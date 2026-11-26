using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text.Json;
using Gravicode.Science.DaskNet.Streaming;
using Gravicode.Science.GraviFrame;
using MQTTnet;
using MQTTnet.Protocol;
using MQTTnet.Server;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet.Streaming;

/// <summary>Stream-stream joins are pinned against a brute-force join over every row at once, under
/// several batch splits and arrival orders; the broker sources against what was published.</summary>
public sealed class JoinAndBrokerTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "dasknet-join-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
    }

    private static readonly DateTime T0 = new(2026, 3, 1, 8, 0, 0);
    private static readonly string[] Users = ["ana", "budi", "citra", "dewi", "eko"];

    /// <summary>Events in time order with jitter: (id, user, time), user missing now and then.</summary>
    private static DataFrame Events(string idColumn, string timeColumn, int count, int seed, double meanGapSeconds, double jitterSeconds)
    {
        var random = new Random(seed);
        var ids = new double[count];
        var users = new string?[count];
        var times = new DateTime?[count];
        var clock = 0.0;
        for (var i = 0; i < count; i++)
        {
            clock += random.NextDouble() * 2 * meanGapSeconds;
            ids[i] = seed * 100_000 + i;
            users[i] = i % 37 == 5 ? null : Users[random.Next(Users.Length)];
            times[i] = T0.AddSeconds(clock - random.NextDouble() * jitterSeconds);
        }
        return new DataFrame([new NumericSeries(idColumn, ids), new TextSeries("user", users), new DateTimeSeries(timeColumn, times)]);
    }

    private static List<DataFrame> Split(DataFrame frame, int[] sizes)
    {
        var result = new List<DataFrame>();
        var at = 0;
        for (var k = 0; at < frame.RowCount; k++)
        {
            var n = Math.Min(sizes[k % sizes.Length], frame.RowCount - at);
            result.Add(frame.Take(Enumerable.Range(at, n).ToArray()));
            at += n;
        }
        return result;
    }

    /// <summary>Every (left id, right id) whose users are equal (and present) and whose times satisfy
    /// right - left in [lower, upper].</summary>
    private static HashSet<(double, double)> BruteForce(DataFrame left, DataFrame right, TimeSpan lower, TimeSpan upper)
    {
        var result = new HashSet<(double, double)>();
        for (var i = 0; i < left.RowCount; i++)
            for (var j = 0; j < right.RowCount; j++)
            {
                if (left.Text("user")[i] is not { } u || right.Text("user")[j] != u) continue;
                var gap = right.DateTimes("click_time")[j]!.Value - left.DateTimes("view_time")[i]!.Value;
                if (gap >= lower && gap <= upper) result.Add((left.Numeric("view_id")[i], right.Numeric("click_id")[j]));
            }
        return result;
    }

    private static readonly TimeSpan Lower = TimeSpan.Zero;
    private static readonly TimeSpan Upper = TimeSpan.FromMinutes(3);
    private static readonly TimeSpan Lateness = TimeSpan.FromMinutes(2);

    private static JoinSpec Spec(StreamJoinType how)
        => new(DaskStream.From(Array.Empty<DataFrame>()), ["user"], "view_time", "click_time", Lower, Upper, how, Lateness, "_right");

    /// <summary>Feeds the batches to a join state in a given interleaving and collects its output.</summary>
    private static (DataFrame Output, JoinState State, int MaxState) Drive(JoinSpec spec, List<DataFrame> left, List<DataFrame> right, string order, int seed)
    {
        var state = new JoinState(spec);
        var outputs = new List<DataFrame>();
        int li = 0, ri = 0, max = 0;
        var random = new Random(seed);
        while (li < left.Count || ri < right.Count)
        {
            var takeLeft = ri >= right.Count || li < left.Count && order switch
            {
                "left-first" => true,
                "right-first" => false,
                "alternate" => (li + ri) % 2 == 0,
                _ => random.Next(2) == 0,
            };
            var output = takeLeft ? state.Add(true, left[li++]) : state.Add(false, right[ri++]);
            if (output is not null) outputs.Add(output);
            max = Math.Max(max, state.Left.Count + state.Right.Count);
        }
        if (state.Flush() is { } rest) outputs.Add(rest);
        return (outputs.Count == 0 ? DataFrame.Empty() : DataFrame.Concat(outputs), state, max);
    }

    private static HashSet<(double, double)> Pairs(DataFrame output)
    {
        var result = new HashSet<(double, double)>();
        for (var i = 0; i < output.RowCount; i++)
        {
            var l = output.Numeric("view_id")[i];
            var r = output.Numeric("click_id")[i];
            if (!double.IsNaN(l) && !double.IsNaN(r)) Assert.True(result.Add((l, r)), $"pair ({l}, {r}) emitted twice");
        }
        return result;
    }

    [Theory]
    [InlineData("left-first", 7)]
    [InlineData("right-first", 7)]
    [InlineData("alternate", 13)]
    [InlineData("random", 1)]
    [InlineData("random", 40)]
    public void Inner_join_equals_brute_force_whatever_the_interleaving(string order, int batchSize)
    {
        var views = Events("view_id", "view_time", 300, 1, 20, 30);
        var clicks = Events("click_id", "click_time", 250, 2, 24, 30);
        var expected = BruteForce(views, clicks, Lower, Upper);
        Assert.True(expected.Count > 50, $"the data should produce matches ({expected.Count})");

        var (output, state, _) = Drive(Spec(StreamJoinType.Inner), Split(views, [batchSize, batchSize + 3]), Split(clicks, [batchSize + 1]), order, batchSize);
        Assert.Equal(0, state.LateRows);
        Assert.Equal(expected.Count, output.RowCount);
        Assert.Equal(expected.OrderBy(p => p), Pairs(output).OrderBy(p => p));
        Assert.Equal(0, state.Left.Count + state.Right.Count);                  // flushed
    }

    [Theory]
    [InlineData(StreamJoinType.Left)]
    [InlineData(StreamJoinType.Right)]
    [InlineData(StreamJoinType.Outer)]
    public void Outer_joins_emit_exactly_the_rows_that_never_matched(StreamJoinType how)
    {
        var views = Events("view_id", "view_time", 200, 3, 20, 20);
        var clicks = Events("click_id", "click_time", 200, 4, 20, 20);
        var expected = BruteForce(views, clicks, Lower, Upper);
        var (output, state, _) = Drive(Spec(how), Split(views, [9]), Split(clicks, [11]), "alternate", 0);
        Assert.Equal(0, state.LateRows);
        Assert.Equal(expected.OrderBy(p => p), Pairs(output).OrderBy(p => p));

        var leftAlone = Enumerable.Range(0, views.RowCount).Select(i => views.Numeric("view_id")[i]).Where(id => expected.All(p => p.Item1 != id)).ToHashSet();
        var rightAlone = Enumerable.Range(0, clicks.RowCount).Select(i => clicks.Numeric("click_id")[i]).Where(id => expected.All(p => p.Item2 != id)).ToHashSet();
        var gotLeft = new List<double>();
        var gotRight = new List<double>();
        for (var i = 0; i < output.RowCount; i++)
        {
            var l = output.Numeric("view_id")[i];
            var r = output.Numeric("click_id")[i];
            if (double.IsNaN(r)) gotLeft.Add(l);
            if (double.IsNaN(l)) gotRight.Add(r);
            if (double.IsNaN(r)) Assert.True(output.DateTimes("click_time")[i] is null);
        }
        Assert.Equal(how is StreamJoinType.Left or StreamJoinType.Outer ? leftAlone.Order() : [], gotLeft.Order());
        Assert.Equal(how is StreamJoinType.Right or StreamJoinType.Outer ? rightAlone.Order() : [], gotRight.Order());
        Assert.Equal(["user", "view_id", "view_time", "click_id", "click_time"], output.ColumnNames);
    }

    [Fact]
    public void Join_state_is_bounded_by_the_interval_not_the_stream()
    {
        var views = Events("view_id", "view_time", 4000, 5, 10, 10);
        var clicks = Events("click_id", "click_time", 4000, 6, 10, 10);
        var (output, _, max) = Drive(Spec(StreamJoinType.Inner), Split(views, [50]), Split(clicks, [50]), "alternate", 0);
        Assert.Equal(BruteForce(views, clicks, Lower, Upper).Count, output.RowCount);
        // (3 min interval + 2 min lateness) at one row per 10 s per side is ~60 rows a side, plus a
        // batch each; 8000 rows must never all be held.
        Assert.True(max < 400, $"state peaked at {max} rows");
    }

    [Fact]
    public void Rows_older_than_their_sides_watermark_are_late()
    {
        var state = new JoinState(Spec(StreamJoinType.Inner));
        DataFrame View(double id, string user, DateTime t) => new([new NumericSeries("view_id", [id]), new TextSeries("user", [user]), new DateTimeSeries("view_time", [t])]);
        DataFrame Click(double id, string user, DateTime t) => new([new NumericSeries("click_id", [id]), new TextSeries("user", [user]), new DateTimeSeries("click_time", [t])]);
        Assert.Null(state.Add(true, View(1, "ana", T0.AddMinutes(30))));         // left watermark: 08:28
        Assert.Null(state.Add(true, View(2, "ana", T0.AddMinutes(10))));         // older than 08:28: late
        Assert.Equal(1, state.LateRows);
        var matched = state.Add(false, Click(9, "ana", T0.AddMinutes(31)));
        Assert.NotNull(matched);
        Assert.Equal(1, matched.RowCount);
        Assert.Equal(1, matched.Numeric("view_id")[0]);
    }

    [Fact]
    public async Task A_joined_query_runs_both_sources_concurrently_and_restarts_from_its_checkpoint()
    {
        var views = Events("view_id", "view_time", 240, 7, 20, 20);
        var clicks = Events("click_id", "click_time", 240, 8, 20, 20);
        var expected = BruteForce(views, clicks, Lower, Upper);
        var lb = Split(views, [17]);
        var rb = Split(clicks, [19]);

        // One uninterrupted run.
        var whole = DaskStream.From(lb).IntervalJoin(DaskStream.From(rb), ["user"], "view_time", "click_time", Lower, Upper,
            StreamJoinType.Left, Lateness).Collect();
        await whole.RunAsync();
        var all = whole.CollectedFrame();
        Assert.Equal(expected.OrderBy(p => p), Pairs(all).OrderBy(p => p));

        // The same in two runs: the first stops halfway without flushing, the second resumes.
        var checkpoint = Path.Combine(_root, "ckpt");
        var first = DaskStream.From(lb.Take(lb.Count / 2)).IntervalJoin(DaskStream.From(rb.Take(rb.Count / 2)), ["user"], "view_time", "click_time",
            Lower, Upper, StreamJoinType.Left, Lateness).Collect();
        first.FlushOnEnd = false;
        await first.RunAsync(checkpoint);
        Assert.True(first.Statistics.JoinStateRows > 0);
        var second = DaskStream.From(lb.Skip(lb.Count / 2)).IntervalJoin(DaskStream.From(rb.Skip(rb.Count / 2)), ["user"], "view_time", "click_time",
            Lower, Upper, StreamJoinType.Left, Lateness).Collect();
        await second.RunAsync(checkpoint);
        var resumed = DataFrame.Concat([first.CollectedFrame(), second.CollectedFrame()]);
        Assert.Equal(Pairs(all).OrderBy(p => p), Pairs(resumed).OrderBy(p => p));
        Assert.Equal(all.RowCount, resumed.RowCount);                            // unmatched left rows too

        // A different join refuses the checkpoint.
        var other = DaskStream.From(lb).Join(DaskStream.From(rb), ["user"], "view_time", "click_time", TimeSpan.FromMinutes(1)).Collect();
        await Assert.ThrowsAsync<InvalidOperationException>(() => other.RunAsync(checkpoint));
    }

    [Fact]
    public async Task A_join_can_be_windowed()
    {
        var views = Events("view_id", "view_time", 200, 9, 20, 0);
        var clicks = Events("click_id", "click_time", 200, 10, 20, 0);
        var expected = BruteForce(views, clicks, Lower, Upper);
        var query = DaskStream.From(Split(views, [10])).Join(DaskStream.From(Split(clicks, [10])), ["user"], "view_time", "click_time",
                TimeSpan.FromMinutes(3), allowedLateness: Lateness)
            .Window("click_time", TimeSpan.FromMinutes(10), ["user"], [(null, "count")], allowedLateness: TimeSpan.FromMinutes(10)).Collect();
        await query.RunAsync();
        var windows = query.CollectedFrame();
        // A symmetric 3-minute join is a superset of the [0, 3] one checked elsewhere; count it here.
        var symmetric = BruteForce(views, clicks, -Upper, Upper);
        Assert.True(symmetric.Count >= expected.Count);
        Assert.Equal(symmetric.Count, (int)Enumerable.Range(0, windows.RowCount).Sum(i => windows.Numeric("count")[i]));
    }

    // ------------------------------------------------------------------ message decoding

    [Fact]
    public void Csv_messages_split_quoted_fields_and_pin_types()
    {
        var options = new MessageSourceOptions { TopicColumn = "topic" };
        var decoder = new MessageDecoder(MessageFormat.Csv(["id", "name", "at", "ok"]), options);
        var first = decoder.Decode([
            new RawMessage("t/1", "1,\"Smith, \"\"J\"\"\",2026-03-01T08:00:00Z,true", null),
            new RawMessage("t/2", "2,,2026-03-01T09:30:00Z,false", null),
        ]);
        Assert.Equal(["id", "name", "at", "ok", "topic"], first.ColumnNames);
        Assert.Equal("Smith, \"J\"", first.Text("name")[0]);
        Assert.Null(first.Text("name")[1]);
        Assert.Equal(new DateTime(2026, 3, 1, 9, 30, 0), first.DateTimes("at")[1]);
        Assert.Equal(false, first.Booleans("ok")[1]);
        // Pinned: a later non-number in "id" is missing, not a type change.
        var second = decoder.Decode([new RawMessage("t/1", "x,n,2026-03-01T10:00:00Z,true", null)]);
        Assert.IsType<NumericSeries>(second["id"]);
        Assert.True(double.IsNaN(second.Numeric("id")[0]));
        Assert.Null(MessageDecoder.SplitCsv("\"open", ','));
    }

    [Fact]
    public void Json_messages_add_new_fields_and_keep_string_ids_as_text()
    {
        var decoder = new MessageDecoder(MessageFormat.Json(("value", ColumnType.Number)), new MessageSourceOptions { TopicColumn = null });
        var a = decoder.Decode([new RawMessage("s", "{\"value\": \"4.5\", \"code\": \"007\", \"at\": \"2026-03-01T08:00:00Z\"}", null)]);
        Assert.Equal(4.5, a.Numeric("value")[0]);
        Assert.Equal("007", a.Text("code")[0]);
        Assert.IsType<DateTimeSeries>(a["at"]);
        var b = decoder.Decode([new RawMessage("s", "{\"value\": 1, \"extra\": {\"x\": 1}}", null), new RawMessage("s", "not json", null)]);
        Assert.Equal(["value", "code", "at", "extra"], b.ColumnNames);
        Assert.Equal("{\"x\": 1}", b.Text("extra")[0]);
        Assert.Equal(1, decoder.Malformed);
        Assert.True(double.IsNaN(b.Numeric("value")[1]));
    }

    // ------------------------------------------------------------------ MQTT (in-process broker)

    private static int FreePort()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    [Fact]
    public async Task Mqtt_messages_published_to_a_broker_come_out_as_windows()
    {
        var port = FreePort();
        var serverFactory = new MqttServerFactory();
        using var server = serverFactory.CreateMqttServer(serverFactory.CreateServerOptionsBuilder().WithDefaultEndpoint().WithDefaultEndpointPort(port).Build());
        var subscribed = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        server.ClientSubscribedTopicAsync += _ => { subscribed.TrySetResult(); return Task.CompletedTask; };
        await server.StartAsync();
        try
        {
            var query = DaskStream.FromMqtt("127.0.0.1", ["plant/+/temperature"], MessageFormat.Json(("time", ColumnType.DateTime), ("value", ColumnType.Number)),
                    new MqttOptions { Port = port, MaxBatchRows = 25, MaxBatchDelay = TimeSpan.FromMilliseconds(100), IdleTimeout = TimeSpan.FromSeconds(3), TimestampColumn = "received" })
                .WithColumn("sensor", Gravicode.Science.DaskNet.Expr.Col("topic"))
                .Window("time", TimeSpan.FromMinutes(10), ["sensor"], [(null, "count"), ("value", "sum")], allowedLateness: TimeSpan.FromHours(1))
                .Collect();
            var run = query.RunAsync();
            await subscribed.Task.WaitAsync(TimeSpan.FromSeconds(20));

            var random = new Random(3);
            var expected = new Dictionary<(DateTime, string), (double Count, double Sum)>();
            var factory = new MqttClientFactory();
            using var publisher = factory.CreateMqttClient();
            await publisher.ConnectAsync(new MqttClientOptionsBuilder().WithTcpServer("127.0.0.1", port).WithClientId("publisher").Build());
            for (var i = 0; i < 300; i++)
            {
                var sensor = new[] { "a", "b", "c" }[random.Next(3)];
                var time = T0.AddSeconds(i * 13);
                var value = Math.Round(random.NextDouble() * 50, 2);
                var topic = $"plant/{sensor}/temperature";
                var payload = JsonSerializer.Serialize(new { time = time.ToString("yyyy-MM-ddTHH:mm:ssZ", CultureInfo.InvariantCulture), value });
                await publisher.PublishAsync(new MqttApplicationMessageBuilder().WithTopic(topic).WithPayload(payload)
                    .WithQualityOfServiceLevel(MqttQualityOfServiceLevel.AtLeastOnce).Build());
                var start = new DateTime(time.Ticks / TimeSpan.FromMinutes(10).Ticks * TimeSpan.FromMinutes(10).Ticks);
                var (c, s) = expected.GetValueOrDefault((start, topic));
                expected[(start, topic)] = (c + 1, s + value);
            }
            await publisher.PublishAsync(new MqttApplicationMessageBuilder().WithTopic("plant/a/humidity").WithPayload("{\"time\":\"2026-03-01T08:00:00Z\",\"value\":1}").Build());
            await publisher.DisconnectAsync();

            var stats = await run.WaitAsync(TimeSpan.FromSeconds(60));
            Assert.Equal(300, stats.InputRows);                                  // the humidity topic is not subscribed
            Assert.True(stats.Batches > 1, $"{stats.Batches} batch(es)");
            var windows = query.CollectedFrame();
            var got = Enumerable.Range(0, windows.RowCount).ToDictionary(
                i => (windows.DateTimes("window_start")[i]!.Value, windows.Text("sensor")[i]!),
                i => (windows.Numeric("count")[i], Math.Round(windows.Numeric("value_sum")[i], 6)));
            Assert.Equal(expected.OrderBy(e => e.Key).Select(e => (e.Key, (e.Value.Count, Math.Round(e.Value.Sum, 6)))),
                got.OrderBy(e => e.Key).Select(e => (e.Key, e.Value)));
        }
        finally
        {
            await server.StopAsync();
        }
    }
    [Fact]
    public async Task Mqtt_messages_a_failed_sink_never_finished_are_redelivered_on_restart()
    {
        var port = FreePort();
        var serverFactory = new MqttServerFactory();
        using var server = serverFactory.CreateMqttServer(serverFactory.CreateServerOptionsBuilder()
            .WithDefaultEndpoint().WithDefaultEndpointPort(port).WithPersistentSessions(true).Build());
        var subscribed = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        server.ClientSubscribedTopicAsync += _ => { subscribed.TrySetResult(); return Task.CompletedTask; };
        await server.StartAsync();
        try
        {
            // A fixed client id and a persistent session: what the broker has not had a PUBACK for
            // stays its responsibility. With acknowledgement after the sink, that is every message
            // of a batch whose sink threw, and everything behind it.
            MqttOptions Options() => new()
            {
                Port = port, ClientId = "dasknet-redelivery", CleanSession = false,
                MaxBatchRows = 10, MaxBatchDelay = TimeSpan.FromMilliseconds(100), IdleTimeout = TimeSpan.FromSeconds(3),
            };
            var format = MessageFormat.Json(("id", ColumnType.Number));
            var failing = DaskStream.FromMqtt("127.0.0.1", ["jobs/#"], format, Options())
                .ForEachBatch((_, _) => throw new InvalidOperationException("sink down"));
            var first = failing.RunAsync();
            await subscribed.Task.WaitAsync(TimeSpan.FromSeconds(20));

            var factory = new MqttClientFactory();
            using var publisher = factory.CreateMqttClient();
            await publisher.ConnectAsync(new MqttClientOptionsBuilder().WithTcpServer("127.0.0.1", port).WithClientId("publisher").Build());
            for (var i = 0; i < 40; i++)
                await publisher.PublishAsync(new MqttApplicationMessageBuilder().WithTopic("jobs/a").WithPayload($"{{\"id\":{i}}}")
                    .WithQualityOfServiceLevel(MqttQualityOfServiceLevel.AtLeastOnce).Build());
            var error = await Assert.ThrowsAnyAsync<Exception>(() => first.WaitAsync(TimeSpan.FromSeconds(30)));
            Assert.Contains("sink down", error.ToString());

            var seen = new System.Collections.Concurrent.ConcurrentBag<double>();
            var stats = await DaskStream.FromMqtt("127.0.0.1", ["jobs/#"], format, Options())
                .ForEachBatch((frame, _) =>
                {
                    foreach (var v in frame.Numeric("id").Values) seen.Add(v);
                    return Task.CompletedTask;
                })
                .RunAsync().WaitAsync(TimeSpan.FromSeconds(60));
            await publisher.DisconnectAsync();
            // At least once: every id comes back, and nothing that was never published.
            Assert.Equal(Enumerable.Range(0, 40).Select(i => (double)i), seen.Distinct().Order());
            Assert.True(stats.InputRows >= 40, $"{stats.InputRows} rows");
        }
        finally
        {
            await server.StopAsync();
        }
    }
}
