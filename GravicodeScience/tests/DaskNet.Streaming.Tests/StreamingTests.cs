using System.Globalization;
using System.Threading.Channels;
using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.DaskNet.Streaming;
using Gravicode.Science.GraviFrame;
using Xunit;
using static Gravicode.Science.DaskNet.Expr;

namespace Gravicode.Science.Tests.DaskNet.Streaming;

/// <summary>Windows are pinned against a brute-force group-by over all rows at once; restarts
/// against one uninterrupted run.</summary>
public sealed class StreamingTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "dasknet-stream-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
    }

    private static readonly DateTime T0 = new(2026, 3, 1, 8, 0, 0);
    private static readonly string[] Sensors = ["north", "south", "east"];

    /// <summary>Readings in event-time order (with small jitter), split into batches.</summary>
    private static List<DataFrame> Readings(int batches, int perBatch, int seed, double jitterSeconds = 0)
    {
        var random = new Random(seed);
        var result = new List<DataFrame>();
        var clock = 0.0;
        for (var b = 0; b < batches; b++)
        {
            var times = new DateTime?[perBatch];
            var sensor = new string?[perBatch];
            var value = new double[perBatch];
            for (var i = 0; i < perBatch; i++)
            {
                clock += random.NextDouble() * 20;
                times[i] = T0.AddSeconds(clock - random.NextDouble() * jitterSeconds);
                sensor[i] = Sensors[random.Next(3)];
                value[i] = i % 23 == 0 ? double.NaN : Math.Round(random.NextDouble() * 100, 2);
            }
            result.Add(new DataFrame([new DateTimeSeries("time", times), new TextSeries("sensor", sensor), new NumericSeries("value", value)]));
        }
        return result;
    }

    /// <summary>The windows of every row at once: (start, sensor) -> (count rows, sum, count, max).</summary>
    private static SortedDictionary<(DateTime, string), (double Rows, double Sum, double Count, double Max)> BruteForce(
        IEnumerable<DataFrame> batches, TimeSpan size, TimeSpan slide)
    {
        var result = new SortedDictionary<(DateTime, string), (double, double, double, double)>();
        foreach (var b in batches)
            for (var i = 0; i < b.RowCount; i++)
            {
                var t = b.DateTimes("time")[i]!.Value;
                var since = (t - DateTime.UnixEpoch).Ticks;
                for (var start = since / slide.Ticks * slide.Ticks; start > since - size.Ticks; start -= slide.Ticks)
                {
                    var key = (DateTime.UnixEpoch.AddTicks(start), b.Text("sensor")[i]!);
                    var (rows, sum, count, max) = result.GetValueOrDefault(key, (0, 0, 0, double.NegativeInfinity));
                    var v = b.Numeric("value")[i];
                    result[key] = double.IsNaN(v) ? (rows + 1, sum, count, max) : (rows + 1, sum + v, count + 1, Math.Max(max, v));
                }
            }
        return result;
    }

    private static List<((DateTime, string), (double, double, double, double))> Rows(DataFrame f)
        => [.. Enumerable.Range(0, f.RowCount).Select(i => ((f.DateTimes("window_start")[i]!.Value, f.Text("sensor")[i]!),
            (f.Numeric("count")[i], f.Numeric("value_sum")[i], f.Numeric("value_count")[i], f.Numeric("value_max")[i])))
            .OrderBy(r => r.Item1)];

    private static readonly (string?, string)[] Aggs = [(null, "count"), ("value", "sum"), ("value", "count"), ("value", "max")];

    [Theory]
    [InlineData(10, 10)]       // tumbling
    [InlineData(10, 4)]        // sliding: each row in two or three windows
    public async Task Windows_equal_a_brute_force_group_by(int sizeMinutes, int slideMinutes)
    {
        var batches = Readings(8, 120, seed: sizeMinutes + slideMinutes);
        var (size, slide) = (TimeSpan.FromMinutes(sizeMinutes), TimeSpan.FromMinutes(slideMinutes));
        var emitted = new List<(StreamBatch Info, DataFrame Frame)>();
        var query = DaskStream.From(batches)
            .Window("time", size, ["sensor"], Aggs, slide: slide)
            .ForEachBatch((f, info) => { emitted.Add((info, f)); return Task.CompletedTask; });
        var stats = await query.RunAsync();

        var all = DataFrame.Concat([.. emitted.Select(e => e.Frame)]);
        Assert.Equal(["window_start", "window_end", "sensor", "count", "value_sum", "value_count", "value_max"], all.ColumnNames);
        Assert.Equal(BruteForce(batches, size, slide).Select(p => (p.Key, p.Value)), Rows(all));
        Assert.Equal(0, stats.LateRows);
        Assert.Equal(all.RowCount, stats.WindowsEmitted);
        Assert.Equal(8 * 120, stats.InputRows);

        // Windows close as the watermark passes them, not all at the end.
        Assert.True(emitted.Count(e => e.Info.InputRows > 0) >= 5, "windows should be emitted while the stream runs");
        foreach (var (info, frame) in emitted.Where(e => e.Info.InputRows > 0))
            Assert.All(Enumerable.Range(0, frame.RowCount), r => Assert.True(frame.DateTimes("window_end")[r] <= info.Watermark));
        Assert.All(Enumerable.Range(0, all.RowCount), r => Assert.Equal(size, all.DateTimes("window_end")[r] - all.DateTimes("window_start")[r]));
    }

    [Fact]
    public async Task Late_rows_are_dropped_unless_lateness_allows_them()
    {
        DataFrame Batch(params (int Minute, double Value)[] rows) => new([
            new DateTimeSeries("time", rows.Select(r => (DateTime?)T0.AddMinutes(r.Minute)).ToArray()),
            new TextSeries("sensor", rows.Select(_ => (string?)"north").ToArray()),
            new NumericSeries("value", rows.Select(r => r.Value).ToArray())]);
        DataFrame[] batches = [Batch((1, 1), (12, 2)), Batch((3, 4)), Batch((25, 8)), Batch((14, 16))];

        // No lateness: minute 3 arrives after the watermark (12) closed [0, 10), and minute 14
        // after minute 25 closed [10, 20).
        var strict = DaskStream.From(batches).Window("time", TimeSpan.FromMinutes(10), ["sensor"], [("value", "sum")]).Collect();
        var s = await strict.RunAsync();
        Assert.Equal(2, s.LateRows);
        Assert.Equal([1.0, 2, 8], strict.CollectedFrame().Numeric("value_sum").Values.ToArray());

        // Twelve minutes of lateness: the watermark trails at 0, 0, 13, so [10, 20) is still open
        // when minute 14 arrives.
        var lenient = DaskStream.From(batches).Window("time", TimeSpan.FromMinutes(10), ["sensor"], [("value", "sum")], allowedLateness: TimeSpan.FromMinutes(12)).Collect();
        var l = await lenient.RunAsync();
        Assert.Equal(0, l.LateRows);
        Assert.Equal([1.0 + 4, 2 + 16, 8], lenient.CollectedFrame().Numeric("value_sum").Values.ToArray());
    }

    [Fact]
    public async Task Stateless_steps_equal_one_pass_over_everything_locally_and_on_a_cluster()
    {
        var batches = Readings(5, 200, seed: 4);
        var whole = DaskFrame.FromFrame(DataFrame.Concat(batches), 3)
            .Filter(Col("value") > 50).WithColumn("double", Col("value") * 2).Select("sensor", "double").Compute();

        await using var cluster = await LocalCluster.StartAsync(workers: 2, threadsPerWorker: 2);
        foreach (var scheduler in new IScheduler?[] { null, cluster.Client })
        {
            var stream = DaskStream.From(batches).Filter(Col("value") > 50).WithColumn("double", Col("value") * 2).Select("sensor", "double").Partitions(3);
            var query = (scheduler is null ? stream : stream.WithScheduler(scheduler)).Collect();
            var stats = await query.RunAsync();
            var got = query.CollectedFrame();
            Assert.Equal(whole.Text("sensor").Values.ToArray(), got.Text("sensor").Values.ToArray());
            Assert.Equal(whole.Numeric("double").Values.ToArray(), got.Numeric("double").Values.ToArray());
            Assert.Equal(5, stats.Batches);
            Assert.Equal(whole.RowCount, stats.OutputRows);
        }

        // Map is in-process, so it cannot go to the cluster.
        var local = DaskStream.From(batches).Map(f => f.Head(1)).WithScheduler(cluster.Client).Collect();
        await Assert.ThrowsAsync<NotSupportedException>(() => local.RunAsync());
    }

    [Fact]
    public async Task Sql_runs_per_batch_and_channels_end_when_completed()
    {
        var channel = Channel.CreateUnbounded<DataFrame>();
        var batches = Readings(3, 90, seed: 6);
        var query = DaskStream.FromChannel(channel.Reader)
            .Sql("SELECT sensor, COUNT(*) AS n, MAX(value) AS top FROM batch GROUP BY sensor ORDER BY sensor").Collect();
        var run = query.RunAsync();
        foreach (var b in batches) await channel.Writer.WriteAsync(b);
        channel.Writer.Complete();
        await run;
        Assert.Equal(3, query.Results.Count);
        for (var i = 0; i < 3; i++)
        {
            Assert.Equal(["east", "north", "south"], query.Results[i].Text("sensor").Values.ToArray().Select(v => v!));
            Assert.Equal(90, query.Results[i].Numeric("n").Values.ToArray().Sum());
        }
    }

    private static void Drop(string directory, string name, DataFrame frame)
    {
        Directory.CreateDirectory(directory);
        var temp = Path.Combine(directory, name + ".partial");
        frame.WriteCsv(temp);
        File.Move(temp, Path.Combine(directory, name));
    }

    [Fact]
    public async Task A_checkpointed_restart_consumes_only_new_files_and_carries_windows_over()
    {
        var batches = Readings(6, 80, seed: 9);
        var inbox = Path.Combine(_root, "inbox");
        var checkpoint = Path.Combine(_root, "checkpoint");
        var output = Path.Combine(_root, "out");
        for (var i = 0; i < 4; i++) Drop(inbox, $"part-{i:D3}.csv", batches[i]);

        StreamQuery Query() => DaskStream.FromDirectory(inbox)
            .Window("time", TimeSpan.FromMinutes(15), ["sensor"], Aggs).ToCsvDirectory(output);

        var first = Query();
        first.FlushOnEnd = false;                         // the stream is not over; keep open windows
        var s1 = await first.RunAsync(checkpoint);
        Assert.Equal(4, s1.Batches);
        Assert.True(File.Exists(Path.Combine(checkpoint, "checkpoint.json")));

        for (var i = 4; i < 6; i++) Drop(inbox, $"part-{i:D3}.csv", batches[i]);
        var s2 = await Query().RunAsync(checkpoint);
        Assert.Equal(2, s2.Batches);                      // the four old files are not read again
        Assert.Equal(160, s2.InputRows);

        var files = Directory.GetFiles(output, "batch-*.csv").Order(StringComparer.Ordinal).ToArray();
        Assert.Contains(files, f => f.EndsWith("batch-000006.csv"));      // ids continue across the restart
        var written = DataFrame.Concat([.. files.Select(f => DataFrame.ReadCsv(f))]);
        var expected = BruteForce(batches.Select(b => DataFrame.ParseCsv(ToCsv(b))), TimeSpan.FromMinutes(15), TimeSpan.FromMinutes(15));
        var actual = Rows(written);
        Assert.Equal(expected.Count, actual.Count);
        foreach (var ((key, e), (akey, a)) in expected.Select(p => (p.Key, p.Value)).Zip(actual))
        {
            Assert.Equal(key, akey);
            Assert.Equal(e.Rows, a.Item1);
            Assert.Equal(e.Sum, a.Item2, 9);              // through CSV and back
            Assert.Equal(e.Count, a.Item3);
            Assert.Equal(e.Max, a.Item4, 9);
        }

        // A checkpoint is tied to its window definition.
        var other = DaskStream.FromDirectory(inbox).Window("time", TimeSpan.FromMinutes(5), ["sensor"], Aggs).Collect();
        await Assert.ThrowsAsync<InvalidOperationException>(() => other.RunAsync(checkpoint));
    }

    private static string ToCsv(DataFrame f)
    {
        var path = Path.GetTempFileName();
        try { f.WriteCsv(path); return File.ReadAllText(path); }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task A_watched_directory_picks_up_files_as_they_arrive()
    {
        var inbox = Path.Combine(_root, "watched");
        Directory.CreateDirectory(inbox);
        var batches = Readings(3, 30, seed: 12);
        var seen = new List<StreamBatch>();
        var got = new TaskCompletionSource();
        var query = DaskStream.FromDirectory(inbox, watch: true, pollInterval: TimeSpan.FromMilliseconds(50))
            .Filter(Col("value") >= 0)
            .ForEachBatch((f, info) =>
            {
                lock (seen) { seen.Add(info); if (seen.Count == 3) got.TrySetResult(); }
                return Task.CompletedTask;
            });
        using var stop = new CancellationTokenSource();
        var run = query.RunAsync(cancellationToken: stop.Token);
        for (var i = 0; i < 3; i++)
        {
            await Task.Delay(120);
            Drop(inbox, $"f{i}.csv", batches[i]);
        }
        Assert.Same(got.Task, await Task.WhenAny(got.Task, Task.Delay(TimeSpan.FromSeconds(20))));
        await stop.CancelAsync();
        var stats = await run;                            // cancellation ends a watch normally
        Assert.Equal(3, stats.Batches);
        Assert.Equal(["f0.csv", "f1.csv", "f2.csv"], seen.Select(s => s.Files.Single()));
        Assert.Equal([0L, 1, 2], seen.Select(s => s.Id));
    }

    [Fact]
    public void Window_definitions_are_validated()
    {
        var s = DaskStream.From(Readings(1, 5, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => s.Window("time", TimeSpan.Zero, [], [(null, "count")]));
        Assert.Throws<ArgumentOutOfRangeException>(() => s.Window("time", TimeSpan.FromMinutes(1), [], [(null, "count")], slide: TimeSpan.FromMinutes(2)));
        Assert.Throws<ArgumentException>(() => s.Window("time", TimeSpan.FromMinutes(1), [], [("value", "median")]));
        Assert.Throws<ArgumentException>(() => s.Window("time", TimeSpan.FromMinutes(1), [], [(null, "sum")]));
        Assert.Throws<ArgumentException>(() => s.Window("time", TimeSpan.FromMinutes(1), [], []));
    }

    [Fact]
    public async Task A_numeric_time_column_is_refused_with_a_reason()
    {
        var frame = new DataFrame([new NumericSeries("time", [1.0, 2.0]), new NumericSeries("value", [1.0, 2.0])]);
        var q = DaskStream.From([frame]).Window("time", TimeSpan.FromMinutes(1), [], [("value", "sum")]).Collect();
        var e = await Assert.ThrowsAsync<InvalidOperationException>(() => q.RunAsync());
        Assert.Contains("dates and times", e.Message);
        _ = CultureInfo.InvariantCulture;
    }
}
