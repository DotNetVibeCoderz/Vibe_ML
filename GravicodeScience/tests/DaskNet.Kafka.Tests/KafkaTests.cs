using Gravicode.Science.DaskNet;
using System.Text.Json;
using Confluent.Kafka;
using Gravicode.Science.DaskNet.Streaming;
using Gravicode.Science.GraviFrame;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet.Kafka;

/// <summary>
/// The Kafka source and sink: offsets and transactions against an in-memory model of a broker
/// (consumer groups, committed offsets, transactions that land whole or not at all), with crashes
/// injected on either side of a commit; and, with <c>DASKNET_KAFKA</c> set, against a real broker.
/// </summary>
public sealed class KafkaTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "dasknet-kafka-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root)) try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
    }

    private static readonly DateTime T0 = new(2026, 3, 1, 8, 0, 0);

    // ------------------------------------------------------------------ Kafka (in-memory broker model)

    /// <summary>A topic's partitions and a group's committed offsets, shared by consumers.</summary>
    private sealed class FakeBroker
    {
        public List<List<RawMessage>> Partitions { get; } = [];
        public Dictionary<(string, int), long> Committed { get; } = [];
        public int Started { get; set; }
    }

    private sealed class FakeConsumer(FakeBroker broker) : IKafkaConsumer
    {
        private readonly long[] _position = new long[broker.Partitions.Count];
        private int _turn;

        public void Start(IReadOnlyDictionary<(string Topic, int Partition), long> resume)
        {
            broker.Started++;
            for (var p = 0; p < _position.Length; p++)
                _position[p] = resume.TryGetValue(("orders", p), out var o) ? o : broker.Committed.GetValueOrDefault(("orders", p));
        }

        public RawMessage? Consume(TimeSpan timeout, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (var k = 0; k < _position.Length; k++)
            {
                var p = (_turn + k) % _position.Length;
                if (_position[p] < broker.Partitions[p].Count)
                {
                    _turn = p + 1;
                    return broker.Partitions[p][(int)_position[p]++];
                }
            }
            return null;                                                         // nothing new: as if the timeout passed
        }

        public void Commit(IReadOnlyDictionary<(string Topic, int Partition), long> next)
        {
            foreach (var (k, v) in next) broker.Committed[k] = v;
        }

        public void Dispose() { }
    }

    private static FakeBroker Orders(int partitions, int perPartition)
    {
        var broker = new FakeBroker();
        for (var p = 0; p < partitions; p++)
            broker.Partitions.Add([.. Enumerable.Range(0, perPartition).Select(o =>
                new RawMessage("orders", $"{{\"order\": {p * 1000 + o}, \"amount\": {o % 17}}}", T0.AddSeconds(o), $"k{o % 3}", p, o))]);
        return broker;
    }

    private static KafkaOptions KafkaTestOptions() => new() { MaxBatchRows = 40, MaxBatchDelay = TimeSpan.FromSeconds(5), IdleTimeout = TimeSpan.FromMilliseconds(50), KeyColumn = "key", PartitionColumn = "partition", OffsetColumn = "offset", TimestampColumn = "ts" };

    [Fact]
    public async Task Kafka_offsets_are_checkpointed_and_committed_only_after_the_sink()
    {
        var broker = Orders(partitions: 3, perPartition: 70);
        var checkpoint = Path.Combine(_root, "kafka");
        var seen = new List<double>();
        var batches = 0;
        long committedBeforeFailure = 0;
        Task Sink(DataFrame f, StreamBatch b)
        {
            if (++batches == 3)
            {
                committedBeforeFailure = broker.Committed.Values.Sum();
                throw new IOException("sink down");
            }
            seen.AddRange(Enumerable.Range(0, f.RowCount).Select(i => f.Numeric("order")[i]));
            return Task.CompletedTask;
        }

        var stream = KafkaStreams.FromKafkaConsumer(() => new FakeConsumer(broker), MessageFormat.Json(), KafkaTestOptions());
        await Assert.ThrowsAsync<IOException>(() => stream.ForEachBatch(Sink).RunAsync(checkpoint));
        Assert.Equal(80, seen.Count);                                            // two batches of 40 reached the sink
        Assert.Equal(80, committedBeforeFailure);                                // ...and only they were committed

        // Restart: the failed batch is replayed, nothing is skipped or repeated.
        var stats = await stream.ForEachBatch(Sink).RunAsync(checkpoint);
        Assert.Equal(2, broker.Started);
        Assert.Equal(210 - 80, stats.InputRows);
        var all = Enumerable.Range(0, 3).SelectMany(p => Enumerable.Range(0, 70).Select(o => (double)(p * 1000 + o))).Order();
        Assert.Equal(all, seen.Order());
        Assert.Equal(new[] { 70L, 70, 70 }, Enumerable.Range(0, 3).Select(p => broker.Committed[("orders", p)]));

        // A new message after the end: a third run reads only it.
        broker.Partitions[1].Add(new RawMessage("orders", "{\"order\": 9999, \"amount\": 1}", T0, null, 1, 70));
        var third = await stream.ForEachBatch(Sink).RunAsync(checkpoint);
        Assert.Equal(1, third.InputRows);
        Assert.Equal(9999, seen[^1]);
    }

    [Fact]
    public async Task Kafka_without_a_checkpoint_resumes_from_the_groups_commits_and_carries_metadata()
    {
        var broker = Orders(partitions: 2, perPartition: 30);
        broker.Committed[("orders", 0)] = 25;                                    // the group had read most of p0
        var query = KafkaStreams.FromKafkaConsumer(() => new FakeConsumer(broker), MessageFormat.Json(("order", ColumnType.Number)), KafkaTestOptions()).Collect();
        var stats = await query.RunAsync();
        Assert.Equal(5 + 30, stats.InputRows);
        var frame = query.CollectedFrame();
        Assert.Equal(["order", "amount", "topic", "ts", "key", "partition", "offset"], frame.ColumnNames);
        var p0 = Enumerable.Range(0, frame.RowCount).Where(i => frame.Numeric("partition")[i] == 0).Select(i => frame.Numeric("offset")[i]).Order();
        Assert.Equal([25.0, 26, 27, 28, 29], p0);
        Assert.Equal(T0.AddSeconds(25), Enumerable.Range(0, frame.RowCount).Where(i => frame.Numeric("order")[i] == 25).Select(i => frame.DateTimes("ts")[i]).Single());
        Assert.Equal(30, broker.Committed[("orders", 0)]);
    }

    // ------------------------------------------------------------------ the sink (in-memory transactions)

    /// <summary>An output topic with transactions: a write lands whole or not at all, together
    /// with the batch id it records. <see cref="Fault"/> crashes a write before or after its commit.</summary>
    private sealed class FakeTopic
    {
        public List<(string? Key, string Value)> Messages { get; } = [];
        public long LastBatch { get; set; } = -1;
        public int Starts { get; set; }
        public int Transactions { get; set; }
        public Func<long, string?>? Fault { get; set; }
    }

    private sealed class FakeProducer(FakeTopic topic, bool transactional) : IKafkaProducer
    {
        public void Start() => topic.Starts++;

        public long LastCommittedBatch() => transactional ? topic.LastBatch : -1;

        public void Write(IReadOnlyList<(string? Key, string Value)> messages, long batchId)
        {
            var fault = topic.Fault?.Invoke(batchId);
            if (fault == "before") throw new IOException($"crashed before committing batch {batchId}");
            topic.Messages.AddRange(messages);
            topic.Transactions++;
            if (transactional) topic.LastBatch = batchId;
            if (fault == "after") throw new IOException($"crashed after committing batch {batchId}");
        }

        public void Dispose() { }
    }

    private static StreamQuery ToFake(DaskStream stream, FakeTopic topic, KafkaSinkOptions options)
        => KafkaStreams.WithProducer(() => new FakeProducer(topic, options.TransactionalId is not null), options,
            s => StreamQuery.For(stream, s), f => StreamQuery.For(stream, f));

    private static StreamQuery ToFake(WindowedStream stream, FakeTopic topic, KafkaSinkOptions options)
        => KafkaStreams.WithProducer(() => new FakeProducer(topic, options.TransactionalId is not null), options,
            s => StreamQuery.For(stream, s), f => StreamQuery.For(stream, f));

    private static double[] Orders(FakeTopic topic)
        => [.. topic.Messages.Select(m => JsonDocument.Parse(m.Value).RootElement.GetProperty("order").GetDouble())];

    [Fact]
    public async Task A_transactional_sink_writes_every_row_exactly_once_across_crashes()
    {
        var broker = Orders(partitions: 3, perPartition: 70);
        var topic = new FakeTopic();
        var checkpoint = Path.Combine(_root, "exactly-once");
        var stream = KafkaStreams.FromKafkaConsumer(() => new FakeConsumer(broker), MessageFormat.Json(), KafkaTestOptions());
        var options = new KafkaSinkOptions { TransactionalId = "orders-copy", KeyColumn = "key" };

        // Batch 2 commits, then the process dies before its checkpoint is promoted: the restart
        // must see that it landed and not write it again.
        topic.Fault = id => id == 2 ? "after" : null;
        await Assert.ThrowsAsync<IOException>(() => ToFake(stream, topic, options).RunAsync(checkpoint));
        Assert.Equal(120, topic.Messages.Count);
        Assert.True(File.Exists(Path.Combine(checkpoint, "checkpoint.json.pending")));

        // Batch 4 dies before committing: the restart must write it, once.
        topic.Fault = id => id == 4 ? "before" : null;
        await Assert.ThrowsAsync<IOException>(() => ToFake(stream, topic, options).RunAsync(checkpoint));
        Assert.Equal(160, topic.Messages.Count);

        topic.Fault = null;
        var stats = await ToFake(stream, topic, options).RunAsync(checkpoint);
        var all = Enumerable.Range(0, 3).SelectMany(p => Enumerable.Range(0, 70).Select(o => (double)(p * 1000 + o))).Order().ToArray();
        Assert.Equal(all, Orders(topic).Order());                              // every order, none twice
        Assert.Equal(210 - 160, stats.InputRows);
        Assert.Equal(3, topic.Starts);
        Assert.False(File.Exists(Path.Combine(checkpoint, "checkpoint.json.pending")));
        Assert.All(topic.Messages, m => Assert.StartsWith("k", m.Key));       // the key column became the key...
        Assert.All(topic.Messages, m => Assert.DoesNotContain("\"key\"", m.Value, StringComparison.Ordinal));   // ...and left the value
        Assert.Equal(new[] { 70L, 70, 70 }, Enumerable.Range(0, 3).Select(p => broker.Committed[("orders", p)]));
    }

    [Fact]
    public async Task Windows_through_a_transactional_sink_are_emitted_exactly_once()
    {
        // Window state lives in the checkpoint, so promoting a pending checkpoint whose batch did
        // not land (or discarding one whose batch did) would lose or repeat whole windows.
        var broker = Orders(partitions: 3, perPartition: 70);
        var expected = new Dictionary<(string, string), double>();
        for (var p = 0; p < 3; p++)
            for (var o = 0; o < 70; o++)
            {
                var start = T0.AddSeconds(o / 10 * 10).ToString("o");
                var k = (start, $"k{o % 3}");
                expected[k] = expected.GetValueOrDefault(k) + o % 17;
            }

        var topic = new FakeTopic();
        var checkpoint = Path.Combine(_root, "windows");
        var stream = KafkaStreams.FromKafkaConsumer(() => new FakeConsumer(broker), MessageFormat.Json(), KafkaTestOptions())
            .Window("ts", TimeSpan.FromSeconds(10), ["key"], [("amount", "sum")], allowedLateness: TimeSpan.FromHours(1));
        var options = new KafkaSinkOptions { TransactionalId = "windows" };
        var crashes = new Queue<(long, string)>([(1, "after"), (2, "before"), (3, "after")]);
        while (true)
        {
            var (at, how) = crashes.Count > 0 ? crashes.Dequeue() : (-1, "");
            topic.Fault = id => id == at ? how : null;
            try
            {
                await ToFake(stream, topic, options).RunAsync(checkpoint);
                if (at < 0) break;
            }
            catch (IOException) { }
        }

        var got = new Dictionary<(string, string), double>();
        foreach (var m in topic.Messages)
        {
            var row = JsonDocument.Parse(m.Value).RootElement;
            var key = (DateTime.Parse(row.GetProperty("window_start").GetString()!, null, System.Globalization.DateTimeStyles.RoundtripKind).ToString("o"), row.GetProperty("key").GetString()!);
            Assert.True(got.TryAdd(key, row.GetProperty("amount_sum").GetDouble()), $"window {key} emitted twice");
        }
        Assert.Equal(expected.OrderBy(e => e.Key), got.OrderBy(e => e.Key));
    }

    [Fact]
    public async Task Without_a_transactional_id_the_sink_is_plain_and_at_least_once()
    {
        var broker = Orders(partitions: 2, perPartition: 30);
        var topic = new FakeTopic();
        var stream = KafkaStreams.FromKafkaConsumer(() => new FakeConsumer(broker), MessageFormat.Json(), KafkaTestOptions());
        var checkpoint = Path.Combine(_root, "plain");
        topic.Fault = id => id == 1 ? "after" : null;                         // written, then the run dies
        await Assert.ThrowsAsync<IOException>(() => ToFake(stream, topic, new KafkaSinkOptions()).RunAsync(checkpoint));
        topic.Fault = null;
        await ToFake(stream, topic, new KafkaSinkOptions()).RunAsync(checkpoint);
        var orders = Orders(topic);
        Assert.Equal(60, orders.Distinct().Count());                           // nothing lost...
        Assert.Equal(60 + 20, orders.Length);                                  // ...batch 1 written twice
        Assert.All(topic.Messages, m => Assert.Null(m.Key));
    }

    [Fact]
    public void Rows_encode_as_json_objects()
    {
        var frame = new DataFrame([
            new NumericSeries("id", [1, 2.5, double.NaN]),
            new TextSeries("name", ["a", null, "c\"q"]),
            new BooleanSeries("ok", [true, null, false]),
            new DateTimeSeries("at", [new DateTime(2026, 1, 2, 3, 4, 5, DateTimeKind.Utc), null, null]),
        ]);
        var messages = KafkaEncoding.Encode(frame, "name");
        Assert.Equal(["a", null, "c\"q"], messages.Select(m => m.Key));
        Assert.Equal("{\"id\":1,\"ok\":true,\"at\":\"2026-01-02T03:04:05.0000000Z\"}", messages[0].Value);
        Assert.Equal("{\"id\":2.5,\"ok\":null,\"at\":null}", messages[1].Value);
        Assert.Equal("{\"id\":null,\"ok\":false,\"at\":null}", messages[2].Value);
        Assert.Throws<ArgumentException>(() => KafkaEncoding.Encode(frame, "nope"));
    }

    // ------------------------------------------------------------------ a real broker (opt-in)

    private static string Servers()
    {
        var servers = Environment.GetEnvironmentVariable("DASKNET_KAFKA");
        if (string.IsNullOrEmpty(servers)) Assert.Skip("Set DASKNET_KAFKA to a broker's bootstrap servers to run this test.");
        return servers;
    }

    private static async Task<string> Topic(string servers, int partitions)
    {
        var topic = "dasknet-test-" + Guid.NewGuid().ToString("N")[..8];
        using var admin = new AdminClientBuilder(new AdminClientConfig { BootstrapServers = servers }).Build();
        await admin.CreateTopicsAsync([new Confluent.Kafka.Admin.TopicSpecification { Name = topic, NumPartitions = partitions, ReplicationFactor = 1 }]);
        return topic;
    }

    private static async Task Produce(string servers, string topic, IEnumerable<(string Key, string Value)> messages)
    {
        using var producer = new ProducerBuilder<string, string>(new ProducerConfig { BootstrapServers = servers }).Build();
        foreach (var (k, v) in messages) await producer.ProduceAsync(topic, new Message<string, string> { Key = k, Value = v });
        producer.Flush(TimeSpan.FromSeconds(10));
    }

    [Fact]
    public async Task Kafka_source_against_a_real_broker()
    {
        var servers = Servers();
        var topic = await Topic(servers, 3);
        await Produce(servers, topic, Enumerable.Range(0, 500).Select(i => ($"k{i % 7}", $"{{\"n\": {i}, \"region\": \"r{i % 4}\"}}")));
        var checkpoint = Path.Combine(_root, "real-kafka");
        var options = new KafkaOptions { MaxBatchRows = 100, MaxBatchDelay = TimeSpan.FromSeconds(1), IdleTimeout = TimeSpan.FromSeconds(15), KeyColumn = "key" };
        var first = DaskStream.FromKafka(servers, [topic], "dasknet-tests", MessageFormat.Json(), options).Collect();
        var stats = await first.RunAsync(checkpoint);
        Assert.Equal(500, stats.InputRows);
        var frame = first.CollectedFrame();
        Assert.Equal(Enumerable.Range(0, 500).Select(i => (double)i), Enumerable.Range(0, frame.RowCount).Select(i => frame.Numeric("n")[i]).Order());

        await Produce(servers, topic, Enumerable.Range(500, 20).Select(i => ("k", $"{{\"n\": {i}, \"region\": \"r0\"}}")));
        var second = DaskStream.FromKafka(servers, [topic], "dasknet-tests", MessageFormat.Json(), options).Collect();
        var again = await second.RunAsync(checkpoint);
        Assert.Equal(20, again.InputRows);
    }

    [Fact]
    public async Task Kafka_to_kafka_is_exactly_once_against_a_real_broker()
    {
        var servers = Servers();
        var input = await Topic(servers, 3);
        var output = await Topic(servers, 2);
        await Produce(servers, input, Enumerable.Range(0, 400).Select(i => ($"k{i % 5}", $"{{\"n\": {i}}}")));
        var checkpoint = Path.Combine(_root, "real-eos");
        var source = new KafkaOptions { MaxBatchRows = 50, MaxBatchDelay = TimeSpan.FromSeconds(1), IdleTimeout = TimeSpan.FromSeconds(10), KeyColumn = "key" };
        var sink = new KafkaSinkOptions { TransactionalId = "dasknet-eos-" + output, KeyColumn = "key" };

        // The first run dies in the middle of its third batch, before that batch's transaction.
        var batches = 0;
        var failing = DaskStream.FromKafka(servers, [input], "eos-" + output, MessageFormat.Json(), source)
            .Map(f => ++batches == 3 ? throw new IOException("worker died") : f)
            .WithColumn("doubled", Gravicode.Science.DaskNet.Expr.Col("n") * 2);
        var died = await Assert.ThrowsAsync<DaskTaskException>(() => failing.ToKafka(servers, output, sink).RunAsync(checkpoint));
        Assert.IsType<IOException>(died.InnerException);          // thrown inside the batch's frame task
        var resumed = DaskStream.FromKafka(servers, [input], "eos-" + output, MessageFormat.Json(), source)
            .WithColumn("doubled", Gravicode.Science.DaskNet.Expr.Col("n") * 2);
        var stats = await resumed.ToKafka(servers, output, sink).RunAsync(checkpoint);
        Assert.Equal(400 - 100, stats.InputRows);

        // Read the output committed-only: every n exactly once, with its key.
        var seen = new List<double>();
        using (var consumer = new ConsumerBuilder<string, string>(new ConsumerConfig
        {
            BootstrapServers = servers, GroupId = "verify-" + output, AutoOffsetReset = AutoOffsetReset.Earliest,
            IsolationLevel = IsolationLevel.ReadCommitted, EnableAutoCommit = false,
        }).Build())
        {
            consumer.Subscribe(output);
            var idle = DateTime.UtcNow;
            while (DateTime.UtcNow - idle < TimeSpan.FromSeconds(8))
            {
                var r = consumer.Consume(TimeSpan.FromMilliseconds(500));
                if (r?.Message is null) continue;
                idle = DateTime.UtcNow;
                var row = JsonDocument.Parse(r.Message.Value).RootElement;
                Assert.Equal(row.GetProperty("n").GetDouble() * 2, row.GetProperty("doubled").GetDouble());
                Assert.Equal($"k{(int)row.GetProperty("n").GetDouble() % 5}", r.Message.Key);
                seen.Add(row.GetProperty("n").GetDouble());
            }
        }
        Assert.Equal(Enumerable.Range(0, 400).Select(i => (double)i), seen.Order());

        // The sink's record of the last batch is what a restart reads, and it agrees with the
        // checkpoint. How many batches there were depends on timing (a batch may close on
        // MaxBatchDelay), so the count itself is not pinned - only that both sides agree.
        using var producer = new ConfluentProducer(servers, output, sink);
        producer.Start();
        Assert.Equal(Checkpoint.ReadBatches(Path.Combine(Path.GetFullPath(checkpoint), "checkpoint.json")) - 1, producer.LastCommittedBatch());
        Assert.True(producer.LastCommittedBatch() >= 7, "400 rows in batches of at most 50");
    }
}
