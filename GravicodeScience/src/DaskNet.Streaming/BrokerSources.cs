using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Channels;
using Confluent.Kafka;
using MQTTnet;
using MQTTnet.Protocol;

namespace Gravicode.Science.DaskNet.Streaming;

/// <summary>Connection and batching settings for <see cref="DaskStream.FromMqtt"/>.</summary>
public sealed class MqttOptions : MessageSourceOptions
{
    /// <summary>The broker's port.</summary>
    public int Port { get; set; } = 1883;

    /// <summary>The client id; a random one by default. Give a fixed one with
    /// <see cref="CleanSession"/> off for the broker to queue QoS 1/2 messages while the query is down.</summary>
    public string? ClientId { get; set; }

    /// <summary>Start a fresh session (the default), or resume the broker-side one.</summary>
    public bool CleanSession { get; set; } = true;

    /// <summary>Subscription quality of service: 0 at most once, 1 at least once, 2 exactly once.</summary>
    public int QualityOfService { get; set; } = 1;

    /// <summary>User name, if the broker requires one.</summary>
    public string? UserName { get; set; }

    /// <summary>Password, with <see cref="UserName"/>.</summary>
    public string? Password { get; set; }

    /// <summary>Connect over TLS (validated against the system's trust store).</summary>
    public bool UseTls { get; set; }

    /// <summary>Messages buffered between the broker connection and the query; a full buffer
    /// stops reading from the socket, which is the backpressure MQTT has.</summary>
    public int BufferCapacity { get; set; } = 10_000;
}

/// <summary>Connection and batching settings for <see cref="DaskStream.FromKafka"/>.</summary>
public sealed class KafkaOptions : MessageSourceOptions
{
    /// <summary>Where a partition with no checkpointed or committed offset starts: the earliest
    /// retained message (the default) or only new ones.</summary>
    public bool StartFromEarliest { get; set; } = true;

    /// <summary>A column holding each message's key as text; null for none.</summary>
    public string? KeyColumn { get; set; }

    /// <summary>A column holding each message's partition; null for none.</summary>
    public string? PartitionColumn { get; set; }

    /// <summary>A column holding each message's offset; null for none.</summary>
    public string? OffsetColumn { get; set; }

    /// <summary>Further librdkafka settings (security.protocol, sasl.*, ...), passed through as given.</summary>
    public IDictionary<string, string> Config { get; } = new Dictionary<string, string>();
}

/// <summary>The MQTT source: subscribes, decodes, batches.</summary>
internal sealed class MqttSource(string host, string[] topics, MessageFormat format, MqttOptions options) : StreamSource
{
    internal override async IAsyncEnumerable<SourceBatch> ReadAsync(SourceState state, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var decoder = new MessageDecoder(format, options);
        var channel = Channel.CreateBounded<RawMessage>(new BoundedChannelOptions(options.BufferCapacity) { SingleReader = true, FullMode = BoundedChannelFullMode.Wait });
        var factory = new MqttClientFactory();
        using var client = factory.CreateMqttClient();
        client.ApplicationMessageReceivedAsync += async e =>
        {
            var m = e.ApplicationMessage;
            var text = Encoding.UTF8.GetString(m.Payload);
            await channel.Writer.WriteAsync(new RawMessage(m.Topic, text, DateTime.UtcNow), cancellationToken).ConfigureAwait(false);
        };
        var builder = new MqttClientOptionsBuilder()
            .WithTcpServer(host, options.Port)
            .WithClientId(options.ClientId ?? "dasknet-" + Guid.NewGuid().ToString("N"))
            .WithCleanSession(options.CleanSession);
        if (options.UserName is not null) builder = builder.WithCredentials(options.UserName, options.Password);
        if (options.UseTls) builder = builder.WithTlsOptions(o => o.UseTls());
        await client.ConnectAsync(builder.Build(), cancellationToken).ConfigureAwait(false);
        var subscribe = factory.CreateSubscribeOptionsBuilder();
        foreach (var t in topics)
            subscribe = subscribe.WithTopicFilter(f => f.WithTopic(t).WithQualityOfServiceLevel((MqttQualityOfServiceLevel)options.QualityOfService));
        await client.SubscribeAsync(subscribe.Build(), cancellationToken).ConfigureAwait(false);

        try
        {
            await foreach (var batch in MessageDecoder.Batches((timeout, ct) => ReadAsync(channel.Reader, timeout, ct), options, cancellationToken).ConfigureAwait(false))
                yield return new SourceBatch(decoder.Decode(batch), []);
        }
        finally
        {
            if (client.IsConnected)
                try { await client.DisconnectAsync().ConfigureAwait(false); } catch (Exception) { /* closing anyway */ }
        }
    }

    internal static async ValueTask<RawMessage?> ReadAsync(ChannelReader<RawMessage> reader, TimeSpan timeout, CancellationToken cancellationToken)
    {
        if (reader.TryRead(out var ready)) return ready;
        using var timer = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        if (timeout != Timeout.InfiniteTimeSpan) timer.CancelAfter(timeout);
        try { return await reader.ReadAsync(timer.Token).ConfigureAwait(false); }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested) { return null; }
        catch (ChannelClosedException) { return null; }
    }
}

/// <summary>What the Kafka source needs from a consumer, so tests can model one in memory.</summary>
internal interface IKafkaConsumer : IDisposable
{
    /// <summary>Joins the group; partitions with an entry in <paramref name="resume"/> start at that
    /// offset, others at the group's committed offset or the reset policy.</summary>
    void Start(IReadOnlyDictionary<(string Topic, int Partition), long> resume);

    /// <summary>The next message, or null if none arrives within <paramref name="timeout"/>.</summary>
    RawMessage? Consume(TimeSpan timeout, CancellationToken cancellationToken);

    /// <summary>Commits the next offsets to read to the consumer group.</summary>
    void Commit(IReadOnlyDictionary<(string Topic, int Partition), long> next);
}

/// <summary>The Confluent (librdkafka) consumer behind <see cref="IKafkaConsumer"/>.</summary>
internal sealed class ConfluentConsumer(string servers, string[] topics, string groupId, KafkaOptions options) : IKafkaConsumer
{
    private IConsumer<byte[]?, byte[]?>? _consumer;

    public void Start(IReadOnlyDictionary<(string Topic, int Partition), long> resume)
    {
        var config = new ConsumerConfig
        {
            BootstrapServers = servers,
            GroupId = groupId,
            EnableAutoCommit = false,
            EnableAutoOffsetStore = false,
            AutoOffsetReset = options.StartFromEarliest ? AutoOffsetReset.Earliest : AutoOffsetReset.Latest,
        };
        foreach (var (k, v) in options.Config) config.Set(k, v);
        _consumer = new ConsumerBuilder<byte[]?, byte[]?>(config)
            .SetKeyDeserializer(Deserializers.ByteArray)
            .SetValueDeserializer(Deserializers.ByteArray)
            .SetPartitionsAssignedHandler((_, assigned) => assigned.Select(tp =>
                resume.TryGetValue((tp.Topic, tp.Partition.Value), out var o) ? new TopicPartitionOffset(tp, new Offset(o)) : new TopicPartitionOffset(tp, Offset.Unset)))
            .Build();
        _consumer.Subscribe(topics);
    }

    public RawMessage? Consume(TimeSpan timeout, CancellationToken cancellationToken)
    {
        var consumer = _consumer ?? throw new InvalidOperationException("Start the consumer first.");
        var deadline = timeout == Timeout.InfiniteTimeSpan ? DateTime.MaxValue : DateTime.UtcNow + timeout;
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var left = deadline - DateTime.UtcNow;
            if (left <= TimeSpan.Zero) return null;
            var slice = left < TimeSpan.FromMilliseconds(200) ? left : TimeSpan.FromMilliseconds(200);   // stay cancellable
            var r = consumer.Consume(slice);
            if (r is null || r.IsPartitionEOF || r.Message is null) continue;
            var value = r.Message.Value is { } v ? Encoding.UTF8.GetString(v) : "";
            var key = r.Message.Key is { } k ? Encoding.UTF8.GetString(k) : null;
            var time = r.Message.Timestamp.Type == TimestampType.NotAvailable ? (DateTime?)null : r.Message.Timestamp.UtcDateTime;
            return new RawMessage(r.Topic, value, time, key, r.Partition.Value, r.Offset.Value);
        }
    }

    public void Commit(IReadOnlyDictionary<(string Topic, int Partition), long> next)
    {
        if (_consumer is null || next.Count == 0) return;
        try { _consumer.Commit(next.Select(kv => new TopicPartitionOffset(kv.Key.Topic, new Partition(kv.Key.Partition), new Offset(kv.Value)))); }
        catch (KafkaException) { /* a rebalance took the partition; the checkpoint still has the position */ }
    }

    public void Dispose()
    {
        if (_consumer is null) return;
        try { _consumer.Close(); } catch (KafkaException) { }
        _consumer.Dispose();
    }
}

/// <summary>
/// The Kafka source. Each batch carries the next offset per partition; the query saves them in its
/// checkpoint and commits them to the group only after the batch's sink has returned.
/// </summary>
internal sealed class KafkaSource(Func<IKafkaConsumer> create, MessageFormat format, KafkaOptions options) : StreamSource
{
    internal override async IAsyncEnumerable<SourceBatch> ReadAsync(SourceState state, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var extra = new List<(string, Func<RawMessage, object?>, ColumnType)>();
        if (options.KeyColumn is { } key) extra.Add((key, m => m.Key, ColumnType.Text));
        if (options.PartitionColumn is { } partition) extra.Add((partition, m => (double)m.Partition, ColumnType.Number));
        if (options.OffsetColumn is { } offset) extra.Add((offset, m => (double)m.Offset, ColumnType.Number));
        var decoder = new MessageDecoder(format, options, [.. extra]);

        Dictionary<(string, int), long> positions;
        lock (state) positions = new(state.Offsets);
        using var consumer = create();
        consumer.Start(positions);

        ValueTask<RawMessage?> Next(TimeSpan timeout, CancellationToken ct)
        {
            // A message below a known position was already processed (a rebalance or a broker that
            // ignored the requested start): skip it rather than deliver it twice.
            var deadline = timeout == Timeout.InfiniteTimeSpan ? DateTime.MaxValue : DateTime.UtcNow + timeout;
            while (true)
            {
                var left = deadline == DateTime.MaxValue ? Timeout.InfiniteTimeSpan : deadline - DateTime.UtcNow;
                if (left != Timeout.InfiniteTimeSpan && left <= TimeSpan.Zero) return ValueTask.FromResult<RawMessage?>(null);
                var m = consumer.Consume(left, ct);
                if (m is null) return ValueTask.FromResult<RawMessage?>(null);
                if (positions.TryGetValue((m.Topic, m.Partition), out var p) && m.Offset < p) continue;
                return ValueTask.FromResult<RawMessage?>(m);
            }
        }

        await foreach (var batch in MessageDecoder.Batches(Next, options, cancellationToken).ConfigureAwait(false))
        {
            foreach (var m in batch) positions[(m.Topic, m.Partition)] = Math.Max(positions.GetValueOrDefault((m.Topic, m.Partition)), m.Offset + 1);
            var next = new Dictionary<(string, int), long>(positions);
            yield return new SourceBatch(decoder.Decode(batch), [], next, () => consumer.Commit(next));
        }
    }
}
