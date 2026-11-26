using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Confluent.Kafka;
using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet.Streaming;

/// <summary>Settings for <see cref="KafkaStreams.ToKafka(DaskStream, string, string, KafkaSinkOptions?)"/>.</summary>
public sealed class KafkaSinkOptions
{
    /// <summary>
    /// A transactional id makes the sink exactly-once with a checkpoint: each batch is one
    /// transaction, which also records the batch's id, so a restart knows whether the batch it was
    /// writing landed. Keep it stable across restarts of one query and unique across queries -
    /// starting a producer with it fences off any older one still running. Null (the default)
    /// writes without transactions, which is at-least-once.
    /// </summary>
    public string? TransactionalId { get; set; }

    /// <summary>A column whose value (as text) becomes each message's key, and is left out of its
    /// value; null for keyless messages.</summary>
    public string? KeyColumn { get; set; }

    /// <summary>How long a transaction may stay open before the broker aborts it.</summary>
    public TimeSpan TransactionTimeout { get; set; } = TimeSpan.FromMinutes(1);

    /// <summary>Further librdkafka producer settings, passed through as given.</summary>
    public IDictionary<string, string> Config { get; } = new Dictionary<string, string>();
}

public static partial class KafkaStreams
{
    /// <summary>
    /// Writes each output row to <paramref name="topic"/> as a JSON object (numbers, text, booleans,
    /// ISO 8601 times; missing values as null). With <see cref="KafkaSinkOptions.TransactionalId"/>
    /// and a checkpoint directory, a restarted query neither loses nor repeats a row; read the
    /// topic with read-committed isolation (the default of <see cref="KafkaOptions"/>).
    /// </summary>
    public static StreamQuery ToKafka(this DaskStream stream, string bootstrapServers, string topic, KafkaSinkOptions? options = null)
        => Sink(bootstrapServers, topic, options, s => StreamQuery.For(stream, s), f => StreamQuery.For(stream, f));

    /// <summary>Writes each batch of closed windows to <paramref name="topic"/> (see the <see cref="DaskStream"/> overload).</summary>
    public static StreamQuery ToKafka(this WindowedStream stream, string bootstrapServers, string topic, KafkaSinkOptions? options = null)
        => Sink(bootstrapServers, topic, options, s => StreamQuery.For(stream, s), f => StreamQuery.For(stream, f));

    /// <summary>Writes each batch of joined rows to <paramref name="topic"/> (see the <see cref="DaskStream"/> overload).</summary>
    public static StreamQuery ToKafka(this JoinedStream stream, string bootstrapServers, string topic, KafkaSinkOptions? options = null)
        => Sink(bootstrapServers, topic, options, s => StreamQuery.For(stream, s), f => StreamQuery.For(stream, f));

    private static StreamQuery Sink(string servers, string topic, KafkaSinkOptions? options,
        Func<ITransactionalSink, StreamQuery> transactional, Func<Func<DataFrame, StreamBatch, Task>, StreamQuery> plain)
    {
        ArgumentException.ThrowIfNullOrEmpty(servers);
        ArgumentException.ThrowIfNullOrEmpty(topic);
        options ??= new KafkaSinkOptions();
        return WithProducer(() => new ConfluentProducer(servers, topic, options), options, transactional, plain);
    }

    /// <summary>A Kafka sink over any producer (tests model a broker's transactions in memory).</summary>
    internal static StreamQuery WithProducer(Func<IKafkaProducer> create, KafkaSinkOptions options,
        Func<ITransactionalSink, StreamQuery> transactional, Func<Func<DataFrame, StreamBatch, Task>, StreamQuery> plain)
    {
        if (options.TransactionalId is not null) return transactional(new KafkaTransactionalSink(create, options));
        return plain((frame, batch) =>
        {
            using var producer = create();
            producer.Start();
            producer.Write(KafkaEncoding.Encode(frame, options.KeyColumn), batch.Id);
            return Task.CompletedTask;
        });
    }
}

/// <summary>What the sink needs from a producer.</summary>
internal interface IKafkaProducer : IDisposable
{
    /// <summary>Connects; a transactional producer fences older ones with its id, which aborts
    /// whatever transaction they left open.</summary>
    void Start();

    /// <summary>The id of the last batch committed under this transactional id, or -1.</summary>
    long LastCommittedBatch();

    /// <summary>Writes the messages - in one transaction that also records <paramref name="batchId"/>,
    /// for a transactional producer - and returns once they are acknowledged.</summary>
    void Write(IReadOnlyList<(string? Key, string Value)> messages, long batchId);
}

internal sealed class KafkaTransactionalSink(Func<IKafkaProducer> create, KafkaSinkOptions options) : ITransactionalSink
{
    public ITransactionalSession Open()
    {
        var producer = create();
        try { producer.Start(); }
        catch
        {
            producer.Dispose();
            throw;
        }
        return new Session(producer, options);
    }

    private sealed class Session(IKafkaProducer producer, KafkaSinkOptions options) : ITransactionalSession
    {
        public long LastCommitted() => producer.LastCommittedBatch();
        public void Write(DataFrame output, StreamBatch batch) => producer.Write(KafkaEncoding.Encode(output, options.KeyColumn), batch.Id);
        public void Dispose() => producer.Dispose();
    }
}

/// <summary>
/// The Confluent producer. A transactional one records each committed batch as a consumer-group
/// offset - group <c>{transactionalId}.dasknet</c>, partition 0 of the output topic, offset
/// batch id + 1 - committed inside the batch's own transaction, so the record and the rows land
/// together or not at all (Kafka does not check an offset commit against the log).
/// </summary>
internal sealed class ConfluentProducer(string servers, string topic, KafkaSinkOptions options) : IKafkaProducer
{
    private IProducer<byte[]?, byte[]>? _producer;
    private IConsumer<Ignore, Ignore>? _marker;
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(30);

    private bool Transactional => options.TransactionalId is not null;
    private string MarkerGroup => options.TransactionalId + ".dasknet";

    public void Start()
    {
        var config = new ProducerConfig
        {
            BootstrapServers = servers,
            EnableIdempotence = true,
            Acks = Acks.All,
            LingerMs = 5,
        };
        if (Transactional)
        {
            config.TransactionalId = options.TransactionalId;
            config.TransactionTimeoutMs = (int)options.TransactionTimeout.TotalMilliseconds;
        }
        foreach (var (k, v) in options.Config) config.Set(k, v);
        _producer = new ProducerBuilder<byte[]?, byte[]>(config).SetKeySerializer(Serializers.ByteArray).SetValueSerializer(Serializers.ByteArray).Build();
        if (!Transactional) return;
        _producer.InitTransactions(Timeout);
        _marker = new ConsumerBuilder<Ignore, Ignore>(new ConsumerConfig
        {
            BootstrapServers = servers,
            GroupId = MarkerGroup,
            EnableAutoCommit = false,
            IsolationLevel = IsolationLevel.ReadCommitted,   // committed offsets only, never a pending transaction's
        }).Build();
    }

    public long LastCommittedBatch()
    {
        if (_marker is null) return -1;
        var committed = _marker.Committed([new TopicPartition(topic, new Partition(0))], Timeout);
        var offset = committed.Count == 0 ? Offset.Unset : committed[0].Offset;
        return offset.IsSpecial ? -1 : offset.Value - 1;
    }

    public void Write(IReadOnlyList<(string? Key, string Value)> messages, long batchId)
    {
        var producer = _producer ?? throw new InvalidOperationException("Start the producer first.");
        Error? failure = null;
        void Delivered(DeliveryReport<byte[]?, byte[]> r)
        {
            if (r.Error.IsError) failure ??= r.Error;
        }

        if (Transactional) producer.BeginTransaction();
        try
        {
            foreach (var (key, value) in messages)
            {
                var message = new Message<byte[]?, byte[]> { Key = key is null ? null : Encoding.UTF8.GetBytes(key), Value = Encoding.UTF8.GetBytes(value) };
                while (true)
                {
                    try
                    {
                        producer.Produce(topic, message, Delivered);
                        break;
                    }
                    catch (ProduceException<byte[]?, byte[]> e) when (e.Error.Code == ErrorCode.Local_QueueFull)
                    {
                        producer.Poll(TimeSpan.FromMilliseconds(100));   // let deliveries drain the queue
                    }
                }
            }
            if (Transactional)
            {
                producer.SendOffsetsToTransaction([new TopicPartitionOffset(topic, new Partition(0), new Offset(batchId + 1))],
                    _marker!.ConsumerGroupMetadata, Timeout);
                producer.CommitTransaction(Timeout);                       // flushes, and fails if any delivery did
            }
            else
            {
                producer.Flush(Timeout);
                if (failure is { } e) throw new KafkaException(e);
            }
        }
        catch (KafkaException) when (Transactional)
        {
            try { producer.AbortTransaction(Timeout); } catch (KafkaException) { /* fenced or gone: the broker aborts it */ }
            throw;
        }
    }

    public void Dispose()
    {
        _marker?.Close();
        _marker?.Dispose();
        _producer?.Dispose();
    }
}

/// <summary>Rows to JSON message values.</summary>
internal static class KafkaEncoding
{
    public static List<(string? Key, string Value)> Encode(DataFrame frame, string? keyColumn)
    {
        if (keyColumn is not null && !frame.HasColumn(keyColumn) && frame.RowCount > 0)
            throw new ArgumentException($"The key column '{keyColumn}' is not in the output ({string.Join(", ", frame.ColumnNames)}).", nameof(keyColumn));
        var columns = frame.Columns.Where(c => c.Name != keyColumn).ToArray();
        var key = keyColumn is not null && frame.HasColumn(keyColumn) ? frame[keyColumn] : null;
        var messages = new List<(string?, string)>(frame.RowCount);
        var buffer = new ArrayBufferWriter<byte>();
        for (var i = 0; i < frame.RowCount; i++)
        {
            buffer.Clear();
            using (var w = new Utf8JsonWriter(buffer))
            {
                w.WriteStartObject();
                foreach (var c in columns)
                {
                    w.WritePropertyName(c.Name);
                    Write(w, c.IsMissing(i) ? null : c.GetValue(i));
                }
                w.WriteEndObject();
            }
            messages.Add((key is null || key.IsMissing(i) ? null : Text(key.GetValue(i)), Encoding.UTF8.GetString(buffer.WrittenSpan)));
        }
        return messages;
    }

    private static void Write(Utf8JsonWriter w, object? value)
    {
        switch (value)
        {
            case null: w.WriteNullValue(); break;
            case double d when double.IsFinite(d): w.WriteNumberValue(d); break;
            case double: w.WriteNullValue(); break;
            case bool b: w.WriteBooleanValue(b); break;
            case DateTime t: w.WriteStringValue(t.ToString("o", CultureInfo.InvariantCulture)); break;
            default: w.WriteStringValue(Text(value)); break;
        }
    }

    private static string? Text(object? value) => value switch
    {
        null => null,
        double d => d.ToString("R", CultureInfo.InvariantCulture),
        DateTime t => t.ToString("o", CultureInfo.InvariantCulture),
        _ => Convert.ToString(value, CultureInfo.InvariantCulture),
    };
}
