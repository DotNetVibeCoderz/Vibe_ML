using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Gravicode.Science.DaskNet.Sql;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;

namespace Gravicode.Science.DaskNet.Streaming;

/// <summary>Where a stream's batches come from.</summary>
public abstract class StreamSource
{
    /// <summary>The batches, skipping whatever <paramref name="state"/> says was already consumed.</summary>
    internal abstract IAsyncEnumerable<SourceBatch> ReadAsync(SourceState state, CancellationToken cancellationToken);
}

/// <summary>One batch from a source: the files it came from, or the next offset per Kafka
/// partition after it (both for the checkpoint), and what to call once it is safely processed
/// (commit Kafka offsets, acknowledge MQTT messages).</summary>
internal sealed record SourceBatch(DataFrame Frame, string[] Files,
    IReadOnlyDictionary<(string Topic, int Partition), long>? Offsets = null, Func<Task>? Commit = null);

/// <summary>What a source has consumed, carried in the checkpoint. Lock the instance to read or
/// change it: a joined query's sources run beside the loop that records their progress.</summary>
internal sealed class SourceState
{
    public HashSet<string> Files { get; } = new(StringComparer.OrdinalIgnoreCase);
    public Dictionary<(string Topic, int Partition), long> Offsets { get; } = [];
    public long Batches { get; set; }

    /// <summary>Records a processed batch.</summary>
    public void Consumed(SourceBatch batch)
    {
        lock (this)
        {
            foreach (var f in batch.Files) Files.Add(f);
            if (batch.Offsets is { } offsets)
                foreach (var (k, v) in offsets) Offsets[k] = v;
        }
    }
}

/// <summary>
/// A micro-batch stream of frames (Streamz / Spark Structured Streaming): build it with stateless
/// steps, optionally a window, then a sink, and <see cref="StreamQuery.RunAsync"/> it.
/// </summary>
/// <remarks>
/// Stateless steps (<see cref="Filter"/>, <see cref="WithColumns"/>, <see cref="Select(string[])"/>,
/// <see cref="Sql"/>) run on each batch as a <see cref="DaskFrame"/> of
/// <see cref="Partitions"/> partitions on the query's scheduler, so a
/// <see cref="Distributed.Client"/> spreads every batch over a cluster. <see cref="Map"/> runs a
/// function in-process and keeps a query local. Steps apply in the order they are added; the
/// builder is immutable, so a stream can fan out into several queries.
/// </remarks>
public sealed class DaskStream
{
    internal StreamSource Source { get; }
    internal IReadOnlyList<Func<DaskFrame, DaskFrame>> Steps { get; }
    internal int PartitionCount { get; }
    internal IScheduler? Scheduler { get; }

    private DaskStream(StreamSource source, IReadOnlyList<Func<DaskFrame, DaskFrame>> steps, int partitions, IScheduler? scheduler)
    {
        Source = source;
        Steps = steps;
        PartitionCount = partitions;
        Scheduler = scheduler;
    }

    private DaskStream With(Func<DaskFrame, DaskFrame> step) => new(Source, [.. Steps, step], PartitionCount, Scheduler);

    // ------------------------------------------------------------------ sources

    /// <summary>A stream over any asynchronous sequence of frames.</summary>
    public static DaskStream From(IAsyncEnumerable<DataFrame> batches) => new(new EnumerableSource(batches), [], 1, null);

    /// <summary>A stream over a finite sequence of frames (tests, replays).</summary>
    public static DaskStream From(IEnumerable<DataFrame> batches) => From(ToAsync(batches));

    /// <summary>A stream that ends when the channel's writer completes.</summary>
    public static DaskStream FromChannel(ChannelReader<DataFrame> reader) => From(reader.ReadAllAsync());

    /// <summary>
    /// Incremental files: every CSV file in <paramref name="directory"/> matching
    /// <paramref name="pattern"/>, in name order, each read once - a checkpoint remembers which
    /// files were consumed, so a restarted query picks up only new ones. With
    /// <paramref name="watch"/> the directory is polled every <paramref name="pollInterval"/>
    /// until the query is cancelled; without it the stream ends once the files present are read.
    /// </summary>
    /// <remarks>Write each file under another name (or another folder) and rename it into place:
    /// a file is read as soon as it is seen, so a half-written one would be read half.</remarks>
    public static DaskStream FromDirectory(string directory, string pattern = "*.csv", bool watch = false,
        TimeSpan? pollInterval = null, CsvOptions? options = null, int filesPerBatch = 1)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        ArgumentOutOfRangeException.ThrowIfLessThan(filesPerBatch, 1);
        return new(new DirectorySource(Path.GetFullPath(directory), pattern, watch, pollInterval ?? TimeSpan.FromSeconds(1), options, filesPerBatch), [], 1, null);
    }

    /// <summary>
    /// An MQTT subscription: every message on <paramref name="topics"/> (filters, so <c>+</c> and
    /// <c>#</c> work) becomes a row, decoded by <paramref name="format"/>, in batches of up to
    /// <see cref="MessageSourceOptions.MaxBatchRows"/> messages or
    /// <see cref="MessageSourceOptions.MaxBatchDelay"/>.
    /// </summary>
    /// <remarks>MQTT cannot replay, so a checkpoint keeps the windows and the batch count but not
    /// a position: what is published while the query is down is lost, unless a fixed
    /// <see cref="MqttOptions.ClientId"/> with <see cref="MqttOptions.CleanSession"/> off has the
    /// broker queue it (QoS 1 or 2). QoS 1/2 messages are acknowledged only after their batch's
    /// sink has returned and the checkpoint is saved (<see cref="MqttOptions.AcknowledgeAfterSink"/>),
    /// so with such a session a crash makes the broker redeliver them: at-least-once.</remarks>
    public static DaskStream FromMqtt(string host, string[] topics, MessageFormat format, MqttOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(host);
        ArgumentNullException.ThrowIfNull(topics);
        if (topics.Length == 0) throw new ArgumentException("Subscribe to at least one topic.", nameof(topics));
        ArgumentNullException.ThrowIfNull(format);
        options ??= new MqttOptions();
        options.Validate();
        if (options.QualityOfService is < 0 or > 2) throw new ArgumentOutOfRangeException(nameof(options), "QoS is 0, 1 or 2.");
        return new(new MqttSource(host, topics, format, options), [], 1, null);
    }

    /// <summary>A stream over a source defined in another package (Kafka).</summary>
    internal static DaskStream FromSource(StreamSource source) => new(source, [], 1, null);

    // ------------------------------------------------------------------ steps

    /// <summary>Keeps the rows matching <paramref name="predicate"/>.</summary>
    public DaskStream Filter(Expr predicate) => With(f => f.Filter(predicate));

    /// <summary>Adds or replaces columns.</summary>
    public DaskStream WithColumns(params Expr[] items) => With(f => f.WithColumns(items));

    /// <summary>Adds or replaces one column.</summary>
    public DaskStream WithColumn(string name, Expr value) => With(f => f.WithColumn(name, value));

    /// <summary>Keeps (and orders) columns.</summary>
    public DaskStream Select(params string[] columns) => With(f => f.Select(columns));

    /// <summary>Projects expressions.</summary>
    public DaskStream Select(params Expr[] items) => With(f => f.Select(items));

    /// <summary>Drops columns.</summary>
    public DaskStream Drop(params string[] columns) => With(f => f.Drop(columns));

    /// <summary>
    /// Runs a SQL query over each batch, which is registered as <paramref name="table"/>. Joins
    /// against static tables work through <paramref name="context"/>: register them there first.
    /// Aggregates are per batch - use <see cref="Window"/> for aggregates over time.
    /// </summary>
    public DaskStream Sql(string query, string table = "batch", SqlContext? context = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(query);
        return With(f =>
        {
            var sql = context ?? new SqlContext();
            lock (sql)
            {
                sql.Register(table, f);
                return sql.Sql(query);
            }
        });
    }

    /// <summary>Applies a function to each batch, in-process (local schedulers only).</summary>
    public DaskStream Map(Func<DataFrame, DataFrame> function)
    {
        ArgumentNullException.ThrowIfNull(function);
        return With(f => f.MapPartitions(function));
    }

    /// <summary>Splits each batch into <paramref name="partitions"/> partitions for the scheduler.</summary>
    public DaskStream Partitions(int partitions)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(partitions, 1);
        return new(Source, Steps, partitions, Scheduler);
    }

    /// <summary>Runs every batch's steps on <paramref name="scheduler"/> (a cluster client, say).</summary>
    public DaskStream WithScheduler(IScheduler scheduler)
    {
        ArgumentNullException.ThrowIfNull(scheduler);
        return new(Source, Steps, PartitionCount, scheduler);
    }

    /// <summary>
    /// Event-time windowed aggregation. Rows are assigned to windows by <paramref name="timeColumn"/>;
    /// a window is emitted once the watermark (latest event time seen minus
    /// <paramref name="allowedLateness"/>) passes its end, and rows that arrive for an emitted
    /// window are dropped and counted as late. Pass <paramref name="slide"/> for sliding
    /// (hopping) windows; omit it for tumbling ones.
    /// </summary>
    /// <remarks>Aggregations are (column, how) with how one of sum, count, mean, min, max;
    /// (null, "count") counts rows. Outputs are named <c>{column}_{how}</c>, or <c>count</c>.</remarks>
    public WindowedStream Window(string timeColumn, TimeSpan size, string[] keys, (string? Column, string How)[] aggregations,
        TimeSpan? slide = null, TimeSpan allowedLateness = default)
        => new(this, new WindowSpec(timeColumn, size, slide ?? size, keys, aggregations, allowedLateness));

    /// <summary>
    /// Joins with another stream on equal <paramref name="on"/> keys and event times at most
    /// <paramref name="within"/> apart (Spark's stream-stream join). Both sides are buffered and
    /// evicted by each other's watermark (latest event time minus <paramref name="allowedLateness"/>),
    /// so state stays bounded; unmatched rows of an outer side are emitted once they can no longer
    /// match. Each side runs its own steps first.
    /// </summary>
    public JoinedStream Join(DaskStream right, string[] on, string leftTime, string rightTime, TimeSpan within,
        StreamJoinType how = StreamJoinType.Inner, TimeSpan allowedLateness = default, string suffix = "_right")
    {
        if (within < TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(within));
        return IntervalJoin(right, on, leftTime, rightTime, -within, within, how, allowedLateness, suffix);
    }

    /// <summary>
    /// Like <see cref="Join"/> with an asymmetric interval: a right row matches a left row when
    /// <c>rightTime - leftTime</c> is in [<paramref name="lower"/>, <paramref name="upper"/>] - for
    /// "a click within an hour after the impression", lower 0 and upper one hour.
    /// </summary>
    public JoinedStream IntervalJoin(DaskStream right, string[] on, string leftTime, string rightTime, TimeSpan lower, TimeSpan upper,
        StreamJoinType how = StreamJoinType.Inner, TimeSpan allowedLateness = default, string suffix = "_right")
        => new(this, new JoinSpec(right, on, leftTime, rightTime, lower, upper, how, allowedLateness, suffix));

    // ------------------------------------------------------------------ sinks

    /// <summary>Calls <paramref name="sink"/> with each non-empty output batch.</summary>
    public StreamQuery ForEachBatch(Func<DataFrame, StreamBatch, Task> sink) => new(this, null, sink);

    /// <summary>Keeps every output batch in <see cref="StreamQuery.Results"/>.</summary>
    public StreamQuery Collect() => new(this, null, null);

    /// <summary>
    /// Writes each output batch to <c>batch-{id}.csv</c> in <paramref name="directory"/>. Batch ids
    /// continue across checkpointed restarts, so a batch replayed after a crash overwrites its own
    /// file rather than duplicating it.
    /// </summary>
    public StreamQuery ToCsvDirectory(string directory) => new(this, null, StreamQuery.CsvSink(directory));

    internal async Task<DataFrame> ApplyAsync(DataFrame batch, CancellationToken cancellationToken)
    {
        if (Steps.Count == 0) return batch;
        var frame = DaskFrame.FromFrame(batch, Math.Max(1, Math.Min(PartitionCount, batch.RowCount)));
        foreach (var step in Steps) frame = step(frame);
        return await frame.ComputeAsync(Scheduler, cancellationToken).ConfigureAwait(false);
    }

    private static async IAsyncEnumerable<DataFrame> ToAsync(IEnumerable<DataFrame> batches)
    {
        foreach (var b in batches)
        {
            yield return b;
            await Task.Yield();
        }
    }

    private sealed class EnumerableSource(IAsyncEnumerable<DataFrame> batches) : StreamSource
    {
        internal override async IAsyncEnumerable<SourceBatch> ReadAsync(SourceState state, [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            await foreach (var b in batches.WithCancellation(cancellationToken).ConfigureAwait(false))
                yield return new SourceBatch(b, []);
        }
    }

    private sealed class DirectorySource(string directory, string pattern, bool watch, TimeSpan poll, CsvOptions? options, int perBatch) : StreamSource
    {
        internal override async IAsyncEnumerable<SourceBatch> ReadAsync(SourceState state, [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            while (true)
            {
                var present = Directory.Exists(directory) ? Directory.EnumerateFiles(directory, pattern).Select(n => Path.GetFileName(n)!).ToArray() : [];
                string[] fresh;
                lock (state) fresh = [.. present.Where(n => !state.Files.Contains(n)).Order(StringComparer.Ordinal)];
                foreach (var chunk in fresh.Chunk(perBatch))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var frames = chunk.Select(n => DataFrame.ReadCsv(Path.Combine(directory, n), options)).ToArray();
                    yield return new SourceBatch(frames.Length == 1 ? frames[0] : DataFrame.Concat(frames), chunk);
                }
                if (!watch) yield break;
                try { await Task.Delay(poll, cancellationToken).ConfigureAwait(false); }
                catch (OperationCanceledException) { yield break; }
            }
        }
    }
}

/// <summary>A stream with a window; finish it with a sink.</summary>
public sealed class WindowedStream
{
    internal WindowedStream(DaskStream stream, WindowSpec spec, JoinSpec? join = null)
    {
        spec.Validate();
        Stream = stream;
        Spec = spec;
        Join = join;
    }

    internal DaskStream Stream { get; }
    internal WindowSpec Spec { get; }
    internal JoinSpec? Join { get; }

    /// <summary>Calls <paramref name="sink"/> with each batch of closed windows.</summary>
    public StreamQuery ForEachBatch(Func<DataFrame, StreamBatch, Task> sink) => new(Stream, Spec, sink, Join);

    /// <summary>Keeps every batch of closed windows in <see cref="StreamQuery.Results"/>.</summary>
    public StreamQuery Collect() => new(Stream, Spec, null, Join);

    /// <summary>Writes each batch of closed windows to <c>batch-{id}.csv</c>.</summary>
    public StreamQuery ToCsvDirectory(string directory) => new(Stream, Spec, StreamQuery.CsvSink(directory), Join);
}

/// <summary>The batch a sink is given.</summary>
/// <param name="Id">The batch number, counting from 0 and continuing across restarts.</param>
/// <param name="InputRows">Rows the source delivered for this batch.</param>
/// <param name="Files">The source files it came from, for a directory source.</param>
/// <param name="Watermark">The event-time watermark after this batch, for a windowed query.</param>
public sealed record StreamBatch(long Id, int InputRows, IReadOnlyList<string> Files, DateTime? Watermark);
