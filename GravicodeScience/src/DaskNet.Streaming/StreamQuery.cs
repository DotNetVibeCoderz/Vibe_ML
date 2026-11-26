using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Channels;
using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet.Streaming;

/// <summary>Counters for a query run.</summary>
public sealed class StreamStatistics
{
    /// <summary>Batches processed by this run (not counting those before a restart).</summary>
    public long Batches { get; internal set; }

    /// <summary>Rows the source delivered.</summary>
    public long InputRows { get; internal set; }

    /// <summary>Rows handed to the sink.</summary>
    public long OutputRows { get; internal set; }

    /// <summary>Rows dropped as late over the query's life: for a window, rows whose every window
    /// had closed; for a join, rows older than their own side's watermark.</summary>
    public long LateRows { get; internal set; }

    /// <summary>Windows emitted by this run.</summary>
    public long WindowsEmitted { get; internal set; }

    /// <summary>Rows a join holds in its state after the run (both sides).</summary>
    public long JoinStateRows { get; internal set; }

    /// <summary>Wall-clock time of the run.</summary>
    public TimeSpan Elapsed { get; internal set; }

    /// <inheritdoc/>
    public override string ToString()
        => $"{Batches} batch(es), {InputRows} row(s) in, {OutputRows} out, {LateRows} late, {Elapsed.TotalMilliseconds:F0} ms";
}

/// <summary>
/// A stream with a sink, ready to run. With a checkpoint directory the query is restartable: after
/// each batch's sink returns, the consumed files or Kafka offsets, the batch count, the open
/// windows and any join state are saved; a new run of the same query resumes from there.
/// </summary>
/// <remarks>
/// Delivery is at-least-once: a crash after the sink but before the checkpoint replays that batch.
/// The CSV sink names files by batch id, so a replay overwrites rather than duplicates - which makes
/// it effectively exactly-once. Other sinks should be idempotent by <see cref="StreamBatch.Id"/>.
/// </remarks>
public sealed class StreamQuery
{
    private readonly DaskStream _stream;
    private readonly WindowSpec? _window;
    private readonly JoinSpec? _join;
    private readonly Func<DataFrame, StreamBatch, Task>? _sink;
    private readonly List<DataFrame> _results = [];

    internal StreamQuery(DaskStream stream, WindowSpec? window, Func<DataFrame, StreamBatch, Task>? sink, JoinSpec? join = null)
    {
        _stream = stream;
        _window = window;
        _sink = sink;
        _join = join;
    }

    /// <summary>Output batches kept by a <c>Collect()</c> query.</summary>
    public IReadOnlyList<DataFrame> Results => _results;

    /// <summary>Every collected batch as one frame.</summary>
    public DataFrame CollectedFrame() => _results.Count == 0 ? DataFrame.Empty() : DataFrame.Concat(_results);

    /// <summary>Whether a windowed or outer-joined query emits what it still holds when a finite
    /// source ends. On by default; a cancelled run never flushes (the state stays in the checkpoint).</summary>
    public bool FlushOnEnd { get; set; } = true;

    /// <summary>The counters of the current or last run.</summary>
    public StreamStatistics Statistics { get; private set; } = new();

    /// <summary>
    /// Runs until the source ends or <paramref name="cancellationToken"/> fires (a watched directory
    /// or a broker without an idle timeout only ends that way; cancellation then returns normally
    /// with the statistics so far). A joined query ends when both of its sources have.
    /// </summary>
    public async Task<StreamStatistics> RunAsync(string? checkpointDirectory = null, CancellationToken cancellationToken = default)
    {
        var stats = Statistics = new StreamStatistics();
        var clock = Stopwatch.StartNew();
        var source = new SourceState();
        var windows = _window is null ? null : new WindowState(_window);
        var join = _join is null ? null : new JoinState(_join);
        var path = checkpointDirectory is null ? null : Path.Combine(Path.GetFullPath(checkpointDirectory), "checkpoint.json");
        if (path is not null && File.Exists(path)) Checkpoint.Load(path, source, windows, join);
        stats.LateRows = (windows?.LateRows ?? 0) + (join?.LateRows ?? 0);

        var finished = false;
        try
        {
            await foreach (var (side, batch) in ReadAsync(source, join, cancellationToken).ConfigureAwait(false))
            {
                var stream = side == 1 ? _join!.Right : _stream;
                var output = await stream.ApplyAsync(batch.Frame, cancellationToken).ConfigureAwait(false);
                if (join is not null) output = join.Add(side == 0, output) ?? DataFrame.Empty();
                if (windows is not null) output = windows.Add(output) ?? DataFrame.Empty();
                stats.LateRows = (windows?.LateRows ?? 0) + (join?.LateRows ?? 0);
                await EmitAsync(output, new StreamBatch(source.Batches, batch.Frame.RowCount, batch.Files, windows?.Watermark ?? join?.Watermark), stats).ConfigureAwait(false);
                (join is null ? source : side == 0 ? join.Left.Source : join.Right.Source).Consumed(batch);
                source.Batches++;
                stats.Batches++;
                stats.InputRows += batch.Frame.RowCount;
                if (path is not null) Checkpoint.Save(path, source, windows, join);
                batch.Commit?.Invoke();                            // only now is the batch safe
            }
            finished = !cancellationToken.IsCancellationRequested;
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { }

        if (finished && FlushOnEnd && (windows is not null || join is not null))
        {
            var rest = join?.Flush();
            if (windows is not null)
            {
                var closed = rest is null ? null : windows.Add(rest);
                var open = windows.Flush();
                rest = closed is null ? open : open is null ? closed : DataFrame.Concat([closed, open]);
            }
            if (rest is { RowCount: > 0 })
            {
                await EmitAsync(rest, new StreamBatch(source.Batches, 0, [], windows?.Watermark ?? join?.Watermark), stats).ConfigureAwait(false);
                source.Batches++;
                if (path is not null) Checkpoint.Save(path, source, windows, join);
            }
        }
        stats.JoinStateRows = join is null ? 0 : join.Left.Count + join.Right.Count;
        stats.Elapsed = clock.Elapsed;
        return stats;
    }

    /// <summary>
    /// The batches to process, tagged 0 (left / only) or 1 (right). A joined query reads both
    /// sources concurrently, each at most one batch ahead, in whatever order they arrive.
    /// </summary>
    private async IAsyncEnumerable<(int Side, SourceBatch Batch)> ReadAsync(SourceState source, JoinState? join,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        if (join is null)
        {
            await foreach (var b in _stream.Source.ReadAsync(source, cancellationToken).ConfigureAwait(false)) yield return (0, b);
            yield break;
        }

        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var channel = Channel.CreateBounded<(int, SourceBatch)>(new BoundedChannelOptions(2) { SingleReader = true });
        async Task Pump(int side, StreamSource s, SourceState state)
        {
            await foreach (var b in s.ReadAsync(state, stop.Token).ConfigureAwait(false))
                await channel.Writer.WriteAsync((side, b), stop.Token).ConfigureAwait(false);
        }
        var pumps = new[] { Pump(0, _stream.Source, join.Left.Source), Pump(1, _join!.Right.Source, join.Right.Source) };
        _ = Task.WhenAll(pumps).ContinueWith(t => channel.Writer.TryComplete(t.Exception?.InnerException), TaskScheduler.Default);
        try
        {
            await foreach (var item in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false)) yield return item;
        }
        finally
        {
            stop.Cancel();
            try { await Task.WhenAll(pumps).ConfigureAwait(false); }
            catch (OperationCanceledException) { }
            catch (ChannelClosedException) { }
        }
    }

    private async Task EmitAsync(DataFrame output, StreamBatch info, StreamStatistics stats)
    {
        if (output.RowCount == 0) return;
        if (_window is not null) stats.WindowsEmitted += output.RowCount;
        stats.OutputRows += output.RowCount;
        if (_sink is null) _results.Add(output);
        else await _sink(output, info).ConfigureAwait(false);
    }

    internal static Func<DataFrame, StreamBatch, Task> CsvSink(string directory)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        var full = Path.GetFullPath(directory);
        return (frame, batch) =>
        {
            Directory.CreateDirectory(full);
            var target = Path.Combine(full, $"batch-{batch.Id:D6}.csv");
            var temp = target + ".tmp";
            frame.WriteCsv(temp);
            File.Move(temp, target, overwrite: true);
            return Task.CompletedTask;
        };
    }
}

/// <summary>The checkpoint file: JSON, written to a temporary name and renamed into place.</summary>
internal static class Checkpoint
{
    private static readonly JsonSerializerOptions Options = new()
    {
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
        WriteIndented = true,
    };

    private sealed class Document
    {
        public int Version { get; set; } = 1;
        public long Batches { get; set; }
        public List<string> Files { get; set; } = [];
        public List<OffsetDocument> Offsets { get; set; } = [];
        public long? MaxEvent { get; set; }
        public long LateRows { get; set; }
        public string[]? KeyKinds { get; set; }
        public string? Window { get; set; }
        public List<WindowDocument> Windows { get; set; } = [];
        public JoinDocument? Join { get; set; }
    }

    private sealed class OffsetDocument
    {
        public string Topic { get; set; } = "";
        public int Partition { get; set; }
        public long Next { get; set; }
    }

    private sealed class WindowDocument
    {
        public long Start { get; set; }
        public string?[] Keys { get; set; } = [];
        public double[] Sum { get; set; } = [];
        public double[] Count { get; set; } = [];
        public double[] Min { get; set; } = [];
        public double[] Max { get; set; } = [];
    }

    private sealed class JoinDocument
    {
        public string Definition { get; set; } = "";
        public long LateRows { get; set; }
        public SideDocument Left { get; set; } = new();
        public SideDocument Right { get; set; } = new();
    }

    private sealed class SideDocument
    {
        public List<string> Files { get; set; } = [];
        public List<OffsetDocument> Offsets { get; set; } = [];
        public long? MaxEvent { get; set; }
        public string[]? Columns { get; set; }
        public string[]? Kinds { get; set; }
        public List<RowDocument> Rows { get; set; } = [];
    }

    private sealed class RowDocument
    {
        public long Time { get; set; }
        public string?[] Values { get; set; } = [];
        public bool Matched { get; set; }
    }

    private static string Describe(WindowSpec s)
        => $"{s.TimeColumn}|{s.Size.Ticks}|{s.Slide.Ticks}|{string.Join(",", s.Keys)}|{string.Join(",", s.OutputNames)}";

    private static List<OffsetDocument> Offsets(SourceState s)
    {
        lock (s) return [.. s.Offsets.OrderBy(o => o.Key.Topic, StringComparer.Ordinal).ThenBy(o => o.Key.Partition)
            .Select(o => new OffsetDocument { Topic = o.Key.Topic, Partition = o.Key.Partition, Next = o.Value })];
    }

    private static List<string> Files(SourceState s)
    {
        lock (s) return [.. s.Files.Order(StringComparer.Ordinal)];
    }

    private static void Restore(SourceState s, List<string> files, List<OffsetDocument> offsets)
    {
        lock (s)
        {
            foreach (var f in files) s.Files.Add(f);
            foreach (var o in offsets) s.Offsets[(o.Topic, o.Partition)] = o.Next;
        }
    }

    private static SideDocument Side(JoinSide side) => new()
    {
        Files = Files(side.Source),
        Offsets = Offsets(side.Source),
        MaxEvent = side.MaxEvent,
        Columns = side.Columns,
        Kinds = side.Kinds,
        Rows = [.. side.Buffer.Values.SelectMany(b => b).Select(r => new RowDocument
        {
            Time = r.Time!.Value, Values = [.. r.Values.Select(WindowState.Format)], Matched = r.Matched,
        })],
    };

    private static void Side(SideDocument doc, JoinSide side, string[] on)
    {
        Restore(side.Source, doc.Files, doc.Offsets);
        side.MaxEvent = doc.MaxEvent;
        side.Columns = doc.Columns;
        side.Kinds = doc.Kinds;
        if (doc.Columns is null || doc.Kinds is null) return;
        var keyIndex = on.Select(k => Array.IndexOf(doc.Columns, k)).ToArray();
        foreach (var r in doc.Rows)
        {
            var values = r.Values.Select((v, i) => WindowState.Parse(v, doc.Kinds[i])).ToArray();
            var key = string.Join("\u001f", keyIndex.Select(k => WindowState.Format(values[k])));
            side.Add(new JoinRow(r.Time, key, values) { Matched = r.Matched });
        }
    }

    public static void Save(string path, SourceState source, WindowState? windows, JoinState? join = null)
    {
        var doc = new Document { Batches = source.Batches, Files = Files(source), Offsets = Offsets(source) };
        if (windows is not null)
        {
            doc.MaxEvent = windows.MaxEvent;
            doc.LateRows = windows.LateRows;
            doc.KeyKinds = windows.KeyKinds;
            doc.Window = Describe(windows.Spec);
            doc.Windows = [.. windows.Open.Values.OrderBy(w => w.Start).Select(w => new WindowDocument
            {
                Start = w.Start, Keys = [.. w.Keys.Select(WindowState.Format)], Sum = w.Sum, Count = w.Count, Min = w.Min, Max = w.Max,
            })];
        }
        if (join is not null)
            doc.Join = new JoinDocument { Definition = join.Spec.Describe(), LateRows = join.LateRows, Left = Side(join.Left), Right = Side(join.Right) };
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var temp = path + ".tmp";
        File.WriteAllText(temp, JsonSerializer.Serialize(doc, Options));
        File.Move(temp, path, overwrite: true);
    }

    public static void Load(string path, SourceState source, WindowState? windows, JoinState? join = null)
    {
        var doc = JsonSerializer.Deserialize<Document>(File.ReadAllText(path), Options)
            ?? throw new InvalidDataException($"{path} is not a stream checkpoint.");
        if (doc.Version != 1) throw new InvalidDataException($"Checkpoint version {doc.Version} is not supported.");
        source.Batches = doc.Batches;
        Restore(source, doc.Files, doc.Offsets);
        if (join is not null)
        {
            if (doc.Join is null || doc.Join.Definition != join.Spec.Describe())
                throw new InvalidOperationException($"The checkpoint at {path} belongs to a different join ({doc.Join?.Definition ?? "none"}); use a new checkpoint directory.");
            join.LateRows = doc.Join.LateRows;
            Side(doc.Join.Left, join.Left, join.Spec.On);
            Side(doc.Join.Right, join.Right, join.Spec.On);
        }
        else if (doc.Join is not null)
            throw new InvalidOperationException($"The checkpoint at {path} belongs to a joined query; use a new checkpoint directory.");
        if (windows is null) return;
        if (doc.Window is not null && doc.Window != Describe(windows.Spec))
            throw new InvalidOperationException($"The checkpoint at {path} belongs to a different window ({doc.Window}); use a new checkpoint directory.");
        windows.MaxEvent = doc.MaxEvent;
        windows.LateRows = doc.LateRows;
        windows.KeyKinds = doc.KeyKinds;
        foreach (var w in doc.Windows)
        {
            var keys = w.Keys.Select((k, i) => WindowState.Parse(k, doc.KeyKinds?[i] ?? "text")).ToArray();
            var acc = new WindowAccumulator(w.Start, keys, w.Sum.Length) { Sum = w.Sum, Count = w.Count, Min = w.Min, Max = w.Max };
            windows.Open[(w.Start, string.Join("\u001f", keys.Select(WindowState.Format)))] = acc;
        }
    }
}
