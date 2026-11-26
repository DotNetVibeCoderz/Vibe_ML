using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
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

    /// <summary>Rows dropped because every window they belong to had already closed, over the query's life.</summary>
    public long LateRows { get; internal set; }

    /// <summary>Windows emitted by this run.</summary>
    public long WindowsEmitted { get; internal set; }

    /// <summary>Wall-clock time of the run.</summary>
    public TimeSpan Elapsed { get; internal set; }

    /// <inheritdoc/>
    public override string ToString()
        => $"{Batches} batch(es), {InputRows} row(s) in, {OutputRows} out, {LateRows} late, {Elapsed.TotalMilliseconds:F0} ms";
}

/// <summary>
/// A stream with a sink, ready to run. With a checkpoint directory the query is restartable: after
/// each batch's sink returns, the consumed files, the batch count and the open windows are saved;
/// a new run of the same query resumes from there.
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
    private readonly Func<DataFrame, StreamBatch, Task>? _sink;
    private readonly List<DataFrame> _results = [];

    internal StreamQuery(DaskStream stream, WindowSpec? window, Func<DataFrame, StreamBatch, Task>? sink)
    {
        _stream = stream;
        _window = window;
        _sink = sink;
    }

    /// <summary>Output batches kept by a <c>Collect()</c> query.</summary>
    public IReadOnlyList<DataFrame> Results => _results;

    /// <summary>Every collected batch as one frame.</summary>
    public DataFrame CollectedFrame() => _results.Count == 0 ? DataFrame.Empty() : DataFrame.Concat(_results);

    /// <summary>Whether a windowed query emits its still-open windows when a finite source ends.
    /// On by default; a cancelled run never flushes (its windows stay in the checkpoint).</summary>
    public bool FlushOnEnd { get; set; } = true;

    /// <summary>The counters of the current or last run.</summary>
    public StreamStatistics Statistics { get; private set; } = new();

    /// <summary>
    /// Runs until the source ends or <paramref name="cancellationToken"/> fires (a watched directory
    /// only ends that way; cancellation then returns normally with the statistics so far).
    /// </summary>
    public async Task<StreamStatistics> RunAsync(string? checkpointDirectory = null, CancellationToken cancellationToken = default)
    {
        var stats = Statistics = new StreamStatistics();
        var clock = Stopwatch.StartNew();
        var source = new SourceState();
        var windows = _window is null ? null : new WindowState(_window);
        var path = checkpointDirectory is null ? null : Path.Combine(Path.GetFullPath(checkpointDirectory), "checkpoint.json");
        if (path is not null && File.Exists(path)) Checkpoint.Load(path, source, windows);
        stats.LateRows = windows?.LateRows ?? 0;

        var finished = false;
        try
        {
            await foreach (var batch in _stream.Source.ReadAsync(source, cancellationToken).ConfigureAwait(false))
            {
                var output = await _stream.ApplyAsync(batch.Frame, cancellationToken).ConfigureAwait(false);
                if (windows is not null)
                {
                    output = windows.Add(output) ?? DataFrame.Empty();
                    stats.LateRows = windows.LateRows;
                }
                await EmitAsync(output, new StreamBatch(source.Batches, batch.Frame.RowCount, batch.Files, windows?.Watermark), stats).ConfigureAwait(false);
                foreach (var f in batch.Files) source.Files.Add(f);
                source.Batches++;
                stats.Batches++;
                stats.InputRows += batch.Frame.RowCount;
                if (path is not null) Checkpoint.Save(path, source, windows);
            }
            finished = !cancellationToken.IsCancellationRequested;
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { }

        if (finished && windows is not null && FlushOnEnd && windows.Flush() is { } rest)
        {
            await EmitAsync(rest, new StreamBatch(source.Batches, 0, [], windows.Watermark), stats).ConfigureAwait(false);
            source.Batches++;
            if (path is not null) Checkpoint.Save(path, source, windows);
        }
        stats.Elapsed = clock.Elapsed;
        return stats;
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
        public long? MaxEvent { get; set; }
        public long LateRows { get; set; }
        public string[]? KeyKinds { get; set; }
        public string? Window { get; set; }
        public List<WindowDocument> Windows { get; set; } = [];
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

    private static string Describe(WindowSpec s)
        => $"{s.TimeColumn}|{s.Size.Ticks}|{s.Slide.Ticks}|{string.Join(",", s.Keys)}|{string.Join(",", s.OutputNames)}";

    public static void Save(string path, SourceState source, WindowState? windows)
    {
        var doc = new Document { Batches = source.Batches, Files = [.. source.Files.Order(StringComparer.Ordinal)] };
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
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var temp = path + ".tmp";
        File.WriteAllText(temp, JsonSerializer.Serialize(doc, Options));
        File.Move(temp, path, overwrite: true);
    }

    public static void Load(string path, SourceState source, WindowState? windows)
    {
        var doc = JsonSerializer.Deserialize<Document>(File.ReadAllText(path), Options)
            ?? throw new InvalidDataException($"{path} is not a stream checkpoint.");
        if (doc.Version != 1) throw new InvalidDataException($"Checkpoint version {doc.Version} is not supported.");
        source.Batches = doc.Batches;
        foreach (var f in doc.Files) source.Files.Add(f);
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
