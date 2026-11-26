using System.Globalization;
using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet.Streaming;

/// <summary>A window definition.</summary>
internal sealed record WindowSpec(string TimeColumn, TimeSpan Size, TimeSpan Slide, string[] Keys,
    (string? Column, string How)[] Aggregations, TimeSpan AllowedLateness)
{
    internal static readonly string[] Supported = ["sum", "count", "mean", "min", "max"];

    public void Validate()
    {
        ArgumentException.ThrowIfNullOrEmpty(TimeColumn);
        if (Size <= TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(Size), "A window needs a positive size.");
        if (Slide <= TimeSpan.Zero || Slide > Size) throw new ArgumentOutOfRangeException(nameof(Slide), "The slide must be positive and no longer than the window.");
        if (AllowedLateness < TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(AllowedLateness));
        if (Aggregations.Length == 0) throw new ArgumentException("Give at least one aggregation.");
        foreach (var (column, how) in Aggregations)
        {
            if (!Supported.Contains(how)) throw new ArgumentException($"Unsupported window aggregation '{how}'. Use one of: {string.Join(", ", Supported)}.");
            if (column is null && how != "count") throw new ArgumentException($"'{how}' needs a column; only count can count rows.");
        }
        var names = OutputNames;
        if (names.Distinct().Count() != names.Length) throw new ArgumentException("Two aggregations would produce the same column.");
    }

    public string[] OutputNames => [.. Aggregations.Select(a => a.Column is null ? "count" : $"{a.Column}_{a.How}")];
}

/// <summary>One open window's running aggregates for one key.</summary>
internal sealed class WindowAccumulator(long start, object?[] keys, int aggregations)
{
    public long Start { get; } = start;
    public object?[] Keys { get; } = keys;
    public double[] Sum { get; init; } = new double[aggregations];
    public double[] Count { get; init; } = new double[aggregations];
    public double[] Min { get; init; } = Enumerable.Repeat(double.PositiveInfinity, aggregations).ToArray();
    public double[] Max { get; init; } = Enumerable.Repeat(double.NegativeInfinity, aggregations).ToArray();
}

/// <summary>
/// Event-time window state: open windows per key, the watermark, and the key columns' types.
/// Window starts are aligned to the Unix epoch, so a 1-hour window starts on the hour.
/// </summary>
internal sealed class WindowState(WindowSpec spec)
{
    public WindowSpec Spec { get; } = spec;
    public Dictionary<(long Start, string Key), WindowAccumulator> Open { get; } = [];
    public long? MaxEvent { get; set; }
    public string[]? KeyKinds { get; set; }
    public long LateRows { get; set; }

    private static readonly long Epoch = DateTime.UnixEpoch.Ticks;

    public DateTime? Watermark => MaxEvent is { } m ? new DateTime(Math.Max(0, m - Spec.AllowedLateness.Ticks)) : null;

    /// <summary>Adds a batch's rows, then removes and returns the windows the new watermark closes.</summary>
    public DataFrame? Add(DataFrame batch)
    {
        if (batch.RowCount == 0) return CloseUpTo(Watermark);
        if (batch[Spec.TimeColumn] is not DateTimeSeries times)
            throw new InvalidOperationException($"The window's time column '{Spec.TimeColumn}' must hold dates and times (it is {batch[Spec.TimeColumn].GetType().Name}).");
        KeyKinds ??= [.. Spec.Keys.Select(k => Kind(batch[k]))];
        var keySeries = Spec.Keys.Select(k => batch[k]).ToArray();
        var values = Spec.Aggregations.Select(a => a.Column is null ? null : batch.Numeric(a.Column).Values.ToArray()).ToArray();
        var size = Spec.Size.Ticks;
        var slide = Spec.Slide.Ticks;
        var before = Watermark?.Ticks;                   // rows are late against the watermark they arrived under
        long? max = MaxEvent;

        for (var i = 0; i < batch.RowCount; i++)
        {
            if (times[i] is not { } time) continue;      // no event time, no window
            var t = time.Ticks - Epoch;
            max = max is { } m ? Math.Max(m, time.Ticks) : time.Ticks;
            var keys = keySeries.Select(s => s.IsMissing(i) ? null : s.GetValue(i)).ToArray();
            var keyId = string.Join("\u001f", keys.Select(Format));
            var any = false;
            for (var k = FloorDiv(t - size, slide) + 1; k <= FloorDiv(t, slide); k++)
            {
                var start = k * slide + Epoch;
                if (before is { } w && start + size <= w) continue;          // that window already closed
                any = true;
                if (!Open.TryGetValue((start, keyId), out var acc)) Open[(start, keyId)] = acc = new WindowAccumulator(start, keys, values.Length);
                for (var a = 0; a < values.Length; a++)
                {
                    if (values[a] is not { } column) { acc.Count[a]++; continue; }
                    var v = column[i];
                    if (double.IsNaN(v)) continue;
                    acc.Sum[a] += v;
                    acc.Count[a]++;
                    acc.Min[a] = Math.Min(acc.Min[a], v);
                    acc.Max[a] = Math.Max(acc.Max[a], v);
                }
            }
            if (!any) LateRows++;
        }
        MaxEvent = max;
        return CloseUpTo(Watermark);
    }

    /// <summary>Every open window, closed now (the end of a finite stream).</summary>
    public DataFrame? Flush() => CloseUpTo(DateTime.MaxValue);

    private DataFrame? CloseUpTo(DateTime? watermark)
    {
        if (watermark is not { } w) return null;
        var closed = Open.Values.Where(a => a.Start + Spec.Size.Ticks <= w.Ticks)
            .OrderBy(a => a.Start).ThenBy(a => string.Join("\u001f", a.Keys.Select(Format)), StringComparer.Ordinal).ToArray();
        if (closed.Length == 0) return null;
        foreach (var a in closed) Open.Remove((a.Start, string.Join("\u001f", a.Keys.Select(Format))));
        return ToFrame(closed);
    }

    private DataFrame ToFrame(WindowAccumulator[] windows)
    {
        var columns = new List<Series>
        {
            new DateTimeSeries("window_start", windows.Select(w => new DateTime(w.Start)).ToArray()),
            new DateTimeSeries("window_end", windows.Select(w => new DateTime(w.Start + Spec.Size.Ticks)).ToArray()),
        };
        for (var k = 0; k < Spec.Keys.Length; k++)
        {
            var kind = KeyKinds?[k] ?? "text";
            var name = Spec.Keys[k];
            var keys = windows.Select(w => w.Keys[k]).ToArray();
            columns.Add(kind switch
            {
                "number" => new NumericSeries(name, keys.Select(v => v is null ? double.NaN : Convert.ToDouble(v, CultureInfo.InvariantCulture)).ToArray()),
                "bool" => new BooleanSeries(name, keys.Select(v => (bool?)v).ToArray()),
                "date" => new DateTimeSeries(name, keys.Select(v => (DateTime?)v).ToArray()),
                _ => new TextSeries(name, keys.Select(v => v?.ToString()).ToArray()),
            });
        }
        var names = Spec.OutputNames;
        for (var a = 0; a < names.Length; a++)
        {
            var how = Spec.Aggregations[a].How;
            columns.Add(new NumericSeries(names[a], windows.Select(w => how switch
            {
                "sum" => w.Sum[a],
                "count" => w.Count[a],
                "mean" => w.Count[a] > 0 ? w.Sum[a] / w.Count[a] : double.NaN,
                "min" => w.Count[a] > 0 ? w.Min[a] : double.NaN,
                _ => w.Count[a] > 0 ? w.Max[a] : double.NaN,
            }).ToArray()));
        }
        return new DataFrame(columns);
    }

    private static long FloorDiv(long a, long b) => a >= 0 ? a / b : -((-a + b - 1) / b);

    private static string Kind(Series s) => s switch
    {
        NumericSeries => "number",
        BooleanSeries => "bool",
        DateTimeSeries => "date",
        _ => "text",
    };

    /// <summary>A key value as invariant text, the inverse of <see cref="Parse"/>.</summary>
    internal static string? Format(object? v) => v switch
    {
        null => null,
        double d => d.ToString("R", CultureInfo.InvariantCulture),
        DateTime t => t.Ticks.ToString(CultureInfo.InvariantCulture),
        bool b => b ? "true" : "false",
        _ => v.ToString(),
    };

    internal static object? Parse(string? s, string kind) => s is null ? null : kind switch
    {
        "number" => double.Parse(s, CultureInfo.InvariantCulture),
        "date" => new DateTime(long.Parse(s, CultureInfo.InvariantCulture)),
        "bool" => s == "true",
        _ => s,
    };
}
