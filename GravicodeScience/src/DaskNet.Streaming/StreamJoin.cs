using System.Globalization;
using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet.Streaming;

/// <summary>Which unmatched rows a stream-stream join keeps.</summary>
public enum StreamJoinType
{
    /// <summary>Matched pairs only.</summary>
    Inner,
    /// <summary>Plus left rows that can no longer match, with the right columns missing.</summary>
    Left,
    /// <summary>Plus right rows that can no longer match.</summary>
    Right,
    /// <summary>Plus unmatched rows of both sides.</summary>
    Outer,
}

/// <summary>A stream-stream join definition: rows match when their keys are equal and
/// <c>rightTime - leftTime</c> lies in [<see cref="Lower"/>, <see cref="Upper"/>].</summary>
internal sealed record JoinSpec(DaskStream Right, string[] On, string LeftTime, string RightTime,
    TimeSpan Lower, TimeSpan Upper, StreamJoinType How, TimeSpan AllowedLateness, string Suffix)
{
    public void Validate()
    {
        ArgumentNullException.ThrowIfNull(Right);
        ArgumentNullException.ThrowIfNull(On);
        if (On.Length == 0) throw new ArgumentException("Join on at least one key column.", nameof(On));
        ArgumentException.ThrowIfNullOrEmpty(LeftTime);
        ArgumentException.ThrowIfNullOrEmpty(RightTime);
        if (Lower > Upper) throw new ArgumentException("The interval's lower bound is above its upper bound.");
        if (AllowedLateness < TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(AllowedLateness));
        ArgumentException.ThrowIfNullOrEmpty(Suffix);
    }

    public string Describe() => $"{string.Join(",", On)}|{LeftTime}|{RightTime}|{Lower.Ticks}|{Upper.Ticks}|{How}|{AllowedLateness.Ticks}|{Suffix}";
}

/// <summary>One buffered row of one side.</summary>
internal sealed class JoinRow(long? time, string? key, object?[] values)
{
    public long? Time { get; } = time;
    public string? Key { get; } = key;
    public object?[] Values { get; } = values;
    public bool Matched { get; set; }
}

/// <summary>One side's buffered rows, watermark and pinned schema.</summary>
internal sealed class JoinSide(string timeColumn)
{
    public string TimeColumn { get; } = timeColumn;
    public string[]? Columns { get; set; }
    public string[]? Kinds { get; set; }
    public Dictionary<string, List<JoinRow>> Buffer { get; } = new(StringComparer.Ordinal);
    public long? MaxEvent { get; set; }
    public SourceState Source { get; } = new();

    public int Count => Buffer.Values.Sum(b => b.Count);

    public void Add(JoinRow row)
    {
        if (!Buffer.TryGetValue(row.Key!, out var list)) Buffer[row.Key!] = list = [];
        list.Add(row);
    }
}

/// <summary>
/// Event-time interval join state (Spark's stream-stream join). Each side's rows are buffered
/// until the other side's watermark proves they can no longer match, then evicted - emitted with
/// missing partner columns first, for an outer side that never matched - so memory is bounded by
/// the interval and the lateness, not by the stream.
/// </summary>
/// <remarks>
/// A side's watermark is its latest event time minus the allowed lateness; a row older than its
/// own side's watermark from before its batch is late and dropped. A left row at <c>l</c> can only
/// meet right rows in [l + lower, l + upper] and every future right row is at or after the right
/// watermark, so it is evicted once <c>l + upper</c> falls below it (symmetrically for the right).
/// Both conditions depend on one side's own progress only, so the matches do not depend on how the
/// two streams' batches interleave.
/// </remarks>
internal sealed class JoinState(JoinSpec spec)
{
    public JoinSpec Spec { get; } = spec;
    public JoinSide Left { get; } = new(spec.LeftTime);
    public JoinSide Right { get; } = new(spec.RightTime);
    public long LateRows { get; set; }
    private readonly List<(JoinRow? Left, JoinRow? Right)> _pending = [];

    private long Lateness => Spec.AllowedLateness.Ticks;
    private static long? Mark(JoinSide s, long lateness) => s.MaxEvent is { } m ? m - lateness : null;

    /// <summary>The query's watermark: the slower side's.</summary>
    public DateTime? Watermark => Mark(Left, Lateness) is { } l && Mark(Right, Lateness) is { } r ? new DateTime(Math.Max(0, Math.Min(l, r))) : null;

    /// <summary>Adds one side's batch and returns the rows it produced (matches and evictions).</summary>
    public DataFrame? Add(bool left, DataFrame batch)
    {
        var side = left ? Left : Right;
        var other = left ? Right : Left;
        if (batch.RowCount > 0)
        {
            Pin(side, batch);
            var before = Mark(side, Lateness);
            var times = batch[side.TimeColumn] as DateTimeSeries
                ?? throw new InvalidOperationException($"The join's time column '{side.TimeColumn}' must hold dates and times (it is {batch[side.TimeColumn].GetType().Name}).");
            var sources = side.Columns!.Select(c => batch.ColumnNames.Contains(c) ? batch[c] : null).ToArray();
            var keyIndex = Spec.On.Select(k => Array.IndexOf(side.Columns!, k)).ToArray();
            long? max = side.MaxEvent;
            for (var i = 0; i < batch.RowCount; i++)
            {
                var values = new object?[sources.Length];
                for (var c = 0; c < sources.Length; c++) values[c] = Value(sources[c], i, side.Kinds![c]);
                var keyParts = keyIndex.Select(k => values[k]).ToArray();
                var key = keyParts.Any(v => v is null) ? null : string.Join("\u001f", keyParts.Select(WindowState.Format));
                var time = times[i]?.Ticks;
                var row = new JoinRow(time, key, values);
                if (time is not { } t || key is null)
                {
                    // Can never match: emit now if this side is preserved, else drop.
                    if (Preserved(left)) _pending.Add(left ? (row, null) : (null, row));
                    continue;
                }
                if (before is { } w && t < w) { LateRows++; continue; }
                max = max is { } m ? Math.Max(m, t) : t;
                if (other.Buffer.TryGetValue(key, out var candidates))
                    foreach (var o in candidates)
                    {
                        var gap = left ? o.Time!.Value - t : t - o.Time!.Value;          // right - left
                        if (gap < Spec.Lower.Ticks || gap > Spec.Upper.Ticks) continue;
                        row.Matched = o.Matched = true;
                        _pending.Add(left ? (row, o) : (o, row));
                    }
                side.Add(row);
            }
            side.MaxEvent = max;
        }
        Evict(flush: false);
        return Drain(flush: false);
    }

    /// <summary>Emits every still-buffered unmatched row of a preserved side (the end of finite streams).</summary>
    public DataFrame? Flush()
    {
        Evict(flush: true);
        return Drain(flush: true);
    }

    private bool Preserved(bool left) => Spec.How == StreamJoinType.Outer || Spec.How == (left ? StreamJoinType.Left : StreamJoinType.Right);

    private void Evict(bool flush)
    {
        var rightMark = Mark(Right, Lateness);
        var leftMark = Mark(Left, Lateness);
        // A left row can still meet a right row at or after rightMark iff l + upper >= rightMark.
        EvictSide(Left, true, r => flush || rightMark is { } w && r.Time!.Value + Spec.Upper.Ticks < w);
        // A right row can still meet a left row at or after leftMark iff r - lower >= leftMark.
        EvictSide(Right, false, r => flush || leftMark is { } w && r.Time!.Value - Spec.Lower.Ticks < w);
    }

    private void EvictSide(JoinSide side, bool left, Func<JoinRow, bool> expired)
    {
        var keep = Preserved(left);
        foreach (var (key, rows) in side.Buffer.ToArray())
        {
            var gone = rows.RemoveAll(r =>
            {
                if (!expired(r)) return false;
                if (keep && !r.Matched) _pending.Add(left ? (r, null) : (null, r));
                return true;
            });
            if (gone > 0 && rows.Count == 0) side.Buffer.Remove(key);
        }
    }

    private static void Pin(JoinSide side, DataFrame batch)
    {
        if (side.Columns is not null) return;
        side.Columns = [.. batch.ColumnNames];
        side.Kinds = [.. side.Columns.Select(c => Kind(batch[c]))];
    }

    internal static string Kind(Series s) => s switch
    {
        NumericSeries => "number",
        BooleanSeries => "bool",
        DateTimeSeries => "date",
        _ => "text",
    };

    private static object? Value(Series? s, int i, string kind)
    {
        if (s is null || s.IsMissing(i)) return null;
        var v = s.GetValue(i);
        return kind switch
        {
            "number" => Convert.ToDouble(v, CultureInfo.InvariantCulture),
            "date" => v is DateTime t ? t : null,
            "bool" => v is bool b ? b : null,
            _ => v?.ToString(),
        };
    }

    /// <summary>The output columns: keys once, then the left's other columns, then the right's,
    /// suffixed where a name is taken.</summary>
    private (string Name, string Kind, Func<JoinRow?, JoinRow?, object?> Value)[] Layout()
    {
        var layout = new List<(string, string, Func<JoinRow?, JoinRow?, object?>)>();
        var taken = new HashSet<string>(StringComparer.Ordinal);
        var lc = Left.Columns ?? [];
        var rc = Right.Columns ?? [];
        foreach (var k in Spec.On)
        {
            var li = Array.IndexOf(lc, k);
            var ri = Array.IndexOf(rc, k);
            var kind = li >= 0 ? Left.Kinds![li] : ri >= 0 ? Right.Kinds![ri] : "text";
            layout.Add((k, kind, (l, r) => l is not null && li >= 0 ? l.Values[li] : r is not null && ri >= 0 ? r.Values[ri] : null));
            taken.Add(k);
        }
        for (var i = 0; i < lc.Length; i++)
        {
            if (Spec.On.Contains(lc[i])) continue;
            var c = i;
            layout.Add((lc[i], Left.Kinds![i], (l, _) => l?.Values[c]));
            taken.Add(lc[i]);
        }
        for (var i = 0; i < rc.Length; i++)
        {
            if (Spec.On.Contains(rc[i])) continue;
            var c = i;
            var name = rc[i];
            while (!taken.Add(name)) name += Spec.Suffix;
            layout.Add((name, Right.Kinds![i], (_, r) => r?.Values[c]));
        }
        return [.. layout];
    }

    private DataFrame? Drain(bool flush)
    {
        if (_pending.Count == 0) return null;
        // Rows that can never match (no time or key) may arrive before the other side has shown its
        // columns; hold them so every batch has the same schema. Only the end of the streams
        // emits them without it.
        if (!flush && (Left.Columns is null || Right.Columns is null)) return null;
        var layout = Layout();
        var columns = new List<Series>(layout.Length);
        foreach (var (name, kind, value) in layout)
        {
            var values = _pending.Select(p => value(p.Left, p.Right)).ToArray();
            columns.Add(kind switch
            {
                "number" => new NumericSeries(name, values.Select(v => v is null ? double.NaN : (double)v).ToArray()),
                "bool" => new BooleanSeries(name, values.Select(v => (bool?)v).ToArray()),
                "date" => new DateTimeSeries(name, values.Select(v => (DateTime?)v).ToArray()),
                _ => new TextSeries(name, values.Select(v => (string?)v).ToArray()),
            });
        }
        _pending.Clear();
        return new DataFrame(columns);
    }
}

/// <summary>Two streams joined on keys and event time; finish it with a sink, or window it.</summary>
public sealed class JoinedStream
{
    internal JoinedStream(DaskStream left, JoinSpec spec)
    {
        spec.Validate();
        Left = left;
        Spec = spec;
    }

    internal DaskStream Left { get; }
    internal JoinSpec Spec { get; }

    /// <summary>Calls <paramref name="sink"/> with each batch of joined rows.</summary>
    public StreamQuery ForEachBatch(Func<DataFrame, StreamBatch, Task> sink) => new(Left, null, sink, Spec);

    /// <summary>Keeps every batch of joined rows in <see cref="StreamQuery.Results"/>.</summary>
    public StreamQuery Collect() => new(Left, null, null, Spec);

    /// <summary>Writes each batch of joined rows to <c>batch-{id}.csv</c>.</summary>
    public StreamQuery ToCsvDirectory(string directory) => new(Left, null, StreamQuery.CsvSink(directory), Spec);

    /// <summary>Windowed aggregation over the joined rows (see <see cref="DaskStream.Window"/>).</summary>
    public WindowedStream Window(string timeColumn, TimeSpan size, string[] keys, (string? Column, string How)[] aggregations,
        TimeSpan? slide = null, TimeSpan allowedLateness = default)
        => new(Left, new WindowSpec(timeColumn, size, slide ?? size, keys, aggregations, allowedLateness), Spec);
}
