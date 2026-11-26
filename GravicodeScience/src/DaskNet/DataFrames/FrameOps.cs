using System.Globalization;
using System.Text;
using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet;

/// <summary>A row-wise operation on one partition. A chain of them runs in one task.</summary>
internal abstract record FrameOp
{
    public abstract DataFrame Apply(DataFrame frame);
    public abstract void Write(BinaryWriter w);
    public abstract string Describe();

    public static byte[] Encode(IReadOnlyList<FrameOp> ops)
    {
        using var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
        {
            w.Write(ops.Count);
            foreach (var op in ops) op.Write(w);
        }
        return ms.ToArray();
    }

    public static FrameOp[] Decode(byte[] bytes)
    {
        if (bytes.Length == 0) return [];
        using var r = new BinaryReader(new MemoryStream(bytes), Encoding.UTF8);
        var ops = new FrameOp[r.ReadInt32()];
        for (var i = 0; i < ops.Length; i++)
        {
            ops[i] = r.ReadByte() switch
            {
                1 => new FilterOp(Expr.Read(r)),
                2 => new SelectOp(ReadExprs(r)),
                3 => new WithColumnsOp(ReadExprs(r)),
                4 => new DropOp(ReadStrings(r)),
                5 => new RenameOp(ReadStrings(r).Zip(ReadStrings(r)).ToArray()),
                var tag => throw new InvalidDataException($"Unknown frame operation {tag}."),
            };
        }
        return ops;
    }

    public static DataFrame ApplyAll(DataFrame frame, IEnumerable<FrameOp> ops)
    {
        foreach (var op in ops) frame = op.Apply(frame);
        return frame;
    }

    protected static void WriteExprs(BinaryWriter w, IReadOnlyList<Expr> exprs)
    {
        w.Write(exprs.Count);
        foreach (var e in exprs) e.Write(w);
    }

    private static Expr[] ReadExprs(BinaryReader r)
    {
        var a = new Expr[r.ReadInt32()];
        for (var i = 0; i < a.Length; i++) a[i] = Expr.Read(r);
        return a;
    }

    protected static void WriteStrings(BinaryWriter w, IReadOnlyList<string> s)
    {
        w.Write(s.Count);
        foreach (var x in s) w.Write(x);
    }

    private static string[] ReadStrings(BinaryReader r)
    {
        var a = new string[r.ReadInt32()];
        for (var i = 0; i < a.Length; i++) a[i] = r.ReadString();
        return a;
    }
}

internal sealed record FilterOp(Expr Predicate) : FrameOp
{
    public override DataFrame Apply(DataFrame frame) => frame.Filter(Predicate.EvaluateMask(frame));
    public override void Write(BinaryWriter w) { w.Write((byte)1); Predicate.Write(w); }
    public override string Describe() => $"Filter {Predicate}";
}

internal sealed record SelectOp(Expr[] Items) : FrameOp
{
    public override DataFrame Apply(DataFrame frame)
    {
        var columns = Items.Select(e => e is ColumnExpr c ? frame[c.Name] : e.EvaluateSeries(frame)).ToList();
        var duplicate = columns.GroupBy(c => c.Name).FirstOrDefault(g => g.Count() > 1);
        if (duplicate is not null) throw new InvalidOperationException($"Column '{duplicate.Key}' is selected twice; alias one of them.");
        return new DataFrame(columns);
    }
    public override void Write(BinaryWriter w) { w.Write((byte)2); WriteExprs(w, Items); }
    public override string Describe() => $"Select {string.Join(", ", Items.Select(i => i is AliasExpr || i is ColumnExpr ? i.ToString() : $"{i} AS {i.OutputName}"))}";
}

internal sealed record WithColumnsOp(Expr[] Items) : FrameOp
{
    public override DataFrame Apply(DataFrame frame)
    {
        // Evaluated against the input frame, so columns added together do not see each other.
        var added = Items.Select(e => e.EvaluateSeries(frame)).ToArray();
        foreach (var s in added) frame = frame.WithColumn(s);
        return frame;
    }
    public override void Write(BinaryWriter w) { w.Write((byte)3); WriteExprs(w, Items); }
    public override string Describe() => $"WithColumns {string.Join(", ", Items.Select(i => $"{i.OutputName} = {(i is AliasExpr a ? a.Inner : i)}"))}";
}

internal sealed record DropOp(string[] Names) : FrameOp
{
    public override DataFrame Apply(DataFrame frame) => frame.Drop(Names);
    public override void Write(BinaryWriter w) { w.Write((byte)4); WriteStrings(w, Names); }
    public override string Describe() => $"Drop {string.Join(", ", Names)}";
}

internal sealed record RenameOp((string From, string To)[] Pairs) : FrameOp
{
    public override DataFrame Apply(DataFrame frame)
    {
        var map = Pairs.ToDictionary(p => p.From, p => p.To, StringComparer.Ordinal);
        return new DataFrame(frame.Columns.Select(c => map.TryGetValue(c.Name, out var to) ? c.Rename(to) : c));
    }
    public override void Write(BinaryWriter w)
    {
        w.Write((byte)5);
        WriteStrings(w, Pairs.Select(p => p.From).ToArray());
        WriteStrings(w, Pairs.Select(p => p.To).ToArray());
    }
    public override string Describe() => $"Rename {string.Join(", ", Pairs.Select(p => $"{p.From} -> {p.To}"))}";
}

/// <summary>One aggregation: <c>Output = How(Column)</c>.</summary>
/// <param name="Output">The result column's name.</param>
/// <param name="How">One of <c>sum</c>, <c>mean</c>, <c>count</c> (non-missing), <c>size</c> (rows),
/// <c>min</c>, <c>max</c>, <c>var</c>, <c>std</c> (sample, ddof 1), <c>var_pop</c>, <c>std_pop</c>.</param>
/// <param name="Column">The input column; <c>null</c> only for <c>size</c>.</param>
public sealed record Aggregation(string Output, string How, string? Column)
{
    internal static readonly string[] Supported = ["sum", "mean", "count", "size", "min", "max", "var", "std", "var_pop", "std_pop"];

    /// <summary>Checks the operation name.</summary>
    public Aggregation Validate()
    {
        if (!Supported.Contains(How)) throw new ArgumentException($"Unknown aggregation '{How}'; expected one of {string.Join(", ", Supported)}.");
        if (Column is null && How != "size") throw new ArgumentException($"Aggregation '{How}' needs a column.");
        return this;
    }

    internal static byte[] Encode(IReadOnlyList<string> keys, IReadOnlyList<Aggregation> aggs)
    {
        using var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
        {
            w.Write(keys.Count);
            foreach (var k in keys) w.Write(k);
            w.Write(aggs.Count);
            foreach (var a in aggs)
            {
                w.Write(a.Output);
                w.Write(a.How);
                w.Write(a.Column ?? "");
            }
        }
        return ms.ToArray();
    }

    internal static (string[] Keys, Aggregation[] Aggs) Decode(byte[] bytes)
    {
        using var r = new BinaryReader(new MemoryStream(bytes), Encoding.UTF8);
        var keys = new string[r.ReadInt32()];
        for (var i = 0; i < keys.Length; i++) keys[i] = r.ReadString();
        var aggs = new Aggregation[r.ReadInt32()];
        for (var i = 0; i < aggs.Length; i++)
        {
            var output = r.ReadString();
            var how = r.ReadString();
            var column = r.ReadString();
            aggs[i] = new Aggregation(output, how, column.Length == 0 ? null : column);
        }
        return (keys, aggs);
    }
}

/// <summary>Grouping helpers shared by the aggregation, shuffle and join kernels.</summary>
internal static class Grouping
{
    /// <summary>The key signature of one row; the same rule GraviFrame's joins use, so a hash
    /// shuffle sends rows to the bucket where their join partners are.</summary>
    public static string Signature(Series[] keys, int row)
    {
        if (keys.Length == 1) return keys[0].GetValue(row)?.ToString() ?? "\u0000";
        var sb = new StringBuilder();
        for (var k = 0; k < keys.Length; k++)
        {
            if (k > 0) sb.Append('\u001F');
            sb.Append(keys[k].GetValue(row)?.ToString() ?? "\u0000");
        }
        return sb.ToString();
    }

    /// <summary>
    /// A hash that is the same in every process: FNV-1a over UTF-8. <c>string.GetHashCode</c> is
    /// randomised per process, which would send one key to different buckets on different
    /// workers and silently drop join matches.
    /// </summary>
    public static ulong StableHash(string s)
    {
        var hash = 14695981039346656037UL;
        foreach (var b in Encoding.UTF8.GetBytes(s))
        {
            hash ^= b;
            hash *= 1099511628211UL;
        }
        return hash;
    }

    /// <summary>Groups rows by key, in first-seen order: (first row of each group, rows per group).</summary>
    public static (List<int> First, List<List<int>> Rows) Group(DataFrame frame, IReadOnlyList<string> keys)
    {
        var first = new List<int>();
        var rows = new List<List<int>>();
        if (keys.Count == 0)
        {
            first.Add(0);
            rows.Add(Enumerable.Range(0, frame.RowCount).ToList());
            return (first, rows);
        }

        var series = keys.Select(k => frame[k]).ToArray();
        var lookup = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < frame.RowCount; i++)
        {
            var sig = Signature(series, i);
            if (!lookup.TryGetValue(sig, out var g))
            {
                g = rows.Count;
                lookup[sig] = g;
                first.Add(i);
                rows.Add([]);
            }
            rows[g].Add(i);
        }
        return (first, rows);
    }

    /// <summary>Compares two values of one column for sorting: missing last, numbers numerically,
    /// text ordinally.</summary>
    public static int CompareValues(object? a, object? b)
    {
        if (a is null) return b is null ? 0 : 1;
        if (b is null) return -1;
        return (a, b) switch
        {
            (double x, double y) => x.CompareTo(y),
            (string x, string y) => string.CompareOrdinal(x, y),
            (DateTime x, DateTime y) => x.CompareTo(y),
            (bool x, bool y) => x.CompareTo(y),
            _ => string.CompareOrdinal(Convert.ToString(a, CultureInfo.InvariantCulture), Convert.ToString(b, CultureInfo.InvariantCulture)),
        };
    }

    /// <summary>A typed empty column of the same kind as <paramref name="like"/>.</summary>
    public static Series EmptyLike(Series like) => like switch
    {
        NumericSeries => new NumericSeries(like.Name, []),
        BooleanSeries => new BooleanSeries(like.Name, Array.Empty<bool?>()),
        DateTimeSeries => new DateTimeSeries(like.Name, Array.Empty<DateTime?>()),
        _ => new TextSeries(like.Name, []),
    };
}
