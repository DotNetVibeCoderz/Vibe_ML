using System.Text;
using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet;

/// <summary>
/// One window function: <c>Output = How(Column) OVER (PARTITION BY ... ORDER BY ...)</c>.
/// </summary>
/// <param name="Output">The result column's name.</param>
/// <param name="How">
/// <c>row_number</c>, <c>rank</c>, <c>dense_rank</c> (by the window's order); <c>lag</c>/<c>lead</c>
/// (the value <paramref name="Offset"/> rows before/after, missing past the edge); or an aggregate -
/// <c>sum</c>, <c>mean</c>, <c>count</c> (non-missing), <c>size</c> (rows), <c>min</c>, <c>max</c>. An
/// aggregate covers the whole partition when the window has no order, and otherwise runs from the
/// partition's start through the current row <em>and its ties</em> (SQL's default
/// <c>RANGE UNBOUNDED PRECEDING</c> frame), so tied rows get the same running total.
/// </param>
/// <param name="Column">The input column; <c>null</c> for the ranking functions and <c>size</c>.</param>
/// <param name="Offset">How far <c>lag</c>/<c>lead</c> look.</param>
public sealed record WindowFunction(string Output, string How, string? Column = null, int Offset = 1)
{
    internal static readonly string[] Ranking = ["row_number", "rank", "dense_rank"];
    internal static readonly string[] Shifts = ["lag", "lead"];
    internal static readonly string[] Aggregates = ["sum", "mean", "count", "size", "min", "max"];

    /// <summary>Checks the function name and its arguments.</summary>
    public WindowFunction Validate()
    {
        if (!Ranking.Contains(How) && !Shifts.Contains(How) && !Aggregates.Contains(How))
            throw new ArgumentException($"Unknown window function '{How}'; expected one of {string.Join(", ", Ranking.Concat(Shifts).Concat(Aggregates))}.");
        var needsColumn = Shifts.Contains(How) || (Aggregates.Contains(How) && How != "size");
        if (needsColumn && Column is null) throw new ArgumentException($"Window function '{How}' needs a column.");
        if (!needsColumn && Column is not null) throw new ArgumentException($"Window function '{How}' takes no column.");
        if (Offset < 0) throw new ArgumentException("A lag/lead offset cannot be negative.");
        return this;
    }
}

/// <summary>A window: how rows are partitioned and ordered, and the functions computed over it.</summary>
internal sealed record WindowSpec(string[] PartitionBy, (string Column, bool Ascending)[] OrderBy, WindowFunction[] Functions)
{
    public byte[] Encode()
    {
        using var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
        {
            w.Write(PartitionBy.Length);
            foreach (var k in PartitionBy) w.Write(k);
            w.Write(OrderBy.Length);
            foreach (var (c, asc) in OrderBy) { w.Write(c); w.Write(asc); }
            w.Write(Functions.Length);
            foreach (var f in Functions)
            {
                w.Write(f.Output);
                w.Write(f.How);
                w.Write(f.Column ?? "");
                w.Write(f.Offset);
            }
        }
        return ms.ToArray();
    }

    public static WindowSpec Decode(byte[] bytes)
    {
        using var r = new BinaryReader(new MemoryStream(bytes), Encoding.UTF8);
        var keys = new string[r.ReadInt32()];
        for (var i = 0; i < keys.Length; i++) keys[i] = r.ReadString();
        var order = new (string, bool)[r.ReadInt32()];
        for (var i = 0; i < order.Length; i++) order[i] = (r.ReadString(), r.ReadBoolean());
        var functions = new WindowFunction[r.ReadInt32()];
        for (var i = 0; i < functions.Length; i++)
        {
            var output = r.ReadString();
            var how = r.ReadString();
            var column = r.ReadString();
            functions[i] = new WindowFunction(output, how, column.Length == 0 ? null : column, r.ReadInt32());
        }
        return new WindowSpec(keys, order, functions);
    }

    public IEnumerable<string> InputColumns
        => PartitionBy.Concat(OrderBy.Select(o => o.Column)).Concat(Functions.Where(f => f.Column is not null).Select(f => f.Column!));

    public string Describe()
        => $"Window{(PartitionBy.Length > 0 ? $" partition by [{string.Join(", ", PartitionBy)}]" : "")}"
           + (OrderBy.Length > 0 ? $" order by [{string.Join(", ", OrderBy.Select(o => $"{o.Column} {(o.Ascending ? "ASC" : "DESC")}"))}]" : "")
           + $": {string.Join(", ", Functions.Select(f => $"{f.Output} = {f.How}({f.Column ?? ""}{(Shifts(f) ? $", {f.Offset}" : "")})"))}";

    private static bool Shifts(WindowFunction f) => WindowFunction.Shifts.Contains(f.How);
}

/// <summary>
/// Adds window-function columns. Rows are hash-shuffled by the partition keys so every window
/// partition lands whole in one task (with no keys, everything goes to one task), then each task
/// sorts its groups and computes the functions. Row order within a task is kept.
/// </summary>
internal sealed class WindowNode(PlanNode input, WindowSpec spec) : PlanNode
{
    public PlanNode Input { get; } = input;
    public WindowSpec Spec { get; } = spec;
    public override IReadOnlyList<PlanNode> Children => [Input];
    public override IReadOnlyList<string>? Schema => Input.Schema?.Concat(Spec.Functions.Select(f => f.Output)).ToArray();
    public override int Partitions => Spec.PartitionBy.Length == 0 ? 1 : Input.Partitions;
    public override string Describe()
        => Spec.Describe() + (Spec.PartitionBy.Length == 0 || Input.Partitions == 1 ? " (one partition)" : $" (hash shuffle, {Partitions} partitions)");
    public override PlanNode WithChildren(IReadOnlyList<PlanNode> c) => new WindowNode(c[0], Spec);
}

public static partial class FrameKernels
{
    /// <summary>Computes window functions over one partition that holds whole window partitions.</summary>
    [RemoteTask]
    public static DataFrame WindowFrame(DataFrame frame, byte[] spec)
    {
        var s = WindowSpec.Decode(spec);
        var n = frame.RowCount;
        var (_, groups) = Grouping.Group(frame, s.PartitionBy);
        if (n == 0) groups.Clear();
        var orderColumns = s.OrderBy.Select(o => frame[o.Column]).ToArray();

        int Compare(int a, int b)
        {
            for (var k = 0; k < orderColumns.Length; k++)
            {
                var c = Grouping.CompareValues(orderColumns[k].GetValue(a), orderColumns[k].GetValue(b));
                if (c != 0) return s.OrderBy[k].Ascending ? c : -c;
            }
            return 0;
        }

        var results = new List<Series>();
        var sorted = groups.Select(g => g.OrderBy(r => r, Comparer<int>.Create(Compare)).ToArray()).ToArray();   // stable
        // Peer runs: consecutive sorted rows whose order values tie. Without ORDER BY, the whole
        // partition is one run, which is what makes an unordered aggregate cover all of it.
        var runs = sorted.Select(rows =>
        {
            var starts = new List<int>();
            for (var i = 0; i < rows.Length; i++)
                if (i == 0 || Compare(rows[i - 1], rows[i]) != 0) starts.Add(i);
            return starts.ToArray();
        }).ToArray();

        foreach (var f in s.Functions)
        {
            if (WindowFunction.Shifts.Contains(f.How))
            {
                var take = new int[n];
                foreach (var rows in sorted)
                    for (var i = 0; i < rows.Length; i++)
                    {
                        var j = f.How == "lag" ? i - f.Offset : i + f.Offset;
                        take[rows[i]] = j >= 0 && j < rows.Length ? rows[j] : -1;
                    }
                results.Add(SeriesOperations.TakeAllowingMissing(frame[f.Column!], take).Rename(f.Output));
                continue;
            }

            var values = new double[n];
            if (WindowFunction.Ranking.Contains(f.How))
            {
                for (var g = 0; g < sorted.Length; g++)
                {
                    var rows = sorted[g];
                    var run = 0;
                    for (var i = 0; i < rows.Length; i++)
                    {
                        if (run + 1 < runs[g].Length && runs[g][run + 1] == i) run++;
                        values[rows[i]] = f.How switch
                        {
                            "row_number" => i + 1,
                            "rank" => runs[g][run] + 1,
                            _ => run + 1,
                        };
                    }
                }
                results.Add(new NumericSeries(f.Output, values));
                continue;
            }

            var source = f.Column is null ? null : frame[f.Column];
            Value? numbers = null;
            if (f.How is not ("size" or "count"))
            {
                numbers = Value.FromSeries(source!);
                if (numbers.Kind is not (DataType.Numeric or DataType.Boolean))
                    throw new InvalidOperationException($"Window function '{f.How}' needs a numeric column; '{f.Column}' is {numbers.Kind}.");
            }
            for (var g = 0; g < sorted.Length; g++)
            {
                var rows = sorted[g];
                double count = 0, sum = 0, compensation = 0, min = double.PositiveInfinity, max = double.NegativeInfinity;
                for (var run = 0; run < runs[g].Length; run++)
                {
                    var end = run + 1 < runs[g].Length ? runs[g][run + 1] : rows.Length;
                    for (var i = runs[g][run]; i < end; i++)
                    {
                        var r = rows[i];
                        if (f.How == "size") { count++; continue; }
                        if (f.How == "count") { if (!source!.IsMissing(r)) count++; continue; }
                        var x = numbers!.NumberAt(r);
                        if (double.IsNaN(x)) continue;
                        count++;
                        var y = x - compensation;   // Kahan, as the group aggregation sums
                        var t = sum + y;
                        compensation = t - sum - y;
                        sum = t;
                        min = Math.Min(min, x);
                        max = Math.Max(max, x);
                    }
                    // SQL: an aggregate over no non-missing values is NULL (COUNT is 0).
                    var value = f.How switch
                    {
                        "size" or "count" => count,
                        _ when count == 0 => double.NaN,
                        "sum" => sum,
                        "mean" => sum / count,
                        "min" => min,
                        _ => max,
                    };
                    for (var i = runs[g][run]; i < end; i++) values[rows[i]] = value;
                }
            }
            results.Add(new NumericSeries(f.Output, values));
        }

        var frameColumns = frame.Columns.ToList();
        frameColumns.AddRange(results);
        return new DataFrame(frameColumns);
    }
}
