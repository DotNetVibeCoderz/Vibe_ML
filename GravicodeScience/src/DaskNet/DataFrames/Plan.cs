using System.Reflection;
using System.Text;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;

namespace Gravicode.Science.DaskNet;

/// <summary>A node of a <see cref="DaskFrame"/>'s logical plan.</summary>
internal abstract class PlanNode
{
    public abstract IReadOnlyList<PlanNode> Children { get; }

    /// <summary>Output column names, or <c>null</c> when an opaque function hides them.</summary>
    public abstract IReadOnlyList<string>? Schema { get; }

    public abstract int Partitions { get; }

    public abstract string Describe();

    public abstract PlanNode WithChildren(IReadOnlyList<PlanNode> children);

    /// <summary>Whether every task this node lowers to can run on a cluster.</summary>
    public virtual bool Remotable => Children.All(c => c.Remotable);

    public string Explain()
    {
        var sb = new StringBuilder();
        Write(sb, 0);
        return sb.ToString();
    }

    private void Write(StringBuilder sb, int depth)
    {
        sb.Append(' ', depth * 2).AppendLine(Describe());
        foreach (var c in Children) c.Write(sb, depth + 1);
    }

    protected static string List(IEnumerable<string> names) => string.Join(", ", names);
}

/// <summary>Where a scan's partitions come from.</summary>
internal abstract class ScanSource
{
    public abstract IReadOnlyList<string> Columns { get; }
    public abstract int Partitions { get; }
    public abstract bool Remotable { get; }
    public abstract string Describe();

    /// <summary>Tasks producing the final partitions: only <paramref name="columns"/> (all if null),
    /// with <paramref name="ops"/> applied.</summary>
    public abstract TaskSpec[] Build(string name, string[]? columns, IReadOnlyList<FrameOp> ops);

    protected static readonly MethodInfo ApplyOpsMethod = typeof(FrameKernels).GetMethod(nameof(FrameKernels.ApplyOps))!;

    protected static List<FrameOp> WithProjection(string[]? columns, IReadOnlyList<FrameOp> ops)
    {
        var all = new List<FrameOp>();
        if (columns is not null) all.Add(new SelectOp(columns.Select(Expr.Col).ToArray()));
        all.AddRange(ops);
        return all;
    }
}

internal sealed class CsvSource(IReadOnlyList<(string Path, long Start, long End)> blocks, string[] names, DataType[] types,
    string delimiter, string[] missing) : ScanSource
{
    private static readonly MethodInfo ReadMethod = typeof(FrameKernels).GetMethod(nameof(FrameKernels.ReadCsvBlock))!;

    public override IReadOnlyList<string> Columns => names;
    public override int Partitions => blocks.Count;
    public override bool Remotable => true;
    public IReadOnlyList<DataType> Types => types;

    public override string Describe()
        => $"ReadCsv {blocks.Select(b => Path.GetFileName(b.Path)).Distinct().Count()} file(s), {blocks.Count} block(s)";

    public override TaskSpec[] Build(string name, string[]? columns, IReadOnlyList<FrameOp> ops)
    {
        var keep = (columns ?? names).Select(c => Array.IndexOf(names, c)).ToArray();
        if (keep.Any(k => k < 0)) throw new KeyNotFoundException($"Unknown column in [{string.Join(", ", columns!)}].");
        var encoded = FrameOp.Encode(ops);
        var typeCodes = types.Select(t => (int)t).ToArray();
        return blocks.Select((b, i) => TaskSpec.Call(new TaskKey(name, i), ReadMethod,
            [b.Path, b.Start, b.End, names, typeCodes, keep, delimiter, missing, encoded], "read-csv")).ToArray();
    }
}

internal sealed class ArrowSource(string[] files, string[] columns) : ScanSource
{
    private static readonly MethodInfo ReadMethod = typeof(FrameKernels).GetMethod(nameof(FrameKernels.ReadArrow))!;

    public override IReadOnlyList<string> Columns => columns;
    public override int Partitions => files.Length;
    public override bool Remotable => true;
    public override string Describe() => $"ReadArrow {files.Length} file(s)";

    public override TaskSpec[] Build(string name, string[]? cols, IReadOnlyList<FrameOp> ops)
    {
        var encoded = FrameOp.Encode(ops);
        return files.Select((f, i) => TaskSpec.Call(new TaskKey(name, i), ReadMethod, [f, cols ?? [], encoded], "read-arrow")).ToArray();
    }
}

internal sealed class MemorySource(DataFrame[] partitions, string label = "from-frame") : ScanSource
{
    public override IReadOnlyList<string> Columns => partitions[0].ColumnNames;
    public override int Partitions => partitions.Length;
    public override bool Remotable => true;
    public override string Describe() => $"{(label == "from-frame" ? "FromFrame" : label)} {partitions.Length} partition(s), {partitions.Sum(p => (long)p.RowCount)} rows";

    public override TaskSpec[] Build(string name, string[]? columns, IReadOnlyList<FrameOp> ops)
    {
        var all = WithProjection(columns is not null && columns.SequenceEqual(Columns) ? null : columns, ops);
        if (all.Count == 0)
            return partitions.Select((p, i) => TaskSpec.Value(new TaskKey(name, i), p, label)).ToArray();
        var encoded = FrameOp.Encode(all);
        return partitions.Select((p, i) => TaskSpec.Call(new TaskKey(name, i), ApplyOpsMethod, [p, encoded], label)).ToArray();
    }
}

internal sealed class DelegateSource(string[] files, Func<string, DataFrame> reader, string[] columns) : ScanSource
{
    public override IReadOnlyList<string> Columns => columns;
    public override int Partitions => files.Length;
    public override bool Remotable => false;
    public override string Describe() => $"ReadFiles {files.Length} file(s) with {reader.Method.Name}";

    public override TaskSpec[] Build(string name, string[]? cols, IReadOnlyList<FrameOp> ops)
    {
        var all = WithProjection(cols, ops);
        return files.Select((f, i) => TaskSpec.Create(new TaskKey(name, i), [], _ => FrameOp.ApplyAll(reader(f), all), "read-files")).ToArray();
    }
}

internal sealed class DelayedSource(Delayed<DataFrame>[] parts, string[] columns) : ScanSource
{
    public override IReadOnlyList<string> Columns => columns;
    public override int Partitions => parts.Length;
    public override bool Remotable => false;
    public override string Describe() => $"FromDelayed {parts.Length} partition(s)";
    public IReadOnlyList<GraphLayer> Layers => parts.Select(p => p.Layer).ToArray();

    public override TaskSpec[] Build(string name, string[]? cols, IReadOnlyList<FrameOp> ops)
    {
        var encoded = FrameOp.Encode(WithProjection(cols, ops));
        return parts.Select((p, i) => TaskSpec.Call(new TaskKey(name, i), ApplyOpsMethod, [new TaskRef(p.Key), encoded], "from-delayed")).ToArray();
    }
}

internal sealed class ScanNode(ScanSource source, string[]? columns = null, Expr? predicate = null) : PlanNode
{
    public ScanSource Source { get; } = source;
    public string[]? Columns { get; } = columns;
    public Expr? Predicate { get; } = predicate;
    public override IReadOnlyList<PlanNode> Children => [];
    public override IReadOnlyList<string> Schema => Columns ?? Source.Columns;
    public override int Partitions => Source.Partitions;
    public override bool Remotable => Source.Remotable;
    public override PlanNode WithChildren(IReadOnlyList<PlanNode> children) => this;

    public override string Describe()
    {
        var sb = new StringBuilder(Source.Describe());
        if (Columns is not null) sb.Append($" columns=[{List(Columns)}]");
        if (Predicate is not null) sb.Append($" filter={Predicate}");
        return sb.ToString();
    }
}

internal sealed class FilterNode(PlanNode input, Expr predicate) : PlanNode
{
    public PlanNode Input { get; } = input;
    public Expr Predicate { get; } = predicate;
    public override IReadOnlyList<PlanNode> Children => [Input];
    public override IReadOnlyList<string>? Schema => Input.Schema;
    public override int Partitions => Input.Partitions;
    public override string Describe() => $"Filter {Predicate}";
    public override PlanNode WithChildren(IReadOnlyList<PlanNode> c) => new FilterNode(c[0], Predicate);
}

internal sealed class ProjectNode(PlanNode input, Expr[] items) : PlanNode
{
    public PlanNode Input { get; } = input;
    public Expr[] Items { get; } = items;
    public override IReadOnlyList<PlanNode> Children => [Input];
    public override IReadOnlyList<string> Schema => Items.Select(i => i.OutputName).ToArray();
    public override int Partitions => Input.Partitions;
    public override string Describe() => new SelectOp(Items).Describe();
    public override PlanNode WithChildren(IReadOnlyList<PlanNode> c) => new ProjectNode(c[0], Items);
}

internal sealed class WithColumnsNode(PlanNode input, Expr[] items) : PlanNode
{
    public PlanNode Input { get; } = input;
    public Expr[] Items { get; } = items;
    public override IReadOnlyList<PlanNode> Children => [Input];
    public override IReadOnlyList<string>? Schema
        => Input.Schema is null ? null : Input.Schema.Concat(Items.Select(i => i.OutputName).Where(n => !Input.Schema.Contains(n))).ToArray();
    public override int Partitions => Input.Partitions;
    public override string Describe() => new WithColumnsOp(Items).Describe();
    public override PlanNode WithChildren(IReadOnlyList<PlanNode> c) => new WithColumnsNode(c[0], Items);
}

internal sealed class DropNode(PlanNode input, string[] names) : PlanNode
{
    public PlanNode Input { get; } = input;
    public string[] Names { get; } = names;
    public override IReadOnlyList<PlanNode> Children => [Input];
    public override IReadOnlyList<string>? Schema => Input.Schema?.Where(n => !Names.Contains(n)).ToArray();
    public override int Partitions => Input.Partitions;
    public override string Describe() => new DropOp(Names).Describe();
    public override PlanNode WithChildren(IReadOnlyList<PlanNode> c) => new DropNode(c[0], Names);
}

internal sealed class RenameNode(PlanNode input, (string From, string To)[] pairs) : PlanNode
{
    public PlanNode Input { get; } = input;
    public (string From, string To)[] Pairs { get; } = pairs;
    public override IReadOnlyList<PlanNode> Children => [Input];
    public override IReadOnlyList<string>? Schema => Input.Schema?.Select(Forward).ToArray();
    public override int Partitions => Input.Partitions;
    public override string Describe() => new RenameOp(Pairs).Describe();
    public override PlanNode WithChildren(IReadOnlyList<PlanNode> c) => new RenameNode(c[0], Pairs);
    public string Forward(string n) => Pairs.FirstOrDefault(p => p.From == n).To ?? n;
    public string Backward(string n) => Pairs.FirstOrDefault(p => p.To == n).From ?? n;
}

internal sealed class MapPartitionsNode(PlanNode input, Func<DataFrame, DataFrame> function, string[]? schema, string label) : PlanNode
{
    public PlanNode Input { get; } = input;
    public Func<DataFrame, DataFrame> Function { get; } = function;
    public string Label { get; } = label;
    public override IReadOnlyList<PlanNode> Children => [Input];
    public override IReadOnlyList<string>? Schema => schema;
    public override int Partitions => Input.Partitions;
    public override bool Remotable => false;
    public override string Describe() => $"MapPartitions {Label} (opaque: blocks pushdown)";
    public override PlanNode WithChildren(IReadOnlyList<PlanNode> c) => new MapPartitionsNode(c[0], Function, schema, Label);
}

internal sealed class AggregateNode(PlanNode input, string[] keys, Aggregation[] aggs, int splitEvery) : PlanNode
{
    public PlanNode Input { get; } = input;
    public string[] Keys { get; } = keys;
    public Aggregation[] Aggs { get; } = aggs;
    public int SplitEvery { get; } = splitEvery;
    public override IReadOnlyList<PlanNode> Children => [Input];
    public override IReadOnlyList<string> Schema => Keys.Concat(Aggs.Select(a => a.Output)).ToArray();
    public override int Partitions => 1;
    public override string Describe()
        => $"Aggregate{(Keys.Length > 0 ? $" by [{List(Keys)}]" : "")}: {string.Join(", ", Aggs.Select(a => $"{a.Output} = {a.How}({a.Column ?? "*"})"))}";
    public override PlanNode WithChildren(IReadOnlyList<PlanNode> c) => new AggregateNode(c[0], Keys, Aggs, SplitEvery);
}

internal sealed class JoinNode(PlanNode left, PlanNode right, string[] leftOn, string[] rightOn, JoinKind kind, string suffix,
    int? partitions) : PlanNode
{
    public PlanNode Left { get; } = left;
    public PlanNode Right { get; } = right;
    public string[] LeftOn { get; } = leftOn;
    public string[] RightOn { get; } = rightOn;
    public JoinKind Kind { get; } = kind;
    public string Suffix { get; } = suffix;
    public int? RequestedPartitions { get; } = partitions;
    public override IReadOnlyList<PlanNode> Children => [Left, Right];

    /// <summary>Broadcast when the right side is one partition and every right row is either matched
    /// or dropped - true for inner and left joins only.</summary>
    public bool Broadcast => Right.Partitions == 1 && Kind is JoinKind.Inner or JoinKind.Left && RequestedPartitions is null;

    public override int Partitions => Broadcast ? Left.Partitions : RequestedPartitions ?? Math.Max(Left.Partitions, Right.Partitions);

    public override IReadOnlyList<string>? Schema
    {
        get
        {
            if (Left.Schema is null || Right.Schema is null) return null;
            var left = Left.Schema.ToList();
            var names = new HashSet<string>(left, StringComparer.Ordinal);
            var single = LeftOn.Length == 1;
            foreach (var r in Right.Schema)
            {
                if (single && r == RightOn[0] && RightOn[0] == LeftOn[0]) continue;
                left.Add(names.Contains(r) ? r + Suffix : r);
            }
            return left;
        }
    }

    public override string Describe()
        => $"Join {Kind} on {(LeftOn.SequenceEqual(RightOn) ? List(LeftOn) : $"[{List(LeftOn)}] = [{List(RightOn)}]")}"
           + (Broadcast ? " (broadcast right)" : $" (hash shuffle, {Partitions} partitions)");

    public override PlanNode WithChildren(IReadOnlyList<PlanNode> c) => new JoinNode(c[0], c[1], LeftOn, RightOn, Kind, Suffix, RequestedPartitions);
}

internal sealed class SortNode(PlanNode input, string[] columns, bool[] ascending, int? partitions) : PlanNode
{
    public PlanNode Input { get; } = input;
    public string[] Columns { get; } = columns;
    public bool[] Ascending { get; } = ascending;
    public int? RequestedPartitions { get; } = partitions;
    public override IReadOnlyList<PlanNode> Children => [Input];
    public override IReadOnlyList<string>? Schema => Input.Schema;
    public override int Partitions => RequestedPartitions ?? Input.Partitions;
    public override string Describe()
        => $"Sort by {string.Join(", ", Columns.Select((c, i) => $"{c} {(Ascending[i] ? "ASC" : "DESC")}"))} (range partitioned, {Partitions} partitions)";
    public override PlanNode WithChildren(IReadOnlyList<PlanNode> c) => new SortNode(c[0], Columns, Ascending, RequestedPartitions);
}

internal sealed class RepartitionNode(PlanNode input, int partitions) : PlanNode
{
    public PlanNode Input { get; } = input;
    public override IReadOnlyList<PlanNode> Children => [Input];
    public override IReadOnlyList<string>? Schema => Input.Schema;
    public override int Partitions { get; } = partitions;
    public override string Describe() => $"Repartition {Input.Partitions} -> {Partitions}";
    public override PlanNode WithChildren(IReadOnlyList<PlanNode> c) => new RepartitionNode(c[0], Partitions);
}

internal sealed class ConcatNode(PlanNode[] inputs) : PlanNode
{
    public PlanNode[] Inputs { get; } = inputs;
    public override IReadOnlyList<PlanNode> Children => Inputs;
    public override IReadOnlyList<string>? Schema => Inputs[0].Schema;
    public override int Partitions => Inputs.Sum(i => i.Partitions);
    public override string Describe() => $"Concat {Inputs.Length} inputs";
    public override PlanNode WithChildren(IReadOnlyList<PlanNode> c) => new ConcatNode(c.ToArray());
}

internal sealed class LimitNode(PlanNode input, int count) : PlanNode
{
    public PlanNode Input { get; } = input;
    public int Count { get; } = count;
    public override IReadOnlyList<PlanNode> Children => [Input];
    public override IReadOnlyList<string>? Schema => Input.Schema;
    public override int Partitions => 1;
    public override string Describe() => $"Limit {Count}";
    public override PlanNode WithChildren(IReadOnlyList<PlanNode> c) => new LimitNode(c[0], Count);
}
