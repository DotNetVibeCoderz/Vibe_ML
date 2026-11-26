using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet;

/// <summary>
/// Size estimates for plan nodes, and the join choices made from them.
/// </summary>
/// <remarks>
/// <para>
/// Leaves are measured: in-memory partitions by <see cref="SizeOf"/>, files by their length on
/// disk. Everything above is a rule of thumb - a filter keeps half its input, a projection keeps
/// its share of the columns - and wherever a node's output cannot be bounded from its input
/// (grouped aggregates, joins), the estimate errs large. An underestimate is the costly mistake:
/// it broadcasts something big to every partition. An overestimate only keeps a shuffle that
/// would have happened anyway.
/// </para>
/// <para>
/// With estimates, a join whose right side is small enough is broadcast even when it has several
/// partitions (they are concatenated first), and an inner join whose <i>left</i> side is the small
/// one is turned around, so the big side is never shuffled to meet a small one. Turning a join
/// around changes nothing but row order, which a hash-shuffled join never promised anyway.
/// </para>
/// </remarks>
internal static class CostModel
{
    public const double FilterSelectivity = 0.5;

    /// <summary>Estimated bytes a node produces, or null when nothing is known (opaque functions,
    /// delayed partitions).</summary>
    public static long? Bytes(PlanNode node) => node switch
    {
        ScanNode s => s.Source.EstimatedBytes is { } b
            ? Scale(Scale(b, s.Schema.Count, s.Source.Columns.Count), s.Predicate is null ? 1 : FilterSelectivity)
            : null,
        FilterNode f => Scale(Bytes(f.Input), FilterSelectivity),
        ProjectNode or WithColumnsNode or DropNode or RenameNode or WindowNode => Columns(node),
        AggregateNode a => a.Keys.Length == 0 ? 1024 : Bytes(a.Input),   // no fewer groups than... at most one per row
        JoinNode j => Bytes(j.Left) is { } l && Bytes(j.Right) is { } r ? l + r : null,
        SortNode or RepartitionNode => Bytes(node.Children[0]),
        ConcatNode c => c.Inputs.Select(Bytes).Aggregate((long?)0, (acc, b) => acc is null || b is null ? null : acc + b),
        LimitNode l => Bytes(l.Input) is { } b ? Math.Min(b, 64L * l.Count * Math.Max(1, l.Schema?.Count ?? 1)) : null,
        _ => null,
    };

    private static long? Columns(PlanNode node)
    {
        var input = node.Children[0];
        if (Bytes(input) is not { } b) return null;
        var before = input.Schema?.Count ?? 0;
        var after = node.Schema?.Count ?? before;
        return before == 0 ? b : Scale(b, after, before);
    }

    private static long Scale(long bytes, int part, int whole) => whole == 0 ? bytes : bytes * part / whole;

    private static long? Scale(long? bytes, double factor) => bytes is { } b ? (long)(b * factor) : null;

    // ------------------------------------------------------------------ join choice

    /// <summary>Rewrites joins bottom-up by estimated size; a no-op when the threshold is 0.</summary>
    public static PlanNode ChooseJoins(PlanNode node, long threshold)
    {
        if (threshold <= 0) return node;
        var children = node.Children.Select(c => ChooseJoins(c, threshold)).ToArray();
        if (!children.SequenceEqual(node.Children)) node = node.WithChildren(children);
        if (node is not JoinNode { RequestedPartitions: null, BroadcastHint: null } join || join.Broadcast) return node;

        bool Small(PlanNode side) => Bytes(side) is { } b && b <= threshold;
        if (join.Kind is JoinKind.Inner or JoinKind.Left && Small(join.Right))
            return new JoinNode(join.Left, new RepartitionNode(join.Right, 1), join.LeftOn, join.RightOn, join.Kind, join.Suffix, null);
        if (join.Kind is JoinKind.Inner && Small(join.Left) && join.Left.Schema is not null && join.Right.Schema is not null)
            return Swap(join);
        return node;
    }

    /// <summary>
    /// <c>small JOIN big</c> as <c>big JOIN small</c> (broadcast), projected back to the original
    /// column names and order. Each output column is traced to (side, source column) in both joins.
    /// </summary>
    private static PlanNode Swap(JoinNode join)
    {
        var swapped = new JoinNode(join.Right, new RepartitionNode(join.Left, 1), join.RightOn, join.LeftOn, JoinKind.Inner, join.Suffix, null);
        var original = Origins(join);
        var turned = Origins(swapped).ToDictionary(o => (!o.FromLeft, o.Source), o => o.Output);
        var items = new List<Expr>();
        foreach (var (fromLeft, source, output) in original)
        {
            if (!turned.TryGetValue((fromLeft, source), out var name))
            {
                // A shared single key is kept once, from the left; an inner join's keys are equal,
                // so the other side's copy is the same value.
                var k = Array.IndexOf(fromLeft ? join.LeftOn : join.RightOn, source);
                name = turned[(!fromLeft, fromLeft ? join.RightOn[k] : join.LeftOn[k])];
            }
            items.Add(Expr.Col(name).As(output));
        }
        return new ProjectNode(swapped, items.ToArray());
    }

    /// <summary>The output columns of a join in order, each with the side and column it comes from
    /// (the same naming rule as <see cref="JoinNode.Schema"/>).</summary>
    private static List<(bool FromLeft, string Source, string Output)> Origins(JoinNode join)
    {
        var result = join.Left.Schema!.Select(c => (true, c, c)).ToList();
        var names = new HashSet<string>(join.Left.Schema!, StringComparer.Ordinal);
        var single = join.LeftOn.Length == 1;
        foreach (var r in join.Right.Schema!)
        {
            if (single && r == join.RightOn[0] && join.RightOn[0] == join.LeftOn[0]) continue;
            result.Add((false, r, names.Contains(r) ? r + join.Suffix : r));
        }
        return result;
    }

    /// <summary>"1.5 MB" for plan descriptions.</summary>
    public static string Format(long? bytes) => bytes switch
    {
        null => "size unknown",
        < 1L << 10 => $"{bytes} B",
        < 1L << 20 => $"{bytes / 1024.0:0.#} KB",
        < 1L << 30 => $"{bytes / (1024.0 * 1024):0.#} MB",
        _ => $"{bytes / (1024.0 * 1024 * 1024):0.##} GB",
    };

    /// <summary>Total length of files that exist, or null if none can be measured.</summary>
    public static long? FileBytes(IEnumerable<string> files)
    {
        long total = 0;
        foreach (var f in files)
        {
            try { total += new FileInfo(f).Length; }
            catch (IOException) { return null; }
            catch (UnauthorizedAccessException) { return null; }
        }
        return total;
    }
}
