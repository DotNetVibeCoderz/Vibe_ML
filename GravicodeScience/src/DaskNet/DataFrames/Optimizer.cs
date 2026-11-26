using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet;

/// <summary>
/// Rewrites of a <see cref="DaskFrame"/> plan: predicate pushdown, projection pushdown, filter
/// merging, then join strategy by estimated size (<see cref="CostModel"/>). Row-wise operations left adjacent afterwards are fused into one task per
/// partition when the plan is lowered.
/// </summary>
/// <remarks>
/// Every rule preserves results; the tests compare optimised and unoptimised plans on the same
/// data for exactly that reason. The rules that are easy to get wrong are the ones through joins:
/// a predicate on the right side of a <i>left</i> join must stay above it, because below it would
/// turn rows the join keeps (with missing right values) into rows it drops.
/// </remarks>
internal static class Optimizer
{
    public static PlanNode Optimize(PlanNode plan)
    {
        var current = plan;
        for (var pass = 0; pass < 16; pass++)
        {
            var next = PushFilters(current);
            if (ReferenceEquals(next, current)) break;
            current = next;
        }
        return CostModel.ChooseJoins(Prune(current, null), Dask.BroadcastJoinBytes);
    }

    // ------------------------------------------------------------------ predicate pushdown

    private static PlanNode PushFilters(PlanNode node)
    {
        var children = node.Children.Select(PushFilters).ToArray();
        if (!children.SequenceEqual(node.Children)) node = node.WithChildren(children);
        return node is FilterNode f ? PushOne(f) ?? node : node;
    }

    /// <summary>Moves one filter down a level, or returns null if it cannot move.</summary>
    private static PlanNode? PushOne(FilterNode filter)
    {
        var p = filter.Predicate;
        switch (filter.Input)
        {
            case FilterNode inner:
                return new FilterNode(inner.Input, inner.Predicate & p);

            case ScanNode scan when Within(p, scan.Source.Columns):
                return new ScanNode(scan.Source, scan.Columns, scan.Predicate is null ? p : scan.Predicate & p);

            case ProjectNode project:
            {
                // Only through items that pass a column through, possibly renamed.
                var map = new Dictionary<string, string>(StringComparer.Ordinal);
                foreach (var item in project.Items)
                {
                    var inner = item is AliasExpr a ? a.Inner : item;
                    if (inner is ColumnExpr c) map[item.OutputName] = c.Name;
                }
                if (!p.Columns().All(map.ContainsKey)) return null;
                return new ProjectNode(new FilterNode(project.Input, p.RenameColumns(n => map[n])), project.Items);
            }

            case WithColumnsNode with when !p.Columns().Overlaps(with.Items.Select(i => i.OutputName)):
                return new WithColumnsNode(new FilterNode(with.Input, p), with.Items);

            case DropNode drop:
                return new DropNode(new FilterNode(drop.Input, p), drop.Names);

            case RenameNode rename:
                return new RenameNode(new FilterNode(rename.Input, p.RenameColumns(rename.Backward)), rename.Pairs);

            case SortNode sort:
                return new SortNode(new FilterNode(sort.Input, p), sort.Columns, sort.Ascending, sort.RequestedPartitions);

            case RepartitionNode re:
                return new RepartitionNode(new FilterNode(re.Input, p), re.Partitions);

            case ConcatNode concat:
                return new ConcatNode(concat.Inputs.Select(i => (PlanNode)new FilterNode(i, p)).ToArray());

            case AggregateNode agg:
            {
                // Conjuncts on group keys only can be applied before grouping (HAVING on a key).
                var (down, up) = Partition(p, c => agg.Keys.Contains(c));
                if (down is null) return null;
                PlanNode moved = new AggregateNode(new FilterNode(agg.Input, down), agg.Keys, agg.Aggs, agg.SplitEvery);
                return up is null ? moved : new FilterNode(moved, up);
            }

            case JoinNode join:
                return PushIntoJoin(join, p);

            default:
                return null;
        }
    }

    private static PlanNode? PushIntoJoin(JoinNode join, Expr predicate)
    {
        if (join.Left.Schema is null || join.Right.Schema is null) return null;
        var leftNames = join.Left.Schema.ToHashSet(StringComparer.Ordinal);
        var rightNames = join.Right.Schema.ToHashSet(StringComparer.Ordinal);
        var sharedKey = join.LeftOn.Length == 1 && join.LeftOn[0] == join.RightOn[0] ? join.LeftOn[0] : null;
        var canLeft = join.Kind is JoinKind.Inner or JoinKind.Left;
        var canRight = join.Kind is JoinKind.Inner or JoinKind.Right;

        Expr? toLeft = null, toRight = null, stay = null;
        foreach (var conjunct in Expr.SplitAnd(predicate))
        {
            var cols = conjunct.Columns();
            if (cols.Count == 0) { stay = stay is null ? conjunct : stay & conjunct; continue; }

            // An inner join on a shared key: a key predicate filters both sides.
            if (sharedKey is not null && join.Kind == JoinKind.Inner && cols.All(c => c == sharedKey))
            {
                toLeft = toLeft is null ? conjunct : toLeft & conjunct;
                toRight = toRight is null ? conjunct : toRight & conjunct;
                continue;
            }
            // A column the output takes unchanged from the left (right duplicates are suffixed).
            if (canLeft && cols.All(leftNames.Contains))
            {
                toLeft = toLeft is null ? conjunct : toLeft & conjunct;
                continue;
            }
            // From the right only if the name is not also on the left (which would make it suffixed).
            if (canRight && cols.All(c => rightNames.Contains(c) && !leftNames.Contains(c)))
            {
                toRight = toRight is null ? conjunct : toRight & conjunct;
                continue;
            }
            stay = stay is null ? conjunct : stay & conjunct;
        }

        if (toLeft is null && toRight is null) return null;
        PlanNode moved = new JoinNode(
            toLeft is null ? join.Left : new FilterNode(join.Left, toLeft),
            toRight is null ? join.Right : new FilterNode(join.Right, toRight),
            join.LeftOn, join.RightOn, join.Kind, join.Suffix, join.RequestedPartitions, join.BroadcastHint);
        return stay is null ? moved : new FilterNode(moved, stay);
    }

    private static (Expr? Down, Expr? Up) Partition(Expr predicate, Func<string, bool> canMove)
    {
        var down = new List<Expr>();
        var up = new List<Expr>();
        foreach (var c in Expr.SplitAnd(predicate))
            (c.Columns().Count > 0 && c.Columns().All(canMove) ? down : up).Add(c);
        return (Expr.JoinAnd(down), Expr.JoinAnd(up));
    }

    private static bool Within(Expr e, IReadOnlyList<string> schema) => e.Columns().All(schema.Contains);

    // ------------------------------------------------------------------ projection pushdown

    /// <summary>Rewrites <paramref name="node"/> so it produces at least <paramref name="required"/>
    /// (null: everything) while reading as few columns as possible.</summary>
    private static PlanNode Prune(PlanNode node, HashSet<string>? required)
    {
        switch (node)
        {
            case ScanNode scan:
            {
                if (required is null && scan.Predicate is null) return scan;
                var need = required is null ? scan.Source.Columns.ToHashSet() : new HashSet<string>(required);
                if (scan.Predicate is not null) need.UnionWith(scan.Predicate.Columns());
                var columns = scan.Source.Columns.Where(need.Contains).ToArray();
                // A frame with no columns has no rows either, so COUNT(*) or SELECT 1 - which name
                // no column - would count nothing. Keep one column to carry the row count.
                if (columns.Length == 0 && scan.Source.Columns.Count > 0) columns = [scan.Source.Columns[0]];
                return new ScanNode(scan.Source, columns.Length == scan.Source.Columns.Count ? null : columns, scan.Predicate);
            }

            case FilterNode f:
                return new FilterNode(Prune(f.Input, Union(required, f.Predicate.Columns())), f.Predicate);

            case ProjectNode project:
            {
                var items = required is null ? project.Items : project.Items.Where(i => required.Contains(i.OutputName)).ToArray();
                if (items.Length == 0) items = project.Items.Take(1).ToArray();
                var need = items.SelectMany(i => i.Columns()).ToHashSet(StringComparer.Ordinal);
                return new ProjectNode(Prune(project.Input, need), items);
            }

            case WithColumnsNode with:
            {
                var items = required is null ? with.Items : with.Items.Where(i => required.Contains(i.OutputName)).ToArray();
                HashSet<string>? need = null;
                if (required is not null)
                {
                    need = new HashSet<string>(required.Except(items.Select(i => i.OutputName)), StringComparer.Ordinal);
                    foreach (var i in items) need.UnionWith(i.Columns());
                }
                var input = Prune(with.Input, need);
                return items.Length == 0 ? input : new WithColumnsNode(input, items);
            }

            case DropNode drop:
            {
                if (required is null) return new DropNode(Prune(drop.Input, null), drop.Names);
                // Once the input is pruned to what is required, the dropped columns are gone already.
                var input = Prune(drop.Input, required);
                var stillThere = drop.Names.Where(n => input.Schema?.Contains(n) ?? true).ToArray();
                return stillThere.Length == 0 ? input : new DropNode(input, stillThere);
            }

            case RenameNode rename:
                return new RenameNode(Prune(rename.Input, required?.Select(rename.Backward).ToHashSet(StringComparer.Ordinal)), rename.Pairs);

            case MapPartitionsNode map:
                return map.WithChildren([Prune(map.Input, null)]);

            case AggregateNode agg:
            {
                var need = agg.Keys.Concat(agg.Aggs.Where(a => a.Column is not null).Select(a => a.Column!))
                    .ToHashSet(StringComparer.Ordinal);
                return agg.WithChildren([Prune(agg.Input, need)]);
            }

            case JoinNode join:
            {
                if (required is null || join.Left.Schema is null || join.Right.Schema is null)
                    return join.WithChildren([Prune(join.Left, null), Prune(join.Right, null)]);
                var left = join.Left.Schema.ToHashSet(StringComparer.Ordinal);
                var right = join.Right.Schema.ToHashSet(StringComparer.Ordinal);
                var needLeft = new HashSet<string>(join.LeftOn, StringComparer.Ordinal);
                var needRight = new HashSet<string>(join.RightOn, StringComparer.Ordinal);
                foreach (var r in required)
                {
                    if (left.Contains(r)) needLeft.Add(r);
                    if (right.Contains(r)) needRight.Add(r);
                    if (r.EndsWith(join.Suffix, StringComparison.Ordinal) && right.Contains(r[..^join.Suffix.Length]))
                        needRight.Add(r[..^join.Suffix.Length]);
                }
                // A right column whose name the left also has gets a suffix; keeping the left copy
                // keeps that naming stable whether or not the left one was otherwise needed.
                foreach (var r in needRight) if (left.Contains(r)) needLeft.Add(r);
                return join.WithChildren([Prune(join.Left, needLeft), Prune(join.Right, needRight)]);
            }

            case SortNode sort:
                return sort.WithChildren([Prune(sort.Input, Union(required, sort.Columns))]);

            case WindowNode window:
            {
                if (required is null) return window.WithChildren([Prune(window.Input, null)]);
                var outputs = window.Spec.Functions.Select(f => f.Output).ToHashSet(StringComparer.Ordinal);
                var need = required.Where(r => !outputs.Contains(r)).Concat(window.Spec.InputColumns).ToHashSet(StringComparer.Ordinal);
                return window.WithChildren([Prune(window.Input, need)]);
            }

            case RepartitionNode or LimitNode:
                return node.WithChildren([Prune(node.Children[0], required)]);

            case ConcatNode concat:
                // Every input must keep the same columns in the same order.
                return concat.WithChildren(concat.Inputs.Select(i => Prune(i, required ?? i.Schema?.ToHashSet())).ToArray());

            default:
                return node.WithChildren(node.Children.Select(c => Prune(c, null)).ToArray());
        }
    }

    private static HashSet<string>? Union(HashSet<string>? a, IEnumerable<string> b)
    {
        if (a is null) return null;
        var r = new HashSet<string>(a, StringComparer.Ordinal);
        r.UnionWith(b);
        return r;
    }
}
