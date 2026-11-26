using System.Reflection;
using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet;

/// <summary>
/// Turns an optimised plan into a task graph. Row-wise operations are not given tasks of their
/// own: they ride down to the nearest task that has to exist (a file scan, a shuffle output) and
/// run inside it - pipeline fusion at plan level, before graph-level fusion sees anything.
/// </summary>
internal sealed class Lowering(IScheduler scheduler)
{
    private static MethodInfo K(string name) => typeof(FrameKernels).GetMethod(name)!;
    private static readonly MethodInfo ApplyOps = K(nameof(FrameKernels.ApplyOps));
    private static readonly MethodInfo Partial = K(nameof(FrameKernels.PartialAggregate));
    private static readonly MethodInfo Combine = K(nameof(FrameKernels.CombineAggregate));
    private static readonly MethodInfo Finish = K(nameof(FrameKernels.FinishAggregate));
    private static readonly MethodInfo Concat = K(nameof(FrameKernels.ConcatFrames));
    private static readonly MethodInfo Head = K(nameof(FrameKernels.HeadFrame));
    private static readonly MethodInfo Split = K(nameof(FrameKernels.SplitFrame));
    private static readonly MethodInfo Pick = K(nameof(FrameKernels.PickFrame));
    private static readonly MethodInfo HashSplit = K(nameof(FrameKernels.HashSplit));
    private static readonly MethodInfo Join = K(nameof(FrameKernels.JoinFrames));
    private static readonly MethodInfo Sample = K(nameof(FrameKernels.SampleColumn));
    private static readonly MethodInfo RangeSplit = K(nameof(FrameKernels.RangeSplit));
    private static readonly MethodInfo Sort = K(nameof(FrameKernels.SortFrame));
    private static readonly MethodInfo WindowKernel = K(nameof(FrameKernels.WindowFrame));

    /// <summary>The partitions of a lowered node: their keys and the layer that holds them.</summary>
    public sealed record Parts(GraphLayer Layer, TaskKey[] Keys) : IDaskCollection
    {
        IReadOnlyList<TaskKey> IDaskCollection.OutputKeys => Keys;
    }

    public Parts Lower(PlanNode node) => Lower(node, []);

    private Parts Lower(PlanNode node, List<FrameOp> pending)
    {
        switch (node)
        {
            case FilterNode f: return Lower(f.Input, Prepend(new FilterOp(f.Predicate), pending));
            case ProjectNode p: return Lower(p.Input, Prepend(new SelectOp(p.Items), pending));
            case WithColumnsNode w: return Lower(w.Input, Prepend(new WithColumnsOp(w.Items), pending));
            case DropNode d: return Lower(d.Input, Prepend(new DropOp(d.Names), pending));
            case RenameNode r: return Lower(r.Input, Prepend(new RenameOp(r.Pairs), pending));

            case ScanNode scan:
            {
                var ops = new List<FrameOp>();
                if (scan.Predicate is not null) ops.Add(new FilterOp(scan.Predicate));
                ops.AddRange(pending);
                var name = TaskKey.NewName("read");
                var tasks = scan.Source.Build(name, scan.Columns, ops);
                var parents = scan.Source is DelayedSource ds ? ds.Layers : [];
                return new Parts(new GraphLayer(name, tasks, parents), tasks.Select(t => t.Key).ToArray());
            }
        }

        var parts = node switch
        {
            MapPartitionsNode m => LowerMap(m),
            AggregateNode a => LowerAggregate(a),
            JoinNode j => LowerJoin(j),
            SortNode s => LowerSort(s),
            RepartitionNode r => LowerRepartition(r),
            ConcatNode c => LowerConcat(c),
            LimitNode l => LowerLimit(l),
            WindowNode w => LowerWindow(w),
            _ => throw new NotSupportedException(node.GetType().Name),
        };
        return pending.Count == 0 ? parts : Apply(parts, pending);
    }

    private static List<FrameOp> Prepend(FrameOp op, List<FrameOp> rest) => [op, .. rest];

    private static Parts Apply(Parts parts, IReadOnlyList<FrameOp> ops)
    {
        var name = TaskKey.NewName("partition-ops");
        var encoded = FrameOp.Encode(ops);
        var tasks = parts.Keys.Select((k, i) => TaskSpec.Call(new TaskKey(name, i), ApplyOps, [new TaskRef(k), encoded], "partition-ops")).ToArray();
        return new Parts(new GraphLayer(name, tasks, [parts.Layer]), tasks.Select(t => t.Key).ToArray());
    }

    private Parts LowerMap(MapPartitionsNode m)
    {
        var input = Lower(m.Input);
        var name = TaskKey.NewName(m.Label);
        var fn = m.Function;
        var tasks = input.Keys.Select((k, i) => TaskSpec.Create(new TaskKey(name, i), [k], v => fn((DataFrame)v[0]!), m.Label)).ToArray();
        return new Parts(new GraphLayer(name, tasks, [input.Layer]), tasks.Select(t => t.Key).ToArray());
    }

    private Parts LowerAggregate(AggregateNode a)
    {
        var input = Lower(a.Input);
        var spec = Aggregation.Encode(a.Keys, a.Aggs);
        var partialName = TaskKey.NewName("aggregate-partial");
        var partials = input.Keys.Select((k, i) =>
            TaskSpec.Call(new TaskKey(partialName, i), Partial, [new TaskRef(k), spec], "aggregate-partial")).ToArray();
        var partialLayer = new GraphLayer(partialName, partials, [input.Layer]);

        var treeName = TaskKey.NewName("aggregate-combine");
        var (tree, root) = GraphBuilders.TreeReduce(treeName, partials.Select(p => p.Key).ToArray(), a.SplitEvery,
            (key, inputs) => TaskSpec.Call(key, Combine, [new TaskRefList(inputs), spec], "aggregate-combine"));
        var treeLayer = new GraphLayer(treeName, tree, [partialLayer]);

        var finalName = TaskKey.NewName("aggregate");
        var final = TaskSpec.Call(new TaskKey(finalName, 0), Finish, [new TaskRef(root), spec], "aggregate");
        return new Parts(new GraphLayer(finalName, [final], [treeLayer]), [final.Key]);
    }

    private Parts LowerJoin(JoinNode j)
    {
        var left = Lower(j.Left);
        var right = Lower(j.Right);
        var name = TaskKey.NewName("join");

        if (j.Broadcast)
        {
            var tasks = left.Keys.Select((k, i) => TaskSpec.Call(new TaskKey(name, i), Join,
                [new TaskRef(k), new TaskRef(right.Keys[0]), j.LeftOn, j.RightOn, (int)j.Kind, j.Suffix], "join-broadcast")).ToArray();
            return new Parts(new GraphLayer(name, tasks, [left.Layer, right.Layer]), tasks.Select(t => t.Key).ToArray());
        }

        var n = j.Partitions;
        var (leftBuckets, leftLayer) = Shuffle(left, j.LeftOn, n, "shuffle-left");
        var (rightBuckets, rightLayer) = Shuffle(right, j.RightOn, n, "shuffle-right");
        var joins = Enumerable.Range(0, n).Select(b => TaskSpec.FromRemote(new TaskKey(name, b), new RemoteCall(Join,
            new RemoteCall(Concat, new TaskRefList(leftBuckets[b])),
            new RemoteCall(Concat, new TaskRefList(rightBuckets[b])),
            j.LeftOn, j.RightOn, (int)j.Kind, j.Suffix), "join-shuffle")).ToArray();
        return new Parts(new GraphLayer(name, joins, [leftLayer, rightLayer]), joins.Select(t => t.Key).ToArray());
    }

    /// <summary>Hash-splits every partition into <paramref name="n"/> buckets; returns, per
    /// bucket, the keys of its piece from each input partition.</summary>
    private static (List<TaskKey>[] Buckets, GraphLayer Layer) Shuffle(Parts input, string[] keys, int n, string label)
    {
        var splitName = TaskKey.NewName(label + "-split");
        var splits = input.Keys.Select((k, i) =>
            TaskSpec.Call(new TaskKey(splitName, i), HashSplit, [new TaskRef(k), keys, n], label + "-split")).ToArray();
        var splitLayer = new GraphLayer(splitName, splits, [input.Layer]);
        return Pieces(splits, n, label, splitLayer);
    }

    private static (List<TaskKey>[] Buckets, GraphLayer Layer) Pieces(TaskSpec[] splits, int n, string label, GraphLayer splitLayer)
    {
        var pickName = TaskKey.NewName(label + "-pick");
        var picks = new List<TaskSpec>();
        var buckets = Enumerable.Range(0, n).Select(_ => new List<TaskKey>()).ToArray();
        for (var i = 0; i < splits.Length; i++)
            for (var b = 0; b < n; b++)
            {
                var key = new TaskKey(pickName, i * n + b);
                picks.Add(TaskSpec.Call(key, Pick, [new TaskRef(splits[i].Key), b], label + "-pick"));
                buckets[b].Add(key);
            }
        return (buckets, new GraphLayer(pickName, picks, [splitLayer]));
    }

    private Parts LowerSort(SortNode s)
    {
        var input = Lower(s.Input);
        var n = s.Partitions;
        var column = s.Columns[0];
        var name = TaskKey.NewName("sort");

        if (n == 1)
        {
            var single = TaskSpec.FromRemote(new TaskKey(name, 0), new RemoteCall(Sort,
                new RemoteCall(Concat, new TaskRefList(input.Keys)), s.Columns, s.Ascending), "sort");
            return new Parts(new GraphLayer(name, [single], [input.Layer]), [single.Key]);
        }

        // Range boundaries from a sample of every partition, computed now - the shape of the rest
        // of the graph depends on them (Dask's set_index does the same).
        var sampleName = TaskKey.NewName("sort-sample");
        var samples = input.Keys.Select((k, i) =>
            TaskSpec.Call(new TaskKey(sampleName, i), Sample, [new TaskRef(k), column, 20 * n], "sort-sample")).ToArray();
        var sampleLayer = new GraphLayer(sampleName, samples, [input.Layer]);
        var sampled = scheduler.ExecuteAsync(TaskGraph.FromLayers([sampleLayer]), samples.Select(t => t.Key).ToArray())
            .GetAwaiter().GetResult();
        var boundaries = Boundaries(samples.Select(t => (DataFrame)sampled[t.Key]!).ToArray(), column, n);

        var splitName = TaskKey.NewName("sort-split");
        var splits = input.Keys.Select((k, i) => TaskSpec.Call(new TaskKey(splitName, i), RangeSplit,
            [new TaskRef(k), column, boundaries, s.Ascending[0]], "sort-split")).ToArray();
        var splitLayer = new GraphLayer(splitName, splits, [input.Layer]);
        var buckets = boundaries.RowCount + 1;
        var (pieces, pickLayer) = Pieces(splits, buckets, "sort", splitLayer);

        var tasks = Enumerable.Range(0, buckets).Select(b => TaskSpec.FromRemote(new TaskKey(name, b), new RemoteCall(Sort,
            new RemoteCall(Concat, new TaskRefList(pieces[b])), s.Columns, s.Ascending), "sort")).ToArray();
        return new Parts(new GraphLayer(name, tasks, [pickLayer]), tasks.Select(t => t.Key).ToArray());
    }

    /// <summary>n - 1 distinct boundary values at even quantiles of the sample.</summary>
    private static DataFrame Boundaries(DataFrame[] samples, string column, int n)
    {
        var nonEmpty = samples.Where(f => f.RowCount > 0).ToArray();
        if (nonEmpty.Length == 0) return samples[0];
        var all = DataFrame.Concat(nonEmpty);
        var values = Enumerable.Range(0, all.RowCount).Select(i => all[column].GetValue(i)).ToArray();
        var order = Enumerable.Range(0, values.Length).ToArray();
        Array.Sort(order, (a, b) => Grouping.CompareValues(values[a], values[b]));
        var picks = new List<int>();
        object? last = null;
        for (var q = 1; q < n; q++)
        {
            var idx = order[Math.Min(order.Length - 1, q * order.Length / n)];
            if (picks.Count > 0 && Grouping.CompareValues(values[idx], last) == 0) continue;
            picks.Add(idx);
            last = values[idx];
        }
        return all.SelectColumns(column).Take(picks);
    }

    private Parts LowerRepartition(RepartitionNode r)
    {
        var input = Lower(r.Input);
        var m = input.Keys.Length;
        var n = r.Partitions;
        var name = TaskKey.NewName("repartition");
        if (n == m) return input;

        if (n < m)
        {
            // Merge runs of adjacent partitions, keeping order.
            var tasks = Enumerable.Range(0, n).Select(j =>
            {
                var (start, count) = Bag.Slice(m, n, j);
                return TaskSpec.Call(new TaskKey(name, j), Concat, [new TaskRefList(input.Keys.Skip(start).Take(count).ToArray())], "repartition");
            }).ToArray();
            return new Parts(new GraphLayer(name, tasks, [input.Layer]), tasks.Select(t => t.Key).ToArray());
        }

        // Split each partition into as many pieces as its share of the output.
        var splitName = TaskKey.NewName("repartition-split");
        var splits = new List<TaskSpec>();
        var outputs = new List<TaskSpec>();
        for (var i = 0; i < m; i++)
        {
            var (_, pieces) = Bag.Slice(n, m, i);
            var split = TaskSpec.Call(new TaskKey(splitName, i), Split, [new TaskRef(input.Keys[i]), pieces], "repartition-split");
            splits.Add(split);
            for (var p = 0; p < pieces; p++)
                outputs.Add(TaskSpec.Call(new TaskKey(name, outputs.Count), Pick, [new TaskRef(split.Key), p], "repartition"));
        }
        var splitLayer = new GraphLayer(splitName, splits, [input.Layer]);
        return new Parts(new GraphLayer(name, outputs, [splitLayer]), outputs.Select(t => t.Key).ToArray());
    }

    private Parts LowerConcat(ConcatNode c)
    {
        var inputs = c.Inputs.Select(Lower).ToArray();
        var name = TaskKey.NewName("concat");
        var keys = inputs.SelectMany(i => i.Keys).ToArray();
        var tasks = keys.Select((k, i) => TaskSpec.FromRemote(new TaskKey(name, i),
            new RemoteCall(Builtins.IdentityMethod, new TaskRef(k)), "concat")).ToArray();
        return new Parts(new GraphLayer(name, tasks, inputs.Select(i => i.Layer)), tasks.Select(t => t.Key).ToArray());
    }

    private Parts LowerWindow(WindowNode w)
    {
        var input = Lower(w.Input);
        var spec = w.Spec.Encode();
        var name = TaskKey.NewName("window");
        if (w.Spec.PartitionBy.Length == 0 || input.Keys.Length == 1)
        {
            var single = TaskSpec.FromRemote(new TaskKey(name, 0), new RemoteCall(WindowKernel,
                new RemoteCall(Concat, new TaskRefList(input.Keys)), spec), "window");
            return new Parts(new GraphLayer(name, [single], [input.Layer]), [single.Key]);
        }
        var n = input.Keys.Length;
        var (buckets, layer) = Shuffle(input, w.Spec.PartitionBy, n, "window-shuffle");
        var tasks = Enumerable.Range(0, n).Select(b => TaskSpec.FromRemote(new TaskKey(name, b), new RemoteCall(WindowKernel,
            new RemoteCall(Concat, new TaskRefList(buckets[b])), spec), "window")).ToArray();
        return new Parts(new GraphLayer(name, tasks, [layer]), tasks.Select(t => t.Key).ToArray());
    }

    private Parts LowerLimit(LimitNode l)
    {
        var input = Lower(l.Input);
        var headName = TaskKey.NewName("limit-partial");
        var heads = input.Keys.Select((k, i) => TaskSpec.Call(new TaskKey(headName, i), Head, [new TaskRef(k), l.Count], "limit-partial")).ToArray();
        var name = TaskKey.NewName("limit");
        var final = TaskSpec.FromRemote(new TaskKey(name, 0), new RemoteCall(Head,
            new RemoteCall(Concat, new TaskRefList(heads.Select(h => h.Key).ToArray())), l.Count), "limit");
        var headLayer = new GraphLayer(headName, heads, [input.Layer]);
        return new Parts(new GraphLayer(name, [final], [headLayer]), [final.Key]);
    }
}
