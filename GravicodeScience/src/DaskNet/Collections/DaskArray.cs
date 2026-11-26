using System.Reflection;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.DaskNet;

/// <summary>
/// A lazy n-dimensional array cut into blocks (Dask's <c>dask.array</c>): each block is an
/// <see cref="NdArray"/> computed by one task, so arrays larger than memory can be reduced,
/// combined and multiplied block by block.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="Chunks"/> lists, per axis, the extent of each block along it; blocks are numbered
/// in C order over that grid. Element-wise operations need matching chunks and rechunk the right
/// operand when they differ. Reductions build a tree whose combine order is fixed by the graph,
/// so a result does not depend on which task finished first; it does depend on the chunking, as
/// any floating-point sum does on its grouping.
/// </para>
/// <para>
/// Every operation here except <see cref="MapBlocks"/> uses <see cref="ArrayKernels"/>, so array
/// graphs run on a cluster as well as on threads.
/// </para>
/// </remarks>
public sealed class DaskArray : IDaskCollection
{
    private readonly string _name;

    internal DaskArray(GraphLayer layer, string name, int[] shape, int[][] chunks)
    {
        Layer = layer;
        _name = name;
        Shape = shape;
        Chunks = chunks;
        OutputKeys = Enumerable.Range(0, BlockCount).Select(i => new TaskKey(name, i)).ToArray();
    }

    /// <inheritdoc />
    public GraphLayer Layer { get; }

    /// <inheritdoc />
    public IReadOnlyList<TaskKey> OutputKeys { get; }

    /// <summary>The array's shape.</summary>
    public int[] Shape { get; }

    /// <summary>Per axis, the extent of each block along it.</summary>
    public int[][] Chunks { get; }

    /// <summary>The number of axes.</summary>
    public int Rank => Shape.Length;

    /// <summary>The number of elements.</summary>
    public long Size => Shape.Aggregate(1L, (a, b) => a * b);

    /// <summary>Blocks per axis.</summary>
    public int[] BlockGrid => Chunks.Select(c => c.Length).ToArray();

    /// <summary>The total number of blocks.</summary>
    public int BlockCount => Chunks.Aggregate(1, (a, c) => a * c.Length);

    private static readonly MethodInfo FullMethod = Kernel(nameof(ArrayKernels.Full));
    private static readonly MethodInfo RandomMethod = Kernel(nameof(ArrayKernels.Random));
    private static readonly MethodInfo ArangeMethod = Kernel(nameof(ArrayKernels.Arange));
    private static readonly MethodInfo BinaryMethod = Kernel(nameof(ArrayKernels.Binary));
    private static readonly MethodInfo ScalarMethod = Kernel(nameof(ArrayKernels.Scalar));
    private static readonly MethodInfo UnaryMethod = Kernel(nameof(ArrayKernels.Unary));
    private static readonly MethodInfo MatMulMethod = Kernel(nameof(ArrayKernels.MatMul));
    private static readonly MethodInfo SumBlocksMethod = Kernel(nameof(ArrayKernels.SumBlocks));
    private static readonly MethodInfo TransposeMethod = Kernel(nameof(ArrayKernels.Transpose));
    private static readonly MethodInfo ExpandDimsMethod = Kernel(nameof(ArrayKernels.ExpandDims));
    private static readonly MethodInfo AssembleMethod = Kernel(nameof(ArrayKernels.Assemble));
    private static readonly MethodInfo ReduceMethod = Kernel(nameof(ArrayKernels.Reduce));
    private static readonly MethodInfo CombineReduceMethod = Kernel(nameof(ArrayKernels.CombineReduce));
    private static readonly MethodInfo FinishReduceMethod = Kernel(nameof(ArrayKernels.FinishReduce));
    private static readonly MethodInfo ReduceAxisMethod = Kernel(nameof(ArrayKernels.ReduceAxis));
    private static readonly MethodInfo CombineAxisMethod = Kernel(nameof(ArrayKernels.CombineAxis));
    private static readonly MethodInfo FinishMomentsMethod = Kernel(nameof(ArrayKernels.FinishMoments));

    private static MethodInfo Kernel(string name) => typeof(ArrayKernels).GetMethod(name)!;

    // ------------------------------------------------------------------ creation

    /// <summary>Splits an in-memory array into blocks of at most <paramref name="chunkSize"/> per axis
    /// (one value for every axis, or one per axis).</summary>
    public static DaskArray FromArray(NdArray array, params int[] chunkSize)
    {
        var shape = array.Shape.ToArray();
        var chunks = NormaliseChunks(shape, chunkSize);
        var name = TaskKey.NewName("array");
        var grid = chunks.Select(c => c.Length).ToArray();
        var tasks = new TaskSpec[Shapes.Size(grid)];
        for (var b = 0; b < tasks.Length; b++)
        {
            var (start, size) = BlockExtent(chunks, Unravel(b, grid));
            var block = array.Slice(start.Select((s, d) => Slice.Range(s, s + size[d])).ToArray()).Copy();
            tasks[b] = TaskSpec.Value(new TaskKey(name, b), block, "array");
        }
        return new DaskArray(new GraphLayer(name, tasks), name, shape, chunks);
    }

    /// <summary>An array of <paramref name="value"/>, generated inside the tasks.</summary>
    public static DaskArray Full(double value, int[] shape, params int[] chunkSize)
        => Generate("full", shape, chunkSize, (_, s) => [s, value], FullMethod);

    /// <summary>An array of zeros.</summary>
    public static DaskArray Zeros(int[] shape, params int[] chunkSize) => Full(0.0, shape, chunkSize);

    /// <summary>An array of ones.</summary>
    public static DaskArray Ones(int[] shape, params int[] chunkSize) => Full(1.0, shape, chunkSize);

    /// <summary>Uniform [0, 1) random numbers; each block has its own stream derived from
    /// <paramref name="seed"/>, so the result does not depend on scheduling.</summary>
    public static DaskArray Random(int[] shape, int[] chunkSize, long seed = 42)
        => Generate("random", shape, chunkSize, (b, s) => [s, seed, b, false], RandomMethod);

    /// <summary>Standard normal random numbers, seeded per block.</summary>
    public static DaskArray RandomNormal(int[] shape, int[] chunkSize, long seed = 42)
        => Generate("random-normal", shape, chunkSize, (b, s) => [s, seed, b, true], RandomMethod);

    /// <summary><c>0, 1, ..., count - 1</c> in blocks of <paramref name="chunkSize"/>.</summary>
    public static DaskArray Arange(int count, int chunkSize)
    {
        var chunks = NormaliseChunks([count], [chunkSize]);
        var name = TaskKey.NewName("arange");
        var offset = 0;
        var tasks = new TaskSpec[chunks[0].Length];
        for (var b = 0; b < tasks.Length; b++)
        {
            tasks[b] = TaskSpec.Call(new TaskKey(name, b), ArangeMethod, [(double)offset, 1.0, chunks[0][b]], "arange");
            offset += chunks[0][b];
        }
        return new DaskArray(new GraphLayer(name, tasks), name, [count], chunks);
    }

    private static DaskArray Generate(string label, int[] shape, int[] chunkSize, Func<int, int[], object?[]> args, MethodInfo method)
    {
        var chunks = NormaliseChunks(shape, chunkSize);
        var name = TaskKey.NewName(label);
        var grid = chunks.Select(c => c.Length).ToArray();
        var tasks = new TaskSpec[Shapes.Size(grid)];
        for (var b = 0; b < tasks.Length; b++)
        {
            var (_, size) = BlockExtent(chunks, Unravel(b, grid));
            tasks[b] = TaskSpec.Call(new TaskKey(name, b), method, args(b, size), label);
        }
        return new DaskArray(new GraphLayer(name, tasks), name, shape.ToArray(), chunks);
    }

    // ------------------------------------------------------------------ element-wise

    /// <summary>
    /// Element-wise combination with another array, broadcasting as NumPy does: shapes are aligned
    /// from the right and an axis of length 1 (or a missing leading axis) stretches to match, so
    /// <c>x - x.Mean(axis: 0)</c> centres every column.
    /// </summary>
    /// <remarks>
    /// The output takes its chunking from whichever operand spans each axis (this one when both
    /// do; the other is rechunked to it). A broadcast operand is not copied per block: each output
    /// block reads block 0 of it along the stretched axes, and GraviNum broadcasts inside the kernel.
    /// </remarks>
    public DaskArray Elementwise(DaskArray other, BinaryOp op)
    {
        if (Shape.SequenceEqual(other.Shape))
        {
            var same = SameChunks(other) ? other : other.RechunkTo(Chunks);
            return BlockWise($"{op}".ToLowerInvariant(), [Layer, same.Layer], b =>
                new RemoteCall(BinaryMethod, new TaskRef(OutputKeys[b]), new TaskRef(same.OutputKeys[b]), (int)op));
        }

        var rank = Math.Max(Rank, other.Rank);
        var shape = new int[rank];
        var chunks = new int[rank][];
        for (var ax = 0; ax < rank; ax++)
        {
            int ia = ax - (rank - Rank), ib = ax - (rank - other.Rank);
            int da = ia >= 0 ? Shape[ia] : 1, db = ib >= 0 ? other.Shape[ib] : 1;
            if (da != db && da != 1 && db != 1)
                throw new ArgumentException(
                    $"Shapes [{string.Join(", ", Shape)}] and [{string.Join(", ", other.Shape)}] cannot be broadcast together (axis {ax}: {da} vs {db}).");
            shape[ax] = Math.Max(da, db);
            chunks[ax] = da == shape[ax] && ia >= 0 ? Chunks[ia] : db == shape[ax] && ib >= 0 ? other.Chunks[ib] : [shape[ax]];
        }

        // Each operand keeps one block along a stretched axis and takes the output's chunks elsewhere.
        DaskArray Fit(DaskArray x) => x.SameChunks(Target(x)) ? x : x.RechunkTo(Target(x));
        int[][] Target(DaskArray x) => Enumerable.Range(0, x.Rank).Select(k =>
        {
            var ax = k + rank - x.Rank;
            return x.Shape[k] == 1 && shape[ax] != 1 ? [1] : chunks[ax];
        }).ToArray();
        var left = Fit(this);
        var right = Fit(other);

        var grid = chunks.Select(c => c.Length).ToArray();
        int Source(DaskArray x, int[] coords)
        {
            var own = Enumerable.Range(0, x.Rank).Select(k => x.BlockGrid[k] == 1 ? 0 : coords[k + rank - x.Rank]).ToArray();
            return Ravel(own, x.BlockGrid);
        }
        var label = $"{op}".ToLowerInvariant();
        var name = TaskKey.NewName(label);
        var tasks = new TaskSpec[grid.Aggregate(1, (p, g) => p * g)];
        for (var b = 0; b < tasks.Length; b++)
        {
            var coords = Unravel(b, grid);
            var call = new RemoteCall(BinaryMethod, new TaskRef(left.OutputKeys[Source(left, coords)]),
                new TaskRef(right.OutputKeys[Source(right, coords)]), (int)op);
            tasks[b] = TaskSpec.FromRemote(new TaskKey(name, b), call, label);
        }
        return new DaskArray(new GraphLayer(name, tasks, [left.Layer, right.Layer]), name, shape, chunks);
    }

    /// <summary>Element-wise combination with a scalar.</summary>
    public DaskArray Elementwise(double scalar, BinaryOp op, bool scalarFirst = false)
        => BlockWise($"{op}".ToLowerInvariant(), [Layer], b =>
            new RemoteCall(ScalarMethod, new TaskRef(OutputKeys[b]), scalar, (int)op, scalarFirst));

    /// <summary>A unary function of every element.</summary>
    public DaskArray Apply(UnaryOp op)
        => BlockWise($"{op}".ToLowerInvariant(), [Layer], b => new RemoteCall(UnaryMethod, new TaskRef(OutputKeys[b]), (int)op));

    /// <summary>e^x</summary>
    public DaskArray Exp() => Apply(UnaryOp.Exp);
    /// <summary>ln x</summary>
    public DaskArray Log() => Apply(UnaryOp.Log);
    /// <summary>√x</summary>
    public DaskArray Sqrt() => Apply(UnaryOp.Sqrt);
    /// <summary>|x|</summary>
    public DaskArray Abs() => Apply(UnaryOp.Abs);
    /// <summary>x²</summary>
    public DaskArray Square() => Apply(UnaryOp.Square);
    /// <summary>sin x</summary>
    public DaskArray Sin() => Apply(UnaryOp.Sin);
    /// <summary>cos x</summary>
    public DaskArray Cos() => Apply(UnaryOp.Cos);
    /// <summary>tanh x</summary>
    public DaskArray Tanh() => Apply(UnaryOp.Tanh);
    /// <summary>max(a, b) element-wise.</summary>
    public DaskArray Maximum(DaskArray other) => Elementwise(other, BinaryOp.Maximum);
    /// <summary>min(a, b) element-wise.</summary>
    public DaskArray Minimum(DaskArray other) => Elementwise(other, BinaryOp.Minimum);
    /// <summary>a ^ p element-wise.</summary>
    public DaskArray Power(double p) => Elementwise(p, BinaryOp.Power);

    /// <summary>
    /// Applies a function to every block (Dask's <c>map_blocks</c>). The function must keep each
    /// block's shape. Runs locally only: a delegate cannot be shipped to a cluster.
    /// </summary>
    public DaskArray MapBlocks(Func<NdArray, NdArray> function)
    {
        ArgumentNullException.ThrowIfNull(function);
        var name = TaskKey.NewName("map-blocks");
        var tasks = OutputKeys.Select((k, b) => TaskSpec.Create(new TaskKey(name, b), [k], v =>
        {
            var input = (NdArray)v[0]!;
            var output = function(input);
            if (!output.Shape.SequenceEqual(input.Shape))
                throw new InvalidOperationException("map_blocks functions must keep the block shape.");
            return output;
        }, "map-blocks")).ToArray();
        return new DaskArray(new GraphLayer(name, tasks, [Layer]), name, Shape, Chunks);
    }

    /// <summary>Applies a scalar function to every element. Runs locally only.</summary>
    public DaskArray Map(Func<double, double> function) => MapBlocks(b => UFunc.Unary(b, function));

    private DaskArray BlockWise(string label, GraphLayer[] parents, Func<int, RemoteCall> call)
    {
        var name = TaskKey.NewName(label);
        var tasks = Enumerable.Range(0, BlockCount)
            .Select(b => TaskSpec.FromRemote(new TaskKey(name, b), call(b), label)).ToArray();
        return new DaskArray(new GraphLayer(name, tasks, parents), name, Shape, Chunks);
    }

    /// <summary>a + b</summary>
    public static DaskArray operator +(DaskArray a, DaskArray b) => a.Elementwise(b, BinaryOp.Add);
    /// <summary>a - b</summary>
    public static DaskArray operator -(DaskArray a, DaskArray b) => a.Elementwise(b, BinaryOp.Subtract);
    /// <summary>a * b</summary>
    public static DaskArray operator *(DaskArray a, DaskArray b) => a.Elementwise(b, BinaryOp.Multiply);
    /// <summary>a / b</summary>
    public static DaskArray operator /(DaskArray a, DaskArray b) => a.Elementwise(b, BinaryOp.Divide);
    /// <summary>a + s</summary>
    public static DaskArray operator +(DaskArray a, double s) => a.Elementwise(s, BinaryOp.Add);
    /// <summary>s + a</summary>
    public static DaskArray operator +(double s, DaskArray a) => a.Elementwise(s, BinaryOp.Add, scalarFirst: true);
    /// <summary>a - s</summary>
    public static DaskArray operator -(DaskArray a, double s) => a.Elementwise(s, BinaryOp.Subtract);
    /// <summary>s - a</summary>
    public static DaskArray operator -(double s, DaskArray a) => a.Elementwise(s, BinaryOp.Subtract, scalarFirst: true);
    /// <summary>a * s</summary>
    public static DaskArray operator *(DaskArray a, double s) => a.Elementwise(s, BinaryOp.Multiply);
    /// <summary>s * a</summary>
    public static DaskArray operator *(double s, DaskArray a) => a.Elementwise(s, BinaryOp.Multiply, scalarFirst: true);
    /// <summary>a / s</summary>
    public static DaskArray operator /(DaskArray a, double s) => a.Elementwise(s, BinaryOp.Divide);
    /// <summary>s / a</summary>
    public static DaskArray operator /(double s, DaskArray a) => a.Elementwise(s, BinaryOp.Divide, scalarFirst: true);
    /// <summary>-a</summary>
    public static DaskArray operator -(DaskArray a) => a.Apply(UnaryOp.Negate);

    // ------------------------------------------------------------------ reductions

    /// <summary>The sum of every element.</summary>
    public Delayed<double> Sum() => FullReduce(ArrayKernels.SumCode, ArrayKernels.FinishFirst, 0, "sum");

    /// <summary>The smallest element.</summary>
    public Delayed<double> Min() => FullReduce(ArrayKernels.MinCode, ArrayKernels.FinishFirst, 0, "min");

    /// <summary>The largest element.</summary>
    public Delayed<double> Max() => FullReduce(ArrayKernels.MaxCode, ArrayKernels.FinishFirst, 0, "max");

    /// <summary>The mean of every element.</summary>
    public Delayed<double> Mean() => FullReduce(ArrayKernels.MomentsCode, ArrayKernels.FinishMean, 0, "mean");

    /// <summary>The variance of every element (population by default, like NumPy).</summary>
    public Delayed<double> Var(int ddof = 0) => FullReduce(ArrayKernels.MomentsCode, ArrayKernels.FinishVar, ddof, "var");

    /// <summary>The standard deviation of every element.</summary>
    public Delayed<double> Std(int ddof = 0) => FullReduce(ArrayKernels.MomentsCode, ArrayKernels.FinishStd, ddof, "std");

    private Delayed<double> FullReduce(int how, int finish, int ddof, string label)
    {
        var splitEvery = SplitEvery;
        var partialName = TaskKey.NewName(label + "-partial");
        var partials = OutputKeys.Select((k, b) =>
            TaskSpec.Call(new TaskKey(partialName, b), ReduceMethod, [new TaskRef(k), how], label + "-partial")).ToArray();
        var treeName = TaskKey.NewName(label + "-aggregate");
        var (tree, root) = GraphBuilders.TreeReduce(treeName, partials.Select(p => p.Key).ToArray(), splitEvery,
            (key, inputs) => TaskSpec.Call(key, CombineReduceMethod, [new TaskRefList(inputs), how], label + "-aggregate"));
        var partialLayer = new GraphLayer(partialName, partials, [Layer]);
        var treeLayer = new GraphLayer(treeName, tree, [partialLayer]);
        var finalKey = new TaskKey(TaskKey.NewName(label));
        var final = TaskSpec.Call(finalKey, FinishReduceMethod, [new TaskRef(root), finish, ddof], label);
        return new Delayed<double>(new GraphLayer(finalKey.Name, [final], [treeLayer]), finalKey);
    }

    /// <summary>The sum along one axis.</summary>
    public DaskArray Sum(int axis) => AxisReduce(axis, ArrayKernels.SumCode, "sum", null);

    /// <summary>The minimum along one axis.</summary>
    public DaskArray Min(int axis) => AxisReduce(axis, ArrayKernels.MinCode, "min", null);

    /// <summary>The maximum along one axis.</summary>
    public DaskArray Max(int axis) => AxisReduce(axis, ArrayKernels.MaxCode, "max", null);

    /// <summary>The mean along one axis.</summary>
    public DaskArray Mean(int axis) => AxisReduce(axis, ArrayKernels.MomentsCode, "mean", -1);

    /// <summary>The variance along one axis. Both arguments are required, so that <c>Var(1)</c>
    /// cannot be read as an axis when it is a <c>ddof</c> (or the reverse).</summary>
    public DaskArray Var(int axis, int ddof) => AxisReduce(axis, ArrayKernels.MomentsCode, "var", ddof);

    /// <summary>The standard deviation along one axis; see <see cref="Var(int, int)"/>.</summary>
    public DaskArray Std(int axis, int ddof) => Var(axis, ddof).Sqrt();

    /// <summary>How many partial results each node of a reduction tree combines (Dask's
    /// <c>split_every</c>). Larger is shallower; smaller holds less per task.</summary>
    public int SplitEvery { get; private init; } = 8;

    /// <summary>The same array, reduced with trees of <paramref name="splitEvery"/> inputs per node.</summary>
    public DaskArray WithSplitEvery(int splitEvery)
        => splitEvery >= 2
            ? new DaskArray(Layer, _name, Shape, Chunks) { SplitEvery = splitEvery }
            : throw new ArgumentOutOfRangeException(nameof(splitEvery), "split_every must be at least 2.");

    private DaskArray AxisReduce(int axis, int how, string label, int? finishDdof)
    {
        var splitEvery = SplitEvery;
        if (axis < 0) axis += Rank;
        if (axis < 0 || axis >= Rank) throw new ArgumentOutOfRangeException(nameof(axis));

        var grid = BlockGrid;
        var partialName = TaskKey.NewName(label + "-partial");
        var partials = OutputKeys.Select((k, b) =>
            TaskSpec.Call(new TaskKey(partialName, b), ReduceAxisMethod, [new TaskRef(k), axis, how], label + "-partial")).ToArray();
        var partialLayer = new GraphLayer(partialName, partials, [Layer]);

        var outChunks = Chunks.Where((_, d) => d != axis).ToArray();
        var outShape = Shape.Where((_, d) => d != axis).ToArray();
        if (outShape.Length == 0) { outShape = [1]; outChunks = [[1]]; }
        var outGrid = outChunks.Select(c => c.Length).ToArray();
        var outCount = Shapes.Size(outGrid);

        var treeName = TaskKey.NewName(label + "-aggregate");
        var finalName = TaskKey.NewName(label);
        var tree = new List<TaskSpec>();
        var finals = new List<TaskSpec>();
        var treeIndex = 0;

        for (var o = 0; o < outCount; o++)
        {
            var outCoords = Rank == 1 ? [] : Unravel(o, outGrid);
            var leaves = new List<TaskKey>();
            for (var i = 0; i < grid[axis]; i++)
            {
                var coords = new int[Rank];
                for (int d = 0, od = 0; d < Rank; d++) coords[d] = d == axis ? i : outCoords[od++];
                leaves.Add(partials[Ravel(coords, grid)].Key);
            }

            // Each output block gets its own reduction tree; their keys share one layer.
            var (subtree, root) = GraphBuilders.TreeReduce(treeName, leaves, splitEvery,
                (key, inputs) => TaskSpec.Call(key, CombineAxisMethod, [new TaskRefList(inputs), how], label + "-aggregate"),
                () => new TaskKey(treeName, treeIndex++));
            tree.AddRange(subtree);

            var finalKey = new TaskKey(finalName, o);
            finals.Add(finishDdof is { } ddof
                ? TaskSpec.Call(finalKey, FinishMomentsMethod, [new TaskRef(root), ddof], label)
                : TaskSpec.FromRemote(finalKey, new RemoteCall(Builtins.IdentityMethod, new TaskRef(root)), label));
        }

        var treeLayer = new GraphLayer(treeName, tree, [partialLayer]);
        return new DaskArray(new GraphLayer(finalName, finals, [treeLayer]), finalName, outShape, outChunks);
    }

    // ------------------------------------------------------------------ linear algebra and layout

    /// <summary>
    /// The matrix product of two 2-D arrays, block by block: <c>C[i,j] = Σ_p A[i,p] B[p,j]</c>,
    /// each sum a tree over <c>p</c>. Rechunks <paramref name="other"/>'s rows to this array's
    /// column chunks when they differ.
    /// </summary>
    public DaskArray Dot(DaskArray other, int splitEvery = 8)
    {
        if (Rank != 2 || other.Rank != 2) throw new ArgumentException("Dot needs two 2-D arrays.");
        if (Shape[1] != other.Shape[0])
            throw new ArgumentException($"Cannot multiply {Shape[0]}x{Shape[1]} by {other.Shape[0]}x{other.Shape[1]}.");

        var right = other.Chunks[0].SequenceEqual(Chunks[1]) ? other : other.RechunkTo([Chunks[1], other.Chunks[1]]);
        var (mi, kp, nj) = (Chunks[0].Length, Chunks[1].Length, right.Chunks[1].Length);

        var productName = TaskKey.NewName("matmul-block");
        var products = new List<TaskSpec>();
        var treeName = TaskKey.NewName("matmul-sum");
        var tree = new List<TaskSpec>();
        var finalName = TaskKey.NewName("matmul");
        var finals = new List<TaskSpec>();
        var treeIndex = 0;

        for (var i = 0; i < mi; i++)
            for (var j = 0; j < nj; j++)
            {
                var leaves = new List<TaskKey>();
                for (var p = 0; p < kp; p++)
                {
                    var key = new TaskKey(productName, (i * nj + j) * kp + p);
                    products.Add(TaskSpec.Call(key, MatMulMethod,
                        [new TaskRef(OutputKeys[i * kp + p]), new TaskRef(right.OutputKeys[p * nj + j])], "matmul-block"));
                    leaves.Add(key);
                }
                var (subtree, root) = GraphBuilders.TreeReduce(treeName, leaves, splitEvery,
                    (key, inputs) => TaskSpec.Call(key, SumBlocksMethod, [new TaskRefList(inputs)], "matmul-sum"),
                    () => new TaskKey(treeName, treeIndex++));
                tree.AddRange(subtree);
                finals.Add(TaskSpec.FromRemote(new TaskKey(finalName, i * nj + j),
                    new RemoteCall(Builtins.IdentityMethod, new TaskRef(root)), "matmul"));
            }

        var productLayer = new GraphLayer(productName, products, [Layer, right.Layer]);
        var treeLayer = new GraphLayer(treeName, tree, [productLayer]);
        return new DaskArray(new GraphLayer(finalName, finals, [treeLayer]), finalName,
            [Shape[0], right.Shape[1]], [Chunks[0], right.Chunks[1]]);
    }

    /// <summary>The array with its axes reversed (for 2-D, the transpose).</summary>
    public DaskArray Transpose()
    {
        var grid = BlockGrid;
        var outGrid = grid.Reverse().ToArray();
        var name = TaskKey.NewName("transpose");
        var tasks = new TaskSpec[BlockCount];
        for (var b = 0; b < tasks.Length; b++)
        {
            var source = Ravel(Unravel(b, outGrid).Reverse().ToArray(), grid);
            tasks[b] = TaskSpec.Call(new TaskKey(name, b), TransposeMethod, [new TaskRef(OutputKeys[source])], "transpose");
        }
        return new DaskArray(new GraphLayer(name, tasks, [Layer]), name,
            Shape.Reverse().ToArray(), Chunks.Reverse().ToArray());
    }

    /// <summary>The transpose.</summary>
    public DaskArray T => Transpose();

    /// <summary>
    /// A unit axis inserted at <paramref name="axis"/> (NumPy's <c>expand_dims</c>, or <c>[:, None]</c>),
    /// so a reduced array broadcasts back along the axis it lost: <c>x / x.Sum(axis: 1).ExpandDims(1)</c>.
    /// </summary>
    public DaskArray ExpandDims(int axis)
    {
        if (axis < 0) axis += Rank + 1;
        if (axis < 0 || axis > Rank) throw new ArgumentOutOfRangeException(nameof(axis));
        var name = TaskKey.NewName("expand-dims");
        var tasks = Enumerable.Range(0, BlockCount).Select(b => TaskSpec.FromRemote(new TaskKey(name, b),
            new RemoteCall(ExpandDimsMethod, new TaskRef(OutputKeys[b]), axis), "expand-dims")).ToArray();
        var shape = Shape.ToList();
        shape.Insert(axis, 1);
        var chunks = Chunks.ToList();
        chunks.Insert(axis, [1]);
        return new DaskArray(new GraphLayer(name, tasks, [Layer]), name, shape.ToArray(), chunks.ToArray());
    }

    /// <summary>The same values in blocks of at most <paramref name="chunkSize"/> per axis.</summary>
    public DaskArray Rechunk(params int[] chunkSize) => RechunkTo(NormaliseChunks(Shape, chunkSize));

    /// <summary>The same values with exactly the given chunks.</summary>
    public DaskArray RechunkTo(int[][] chunks)
    {
        if (chunks.Length != Rank) throw new ArgumentException("One chunk list per axis is needed.");
        for (var d = 0; d < Rank; d++)
            if (chunks[d].Sum() != Shape[d]) throw new ArgumentException($"Chunks along axis {d} do not add up to {Shape[d]}.");
        if (SameChunks(chunks)) return this;

        var grid = BlockGrid;
        var outGrid = chunks.Select(c => c.Length).ToArray();
        var name = TaskKey.NewName("rechunk");
        var tasks = new TaskSpec[Shapes.Size(outGrid)];
        for (var o = 0; o < tasks.Length; o++)
        {
            var (start, size) = BlockExtent(chunks, Unravel(o, outGrid));

            // The source blocks overlapping this target block, per axis then as a product.
            var ranges = new List<int>[Rank];
            for (var d = 0; d < Rank; d++)
            {
                ranges[d] = [];
                var offset = 0;
                for (var i = 0; i < Chunks[d].Length; i++)
                {
                    var end = offset + Chunks[d][i];
                    if (offset < start[d] + size[d] && end > start[d]) ranges[d].Add(i);
                    offset = end;
                }
            }

            var sources = new List<TaskKey>();
            var starts = new List<int>();
            foreach (var coords in CartesianProduct(ranges))
            {
                sources.Add(OutputKeys[Ravel(coords, grid)]);
                starts.AddRange(BlockExtent(Chunks, coords).Start);
            }
            tasks[o] = TaskSpec.Call(new TaskKey(name, o), AssembleMethod,
                [new TaskRefList(sources), starts.ToArray(), start, size], "rechunk");
        }
        return new DaskArray(new GraphLayer(name, tasks, [Layer]), name, Shape, chunks.Select(c => c.ToArray()).ToArray());
    }

    // ------------------------------------------------------------------ results

    /// <summary>Computes every block and assembles the array.</summary>
    public NdArray Compute(IScheduler? scheduler = null, CancellationToken cancellationToken = default)
        => Assemble(Dask.Run([this], scheduler, cancellationToken));

    /// <summary>Computes every block asynchronously and assembles the array.</summary>
    public async Task<NdArray> ComputeAsync(IScheduler? scheduler = null, CancellationToken cancellationToken = default)
        => Assemble(await Dask.RunAsync([this], scheduler, cancellationToken).ConfigureAwait(false));

    /// <summary>Computes now and returns an array over the in-memory blocks.</summary>
    public DaskArray Persist(IScheduler? scheduler = null)
    {
        var results = Dask.Run([this], scheduler, CancellationToken.None);
        var name = TaskKey.NewName("persisted");
        var tasks = OutputKeys.Select((k, b) => TaskSpec.Value(new TaskKey(name, b), results[k], "persisted")).ToArray();
        return new DaskArray(new GraphLayer(name, tasks), name, Shape, Chunks);
    }

    /// <summary>The blocks as delayed arrays, in C order over the block grid.</summary>
    public Delayed<NdArray>[] ToDelayed() => OutputKeys.Select(k => new Delayed<NdArray>(Layer, k)).ToArray();

    /// <summary>The graph that would run, as Graphviz DOT.</summary>
    public string Visualize(bool highLevel = false) => Dask.Graph(this).ToDot(highLevel);

    private NdArray Assemble(IReadOnlyDictionary<TaskKey, object?> results)
    {
        var result = NdArray.Zeros(Shape);
        var grid = BlockGrid;
        for (var b = 0; b < BlockCount; b++)
        {
            var (start, size) = BlockExtent(Chunks, Unravel(b, grid));
            var block = (NdArray)results[OutputKeys[b]]!;
            if (block.Size != Shapes.Size(size))
                throw new InvalidOperationException($"Block {b} has {block.Size} elements, expected {Shapes.Size(size)}.");
            result.Slice(start.Select((s, d) => Slice.Range(s, s + size[d])).ToArray()).Assign(block.Reshape(size));
        }
        return result;
    }

    /// <inheritdoc />
    public override string ToString()
        => $"DaskArray(shape=({string.Join(", ", Shape)}), chunks=({string.Join(", ", Chunks.Select(c => c.Length > 0 ? c.Max() : 0))}), blocks={BlockCount})";

    // ------------------------------------------------------------------ block geometry

    private bool SameChunks(DaskArray other) => SameChunks(other.Chunks);

    private bool SameChunks(int[][] chunks)
        => chunks.Length == Rank && Enumerable.Range(0, Rank).All(d => chunks[d].SequenceEqual(Chunks[d]));

    internal static int[][] NormaliseChunks(int[] shape, int[] chunkSize)
    {
        if (chunkSize.Length != 1 && chunkSize.Length != shape.Length)
            throw new ArgumentException("Give one chunk size, or one per axis.");
        var chunks = new int[shape.Length][];
        for (var d = 0; d < shape.Length; d++)
        {
            var size = chunkSize.Length == 1 ? chunkSize[0] : chunkSize[d];
            if (size <= 0) size = Math.Max(shape[d], 1);
            var list = new List<int>();
            for (var s = 0; s < shape[d]; s += size) list.Add(Math.Min(size, shape[d] - s));
            if (list.Count == 0) list.Add(0);
            chunks[d] = list.ToArray();
        }
        return chunks;
    }

    internal static int[] Unravel(int flat, int[] grid)
    {
        var coords = new int[grid.Length];
        for (var d = grid.Length - 1; d >= 0; d--)
        {
            coords[d] = flat % grid[d];
            flat /= grid[d];
        }
        return coords;
    }

    internal static int Ravel(int[] coords, int[] grid)
    {
        var flat = 0;
        for (var d = 0; d < grid.Length; d++) flat = flat * grid[d] + coords[d];
        return flat;
    }

    internal static (int[] Start, int[] Size) BlockExtent(int[][] chunks, int[] coords)
    {
        var start = new int[chunks.Length];
        var size = new int[chunks.Length];
        for (var d = 0; d < chunks.Length; d++)
        {
            for (var i = 0; i < coords[d]; i++) start[d] += chunks[d][i];
            size[d] = chunks[d][coords[d]];
        }
        return (start, size);
    }

    private static IEnumerable<int[]> CartesianProduct(List<int>[] ranges)
    {
        var index = new int[ranges.Length];
        while (true)
        {
            yield return index.Select((i, d) => ranges[d][i]).ToArray();
            var axis = ranges.Length - 1;
            while (axis >= 0 && ++index[axis] == ranges[axis].Count) index[axis--] = 0;
            if (axis < 0) yield break;
        }
    }
}
