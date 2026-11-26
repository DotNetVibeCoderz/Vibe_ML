using System.ComponentModel;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.DaskNet;

/// <summary>Element-wise binary operations on array blocks.</summary>
public enum BinaryOp
{
    /// <summary>a + b</summary>
    Add,
    /// <summary>a - b</summary>
    Subtract,
    /// <summary>a * b</summary>
    Multiply,
    /// <summary>a / b</summary>
    Divide,
    /// <summary>a ^ b</summary>
    Power,
    /// <summary>max(a, b)</summary>
    Maximum,
    /// <summary>min(a, b)</summary>
    Minimum,
}

/// <summary>Element-wise unary operations on array blocks.</summary>
public enum UnaryOp
{
    /// <summary>-x</summary>
    Negate,
    /// <summary>|x|</summary>
    Abs,
    /// <summary>e^x</summary>
    Exp,
    /// <summary>ln x</summary>
    Log,
    /// <summary>√x</summary>
    Sqrt,
    /// <summary>x²</summary>
    Square,
    /// <summary>sin x</summary>
    Sin,
    /// <summary>cos x</summary>
    Cos,
    /// <summary>tanh x</summary>
    Tanh,
}

/// <summary>
/// The block-level kernels of <see cref="DaskArray"/>. Public and <see cref="RemoteTaskAttribute"/>
/// so a cluster worker can run them; not meant to be called directly.
/// </summary>
/// <remarks>Operation codes are <c>int</c> rather than the enums because they cross the wire as
/// plain integers.</remarks>
[EditorBrowsable(EditorBrowsableState.Never)]
public static class ArrayKernels
{
    /// <summary>One block of a constant array.</summary>
    [RemoteTask]
    public static NdArray Full(int[] shape, double value) => NdArray.Full(value, shape);

    /// <summary>One block of uniform or normal random numbers, seeded by block.</summary>
    [RemoteTask]
    public static NdArray Random(int[] shape, long seed, int blockIndex, bool normal)
    {
        // SplitMix-style mixing: neighbouring blocks must not get correlated streams from
        // neighbouring seeds.
        var mixed = (ulong)seed * 0x9E3779B97F4A7C15UL ^ (ulong)(blockIndex + 1) * 0xBF58476D1CE4E5B9UL;
        var rng = new GraviRandom(mixed);
        var data = new double[Shapes.Size(shape)];
        for (var i = 0; i < data.Length; i++) data[i] = normal ? rng.Normal() : rng.NextDouble();
        return new NdArray(data, shape);
    }

    /// <summary>One block of <c>start, start + step, ...</c>.</summary>
    [RemoteTask]
    public static NdArray Arange(double start, double step, int count)
    {
        var data = new double[count];
        for (var i = 0; i < count; i++) data[i] = start + i * step;
        return new NdArray(data, count);
    }

    /// <summary>Two blocks combined element-wise.</summary>
    [RemoteTask]
    public static NdArray Binary(NdArray a, NdArray b, int op) => (BinaryOp)op switch
    {
        BinaryOp.Add => a + b,
        BinaryOp.Subtract => a - b,
        BinaryOp.Multiply => a * b,
        BinaryOp.Divide => a / b,
        BinaryOp.Power => UFunc.Power(a, b),
        BinaryOp.Maximum => UFunc.Maximum(a, b),
        BinaryOp.Minimum => UFunc.Minimum(a, b),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    /// <summary>A block combined with a scalar; <paramref name="scalarFirst"/> puts the scalar on the left.</summary>
    [RemoteTask]
    public static NdArray Scalar(NdArray a, double s, int op, bool scalarFirst)
    {
        var b = NdArray.Full(s, a.Shape.ToArray());
        return scalarFirst ? Binary(b, a, op) : Binary(a, b, op);
    }

    /// <summary>A unary function of every element.</summary>
    [RemoteTask]
    public static NdArray Unary(NdArray a, int op) => (UnaryOp)op switch
    {
        UnaryOp.Negate => UFunc.Negate(a),
        UnaryOp.Abs => UFunc.Abs(a),
        UnaryOp.Exp => UFunc.Exp(a),
        UnaryOp.Log => UFunc.Log(a),
        UnaryOp.Sqrt => UFunc.Sqrt(a),
        UnaryOp.Square => UFunc.Square(a),
        UnaryOp.Sin => UFunc.Sin(a),
        UnaryOp.Cos => UFunc.Cos(a),
        UnaryOp.Tanh => UFunc.Tanh(a),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    /// <summary>A block's matrix product.</summary>
    [RemoteTask]
    public static NdArray MatMul(NdArray a, NdArray b) => LinAlg.Dot(a, b);

    /// <summary>The element-wise sum of several equal-shape blocks, added in order.</summary>
    [RemoteTask]
    public static NdArray SumBlocks(NdArray[] blocks)
    {
        var total = blocks[0].Copy();
        for (var i = 1; i < blocks.Length; i++) total = total + blocks[i];
        return total;
    }

    /// <summary>A block transposed (axes reversed).</summary>
    [RemoteTask]
    public static NdArray Transpose(NdArray a) => a.Transpose().Copy();

    /// <summary>
    /// Assembles a target block from the source blocks that overlap it.
    /// </summary>
    /// <param name="blocks">The overlapping source blocks.</param>
    /// <param name="blockStarts">Each block's origin in the full array, flattened (rank values per block).</param>
    /// <param name="targetStart">The target block's origin.</param>
    /// <param name="targetShape">The target block's shape.</param>
    [RemoteTask]
    public static NdArray Assemble(NdArray[] blocks, int[] blockStarts, int[] targetStart, int[] targetShape)
    {
        var rank = targetShape.Length;
        var result = NdArray.Zeros(targetShape);
        for (var b = 0; b < blocks.Length; b++)
        {
            var block = blocks[b];
            var srcSel = new Slice[rank];
            var dstSel = new Slice[rank];
            for (var d = 0; d < rank; d++)
            {
                var blockStart = blockStarts[b * rank + d];
                var lo = Math.Max(blockStart, targetStart[d]);
                var hi = Math.Min(blockStart + block.Shape[d], targetStart[d] + targetShape[d]);
                srcSel[d] = Slice.Range(lo - blockStart, hi - blockStart);
                dstSel[d] = Slice.Range(lo - targetStart[d], hi - targetStart[d]);
            }
            result.Slice(dstSel).Assign(block.Slice(srcSel));
        }
        return result;
    }

    // ------------------------------------------------------------------ reductions

    /// <summary>Codes for <see cref="Reduce"/> and <see cref="ReduceAxis"/>.</summary>
    internal const int SumCode = 0, MinCode = 1, MaxCode = 2, MomentsCode = 3;

    /// <summary>
    /// A whole-block summary: <c>[sum]</c>, <c>[min]</c>, <c>[max]</c> or
    /// <c>[count, mean, m2]</c>.
    /// </summary>
    [RemoteTask]
    public static double[] Reduce(NdArray block, int how)
    {
        var data = block.ToArray();
        return how switch
        {
            SumCode => [KahanSum(data)],
            MinCode => [data.Length == 0 ? double.PositiveInfinity : data.Min()],
            MaxCode => [data.Length == 0 ? double.NegativeInfinity : data.Max()],
            MomentsCode => MomentsArray(Moments.Of(data, skipMissing: false)),
            _ => throw new ArgumentOutOfRangeException(nameof(how)),
        };
    }

    /// <summary>Merges whole-block summaries produced by <see cref="Reduce"/>, in order.</summary>
    [RemoteTask]
    public static double[] CombineReduce(double[][] parts, int how) => how switch
    {
        SumCode => [KahanSum(parts.Select(p => p[0]).ToArray())],
        MinCode => [parts.Min(p => p[0])],
        MaxCode => [parts.Max(p => p[0])],
        MomentsCode => MomentsArray(parts.Select(FromArray).Aggregate(Moments.Merge)),
        _ => throw new ArgumentOutOfRangeException(nameof(how)),
    };

    internal const int FinishFirst = 0, FinishMean = 1, FinishVar = 2, FinishStd = 3;

    /// <summary>Turns a combined whole-array summary into the answer.</summary>
    [RemoteTask]
    public static double FinishReduce(double[] partial, int finish, int ddof) => finish switch
    {
        FinishFirst => partial[0],
        FinishMean => partial[0] == 0 ? double.NaN : partial[1],
        FinishVar => FromArray(partial).Variance(ddof),
        FinishStd => Math.Sqrt(FromArray(partial).Variance(ddof)),
        _ => throw new ArgumentOutOfRangeException(nameof(finish)),
    };

    /// <summary>
    /// A block reduced along one axis. For sums, minima and maxima the result has the axis
    /// removed; for moments it gains a leading axis of length 3 holding count, mean and m2.
    /// </summary>
    [RemoteTask]
    public static NdArray ReduceAxis(NdArray block, int axis, int how)
    {
        var (rows, length, outShape) = MoveAxisLast(block, axis, out var data);
        if (how == MomentsCode)
        {
            var result = new double[3 * rows];
            for (var r = 0; r < rows; r++)
            {
                var m = Moments.Of(new ArraySegment<double>(data, r * length, length), skipMissing: false);
                result[r] = m.Count;
                result[rows + r] = m.Mean;
                result[2 * rows + r] = m.M2;
            }
            return new NdArray(result, [3, .. outShape]);
        }

        var values = new double[rows];
        for (var r = 0; r < rows; r++)
        {
            var span = new ReadOnlySpan<double>(data, r * length, length);
            values[r] = how switch
            {
                SumCode => KahanSum(span),
                MinCode => length == 0 ? double.PositiveInfinity : Min(span),
                MaxCode => length == 0 ? double.NegativeInfinity : Max(span),
                _ => throw new ArgumentOutOfRangeException(nameof(how)),
            };
        }
        return new NdArray(values, outShape.Length == 0 ? [1] : outShape);
    }

    /// <summary>Merges per-block axis reductions produced by <see cref="ReduceAxis"/>, in order.</summary>
    [RemoteTask]
    public static NdArray CombineAxis(NdArray[] parts, int how)
    {
        if (how == SumCode) return SumBlocks(parts);
        if (how == MinCode) return parts.Aggregate(UFunc.Minimum);
        if (how == MaxCode) return parts.Aggregate(UFunc.Maximum);

        var shape = parts[0].Shape.ToArray();
        var n = parts[0].Size / 3;
        var result = new double[3 * n];
        var arrays = parts.Select(p => p.ToArray()).ToArray();
        for (var i = 0; i < n; i++)
        {
            var m = new Moments(0, 0, 0);
            foreach (var a in arrays) m = Moments.Merge(m, new Moments((long)a[i], a[n + i], a[2 * n + i]));
            result[i] = m.Count;
            result[n + i] = m.Mean;
            result[2 * n + i] = m.M2;
        }
        return new NdArray(result, shape);
    }

    /// <summary>The mean (<paramref name="ddof"/> &lt; 0) or variance from combined moments.</summary>
    [RemoteTask]
    public static NdArray FinishMoments(NdArray moments, int ddof)
    {
        var shape = moments.Shape[1..].ToArray();
        var n = moments.Size / 3;
        var a = moments.ToArray();
        var result = new double[n];
        for (var i = 0; i < n; i++)
            result[i] = ddof < 0 ? a[n + i] : new Moments((long)a[i], a[n + i], a[2 * n + i]).Variance(ddof);
        return new NdArray(result, shape.Length == 0 ? [1] : shape);
    }

    internal static double[] MomentsArray(Moments m) => [m.Count, m.Mean, m.M2];

    internal static Moments FromArray(double[] a) => new((long)a[0], a[1], a[2]);

    private static (int Rows, int Length, int[] OutShape) MoveAxisLast(NdArray block, int axis, out double[] data)
    {
        var rank = block.Rank;
        var order = Enumerable.Range(0, rank).Where(d => d != axis).Append(axis).ToArray();
        data = block.Transpose(order).ToArray();
        var length = block.Shape[axis];
        var outShape = Enumerable.Range(0, rank).Where(d => d != axis).Select(d => block.Shape[d]).ToArray();
        var rows = length == 0 ? Shapes.Size(outShape) : data.Length / length;
        return (rows, length, outShape);
    }

    /// <summary>Compensated summation: a sum over a million chunks should not depend on chunking.</summary>
    private static double KahanSum(ReadOnlySpan<double> values)
    {
        double sum = 0, c = 0;
        foreach (var v in values)
        {
            var y = v - c;
            var t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        return sum;
    }

    private static double Min(ReadOnlySpan<double> s) { var m = s[0]; foreach (var v in s) if (v < m || double.IsNaN(v)) m = v; return m; }

    private static double Max(ReadOnlySpan<double> s) { var m = s[0]; foreach (var v in s) if (v > m || double.IsNaN(v)) m = v; return m; }
}
