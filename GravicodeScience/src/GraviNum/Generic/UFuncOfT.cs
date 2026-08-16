using System.Numerics;
using System.Runtime.CompilerServices;

namespace Gravicode.Science.GraviNum.Generic;

/// <summary>
/// Element-wise and linear-algebra kernels over <see cref="NdArray{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// One implementation serves every element width. <c>Vector&lt;T&gt;</c> is itself generic, so the
/// same source compiles to four lanes for <c>double</c> and eight for <c>float</c> — which is
/// exactly where the measured single-precision gain comes from, along with halving the bytes moved.
/// </para>
/// <para>
/// The structure mirrors the <c>double</c> kernels deliberately, including the parts that look
/// fussy: pointers rather than copies on the parallel path, because element-wise work is bound by
/// memory bandwidth and an extra pass costs more than the threads save; and a tile of C held in
/// registers across a slice of <c>k</c> in the matrix product, because that is what keeps the
/// strided walk through B inside cache.
/// </para>
/// </remarks>
public static class UFunc<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    /// <summary>Element count above which work is spread across cores.</summary>
    public const int ParallelThreshold = 24_000;

    /// <summary>Lanes a <see cref="Vector{T}"/> holds for this element type.</summary>
    public static int VectorWidth => Vector<T>.Count;

    /// <summary>Element-wise sum, for equally shaped operands.</summary>
    public static NdArray<T> Add(NdArray<T> a, NdArray<T> b)
        => Binary(a, b, static (x, y) => x + y, static (x, y) => x + y);

    /// <summary>Element-wise difference.</summary>
    public static NdArray<T> Subtract(NdArray<T> a, NdArray<T> b)
        => Binary(a, b, static (x, y) => x - y, static (x, y) => x - y);

    /// <summary>Element-wise product.</summary>
    public static NdArray<T> Multiply(NdArray<T> a, NdArray<T> b)
        => Binary(a, b, static (x, y) => x * y, static (x, y) => x * y);

    /// <summary>Element-wise quotient.</summary>
    public static NdArray<T> Divide(NdArray<T> a, NdArray<T> b)
        => Binary(a, b, static (x, y) => x / y, static (x, y) => x / y);

    /// <summary>Element-wise square root.</summary>
    public static NdArray<T> Sqrt(NdArray<T> a)
        => Unary(a, T.Sqrt, static v => Vector.SquareRoot(v));

    /// <summary>Element-wise absolute value.</summary>
    public static NdArray<T> Abs(NdArray<T> a) => Unary(a, T.Abs, static v => Vector.Abs(v));

    /// <summary>Element-wise exponential.</summary>
    /// <remarks>No vector form: <c>Vector&lt;T&gt;</c> has no transcendental functions, and .NET's
    /// scalar intrinsics already beat a hand-written polynomial — measured, not assumed.</remarks>
    public static NdArray<T> Exp(NdArray<T> a) => Unary(a, T.Exp);

    /// <summary>Element-wise natural logarithm.</summary>
    public static NdArray<T> Log(NdArray<T> a) => Unary(a, T.Log);

    /// <summary>Sum of every element.</summary>
    /// <remarks>
    /// Pairwise rather than a running total: summing a million values in order accumulates error
    /// proportional to the count, while halving the problem keeps it proportional to its logarithm.
    /// That matters far more in <c>float</c>, where there are only about seven digits to lose.
    /// </remarks>
    public static T Sum(NdArray<T> a)
    {
        if (a.Size == 0) return T.Zero;
        return a.IsContiguous ? PairwiseSum(a.AsSpan()) : PairwiseSum(a.ToArray());
    }

    /// <summary>Arithmetic mean.</summary>
    public static T Mean(NdArray<T> a)
        => a.Size == 0 ? T.Zero : Sum(a) / T.CreateChecked(a.Size);

    private static T PairwiseSum(ReadOnlySpan<T> values)
    {
        const int blockSize = 128;
        if (values.Length <= blockSize)
        {
            var total = T.Zero;
            foreach (var v in values) total += v;
            return total;
        }

        var half = values.Length / 2;
        return PairwiseSum(values[..half]) + PairwiseSum(values[half..]);
    }

    // ---------------------------------------------------------------- kernels

    /// <summary>Applies a function to every element.</summary>
    public static unsafe NdArray<T> Unary(NdArray<T> a, Func<T, T> scalar,
        Func<Vector<T>, Vector<T>>? vector = null)
    {
        var result = NdArray<T>.Zeros(a.Shape.ToArray());
        var source = a.AsContiguous();
        var n = source.Size;

        var src = source.AsSpan();
        var dst = result.AsSpan();

        if (n >= ParallelThreshold)
        {
            fixed (T* s = src, d = dst)
            {
                var sp = s;
                var dp = d;
                Parallel.For(0, Environment.ProcessorCount, worker =>
                {
                    var (from, to) = Partition(n, worker);
                    if (to > from)
                        UnarySpan(new ReadOnlySpan<T>(sp + from, to - from),
                            new Span<T>(dp + from, to - from), scalar, vector);
                });
            }
            return result;
        }

        UnarySpan(src, dst, scalar, vector);
        return result;
    }

    private static void UnarySpan(ReadOnlySpan<T> src, Span<T> dst,
        Func<T, T> scalar, Func<Vector<T>, Vector<T>>? vector)
    {
        var n = src.Length;
        var i = 0;

        if (vector is not null && Vector.IsHardwareAccelerated && n >= Vector<T>.Count)
        {
            var width = Vector<T>.Count;
            for (; i <= n - width; i += width)
                vector(new Vector<T>(src.Slice(i, width))).CopyTo(dst.Slice(i, width));
        }

        for (; i < n; i++) dst[i] = scalar(src[i]);
    }

    /// <summary>Applies a binary function element-wise. Both operands must have the same shape.</summary>
    public static unsafe NdArray<T> Binary(NdArray<T> a, NdArray<T> b,
        Func<T, T, T> scalar, Func<Vector<T>, Vector<T>, Vector<T>>? vector = null)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException(
                $"Shapes ({Shapes.Describe(a.Shape)}) and ({Shapes.Describe(b.Shape)}) differ. "
                + "The generic kernels do not broadcast; reshape explicitly.");

        var left = a.AsContiguous();
        var right = b.AsContiguous();
        var result = NdArray<T>.Zeros(a.Shape.ToArray());
        var n = result.Size;

        var ls = left.AsSpan();
        var rs = right.AsSpan();
        var ds = result.AsSpan();

        if (n >= ParallelThreshold)
        {
            // Pinned pointers rather than copies: this is bandwidth-bound work, and copying the
            // operands to satisfy a lambda capture costs three extra passes over memory.
            fixed (T* lp = ls, rp = rs, dp = ds)
            {
                var l = lp;
                var r = rp;
                var d = dp;
                Parallel.For(0, Environment.ProcessorCount, worker =>
                {
                    var (from, to) = Partition(n, worker);
                    if (to > from)
                        BinarySpan(new ReadOnlySpan<T>(l + from, to - from),
                            new ReadOnlySpan<T>(r + from, to - from),
                            new Span<T>(d + from, to - from), scalar, vector);
                });
            }
            return result;
        }

        BinarySpan(ls, rs, ds, scalar, vector);
        return result;
    }

    private static void BinarySpan(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> dst,
        Func<T, T, T> scalar, Func<Vector<T>, Vector<T>, Vector<T>>? vector)
    {
        var n = left.Length;
        var i = 0;

        if (vector is not null && Vector.IsHardwareAccelerated && n >= Vector<T>.Count)
        {
            var width = Vector<T>.Count;
            for (; i <= n - width; i += width)
                vector(new Vector<T>(left.Slice(i, width)), new Vector<T>(right.Slice(i, width)))
                    .CopyTo(dst.Slice(i, width));
        }

        for (; i < n; i++) dst[i] = scalar(left[i], right[i]);
    }

    /// <summary>
    /// Matrix product, using the same register-blocked kernel as the <c>double</c> path.
    /// </summary>
    /// <remarks>
    /// Four rows of C are accumulated in registers across a slice of <c>k</c>. Both halves matter:
    /// registers keep C out of memory, and slicing <c>k</c> bounds the strided walk through B so
    /// it stays in cache. Doing only the first is slower at 1024 and above.
    /// </remarks>
    public static unsafe NdArray<T> Dot(NdArray<T> a, NdArray<T> b)
    {
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException($"Dot expects two rank 2 arrays, got {a.Rank} and {b.Rank}.");
        if (a.Shape[1] != b.Shape[0])
            throw new ArgumentException(
                $"Shapes ({Shapes.Describe(a.Shape)}) and ({Shapes.Describe(b.Shape)}) are not aligned.");

        var m = a.Shape[0];
        var k = a.Shape[1];
        var n = b.Shape[1];

        var left = a.AsContiguous();
        var right = b.AsContiguous();
        var result = NdArray<T>.Zeros(m, n);

        var slice = KSlice(k, n);
        var blocks = (m + 3) / 4;
        var parallel = m >= 64 || (long)m * n * k > 1_000_000;

        fixed (T* ap = left.AsSpan(), bp = right.AsSpan(), cp = result.AsSpan())
        {
            var aPtr = ap;
            var bPtr = bp;
            var cPtr = cp;

            for (var p0 = 0; p0 < k; p0 += slice)
            {
                var pn = Math.Min(slice, k - p0);
                var start = p0;

                if (parallel)
                    Parallel.For(0, blocks, block => RowBlock(aPtr, bPtr, cPtr, m, n, k, block, start, pn));
                else
                    for (var block = 0; block < blocks; block++)
                        RowBlock(aPtr, bPtr, cPtr, m, n, k, block, start, pn);
            }
        }

        return result;
    }

    private static unsafe void RowBlock(T* a, T* b, T* c, int m, int n, int k,
        int block, int p0, int pn)
    {
        var i0 = block * 4;
        var rows = Math.Min(4, m - i0);
        var width = Vector<T>.Count;

        var aRow = a + (long)i0 * k + p0;
        var bRow = b + (long)p0 * n;
        var c0 = c + (long)i0 * n;

        var j = 0;
        if (Vector.IsHardwareAccelerated && n >= width && rows == 4)
        {
            for (; j <= n - width; j += width)
            {
                Vector<T> acc0 = default, acc1 = default, acc2 = default, acc3 = default;
                var bp = bRow + j;

                for (var p = 0; p < pn; p++, bp += n)
                {
                    var bVec = Vector.Load(bp);
                    acc0 += new Vector<T>(aRow[p]) * bVec;
                    acc1 += new Vector<T>(aRow[k + p]) * bVec;
                    acc2 += new Vector<T>(aRow[2 * k + p]) * bVec;
                    acc3 += new Vector<T>(aRow[3 * k + p]) * bVec;
                }

                Vector.Store(Vector.Load(c0 + j) + acc0, c0 + j);
                Vector.Store(Vector.Load(c0 + n + j) + acc1, c0 + n + j);
                Vector.Store(Vector.Load(c0 + 2 * n + j) + acc2, c0 + 2 * n + j);
                Vector.Store(Vector.Load(c0 + 3 * n + j) + acc3, c0 + 3 * n + j);
            }
        }

        for (; j < n; j++)
        {
            T s0 = T.Zero, s1 = T.Zero, s2 = T.Zero, s3 = T.Zero;
            var bp = bRow + j;

            for (var p = 0; p < pn; p++, bp += n)
            {
                var bValue = *bp;
                s0 += aRow[p] * bValue;
                if (rows > 1) s1 += aRow[k + p] * bValue;
                if (rows > 2) s2 += aRow[2 * k + p] * bValue;
                if (rows > 3) s3 += aRow[3 * k + p] * bValue;
            }

            c0[j] += s0;
            if (rows > 1) c0[n + j] += s1;
            if (rows > 2) c0[2 * n + j] += s2;
            if (rows > 3) c0[3 * n + j] += s3;
        }
    }

    /// <summary>Rows of B per pass, sized so the live block stays in cache.</summary>
    /// <remarks>
    /// Scaled by <c>sizeof(T)</c>, so a <c>float</c> array gets twice as many rows in the same
    /// cache budget — the second half of the single-precision gain, after the wider vectors.
    /// </remarks>
    private static unsafe int KSlice(int k, int n)
    {
        const long target = 4 * 1024 * 1024;
        if (k <= 8) return k;
        return (int)Math.Clamp(target / Math.Max(1, (long)n * sizeof(T)), 8, k);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (int From, int To) Partition(int total, int worker)
    {
        var workers = Environment.ProcessorCount;
        var chunk = (total + workers - 1) / workers;
        var from = worker * chunk;
        return (Math.Min(from, total), Math.Min(from + chunk, total));
    }
}
