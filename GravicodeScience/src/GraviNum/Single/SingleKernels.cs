using System.Numerics;

namespace Gravicode.Science.GraviNum.Single;

/// <summary>
/// Single-precision versions of the two kernels that dominate runtime, for measuring what a
/// <c>float</c> path would actually buy.
/// </summary>
/// <remarks>
/// <para>
/// Everything else in this library is <c>double</c>. Making that generic — <c>NdArray&lt;T&gt;</c>
/// over <c>INumber&lt;T&gt;</c> — is the largest single change on the roadmap, because
/// <see cref="NdArray"/> is the type all six libraries are written against. It is not a change to
/// start without knowing the payoff, so this is the prototype that measures it: the element-wise
/// path and the matrix product, the two places the time actually goes.
/// </para>
/// <para>
/// Two effects are expected and worth separating. A <c>Vector&lt;float&gt;</c> holds twice the
/// lanes of a <c>Vector&lt;double&gt;</c>, which helps compute-bound work; and every value is half
/// the bytes, which helps bandwidth-bound work. Element-wise arithmetic is bound by bandwidth and
/// the matrix product by neither purely, so the two should not gain equally — and the difference
/// is the useful result.
/// </para>
/// <para>
/// <b>This is a measurement tool, not a parallel API.</b> It deliberately does not grow an
/// <c>NdArray</c>-shaped surface; a half-finished second numeric stack would be worse than none.
/// </para>
/// </remarks>
public static class SingleKernels
{
    /// <summary>Lanes a <see cref="Vector{T}"/> holds for <see cref="float"/> on this machine.</summary>
    public static int VectorWidth => Vector<float>.Count;

    /// <summary>Element count above which the work is spread across cores.</summary>
    public const int ParallelThreshold = 24_000;

    /// <summary>Element-wise sum of two equally sized arrays.</summary>
    /// <remarks>
    /// Pins and passes pointers rather than copying, for the same reason the <c>double</c> path
    /// does: this operation is bound by memory bandwidth, so an extra pass over the data costs
    /// more than the parallelism saves.
    /// </remarks>
    public static unsafe void Add(float[] a, float[] b, float[] result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("Add needs three arrays of the same length.");

        var n = a.Length;

        if (n >= ParallelThreshold)
        {
            fixed (float* pa = a, pb = b, pr = result)
            {
                var x = pa;
                var y = pb;
                var z = pr;

                Parallel.For(0, Environment.ProcessorCount, worker =>
                {
                    var (from, to) = Partition(n, worker);
                    if (to > from) AddSpan(new ReadOnlySpan<float>(x + from, to - from),
                        new ReadOnlySpan<float>(y + from, to - from),
                        new Span<float>(z + from, to - from));
                });
            }
            return;
        }

        AddSpan(a, b, result);
    }

    private static void AddSpan(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        var width = Vector<float>.Count;
        var n = a.Length;
        var i = 0;

        if (Vector.IsHardwareAccelerated && n >= width)
            for (; i <= n - width; i += width)
                (new Vector<float>(a.Slice(i, width)) + new Vector<float>(b.Slice(i, width)))
                    .CopyTo(result.Slice(i, width));

        for (; i < n; i++) result[i] = a[i] + b[i];
    }

    /// <summary>
    /// Row-major <c>C = A B</c>, using the same four-row register-blocked kernel as the
    /// <c>double</c> path so the comparison is like for like.
    /// </summary>
    public static unsafe void Multiply(float[] a, float[] b, float[] c, int m, int n, int k)
    {
        var slice = KSlice(k, n);
        var blocks = (m + 3) / 4;

        fixed (float* aBase = a, bBase = b, cBase = c)
        {
            var aPtr = aBase;
            var bPtr = bBase;
            var cPtr = cBase;

            for (var p0 = 0; p0 < k; p0 += slice)
            {
                var pn = Math.Min(slice, k - p0);
                var start = p0;

                Parallel.For(0, blocks, block =>
                {
                    var i0 = block * 4;
                    var rows = Math.Min(4, m - i0);
                    var width = Vector<float>.Count;

                    var aRow = aPtr + (long)i0 * k + start;
                    var bRow = bPtr + (long)start * n;
                    var c0 = cPtr + (long)i0 * n;

                    var j = 0;
                    if (Vector.IsHardwareAccelerated && n >= width && rows == 4)
                    {
                        for (; j <= n - width; j += width)
                        {
                            Vector<float> acc0 = default, acc1 = default, acc2 = default, acc3 = default;
                            var bp = bRow + j;

                            for (var p = 0; p < pn; p++, bp += n)
                            {
                                var bVec = Vector.Load(bp);
                                acc0 += new Vector<float>(aRow[p]) * bVec;
                                acc1 += new Vector<float>(aRow[k + p]) * bVec;
                                acc2 += new Vector<float>(aRow[2 * k + p]) * bVec;
                                acc3 += new Vector<float>(aRow[3 * k + p]) * bVec;
                            }

                            Vector.Store(Vector.Load(c0 + j) + acc0, c0 + j);
                            Vector.Store(Vector.Load(c0 + n + j) + acc1, c0 + n + j);
                            Vector.Store(Vector.Load(c0 + 2 * n + j) + acc2, c0 + 2 * n + j);
                            Vector.Store(Vector.Load(c0 + 3 * n + j) + acc3, c0 + 3 * n + j);
                        }
                    }

                    for (; j < n; j++)
                    {
                        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
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
                });
            }
        }
    }

    /// <summary>Rows of B per pass, sized so the live block stays in cache.</summary>
    /// <remarks>
    /// Half the bytes per element means twice as many rows fit the same cache budget as the
    /// <c>double</c> path — one of the two effects being measured.
    /// </remarks>
    private static int KSlice(int k, int n)
    {
        const long target = 4 * 1024 * 1024;
        if (k <= 8) return k;
        return (int)Math.Clamp(target / Math.Max(1, (long)n * 4), 8, k);
    }

    private static (int From, int To) Partition(int total, int worker)
    {
        var workers = Environment.ProcessorCount;
        var chunk = (total + workers - 1) / workers;
        var from = worker * chunk;
        return (Math.Min(from, total), Math.Min(from + chunk, total));
    }

    /// <summary>Converts a <c>double</c> array to <c>float</c>.</summary>
    public static float[] ToSingle(IReadOnlyList<double> values)
    {
        var result = new float[values.Count];
        for (var i = 0; i < values.Count; i++) result[i] = (float)values[i];
        return result;
    }
}
