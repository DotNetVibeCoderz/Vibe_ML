using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Gravicode.Science.SciNet.Sparse;

/// <summary>
/// The level-2 kernels of the Lanczos/Arnoldi iterations: <c>h = Vᵀw</c> and <c>w −= V h</c> on a
/// basis stored one contiguous vector per row. Four basis vectors are processed per sweep, so
/// <c>w</c> is read once per four vectors instead of once per vector — these loops are
/// memory-bound, and that halves their traffic against the one-dot-at-a-time version.
/// </summary>
internal static unsafe class SparseKrylovKernels
{
    /// <summary>
    /// Basis size × n above which the sweeps are split across threads. Measured on Poisson
    /// <c>eigsh</c>: no gain at n = 10k, about 1.2× at n = 40k. Settable only for benchmarks.
    /// </summary>
    internal static long ParallelThreshold = 400_000;

    private static readonly bool UseFma = Fma.IsSupported && Avx2.IsSupported;

    /// <summary><c>h[i] = V[i]·w</c> for i &lt; count.</summary>
    internal static void MultiDot(double[] v, int n, int count, ReadOnlySpan<double> w, Span<double> h)
    {
        fixed (double* pv = v)
        fixed (double* pw = w)
        fixed (double* ph = h)
        {
            if ((long)count * n >= ParallelThreshold && Environment.ProcessorCount > 1)
            {
                var chunks = Math.Min(Environment.ProcessorCount, 8);
                var len = ((n + chunks - 1) / chunks + 7) & ~7;
                var partial = new double[chunks * count];
                nint av = (nint)pv, aw = (nint)pw;
                Parallel.For(0, chunks, c =>
                {
                    var s0 = c * len;
                    var s1 = Math.Min(n, s0 + len);
                    if (s1 <= s0) return;
                    fixed (double* pp = &partial[c * count])
                        MultiDotRange((double*)av, n, count, (double*)aw, pp, s0, s1);
                });
                for (var i = 0; i < count; i++)
                {
                    var s = 0.0;
                    for (var c = 0; c < chunks; c++) s += partial[c * count + i];
                    ph[i] = s;
                }
                return;
            }
            MultiDotRange(pv, n, count, pw, ph, 0, n);
        }
    }

    private static void MultiDotRange(double* v, int n, int count, double* w, double* h, int s0, int s1)
    {
        var i = 0;
        for (; i + 4 <= count; i += 4)
        {
            double* v0 = v + (long)i * n, v1 = v0 + n, v2 = v1 + n, v3 = v2 + n;
            var t = s0;
            double d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            if (UseFma)
            {
                Vector256<double> a0 = default, a1 = default, a2 = default, a3 = default;
                for (; t + 4 <= s1; t += 4)
                {
                    var x = Avx.LoadVector256(w + t);
                    a0 = Fma.MultiplyAdd(Avx.LoadVector256(v0 + t), x, a0);
                    a1 = Fma.MultiplyAdd(Avx.LoadVector256(v1 + t), x, a1);
                    a2 = Fma.MultiplyAdd(Avx.LoadVector256(v2 + t), x, a2);
                    a3 = Fma.MultiplyAdd(Avx.LoadVector256(v3 + t), x, a3);
                }
                d0 = Vector256.Sum(a0); d1 = Vector256.Sum(a1); d2 = Vector256.Sum(a2); d3 = Vector256.Sum(a3);
            }
            for (; t < s1; t++)
            {
                var x = w[t];
                d0 += v0[t] * x; d1 += v1[t] * x; d2 += v2[t] * x; d3 += v3[t] * x;
            }
            h[i] = d0; h[i + 1] = d1; h[i + 2] = d2; h[i + 3] = d3;
        }
        for (; i < count; i++)
            h[i] = SparseDenseKernels.Dot(v + (long)i * n + s0, w + s0, s1 - s0);
    }

    /// <summary><c>w −= Σ h[i]·V[i]</c> for i &lt; count.</summary>
    internal static void MultiAxpy(double[] v, int n, int count, ReadOnlySpan<double> h, Span<double> w)
    {
        fixed (double* pv = v)
        fixed (double* pw = w)
        fixed (double* ph = h)
        {
            if ((long)count * n >= ParallelThreshold && Environment.ProcessorCount > 1)
            {
                var chunks = Math.Min(Environment.ProcessorCount, 8);
                var len = ((n + chunks - 1) / chunks + 7) & ~7;
                nint av = (nint)pv, aw = (nint)pw, ah = (nint)ph;
                Parallel.For(0, chunks, c =>
                {
                    var s0 = c * len;
                    var s1 = Math.Min(n, s0 + len);
                    if (s1 > s0) MultiAxpyRange((double*)av, n, count, (double*)ah, (double*)aw, s0, s1);
                });
                return;
            }
            MultiAxpyRange(pv, n, count, ph, pw, 0, n);
        }
    }

    private static void MultiAxpyRange(double* v, int n, int count, double* h, double* w, int s0, int s1)
    {
        var i = 0;
        for (; i + 4 <= count; i += 4)
        {
            double* v0 = v + (long)i * n, v1 = v0 + n, v2 = v1 + n, v3 = v2 + n;
            double h0 = h[i], h1 = h[i + 1], h2 = h[i + 2], h3 = h[i + 3];
            var t = s0;
            if (UseFma)
            {
                Vector256<double> b0 = Vector256.Create(h0), b1 = Vector256.Create(h1), b2 = Vector256.Create(h2), b3 = Vector256.Create(h3);
                for (; t + 4 <= s1; t += 4)
                {
                    var acc = Avx.Multiply(Avx.LoadVector256(v0 + t), b0);
                    acc = Fma.MultiplyAdd(Avx.LoadVector256(v1 + t), b1, acc);
                    acc = Fma.MultiplyAdd(Avx.LoadVector256(v2 + t), b2, acc);
                    acc = Fma.MultiplyAdd(Avx.LoadVector256(v3 + t), b3, acc);
                    Avx.Store(w + t, Avx.Subtract(Avx.LoadVector256(w + t), acc));
                }
            }
            for (; t < s1; t++) w[t] -= h0 * v0[t] + h1 * v1[t] + h2 * v2[t] + h3 * v3[t];
        }
        for (; i < count; i++)
            SparseDenseKernels.AxpySub(h[i], v + (long)i * n + s0, w + s0, s1 - s0);
    }

    /// <summary>
    /// <c>out[c] = Σ_r coef[r, c] · V[r]</c> for c &lt; cols (coef row-major m×cols): the basis
    /// rotation of a restart, as one GEMM.
    /// </summary>
    internal static void Combine(double[] v, int n, int m, double[] coef, int cols, double[] output)
    {
        // out (cols × n) = coefᵀ (cols × m) · V (m × n); GemmSub computes C −= A B, so negate A.
        var a = new double[cols * m];
        for (var c = 0; c < cols; c++)
            for (var r = 0; r < m; r++) a[c * m + r] = -coef[r * cols + c];
        Array.Clear(output, 0, cols * n);
        fixed (double* pa = a)
        fixed (double* pv = v)
        fixed (double* po = output)
            SparseDenseKernels.GemmSub(cols, n, m, pa, m, pv, n, po, n);
    }

    /// <summary>
    /// Classical Gram–Schmidt of <paramref name="w"/> against the first <paramref name="count"/>
    /// basis vectors with the DGKS correction (Daniel, Gragg, Kaufman &amp; Stewart 1976), as
    /// ARPACK's <c>dsaitr</c>/<c>dnaitr</c>: a second pass only when the first removed more than
    /// ~30% of the norm (‖w‖ &lt; 0.717·‖w₀‖), which is when cancellation can have left
    /// components behind. Accumulates the coefficients into <paramref name="h"/>; returns ‖w‖.
    /// </summary>
    internal static double Orthogonalize(double[] v, int n, int count, Span<double> w, Span<double> h, Span<double> scratch, double norm0)
    {
        MultiDot(v, n, count, w, h);
        MultiAxpy(v, n, count, h, w);
        var norm = Norm(w);
        for (var pass = 0; pass < 2 && norm < 0.717 * norm0; pass++)
        {
            MultiDot(v, n, count, w, scratch);
            MultiAxpy(v, n, count, scratch, w);
            for (var i = 0; i < count; i++) h[i] += scratch[i];
            var prev = norm;
            norm = Norm(w);
            if (norm >= 0.717 * prev) break;
            norm0 = prev;
        }
        return norm;
    }

    internal static double Norm(ReadOnlySpan<double> x)
    {
        // Scaled only if the plain sum over/underflows; the plain sum is what is fast.
        fixed (double* p = x)
        {
            var s = SparseDenseKernels.Dot(p, p, x.Length);
            if (s > 1e-280 && s < 1e280) return Math.Sqrt(s);
        }
        var scale = 0.0;
        foreach (var v in x) scale = Math.Max(scale, Math.Abs(v));
        if (scale == 0) return 0;
        var acc = 0.0;
        foreach (var v in x) { var t = v / scale; acc += t * t; }
        return scale * Math.Sqrt(acc);
    }
}
