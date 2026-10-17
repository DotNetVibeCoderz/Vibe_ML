using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Gravicode.Science.SciNet.Sparse;

/// <summary>
/// Dense kernels for the frontal matrices of the supernodal factorisations. Everything is
/// row-major with an explicit leading dimension, on raw pointers: the fronts live in one reused
/// buffer and the hot loops must not bounds-check.
/// </summary>
internal static unsafe class SparseDenseKernels
{
    /// <summary>Multiply-adds above which <see cref="GemmSub"/> splits rows across threads (measured).</summary>
    internal const long ParallelGemmThreshold = 1_500_000;

    private static readonly bool UseFma = Fma.IsSupported && Avx2.IsSupported;

    /// <summary><c>Σ x[i]·y[i]</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double Dot(double* x, double* y, int len)
    {
        var i = 0;
        double s = 0;
        if (UseFma && len >= 8)
        {
            var a0 = Vector256<double>.Zero;
            var a1 = Vector256<double>.Zero;
            for (; i + 8 <= len; i += 8)
            {
                a0 = Fma.MultiplyAdd(Avx.LoadVector256(x + i), Avx.LoadVector256(y + i), a0);
                a1 = Fma.MultiplyAdd(Avx.LoadVector256(x + i + 4), Avx.LoadVector256(y + i + 4), a1);
            }
            s = Vector256.Sum(Avx.Add(a0, a1));
        }
        for (; i < len; i++) s += x[i] * y[i];
        return s;
    }

    /// <summary><c>y −= α·x</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void AxpySub(double alpha, double* x, double* y, int len)
    {
        var i = 0;
        if (UseFma && len >= 4)
        {
            var va = Vector256.Create(alpha);
            for (; i + 4 <= len; i += 4)
                Avx.Store(y + i, Fma.MultiplyAddNegated(va, Avx.LoadVector256(x + i), Avx.LoadVector256(y + i)));
        }
        for (; i < len; i++) y[i] -= alpha * x[i];
    }

    /// <summary>
    /// <c>C −= A·B</c> with A m×k, B k×n, C m×n, all row-major with leading dimensions.
    /// A 4×8 register tile (eight FMA accumulators) when AVX2/FMA is present; rows are split across
    /// threads above <see cref="ParallelGemmThreshold"/>.
    /// </summary>
    internal static void GemmSub(int m, int n, int k, double* a, int lda, double* b, int ldb, double* c, int ldc, bool allowParallel = true)
    {
        if (m <= 0 || n <= 0 || k <= 0) return;
        var work = (long)m * n * k;
        if (allowParallel && work >= ParallelGemmThreshold && m >= 32 && Environment.ProcessorCount > 1)
        {
            var chunks = Math.Min(Environment.ProcessorCount * 2, m / 16);
            var rowsPer = ((m + chunks - 1) / chunks + 3) & ~3;
            var nb = (m + rowsPer - 1) / rowsPer;
            var pa = (nint)a; var pb = (nint)b; var pc = (nint)c;
            Parallel.For(0, nb, t =>
            {
                var r0 = t * rowsPer;
                var r1 = Math.Min(m, r0 + rowsPer);
                GemmRows(r1 - r0, n, k, (double*)pa + (long)r0 * lda, lda, (double*)pb, ldb, (double*)pc + (long)r0 * ldc, ldc);
            });
            return;
        }
        GemmRows(m, n, k, a, lda, b, ldb, c, ldc);
    }

    private static void GemmRows(int m, int n, int k, double* a, int lda, double* b, int ldb, double* c, int ldc)
    {
        if (!UseFma)
        {
            GemmReference(m, n, k, a, lda, b, ldb, c, ldc);
            return;
        }
        // Column panels of 256 keep B's panel (k × 256) in L2 while the rows stream past.
        const int NC = 256;
        for (var j0 = 0; j0 < n; j0 += NC)
        {
            var nn = Math.Min(NC, n - j0);
            var i = 0;
            for (; i + 4 <= m; i += 4)
            {
                var a0 = a + (long)i * lda;
                var c0 = c + (long)i * ldc + j0;
                var j = 0;
                for (; j + 8 <= nn; j += 8) Tile4x8(k, a0, lda, b + j0 + j, ldb, c0 + j, ldc);
                for (; j + 4 <= nn; j += 4) Tile4x4(k, a0, lda, b + j0 + j, ldb, c0 + j, ldc);
                for (; j < nn; j++)
                {
                    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    var bj = b + j0 + j;
                    for (var t = 0; t < k; t++)
                    {
                        var bv = bj[(long)t * ldb];
                        s0 += a0[t] * bv; s1 += a0[lda + t] * bv; s2 += a0[2 * lda + t] * bv; s3 += a0[3 * lda + t] * bv;
                    }
                    c0[j] -= s0; c0[ldc + j] -= s1; c0[2 * ldc + j] -= s2; c0[3 * ldc + j] -= s3;
                }
            }
            for (; i < m; i++)
            {
                var ai = a + (long)i * lda;
                var ci = c + (long)i * ldc + j0;
                for (var t = 0; t < k; t++)
                {
                    var av = ai[t];
                    if (av != 0) AxpySub(av, b + (long)t * ldb + j0, ci, nn);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Tile4x8(int k, double* a, int lda, double* b, int ldb, double* c, int ldc)
    {
        Vector256<double> c00 = default, c01 = default, c10 = default, c11 = default,
            c20 = default, c21 = default, c30 = default, c31 = default;
        double* a1 = a + lda, a2 = a + 2 * lda, a3 = a + 3 * lda;
        for (var t = 0; t < k; t++)
        {
            var bp = b + (long)t * ldb;
            var b0 = Avx.LoadVector256(bp);
            var b1 = Avx.LoadVector256(bp + 4);
            var av = Vector256.Create(a[t]);
            c00 = Fma.MultiplyAdd(av, b0, c00); c01 = Fma.MultiplyAdd(av, b1, c01);
            av = Vector256.Create(a1[t]);
            c10 = Fma.MultiplyAdd(av, b0, c10); c11 = Fma.MultiplyAdd(av, b1, c11);
            av = Vector256.Create(a2[t]);
            c20 = Fma.MultiplyAdd(av, b0, c20); c21 = Fma.MultiplyAdd(av, b1, c21);
            av = Vector256.Create(a3[t]);
            c30 = Fma.MultiplyAdd(av, b0, c30); c31 = Fma.MultiplyAdd(av, b1, c31);
        }
        Avx.Store(c, Avx.Subtract(Avx.LoadVector256(c), c00)); Avx.Store(c + 4, Avx.Subtract(Avx.LoadVector256(c + 4), c01));
        c += ldc;
        Avx.Store(c, Avx.Subtract(Avx.LoadVector256(c), c10)); Avx.Store(c + 4, Avx.Subtract(Avx.LoadVector256(c + 4), c11));
        c += ldc;
        Avx.Store(c, Avx.Subtract(Avx.LoadVector256(c), c20)); Avx.Store(c + 4, Avx.Subtract(Avx.LoadVector256(c + 4), c21));
        c += ldc;
        Avx.Store(c, Avx.Subtract(Avx.LoadVector256(c), c30)); Avx.Store(c + 4, Avx.Subtract(Avx.LoadVector256(c + 4), c31));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Tile4x4(int k, double* a, int lda, double* b, int ldb, double* c, int ldc)
    {
        Vector256<double> c0 = default, c1 = default, c2 = default, c3 = default;
        double* a1 = a + lda, a2 = a + 2 * lda, a3 = a + 3 * lda;
        for (var t = 0; t < k; t++)
        {
            var b0 = Avx.LoadVector256(b + (long)t * ldb);
            c0 = Fma.MultiplyAdd(Vector256.Create(a[t]), b0, c0);
            c1 = Fma.MultiplyAdd(Vector256.Create(a1[t]), b0, c1);
            c2 = Fma.MultiplyAdd(Vector256.Create(a2[t]), b0, c2);
            c3 = Fma.MultiplyAdd(Vector256.Create(a3[t]), b0, c3);
        }
        Avx.Store(c, Avx.Subtract(Avx.LoadVector256(c), c0)); c += ldc;
        Avx.Store(c, Avx.Subtract(Avx.LoadVector256(c), c1)); c += ldc;
        Avx.Store(c, Avx.Subtract(Avx.LoadVector256(c), c2)); c += ldc;
        Avx.Store(c, Avx.Subtract(Avx.LoadVector256(c), c3));
    }

    /// <summary>The scalar triple loop: the reference the tiled kernel is tested against.</summary>
    internal static void GemmReference(int m, int n, int k, double* a, int lda, double* b, int ldb, double* c, int ldc)
    {
        for (var i = 0; i < m; i++)
            for (var t = 0; t < k; t++)
            {
                var av = a[(long)i * lda + t];
                if (av == 0) continue;
                var bt = b + (long)t * ldb;
                var ci = c + (long)i * ldc;
                for (var j = 0; j < n; j++) ci[j] -= av * bt[j];
            }
    }
}
