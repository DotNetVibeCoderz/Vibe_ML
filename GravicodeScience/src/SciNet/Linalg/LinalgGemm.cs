using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Packed, register-blocked, parallel matrix product <c>C = alpha·A·B + beta·C</c> on row-major
/// buffers with leading dimensions, so it can update a sub-block in place (the trailing update of
/// a blocked LU, the off-diagonal update of a blocked triangular solve).
/// </summary>
/// <remarks>
/// <para>
/// Why SciNet carries its own instead of calling GraviNum's <c>LinAlg.Dot</c>: Dot allocates its
/// result, takes no leading dimension (so it cannot write into a block of a larger matrix), and
/// below its packing threshold of eight million multiply-adds — exactly the 100–200 sizes that
/// expm, Sylvester and blocked factorisations live at — it runs an unpacked 4×1-vector kernel
/// without FMA. Measured at n = 200 that path took 4.7 ms for a product while a 400-cube took
/// 3.4 ms. This kernel packs at every size (packing is O(n²) against O(n³) arithmetic, and the
/// panels are what keep the inner loop out of main memory) and uses a 4×8 FMA tile.
/// </para>
/// <para>
/// Structure is the usual Goto/BLIS one in its simplest form: a k-slice of B is packed once into
/// panels of <see cref="Nr"/> columns and shared; each parallel task packs its own row block of A
/// into panels of <see cref="Mr"/> rows and runs the micro-kernel over its share of the columns.
/// Tasks split rows first and columns only when there are too few row blocks to keep every core
/// busy, which is what a tall-skinny update needs.
/// </para>
/// <para>
/// When a native BLAS is installed, whole-matrix products go through GraviNum (which calls it);
/// this kernel is the managed path, and the one the blocked factorisations always use.
/// </para>
/// </remarks>
internal static unsafe class LinalgGemm
{
    /// <summary>Rows of the micro-kernel tile.</summary>
    internal const int Mr = 4;

    /// <summary>Columns of the micro-kernel tile: two 256-bit vectors.</summary>
    internal const int Nr = 8;

    /// <summary>Depth of one packed slice; Kc·Nr·8 = 16 KB of B stays in L1 across a row block.</summary>
    private const int Kc = 256;

    /// <summary>Rows of A per packed block; Mc·Kc·8 = 128 KB stays in L2.</summary>
    private const int Mc = 64;

    /// <summary>Columns of B per packed slice, bounding the packed buffer to 4 MB.</summary>
    private const int Nc = 2048;

    /// <summary>Below this many multiply-adds one thread is faster than starting several.</summary>
    private const long ParallelThreshold = 96L * 96 * 96;

    [ThreadStatic] private static double[]? _packA;

    /// <summary><c>C = A B</c> for row-major operands, into a new array.</summary>
    public static double[] Multiply(ReadOnlySpan<double> a, ReadOnlySpan<double> b, int m, int k, int n)
    {
        var c = new double[m * n];
        fixed (double* pa = a, pb = b, pc = c)
            Gemm(m, n, k, 1.0, pa, k, pb, n, 0.0, pc, n);
        return c;
    }

    /// <summary>
    /// <c>C ← alpha·A·B + beta·C</c>. A is m×k with row stride <paramref name="lda"/>, B is k×n with
    /// stride <paramref name="ldb"/>, C is m×n with stride <paramref name="ldc"/>. C must not alias A or B.
    /// </summary>
    public static void Gemm(int m, int n, int k, double alpha, double* a, int lda, double* b, int ldb,
        double beta, double* c, int ldc, bool allowParallel = true)
    {
        if (m <= 0 || n <= 0) return;
        if (beta != 1.0)
        {
            for (var i = 0; i < m; i++)
            {
                var row = new Span<double>(c + (long)i * ldc, n);
                if (beta == 0.0) row.Clear();
                else for (var j = 0; j < n; j++) row[j] *= beta;
            }
        }
        if (k <= 0 || alpha == 0.0) return;
        if (!Vector256.IsHardwareAccelerated)
        {
            // No 256-bit SIMD (e.g. Arm64): Vector256 would run in software, so use plain
            // row-axpy loops instead — correct everywhere, fast nowhere in particular.
            for (var i = 0; i < m; i++)
            {
                var ci = new Span<double>(c + (long)i * ldc, n);
                for (var p = 0; p < k; p++)
                {
                    var aip = alpha * a[(long)i * lda + p];
                    if (aip == 0.0) continue;
                    LinAlg.AxpySpan(aip, new ReadOnlySpan<double>(b + (long)p * ldb, n), ci);
                }
            }
            return;
        }

        var parallel = allowParallel && (long)m * n * k >= ParallelThreshold && Environment.ProcessorCount > 1;
        var ncMax = Math.Min(n, Nc);
        var kcMax = Math.Min(k, Kc);
        var panelsMax = (ncMax + Nr - 1) / Nr;
        var packedB = ArrayPool<double>.Shared.Rent(panelsMax * Nr * kcMax);
        try
        {
            fixed (double* pbBase = packedB)
            {
                var pb = pbBase;
                for (var jc = 0; jc < n; jc += Nc)
                {
                    var nc = Math.Min(Nc, n - jc);
                    var panels = (nc + Nr - 1) / Nr;
                    for (var pc = 0; pc < k; pc += Kc)
                    {
                        var kc = Math.Min(Kc, k - pc);
                        PackB(b + (long)pc * ldb + jc, ldb, pb, kc, nc);
                        var aSlice = a + pc;
                        var cBlock = c + jc;

                        if (!parallel)
                        {
                            for (var ic = 0; ic < m; ic += Mc)
                                RowBlock(aSlice, lda, pb, cBlock, ldc, ic, Math.Min(Mc, m - ic), 0, panels, nc, kc, alpha);
                            continue;
                        }

                        // Rows first; columns as well only when rows alone cannot occupy every core.
                        var workers = Environment.ProcessorCount;
                        var rowsPer = Math.Max(Mr, Math.Min(Mc, RoundUp((m + workers - 1) / workers, Mr)));
                        var rowBlocks = (m + rowsPer - 1) / rowsPer;
                        var colChunks = Math.Max(1, Math.Min(panels, (2 * workers + rowBlocks - 1) / rowBlocks));
                        var panelsPer = (panels + colChunks - 1) / colChunks;
                        colChunks = (panels + panelsPer - 1) / panelsPer;
                        var tasks = rowBlocks * colChunks;
                        var aS = (nint)aSlice; var pbS = (nint)pb; var cS = (nint)cBlock;
                        var kcL = kc; var ncL = nc;
                        Parallel.For(0, tasks, t =>
                        {
                            var rb = t / colChunks;
                            var cb = t % colChunks;
                            var ic = rb * rowsPer;
                            var p0 = cb * panelsPer;
                            var p1 = Math.Min(panels, p0 + panelsPer);
                            RowBlock((double*)aS, lda, (double*)pbS, (double*)cS, ldc, ic, Math.Min(rowsPer, m - ic), p0, p1, ncL, kcL, alpha);
                        });
                    }
                }
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(packedB);
        }
    }

    private static int RoundUp(int x, int to) => (x + to - 1) / to * to;

    /// <summary>Packs one row block of A and runs the micro-kernel over panels [p0, p1) of packed B.</summary>
    private static void RowBlock(double* a, int lda, double* pb, double* c, int ldc, int ic, int mc,
        int p0, int p1, int nc, int kc, double alpha)
    {
        var mPanels = (mc + Mr - 1) / Mr;
        var need = mPanels * Mr * kc;
        var buf = _packA;
        if (buf is null || buf.Length < need) _packA = buf = new double[Math.Max(need, Mc * Kc)];
        fixed (double* pa = buf)
        {
            PackA(a + (long)ic * lda, lda, pa, mc, kc);
            var tile = stackalloc double[Mr * Nr];
            for (var jp = p0; jp < p1; jp++)
            {
                var j = jp * Nr;
                var cols = Math.Min(Nr, nc - j);
                var bPanel = pb + (long)jp * Nr * kc;
                for (var ip = 0; ip < mPanels; ip++)
                {
                    var i = ip * Mr;
                    var rows = Math.Min(Mr, mc - i);
                    var cTile = c + (long)(ic + i) * ldc + j;
                    if (rows == Mr && cols == Nr)
                        Kernel(pa + (long)ip * Mr * kc, bPanel, kc, alpha, cTile, ldc);
                    else
                    {
                        new Span<double>(tile, Mr * Nr).Clear();
                        Kernel(pa + (long)ip * Mr * kc, bPanel, kc, alpha, tile, Nr);
                        for (var r = 0; r < rows; r++)
                            for (var q = 0; q < cols; q++) cTile[(long)r * ldc + q] += tile[r * Nr + q];
                    }
                }
            }
        }
    }

    /// <summary>Copies a kc×nc block of B into Nr-wide panels, zero-padding the ragged last one.</summary>
    private static void PackB(double* b, int ldb, double* dst, int kc, int nc)
    {
        for (var j = 0; j < nc; j += Nr)
        {
            var span = Math.Min(Nr, nc - j);
            var d = dst + (long)(j / Nr) * Nr * kc;
            if (span == Nr)
            {
                for (var p = 0; p < kc; p++)
                {
                    var src = b + (long)p * ldb + j;
                    Vector256.Store(Vector256.Load(src), d);
                    Vector256.Store(Vector256.Load(src + 4), d + 4);
                    d += Nr;
                }
            }
            else
            {
                for (var p = 0; p < kc; p++)
                {
                    var src = b + (long)p * ldb + j;
                    for (var t = 0; t < span; t++) d[t] = src[t];
                    for (var t = span; t < Nr; t++) d[t] = 0.0;
                    d += Nr;
                }
            }
        }
    }

    /// <summary>Copies an mc×kc block of A into Mr-row panels stored k-major, zero-padding the tail.</summary>
    private static void PackA(double* a, int lda, double* dst, int mc, int kc)
    {
        for (var i = 0; i < mc; i += Mr)
        {
            var span = Math.Min(Mr, mc - i);
            var d = dst + (long)(i / Mr) * Mr * kc;
            if (span == Mr)
            {
                var r0 = a + (long)i * lda; var r1 = r0 + lda; var r2 = r1 + lda; var r3 = r2 + lda;
                for (var p = 0; p < kc; p++)
                {
                    d[0] = r0[p]; d[1] = r1[p]; d[2] = r2[p]; d[3] = r3[p];
                    d += Mr;
                }
            }
            else
            {
                for (var p = 0; p < kc; p++)
                {
                    for (var t = 0; t < span; t++) d[t] = a[(long)(i + t) * lda + p];
                    for (var t = span; t < Mr; t++) d[t] = 0.0;
                    d += Mr;
                }
            }
        }
    }

    /// <summary>The 4×8 register tile: <c>C[0..4, 0..8] += alpha · Apanel · Bpanel</c> over kc.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel(double* a, double* b, int kc, double alpha, double* c, int ldc)
    {
        Vector256<double> c00 = default, c01 = default, c10 = default, c11 = default;
        Vector256<double> c20 = default, c21 = default, c30 = default, c31 = default;
        if (Fma.IsSupported)
        {
            for (var p = 0; p < kc; p++)
            {
                var b0 = Vector256.Load(b);
                var b1 = Vector256.Load(b + 4);
                var a0 = Vector256.Create(a[0]);
                c00 = Fma.MultiplyAdd(a0, b0, c00); c01 = Fma.MultiplyAdd(a0, b1, c01);
                var a1 = Vector256.Create(a[1]);
                c10 = Fma.MultiplyAdd(a1, b0, c10); c11 = Fma.MultiplyAdd(a1, b1, c11);
                var a2 = Vector256.Create(a[2]);
                c20 = Fma.MultiplyAdd(a2, b0, c20); c21 = Fma.MultiplyAdd(a2, b1, c21);
                var a3 = Vector256.Create(a[3]);
                c30 = Fma.MultiplyAdd(a3, b0, c30); c31 = Fma.MultiplyAdd(a3, b1, c31);
                a += Mr; b += Nr;
            }
        }
        else
        {
            for (var p = 0; p < kc; p++)
            {
                var b0 = Vector256.Load(b);
                var b1 = Vector256.Load(b + 4);
                var a0 = Vector256.Create(a[0]); c00 += a0 * b0; c01 += a0 * b1;
                var a1 = Vector256.Create(a[1]); c10 += a1 * b0; c11 += a1 * b1;
                var a2 = Vector256.Create(a[2]); c20 += a2 * b0; c21 += a2 * b1;
                var a3 = Vector256.Create(a[3]); c30 += a3 * b0; c31 += a3 * b1;
                a += Mr; b += Nr;
            }
        }
        var al = Vector256.Create(alpha);
        Accumulate(c, c00, c01, al);
        Accumulate(c + ldc, c10, c11, al);
        Accumulate(c + 2L * ldc, c20, c21, al);
        Accumulate(c + 3L * ldc, c30, c31, al);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Accumulate(double* c, Vector256<double> lo, Vector256<double> hi, Vector256<double> alpha)
    {
        Vector256.Store(Vector256.Load(c) + alpha * lo, c);
        Vector256.Store(Vector256.Load(c + 4) + alpha * hi, c + 4);
    }

    // ================================================================ conveniences

    /// <summary>
    /// Whole-matrix product into a new <see cref="NdArray"/>: native BLAS through GraviNum when one
    /// is installed and the product is large enough, this kernel otherwise.
    /// </summary>
    public static NdArray Mul(NdArray a, NdArray b)
    {
        if (a.Rank != 2 || b.Rank != 2) return LinAlg.Dot(a, b);
        var m = a.Shape[0];
        var k = a.Shape[1];
        var n = b.Shape[1];
        if (b.Shape[0] != k) throw new ArgumentException($"Matrix product shape mismatch: ({m}x{k}) @ ({b.Shape[0]}x{n}).");
        if (Gravicode.Science.GraviNum.Compute.NativeBlas.ShouldUse(m, n, k)) return LinAlg.Dot(a, b);
        var av = a.AsContiguous().AsSpan();
        var bv = b.AsContiguous().AsSpan();
        return new NdArray(Multiply(av, bv, m, k, n), m, n);
    }
}
