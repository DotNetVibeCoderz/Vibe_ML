using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Gravicode.Science.GraviNum;

/// <summary>
/// Packed, register-blocked, parallel matrix product <c>C = alpha·A·B + beta·C</c> on strided
/// operands — the managed GEMM behind <see cref="LinAlg.Dot"/> at every size above a tiny one.
/// </summary>
/// <remarks>
/// <para>
/// Structure is the Goto/BLIS one in its simplest form: a <c>Kc</c>-deep slice of B is packed
/// once into panels of <c>Nr</c> columns and shared; each parallel task packs its own row block
/// of A into panels of <see cref="Mr"/> rows and runs a <c>Mr × Nr</c> register tile (two
/// <see cref="Vector{T}"/> wide, with FMA where the hardware has it) over its share of the
/// columns. Tasks split rows first and columns as well only when there are too few row blocks to
/// occupy every core — which is what a medium product (100–400) needs and what the previous
/// version of this class, parallel over 128-row blocks only, could not do: a 200-cube ran on two
/// threads.
/// </para>
/// <para>
/// Packing reads each operand through a row and a column stride, so a transposed or sliced view
/// is multiplied in place — the copy <see cref="NdArray.AsContiguous"/> would make is replaced by
/// the pack the kernel does anyway.
/// </para>
/// <para>
/// Deterministic: every element of C is summed over k in the same order — <c>Kc</c> slices in
/// sequence, each slice accumulated in one register from zero — whatever the thread count or
/// task split, because parallelism only ever divides C, never k. Different dispatch paths
/// (this kernel, the small blocked kernel, a native BLAS) do round differently from each other.
/// </para>
/// <para>
/// The micro-kernel is written on <see cref="Vector{T}"/>, not <c>Vector256</c>, so it is the
/// same code on Arm64 (two-lane vectors, a 4×4 tile) as on AVX2 (four lanes, 4×8).
/// </para>
/// </remarks>
public static unsafe class PackedMatMul
{
    /// <summary>
    /// Whether the packed kernel is used, when no native BLAS is present.
    /// </summary>
    /// <remarks>
    /// Public for the same reason <see cref="Compute.NativeBlas.Enabled"/> is: turning a fast path
    /// off is how it gets compared against the simpler one it replaced, and the simpler one stays
    /// in place as that reference.
    /// </remarks>
    public static bool Enabled { get; set; } = true;

    /// <summary>Rows of the micro-kernel tile.</summary>
    /// <remarks>
    /// Six rows (BLIS's Haswell shape, twelve accumulators) was measured against four in one
    /// process: 0.89–0.97x at 32–200 and 1.03–1.11x from 256 up. The medium sizes are the point
    /// of this kernel, so it stays at four.
    /// </remarks>
    public const int Mr = 4;

    /// <summary>Columns of the micro-kernel tile: two hardware vectors.</summary>
    public static int Nr => 2 * Vector<double>.Count;

    /// <summary>Depth of one packed slice; <c>Kc·Nr·8</c> = 16 KB of B stays in L1 across a row block.</summary>
    private const int Kc = 256;

    /// <summary>Rows of A per packed block; <c>Mc·Kc·8</c> = 128 KB stays in L2.</summary>
    private const int Mc = 64;

    /// <summary>Columns of B per packed slice, bounding the packed buffer to 4 MB.</summary>
    private const int Nc = 2048;

    /// <summary>
    /// Multiply-adds below which the simple blocked kernel in <see cref="LinAlg"/> wins: packing
    /// both operands is a fixed cost that a tiny product cannot repay.
    /// </summary>
    public const long PackThreshold = 2048;

    /// <summary>Multiply-adds below which one thread is faster than starting several.</summary>
    public const long ParallelThreshold = 64L * 64 * 64;

    private static readonly bool HasFma = Fma.IsSupported || AdvSimd.IsSupported;

    [ThreadStatic] private static double[]? _packA;

    /// <summary>Whether <see cref="LinAlg.Dot"/> routes a product of this shape to the packed kernel.</summary>
    /// <remarks>
    /// <para>
    /// Every rule here was measured against the blocked kernel on a 4-core AVX2 laptop, both
    /// kernels alternating in one process. Packing wins from a 16-cube up (1.2x there, 2x by 48);
    /// at 12 and below its fixed cost loses. Two thin shapes go the other way:
    /// </para>
    /// <list type="bullet">
    /// <item><c>m == 1</c> — a vector·matrix product would pad the four-row tile with three rows
    /// of zeros (1×500×500: blocked 0.12 ms, packed 0.21 ms).</item>
    /// <item><c>n</c> a multiple of the vector width but narrower than the tile — exactly the
    /// blocked kernel's shape with no scalar tail, while packed computes zeros in half the tile
    /// (500×500×4: 0.085 against 0.13 ms).</item>
    /// </list>
    /// <para>
    /// Shallow products (<c>k ≤ 16</c>) looked like a third case in one run and reversed in the
    /// next; on a shared, throttling machine that is noise, so there is no rule for it.
    /// A one-column <c>n == 1</c> product never gets here: <see cref="LinAlg.Dot"/> sends it to
    /// the matrix-vector path, which beat both (500×500×1: 0.043 ms against 0.065 blocked).
    /// </para>
    /// </remarks>
    public static bool ShouldUse(int m, int n, int k)
    {
        if (!Enabled || !Vector.IsHardwareAccelerated) return false;
        if ((long)m * n * k < PackThreshold || m == 1) return false;
        var w = Vector<double>.Count;
        if (n % w == 0 && n < 2 * w) return false;
        return true;
    }

    /// <summary>Computes <c>C = A B</c> for row-major contiguous operands; C is overwritten.</summary>
    public static void Multiply(
        double[] a, int aOffset, double[] b, int bOffset, double[] c, int m, int n, int k)
    {
        fixed (double* pa = a, pb = b, pc = c)
            Gemm(m, n, k, 1.0, pa + aOffset, k, 1, pb + bOffset, n, 1, 0.0, pc, n);
    }

    /// <summary>
    /// <c>C ← alpha·A·B + beta·C</c>. Element <c>(i, p)</c> of A is
    /// <c>a[i·rowStrideA + p·colStrideA]</c>, likewise for B; C is row-major with row stride
    /// <paramref name="ldc"/> and must not alias A or B.
    /// </summary>
    /// <param name="m">Rows of A and C.</param>
    /// <param name="n">Columns of B and C.</param>
    /// <param name="k">Columns of A, rows of B.</param>
    /// <param name="alpha">Scale applied to the product.</param>
    /// <param name="a">Element (0, 0) of A.</param>
    /// <param name="rowStrideA">Distance between rows of A, in elements.</param>
    /// <param name="colStrideA">Distance between columns of A, in elements.</param>
    /// <param name="b">Element (0, 0) of B.</param>
    /// <param name="rowStrideB">Distance between rows of B, in elements.</param>
    /// <param name="colStrideB">Distance between columns of B, in elements.</param>
    /// <param name="beta">Scale applied to C before accumulating; 0 overwrites it.</param>
    /// <param name="c">Element (0, 0) of C.</param>
    /// <param name="ldc">Distance between rows of C, in elements.</param>
    /// <param name="allowParallel">False keeps the call on the calling thread.</param>
    public static void Gemm(int m, int n, int k, double alpha,
        double* a, long rowStrideA, long colStrideA,
        double* b, long rowStrideB, long colStrideB,
        double beta, double* c, long ldc, bool allowParallel = true)
    {
        if (m <= 0 || n <= 0) return;
        if (beta != 1.0)
        {
            for (var i = 0; i < m; i++)
            {
                var row = new Span<double>(c + i * ldc, n);
                if (beta == 0.0) row.Clear();
                else for (var j = 0; j < n; j++) row[j] *= beta;
            }
        }
        if (k <= 0 || alpha == 0.0) return;

        if (!Vector.IsHardwareAccelerated)
        {
            // No SIMD at all: Vector<T> would run in software, so use plain row-axpy loops —
            // correct everywhere, fast nowhere in particular.
            for (var i = 0; i < m; i++)
            {
                var ci = c + i * ldc;
                for (var p = 0; p < k; p++)
                {
                    var aip = alpha * a[i * rowStrideA + p * colStrideA];
                    if (aip == 0.0) continue;
                    var bp = b + p * rowStrideB;
                    for (var j = 0; j < n; j++) ci[j] += aip * bp[j * colStrideB];
                }
            }
            return;
        }

        var nr = Nr;
        var parallel = allowParallel && (long)m * n * k >= ParallelThreshold && Environment.ProcessorCount > 1;
        var ncMax = Math.Min(n, Nc);
        var kcMax = Math.Min(k, Kc);
        var panelsMax = (ncMax + nr - 1) / nr;
        var packedB = ArrayPool<double>.Shared.Rent(panelsMax * nr * kcMax);
        try
        {
            fixed (double* pbBase = packedB)
            {
                var pb = pbBase;
                for (var jc = 0; jc < n; jc += Nc)
                {
                    var nc = Math.Min(Nc, n - jc);
                    var panels = (nc + nr - 1) / nr;
                    for (var pc = 0; pc < k; pc += Kc)
                    {
                        var kc = Math.Min(Kc, k - pc);
                        var bSlice = b + pc * rowStrideB + jc * colStrideB;
                        var aSlice = a + pc * colStrideA;
                        var cBlock = c + jc;

                        if (!parallel)
                        {
                            PackB(bSlice, rowStrideB, colStrideB, pb, kc, nc, 0, panels);
                            for (var ic = 0; ic < m; ic += Mc)
                                RowBlock(aSlice, rowStrideA, colStrideA, pb, cBlock, ldc, ic, Math.Min(Mc, m - ic),
                                    0, panels, nc, kc, alpha);
                            continue;
                        }

                        var workers = Environment.ProcessorCount;
                        var bS = (nint)bSlice; var pbS = (nint)pb; var aS = (nint)aSlice; var cS = (nint)cBlock;
                        var kcL = kc; var ncL = nc;

                        // Packing B is a pass over kc×nc; split it by panels when it is big
                        // enough that a serial copy would show up next to the arithmetic.
                        if ((long)kc * nc >= 64 * 1024)
                        {
                            var chunks = Math.Min(workers, panels);
                            var per = (panels + chunks - 1) / chunks;
                            Parallel.For(0, chunks, t =>
                            {
                                var p0 = t * per;
                                var p1 = Math.Min(panels, p0 + per);
                                if (p0 < p1) PackB((double*)bS, rowStrideB, colStrideB, (double*)pbS, kcL, ncL, p0, p1);
                            });
                        }
                        else
                        {
                            PackB(bSlice, rowStrideB, colStrideB, pb, kc, nc, 0, panels);
                        }

                        // Rows first; columns as well only when rows alone cannot occupy every core.
                        var rowsPer = Math.Max(Mr, Math.Min(Mc, RoundUp((m + workers - 1) / workers, Mr)));
                        var rowBlocks = (m + rowsPer - 1) / rowsPer;
                        var colChunks = Math.Max(1, Math.Min(panels, (2 * workers + rowBlocks - 1) / rowBlocks));
                        var panelsPer = (panels + colChunks - 1) / colChunks;
                        colChunks = (panels + panelsPer - 1) / panelsPer;
                        var tasks = rowBlocks * colChunks;
                        Parallel.For(0, tasks, t =>
                        {
                            var rb = t / colChunks;
                            var cb = t % colChunks;
                            var ic = rb * rowsPer;
                            var p0 = cb * panelsPer;
                            var p1 = Math.Min(panels, p0 + panelsPer);
                            RowBlock((double*)aS, rowStrideA, colStrideA, (double*)pbS, (double*)cS, ldc, ic,
                                Math.Min(rowsPer, m - ic), p0, p1, ncL, kcL, alpha);
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
    private static void RowBlock(double* a, long rsA, long csA, double* pb, double* c, long ldc, int ic, int mc,
        int p0, int p1, int nc, int kc, double alpha)
    {
        var nr = Nr;
        var mPanels = (mc + Mr - 1) / Mr;
        var need = mPanels * Mr * kc;
        var buf = _packA;
        if (buf is null || buf.Length < need) _packA = buf = new double[Math.Max(need, Mc * Kc)];
        fixed (double* pa = buf)
        {
            PackA(a + ic * rsA, rsA, csA, pa, mc, kc);
            var tile = stackalloc double[Mr * nr];
            for (var jp = p0; jp < p1; jp++)
            {
                var j = jp * nr;
                var cols = Math.Min(nr, nc - j);
                var bPanel = pb + (long)jp * nr * kc;
                for (var ip = 0; ip < mPanels; ip++)
                {
                    var i = ip * Mr;
                    var rows = Math.Min(Mr, mc - i);
                    var cTile = c + (ic + i) * ldc + j;
                    if (rows == Mr && cols == nr)
                    {
                        Kernel(pa + (long)ip * Mr * kc, bPanel, kc, alpha, cTile, ldc);
                    }
                    else
                    {
                        // Edge tile: the panels are zero-padded, so the kernel runs unchanged into
                        // a scratch tile and only the valid part is added to C.
                        new Span<double>(tile, Mr * nr).Clear();
                        Kernel(pa + (long)ip * Mr * kc, bPanel, kc, alpha, tile, nr);
                        for (var r = 0; r < rows; r++)
                            for (var q = 0; q < cols; q++) cTile[r * ldc + q] += tile[r * nr + q];
                    }
                }
            }
        }
    }

    /// <summary>
    /// Copies panels [p0, p1) of a kc×nc block of B into <c>Nr</c>-wide, k-major panels,
    /// zero-padding the ragged last one.
    /// </summary>
    private static void PackB(double* b, long rs, long cs, double* dst, int kc, int nc, int p0, int p1)
    {
        var nr = Nr;
        var width = Vector<double>.Count;
        for (var jp = p0; jp < p1; jp++)
        {
            var j = jp * nr;
            var span = Math.Min(nr, nc - j);
            var d = dst + (long)jp * nr * kc;
            if (span == nr && cs == 1)
            {
                for (var p = 0; p < kc; p++)
                {
                    var src = b + p * rs + j;
                    Vector.Store(Vector.Load(src), d);
                    Vector.Store(Vector.Load(src + width), d + width);
                    d += nr;
                }
            }
            else if (rs == 1)
            {
                // B is a transposed view: walking k is the unit stride, so make it the inner loop.
                for (var t = 0; t < span; t++)
                {
                    var src = b + (j + t) * cs;
                    for (var p = 0; p < kc; p++) d[(long)p * nr + t] = src[p];
                }
                for (var t = span; t < nr; t++)
                    for (var p = 0; p < kc; p++) d[(long)p * nr + t] = 0.0;
            }
            else
            {
                for (var p = 0; p < kc; p++)
                {
                    var src = b + p * rs + j * cs;
                    for (var t = 0; t < span; t++) d[t] = src[t * cs];
                    for (var t = span; t < nr; t++) d[t] = 0.0;
                    d += nr;
                }
            }
        }
    }

    /// <summary>Copies an mc×kc block of A into <see cref="Mr"/>-row, k-major panels, zero-padding the tail.</summary>
    private static void PackA(double* a, long rs, long cs, double* dst, int mc, int kc)
    {
        for (var i = 0; i < mc; i += Mr)
        {
            var span = Math.Min(Mr, mc - i);
            var d = dst + (long)(i / Mr) * Mr * kc;
            if (span == Mr && cs == 1)
            {
                var r0 = a + i * rs; var r1 = r0 + rs; var r2 = r1 + rs; var r3 = r2 + rs;
                for (var p = 0; p < kc; p++)
                {
                    d[0] = r0[p]; d[1] = r1[p]; d[2] = r2[p]; d[3] = r3[p];
                    d += Mr;
                }
            }
            else if (span == Mr && rs == 1)
            {
                // A is a transposed view: the four rows of one k step are adjacent in memory.
                var src = a + i;
                for (var p = 0; p < kc; p++, src += cs)
                {
                    d[0] = src[0]; d[1] = src[1]; d[2] = src[2]; d[3] = src[3];
                    d += Mr;
                }
            }
            else
            {
                for (var p = 0; p < kc; p++)
                {
                    for (var t = 0; t < span; t++) d[t] = a[(i + t) * rs + p * cs];
                    for (var t = span; t < Mr; t++) d[t] = 0.0;
                    d += Mr;
                }
            }
        }
    }

    /// <summary>The register tile: <c>C[0..Mr, 0..Nr] += alpha · Apanel · Bpanel</c> over kc.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel(double* a, double* b, int kc, double alpha, double* c, long ldc)
    {
        var w = Vector<double>.Count;
        var nr = 2 * w;
        Vector<double> c00 = default, c01 = default, c10 = default, c11 = default;
        Vector<double> c20 = default, c21 = default, c30 = default, c31 = default;
        if (HasFma)
        {
            for (var p = 0; p < kc; p++)
            {
                var b0 = Vector.Load(b);
                var b1 = Vector.Load(b + w);
                var a0 = new Vector<double>(a[0]);
                c00 = Vector.FusedMultiplyAdd(a0, b0, c00); c01 = Vector.FusedMultiplyAdd(a0, b1, c01);
                var a1 = new Vector<double>(a[1]);
                c10 = Vector.FusedMultiplyAdd(a1, b0, c10); c11 = Vector.FusedMultiplyAdd(a1, b1, c11);
                var a2 = new Vector<double>(a[2]);
                c20 = Vector.FusedMultiplyAdd(a2, b0, c20); c21 = Vector.FusedMultiplyAdd(a2, b1, c21);
                var a3 = new Vector<double>(a[3]);
                c30 = Vector.FusedMultiplyAdd(a3, b0, c30); c31 = Vector.FusedMultiplyAdd(a3, b1, c31);
                a += Mr; b += nr;
            }
        }
        else
        {
            for (var p = 0; p < kc; p++)
            {
                var b0 = Vector.Load(b);
                var b1 = Vector.Load(b + w);
                var a0 = new Vector<double>(a[0]); c00 += a0 * b0; c01 += a0 * b1;
                var a1 = new Vector<double>(a[1]); c10 += a1 * b0; c11 += a1 * b1;
                var a2 = new Vector<double>(a[2]); c20 += a2 * b0; c21 += a2 * b1;
                var a3 = new Vector<double>(a[3]); c30 += a3 * b0; c31 += a3 * b1;
                a += Mr; b += nr;
            }
        }

        if (alpha != 1.0)
        {
            var al = new Vector<double>(alpha);
            c00 *= al; c01 *= al; c10 *= al; c11 *= al; c20 *= al; c21 *= al; c30 *= al; c31 *= al;
        }
        Accumulate(c, c00, c01, w);
        Accumulate(c + ldc, c10, c11, w);
        Accumulate(c + 2 * ldc, c20, c21, w);
        Accumulate(c + 3 * ldc, c30, c31, w);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Accumulate(double* c, Vector<double> lo, Vector<double> hi, int w)
    {
        Vector.Store(Vector.Load(c) + lo, c);
        Vector.Store(Vector.Load(c + w) + hi, c + w);
    }

    /// <summary>
    /// The scalar reference: a plain triple loop over strided operands, <c>C = A B</c>. Not used by
    /// any dispatch; it is what the tests and benchmarks check every fast path against.
    /// </summary>
    public static NdArray Reference(NdArray a, NdArray b)
    {
        if (a.Rank != 2 || b.Rank != 2 || a.Shape[1] != b.Shape[0])
            throw new ArgumentException("Reference expects aligned rank 2 operands.");
        int m = a.Shape[0], k = a.Shape[1], n = b.Shape[1];
        var c = NdArray.Zeros(m, n);
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
            {
                var s = 0.0;
                for (var p = 0; p < k; p++) s += a[i, p] * b[p, j];
                c[i, j] = s;
            }
        return c;
    }
}
