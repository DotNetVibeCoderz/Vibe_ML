using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// <c>C = alpha·A·B + beta·C</c> on row-major buffers with leading dimensions, so it can update a
/// sub-block in place (the trailing update of a blocked LU, the off-diagonal update of a blocked
/// triangular solve).
/// </summary>
/// <remarks>
/// <para>
/// This used to be SciNet's own packed 4×8 FMA kernel, written because GraviNum's <c>LinAlg.Dot</c>
/// ran an unpacked, FMA-less kernel below eight million multiply-adds (a 200-cube took 4.7 ms).
/// That kernel has since moved into GraviNum as <see cref="PackedMatMul.Gemm"/> — with strided
/// operands and a parallel B pack — and this class forwards to it. Measured in one process the two
/// were bit-identical and within noise of each other (0.93–1.07x from 32 to 1024), so the
/// duplicate went.
/// </para>
/// <para>
/// When a native BLAS is installed, whole-matrix products go through <c>LinAlg.Dot</c> (which
/// calls it); the blocked factorisations always use the managed kernel.
/// </para>
/// </remarks>
internal static unsafe class LinalgGemm
{
    /// <summary>Rows of the micro-kernel tile.</summary>
    internal const int Mr = PackedMatMul.Mr;

    /// <summary>Column granularity callers round work chunks to: one AVX2 tile width.</summary>
    internal const int Nr = 8;

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
        => PackedMatMul.Gemm(m, n, k, alpha, a, lda, 1, b, ldb, 1, beta, c, ldc, allowParallel);

    /// <summary>
    /// Whole-matrix product into a new <see cref="NdArray"/>: <c>LinAlg.Dot</c>, which picks native
    /// BLAS, the packed kernel or the small blocked one by shape.
    /// </summary>
    public static NdArray Mul(NdArray a, NdArray b)
    {
        if (a.Rank == 2 && b.Rank == 2 && b.Shape[0] != a.Shape[1])
            throw new ArgumentException(
                $"Matrix product shape mismatch: ({a.Shape[0]}x{a.Shape[1]}) @ ({b.Shape[0]}x{b.Shape[1]}).");
        return LinAlg.Dot(a, b);
    }
}
