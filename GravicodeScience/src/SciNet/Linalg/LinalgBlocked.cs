namespace Gravicode.Science.SciNet;

/// <summary>
/// Level-3 LU factorisation and triangular solves with many right-hand sides, built on
/// <see cref="LinalgGemm"/>.
/// </summary>
/// <remarks>
/// <para>
/// The unblocked kernels in <see cref="LinalgKernels"/> do every update as an axpy over a row, so
/// each element of the trailing matrix is read and written once per pivot: n passes over an n² array.
/// Above about a hundred rows that array no longer fits in L2, and the factorisation runs at memory
/// speed. Here almost all of the arithmetic is matrix products whose operands are packed and
/// cache-resident.
/// </para>
/// <para>
/// The LU is the recursive one of LAPACK's <c>dgetrf2</c> (Toledo 1997) applied to the whole
/// matrix: factor the left half of the columns, solve for the top-right block, update the
/// bottom-right with one product, recurse. A right-looking blocked LU with an unblocked panel was
/// tried first; at n = 200 its panel — n·nb/2 short axpys on one core — was most of the runtime.
/// Recursion turns the panel into products too.
/// </para>
/// <para>
/// Same pivots as the unblocked version in exact arithmetic (partial pivoting by the largest
/// magnitude in the column, row swaps applied to whole rows as <c>getrf</c> does); the rounding of
/// the updates differs because the sums are accumulated in a different order.
/// </para>
/// </remarks>
internal static unsafe class LinalgBlocked
{
    /// <summary>Block height for the triangular solves.</summary>
    internal const int BlockSize = 48;

    /// <summary>Order at and above which <see cref="LinalgKernels.LuInPlace"/> switches to the recursive form.</summary>
    internal const int LuThreshold = 64;

    /// <summary>Column count at which the recursion stops and eliminates column by column.</summary>
    private const int LuLeaf = 8;

    /// <summary>
    /// In-place recursive LU with partial pivoting (LAPACK <c>getrf</c> semantics, 0-based swaps).
    /// Returns -1 on success, otherwise the first column with an exactly zero pivot.
    /// </summary>
    public static int Lu(Span<double> a, int n, int[] swaps)
    {
        var firstZero = -1;
        fixed (double* p = a)
            LuRecursive(p, n, 0, n, n, swaps, ref firstZero);
        return firstZero;
    }

    /// <summary>
    /// Factors the m×w block whose top-left corner is the diagonal entry (c0, c0), m = n - c0,
    /// swapping whole rows of the n×n matrix as pivots are chosen.
    /// </summary>
    private static void LuRecursive(double* p, int n, int c0, int m, int w, int[] swaps, ref int firstZero)
    {
        if (w <= LuLeaf)
        {
            LuLeafColumns(p, n, c0, m, w, swaps, ref firstZero);
            return;
        }
        var n1 = w / 2;
        var n2 = w - n1;
        LuRecursive(p, n, c0, m, n1, swaps, ref firstZero);

        // A12 ← L11⁻¹ A12, then A22 ← A22 − A21 A12, then factor A22.
        var a11 = p + (long)c0 * n + c0;
        var a12 = a11 + n1;
        var a21 = a11 + (long)n1 * n;
        var a22 = a21 + n1;
        Striped(a11, n, n1, a12, n, n2, lower: true);
        LinalgGemm.Gemm(m - n1, n2, n1, -1.0, a21, n, a12, n, 1.0, a22, n);
        LuRecursive(p, n, c0 + n1, m - n1, n2, swaps, ref firstZero);
    }

    /// <summary>Column-by-column elimination of a narrow block (columns [c0, c0 + w), rows [c0, n)).</summary>
    private static void LuLeafColumns(double* p, int n, int c0, int m, int w, int[] swaps, ref int firstZero)
    {
        var end = c0 + w;
        for (var j = c0; j < end; j++)
        {
            var piv = j;
            var best = Math.Abs(p[(long)j * n + j]);
            for (var i = j + 1; i < c0 + m; i++)
            {
                var v = Math.Abs(p[(long)i * n + j]);
                if (v > best) { best = v; piv = i; }
            }
            swaps[j] = piv;
            if (piv != j)
                new Span<double>(p + (long)j * n, n).SwapWith(new Span<double>(p + (long)piv * n, n));

            var pivot = p[(long)j * n + j];
            if (pivot == 0.0)
            {
                if (firstZero < 0) firstZero = j;
                continue;
            }
            var inv = 1.0 / pivot;
            var width = end - j - 1;
            var pivotRow = p + (long)j * n + j + 1;
            for (var i = j + 1; i < c0 + m; i++)
            {
                var row = p + (long)i * n;
                var l = row[j] * inv;
                row[j] = l;
                if (l != 0.0)
                    for (var q = 0; q < width; q++) row[j + 1 + q] -= l * pivotRow[q];
            }
        }
    }

    /// <summary>
    /// Solves <c>L X = B</c> in place for unit lower-triangular L (read from the strict lower part
    /// of <paramref name="t"/>), B row-major n×k.
    /// </summary>
    /// <remarks>
    /// The columns of X are independent, so with many right-hand sides they are split into
    /// stripes solved in parallel, each by the sequential blocked algorithm. Parallelising inside
    /// one solve instead would leave the diagonal-block axpys — a fixed n·nb/2 of them —
    /// running on one core, and at n = 200 those were most of the time.
    /// </remarks>
    public static void LowerUnitSolve(ReadOnlySpan<double> t, int n, Span<double> b, int k)
    {
        fixed (double* tp = t, bp = b)
            Striped(tp, n, n, bp, k, k, lower: true);
    }

    /// <summary>
    /// Solves <c>U X = B</c> in place for upper-triangular U (diagonal included), B row-major n×k.
    /// Throws <see cref="LinalgSingularMatrixException"/> on an exactly zero diagonal entry.
    /// </summary>
    public static void UpperSolve(ReadOnlySpan<double> t, int n, Span<double> b, int k)
    {
        for (var i = 0; i < n; i++)
            if (t[i * n + i] == 0.0) throw new LinalgSingularMatrixException("Matrix is singular: zero on the diagonal of a triangular factor.");
        fixed (double* tp = t, bp = b)
            Striped(tp, n, n, bp, k, k, lower: false);
    }

    /// <summary>
    /// Triangular solve with the order-<paramref name="order"/> triangle at <paramref name="tp"/>
    /// (row stride <paramref name="ldt"/>) against <paramref name="k"/> columns of B (row stride
    /// <paramref name="ldb"/>), split into parallel column stripes when large enough.
    /// </summary>
    private static void Striped(double* tp, int ldt, int order, double* bp, int ldb, int k, bool lower)
    {
        var workers = Environment.ProcessorCount;
        var flops = (long)order * order * k;
        if (workers == 1 || flops < 64L * 64 * 64 || k < 32)
        {
            if (lower) LowerStripe(tp, ldt, order, bp, ldb, k);
            else UpperStripe(tp, ldt, order, bp, ldb, k);
            return;
        }
        // Stripes a multiple of the kernel width, at least 16 wide so the axpys still vectorise.
        var width = Math.Max(16, (k / workers + LinalgGemm.Nr - 1) / LinalgGemm.Nr * LinalgGemm.Nr);
        var stripes = (k + width - 1) / width;
        var tS = (nint)tp; var bS = (nint)bp;
        Parallel.For(0, stripes, s =>
        {
            var c0 = s * width;
            var w = Math.Min(width, k - c0);
            if (lower) LowerStripe((double*)tS, ldt, order, (double*)bS + c0, ldb, w);
            else UpperStripe((double*)tS, ldt, order, (double*)bS + c0, ldb, w);
        });
    }

    /// <summary>Sequential blocked unit-lower solve on <paramref name="w"/> columns of B.</summary>
    private static void LowerStripe(double* tp, int ldt, int n, double* bp, int ldb, int w)
    {
        const int nb = BlockSize;
        for (var i0 = 0; i0 < n; i0 += nb)
        {
            var i1 = Math.Min(n, i0 + nb);
            for (var i = i0 + 1; i < i1; i++)
            {
                var xi = bp + (long)i * ldb;
                var ti = tp + (long)i * ldt;
                for (var j = i0; j < i; j++)
                {
                    var tij = ti[j];
                    if (tij != 0.0) Axpy(-tij, bp + (long)j * ldb, xi, w);
                }
            }
            if (i1 < n)
                LinalgGemm.Gemm(n - i1, w, i1 - i0, -1.0, tp + (long)i1 * ldt + i0, ldt, bp + (long)i0 * ldb, ldb,
                    1.0, bp + (long)i1 * ldb, ldb, allowParallel: false);
        }
    }

    /// <summary>Sequential blocked upper solve on <paramref name="w"/> columns of B; the diagonal must be non-zero.</summary>
    private static void UpperStripe(double* tp, int ldt, int n, double* bp, int ldb, int w)
    {
        const int nb = BlockSize;
        for (var i1 = n; i1 > 0; i1 -= nb)
        {
            var i0 = Math.Max(0, i1 - nb);
            for (var i = i1 - 1; i >= i0; i--)
            {
                var xi = bp + (long)i * ldb;
                var ti = tp + (long)i * ldt;
                for (var j = i + 1; j < i1; j++)
                {
                    var tij = ti[j];
                    if (tij != 0.0) Axpy(-tij, bp + (long)j * ldb, xi, w);
                }
                var inv = 1.0 / ti[i];
                for (var q = 0; q < w; q++) xi[q] *= inv;
            }
            if (i0 > 0)
                LinalgGemm.Gemm(i0, w, i1 - i0, -1.0, tp + i0, ldt, bp + (long)i0 * ldb, ldb, 1.0, bp, ldb, allowParallel: false);
        }
    }

    /// <summary><c>y += alpha x</c> over raw pointers, four lanes at a time.</summary>
    internal static void Axpy(double alpha, double* x, double* y, int len)
    {
        var i = 0;
        var va = System.Runtime.Intrinsics.Vector256.Create(alpha);
        for (; i <= len - 4; i += 4)
            System.Runtime.Intrinsics.Vector256.Store(
                System.Runtime.Intrinsics.Vector256.Load(y + i) + va * System.Runtime.Intrinsics.Vector256.Load(x + i), y + i);
        for (; i < len; i++) y[i] += alpha * x[i];
    }
}
