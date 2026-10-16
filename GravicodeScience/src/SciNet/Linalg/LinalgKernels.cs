using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Span-level kernels shared by the <see cref="Linalg"/> facade: LU and Cholesky on flat
/// row-major buffers, triangular solves, and the handful of matrix helpers every routine needs.
/// </summary>
/// <remarks>
/// <para>
/// GraviNum's <c>Decomposition.Lu</c> and <c>Decomposition.Cholesky</c> route to native LAPACK
/// above 64 rows and are used there. Below that, and whenever no LAPACK is installed, their
/// managed paths walk the matrix through the <see cref="NdArray"/> indexer, whose stride maths
/// and bounds checks cost roughly 4x inside an O(n³) loop. The scipy.linalg surface also wants
/// the <em>packed</em> LU with LAPACK-style swap pivots (<c>lu_factor</c>), which the unpacked
/// GraviNum record would have to be re-packed into. So the managed fallback lives here, written
/// row-wise so every inner loop is a vectorised axpy over a contiguous row.
/// </para>
/// </remarks>
internal static class LinalgKernels
{
    /// <summary>Machine epsilon for double, <c>2^-52</c>.</summary>
    public const double Eps = 2.220446049250313e-16;

    public static void RequireSquare(NdArray a, string caller)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1])
            throw new ArgumentException($"{caller} requires a square 2-D matrix, got rank {a.Rank} with shape [{string.Join(", ", a.Shape.ToArray())}].");
    }

    public static void RequireMatrix(NdArray a, string caller)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Rank != 2)
            throw new ArgumentException($"{caller} requires a 2-D matrix, got rank {a.Rank}.");
    }

    public static void RequireFinite(ReadOnlySpan<double> a, string caller)
    {
        foreach (var v in a)
            if (!double.IsFinite(v))
                throw new ArgumentException($"{caller}: the array must not contain infs or NaNs.");
    }

    /// <summary>A contiguous row-major copy of a matrix.</summary>
    public static double[] Buffer(NdArray a) => a.ToArray();

    public static NdArray Identity(int n) => NdArray.Eye(n);

    public static double[] IdentityBuffer(int n)
    {
        var r = new double[n * n];
        for (var i = 0; i < n; i++) r[i * n + i] = 1.0;
        return r;
    }

    /// <summary>Matrix product through GraviNum's blocked, parallel <c>LinAlg.Dot</c>.</summary>
    public static NdArray Mul(NdArray a, NdArray b) => LinAlg.Dot(a, b);

    public static double[] Mul(double[] a, double[] b, int m, int k, int n)
        => LinAlg.Dot(new NdArray(a, m, k), new NdArray(b, k, n)).ToArray();

    /// <summary><c>alpha*a + beta*b</c> element-wise into a new array.</summary>
    public static NdArray Combine(double alpha, NdArray a, double beta, NdArray b)
    {
        var x = a.AsContiguous().AsSpan();
        var y = b.AsContiguous().AsSpan();
        var r = new double[x.Length];
        for (var i = 0; i < r.Length; i++) r[i] = alpha * x[i] + beta * y[i];
        return new NdArray(r, a.Shape.ToArray());
    }

    /// <summary>Adds <paramref name="alpha"/> times the identity to a square buffer in place.</summary>
    public static void AddDiagonal(Span<double> a, int n, double alpha)
    {
        for (var i = 0; i < n; i++) a[i * n + i] += alpha;
    }

    /// <summary>Matrix 1-norm: the largest absolute column sum.</summary>
    public static double OneNorm(ReadOnlySpan<double> a, int m, int n)
    {
        var sums = n <= 256 ? stackalloc double[n] : new double[n];
        sums.Clear();
        for (var i = 0; i < m; i++)
        {
            var row = a.Slice(i * n, n);
            for (var j = 0; j < n; j++) sums[j] += Math.Abs(row[j]);
        }
        var best = 0.0;
        foreach (var s in sums) best = Math.Max(best, s);
        return best;
    }

    public static double OneNorm(NdArray a) => OneNorm(a.AsContiguous().AsSpan(), a.Shape[0], a.Shape[1]);

    /// <summary>Infinity norm: the largest absolute row sum.</summary>
    public static double InfNorm(ReadOnlySpan<double> a, int m, int n)
    {
        var best = 0.0;
        for (var i = 0; i < m; i++)
        {
            var s = 0.0;
            foreach (var v in a.Slice(i * n, n)) s += Math.Abs(v);
            best = Math.Max(best, s);
        }
        return best;
    }

    public static double FrobeniusNorm(ReadOnlySpan<double> a)
    {
        // Scaled sum of squares so a matrix of 1e200s does not overflow.
        var scale = 0.0;
        var ssq = 1.0;
        foreach (var v in a)
        {
            if (v == 0.0) continue;
            var av = Math.Abs(v);
            if (scale < av) { ssq = 1.0 + ssq * (scale / av) * (scale / av); scale = av; }
            else ssq += (av / scale) * (av / scale);
        }
        return scale * Math.Sqrt(ssq);
    }

    // ================================================================ LU

    /// <summary>
    /// In-place LU with partial pivoting (LAPACK <c>getrf</c> semantics, 0-based).
    /// </summary>
    /// <param name="a">Row-major <c>n×n</c>; overwritten with L (unit, below) and U (on/above).</param>
    /// <param name="n">Order.</param>
    /// <param name="swaps">Receives the row <em>swaps</em>: at step i, row i was exchanged with row swaps[i].</param>
    /// <returns>-1 on success, otherwise the first column with an exactly zero pivot.</returns>
    public static int LuInPlace(Span<double> a, int n, int[] swaps)
    {
        var firstZero = -1;
        for (var k = 0; k < n; k++)
        {
            var p = k;
            var best = Math.Abs(a[k * n + k]);
            for (var i = k + 1; i < n; i++)
            {
                var v = Math.Abs(a[i * n + k]);
                if (v > best) { best = v; p = i; }
            }
            swaps[k] = p;
            if (p != k) a.Slice(k * n, n).SwapWith(a.Slice(p * n, n));

            var pivot = a[k * n + k];
            if (pivot == 0.0)
            {
                if (firstZero < 0) firstZero = k;
                continue;
            }

            var inv = 1.0 / pivot;
            var tail = n - k - 1;
            if (tail == 0) continue;
            var pivotRow = a.Slice(k * n + k + 1, tail);
            for (var i = k + 1; i < n; i++)
            {
                var l = a[i * n + k] * inv;
                a[i * n + k] = l;
                if (l != 0.0) LinAlg.AxpySpan(-l, pivotRow, a.Slice(i * n + k + 1, tail));
            }
        }
        return firstZero;
    }

    /// <summary>
    /// LU factorisation into the packed form, using native LAPACK through GraviNum when it is
    /// installed and large enough to win.
    /// </summary>
    public static (double[] Lu, int[] Swaps, int FirstZero) LuFactor(NdArray a)
    {
        var n = a.Shape[0];
        if (Gravicode.Science.GraviNum.Compute.NativeLapack.ShouldUse(n))
        {
            var lu = Decomposition.Lu(a);
            var packed = new double[n * n];
            var l = lu.Lower.AsContiguous().AsSpan();
            var u = lu.Upper.AsContiguous().AsSpan();
            var zero = -1;
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < i; j++) packed[i * n + j] = l[i * n + j];
                for (var j = i; j < n; j++) packed[i * n + j] = u[i * n + j];
                if (zero < 0 && u[i * n + i] == 0.0) zero = i;
            }
            return (packed, PermutationToSwaps(lu.Pivot), zero);
        }

        var buf = Buffer(a);
        var swaps = new int[n];
        var first = LuInPlace(buf, n, swaps);
        return (buf, swaps, first);
    }

    /// <summary>
    /// Converts a finished permutation (row i of PA is row perm[i] of A) into the sequence of
    /// swaps that produces it — the inverse of what <c>dgetrf</c> reports.
    /// </summary>
    public static int[] PermutationToSwaps(int[] perm)
    {
        var n = perm.Length;
        var current = new int[n];      // current[i] = original row now sitting at position i
        var where = new int[n];        // where[r] = position of original row r
        for (var i = 0; i < n; i++) { current[i] = i; where[i] = i; }
        var swaps = new int[n];
        for (var i = 0; i < n; i++)
        {
            var want = perm[i];
            var at = where[want];
            swaps[i] = at;
            if (at != i)
            {
                var displaced = current[i];
                current[i] = want; current[at] = displaced;
                where[want] = i; where[displaced] = at;
            }
        }
        return swaps;
    }

    /// <summary>Replays the swaps: <c>perm[i]</c> is the original row of output row <c>i</c>.</summary>
    public static int[] SwapsToPermutation(int[] swaps)
    {
        var n = swaps.Length;
        var perm = new int[n];
        for (var i = 0; i < n; i++) perm[i] = i;
        for (var i = 0; i < n; i++)
            if (swaps[i] != i) (perm[i], perm[swaps[i]]) = (perm[swaps[i]], perm[i]);
        return perm;
    }

    /// <summary>Solves <c>A X = B</c> (or <c>Aᵀ X = B</c>) from a packed LU, overwriting <paramref name="b"/>.</summary>
    public static void LuSolveInPlace(ReadOnlySpan<double> lu, int[] swaps, int n, Span<double> b, int k, bool transpose)
    {
        if (!transpose)
        {
            for (var i = 0; i < n; i++)
                if (swaps[i] != i) b.Slice(i * k, k).SwapWith(b.Slice(swaps[i] * k, k));
            TriangularSolveInPlace(lu, n, b, k, lower: true, transpose: false, unitDiagonal: true);
            TriangularSolveInPlace(lu, n, b, k, lower: false, transpose: false, unitDiagonal: false);
        }
        else
        {
            // (P^T L U)^T x = b  ->  U^T L^T P x = b.
            TriangularSolveInPlace(lu, n, b, k, lower: false, transpose: true, unitDiagonal: false);
            TriangularSolveInPlace(lu, n, b, k, lower: true, transpose: true, unitDiagonal: true);
            for (var i = n - 1; i >= 0; i--)
                if (swaps[i] != i) b.Slice(i * k, k).SwapWith(b.Slice(swaps[i] * k, k));
        }
    }

    /// <summary>
    /// Triangular solve with <c>k</c> right-hand sides stored row-major in <paramref name="b"/>,
    /// reading only the requested triangle of <paramref name="t"/>.
    /// </summary>
    /// <remarks>
    /// Every update is an axpy of one solved row of X into another row of B, so the inner loop
    /// runs along contiguous memory whichever triangle and orientation is asked for.
    /// </remarks>
    public static void TriangularSolveInPlace(
        ReadOnlySpan<double> t, int n, Span<double> b, int k, bool lower, bool transpose, bool unitDiagonal)
    {
        // Effective lower-triangular solve when (lower && !transpose) or (!lower && transpose).
        var forward = lower != transpose;
        if (forward)
        {
            for (var i = 0; i < n; i++)
            {
                var xi = b.Slice(i * k, k);
                if (!transpose)
                {
                    // x_i = (b_i - sum_{j<i} T[i,j] x_j) / T[i,i]
                    for (var j = 0; j < i; j++)
                    {
                        var tij = t[i * n + j];
                        if (tij != 0.0) LinAlg.AxpySpan(-tij, b.Slice(j * k, k), xi);
                    }
                    if (!unitDiagonal) Divide(xi, t[i * n + i]);
                }
                else
                {
                    // Upper T, solving T^T x = b: x_i final once earlier rows pushed into it.
                    if (!unitDiagonal) Divide(xi, t[i * n + i]);
                    for (var j = i + 1; j < n; j++)
                    {
                        var tij = t[i * n + j];
                        if (tij != 0.0) LinAlg.AxpySpan(-tij, xi, b.Slice(j * k, k));
                    }
                }
            }
        }
        else
        {
            for (var i = n - 1; i >= 0; i--)
            {
                var xi = b.Slice(i * k, k);
                if (!transpose)
                {
                    for (var j = i + 1; j < n; j++)
                    {
                        var tij = t[i * n + j];
                        if (tij != 0.0) LinAlg.AxpySpan(-tij, b.Slice(j * k, k), xi);
                    }
                    if (!unitDiagonal) Divide(xi, t[i * n + i]);
                }
                else
                {
                    // Lower T, solving T^T x = b (upper system), column i of T^T is row i of T.
                    if (!unitDiagonal) Divide(xi, t[i * n + i]);
                    for (var j = 0; j < i; j++)
                    {
                        var tij = t[i * n + j];
                        if (tij != 0.0) LinAlg.AxpySpan(-tij, xi, b.Slice(j * k, k));
                    }
                }
            }
        }
    }

    private static void Divide(Span<double> x, double d)
    {
        if (d == 0.0) throw new LinalgSingularMatrixException("Matrix is singular: zero on the diagonal of a triangular factor.");
        var inv = 1.0 / d;
        for (var i = 0; i < x.Length; i++) x[i] *= inv;
    }

    /// <summary>Reciprocal condition estimate from the diagonal of U, cheap and only a warning signal.</summary>
    public static double DiagonalRatio(ReadOnlySpan<double> lu, int n)
    {
        var min = double.PositiveInfinity;
        var max = 0.0;
        for (var i = 0; i < n; i++)
        {
            var v = Math.Abs(lu[i * n + i]);
            min = Math.Min(min, v);
            max = Math.Max(max, v);
        }
        return max == 0.0 ? 0.0 : min / max;
    }

    // ================================================================ Cholesky

    /// <summary>
    /// Lower Cholesky factor of a symmetric positive-definite buffer, in place; the strict upper
    /// triangle is zeroed. Returns -1 on success or the index of the failing leading minor.
    /// </summary>
    public static int CholeskyInPlace(Span<double> a, int n)
    {
        for (var j = 0; j < n; j++)
        {
            var rowJ = a.Slice(j * n, j);
            var d = a[j * n + j] - LinAlg.InnerSpan(rowJ, rowJ);
            if (!(d > 0.0)) return j;
            d = Math.Sqrt(d);
            a[j * n + j] = d;
            var inv = 1.0 / d;
            for (var i = j + 1; i < n; i++)
                a[i * n + j] = (a[i * n + j] - LinAlg.InnerSpan(a.Slice(i * n, j), rowJ)) * inv;
        }
        for (var i = 0; i < n; i++)
            a.Slice(i * n + i + 1, n - i - 1).Clear();
        return -1;
    }

    /// <summary>Lower Cholesky factor, native when GraviNum would use LAPACK; throws when not positive definite.</summary>
    public static double[] CholeskyLower(NdArray a, string caller)
    {
        var n = a.Shape[0];
        if (Gravicode.Science.GraviNum.Compute.NativeLapack.ShouldUse(n))
        {
            try { return Decomposition.Cholesky(a).ToArray(); }
            catch (InvalidOperationException ex) { throw new LinalgException($"{caller}: {ex.Message}"); }
        }
        var buf = Buffer(a);
        // Read the lower triangle only, as LAPACK does.
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++) buf[i * n + j] = 0.0;
        var info = CholeskyInPlace(buf, n);
        if (info >= 0)
            throw new LinalgException($"{caller}: the leading minor of order {info + 1} is not positive definite.");
        return buf;
    }

    // ================================================================ misc

    public static double[] Transpose(ReadOnlySpan<double> a, int m, int n)
    {
        var r = new double[m * n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++) r[j * m + i] = a[i * n + j];
        return r;
    }

    public static void Symmetrize(Span<double> a, int n)
    {
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
            {
                var v = 0.5 * (a[i * n + j] + a[j * n + i]);
                a[i * n + j] = v; a[j * n + i] = v;
            }
    }

    public static bool IsUpperTriangular(ReadOnlySpan<double> a, int n)
    {
        for (var i = 1; i < n; i++)
            for (var j = 0; j < i; j++)
                if (a[i * n + j] != 0.0) return false;
        return true;
    }

    public static bool IsLowerTriangular(ReadOnlySpan<double> a, int n)
    {
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
                if (a[i * n + j] != 0.0) return false;
        return true;
    }

    /// <summary>Right-hand side as a row-major <c>n×k</c> buffer, remembering whether it was a vector.</summary>
    public static (double[] Data, int Columns, bool WasVector) Rhs(NdArray b, int n, string caller)
    {
        ArgumentNullException.ThrowIfNull(b);
        if (b.Rank == 1)
        {
            if (b.Size != n) throw new ArgumentException($"{caller}: right-hand side has length {b.Size}, expected {n}.");
            return (b.ToArray(), 1, true);
        }
        if (b.Rank != 2 || b.Shape[0] != n)
            throw new ArgumentException($"{caller}: right-hand side must have {n} rows.");
        return (b.ToArray(), b.Shape[1], false);
    }

    public static NdArray Shaped(double[] x, int n, int k, bool vector) => vector ? new NdArray(x, n) : new NdArray(x, n, k);

    // ================================================================ complex helpers

    public static Complex[] ToComplex(ReadOnlySpan<double> a)
    {
        var r = new Complex[a.Length];
        for (var i = 0; i < a.Length; i++) r[i] = a[i];
        return r;
    }

    /// <summary>Complex product C = A B (m×k times k×n), i-k-j order so the inner loop is contiguous.</summary>
    public static Complex[] CMul(Complex[] a, Complex[] b, int m, int k, int n)
    {
        var c = new Complex[m * n];
        void Row(int i)
        {
            var ci = c.AsSpan(i * n, n);
            for (var p = 0; p < k; p++)
            {
                var aip = a[i * k + p];
                if (aip == Complex.Zero) continue;
                var bp = b.AsSpan(p * n, n);
                for (var j = 0; j < n; j++) ci[j] += aip * bp[j];
            }
        }
        if ((long)m * k * n >= 1L << 18) Parallel.For(0, m, Row);
        else for (var i = 0; i < m; i++) Row(i);
        return c;
    }

    public static Complex[] CConjTranspose(Complex[] a, int m, int n)
    {
        var r = new Complex[m * n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++) r[j * m + i] = Complex.Conjugate(a[i * n + j]);
        return r;
    }

    /// <summary>Z F Z^H for unitary Z, the back-transform of a function of a Schur form.</summary>
    public static Complex[] CSandwich(Complex[] z, Complex[] f, int n)
        => CMul(CMul(z, f, n, n, n), CConjTranspose(z, n, n), n, n, n);

    /// <summary>
    /// Largest imaginary part relative to the largest magnitude; used to decide whether a result
    /// that should be real for real input actually is.
    /// </summary>
    public static double RelativeImaginary(Complex[] a)
    {
        var maxIm = 0.0;
        var maxAbs = 0.0;
        foreach (var v in a)
        {
            maxIm = Math.Max(maxIm, Math.Abs(v.Imaginary));
            maxAbs = Math.Max(maxAbs, Complex.Abs(v));
        }
        return maxAbs == 0.0 ? 0.0 : maxIm / maxAbs;
    }

    public static double[] RealPart(Complex[] a)
    {
        var r = new double[a.Length];
        for (var i = 0; i < a.Length; i++) r[i] = a[i].Real;
        return r;
    }

    /// <summary>Solves an upper-triangular complex system in place for one right-hand side.</summary>
    public static void CUpperSolve(ReadOnlySpan<Complex> u, int n, Span<Complex> x, Complex shift)
    {
        for (var i = n - 1; i >= 0; i--)
        {
            var s = x[i];
            var row = u.Slice(i * n, n);
            for (var j = i + 1; j < n; j++) s -= row[j] * x[j];
            var d = row[i] + shift;
            if (d == Complex.Zero) throw new LinalgSingularMatrixException("Singular triangular system.");
            x[i] = s / d;
        }
    }

    /// <summary>Real 2n×2n embedding <c>[[Re, -Im], [Im, Re]]</c> of a complex matrix.</summary>
    public static NdArray RealEmbedding(Complex[] a, int n)
    {
        var m = 2 * n;
        var r = new double[m * m];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                var v = a[i * n + j];
                r[i * m + j] = v.Real;
                r[i * m + j + n] = -v.Imaginary;
                r[(i + n) * m + j] = v.Imaginary;
                r[(i + n) * m + j + n] = v.Real;
            }
        return new NdArray(r, m, m);
    }

    public static Complex[] FromRealEmbedding(ReadOnlySpan<double> r, int n)
    {
        var m = 2 * n;
        var a = new Complex[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
                a[i * n + j] = new Complex(r[i * m + j], r[(i + n) * m + j]);
        return a;
    }
}

/// <summary>Span helpers for the Linalg kernels.</summary>
internal static class LinalgSpanExtensions
{
    /// <summary>Exchanges the contents of two equal-length spans.</summary>
    public static void SwapWith(this Span<double> a, Span<double> b)
    {
        for (var i = 0; i < a.Length; i++) (a[i], b[i]) = (b[i], a[i]);
    }
}
