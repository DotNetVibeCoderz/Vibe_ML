using System.Numerics;
using Gravicode.Science.GraviNum;
using GFft = Gravicode.Science.GraviNum.Signal.Fft;

namespace Gravicode.Science.SciNet;

public static partial class Linalg
{
    // ================================================================ dense solve

    /// <summary>
    /// Solves <c>A x = b</c> for square A, scipy's <c>solve</c>; b may be a vector or an n×k matrix.
    /// </summary>
    /// <param name="a">Square coefficient matrix.</param>
    /// <param name="b">Right-hand side(s).</param>
    /// <param name="assumeA">
    /// Structure to exploit. Null (scipy's default) detects it: diagonal, triangular, tridiagonal,
    /// then symmetric with positive diagonal (Cholesky tried, LU if it fails), else general LU.
    /// </param>
    /// <param name="lower">For symmetric/positive-definite structure, read the lower triangle; scipy's
    /// default (false) reads the upper one.</param>
    /// <param name="transposed">Solve <c>Aᵀ x = b</c> instead.</param>
    /// <exception cref="LinalgSingularMatrixException">A has an exactly zero pivot.</exception>
    public static NdArray Solve(NdArray a, NdArray b, LinalgMatrixStructure? assumeA = null, bool lower = false, bool transposed = false)
    {
        LinalgKernels.RequireSquare(a, nameof(Solve));
        var n = a.Shape[0];
        var aData = a.ToArray();
        LinalgKernels.RequireFinite(aData, nameof(Solve));
        if (transposed)
        {
            aData = LinalgKernels.Transpose(aData, n, n);
            a = new NdArray(aData, n, n);
            if (assumeA is LinalgMatrixStructure.UpperTriangular) assumeA = LinalgMatrixStructure.LowerTriangular;
            else if (assumeA is LinalgMatrixStructure.LowerTriangular) assumeA = LinalgMatrixStructure.UpperTriangular;
            else if (assumeA is LinalgMatrixStructure.Symmetric or LinalgMatrixStructure.Hermitian or LinalgMatrixStructure.PositiveDefinite)
                lower = !lower;
        }
        var (x, k, vec) = LinalgKernels.Rhs(b, n, nameof(Solve));
        LinalgKernels.RequireFinite(x, nameof(Solve));

        var structure = assumeA ?? Detect(aData, n);
        switch (structure)
        {
            case LinalgMatrixStructure.Diagonal:
                for (var i = 0; i < n; i++)
                {
                    var d = aData[i * n + i];
                    if (d == 0.0) throw new LinalgSingularMatrixException("Solve: singular diagonal matrix.");
                    for (var j = 0; j < k; j++) x[i * k + j] /= d;
                }
                break;
            case LinalgMatrixStructure.UpperTriangular:
                LinalgKernels.TriangularSolveInPlace(aData, n, x, k, lower: false, transpose: false, unitDiagonal: false);
                break;
            case LinalgMatrixStructure.LowerTriangular:
                LinalgKernels.TriangularSolveInPlace(aData, n, x, k, lower: true, transpose: false, unitDiagonal: false);
                break;
            case LinalgMatrixStructure.Tridiagonal:
                return SolveBanded(1, 1, DenseToBand(aData, n, 1, 1), b);
            case LinalgMatrixStructure.PositiveDefinite:
            {
                var l = LinalgKernels.CholeskyLower(new NdArray(SymmetricFromTriangle(a, lower), n, n), nameof(Solve));
                LinalgKernels.TriangularSolveInPlace(l, n, x, k, lower: true, transpose: false, unitDiagonal: false);
                LinalgKernels.TriangularSolveInPlace(l, n, x, k, lower: true, transpose: true, unitDiagonal: false);
                break;
            }
            case LinalgMatrixStructure.Symmetric:
            case LinalgMatrixStructure.Hermitian:
                return LdlSolve(Ldl(new NdArray(SymmetricFromTriangle(a, lower), n, n)), b);
            case LinalgMatrixStructure.General:
            default:
                return GeneralSolve(a, x, n, k, vec);
        }
        return LinalgKernels.Shaped(x, n, k, vec);
    }

    private static NdArray GeneralSolve(NdArray a, double[] x, int n, int k, bool vec)
    {
        var (lu, swaps, zero) = LinalgKernels.LuFactor(a);
        if (zero >= 0) throw new LinalgSingularMatrixException($"Solve: matrix is exactly singular (zero pivot at {zero}).");
        LinalgKernels.LuSolveInPlace(lu, swaps, n, x, k, transpose: false);
        return LinalgKernels.Shaped(x, n, k, vec);
    }

    private static LinalgMatrixStructure Detect(double[] a, int n)
    {
        var (lo, up) = Bandwidth(a, n);
        if (lo == 0 && up == 0) return LinalgMatrixStructure.Diagonal;
        if (lo == 0) return LinalgMatrixStructure.UpperTriangular;
        if (up == 0) return LinalgMatrixStructure.LowerTriangular;
        if (lo == 1 && up == 1 && n > 3) return LinalgMatrixStructure.Tridiagonal;

        var symmetric = true;
        for (var i = 0; i < n && symmetric; i++)
            for (var j = i + 1; j < n; j++)
                if (a[i * n + j] != a[j * n + i]) { symmetric = false; break; }
        if (symmetric)
        {
            var positiveDiagonal = true;
            for (var i = 0; i < n; i++) if (!(a[i * n + i] > 0)) positiveDiagonal = false;
            if (positiveDiagonal)
            {
                var trial = (double[])a.Clone();
                if (LinalgKernels.CholeskyInPlace(trial, n) < 0) return LinalgMatrixStructure.PositiveDefinite;
            }
        }
        return LinalgMatrixStructure.General;
    }

    /// <summary>Solves from an <see cref="Ldl"/> factorisation: P-permuted L, block-diagonal D.</summary>
    internal static NdArray LdlSolve(LdlResult f, NdArray b)
    {
        var n = f.D.Shape[0];
        var (x, k, vec) = LinalgKernels.Rhs(b, n, nameof(Solve));
        var p = f.Perm;
        var lu = f.Lu.AsContiguous().AsSpan();
        var l = new double[n * n];
        for (var i = 0; i < n; i++) lu.Slice(p[i] * n, n).CopyTo(l.AsSpan(i * n, n));
        var y = new double[n * k];
        for (var i = 0; i < n; i++) Array.Copy(x, p[i] * k, y, i * k, k);
        LinalgKernels.TriangularSolveInPlace(l, n, y, k, lower: true, transpose: false, unitDiagonal: true);

        var d = f.D.AsContiguous().AsSpan();
        for (var i = 0; i < n;)
        {
            if (i + 1 < n && d[(i + 1) * n + i] != 0.0)
            {
                var a11 = d[i * n + i]; var a21 = d[(i + 1) * n + i]; var a22 = d[(i + 1) * n + i + 1];
                var det = a11 * a22 - a21 * a21;
                if (det == 0.0) throw new LinalgSingularMatrixException("Solve: singular 2x2 pivot block.");
                for (var j = 0; j < k; j++)
                {
                    var y1 = y[i * k + j]; var y2 = y[(i + 1) * k + j];
                    y[i * k + j] = (a22 * y1 - a21 * y2) / det;
                    y[(i + 1) * k + j] = (a11 * y2 - a21 * y1) / det;
                }
                i += 2;
            }
            else
            {
                var di = d[i * n + i];
                if (di == 0.0) throw new LinalgSingularMatrixException("Solve: matrix is exactly singular.");
                for (var j = 0; j < k; j++) y[i * k + j] /= di;
                i++;
            }
        }
        LinalgKernels.TriangularSolveInPlace(l, n, y, k, lower: true, transpose: true, unitDiagonal: true);
        for (var i = 0; i < n; i++) Array.Copy(y, i * k, x, p[i] * k, k);
        return LinalgKernels.Shaped(x, n, k, vec);
    }

    /// <summary>
    /// Solves <c>T x = b</c> for triangular T, scipy's <c>solve_triangular</c>. Only the named
    /// triangle is read.
    /// </summary>
    /// <param name="a">Triangular matrix.</param>
    /// <param name="b">Right-hand side(s).</param>
    /// <param name="trans">0 solves <c>T x = b</c>, 1 <c>Tᵀ x = b</c>, 2 <c>Tᴴ x = b</c> (same as 1 for real).</param>
    /// <param name="lower">Use the lower triangle (default upper, as scipy).</param>
    /// <param name="unitDiagonal">Assume ones on the diagonal without reading it.</param>
    public static NdArray SolveTriangular(NdArray a, NdArray b, int trans = 0, bool lower = false, bool unitDiagonal = false)
    {
        LinalgKernels.RequireSquare(a, nameof(SolveTriangular));
        if (trans is < 0 or > 2) throw new ArgumentException("SolveTriangular: trans must be 0, 1 or 2.");
        var n = a.Shape[0];
        var (x, k, vec) = LinalgKernels.Rhs(b, n, nameof(SolveTriangular));
        LinalgKernels.TriangularSolveInPlace(a.AsContiguous().AsSpan(), n, x, k, lower, trans != 0, unitDiagonal);
        return LinalgKernels.Shaped(x, n, k, vec);
    }

    // ================================================================ banded

    /// <summary>
    /// Solves a banded system from LAPACK band storage, scipy's <c>solve_banded((l, u), ab, b)</c>:
    /// <c>ab[u + i - j, j] = a[i, j]</c>, shape (l + u + 1, n).
    /// </summary>
    /// <remarks>
    /// Gaussian elimination with partial pivoting inside the band (<c>gbsv</c>): pivoting can push
    /// fill-in up to l extra superdiagonals, so each row is kept in a window of width 2l + u + 1.
    /// O(n·l·(l+u)) time — for a tridiagonal system that is linear in n where dense LU is cubic.
    /// </remarks>
    public static NdArray SolveBanded(int l, int u, NdArray ab, NdArray b)
    {
        LinalgKernels.RequireMatrix(ab, nameof(SolveBanded));
        if (l < 0 || u < 0) throw new ArgumentException("SolveBanded: l and u must be non-negative.");
        if (ab.Shape[0] != l + u + 1)
            throw new ArgumentException($"SolveBanded: ab must have l + u + 1 = {l + u + 1} rows, got {ab.Shape[0]}.");
        var n = ab.Shape[1];
        var (x, k, vec) = LinalgKernels.Rhs(b, n, nameof(SolveBanded));
        var src = ab.AsContiguous().AsSpan();
        var width = 2 * l + u + 1;
        var w = new double[n * width];
        // Window column for (row i, column c) is c - i + l.
        for (var j = 0; j < n; j++)
            for (var r = 0; r <= l + u; r++)
            {
                var i = r + j - u;
                if (i < 0 || i >= n) continue;
                w[i * width + (j - i + l)] = src[r * n + j];
            }

        for (var kk = 0; kk < n; kk++)
        {
            var last = Math.Min(n - 1, kk + l);
            var p = kk;
            var best = Math.Abs(w[kk * width + l]);
            for (var i = kk + 1; i <= last; i++)
            {
                var v = Math.Abs(w[i * width + (kk - i + l)]);
                if (v > best) { best = v; p = i; }
            }
            if (best == 0.0) throw new LinalgSingularMatrixException($"SolveBanded: matrix is singular (zero pivot at {kk}).");
            var colEnd = Math.Min(n - 1, kk + l + u);
            if (p != kk)
            {
                for (var c = kk; c <= colEnd; c++)
                    (w[kk * width + (c - kk + l)], w[p * width + (c - p + l)]) = (w[p * width + (c - p + l)], w[kk * width + (c - kk + l)]);
                x.AsSpan(kk * k, k).SwapWith(x.AsSpan(p * k, k));
            }
            var piv = w[kk * width + l];
            for (var i = kk + 1; i <= last; i++)
            {
                var f = w[i * width + (kk - i + l)] / piv;
                if (f == 0.0) continue;
                w[i * width + (kk - i + l)] = 0.0;
                for (var c = kk + 1; c <= colEnd; c++)
                    w[i * width + (c - i + l)] -= f * w[kk * width + (c - kk + l)];
                LinAlg.AxpySpan(-f, x.AsSpan(kk * k, k), x.AsSpan(i * k, k));
            }
        }
        for (var i = n - 1; i >= 0; i--)
        {
            var xi = x.AsSpan(i * k, k);
            var colEnd = Math.Min(n - 1, i + l + u);
            for (var c = i + 1; c <= colEnd; c++)
            {
                var t = w[i * width + (c - i + l)];
                if (t != 0.0) LinAlg.AxpySpan(-t, x.AsSpan(c * k, k), xi);
            }
            var d = 1.0 / w[i * width + l];
            for (var j = 0; j < k; j++) xi[j] *= d;
        }
        return LinalgKernels.Shaped(x, n, k, vec);
    }

    /// <summary>Converts a dense matrix to LAPACK band storage with l sub- and u superdiagonals.</summary>
    public static NdArray DenseToBand(NdArray a, int l, int u)
    {
        LinalgKernels.RequireSquare(a, nameof(DenseToBand));
        return DenseToBand(a.ToArray(), a.Shape[0], l, u);
    }

    private static NdArray DenseToBand(double[] a, int n, int l, int u)
    {
        var rows = l + u + 1;
        var ab = new double[rows * n];
        for (var j = 0; j < n; j++)
            for (var r = 0; r < rows; r++)
            {
                var i = r + j - u;
                if (i >= 0 && i < n) ab[r * n + j] = a[i * n + j];
            }
        return new NdArray(ab, rows, n);
    }

    /// <summary>
    /// Cholesky factor of a symmetric positive-definite banded matrix in band storage, scipy's
    /// <c>cholesky_banded</c>. Upper form: <c>ab[u + i - j, j] = a[i, j]</c> for i ≤ j, factor U with
    /// <c>A = Uᵀ U</c>; lower form: <c>ab[i - j, j]</c>, factor L with <c>A = L Lᵀ</c>. Same storage out.
    /// </summary>
    public static NdArray CholeskyBanded(NdArray ab, bool lower = false)
    {
        LinalgKernels.RequireMatrix(ab, nameof(CholeskyBanded));
        var bw = ab.Shape[0] - 1;
        var n = ab.Shape[1];
        var s = ab.AsContiguous().AsSpan();
        // Lower window: L[i, i - bw .. i] at w[i*(bw+1) + (j - i + bw)].
        var width = bw + 1;
        var w = new double[n * width];
        for (var j = 0; j < n; j++)
            for (var r = 0; r <= bw; r++)
            {
                // Element (i, j) with i >= j in the lower triangle.
                int i;
                double v;
                if (lower) { i = j + r; if (i >= n) continue; v = s[r * n + j]; }
                else
                {
                    // Upper storage holds (row, col) = (j - bw + r, j); transpose to (j, row).
                    var row = j - bw + r;
                    if (row < 0) continue;
                    i = j; v = s[r * n + j];
                    w[i * width + (row - i + bw)] = v;
                    continue;
                }
                w[i * width + (j - i + bw)] = v;
            }

        for (var j = 0; j < n; j++)
        {
            var start = Math.Max(0, j - bw);
            var acc = w[j * width + bw];
            for (var kk = start; kk < j; kk++)
            {
                var v = w[j * width + (kk - j + bw)];
                acc -= v * v;
            }
            if (!(acc > 0.0))
                throw new LinalgException($"CholeskyBanded: the leading minor of order {j + 1} is not positive definite.");
            var d = Math.Sqrt(acc);
            w[j * width + bw] = d;
            var end = Math.Min(n - 1, j + bw);
            for (var i = j + 1; i <= end; i++)
            {
                var sum = w[i * width + (j - i + bw)];
                var from = Math.Max(0, i - bw);
                for (var kk = from; kk < j; kk++)
                    sum -= w[i * width + (kk - i + bw)] * w[j * width + (kk - j + bw)];
                w[i * width + (j - i + bw)] = sum / d;
            }
        }

        var outp = new double[(bw + 1) * n];
        for (var i = 0; i < n; i++)
            for (var j = Math.Max(0, i - bw); j <= i; j++)
            {
                var v = w[i * width + (j - i + bw)];
                if (lower) outp[(i - j) * n + j] = v;               // L[i, j]
                else outp[(bw + j - i) * n + i] = v;                // U[j, i] = L[i, j]
            }
        return new NdArray(outp, bw + 1, n);
    }

    /// <summary>Solves <c>A x = b</c> from a <see cref="CholeskyBanded"/> factor, scipy's <c>cho_solve_banded</c>.</summary>
    public static NdArray ChoSolveBanded(NdArray cb, bool lower, NdArray b)
    {
        LinalgKernels.RequireMatrix(cb, nameof(ChoSolveBanded));
        var bw = cb.Shape[0] - 1;
        var n = cb.Shape[1];
        var s = cb.ToArray();
        var (x, k, vec) = LinalgKernels.Rhs(b, n, nameof(ChoSolveBanded));
        double L(int i, int j) => lower ? s[(i - j) * n + j] : s[(bw + j - i) * n + i];

        // Forward: L y = b.
        for (var i = 0; i < n; i++)
        {
            var xi = x.AsSpan(i * k, k);
            for (var j = Math.Max(0, i - bw); j < i; j++)
                LinAlg.AxpySpan(-L(i, j), x.AsSpan(j * k, k), xi);
            var d = 1.0 / L(i, i);
            for (var c = 0; c < k; c++) xi[c] *= d;
        }
        // Backward: L^T x = y.
        for (var i = n - 1; i >= 0; i--)
        {
            var xi = x.AsSpan(i * k, k);
            for (var j = i + 1; j <= Math.Min(n - 1, i + bw); j++)
                LinAlg.AxpySpan(-L(j, i), x.AsSpan(j * k, k), xi);
            var d = 1.0 / L(i, i);
            for (var c = 0; c < k; c++) xi[c] *= d;
        }
        return LinalgKernels.Shaped(x, n, k, vec);
    }

    /// <summary>
    /// Solves a symmetric/Hermitian positive-definite banded system, scipy's <c>solveh_banded</c>,
    /// by banded Cholesky. Storage as <see cref="CholeskyBanded"/>.
    /// </summary>
    public static NdArray SolveHBanded(NdArray ab, NdArray b, bool lower = false)
        => ChoSolveBanded(CholeskyBanded(ab, lower), lower, b);

    // ================================================================ Toeplitz / circulant

    /// <summary>
    /// Solves <c>T x = b</c> for the Toeplitz matrix with first column <paramref name="c"/> and first
    /// row <paramref name="r"/> (default <c>r = c</c>, symmetric), scipy's <c>solve_toeplitz</c>.
    /// </summary>
    /// <remarks>
    /// Levinson recursion with forward and backward vectors, O(n²) and O(n) memory, all right-hand
    /// sides advanced together. Like scipy's it needs every leading principal minor to be
    /// non-singular, which a perfectly solvable system can violate (e.g. a zero <c>c[0]</c>); it throws
    /// rather than guess. It is not backward stable for indefinite matrices — check the residual
    /// if the matrix is far from positive definite.
    /// </remarks>
    public static NdArray SolveToeplitz(double[] c, double[]? r, NdArray b)
    {
        ArgumentNullException.ThrowIfNull(c);
        r ??= c;
        var n = c.Length;
        if (r.Length != n) throw new ArgumentException("SolveToeplitz: c and r must have the same length.");
        var (y, k, vec) = LinalgKernels.Rhs(b, n, nameof(SolveToeplitz));
        if (n == 0) return LinalgKernels.Shaped(y, 0, k, vec);
        if (c[0] == 0.0) throw new LinalgSingularMatrixException("SolveToeplitz: c[0] is zero, a leading minor is singular.");

        var f = new double[n];
        var bk = new double[n];
        var fn = new double[n];
        var bn = new double[n];
        var x = new double[n * k];
        f[0] = 1.0 / c[0];
        bk[0] = 1.0 / c[0];
        for (var j = 0; j < k; j++) x[j] = y[j] / c[0];
        var ex = new double[k];

        for (var m = 1; m < n; m++)
        {
            double ef = 0, eb = 0;
            for (var j = 0; j < m; j++)
            {
                ef += c[m - j] * f[j];
                eb += r[j + 1] * bk[j];
            }
            var denom = 1.0 - ef * eb;
            if (denom == 0.0 || !double.IsFinite(denom))
                throw new LinalgSingularMatrixException($"SolveToeplitz: leading minor of order {m + 1} is singular.");
            var inv = 1.0 / denom;
            for (var j = 0; j <= m; j++)
            {
                var fe = j < m ? f[j] : 0.0;
                var be = j > 0 ? bk[j - 1] : 0.0;
                fn[j] = (fe - ef * be) * inv;
                bn[j] = (be - eb * fe) * inv;
            }
            Array.Copy(fn, f, m + 1);
            Array.Copy(bn, bk, m + 1);

            Array.Clear(ex);
            for (var j = 0; j < m; j++)
                LinAlg.AxpySpan(c[m - j], x.AsSpan(j * k, k), ex);
            for (var col = 0; col < k; col++)
            {
                var coef = y[m * k + col] - ex[col];
                for (var j = 0; j <= m; j++) x[j * k + col] += coef * bk[j];
            }
        }
        return LinalgKernels.Shaped(x, n, k, vec);
    }

    /// <summary>
    /// Solves <c>C x = b</c> for the circulant matrix with first column <paramref name="c"/>, scipy's
    /// <c>solve_circulant</c>, in O(n log n) through GraviNum's FFT: C is diagonalised by the DFT, so
    /// <c>x = ifft(fft(b) / fft(c))</c>.
    /// </summary>
    /// <param name="c">First column of C.</param>
    /// <param name="b">Right-hand side(s).</param>
    /// <param name="singularLstsq">When an eigenvalue <c>|fft(c)_k| ≤ tol</c>, return the minimum-norm
    /// least-squares solution (scipy's <c>singular='lstsq'</c>) instead of throwing.</param>
    /// <param name="tol">Near-zero eigenvalue threshold; default <c>n·eps·max|fft(c)|</c>.</param>
    public static NdArray SolveCirculant(double[] c, NdArray b, bool singularLstsq = false, double? tol = null)
    {
        ArgumentNullException.ThrowIfNull(c);
        var n = c.Length;
        var (y, k, vec) = LinalgKernels.Rhs(b, n, nameof(SolveCirculant));
        var cc = new Complex[n];
        for (var i = 0; i < n; i++) cc[i] = c[i];
        var lambda = GFft.Forward(cc);
        var maxAbs = 0.0;
        foreach (var v in lambda) maxAbs = Math.Max(maxAbs, v.Magnitude);
        var t = tol ?? maxAbs * n * LinalgKernels.Eps;
        var near = new bool[n];
        for (var i = 0; i < n; i++)
        {
            near[i] = lambda[i].Magnitude <= t;
            if (near[i] && !singularLstsq) throw new LinalgSingularMatrixException("SolveCirculant: near singular circulant matrix.");
        }

        var x = new double[n * k];
        var col = new Complex[n];
        for (var j = 0; j < k; j++)
        {
            for (var i = 0; i < n; i++) col[i] = y[i * k + j];
            var fb = GFft.Forward(col);
            for (var i = 0; i < n; i++) fb[i] = near[i] ? Complex.Zero : fb[i] / lambda[i];
            var xs = GFft.Inverse(fb);
            for (var i = 0; i < n; i++) x[i * k + j] = xs[i].Real;
        }
        return LinalgKernels.Shaped(x, n, k, vec);
    }

    // ================================================================ least squares / inverses

    /// <summary>
    /// Least-squares solution of <c>A x ≈ b</c> from the SVD (the <c>gelsd</c> answer), scipy's
    /// <c>lstsq</c>. Singular values below <c>cond·s_max</c> are treated as zero, giving the
    /// minimum-norm solution for rank-deficient A.
    /// </summary>
    /// <param name="a">m×n matrix.</param>
    /// <param name="b">Vector of length m or m×k matrix.</param>
    /// <param name="cond">Relative cut-off; default machine epsilon (scipy's default).</param>
    public static LstsqResult Lstsq(NdArray a, NdArray b, double? cond = null)
    {
        LinalgKernels.RequireMatrix(a, nameof(Lstsq));
        var m = a.Shape[0];
        var n = a.Shape[1];
        var (bb, k, vec) = LinalgKernels.Rhs(b, m, nameof(Lstsq));
        var svd = Decomposition.Svd(a);
        var s = svd.SingularValues.ToArray();
        var cut = (cond ?? LinalgKernels.Eps) * (s.Length > 0 ? s[0] : 0.0);
        var rank = 0;
        foreach (var v in s) if (v > cut) rank++;

        // x = V diag(1/s) U^T b over the retained singular triplets.
        var u = svd.U.AsContiguous().AsSpan();
        var vv = svd.V.AsContiguous().AsSpan();
        var p = s.Length;
        var utb = new double[p * k];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < rank; j++)
            {
                var uij = u[i * p + j];
                if (uij != 0.0) LinAlg.AxpySpan(uij, bb.AsSpan(i * k, k), utb.AsSpan(j * k, k));
            }
        for (var j = 0; j < rank; j++)
        {
            var inv = 1.0 / s[j];
            for (var c = 0; c < k; c++) utb[j * k + c] *= inv;
        }
        var x = new double[n * k];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < rank; j++)
            {
                var vij = vv[i * p + j];
                if (vij != 0.0) LinAlg.AxpySpan(vij, utb.AsSpan(j * k, k), x.AsSpan(i * k, k));
            }

        var residues = Array.Empty<double>();
        if (m > n && rank == n)
        {
            residues = new double[k];
            var ax = LinalgKernels.Mul(a.AsContiguous(), new NdArray(x, n, k)).AsSpan();
            for (var i = 0; i < m; i++)
                for (var c = 0; c < k; c++)
                {
                    var d = bb[i * k + c] - ax[i * k + c];
                    residues[c] += d * d;
                }
        }
        return new LstsqResult(LinalgKernels.Shaped(x, n, k, vec), residues, rank, s);
    }

    /// <summary>Matrix inverse by LU, scipy's <c>inv</c>.</summary>
    /// <exception cref="LinalgSingularMatrixException">A has an exactly zero pivot.</exception>
    public static NdArray Inv(NdArray a)
    {
        LinalgKernels.RequireSquare(a, nameof(Inv));
        var n = a.Shape[0];
        var (lu, swaps, zero) = LinalgKernels.LuFactor(a);
        if (zero >= 0) throw new LinalgSingularMatrixException("Inv: singular matrix.");
        var x = LinalgKernels.IdentityBuffer(n);
        LinalgKernels.LuSolveInPlace(lu, swaps, n, x, n, transpose: false);
        return new NdArray(x, n, n);
    }

    /// <summary>
    /// Moore–Penrose pseudo-inverse from the SVD, scipy's <c>pinv</c>: singular values at or below
    /// <c>atol + rtol·s_max</c> are dropped; <c>rtol</c> defaults to <c>max(m, n)·eps</c>.
    /// </summary>
    public static NdArray Pinv(NdArray a, double atol = 0.0, double? rtol = null)
    {
        LinalgKernels.RequireMatrix(a, nameof(Pinv));
        var m = a.Shape[0];
        var n = a.Shape[1];
        var svd = Decomposition.Svd(a);
        var s = svd.SingularValues.ToArray();
        var val = atol + (s.Length > 0 ? s[0] : 0.0) * (rtol ?? Math.Max(m, n) * LinalgKernels.Eps);
        var p = s.Length;
        var vs = svd.V.Copy();
        var span = vs.AsSpan();
        for (var i = 0; i < n; i++)
            for (var j = 0; j < p; j++) span[i * p + j] = s[j] > val ? span[i * p + j] / s[j] : 0.0;
        return LinalgKernels.Mul(vs, svd.U.T);
    }

    /// <summary>
    /// Pseudo-inverse of a symmetric matrix from its eigen decomposition, scipy's <c>pinvh</c>;
    /// eigenvalues with |λ| ≤ <c>atol + rtol·max|λ|</c> are dropped (rtol default <c>n·eps</c>).
    /// </summary>
    public static NdArray Pinvh(NdArray a, double atol = 0.0, double? rtol = null, bool lower = true)
    {
        LinalgKernels.RequireSquare(a, nameof(Pinvh));
        var n = a.Shape[0];
        var e = Eigh(a, lower: lower);
        var maxAbs = 0.0;
        foreach (var v in e.Values) maxAbs = Math.Max(maxAbs, Math.Abs(v));
        var val = atol + maxAbs * (rtol ?? n * LinalgKernels.Eps);
        var u = e.Vectors.AsContiguous().AsSpan();
        var scaled = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
                scaled[i * n + j] = Math.Abs(e.Values[j]) > val ? u[i * n + j] / e.Values[j] : 0.0;
        return LinalgKernels.Mul(new NdArray(scaled, n, n), e.Vectors.T);
    }

    /// <summary>Determinant by LU, scipy's <c>det</c>.</summary>
    public static double Det(NdArray a)
    {
        LinalgKernels.RequireSquare(a, nameof(Det));
        var n = a.Shape[0];
        if (n == 0) return 1.0;
        var (lu, swaps, _) = LinalgKernels.LuFactor(a);
        var det = 1.0;
        for (var i = 0; i < n; i++)
        {
            det *= lu[i * n + i];
            if (swaps[i] != i) det = -det;
        }
        return det;
    }
}
