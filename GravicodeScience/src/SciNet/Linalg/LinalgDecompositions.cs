using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// scipy.linalg for .NET: solvers, factorisations, eigenproblems, matrix functions, matrix
/// equations, special matrices and norms beyond what numpy.linalg (GraviNum <c>LinAlg</c>) has.
/// </summary>
/// <remarks>
/// Matrices are GraviNum <see cref="NdArray"/>s, row-major. Products go through GraviNum's
/// blocked, parallel <c>LinAlg.Dot</c>; SVD and symmetric eigenproblems through
/// <c>Decomposition.Svd</c>/<c>SymmetricEigen</c> (and native LAPACK when installed). Note the
/// spelling: this is <c>Linalg</c> (scipy.linalg), GraviNum's is <c>LinAlg</c> (numpy.linalg).
/// </remarks>
public static partial class Linalg
{
    // ================================================================ LU / Cholesky

    /// <summary>
    /// Packed LU factorisation with partial pivoting, scipy's <c>lu_factor</c>.
    /// </summary>
    /// <remarks>
    /// A singular matrix is factorised anyway (as LAPACK does) — <see cref="LuSolve"/> is where it
    /// fails. Pivots are row swaps, not a permutation; reading one as the other reconstructs the
    /// wrong matrix while every factor still looks valid.
    /// </remarks>
    public static LuFactorResult LuFactor(NdArray a)
    {
        LinalgKernels.RequireSquare(a, nameof(LuFactor));
        var n = a.Shape[0];
        var (lu, swaps, _) = LinalgKernels.LuFactor(a);
        return new LuFactorResult(new NdArray(lu, n, n), swaps);
    }

    /// <summary>Solves <c>A x = b</c> (or <c>Aᵀ x = b</c> with <paramref name="trans"/>) from <see cref="LuFactor"/>.</summary>
    /// <exception cref="LinalgSingularMatrixException">U has an exactly zero diagonal entry.</exception>
    public static NdArray LuSolve(LuFactorResult luAndPiv, NdArray b, bool trans = false)
    {
        ArgumentNullException.ThrowIfNull(luAndPiv);
        var n = luAndPiv.Lu.Shape[0];
        var (x, k, vec) = LinalgKernels.Rhs(b, n, nameof(LuSolve));
        LinalgKernels.LuSolveInPlace(luAndPiv.Lu.AsContiguous().AsSpan(), luAndPiv.Piv, n, x, k, trans);
        return LinalgKernels.Shaped(x, n, k, vec);
    }

    /// <summary>
    /// The full <c>A = P L U</c> factorisation, scipy's <c>lu</c>; P here is the permutation
    /// matrix with <c>A = P L U</c> (scipy's convention, the transpose of GraviNum's <c>P A = L U</c>).
    /// </summary>
    public static (NdArray P, NdArray L, NdArray U) Lu(NdArray a)
    {
        LinalgKernels.RequireSquare(a, nameof(Lu));
        var n = a.Shape[0];
        var (lu, swaps, _) = LinalgKernels.LuFactor(a);
        var perm = LinalgKernels.SwapsToPermutation(swaps);
        var p = new double[n * n];
        var l = new double[n * n];
        var u = new double[n * n];
        for (var i = 0; i < n; i++)
        {
            p[perm[i] * n + i] = 1.0;
            for (var j = 0; j < i; j++) l[i * n + j] = lu[i * n + j];
            l[i * n + i] = 1.0;
            for (var j = i; j < n; j++) u[i * n + j] = lu[i * n + j];
        }
        return (new NdArray(p, n, n), new NdArray(l, n, n), new NdArray(u, n, n));
    }

    /// <summary>Cholesky factor, scipy's <c>cholesky</c>: upper <c>U</c> with <c>A = Uᵀ U</c> by default.</summary>
    /// <remarks>Only the requested triangle of <paramref name="a"/> is read, as LAPACK does.</remarks>
    /// <exception cref="LinalgException">The matrix is not positive definite.</exception>
    public static NdArray Cholesky(NdArray a, bool lower = false)
    {
        LinalgKernels.RequireSquare(a, nameof(Cholesky));
        var n = a.Shape[0];
        var src = a;
        if (!lower) src = a.T.Copy(); // the lower triangle of A^T is the upper triangle of A
        var l = LinalgKernels.CholeskyLower(src, nameof(Cholesky));
        return lower ? new NdArray(l, n, n) : new NdArray(LinalgKernels.Transpose(l, n, n), n, n);
    }

    /// <summary>Cholesky factor for <see cref="ChoSolve"/>, scipy's <c>cho_factor</c>.</summary>
    public static ChoFactorResult ChoFactor(NdArray a, bool lower = false) => new(Cholesky(a, lower), lower);

    /// <summary>Solves <c>A x = b</c> from <see cref="ChoFactor"/>.</summary>
    public static NdArray ChoSolve(ChoFactorResult factor, NdArray b)
    {
        ArgumentNullException.ThrowIfNull(factor);
        var n = factor.C.Shape[0];
        var c = factor.C.AsContiguous().AsSpan();
        var (x, k, vec) = LinalgKernels.Rhs(b, n, nameof(ChoSolve));
        if (factor.Lower)
        {
            LinalgKernels.TriangularSolveInPlace(c, n, x, k, lower: true, transpose: false, unitDiagonal: false);
            LinalgKernels.TriangularSolveInPlace(c, n, x, k, lower: true, transpose: true, unitDiagonal: false);
        }
        else
        {
            LinalgKernels.TriangularSolveInPlace(c, n, x, k, lower: false, transpose: true, unitDiagonal: false);
            LinalgKernels.TriangularSolveInPlace(c, n, x, k, lower: false, transpose: false, unitDiagonal: false);
        }
        return LinalgKernels.Shaped(x, n, k, vec);
    }

    // ================================================================ LDL^T

    /// <summary>
    /// Bunch–Kaufman <c>A = Lu D Luᵀ</c> for a symmetric (possibly indefinite) matrix, scipy's
    /// <c>ldl</c> with <c>lower=True</c>. Only the lower triangle of <paramref name="a"/> is read.
    /// </summary>
    /// <remarks>
    /// Partial pivoting with the Bunch–Kaufman threshold α = (1+√17)/8, choosing 2×2 pivots when no
    /// diagonal entry is large enough. Pivot choices can differ from LAPACK's blocked
    /// <c>sytrf</c> on ties, so <c>Lu</c> and <c>D</c> may differ from scipy while
    /// <c>Lu D Luᵀ = A</c> holds either way.
    /// </remarks>
    public static LdlResult Ldl(NdArray a)
    {
        LinalgKernels.RequireSquare(a, nameof(Ldl));
        var n = a.Shape[0];
        var w = a.ToArray();
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++) w[i * n + j] = w[j * n + i];

        var l = new double[n * n];
        var d = new double[n * n];
        var p = new int[n];
        for (var i = 0; i < n; i++) p[i] = i;
        var alpha = (1.0 + Math.Sqrt(17.0)) / 8.0;

        var k = 0;
        while (k < n)
        {
            var absakk = Math.Abs(w[k * n + k]);
            var r = k;
            var colmax = 0.0;
            for (var i = k + 1; i < n; i++)
            {
                var v = Math.Abs(w[i * n + k]);
                if (v > colmax) { colmax = v; r = i; }
            }

            int kstep, kp;
            if (Math.Max(absakk, colmax) == 0.0 || absakk >= alpha * colmax)
            {
                kstep = 1; kp = k;
            }
            else
            {
                var rowmax = 0.0;
                for (var j = k; j < n; j++)
                    if (j != r) rowmax = Math.Max(rowmax, Math.Abs(w[r * n + j]));
                if (absakk * rowmax >= alpha * colmax * colmax) { kstep = 1; kp = k; }
                else if (Math.Abs(w[r * n + r]) >= alpha * rowmax) { kstep = 1; kp = r; }
                else { kstep = 2; kp = r; }
            }

            var kk = k + kstep - 1;
            if (kp != kk)
            {
                // Symmetric interchange of kk and kp in the working matrix, and in the rows of L built so far.
                w.AsSpan(kk * n, n).SwapWith(w.AsSpan(kp * n, n));
                for (var i = 0; i < n; i++) (w[i * n + kk], w[i * n + kp]) = (w[i * n + kp], w[i * n + kk]);
                for (var j = 0; j < k; j++) (l[kk * n + j], l[kp * n + j]) = (l[kp * n + j], l[kk * n + j]);
                (p[kk], p[kp]) = (p[kp], p[kk]);
            }

            if (kstep == 1)
            {
                var dk = w[k * n + k];
                d[k * n + k] = dk;
                l[k * n + k] = 1.0;
                if (dk != 0.0)
                {
                    for (var i = k + 1; i < n; i++) l[i * n + k] = w[i * n + k] / dk;
                    for (var i = k + 1; i < n; i++)
                    {
                        var f = l[i * n + k] * dk;
                        if (f == 0.0) continue;
                        for (var j = k + 1; j < n; j++) w[i * n + j] -= f * l[j * n + k];
                    }
                }
            }
            else
            {
                var e11 = w[k * n + k];
                var e21 = w[(k + 1) * n + k];
                var e22 = w[(k + 1) * n + k + 1];
                var det = e11 * e22 - e21 * e21;
                d[k * n + k] = e11; d[k * n + k + 1] = e21; d[(k + 1) * n + k] = e21; d[(k + 1) * n + k + 1] = e22;
                l[k * n + k] = 1.0; l[(k + 1) * n + k + 1] = 1.0;
                for (var i = k + 2; i < n; i++)
                {
                    var wi1 = w[i * n + k];
                    var wi2 = w[i * n + k + 1];
                    // [l1 l2] = [wi1 wi2] E^{-1}
                    l[i * n + k] = (wi1 * e22 - wi2 * e21) / det;
                    l[i * n + k + 1] = (wi2 * e11 - wi1 * e21) / det;
                }
                for (var i = k + 2; i < n; i++)
                {
                    var wi1 = w[i * n + k];
                    var wi2 = w[i * n + k + 1];
                    for (var j = k + 2; j < n; j++)
                        w[i * n + j] -= wi1 * l[j * n + k] + wi2 * l[j * n + k + 1];
                }
            }
            k += kstep;
        }

        // Lu = P^T L: row p[i] of Lu is row i of L.
        var lu = new double[n * n];
        for (var i = 0; i < n; i++) Array.Copy(l, i * n, lu, p[i] * n, n);
        return new LdlResult(new NdArray(lu, n, n), new NdArray(d, n, n), p);
    }

    // ================================================================ QR variants

    /// <summary>
    /// QR with column pivoting (Businger–Golub), scipy's <c>qr(a, pivoting=True)</c>:
    /// <c>A[:, P] = Q R</c> with |R[k,k]| non-increasing, which is what makes it rank-revealing.
    /// </summary>
    /// <param name="a">An m×n matrix.</param>
    /// <param name="economic">Return Q as m×k and R as k×n (k = min(m, n)) instead of m×m and m×n.</param>
    /// <remarks>
    /// Column norms are recomputed at every step rather than down-dated. Down-dating is the usual
    /// economy but loses all accuracy once a norm cancels below sqrt(eps) of its start, which is
    /// precisely the rank-deficient case pivoting exists for; recomputing costs the same order of
    /// work as the reflections themselves.
    /// </remarks>
    public static QrPivotingResult QrPivoting(NdArray a, bool economic = false)
    {
        LinalgKernels.RequireMatrix(a, nameof(QrPivoting));
        var m = a.Shape[0];
        var n = a.Shape[1];
        var r = a.ToArray();
        var perm = new int[n];
        for (var j = 0; j < n; j++) perm[j] = j;
        var steps = Math.Min(m, n);
        var vs = new double[steps][];
        var betas = new double[steps];
        var norms = new double[n];
        var w = new double[n];

        for (var k = 0; k < steps; k++)
        {
            Array.Clear(norms);
            for (var i = k; i < m; i++)
            {
                var row = r.AsSpan(i * n, n);
                for (var j = k; j < n; j++) norms[j] += row[j] * row[j];
            }
            var best = k;
            for (var j = k + 1; j < n; j++) if (norms[j] > norms[best]) best = j;
            var normX = Math.Sqrt(norms[best]);
            if (best != k)
            {
                for (var i = 0; i < m; i++) (r[i * n + k], r[i * n + best]) = (r[i * n + best], r[i * n + k]);
                (perm[k], perm[best]) = (perm[best], perm[k]);
            }

            var v = new double[m - k];
            vs[k] = v;
            if (normX == 0.0) { betas[k] = 0.0; continue; }
            var x0 = r[k * n + k];
            var alpha = x0 > 0 ? -normX : normX;
            for (var i = k; i < m; i++) v[i - k] = r[i * n + k];
            v[0] -= alpha;
            var vtv = 0.0;
            foreach (var t in v) vtv += t * t;
            if (vtv == 0.0) { betas[k] = 0.0; continue; }
            var beta = 2.0 / vtv;
            betas[k] = beta;

            var width = n - k;
            var ws = w.AsSpan(0, width);
            ws.Clear();
            for (var i = k; i < m; i++) LinAlg.AxpySpan(v[i - k], r.AsSpan(i * n + k, width), ws);
            for (var i = k; i < m; i++) LinAlg.AxpySpan(-beta * v[i - k], ws, r.AsSpan(i * n + k, width));
            r[k * n + k] = alpha;
            for (var i = k + 1; i < m; i++) r[i * n + k] = 0.0;
        }

        var qCols = economic ? steps : m;
        var q = new double[m * qCols];
        for (var i = 0; i < Math.Min(m, qCols); i++) q[i * qCols + i] = 1.0;
        var qw = new double[qCols];
        for (var k = steps - 1; k >= 0; k--)
        {
            if (betas[k] == 0.0) continue;
            var v = vs[k];
            var ws = qw.AsSpan();
            ws.Clear();
            for (var i = k; i < m; i++) LinAlg.AxpySpan(v[i - k], q.AsSpan(i * qCols, qCols), ws);
            for (var i = k; i < m; i++) LinAlg.AxpySpan(-betas[k] * v[i - k], ws, q.AsSpan(i * qCols, qCols));
        }

        var rRows = economic ? steps : m;
        var rOut = new double[rRows * n];
        Array.Copy(r, rOut, rRows * n);
        return new QrPivotingResult(new NdArray(q, m, qCols), new NdArray(rOut, rRows, n), perm);
    }

    /// <summary>
    /// RQ factorisation <c>A = R Q</c>, scipy's <c>rq</c>: R upper trapezoidal (m×n, its last
    /// min(m,n) columns triangular), Q orthogonal n×n; economic gives R m×k and Q k×n.
    /// </summary>
    /// <remarks>
    /// Computed from GraviNum's QR of the row-reversed transpose: with J the reversal,
    /// <c>(J A)ᵀ = Q₁ R₁</c> gives <c>A = (J R₁ᵀ J)(J Q₁ᵀ)</c>, and reversing a triangle's rows and
    /// columns turns lower into upper.
    /// </remarks>
    public static RqResult Rq(NdArray a, bool economic = false)
    {
        LinalgKernels.RequireMatrix(a, nameof(Rq));
        var m = a.Shape[0];
        var n = a.Shape[1];
        var src = a.AsContiguous().AsSpan();
        var b = new double[n * m]; // (J A)^T, n x m
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++) b[j * m + i] = src[(m - 1 - i) * n + j];
        var qr = Decomposition.Qr(new NdArray(b, n, m), reduced: false);
        var q1 = qr.Q.AsContiguous().AsSpan(); // n x n
        var r1 = qr.R.AsContiguous().AsSpan(); // n x m

        var r = new double[m * n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++) r[i * n + j] = r1[(n - 1 - j) * m + (m - 1 - i)];
        var q = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++) q[i * n + j] = q1[j * n + (n - 1 - i)];

        if (!economic || m >= n) return new RqResult(new NdArray(r, m, n), new NdArray(q, n, n));

        var k = m;
        var re = new double[m * k];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < k; j++) re[i * k + j] = r[i * n + (n - k + j)];
        var qe = new double[k * n];
        Array.Copy(q, (n - k) * n, qe, 0, k * n);
        return new RqResult(new NdArray(re, m, k), new NdArray(qe, k, n));
    }

    // ================================================================ Hessenberg / Schur / Eig

    /// <summary>
    /// Upper Hessenberg form by Householder reflections, scipy's <c>hessenberg</c>:
    /// <c>A = Q H Qᵀ</c>. Q is returned only when <paramref name="calcQ"/> is set.
    /// </summary>
    public static HessenbergResult Hessenberg(NdArray a, bool calcQ = false)
    {
        LinalgKernels.RequireSquare(a, nameof(Hessenberg));
        var n = a.Shape[0];
        var h = a.ToArray();
        LinalgKernels.RequireFinite(h, nameof(Hessenberg));
        var q = calcQ ? new double[n * n] : null;
        LinalgSchurKernel.Hessenberg(h, q, n);
        return new HessenbergResult(new NdArray(h, n, n), q is null ? null : new NdArray(q, n, n));
    }

    /// <summary>
    /// Real Schur decomposition <c>A = Z T Zᵀ</c>, scipy's <c>schur(a, output='real')</c>.
    /// </summary>
    /// <remarks>
    /// Householder Hessenberg reduction, then the Francis implicit double-shift QR iteration
    /// (EISPACK <c>hqr2</c>) accumulating Z, then each 2×2 block standardised as LAPACK's
    /// <c>dlanv2</c> does, so T matches scipy's up to the signs of Schur vectors. No eigenvalue
    /// reordering (scipy's <c>sort=</c>) is provided.
    /// </remarks>
    /// <exception cref="LinalgException">The QR iteration failed to converge.</exception>
    public static SchurResult Schur(NdArray a)
    {
        LinalgKernels.RequireSquare(a, nameof(Schur));
        var n = a.Shape[0];
        var t = a.ToArray();
        LinalgKernels.RequireFinite(t, nameof(Schur));
        var z = new double[n * n];
        LinalgSchurKernel.RealSchur(t, z, n);
        return new SchurResult(new NdArray(t, n, n), new NdArray(z, n, n));
    }

    /// <summary>
    /// Complex Schur decomposition <c>A = Z T Zᴴ</c> with T upper triangular, scipy's
    /// <c>schur(a, output='complex')</c> — the real Schur form converted block by block (<c>rsf2csf</c>).
    /// </summary>
    public static ComplexSchurResult SchurComplex(NdArray a)
    {
        LinalgKernels.RequireSquare(a, nameof(SchurComplex));
        var n = a.Shape[0];
        var (t, z) = LinalgSchurKernel.ComplexSchur(a);
        return new ComplexSchurResult(new ComplexNdArray(t, n, n), new ComplexNdArray(z, n, n));
    }

    /// <summary>
    /// Eigenvalues of a general real matrix, scipy's <c>eigvals</c>, from the real Schur form of the
    /// balanced matrix. Complex pairs are adjacent with the positive imaginary part first; the
    /// order is otherwise the Schur order, not sorted.
    /// </summary>
    public static Complex[] Eigvals(NdArray a)
    {
        LinalgKernels.RequireSquare(a, nameof(Eigvals));
        var n = a.Shape[0];
        var t = a.ToArray();
        LinalgKernels.RequireFinite(t, nameof(Eigvals));
        LinalgSchurKernel.Balance(t, n);
        var (re, im) = LinalgSchurKernel.RealSchur(t, null, n);
        var w = new Complex[n];
        for (var i = 0; i < n; i++) w[i] = new Complex(re[i], im[i]);
        return w;
    }

    /// <summary>
    /// Eigenvalues and right eigenvectors of a general real matrix, scipy's <c>eig</c>.
    /// </summary>
    /// <remarks>
    /// Balance (diagonal scaling by powers of two, like LAPACK <c>gebal</c>), real Schur form, the
    /// complex Schur form from it, back-substitution on the triangle, and back-transformation.
    /// Each vector has unit 2-norm; a complex vector is rotated so its largest component is real
    /// and positive, and the vector of <c>λ̄</c> is the exact conjugate of that of <c>λ</c>. For a
    /// defective matrix the computed vectors are (nearly) parallel, as in LAPACK.
    /// </remarks>
    public static LinalgEigResult Eig(NdArray a)
    {
        LinalgKernels.RequireSquare(a, nameof(Eig));
        var n = a.Shape[0];
        var t = a.ToArray();
        LinalgKernels.RequireFinite(t, nameof(Eig));
        var scale = LinalgSchurKernel.Balance(t, n);
        var z = new double[n * n];
        var (re, im) = LinalgSchurKernel.RealSchur(t, z, n);
        var (ct, cz) = LinalgSchurKernel.ToComplexSchur(t, z, n);

        var values = new Complex[n];
        var vectors = new Complex[n * n];
        var x = new Complex[n];
        var v = new Complex[n];
        for (var k = 0; k < n; k++)
        {
            values[k] = new Complex(re[k], im[k]);
            if (im[k] < 0 && k > 0 && im[k - 1] > 0)
            {
                for (var i = 0; i < n; i++) vectors[i * n + k] = Complex.Conjugate(vectors[i * n + k - 1]);
                continue;
            }

            LinalgSchurKernel.EigenvectorOfTriangular(ct, n, k, x);
            for (var i = 0; i < n; i++)
            {
                var s = Complex.Zero;
                var row = cz.AsSpan(i * n, k + 1);
                for (var j = 0; j <= k; j++) s += row[j] * x[j];
                v[i] = s * scale[i];
            }

            var isReal = im[k] == 0.0;
            var norm = 0.0;
            var big = 0;
            for (var i = 0; i < n; i++)
            {
                if (isReal) v[i] = v[i].Real;
                var mag = Complex.Abs(v[i]);
                norm += mag * mag;
                if (mag > Complex.Abs(v[big])) big = i;
            }
            norm = Math.Sqrt(norm);
            var phase = isReal ? Complex.One : Complex.Conjugate(v[big]) / Complex.Abs(v[big]);
            for (var i = 0; i < n; i++) vectors[i * n + k] = v[i] * phase / norm;
        }
        return new LinalgEigResult(values, new ComplexNdArray(vectors, n, n));
    }

    /// <summary>
    /// Eigen decomposition of a real symmetric matrix, or of the generalised symmetric-definite
    /// problem <c>A v = λ B v</c>, scipy's <c>eigh</c>. Eigenvalues ascending.
    /// </summary>
    /// <param name="a">Symmetric matrix; only the triangle named by <paramref name="lower"/> is read.</param>
    /// <param name="b">Optional symmetric positive-definite B.</param>
    /// <param name="lower">Read the lower (default) or upper triangle of a and b.</param>
    /// <param name="eigvalsOnly">Skip the eigenvectors (Vectors is then an empty 0×0 array).</param>
    /// <param name="subsetByIndex">Inclusive ascending-index range <c>(lo, hi)</c> to return.</param>
    /// <param name="subsetByValue">Half-open value interval <c>(lo, hi]</c> to return.</param>
    /// <remarks>
    /// The standard problem delegates to GraviNum's <c>SymmetricEigen</c> (Householder tridiagonal
    /// + implicit QL, or LAPACK) and reverses its descending order. The generalised problem is
    /// reduced with the Cholesky factor B = L Lᵀ to <c>C = L⁻¹ A L⁻ᵀ</c>; eigenvectors are mapped back
    /// by <c>L⁻ᵀ</c>, so they satisfy <c>Vᵀ B V = I</c> as scipy's do. That reduction is accurate when
    /// B is well conditioned; for nearly singular B it loses digits in proportion to cond(B).
    /// Subsets are selected after a full solve, so they save no time.
    /// </remarks>
    public static EighResult Eigh(NdArray a, NdArray? b = null, bool lower = true, bool eigvalsOnly = false,
        (int Lo, int Hi)? subsetByIndex = null, (double Lo, double Hi)? subsetByValue = null)
    {
        LinalgKernels.RequireSquare(a, nameof(Eigh));
        var n = a.Shape[0];
        var c = SymmetricFromTriangle(a, lower);

        double[]? lFactor = null;
        if (b is not null)
        {
            LinalgKernels.RequireSquare(b, nameof(Eigh));
            if (b.Shape[0] != n) throw new ArgumentException("Eigh: a and b must have the same shape.");
            var bs = SymmetricFromTriangle(b, lower);
            lFactor = LinalgKernels.CholeskyLower(new NdArray(bs, n, n), nameof(Eigh));
            // C = L^{-1} A L^{-T}: solve L Y = A, then L C^T = Y^T (C symmetric).
            LinalgKernels.TriangularSolveInPlace(lFactor, n, c, n, lower: true, transpose: false, unitDiagonal: false);
            var yt = LinalgKernels.Transpose(c, n, n);
            LinalgKernels.TriangularSolveInPlace(lFactor, n, yt, n, lower: true, transpose: false, unitDiagonal: false);
            c = yt;
            LinalgKernels.Symmetrize(c, n);
        }

        var eig = Decomposition.SymmetricEigen(new NdArray(c, n, n));
        var desc = eig.Values.ToArray();
        var vecDesc = eig.Vectors.AsContiguous().AsSpan();

        var idx = new List<int>(n);
        for (var i = 0; i < n; i++)
        {
            var src = n - 1 - i; // ascending position i
            var val = desc[src];
            if (subsetByIndex is { } si && (i < si.Lo || i > si.Hi)) continue;
            if (subsetByValue is { } sv && !(val > sv.Lo && val <= sv.Hi)) continue;
            idx.Add(src);
        }
        if (subsetByIndex is { } chk && (chk.Lo < 0 || chk.Hi >= n || chk.Lo > chk.Hi))
            throw new ArgumentException("Eigh: subsetByIndex must satisfy 0 <= lo <= hi < n.");

        var values = new double[idx.Count];
        for (var j = 0; j < idx.Count; j++) values[j] = desc[idx[j]];
        if (eigvalsOnly) return new EighResult(values, NdArray.Zeros(0, 0));

        var m = idx.Count;
        var vecs = new double[n * m];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < m; j++) vecs[i * m + j] = vecDesc[i * n + idx[j]];
        if (lFactor is not null)
            LinalgKernels.TriangularSolveInPlace(lFactor, n, vecs, m, lower: true, transpose: true, unitDiagonal: false);
        return new EighResult(values, new NdArray(vecs, n, m));
    }

    /// <summary>Eigenvalues of a symmetric (or generalised symmetric-definite) problem, ascending; scipy's <c>eigvalsh</c>.</summary>
    public static double[] Eigvalsh(NdArray a, NdArray? b = null, bool lower = true)
        => Eigh(a, b, lower, eigvalsOnly: true).Values;

    /// <summary>
    /// Eigen decomposition of the symmetric tridiagonal matrix with diagonal <paramref name="d"/>
    /// and off-diagonal <paramref name="e"/>, scipy's <c>eigh_tridiagonal</c>. Eigenvalues ascending.
    /// </summary>
    /// <remarks>
    /// Implicit QL with Wilkinson shifts directly on the tridiagonal (EISPACK <c>tql2</c>):
    /// O(n²) for values alone, with no O(n³) reduction step because there is nothing to reduce.
    /// </remarks>
    public static EighResult EighTridiagonal(double[] d, double[] e, bool eigvalsOnly = false)
    {
        ArgumentNullException.ThrowIfNull(d);
        ArgumentNullException.ThrowIfNull(e);
        var n = d.Length;
        if (e.Length != Math.Max(n - 1, 0)) throw new ArgumentException("EighTridiagonal: e must have length len(d) - 1.");
        var dd = (double[])d.Clone();
        var ee = new double[n];
        Array.Copy(e, ee, n - 1 < 0 ? 0 : n - 1);
        var vt = eigvalsOnly ? null : LinalgKernels.IdentityBuffer(n);
        LinalgTridiagonal.Tql2(dd, ee, vt, n);

        // Sort ascending, carrying the eigenvector rows (vt holds vectors as rows).
        var order = new int[n];
        for (var i = 0; i < n; i++) order[i] = i;
        Array.Sort((double[])dd.Clone(), order);
        var values = new double[n];
        for (var i = 0; i < n; i++) values[i] = dd[order[i]];
        if (vt is null) return new EighResult(values, NdArray.Zeros(0, 0));
        var vecs = new double[n * n];
        for (var j = 0; j < n; j++)
        {
            var src = order[j];
            for (var i = 0; i < n; i++) vecs[i * n + j] = vt[src * n + i];
        }
        return new EighResult(values, new NdArray(vecs, n, n));
    }

    /// <summary>
    /// Eigenvalues of a symmetric banded matrix in LAPACK upper band storage
    /// (<c>ab[u + i - j, j] = a[i, j]</c>) or lower (<c>ab[i - j, j]</c>), scipy's <c>eigvals_banded</c>.
    /// </summary>
    /// <remarks>Expands to dense and calls <see cref="Eigh"/>; it is a convenience, not a band algorithm.</remarks>
    public static double[] EigvalsBanded(NdArray ab, bool lower = false)
    {
        var dense = BandToDenseSymmetric(ab, lower);
        return Eigh(dense, eigvalsOnly: true).Values;
    }

    // ================================================================ SVD family

    /// <summary>
    /// Singular value decomposition, scipy's <c>svd</c>: <c>A = U diag(S) Vh</c>, S descending.
    /// </summary>
    /// <param name="a">An m×n matrix.</param>
    /// <param name="fullMatrices">U m×m and Vh n×n (default, as scipy); otherwise k = min(m, n) columns/rows.</param>
    /// <remarks>
    /// Delegates to GraviNum's Golub–Kahan–Reinsch <c>Decomposition.Svd</c> (thin). The full factors
    /// are completed with an orthonormal basis of the complement from a Householder QR of the thin
    /// factor; those extra columns are arbitrary (as in LAPACK) — only their span is defined.
    /// </remarks>
    public static LinalgSvdResult Svd(NdArray a, bool fullMatrices = true)
    {
        LinalgKernels.RequireMatrix(a, nameof(Svd));
        var m = a.Shape[0];
        var n = a.Shape[1];
        var svd = Decomposition.Svd(a);
        var s = svd.SingularValues.ToArray();
        var u = svd.U;
        var v = svd.V;
        if (fullMatrices)
        {
            u = CompleteBasis(u);
            v = CompleteBasis(v);
        }
        return new LinalgSvdResult(u.AsContiguous(), s, v.T.Copy());
    }

    /// <summary>Singular values only, descending; scipy's <c>svdvals</c>.</summary>
    public static double[] Svdvals(NdArray a)
    {
        LinalgKernels.RequireMatrix(a, nameof(Svdvals));
        return Decomposition.SingularValues(a).ToArray();
    }

    /// <summary>The diagonal Σ (m×n) from singular values, scipy's <c>diagsvd</c>.</summary>
    public static NdArray Diagsvd(double[] s, int m, int n)
    {
        var r = new double[m * n];
        for (var i = 0; i < Math.Min(s.Length, Math.Min(m, n)); i++) r[i * n + i] = s[i];
        return new NdArray(r, m, n);
    }

    /// <summary>Extends an m×k matrix with orthonormal columns to an m×m orthogonal matrix.</summary>
    internal static NdArray CompleteBasis(NdArray q)
    {
        var m = q.Shape[0];
        var k = q.Shape[1];
        if (k >= m) return q;
        var full = Decomposition.Qr(q.AsContiguous(), reduced: false).Q.AsContiguous().AsSpan();
        var qs = q.AsContiguous().AsSpan();
        var r = new double[m * m];
        for (var i = 0; i < m; i++)
        {
            for (var j = 0; j < k; j++) r[i * m + j] = qs[i * k + j];
            for (var j = k; j < m; j++) r[i * m + j] = full[i * m + j];
        }
        return new NdArray(r, m, m);
    }

    /// <summary>
    /// Polar decomposition, scipy's <c>polar</c>: <c>A = U P</c> (side "right") or <c>A = P U</c>
    /// ("left"), U with orthonormal columns/rows and P symmetric positive semi-definite; from the SVD.
    /// </summary>
    public static PolarResult Polar(NdArray a, string side = "right")
    {
        LinalgKernels.RequireMatrix(a, nameof(Polar));
        if (side is not ("right" or "left")) throw new ArgumentException("Polar: side must be 'right' or 'left'.");
        var svd = Decomposition.Svd(a);
        var w = svd.U;                 // m x k
        var v = svd.V;                 // n x k
        var s = svd.SingularValues.ToArray();
        var u = LinalgKernels.Mul(w, v.T);
        var basis = side == "right" ? v : w;
        var bs = basis.AsContiguous();
        var rows = bs.Shape[0];
        var k = s.Length;
        var scaled = bs.Copy();
        var sp = scaled.AsSpan();
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < k; j++) sp[i * k + j] *= s[j];
        var p = LinalgKernels.Mul(scaled, bs.T);
        return new PolarResult(u, p);
    }

    /// <summary>
    /// Orthonormal basis of the null space from the SVD, scipy's <c>null_space</c>. Singular values
    /// below <c>rcond·max(s)</c> count as zero; default <c>rcond = eps·max(m, n)</c>.
    /// </summary>
    public static NdArray NullSpace(NdArray a, double? rcond = null)
    {
        LinalgKernels.RequireMatrix(a, nameof(NullSpace));
        var m = a.Shape[0];
        var n = a.Shape[1];
        var svd = Svd(a, fullMatrices: true);
        var tol = (rcond ?? LinalgKernels.Eps * Math.Max(m, n)) * (svd.S.Length > 0 ? svd.S[0] : 0.0);
        var num = 0;
        foreach (var s in svd.S) if (s > tol) num++;
        var vh = svd.Vh.AsContiguous().AsSpan();
        var cols = n - num;
        var r = new double[n * cols];
        for (var j = 0; j < cols; j++)
            for (var i = 0; i < n; i++) r[i * cols + j] = vh[(num + j) * n + i];
        return new NdArray(r, n, cols);
    }

    /// <summary>
    /// Orthonormal basis of the range from the SVD, scipy's <c>orth</c>; same rank cut-off as
    /// <see cref="NullSpace"/>.
    /// </summary>
    public static NdArray Orth(NdArray a, double? rcond = null)
    {
        LinalgKernels.RequireMatrix(a, nameof(Orth));
        var m = a.Shape[0];
        var n = a.Shape[1];
        var svd = Decomposition.Svd(a);
        var s = svd.SingularValues.ToArray();
        var tol = (rcond ?? LinalgKernels.Eps * Math.Max(m, n)) * (s.Length > 0 ? s[0] : 0.0);
        var num = 0;
        foreach (var x in s) if (x > tol) num++;
        var u = svd.U.AsContiguous().AsSpan();
        var k = s.Length;
        var r = new double[m * num];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < num; j++) r[i * num + j] = u[i * k + j];
        return new NdArray(r, m, num);
    }

    /// <summary>
    /// Principal angles between the column spaces of A and B in radians, largest first; scipy's
    /// <c>subspace_angles</c> (Knyazev–Argentati: arccos for large angles, arcsin of the
    /// complement's singular values for small ones, where arccos has lost all precision).
    /// </summary>
    public static double[] SubspaceAngles(NdArray a, NdArray b)
    {
        LinalgKernels.RequireMatrix(a, nameof(SubspaceAngles));
        LinalgKernels.RequireMatrix(b, nameof(SubspaceAngles));
        if (a.Shape[0] != b.Shape[0]) throw new ArgumentException("SubspaceAngles: a and b must have the same number of rows.");
        var qa = Orth(a);
        var qb = Orth(b);
        var qaHqb = LinalgKernels.Mul(qa.T, qb);
        var sigma = Svdvals(qaHqb);
        NdArray bm = qa.Shape[1] >= qb.Shape[1]
            ? LinalgKernels.Combine(1.0, qb, -1.0, LinalgKernels.Mul(qa, qaHqb))
            : LinalgKernels.Combine(1.0, qa, -1.0, LinalgKernels.Mul(qb, qaHqb.T));
        var k = sigma.Length;
        var anyMask = false;
        foreach (var s in sigma) if (s * s >= 0.5) anyMask = true;
        var muArcsin = new double[k];
        if (anyMask)
        {
            var sb = Svdvals(bm);
            for (var i = 0; i < k; i++) muArcsin[i] = i < sb.Length ? Math.Asin(Math.Clamp(sb[i], -1.0, 1.0)) : 0.0;
        }
        var theta = new double[k];
        for (var i = 0; i < k; i++)
            theta[i] = sigma[i] * sigma[i] >= 0.5 ? muArcsin[i] : Math.Acos(Math.Clamp(sigma[k - 1 - i], -1.0, 1.0));
        return theta;
    }

    // ================================================================ helpers

    internal static double[] SymmetricFromTriangle(NdArray a, bool lower)
    {
        var n = a.Shape[0];
        var c = a.ToArray();
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
            {
                if (lower) c[i * n + j] = c[j * n + i];
                else c[j * n + i] = c[i * n + j];
            }
        return c;
    }

    internal static NdArray BandToDenseSymmetric(NdArray ab, bool lower)
    {
        LinalgKernels.RequireMatrix(ab, "band storage");
        var rows = ab.Shape[0];
        var n = ab.Shape[1];
        var s = ab.AsContiguous().AsSpan();
        var a = new double[n * n];
        var u = rows - 1;
        for (var j = 0; j < n; j++)
            for (var r = 0; r < rows; r++)
            {
                var i = lower ? j + r : j - u + r;
                if (i < 0 || i >= n) continue;
                var v = s[r * n + j];
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        return new NdArray(a, n, n);
    }
}
