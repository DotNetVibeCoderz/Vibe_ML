using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Sparse;

/// <summary>Which eigenvalues <see cref="SparseLinalg.Eigsh"/> looks for (SciPy's <c>which</c>).</summary>
public enum EigshWhich
{
    /// <summary>Largest magnitude.</summary>
    LM,

    /// <summary>Smallest magnitude (slow without <c>sigma</c>; prefer <c>sigma: 0</c>).</summary>
    SM,

    /// <summary>Largest algebraic.</summary>
    LA,

    /// <summary>Smallest algebraic.</summary>
    SA,

    /// <summary>Both ends: half from the top, half from the bottom (the extra one from the top).</summary>
    BE,
}

/// <summary>Which eigenvalues <see cref="SparseLinalg.Eigs"/> looks for (SciPy's <c>which</c>).</summary>
public enum EigsWhich
{
    /// <summary>Largest magnitude.</summary>
    LM,

    /// <summary>Smallest magnitude.</summary>
    SM,

    /// <summary>Largest real part.</summary>
    LR,

    /// <summary>Smallest real part.</summary>
    SR,

    /// <summary>
    /// Largest imaginary part in magnitude — ARPACK's (and so SciPy's) actual criterion, which keeps
    /// complex-conjugate pairs together; in real arithmetic a pair cannot be separated anyway.
    /// </summary>
    LI,

    /// <summary>Smallest imaginary part in magnitude (see <see cref="LI"/>).</summary>
    SI,
}

/// <summary>Which singular values <see cref="SparseLinalg.Svds"/> looks for.</summary>
public enum SvdsWhich
{
    /// <summary>Largest.</summary>
    LM,

    /// <summary>Smallest.</summary>
    SM,
}

/// <summary>Eigenpairs from <see cref="SparseLinalg.Eigsh"/>.</summary>
/// <param name="Values">Eigenvalues in ascending order (SciPy's order).</param>
/// <param name="Vectors">Unit eigenvectors as columns (n×k), aligned with <paramref name="Values"/>; null when not requested.</param>
/// <param name="Restarts">Restart cycles used.</param>
/// <param name="MatVecs">Operator applications used.</param>
public sealed record EigshResult(double[] Values, NdArray? Vectors, int Restarts, int MatVecs);

/// <summary>Eigenpairs from <see cref="SparseLinalg.Eigs"/>.</summary>
/// <param name="Values">Eigenvalues, ordered by the <c>which</c> criterion (best first).</param>
/// <param name="Vectors">Unit eigenvectors as columns (n×k); null when not requested.</param>
/// <param name="Restarts">Restart cycles used.</param>
/// <param name="MatVecs">Operator applications used.</param>
public sealed record EigsResult(Complex[] Values, ComplexNdArray? Vectors, int Restarts, int MatVecs);

/// <summary>Partial SVD from <see cref="SparseLinalg.Svds"/>: <c>A ≈ U diag(S) Vt</c>.</summary>
/// <param name="U">Left singular vectors (m×k); null when not requested.</param>
/// <param name="S">Singular values in ascending order (SciPy's order).</param>
/// <param name="Vt">Right singular vectors as rows (k×n); null when not requested.</param>
public sealed record SvdsResult(NdArray? U, double[] S, NdArray? Vt);

/// <summary>Thrown when ARPACK-style iteration does not converge within <c>maxiter</c> restarts (SciPy's <c>ArpackNoConvergence</c>).</summary>
public sealed class SparseEigenNoConvergenceException(string message, int converged) : Exception(message)
{
    /// <summary>How many of the requested eigenpairs had converged.</summary>
    public int Converged { get; } = converged;
}

public static partial class SparseLinalg
{
    private static readonly double Eps23 = Math.Pow(Eps, 2.0 / 3.0);

    /// <summary>
    /// A few eigenpairs of a real symmetric operator — <c>scipy.sparse.linalg.eigsh</c>.
    /// </summary>
    /// <param name="a">Symmetric operator (n×n).</param>
    /// <param name="k">Number of eigenpairs; must be below n.</param>
    /// <param name="which">Which end of the spectrum.</param>
    /// <param name="sigma">
    /// Shift-invert: iterate with <c>(A − σI)⁻¹</c> (a sparse LU of <c>A − σI</c>, or
    /// <paramref name="opInv"/>), so <paramref name="which"/> applies to <c>1/(λ − σ)</c> and LM
    /// returns the eigenvalues nearest σ — the fast way to get interior or smallest eigenvalues.
    /// </param>
    /// <param name="v0">Starting vector (default: a fixed pseudo-random vector, so results are reproducible).</param>
    /// <param name="ncv">Krylov basis size (default <c>max(2k + 1, 20)</c>, capped at n).</param>
    /// <param name="maxiter">Maximum restarts (default <c>10 n</c>).</param>
    /// <param name="tol">Relative accuracy of the Ritz values; 0 means machine precision.</param>
    /// <param name="returnEigenvectors">Also compute eigenvectors.</param>
    /// <param name="opInv">Explicit <c>(A − σI)⁻¹</c> for operators that are not sparse matrices.</param>
    /// <remarks>
    /// Thick-restart Lanczos (Wu &amp; Simon 2000), equivalently Krylov–Schur for symmetric
    /// matrices, with full classical Gram–Schmidt reorthogonalisation applied twice. Full
    /// reorthogonalisation costs O(n·ncv) per step but removes the ghost eigenvalues of plain
    /// Lanczos, which is what makes the result trustworthy without ARPACK's selective schemes.
    /// Convergence uses ARPACK's test <c>‖r‖ ≤ tol·max(ε^{2/3}, |θ|)</c>. When <c>ncv ≥ n</c> the
    /// problem is solved densely (the Krylov space would be the whole space anyway).
    /// </remarks>
    public static EigshResult Eigsh(ILinearOperator a, int k = 6, EigshWhich which = EigshWhich.LM, double? sigma = null,
        double[]? v0 = null, int? ncv = null, int? maxiter = null, double tol = 0, bool returnEigenvectors = true, ILinearOperator? opInv = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Rows != a.Columns) throw new ArgumentException("Eigsh needs a square operator.");
        var n = a.Rows;
        if (k <= 0 || k >= n) throw new ArgumentException($"k must be between 1 and n - 1 = {n - 1}, got {k}.");
        var op = sigma is null ? a : (opInv ?? ShiftInvert(a, sigma.Value));
        var m = Math.Min(n, ncv ?? Math.Max(2 * k + 1, 20));
        if (m <= k) throw new ArgumentException("ncv must exceed k.");
        double[] theta;
        double[] vecs; // column-major n×k
        int restarts = 0, matvecs = 0;
        if (m >= n)
        {
            var dense = op.ToDenseArray();
            var sym = Symmetrize(dense.AsContiguous().ToArray(), n);
            var eig = Linalg.Eigh(new NdArray(sym, n, n));
            var order = RankSymmetric(eig.Values, which, k);
            theta = order.Take(k).Select(i => eig.Values[i]).ToArray();
            var vbuf = eig.Vectors.AsContiguous().ToArray();
            vecs = new double[n * k];
            for (var c = 0; c < k; c++)
                for (var i = 0; i < n; i++) vecs[c * n + i] = vbuf[i * n + order[c]];
            matvecs = n;
        }
        else
            (theta, vecs, restarts, matvecs) = ThickRestartLanczos(op, n, k, m, which, v0, maxiter ?? 10 * n, tol > 0 ? tol : Eps, returnEigenvectors);

        var lambda = theta.Select(t => sigma is null ? t : sigma.Value + 1.0 / t).ToArray();
        var idx = Enumerable.Range(0, k).OrderBy(i => lambda[i]).ToArray();
        var values = idx.Select(i => lambda[i]).ToArray();
        NdArray? vectors = null;
        if (returnEigenvectors)
        {
            var buf = new double[n * k];
            for (var c = 0; c < k; c++)
                for (var i = 0; i < n; i++) buf[i * k + c] = vecs[idx[c] * n + i];
            vectors = new NdArray(buf, n, k);
        }
        return new EigshResult(values, vectors, restarts, matvecs);
    }

    private static double[] Symmetrize(double[] a, int n)
    {
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
            {
                var v = 0.5 * (a[i * n + j] + a[j * n + i]);
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        return a;
    }

    private static ILinearOperator ShiftInvert(ILinearOperator a, double sigma)
    {
        if (a is not SparseMatrixBase s)
            throw new ArgumentException("Shift-invert needs a sparse matrix (to factorise A − σI) or an explicit opInv.");
        var shifted = s.ToCsr().Subtract(SparseMatrices.Identity(s.Rows).Scale(sigma));
        return Splu(shifted);
    }

    /// <summary>Indices of the values ordered from most to least wanted.</summary>
    private static int[] RankSymmetric(double[] v, EigshWhich which, int k)
    {
        var idx = Enumerable.Range(0, v.Length).ToArray();
        switch (which)
        {
            case EigshWhich.LM: return idx.OrderByDescending(i => Math.Abs(v[i])).ThenByDescending(i => v[i]).ToArray();
            case EigshWhich.SM: return idx.OrderBy(i => Math.Abs(v[i])).ThenBy(i => v[i]).ToArray();
            case EigshWhich.LA: return idx.OrderByDescending(i => v[i]).ToArray();
            case EigshWhich.SA: return idx.OrderBy(i => v[i]).ToArray();
            default:
            {
                // BE: alternate top and bottom; k/2 from the bottom, the rest from the top.
                var asc = idx.OrderBy(i => v[i]).ToArray();
                var result = new List<int>(v.Length);
                int lo = 0, hi = asc.Length - 1;
                var fromTop = k - k / 2;
                var fromBottom = k / 2;
                while (fromTop > 0 || fromBottom > 0)
                {
                    if (fromTop-- > 0 && lo <= hi) result.Add(asc[hi--]);
                    if (fromBottom-- > 0 && lo <= hi) result.Add(asc[lo++]);
                }
                // Unwanted remainder: closest to the middle are least wanted.
                while (lo <= hi)
                {
                    result.Add(asc[hi--]);
                    if (lo <= hi) result.Add(asc[lo++]);
                }
                return result.ToArray();
            }
        }
    }

    private static double[] StartVector(int n, double[]? v0)
    {
        if (v0 is not null)
        {
            if (v0.Length != n) throw new ArgumentException($"v0 has {v0.Length} entries, expected {n}.");
            return (double[])v0.Clone();
        }
        var rng = new GraviRandom(20240917);
        var v = new double[n];
        for (var i = 0; i < n; i++) v[i] = rng.Uniform(-1, 1);
        return v;
    }

    /// <summary>
    /// Orthogonalises <paramref name="w"/> against columns 0..<paramref name="count"/>−1 of the row-major
    /// basis <paramref name="v"/> (each basis vector contiguous), twice, accumulating the coefficients.
    /// </summary>
    private static void Orthogonalize(double[] v, int n, int count, double[] w, Span<double> coef)
    {
        coef.Slice(0, count).Clear();
        for (var pass = 0; pass < 2; pass++)
            for (var i = 0; i < count; i++)
            {
                var vi = v.AsSpan(i * n, n);
                var c = Dot(vi, w);
                coef[i] += c;
                Axpy(-c, vi, w);
            }
    }

    /// <summary>A unit vector orthogonal to the basis, used when the Krylov space becomes invariant.</summary>
    private static double[] FreshDirection(double[] v, int n, int count, GraviRandom rng)
    {
        var coef = new double[count];
        for (var attempt = 0; attempt < 5; attempt++)
        {
            var w = new double[n];
            for (var i = 0; i < n; i++) w[i] = rng.Uniform(-1, 1);
            Orthogonalize(v, n, count, w, coef);
            var nw = Nrm2(w);
            if (nw > 1e-8)
            {
                ScaleInPlace(w, 1 / nw);
                return w;
            }
        }
        throw new InvalidOperationException("Could not extend the Krylov basis: the space is exhausted.");
    }

    private static (double[] Theta, double[] Vecs, int Restarts, int MatVecs) ThickRestartLanczos(
        ILinearOperator op, int n, int k, int m, EigshWhich which, double[]? v0, int maxRestarts, double tol, bool wantVectors)
    {
        var v = new double[(m + 1) * n];
        var start = StartVector(n, v0);
        var s0 = Nrm2(start);
        if (s0 == 0) throw new ArgumentException("v0 must be non-zero.");
        for (var i = 0; i < n; i++) v[i] = start[i] / s0;
        var h = new double[(m + 1) * m]; // row-major (m+1)×m
        var coef = new double[m + 1];
        var w = new double[n];
        var rng = new GraviRandom(7);
        var j0 = 0;
        var matvecs = 0;
        for (var restart = 0; ; restart++)
        {
            for (var j = j0; j < m; j++)
            {
                op.MatVec(v.AsSpan(j * n, n), w);
                matvecs++;
                var wn0 = Nrm2(w);
                Orthogonalize(v, n, j + 1, w, coef);
                for (var i = 0; i <= j; i++) h[i * m + j] = coef[i];
                var beta = Nrm2(w);
                if (beta <= 1e-12 * Math.Max(wn0, 1e-300) || beta == 0)
                {
                    h[(j + 1) * m + j] = 0;
                    FreshDirection(v, n, j + 1, rng).CopyTo(v, (j + 1) * n);
                }
                else
                {
                    h[(j + 1) * m + j] = beta;
                    for (var i = 0; i < n; i++) v[(j + 1) * n + i] = w[i] / beta;
                }
            }
            // Ritz pairs of the symmetric m×m projection.
            var t = new double[m * m];
            for (var i = 0; i < m; i++)
                for (var j = 0; j < m; j++) t[i * m + j] = 0.5 * (h[i * m + j] + h[j * m + i]);
            var eig = Linalg.Eigh(new NdArray(t, m, m));
            var y = eig.Vectors.AsContiguous().ToArray(); // m×m, columns are eigenvectors
            var thetaAll = eig.Values;
            var bm = h[m * m + (m - 1)];
            var rank = RankSymmetric(thetaAll, which, k);
            var converged = 0;
            for (var c = 0; c < k; c++)
            {
                var i = rank[c];
                var res = Math.Abs(bm * y[(m - 1) * m + i]);
                if (res <= tol * Math.Max(Eps23, Math.Abs(thetaAll[i]))) converged++;
            }
            if (converged == k || restart >= maxRestarts)
            {
                if (converged < k)
                    throw new SparseEigenNoConvergenceException($"Eigsh: {converged} of {k} eigenpairs converged after {restart} restarts.", converged);
                var theta = new double[k];
                var vecs = new double[wantVectors ? n * k : 0];
                for (var c = 0; c < k; c++)
                {
                    var i = rank[c];
                    theta[c] = thetaAll[i];
                    if (!wantVectors) continue;
                    var dst = vecs.AsSpan(c * n, n);
                    for (var r = 0; r < m; r++) Axpy(y[r * m + i], v.AsSpan(r * n, n), dst);
                    ScaleInPlace(dst, 1 / Nrm2(dst));
                }
                if (wantVectors) VerifyRitzPairs(op, n, theta, vecs, thetaAll, tol);
                return (theta, vecs, restart, matvecs);
            }
            // Thick restart: keep the best `keep` Ritz vectors plus the residual direction.
            var keep = Math.Min(m - 1, k + Math.Max(1, Math.Min(converged, (m - k) / 2)));
            keep = Math.Max(keep, Math.Min(m - 1, k + (m - k) / 2));
            var newV = new double[(m + 1) * n];
            for (var c = 0; c < keep; c++)
            {
                var i = rank[c];
                var dst = newV.AsSpan(c * n, n);
                for (var r = 0; r < m; r++) Axpy(y[r * m + i], v.AsSpan(r * n, n), dst);
            }
            v.AsSpan(m * n, n).CopyTo(newV.AsSpan(keep * n, n));
            Array.Copy(newV, v, newV.Length);
            Array.Clear(h);
            for (var c = 0; c < keep; c++)
            {
                var i = rank[c];
                h[c * m + c] = thetaAll[i];
                h[keep * m + c] = bm * y[(m - 1) * m + i];
            }
            j0 = keep;
        }
    }

    /// <summary>
    /// Re-checks converged Ritz pairs against the operator itself. The convergence test only reads
    /// the projected matrix; if the Krylov relation has drifted (lost orthogonality over many
    /// restarts) the projection can report pairs that are not eigenpairs at all. That must throw,
    /// not return a plausible wrong answer.
    /// </summary>
    private static double RitzTolerance(double tol, double value, double scale)
        => Math.Max(Math.Sqrt(tol), 1e-8) * Math.Max(Math.Abs(value), scale);

    private static void VerifyRitzPairs(ILinearOperator op, int n, double[] theta, double[] vecs, double[] allTheta, double tol)
    {
        var scale = allTheta.Max(Math.Abs);
        var y = new double[n];
        for (var c = 0; c < theta.Length; c++)
        {
            var x = vecs.AsSpan(c * n, n);
            op.MatVec(x, y);
            Axpy(-theta[c], x, y);
            if (Nrm2(y) > RitzTolerance(tol, theta[c], scale))
                throw new SparseEigenNoConvergenceException($"Ritz pair {c} has true residual {Nrm2(y):G3}: the Lanczos basis lost orthogonality.", c);
        }
    }

    private static void VerifyComplexRitzPairs(ILinearOperator op, int n, Complex[] mu, Complex[] vecs, Complex[] allMu, double tol)
    {
        var scale = allMu.Max(z => z.Magnitude);
        var re = new double[n]; var im = new double[n]; var are = new double[n]; var aim = new double[n];
        for (var c = 0; c < mu.Length; c++)
        {
            for (var i = 0; i < n; i++) { re[i] = vecs[c * n + i].Real; im[i] = vecs[c * n + i].Imaginary; }
            op.MatVec(re, are);
            op.MatVec(im, aim);
            var r = 0.0;
            for (var i = 0; i < n; i++)
            {
                var d = new Complex(are[i], aim[i]) - mu[c] * vecs[c * n + i];
                r += d.Real * d.Real + d.Imaginary * d.Imaginary;
            }
            r = Math.Sqrt(r);
            if (r > RitzTolerance(tol, mu[c].Magnitude, scale))
                throw new SparseEigenNoConvergenceException($"Ritz pair {c} has true residual {r:G3}: the Arnoldi basis lost orthogonality.", c);
        }
    }

    /// <summary>
    /// A few eigenpairs of a general real square operator — <c>scipy.sparse.linalg.eigs</c>.
    /// </summary>
    /// <param name="a">Square operator (n×n).</param>
    /// <param name="k">Number of eigenpairs; must be below n − 1 (as ARPACK requires).</param>
    /// <param name="which">Which part of the spectrum.</param>
    /// <param name="sigma">Real shift for shift-invert mode (<paramref name="which"/> then applies to <c>1/(λ − σ)</c>).</param>
    /// <param name="v0">Starting vector.</param>
    /// <param name="ncv">Krylov basis size (default <c>max(2k + 1, 20)</c>, capped at n).</param>
    /// <param name="maxiter">Maximum restarts (default <c>10 n</c>).</param>
    /// <param name="tol">Relative accuracy; 0 means machine precision.</param>
    /// <param name="returnEigenvectors">Also compute eigenvectors.</param>
    /// <param name="opInv">Explicit <c>(A − σI)⁻¹</c>.</param>
    /// <remarks>
    /// Implicitly restarted Arnoldi (Sorensen 1992, the method of ARPACK): after each Arnoldi cycle
    /// the unwanted Ritz values are applied as exact shifts — single shifts for real ones, double
    /// (real-arithmetic) shifts for complex-conjugate pairs — and the wanted set is never allowed
    /// to split a conjugate pair. The shifted QR steps are done explicitly on the small Hessenberg
    /// matrix. The projected eigenproblem uses SciNet's dense <c>Linalg.Eig</c>. Complex
    /// <paramref name="sigma"/> is not supported.
    /// </remarks>
    public static EigsResult Eigs(ILinearOperator a, int k = 6, EigsWhich which = EigsWhich.LM, double? sigma = null,
        double[]? v0 = null, int? ncv = null, int? maxiter = null, double tol = 0, bool returnEigenvectors = true, ILinearOperator? opInv = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Rows != a.Columns) throw new ArgumentException("Eigs needs a square operator.");
        var n = a.Rows;
        if (k <= 0 || k >= n - 1) throw new ArgumentException($"k must be between 1 and n - 2 = {n - 2}, got {k}.");
        var op = sigma is null ? a : (opInv ?? ShiftInvert(a, sigma.Value));
        var m = Math.Min(n, ncv ?? Math.Max(2 * k + 1, 20));
        if (m <= k + 1) throw new ArgumentException("ncv must exceed k + 1.");
        Complex[] mu;
        Complex[]? vecs;
        int restarts = 0, matvecs = 0;
        if (m >= n)
        {
            var eig = Linalg.Eig(op.ToDenseArray());
            var rank = RankComplex(eig.Values, which);
            mu = rank.Take(k).Select(i => eig.Values[i]).ToArray();
            vecs = null;
            if (returnEigenvectors)
            {
                vecs = new Complex[n * k];
                for (var c = 0; c < k; c++)
                    for (var i = 0; i < n; i++) vecs[c * n + i] = eig.Vectors[i, rank[c]];
            }
            matvecs = n;
        }
        else
            (mu, vecs, restarts, matvecs) = ImplicitlyRestartedArnoldi(op, n, k, m, which, v0, maxiter ?? 10 * n, tol > 0 ? tol : Eps, returnEigenvectors);

        var values = mu.Select(t => sigma is null ? t : sigma.Value + 1.0 / t).ToArray();
        ComplexNdArray? vectors = null;
        if (vecs is not null)
        {
            var buf = new Complex[n * k];
            for (var c = 0; c < k; c++)
                for (var i = 0; i < n; i++) buf[i * k + c] = vecs[c * n + i];
            vectors = new ComplexNdArray(buf, n, k);
        }
        return new EigsResult(values, vectors, restarts, matvecs);
    }

    private static int[] RankComplex(Complex[] v, EigsWhich which)
    {
        var idx = Enumerable.Range(0, v.Length);
        return which switch
        {
            EigsWhich.LM => idx.OrderByDescending(i => v[i].Magnitude).ThenByDescending(i => v[i].Imaginary).ToArray(),
            EigsWhich.SM => idx.OrderBy(i => v[i].Magnitude).ThenByDescending(i => v[i].Imaginary).ToArray(),
            EigsWhich.LR => idx.OrderByDescending(i => v[i].Real).ThenByDescending(i => v[i].Imaginary).ToArray(),
            EigsWhich.SR => idx.OrderBy(i => v[i].Real).ThenByDescending(i => v[i].Imaginary).ToArray(),
            EigsWhich.LI => idx.OrderByDescending(i => Math.Abs(v[i].Imaginary)).ThenByDescending(i => v[i].Real).ThenByDescending(i => v[i].Imaginary).ToArray(),
            _ => idx.OrderBy(i => Math.Abs(v[i].Imaginary)).ThenByDescending(i => v[i].Real).ThenByDescending(i => v[i].Imaginary).ToArray(),
        };
    }

    private static bool IsConjugatePair(Complex a, Complex b)
        => a.Imaginary != 0 && Math.Abs(a.Real - b.Real) <= 1e-12 * Math.Max(1, a.Magnitude)
           && Math.Abs(a.Imaginary + b.Imaginary) <= 1e-12 * Math.Max(1, a.Magnitude);

    private static (Complex[] Mu, Complex[]? Vecs, int Restarts, int MatVecs) ImplicitlyRestartedArnoldi(
        ILinearOperator op, int n, int k, int m, EigsWhich which, double[]? v0, int maxRestarts, double tol, bool wantVectors)
    {
        var v = new double[(m + 1) * n];
        var start = StartVector(n, v0);
        var s0 = Nrm2(start);
        if (s0 == 0) throw new ArgumentException("v0 must be non-zero.");
        for (var i = 0; i < n; i++) v[i] = start[i] / s0;
        var h = new double[(m + 1) * m];
        var coef = new double[m + 1];
        var w = new double[n];
        var rng = new GraviRandom(11);
        var j0 = 0;
        var matvecs = 0;
        for (var restart = 0; ; restart++)
        {
            for (var j = j0; j < m; j++)
            {
                op.MatVec(v.AsSpan(j * n, n), w);
                matvecs++;
                var wn0 = Nrm2(w);
                Orthogonalize(v, n, j + 1, w, coef);
                for (var i = 0; i <= j; i++) h[i * m + j] = coef[i];
                var beta = Nrm2(w);
                if (beta <= 1e-12 * Math.Max(wn0, 1e-300) || beta == 0)
                {
                    h[(j + 1) * m + j] = 0;
                    FreshDirection(v, n, j + 1, rng).CopyTo(v, (j + 1) * n);
                }
                else
                {
                    h[(j + 1) * m + j] = beta;
                    for (var i = 0; i < n; i++) v[(j + 1) * n + i] = w[i] / beta;
                }
            }
            var hm = new double[m * m];
            Array.Copy(h, hm, m * m);
            var eig = Linalg.Eig(new NdArray(hm, m, m));
            var mu = eig.Values;
            var y = eig.Vectors;
            var bm = h[m * m + (m - 1)];
            var rank = RankComplex(mu, which);
            var converged = 0;
            for (var c = 0; c < k; c++)
            {
                var i = rank[c];
                var res = Math.Abs(bm) * y[m - 1, i].Magnitude;
                if (res <= tol * Math.Max(Eps23, mu[i].Magnitude)) converged++;
            }
            if (converged == k || restart >= maxRestarts)
            {
                if (converged < k)
                    throw new SparseEigenNoConvergenceException($"Eigs: {converged} of {k} eigenpairs converged after {restart} restarts.", converged);
                var outMu = new Complex[k];
                Complex[]? vecs = wantVectors ? new Complex[n * k] : null;
                for (var c = 0; c < k; c++)
                {
                    var i = rank[c];
                    outMu[c] = mu[i];
                    if (vecs is null) continue;
                    var norm = 0.0;
                    for (var r = 0; r < n; r++)
                    {
                        var s = Complex.Zero;
                        for (var q = 0; q < m; q++) s += v[q * n + r] * y[q, i];
                        vecs[c * n + r] = s;
                        norm += s.Real * s.Real + s.Imaginary * s.Imaginary;
                    }
                    norm = Math.Sqrt(norm);
                    for (var r = 0; r < n; r++) vecs[c * n + r] /= norm;
                }
                if (vecs is not null) VerifyComplexRitzPairs(op, n, outMu, vecs, mu, tol);
                return (outMu, vecs, restart, matvecs);
            }

            // Number kept; never split a complex-conjugate pair across the wanted/unwanted boundary.
            var kk = Math.Min(m - 2, k + Math.Max(1, (m - k) / 2));
            if (kk < m && IsConjugatePair(mu[rank[kk - 1]], mu[rank[kk]])) kk = kk + 1 < m ? kk + 1 : kk - 1;
            if (kk < k) kk = k;

            // Apply the unwanted Ritz values as exact shifts to H (explicit shifted QR steps).
            var hmat = new double[m, m];
            for (var i = 0; i < m; i++)
                for (var j = 0; j < m; j++) hmat[i, j] = h[i * m + j];
            var qtot = new double[m, m];
            for (var i = 0; i < m; i++) qtot[i, i] = 1;
            var used = new bool[m];
            for (var c = kk; c < m; c++)
            {
                var idx = rank[c];
                if (used[idx]) continue;
                used[idx] = true;
                var s = mu[idx];
                if (s.Imaginary == 0)
                    ApplyShift(hmat, qtot, m, s.Real, 0, single: true);
                else
                {
                    var paired = false;
                    for (var d = kk; d < m; d++)
                        if (!used[rank[d]] && IsConjugatePair(s, mu[rank[d]])) { used[rank[d]] = true; paired = true; break; }
                    // A complex shift whose conjugate is wanted cannot be applied in real arithmetic
                    // without filtering the wanted one out too: skip it.
                    if (!paired) continue;
                    ApplyShift(hmat, qtot, m, 2 * s.Real, s.Real * s.Real + s.Imaginary * s.Imaginary, single: false);
                }
            }
            // f_new = V Q[:, kk] H[kk, kk-1] + f Q[m-1, kk-1], with f = bm v_m.
            var f = new double[n];
            var sub = hmat[kk, kk - 1];
            for (var q = 0; q < m; q++) Axpy(qtot[q, kk] * sub, v.AsSpan(q * n, n), f);
            Axpy(bm * qtot[m - 1, kk - 1], v.AsSpan(m * n, n), f);
            var newV = new double[(m + 1) * n];
            for (var c = 0; c < kk; c++)
            {
                var dst = newV.AsSpan(c * n, n);
                for (var q = 0; q < m; q++) Axpy(qtot[q, c], v.AsSpan(q * n, n), dst);
            }
            Array.Copy(newV, v, newV.Length);
            Array.Clear(h);
            for (var i = 0; i < kk; i++)
                for (var j = Math.Max(0, i - 1); j < kk; j++) h[i * m + j] = hmat[i, j];
            // Re-orthogonalise f; the coefficients are O(ε) drift and are dropped so H stays Hessenberg.
            Orthogonalize(v, n, kk, f, coef);
            var fn = Nrm2(f);
            if (fn <= 1e-14 * Math.Max(1, Math.Abs(bm)))
            {
                h[kk * m + kk - 1] = 0;
                FreshDirection(v, n, kk, rng).CopyTo(v, kk * n);
            }
            else
            {
                h[kk * m + kk - 1] = fn;
                for (var i = 0; i < n; i++) v[kk * n + i] = f[i] / fn;
            }
            j0 = kk;
        }
    }

    /// <summary>
    /// One explicit shifted QR step on the m×m Hessenberg <paramref name="h"/>: QR-factorise
    /// <c>H − μI</c> (single) or <c>H² − sH + tI</c> (double, real arithmetic for a conjugate
    /// pair), then <c>H ← QᵀHQ</c> and <c>Qtot ← Qtot Q</c>. Entries below the subdiagonal are
    /// rounding noise afterwards and are zeroed.
    /// </summary>
    private static void ApplyShift(double[,] h, double[,] qtot, int m, double s, double t, bool single)
    {
        var mat = new double[m * m];
        if (single)
        {
            for (var i = 0; i < m; i++)
                for (var j = 0; j < m; j++) mat[i * m + j] = h[i, j] - (i == j ? s : 0);
        }
        else
        {
            for (var i = 0; i < m; i++)
                for (var j = 0; j < m; j++)
                {
                    var acc = 0.0;
                    for (var l = 0; l < m; l++) acc += h[i, l] * h[l, j];
                    mat[i * m + j] = acc - s * h[i, j] + (i == j ? t : 0);
                }
        }
        var q = Decomposition.Qr(new NdArray(mat, m, m)).Q.AsContiguous().ToArray();
        var tmp = new double[m, m];
        // tmp = H Q
        for (var i = 0; i < m; i++)
            for (var j = 0; j < m; j++)
            {
                var acc = 0.0;
                for (var l = 0; l < m; l++) acc += h[i, l] * q[l * m + j];
                tmp[i, j] = acc;
            }
        // H = Qᵀ tmp
        for (var i = 0; i < m; i++)
            for (var j = 0; j < m; j++)
            {
                if (i > j + 1) { h[i, j] = 0; continue; }
                var acc = 0.0;
                for (var l = 0; l < m; l++) acc += q[l * m + i] * tmp[l, j];
                h[i, j] = acc;
            }
        var qt = new double[m, m];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < m; j++)
            {
                var acc = 0.0;
                for (var l = 0; l < m; l++) acc += qtot[i, l] * q[l * m + j];
                qt[i, j] = acc;
            }
        Array.Copy(qt, qtot, qt.Length);
    }

    /// <summary>
    /// A few singular triplets of a sparse matrix or operator — <c>scipy.sparse.linalg.svds</c>.
    /// </summary>
    /// <param name="a">The m×n operator (needs <c>RMatVec</c>).</param>
    /// <param name="k">Number of singular values; below <c>min(m, n)</c>.</param>
    /// <param name="which">Largest or smallest.</param>
    /// <param name="ncv">Lanczos basis size for the underlying <see cref="Eigsh"/>.</param>
    /// <param name="tol">Accuracy of the eigenvalues of the normal operator.</param>
    /// <param name="maxiter">Maximum restarts.</param>
    /// <param name="returnSingularVectors">Also compute U and Vt.</param>
    /// <param name="v0">Starting vector for the normal-operator Lanczos.</param>
    /// <remarks>
    /// Runs <see cref="Eigsh"/> on the smaller of <c>AᵀA</c> and <c>AAᵀ</c> (as SciPy's ARPACK
    /// path does), then recovers the other factor as <c>A v / σ</c> and re-orthonormalises it.
    /// Squaring is the accuracy trade-off: a singular value σ comes out with absolute error about
    /// <c>ε·σ_max²/σ</c>, so values below roughly <c>√ε·σ_max</c> are unreliable and the smallest
    /// singular values of an ill-conditioned matrix should not be trusted to more than a few digits.
    /// </remarks>
    public static SvdsResult Svds(ILinearOperator a, int k = 6, SvdsWhich which = SvdsWhich.LM, int? ncv = null, double tol = 0,
        int? maxiter = null, bool returnSingularVectors = true, double[]? v0 = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        int m = a.Rows, n = a.Columns;
        if (k <= 0 || k >= Math.Min(m, n)) throw new ArgumentException($"k must be between 1 and min(m, n) - 1 = {Math.Min(m, n) - 1}.");
        var tall = m >= n;
        var dim = tall ? n : m;
        var inner = tall ? m : n;
        MatVecFunction normal = (x, y) =>
        {
            var t = new double[inner];
            if (tall) { a.MatVec(x, t); a.RMatVec(t, y); }
            else { a.RMatVec(x, t); a.MatVec(t, y); }
        };
        var gram = new LinearOperator(dim, dim, normal, normal);
        var eig = Eigsh(gram, k, which == SvdsWhich.LM ? EigshWhich.LA : EigshWhich.SA, ncv: ncv, tol: tol, maxiter: maxiter, v0: v0,
            returnEigenvectors: true);
        var s = eig.Values.Select(l => Math.Sqrt(Math.Max(l, 0))).ToArray(); // ascending already
        if (!returnSingularVectors) return new SvdsResult(null, s, null);
        var x = eig.Vectors!.AsContiguous().ToArray(); // dim×k
        var other = new double[inner * k];
        var col = new double[dim];
        var res = new double[inner];
        for (var c = 0; c < k; c++)
        {
            for (var i = 0; i < dim; i++) col[i] = x[i * k + c];
            if (tall) a.MatVec(col, res); else a.RMatVec(col, res);
            var nrm = Nrm2(res);
            for (var i = 0; i < inner; i++) other[i * k + c] = nrm > 0 ? res[i] / nrm : 0;
        }
        // Re-orthonormalise the recovered factor (columns for tiny σ are otherwise noise).
        var q = Decomposition.Qr(new NdArray(other, inner, k)).Q.AsContiguous().ToArray();
        var qk = q.Length / inner;
        var otherOrtho = new double[inner * k];
        for (var c = 0; c < k; c++)
        {
            var dot = 0.0;
            for (var i = 0; i < inner; i++) dot += q[i * qk + c] * other[i * k + c];
            var sign = dot < 0 ? -1.0 : 1.0;
            for (var i = 0; i < inner; i++) otherOrtho[i * k + c] = sign * q[i * qk + c];
        }
        var small = new NdArray(x, dim, k);
        var big = new NdArray(otherOrtho, inner, k);
        return tall ? new SvdsResult(big, s, small.T.Copy()) : new SvdsResult(small, s, big.T.Copy());
    }
}
