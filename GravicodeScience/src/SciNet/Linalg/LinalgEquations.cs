using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Linalg
{
    /// <summary>
    /// Solves the Sylvester equation <c>A X + X B = Q</c>, scipy's <c>solve_sylvester</c>.
    /// </summary>
    /// <remarks>
    /// Bartels–Stewart: Schur forms of A and B, transform Q, solve the triangular equation
    /// column by column, transform back. The triangular stage uses the <em>complex</em> Schur forms
    /// (converted from the real ones), so each column is one upper-triangular solve with
    /// <c>R + S[j,j] I</c> instead of LAPACK <c>trsyl</c>'s 1×1/2×2 block cases. For real data the
    /// solution is real and is returned as such. Unique iff no eigenvalue of A equals minus one of B.
    /// </remarks>
    /// <exception cref="LinalgSingularMatrixException">A and −B share an eigenvalue.</exception>
    public static NdArray SolveSylvester(NdArray a, NdArray b, NdArray q)
    {
        LinalgKernels.RequireSquare(a, nameof(SolveSylvester));
        LinalgKernels.RequireSquare(b, nameof(SolveSylvester));
        LinalgKernels.RequireMatrix(q, nameof(SolveSylvester));
        var m = a.Shape[0];
        var n = b.Shape[0];
        if (q.Shape[0] != m || q.Shape[1] != n) throw new ArgumentException("SolveSylvester: q must be (rows of a) × (rows of b).");

        var (r, u) = LinalgSchurKernel.ComplexSchur(a);
        var (s, v) = LinalgSchurKernel.ComplexSchur(b);
        // F = U^H Q V
        var f = LinalgKernels.CMul(LinalgKernels.CMul(LinalgKernels.CConjTranspose(u, m, m), LinalgKernels.ToComplex(q.AsContiguous().AsSpan()), m, m, n), v, m, n, n);

        // R Y + Y S = F, S upper triangular: column j depends on columns < j.
        var y = new Complex[m * n];
        var col = new Complex[m];
        for (var j = 0; j < n; j++)
        {
            for (var i = 0; i < m; i++)
            {
                var acc = f[i * n + j];
                for (var k = 0; k < j; k++) acc -= y[i * n + k] * s[k * n + j];
                col[i] = acc;
            }
            UpperSolveChecked(r, m, col, s[j * n + j], nameof(SolveSylvester));
            for (var i = 0; i < m; i++) y[i * n + j] = col[i];
        }
        var x = LinalgKernels.CMul(LinalgKernels.CMul(u, y, m, m, n), LinalgKernels.CConjTranspose(v, n, n), m, n, n);
        return new NdArray(LinalgKernels.RealPart(x), m, n);
    }

    private static void UpperSolveChecked(Complex[] r, int n, Span<Complex> x, Complex shift, string caller)
    {
        var scale = 0.0;
        for (var i = 0; i < n; i++) scale = Math.Max(scale, Complex.Abs(r[i * n + i]));
        scale = Math.Max(scale, Complex.Abs(shift));
        for (var i = n - 1; i >= 0; i--)
        {
            var acc = x[i];
            for (var j = i + 1; j < n; j++) acc -= r[i * n + j] * x[j];
            var d = r[i * n + i] + shift;
            if (Complex.Abs(d) <= 1e-14 * Math.Max(scale, 1e-300))
                throw new LinalgSingularMatrixException($"{caller}: the equation is singular (eigenvalues sum to zero); no unique solution.");
            x[i] = acc / d;
        }
    }

    /// <summary>
    /// Solves the continuous Lyapunov equation <c>A X + X Aᵀ = Q</c>, scipy's
    /// <c>solve_continuous_lyapunov</c> (a Sylvester equation with B = Aᵀ).
    /// </summary>
    public static NdArray SolveContinuousLyapunov(NdArray a, NdArray q)
    {
        LinalgKernels.RequireSquare(a, nameof(SolveContinuousLyapunov));
        return SolveSylvester(a, a.T.Copy(), q);
    }

    /// <summary>
    /// Solves the discrete Lyapunov (Stein) equation <c>A X Aᵀ − X + Q = 0</c>, scipy's
    /// <c>solve_discrete_lyapunov</c>.
    /// </summary>
    /// <remarks>
    /// A direct Bartels–Stewart-style solve on the complex Schur form <c>A = U R Uᴴ</c>: with
    /// <c>Y = Uᴴ X U</c>, column j (taken from the last) satisfies
    /// <c>(I − conj(R[j,j]) R) y_j = f_j + R Σ_{l&gt;j} conj(R[j,l]) y_l</c>, one triangular solve per
    /// column, O(n³) in all, with no inverse of <c>A + I</c> as in scipy's bilinear route. Unique iff
    /// <c>λᵢ λ̄ⱼ ≠ 1</c> for all eigenvalue pairs; otherwise this throws.
    /// </remarks>
    public static NdArray SolveDiscreteLyapunov(NdArray a, NdArray q)
    {
        LinalgKernels.RequireSquare(a, nameof(SolveDiscreteLyapunov));
        LinalgKernels.RequireSquare(q, nameof(SolveDiscreteLyapunov));
        var n = a.Shape[0];
        if (q.Shape[0] != n) throw new ArgumentException("SolveDiscreteLyapunov: a and q must have the same shape.");
        var (r, u) = LinalgSchurKernel.ComplexSchur(a);
        var uh = LinalgKernels.CConjTranspose(u, n, n);
        var f = LinalgKernels.CMul(LinalgKernels.CMul(uh, LinalgKernels.ToComplex(q.AsContiguous().AsSpan()), n, n, n), u, n, n, n);

        var y = new Complex[n * n];
        var w = new Complex[n];
        var rw = new Complex[n];
        var shifted = new Complex[n * n];
        for (var j = n - 1; j >= 0; j--)
        {
            Array.Clear(w);
            for (var l = j + 1; l < n; l++)
            {
                var c = Complex.Conjugate(r[j * n + l]);
                if (c == Complex.Zero) continue;
                for (var i = 0; i < n; i++) w[i] += y[i * n + l] * c;
            }
            for (var i = 0; i < n; i++)
            {
                var acc = Complex.Zero;
                for (var k = i; k < n; k++) acc += r[i * n + k] * w[k];
                rw[i] = f[i * n + j] + acc;
            }
            var cj = Complex.Conjugate(r[j * n + j]);
            for (var i = 0; i < n * n; i++) shifted[i] = -cj * r[i];
            for (var i = 0; i < n; i++) shifted[i * n + i] += 1.0;
            UpperSolveChecked(shifted, n, rw, Complex.Zero, nameof(SolveDiscreteLyapunov));
            for (var i = 0; i < n; i++) y[i * n + j] = rw[i];
        }
        var x = LinalgKernels.CMul(LinalgKernels.CMul(u, y, n, n, n), uh, n, n, n);
        var xr = LinalgKernels.RealPart(x);
        LinalgKernels.Symmetrize(xr, n);
        return new NdArray(xr, n, n);
    }

    /// <summary>
    /// Stabilising solution of the continuous algebraic Riccati equation, scipy's
    /// <c>solve_continuous_are</c>:
    /// <c>AᵀXE + EᵀXA − (EᵀXB + S) R⁻¹ (BᵀXE + Sᵀ) + Q = 0</c>.
    /// </summary>
    /// <param name="a">n×n state matrix.</param>
    /// <param name="b">n×m input matrix.</param>
    /// <param name="q">n×n symmetric state weight.</param>
    /// <param name="r">m×m symmetric positive-definite input weight.</param>
    /// <param name="e">Optional non-singular n×n descriptor matrix.</param>
    /// <param name="s">Optional n×m cross term.</param>
    /// <remarks>
    /// <para>
    /// Matrix sign function of the Hamiltonian <c>H = [[A, −G], [−Q, −Aᵀ]]</c>, G = B R⁻¹ Bᵀ
    /// (Roberts 1971; Byers 1987), by scaled Newton iteration; X then solves the overdetermined
    /// <c>[W₁₂; W₂₂ + I] X = −[W₁₁ + I; W₂₁]</c> in the least-squares sense. Two Newton–Kleinman
    /// steps (each a Lyapunov solve) polish the result to the residual LAPACK's ordered-QZ method
    /// reaches. scipy uses the ordered-QZ method on the extended pencil; the stabilising solution is
    /// unique, so the answers agree to rounding.
    /// </para>
    /// <para>
    /// E is handled by reduction (<c>A ← E⁻¹A</c>, <c>B ← E⁻¹B</c>, solve for <c>Y = EᵀXE</c>); S by
    /// completing the square (<c>A ← A − B R⁻¹ Sᵀ</c>, <c>Q ← Q − S R⁻¹ Sᵀ</c>). Requires (A, B)
    /// stabilisable and no eigenvalue of H on the imaginary axis; otherwise this throws.
    /// </para>
    /// </remarks>
    public static NdArray SolveContinuousAre(NdArray a, NdArray b, NdArray q, NdArray r, NdArray? e = null, NdArray? s = null)
    {
        var (aa, g, qq, n) = PrepareRiccati(a, b, q, r, e, s, nameof(SolveContinuousAre), discrete: false);

        var m2 = 2 * n;
        var h = new double[m2 * m2];
        var A = aa.AsContiguous().AsSpan();
        var G = g.AsSpan();
        var Q = qq.AsSpan();
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                h[i * m2 + j] = A[i * n + j];
                h[i * m2 + n + j] = -G[i * n + j];
                h[(n + i) * m2 + j] = -Q[i * n + j];
                h[(n + i) * m2 + n + j] = -A[j * n + i];
            }
        var w = SignNewton(new NdArray(h, m2, m2), nameof(SolveContinuousAre)).AsSpan();

        // [W12; W22 + I] X = -[W11 + I; W21]
        var lhs = new double[m2 * n];
        var rhs = new double[m2 * n];
        for (var i = 0; i < m2; i++)
            for (var j = 0; j < n; j++)
            {
                lhs[i * n + j] = w[i * m2 + n + j] + (i - n == j ? 1.0 : 0.0);
                rhs[i * n + j] = -(w[i * m2 + j] + (i == j ? 1.0 : 0.0));
            }
        var x = Lstsq(new NdArray(lhs, m2, n), new NdArray(rhs, m2, n)).X.AsContiguous().Copy();
        LinalgKernels.Symmetrize(x.AsSpan(), n);

        // Newton–Kleinman refinement: (A - G X)^T X+ + X+ (A - G X) = -(Q + X G X).
        for (var step = 0; step < 2; step++)
        {
            var gx = LinalgKernels.Mul(g, x);
            var ac = LinalgKernels.Combine(1.0, aa, -1.0, gx);
            var rhsK = LinalgKernels.Combine(-1.0, qq, -1.0, LinalgKernels.Mul(x, gx));
            var before = CareResidual(aa, g, qq, x);
            NdArray xn;
            try { xn = SolveContinuousLyapunov(ac.T.Copy(), rhsK).AsContiguous().Copy(); }
            catch (LinalgException) { break; }
            LinalgKernels.Symmetrize(xn.AsSpan(), n);
            if (CareResidual(aa, g, qq, xn) < before) x = xn; else break;
        }

        if (e is not null)
        {
            // X = E^{-T} Y E^{-1}
            var einv = Inv(e);
            x = LinalgKernels.Mul(LinalgKernels.Mul(einv.T, x), einv).AsContiguous().Copy();
            LinalgKernels.Symmetrize(x.AsSpan(), n);
        }
        return x;
    }

    private static double CareResidual(NdArray a, NdArray g, NdArray q, NdArray x)
    {
        var res = LinalgKernels.Combine(1.0, LinalgKernels.Mul(a.T, x), 1.0, LinalgKernels.Mul(x, a));
        res = LinalgKernels.Combine(1.0, res, -1.0, LinalgKernels.Mul(LinalgKernels.Mul(x, g), x));
        res = LinalgKernels.Combine(1.0, res, 1.0, q);
        return LinalgKernels.OneNorm(res);
    }

    /// <summary>
    /// Stabilising solution of the discrete algebraic Riccati equation, scipy's
    /// <c>solve_discrete_are</c>:
    /// <c>AᵀXA − EᵀXE − (AᵀXB + S)(R + BᵀXB)⁻¹(BᵀXA + Sᵀ) + Q = 0</c>.
    /// </summary>
    /// <remarks>
    /// Structured doubling algorithm (Chu, Fan, Lin &amp; Wang 2004) on
    /// <c>X = Aᵀ X (I + G X)⁻¹ A + Q</c>, G = B R⁻¹ Bᵀ: <c>A ← A(I+GH)⁻¹A</c>,
    /// <c>G ← G + A(I+GH)⁻¹GAᵀ</c>, <c>H ← H + AᵀH(I+GH)⁻¹A</c>, H → X quadratically. E is absorbed as
    /// <c>A ← A E⁻¹</c>, <c>Q ← E⁻ᵀ Q E⁻¹</c> with the cross term mapped likewise; S by completing the
    /// square. scipy uses ordered QZ; the stabilising solution is unique, so they agree to rounding.
    /// Requires (A, B) stabilisable and no unit-circle eigenvalue of the symplectic pencil; otherwise
    /// the doubling diverges and this throws.
    /// </remarks>
    public static NdArray SolveDiscreteAre(NdArray a, NdArray b, NdArray q, NdArray r, NdArray? e = null, NdArray? s = null)
    {
        var (ak, gk, hk, n) = PrepareRiccati(a, b, q, r, e, s, nameof(SolveDiscreteAre), discrete: true);
        for (var iter = 0; iter < 100; iter++)
        {
            var w = LinalgKernels.Mul(gk, hk);
            LinalgKernels.AddDiagonal(w.AsSpan(), n, 1.0);
            var luW = LuFactor(w);
            if (HasZeroPivot(luW)) throw new LinalgSingularMatrixException("SolveDiscreteAre: I + G H became singular.");
            // W^{-1} A and W^{-1} G
            var wInvA = LuSolve(luW, ak);
            var wInvG = LuSolve(luW, gk);
            var aNext = LinalgKernels.Mul(ak, wInvA);
            var gNext = LinalgKernels.Combine(1.0, gk, 1.0, LinalgKernels.Mul(LinalgKernels.Mul(ak, wInvG), ak.T));
            // A^T H W^{-1} A
            var hNext = LinalgKernels.Combine(1.0, hk, 1.0, LinalgKernels.Mul(LinalgKernels.Mul(ak.T, hk), wInvA));
            var diff = LinalgKernels.OneNorm(LinalgKernels.Combine(1.0, hNext, -1.0, hk));
            var size = LinalgKernels.OneNorm(hNext);
            ak = aNext; gk = gNext; hk = hNext;
            if (!double.IsFinite(diff)) break;
            if (diff <= 1e-15 * Math.Max(size, 1e-300))
            {
                var x = hk.AsContiguous().Copy();
                LinalgKernels.Symmetrize(x.AsSpan(), n);
                return x;
            }
        }
        throw new LinalgException("SolveDiscreteAre: structured doubling did not converge; is (A, B) stabilisable?");
    }

    private static bool HasZeroPivot(LuFactorResult lu)
    {
        var n = lu.Lu.Shape[0];
        var s = lu.Lu.AsSpan();
        for (var i = 0; i < n; i++) if (s[i * n + i] == 0.0) return true;
        return false;
    }

    /// <summary>Reduces E and S away and forms G = B R⁻¹ Bᵀ; returns (A', G, Q', n).</summary>
    private static (NdArray A, NdArray G, NdArray Q, int N) PrepareRiccati(
        NdArray a, NdArray b, NdArray q, NdArray r, NdArray? e, NdArray? s, string caller, bool discrete)
    {
        LinalgKernels.RequireSquare(a, caller);
        LinalgKernels.RequireMatrix(b, caller);
        LinalgKernels.RequireSquare(q, caller);
        LinalgKernels.RequireSquare(r, caller);
        var n = a.Shape[0];
        var m = b.Shape[1];
        if (b.Shape[0] != n || q.Shape[0] != n || r.Shape[0] != m)
            throw new ArgumentException($"{caller}: incompatible shapes (a n×n, b n×m, q n×n, r m×m).");
        if (s is not null && (s.Rank != 2 || s.Shape[0] != n || s.Shape[1] != m))
            throw new ArgumentException($"{caller}: s must be n×m.");
        if (!LinAlg.IsSymmetric(q, 1e-12 * Math.Max(1.0, LinalgKernels.OneNorm(q.AsContiguous()))) ||
            !LinAlg.IsSymmetric(r, 1e-12 * Math.Max(1.0, LinalgKernels.OneNorm(r.AsContiguous()))))
            throw new ArgumentException($"{caller}: q and r must be symmetric.");

        var aa = a.AsContiguous().Copy();
        var bb = b.AsContiguous().Copy();
        var qq = q.AsContiguous().Copy();
        var ss = s?.AsContiguous().Copy();
        if (e is not null)
        {
            LinalgKernels.RequireSquare(e, caller);
            if (e.Shape[0] != n) throw new ArgumentException($"{caller}: e must be n×n.");
            if (!discrete)
            {
                // Continuous: A <- E^{-1} A, B <- E^{-1} B, solving for Y = E^T X E.
                aa = Solve(e, aa, LinalgMatrixStructure.General);
                bb = Solve(e, bb, LinalgMatrixStructure.General);
            }
            else
            {
                // Discrete: A <- A E^{-1}, Q <- E^{-T} Q E^{-1}, S <- E^{-T} S.
                var einv = Inv(e);
                aa = LinalgKernels.Mul(aa, einv);
                qq = LinalgKernels.Mul(LinalgKernels.Mul(einv.T, qq), einv);
                if (ss is not null) ss = LinalgKernels.Mul(einv.T, ss);
            }
        }

        var rChol = ChoFactor(r, lower: true);
        var rInvBt = ChoSolve(rChol, bb.T.Copy()); // m x n
        var g = LinalgKernels.Mul(bb, rInvBt);
        if (ss is not null)
        {
            var rInvSt = ChoSolve(rChol, ss.T.Copy()); // m x n
            aa = LinalgKernels.Combine(1.0, aa, -1.0, LinalgKernels.Mul(bb, rInvSt));
            qq = LinalgKernels.Combine(1.0, qq, -1.0, LinalgKernels.Mul(ss, rInvSt));
        }
        var gs = g.AsContiguous().Copy();
        LinalgKernels.Symmetrize(gs.AsSpan(), n);
        var qs = qq.AsContiguous().Copy();
        LinalgKernels.Symmetrize(qs.AsSpan(), n);
        return (aa.AsContiguous(), gs, qs, n);
    }
}
