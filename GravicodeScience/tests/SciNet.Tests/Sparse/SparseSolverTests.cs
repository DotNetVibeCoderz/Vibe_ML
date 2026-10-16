using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Sparse;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SparseAssert;

namespace Gravicode.Science.Tests.SciNet;

public class SparseSolverTests
{
    private static double[] Rhs(int n, int seed)
    {
        var rng = new GraviRandom(seed);
        return Enumerable.Range(0, n).Select(_ => rng.Uniform(-1, 1)).ToArray();
    }

    private static double[] DenseSolve(SparseMatrixBase a, double[] b)
        => LinAlg.Solve(a.ToDense(), new NdArray(b, b.Length)).ToArray();

    private delegate IterativeSolveResult Solver(ILinearOperator a, double[] b, ILinearOperator? m);

    private static readonly (string Name, Solver Solve)[] General =
    [
        ("Gmres", (a, b, m) => SparseLinalg.Gmres(a, b, rtol: 1e-10, m: m)),
        ("Lgmres", (a, b, m) => SparseLinalg.Lgmres(a, b, rtol: 1e-10, m: m)),
        ("Bicgstab", (a, b, m) => SparseLinalg.Bicgstab(a, b, rtol: 1e-10, m: m)),
        ("Bicg", (a, b, m) => SparseLinalg.Bicg(a, b, rtol: 1e-10, m: m)),
        ("Cgs", (a, b, m) => SparseLinalg.Cgs(a, b, rtol: 1e-10, m: m)),
        ("Qmr", (a, b, m) => SparseLinalg.Qmr(a, b, rtol: 1e-10, m1: m)),
        ("Tfqmr", (a, b, m) => SparseLinalg.Tfqmr(a, b, rtol: 1e-10, m: m)),
    ];

    [Fact]
    public void Cg_And_Minres_SolvePoisson2D()
    {
        var a = Poisson2D(12, 10);
        var b = Rhs(a.Rows, 1);
        var exact = DenseSolve(a, b);
        var iterates = 0;
        var cg = SparseLinalg.Cg(a, b, rtol: 1e-10, callback: _ => iterates++);
        Assert.True(cg.Converged);
        Assert.Equal(cg.Iterations, iterates);
        Assert.True(cg.ResidualNorm <= 1e-10 * Norm(b) * 1.01, $"CG residual {cg.ResidualNorm}");
        Close(exact, cg.X, 1e-8, "cg");
        // CG on the 120-unknown Poisson problem needs well under n iterations.
        Assert.InRange(cg.Iterations, 10, 120);

        var mr = SparseLinalg.Minres(a, b, rtol: 1e-12);
        Assert.True(mr.Converged);
        Close(exact, mr.X, 1e-7, "minres");

        // Jacobi preconditioning and ILU(0) both cut iterations for CG.
        var ilu = SparseLinalg.Ilu0(a);
        var pcg = SparseLinalg.Cg(a, b, rtol: 1e-10, m: ilu);
        Assert.True(pcg.Converged);
        Assert.True(pcg.Iterations < cg.Iterations, $"ILU0-CG {pcg.Iterations} vs CG {cg.Iterations}");
        Close(exact, pcg.X, 1e-8, "pcg");
        var jcg = SparseLinalg.Cg(a, b, rtol: 1e-10, m: SparseLinalg.JacobiPreconditioner(a));
        Assert.True(jcg.Converged);
        Close(exact, jcg.X, 1e-8, "jacobi cg");
    }

    [Fact]
    public void Minres_HandlesSymmetricIndefinite()
    {
        // Poisson shifted into the middle of its spectrum: symmetric but indefinite, where CG is not valid.
        var p = Poisson2D(8, 8);
        var a = p.Subtract(SparseMatrices.Identity(64).Scale(3.1));
        var b = Rhs(64, 2);
        var r = SparseLinalg.Minres(a, b, rtol: 1e-12, maxiter: 2000);
        Assert.True(r.Converged);
        Close(DenseSolve(a, b), r.X, 1e-6, "minres indefinite");
        var shifted = SparseLinalg.Minres(p, b, rtol: 1e-12, shift: 3.1, maxiter: 2000);
        Close(r.X, shifted.X, 1e-6, "shift parameter");
    }

    [Fact]
    public void NonSymmetricSolvers_ConvergeOnConvectionDiffusion()
    {
        var a = ConvectionDiffusion(10, peclet: 20);
        var b = Rhs(a.Rows, 3);
        var exact = DenseSolve(a, b);
        foreach (var (name, solve) in General)
        {
            var r = solve(a, b, null);
            Assert.True(r.Converged, $"{name} info={r.Info} after {r.Iterations}");
            Assert.True(r.ResidualNorm <= 2e-10 * Norm(b), $"{name} residual {r.ResidualNorm}");
            Close(exact, r.X, 1e-7, name);
        }
    }

    [Fact]
    public void NonSymmetricSolvers_WithIlutPreconditioner()
    {
        var a = ConvectionDiffusion(14, peclet: 50);
        var b = Rhs(a.Rows, 4);
        var exact = DenseSolve(a, b);
        var ilu = SparseLinalg.Spilu(a, dropTol: 1e-3, fillFactor: 4);
        foreach (var (name, solve) in General)
        {
            var plain = solve(a, b, null);
            var pre = solve(a, b, ilu);
            Assert.True(pre.Converged, $"{name} preconditioned info={pre.Info}");
            Close(exact, pre.X, 1e-7, name + " + ILUT");
            Assert.True(pre.Iterations <= plain.Iterations, $"{name}: {pre.Iterations} preconditioned vs {plain.Iterations} plain");
        }
    }

    [Fact]
    public void Gmres_RestartStillConverges_AndX0IsHonoured()
    {
        var a = ConvectionDiffusion(12, 10);
        var b = Rhs(a.Rows, 5);
        var exact = DenseSolve(a, b);
        var r = SparseLinalg.Gmres(a, b, rtol: 1e-10, restart: 5);
        Assert.True(r.Converged);
        Close(exact, r.X, 1e-7);
        var warm = SparseLinalg.Gmres(a, b, x0: exact, rtol: 1e-10);
        Assert.Equal(0, warm.Iterations);
        var zero = SparseLinalg.Cg(Poisson2D(3, 3), new double[9]);
        Assert.All(zero.X, v => Assert.Equal(0.0, v));
    }

    [Fact]
    public void IterationLimit_IsReportedNotHidden()
    {
        var a = Poisson2D(20, 20);
        var r = SparseLinalg.Cg(a, Rhs(400, 6), rtol: 1e-12, maxiter: 3);
        Assert.False(r.Converged);
        Assert.Equal(3, r.Info);
    }

    [Fact]
    public void Lsqr_And_Lsmr_MatchDenseLeastSquares()
    {
        var a = CsrMatrix.FromDense(RandomDense(40, 12, 0.3, 61));
        var a2 = a.Add(SparseMatrices.Eye(40, 12)); // full column rank
        var b = Rhs(40, 7);
        var dense = a2.ToDense();
        var normal = LinAlg.Dot(dense.T, dense);
        var rhs = LinAlg.Dot(dense.T, new NdArray(b, 40));
        var exact = LinAlg.Solve(normal, rhs).ToArray();
        var lsqr = SparseLinalg.Lsqr(a2, b, atol: 1e-12, btol: 1e-12);
        Assert.InRange(lsqr.IStop, 1, 2);
        Close(exact, lsqr.X, 1e-8, "lsqr");
        var lsmr = SparseLinalg.Lsmr(a2, b, atol: 1e-12, btol: 1e-12, maxiter: 200);
        Close(exact, lsmr.X, 1e-8, "lsmr");
        var r = Residual(a2, lsqr.X, b);
        Close(Norm(r), lsqr.R1Norm, 1e-8, "r1norm");

        // Damped: (AᵀA + λ²I) x = Aᵀb.
        const double damp = 0.7;
        var damped = normal.ToArray();
        for (var i = 0; i < 12; i++) damped[i * 12 + i] += damp * damp;
        var exactDamped = LinAlg.Solve(new NdArray(damped, 12, 12), rhs).ToArray();
        Close(exactDamped, SparseLinalg.Lsqr(a2, b, damp: damp, atol: 1e-12, btol: 1e-12).X, 1e-8, "lsqr damped");
        Close(exactDamped, SparseLinalg.Lsmr(a2, b, damp: damp, atol: 1e-12, btol: 1e-12, maxiter: 200).X, 1e-8, "lsmr damped");
    }

    // ------------------------------------------------------------------ direct

    [Fact]
    public void Splu_NeedsPivoting_AndReconstructsA()
    {
        // Zero diagonal everywhere: without row pivoting the first step divides by zero.
        var d = new NdArray([
            0, 2, 0, 1, 0,
            3, 0, 0, 0, 1,
            0, 1, 0, 4, 0,
            1, 0, 5, 0, 0,
            0, 0, 1, 0, 2.0], 5, 5);
        var a = CsrMatrix.FromDense(d);
        foreach (var ordering in Enum.GetValues<ColumnOrdering>())
        {
            var lu = SparseLinalg.Splu(a, ordering);
            // A = Prᵀ L U Pcᵀ  (SciPy's reconstruction recipe)
            var n = 5;
            var pr = CsrMatrix.FromTriplets(n, n, lu.PermR, Enumerable.Range(0, n).ToArray(), Enumerable.Repeat(1.0, n).ToArray());
            var pc = CsrMatrix.FromTriplets(n, n, Enumerable.Range(0, n).ToArray(), lu.PermC, Enumerable.Repeat(1.0, n).ToArray());
            var rebuilt = pr.Transpose().Dot(lu.L.Dot(lu.U)).Dot(pc.Transpose());
            Close(d, rebuilt.ToDense(), 1e-14, ordering.ToString());
            AssertTriangular(lu.L, lower: true, unit: true);
            AssertTriangular(lu.U, lower: false, unit: false);
            var b = Rhs(5, 8);
            Close(DenseSolve(a, b), lu.Solve(b), 1e-12, ordering + " solve");
            Close(LinAlg.Solve(d.T.Copy(), new NdArray(b, 5)).ToArray(), lu.Solve(b, transpose: true), 1e-12, ordering + " solve T");
        }
    }

    private static void AssertTriangular(CscMatrix m, bool lower, bool unit)
    {
        foreach (var (i, j, v) in m.EnumerateEntries())
        {
            if (lower) Assert.True(i >= j, $"L has ({i},{j})");
            else Assert.True(i <= j, $"U has ({i},{j})");
            if (unit && i == j) Assert.Equal(1.0, v);
        }
        // Every L multiplier is bounded by 1 under partial pivoting.
        if (lower) foreach (var v in m.Data) Assert.True(Math.Abs(v) <= 1.0 + 1e-15);
    }

    [Fact]
    public void Splu_FillReducingOrdering_BeatsNaturalOnAShuffledGrid()
    {
        var p = Poisson2D(15, 15);
        var rng = new GraviRandom(17);
        var perm = rng.Permutation(225);
        var shuffled = p.GetRows(perm).GetColumns(perm);
        var natural = SparseLinalg.Splu(shuffled, ColumnOrdering.Natural);
        var amd = SparseLinalg.Splu(shuffled, ColumnOrdering.MmdAtPlusA);
        var ata = SparseLinalg.Splu(shuffled, ColumnOrdering.MmdAtA);
        var colamd = SparseLinalg.Splu(shuffled);
        Assert.Equal(amd.Nnz, colamd.Nnz); // symmetric pattern: the default orders A + Aᵀ
        Assert.True(amd.Nnz < natural.Nnz / 2, $"MMD fill {amd.Nnz} vs natural {natural.Nnz}");
        Assert.True(ata.Nnz < natural.Nnz, $"AᵀA fill {ata.Nnz} vs natural {natural.Nnz}");
        var b = Rhs(225, 9);
        var exact = DenseSolve(shuffled, b);
        Close(exact, amd.Solve(b), 1e-10, "mmd");
        Close(exact, ata.Solve(b), 1e-10, "ata");
        Close(exact, SparseLinalg.Spsolve(shuffled, b), 1e-10, "spsolve");
        Close(exact, SparseLinalg.Factorized(shuffled)(b), 1e-10, "factorized");
    }

    [Fact]
    public void Splu_LargeNonSymmetric_SolvesToRoundoff()
    {
        var a = ConvectionDiffusion(30, 40); // 900 unknowns
        var b = Rhs(900, 10);
        var lu = SparseLinalg.Splu(a);
        var x = lu.Solve(b);
        Assert.True(Norm(Residual(a, x, b)) <= 1e-11 * Norm(b), "residual");
        var m = lu.Solve(new NdArray(b.Concat(b.Select(v => 2 * v)).ToArray(), 2, 900).T.Copy());
        Close(x, m.Column(0).ToArray(), 1e-12);
        Close(x.Select(v => 2 * v).ToArray(), m.Column(1).ToArray(), 1e-12);
    }

    [Fact]
    public void Splu_SingularMatrix_Throws()
    {
        var s = CsrMatrix.FromDense(new NdArray([1.0, 2, 2, 4], 2, 2));
        Assert.Throws<InvalidOperationException>(() => SparseLinalg.Splu(s));
        var structural = CsrMatrix.FromDense(new NdArray([1.0, 0, 1, 0], 2, 2));
        Assert.Throws<InvalidOperationException>(() => SparseLinalg.Splu(structural));
    }

    [Fact]
    public void Inv_And_SparseRhs()
    {
        var a = ConvectionDiffusion(4, 3);
        var inv = SparseLinalg.Inv(a);
        Close(NdArray.Eye(16), a.Dot(inv).ToDense(), 1e-13, "A A⁻¹");
        var x = SparseLinalg.Spsolve(a, SparseMatrices.Eye(16, 3));
        Close(inv.ToCsr().Slice(0, 16, 0, 3).ToDense(), x.ToDense(), 1e-14);
    }

    [Fact]
    public void Ilu0_IsExactForTridiagonal_AndSpiluRespectsPattern()
    {
        var t = Laplacian1D(20);
        var ilu = SparseLinalg.Ilu0(t);
        Close(t.ToDense(), ilu.L.Dot(ilu.U).ToDense(), 1e-13, "ILU(0) of tridiagonal is exact");
        var b = Rhs(20, 11);
        Close(DenseSolve(t, b), ilu.Solve(b), 1e-12);
        Close(LinAlg.Solve(t.ToDense().T.Copy(), new NdArray(b, 20)).ToArray(), ilu.RDot(b), 1e-12, "transpose");
        // ILU(0) on a 2-D grid stays within A's pattern.
        var p = Poisson2D(6, 6);
        var ilu2 = SparseLinalg.Ilu0(p);
        foreach (var (i, j, _) in ilu2.U.EnumerateEntries()) Assert.NotEqual(0.0, p[i, j]);
        // Spilu with no dropping and unlimited fill is a full LU.
        var full = SparseLinalg.Spilu(p, dropTol: 0, fillFactor: 1000);
        Close(p.ToDense(), full.L.Dot(full.U).ToDense(), 1e-12, "ILUT(0, ∞) = LU");
        var dropped = SparseLinalg.Spilu(p, dropTol: 0.1, fillFactor: 1);
        Assert.True(dropped.Nnz < full.Nnz);
    }

    [Fact]
    public void SpsolveTriangular_MatchesDense()
    {
        var d = RandomDense(8, 8, 0.6, 71).ToArray();
        for (var i = 0; i < 8; i++) d[i * 9] = 2 + i;
        var full = CsrMatrix.FromDense(new NdArray(d, 8, 8));
        var lower = full.Tril();
        var upper = full.Triu();
        var b = Rhs(8, 12);
        Close(DenseSolve(lower, b), SparseLinalg.SpsolveTriangular(lower, b), 1e-12, "lower");
        Close(DenseSolve(upper, b), SparseLinalg.SpsolveTriangular(full, b, lower: false), 1e-12, "upper ignores the other triangle");
        var unit = lower.SetDiag(1.0);
        Close(DenseSolve(unit, b), SparseLinalg.SpsolveTriangular(lower, b, unitDiagonal: true), 1e-12, "unit");
        Assert.Throws<InvalidOperationException>(() => SparseLinalg.SpsolveTriangular(lower.SetDiag(0.0), b));
    }
}
