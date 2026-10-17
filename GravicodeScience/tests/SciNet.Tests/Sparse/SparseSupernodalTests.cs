using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Sparse;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SparseAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// The multifrontal LU, AMD ordering and dense frontal kernels, each pinned against an
/// independent route: dense LU, the old left-looking Gilbert–Peierls factorisation, and the
/// scalar GEMM.
/// </summary>
public class SparseSupernodalTests
{
    private static double[] Rhs(int n, int seed)
    {
        var rng = new GraviRandom(seed);
        return Enumerable.Range(0, n).Select(_ => rng.Uniform(-1, 1)).ToArray();
    }

    private static double[] DenseSolve(SparseMatrixBase a, double[] b)
        => LinAlg.Solve(a.ToDense(), new NdArray(b, b.Length)).ToArray();

    /// <summary>A random sparse matrix whose diagonal is deliberately weak, so pivots must be delayed.</summary>
    private static CsrMatrix WeakDiagonal(int n, double density, int seed, double diagScale)
    {
        var d = RandomDense(n, n, density, seed).AsContiguous().ToArray();
        var rng = new GraviRandom(seed + 1);
        for (var i = 0; i < n; i++)
        {
            d[i * n + i] = diagScale * rng.Uniform(0.5, 1); // structurally present, numerically small
            // keep it non-singular: a guaranteed entry on a random permutation
            var j = (i * 7 + 3) % n;
            if (d[i * n + j] == 0) d[i * n + j] = rng.Uniform(0.5, 1);
        }
        return CsrMatrix.FromDense(new NdArray(d, n, n));
    }

    private static void AssertReconstructs(SparseMatrixBase a, SparseLuFactorization lu, double tol, double thresh)
    {
        var n = a.Rows;
        var pr = CsrMatrix.FromTriplets(n, n, lu.PermR, Enumerable.Range(0, n).ToArray(), Enumerable.Repeat(1.0, n).ToArray());
        var pc = CsrMatrix.FromTriplets(n, n, Enumerable.Range(0, n).ToArray(), lu.PermC, Enumerable.Repeat(1.0, n).ToArray());
        var rebuilt = pr.Transpose().Dot(lu.L.Dot(lu.U)).Dot(pc.Transpose());
        Close(a.ToDense(), rebuilt.ToDense(), tol, "A = Prᵀ L U Pcᵀ");
        foreach (var (i, j, v) in lu.L.EnumerateEntries())
        {
            Assert.True(i >= j, $"L has ({i},{j})");
            if (i == j) Assert.Equal(1.0, v);
            Assert.True(Math.Abs(v) <= 1.0 / thresh + 1e-12, $"multiplier {v} exceeds 1/u");
        }
        foreach (var (i, j, _) in lu.U.EnumerateEntries()) Assert.True(i <= j, $"U has ({i},{j})");
        Assert.Equal(lu.L.Nnz + lu.U.Nnz, lu.Nnz);
    }

    [Theory]
    [InlineData(1.0)]
    [InlineData(0.1)]
    public void Splu_DelayedPivots_ReconstructAndSolve(double thresh)
    {
        // 150 unknowns, ~8% dense, diagonal 1e-3: most diagonal pivots fail the threshold, so
        // columns are delayed up the tree and fronts span several Nb-column blocks.
        var a = WeakDiagonal(150, 0.08, 31, 1e-3);
        var lu = SparseLinalg.Splu(a, diagPivotThresh: thresh);
        Assert.True(lu.Supernodal!.Delayed > 0, "the test is meant to exercise delayed pivots");
        AssertReconstructs(a, lu, 1e-11, thresh);
        var b = Rhs(150, 4);
        Close(DenseSolve(a, b), lu.Solve(b), 1e-9, "solve");
        Close(LinAlg.Solve(a.ToDense().T.Copy(), new NdArray(b, 150)).ToArray(), lu.Solve(b, transpose: true), 1e-9, "solve T");
    }

    [Fact]
    public void Splu_MatchesGilbertPeierlsReference_OnEveryOrdering()
    {
        var a = ConvectionDiffusion(17, 30).Add(WeakDiagonal(289, 0.01, 5, 0.0)); // non-symmetric pattern
        var b = Rhs(289, 6);
        foreach (var o in Enum.GetValues<ColumnOrdering>())
        {
            var x = SparseLinalg.Splu(a, o).Solve(b);
            var r = SparseLinalg.SpluGilbertPeierls(a, o).Solve(b);
            Close(r, x, 1e-9, o.ToString());
            Assert.True(Norm(Residual(a, x, b)) <= 1e-12 * Norm(b) * 1e3, $"{o} residual");
        }
    }

    [Fact]
    public void Splu_ZeroDiagonal_UsesTransversal()
    {
        // A permuted Poisson matrix: every diagonal entry is zero, the transversal restores it.
        var p = Poisson2D(9, 8);
        var n = p.Rows;
        var shift = Enumerable.Range(0, n).Select(i => (i + 5) % n).ToArray();
        var a = p.GetRows(shift);
        Assert.All(a.Diagonal(), v => Assert.Equal(0.0, v));
        var lu = SparseLinalg.Splu(a);
        AssertReconstructs(a, lu, 1e-12, 1.0);
        var b = Rhs(n, 2);
        Close(DenseSolve(a, b), lu.Solve(b), 1e-11, "solve");
    }

    [Fact]
    public void Splu_SingularInsideTheTree_Throws()
    {
        // Two identical rows in a 2-D grid: numerically singular, structurally fine.
        var p = Poisson2D(6, 6).ToDense().AsContiguous().ToArray();
        for (var j = 0; j < 36; j++) p[20 * 36 + j] = p[3 * 36 + j];
        Assert.Throws<InvalidOperationException>(() => SparseLinalg.Splu(CsrMatrix.FromDense(new NdArray(p, 36, 36))));
    }

    [Fact]
    public void Splu_LargeGrid_SolvesToRoundoff_AndBeatsNaturalFill()
    {
        var a = ConvectionDiffusion(60, 25); // 3600 unknowns, fronts of a few hundred
        var b = Rhs(3600, 7);
        var lu = SparseLinalg.Splu(a);
        var x = lu.Solve(b);
        Assert.True(Norm(Residual(a, x, b)) <= 1e-13 * Norm(b) * 100, "residual");
        var xt = lu.Solve(b, transpose: true);
        var rt = a.Transpose().Dot(xt);
        for (var i = 0; i < rt.Length; i++) rt[i] -= b[i];
        Assert.True(Norm(rt) <= 1e-11 * Norm(b), "transpose residual");
        // AMD on a 60×60 grid: far below the natural (banded) fill of ~2·n·60.
        Assert.True(lu.Nnz < 2 * 3600 * 60 / 2, $"fill {lu.Nnz}");
    }

    [Fact]
    public void Amd_IsAPermutation_AndReducesFillOnAShuffledGrid()
    {
        var p = Poisson2D(20, 20);
        var perm = new GraviRandom(3).Permutation(400);
        var a = p.GetRows(perm).GetColumns(perm);
        foreach (var o in new[] { ColumnOrdering.MmdAtPlusA, ColumnOrdering.MmdAtA, ColumnOrdering.Colamd })
        {
            var q = SparseLinalg.ColumnPermutation(a, o);
            Assert.Equal(Enumerable.Range(0, 400), q.OrderBy(v => v));
        }
        var natural = SparseLinalg.SpluGilbertPeierls(a, ColumnOrdering.Natural).Nnz;
        var amd = SparseLinalg.SpluGilbertPeierls(a, ColumnOrdering.MmdAtPlusA).Nnz;
        // Measured: 7600 entries in L + U against 28286 for the natural order of the shuffled grid.
        Assert.True(amd < natural / 3, $"AMD fill {amd} vs natural {natural}");
        Assert.True(amd < 9_000, $"AMD fill {amd}");
    }

    [Fact]
    public void Amd_HandlesDenseRowsAndEmptyColumns()
    {
        // An arrow matrix (one dense row/column) plus an isolated node: dense nodes go last.
        var n = 300;
        var rows = new List<int>(); var cols = new List<int>(); var vals = new List<double>();
        for (var i = 0; i < n; i++)
        {
            rows.Add(i); cols.Add(i); vals.Add(4);
            if (i > 0 && i < n - 1) { rows.Add(0); cols.Add(i); vals.Add(1); rows.Add(i); cols.Add(0); vals.Add(1); }
            if (i > 1 && i < n - 1) { rows.Add(i); cols.Add(i - 1); vals.Add(-1); rows.Add(i - 1); cols.Add(i); vals.Add(-1); }
        }
        var a = CsrMatrix.FromTriplets(n, n, rows.ToArray(), cols.ToArray(), vals.ToArray());
        var q = SparseLinalg.ColumnPermutation(a, ColumnOrdering.MmdAtPlusA);
        Assert.Equal(Enumerable.Range(0, n), q.OrderBy(v => v));
        Assert.True(Array.IndexOf(q, 0) >= n - 2, "the dense node is ordered last");
        var b = Rhs(n, 1);
        var x = SparseLinalg.Splu(a).Solve(b);
        Assert.True(Norm(Residual(a, x, b)) <= 1e-12 * Norm(b), "arrow residual");
    }

    // ------------------------------------------------------------------ Cholesky

    /// <summary>An SPD matrix with a non-grid pattern: BᵀB + I for a random sparse B.</summary>
    private static CsrMatrix RandomSpd(int n, double density, int seed)
    {
        var b = CsrMatrix.FromDense(RandomDense(n, n, density, seed));
        return b.Transpose().ToCsr().MatMul(b).Add(SparseMatrices.Identity(n));
    }

    [Fact]
    public void Cholesky_ReconstructsPermutedMatrix_AndSolves()
    {
        foreach (var (a, name) in new[] { (Poisson2D(13, 11), "poisson"), (RandomSpd(120, 0.04, 8), "BᵀB + I") })
        {
            var n = a.Rows;
            foreach (var o in new[] { ColumnOrdering.Natural, ColumnOrdering.MmdAtPlusA })
            {
                var ch = SparseLinalg.Cholesky(a, o);
                var perm = ch.Perm;
                Assert.Equal(Enumerable.Range(0, n), perm.OrderBy(v => v));
                var l = ch.L;
                foreach (var (i, j, _) in l.EnumerateEntries()) Assert.True(i >= j, $"L has ({i},{j})");
                Close(a.GetRows(perm).GetColumns(perm).ToDense(), l.Dot(l.Transpose()).ToDense(), 1e-12, $"{name} {o}: P A Pᵀ = L Lᵀ");
                Assert.Equal(l.Nnz, ch.Nnz);
                var b = Rhs(n, 3);
                Close(DenseSolve(a, b), ch.Solve(b), 1e-11, $"{name} {o} solve");
                // log det against the dense LU (product of U's diagonal; SPD so it is positive).
                var lu = SparseLinalg.SpluGilbertPeierls(a);
                var logdet = lu.U.Diagonal().Sum(v => Math.Log(Math.Abs(v)));
                Close(logdet, ch.LogDeterminant(), 1e-10 * Math.Abs(logdet), $"{name} {o} log det");
            }
        }
    }

    [Fact]
    public void Cholesky_LargeGrid_MatchesLu()
    {
        var a = Poisson2D(70, 50); // 3500 unknowns, multi-block fronts and parallel updates
        var b = Rhs(a.Rows, 12);
        var x = SparseLinalg.Cholesky(a).Solve(b);
        Assert.True(Norm(Residual(a, x, b)) <= 1e-13 * Norm(b) * 100, "residual");
        Close(SparseLinalg.Splu(a).Solve(b), x, 1e-9, "against LU");
        var m = SparseLinalg.Cholesky(a).Solve(new NdArray(b.Concat(b).ToArray(), 2, a.Rows).T.Copy());
        Close(x, m.Column(1).ToArray(), 1e-14, "matrix right-hand side");
    }

    [Fact]
    public void Cholesky_RejectsIndefiniteAndNonSymmetric()
    {
        var p = Poisson2D(8, 8);
        var shifted = p.Subtract(SparseMatrices.Identity(64).Scale(1.0)); // λ_min ≈ 0.12 − 1 < 0
        var e = Assert.Throws<SparseNotPositiveDefiniteException>(() => SparseLinalg.Cholesky(shifted));
        Assert.True(e.Column >= 0);
        Assert.Throws<ArgumentException>(() => SparseLinalg.Cholesky(ConvectionDiffusion(6, 10)));
    }

    [Fact]
    public unsafe void GemmKernel_MatchesScalarReference()
    {
        var rng = new GraviRandom(9);
        foreach (var (m, n, k) in new[] { (1, 1, 1), (5, 13, 3), (37, 41, 32), (130, 70, 17), (64, 300, 48), (200, 200, 48) })
        {
            int lda = k + 3, ldb = n + 2, ldc = n + 5;
            var a = Enumerable.Range(0, m * lda).Select(_ => rng.Uniform(-1, 1)).ToArray();
            var b = Enumerable.Range(0, k * ldb).Select(_ => rng.Uniform(-1, 1)).ToArray();
            var c1 = Enumerable.Range(0, m * ldc).Select(_ => rng.Uniform(-1, 1)).ToArray();
            var c2 = (double[])c1.Clone();
            fixed (double* pa = a, pb = b, p1 = c1, p2 = c2)
            {
                SparseDenseKernels.GemmSub(m, n, k, pa, lda, pb, ldb, p1, ldc);
                SparseDenseKernels.GemmReference(m, n, k, pa, lda, pb, ldb, p2, ldc);
            }
            Close(c2, c1, 1e-12, $"gemm {m}x{n}x{k}");
        }
    }
}
