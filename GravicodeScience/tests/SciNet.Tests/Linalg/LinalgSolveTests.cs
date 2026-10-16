using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.LinalgAssert;

namespace Gravicode.Science.Tests.SciNet;

public class LinalgSolveTests
{
    [Fact]
    public void Solve_General_MatchesScipyExample()
    {
        // scipy.linalg.solve docstring example.
        var a = M(3, 3, 3, 2, 0, 1, -1, 0, 0, 5, 1);
        var x = Linalg.Solve(a, V(2, 4, -1));
        Close([2, -2, 9], x.ToArray(), 1e-14, "x");
    }

    [Theory]
    [InlineData(LinalgMatrixStructure.General)]
    [InlineData(LinalgMatrixStructure.Symmetric)]
    [InlineData(LinalgMatrixStructure.PositiveDefinite)]
    [InlineData(null)]
    public void Solve_StructuresAgreeOnSpdMatrix(LinalgMatrixStructure? s)
    {
        var r = Random(6, 6, 3);
        var a = Add(Mul(r.T, r), Mul(NdArray.Eye(6), NdArray.Eye(6)));
        var b = Random(6, 2, 4);
        var x = Linalg.Solve(a, b, s);
        Close(b, Mul(a, x), 1e-12, $"A x = b for {s}");
    }

    [Fact]
    public void Solve_SymmetricIndefinite_UsesLdlAndOnlyUpperTriangle()
    {
        var a = M(4, 4, 0, 1, 2, 3, 1, 0, 4, 5, 2, 4, 0, 6, 3, 5, 6, 0);
        var b = V(1, 2, 3, 4);
        var expected = Linalg.Solve(a, b, LinalgMatrixStructure.General);
        var garbage = a.Copy();
        for (var i = 0; i < 4; i++) for (var j = 0; j < i; j++) garbage[i, j] = 99; // lower ignored (lower=false)
        Close(expected, Linalg.Solve(garbage, b, LinalgMatrixStructure.Symmetric), 1e-12, "sym");
    }

    [Fact]
    public void Solve_TriangularTridiagonalDiagonal_Transposed()
    {
        var a = M(5, 5,
            4, 1, 0, 0, 0,
            2, 5, 1, 0, 0,
            0, 1, 6, 2, 0,
            0, 0, 3, 7, 1,
            0, 0, 0, 1, 3);
        var b = V(1, 2, 3, 4, 5);
        var dense = Linalg.Solve(a, b, LinalgMatrixStructure.General);
        Close(dense, Linalg.Solve(a, b, LinalgMatrixStructure.Tridiagonal), 1e-13, "tridiagonal");
        Close(dense, Linalg.Solve(a, b), 1e-13, "auto-detect");
        var xt = Linalg.Solve(a, b, transposed: true);
        Close(b, Mul(a.T, xt), 1e-13, "Aᵀ x = b");

        var u = Linalg.Tril(a).T.Copy();
        Close(Linalg.Solve(u, b, LinalgMatrixStructure.General), Linalg.Solve(u, b), 1e-13, "upper");
        Assert.Throws<LinalgSingularMatrixException>(() => Linalg.Solve(M(2, 2, 1, 0, 0, 0), V(1, 1)));
    }

    [Fact]
    public void LuFactor_LuSolve_AndPermutedFactors()
    {
        var a = A4;
        var lu = Linalg.LuFactor(a);
        var b = Random(4, 3, 7);
        Close(b, Mul(a, Linalg.LuSolve(lu, b)), 1e-12, "LuSolve");
        Close(b, Mul(a.T, Linalg.LuSolve(lu, b, trans: true)), 1e-12, "LuSolve trans");
        var (p, l, u) = Linalg.Lu(a);
        Close(a, Mul(p, Mul(l, u)), 1e-13, "A = P L U");
        Close(1654.0, Linalg.Det(a), 1e-13, "det (scipy 1.17.1: 1654.0)");
    }

    [Fact]
    public void Cholesky_ChoSolve_Ldl()
    {
        var a = M(3, 3, 4, 1, 2, 1, 5, 3, 2, 3, 6);
        var u = Linalg.Cholesky(a);
        Close(a, Mul(u.T, u), 1e-13, "UᵀU");
        Assert.Equal(0.0, u[1, 0]);
        var l = Linalg.Cholesky(a, lower: true);
        Close(a, Mul(l, l.T), 1e-13, "LLᵀ");
        var b = V(1, -2, 0.5);
        Close(b, Mul(a, Linalg.ChoSolve(Linalg.ChoFactor(a), b)), 1e-13, "ChoSolve");
        Assert.Throws<LinalgException>(() => Linalg.Cholesky(M(2, 2, 1, 2, 2, 1)));

        // A matrix forcing a 2x2 Bunch–Kaufman pivot (zero diagonal).
        var ind = M(4, 4, 0, 1, 2, 3, 1, 0, 4, 5, 2, 4, 0, 6, 3, 5, 6, 0);
        var f = Linalg.Ldl(ind);
        Close(ind, Mul(Mul(f.Lu, f.D), f.Lu.T), 1e-12, "Lu D Luᵀ");
        var perm = f.Lu.Take(f.Perm); // rows in perm order must be lower triangular
        for (var i = 0; i < 4; i++) for (var j = i + 1; j < 4; j++) Assert.Equal(0.0, perm[i, j]);
        var hasBlock = false;
        for (var i = 0; i < 3; i++) if (f.D[i + 1, i] != 0) hasBlock = true;
        Assert.True(hasBlock, "expected a 2x2 pivot block");
    }

    [Fact]
    public void SolveTriangular_AllVariants()
    {
        var t = M(3, 3, 2, 1, -1, 0, 3, 4, 0, 0, 5);
        var b = V(1, 2, 3);
        Close(b, Mul(t, Linalg.SolveTriangular(t, b)), 1e-14, "upper");
        Close(b, Mul(t.T, Linalg.SolveTriangular(t, b, trans: 1)), 1e-14, "upper T");
        var l = t.T.Copy();
        Close(b, Mul(l, Linalg.SolveTriangular(l, b, lower: true)), 1e-14, "lower");
        Close(b, Mul(l.T, Linalg.SolveTriangular(l, b, trans: 2, lower: true)), 1e-14, "lower T");
        var unit = M(3, 3, 1, 1, -1, 0, 1, 4, 0, 0, 1);
        Close(b, Mul(unit, Linalg.SolveTriangular(t, b, unitDiagonal: true)), 1e-14, "unit diagonal ignores stored diagonal");
    }

    [Fact]
    public void SolveBanded_MatchesDense_IncludingPivoting()
    {
        // l = 2, u = 1, with small diagonal entries so pivoting creates fill-in.
        const int n = 9;
        var rng = new GraviRandom(11);
        var a = NdArray.Zeros(n, n);
        for (var i = 0; i < n; i++)
            for (var j = Math.Max(0, i - 2); j <= Math.Min(n - 1, i + 1); j++)
                a[i, j] = i == j ? 0.01 * rng.NextDouble() : rng.NextDouble() * 2 - 1;
        var ab = Linalg.DenseToBand(a, 2, 1);
        var b = Random(n, 2, 12);
        Close(Linalg.Solve(a, b, LinalgMatrixStructure.General), Linalg.SolveBanded(2, 1, ab, b), 1e-10, "banded");
    }

    [Fact]
    public void CholeskyBanded_AndSolveHBanded_MatchScipyAndDense()
    {
        // Upper storage of the SPD pentadiagonal from the scipy docs (solveh_banded example).
        var ab = M(3, 6,
            0, 0, -1, -1, -1, -1,
            0, 2, 2, 2, 2, 2,
            7, 6, 6, 6, 6, 7);
        var b = V(1, 1, 1, 1, 1, 1);
        var x = Linalg.SolveHBanded(ab, b);
        // scipy 1.17.1 solveh_banded(ab, b)
        Close([0.13240418118466898, 0.1010452961672474, 0.1289198606271777, 0.1289198606271777, 0.10104529616724736, 0.13240418118466898], x.ToArray(), 1e-13, "x");

        var dense = NdArray.Zeros(6, 6);
        for (var j = 0; j < 6; j++)
            for (var r = 0; r < 3; r++)
            {
                var i = j - 2 + r;
                if (i < 0) continue;
                dense[i, j] = ab[r, j];
                dense[j, i] = ab[r, j];
            }
        var lowerAb = NdArray.Zeros(3, 6);
        for (var j = 0; j < 6; j++)
            for (var r = 0; r < 3; r++) if (j + r < 6) lowerAb[r, j] = dense[j + r, j];
        Close(x, Linalg.SolveHBanded(lowerAb, b, lower: true), 1e-13, "lower storage");

        var cu = Linalg.CholeskyBanded(ab);
        var u = NdArray.Zeros(6, 6);
        for (var j = 0; j < 6; j++) for (var r = 0; r < 3; r++) { var i = j - 2 + r; if (i >= 0) u[i, j] = cu[r, j]; }
        Close(dense, Mul(u.T, u), 1e-13, "UᵀU");
    }

    [Fact]
    public void SolveToeplitz_And_SolveCirculant_MatchDense()
    {
        double[] c = [1, 3, 6, 10];
        double[] r = [1, -1, -2, -3];
        var b = M(4, 2, 1, 0, 2, 1, 2, 0, 5, 1);
        var t = Linalg.Toeplitz(c, r);
        Close(Linalg.Solve(t, b, LinalgMatrixStructure.General), Linalg.SolveToeplitz(c, r, b), 1e-12, "toeplitz");
        // scipy docstring: solve_toeplitz((c, r), [1, 2, 2, 5]) = [1.66666667, -1, -2.66666667, 2.33333333]
        Close([5.0 / 3, -1, -8.0 / 3, 7.0 / 3], Linalg.SolveToeplitz(c, r, V(1, 2, 2, 5)).ToArray(), 1e-12, "doc example");
        double[] sym = [4, 1, 0.5, 0.2, 0.1];
        var bs = V(1, 2, 3, 4, 5);
        Close(Linalg.Solve(Linalg.Toeplitz(sym), bs, LinalgMatrixStructure.General), Linalg.SolveToeplitz(sym, null, bs), 1e-13, "symmetric");

        double[] cc = [2, -1, 0, 0.5, 1, 3, 0.25];
        var bc = Random(7, 3, 21);
        Close(Linalg.Solve(Linalg.Circulant(cc), bc, LinalgMatrixStructure.General), Linalg.SolveCirculant(cc, bc), 1e-12, "circulant");
        Assert.Throws<LinalgSingularMatrixException>(() => Linalg.SolveCirculant([1, 1, 1, 1], V(1, 2, 3, 4)));
        // Singular with lstsq: c = [1,1,1,1] has eigenvalues (4, 0, 0, 0); the min-norm solution of the
        // consistent system C x = (4,4,4,4) is (1,1,1,1).
        Close([1, 1, 1, 1], Linalg.SolveCirculant([1, 1, 1, 1], V(4, 4, 4, 4), singularLstsq: true).ToArray(), 1e-13, "lstsq");
    }

    [Fact]
    public void Lstsq_Pinv_Pinvh_MatchScipy()
    {
        var m = M(4, 2, 1, 2, 3, 4, 5, 6, 7, 9);
        var r = Linalg.Lstsq(m, V(1, 2, 2, 5));
        // scipy 1.17.1
        Close([-0.7627118644067775, 1.0847457627118628], r.X.ToArray(), 1e-12, "x");
        Close([0.9830508474576293], r.Residues, 1e-12, "residues");
        Assert.Equal(2, r.Rank);
        Close([-1.3050847457627106, -0.2881355932203378, 0.7288135593220314, -0.06779661016949076,
               1.0338983050847448, 0.2542372881355923, -0.5254237288135573, 0.1186440677966096],
            Linalg.Pinv(m).ToArray(), 1e-12, "pinv");

        // Rank-deficient: minimum-norm solution, no residues reported.
        var rd = M(3, 3, 1, 2, 3, 2, 4, 6, 1, 1, 1);
        var lr = Linalg.Lstsq(rd, V(1, 2, 3));
        Assert.Equal(2, lr.Rank);
        Assert.Empty(lr.Residues);
        Close(Mul(Linalg.Pinv(rd), V(1, 2, 3)), lr.X, 1e-12, "min-norm");

        var s = M(3, 3, 2, 1, 0, 1, 2, 0, 0, 0, 0);
        var ph = Linalg.Pinvh(s);
        Close(Linalg.Pinv(s), ph, 1e-13, "pinvh");
        Close(s, Mul(Mul(s, ph), s), 1e-13, "A A+ A = A");

        Identity(Mul(A4, Linalg.Inv(A4)), 1e-13, "inv");
        Assert.Throws<LinalgSingularMatrixException>(() => Linalg.Inv(rd));
    }
}
