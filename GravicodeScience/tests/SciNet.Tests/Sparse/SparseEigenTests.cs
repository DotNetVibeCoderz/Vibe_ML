using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Gravicode.Science.SciNet.Sparse;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SparseAssert;

namespace Gravicode.Science.Tests.SciNet;

public class SparseEigenTests
{
    private const int Nx = 20, Ny = 15;

    /// <summary>Closed-form spectrum of the 5-point Laplacian: 4sin²(iπ/2(nx+1)) + 4sin²(jπ/2(ny+1)).</summary>
    private static double[] PoissonSpectrum()
    {
        var list = new List<double>();
        for (var i = 1; i <= Nx; i++)
            for (var j = 1; j <= Ny; j++)
                list.Add(4 * Math.Pow(Math.Sin(i * Math.PI / (2 * (Nx + 1))), 2) + 4 * Math.Pow(Math.Sin(j * Math.PI / (2 * (Ny + 1))), 2));
        list.Sort();
        return list.ToArray();
    }

    private static void AssertEigenpairs(ILinearOperator a, double[] values, NdArray vectors, double tol)
    {
        var n = a.Rows;
        var k = values.Length;
        var v = vectors.AsContiguous().ToArray();
        for (var c = 0; c < k; c++)
        {
            var x = new double[n];
            for (var i = 0; i < n; i++) x[i] = v[i * k + c];
            Close(1.0, Norm(x), 1e-12, "unit norm");
            var ax = a.Dot(x);
            var r = 0.0;
            for (var i = 0; i < n; i++) r += Math.Pow(ax[i] - values[c] * x[i], 2);
            Assert.True(Math.Sqrt(r) <= tol, $"‖Av − λv‖ = {Math.Sqrt(r)} for λ = {values[c]}");
        }
    }

    [Fact]
    public void Eigsh_LargestOfPoisson_MatchClosedForm()
    {
        var a = Poisson2D(Nx, Ny);
        var exact = PoissonSpectrum();
        var r = SparseLinalg.Eigsh(a, k: 5);
        Close(exact[^5..], r.Values, 1e-10, "LM");
        AssertEigenpairs(a, r.Values, r.Vectors!, 1e-8);
        var la = SparseLinalg.Eigsh(a, k: 3, which: EigshWhich.LA, returnEigenvectors: false);
        Close(exact[^3..], la.Values, 1e-10, "LA");
        Assert.Null(la.Vectors);
    }

    [Fact]
    public void Eigsh_ShiftInvert_FindsSmallestAndInterior()
    {
        var a = Poisson2D(Nx, Ny);
        var exact = PoissonSpectrum();
        var smallest = SparseLinalg.Eigsh(a, k: 4, sigma: 0);
        Close(exact[..4], smallest.Values, 1e-10, "sigma=0");
        AssertEigenpairs(a, smallest.Values, smallest.Vectors!, 1e-9);
        // Interior: the 4 eigenvalues nearest 3.3.
        var interior = SparseLinalg.Eigsh(a, k: 4, sigma: 3.3);
        var nearest = exact.OrderBy(l => Math.Abs(l - 3.3)).Take(4).OrderBy(l => l).ToArray();
        Close(nearest, interior.Values, 1e-10, "sigma=3.3");
        var sa = SparseLinalg.Eigsh(a, k: 3, which: EigshWhich.SA, ncv: 40, maxiter: 5000);
        Close(exact[..3], sa.Values, 1e-8, "SA without shift (slow but correct)");
    }

    [Fact]
    public void Eigsh_BothEnds()
    {
        var a = Poisson2D(Nx, Ny);
        var exact = PoissonSpectrum();
        var be = SparseLinalg.Eigsh(a, k: 4, which: EigshWhich.BE, ncv: 40, maxiter: 5000);
        Close([exact[0], exact[1], exact[^2], exact[^1]], be.Values, 1e-8, "BE");
    }

    [Fact]
    public void Eigsh_RandomSymmetric_MatchesDenseEigh()
    {
        var b = SparseMatrices.Random(150, 150, 0.04, seed: 81);
        var a = b.Add(b.ToCsc().Transpose()); // symmetric
        var dense = Linalg.Eigh(a.ToDense()).Values; // ascending
        var lm = SparseLinalg.Eigsh(a, k: 6);
        var expected = dense.OrderByDescending(Math.Abs).Take(6).OrderBy(x => x).ToArray();
        Close(expected, lm.Values, 1e-9, "random LM");
        AssertEigenpairs(a, lm.Values, lm.Vectors!, 1e-8);
        // An explicit opInv works for operators that are not sparse matrices.
        var shifted = a.Subtract(SparseMatrices.Identity(150).Scale(0.5));
        var op = a.AsLinearOperator();
        var si = SparseLinalg.Eigsh(op, k: 3, sigma: 0.5, opInv: SparseLinalg.Splu(shifted));
        var near = dense.OrderBy(l => Math.Abs(l - 0.5)).Take(3).OrderBy(l => l).ToArray();
        Close(near, si.Values, 1e-9, "opInv");
    }

    [Fact]
    public void Eigsh_SmallProblem_UsesDensePath()
    {
        var a = Laplacian1D(10);
        var r = SparseLinalg.Eigsh(a, k: 2, which: EigshWhich.SA);
        Close([4 * Math.Pow(Math.Sin(Math.PI / 22), 2), 4 * Math.Pow(Math.Sin(2 * Math.PI / 22), 2)], r.Values, 1e-13);
        Assert.Throws<ArgumentException>(() => SparseLinalg.Eigsh(a, k: 10));
    }

    /// <summary>Block-diagonal 2×2 rotations (eigenvalues a ± bi) with a sparse perturbation: complex spectrum.</summary>
    private static CsrMatrix ComplexSpectrumMatrix(int blocks, int seed)
    {
        var r = new List<int>(); var c = new List<int>(); var v = new List<double>();
        for (var t = 0; t < blocks; t++)
        {
            var a = 1.0 + t;
            var b = t % 3 == 0 ? 0.0 : 0.05 * t;
            int i = 2 * t, j = 2 * t + 1;
            r.AddRange([i, i, j, j]); c.AddRange([i, j, i, j]); v.AddRange([a, b, -b, a + (b == 0 ? 0.5 : 0)]);
        }
        var n = 2 * blocks;
        var block = CsrMatrix.FromTriplets(n, n, r.ToArray(), c.ToArray(), v.ToArray());
        return block.Add(SparseMatrices.Random(n, n, 0.02, seed).Scale(0.05));
    }

    private static Complex[] DenseEigenvalues(SparseMatrixBase a) => Linalg.Eig(a.ToDense()).Values;

    private static void AssertSameSet(Complex[] expected, Complex[] actual, double tol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        var used = new bool[actual.Length];
        foreach (var e in expected)
        {
            var best = -1;
            for (var i = 0; i < actual.Length; i++)
                if (!used[i] && (best < 0 || (actual[i] - e).Magnitude < (actual[best] - e).Magnitude)) best = i;
            Assert.True((actual[best] - e).Magnitude <= tol, $"{what}: {e} not found (closest {actual[best]})");
            used[best] = true;
        }
    }

    private static void AssertComplexEigenpairs(SparseMatrixBase a, EigsResult r, double tol)
    {
        var n = a.Rows;
        var k = r.Values.Length;
        var dense = a.ToDense().To2DArray();
        for (var c = 0; c < k; c++)
        {
            var res = 0.0;
            for (var i = 0; i < n; i++)
            {
                var s = Complex.Zero;
                for (var j = 0; j < n; j++) s += dense[i, j] * r.Vectors![j, c];
                res += (s - r.Values[c] * r.Vectors![i, c]).Magnitude * (s - r.Values[c] * r.Vectors[i, c]).Magnitude;
            }
            Assert.True(Math.Sqrt(res) <= tol, $"‖Av − λv‖ = {Math.Sqrt(res)} for λ = {r.Values[c]}");
        }
    }

    [Fact]
    public void Eigs_NonSymmetric_MatchesDenseEig()
    {
        var a = ComplexSpectrumMatrix(60, 91); // n = 120
        var all = DenseEigenvalues(a);
        Assert.Contains(all, z => Math.Abs(z.Imaginary) > 0.1); // the test needs complex pairs
        var lm = SparseLinalg.Eigs(a, k: 6);
        AssertSameSet(all.OrderByDescending(z => z.Magnitude).Take(6).ToArray(), lm.Values, 1e-8, "LM");
        AssertComplexEigenpairs(a, lm, 1e-7);
        var sr = SparseLinalg.Eigs(a, k: 4, which: EigsWhich.SR, ncv: 30, maxiter: 5000);
        AssertSameSet(all.OrderBy(z => z.Real).Take(4).ToArray(), sr.Values, 1e-7, "SR");
        var li = SparseLinalg.Eigs(a, k: 3, which: EigsWhich.LI, ncv: 40, maxiter: 5000);
        Assert.Equal(all.Max(z => z.Imaginary), li.Values.Max(z => z.Imaginary), 1e-7);
    }

    [Fact]
    public void Eigs_ShiftInvert_FindsEigenvaluesNearSigma()
    {
        var a = ComplexSpectrumMatrix(60, 92);
        var all = DenseEigenvalues(a);
        const double sigma = 30.2;
        var r = SparseLinalg.Eigs(a, k: 4, sigma: sigma);
        AssertSameSet(all.OrderBy(z => (z - sigma).Magnitude).Take(4).ToArray(), r.Values, 1e-9, "sigma");
        AssertComplexEigenpairs(a, r, 1e-8);
    }

    [Fact]
    public void Eigs_ConvectionDiffusion_RealSpectrum()
    {
        var a = ConvectionDiffusion(12, 5);
        var all = DenseEigenvalues(a);
        var r = SparseLinalg.Eigs(a, k: 5, which: EigsWhich.LR);
        AssertSameSet(all.OrderByDescending(z => z.Real).Take(5).ToArray(), r.Values, 1e-8, "LR");
    }

    [Fact]
    public void Svds_MatchesDenseSvd()
    {
        var a = SparseMatrices.Random(80, 50, 0.1, seed: 101).Add(SparseMatrices.Eye(80, 50).Scale(0.5));
        var s = Decomposition.SingularValues(a.ToDense()).ToArray(); // descending
        var r = SparseLinalg.Svds(a, k: 4);
        Close(s.Take(4).Reverse().ToArray(), r.S, 1e-9, "largest");
        AssertSvdTriplets(a, r, 1e-8);
        var wide = a.ToCsc().Transpose(); // 50×80 exercises the AAᵀ branch
        var rw = SparseLinalg.Svds(wide, k: 3);
        Close(s.Take(3).Reverse().ToArray(), rw.S, 1e-9, "wide");
        AssertSvdTriplets(wide, rw, 1e-8);
        var small = SparseLinalg.Svds(a, k: 2, which: SvdsWhich.SM, ncv: 30, maxiter: 5000);
        // Smallest singular values via AᵀA lose accuracy in proportion to (σmax/σ)²·ε; still good here.
        Close(s[^2..].Reverse().ToArray(), small.S, 1e-7, "smallest");
    }

    private static void AssertSvdTriplets(SparseMatrixBase a, SvdsResult r, double tol)
    {
        var u = r.U!.AsContiguous().ToArray();
        var vt = r.Vt!.AsContiguous().ToArray();
        int m = a.Rows, n = a.Columns, k = r.S.Length;
        for (var c = 0; c < k; c++)
        {
            var v = new double[n];
            for (var j = 0; j < n; j++) v[j] = vt[c * n + j];
            var av = a.Dot(v);
            for (var i = 0; i < m; i++) Assert.True(Math.Abs(av[i] - r.S[c] * u[i * k + c]) <= tol, $"A v ≠ σ u at {i}");
        }
        var utu = LinAlg.Dot(r.U!.T, r.U!);
        Close(NdArray.Eye(k), utu, 1e-10, "UᵀU");
    }
}
