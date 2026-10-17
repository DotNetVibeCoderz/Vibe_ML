using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Sparse;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SparseAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// The restarted Lanczos/Arnoldi drivers after the ARPACK-policy rewrite, and the blocked
/// Gram–Schmidt kernels underneath them.
/// </summary>
public class SparseKrylovTests
{
    private static double[] PoissonSpectrum(int nx, int ny)
    {
        var list = new List<double>();
        for (var i = 1; i <= nx; i++)
            for (var j = 1; j <= ny; j++)
                list.Add(4 * Math.Pow(Math.Sin(i * Math.PI / (2 * (nx + 1))), 2) + 4 * Math.Pow(Math.Sin(j * Math.PI / (2 * (ny + 1))), 2));
        list.Sort();
        return list.ToArray();
    }

    private static void AssertResiduals(ILinearOperator a, double[] values, NdArray vectors, double tol)
    {
        var n = a.Rows;
        var k = values.Length;
        var v = vectors.AsContiguous().ToArray();
        for (var c = 0; c < k; c++)
        {
            var x = new double[n];
            for (var i = 0; i < n; i++) x[i] = v[i * k + c];
            var ax = a.Dot(x);
            for (var i = 0; i < n; i++) ax[i] -= values[c] * x[i];
            Assert.True(Norm(ax) <= tol, $"‖Av − λv‖ = {Norm(ax):G3} for λ = {values[c]}");
        }
    }

    [Fact]
    public void Eigsh_SmallestAlgebraic_ConvergesWithDefaultNcv()
    {
        // The smallest eigenvalues of a Laplacian are tiny next to ‖A‖ = 8. Folding the
        // reorthogonalisation coefficients back into the projected matrix left O(ε‖A‖) noise in
        // its couplings, the relative test ‖r‖ ≤ ε·|θ| became unreachable, and this never
        // converged. Values: closed form, and scipy 1.17.1 eigsh(k=6, which='SA') agrees:
        // 0.016129750848728714, 0.0337005056555119, 0.04680851512752961, ...
        var a = Poisson2D(40, 30);
        var exact = PoissonSpectrum(40, 30);
        var r = SparseLinalg.Eigsh(a, k: 6, which: EigshWhich.SA);
        Close(exact[..6], r.Values, 1e-10, "SA");
        Close(0.016129750848728714, r.Values[0], 1e-12, "scipy SA[0]");
        Close(0.0337005056555119, r.Values[1], 1e-12, "scipy SA[1]");
        AssertResiduals(a, r.Values, r.Vectors!, 1e-7);
    }

    [Fact]
    public void Eigsh_LargestMagnitude_ClusteredTop_MatchesClosedForm()
    {
        var a = Poisson2D(40, 30);
        var exact = PoissonSpectrum(40, 30);
        var r = SparseLinalg.Eigsh(a, k: 6);
        Close(exact[^6..], r.Values, 1e-10, "LM");
        AssertResiduals(a, r.Values, r.Vectors!, 1e-7);
        // Restarted, not solved densely: the default basis is max(2k + 1, 20) = 20 vectors.
        Assert.True(r.Restarts > 0);
    }

    [Fact]
    public void Eigsh_ShiftInvert_CholeskyAndLuRoutesAgree()
    {
        var a = Poisson2D(25, 20);
        var exact = PoissonSpectrum(25, 20);
        // σ = −0.5 is below the Gershgorin bound (0): A − σI is SPD and goes through Cholesky.
        var below = SparseLinalg.Eigsh(a, k: 5, sigma: -0.5);
        Close(exact[..5], below.Values, 1e-11, "σ below the spectrum");
        AssertResiduals(a, below.Values, below.Vectors!, 1e-9);
        // The same shift through an explicit LU operator.
        var shifted = a.Subtract(SparseMatrices.Identity(a.Rows).Scale(-0.5));
        var viaLu = SparseLinalg.Eigsh(a, k: 5, sigma: -0.5, opInv: SparseLinalg.Splu(shifted));
        Close(below.Values, viaLu.Values, 1e-11, "LU route");
        // σ inside the spectrum: indefinite, LU only.
        var inside = SparseLinalg.Eigsh(a, k: 4, sigma: 2.9);
        var nearest = exact.OrderBy(l => Math.Abs(l - 2.9)).Take(4).OrderBy(l => l).ToArray();
        Close(nearest, inside.Values, 1e-10, "σ inside");
    }

    [Fact]
    public void Eigs_ConvectionDiffusion_MatchesScipy()
    {
        // scipy 1.17.1: A = kronsum(diags([-1-c, 2+c, -1]), lap(12)), c = 40/13.
        var a = ConvectionDiffusion(12, 40);
        var lm = SparseLinalg.Eigs(a, k: 4);
        double[] lmRef = [12.939740065329955, 12.768768481784342, 12.594524640264058, 12.494877926820102];
        for (var i = 0; i < 4; i++)
        {
            Close(lmRef[i], lm.Values[i].Real, 1e-10, $"LM[{i}]");
            Assert.True(Math.Abs(lm.Values[i].Imaginary) <= 1e-10, "real spectrum");
        }
        var si = SparseLinalg.Eigs(a, k: 4, sigma: 1.0);
        double[] siRef = [1.2141060885161734, 1.3850776720618583, 1.5593215135822667, 1.6589682270260755];
        var got = si.Values.Select(z => z.Real).OrderBy(x => x).ToArray();
        Close(siRef, got, 1e-10, "σ = 1");
        // Eigenvectors satisfy A x = λ x (complex arithmetic, real spectrum here).
        var v = si.Vectors!;
        for (var c = 0; c < 4; c++)
        {
            var re = new double[a.Rows]; var im = new double[a.Rows];
            for (var i = 0; i < a.Rows; i++) { re[i] = v[i, c].Real; im[i] = v[i, c].Imaginary; }
            var are = a.Dot(re); var aim = a.Dot(im);
            var r = 0.0;
            for (var i = 0; i < a.Rows; i++)
            {
                var d = new Complex(are[i], aim[i]) - si.Values[c] * v[i, c];
                r += d.Real * d.Real + d.Imaginary * d.Imaginary;
            }
            Assert.True(Math.Sqrt(r) <= 1e-9, $"residual {Math.Sqrt(r):G3}");
        }
    }

    [Fact]
    public void GramSchmidtKernels_MatchScalarLoops()
    {
        var rng = new GraviRandom(4);
        foreach (var (count, n) in new[] { (1, 7), (5, 33), (9, 1001), (13, 40_000) }) // the last one runs in parallel
        {
            var v = Enumerable.Range(0, count * n).Select(_ => rng.Uniform(-1, 1)).ToArray();
            var w = Enumerable.Range(0, n).Select(_ => rng.Uniform(-1, 1)).ToArray();
            var h = new double[count];
            SparseKrylovKernels.MultiDot(v, n, count, w, h);
            var w2 = (double[])w.Clone();
            SparseKrylovKernels.MultiAxpy(v, n, count, h, w2);
            for (var i = 0; i < count; i++)
            {
                var d = 0.0;
                for (var t = 0; t < n; t++) d += v[i * n + t] * w[t];
                Close(d, h[i], 1e-11 * Math.Sqrt(n), $"dot {i} ({count}×{n})");
            }
            for (var t = 0; t < n; t++)
            {
                var e = w[t];
                for (var i = 0; i < count; i++) e -= h[i] * v[i * n + t];
                Close(e, w2[t], 1e-12, $"axpy ({count}×{n})");
            }
        }
    }

    [Fact]
    public void Orthogonalize_LeavesWOrthogonalToTheBasis()
    {
        // A basis of 12 orthonormal vectors and a w that is almost inside its span: the case
        // where one CGS pass is not enough and the DGKS correction must fire.
        const int n = 500, m = 12;
        var rng = new GraviRandom(8);
        var q = Decomposition.Qr(new NdArray(Enumerable.Range(0, n * m).Select(_ => rng.Uniform(-1, 1)).ToArray(), n, m)).Q.AsContiguous().ToArray();
        var v = new double[m * n];
        for (var i = 0; i < n; i++)
            for (var c = 0; c < m; c++) v[c * n + i] = q[i * m + c];
        var w = new double[n];
        for (var c = 0; c < m; c++)
            for (var i = 0; i < n; i++) w[i] += (c + 1) * v[c * n + i];
        for (var i = 0; i < n; i++) w[i] += 1e-9 * rng.Uniform(-1, 1);
        var h = new double[m];
        var nrm = SparseKrylovKernels.Orthogonalize(v, n, m, w, h, new double[m], Norm(w));
        for (var c = 0; c < m; c++)
        {
            var d = 0.0;
            for (var i = 0; i < n; i++) d += v[c * n + i] * w[i];
            Assert.True(Math.Abs(d) <= 1e-14 * Math.Max(nrm, 1e-300) * 1e3 + 1e-24, $"⟨v{c}, w⟩ = {d:G3}");
            Close(c + 1, h[c], 1e-8, $"coefficient {c}");
        }
    }
}
