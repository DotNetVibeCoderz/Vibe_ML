using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsExtraAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Matrix, multivariate-discrete, directional and empirical distributions and the Covariance family. Values from scipy 1.17.1.</summary>
public class StatsDistributionsExtraTests
{
    private static readonly NdArray M = new([1.0, 2, 0, 0, -1, 3], 2, 3);
    private static readonly NdArray U = new([2.0, 0.3, 0.3, 1], 2, 2);
    private static readonly NdArray V = new([1.0, 0.2, 0.1, 0.2, 1.5, 0.0, 0.1, 0, 0.8], 3, 3);
    private static readonly NdArray X = new([0.5, 1.2, 0.3, -0.4, -0.2, 2.1], 2, 3);

    private static NdArray T(NdArray a) => new(Enumerable.Range(0, a.Shape[0] * a.Shape[1]).Select(k => a.AsContiguous().ToArray()[(k % a.Shape[0]) * a.Shape[1] + k / a.Shape[0]]).ToArray(), a.Shape[1], a.Shape[0]);

    [Fact]
    public void MatrixNormalAndMatrixTMatchScipy()
    {
        var mn = Stats.MatrixNormal(M, U, V);
        Near(-7.748784052888716, mn.LogPdf(X), 1e-12, 0, "matrix_normal logpdf");
        Near(9.626653303602765, mn.Entropy(), 1e-12, 0, "matrix_normal entropy");
        Near(-8.090420948347056, Stats.MatrixT(M, U, V, 1).LogPdf(X), 1e-12, 0, "matrix_t df=1");
        Near(-7.048355517051215, Stats.MatrixT(M, U, V, 3.5).LogPdf(X), 1e-12, 0, "matrix_t df=3.5");
        Near(-7.185707454621017, Stats.MatrixT(T(M), V, U, 2.5).LogPdf(T(X)), 1e-12, 0, "matrix_t transposed");
        // The matrix normal is a multivariate normal on vec(X) with covariance V ⊗ U (column-major vec).
        var u = U.AsContiguous().ToArray();
        var v = V.AsContiguous().ToArray();
        var k = new double[36];
        for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
            k[(a * 2 + i) * 6 + b * 2 + j] = v[a * 3 + b] * u[i * 2 + j];
        var vec = new double[6];
        var mvec = new double[6];
        for (int c = 0; c < 3; c++) for (int r = 0; r < 2; r++) { vec[c * 2 + r] = X.AsContiguous().ToArray()[r * 3 + c]; mvec[c * 2 + r] = M.AsContiguous().ToArray()[r * 3 + c]; }
        Near(Stats.MultivariateNormal(mvec, new NdArray(k, 6, 6)).LogPdf(vec), mn.LogPdf(X), 1e-12, 0, "vec identity");
        // Draws: sample mean and row covariance (E[(X−M)(X−M)ᵀ] = tr(V)·U) of 4000 draws.
        var draws = mn.Rvs(4000, seed: 3);
        var rowCov = new double[4];
        foreach (var d in draws)
        {
            var a = d.AsContiguous().ToArray();
            var mm = M.AsContiguous().ToArray();
            for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) for (int c = 0; c < 3; c++) rowCov[i * 2 + j] += (a[i * 3 + c] - mm[i * 3 + c]) * (a[j * 3 + c] - mm[j * 3 + c]) / 4000;
        }
        All([3.3 * 2, 3.3 * 0.3, 3.3 * 0.3, 3.3], rowCov, 0.06, 0.05, "row covariance");
        var tDraws = Stats.MatrixT(M, U, V, 6).Rvs(3000, seed: 4);
        var m00 = tDraws.Average(d => d.AsContiguous().ToArray()[0]);
        Near(1.0, m00, 0, 0.06, "matrix_t mean");
    }

    [Fact]
    public void MultivariateDiscreteMatchScipy()
    {
        var mh = Stats.MultivariateHypergeom([5, 7, 3], 6);
        Near(-1.561647248597751, mh.LogPmf([2, 3, 1]), 1e-12, 0, "mh logpmf");
        Near(0.06993006993006996, mh.Pmf([2, 4, 0]), 1e-12, 0, "mh pmf");
        Assert.Equal(double.NegativeInfinity, mh.LogPmf([2, 3, 2]));
        All([2.0, 2.8, 1.2], mh.Mean(), 1e-14, 1e-14, "mh mean");
        All([0.8571428571428571, 0.96, 0.6171428571428571], mh.Var(), 1e-13, 0, "mh var");
        Near(-0.6, mh.Cov().AsContiguous().ToArray()[1], 1e-13, 0, "mh cov");
        var draws = mh.Rvs(5000, seed: 1).AsContiguous().ToArray();
        for (var i = 0; i < 5000; i++) Assert.Equal(6.0, draws[i * 3] + draws[i * 3 + 1] + draws[i * 3 + 2]);
        Near(2.0, Enumerable.Range(0, 5000).Average(i => draws[i * 3]), 0, 0.05, "mh sample mean");

        var dm = Stats.DirichletMultinomial([1.5, 0.7, 2.2], 8);
        Near(-3.2821572529883363, dm.LogPmf([3, 1, 4]), 1e-12, 0, "dm logpmf");
        All([2.727272727272727, 1.2727272727272725, 4.0], dm.Mean(), 1e-13, 0, "dm mean");
        All([4.127640036730946, 2.4576063666972754, 4.592592592592593], dm.Var(), 1e-12, 0, "dm var");
        Near(-3.1313131313131315, dm.Cov().AsContiguous().ToArray()[2], 1e-12, 0, "dm cov");
        // Total probability over the simplex {x : Σx = 8}.
        var total = 0.0;
        for (var a = 0; a <= 8; a++) for (var b = 0; a + b <= 8; b++) total += dm.Pmf([a, b, 8 - a - b]);
        Near(1.0, total, 1e-12, 0, "dm sums to 1");
        var dd = dm.Rvs(6000, seed: 2).AsContiguous().ToArray();
        Near(4.0, Enumerable.Range(0, 6000).Average(i => dd[i * 3 + 2]), 0, 0.12, "dm sample mean");
    }

    [Fact]
    public void VonMisesFisherMatchesScipy()
    {
        var vm = Stats.VonMisesFisher([0.0, 0.6, 0.8], 3.2);
        Near(-0.9034633174132369, vm.LogPdf([0.36, 0.48, 0.8]), 1e-12, 0, "vmf logpdf");
        Near(1.6624116525136392, vm.Entropy(), 1e-12, 0, "vmf entropy");
        var vm2 = Stats.VonMisesFisher([0.6, 0.8], 40.0);
        Near(-15.077663927761705, vm2.LogPdf([1.0, 0.0]), 1e-12, 0, "vmf2 logpdf");
        Near(-0.419129725692315, vm2.Entropy(), 1e-11, 0, "vmf2 entropy");
        var vm5 = Stats.VonMisesFisher([1.0, 0, 0, 0, 0], 0.7);
        Near(-3.318950918280681, vm5.LogPdf([0.0, 1, 0, 0, 0]), 1e-12, 0, "vmf5 logpdf");
        Near(3.222293726172689, vm5.Entropy(), 1e-12, 0, "vmf5 entropy");
        double[] s =
        [
            -0.06954424512662631, 0.5220976305123682, 0.8500456824095621, 0.22914968052972076, 0.19386463983849395, 0.9538904157886366, 0.1014676592955032, 0.574747563315844, 0.8120157341945977,
            -0.12315936790551686, 0.8305455425250261, 0.5431628410420982, 0.19580964331506306, 0.8103487339339406, 0.5522621777711085, 0.46800878057766504, -0.02190245399334345, 0.8834523551450166,
            -0.054310493750633494, 0.6622658701013033, 0.7472979911437779, -0.7143516669034916, -0.5682485429253794, 0.408405790183472, 0.5030068932203393, 0.0588656176491833, 0.8622754226067229,
            0.03414978270265964, 0.05881028370639235, 0.9976848915723515, -0.015725570520234544, 0.32097903785239396, 0.9469557348108542, 0.037590598322970126, 0.9943778996722835, 0.09899262377096163,
        ];
        var (mu, kappa) = VonMisesFisherDistribution.Fit(new NdArray(s, 12, 3));
        All([0.06075713700733484, 0.45527464946787605, 0.8882756125519683], mu, 1e-12, 1e-14, "fit mu");
        Near(5.320674074881894, kappa, 1e-8, 0, "fit kappa");
        // Draws: unit norm and the mean resultant length A_p(κ) along μ.
        var d = Stats.VonMisesFisher([0.0, 0.6, 0.8], 5.0).Rvs(4000, seed: 9).AsContiguous().ToArray();
        var proj = 0.0;
        for (var i = 0; i < 4000; i++)
        {
            Near(1.0, d[i * 3] * d[i * 3] + d[i * 3 + 1] * d[i * 3 + 1] + d[i * 3 + 2] * d[i * 3 + 2], 1e-12, 0, "unit norm");
            proj += (0.6 * d[i * 3 + 1] + 0.8 * d[i * 3 + 2]) / 4000;
        }
        // A_3(5) = coth(5) − 1/5.
        Near(1 / Math.Tanh(5) - 0.2, proj, 0, 0.01, "mean resultant");
        var u2 = Stats.VonMisesFisher([1.0, 0], 0).Rvs(4000, seed: 1).AsContiguous().ToArray();
        Near(0.0, Enumerable.Range(0, 4000).Average(i => u2[2 * i]), 0, 0.04, "uniform circle mean");
    }

    [Fact]
    public void NormalInverseGammaMatchesScipy()
    {
        var nig = Stats.NormalInverseGamma(1.5, 2.0, 3.0, 1.2);
        Near(0.004760840330916505, nig.LogPdf(1.1, 0.4), 1e-10, 0, "logpdf");
        Near(0.1563690682522789, nig.Pdf(2.0, 0.9), 1e-12, 0, "pdf");
        Assert.Equal((1.5, 0.6), nig.Mean());
        var (vx, vs) = nig.Var();
        Near(0.3, vx, 1e-14, 0, "var x");
        Near(0.36, vs, 1e-14, 0, "var s2");
        var (x, s2) = nig.Rvs(20000, seed: 5);
        Near(0.6, s2.Average(), 0, 0.02, "s2 mean");
        Near(1.5, x.Average(), 0, 0.02, "x mean");
    }

    [Fact]
    public void RvHistogramAndRvDiscreteMatchScipy()
    {
        var h = Stats.RvHistogram([1, 3, 0, 2], [0, 1, 2, 4, 5]);
        All([0, 1 / 6.0, 1 / 6.0, 0.5, 0, 0, 0], h.Pdf([-1, 0, 0.5, 1, 2, 3.9, 5]), 1e-15, 1e-15, "pdf");
        All([1 / 12.0, 1 / 6.0, 2 / 3.0, 2 / 3.0, 5 / 6.0], h.Cdf([0.5, 1, 2, 3, 4.5]), 1e-15, 1e-15, "cdf");
        All([0.6, 1.6666666666666667, 4.1, 4.85], h.Ppf([0.1, 0.5, 0.7, 0.95]), 1e-14, 0, "ppf");
        Near(2.3333333333333335, h.Mean(), 1e-13, 0, "mean");
        Near(2.5555555555555536, h.Var(), 1e-12, 0, "var");
        Near(1.0114042647073518, h.Entropy(), 1e-12, 0, "entropy");
        Near(0.5076872690085319, h.Skew(), 1e-11, 0, "skew");
        Near(-1.3587901701323264, h.Kurtosis(), 1e-11, 0, "kurtosis");
        var hd = Stats.RvHistogram([1, 3, 0, 2], [0, 1, 2, 4, 5], density: true);
        All([1 / 6.0, 0.5, 1 / 3.0], hd.Pdf([0.5, 1.5, 4.5]), 1e-15, 0, "density pdf");

        var d = Stats.RvDiscrete([0.5, 2, 1.5, 4], [0.1, 0.4, 0.3, 0.2]);
        All([-0.5, 0.5, 1.5, 1.5, 2, 4], [d.Ppf(0), d.Ppf(0.1), d.Ppf(0.3), d.Ppf(0.4), d.Ppf(0.41), d.Ppf(1)], 0, 1e-15, "ppf");
        All([4, 2, 2, -0.5], [d.Isf(0), d.Isf(0.2), d.Isf(0.5), d.Isf(1)], 0, 1e-15, "isf");
        All([0, 0.4, 0.4, 1], [d.Cdf(0), d.Cdf(1.5), d.Cdf(1.7), d.Cdf(4)], 1e-15, 1e-15, "cdf");
        Near(0.6, d.Sf(1.5), 1e-15, 0, "sf");
        Near(2.1, d.Mean(), 1e-15, 0, "mean");
        Near(1.0899999999999999, d.Var(), 1e-14, 0, "var");
        Near(0.7882295209572215, d.Skew(), 1e-13, 0, "skew");
        Near(-0.22186684622505926, d.Kurtosis(), 1e-12, 0, "kurt");
        Near(1.2798542258336674, d.Entropy(), 1e-14, 0, "entropy");
        Near(17.025, d.Moment(3), 1e-14, 0, "moment 3");
        Assert.Equal((1.5, 2.0), d.Interval(0.5));
        Assert.Equal(0.0, d.Pmf(1.6));
        Assert.Throws<ArgumentException>(() => Stats.RvDiscrete([1, 1, 2], [0.2, 0.3, 0.5]));
        var draws = d.Rvs(20000, seed: 3);
        Near(0.4, draws.Count(v => v == 2) / 20000.0, 0, 0.015, "sample frequency");
    }

    [Fact]
    public void RandomTableMatchesScipy()
    {
        var t = Stats.RandomTable([3, 2, 4], [4, 5]);
        Near(-1.2527629684953676, t.LogPmf(new NdArray([1.0, 2, 1, 1, 2, 2], 3, 2)), 1e-12, 0, "logpmf");
        All([1.3333333333333333, 1.6666666666666667, 0.8888888888888888, 1.1111111111111112, 1.7777777777777777, 2.2222222222222223], t.Mean().AsContiguous().ToArray(), 1e-14, 0, "mean");
        // Both samplers preserve the margins and agree with the exact pmf of each table.
        foreach (var method in new[] { "boyett", "patefield" })
        {
            var draws = t.Rvs(6000, seed: 2, method: method);
            var counts = new Dictionary<string, int>();
            foreach (var d in draws)
            {
                var a = d.AsContiguous().ToArray();
                Assert.Equal(3.0, a[0] + a[1]);
                Assert.Equal(4.0, a[0] + a[2] + a[4]);
                var key = string.Join(",", a);
                counts[key] = counts.GetValueOrDefault(key) + 1;
            }
            var probe = new NdArray([1.0, 2, 1, 1, 2, 2], 3, 2);
            Near(t.Pmf(probe), counts.GetValueOrDefault("1,2,1,1,2,2") / 6000.0, 0, 0.02, $"{method} frequency");
        }
    }

    [Fact]
    public void RandomCorrelationHasUnitDiagonalAndGivenSpectrum()
    {
        double[] eigs = [0.5, 0.8, 1.2, 1.5];
        var c = Stats.RandomCorrelation(eigs, seed: 7);
        var a = c.AsContiguous().ToArray();
        for (var i = 0; i < 4; i++)
        {
            Near(1.0, a[i * 4 + i], 0, 1e-12, "diag");
            for (var j = 0; j < 4; j++) Near(a[i * 4 + j], a[j * 4 + i], 0, 1e-12, "symmetric");
        }
        var w = Linalg.Eigh(c).Values.OrderBy(v => v).ToArray();
        All(eigs, w, 0, 1e-10, "spectrum preserved");
        Assert.Throws<ArgumentException>(() => Stats.RandomCorrelation([0.5, 0.5, 0.5]));
    }

    [Fact]
    public void UnitaryGroupIsUnitaryAndHaarPhaseFree()
    {
        var q = Stats.UnitaryGroup(5, seed: 4);
        Assert.Equal([5, 5], q.Shape);
        for (var i = 0; i < 5; i++)
            for (var j = 0; j < 5; j++)
            {
                var s = Complex.Zero;
                for (var k = 0; k < 5; k++) s += Complex.Conjugate(q[k, i]) * q[k, j];
                Near(i == j ? 1 : 0, s.Real, 0, 1e-13, "QᴴQ real");
                Near(0, s.Imaginary, 0, 1e-13, "QᴴQ imag");
            }
        // Haar: E|Q_00|² = 1/n; and the trace's mean is zero (no phase bias, which the R-diagonal correction removes).
        double m2 = 0;
        var tr = Complex.Zero;
        for (var s = 0; s < 3000; s++)
        {
            var u = Stats.UnitaryGroup(3, seed: 100 + s);
            m2 += Complex.Abs(u[0, 0]) * Complex.Abs(u[0, 0]) / 3000;
            tr += (u[0, 0] + u[1, 1] + u[2, 2]) / 3000;
        }
        Near(1 / 3.0, m2, 0, 0.02, "E|Q00|²");
        Near(0, Complex.Abs(tr), 0, 0.05, "E tr Q");
    }

    private static readonly double[] Mean3 = [0.2, -0.1, 0.5];
    private static readonly NdArray A3 = new([2.0, 0.5, 0.1, 0.5, 1.0, 0.3, 0.1, 0.3, 1.5], 3, 3);
    private static readonly double[] W3 = [0.7172007490371514, 1.5166743934526727, 2.2661248575101762];
    private static readonly NdArray V3 = new([-0.32571614664802717, -0.36855136369846414, -0.8706772560075082, 0.896012604641014, 0.1736102037580609, -0.40868191723577835, -0.3017787337391217, 0.9132520952437433, -0.273679020744604], 3, 3);

    [Fact]
    public void CovarianceFormsMatchScipy()
    {
        double[] x = [0.7, 0.4, -0.3];
        var inv = Linalg.Inv(A3);
        var chol = Linalg.Cholesky(A3, lower: true);
        var forms = new (Covariance C, double[] W)[]
        {
            (Covariance.FromPrecision(inv), [0.3668918660341864, 0.474453773279045, -0.2449489742783178]),
            (Covariance.FromCholesky(chol), [0.4949747468305832, 0.24053511772118202, -0.34184628341350726]),
            (Covariance.FromEigendecomposition(W3, V3), [0.2608848091895568, -0.3755623032555788, -0.4589205781486207]),
        };
        foreach (var (c, w) in forms)
        {
            var mvn = Stats.MultivariateNormal(Mean3, c);
            Near(-3.666329353259925, mvn.LogPdf(x), 1e-12, 0, $"{c.GetType().Name} logpdf");
            Near(4.707911503361345, mvn.Entropy(), 1e-12, 0, $"{c.GetType().Name} entropy");
            Near(0.9021918074946532, c.LogPdet, 1e-12, 0, $"{c.GetType().Name} log_pdet");
            // scipy's whitening convention differs per form (any W with WWᵀ = Σ⁻¹ is valid), so pin each.
            All(w, c.Whiten(x), 1e-12, 1e-14, $"{c.GetType().Name} whiten");
            All(x, c.Colorize(c.Whiten(x)), 1e-12, 1e-14, $"{c.GetType().Name} colorize∘whiten");
            All(A3.AsContiguous().ToArray(), c.CovarianceMatrix.AsContiguous().ToArray(), 1e-12, 1e-14, $"{c.GetType().Name} matrix");
            // Same as the matrix-based distribution.
            Near(Stats.MultivariateNormal(Mean3, A3).LogPdf(x), mvn.LogPdf(x), 1e-12, 0, "vs matrix form");
        }
        var cd = Covariance.FromDiagonal([2.0, 0.0, 0.5]);
        var md = Stats.MultivariateNormal(Mean3, cd);
        Near(-2.0878770664093453, md.LogPdf([1.0, -0.1, 0.2]), 1e-12, 0, "diagonal singular logpdf");
        Assert.Equal(double.NegativeInfinity, md.LogPdf([1.0, 0.0, 0.2]));
        Assert.Equal(2, cd.Rank);
        Near(0.0, cd.LogPdet, 0, 1e-15, "diag log_pdet");
        var ce = Covariance.FromEigendecomposition([0.0, 1.0, 2.0], V3);
        var v = V3.AsContiguous().ToArray();
        var pt = Enumerable.Range(0, 3).Select(i => Mean3[i] + v[i * 3 + 1] * 0.3 + v[i * 3 + 2] * 0.5).ToArray();
        Near(-2.291950656689318, Stats.MultivariateNormal(Mean3, ce).LogPdf(pt), 1e-12, 0, "eigen singular logpdf");
        Assert.Equal(2, ce.Rank);
        // Draws through colorize have covariance Σ.
        var draws = Stats.MultivariateNormal(Mean3, Covariance.FromCholesky(chol)).Rvs(20000, seed: 1).AsContiguous().ToArray();
        var c01 = 0.0;
        for (var i = 0; i < 20000; i++) c01 += (draws[i * 3] - 0.2) * (draws[i * 3 + 1] + 0.1) / 20000;
        Near(0.5, c01, 0, 0.04, "sample covariance");
    }
}
