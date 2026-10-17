using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsExtraAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>bws_test, quantile_test, mvsdist, ppcc, goodness_of_fit, tukeylambda, gengamma. Values from scipy 1.17.1.</summary>
public class StatsTestsExtraTests
{
    [Fact]
    public void BwsExactMatchesScipy()
    {
        double[] x = [3.1, 4.5, 2.2, 5.0, 3.3], y = [4.4, 6.1, 5.2, 7.0, 3.9, 5.5];
        var r = Stats.BwsTest(x, y);
        Near(2.6014116161616165, r.Statistic, 1e-13, 0, "stat");
        Near(0.05194805194805195, r.Pvalue, 1e-12, 0, "p");
        Assert.Equal(462, r.NullDistribution.Length);
        r = Stats.BwsTest(x, y, Alternative.Less);
        Near(-2.6014116161616165, r.Statistic, 1e-13, 0, "stat less");
        Near(0.023809523809523808, r.Pvalue, 1e-12, 0, "p less");
        r = Stats.BwsTest(x, y, Alternative.Greater);
        Near(0.9783549783549783, r.Pvalue, 1e-12, 0, "p greater");
    }

    [Fact]
    public void BwsTiedAndRandomisedMatchScipy()
    {
        // Tied ranks, exact (924 partitions).
        double[] x = [3, 4, 2, 6, 2, 5], y = [9, 7, 5, 10, 6, 8];
        var r = Stats.BwsTest(x, y, seed: 1);
        Near(4.2875, r.Statistic, 1e-12, 0, "stat");
        Near(0.010822510822510822, r.Pvalue, 1e-12, 0, "exact p");
        // 12870 partitions exceed 9999 resamples, so the p-value is Monte Carlo (scipy 1.17.1 got 0.0112 with its own stream).
        double[] a = [68, 68, 59, 72, 64, 67, 70, 74], b = [60, 67, 61, 62, 67, 63, 56, 58];
        var m = Stats.BwsTest(a, b, seed: 7);
        Near(4.0152204241071425, m.Statistic, 1e-13, 0, "stat mc");
        Near(0.0112, m.Pvalue, 0, 0.006, "p mc");
    }

    [Fact]
    public void QuantileTestMatchesScipy()
    {
        var x = Enumerable.Range(0, 21).Select(i => Math.Exp(i * 0.05)).ToArray();
        var exp = new (Alternative Alt, double Stat, int Type, double P, double Lo, double Hi)[]
        {
            (Alternative.TwoSided, 10, 2, 0.617116375770549, 1.2214027581601699, 1.822118800390509),
            (Alternative.Less, 10, 2, 0.3085581878852745, double.NegativeInfinity, 1.7332530178673953),
            (Alternative.Greater, 10, 1, 0.825622133638227, 1.2840254166877414, double.PositiveInfinity),
        };
        foreach (var e in exp)
        {
            var r = Stats.QuantileTest(x, q: 1.6, p: 0.4, alternative: e.Alt);
            Assert.Equal(e.Stat, r.Statistic);
            Assert.Equal(e.Type, r.StatisticType);
            Near(e.P, r.Pvalue, 1e-12, 0, $"p {e.Alt}");
            var ci = r.ConfidenceInterval(0.9);
            Near(e.Lo, ci.Low, 1e-15, 0, $"lo {e.Alt}");
            Near(e.Hi, ci.High, 1e-15, 0, $"hi {e.Alt}");
        }
    }

    [Fact]
    public void MvsdistMatchesScipy()
    {
        var (m, v, s) = Stats.Mvsdist([3.1, 2.4, 5.6, 4.4, 3.9, 4.1, 2.2, 3.3]);
        var exp = new[]
        {
            new[] { 3.625, 0.4667708217101835, 3.0668219255143305, 0.0785704607765031, 0.0005823578162404113 },
            new[] { 1.7429999999999997, 1.4231535405570261, 0.7252203910865905, 0.8936491776893337, 0.6059087611067816 },
            new[] { 1.2562396166261733, 0.4060320499898059, 0.8515987265646834, 0.9953233168226829, 1.1725773644521178 },
        };
        var ds = new ContinuousDistribution[] { m, v, s };
        for (var i = 0; i < 3; i++)
            All(exp[i], [ds[i].Mean(), ds[i].Std(), ds[i].Ppf(0.1), ds[i].Cdf(3.0), ds[i].Pdf(1.2)], 1e-10, 1e-15, $"mvsdist {i}");
        // BayesMvs is the same posteriors summarised.
        var b = Stats.BayesMvs([3.1, 2.4, 5.6, 4.4, 3.9, 4.1, 2.2, 3.3]);
        Near(b.StdDev.Statistic, s.Mean(), 1e-12, 0, "std mean");
    }

    [Fact]
    public void TukeyLambdaAndGenGammaMatchScipy()
    {
        var exp = new (double Lam, double[] V)[]
        {
            (-0.3, [-9.926840182636317, 0.280729864458813, 0.18088116858291023, 0.1417628652534617, 13.354712585021783, double.NaN]),
            (0, [-4.59511985013459, 0.23147521650098238, 0.22171287329310907, 0.07585818002124345, 3.289868133696453, 1.2]),
            (0.14, [-3.3841891582663126, 0.20412100166084457, 0.24463660053096753, 0.043072579592610793, 2.1102970222144966, -0.027083773532802446]),
            (0.5, [-1.78997487421324, 0.11581254575403221, 0.3202042977178211, 0.0, 0.8584073464102073, -0.9183035664374515]),
            (1.5, [-0.6560250418237026, 0.0, 0.0, 0.0, 0.15677237527243484, -1.2469231426098304]),
        };
        foreach (var (lam, v) in exp)
        {
            var t = Stats.TukeyLambda(lam);
            All(v, [t.Ppf(0.01), t.Cdf(-1.2), t.Pdf(0.7), t.Sf(2.5), t.Var(), t.Kurtosis()], 1e-10, 1e-14, $"tukeylambda {lam}");
        }
        var g = Stats.GenGamma(2.5, -1.5, scale: 1.3);
        All([1.097575241435366, 0.5290232809956784, 0.6205403483992808, 0.9198942773481852, 0.33321078221937783, 0.4149193314556424],
            [g.Pdf(0.8), g.Cdf(0.8), g.Ppf(0.3), g.Mean(), g.Var(), g.Entropy()], 1e-11, 0, "gengamma c<0");
        var g2 = Stats.GenGamma(1.7, 2.2);
        All([0.7123300848413635, 0.19345677923114657, 2.2694922353999343, 1.1836602072802467, 0.5559980435225852],
            [g2.Pdf(0.8), g2.Cdf(0.8), g2.Isf(0.01), g2.Mean(), g2.Entropy()], 1e-11, 0, "gengamma c>0");
    }

    private static readonly double[] Z =
    [
        -1.6502230342896176, -0.9100395068592103, 1.0661451228934795, 0.2683944825505409, -1.595354954302351, -0.2180490121954659, -0.06808625010694425, -1.2410986152524772, 0.7996833296630184, -1.4787227958568836,
        -0.35679859280778464, 0.05440111227102107, -0.22625911980322044, 0.28372082253207187, 0.8120109946278384, 1.9853848906953078, -0.7297769139589493, 0.4914750931267825, 0.6586430610342737, -0.6986243138476147,
        -2.858863396867525, 2.1844016842234772, -0.6780316411015115, -0.6222105612713176, 1.5096526787998528, 0.27831077149958, -0.09327614459030076, 0.9502448250124151, -2.1344808447348025, 0.6974818073938549,
        -0.4138780011783525, -1.6162029214054117, 0.5327106371680714, 1.7714783993826086, -1.0303694628914932, 0.4285395607876454, -0.6788921510356647, 0.8269068941307327, 0.7532763893447805, -0.9827107571643383,
        1.192690789605005, 0.5228490016723, 0.6108102052583531, 1.1482277296420695, -0.2330163008851196, 0.8924000595072036, 1.4350355402189785, -1.5449627143396403, 1.2868651429442268, -0.34777853733454633,
        -0.06602875864483393, -1.306050513321278, 0.6665903006955292, -0.701324993571975, 1.39553040970354, -0.762413571880544, 0.20143372362436185, -0.3286758984022818, 0.370651923529334, -1.0750685351049893,
    ];

    [Fact]
    public void PpccMatchesScipy()
    {
        Near(0.1828135869694026, Stats.PpccMax(Z), 0, 1e-6, "ppcc_max tukeylambda");
        var shifted = Z.Select(v => Math.Abs(v) + 0.1).ToArray();
        Near(1.4922699721640733, Stats.PpccMax(shifted, [0.5, 3], c => Stats.WeibullMin(c)), 0, 1e-6, "ppcc_max weibull_min");
        var (sv, pp) = Stats.PpccPlot(Z, -0.5, 1.0, n: 4);
        All([-0.5, 0.0, 0.5, 1.0], sv, 1e-15, 1e-15, "svals");
        All([0.9248943674364276, 0.9923694856939256, 0.991137157482315, 0.9827328323444363], pp, 1e-10, 0, "ppcc");
    }

    private static readonly double[] W =
    [
        3.4741047822349733, -2.916571431671465, 2.1840605295145146, -0.8922989906237211, 4.271663134274451, -4.102501820330284, -0.743483613983766, 4.128739963226202, 5.469203145296471, -4.474016140378624,
        0.50588046574609, 2.984060277627731, 0.17235158615038815, 6.771920693969431, -1.5736800448533423, 3.063595838860778, -1.1452165556335334, 6.2178888294046555, 1.9350463128333248, 0.8832483079981523,
        -3.154554849197849, 7.045476635200042, 4.258335778092163, 4.260691512528085, 5.413643776753344,
    ];

    [Fact]
    public void GoodnessOfFitMatchesScipy()
    {
        // Statistics are deterministic; p-values are Monte Carlo (scipy's stream differs), so they get a
        // binomial tolerance: sd ≈ √(p(1−p)/B) ≈ 0.016 at B = 999.
        var exp = new (string S, double Stat, double P)[]
        {
            ("ad", 0.3529611137023245, 0.4411), ("ks", 0.11797941878122542, 0.484), ("cvm", 0.05607379450255794, 0.4174), ("filliben", 0.9851064350132566, 0.5594),
        };
        foreach (var e in exp)
        {
            var r = Stats.GoodnessOfFit(Stats.Norm(), W, statistic: e.S, nMcSamples: 999, seed: 3);
            Near(1.7615035249215287, r.FitResult.Loc, 1e-13, 0, "loc");
            Near(3.4627149551514393, r.FitResult.Scale, 1e-13, 0, "scale");
            Near(e.Stat, r.Statistic, 1e-11, 0, $"{e.S} stat");
            Near(e.P, r.Pvalue, 0, 0.065, $"{e.S} p");
        }
        var known = Stats.GoodnessOfFit(Stats.Norm(), W, knownParams: new Dictionary<string, double> { ["loc"] = 2 }, nMcSamples: 999, seed: 3);
        Near(2.0, known.FitResult.Loc, 0, 0, "known loc");
        Near(3.4011262303579843, known.FitResult.Scale, 1e-13, 0, "known scale");
        Near(0.39251232634144984, known.Statistic, 1e-11, 0, "known stat");
        Near(0.7642, known.Pvalue, 0, 0.06, "known p");
        var all = Stats.GoodnessOfFit(Stats.Norm(), W, knownParams: new Dictionary<string, double> { ["loc"] = 2, ["scale"] = 3 }, statistic: "ks", nMcSamples: 999, seed: 3);
        Near(0.13977147669375323, all.Statistic, 1e-12, 0, "all-known stat");
        Near(0.6635, all.Pvalue, 0, 0.065, "all-known p");
        Assert.Throws<ArgumentException>(() => Stats.GoodnessOfFit(Stats.Norm(), W, knownParams: new Dictionary<string, double> { ["mu"] = 1 }));
    }

    [Fact]
    public void NormplotsAndUniformDirectionMatchScipy()
    {
        double[] x = [0.8, 1.5, 2.2, 3.9, 0.4, 6.1, 2.7, 1.1, 9.5, 3.3];
        var (l, r) = Stats.BoxcoxNormplot(x, -1, 1, 3);
        All([-1.0, 0.0, 1.0], l, 0, 1e-15, "lambdas");
        All([0.8711337104515093, 0.9962409483960205, 0.9245326211306526], r, 1e-10, 0, "boxcox_normplot");
        var (_, ry) = Stats.YeojohnsonNormplot(x.Select(v => v - 2).ToArray(), -1, 2, 3);
        All([0.892566766058564, 0.9902076747244873, 0.7614035527436597], ry, 1e-10, 0, "yeojohnson_normplot");
        var u = Stats.UniformDirection(4, 3000, seed: 1).AsContiguous().ToArray();
        double m0 = 0, s0 = 0;
        for (var i = 0; i < 3000; i++)
        {
            Near(1.0, u[i * 4] * u[i * 4] + u[i * 4 + 1] * u[i * 4 + 1] + u[i * 4 + 2] * u[i * 4 + 2] + u[i * 4 + 3] * u[i * 4 + 3], 1e-12, 0, "unit");
            m0 += u[i * 4] / 3000;
            s0 += u[i * 4] * u[i * 4] / 3000;
        }
        Near(0, m0, 0, 0.03, "mean");
        Near(0.25, s0, 0, 0.02, "E x0² = 1/d");
    }
}
