using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsExtraAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>multiscale_graphcorr, sobol_indices and wasserstein_distance_nd. Values from scipy 1.17.1.</summary>
public class StatsMgcSobolTests
{
    private static readonly double[] X = [0.6100058474907604, 0.6158815794729875, 0.030651122084284, -0.4283972398237168, -0.8921385952366871, -0.23326223842896354, -0.1830535891600027, -0.9094496121951097, -0.9024845785456639, 0.9983522301301428, 0.30473822317597543, -0.5309795966603521, -0.13010489554971594, 0.9483723865185107, 0.7953552162170976, 0.6884620752174819, -0.21519067133044367, -0.013953962536514819, 0.3533787036621321, -0.8783945740838879, 0.11119223384144683, -0.45709679093979694, 0.7593023466698443, -0.871571125375618, 0.35836306604272994, 0.7401770046550067, -0.5453629496781838, 0.790896478828252, 0.74439093604867, -0.9629655646595785];
    private static readonly double[] Y = [0.2431652592975349, 0.38137915933790095, -0.0028490828193811404, 0.1530904199927302, 0.6911186225988648, 0.014792238829581361, -0.07562427366521185, 0.6915777225013615, 0.836956987758734, 0.8857721816166931, 0.20989499478227996, 0.3535980976569738, -0.18285438539896962, 0.9266230704520644, 0.5224182572056415, 0.47728575102858867, 0.05067022428458919, -0.19864826575264163, 0.10153425582535197, 0.7459980246400757, 0.10856376605096044, 0.09079279549183886, 0.6503442434429008, 0.6497389502848805, 0.09529499783355436, 0.46381468137783666, 0.44229387577385637, 0.6823385502017569, 0.7972911159559364, 0.9914943166283728];
    private static readonly double[] X2 = [0.8449927337191754, 0.8406828762653401, -0.6066115359095516, -0.07002844663838208, 1.350388867744626, -0.3965507651729716, 0.18879953129109867, -0.021223460417289796, 0.6092164928327407, -0.3649087419473726, -0.15236188875684828, 0.24238142867215043, 0.10302313848680769, -0.8649727462818201, 0.8957830431894438, -1.298481208246912, -1.2011155488035266, -1.282491794740462, 0.9669722789332148, -0.36060836889927, -0.9710363785210655, -1.1360213941896466, 0.42113113746240616, -1.054840662577835];
    private static readonly double[] Y2 = [1.0539463623895298, -0.9062265405846728, 2.1024238718929666, 0.2163799872064987, 1.2150522154922205, -0.5273914609786027, 0.17899929617589125, 2.4609528359062733, -2.6606781739016907, 1.3042909784273158, -2.8604456096389708, 0.7398792165078788];
    private static readonly double[] A = [-0.3522570104112305, 0.2650909069303235, -0.4642445916374816, -0.4786384630527245, -0.7213157478486046, -0.5197597723701788, 0.1602267063174243, -0.3803527272151388, 0.1004418528325065, 1.9011872000332568, 0.4791181507969435, -1.5760898454352261, 1.7335295856131068, 0.34781036998853543];
    private static readonly double[] B = [-0.1414132864509846, 1.70704895768581, 0.817655610909839, 0.18478146696493092, 0.16649107456638368, -0.1934317902518573, 0.8481189839898386, 1.8688166937495658, 0.47494790119362135, 1.220824125056355, 2.675646053775279, -0.41459063297984966, 1.0575223239234917, 0.4935630731481257, -0.2592560998278446, -0.22584719943659715, 0.7847148568346378, 1.2338634112001694];
    private static readonly double[] U = [-0.531589820743032, 0.05439054435949247, 0.7222086544347099, 0.2965612469178253, 0.8686701897408925, 0.4613956081509272, 0.4880056756807095, 1.8266340649418156, 0.6231718736925668, 0.12947113235560923, 0.7099438034419987, -0.9191236755082103, -0.3377047857608698, 0.793220854194603, 0.6307867000938839, 1.5483616908312823, 0.010473028918682835, -1.4623277904204515];
    private static readonly double[] V = [2.2472469308762064, 1.39289281055875, -0.7587374336035666, 1.6758236699684164, 0.33240279877908363, -1.513928769904867, -0.12035392142294682, -0.20890383399376783, 1.890848160322242, -0.49206869541952386, 0.04638265962063193, 0.02323919315462858, -0.0789727629772628, -0.6140740610326185, 0.5190869090256768, 1.3768046464135546, 0.9239776511717521, -0.6274991160593102];
    private static readonly double[] FA = [-1.1498157571616898, 0.11897355358204866, -0.7065801636237695, -0.6301656488493008, -1.6796945788401643, 1.9504916026999646, 0.9166191525408893, -0.9739073319069764];
    private static readonly double[] FB = [0.9082102667945468, 1.3424871058246228, -2.389553349676638, -0.548944785564265, -0.3879732181256381, 0.6482544703432906, -0.12146542230081914, -0.2304313358523204];
    private static readonly double[] FAB = [-0.05837138907215449, 1.8533257078420022, 2.159980469779012, -0.5248264370136562, -0.9262439932504336, 2.6925531473868567, -0.9797446331927372, -0.5734020178922011, 0.036581509348738155, 0.4832462441452338, 1.029035549437811, 0.39097722167013954, -0.8723830716951919, 0.5063823263497103, 0.24991757985122218, 1.876918992454176, -0.01499979815571943, -1.3369728999172332, -1.0450199143839616, 1.450153058953162, -0.5401312070783946, -2.1044660100855603, -0.5806997054894565, 1.5099831293121058e-05];

    [Fact]
    public void MgcMatchesScipyOneDimensional()
    {
        var r = Stats.MultiscaleGraphcorr(X, Y, reps: 200, seed: 1);
        Near(0.4984296127496668, r.Statistic, 1e-12, 0, "stat");
        Assert.Equal((6, 24), r.OptScale);
        Assert.Equal([30, 30], r.MgcMap.Shape);
        var map = r.MgcMap.AsContiguous().ToArray();
        Near(0.4496906863559994, map[5 * 30 + 7], 1e-12, 0, "map[5,7]");
        Near(0.17056302153546807, map[^1], 1e-12, 0, "global dcorr");
        // A strong dependence: scipy got 1/201, the smallest possible p.
        Near(1.0 / 201, r.Pvalue, 0, 0.011, "p");
    }

    [Fact]
    public void MgcMatchesScipyMultivariateAndTwoSample()
    {
        var r = Stats.MultiscaleGraphcorr(new NdArray(X2, 12, 2), new NdArray(Y2, 12, 1), reps: 200, seed: 1);
        Near(0.8135401560925078, r.Statistic, 1e-12, 0, "stat");
        Assert.Equal((8, 9), r.OptScale);
        Near(0.6726246149524963, r.MgcMap.AsContiguous().ToArray()[3 * r.MgcMap.Shape[1] + 4], 1e-12, 0, "map[3,4]");
        var t = Stats.MultiscaleGraphcorr(new NdArray(A, 7, 2), new NdArray(B, 9, 2), reps: 400, seed: 1);
        Near(0.0824288072056061, t.Statistic, 1e-12, 0, "two-sample stat");
        Assert.Equal((16, 2), t.OptScale);
        Near(0.14925373134328357, t.Pvalue, 0, 0.07, "two-sample p");
        Assert.Throws<ArgumentException>(() => Stats.MultiscaleGraphcorr([1.0, 2, 3, 4], [1.0, 2, 3, 4]));
    }

    [Fact]
    public void WassersteinNdMatchesScipy()
    {
        var u = new NdArray(U, 6, 3);
        var v = new NdArray(V, 6, 3);
        Near(1.2621772798974167, Stats.WassersteinDistanceNd(u, v), 1e-12, 0, "equal weights (assignment)");
        double[] uw = [1, 2, 1, 3, 1, 1.5], vw = [2, 1, 1, 1, 0.5, 2];
        Near(1.4066614105525792, Stats.WassersteinDistanceNd(u, v, uw, vw), 1e-8, 0, "weighted (LP)");
        Near(1.5204816588455325, Stats.WassersteinDistanceNd(new NdArray(U[..12], 4, 3), v), 1e-8, 0, "unequal sizes (LP)");
        var u0 = Enumerable.Range(0, 6).Select(i => U[i * 3]).ToArray();
        var v0 = Enumerable.Range(0, 6).Select(i => V[i * 3]).ToArray();
        Near(0.8597827537001925, Stats.WassersteinDistanceNd(new NdArray(u0, 6), new NdArray(v0, 6), uw, vw), 1e-12, 0, "1-D");
        // Uniform LP and assignment must agree: weights of all ones force the LP path.
        var ones = Enumerable.Repeat(1.0, 6).ToArray();
        Near(1.2621772798974167, Stats.WassersteinDistanceNd(u, v, ones, ones), 1e-8, 0, "LP = assignment");
    }

    [Fact]
    public void SobolIndicesFromEvaluationsMatchScipy()
    {
        var fab = new double[3][][];
        for (var i = 0; i < 3; i++) fab[i] = [FAB[(i * 8)..((i + 1) * 8)]];
        var r = Stats.SobolIndices([FA], [FB], fab);
        All([-0.22679275936028312, -0.40164932105186235, -0.48905825005563175], r.FirstOrder, 1e-12, 1e-15, "first");
        All([0.8925099701833451, 0.8719694140209822, 1.4846400047175001], r.TotalOrder, 1e-12, 1e-15, "total");
    }

    [Fact]
    public void SobolIndicesRecoverIshigamiAnalyticValues()
    {
        // Ishigami (a = 7, b = 0.1) on U(−π, π)³: S = (0.3139, 0.4424, 0), S_T = (0.5576, 0.4424, 0.2437)
        // (Saltelli et al. 2008). The estimator's Monte Carlo error at n = 2¹² is a few 1e-2.
        var dists = Enumerable.Repeat<IUnivariateDistribution>(Stats.Uniform(-Math.PI, 2 * Math.PI), 3).ToArray();
        var r = Stats.SobolIndices(p => Math.Sin(p[0]) + 7 * Math.Pow(Math.Sin(p[1]), 2) + 0.1 * Math.Pow(p[2], 4) * Math.Sin(p[0]), 4096, dists, seed: 11);
        All([0.3139, 0.4424, 0.0], r.FirstOrder, 0, 0.04, "first");
        All([0.5576, 0.4424, 0.2437], r.TotalOrder, 0, 0.04, "total");
        var (first, total) = r.Bootstrap(nResamples: 999, seed: 2);
        for (var i = 0; i < 3; i++)
        {
            Assert.True(first[i].ConfidenceInterval.Low <= r.FirstOrder[i] && r.FirstOrder[i] <= first[i].ConfidenceInterval.High, $"first CI {i}");
            Assert.True(total[i].StandardError > 0 && total[i].StandardError < 0.1, $"total se {i}");
        }
        Assert.Throws<ArgumentException>(() => Stats.SobolIndices(p => p[0], 1000, dists));
    }

    [Fact]
    public void SobolBootstrapSumUpdatesMatchGenericBootstrap()
    {
        // The O(n·d) resample/jackknife-by-removal path must equal Stats.Bootstrap re-running the naive
        // estimator on resampled point indices (same seed ⇒ same resamples).
        var fab = new double[3][][];
        for (var i = 0; i < 3; i++) fab[i] = [FAB[(i * 8)..((i + 1) * 8)]];
        var r = Stats.SobolIndices([FA], [FB], fab);
        var (first, total) = r.Bootstrap(0.9, 199, seed: 5);
        var idx = Enumerable.Range(0, 8).Select(i => (double)i).ToArray();
        for (var k = 0; k < 3; k++)
        {
            double Naive(double[] sel, bool tot)
            {
                var a = sel.Select(t => r.FA[0][(int)t]).ToArray();
                var b = sel.Select(t => r.FB[0][(int)t]).ToArray();
                var ab = sel.Select(t => r.FAB[k][0][(int)t]).ToArray();
                var mu = (a.Sum() + b.Sum()) / (2.0 * a.Length);
                var v = (a.Sum(x => (x - mu) * (x - mu)) + b.Sum(x => (x - mu) * (x - mu))) / (2.0 * a.Length);
                var e = tot ? 0.5 * a.Zip(ab, (x, y) => (x - y) * (x - y)).Average() : b.Select((x, j) => x * (ab[j] - a[j])).Average();
                return e / v;
            }
            var g1 = Stats.Bootstrap(idx, s => Naive(s, false), 199, 0.9, seed: 5);
            var g2 = Stats.Bootstrap(idx, s => Naive(s, true), 199, 0.9, seed: 5);
            All(g1.BootstrapDistribution, first[k].BootstrapDistribution, 1e-10, 1e-12, $"first dist {k}");
            Near(g1.ConfidenceInterval.Low, first[k].ConfidenceInterval.Low, 1e-9, 1e-12, $"first lo {k}");
            Near(g1.ConfidenceInterval.High, first[k].ConfidenceInterval.High, 1e-9, 1e-12, $"first hi {k}");
            Near(g2.ConfidenceInterval.Low, total[k].ConfidenceInterval.Low, 1e-9, 1e-12, $"total lo {k}");
            Near(g2.ConfidenceInterval.High, total[k].ConfidenceInterval.High, 1e-9, 1e-12, $"total hi {k}");
            Near(g2.StandardError, total[k].StandardError, 1e-9, 1e-12, $"total se {k}");
        }
    }
}
