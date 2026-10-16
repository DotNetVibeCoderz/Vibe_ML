using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class SignalWindowTests
{
    private static double[] Make(string name, int n, bool sym) => name switch
    {
        "exponential" => Signal.Exponential(n, null, 2.5, sym),
        _ => Signal.GetWindow(name switch
        {
            "tukey" => ("tukey", 0.3),
            "kaiser" => ("kaiser", 8.6),
            "gaussian" => ("gaussian", 1.7),
            "general_gaussian" => ("general_gaussian", 1.5, 2.0),
            "general_hamming" => ("general_hamming", 0.6),
            "chebwin" => ("chebwin", 60),
            "taylor" => ("taylor", 5, 35),
            "dpss" => ("dpss", 2.5),
            _ => name,
        }, n, fftbins: !sym),
    };

    [Fact]
    public void AllNamedWindowsMatchScipy()
    {
        // Every (name, n, sym) case in the generated table: scipy 1.17.1 get_window values.
        var count = 0;
        foreach (var (key, expected) in SignalWindowRef.Windows)
        {
            var parts = key.Split('|');
            var name = parts[0];
            if (name.Contains('_') && !(name is "general_gaussian" or "general_hamming")) continue;
            if (name is "kbd") continue;
            var n = int.Parse(parts[1]);
            var sym = parts[2] == "1";
            var tol = name == "dpss" ? 1e-9 : name == "chebwin" ? 1e-11 : 1e-13;
            SignalAssert.Close(expected, Make(name, n, sym), tol, tol, key);
            count++;
        }
        Assert.True(count > 250, $"only {count} cases ran");
    }

    [Fact]
    public void SpecialWindowsMatchScipy()
    {
        var r = SignalWindowRef.Windows;
        SignalAssert.Close(r["exponential_center|9|0"], Signal.Exponential(9, 2.0, 1.5, sym: false), 1e-14, 1e-15, "exp center");
        SignalAssert.Close(r["kbd|8|1"], Signal.KaiserBesselDerived(8, 5.0), 1e-13, 1e-14, "kbd8");
        SignalAssert.Close(r["kbd|32|1"], Signal.KaiserBesselDerived(32, 8.0), 1e-13, 1e-14, "kbd32");
        SignalAssert.Close(r["general_cosine|9|1"], Signal.GeneralCosine(9, [0.5, 0.3, 0.2]), 1e-14, 1e-15, "gc");
        SignalAssert.Close(r["taylor_nonorm|12|1"], Signal.Taylor(12, 4, 30, norm: false), 1e-13, 1e-14, "taylor");
        SignalAssert.Close(r["chebwin_x|101|1"], Signal.Chebwin(101, 100), 1e-10, 1e-12, "cheb101");
        SignalAssert.Close(r["chebwin_x|100|0"], Signal.Chebwin(100, 80, sym: false), 1e-10, 1e-12, "cheb100");
        SignalAssert.Close(r["kaiser_x|51|1"], Signal.Kaiser(51, 14), 1e-13, 1e-16, "kaiser51");
    }

    [Fact]
    public void DpssMatchesScipy()
    {
        var r = SignalWindowRef.Windows;
        SignalAssert.Close(r["dpss_multi|64|1"], Signal.Dpss(64, 3.0, 5).ToArray(), 1e-9, 1e-11, "multi64");
        SignalAssert.Close(r["dpss_multi|31|0"], Signal.Dpss(31, 2.0, 3, sym: false).ToArray(), 1e-9, 1e-11, "multi31");
        var (_, ratios) = Signal.DpssWithRatios(64, 3.0, 5);
        SignalAssert.Close(r["dpss_ratios|64|1"], ratios, 1e-10, 1e-12, "ratios");
        SignalAssert.Close(r["dpss_sub|32|1"], Signal.Dpss(32, 2.5, norm: DpssNorm.Subsample), 1e-9, 1e-11, "subsample");
        SignalAssert.Close(r["dpss_l2|33|1"], Signal.Dpss(33, 2.5, norm: DpssNorm.L2), 1e-9, 1e-11, "l2");
        SignalAssert.Close(r["dpss_x|512|1"], Signal.Dpss(512, 4.0), 1e-8, 1e-10, "dpss512");
    }

    [Fact]
    public void PeriodicHannSumsToConstantAtHalfOverlap()
    {
        // The defining property of the periodic form (COLA): shifted copies sum to exactly one.
        var w = Signal.Hann(16, sym: false);
        for (var i = 0; i < 8; i++) Assert.True(Math.Abs(w[i] + w[i + 8] - 1.0) < 1e-15);
    }

    [Fact]
    public void BadInputsThrow()
    {
        Assert.Throws<ArgumentException>(() => Signal.GetWindow("nosuch", 8));
        Assert.Throws<ArgumentException>(() => Signal.GetWindow("kaiser", 8));
        Assert.Throws<ArgumentException>(() => Signal.GetWindow(("hann", 1.0), 8));
        Assert.Throws<ArgumentException>(() => Signal.KaiserBesselDerived(7, 3));
        Assert.Throws<ArgumentException>(() => Signal.Exponential(8, 2.0, sym: true));
        Assert.Throws<ArgumentException>(() => Signal.Dpss(8, 4.0));
    }
}
