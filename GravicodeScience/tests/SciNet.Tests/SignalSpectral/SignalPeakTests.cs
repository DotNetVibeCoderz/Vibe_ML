using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpectralAssert;

namespace Gravicode.Science.Tests.SciNet;

public class SignalPeakTests
{
    private static FindPeaksResult Run(string name, double[] x) => name switch
    {
        "none" => Signal.FindPeaks(x),
        "height" => Signal.FindPeaks(x, height: 1.5),
        "height_iv" => Signal.FindPeaks(x, height: (1.0, 4.0)),
        "height_arr" => Signal.FindPeaks(x, height: Enumerable.Range(0, x.Length).Select(i => 3.0 * i / (x.Length - 1)).ToArray()),
        "thr" => Signal.FindPeaks(x, threshold: 0.5),
        "thr_iv" => Signal.FindPeaks(x, threshold: (0.2, 3.0)),
        "dist" => Signal.FindPeaks(x, distance: 5),
        "dist_f" => Signal.FindPeaks(x, distance: 3.5),
        "prom" => Signal.FindPeaks(x, prominence: 1.0),
        "prom_wlen" => Signal.FindPeaks(x, prominence: 0.5, wlen: 9),
        "prom_wlenf" => Signal.FindPeaks(x, prominence: (0.5, 4.0), wlen: 6.5),
        "width" => Signal.FindPeaks(x, width: 2.0),
        "width_rel" => Signal.FindPeaks(x, width: (1.0, 6.0), relHeight: 0.8),
        "width_wlen" => Signal.FindPeaks(x, width: 1.0, wlen: 11, relHeight: 1.0),
        "plateau" => Signal.FindPeaks(x, plateauSize: 2.0),
        "plateau_iv" => Signal.FindPeaks(x, plateauSize: (1.0, 3.0)),
        "all" => Signal.FindPeaks(x, height: 0.5, threshold: 0.01, distance: 2, prominence: 0.3, width: 1.0, wlen: 21, relHeight: 0.6, plateauSize: (1.0, 5.0)),
        _ => throw new ArgumentException(name),
    };

    [Fact]
    public void FindPeaks_EveryOptionCombination_MatchesScipy()
    {
        var x = R("xp");
        foreach (var spec in SignalSpectralRef.PeakCases)
        {
            var name = spec[..spec.IndexOf(':')];
            var keys = spec[(spec.IndexOf(':') + 1)..].Split(',', StringSplitOptions.RemoveEmptyEntries).ToHashSet();
            var (peaks, props) = Run(name, x);
            Equal($"fp_{name}", peaks);
            var map = new Dictionary<string, object?>
            {
                ["peak_heights"] = props.PeakHeights, ["left_thresholds"] = props.LeftThresholds, ["right_thresholds"] = props.RightThresholds,
                ["prominences"] = props.Prominences, ["left_bases"] = props.LeftBases, ["right_bases"] = props.RightBases,
                ["width_heights"] = props.WidthHeights, ["widths"] = props.Widths, ["left_ips"] = props.LeftIps, ["right_ips"] = props.RightIps,
                ["plateau_sizes"] = props.PlateauSizes, ["left_edges"] = props.LeftEdges, ["right_edges"] = props.RightEdges,
            };
            foreach (var (k, v) in map)
            {
                if (!keys.Contains(k)) { Assert.True(v is null, $"{name}: {k} should be null"); continue; }
                Assert.True(v is not null, $"{name}: {k} missing");
                if (v is int[] iv) Equal($"fp_{name}_{k}", iv);
                else Close($"fp_{name}_{k}", (double[])v!, 1e-12);
            }
        }
    }

    [Fact]
    public void PeakProminencesAndWidths_MatchScipy()
    {
        var x = R("xp");
        var peaks = Signal.FindPeaks(x).Peaks;
        var (p, lb, rb) = Signal.PeakProminences(x, peaks);
        Close("pp_prom", p, 1e-13);
        Equal("pp_lb", lb);
        Equal("pp_rb", rb);
        (p, lb, rb) = Signal.PeakProminences(x, peaks, wlen: 7);
        Close("pp7_prom", p, 1e-13);
        Equal("pp7_lb", lb);
        Equal("pp7_rb", rb);
        foreach (var rh in new[] { 0.5, 1.0, 0.0, 0.25 })
        {
            var (w, wh, l, r) = Signal.PeakWidths(x, peaks, rh);
            var tag = rh.ToString("0.0#", System.Globalization.CultureInfo.InvariantCulture);
            Close($"pw_{tag}_w", w, 1e-12);
            Close($"pw_{tag}_wh", wh, 1e-12);
            Close($"pw_{tag}_l", l, 1e-12);
            Close($"pw_{tag}_r", r, 1e-12);
        }
    }

    [Fact]
    public void FindPeaks_LongSignalWithProminence_MatchesNaiveDefinition()
    {
        var rng = new Random(8);
        var x = Enumerable.Range(0, 100_000).Select(i => Math.Sin(i * 0.01) + 0.2 * (rng.NextDouble() - 0.5)).ToArray();
        var (peaks, props) = Signal.FindPeaks(x, prominence: 0.5);
        Assert.NotEmpty(peaks);
        foreach (var k in new[] { 0, peaks.Length / 2, peaks.Length - 1 })
        {
            var pk = peaks[k];
            double lmin = x[pk], rmin = x[pk];
            for (var i = pk; i >= 0 && x[i] <= x[pk]; i--) lmin = Math.Min(lmin, x[i]);
            for (var i = pk; i < x.Length && x[i] <= x[pk]; i++) rmin = Math.Min(rmin, x[i]);
            Assert.Equal(x[pk] - Math.Max(lmin, rmin), props.Prominences![k]);
        }
    }

    [Fact]
    public void ArgRelExtrema_MatchScipy()
    {
        var x = R("xp");
        foreach (var order in new[] { 1, 2, 3 })
            foreach (var (mode, name) in new[] { (ArgRelMode.Clip, "clip"), (ArgRelMode.Wrap, "wrap") })
            {
                Equal($"arm_{order}_{name}", Signal.ArgRelMax(x, order, mode));
                Equal($"armin_{order}_{name}", Signal.ArgRelMin(x, order, mode));
            }
        var im = Nd("im");
        var r = Signal.ArgRelExtrema(im, (a, b) => a > b, axis: 1);
        Equal("arm2d_0", r[0]);
        Equal("arm2d_1", r[1]);
        r = Signal.ArgRelExtrema(im, (a, b) => a < b, axis: 0, order: 2, mode: ArgRelMode.Wrap);
        Equal("armin2d_0", r[0]);
        Equal("armin2d_1", r[1]);
    }

    [Fact]
    public void FindPeaksCwt_AndWavelets_MatchScipy()
    {
        var x = R("xcw");
        Equal("fpcwt", Signal.FindPeaksCwt(x, Enumerable.Range(1, 19).Select(i => (double)i).ToArray()));
        Equal("fpcwt2", Signal.FindPeaksCwt(x, Enumerable.Range(0, 12).Select(i => 5.0 + 3 * i).ToArray(), minSnr: 2, noisePerc: 20));
        Equal("fpcwt3", Signal.FindPeaksCwt(x, [2.5, 4, 7.5, 12], gapThresh: 2, minLength: 2));
        Close("ricker_10_2", Signal.Ricker(10, 2.0), 1e-14);
        Close("ricker_9_1.5", Signal.Ricker(9, 1.5), 1e-14);
        Close("cwt_ricker", Signal.Cwt(x[..80], Signal.Ricker, [1, 2, 4, 9]), 1e-12);
        Close("morlet2_12_2", Signal.Morlet2(12, 2.0), 1e-14);
        Close("cwt_morlet2", Signal.CwtComplex(x[..80], (m, s) => Signal.Morlet2(m, s), [1, 2, 4, 9]), 1e-12);
    }

    [Fact]
    public void FindPeaks_RejectsBadArguments()
    {
        Assert.Throws<ArgumentException>(() => Signal.FindPeaks([0, 1, 0], distance: 0.5));
        Assert.Throws<ArgumentException>(() => Signal.FindPeaks([0, 1, 0], prominence: 0.1, wlen: 1));
        Assert.Throws<ArgumentException>(() => Signal.PeakWidths([0, 1, 0], [1], -0.1));
    }
}
