using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class SignalIirTests
{
    private static readonly Dictionary<string, double[]> R = SignalDesignRef.D;

    // Each case: the zpk designer and whether it is analog (which picks the sos pairing rule).
    private static readonly (string Key, Func<FilterZpk> Zpk, bool Analog)[] Cases =
    [
        ("butter4_lp", () => Signal.ButterZpk(4, 0.3), false),
        ("butter5_hp", () => Signal.ButterZpk(5, 0.25, FilterType.Highpass), false),
        ("butter3_bp", () => Signal.ButterZpk(3, (0.2, 0.4), FilterType.Bandpass), false),
        ("butter4_bs", () => Signal.ButterZpk(4, (100, 200), FilterType.Bandstop, fs: 1000), false),
        ("butter8_lp", () => Signal.ButterZpk(8, 0.1), false),
        ("butter4_analog", () => Signal.ButterZpk(4, 10.0, analog: true), true),
        ("butter3_analog_bp", () => Signal.ButterZpk(3, (2.0, 5.0), FilterType.Bandpass, analog: true), true),
        ("cheby1_4_lp", () => Signal.Cheby1Zpk(4, 1.0, 0.3), false),
        ("cheby1_5_hp", () => Signal.Cheby1Zpk(5, 0.5, 0.4, FilterType.Highpass), false),
        ("cheby1_3_bp", () => Signal.Cheby1Zpk(3, 2.0, (0.1, 0.3), FilterType.Bandpass), false),
        ("cheby1_4_analog", () => Signal.Cheby1Zpk(4, 3.0, 1.0, analog: true), true),
        ("cheby2_4_lp", () => Signal.Cheby2Zpk(4, 40, 0.3), false),
        ("cheby2_5_hp", () => Signal.Cheby2Zpk(5, 50, 0.4, FilterType.Highpass), false),
        ("cheby2_4_bs", () => Signal.Cheby2Zpk(4, 30, (0.2, 0.5), FilterType.Bandstop), false),
        ("cheby2_3_analog", () => Signal.Cheby2Zpk(3, 40, 2.0, analog: true), true),
        ("ellip4_lp", () => Signal.EllipZpk(4, 1, 40, 0.3), false),
        ("ellip5_hp", () => Signal.EllipZpk(5, 0.5, 60, 0.25, FilterType.Highpass), false),
        ("ellip3_bp", () => Signal.EllipZpk(3, 1, 50, (0.2, 0.35), FilterType.Bandpass), false),
        ("ellip6_bs", () => Signal.EllipZpk(6, 0.1, 70, (1000, 2000), FilterType.Bandstop, fs: 8000), false),
        ("ellip1_lp", () => Signal.EllipZpk(1, 1, 30, 0.3), false),
        ("ellip2_analog", () => Signal.EllipZpk(2, 1, 30, 1.0, analog: true), true),
        ("ellip7_lp", () => Signal.EllipZpk(7, 0.2, 80, 0.15), false),
        ("bessel4_lp", () => Signal.BesselZpk(4, 0.3), false),
        ("bessel5_hp_delay", () => Signal.BesselZpk(5, 0.3, FilterType.Highpass, norm: BesselNorm.Delay), false),
        ("bessel3_mag", () => Signal.BesselZpk(3, 0.2, norm: BesselNorm.Mag), false),
        ("bessel6_analog", () => Signal.BesselZpk(6, 1.0, analog: true), true),
        ("bessel4_bp", () => Signal.BesselZpk(4, (0.2, 0.4), FilterType.Bandpass), false),
        ("bessel10_mag_analog", () => Signal.BesselZpk(10, 1.0, analog: true, norm: BesselNorm.Mag), true),
    ];

    [Fact]
    public void ZpkDesignsMatchScipy()
    {
        foreach (var (key, make, _) in Cases)
        {
            var zpk = make();
            SignalAssert.SameRoots(R[key + "_z"], zpk.Z, 1e-10, key + " zeros");
            SignalAssert.SameRoots(R[key + "_p"], zpk.P, 1e-10, key + " poles");
            SignalAssert.Close(R[key + "_k"][0], zpk.K, 1e-10, 0, key + " gain");
        }
    }

    [Fact]
    public void BaDesignsMatchScipy()
    {
        foreach (var (key, make, _) in Cases)
        {
            var (b, a) = Signal.Zpk2Tf(make());
            SignalAssert.Close(R[key + "_b"], b, 1e-9, 1e-11 * R[key + "_b"].Max(Math.Abs), key + " b");
            SignalAssert.Close(R[key + "_a"], a, 1e-9, 1e-11 * R[key + "_a"].Max(Math.Abs), key + " a");
        }
        var (bb, ba) = Signal.Butter(4, 0.3);
        SignalAssert.Close(R["butter4_lp_b"], bb, 1e-10, 1e-14, "butter ba");
        SignalAssert.Close(R["butter4_lp_a"], ba, 1e-10, 1e-14, "butter ba");
        SignalAssert.Close(R["ellip4_lp_b"], Signal.Ellip(4, 1, 40, 0.3).B, 1e-10, 1e-14, "ellip ba");
        var res = Signal.Iirfilter(4, 0.3, rp: 1, rs: 40, btype: FilterType.Lowpass, ftype: IirDesign.Ellip);
        SignalAssert.Close(R["ellip4_lp_a"], res.Ba!.A, 1e-10, 1e-14, "iirfilter ba");
    }

    [Fact]
    public void SosDesignsMatchScipyIncludingPairing()
    {
        foreach (var (key, make, analog) in Cases)
        {
            var sos = Signal.Zpk2Sos(make(), analog: analog);
            var exp = R[key + "_sos"];
            Assert.True(exp.Length == sos.Size, $"{key}: {sos.Size / 6} sections vs {exp.Length / 6}");
            SignalAssert.Close(exp, sos.ToArray(), 1e-9, 1e-11 * exp.Max(Math.Abs), key + " sos");
        }
        SignalAssert.Close(R["ellip7_lp_sos"], Signal.EllipSos(7, 0.2, 80, 0.15).ToArray(), 1e-9, 1e-12, "EllipSos");
    }

    [Fact]
    public void PrototypesMatchScipy()
    {
        foreach (var n in new[] { 1, 2, 5, 8 })
        {
            var e = Signal.Ellipap(n, 0.5, 45);
            SignalAssert.SameRoots(R[$"ellipap{n}_z"], e.Z, 1e-11, $"ellipap{n} z");
            SignalAssert.SameRoots(R[$"ellipap{n}_p"], e.P, 1e-11, $"ellipap{n} p");
            SignalAssert.Close(R[$"ellipap{n}_k"][0], e.K, 1e-11, 0, $"ellipap{n} k");
            var b = Signal.Besselap(n);
            SignalAssert.SameRoots(R[$"besselap{n}_p"], b.P, 1e-12, $"besselap{n} p");
        }
        foreach (var n in new[] { 12, 20 })
        {
            var e = Signal.Ellipap(n, 0.1, 90);
            SignalAssert.SameRoots(R[$"ellipap{n}_z"], e.Z, 1e-9, $"ellipap{n} z");
            SignalAssert.SameRoots(R[$"ellipap{n}_p"], e.P, 1e-9, $"ellipap{n} p");
            SignalAssert.Close(R[$"ellipap{n}_k"][0], e.K, 1e-8, 0, $"ellipap{n} k");
        }
        // High order Bessel: the recurrence-evaluated Aberth iteration holds up at N = 25.
        var b25 = Signal.BesselZpk(25, 1.0, analog: true);
        SignalAssert.SameRoots(R["bessel25_analog_p"], b25.P, 1e-10, "bessel25");
    }

    [Theory]
    [InlineData("buttord_lp")]
    [InlineData("buttord_hp")]
    [InlineData("buttord_bp")]
    [InlineData("buttord_bs")]
    [InlineData("buttord_analog")]
    [InlineData("buttord_fs")]
    [InlineData("cheb1ord_lp")]
    [InlineData("cheb1ord_bp")]
    [InlineData("cheb1ord_bs")]
    [InlineData("cheb2ord_lp")]
    [InlineData("cheb2ord_hp")]
    [InlineData("cheb2ord_bp")]
    [InlineData("cheb2ord_bs")]
    [InlineData("cheb2ord_analog")]
    [InlineData("ellipord_lp")]
    [InlineData("ellipord_hp")]
    [InlineData("ellipord_bp")]
    [InlineData("ellipord_bs")]
    [InlineData("ellipord_fs")]
    public void OrderSelectionMatchesScipy(string key)
    {
        var (n, wn) = key switch
        {
            "buttord_lp" => Signal.Buttord(0.2, 0.3, 3, 40),
            "buttord_hp" => Signal.Buttord(0.3, 0.2, 1, 60),
            "buttord_bp" => Signal.Buttord((0.2, 0.5), (0.1, 0.6), 3, 40),
            "buttord_bs" => Signal.Buttord((0.1, 0.6), (0.2, 0.5), 3, 40),
            "buttord_analog" => Signal.Buttord(10, 30, 3, 40, analog: true),
            "buttord_fs" => Signal.Buttord(100, 150, 1, 50, fs: 1000),
            "cheb1ord_lp" => Signal.Cheb1ord(0.2, 0.3, 3, 40),
            "cheb1ord_bp" => Signal.Cheb1ord((0.2, 0.5), (0.1, 0.6), 1, 60),
            "cheb1ord_bs" => Signal.Cheb1ord((0.1, 0.6), (0.2, 0.5), 3, 40),
            "cheb2ord_lp" => Signal.Cheb2ord(0.2, 0.3, 3, 40),
            "cheb2ord_hp" => Signal.Cheb2ord(0.3, 0.2, 3, 40),
            "cheb2ord_bp" => Signal.Cheb2ord((0.2, 0.5), (0.1, 0.6), 3, 40),
            "cheb2ord_bs" => Signal.Cheb2ord((0.1, 0.6), (0.2, 0.5), 3, 40),
            "cheb2ord_analog" => Signal.Cheb2ord(1, 3, 1, 50, analog: true),
            "ellipord_lp" => Signal.Ellipord(0.2, 0.3, 3, 40),
            "ellipord_hp" => Signal.Ellipord(0.3, 0.2, 0.5, 70),
            "ellipord_bp" => Signal.Ellipord((0.2, 0.5), (0.1, 0.6), 3, 40),
            "ellipord_bs" => Signal.Ellipord((0.1, 0.6), (0.2, 0.5), 3, 40),
            "ellipord_fs" => Signal.Ellipord((1000, 2000), (800, 2400), 1, 60, fs: 8000),
            _ => throw new ArgumentException(key),
        };
        var exp = R[key];
        Assert.Equal((int)exp[0], n);
        // Band-stop edges come from a bounded Brent search (xtol 1e-5) ported step for step.
        SignalAssert.Close(exp[1..], wn, 1e-10, 1e-12, key);
    }

    [Fact]
    public void IirdesignNotchPeakCombGammatoneMatchScipy()
    {
        var d = Signal.Iirdesign((0.2, 0.5), (0.1, 0.6), 1, 40, ftype: IirDesign.Cheby2).Ba!;
        SignalAssert.Close(R["iirdesign_b"], d.B, 1e-9, 1e-12, "iirdesign b");
        SignalAssert.Close(R["iirdesign_a"], d.A, 1e-9, 1e-12, "iirdesign a");
        var sos = Signal.Iirdesign(0.3, 0.4, 0.5, 60, ftype: IirDesign.Ellip, output: FilterOutput.Sos).Sos!;
        SignalAssert.Close(R["iirdesign_sos"], sos.ToArray(), 1e-9, 1e-12, "iirdesign sos");
        var n = Signal.Iirnotch(60, 30, fs: 1000);
        SignalAssert.Close(R["notch_b"], n.B, 1e-14, 1e-15, "notch b");
        SignalAssert.Close(R["notch_a"], n.A, 1e-14, 1e-15, "notch a");
        var p = Signal.Iirpeak(0.3, 10);
        SignalAssert.Close(R["peak_b"], p.B, 1e-14, 1e-15, "peak b");
        SignalAssert.Close(R["peak_a"], p.A, 1e-14, 1e-15, "peak a");
        var c = Signal.Iircomb(100, 30, fs: 1000);
        SignalAssert.Close(R["comb_b"], c.B, 1e-14, 1e-15, "comb b");
        SignalAssert.Close(R["comb_a"], c.A, 1e-14, 1e-15, "comb a");
        var cp = Signal.Iircomb(250, 10, peak: true, fs: 1000, passZero: true);
        SignalAssert.Close(R["combp_b"], cp.B, 1e-14, 1e-15, "combp b");
        SignalAssert.Close(R["combp_a"], cp.A, 1e-14, 1e-15, "combp a");
        SignalAssert.Close(R["gt_fir_b"], Signal.Gammatone(440, fir: true, fs: 16000).B, 1e-12, 1e-16, "gammatone fir");
        var gi = Signal.Gammatone(440, fir: false, fs: 16000);
        SignalAssert.Close(R["gt_iir_b"], gi.B, 1e-10, 1e-20, "gammatone iir b");
        SignalAssert.Close(R["gt_iir_a"], gi.A, 1e-12, 1e-14, "gammatone iir a");
    }

    [Fact]
    public void ButterworthIsMinus3DbAtCutoff()
    {
        // Closed form: |H(e^{jπWn})| = 1/√2 for every order.
        foreach (var order in new[] { 1, 2, 5, 9 })
        {
            var sos = Signal.ButterSos(order, 0.37);
            var (_, h) = Signal.Sosfreqz(sos, [0.37], fs: 2.0);
            SignalAssert.Close(1 / Math.Sqrt(2), Complex.Abs(h[0]), 1e-12, 0, $"order {order}");
        }
    }

    [Fact]
    public void InvalidDesignsThrow()
    {
        Assert.Throws<ArgumentException>(() => Signal.Butter(4, 1.2));
        Assert.Throws<ArgumentException>(() => Signal.Butter(4, (0.4, 0.2), FilterType.Bandpass));
        Assert.Throws<ArgumentException>(() => Signal.Butter(4, 0.2, FilterType.Bandpass));
        Assert.Throws<ArgumentException>(() => Signal.Butter(4, 10, analog: true, fs: 100));
        Assert.Throws<ArgumentException>(() => Signal.Iirfilter(4, 0.3, btype: FilterType.Lowpass, ftype: IirDesign.Cheby1));
        Assert.Throws<ArgumentException>(() => Signal.Buttord(0.2, 0.3, 50, 40));
    }

    [Fact]
    public void CriticalFrequencies_AcceptACollectionExpression()
    {
        // [5.0, 50.0] is how a band is written in modern C#; it must mean the same as an array.
        var fromCollection = Signal.Butter(4, [5.0, 50.0], FilterType.Bandpass, fs: 1000);
        var fromArray = Signal.Butter(4, new[] { 5.0, 50.0 }, FilterType.Bandpass, fs: 1000);
        Assert.Equal(fromArray.B, fromCollection.B);
        Assert.Equal(fromArray.A, fromCollection.A);
    }
}
