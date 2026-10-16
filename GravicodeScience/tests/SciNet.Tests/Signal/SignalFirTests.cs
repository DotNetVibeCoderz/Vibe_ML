using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class SignalFirTests
{
    private static readonly Dictionary<string, double[]> R = SignalDesignRef.D;

    [Fact]
    public void FirwinMatchesScipy()
    {
        SignalAssert.Close(R["firwin_lp"], Signal.Firwin(31, 0.3), 1e-12, 1e-15, "lp");
        SignalAssert.Close(R["firwin_hp"], Signal.Firwin(31, 0.3, passZero: false), 1e-12, 1e-15, "hp");
        SignalAssert.Close(R["firwin_bp"], Signal.Firwin(41, new[] { 0.2, 0.5 }, passZero: false, window: "blackman"), 1e-12, 1e-15, "bp");
        SignalAssert.Close(R["firwin_bs"], Signal.Firwin(41, (0.2, 0.5), window: ("kaiser", 6.0)), 1e-12, 1e-15, "bs");
        SignalAssert.Close(R["firwin_multi"], Signal.Firwin(51, new[] { 0.1, 0.3, 0.5, 0.7 }, passZero: false), 1e-12, 1e-15, "multi");
        SignalAssert.Close(R["firwin_width"], Signal.Firwin(45, 100, width: 40, fs: 1000), 1e-12, 1e-15, "width");
        SignalAssert.Close(R["firwin_noscale"], Signal.Firwin(20, 0.4, scale: false, window: "hann"), 1e-12, 1e-15, "noscale");
        SignalAssert.Close(R["firwin_bp"], Signal.Firwin(41, new[] { 0.2, 0.5 }, FilterType.Bandpass, window: "blackman"), 1e-12, 1e-15, "bp typed");
        Assert.Throws<ArgumentException>(() => Signal.Firwin(30, 0.3, passZero: false)); // even taps cannot pass Nyquist
        Assert.Throws<ArgumentException>(() => Signal.Firwin(31, new[] { 0.2, 0.5 }, FilterType.Lowpass));
    }

    [Fact]
    public void Firwin2MatchesScipy()
    {
        SignalAssert.Close(R["firwin2_lp"], Signal.Firwin2(31, [0, 0.3, 0.5, 1], [1, 1, 0, 0]), 1e-10, 1e-14, "lp");
        SignalAssert.Close(R["firwin2_step"], Signal.Firwin2(32, [0, 0.5, 0.5, 1], [1, 1, 0, 0], window: ("kaiser", 5)), 1e-10, 1e-14, "step");
        SignalAssert.Close(R["firwin2_anti3"], Signal.Firwin2(31, [0, 0.2, 0.8, 1], [0, 1, 1, 0], antisymmetric: true), 1e-10, 1e-14, "anti3");
        SignalAssert.Close(R["firwin2_anti4"], Signal.Firwin2(30, [0, 0.5, 1], [0, 0.5, 1], antisymmetric: true, fs: 2), 1e-10, 1e-14, "anti4");
        SignalAssert.Close(R["firwin2_fs"], Signal.Firwin2(25, [0, 100, 200, 500], [1, 1, 0, 0], nfreqs: 129, fs: 1000, noWindow: true), 1e-10, 1e-14, "fs");
    }

    [Fact]
    public void FirlsMatchesScipy()
    {
        SignalAssert.Close(R["firls_lp"], Signal.Firls(31, [0, 0.3, 0.4, 1], [1, 1, 0, 0]), 1e-9, 1e-12, "lp");
        SignalAssert.Close(R["firls_w"], Signal.Firls(41, [0, 100, 150, 250, 300, 500], [0, 0, 1, 1, 0, 0], [2, 1, 2], fs: 1000), 1e-9, 1e-12, "weighted");
        SignalAssert.Close(R["firls_slope"], Signal.Firls(21, [0, 0.5, 0.6, 1], [0, 1, 0.5, 0.2]), 1e-9, 1e-12, "slope");
    }

    [Fact]
    public void RemezMatchesScipy()
    {
        SignalAssert.Close(R["remez_lp"], Signal.Remez(31, [0, 0.2, 0.25, 0.5], [1, 0]), 1e-9, 1e-12, "lp");
        SignalAssert.Close(R["remez_lp_even"], Signal.Remez(32, [0, 0.2, 0.25, 0.5], [1, 0]), 1e-9, 1e-12, "lp even");
        SignalAssert.Close(R["remez_bp"], Signal.Remez(41, [0, 100, 150, 250, 300, 500], [0, 1, 0], [1, 2, 1], fs: 1000), 1e-9, 1e-12, "bp");
        SignalAssert.Close(R["remez_hilbert_even"], Signal.Remez(30, [0.05, 0.5], [1], type: RemezType.Hilbert), 1e-9, 1e-12, "hilbert even");
        SignalAssert.Close(R["remez_diff"], Signal.Remez(31, [0, 0.4], [1], type: RemezType.Differentiator), 1e-9, 1e-12, "diff");
        SignalAssert.Close(R["remez_diff_even"], Signal.Remez(30, [0.0, 0.45], [1], type: RemezType.Differentiator), 1e-9, 1e-12, "diff even");
        SignalAssert.Close(R["remez_hilbert"], Signal.Remez(31, [0.05, 0.45], [1], type: RemezType.Hilbert), 1e-9, 1e-12, "hilbert");
        SignalAssert.Close(R["remez_hp"], Signal.Remez(33, [0, 0.15, 0.2, 0.5], [0, 1], gridDensity: 20), 1e-9, 1e-12, "hp");
    }

    [Fact]
    public void MinimumPhaseMatchesScipy()
    {
        var h = Signal.Remez(41, [0, 0.2, 0.3, 0.5], [1, 0]);
        SignalAssert.Close(R["minphase_h"], Signal.MinimumPhase(h), 1e-8, 1e-11, "homomorphic");
        SignalAssert.Close(R["minphase_hfull"], Signal.MinimumPhase(h, half: false), 1e-8, 1e-11, "homomorphic full");
        // The Hilbert method takes sqrt of a spectrum shifted to touch zero, so last-bit FFT
        // differences near that zero are amplified to ~1e-4; scipy's own result is that sensitive.
        SignalAssert.Close(R["minphase_hilb"], Signal.MinimumPhase(h, MinimumPhaseMethod.Hilbert), 0, 1e-3, "hilbert");
    }

    [Fact]
    public void KaiserFormulas()
    {
        var (n, beta) = Signal.Kaiserord(65, 0.05);
        Assert.Equal((int)R["kaiserord"][0], n);
        SignalAssert.Close(R["kaiserord"][1], beta, 1e-14, 0, "beta");
        // KaiserAtten inverts kaiserord's length formula.
        SignalAssert.Close(65, Signal.KaiserAtten(1 + (int)Math.Round((65 - 7.95) / 2.285 / (Math.PI * 0.05)), 0.05), 2e-3, 0, "atten");
        Assert.Equal(0.0, Signal.KaiserBeta(20));
    }
}
