using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class SignalLtiTests
{
    private static readonly Dictionary<string, double[]> R = SignalLtiRef.D;
    private static readonly double[] Num = [1.0, 3.0];
    private static readonly double[] Den = [1.0, 2.0, 5.0, 4.0];

    private static StateSpace Sys3() => new(
        new NdArray([0, 1.0, 0, 0, 0, 1, -4, -5, -2], 3, 3),
        new NdArray([0.0, 0, 1], 3, 1),
        new NdArray([1.0, 0.5, 0], 1, 3),
        new NdArray([0.1], 1, 1));

    [Fact]
    public void RepresentationConversionsMatchScipy()
    {
        var (a, b, c, d) = Signal.Tf2Ss(Num, Den);
        SignalAssert.Close(R["tf2ss_A"], a.ToArray(), 1e-15, 0, "A");
        SignalAssert.Close(R["tf2ss_B"], b.ToArray(), 1e-15, 0, "B");
        SignalAssert.Close(R["tf2ss_C"], c.ToArray(), 1e-15, 0, "C");
        SignalAssert.Close(R["tf2ss_D"], d.ToArray(), 1e-15, 0, "D");
        var (a2, _, c2, d2) = Signal.Tf2Ss([2, 1, 3], [1, 4, 6]);
        SignalAssert.Close(R["tf2ss2_A"], a2.ToArray(), 1e-15, 0, "A2");
        SignalAssert.Close(R["tf2ss2_C"], c2.ToArray(), 1e-15, 0, "C2");
        SignalAssert.Close(R["tf2ss2_D"], d2.ToArray(), 1e-15, 0, "D2");

        var am = new NdArray([0, 1.0, -2, -3], 2, 2);
        var bm = new NdArray([0.0, 1], 2, 1);
        var (num, den) = Signal.Ss2Tf(am, bm, new NdArray([1.0, 0, 0.5, 1], 2, 2), new NdArray([0.0, 0.2], 2, 1));
        SignalAssert.Close(R["ss2tf_num"], num.ToArray(), 1e-13, 1e-14, "ss2tf num");
        SignalAssert.Close(R["ss2tf_den"], den, 1e-13, 1e-14, "ss2tf den");
        var zpk = Signal.Ss2Zpk(am, bm, new NdArray([1.0, 0], 1, 2), new NdArray([0.0], 1, 1));
        Assert.Empty(zpk.Z);
        SignalAssert.SameRoots(R["ss2zpk_p"], zpk.P, 1e-13, "ss2zpk p");
        SignalAssert.Close(R["ss2zpk_k"][0], zpk.K, 1e-13, 0, "ss2zpk k");

        // Round trip tf -> ss -> tf is the identity.
        var back = new TransferFunction(Num, Den).ToSs().ToTf();
        SignalAssert.Close(Num, back.Num, 1e-13, 1e-13, "roundtrip num");
        SignalAssert.Close(Den, back.Den, 1e-13, 1e-13, "roundtrip den");
    }

    [Theory]
    [InlineData("zoh", DiscretizationMethod.Zoh)]
    [InlineData("foh", DiscretizationMethod.Foh)]
    [InlineData("bilinear", DiscretizationMethod.Bilinear)]
    [InlineData("euler", DiscretizationMethod.Euler)]
    [InlineData("backward_diff", DiscretizationMethod.BackwardDiff)]
    [InlineData("gbt", DiscretizationMethod.Gbt)]
    [InlineData("impulse", DiscretizationMethod.Impulse)]
    public void Cont2DiscreteMatchesScipy(string key, DiscretizationMethod method)
    {
        var sys = Sys3();
        if (method == DiscretizationMethod.Impulse) sys = new StateSpace(sys.A, sys.B, sys.C, NdArray.Zeros(1, 1));
        var sd = Signal.Cont2Discrete(sys, 0.1, method, alpha: 0.3);
        Assert.Equal(0.1, sd.Dt);
        SignalAssert.Close(R[$"c2d_{key}_A"], sd.A.ToArray(), 1e-12, 1e-14, key + " A");
        SignalAssert.Close(R[$"c2d_{key}_B"], sd.B.ToArray(), 1e-12, 1e-14, key + " B");
        SignalAssert.Close(R[$"c2d_{key}_C"], sd.C.ToArray(), 1e-12, 1e-14, key + " C");
        SignalAssert.Close(R[$"c2d_{key}_D"], sd.D.ToArray(), 1e-12, 1e-14, key + " D");
    }

    [Fact]
    public void TransferFunctionDiscretisationMatchesScipy()
    {
        var tf = (TransferFunction)Signal.Cont2Discrete(new TransferFunction(Num, Den), 0.2);
        // cont2discrete returns ss2tf's raw numerator with an exact leading zero; TransferFunction normalises it away.
        SignalAssert.Close(R["c2d_tf_num"][1..], tf.Num, 1e-10, 1e-13, "num");
        SignalAssert.Close(R["c2d_tf_den"], tf.Den, 1e-12, 1e-13, "den");
        var bl = (TransferFunction)new TransferFunction([1.0], [1.0, 1.0]).ToDiscrete(0.5, DiscretizationMethod.Bilinear);
        SignalAssert.Close(R["todisc_num"], bl.Num, 1e-13, 1e-14, "bilinear num");
        SignalAssert.Close(R["todisc_den"], bl.Den, 1e-13, 1e-14, "bilinear den");
        // gbt with alpha = 0.5 is Tustin, which is exactly Signal.Bilinear on the transfer function.
        var (bb, ba) = Signal.Bilinear([1.0], [1.0, 1.0], fs: 2.0);
        SignalAssert.Close(bb, bl.Num, 1e-13, 1e-14, "vs Bilinear");
        SignalAssert.Close(ba, bl.Den, 1e-13, 1e-14, "vs Bilinear");
    }

    [Fact]
    public void LsimMatchesScipy()
    {
        var sys = new TransferFunction(Num, Den);
        var t = Enumerable.Range(0, 51).Select(i => i * 0.1).ToArray();
        var u = t.Select(Math.Sin).ToArray();
        var r = Signal.Lsim(sys, u, t);
        SignalAssert.Close(R["lsim_y"], r.Y, 1e-10, 1e-12, "lsim");
        SignalAssert.Close(R["lsim_x"], r.Xout.ToArray(), 1e-10, 1e-12, "lsim x");
        SignalAssert.Close(R["lsim_zoh_y"], Signal.Lsim(sys, u, t, interp: false).Y, 1e-10, 1e-12, "zoh");
        SignalAssert.Close(R["lsim_x0_y"], Signal.Lsim(sys, u, t, [1.0, -0.5, 0.2]).Y, 1e-10, 1e-12, "x0");
        SignalAssert.Close(R["lsim_free_y"], Signal.Lsim(sys, (double[]?)null, t, [1.0, -0.5, 0.2]).Y, 1e-10, 1e-12, "free");
        var t2 = Enumerable.Range(0, 26).Select(i => 0.5 + i * 0.1).ToArray();
        SignalAssert.Close(R["lsim_t0_y"], Signal.Lsim(sys, t2.Select(Math.Cos).ToArray(), t2).Y, 1e-10, 1e-12, "t0 > 0");
    }

    [Fact]
    public void ImpulseStepBodeMatchScipy()
    {
        var sys = new TransferFunction(Num, Den);
        var (ti, yi) = Signal.Impulse(sys);
        SignalAssert.Close(R["impulse_t"], ti, 1e-12, 1e-14, "impulse t");
        SignalAssert.Close(R["impulse_y"], yi, 1e-10, 1e-12, "impulse y");
        var (ts, ys) = Signal.Step(sys);
        SignalAssert.Close(R["step_t"], ts, 1e-12, 1e-14, "step t");
        SignalAssert.Close(R["step_y"], ys, 1e-10, 1e-12, "step y");
        var t = Enumerable.Range(0, 51).Select(i => i * 0.1).ToArray();
        SignalAssert.Close(R["step_tt_y"], Signal.Step(sys, t: t).Y, 1e-10, 1e-12, "step given t");
        var (w, mag, ph) = Signal.Bode(sys);
        SignalAssert.Close(R["bode_w"], w, 1e-12, 0, "bode w");
        SignalAssert.Close(R["bode_mag"], mag, 1e-10, 1e-10, "bode mag");
        SignalAssert.Close(R["bode_ph"], ph, 1e-10, 1e-9, "bode phase");
        var (_, m2, p2) = Signal.Bode(new TransferFunction([1], [1, 0.2, 1]), [0.1, 0.5, 1, 2, 5]);
        SignalAssert.Close(R["bode2_mag"], m2, 1e-11, 1e-11, "bode2 mag");
        SignalAssert.Close(R["bode2_ph"], p2, 1e-11, 1e-10, "bode2 phase");
        var (fw, fh) = Signal.Freqresp(new ZerosPolesGain([-1], [-2, new Complex(-3, 1), new Complex(-3, -1)], 5.0), n: 30);
        SignalAssert.Close(R["freqresp_zpk_w"], fw, 1e-12, 0, "freqresp w");
        SignalAssert.Close(R["freqresp_zpk_h"], fh, 1e-11, 1e-13, "freqresp h");
        var (_, sh) = Signal.Freqresp(Sys3(), [0.1, 1, 10]);
        SignalAssert.Close(R["freqresp_ss_h"], sh, 1e-10, 1e-12, "freqresp ss");
    }

    [Fact]
    public void StepOfFirstOrderLagIsClosedForm()
    {
        // H = 1/(s+2): y(t) = (1 - e^{-2t})/2.
        var t = Enumerable.Range(0, 40).Select(i => i * 0.05).ToArray();
        var (_, y) = Signal.Step(new TransferFunction([1], [1, 2]), t: t);
        for (var i = 0; i < t.Length; i++) SignalAssert.Close((1 - Math.Exp(-2 * t[i])) / 2, y[i], 1e-12, 1e-14, $"t={t[i]}");
    }

    [Fact]
    public void DiscreteResponsesMatchScipy()
    {
        var dsys = new TransferFunction([0.5, 0.2], [1, -0.6, 0.08], dt: 0.1);
        var ud = Enumerable.Range(0, 20).Select(i => Math.Cos(i * 0.3)).ToArray();
        var r = Signal.Dlsim(dsys, ud);
        SignalAssert.Close(R["dlsim_t"], r.T, 1e-14, 1e-15, "t");
        SignalAssert.Close(R["dlsim_y"], r.Y, 1e-12, 1e-14, "y");
        SignalAssert.Close(R["dlsim_x0_y"], Signal.Dlsim(dsys, ud, x0: [1.0, 0.5]).Y, 1e-12, 1e-14, "x0");
        var tt = Enumerable.Range(0, 7).Select(i => 1.9 * i / 6).ToArray();
        var rt = Signal.Dlsim(dsys, tt.Select(Math.Sin).ToArray(), tt);
        SignalAssert.Close(R["dlsim_tt_t"], rt.T, 1e-14, 1e-15, "tt t");
        SignalAssert.Close(R["dlsim_tt_y"], rt.Y, 1e-12, 1e-14, "tt y");
        var (ti, yi) = Signal.Dimpulse(dsys, n: 15);
        SignalAssert.Close(R["dimpulse_t"], ti, 1e-14, 1e-15, "dimpulse t");
        SignalAssert.Close(R["dimpulse_y"], yi[0].ToArray(), 1e-12, 1e-14, "dimpulse y");
        SignalAssert.Close(R["dstep_y"], Signal.Dstep(dsys, n: 15).Y[0].ToArray(), 1e-12, 1e-14, "dstep");
        // The impulse response of b/a is lfilter of a unit impulse.
        var imp = new double[15]; imp[0] = 1;
        SignalAssert.Close(Signal.Lfilter([0, 0.5, 0.2], [1, -0.6, 0.08], imp), yi[0].ToArray(), 1e-12, 1e-14, "vs lfilter");
        var (w, h) = Signal.Dfreqresp(dsys, n: 16);
        SignalAssert.Close(R["dfreq_w"], w, 1e-14, 1e-15, "dfreq w");
        SignalAssert.Close(R["dfreq_h"], h, 1e-12, 1e-14, "dfreq h");
        var (bw, bm, bp) = Signal.Dbode(dsys, n: 16);
        SignalAssert.Close(R["dbode_w"], bw, 1e-13, 1e-14, "dbode w");
        SignalAssert.Close(R["dbode_mag"], bm, 1e-11, 1e-12, "dbode mag");
        SignalAssert.Close(R["dbode_ph"], bp, 1e-11, 1e-10, "dbode phase");
        var zsys = new ZerosPolesGain([0.5], [0.9, new Complex(0.3, 0.2), new Complex(0.3, -0.2)], 0.7, dt: 0.5);
        SignalAssert.Close(R["dfreq_zpk_h"], Signal.Dfreqresp(zsys, [0.1, 0.7, 2.0]).H, 1e-12, 1e-14, "dfreq zpk");
    }

    [Fact]
    public void InvalidUseThrows()
    {
        var c = new TransferFunction([1], [1, 1]);
        var d = new TransferFunction([1], [1, 0.5], dt: 1);
        Assert.Throws<ArgumentException>(() => Signal.Dlsim(c, [1.0, 2.0]));
        Assert.Throws<ArgumentException>(() => Signal.Lsim(d, [1.0, 2.0], [0, 1.0]));
        Assert.Throws<ArgumentException>(() => Signal.Lsim(c, [1.0, 2.0, 3.0], [0, 1.0, 3.0]));
        Assert.Throws<ArgumentException>(() => Signal.Cont2Discrete(Sys3(), 0.1, DiscretizationMethod.Gbt));
        Assert.Throws<ArgumentException>(() => Signal.Cont2Discrete(Sys3(), 0.1, DiscretizationMethod.Impulse));
    }
}
