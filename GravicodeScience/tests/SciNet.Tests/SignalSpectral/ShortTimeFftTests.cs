using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>ShortTimeFft against scipy 1.17.1's ShortTimeFFT (values printed by scipy, hardcoded).</summary>
public class ShortTimeFftTests
{
    private const int N = 50;

    private static double[] X()
    {
        var x = new double[N];
        for (var k = 0; k < N; k++) x[k] = Math.Sin(0.37 * k) + 0.02 * k * k / 50 - 0.3 * Math.Cos(1.3 * k);
        return x;
    }

    private static void Close(double e, double a, double tol, string what)
        => Assert.True(Math.Abs(e - a) <= tol * Math.Max(1, Math.Abs(e)), $"{what}: expected {e:R}, got {a:R}");

    private static void Close(Complex e, Complex a, double tol, string what)
        => Assert.True((e - a).Magnitude <= tol * Math.Max(1, e.Magnitude), $"{what}: expected {e}, got {a}");

    private static double AbsSum(ComplexNdArray s) { double t = 0; foreach (var v in s.Data) t += v.Magnitude; return t; }

    private static (ShortTimeFft Sft, string Name) Config(string name) => name switch
    {
        "A" => (ShortTimeFft.FromWindow("hann", 10, 16, 12, scaleTo: ShortTimeFftScaling.Magnitude), name),
        "B" => (ShortTimeFft.FromWindow(new WindowSpec("kaiser", 6.0), 2, 15, 10, fftMode: ShortTimeFftMode.Twosided, mfft: 20, phaseShift: null), name),
        "C" => (ShortTimeFft.FromWindow("hamming", 1, 12, 4, symmetricWin: true, fftMode: ShortTimeFftMode.Centered, phaseShift: 3), name),
        "D" => (ShortTimeFft.FromWindow("hann", 8, 16, 8, fftMode: ShortTimeFftMode.Onesided2X, scaleTo: ShortTimeFftScaling.Psd, mfft: 21), name),
        _ => (ShortTimeFft.FromWindow("hann", 1, 16, 12), name),
    };

    public static TheoryData<string> Configs => ["A", "B", "C", "D"];

    [Theory]
    [MemberData(nameof(Configs))]
    public void Stft_MatchesScipyAndInverts(string name)
    {
        // scipy 1.17.1: k_min, p_min, k_max(50), p_max(50), lower_border_end, upper_border_begin(50), f_pts,
        // slices, f[1], t(50)[0], delta_t; then sum|S|, S[1,2], S[-1,-1], S[f/2, p/2]; dual_win[3].
        var e = name switch
        {
            "A" => (-12, -1, 64, 15, (13, 2), (36, 11), 9, 16, 0.625, -0.4, 0.4, 18.216803661994085, new Complex(0.4923227546245052, -0.03155694394729332),
                new Complex(-0.0005113236121406013, 0), new Complex(-0.030122121961696127, 0.10511482167231039), 1.6461775136930938),
            "B" => (-12, -1, 63, 12, (13, 2), (38, 9), 20, 13, 0.1, -2.5, 2.5, 238.40229831030598, new Complex(-1.6063078483477213, -3.6564925124733785),
                new Complex(0.003851143318974903, 0.002397996135609113), new Complex(-0.0034009117639303144, 0), 0.3018106589579927),
            "C" => (-6, 0, 54, 7, (6, 1), (42, 6), 12, 7, -0.41666666666666663, 0.0, 8.0, 78.92586816803482, new Complex(0.021059985372557932, -0.0724478927234588),
                new Complex(0.11244950865419587, -0.22152099807931475), new Complex(4.126734278792133, 0), 1.623283820295444),
            _ => (-8, 0, 64, 8, (9, 1), (40, 6), 11, 8, 0.38095238095238093, 0.0, 1.0, 16.79967163770862, new Complex(-0.19421086805986637, -0.7209916939963295),
                new Complex(-0.00041749398106041564, 0.0007231207870508384), new Complex(0.15768032072972143, 0.14374430760026358), 3.7305659093919945),
        };
        var (sft, _) = Config(name);
        var x = X();
        Assert.Equal(e.Item1, sft.KMin);
        Assert.Equal(e.Item2, sft.PMin);
        Assert.Equal(e.Item3, sft.KMax(N));
        Assert.Equal(e.Item4, sft.PMax(N));
        Assert.Equal(e.Item5, sft.LowerBorderEnd);
        Assert.Equal(e.Item6, sft.UpperBorderBegin(N));
        Assert.Equal(e.Item7, sft.FPts);
        Close(e.Item9, sft.F[1], 1e-14, "f[1]");
        Close(e.Item10, sft.Times(N)[0], 1e-14, "t[0]");
        Close(e.Item11, sft.DeltaT, 1e-14, "delta_t");

        var s = sft.Stft(x);
        Assert.Equal(e.Item7, s.Shape[0]);
        Assert.Equal(e.Item8, s.Shape[1]);
        Close(e.Item12, AbsSum(s), 1e-12, "sum|S|");
        var cols = s.Shape[1];
        Close(e.Item13, s.Data[1 * cols + 2], 1e-12, "S[1,2]");
        Close(e.Item14, s.Data[s.Data.Length - 1], 1e-12, "S[-1,-1]");
        Close(e.Item15, s.Data[s.Shape[0] / 2 * cols + cols / 2], 1e-12, "S[mid]");
        Close(e.Item16, sft.DualWin[3], 1e-12, "dual_win[3]");

        // Perfect reconstruction (scipy: max error ~5e-16).
        var back = sft.OnesidedFft ? sft.Istft(s, k1: N) : sft.IstftComplex(s, k1: N).Select(v => v.Real).ToArray();
        Assert.Equal(N, back.Length);
        for (var i = 0; i < N; i++) Assert.True(Math.Abs(back[i] - x[i]) < 1e-12, $"istft[{i}]");
    }

    [Fact]
    public void Stft_PaddingSliceRangeAndOffsetMatchScipy()
    {
        var (sft, _) = Config("E");
        var x = X();
        // scipy 1.17.1: stft(x, padding='even', p0=0, p1=8); padding='odd', k_offset=3; padding='edge'.
        var e = sft.Stft(x, 0, 8, padding: ShortTimeFftPadding.Even);
        Assert.Equal(8, e.Shape[1]);
        Close(89.09476487022373, AbsSum(e), 1e-12, "even sum");
        Close(new Complex(0.8088597044032279, 3.7632613445342), e.Data[1 * 8 + 2], 1e-12, "even S[1,2]");
        Close(new Complex(0.3210710358312797, -0.8260151961313068), e.Data[4 * 8 + 4], 1e-12, "even S[mid]");
        var o = sft.Stft(x, kOffset: 3, padding: ShortTimeFftPadding.Odd);
        Close(184.3457908973637, AbsSum(o), 1e-12, "odd sum");
        Close(new Complex(2.1566286038550757, 3.286288512288034), o.Data[1 * 16 + 2], 1e-12, "odd S[1,2]");
        var g = sft.Stft(x, padding: ShortTimeFftPadding.Edge);
        Close(149.79647482512593, AbsSum(g), 1e-12, "edge sum");
        Close(new Complex(-0.24097697569356902, 0.8409185733784831), g.Data[4 * 16 + 8], 1e-12, "edge S[mid]");
    }

    [Fact]
    public void Spectrogram_DetrendAndCrossMatchScipy()
    {
        var (sft, _) = Config("E");
        var x = X();
        var lin = sft.Spectrogram(x, DetrendType.Linear).ToArray();
        Close(224.00762223038444, lin.Sum(), 1e-12, "linear sum");
        Close(2.991047206837183, lin[2 * 16 + 3], 1e-12, "linear [2,3]");
        // A custom callback, per slice: subtract the mean.
        var cus = sft.Spectrogram(x, s => { var m = s.Average(); return s.Select(v => v - m).ToArray(); }).ToArray();
        Close(302.3783069558648, cus.Sum(), 1e-12, "custom sum");
        Close(4.509599219722902, cus[2 * 16 + 3], 1e-12, "custom [2,3]");
        Assert.Equal(cus, sft.Spectrogram(x, DetrendType.Constant).ToArray());

        var y = Enumerable.Range(0, N).Select(k => Math.Cos(0.2 * k)).ToArray();
        var c = sft.Spectrogram(x, y);
        var sum = Complex.Zero;
        foreach (var v in c.Data) sum += v;
        Close(new Complex(52.97142528412618, -35.73052178372907), sum, 1e-12, "cross sum");
        Close(new Complex(1.6275999610851113, -0.4372539682601955), c.Data[2 * 16 + 3], 1e-12, "cross [2,3]");
    }

    [Fact]
    public void DualWindows_MatchScipy()
    {
        var w = Signal.GetWindow("hann", 16);
        var (d, a) = Signal.ClosestStftDualWindow(w, 4);
        Close(0.5, a, 1e-14, "alpha");
        foreach (var v in d) Close(0.5, v, 1e-13, "dual");
        var desired = Enumerable.Range(0, 16).Select(i => 0.5 + i / 15.0).ToArray();
        var (d2, a2) = Signal.ClosestStftDualWindow(w, 4, desired, scaled: false);
        Assert.Equal(1.0, a2);
        double[] e2 = [0.5, 0.539322867879405, 0.5291935221996613, 0.48277020129800463];
        for (var i = 0; i < 4; i++) Close(e2[i], d2[i], 1e-13, $"dual2[{i}]");

        var fromDual = ShortTimeFft.FromDual(d, 4, 1);
        foreach (var v in fromDual.Win.Take(4)) Close(0.5, v, 1e-13, "from_dual win");
        var wed = ShortTimeFft.FromWinEqualsDual(w, 4, 1);
        double[] e3 = [0.0, 0.031076050721577257, 0.11957315586905017, 0.2520184334119718];
        for (var i = 0; i < 4; i++) Close(e3[i], wed.Win[i], 1e-13, $"wed[{i}]");
        // A window equal to its dual inverts with itself.
        for (var i = 0; i < 16; i++) Close(wed.Win[i], wed.DualWin[i], 1e-12, "self-dual");
    }

    [Fact]
    public void Extent_AndValidation()
    {
        var (sft, _) = Config("E");
        Assert.Equal((-4.0, 60.0, 0.0, 0.5625), sft.Extent(N));
        var (f0, f1, t0, t1) = sft.Extent(N, axesSeqTf: false, centerBins: true);
        Close(-0.03125, f0, 1e-15, "f0"); Close(0.53125, f1, 1e-15, "f1"); Close(-6.0, t0, 1e-15, "t0"); Close(58.0, t1, 1e-15, "t1");
        Assert.Throws<ArgumentException>(() => sft.Mfft = 8);
        Assert.Throws<ArgumentException>(() => sft.FftMode = ShortTimeFftMode.Onesided2X); // needs a scaling
        Assert.Throws<ArgumentException>(() => sft.Stft(new Complex[N]));                  // complex input, one-sided
        Assert.Throws<ArgumentException>(() => new ShortTimeFft([1.0, 1.0], 3, 1).DualWin); // hop > window
        Assert.False(new ShortTimeFft([1.0, 0.0, 0.0, 0.0], 2, 1).Invertible);
    }

    [Fact]
    public void ComplexSignal_TwoSidedRoundTrip()
    {
        var sft = ShortTimeFft.FromWindow("hann", 1, 16, 12, fftMode: ShortTimeFftMode.Centered);
        var z = Enumerable.Range(0, 40).Select(k => Complex.FromPolarCoordinates(1 + 0.01 * k, 0.3 * k)).ToArray();
        var s = sft.Stft(z);
        var back = sft.IstftComplex(s, k1: 40);
        for (var i = 0; i < 40; i++) Assert.True((back[i] - z[i]).Magnitude < 1e-12, $"[{i}]");
    }
}
