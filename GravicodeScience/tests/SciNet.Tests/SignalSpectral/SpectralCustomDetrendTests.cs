using System.Numerics;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Welch/Csd/Spectrogram/Stft with a detrend callback (median removal), against scipy 1.17.1.</summary>
public class SpectralCustomDetrendTests
{
    private static double[] Median(double[] s)
    {
        var c = (double[])s.Clone();
        Array.Sort(c);
        var m = c.Length % 2 == 1 ? c[c.Length / 2] : 0.5 * (c[c.Length / 2 - 1] + c[c.Length / 2]);
        return s.Select(v => v - m).ToArray();
    }

    private static readonly double[] X = Enumerable.Range(0, 300).Select(k => Math.Sin(0.37 * k) + 0.002 * k * k / 50 - 0.3 * Math.Cos(1.3 * k)).ToArray();
    private static readonly double[] Y = Enumerable.Range(0, 300).Select(k => Math.Cos(0.2 * k) + 0.01 * k).ToArray();

    private static void Close(double e, double a, double tol, string w) => Assert.True(Math.Abs(e - a) <= tol * Math.Max(1, Math.Abs(e)), $"{w}: {e:R} vs {a:R}");
    private static void Close(Complex e, Complex a, double tol, string w) => Assert.True((e - a).Magnitude <= tol * Math.Max(1, e.Magnitude), $"{w}: {e} vs {a}");

    [Fact]
    public void CustomDetrend_MatchesScipy()
    {
        var (_, p) = Signal.Welch(X, detrend: Median, fs: 2, nperseg: 64);
        Close(18.420464002773365, p.Sum(), 1e-12, "welch sum");
        Close(4.8289754178666895, p[3], 1e-12, "welch[3]");

        var (_, c) = Signal.Csd(X, Y, detrend: Median, nperseg: 50);
        var cs = Complex.Zero; foreach (var v in c) cs += v;
        Close(new Complex(0.30291998904886325, -0.9971510842922987), cs, 1e-12, "csd sum");
        Close(new Complex(-0.25582390064977917, -0.7510722049626489), c[2], 1e-12, "csd[2]");

        var (_, t, s, _) = Signal.Spectrogram(X, detrend: Median, nperseg: 40);
        var sa = s!.ToArray();
        Close(204.66937933658923, sa.Sum(), 1e-12, "spec sum");
        Close(15.093997998365088, sa[2 * t.Length + 3], 1e-12, "spec[2,3]");

        var (_, tz, z) = Signal.Stft(X, detrend: Median, nperseg: 32);
        double az = 0; foreach (var v in z.Data) az += v.Magnitude;
        Close(31.58163333293858, az, 1e-12, "stft sum|Z|");
        Close(new Complex(-0.4404643697709726, -0.23240195063945895), z.Data[2 * tz.Length + 3], 1e-12, "stft[2,3]");
    }

    [Fact]
    public void CustomDetrend_EqualsBuiltInWhenItIsTheSame()
    {
        static double[] Mean(double[] s) { var m = s.Average(); return s.Select(v => v - m).ToArray(); }
        var a = Signal.Welch(X, detrend: Mean, nperseg: 64).Pxx;
        var b = Signal.Welch(X, nperseg: 64).Pxx;
        for (var i = 0; i < a.Length; i++) Close(b[i], a[i], 1e-13, $"[{i}]");
        Assert.Throws<ArgumentException>(() => Signal.Welch(X, detrend: s => s[1..], nperseg: 64));
    }
}
