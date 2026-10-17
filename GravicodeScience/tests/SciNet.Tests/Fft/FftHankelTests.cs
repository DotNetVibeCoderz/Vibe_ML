using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>FFTLog (fht/ifht/fhtoffset) and the n-D Hermitian transforms against scipy 1.17.1.</summary>
public class FftHankelTests
{
    private const int N = 64;
    private static readonly double Dln = Math.Log(10) / 8;
    private static readonly int[] Idx = [0, 10, 31, 32, 50, 63];

    private static double[] Gauss(double mu)
    {
        var a = new double[N];
        for (var j = 0; j < N; j++)
        {
            var r = Math.Exp((j - (N - 1) / 2.0) * Dln);
            a[j] = Math.Pow(r, mu + 1) * Math.Exp(-r * r / 2);
        }
        return a;
    }

    private static void Close(double e, double a, double tol, string w)
        => Assert.True(Math.Abs(e - a) <= tol, $"{w}: expected {e:R}, got {a:R}");

    private static void Pick(double[] expected, double[] got, double tol, string w)
    {
        for (var i = 0; i < Idx.Length; i++) Close(expected[i], got[Idx[i]], tol, $"{w}[{Idx[i]}]");
    }

    [Fact]
    public void Fht_MatchesScipy()
    {
        var a = Gauss(0);
        Pick([6.868541706628317e-05, 0.002051597321929756, 0.5952679939586007, 0.5927500182068944, 1.9194963500994422e-05, 4.5124283098441076e-05],
            Fft.Fht(a, Dln, 0), 1e-13, "fht");
        var off = Fft.Fhtoffset(Dln, 0);
        Close(-0.04757498683841151, off, 1e-13, "fhtoffset");
        Pick([6.332602906206075e-05, 0.0019594745970270278, 0.5871955785426038, 0.600549963919373, 1.1658673757936056e-06, 4.5100457657043513e-05],
            Fft.Fht(a, Dln, 0, offset: off), 1e-13, "fht offset");
        Pick([-0.02724303816675411, -1.1053780938446616e-06, 0.6238810637482658, 0.6378252443708072, 0.001803452937703686, 8.38434023506226e-06],
            Fft.Fht(a, Dln, 0.5, offset: 0.3, bias: 0.4), 1e-13, "fht bias");
        Pick([0.03700463292844452, 0.00044646729428139085, 0.14638490365177248, 0.2627409253062492, 0.008916659368570013, 0.00021396358674605375],
            Fft.Ifht(a, Dln, 1.5, offset: -0.2, bias: -0.3), 1e-13, "ifht");
        Close(0.6692745478882607, Fft.Fhtoffset(0.1, 2.5, initial: 0.7, bias: 0.3), 1e-13, "fhtoffset 2");
        Close(-0.022027286150683525, Fft.Fhtoffset(0.05, -0.5), 1e-13, "fhtoffset 3");

        var b = Enumerable.Range(0, 33).Select(j => Math.Exp(-j * j / 50.0)).ToArray();
        var fb = Fft.Fht(b, 0.2, 1.0);
        double[] eb = [0.7735162289755343, -0.28027176435808215, -0.019802512928575267, 0.7811987065621029];
        int[] ib = [0, 5, 16, 32];
        for (var i = 0; i < 4; i++) Close(eb[i], fb[ib[i]], 1e-13, $"odd n [{ib[i]}]");
    }

    [Fact]
    public void Fht_AnalyticPairAndInverse()
    {
        // ∫ r^(μ+1) e^(−r²/2) J_μ(kr) k dr = k^(μ+1) e^(−k²/2): the Gaussian is its own Hankel
        // transform. On a 32-points-per-decade grid FFTLog reproduces it to ~1e-10.
        const double mu = 1.0;
        const int n = 256;
        var dln = Math.Log(10) / 32;
        var a = new double[n];
        for (var j = 0; j < n; j++)
        {
            var r = Math.Exp((j - (n - 1) / 2.0) * dln);
            a[j] = Math.Pow(r, mu + 1) * Math.Exp(-r * r / 2);
        }
        var off = Fft.Fhtoffset(dln, mu);
        var A = Fft.Fht(a, dln, mu, off);
        var kc = Math.Exp(off); // r_c = 1
        for (var j = 64; j < 192; j++)
        {
            var k = kc * Math.Exp((j - (n - 1) / 2.0) * dln);
            Close(Math.Pow(k, mu + 1) * Math.Exp(-k * k / 2), A[j], 1e-9, $"pair[{j}]");
        }
        var back = Fft.Ifht(A, dln, mu, off);
        for (var j = 0; j < n; j++) Close(a[j], back[j], 1e-13, $"inverse[{j}]");
    }

    [Fact]
    public void ComplexLogGamma_MatchesScipy()
    {
        void C(Complex e, Complex z) => Assert.True((Fft.FftLogGamma(z) - e).Magnitude < 1e-12 * Math.Max(1, e.Magnitude), $"loggamma({z})");
        C(new Complex(-7.478236042050316, -5.726104271910385), new Complex(-2.5, 3));
        C(new Complex(-62.65068605396814, 107.24156057988668), new Complex(0.3, 40));
        C(new Complex(-7.383486336994913, -24.532008881328785), new Complex(-7.2, 0.1));
    }

    [Fact]
    public void HermitianNd_MatchesScipy()
    {
        var data = new Complex[12];
        for (var i = 0; i < 12; i++) data[i] = new Complex(i, Math.Cos(i));
        var x = new ComplexNdArray(data, 3, 4);
        var h = Fft.Hfft2(x);
        Assert.Equal(new[] { 3, 6 }, h.Shape.ToArray());
        var hv = h.ToArray();
        Close(99.0, hv[0], 1e-12, "hfft2[0,0]");
        Close(-0.07901284524955621, hv[1 * 6 + 2], 1e-12, "hfft2[1,2]");
        Close(0.0, hv.Sum(), 1e-12, "hfft2 sum");
        var hn = Fft.HfftN(x, [3, 7], norm: FftNorm.Ortho);
        Assert.Equal(new[] { 3, 7 }, hn.Shape.ToArray());
        Close(3.7036419065857795, hn.ToArray()[2 * 7 + 6], 1e-12, "hfftn[2,6]");

        var y = new NdArray(Enumerable.Range(0, 20).Select(i => Math.Sin(i)).ToArray(), 4, 5);
        var ih = Fft.Ihfft2(y);
        Assert.Equal(new[] { 4, 3 }, ih.Shape.ToArray());
        var s = Complex.Zero;
        foreach (var v in ih.Data) s += v;
        Assert.True((s - new Complex(0.11350859243855169, 0.43608664138695263)).Magnitude < 1e-13, "ihfft2 sum");
        Assert.True((ih.Data[1 * 3 + 2] - new Complex(0.04308302826178977, -0.01606702344785916)).Magnitude < 1e-13, "ihfft2[1,2]");
        var ihn = Fft.IhfftN(y, norm: FftNorm.Forward);
        Assert.True((ihn.Data[3 * 3 + 1] - new Complex(-0.683916311554116, 8.669921721450951)).Magnitude < 1e-12, "ihfftn[3,1]");
        // Round trip: hfftn(ihfftn(y), s = y.shape) = y.
        var rt = Fft.HfftN(Fft.IhfftN(y), [4, 5]).ToArray();
        var yv = y.ToArray();
        for (var i = 0; i < 20; i++) Close(yv[i], rt[i], 1e-13, $"round trip [{i}]");
    }
}
