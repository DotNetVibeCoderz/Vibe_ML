using System.Numerics;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class FftEngineTests
{
    public static IEnumerable<object[]> SmallLengths() => Enumerable.Range(1, 64).Select(n => new object[] { n });

    [Theory]
    [MemberData(nameof(SmallLengths))]
    public void Fft_MatchesNaiveDft_ForEveryLengthUpTo64(int n)
    {
        var x = FftAssert.RandomComplex(n, n);
        FftAssert.Close(FftAssert.NaiveDft(x), Fft.Fft1D(x), 1e-13, $"fft n={n}");
        FftAssert.Close(FftAssert.NaiveDft(x, inverse: true), Fft.Ifft(x, norm: FftNorm.Forward), 1e-13, $"ifft n={n}");
    }

    [Theory]
    [InlineData(97)]      // prime, generic butterfly
    [InlineData(1009)]    // prime, Bluestein
    [InlineData(1000)]    // 2^3 5^3
    [InlineData(1001)]    // 7 11 13: generic radices, no Bluestein
    [InlineData(1536)]    // 2^9 3
    [InlineData(2187)]    // 3^7
    [InlineData(3125)]    // 5^5
    [InlineData(4096)]
    [InlineData(2 * 1009)]
    public void Fft_MatchesNaiveDft_ForAwkwardLengths(int n)
    {
        var x = FftAssert.RandomComplex(n, 7 * n);
        var tol = 1e-13 * Math.Log2(n);
        FftAssert.Close(FftAssert.NaiveDft(x), Fft.Fft1D(x), tol, $"fft n={n}");
        FftAssert.Close(FftAssert.NaiveDft(x, inverse: true), Fft.Ifft(x, norm: FftNorm.Forward), tol, $"ifft n={n}");
    }

    [Fact]
    public void Fft_LargePrime_MatchesNaiveBins()
    {
        const int n = 10007;
        Assert.True(FftPlan.Get(n).UsesBluestein);
        var x = FftAssert.RandomComplex(n, 3);
        var y = Fft.Fft1D(x);
        var scale = FftAssert.MaxAbs(y);
        foreach (var k in new[] { 0, 1, 2, 17, 500, 5003, 5004, 9999, 10006 })
        {
            var err = (FftAssert.NaiveBin(x, k) - y[k]).Magnitude / scale;
            Assert.True(err < 1e-12, $"bin {k}: relative error {err:E2}");
        }
    }

    [Fact]
    public void Plans_ChooseDirectRadicesForSmoothLengths()
    {
        Assert.Equal(new[] { 2, 4, 4, 4, 4, 3 }, FftPlan.Get(1536).Factors);
        Assert.False(FftPlan.Get(1001).UsesBluestein);
        Assert.False(FftPlan.Get(3125).UsesBluestein);
        Assert.True(FftPlan.Get(1009).UsesBluestein);
        Assert.Same(FftPlan.Get(4096), FftPlan.Get(4096));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(16)]
    [InlineData(105)]
    [InlineData(1024)]
    [InlineData(10007)]
    [InlineData(65536)]
    [InlineData(3 * 5 * 7 * 11 * 13)]
    public void RoundTrip_RecoversInput(int n)
    {
        var x = FftAssert.RandomComplex(n, n + 1);
        FftAssert.Close(x, Fft.Ifft(Fft.Fft1D(x)), 1e-14 * Math.Log2(n + 1) + 1e-15, $"n={n}");
    }

    [Theory]
    [InlineData(4096)]
    [InlineData(1000)]
    [InlineData(1009)]
    public void Parseval_HoldsUnderOrthoNorm(int n)
    {
        var x = FftAssert.RandomComplex(n, 11);
        var y = Fft.Fft1D(x, norm: FftNorm.Ortho);
        double ex = 0, ey = 0;
        foreach (var v in x) ex += v.Real * v.Real + v.Imaginary * v.Imaginary;
        foreach (var v in y) ey += v.Real * v.Real + v.Imaginary * v.Imaginary;
        Assert.True(Math.Abs(ex - ey) <= 1e-12 * ex, $"energy {ex} vs {ey}");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(48)]
    [InlineData(4096)]
    [InlineData(1536)]
    [InlineData(3125)]
    [InlineData(1001)]
    [InlineData(1009)]
    public void SimdPath_AgreesWithScalarReferencePath(int n)
    {
        var x = FftAssert.RandomComplex(n, 5);
        var plan = new FftPlan(n);
        var simd = new Complex[n];
        var scalar = new Complex[n];
        var simdInv = new Complex[n];
        var scalarInv = new Complex[n];
        lock (SimdSwitch.Gate)
        {
            try
            {
                FftPlan.SimdEnabled = true;
                plan.Execute(x, simd);
                plan.Execute(x, simdInv, inverse: true);
                FftPlan.SimdEnabled = false;
                plan.Execute(x, scalar);
                plan.Execute(x, scalarInv, inverse: true);
            }
            finally { FftPlan.SimdEnabled = true; }
        }
        FftAssert.Close(scalar, simd, 1e-14, "forward");
        FftAssert.Close(scalarInv, simdInv, 1e-14, "backward");
    }

    [Theory]
    [InlineData(64)]
    [InlineData(48)]
    [InlineData(4096)]
    [InlineData(1009)]
    public void Plan_InPlaceAndOutOfPlaceAgree(int n)
    {
        var x = FftAssert.RandomComplex(n, 9);
        var plan = FftPlan.Get(n);
        var outOfPlace = new Complex[n];
        plan.Execute(x, outOfPlace);
        var inPlace = (Complex[])x.Clone();
        plan.ExecuteInPlace(inPlace);
        FftAssert.Close(outOfPlace, inPlace, 0, "in place");
        var scratch = new Complex[plan.ScratchLength];
        var explicitScratch = new Complex[n];
        plan.Execute(x, explicitScratch, scratch);
        FftAssert.Close(outOfPlace, explicitScratch, 0, "explicit scratch");
    }

    [Fact]
    public void SpanApi_TransformsInPlace()
    {
        var x = FftAssert.RandomComplex(360, 1);
        Span<Complex> data = (Complex[])x.Clone();
        Fft.Forward(data);
        FftAssert.Close(Fft.Fft1D(x), data, 0, "forward");
        Fft.Inverse(data);
        FftAssert.Close(x, data, 1e-14, "inverse");
    }

    [Fact]
    public void Plan_RejectsPartialOverlap()
    {
        var buffer = new Complex[20];
        var plan = FftPlan.Get(10);
        Assert.Throws<ArgumentException>(() => plan.Execute(buffer.AsSpan(0, 10), buffer.AsSpan(5, 10)));
        Assert.Throws<ArgumentException>(() => new FftPlan(0));
        Assert.Throws<ArgumentException>(() => Fft.Fft1D(Array.Empty<Complex>()));
    }

    [Fact]
    public void Plans_AreSafeToShareAcrossThreads()
    {
        var x = FftAssert.RandomComplex(1009, 4);
        var expected = Fft.Fft1D(x);
        var failures = 0;
        Parallel.For(0, 64, _ =>
        {
            var y = Fft.Fft1D(x);
            for (var i = 0; i < y.Length; i++)
                if (y[i] != expected[i]) { Interlocked.Increment(ref failures); break; }
        });
        Assert.Equal(0, failures);
    }

    [Fact]
    public void Fft_PadsAndTruncatesToN()
    {
        var x = FftAssert.RandomComplex(10, 2);
        var padded = new Complex[16];
        Array.Copy(x, padded, 10);
        FftAssert.Close(FftAssert.NaiveDft(padded), Fft.Fft1D(x, n: 16), 1e-14, "padded");
        FftAssert.Close(FftAssert.NaiveDft(x[..6]), Fft.Fft1D(x, n: 6), 1e-14, "truncated");
    }

    [Fact]
    public void Fft_DeltaAndConstant_HaveClosedFormSpectra()
    {
        // delta -> all ones; constant -> n at DC.
        var delta = new Complex[12];
        delta[0] = 1;
        foreach (var v in Fft.Fft1D(delta)) Assert.True((v - 1).Magnitude < 1e-15);
        var ones = Enumerable.Repeat(Complex.One, 12).ToArray();
        var y = Fft.Fft1D(ones);
        Assert.True((y[0] - 12).Magnitude < 1e-14);
        for (var k = 1; k < 12; k++) Assert.True(y[k].Magnitude < 1e-14);
    }
}

/// <summary>Serialises tests that flip the process-wide SIMD switch.</summary>
internal static class SimdSwitch
{
    public static readonly object Gate = new();
}
