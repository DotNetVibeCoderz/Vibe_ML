using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Signal;
using Xunit;

namespace GraviNum.Tests;

/// <summary>
/// Tests for the discrete Fourier transform.
/// </summary>
/// <remarks>
/// Three kinds of check, in increasing order of what they prove. Against the direct
/// <c>O(n²)</c> DFT, which shares no code with the fast paths. Against closed-form answers — a
/// sinusoid, a constant, an impulse — where the right answer is known without computing anything.
/// And round trips, which catch a forward and inverse that are wrong in mirror-image ways.
/// </remarks>
public class FftTests
{
    private static Complex[] Random(int n, int seed)
    {
        var rng = new GraviRandom(seed);
        var values = new Complex[n];
        for (var i = 0; i < n; i++) values[i] = new Complex(rng.Normal(), rng.Normal());
        return values;
    }

    private static double MaxDiff(Complex[] a, Complex[] b)
    {
        var worst = 0.0;
        for (var i = 0; i < a.Length; i++) worst = Math.Max(worst, (a[i] - b[i]).Magnitude);
        return worst;
    }

    // ---------------------------------------------------------------- against the definition

    [Theory]
    [InlineData(2)]
    [InlineData(8)]
    [InlineData(64)]
    [InlineData(1024)]
    public void PowerOfTwoTransformMatchesTheDirectDft(int n)
    {
        var input = Random(n, 3);
        var worst = MaxDiff(Fft.Forward(input), Fft.DirectDft(input));

        // The error grows slowly with n, since both accumulate rounding over n terms.
        Assert.True(worst < 1e-9 * n, $"n={n}: max difference {worst:E2}");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(7)]
    [InlineData(12)]
    [InlineData(100)]
    [InlineData(101)]     // prime, so no radix decomposition helps at all
    [InlineData(360)]
    public void ArbitraryLengthTransformMatchesTheDirectDft(int n)
    {
        // This is the Bluestein path, and the reason the library does not simply demand a
        // power-of-two length. A prime length is the worst case for any radix approach.
        var input = Random(n, 5);
        var worst = MaxDiff(Fft.Forward(input), Fft.DirectDft(input));

        Assert.True(worst < 1e-9 * n, $"n={n}: max difference {worst:E2}");
    }

    [Theory]
    [InlineData(16)]
    [InlineData(37)]
    public void InverseTransformMatchesTheDirectInverse(int n)
    {
        var input = Random(n, 7);
        var worst = MaxDiff(Fft.Inverse(input), Fft.DirectDft(input, inverse: true));

        Assert.True(worst < 1e-9 * n, $"n={n}: max difference {worst:E2}");
    }

    // ---------------------------------------------------------------- closed forms

    [Fact]
    public void AConstantSignalPutsAllItsEnergyInBinZero()
    {
        // The mean is the zero-frequency component, so a constant signal has one non-zero bin and
        // its value is n times the constant.
        const int n = 64;
        var signal = new double[n];
        Array.Fill(signal, 2.5);

        var spectrum = Fft.ForwardReal(signal);

        Assert.Equal(2.5 * n, spectrum[0].Real, 9);
        for (var k = 1; k < spectrum.Length; k++)
            Assert.True(spectrum[k].Magnitude < 1e-9, $"bin {k} held {spectrum[k].Magnitude:E2}");
    }

    [Fact]
    public void AnImpulseHasAFlatSpectrum()
    {
        // A unit impulse contains every frequency equally — the transform pair that makes the
        // uncertainty principle concrete.
        const int n = 32;
        var signal = new double[n];
        signal[0] = 1.0;

        var spectrum = Fft.ForwardReal(signal);

        foreach (var bin in spectrum)
            Assert.Equal(1.0, bin.Magnitude, 9);
    }

    [Fact]
    public void ASinusoidPeaksAtItsOwnFrequency()
    {
        // 5 Hz sampled at 64 Hz for one second: the peak must land in bin 5 and nowhere else.
        const int n = 64;
        const double frequency = 5.0;

        var signal = new double[n];
        for (var i = 0; i < n; i++) signal[i] = Math.Sin(2 * Math.PI * frequency * i / n);

        var spectrum = Fft.ForwardReal(signal);
        var magnitudes = spectrum.Select(c => c.Magnitude).ToArray();

        var peak = Array.IndexOf(magnitudes, magnitudes.Max());
        Assert.Equal(5, peak);

        // A pure tone at an exact bin frequency leaves everything else at zero — no leakage,
        // because the signal contains a whole number of cycles.
        for (var k = 0; k < magnitudes.Length; k++)
            if (k != peak) Assert.True(magnitudes[k] < 1e-9, $"bin {k} leaked {magnitudes[k]:E2}");

        // And the frequency helper must agree about what bin 5 means.
        Assert.Equal(frequency, Fft.FrequencyBins(n, sampleRate: n).At(peak), 9);
    }

    [Fact]
    public void TwoTonesAppearAsTwoPeaks()
    {
        const int n = 128;
        var signal = new double[n];
        for (var i = 0; i < n; i++)
            signal[i] = Math.Sin(2 * Math.PI * 7 * i / n) + 0.5 * Math.Sin(2 * Math.PI * 20 * i / n);

        var magnitudes = Fft.Magnitude(signal);

        // The amplitudes are 1 and 0.5, so the peaks should stand in the same ratio.
        Assert.True(magnitudes.At(7) > magnitudes.At(20));
        Assert.Equal(2.0, magnitudes.At(7) / magnitudes.At(20), 6);

        for (var k = 0; k < magnitudes.Size; k++)
            if (k != 7 && k != 20) Assert.True(magnitudes.At(k) < 1e-9);
    }

    // ---------------------------------------------------------------- round trips

    [Theory]
    [InlineData(8)]
    [InlineData(13)]      // Bluestein both ways
    [InlineData(256)]
    public void ForwardThenInverseReturnsTheOriginal(int n)
    {
        var input = Random(n, 11);
        var worst = MaxDiff(Fft.Inverse(Fft.Forward(input)), input);

        Assert.True(worst < 1e-10, $"n={n}: round trip drifted by {worst:E2}");
    }

    [Theory]
    [InlineData(64)]
    [InlineData(75)]
    public void ARealSignalSurvivesTheHalfSpectrumRoundTrip(int n)
    {
        var rng = new GraviRandom(13);
        var signal = new double[n];
        for (var i = 0; i < n; i++) signal[i] = rng.Normal();

        var restored = Fft.InverseReal(Fft.ForwardReal(signal), n);

        for (var i = 0; i < n; i++)
            Assert.True(Math.Abs(signal[i] - restored[i]) < 1e-10,
                $"n={n}, sample {i}: {signal[i]:G17} vs {restored[i]:G17}");
    }

    [Fact]
    public void TheRealTransformKeepsOnlyTheDistinctBins()
    {
        // The upper half of a real signal's spectrum is the conjugate of the lower half, so
        // returning it would imply structure that is not there.
        const int n = 64;
        var rng = new GraviRandom(17);
        var signal = new double[n];
        for (var i = 0; i < n; i++) signal[i] = rng.Normal();

        var half = Fft.ForwardReal(signal);
        Assert.Equal(n / 2 + 1, half.Length);

        // Check the symmetry actually holds against the full transform.
        var full = Fft.Forward([.. signal.Select(v => new Complex(v, 0))]);
        for (var k = 1; k < n / 2; k++)
            Assert.True((full[n - k] - Complex.Conjugate(full[k])).Magnitude < 1e-9);
    }

    // ---------------------------------------------------------------- convolution

    [Theory]
    [InlineData(8, 5)]
    [InlineData(50, 30)]
    [InlineData(17, 17)]
    public void ConvolutionMatchesTheDirectSum(int m, int n)
    {
        var rng = new GraviRandom(19);
        var a = new double[m];
        var b = new double[n];
        for (var i = 0; i < m; i++) a[i] = rng.Normal();
        for (var i = 0; i < n; i++) b[i] = rng.Normal();

        var fast = Fft.Convolve(a, b);

        // The definition, O(n m), computed independently.
        var expected = new double[m + n - 1];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                expected[i + j] += a[i] * b[j];

        Assert.Equal(expected.Length, fast.Length);
        for (var i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - fast[i]) < 1e-9,
                $"index {i}: {expected[i]:G17} vs {fast[i]:G17}");
    }

    [Fact]
    public void ConvolutionIsLinearNotCircular()
    {
        // Without zero-padding to n + m - 1 the tail wraps around and corrupts the start, which
        // is the classic bug in an FFT-based convolution. The full-length result is the tell.
        double[] a = [1, 2, 3];
        double[] b = [4, 5];

        var result = Fft.Convolve(a, b);

        Assert.Equal(4, result.Length);           // 3 + 2 - 1, not 3
        Assert.Equal(4.0, result[0], 9);          // 1*4
        Assert.Equal(13.0, result[1], 9);         // 1*5 + 2*4
        Assert.Equal(22.0, result[2], 9);         // 2*5 + 3*4
        Assert.Equal(15.0, result[3], 9);         // 3*5
    }

    [Fact]
    public void ConvolvingWithAnImpulseReturnsTheSignal()
    {
        double[] signal = [3, 1, 4, 1, 5];
        var result = Fft.Convolve(signal, [1.0]);

        for (var i = 0; i < signal.Length; i++) Assert.Equal(signal[i], result[i], 9);
    }

    // ---------------------------------------------------------------- edges

    [Fact]
    public void DegenerateLengthsAreHandled()
    {
        Assert.Empty(Fft.Convolve([], [1.0]));
        Assert.Single(Fft.Forward([new Complex(2, 3)]));
        Assert.Equal(new Complex(2, 3), Fft.Forward([new Complex(2, 3)])[0]);
    }

    [Fact]
    public void ALongPrimeLengthStaysAccurate()
    {
        // Bluestein computes t² mod 2n rather than t² so the phase does not drift as n grows.
        // Squaring 1531 directly is fine; the guard matters for much longer inputs, and this
        // pins the behaviour before anyone "simplifies" it away.
        const int n = 1531;
        var input = Random(n, 23);

        var worst = MaxDiff(Fft.Forward(input), Fft.DirectDft(input));
        Assert.True(worst < 1e-6, $"max difference {worst:E2}");
    }
}
