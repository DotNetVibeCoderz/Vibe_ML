using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class FftRealAndNdTests
{
    private static Complex[] ToComplex(double[] x) => x.Select(v => new Complex(v, 0)).ToArray();

    public static IEnumerable<object[]> RealLengths()
        => new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 30, 64, 97, 100, 1000, 1009, 2048, 2 * 1009 }.Select(n => new object[] { n });

    [Theory]
    [MemberData(nameof(RealLengths))]
    public void Rfft_AgreesWithComplexTransform(int n)
    {
        var x = FftAssert.RandomReal(n, n);
        var full = FftAssert.NaiveDft(ToComplex(x));
        var half = Fft.Rfft(x);
        Assert.Equal(n / 2 + 1, half.Length);
        FftAssert.Close(full[..(n / 2 + 1)], half, 1e-13, $"rfft n={n}");
        FftAssert.Close(full, Fft.Fft1D(x), 1e-13, $"fft(real) n={n}");
    }

    [Theory]
    [MemberData(nameof(RealLengths))]
    public void Irfft_RoundTrips_GivenN(int n)
    {
        var x = FftAssert.RandomReal(n, n + 3);
        FftAssert.Close(x, Fft.Irfft(Fft.Rfft(x), n), 1e-14 * Math.Log2(n + 2), $"n={n}");
        FftAssert.Close(x, Fft.Irfft(Fft.Rfft(x, norm: FftNorm.Ortho), n, FftNorm.Ortho), 1e-13, $"ortho n={n}");
    }

    [Fact]
    public void Irfft_IgnoresImaginaryPartOfDcAndNyquist_LikeNumpy()
    {
        var spectrum = new[] { new Complex(4, 3), new Complex(1, -1), new Complex(2, 5) };
        var a = Fft.Irfft(spectrum);
        var b = Fft.Irfft([new Complex(4, 0), new Complex(1, -1), new Complex(2, 0)]);
        FftAssert.Close(b, a, 0, "imag ignored");
    }

    [Fact]
    public void Rfft_LargePowerOfTwo_RoundTrips()
    {
        var n = 1 << 20;
        var x = FftAssert.RandomReal(n, 42);
        var back = Fft.Irfft(Fft.Rfft(x), n);
        double worst = 0;
        for (var i = 0; i < n; i++) worst = Math.Max(worst, Math.Abs(back[i] - x[i]));
        Assert.True(worst < 1e-14, $"max error {worst:E2}");
    }

    [Fact]
    public void Hfft_AndIhfft_MatchScipy()
    {
        // scipy 1.17.1: hfft([1+2j, -0.5+0.25j, 3-1j, 0.75j]) and with n=7; ihfft([1, 2, -1, 0.5, 3]).
        Complex[] z = [new(1, 2), new(-0.5, 0.25), new(3, -1), new(0, 0.75)];
        FftAssert.Abs([6.0, -3.799038105676658, 0.6650635094610964, 8.0, -3.6650635094610964, -1.200961894323342], Fft.Hfft(z), 1e-13, "hfft");
        FftAssert.Abs([6.0, -1.866729880049915, -4.000808062834217, 8.884904381822395, 2.398910976287246, -4.365776484082183, -0.05050093114332477],
            Fft.Hfft(z, 7), 1e-13, "hfft n=7");
        var ih = Fft.Ihfft([1.0, 2.0, -1.0, 0.5, 3.0]);
        FftAssert.Close([new(1.1, 0), new(0.5899186938124422, -0.3665468789467727), new(-0.6399186938124423, 0.16775990443005143)], ih, 1e-14, "ihfft");
    }

    [Fact]
    public void Fft2_MatchesSeparableComposition_AndNaive2D()
    {
        const int rows = 6, cols = 10;
        var data = FftAssert.RandomComplex(rows * cols, 21);
        var x = new ComplexNdArray((Complex[])data.Clone(), rows, cols);
        var y = Fft.Fft2(x).ToArray();

        // Naive 2-D DFT.
        var naive = new Complex[rows * cols];
        for (var k = 0; k < rows; k++)
        for (var l = 0; l < cols; l++)
        {
            var s = Complex.Zero;
            for (var r = 0; r < rows; r++)
            for (var c = 0; c < cols; c++)
                s += data[r * cols + c] * Complex.Exp(new Complex(0, -2 * Math.PI * ((double)k * r / rows + (double)l * c / cols)));
            naive[k * cols + l] = s;
        }
        FftAssert.Close(naive, y, 1e-13, "fft2 vs naive");
        FftAssert.Close(Fft.Fft1D(Fft.Fft1D(x, axis: 0), axis: 1).ToArray(), y, 1e-15, "fft2 vs composition");
        FftAssert.Close(data, Fft.Ifft2(Fft.Fft2(x)).ToArray(), 1e-14, "ifft2");
    }

    [Fact]
    public void FftN_OverChosenAxes_WithPadding()
    {
        var data = FftAssert.RandomComplex(4 * 5 * 6, 2);
        var x = new ComplexNdArray(data, 4, 5, 6);
        var y = Fft.FftN(x, s: [8, 7], axes: [0, 2]);
        Assert.Equal(new[] { 8, 5, 7 }, y.Shape.ToArray());
        var manual = Fft.Fft1D(Fft.Fft1D(x, n: 8, axis: 0), n: 7, axis: 2);
        FftAssert.Close(manual.ToArray(), y.ToArray(), 1e-15, "fftn");
        var all = Fft.FftN(x);
        FftAssert.Close(data, Fft.IfftN(all).ToArray(), 1e-14, "ifftn");
    }

    [Fact]
    public void RfftN_AgreesWithFftN_AndInverts()
    {
        var data = FftAssert.RandomReal(6 * 4 * 9, 13);
        var x = new NdArray(data, 6, 4, 9);
        var r = Fft.RfftN(x);
        Assert.Equal(new[] { 6, 4, 5 }, r.Shape.ToArray());
        var full = Fft.FftN(x).ToArray();
        var half = r.ToArray();
        for (var i = 0; i < 6 * 4; i++)
            FftAssert.Close(full.AsSpan(i * 9, 5), half.AsSpan(i * 5, 5), 1e-14, $"line {i}");
        FftAssert.Close(data, Fft.IrfftN(r, s: [6, 4, 9]).ToArray(), 1e-14, "irfftn odd last axis");

        var r2 = Fft.Rfft2(new NdArray(data[..48], 6, 8));
        FftAssert.Close(data[..48], Fft.Irfft2(r2).ToArray(), 1e-14, "irfft2");
    }

    [Fact]
    public void AxisTransforms_ParallelAndSerialAgree()
    {
        // 64 x 1024 along axis 0 (strided lines) and axis 1 (contiguous): big enough to go parallel.
        var data = FftAssert.RandomComplex(64 * 1024, 77);
        var x = new ComplexNdArray(data, 64, 1024);
        foreach (var axis in new[] { 0, 1 })
        {
            var serial = Fft.Fft1D(x, axis: axis, workers: 1).ToArray();
            var parallel = Fft.Fft1D(x, axis: axis).ToArray();
            FftAssert.Close(serial, parallel, 0, $"axis {axis}");
        }
        var real = new NdArray(FftAssert.RandomReal(64 * 1024, 5), 1024, 64);
        FftAssert.Close(Fft.Rfft(real, axis: 0, workers: 1).ToArray(), Fft.Rfft(real, axis: 0, workers: -1).ToArray(), 0, "rfft axis 0");
    }

    [Fact]
    public void StridedAxis_MatchesColumnByColumn()
    {
        const int rows = 9, cols = 13; // 13 columns: one full block of 8 and a ragged one
        var data = FftAssert.RandomComplex(rows * cols, 6);
        var y = Fft.Fft1D(new ComplexNdArray(data, rows, cols), axis: 0).ToArray();
        for (var c = 0; c < cols; c++)
        {
            var column = Enumerable.Range(0, rows).Select(r => data[r * cols + c]).ToArray();
            var expected = FftAssert.NaiveDft(column);
            var actual = Enumerable.Range(0, rows).Select(r => y[r * cols + c]).ToArray();
            FftAssert.Close(expected, actual, 1e-14, $"column {c}");
        }
    }

    [Fact]
    public void Freqs_And_Shifts_MatchScipy()
    {
        // scipy 1.17.1.
        FftAssert.Abs([0.0, 0.2857142857142857, 0.5714285714285714, 0.8571428571428571, -0.8571428571428571, -0.5714285714285714, -0.2857142857142857],
            Fft.FftFreq(7, 0.5), 1e-16, "fftfreq");
        FftAssert.Abs([0.0, 1.25, 2.5, 3.75, 5.0], Fft.RfftFreq(8, 0.1), 1e-15, "rfftfreq");
        double[] r7 = [0, 1, 2, 3, 4, 5, 6];
        Assert.Equal(new double[] { 4, 5, 6, 0, 1, 2, 3 }, Fft.FftShift(r7));
        Assert.Equal(new double[] { 3, 4, 5, 6, 0, 1, 2 }, Fft.IfftShift(r7));
        Assert.Equal(r7, Fft.IfftShift(Fft.FftShift(r7)));
        var a = new NdArray(Enumerable.Range(0, 12).Select(v => (double)v).ToArray(), 3, 4);
        Assert.Equal(new double[] { 10, 11, 8, 9, 2, 3, 0, 1, 6, 7, 4, 5 }, Fft.FftShift(a).ToArray());
        Assert.Equal(new double[] { 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3 }, Fft.IfftShift(a, axes: [0]).ToArray());
        var c = Fft.FftShift(new Complex[] { 0, 1, 2, 3 });
        Assert.Equal(new Complex[] { 2, 3, 0, 1 }, c);
    }

    [Fact]
    public void NextFastLen_MatchesScipy()
    {
        int[] targets = [0, 1, 2, 3, 5, 6, 7, 11, 12, 13, 14, 17, 97, 1000, 1001, 1025, 10007, 65537, 100003];
        // scipy 1.17.1 next_fast_len(t) and next_fast_len(t, real=True).
        int[] complexExpected = [0, 1, 2, 3, 5, 6, 7, 11, 12, 14, 14, 18, 98, 1000, 1008, 1029, 10080, 65610, 100352];
        int[] realExpected = [0, 1, 2, 3, 5, 6, 8, 12, 12, 15, 15, 18, 100, 1000, 1024, 1080, 10125, 65610, 101250];
        for (var i = 0; i < targets.Length; i++)
        {
            Assert.Equal(complexExpected[i], Fft.NextFastLen(targets[i]));
            Assert.Equal(realExpected[i], Fft.NextFastLen(targets[i], real: true));
        }
        // prev_fast_len(1000) = 1000, prev_fast_len(10007) = 10000 (both real and complex).
        Assert.Equal(1000, Fft.PrevFastLen(1000));
        Assert.Equal(10000, Fft.PrevFastLen(10007));
        Assert.Equal(10000, Fft.PrevFastLen(10007, real: true));
        // Every result is smooth and minimal over a dense sweep.
        for (var t = 1; t < 3000; t++)
        {
            var r = Fft.NextFastLen(t, real: true);
            Assert.True(r >= t && IsSmooth(r, [2, 3, 5]));
            for (var m = t; m < r; m++) Assert.False(IsSmooth(m, [2, 3, 5]), $"{m} is smooth but {r} was returned for {t}");
        }
    }

    private static bool IsSmooth(int n, int[] primes)
    {
        foreach (var p in primes) while (n % p == 0) n /= p;
        return n == 1;
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(5, 3)]
    [InlineData(100, 37)]
    [InlineData(1000, 999)]
    public void Convolve_MatchesDirectSum(int n, int m)
    {
        var a = FftAssert.RandomReal(n, n);
        var b = FftAssert.RandomReal(m, m + 1);
        var direct = new double[n + m - 1];
        for (var i = 0; i < n; i++)
        for (var j = 0; j < m; j++)
            direct[i + j] += a[i] * b[j];
        FftAssert.Close(direct, Fft.Convolve(a, b), 1e-13, "convolve");

        var corr = new double[n + m - 1];
        for (var k = 0; k < n + m - 1; k++)
        for (var j = 0; j < m; j++)
        {
            var i = j + k - (m - 1);
            if (i >= 0 && i < n) corr[k] += a[i] * b[j];
        }
        FftAssert.Close(corr, Fft.Correlate(a, b), 1e-13, "correlate");
    }

    [Fact]
    public void Convolve2D_MatchesDirectSum()
    {
        var a = new NdArray(FftAssert.RandomReal(5 * 7, 1), 5, 7);
        var b = new NdArray(FftAssert.RandomReal(3 * 4, 2), 3, 4);
        var y = Fft.Convolve(a, b);
        Assert.Equal(new[] { 7, 10 }, y.Shape.ToArray());
        var ad = a.ToArray();
        var bd = b.ToArray();
        var direct = new double[7 * 10];
        for (var i = 0; i < 5; i++)
        for (var j = 0; j < 7; j++)
        for (var p = 0; p < 3; p++)
        for (var q = 0; q < 4; q++)
            direct[(i + p) * 10 + j + q] += ad[i * 7 + j] * bd[p * 4 + q];
        FftAssert.Close(direct, y.ToArray(), 1e-13, "convolve2d");
    }
}
