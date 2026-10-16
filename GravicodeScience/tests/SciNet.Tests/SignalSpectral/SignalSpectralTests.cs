using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpectralAssert;

namespace Gravicode.Science.Tests.SciNet;

public class SignalSpectralTests
{
    private const double Fs = 100.0;

    [Fact]
    public void Periodogram_MatchesScipy()
    {
        var x = R("xs");
        var (f, p) = Signal.Periodogram(x, Fs);
        Close("pg_f", f, 1e-14);
        Close("pg", p, 1e-11);
        Close("pg_hann", Signal.Periodogram(x, Fs, "hann").Pxx, 1e-11);
        Close("pg_nfft3000", Signal.Periodogram(x, Fs, nfft: 3000).Pxx, 1e-11);
        Close("pg_nfft1001", Signal.Periodogram(x, Fs, nfft: 1001).Pxx, 1e-11);
        Close("pg_spec", Signal.Periodogram(x, Fs, scaling: SpectralScaling.Spectrum).Pxx, 1e-11);
        Close("pg_two", Signal.Periodogram(x[..101], Fs, returnOnesided: false).Pxx, 1e-11);
        Close("pg_lin", Signal.Periodogram(x, Fs, detrend: DetrendType.Linear).Pxx, 1e-11);
        Close("pg_none", Signal.Periodogram(x[..300], Fs, detrend: null).Pxx, 1e-11);
    }

    [Fact]
    public void Welch_AllOptionsMatchScipy()
    {
        var x = R("xs");
        var (f, p) = Signal.Welch(x, Fs);
        Close("w_f", f, 1e-14);
        Close("w", p, 1e-11);
        Close("w_128_32", Signal.Welch(x, Fs, nperseg: 128, noverlap: 32).Pxx, 1e-11);
        Close("w_nfft300", Signal.Welch(x, Fs, nperseg: 200, nfft: 300).Pxx, 1e-11);
        Close("w_nfft301", Signal.Welch(x, Fs, nperseg: 200, nfft: 301).Pxx, 1e-11);
        Close("w_med", Signal.Welch(x, Fs, nperseg: 256, average: SpectralAverage.Median).Pxx, 1e-11);
        Close("w_med_even", Signal.Welch(x[..1800], Fs, nperseg: 256, noverlap: 0, average: SpectralAverage.Median).Pxx, 1e-11);
        Close("w_two", Signal.Welch(x, Fs, nperseg: 100, returnOnesided: false).Pxx, 1e-11);
        Close("w_spec", Signal.Welch(x, Fs, scaling: SpectralScaling.Spectrum).Pxx, 1e-11);
        Close("w_kaiser", Signal.Welch(x, Fs, ("kaiser", 8.0), nperseg: 150).Pxx, 1e-11);
        Close("w_arr", Signal.Welch(x, Fs, Signal.Hamming(90)).Pxx, 1e-11);
        Close("w_lin", Signal.Welch(x, Fs, nperseg: 333, noverlap: 100, detrend: DetrendType.Linear).Pxx, 1e-11);
        Close("w_short", Signal.Welch(x[..100], Fs).Pxx, 1e-11);
        Close("w_odd", Signal.Welch(x, Fs, nperseg: 127).Pxx, 1e-11);
    }

    [Fact]
    public void Welch_ParsevalForOneBoxcarSegment()
    {
        // Spectrum-scaled periodogram with a boxcar: Σ P·(fs/N) = mean(x²) for a zero-mean signal.
        var x = Signal.Detrend(R("xs"), DetrendType.Constant);
        var (_, p) = Signal.Periodogram(x, Fs, detrend: null);
        var integral = p.Sum() * Fs / x.Length;
        var power = x.Sum(v => v * v) / x.Length;
        Assert.True(Math.Abs(integral - power) < 1e-10 * power);
    }

    [Fact]
    public void Welch_NdAlongAxis_MatchesRows()
    {
        var x = R("xs");
        var two = new double[2 * 1024];
        Array.Copy(x, 0, two, 0, 1024);
        Array.Copy(x, 1024, two, 1024, 1024);
        var (f, p) = Signal.Welch(new NdArray(two, 2, 1024), Fs, nperseg: 128);
        var row1 = Signal.Welch(x[1024..], Fs, nperseg: 128).Pxx;
        Assert.Equal(new[] { 2, f.Length }, p.Shape.ToArray());
        Close(row1, p.AsContiguous().AsSpan().Slice(f.Length, f.Length), 1e-14, "row 1");
    }

    [Fact]
    public void Welch_LongSignal_ParallelPathEqualsSerialDefinition()
    {
        var rng = new Random(4);
        var x = Enumerable.Range(0, 300_000).Select(_ => rng.NextDouble() - 0.5).ToArray();
        var (_, p) = Signal.Welch(x, 1.0, nperseg: 1024, detrend: null, scaling: SpectralScaling.Spectrum);
        // Bin 5 by brute force over the same 584 half-overlapping Hann segments.
        var w = Signal.GetWindow("hann", 1024);
        var wsum = w.Sum();
        var nseg = (x.Length - 1024) / 512 + 1;
        double acc = 0;
        for (var s = 0; s < nseg; s++)
        {
            Complex z = 0;
            for (var i = 0; i < 1024; i++) z += x[s * 512 + i] * w[i] * Complex.Exp(new Complex(0, -2 * Math.PI * 5 * i / 1024));
            acc += z.Real * z.Real + z.Imaginary * z.Imaginary;
        }
        var expect = 2 * acc / nseg / (wsum * wsum);
        Assert.True(Math.Abs(p[5] - expect) < 1e-10 * expect);
    }

    [Fact]
    public void CsdAndCoherence_MatchScipy()
    {
        var x = R("xs");
        var y = R("ys");
        Close("csd", Signal.Csd(x, y, Fs, nperseg: 256).Pxy, 1e-11);
        Close("csd_med", Signal.Csd(x, y, Fs, nperseg: 256, average: SpectralAverage.Median).Pxy, 1e-11);
        Close("csd_unequal", Signal.Csd(x, y[..1500], Fs, nperseg: 128).Pxy, 1e-11);
        Close("csd_two", Signal.Csd(x, y, Fs, nperseg: 64, returnOnesided: false).Pxy, 1e-11);
        var (_, c) = Signal.Coherence(x, y, Fs, nperseg: 128);
        Close("coh", c, 1e-10);
        Assert.All(c, v => Assert.InRange(v, 0.0, 1.0 + 1e-12));
    }

    [Fact]
    public void Spectrogram_AllModesMatchScipy()
    {
        var x = R("xs");
        foreach (var mode in new[] { SpectrogramMode.Psd, SpectrogramMode.Magnitude, SpectrogramMode.Angle, SpectrogramMode.Phase })
        {
            var (f, t, s, _) = Signal.Spectrogram(x, Fs, nperseg: 128, mode: mode);
            Close($"spg_{mode.ToString().ToLowerInvariant()}", s!, 1e-10);
            Close("spg_t", t, 1e-14);
            Close("spg_f", f, 1e-14);
        }
        var (_, _, _, sc) = Signal.Spectrogram(x, Fs, nperseg: 128, mode: SpectrogramMode.Complex);
        Close("spg_complex", sc!, 1e-11);
        var (_, td, sd, _) = Signal.Spectrogram(x, Fs);
        Close("spg_def", sd!, 1e-11);
        Close("spg_def_t", td, 1e-14);
        Close("spg_spec", Signal.Spectrogram(x, Fs, nperseg: 100, noverlap: 30, nfft: 128, detrend: DetrendType.Linear, scaling: SpectralScaling.Spectrum).Sxx!, 1e-11);
    }

    [Fact]
    public void Stft_AllBoundariesMatchScipy()
    {
        var x = R("xs")[..500];
        var (f, t, z) = Signal.Stft(x, Fs, nperseg: 64);
        Close("stft", z, 1e-11);
        Close("stft_t", t, 1e-14);
        Close("stft_f", f, 1e-14);
        foreach (var (name, bd) in new[] { ("even", StftBoundary.Even), ("odd", StftBoundary.Odd), ("constant", StftBoundary.Constant), ("None", StftBoundary.None) })
        {
            var r = Signal.Stft(x, Fs, nperseg: 64, noverlap: 40, boundary: bd);
            Close($"stft_{name}", r.Zxx, 1e-11);
            Close($"stft_{name}_t", r.T, 1e-14);
        }
        Close("stft_nopad", Signal.Stft(x, Fs, nperseg: 64, padded: false).Zxx, 1e-11);
        Close("stft_psd", Signal.Stft(x, Fs, nperseg: 50, scaling: SpectralScaling.Density).Zxx, 1e-11);
        Close("stft_nfft", Signal.Stft(x, Fs, nperseg: 64, nfft: 90, detrend: DetrendType.Constant).Zxx, 1e-11);
    }

    [Fact]
    public void Istft_RoundTripsAndMatchesScipy()
    {
        var x = R("xs")[..500];
        var (_, _, z) = Signal.Stft(x, Fs, nperseg: 64);
        var (t, xr) = Signal.Istft(z, Fs, nperseg: 64);
        Close("istft", xr, 1e-11);
        Close("istft_t", t, 1e-14);
        for (var i = 0; i < x.Length; i++) Assert.True(Math.Abs(xr[i] - x[i]) < 1e-12, $"round trip [{i}]");
        var z2 = Signal.Stft(x, Fs, "hamming", nperseg: 50, noverlap: 30, scaling: SpectralScaling.Density).Zxx;
        Close("istft2", Signal.Istft(z2, Fs, "hamming", nperseg: 50, noverlap: 30, scaling: SpectralScaling.Density).X, 1e-11);
        Close("istft_noboundary", Signal.Istft(z, Fs, nperseg: 64, boundary: false).X, 1e-11);
    }

    [Fact]
    public void ColaAndNola_MatchScipy()
    {
        // scipy 1.17.1: check_COLA / check_NOLA on these arguments.
        Assert.True(Signal.CheckCola("hann", 256, 128));
        Assert.False(Signal.CheckCola("hann", 256, 100));
        Assert.True(Signal.CheckCola("boxcar", 100, 0));
        Assert.True(Signal.CheckCola(("tukey", 0.25), 64, 8));
        Assert.False(Signal.CheckCola("hamming", 50, 30));
        Assert.True(Signal.CheckNola("hann", 256, 128));
        Assert.True(Signal.CheckNola("hann", 256, 100));
        Assert.False(Signal.CheckNola("hann_symmetric", 64, 0));
        Assert.True(Signal.CheckNola(new double[] { 0, 1, 0 }, 3, 1) == false);
    }

    [Fact]
    public void Lombscargle_MatchesScipy()
    {
        var t = R("ls_t");
        var y = R("ls_y");
        var w = R("ls_w");
        var f = R("ls_f");
        Close("ls", Signal.Lombscargle(t, y, f), 1e-10);
        Close("ls_norm", Signal.Lombscargle(t, y, f, normalize: LombscargleNormalization.Normalize), 1e-10);
        Close("ls_fm", Signal.Lombscargle(t, y, f, floatingMean: true), 1e-10);
        Close("ls_fm_norm_w", Signal.Lombscargle(t, y, f, true, LombscargleNormalization.Normalize, w), 1e-10);
        Close("ls_amp", Signal.LombscargleAmplitude(t, y, f), 1e-10);
        Close("ls_amp_fm", Signal.LombscargleAmplitude(t, y, f, true, w), 1e-10);
        var p = Signal.Lombscargle(t, y, f, floatingMean: true);
        Assert.InRange(f[Array.IndexOf(p, p.Max())], 1.6, 1.8); // the 1.7 rad/s sinusoid
    }
}
