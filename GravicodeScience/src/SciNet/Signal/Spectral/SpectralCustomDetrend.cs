using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Signal
{
    // Overloads taking a detrend callback, scipy's "detrend: callable". The callback receives a copy
    // of one segment (nperseg samples, before windowing) and must return as many values. scipy hands
    // a callable all segments at once, stacked along the last axis; a function that works along the
    // last axis gives the same result either way. Segments are processed serially when a callback is
    // given, since it need not be thread-safe.

    /// <summary><see cref="Welch(double[], double, SpectralWindow, int?, int?, int?, DetrendType?, bool, SpectralScaling, SpectralAverage)"/>
    /// with a custom per-segment detrend callback.</summary>
    public static (double[] F, double[] Pxx) Welch(double[] x, Func<double[], double[]> detrend, double fs = 1.0, SpectralWindow? window = null,
        int? nperseg = null, int? noverlap = null, int? nfft = null, bool returnOnesided = true,
        SpectralScaling scaling = SpectralScaling.Density, SpectralAverage average = SpectralAverage.Mean)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(detrend);
        if (x.Length == 0) return ([], []);
        var (f, pxy, preal) = CsdCore(x, null, fs, window, nperseg, noverlap, nfft, null, returnOnesided, scaling, average, detrend);
        return (f, preal ?? pxy.Select(v => v.Real).ToArray());
    }

    /// <summary><see cref="Csd(double[], double[], double, SpectralWindow, int?, int?, int?, DetrendType?, bool, SpectralScaling, SpectralAverage)"/>
    /// with a custom per-segment detrend callback (applied to both signals).</summary>
    public static (double[] F, Complex[] Pxy) Csd(double[] x, double[] y, Func<double[], double[]> detrend, double fs = 1.0, SpectralWindow? window = null,
        int? nperseg = null, int? noverlap = null, int? nfft = null, bool returnOnesided = true,
        SpectralScaling scaling = SpectralScaling.Density, SpectralAverage average = SpectralAverage.Mean)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        ArgumentNullException.ThrowIfNull(detrend);
        if (x.Length == 0 || y.Length == 0) return ([], []);
        if (x.Length != y.Length)
        {
            var n = Math.Max(x.Length, y.Length);
            if (x.Length < n) { var t = new double[n]; x.CopyTo(t, 0); x = t; }
            if (y.Length < n) { var t = new double[n]; y.CopyTo(t, 0); y = t; }
        }
        var (f, pxy, _) = CsdCore(x, y, fs, window, nperseg, noverlap, nfft, null, returnOnesided, scaling, average, detrend);
        return (f, pxy);
    }

    /// <summary><see cref="Spectrogram(double[], double, SpectralWindow, int?, int?, int?, DetrendType?, bool, SpectralScaling, SpectrogramMode)"/>
    /// with a custom per-segment detrend callback.</summary>
    public static (double[] F, double[] T, NdArray? Sxx, ComplexNdArray? SxxComplex) Spectrogram(double[] x, Func<double[], double[]> detrend,
        double fs = 1.0, SpectralWindow? window = null, int? nperseg = null, int? noverlap = null, int? nfft = null,
        bool returnOnesided = true, SpectralScaling scaling = SpectralScaling.Density, SpectrogramMode mode = SpectrogramMode.Psd)
    {
        ArgumentNullException.ThrowIfNull(detrend);
        return SpectrogramCore(x, fs, window, nperseg, noverlap, nfft, null, detrend, returnOnesided, scaling, mode);
    }

    /// <summary><see cref="Stft(double[], double, SpectralWindow, int, int?, int?, DetrendType?, bool, StftBoundary, bool, SpectralScaling)"/>
    /// with a custom per-segment detrend callback.</summary>
    public static (double[] F, double[] T, ComplexNdArray Zxx) Stft(double[] x, Func<double[], double[]> detrend, double fs = 1.0,
        SpectralWindow? window = null, int nperseg = 256, int? noverlap = null, int? nfft = null, bool returnOnesided = true,
        StftBoundary boundary = StftBoundary.Zeros, bool padded = true, SpectralScaling scaling = SpectralScaling.Spectrum)
    {
        ArgumentNullException.ThrowIfNull(detrend);
        return StftCore(x, fs, window, nperseg, noverlap, nfft, null, detrend, returnOnesided, boundary, padded, scaling);
    }
}
