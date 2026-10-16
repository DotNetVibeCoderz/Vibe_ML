using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Trend removed by <see cref="Signal.Detrend(double[], DetrendType, int[])"/>.</summary>
public enum DetrendType
{
    /// <summary>Least-squares straight line (per segment between breakpoints).</summary>
    Linear,
    /// <summary>The mean.</summary>
    Constant,
}

/// <summary>What <see cref="Signal.Envelope"/> returns besides the envelope.</summary>
public enum EnvelopeResidual
{
    /// <summary>Nothing.</summary>
    None,
    /// <summary>The part of the spectrum below the band (scipy's default).</summary>
    Lowpass,
    /// <summary>Everything outside the band.</summary>
    All,
}

public static partial class Signal
{
    // ================================================================== Hilbert

    /// <summary>
    /// The analytic signal <c>x + i·H{x}</c> (<c>scipy.signal.hilbert</c>), by zeroing the negative
    /// frequencies of an <paramref name="n"/>-point FFT and doubling the positive ones. The real
    /// part reproduces <paramref name="x"/>; the magnitude is the envelope.
    /// </summary>
    /// <param name="x">Real signal.</param>
    /// <param name="n">FFT length; default <c>len x</c> (shorter truncates, longer zero-pads).</param>
    public static Complex[] Hilbert(double[] x, int? n = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var len = n ?? x.Length;
        if (len <= 0) throw new ArgumentException("N must be positive.", nameof(n));
        var xf = Fft.Fft1D(x, len);
        HilbertMask(xf);
        FftPlan.Get(len).ExecuteInPlace(xf, inverse: true);
        var s = 1.0 / len;
        for (var i = 0; i < len; i++) xf[i] *= s;
        return xf;
    }

    /// <summary><see cref="Hilbert(double[], int?)"/> along one axis of an n-D array.</summary>
    public static ComplexNdArray Hilbert(NdArray x, int? n = null, int axis = -1)
    {
        ArgumentNullException.ThrowIfNull(x);
        var rank = x.Rank;
        var ax = axis < 0 ? axis + rank : axis;
        if (ax < 0 || ax >= rank) throw new ArgumentException("axis out of range.", nameof(axis));
        var shape = x.Shape.ToArray();
        var len = shape[ax];
        var outLen = n ?? len;
        if (outLen <= 0) throw new ArgumentException("N must be positive.", nameof(n));
        var src = x.AsContiguous().ToArray();
        var outer = 1;
        for (var i = 0; i < ax; i++) outer *= shape[i];
        var inner = 1;
        for (var i = ax + 1; i < rank; i++) inner *= shape[i];
        var oshape = (int[])shape.Clone();
        oshape[ax] = outLen;
        var dst = new Complex[outer * inner * outLen];
        Parallel.For(0, outer * inner, new ParallelOptions { MaxDegreeOfParallelism = (long)outer * inner * outLen > 1 << 15 ? -1 : 1 }, li =>
        {
            var o = li / inner;
            var ii = li % inner;
            var line = new double[len];
            for (var k = 0; k < len; k++) line[k] = src[(o * len + k) * inner + ii];
            var h = Hilbert(line, outLen);
            for (var k = 0; k < outLen; k++) dst[(o * outLen + k) * inner + ii] = h[k];
        });
        return new ComplexNdArray(dst, oshape);
    }

    private static void HilbertMask(Span<Complex> xf)
    {
        var n = xf.Length;
        var half = n % 2 == 0 ? n / 2 : (n + 1) / 2;
        for (var i = 1; i < half; i++) xf[i] *= 2;
        var zeroFrom = n % 2 == 0 ? n / 2 + 1 : (n + 1) / 2;
        for (var i = zeroFrom; i < n; i++) xf[i] = Complex.Zero;
    }

    /// <summary>
    /// 2-D analytic signal (<c>scipy.signal.hilbert2</c>) over the last two axes: the 2-D spectrum
    /// is kept only in the first quadrant, with the positive frequencies on each axis doubled.
    /// </summary>
    /// <param name="x">Real array of rank ≥ 2 (rank 1 is treated as a single row).</param>
    /// <param name="n">FFT shape; default the input's last two lengths.</param>
    public static ComplexNdArray Hilbert2(NdArray x, (int Rows, int Cols)? n = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Rank == 1) x = new NdArray(x.AsContiguous().ToArray(), 1, x.Shape[0]);
        var rank = x.Rank;
        var r = n?.Rows ?? x.Shape[rank - 2];
        var c = n?.Cols ?? x.Shape[rank - 1];
        if (r <= 0 || c <= 0) throw new ArgumentException("N must hold exactly two positive integers.", nameof(n));
        var xf = Fft.Fft2(x, [r, c], [rank - 2, rank - 1]);
        var data = xf.Data;
        var k0 = (r + 1) / 2;
        var k1 = (c + 1) / 2;
        var block = r * c;
        for (var b = 0; b < data.Length / block; b++)
            for (var i = 0; i < r; i++)
                for (var j = 0; j < c; j++)
                {
                    ref var v = ref data[b * block + i * c + j];
                    if (i >= k0 || j >= k1) { v = Complex.Zero; continue; }
                    if (k0 > 1 && i >= 1) v *= 2;
                    if (k1 > 1 && j >= 1) v *= 2;
                }
        return Fft.Ifft2(xf, null, [rank - 2, rank - 1]);
    }

    // ================================================================== envelope

    /// <summary>
    /// Envelope of a real signal from its band-limited analytic signal (<c>scipy.signal.envelope</c>,
    /// scipy 1.16+), optionally resampled to <paramref name="nOut"/> samples, plus the residual.
    /// </summary>
    /// <param name="z">Real signal of length n.</param>
    /// <param name="bpStart">First FFT bin of the band (default 1, i.e. drop DC); negative bins
    /// count from −n/2, null means −n/2.</param>
    /// <param name="bpStop">One past the last bin; null means ⌈n/2⌉.</param>
    /// <param name="nOut">Output length; default n.</param>
    /// <param name="squared">Return the squared envelope.</param>
    /// <param name="residual">Which residual to return (null array for <see cref="EnvelopeResidual.None"/>).</param>
    public static (double[] Envelope, double[]? Residual) Envelope(double[] z, int? bpStart = 1, int? bpStop = null, int? nOut = null,
        bool squared = false, EnvelopeResidual residual = EnvelopeResidual.Lowpass)
    {
        ArgumentNullException.ThrowIfNull(z);
        var n = z.Length;
        if (n == 0) throw new ArgumentException("z must not be empty.", nameof(z));
        var no = nOut ?? n;
        if (no <= 0) throw new ArgumentException("n_out must be positive.", nameof(nOut));
        var fak = (double)no / n;
        var start = bpStart ?? -(n / 2);
        var stop = bpStop ?? (n + 1) / 2;
        var negHalf = (int)Math.Floor(-n / 2.0);
        if (!(negHalf <= start && start < stop && stop <= (n + 1) / 2))
            throw new ArgumentException("-n//2 <= bp_in[0] < bp_in[1] <= (n+1)//2 does not hold.");
        var Z = new Complex[n];
        Fft.Rfft(z).CopyTo(Z, 0);
        if (start > 0) { var (lo, hi) = PySlice(n, start, stop); for (var i = lo; i < hi; i++) Z[i] *= 2; }
        else if (stop > 0) for (var i = 1; i < Math.Min(stop, n); i++) Z[i] *= 2;

        Complex[] band;
        var straddles = start <= 0 && 0 < stop;
        if (!straddles)
        {
            var (lo, hi) = PySlice(n, start, stop);
            band = Z.AsSpan(lo, hi - lo).ToArray();
        }
        else
        {
            var shifted = Fft.FftShift(Z);
            var (lo, hi) = PySlice(n, start + n / 2, stop + n / 2);
            band = shifted.AsSpan(lo, hi - lo).ToArray();
        }
        var bb = Fft.Ifft(band, no);
        var env = new double[no];
        for (var i = 0; i < no; i++)
        {
            var v = bb[i] * fak;
            env[i] = squared ? v.Real * v.Real + v.Imaginary * v.Imaginary : Complex.Abs(v);
        }
        if (residual == EnvelopeResidual.None) return (env, null);

        if (!straddles) { var (lo, hi) = PySlice(n, start, stop); for (var i = lo; i < hi; i++) Z[i] = 0; }
        else
        {
            var (lo, hi) = PySlice(n, null, stop);
            for (var i = lo; i < hi; i++) Z[i] = 0;
            (lo, hi) = PySlice(n, start, null);
            for (var i = lo; i < hi; i++) Z[i] = 0;
        }
        if (residual == EnvelopeResidual.Lowpass)
        {
            if (stop > 0) { var (lo, hi) = PySlice(n, stop, (n + 1) / 2); for (var i = lo; i < hi; i++) Z[i] = 0; }
            else
            {
                var (lo, hi) = PySlice(n, start, null);
                for (var i = lo; i < hi; i++) Z[i] = 0;
                (lo, hi) = PySlice(n, 0, (n + 1) / 2);
                for (var i = lo; i < hi; i++) Z[i] = 0;
            }
        }
        var m = Math.Min(n, no);
        if (no != n && m % 2 == 0) Z[m / 2] *= no < n ? 2.0 : 0.5;
        var bins = no / 2 + 1;
        var spec = new Complex[bins];
        Array.Copy(Z, spec, Math.Min(bins, n));
        var res = Fft.Irfft(spec, no);
        for (var i = 0; i < no; i++) res[i] *= fak;
        return (env, res);
    }

    /// <summary>numpy slice bounds <c>[start:stop]</c> on a length-n axis.</summary>
    private static (int Lo, int Hi) PySlice(int n, int? start, int? stop)
    {
        static int Norm(int v, int n) => v < 0 ? Math.Max(0, v + n) : Math.Min(v, n);
        var lo = start is { } s ? Norm(s, n) : 0;
        var hi = stop is { } e ? Norm(e, n) : n;
        return (lo, Math.Max(lo, hi));
    }

    // ================================================================== detrend

    /// <summary>
    /// Removes a linear (per segment) or constant trend (<c>scipy.signal.detrend</c>).
    /// </summary>
    /// <remarks>Each segment's line is fitted in centred form, <c>y − ȳ − β(t − t̄)</c>, which
    /// is the same projection as scipy's <c>lstsq</c> on <c>[t/N, 1]</c> but avoids the
    /// cancellation of an uncentred normal equation on long segments.</remarks>
    /// <param name="data">Signal.</param>
    /// <param name="type">Linear or constant.</param>
    /// <param name="bp">Breakpoints for the linear fit; a separate line is fitted between consecutive breakpoints.</param>
    public static double[] Detrend(double[] data, DetrendType type = DetrendType.Linear, int[]? bp = null)
    {
        ArgumentNullException.ThrowIfNull(data);
        var y = (double[])data.Clone();
        DetrendInPlace(y, type, bp);
        return y;
    }

    /// <summary><see cref="Detrend(double[], DetrendType, int[])"/> along one axis of an n-D array.</summary>
    public static NdArray Detrend(NdArray data, int axis = -1, DetrendType type = DetrendType.Linear, int[]? bp = null)
    {
        var n = SignalAxisLines.Length(data, axis);
        return SignalAxisLines.Map(data, axis, n, (s, d) => { s.CopyTo(d); DetrendInPlace(d, type, bp); });
    }

    internal static void DetrendInPlace(Span<double> y, DetrendType type, int[]? bp = null)
    {
        var n = y.Length;
        if (n == 0) return;
        if (type == DetrendType.Constant)
        {
            double s = 0;
            foreach (var v in y) s += v;
            var mean = s / n;
            for (var i = 0; i < n; i++) y[i] -= mean;
            return;
        }
        var points = new SortedSet<int> { 0, n };
        if (bp is not null)
            foreach (var b in bp)
            {
                if (b > n) throw new ArgumentException("Breakpoints must be less than length of data along given axis.", nameof(bp));
                if (b < 0) throw new ArgumentException("Breakpoints must be non-negative.", nameof(bp));
                points.Add(b);
            }
        var arr = points.ToArray();
        for (var m = 0; m < arr.Length - 1; m++) DetrendLinearSegment(y.Slice(arr[m], arr[m + 1] - arr[m]));
    }

    private static void DetrendLinearSegment(Span<double> y)
    {
        var n = y.Length;
        if (n == 0) return;
        var tMean = (n - 1) / 2.0;
        double yMean = 0;
        foreach (var v in y) yMean += v;
        yMean /= n;
        double sxy = 0, sxx = 0;
        for (var i = 0; i < n; i++)
        {
            var dt = i - tMean;
            sxy += dt * (y[i] - yMean);
            sxx += dt * dt;
        }
        var slope = sxx == 0 ? 0 : sxy / sxx;
        for (var i = 0; i < n; i++) y[i] -= yMean + slope * (i - tMean);
    }

    // ================================================================== vector strength

    /// <summary>
    /// Vector strength and phase of an event train at one period (<c>scipy.signal.vectorstrength</c>):
    /// the length and angle of the mean of <c>exp(2πi·t/period)</c>. 1 means perfectly phase-locked.
    /// </summary>
    public static (double Strength, double Phase) Vectorstrength(double[] events, double period)
    {
        var (s, p) = Vectorstrength(events, [period]);
        return (s[0], p[0]);
    }

    /// <summary>Vector strength and phase for several periods at once.</summary>
    public static (double[] Strength, double[] Phase) Vectorstrength(double[] events, double[] periods)
    {
        ArgumentNullException.ThrowIfNull(events);
        ArgumentNullException.ThrowIfNull(periods);
        var strength = new double[periods.Length];
        var phase = new double[periods.Length];
        for (var k = 0; k < periods.Length; k++)
        {
            if (!(periods[k] > 0)) throw new ArgumentException("periods must be positive.", nameof(periods));
            double re = 0, im = 0;
            var w = 2 * Math.PI / periods[k];
            foreach (var t in events)
            {
                var (sn, cs) = Math.SinCos(w * t);
                re += cs;
                im += sn;
            }
            re /= events.Length;
            im /= events.Length;
            strength[k] = Math.Sqrt(re * re + im * im);
            phase[k] = Math.Atan2(im, re);
        }
        return (strength, phase);
    }

    // ================================================================== unwrap

    /// <summary>
    /// Phase unwrapping (<c>numpy.unwrap</c>): adds multiples of <paramref name="period"/> wherever
    /// consecutive samples jump by more than <paramref name="discont"/> (default half the period).
    /// </summary>
    public static double[] UnwrapPhase(double[] p, double? discont = null, double period = 2 * Math.PI)
    {
        ArgumentNullException.ThrowIfNull(p);
        var y = (double[])p.Clone();
        UnwrapInPlace(y, discont, period);
        return y;
    }

    internal static void UnwrapInPlace(Span<double> p, double? discont = null, double period = 2 * Math.PI)
    {
        if (p.Length < 2) return;
        var half = period / 2;
        var disc = Math.Max(discont ?? half, half);
        double correction = 0;
        var prev = p[0];
        for (var i = 1; i < p.Length; i++)
        {
            var orig = p[i];
            var dd = orig - prev;
            // numpy: ddmod = mod(dd + half, period) - half, with the boundary case kept on the sign of dd.
            var ddmod = ((dd + half) % period + period) % period - half;
            if (ddmod == -half && dd > 0) ddmod = half;
            var corr = ddmod - dd;
            if (Math.Abs(dd) < disc) corr = 0;
            correction += corr;
            prev = orig;
            p[i] = orig + correction;
        }
    }
}
