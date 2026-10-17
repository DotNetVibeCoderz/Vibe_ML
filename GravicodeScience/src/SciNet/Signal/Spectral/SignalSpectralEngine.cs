using System.Numerics;

namespace Gravicode.Science.SciNet;

/// <summary>
/// The segment loop shared by every Welch-type estimator: slice, detrend, window, real FFT.
/// Segments are processed in parallel chunks; each worker owns its FFT buffers (and, when
/// averaging, its own partial sums, combined in a fixed order at the end so the result does not
/// depend on scheduling).
/// </summary>
internal sealed class SignalSpectralEngine
{
    private readonly double[] _win;
    private readonly int _nperseg;
    private readonly int _step;
    private readonly int _nfft;
    private readonly DetrendType? _detrend;
    private readonly Func<double[], double[]>? _custom;
    private readonly bool _onesided;
    private readonly RfftPlan _plan;

    // custom: a user detrend callback; when given it replaces detrend and segments are processed
    // serially, since the callback need not be thread-safe.
    public SignalSpectralEngine(double[] win, int nperseg, int noverlap, int nfft, DetrendType? detrend, bool onesided,
        Func<double[], double[]>? custom = null)
    {
        _custom = custom;
        _win = win;
        _nperseg = nperseg;
        _step = nperseg - noverlap;
        _nfft = nfft;
        _detrend = detrend;
        _onesided = onesided;
        _plan = RfftPlan.Get(nfft);
    }

    /// <summary>Number of frequency bins produced per segment.</summary>
    public int Bins => _onesided ? _nfft / 2 + 1 : _nfft;

    /// <summary>Number of full segments in a signal of length n.</summary>
    public int Segments(int n) => n < _nperseg ? 0 : (n - _nperseg) / _step + 1;

    private sealed class Worker
    {
        public readonly double[] Buf;
        public readonly Complex[] Spec;
        public readonly Complex[] Spec2;
        public readonly Complex[] Scratch;

        public Worker(int nfft, int scratch)
        {
            Buf = new double[nfft];
            Spec = new Complex[nfft / 2 + 1];
            Spec2 = new Complex[nfft / 2 + 1];
            Scratch = new Complex[Math.Max(1, scratch)];
        }
    }

    private void Transform(double[] x, int seg, Worker w, Complex[] spec)
    {
        var buf = w.Buf;
        var start = seg * _step;
        var s = buf.AsSpan(0, _nperseg);
        x.AsSpan(start, _nperseg).CopyTo(s);
        if (_custom is not null)
        {
            var r = _custom(s.ToArray());
            if (r is null || r.Length != _nperseg) throw new ArgumentException($"The detrend callback must return {_nperseg} values.");
            r.AsSpan().CopyTo(s);
        }
        else if (_detrend is { } d) Signal.DetrendInPlace(s, d);
        for (var i = 0; i < _nperseg; i++) buf[i] *= _win[i];
        if (_nfft > _nperseg) Array.Clear(buf, _nperseg, _nfft - _nperseg);
        _plan.Forward(buf, spec, w.Scratch);
    }

    /// <summary>Bin k of the (possibly two-sided) spectrum from the one-sided one.</summary>
    private Complex Bin(Complex[] spec, int k) => k < spec.Length ? spec[k] : Complex.Conjugate(spec[_nfft - k]);

    /// <summary>Runs the chunk bodies; a single chunk runs inline so its exceptions surface unwrapped.</summary>
    private static void ForChunks(int chunks, Action<int> body)
    {
        if (chunks == 1) body(0);
        else Parallel.For(0, chunks, body);
    }

    private (int Chunks, int Per) Chunking(int nseg)
    {
        var work = (long)nseg * _nfft;
        var chunks = work < 1 << 16 || nseg < 4 || _custom is not null ? 1 : Math.Min(Environment.ProcessorCount * 2, nseg);
        return (chunks, (nseg + chunks - 1) / chunks);
    }

    /// <summary>
    /// Σ over segments of <c>conj(X)·Y</c> (or |X|² when <paramref name="y"/> is null), one value per bin.
    /// </summary>
    public Complex[] SumCross(double[] x, double[]? y, int nseg, out double[]? realOnly)
    {
        var bins = Bins;
        var (chunks, per) = Chunking(nseg);
        var partialR = y is null ? new double[chunks][] : null;
        var partialC = y is null ? null : new Complex[chunks][];
        ForChunks(chunks, c =>
        {
            var w = new Worker(_nfft, _plan.ScratchLength);
            var s0 = c * per;
            var s1 = Math.Min(nseg, s0 + per);
            if (y is null)
            {
                var acc = new double[bins];
                for (var s = s0; s < s1; s++)
                {
                    Transform(x, s, w, w.Spec);
                    var sp = w.Spec;
                    for (var k = 0; k < bins; k++)
                    {
                        var v = Bin(sp, k);
                        acc[k] += v.Real * v.Real + v.Imaginary * v.Imaginary;
                    }
                }
                partialR![c] = acc;
            }
            else
            {
                var acc = new Complex[bins];
                for (var s = s0; s < s1; s++)
                {
                    Transform(x, s, w, w.Spec);
                    Transform(y, s, w, w.Spec2);
                    for (var k = 0; k < bins; k++) acc[k] += Complex.Conjugate(Bin(w.Spec, k)) * Bin(w.Spec2, k);
                }
                partialC![c] = acc;
            }
        });
        if (y is null)
        {
            var tot = partialR![0];
            for (var c = 1; c < chunks; c++) for (var k = 0; k < bins; k++) tot[k] += partialR[c][k];
            realOnly = tot;
            return [];
        }
        var totC = partialC![0];
        for (var c = 1; c < chunks; c++) for (var k = 0; k < bins; k++) totC[k] += partialC[c][k];
        realOnly = null;
        return totC;
    }

    /// <summary>Per-segment spectra laid out bins × segments (row-major), optionally crossed with y.</summary>
    public Complex[] AllSegments(double[] x, double[]? y, int nseg, bool power)
    {
        var bins = Bins;
        var result = new Complex[(long)bins * nseg];
        var (chunks, per) = Chunking(nseg);
        ForChunks(chunks, c =>
        {
            var w = new Worker(_nfft, _plan.ScratchLength);
            var s0 = c * per;
            var s1 = Math.Min(nseg, s0 + per);
            for (var s = s0; s < s1; s++)
            {
                Transform(x, s, w, w.Spec);
                if (y is not null)
                {
                    Transform(y, s, w, w.Spec2);
                    for (var k = 0; k < bins; k++) result[(long)k * nseg + s] = Complex.Conjugate(Bin(w.Spec, k)) * Bin(w.Spec2, k);
                }
                else if (power)
                {
                    for (var k = 0; k < bins; k++)
                    {
                        var v = Bin(w.Spec, k);
                        result[(long)k * nseg + s] = v.Real * v.Real + v.Imaginary * v.Imaginary;
                    }
                }
                else
                    for (var k = 0; k < bins; k++) result[(long)k * nseg + s] = Bin(w.Spec, k);
            }
        });
        return result;
    }
}
