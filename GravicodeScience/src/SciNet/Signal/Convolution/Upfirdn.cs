using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Signal
{
    /// <summary>
    /// Upsample by zero insertion, FIR filter, downsample (<c>scipy.signal.upfirdn</c>), computed
    /// polyphase: only the outputs that survive decimation are formed, each as one SIMD dot product
    /// of a contiguous slice of the (padded) input with one reversed polyphase branch of
    /// <paramref name="h"/>. Nothing is allocated per output.
    /// </summary>
    /// <param name="h">FIR coefficients.</param>
    /// <param name="x">Input signal.</param>
    /// <param name="up">Upsampling factor (≥ 1).</param>
    /// <param name="down">Downsampling factor (≥ 1).</param>
    /// <param name="mode">Edge extension of <paramref name="x"/>; the statistic modes are not allowed here.</param>
    /// <param name="cval">Pad value for <see cref="UpfirdnPad.Constant"/>.</param>
    /// <returns><c>⌈((len x − 1)·up + len h) / down⌉</c> samples.</returns>
    public static double[] Upfirdn(double[] h, double[] x, int up = 1, int down = 1, UpfirdnPad mode = UpfirdnPad.Constant, double cval = 0)
    {
        ArgumentNullException.ThrowIfNull(h);
        ArgumentNullException.ThrowIfNull(x);
        var u = new UpfirdnPlan(h, up, down);
        var y = new double[u.OutputLength(x.Length)];
        u.Apply(x, y, mode, cval);
        return y;
    }

    /// <summary><see cref="Upfirdn(double[], double[], int, int, UpfirdnPad, double)"/> along one axis of an n-D array.</summary>
    public static NdArray Upfirdn(double[] h, NdArray x, int up, int down, int axis = -1, UpfirdnPad mode = UpfirdnPad.Constant, double cval = 0)
    {
        ArgumentNullException.ThrowIfNull(h);
        var u = new UpfirdnPlan(h, up, down);
        var n = SignalAxisLines.Length(x, axis);
        return SignalAxisLines.Map(x, axis, u.OutputLength(n), (s, d) => u.Apply(s, d, mode, cval));
    }

    /// <summary>
    /// The polyphase decomposition of a filter for one (up, down) pair; reusable across signals
    /// and thread-safe to apply concurrently.
    /// </summary>
    internal sealed class UpfirdnPlan
    {
        private readonly double[][] _phases; // phase p: h[p], h[p+up], … reversed
        private readonly int _hLen;
        private readonly int _maxPhase;
        private readonly int _up;
        private readonly int _down;

        public UpfirdnPlan(double[] h, int up, int down)
        {
            if (h.Length == 0) throw new ArgumentException("h must be 1-D with non-zero length.", nameof(h));
            if (up < 1 || down < 1) throw new ArgumentException("Both up and down must be >= 1.");
            _up = up;
            _down = down;
            _hLen = h.Length;
            _phases = new double[up][];
            for (var p = 0; p < up; p++)
            {
                var lp = p < h.Length ? (h.Length - p + up - 1) / up : 0;
                var ph = new double[lp];
                for (var j = 0; j < lp; j++) ph[lp - 1 - j] = h[p + j * up];
                _phases[p] = ph;
                _maxPhase = Math.Max(_maxPhase, lp);
            }
        }

        /// <summary>scipy's <c>_output_len</c>: <c>((n − 1)·up + len h − 1) / down + 1</c>.</summary>
        public int OutputLength(int inLen) => (int)(((long)(inLen - 1) * _up + _hLen - 1) / _down + 1);

        public void Apply(ReadOnlySpan<double> x, Span<double> y, UpfirdnPad mode, double cval)
        {
            if (mode is UpfirdnPad.Mean or UpfirdnPad.Median or UpfirdnPad.Minimum or UpfirdnPad.Maximum)
                throw new ArgumentException("Statistic pad types are only valid for ResamplePoly.", nameof(mode));
            var n = x.Length;
            if (n == 0) { y.Clear(); return; }
            var outLen = y.Length;
            var left = _maxPhase - 1;
            var lastBase = (long)(outLen - 1) * _down / _up;
            var right = (int)Math.Max(0, lastBase - (n - 1));
            var padded = new double[left + n + right];
            for (var i = 0; i < left; i++) padded[i] = SignalPadding.Extend(x, i - left, mode, cval);
            x.CopyTo(padded.AsSpan(left));
            for (var i = 0; i < right; i++) padded[left + n + i] = SignalPadding.Extend(x, n + i, mode, cval);
            ApplyPadded(padded, left, y);
        }

        /// <summary>Fills <paramref name="y"/> from a signal already carrying <paramref name="left"/> pad samples.</summary>
        private void ApplyPadded(double[] padded, int left, Span<double> y)
        {
            var outLen = y.Length;
            if ((long)outLen * _maxPhase < 1 << 17 || outLen < 256)
            {
                Range(padded, left, y, 0);
                return;
            }
            var result = new double[outLen];
            var chunks = Math.Min(Environment.ProcessorCount * 4, outLen / 128);
            var per = (outLen + chunks - 1) / chunks;
            Parallel.For(0, chunks, c =>
            {
                var s = c * per;
                var e = Math.Min(outLen, s + per);
                if (s < e) Range(padded, left, result.AsSpan(s, e - s), s);
            });
            result.CopyTo(y);
        }

        private void Range(double[] padded, int left, Span<double> y, int first)
        {
            for (var t = 0; t < y.Length; t++)
            {
                var pos = (long)(first + t) * _down;
                var phase = (int)(pos % _up);
                var baseIdx = (int)(pos / _up) + left; // index of x[base] in padded
                var ph = _phases[phase];
                var lp = ph.Length;
                if (lp == 0) { y[t] = 0; continue; }
                var s = baseIdx - lp + 1;
                y[t] = SignalConvKernels.Dot(padded.AsSpan(s, lp), ph);
            }
        }

    }
}

/// <summary>Edge-extension rules shared by upfirdn, resample_poly and the savgol/medfilt modes.</summary>
internal static class SignalPadding
{
    /// <summary>
    /// The value of <paramref name="x"/> at index <paramref name="idx"/> (which may lie outside
    /// <c>[0, n)</c>) under an extension rule — scipy's <c>_extend_left</c>/<c>_extend_right</c>,
    /// including the multiple-reflection cases a long filter reaches on a short signal. The right
    /// edge is the left edge of the reversed signal for every mode except wrap.
    /// </summary>
    public static double Extend(ReadOnlySpan<double> x, long idx, UpfirdnPad mode, double cval)
    {
        var n = x.Length;
        if (idx >= 0 && idx < n) return x[(int)idx];
        if (mode == UpfirdnPad.Constant) return cval;
        if (mode == UpfirdnPad.Wrap) return x[(int)(((idx % n) + n) % n)];
        if (idx < 0) return Left(x, idx, mode, reversed: false);
        return Left(x, n - 1 - idx, mode, reversed: true);
    }

    private static double Left(ReadOnlySpan<double> x, long idx, UpfirdnPad mode, bool reversed)
    {
        var n = x.Length;
        switch (mode)
        {
            case UpfirdnPad.Edge:
                return At(x, reversed, 0);
            case UpfirdnPad.Symmetric:
                {
                    var k = (-idx - 1) % (2L * n);
                    return k < n ? At(x, reversed, k) : At(x, reversed, n - 1 - (k - n));
                }
            case UpfirdnPad.Reflect:
                {
                    if (n == 1) return At(x, reversed, 0);
                    var k = (-idx - 1) % (2L * (n - 1));
                    return k < n - 1 ? At(x, reversed, k + 1) : At(x, reversed, n - 2 - (k - (n - 1)));
                }
            case UpfirdnPad.Antisymmetric:
                {
                    var k = (-idx - 1) % (2L * n);
                    return k < n ? -At(x, reversed, k) : At(x, reversed, n - 1 - (k - n));
                }
            case UpfirdnPad.Antireflect:
                {
                    if (n == 1) return At(x, reversed, 0);
                    if (-idx < n) return At(x, reversed, 0) - (At(x, reversed, -idx) - At(x, reversed, 0));
                    var le = At(x, reversed, 0) + (At(x, reversed, 0) - At(x, reversed, n - 1)) * ((-idx - 1) / (n - 1));
                    var k = (-idx - 1) % (2L * (n - 1));
                    return k < n - 1 ? le - (At(x, reversed, k + 1) - At(x, reversed, 0)) : le - (At(x, reversed, n - 1) - At(x, reversed, n - 2 - (k - (n - 1))));
                }
            case UpfirdnPad.Smooth:
                return n == 1 ? At(x, reversed, 0) : At(x, reversed, 0) + idx * (At(x, reversed, 1) - At(x, reversed, 0));
            case UpfirdnPad.Line:
                return n == 1 ? At(x, reversed, 0) : At(x, reversed, 0) + idx * (At(x, reversed, n - 1) - At(x, reversed, 0)) / (n - 1);
            default:
                throw new ArgumentException($"Unsupported extension mode {mode}.", nameof(mode));
        }
    }

    private static double At(ReadOnlySpan<double> x, bool reversed, long i) => reversed ? x[(int)(x.Length - 1 - i)] : x[(int)i];
}
