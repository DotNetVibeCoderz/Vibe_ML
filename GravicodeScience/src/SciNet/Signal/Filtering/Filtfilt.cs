using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Signal
{
    /// <summary>
    /// Zero-phase filtering: forward then backward (<c>scipy.signal.filtfilt</c>). The result has
    /// the squared magnitude response and no phase shift.
    /// </summary>
    /// <param name="b">Numerator.</param>
    /// <param name="a">Denominator.</param>
    /// <param name="x">Signal.</param>
    /// <param name="padtype">Edge extension (default odd).</param>
    /// <param name="padlen">Extension length; default <c>3·max(len a, len b)</c>. Must be less than the signal length.</param>
    /// <param name="method">Pad (default) or Gustafsson.</param>
    /// <param name="irlen">Gust only: impulse-response length used to limit the optimisation to the ends.</param>
    public static double[] Filtfilt(double[] b, double[] a, double[] x, FiltfiltPad padtype = FiltfiltPad.Odd, int? padlen = null,
        FiltfiltMethod method = FiltfiltMethod.Pad, int? irlen = null)
    {
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(x);
        if (method == FiltfiltMethod.Gust) return FiltfiltGust(b, a, x, irlen);
        var (edge, ext) = PadSignal(x, padtype, padlen, Math.Max(a.Length, b.Length));
        var zi = LfilterZi(b, a);
        var z = new double[zi.Length];
        for (var i = 0; i < z.Length; i++) z[i] = zi[i] * ext[0];
        var y = new double[ext.Length];
        Lfilter(b, a, ext, y, z);
        var y0 = y[^1];
        Array.Reverse(y);
        for (var i = 0; i < z.Length; i++) z[i] = zi[i] * y0;
        Lfilter(b, a, y, y, z);
        Array.Reverse(y);
        return edge > 0 ? y[edge..^edge] : y;
    }

    /// <summary>Zero-phase filtering of every line of <paramref name="x"/> along <paramref name="axis"/>.</summary>
    public static NdArray Filtfilt(double[] b, double[] a, NdArray x, int axis = -1, FiltfiltPad padtype = FiltfiltPad.Odd, int? padlen = null,
        FiltfiltMethod method = FiltfiltMethod.Pad)
        => ApplyAlongAxis(x, axis, (src, dst) => Filtfilt(b, a, src.ToArray(), padtype, padlen, method).CopyTo(dst));

    /// <summary>
    /// Zero-phase filtering with second-order sections (<c>scipy.signal.sosfiltfilt</c>) — the
    /// numerically safe choice for high-order or narrow-band filters.
    /// </summary>
    public static double[] Sosfiltfilt(NdArray sos, double[] x, FiltfiltPad padtype = FiltfiltPad.Odd, int? padlen = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var s = ValidateSos(sos, out var n).ToArray();
        var ntaps = 2 * n + 1;
        int zb = 0, za = 0;
        for (var i = 0; i < n; i++) { if (s[i * 6 + 2] == 0) zb++; if (s[i * 6 + 5] == 0) za++; }
        ntaps -= Math.Min(zb, za);
        var (edge, ext) = PadSignal(x, padtype, padlen, ntaps);
        var zi = SosfiltZi(sos).ToArray();
        var z = new double[zi.Length];
        for (var i = 0; i < z.Length; i++) z[i] = zi[i] * ext[0];
        var y = new double[ext.Length];
        Sosfilt(s, ext, y, z);
        var y0 = y[^1];
        Array.Reverse(y);
        for (var i = 0; i < z.Length; i++) z[i] = zi[i] * y0;
        Sosfilt(s, y, y, z);
        Array.Reverse(y);
        return edge > 0 ? y[edge..^edge] : y;
    }

    /// <summary>Zero-phase SOS filtering of every line of <paramref name="x"/> along <paramref name="axis"/>.</summary>
    public static NdArray Sosfiltfilt(NdArray sos, NdArray x, int axis = -1, FiltfiltPad padtype = FiltfiltPad.Odd, int? padlen = null)
        => ApplyAlongAxis(x, axis, (src, dst) => Sosfiltfilt(sos, src.ToArray(), padtype, padlen).CopyTo(dst));

    private static (int Edge, double[] Ext) PadSignal(double[] x, FiltfiltPad padtype, int? padlen, int ntaps)
    {
        var edge = padtype == FiltfiltPad.None ? 0 : padlen ?? ntaps * 3;
        if (x.Length <= edge) throw new ArgumentException($"The length of the input vector x must be greater than padlen, which is {edge}.");
        if (padtype == FiltfiltPad.None || edge == 0) return (edge, x.ToArray());
        var n = x.Length;
        var ext = new double[n + 2 * edge];
        Array.Copy(x, 0, ext, edge, n);
        for (var i = 0; i < edge; i++)
        {
            // left: x[edge - i], right: x[n - 2 - i]
            double l = x[edge - i], r = x[n - 2 - i];
            switch (padtype)
            {
                case FiltfiltPad.Odd:
                    ext[i] = 2 * x[0] - l;
                    ext[edge + n + i] = 2 * x[n - 1] - r;
                    break;
                case FiltfiltPad.Even:
                    ext[i] = l;
                    ext[edge + n + i] = r;
                    break;
                default:
                    ext[i] = x[0];
                    ext[edge + n + i] = x[n - 1];
                    break;
            }
        }
        return (edge, ext);
    }

    /// <summary>
    /// Gustafsson's method (Gustafsson 1996): choose the forward and backward initial states by
    /// least squares so that forward-backward and backward-forward filtering agree.
    /// </summary>
    private static double[] FiltfiltGust(double[] b, double[] a, double[] x, int? irlen)
    {
        var order = Math.Max(b.Length, a.Length) - 1;
        if (order == 0)
        {
            var sc = Math.Pow(b[0] / a[0], 2);
            return x.Select(v => sc * v).ToArray();
        }
        var n = x.Length;
        var m = irlen is int ir && n > 2 * ir ? ir : n;
        // Observability matrix: response to unit initial state in each delay.
        var obs = new double[m, order];
        var zi = new double[order];
        zi[0] = 1;
        var (o0, _) = Lfilter(b, a, new double[m], zi);
        for (var i = 0; i < m; i++) obs[i, 0] = o0[i];
        for (var k = 1; k < order; k++)
            for (var i = k; i < m; i++) obs[i, k] = obs[i - k, 0];
        // S = lfilter(b, a, Obs[::-1]) column-wise; Sr = S[::-1]; Obsr = Obs[::-1]
        var S = new double[m, order];
        for (var k = 0; k < order; k++)
        {
            var col = new double[m];
            for (var i = 0; i < m; i++) col[i] = obs[m - 1 - i, k];
            var f = Lfilter(b, a, col);
            for (var i = 0; i < m; i++) S[i, k] = f[i];
        }
        var rows = m == n ? m : 2 * m;
        var cols = 2 * order;
        var M = new double[rows * cols];
        var W = new double[rows, cols];
        for (var i = 0; i < m; i++)
            for (var k = 0; k < order; k++)
            {
                var sr = S[m - 1 - i, k];
                var obsr = obs[m - 1 - i, k];
                if (m == n)
                {
                    M[i * cols + k] = sr - obs[i, k];
                    M[i * cols + order + k] = obsr - S[i, k];
                    W[i, k] = sr;
                    W[i, order + k] = obsr;
                }
                else
                {
                    M[i * cols + k] = sr - obs[i, k];
                    M[(m + i) * cols + order + k] = obsr - S[i, k];
                    W[i, k] = sr;
                    W[m + i, order + k] = obsr;
                }
            }
        var yf = Lfilter(b, a, x);
        var yfb = Lfilter(b, a, yf.Reverse().ToArray()); Array.Reverse(yfb);
        var yb = Lfilter(b, a, x.Reverse().ToArray()); Array.Reverse(yb);
        var ybf = Lfilter(b, a, yb);
        var delta = new double[rows];
        if (m == n) for (var i = 0; i < n; i++) delta[i] = ybf[i] - yfb[i];
        else
        {
            for (var i = 0; i < m; i++) delta[i] = ybf[i] - yfb[i];
            for (var i = 0; i < m; i++) delta[m + i] = ybf[n - m + i] - yfb[n - m + i];
        }
        var ic = Decomposition.Qr(new NdArray(M, rows, cols)).Solve(new NdArray(delta, rows)).ToArray();
        var wic = new double[rows];
        for (var i = 0; i < rows; i++)
        {
            var s = 0.0;
            for (var k = 0; k < cols; k++) s += ic[k] * W[i, k];
            wic[i] = s;
        }
        var y = yfb;
        if (m == n) for (var i = 0; i < n; i++) y[i] += wic[i];
        else
        {
            for (var i = 0; i < m; i++) y[i] += wic[i];
            for (var i = 0; i < m; i++) y[n - m + i] += wic[m + i];
        }
        return y;
    }
}
