using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Signal
{
    /// <summary>
    /// Ricker ("Mexican hat") wavelet of <paramref name="points"/> samples and width
    /// <paramref name="a"/>: <c>A (1 − t²/a²) exp(−t²/(2a²))</c>, <c>A = 2/(√(3a) π^¼)</c>
    /// (the former <c>scipy.signal.ricker</c>, removed in scipy 1.15 and kept private there).
    /// </summary>
    public static double[] Ricker(int points, double a) => RickerCore(points, a);

    private static double[] RickerCore(double points, double a)
    {
        if (!(points > 0)) return [];
        var count = (int)Math.Ceiling(points);
        var amp = 2 / (Math.Sqrt(3 * a) * Math.Pow(Math.PI, 0.25));
        var wsq = a * a;
        var r = new double[count];
        for (var i = 0; i < count; i++)
        {
            var v = i - (points - 1.0) / 2;
            var xsq = v * v;
            r[i] = amp * (1 - xsq / wsq) * Math.Exp(-xsq / (2 * wsq));
        }
        return r;
    }

    /// <summary>
    /// Complex Morlet wavelet for <see cref="CwtComplex"/> (the former <c>scipy.signal.morlet2</c>):
    /// <c>π^(−¼) √(1/s) exp(i w t/s) exp(−(t/s)²/2)</c>, t centred on the window.
    /// </summary>
    /// <param name="m">Length.</param>
    /// <param name="s">Width (scale).</param>
    /// <param name="w">Omega0; 5 gives a nearly admissible wavelet.</param>
    public static Complex[] Morlet2(int m, double s, double w = 5.0)
    {
        var r = new Complex[Math.Max(0, m)];
        var norm = Math.Sqrt(1 / s) * Math.Pow(Math.PI, -0.25);
        for (var i = 0; i < r.Length; i++)
        {
            var x = (i - (m - 1) / 2.0) / s;
            r[i] = norm * Complex.Exp(new Complex(0, w * x)) * Math.Exp(-0.5 * x * x);
        }
        return r;
    }

    /// <summary>
    /// The older complex Morlet wavelet (former <c>scipy.signal.morlet</c>), sampled over
    /// <c>[−2πs, 2πs]</c>; <paramref name="complete"/> subtracts the correction term that makes it admissible.
    /// </summary>
    public static Complex[] Morlet(int m, double w = 5.0, double s = 1.0, bool complete = true)
    {
        var r = new Complex[Math.Max(0, m)];
        for (var i = 0; i < r.Length; i++)
        {
            var x = m == 1 ? -s * 2 * Math.PI : -s * 2 * Math.PI + i * (4 * Math.PI * s) / (m - 1);
            var v = Complex.Exp(new Complex(0, w * x));
            if (complete) v -= Math.Exp(-0.5 * w * w);
            r[i] = v * Math.Exp(-0.5 * x * x) * Math.Pow(Math.PI, -0.25);
        }
        return r;
    }

    /// <summary>
    /// Continuous wavelet transform with a real wavelet (the former <c>scipy.signal.cwt</c>): row
    /// i is <paramref name="data"/> convolved (<see cref="ConvolveMode.Same"/>) with the reversed
    /// wavelet of width <c>widths[i]</c> sampled on <c>⌈min(10·width, n)⌉</c> points.
    /// </summary>
    /// <param name="data">Signal of length n.</param>
    /// <param name="wavelet">(points, width) → samples, e.g. <see cref="Ricker"/>.</param>
    /// <param name="widths">Widths to use.</param>
    /// <returns>len(widths) × n.</returns>
    public static NdArray Cwt(double[] data, Func<int, double, double[]> wavelet, double[] widths)
    {
        ArgumentNullException.ThrowIfNull(wavelet);
        return CwtCore(data, (p, w) => wavelet((int)Math.Ceiling(p), w), widths);
    }

    private static NdArray CwtCore(double[] data, Func<double, double, double[]> wavelet, double[] widths)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(widths);
        var n = data.Length;
        var outp = new double[widths.Length * n];
        for (var i = 0; i < widths.Length; i++)
        {
            var pts = Math.Min(10 * widths[i], n);
            var wv = wavelet(pts, widths[i]);
            Array.Reverse(wv);
            var row = Convolve(data, wv, ConvolveMode.Same);
            Array.Copy(row, 0, outp, i * n, n);
        }
        return new NdArray(outp, widths.Length, n);
    }

    /// <summary>
    /// Continuous wavelet transform with a complex wavelet such as <see cref="Morlet2"/>: row i is
    /// <paramref name="data"/> convolved with the conjugated, reversed wavelet.
    /// </summary>
    public static ComplexNdArray CwtComplex(double[] data, Func<int, double, Complex[]> wavelet, double[] widths)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(wavelet);
        ArgumentNullException.ThrowIfNull(widths);
        var n = data.Length;
        var outp = new Complex[widths.Length * n];
        for (var i = 0; i < widths.Length; i++)
        {
            var pts = (int)Math.Ceiling(Math.Min(10 * widths[i], n));
            var wv = wavelet(pts, widths[i]);
            var re = new double[wv.Length];
            var im = new double[wv.Length];
            for (var k = 0; k < wv.Length; k++)
            {
                var c = Complex.Conjugate(wv[wv.Length - 1 - k]);
                re[k] = c.Real;
                im[k] = c.Imaginary;
            }
            var cr = Convolve(data, re, ConvolveMode.Same);
            var ci = Convolve(data, im, ConvolveMode.Same);
            for (var k = 0; k < n; k++) outp[i * n + k] = new Complex(cr[k], ci[k]);
        }
        return new ComplexNdArray(outp, widths.Length, n);
    }

    /// <summary>
    /// Peak detection by ridge lines in the continuous wavelet transform
    /// (<c>scipy.signal.find_peaks_cwt</c>) — robust to noise for peaks whose width is roughly
    /// known, at the cost of a CWT per width.
    /// </summary>
    /// <param name="vector">Signal.</param>
    /// <param name="widths">Expected peak widths (ascending).</param>
    /// <param name="wavelet">Wavelet (points, width) → samples; default Ricker evaluated exactly as scipy does.</param>
    /// <param name="maxDistances">Per-row maximum column jump of a ridge; default widths/4.</param>
    /// <param name="gapThresh">Rows a ridge may skip; default ⌈widths[0]⌉.</param>
    /// <param name="minLength">Minimum ridge length in rows; default ⌈rows/4⌉.</param>
    /// <param name="minSnr">Minimum signal-to-noise ratio.</param>
    /// <param name="noisePerc">Percentile of the smallest-width row used as the noise floor.</param>
    /// <param name="windowSize">Window for the noise floor; default ⌈n/20⌉.</param>
    public static int[] FindPeaksCwt(double[] vector, double[] widths, Func<int, double, double[]>? wavelet = null, double[]? maxDistances = null,
        double? gapThresh = null, double? minLength = null, double minSnr = 1, double noisePerc = 10, int? windowSize = null)
    {
        ArgumentNullException.ThrowIfNull(vector);
        ArgumentNullException.ThrowIfNull(widths);
        if (widths.Length == 0) throw new ArgumentException("widths must not be empty.", nameof(widths));
        var gap = gapThresh ?? Math.Ceiling(widths[0]);
        var maxd = maxDistances ?? widths.Select(w => w / 4.0).ToArray();
        var cwt = wavelet is null ? CwtCore(vector, RickerCore, widths) : Cwt(vector, wavelet, widths);
        var rows = widths.Length;
        var cols = vector.Length;
        var m = cwt.AsContiguous().ToArray();
        var ridges = IdentifyRidgeLines(m, rows, cols, maxd, gap);
        var minLen = minLength ?? Math.Ceiling(rows / 4.0);
        var ws = windowSize ?? (int)Math.Ceiling(cols / 20.0);
        var hf = ws / 2;
        var odd = ws % 2;
        var noises = new double[cols];
        for (var ind = 0; ind < cols; ind++)
        {
            var s = Math.Max(ind - hf, 0);
            var e = Math.Min(ind + hf + odd, cols);
            noises[ind] = ScoreAtPercentile(m.AsSpan(s, Math.Max(0, e - s)), noisePerc);
        }
        var locs = new List<int>();
        foreach (var (lr, lc) in ridges)
        {
            if (lr.Length < minLen) continue;
            var snr = Math.Abs(m[lr[0] * cols + lc[0]] / noises[lc[0]]);
            if (snr < minSnr) continue;
            locs.Add(lc[0]);
        }
        locs.Sort();
        return locs.ToArray();
    }

    private static double ScoreAtPercentile(ReadOnlySpan<double> a, double per)
    {
        if (a.Length == 0) return double.NaN;
        var v = a.ToArray();
        Array.Sort(v);
        var idx = per / 100.0 * (v.Length - 1);
        var i = (int)Math.Floor(idx);
        if (i == idx || i + 1 >= v.Length) return v[i];
        return v[i] + (v[i + 1] - v[i]) * (idx - i);
    }

    private sealed class RidgeLine
    {
        public readonly List<int> Rows = [];
        public readonly List<int> Cols = [];
        public int Gap;
    }

    /// <summary>scipy's <c>_identify_ridge_lines</c>, line for line.</summary>
    private static List<(int[] Rows, int[] Cols)> IdentifyRidgeLines(double[] m, int rows, int cols, double[] maxDistances, double gapThresh)
    {
        if (maxDistances.Length < rows) throw new ArgumentException("Max_distances must have at least as many rows as matr.");
        var isMax = new bool[rows * cols];
        var anyRow = -1;
        for (var r = 0; r < rows; r++)
            for (var c = 0; c < cols; c++)
            {
                var v = m[r * cols + c];
                var ok = v > m[r * cols + Math.Min(c + 1, cols - 1)] && v > m[r * cols + Math.Max(c - 1, 0)];
                isMax[r * cols + c] = ok;
                if (ok) anyRow = r;
            }
        if (anyRow < 0) return [];
        var startRow = anyRow;
        var ridges = new List<RidgeLine>();
        for (var c = 0; c < cols; c++)
            if (isMax[startRow * cols + c])
            {
                var l = new RidgeLine();
                l.Rows.Add(startRow);
                l.Cols.Add(c);
                ridges.Add(l);
            }
        var final = new List<RidgeLine>();
        for (var row = startRow - 1; row >= 0; row--)
        {
            foreach (var l in ridges) l.Gap++;
            var prev = ridges.Select(l => l.Cols[^1]).ToArray();
            for (var col = 0; col < cols; col++)
            {
                if (!isMax[row * cols + col]) continue;
                RidgeLine? line = null;
                if (prev.Length > 0)
                {
                    var closest = 0;
                    var best = Math.Abs(col - prev[0]);
                    for (var q = 1; q < prev.Length; q++)
                    {
                        var d = Math.Abs(col - prev[q]);
                        if (d < best) { best = d; closest = q; }
                    }
                    if (best <= maxDistances[row]) line = ridges[closest];
                }
                if (line is not null)
                {
                    line.Cols.Add(col);
                    line.Rows.Add(row);
                    line.Gap = 0;
                }
                else
                {
                    var nl = new RidgeLine();
                    nl.Rows.Add(row);
                    nl.Cols.Add(col);
                    ridges.Add(nl);
                }
            }
            for (var ind = ridges.Count - 1; ind >= 0; ind--)
                if (ridges[ind].Gap > gapThresh)
                {
                    final.Add(ridges[ind]);
                    ridges.RemoveAt(ind);
                }
        }
        var outLines = new List<(int[], int[])>();
        foreach (var l in final.Concat(ridges))
        {
            var idx = Enumerable.Range(0, l.Rows.Count).OrderBy(i => l.Rows[i]).ToArray();
            outLines.Add((idx.Select(i => l.Rows[i]).ToArray(), idx.Select(i => l.Cols[i]).ToArray()));
        }
        return outLines;
    }
}
