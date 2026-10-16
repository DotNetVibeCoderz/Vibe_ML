using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Signal
{
    /// <summary>
    /// The first discrete prolate spheroidal sequence (Slepian window) — the window of length
    /// <paramref name="m"/> with the most energy inside the band <c>|f| &lt; NW/M</c>.
    /// </summary>
    /// <param name="m">Window length.</param>
    /// <param name="nw">Standardised half bandwidth, <c>2·NW = BW·f0</c>; must be in (0, M/2).</param>
    /// <param name="sym">Symmetric or periodic.</param>
    /// <param name="norm">Normalisation; scipy's default for a single window is <see cref="DpssNorm.Approximate"/>.</param>
    public static double[] Dpss(int m, double nw, bool sym = true, DpssNorm norm = DpssNorm.Approximate)
    {
        var (w, _) = DpssCore(m, nw, 1, sym, norm, false);
        return w[0];
    }

    /// <summary>
    /// The first <paramref name="kmax"/> DPSS tapers as a <c>kmax × M</c> array, ordered by
    /// decreasing concentration. scipy's default normalisation here is unit energy.
    /// </summary>
    /// <remarks>
    /// Solved as scipy does, via the tridiagonal matrix that commutes with the concentration
    /// operator (Percival &amp; Walden 1993): bisection on its Sturm sequence for the largest
    /// eigenvalues, then inverse iteration for the vectors. Signs follow scipy: even tapers have a
    /// positive sum, odd tapers start positive at their first significant sample.
    /// </remarks>
    public static NdArray Dpss(int m, double nw, int kmax, bool sym = true, DpssNorm norm = DpssNorm.L2)
    {
        var (w, _) = DpssCore(m, nw, kmax, sym, norm, false);
        return Stack(w, m);
    }

    /// <summary>
    /// The first <paramref name="kmax"/> DPSS tapers together with their concentration ratios
    /// (fraction of energy inside the band), as scipy's <c>return_ratios=True</c>.
    /// </summary>
    public static (NdArray Windows, double[] Ratios) DpssWithRatios(int m, double nw, int kmax, bool sym = true, DpssNorm norm = DpssNorm.L2)
    {
        var (w, r) = DpssCore(m, nw, kmax, sym, norm, true);
        return (Stack(w, m), r!);
    }

    private static NdArray Stack(double[][] rows, int m)
    {
        var data = new double[rows.Length * m];
        for (var i = 0; i < rows.Length; i++) Array.Copy(rows[i], 0, data, i * m, m);
        return new NdArray(data, rows.Length, m);
    }

    private static (double[][] Windows, double[]? Ratios) DpssCore(int m, double nw, int kmax, bool sym, DpssNorm norm, bool ratios)
    {
        if (LenGuard(m))
        {
            var ones = new double[kmax][];
            for (var i = 0; i < kmax; i++) ones[i] = Ones(m);
            return (ones, ratios ? Enumerable.Repeat(1.0, kmax).ToArray() : null);
        }
        if (kmax <= 0 || kmax > m) throw new ArgumentException("Kmax must be greater than 0 and less than M.", nameof(kmax));
        if (nw >= m / 2.0) throw new ArgumentException("NW must be less than M/2.", nameof(nw));
        if (nw <= 0) throw new ArgumentException("NW must be positive.", nameof(nw));

        var mm = sym ? m : m + 1;
        var wBand = nw / mm;
        var d = new double[mm];
        var e = new double[mm - 1];
        var c = Math.Cos(2 * Math.PI * wBand);
        for (var i = 0; i < mm; i++)
        {
            var t = (mm - 1 - 2.0 * i) / 2.0;
            d[i] = t * t * c;
        }
        for (var i = 1; i < mm; i++) e[i - 1] = i * (double)(mm - i) / 2.0;

        var windows = new double[kmax][];
        for (var k = 0; k < kmax; k++)
        {
            // k-th largest eigenvalue: index mm-1-k in ascending order.
            var lambda = SignalTridiagonal.Eigenvalue(d, e, mm - 1 - k);
            windows[k] = SignalTridiagonal.InverseIteration(d, e, lambda, windows.AsSpan(0, k));
        }

        // scipy's sign conventions.
        for (var k = 0; k < kmax; k += 2)
        {
            var s = 0.0;
            foreach (var v in windows[k]) s += v;
            if (s < 0) Negate(windows[k]);
        }
        var thresh = Math.Max(1e-7, 1.0 / mm);
        for (var k = 1; k < kmax; k += 2)
        {
            foreach (var v in windows[k])
            {
                if (v * v > thresh)
                {
                    if (v < 0) Negate(windows[k]);
                    break;
                }
            }
        }

        double[]? r = null;
        if (ratios)
        {
            r = new double[kmax];
            var rr = new double[mm];
            for (var i = 0; i < mm; i++) rr[i] = 4 * wBand * Sinc(2 * wBand * i);
            rr[0] = 2 * wBand;
            for (var k = 0; k < kmax; k++)
            {
                var ac = Autocorrelation(windows[k]);
                var s = 0.0;
                for (var i = 0; i < mm; i++) s += ac[i] * rr[i];
                r[k] = s;
            }
        }

        if (norm != DpssNorm.L2)
        {
            var max = double.NegativeInfinity;
            foreach (var row in windows) foreach (var v in row) max = Math.Max(max, v);
            foreach (var row in windows) for (var i = 0; i < mm; i++) row[i] /= max;
            if (mm % 2 == 0)
            {
                double correction;
                if (norm == DpssNorm.Approximate) correction = mm * (double)mm / (mm * (double)mm + nw);
                else
                {
                    // Sub-sample shift of the first taper's spectrum to its true centre.
                    var spec = SignalDft(windows[0].Select(v => new Complex(v, 0)).ToArray());
                    var sum = spec[0].Real;
                    for (var i = 1; i <= mm / 2; i++)
                    {
                        var shift = -(1 - 1.0 / mm) * i;
                        sum += (2 * spec[i] * Complex.Exp(new Complex(0, -Math.PI * shift))).Real;
                    }
                    correction = mm / sum;
                }
                foreach (var row in windows) for (var i = 0; i < mm; i++) row[i] *= correction;
            }
        }
        if (!sym)
            for (var k = 0; k < kmax; k++) windows[k] = windows[k][..^1];
        return (windows, r);
    }

    private static void Negate(double[] v)
    {
        for (var i = 0; i < v.Length; i++) v[i] = -v[i];
    }

    /// <summary>Linear autocorrelation <c>c[k] = Σ x[n]x[n+k]</c>, k = 0..N−1.</summary>
    private static double[] Autocorrelation(double[] x)
    {
        var n = x.Length;
        var c = new double[n];
        if (n <= 256)
        {
            for (var k = 0; k < n; k++)
            {
                var s = 0.0;
                for (var i = 0; i + k < n; i++) s += x[i] * x[i + k];
                c[k] = s;
            }
            return c;
        }
        var len = 1;
        while (len < 2 * n - 1) len <<= 1;
        var buf = new Complex[len];
        for (var i = 0; i < n; i++) buf[i] = x[i];
        var f = SignalDft(buf);
        for (var i = 0; i < len; i++) f[i] = f[i] * Complex.Conjugate(f[i]);
        var inv = SignalIdft(f);
        for (var k = 0; k < n; k++) c[k] = inv[k].Real;
        return c;
    }
}

/// <summary>
/// Eigenpairs of a symmetric tridiagonal matrix by Sturm bisection and inverse iteration — the
/// LAPACK <c>stebz</c>/<c>stein</c> route that <c>scipy.linalg.eigh_tridiagonal(select='i')</c>
/// takes. Selecting a few eigenpairs this way is O(n) each, where a full QL sweep is O(n²) memory.
/// </summary>
internal static class SignalTridiagonal
{
    /// <summary>Number of eigenvalues strictly less than <paramref name="x"/> (Sturm count).</summary>
    private static int CountBelow(double[] d, double[] e, double x, double pivmin)
    {
        var count = 0;
        var q = d[0] - x;
        if (Math.Abs(q) < pivmin) q = -pivmin;
        if (q < 0) count++;
        for (var i = 1; i < d.Length; i++)
        {
            q = d[i] - x - e[i - 1] * e[i - 1] / q;
            if (Math.Abs(q) < pivmin) q = -pivmin;
            if (q < 0) count++;
        }
        return count;
    }

    /// <summary>The eigenvalue with ascending index <paramref name="index"/>, by bisection.</summary>
    public static double Eigenvalue(double[] d, double[] e, int index)
    {
        var n = d.Length;
        double lo = double.PositiveInfinity, hi = double.NegativeInfinity, emax = 0;
        for (var i = 0; i < n; i++)
        {
            var r = (i > 0 ? Math.Abs(e[i - 1]) : 0) + (i < n - 1 ? Math.Abs(e[i]) : 0);
            lo = Math.Min(lo, d[i] - r);
            hi = Math.Max(hi, d[i] + r);
        }
        foreach (var v in e) emax = Math.Max(emax, v * v);
        var pivmin = double.Epsilon * 4 * Math.Max(1.0, emax);
        var norm = Math.Max(Math.Abs(lo), Math.Abs(hi));
        lo -= 2 * double.Epsilon + 1e-15 * norm;
        hi += 2 * double.Epsilon + 1e-15 * norm;
        for (var it = 0; it < 200; it++)
        {
            var mid = 0.5 * (lo + hi);
            if (mid <= lo || mid >= hi) break;
            if (CountBelow(d, e, mid, pivmin) > index) hi = mid; else lo = mid;
            if (hi - lo <= 2.2e-16 * Math.Max(Math.Abs(lo), Math.Abs(hi))) break;
        }
        return 0.5 * (lo + hi);
    }

    /// <summary>
    /// Unit eigenvector for an (accurate) eigenvalue by inverse iteration with a pivoted
    /// tridiagonal LU, orthogonalised against <paramref name="previous"/>.
    /// </summary>
    public static double[] InverseIteration(double[] d0, double[] e, double lambda, ReadOnlySpan<double[]> previous)
    {
        var n = d0.Length;
        var scale = 0.0;
        for (var i = 0; i < n; i++) scale = Math.Max(scale, Math.Abs(d0[i]) + (i < n - 1 ? Math.Abs(e[i]) : 0));
        var tiny = Math.Max(scale, 1e-300) * 1e-15;

        var d = new double[n];
        var dl = new double[n];
        var du = new double[n];
        var du2 = new double[n];
        var piv = new bool[n];
        for (var i = 0; i < n; i++) d[i] = d0[i] - lambda;
        for (var i = 0; i < n - 1; i++) { dl[i] = e[i]; du[i] = e[i]; }

        for (var i = 0; i < n - 1; i++)
        {
            if (Math.Abs(d[i]) >= Math.Abs(dl[i]))
            {
                if (d[i] == 0) d[i] = tiny;
                var fact = dl[i] / d[i];
                dl[i] = fact;
                d[i + 1] -= fact * du[i];
                du2[i] = 0;
            }
            else
            {
                piv[i] = true;
                var fact = d[i] / dl[i];
                d[i] = dl[i];
                dl[i] = fact;
                var temp = du[i];
                du[i] = d[i + 1];
                d[i + 1] = temp - fact * d[i + 1];
                if (i < n - 2)
                {
                    du2[i] = du[i + 1];
                    du[i + 1] = -fact * du[i + 1];
                }
            }
        }
        if (d[n - 1] == 0) d[n - 1] = tiny;

        // Deterministic, non-degenerate start vector.
        var x = new double[n];
        for (var i = 0; i < n; i++) x[i] = 1.0 + 0.1 * Math.Sin(1.7 * i + 0.3);
        for (var iter = 0; iter < 4; iter++)
        {
            foreach (var p in previous)
            {
                var dot = 0.0;
                for (var i = 0; i < n; i++) dot += p[i] * x[i];
                for (var i = 0; i < n; i++) x[i] -= dot * p[i];
            }
            for (var i = 0; i < n - 1; i++)
            {
                if (!piv[i]) x[i + 1] -= dl[i] * x[i];
                else
                {
                    var t = x[i];
                    x[i] = x[i + 1];
                    x[i + 1] = t - dl[i] * x[i];
                }
            }
            x[n - 1] /= d[n - 1];
            if (n > 1) x[n - 2] = (x[n - 2] - du[n - 2] * x[n - 1]) / d[n - 2];
            for (var i = n - 3; i >= 0; i--) x[i] = (x[i] - du[i] * x[i + 1] - du2[i] * x[i + 2]) / d[i];

            var nrm = 0.0;
            foreach (var v in x) nrm += v * v;
            nrm = Math.Sqrt(nrm);
            for (var i = 0; i < n; i++) x[i] /= nrm;
        }
        foreach (var p in previous)
        {
            var dot = 0.0;
            for (var i = 0; i < n; i++) dot += p[i] * x[i];
            for (var i = 0; i < n; i++) x[i] -= dot * p[i];
        }
        var nn = 0.0;
        foreach (var v in x) nn += v * v;
        nn = Math.Sqrt(nn);
        for (var i = 0; i < n; i++) x[i] /= nn;
        return x;
    }
}
