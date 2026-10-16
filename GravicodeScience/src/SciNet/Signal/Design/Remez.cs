namespace Gravicode.Science.SciNet;

public static partial class Signal
{
    /// <summary>
    /// Minimax (equiripple) linear-phase FIR by the Parks–McClellan / Remez exchange algorithm
    /// (<c>scipy.signal.remez</c>). This follows Janovetz's implementation that scipy wraps: dense
    /// grid, barycentric Lagrange interpolation of the error, extremal exchange with scipy's
    /// alternation-repair rule, then frequency sampling for the taps.
    /// </summary>
    /// <param name="numtaps">Filter length.</param>
    /// <param name="bands">Band edges, flattened pairs, monotonic, from 0 to at most fs/2.</param>
    /// <param name="desired">Desired gain per band (one value per pair).</param>
    /// <param name="weight">Relative weight per band (default ones).</param>
    /// <param name="type">Bandpass, differentiator or Hilbert transformer.</param>
    /// <param name="maxiter">Maximum exchange iterations; like scipy, hitting it returns the current design.</param>
    /// <param name="gridDensity">Grid points per extremal.</param>
    /// <param name="fs">Sampling frequency (default 1, so bands are in cycles/sample up to 0.5).</param>
    /// <exception cref="InvalidOperationException">The exchange lost track of the extremal set (scipy's "too many/few extremals").</exception>
    public static double[] Remez(int numtaps, double[] bands, double[] desired, double[]? weight = null,
        RemezType type = RemezType.Bandpass, int maxiter = 25, int gridDensity = 16, double? fs = null)
    {
        ArgumentNullException.ThrowIfNull(bands);
        ArgumentNullException.ThrowIfNull(desired);
        var hz = fs ?? 1.0;
        if (bands.Length % 2 != 0) throw new ArgumentException("bands must contain frequency pairs.", nameof(bands));
        var numband = bands.Length / 2;
        if (desired.Length != numband) throw new ArgumentException("desired must have one entry per band.", nameof(desired));
        var wt = weight ?? Enumerable.Repeat(1.0, numband).ToArray();
        if (wt.Length != numband) throw new ArgumentException("weight must have one entry per band.", nameof(weight));
        if (numtaps < 3) throw new ArgumentException("numtaps must be at least 3.", nameof(numtaps));
        var b = new double[bands.Length];
        var old = -1.0;
        for (var k = 0; k < bands.Length; k++)
        {
            if (bands[k] < old) throw new ArgumentException("Bands must be monotonic starting at zero.");
            if (bands[k] * 2 > hz) throw new ArgumentException("Band edges should be less than 1/2 the sampling frequency.");
            old = bands[k];
            b[k] = bands[k] / hz;
        }

        var positive = type == RemezType.Bandpass;
        var r = numtaps / 2;
        if (numtaps % 2 == 1 && positive) r++;

        // Dense grid.
        var delf = 0.5 / (gridDensity * r);
        var grid0 = !positive && delf > b[0] ? delf : b[0];
        var grid = new List<double>();
        var d = new List<double>();
        var w = new List<double>();
        for (var band = 0; band < numband; band++)
        {
            var lowf = band == 0 ? grid0 : b[2 * band];
            var highf = b[2 * band + 1];
            // Count from the band edge, not grid0: a shifted first point must not shorten the band.
            var k = (int)((highf - b[2 * band]) / delf + 0.5);
            if (band == 0 && !positive) k--;
            for (var i = 0; i < k; i++)
            {
                d.Add(desired[band]);
                w.Add(wt[band]);
                grid.Add(lowf);
                lowf += delf;
            }
            if (k > 0) grid[^1] = highf;
        }
        var gridsize = grid.Count;
        if (gridsize <= r) throw new ArgumentException("Bands too narrow for the requested number of taps.");
        if (!positive && grid[^1] > 0.5 - delf && numtaps % 2 == 1) grid[^1] = 0.5 - delf;

        var G = grid.ToArray();
        var D = d.ToArray();
        var W = w.ToArray();
        if (type == RemezType.Differentiator)
        {
            // The desired response of a differentiator is desired·f (a slope), weighted by 1/f so
            // the error is relative — evaluated on the final grid, edges included.
            for (var i = 0; i < gridsize; i++) D[i] *= G[i];
            for (var i = 0; i < gridsize; i++)
                if (D[i] > 0.0001) W[i] /= G[i];
        }
        for (var i = 0; i < gridsize; i++)
        {
            double c;
            if (positive)
            {
                if (numtaps % 2 == 1) continue;
                c = Math.Cos(Math.PI * G[i]);
            }
            else c = numtaps % 2 == 1 ? Math.Sin(2 * Math.PI * G[i]) : Math.Sin(Math.PI * G[i]);
            D[i] /= c;
            W[i] *= c;
        }

        var ext = new int[r + 1];
        for (var i = 0; i <= r; i++) ext[i] = (int)((long)i * (gridsize - 1) / r);
        var ad = new double[r + 1];
        var x = new double[r + 1];
        var y = new double[r + 1];
        var e = new double[gridsize];
        for (var iter = 0; iter < maxiter; iter++)
        {
            RemezCalcParms(r, ext, G, D, W, ad, x, y);
            for (var i = 0; i < gridsize; i++) e[i] = W[i] * (D[i] - RemezComputeA(G[i], r, ad, x, y));
            var err = RemezSearch(r, ext, gridsize, e);
            if (err == -2) throw new InvalidOperationException("Remez: too few extremals -- cannot continue.");
            if (err == -3) throw new InvalidOperationException("Remez: too many extremals -- cannot continue.");
            if (RemezIsDone(r, ext, e)) break;
        }
        RemezCalcParms(r, ext, G, D, W, ad, x, y);

        var taps = new double[numtaps / 2 + 1];
        for (var i = 0; i <= numtaps / 2; i++)
        {
            double c;
            if (positive) c = numtaps % 2 == 1 ? 1 : Math.Cos(Math.PI * i / numtaps);
            else c = numtaps % 2 == 1 ? Math.Sin(2 * Math.PI * i / numtaps) : Math.Sin(Math.PI * i / numtaps);
            taps[i] = RemezComputeA((double)i / numtaps, r, ad, x, y) * c;
        }
        var h = RemezFreqSample(numtaps, taps, positive);
        // Odd-symmetric designs (Hilbert, differentiator) come out of the sine-series frequency
        // sampling with the opposite sign to scipy's convention, h[M+1] < 0 for a positive slope.
        if (!positive) for (var i = 0; i < h.Length; i++) h[i] = -h[i];
        return h;
    }

    private static void RemezCalcParms(int r, int[] ext, double[] grid, double[] d, double[] w, double[] ad, double[] x, double[] y)
    {
        for (var i = 0; i <= r; i++) x[i] = Math.Cos(2 * Math.PI * grid[ext[i]]);
        var ld = (r - 1) / 15 + 1; // interleave the product to avoid overflow/underflow
        for (var i = 0; i <= r; i++)
        {
            var denom = 1.0;
            var xi = x[i];
            for (var j = 0; j < ld; j++)
                for (var k = j; k <= r; k += ld)
                    if (k != i) denom *= 2.0 * (xi - x[k]);
            if (Math.Abs(denom) < 0.00001) denom = 0.00001;
            ad[i] = 1.0 / denom;
        }
        double numer = 0, den = 0;
        var sign = 1;
        for (var i = 0; i <= r; i++)
        {
            numer += ad[i] * d[ext[i]];
            den += sign * ad[i] / w[ext[i]];
            sign = -sign;
        }
        var delta = numer / den;
        sign = 1;
        for (var i = 0; i <= r; i++)
        {
            y[i] = d[ext[i]] - sign * delta / w[ext[i]];
            sign = -sign;
        }
    }

    private static double RemezComputeA(double freq, int r, double[] ad, double[] x, double[] y)
    {
        double denom = 0, numer = 0;
        var xc = Math.Cos(2 * Math.PI * freq);
        for (var i = 0; i <= r; i++)
        {
            var c = xc - x[i];
            if (Math.Abs(c) < 1.0e-7)
            {
                numer = y[i];
                denom = 1;
                break;
            }
            c = ad[i] / c;
            denom += c;
            numer += c * y[i];
        }
        return numer / denom;
    }

    private static int RemezSearch(int r, int[] ext, int gridsize, double[] e)
    {
        var found = new int[2 * r + 2];
        var k = 0;
        if ((e[0] > 0.0 && e[0] > e[1]) || (e[0] < 0.0 && e[0] < e[1])) found[k++] = 0;
        for (var i = 1; i < gridsize - 1; i++)
        {
            if ((e[i] >= e[i - 1] && e[i] > e[i + 1] && e[i] > 0.0) || (e[i] <= e[i - 1] && e[i] < e[i + 1] && e[i] < 0.0))
            {
                if (k >= 2 * r) return -3;
                found[k++] = i;
            }
        }
        var jj = gridsize - 1;
        if ((e[jj] > 0.0 && e[jj] > e[jj - 1]) || (e[jj] < 0.0 && e[jj] < e[jj - 1]))
        {
            if (k >= 2 * r) return -3;
            found[k++] = jj;
        }
        if (k < r + 1) return -2;
        var extra = k - (r + 1);
        while (extra > 0)
        {
            var up = e[found[0]] > 0.0;
            var l = 0;
            var alt = true;
            for (var j = 1; j < k; j++)
            {
                if (Math.Abs(e[found[j]]) < Math.Abs(e[found[l]])) l = j;
                if (up && e[found[j]] < 0.0) up = false;
                else if (!up && e[found[j]] > 0.0) up = true;
                else { alt = false; break; }
            }
            if (alt && extra == 1) l = Math.Abs(e[found[k - 1]]) < Math.Abs(e[found[0]]) ? k - 1 : 0;
            for (var j = l; j < k - 1; j++) found[j] = found[j + 1];
            k--;
            extra--;
        }
        for (var i = 0; i <= r; i++) ext[i] = found[i];
        return 0;
    }

    private static bool RemezIsDone(int r, int[] ext, double[] e)
    {
        double min, max;
        min = max = Math.Abs(e[ext[0]]);
        for (var i = 1; i <= r; i++)
        {
            var cur = Math.Abs(e[ext[i]]);
            if (cur < min) min = cur;
            if (cur > max) max = cur;
        }
        return (max - min) / max < 0.0001;
    }

    private static double[] RemezFreqSample(int n, double[] a, bool positive)
    {
        var h = new double[n];
        var m = (n - 1.0) / 2.0;
        for (var i = 0; i < n; i++)
        {
            var x = 2 * Math.PI * (i - m) / n;
            double val;
            if (positive)
            {
                val = a[0];
                var kmax = n % 2 == 1 ? (int)m : n / 2 - 1;
                for (var k = 1; k <= kmax; k++) val += 2.0 * a[k] * Math.Cos(x * k);
            }
            else if (n % 2 == 1)
            {
                val = 0;
                for (var k = 1; k <= (int)m; k++) val += 2.0 * a[k] * Math.Sin(x * k);
            }
            else
            {
                val = a[n / 2] * Math.Sin(Math.PI * (i - m));
                for (var k = 1; k <= n / 2 - 1; k++) val += 2.0 * a[k] * Math.Sin(x * k);
            }
            h[i] = val / n;
        }
        return h;
    }
}
