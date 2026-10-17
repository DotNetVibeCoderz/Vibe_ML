namespace Gravicode.Science.SciNet;

/// <summary>
/// An empirical CDF or survival function, scipy's <c>EmpiricalDistributionFunction</c>: a right-continuous
/// step function through (<see cref="Quantiles"/>, <see cref="Probabilities"/>).
/// </summary>
public sealed class EmpiricalDistributionFunction
{
    private readonly double[]? _n;
    private readonly double[]? _d;
    private readonly double[] _sf;

    internal EmpiricalDistributionFunction(double[] q, double[] p, double[]? n, double[]? d, bool isCdf)
    {
        Quantiles = q;
        Probabilities = p;
        _n = n;
        _d = d;
        IsCdf = isCdf;
        _sf = isCdf ? p.Select(v => 1 - v).ToArray() : p;
    }

    /// <summary>The distinct sample values at which the function steps.</summary>
    public double[] Quantiles { get; }

    /// <summary>The function's value at each of <see cref="Quantiles"/>.</summary>
    public double[] Probabilities { get; }

    /// <summary>True for the CDF, false for the survival function.</summary>
    public bool IsCdf { get; }

    /// <summary>
    /// The step function at <paramref name="x"/>: the value at the largest quantile ≤ x, and 0 (CDF) or
    /// 1 (SF) below the smallest — scipy's <c>interp1d(kind='previous')</c> with ±∞ end points.
    /// </summary>
    public double Evaluate(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (double.IsPositiveInfinity(x)) return IsCdf ? 1.0 : 0.0;
        int lo = 0, hi = Quantiles.Length; // first index with q > x
        while (lo < hi)
        {
            var mid = (lo + hi) >>> 1;
            if (Quantiles[mid] <= x) lo = mid + 1; else hi = mid;
        }
        return lo == 0 ? (IsCdf ? 0.0 : 1.0) : Probabilities[lo - 1];
    }

    /// <summary>Element-wise <see cref="Evaluate(double)"/>.</summary>
    public double[] Evaluate(double[] x)
    {
        ArgumentNullException.ThrowIfNull(x);
        var r = new double[x.Length];
        for (var i = 0; i < x.Length; i++) r[i] = Evaluate(x[i]);
        return r;
    }

    /// <summary>
    /// Pointwise confidence bands at <see cref="Quantiles"/>: Greenwood's formula (<c>"linear"</c>, the
    /// default) or the exponential Greenwood formula (<c>"log-log"</c>), clipped to [0, 1] as scipy.
    /// NaN where the formula is undefined (e.g. the last step of an uncensored sample, where n = d),
    /// which is a property of the formulas, not an error. Bands of bands cannot be computed.
    /// </summary>
    public (EmpiricalDistributionFunction Low, EmpiricalDistributionFunction High) ConfidenceInterval(double confidenceLevel = 0.95, string method = "linear")
    {
        if (_n == null || _d == null) throw new InvalidOperationException("Confidence interval bounds do not implement a confidence_interval method.");
        if (!(confidenceLevel >= 0 && confidenceLevel <= 1)) throw new ArgumentException("confidenceLevel must be a scalar between 0 and 1.", nameof(confidenceLevel));
        var m = (method ?? "").ToLowerInvariant();
        if (m != "linear" && m != "log-log") throw new ArgumentException("method must be one of {'linear', 'log-log'}.", nameof(method));
        var k = Quantiles.Length;
        var z = Special.Ndtri(0.5 + confidenceLevel / 2);
        var low = new double[k];
        var high = new double[k];
        var cum = 0.0;
        for (var i = 0; i < k; i++)
        {
            cum += _d[i] / (_n[i] * (_n[i] - _d[i]));
            var sf = _sf[i];
            if (m == "linear")
            {
                var se = Math.Sqrt(sf * sf * cum);
                low[i] = Probabilities[i] - z * se;
                high[i] = Probabilities[i] + z * se;
            }
            else
            {
                var ls = Math.Log(sf);
                var se = Math.Sqrt(1 / (ls * ls) * cum);
                var lnl = Math.Log(-ls);
                var lo = Math.Exp(-Math.Exp(lnl + z * se));
                var hi = Math.Exp(-Math.Exp(lnl - z * se));
                if (IsCdf) (lo, hi) = (1 - hi, 1 - lo);
                low[i] = lo;
                high[i] = hi;
            }
            low[i] = double.IsNaN(low[i]) ? double.NaN : Math.Clamp(low[i], 0, 1);
            high[i] = double.IsNaN(high[i]) ? double.NaN : Math.Clamp(high[i], 0, 1);
        }
        return (new EmpiricalDistributionFunction(Quantiles, low, null, null, IsCdf),
                new EmpiricalDistributionFunction(Quantiles, high, null, null, IsCdf));
    }

    internal double[] AtRisk => _n!;
    internal double[] Events => _d!;
}

/// <summary>Result of <see cref="Stats.Ecdf(CensoredData)"/> (scipy's <c>ECDFResult</c>).</summary>
/// <param name="Cdf">The empirical CDF.</param>
/// <param name="Sf">The empirical survival function (Kaplan–Meier for right-censored data).</param>
public sealed record EcdfResult(EmpiricalDistributionFunction Cdf, EmpiricalDistributionFunction Sf);

/// <summary>scipy.stats survival analysis: <c>ecdf</c> and <c>logrank</c>.</summary>
public static partial class Stats
{
    /// <summary>The empirical CDF and survival function of an uncensored sample, scipy's <c>ecdf</c>.</summary>
    public static EcdfResult Ecdf(double[] sample)
    {
        ArgumentNullException.ThrowIfNull(sample);
        if (sample.Length == 0) throw new ArgumentException("sample must not be empty.", nameof(sample));
        return Ecdf(new CensoredData(uncensored: sample));
    }

    /// <summary>
    /// Empirical distribution of possibly right-censored data, scipy's <c>ecdf</c>: steps of 1/n for
    /// uncensored data, the Kaplan–Meier product-limit estimator when some values are right-censored.
    /// Tied death and censoring times count the censored subjects as still at risk at that time, as
    /// scipy. Left- and interval-censored data throw <see cref="NotSupportedException"/>, as scipy.
    /// </summary>
    public static EcdfResult Ecdf(CensoredData sample)
    {
        ArgumentNullException.ThrowIfNull(sample);
        if (sample.Left.Length > 0 || sample.Interval.GetLength(0) > 0)
            throw new NotSupportedException("Currently, only uncensored and right-censored data is supported.");
        if (sample.Count == 0) throw new ArgumentException("sample must not be empty.", nameof(sample));
        double[] t, cdf, sf, n, d;
        if (sample.Right.Length == 0)
        {
            var s = (double[])sample.Uncensored.Clone();
            Array.Sort(s);
            var total = s.Length;
            var xs = new List<double>();
            var counts = new List<double>();
            for (var i = 0; i < s.Length;)
            {
                var j = i;
                while (j < s.Length && s[j] == s[i]) j++;
                xs.Add(s[i]);
                counts.Add(j - i);
                i = j;
            }
            var k = xs.Count;
            t = [.. xs];
            d = [.. counts];
            cdf = new double[k];
            sf = new double[k];
            n = new double[k];
            var events = 0.0;
            for (var i = 0; i < k; i++)
            {
                n[i] = total - events;
                events += d[i];
                cdf[i] = events / total;
                sf[i] = 1 - cdf[i];
            }
        }
        else
        {
            var m = sample.Count;
            var times = new double[m];
            var died = new int[m];
            sample.Uncensored.CopyTo(times, 0);
            sample.Right.CopyTo(times, sample.Uncensored.Length);
            for (var i = 0; i < sample.Uncensored.Length; i++) died[i] = 1;
            // Stable sort so that, as numpy's argsort on the concatenation, deaths precede losses at ties
            // (only the per-time totals matter, so the order inside a tie does not change the result).
            var idx = Enumerable.Range(0, m).OrderBy(i => times[i]).ToArray();
            var tl = new List<double>();
            var nl = new List<double>();
            var dl = new List<double>();
            for (var p = 0; p < m;)
            {
                var q = p;
                var deaths = 0;
                while (q < m && times[idx[q]] == times[idx[p]]) { deaths += died[idx[q]]; q++; }
                tl.Add(times[idx[p]]);
                nl.Add(m - p);
                dl.Add(deaths);
                p = q;
            }
            t = [.. tl];
            n = [.. nl];
            d = [.. dl];
            var k = t.Length;
            sf = new double[k];
            cdf = new double[k];
            var prod = 1.0;
            for (var i = 0; i < k; i++)
            {
                prod *= (n[i] - d[i]) / n[i];
                sf[i] = prod;
                cdf[i] = 1 - prod;
            }
        }
        return new EcdfResult(new EmpiricalDistributionFunction(t, cdf, n, d, true), new EmpiricalDistributionFunction(t, sf, n, d, false));
    }

    /// <summary>Logrank test of two uncensored samples; see <see cref="Logrank(CensoredData, CensoredData, Alternative)"/>.</summary>
    public static SignificanceResult Logrank(double[] x, double[] y, Alternative alternative = Alternative.TwoSided)
        => Logrank(new CensoredData(uncensored: x), new CensoredData(uncensored: y), alternative);

    /// <summary>
    /// The logrank (Mantel–Cox) test comparing the survival of two right-censored samples, scipy's
    /// <c>logrank</c>. The statistic is the signed Z = (O_x − E_x)/√V, the square root of the χ²₁
    /// statistic other packages report, so one-sided alternatives are available: <c>Less</c> means
    /// group x's hazard is lower (its survival favoured).
    /// </summary>
    public static SignificanceResult Logrank(CensoredData x, CensoredData y, Alternative alternative = Alternative.TwoSided)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        var xy = new CensoredData(uncensored: [.. x.Uncensored, .. y.Uncensored], right: [.. x.Right, .. y.Right]);
        var res = Ecdf(xy).Sf;
        var resX = Ecdf(x).Sf;
        double sumVar = 0, sumExp = 0;
        var qx = resX.Quantiles;
        for (var i = 0; i < res.Quantiles.Length; i++)
        {
            var dj = res.Events[i];
            if (dj == 0) continue;
            var nj = res.AtRisk[i];
            // At risk in x at this time: x's at-risk count at its first time ≥ t (searchsorted left), 0 past its end.
            int lo = 0, hi = qx.Length;
            while (lo < hi) { var mid = (lo + hi) >>> 1; if (qx[mid] < res.Quantiles[i]) lo = mid + 1; else hi = mid; }
            var nx = lo < qx.Length ? resX.AtRisk[lo] : 0.0;
            var ny = nj - nx;
            if (nj > 1) sumVar += nx * ny * dj * (nj - dj) / (nj * nj * (nj - 1));
            sumExp += nx * (dj / nj);
        }
        var stat = (x.Uncensored.Length - sumExp) / Math.Sqrt(sumVar);
        double p = alternative switch
        {
            Alternative.Less => Special.Ndtr(stat),
            Alternative.Greater => Special.Ndtr(-stat),
            _ => 2 * Special.Ndtr(-Math.Abs(stat)),
        };
        return new SignificanceResult(stat, Math.Min(p, 1.0));
    }
}
