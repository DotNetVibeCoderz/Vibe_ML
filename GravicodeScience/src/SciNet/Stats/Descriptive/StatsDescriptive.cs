using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>scipy.stats descriptive statistics: moments, means, trimmed statistics and scales.</summary>
public static partial class Stats
{
    // ------------------------------------------------------------------ moments

    /// <summary>
    /// Central moment of the given order, scipy's <c>moment</c>: mean((x − c)^order) about
    /// <paramref name="center"/> (the sample mean when null). Order 0 is 1; order 1 about the mean is 0.
    /// </summary>
    public static double Moment(double[] a, int order = 1, double? center = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (order < 0) throw new ArgumentException("order must be non-negative.", nameof(order));
        if (a.Length == 0) return double.NaN;
        if (order == 0) return 1.0;
        if (order == 1 && center is null) return 0.0;
        var c = center ?? StatsTestMath.Mean(a);
        return CentralMoment(a, c, order);
    }

    /// <summary><see cref="Moment(double[], int, double?)"/> along an axis.</summary>
    public static NdArray Moment(NdArray a, int order = 1, int axis = 0, double? center = null)
        => StatsAxis.Reduce(a, axis, x => Moment(x, order, center));

    private static double CentralMoment(ReadOnlySpan<double> a, double c, int order)
    {
        var buf = new double[a.Length];
        for (var i = 0; i < a.Length; i++)
        {
            var d = a[i] - c;
            var p = order switch { 1 => d, 2 => d * d, 3 => d * d * d, 4 => d * d * d * d, _ => Math.Pow(d, order) };
            buf[i] = p;
        }
        return StatsTestMath.Mean(buf);
    }

    /// <summary>
    /// Sample skewness, scipy's <c>skew</c>: m₃/m₂^{3/2}, or the adjusted Fisher–Pearson G₁ when
    /// <paramref name="bias"/> is false. A sample whose spread is lost in the rounding of its mean
    /// returns NaN rather than a huge meaningless ratio (scipy's precision check).
    /// </summary>
    public static double Skew(double[] a, bool bias = true)
    {
        ArgumentNullException.ThrowIfNull(a);
        var n = a.Length;
        if (n == 0) return double.NaN;
        var mean = StatsTestMath.Mean(a);
        var m2 = CentralMoment(a, mean, 2);
        var m3 = CentralMoment(a, mean, 3);
        var eps = Math.Pow(2, -52);
        if (m2 <= (eps * mean) * (eps * mean)) return double.NaN;
        var v = m3 / Math.Pow(m2, 1.5);
        if (!bias && n > 2) v = Math.Sqrt((n - 1.0) * n) / (n - 2.0) * v;
        return v;
    }

    /// <summary><see cref="Skew(double[], bool)"/> along an axis.</summary>
    public static NdArray Skew(NdArray a, int axis = 0, bool bias = true) => StatsAxis.Reduce(a, axis, x => Skew(x, bias));

    /// <summary>
    /// Sample kurtosis, scipy's <c>kurtosis</c>: m₄/m₂², minus 3 when <paramref name="fisher"/> (the
    /// default, so the normal scores 0). <paramref name="bias"/> false applies the k-statistic correction.
    /// </summary>
    public static double Kurtosis(double[] a, bool fisher = true, bool bias = true)
    {
        ArgumentNullException.ThrowIfNull(a);
        var n = a.Length;
        if (n == 0) return double.NaN;
        var mean = StatsTestMath.Mean(a);
        var m2 = CentralMoment(a, mean, 2);
        var m4 = CentralMoment(a, mean, 4);
        var eps = Math.Pow(2, -52);
        if (m2 <= (eps * mean) * (eps * mean)) return double.NaN;
        var v = m4 / (m2 * m2);
        if (!bias && n > 3)
            v = 1.0 / (n - 2) / (n - 3) * ((n * (double)n - 1.0) * m4 / (m2 * m2) - 3 * (n - 1.0) * (n - 1.0)) + 3.0;
        return fisher ? v - 3 : v;
    }

    /// <summary><see cref="Kurtosis(double[], bool, bool)"/> along an axis.</summary>
    public static NdArray Kurtosis(NdArray a, int axis = 0, bool fisher = true, bool bias = true)
        => StatsAxis.Reduce(a, axis, x => Kurtosis(x, fisher, bias));

    /// <summary>
    /// Summary statistics, scipy's <c>describe</c>: count, extremes, mean, variance (ddof 1 by default),
    /// skewness and kurtosis (biased by default, as scipy).
    /// </summary>
    public static DescribeResult Describe(double[] a, int ddof = 1, bool bias = true)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Length == 0) throw new ArgumentException("describe needs at least one observation.", nameof(a));
        return new DescribeResult(a.Length, a.Min(), a.Max(), StatsTestMath.Mean(a), StatsTestMath.Var(a, ddof),
            Skew(a, bias), Kurtosis(a, true, bias));
    }

    /// <summary>Coefficient of variation std/mean (scipy's <c>variation</c>, ddof 0 by default).</summary>
    public static double Variation(double[] a, int ddof = 0)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Length == 0) return double.NaN;
        return Math.Sqrt(StatsTestMath.Var(a, ddof)) / StatsTestMath.Mean(a);
    }

    /// <summary><see cref="Variation(double[], int)"/> along an axis.</summary>
    public static NdArray Variation(NdArray a, int axis = 0, int ddof = 0) => StatsAxis.Reduce(a, axis, x => Variation(x, ddof));

    // ------------------------------------------------------------------ means

    /// <summary>Weighted geometric mean exp(Σ wᵢ log aᵢ / Σ wᵢ), scipy's <c>gmean</c>.</summary>
    public static double Gmean(double[] a, double[]? weights = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        var logs = new double[a.Length];
        for (var i = 0; i < a.Length; i++) logs[i] = Math.Log(a[i]);
        return Math.Exp(WeightedMean(logs, weights));
    }

    /// <summary><see cref="Gmean(double[], double[])"/> along an axis.</summary>
    public static NdArray Gmean(NdArray a, int axis = 0) => StatsAxis.Reduce(a, axis, x => Gmean(x));

    /// <summary>
    /// Weighted harmonic mean Σwᵢ / Σ(wᵢ/aᵢ), scipy's <c>hmean</c>; a zero entry gives 0. Negative
    /// values are rejected, as scipy, because the harmonic mean of mixed signs is meaningless.
    /// </summary>
    public static double Hmean(double[] a, double[]? weights = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        foreach (var v in a) if (v < 0) throw new ArgumentException("Harmonic mean only defined if all elements are greater than or equal to zero.", nameof(a));
        var inv = new double[a.Length];
        for (var i = 0; i < a.Length; i++) inv[i] = 1.0 / a[i];
        return 1.0 / WeightedMean(inv, weights);
    }

    /// <summary><see cref="Hmean(double[], double[])"/> along an axis.</summary>
    public static NdArray Hmean(NdArray a, int axis = 0) => StatsAxis.Reduce(a, axis, x => Hmean(x));

    /// <summary>
    /// Power (Hölder) mean (Σwᵢaᵢᵖ/Σwᵢ)^{1/p}, scipy's <c>pmean</c>; p = 0 is the geometric mean.
    /// </summary>
    public static double Pmean(double[] a, double p, double[]? weights = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        foreach (var v in a) if (v < 0) throw new ArgumentException("Power mean only defined if all elements are greater than or equal to zero.", nameof(a));
        if (p == 0) return Gmean(a, weights);
        var pw = new double[a.Length];
        for (var i = 0; i < a.Length; i++) pw[i] = Math.Pow(a[i], p);
        return Math.Pow(WeightedMean(pw, weights), 1.0 / p);
    }

    private static double WeightedMean(double[] x, double[]? w)
    {
        if (w is null) return x.Length == 0 ? double.NaN : StatsTestMath.Mean(x);
        if (w.Length != x.Length) throw new ArgumentException("weights must have the same length as the data.", nameof(w));
        double s = 0, sw = 0;
        for (var i = 0; i < x.Length; i++) { s += w[i] * x[i]; sw += w[i]; }
        return s / sw;
    }

    // ------------------------------------------------------------------ trimmed statistics

    private static double[] Within(double[] a, (double Lower, double Upper)? limits, (bool Lower, bool Upper) inclusive)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (limits is null) return a;
        var (lo, hi) = limits.Value;
        var r = new List<double>(a.Length);
        foreach (var v in a)
        {
            var okLo = inclusive.Lower ? v >= lo : v > lo;
            var okHi = inclusive.Upper ? v <= hi : v < hi;
            if (okLo && okHi) r.Add(v);
        }
        if (r.Count == 0) throw new ArgumentException("No array values within given limits.");
        return r.ToArray();
    }

    /// <summary>Mean of the values inside <paramref name="limits"/>, scipy's <c>tmean</c>.</summary>
    public static double Tmean(double[] a, (double Lower, double Upper)? limits = null, (bool Lower, bool Upper)? inclusive = null)
        => StatsTestMath.Mean(Within(a, limits, inclusive ?? (true, true)));

    /// <summary>Variance (ddof 1) of the values inside the limits, scipy's <c>tvar</c>.</summary>
    public static double Tvar(double[] a, (double Lower, double Upper)? limits = null, (bool Lower, bool Upper)? inclusive = null, int ddof = 1)
        => StatsTestMath.Var(Within(a, limits, inclusive ?? (true, true)), ddof);

    /// <summary>Standard deviation of the values inside the limits, scipy's <c>tstd</c>.</summary>
    public static double Tstd(double[] a, (double Lower, double Upper)? limits = null, (bool Lower, bool Upper)? inclusive = null, int ddof = 1)
        => Math.Sqrt(Tvar(a, limits, inclusive, ddof));

    /// <summary>Standard error of the trimmed mean, scipy's <c>tsem</c>.</summary>
    public static double Tsem(double[] a, (double Lower, double Upper)? limits = null, (bool Lower, bool Upper)? inclusive = null, int ddof = 1)
    {
        var w = Within(a, limits, inclusive ?? (true, true));
        return Math.Sqrt(StatsTestMath.Var(w, ddof)) / Math.Sqrt(w.Length);
    }

    /// <summary>Minimum of the values above <paramref name="lowerlimit"/>, scipy's <c>tmin</c>.</summary>
    public static double Tmin(double[] a, double? lowerlimit = null, bool inclusive = true)
    {
        ArgumentNullException.ThrowIfNull(a);
        var m = double.PositiveInfinity;
        var any = false;
        foreach (var v in a)
            if (lowerlimit is null || (inclusive ? v >= lowerlimit : v > lowerlimit)) { any = true; if (v < m) m = v; }
        if (!any) throw new ArgumentException("No array values within given limits.");
        return m;
    }

    /// <summary>Maximum of the values below <paramref name="upperlimit"/>, scipy's <c>tmax</c>.</summary>
    public static double Tmax(double[] a, double? upperlimit = null, bool inclusive = true)
    {
        ArgumentNullException.ThrowIfNull(a);
        var m = double.NegativeInfinity;
        var any = false;
        foreach (var v in a)
            if (upperlimit is null || (inclusive ? v <= upperlimit : v < upperlimit)) { any = true; if (v > m) m = v; }
        if (!any) throw new ArgumentException("No array values within given limits.");
        return m;
    }

    /// <summary>
    /// Mean after cutting ⌊proportion·n⌋ observations from each end, scipy's <c>trim_mean</c>.
    /// </summary>
    public static double TrimMean(double[] a, double proportiontocut)
    {
        var s = Trimboth(a, proportiontocut);
        return s.Length == 0 ? double.NaN : StatsTestMath.Mean(s);
    }

    /// <summary><see cref="TrimMean(double[], double)"/> along an axis.</summary>
    public static NdArray TrimMean(NdArray a, double proportiontocut, int axis = 0)
        => StatsAxis.Reduce(a, axis, x => TrimMean(x, proportiontocut));

    /// <summary>
    /// Removes ⌊proportion·n⌋ observations from each end, scipy's <c>trimboth</c>. scipy returns them in
    /// partition order; these come back sorted.
    /// </summary>
    public static double[] Trimboth(double[] a, double proportiontocut)
    {
        ArgumentNullException.ThrowIfNull(a);
        var n = a.Length;
        var lo = (int)(proportiontocut * n);
        var hi = n - lo;
        if (lo >= hi) throw new ArgumentException("Proportion too big.", nameof(proportiontocut));
        return StatsTestMath.Sorted(a)[lo..hi];
    }

    /// <summary>Removes ⌊proportion·n⌋ observations from one tail (<c>"right"</c> or <c>"left"</c>), scipy's <c>trim1</c>.</summary>
    public static double[] Trim1(double[] a, double proportiontocut, string tail = "right")
    {
        ArgumentNullException.ThrowIfNull(a);
        var n = a.Length;
        var cut = (int)(proportiontocut * n);
        if (proportiontocut >= 1) return [];
        var s = StatsTestMath.Sorted(a);
        return tail.ToLowerInvariant() switch
        {
            "right" => s[..(n - cut)],
            "left" => s[cut..],
            _ => throw new ArgumentException("tail must be 'left' or 'right'.", nameof(tail)),
        };
    }

    // ------------------------------------------------------------------ scales and standardisation

    /// <summary>Standard error of the mean, std(ddof)/√n (scipy's <c>sem</c>, ddof 1).</summary>
    public static double Sem(double[] a, int ddof = 1)
    {
        ArgumentNullException.ThrowIfNull(a);
        return Math.Sqrt(StatsTestMath.Var(a, ddof)) / Math.Sqrt(a.Length);
    }

    /// <summary><see cref="Sem(double[], int)"/> along an axis.</summary>
    public static NdArray Sem(NdArray a, int axis = 0, int ddof = 1) => StatsAxis.Reduce(a, axis, x => Sem(x, ddof));

    /// <summary>
    /// z-scores of <paramref name="scores"/> relative to <paramref name="compare"/>, scipy's <c>zmap</c>.
    /// A constant comparison sample yields NaN everywhere (scipy's behaviour), not ±∞.
    /// </summary>
    public static double[] Zmap(double[] scores, double[] compare, int ddof = 0)
    {
        ArgumentNullException.ThrowIfNull(scores);
        ArgumentNullException.ThrowIfNull(compare);
        var mn = StatsTestMath.Mean(compare);
        var sd = Math.Sqrt(StatsTestMath.Var(compare, ddof));
        var isConst = true;
        foreach (var v in compare) if (v != compare[0]) { isConst = false; break; }
        var z = new double[scores.Length];
        for (var i = 0; i < z.Length; i++) z[i] = isConst ? double.NaN : (scores[i] - mn) / sd;
        return z;
    }

    /// <summary>z-scores (x − mean)/std (scipy's <c>zscore</c>, ddof 0).</summary>
    public static double[] Zscore(double[] a, int ddof = 0) => Zmap(a, a, ddof);

    /// <summary><see cref="Zscore(double[], int)"/> along an axis.</summary>
    public static NdArray Zscore(NdArray a, int axis = 0, int ddof = 0) => StatsAxis.Transform(a, axis, x => Zscore(x, ddof));

    /// <summary>Geometric z-scores, z-scores of log(a) (scipy's <c>gzscore</c>).</summary>
    public static double[] Gzscore(double[] a, int ddof = 0)
    {
        ArgumentNullException.ThrowIfNull(a);
        var l = new double[a.Length];
        for (var i = 0; i < l.Length; i++) l[i] = Math.Log(a[i]);
        return Zscore(l, ddof);
    }

    /// <summary>
    /// Interquartile (or any percentile) range, scipy's <c>iqr</c>, divided by <paramref name="scale"/>.
    /// Use <see cref="NormalIqrScale"/> to make it a consistent estimator of σ for normal data.
    /// </summary>
    public static double Iqr(double[] x, (double Low, double High)? rng = null, double scale = 1.0)
    {
        ArgumentNullException.ThrowIfNull(x);
        var (lo, hi) = rng ?? (25.0, 75.0);
        if (!(lo >= 0 && hi <= 100 && lo <= hi)) throw new ArgumentException("rng must satisfy 0 <= low <= high <= 100.", nameof(rng));
        if (x.Length == 0) return double.NaN;
        var s = StatsTestMath.Sorted(x);
        return (StatsTestMath.PercentileSorted(s, hi) - StatsTestMath.PercentileSorted(s, lo)) / scale;
    }

    /// <summary><see cref="Iqr(double[], ValueTuple{double, double}?, double)"/> along an axis.</summary>
    public static NdArray Iqr(NdArray x, int axis = 0, (double Low, double High)? rng = null, double scale = 1.0)
        => StatsAxis.Reduce(x, axis, v => Iqr(v, rng, scale));

    /// <summary>scipy's <c>scale='normal'</c> for <see cref="Iqr(double[], ValueTuple{double, double}?, double)"/>: 2√2 erf⁻¹(½) ≈ 1.349.</summary>
    public const double NormalIqrScale = 1.3489795003921634;

    /// <summary>scipy's <c>scale='normal'</c> for <see cref="MedianAbsDeviation(double[], double, Func{double[], double})"/>: Φ⁻¹(¾) ≈ 0.6745.</summary>
    public const double NormalMadScale = 0.6744897501960817;

    /// <summary>
    /// Median absolute deviation median(|x − center|)/scale, scipy's <c>median_abs_deviation</c>; the
    /// center is the median unless given. Pass <see cref="NormalMadScale"/> for a σ estimate.
    /// </summary>
    public static double MedianAbsDeviation(double[] x, double scale = 1.0, Func<double[], double>? center = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Length == 0) return double.NaN;
        var c = center?.Invoke(x) ?? StatsTestMath.Median(x);
        var d = new double[x.Length];
        for (var i = 0; i < d.Length; i++) d[i] = Math.Abs(x[i] - c);
        return StatsTestMath.Median(d) / scale;
    }

    /// <summary><see cref="MedianAbsDeviation(double[], double, Func{double[], double})"/> along an axis.</summary>
    public static NdArray MedianAbsDeviation(NdArray x, int axis = 0, double scale = 1.0)
        => StatsAxis.Reduce(x, axis, v => MedianAbsDeviation(v, scale));

    /// <summary>
    /// The smallest of the most frequent values and its count, scipy's <c>mode</c>.
    /// </summary>
    public static ModeResult Mode(double[] a)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Length == 0) return new ModeResult(double.NaN, 0);
        var s = StatsTestMath.Sorted(a);
        double best = s[0];
        int bestCount = 0, start = 0;
        while (start < s.Length)
        {
            var end = start + 1;
            while (end < s.Length && (s[end] == s[start] || (double.IsNaN(s[end]) && double.IsNaN(s[start])))) end++;
            if (end - start > bestCount) { bestCount = end - start; best = s[start]; }
            start = end;
        }
        return new ModeResult(best, bestCount);
    }

    /// <summary>
    /// O'Brien's transformation of each sample (scipy's <c>obrientransform</c>): values whose means are
    /// the sample variances, so a one-way ANOVA on them tests equality of variances.
    /// </summary>
    public static double[][] Obrientransform(params double[][] samples)
    {
        ArgumentNullException.ThrowIfNull(samples);
        var r = new double[samples.Length][];
        for (var k = 0; k < samples.Length; k++)
        {
            var a = samples[k];
            var n = a.Length;
            var mu = StatsTestMath.Mean(a);
            var sq = new double[n];
            for (var i = 0; i < n; i++) sq[i] = (a[i] - mu) * (a[i] - mu);
            var sumsq = StatsTestMath.Sum(sq);
            var t = new double[n];
            for (var i = 0; i < n; i++) t[i] = ((n - 1.5) * n * sq[i] - 0.5 * sumsq) / ((n - 1.0) * (n - 2.0));
            var check = StatsTestMath.Mean(t);
            var v = StatsTestMath.Var(a, 1);
            if (Math.Abs(check - v) > 1e-8 + 1e-5 * Math.Abs(v)) throw new ArgumentException("Lack of convergence in obrientransform.");
            r[k] = t;
        }
        return r;
    }

    /// <summary>
    /// Iterative sigma clipping (scipy's <c>sigmaclip</c>): drop values outside
    /// [mean − low·std, mean + high·std] until nothing more is dropped.
    /// </summary>
    public static SigmaclipResult Sigmaclip(double[] a, double low = 4.0, double high = 4.0)
    {
        ArgumentNullException.ThrowIfNull(a);
        var c = a;
        double lo, hi;
        while (true)
        {
            var sd = Math.Sqrt(StatsTestMath.Var(c));
            var mn = StatsTestMath.Mean(c);
            lo = mn - sd * low;
            hi = mn + sd * high;
            double cl = lo, ch = hi;
            var next = c.Where(v => v >= cl && v <= ch).ToArray();
            var delta = c.Length - next.Length;
            c = next;
            if (delta == 0) break;
        }
        return new SigmaclipResult(c, lo, hi);
    }

    // ------------------------------------------------------------------ L-moments and expectiles

    /// <summary>
    /// Sample L-moments λ₁…λ_order (scipy's <c>lmoment</c>, unbiased probability-weighted moments).
    /// With <paramref name="standardize"/> the third and higher are divided by λ₂ (L-skewness, L-kurtosis).
    /// </summary>
    public static double[] Lmoment(double[] sample, int order = 4, bool standardize = true)
    {
        ArgumentNullException.ThrowIfNull(sample);
        if (order < 1) throw new ArgumentException("order must be positive.", nameof(order));
        var x = StatsTestMath.Sorted(sample);
        var n = x.Length;
        var b = new double[order];
        for (var r = 0; r < order; r++)
        {
            if (r >= n) { b[r] = 0; continue; }
            var s = 0.0;
            for (var j = 0; j < n; j++) s += Special.Binom(j, r) * x[j];
            b[r] = s / Special.Binom(n - 1, r) / n;
        }
        var l = new double[order];
        for (var r = 0; r < order; r++)
        {
            var s = 0.0;
            for (var k = 0; k <= r; k++)
                s += ((r - k) % 2 == 0 ? 1 : -1) * Special.Binom(r, k) * Special.Binom(r + k, k) * b[k];
            l[r] = s;
        }
        if (standardize && order > 2) for (var r = 2; r < order; r++) l[r] /= l[1];
        for (var r = n; r < order; r++) l[r] = double.NaN;
        return l;
    }

    /// <summary>
    /// The α-expectile (scipy's <c>expectile</c>): the t solving Σ wᵢ|1{aᵢ ≤ t} − α|(t − aᵢ) = 0. The
    /// identification function is monotone and piecewise linear, so a bracketed solve is exact.
    /// </summary>
    public static double Expectile(double[] a, double alpha = 0.5, double[]? weights = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (alpha < 0 || alpha > 1) throw new ArgumentException("The expectile level alpha must be in the range [0, 1].", nameof(alpha));
        if (a.Length == 0) return double.NaN;
        double F(double t)
        {
            double s = 0, sw = 0;
            for (var i = 0; i < a.Length; i++)
            {
                var w = weights?[i] ?? 1.0;
                s += w * Math.Abs((a[i] <= t ? 1.0 : 0.0) - alpha) * (t - a[i]);
                sw += w;
            }
            return s / sw;
        }
        double lo = a.Min(), hi = a.Max();
        if (lo == hi) return lo;
        // F is increasing; it is piecewise linear between data points, so locate the piece and
        // solve the line exactly instead of iterating.
        var pts = StatsTestMath.Sorted(a);
        double fl = F(lo), fh = F(hi);
        if (fl >= 0) return lo;
        if (fh <= 0) return hi;
        int l = 0, h = pts.Length - 1;
        while (h - l > 1)
        {
            var m = (l + h) / 2;
            if (F(pts[m]) < 0) l = m; else h = m;
        }
        double x0 = pts[l], x1 = pts[h];
        if (x0 == x1) return x0;
        double f0 = F(x0), f1 = F(x1);
        if (f1 == 0) return x1;
        // Linear on (x0, x1): but F jumps in slope at x0 only, so evaluate just inside.
        var xm = x0 + (x1 - x0) / 2;
        var fm = F(xm);
        var slope = (f1 - fm) / (x1 - xm);
        return x1 - f1 / slope;
    }
}
