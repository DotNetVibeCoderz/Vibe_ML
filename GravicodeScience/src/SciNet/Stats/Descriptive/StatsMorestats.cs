using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>How <see cref="Stats.BoxcoxNormmax"/> chooses λ.</summary>
public enum BoxcoxNormmaxMethod
{
    /// <summary>Maximise the probability-plot correlation with the normal (scipy's default).</summary>
    Pearsonr,
    /// <summary>Maximise the Box–Cox log-likelihood.</summary>
    Mle,
}

/// <summary>Where <see cref="Stats.MedianTest"/> counts values equal to the grand median.</summary>
public enum MedianTestTies
{
    /// <summary>Count them as below (scipy's default).</summary>
    Below,
    /// <summary>Count them as above.</summary>
    Above,
    /// <summary>Leave them out.</summary>
    Ignore,
}

/// <summary>Result of <see cref="Stats.MedianTest"/> (scipy's <c>MedianTestResult</c>).</summary>
/// <param name="Statistic">The chi-square (or power-divergence) statistic of the 2 × k table.</param>
/// <param name="Pvalue">The p-value.</param>
/// <param name="Median">The grand median of all samples.</param>
/// <param name="Table">Counts above (row 0) and below (row 1) the grand median, per sample.</param>
public sealed record MedianTestResult(double Statistic, double Pvalue, double Median, int[,] Table);

/// <summary>A Bayesian point estimate with its credible interval (scipy's <c>Mean</c>/<c>Variance</c>/<c>Std_dev</c>).</summary>
/// <param name="Statistic">Posterior mean.</param>
/// <param name="Minmax">Equal-tailed credible interval.</param>
public sealed record BayesMvsEstimate(double Statistic, (double Lower, double Upper) Minmax);

/// <summary>Result of <see cref="Stats.DirectionalStats"/> (scipy's <c>DirectionalStats</c>).</summary>
/// <param name="MeanDirection">Unit vector of the mean direction.</param>
/// <param name="MeanResultantLength">Length of the mean resultant vector, in [0, 1] for normalised input.</param>
public sealed record DirectionalStatsResult(double[] MeanDirection, double MeanResultantLength);

/// <summary>Result of <see cref="Stats.Probplot"/>.</summary>
/// <param name="Osm">Theoretical quantiles (order statistic medians through the ppf).</param>
/// <param name="Osr">Ordered data.</param>
/// <param name="Slope">Least-squares slope of osr on osm.</param>
/// <param name="Intercept">Intercept.</param>
/// <param name="R">Correlation coefficient.</param>
public sealed record ProbplotResult(double[] Osm, double[] Osr, double Slope, double Intercept, double R);

public static partial class Stats
{
    // ================================================================== Box–Cox

    /// <summary>The Box–Cox transform with a given λ (scipy.stats.boxcox with lmbda).</summary>
    public static double[] Boxcox(double[] x, double lmbda)
    {
        ArgumentNullException.ThrowIfNull(x);
        return x.Select(v => Special.Boxcox(v, lmbda)).ToArray();
    }

    /// <summary>
    /// The Box–Cox transform with the maximum-likelihood λ (scipy.stats.boxcox): data must be
    /// positive and not constant.
    /// </summary>
    public static (double[] Y, double Lmbda) Boxcox(double[] x)
    {
        CheckBoxcoxData(x);
        var lmax = BoxcoxNormmax(x, method: BoxcoxNormmaxMethod.Mle);
        return (Boxcox(x, lmax), lmax);
    }

    /// <summary>
    /// scipy.stats.boxcox with alpha: <see cref="Boxcox(double[])"/> plus the profile-likelihood confidence interval for λ at level
    /// 1 − <paramref name="alpha"/> (the λ where the log-likelihood falls by χ²₁(1 − α)/2).
    /// </summary>
    public static (double[] Y, double Lmbda, (double Lower, double Upper) Interval) BoxcoxInterval(double[] x, double alpha)
    {
        var (y, lmax) = Boxcox(x);
        var fac = 0.5 * Special.Chdtri(1, alpha);
        var target = BoxcoxLlf(lmax, x) - fac;
        double Root(double l) => BoxcoxLlf(l, x) - target;
        double Find(double dir)
        {
            var newlm = lmax + 0.5 * dir;
            var n = 0;
            while (Root(newlm) > 0 && n < 500) { newlm += 0.1 * dir; n++; }
            if (n == 500) throw new InvalidOperationException("Could not find endpoint.");
            return dir > 0 ? Optimize.Brentq(Root, lmax, newlm).Root : Optimize.Brentq(Root, newlm, lmax).Root;
        }
        return (y, lmax, (Find(-1), Find(1)));
    }

    private static void CheckBoxcoxData(double[] x)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Length == 0) throw new ArgumentException("Data must not be empty.", nameof(x));
        if (x.All(v => v == x[0])) throw new ArgumentException("Data must not be constant.", nameof(x));
        if (x.Any(v => v <= 0)) throw new ArgumentException("Data must be positive.", nameof(x));
    }

    /// <summary>
    /// Box–Cox log-likelihood (scipy.stats.boxcox_llf):
    /// <c>(λ − 1) Σ log x − n/2 · log var(y)</c>, with the variance of the transformed data computed in
    /// log space (logsumexp), so large |λ| neither overflows nor cancels.
    /// </summary>
    public static double BoxcoxLlf(double lmb, double[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        var n = data.Length;
        if (n == 0) return double.NaN;
        var logdata = data.Select(v => Math.Log(v)).ToArray();
        double logvar;
        if (lmb == 0) logvar = Math.Log(StatsTestMath.Var(logdata));
        else
        {
            var logx = logdata.Select(v => lmb * v).ToArray();
            logvar = LogVar(logx) - 2 * Math.Log(Math.Abs(lmb));
        }
        return (lmb - 1) * StatsTestMath.Sum(logdata) - n / 2.0 * logvar;
    }

    /// <summary>log of the population variance of exp(logx), all in log space.</summary>
    private static double LogVar(double[] logx)
    {
        var n = logx.Length;
        var logmean = Special.LogSumExp(logx) - Math.Log(n);
        var t = new double[n];
        for (var i = 0; i < n; i++)
        {
            var a = logx[i];
            var d = Math.Abs(a - logmean);
            t[i] = d == 0 ? double.NegativeInfinity : 2 * (Math.Max(a, logmean) + Math.Log(-Special.Expm1(-d)));
        }
        return Special.LogSumExp(t) - Math.Log(n);
    }

    /// <summary>
    /// Optimal Box–Cox λ (scipy.stats.boxcox_normmax), by Brent's method from the bracket
    /// (default (−2, 2)). When the optimal λ would send the extreme datum beyond
    /// <paramref name="ymax"/> (default double.MaxValue/10000) in absolute value, the constrained λ
    /// at which it equals ±ymax is returned instead (via the Lambert W function, as scipy).
    /// </summary>
    public static double BoxcoxNormmax(double[] x, double[]? brack = null, BoxcoxNormmaxMethod method = BoxcoxNormmaxMethod.Pearsonr, double? ymax = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (!x.All(v => double.IsFinite(v) && v >= 0))
            throw new ArgumentException("The `x` argument of `boxcox_normmax` must contain only positive, finite, real numbers.", nameof(x));
        var ym = ymax ?? double.MaxValue / 10000;
        if (!(ym > 0)) throw new ArgumentException("`ymax` must be strictly positive", nameof(ymax));
        brack ??= [-2.0, 2.0];
        var res = method == BoxcoxNormmaxMethod.Mle ? BoxcoxMle(x, brack) : BoxcoxPearsonr(x, brack);
        return ConstrainBoxcox(x, res, ym);
    }

    /// <summary>scipy's <c>boxcox_normmax(method='all')</c>: the Pearson-r and the MLE λ.</summary>
    public static (double Pearsonr, double Mle) BoxcoxNormmaxAll(double[] x, double[]? brack = null, double? ymax = null)
        => (BoxcoxNormmax(x, brack, BoxcoxNormmaxMethod.Pearsonr, ymax), BoxcoxNormmax(x, brack, BoxcoxNormmaxMethod.Mle, ymax));

    private static double BoxcoxMle(double[] x, double[] brack) => Optimize.Brent(l => -BoxcoxLlf(l, x), brack).X;

    private static double BoxcoxPearsonr(double[] x, double[] brack)
    {
        var osm = UniformOrderStatisticMedians(x.Length).Select(v => Special.Ndtri(v)).ToArray();
        double F(double l)
        {
            var y = StatsTestMath.Sorted(Boxcox(x, l));
            return 1 - PearsonR(osm, y);
        }
        return Optimize.Brent(F, brack).X;
    }

    private static double PearsonR(double[] a, double[] b)
    {
        var n = a.Length;
        double ma = StatsTestMath.Mean(a), mb = StatsTestMath.Mean(b);
        double sab = 0, saa = 0, sbb = 0;
        for (var i = 0; i < n; i++)
        {
            var da = a[i] - ma; var db = b[i] - mb;
            sab += da * db; saa += da * da; sbb += db * db;
        }
        return Math.Clamp(sab / Math.Sqrt(saa * sbb), -1, 1);
    }

    private static double ConstrainBoxcox(double[] x, double res, double ymax)
    {
        if (double.IsInfinity(ymax)) return res;
        double xmax = x.Max(), xmin = x.Min();
        double xt;
        if (xmin >= 1) xt = xmax;
        else if (xmax <= 1) xt = xmin;
        else xt = Special.Boxcox(xmax, res) > Math.Abs(Special.Boxcox(xmin, res)) ? xmax : xmin;
        if (!(Math.Abs(Special.Boxcox(xt, res)) > ymax)) return res;
        // Solve boxcox(xt, λ) = ±ymax: λ = −W₋₁(−xt^(−1/y) log xt / y)/log xt − 1/y.
        var y = ymax * Math.Sign(xt - 1);
        var arg = -Math.Pow(xt, -1 / y) * Math.Log(xt) / y;
        var w = Special.LambertW(new System.Numerics.Complex(arg, 0), -1);
        return (-w / Math.Log(xt) - 1 / y).Real;
    }

    /// <summary>Filliben's estimate of the uniform order-statistic medians (scipy's helper for probplot).</summary>
    private static double[] UniformOrderStatisticMedians(int n)
    {
        var v = new double[n];
        v[n - 1] = Math.Pow(0.5, 1.0 / n);
        v[0] = 1 - v[n - 1];
        for (var i = 2; i < n; i++) v[i - 1] = (i - 0.3175) / (n + 0.365);
        return v;
    }

    // ================================================================== Yeo–Johnson

    /// <summary>The Yeo–Johnson transform with a given λ (scipy.stats.yeojohnson with lmbda), defined for all real x.</summary>
    public static double[] Yeojohnson(double[] x, double lmbda)
    {
        ArgumentNullException.ThrowIfNull(x);
        const double eps = 2.220446049250313e-16;
        var o = new double[x.Length];
        for (var i = 0; i < x.Length; i++)
        {
            var v = x[i];
            if (v >= 0) o[i] = Math.Abs(lmbda) < eps ? Special.Log1p(v) : Special.Expm1(lmbda * Special.Log1p(v)) / lmbda;
            else o[i] = Math.Abs(lmbda - 2) > eps ? -Special.Expm1((2 - lmbda) * Special.Log1p(-v)) / (2 - lmbda) : -Special.Log1p(-v);
        }
        return o;
    }

    /// <summary>The Yeo–Johnson transform with the maximum-likelihood λ (scipy.stats.yeojohnson).</summary>
    public static (double[] Y, double Lmbda) Yeojohnson(double[] x)
    {
        var l = YeojohnsonNormmax(x);
        return (Yeojohnson(x, l), l);
    }

    /// <summary>Yeo–Johnson log-likelihood (scipy.stats.yeojohnson_llf); −∞ when the transformed data have zero variance.</summary>
    public static double YeojohnsonLlf(double lmb, double[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        var n = data.Length;
        if (n == 0) return double.NaN;
        var y = Yeojohnson(data, lmb);
        var sigma = StatsTestMath.Var(y);
        var logSigma = sigma >= 2.2250738585072014e-308 ? Math.Log(sigma) : double.NegativeInfinity;
        double s = 0;
        foreach (var v in data) s += Math.Sign(v) * Special.Log1p(Math.Abs(v));
        return -n / 2.0 * logSigma + (lmb - 1) * s;
    }

    /// <summary>
    /// Optimal Yeo–Johnson λ (scipy.stats.yeojohnson_normmax): Brent from <paramref name="brack"/> when
    /// given; otherwise a bounded search over the λ range in which the transform of 20·max|x| neither
    /// overflows nor underflows (scipy's bounds), with xtol 1.48e-8.
    /// </summary>
    public static double YeojohnsonNormmax(double[] x, double[]? brack = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (!x.All(double.IsFinite)) throw new ArgumentException("Yeo-Johnson input must be finite.", nameof(x));
        if (x.All(v => v == 0)) return 1.0;
        double Neg(double l)
        {
            var v = YeojohnsonLlf(l, x);
            return double.IsInfinity(v) ? double.PositiveInfinity : -v;
        }
        if (brack is not null) return Optimize.Brent(Neg, brack).X;
        var log1pMax = Special.Log1p(20 * x.Max(v => Math.Abs(v)));
        var logEps = Math.Log(2.220446049250313e-16);
        var logTiny = (Math.Log(2.2250738585072014e-308) - logEps) / 2;
        var logMax = (Math.Log(double.MaxValue) + logEps) / 2;
        var lb = logTiny / log1pMax;
        var ub = logMax / log1pMax;
        if (x.All(v => v < 0)) (lb, ub) = (2 - ub, 2 - lb);
        else if (x.Any(v => v < 0)) (lb, ub) = (Math.Max(2 - ub, lb), Math.Min(2 - lb, ub));
        return Optimize.Fminbound(Neg, lb, ub, xatol: 1.48e-8).X;
    }

    // ================================================================== circular statistics

    /// <summary>
    /// Circular mean of angles on [low, high) (scipy.stats.circmean): the angle of the mean
    /// resultant vector, mapped back into [low, high).
    /// </summary>
    public static double Circmean(double[] samples, double high = 2 * Math.PI, double low = 0)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Length == 0) return double.NaN;
        var period = high - low;
        var (s, c) = CircSums(samples, period);
        var res = Math.Atan2(s, c);
        var v = res * (period / (2.0 * Math.PI)) - low;
        return v - period * Math.Floor(v / period) + low;
    }

    /// <summary>Circular variance 1 − R, R the mean resultant length (scipy.stats.circvar), in [0, 1].</summary>
    public static double Circvar(double[] samples, double high = 2 * Math.PI, double low = 0)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Length == 0) return double.NaN;
        var (s, c) = CircSums(samples, high - low);
        var n = samples.Length;
        var r = Math.Min(Math.Sqrt(s / n * (s / n) + c / n * (c / n)), 1.0);
        return 1.0 - r;
    }

    /// <summary>
    /// Circular standard deviation √(−2 ln R) (scipy.stats.circstd), in the units of [low, high)
    /// unless <paramref name="normalize"/>, which leaves it in radians.
    /// </summary>
    public static double Circstd(double[] samples, double high = 2 * Math.PI, double low = 0, bool normalize = false)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Length == 0) return double.NaN;
        var (s, c) = CircSums(samples, high - low);
        var n = samples.Length;
        var r = Math.Min(Math.Sqrt(s / n * (s / n) + c / n * (c / n)), 1.0);
        var res = Math.Sqrt(-2 * Math.Log(r)) + 0.0;
        return normalize ? res : res * (high - low) / (2.0 * Math.PI);
    }

    private static (double Sin, double Cos) CircSums(double[] samples, double period)
    {
        double s = 0, c = 0;
        var k = 2.0 * Math.PI / period;
        foreach (var v in samples)
        {
            var (sn, cs) = Math.SinCos(v * k);
            s += sn; c += cs;
        }
        return (s, c);
    }

    /// <summary>
    /// Mean direction and mean resultant length of unit vectors (scipy.stats.directional_stats);
    /// <paramref name="samples"/> is n × d, rows normalised first unless <paramref name="normalize"/> is false.
    /// </summary>
    public static DirectionalStatsResult DirectionalStats(NdArray samples, bool normalize = true)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Rank < 2) throw new ArgumentException("samples must at least be two-dimensional.", nameof(samples));
        var (n, d, a) = StatsTestMath.Matrix(samples, nameof(samples));
        var mean = new double[d];
        for (var i = 0; i < n; i++)
        {
            double norm = 1;
            if (normalize)
            {
                norm = 0;
                for (var j = 0; j < d; j++) norm += a[i * d + j] * a[i * d + j];
                norm = Math.Sqrt(norm);
            }
            for (var j = 0; j < d; j++) mean[j] += a[i * d + j] / norm / n;
        }
        var len = Math.Sqrt(mean.Sum(v => v * v));
        return new DirectionalStatsResult(mean.Select(v => v / len).ToArray(), len);
    }

    // ================================================================== Mood's median test

    /// <summary>
    /// Mood's median test (scipy.stats.median_test): a chi-square test of the 2 × k table of counts
    /// above/below the grand median. <paramref name="lambda"/> selects the power-divergence statistic
    /// (1 = Pearson), <paramref name="correction"/> applies Yates' correction when the table has one
    /// degree of freedom.
    /// </summary>
    public static MedianTestResult MedianTest(double[][] samples, MedianTestTies ties = MedianTestTies.Below, bool correction = true, double lambda = 1)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Length < 2) throw new ArgumentException("median_test requires two or more samples.", nameof(samples));
        for (var k = 0; k < samples.Length; k++)
            if (samples[k] is null || samples[k].Length == 0) throw new ArgumentException($"Sample {k + 1} is empty. All samples must contain at least one value.");
        var all = StatsTestMath.Concat(samples);
        if (all.Any(double.IsNaN)) return new MedianTestResult(double.NaN, double.NaN, double.NaN, new int[2, samples.Length]);
        var med = StatsTestMath.Median(all);
        var table = new int[2, samples.Length];
        for (var k = 0; k < samples.Length; k++)
        {
            int above = 0, below = 0;
            foreach (var v in samples[k]) { if (v > med) above++; else if (v < med) below++; }
            var equal = samples[k].Length - above - below;
            table[0, k] = above + (ties == MedianTestTies.Above ? equal : 0);
            table[1, k] = below + (ties == MedianTestTies.Below ? equal : 0);
        }
        int r0 = 0, r1 = 0;
        for (var k = 0; k < samples.Length; k++) { r0 += table[0, k]; r1 += table[1, k]; }
        if (r0 == 0) throw new ArgumentException($"All values are below the grand median ({med}).");
        if (r1 == 0) throw new ArgumentException($"All values are above the grand median ({med}).");
        if (ties == MedianTestTies.Ignore)
            for (var k = 0; k < samples.Length; k++)
                if (table[0, k] == 0 && table[1, k] == 0)
                    throw new ArgumentException($"All values in sample {k + 1} are equal to the grand median ({med}), so they are ignored, resulting in an empty sample.");
        var obs = new double[2 * samples.Length];
        for (var k = 0; k < samples.Length; k++) { obs[k] = table[0, k]; obs[samples.Length + k] = table[1, k]; }
        var res = Chi2Contingency(new NdArray(obs, 2, samples.Length), correction, lambda);
        return new MedianTestResult(res.Statistic, res.Pvalue, med, table);
    }

    // ================================================================== small descriptive helpers

    /// <summary>Geometric standard deviation exp(std(log a, ddof)) (scipy.stats.gstd); NaN if any value ≤ 0.</summary>
    public static double Gstd(double[] a, int ddof = 1)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Any(v => v <= 0)) return double.NaN;
        return Math.Exp(Math.Sqrt(StatsTestMath.Var(a.Select(v => Math.Log(v)).ToArray(), ddof)));
    }

    /// <summary>The n-th k-statistic, n = 1..4 (scipy.stats.kstat): the unique symmetric unbiased estimator of the n-th cumulant.</summary>
    public static double Kstat(double[] data, int n = 2)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (n is > 4 or < 1) throw new ArgumentException("k-statistics only supported for 1<=n<=4", nameof(n));
        double N = data.Length;
        double s1 = 0, s2 = 0, s3 = 0, s4 = 0;
        foreach (var v in data) { var v2 = v * v; s1 += v; s2 += v2; s3 += v2 * v; s4 += v2 * v2; }
        return n switch
        {
            1 => s1 / N,
            2 => (N * s2 - s1 * s1) / (N * (N - 1.0)),
            3 => (2 * s1 * s1 * s1 - 3 * N * s1 * s2 + N * N * s3) / (N * (N - 1.0) * (N - 2.0)),
            _ => (-6 * Math.Pow(s1, 4) + 12 * N * s1 * s1 * s2 - 3 * N * (N - 1.0) * s2 * s2 - 4 * N * (N + 1) * s1 * s3 + N * N * (N + 1) * s4)
                 / (N * (N - 1.0) * (N - 2.0) * (N - 3.0)),
        };
    }

    /// <summary>Unbiased estimate of the variance of <see cref="Kstat"/> for n = 1 or 2 (scipy.stats.kstatvar).</summary>
    public static double Kstatvar(double[] data, int n = 2)
    {
        ArgumentNullException.ThrowIfNull(data);
        double N = data.Length;
        if (n == 1) return Kstat(data, 2) / N;
        if (n == 2)
        {
            var k2 = Kstat(data, 2);
            var k4 = Kstat(data, 4);
            return (2 * N * k2 * k2 + (N - 1) * k4) / (N * (N + 1));
        }
        throw new ArgumentException("Only n=1 or n=2 supported.", nameof(n));
    }

    /// <summary>Tie correction factor 1 − Σ(t³ − t)/(n³ − n) for Mann–Whitney/Kruskal–Wallis (scipy.stats.tiecorrect).</summary>
    public static double Tiecorrect(double[] rankvals)
    {
        ArgumentNullException.ThrowIfNull(rankvals);
        double size = rankvals.Length;
        if (size < 2) return 1.0;
        var s = StatsTestMath.Sorted(rankvals);
        double acc = 0;
        foreach (var t in StatsTestMath.TieCounts(s)) acc += (double)t * t * t - t;
        return 1.0 - acc / (size * size * size - size);
    }

    /// <summary>
    /// Bayesian confidence intervals for mean, variance and standard deviation under Jeffreys' prior
    /// (scipy.stats.bayes_mvs): t, inverse-gamma and generalised-gamma posteriors (normal
    /// approximations above 1000 points).
    /// </summary>
    public static (BayesMvsEstimate Mean, BayesMvsEstimate Variance, BayesMvsEstimate StdDev) BayesMvs(double[] data, double alpha = 0.9)
    {
        ArgumentNullException.ThrowIfNull(data);
        var n = data.Length;
        if (n < 2) throw new ArgumentException("Need at least 2 data-points.", nameof(data));
        if (!(alpha > 0 && alpha < 1)) throw new ArgumentException($"0 < alpha < 1 is required, but alpha={alpha} was given.", nameof(alpha));
        var xbar = StatsTestMath.Mean(data);
        var c = StatsTestMath.Var(data);
        if (n > 1000)
        {
            var md = new NormalDistribution(xbar, Math.Sqrt(c / n));
            var sd = new NormalDistribution(Math.Sqrt(c), Math.Sqrt(c / (2.0 * n)));
            var vd = new NormalDistribution(c, Math.Sqrt(2.0 / n) * c);
            return (new(md.Mean(), md.Interval(alpha)), new(vd.Mean(), vd.Interval(alpha)), new(sd.Mean(), sd.Interval(alpha)));
        }
        var nm1 = n - 1;
        var fac = n * c / 2.0;
        var val = nm1 / 2.0;
        var mdist = new StudentTDistribution(nm1, xbar, Math.Sqrt(c / nm1));
        var vdist = new InvGammaDistribution(val, 0, fac);
        // std ~ gengamma(val, c=-2, scale=√fac): X = √fac·Y^(−1/2), Y ~ Gamma(val).
        var sScale = Math.Sqrt(fac);
        var sMean = sScale * Math.Exp(Special.GammaLn(val - 0.5) - Special.GammaLn(val));
        var lo = sScale / Math.Sqrt(Special.GammaIncInv(val, 1 - (1 - alpha) / 2));
        var hi = sScale / Math.Sqrt(Special.GammaIncInv(val, (1 - alpha) / 2));
        return (new(mdist.Mean(), mdist.Interval(alpha)), new(vdist.Mean(), vdist.Interval(alpha)), new(sMean, (lo, hi)));
    }

    /// <summary>
    /// Probability-plot data against a distribution (scipy.stats.probplot): theoretical quantiles at
    /// Filliben's order-statistic medians, the sorted data, and the least-squares line through them.
    /// The default is the standard normal.
    /// </summary>
    public static ProbplotResult Probplot(double[] x, ContinuousDistribution? dist = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Length == 0) return new ProbplotResult([], [], double.NaN, double.NaN, 0.0);
        var u = UniformOrderStatisticMedians(x.Length);
        var osm = dist is null ? u.Select(v => Special.Ndtri(v)).ToArray() : u.Select(v => dist.Ppf(v)).ToArray();
        var osr = StatsTestMath.Sorted(x);
        var lr = Linregress(osm, osr);
        return new ProbplotResult(osm, osr, lr.Slope, lr.Intercept, lr.Rvalue);
    }
}
