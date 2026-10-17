using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>scipy.stats goodness-of-fit and normality tests.</summary>
public static partial class Stats
{
    // ------------------------------------------------------------------ Kolmogorov–Smirnov

    /// <summary>
    /// One-sample Kolmogorov–Smirnov test against a continuous CDF, scipy's <c>ks_1samp</c>.
    /// <paramref name="method"/>: <c>"auto"</c>/<c>"exact"</c> (the exact distribution of Dₙ),
    /// <c>"approx"</c> (twice the one-sided Smirnov tail) or <c>"asymp"</c> (Kolmogorov's limit).
    /// One-sided alternatives use the exact Smirnov distribution of D⁺/D⁻.
    /// </summary>
    public static KstestResult Ks1samp(double[] x, Func<double, double> cdf, Alternative alternative = Alternative.TwoSided, string method = "auto")
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(cdf);
        var n = x.Length;
        if (n == 0) throw new ArgumentException("x must not be empty.", nameof(x));
        var s = StatsTestMath.Sorted(x);
        var cv = new double[n];
        for (var i = 0; i < n; i++) cv[i] = cdf(s[i]);
        int ip = 0, im = 0;
        double dplus = double.NegativeInfinity, dminus = double.NegativeInfinity;
        for (var i = 0; i < n; i++)
        {
            var dp = (i + 1.0) / n - cv[i];
            var dm = cv[i] - (double)i / n;
            if (dp > dplus) { dplus = dp; ip = i; }
            if (dm > dminus) { dminus = dm; im = i; }
        }
        if (alternative == Alternative.Greater) return new KstestResult(dplus, Special.Smirnov(n, dplus), s[ip], 1);
        if (alternative == Alternative.Less) return new KstestResult(dminus, Special.Smirnov(n, dminus), s[im], -1);
        double d; double loc; int sign;
        if (dplus > dminus) { d = dplus; loc = s[ip]; sign = 1; }
        else { d = dminus; loc = s[im]; sign = -1; }
        double p = method.ToLowerInvariant() switch
        {
            "auto" or "exact" => new KSTwoDistribution(n).Sf(d),
            "asymp" => Special.Kolmogorov(d * Math.Sqrt(n)),
            "approx" => 2 * Special.Smirnov(n, d),
            _ => throw new ArgumentException("method must be one of auto, exact, approx, asymp.", nameof(method)),
        };
        return new KstestResult(d, Math.Clamp(p, 0, 1), loc, sign);
    }

    /// <summary>One-sample KS test against a frozen distribution.</summary>
    public static KstestResult Ks1samp(double[] x, IUnivariateDistribution dist, Alternative alternative = Alternative.TwoSided, string method = "auto")
    {
        ArgumentNullException.ThrowIfNull(dist);
        return Ks1samp(x, dist.Cdf, alternative, method);
    }

    /// <summary>scipy's <c>kstest</c> with a reference distribution: the one-sample test.</summary>
    public static KstestResult Kstest(double[] rvs, IUnivariateDistribution cdf, Alternative alternative = Alternative.TwoSided, string method = "auto")
        => Ks1samp(rvs, cdf, alternative, method);

    /// <summary>scipy's <c>kstest</c> with a reference CDF function: the one-sample test.</summary>
    public static KstestResult Kstest(double[] rvs, Func<double, double> cdf, Alternative alternative = Alternative.TwoSided, string method = "auto")
        => Ks1samp(rvs, cdf, alternative, method);

    /// <summary>scipy's <c>kstest</c> with a second sample: the two-sample test.</summary>
    public static KstestResult Kstest(double[] rvs, double[] other, Alternative alternative = Alternative.TwoSided, string method = "auto")
        => Ks2samp(rvs, other, alternative, method);

    /// <summary>
    /// Two-sample Kolmogorov–Smirnov test, scipy's <c>ks_2samp</c>. <c>"auto"</c> is exact when both
    /// samples have at most 10 000 values (lattice-path counting, Hodges 1958); <c>"asymp"</c> uses the
    /// KS distribution at the effective size (two-sided) or the Hodges exponential bound (one-sided).
    /// An exact computation that overflows falls back to the asymptotic one, as scipy.
    /// </summary>
    public static KstestResult Ks2samp(double[] data1, double[] data2, Alternative alternative = Alternative.TwoSided, string method = "auto")
    {
        ArgumentNullException.ThrowIfNull(data1);
        ArgumentNullException.ThrowIfNull(data2);
        var mode = method.ToLowerInvariant();
        if (mode is not ("auto" or "exact" or "asymp")) throw new ArgumentException($"Invalid value for mode: {method}", nameof(method));
        var a = StatsTestMath.Sorted(data1);
        var b = StatsTestMath.Sorted(data2);
        int n1 = a.Length, n2 = b.Length;
        if (Math.Min(n1, n2) == 0) throw new ArgumentException("Data passed to ks_2samp must not be empty");
        var all = new double[n1 + n2];
        a.CopyTo(all, 0);
        b.CopyTo(all, n1);
        double minS = double.PositiveInfinity, maxS = double.NegativeInfinity;
        double locMin = 0, locMax = 0;
        foreach (var v in all)
        {
            var diff = (double)UpperBound(a, v) / n1 - (double)UpperBound(b, v) / n2;
            if (diff < minS) { minS = diff; locMin = v; }
            if (diff > maxS) { maxS = diff; locMax = v; }
        }
        // argmin/argmax take the first occurrence in the concatenated order; the loop above does too.
        minS = Math.Clamp(-minS, 0, 1);
        double d, loc;
        int sign;
        if (alternative == Alternative.Less || (alternative == Alternative.TwoSided && minS > maxS)) { d = minS; loc = locMin; sign = -1; }
        else { d = maxS; loc = locMax; sign = 1; }
        var g = (int)Gcd(n1, n2);
        double prob = double.NaN;
        if (mode == "auto") mode = Math.Max(n1, n2) <= 10000 ? "exact" : "asymp";
        else if (mode == "exact" && n1 / g >= int.MaxValue / (double)(n2 / g)) mode = "asymp";
        if (mode == "exact")
        {
            var ok = StatsKsKernels.AttemptExact2Samp(n1, n2, g, d, alternative, out var dd, out prob);
            d = dd;
            if (!ok) mode = "asymp";
        }
        if (mode == "asymp")
        {
            double m = Math.Max(n1, n2), n = Math.Min(n1, n2);
            var en = m * n / (m + n);
            if (alternative == Alternative.TwoSided) prob = new KSTwoDistribution((int)Math.Round(en, MidpointRounding.ToEven)).Sf(d);
            else
            {
                var z = Math.Sqrt(en) * d;
                var expt = -2 * z * z - 2 * z * (m + 2 * n) / Math.Sqrt(m * n * (m + n)) / 3.0;
                prob = Math.Exp(expt);
            }
        }
        return new KstestResult(d, Math.Clamp(prob, 0, 1), loc, sign);
    }

    private static long Gcd(long a, long b)
    {
        while (b != 0) (a, b) = (b, a % b);
        return a;
    }

    // ------------------------------------------------------------------ Cramér–von Mises

    /// <summary>
    /// One-sample Cramér–von Mises test against a continuous CDF, scipy's <c>cramervonmises</c>. The
    /// p-value uses Csörgő &amp; Faraway's (1996) finite-n correction to the limiting distribution.
    /// </summary>
    public static SignificanceResult CramerVonMises(double[] rvs, Func<double, double> cdf)
    {
        ArgumentNullException.ThrowIfNull(rvs);
        ArgumentNullException.ThrowIfNull(cdf);
        var n = rvs.Length;
        if (n <= 1) throw new ArgumentException("The sample must contain at least two observations.", nameof(rvs));
        var s = StatsTestMath.Sorted(rvs);
        var terms = new double[n];
        for (var i = 0; i < n; i++)
        {
            var u = (2.0 * (i + 1) - 1) / (2.0 * n);
            var t = u - cdf(s[i]);
            terms[i] = t * t;
        }
        var w = 1.0 / (12 * n) + StatsTestMath.Sum(terms);
        var p = Math.Max(1.0 - StatsKsKernels.CdfCvm(w, n), 0.0);
        return new SignificanceResult(w, p);
    }

    /// <summary>One-sample Cramér–von Mises test against a frozen distribution.</summary>
    public static SignificanceResult CramerVonMises(double[] rvs, IUnivariateDistribution dist)
    {
        ArgumentNullException.ThrowIfNull(dist);
        return CramerVonMises(rvs, dist.Cdf);
    }

    /// <summary>
    /// Two-sample Cramér–von Mises test, scipy's <c>cramervonmises_2samp</c>: exact (lattice enumeration)
    /// when both samples have at most 20 values under <c>"auto"</c>, asymptotic otherwise.
    /// </summary>
    public static SignificanceResult CramerVonMises2Samp(double[] x, double[] y, string method = "auto")
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        int nx = x.Length, ny = y.Length;
        if (nx <= 1 || ny <= 1) throw new ArgumentException("x and y must contain at least two observations.");
        var m = method.ToLowerInvariant();
        if (m is not ("auto" or "exact" or "asymptotic")) throw new ArgumentException("method must be either auto, exact or asymptotic.", nameof(method));
        if (m == "auto") m = Math.Max(nx, ny) > 20 ? "asymptotic" : "exact";
        var z = new double[nx + ny];
        StatsTestMath.Sorted(x).CopyTo(z, 0);
        StatsTestMath.Sorted(y).CopyTo(z, nx);
        var r = Rankdata(z);
        double sx = 0, sy = 0;
        for (var i = 0; i < nx; i++) sx += (r[i] - (i + 1)) * (r[i] - (i + 1));
        for (var i = 0; i < ny; i++) sy += (r[nx + i] - (i + 1)) * (r[nx + i] - (i + 1));
        var u = nx * sx + ny * sy;
        double k = (double)nx * ny, N = nx + ny;
        var t = u / (k * N) - (4 * k - 1) / (6 * N);
        double p;
        if (m == "exact") p = StatsKsKernels.CvmTwoSampleExact(u, nx, ny);
        else
        {
            var et = (1 + 1 / N) / 6;
            var vt = (N + 1) * (4 * k * N - 3 * ((double)nx * nx + (double)ny * ny) - 2 * k);
            vt /= 45 * N * N * 4 * k;
            var tn = 1.0 / 6 + (t - et) / Math.Sqrt(45 * vt);
            p = tn >= 0.003 ? Math.Max(1.0 - StatsKsKernels.CdfCvmInf(tn), 0) : 1.0;
        }
        return new SignificanceResult(t, p);
    }

    // ------------------------------------------------------------------ Anderson–Darling

    private static readonly double[] AvalsNorm = [0.561, 0.631, 0.752, 0.873, 1.035];
    private static readonly double[] AvalsExpon = [0.916, 1.062, 1.321, 1.591, 1.959];
    private static readonly double[] AvalsLogistic = [0.426, 0.563, 0.66, 0.769, 0.906, 1.01];
    private static readonly double[] AvalsGumbel = [0.474, 0.637, 0.757, 0.877, 1.038];

    // Lockhart & Stephens (1994), "Estimation and tests of fit for the three-parameter Weibull
    // distribution", JRSS B 56(3), table 1: rows c = 1/m = 0, 0.05, ..., 0.5.
    private static readonly double[,] AvalsWeibull =
    {
        { 0.292, 0.395, 0.467, 0.522, 0.617, 0.711, 0.836, 0.931 },
        { 0.295, 0.399, 0.471, 0.527, 0.623, 0.719, 0.845, 0.941 },
        { 0.298, 0.403, 0.476, 0.534, 0.631, 0.728, 0.856, 0.954 },
        { 0.301, 0.408, 0.483, 0.541, 0.640, 0.738, 0.869, 0.969 },
        { 0.305, 0.414, 0.490, 0.549, 0.650, 0.751, 0.885, 0.986 },
        { 0.309, 0.421, 0.498, 0.559, 0.662, 0.765, 0.902, 1.007 },
        { 0.314, 0.429, 0.508, 0.570, 0.676, 0.782, 0.923, 1.030 },
        { 0.320, 0.438, 0.519, 0.583, 0.692, 0.802, 0.947, 1.057 },
        { 0.327, 0.448, 0.532, 0.598, 0.711, 0.824, 0.974, 1.089 },
        { 0.334, 0.469, 0.547, 0.615, 0.732, 0.850, 1.006, 1.125 },
        { 0.342, 0.472, 0.563, 0.636, 0.757, 0.879, 1.043, 1.167 },
    };

    /// <summary>
    /// Three-parameter Weibull MLE for the Anderson-Darling test: the root (m, u) of the first-order
    /// conditions (Lockhart &amp; Stephens eqs. 5-6, scale profiled out by eq. 7), started from the
    /// numerical MLE. Throws, as scipy does, when the fit puts the location at the sample minimum or
    /// the shape below 1, where the tabulated critical values do not apply.
    /// </summary>
    private static (double M, double U, double S) AndersonWeibullFit(double[] x)
    {
        var n = x.Length;
        var fit = WeibullMinDistribution.Fit(x);
        double m = fit.C, u = fit.Loc;
        var xmin = x.Min();
        const string suggestion = " Maximum likelihood estimation is known to be challenging for the three-parameter Weibull distribution.";
        bool AtMin(double uu) => Math.Abs(uu - xmin) <= 1e-8 + 1e-5 * Math.Abs(xmin);
        if (AtMin(u) || m < 1)
            throw new ArgumentException("Maximum likelihood estimation has converged to a solution in which the location is equal to the minimum of the data, the shape parameter is less than 1, or both. The table of critical values does not include this case." + suggestion);
        (double, double) F(double mm, double uu)
        {
            double sxm = 0, sxmlog = 0, slog = 0, sinv = 0, sxm1 = 0;
            for (var i = 0; i < n; i++)
            {
                var xu = x[i] - uu;
                var lxu = Math.Log(xu);
                var p = Math.Pow(xu, mm);
                sxm += p; sxmlog += p * lxu; slog += lxu; sinv += 1 / xu; sxm1 += p / xu;
            }
            return (1 / mm - sxmlog / sxm + slog / n, (mm - 1) / mm * sinv - n * sxm1 / sxm);
        }
        // Damped Newton with a forward-difference Jacobian; u stays below the minimum.
        var (f1, f2) = F(m, u);
        for (var it = 0; it < 200; it++)
        {
            var norm = Math.Abs(f1) + Math.Abs(f2);
            if (norm < 1e-14 * n) break;
            var hm = 1e-7 * Math.Max(1, Math.Abs(m));
            var hu = 1e-7 * Math.Max(1e-3, Math.Abs(xmin - u));
            var (a1, a2) = F(m + hm, u);
            var (b1, b2) = F(m, u - hu);
            double j11 = (a1 - f1) / hm, j21 = (a2 - f2) / hm, j12 = -(b1 - f1) / hu, j22 = -(b2 - f2) / hu;
            var det = j11 * j22 - j12 * j21;
            if (det == 0 || !double.IsFinite(det)) break;
            var dm = -(j22 * f1 - j12 * f2) / det;
            var du = -(-j21 * f1 + j11 * f2) / det;
            var t = 1.0;
            var improved = false;
            for (var ls = 0; ls < 60; ls++)
            {
                var mn = m + t * dm; var un = u + t * du;
                if (mn > 0 && un < xmin)
                {
                    var (g1, g2) = F(mn, un);
                    if (double.IsFinite(g1) && double.IsFinite(g2) && Math.Abs(g1) + Math.Abs(g2) < norm)
                    {
                        m = mn; u = un; f1 = g1; f2 = g2; improved = true;
                        break;
                    }
                }
                t *= 0.5;
            }
            if (!improved) break;
        }
        if (!(Math.Abs(f1) < 1e-6 && Math.Abs(f2) < 1e-6 * n))
            throw new ArgumentException("Solution of MLE first-order conditions failed; `anderson` cannot continue." + suggestion);
        double s = 0;
        for (var i = 0; i < n; i++) s += Math.Pow(x[i] - u, m) / n;
        return (m, u, Math.Pow(s, 1 / m));
    }

    /// <summary>
    /// Anderson–Darling test that the data come from a family with estimated parameters, scipy's
    /// <c>anderson</c>: <c>"norm"</c>, <c>"expon"</c>, <c>"logistic"</c>, <c>"gumbel_r"</c>,
    /// <c>"gumbel_l"</c> (alias <c>"gumbel"</c>/<c>"extreme1"</c>) and <c>"weibull_min"</c> (three-parameter
    /// MLE, Lockhart &amp; Stephens' asymptotic critical values with significance levels given as CDF
    /// levels 0.5...0.995, as scipy 1.17). Other critical values are Stephens' (1974,
    /// 1977) small-sample-adjusted tables rounded to 3 decimals, as scipy; <see cref="AndersonResult.Pvalue"/>
    /// interpolates them (scipy's <c>method='interpolate'</c>). The logistic fit is solved to full
    /// precision where scipy stops at fsolve's xtol = 1e-5, so the statistic can differ in the 6th digit.
    /// </summary>
    public static AndersonResult Anderson(double[] x, string dist = "norm")
    {
        ArgumentNullException.ThrowIfNull(x);
        var d = dist.ToLowerInvariant();
        if (d is "extreme1" or "gumbel") d = "gumbel_l";
        var y = StatsTestMath.Sorted(x);
        var n = y.Length;
        if (n == 0) throw new ArgumentException("x must not be empty.", nameof(x));
        var xbar = StatsTestMath.Mean(x);
        double[] logcdf = new double[n], logsf = new double[n];
        double[] sig, crit;
        double loc, scale;
        switch (d)
        {
            case "norm":
            {
                var s = Math.Sqrt(StatsTestMath.Var(x, 1));
                for (var i = 0; i < n; i++)
                {
                    var w = (y[i] - xbar) / s;
                    logcdf[i] = Special.LogNdtr(w);
                    logsf[i] = Special.LogNdtr(-w);
                }
                (loc, scale) = (xbar, s);
                sig = [15, 10, 5, 2.5, 1];
                crit = AvalsNorm.Select(v => Math.Round(v / (1.0 + 0.75 / n + 2.25 / n / n), 3, MidpointRounding.ToEven)).ToArray();
                break;
            }
            case "expon":
            {
                for (var i = 0; i < n; i++)
                {
                    var w = y[i] / xbar;
                    logcdf[i] = w < 0 ? double.NegativeInfinity : Math.Log(-Special.Expm1(-w));
                    logsf[i] = w < 0 ? 0 : -w;
                }
                (loc, scale) = (0, xbar);
                sig = [15, 10, 5, 2.5, 1];
                crit = AvalsExpon.Select(v => Math.Round(v / (1.0 + 0.6 / n), 3, MidpointRounding.ToEven)).ToArray();
                break;
            }
            case "logistic":
            {
                (loc, scale) = StatsKsKernels.LogisticFit(x, xbar, Math.Sqrt(StatsTestMath.Var(x, 1)));
                for (var i = 0; i < n; i++)
                {
                    var w = (y[i] - loc) / scale;
                    logcdf[i] = -Special.Log1p(Math.Exp(-w));
                    logsf[i] = -Special.Log1p(Math.Exp(w));
                    if (double.IsInfinity(logcdf[i]) || double.IsInfinity(logsf[i]))
                    {
                        logcdf[i] = w < -30 ? w : logcdf[i];
                        logsf[i] = w > 30 ? -w : logsf[i];
                    }
                }
                sig = [25, 10, 5, 2.5, 1, 0.5];
                crit = AvalsLogistic.Select(v => Math.Round(v / (1.0 + 0.25 / n), 3, MidpointRounding.ToEven)).ToArray();
                break;
            }
            case "gumbel_r":
            case "gumbel_l":
            {
                var right = d == "gumbel_r";
                ContinuousDistribution fit = right ? GumbelRDistribution.Fit(x) : GumbelLDistribution.Fit(x);
                (loc, scale) = (fit.Loc, fit.Scale);
                for (var i = 0; i < n; i++)
                {
                    var w = (y[i] - loc) / scale;
                    if (right)
                    {
                        logcdf[i] = -Math.Exp(-w);
                        logsf[i] = Math.Log(-Special.Expm1(-Math.Exp(-w)));
                    }
                    else
                    {
                        logcdf[i] = Math.Log(-Special.Expm1(-Math.Exp(w)));
                        logsf[i] = -Math.Exp(w);
                    }
                }
                sig = [25, 10, 5, 2.5, 1];
                crit = AvalsGumbel.Select(v => Math.Round(v / (1.0 + 0.2 / Math.Sqrt(n)), 3, MidpointRounding.ToEven)).ToArray();
                break;
            }
            case "weibull_min":
            {
                var (m, u, sc) = AndersonWeibullFit(y);
                for (var i = 0; i < n; i++)
                {
                    var w = Math.Pow((y[i] - u) / sc, m);
                    logcdf[i] = Math.Log(-Special.Expm1(-w));
                    logsf[i] = -w;
                }
                var cTab = 1 / m;
                sig = [0.5, 0.75, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995];
                crit = new double[8];
                // Lockhart & Stephens (1994) table 1, linear in c = 1/m over [0, 0.5], last row beyond;
                // rounded up (+0.0005) to stay conservative, as scipy.
                var pos = Math.Clamp(cTab / 0.05, 0, 10);
                var lo = (int)Math.Min(Math.Floor(pos), 9);
                var f = pos - lo;
                for (var j = 0; j < 8; j++)
                {
                    var v = cTab >= 0.5 ? AvalsWeibull[10, j] : AvalsWeibull[lo, j] + f * (AvalsWeibull[lo + 1, j] - AvalsWeibull[lo, j]);
                    crit[j] = Math.Round(v + 0.0005, 3, MidpointRounding.ToEven);
                }
                var termsW = new double[n];
                for (var i = 0; i < n; i++) termsW[i] = (2 * (i + 1) - 1.0) / n * (logcdf[i] + logsf[n - 1 - i]);
                return new AndersonResult(-n - StatsTestMath.Sum(termsW), crit, sig, u, sc) { Distribution = "weibull_min", FitShape = m };
            }
            default:
                throw new ArgumentException("Invalid distribution; dist must be one of norm, expon, gumbel_l, gumbel_r, logistic, weibull_min.", nameof(dist));
        }
        var terms = new double[n];
        for (var i = 0; i < n; i++) terms[i] = (2 * (i + 1) - 1.0) / n * (logcdf[i] + logsf[n - 1 - i]);
        var a2 = -n - StatsTestMath.Sum(terms);
        return new AndersonResult(a2, crit, sig, loc, scale) { Distribution = d };
    }

    /// <summary>
    /// k-sample Anderson–Darling test (Scholz &amp; Stephens 1987), scipy's <c>anderson_ksamp</c>.
    /// <paramref name="variant"/>: <c>"midrank"</c> (the default for discrete or tied data),
    /// <c>"right"</c> or <c>"continuous"</c>. The p-value interpolates the tabulated critical values
    /// (quadratic fit in log p) and is capped to [0.001, 0.25] as scipy.
    /// </summary>
    public static AndersonKsampResult AndersonKsamp(double[][] samples, string variant = "midrank")
    {
        StatsTestMath.CheckSamples(samples, 2);
        var k = samples.Length;
        var Z = StatsTestMath.Sorted(StatsTestMath.Concat(samples));
        var N = Z.Length;
        var zstar = Z.Distinct().ToArray();
        if (zstar.Length < 2) throw new ArgumentException("anderson_ksamp needs more than one distinct observation");
        var n = samples.Select(s => s.Length).ToArray();
        if (n.Any(v => v == 0)) throw new ArgumentException("anderson_ksamp encountered sample without observations");
        var a2kn = variant.ToLowerInvariant() switch
        {
            "midrank" => StatsKsKernels.AndersonMidrank(samples, Z, zstar, n, N),
            "right" => StatsKsKernels.AndersonRight(samples, Z, zstar, n, N),
            "continuous" => StatsKsKernels.AndersonContinuous(samples, Z, n, N),
            _ => throw new ArgumentException("`variant` must be one of 'midrank', 'right', or 'continuous'.", nameof(variant)),
        };
        var H = n.Sum(v => 1.0 / v);
        var hsCs = new double[N - 2];
        var acc = 0.0;
        for (var i = 0; i < N - 2; i++) { acc += 1.0 / (N - 1 - i); hsCs[i] = acc; }
        var h = hsCs[^1] + 1;
        var g = 0.0;
        for (var i = 0; i < N - 2; i++) g += hsCs[i] / (i + 2);
        double kk = k, NN = N;
        var a = (4 * g - 6) * (kk - 1) + (10 - 6 * g) * H;
        var b = (2 * g - 4) * kk * kk + 8 * h * kk + (2 * g - 14 * h - 4) * H - 8 * h + 4 * g - 6;
        var c = (6 * h + 2 * g - 2) * kk * kk + (4 * h - 4 * g + 6) * kk + (2 * h - 6) * H + 4 * h;
        var d = (2 * h + 6) * kk * kk - 4 * h * kk;
        var sigmasq = (a * NN * NN * NN + b * NN * NN + c * NN + d) / ((NN - 1) * (NN - 2) * (NN - 3));
        var m = k - 1.0;
        var A2 = (a2kn - m) / Math.Sqrt(sigmasq);
        double[] b0 = [0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085];
        double[] b1 = [-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615];
        double[] b2 = [-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154];
        var critical = new double[7];
        for (var i = 0; i < 7; i++) critical[i] = b0[i] + b1[i] / Math.Sqrt(m) + b2[i] / m;
        double[] sig = [0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001];
        double p;
        if (A2 < critical.Min()) p = 0.25;
        else if (A2 > critical.Max()) p = 0.001;
        else
        {
            var coef = StatsKsKernels.PolyFit2(critical, sig.Select(v => Math.Log(v)).ToArray());
            p = Math.Exp((coef[0] * A2 + coef[1]) * A2 + coef[2]);
        }
        return new AndersonKsampResult(A2, critical, p);
    }

    // ------------------------------------------------------------------ normality

    /// <summary>
    /// Shapiro–Wilk test of normality, scipy's <c>shapiro</c> — Royston's (1995) algorithm AS R94 for
    /// 3 ≤ n ≤ 5000, with the median subtracted first as scipy does.
    /// </summary>
    public static SignificanceResult Shapiro(double[] x)
    {
        ArgumentNullException.ThrowIfNull(x);
        var n = x.Length;
        if (n < 3) throw new ArgumentException("Data must be at least length 3.", nameof(x));
        var y = StatsTestMath.Sorted(x);
        var shift = x[n / 2];
        for (var i = 0; i < n; i++) y[i] -= shift;
        var (w, pw) = StatsKsKernels.Swilk(y);
        return new SignificanceResult(w, pw);
    }

    /// <summary>
    /// D'Agostino's test that the skewness is that of a normal distribution, scipy's <c>skewtest</c>
    /// (n ≥ 8).
    /// </summary>
    public static SignificanceResult Skewtest(double[] a, Alternative alternative = Alternative.TwoSided)
    {
        ArgumentNullException.ThrowIfNull(a);
        var z = SkewtestZ(a);
        return new SignificanceResult(z, StatsTestMath.NormPvalue(z, alternative));
    }

    private static double SkewtestZ(double[] a)
    {
        var b2 = Skew(a);
        double n = a.Length;
        if (n < 8) throw new ArgumentException($"skewtest is not valid with less than 8 samples; {n} samples were given.", nameof(a));
        var y = b2 * Math.Sqrt((n + 1) * (n + 3) / (6.0 * (n - 2)));
        var beta2 = 3.0 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2.0) * (n + 5) * (n + 7) * (n + 9));
        var w2 = -1 + Math.Sqrt(2 * (beta2 - 1));
        var delta = 1 / Math.Sqrt(0.5 * Math.Log(w2));
        var alpha = Math.Sqrt(2.0 / (w2 - 1));
        if (y == 0) y = 1;
        return delta * Math.Log(y / alpha + Math.Sqrt((y / alpha) * (y / alpha) + 1));
    }

    /// <summary>
    /// Anscombe–Glynn test that the kurtosis is that of a normal distribution, scipy's
    /// <c>kurtosistest</c> (n ≥ 5).
    /// </summary>
    public static SignificanceResult Kurtosistest(double[] a, Alternative alternative = Alternative.TwoSided)
    {
        ArgumentNullException.ThrowIfNull(a);
        var z = KurtosistestZ(a);
        return new SignificanceResult(z, StatsTestMath.NormPvalue(z, alternative));
    }

    private static double KurtosistestZ(double[] a)
    {
        double n = a.Length;
        if (n < 5) throw new ArgumentException($"kurtosistest requires at least 5 observations; {n} observations were given.", nameof(a));
        var b2 = Kurtosis(a, fisher: false);
        var e = 3.0 * (n - 1) / (n + 1);
        var varb2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1) * (n + 3) * (n + 5));
        var x = (b2 - e) / Math.Sqrt(varb2);
        var sqrtbeta1 = 6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) * Math.Sqrt(6.0 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)));
        var A = 6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + Math.Sqrt(1 + 4.0 / (sqrtbeta1 * sqrtbeta1)));
        var term1 = 1 - 2 / (9.0 * A);
        var denom = 1 + x * Math.Sqrt(2 / (A - 4.0));
        var term2 = denom == 0 ? double.NaN : Math.Sign(denom) * Math.Cbrt((1 - 2.0 / A) / Math.Abs(denom));
        return (term1 - term2) / Math.Sqrt(2 / (9.0 * A));
    }

    /// <summary>
    /// D'Agostino–Pearson omnibus normality test (skewtest² + kurtosistest², χ²₂), scipy's <c>normaltest</c>.
    /// </summary>
    public static SignificanceResult Normaltest(double[] a)
    {
        ArgumentNullException.ThrowIfNull(a);
        var s = SkewtestZ(a);
        var k = KurtosistestZ(a);
        var stat = s * s + k * k;
        return new SignificanceResult(stat, StatsTestMath.Chi2Sf(stat, 2));
    }

    /// <summary>Jarque–Bera normality test n/6 (S² + K²/4), χ²₂ p-value, scipy's <c>jarque_bera</c>.</summary>
    public static SignificanceResult JarqueBera(double[] x)
    {
        ArgumentNullException.ThrowIfNull(x);
        var mu = StatsTestMath.Mean(x);
        var dx = x.Select(v => v - mu).ToArray();
        var s = Skew(dx);
        var k = Kurtosis(dx);
        var stat = x.Length / 6.0 * (s * s + k * k / 4);
        return new SignificanceResult(stat, StatsTestMath.Chi2Sf(stat, 2));
    }

    /// <summary>
    /// Epps–Singleton two-sample test on the empirical characteristic functions at
    /// <paramref name="t"/>/σ (σ = IQR/2 of the pooled data), scipy's <c>epps_singleton_2samp</c>.
    /// </summary>
    public static SignificanceResult EppsSingleton2samp(double[] x, double[] y, double[]? t = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        t ??= [0.4, 0.8];
        int nx = x.Length, ny = y.Length;
        if (nx < 5 || ny < 5) throw new ArgumentException($"x and y should have at least 5 elements, but len(x) = {nx} and len(y) = {ny}.");
        if (t.Any(v => v <= 0)) throw new ArgumentException("t must contain positive elements only.", nameof(t));
        if (!x.All(double.IsFinite) || !y.All(double.IsFinite)) return new SignificanceResult(double.NaN, double.NaN);
        var n = nx + ny;
        var all = x.Concat(y).ToArray();
        var sigma = Iqr(all) / 2;
        var dim = 2 * t.Length;
        double[,] G(double[] v)
        {
            var g = new double[dim, v.Length];
            for (var j = 0; j < t.Length; j++)
                for (var i = 0; i < v.Length; i++)
                {
                    var arg = t[j] / sigma * v[i];
                    g[j, i] = Math.Cos(arg);
                    g[t.Length + j, i] = Math.Sin(arg);
                }
            return g;
        }
        var gx = G(x);
        var gy = G(y);
        double[] Means(double[,] g, int m)
        {
            var r = new double[dim];
            for (var a = 0; a < dim; a++) { var s = 0.0; for (var i = 0; i < m; i++) s += g[a, i]; r[a] = s / m; }
            return r;
        }
        var mx = Means(gx, nx);
        var my = Means(gy, ny);
        var cov = new double[dim * dim];
        for (var a = 0; a < dim; a++)
            for (var b = 0; b < dim; b++)
            {
                double sx = 0, sy = 0;
                for (var i = 0; i < nx; i++) sx += (gx[a, i] - mx[a]) * (gx[b, i] - mx[b]);
                for (var i = 0; i < ny; i++) sy += (gy[a, i] - my[a]) * (gy[b, i] - my[b]);
                cov[a * dim + b] = (double)n / nx * (sx / nx) + (double)n / ny * (sy / ny);
            }
        var eig = Decomposition.SymmetricEigen(new NdArray(cov, dim, dim));
        var vals = eig.Values.AsContiguous().ToArray();
        var vecs = eig.Vectors.AsContiguous().ToArray();
        var maxv = vals.Max(v => Math.Abs(v));
        var tol = maxv * dim * Math.Pow(2, -52);
        var rank = 0;
        var gd = new double[dim];
        for (var a = 0; a < dim; a++) gd[a] = mx[a] - my[a];
        var w = 0.0;
        for (var e = 0; e < dim; e++)
        {
            if (Math.Abs(vals[e]) <= tol) continue;
            rank++;
            var proj = 0.0;
            for (var a = 0; a < dim; a++) proj += vecs[a * dim + e] * gd[a];
            w += proj * proj / vals[e];
        }
        w *= n;
        if (Math.Max(nx, ny) < 25)
            w *= 1.0 / (1.0 + Math.Pow(n, -0.45) + 10.1 * (Math.Pow(nx, -1.7) + Math.Pow(ny, -1.7)));
        return new SignificanceResult(w, StatsTestMath.Chi2Sf(w, rank));
    }
}
