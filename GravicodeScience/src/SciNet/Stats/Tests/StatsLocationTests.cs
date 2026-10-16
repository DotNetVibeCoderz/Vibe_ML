namespace Gravicode.Science.SciNet;

/// <summary>scipy.stats location tests: t-tests, ANOVA and the rank-based alternatives.</summary>
public static partial class Stats
{
    // ------------------------------------------------------------------ t-tests

    /// <summary>
    /// One-sample t-test of H₀: mean = <paramref name="popmean"/>, scipy's <c>ttest_1samp</c>. The
    /// result's <see cref="TtestResult.ConfidenceInterval"/> is for the population mean.
    /// </summary>
    public static TtestResult Ttest1samp(double[] a, double popmean, Alternative alternative = Alternative.TwoSided, NanPolicy nanPolicy = NanPolicy.Propagate)
    {
        a = StatsTestMath.Prepare(a, nameof(a), nanPolicy, out var nan);
        var n = a.Length;
        if (nan || n == 0) return new TtestResult(double.NaN, double.NaN, n - 1, double.NaN, double.NaN, alternative);
        var mean = StatsTestMath.Mean(a);
        var se = Math.Sqrt(StatsTestMath.Var(a, 1) / n);
        var t = (mean - popmean) / se;
        return new TtestResult(t, StatsTestMath.TPvalue(t, n - 1, alternative), n - 1, mean, se, alternative);
    }

    /// <summary>
    /// Two-sample t-test for independent samples, scipy's <c>ttest_ind</c>: Student's pooled-variance
    /// test, or Welch's when <paramref name="equalVar"/> is false. A positive <paramref name="trim"/>
    /// gives Yuen's trimmed-mean test (winsorised variances, n − 2g effective sizes).
    /// </summary>
    public static TtestResult TtestInd(double[] a, double[] b, bool equalVar = true, Alternative alternative = Alternative.TwoSided,
        double trim = 0.0, NanPolicy nanPolicy = NanPolicy.Propagate)
    {
        a = StatsTestMath.Prepare(a, nameof(a), nanPolicy, out var na);
        b = StatsTestMath.Prepare(b, nameof(b), nanPolicy, out var nb);
        if (!(trim >= 0 && trim < 0.5)) throw new ArgumentException("Trimming percentage should be 0 <= `trim` < .5.", nameof(trim));
        if (na || nb || a.Length == 0 || b.Length == 0)
            return new TtestResult(double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, alternative);
        double v1, v2, m1, m2, n1, n2;
        if (trim == 0)
        {
            v1 = StatsTestMath.Var(a, 1); v2 = StatsTestMath.Var(b, 1);
            m1 = StatsTestMath.Mean(a); m2 = StatsTestMath.Mean(b);
            n1 = a.Length; n2 = b.Length;
        }
        else
        {
            (v1, m1, n1) = TrimVarMeanLen(a, trim);
            (v2, m2, n2) = TrimVarMeanLen(b, trim);
        }
        var (df, denom) = equalVar ? EqualVarDenom(v1, n1, v2, n2) : UnequalVarDenom(v1, n1, v2, n2);
        var t = (m1 - m2) / denom;
        return new TtestResult(t, StatsTestMath.TPvalue(t, df, alternative), df, m1 - m2, denom, alternative);
    }

    private static (double V, double M, double N) TrimVarMeanLen(double[] a, double trim)
    {
        var s = StatsTestMath.Sorted(a);
        var n = s.Length;
        var g = (int)(n * trim);
        double v;
        if (g == 0) v = StatsTestMath.Var(s, 1);
        else
        {
            var w = (double[])s.Clone();
            for (var i = 0; i < g; i++) { w[i] = s[g]; w[n - 1 - i] = s[n - g - 1]; }
            v = StatsTestMath.Var(w, 2 * g + 1);
        }
        return (v, TrimMean(s, trim), n - 2 * g);
    }

    private static (double Df, double Denom) EqualVarDenom(double v1, double n1, double v2, double n2)
    {
        var df = n1 + n2 - 2.0;
        var svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / df;
        return (df, Math.Sqrt(svar * (1.0 / n1 + 1.0 / n2)));
    }

    private static (double Df, double Denom) UnequalVarDenom(double v1, double n1, double v2, double n2)
    {
        var vn1 = v1 / n1;
        var vn2 = v2 / n2;
        var df = (vn1 + vn2) * (vn1 + vn2) / (vn1 * vn1 / (n1 - 1) + vn2 * vn2 / (n2 - 1));
        // scipy replaces an undefined Welch df (both variances zero) by 1.
        if (double.IsNaN(df)) df = 1;
        return (df, Math.Sqrt(vn1 + vn2));
    }

    /// <summary>
    /// t-test for two independent samples given only their summary statistics, scipy's
    /// <c>ttest_ind_from_stats</c> (<paramref name="std1"/> and <paramref name="std2"/> are ddof-1 standard deviations).
    /// </summary>
    public static TtestResult TtestIndFromStats(double mean1, double std1, double nobs1, double mean2, double std2, double nobs2,
        bool equalVar = true, Alternative alternative = Alternative.TwoSided)
    {
        var (df, denom) = equalVar ? EqualVarDenom(std1 * std1, nobs1, std2 * std2, nobs2) : UnequalVarDenom(std1 * std1, nobs1, std2 * std2, nobs2);
        var t = (mean1 - mean2) / denom;
        return new TtestResult(t, StatsTestMath.TPvalue(t, df, alternative), df, mean1 - mean2, denom, alternative);
    }

    /// <summary>
    /// Paired-sample t-test on the differences a − b, scipy's <c>ttest_rel</c>. The confidence interval is
    /// for the mean difference.
    /// </summary>
    public static TtestResult TtestRel(double[] a, double[] b, Alternative alternative = Alternative.TwoSided, NanPolicy nanPolicy = NanPolicy.Propagate)
    {
        var (x, y) = StatsTestMath.PreparePaired(a, b, nanPolicy, out var nan);
        var d = new double[x.Length];
        for (var i = 0; i < d.Length; i++) d[i] = x[i] - y[i];
        if (nan) return new TtestResult(double.NaN, double.NaN, d.Length - 1, double.NaN, double.NaN, alternative);
        return Ttest1samp(d, 0.0, alternative);
    }

    // ------------------------------------------------------------------ ANOVA family

    /// <summary>
    /// One-way ANOVA, scipy's <c>f_oneway</c>; the overload taking <c>equalVar: false</c> gives Welch's ANOVA.
    /// Every group constant but different gives F = ∞ (p = 0); all values equal gives NaN, as scipy.
    /// </summary>
    public static SignificanceResult FOneway(params double[][] samples) => FOneway(samples, true);

    /// <summary><see cref="FOneway(double[][])"/> with the choice of classic (equal variances) or Welch's ANOVA.</summary>
    public static SignificanceResult FOneway(double[][] samples, bool equalVar)
    {
        StatsTestMath.CheckSamples(samples, 2);
        var k = samples.Length;
        foreach (var s in samples)
            if (s.Length == 0) return new SignificanceResult(double.NaN, double.NaN);
        var all = StatsTestMath.Concat(samples);
        var bign = all.Length;
        if (bign <= k) return new SignificanceResult(double.NaN, double.NaN);
        var allConst = samples.All(s => s.All(v => v == s[0]));
        var allSame = all.All(v => v == all[0]);
        double f, dfn, dfd;
        if (equalVar)
        {
            var offset = StatsTestMath.Mean(all);
            var c = new double[bign];
            for (var i = 0; i < bign; i++) c[i] = all[i] - offset;
            var sumc = StatsTestMath.Sum(c);
            var normalizedSs = sumc * sumc / bign;
            var sq = 0.0;
            foreach (var v in c) sq += v * v;
            var sstot = sq - normalizedSs;
            var ssbn = 0.0;
            foreach (var s in samples)
            {
                var t = new double[s.Length];
                for (var i = 0; i < s.Length; i++) t[i] = s[i] - offset;
                var st = StatsTestMath.Sum(t);
                ssbn += st * st / s.Length;
            }
            ssbn -= normalizedSs;
            var sswn = sstot - ssbn;
            dfn = k - 1;
            dfd = bign - k;
            f = ssbn / dfn / (sswn / dfd);
        }
        else
        {
            var yt = samples.Select(s => StatsTestMath.Mean(s)).ToArray();
            var nt = samples.Select(s => (double)s.Length).ToArray();
            var wt = new double[k];
            for (var i = 0; i < k; i++) wt[i] = nt[i] / StatsTestMath.Var(samples[i], 1);
            var sw = wt.Sum();
            var yhat = 0.0;
            for (var i = 0; i < k; i++) yhat += wt[i] * yt[i];
            yhat /= sw;
            double num = 0, lam = 0;
            for (var i = 0; i < k; i++)
            {
                num += wt[i] * (yt[i] - yhat) * (yt[i] - yhat);
                lam += 1 / (nt[i] - 1) * (1 - wt[i] / sw) * (1 - wt[i] / sw);
            }
            num /= k - 1;
            f = num / (1 + 2.0 * (k - 2) / (k * (double)k - 1) * lam);
            dfn = k - 1;
            dfd = (k * (double)k - 1) / (3 * lam);
        }
        if (allConst) f = double.PositiveInfinity;
        if (allSame) f = double.NaN;
        var p = double.IsNaN(f) ? double.NaN : double.IsPositiveInfinity(f) ? 0.0 : Special.Fdtrc(dfn, dfd, f);
        return new SignificanceResult(f, p);
    }

    /// <summary>
    /// Alexander–Govern test of equal means without assuming equal variances, scipy's
    /// <c>alexandergovern</c>: each group's t statistic against the variance-weighted mean is normalised
    /// by the Hill (1970) expansion and the squares summed into a χ²(k − 1) statistic.
    /// </summary>
    public static SignificanceResult AlexanderGovern(params double[][] samples)
    {
        StatsTestMath.CheckSamples(samples, 2);
        foreach (var s in samples) if (s.Length <= 1) throw new ArgumentException("Input sample size must be greater than one.", nameof(samples));
        var k = samples.Length;
        var means = new double[k];
        var se2 = new double[k];
        var se = new double[k];
        for (var i = 0; i < k; i++)
        {
            means[i] = StatsTestMath.Mean(samples[i]);
            se2[i] = StatsTestMath.Var(samples[i], 1) / samples[i].Length;
            se[i] = Math.Sqrt(se2[i]);
            if (se[i] <= Math.Abs(Math.Pow(2, -52) * means[i])) se[i] = double.NaN;
        }
        var winv = 0.0;
        for (var i = 0; i < k; i++) winv += 1 / se2[i];
        var varW = 0.0;
        for (var i = 0; i < k; i++) varW += 1 / se2[i] / winv * means[i];
        var A = 0.0;
        for (var i = 0; i < k; i++)
        {
            var t = (means[i] - varW) / se[i];
            var v = samples[i].Length - 1.0;
            var a = v - 0.5;
            var b = 48 * a * a;
            var c = Math.Sqrt(a * Math.Log(1 + t * t / v));
            var c3 = c * c * c; var c5 = c3 * c * c; var c7 = c5 * c * c;
            var z = c + (c3 + 3 * c) / b - (4 * c7 + 33 * c5 + 240 * c3 + 855 * c) / (b * b * 10 + 8 * b * c * c * c * c + 1000 * b);
            A += z * z;
        }
        return new SignificanceResult(A, StatsTestMath.Chi2Sf(A, k - 1));
    }

    /// <summary>
    /// Kruskal–Wallis H-test, scipy's <c>kruskal</c>, with the tie correction 1 − Σ(t³ − t)/(N³ − N).
    /// Throws when every value is identical (the correction is zero), as scipy.
    /// </summary>
    public static SignificanceResult Kruskal(params double[][] samples)
    {
        StatsTestMath.CheckSamples(samples, 2);
        foreach (var s in samples) if (s.Length == 0) return new SignificanceResult(double.NaN, double.NaN);
        var all = StatsTestMath.Concat(samples);
        if (all.Any(double.IsNaN)) return new SignificanceResult(double.NaN, double.NaN);
        var n = (double)all.Length;
        var ranks = StatsTestMath.RankAverage(all, out var tieSum);
        var ties = 1 - tieSum / (n * n * n - n);
        if (ties == 0) throw new ArgumentException("All numbers are identical in kruskal.");
        var h = 0.0;
        var o = 0;
        foreach (var s in samples)
        {
            var rs = 0.0;
            for (var i = 0; i < s.Length; i++) rs += ranks[o + i];
            o += s.Length;
            h += rs * rs / s.Length;
        }
        h = 12.0 / (n * (n + 1)) * h - 3 * (n + 1);
        h /= ties;
        return new SignificanceResult(h, StatsTestMath.Chi2Sf(h, samples.Length - 1));
    }

    /// <summary>
    /// Friedman test for repeated measurements, scipy's <c>friedmanchisquare</c>: each argument is one
    /// treatment measured on the same n subjects; ranks are taken within subjects.
    /// </summary>
    public static SignificanceResult FriedmanChiSquare(params double[][] samples)
    {
        StatsTestMath.CheckSamples(samples, 3);
        var k = samples.Length;
        var n = samples[0].Length;
        foreach (var s in samples) if (s.Length != n) throw new ArgumentException("Unequal N in friedmanchisquare.  Aborting.", nameof(samples));
        if (n == 0) throw new ArgumentException("One or more sample arguments is too small.", nameof(samples));
        var colSums = new double[k];
        var ties = 0.0;
        var row = new double[k];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < k; j++) row[j] = samples[j][i];
            var r = StatsTestMath.RankAverage(row, out var t);
            ties += t;
            for (var j = 0; j < k; j++) colSums[j] += r[j];
        }
        var c = 1 - ties / (k * ((double)k * k - 1) * n);
        var ssbn = 0.0;
        foreach (var s in colSums) ssbn += s * s;
        var stat = (12.0 / (k * n * (k + 1.0)) * ssbn - 3.0 * n * (k + 1)) / c;
        return new SignificanceResult(stat, StatsTestMath.Chi2Sf(stat, k - 1));
    }

    // ------------------------------------------------------------------ two-sample rank tests

    /// <summary>
    /// Mann–Whitney U test, scipy's <c>mannwhitneyu</c>. The statistic is U₁ for <paramref name="x"/>.
    /// <paramref name="method"/>: <c>"auto"</c> (exact when either sample has ≤ 8 values and there are no
    /// ties), <c>"exact"</c> (the null distribution by the Löffler/Harding recursion; ties are ignored by
    /// it, as scipy), or <c>"asymptotic"</c> (normal with tie-corrected variance and optional continuity
    /// correction).
    /// </summary>
    public static SignificanceResult MannWhitneyU(double[] x, double[] y, bool useContinuity = true, Alternative alternative = Alternative.TwoSided, string method = "auto")
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (x.Any(double.IsNaN) || y.Any(double.IsNaN)) throw new ArgumentException("`x` and `y` must not contain NaNs.");
        if (x.Length == 0 || y.Length == 0) throw new ArgumentException("`x` and `y` must be of nonzero size.");
        int n1 = x.Length, n2 = y.Length;
        var xy = new double[n1 + n2];
        x.CopyTo(xy, 0);
        y.CopyTo(xy, n1);
        var ranks = StatsTestMath.RankAverage(xy, out var tieSum);
        var r1 = StatsTestMath.Sum(ranks.AsSpan(0, n1));
        var u1 = r1 - n1 * (n1 + 1.0) / 2;
        var u2 = (double)n1 * n2 - u1;
        double u;
        double f = 1;
        switch (alternative)
        {
            case Alternative.Greater: u = u1; break;
            case Alternative.Less: u = u2; break;
            default: u = Math.Max(u1, u2); f = 2; break;
        }
        var m = method.ToLowerInvariant();
        if (m == "auto") m = (n1 > 8 && n2 > 8) || tieSum > 0 ? "asymptotic" : "exact";
        double p;
        if (m == "exact")
            p = StatsRankDistributions.MwuSf(n1, n2, (long)u);
        else if (m == "asymptotic")
        {
            var mu = n1 * (double)n2 / 2;
            var n = (double)(n1 + n2);
            var s = Math.Sqrt(n1 * (double)n2 / 12 * ((n + 1) - tieSum / (n * (n - 1))));
            var num = u - mu;
            if (useContinuity) num -= 0.5;
            p = Special.Ndtr(-(num / s));
        }
        else throw new ArgumentException("method must be one of auto, exact, asymptotic.", nameof(method));
        p = Math.Clamp(p * f, 0.0, 1.0);
        return new SignificanceResult(u1, p);
    }

    /// <summary>
    /// Wilcoxon rank-sum test (normal approximation, no tie correction), scipy's <c>ranksums</c>.
    /// </summary>
    public static SignificanceResult Ranksums(double[] x, double[] y, Alternative alternative = Alternative.TwoSided)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        int n1 = x.Length, n2 = y.Length;
        var all = new double[n1 + n2];
        x.CopyTo(all, 0);
        y.CopyTo(all, n1);
        var ranks = Rankdata(all);
        var s = 0.0;
        for (var i = 0; i < n1; i++) s += ranks[i];
        var expected = n1 * (n1 + n2 + 1) / 2.0;
        var z = (s - expected) / Math.Sqrt(n1 * (double)n2 * (n1 + n2 + 1) / 12.0);
        return new SignificanceResult(z, StatsTestMath.NormPvalue(z, alternative));
    }

    /// <summary>
    /// Wilcoxon signed-rank test on <paramref name="x"/> − <paramref name="y"/> (or on <paramref name="x"/>
    /// alone), scipy's <c>wilcoxon</c>. <paramref name="zeroMethod"/>: <c>"wilcox"</c> drops zero
    /// differences, <c>"pratt"</c> ranks then drops them, <c>"zsplit"</c> splits their ranks.
    /// <paramref name="method"/> <c>"auto"</c> follows scipy: exact for n ≤ 50 without ties or zeros, an
    /// exact sign-flip permutation test for n ≤ 13 otherwise, the normal approximation beyond.
    /// The statistic is min(R⁺, R⁻) two-sided, R⁺ one-sided.
    /// </summary>
    public static SignificanceResult Wilcoxon(double[] x, double[]? y = null, string zeroMethod = "wilcox", bool correction = false,
        Alternative alternative = Alternative.TwoSided, string method = "auto")
    {
        ArgumentNullException.ThrowIfNull(x);
        double[] d;
        if (y is null) d = (double[])x.Clone();
        else
        {
            if (y.Length != x.Length) throw new ArgumentException("`x` and `y` must have the same length along `axis`.", nameof(y));
            d = new double[x.Length];
            for (var i = 0; i < d.Length; i++) d[i] = x[i] - y[i];
        }
        var zm = zeroMethod.ToLowerInvariant();
        if (zm is not ("wilcox" or "pratt" or "zsplit")) throw new ArgumentException("zero_method must be one of wilcox, pratt, zsplit.", nameof(zeroMethod));
        var meth = method.ToLowerInvariant();
        if (meth is not ("auto" or "exact" or "asymptotic")) throw new ArgumentException("method must be one of auto, exact, asymptotic.", nameof(method));
        if (d.Length == 0 || d.Any(double.IsNaN)) return new SignificanceResult(double.NaN, double.NaN);
        var nZero = d.Count(v => v == 0);
        if (meth == "auto" && d.Length > 50) meth = "asymptotic";
        var st = StatsRankDistributions.WilcoxonStatistic(d, zm);
        if (meth == "auto")
        {
            if (!(st.HasTies || nZero > 0)) meth = "exact";
            else if (d.Length <= 13) meth = "permutation";
            else meth = "asymptotic";
        }
        double p;
        if (meth == "asymptotic")
        {
            var z = (st.RPlus - st.Mean) / st.Se;
            if (correction)
            {
                var sign = alternative switch { Alternative.Greater => 1.0, Alternative.Less => -1.0, _ => Math.Sign(z) };
                z -= sign * 0.5 / st.Se;
            }
            p = StatsTestMath.NormPvalue(z, alternative);
        }
        else if (meth == "exact")
        {
            var count = st.Count;
            var pmf = StatsRankDistributions.WilcoxonPmf(count);
            double Cdf(double k) => StatsRankDistributions.WilcoxonCdf(pmf, count, (int)k);
            double Sf(double k) => StatsRankDistributions.WilcoxonSf(pmf, count, (int)k);
            p = alternative switch
            {
                Alternative.Less => Cdf(Math.Ceiling(st.RPlus)),
                Alternative.Greater => Sf(Math.Floor(st.RPlus)),
                _ => Math.Clamp(2 * Math.Min(Sf(Math.Floor(st.RPlus)), Cdf(Math.Ceiling(st.RPlus))), 0, 1),
            };
        }
        else
        {
            // Exact sign-flip enumeration: scipy's PermutationMethod with 9999 resamples is exhaustive
            // for n ≤ 13 (2¹³ = 8192 patterns).
            p = StatsRankDistributions.WilcoxonSignFlipPvalue(d, zm, st.RPlus, alternative);
        }
        var stat = alternative == Alternative.TwoSided ? Math.Min(st.RPlus, st.RMinus) : st.RPlus;
        return new SignificanceResult(stat, p);
    }

    /// <summary>
    /// Brunner–Munzel test of stochastic equality, scipy's <c>brunnermunzel</c>, with the t
    /// (Welch–Satterthwaite df) or normal reference distribution.
    /// </summary>
    public static SignificanceResult BrunnerMunzel(double[] x, double[] y, Alternative alternative = Alternative.TwoSided, string distribution = "t")
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        int nx = x.Length, ny = y.Length;
        if (nx == 0 || ny == 0) return new SignificanceResult(double.NaN, double.NaN);
        var all = new double[nx + ny];
        x.CopyTo(all, 0);
        y.CopyTo(all, nx);
        var rc = Rankdata(all);
        var rx = Rankdata(x);
        var ry = Rankdata(y);
        var rcx = rc.AsSpan(0, nx).ToArray();
        var rcy = rc.AsSpan(nx).ToArray();
        double rcxm = StatsTestMath.Mean(rcx), rcym = StatsTestMath.Mean(rcy), rxm = StatsTestMath.Mean(rx), rym = StatsTestMath.Mean(ry);
        double sx = 0, sy = 0;
        for (var i = 0; i < nx; i++) { var t = rcx[i] - rx[i] - rcxm + rxm; sx += t * t; }
        for (var i = 0; i < ny; i++) { var t = rcy[i] - ry[i] - rcym + rym; sy += t * t; }
        sx /= nx - 1;
        sy /= ny - 1;
        var w = nx * (double)ny * (rcym - rcxm) / ((nx + ny) * Math.Sqrt(nx * sx + ny * sy));
        double p;
        if (distribution == "t")
        {
            var dfNum = Math.Pow(nx * sx + ny * sy, 2);
            var dfDen = Math.Pow(nx * sx, 2) / (nx - 1) + Math.Pow(ny * sy, 2) / (ny - 1);
            p = StatsTestMath.TPvalue(-w, dfNum / dfDen, alternative);
        }
        else if (distribution == "normal") p = StatsTestMath.NormPvalue(-w, alternative);
        else throw new ArgumentException("distribution should be 't' or 'normal'", nameof(distribution));
        return new SignificanceResult(w, p);
    }
}
