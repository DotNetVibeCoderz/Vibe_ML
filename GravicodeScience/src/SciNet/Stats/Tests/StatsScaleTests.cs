namespace Gravicode.Science.SciNet;

/// <summary>scipy.stats tests of equal variance / scale.</summary>
public static partial class Stats
{
    private static Func<double[], double> ScaleCenter(string center, double proportiontocut) => center switch
    {
        "median" => x => StatsTestMath.Median(x),
        "mean" => x => StatsTestMath.Mean(x),
        "trimmed" => x => TrimMean(x, proportiontocut),
        _ => throw new ArgumentException("center must be 'mean', 'median' or 'trimmed'.", nameof(center)),
    };

    /// <summary>
    /// Levene's test of equal variances, scipy's <c>levene</c>. <paramref name="center"/> <c>"median"</c>
    /// (the default) is the Brown–Forsythe variant, robust to non-normal data; <c>"mean"</c> is Levene's
    /// original; <c>"trimmed"</c> uses the trimmed mean.
    /// </summary>
    public static SignificanceResult Levene(double[][] samples, string center = "median", double proportiontocut = 0.05)
    {
        StatsTestMath.CheckSamples(samples, 2);
        var f = ScaleCenter(center, proportiontocut);
        var k = samples.Length;
        var ntot = samples.Sum(s => s.Length);
        var zij = new double[k][];
        var zbari = new double[k];
        for (var i = 0; i < k; i++)
        {
            var c = f(samples[i]);
            zij[i] = samples[i].Select(v => Math.Abs(v - c)).ToArray();
            zbari[i] = StatsTestMath.Mean(zij[i]);
        }
        var zbar = 0.0;
        for (var i = 0; i < k; i++) zbar += samples[i].Length * zbari[i];
        zbar /= ntot;
        var numer = 0.0;
        for (var i = 0; i < k; i++) numer += samples[i].Length * (zbari[i] - zbar) * (zbari[i] - zbar);
        numer *= ntot - k;
        var denom = 0.0;
        for (var i = 0; i < k; i++) foreach (var z in zij[i]) denom += (z - zbari[i]) * (z - zbari[i]);
        denom *= k - 1.0;
        var w = numer / denom;
        return new SignificanceResult(w, StatsTestMath.FSf(w, k - 1.0, ntot - k));
    }

    /// <summary>Levene's test over the given samples with the median center.</summary>
    public static SignificanceResult Levene(params double[][] samples) => Levene(samples, "median");

    /// <summary>Bartlett's test of equal variances for normal samples, scipy's <c>bartlett</c>.</summary>
    public static SignificanceResult Bartlett(params double[][] samples)
    {
        StatsTestMath.CheckSamples(samples, 2);
        var k = samples.Length;
        var ni = samples.Select(s => (double)s.Length).ToArray();
        var ssq = samples.Select(s => StatsTestMath.Var(s, 1)).ToArray();
        var ntot = ni.Sum();
        var spsq = 0.0;
        for (var i = 0; i < k; i++) spsq += (ni[i] - 1) * ssq[i];
        spsq /= ntot - k;
        var numer = (ntot - k) * Math.Log(spsq);
        for (var i = 0; i < k; i++) numer -= (ni[i] - 1) * Math.Log(ssq[i]);
        var sinv = 0.0;
        for (var i = 0; i < k; i++) sinv += 1 / (ni[i] - 1);
        var denom = 1 + 1.0 / (3 * (k - 1)) * (sinv - 1 / (ntot - k));
        var t = numer / denom;
        var p = StatsTestMath.Chi2Sf(t, k - 1);
        return new SignificanceResult(Math.Max(t, 0), p);
    }

    /// <summary>
    /// Fligner–Killeen test of equal variances (median-centred, normal scores), scipy's <c>fligner</c>.
    /// </summary>
    public static SignificanceResult Fligner(double[][] samples, string center = "median", double proportiontocut = 0.05)
    {
        StatsTestMath.CheckSamples(samples, 2);
        if (samples.Any(s => s.Length == 0)) return new SignificanceResult(double.NaN, double.NaN);
        var f = ScaleCenter(center, proportiontocut);
        var k = samples.Length;
        var all = new List<double>();
        foreach (var s in samples)
        {
            var c = f(s);
            all.AddRange(s.Select(v => Math.Abs(v - c)));
        }
        var n = all.Count;
        var ranks = Rankdata(all.ToArray());
        var a = ranks.Select(r => Special.Ndtri(r / (2 * (n + 1.0)) + 0.5)).ToArray();
        var abar = StatsTestMath.Mean(a);
        var v2 = StatsTestMath.Var(a, 1);
        var stat = 0.0;
        var o = 0;
        foreach (var s in samples)
        {
            var ai = StatsTestMath.Mean(a.AsSpan(o, s.Length));
            stat += s.Length * (ai - abar) * (ai - abar);
            o += s.Length;
        }
        stat /= v2;
        return new SignificanceResult(stat, StatsTestMath.Chi2Sf(stat, k - 1));
    }

    /// <summary>Fligner–Killeen test over the given samples with the median center.</summary>
    public static SignificanceResult Fligner(params double[][] samples) => Fligner(samples, "median");

    /// <summary>
    /// Mood's two-sample test for scale, scipy's <c>mood</c>, with Mielke's (1967) tie-corrected
    /// statistic and variance when ties are present.
    /// </summary>
    public static SignificanceResult Mood(double[] x, double[] y, Alternative alternative = Alternative.TwoSided)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        int m = x.Length, n = y.Length, N = m + n;
        if (m == 0 || n == 0 || N < 3) return new SignificanceResult(double.NaN, double.NaN);
        var xy = x.Concat(y).ToArray();
        var r = StatsTestMath.RankAverage(xy, out var tieSum);
        double z;
        var c = (N + 1) / 2.0;
        if (tieSum == 0)
        {
            var M = 0.0;
            for (var i = 0; i < m; i++) M += (r[i] - c) * (r[i] - c);
            var e0 = m * ((double)N * N - 1.0) / 12;
            var varM = m * (double)n * (N + 1.0) * (N + 2) * (N - 2) / 180;
            z = (M - e0) / Math.Sqrt(varM);
        }
        else
        {
            // Walk the tie groups of the pooled sorted data: each group with t members spans ranks
            // S_{i−1}+1..S_i and contributes its mean squared deviation from c per x member.
            var keys = (double[])xy.Clone();
            var isX = new bool[N];
            for (var i = 0; i < m; i++) isX[i] = true;
            Array.Sort(keys, isX);
            double T = 0, tieTerm = 0;
            var start = 0;
            while (start < N)
            {
                var end = start + 1;
                while (end < N && keys[end] == keys[start]) end++;
                double t = end - start;
                double sPrev = start, sCur = end;
                var phi = 0.0;
                for (var rr = start + 1; rr <= end; rr++) phi += (rr - c) * (rr - c);
                phi /= t;
                var xs = 0;
                for (var q = start; q < end; q++) if (isX[q]) xs++;
                T += xs * phi;
                tieTerm += t * (t * t - 1) * (t * t - 4 + 15 * (N - sCur - sPrev) * (N - sCur - sPrev));
                start = end;
            }
            var e0 = m * ((double)N * N - 1) / 12;
            var varM = m * (double)n * (N + 1.0) * ((double)N * N - 4) / 180 - m * (double)n / (180.0 * N * (N - 1)) * tieTerm;
            z = (T - e0) / Math.Sqrt(varM);
        }
        return new SignificanceResult(z, StatsTestMath.NormPvalue(z, alternative));
    }

    /// <summary>
    /// Ansari–Bradley test for equal scale, scipy's <c>ansari</c>: exact null distribution when both
    /// samples have fewer than 55 values and there are no ties, normal approximation otherwise (with
    /// the tie-corrected variance when needed).
    /// </summary>
    public static SignificanceResult Ansari(double[] x, double[] y, Alternative alternative = Alternative.TwoSided)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        int n = x.Length, m = y.Length;
        if (m < 1) throw new ArgumentException("Not enough other observations.", nameof(y));
        if (n < 1) throw new ArgumentException("Not enough test observations.", nameof(x));
        var N = m + n;
        var xy = x.Concat(y).ToArray();
        var rank = StatsTestMath.RankAverage(xy, out var tieSum);
        var sym = rank.Select(r => Math.Min(r, N - r + 1)).ToArray();
        var ab = 0.0;
        for (var i = 0; i < n; i++) ab += sym[i];
        var repeats = tieSum > 0;
        if (m < 55 && n < 55 && !repeats)
        {
            var (start, freq) = StatsRankDistributions.AnsariFrequencies(n, m);
            var total = freq.Sum();
            double Cdf(double k)
            {
                var ind = (int)Math.Ceiling(k - start);
                var s = 0.0;
                for (var i = 0; i <= Math.Min(ind, freq.Length - 1); i++) s += freq[i];
                return s / total;
            }
            double Sf(double k)
            {
                var ind = Math.Max((int)Math.Floor(k - start), 0);
                var s = 0.0;
                for (var i = ind; i < freq.Length; i++) s += freq[i];
                return s / total;
            }
            var pv = alternative switch
            {
                Alternative.TwoSided => 2.0 * Math.Min(Cdf(ab), Sf(ab)),
                Alternative.Greater => Cdf(ab),
                _ => Sf(ab),
            };
            return new SignificanceResult(ab, Math.Min(pv, 1.0));
        }
        var mnAB = N % 2 == 1 ? n * (N + 1.0) * (N + 1.0) / 4.0 / N : n * (N + 2.0) / 4.0;
        double varAB;
        if (repeats)
        {
            var fac = sym.Sum(v => v * v);
            varAB = N % 2 == 1
                ? m * (double)n * (16 * N * fac - Math.Pow(N + 1, 4)) / (16.0 * N * N * (N - 1))
                : m * (double)n * (16 * fac - N * (N + 2.0) * (N + 2.0)) / (16.0 * N * (N - 1));
        }
        else
        {
            varAB = N % 2 == 1
                ? n * (double)m * (N + 1.0) * (3 + (double)N * N) / (48.0 * N * N)
                : m * (double)n * (N + 2) * (N - 2.0) / 48 / (N - 1.0);
        }
        var z = (mnAB - ab) / Math.Sqrt(varAB);
        return new SignificanceResult(ab, StatsTestMath.NormPvalue(z, alternative));
    }
}
