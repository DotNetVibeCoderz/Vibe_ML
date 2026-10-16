using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>scipy.stats ranking, percentiles, histograms, binned statistics, distances and entropy.</summary>
public static partial class Stats
{
    // ------------------------------------------------------------------ ranks and percentiles

    /// <summary>
    /// Ranks 1…n with ties resolved by <paramref name="method"/> — <c>"average"</c>, <c>"min"</c>,
    /// <c>"max"</c>, <c>"dense"</c> or <c>"ordinal"</c> (first occurrence ranks first) — scipy's
    /// <c>rankdata</c>. Any NaN makes every rank NaN, as scipy's default policy.
    /// </summary>
    public static double[] Rankdata(double[] a, string method = "average")
    {
        ArgumentNullException.ThrowIfNull(a);
        var n = a.Length;
        foreach (var v in a)
            if (double.IsNaN(v)) return Enumerable.Repeat(double.NaN, n).ToArray();
        var keys = (double[])a.Clone();
        var idx = new int[n];
        for (var i = 0; i < n; i++) idx[i] = i;
        Array.Sort(keys, idx);
        var r = new double[n];
        var start = 0;
        var dense = 0;
        var m = method.ToLowerInvariant();
        if (m is not ("average" or "min" or "max" or "dense" or "ordinal")) throw new ArgumentException($"unknown method '{method}'.", nameof(method));
        while (start < n)
        {
            var end = start + 1;
            while (end < n && keys[end] == keys[start]) end++;
            dense++;
            if (m == "ordinal")
            {
                // Array.Sort is unstable; restore first-occurrence order inside the tie run.
                Array.Sort(idx, start, end - start);
                for (var k = start; k < end; k++) r[idx[k]] = k + 1;
            }
            else
            {
                double v = m switch
                {
                    "average" => (start + end + 1) / 2.0,
                    "min" => start + 1,
                    "max" => end,
                    _ => dense,
                };
                for (var k = start; k < end; k++) r[idx[k]] = v;
            }
            start = end;
        }
        return r;
    }

    /// <summary><see cref="Rankdata(double[], string)"/> along an axis.</summary>
    public static NdArray Rankdata(NdArray a, int axis, string method = "average") => StatsAxis.Transform(a, axis, x => Rankdata(x, method));

    /// <summary>
    /// Percentile rank of <paramref name="score"/> in <paramref name="a"/>, scipy's
    /// <c>percentileofscore</c>; <paramref name="kind"/> is <c>"rank"</c>, <c>"weak"</c>,
    /// <c>"strict"</c> or <c>"mean"</c>.
    /// </summary>
    public static double Percentileofscore(double[] a, double score, string kind = "rank")
    {
        ArgumentNullException.ThrowIfNull(a);
        var n = a.Length;
        if (n == 0 || double.IsNaN(score) || a.Any(double.IsNaN)) return double.NaN;
        int left = 0, right = 0;
        foreach (var v in a)
        {
            if (v < score) left++;
            if (v <= score) right++;
        }
        return kind.ToLowerInvariant() switch
        {
            "rank" => (left + right + (left < right ? 1 : 0)) * (50.0 / n),
            "strict" => left * (100.0 / n),
            "weak" => right * (100.0 / n),
            "mean" => (left + right) * (50.0 / n),
            _ => throw new ArgumentException("kind can only be 'rank', 'strict', 'weak' or 'mean'.", nameof(kind)),
        };
    }

    /// <summary>
    /// Score at percentile <paramref name="per"/> (0…100), scipy's <c>scoreatpercentile</c>;
    /// <paramref name="interpolationMethod"/> is <c>"fraction"</c>, <c>"lower"</c> or <c>"higher"</c>.
    /// </summary>
    public static double Scoreatpercentile(double[] a, double per, (double Lower, double Upper)? limit = null, string interpolationMethod = "fraction")
    {
        ArgumentNullException.ThrowIfNull(a);
        var data = a;
        if (limit is { } lim) data = a.Where(v => v >= lim.Lower && v <= lim.Upper).ToArray();
        if (data.Length == 0) return double.NaN;
        var s = StatsTestMath.Sorted(data);
        var idx = per / 100.0 * (s.Length - 1);
        if ((int)idx != idx)
        {
            switch (interpolationMethod.ToLowerInvariant())
            {
                case "lower": idx = Math.Floor(idx); break;
                case "higher": idx = Math.Ceiling(idx); break;
                case "fraction": break;
                default: throw new ArgumentException("interpolation_method can only be 'fraction', 'lower' or 'higher'", nameof(interpolationMethod));
            }
        }
        var i = (int)idx;
        if (i == idx) return s[i];
        return s[i] + (s[i + 1] - s[i]) * (idx - i);
    }

    /// <summary>
    /// Sample quantiles by any of the Hyndman–Fan definitions, scipy 1.15's <c>stats.quantile</c>:
    /// <c>"inverted_cdf"</c>, <c>"averaged_inverted_cdf"</c>, <c>"closest_observation"</c>,
    /// <c>"interpolated_inverted_cdf"</c>, <c>"hazen"</c>, <c>"weibull"</c>, <c>"linear"</c> (numpy's
    /// default), <c>"median_unbiased"</c>, <c>"normal_unbiased"</c>, plus <c>"harrell-davis"</c> and
    /// <c>"round_outward"</c>/<c>"round_inward"</c>/<c>"round_nearest"</c>. Probabilities outside [0, 1] give NaN.
    /// </summary>
    public static double[] Quantile(double[] x, double[] p, string method = "linear")
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(p);
        var y = StatsTestMath.Sorted(x);
        var n = y.Length;
        var r = new double[p.Length];
        for (var k = 0; k < p.Length; k++)
        {
            var pk = p[k];
            if (!(pk >= 0 && pk <= 1) || n == 0) { r[k] = double.NaN; continue; }
            r[k] = method switch
            {
                "harrell-davis" => QuantileHd(y, pk),
                "round_outward" or "round_inward" or "round_nearest" => QuantileWinsor(y, pk, method),
                _ => QuantileHf(y, pk, method),
            };
        }
        return r;
    }

    /// <summary>One quantile; see <see cref="Quantile(double[], double[], string)"/>.</summary>
    public static double Quantile(double[] x, double p, string method = "linear") => Quantile(x, [p], method)[0];

    private static double QuantileHf(double[] y, double p, string method)
    {
        var n = y.Length;
        double m = method switch
        {
            "inverted_cdf" or "averaged_inverted_cdf" or "interpolated_inverted_cdf" => 0,
            "closest_observation" => -0.5,
            "hazen" => 0.5,
            "weibull" => p,
            "linear" => 1 - p,
            "median_unbiased" => p / 3 + 1.0 / 3,
            "normal_unbiased" => p / 4 + 3.0 / 8,
            _ => throw new ArgumentException($"unknown quantile method '{method}'.", nameof(method)),
        };
        var jg = p * n + m;
        var jp1 = Math.Floor(jg);
        var j = jp1 - 1;
        var g = jg - Math.Floor(jg);
        if (method == "inverted_cdf") g = g > 0 ? 1 : 0;
        else if (method == "averaged_inverted_cdf") g = (1 + (g > 0 ? 1 : 0)) / 2.0;
        else if (method == "closest_observation") g = 1 - (g == 0 && Math.Abs(j % 2) == 1 ? 1 : 0);
        if (method is "inverted_cdf" or "averaged_inverted_cdf" or "closest_observation" && jg < 0) g = 0;
        if (j < 0) g = 0;
        var ji = (int)Math.Clamp(j, 0, n - 1);
        var jpi = (int)Math.Clamp(jp1, 0, n - 1);
        return (1 - g) * y[ji] + g * y[jpi];
    }

    private static double QuantileHd(double[] y, double p)
    {
        var n = y.Length;
        double a = p * (n + 1), b = (1 - p) * (n + 1);
        // p = 0 or 1 makes a shape parameter zero: the beta distribution degenerates to a point mass
        // at 0 (a = 0) or 1 (b = 0), and the weights collapse onto the first or last order statistic.
        double I(double t) => a == 0 ? (t > 0 ? 1 : 0) : b == 0 ? (t < 1 ? 0 : 1) : Special.BetaInc(a, b, t);
        var prev = I(0.0);
        var s = 0.0;
        for (var i = 1; i <= n; i++)
        {
            var cur = I((double)i / n);
            var w = cur - prev;
            if (!double.IsNaN(w)) s += w * y[i - 1];
            prev = cur;
        }
        return s;
    }

    private static double QuantileWinsor(double[] y, double p, string method)
    {
        var n = y.Length;
        double j = method switch
        {
            "round_outward" => p < 0.5 ? Math.Floor(p * n) : Math.Ceiling(n * p - 1),
            "round_inward" => p < 0.5 ? Math.Ceiling(p * n) : Math.Floor(n * p - 1),
            _ => p < 0.5 ? Math.Round(p * n, MidpointRounding.ToEven) : Math.Round(n * p - 1, MidpointRounding.ToEven),
        };
        return y[(int)Math.Clamp(j, 0, n - 1)];
    }

    // ------------------------------------------------------------------ histograms

    /// <summary>numpy's uniform-bin histogram with an explicit range, including its edge corrections.</summary>
    internal static double[] UniformHistogram(double[] a, int bins, double lo, double hi, double[]? weights, out double[] edges)
    {
        edges = Linspace(lo, hi, bins + 1);
        var h = new double[bins];
        var norm = bins / (hi - lo);
        for (var i = 0; i < a.Length; i++)
        {
            var v = a[i];
            if (!(v >= lo && v <= hi)) continue;
            var k = (int)((v - lo) * norm);
            if (k == bins) k--;
            if (v < edges[k]) k--;
            else if (k != bins - 1 && v >= edges[k + 1]) k++;
            h[k] += weights?[i] ?? 1.0;
        }
        return h;
    }

    /// <summary>numpy's <c>linspace</c> (start + i·step, last point set exactly).</summary>
    internal static double[] Linspace(double start, double stop, int num)
    {
        var r = new double[num];
        if (num == 1) { r[0] = start; return r; }
        var step = (stop - start) / (num - 1);
        for (var i = 0; i < num; i++) r[i] = i * step + start;
        r[num - 1] = stop;
        return r;
    }

    private static FrequencyResult Histogram(double[] a, int numbins, (double Lower, double Upper)? limits, double[]? weights)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (numbins < 1) throw new ArgumentException("numbins must be positive.", nameof(numbins));
        double lo, hi;
        if (limits is { } l) (lo, hi) = l;
        else if (a.Length == 0) (lo, hi) = (0, 1);
        else
        {
            var mn = a.Min();
            var mx = a.Max();
            var s = (mx - mn) / (2.0 * (numbins - 1.0));
            (lo, hi) = (mn - s, mx + s);
        }
        var h = UniformHistogram(a, numbins, lo, hi, weights, out var edges);
        var extra = a.Count(v => lo > v || v > hi);
        return new FrequencyResult(h, lo, edges[1] - edges[0], extra);
    }

    /// <summary>Cumulative histogram, scipy's <c>cumfreq</c> (default limits pad the range by half a bin).</summary>
    public static FrequencyResult Cumfreq(double[] a, int numbins = 10, (double Lower, double Upper)? defaultreallimits = null, double[]? weights = null)
    {
        var h = Histogram(a, numbins, defaultreallimits, weights);
        var c = new double[h.Frequency.Length];
        var s = 0.0;
        for (var i = 0; i < c.Length; i++) { s += h.Frequency[i]; c[i] = s; }
        return h with { Frequency = c };
    }

    /// <summary>Relative-frequency histogram (counts / n), scipy's <c>relfreq</c>.</summary>
    public static FrequencyResult Relfreq(double[] a, int numbins = 10, (double Lower, double Upper)? defaultreallimits = null, double[]? weights = null)
    {
        var h = Histogram(a, numbins, defaultreallimits, weights);
        var r = new double[h.Frequency.Length];
        for (var i = 0; i < r.Length; i++) r[i] = h.Frequency[i] / a.Length;
        return h with { Frequency = r };
    }

    // ------------------------------------------------------------------ binned statistics

    /// <summary>
    /// A statistic of <paramref name="values"/> within bins of <paramref name="x"/>, scipy's
    /// <c>binned_statistic</c>. <paramref name="statistic"/> is <c>"mean"</c>, <c>"std"</c>, <c>"median"</c>,
    /// <c>"count"</c>, <c>"sum"</c>, <c>"min"</c> or <c>"max"</c>; empty bins hold NaN (0 for count and sum).
    /// </summary>
    public static BinnedStatisticResult BinnedStatistic(double[] x, double[] values, string statistic = "mean", int bins = 10, (double Lower, double Upper)? range = null)
        => BinnedStatisticDd([x], values, statistic, [bins], null, range is null ? null : [range.Value], null);

    /// <summary><see cref="BinnedStatistic(double[], double[], string, int, ValueTuple{double, double}?)"/> with explicit bin edges.</summary>
    public static BinnedStatisticResult BinnedStatistic(double[] x, double[] values, string statistic, double[] binEdges)
        => BinnedStatisticDd([x], values, statistic, null, [binEdges], null, null);

    /// <summary>Binned statistic with a custom reducer (scipy's callable <c>statistic</c>); empty bins get <c>func([])</c> or NaN.</summary>
    public static BinnedStatisticResult BinnedStatistic(double[] x, double[] values, Func<double[], double> statistic, int bins = 10, (double Lower, double Upper)? range = null)
        => BinnedStatisticDd([x], values, "custom", [bins], null, range is null ? null : [range.Value], statistic);

    /// <summary>Two-dimensional binned statistic, scipy's <c>binned_statistic_2d</c>; the result is (nx, ny).</summary>
    public static BinnedStatisticResult BinnedStatistic2d(double[] x, double[] y, double[] values, string statistic = "mean", (int Nx, int Ny)? bins = null,
        ((double Lower, double Upper) X, (double Lower, double Upper) Y)? range = null)
    {
        var b = bins ?? (10, 10);
        return BinnedStatisticDd([x, y], values, statistic, [b.Nx, b.Ny], null, range is { } r ? [r.X, r.Y] : null, null);
    }

    /// <summary>
    /// d-dimensional binned statistic, scipy's <c>binned_statistic_dd</c>. <paramref name="sample"/> holds one
    /// coordinate array per dimension. Give either <paramref name="bins"/> (counts per dimension) or
    /// <paramref name="binEdges"/>. Bin numbers include the outlier bins on each side, as scipy.
    /// </summary>
    public static BinnedStatisticResult BinnedStatisticDd(double[][] sample, double[] values, string statistic = "mean", int[]? bins = null,
        double[][]? binEdges = null, (double Lower, double Upper)[]? range = null, Func<double[], double>? custom = null)
    {
        ArgumentNullException.ThrowIfNull(sample);
        var d = sample.Length;
        if (d == 0) throw new ArgumentException("sample needs at least one dimension.", nameof(sample));
        var n = sample[0].Length;
        foreach (var s in sample) if (s.Length != n) throw new ArgumentException("every dimension of sample must have the same length.", nameof(sample));
        if (statistic != "count" && (values is null || values.Length != n))
            throw new ArgumentException("The number of values elements must match the length of each sample dimension.", nameof(values));
        var edges = new double[d][];
        var nbin = new int[d];
        for (var i = 0; i < d; i++)
        {
            if (binEdges != null) edges[i] = (double[])binEdges[i].Clone();
            else
            {
                double lo, hi;
                if (range != null) (lo, hi) = range[i];
                else if (n == 0) (lo, hi) = (0, 0);
                else { lo = sample[i].Min(); hi = sample[i].Max(); }
                if (hi < lo) throw new ArgumentException("In range, start must be <= stop.", nameof(range));
                if (lo == hi) { lo -= 0.5; hi += 0.5; }
                var nb = bins is null ? 10 : bins.Length == 1 ? bins[0] : bins[i];
                edges[i] = Linspace(lo, hi, nb + 1);
            }
            nbin[i] = edges[i].Length + 1;
        }
        // Bin numbers: digitize (right-closed search), then pull points sitting on the last edge
        // back into the last bin with scipy's decimal rounding test.
        var binnumber = new int[n];
        for (var i = 0; i < d; i++)
        {
            var e = edges[i];
            var dmin = double.PositiveInfinity;
            for (var k = 1; k < e.Length; k++) dmin = Math.Min(dmin, e[k] - e[k - 1]);
            if (dmin == 0) throw new ArgumentException("The smallest edge difference is numerically 0.");
            var dec = (int)(-Math.Log10(dmin)) + 6;
            var last = e[^1];
            var lastR = NpAround(last, dec);
            for (var j = 0; j < n; j++)
            {
                var v = sample[i][j];
                var b = UpperBound(e, v);
                if (v >= last && NpAround(v, dec) == lastR) b--;
                binnumber[j] = binnumber[j] * nbin[i] + b;
            }
        }
        var total = 1;
        foreach (var nb in nbin) total *= nb;
        var result = new double[total];
        var groups = new List<double>?[total];
        var stat = statistic.ToLowerInvariant();
        if (stat is "count" or "sum")
        {
            for (var j = 0; j < n; j++) result[binnumber[j]] += stat == "count" ? 1 : values![j];
        }
        else
        {
            for (var j = 0; j < n; j++) (groups[binnumber[j]] ??= new List<double>()).Add(values![j]);
            double empty = double.NaN;
            if (stat == "custom" && custom != null)
                try { empty = custom([]); } catch { empty = double.NaN; }
            for (var b = 0; b < total; b++)
            {
                var g = groups[b];
                if (g is null) { result[b] = empty; continue; }
                var arr = g.ToArray();
                result[b] = stat switch
                {
                    "mean" => SeqSum(arr) / arr.Length,
                    "std" => BinStd(arr),
                    "median" => StatsTestMath.Median(arr),
                    "min" => arr.Min(),
                    "max" => arr.Max(),
                    "custom" => (custom ?? throw new ArgumentException("custom statistic missing."))(arr),
                    _ => throw new ArgumentException($"unknown statistic '{statistic}'.", nameof(statistic)),
                };
            }
        }
        // Strip the outlier bins from each dimension.
        var core = new int[d];
        var coreTotal = 1;
        for (var i = 0; i < d; i++) { core[i] = nbin[i] - 2; coreTotal *= core[i]; }
        var outArr = new double[coreTotal];
        var idx = new int[d];
        for (var c = 0; c < coreTotal; c++)
        {
            var rem = c;
            for (var i = d - 1; i >= 0; i--) { idx[i] = rem % core[i] + 1; rem /= core[i]; }
            var flat = 0;
            for (var i = 0; i < d; i++) flat = flat * nbin[i] + idx[i];
            outArr[c] = result[flat];
        }
        return new BinnedStatisticResult(new NdArray(outArr, core), edges, binnumber);
    }

    private static double SeqSum(double[] a)
    {
        var s = 0.0;
        foreach (var v in a) s += v;
        return s;
    }

    private static double BinStd(double[] a)
    {
        var m = SeqSum(a) / a.Length;
        var s = 0.0;
        foreach (var v in a) s += (v - m) * (v - m);
        return Math.Sqrt(s / a.Length);
    }

    private static double NpAround(double x, int decimals)
    {
        // numpy rounds by scaling, rounding half to even, then unscaling.
        if (decimals >= 0)
        {
            var f = Math.Pow(10, decimals);
            var y = x * f;
            if (double.IsInfinity(y)) return x;
            return Math.Round(y, MidpointRounding.ToEven) / f;
        }
        var g = Math.Pow(10, -decimals);
        return Math.Round(x / g, MidpointRounding.ToEven) * g;
    }

    /// <summary>Index of the first element strictly greater than <paramref name="v"/> (searchsorted right).</summary>
    internal static int UpperBound(double[] sorted, double v)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            var m = (lo + hi) >>> 1;
            if (sorted[m] <= v) lo = m + 1; else hi = m;
        }
        return lo;
    }

    /// <summary>Index of the first element not less than <paramref name="v"/> (searchsorted left).</summary>
    internal static int LowerBound(double[] sorted, double v)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            var m = (lo + hi) >>> 1;
            if (sorted[m] < v) lo = m + 1; else hi = m;
        }
        return lo;
    }

    // ------------------------------------------------------------------ distances between distributions

    /// <summary>
    /// First Wasserstein (earth mover's) distance between two 1-D empirical distributions, optionally
    /// weighted, scipy's <c>wasserstein_distance</c>: ∫|U(t) − V(t)| dt over the merged support.
    /// </summary>
    public static double WassersteinDistance(double[] uValues, double[] vValues, double[]? uWeights = null, double[]? vWeights = null)
        => CdfDistance(1, uValues, vValues, uWeights, vWeights);

    /// <summary>
    /// Energy distance √2 · (∫(U − V)²)^{1/2} between two 1-D distributions, scipy's <c>energy_distance</c>.
    /// </summary>
    public static double EnergyDistance(double[] uValues, double[] vValues, double[]? uWeights = null, double[]? vWeights = null)
        => Math.Sqrt(2) * CdfDistance(2, uValues, vValues, uWeights, vWeights);

    private static double CdfDistance(int p, double[] u, double[] v, double[]? uw, double[]? vw)
    {
        ArgumentNullException.ThrowIfNull(u);
        ArgumentNullException.ThrowIfNull(v);
        if (u.Length == 0 || v.Length == 0) throw new ArgumentException("Distribution can't be empty.");
        (double[] Sorted, double[]? Cum) Prep(double[] x, double[]? w)
        {
            var s = (double[])x.Clone();
            if (w is null) { Array.Sort(s); return (s, null); }
            if (w.Length != x.Length) throw new ArgumentException("Value and weight array-likes for the same empirical distribution must be of the same size.");
            foreach (var wi in w) if (wi < 0 || !double.IsFinite(wi)) throw new ArgumentException("All weights must be non-negative and finite.");
            var ws = (double[])w.Clone();
            Array.Sort(s, ws);
            var cum = new double[ws.Length + 1];
            for (var i = 0; i < ws.Length; i++) cum[i + 1] = cum[i] + ws[i];
            if (cum[^1] <= 0) throw new ArgumentException("Weight array-like sums must be positive.");
            return (s, cum);
        }
        var (us, uc) = Prep(u, uw);
        var (vs, vc) = Prep(v, vw);
        var all = new double[u.Length + v.Length];
        u.CopyTo(all, 0);
        v.CopyTo(all, u.Length);
        Array.Sort(all);
        var acc = 0.0;
        for (var i = 0; i + 1 < all.Length; i++)
        {
            var delta = all[i + 1] - all[i];
            var iu = UpperBound(us, all[i]);
            var iv = UpperBound(vs, all[i]);
            var cu = uc is null ? (double)iu / us.Length : uc[iu] / uc[^1];
            var cv = vc is null ? (double)iv / vs.Length : vc[iv] / vc[^1];
            var diff = Math.Abs(cu - cv);
            acc += (p == 1 ? diff : diff * diff) * delta;
        }
        return p == 1 ? acc : Math.Sqrt(acc);
    }

    // ------------------------------------------------------------------ entropy

    /// <summary>
    /// Shannon entropy of the distribution <paramref name="pk"/> (normalised to sum 1), or the
    /// Kullback–Leibler divergence D(pk‖qk) when <paramref name="qk"/> is given, scipy's <c>entropy</c>.
    /// Nats unless a <paramref name="logBase"/> is given.
    /// </summary>
    public static double Entropy(double[] pk, double[]? qk = null, double? logBase = null)
    {
        ArgumentNullException.ThrowIfNull(pk);
        if (logBase is <= 0) throw new ArgumentException("base must be positive.", nameof(logBase));
        var sp = StatsTestMath.Sum(pk);
        var s = 0.0;
        if (qk is null)
        {
            var vec = new double[pk.Length];
            for (var i = 0; i < pk.Length; i++) vec[i] = Special.Entr(pk[i] / sp);
            s = StatsTestMath.Sum(vec);
        }
        else
        {
            if (qk.Length != pk.Length) throw new ArgumentException("qk and pk must have same shape.", nameof(qk));
            var sq = StatsTestMath.Sum(qk);
            var vec = new double[pk.Length];
            for (var i = 0; i < pk.Length; i++) vec[i] = Special.RelEntr(pk[i] / sp, qk[i] / sq);
            s = StatsTestMath.Sum(vec);
        }
        return logBase is { } b ? s / Math.Log(b) : s;
    }

    /// <summary><see cref="Entropy(double[], double[], double?)"/> of each lane along an axis.</summary>
    public static NdArray Entropy(NdArray pk, int axis = 0, double? logBase = null) => StatsAxis.Reduce(pk, axis, x => Entropy(x, null, logBase));

    /// <summary>
    /// Differential entropy estimated from a sample by m-spacings, scipy's <c>differential_entropy</c>.
    /// <paramref name="method"/>: <c>"vasicek"</c>, <c>"van es"</c>, <c>"ebrahimi"</c>, <c>"correa"</c> or
    /// <c>"auto"</c> (van Es for n ≤ 10, Ebrahimi for n ≤ 1000, Vasicek beyond). The window defaults to
    /// round(√n).
    /// </summary>
    public static double DifferentialEntropy(double[] values, int? windowLength = null, double? logBase = null, string method = "auto")
    {
        ArgumentNullException.ThrowIfNull(values);
        var n = values.Length;
        var m = windowLength ?? (int)Math.Floor(Math.Sqrt(n) + 0.5);
        if (!(2 <= 2 * m && 2 * m < n)) throw new ArgumentException($"Window length ({m}) must be positive and less than half the sample size ({n}).", nameof(windowLength));
        if (logBase is <= 0) throw new ArgumentException("`base` must be a positive number or `None`.", nameof(logBase));
        var x = StatsTestMath.Sorted(values);
        var meth = method.ToLowerInvariant();
        if (meth == "auto") meth = n <= 10 ? "van es" : n <= 1000 ? "ebrahimi" : "vasicek";
        double X(int i) => x[Math.Clamp(i, 0, n - 1)]; // padded with the end values
        double res;
        switch (meth)
        {
            case "vasicek":
            {
                var logs = new double[n];
                for (var i = 0; i < n; i++) logs[i] = Math.Log(n / (2.0 * m) * (X(i + m) - X(i - m)));
                res = StatsTestMath.Mean(logs);
                break;
            }
            case "van es":
            {
                var logs = new double[n - m];
                for (var i = 0; i < n - m; i++) logs[i] = Math.Log((n + 1.0) / m * (x[i + m] - x[i]));
                var t1 = 1.0 / (n - m) * StatsTestMath.Sum(logs);
                var h = new double[n - m + 1];
                for (var k = m; k <= n; k++) h[k - m] = 1.0 / k;
                res = t1 + StatsTestMath.Sum(h) + Math.Log(m) - Math.Log(n + 1.0);
                break;
            }
            case "ebrahimi":
            {
                var logs = new double[n];
                for (var i = 0; i < n; i++)
                {
                    var ii = i + 1.0;
                    var ci = ii <= m ? 1 + (ii - 1) / m : 2.0;
                    if (ii >= n - m + 1) ci = 1 + (n - ii) / m;
                    logs[i] = Math.Log(n * (X(i + m) - X(i - m)) / (ci * m));
                }
                res = StatsTestMath.Mean(logs);
                break;
            }
            case "correa":
            {
                var logs = new double[n];
                for (var i = 0; i < n; i++)
                {
                    var mean = 0.0;
                    for (var j = -m; j <= m; j++) mean += X(i + j);
                    mean /= 2 * m + 1;
                    double num = 0, den = 0;
                    for (var j = -m; j <= m; j++)
                    {
                        var dlt = X(i + j) - mean;
                        num += dlt * j;
                        den += dlt * dlt;
                    }
                    logs[i] = Math.Log(num / (n * den));
                }
                res = -StatsTestMath.Mean(logs);
                break;
            }
            default:
                throw new ArgumentException("method must be one of vasicek, van es, ebrahimi, correa, auto.", nameof(method));
        }
        return logBase is { } b ? res / Math.Log(b) : res;
    }
}
