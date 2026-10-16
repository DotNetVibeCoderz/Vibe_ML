using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Plumbing shared by the hypothesis tests and descriptive statistics: validation, NaN policy,
/// sums, ranking with tie bookkeeping and p-values from the reference distributions.
/// </summary>
internal static class StatsTestMath
{
    /// <summary>Validates and applies the NaN policy; <paramref name="hasNan"/> tells a Propagate caller to return NaN.</summary>
    public static double[] Prepare(double[] x, string name, NanPolicy policy, out bool hasNan)
    {
        ArgumentNullException.ThrowIfNull(x, name);
        hasNan = false;
        foreach (var v in x)
            if (double.IsNaN(v)) { hasNan = true; break; }
        if (!hasNan) return x;
        switch (policy)
        {
            case NanPolicy.Raise:
                throw new ArgumentException("The input contains NaN.", name);
            case NanPolicy.Omit:
                hasNan = false;
                return x.Where(v => !double.IsNaN(v)).ToArray();
            default:
                return x;
        }
    }

    /// <summary>Drops pairs where either side is NaN (Omit), or flags them.</summary>
    public static (double[] X, double[] Y) PreparePaired(double[] x, double[] y, NanPolicy policy, out bool hasNan)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (x.Length != y.Length) throw new ArgumentException("x and y must have the same length.");
        hasNan = false;
        for (var i = 0; i < x.Length; i++)
            if (double.IsNaN(x[i]) || double.IsNaN(y[i])) { hasNan = true; break; }
        if (!hasNan) return (x, y);
        if (policy == NanPolicy.Raise) throw new ArgumentException("The input contains NaN.");
        if (policy == NanPolicy.Propagate) return (x, y);
        hasNan = false;
        var xs = new List<double>();
        var ys = new List<double>();
        for (var i = 0; i < x.Length; i++)
            if (!double.IsNaN(x[i]) && !double.IsNaN(y[i])) { xs.Add(x[i]); ys.Add(y[i]); }
        return (xs.ToArray(), ys.ToArray());
    }

    /// <summary>Pairwise (cascade) summation: the error grows as O(log n) rather than O(n), as numpy's sum.</summary>
    public static double Sum(ReadOnlySpan<double> x)
    {
        if (x.Length <= 128)
        {
            // numpy unrolls by 8 below its block size; matching that keeps the last digits identical.
            if (x.Length < 8)
            {
                var s = 0.0;
                foreach (var v in x) s += v;
                return s;
            }
            double r0 = x[0], r1 = x[1], r2 = x[2], r3 = x[3], r4 = x[4], r5 = x[5], r6 = x[6], r7 = x[7];
            var i = 8;
            for (; i + 8 <= x.Length; i += 8)
            {
                r0 += x[i]; r1 += x[i + 1]; r2 += x[i + 2]; r3 += x[i + 3];
                r4 += x[i + 4]; r5 += x[i + 5]; r6 += x[i + 6]; r7 += x[i + 7];
            }
            var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
            for (; i < x.Length; i++) res += x[i];
            return res;
        }
        var half = x.Length / 2;
        half -= half % 8;
        return Sum(x[..half]) + Sum(x[half..]);
    }

    /// <summary>Arithmetic mean by pairwise summation.</summary>
    public static double Mean(ReadOnlySpan<double> x) => Sum(x) / x.Length;

    /// <summary>Variance with <paramref name="ddof"/> delta degrees of freedom, two-pass.</summary>
    public static double Var(ReadOnlySpan<double> x, int ddof = 0)
    {
        var n = x.Length;
        if (n - ddof <= 0) return double.NaN;
        var m = Mean(x);
        var s = 0.0;
        var c = 0.0;
        foreach (var v in x)
        {
            var d = v - m;
            s += d * d;
            c += d;
        }
        // The compensation term removes the rounding of the mean (Chan, Golub & LeVeque 1983).
        return (s - c * c / n) / (n - ddof);
    }

    /// <summary>A sorted copy.</summary>
    public static double[] Sorted(ReadOnlySpan<double> x)
    {
        var s = x.ToArray();
        Array.Sort(s);
        return s;
    }

    /// <summary>Median of a sample (copying, not mutating).</summary>
    public static double Median(ReadOnlySpan<double> x)
    {
        if (x.Length == 0) return double.NaN;
        var s = Sorted(x);
        var n = s.Length;
        return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
    }

    /// <summary>numpy's default ('linear') percentile of already sorted data; <paramref name="q"/> in [0, 100].</summary>
    public static double PercentileSorted(ReadOnlySpan<double> sorted, double q)
    {
        var n = sorted.Length;
        if (n == 0) return double.NaN;
        var h = (n - 1) * (q / 100.0);
        var lo = (int)Math.Floor(h);
        if (lo >= n - 1) return sorted[n - 1];
        if (lo < 0) return sorted[0];
        var g = h - lo;
        var a = sorted[lo];
        var b = sorted[lo + 1];
        // numpy's _lerp: switch formula at g ≥ 0.5 so that g = 1 returns b exactly.
        var diff = b - a;
        return g >= 0.5 ? b - diff * (1 - g) : a + diff * g;
    }

    /// <summary>
    /// Average ranks (1-based, ties share the mean rank) and Σ(t³ − t) over tie groups, the term every
    /// rank test's tie correction needs.
    /// </summary>
    public static double[] RankAverage(ReadOnlySpan<double> x, out double tieSum)
    {
        var n = x.Length;
        var keys = x.ToArray();
        var idx = new int[n];
        for (var i = 0; i < n; i++) idx[i] = i;
        Array.Sort(keys, idx);
        var ranks = new double[n];
        tieSum = 0;
        var start = 0;
        while (start < n)
        {
            var end = start + 1;
            while (end < n && keys[end] == keys[start]) end++;
            var t = end - start;
            var r = (start + end + 1) / 2.0;
            for (var k = start; k < end; k++) ranks[idx[k]] = r;
            if (t > 1) tieSum += (double)t * t * t - t;
            start = end;
        }
        return ranks;
    }

    /// <summary>Sizes of the runs of equal values in sorted data.</summary>
    public static List<int> TieCounts(ReadOnlySpan<double> sorted)
    {
        var counts = new List<int>();
        var start = 0;
        while (start < sorted.Length)
        {
            var end = start + 1;
            while (end < sorted.Length && sorted[end] == sorted[start]) end++;
            counts.Add(end - start);
            start = end;
        }
        return counts;
    }

    /// <summary>Concatenation of several samples.</summary>
    public static double[] Concat(IReadOnlyList<double[]> samples)
    {
        var total = 0;
        foreach (var s in samples) total += s.Length;
        var all = new double[total];
        var o = 0;
        foreach (var s in samples) { s.CopyTo(all, o); o += s.Length; }
        return all;
    }

    /// <summary>Validates a list of at least <paramref name="min"/> samples, each non-null.</summary>
    public static void CheckSamples(double[][] samples, int min, string name = "samples")
    {
        ArgumentNullException.ThrowIfNull(samples, name);
        if (samples.Length < min) throw new ArgumentException($"At least {min} samples are required.", name);
        foreach (var s in samples) ArgumentNullException.ThrowIfNull(s, name);
    }

    // ------------------------------------------------------------------ p-values

    /// <summary>scipy's <c>_get_pvalue</c> for the standard normal (symmetric).</summary>
    public static double NormPvalue(double z, Alternative alt) => alt switch
    {
        Alternative.Less => Special.Ndtr(z),
        Alternative.Greater => Special.Ndtr(-z),
        _ => double.IsNaN(z) ? double.NaN : 2 * Special.Ndtr(-Math.Abs(z)),
    };

    /// <summary>scipy's <c>_get_pvalue</c> for Student's t (symmetric).</summary>
    public static double TPvalue(double t, double df, Alternative alt) => alt switch
    {
        Alternative.Less => Special.Stdtr(df, t),
        Alternative.Greater => Special.Stdtr(df, -t),
        _ => double.IsNaN(t) ? double.NaN : 2 * Special.Stdtr(df, -Math.Abs(t)),
    };

    /// <summary>Upper tail of chi-square with <paramref name="df"/> degrees of freedom.</summary>
    public static double Chi2Sf(double x, double df) => double.IsNaN(x) ? double.NaN : Special.Chdtrc(df, Math.Max(x, 0));

    /// <summary>Upper tail of F(dfn, dfd).</summary>
    public static double FSf(double x, double dfn, double dfd) => double.IsNaN(x) ? double.NaN : x <= 0 ? 1.0 : Special.Fdtrc(dfn, dfd, x);

    /// <summary>Two-sided p-value of a discrete null as 2·min(cdf, sf) clipped to 1.</summary>
    public static double Clip01(double p) => double.IsNaN(p) ? p : Math.Clamp(p, 0.0, 1.0);

    /// <summary>Copies a 2-D NdArray to row-major (rows, cols, data).</summary>
    public static (int Rows, int Cols, double[] Data) Matrix(NdArray a, string name)
    {
        ArgumentNullException.ThrowIfNull(a, name);
        if (a.Rank != 2) throw new ArgumentException($"{name} must be 2-D.", name);
        return (a.Shape[0], a.Shape[1], a.AsContiguous().ToArray());
    }
}

/// <summary>
/// Applies a 1-D statistic along one axis of an <see cref="NdArray"/> — the plumbing behind the
/// <c>axis=</c> overloads. Each lane is copied once into a contiguous buffer; lanes run in parallel
/// when there is enough work to pay for the hand-off.
/// </summary>
internal static class StatsAxis
{
    private const int ParallelWork = 1 << 15;

    /// <summary>Splits the shape around <paramref name="axis"/>: outer × length × inner.</summary>
    public static (int Outer, int Length, int Inner, int[] Reduced) Split(NdArray a, int axis)
    {
        var rank = a.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentOutOfRangeException(nameof(axis), $"axis is out of range for a {rank}-D array.");
        var shape = a.Shape;
        int outer = 1, inner = 1;
        for (var i = 0; i < axis; i++) outer *= shape[i];
        for (var i = axis + 1; i < rank; i++) inner *= shape[i];
        var reduced = new int[rank - 1];
        for (int i = 0, k = 0; i < rank; i++) if (i != axis) reduced[k++] = shape[i];
        return (outer, shape[axis], inner, reduced);
    }

    /// <summary>Reduces each lane along <paramref name="axis"/> to a scalar.</summary>
    public static NdArray Reduce(NdArray a, int axis, Func<double[], double> f)
    {
        ArgumentNullException.ThrowIfNull(a);
        var (outer, len, inner, reduced) = Split(a, axis);
        var src = a.AsContiguous().ToArray();
        var lanes = outer * inner;
        var result = new double[lanes];
        void Lane(int l, double[] buf)
        {
            int o = l / inner, i = l % inner;
            var b = o * len * inner + i;
            for (var k = 0; k < len; k++) buf[k] = src[b + k * inner];
            result[l] = f(buf);
        }
        if ((long)lanes * len >= ParallelWork && lanes > 1)
            Parallel.For(0, lanes, () => new double[len], (l, _, buf) => { Lane(l, buf); return buf; }, _ => { });
        else
        {
            var buf = new double[len];
            for (var l = 0; l < lanes; l++) Lane(l, buf);
        }
        return reduced.Length == 0 ? new NdArray(result, 1) : new NdArray(result, reduced);
    }

    /// <summary>Transforms each lane along <paramref name="axis"/>, keeping the shape.</summary>
    public static NdArray Transform(NdArray a, int axis, Func<double[], double[]> f)
    {
        ArgumentNullException.ThrowIfNull(a);
        var (outer, len, inner, _) = Split(a, axis);
        var src = a.AsContiguous().ToArray();
        var dst = new double[src.Length];
        var lanes = outer * inner;
        void Lane(int l)
        {
            int o = l / inner, i = l % inner;
            var b = o * len * inner + i;
            var buf = new double[len];
            for (var k = 0; k < len; k++) buf[k] = src[b + k * inner];
            var r = f(buf);
            for (var k = 0; k < len; k++) dst[b + k * inner] = r[k];
        }
        if ((long)lanes * len >= ParallelWork && lanes > 1) Parallel.For(0, lanes, Lane);
        else for (var l = 0; l < lanes; l++) Lane(l);
        return new NdArray(dst, a.Shape.ToArray());
    }
}
