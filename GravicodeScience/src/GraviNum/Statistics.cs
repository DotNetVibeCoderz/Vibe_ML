namespace Gravicode.Science.GraviNum;

/// <summary>
/// Descriptive statistics and reductions over <see cref="NdArray"/>, whole-array or along one axis.
/// </summary>
public static class Statistics
{
    // ---------------------------------------------------------------- whole array

    /// <summary>Sum of every element, using pairwise summation to limit rounding drift.</summary>
    public static double Sum(NdArray a)
    {
        if (a.Size == 0) return 0.0;
        if (a.IsContiguous) return PairwiseSum(a.AsSpan());

        // Kahan compensation for the strided path, where we cannot slice cheaply.
        var sum = 0.0;
        var comp = 0.0;
        for (var i = 0; i < a.Size; i++)
        {
            var y = a.At(i) - comp;
            var t = sum + y;
            comp = t - sum - y;
            sum = t;
        }
        return sum;
    }

    private static double PairwiseSum(ReadOnlySpan<double> values)
    {
        const int blockSize = 128;
        if (values.Length <= blockSize)
        {
            var acc = 0.0;
            foreach (var v in values) acc += v;
            return acc;
        }
        var half = values.Length / 2;
        return PairwiseSum(values[..half]) + PairwiseSum(values[half..]);
    }

    /// <summary>Product of every element.</summary>
    public static double Product(NdArray a)
    {
        var p = 1.0;
        for (var i = 0; i < a.Size; i++) p *= a.At(i);
        return p;
    }

    /// <summary>Arithmetic mean.</summary>
    public static double Mean(NdArray a) => a.Size == 0 ? double.NaN : Sum(a) / a.Size;

    /// <summary>Smallest element.</summary>
    public static double Min(NdArray a)
    {
        if (a.Size == 0) return double.NaN;
        var m = double.PositiveInfinity;
        for (var i = 0; i < a.Size; i++) { var v = a.At(i); if (v < m) m = v; }
        return m;
    }

    /// <summary>Largest element.</summary>
    public static double Max(NdArray a)
    {
        if (a.Size == 0) return double.NaN;
        var m = double.NegativeInfinity;
        for (var i = 0; i < a.Size; i++) { var v = a.At(i); if (v > m) m = v; }
        return m;
    }

    /// <summary>Flat index of the smallest element.</summary>
    public static int ArgMin(NdArray a)
    {
        var best = 0;
        var m = double.PositiveInfinity;
        for (var i = 0; i < a.Size; i++) { var v = a.At(i); if (v < m) { m = v; best = i; } }
        return best;
    }

    /// <summary>Flat index of the largest element.</summary>
    public static int ArgMax(NdArray a)
    {
        var best = 0;
        var m = double.NegativeInfinity;
        for (var i = 0; i < a.Size; i++) { var v = a.At(i); if (v > m) { m = v; best = i; } }
        return best;
    }

    /// <summary>
    /// Variance. <paramref name="ddof"/> is the delta degrees of freedom: 0 for the population
    /// variance (NumPy's default), 1 for the unbiased sample variance (pandas' default).
    /// </summary>
    public static double Var(NdArray a, int ddof = 0)
    {
        var n = a.Size;
        if (n - ddof <= 0) return double.NaN;

        // Welford's online algorithm: one pass, no catastrophic cancellation.
        var mean = 0.0;
        var m2 = 0.0;
        for (var i = 0; i < n; i++)
        {
            var x = a.At(i);
            var delta = x - mean;
            mean += delta / (i + 1);
            m2 += delta * (x - mean);
        }
        return m2 / (n - ddof);
    }

    /// <summary>Standard deviation; see <see cref="Var"/> for <paramref name="ddof"/>.</summary>
    public static double Std(NdArray a, int ddof = 0) => Math.Sqrt(Var(a, ddof));

    /// <summary>Median value.</summary>
    public static double Median(NdArray a) => Percentile(a, 50);

    /// <summary>
    /// Linear-interpolated percentile, <paramref name="percent"/> in <c>[0, 100]</c>.
    /// </summary>
    public static double Percentile(NdArray a, double percent)
    {
        if (a.Size == 0) return double.NaN;
        var sorted = a.ToArray();
        Array.Sort(sorted);
        return PercentileOfSorted(sorted, percent);
    }

    /// <summary>Percentile of an already-sorted buffer.</summary>
    public static double PercentileOfSorted(double[] sorted, double percent)
    {
        if (sorted.Length == 0) return double.NaN;
        if (sorted.Length == 1) return sorted[0];
        var rank = Math.Clamp(percent, 0, 100) / 100.0 * (sorted.Length - 1);
        var lo = (int)Math.Floor(rank);
        var hi = (int)Math.Ceiling(rank);
        if (lo == hi) return sorted[lo];
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
    }

    /// <summary>Quantile, <paramref name="q"/> in <c>[0, 1]</c>.</summary>
    public static double Quantile(NdArray a, double q) => Percentile(a, q * 100.0);

    /// <summary>Difference between the 75th and 25th percentiles.</summary>
    public static double InterQuartileRange(NdArray a) => Percentile(a, 75) - Percentile(a, 25);

    /// <summary>Most frequently occurring value (first one wins on ties).</summary>
    public static double Mode(NdArray a)
    {
        var counts = new Dictionary<double, int>();
        var best = double.NaN;
        var bestCount = 0;
        for (var i = 0; i < a.Size; i++)
        {
            var v = a.At(i);
            counts.TryGetValue(v, out var c);
            counts[v] = ++c;
            if (c > bestCount) { bestCount = c; best = v; }
        }
        return best;
    }

    /// <summary>Fisher-Pearson skewness.</summary>
    public static double Skewness(NdArray a)
    {
        var n = a.Size;
        if (n < 3) return double.NaN;
        var mean = Mean(a);
        var sd = Std(a);
        if (sd == 0) return 0.0;
        var sum = 0.0;
        for (var i = 0; i < n; i++) sum += Math.Pow((a.At(i) - mean) / sd, 3);
        return sum / n;
    }

    /// <summary>Excess kurtosis (normal distribution scores 0).</summary>
    public static double Kurtosis(NdArray a)
    {
        var n = a.Size;
        if (n < 4) return double.NaN;
        var mean = Mean(a);
        var sd = Std(a);
        if (sd == 0) return 0.0;
        var sum = 0.0;
        for (var i = 0; i < n; i++) sum += Math.Pow((a.At(i) - mean) / sd, 4);
        return sum / n - 3.0;
    }

    // ---------------------------------------------------------------- axis reductions

    /// <summary>
    /// Reduces along a single axis. The axis is removed from the result shape; reducing a
    /// rank-1 array therefore yields a single-element array.
    /// </summary>
    public static NdArray Reduce(NdArray a, int axis, Func<double[], double> reducer)
    {
        var ax = Shapes.NormalizeAxis(axis, a.Rank);
        var length = a.Shape[ax];

        var outShape = a.Rank == 1
            ? new[] { 1 }
            : Enumerable.Range(0, a.Rank).Where(d => d != ax).Select(d => a.Shape[d]).ToArray();

        var result = NdArray.Zeros(outShape);
        var outSize = result.Size;
        var buffer = new int[a.Rank];
        var slice = new double[length];

        for (var o = 0; o < outSize; o++)
        {
            // Rebuild the source index from the flat output index, leaving the reduced axis open.
            var rem = o;
            for (var d = a.Rank - 1; d >= 0; d--)
            {
                if (d == ax) { buffer[d] = 0; continue; }
                buffer[d] = rem % a.Shape[d];
                rem /= a.Shape[d];
            }

            for (var k = 0; k < length; k++)
            {
                buffer[ax] = k;
                slice[k] = a[buffer];
            }
            result.SetAt(o, reducer(slice));
        }
        return result;
    }

    /// <summary>Sum along an axis.</summary>
    public static NdArray Sum(NdArray a, int axis) => Reduce(a, axis, static s => s.Sum());

    /// <summary>Mean along an axis.</summary>
    public static NdArray Mean(NdArray a, int axis) => Reduce(a, axis, static s => s.Average());

    /// <summary>Minimum along an axis.</summary>
    public static NdArray Min(NdArray a, int axis) => Reduce(a, axis, static s => s.Min());

    /// <summary>Maximum along an axis.</summary>
    public static NdArray Max(NdArray a, int axis) => Reduce(a, axis, static s => s.Max());

    /// <summary>Standard deviation along an axis.</summary>
    public static NdArray Std(NdArray a, int axis, int ddof = 0)
        => Reduce(a, axis, s => Std(new NdArray(s, s.Length), ddof));

    /// <summary>Variance along an axis.</summary>
    public static NdArray Var(NdArray a, int axis, int ddof = 0)
        => Reduce(a, axis, s => Var(new NdArray(s, s.Length), ddof));

    /// <summary>Index of the maximum along an axis.</summary>
    public static NdArray ArgMax(NdArray a, int axis)
        => Reduce(a, axis, static s => Array.IndexOf(s, s.Max()));

    /// <summary>Index of the minimum along an axis.</summary>
    public static NdArray ArgMin(NdArray a, int axis)
        => Reduce(a, axis, static s => Array.IndexOf(s, s.Min()));

    /// <summary>Median along an axis.</summary>
    public static NdArray Median(NdArray a, int axis)
        => Reduce(a, axis, static s => { var c = (double[])s.Clone(); Array.Sort(c); return PercentileOfSorted(c, 50); });

    // ---------------------------------------------------------------- cumulative

    /// <summary>Running total over the flattened array.</summary>
    public static NdArray CumulativeSum(NdArray a)
    {
        var result = NdArray.Zeros(a.Size);
        var acc = 0.0;
        for (var i = 0; i < a.Size; i++) { acc += a.At(i); result.SetAt(i, acc); }
        return result;
    }

    /// <summary>Running product over the flattened array.</summary>
    public static NdArray CumulativeProduct(NdArray a)
    {
        var result = NdArray.Zeros(a.Size);
        var acc = 1.0;
        for (var i = 0; i < a.Size; i++) { acc *= a.At(i); result.SetAt(i, acc); }
        return result;
    }

    /// <summary>First-order differences over the flattened array.</summary>
    public static NdArray Diff(NdArray a)
    {
        if (a.Size < 2) return NdArray.Zeros(0);
        var result = NdArray.Zeros(a.Size - 1);
        for (var i = 1; i < a.Size; i++) result.SetAt(i - 1, a.At(i) - a.At(i - 1));
        return result;
    }

    // ---------------------------------------------------------------- relationships

    /// <summary>Covariance between two equally sized 1-D arrays.</summary>
    public static double Covariance(NdArray x, NdArray y, int ddof = 1)
    {
        if (x.Size != y.Size) throw new ArgumentException("Covariance needs two equally sized arrays.");
        var n = x.Size;
        if (n - ddof <= 0) return double.NaN;
        var mx = Mean(x);
        var my = Mean(y);
        var acc = 0.0;
        for (var i = 0; i < n; i++) acc += (x.At(i) - mx) * (y.At(i) - my);
        return acc / (n - ddof);
    }

    /// <summary>Pearson correlation coefficient between two 1-D arrays.</summary>
    public static double Correlation(NdArray x, NdArray y)
    {
        var sx = Std(x);
        var sy = Std(y);
        if (sx == 0 || sy == 0) return double.NaN;
        return Covariance(x, y, ddof: 0) / (sx * sy);
    }

    /// <summary>Spearman rank correlation between two 1-D arrays.</summary>
    public static double SpearmanCorrelation(NdArray x, NdArray y)
        => Correlation(new NdArray(Ranks(x), x.Size), new NdArray(Ranks(y), y.Size));

    private static double[] Ranks(NdArray a)
    {
        var n = a.Size;
        var order = Enumerable.Range(0, n).OrderBy(i => a.At(i)).ToArray();
        var ranks = new double[n];
        var i2 = 0;
        while (i2 < n)
        {
            // Ties share the average of the ranks they span.
            var j = i2;
            while (j + 1 < n && a.At(order[j + 1]) == a.At(order[i2])) j++;
            var avg = (i2 + j) / 2.0 + 1;
            for (var k = i2; k <= j; k++) ranks[order[k]] = avg;
            i2 = j + 1;
        }
        return ranks;
    }

    /// <summary>
    /// Covariance matrix of a samples-by-features matrix; the result is features-by-features.
    /// </summary>
    public static NdArray CovarianceMatrix(NdArray data, int ddof = 1)
    {
        if (data.Rank != 2) throw new ArgumentException("CovarianceMatrix expects a rank 2 array.");
        var features = data.Shape[1];
        var result = NdArray.Zeros(features, features);
        var columns = new NdArray[features];
        for (var j = 0; j < features; j++) columns[j] = data.Column(j).Copy();

        for (var i = 0; i < features; i++)
            for (var j = i; j < features; j++)
            {
                var c = Covariance(columns[i], columns[j], ddof);
                result[i, j] = c;
                result[j, i] = c;
            }
        return result;
    }

    /// <summary>Correlation matrix of a samples-by-features matrix.</summary>
    public static NdArray CorrelationMatrix(NdArray data)
    {
        if (data.Rank != 2) throw new ArgumentException("CorrelationMatrix expects a rank 2 array.");
        var features = data.Shape[1];
        var result = NdArray.Zeros(features, features);
        var columns = new NdArray[features];
        for (var j = 0; j < features; j++) columns[j] = data.Column(j).Copy();

        for (var i = 0; i < features; i++)
            for (var j = i; j < features; j++)
            {
                var c = i == j ? 1.0 : Correlation(columns[i], columns[j]);
                result[i, j] = c;
                result[j, i] = c;
            }
        return result;
    }

    /// <summary>Counts values into <paramref name="bins"/> equal-width buckets.</summary>
    public static (double[] Edges, int[] Counts) Histogram(NdArray a, int bins = 10)
    {
        var min = Min(a);
        var max = Max(a);
        if (min == max) { max = min + 1; }
        var edges = new double[bins + 1];
        for (var i = 0; i <= bins; i++) edges[i] = min + (max - min) * i / bins;

        var counts = new int[bins];
        for (var i = 0; i < a.Size; i++)
        {
            var idx = (int)((a.At(i) - min) / (max - min) * bins);
            counts[Math.Clamp(idx, 0, bins - 1)]++;
        }
        return (edges, counts);
    }

    /// <summary>Z-scores: <c>(x - mean) / std</c> over the whole array.</summary>
    public static NdArray Standardize(NdArray a)
    {
        var mean = Mean(a);
        var sd = Std(a);
        return sd == 0 ? a.Map(_ => 0.0) : a.Map(x => (x - mean) / sd);
    }

    /// <summary>Rescales the whole array into <c>[0, 1]</c>.</summary>
    public static NdArray MinMaxScale(NdArray a)
    {
        var min = Min(a);
        var max = Max(a);
        var range = max - min;
        return range == 0 ? a.Map(_ => 0.0) : a.Map(x => (x - min) / range);
    }

    /// <summary>A quick five-number-style summary, handy in notebooks.</summary>
    public static IReadOnlyDictionary<string, double> Describe(NdArray a) => new Dictionary<string, double>
    {
        ["count"] = a.Size,
        ["mean"] = Mean(a),
        ["std"] = Std(a, ddof: 1),
        ["min"] = Min(a),
        ["25%"] = Percentile(a, 25),
        ["50%"] = Percentile(a, 50),
        ["75%"] = Percentile(a, 75),
        ["max"] = Max(a),
    };
}
