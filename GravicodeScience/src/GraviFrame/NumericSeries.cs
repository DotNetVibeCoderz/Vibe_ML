using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviFrame;

/// <summary>
/// A column of double precision numbers, with <see cref="double.NaN"/> standing for "missing".
/// </summary>
/// <remarks>
/// The values live in a bare <c>double[]</c>, so every arithmetic operation can be handed to
/// GraviNum's vectorised <see cref="UFunc"/> kernels rather than looping over boxed objects.
/// Using NaN as the missing marker is what keeps that possible: no parallel null mask to check
/// inside the hot loop, and the IEEE rules propagate missing-ness for free.
/// </remarks>
public sealed class NumericSeries : Series
{
    private readonly double[] _values;

    /// <summary>Wraps <paramref name="values"/> without copying.</summary>
    public NumericSeries(string name, double[] values) : base(name, values.Length) => _values = values;

    /// <summary>Builds a column of <paramref name="length"/> missing values.</summary>
    public NumericSeries(string name, int length) : base(name, length)
    {
        _values = new double[length];
        Array.Fill(_values, double.NaN);
    }

    /// <inheritdoc />
    public override DataType DataType => DataType.Numeric;

    /// <summary>The raw buffer. Mutating it mutates the column.</summary>
    public double[] Values => _values;

    /// <summary>Reads or writes a value.</summary>
    public double this[int index]
    {
        get => _values[index];
        set => _values[index] = value;
    }

    /// <inheritdoc />
    public override object? GetValue(int index) => double.IsNaN(_values[index]) ? null : _values[index];

    /// <inheritdoc />
    public override bool IsMissing(int index) => double.IsNaN(_values[index]);

    /// <inheritdoc />
    public override Series Take(IReadOnlyList<int> indices) => TakeNumeric(indices);

    /// <summary>Reorders rows, keeping the numeric type.</summary>
    public NumericSeries TakeNumeric(IReadOnlyList<int> indices)
    {
        var result = new double[indices.Count];
        for (var i = 0; i < indices.Count; i++) result[i] = _values[indices[i]];
        return new NumericSeries(Name, result);
    }

    /// <inheritdoc />
    public override Series Rename(string name) => new NumericSeries(name, (double[])_values.Clone());

    /// <summary>A view of the column as a GraviNum array; no data is copied.</summary>
    public NdArray ToNdArray() => new(_values, _values.Length);

    /// <summary>The present values only, as a GraviNum array.</summary>
    public NdArray ToNdArrayDropMissing()
    {
        var present = _values.Where(v => !double.IsNaN(v)).ToArray();
        return new NdArray(present, present.Length);
    }

    /// <summary>Builds a column from a GraviNum array.</summary>
    public static NumericSeries FromNdArray(string name, NdArray array) => new(name, array.ToArray());

    // ---------------------------------------------------------------- arithmetic

    /// <summary>Element-wise sum of two columns.</summary>
    public static NumericSeries operator +(NumericSeries a, NumericSeries b) => a.Combine(b, "sum", static (x, y) => x + y);

    /// <summary>Element-wise difference of two columns.</summary>
    public static NumericSeries operator -(NumericSeries a, NumericSeries b) => a.Combine(b, "diff", static (x, y) => x - y);

    /// <summary>Element-wise product of two columns.</summary>
    public static NumericSeries operator *(NumericSeries a, NumericSeries b) => a.Combine(b, "product", static (x, y) => x * y);

    /// <summary>Element-wise quotient of two columns.</summary>
    public static NumericSeries operator /(NumericSeries a, NumericSeries b) => a.Combine(b, "ratio", static (x, y) => x / y);

    /// <summary>Adds a scalar to every value.</summary>
    public static NumericSeries operator +(NumericSeries a, double s) => a.Apply(x => x + s);

    /// <summary>Subtracts a scalar from every value.</summary>
    public static NumericSeries operator -(NumericSeries a, double s) => a.Apply(x => x - s);

    /// <summary>Scales every value.</summary>
    public static NumericSeries operator *(NumericSeries a, double s) => a.Apply(x => x * s);

    /// <summary>Divides every value by a scalar.</summary>
    public static NumericSeries operator /(NumericSeries a, double s) => a.Apply(x => x / s);

    private NumericSeries Combine(NumericSeries other, string suffix, Func<double, double, double> f)
    {
        if (other.Length != Length)
            throw new InvalidOperationException($"Cannot combine columns of length {Length} and {other.Length}.");
        var result = new double[Length];
        for (var i = 0; i < Length; i++) result[i] = f(_values[i], other._values[i]);
        return new NumericSeries($"{Name}_{suffix}", result);
    }

    /// <summary>Applies a function to every value, keeping the column name.</summary>
    public NumericSeries Apply(Func<double, double> f)
    {
        var result = new double[Length];
        for (var i = 0; i < Length; i++) result[i] = f(_values[i]);
        return new NumericSeries(Name, result);
    }

    // ---------------------------------------------------------------- statistics

    private double[] Present() => _values.Where(v => !double.IsNaN(v)).ToArray();

    /// <summary>Sum of the present values.</summary>
    public double Sum()
    {
        var acc = 0.0;
        foreach (var v in _values) if (!double.IsNaN(v)) acc += v;
        return acc;
    }

    /// <summary>Mean of the present values.</summary>
    public double Mean()
    {
        var acc = 0.0;
        var n = 0;
        foreach (var v in _values) if (!double.IsNaN(v)) { acc += v; n++; }
        return n == 0 ? double.NaN : acc / n;
    }

    /// <summary>Smallest present value.</summary>
    public double Min()
    {
        var m = double.PositiveInfinity;
        foreach (var v in _values) if (!double.IsNaN(v) && v < m) m = v;
        return double.IsPositiveInfinity(m) ? double.NaN : m;
    }

    /// <summary>Largest present value.</summary>
    public double Max()
    {
        var m = double.NegativeInfinity;
        foreach (var v in _values) if (!double.IsNaN(v) && v > m) m = v;
        return double.IsNegativeInfinity(m) ? double.NaN : m;
    }

    /// <summary>Product of the present values.</summary>
    public double Product()
    {
        var acc = 1.0;
        foreach (var v in _values) if (!double.IsNaN(v)) acc *= v;
        return acc;
    }

    /// <summary>Sample variance of the present values (ddof = 1, matching pandas).</summary>
    public double Var(int ddof = 1)
    {
        var present = Present();
        return present.Length <= ddof ? double.NaN : Statistics.Var(new NdArray(present, present.Length), ddof);
    }

    /// <summary>Sample standard deviation of the present values.</summary>
    public double Std(int ddof = 1) => Math.Sqrt(Var(ddof));

    /// <summary>Median of the present values.</summary>
    public double Median() => Quantile(0.5);

    /// <summary>Quantile of the present values, <paramref name="q"/> in <c>[0, 1]</c>.</summary>
    public double Quantile(double q)
    {
        var present = Present();
        if (present.Length == 0) return double.NaN;
        Array.Sort(present);
        return Statistics.PercentileOfSorted(present, q * 100.0);
    }

    /// <summary>Skewness of the present values.</summary>
    public double Skewness()
    {
        var present = Present();
        return present.Length < 3 ? double.NaN : Statistics.Skewness(new NdArray(present, present.Length));
    }

    /// <summary>Excess kurtosis of the present values.</summary>
    public double Kurtosis()
    {
        var present = Present();
        return present.Length < 4 ? double.NaN : Statistics.Kurtosis(new NdArray(present, present.Length));
    }

    /// <summary>Pearson correlation with another column, over rows present in both.</summary>
    public double Correlation(NumericSeries other)
    {
        var x = new List<double>();
        var y = new List<double>();
        for (var i = 0; i < Length; i++)
        {
            if (double.IsNaN(_values[i]) || double.IsNaN(other._values[i])) continue;
            x.Add(_values[i]);
            y.Add(other._values[i]);
        }
        if (x.Count < 2) return double.NaN;
        return Statistics.Correlation(NdArray.FromValues(x), NdArray.FromValues(y));
    }

    /// <summary>The pandas-style descriptive summary.</summary>
    public IReadOnlyDictionary<string, double> Describe() => new Dictionary<string, double>
    {
        ["count"] = Count,
        ["mean"] = Mean(),
        ["std"] = Std(),
        ["min"] = Min(),
        ["25%"] = Quantile(0.25),
        ["50%"] = Quantile(0.50),
        ["75%"] = Quantile(0.75),
        ["max"] = Max(),
    };

    // ---------------------------------------------------------------- missing data

    /// <summary>Replaces missing values with a constant.</summary>
    public NumericSeries FillMissing(double value)
        => new(Name, _values.Select(v => double.IsNaN(v) ? value : v).ToArray());

    /// <summary>Replaces missing values with the column mean.</summary>
    public NumericSeries FillMissingWithMean() => FillMissing(Mean());

    /// <summary>Replaces missing values with the column median.</summary>
    public NumericSeries FillMissingWithMedian() => FillMissing(Median());

    /// <summary>Carries the last present value forward over gaps.</summary>
    public NumericSeries ForwardFill()
    {
        var result = new double[Length];
        var last = double.NaN;
        for (var i = 0; i < Length; i++)
        {
            if (!double.IsNaN(_values[i])) last = _values[i];
            result[i] = last;
        }
        return new NumericSeries(Name, result);
    }

    /// <summary>Carries the next present value backward over gaps.</summary>
    public NumericSeries BackwardFill()
    {
        var result = new double[Length];
        var next = double.NaN;
        for (var i = Length - 1; i >= 0; i--)
        {
            if (!double.IsNaN(_values[i])) next = _values[i];
            result[i] = next;
        }
        return new NumericSeries(Name, result);
    }

    /// <summary>Fills gaps by linear interpolation between the surrounding present values.</summary>
    public NumericSeries Interpolate()
    {
        var result = (double[])_values.Clone();
        var i = 0;
        while (i < Length)
        {
            if (!double.IsNaN(result[i])) { i++; continue; }

            var start = i - 1;
            var end = i;
            while (end < Length && double.IsNaN(result[end])) end++;

            if (start >= 0 && end < Length)
            {
                var span = end - start;
                for (var k = start + 1; k < end; k++)
                    result[k] = result[start] + (result[end] - result[start]) * (k - start) / span;
            }
            i = end + 1;
        }
        return new NumericSeries(Name, result);
    }

    /// <summary>Drops missing values, shortening the column.</summary>
    public NumericSeries DropMissing() => new(Name, Present());

    // ---------------------------------------------------------------- transforms

    /// <summary>Z-scores over the present values.</summary>
    public NumericSeries Standardize()
    {
        var mean = Mean();
        var sd = Std();
        return sd == 0 || double.IsNaN(sd) ? Apply(_ => 0.0) : Apply(x => (x - mean) / sd);
    }

    /// <summary>Rescales the present values into <c>[0, 1]</c>.</summary>
    public NumericSeries MinMaxScale()
    {
        var min = Min();
        var range = Max() - min;
        return range == 0 ? Apply(_ => 0.0) : Apply(x => (x - min) / range);
    }

    /// <summary>Clamps values into <c>[min, max]</c>.</summary>
    public NumericSeries Clip(double min, double max) => Apply(x => double.IsNaN(x) ? x : Math.Clamp(x, min, max));

    /// <summary>Absolute values.</summary>
    public NumericSeries Abs() => Apply(Math.Abs);

    /// <summary>Running total, skipping missing values.</summary>
    public NumericSeries CumulativeSum()
    {
        var result = new double[Length];
        var acc = 0.0;
        for (var i = 0; i < Length; i++)
        {
            if (!double.IsNaN(_values[i])) acc += _values[i];
            result[i] = acc;
        }
        return new NumericSeries($"{Name}_cumsum", result);
    }

    /// <summary>Running maximum.</summary>
    public NumericSeries CumulativeMax()
    {
        var result = new double[Length];
        var best = double.NegativeInfinity;
        for (var i = 0; i < Length; i++)
        {
            if (!double.IsNaN(_values[i])) best = Math.Max(best, _values[i]);
            result[i] = double.IsNegativeInfinity(best) ? double.NaN : best;
        }
        return new NumericSeries($"{Name}_cummax", result);
    }

    /// <summary>Competition ranks (1 = smallest); ties share the average rank.</summary>
    public NumericSeries Rank()
    {
        var order = Enumerable.Range(0, Length)
            .Where(i => !double.IsNaN(_values[i]))
            .OrderBy(i => _values[i])
            .ToArray();

        var result = new double[Length];
        Array.Fill(result, double.NaN);

        var k = 0;
        while (k < order.Length)
        {
            var j = k;
            while (j + 1 < order.Length && _values[order[j + 1]] == _values[order[k]]) j++;
            var average = (k + j) / 2.0 + 1;
            for (var t = k; t <= j; t++) result[order[t]] = average;
            k = j + 1;
        }
        return new NumericSeries($"{Name}_rank", result);
    }

    /// <summary>A boolean mask of the rows satisfying <paramref name="predicate"/>.</summary>
    public bool[] Where(Func<double, bool> predicate)
    {
        var mask = new bool[Length];
        for (var i = 0; i < Length; i++) mask[i] = !double.IsNaN(_values[i]) && predicate(_values[i]);
        return mask;
    }

    /// <summary>Buckets values into <paramref name="bins"/> equal-width intervals, as bin indices.</summary>
    public NumericSeries Bin(int bins)
    {
        var min = Min();
        var max = Max();
        var range = max - min;
        if (range == 0) return Apply(_ => 0.0);
        return Apply(x => double.IsNaN(x) ? x : Math.Min(bins - 1, Math.Floor((x - min) / range * bins)));
    }
}
