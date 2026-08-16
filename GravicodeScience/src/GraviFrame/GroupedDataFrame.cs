namespace Gravicode.Science.GraviFrame;

/// <summary>
/// The result of <see cref="DataFrame.GroupBy"/>: rows bucketed by the distinct values of one or
/// more key columns, ready to be aggregated.
/// </summary>
/// <remarks>
/// Grouping is done once, eagerly, into an index list per key - so <c>Mean</c>, <c>Sum</c> and
/// <c>Count</c> on the same <c>GroupBy</c> each cost one pass over the values rather than
/// re-bucketing the frame. Groups keep first-seen order, which makes results reproducible.
/// </remarks>
public sealed class GroupedDataFrame
{
    private readonly DataFrame _source;
    private readonly string[] _keys;
    private readonly List<(object?[] Key, List<int> Rows)> _groups;

    internal GroupedDataFrame(DataFrame source, string[] keys)
    {
        if (keys.Length == 0) throw new ArgumentException("GroupBy needs at least one key column.", nameof(keys));
        _source = source;
        _keys = keys;

        var keyColumns = keys.Select(k => source[k]).ToArray();
        var lookup = new Dictionary<string, int>(StringComparer.Ordinal);
        _groups = [];

        for (var row = 0; row < source.RowCount; row++)
        {
            var values = new object?[keys.Length];
            for (var k = 0; k < keys.Length; k++) values[k] = keyColumns[k].GetValue(row);

            // Unit separator plus a null sentinel keep composite keys unambiguous.
            var signature = string.Join('\u001F', values.Select(v => v?.ToString() ?? "\u0000"));
            if (!lookup.TryGetValue(signature, out var index))
            {
                index = _groups.Count;
                lookup[signature] = index;
                _groups.Add((values, []));
            }
            _groups[index].Rows.Add(row);
        }
    }

    /// <summary>Number of distinct groups.</summary>
    public int GroupCount => _groups.Count;

    /// <summary>The key columns this frame was grouped by.</summary>
    public IReadOnlyList<string> Keys => _keys;

    /// <summary>The groups, as key values plus the rows that carry them.</summary>
    public IEnumerable<(object?[] Key, IReadOnlyList<int> Rows)> Groups
        => _groups.Select(g => (g.Key, (IReadOnlyList<int>)g.Rows));

    /// <summary>Materialises one group as its own frame.</summary>
    public DataFrame GetGroup(int index) => _source.Take(_groups[index].Rows);

    /// <summary>Row count per group.</summary>
    public DataFrame Count(string name = "count")
        => Build([(name, name, _ => 0.0)], countMode: true, name);

    /// <summary>Mean of one column per group.</summary>
    public DataFrame Mean(params string[] columns) => Aggregate("mean", columns);

    /// <summary>Sum of one column per group.</summary>
    public DataFrame Sum(params string[] columns) => Aggregate("sum", columns);

    /// <summary>Minimum of one column per group.</summary>
    public DataFrame Min(params string[] columns) => Aggregate("min", columns);

    /// <summary>Maximum of one column per group.</summary>
    public DataFrame Max(params string[] columns) => Aggregate("max", columns);

    /// <summary>Median of one column per group.</summary>
    public DataFrame Median(params string[] columns) => Aggregate("median", columns);

    /// <summary>Standard deviation of one column per group.</summary>
    public DataFrame Std(params string[] columns) => Aggregate("std", columns);

    /// <summary>Variance of one column per group.</summary>
    public DataFrame Var(params string[] columns) => Aggregate("var", columns);

    /// <summary>
    /// Applies a named aggregate to the given numeric columns (all of them when none are named).
    /// </summary>
    public DataFrame Aggregate(string how, params string[] columns)
    {
        var targets = columns.Length > 0
            ? columns
            : _source.NumericColumns.Select(c => c.Name).Where(n => !_keys.Contains(n)).ToArray();

        Func<double[], double> reducer = how.ToLowerInvariant() switch
        {
            "mean" => Reducers.Mean,
            "sum" => Reducers.Sum,
            "min" => Reducers.Min,
            "max" => Reducers.Max,
            "median" => Reducers.Median,
            "std" => Reducers.Std,
            "var" => Reducers.Var,
            "count" => v => v.Length,
            "first" => v => v.Length == 0 ? double.NaN : v[0],
            "last" => v => v.Length == 0 ? double.NaN : v[^1],
            _ => throw new ArgumentException($"Unknown aggregate '{how}'. Use mean, sum, min, max, median, std, var, count, first or last."),
        };

        return Build(targets.Select(t => (t, t, reducer)).ToArray(), countMode: false, null);
    }

    /// <summary>Applies a custom reducer to one column per group.</summary>
    public DataFrame Aggregate(string column, Func<double[], double> reducer, string? resultName = null)
        => Build([(column, resultName ?? column, reducer)], countMode: false, null);

    /// <summary>Applies several named aggregates at once, e.g. <c>("Sales", "sum"), ("Sales", "mean")</c>.</summary>
    public DataFrame AggregateMany(IReadOnlyList<(string Column, string How)> specifications)
    {
        var frame = KeyFrame();
        foreach (var (column, how) in specifications)
        {
            var aggregated = Aggregate(how, column);
            frame = frame.WithColumn(aggregated.Numeric(column).Rename($"{column}_{how}"));
        }
        return frame;
    }

    private DataFrame KeyFrame()
    {
        var columns = new List<Series>();
        for (var k = 0; k < _keys.Length; k++)
        {
            var source = _source[_keys[k]];
            var index = k;
            columns.Add(source.DataType switch
            {
                DataType.Numeric => new NumericSeries(_keys[k],
                    _groups.Select(g => g.Key[index] is double d ? d : double.NaN).ToArray()),
                DataType.DateTime => new DateTimeSeries(_keys[k],
                    _groups.Select(g => g.Key[index] as DateTime?).ToArray()),
                DataType.Boolean => new BooleanSeries(_keys[k],
                    _groups.Select(g => g.Key[index] as bool?).ToArray()),
                _ => new TextSeries(_keys[k], _groups.Select(g => g.Key[index]?.ToString()).ToArray()),
            });
        }
        return new DataFrame(columns);
    }

    private DataFrame Build(
        (string Source, string Output, Func<double[], double> Reducer)[] specifications,
        bool countMode, string? countName)
    {
        var frame = KeyFrame();

        if (countMode)
        {
            var counts = _groups.Select(g => (double)g.Rows.Count).ToArray();
            return frame.WithColumn(new NumericSeries(countName ?? "count", counts));
        }

        foreach (var (column, output, reducer) in specifications)
        {
            var source = _source.Numeric(column);
            var results = new double[_groups.Count];
            for (var g = 0; g < _groups.Count; g++)
            {
                var rows = _groups[g].Rows;
                var values = new List<double>(rows.Count);
                foreach (var row in rows)
                {
                    var v = source[row];
                    if (!double.IsNaN(v)) values.Add(v);
                }
                results[g] = reducer(values.ToArray());
            }
            frame = frame.WithColumn(new NumericSeries(output, results));
        }
        return frame;
    }
}

/// <summary>The reducers used by <see cref="GroupedDataFrame.Aggregate(string, string[])"/>.</summary>
internal static class Reducers
{
    public static double Sum(double[] v)
    {
        var acc = 0.0;
        foreach (var x in v) acc += x;
        return acc;
    }

    public static double Mean(double[] v) => v.Length == 0 ? double.NaN : Sum(v) / v.Length;

    public static double Min(double[] v) => v.Length == 0 ? double.NaN : v.Min();

    public static double Max(double[] v) => v.Length == 0 ? double.NaN : v.Max();

    public static double Median(double[] v)
    {
        if (v.Length == 0) return double.NaN;
        var sorted = (double[])v.Clone();
        Array.Sort(sorted);
        return GraviNum.Statistics.PercentileOfSorted(sorted, 50);
    }

    public static double Var(double[] v)
    {
        if (v.Length < 2) return double.NaN;
        var mean = Mean(v);
        var acc = 0.0;
        foreach (var x in v) acc += (x - mean) * (x - mean);
        return acc / (v.Length - 1);
    }

    public static double Std(double[] v) => Math.Sqrt(Var(v));
}
