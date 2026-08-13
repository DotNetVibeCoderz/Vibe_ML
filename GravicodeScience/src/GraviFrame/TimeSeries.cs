namespace Gravicode.Science.GraviFrame;

/// <summary>Time-series operations on <see cref="NumericSeries"/>: shifting, rolling and resampling.</summary>
public static class TimeSeriesOps
{
    /// <summary>Shifts values forward (positive) or backward (negative), padding with missing.</summary>
    public static NumericSeries Shift(this NumericSeries series, int periods)
    {
        var result = new double[series.Length];
        Array.Fill(result, double.NaN);

        if (periods >= 0)
            for (var i = periods; i < series.Length; i++) result[i] = series[i - periods];
        else
            for (var i = 0; i < series.Length + periods; i++) result[i] = series[i - periods];

        return new NumericSeries(series.Name, result);
    }

    /// <summary>Difference against the value <paramref name="periods"/> rows earlier.</summary>
    public static NumericSeries Diff(this NumericSeries series, int periods = 1)
    {
        var shifted = series.Shift(periods);
        var result = new double[series.Length];
        for (var i = 0; i < series.Length; i++) result[i] = series[i] - shifted[i];
        return new NumericSeries($"{series.Name}_diff", result);
    }

    /// <summary>Fractional change against the value <paramref name="periods"/> rows earlier.</summary>
    public static NumericSeries PercentChange(this NumericSeries series, int periods = 1)
    {
        var shifted = series.Shift(periods);
        var result = new double[series.Length];
        for (var i = 0; i < series.Length; i++) result[i] = series[i] / shifted[i] - 1.0;
        return new NumericSeries($"{series.Name}_pct_change", result);
    }

    /// <summary>Starts a rolling window over the column.</summary>
    public static RollingWindow Rolling(this NumericSeries series, int window, int? minPeriods = null)
        => new(series, window, minPeriods ?? window);

    /// <summary>Starts an expanding (cumulative) window over the column.</summary>
    public static RollingWindow Expanding(this NumericSeries series, int minPeriods = 1)
        => new(series, series.Length, minPeriods, expanding: true);

    /// <summary>Exponentially weighted moving average with the given smoothing factor.</summary>
    public static NumericSeries ExponentialMovingAverage(this NumericSeries series, double alpha)
    {
        if (alpha is <= 0 or > 1) throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be in (0, 1].");

        var result = new double[series.Length];
        var accumulator = double.NaN;
        for (var i = 0; i < series.Length; i++)
        {
            var v = series[i];
            if (double.IsNaN(v)) { result[i] = accumulator; continue; }
            accumulator = double.IsNaN(accumulator) ? v : alpha * v + (1 - alpha) * accumulator;
            result[i] = accumulator;
        }
        return new NumericSeries($"{series.Name}_ema", result);
    }

    /// <summary>Exponentially weighted moving average specified by a span, as in pandas.</summary>
    public static NumericSeries ExponentialMovingAverageBySpan(this NumericSeries series, int span)
        => series.ExponentialMovingAverage(2.0 / (span + 1));
}

/// <summary>
/// A rolling (or expanding) window over a numeric column, aggregated one position at a time.
/// </summary>
/// <remarks>
/// <see cref="Mean"/> and <see cref="Sum"/> use an incremental accumulator, so a window of any
/// width costs one add and one subtract per row rather than a full re-scan. Order statistics
/// (<see cref="Median"/>, <see cref="Quantile"/>) do have to re-sort each window, which is why
/// they are noticeably slower on wide windows.
/// </remarks>
public sealed class RollingWindow
{
    private readonly NumericSeries _series;
    private readonly int _window;
    private readonly int _minPeriods;
    private readonly bool _expanding;

    internal RollingWindow(NumericSeries series, int window, int minPeriods, bool expanding = false)
    {
        if (window <= 0) throw new ArgumentOutOfRangeException(nameof(window), "Window must be positive.");
        _series = series;
        _window = window;
        _minPeriods = Math.Max(1, minPeriods);
        _expanding = expanding;
    }

    /// <summary>Rolling mean.</summary>
    public NumericSeries Mean() => RunningSum(divide: true, "mean");

    /// <summary>Rolling sum.</summary>
    public NumericSeries Sum() => RunningSum(divide: false, "sum");

    private NumericSeries RunningSum(bool divide, string suffix)
    {
        var n = _series.Length;
        var result = new double[n];
        var sum = 0.0;
        var count = 0;

        for (var i = 0; i < n; i++)
        {
            var incoming = _series[i];
            if (!double.IsNaN(incoming)) { sum += incoming; count++; }

            if (!_expanding && i >= _window)
            {
                var outgoing = _series[i - _window];
                if (!double.IsNaN(outgoing)) { sum -= outgoing; count--; }
            }

            result[i] = count >= _minPeriods ? (divide ? sum / count : sum) : double.NaN;
        }
        return new NumericSeries($"{_series.Name}_{suffix}{(_expanding ? "_expanding" : $"_{_window}")}", result);
    }

    /// <summary>Rolling standard deviation.</summary>
    public NumericSeries Std(int ddof = 1) => Apply(v => Reducers.Std(v), $"std_{_window}", ddof + 1);

    /// <summary>Rolling variance.</summary>
    public NumericSeries Var(int ddof = 1) => Apply(v => Reducers.Var(v), $"var_{_window}", ddof + 1);

    /// <summary>Rolling minimum.</summary>
    public NumericSeries Min() => Apply(Reducers.Min, $"min_{_window}");

    /// <summary>Rolling maximum.</summary>
    public NumericSeries Max() => Apply(Reducers.Max, $"max_{_window}");

    /// <summary>Rolling median.</summary>
    public NumericSeries Median() => Apply(Reducers.Median, $"median_{_window}");

    /// <summary>Rolling quantile.</summary>
    public NumericSeries Quantile(double q) => Apply(v =>
    {
        if (v.Length == 0) return double.NaN;
        var sorted = (double[])v.Clone();
        Array.Sort(sorted);
        return GraviNum.Statistics.PercentileOfSorted(sorted, q * 100.0);
    }, $"q{q:0.##}_{_window}");

    /// <summary>Rolling count of present values.</summary>
    public NumericSeries Count() => Apply(v => v.Length, $"count_{_window}", 1);

    /// <summary>Applies an arbitrary reducer to each window.</summary>
    public NumericSeries Apply(Func<double[], double> reducer, string suffix = "apply", int? minimum = null)
    {
        var n = _series.Length;
        var result = new double[n];
        var required = Math.Max(_minPeriods, minimum ?? _minPeriods);
        var buffer = new List<double>(_window);

        for (var i = 0; i < n; i++)
        {
            buffer.Clear();
            var from = _expanding ? 0 : Math.Max(0, i - _window + 1);
            for (var k = from; k <= i; k++)
            {
                var v = _series[k];
                if (!double.IsNaN(v)) buffer.Add(v);
            }
            result[i] = buffer.Count >= required ? reducer(buffer.ToArray()) : double.NaN;
        }
        return new NumericSeries($"{_series.Name}_{suffix}", result);
    }
}

/// <summary>How a resampling bucket is labelled and how long it runs.</summary>
public enum ResampleFrequency
{
    /// <summary>Calendar day.</summary>
    Daily,

    /// <summary>Seven-day bucket starting on the first observation's weekday.</summary>
    Weekly,

    /// <summary>Calendar month.</summary>
    Monthly,

    /// <summary>Calendar quarter.</summary>
    Quarterly,

    /// <summary>Calendar year.</summary>
    Yearly,

    /// <summary>Clock hour.</summary>
    Hourly,
}

/// <summary>
/// Resamples a frame with a timestamp column onto a regular calendar frequency.
/// </summary>
/// <remarks>
/// Buckets are derived from the calendar rather than from fixed tick counts, so a monthly
/// resample of daily data lands on real month boundaries regardless of month length or leap years.
/// </remarks>
public static class Resampling
{
    /// <summary>Aggregates numeric columns into calendar buckets of the given frequency.</summary>
    public static DataFrame Resample(DataFrame frame, string timeColumn, ResampleFrequency frequency,
        string aggregate = "mean", IReadOnlyList<string>? valueColumns = null)
    {
        var times = frame.DateTimes(timeColumn);
        var targets = valueColumns
            ?? frame.NumericColumns.Select(c => c.Name).ToList();

        var buckets = new Dictionary<DateTime, List<int>>();
        for (var i = 0; i < frame.RowCount; i++)
        {
            var t = times[i];
            if (t is null) continue;
            var key = Floor(t.Value, frequency);
            if (!buckets.TryGetValue(key, out var rows)) buckets[key] = rows = [];
            rows.Add(i);
        }

        var ordered = buckets.OrderBy(kv => kv.Key).ToList();
        var columns = new List<Series>
        {
            new DateTimeSeries(timeColumn, ordered.Select(kv => (DateTime?)kv.Key).ToArray()),
        };

        var reducer = aggregate.ToLowerInvariant() switch
        {
            "mean" => Reducers.Mean,
            "sum" => Reducers.Sum,
            "min" => Reducers.Min,
            "max" => Reducers.Max,
            "median" => Reducers.Median,
            "std" => Reducers.Std,
            "count" => new Func<double[], double>(v => v.Length),
            "first" => v => v.Length == 0 ? double.NaN : v[0],
            "last" => v => v.Length == 0 ? double.NaN : v[^1],
            _ => throw new ArgumentException($"Unknown aggregate '{aggregate}'."),
        };

        foreach (var name in targets)
        {
            var source = frame.Numeric(name);
            var values = new double[ordered.Count];
            for (var b = 0; b < ordered.Count; b++)
            {
                var present = ordered[b].Value
                    .Select(row => source[row])
                    .Where(v => !double.IsNaN(v))
                    .ToArray();
                values[b] = reducer(present);
            }
            columns.Add(new NumericSeries(name, values));
        }

        return new DataFrame(columns);
    }

    /// <summary>Rounds a timestamp down to the start of its bucket.</summary>
    public static DateTime Floor(DateTime value, ResampleFrequency frequency) => frequency switch
    {
        ResampleFrequency.Hourly => new DateTime(value.Year, value.Month, value.Day, value.Hour, 0, 0, value.Kind),
        ResampleFrequency.Daily => value.Date,
        ResampleFrequency.Weekly => value.Date.AddDays(-(int)value.DayOfWeek),
        ResampleFrequency.Monthly => new DateTime(value.Year, value.Month, 1, 0, 0, 0, value.Kind),
        ResampleFrequency.Quarterly => new DateTime(value.Year, (value.Month - 1) / 3 * 3 + 1, 1, 0, 0, 0, value.Kind),
        ResampleFrequency.Yearly => new DateTime(value.Year, 1, 1, 0, 0, 0, value.Kind),
        _ => value,
    };
}
