namespace Gravicode.Science.GraviFrame;

/// <summary>
/// Window functions evaluated within groups, and time-series joins that match on nearest key.
/// </summary>
/// <remarks>
/// Both of these exist because the obvious alternative is silently wrong. A rolling mean computed
/// over a frame holding several customers mixes one customer's history into another's; an ordinary
/// join on a timestamp matches only where two systems recorded the identical instant, which real
/// clocks never do.
/// </remarks>
public static class Windowing
{
    /// <summary>
    /// A rank within each partition, starting at 1.
    /// </summary>
    /// <param name="frame">The rows to rank.</param>
    /// <param name="partitionBy">Columns that define the groups; rows only compete within a group.</param>
    /// <param name="orderBy">Column ranked on.</param>
    /// <param name="descending">Rank the largest value first.</param>
    /// <param name="dense">
    /// With ties, <c>false</c> leaves gaps after them (1, 2, 2, 4) and <c>true</c> does not
    /// (1, 2, 2, 3).
    /// </param>
    /// <param name="name">Name of the returned series.</param>
    public static NumericSeries Rank(DataFrame frame, IReadOnlyList<string> partitionBy,
        string orderBy, bool descending = false, bool dense = false, string name = "rank")
    {
        ArgumentNullException.ThrowIfNull(frame);
        var values = frame.Numeric(orderBy);
        var result = new double[frame.RowCount];

        foreach (var group in Partitions(frame, partitionBy))
        {
            var ordered = group
                .OrderBy(i => descending ? -values[i] : values[i])
                .ToArray();

            var rank = 0;
            var seen = 0;
            var previous = double.NaN;

            foreach (var row in ordered)
            {
                seen++;
                // A tie keeps the previous rank; what differs is whether the next distinct value
                // resumes at seen (leaving a gap) or at rank + 1.
                if (seen == 1 || values[row] != previous) rank = dense ? rank + 1 : seen;
                previous = values[row];
                result[row] = rank;
            }
        }

        return new NumericSeries(name, result);
    }

    /// <summary>A running total within each partition, in the frame's current row order.</summary>
    public static NumericSeries CumulativeSum(DataFrame frame, IReadOnlyList<string> partitionBy,
        string column, string name = "cumsum")
    {
        var values = frame.Numeric(column);
        var result = new double[frame.RowCount];

        foreach (var group in Partitions(frame, partitionBy))
        {
            var running = 0.0;
            foreach (var row in group)
            {
                running += values[row];
                result[row] = running;
            }
        }

        return new NumericSeries(name, result);
    }

    /// <summary>
    /// A rolling mean over the preceding <paramref name="window"/> rows within each partition.
    /// </summary>
    /// <remarks>
    /// Rows before the window is full are left missing rather than averaged over what is there.
    /// Filling them with a partial average is the more common choice and the more misleading one:
    /// the first few values then carry far more variance than the rest, and nothing in the output
    /// says so.
    /// </remarks>
    public static NumericSeries RollingMean(DataFrame frame, IReadOnlyList<string> partitionBy,
        string column, int window, string name = "rolling_mean")
    {
        if (window <= 0) throw new ArgumentOutOfRangeException(nameof(window));

        var values = frame.Numeric(column);
        var result = new double[frame.RowCount];
        Array.Fill(result, double.NaN);

        foreach (var group in Partitions(frame, partitionBy))
        {
            var ordered = group.ToArray();
            var running = 0.0;

            for (var i = 0; i < ordered.Length; i++)
            {
                running += values[ordered[i]];
                if (i >= window) running -= values[ordered[i - window]];
                if (i >= window - 1) result[ordered[i]] = running / window;
            }
        }

        return new NumericSeries(name, result);
    }

    /// <summary>Shifts a column forward within each partition, so row <c>i</c> sees row <c>i - offset</c>.</summary>
    /// <remarks>
    /// The building block for "change since last time", and the one place a partition boundary
    /// matters most: without it, the first row of each group would lag onto the previous group's
    /// last row, which is the sort of leak that quietly inflates a model's score.
    /// </remarks>
    public static NumericSeries Lag(DataFrame frame, IReadOnlyList<string> partitionBy,
        string column, int offset = 1, string name = "lag")
    {
        var values = frame.Numeric(column);
        var result = new double[frame.RowCount];
        Array.Fill(result, double.NaN);

        foreach (var group in Partitions(frame, partitionBy))
        {
            var ordered = group.ToArray();
            for (var i = offset; i < ordered.Length; i++)
                result[ordered[i]] = values[ordered[i - offset]];
        }

        return new NumericSeries(name, result);
    }

    /// <summary>Shifts a column backward, so row <c>i</c> sees row <c>i + offset</c>.</summary>
    public static NumericSeries Lead(DataFrame frame, IReadOnlyList<string> partitionBy,
        string column, int offset = 1, string name = "lead")
    {
        var values = frame.Numeric(column);
        var result = new double[frame.RowCount];
        Array.Fill(result, double.NaN);

        foreach (var group in Partitions(frame, partitionBy))
        {
            var ordered = group.ToArray();
            for (var i = 0; i + offset < ordered.Length; i++)
                result[ordered[i]] = values[ordered[i + offset]];
        }

        return new NumericSeries(name, result);
    }

    /// <summary>
    /// Joins each left row to the most recent right row at or before its key.
    /// </summary>
    /// <param name="left">The frame whose rows are all kept.</param>
    /// <param name="right">The frame searched for a match.</param>
    /// <param name="on">The ordering key, present in both frames.</param>
    /// <param name="tolerance">
    /// The furthest back a match may be. A right row older than this leaves the columns missing
    /// rather than matching something stale.
    /// </param>
    /// <param name="suffix">Appended to right-hand column names that collide.</param>
    /// <remarks>
    /// <para>
    /// The join real time-series work needs. Two systems almost never stamp the same instant, so
    /// an equality join between a trade log and a quote feed matches almost nothing; what is
    /// wanted is the quote that was in force when the trade happened.
    /// </para>
    /// <para>
    /// The direction is deliberately backward-only. Matching the <em>nearest</em> row in either
    /// direction is easy to write and is look-ahead: it lets a value recorded after the event
    /// inform a row describing the event, which is how a backtest ends up predicting the past.
    /// </para>
    /// </remarks>
    public static DataFrame AsOfJoin(DataFrame left, DataFrame right, string on,
        double? tolerance = null, string suffix = "_right")
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        var leftKeys = left.Numeric(on);
        var rightKeys = right.Numeric(on);

        // Both sides have to be in key order for the scan to mean anything; sorting here rather
        // than demanding it of the caller avoids a whole class of silently wrong results.
        var rightOrder = Enumerable.Range(0, right.RowCount).OrderBy(i => rightKeys[i]).ToArray();
        var sortedRightKeys = rightOrder.Select(i => rightKeys[i]).ToArray();

        var matches = new int[left.RowCount];

        for (var i = 0; i < left.RowCount; i++)
        {
            var key = leftKeys[i];
            var position = UpperBound(sortedRightKeys, key) - 1;

            if (position < 0)
            {
                matches[i] = -1;
                continue;
            }

            var candidate = rightOrder[position];
            matches[i] = tolerance is null || key - rightKeys[candidate] <= tolerance.Value
                ? candidate
                : -1;
        }

        return Combine(left, right, on, matches, suffix);
    }

    // ---------------------------------------------------------------- helpers

    /// <summary>Row indices grouped by the partition columns, each group in frame order.</summary>
    /// <remarks>An empty partition list makes the whole frame one group, which is what SQL does.</remarks>
    private static IEnumerable<List<int>> Partitions(DataFrame frame, IReadOnlyList<string> partitionBy)
    {
        if (partitionBy.Count == 0)
        {
            yield return [.. Enumerable.Range(0, frame.RowCount)];
            yield break;
        }

        var series = partitionBy.Select(n => frame[n]).ToArray();
        var groups = new Dictionary<string, List<int>>(StringComparer.Ordinal);

        for (var row = 0; row < frame.RowCount; row++)
        {
            var key = string.Join('\u001f', series.Select(s => s.GetValue(row)?.ToString() ?? "\0"));
            if (!groups.TryGetValue(key, out var list)) groups[key] = list = [];
            list.Add(row);
        }

        foreach (var group in groups.Values) yield return group;
    }

    /// <summary>Index of the first key strictly greater than <paramref name="value"/>.</summary>
    private static int UpperBound(double[] sorted, double value)
    {
        int low = 0, high = sorted.Length;
        while (low < high)
        {
            var middle = (low + high) / 2;
            if (sorted[middle] <= value) low = middle + 1;
            else high = middle;
        }
        return low;
    }

    /// <summary>Builds the joined frame from the row correspondence.</summary>
    private static DataFrame Combine(DataFrame left, DataFrame right, string on,
        int[] matches, string suffix)
    {
        var columns = new List<Series>();
        foreach (var name in left.ColumnNames) columns.Add(left[name].Rename(name));

        foreach (var name in right.ColumnNames)
        {
            if (name == on) continue;      // the key is already there from the left frame

            var source = right[name];
            var target = left.ColumnNames.Contains(name) ? name + suffix : name;

            columns.Add(source switch
            {
                NumericSeries numeric => new NumericSeries(target,
                    [.. matches.Select(m => m < 0 ? double.NaN : numeric[m])]),
                TextSeries text => new TextSeries(target,
                    [.. matches.Select(m => m < 0 ? null : text[m])]),
                BooleanSeries boolean => new BooleanSeries(target,
                    [.. matches.Select(m => m < 0 ? (bool?)null : boolean[m])]),
                _ => throw new NotSupportedException($"Cannot as-of join a {source.DataType} column."),
            });
        }

        return new DataFrame(columns);
    }
}

