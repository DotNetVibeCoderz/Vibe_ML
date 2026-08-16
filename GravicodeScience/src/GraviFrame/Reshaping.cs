namespace Gravicode.Science.GraviFrame;

/// <summary>Long-to-wide and wide-to-long reshaping: <c>Pivot</c>, <c>PivotTable</c> and <c>Melt</c>.</summary>
public static class Reshaping
{
    /// <summary>
    /// Turns long data into a wide table: one row per distinct <paramref name="index"/> value,
    /// one column per distinct <paramref name="columns"/> value, cells aggregated from
    /// <paramref name="values"/>. Missing combinations come back as NaN rather than as an error,
    /// which is what makes this usable on real, ragged data.
    /// </summary>
    public static DataFrame Pivot(DataFrame frame, string index, string columns, string values, string aggregate = "mean")
    {
        var indexColumn = frame[index];
        var columnColumn = frame[columns];
        var valueColumn = frame.Numeric(values);

        var indexKeys = new List<object?>();
        var indexLookup = new Dictionary<string, int>(StringComparer.Ordinal);
        var columnKeys = new List<string>();
        var columnLookup = new Dictionary<string, int>(StringComparer.Ordinal);

        // First pass: discover the axes, preserving first-seen order.
        for (var row = 0; row < frame.RowCount; row++)
        {
            var indexKey = indexColumn.GetValue(row)?.ToString() ?? "";
            if (!indexLookup.ContainsKey(indexKey))
            {
                indexLookup[indexKey] = indexKeys.Count;
                indexKeys.Add(indexColumn.GetValue(row));
            }

            var columnKey = columnColumn.GetValue(row)?.ToString() ?? "";
            if (!columnLookup.ContainsKey(columnKey))
            {
                columnLookup[columnKey] = columnKeys.Count;
                columnKeys.Add(columnKey);
            }
        }

        // Second pass: collect the values falling into each cell.
        var buckets = new List<double>[indexKeys.Count, columnKeys.Count];
        for (var row = 0; row < frame.RowCount; row++)
        {
            var v = valueColumn[row];
            if (double.IsNaN(v)) continue;
            var i = indexLookup[indexColumn.GetValue(row)?.ToString() ?? ""];
            var j = columnLookup[columnColumn.GetValue(row)?.ToString() ?? ""];
            (buckets[i, j] ??= []).Add(v);
        }

        var reducer = ResolveReducer(aggregate);
        var result = new List<Series>
        {
            BuildIndexColumn(index, indexColumn.DataType, indexKeys),
        };

        for (var j = 0; j < columnKeys.Count; j++)
        {
            var cells = new double[indexKeys.Count];
            for (var i = 0; i < indexKeys.Count; i++)
            {
                var bucket = buckets[i, j];
                cells[i] = bucket is null ? double.NaN : reducer(bucket.ToArray());
            }
            result.Add(new NumericSeries(columnKeys[j], cells));
        }

        return new DataFrame(result);
    }

    /// <summary>
    /// A pivot with several value columns at once; each produces one output column per
    /// <paramref name="columns"/> value, named <c>value_column</c>.
    /// </summary>
    public static DataFrame PivotTable(DataFrame frame, string index, string columns,
        IReadOnlyList<string> values, string aggregate = "mean")
    {
        DataFrame? result = null;
        foreach (var value in values)
        {
            var pivoted = Pivot(frame, index, columns, value, aggregate);
            if (result is null)
            {
                result = pivoted;
                foreach (var name in pivoted.ColumnNames.Skip(1))
                    result = result.Rename(name, $"{value}_{name}");
                continue;
            }
            foreach (var name in pivoted.ColumnNames.Skip(1))
                result = result.WithColumn(pivoted.Numeric(name).Rename($"{value}_{name}"));
        }
        return result ?? DataFrame.Empty();
    }

    /// <summary>
    /// Collapses <paramref name="valueColumns"/> into two columns - one holding the original
    /// column name, one holding the value - repeating <paramref name="idColumns"/> for each.
    /// </summary>
    public static DataFrame Melt(DataFrame frame, IReadOnlyList<string> idColumns,
        IReadOnlyList<string> valueColumns, string variableName = "variable", string valueName = "value")
    {
        var rows = frame.RowCount;
        var total = rows * valueColumns.Count;

        var sourceIndices = new int[total];
        var variables = new string?[total];
        var measurements = new double[total];

        var k = 0;
        foreach (var column in valueColumns)
        {
            var series = frame.Numeric(column);
            for (var row = 0; row < rows; row++)
            {
                sourceIndices[k] = row;
                variables[k] = column;
                measurements[k] = series[row];
                k++;
            }
        }

        var result = new List<Series>();
        foreach (var id in idColumns) result.Add(frame[id].Take(sourceIndices));
        result.Add(new TextSeries(variableName, variables));
        result.Add(new NumericSeries(valueName, measurements));
        return new DataFrame(result);
    }

    /// <summary>Turns a categorical column into one 0/1 column per category.</summary>
    public static DataFrame OneHot(DataFrame frame, string column, string prefix = "", bool drop = true)
    {
        var series = frame[column];
        var categories = series.Unique().Select(v => v?.ToString() ?? "missing").Distinct().OrderBy(v => v, StringComparer.Ordinal).ToList();

        var result = drop ? frame.Drop(column) : frame;
        foreach (var category in categories)
        {
            var indicator = new double[frame.RowCount];
            for (var row = 0; row < frame.RowCount; row++)
                indicator[row] = (series.GetValue(row)?.ToString() ?? "missing") == category ? 1.0 : 0.0;
            result = result.WithColumn(new NumericSeries($"{(prefix.Length > 0 ? prefix : column)}_{category}", indicator));
        }
        return result;
    }

    private static Series BuildIndexColumn(string name, DataType kind, List<object?> keys) => kind switch
    {
        DataType.Numeric => new NumericSeries(name, keys.Select(k => k is double d ? d : double.NaN).ToArray()),
        DataType.DateTime => new DateTimeSeries(name, keys.Select(k => k as DateTime?).ToArray()),
        DataType.Boolean => new BooleanSeries(name, keys.Select(k => k as bool?).ToArray()),
        _ => new TextSeries(name, keys.Select(k => k?.ToString()).ToArray()),
    };

    private static Func<double[], double> ResolveReducer(string aggregate) => aggregate.ToLowerInvariant() switch
    {
        "mean" => Reducers.Mean,
        "sum" => Reducers.Sum,
        "min" => Reducers.Min,
        "max" => Reducers.Max,
        "median" => Reducers.Median,
        "std" => Reducers.Std,
        "count" => v => v.Length,
        "first" => v => v.Length == 0 ? double.NaN : v[0],
        _ => throw new ArgumentException($"Unknown aggregate '{aggregate}'."),
    };
}
