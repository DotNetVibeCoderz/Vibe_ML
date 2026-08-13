using System.Text;
using Gravicode.Science.GraviFrame.Io;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviFrame;

/// <summary>
/// A two-dimensional table of named, strongly typed columns - the pandas <c>DataFrame</c> for .NET.
/// </summary>
/// <remarks>
/// Storage is columnar: the frame owns an ordered list of <see cref="Series"/>, each backed by a
/// single typed array. That layout is what makes column statistics, filters and group-bys fast,
/// because each one touches a contiguous buffer instead of striding across row objects. Row-wise
/// operations (filter, sort, join) work by producing an index vector and then asking every column
/// to <see cref="Series.Take"/> it, so the cost is one pass per column rather than per cell.
/// </remarks>
public sealed class DataFrame
{
    private readonly List<Series> _columns;
    private readonly Dictionary<string, int> _lookup;

    /// <summary>Builds a frame from an ordered set of equally long columns.</summary>
    public DataFrame(IEnumerable<Series> columns)
    {
        _columns = columns.ToList();
        if (_columns.Count > 0)
        {
            var length = _columns[0].Length;
            foreach (var c in _columns)
                if (c.Length != length)
                    throw new ArgumentException(
                        $"Column '{c.Name}' has {c.Length} rows but '{_columns[0].Name}' has {length}.");
        }

        _lookup = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < _columns.Count; i++)
        {
            if (!_lookup.TryAdd(_columns[i].Name, i))
                throw new ArgumentException($"Duplicate column name '{_columns[i].Name}'.");
        }
    }

    /// <summary>An empty frame.</summary>
    public static DataFrame Empty() => new([]);

    /// <summary>Number of rows.</summary>
    public int RowCount => _columns.Count == 0 ? 0 : _columns[0].Length;

    /// <summary>Number of columns.</summary>
    public int ColumnCount => _columns.Count;

    /// <summary>Rows and columns, like pandas' <c>shape</c>.</summary>
    public (int Rows, int Columns) Shape => (RowCount, ColumnCount);

    /// <summary>The columns, in order.</summary>
    public IReadOnlyList<Series> Columns => _columns;

    /// <summary>Column names, in order.</summary>
    public IReadOnlyList<string> ColumnNames => _columns.Select(c => c.Name).ToList();

    /// <summary>Looks up a column by name.</summary>
    public Series this[string name] => _lookup.TryGetValue(name, out var i)
        ? _columns[i]
        : throw new KeyNotFoundException($"No column named '{name}'. Available: {string.Join(", ", ColumnNames)}.");

    /// <summary>True when a column with this name exists.</summary>
    public bool HasColumn(string name) => _lookup.ContainsKey(name);

    /// <summary>Looks up a numeric column, failing clearly if the type is wrong.</summary>
    public NumericSeries Numeric(string name) => this[name] as NumericSeries
        ?? throw new InvalidOperationException($"Column '{name}' is {this[name].DataType}, not Numeric.");

    /// <summary>Looks up a text column.</summary>
    public TextSeries Text(string name) => this[name] as TextSeries
        ?? throw new InvalidOperationException($"Column '{name}' is {this[name].DataType}, not Text.");

    /// <summary>Looks up a timestamp column.</summary>
    public DateTimeSeries DateTimes(string name) => this[name] as DateTimeSeries
        ?? throw new InvalidOperationException($"Column '{name}' is {this[name].DataType}, not DateTime.");

    /// <summary>Looks up a boolean column.</summary>
    public BooleanSeries Booleans(string name) => this[name] as BooleanSeries
        ?? throw new InvalidOperationException($"Column '{name}' is {this[name].DataType}, not Boolean.");

    /// <summary>Every numeric column, in order.</summary>
    public IReadOnlyList<NumericSeries> NumericColumns => _columns.OfType<NumericSeries>().ToList();

    // ---------------------------------------------------------------- construction

    /// <summary>Reads a CSV file, inferring each column's type from its contents.</summary>
    public static DataFrame ReadCsv(string path, CsvOptions? options = null)
        => CsvReader.Read(path, options ?? new CsvOptions());

    /// <summary>Reads CSV content held in a string.</summary>
    public static DataFrame ParseCsv(string content, CsvOptions? options = null)
        => CsvReader.Parse(content, options ?? new CsvOptions());

    /// <summary>
    /// Reads a CSV file through a memory-mapped view, which keeps peak memory near the size of a
    /// single chunk rather than the whole file.
    /// </summary>
    public static DataFrame ReadCsvMemoryMapped(string path, CsvOptions? options = null)
        => CsvReader.ReadMemoryMapped(path, options ?? new CsvOptions());

    /// <summary>Writes the frame as CSV.</summary>
    public void WriteCsv(string path, string delimiter = ",") => CsvWriter.Write(this, path, delimiter);

    /// <summary>Builds a frame from a numeric matrix and column names.</summary>
    public static DataFrame FromMatrix(NdArray matrix, IReadOnlyList<string> columnNames)
    {
        if (matrix.Rank != 2) throw new ArgumentException("FromMatrix expects a rank 2 array.");
        if (columnNames.Count != matrix.Shape[1])
            throw new ArgumentException($"Got {columnNames.Count} names for {matrix.Shape[1]} columns.");

        var columns = new List<Series>();
        for (var j = 0; j < matrix.Shape[1]; j++)
        {
            var values = new double[matrix.Shape[0]];
            for (var i = 0; i < values.Length; i++) values[i] = matrix[i, j];
            columns.Add(new NumericSeries(columnNames[j], values));
        }
        return new DataFrame(columns);
    }

    /// <summary>Builds a frame from a dictionary of column name to values.</summary>
    public static DataFrame FromColumns(IReadOnlyDictionary<string, double[]> columns)
        => new(columns.Select(kv => (Series)new NumericSeries(kv.Key, kv.Value)));

    // ---------------------------------------------------------------- shaping

    /// <summary>A new frame with the given column added or replaced.</summary>
    public DataFrame WithColumn(Series column)
    {
        if (RowCount > 0 && column.Length != RowCount)
            throw new ArgumentException($"Column '{column.Name}' has {column.Length} rows, expected {RowCount}.");

        var columns = _columns.ToList();
        if (_lookup.TryGetValue(column.Name, out var existing)) columns[existing] = column;
        else columns.Add(column);
        return new DataFrame(columns);
    }

    /// <summary>A new frame with a computed numeric column added.</summary>
    public DataFrame WithColumn(string name, Func<int, double> compute)
    {
        var values = new double[RowCount];
        for (var i = 0; i < RowCount; i++) values[i] = compute(i);
        return WithColumn(new NumericSeries(name, values));
    }

    /// <summary>A new frame without the named columns.</summary>
    public DataFrame Drop(params string[] names)
    {
        var drop = new HashSet<string>(names, StringComparer.Ordinal);
        return new DataFrame(_columns.Where(c => !drop.Contains(c.Name)));
    }

    /// <summary>A new frame containing only the named columns, in the order given.</summary>
    public DataFrame SelectColumns(params string[] names)
        => new(names.Select(n => this[n]));

    /// <summary>A new frame with one column renamed.</summary>
    public DataFrame Rename(string from, string to)
        => new(_columns.Select(c => c.Name == from ? c.Rename(to) : c));

    /// <summary>The first <paramref name="count"/> rows.</summary>
    public DataFrame Head(int count = 5) => Take(Enumerable.Range(0, Math.Min(count, RowCount)).ToArray());

    /// <summary>The last <paramref name="count"/> rows.</summary>
    public DataFrame Tail(int count = 5)
    {
        var start = Math.Max(0, RowCount - count);
        return Take(Enumerable.Range(start, RowCount - start).ToArray());
    }

    /// <summary>A contiguous row range, <c>[start, start + count)</c>.</summary>
    public DataFrame Rows(int start, int count) => Take(Enumerable.Range(start, count).ToArray());

    /// <summary>Rows selected (and reordered) by index.</summary>
    public DataFrame Take(IReadOnlyList<int> indices) => new(_columns.Select(c => c.Take(indices)));

    /// <summary>Rows where <paramref name="mask"/> is true.</summary>
    public DataFrame Filter(bool[] mask)
    {
        if (mask.Length != RowCount)
            throw new ArgumentException($"Mask has {mask.Length} entries but the frame has {RowCount} rows.");
        var indices = new List<int>();
        for (var i = 0; i < mask.Length; i++) if (mask[i]) indices.Add(i);
        return Take(indices);
    }

    /// <summary>Rows where the predicate holds, evaluated against a row accessor.</summary>
    public DataFrame Filter(Func<RowView, bool> predicate)
    {
        var indices = new List<int>();
        for (var i = 0; i < RowCount; i++) if (predicate(new RowView(this, i))) indices.Add(i);
        return Take(indices);
    }

    /// <summary>Rows where a numeric column satisfies <paramref name="predicate"/>.</summary>
    public DataFrame FilterBy(string column, Func<double, bool> predicate)
        => Filter(Numeric(column).Where(predicate));

    /// <summary>A random subset of rows.</summary>
    public DataFrame Sample(int count, int seed = 42)
    {
        var rng = new GraviRandom(seed);
        return Take(rng.Choice(RowCount, Math.Min(count, RowCount), replace: false));
    }

    /// <summary>Rows sorted by one column.</summary>
    public DataFrame SortBy(string column, bool ascending = true)
    {
        var series = this[column];
        var indices = Enumerable.Range(0, RowCount).ToArray();

        Comparison<int> comparison = series switch
        {
            NumericSeries n => (a, b) => n[a].CompareTo(n[b]),
            DateTimeSeries d => (a, b) => Nullable.Compare(d[a], d[b]),
            BooleanSeries bo => (a, b) => Nullable.Compare(bo[a], bo[b]),
            _ => (a, b) => string.CompareOrdinal(series.GetValue(a)?.ToString(), series.GetValue(b)?.ToString()),
        };

        Array.Sort(indices, ascending ? comparison : (a, b) => comparison(b, a));
        return Take(indices);
    }

    /// <summary>Rows sorted by several columns, each ascending or descending.</summary>
    public DataFrame SortBy(IReadOnlyList<(string Column, bool Ascending)> keys)
    {
        var result = this;
        // Stable sorts applied from the least significant key upwards give lexicographic order.
        for (var i = keys.Count - 1; i >= 0; i--)
            result = result.StableSortBy(keys[i].Column, keys[i].Ascending);
        return result;
    }

    private DataFrame StableSortBy(string column, bool ascending)
    {
        var series = this[column];
        var indices = Enumerable.Range(0, RowCount);

        var ordered = series switch
        {
            NumericSeries n => ascending
                ? indices.OrderBy(i => n[i]).ToArray()
                : indices.OrderByDescending(i => n[i]).ToArray(),
            DateTimeSeries d => ascending
                ? indices.OrderBy(i => d[i]).ToArray()
                : indices.OrderByDescending(i => d[i]).ToArray(),
            _ => ascending
                ? indices.OrderBy(i => series.GetValue(i)?.ToString(), StringComparer.Ordinal).ToArray()
                : indices.OrderByDescending(i => series.GetValue(i)?.ToString(), StringComparer.Ordinal).ToArray(),
        };
        return Take(ordered);
    }

    /// <summary>Stacks frames with identical columns on top of each other.</summary>
    public static DataFrame Concat(IReadOnlyList<DataFrame> frames)
    {
        if (frames.Count == 0) return Empty();
        var names = frames[0].ColumnNames;
        foreach (var f in frames)
            if (!f.ColumnNames.SequenceEqual(names))
                throw new ArgumentException("Concat requires identical column names in the same order.");

        var columns = new List<Series>();
        foreach (var name in names)
        {
            var pieces = frames.Select(f => f[name]).ToList();
            columns.Add(SeriesOperations.Concat(pieces));
        }
        return new DataFrame(columns);
    }

    // ---------------------------------------------------------------- missing data

    /// <summary>Drops rows that have any missing value (or, with <paramref name="how"/> = <c>all</c>, only fully missing rows).</summary>
    public DataFrame DropMissing(string how = "any", IReadOnlyList<string>? subset = null)
    {
        var considered = subset is null ? _columns : subset.Select(n => this[n]).ToList();
        var keep = new List<int>();
        for (var i = 0; i < RowCount; i++)
        {
            var missing = considered.Count(c => c.IsMissing(i));
            var drop = how == "all" ? missing == considered.Count : missing > 0;
            if (!drop) keep.Add(i);
        }
        return Take(keep);
    }

    /// <summary>Replaces missing values in every numeric column with a constant.</summary>
    public DataFrame FillMissing(double value)
        => new(_columns.Select(c => c is NumericSeries n ? n.FillMissing(value) : c));

    /// <summary>Replaces missing values in every numeric column with that column's mean.</summary>
    public DataFrame FillMissingWithMean()
        => new(_columns.Select(c => c is NumericSeries n ? n.FillMissingWithMean() : c));

    /// <summary>Missing value count per column.</summary>
    public IReadOnlyDictionary<string, int> MissingCounts()
        => _columns.ToDictionary(c => c.Name, c => c.MissingCount);

    // ---------------------------------------------------------------- analytics

    /// <summary>Descriptive statistics for every numeric column.</summary>
    public DataFrame Describe()
    {
        var statistics = new[] { "count", "mean", "std", "min", "25%", "50%", "75%", "max" };
        var columns = new List<Series> { new TextSeries("statistic", statistics.Cast<string?>().ToArray()) };

        foreach (var numeric in NumericColumns)
        {
            var summary = numeric.Describe();
            columns.Add(new NumericSeries(numeric.Name, statistics.Select(s => summary[s]).ToArray()));
        }
        return new DataFrame(columns);
    }

    /// <summary>Correlation matrix over the numeric columns.</summary>
    public DataFrame CorrelationMatrix()
    {
        var numeric = NumericColumns;
        var columns = new List<Series>
        {
            new TextSeries("column", numeric.Select(n => (string?)n.Name).ToArray()),
        };

        foreach (var a in numeric)
            columns.Add(new NumericSeries(a.Name, numeric.Select(b => a.Correlation(b)).ToArray()));

        return new DataFrame(columns);
    }

    /// <summary>Groups rows by the distinct values of one or more columns.</summary>
    public GroupedDataFrame GroupBy(params string[] keys) => new(this, keys);

    /// <summary>
    /// Long to wide: one row per <paramref name="index"/> value, one column per
    /// <paramref name="columns"/> value, filled from <paramref name="values"/>.
    /// </summary>
    public DataFrame Pivot(string index, string columns, string values, string aggregate = "mean")
        => Reshaping.Pivot(this, index, columns, values, aggregate);

    /// <summary>Wide to long: collapses <paramref name="valueColumns"/> into name/value pairs.</summary>
    public DataFrame Melt(IReadOnlyList<string> idColumns, IReadOnlyList<string> valueColumns,
        string variableName = "variable", string valueName = "value")
        => Reshaping.Melt(this, idColumns, valueColumns, variableName, valueName);

    /// <summary>Joins with another frame on a shared key column.</summary>
    public DataFrame Join(DataFrame other, string on, JoinKind kind = JoinKind.Inner, string suffix = "_right")
        => Joins.Merge(this, other, on, on, kind, suffix);

    /// <summary>Joins with another frame on differently named key columns.</summary>
    public DataFrame Merge(DataFrame other, string leftOn, string rightOn,
        JoinKind kind = JoinKind.Inner, string suffix = "_right")
        => Joins.Merge(this, other, leftOn, rightOn, kind, suffix);

    /// <summary>Every numeric column as a rows-by-features matrix.</summary>
    public NdArray ToNdArray()
    {
        var numeric = NumericColumns;
        if (numeric.Count == 0) throw new InvalidOperationException("The frame has no numeric columns.");

        var matrix = NdArray.Zeros(RowCount, numeric.Count);
        for (var j = 0; j < numeric.Count; j++)
            for (var i = 0; i < RowCount; i++)
                matrix[i, j] = numeric[j][i];
        return matrix;
    }

    /// <summary>The named columns as a rows-by-features matrix, in the order given.</summary>
    public NdArray ToNdArray(params string[] columns)
    {
        var matrix = NdArray.Zeros(RowCount, columns.Length);
        for (var j = 0; j < columns.Length; j++)
        {
            var series = Numeric(columns[j]);
            for (var i = 0; i < RowCount; i++) matrix[i, j] = series[i];
        }
        return matrix;
    }

    /// <summary>A read-only view of one row, for row-wise predicates.</summary>
    public RowView Row(int index) => new(this, index);

    /// <summary>Enumerates rows as views.</summary>
    public IEnumerable<RowView> EnumerateRows()
    {
        for (var i = 0; i < RowCount; i++) yield return new RowView(this, i);
    }

    // ---------------------------------------------------------------- rendering

    /// <inheritdoc />
    public override string ToString() => ToString(10);

    /// <summary>Renders the frame as a fixed-width table, truncated to <paramref name="maxRows"/>.</summary>
    public string ToString(int maxRows)
    {
        if (ColumnCount == 0) return "DataFrame(empty)";

        var shown = Math.Min(maxRows, RowCount);
        var headers = ColumnNames.ToArray();
        var cells = new string[shown][];

        for (var i = 0; i < shown; i++)
        {
            cells[i] = new string[ColumnCount];
            for (var j = 0; j < ColumnCount; j++)
            {
                var series = _columns[j];
                cells[i][j] = series.IsMissing(i)
                    ? "NaN"
                    : series is NumericSeries n
                        ? FormatNumber(n[i])
                        : series.GetValue(i)?.ToString() ?? "NaN";
            }
        }

        var widths = new int[ColumnCount];
        for (var j = 0; j < ColumnCount; j++)
        {
            widths[j] = headers[j].Length;
            for (var i = 0; i < shown; i++) widths[j] = Math.Max(widths[j], cells[i][j].Length);
            widths[j] = Math.Min(widths[j], 24);
        }

        var sb = new StringBuilder();
        sb.Append("  ");
        for (var j = 0; j < ColumnCount; j++) sb.Append(Pad(headers[j], widths[j])).Append("  ");
        sb.AppendLine();
        sb.Append("  ");
        for (var j = 0; j < ColumnCount; j++) sb.Append(new string('-', widths[j])).Append("  ");
        sb.AppendLine();

        for (var i = 0; i < shown; i++)
        {
            sb.Append("  ");
            for (var j = 0; j < ColumnCount; j++) sb.Append(Pad(cells[i][j], widths[j])).Append("  ");
            sb.AppendLine();
        }

        if (shown < RowCount) sb.AppendLine($"  ... {RowCount - shown} more rows");
        sb.Append($"[{RowCount} rows x {ColumnCount} columns]");
        return sb.ToString();
    }

    private static string FormatNumber(double value)
        => value == Math.Floor(value) && Math.Abs(value) < 1e15
            ? value.ToString("0")
            : value.ToString("0.####");

    private static string Pad(string text, int width)
        => text.Length > width ? text[..(width - 1)] + "…" : text.PadRight(width);

    /// <summary>A compact structural summary, like pandas' <c>info()</c>.</summary>
    public string Info()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"DataFrame: {RowCount} rows x {ColumnCount} columns");
        foreach (var c in _columns)
            sb.AppendLine($"  {c.Name,-24} {c.DataType,-10} non-null: {c.Count,-8} missing: {c.MissingCount}");
        return sb.ToString();
    }
}

/// <summary>A read-only accessor for one row, used by row-wise predicates.</summary>
public readonly struct RowView(DataFrame frame, int index)
{
    /// <summary>The frame this row belongs to.</summary>
    public DataFrame Frame { get; } = frame;

    /// <summary>The row's position in the frame.</summary>
    public int Index { get; } = index;

    /// <summary>Reads a cell as an object.</summary>
    public object? this[string column] => Frame[column].GetValue(Index);

    /// <summary>Reads a numeric cell.</summary>
    public double Number(string column) => Frame.Numeric(column)[Index];

    /// <summary>Reads a text cell.</summary>
    public string? String(string column) => Frame[column].GetValue(Index)?.ToString();

    /// <summary>Reads a timestamp cell.</summary>
    public DateTime? Date(string column) => Frame.DateTimes(column)[Index];

    /// <summary>True when the cell is missing.</summary>
    public bool IsMissing(string column) => Frame[column].IsMissing(Index);
}
