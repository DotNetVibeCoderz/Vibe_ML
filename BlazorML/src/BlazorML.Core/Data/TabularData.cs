using System.Globalization;
using System.Text;
using BlazorML.Core.Analysis;
using BlazorML.Core.Domain;

namespace BlazorML.Core.Data;

/// <summary>
/// The payload that travels along every <see cref="PortType.Dataset"/> edge.
/// <para>
/// Deliberately an in-memory table rather than an ML.NET <c>IDataView</c>: transforms, LLM
/// modules and user scripts all need to read and write individual cells, which a lazy
/// column-oriented view makes awkward. The ML layer converts to <c>IDataView</c> only at the
/// point a trainer needs one.
/// </para>
/// </summary>
public sealed class TabularData
{
    public List<DataColumn> Columns { get; } = new();
    public List<object?[]> Rows { get; } = new();

    public int RowCount => Rows.Count;
    public int ColumnCount => Columns.Count;

    public TabularData() { }

    public TabularData(IEnumerable<DataColumn> columns)
    {
        Columns.AddRange(columns);
    }

    public static TabularData WithColumns(params string[] names) =>
        new(names.Select(n => new DataColumn(n, ColumnDataType.Text)));

    public int IndexOf(string columnName)
    {
        for (var i = 0; i < Columns.Count; i++)
        {
            if (string.Equals(Columns[i].Name, columnName, StringComparison.OrdinalIgnoreCase))
            {
                return i;
            }
        }

        return -1;
    }

    public bool Has(string columnName) => IndexOf(columnName) >= 0;

    public DataColumn? Column(string columnName)
    {
        var i = IndexOf(columnName);
        return i < 0 ? null : Columns[i];
    }

    public object? Value(int row, string columnName)
    {
        var i = IndexOf(columnName);
        return i < 0 ? null : Rows[row][i];
    }

    public string? Text(int row, string columnName)
    {
        var v = Value(row, columnName);
        return v is null ? null : Convert.ToString(v, CultureInfo.InvariantCulture);
    }

    /// <summary>Appends a column, widening every existing row so the table stays rectangular.</summary>
    public int AddColumn(string name, ColumnDataType type = ColumnDataType.Text)
    {
        var unique = UniqueName(name);
        Columns.Add(new DataColumn(unique, type));

        for (var r = 0; r < Rows.Count; r++)
        {
            var widened = new object?[Columns.Count];
            Array.Copy(Rows[r], widened, Rows[r].Length);
            Rows[r] = widened;
        }

        return Columns.Count - 1;
    }

    /// <summary>Returns <paramref name="name"/>, or <c>name_2</c>, <c>name_3</c>… if it is taken.</summary>
    public string UniqueName(string name)
    {
        if (!Has(name))
        {
            return name;
        }

        var n = 2;
        while (Has($"{name}_{n}"))
        {
            n++;
        }

        return $"{name}_{n}";
    }

    public void RemoveColumn(string name)
    {
        var index = IndexOf(name);
        if (index < 0)
        {
            return;
        }

        Columns.RemoveAt(index);
        for (var r = 0; r < Rows.Count; r++)
        {
            var narrowed = new object?[Columns.Count];
            var source = Rows[r];
            for (int s = 0, d = 0; s < source.Length; s++)
            {
                if (s == index)
                {
                    continue;
                }

                if (d < narrowed.Length)
                {
                    narrowed[d++] = source[s];
                }
            }

            Rows[r] = narrowed;
        }
    }

    public object?[] NewRow() => new object?[Columns.Count];

    public void AddRow(params object?[] values)
    {
        var row = NewRow();
        Array.Copy(values, row, Math.Min(values.Length, row.Length));
        Rows.Add(row);
    }

    /// <summary>Shallow structural copy: same columns, same row instances are not shared.</summary>
    public TabularData Clone()
    {
        var copy = new TabularData(Columns.Select(c => c with { }));
        foreach (var row in Rows)
        {
            copy.Rows.Add((object?[])row.Clone());
        }

        return copy;
    }

    /// <summary>Columns only, no rows — the starting point for most transforms.</summary>
    public TabularData CloneSchema() => new(Columns.Select(c => c with { }));

    public IEnumerable<double?> NumericValues(string columnName)
    {
        var i = IndexOf(columnName);
        if (i < 0)
        {
            yield break;
        }

        foreach (var row in Rows)
        {
            yield return ToNumber(row[i]);
        }
    }

    public static double? ToNumber(object? value) => value switch
    {
        null => null,
        double d => double.IsNaN(d) ? null : d,
        float f => float.IsNaN(f) ? null : f,
        int i => i,
        long l => l,
        decimal m => (double)m,
        bool b => b ? 1d : 0d,
        DateTime dt => dt.ToOADate(),
        string s when double.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var parsed) => parsed,
        _ => null
    };

    public static bool IsBlank(object? value) =>
        value is null || (value is string s && string.IsNullOrWhiteSpace(s));

    /// <summary>Rows shaped as dictionaries, the form the script and LLM modules expect.</summary>
    public List<Dictionary<string, object?>> ToDictionaries(int limit = int.MaxValue)
    {
        var result = new List<Dictionary<string, object?>>();
        foreach (var row in Rows.Take(limit))
        {
            var dict = new Dictionary<string, object?>(Columns.Count, StringComparer.OrdinalIgnoreCase);
            for (var c = 0; c < Columns.Count; c++)
            {
                dict[Columns[c].Name] = c < row.Length ? row[c] : null;
            }

            result.Add(dict);
        }

        return result;
    }

    public static TabularData FromDictionaries(IEnumerable<Dictionary<string, object?>> rows)
    {
        var table = new TabularData();
        foreach (var dict in rows)
        {
            foreach (var key in dict.Keys)
            {
                if (!table.Has(key))
                {
                    table.AddColumn(key);
                }
            }

            var row = table.NewRow();
            foreach (var (key, value) in dict)
            {
                var i = table.IndexOf(key);
                if (i >= 0)
                {
                    row[i] = value;
                }
            }

            table.Rows.Add(row);
        }

        return table;
    }

    /// <summary>Header plus the first <paramref name="rows"/> rows, for the node preview panel.</summary>
    public List<List<string>> Preview(int rows = 10)
    {
        var preview = new List<List<string>> { Columns.Select(c => c.Name).ToList() };
        foreach (var row in Rows.Take(rows))
        {
            preview.Add(Columns.Select((_, c) =>
                c < row.Length ? Convert.ToString(row[c], CultureInfo.InvariantCulture) ?? string.Empty : string.Empty).ToList());
        }

        return preview;
    }

    public string ToCsv(char delimiter = ',')
    {
        var sb = new StringBuilder();
        sb.AppendLine(string.Join(delimiter, Columns.Select(c => Escape(c.Name, delimiter))));

        foreach (var row in Rows)
        {
            var cells = new string[Columns.Count];
            for (var c = 0; c < Columns.Count; c++)
            {
                var raw = c < row.Length ? Convert.ToString(row[c], CultureInfo.InvariantCulture) : null;
                cells[c] = Escape(raw ?? string.Empty, delimiter);
            }

            sb.AppendLine(string.Join(delimiter, cells));
        }

        return sb.ToString();
    }

    private static string Escape(string value, char delimiter)
    {
        var needsQuotes = value.Contains(delimiter) || value.Contains('"') || value.Contains('\n') || value.Contains('\r');
        return needsQuotes ? '"' + value.Replace("\"", "\"\"") + '"' : value;
    }

    /// <summary>
    /// Infers a data type per column from the values present. Called after loading a file and
    /// again after transforms that can change a column's shape.
    /// </summary>
    public void InferTypes(int sampleSize = 500)
    {
        for (var c = 0; c < Columns.Count; c++)
        {
            var seen = 0;
            var numeric = 0;
            var boolean = 0;
            var dates = 0;
            var distinct = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var longText = false;

            foreach (var row in Rows.Take(sampleSize))
            {
                var value = c < row.Length ? row[c] : null;
                if (IsBlank(value))
                {
                    continue;
                }

                seen++;
                var text = Convert.ToString(value, CultureInfo.InvariantCulture) ?? string.Empty;
                distinct.Add(text);

                if (text.Length > 60)
                {
                    longText = true;
                }

                if (double.TryParse(text, NumberStyles.Any, CultureInfo.InvariantCulture, out _))
                {
                    numeric++;
                }
                else if (bool.TryParse(text, out _))
                {
                    // Only literals ML.NET's TextLoader can itself parse as Boolean count here.
                    // "ya"/"tidak" and "yes"/"no" read as boolean to a human but blow the loader
                    // up when the column is declared Boolean, so they stay categorical — which
                    // one-hot encoding handles correctly anyway.
                    boolean++;
                }
                else if (DateTime.TryParse(text, CultureInfo.InvariantCulture, DateTimeStyles.None, out _))
                {
                    dates++;
                }
            }

            Columns[c] = Columns[c] with
            {
                DataType = seen == 0 ? ColumnDataType.Unknown
                    : numeric >= seen * 0.9 ? ColumnDataType.Numeric
                    : dates >= seen * 0.9 ? ColumnDataType.DateTime
                    : boolean >= seen * 0.9 ? ColumnDataType.Boolean
                    : longText || distinct.Count > seen * 0.7 ? ColumnDataType.Text
                    : ColumnDataType.Categorical
            };
        }
    }

    /// <summary>Full column statistics, used by the dataset page and the agent's data questions.</summary>
    public DatasetProfile Profile(int histogramBins = 12, int topValues = 8)
    {
        InferTypes();
        var profile = new DatasetProfile { RowCount = RowCount, SampleRows = Preview(20) };

        for (var c = 0; c < Columns.Count; c++)
        {
            var column = Columns[c];
            var stat = new ColumnProfile { Name = column.Name, DataType = column.DataType };
            var numbers = new List<double>();
            var counts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

            foreach (var row in Rows)
            {
                var value = c < row.Length ? row[c] : null;
                if (IsBlank(value))
                {
                    stat.MissingCount++;
                    continue;
                }

                var text = Convert.ToString(value, CultureInfo.InvariantCulture) ?? string.Empty;
                counts[text] = counts.GetValueOrDefault(text) + 1;

                var number = ToNumber(value);
                if (number.HasValue)
                {
                    numbers.Add(number.Value);
                }
            }

            stat.DistinctCount = counts.Count;
            stat.TopValues = counts.OrderByDescending(kv => kv.Value).Take(topValues)
                .Select(kv => new CategoryCount(kv.Key, kv.Value)).ToList();

            if (column.DataType == ColumnDataType.Numeric && numbers.Count > 0)
            {
                numbers.Sort();
                stat.Min = numbers[0];
                stat.Max = numbers[^1];
                stat.Mean = numbers.Average();
                stat.Median = numbers[numbers.Count / 2];

                var mean = stat.Mean.Value;
                stat.StdDev = numbers.Count > 1
                    ? Math.Sqrt(numbers.Sum(n => (n - mean) * (n - mean)) / (numbers.Count - 1))
                    : 0;

                stat.Histogram = BuildHistogram(numbers, histogramBins);
            }

            profile.Columns.Add(stat);
        }

        return profile;
    }

    private static List<HistogramBin> BuildHistogram(List<double> sorted, int bins)
    {
        var result = new List<HistogramBin>();
        var min = sorted[0];
        var max = sorted[^1];

        if (Math.Abs(max - min) < double.Epsilon)
        {
            result.Add(new HistogramBin(min, max, sorted.Count));
            return result;
        }

        var width = (max - min) / bins;
        var counters = new int[bins];

        foreach (var value in sorted)
        {
            var index = (int)((value - min) / width);
            counters[Math.Clamp(index, 0, bins - 1)]++;
        }

        for (var b = 0; b < bins; b++)
        {
            result.Add(new HistogramBin(min + b * width, min + (b + 1) * width, counters[b]));
        }

        return result;
    }
}

public readonly record struct DataColumn(string Name, ColumnDataType DataType);
