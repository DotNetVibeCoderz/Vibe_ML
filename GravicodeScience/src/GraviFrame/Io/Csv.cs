using System.Globalization;
using System.IO.MemoryMappedFiles;
using System.Text;

namespace Gravicode.Science.GraviFrame.Io;

/// <summary>Settings for CSV parsing.</summary>
public sealed class CsvOptions
{
    /// <summary>Field separator.</summary>
    public string Delimiter { get; set; } = ",";

    /// <summary>Whether the first line holds column names.</summary>
    public bool HasHeader { get; set; } = true;

    /// <summary>Rows sampled to infer each column's type. Zero means "use every row".</summary>
    public int TypeInferenceRows { get; set; } = 1000;

    /// <summary>Tokens treated as missing, in addition to the empty string.</summary>
    public HashSet<string> MissingTokens { get; set; } =
        new(StringComparer.OrdinalIgnoreCase) { "na", "n/a", "nan", "null", "none", "-", "?" };

    /// <summary>Explicit column types, overriding inference.</summary>
    public Dictionary<string, DataType> ColumnTypes { get; set; } = new(StringComparer.Ordinal);

    /// <summary>Stop after this many data rows; zero means "read everything".</summary>
    public int MaxRows { get; set; }

    /// <summary>Column names to use when <see cref="HasHeader"/> is false.</summary>
    public string[]? ColumnNames { get; set; }
}

/// <summary>Parses CSV into a <see cref="DataFrame"/>, inferring column types.</summary>
/// <remarks>
/// Parsing is two-phase: fields are collected as strings, then each column is inferred and
/// converted once into a typed array. Doing conversion per column rather than per cell is what
/// keeps the numeric path cheap - a whole column of doubles is one tight loop, not a switch per value.
/// </remarks>
public static class CsvReader
{
    /// <summary>Reads a CSV file from disk.</summary>
    public static DataFrame Read(string path, CsvOptions options)
    {
        using var reader = new StreamReader(path, Encoding.UTF8);
        return Build(ReadLines(reader), options);
    }

    /// <summary>Parses CSV held in a string.</summary>
    public static DataFrame Parse(string content, CsvOptions options)
    {
        using var reader = new StringReader(content);
        return Build(ReadLines(reader), options);
    }

    /// <summary>
    /// Reads a CSV file through a memory-mapped view.
    /// </summary>
    /// <remarks>
    /// The file is never loaded into a single managed string. Instead the mapping is scanned for
    /// line breaks and each line is decoded into a reusable buffer, so peak managed memory tracks
    /// the parsed frame rather than the file. For files in the gigabyte range this is the
    /// difference between working and an <c>OutOfMemoryException</c>.
    /// </remarks>
    public static DataFrame ReadMemoryMapped(string path, CsvOptions options)
    {
        var info = new FileInfo(path);
        if (info.Length == 0) return DataFrame.Empty();

        using var file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, mapName: null, 0, MemoryMappedFileAccess.Read);

        // The view size must be the exact file length. Passing 0 ("to the end") rounds the
        // mapping up to the system page size, and the trailing NUL bytes then decode as one
        // extra, bogus line.
        using var stream = file.CreateViewStream(0, info.Length, MemoryMappedFileAccess.Read);
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, bufferSize: 1 << 20);
        return Build(ReadLines(reader), options);
    }

    private static IEnumerable<string> ReadLines(TextReader reader)
    {
        string? line;
        while ((line = reader.ReadLine()) is not null) yield return line;
    }

    private static DataFrame Build(IEnumerable<string> lines, CsvOptions options)
    {
        string[]? header = null;
        var rows = new List<string[]>();

        foreach (var line in lines)
        {
            if (line.Length == 0) continue;
            var fields = SplitLine(line, options.Delimiter);

            if (header is null && options.HasHeader)
            {
                header = fields;
                continue;
            }

            rows.Add(fields);
            if (options.MaxRows > 0 && rows.Count >= options.MaxRows) break;
        }

        var columnCount = header?.Length ?? (rows.Count > 0 ? rows[0].Length : 0);
        if (columnCount == 0) return DataFrame.Empty();

        header ??= options.ColumnNames
            ?? Enumerable.Range(0, columnCount).Select(i => $"column{i}").ToArray();

        // Duplicate or blank headers would break the frame's name lookup; make them unique.
        header = MakeUnique(header);

        var columns = new List<Series>(columnCount);
        for (var j = 0; j < columnCount; j++)
        {
            var raw = new string?[rows.Count];
            for (var i = 0; i < rows.Count; i++)
            {
                var value = j < rows[i].Length ? rows[i][j] : null;
                raw[i] = IsMissing(value, options) ? null : value;
            }

            var name = j < header.Length ? header[j] : $"column{j}";
            var type = options.ColumnTypes.TryGetValue(name, out var forced) ? forced : Infer(raw, options);
            columns.Add(Convert(name, raw, type));
        }

        return new DataFrame(columns);
    }

    private static string[] MakeUnique(string[] names)
    {
        var seen = new Dictionary<string, int>(StringComparer.Ordinal);
        var result = new string[names.Length];
        for (var i = 0; i < names.Length; i++)
        {
            var name = string.IsNullOrWhiteSpace(names[i]) ? $"column{i}" : names[i].Trim();
            if (seen.TryGetValue(name, out var count))
            {
                seen[name] = count + 1;
                name = $"{name}_{count}";
            }
            else seen[name] = 1;
            result[i] = name;
        }
        return result;
    }

    private static bool IsMissing(string? value, CsvOptions options)
        => string.IsNullOrWhiteSpace(value) || options.MissingTokens.Contains(value.Trim());

    /// <summary>Splits one line, honouring double-quoted fields and doubled escape quotes.</summary>
    public static string[] SplitLine(string line, string delimiter)
    {
        if (!line.Contains('"')) return line.Split(delimiter).Select(f => f.Trim()).ToArray();

        var fields = new List<string>();
        var sb = new StringBuilder();
        var inQuotes = false;
        var i = 0;

        while (i < line.Length)
        {
            var c = line[i];

            if (inQuotes)
            {
                if (c == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"') { sb.Append('"'); i += 2; continue; }
                    inQuotes = false;
                    i++;
                    continue;
                }
                sb.Append(c);
                i++;
                continue;
            }

            if (c == '"') { inQuotes = true; i++; continue; }

            if (string.CompareOrdinal(line, i, delimiter, 0, delimiter.Length) == 0)
            {
                fields.Add(sb.ToString().Trim());
                sb.Clear();
                i += delimiter.Length;
                continue;
            }

            sb.Append(c);
            i++;
        }

        fields.Add(sb.ToString().Trim());
        return fields.ToArray();
    }

    private static DataType Infer(string?[] values, CsvOptions options)
    {
        var limit = options.TypeInferenceRows <= 0 ? values.Length : Math.Min(options.TypeInferenceRows, values.Length);

        var numeric = true;
        var boolean = true;
        var date = true;
        var seenAny = false;

        for (var i = 0; i < limit; i++)
        {
            var v = values[i];
            if (v is null) continue;
            seenAny = true;

            if (numeric && !double.TryParse(v, NumberStyles.Any, CultureInfo.InvariantCulture, out _)) numeric = false;
            if (boolean && !TryParseBoolean(v, out _)) boolean = false;
            if (date && !DateTime.TryParse(v, CultureInfo.InvariantCulture, DateTimeStyles.None, out _)) date = false;
            if (!numeric && !boolean && !date) break;
        }

        if (!seenAny) return DataType.Text;
        // Booleans are checked before numbers so a 0/1 flag column does not silently become numeric
        // only when it also contains "true"; a pure 0/1 column stays numeric, which is more useful.
        if (numeric) return DataType.Numeric;
        if (boolean) return DataType.Boolean;
        if (date) return DataType.DateTime;
        return DataType.Text;
    }

    private static bool TryParseBoolean(string value, out bool result)
    {
        switch (value.Trim().ToLowerInvariant())
        {
            case "true" or "yes" or "y" or "t": result = true; return true;
            case "false" or "no" or "n" or "f": result = false; return true;
            default: result = false; return false;
        }
    }

    private static Series Convert(string name, string?[] raw, DataType type)
    {
        switch (type)
        {
            case DataType.Numeric:
            {
                var values = new double[raw.Length];
                for (var i = 0; i < raw.Length; i++)
                    values[i] = raw[i] is null
                        ? double.NaN
                        : double.TryParse(raw[i], NumberStyles.Any, CultureInfo.InvariantCulture, out var v) ? v : double.NaN;
                return new NumericSeries(name, values);
            }
            case DataType.Boolean:
            {
                var values = new bool?[raw.Length];
                for (var i = 0; i < raw.Length; i++)
                    values[i] = raw[i] is not null && TryParseBoolean(raw[i]!, out var b) ? b : null;
                return new BooleanSeries(name, values);
            }
            case DataType.DateTime:
            {
                var values = new DateTime?[raw.Length];
                for (var i = 0; i < raw.Length; i++)
                    values[i] = raw[i] is not null
                        && DateTime.TryParse(raw[i], CultureInfo.InvariantCulture, DateTimeStyles.None, out var d)
                        ? d : null;
                return new DateTimeSeries(name, values);
            }
            default:
                return new TextSeries(name, raw);
        }
    }
}

/// <summary>Writes a <see cref="DataFrame"/> back out as CSV.</summary>
public static class CsvWriter
{
    /// <summary>Writes the frame to <paramref name="path"/>.</summary>
    public static void Write(DataFrame frame, string path, string delimiter = ",")
    {
        using var writer = new StreamWriter(path, false, Encoding.UTF8);
        Write(frame, writer, delimiter);
    }

    /// <summary>Writes the frame to any text writer.</summary>
    public static void Write(DataFrame frame, TextWriter writer, string delimiter = ",")
    {
        writer.WriteLine(string.Join(delimiter, frame.ColumnNames.Select(n => Quote(n, delimiter))));

        var sb = new StringBuilder();
        for (var i = 0; i < frame.RowCount; i++)
        {
            sb.Clear();
            for (var j = 0; j < frame.ColumnCount; j++)
            {
                if (j > 0) sb.Append(delimiter);
                var column = frame.Columns[j];
                if (column.IsMissing(i)) continue;

                var text = column is NumericSeries n
                    ? n[i].ToString("R", CultureInfo.InvariantCulture)
                    : column is DateTimeSeries d
                        ? d[i]!.Value.ToString("O", CultureInfo.InvariantCulture)
                        : column.GetValue(i)?.ToString() ?? "";
                sb.Append(Quote(text, delimiter));
            }
            writer.WriteLine(sb.ToString());
        }
    }

    /// <summary>Renders the frame as a CSV string.</summary>
    public static string ToCsv(DataFrame frame, string delimiter = ",")
    {
        using var writer = new StringWriter();
        Write(frame, writer, delimiter);
        return writer.ToString();
    }

    private static string Quote(string value, string delimiter)
        => value.Contains(delimiter) || value.Contains('"') || value.Contains('\n')
            ? '"' + value.Replace("\"", "\"\"") + '"'
            : value;
}
