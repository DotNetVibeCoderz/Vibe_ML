using Parquet;
using Parquet.Data;
using Parquet.Schema;

namespace Gravicode.Science.GraviFrame.Io;

/// <summary>
/// Apache Parquet support for <see cref="DataFrame"/>.
/// </summary>
/// <remarks>
/// Parquet is columnar on disk, which lines up exactly with how a frame is stored in memory:
/// each <see cref="Series"/> maps to one Parquet column, so a round trip preserves types without
/// re-inference and reading a subset of columns never touches the rest of the file. That makes it
/// the right format for anything larger than a demo CSV.
/// </remarks>
public static class ParquetIO
{
    /// <summary>Writes the frame to a Parquet file.</summary>
    public static async Task WriteAsync(DataFrame frame, string path, CancellationToken cancellationToken = default)
    {
        var fields = frame.Columns.Select(ToField).ToArray();
        var schema = new ParquetSchema(fields);

        await using var stream = File.Create(path);
        using var writer = await ParquetWriter.CreateAsync(schema, stream, cancellationToken: cancellationToken);
        using var group = writer.CreateRowGroup();

        for (var i = 0; i < frame.ColumnCount; i++)
            await group.WriteColumnAsync(new DataColumn(fields[i], ToArray(frame.Columns[i])), cancellationToken);
    }

    /// <summary>Writes the frame to a Parquet file, blocking until done.</summary>
    public static void Write(DataFrame frame, string path) => WriteAsync(frame, path).GetAwaiter().GetResult();

    /// <summary>Reads a Parquet file into a frame.</summary>
    public static async Task<DataFrame> ReadAsync(string path, IReadOnlyList<string>? columns = null,
        CancellationToken cancellationToken = default)
    {
        await using var stream = File.OpenRead(path);
        using var reader = await ParquetReader.CreateAsync(stream, cancellationToken: cancellationToken);

        var wanted = reader.Schema.GetDataFields()
            .Where(f => columns is null || columns.Contains(f.Name))
            .ToArray();

        // Row groups are read in order and concatenated, so a file written by any tool works.
        var pieces = wanted.ToDictionary(f => f.Name, _ => new List<Series>(), StringComparer.Ordinal);

        for (var g = 0; g < reader.RowGroupCount; g++)
        {
            using var group = reader.OpenRowGroupReader(g);
            foreach (var field in wanted)
            {
                var column = await group.ReadColumnAsync(field, cancellationToken);
                pieces[field.Name].Add(FromArray(field.Name, column.Data));
            }
        }

        var result = wanted.Select(f => pieces[f.Name].Count == 1
            ? pieces[f.Name][0]
            : SeriesOperations.Concat(pieces[f.Name]));

        return new DataFrame(result);
    }

    /// <summary>Reads a Parquet file into a frame, blocking until done.</summary>
    public static DataFrame Read(string path, IReadOnlyList<string>? columns = null)
        => ReadAsync(path, columns).GetAwaiter().GetResult();

    /// <summary>Column names and types in a Parquet file, without reading any data.</summary>
    public static async Task<IReadOnlyList<(string Name, string Type)>> SchemaAsync(string path)
    {
        await using var stream = File.OpenRead(path);
        using var reader = await ParquetReader.CreateAsync(stream);
        return reader.Schema.GetDataFields()
            .Select(f => (f.Name, f.ClrType.Name))
            .ToList();
    }

    private static DataField ToField(Series series) => series.DataType switch
    {
        DataType.Numeric => new DataField<double?>(series.Name),
        DataType.Boolean => new DataField<bool?>(series.Name),
        DataType.DateTime => new DataField<DateTime?>(series.Name),
        _ => new DataField<string>(series.Name),
    };

    private static Array ToArray(Series series)
    {
        switch (series)
        {
            case NumericSeries n:
            {
                var values = new double?[n.Length];
                for (var i = 0; i < n.Length; i++) values[i] = double.IsNaN(n[i]) ? null : n[i];
                return values;
            }
            case BooleanSeries b:
            {
                var values = new bool?[b.Length];
                for (var i = 0; i < b.Length; i++) values[i] = b[i];
                return values;
            }
            case DateTimeSeries d:
            {
                var values = new DateTime?[d.Length];
                for (var i = 0; i < d.Length; i++) values[i] = d[i];
                return values;
            }
            default:
            {
                var values = new string?[series.Length];
                for (var i = 0; i < series.Length; i++) values[i] = series.GetValue(i)?.ToString();
                return values;
            }
        }
    }

    private static Series FromArray(string name, Array data)
    {
        switch (data)
        {
            case double?[] doubles:
                return new NumericSeries(name, doubles.Select(v => v ?? double.NaN).ToArray());
            case double[] doubles:
                return new NumericSeries(name, doubles);
            case float?[] floats:
                return new NumericSeries(name, floats.Select(v => v.HasValue ? (double)v.Value : double.NaN).ToArray());
            case int?[] ints:
                return new NumericSeries(name, ints.Select(v => v.HasValue ? (double)v.Value : double.NaN).ToArray());
            case long?[] longs:
                return new NumericSeries(name, longs.Select(v => v.HasValue ? (double)v.Value : double.NaN).ToArray());
            case bool?[] booleans:
                return new BooleanSeries(name, booleans);
            case DateTime?[] dates:
                return new DateTimeSeries(name, dates);
            case DateTimeOffset?[] offsets:
                return new DateTimeSeries(name, offsets.Select(v => (DateTime?)v?.UtcDateTime).ToArray());
            case string?[] strings:
                return new TextSeries(name, strings);
            default:
            {
                // Anything else is rendered as text rather than dropped.
                var values = new string?[data.Length];
                for (var i = 0; i < data.Length; i++) values[i] = data.GetValue(i)?.ToString();
                return new TextSeries(name, values);
            }
        }
    }
}
