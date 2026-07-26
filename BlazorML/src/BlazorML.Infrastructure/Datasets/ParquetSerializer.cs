using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using Parquet;
using Parquet.Schema;
using ParquetDataColumn = Parquet.Data.DataColumn;

namespace BlazorML.Infrastructure.Datasets;

/// <summary>
/// Reads and writes Parquet. Columnar on disk and row-oriented in memory, so both directions
/// transpose — which is also why Parquet files are so much smaller than the equivalent CSV.
/// </summary>
internal static class ParquetSerializer
{
    public static async Task<TabularData> ReadAsync(Stream stream, int rowLimit, CancellationToken ct)
    {
        // Parquet needs random access to read its footer; a response or upload stream may not
        // support seeking, so buffer first.
        var seekable = stream;
        MemoryStream? buffer = null;

        if (!stream.CanSeek)
        {
            buffer = new MemoryStream();
            await stream.CopyToAsync(buffer, ct);
            buffer.Position = 0;
            seekable = buffer;
        }

        try
        {
            // A table with no columns has no schema to write, so it round-trips as an empty
            // stream. Reading that back would otherwise surface as "not a Parquet file", which
            // says nothing useful about a dataset that is simply empty.
            if (seekable.CanSeek && seekable.Length < 8)
            {
                return new TabularData();
            }

            using var reader = await ParquetReader.CreateAsync(seekable, cancellationToken: ct);

            var fields = reader.Schema.GetDataFields();
            var table = new TabularData();

            foreach (var field in fields)
            {
                table.AddColumn(field.Name, MapType(field.ClrType));
            }

            for (var group = 0; group < reader.RowGroupCount; group++)
            {
                ct.ThrowIfCancellationRequested();

                using var rowGroup = reader.OpenRowGroupReader(group);

                // A row group arrives one column at a time; assemble the rows once all the
                // columns for this group are in hand.
                var columns = new ParquetDataColumn[fields.Length];

                for (var c = 0; c < fields.Length; c++)
                {
                    columns[c] = await rowGroup.ReadColumnAsync(fields[c], ct);
                }

                var rowsInGroup = columns.Length == 0 ? 0 : columns[0].Data.Length;

                for (var r = 0; r < rowsInGroup; r++)
                {
                    var row = table.NewRow();

                    for (var c = 0; c < columns.Length; c++)
                    {
                        row[c] = r < columns[c].Data.Length ? columns[c].Data.GetValue(r) : null;
                    }

                    table.Rows.Add(row);

                    if (rowLimit > 0 && table.RowCount >= rowLimit)
                    {
                        table.InferTypes();
                        return table;
                    }
                }
            }

            table.InferTypes();
            return table;
        }
        finally
        {
            buffer?.Dispose();
        }
    }

    public static async Task<Stream> WriteAsync(TabularData data, CancellationToken ct)
    {
        data.InferTypes();

        var output = new MemoryStream();

        // Parquet has no representation for a file with no columns; the reader above recognises
        // the empty stream and gives back an empty table.
        if (data.ColumnCount == 0)
        {
            return output;
        }

        var fields = data.Columns.Select(BuildField).ToArray();
        var schema = new ParquetSchema(fields.Cast<Field>().ToArray());

        using (var writer = await ParquetWriter.CreateAsync(schema, output, cancellationToken: ct))
        {
            // Written as one row group: the datasets this app holds are already capped to what
            // fits in memory, so splitting would add complexity for no gain.
            using var rowGroup = writer.CreateRowGroup();

            for (var c = 0; c < fields.Length; c++)
            {
                ct.ThrowIfCancellationRequested();

                var values = BuildColumn(data, c, data.Columns[c].DataType);
                await rowGroup.WriteColumnAsync(new ParquetDataColumn(fields[c], values), ct);
            }
        }

        output.Position = 0;
        return output;
    }

    /// <summary>
    /// Every column is written nullable. A column that happens to have no gaps today can gain
    /// them after a transform, and a non-nullable Parquet column cannot represent that.
    /// </summary>
    private static DataField BuildField(DataColumn column) => column.DataType switch
    {
        ColumnDataType.Numeric => new DataField<double?>(column.Name),
        ColumnDataType.Boolean => new DataField<bool?>(column.Name),
        ColumnDataType.DateTime => new DataField<DateTime?>(column.Name),
        _ => new DataField<string?>(column.Name)
    };

    private static Array BuildColumn(TabularData data, int index, ColumnDataType type)
    {
        switch (type)
        {
            case ColumnDataType.Numeric:
            {
                var values = new double?[data.RowCount];
                for (var r = 0; r < data.RowCount; r++)
                {
                    values[r] = TabularData.ToNumber(Cell(data, r, index));
                }

                return values;
            }

            case ColumnDataType.Boolean:
            {
                var values = new bool?[data.RowCount];
                for (var r = 0; r < data.RowCount; r++)
                {
                    var cell = Cell(data, r, index);
                    values[r] = TabularData.IsBlank(cell)
                        ? null
                        : cell as bool? ?? bool.TryParse(Convert.ToString(cell), out var parsed) && parsed;
                }

                return values;
            }

            case ColumnDataType.DateTime:
            {
                var values = new DateTime?[data.RowCount];
                for (var r = 0; r < data.RowCount; r++)
                {
                    var cell = Cell(data, r, index);
                    values[r] = cell switch
                    {
                        null => null,
                        DateTime dt => dt,
                        _ => DateTime.TryParse(Convert.ToString(cell),
                            System.Globalization.CultureInfo.InvariantCulture,
                            System.Globalization.DateTimeStyles.None, out var parsed) ? parsed : null
                    };
                }

                return values;
            }

            default:
            {
                var values = new string?[data.RowCount];
                for (var r = 0; r < data.RowCount; r++)
                {
                    var cell = Cell(data, r, index);
                    values[r] = TabularData.IsBlank(cell)
                        ? null
                        : Convert.ToString(cell, System.Globalization.CultureInfo.InvariantCulture);
                }

                return values;
            }
        }
    }

    private static object? Cell(TabularData data, int row, int column) =>
        column < data.Rows[row].Length ? data.Rows[row][column] : null;

    private static ColumnDataType MapType(Type clrType)
    {
        var underlying = Nullable.GetUnderlyingType(clrType) ?? clrType;

        if (underlying == typeof(bool))
        {
            return ColumnDataType.Boolean;
        }

        if (underlying == typeof(DateTime) || underlying == typeof(DateTimeOffset))
        {
            return ColumnDataType.DateTime;
        }

        return underlying == typeof(string) ? ColumnDataType.Text : ColumnDataType.Numeric;
    }
}
