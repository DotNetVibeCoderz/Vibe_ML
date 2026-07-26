using System.Globalization;
using System.Text;
using System.Text.Json;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using CsvHelper;
using CsvHelper.Configuration;

namespace BlazorML.Infrastructure.Datasets;

/// <summary>Reads and writes <see cref="TabularData"/> in the formats the importer accepts.</summary>
public static class TabularSerializer
{
    public static async Task<TabularData> ReadAsync(Stream stream, DatasetFormat format,
        int rowLimit = 0, CancellationToken ct = default)
    {
        return format switch
        {
            DatasetFormat.Json => await ReadJsonAsync(stream, rowLimit, ct),
            DatasetFormat.Tsv => await ReadDelimitedAsync(stream, '\t', rowLimit, ct),
            DatasetFormat.Parquet => await ParquetSerializer.ReadAsync(stream, rowLimit, ct),
            _ => await ReadDelimitedAsync(stream, ',', rowLimit, ct)
        };
    }

    public static async Task<Stream> WriteAsync(TabularData data, DatasetFormat format, CancellationToken ct = default)
    {
        if (format == DatasetFormat.Parquet)
        {
            return await ParquetSerializer.WriteAsync(data, ct);
        }

        var buffer = new MemoryStream();

        if (format == DatasetFormat.Json)
        {
            await JsonSerializer.SerializeAsync(buffer, data.ToDictionaries(), cancellationToken: ct);
        }
        else
        {
            var delimiter = format == DatasetFormat.Tsv ? '\t' : ',';
            var bytes = Encoding.UTF8.GetBytes(data.ToCsv(delimiter));
            await buffer.WriteAsync(bytes, ct);
        }

        buffer.Position = 0;
        return buffer;
    }

    private static async Task<TabularData> ReadDelimitedAsync(Stream stream, char delimiter,
        int rowLimit, CancellationToken ct)
    {
        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            Delimiter = delimiter.ToString(),
            DetectColumnCountChanges = false,
            MissingFieldFound = null,
            BadDataFound = null,
            TrimOptions = TrimOptions.Trim
        };

        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, leaveOpen: true);
        using var csv = new CsvReader(reader, config);

        var table = new TabularData();

        if (!await csv.ReadAsync())
        {
            return table;
        }

        csv.ReadHeader();
        var header = csv.HeaderRecord ?? [];
        foreach (var name in header)
        {
            table.AddColumn(string.IsNullOrWhiteSpace(name) ? $"column_{table.ColumnCount + 1}" : name);
        }

        while (await csv.ReadAsync())
        {
            ct.ThrowIfCancellationRequested();

            var row = table.NewRow();
            for (var c = 0; c < table.ColumnCount; c++)
            {
                row[c] = csv.TryGetField<string>(c, out var value) ? value : null;
            }

            table.Rows.Add(row);

            if (rowLimit > 0 && table.RowCount >= rowLimit)
            {
                break;
            }
        }

        table.InferTypes();
        return table;
    }

    private static async Task<TabularData> ReadJsonAsync(Stream stream, int rowLimit, CancellationToken ct)
    {
        // A row limit is only worth having if it stops the reading, and JsonDocument.ParseAsync
        // pulls the whole file in before the first row is looked at. A bare array — what an export
        // almost always is — can be streamed element by element instead.
        if (rowLimit > 0 && await LooksLikeArrayAsync(stream, ct))
        {
            return await ReadJsonArrayAsync(stream, rowLimit, ct);
        }

        using var document = await JsonDocument.ParseAsync(stream, cancellationToken: ct);
        var root = document.RootElement;

        // Accept either a bare array or the common { "data": [...] } / { "items": [...] } wrappers.
        if (root.ValueKind == JsonValueKind.Object)
        {
            foreach (var name in (string[])["data", "items", "results", "rows", "value"])
            {
                if (root.TryGetProperty(name, out var wrapped) && wrapped.ValueKind == JsonValueKind.Array)
                {
                    root = wrapped;
                    break;
                }
            }
        }

        if (root.ValueKind != JsonValueKind.Array)
        {
            throw new InvalidDataException(
                "The JSON must be an array of objects, or an object with a data/items/results array.");
        }

        var table = new TabularData();
        foreach (var element in root.EnumerateArray())
        {
            ct.ThrowIfCancellationRequested();

            if (AppendJsonRow(table, element) && rowLimit > 0 && table.RowCount >= rowLimit)
            {
                break;
            }
        }

        table.InferTypes();
        return table;
    }

    /// <summary>
    /// Streams a top-level JSON array, stopping once <paramref name="rowLimit"/> rows are in hand.
    /// </summary>
    private static async Task<TabularData> ReadJsonArrayAsync(Stream stream, int rowLimit,
        CancellationToken ct)
    {
        var table = new TabularData();

        await foreach (var element in JsonSerializer
                           .DeserializeAsyncEnumerable<JsonElement>(stream, cancellationToken: ct))
        {
            if (AppendJsonRow(table, element) && table.RowCount >= rowLimit)
            {
                break;
            }
        }

        table.InferTypes();
        return table;
    }

    /// <summary>
    /// Adds one element as a row, widening the table for any column it introduces. Returns false
    /// for anything that is not an object — a stray scalar in the array is skipped, not counted.
    /// </summary>
    private static bool AppendJsonRow(TabularData table, JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            return false;
        }

        foreach (var property in element.EnumerateObject())
        {
            if (!table.Has(property.Name))
            {
                table.AddColumn(property.Name);
            }
        }

        var row = table.NewRow();
        foreach (var property in element.EnumerateObject())
        {
            var index = table.IndexOf(property.Name);
            if (index >= 0)
            {
                row[index] = ReadScalar(property.Value);
            }
        }

        table.Rows.Add(row);
        return true;
    }

    /// <summary>
    /// Peeks at the first meaningful character to tell a bare array from a wrapped object, and
    /// rewinds.
    /// <para>
    /// Only attempted when the stream can seek. A wrapped object cannot be streamed anyway — its
    /// array may be the last property — so answering "no" here simply keeps the original path,
    /// which is also the right answer for a stream that cannot be rewound.
    /// </para>
    /// </summary>
    private static async Task<bool> LooksLikeArrayAsync(Stream stream, CancellationToken ct)
    {
        if (!stream.CanSeek)
        {
            return false;
        }

        var origin = stream.Position;
        var buffer = new byte[1];

        try
        {
            while (await stream.ReadAsync(buffer.AsMemory(), ct) == 1)
            {
                // Skip whitespace and a UTF-8 byte order mark.
                if (buffer[0] is (byte)' ' or (byte)'\t' or (byte)'\r' or (byte)'\n'
                    or 0xEF or 0xBB or 0xBF)
                {
                    continue;
                }

                return buffer[0] == (byte)'[';
            }

            return false;
        }
        finally
        {
            stream.Position = origin;
        }
    }

    private static object? ReadScalar(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.String => element.GetString(),
        JsonValueKind.Number => element.TryGetDouble(out var d) ? d : element.GetRawText(),
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        JsonValueKind.Null or JsonValueKind.Undefined => null,
        // Nested objects and arrays are kept as their JSON text so nothing is silently dropped.
        _ => element.GetRawText()
    };

    public static DatasetFormat GuessFormat(string fileName) =>
        Path.GetExtension(fileName).ToLowerInvariant() switch
        {
            ".tsv" or ".tab" => DatasetFormat.Tsv,
            ".json" => DatasetFormat.Json,
            ".parquet" => DatasetFormat.Parquet,
            _ => DatasetFormat.Csv
        };
}
