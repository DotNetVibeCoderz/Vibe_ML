using System.Globalization;
using System.Text;
using System.Text.Json;

namespace Gravicode.Science.GraviNum.Io;

/// <summary>
/// Reading and writing <see cref="NdArray"/> as CSV, JSON or a compact binary format.
/// </summary>
/// <remarks>
/// The binary format (<c>.gnb</c>) is the one to reach for when arrays are large: a 16-byte header
/// followed by raw little-endian doubles, which means loading is a single block read and the file
/// can also be opened as a <see cref="MemoryMappedArray"/> without parsing anything.
/// </remarks>
public static class NdIO
{
    private const uint Magic = 0x314D4E47; // "GNM1"

    /// <summary>Writes a 1-D or 2-D array as CSV.</summary>
    public static void SaveCsv(NdArray array, string path, string delimiter = ",", IReadOnlyList<string>? header = null)
    {
        using var writer = new StreamWriter(path, false, Encoding.UTF8);
        if (header is not null) writer.WriteLine(string.Join(delimiter, header));

        if (array.Rank == 1)
        {
            for (var i = 0; i < array.Size; i++)
                writer.WriteLine(array.At(i).ToString("R", CultureInfo.InvariantCulture));
            return;
        }
        if (array.Rank != 2) throw new ArgumentException("SaveCsv supports rank 1 and 2 arrays.");

        var sb = new StringBuilder();
        for (var i = 0; i < array.Shape[0]; i++)
        {
            sb.Clear();
            for (var j = 0; j < array.Shape[1]; j++)
            {
                if (j > 0) sb.Append(delimiter);
                sb.Append(array[i, j].ToString("R", CultureInfo.InvariantCulture));
            }
            writer.WriteLine(sb.ToString());
        }
    }

    /// <summary>Reads a numeric CSV file into a 2-D array.</summary>
    public static NdArray LoadCsv(string path, string delimiter = ",", bool hasHeader = false)
    {
        var rows = new List<double[]>();
        using var reader = new StreamReader(path);
        if (hasHeader) reader.ReadLine();

        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            if (line.Length == 0) continue;
            var parts = line.Split(delimiter, StringSplitOptions.TrimEntries);
            var values = new double[parts.Length];
            for (var i = 0; i < parts.Length; i++)
                values[i] = double.TryParse(parts[i], NumberStyles.Any, CultureInfo.InvariantCulture, out var v)
                    ? v
                    : double.NaN;
            rows.Add(values);
        }

        if (rows.Count == 0) return NdArray.Zeros(0, 0);
        return rows[0].Length == 1
            ? NdArray.FromValues(rows.Select(r => r[0]))
            : NdArray.FromRows(rows);
    }

    /// <summary>Reads the header row of a CSV file.</summary>
    public static string[] ReadCsvHeader(string path, string delimiter = ",")
    {
        using var reader = new StreamReader(path);
        var line = reader.ReadLine();
        return line is null ? [] : line.Split(delimiter, StringSplitOptions.TrimEntries);
    }

    /// <summary>Writes an array in the compact <c>.gnb</c> binary format.</summary>
    public static void SaveBinary(NdArray array, string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        writer.Write(Magic);
        writer.Write(array.Rank);
        for (var i = 0; i < array.Rank; i++) writer.Write(array.Shape[i]);

        var data = array.ToArray();
        var bytes = new byte[data.Length * sizeof(double)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        writer.Write(bytes);
    }

    /// <summary>Reads an array written by <see cref="SaveBinary"/>.</summary>
    public static NdArray LoadBinary(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        var magic = reader.ReadUInt32();
        if (magic != Magic) throw new InvalidDataException($"'{path}' is not a Gravicode binary array file.");

        var rank = reader.ReadInt32();
        var shape = new int[rank];
        for (var i = 0; i < rank; i++) shape[i] = reader.ReadInt32();

        var size = Shapes.Size(shape);
        var bytes = reader.ReadBytes(size * sizeof(double));
        var data = new double[size];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        return new NdArray(data, shape);
    }

    /// <summary>Serialises an array as JSON with its shape and flattened data.</summary>
    public static void SaveJson(NdArray array, string path, bool indented = false)
    {
        var payload = new ArrayPayload(array.Shape.ToArray(), array.ToArray());
        File.WriteAllText(path, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = indented }));
    }

    /// <summary>Reads an array written by <see cref="SaveJson"/>.</summary>
    public static NdArray LoadJson(string path)
    {
        var payload = JsonSerializer.Deserialize<ArrayPayload>(File.ReadAllText(path))
            ?? throw new InvalidDataException($"'{path}' does not contain an array payload.");
        return new NdArray(payload.Data, payload.Shape);
    }

    /// <summary>Serialises an array to a JSON string.</summary>
    public static string ToJson(NdArray array)
        => JsonSerializer.Serialize(new ArrayPayload(array.Shape.ToArray(), array.ToArray()));

    /// <summary>Parses an array from a JSON string produced by <see cref="ToJson"/>.</summary>
    public static NdArray FromJson(string json)
    {
        var payload = JsonSerializer.Deserialize<ArrayPayload>(json)
            ?? throw new InvalidDataException("JSON does not contain an array payload.");
        return new NdArray(payload.Data, payload.Shape);
    }

    private sealed record ArrayPayload(int[] Shape, double[] Data);
}
