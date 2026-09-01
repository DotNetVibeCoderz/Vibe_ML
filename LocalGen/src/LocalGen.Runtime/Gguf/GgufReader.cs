using System.Text;

namespace LocalGen.Runtime.Gguf;

/// <summary>
/// Reads the metadata header of a GGUF file.
/// </summary>
/// <remarks>
/// The model browser needs context length, architecture and quantization for files that can be
/// tens of gigabytes, so loading them through llama.cpp just to read a few fields is not viable.
/// GGUF puts all of that in a header at the start of the file, which is what this parses.
/// Only the key/value block is read; tensor data is never touched.
/// </remarks>
public static class GgufReader
{
    private const uint Magic = 0x46554747; // "GGUF" little-endian

    /// <summary>Guards against a corrupt length field causing a huge allocation.</summary>
    private const int MaxStringLength = 64 * 1024 * 1024;

    /// <summary>
    /// Reads metadata from a GGUF file, or returns null when the file is not GGUF or is truncated.
    /// </summary>
    public static async Task<GgufMetadata?> TryReadAsync(
        string path,
        CancellationToken cancellationToken = default)
    {
        try
        {
            await using var stream = new FileStream(
                path,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                bufferSize: 1 << 16,
                FileOptions.SequentialScan | FileOptions.Asynchronous);

            return await Task.Run(() => Read(stream), cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or EndOfStreamException)
        {
            return null;
        }
    }

    private static GgufMetadata? Read(Stream stream)
    {
        var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        if (reader.ReadUInt32() != Magic)
        {
            return null;
        }

        var version = reader.ReadUInt32();

        // v1 laid out counts as 32-bit; every model in circulation today is v2 or v3.
        if (version < 2)
        {
            return null;
        }

        var tensorCount = (long)reader.ReadUInt64();
        var kvCount = (long)reader.ReadUInt64();

        if (kvCount is < 0 or > 100_000)
        {
            return null;
        }

        var values = new Dictionary<string, string>(StringComparer.Ordinal);

        for (long i = 0; i < kvCount; i++)
        {
            var key = ReadString(reader);
            var type = (GgufType)reader.ReadUInt32();
            values[key] = ReadValue(reader, type);
        }

        var architecture = values.GetValueOrDefault("general.architecture", string.Empty);

        return new GgufMetadata
        {
            Version = version,
            TensorCount = (int)tensorCount,
            Architecture = architecture,
            Name = values.GetValueOrDefault("general.name", string.Empty),
            SizeLabel = values.GetValueOrDefault("general.size_label", string.Empty),
            ChatTemplate = values.GetValueOrDefault("tokenizer.chat_template"),
            ContextLength = ReadInt(values, $"{architecture}.context_length"),
            EmbeddingLength = ReadInt(values, $"{architecture}.embedding_length"),
            BlockCount = ReadInt(values, $"{architecture}.block_count"),
            ParameterCount = ReadLong(values, "general.parameter_count"),
            QuantizationType = DescribeFileType(values.GetValueOrDefault("general.file_type")),
            All = values
        };
    }

    private static string ReadString(BinaryReader reader)
    {
        var length = (long)reader.ReadUInt64();
        if (length is < 0 or > MaxStringLength)
        {
            throw new InvalidDataException($"GGUF string length {length} is out of range.");
        }

        return Encoding.UTF8.GetString(reader.ReadBytes((int)length));
    }

    private static string ReadValue(BinaryReader reader, GgufType type) => type switch
    {
        GgufType.UInt8 => reader.ReadByte().ToString(),
        GgufType.Int8 => reader.ReadSByte().ToString(),
        GgufType.UInt16 => reader.ReadUInt16().ToString(),
        GgufType.Int16 => reader.ReadInt16().ToString(),
        GgufType.UInt32 => reader.ReadUInt32().ToString(),
        GgufType.Int32 => reader.ReadInt32().ToString(),
        GgufType.Float32 => reader.ReadSingle().ToString("R"),
        GgufType.Bool => (reader.ReadByte() != 0).ToString(),
        GgufType.String => ReadString(reader),
        GgufType.UInt64 => reader.ReadUInt64().ToString(),
        GgufType.Int64 => reader.ReadInt64().ToString(),
        GgufType.Float64 => reader.ReadDouble().ToString("R"),
        GgufType.Array => ReadArray(reader),
        _ => throw new InvalidDataException($"Unknown GGUF value type {(int)type}.")
    };

    /// <summary>
    /// Reads an array value. Token vocabularies are stored as arrays with hundreds of thousands
    /// of entries, so the contents are skipped and only a summary is kept.
    /// </summary>
    private static string ReadArray(BinaryReader reader)
    {
        var elementType = (GgufType)reader.ReadUInt32();
        var count = (long)reader.ReadUInt64();

        const int previewLimit = 8;
        var preview = new List<string>(previewLimit);

        for (long i = 0; i < count; i++)
        {
            var value = ReadValue(reader, elementType);
            if (i < previewLimit)
            {
                preview.Add(value);
            }
        }

        return count <= previewLimit
            ? $"[{string.Join(", ", preview)}]"
            : $"[{string.Join(", ", preview)}, … {count} items]";
    }

    private static int ReadInt(IReadOnlyDictionary<string, string> values, string key) =>
        values.TryGetValue(key, out var raw) && int.TryParse(raw, out var parsed) ? parsed : 0;

    private static long ReadLong(IReadOnlyDictionary<string, string> values, string key) =>
        values.TryGetValue(key, out var raw) && long.TryParse(raw, out var parsed) ? parsed : 0;

    /// <summary>
    /// Maps llama.cpp's <c>general.file_type</c> enum to the quantization name people recognise
    /// from file names. Values follow <c>llama_ftype</c> in llama.cpp.
    /// </summary>
    private static string DescribeFileType(string? raw) =>
        int.TryParse(raw, out var fileType)
            ? fileType switch
            {
                0 => "F32",
                1 => "F16",
                2 => "Q4_0",
                3 => "Q4_1",
                7 => "Q8_0",
                8 => "Q5_0",
                9 => "Q5_1",
                10 => "Q2_K",
                11 => "Q3_K_S",
                12 => "Q3_K_M",
                13 => "Q3_K_L",
                14 => "Q4_K_S",
                15 => "Q4_K_M",
                16 => "Q5_K_S",
                17 => "Q5_K_M",
                18 => "Q6_K",
                30 => "BF16",
                _ => $"type_{fileType}"
            }
            : string.Empty;

    private enum GgufType : uint
    {
        UInt8 = 0,
        Int8 = 1,
        UInt16 = 2,
        Int16 = 3,
        UInt32 = 4,
        Int32 = 5,
        Float32 = 6,
        Bool = 7,
        String = 8,
        Array = 9,
        UInt64 = 10,
        Int64 = 11,
        Float64 = 12
    }
}
