using System.Buffers.Binary;
using System.Globalization;
using System.Text.Json;

namespace Gravicode.Science.DaskNet.Lakehouse;

/// <summary>
/// Delta Lake deletion vectors: which row positions of a data file are deleted.
/// </summary>
/// <remarks>
/// <para>
/// A vector is stored inline in the log (<c>i</c>, Z85 text), in a file next to the table named by
/// a Z85-encoded UUID with an optional directory prefix (<c>u</c>), or at an absolute path
/// (<c>p</c>). A stored vector sits at <c>offset</c> in its file as a big-endian length, the
/// bitmap, and a CRC-32 of it. The bitmap is a <em>RoaringBitmapArray</em>: a magic number, a
/// 64-bit bucket count, then per bucket a 32-bit key (the high half of the row index) and a
/// standard portable 32-bit Roaring bitmap (array, bitmap and run containers).
/// </para>
/// <para>
/// A descriptor crosses the wire to the worker as text - <c>file\n{path}\n{offset}\n{size}</c> or
/// <c>inline\n{z85}\n{size}</c> - and is decoded there, so a large vector is never sent.
/// </para>
/// </remarks>
internal static class DeletionVectors
{
    private const int RoaringArrayMagic = 1681511377;
    private const string Z85Alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#";
    private static readonly sbyte[] Z85Index = BuildIndex();

    /// <summary>The wire descriptor of a <c>deletionVector</c> log entry, paths resolved against <paramref name="root"/>.</summary>
    public static string Describe(string storageType, string pathOrInline, int? offset, int size, string root)
    {
        switch (storageType)
        {
            case "i":
                return $"inline\n{pathOrInline}\n{size.ToString(CultureInfo.InvariantCulture)}";
            case "u":
            {
                if (pathOrInline.Length < 20) throw new InvalidDataException($"Delta table at {root}: deletion vector id '{pathOrInline}' is too short.");
                var prefix = pathOrInline[..^20];
                var uuid = new Guid(Z85Decode(pathOrInline[^20..]), bigEndian: true);
                var file = Path.Combine(root, prefix, $"deletion_vector_{uuid:D}.bin");
                return Stored(file, offset, size);
            }
            case "p":
            {
                var file = Uri.TryCreate(pathOrInline, UriKind.Absolute, out var uri) && uri.IsFile ? uri.LocalPath : pathOrInline;
                return Stored(file, offset, size);
            }
            default:
                throw new NotSupportedException($"Delta table at {root}: deletion vector storage type '{storageType}' is not supported.");
        }
    }

    private static string Stored(string file, int? offset, int size)
        => $"file\n{Path.GetFullPath(file)}\n{(offset ?? 1).ToString(CultureInfo.InvariantCulture)}\n{size.ToString(CultureInfo.InvariantCulture)}";

    /// <summary>The descriptor of a log entry's <c>deletionVector</c> object, or null when there is none.</summary>
    public static string? FromJson(JsonElement add, string root)
    {
        if (!add.TryGetProperty("deletionVector", out var dv) || dv.ValueKind != JsonValueKind.Object) return null;
        return Describe(dv.GetProperty("storageType").GetString()!, dv.GetProperty("pathOrInlineDv").GetString()!,
            dv.TryGetProperty("offset", out var o) && o.ValueKind == JsonValueKind.Number ? o.GetInt32() : null,
            dv.GetProperty("sizeInBytes").GetInt32(), root);
    }

    /// <summary>The deleted row positions a descriptor names, ascending.</summary>
    public static long[] Read(string descriptor)
    {
        var parts = descriptor.Split('\n');
        byte[] bitmap;
        if (parts[0] == "inline")
        {
            var size = int.Parse(parts[2], CultureInfo.InvariantCulture);
            var bytes = Z85Decode(parts[1]);
            if (bytes.Length < size) throw new InvalidDataException($"An inline deletion vector holds {bytes.Length} bytes; the log says {size}.");
            bitmap = bytes[..size];
        }
        else if (parts[0] == "file")
        {
            var offset = long.Parse(parts[2], CultureInfo.InvariantCulture);
            var size = int.Parse(parts[3], CultureInfo.InvariantCulture);
            using var stream = File.OpenRead(parts[1]);
            stream.Position = offset;
            Span<byte> header = stackalloc byte[4];
            stream.ReadExactly(header);
            var stored = BinaryPrimitives.ReadInt32BigEndian(header);
            if (stored != size) throw new InvalidDataException($"{parts[1]}: the deletion vector at offset {offset} is {stored} bytes; the log says {size}.");
            bitmap = new byte[size];
            stream.ReadExactly(bitmap);
            Span<byte> crc = stackalloc byte[4];
            if (stream.Read(crc) == 4 && BinaryPrimitives.ReadUInt32BigEndian(crc) != Crc32(bitmap))
                throw new InvalidDataException($"{parts[1]}: the deletion vector at offset {offset} fails its checksum.");
        }
        else throw new InvalidDataException($"Unknown deletion vector descriptor '{parts[0]}'.");
        return DecodeRoaringArray(bitmap);
    }

    /// <summary>Decodes a RoaringBitmapArray (the 64-bit portable format Delta writes).</summary>
    internal static long[] DecodeRoaringArray(ReadOnlySpan<byte> data)
    {
        if (data.Length < 12 || BinaryPrimitives.ReadInt32LittleEndian(data) != RoaringArrayMagic)
            throw new InvalidDataException("Not a Delta deletion vector: the RoaringBitmapArray magic number is missing.");
        var buckets = BinaryPrimitives.ReadInt64LittleEndian(data[4..]);
        var pos = 12;
        var values = new List<long>();
        for (long b = 0; b < buckets; b++)
        {
            var high = (long)BinaryPrimitives.ReadUInt32LittleEndian(data[pos..]) << 32;
            pos += 4;
            pos += DecodeRoaring32(data[pos..], v => values.Add(high | v));
        }
        return values.ToArray();
    }

    /// <summary>Decodes one portable 32-bit Roaring bitmap, returning the bytes it used.</summary>
    internal static int DecodeRoaring32(ReadOnlySpan<byte> data, Action<uint> emit)
    {
        const int SerialCookieNoRun = 12346, SerialCookie = 12347, NoOffsetThreshold = 4;
        var pos = 0;
        var cookie = BinaryPrimitives.ReadUInt32LittleEndian(data);
        pos += 4;
        int containers;
        ReadOnlySpan<byte> runFlags = default;
        var hasRun = (cookie & 0xFFFF) == SerialCookie;
        if (hasRun)
        {
            containers = (int)(cookie >> 16) + 1;
            var flagBytes = (containers + 7) / 8;
            runFlags = data.Slice(pos, flagBytes);
            pos += flagBytes;
        }
        else if (cookie == SerialCookieNoRun)
        {
            containers = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(data[pos..]));
            pos += 4;
        }
        else throw new InvalidDataException($"Not a portable Roaring bitmap (cookie {cookie}).");

        var keys = new ushort[containers];
        var cards = new int[containers];
        for (var i = 0; i < containers; i++)
        {
            keys[i] = BinaryPrimitives.ReadUInt16LittleEndian(data[pos..]);
            cards[i] = BinaryPrimitives.ReadUInt16LittleEndian(data[(pos + 2)..]) + 1;
            pos += 4;
        }
        if (!hasRun || containers >= NoOffsetThreshold) pos += 4 * containers;   // offsets: the containers follow in order anyway

        for (var i = 0; i < containers; i++)
        {
            var key = (uint)keys[i] << 16;
            if (hasRun && (runFlags[i / 8] & (1 << (i % 8))) != 0)
            {
                var runs = BinaryPrimitives.ReadUInt16LittleEndian(data[pos..]);
                pos += 2;
                for (var r = 0; r < runs; r++)
                {
                    var start = BinaryPrimitives.ReadUInt16LittleEndian(data[pos..]);
                    var length = BinaryPrimitives.ReadUInt16LittleEndian(data[(pos + 2)..]);
                    pos += 4;
                    for (var v = 0; v <= length; v++) emit(key | (uint)(start + v));
                }
            }
            else if (cards[i] <= 4096)
            {
                for (var k = 0; k < cards[i]; k++) emit(key | BinaryPrimitives.ReadUInt16LittleEndian(data[(pos + 2 * k)..]));
                pos += 2 * cards[i];
            }
            else
            {
                for (var w = 0; w < 1024; w++)
                {
                    var word = BinaryPrimitives.ReadUInt64LittleEndian(data[(pos + 8 * w)..]);
                    while (word != 0)
                    {
                        var bit = System.Numerics.BitOperations.TrailingZeroCount(word);
                        emit(key | (uint)(w * 64 + bit));
                        word &= word - 1;
                    }
                }
                pos += 8192;
            }
        }
        return pos;
    }

    /// <summary>Z85 (ZeroMQ base-85): five characters to four big-endian bytes. Delta pads the
    /// last group to a whole one, so the output can be longer than the payload.</summary>
    internal static byte[] Z85Decode(string text)
    {
        if (text.Length % 5 != 0) throw new InvalidDataException($"Z85 text must be a multiple of 5 characters; '{text}' is {text.Length}.");
        var output = new byte[text.Length / 5 * 4];
        for (int i = 0, o = 0; i < text.Length; i += 5, o += 4)
        {
            uint value = 0;
            for (var k = 0; k < 5; k++)
            {
                var c = text[i + k];
                var digit = c < 128 ? Z85Index[c] : -1;
                if (digit < 0) throw new InvalidDataException($"'{c}' is not a Z85 character.");
                value = checked(value * 85 + (uint)digit);
            }
            BinaryPrimitives.WriteUInt32BigEndian(output.AsSpan(o), value);
        }
        return output;
    }

    private static sbyte[] BuildIndex()
    {
        var index = new sbyte[128];
        Array.Fill(index, (sbyte)-1);
        for (var i = 0; i < Z85Alphabet.Length; i++) index[Z85Alphabet[i]] = (sbyte)i;
        return index;
    }

    private static uint Crc32(ReadOnlySpan<byte> data)
    {
        var crc = 0xFFFFFFFFu;
        foreach (var b in data)
        {
            crc ^= b;
            for (var k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & (uint)-(int)(crc & 1));
        }
        return ~crc;
    }
}
