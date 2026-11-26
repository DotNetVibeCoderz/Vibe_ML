using System.IO.Compression;
using System.Text;
using System.Text.Json;

namespace Gravicode.Science.DaskNet.Lakehouse;

/// <summary>
/// A reader for Avro object container files - enough for Iceberg's manifest lists and manifests.
/// </summary>
/// <remarks>
/// Records decode to dictionaries keyed by field name, arrays to lists, maps to dictionaries,
/// unions to their branch's value, <c>bytes</c>/<c>fixed</c> to byte arrays and enums to their
/// symbol. Codecs: <c>null</c> and <c>deflate</c> (raw RFC 1951, which is what Avro means by
/// deflate); anything else is refused by name rather than misread.
/// </remarks>
internal static class Avro
{
    private static readonly byte[] Magic = [(byte)'O', (byte)'b', (byte)'j', 1];

    /// <summary>The file's metadata (e.g. <c>avro.schema</c>, or Iceberg's <c>schema</c>) and records.</summary>
    public static (IReadOnlyDictionary<string, string> Metadata, List<object?> Records) Read(string path)
    {
        using var stream = File.OpenRead(path);
        var r = new BinaryReader(stream);
        if (!r.ReadBytes(4).AsSpan().SequenceEqual(Magic)) throw new InvalidDataException($"{path} is not an Avro container file.");

        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var count = ReadLong(r); count != 0; count = ReadLong(r))
        {
            if (count < 0) { count = -count; ReadLong(r); }
            for (var i = 0; i < count; i++)
                metadata[ReadString(r)] = Encoding.UTF8.GetString(ReadBytes(r));
        }
        var sync = r.ReadBytes(16);
        var schema = JsonDocument.Parse(metadata["avro.schema"]).RootElement.Clone();
        var codec = metadata.GetValueOrDefault("avro.codec", "null");
        if (codec is not ("null" or "deflate"))
            throw new NotSupportedException($"{path} uses the Avro codec '{codec}'; only null and deflate are supported.");

        var named = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        Register(schema, named);
        var records = new List<object?>();
        while (stream.Position < stream.Length)
        {
            var objects = ReadLong(r);
            var size = ReadLong(r);
            var block = r.ReadBytes(checked((int)size));
            using var data = codec == "deflate" ? Inflate(block) : new MemoryStream(block);
            var br = new BinaryReader(data);
            for (var i = 0; i < objects; i++) records.Add(ReadValue(br, schema, named));
            if (!r.ReadBytes(16).AsSpan().SequenceEqual(sync)) throw new InvalidDataException($"{path}: Avro sync marker mismatch.");
        }
        return (metadata, records);
    }

    /// <summary>A schema prepared for <see cref="Decode"/>: the parsed root and its named types.</summary>
    public sealed class Schema
    {
        internal Schema(JsonElement root)
        {
            Root = root;
            Register(root, Named);
        }

        public JsonElement Root { get; }
        internal Dictionary<string, JsonElement> Named { get; } = new(StringComparer.Ordinal);

        public static Schema Parse(string json) => new(JsonDocument.Parse(json).RootElement.Clone());
    }

    /// <summary>One value in Avro's binary encoding, without a container (Hudi log blocks store
    /// records this way, the schema travelling in the block header).</summary>
    public static object? Decode(byte[] data, int offset, int count, Schema schema)
    {
        using var stream = new MemoryStream(data, offset, count, writable: false);
        return ReadValue(new BinaryReader(stream), schema.Root, schema.Named);
    }

    private static MemoryStream Inflate(byte[] block)
    {
        var output = new MemoryStream();
        using (var inflater = new DeflateStream(new MemoryStream(block), CompressionMode.Decompress)) inflater.CopyTo(output);
        output.Position = 0;
        return output;
    }

    /// <summary>Records every named type so later references by name resolve.</summary>
    private static void Register(JsonElement schema, Dictionary<string, JsonElement> named)
    {
        switch (schema.ValueKind)
        {
            case JsonValueKind.Array:
                foreach (var branch in schema.EnumerateArray()) Register(branch, named);
                break;
            case JsonValueKind.Object:
                var type = schema.GetProperty("type");
                if (type.ValueKind != JsonValueKind.String) { Register(type, named); break; }
                if (schema.TryGetProperty("name", out var name) && type.GetString() is "record" or "enum" or "fixed")
                    named[name.GetString()!] = schema;
                if (type.GetString() == "record")
                    foreach (var f in schema.GetProperty("fields").EnumerateArray()) Register(f.GetProperty("type"), named);
                else if (type.GetString() == "array") Register(schema.GetProperty("items"), named);
                else if (type.GetString() == "map") Register(schema.GetProperty("values"), named);
                break;
        }
    }

    private static object? ReadValue(BinaryReader r, JsonElement schema, Dictionary<string, JsonElement> named)
    {
        if (schema.ValueKind == JsonValueKind.Array)
        {
            var branch = (int)ReadLong(r);
            return ReadValue(r, schema[branch], named);
        }
        if (schema.ValueKind == JsonValueKind.String)
        {
            var name = schema.GetString()!;
            return name switch
            {
                "null" => null,
                "boolean" => r.ReadByte() != 0,
                "int" or "long" => ReadLong(r),
                "float" => (double)r.ReadSingle(),
                "double" => r.ReadDouble(),
                "bytes" => ReadBytes(r),
                "string" => ReadString(r),
                _ => named.TryGetValue(name, out var type) ? ReadValue(r, type, named)
                    : throw new InvalidDataException($"Unknown Avro type '{name}'."),
            };
        }

        var kind = schema.GetProperty("type");
        if (kind.ValueKind != JsonValueKind.String) return ReadValue(r, kind, named);
        switch (kind.GetString())
        {
            case "record":
            {
                var record = new Dictionary<string, object?>(StringComparer.Ordinal);
                foreach (var f in schema.GetProperty("fields").EnumerateArray())
                    record[f.GetProperty("name").GetString()!] = ReadValue(r, f.GetProperty("type"), named);
                return record;
            }
            case "array":
            {
                var items = new List<object?>();
                var itemType = schema.GetProperty("items");
                for (var count = ReadLong(r); count != 0; count = ReadLong(r))
                {
                    if (count < 0) { count = -count; ReadLong(r); }
                    for (var i = 0; i < count; i++) items.Add(ReadValue(r, itemType, named));
                }
                return items;
            }
            case "map":
            {
                var map = new Dictionary<string, object?>(StringComparer.Ordinal);
                var valueType = schema.GetProperty("values");
                for (var count = ReadLong(r); count != 0; count = ReadLong(r))
                {
                    if (count < 0) { count = -count; ReadLong(r); }
                    for (var i = 0; i < count; i++) map[ReadString(r)] = ReadValue(r, valueType, named);
                }
                return map;
            }
            case "enum":
                return schema.GetProperty("symbols")[(int)ReadLong(r)].GetString();
            case "fixed":
                return r.ReadBytes(schema.GetProperty("size").GetInt32());
            default:
                return ReadValue(r, kind, named);   // a primitive written as {"type": "long", ...}
        }
    }

    /// <summary>Zig-zag variable-length integer (Avro's int and long share it).</summary>
    private static long ReadLong(BinaryReader r)
    {
        ulong value = 0;
        var shift = 0;
        while (true)
        {
            var b = r.ReadByte();
            value |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
            if (shift > 63) throw new InvalidDataException("Avro varint is too long.");
        }
        return (long)(value >> 1) ^ -(long)(value & 1);
    }

    private static byte[] ReadBytes(BinaryReader r) => r.ReadBytes(checked((int)ReadLong(r)));

    private static string ReadString(BinaryReader r) => Encoding.UTF8.GetString(ReadBytes(r));
}
