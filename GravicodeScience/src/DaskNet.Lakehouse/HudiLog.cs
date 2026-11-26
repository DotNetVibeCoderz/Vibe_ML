using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet.Lakehouse;

/// <summary>
/// Hudi's log file format: a sequence of blocks, each <c>#HUDI#</c>, a length, a format version,
/// a block type, a header map, the content, a footer map and the block's total length (all
/// big-endian, as Java's DataOutputStream writes them).
/// </summary>
internal static class HudiLog
{
    private static readonly byte[] Magic = "#HUDI#"u8.ToArray();

    /// <summary>Block types, by their ordinal in Hudi's <c>HoodieLogBlockType</c>.</summary>
    internal enum BlockType { Command = 0, Delete = 1, Corrupt = 2, AvroData = 3, HFileData = 4, ParquetData = 5, CdcData = 6 }

    /// <summary>Header keys, by their ordinal in <c>HeaderMetadataType</c>.</summary>
    internal const int InstantTime = 0, TargetInstantTime = 1, SchemaKey = 2, CommandBlockType = 3;

    internal sealed record Block(BlockType Type, IReadOnlyDictionary<int, string> Header, byte[] Content)
    {
        public string? Instant => Header.GetValueOrDefault(InstantTime);
    }

    /// <summary>The blocks of one log file. A block that does not start with the magic or runs
    /// past the end is where a crashed writer stopped: reading ends there, as Hudi's does.</summary>
    public static List<Block> Read(string path)
    {
        var data = File.ReadAllBytes(path);
        var blocks = new List<Block>();
        var pos = 0;
        while (pos + Magic.Length + 8 <= data.Length && data.AsSpan(pos, Magic.Length).SequenceEqual(Magic))
        {
            var size = BinaryPrimitives.ReadInt64BigEndian(data.AsSpan(pos + 6));
            var end = pos + 14 + size;
            if (size < 0 || end > data.Length) break;
            var p = pos + 14;
            var version = Int(data, ref p);
            var type = (BlockType)Int(data, ref p);
            var header = version >= 1 ? Map(data, ref p) : new Dictionary<int, string>();
            var length = version >= 1 ? checked((int)Long(data, ref p)) : checked((int)(end - p));
            var content = data.AsSpan(p, length).ToArray();
            blocks.Add(new Block(type, header, content));
            pos = checked((int)end);
        }
        return blocks;
    }

    private static int Int(byte[] d, ref int p)
    {
        var v = BinaryPrimitives.ReadInt32BigEndian(d.AsSpan(p));
        p += 4;
        return v;
    }

    private static long Long(byte[] d, ref int p)
    {
        var v = BinaryPrimitives.ReadInt64BigEndian(d.AsSpan(p));
        p += 8;
        return v;
    }

    private static Dictionary<int, string> Map(byte[] d, ref int p)
    {
        var count = Int(d, ref p);
        var map = new Dictionary<int, string>(count);
        for (var i = 0; i < count; i++)
        {
            var key = Int(d, ref p);
            var length = Int(d, ref p);
            map[key] = Encoding.UTF8.GetString(d, p, length);
            p += length;
        }
        return map;
    }

    /// <summary>An Avro data block's records: a version, a count, then length-prefixed records.</summary>
    public static IEnumerable<IDictionary<string, object?>> Records(Block block, Avro.Schema schema)
    {
        var d = block.Content;
        var p = 0;
        var version = Int(d, ref p);
        var count = version >= 2 ? Int(d, ref p) : int.MaxValue;
        for (var i = 0; i < count && p < d.Length; i++)
        {
            var length = version >= 2 ? Int(d, ref p) : d.Length - p;
            yield return (IDictionary<string, object?>)Avro.Decode(d, p, length, schema)!;
            p += length;
        }
    }

    /// <summary><c>HoodieDeleteRecordList</c> (log block version 3), from Hudi 1.0's model classes.
    /// Each ordering value is a union of single-field wrapper records.</summary>
    private static readonly Avro.Schema DeleteSchema = Avro.Schema.Parse("""
        {"type":"record","name":"HoodieDeleteRecordList","namespace":"org.apache.hudi.avro.model","fields":[{"name":"deleteRecordList","type":{"type":"array","items":{"type":"record","name":"HoodieDeleteRecord","fields":[
         {"name":"recordKey","type":["null","string"]},{"name":"partitionPath","type":["null","string"]},
         {"name":"orderingVal","type":["null",
          {"type":"record","name":"BooleanWrapper","fields":[{"name":"value","type":"boolean"}]},
          {"type":"record","name":"IntWrapper","fields":[{"name":"value","type":"int"}]},
          {"type":"record","name":"LongWrapper","fields":[{"name":"value","type":"long"}]},
          {"type":"record","name":"FloatWrapper","fields":[{"name":"value","type":"float"}]},
          {"type":"record","name":"DoubleWrapper","fields":[{"name":"value","type":"double"}]},
          {"type":"record","name":"BytesWrapper","fields":[{"name":"value","type":"bytes"}]},
          {"type":"record","name":"StringWrapper","fields":[{"name":"value","type":"string"}]},
          {"type":"record","name":"DateWrapper","fields":[{"name":"value","type":"int"}]},
          {"type":"record","name":"DecimalWrapper","fields":[{"name":"value","type":"bytes"}]},
          {"type":"record","name":"TimeMicrosWrapper","fields":[{"name":"value","type":"long"}]},
          {"type":"record","name":"TimestampMicrosWrapper","fields":[{"name":"value","type":"long"}]}]}]}}}]}
        """);

    /// <summary>A delete block's (record key, ordering value) pairs. Version 3 is Avro; older
    /// versions are Kryo-serialised Java objects and are refused rather than guessed at.</summary>
    public static IEnumerable<(string Key, object? Ordering)> Deletes(Block block, string path)
    {
        var d = block.Content;
        var p = 0;
        var version = Int(d, ref p);
        if (version < 3)
            throw new NotSupportedException($"{path} has a version {version} delete block (Kryo-serialised, written before Hudi 0.14), which is not supported.");
        var length = Int(d, ref p);
        var list = (IDictionary<string, object?>)Avro.Decode(d, p, length, DeleteSchema)!;
        foreach (var item in (List<object?>)list["deleteRecordList"]!)
        {
            var r = (IDictionary<string, object?>)item!;
            if (r["recordKey"] is not string key) continue;
            var ordering = r["orderingVal"] is IDictionary<string, object?> wrapper ? wrapper["value"] : null;
            yield return (key, ordering);
        }
    }
}

/// <summary>The remote kernel that reads one merge-on-read file slice.</summary>
public static class HudiKernels
{
    private static readonly ConcurrentDictionary<string, Avro.Schema> Schemas = new(StringComparer.Ordinal);

    /// <summary>
    /// Reads a file slice - its base file, if any, then its log files in order - and merges them
    /// by record key the way Hudi's snapshot query does: a log record replaces what is there unless
    /// (event-time ordering) what is there has a greater ordering value; a delete removes the key
    /// unless it carries a non-default ordering value smaller than the current one. Only blocks of
    /// <paramref name="instants"/> count, minus instants a rollback command block names. The
    /// merged rows are converted to the table schema and <paramref name="ops"/> applied.
    /// </summary>
    [RemoteTask]
    public static DataFrame ReadFileSlice(string basePath, string[] logs, string[] instants, string recordKey, string orderingField, bool eventTime,
        string[] constantNames, string?[] constants, string[] names, int[] types, string[] keep, byte[] ops)
    {
        var all = names.Contains(recordKey) ? names : [.. names, recordKey];
        var allTypes = all.Select((n, i) => i < names.Length ? types[i] : (int)DataType.Text).ToArray();
        var keyIndex = Array.IndexOf(all, recordKey);
        var orderIndex = orderingField.Length > 0 ? Array.IndexOf(all, orderingField) : -1;

        // Base file rows, in file order; keys inserted by logs are appended after them.
        var rows = new Dictionary<string, object?[]>(StringComparer.Ordinal);
        var order = new List<string>();
        if (basePath.Length > 0)
        {
            var baseFrame = LakeKernels.ReadLakeFile(basePath, [.. all.Except(constantNames)], [.. all.Except(constantNames)],
                new int[all.Length - all.Count(constantNames.Contains)], "", constantNames, constants, all, allTypes, FrameOp.Encode([]));
            var columns = all.Select(n => baseFrame[n]).ToArray();
            for (var r = 0; r < baseFrame.RowCount; r++)
            {
                var values = columns.Select(c => c.GetValue(r)).ToArray();
                if (values[keyIndex] is not string key) continue;
                if (rows.TryAdd(key, values)) order.Add(key);
            }
        }

        var visible = instants.ToHashSet(StringComparer.Ordinal);
        var blocks = logs.SelectMany(log => HudiLog.Read(log).Select(b => (Log: log, Block: b))).ToList();
        // A rollback command block cancels the blocks of its target instant, wherever they are.
        var rolledBack = blocks.Where(b => b.Block.Type == HudiLog.BlockType.Command && b.Block.Header.GetValueOrDefault(HudiLog.CommandBlockType) == "0")
            .Select(b => b.Block.Header.GetValueOrDefault(HudiLog.TargetInstantTime)).OfType<string>().ToHashSet(StringComparer.Ordinal);

        foreach (var (log, block) in blocks)
        {
            if (block.Instant is not { } instant || !visible.Contains(instant) || rolledBack.Contains(instant)) continue;
            switch (block.Type)
            {
                case HudiLog.BlockType.AvroData:
                {
                    var json = block.Header.GetValueOrDefault(HudiLog.SchemaKey)
                        ?? throw new InvalidDataException($"{log}: an Avro data block has no schema in its header.");
                    var schema = Schemas.GetOrAdd(json, Avro.Schema.Parse);
                    var convert = Converters(schema, all, allTypes, constantNames, constants, log);
                    foreach (var record in HudiLog.Records(block, schema))
                    {
                        var values = convert(record);
                        if (values[keyIndex] is not string key) continue;
                        var deleted = record.TryGetValue("_hoodie_is_deleted", out var flag) && flag is true;
                        if (rows.TryGetValue(key, out var current))
                        {
                            if (eventTime && orderIndex >= 0 && Compare(current[orderIndex], values[orderIndex]) > 0) continue;
                            if (deleted) rows.Remove(key);
                            else rows[key] = values;
                        }
                        else if (!deleted)
                        {
                            rows[key] = values;
                            order.Add(key);
                        }
                    }
                    break;
                }
                case HudiLog.BlockType.Delete:
                    foreach (var (key, ordering) in HudiLog.Deletes(block, log))
                    {
                        if (!rows.TryGetValue(key, out var current)) continue;
                        // 0 is Hudi's "natural order" default: such a delete always applies.
                        var natural = ordering is null || (ordering is long l && l == 0);
                        if (eventTime && !natural && orderIndex >= 0 && Compare(current[orderIndex], Normalize(ordering)) > 0) continue;
                        rows.Remove(key);
                    }
                    break;
                case HudiLog.BlockType.Command:
                case HudiLog.BlockType.Corrupt:
                    break;
                default:
                    throw new NotSupportedException($"{log} has a {block.Type} block; only Avro data, delete and command blocks are supported.");
            }
        }

        var live = order.Where(rows.ContainsKey).Distinct(StringComparer.Ordinal).Select(k => rows[k]).ToList();
        var frame = new DataFrame(keep.Select(n =>
        {
            var i = Array.IndexOf(all, n);
            return Typed.Build(n, (DataType)allTypes[i], live.Select(v => v[i]).ToArray());
        }));
        return FrameOp.ApplyAll(frame, FrameOp.Decode(ops));
    }

    /// <summary>Values compare numerically when both are numbers, else as text or times.</summary>
    private static int Compare(object? a, object? b)
    {
        if (a is null || b is null) return a is null ? (b is null ? 0 : -1) : 1;
        if (a is double x && b is double y) return x.CompareTo(y);
        if (a is DateTime s && b is DateTime t) return s.CompareTo(t);
        return string.CompareOrdinal(Convert.ToString(a, CultureInfo.InvariantCulture), Convert.ToString(b, CultureInfo.InvariantCulture));
    }

    private static object? Normalize(object? v) => v switch
    {
        long l => (double)l,
        int i => (double)i,
        float f => (double)f,
        _ => v,
    };

    /// <summary>Builds the record-to-row conversion for one block schema: Avro values to the
    /// table's types, timestamps from their logical unit, partition constants filled in.</summary>
    private static Func<IDictionary<string, object?>, object?[]> Converters(Avro.Schema schema, string[] all, int[] types,
        string[] constantNames, string?[] constants, string log)
    {
        var logical = new Dictionary<string, string?>(StringComparer.Ordinal);
        foreach (var field in schema.Root.GetProperty("fields").EnumerateArray())
            logical[field.GetProperty("name").GetString()!] = Logical(field.GetProperty("type"));

        var converters = all.Select((name, i) =>
        {
            var c = Array.IndexOf(constantNames, name);
            if (c >= 0)
            {
                var constant = constants[c];
                return (Func<IDictionary<string, object?>, object?>)(_ => constant);
            }
            var kind = logical.GetValueOrDefault(name);
            var type = (DataType)types[i];
            return record =>
            {
                if (!record.TryGetValue(name, out var v) || v is null) return null;
                return (type, v, kind) switch
                {
                    (DataType.DateTime, long micros, "timestamp-micros" or "local-timestamp-micros") => DateTime.UnixEpoch.AddTicks(micros * 10),
                    (DataType.DateTime, long millis, "timestamp-millis" or "local-timestamp-millis") => DateTime.UnixEpoch.AddMilliseconds(millis),
                    (DataType.DateTime, long days, "date") => DateTime.UnixEpoch.AddDays(days),
                    (DataType.Numeric, byte[], "decimal") => throw new NotSupportedException($"{log}: decimal column '{name}' in a log block is not supported."),
                    (DataType.Numeric, _, _) => Convert.ToDouble(v, CultureInfo.InvariantCulture),
                    _ => v,
                };
            };
        }).ToArray();
        return record => converters.Select(f => f(record)).ToArray();
    }

    private static string? Logical(JsonElement type)
    {
        if (type.ValueKind == JsonValueKind.Array)
            return type.EnumerateArray().Select(Logical).FirstOrDefault(l => l is not null);
        return type.ValueKind == JsonValueKind.Object && type.TryGetProperty("logicalType", out var l) ? l.GetString() : null;
    }
}
