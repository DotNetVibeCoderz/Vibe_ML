using System.Globalization;
using System.Text;
using System.Text.Json;
using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet.Lakehouse;

/// <summary>One completed instant on a Hudi table's timeline.</summary>
/// <param name="Time">The instant time (<c>yyyyMMddHHmmssSSS</c>), which names the files it wrote.</param>
/// <param name="Action"><c>commit</c>, <c>deltacommit</c> or <c>replacecommit</c>.</param>
/// <param name="CompletionTime">When it completed, for timeline layout 2 (Hudi 1.x); null before.</param>
public sealed record HudiInstant(string Time, string Action, string? CompletionTime);

/// <summary>
/// Apache Hudi tables as partitioned <see cref="DaskFrame"/>s: one partition per file group's
/// latest base file, filters on partition columns pruning whole files.
/// </summary>
/// <remarks>
/// <para>
/// The table is read the way Hudi's file-system view builds it: the completed instants on the
/// timeline (<c>.hoodie/timeline/</c> for layout 2, <c>.hoodie/</c> for layout 1) decide which
/// files exist; base files are named <c>{fileId}_{writeToken}_{instant}.parquet</c>, and the newest
/// visible one per file group is that group's state. File groups a <c>replacecommit</c> replaced
/// (insert-overwrite, clustering) drop out from that instant on. Time travel is by instant;
/// uncommitted and rolled-back writes are never visible because their instants never completed.
/// </para>
/// <para>
/// Copy-on-write tables read fully. A merge-on-read table whose updates still sit in log files
/// is refused for a snapshot read, since merging Avro log blocks is not implemented; pass
/// <c>readOptimized: true</c> to read its base files only, which is Hudi's own read-optimized
/// query (it lags the log until compaction). The schema is the writer schema recorded in the
/// newest commit's metadata; the <c>_hoodie_*</c> meta columns are left out unless asked for.
/// Nested types and non-Parquet base files are refused. Local paths only.
/// </para>
/// </remarks>
public static class Hudi
{
    /// <summary>The meta columns Hudi writes into every base file.</summary>
    public static IReadOnlyList<string> MetaColumns { get; } =
        ["_hoodie_commit_time", "_hoodie_commit_seqno", "_hoodie_record_key", "_hoodie_partition_path", "_hoodie_file_name"];

    /// <summary>Reads the table at its latest instant, or as of <paramref name="instant"/>.</summary>
    /// <param name="path">The table directory (holding <c>.hoodie/</c>).</param>
    /// <param name="instant">An instant time from <see cref="Instants"/>; the latest when null.</param>
    /// <param name="readOptimized">Read a merge-on-read table's base files only, ignoring its logs.</param>
    /// <param name="includeMetaColumns">Keep the <c>_hoodie_*</c> columns.</param>
    public static DaskFrame Read(string path, string? instant = null, bool readOptimized = false, bool includeMetaColumns = false)
        => Table.Load(path).Frame(instant, readOptimized, includeMetaColumns);

    /// <summary>Reads the table as it was at <paramref name="asOf"/>: the newest instant at or
    /// before it, instant times taken in the table's timeline time zone (local time unless the
    /// table says UTC).</summary>
    public static DaskFrame Read(string path, DateTimeOffset asOf, bool readOptimized = false, bool includeMetaColumns = false)
    {
        var table = Table.Load(path);
        var at = table.Utc ? asOf.UtcDateTime : asOf.ToLocalTime().DateTime;
        var instant = table.Instants.Where(i => ParseInstant(i.Time) <= at).Select(i => i.Time).LastOrDefault()
            ?? throw new ArgumentException($"The table at {path} has no instant at or before {asOf:o}.", nameof(asOf));
        return table.Frame(instant, readOptimized, includeMetaColumns);
    }

    /// <summary>The completed instants on the active timeline, oldest first.</summary>
    public static IReadOnlyList<HudiInstant> Instants(string path) => Table.Load(path).Instants;

    /// <summary>The table's <c>hoodie.properties</c>.</summary>
    public static IReadOnlyDictionary<string, string> Properties(string path) => Table.Load(path).Properties;

    internal static DateTime ParseInstant(string time)
    {
        var format = time.Length >= 17 ? "yyyyMMddHHmmssfff" : "yyyyMMddHHmmss";
        return DateTime.ParseExact(time[..Math.Min(time.Length, 17)], format, CultureInfo.InvariantCulture);
    }

    internal sealed class Table
    {
        private static readonly string[] DataActions = ["commit", "deltacommit", "replacecommit"];

        private readonly string _root;
        private readonly Dictionary<string, string> _instantFiles;

        private Table(string root, Dictionary<string, string> properties, List<HudiInstant> instants, Dictionary<string, string> files)
        {
            _root = root;
            Properties = properties;
            Instants = instants;
            _instantFiles = files;
        }

        public IReadOnlyDictionary<string, string> Properties { get; }
        public IReadOnlyList<HudiInstant> Instants { get; }
        public bool Utc => Properties.GetValueOrDefault("hoodie.table.timeline.timezone", "LOCAL").Equals("UTC", StringComparison.OrdinalIgnoreCase);
        private bool MergeOnRead => Properties.GetValueOrDefault("hoodie.table.type", "COPY_ON_WRITE") == "MERGE_ON_READ";

        public static Table Load(string path)
        {
            var root = Path.GetFullPath(path);
            var meta = Path.Combine(root, ".hoodie");
            var propertiesFile = Path.Combine(meta, "hoodie.properties");
            if (!File.Exists(propertiesFile)) throw new DirectoryNotFoundException($"{path} is not a Hudi table: it has no .hoodie/hoodie.properties.");
            var properties = ReadProperties(propertiesFile);

            var format = properties.GetValueOrDefault("hoodie.table.base.file.format", "PARQUET");
            if (!format.Equals("PARQUET", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException($"Hudi table at {root} stores base files as {format}; only Parquet is supported.");

            // Layout 2 (Hudi 1.x): timeline/<requested>_<completed>.<action>. Layout 1: <instant>.<action>.
            var layout2 = properties.GetValueOrDefault("hoodie.timeline.layout.version", "1") == "2";
            var timeline = layout2 ? Path.Combine(meta, properties.GetValueOrDefault("hoodie.timeline.path", "timeline")) : meta;
            var instants = new List<HudiInstant>();
            var files = new Dictionary<string, string>(StringComparer.Ordinal);
            if (Directory.Exists(timeline))
                foreach (var file in Directory.EnumerateFiles(timeline))
                {
                    var name = Path.GetFileName(file);
                    if (name.StartsWith('.')) continue;
                    var dot = name.IndexOf('.');
                    if (dot <= 0) continue;
                    var action = name[(dot + 1)..];
                    if (!DataActions.Contains(action)) continue;   // .inflight / .requested never complete here
                    var stem = name[..dot];
                    var underscore = stem.IndexOf('_');
                    var time = underscore < 0 ? stem : stem[..underscore];
                    if (!time.All(char.IsAsciiDigit)) continue;
                    instants.Add(new HudiInstant(time, action, underscore < 0 ? null : stem[(underscore + 1)..]));
                    files[time] = file;
                }
            instants.Sort((a, b) => string.CompareOrdinal(a.Time, b.Time));
            return new Table(root, properties, instants, files);
        }

        public DaskFrame Frame(string? instant, bool readOptimized, bool includeMetaColumns)
        {
            if (Instants.Count == 0) throw new InvalidDataException($"Hudi table at {_root} has no completed commits.");
            var asOf = instant ?? Instants[^1].Time;
            if (instant is not null && Instants.All(i => string.CompareOrdinal(i.Time, instant) > 0))
                throw new ArgumentException($"Hudi table at {_root} has no completed instant at or before {instant}.", nameof(instant));
            var visible = Instants.Where(i => string.CompareOrdinal(i.Time, asOf) <= 0).ToList();
            var completed = visible.Select(i => i.Time).ToHashSet(StringComparer.Ordinal);

            // File groups replaced by a visible replacecommit.
            var replaced = new HashSet<(string Partition, string FileId)>();
            foreach (var r in visible.Where(i => i.Action == "replacecommit"))
                foreach (var (partition, ids) in StringListMap(Metadata(r.Time), "partitionToReplaceFileIds"))
                    foreach (var id in ids) replaced.Add((Normalize(partition), id));

            var partitionFields = Properties.GetValueOrDefault("hoodie.table.partition.fields", "")
                .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(f => f.Contains(':') ? f[..f.IndexOf(':')] : f).ToArray();

            var latest = new Dictionary<(string Partition, string FileId), (string Instant, string Path, long Size)>();
            var logs = new List<string>();
            foreach (var (partition, directory) in PartitionDirectories(partitionFields.Length))
                foreach (var file in Directory.EnumerateFiles(directory))
                {
                    var name = Path.GetFileName(file);
                    if (name.StartsWith('.'))
                    {
                        // .{fileId}_{instant}.log.{version}_{token}
                        var log = name.IndexOf(".log.", StringComparison.Ordinal);
                        if (log < 0) continue;
                        var head = name[1..log];
                        var cut = head.LastIndexOf('_');
                        if (cut > 0 && completed.Contains(head[(cut + 1)..]) && !replaced.Contains((partition, head[..cut]))) logs.Add(file);
                        continue;
                    }
                    if (!name.EndsWith(".parquet", StringComparison.OrdinalIgnoreCase)) continue;
                    var stem = name[..^".parquet".Length];
                    var last = stem.LastIndexOf('_');
                    var token = last > 0 ? stem.LastIndexOf('_', last - 1) : -1;
                    if (token <= 0) continue;
                    var (fileId, time) = (stem[..token], stem[(last + 1)..]);
                    if (!completed.Contains(time) || replaced.Contains((partition, fileId))) continue;
                    if (!latest.TryGetValue((partition, fileId), out var seen) || string.CompareOrdinal(time, seen.Instant) > 0)
                        latest[(partition, fileId)] = (time, file, new FileInfo(file).Length);
                }

            if (MergeOnRead && !readOptimized && logs.Count > 0)
                throw new NotSupportedException(
                    $"Hudi merge-on-read table at {_root} has updates in log files (e.g. {Path.GetFileName(logs[0])}) that a snapshot read would have to merge, " +
                    "which is not supported; pass readOptimized: true to read the base files only (Hudi's read-optimized query).");

            var (names, types) = Schema(visible, includeMetaColumns);
            var inFile = !Properties.GetValueOrDefault("hoodie.datasource.write.drop.partition.columns", "false").Equals("true", StringComparison.OrdinalIgnoreCase);
            foreach (var field in partitionFields)
                if (!names.Contains(field)) throw new InvalidDataException($"Hudi table at {_root}: partition field '{field}' is not in the schema.");

            var files = latest.OrderBy(f => f.Key.Partition, StringComparer.Ordinal).ThenBy(f => f.Key.FileId, StringComparer.Ordinal)
                .Select(f => new LakeFile(f.Value.Path, f.Value.Size, PartitionValues(f.Key.Partition, partitionFields), new Dictionary<string, string>()))
                .ToArray();
            var source = new LakeSource($"Hudi {Path.GetFileName(_root)} @{asOf}", names, types, files, partitionFields, partitionsInFile: inFile);
            return new DaskFrame(new ScanNode(source));
        }

        /// <summary>Partition directories relative to the root, '/'-separated; "" for a table that is not partitioned.</summary>
        private IEnumerable<(string Partition, string Directory)> PartitionDirectories(int depth)
        {
            if (depth == 0) { yield return ("", _root); yield break; }
            var level = new List<string> { _root };
            for (var d = 0; d < depth; d++)
                level = level.SelectMany(dir => Directory.EnumerateDirectories(dir).Where(x => !Path.GetFileName(x).StartsWith('.'))).ToList();
            foreach (var dir in level)
                yield return (Path.GetRelativePath(_root, dir).Replace(Path.DirectorySeparatorChar, '/'), dir);
        }

        private static string Normalize(string partition) => partition.Replace('\\', '/').Trim('/');

        private static Dictionary<string, string?> PartitionValues(string partition, string[] fields)
        {
            var values = new Dictionary<string, string?>(StringComparer.Ordinal);
            if (fields.Length == 0) return values;
            var segments = partition.Split('/');
            for (var i = 0; i < fields.Length && i < segments.Length; i++)
            {
                var segment = segments[i];
                var eq = segment.IndexOf('=');
                var value = Uri.UnescapeDataString(eq >= 0 ? segment[(eq + 1)..] : segment);
                values[fields[i]] = value is "__HIVE_DEFAULT_PARTITION__" or "default" ? null : value;
            }
            return values;
        }

        /// <summary>The writer schema of the newest visible commit that records one, then the meta columns.</summary>
        private (string[] Names, DataType[] Types) Schema(List<HudiInstant> visible, bool includeMeta)
        {
            string? schema = null;
            for (var i = visible.Count - 1; i >= 0 && schema is null; i--)
                if (StringMap(Metadata(visible[i].Time), "extraMetadata").GetValueOrDefault("schema") is { Length: > 0 } s) schema = s;
            if (schema is null) throw new InvalidDataException($"Hudi table at {_root}: no commit records a writer schema.");

            var names = new List<string>();
            var types = new List<DataType>();
            if (includeMeta)
                foreach (var m in MetaColumns) { names.Add(m); types.Add(DataType.Text); }
            using var doc = JsonDocument.Parse(schema);
            foreach (var field in doc.RootElement.GetProperty("fields").EnumerateArray())
            {
                var name = field.GetProperty("name").GetString()!;
                if (MetaColumns.Contains(name)) continue;
                names.Add(name);
                types.Add(AvroType(field.GetProperty("type"), name));
            }
            return (names.ToArray(), types.ToArray());
        }

        private DataType AvroType(JsonElement type, string column)
        {
            if (type.ValueKind == JsonValueKind.Array)
            {
                var branches = type.EnumerateArray().Where(b => !(b.ValueKind == JsonValueKind.String && b.GetString() == "null")).ToArray();
                if (branches.Length != 1) throw new NotSupportedException($"Hudi table at {_root}: column '{column}' is a union type, which is not supported.");
                return AvroType(branches[0], column);
            }
            string primitive;
            string? logical = null;
            if (type.ValueKind == JsonValueKind.Object)
            {
                primitive = type.GetProperty("type").ValueKind == JsonValueKind.String ? type.GetProperty("type").GetString()! : "record";
                if (type.TryGetProperty("logicalType", out var l)) logical = l.GetString();
            }
            else primitive = type.GetString()!;
            return (primitive, logical) switch
            {
                (_, "timestamp-millis" or "timestamp-micros" or "local-timestamp-millis" or "local-timestamp-micros" or "date") => DataType.DateTime,
                (_, "decimal") => DataType.Numeric,
                ("int" or "long" or "float" or "double", _) => DataType.Numeric,
                ("string" or "enum" or "bytes" or "fixed", _) => DataType.Text,
                ("boolean", _) => DataType.Boolean,
                _ => throw new NotSupportedException($"Hudi table at {_root}: column '{column}' has type '{primitive}', which is not supported."),
            };
        }

        /// <summary>A completed instant's commit metadata: Avro for layout 2, JSON for layout 1.</summary>
        private IDictionary<string, object?> Metadata(string time)
        {
            var file = _instantFiles[time];
            if (new FileInfo(file).Length == 0) return new Dictionary<string, object?>();
            using (var probe = File.OpenRead(file))
            {
                Span<byte> magic = stackalloc byte[4];
                if (probe.Read(magic) == 4 && magic.SequenceEqual("Obj\u0001"u8))
                {
                    probe.Close();
                    var (_, records) = Avro.Read(file);
                    return records.FirstOrDefault() as IDictionary<string, object?> ?? new Dictionary<string, object?>();
                }
            }
            using var doc = JsonDocument.Parse(File.ReadAllText(file, Encoding.UTF8));
            return (IDictionary<string, object?>)FromJson(doc.RootElement)!;
        }

        private static object? FromJson(JsonElement e) => e.ValueKind switch
        {
            JsonValueKind.Object => e.EnumerateObject().ToDictionary(p => p.Name, p => FromJson(p.Value), StringComparer.Ordinal),
            JsonValueKind.Array => e.EnumerateArray().Select(FromJson).ToList(),
            JsonValueKind.String => e.GetString(),
            JsonValueKind.Number => e.GetDouble(),
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => null,
        };

        private static Dictionary<string, string> StringMap(IDictionary<string, object?> metadata, string key)
            => metadata.TryGetValue(key, out var v) && v is IDictionary<string, object?> map
                ? map.Where(e => e.Value is string).ToDictionary(e => e.Key, e => (string)e.Value!, StringComparer.Ordinal)
                : [];

        private static IEnumerable<(string Key, List<string> Values)> StringListMap(IDictionary<string, object?> metadata, string key)
        {
            if (!metadata.TryGetValue(key, out var v) || v is not IDictionary<string, object?> map) yield break;
            foreach (var (k, list) in map)
                if (list is IEnumerable<object?> items) yield return (k, items.OfType<string>().ToList());
        }

        private static Dictionary<string, string> ReadProperties(string file)
        {
            var properties = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var raw in File.ReadLines(file))
            {
                var line = raw.Trim();
                if (line.Length == 0 || line[0] is '#' or '!') continue;
                var eq = line.IndexOf('=');
                if (eq <= 0) continue;
                properties[line[..eq].Trim()] = line[(eq + 1)..].Trim().Replace("\\:", ":").Replace("\\=", "=");
            }
            return properties;
        }
    }
}
