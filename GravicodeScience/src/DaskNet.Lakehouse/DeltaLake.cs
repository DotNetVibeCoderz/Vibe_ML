using System.Globalization;
using System.Text.Json;
using Gravicode.Science.GraviFrame;
using Parquet.Serialization;

namespace Gravicode.Science.DaskNet.Lakehouse;

/// <summary>One commit in a Delta table's log.</summary>
/// <param name="Version">The table version the commit produced.</param>
/// <param name="Timestamp">When it was committed (from its <c>commitInfo</c>), if recorded.</param>
/// <param name="Operation">What it did (<c>WRITE</c>, <c>DELETE</c>, <c>MERGE</c> ...), if recorded.</param>
public sealed record DeltaCommit(long Version, DateTimeOffset? Timestamp, string? Operation);

/// <summary>
/// Delta Lake tables as partitioned <see cref="DaskFrame"/>s: one partition per data file of the
/// chosen version, partition columns filled from the log, filters on them pruning whole files.
/// </summary>
/// <remarks>
/// <para>
/// The state of a version is replayed from the transaction log: the newest checkpoint at or
/// before it (single- or multi-part Parquet), then every JSON commit after it, in order. Time
/// travel is by version or by commit timestamp. The schema is the one in force at that version,
/// so a column added later does not appear when reading an earlier version, and reads as missing
/// in files written before it was added.
/// </para>
/// <para>
/// Column mapping (<c>name</c> and <c>id</c> modes) is followed: columns are read from the files
/// under their physical names (or, in <c>id</c> mode, by Parquet field id) and come back under the
/// table's logical names, so renamed and dropped columns read as the table says. Deletion vectors
/// - inline, beside the table, or at an absolute path - are applied on the worker as each file is
/// read: the rows they mark are dropped before any filter runs.
/// </para>
/// <para>
/// Refused rather than misread: v2 (UUID-named) checkpoints, nested column types, and a reader
/// version above 3 or a reader feature not listed in <see cref="SupportedReaderFeatures"/>. Local
/// paths only; reading cloud storage needs it mounted or synced.
/// </para>
/// </remarks>
public static class DeltaLake
{
    /// <summary>Reader features a table may declare and still be read.</summary>
    public static IReadOnlyCollection<string> SupportedReaderFeatures { get; } = ["timestampNtz", "vacuumProtocolCheck", "deletionVectors", "columnMapping"];

    /// <summary>Reads the table at <paramref name="path"/>, at its latest version or at <paramref name="version"/>.</summary>
    public static DaskFrame Read(string path, long? version = null)
    {
        var snapshot = Load(path, version);
        return snapshot.ToFrame();
    }

    /// <summary>Reads the table as it was at <paramref name="asOf"/>: the newest version committed
    /// at or before it.</summary>
    public static DaskFrame Read(string path, DateTimeOffset asOf)
    {
        var commit = History(path).Where(c => c.Timestamp is { } t && t <= asOf).MaxBy(c => c.Version)
            ?? throw new ArgumentException($"The table at {path} has no version committed at or before {asOf:o}.", nameof(asOf));
        return Read(path, commit.Version);
    }

    /// <summary>The commits still in the log, oldest first.</summary>
    public static IReadOnlyList<DeltaCommit> History(string path)
    {
        var log = LogDirectory(path);
        var commits = new List<DeltaCommit>();
        foreach (var (v, file) in Commits(log))
        {
            DateTimeOffset? timestamp = null;
            string? operation = null;
            foreach (var line in File.ReadLines(file))
            {
                if (line.Length == 0) continue;
                using var doc = JsonDocument.Parse(line);
                if (!doc.RootElement.TryGetProperty("commitInfo", out var info)) continue;
                if (info.TryGetProperty("timestamp", out var ts) && ts.ValueKind == JsonValueKind.Number)
                    timestamp = DateTimeOffset.FromUnixTimeMilliseconds(ts.GetInt64());
                if (info.TryGetProperty("operation", out var op)) operation = op.GetString();
            }
            commits.Add(new DeltaCommit(v, timestamp ?? File.GetLastWriteTimeUtc(file), operation));
        }
        return commits;
    }

    /// <summary>The newest version in the log.</summary>
    public static long LatestVersion(string path)
    {
        var log = LogDirectory(path);
        var commits = Commits(log).Select(c => c.Version).DefaultIfEmpty(-1).Max();
        var checkpoints = Checkpoints(log).Select(c => c.Version).DefaultIfEmpty(-1).Max();
        var latest = Math.Max(commits, checkpoints);
        return latest >= 0 ? latest : throw new InvalidDataException($"{log} holds no commits.");
    }

    // ------------------------------------------------------------------ log replay

    internal sealed class Snapshot
    {
        public required string Root { get; init; }
        public long Version { get; set; }
        public string? SchemaString { get; set; }
        public string[] PartitionColumns { get; set; } = [];
        public Dictionary<string, string> Configuration { get; set; } = [];
        public Dictionary<string, (long Size, Dictionary<string, string?> Values, string? DeletionVector)> Files { get; } = new(StringComparer.Ordinal);

        public DaskFrame ToFrame()
        {
            if (SchemaString is null) throw new InvalidDataException($"Delta table at {Root} has no metaData action at version {Version}.");
            var mode = Configuration.GetValueOrDefault("delta.columnMapping.mode", "none");
            if (mode is not ("none" or "name" or "id"))
                throw new NotSupportedException($"Delta table at {Root} uses column mapping mode '{mode}', which is not supported.");
            var schema = Schema(SchemaString, Root, mode);
            var names = schema.Select(c => c.Name).ToArray();
            var types = schema.Select(c => c.Type).ToArray();

            // Mapped tables name file columns, and key partition values, by physical name.
            var fileNames = schema.Where(c => c.Physical != c.Name).ToDictionary(c => c.Name, c => c.Physical);
            var fieldIds = mode == "id" ? schema.Where(c => c.Id > 0).ToDictionary(c => c.Name, c => c.Id) : [];
            var files = Files.OrderBy(f => f.Key, StringComparer.Ordinal).Select(f => new LakeFile(
                Path.GetFullPath(Path.Combine(Root, Uri.UnescapeDataString(PathOf(f.Key)).Replace('/', Path.DirectorySeparatorChar))),
                f.Value.Size, LogicalPartitionValues(f.Value.Values, schema, PartitionColumns), fileNames)
            {
                FieldIds = fieldIds,
                DeletionVector = f.Value.DeletionVector,
            }).ToArray();
            var source = new LakeSource($"Delta {Path.GetFileName(Root)} v{Version}", names, types, files, PartitionColumns, partitionsInFile: false);
            return new DaskFrame(new ScanNode(source));
        }

        private static Dictionary<string, string?> LogicalPartitionValues(Dictionary<string, string?> stored, IReadOnlyList<SchemaColumn> schema, string[] partitionColumns)
        {
            var values = new Dictionary<string, string?>(StringComparer.Ordinal);
            foreach (var column in partitionColumns)
            {
                var physical = schema.FirstOrDefault(c => c.Name == column).Physical ?? column;
                values[column] = stored.TryGetValue(physical, out var v) ? v : stored.GetValueOrDefault(column);
            }
            return values;
        }
    }

    internal static Snapshot Load(string path, long? version)
    {
        var root = Path.GetFullPath(path);
        var log = LogDirectory(root);
        var target = version ?? LatestVersion(root);
        if (target < 0) throw new ArgumentOutOfRangeException(nameof(version));
        var snapshot = new Snapshot { Root = root, Version = target };

        var checkpoint = Checkpoints(log).Where(c => c.Version <= target).OrderByDescending(c => c.Version).FirstOrDefault();
        var start = 0L;
        if (checkpoint.Files is not null)
        {
            foreach (var part in checkpoint.Files) ReadCheckpoint(part, snapshot);
            start = checkpoint.Version + 1;
        }

        var commits = Commits(log).Where(c => c.Version >= start && c.Version <= target).ToList();
        for (var v = start; v <= target; v++)
            if (commits.All(c => c.Version != v))
                throw new InvalidDataException($"The Delta log at {log} has no commit {v} and no checkpoint covering it; version {target} cannot be rebuilt.");
        foreach (var (_, file) in commits)
            foreach (var line in File.ReadLines(file))
                if (line.Length > 0) Apply(line, snapshot);
        return snapshot;
    }

    private static string LogDirectory(string path)
    {
        var log = Path.Combine(Path.GetFullPath(path), "_delta_log");
        return Directory.Exists(log) ? log : throw new DirectoryNotFoundException($"{path} is not a Delta table: it has no _delta_log directory.");
    }

    private static IEnumerable<(long Version, string File)> Commits(string log)
        => Directory.EnumerateFiles(log, "*.json")
            .Select(f => (Name: Path.GetFileNameWithoutExtension(f), File: f))
            .Where(f => f.Name.Length == 20 && f.Name.All(char.IsAsciiDigit))
            .Select(f => (long.Parse(f.Name, CultureInfo.InvariantCulture), f.File))
            .OrderBy(f => f.Item1);

    /// <summary>Complete checkpoints: single-part, or every part of a multi-part one.</summary>
    private static IEnumerable<(long Version, string[]? Files)> Checkpoints(string log)
    {
        var parts = new Dictionary<long, (int Total, List<string> Files)>();
        foreach (var file in Directory.EnumerateFiles(log, "*.checkpoint*.parquet"))
        {
            var pieces = Path.GetFileName(file).Split('.');
            // 00000000000000000010.checkpoint.parquet or ...checkpoint.0000000001.0000000002.parquet
            if (pieces.Length < 3 || !long.TryParse(pieces[0], CultureInfo.InvariantCulture, out var v) || pieces[1] != "checkpoint") continue;
            if (pieces.Length == 3) { yield return (v, [file]); continue; }
            if (pieces.Length == 5 && int.TryParse(pieces[3], CultureInfo.InvariantCulture, out var total))
            {
                if (!parts.TryGetValue(v, out var entry)) parts[v] = entry = (total, []);
                entry.Files.Add(file);
            }
            else if (pieces.Length == 4)
                throw new NotSupportedException($"{file} is a v2 (UUID-named) checkpoint, which is not supported.");
        }
        foreach (var (v, (total, files)) in parts)
            if (files.Count == total) yield return (v, files.Order(StringComparer.Ordinal).ToArray());
    }

    private static void Apply(string line, Snapshot snapshot)
    {
        using var doc = JsonDocument.Parse(line);
        foreach (var action in doc.RootElement.EnumerateObject())
        {
            var a = action.Value;
            switch (action.Name)
            {
                case "add":
                    snapshot.Files[FileKey(a.GetProperty("path").GetString()!, VectorId(a))] = (
                        a.TryGetProperty("size", out var size) ? size.GetInt64() : 0,
                        PartitionValues(a),
                        DeletionVectors.FromJson(a, snapshot.Root));
                    break;
                case "remove":
                    snapshot.Files.Remove(FileKey(a.GetProperty("path").GetString()!, VectorId(a)));
                    break;
                case "metaData":
                    snapshot.SchemaString = a.GetProperty("schemaString").GetString();
                    snapshot.PartitionColumns = a.TryGetProperty("partitionColumns", out var pc)
                        ? pc.EnumerateArray().Select(e => e.GetString()!).ToArray() : [];
                    snapshot.Configuration = a.TryGetProperty("configuration", out var conf) && conf.ValueKind == JsonValueKind.Object
                        ? conf.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetString() ?? "") : [];
                    break;
                case "protocol":
                    CheckProtocol(
                        a.GetProperty("minReaderVersion").GetInt32(),
                        a.TryGetProperty("readerFeatures", out var rf) && rf.ValueKind == JsonValueKind.Array
                            ? rf.EnumerateArray().Select(e => e.GetString()!).ToArray() : null,
                        snapshot.Root);
                    break;
            }
        }
    }

    // A file's identity is its path *and* its deletion vector: deleting more rows of a file adds
    // (path, new vector) and removes (path, old vector) in one commit, in either order.
    private static string FileKey(string path, string vectorId) => vectorId.Length == 0 ? path : path + "\u0000" + vectorId;

    private static string PathOf(string key) => key.IndexOf('\u0000') is var cut and >= 0 ? key[..cut] : key;

    private static string VectorId(JsonElement action)
        => action.TryGetProperty("deletionVector", out var dv) && dv.ValueKind == JsonValueKind.Object
            ? VectorId(dv.GetProperty("storageType").GetString()!, dv.GetProperty("pathOrInlineDv").GetString(),
                dv.TryGetProperty("offset", out var o) && o.ValueKind == JsonValueKind.Number ? o.GetInt64() : null)
            : "";

    private static string VectorId(string storageType, string? pathOrInline, long? offset)
        => offset is { } at ? $"{storageType}{pathOrInline}@{at.ToString(CultureInfo.InvariantCulture)}" : storageType + pathOrInline;

    private static Dictionary<string, string?> PartitionValues(JsonElement add)
        => add.TryGetProperty("partitionValues", out var pv) && pv.ValueKind == JsonValueKind.Object
            ? pv.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.ValueKind == JsonValueKind.Null ? null : p.Value.GetString())
            : [];

    private static void CheckProtocol(int minReader, IReadOnlyCollection<string>? features, string root)
    {
        if (minReader > 3)
            throw new NotSupportedException($"Delta table at {root} needs reader version {minReader}; this reader implements up to 3.");
        var unknown = features?.Where(f => !SupportedReaderFeatures.Contains(f)).ToArray() ?? [];
        if (unknown.Length > 0)
            throw new NotSupportedException($"Delta table at {root} needs reader feature(s) {string.Join(", ", unknown)}, which are not supported.");
    }

    // ------------------------------------------------------------------ checkpoints

    /// <summary>
    /// Reads a checkpoint with the file's own schema (untyped), because writers disagree on which
    /// fields are required and a typed read insists the class's nullability matches the file's.
    /// </summary>
    private static void ReadCheckpoint(string file, Snapshot snapshot)
    {
        using var stream = File.OpenRead(file);
        var result = ParquetSerializer.DeserializeAsync(stream).GetAwaiter().GetResult();
        foreach (var row in result.Data)
        {
            IDictionary<string, object?> fields = row!;
            if (Struct(fields, "protocol") is { } p && Get(p, "minReaderVersion") is { } min)
                CheckProtocol(Convert.ToInt32(min, CultureInfo.InvariantCulture), Strings(Get(p, "readerFeatures")), snapshot.Root);
            if (Struct(fields, "metaData") is { } m && Get(m, "schemaString") is string schema)
            {
                snapshot.SchemaString = schema;
                snapshot.PartitionColumns = Strings(Get(m, "partitionColumns"))?.ToArray() ?? [];
                snapshot.Configuration = StringMap(Get(m, "configuration"))
                    .ToDictionary(e => e.Key, e => e.Value ?? "");
            }
            if (Struct(fields, "add") is { } a && Get(a, "path") is string path)
                snapshot.Files[FileKey(path, Struct(a, "deletionVector") is { } id && Get(id, "storageType") is string kind
                    ? VectorId(kind, (string?)Get(id, "pathOrInlineDv"), Get(id, "offset") is { } at ? Convert.ToInt64(at, CultureInfo.InvariantCulture) : null)
                    : "")] = (
                    Get(a, "size") is { } size ? Convert.ToInt64(size, CultureInfo.InvariantCulture) : 0,
                    StringMap(Get(a, "partitionValues")),
                    Struct(a, "deletionVector") is { } dv && Get(dv, "storageType") is string storage
                        ? DeletionVectors.Describe(storage, (string)Get(dv, "pathOrInlineDv")!,
                            Get(dv, "offset") is { } offset ? Convert.ToInt32(offset, CultureInfo.InvariantCulture) : null,
                            Convert.ToInt32(Get(dv, "sizeInBytes"), CultureInfo.InvariantCulture), snapshot.Root)
                        : null);
            // A checkpoint's remove actions are tombstones of files already gone; nothing to undo.
        }
    }

    private static IDictionary<string, object?>? Struct(IDictionary<string, object?> row, string name)
        => row.TryGetValue(name, out var v) ? Struct(v) : null;

    private static IDictionary<string, object?>? Struct(object? value) => value switch
    {
        IDictionary<string, object?> d => d,
        _ => null,
    };

    private static List<string>? Strings(object? value)
        => value is System.Collections.IEnumerable items and not string
            ? items.Cast<object?>().Where(x => x is not null).Select(x => x!.ToString()!).ToList()
            : null;

    /// <summary>A Parquet map, however the reader shaped it: a dictionary, or key/value records.</summary>
    private static Dictionary<string, string?> StringMap(object? value)
    {
        var map = new Dictionary<string, string?>(StringComparer.Ordinal);
        switch (value)
        {
            case null:
                break;
            case System.Collections.IDictionary d:
                foreach (System.Collections.DictionaryEntry e in d) map[e.Key.ToString()!] = e.Value?.ToString();
                break;
            case System.Collections.IEnumerable items:
                foreach (var item in items)
                    if (Struct(item) is { } kv && Get(kv, "key") is { } k) map[k.ToString()!] = Get(kv, "value")?.ToString();
                break;
        }
        return map;
    }

    private static object? Get(IDictionary<string, object?> d, string key) => d.TryGetValue(key, out var v) ? v : null;

    // ------------------------------------------------------------------ schema

    /// <summary>A table column: its logical name and type, and (under column mapping) the name and
    /// field id its files use.</summary>
    internal readonly record struct SchemaColumn(string Name, DataType Type, string Physical, int Id);

    internal static IReadOnlyList<SchemaColumn> Schema(string schemaString, string root, string mode = "none")
    {
        using var doc = JsonDocument.Parse(schemaString);
        var columns = new List<SchemaColumn>();
        foreach (var field in doc.RootElement.GetProperty("fields").EnumerateArray())
        {
            var name = field.GetProperty("name").GetString()!;
            var type = field.GetProperty("type");
            if (type.ValueKind != JsonValueKind.String)
                throw new NotSupportedException($"Delta table at {root}: column '{name}' is a nested type, which is not supported.");
            var physical = name;
            var id = 0;
            if (mode != "none")
            {
                if (!field.TryGetProperty("metadata", out var meta) || meta.ValueKind != JsonValueKind.Object
                    || !meta.TryGetProperty("delta.columnMapping.physicalName", out var p))
                    throw new InvalidDataException($"Delta table at {root} uses column mapping, but column '{name}' has no physical name.");
                physical = p.GetString()!;
                if (meta.TryGetProperty("delta.columnMapping.id", out var i) && i.ValueKind == JsonValueKind.Number) id = i.GetInt32();
            }
            columns.Add(new SchemaColumn(name, TypeOf(type.GetString()!, name, root), physical, id));
        }
        return columns;
    }

    private static DataType TypeOf(string type, string column, string root) => type switch
    {
        "long" or "integer" or "short" or "byte" or "float" or "double" => DataType.Numeric,
        _ when type.StartsWith("decimal", StringComparison.Ordinal) => DataType.Numeric,
        "string" or "binary" => DataType.Text,
        "boolean" => DataType.Boolean,
        "timestamp" or "timestamp_ntz" or "date" => DataType.DateTime,
        _ => throw new NotSupportedException($"Delta table at {root}: column '{column}' has type '{type}', which is not supported."),
    };
}
