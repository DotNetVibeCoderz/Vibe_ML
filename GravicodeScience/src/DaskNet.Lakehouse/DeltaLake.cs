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
/// Refused rather than misread: column mapping (renamed or dropped columns, where the file's
/// column names are not the table's), deletion vectors on any live file, v2 (UUID-named)
/// checkpoints, nested column types, and a reader version above 3 or a reader feature not listed
/// in <see cref="SupportedReaderFeatures"/>. Local paths only; reading cloud storage needs it
/// mounted or synced.
/// </para>
/// </remarks>
public static class DeltaLake
{
    /// <summary>Reader features a table may declare and still be read.</summary>
    public static IReadOnlyCollection<string> SupportedReaderFeatures { get; } = ["timestampNtz", "vacuumProtocolCheck", "deletionVectors"];

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
        public Dictionary<string, (long Size, Dictionary<string, string?> Values, bool HasDeletionVector)> Files { get; } = new(StringComparer.Ordinal);

        public DaskFrame ToFrame()
        {
            if (SchemaString is null) throw new InvalidDataException($"Delta table at {Root} has no metaData action at version {Version}.");
            var mode = Configuration.GetValueOrDefault("delta.columnMapping.mode", "none");
            if (mode != "none")
                throw new NotSupportedException($"Delta table at {Root} uses column mapping (mode '{mode}'); its files' column names are not the table's.");
            var withVectors = Files.Where(f => f.Value.HasDeletionVector).Select(f => f.Key).FirstOrDefault();
            if (withVectors is not null)
                throw new NotSupportedException($"Delta table at {Root} has deletion vectors on live files (e.g. {withVectors}); rows marked deleted would be read.");

            var (names, types) = Schema(SchemaString, Root);
            var files = Files.OrderBy(f => f.Key, StringComparer.Ordinal).Select(f => new LakeFile(
                Path.GetFullPath(Path.Combine(Root, Uri.UnescapeDataString(f.Key).Replace('/', Path.DirectorySeparatorChar))),
                f.Value.Size, f.Value.Values, new Dictionary<string, string>())).ToArray();
            var source = new LakeSource($"Delta {Path.GetFileName(Root)} v{Version}", names, types, files, PartitionColumns, partitionsInFile: false);
            return new DaskFrame(new ScanNode(source));
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
                    snapshot.Files[a.GetProperty("path").GetString()!] = (
                        a.TryGetProperty("size", out var size) ? size.GetInt64() : 0,
                        PartitionValues(a),
                        a.TryGetProperty("deletionVector", out var dv) && dv.ValueKind == JsonValueKind.Object);
                    break;
                case "remove":
                    snapshot.Files.Remove(a.GetProperty("path").GetString()!);
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
                snapshot.Files[path] = (
                    Get(a, "size") is { } size ? Convert.ToInt64(size, CultureInfo.InvariantCulture) : 0,
                    StringMap(Get(a, "partitionValues")),
                    (Struct(a, "deletionVector") is { } dv ? Get(dv, "storageType") : null) is not null);
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

    internal static (string[] Names, DataType[] Types) Schema(string schemaString, string root)
    {
        using var doc = JsonDocument.Parse(schemaString);
        var names = new List<string>();
        var types = new List<DataType>();
        foreach (var field in doc.RootElement.GetProperty("fields").EnumerateArray())
        {
            var name = field.GetProperty("name").GetString()!;
            var type = field.GetProperty("type");
            if (type.ValueKind != JsonValueKind.String)
                throw new NotSupportedException($"Delta table at {root}: column '{name}' is a nested type, which is not supported.");
            names.Add(name);
            types.Add(TypeOf(type.GetString()!, name, root));
        }
        return (names.ToArray(), types.ToArray());
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
