using System.Globalization;
using System.Text.Json;
using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet.Lakehouse;

/// <summary>One snapshot of an Iceberg table.</summary>
public sealed record IcebergSnapshot(long Id, long? ParentId, DateTimeOffset Timestamp, string? Operation, int? SchemaId);

/// <summary>
/// Apache Iceberg tables (format versions 1 and 2) as partitioned <see cref="DaskFrame"/>s: one
/// partition per live data file of the chosen snapshot, identity-partition filters pruning files.
/// </summary>
/// <remarks>
/// <para>
/// The table is located through its metadata: <c>metadata/version-hint.text</c> when present,
/// otherwise the newest <c>*.metadata.json</c>, or a metadata file passed directly. The snapshot's
/// manifest list (Avro) names manifests, whose entries name the data files; entries marked deleted
/// are skipped. Reads use the schema the snapshot was written with, so time travel shows the
/// columns that existed then.
/// </para>
/// <para>
/// Columns are matched by field id, not by name: a manifest records the schema its files were
/// written with, so a column renamed since is read under its old name and comes back under its
/// new one. Paths in the metadata are absolute; anything under the table's recorded location is
/// resolved against where the table actually is, so a copied or moved table still reads.
/// </para>
/// <para>
/// Refused rather than misread: delete files (merge-on-read position or equality deletes), data
/// files that are not Parquet, and nested column types. Local paths only.
/// </para>
/// </remarks>
public static class Iceberg
{
    /// <summary>Reads the table at its current snapshot, or at <paramref name="snapshotId"/>.</summary>
    /// <param name="path">The table directory (holding <c>metadata/</c>) or a <c>*.metadata.json</c> file.</param>
    /// <param name="snapshotId">A snapshot id from <see cref="Snapshots"/>; the current one when null.</param>
    public static DaskFrame Read(string path, long? snapshotId = null)
    {
        var table = Table.Load(path);
        var id = snapshotId ?? table.CurrentSnapshotId;
        return table.Frame(id);
    }

    /// <summary>Reads the table as it was at <paramref name="asOf"/>: the newest snapshot taken at or before it.</summary>
    public static DaskFrame Read(string path, DateTimeOffset asOf)
    {
        var snapshot = Snapshots(path).Where(s => s.Timestamp <= asOf).MaxBy(s => s.Timestamp)
            ?? throw new ArgumentException($"The table at {path} has no snapshot at or before {asOf:o}.", nameof(asOf));
        return Read(path, snapshot.Id);
    }

    /// <summary>The table's snapshots, oldest first.</summary>
    public static IReadOnlyList<IcebergSnapshot> Snapshots(string path) => Table.Load(path).Snapshots;

    internal sealed class Table
    {
        private readonly JsonElement _metadata;
        private readonly string _root;
        private readonly string _location;

        private Table(JsonElement metadata, string root)
        {
            _metadata = metadata;
            _root = root;
            _location = Normalize(metadata.GetProperty("location").GetString()!);
            var format = metadata.TryGetProperty("format-version", out var fv) ? fv.GetInt32() : 1;
            if (format > 2) throw new NotSupportedException($"Iceberg table at {root} is format version {format}; versions 1 and 2 are supported.");
        }

        public long? CurrentSnapshotId => _metadata.TryGetProperty("current-snapshot-id", out var c) && c.ValueKind == JsonValueKind.Number && c.GetInt64() >= 0
            ? c.GetInt64() : null;

        public IReadOnlyList<IcebergSnapshot> Snapshots =>
            _metadata.TryGetProperty("snapshots", out var s)
                ? s.EnumerateArray().Select(e => new IcebergSnapshot(
                    e.GetProperty("snapshot-id").GetInt64(),
                    e.TryGetProperty("parent-snapshot-id", out var p) && p.ValueKind == JsonValueKind.Number ? p.GetInt64() : null,
                    DateTimeOffset.FromUnixTimeMilliseconds(e.GetProperty("timestamp-ms").GetInt64()),
                    e.TryGetProperty("summary", out var sum) && sum.TryGetProperty("operation", out var op) ? op.GetString() : null,
                    e.TryGetProperty("schema-id", out var sid) ? sid.GetInt32() : null))
                    .OrderBy(x => x.Timestamp).ThenBy(x => x.Id).ToList()
                : [];

        public static Table Load(string path)
        {
            var full = Path.GetFullPath(path);
            string file;
            if (File.Exists(full)) file = full;
            else
            {
                var dir = Path.Combine(full, "metadata");
                if (!Directory.Exists(dir)) throw new DirectoryNotFoundException($"{path} is not an Iceberg table: it has no metadata directory.");
                var hint = Path.Combine(dir, "version-hint.text");
                file = File.Exists(hint) && int.TryParse(File.ReadAllText(hint).Trim(), CultureInfo.InvariantCulture, out var v)
                       && File.Exists(Path.Combine(dir, $"v{v}.metadata.json"))
                    ? Path.Combine(dir, $"v{v}.metadata.json")
                    : Directory.EnumerateFiles(dir, "*.metadata.json").MaxBy(Sequence)
                      ?? throw new FileNotFoundException($"{dir} holds no *.metadata.json.");
            }
            var root = Path.GetDirectoryName(Path.GetDirectoryName(file))!;
            using var doc = JsonDocument.Parse(File.ReadAllText(file));
            return new Table(doc.RootElement.Clone(), root);
        }

        /// <summary>00012-uuid.metadata.json and v12.metadata.json both sort by 12.</summary>
        private static long Sequence(string file)
        {
            var name = Path.GetFileName(file);
            var digits = new string(name.SkipWhile(c => c == 'v').TakeWhile(char.IsAsciiDigit).ToArray());
            return long.TryParse(digits, CultureInfo.InvariantCulture, out var n) ? n : -1;
        }

        public DaskFrame Frame(long? snapshotId)
        {
            JsonElement? snapshot = null;
            if (snapshotId is { } id)
            {
                snapshot = _metadata.GetProperty("snapshots").EnumerateArray()
                    .Cast<JsonElement?>().FirstOrDefault(s => s!.Value.GetProperty("snapshot-id").GetInt64() == id)
                    ?? throw new ArgumentException($"Iceberg table at {_root} has no snapshot {id}.", nameof(snapshotId));
            }

            var schemaId = snapshot is { } sn && sn.TryGetProperty("schema-id", out var sid) ? sid.GetInt32()
                : _metadata.TryGetProperty("current-schema-id", out var cur) ? cur.GetInt32() : (int?)null;
            var schema = SchemaById(schemaId);
            var fields = Fields(schema);
            var names = fields.Select(f => f.Name).ToArray();
            var types = fields.Select(f => f.Type).ToArray();

            var files = new List<LakeFile>();
            var partitionColumns = new HashSet<string>(StringComparer.Ordinal);
            if (snapshot is { } s)
                foreach (var manifest in ManifestList(s))
                    files.AddRange(DataFiles(manifest, fields, partitionColumns));

            var label = $"Iceberg {Path.GetFileName(_root)} {(snapshotId is { } x ? x.ToString(CultureInfo.InvariantCulture) : "empty")}";
            return new DaskFrame(new ScanNode(new LakeSource(label, names, types, files.ToArray(), partitionColumns, partitionsInFile: true)));
        }

        private JsonElement SchemaById(int? id)
        {
            if (_metadata.TryGetProperty("schemas", out var schemas))
            {
                foreach (var s in schemas.EnumerateArray())
                    if (id is null || s.GetProperty("schema-id").GetInt32() == id) return s;
            }
            return _metadata.GetProperty("schema");   // format v1
        }

        internal static (int Id, string Name, DataType Type)[] Fields(JsonElement schema)
            => schema.GetProperty("fields").EnumerateArray().Select(f =>
            {
                var name = f.GetProperty("name").GetString()!;
                var type = f.GetProperty("type");
                if (type.ValueKind != JsonValueKind.String)
                    throw new NotSupportedException($"Iceberg column '{name}' is a nested type, which is not supported.");
                return (f.GetProperty("id").GetInt32(), name, TypeOf(type.GetString()!, name));
            }).ToArray();

        private static DataType TypeOf(string type, string column) => type switch
        {
            "int" or "long" or "float" or "double" => DataType.Numeric,
            _ when type.StartsWith("decimal", StringComparison.Ordinal) => DataType.Numeric,
            "string" or "uuid" => DataType.Text,
            "boolean" => DataType.Boolean,
            "date" or "timestamp" or "timestamptz" or "timestamp_ns" or "timestamptz_ns" => DataType.DateTime,
            _ => throw new NotSupportedException($"Iceberg column '{column}' has type '{type}', which is not supported."),
        };

        private IEnumerable<string> ManifestList(JsonElement snapshot)
        {
            if (snapshot.TryGetProperty("manifest-list", out var list))
            {
                foreach (var record in Avro.Read(Resolve(list.GetString()!)).Records.Cast<Dictionary<string, object?>>())
                {
                    if (record.TryGetValue("content", out var content) && content is long c && c != 0)
                    {
                        var added = record.GetValueOrDefault("added_files_count") as long? ?? 0;
                        var existing = record.GetValueOrDefault("existing_files_count") as long? ?? 0;
                        if (added + existing > 0)
                            throw new NotSupportedException($"Iceberg table at {_root} has delete files (merge-on-read deletes), which are not supported.");
                        continue;
                    }
                    yield return Resolve((string)record["manifest_path"]!);
                }
            }
            else if (snapshot.TryGetProperty("manifests", out var manifests))   // format v1 may inline them
                foreach (var m in manifests.EnumerateArray()) yield return Resolve(m.GetString()!);
        }

        private IEnumerable<LakeFile> DataFiles(string manifest, (int Id, string Name, DataType Type)[] fields, HashSet<string> partitionColumns)
        {
            var (metadata, records) = Avro.Read(manifest);
            if (metadata.TryGetValue("content", out var content) && content == "deletes")
            {
                if (records.Count > 0)
                    throw new NotSupportedException($"Iceberg table at {_root} has delete files (merge-on-read deletes), which are not supported.");
                yield break;
            }

            // The schema these files were written with: names by field id, for renames since.
            var written = metadata.TryGetValue("schema", out var s)
                ? Fields(JsonDocument.Parse(s).RootElement).ToDictionary(f => f.Id, f => f.Name) : [];
            var renames = fields.Where(f => written.TryGetValue(f.Id, out var old) && old != f.Name)
                .ToDictionary(f => f.Name, f => written[f.Id]);

            // Identity partitions name a source column; their values can prune files.
            var identity = new Dictionary<string, string>(StringComparer.Ordinal);
            if (metadata.TryGetValue("partition-spec", out var spec))
                foreach (var p in JsonDocument.Parse(spec).RootElement.EnumerateArray())
                    if (p.GetProperty("transform").GetString() == "identity"
                        && fields.FirstOrDefault(f => f.Id == p.GetProperty("source-id").GetInt32()) is { Name: { } source })
                        identity[p.GetProperty("name").GetString()!] = source;

            foreach (var entry in records.Cast<Dictionary<string, object?>>())
            {
                if (entry["status"] is long status && status == 2) continue;   // DELETED
                var file = (Dictionary<string, object?>)entry["data_file"]!;
                if (file.GetValueOrDefault("content") is long c && c != 0)
                    throw new NotSupportedException($"Iceberg table at {_root} has delete files (merge-on-read deletes), which are not supported.");
                var format = (string)file["file_format"]!;
                if (!format.Equals("PARQUET", StringComparison.OrdinalIgnoreCase))
                    throw new NotSupportedException($"Iceberg data file {file["file_path"]} is {format}; only Parquet is supported.");

                var values = new Dictionary<string, string?>(StringComparer.Ordinal);
                if (file.GetValueOrDefault("partition") is Dictionary<string, object?> partition)
                    foreach (var (name, value) in partition)
                        if (identity.TryGetValue(name, out var source))
                        {
                            values[source] = value switch
                            {
                                null => null,
                                IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
                                var o => o.ToString(),
                            };
                            partitionColumns.Add(source);
                        }
                yield return new LakeFile(Resolve((string)file["file_path"]!), file.GetValueOrDefault("file_size_in_bytes") as long? ?? 0,
                    values, renames);
            }
        }

        /// <summary>A path recorded under the table's location, re-rooted where the table is now.</summary>
        private string Resolve(string recorded)
        {
            var path = Normalize(recorded);
            if (path.StartsWith(_location, StringComparison.OrdinalIgnoreCase))
                path = _root + path[_location.Length..];
            return Path.GetFullPath(path.Replace('/', Path.DirectorySeparatorChar));
        }

        private static string Normalize(string path)
        {
            if (path.StartsWith("file:", StringComparison.OrdinalIgnoreCase))
            {
                path = Uri.UnescapeDataString(path["file:".Length..]).TrimStart('/');
                // file:///C:/x -> C:/x on Windows; file:///data/x -> /data/x elsewhere.
                if (!(path.Length > 1 && path[1] == ':')) path = "/" + path;
            }
            return path.Replace('\\', '/').TrimEnd('/');
        }
    }
}
