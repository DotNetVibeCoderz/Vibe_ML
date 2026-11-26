using System.Data.Common;
using System.Globalization;
using System.Reflection;
using System.Text;
using System.Text.Json;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;

namespace Gravicode.Science.DaskNet;

/// <summary>One partition of a <see cref="DaskFrame.ReadSql(DbProviderFactory, string, IReadOnlyList{SqlPartition})"/>
/// read: a query and its parameters.</summary>
/// <param name="Sql">The query. Put values in <paramref name="Parameters"/>, never in the text.</param>
/// <param name="Parameters">Bound by name (without the provider's prefix); values must be numbers,
/// text, dates, booleans or null so they can travel to a worker.</param>
public sealed record SqlPartition(string Sql, IReadOnlyDictionary<string, object?>? Parameters = null);

public sealed partial class DaskFrame
{
    // ------------------------------------------------------------------ JSON

    /// <summary>
    /// Reads JSON files (a path or glob). With <paramref name="lines"/> (JSON Lines, one object per
    /// line, as <c>dask.dataframe.read_json(lines=True)</c>) a file is split into byte blocks like
    /// CSV, so one large file still gives many partitions. Without it, each file is one JSON array
    /// of objects and one partition.
    /// </summary>
    /// <remarks>
    /// Columns and their types are fixed from the first <paramref name="inferenceRows"/> objects of
    /// the first file and every partition is read to that schema, so block boundaries cannot change
    /// a column's type. Properties that first appear later are ignored; a value that does not fit
    /// its column's type is missing (a numeric string in a numeric column is parsed). Nested objects
    /// and arrays are kept as their JSON text. Strings are dates only if every sampled value is an
    /// ISO-8601 date.
    /// </remarks>
    public static DaskFrame ReadJson(string pattern, bool lines = true, long blockSize = 64L << 20, int inferenceRows = 1000)
    {
        var files = GraphBuilders.ExpandPaths(pattern);
        var (names, types) = JsonFormat.InferSchema(files[0], lines, Math.Max(1, inferenceRows));
        if (!lines) return new DaskFrame(new ScanNode(new JsonSource(files.Select(f => (f, 0L, -1L)).ToArray(), names, types)));
        var blocks = new List<(string, long, long)>();
        foreach (var file in files)
            foreach (var (start, end) in GraphBuilders.ByteRanges(new FileInfo(file).Length, blockSize))
                blocks.Add((file, start, end));
        return new DaskFrame(new ScanNode(new JsonSource(blocks.ToArray(), names, types)));
    }

    /// <summary>Writes one JSON Lines file per partition (<c>part-00000.jsonl</c>, ...) and returns the paths.
    /// Missing values are <c>null</c>, dates ISO-8601.</summary>
    public string[] ToJson(string directory, IScheduler? scheduler = null)
        => Write(directory, "jsonl", nameof(FrameKernels.WriteJsonPartition), scheduler);

    // ------------------------------------------------------------------ databases

    /// <summary>
    /// Reads a table through any ADO.NET provider, split into <paramref name="partitions"/> ranges
    /// of a numeric <paramref name="index"/> column (Dask's <c>read_sql_table</c>). Rows whose index
    /// is NULL go to the last partition, so every row is read once.
    /// </summary>
    /// <param name="factory">The provider, e.g. <c>SqliteFactory.Instance</c> or <c>NpgsqlFactory.Instance</c>.
    /// Workers create their own connections from it, so a cluster worker must have the provider's
    /// assembly loaded (<c>--preload</c>).</param>
    /// <param name="connectionString">Sent to the workers with each task - use TLS on a cluster.</param>
    /// <param name="table">A plain identifier, optionally schema-qualified (<c>sales.orders</c>).</param>
    /// <param name="index">A numeric column to split on; plain identifier.</param>
    /// <param name="partitions">How many ranges.</param>
    /// <param name="bounds">The index range; when null, <c>MIN</c>/<c>MAX</c> are queried now.</param>
    /// <remarks>
    /// Only the columns a query needs are selected (projection pushdown reaches the SQL). Filters
    /// run after the rows are read. Column types come from the provider's description of an empty
    /// result (<c>WHERE 1 = 0</c>) and are pinned, so an empty range cannot come back with
    /// different types. A database that types by value - an SQLite column declared with no type -
    /// can only describe such a column as text there; read it with the query overload, whose first
    /// query is run to learn the types.
    /// </remarks>
    public static DaskFrame ReadSql(DbProviderFactory factory, string connectionString, string table, string index,
        int partitions, (double Lower, double Upper)? bounds = null)
    {
        ArgumentNullException.ThrowIfNull(factory);
        SqlSource.RequireIdentifier(table, "table");
        SqlSource.RequireIdentifier(index, "index column");
        if (partitions <= 0) throw new ArgumentOutOfRangeException(nameof(partitions));

        var schema = SqlSource.Query(factory, connectionString, $"SELECT * FROM {table} WHERE 1 = 0", null);
        if (!schema.ColumnNames.Contains(index)) throw new ArgumentException($"Table {table} has no column '{index}'.", nameof(index));
        if (bounds is null)
        {
            var range = SqlSource.Query(factory, connectionString, $"SELECT MIN({index}) AS lo, MAX({index}) AS hi FROM {table}", null);
            var lo = range["lo"].GetValue(0);
            var hi = range["hi"].GetValue(0);
            if (lo is null || hi is null) partitions = 1;   // empty, or every index NULL
            else
            {
                if (lo is not double l || hi is not double h)
                    throw new ArgumentException($"Index column '{index}' must be numeric to split into ranges; use ReadSql with queries instead.", nameof(index));
                bounds = (l, h);
            }
        }

        // Interior edges split [lower, upper] evenly. The first range is open below and the last
        // open above (and takes NULLs), so every row is read exactly once even when the caller's
        // bounds are too narrow.
        var parts = new List<(string Where, string[] Names, object?[] Values)>();
        if (bounds is not { } b || partitions == 1) parts.Add(("1 = 1", [], []));
        else
        {
            var edges = Enumerable.Range(1, partitions - 1).Select(i => b.Lower + (b.Upper - b.Lower) * i / partitions).ToArray();
            parts.Add(($"{index} < @hi", ["hi"], [edges[0]]));
            for (var i = 1; i < edges.Length; i++)
                parts.Add(($"{index} >= @lo AND {index} < @hi", ["lo", "hi"], [edges[i - 1], edges[i]]));
            parts.Add(($"{index} >= @lo OR {index} IS NULL", ["lo"], [edges[^1]]));
        }
        var source = new SqlSource(factory, connectionString, schema, table, parts.ToArray());
        return new DaskFrame(new ScanNode(source));
    }

    /// <summary>
    /// One partition per query, through any ADO.NET provider - for any split a table range cannot
    /// express (by date, by tenant, by shard). Every query must return the same columns. The first
    /// query runs once now, to learn the columns and their types, which every partition is then
    /// converted to.
    /// </summary>
    public static DaskFrame ReadSql(DbProviderFactory factory, string connectionString, IReadOnlyList<SqlPartition> queries)
    {
        ArgumentNullException.ThrowIfNull(factory);
        if (queries.Count == 0) throw new ArgumentException("Give at least one query.", nameof(queries));
        var schema = SqlSource.Query(factory, connectionString, queries[0].Sql, queries[0].Parameters);
        var parts = queries.Select(q => (q.Sql, q.Parameters?.Keys.ToArray() ?? [], q.Parameters?.Values.ToArray() ?? [])).ToArray();
        return new DaskFrame(new ScanNode(new SqlSource(factory, connectionString, schema.Head(0), null, parts)));
    }
}

/// <summary>JSON files: byte blocks of JSON Lines, or whole files holding an array (End = -1).</summary>
internal sealed class JsonSource((string Path, long Start, long End)[] blocks, string[] names, DataType[] types) : ScanSource
{
    private static readonly MethodInfo ReadMethod = typeof(FrameKernels).GetMethod(nameof(FrameKernels.ReadJsonBlock))!;

    public override IReadOnlyList<string> Columns => names;
    public override int Partitions => blocks.Length;
    public override bool Remotable => true;
    public override long? EstimatedBytes => blocks.All(b => b.End >= 0) ? blocks.Sum(b => b.End - b.Start) : CostModel.FileBytes(blocks.Select(b => b.Path));
    public override string Describe()
        => $"ReadJson {blocks.Select(b => Path.GetFileName(b.Path)).Distinct().Count()} file(s), {blocks.Length} {(blocks[0].End < 0 ? "array file(s)" : "block(s)")}";

    public override TaskSpec[] Build(string name, string[]? columns, IReadOnlyList<FrameOp> ops)
    {
        var keep = columns ?? names;
        var keepTypes = keep.Select(c => (int)types[Array.IndexOf(names, c)]).ToArray();
        var encoded = FrameOp.Encode(ops);
        return blocks.Select((b, i) => TaskSpec.Call(new TaskKey(name, i), ReadMethod,
            [b.Path, b.Start, b.End, keep, keepTypes, encoded], "read-json")).ToArray();
    }
}

/// <summary>SQL partitions: per-partition WHERE clauses on a table (projection pushed into the
/// SELECT), or whole queries.</summary>
internal sealed class SqlSource : ScanSource
{
    private static readonly MethodInfo ReadMethod = typeof(FrameKernels).GetMethod(nameof(FrameKernels.ReadSqlPartition))!;
    private readonly string _factoryType;
    private readonly string _connectionString;
    private readonly string[] _names;
    private readonly int[] _types;
    private readonly string? _table;
    private readonly (string Text, string[] Names, object?[] Values)[] _parts;

    public SqlSource(DbProviderFactory factory, string connectionString, DataFrame schema, string? table,
        (string Text, string[] Names, object?[] Values)[] parts)
    {
        _factoryType = factory.GetType().AssemblyQualifiedName!;
        _connectionString = connectionString;
        _names = schema.ColumnNames.ToArray();
        _types = schema.Columns.Select(c => (int)c.DataType).ToArray();
        _table = table;
        _parts = parts;
    }

    public override IReadOnlyList<string> Columns => _names;
    public override int Partitions => _parts.Length;
    public override bool Remotable => true;
    public override string Describe() => _table is null ? $"ReadSql {_parts.Length} quer{(_parts.Length == 1 ? "y" : "ies")}" : $"ReadSql {_table}, {_parts.Length} range(s)";

    public override TaskSpec[] Build(string name, string[]? columns, IReadOnlyList<FrameOp> ops)
    {
        var keep = columns ?? _names;
        var types = keep.Select(c => _types[Array.IndexOf(_names, c)]).ToArray();
        // A table read selects only what is needed; a caller's query is run as written.
        var pushed = _table is not null && keep.All(IsIdentifier);
        var encoded = FrameOp.Encode(pushed || columns is null ? ops : [new SelectOp(keep.Select(Expr.Col).ToArray()), .. ops]);
        return _parts.Select((p, i) =>
        {
            var sql = _table is null ? p.Text
                : $"SELECT {(pushed ? string.Join(", ", keep) : "*")} FROM {_table} WHERE {p.Text}";
            return TaskSpec.Call(new TaskKey(name, i), ReadMethod,
                [_factoryType, _connectionString, sql, p.Names, p.Values, keep, types, encoded], "read-sql");
        }).ToArray();
    }

    public static DataFrame Query(DbProviderFactory factory, string connectionString, string sql, IReadOnlyDictionary<string, object?>? parameters)
    {
        using var connection = factory.CreateConnection() ?? throw new InvalidOperationException($"{factory.GetType().Name} made no connection.");
        connection.ConnectionString = connectionString;
        return SqlReader.Read(connection, sql, parameters);
    }

    private static bool IsIdentifier(string s)
        => s.Length > 0 && (char.IsLetter(s[0]) || s[0] == '_') && s.All(c => char.IsLetterOrDigit(c) || c is '_' or '.');

    public static void RequireIdentifier(string value, string what)
    {
        if (!IsIdentifier(value))
            throw new ArgumentException($"'{value}' is not a plain identifier; a {what} is written into the SQL, so it cannot come from user input.");
    }
}

/// <summary>JSON reading and writing to a pinned schema.</summary>
internal static class JsonFormat
{
    public static (string[] Names, DataType[] Types) InferSchema(string path, bool lines, int rows)
    {
        var sample = Records(path, lines).Take(rows).ToList();
        var names = new List<string>();
        var kinds = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        foreach (var record in sample)
            foreach (var p in record.EnumerateObject())
            {
                if (!kinds.TryGetValue(p.Name, out var seen)) { names.Add(p.Name); kinds[p.Name] = seen = []; }
                seen.Add(p.Value.ValueKind switch
                {
                    JsonValueKind.Number => "number",
                    JsonValueKind.True or JsonValueKind.False => "bool",
                    JsonValueKind.String => IsIsoDate(p.Value.GetString()!) ? "date" : "string",
                    JsonValueKind.Null => "null",
                    _ => "nested",
                });
            }
        var types = names.Select(n =>
        {
            var seen = kinds[n].Where(k => k != "null").ToHashSet();
            return seen.Count switch
            {
                0 => DataType.Text,
                1 when seen.Contains("number") => DataType.Numeric,
                1 when seen.Contains("bool") => DataType.Boolean,
                1 when seen.Contains("date") => DataType.DateTime,
                _ => DataType.Text,
            };
        }).ToArray();
        return (names.ToArray(), types);
    }

    private static bool IsIsoDate(string s)
        => s.Length >= 10 && char.IsAsciiDigit(s[0]) && s[4] == '-' && s[7] == '-'
           && DateTime.TryParse(s, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out _);

    /// <summary>The objects of a file: one per non-empty line, or the elements of a top-level array.</summary>
    public static IEnumerable<JsonElement> Records(string path, bool lines)
    {
        if (lines)
        {
            foreach (var line in File.ReadLines(path))
                if (!string.IsNullOrWhiteSpace(line)) yield return Parse(line, path);
            yield break;
        }
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        if (doc.RootElement.ValueKind != JsonValueKind.Array)
            throw new InvalidDataException($"{path} is not a JSON array; read JSON Lines with lines: true.");
        foreach (var e in doc.RootElement.EnumerateArray())
        {
            if (e.ValueKind != JsonValueKind.Object) throw new InvalidDataException($"{path}: array elements must be objects.");
            yield return e.Clone();
        }
    }

    public static JsonElement Parse(string line, string path)
    {
        try
        {
            using var doc = JsonDocument.Parse(line);
            if (doc.RootElement.ValueKind != JsonValueKind.Object) throw new InvalidDataException($"{path}: each line must be a JSON object.");
            return doc.RootElement.Clone();
        }
        catch (JsonException e)
        {
            throw new InvalidDataException($"{path}: malformed JSON line starting '{(line.Length > 60 ? line[..60] + "..." : line)}': {e.Message}", e);
        }
    }

    public static DataFrame ToFrame(IReadOnlyList<JsonElement> records, string[] names, DataType[] types)
    {
        var columns = new List<Series>(names.Length);
        for (var c = 0; c < names.Length; c++)
        {
            var values = new object?[records.Count];
            for (var r = 0; r < records.Count; r++)
                values[r] = records[r].TryGetProperty(names[c], out var v) ? Convert(v, types[c]) : null;
            columns.Add(Typed.Build(names[c], types[c], values));
        }
        return new DataFrame(columns);
    }

    private static object? Convert(JsonElement v, DataType type)
    {
        if (v.ValueKind == JsonValueKind.Null) return null;
        switch (type)
        {
            case DataType.Numeric:
                if (v.ValueKind == JsonValueKind.Number) return v.GetDouble();
                if (v.ValueKind == JsonValueKind.String && double.TryParse(v.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var d)) return d;
                if (v.ValueKind is JsonValueKind.True or JsonValueKind.False) return v.GetBoolean() ? 1.0 : 0.0;
                return null;
            case DataType.Boolean:
                if (v.ValueKind is JsonValueKind.True or JsonValueKind.False) return v.GetBoolean();
                if (v.ValueKind == JsonValueKind.String && bool.TryParse(v.GetString(), out var b)) return b;
                return null;
            case DataType.DateTime:
                return v.ValueKind == JsonValueKind.String
                       && DateTime.TryParse(v.GetString(), CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out var t) ? t : null;
            default:
                return v.ValueKind switch
                {
                    JsonValueKind.String => v.GetString(),
                    _ => v.GetRawText(),
                };
        }
    }

    public static void WriteLines(DataFrame frame, string path)
    {
        using var stream = File.Create(path);
        var newline = "\n"u8.ToArray();
        for (var r = 0; r < frame.RowCount; r++)
        {
            using (var w = new Utf8JsonWriter(stream))
            {
                w.WriteStartObject();
                foreach (var column in frame.Columns)
                {
                    w.WritePropertyName(column.Name);
                    switch (column.GetValue(r))
                    {
                        case null: w.WriteNullValue(); break;
                        case double d when double.IsNaN(d) || double.IsInfinity(d): w.WriteNullValue(); break;
                        case double d: w.WriteNumberValue(d); break;
                        case bool b: w.WriteBooleanValue(b); break;
                        case DateTime t: w.WriteStringValue(t.ToString("o", CultureInfo.InvariantCulture)); break;
                        case var o: w.WriteStringValue(System.Convert.ToString(o, CultureInfo.InvariantCulture)); break;
                    }
                }
                w.WriteEndObject();
            }
            stream.Write(newline);
        }
    }
}

/// <summary>Builds a column of a pinned type from loosely typed values.</summary>
internal static class Typed
{
    public static Series Build(string name, DataType type, IReadOnlyList<object?> values)
    {
        switch (type)
        {
            case DataType.Numeric:
                return new NumericSeries(name, values.Select(v => v switch
                {
                    null => double.NaN,
                    double d => d,
                    bool b => b ? 1.0 : 0.0,
                    string s => double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var x) ? x : double.NaN,
                    DateTime => double.NaN,
                    var o => System.Convert.ToDouble(o, CultureInfo.InvariantCulture),
                }).ToArray());
            case DataType.Boolean:
                return new BooleanSeries(name, values.Select(v => v switch
                {
                    null => (bool?)null,
                    bool b => b,
                    double d => double.IsNaN(d) ? null : d != 0,
                    string s => bool.TryParse(s, out var x) ? x : null,
                    _ => null,
                }).ToArray());
            case DataType.DateTime:
                return new DateTimeSeries(name, values.Select(v => v switch
                {
                    DateTime t => t,
                    string s when DateTime.TryParse(s, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out var x) => x,
                    _ => (DateTime?)null,
                }).ToArray());
            default:
                return new TextSeries(name, values.Select(v => v switch
                {
                    null => null,
                    double d when double.IsNaN(d) => null,
                    DateTime t => t.ToString("o", CultureInfo.InvariantCulture),
                    var o => System.Convert.ToString(o, CultureInfo.InvariantCulture),
                }).ToArray());
        }
    }

    /// <summary>The frame with each column converted to its pinned type (unchanged where it matches).</summary>
    public static DataFrame Pin(DataFrame frame, string[] names, int[] types)
        => new(names.Select((n, i) =>
        {
            var column = frame[n];
            var type = (DataType)types[i];
            var actual = column.DataType == DataType.Categorical ? DataType.Text : column.DataType;
            return actual == type ? column : Build(n, type, Enumerable.Range(0, column.Length).Select(column.GetValue).ToArray());
        }));
}

public static partial class FrameKernels
{
    /// <summary>Reads a JSON Lines byte block (<paramref name="end"/> ≥ 0) or a whole JSON array file
    /// (<paramref name="end"/> = -1) to the pinned schema, then applies <paramref name="ops"/>.</summary>
    [RemoteTask]
    public static DataFrame ReadJsonBlock(string path, long start, long end, string[] names, int[] types, byte[] ops)
    {
        var records = end < 0
            ? JsonFormat.Records(path, lines: false).ToList()
            : TextBlocks.ReadLines(path, start, end).Where(l => !string.IsNullOrWhiteSpace(l)).Select(l => JsonFormat.Parse(l, path)).ToList();
        var frame = JsonFormat.ToFrame(records, names, types.Select(t => (DataType)t).ToArray());
        return FrameOp.ApplyAll(frame, FrameOp.Decode(ops));
    }

    /// <summary>Writes a partition as JSON Lines and returns the path.</summary>
    [RemoteTask]
    public static string WriteJsonPartition(DataFrame frame, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        JsonFormat.WriteLines(frame, path);
        return path;
    }

    /// <summary>
    /// Runs one partition's query on a connection made from the provider factory named by
    /// <paramref name="factoryType"/> (its static <c>Instance</c>, the ADO.NET convention), pins the
    /// column types, then applies <paramref name="ops"/>.
    /// </summary>
    [RemoteTask]
    public static DataFrame ReadSqlPartition(string factoryType, string connectionString, string sql, string[] parameterNames,
        object?[] parameterValues, string[] names, int[] types, byte[] ops)
    {
        var type = Type.GetType(factoryType)
                   ?? throw new InvalidOperationException($"ADO.NET provider '{factoryType}' is not loaded here; preload its assembly on the workers.");
        var factory = type.GetField("Instance", BindingFlags.Public | BindingFlags.Static)?.GetValue(null) as DbProviderFactory
                      ?? throw new InvalidOperationException($"{type.FullName} has no public static Instance.");
        var parameters = parameterNames.Select((n, i) => (n, parameterValues[i])).ToDictionary(p => p.n, p => p.Item2);
        var frame = SqlSource.Query(factory, connectionString, sql, parameters);
        var named = new DataFrame(frame.Columns.Where(c => names.Contains(c.Name)));
        return FrameOp.ApplyAll(Typed.Pin(named, names, types), FrameOp.Decode(ops));
    }
}
