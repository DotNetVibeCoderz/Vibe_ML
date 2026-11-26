using System.Reflection;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;
using Parquet;

namespace Gravicode.Science.DaskNet.Lakehouse;

/// <summary>One data file of a table snapshot.</summary>
/// <param name="Path">Absolute local path of the Parquet file.</param>
/// <param name="Size">Bytes on disk, for the cost model.</param>
/// <param name="PartitionValues">Partition column values as the table format stores them (text),
/// keyed by table column name. Delta keeps these out of the file; Iceberg keeps them in it.</param>
/// <param name="FileNames">Where a column has a different name in this file than in the table
/// (an Iceberg rename, a Delta mapped column): table name to file name.</param>
internal sealed record LakeFile(string Path, long Size, IReadOnlyDictionary<string, string?> PartitionValues,
    IReadOnlyDictionary<string, string> FileNames)
{
    /// <summary>Parquet field ids to match columns by, before names (Delta's column mapping by id).</summary>
    public IReadOnlyDictionary<string, int> FieldIds { get; init; } = new Dictionary<string, int>();

    /// <summary>A <see cref="DeletionVectors"/> descriptor of the rows deleted from the file, if any.</summary>
    public string? DeletionVector { get; init; }

    /// <summary>Hudi log files to merge over this base file (or over nothing: a slice that is all
    /// log), in the order they apply. Only read when the source has a <see cref="HudiMerge"/>.</summary>
    public string[] Logs { get; init; } = [];
}

/// <summary>How a Hudi merge-on-read snapshot merges a file slice's logs into its base file.</summary>
/// <param name="RecordKey">The column holding record keys (<c>_hoodie_record_key</c>).</param>
/// <param name="OrderingField">The precombine / ordering column; empty for none.</param>
/// <param name="EventTime">Event-time ordering: a record with a smaller ordering value loses.</param>
/// <param name="Instants">The completed instants a log block may belong to.</param>
internal sealed record HudiMerge(string RecordKey, string OrderingField, bool EventTime, string[] Instants);

/// <summary>
/// A table snapshot as a scan: one partition per data file, columns pinned to the table schema.
/// </summary>
/// <remarks>
/// Filters that read only partition columns are evaluated once per file against its partition
/// values; a file they rule out becomes an empty partition and is never opened. Columns are
/// projected into the Parquet read. Columns missing from an older file (added later by schema
/// evolution) read as missing values, which is what both formats specify.
/// </remarks>
internal sealed class LakeSource(string label, string[] names, DataType[] types, LakeFile[] files,
    IReadOnlyCollection<string> partitionColumns, bool partitionsInFile, HudiMerge? merge = null) : ScanSource
{
    private static readonly MethodInfo ReadMethod = typeof(LakeKernels).GetMethod(nameof(LakeKernels.ReadLakeFile))!;
    private static readonly MethodInfo SliceMethod = typeof(HudiKernels).GetMethod(nameof(HudiKernels.ReadFileSlice))!;

    public override IReadOnlyList<string> Columns => names;
    public override int Partitions => Math.Max(1, files.Length);
    public override bool Remotable => true;
    public override long? EstimatedBytes => files.Sum(f => f.Size);
    public override string Describe() => $"{label}, {files.Length} file(s)";

    public IReadOnlyList<LakeFile> Files => files;

    public override TaskSpec[] Build(string name, string[]? columns, IReadOnlyList<FrameOp> ops)
    {
        var keep = columns ?? names;
        var keepTypes = keep.Select(c => (int)types[Array.IndexOf(names, c)]).ToArray();
        var encoded = FrameOp.Encode(ops);
        if (files.Length == 0)
            return [Task(name, 0, null, keep, keepTypes, encoded)];

        var pruning = ops.TakeWhile(o => o is FilterOp).Select(o => ((FilterOp)o).Predicate)
            .SelectMany(Conjuncts)
            .Where(p => p.Columns().Count > 0 && p.Columns().All(partitionColumns.Contains))
            .ToArray();
        return files.Select((f, i) => Task(name, i, Excluded(f, pruning) ? null : f, keep, keepTypes, encoded)).ToArray();
    }

    private TaskSpec Task(string name, int i, LakeFile? file, string[] keep, int[] keepTypes, byte[] encoded)
    {
        // Delta partition columns are not in the file: they come from the log as constants.
        var fromLog = partitionsInFile || file is null ? [] : keep.Where(partitionColumns.Contains).ToArray();
        var read = keep.Except(fromLog).ToArray();
        var fileNames = read.Select(c => file?.FileNames.GetValueOrDefault(c) ?? c).ToArray();
        var fieldIds = read.Select(c => file?.FieldIds.GetValueOrDefault(c) ?? 0).ToArray();
        string?[] constants = fromLog.Select(c => file!.PartitionValues.GetValueOrDefault(c)).ToArray();
        if (merge is not null && file is { Logs.Length: > 0 })
        {
            // A merged slice needs the key and ordering columns whatever the projection keeps, so
            // it reads the whole table schema and projects after merging.
            var tableConstants = partitionsInFile ? [] : names.Where(partitionColumns.Contains).ToArray();
            string?[] tableValues = tableConstants.Select(c => file.PartitionValues.GetValueOrDefault(c)).ToArray();
            return TaskSpec.Call(new TaskKey(name, i), SliceMethod,
                [file.Path, file.Logs, merge.Instants, merge.RecordKey, merge.OrderingField, merge.EventTime,
                 tableConstants, tableValues, names, types.Select(t => (int)t).ToArray(), keep, encoded],
                "read-" + label.Split(' ')[0].ToLowerInvariant());
        }
        return TaskSpec.Call(new TaskKey(name, i), ReadMethod,
            [file?.Path ?? "", read, fileNames, fieldIds, file?.DeletionVector ?? "", fromLog, constants, keep, keepTypes, encoded],
            "read-" + label.Split(' ')[0].ToLowerInvariant());
    }

    /// <summary>True when a partition filter is not true for the file's partition values.</summary>
    private bool Excluded(LakeFile file, Expr[] pruning)
    {
        if (pruning.Length == 0) return false;
        var row = new DataFrame(partitionColumns.Select(c =>
            Typed.Build(c, types[Array.IndexOf(names, c)], [file.PartitionValues.GetValueOrDefault(c)])));
        return pruning.Any(p => FrameOp.ApplyAll(row, [new FilterOp(p)]).RowCount == 0);
    }

    private static IEnumerable<Expr> Conjuncts(Expr e) => e switch
    {
        AliasExpr a => Conjuncts(a.Inner),
        BinaryExpr { Kind: BinaryKind.And } b => Conjuncts(b.Left).Concat(Conjuncts(b.Right)),
        _ => [e],
    };
}

/// <summary>Remote kernels of the table readers.</summary>
public static class LakeKernels
{
    /// <summary>
    /// Reads one data file to the table schema: <paramref name="read"/> columns from the file
    /// (named <paramref name="fileNames"/> there), <paramref name="constants"/> for
    /// <paramref name="constantNames"/>, missing values for anything the file lacks, all converted
    /// to <paramref name="types"/>; then applies <paramref name="ops"/>. An empty path is a file
    /// that was pruned, or a table with no files: an empty frame of the right shape.
    /// </summary>
    /// <remarks>A positive entry of <c>fieldIds</c> finds that column by Parquet field id first,
    /// whatever it is called in the file; a non-empty <c>deletionVector</c> names row positions
    /// to drop before anything else runs.</remarks>
    [RemoteTask]
    public static DataFrame ReadLakeFile(string path, string[] read, string[] fileNames, int[] fieldIds, string deletionVector,
        string[] constantNames, string?[] constants, string[] names, int[] types, byte[] ops)
    {
        DataFrame frame;
        long rows;
        if (path.Length == 0)
        {
            frame = new DataFrame(Array.Empty<Series>());
            rows = 0;
        }
        else
        {
            var (present, byId) = Present(path);
            if (fieldIds.Any(id => id > 0))
            {
                fileNames = (string[])fileNames.Clone();
                for (var i = 0; i < fileNames.Length; i++)
                    if (fieldIds[i] > 0 && byId.TryGetValue(fieldIds[i], out var named)) fileNames[i] = named;
            }
            var wanted = fileNames.Where(present.Contains).Distinct().ToArray();
            frame = wanted.Length > 0 ? ParquetIO.Read(path, wanted) : new DataFrame(Array.Empty<Series>());
            rows = wanted.Length > 0 ? frame.RowCount : RowCount(path);
        }

        var count = checked((int)rows);
        var columns = names.Select((n, i) =>
        {
            var type = (DataType)types[i];
            var c = Array.IndexOf(constantNames, n);
            if (c >= 0) return Typed.Build(n, type, Enumerable.Repeat<object?>(constants[c], count).ToArray());
            var fileName = fileNames[Array.IndexOf(read, n)];
            if (frame.HasColumn(fileName)) return Typed.Build(n, type, Enumerable.Range(0, count).Select(frame[fileName].GetValue).ToArray());
            return Typed.Build(n, type, new object?[count]);
        });
        var table = new DataFrame(columns);
        if (deletionVector.Length > 0 && count > 0)
        {
            var keep = new bool[count];
            Array.Fill(keep, true);
            foreach (var row in DeletionVectors.Read(deletionVector))
            {
                if (row >= count) throw new InvalidDataException($"{path} has {count} rows; its deletion vector deletes row {row}.");
                keep[row] = false;
            }
            table = table.Filter(keep);
        }
        return FrameOp.ApplyAll(table, FrameOp.Decode(ops));
    }

    private static (HashSet<string> Names, Dictionary<int, string> ById) Present(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = ParquetReader.CreateAsync(stream).GetAwaiter().GetResult();
        var fields = reader.Schema.GetDataFields();
        var byId = new Dictionary<int, string>();
        foreach (var f in fields)
            if (f.FieldId is int id && id > 0) byId.TryAdd(id, f.Name);
        return (fields.Select(f => f.Name).ToHashSet(StringComparer.Ordinal), byId);
    }

    private static long RowCount(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = ParquetReader.CreateAsync(stream).GetAwaiter().GetResult();
        return reader.Metadata?.NumRows ?? 0;
    }
}
