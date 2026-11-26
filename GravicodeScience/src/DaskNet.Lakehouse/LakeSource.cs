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
/// (an Iceberg rename): table name to file name.</param>
internal sealed record LakeFile(string Path, long Size, IReadOnlyDictionary<string, string?> PartitionValues,
    IReadOnlyDictionary<string, string> FileNames);

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
    IReadOnlyCollection<string> partitionColumns, bool partitionsInFile) : ScanSource
{
    private static readonly MethodInfo ReadMethod = typeof(LakeKernels).GetMethod(nameof(LakeKernels.ReadLakeFile))!;

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
        string?[] constants = fromLog.Select(c => file!.PartitionValues.GetValueOrDefault(c)).ToArray();
        return TaskSpec.Call(new TaskKey(name, i), ReadMethod,
            [file?.Path ?? "", read, fileNames, fromLog, constants, keep, keepTypes, encoded], "read-" + label.Split(' ')[0].ToLowerInvariant());
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
    [RemoteTask]
    public static DataFrame ReadLakeFile(string path, string[] read, string[] fileNames, string[] constantNames,
        string?[] constants, string[] names, int[] types, byte[] ops)
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
            var present = Present(path);
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
        return FrameOp.ApplyAll(new DataFrame(columns), FrameOp.Decode(ops));
    }

    private static HashSet<string> Present(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = ParquetReader.CreateAsync(stream).GetAwaiter().GetResult();
        return reader.Schema.GetDataFields().Select(f => f.Name).ToHashSet(StringComparer.Ordinal);
    }

    private static long RowCount(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = ParquetReader.CreateAsync(stream).GetAwaiter().GetResult();
        return reader.Metadata?.NumRows ?? 0;
    }
}
