using System.ComponentModel;
using System.Globalization;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;

namespace Gravicode.Science.DaskNet;

/// <summary>
/// The partition-level kernels of <see cref="DaskFrame"/>. Public and
/// <see cref="RemoteTaskAttribute"/> so a cluster worker can run them; not meant to be called directly.
/// </summary>
/// <remarks>Operation lists and aggregation specs arrive as bytes (see <see cref="FrameOp"/>):
/// that is the one form every transport carries unchanged.</remarks>
[EditorBrowsable(EditorBrowsableState.Never)]
public static partial class FrameKernels
{
    // ------------------------------------------------------------------ sources

    /// <summary>
    /// Reads the rows of one byte range of a CSV file, parsing only the columns asked for, then
    /// applies the fused operations.
    /// </summary>
    /// <param name="path">The file.</param>
    /// <param name="start">First byte of the block; 0 means the block holds the header line.</param>
    /// <param name="end">One past the last byte; lines starting before it belong to the block.</param>
    /// <param name="names">Every column of the file, in order.</param>
    /// <param name="types">Each column's <see cref="DataType"/> as an int, pinned from a sample so
    /// every block agrees.</param>
    /// <param name="keep">Indices of the columns to parse (projection pushdown).</param>
    /// <param name="delimiter">The field separator.</param>
    /// <param name="missing">Tokens read as missing, beside the empty string.</param>
    /// <param name="ops">Encoded row-wise operations applied after parsing.</param>
    [RemoteTask]
    public static DataFrame ReadCsvBlock(string path, long start, long end, string[] names, int[] types, int[] keep,
        string delimiter, string[] missing, byte[] ops)
    {
        var lines = TextBlocks.ReadLines(path, start, end);
        if (start == 0 && lines.Count > 0) lines.RemoveAt(0); // the header
        var missingSet = new HashSet<string>(missing, StringComparer.OrdinalIgnoreCase);

        var rows = new List<string[]>(lines.Count);
        foreach (var line in lines)
            if (line.Length > 0) rows.Add(CsvReader.SplitLine(line, delimiter));

        var columns = new List<Series>(keep.Length);
        foreach (var j in keep)
        {
            var raw = new string?[rows.Count];
            for (var i = 0; i < rows.Count; i++)
            {
                var v = j < rows[i].Length ? rows[i][j] : null;
                raw[i] = string.IsNullOrWhiteSpace(v) || missingSet.Contains(v.Trim()) ? null : v;
            }
            columns.Add(Convert(names[j], raw, (DataType)types[j]));
        }
        return FrameOp.ApplyAll(new DataFrame(columns), FrameOp.Decode(ops));
    }

    /// <summary>Reads an Arrow IPC file and applies the fused operations.</summary>
    [RemoteTask]
    public static DataFrame ReadArrow(string path, string[] columns, byte[] ops)
    {
        var frame = ArrowFile.Read(path);
        if (columns.Length > 0) frame = frame.SelectColumns(columns);
        return FrameOp.ApplyAll(frame, FrameOp.Decode(ops));
    }

    /// <summary>Applies encoded row-wise operations to a partition.</summary>
    [RemoteTask]
    public static DataFrame ApplyOps(DataFrame frame, byte[] ops) => FrameOp.ApplyAll(frame, FrameOp.Decode(ops));

    // ------------------------------------------------------------------ aggregation

    /// <summary>One partition's per-group aggregation states.</summary>
    [RemoteTask]
    public static DataFrame PartialAggregate(DataFrame frame, byte[] spec)
    {
        var (keys, aggs) = Aggregation.Decode(spec);
        var (first, rows) = Grouping.Group(frame, keys);
        if (keys.Length > 0 && frame.RowCount == 0) { first.Clear(); rows.Clear(); }

        var columns = new List<Series>();
        foreach (var k in keys) columns.Add(frame[k].Take(first));

        for (var a = 0; a < aggs.Length; a++)
        {
            var agg = aggs[a];
            var values = agg.Column is null ? null : Value.FromSeries(frame[agg.Column]);
            var source = agg.Column is null ? null : frame[agg.Column];
            var g = rows.Count;
            var count = new double[g];
            var sum = new double[g];
            var mean = new double[g];
            var m2 = new double[g];
            var min = new double[g];
            var max = new double[g];

            for (var i = 0; i < g; i++)
            {
                if (agg.How == "size") { count[i] = rows[i].Count; continue; }
                if (agg.How == "count") { count[i] = rows[i].Count(r => !source!.IsMissing(r)); continue; }

                var data = values!.Kind is DataType.Numeric or DataType.Boolean
                    ? rows[i].Select(values.NumberAt).Where(x => !double.IsNaN(x)).ToArray()
                    : throw new InvalidOperationException($"Aggregation '{agg.How}' needs a numeric column; '{agg.Column}' is {values.Kind}.");
                var m = Moments.Of(data);
                count[i] = m.Count;
                mean[i] = m.Mean;
                m2[i] = m.M2;
                sum[i] = m.Count * m.Mean;
                min[i] = m.Count == 0 ? double.PositiveInfinity : data.Min();
                max[i] = m.Count == 0 ? double.NegativeInfinity : data.Max();
                if (agg.How == "sum") sum[i] = KahanSum(data);
            }

            columns.Add(new NumericSeries($"__{a}_c", count));
            columns.Add(new NumericSeries($"__{a}_s", sum));
            columns.Add(new NumericSeries($"__{a}_m", mean));
            columns.Add(new NumericSeries($"__{a}_m2", m2));
            columns.Add(new NumericSeries($"__{a}_min", min));
            columns.Add(new NumericSeries($"__{a}_max", max));
        }
        return new DataFrame(columns);
    }

    /// <summary>Merges aggregation states from several partitions, in order.</summary>
    [RemoteTask]
    public static DataFrame CombineAggregate(DataFrame[] parts, byte[] spec)
    {
        var (keys, aggs) = Aggregation.Decode(spec);
        var nonEmpty = parts.Where(p => p.RowCount > 0).ToArray();
        var all = DataFrame.Concat(nonEmpty.Length > 0 ? nonEmpty : [parts[0]]);
        var (first, rows) = Grouping.Group(all, keys);
        if (all.RowCount == 0) return all;

        var columns = new List<Series>();
        foreach (var k in keys) columns.Add(all[k].Take(first));
        for (var a = 0; a < aggs.Length; a++)
        {
            var c = all.Numeric($"__{a}_c").Values;
            var s = all.Numeric($"__{a}_s").Values;
            var m = all.Numeric($"__{a}_m").Values;
            var q = all.Numeric($"__{a}_m2").Values;
            var lo = all.Numeric($"__{a}_min").Values;
            var hi = all.Numeric($"__{a}_max").Values;
            var g = rows.Count;
            var (rc, rs, rm, rq, rlo, rhi) = (new double[g], new double[g], new double[g], new double[g], new double[g], new double[g]);
            for (var i = 0; i < g; i++)
            {
                var moments = new Moments(0, 0, 0);
                double total = 0, comp = 0, lowest = double.PositiveInfinity, highest = double.NegativeInfinity, n = 0;
                foreach (var r in rows[i])
                {
                    n += c[r];
                    var y = s[r] - comp; var t = total + y; comp = (t - total) - y; total = t;
                    moments = Moments.Merge(moments, new Moments((long)c[r], m[r], q[r]));
                    lowest = Math.Min(lowest, lo[r]);
                    highest = Math.Max(highest, hi[r]);
                }
                (rc[i], rs[i], rm[i], rq[i], rlo[i], rhi[i]) = (n, total, moments.Mean, moments.M2, lowest, highest);
            }
            columns.Add(new NumericSeries($"__{a}_c", rc));
            columns.Add(new NumericSeries($"__{a}_s", rs));
            columns.Add(new NumericSeries($"__{a}_m", rm));
            columns.Add(new NumericSeries($"__{a}_m2", rq));
            columns.Add(new NumericSeries($"__{a}_min", rlo));
            columns.Add(new NumericSeries($"__{a}_max", rhi));
        }
        return new DataFrame(columns);
    }

    /// <summary>Turns merged states into result columns, sorted by key.</summary>
    [RemoteTask]
    public static DataFrame FinishAggregate(DataFrame states, byte[] spec)
    {
        var (keys, aggs) = Aggregation.Decode(spec);
        var columns = new List<Series>();
        foreach (var k in keys) columns.Add(states[k]);
        var rows = states.RowCount;

        // A global aggregate over no rows still has one row, like SQL's COUNT(*) on an empty table.
        if (keys.Length == 0 && rows == 0)
        {
            return new DataFrame(aggs.Select(a => (Series)new NumericSeries(a.Output,
                [a.How is "count" or "size" or "sum" ? 0 : double.NaN])));
        }

        for (var a = 0; a < aggs.Length; a++)
        {
            var c = states.Numeric($"__{a}_c").Values;
            var s = states.Numeric($"__{a}_s").Values;
            var m = states.Numeric($"__{a}_m").Values;
            var q = states.Numeric($"__{a}_m2").Values;
            var lo = states.Numeric($"__{a}_min").Values;
            var hi = states.Numeric($"__{a}_max").Values;
            var result = new double[rows];
            for (var i = 0; i < rows; i++)
            {
                var moments = new Moments((long)c[i], m[i], q[i]);
                result[i] = aggs[a].How switch
                {
                    "sum" => s[i],
                    "mean" => c[i] == 0 ? double.NaN : m[i],
                    "count" or "size" => c[i],
                    "min" => c[i] == 0 ? double.NaN : lo[i],
                    "max" => c[i] == 0 ? double.NaN : hi[i],
                    "var" => moments.Variance(1),
                    "std" => Math.Sqrt(moments.Variance(1)),
                    "var_pop" => moments.Variance(0),
                    _ => Math.Sqrt(moments.Variance(0)),
                };
            }
            columns.Add(new NumericSeries(aggs[a].Output, result));
        }

        var frame = new DataFrame(columns);
        if (keys.Length == 0 || rows <= 1) return frame;
        var order = Enumerable.Range(0, rows).ToArray();
        var keySeries = keys.Select(k => frame[k]).ToArray();
        Array.Sort(order, (x, y) =>
        {
            foreach (var k in keySeries)
            {
                var c = Grouping.CompareValues(k.GetValue(x), k.GetValue(y));
                if (c != 0) return c;
            }
            return x.CompareTo(y);
        });
        return frame.Take(order);
    }

    // ------------------------------------------------------------------ counting

    /// <summary>A partition's row count.</summary>
    [RemoteTask]
    public static long CountRows(DataFrame frame) => frame.RowCount;

    /// <summary>The sum of several counts.</summary>
    [RemoteTask]
    public static long SumCounts(long[] counts) => counts.Sum();

    // ------------------------------------------------------------------ reshaping

    /// <summary>Concatenates partitions, in order, skipping empty ones that would only cost a copy.</summary>
    [RemoteTask]
    public static DataFrame ConcatFrames(DataFrame[] parts)
    {
        var nonEmpty = parts.Where(p => p.RowCount > 0).ToArray();
        return nonEmpty.Length switch
        {
            0 => parts.Length > 0 ? parts[0] : DataFrame.Empty(),
            1 => nonEmpty[0],
            _ => DataFrame.Concat(nonEmpty),
        };
    }

    /// <summary>The first <paramref name="count"/> rows.</summary>
    [RemoteTask]
    public static DataFrame HeadFrame(DataFrame frame, int count) => frame.RowCount <= count ? frame : frame.Head(count);

    /// <summary>Splits a partition into <paramref name="pieces"/> consecutive pieces of near-equal size.</summary>
    [RemoteTask]
    public static DataFrame[] SplitFrame(DataFrame frame, int pieces)
        => Enumerable.Range(0, pieces).Select(p =>
        {
            var (start, count) = Bag.Slice(frame.RowCount, pieces, p);
            return frame.Rows(start, count);
        }).ToArray();

    /// <summary>One element of an array of partitions.</summary>
    [RemoteTask]
    public static DataFrame PickFrame(DataFrame[] frames, int index) => frames[index];

    /// <summary>Splits a partition by the stable hash of its key columns.</summary>
    [RemoteTask]
    public static DataFrame[] HashSplit(DataFrame frame, string[] keys, int buckets)
    {
        var series = keys.Select(k => frame[k]).ToArray();
        var lists = new List<int>[buckets];
        for (var b = 0; b < buckets; b++) lists[b] = [];
        for (var i = 0; i < frame.RowCount; i++)
            lists[(int)(Grouping.StableHash(Grouping.Signature(series, i)) % (ulong)buckets)].Add(i);
        return lists.Select(l => frame.Take(l)).ToArray();
    }

    /// <summary>Joins two partitions (GraviFrame's hash join).</summary>
    [RemoteTask]
    public static DataFrame JoinFrames(DataFrame left, DataFrame right, string[] leftOn, string[] rightOn, int kind, string suffix)
    {
        if (leftOn.Length == 1) return Joins.Merge(left, right, leftOn[0], rightOn[0], (JoinKind)kind, suffix);
        if (!leftOn.SequenceEqual(rightOn))
            throw new NotSupportedException("A join on several keys needs the same key names on both sides.");
        return Joins.MergeOn(left, right, leftOn, (JoinKind)kind, suffix);
    }

    /// <summary>Up to <paramref name="count"/> evenly spaced values of a column, for picking
    /// range-partition boundaries.</summary>
    [RemoteTask]
    public static DataFrame SampleColumn(DataFrame frame, string column, int count)
    {
        var present = Enumerable.Range(0, frame.RowCount).Where(i => !frame[column].IsMissing(i)).ToArray();
        if (present.Length <= count) return new DataFrame([frame[column].Take(present)]);
        var step = (double)present.Length / count;
        return new DataFrame([frame[column].Take(Enumerable.Range(0, count).Select(i => present[(int)(i * step)]).ToArray())]);
    }

    /// <summary>Splits a partition into ranges by sorted boundary values; missing keys go last.</summary>
    [RemoteTask]
    public static DataFrame[] RangeSplit(DataFrame frame, string column, DataFrame boundaries, bool ascending)
    {
        var bounds = boundaries[column];
        var values = Enumerable.Range(0, bounds.Length).Select(bounds.GetValue).ToArray();
        var buckets = values.Length + 1;
        var lists = new List<int>[buckets];
        for (var b = 0; b < buckets; b++) lists[b] = [];
        var key = frame[column];
        for (var i = 0; i < frame.RowCount; i++)
        {
            var v = key.GetValue(i);
            int bucket;
            if (v is null) bucket = buckets - 1;
            else
            {
                // The number of boundaries strictly below v: equal keys share a bucket.
                int lo = 0, hi = values.Length;
                while (lo < hi)
                {
                    var mid = (lo + hi) / 2;
                    if (Grouping.CompareValues(values[mid], v) < 0) lo = mid + 1; else hi = mid;
                }
                bucket = ascending ? lo : values.Length - lo;
            }
            lists[bucket].Add(i);
        }
        return lists.Select(l => frame.Take(l)).ToArray();
    }

    /// <summary>Sorts a partition by several keys; missing keys last, stable.</summary>
    [RemoteTask]
    public static DataFrame SortFrame(DataFrame frame, string[] columns, bool[] ascending)
    {
        var order = Enumerable.Range(0, frame.RowCount).ToArray();
        var keys = columns.Select(c => frame[c]).ToArray();
        var comparer = Comparer<int>.Create((x, y) =>
        {
            for (var k = 0; k < keys.Length; k++)
            {
                var a = keys[k].GetValue(x);
                var b = keys[k].GetValue(y);
                if (a is null || b is null)
                {
                    if (a is null && b is null) continue;
                    return a is null ? 1 : -1;
                }
                var c = Grouping.CompareValues(a, b);
                if (c != 0) return ascending[k] ? c : -c;
            }
            return x.CompareTo(y);
        });
        Array.Sort(order, comparer);
        return frame.Take(order);
    }

    // ------------------------------------------------------------------ output

    /// <summary>Writes a partition to CSV and returns the path.</summary>
    [RemoteTask]
    public static string WriteCsvPartition(DataFrame frame, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        frame.WriteCsv(path);
        return path;
    }

    /// <summary>Writes a partition to Arrow IPC (written to a temporary name and renamed, so a
    /// crash never leaves a truncated file that a resumed run would trust) and returns the path.</summary>
    [RemoteTask]
    public static string WriteArrowPartition(DataFrame frame, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        var staging = path + ".partial";
        ArrowFile.Write(frame, staging);
        File.Move(staging, path, overwrite: true);
        return path;
    }

    // ------------------------------------------------------------------ helpers

    internal static Series Convert(string name, string?[] raw, DataType type)
    {
        switch (type)
        {
            case DataType.Numeric:
            {
                var values = new double[raw.Length];
                for (var i = 0; i < raw.Length; i++)
                    values[i] = raw[i] is not null && double.TryParse(raw[i], NumberStyles.Any, CultureInfo.InvariantCulture, out var v)
                        ? v : double.NaN;
                return new NumericSeries(name, values);
            }
            case DataType.Boolean:
            {
                var values = new bool?[raw.Length];
                for (var i = 0; i < raw.Length; i++)
                    values[i] = raw[i]?.Trim().ToLowerInvariant() switch
                    {
                        "true" or "yes" or "y" or "t" => true,
                        "false" or "no" or "n" or "f" => false,
                        _ => null,
                    };
                return new BooleanSeries(name, values);
            }
            case DataType.DateTime:
            {
                var values = new DateTime?[raw.Length];
                for (var i = 0; i < raw.Length; i++)
                    values[i] = raw[i] is not null && DateTime.TryParse(raw[i], CultureInfo.InvariantCulture, DateTimeStyles.None, out var d)
                        ? d : null;
                return new DateTimeSeries(name, values);
            }
            default:
                return new TextSeries(name, raw);
        }
    }

    private static double KahanSum(IEnumerable<double> values)
    {
        double sum = 0, c = 0;
        foreach (var v in values)
        {
            var y = v - c;
            var t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        return sum;
    }
}
