using System.Globalization;
using Gravicode.Science.GraviFrame.Io;

namespace Gravicode.Science.GraviFrame;

/// <summary>
/// A frame read in bounded pieces, so the whole of it never has to be in memory at once.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="DataFrame"/> holds every column as an array, so a frame is capped by RAM.
/// <c>MemoryMappedArray</c> already lifts that cap for a single array; this lifts it for a table,
/// by never materialising more than <see cref="ChunkRows"/> rows at a time.
/// </para>
/// <para>
/// What this is <b>not</b> is a lazy query engine. There is no optimiser, no predicate pushdown and
/// no plan: a chunked frame is an <see cref="IEnumerable{T}"/> of ordinary frames, and the
/// operations in <see cref="Streaming"/> are hand-written single passes over it. That keeps the
/// cost model obvious, which matters more here than cleverness — the reason to reach for this is
/// that the data does not fit, and a surprise materialisation defeats the entire purpose.
/// </para>
/// <para>
/// <b>Chunks are produced on demand and not retained.</b> Enumerating twice re-reads the source,
/// which is the right default for something too large to keep, and the reason every operation here
/// is written as one pass.
/// </para>
/// </remarks>
public sealed class ChunkedFrame
{
    private readonly Func<IEnumerable<DataFrame>> _source;

    private ChunkedFrame(Func<IEnumerable<DataFrame>> source, int chunkRows, IReadOnlyList<string> columns)
    {
        _source = source;
        ChunkRows = chunkRows;
        ColumnNames = columns;
    }

    /// <summary>How many rows a chunk holds at most.</summary>
    public int ChunkRows { get; }

    /// <summary>The column names, read from the source's header without loading any data.</summary>
    public IReadOnlyList<string> ColumnNames { get; }

    /// <summary>The chunks, produced on demand.</summary>
    public IEnumerable<DataFrame> Chunks => _source();

    /// <summary>
    /// Reads a CSV file in chunks.
    /// </summary>
    /// <param name="path">The file to read.</param>
    /// <param name="chunkRows">Rows per chunk. Larger is faster and uses more memory.</param>
    /// <param name="options">Parsing options. <c>MaxRows</c> is ignored, since chunking sets it.</param>
    /// <remarks>
    /// <para>
    /// <b>Column types are inferred once, from a sample, and then pinned for every chunk.</b>
    /// Inferring per chunk would be simpler and is a trap: a column that parses as numeric for the
    /// first million rows and turns textual later would come back numeric in the early chunks and
    /// textual in the late ones, so the same query would give different answers at different chunk
    /// sizes. Pinning makes the result independent of the chunk size, which is the only defensible
    /// behaviour when the chunk size is a memory knob rather than part of the question.
    /// </para>
    /// <para>
    /// The sample is the first <c>TypeInferenceRows</c> data rows. That is still a guess — a file
    /// too large to load is also too large to inspect — so pass explicit <c>ColumnTypes</c> when
    /// the schema is known. Anything the caller specifies wins over the sample.
    /// </para>
    /// </remarks>
    public static ChunkedFrame FromCsv(string path, int chunkRows = 100_000, CsvOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (chunkRows <= 0) throw new ArgumentOutOfRangeException(nameof(chunkRows));

        options ??= new CsvOptions();

        var header = ReadHeader(path, options);
        var pinned = PinTypes(path, options, header);

        return new ChunkedFrame(() => ReadChunks(path, chunkRows, pinned, header), chunkRows, header);
    }

    /// <summary>
    /// Reads a sample of the file, infers each column's type from it, and returns options that
    /// state those types explicitly.
    /// </summary>
    private static CsvOptions PinTypes(string path, CsvOptions options, string[] header)
    {
        var sampleRows = options.TypeInferenceRows > 0 ? options.TypeInferenceRows : 1000;

        var sample = CsvReader.Read(path, new CsvOptions
        {
            Delimiter = options.Delimiter,
            HasHeader = options.HasHeader,
            ColumnNames = options.ColumnNames,
            TypeInferenceRows = sampleRows,
            MissingTokens = options.MissingTokens,
            ColumnTypes = options.ColumnTypes,
            MaxRows = sampleRows,
        });

        var types = new Dictionary<string, DataType>(options.ColumnTypes, StringComparer.Ordinal);

        foreach (var name in header)
        {
            // A type the caller stated is never overridden by the sample.
            if (types.ContainsKey(name)) continue;
            if (sample.ColumnNames.Contains(name)) types[name] = sample[name].DataType;
        }

        return new CsvOptions
        {
            Delimiter = options.Delimiter,
            HasHeader = options.HasHeader,
            ColumnNames = options.ColumnNames,
            TypeInferenceRows = options.TypeInferenceRows,
            MissingTokens = options.MissingTokens,
            ColumnTypes = types,
        };
    }

    /// <summary>Wraps an in-memory frame, splitting it into chunks.</summary>
    /// <remarks>Mostly for testing a streaming pipeline against a frame small enough to check by hand.</remarks>
    public static ChunkedFrame FromFrame(DataFrame frame, int chunkRows = 100_000)
    {
        ArgumentNullException.ThrowIfNull(frame);
        if (chunkRows <= 0) throw new ArgumentOutOfRangeException(nameof(chunkRows));

        return new ChunkedFrame(() => Split(frame, chunkRows), chunkRows, frame.ColumnNames);
    }

    /// <summary>Wraps an arbitrary sequence of frames.</summary>
    public static ChunkedFrame FromChunks(Func<IEnumerable<DataFrame>> source, IReadOnlyList<string> columns)
        => new(source, 0, columns);

    private static IEnumerable<DataFrame> Split(DataFrame frame, int chunkRows)
    {
        for (var start = 0; start < frame.RowCount; start += chunkRows)
        {
            var count = Math.Min(chunkRows, frame.RowCount - start);
            yield return frame.Take([.. Enumerable.Range(start, count)]);
        }
    }

    /// <summary>Reads only the header line, so the schema is known before any data is touched.</summary>
    private static string[] ReadHeader(string path, CsvOptions options)
    {
        if (!options.HasHeader)
            return options.ColumnNames?.ToArray()
                   ?? throw new ArgumentException(
                       "A headerless CSV needs explicit ColumnNames to be read in chunks.", nameof(options));

        using var reader = new StreamReader(path);
        var line = reader.ReadLine()
                   ?? throw new InvalidDataException($"'{path}' is empty.");

        return CsvReader.SplitLine(line, options.Delimiter);
    }

    /// <summary>
    /// Reads the file line by line, parsing a frame every <paramref name="chunkRows"/> lines.
    /// </summary>
    /// <remarks>
    /// The header is prepended to every chunk's text so the existing parser can infer types and
    /// name columns without a separate code path. That costs one line of parsing per chunk and
    /// saves an entire parallel implementation.
    /// </remarks>
    private static IEnumerable<DataFrame> ReadChunks(
        string path, int chunkRows, CsvOptions options, string[] header)
    {
        using var reader = new StreamReader(path);

        if (options.HasHeader) reader.ReadLine();

        var headerLine = string.Join(options.Delimiter, header);
        var buffer = new List<string>(chunkRows);

        while (reader.ReadLine() is { } line)
        {
            if (line.Length == 0) continue;

            buffer.Add(line);
            if (buffer.Count < chunkRows) continue;

            yield return Parse(headerLine, buffer, options);
            buffer.Clear();
        }

        if (buffer.Count > 0) yield return Parse(headerLine, buffer, options);
    }

    private static DataFrame Parse(string headerLine, List<string> rows, CsvOptions options)
    {
        var text = headerLine + "\n" + string.Join("\n", rows);

        return CsvReader.Parse(text, new CsvOptions
        {
            Delimiter = options.Delimiter,
            HasHeader = true,
            TypeInferenceRows = options.TypeInferenceRows,
            MissingTokens = options.MissingTokens,
            ColumnTypes = options.ColumnTypes,
        });
    }
}

/// <summary>
/// Single-pass operations over a <see cref="ChunkedFrame"/>.
/// </summary>
/// <remarks>
/// Each of these is bounded by something other than the input size, and which something it is
/// varies. That is the thing to know before using them: streaming does not make memory go away, it
/// moves what the memory is proportional to.
/// </remarks>
public static class Streaming
{
    /// <summary>Counts rows without materialising them.</summary>
    public static long CountRows(ChunkedFrame source)
    {
        ArgumentNullException.ThrowIfNull(source);

        var total = 0L;
        foreach (var chunk in source.Chunks) total += chunk.RowCount;
        return total;
    }

    /// <summary>
    /// Aggregates by key in one pass.
    /// </summary>
    /// <param name="source">The chunked input.</param>
    /// <param name="keys">Columns to group by.</param>
    /// <param name="aggregates">Column and aggregate name pairs: sum, mean, min, max, count.</param>
    /// <remarks>
    /// <para>
    /// <b>Memory is proportional to the number of distinct groups, not to the input.</b> That is
    /// the whole trick and also the whole limitation: grouping a billion rows by country is
    /// trivial, and grouping the same rows by user id is not, because the accumulator table then
    /// holds a billion entries. If the key is high-cardinality, this will run out of memory just
    /// as loading the file would — see <see cref="SortToFile"/> for the alternative.
    /// </para>
    /// <para>
    /// Every aggregate here is computable from a fixed-size accumulator. Median deliberately is not
    /// offered: it needs the values, so a streaming version would have to keep them all and would
    /// only look like it was streaming.
    /// </para>
    /// </remarks>
    public static DataFrame GroupBy(ChunkedFrame source, IReadOnlyList<string> keys,
        params (string Column, string How)[] aggregates)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(keys);

        if (aggregates.Length == 0)
            throw new ArgumentException("At least one aggregate is needed.", nameof(aggregates));

        foreach (var (_, how) in aggregates)
            if (!IsSupported(how))
                throw new ArgumentException(
                    $"'{how}' cannot be computed in one pass with bounded memory. " +
                    "Supported: sum, mean, min, max, count. For median, sort first.", nameof(aggregates));

        var accumulators = new Dictionary<string, Accumulator[]>(StringComparer.Ordinal);
        var keyValues = new Dictionary<string, string[]>(StringComparer.Ordinal);

        foreach (var chunk in source.Chunks)
        {
            var keyColumns = keys.Select(k => chunk[k]).ToArray();
            var valueColumns = aggregates.Select(a => chunk.Numeric(a.Column)).ToArray();

            for (var row = 0; row < chunk.RowCount; row++)
            {
                var parts = new string[keys.Count];
                for (var k = 0; k < keys.Count; k++)
                    parts[k] = keyColumns[k].GetValue(row)?.ToString() ?? "";

                // Joined on the unit separator so that two rows holding "a|b" and "a", "b" cannot
                // collide into one group.
                var key = string.Join('\u001f', parts);

                if (!accumulators.TryGetValue(key, out var slots))
                {
                    slots = new Accumulator[aggregates.Length];
                    for (var a = 0; a < aggregates.Length; a++) slots[a] = new Accumulator();

                    accumulators[key] = slots;
                    keyValues[key] = parts;
                }

                for (var a = 0; a < aggregates.Length; a++) slots[a].Add(valueColumns[a][row]);
            }
        }

        return Materialise(keys, aggregates, accumulators, keyValues);
    }

    /// <summary>How many distinct groups a key would produce, without aggregating anything.</summary>
    /// <remarks>
    /// Worth calling first on an unfamiliar key. <see cref="GroupBy"/>'s memory is proportional to
    /// this number, so it is the difference between a query that runs and one that does not.
    /// </remarks>
    public static long CountGroups(ChunkedFrame source, IReadOnlyList<string> keys)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);

        foreach (var chunk in source.Chunks)
        {
            var keyColumns = keys.Select(k => chunk[k]).ToArray();

            for (var row = 0; row < chunk.RowCount; row++)
            {
                var parts = new string[keys.Count];
                for (var k = 0; k < keys.Count; k++)
                    parts[k] = keyColumns[k].GetValue(row)?.ToString() ?? "";

                seen.Add(string.Join('\u001f', parts));
            }
        }

        return seen.Count;
    }

    /// <summary>Summary statistics per numeric column, in one pass and constant memory.</summary>
    /// <remarks>
    /// Variance is accumulated by Welford's method rather than as the difference of
    /// <c>E[x²] − E[x]²</c>. The textbook form is one line shorter and catastrophically imprecise
    /// when the mean is large relative to the spread: it subtracts two nearly equal large numbers
    /// and can return a negative variance.
    /// </remarks>
    public static IReadOnlyDictionary<string, (long Count, double Mean, double StandardDeviation, double Min, double Max)>
        Describe(ChunkedFrame source, IReadOnlyList<string>? columns = null)
    {
        ArgumentNullException.ThrowIfNull(source);

        var running = new Dictionary<string, Accumulator>(StringComparer.Ordinal);

        foreach (var chunk in source.Chunks)
        {
            var names = columns ?? [.. chunk.ColumnNames.Where(n => chunk[n].DataType == DataType.Numeric)];

            foreach (var name in names)
            {
                if (!running.TryGetValue(name, out var accumulator))
                    running[name] = accumulator = new Accumulator();

                var column = chunk.Numeric(name);
                for (var row = 0; row < chunk.RowCount; row++) accumulator.Add(column[row]);
            }
        }

        return running.ToDictionary(
            kv => kv.Key,
            kv => (kv.Value.Count, kv.Value.Mean, kv.Value.StandardDeviation, kv.Value.Min, kv.Value.Max));
    }

    /// <summary>Keeps the rows matching a predicate, writing the result out chunk by chunk.</summary>
    /// <remarks>
    /// Memory is proportional to what survives, not to the input — so this is only bounded when the
    /// filter is selective. <see cref="FilterToFile"/> is bounded regardless.
    /// </remarks>
    public static DataFrame Filter(ChunkedFrame source, Func<DataFrame, int, bool> predicate)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(predicate);

        var kept = new List<DataFrame>();

        foreach (var chunk in source.Chunks)
        {
            var rows = new List<int>();
            for (var row = 0; row < chunk.RowCount; row++)
                if (predicate(chunk, row)) rows.Add(row);

            if (rows.Count > 0) kept.Add(chunk.Take(rows));
        }

        return kept.Count == 0
            ? new DataFrame([.. source.ColumnNames.Select(n => (Series)new NumericSeries(n, []))])
            : DataFrame.Concat(kept);
    }

    /// <summary>Filters straight to a CSV file, so neither the input nor the output need fit.</summary>
    public static long FilterToFile(ChunkedFrame source, Func<DataFrame, int, bool> predicate,
        string path, string delimiter = ",")
    {
        using var writer = new StreamWriter(path);
        var written = 0L;
        var wroteHeader = false;

        foreach (var chunk in source.Chunks)
        {
            var rows = new List<int>();
            for (var row = 0; row < chunk.RowCount; row++)
                if (predicate(chunk, row)) rows.Add(row);

            if (rows.Count == 0) continue;

            var block = chunk.Take(rows);
            var text = CsvWriter.ToCsv(block, delimiter);

            // Only the first chunk contributes a header; the rest have theirs stripped.
            var newline = text.IndexOf('\n');
            if (!wroteHeader) { writer.Write(text); wroteHeader = true; }
            else if (newline >= 0) writer.Write(text[(newline + 1)..]);

            if (!text.EndsWith('\n')) writer.Write('\n');
            written += rows.Count;
        }

        if (!wroteHeader) writer.WriteLine(string.Join(delimiter, source.ColumnNames));
        return written;
    }

    /// <summary>
    /// Sorts a chunked frame into a CSV file by external merge sort.
    /// </summary>
    /// <param name="source">The chunked input.</param>
    /// <param name="column">The numeric column to sort on.</param>
    /// <param name="path">Where the sorted result goes.</param>
    /// <param name="descending">Sort direction.</param>
    /// <param name="temporaryDirectory">Where the intermediate runs go. Defaults to the temp path.</param>
    /// <remarks>
    /// <para>
    /// The classic two-phase algorithm, and the reason a database can sort more data than it can
    /// hold: sort each chunk in memory and write it out as a <em>run</em>, then merge the runs
    /// together by repeatedly taking the smallest head. The merge holds one row per run, so its
    /// memory is proportional to the number of runs rather than to the data.
    /// </para>
    /// <para>
    /// <b>It needs disk space roughly equal to the input</b>, which is the trade being made. It is
    /// also two full passes over the data — one to write runs, one to merge — so it is a great deal
    /// slower than an in-memory sort and should not be reached for when the data fits.
    /// </para>
    /// <para>
    /// Missing values sort last in both directions, matching <c>CategoricalSeries.ArgSort</c>:
    /// missing is absent, not extreme.
    /// </para>
    /// <para>
    /// <b>The output is CSV, so a single-column frame whose values are missing writes blank
    /// lines</b> — and a CSV reader cannot tell those from padding, so reading the result back
    /// loses them. The rows are in the file and correctly ordered; it is the round trip that drops
    /// them. With two or more columns the lines carry delimiters and survive.
    /// </para>
    /// </remarks>
    public static long SortToFile(ChunkedFrame source, string column, string path,
        bool descending = false, string? temporaryDirectory = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(column);

        var scratch = temporaryDirectory ?? Path.Combine(Path.GetTempPath(), $"gravi-sort-{Guid.NewGuid():N}");
        Directory.CreateDirectory(scratch);

        var runs = new List<string>();

        try
        {
            // Phase one: sort each chunk in memory and spill it as a run.
            foreach (var chunk in source.Chunks)
            {
                if (chunk.RowCount == 0) continue;

                var values = chunk.Numeric(column);
                var order = Enumerable.Range(0, chunk.RowCount).ToArray();

                Array.Sort(order, (a, b) => Compare(values[a], values[b], descending));

                var run = Path.Combine(scratch, $"run-{runs.Count:D5}.csv");
                CsvWriter.Write(chunk.Take(order), run);
                runs.Add(run);
            }

            if (runs.Count == 0)
            {
                File.WriteAllText(path, string.Join(",", source.ColumnNames) + "\n");
                return 0;
            }

            return Merge(runs, column, path, descending);
        }
        finally
        {
            if (temporaryDirectory is null && Directory.Exists(scratch))
                Directory.Delete(scratch, recursive: true);
        }
    }

    /// <summary>
    /// Phase two: merge sorted runs by repeatedly taking the smallest head.
    /// </summary>
    /// <remarks>
    /// One reader per run and a heap of their heads, so memory is proportional to the run count.
    /// Reading each run fully into memory here would work on the test data and defeat the point on
    /// anything real, so the readers stay line-oriented.
    /// </remarks>
    private static long Merge(List<string> runs, string column, string path, bool descending)
    {
        var readers = new StreamReader[runs.Count];
        var headers = new string[runs.Count];
        var heads = new string?[runs.Count];
        var keys = new double[runs.Count];
        var keyIndex = -1;

        try
        {
            for (var i = 0; i < runs.Count; i++)
            {
                readers[i] = new StreamReader(runs[i]);
                headers[i] = readers[i].ReadLine() ?? "";

                if (keyIndex < 0)
                {
                    var names = CsvReader.SplitLine(headers[i], ",");
                    keyIndex = Array.IndexOf(names, column);

                    if (keyIndex < 0)
                        throw new InvalidOperationException($"The sorted runs have no column '{column}'.");
                }

                heads[i] = readers[i].ReadLine();
                keys[i] = heads[i] is null ? double.NaN : KeyOf(heads[i]!, keyIndex);
            }

            using var writer = new StreamWriter(path);
            writer.WriteLine(headers[0]);

            var written = 0L;

            while (true)
            {
                var best = -1;

                for (var i = 0; i < runs.Count; i++)
                {
                    if (heads[i] is null) continue;
                    if (best < 0 || Compare(keys[i], keys[best], descending) < 0) best = i;
                }

                if (best < 0) break;

                writer.WriteLine(heads[best]);
                written++;

                heads[best] = readers[best].ReadLine();
                keys[best] = heads[best] is null ? double.NaN : KeyOf(heads[best]!, keyIndex);
            }

            return written;
        }
        finally
        {
            foreach (var reader in readers) reader?.Dispose();
        }
    }

    private static double KeyOf(string line, int index)
    {
        var fields = CsvReader.SplitLine(line, ",");
        if (index >= fields.Length) return double.NaN;

        return double.TryParse(fields[index], NumberStyles.Any, CultureInfo.InvariantCulture, out var value)
            ? value
            : double.NaN;
    }

    /// <summary>Orders two keys, putting missing last whichever way the sort runs.</summary>
    private static int Compare(double a, double b, bool descending)
    {
        var aMissing = double.IsNaN(a);
        var bMissing = double.IsNaN(b);

        if (aMissing || bMissing) return aMissing == bMissing ? 0 : aMissing ? 1 : -1;
        return descending ? b.CompareTo(a) : a.CompareTo(b);
    }

    // ------------------------------------------------------------------ helpers

    private static bool IsSupported(string how)
        => how.ToLowerInvariant() is "sum" or "mean" or "min" or "max" or "count";

    /// <summary>
    /// A fixed-size accumulator: everything it reports is computable without keeping the values.
    /// </summary>
    private sealed class Accumulator
    {
        private double _m2;

        public long Count { get; private set; }
        public double Mean { get; private set; }
        public double Sum { get; private set; }
        public double Min { get; private set; } = double.PositiveInfinity;
        public double Max { get; private set; } = double.NegativeInfinity;

        /// <summary>Sample standard deviation, from Welford's running second moment.</summary>
        public double StandardDeviation => Count < 2 ? double.NaN : Math.Sqrt(_m2 / (Count - 1));

        public void Add(double value)
        {
            // Missing values are skipped rather than counted, so a mean over a sparse column is the
            // mean of what is there — matching what the in-memory aggregates do.
            if (double.IsNaN(value)) return;

            Count++;
            Sum += value;
            Min = Math.Min(Min, value);
            Max = Math.Max(Max, value);

            // Welford: numerically stable where E[x²] − E[x]² is not.
            var delta = value - Mean;
            Mean += delta / Count;
            _m2 += delta * (value - Mean);
        }

        public double Value(string how) => how.ToLowerInvariant() switch
        {
            "sum" => Count == 0 ? 0 : Sum,
            "mean" => Count == 0 ? double.NaN : Mean,
            "min" => Count == 0 ? double.NaN : Min,
            "max" => Count == 0 ? double.NaN : Max,
            _ => Count,
        };
    }

    private static DataFrame Materialise(IReadOnlyList<string> keys,
        (string Column, string How)[] aggregates,
        Dictionary<string, Accumulator[]> accumulators,
        Dictionary<string, string[]> keyValues)
    {
        var order = accumulators.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray();
        var columns = new List<Series>(keys.Count + aggregates.Length);

        for (var k = 0; k < keys.Count; k++)
        {
            var index = k;
            columns.Add(new TextSeries(keys[k], [.. order.Select(key => keyValues[key][index])]));
        }

        for (var a = 0; a < aggregates.Length; a++)
        {
            var index = a;
            var (column, how) = aggregates[a];

            columns.Add(new NumericSeries($"{column}_{how.ToLowerInvariant()}",
                [.. order.Select(key => accumulators[key][index].Value(how))]));
        }

        return new DataFrame(columns);
    }
}
