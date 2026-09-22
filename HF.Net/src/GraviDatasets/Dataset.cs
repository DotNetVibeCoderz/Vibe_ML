using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviFrame.Io;

namespace Gravicode.HFNet.GraviDatasets;

/// <summary>
/// A named table of examples: a <see cref="DataFrame"/> plus the operations a training pipeline
/// asks of it - shuffling, splitting, mapping and batching.
/// </summary>
/// <remarks>
/// <para>
/// Every operation returns a new dataset and leaves the original alone. Datasets are passed around
/// between preprocessing steps and a mutating <c>Shuffle</c> would silently change what an earlier
/// split had already taken, which is the kind of leak that inflates a validation score without
/// producing any visible error.
/// </para>
/// <para>
/// The underlying frame is shared rather than copied where an operation only selects rows, so
/// taking a hundred splits of a large dataset does not multiply its memory.
/// </para>
/// </remarks>
public sealed class Dataset
{
    /// <summary>Wraps a frame as a dataset.</summary>
    /// <param name="frame">The rows.</param>
    /// <param name="name">A name for messages and for <see cref="ToString"/>.</param>
    public Dataset(DataFrame frame, string name = "dataset")
    {
        Frame = frame ?? throw new ArgumentNullException(nameof(frame));
        Name = name;
    }

    /// <summary>The rows, as a GraviFrame <see cref="DataFrame"/>.</summary>
    public DataFrame Frame { get; }

    /// <summary>A name used in messages.</summary>
    public string Name { get; }

    /// <summary>Number of examples.</summary>
    public int Count => Frame.RowCount;

    /// <summary>The column names.</summary>
    public IReadOnlyList<string> Columns => Frame.ColumnNames;

    /// <summary>Column names and their types, as a one-line summary per column.</summary>
    public IReadOnlyList<(string Name, string Type)> Features
        => [.. Frame.Columns.Select(c => (c.Name, c.GetType().Name.Replace("Series", "")))];

    // ------------------------------------------------------------------ loading

    /// <summary>
    /// Loads a dataset by name: one of the built-in classics, or a Hugging Face Hub dataset id.
    /// </summary>
    /// <param name="name">
    /// A built-in name such as <c>titanic</c> or <c>iris</c>, or a Hub id such as
    /// <c>stanfordnlp/imdb</c>.
    /// </param>
    /// <param name="split">Which split to return when the source has several.</param>
    /// <remarks>
    /// A name containing a slash is always treated as a Hub id; a bare name is looked up in the
    /// built-in registry first and only then on the Hub. That ordering is what keeps
    /// <c>Dataset.Load("titanic")</c> working without a network connection.
    /// </remarks>
    public static Dataset Load(string name, string split = "train")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        if (!name.Contains('/') && BuiltinDatasets.TryResolve(name, out var path))
        {
            return FromFile(path, name);
        }

        return HubDatasets.Load(name, split);
    }

    /// <summary>Loads every split of a dataset.</summary>
    /// <param name="name">A built-in name or a Hub dataset id.</param>
    public static DatasetDict LoadAll(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        if (!name.Contains('/') && BuiltinDatasets.TryResolve(name, out var path))
        {
            return new DatasetDict(new Dictionary<string, Dataset> { ["train"] = FromFile(path, name) });
        }

        return HubDatasets.LoadAll(name);
    }

    /// <summary>Reads a dataset from a local file, choosing the reader by extension.</summary>
    /// <param name="path">A <c>.csv</c>, <c>.tsv</c>, <c>.parquet</c>, <c>.json</c> or <c>.jsonl</c> file.</param>
    /// <param name="name">A name for the dataset.</param>
    public static Dataset FromFile(string path, string? name = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path)) throw new FileNotFoundException("Dataset file not found.", path);

        var label = name ?? Path.GetFileNameWithoutExtension(path);

        return Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".csv" => new Dataset(DataFrame.ReadCsv(path), label),
            ".tsv" => new Dataset(DataFrame.ReadCsv(path, new CsvOptions { Delimiter = "\t" }), label),
            ".parquet" => new Dataset(Gravicode.Science.GraviFrame.Io.ParquetIO.Read(path), label),
            ".json" or ".jsonl" or ".ndjson" => new Dataset(JsonLines.Read(path), label),
            var other => throw new NotSupportedException(
                $"'{other}' is not a dataset format this loader reads. Use CSV, TSV, Parquet, JSON or JSON Lines."),
        };
    }

    /// <summary>
    /// Reads a CSV through a memory mapping rather than into the heap.
    /// </summary>
    /// <param name="path">The file.</param>
    /// <param name="name">A name for the dataset.</param>
    /// <remarks>
    /// Worth it only for files large relative to RAM. Below that the mapping's page faults cost
    /// more than a straight read, which is what <c>GraviDatasets.Benchmark</c> measures.
    /// </remarks>
    public static Dataset FromCsvMemoryMapped(string path, string? name = null)
        => new(DataFrame.ReadCsvMemoryMapped(path), name ?? Path.GetFileNameWithoutExtension(path));

    /// <summary>Wraps an in-memory frame.</summary>
    public static Dataset FromFrame(DataFrame frame, string name = "dataset") => new(frame, name);

    // ------------------------------------------------------------------ shaping

    /// <summary>The first <paramref name="count"/> examples.</summary>
    public Dataset Head(int count = 5) => new(Frame.Head(count), Name);

    /// <summary>A contiguous range of examples.</summary>
    public Dataset Slice(int start, int count) => new(Frame.Rows(start, count), Name);

    /// <summary>The examples at the given row indices, in that order.</summary>
    public Dataset Select(IReadOnlyList<int> indices) => new(Frame.Take(indices), Name);

    /// <summary>Keeps only the named columns.</summary>
    public Dataset SelectColumns(params string[] columns) => new(Frame.SelectColumns(columns), Name);

    /// <summary>Drops the named columns.</summary>
    public Dataset RemoveColumns(params string[] columns) => new(Frame.Drop(columns), Name);

    /// <summary>Renames a column.</summary>
    public Dataset RenameColumn(string from, string to) => new(Frame.Rename(from, to), Name);

    /// <summary>Shuffles the examples.</summary>
    /// <param name="seed">The seed, so a shuffle can be reproduced.</param>
    public Dataset Shuffle(int seed = 42)
    {
        var order = Enumerable.Range(0, Count).ToArray();
        var random = new Random(seed);

        for (var i = order.Length - 1; i > 0; i--)
        {
            var j = random.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }

        return new Dataset(Frame.Take(order), Name);
    }

    /// <summary>Keeps the examples a predicate accepts.</summary>
    /// <param name="predicate">Called with each row.</param>
    public Dataset Filter(Func<RowView, bool> predicate) => new(Frame.Filter(predicate), Name);

    /// <summary>Adds a computed numeric column.</summary>
    /// <param name="column">The new column's name.</param>
    /// <param name="compute">Called with each row index.</param>
    /// <remarks>
    /// Named <c>Map</c> to match the Python API, though it only adds a column rather than replacing
    /// the example: a dataset here is a typed frame, so a map that could change the schema per row
    /// would have no meaningful result type.
    /// </remarks>
    public Dataset Map(string column, Func<int, double> compute) => new(Frame.WithColumn(column, compute), Name);

    /// <summary>Splits into a training and a test set.</summary>
    /// <param name="testSize">The test fraction, between 0 and 1.</param>
    /// <param name="seed">The shuffle seed.</param>
    /// <param name="shuffle">Whether to shuffle before splitting.</param>
    /// <remarks>
    /// Shuffling is on by default and matters more than it looks: many published CSVs are sorted by
    /// label, and an unshuffled split of one of those puts every positive example on one side.
    /// </remarks>
    public DatasetDict TrainTestSplit(double testSize = 0.2, int seed = 42, bool shuffle = true)
    {
        if (testSize is <= 0 or >= 1) throw new ArgumentOutOfRangeException(nameof(testSize));

        var source = shuffle ? Shuffle(seed) : this;
        var testCount = Math.Max(1, (int)Math.Round(Count * testSize));
        var trainCount = Count - testCount;

        return new DatasetDict(new Dictionary<string, Dataset>
        {
            ["train"] = new(source.Frame.Rows(0, trainCount), Name),
            ["test"] = new(source.Frame.Rows(trainCount, testCount), Name),
        });
    }

    // ------------------------------------------------------------------ access

    /// <summary>Iterates the examples in order, one row at a time.</summary>
    /// <remarks>
    /// This is the streaming read. It holds one row at a time rather than materialising a list, so
    /// a pipeline that only ever looks at each example once need not size its memory to the data.
    /// </remarks>
    public IEnumerable<RowView> Rows()
    {
        for (var i = 0; i < Count; i++) yield return Frame.Row(i);
    }

    /// <summary>Iterates the examples in fixed-size batches.</summary>
    /// <param name="size">Rows per batch.</param>
    /// <param name="dropLast">Whether to discard a final short batch.</param>
    public IEnumerable<Dataset> Batches(int size, bool dropLast = false)
    {
        if (size < 1) throw new ArgumentOutOfRangeException(nameof(size));

        for (var start = 0; start < Count; start += size)
        {
            var take = Math.Min(size, Count - start);
            if (take < size && dropLast) yield break;

            yield return new Dataset(Frame.Rows(start, take), Name);
        }
    }

    /// <summary>The text of one column, as strings.</summary>
    /// <param name="column">The column name.</param>
    /// <remarks>
    /// Works for a text column and for a numeric one, which is what makes it usable on a label
    /// column whose type differs between two otherwise identical published datasets.
    /// </remarks>
    public IReadOnlyList<string> TextColumn(string column)
    {
        var series = Frame[column];

        if (series is TextSeries text)
        {
            return [.. Enumerable.Range(0, text.Length).Select(i => text[i] ?? "")];
        }

        if (series is NumericSeries numeric)
        {
            return [.. Enumerable.Range(0, numeric.Length).Select(i => numeric[i].ToString("G"))];
        }

        return [.. Enumerable.Range(0, series.Length).Select(i => series.GetValue(i)?.ToString() ?? "")];
    }

    /// <summary>One numeric column as an array.</summary>
    public double[] NumericColumn(string column) => Frame.Numeric(column).Values;

    /// <summary>The named columns as a <c>[rows, columns]</c> matrix.</summary>
    public NdArray ToMatrix(params string[] columns)
        => columns.Length == 0 ? Frame.ToNdArray() : Frame.ToNdArray(columns);

    /// <summary>Per-column summary statistics.</summary>
    public DataFrame Describe() => Frame.Describe();

    /// <summary>Writes the dataset to a CSV.</summary>
    public void WriteCsv(string path) => Frame.WriteCsv(path);

    /// <summary>Writes the dataset to a Parquet file.</summary>
    public void WriteParquet(string path) => Gravicode.Science.GraviFrame.Io.ParquetIO.Write(Frame, path);

    /// <inheritdoc />
    public override string ToString()
        => $"Dataset '{Name}' ({Count:N0} rows x {Frame.ColumnCount} columns: {string.Join(", ", Columns.Take(6))}{(Frame.ColumnCount > 6 ? ", ..." : "")})";
}

/// <summary>A dataset's splits, keyed by name.</summary>
/// <remarks>
/// Indexing a missing split throws with the available names listed. A dataset published with
/// <c>train</c> and <c>validation</c> but no <c>test</c> is common enough that the error needs to
/// say which splits exist rather than only which one was asked for.
/// </remarks>
public sealed class DatasetDict
{
    private readonly Dictionary<string, Dataset> _splits;

    /// <summary>Creates a split collection.</summary>
    /// <param name="splits">The splits, keyed by name.</param>
    public DatasetDict(IReadOnlyDictionary<string, Dataset> splits)
        => _splits = new Dictionary<string, Dataset>(splits, StringComparer.OrdinalIgnoreCase);

    /// <summary>The split of that name.</summary>
    public Dataset this[string split] => _splits.TryGetValue(split, out var dataset)
        ? dataset
        : throw new KeyNotFoundException(
            $"No split named '{split}'. This dataset has: {string.Join(", ", _splits.Keys)}.");

    /// <summary>The available split names.</summary>
    public IReadOnlyCollection<string> Splits => _splits.Keys;

    /// <summary>Whether a split of that name exists.</summary>
    public bool Contains(string split) => _splits.ContainsKey(split);

    /// <summary>The training split, by any of its usual names.</summary>
    public Dataset Train => Resolve("train", "training");

    /// <summary>The test split, by any of its usual names.</summary>
    public Dataset Test => Resolve("test", "testing", "eval");

    /// <summary>The validation split, by any of its usual names.</summary>
    public Dataset Validation => Resolve("validation", "valid", "dev");

    private Dataset Resolve(params string[] candidates)
    {
        foreach (var candidate in candidates)
        {
            if (_splits.TryGetValue(candidate, out var dataset)) return dataset;
        }

        throw new KeyNotFoundException(
            $"No split among {string.Join(", ", candidates)}. This dataset has: {string.Join(", ", _splits.Keys)}.");
    }

    /// <summary>Total rows across every split.</summary>
    public int TotalRows => _splits.Values.Sum(d => d.Count);

    /// <inheritdoc />
    public override string ToString()
        => $"DatasetDict({string.Join(", ", _splits.Select(p => $"{p.Key}: {p.Value.Count:N0}"))})";
}
