using System.Reflection;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;

namespace Gravicode.Science.DaskNet;

/// <summary>
/// A lazy DataFrame split into partitions, each a GraviFrame <see cref="DataFrame"/> (Dask's
/// <c>dask.dataframe</c>). Operations build a logical plan; <see cref="Compute"/> optimises it,
/// lowers it to a task graph and runs that on a scheduler.
/// </summary>
/// <remarks>
/// <para>
/// The plan is what makes this more than a loop over files. <see cref="Explain"/> shows it before
/// and after the optimiser: filters move towards the scan and through joins
/// (<i>predicate pushdown</i>), each scan parses only the columns something downstream reads
/// (<i>projection pushdown</i>), and the row-wise steps left in a run execute in one task per
/// partition (<i>pipeline fusion</i>).
/// </para>
/// <para>
/// Filters and projections take <see cref="Expr"/> rather than lambdas so that they can be
/// optimised and shipped to cluster workers. <see cref="MapPartitions"/> takes any function, at
/// the price of being opaque to both.
/// </para>
/// </remarks>
public sealed partial class DaskFrame
{
    internal DaskFrame(PlanNode plan) => Plan = plan;

    internal PlanNode Plan { get; }

    /// <summary>Column names, or empty when an opaque <see cref="MapPartitions"/> hides them.</summary>
    public IReadOnlyList<string> Columns => Plan.Schema ?? [];

    /// <summary>How many partitions the result has.</summary>
    public int PartitionCount => Plan.Partitions;

    /// <summary>Whether this frame's graph can run on a cluster.</summary>
    public bool IsRemotable => Plan.Remotable;

    /// <summary>How many partial results each node of an aggregation tree combines.</summary>
    public static int DefaultSplitEvery { get; set; } = 8;

    // ------------------------------------------------------------------ sources

    /// <summary>Splits an in-memory frame into <paramref name="partitions"/> row ranges.</summary>
    public static DaskFrame FromFrame(DataFrame frame, int partitions = 4)
    {
        ArgumentNullException.ThrowIfNull(frame);
        if (partitions <= 0) throw new ArgumentOutOfRangeException(nameof(partitions));
        partitions = Math.Max(1, Math.Min(partitions, Math.Max(frame.RowCount, 1)));
        var parts = Enumerable.Range(0, partitions).Select(p =>
        {
            var (start, count) = Bag.Slice(frame.RowCount, partitions, p);
            return frame.Rows(start, count);
        }).ToArray();
        return new DaskFrame(new ScanNode(new MemorySource(parts)));
    }

    /// <summary>A frame whose partitions are given frames (same columns, in the same order).</summary>
    public static DaskFrame FromPartitions(IReadOnlyList<DataFrame> partitions)
    {
        if (partitions.Count == 0) throw new ArgumentException("At least one partition is needed.");
        var names = partitions[0].ColumnNames;
        if (partitions.Any(p => !p.ColumnNames.SequenceEqual(names)))
            throw new ArgumentException("Every partition must have the same columns in the same order.");
        return new DaskFrame(new ScanNode(new MemorySource(partitions.ToArray())));
    }

    /// <summary>A frame whose partitions are computed by delayed values (Dask's <c>from_delayed</c>).</summary>
    public static DaskFrame FromDelayed(IReadOnlyList<Delayed<DataFrame>> partitions, IReadOnlyList<string> columns)
        => new(new ScanNode(new DelayedSource(partitions.ToArray(), columns.ToArray())));

    /// <summary>
    /// Reads CSV files lazily, one partition per <paramref name="blockSize"/> bytes of each file
    /// (Dask's <c>read_csv</c>).
    /// </summary>
    /// <param name="pattern">A file, directory or glob such as <c>sales/2024-*.csv</c>.</param>
    /// <param name="blockSize">Bytes per partition; 0 means one partition per file. Default 64 MiB.</param>
    /// <param name="options">Delimiter, missing tokens and forced column types.</param>
    /// <remarks>
    /// Column names and types are read now, from the first file's header and leading rows, and
    /// pinned for every block: inferring per block would type a column as a number in one
    /// partition and text in the next. Quoted fields must not contain line breaks when
    /// <paramref name="blockSize"/> splits a file - a block boundary cannot tell an embedded
    /// newline from a row end (Dask has the same restriction).
    /// </remarks>
    public static DaskFrame ReadCsv(string pattern, long blockSize = 64L << 20, CsvOptions? options = null)
        => ReadCsv([pattern], blockSize, options);

    /// <summary>Reads CSV files matching several patterns as one frame.</summary>
    public static DaskFrame ReadCsv(IEnumerable<string> patterns, long blockSize = 64L << 20, CsvOptions? options = null)
    {
        options ??= new CsvOptions();
        var files = GraphBuilders.ExpandPaths(patterns);
        var sample = CsvReader.Read(files[0], new CsvOptions
        {
            Delimiter = options.Delimiter,
            HasHeader = true,
            MaxRows = Math.Max(options.TypeInferenceRows, 1),
            TypeInferenceRows = options.TypeInferenceRows,
            MissingTokens = options.MissingTokens,
            ColumnTypes = options.ColumnTypes,
        });
        var names = sample.ColumnNames.ToArray();
        var types = sample.Columns.Select(c => c.DataType == DataType.Categorical ? DataType.Text : c.DataType).ToArray();

        var blocks = new List<(string, long, long)>();
        foreach (var file in files)
            foreach (var (start, end) in GraphBuilders.ByteRanges(new FileInfo(file).Length, blockSize))
                blocks.Add((file, start, end));
        return new DaskFrame(new ScanNode(new CsvSource(blocks, names, types, options.Delimiter, options.MissingTokens.ToArray())));
    }

    /// <summary>Reads Arrow IPC files, one partition per file.</summary>
    public static DaskFrame ReadArrow(string pattern)
    {
        var files = GraphBuilders.ExpandPaths(pattern);
        var columns = ArrowFile.Read(files[0]).ColumnNames.ToArray();
        return new DaskFrame(new ScanNode(new ArrowSource(files, columns)));
    }

    /// <summary>
    /// Reads files with any reader, one partition per file - for example
    /// <c>DaskFrame.ReadFiles("data/*.parquet", ParquetIO.Read)</c>. Runs locally only.
    /// </summary>
    public static DaskFrame ReadFiles(string pattern, Func<string, DataFrame> reader)
    {
        var files = GraphBuilders.ExpandPaths(pattern);
        var columns = reader(files[0]).ColumnNames.ToArray();
        return new DaskFrame(new ScanNode(new DelegateSource(files, reader, columns)));
    }

    /// <summary>Stacks frames with the same columns, keeping every partition.</summary>
    public static DaskFrame Concat(params DaskFrame[] frames)
    {
        if (frames.Length == 0) throw new ArgumentException("Nothing to concatenate.");
        var names = frames[0].Columns;
        if (frames.Any(f => !f.Columns.SequenceEqual(names)))
            throw new ArgumentException("Concat needs the same columns in the same order.");
        return new DaskFrame(new ConcatNode(frames.Select(f => f.Plan).ToArray()));
    }

    // ------------------------------------------------------------------ row-wise

    /// <summary>Keeps the rows where <paramref name="predicate"/> is true.</summary>
    public DaskFrame Filter(Expr predicate) => new(new FilterNode(Plan, CheckColumns(predicate)));

    /// <summary>Same as <see cref="Filter"/>.</summary>
    public DaskFrame Where(Expr predicate) => Filter(predicate);

    /// <summary>Keeps only the named columns, in the order given.</summary>
    public DaskFrame Select(params string[] columns) => Select(columns.Select(Expr.Col).ToArray());

    /// <summary>Computes exactly these columns (use <see cref="Expr.As"/> to name them).</summary>
    public DaskFrame Select(params Expr[] items)
    {
        foreach (var i in items) CheckColumns(i);
        var duplicate = items.GroupBy(i => i.OutputName).FirstOrDefault(g => g.Count() > 1);
        if (duplicate is not null) throw new ArgumentException($"Column '{duplicate.Key}' is selected twice; alias one of them.");
        return new DaskFrame(new ProjectNode(Plan, items));
    }

    /// <summary>Adds or replaces a column.</summary>
    public DaskFrame WithColumn(string name, Expr value) => WithColumns(value.As(name));

    /// <summary>Adds or replaces several columns, each computed from the input (they do not see each other).</summary>
    public DaskFrame WithColumns(params Expr[] items)
    {
        foreach (var i in items) CheckColumns(i);
        return new DaskFrame(new WithColumnsNode(Plan, items));
    }

    /// <summary>Removes columns.</summary>
    public DaskFrame Drop(params string[] columns) => new(new DropNode(Plan, columns));

    /// <summary>Renames one column.</summary>
    public DaskFrame Rename(string from, string to) => new(new RenameNode(Plan, [(from, to)]));

    /// <summary>Drops rows with a missing value in any of <paramref name="subset"/> (all columns if none given).</summary>
    public DaskFrame DropMissing(params string[] subset)
    {
        var columns = subset.Length > 0 ? subset : Columns.ToArray();
        if (columns.Length == 0) throw new InvalidOperationException("Name the columns: this frame's schema is unknown.");
        return Filter(Expr.JoinAnd(columns.Select(c => Expr.Col(c).IsNotNull()))!);
    }

    /// <summary>Replaces missing values in the named columns.</summary>
    public DaskFrame FillMissing(Expr value, params string[] columns)
        => WithColumns(columns.Select(c => Expr.Col(c).FillNull(value).As(c)).ToArray());

    /// <summary>
    /// Applies any function to every partition (Dask's <c>map_partitions</c>). Opaque to the
    /// optimiser and local only; pass <paramref name="columns"/> if later steps need the schema.
    /// </summary>
    public DaskFrame MapPartitions(Func<DataFrame, DataFrame> function, IReadOnlyList<string>? columns = null, string label = "map-partitions")
    {
        ArgumentNullException.ThrowIfNull(function);
        return new DaskFrame(new MapPartitionsNode(Plan, function, columns?.ToArray(), label));
    }

    // ------------------------------------------------------------------ reshaping

    /// <summary>
    /// Adds window-function columns (SQL's <c>OVER (PARTITION BY ... ORDER BY ...)</c>): ranks,
    /// lag/lead, and running or per-partition aggregates. Every row is kept. Rows are
    /// hash-shuffled by <paramref name="partitionBy"/> so each window partition is computed whole
    /// in one task; with no partition keys the frame is gathered into one task.
    /// </summary>
    public DaskFrame Window(string[] partitionBy, IReadOnlyList<(string Column, bool Ascending)> orderBy, params WindowFunction[] functions)
    {
        ArgumentNullException.ThrowIfNull(partitionBy);
        ArgumentNullException.ThrowIfNull(orderBy);
        if (functions.Length == 0) throw new ArgumentException("Give at least one window function.", nameof(functions));
        foreach (var k in partitionBy) CheckColumn(k);
        foreach (var (c, _) in orderBy) CheckColumn(c);
        foreach (var f in functions)
        {
            f.Validate();
            if (f.Column is not null) CheckColumn(f.Column);
            if (WindowFunction.Ranking.Contains(f.How) && orderBy.Count == 0)
                throw new ArgumentException($"Window function '{f.How}' needs an order.");
            if (Columns.Contains(f.Output)) throw new ArgumentException($"Column '{f.Output}' already exists.");
        }
        return new DaskFrame(new WindowNode(Plan, new WindowSpec(partitionBy.ToArray(), orderBy.ToArray(), functions.ToArray())));
    }

    /// <summary>Groups rows by key columns; aggregate the result to get a frame.</summary>
    public DaskGroupBy GroupBy(params string[] keys)
    {
        if (keys.Length == 0) throw new ArgumentException("GroupBy needs at least one key.");
        foreach (var k in keys) CheckColumn(k);
        return new DaskGroupBy(this, keys);
    }

    /// <summary>Whole-frame aggregation into a one-row frame.</summary>
    public DaskFrame Aggregate(params Aggregation[] aggregations)
    {
        foreach (var a in aggregations) { a.Validate(); if (a.Column is not null) CheckColumn(a.Column); }
        return new DaskFrame(new AggregateNode(Plan, [], aggregations, DefaultSplitEvery));
    }

    /// <summary>Joins on a column both sides share.</summary>
    public DaskFrame Join(DaskFrame other, string on, JoinKind kind = JoinKind.Inner, string suffix = "_right", int? partitions = null,
        bool? broadcast = null)
        => Merge(other, [on], [on], kind, suffix, partitions, broadcast);

    /// <summary>
    /// Joins on key columns. Inner and left joins <b>broadcast</b> the right side - gather it into
    /// one partition that every left partition reads - when it has one partition already, or when
    /// it is estimated at or below <see cref="Dask.BroadcastJoinBytes"/>; an inner join whose left
    /// side is the small one is turned around. Otherwise both sides are hash-shuffled so matching
    /// keys meet.
    /// </summary>
    /// <param name="other">The right side.</param>
    /// <param name="leftOn">Key columns of this frame.</param>
    /// <param name="rightOn">Key columns of <paramref name="other"/>, matched by position.</param>
    /// <param name="kind">Inner, left, right or outer.</param>
    /// <param name="suffix">Added to right columns whose names the left side also has.</param>
    /// <param name="partitions">Shuffle into this many partitions (and never broadcast).</param>
    /// <param name="broadcast">true to broadcast the right side whatever its size (inner and left
    /// joins only), false to always shuffle, null to decide by size (Dask's <c>broadcast=</c>).</param>
    public DaskFrame Merge(DaskFrame other, string[] leftOn, string[] rightOn, JoinKind kind = JoinKind.Inner,
        string suffix = "_right", int? partitions = null, bool? broadcast = null)
    {
        if (leftOn.Length == 0 || leftOn.Length != rightOn.Length) throw new ArgumentException("Give matching key lists.");
        foreach (var k in leftOn) CheckColumn(k);
        foreach (var k in rightOn) other.CheckColumn(k);
        var right = other.Plan;
        if (broadcast == true)
        {
            if (kind is not (JoinKind.Inner or JoinKind.Left))
                throw new ArgumentException("Only inner and left joins can broadcast the right side; the others must see every right row.", nameof(broadcast));
            if (partitions is not null) throw new ArgumentException("A broadcast join keeps the left partitioning; do not give partitions.", nameof(partitions));
            if (right.Partitions > 1) right = new RepartitionNode(right, 1);
        }
        return new DaskFrame(new JoinNode(Plan, right, leftOn, rightOn, kind, suffix, partitions, broadcast));
    }

    /// <summary>
    /// Sorts by one or more columns into range partitions, so concatenating the partitions in
    /// order gives the sorted frame. Boundaries come from a sample taken when the graph is built,
    /// which reads the input once more.
    /// </summary>
    public DaskFrame SortBy(string column, bool ascending = true, int? partitions = null)
        => SortBy([(column, ascending)], partitions);

    /// <summary>Sorts by several keys; partitions are ranges of the first.</summary>
    public DaskFrame SortBy(IReadOnlyList<(string Column, bool Ascending)> keys, int? partitions = null)
    {
        foreach (var (c, _) in keys) CheckColumn(c);
        return new DaskFrame(new SortNode(Plan, keys.Select(k => k.Column).ToArray(), keys.Select(k => k.Ascending).ToArray(), partitions));
    }

    /// <summary>The same rows in <paramref name="partitions"/> partitions, order kept.</summary>
    public DaskFrame Repartition(int partitions)
        => partitions > 0 ? new DaskFrame(new RepartitionNode(Plan, partitions)) : throw new ArgumentOutOfRangeException(nameof(partitions));

    /// <summary>The first <paramref name="count"/> rows, lazily (SQL's <c>LIMIT</c>).</summary>
    public DaskFrame Limit(int count) => new(new LimitNode(Plan, count));

    /// <summary>Each distinct value of a column with its row count, most frequent first.</summary>
    public DaskFrame ValueCounts(string column)
        => GroupBy(column).Agg(new Aggregation("count", "size", null)).SortBy([("count", false), (column, true)], partitions: 1);

    // ------------------------------------------------------------------ results

    /// <summary>The number of rows.</summary>
    public Delayed<long> Count()
    {
        var parts = Lower(Dask.CurrentScheduler);
        var countName = TaskKey.NewName("count-partial");
        var counts = parts.Keys.Select((k, i) => TaskSpec.Call(new TaskKey(countName, i),
            typeof(FrameKernels).GetMethod(nameof(FrameKernels.CountRows))!, [new TaskRef(k)], "count-partial")).ToArray();
        var sum = typeof(FrameKernels).GetMethod(nameof(FrameKernels.SumCounts))!;
        var treeName = TaskKey.NewName("count");
        var (tree, root) = GraphBuilders.TreeReduce(treeName, counts.Select(c => c.Key).ToArray(), DefaultSplitEvery,
            (key, inputs) => TaskSpec.Call(key, sum, [new TaskRefList(inputs)], "count"));
        var countLayer = new GraphLayer(countName, counts, [parts.Layer]);
        return new Delayed<long>(new GraphLayer(treeName, tree, [countLayer]), root);
    }

    /// <summary>Sum of a column.</summary>
    public Delayed<double> Sum(string column) => Scalar("sum", column);
    /// <summary>Mean of a column.</summary>
    public Delayed<double> Mean(string column) => Scalar("mean", column);
    /// <summary>Minimum of a column.</summary>
    public Delayed<double> Min(string column) => Scalar("min", column);
    /// <summary>Maximum of a column.</summary>
    public Delayed<double> Max(string column) => Scalar("max", column);
    /// <summary>Sample standard deviation of a column.</summary>
    public Delayed<double> Std(string column) => Scalar("std", column);
    /// <summary>Sample variance of a column.</summary>
    public Delayed<double> Var(string column) => Scalar("var", column);

    private Delayed<double> Scalar(string how, string column)
    {
        var frame = Aggregate(new Aggregation(how, how, column));
        var parts = frame.Lower(Dask.CurrentScheduler);
        return new Delayed<DataFrame>(parts.Layer, parts.Keys[0]).Then(f => f.Numeric(how)[0], how);
    }

    /// <summary>count, mean, std, min and max of every numeric column, one row per statistic.</summary>
    public DataFrame Describe(IScheduler? scheduler = null)
    {
        var numeric = NumericColumns(scheduler);
        var stats = new[] { "count", "mean", "std", "min", "max" };
        var aggs = numeric.SelectMany(c => stats.Select(s => new Aggregation($"{c}\u001F{s}", s, c))).ToArray();
        var row = Aggregate(aggs).Compute(scheduler);
        var columns = new List<Series> { new TextSeries("statistic", stats) };
        foreach (var c in numeric)
            columns.Add(new NumericSeries(c, stats.Select(s => row.Numeric($"{c}\u001F{s}")[0]).ToArray()));
        return new DataFrame(columns);
    }

    private string[] NumericColumns(IScheduler? scheduler)
    {
        if (FindCsv(Plan) is { } csv && Plan.Schema is { } schema)
            return schema.Where(c => csv.Columns.Contains(c) && csv.Types[csv.Columns.ToList().IndexOf(c)] == DataType.Numeric).ToArray();
        var head = Head(1, scheduler);
        return head.Columns.OfType<NumericSeries>().Select(c => c.Name).ToArray();
    }

    private static CsvSource? FindCsv(PlanNode node)
        => node is ScanNode { Source: CsvSource csv } ? csv : node.Children.Count == 1 ? FindCsv(node.Children[0]) : null;

    /// <summary>
    /// The first <paramref name="count"/> rows, computing only as many leading partitions as
    /// that needs (Dask's <c>head</c>).
    /// </summary>
    public DataFrame Head(int count = 5, IScheduler? scheduler = null)
    {
        scheduler ??= Dask.CurrentScheduler;
        var parts = Lower(scheduler);
        var collected = new List<DataFrame>();
        var rows = 0;
        foreach (var key in parts.Keys)
        {
            if (rows >= count) break;
            var frame = (DataFrame)scheduler.ExecuteAsync(Graph(new Lowering.Parts(parts.Layer, [key])), [key])
                .GetAwaiter().GetResult()[key]!;
            collected.Add(frame);
            rows += frame.RowCount;
        }
        var all = FrameKernels.ConcatFrames(collected.ToArray());
        return all.RowCount > count ? all.Head(count) : all;
    }

    /// <summary>Computes every partition and concatenates them in order.</summary>
    public DataFrame Compute(IScheduler? scheduler = null, CancellationToken cancellationToken = default)
        => ComputeAsync(scheduler, cancellationToken).GetAwaiter().GetResult();

    /// <summary>Computes every partition asynchronously and concatenates them in order.</summary>
    public async Task<DataFrame> ComputeAsync(IScheduler? scheduler = null, CancellationToken cancellationToken = default)
    {
        var partitions = await ComputePartitionsAsync(scheduler, cancellationToken).ConfigureAwait(false);
        return FrameKernels.ConcatFrames(partitions);
    }

    /// <summary>Computes every partition, separately.</summary>
    public async Task<DataFrame[]> ComputePartitionsAsync(IScheduler? scheduler = null, CancellationToken cancellationToken = default)
    {
        scheduler ??= Dask.CurrentScheduler;
        var parts = Lower(scheduler);
        var results = await scheduler.ExecuteAsync(Graph(parts), parts.Keys, cancellationToken).ConfigureAwait(false);
        return parts.Keys.Select(k => (DataFrame)results[k]!).ToArray();
    }

    /// <summary>Computes now and returns a frame over the in-memory partitions.</summary>
    public DaskFrame Persist(IScheduler? scheduler = null)
    {
        var partitions = ComputePartitionsAsync(scheduler).GetAwaiter().GetResult();
        return new DaskFrame(new ScanNode(new MemorySource(partitions, "persisted")));
    }

    /// <summary>
    /// Writes every partition to <paramref name="directory"/> as Arrow and returns a frame that
    /// reads them back, cutting the lineage. Partitions already written by an earlier run are
    /// reused, so a long pipeline that failed halfway resumes from its last checkpoint.
    /// </summary>
    /// <remarks>
    /// Files are written under a temporary name and renamed, so a file present under its final
    /// name is complete. A checkpoint is only valid for the same plan - delete the directory when
    /// the pipeline changes.
    /// </remarks>
    public DaskFrame Checkpoint(string directory, IScheduler? scheduler = null)
    {
        scheduler ??= Dask.CurrentScheduler;
        Directory.CreateDirectory(directory);
        var parts = Lower(scheduler);
        var paths = parts.Keys.Select((_, i) => Path.Combine(Path.GetFullPath(directory), $"part-{i:D5}.arrow")).ToArray();
        var missing = Enumerable.Range(0, paths.Length).Where(i => !File.Exists(paths[i])).ToArray();
        if (missing.Length > 0)
        {
            var write = typeof(FrameKernels).GetMethod(nameof(FrameKernels.WriteArrowPartition))!;
            var name = TaskKey.NewName("checkpoint");
            var tasks = missing.Select(i => TaskSpec.Call(new TaskKey(name, i), write, [new TaskRef(parts.Keys[i]), paths[i]], "checkpoint")).ToArray();
            var layer = new GraphLayer(name, tasks, [parts.Layer]);
            scheduler.ExecuteAsync(Graph(new Lowering.Parts(layer, tasks.Select(t => t.Key).ToArray())), tasks.Select(t => t.Key).ToArray())
                .GetAwaiter().GetResult();
        }
        return new DaskFrame(new ScanNode(new ArrowSource(paths, Plan.Schema?.ToArray() ?? ArrowFile.Read(paths[0]).ColumnNames.ToArray())));
    }

    /// <summary>Writes one CSV file per partition (<c>part-00000.csv</c>, ...) and returns the paths.</summary>
    public string[] ToCsv(string directory, IScheduler? scheduler = null)
        => Write(directory, "csv", nameof(FrameKernels.WriteCsvPartition), scheduler);

    /// <summary>Writes one Arrow IPC file per partition and returns the paths.</summary>
    public string[] ToArrow(string directory, IScheduler? scheduler = null)
        => Write(directory, "arrow", nameof(FrameKernels.WriteArrowPartition), scheduler);

    private string[] Write(string directory, string extension, string kernel, IScheduler? scheduler)
    {
        scheduler ??= Dask.CurrentScheduler;
        var parts = Lower(scheduler);
        var method = typeof(FrameKernels).GetMethod(kernel)!;
        var name = TaskKey.NewName("to-" + extension);
        var full = Path.GetFullPath(directory);
        var tasks = parts.Keys.Select((k, i) => TaskSpec.Call(new TaskKey(name, i), method,
            [new TaskRef(k), Path.Combine(full, $"part-{i:D5}.{extension}")], "to-" + extension)).ToArray();
        var layer = new GraphLayer(name, tasks, [parts.Layer]);
        var keys = tasks.Select(t => t.Key).ToArray();
        var results = scheduler.ExecuteAsync(Graph(new Lowering.Parts(layer, keys)), keys).GetAwaiter().GetResult();
        return keys.Select(k => (string)results[k]!).ToArray();
    }

    /// <summary>The partitions as delayed frames (the plan is optimised and lowered now).</summary>
    public Delayed<DataFrame>[] ToDelayed(IScheduler? scheduler = null)
    {
        var parts = Lower(scheduler ?? Dask.CurrentScheduler);
        return parts.Keys.Select(k => new Delayed<DataFrame>(parts.Layer, k)).ToArray();
    }

    // ------------------------------------------------------------------ inspection

    /// <summary>The logical plan, optimised by default. Compare <c>Explain(false)</c> with
    /// <c>Explain()</c> to see what the optimiser did.</summary>
    public string Explain(bool optimized = true) => (optimized ? Optimizer.Optimize(Plan) : Plan).Explain();

    /// <summary>The task graph that would run, as Graphviz DOT.</summary>
    public string Visualize(bool highLevel = false) => TaskGraphFor().ToDot(highLevel);

    /// <summary>The task graph that would run.</summary>
    public TaskGraph TaskGraphFor(IScheduler? scheduler = null) => Graph(Lower(scheduler ?? Dask.CurrentScheduler));

    /// <summary>Whether <c>Compute</c> optimises the plan first. On by default; off is for
    /// comparing results and for tests.</summary>
    public static bool OptimizePlans { get; set; } = true;

    internal Lowering.Parts Lower(IScheduler scheduler)
        => new Lowering(scheduler).Lower(OptimizePlans ? Optimizer.Optimize(Plan) : Plan);

    private static TaskGraph Graph(Lowering.Parts parts) => Dask.Graph(parts);

    private Expr CheckColumns(Expr e)
    {
        foreach (var c in e.Columns()) CheckColumn(c);
        return e;
    }

    private void CheckColumn(string column)
    {
        if (Plan.Schema is { } schema && !schema.Contains(column))
            throw new KeyNotFoundException($"Column '{column}' not found; available: {string.Join(", ", schema)}.");
    }

    /// <inheritdoc />
    public override string ToString()
        => $"DaskFrame({PartitionCount} partitions, columns: {(Plan.Schema is null ? "unknown" : string.Join(", ", Plan.Schema))})";
}

/// <summary>A <see cref="DaskFrame"/> grouped by key columns.</summary>
public sealed class DaskGroupBy
{
    private readonly DaskFrame _frame;
    private readonly string[] _keys;

    internal DaskGroupBy(DaskFrame frame, string[] keys)
    {
        _frame = frame;
        _keys = keys;
    }

    /// <summary>
    /// Aggregates each group. Partitions are reduced to per-group states in parallel and the
    /// states merged in a tree, so only one row per group per partition moves; the result is one
    /// partition, sorted by key.
    /// </summary>
    public DaskFrame Agg(params Aggregation[] aggregations)
    {
        foreach (var a in aggregations) a.Validate();
        return new DaskFrame(new AggregateNode(_frame.Plan, _keys, aggregations, DaskFrame.DefaultSplitEvery));
    }

    /// <summary>Aggregates with (column, how) pairs; output columns are named <c>column_how</c>.</summary>
    public DaskFrame Agg(params (string Column, string How)[] specs)
        => Agg(specs.Select(s => new Aggregation($"{s.Column}_{s.How}", s.How, s.Column)).ToArray());

    /// <summary>Sum per group.</summary>
    public DaskFrame Sum(params string[] columns) => Same("sum", columns);
    /// <summary>Mean per group.</summary>
    public DaskFrame Mean(params string[] columns) => Same("mean", columns);
    /// <summary>Minimum per group.</summary>
    public DaskFrame Min(params string[] columns) => Same("min", columns);
    /// <summary>Maximum per group.</summary>
    public DaskFrame Max(params string[] columns) => Same("max", columns);
    /// <summary>Sample standard deviation per group.</summary>
    public DaskFrame Std(params string[] columns) => Same("std", columns);
    /// <summary>Sample variance per group.</summary>
    public DaskFrame Var(params string[] columns) => Same("var", columns);
    /// <summary>Non-missing count per group.</summary>
    public DaskFrame Count(params string[] columns) => Same("count", columns);
    /// <summary>Rows per group, as column <paramref name="name"/>.</summary>
    public DaskFrame Size(string name = "size") => Agg(new Aggregation(name, "size", null));

    private DaskFrame Same(string how, string[] columns)
    {
        if (columns.Length == 0)
            columns = _frame.Columns.Where(c => !_keys.Contains(c)).ToArray();
        return Agg(columns.Select(c => new Aggregation(c, how, c)).ToArray());
    }
}
