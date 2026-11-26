#pragma warning disable CS8714 // Frequencies and Distinct key on T itself; null elements are the caller's to avoid.

namespace Gravicode.Science.DaskNet;

/// <summary>
/// A lazy, partitioned sequence of arbitrary objects (Dask's <c>dask.bag</c>): map, filter,
/// fold and group in parallel over data that need not fit in memory.
/// </summary>
/// <remarks>
/// <para>
/// Each partition is a <c>T[]</c> computed by one task. Element-wise operations add one task per
/// partition and fuse into a single pass when computed; reductions build a tree
/// (<see cref="GraphBuilders.TreeReduce"/>); <see cref="GroupBy{TKey}"/> and
/// <see cref="Distinct"/> shuffle every partition into every output partition.
/// </para>
/// <para>
/// Bags take ordinary delegates, so they run on the thread and synchronous schedulers. A cluster
/// cannot run a lambda: its closure has no form another process can load.
/// </para>
/// <para>LINQ query syntax works: <c>from x in bag where x &gt; 0 select x * 2</c>.</para>
/// </remarks>
public sealed class Bag<T> : IDaskCollection
{
    private readonly TaskKey[] _keys;

    internal Bag(GraphLayer layer, IReadOnlyList<TaskKey> keys)
    {
        Layer = layer;
        _keys = keys.ToArray();
    }

    /// <inheritdoc />
    public GraphLayer Layer { get; }

    /// <inheritdoc />
    public IReadOnlyList<TaskKey> OutputKeys => _keys;

    /// <summary>How many partitions.</summary>
    public int PartitionCount => _keys.Length;

    // ------------------------------------------------------------------ element-wise

    /// <summary>Applies a function to every element.</summary>
    public Bag<TResult> Map<TResult>(Func<T, TResult> selector)
    {
        ArgumentNullException.ThrowIfNull(selector);
        return MapPartitions(p => p.Select(selector), "map");
    }

    /// <summary>
    /// Applies a function to every element together with one shared value computed once and
    /// handed to every partition (a broadcast).
    /// </summary>
    public Bag<TResult> Map<TShared, TResult>(Delayed<TShared> shared, Func<T, TShared, TResult> selector)
    {
        ArgumentNullException.ThrowIfNull(selector);
        var name = TaskKey.NewName("map-broadcast");
        var keys = new TaskKey[_keys.Length];
        var tasks = new List<TaskSpec>();
        for (var i = 0; i < _keys.Length; i++)
        {
            keys[i] = new TaskKey(name, i);
            tasks.Add(TaskSpec.Create(keys[i], [_keys[i], shared.Key], v =>
            {
                var s = (TShared)v[1]!;
                return ((T[])v[0]!).Select(x => selector(x, s)).ToArray();
            }, "map-broadcast"));
        }
        return new Bag<TResult>(new GraphLayer(name, tasks, [Layer, shared.Layer]), keys);
    }

    /// <summary>Keeps the elements that satisfy <paramref name="predicate"/>.</summary>
    public Bag<T> Filter(Func<T, bool> predicate)
    {
        ArgumentNullException.ThrowIfNull(predicate);
        return MapPartitions(p => p.Where(predicate), "filter");
    }

    /// <summary>Replaces every element with zero or more elements.</summary>
    public Bag<TResult> FlatMap<TResult>(Func<T, IEnumerable<TResult>> selector)
    {
        ArgumentNullException.ThrowIfNull(selector);
        return MapPartitions(p => p.SelectMany(selector), "flatmap");
    }

    /// <summary>Applies a function to each whole partition.</summary>
    public Bag<TResult> MapPartitions<TResult>(Func<T[], IEnumerable<TResult>> function, string label = "map-partitions")
    {
        ArgumentNullException.ThrowIfNull(function);
        var name = TaskKey.NewName(label);
        var keys = new TaskKey[_keys.Length];
        var tasks = new TaskSpec[_keys.Length];
        for (var i = 0; i < _keys.Length; i++)
        {
            keys[i] = new TaskKey(name, i);
            tasks[i] = TaskSpec.Create(keys[i], [_keys[i]], v => function((T[])v[0]!).ToArray(), label);
        }
        return new Bag<TResult>(new GraphLayer(name, tasks, [Layer]), keys);
    }

    /// <summary>Each element paired with the value at the same position of <paramref name="other"/>.
    /// Both bags must have the same partitioning.</summary>
    public Bag<(T, TOther)> Zip<TOther>(Bag<TOther> other)
    {
        if (other.PartitionCount != PartitionCount)
            throw new ArgumentException("Zip needs bags with the same number of partitions.");
        var name = TaskKey.NewName("zip");
        var keys = new TaskKey[_keys.Length];
        var tasks = new TaskSpec[_keys.Length];
        for (var i = 0; i < _keys.Length; i++)
        {
            keys[i] = new TaskKey(name, i);
            var index = i;
            tasks[i] = TaskSpec.Create(keys[i], [_keys[i], other._keys[i]], v =>
            {
                var a = (T[])v[0]!;
                var b = (TOther[])v[1]!;
                if (a.Length != b.Length)
                    throw new InvalidOperationException($"Partition {index} has {a.Length} and {b.Length} elements; zip needs equal partitions.");
                return a.Zip(b).ToArray();
            }, "zip");
        }
        return new Bag<(T, TOther)>(new GraphLayer(name, tasks, [Layer, other.Layer]), keys);
    }

    // ------------------------------------------------------------------ reductions

    /// <summary>
    /// A general reduction: <paramref name="perPartition"/> summarises each partition,
    /// <paramref name="combine"/> merges summaries in a tree, <paramref name="finish"/> turns the
    /// last summary into the answer.
    /// </summary>
    public Delayed<TResult> Reduction<TPartial, TResult>(Func<T[], TPartial> perPartition,
        Func<TPartial[], TPartial> combine, Func<TPartial, TResult> finish, int splitEvery = 8, string label = "reduction")
    {
        var partialName = TaskKey.NewName(label + "-partial");
        var partials = new TaskSpec[_keys.Length];
        for (var i = 0; i < _keys.Length; i++)
            partials[i] = TaskSpec.Create(new TaskKey(partialName, i), [_keys[i]], v => perPartition((T[])v[0]!), label + "-partial");

        var treeName = TaskKey.NewName(label + "-aggregate");
        var (tree, root) = GraphBuilders.TreeReduce(treeName, partials.Select(p => p.Key).ToArray(), splitEvery,
            (key, inputs) => TaskSpec.Create(key, inputs, v => combine(v.Select(x => (TPartial)x!).ToArray()), label + "-aggregate"));

        var finalKey = new TaskKey(TaskKey.NewName(label));
        var final = TaskSpec.Create(finalKey, [root], v => finish((TPartial)v[0]!), label);

        var partialLayer = new GraphLayer(partialName, partials, [Layer]);
        var treeLayer = new GraphLayer(treeName, tree, [partialLayer]);
        return new Delayed<TResult>(new GraphLayer(finalKey.Name, [final], [treeLayer]), finalKey);
    }

    /// <summary>Folds the elements: <paramref name="step"/> within partitions, starting from a
    /// fresh <paramref name="seed"/> each, and <paramref name="combine"/> across them.</summary>
    /// <remarks>The seed is a factory because one mutable seed object shared by every partition
    /// would be folded into concurrently.</remarks>
    public Delayed<TAccumulate> Fold<TAccumulate>(Func<TAccumulate> seed, Func<TAccumulate, T, TAccumulate> step,
        Func<TAccumulate, TAccumulate, TAccumulate> combine, int splitEvery = 8)
        => Reduction(p => p.Aggregate(seed(), step), parts => parts.Aggregate(combine), x => x, splitEvery, "fold");

    /// <summary>Folds with an immutable seed value.</summary>
    public Delayed<TAccumulate> Fold<TAccumulate>(TAccumulate seed, Func<TAccumulate, T, TAccumulate> step,
        Func<TAccumulate, TAccumulate, TAccumulate> combine, int splitEvery = 8)
        => Fold(() => seed, step, combine, splitEvery);

    /// <summary>Reduces with an associative binary operator. Throws on an empty bag.</summary>
    public Delayed<T> Reduce(Func<T, T, T> op, int splitEvery = 8)
        => Reduction(p => p.Length == 0 ? new Option<T>() : new Option<T>(p.Aggregate(op)),
            parts => parts.Where(x => x.HasValue).Select(x => x.Value).DefaultIfEmptyOption(op),
            x => x.HasValue ? x.Value : throw new InvalidOperationException("Reduce of an empty bag."),
            splitEvery, "reduce");

    /// <summary>How many elements.</summary>
    public Delayed<long> Count()
        => Reduction(p => (long)p.Length, parts => parts.Sum(), x => x, label: "count");

    /// <summary>The sum of a numeric projection.</summary>
    public Delayed<double> Sum(Func<T, double> selector)
        => Reduction(p => p.Sum(selector), parts => parts.Sum(), x => x, label: "sum");

    /// <summary>The mean of a numeric projection.</summary>
    public Delayed<double> Mean(Func<T, double> selector)
        => Reduction(p => (Count: (long)p.Length, Sum: p.Sum(selector)),
            parts => (parts.Sum(x => x.Count), parts.Sum(x => x.Sum)),
            x => x.Count == 0 ? double.NaN : x.Sum / x.Count, label: "mean");

    /// <summary>The variance of a numeric projection (Chan's parallel update, numerically stable).</summary>
    public Delayed<double> Var(Func<T, double> selector, int ddof = 0)
        => Reduction(p => Moments.Of(p.Select(selector)), parts => parts.Aggregate(Moments.Merge),
            m => m.Variance(ddof), label: "var");

    /// <summary>The standard deviation of a numeric projection.</summary>
    public Delayed<double> Std(Func<T, double> selector, int ddof = 0)
        => Var(selector, ddof).Then(Math.Sqrt, "std");

    /// <summary>The smallest numeric projection.</summary>
    public Delayed<double> Min(Func<T, double> selector)
        => Reduction(p => p.Length == 0 ? double.PositiveInfinity : p.Min(selector), parts => parts.Min(), x => x, label: "min");

    /// <summary>The largest numeric projection.</summary>
    public Delayed<double> Max(Func<T, double> selector)
        => Reduction(p => p.Length == 0 ? double.NegativeInfinity : p.Max(selector), parts => parts.Max(), x => x, label: "max");

    /// <summary>Whether any element satisfies <paramref name="predicate"/>.</summary>
    public Delayed<bool> Any(Func<T, bool> predicate)
        => Reduction(p => p.Any(predicate), parts => parts.Any(x => x), x => x, label: "any");

    /// <summary>Whether every element satisfies <paramref name="predicate"/>.</summary>
    public Delayed<bool> All(Func<T, bool> predicate)
        => Reduction(p => p.All(predicate), parts => parts.All(x => x), x => x, label: "all");

    /// <summary>The <paramref name="k"/> elements with the largest key, largest first.</summary>
    public Delayed<T[]> TopK(int k, Func<T, double> key)
        => Reduction(p => p.OrderByDescending(key).Take(k).ToArray(),
            parts => parts.SelectMany(x => x).OrderByDescending(key).Take(k).ToArray(), x => x, label: "topk");

    /// <summary>How often each distinct element occurs, most frequent first.</summary>
    public Delayed<(T Item, long Count)[]> Frequencies()
        => Reduction(p => p.GroupBy(x => x).ToDictionary(g => g.Key!, g => (long)g.Count()),
            parts =>
            {
                var merged = new Dictionary<T, long>();
                foreach (var d in parts)
                    foreach (var (k, c) in d) merged[k] = merged.GetValueOrDefault(k) + c;
                return merged;
            },
            d => d.Select(p => (p.Key, p.Value)).OrderByDescending(p => p.Value).ToArray(), label: "frequencies");

    /// <summary>
    /// Reduces by key without a full shuffle (Dask's <c>foldby</c>): each partition folds its own
    /// elements per key, then the per-key states are merged in a tree.
    /// </summary>
    /// <remarks>Much cheaper than <see cref="GroupBy{TKey}"/> followed by a reduction, because
    /// only one state per key leaves each partition, not every element.</remarks>
    public Delayed<(TKey Key, TAccumulate Value)[]> FoldBy<TKey, TAccumulate>(Func<T, TKey> keySelector,
        Func<TAccumulate> seed, Func<TAccumulate, T, TAccumulate> step, Func<TAccumulate, TAccumulate, TAccumulate> combine,
        int splitEvery = 8) where TKey : notnull
        => Reduction(p =>
            {
                var states = new Dictionary<TKey, TAccumulate>();
                foreach (var x in p)
                {
                    var key = keySelector(x);
                    states[key] = step(states.TryGetValue(key, out var s) ? s : seed(), x);
                }
                return states;
            },
            parts =>
            {
                var merged = new Dictionary<TKey, TAccumulate>();
                foreach (var d in parts)
                    foreach (var (k, v) in d) merged[k] = merged.TryGetValue(k, out var s) ? combine(s, v) : v;
                return merged;
            },
            d => d.Select(p => (p.Key, p.Value)).ToArray(), splitEvery, "foldby");

    // ------------------------------------------------------------------ shuffles

    /// <summary>
    /// Groups elements by key (Dask's <c>groupby</c>): a full shuffle, after which every key's
    /// elements sit in exactly one output partition.
    /// </summary>
    public Bag<(TKey Key, T[] Items)> GroupBy<TKey>(Func<T, TKey> keySelector, int? partitions = null) where TKey : notnull
    {
        ArgumentNullException.ThrowIfNull(keySelector);
        var shuffled = Shuffle(keySelector, partitions ?? PartitionCount, "groupby");
        return shuffled.MapPartitions(p => p.GroupBy(keySelector).Select(g => (g.Key, g.ToArray())), "groupby-collect");
    }

    /// <summary>The distinct elements, via a shuffle so each value is compared in one partition.</summary>
    public Bag<T> Distinct()
        => Shuffle(x => x!, PartitionCount, "distinct").MapPartitions(p => p.Distinct(), "distinct-collect");

    /// <summary>Moves every element to the partition its key hashes to.</summary>
    public Bag<T> Shuffle<TKey>(Func<T, TKey> keySelector, int partitions, string label = "shuffle") where TKey : notnull
    {
        if (partitions <= 0) throw new ArgumentOutOfRangeException(nameof(partitions));
        var splitName = TaskKey.NewName(label + "-split");
        var splits = new TaskSpec[_keys.Length];
        for (var i = 0; i < _keys.Length; i++)
            splits[i] = TaskSpec.Create(new TaskKey(splitName, i), [_keys[i]], v =>
            {
                var buckets = new List<T>[partitions];
                for (var b = 0; b < partitions; b++) buckets[b] = [];
                foreach (var x in (T[])v[0]!)
                {
                    var h = EqualityComparer<TKey>.Default.GetHashCode(keySelector(x)!) & int.MaxValue;
                    buckets[h % partitions].Add(x);
                }
                return buckets.Select(b => b.ToArray()).ToArray();
            }, label + "-split");

        var splitLayer = new GraphLayer(splitName, splits, [Layer]);
        var joinName = TaskKey.NewName(label);
        var keys = new TaskKey[partitions];
        var joins = new TaskSpec[partitions];
        var splitKeys = splits.Select(s => s.Key).ToArray();
        for (var j = 0; j < partitions; j++)
        {
            keys[j] = new TaskKey(joinName, j);
            var bucket = j;
            joins[j] = TaskSpec.Create(keys[j], splitKeys, v => v.SelectMany(s => ((T[][])s!)[bucket]).ToArray(), label);
        }
        return new Bag<T>(new GraphLayer(joinName, joins, [splitLayer]), keys);
    }

    /// <summary>Regroups the elements into <paramref name="partitions"/> partitions of near-equal size,
    /// keeping their order.</summary>
    public Bag<T> Repartition(int partitions)
    {
        if (partitions <= 0) throw new ArgumentOutOfRangeException(nameof(partitions));
        var allName = TaskKey.NewName("repartition-gather");
        var all = TaskSpec.Create(new TaskKey(allName), _keys, v => v.SelectMany(x => (T[])x!).ToArray(), "repartition-gather");
        var name = TaskKey.NewName("repartition");
        var keys = new TaskKey[partitions];
        var tasks = new TaskSpec[partitions];
        for (var j = 0; j < partitions; j++)
        {
            keys[j] = new TaskKey(name, j);
            var part = j;
            tasks[j] = TaskSpec.Create(keys[j], [all.Key], v =>
            {
                var items = (T[])v[0]!;
                var (start, count) = Bag.Slice(items.Length, partitions, part);
                return items.AsSpan(start, count).ToArray();
            }, "repartition");
        }
        var gather = new GraphLayer(allName, [all], [Layer]);
        return new Bag<T>(new GraphLayer(name, tasks, [gather]), keys);
    }

    // ------------------------------------------------------------------ results

    /// <summary>The partitions as delayed arrays.</summary>
    public Delayed<T[]>[] ToDelayed() => _keys.Select(k => new Delayed<T[]>(Layer, k)).ToArray();

    /// <summary>Computes every element, in partition order.</summary>
    public List<T> Compute(IScheduler? scheduler = null, CancellationToken cancellationToken = default)
    {
        var results = Dask.Run([this], scheduler, cancellationToken);
        return _keys.SelectMany(k => (T[])results[k]!).ToList();
    }

    /// <summary>The first <paramref name="count"/> elements, computing only as many leading
    /// partitions as that needs.</summary>
    public List<T> Take(int count, IScheduler? scheduler = null)
    {
        var taken = new List<T>(count);
        foreach (var key in _keys)
        {
            if (taken.Count >= count) break;
            var single = new Bag<T>(Layer, [key]);
            taken.AddRange(single.Compute(scheduler).Take(count - taken.Count));
        }
        return taken;
    }

    /// <summary>Computes now and returns a bag over the in-memory partitions.</summary>
    public Bag<T> Persist(IScheduler? scheduler = null)
    {
        var results = Dask.Run([this], scheduler, CancellationToken.None);
        return Bag.FromPartitions(_keys.Select(k => (T[])results[k]!).ToArray());
    }

    /// <summary>The graph that would run, as Graphviz DOT.</summary>
    public string Visualize(bool highLevel = false) => Dask.Graph(this).ToDot(highLevel);

    /// <inheritdoc />
    public override string ToString() => $"Bag<{typeof(T).Name}>({PartitionCount} partitions)";
}

/// <summary>Creating bags.</summary>
public static class Bag
{
    /// <summary>Spreads a sequence over <paramref name="partitions"/> partitions of near-equal size.</summary>
    public static Bag<T> FromSequence<T>(IEnumerable<T> items, int partitions = 8)
    {
        if (partitions <= 0) throw new ArgumentOutOfRangeException(nameof(partitions));
        var all = items.ToArray();
        partitions = Math.Max(1, Math.Min(partitions, Math.Max(all.Length, 1)));
        return FromPartitions(Enumerable.Range(0, partitions).Select(p =>
        {
            var (start, count) = Slice(all.Length, partitions, p);
            return all.AsSpan(start, count).ToArray();
        }).ToArray());
    }

    /// <summary>A bag whose partitions are given arrays.</summary>
    public static Bag<T> FromPartitions<T>(IReadOnlyList<T[]> partitions)
    {
        var name = TaskKey.NewName("from-sequence");
        var tasks = partitions.Select((p, i) => TaskSpec.Value(new TaskKey(name, i), p, "from-sequence")).ToArray();
        return new Bag<T>(new GraphLayer(name, tasks), tasks.Select(t => t.Key).ToArray());
    }

    /// <summary>A bag whose partitions are computed by delayed values.</summary>
    public static Bag<T> FromDelayed<T>(IEnumerable<Delayed<T[]>> partitions)
    {
        var list = partitions.ToArray();
        var name = TaskKey.NewName("from-delayed");
        var tasks = list.Select((d, i) => TaskSpec.Create(new TaskKey(name, i), [d.Key], v => v[0], "from-delayed")).ToArray();
        return new Bag<T>(new GraphLayer(name, tasks, list.Select(d => d.Layer)), tasks.Select(t => t.Key).ToArray());
    }

    /// <summary>The integers <c>0 .. count-1</c>, generated inside the tasks.</summary>
    public static Bag<int> Range(int count, int partitions = 8)
    {
        partitions = Math.Max(1, Math.Min(partitions, Math.Max(count, 1)));
        var name = TaskKey.NewName("range");
        var tasks = Enumerable.Range(0, partitions).Select(p =>
        {
            var (start, n) = Slice(count, partitions, p);
            return TaskSpec.Create(new TaskKey(name, p), [], _ => Enumerable.Range(start, n).ToArray(), "range");
        }).ToArray();
        return new Bag<int>(new GraphLayer(name, tasks), tasks.Select(t => t.Key).ToArray());
    }

    /// <summary>
    /// The lines of one or more text files, one partition per file or per
    /// <paramref name="blockSize"/> bytes of a file (Dask's <c>read_text</c>).
    /// </summary>
    /// <param name="pattern">A file, a directory or a glob such as <c>logs/*.txt</c>.</param>
    /// <param name="blockSize">Split files into blocks of about this many bytes; 0 keeps whole files.</param>
    public static Bag<string> ReadText(string pattern, long blockSize = 0)
        => ReadText([pattern], blockSize);

    /// <summary>The lines of several files or patterns.</summary>
    public static Bag<string> ReadText(IEnumerable<string> patterns, long blockSize = 0)
    {
        var files = GraphBuilders.ExpandPaths(patterns);
        var name = TaskKey.NewName("read-text");
        var tasks = new List<TaskSpec>();
        foreach (var file in files)
            foreach (var (start, end) in GraphBuilders.ByteRanges(new FileInfo(file).Length, blockSize))
            {
                var (path, s, e) = (file, start, end);
                tasks.Add(TaskSpec.Create(new TaskKey(name, tasks.Count), [],
                    _ => TextBlocks.ReadLines(path, s, e).ToArray(), "read-text"));
            }
        return new Bag<string>(new GraphLayer(name, tasks), tasks.Select(t => t.Key).ToArray());
    }

    /// <summary>Concatenates bags, keeping every partition.</summary>
    public static Bag<T> Concat<T>(params Bag<T>[] bags)
    {
        var name = TaskKey.NewName("concat");
        var tasks = bags.SelectMany(b => b.OutputKeys).Select((k, i) =>
            TaskSpec.Create(new TaskKey(name, i), [k], v => v[0], "concat")).ToArray();
        return new Bag<T>(new GraphLayer(name, tasks, bags.Select(b => b.Layer)), tasks.Select(t => t.Key).ToArray());
    }

    /// <summary>The <paramref name="part"/>-th of <paramref name="parts"/> near-equal slices of
    /// <paramref name="length"/> items; the remainder goes one each to the first slices.</summary>
    internal static (int Start, int Count) Slice(int length, int parts, int part)
    {
        var baseSize = length / parts;
        var extra = length % parts;
        var start = part * baseSize + Math.Min(part, extra);
        return (start, baseSize + (part < extra ? 1 : 0));
    }

    // ------------------------------------------------------------------ LINQ

    /// <summary>LINQ <c>select</c>.</summary>
    public static Bag<TResult> Select<T, TResult>(this Bag<T> bag, Func<T, TResult> selector) => bag.Map(selector);

    /// <summary>LINQ <c>where</c>.</summary>
    public static Bag<T> Where<T>(this Bag<T> bag, Func<T, bool> predicate) => bag.Filter(predicate);

    /// <summary>LINQ <c>from ... from ...</c>.</summary>
    public static Bag<TResult> SelectMany<T, TResult>(this Bag<T> bag, Func<T, IEnumerable<TResult>> selector)
        => bag.FlatMap(selector);

    /// <summary>LINQ <c>from ... from ... select</c>.</summary>
    public static Bag<TResult> SelectMany<T, TCollection, TResult>(this Bag<T> bag,
        Func<T, IEnumerable<TCollection>> collection, Func<T, TCollection, TResult> result)
        => bag.FlatMap(x => collection(x).Select(c => result(x, c)));
}

/// <summary>A value that may be absent; used where a partition can be empty.</summary>
internal readonly record struct Option<T>(bool HasValue, T Value)
{
    public Option(T value) : this(true, value) { }
}

internal static class OptionExtensions
{
    public static Option<T> DefaultIfEmptyOption<T>(this IEnumerable<T> values, Func<T, T, T> op)
    {
        var list = values.ToList();
        return list.Count == 0 ? new Option<T>() : new Option<T>(list.Aggregate(op));
    }
}

/// <summary>Count, mean and sum of squared deviations, mergeable across partitions.</summary>
/// <remarks>
/// Merged with Chan et al.'s pairwise formula rather than from a sum of squares, which cancels
/// catastrophically when the mean is large relative to the spread (a series sitting at 1e9 with
/// unit noise has no correct digits left in <c>Σx² - n·mean²</c>).
/// </remarks>
public readonly record struct Moments(long Count, double Mean, double M2)
{
    /// <summary>The moments of a sequence (Welford's update).</summary>
    /// <param name="values">The values.</param>
    /// <param name="skipMissing">Skip NaN as missing (pandas) rather than propagate it (NumPy).</param>
    public static Moments Of(IEnumerable<double> values, bool skipMissing = true)
    {
        // Welford on values shifted by the first one: deviations from a nearby point are small
        // and exactly representable, so a series sitting at 1e9 keeps its digits.
        long n = 0;
        double shift = 0, mean = 0, m2 = 0;
        foreach (var value in values)
        {
            if (skipMissing && double.IsNaN(value)) continue;
            if (n == 0) shift = value;
            var x = value - shift;
            n++;
            var delta = x - mean;
            mean += delta / n;
            m2 += delta * (x - mean);
        }
        return new Moments(n, n == 0 ? 0 : shift + mean, m2);
    }

    /// <summary>The moments of two disjoint sets together.</summary>
    public static Moments Merge(Moments a, Moments b)
    {
        if (a.Count == 0) return b;
        if (b.Count == 0) return a;
        var n = a.Count + b.Count;
        var delta = b.Mean - a.Mean;
        var mean = a.Mean + delta * b.Count / n;
        var m2 = a.M2 + b.M2 + delta * delta * ((double)a.Count * b.Count / n);
        return new Moments(n, mean, m2);
    }

    /// <summary>The variance with <paramref name="ddof"/> delta degrees of freedom.</summary>
    public double Variance(int ddof = 0) => Count - ddof <= 0 ? double.NaN : M2 / (Count - ddof);
}
