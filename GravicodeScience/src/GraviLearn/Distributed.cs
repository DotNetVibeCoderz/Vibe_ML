using System.Buffers.Binary;
using Gravicode.Science.GraviLearn.Trees;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn.Distributed;

/// <summary>A contiguous range of work assigned to one worker.</summary>
/// <param name="Start">First index, inclusive.</param>
/// <param name="Count">How many items.</param>
public readonly record struct Shard(int Start, int Count)
{
    /// <summary>One past the last index.</summary>
    public int End => Start + Count;

    /// <summary>The indices this shard covers.</summary>
    public IEnumerable<int> Indices => Enumerable.Range(Start, Count);

    /// <inheritdoc />
    public override string ToString() => $"[{Start}, {End})";
}

/// <summary>
/// How workers exchange results between rounds.
/// </summary>
/// <remarks>
/// The transport is separated from the algorithm because the algorithm does not care. Data-parallel
/// training is "everyone computes a partial result, someone combines them", and whether the workers
/// are threads, processes on one machine, or processes on several is a deployment question rather
/// than a modelling one.
/// </remarks>
public interface IWorkerTransport : IDisposable
{
    /// <summary>How many workers are participating.</summary>
    int WorkerCount { get; }

    /// <summary>Publishes one worker's result for a round.</summary>
    void Publish(int round, int worker, byte[] payload);

    /// <summary>
    /// Waits for every worker's result for a round and returns them in worker order.
    /// </summary>
    /// <remarks>
    /// Ordered by worker rather than by arrival, because floating-point addition is not
    /// associative: summing gradients in arrival order makes the result depend on scheduling, and
    /// two runs of the same job would then differ in the last bits.
    /// </remarks>
    IReadOnlyList<byte[]> Collect(int round);
}

/// <summary>Workers as threads in this process, exchanging results in memory.</summary>
/// <remarks>
/// The right transport when the bottleneck is CPU rather than memory, and the one to test against:
/// it removes serialisation from the picture, so a disagreement with the single-process result is
/// the algorithm's fault and nothing else's.
/// </remarks>
public sealed class InProcessTransport(int workerCount) : IWorkerTransport
{
    private readonly Dictionary<(int Round, int Worker), byte[]> _published = [];
    private readonly Lock _gate = new();

    /// <inheritdoc />
    public int WorkerCount { get; } = workerCount > 0
        ? workerCount
        : throw new ArgumentOutOfRangeException(nameof(workerCount));

    /// <inheritdoc />
    public void Publish(int round, int worker, byte[] payload)
    {
        lock (_gate) _published[(round, worker)] = payload;
    }

    /// <inheritdoc />
    public IReadOnlyList<byte[]> Collect(int round)
    {
        lock (_gate)
        {
            var results = new byte[WorkerCount][];

            for (var w = 0; w < WorkerCount; w++)
                results[w] = _published.TryGetValue((round, w), out var payload)
                    ? payload
                    : throw new InvalidOperationException($"Worker {w} published nothing for round {round}.");

            return results;
        }
    }

    /// <inheritdoc />
    public void Dispose() { }
}

/// <summary>
/// Workers as separate processes, exchanging results through a shared directory.
/// </summary>
/// <remarks>
/// <para>
/// Deliberately the least clever transport that actually crosses a process boundary. A worker
/// writes its payload to a file and then renames it into place; the collector polls for the
/// finished names. That is enough to work across processes on one machine and across machines
/// sharing a filesystem, with no broker, no ports and no protocol to get wrong.
/// </para>
/// <para>
/// The rename is what makes it safe. Writing directly to the final name lets a collector observe a
/// half-written file and read a truncated payload — on a local filesystem a rename is atomic, so a
/// name either does not exist or names a complete file.
/// </para>
/// <para>
/// It is not fast. Every round costs a write, a rename and a poll per worker, so this suits rounds
/// measured in seconds — which is what tree building and full-batch gradient steps are — and not a
/// per-minibatch exchange.
/// </para>
/// </remarks>
public sealed class FileTransport : IWorkerTransport, IDisposable
{
    private readonly string _directory;
    private readonly bool _owned;
    private readonly TimeSpan _timeout;

    /// <summary>Opens or creates a shared exchange directory.</summary>
    /// <param name="directory">A path every worker can see.</param>
    /// <param name="workerCount">How many workers will publish each round.</param>
    /// <param name="timeout">How long <see cref="Collect"/> waits before giving up.</param>
    public FileTransport(string directory, int workerCount, TimeSpan? timeout = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        if (workerCount <= 0) throw new ArgumentOutOfRangeException(nameof(workerCount));

        _directory = directory;
        _owned = !Directory.Exists(directory);
        _timeout = timeout ?? TimeSpan.FromMinutes(5);

        Directory.CreateDirectory(directory);
        WorkerCount = workerCount;
    }

    /// <inheritdoc />
    public int WorkerCount { get; }

    private string PathFor(int round, int worker) => Path.Combine(_directory, $"r{round:D5}-w{worker:D4}.bin");

    /// <inheritdoc />
    public void Publish(int round, int worker, byte[] payload)
    {
        var final = PathFor(round, worker);
        var staging = final + ".partial";

        File.WriteAllBytes(staging, payload);

        // Rename rather than write in place: a collector must never see a half-written payload.
        File.Move(staging, final, overwrite: true);
    }

    /// <inheritdoc />
    public IReadOnlyList<byte[]> Collect(int round)
    {
        var deadline = DateTime.UtcNow + _timeout;
        var results = new byte[WorkerCount][];

        for (var w = 0; w < WorkerCount; w++)
        {
            var path = PathFor(round, w);

            while (!File.Exists(path))
            {
                if (DateTime.UtcNow > deadline)
                    throw new TimeoutException(
                        $"Worker {w} published nothing for round {round} within {_timeout}.");

                Thread.Sleep(20);
            }

            results[w] = File.ReadAllBytes(path);
        }

        return results;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_owned && Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }
}

/// <summary>
/// The parts of data-parallel training that are the same whatever is being trained.
/// </summary>
public static class DataParallel
{
    /// <summary>
    /// Splits <paramref name="items"/> across <paramref name="workers"/> as evenly as possible.
    /// </summary>
    /// <remarks>
    /// The remainder goes to the first shards one item each, rather than all to the last. Dumping
    /// it on one worker makes that worker the critical path, and with 8 workers and 100 items the
    /// difference is a shard of 13 against a shard of 16.
    /// </remarks>
    public static Shard[] Partition(int items, int workers)
    {
        if (items < 0) throw new ArgumentOutOfRangeException(nameof(items));
        if (workers <= 0) throw new ArgumentOutOfRangeException(nameof(workers));

        var shards = new Shard[workers];
        var baseSize = items / workers;
        var remainder = items % workers;

        var start = 0;
        for (var w = 0; w < workers; w++)
        {
            var count = baseSize + (w < remainder ? 1 : 0);
            shards[w] = new Shard(start, count);
            start += count;
        }

        return shards;
    }

    /// <summary>
    /// Combines per-worker mean gradients into the global mean.
    /// </summary>
    /// <param name="gradients">Each worker's mean gradient over its own shard.</param>
    /// <param name="sampleCounts">How many samples each worker averaged over.</param>
    /// <remarks>
    /// <para>
    /// <b>Weighting by sample count is not a refinement, it is the whole correctness condition.</b>
    /// A plain average of per-worker means equals the global mean only when every shard is the same
    /// size. With uneven shards — which <see cref="Partition"/> produces whenever the count does not
    /// divide — an unweighted average silently over-weights the small shards, and the model trains
    /// to something slightly wrong in a way no shape or convergence check would reveal.
    /// </para>
    /// <para>
    /// Summed in worker order, because floating-point addition is not associative and arrival order
    /// would make the result depend on scheduling.
    /// </para>
    /// </remarks>
    public static NdArray AverageGradients(IReadOnlyList<NdArray> gradients, IReadOnlyList<int> sampleCounts)
    {
        ArgumentNullException.ThrowIfNull(gradients);
        ArgumentNullException.ThrowIfNull(sampleCounts);

        if (gradients.Count == 0) throw new ArgumentException("There is nothing to average.", nameof(gradients));
        if (gradients.Count != sampleCounts.Count)
            throw new ArgumentException("There must be one sample count per gradient.", nameof(sampleCounts));

        var total = sampleCounts.Sum();
        if (total <= 0) throw new ArgumentException("The shards hold no samples.", nameof(sampleCounts));

        var shape = gradients[0].Shape.ToArray();
        var result = NdArray.Zeros(shape);

        for (var w = 0; w < gradients.Count; w++)
        {
            if (!gradients[w].Shape.SequenceEqual(shape))
                throw new ArgumentException(
                    $"Worker {w} produced a gradient of shape [{string.Join(", ", gradients[w].Shape.ToArray())}], "
                    + $"but worker 0 produced [{string.Join(", ", shape)}].", nameof(gradients));

            var weight = (double)sampleCounts[w] / total;
            for (var i = 0; i < result.Size; i++) result.SetAt(i, result.At(i) + gradients[w].At(i) * weight);
        }

        return result;
    }

    /// <summary>Serialises an array for a transport.</summary>
    public static byte[] Encode(NdArray array)
    {
        var shape = array.Shape.ToArray();
        var bytes = new byte[4 + shape.Length * 4 + array.Size * 8];

        BinaryPrimitives.WriteInt32LittleEndian(bytes, shape.Length);
        for (var i = 0; i < shape.Length; i++)
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(4 + i * 4), shape[i]);

        var offset = 4 + shape.Length * 4;
        for (var i = 0; i < array.Size; i++)
            BinaryPrimitives.WriteDoubleLittleEndian(bytes.AsSpan(offset + i * 8), array.At(i));

        return bytes;
    }

    /// <summary>Reads an array written by <see cref="Encode"/>.</summary>
    public static NdArray Decode(byte[] bytes)
    {
        var rank = BinaryPrimitives.ReadInt32LittleEndian(bytes);
        var shape = new int[rank];

        for (var i = 0; i < rank; i++)
            shape[i] = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(4 + i * 4));

        var offset = 4 + rank * 4;
        var result = NdArray.Zeros(shape);

        for (var i = 0; i < result.Size; i++)
            result.SetAt(i, BinaryPrimitives.ReadDoubleLittleEndian(bytes.AsSpan(offset + i * 8)));

        return result;
    }
}

/// <summary>
/// Averages gradients across workers, one round per optimisation step.
/// </summary>
/// <remarks>
/// The synchronous variant: every worker computes a gradient over its shard, publishes it, and
/// waits for the average before stepping. That keeps every worker's parameters identical at every
/// step, which is what makes the result reproducible and equal to single-process training.
/// Asynchronous schemes are faster and give up exactly that.
/// </remarks>
public sealed class ParameterServer(IWorkerTransport transport)
{
    private int _round;

    /// <summary>How many rounds have completed.</summary>
    public int Round => _round;

    /// <summary>The transport in use.</summary>
    public IWorkerTransport Transport { get; } = transport;

    /// <summary>Publishes one worker's gradient for the current round.</summary>
    public void Contribute(int worker, NdArray gradient, int sampleCount)
    {
        var payload = DataParallel.Encode(gradient);
        var envelope = new byte[payload.Length + 4];

        BinaryPrimitives.WriteInt32LittleEndian(envelope, sampleCount);
        payload.CopyTo(envelope.AsSpan(4));

        Transport.Publish(_round, worker, envelope);
    }

    /// <summary>Waits for every worker and returns the sample-weighted mean gradient.</summary>
    public NdArray Aggregate()
    {
        var envelopes = Transport.Collect(_round);

        var gradients = new List<NdArray>(envelopes.Count);
        var counts = new List<int>(envelopes.Count);

        foreach (var envelope in envelopes)
        {
            counts.Add(BinaryPrimitives.ReadInt32LittleEndian(envelope));
            gradients.Add(DataParallel.Decode(envelope[4..]));
        }

        _round++;
        return DataParallel.AverageGradients(gradients, counts);
    }
}

/// <summary>
/// A random forest whose trees are grown by several workers.
/// </summary>
/// <remarks>
/// <para>
/// The easiest thing in this file to distribute and the one with the strongest guarantee. Every
/// tree in a random forest is independent of every other, and <see cref="RandomForestClassifier"/>
/// seeds tree <c>t</c> from <c>seed + t * 7919</c> — a function of the index alone. So a worker
/// given tree indices 40 to 79 grows exactly the trees a single process would have grown at those
/// positions, and the assembled forest is <b>bit-identical</b> to the single-process one.
/// </para>
/// <para>
/// That is a stronger claim than "equivalent" and worth having, because it means distribution
/// cannot change an answer. Anything that depends on the worker count — a different split, a
/// different accuracy — would be a bug rather than a rounding difference.
/// </para>
/// <para>
/// What is not distributed is the data: every worker fits on the whole training set. This shards
/// the <em>compute</em>, not the memory, which is the right way round for forests — the trees are
/// the expensive part and the dataset usually fits.
/// </para>
/// </remarks>
public sealed class DistributedForest(
    int nTrees = 100,
    int maxDepth = 0,
    int minSamplesSplit = 2,
    int minSamplesLeaf = 1,
    int maxFeatures = 0,
    int seed = 42,
    SplitCriterion criterion = SplitCriterion.Gini)
{
    private DecisionTree[] _trees = [];
    private double[] _classes = [];

    /// <summary>Total trees across every worker.</summary>
    public int TreeCount { get; } = nTrees;

    /// <summary>Number of features seen during training.</summary>
    public int FeatureCount { get; private set; }

    /// <summary>The distinct class labels, ascending.</summary>
    public IReadOnlyList<double> Classes => _classes;

    /// <summary>The assembled trees.</summary>
    public IReadOnlyList<DecisionTree> Trees => _trees;

    /// <summary>True once fitted.</summary>
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Grows the trees a single worker is responsible for.
    /// </summary>
    /// <param name="x">Features. Every worker sees the whole training set.</param>
    /// <param name="y">Targets.</param>
    /// <param name="shard">Which tree indices this worker owns.</param>
    /// <param name="classes">The global class set, so every worker agrees on column order.</param>
    /// <remarks>
    /// The class set is passed in rather than derived, because a worker whose shard happens to miss
    /// a rare class would otherwise build trees with a different number of output columns, and the
    /// assembled forest would vote over misaligned probabilities.
    /// </remarks>
    public static DecisionTree[] GrowShard(NdArray x, NdArray y, Shard shard, IReadOnlyList<double> classes,
        int maxDepth = 0, int minSamplesSplit = 2, int minSamplesLeaf = 1, int maxFeatures = 0,
        int seed = 42, SplitCriterion criterion = SplitCriterion.Gini)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);

        var samples = x.Shape[0];
        var featuresPerSplit = maxFeatures > 0 ? maxFeatures : Math.Max(1, (int)Math.Sqrt(x.Shape[1]));

        var trees = new DecisionTree[shard.Count];

        Parallel.For(0, shard.Count, local =>
        {
            var t = shard.Start + local;

            // Seeded from the GLOBAL tree index, which is what makes the shard boundary invisible.
            var treeSeed = seed + t * 7919;
            var rng = new GraviRandom(treeSeed);
            var indices = rng.Choice(samples, samples, replace: true);

            var tree = new DecisionTree(criterion, maxDepth, minSamplesSplit, minSamplesLeaf,
                featuresPerSplit, treeSeed);

            tree.FitOnIndices(x, y, indices, [.. classes]);
            trees[local] = tree;
        });

        return trees;
    }

    /// <summary>
    /// Trains across <paramref name="workers"/> workers and assembles the forest.
    /// </summary>
    /// <remarks>
    /// The coordinator runs the shards itself here. In a real deployment each shard runs in its own
    /// process and the trees come back through an <see cref="IWorkerTransport"/>; the split into
    /// <see cref="GrowShard"/> and this method is what makes that substitution possible without
    /// changing the result.
    /// </remarks>
    public DistributedForest Fit(NdArray x, NdArray y, int workers)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);

        if (workers <= 0) throw new ArgumentOutOfRangeException(nameof(workers));
        if (x.Rank != 2) throw new ArgumentException("x must be a rank 2 array.", nameof(x));
        if (y.Size != x.Shape[0]) throw new ArgumentException("There must be one label per row.", nameof(y));

        FeatureCount = x.Shape[1];
        _classes = [.. y.ToArray().Distinct().Order()];

        var shards = DataParallel.Partition(TreeCount, workers);
        var perShard = new DecisionTree[shards.Length][];

        for (var w = 0; w < shards.Length; w++)
            perShard[w] = GrowShard(x, y, shards[w], _classes,
                maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures, seed, criterion);

        // Concatenated in shard order, so the ensemble is ordered by global tree index.
        _trees = [.. perShard.SelectMany(t => t)];

        IsFitted = true;
        return this;
    }

    /// <summary>Class probabilities, averaged over the ensemble.</summary>
    public NdArray PredictProbabilities(NdArray x)
    {
        RequireFitted();

        var result = NdArray.Zeros(x.Shape[0], _classes.Length);

        foreach (var tree in _trees)
        {
            var votes = tree.PredictProbabilities(x);
            for (var i = 0; i < x.Shape[0]; i++)
                for (var c = 0; c < _classes.Length; c++)
                    result[i, c] += votes[i, c] / _trees.Length;
        }

        return result;
    }

    /// <summary>The predicted class per row.</summary>
    public NdArray Predict(NdArray x)
    {
        var probabilities = PredictProbabilities(x);
        var result = NdArray.Zeros(x.Shape[0]);

        for (var i = 0; i < x.Shape[0]; i++)
        {
            var best = 0;
            for (var c = 1; c < _classes.Length; c++)
                if (probabilities[i, c] > probabilities[i, best]) best = c;

            result.SetAt(i, _classes[best]);
        }

        return result;
    }

    /// <summary>Fraction of rows predicted correctly.</summary>
    public double Score(NdArray x, NdArray y)
    {
        var predictions = Predict(x);
        var correct = 0;

        for (var i = 0; i < y.Size; i++)
            if (Math.Abs(predictions.At(i) - y.At(i)) < 1e-9) correct++;

        return (double)correct / y.Size;
    }

    private void RequireFitted()
    {
        if (!IsFitted) throw new InvalidOperationException("The forest must be fitted before use.");
    }
}
