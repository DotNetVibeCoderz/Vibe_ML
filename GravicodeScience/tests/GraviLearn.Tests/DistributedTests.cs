using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviLearn.Distributed;
using Gravicode.Science.GraviLearn.ModelSelection;
using Gravicode.Science.GraviLearn.Trees;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.GraviLearn;

/// <summary>
/// Tests for data-parallel training.
/// </summary>
/// <remarks>
/// <para>
/// Distribution is worth having only if it does not change the answer, so that is what these check.
/// For the forest the claim is the strong one — <b>bit-identical</b> to single-process, because
/// every tree is seeded from its global index — and for gradient averaging it is exactness against
/// the gradient a single worker would have computed over the whole batch.
/// </para>
/// <para>
/// Tests that merely showed a distributed run converging would pass with the shards weighted wrong,
/// which is the mistake this file exists to catch.
/// </para>
/// </remarks>
public class DistributedTests : IDisposable
{
    private readonly List<string> _temporary = [];

    public void Dispose()
    {
        foreach (var path in _temporary)
            if (Directory.Exists(path)) Directory.Delete(path, recursive: true);
    }

    private string TempDirectory()
    {
        var path = Path.Combine(Path.GetTempPath(), $"gravi_dist_{Guid.NewGuid():N}");
        _temporary.Add(path);
        return path;
    }

    // ---------------------------------------------------------------- partitioning

    [Fact]
    public void PartitioningCoversEveryItemExactlyOnce()
    {
        foreach (var (items, workers) in new[] { (100, 8), (7, 3), (5, 5), (0, 4), (1, 9) })
        {
            var shards = DataParallel.Partition(items, workers);

            Assert.Equal(workers, shards.Length);
            Assert.Equal(items, shards.Sum(s => s.Count));

            // Contiguous and non-overlapping.
            var next = 0;
            foreach (var shard in shards)
            {
                Assert.Equal(next, shard.Start);
                next = shard.End;
            }

            Assert.Equal(items, next);
        }
    }

    [Fact]
    public void TheRemainderIsSpreadRatherThanDumpedOnOneWorker()
    {
        // 100 items over 8 workers is 12 remainder 4. Giving the remainder to one worker makes it
        // the critical path: 13 against 16 is the difference.
        var shards = DataParallel.Partition(100, 8);

        var largest = shards.Max(s => s.Count);
        var smallest = shards.Min(s => s.Count);

        Assert.Equal(13, largest);
        Assert.Equal(12, smallest);
        Assert.True(largest - smallest <= 1, "the shards were not balanced to within one item");
    }

    // ---------------------------------------------------------------- gradients

    [Fact]
    public void WeightedAveragingReproducesTheGlobalGradientExactly()
    {
        // The correctness condition. Each worker averages over its own shard; weighting those means
        // by shard size must give the mean over everything.
        var rng = new GraviRandom(7);
        var samples = new NdArray[97];
        for (var i = 0; i < samples.Length; i++) samples[i] = rng.StandardNormal(3, 4);

        var global = NdArray.Zeros(3, 4);
        foreach (var sample in samples)
            for (var k = 0; k < global.Size; k++) global.SetAt(k, global.At(k) + sample.At(k) / samples.Length);

        var shards = DataParallel.Partition(samples.Length, 6);
        var partials = new List<NdArray>();
        var counts = new List<int>();

        foreach (var shard in shards)
        {
            var partial = NdArray.Zeros(3, 4);
            foreach (var i in shard.Indices)
                for (var k = 0; k < partial.Size; k++)
                    partial.SetAt(k, partial.At(k) + samples[i].At(k) / shard.Count);

            partials.Add(partial);
            counts.Add(shard.Count);
        }

        var combined = DataParallel.AverageGradients(partials, counts);

        for (var k = 0; k < global.Size; k++)
            Assert.Equal(global.At(k), combined.At(k), 12);
    }

    [Fact]
    public void UnweightedAveragingIsWrongWhenShardsDiffer()
    {
        // Pinning the trap rather than the fix. With uneven shards a plain average of per-worker
        // means over-weights the small shards, and nothing about the result looks wrong.
        var a = NdArray.FromValues([0.0, 0.0]);       // 90 samples, all zero
        var b = NdArray.FromValues([10.0, 10.0]);     // 10 samples, all ten

        var weighted = DataParallel.AverageGradients([a, b], [90, 10]);
        Assert.Equal(1.0, weighted.At(0), 12);        // (90*0 + 10*10) / 100

        // What an unweighted average would have given.
        var naive = (a.At(0) + b.At(0)) / 2;
        Assert.Equal(5.0, naive);
        Assert.NotEqual(naive, weighted.At(0), 6);
    }

    [Fact]
    public void AveragingIsIndependentOfHowTheWorkIsSplit()
    {
        var rng = new GraviRandom(11);
        var samples = new NdArray[120];
        for (var i = 0; i < samples.Length; i++) samples[i] = rng.StandardNormal(2, 3);

        NdArray Combine(int workers)
        {
            var shards = DataParallel.Partition(samples.Length, workers);
            var partials = new List<NdArray>();
            var counts = new List<int>();

            foreach (var shard in shards)
            {
                var partial = NdArray.Zeros(2, 3);
                foreach (var i in shard.Indices)
                    for (var k = 0; k < partial.Size; k++)
                        partial.SetAt(k, partial.At(k) + samples[i].At(k) / shard.Count);

                partials.Add(partial);
                counts.Add(shard.Count);
            }

            return DataParallel.AverageGradients(partials, counts);
        }

        var reference = Combine(1);
        foreach (var workers in new[] { 2, 3, 7, 16 })
        {
            var combined = Combine(workers);
            for (var k = 0; k < reference.Size; k++)
                Assert.Equal(reference.At(k), combined.At(k), 10);
        }
    }

    [Fact]
    public void MismatchedGradientShapesAreRejected()
    {
        Assert.Throws<ArgumentException>(() => DataParallel.AverageGradients(
            [NdArray.Zeros(2, 3), NdArray.Zeros(3, 2)], [1, 1]));

        Assert.Throws<ArgumentException>(() => DataParallel.AverageGradients(
            [NdArray.Zeros(2, 2)], [1, 1]));
    }

    // ---------------------------------------------------------------- transports

    [Fact]
    public void EncodingRoundTripsAnArrayExactly()
    {
        var rng = new GraviRandom(13);
        var original = rng.StandardNormal(4, 5);

        var back = DataParallel.Decode(DataParallel.Encode(original));

        Assert.Equal(original.Shape.ToArray(), back.Shape.ToArray());
        for (var i = 0; i < original.Size; i++)
            Assert.Equal(original.At(i), back.At(i));      // bit-exact, not approximate
    }

    [Fact]
    public void TheInProcessTransportReturnsResultsInWorkerOrder()
    {
        // Not arrival order: floating-point addition is not associative, so summing in arrival
        // order would make the result depend on scheduling and two runs would differ.
        using var transport = new InProcessTransport(4);

        transport.Publish(0, 2, [2]);
        transport.Publish(0, 0, [0]);
        transport.Publish(0, 3, [3]);
        transport.Publish(0, 1, [1]);

        var collected = transport.Collect(0);
        for (var w = 0; w < 4; w++) Assert.Equal((byte)w, collected[w][0]);
    }

    [Fact]
    public void CollectingBeforeEveryWorkerHasPublishedIsAnError()
    {
        using var transport = new InProcessTransport(3);
        transport.Publish(0, 0, [1]);

        Assert.Throws<InvalidOperationException>(() => transport.Collect(0));
    }

    [Fact]
    public void TheFileTransportCarriesPayloadsBetweenWorkers()
    {
        var directory = TempDirectory();
        using var transport = new FileTransport(directory, workerCount: 3, timeout: TimeSpan.FromSeconds(20));

        var rng = new GraviRandom(17);
        var payloads = Enumerable.Range(0, 3).Select(_ => rng.StandardNormal(2, 2)).ToArray();

        // Published from several threads, which is what a real deployment does.
        Parallel.For(0, 3, w => transport.Publish(0, w, DataParallel.Encode(payloads[w])));

        var collected = transport.Collect(0);
        Assert.Equal(3, collected.Count);

        for (var w = 0; w < 3; w++)
        {
            var back = DataParallel.Decode(collected[w]);
            for (var i = 0; i < back.Size; i++) Assert.Equal(payloads[w].At(i), back.At(i));
        }
    }

    [Fact]
    public void TheFileTransportTimesOutRatherThanHanging()
    {
        var directory = TempDirectory();
        using var transport = new FileTransport(directory, workerCount: 2, timeout: TimeSpan.FromMilliseconds(200));

        transport.Publish(0, 0, [1]);
        Assert.Throws<TimeoutException>(() => transport.Collect(0));
    }

    [Fact]
    public void ParameterServerAggregatesAcrossRounds()
    {
        using var transport = new InProcessTransport(2);
        var server = new ParameterServer(transport);

        for (var round = 0; round < 3; round++)
        {
            server.Contribute(0, NdArray.FromValues([1.0 * round, 2.0]), sampleCount: 30);
            server.Contribute(1, NdArray.FromValues([3.0 * round, 4.0]), sampleCount: 10);

            var averaged = server.Aggregate();

            // Weighted 3:1 towards worker 0.
            Assert.Equal((30 * 1.0 * round + 10 * 3.0 * round) / 40, averaged.At(0), 10);
            Assert.Equal((30 * 2.0 + 10 * 4.0) / 40, averaged.At(1), 10);
        }

        Assert.Equal(3, server.Round);
    }

    // ---------------------------------------------------------------- forest

    private static (NdArray X, NdArray Y) Data(int rows = 400, int seed = 3)
    {
        var rng = new GraviRandom(seed);
        var x = NdArray.Zeros(rows, 6);
        var y = NdArray.Zeros(rows);

        for (var i = 0; i < rows; i++)
        {
            for (var f = 0; f < 6; f++) x[i, f] = rng.Normal();
            y.SetAt(i, x[i, 0] + x[i, 1] > 0 ? 1 : 0);
        }

        return (x, y);
    }

    [Fact]
    public void TheDistributedForestIsIdenticalHoweverManyWorkersAreUsed()
    {
        // The strong claim, and the reason it holds: tree t is seeded from t alone, so a shard
        // boundary cannot change what any tree becomes.
        var (x, y) = Data();

        var reference = new DistributedForest(nTrees: 24, maxDepth: 5, seed: 42).Fit(x, y, workers: 1);
        var referencePredictions = reference.Predict(x);

        foreach (var workers in new[] { 2, 3, 5, 8, 24 })
        {
            var forest = new DistributedForest(nTrees: 24, maxDepth: 5, seed: 42).Fit(x, y, workers);

            Assert.Equal(24, forest.Trees.Count);

            var predictions = forest.Predict(x);
            for (var i = 0; i < x.Shape[0]; i++)
                Assert.Equal(referencePredictions.At(i), predictions.At(i));
        }
    }

    [Fact]
    public void TheProbabilitiesAreIdenticalToo()
    {
        // Equal predictions could survive small differences; equal probabilities could not.
        var (x, y) = Data(rows: 200);

        var one = new DistributedForest(nTrees: 16, maxDepth: 4, seed: 7).Fit(x, y, workers: 1);
        var many = new DistributedForest(nTrees: 16, maxDepth: 4, seed: 7).Fit(x, y, workers: 4);

        var a = one.PredictProbabilities(x);
        var b = many.PredictProbabilities(x);

        for (var i = 0; i < a.Size; i++) Assert.Equal(a.At(i), b.At(i), 12);
    }

    [Fact]
    public void MoreWorkersThanTreesLeavesSomeShardsEmpty()
    {
        // A degenerate but legal configuration, and the empty shards must not produce empty trees
        // or throw.
        var (x, y) = Data(rows: 120);
        var forest = new DistributedForest(nTrees: 5, maxDepth: 4, seed: 11).Fit(x, y, workers: 12);

        Assert.Equal(5, forest.Trees.Count);
        Assert.True(forest.Score(x, y) > 0.7);
    }

    [Fact]
    public void ItReachesAccuracyComparableToTheOrdinaryForest()
    {
        // Not identical: RandomForestClassifier computes an out-of-bag score and this does not, so
        // they are different classes doing the same thing. The accuracies should still match.
        var (x, y) = Data(rows: 500, seed: 19);
        var split = Selection.Split(x, y, testSize: 0.3, seed: 5);

        var distributed = new DistributedForest(nTrees: 40, maxDepth: 6, seed: 42)
            .Fit(split.TrainX, split.TrainY, workers: 4);

        var ordinary = new RandomForestClassifier(nTrees: 40, maxDepth: 6, seed: 42);
        ordinary.Fit(split.TrainX, split.TrainY);

        var distributedScore = distributed.Score(split.TestX, split.TestY);
        var ordinaryScore = ordinary.Score(split.TestX, split.TestY);

        Assert.True(Math.Abs(distributedScore - ordinaryScore) < 0.05,
            $"distributed {distributedScore:P1} against ordinary {ordinaryScore:P1}");
    }

    [Fact]
    public void EveryWorkerAgreesOnTheClassColumnOrder()
    {
        // A worker whose shard missed a rare class would otherwise build trees with a different
        // number of output columns, and the ensemble would average misaligned probabilities.
        var (x, y) = Data(rows: 300);
        var classes = new[] { 0.0, 1.0, 2.0 };      // deliberately wider than the data

        var trees = DistributedForest.GrowShard(x, y, new Shard(0, 4), classes, maxDepth: 3);

        foreach (var tree in trees)
            Assert.Equal(3, tree.PredictProbabilities(x).Shape[1]);
    }

    [Fact]
    public void MalformedRequestsAreRejected()
    {
        var (x, y) = Data(rows: 50);

        Assert.Throws<ArgumentOutOfRangeException>(() => new DistributedForest().Fit(x, y, workers: 0));
        Assert.Throws<ArgumentException>(() => new DistributedForest().Fit(x, NdArray.Zeros(3), workers: 2));
        Assert.Throws<InvalidOperationException>(() => new DistributedForest().Predict(x));
    }
}
