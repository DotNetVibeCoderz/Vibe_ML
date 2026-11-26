using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.DaskNet.ML;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviLearn.Neighbors;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet.ML;

/// <summary>Incremental training is pinned against a plain loop of PartialFit over the same
/// partitions in the same order, and naive Bayes against the batch model.</summary>
public sealed class IncrementalTests
{
    private static DataFrame Table(NdArray x, NdArray y)
    {
        var names = Enumerable.Range(0, x.Shape[1]).Select(j => $"x{j}").ToArray();
        return DataFrame.FromMatrix(x, names).WithColumn(new NumericSeries("y", y.ToArray()));
    }

    private static string[] Names(int d) => Enumerable.Range(0, d).Select(j => $"x{j}").ToArray();

    /// <summary>The reference: the partitions of <paramref name="frame"/> fed to PartialFit in
    /// the order the chain visits them.</summary>
    private static void LoopFit(IIncrementalModel model, DaskFrame frame, string[] features, int epochs, int seed)
    {
        var parts = frame.ToDelayed().Select(p => p.Compute()).ToArray();
        var random = new Random(seed);
        for (var e = 0; e < epochs; e++)
        {
            var order = Enumerable.Range(0, parts.Length).ToArray();
            random.Shuffle(order);
            foreach (var p in order)
                model.PartialFit(MLFrame.Features(parts[p], features), new NdArray(parts[p].Numeric("y").Values.ToArray(), parts[p].RowCount));
        }
    }

    [Fact]
    public async Task Sgd_regression_over_partitions_equals_the_loop_and_finds_the_coefficients()
    {
        var (x, y, truth) = Datasets.MakeRegression(3_000, features: 4, noise: 0.1, seed: 5);
        var frame = DaskFrame.FromFrame(Table(x, y), 6);
        var features = Names(4);

        var reference = new SgdRegressor(learningRate: 0.02);
        LoopFit(reference, frame, features, epochs: 4, seed: 9);
        var local = Incremental.Fit(new SgdRegressor(learningRate: 0.02), frame, features, "y", epochs: 4, seed: 9);
        Assert.Equal(reference.Coefficients, local.Coefficients);              // bit for bit
        Assert.Equal(reference.Intercept, local.Intercept);
        Assert.Equal(4 * 3_000, local.Steps);
        for (var j = 0; j < 4; j++)
            Assert.True(Math.Abs(local.Coefficients[j] - truth.At(j)) < 0.05 * Math.Max(1, Math.Abs(truth.At(j))),
                $"coefficient {j}: {local.Coefficients[j]} vs {truth.At(j)}");

        await using var cluster = await LocalCluster.StartAsync(workers: 3, threadsPerWorker: 2);
        var remote = Incremental.Fit(new SgdRegressor(learningRate: 0.02), frame, features, "y", epochs: 4, seed: 9, scheduler: cluster.Client);
        Assert.Equal(reference.Coefficients, remote.Coefficients);
        Assert.Equal(reference.Intercept, remote.Intercept);

        // Scoring on the cluster ships the state, not a fitted object.
        var scored = frame.Predict(remote, features).Compute(cluster.Client);
        Assert.Equal(remote.Predict(x).ToArray(), scored.Numeric("prediction").Values.ToArray());

        // Training continues from a fitted model's state.
        var more = Incremental.Fit(local, frame, features, "y", epochs: 1, seed: 1);
        Assert.Equal(5 * 3_000, more.Steps);
    }

    [Fact]
    public async Task Sgd_classification_equals_the_loop_and_separates_blobs()
    {
        var data = Datasets.MakeBlobs(1_500, features: 3, centers: 3, spread: 1.0, seed: 2);
        var frame = DaskFrame.FromFrame(Table(data.Features, data.Target), 5);
        var features = Names(3);
        double[] classes = [0, 1, 2];

        var reference = new SgdClassifier(classes, learningRate: 0.05);
        LoopFit(reference, frame, features, epochs: 3, seed: 4);
        await using var cluster = await LocalCluster.StartAsync(workers: 2, threadsPerWorker: 2);
        var remote = Incremental.Fit(new SgdClassifier(classes, learningRate: 0.05), frame, features, "y", epochs: 3, seed: 4, scheduler: cluster.Client);
        Assert.Equal(reference.PredictProbabilities(data.Features).ToArray(), remote.PredictProbabilities(data.Features).ToArray());
        Assert.True(Metrics.Accuracy(data.Target, remote.Predict(data.Features)) > 0.95);

        // Classes are fixed up front: a label outside them is refused, not silently added.
        Assert.Throws<ArgumentException>(() => remote.PartialFit(NdArray.Zeros(1, 3), new NdArray([7.0], 1)));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(9)]
    public void Naive_bayes_learned_in_batches_equals_the_batch_model(int partitions)
    {
        var data = Datasets.LoadIris();
        var frame = DaskFrame.FromFrame(Table(data.Features, data.Target), partitions);
        var features = Names(4);
        var batch = new GaussianNaiveBayes();
        batch.Fit(data.Features, data.Target);

        var incremental = Incremental.Fit(new IncrementalGaussianNaiveBayes([0, 1, 2]), frame, features, "y", seed: partitions);
        Assert.Equal(batch.Predict(data.Features).ToArray(), incremental.Predict(data.Features).ToArray());
        var p = batch.PredictProbabilities(data.Features).ToArray();
        var q = incremental.PredictProbabilities(data.Features).ToArray();
        for (var i = 0; i < p.Length; i++)
            Assert.True(Math.Abs(p[i] - q[i]) <= 1e-9 * Math.Max(1e-300, Math.Abs(p[i])) + 1e-15, $"probability {i}: {p[i]} vs {q[i]}");
    }

    [Fact]
    public void A_class_missing_from_the_data_has_zero_probability()
    {
        var data = Datasets.LoadIris();
        var rows = Enumerable.Range(0, data.Features.Shape[0]).Where(i => data.Target.At(i) != 2).ToArray();
        var model = new IncrementalGaussianNaiveBayes([0, 1, 2]);
        model.PartialFit(data.Features.Take(rows), data.Target.Take(rows));
        var p = model.PredictProbabilities(data.Features);
        Assert.All(Enumerable.Range(0, data.Features.Shape[0]), i => Assert.Equal(0.0, p[i, 2]));
        Assert.DoesNotContain(2.0, model.Predict(data.Features).ToArray());
    }
}
