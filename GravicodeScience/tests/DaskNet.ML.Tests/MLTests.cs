using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.DaskNet.ML;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviLearn.Linear;
using Gravicode.Science.GraviLearn.Neighbors;
using Gravicode.Science.GraviLearn.Trees;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet.ML;

/// <summary>Every distributed score is pinned against a plain loop over the same folds.</summary>
public sealed class MLTests
{
    private static double LoopScore(Func<IEstimator> make, NdArray x, NdArray y, int[] foldOf, int fold, Func<NdArray, NdArray, double> metric)
    {
        var train = Enumerable.Range(0, foldOf.Length).Where(i => foldOf[i] != fold).ToArray();
        var test = Enumerable.Range(0, foldOf.Length).Where(i => foldOf[i] == fold).ToArray();
        var model = make();
        model.Fit(x.Take(train), y.Take(train));
        return metric(y.Take(test), model.Predict(x.Take(test)));
    }

    [Fact]
    public void Folds_are_balanced_seeded_and_cover_every_sample()
    {
        var f = HyperparameterSearch.Folds(103, 5, seed: 7);
        Assert.All(Enumerable.Range(0, 5), k => Assert.InRange(f.Count(v => v == k), 20, 21));
        Assert.Equal(f, HyperparameterSearch.Folds(103, 5, seed: 7));
        Assert.NotEqual(f, HyperparameterSearch.Folds(103, 5, seed: 8));
        Assert.Equal(Enumerable.Range(0, 10).Select(i => i % 5), HyperparameterSearch.Folds(10, 5, shuffle: false));
        Assert.Throws<ArgumentException>(() => HyperparameterSearch.Folds(3, 5));
    }

    [Fact]
    public async Task Grid_search_scores_equal_a_plain_loop_locally_and_on_a_cluster()
    {
        var data = Datasets.MakeMoons(240, noise: 0.3, seed: 3);
        var grid = new Dictionary<string, object?[]>
        {
            ["k"] = [1, 5, 15],
            ["metric"] = [DistanceMetric.Euclidean, DistanceMetric.Manhattan],   // an enum crosses the wire by name
        };
        var foldOf = HyperparameterSearch.Folds(240, 4, seed: 11);

        var local = HyperparameterSearch.Grid<KNearestNeighborsClassifier>(grid, data.Features, data.Target, folds: 4, seed: 11);
        await using var cluster = await LocalCluster.StartAsync(workers: 3, threadsPerWorker: 2);
        var remote = HyperparameterSearch.Grid<KNearestNeighborsClassifier>(grid, data.Features, data.Target, folds: 4, seed: 11, scheduler: cluster.Client);

        Assert.Equal(6, local.Candidates.Count);
        Assert.Equal("accuracy", local.Scoring);
        foreach (var c in local.Candidates)
        {
            var k = (int)c.Spec.Parameters["k"]!;
            var metric = (DistanceMetric)c.Spec.Parameters["metric"]!;
            var expected = Enumerable.Range(0, 4).Select(f => LoopScore(() => new KNearestNeighborsClassifier(k, metric), data.Features, data.Target, foldOf, f, Metrics.Accuracy));
            Assert.Equal(expected, c.FoldScores);
            Assert.Equal(c.FoldScores, remote.Candidates.Single(r => r.Spec.ToString() == c.Spec.ToString()).FoldScores);
        }
        Assert.Equal(local.Candidates.Select(c => c.Spec.ToString()), remote.Candidates.Select(c => c.Spec.ToString()));
        Assert.True(local.Candidates.Zip(local.Candidates.Skip(1)).All(p => p.First.Mean >= p.Second.Mean), "ranked best first");
        Assert.NotEqual(1, (int)local.Best.Spec.Parameters["k"]!);       // 1-NN overfits noisy moons
        Assert.True(cluster.Client.LastRun!.TasksPerWorker.Count >= 2, "the fits should spread over workers");

        Assert.NotNull(local.BestEstimator);
        Assert.True(local.BestEstimator!.IsFitted);
        Assert.True(Metrics.Accuracy(data.Target, local.BestEstimator.Predict(data.Features)) > 0.8);

        var table = local.ToFrame();
        Assert.Equal(["param_k", "param_metric", "mean_score", "std_score", "rank"], table.ColumnNames);
        Assert.Equal(local.Best.Mean, table.Numeric("mean_score")[0]);
        Assert.Equal("Euclidean", table.Text("param_metric").Values.ToArray().Distinct().Order().First());
    }

    [Fact]
    public void Regression_search_negates_errors_and_matches_the_loop()
    {
        var (x, y, _) = Datasets.MakeRegression(150, features: 6, noise: 2.0, seed: 5);
        var result = HyperparameterSearch.Grid<RidgeRegression>(new Dictionary<string, object?[]> { ["alpha"] = [0.01, 1.0, 100.0, 10_000.0] },
            x, y, folds: 5, scoring: "neg_mean_squared_error", refit: false);
        Assert.Null(result.BestEstimator);
        Assert.All(result.Candidates, c => Assert.All(c.FoldScores, s => Assert.True(s <= 0)));
        Assert.Equal(10_000.0, result.Candidates[^1].Spec.Parameters["alpha"]);       // heavy shrinkage is worst
        var foldOf = HyperparameterSearch.Folds(150, 5);
        var best = (double)result.Best.Spec.Parameters["alpha"]!;
        Assert.Equal(Enumerable.Range(0, 5).Select(f => -LoopScore(() => new RidgeRegression(best), x, y, foldOf, f, Metrics.MeanSquaredError)),
            result.Best.FoldScores);

        var r2 = HyperparameterSearch.Grid<RidgeRegression>(new Dictionary<string, object?[]> { ["alpha"] = [1.0] }, x, y, folds: 3);
        Assert.Equal("r2", r2.Scoring);
        Assert.True(r2.Best.Mean > 0.5, $"r2 {r2.Best.Mean}");
    }

    [Fact]
    public void Random_search_draws_distinct_candidates_and_caps_at_the_grid()
    {
        var data = Datasets.MakeMoons(120, seed: 2);
        var space = new Dictionary<string, object?[]> { ["maxDepth"] = [1, 2, 3, 4, 6, 8], ["minSamplesLeaf"] = [1, 2, 5, 10] };
        var some = HyperparameterSearch.Random<DecisionTree>(space, 7, data.Features, data.Target, folds: 3, seed: 4);
        Assert.Equal(7, some.Candidates.Count);
        Assert.Equal(7, some.Candidates.Select(c => c.Spec.ToString()).Distinct().Count());
        Assert.Equal(some.Candidates.Select(c => c.Spec.ToString()),
            HyperparameterSearch.Random<DecisionTree>(space, 7, data.Features, data.Target, folds: 3, seed: 4).Candidates.Select(c => c.Spec.ToString()));
        var all = HyperparameterSearch.Random<DecisionTree>(space, 500, data.Features, data.Target, folds: 3, refit: false);
        Assert.Equal(24, all.Candidates.Count);
    }

    [Fact]
    public async Task Bad_specs_fail_before_anything_runs()
    {
        var data = Datasets.MakeMoons(60);
        var e = Assert.Throws<ArgumentException>(() => HyperparameterSearch.Grid<KNearestNeighborsClassifier>(
            new Dictionary<string, object?[]> { ["neighbours"] = [3] }, data.Features, data.Target));
        Assert.Contains("neighbours", e.Message);
        Assert.Throws<ArgumentException>(() => EstimatorSpec.Of<KNearestNeighborsClassifier>(("k", "three")));
        Assert.Throws<ArgumentException>(() => EstimatorSpec.Of<KNearestNeighborsClassifier>(("metric", "Hamming")));
        Assert.Throws<ArgumentException>(() => new EstimatorSpec(typeof(string)));
        Assert.Throws<ArgumentException>(() => HyperparameterSearch.Grid<RidgeRegression>(
            new Dictionary<string, object?[]> { ["alpha"] = [1.0] }, data.Features, data.Target, scoring: "auc"));
        Assert.Equal("KNearestNeighborsClassifier(k=3, metric=Cosine)", EstimatorSpec.Of<KNearestNeighborsClassifier>(("k", 3), ("metric", "cosine")).ToString());

        // A captured, fitted model is local-only and a cluster says so.
        await using var cluster = await LocalCluster.StartAsync(workers: 1, threadsPerWorker: 1);
        var model = new LogisticRegression();
        model.Fit(data.Features, data.Target);
        var frame = DaskFrame.FromFrame(DataFrame.FromMatrix(data.Features, ["a", "b"]), 2);
        Assert.Throws<NotSupportedException>(() => frame.Predict(model, ["a", "b"]).Compute(cluster.Client));
    }

    [Fact]
    public async Task Batch_prediction_on_a_cluster_equals_the_locally_fitted_model()
    {
        var train = Datasets.MakeMoons(200, noise: 0.25, seed: 8);
        var score = Datasets.MakeMoons(3000, noise: 0.25, seed: 9);
        var table = DataFrame.FromMatrix(score.Features, ["x1", "x2"]).WithColumn(new NumericSeries("id", Enumerable.Range(0, 3000).Select(i => (double)i).ToArray()));
        var frame = DaskFrame.FromFrame(table, 6);
        var spec = EstimatorSpec.Of<RandomForestClassifier>(("nTrees", 15), ("maxDepth", 6), ("seed", 3));

        var reference = spec.Create();
        reference.Fit(train.Features, train.Target);
        var expected = reference.Predict(score.Features).ToArray();
        var expectedP = ((IClassifier)reference).PredictProbabilities(score.Features);

        var local = frame.Predict(reference, ["x1", "x2"], probabilities: true).Compute();
        Assert.Equal(["x1", "x2", "id", "prediction", "prediction_p0", "prediction_p1"], local.ColumnNames);
        Assert.Equal(expected, local.Numeric("prediction").Values.ToArray());

        await using var cluster = await LocalCluster.StartAsync(workers: 2, threadsPerWorker: 2);
        var remote = frame.Predict(spec, train.Features, train.Target, ["x1", "x2"], output: "label", probabilities: true).Compute(cluster.Client);
        Assert.Equal(["x1", "x2", "id", "label", "label_p0", "label_p1"], remote.ColumnNames);
        Assert.Equal(expected, remote.Numeric("label").Values.ToArray());
        Assert.Equal(Enumerable.Range(0, 3000).Select(i => expectedP[i, 1]), remote.Numeric("label_p1").Values.ToArray());
        Assert.InRange(MLKernels.CachedModels, 1, 8);              // fitted once per process, not per partition

        Assert.Throws<ArgumentException>(() => frame.Predict(spec, train.Features, train.Target, ["x3"]));
        Assert.Throws<ArgumentException>(() => frame.Predict(EstimatorSpec.Of<RidgeRegression>(), train.Features, train.Target, ["x1"], probabilities: true));
    }
}
