using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviLearn.Clustering;
using Gravicode.Science.GraviLearn.Decomposition;
using Gravicode.Science.GraviLearn.Linear;
using Gravicode.Science.GraviLearn.ModelSelection;
using Gravicode.Science.GraviLearn.Preprocessing;
using Gravicode.Science.GraviLearn.Trees;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.GraviLearn;

public class DecompositionTests
{
    [Fact]
    public void Pca_CapturesMostVarianceInTheFirstComponent()
    {
        // Data stretched along one direction: one component should explain nearly everything.
        var rng = new GraviRandom(53);
        var x = NdArray.Zeros(300, 2);
        for (var i = 0; i < 300; i++)
        {
            var t = rng.Normal(0, 5);
            x[i, 0] = t + rng.Normal(0, 0.05);
            x[i, 1] = t * 0.5 + rng.Normal(0, 0.05);
        }

        var pca = new PrincipalComponentAnalysis(2);
        pca.Fit(x);

        Assert.True(pca.ExplainedVarianceRatio.At(0) > 0.98);
        Assert.True(pca.ExplainedVarianceRatio.At(0) > pca.ExplainedVarianceRatio.At(1));
    }

    [Fact]
    public void Pca_ReconstructionIsExactWhenNoComponentsAreDropped()
    {
        var rng = new GraviRandom(55);
        var x = rng.StandardNormal(80, 4);

        var pca = new PrincipalComponentAnalysis(4);
        var projected = pca.FitTransform(x);
        var restored = pca.InverseTransform(projected);

        Assert.True(UFunc.AllClose(x, restored, 1e-8));
    }

    [Fact]
    public void Pca_ProjectedComponentsAreUncorrelated()
    {
        var rng = new GraviRandom(57);
        var x = rng.StandardNormal(400, 5);
        var projected = new PrincipalComponentAnalysis(3).FitTransform(x);

        var correlation = Statistics.CorrelationMatrix(projected);
        Assert.Equal(0.0, correlation[0, 1], 6);
        Assert.Equal(0.0, correlation[0, 2], 6);
    }

    [Fact]
    public void Pca_RejectsMoreComponentsThanTheDataSupports()
    {
        var rng = new GraviRandom(59);
        var x = rng.StandardNormal(10, 3);
        Assert.Throws<ArgumentException>(() => new PrincipalComponentAnalysis(5).Fit(x));
    }

    [Fact]
    public void PcaAlias_BehavesIdentically()
    {
        var rng = new GraviRandom(61);
        var x = rng.StandardNormal(50, 4);
        Assert.True(UFunc.AllClose(
            new PCA(2).FitTransform(x),
            new PrincipalComponentAnalysis(2).FitTransform(x),
            1e-12));
    }

    [Fact]
    public void Lda_SeparatesClassesAlongItsFirstDirection()
    {
        var data = Datasets.MakeBlobs(samples: 200, features: 4, centers: 3, spread: 1.0, seed: 63);
        var lda = new LinearDiscriminantAnalysis();
        var projected = lda.FitTransform(data.Features, data.Target);

        Assert.Equal(2, projected.Shape[1]);   // three classes give two directions

        // A trivial classifier on the projection should do well if the separation is real.
        var model = new GaussianNaiveBayesProbe();
        Assert.True(model.SeparationScore(projected, data.Target) > 0.9);
    }

    [Fact]
    public void TSne_ProducesAnEmbeddingThatKeepsClustersApart()
    {
        var data = Datasets.MakeBlobs(samples: 60, features: 5, centers: 3, spread: 0.6, seed: 65);
        var tsne = new TStochasticNeighborEmbedding(components: 2, perplexity: 8, iterations: 250, seed: 65);
        var embedding = tsne.FitTransform(data.Features);

        Assert.Equal(60, embedding.Shape[0]);
        Assert.Equal(2, embedding.Shape[1]);

        // Same-cluster pairs should end up closer than different-cluster pairs on average.
        double same = 0, different = 0;
        int sameCount = 0, differentCount = 0;
        for (var i = 0; i < 60; i++)
            for (var j = i + 1; j < 60; j++)
            {
                var dx = embedding[i, 0] - embedding[j, 0];
                var dy = embedding[i, 1] - embedding[j, 1];
                var d = Math.Sqrt(dx * dx + dy * dy);
                if (data.Target.At(i) == data.Target.At(j)) { same += d; sameCount++; }
                else { different += d; differentCount++; }
            }

        Assert.True(same / sameCount < different / differentCount);
    }
}

/// <summary>A minimal helper that scores how separable a projection is, used by the LDA test.</summary>
internal sealed class GaussianNaiveBayesProbe
{
    public double SeparationScore(NdArray x, NdArray y)
    {
        var model = new Gravicode.Science.GraviLearn.Neighbors.GaussianNaiveBayes();
        model.Fit(x, y);
        return Metrics.Accuracy(y, model.Predict(x));
    }
}

public class ClusteringTests
{
    [Fact]
    public void KMeans_RecoversWellSeparatedBlobs()
    {
        var data = Datasets.MakeBlobs(samples: 300, centers: 3, spread: 0.6, seed: 67);
        var model = new KMeans(clusters: 3, seed: 67);
        var labels = model.FitPredict(data.Features);

        // Cluster ids are arbitrary, so score by agreement rather than by label equality.
        Assert.Equal(3, model.Centroids.Shape[0]);
        Assert.True(ClusterAgreement(data.Target, labels) > 0.95);
        Assert.True(model.Inertia > 0);
    }

    [Fact]
    public void KMeans_InertiaFallsAsClustersAreAdded()
    {
        var data = Datasets.MakeBlobs(samples: 200, centers: 4, spread: 1.0, seed: 69);
        var curve = KMeans.ElbowCurve(data.Features, maxK: 5, seed: 69);

        for (var i = 1; i < curve.Count; i++)
            Assert.True(curve[i].Inertia <= curve[i - 1].Inertia + 1e-6);
    }

    [Fact]
    public void KMeans_RejectsMoreClustersThanSamples()
    {
        var data = Datasets.MakeBlobs(samples: 5, centers: 2, seed: 71);
        Assert.Throws<ArgumentException>(() => new KMeans(clusters: 10).Fit(data.Features));
    }

    [Fact]
    public void Dbscan_FindsClustersAndLabelsOutliersAsNoise()
    {
        var rng = new GraviRandom(73);
        var points = new List<double[]>();

        // Two tight clusters plus three far-away outliers.
        for (var i = 0; i < 60; i++) points.Add([rng.Normal(0, 0.2), rng.Normal(0, 0.2)]);
        for (var i = 0; i < 60; i++) points.Add([rng.Normal(6, 0.2), rng.Normal(6, 0.2)]);
        points.Add([20.0, 20.0]);
        points.Add([-15.0, 18.0]);
        points.Add([25.0, -22.0]);

        var x = NdArray.FromRows(points);
        var model = new Dbscan(epsilon: 0.7, minSamples: 5);
        model.Fit(x);

        Assert.Equal(2, model.ClusterCount);
        Assert.Equal(3, model.NoiseCount);
    }

    [Fact]
    public void Dbscan_RefusesOutOfSamplePrediction()
    {
        var data = Datasets.MakeBlobs(samples: 50, centers: 2, seed: 75);
        var model = new Dbscan(epsilon: 2.0, minSamples: 3);
        model.Fit(data.Features);
        Assert.Throws<NotSupportedException>(() => model.Predict(data.Features));
    }

    [Fact]
    public void AgglomerativeClustering_RecoversTwoObviousGroups()
    {
        var points = new List<double[]>();
        for (var i = 0; i < 20; i++) points.Add([i * 0.01, i * 0.01]);
        for (var i = 0; i < 20; i++) points.Add([10 + i * 0.01, 10 + i * 0.01]);

        var model = new AgglomerativeClustering(clusters: 2);
        var labels = model.FitPredict(NdArray.FromRows(points));

        var first = labels.At(0);
        for (var i = 0; i < 20; i++) Assert.Equal(first, labels.At(i));
        for (var i = 20; i < 40; i++) Assert.NotEqual(first, labels.At(i));
    }

    [Fact]
    public void GaussianMixture_ConvergesAndRecoversComponents()
    {
        var data = Datasets.MakeBlobs(samples: 300, centers: 3, spread: 0.7, seed: 77);
        var model = new GaussianMixture(components: 3, seed: 77);
        var labels = model.FitPredict(data.Features);

        Assert.True(model.Converged);
        Assert.Equal(1.0, model.Weights.Sum(), 8);
        Assert.True(ClusterAgreement(data.Target, labels) > 0.9);
    }

    [Fact]
    public void GaussianMixture_BicPrefersTheTrueComponentCount()
    {
        var data = Datasets.MakeBlobs(samples: 300, features: 2, centers: 3, spread: 0.5, seed: 79);

        var scores = new List<(int Components, double Bic)>();
        for (var k = 1; k <= 5; k++)
        {
            var model = new GaussianMixture(components: k, seed: 79);
            model.Fit(data.Features);
            scores.Add((k, model.Bic(data.Features)));
        }

        var best = scores.MinBy(s => s.Bic).Components;
        Assert.True(best >= 3, $"BIC picked {best} components");
    }

    [Fact]
    public void SilhouetteScore_RewardsSeparationAndPenalisesOverlap()
    {
        // Explicit centres, so the test measures the metric rather than where MakeBlobs
        // happened to place its random centroids.
        var rng = new GraviRandom(81);
        var tight = new List<double[]>();
        var overlapping = new List<double[]>();
        double[][] centres = [[0, 0], [30, 0], [15, 30]];

        foreach (var centre in centres)
            for (var i = 0; i < 50; i++)
            {
                tight.Add([centre[0] + rng.Normal(0, 0.5), centre[1] + rng.Normal(0, 0.5)]);
                overlapping.Add([centre[0] + rng.Normal(0, 14.0), centre[1] + rng.Normal(0, 14.0)]);
            }

        var tightX = NdArray.FromRows(tight);
        var overlappingX = NdArray.FromRows(overlapping);

        var tightScore = Metrics.SilhouetteScore(tightX, new KMeans(3, seed: 81).FitPredict(tightX));
        var overlappingScore = Metrics.SilhouetteScore(overlappingX, new KMeans(3, seed: 81).FitPredict(overlappingX));

        Assert.True(tightScore > 0.9, $"tight silhouette = {tightScore:F4}");
        Assert.True(overlappingScore < tightScore, $"overlapping {overlappingScore:F4} vs tight {tightScore:F4}");
    }

    /// <summary>
    /// Cluster ids are arbitrary, so agreement is the best accuracy achievable over any
    /// one-to-one relabelling of the predicted clusters onto the true classes.
    /// </summary>
    private static double ClusterAgreement(NdArray truth, NdArray predicted)
    {
        var trueLabels = truth.ToArray().Distinct().OrderBy(v => v).ToArray();
        var predictedLabels = predicted.ToArray().Distinct().OrderBy(v => v).ToArray();

        var best = 0.0;
        foreach (var permutation in Permutations(trueLabels))
        {
            var mapping = new Dictionary<double, double>();
            for (var i = 0; i < predictedLabels.Length && i < permutation.Length; i++)
                mapping[predictedLabels[i]] = permutation[i];

            var correct = 0;
            for (var i = 0; i < truth.Size; i++)
                if (mapping.TryGetValue(predicted.At(i), out var mapped) && mapped == truth.At(i)) correct++;

            best = Math.Max(best, (double)correct / truth.Size);
        }
        return best;
    }

    private static IEnumerable<double[]> Permutations(double[] values)
    {
        if (values.Length <= 1) { yield return values; yield break; }
        for (var i = 0; i < values.Length; i++)
        {
            var rest = values.Where((_, k) => k != i).ToArray();
            foreach (var tail in Permutations(rest))
                yield return new[] { values[i] }.Concat(tail).ToArray();
        }
    }
}

public class MetricsTests
{
    [Fact]
    public void Accuracy_And_ConfusionMatrix_AgreeWithHandCounts()
    {
        var truth = NdArray.FromValues([0, 0, 1, 1, 1, 2]);
        var predicted = NdArray.FromValues([0, 1, 1, 1, 2, 2]);

        Assert.Equal(4.0 / 6.0, Metrics.Accuracy(truth, predicted), 9);

        var matrix = Metrics.ConfusionMatrix(truth, predicted);
        Assert.Equal(3, matrix.Shape[0]);
        Assert.Equal(1.0, matrix[0, 0]);
        Assert.Equal(1.0, matrix[0, 1]);
        Assert.Equal(2.0, matrix[1, 1]);
        Assert.Equal(1.0, matrix[2, 2]);
    }

    [Fact]
    public void PrecisionRecallF1_MatchTheirDefinitions()
    {
        var truth = NdArray.FromValues([1, 1, 1, 0, 0]);
        var predicted = NdArray.FromValues([1, 1, 0, 1, 0]);

        // TP = 2, FP = 1, FN = 1
        Assert.Equal(2.0 / 3.0, Metrics.Precision(truth, predicted), 9);
        Assert.Equal(2.0 / 3.0, Metrics.Recall(truth, predicted), 9);
        Assert.Equal(2.0 / 3.0, Metrics.F1Score(truth, predicted), 9);
    }

    [Fact]
    public void RocAuc_IsOneForAPerfectRankingAndAHalfForRandom()
    {
        var truth = NdArray.FromValues([0, 0, 1, 1]);
        Assert.Equal(1.0, Metrics.RocAucScore(truth, NdArray.FromValues([0.1, 0.2, 0.8, 0.9])), 9);
        Assert.Equal(0.0, Metrics.RocAucScore(truth, NdArray.FromValues([0.9, 0.8, 0.2, 0.1])), 9);
        Assert.Equal(0.5, Metrics.RocAucScore(truth, NdArray.FromValues([0.5, 0.5, 0.5, 0.5])), 9);
    }

    [Fact]
    public void ClassificationReport_ContainsEveryClassAndTheAverages()
    {
        var truth = NdArray.FromValues([0, 0, 1, 1, 2, 2]);
        var predicted = NdArray.FromValues([0, 1, 1, 1, 2, 2]);
        var report = Metrics.ClassificationReport(truth, predicted);

        Assert.Contains("precision", report);
        Assert.Contains("accuracy", report);
        Assert.Contains("macro avg", report);
        Assert.Contains("weighted avg", report);
    }

    [Fact]
    public void RegressionMetrics_MatchHandComputedValues()
    {
        var truth = NdArray.FromValues([1.0, 2.0, 3.0]);
        var predicted = NdArray.FromValues([1.0, 2.0, 4.0]);

        Assert.Equal(1.0 / 3.0, Metrics.MeanSquaredError(truth, predicted), 9);
        Assert.Equal(1.0 / 3.0, Metrics.MeanAbsoluteError(truth, predicted), 9);
        Assert.Equal(1.0 - (1.0 / 3.0) / (2.0 / 3.0), Metrics.R2Score(truth, predicted), 9);
    }

    [Fact]
    public void R2_IsOneForAPerfectFitAndZeroForThePlainMean()
    {
        var truth = NdArray.FromValues([1.0, 2.0, 3.0, 4.0]);
        Assert.Equal(1.0, Metrics.R2Score(truth, truth), 9);
        Assert.Equal(0.0, Metrics.R2Score(truth, NdArray.Full(2.5, 4)), 9);
    }

    [Fact]
    public void LogLoss_FallsAsPredictionsImprove()
    {
        var truth = NdArray.FromValues([0, 1]);
        var confident = NdArray.FromArray(new double[,] { { 0.95, 0.05 }, { 0.05, 0.95 } });
        var unsure = NdArray.FromArray(new double[,] { { 0.55, 0.45 }, { 0.45, 0.55 } });

        Assert.True(Metrics.LogLoss(truth, confident) < Metrics.LogLoss(truth, unsure));
    }
}

public class WorkflowTests
{
    [Fact]
    public void TrainTestSplit_PartitionsWithoutOverlap()
    {
        var data = Datasets.MakeBlobs(samples: 200, centers: 2, seed: 83);
        var split = Selection.Split(data.Features, data.Target, testSize: 0.25, seed: 83);

        Assert.Equal(150, split.TrainX.Shape[0]);
        Assert.Equal(50, split.TestX.Shape[0]);
        Assert.Equal(150, split.TrainY.Size);
        Assert.Equal(50, split.TestY.Size);
    }

    [Fact]
    public void StratifiedSplit_PreservesClassProportions()
    {
        // A deliberately imbalanced problem: 90% class 0, 10% class 1.
        var rng = new GraviRandom(85);
        var x = rng.StandardNormal(200, 2);
        var y = NdArray.Zeros(200);
        for (var i = 0; i < 20; i++) y.SetAt(i, 1);

        var split = Selection.Split(x, y, testSize: 0.25, seed: 85, stratify: true);
        var testPositives = split.TestY.ToArray().Count(v => v == 1);
        var trainPositives = split.TrainY.ToArray().Count(v => v == 1);

        Assert.Equal(5, testPositives);
        Assert.Equal(15, trainPositives);
    }

    [Fact]
    public void KFold_CoversEverySampleExactlyOnceAcrossTestFolds()
    {
        var folds = Selection.KFold(103, folds: 5, seed: 87);
        var seen = folds.SelectMany(f => f.Test).ToArray();

        Assert.Equal(103, seen.Length);
        Assert.Equal(103, seen.Distinct().Count());
        foreach (var (train, test) in folds) Assert.Empty(train.Intersect(test));
    }

    [Fact]
    public void CrossValidate_ScoresEveryFold()
    {
        var data = Datasets.MakeBlobs(samples: 200, centers: 3, spread: 1.0, seed: 89);
        var result = Selection.CrossValidate(
            () => new DecisionTree(maxDepth: 4), data.Features, data.Target, folds: 5, stratified: true, seed: 89);

        Assert.Equal(5, result.Scores.Length);
        Assert.True(result.Mean > 0.85);
        Assert.True(result.StandardDeviation < 0.25);
    }

    [Fact]
    public void Pipeline_ChainsPreprocessingIntoAModel()
    {
        var data = Datasets.MakeBlobs(samples: 300, features: 6, centers: 3, spread: 1.2, seed: 91);
        var split = Selection.Split(data.Features, data.Target, testSize: 0.3, seed: 91, stratify: true);

        var pipeline = new Pipeline()
            .Add(new StandardScaler())
            .Add(new PCA(components: 3))
            .Add(new RandomForestClassifier(nTrees: 40, seed: 91));

        pipeline.Fit(split.TrainX, split.TrainY);

        Assert.True(pipeline.IsFitted);
        Assert.True(pipeline.Score(split.TestX, split.TestY) > 0.85);
        Assert.Equal(3, pipeline.Transform(split.TestX).Shape[1]);
    }

    [Fact]
    public void Pipeline_RejectsAnEstimatorThatIsNotLast()
    {
        var pipeline = new Pipeline()
            .Add(new LogisticRegression())
            .Add(new StandardScaler());

        var data = Datasets.MakeBlobs(samples: 40, centers: 2, seed: 93);
        Assert.Throws<InvalidOperationException>(() => pipeline.Fit(data.Features, data.Target));
    }

    [Fact]
    public void Pipeline_MustBeFittedBeforePredicting()
    {
        var pipeline = new Pipeline().Add(new StandardScaler()).Add(new LogisticRegression());
        Assert.Throws<InvalidOperationException>(() => pipeline.Predict(NdArray.Ones(3, 2)));
    }

    [Fact]
    public void GridSearch_FindsTheBetterHyperParameter()
    {
        var data = Datasets.MakeMoons(samples: 250, noise: 0.25, seed: 95);

        var search = new GridSearch(p => new DecisionTree(maxDepth: (int)p["maxDepth"]), folds: 4, seed: 95)
            .AddParameter("maxDepth", 1, 3, 6);
        search.Fit(data.Features, data.Target);

        Assert.Equal(3, search.Results.Count);
        Assert.NotNull(search.Best);
        Assert.NotNull(search.BestModel);
        // Depth 1 cannot separate two moons; the search must not pick it.
        Assert.NotEqual(1, (int)search.Best!.Parameters["maxDepth"]);
    }

    [Fact]
    public void ModelPersistence_RoundTripsALinearModel()
    {
        var (x, y, _) = Datasets.MakeRegression(samples: 100, features: 3, seed: 97);
        var model = new LinearRegression();
        model.Fit(x, y);

        var path = Path.Combine(Path.GetTempPath(), $"gravi-{Guid.NewGuid():N}.json");
        try
        {
            ModelPersistence.Save(ModelPersistence.Capture(model), path);
            var loaded = ModelPersistence.Load(path);

            Assert.Equal(nameof(LinearRegression), loaded.ModelType);
            Assert.Equal(3, loaded.FeatureCount);
            for (var j = 0; j < 3; j++)
                Assert.Equal(model.Coefficients.At(j), loaded.Parameters["coefficients"][j], 12);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }
}

public class RealDatasetTests
{
    [Fact]
    public void Iris_LoadsWithTheExpectedShape()
    {
        var iris = Datasets.LoadIris();
        Assert.Equal(150, iris.SampleCount);
        Assert.Equal(4, iris.FeatureCount);
        Assert.Equal(3, iris.TargetNames.Count);
    }

    [Fact]
    public void Iris_IsClassifiedAccuratelyByARandomForest()
    {
        var iris = Datasets.LoadIris();
        var split = Selection.Split(iris.Features, iris.Target, testSize: 0.3, seed: 101, stratify: true);

        var forest = new RandomForestClassifier(nTrees: 100, seed: 101);
        forest.Fit(split.TrainX, split.TrainY);

        Assert.True(forest.Score(split.TestX, split.TestY) > 0.9);
    }

    [Fact]
    public void Titanic_LoadsAndTrains()
    {
        var titanic = Datasets.LoadTitanic();
        Assert.Equal(891, titanic.SampleCount);
        Assert.Equal(7, titanic.FeatureCount);

        var split = Selection.Split(titanic.Features, titanic.Target, testSize: 0.25, seed: 103, stratify: true);
        var forest = new RandomForestClassifier(nTrees: 80, maxDepth: 8, seed: 103);
        forest.Fit(split.TrainX, split.TrainY);

        // The published baseline for this dataset sits around 0.78-0.83.
        Assert.True(forest.Score(split.TestX, split.TestY) > 0.75);
    }

    [Fact]
    public void Digits_LoadsAndIsClassifiable()
    {
        var digits = Datasets.LoadDigits();
        Assert.Equal(1797, digits.SampleCount);
        Assert.Equal(64, digits.FeatureCount);

        var split = Selection.Split(digits.Features, digits.Target, testSize: 0.3, seed: 105, stratify: true);
        var model = new Gravicode.Science.GraviLearn.Neighbors.KNearestNeighborsClassifier(k: 3);
        model.Fit(split.TrainX, split.TrainY);

        Assert.True(model.Score(split.TestX, split.TestY) > 0.95);
    }
}
