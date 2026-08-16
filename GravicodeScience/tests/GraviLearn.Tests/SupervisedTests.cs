using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviLearn.Linear;
using Gravicode.Science.GraviLearn.ModelSelection;
using Gravicode.Science.GraviLearn.Neighbors;
using Gravicode.Science.GraviLearn.Preprocessing;
using Gravicode.Science.GraviLearn.Trees;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.GraviLearn;

public class LinearModelTests
{
    [Fact]
    public void LinearRegression_RecoversAKnownLinearRelationship()
    {
        // y = 3*x0 - 2*x1 + 5
        var x = NdArray.FromArray(new double[,] { { 1, 1 }, { 2, 1 }, { 3, 2 }, { 4, 3 }, { 5, 5 } });
        var y = NdArray.Zeros(5);
        for (var i = 0; i < 5; i++) y.SetAt(i, 3 * x[i, 0] - 2 * x[i, 1] + 5);

        var model = new LinearRegression();
        model.Fit(x, y);

        Assert.Equal(3.0, model.Coefficients.At(0), 8);
        Assert.Equal(-2.0, model.Coefficients.At(1), 8);
        Assert.Equal(5.0, model.Intercept, 8);
        Assert.Equal(1.0, model.Score(x, y), 8);
    }

    [Fact]
    public void LinearRegression_HandlesNoisyDataWithHighR2()
    {
        var (x, y, coefficients) = Datasets.MakeRegression(samples: 400, features: 4, noise: 0.3, seed: 7);
        var model = new LinearRegression();
        model.Fit(x, y);

        Assert.True(model.Score(x, y) > 0.95);
        for (var j = 0; j < 4; j++)
            Assert.Equal(coefficients.At(j), model.Coefficients.At(j), 1);
    }

    [Fact]
    public void RidgeRegression_ShrinksCoefficientsTowardZero()
    {
        var (x, y, _) = Datasets.MakeRegression(samples: 100, features: 6, noise: 1.0, seed: 11);

        var plain = new LinearRegression();
        plain.Fit(x, y);
        var ridge = new RidgeRegression(alpha: 50.0);
        ridge.Fit(x, y);

        Assert.True(LinAlg.Norm(ridge.Coefficients) < LinAlg.Norm(plain.Coefficients));
    }

    [Fact]
    public void LassoRegression_DrivesIrrelevantCoefficientsToExactlyZero()
    {
        // Only the first two features carry signal; the other six are noise.
        var rng = new GraviRandom(3);
        var x = rng.StandardNormal(300, 8);
        var y = NdArray.Zeros(300);
        for (var i = 0; i < 300; i++) y.SetAt(i, 4 * x[i, 0] - 3 * x[i, 1] + rng.Normal(0, 0.1));

        var lasso = new LassoRegression(alpha: 0.3);
        lasso.Fit(x, y);

        Assert.True(lasso.ZeroCoefficients >= 4);
        Assert.True(Math.Abs(lasso.Coefficients.At(0)) > 1.0);
        Assert.True(Math.Abs(lasso.Coefficients.At(1)) > 1.0);
    }

    [Fact]
    public void LogisticRegression_SeparatesTwoWellSpacedBlobs()
    {
        var data = Datasets.MakeBlobs(samples: 200, centers: 2, spread: 0.6, seed: 5);
        var model = new LogisticRegression(learningRate: 0.5, maxIterations: 2000);
        model.Fit(data.Features, data.Target);

        Assert.True(model.Score(data.Features, data.Target) > 0.95);

        var probabilities = model.PredictProbabilities(data.Features);
        for (var i = 0; i < 10; i++)
            Assert.Equal(1.0, probabilities[i, 0] + probabilities[i, 1], 9);
    }

    [Fact]
    public void LogisticRegression_HandlesThreeClassesWithOneVsRest()
    {
        var data = Datasets.MakeBlobs(samples: 300, centers: 3, spread: 0.8, seed: 9);
        var model = new LogisticRegression(learningRate: 0.5, maxIterations: 2000);
        model.Fit(data.Features, data.Target);

        Assert.Equal(3, model.Classes.Count);
        Assert.True(model.Score(data.Features, data.Target) > 0.9);

        var probabilities = model.PredictProbabilities(data.Features);
        Assert.Equal(3, probabilities.Shape[1]);
        for (var i = 0; i < 10; i++)
        {
            var total = 0.0;
            for (var c = 0; c < 3; c++) total += probabilities[i, c];
            Assert.Equal(1.0, total, 9);
        }
    }

    [Fact]
    public void LinearSvm_SeparatesLinearlySeparableData()
    {
        var data = Datasets.MakeBlobs(samples: 200, centers: 2, spread: 0.5, seed: 13);
        var model = new LinearSupportVectorClassifier(c: 1.0, maxIterations: 200);
        model.Fit(data.Features, data.Target);

        Assert.True(model.Score(data.Features, data.Target) > 0.95);
    }

    [Fact]
    public void UnfittedModel_FailsWithAClearMessage()
    {
        var model = new LinearRegression();
        var ex = Assert.Throws<InvalidOperationException>(() => model.Predict(NdArray.Ones(2, 2)));
        Assert.Contains("fitted", ex.Message);
    }

    [Fact]
    public void PredictingWithTheWrongFeatureCount_Fails()
    {
        var (x, y, _) = Datasets.MakeRegression(samples: 50, features: 3, seed: 1);
        var model = new LinearRegression();
        model.Fit(x, y);

        Assert.Throws<ArgumentException>(() => model.Predict(NdArray.Ones(5, 7)));
    }
}

public class TreeTests
{
    [Fact]
    public void DecisionTree_PerfectlyFitsSeparableTrainingData()
    {
        var data = Datasets.MakeBlobs(samples: 150, centers: 3, spread: 0.5, seed: 17);
        var tree = new DecisionTree();
        tree.Fit(data.Features, data.Target);

        Assert.Equal(1.0, tree.Score(data.Features, data.Target), 6);
        Assert.True(tree.LeafCount >= 3);
    }

    [Fact]
    public void MaxDepth_LimitsTheTree()
    {
        var data = Datasets.MakeBlobs(samples: 200, centers: 4, spread: 1.5, seed: 19);
        var tree = new DecisionTree(maxDepth: 2);
        tree.Fit(data.Features, data.Target);

        Assert.True(tree.Depth <= 2);
        Assert.True(tree.LeafCount <= 4);
    }

    [Fact]
    public void EntropyAndGini_BothProduceUsableTrees()
    {
        var data = Datasets.MakeBlobs(samples: 200, centers: 3, spread: 1.0, seed: 21);
        foreach (var criterion in new[] { SplitCriterion.Gini, SplitCriterion.Entropy })
        {
            var tree = new DecisionTree(criterion);
            tree.Fit(data.Features, data.Target);
            Assert.True(tree.Score(data.Features, data.Target) > 0.95);
        }
    }

    [Fact]
    public void FeatureImportances_SumToOneAndFavourTheInformativeFeature()
    {
        // Only feature 0 determines the label; feature 1 is pure noise.
        var rng = new GraviRandom(23);
        var x = NdArray.Zeros(200, 2);
        var y = NdArray.Zeros(200);
        for (var i = 0; i < 200; i++)
        {
            x[i, 0] = rng.Normal();
            x[i, 1] = rng.Normal();
            y.SetAt(i, x[i, 0] > 0 ? 1 : 0);
        }

        var tree = new DecisionTree(maxDepth: 3);
        tree.Fit(x, y);
        var importances = tree.FeatureImportances;

        Assert.Equal(1.0, importances.Sum(), 8);
        Assert.True(importances.At(0) > importances.At(1));
    }

    [Fact]
    public void DecisionTreeRegressor_FitsAStepFunction()
    {
        var x = NdArray.Zeros(100, 1);
        var y = NdArray.Zeros(100);
        for (var i = 0; i < 100; i++)
        {
            x[i, 0] = i;
            y.SetAt(i, i < 50 ? 10.0 : 30.0);
        }

        var tree = new DecisionTreeRegressor(maxDepth: 2);
        tree.Fit(x, y);
        Assert.Equal(1.0, tree.Score(x, y), 6);
    }

    [Fact]
    public void RandomForest_MatchesOrBeatsASingleTreeOnHeldOutData()
    {
        var data = Datasets.MakeMoons(samples: 300, noise: 0.25, seed: 27);
        var split = Selection.Split(data.Features, data.Target, testSize: 0.3, seed: 27, stratify: true);

        var tree = new DecisionTree();
        tree.Fit(split.TrainX, split.TrainY);

        var forest = new RandomForestClassifier(nTrees: 60, seed: 27);
        forest.Fit(split.TrainX, split.TrainY);

        var treeScore = Metrics.Accuracy(split.TestY, tree.Predict(split.TestX));
        var forestScore = Metrics.Accuracy(split.TestY, forest.Predict(split.TestX));

        Assert.True(forestScore >= treeScore - 0.02, $"forest {forestScore:F3} vs tree {treeScore:F3}");
        Assert.True(forestScore > 0.8);
    }

    [Fact]
    public void RandomForest_ReportsAnOutOfBagScore()
    {
        var data = Datasets.MakeBlobs(samples: 200, centers: 3, spread: 1.0, seed: 29);
        var forest = new RandomForestClassifier(nTrees: 40, seed: 29);
        forest.Fit(data.Features, data.Target);

        Assert.False(double.IsNaN(forest.OutOfBagScore));
        Assert.True(forest.OutOfBagScore > 0.85);
    }

    [Fact]
    public void RandomForest_ProbabilitiesAreNormalised()
    {
        var data = Datasets.MakeBlobs(samples: 120, centers: 3, spread: 1.0, seed: 31);
        var forest = new RandomForestClassifier(nTrees: 20, seed: 31);
        forest.Fit(data.Features, data.Target);

        var probabilities = forest.PredictProbabilities(data.Features);
        for (var i = 0; i < probabilities.Shape[0]; i++)
        {
            var total = 0.0;
            for (var c = 0; c < probabilities.Shape[1]; c++) total += probabilities[i, c];
            Assert.Equal(1.0, total, 9);
        }
    }

    [Fact]
    public void GradientBoosting_ReducesLossEachRoundAndFitsNonLinearData()
    {
        var x = NdArray.Zeros(200, 1);
        var y = NdArray.Zeros(200);
        for (var i = 0; i < 200; i++)
        {
            var t = i / 200.0 * 6 - 3;
            x[i, 0] = t;
            y.SetAt(i, Math.Sin(t) * 3);
        }

        var model = new GradientBoostingRegressor(nTrees: 80, learningRate: 0.1, maxDepth: 3);
        model.Fit(x, y);

        Assert.True(model.Score(x, y) > 0.95);
        Assert.True(model.TrainingLoss[^1] < model.TrainingLoss[0]);
    }

    [Fact]
    public void GradientBoostingClassifier_HandlesBinaryTargets()
    {
        var data = Datasets.MakeMoons(samples: 200, noise: 0.2, seed: 33);
        var model = new GradientBoostingClassifier(nTrees: 60, maxDepth: 3);
        model.Fit(data.Features, data.Target);

        var score = model.Score(data.Features, data.Target);
        Assert.True(score > 0.95, $"accuracy = {score:F4}");
    }
}

public class NeighborAndBayesTests
{
    [Fact]
    public void KNearestNeighbors_ClassifiesBlobs()
    {
        var data = Datasets.MakeBlobs(samples: 200, centers: 3, spread: 0.8, seed: 37);
        var model = new KNearestNeighborsClassifier(k: 5);
        model.Fit(data.Features, data.Target);

        Assert.True(model.Score(data.Features, data.Target) > 0.95);
    }

    [Fact]
    public void KNearestNeighbors_WithK1ReproducesTrainingLabelsExactly()
    {
        var data = Datasets.MakeBlobs(samples: 100, centers: 3, spread: 1.0, seed: 39);
        var model = new KNearestNeighborsClassifier(k: 1);
        model.Fit(data.Features, data.Target);

        Assert.Equal(1.0, model.Score(data.Features, data.Target), 9);
    }

    [Fact]
    public void KNearestNeighbors_RejectsAKLargerThanTheTrainingSet()
    {
        var data = Datasets.MakeBlobs(samples: 10, centers: 2, seed: 41);
        var model = new KNearestNeighborsClassifier(k: 50);
        Assert.Throws<ArgumentException>(() => model.Fit(data.Features, data.Target));
    }

    [Fact]
    public void KNearestNeighborsRegressor_AveragesNeighbourTargets()
    {
        var x = NdArray.Zeros(50, 1);
        var y = NdArray.Zeros(50);
        for (var i = 0; i < 50; i++) { x[i, 0] = i; y.SetAt(i, i * 2.0); }

        var model = new KNearestNeighborsRegressor(k: 3);
        model.Fit(x, y);
        Assert.True(model.Score(x, y) > 0.99);
    }

    [Fact]
    public void GaussianNaiveBayes_ClassifiesWellSeparatedGaussians()
    {
        var data = Datasets.MakeBlobs(samples: 300, centers: 3, spread: 1.0, seed: 43);
        var model = new GaussianNaiveBayes();
        model.Fit(data.Features, data.Target);

        Assert.True(model.Score(data.Features, data.Target) > 0.95);
        Assert.Equal(1.0, model.Priors.Sum(), 9);
    }

    [Fact]
    public void MultinomialNaiveBayes_ClassifiesCountFeatures()
    {
        // Two "topics" with disjoint vocabularies.
        var rng = new GraviRandom(45);
        var x = NdArray.Zeros(200, 6);
        var y = NdArray.Zeros(200);
        for (var i = 0; i < 200; i++)
        {
            var topic = i % 2;
            y.SetAt(i, topic);
            for (var j = 0; j < 6; j++)
            {
                var inTopic = topic == 0 ? j < 3 : j >= 3;
                x[i, j] = rng.Poisson(inTopic ? 6.0 : 0.4);
            }
        }

        var model = new MultinomialNaiveBayes();
        model.Fit(x, y);
        Assert.True(model.Score(x, y) > 0.95);
    }

    [Fact]
    public void DistanceMetrics_MatchTheirDefinitions()
    {
        var a = NdArray.FromArray(new double[,] { { 0, 0 } });
        var b = NdArray.FromArray(new double[,] { { 3, 4 } });

        Assert.Equal(5.0, Distances.Between(a, 0, b, 0, DistanceMetric.Euclidean), 9);
        Assert.Equal(7.0, Distances.Between(a, 0, b, 0, DistanceMetric.Manhattan), 9);
        Assert.Equal(4.0, Distances.Between(a, 0, b, 0, DistanceMetric.Chebyshev), 9);
    }
}

public class PreprocessingTests
{
    [Fact]
    public void StandardScaler_ProducesZeroMeanUnitVariance()
    {
        var rng = new GraviRandom(47);
        var x = rng.Normal(50.0, 12.0, 300, 3);

        var scaler = new StandardScaler();
        var scaled = scaler.FitTransform(x);

        for (var j = 0; j < 3; j++)
        {
            Assert.Equal(0.0, Statistics.Mean(scaled.Column(j).Copy()), 9);
            Assert.Equal(1.0, Statistics.Std(scaled.Column(j).Copy()), 9);
        }
    }

    [Fact]
    public void StandardScaler_InverseTransformRoundTrips()
    {
        var rng = new GraviRandom(49);
        var x = rng.Normal(3.0, 2.0, 100, 4);

        var scaler = new StandardScaler();
        var restored = scaler.InverseTransform(scaler.FitTransform(x));
        Assert.True(UFunc.AllClose(x, restored, 1e-9));
    }

    [Fact]
    public void StandardScaler_LeavesAConstantColumnFinite()
    {
        var x = NdArray.Zeros(20, 2);
        for (var i = 0; i < 20; i++) { x[i, 0] = 5.0; x[i, 1] = i; }

        var scaled = new StandardScaler().FitTransform(x);
        for (var i = 0; i < 20; i++) Assert.Equal(0.0, scaled[i, 0], 12);
    }

    [Fact]
    public void MinMaxScaler_MapsIntoTheRequestedRange()
    {
        var rng = new GraviRandom(51);
        var scaled = new MinMaxScaler().FitTransform(rng.Uniform(-20.0, 30.0, 200, 3));

        Assert.Equal(0.0, Statistics.Min(scaled), 9);
        Assert.Equal(1.0, Statistics.Max(scaled), 9);
    }

    [Fact]
    public void SimpleImputer_FillsMissingValues()
    {
        var x = NdArray.FromArray(new double[,] { { 1, 10 }, { double.NaN, 20 }, { 3, double.NaN } });
        var filled = new SimpleImputer(ImputationStrategy.Mean).FitTransform(x);

        Assert.Equal(2.0, filled[1, 0], 9);
        Assert.Equal(15.0, filled[2, 1], 9);
    }

    [Fact]
    public void LabelEncoder_RoundTripsLabels()
    {
        var labels = NdArray.FromValues([10.0, 20.0, 10.0, 30.0]);
        var encoder = new LabelEncoder();
        var codes = encoder.FitTransform(labels);

        Assert.Equal([0.0, 1.0, 0.0, 2.0], codes.ToArray());
        Assert.True(UFunc.AllClose(encoder.InverseTransform(codes), labels, 0));
    }

    [Fact]
    public void OneHotEncoder_ExpandsCategories()
    {
        var x = NdArray.FromArray(new double[,] { { 0 }, { 1 }, { 2 }, { 1 } });
        var encoded = new OneHotEncoder().FitTransform(x);

        Assert.Equal(3, encoded.Shape[1]);
        Assert.Equal(1.0, encoded[0, 0]);
        Assert.Equal(1.0, encoded[3, 1]);
        Assert.Equal(0.0, encoded[3, 2]);
    }

    [Fact]
    public void PolynomialFeatures_GeneratesInteractionTerms()
    {
        var x = NdArray.FromArray(new double[,] { { 2, 3 } });
        var expanded = new PolynomialFeatures(degree: 2).FitTransform(x);

        // x0, x1, x0^2, x0*x1, x1^2
        Assert.Equal(5, expanded.Shape[1]);
        Assert.Equal(2.0, expanded[0, 0], 9);
        Assert.Equal(3.0, expanded[0, 1], 9);
        Assert.Equal(4.0, expanded[0, 2], 9);
        Assert.Equal(6.0, expanded[0, 3], 9);
        Assert.Equal(9.0, expanded[0, 4], 9);
    }
}
