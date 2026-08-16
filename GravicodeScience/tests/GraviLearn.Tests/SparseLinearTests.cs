using Gravicode.Science.GraviLearn.Linear;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.GraviLearn;

/// <summary>
/// Tests for logistic regression trained directly on a sparse matrix.
/// </summary>
/// <remarks>
/// The reference is the dense implementation. A sparse optimiser that reaches a <em>different</em>
/// model is not a faster version of the same thing, it is a different algorithm — so the central
/// test compares the two coefficient vectors rather than just the accuracies, which would agree
/// even if the weights had drifted.
/// </remarks>
public class SparseLinearTests
{
    /// <summary>A sparse, linearly separable problem: few active features per row.</summary>
    private static (SparseMatrix Sparse, NdArray Dense, NdArray Y) Problem(
        int rows = 400, int features = 300, int active = 8, int seed = 5)
    {
        var rng = new GraviRandom(seed);
        var triplets = new List<(int, int, double)>();
        var dense = NdArray.Zeros(rows, features);
        var y = NdArray.Zeros(rows);

        for (var i = 0; i < rows; i++)
        {
            var positive = i % 2 == 0;
            y.SetAt(i, positive ? 1 : 0);

            for (var k = 0; k < active; k++)
            {
                // The signal: positives draw mostly from the low half of the feature space.
                var column = positive
                    ? (rng.NextDouble() < 0.8 ? rng.Next(0, features / 2) : rng.Next(features / 2, features))
                    : (rng.NextDouble() < 0.8 ? rng.Next(features / 2, features) : rng.Next(0, features / 2));

                var value = 1.0 + rng.NextDouble();

                triplets.Add((i, column, value));
                dense[i, column] += value;
            }
        }

        // Duplicate (row, column) pairs are summed by FromTriplets, matching the dense accumulation.
        return (SparseMatrix.FromTriplets(rows, features, triplets), dense, y);
    }

    [Fact]
    public void ItLearnsTheSameModelAsTheDenseImplementation()
    {
        // The claim that matters. Equal accuracy would not prove this: two different weight
        // vectors can classify a separable problem identically.
        var (sparse, dense, y) = Problem();

        var sparseModel = new SparseLogisticRegression(learningRate: 0.5, maxIterations: 150).Fit(sparse, y);

        var denseModel = new LogisticRegression(learningRate: 0.5, maxIterations: 150);
        denseModel.Fit(dense, y);

        var a = sparseModel.Coefficients;
        var b = denseModel.Coefficients;

        var largestGap = 0.0;
        for (var j = 0; j < sparse.Columns; j++)
            largestGap = Math.Max(largestGap, Math.Abs(a[0, j] - b.At(j)));

        Assert.True(largestGap < 1e-9,
            $"the sparse and dense coefficients diverged by {largestGap:E3}");
        Assert.Equal(denseModel.Intercept, sparseModel.Intercepts.At(0), 9);
    }

    [Fact]
    public void ItSeparatesASeparableProblem()
    {
        var (sparse, _, y) = Problem();
        var model = new SparseLogisticRegression(learningRate: 0.5, maxIterations: 200).Fit(sparse, y);

        Assert.True(model.Score(sparse, y) > 0.95, $"accuracy was {model.Score(sparse, y):P2}");
    }

    [Fact]
    public void ProbabilitiesAreBoundedAndAgreeWithTheLabels()
    {
        var (sparse, _, y) = Problem();
        var model = new SparseLogisticRegression(learningRate: 0.5, maxIterations: 200).Fit(sparse, y);

        var probabilities = model.PredictProbabilities(sparse);

        for (var i = 0; i < sparse.Rows; i++)
            Assert.InRange(probabilities[i, 0], 0.0, 1.0);

        // A confident model should put the positives above 0.5 and the negatives below.
        var agreed = 0;
        for (var i = 0; i < sparse.Rows; i++)
            if ((probabilities[i, 0] >= 0.5) == (y.At(i) == 1)) agreed++;

        Assert.True(agreed > sparse.Rows * 0.95, $"only {agreed}/{sparse.Rows} agreed");
    }

    [Fact]
    public void TheSigmoidDoesNotOverflowOnLargeScores()
    {
        // exp of a large positive number overflows; the sign split avoids it. A row of large
        // values is what triggers it, and the failure is a NaN probability rather than an error.
        var triplets = new List<(int, int, double)>();
        for (var i = 0; i < 20; i++)
            for (var j = 0; j < 5; j++)
                triplets.Add((i, j, i % 2 == 0 ? 1000.0 : -1000.0));

        var sparse = SparseMatrix.FromTriplets(20, 5, triplets);
        var y = NdArray.Zeros(20);
        for (var i = 0; i < 20; i++) y.SetAt(i, i % 2);

        var model = new SparseLogisticRegression(learningRate: 0.1, maxIterations: 50).Fit(sparse, y);
        var probabilities = model.PredictProbabilities(sparse);

        for (var i = 0; i < 20; i++)
        {
            Assert.False(double.IsNaN(probabilities[i, 0]), $"row {i} produced NaN");
            Assert.InRange(probabilities[i, 0], 0.0, 1.0);
        }

        Assert.False(double.IsNaN(model.FinalLoss), "the loss went to NaN");
    }

    [Fact]
    public void MultiClassWorksByOneVersusRest()
    {
        var rng = new GraviRandom(11);
        var triplets = new List<(int, int, double)>();
        var y = NdArray.Zeros(300);

        for (var i = 0; i < 300; i++)
        {
            var label = i % 3;
            y.SetAt(i, label);

            // Each class activates its own band of features.
            for (var k = 0; k < 6; k++)
                triplets.Add((i, label * 30 + rng.Next(30), 1.0 + rng.NextDouble()));
        }

        var sparse = SparseMatrix.FromTriplets(300, 90, triplets);
        var model = new SparseLogisticRegression(learningRate: 0.5, maxIterations: 200).Fit(sparse, y);

        Assert.Equal([0.0, 1.0, 2.0], model.Classes);
        Assert.True(model.Score(sparse, y) > 0.9, $"accuracy was {model.Score(sparse, y):P2}");

        // One-versus-rest fits one problem per class, so the coefficient matrix has three rows.
        Assert.Equal(3, model.Coefficients.Shape[0]);
    }

    [Fact]
    public void MultiClassProbabilitiesSumToOne()
    {
        var rng = new GraviRandom(13);
        var triplets = new List<(int, int, double)>();
        var y = NdArray.Zeros(150);

        for (var i = 0; i < 150; i++)
        {
            var label = i % 3;
            y.SetAt(i, label);
            for (var k = 0; k < 5; k++) triplets.Add((i, label * 20 + rng.Next(20), 1.0));
        }

        var sparse = SparseMatrix.FromTriplets(150, 60, triplets);
        var model = new SparseLogisticRegression(maxIterations: 100).Fit(sparse, y);

        var probabilities = model.PredictProbabilities(sparse);

        for (var i = 0; i < 150; i++)
        {
            var total = 0.0;
            for (var k = 0; k < 3; k++) total += probabilities[i, k];
            Assert.Equal(1.0, total, 9);
        }
    }

    [Fact]
    public void AnEmptyRowStillGetsAPrediction()
    {
        // A document with no known words is a real case, and its score is the intercept alone.
        // Skipping it or dividing by its zero norm would be the easy mistakes.
        var triplets = new List<(int, int, double)>();
        for (var i = 0; i < 40; i++)
            if (i != 7) triplets.Add((i, i % 10, 1.0));

        var sparse = SparseMatrix.FromTriplets(40, 10, triplets);
        var y = NdArray.Zeros(40);
        for (var i = 0; i < 40; i++) y.SetAt(i, i % 2);

        var model = new SparseLogisticRegression(maxIterations: 50).Fit(sparse, y);
        var probabilities = model.PredictProbabilities(sparse);

        Assert.False(double.IsNaN(probabilities[7, 0]));
        Assert.InRange(probabilities[7, 0], 0.0, 1.0);
    }

    [Fact]
    public void TheL2PenaltyShrinksTheCoefficients()
    {
        var (sparse, _, y) = Problem();

        double Magnitude(double penalty)
        {
            var model = new SparseLogisticRegression(
                learningRate: 0.5, maxIterations: 150, l2Penalty: penalty).Fit(sparse, y);

            var total = 0.0;
            for (var j = 0; j < sparse.Columns; j++) total += Math.Abs(model.Coefficients[0, j]);
            return total;
        }

        Assert.True(Magnitude(1.0) < Magnitude(0.0),
            "the penalty did not shrink the weights");
    }

    [Fact]
    public void TopFeaturesRanksByWeight()
    {
        var (sparse, _, y) = Problem();
        var model = new SparseLogisticRegression(learningRate: 0.5, maxIterations: 200).Fit(sparse, y);

        var top = model.TopFeatures(10);

        Assert.Equal(10, top.Count);
        for (var i = 1; i < top.Count; i++)
            Assert.True(top[i].Weight <= top[i - 1].Weight, "the ranking was not descending");

        // The signal lives in the low half of the feature space, so that is where the strongest
        // positive weights should be.
        Assert.True(top.Count(f => f.Feature < sparse.Columns / 2) >= 7,
            "the top features were not concentrated where the signal is");
    }

    [Fact]
    public void ItRefusesAMatrixOfTheWrongWidth()
    {
        var (sparse, _, y) = Problem();
        var model = new SparseLogisticRegression(maxIterations: 20).Fit(sparse, y);

        Assert.Throws<ArgumentException>(
            () => model.Predict(SparseMatrix.FromTriplets(5, 999, [(0, 0, 1.0)])));
    }

    [Fact]
    public void MalformedInputsAreRejected()
    {
        var (sparse, _, _) = Problem(rows: 10, features: 20);

        Assert.Throws<ArgumentException>(
            () => new SparseLogisticRegression().Fit(sparse, NdArray.Zeros(3)));

        Assert.Throws<InvalidOperationException>(
            () => new SparseLogisticRegression().Predict(sparse));
    }

    [Fact]
    public void OnSeparableDataItConvergesOnlyWithAPenalty()
    {
        // Worth pinning because it looks like a bug and is not. Unpenalised logistic regression on
        // separable data has its maximum likelihood at INFINITY: the weights can always grow a
        // little more and shave a little more off the loss, so a tolerance test never fires and the
        // fit runs to the iteration cap. Adding an L2 term makes the objective strictly convex with
        // a finite optimum, and it converges.
        var (sparse, _, y) = Problem();

        var unpenalised = new SparseLogisticRegression(
            learningRate: 0.5, maxIterations: 3000, tolerance: 1e-8).Fit(sparse, y);

        var penalised = new SparseLogisticRegression(
            learningRate: 0.5, maxIterations: 3000, l2Penalty: 0.05, tolerance: 1e-8).Fit(sparse, y);

        Assert.Equal(3000, unpenalised.IterationsRun);
        Assert.True(penalised.IterationsRun < 3000,
            $"the penalised fit also ran to the cap ({penalised.IterationsRun})");

        // Both still classify the data; it is the optimiser that differs, not the answer.
        Assert.True(unpenalised.Score(sparse, y) > 0.95);
        Assert.True(penalised.Score(sparse, y) > 0.95);

        Assert.False(double.IsNaN(unpenalised.FinalLoss));
        Assert.False(double.IsNaN(penalised.FinalLoss));
    }
}
