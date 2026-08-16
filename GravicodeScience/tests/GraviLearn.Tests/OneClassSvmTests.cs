using Gravicode.Science.GraviLearn.Anomaly;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.GraviLearn;

/// <summary>
/// Tests for the one-class SVM.
/// </summary>
/// <remarks>
/// The strongest available reference is theoretical rather than numerical: Schölkopf's nu-property
/// says nu bounds the outlier fraction from above and the support-vector fraction from below. That
/// holds for any correct solver on any data, so it is checked directly — a plausible-looking
/// implementation with a broken constraint violates it immediately.
/// </remarks>
public class OneClassSvmTests
{
    /// <summary>A tight Gaussian blob at the origin.</summary>
    private static NdArray Blob(int rows = 200, int seed = 7, double centre = 0, double spread = 1)
    {
        var rng = new GraviRandom(seed);
        var x = NdArray.Zeros(rows, 2);
        for (var i = 0; i < rows; i++)
        {
            x[i, 0] = centre + rng.Normal() * spread;
            x[i, 1] = centre + rng.Normal() * spread;
        }
        return x;
    }

    [Fact]
    public void PointsFarFromTheTrainingDataAreFlagged()
    {
        var model = new OneClassSvm(nu: 0.1).Fit(Blob());

        var outliers = NdArray.FromArray(new double[,] { { 20, 20 }, { -30, 15 }, { 0, 40 } });
        var predictions = model.Predict(outliers);

        for (var i = 0; i < 3; i++)
            Assert.Equal(-1.0, predictions.At(i));
    }

    [Fact]
    public void PointsInTheMiddleOfTheTrainingDataAreNot()
    {
        var model = new OneClassSvm(nu: 0.1).Fit(Blob());

        var inliers = NdArray.FromArray(new double[,] { { 0, 0 }, { 0.2, -0.1 }, { -0.3, 0.25 } });
        var predictions = model.Predict(inliers);

        for (var i = 0; i < 3; i++)
            Assert.Equal(1.0, predictions.At(i));
    }

    [Theory]
    [InlineData(0.05)]
    [InlineData(0.1)]
    [InlineData(0.3)]
    [InlineData(0.5)]
    public void NuBoundsTheOutlierAndSupportVectorFractions(double nu)
    {
        // The nu-property, which is what makes the parameter meaningful rather than a knob.
        var x = Blob(rows: 200, seed: 11);
        var model = new OneClassSvm(nu).Fit(x);

        var predictions = model.Predict(x);
        var outliers = 0;
        for (var i = 0; i < predictions.Size; i++) if (predictions.At(i) < 0) outliers++;

        var outlierFraction = (double)outliers / x.Shape[0];
        var supportFraction = (double)model.SupportVectorCount / x.Shape[0];

        // Both bounds get a little slack for the solver's finite tolerance.
        Assert.True(outlierFraction <= nu + 0.05,
            $"nu={nu}: {outlierFraction:P1} of points were outliers, above the upper bound");
        Assert.True(supportFraction >= nu - 0.05,
            $"nu={nu}: only {supportFraction:P1} were support vectors, below the lower bound");
    }

    [Fact]
    public void ALargerNuDrawsATighterBoundary()
    {
        // Directly implied by the property above: raising nu buys a tighter fit by declaring more
        // of the training data to be contamination.
        var x = Blob(rows: 200, seed: 13);

        int Rejected(double nu)
        {
            var predictions = new OneClassSvm(nu).Fit(x).Predict(x);
            var count = 0;
            for (var i = 0; i < predictions.Size; i++) if (predictions.At(i) < 0) count++;
            return count;
        }

        Assert.True(Rejected(0.4) > Rejected(0.05),
            "a larger nu should exclude more of the training data");
    }

    [Fact]
    public void TheDecisionFunctionFallsAwayWithDistance()
    {
        // What makes the score usable for ranking alerts rather than only for a yes/no.
        var model = new OneClassSvm(nu: 0.1).Fit(Blob());

        var probe = NdArray.FromArray(new double[,] { { 0, 0 }, { 3, 0 }, { 6, 0 }, { 12, 0 } });
        var scores = model.DecisionFunction(probe);

        for (var i = 1; i < 4; i++)
            Assert.True(scores.At(i) < scores.At(i - 1),
                $"score did not fall between probe {i - 1} and {i}");
    }

    [Fact]
    public void TheSignOfTheDecisionFunctionAgreesWithPredict()
    {
        var x = Blob(rows: 100, seed: 17);
        var model = new OneClassSvm(nu: 0.2).Fit(x);

        var scores = model.DecisionFunction(x);
        var predictions = model.Predict(x);

        for (var i = 0; i < x.Shape[0]; i++)
            Assert.Equal(scores.At(i) >= 0 ? 1.0 : -1.0, predictions.At(i));
    }

    [Fact]
    public void ContaminatedTrainingDataStillYieldsAUsableBoundary()
    {
        // 190 clean points and 10 scattered far away. What nu buys is that the contamination does
        // not drag the boundary out to enclose it — checked on held-out points, because a training
        // point that is a support vector appears in its own decision function and an isolated one
        // then lands exactly on the boundary rather than outside it.
        var clean = Blob(rows: 190, seed: 19);
        var x = NdArray.Zeros(200, 2);

        for (var i = 0; i < 190; i++) { x[i, 0] = clean[i, 0]; x[i, 1] = clean[i, 1]; }

        for (var i = 190; i < 200; i++)
        {
            var angle = (i - 190) * 2 * Math.PI / 10;
            x[i, 0] = 12 * Math.Cos(angle);
            x[i, 1] = 12 * Math.Sin(angle);
        }

        var model = new OneClassSvm(nu: 0.05).Fit(x);

        // Held-out probes in the same ring the contamination sits on: all rejected.
        var probes = NdArray.Zeros(10, 2);
        for (var i = 0; i < 10; i++)
        {
            var angle = (i + 0.5) * 2 * Math.PI / 10;
            probes[i, 0] = 12 * Math.Cos(angle);
            probes[i, 1] = 12 * Math.Sin(angle);
        }

        var predictions = model.Predict(probes);
        for (var i = 0; i < 10; i++)
            Assert.Equal(-1.0, predictions.At(i));

        // And the centre of the clean blob is still comfortably inside.
        Assert.Equal(1.0, model.Predict(NdArray.FromArray(new double[,] { { 0, 0 } })).At(0));
    }

    [Fact]
    public void AnIsolatedTrainingPointScoresAboveAnIdenticalHeldOutOne()
    {
        // Pins the self-contribution effect, so it stays a documented property rather than becoming
        // a mystery the next time someone scores the training set and finds an outlier marked
        // normal. How far above depends on the multiplier the point ends up with — when rho falls
        // below the box bound the training copy lands exactly on the boundary — so the invariant
        // worth asserting is the direction, which holds in every regime.
        var clean = Blob(rows: 99, seed: 29);
        var x = NdArray.Zeros(100, 2);
        for (var i = 0; i < 99; i++) { x[i, 0] = clean[i, 0]; x[i, 1] = clean[i, 1]; }
        x[99, 0] = 40;
        x[99, 1] = 40;

        var model = new OneClassSvm(nu: 0.05).Fit(x);
        var inTraining = model.DecisionFunction(x).At(99);

        // An equally isolated point the model never saw has no self-contribution, so it scores the
        // full negative offset. (A probe at the *same* coordinates would not show this: it would
        // pick the self-similarity term back up off the stored support vector.)
        var heldOut = model.DecisionFunction(NdArray.FromArray(new double[,] { { -40, 40 } })).At(0);

        Assert.True(heldOut < 0, $"a held-out isolated point scored {heldOut:E3}");
        Assert.True(heldOut < inTraining,
            $"held out {heldOut:E3} should score below the training copy's {inTraining:E3}");
    }

    [Fact]
    public void TheLinearKernelStillSeparatesAnObviousCase()
    {
        var model = new OneClassSvm(nu: 0.1, kernel: SvmKernel.Linear).Fit(Blob(centre: 5, spread: 0.3));

        Assert.Equal(1.0, model.Predict(NdArray.FromArray(new double[,] { { 5, 5 } })).At(0));
        Assert.Equal(-1.0, model.Predict(NdArray.FromArray(new double[,] { { -20, -20 } })).At(0));
    }

    [Fact]
    public void GammaIsResolvedFromTheDataWhenNotGiven()
    {
        // The scale heuristic has to adapt, or an unstandardised dataset gets a meaningless width.
        var tight = new OneClassSvm(nu: 0.1).Fit(Blob(spread: 0.1));
        var wide = new OneClassSvm(nu: 0.1).Fit(Blob(spread: 10.0));

        Assert.True(tight.Gamma > wide.Gamma,
            $"tighter data should give a larger gamma: {tight.Gamma:E3} vs {wide.Gamma:E3}");

        // And an explicit value is respected exactly.
        Assert.Equal(0.25, new OneClassSvm(nu: 0.1, gamma: 0.25).Fit(Blob()).Gamma);
    }

    [Fact]
    public void PredictingBeforeFittingThrows()
    {
        Assert.Throws<InvalidOperationException>(
            () => new OneClassSvm().Predict(NdArray.Zeros(1, 2)));
    }

    [Fact]
    public void AnInvalidNuIsRejected()
    {
        // nu of 0 has no feasible solution and nu above 1 is meaningless; both fail at construction
        // rather than producing a boundary that means nothing.
        Assert.Throws<ArgumentOutOfRangeException>(() => new OneClassSvm(nu: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new OneClassSvm(nu: 1.5));
    }

    [Fact]
    public void PredictingWithTheWrongFeatureCountThrows()
    {
        var model = new OneClassSvm(nu: 0.1).Fit(Blob());
        Assert.Throws<ArgumentException>(() => model.Predict(NdArray.Zeros(3, 5)));
    }
}
