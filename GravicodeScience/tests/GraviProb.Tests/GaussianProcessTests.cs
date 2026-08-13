using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviProb;
using Xunit;

namespace Gravicode.Science.Tests.GraviProb;

/// <summary>
/// Tests for Gaussian process regression.
/// </summary>
/// <remarks>
/// A GP is easy to get plausibly wrong: predictions that look smooth and sensible can come from a
/// posterior whose uncertainty is meaningless. So the pins here are the properties that break when
/// the conditioning is wrong — interpolation at zero noise, uncertainty growing away from the data,
/// and agreement with the closed-form Gaussian conditional, which is computed independently through
/// <see cref="MultivariateNormal"/>.
/// </remarks>
public class GaussianProcessTests
{
    private static NdArray Column(params double[] values)
    {
        var result = NdArray.Zeros(values.Length, 1);
        for (var i = 0; i < values.Length; i++) result[i, 0] = values[i];
        return result;
    }

    /// <summary>A smooth function sampled at a handful of points.</summary>
    private static (NdArray X, NdArray Y) SineData(int count = 12)
    {
        var x = NdArray.Zeros(count, 1);
        var y = NdArray.Zeros(count);

        for (var i = 0; i < count; i++)
        {
            var value = i * 2 * Math.PI / count;
            x[i, 0] = value;
            y.SetAt(i, Math.Sin(value));
        }

        return (x, y);
    }

    [Fact]
    public void WithAlmostNoNoiseTheProcessInterpolatesItsTrainingPoints()
    {
        // The defining behaviour of a noise-free GP: conditioning on an observation makes the
        // posterior pass through it. If it does not, the conditioning is wrong.
        var (x, y) = SineData();
        var gp = new GaussianProcess(new RbfKernel(lengthScale: 1.0), noise: 1e-10).Fit(x, y);

        var prediction = gp.Predict(x);

        for (var i = 0; i < y.Size; i++)
        {
            Assert.Equal(y.At(i), prediction.Mean.At(i), 6);
            Assert.True(prediction.Variance.At(i) < 1e-6,
                $"variance at a training point was {prediction.Variance.At(i):E3}");
        }
    }

    [Fact]
    public void UncertaintyGrowsAwayFromTheData()
    {
        // The reason to use a GP rather than a point-estimate regressor, and the thing that is
        // silently wrong if the variance formula loses its explained-variance term.
        var (x, y) = SineData();
        var gp = new GaussianProcess(new RbfKernel(lengthScale: 0.7), noise: 1e-8).Fit(x, y);

        var near = gp.Predict(Column(x[3, 0] + 0.01)).Variance.At(0);
        var far = gp.Predict(Column(50.0)).Variance.At(0);

        Assert.True(far > near, $"variance far from the data ({far:E3}) was not above near ({near:E3})");

        // Far enough away it must return to the prior variance, which is the kernel's own scale.
        Assert.Equal(1.0, far, 6);
    }

    [Fact]
    public void ThePosteriorAgreesWithTheClosedFormGaussianConditional()
    {
        // The strongest available reference: a GP posterior is exactly the conditional of a joint
        // multivariate normal whose covariance is the kernel matrix. That is computed here through
        // MultivariateNormal.Conditional, which is tested separately against textbook formulas.
        var (x, y) = SineData(count: 6);
        var kernel = new RbfKernel(lengthScale: 1.2, variance: 1.5);
        const double noise = 0.05;

        var test = Column(1.0, 2.5);

        // Build the joint over [test; train] and condition on the training outputs.
        var all = NdArray.Zeros(x.Shape[0] + test.Shape[0], 1);
        for (var i = 0; i < test.Shape[0]; i++) all[i, 0] = test[i, 0];
        for (var i = 0; i < x.Shape[0]; i++) all[test.Shape[0] + i, 0] = x[i, 0];

        var joint = kernel.Matrix(all, all);
        for (var i = test.Shape[0]; i < all.Shape[0]; i++) joint[i, i] += noise;   // noise on training only
        for (var i = 0; i < all.Shape[0]; i++) joint[i, i] += 1e-10;               // jitter for the factorisation

        var conditional = new MultivariateNormal(NdArray.Zeros(all.Shape[0]), joint)
            .Conditional([0, 1], [2, 3, 4, 5, 6, 7], Centred(y));

        var gp = new GaussianProcess(kernel, noise).Fit(x, y);
        var prediction = gp.Predict(test);

        // The reference needs a jitter on the test-point diagonal to factorise where the GP does
        // not, so the two agree to about that jitter rather than to machine precision.
        var mean = Mean(y);
        for (var i = 0; i < 2; i++)
        {
            Assert.True(Math.Abs(conditional.Mean.At(i) + mean - prediction.Mean.At(i)) < 1e-9,
                $"mean {i}: {prediction.Mean.At(i):R} against {conditional.Mean.At(i) + mean:R}");
            Assert.True(Math.Abs(conditional.Covariance[i, i] - prediction.Variance.At(i)) < 1e-9,
                $"variance {i}: {prediction.Variance.At(i):R} against {conditional.Covariance[i, i]:R}");
        }
    }

    [Fact]
    public void TheProcessRecoversASmoothFunctionBetweenItsObservations()
    {
        var (x, y) = SineData(count: 20);
        var gp = new GaussianProcess(new RbfKernel(lengthScale: 0.8), noise: 1e-6).Fit(x, y);

        // Points between the training inputs, where interpolation should be accurate.
        foreach (var probe in new[] { 0.5, 1.7, 3.3, 4.9 })
        {
            var predicted = gp.Predict(Column(probe)).Mean.At(0);
            Assert.True(Math.Abs(predicted - Math.Sin(probe)) < 0.05,
                $"at {probe} the process predicted {predicted:F4} against {Math.Sin(probe):F4}");
        }
    }

    [Fact]
    public void TargetsAreCentredSoAnOffsetSeriesIsNotPulledTowardsZero()
    {
        // A GP has a zero prior mean. Without centring, a series sitting at 1000 gets predictions
        // dragged towards zero between observations — a bias that looks like a mysterious sag.
        var x = Column(0, 1, 2, 3, 4);
        var y = NdArray.FromValues([1000, 1001, 1000, 1001, 1000]);

        var gp = new GaussianProcess(new RbfKernel(lengthScale: 0.5), noise: 1e-6).Fit(x, y);

        // Far from the data the mean must return to the series level, not to zero.
        var far = gp.Predict(Column(100.0)).Mean.At(0);
        Assert.True(Math.Abs(far - 1000.4) < 1.0, $"far from the data the mean was {far:F3}");
    }

    [Fact]
    public void MoreNoiseMeansLessInterpolationAndMoreSmoothing()
    {
        var (x, y) = SineData();

        var tight = new GaussianProcess(new RbfKernel(lengthScale: 1.0), noise: 1e-8).Fit(x, y);
        var loose = new GaussianProcess(new RbfKernel(lengthScale: 1.0), noise: 1.0).Fit(x, y);

        var tightError = 0.0;
        var looseError = 0.0;

        for (var i = 0; i < y.Size; i++)
        {
            tightError += Math.Abs(tight.Predict(x).Mean.At(i) - y.At(i));
            looseError += Math.Abs(loose.Predict(x).Mean.At(i) - y.At(i));
        }

        Assert.True(looseError > tightError,
            "a larger noise term should stop the process from chasing every observation");
    }

    [Fact]
    public void TheMarginalLikelihoodPrefersTheRightLengthScale()
    {
        // The quantity used to select hyperparameters, and the property that makes it usable: it is
        // not monotone in flexibility, so it has an interior maximum.
        var (x, y) = SineData(count: 24);

        double Score(double lengthScale)
            => new GaussianProcess(new RbfKernel(lengthScale), noise: 1e-4).Fit(x, y).LogMarginalLikelihood();

        var correct = Score(1.0);
        Assert.True(correct > Score(0.01), "a far too short length scale should score worse");
        Assert.True(correct > Score(100.0), "a far too long length scale should score worse");
    }

    [Fact]
    public void OptimiseFindsAModelThatFitsTheData()
    {
        var (x, y) = SineData(count: 20);
        var gp = GaussianProcess.Optimise(x, y);

        foreach (var probe in new[] { 0.8, 2.2, 4.1 })
        {
            var predicted = gp.PredictMean(Column(probe)).At(0);
            Assert.True(Math.Abs(predicted - Math.Sin(probe)) < 0.15,
                $"at {probe} the optimised process predicted {predicted:F4} against {Math.Sin(probe):F4}");
        }
    }

    [Fact]
    public void CredibleIntervalsBracketTheMean()
    {
        var (x, y) = SineData();
        var gp = new GaussianProcess(new RbfKernel(lengthScale: 1.0), noise: 0.01).Fit(x, y);

        var prediction = gp.Predict(Column(0.5, 10.0));
        var (lower, upper) = prediction.Interval(0.95);

        for (var i = 0; i < 2; i++)
        {
            Assert.True(lower.At(i) <= prediction.Mean.At(i));
            Assert.True(upper.At(i) >= prediction.Mean.At(i));
        }

        // The interval must be wider where the process is less certain.
        Assert.True(upper.At(1) - lower.At(1) > upper.At(0) - lower.At(0));
    }

    [Fact]
    public void PosteriorSamplesPassThroughTheObservations()
    {
        // Sampled functions are constrained by the data in a way a marginal band cannot express.
        var (x, y) = SineData(count: 8);
        var gp = new GaussianProcess(new RbfKernel(lengthScale: 1.0), noise: 1e-8).Fit(x, y);

        var draws = gp.SamplePosterior(x, 20, new GraviRandom(5));

        Assert.Equal([20, 8], draws.Shape.ToArray());

        for (var s = 0; s < 20; s++)
            for (var i = 0; i < 8; i++)
                Assert.True(Math.Abs(draws[s, i] - y.At(i)) < 1e-3,
                    $"draw {s} missed observation {i}: {draws[s, i]:F5} against {y.At(i):F5}");
    }

    [Fact]
    public void PosteriorSamplesVaryWhereTheDataDoesNotConstrainThem()
    {
        var (x, y) = SineData(count: 6);
        var gp = new GaussianProcess(new RbfKernel(lengthScale: 0.5), noise: 1e-6).Fit(x, y);

        var draws = gp.SamplePosterior(Column(50.0), 200, new GraviRandom(7));

        var mean = 0.0;
        for (var s = 0; s < 200; s++) mean += draws[s, 0];
        mean /= 200;

        var variance = 0.0;
        for (var s = 0; s < 200; s++) variance += (draws[s, 0] - mean) * (draws[s, 0] - mean);
        variance /= 200;

        // Far from every observation the spread must be roughly the prior variance.
        Assert.True(variance > 0.5, $"draws far from the data had variance {variance:F4}");
    }

    // ------------------------------------------------------------------ kernels

    [Fact]
    public void EveryKernelIsMaximalAtZeroDistance()
    {
        var origin = NdArray.FromValues([0.0]);
        var away = NdArray.FromValues([2.0]);

        Kernel[] kernels =
        [
            new RbfKernel(),
            new MaternKernel(0.5), new MaternKernel(1.5), new MaternKernel(2.5),
            new PeriodicKernel(period: 10.0),
        ];

        foreach (var kernel in kernels)
        {
            Assert.Equal(1.0, kernel.Evaluate(origin, origin), 10);
            Assert.True(kernel.Evaluate(origin, away) < kernel.Evaluate(origin, origin),
                $"{kernel.Name} did not fall off with distance");
        }
    }

    [Fact]
    public void KernelMatricesAreSymmetricAndPositiveDefinite()
    {
        // The requirement that makes a function a valid covariance at all. A kernel failing this
        // produces a covariance the Cholesky factorisation rejects.
        var x = NdArray.Zeros(8, 1);
        for (var i = 0; i < 8; i++) x[i, 0] = i * 0.7;

        Kernel[] kernels = [new RbfKernel(), new MaternKernel(1.5), new PeriodicKernel(period: 5.0)];

        foreach (var kernel in kernels)
        {
            var matrix = kernel.Matrix(x, x);

            for (var i = 0; i < 8; i++)
                for (var j = 0; j < 8; j++)
                    Assert.Equal(matrix[i, j], matrix[j, i], 12);

            for (var i = 0; i < 8; i++) matrix[i, i] += 1e-8;
            Decomposition.Cholesky(matrix);      // throws if it is not positive definite
        }
    }

    [Fact]
    public void APeriodicKernelIsMaximalOnePeriodApart()
    {
        // What separates it from every other kernel here: distance alone does not decide similarity.
        var kernel = new PeriodicKernel(period: 4.0, lengthScale: 1.0);

        var origin = NdArray.FromValues([0.0]);
        Assert.Equal(1.0, kernel.Evaluate(origin, NdArray.FromValues([4.0])), 10);
        Assert.True(kernel.Evaluate(origin, NdArray.FromValues([2.0])) < 0.5);
    }

    [Fact]
    public void MaternBecomesRougherAsNuFalls()
    {
        // The point of the family: nu controls smoothness, and at a fixed distance a rougher kernel
        // decorrelates faster.
        var a = NdArray.FromValues([0.0]);
        var b = NdArray.FromValues([0.3]);

        var rough = new MaternKernel(0.5).Evaluate(a, b);
        var medium = new MaternKernel(1.5).Evaluate(a, b);
        var smooth = new MaternKernel(2.5).Evaluate(a, b);

        Assert.True(rough < medium, $"nu=0.5 gave {rough:F5} against nu=1.5's {medium:F5}");
        Assert.True(medium < smooth, $"nu=1.5 gave {medium:F5} against nu=2.5's {smooth:F5}");
    }

    [Fact]
    public void SummingKernelsAddsTheirCovariances()
    {
        var a = new RbfKernel(lengthScale: 1.0);
        var b = new PeriodicKernel(period: 3.0);
        var sum = new SumKernel(a, b);

        var x = NdArray.FromValues([0.0]);
        var y = NdArray.FromValues([1.0]);

        Assert.Equal(a.Evaluate(x, y) + b.Evaluate(x, y), sum.Evaluate(x, y), 12);
    }

    [Fact]
    public void MalformedInputsAreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RbfKernel(lengthScale: 0));
        Assert.Throws<ArgumentException>(() => new MaternKernel(nu: 2.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GaussianProcess(noise: -1));

        var gp = new GaussianProcess();
        Assert.Throws<InvalidOperationException>(() => gp.Predict(Column(1.0)));
        Assert.Throws<ArgumentException>(() => gp.Fit(Column(1, 2), NdArray.FromValues([1.0])));
    }

    [Fact]
    public void DuplicateInputsWithoutNoiseFailLoudly()
    {
        // The covariance is singular, so the factorisation cannot succeed. Reporting that beats
        // returning a fit built on a factorisation that silently went wrong.
        var x = Column(1.0, 1.0, 1.0);
        var y = NdArray.FromValues([1.0, 2.0, 3.0]);

        var error = Assert.Throws<InvalidOperationException>(
            () => new GaussianProcess(new RbfKernel(), noise: 0.0).Fit(x, y));

        Assert.Contains("noise", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    // ------------------------------------------------------------------ helpers

    private static double Mean(NdArray y)
    {
        var total = 0.0;
        for (var i = 0; i < y.Size; i++) total += y.At(i);
        return total / y.Size;
    }

    private static NdArray Centred(NdArray y)
    {
        var mean = Mean(y);
        var result = NdArray.Zeros(y.Size);
        for (var i = 0; i < y.Size; i++) result.SetAt(i, y.At(i) - mean);
        return result;
    }
}
