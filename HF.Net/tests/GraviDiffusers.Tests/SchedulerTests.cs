using Gravicode.HFNet.GraviDiffusers;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.HFNet.GraviDiffusers.Tests;

/// <summary>
/// Tests for the noise schedule.
/// </summary>
/// <remarks>
/// The schedule is checked against properties that follow from its definition - monotonicity, the
/// endpoints, the relationship between betas and the cumulative alphas - rather than against a
/// stored table. A table would pass even if the betas were being generated for the wrong schedule.
/// </remarks>
public sealed class NoiseScheduleTests
{
    [Fact]
    public void AlphasCumulativeIsTheRunningProductOfOneMinusBeta()
    {
        var schedule = new NoiseSchedule(50);

        var product = 1.0;
        for (var t = 0; t < schedule.TrainTimesteps; t++)
        {
            product *= 1 - schedule.Betas[t];
            Assert.True(Math.Abs(product - schedule.AlphasCumulative[t]) < 1e-12,
                $"at t={t}: {product} vs {schedule.AlphasCumulative[t]}");
        }
    }

    [Fact]
    public void SignalDecaysMonotonically()
    {
        var schedule = NoiseSchedule.StableDiffusion;

        for (var t = 1; t < schedule.TrainTimesteps; t++)
        {
            Assert.True(schedule.AlphasCumulative[t] < schedule.AlphasCumulative[t - 1],
                $"alpha bar increased at t={t}");
        }
    }

    [Fact]
    public void AlphaBarBeforeTheFirstStepIsExactlyOne()
    {
        // This is what makes the last reverse step land on a clean sample.
        Assert.Equal(1.0, NoiseSchedule.StableDiffusion.AlphaBar(-1));
    }

    [Fact]
    public void ScaledLinearUsesTheSquareOfALinearRampInSqrtBeta()
    {
        var schedule = new NoiseSchedule(1000, 0.00085, 0.012, BetaSchedule.ScaledLinear);

        Assert.True(Math.Abs(schedule.Betas[0] - 0.00085) < 1e-12);
        Assert.True(Math.Abs(schedule.Betas[^1] - 0.012) < 1e-12);

        // Halfway through, sqrt(beta) is halfway between the endpoints - which is not where beta
        // itself would be under a plain linear schedule.
        var expectedMiddle = Math.Pow((Math.Sqrt(0.00085) + Math.Sqrt(0.012)) / 2, 2);
        var linearMiddle = (0.00085 + 0.012) / 2;

        Assert.True(Math.Abs(schedule.Betas[499] + schedule.Betas[500] - 2 * expectedMiddle) < 1e-5);
        Assert.True(Math.Abs(expectedMiddle - linearMiddle) > 1e-4, "the two schedules must differ");
    }

    [Fact]
    public void LinearSpansItsEndpoints()
    {
        var schedule = new NoiseSchedule(1000, 1e-4, 0.02, BetaSchedule.Linear);

        Assert.True(Math.Abs(schedule.Betas[0] - 1e-4) < 1e-15);
        Assert.True(Math.Abs(schedule.Betas[^1] - 0.02) < 1e-15);
    }

    [Fact]
    public void CosineBetasStayBelowTheClip()
    {
        var schedule = new NoiseSchedule(1000, schedule: BetaSchedule.SquaredCosine);

        Assert.All(schedule.Betas, beta => Assert.True(beta is > 0 and <= 0.999, $"beta was {beta}"));
    }

    [Fact]
    public void SigmaGrowsWithTheTimestep()
    {
        var schedule = NoiseSchedule.StableDiffusion;

        Assert.True(schedule.Sigma(0) < schedule.Sigma(500));
        Assert.True(schedule.Sigma(500) < schedule.Sigma(999));
    }
}

/// <summary>Tests for the samplers.</summary>
public sealed class SchedulerTests
{
    /// <summary>
    /// A stand-in for a trained model that always predicts the same noise.
    /// </summary>
    /// <remarks>
    /// A constant prediction is not realistic, but it makes every step's arithmetic checkable by
    /// hand and it is enough to expose a wrong coefficient, a reversed direction or a scheduler
    /// that fails to converge.
    /// </remarks>
    private static NdArray ConstantNoise(NdArray sample, double value = 0.1)
    {
        var output = NdArray.Zeros(sample.Shape.ToArray());
        for (var i = 0; i < output.Size; i++) output.SetAt(i, value);
        return output;
    }

    [Fact]
    public void DdimVisitsTimestepsFromNoisiestToCleanest()
    {
        var timesteps = new DdimScheduler().SetTimesteps(10);

        Assert.Equal(10, timesteps.Count);
        Assert.True(timesteps[0] > timesteps[^1]);

        for (var i = 1; i < timesteps.Count; i++)
        {
            Assert.True(timesteps[i] < timesteps[i - 1], $"timesteps not descending at {i}");
        }

        Assert.Equal(0, timesteps[^1]);
    }

    [Fact]
    public void DdimWithEtaZeroIsDeterministic()
    {
        NdArray Run()
        {
            var scheduler = new DdimScheduler(eta: 0);
            scheduler.SetTimesteps(8);

            var sample = DiffusionPipeline.InitialLatents(4, 4, seed: 7, scale: 1.0);
            for (var step = 0; step < 8; step++)
            {
                sample = scheduler.Step(ConstantNoise(sample), step, sample);
            }
            return sample;
        }

        Assert.Equal(Run().ToArray(), Run().ToArray());
    }

    [Fact]
    public void DdimWithPositiveEtaIsNotDeterministic()
    {
        NdArray Run(int seed)
        {
            var scheduler = new DdimScheduler(eta: 1.0, seed: seed);
            scheduler.SetTimesteps(8);

            var sample = DiffusionPipeline.InitialLatents(4, 4, seed: 7, scale: 1.0);
            for (var step = 0; step < 8; step++)
            {
                sample = scheduler.Step(ConstantNoise(sample), step, sample);
            }
            return sample;
        }

        Assert.NotEqual(Run(1).ToArray(), Run(2).ToArray());
    }

    [Fact]
    public void DdimWithAPerfectNoiseOracleRecoversTheOriginalExactly()
    {
        // The sharpest available check on the coefficients. Build x_t from a known x_0 and a known
        // noise, then let the "model" return that exact noise at every step. DDIM's reconstruction
        // is then exact at each step, so the trajectory must land back on x_0 - and it does so only
        // if sqrt(abar), sqrt(1 - abar) and the direction term are all in their right places.
        var schedule = NoiseSchedule.StableDiffusion;
        var scheduler = new DdimScheduler(schedule, eta: 0);
        var timesteps = scheduler.SetTimesteps(25);

        var original = DiffusionPipeline.InitialLatents(4, 4, seed: 5, scale: 1.0);
        var noise = DiffusionPipeline.InitialLatents(4, 4, seed: 99, scale: 1.0);

        var alphaBar = schedule.AlphaBar(timesteps[0]);
        var sample = NdArray.Zeros(original.Shape.ToArray());

        for (var i = 0; i < sample.Size; i++)
        {
            sample.SetAt(i, Math.Sqrt(alphaBar) * original.At(i) + Math.Sqrt(1 - alphaBar) * noise.At(i));
        }

        for (var step = 0; step < timesteps.Count; step++)
        {
            sample = scheduler.Step(noise, step, sample);
        }

        for (var i = 0; i < sample.Size; i++)
        {
            Assert.True(Math.Abs(sample.At(i) - original.At(i)) < 1e-9,
                $"element {i}: recovered {sample.At(i)}, expected {original.At(i)}");
        }
    }

    [Fact]
    public void AConstantNoisePredictionStaysFinite()
    {
        // A constant prediction is not a contraction - DDIM amplifies by sqrt(abar_0 / abar_T)
        // along the way, which is roughly fifteen-fold on this schedule. What must hold is that
        // nothing overflows or turns into a NaN.
        var scheduler = new DdimScheduler();
        scheduler.SetTimesteps(20);

        var sample = DiffusionPipeline.InitialLatents(8, 8, seed: 3, scale: 1.0);

        for (var step = 0; step < 20; step++)
        {
            sample = scheduler.Step(ConstantNoise(sample), step, sample);
            Assert.All(sample.ToArray(), v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
        }
    }

    [Fact]
    public void EulerScalesTheInputByTheNoiseLevel()
    {
        var scheduler = new EulerScheduler();
        scheduler.SetTimesteps(10);

        var sample = NdArray.Ones(1, 4, 2, 2);
        var scaled = scheduler.ScaleInput(sample, 0);

        var sigma = scheduler.Sigmas[0];
        var expected = 1.0 / Math.Sqrt(sigma * sigma + 1);

        Assert.True(Math.Abs(scaled.At(0) - expected) < 1e-12);
    }

    [Fact]
    public void EulerEndsAtZeroNoise()
    {
        var scheduler = new EulerScheduler();
        scheduler.SetTimesteps(12);

        Assert.Equal(13, scheduler.Sigmas.Count);
        Assert.Equal(0.0, scheduler.Sigmas[^1]);
    }

    [Fact]
    public void EulerInitialNoiseScaleIsTheLargestSigma()
    {
        var scheduler = new EulerScheduler();
        scheduler.SetTimesteps(10);

        Assert.Equal(scheduler.Sigmas[0], scheduler.InitialNoiseScale);
        Assert.True(scheduler.InitialNoiseScale > 1);
    }

    [Fact]
    public void DdpmClampsItsReconstructionIntoTheDataRange()
    {
        // Feeding an implausible noise prediction would send an unclamped reconstruction far out of
        // range; the clamp is what keeps the mean finite.
        var scheduler = new DdpmScheduler();
        scheduler.SetTimesteps(10);

        var sample = NdArray.Full(50.0, 1, 4, 2, 2);
        var next = scheduler.Step(ConstantNoise(sample, 10.0), 0, sample);

        Assert.All(next.ToArray(), v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
    }

    [Fact]
    public void EveryStepPreservesShape()
    {
        foreach (IScheduler scheduler in (IScheduler[])
            [new DdimScheduler(), new DdpmScheduler(), new EulerScheduler()])
        {
            scheduler.SetTimesteps(5);

            var sample = NdArray.Zeros(1, 4, 8, 8);
            var next = scheduler.Step(ConstantNoise(sample), 0, sample);

            Assert.Equal(sample.Shape.ToArray(), next.Shape.ToArray());
        }
    }

}

/// <summary>Tests for the image conversion at the end of a pipeline.</summary>
public sealed class ImageConversionTests
{
    [Fact]
    public void MapsMinusOneToBlackAndOneToWhite()
    {
        var tensor = NdArray.Zeros(1, 3, 1, 2);

        // Pixel 0 is -1 in every channel, pixel 1 is +1.
        for (var channel = 0; channel < 3; channel++)
        {
            tensor[0, channel, 0, 0] = -1;
            tensor[0, channel, 0, 1] = 1;
        }

        using var image = DiffusionPipeline.ToImage(tensor);

        Assert.Equal(2, image.Width);
        Assert.Equal(1, image.Height);
        Assert.Equal(0, image[0, 0].R);
        Assert.Equal(255, image[1, 0].R);
    }

    [Fact]
    public void ClampsValuesOutsideTheExpectedRange()
    {
        var tensor = NdArray.Full(5.0, 1, 3, 1, 1);
        using var image = DiffusionPipeline.ToImage(tensor);

        Assert.Equal(255, image[0, 0].R);
    }

    [Fact]
    public void RejectsATensorThatIsNotAnImage()
    {
        var exception = Assert.Throws<InvalidDataException>(
            () => DiffusionPipeline.ToImage(NdArray.Zeros(1, 4, 8, 8)));

        Assert.Contains("[1, 3, H, W]", exception.Message);
    }

    [Fact]
    public void InitialLatentsAreReproducibleFromASeed()
    {
        var first = DiffusionPipeline.InitialLatents(8, 8, seed: 11, scale: 1.0);
        var second = DiffusionPipeline.InitialLatents(8, 8, seed: 11, scale: 1.0);
        var other = DiffusionPipeline.InitialLatents(8, 8, seed: 12, scale: 1.0);

        Assert.Equal(first.ToArray(), second.ToArray());
        Assert.NotEqual(first.ToArray(), other.ToArray());
    }

    [Fact]
    public void InitialLatentsHaveFourChannelsAtTheLatentResolution()
    {
        var latents = DiffusionPipeline.InitialLatents(64, 64, seed: 1, scale: 1.0);

        Assert.Equal([1, 4, 64, 64], latents.Shape.ToArray());
    }
}
