using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviDiffusers;

/// <summary>How the noise schedule's betas are laid out over training timesteps.</summary>
public enum BetaSchedule
{
    /// <summary>Betas spaced linearly from start to end.</summary>
    Linear,

    /// <summary>
    /// The square of a linear ramp in <c>sqrt(beta)</c>, which is what Stable Diffusion trains with.
    /// </summary>
    ScaledLinear,

    /// <summary>The cosine schedule, which adds noise more gently at both ends.</summary>
    SquaredCosine,
}

/// <summary>
/// The forward noise process a diffusion model was trained against: betas, alphas, and their
/// cumulative products.
/// </summary>
/// <remarks>
/// <para>
/// Every sampler is a different way of walking this schedule backwards, so the schedule has to
/// match the one the weights were trained with or nothing else matters. Stable Diffusion uses
/// <see cref="BetaSchedule.ScaledLinear"/> with <c>beta_start = 0.00085</c> and
/// <c>beta_end = 0.012</c>; using plain linear betas with those endpoints produces images that are
/// recognisably structured and consistently washed out, which is easy to mistake for a bad prompt.
/// </para>
/// </remarks>
public sealed class NoiseSchedule
{
    /// <summary>Builds a schedule.</summary>
    /// <param name="trainTimesteps">How many steps the model was trained over.</param>
    /// <param name="betaStart">Beta at the first timestep.</param>
    /// <param name="betaEnd">Beta at the last timestep.</param>
    /// <param name="schedule">How the betas are spaced.</param>
    public NoiseSchedule(
        int trainTimesteps = 1000,
        double betaStart = 0.00085,
        double betaEnd = 0.012,
        BetaSchedule schedule = BetaSchedule.ScaledLinear)
    {
        if (trainTimesteps < 1) throw new ArgumentOutOfRangeException(nameof(trainTimesteps));

        TrainTimesteps = trainTimesteps;
        Schedule = schedule;

        Betas = new double[trainTimesteps];
        Alphas = new double[trainTimesteps];
        AlphasCumulative = new double[trainTimesteps];

        switch (schedule)
        {
            case BetaSchedule.Linear:
                for (var t = 0; t < trainTimesteps; t++)
                {
                    Betas[t] = betaStart + (betaEnd - betaStart) * t / (trainTimesteps - 1.0);
                }
                break;

            case BetaSchedule.ScaledLinear:
            {
                var lower = Math.Sqrt(betaStart);
                var upper = Math.Sqrt(betaEnd);
                for (var t = 0; t < trainTimesteps; t++)
                {
                    var value = lower + (upper - lower) * t / (trainTimesteps - 1.0);
                    Betas[t] = value * value;
                }
                break;
            }

            case BetaSchedule.SquaredCosine:
            {
                // Betas are derived from the cumulative product rather than the other way round,
                // and clipped: without the clip the last few betas approach 1 and the reverse step
                // divides by something arbitrarily close to zero.
                for (var t = 0; t < trainTimesteps; t++)
                {
                    var t1 = (double)t / trainTimesteps;
                    var t2 = (t + 1.0) / trainTimesteps;
                    Betas[t] = Math.Min(1 - CosineBar(t2) / CosineBar(t1), 0.999);
                }
                break;
            }
        }

        var product = 1.0;
        for (var t = 0; t < trainTimesteps; t++)
        {
            Alphas[t] = 1 - Betas[t];
            product *= Alphas[t];
            AlphasCumulative[t] = product;
        }
    }

    /// <summary>How many steps the model was trained over.</summary>
    public int TrainTimesteps { get; }

    /// <summary>The beta spacing used.</summary>
    public BetaSchedule Schedule { get; }

    /// <summary>Per-step noise variance.</summary>
    public double[] Betas { get; }

    /// <summary>Per-step signal retention, <c>1 - beta</c>.</summary>
    public double[] Alphas { get; }

    /// <summary>Cumulative product of the alphas: the signal left after <c>t</c> steps.</summary>
    public double[] AlphasCumulative { get; }

    /// <summary>The cumulative alpha at a timestep, or 1 for the step before the first.</summary>
    /// <remarks>
    /// Index -1 returning exactly 1 is what makes the final reverse step land on the clean image
    /// rather than on a slightly noisy one.
    /// </remarks>
    public double AlphaBar(int timestep)
        => timestep < 0 ? 1.0 : AlphasCumulative[Math.Min(timestep, TrainTimesteps - 1)];

    /// <summary>The noise level <c>sqrt((1 - abar) / abar)</c> a sigma-parameterised sampler uses.</summary>
    public double Sigma(int timestep)
    {
        var alphaBar = AlphaBar(timestep);
        return Math.Sqrt((1 - alphaBar) / alphaBar);
    }

    /// <summary>The Stable Diffusion schedule.</summary>
    public static NoiseSchedule StableDiffusion => new();

    /// <summary>The schedule from the original DDPM paper.</summary>
    public static NoiseSchedule Ddpm => new(1000, 1e-4, 0.02, BetaSchedule.Linear);

    private static double CosineBar(double t)
    {
        var value = Math.Cos((t + 0.008) / 1.008 * Math.PI / 2);
        return value * value;
    }

    /// <inheritdoc />
    public override string ToString()
        => $"NoiseSchedule({Schedule}, {TrainTimesteps} steps, abar[-1]={AlphasCumulative[^1]:E3})";
}

/// <summary>Walks a <see cref="NoiseSchedule"/> backwards, turning noise into a sample.</summary>
public interface IScheduler
{
    /// <summary>Chooses the timesteps a run will visit, from noisiest to cleanest.</summary>
    /// <param name="steps">How many denoising steps to take.</param>
    IReadOnlyList<int> SetTimesteps(int steps);

    /// <summary>The timesteps the last call to <see cref="SetTimesteps"/> produced.</summary>
    IReadOnlyList<int> Timesteps { get; }

    /// <summary>
    /// Scales a latent before it is handed to the model, where the parameterisation needs it.
    /// </summary>
    /// <param name="sample">The current latent.</param>
    /// <param name="stepIndex">Which entry of <see cref="Timesteps"/> is being taken.</param>
    NdArray ScaleInput(NdArray sample, int stepIndex);

    /// <summary>Takes one reverse step.</summary>
    /// <param name="modelOutput">The noise the model predicted for this latent.</param>
    /// <param name="stepIndex">Which entry of <see cref="Timesteps"/> is being taken.</param>
    /// <param name="sample">The current latent.</param>
    /// <returns>The latent for the next step.</returns>
    NdArray Step(NdArray modelOutput, int stepIndex, NdArray sample);

    /// <summary>The factor initial noise is multiplied by before the first step.</summary>
    double InitialNoiseScale { get; }
}

/// <summary>
/// DDIM: a deterministic reverse process that can skip timesteps.
/// </summary>
/// <remarks>
/// <para>
/// The reason to reach for DDIM over DDPM is that it is <i>non-Markovian</i> - it reconstructs an
/// estimate of the clean sample at every step and re-noises it to the next level, so it stays
/// consistent when the trajectory visits 50 of the 1000 training timesteps rather than all of them.
/// DDPM's update assumes adjacent steps and degrades badly when they are skipped.
/// </para>
/// <para>
/// With <c>eta = 0</c> it is fully deterministic: the same seed and prompt give the same image
/// every time, which is what makes a generation reproducible.
/// </para>
/// </remarks>
public sealed class DdimScheduler(NoiseSchedule? schedule = null, double eta = 0.0, int seed = 42) : IScheduler
{
    private readonly NoiseSchedule _schedule = schedule ?? NoiseSchedule.StableDiffusion;
    private readonly GraviRandom _random = new(seed);
    private int[] _timesteps = [];

    /// <inheritdoc />
    public IReadOnlyList<int> Timesteps => _timesteps;

    /// <inheritdoc />
    public double InitialNoiseScale => 1.0;

    /// <summary>The schedule this sampler walks.</summary>
    public NoiseSchedule Schedule => _schedule;

    /// <inheritdoc />
    public IReadOnlyList<int> SetTimesteps(int steps)
    {
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps));

        var stride = _schedule.TrainTimesteps / steps;
        _timesteps = new int[steps];

        for (var i = 0; i < steps; i++) _timesteps[i] = (steps - 1 - i) * stride;
        return _timesteps;
    }

    /// <inheritdoc />
    public NdArray ScaleInput(NdArray sample, int stepIndex) => sample;

    /// <inheritdoc />
    public NdArray Step(NdArray modelOutput, int stepIndex, NdArray sample)
    {
        ArgumentNullException.ThrowIfNull(modelOutput);
        ArgumentNullException.ThrowIfNull(sample);

        var timestep = _timesteps[stepIndex];
        var previous = stepIndex + 1 < _timesteps.Length ? _timesteps[stepIndex + 1] : -1;

        var alphaBar = _schedule.AlphaBar(timestep);
        var alphaBarPrevious = _schedule.AlphaBar(previous);

        var sqrtAlphaBar = Math.Sqrt(alphaBar);
        var sqrtOneMinus = Math.Sqrt(1 - alphaBar);

        // eta > 0 adds noise back and makes the walk stochastic; eta = 0 is the deterministic DDIM.
        var variance = eta * Math.Sqrt((1 - alphaBarPrevious) / (1 - alphaBar) * (1 - alphaBar / alphaBarPrevious));
        var direction = Math.Sqrt(Math.Max(0, 1 - alphaBarPrevious - variance * variance));

        var result = NdArray.Zeros(sample.Shape.ToArray());

        for (var i = 0; i < sample.Size; i++)
        {
            var noise = modelOutput.At(i);
            var predictedOriginal = (sample.At(i) - sqrtOneMinus * noise) / sqrtAlphaBar;
            var value = Math.Sqrt(alphaBarPrevious) * predictedOriginal + direction * noise;

            if (variance > 0) value += variance * _random.Normal();

            result.SetAt(i, value);
        }

        return result;
    }

    /// <inheritdoc />
    public override string ToString() => $"DdimScheduler(eta={eta}, {_timesteps.Length} steps)";
}

/// <summary>
/// DDPM: the original reverse process, one step per training timestep.
/// </summary>
/// <remarks>
/// Faithful and slow. It is here as the reference the faster samplers are checked against: on a
/// full 1000-step trajectory every sampler should agree, and where one does not, this is the one
/// that is right.
/// </remarks>
public sealed class DdpmScheduler(NoiseSchedule? schedule = null, int seed = 42) : IScheduler
{
    private readonly NoiseSchedule _schedule = schedule ?? NoiseSchedule.Ddpm;
    private readonly GraviRandom _random = new(seed);
    private int[] _timesteps = [];

    /// <inheritdoc />
    public IReadOnlyList<int> Timesteps => _timesteps;

    /// <inheritdoc />
    public double InitialNoiseScale => 1.0;

    /// <summary>The schedule this sampler walks.</summary>
    public NoiseSchedule Schedule => _schedule;

    /// <inheritdoc />
    public IReadOnlyList<int> SetTimesteps(int steps)
    {
        var stride = Math.Max(1, _schedule.TrainTimesteps / steps);
        var list = new List<int>();

        for (var t = _schedule.TrainTimesteps - 1; t >= 0; t -= stride) list.Add(t);

        _timesteps = [.. list];
        return _timesteps;
    }

    /// <inheritdoc />
    public NdArray ScaleInput(NdArray sample, int stepIndex) => sample;

    /// <inheritdoc />
    public NdArray Step(NdArray modelOutput, int stepIndex, NdArray sample)
    {
        var timestep = _timesteps[stepIndex];
        var previous = stepIndex + 1 < _timesteps.Length ? _timesteps[stepIndex + 1] : -1;

        var alphaBar = _schedule.AlphaBar(timestep);
        var alphaBarPrevious = _schedule.AlphaBar(previous);
        var currentAlpha = alphaBar / alphaBarPrevious;
        var currentBeta = 1 - currentAlpha;

        var sqrtAlphaBar = Math.Sqrt(alphaBar);
        var sqrtOneMinus = Math.Sqrt(1 - alphaBar);

        var originalCoefficient = Math.Sqrt(alphaBarPrevious) * currentBeta / (1 - alphaBar);
        var currentCoefficient = Math.Sqrt(currentAlpha) * (1 - alphaBarPrevious) / (1 - alphaBar);

        var variance = previous < 0 ? 0 : currentBeta * (1 - alphaBarPrevious) / (1 - alphaBar);
        var deviation = Math.Sqrt(Math.Max(0, variance));

        var result = NdArray.Zeros(sample.Shape.ToArray());

        for (var i = 0; i < sample.Size; i++)
        {
            var predictedOriginal = (sample.At(i) - sqrtOneMinus * modelOutput.At(i)) / sqrtAlphaBar;

            // Clamping the reconstruction to the data range is what the reference implementation
            // does and it matters: an unclamped estimate early in the trajectory can be far outside
            // [-1, 1] and drags the whole mean with it.
            predictedOriginal = Math.Clamp(predictedOriginal, -1, 1);

            var mean = originalCoefficient * predictedOriginal + currentCoefficient * sample.At(i);
            result.SetAt(i, deviation > 0 ? mean + deviation * _random.Normal() : mean);
        }

        return result;
    }

    /// <inheritdoc />
    public override string ToString() => $"DdpmScheduler({_timesteps.Length} steps)";
}

/// <summary>
/// Euler discrete sampling over the sigma parameterisation.
/// </summary>
/// <remarks>
/// The reverse process is an ordinary differential equation in the noise level, and this takes
/// plain Euler steps along it. That framing is why it produces usable images in twenty or thirty
/// steps where DDPM needs hundreds - the step count is an integration accuracy choice rather than a
/// property of the trained schedule.
/// </remarks>
public sealed class EulerScheduler(NoiseSchedule? schedule = null) : IScheduler
{
    private readonly NoiseSchedule _schedule = schedule ?? NoiseSchedule.StableDiffusion;
    private int[] _timesteps = [];
    private double[] _sigmas = [];

    /// <inheritdoc />
    public IReadOnlyList<int> Timesteps => _timesteps;

    /// <summary>The noise levels visited, one per step plus a trailing zero.</summary>
    public IReadOnlyList<double> Sigmas => _sigmas;

    /// <inheritdoc />
    /// <remarks>
    /// The initial latent is scaled by the largest sigma, because this parameterisation expects the
    /// sample to carry the noise level in its magnitude rather than in a separate variable.
    /// </remarks>
    public double InitialNoiseScale => _sigmas.Length > 0 ? _sigmas[0] : 1.0;

    /// <inheritdoc />
    public IReadOnlyList<int> SetTimesteps(int steps)
    {
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps));

        _timesteps = new int[steps];
        _sigmas = new double[steps + 1];

        for (var i = 0; i < steps; i++)
        {
            var t = (int)Math.Round((_schedule.TrainTimesteps - 1) * (1.0 - (double)i / Math.Max(1, steps - 1)));
            _timesteps[i] = t;
            _sigmas[i] = _schedule.Sigma(t);
        }

        _sigmas[steps] = 0;
        return _timesteps;
    }

    /// <inheritdoc />
    public NdArray ScaleInput(NdArray sample, int stepIndex)
    {
        var sigma = _sigmas[stepIndex];
        var factor = 1.0 / Math.Sqrt(sigma * sigma + 1);

        var result = NdArray.Zeros(sample.Shape.ToArray());
        for (var i = 0; i < sample.Size; i++) result.SetAt(i, sample.At(i) * factor);

        return result;
    }

    /// <inheritdoc />
    public NdArray Step(NdArray modelOutput, int stepIndex, NdArray sample)
    {
        var sigma = _sigmas[stepIndex];
        var next = _sigmas[stepIndex + 1];

        var result = NdArray.Zeros(sample.Shape.ToArray());

        for (var i = 0; i < sample.Size; i++)
        {
            // The model predicts the noise, so the clean estimate is the sample minus it scaled by
            // sigma; the derivative is then the direction from that estimate back to the sample.
            var predictedOriginal = sample.At(i) - sigma * modelOutput.At(i);
            var derivative = (sample.At(i) - predictedOriginal) / sigma;

            result.SetAt(i, sample.At(i) + derivative * (next - sigma));
        }

        return result;
    }

    /// <inheritdoc />
    public override string ToString() => $"EulerScheduler({_timesteps.Length} steps)";
}
