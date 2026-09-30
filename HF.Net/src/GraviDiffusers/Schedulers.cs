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

/// <summary>How a scheduler spreads its inference steps over the training timesteps.</summary>
/// <remarks>diffusers' <c>timestep_spacing</c>, by the same names.</remarks>
public enum TimestepSpacing
{
    /// <summary><c>i * (T / steps)</c>, plus <c>steps_offset</c>: Stable Diffusion 1.x and 2.x.</summary>
    Leading,

    /// <summary>Evenly from <c>T - 1</c> down to 0, fractional where the division is not exact.</summary>
    Linspace,

    /// <summary>From <c>T - 1</c> down in steps of <c>T / steps</c>: newer schedulers and SDXL.</summary>
    Trailing,
}

/// <summary>What the model's output means.</summary>
/// <remarks>diffusers' <c>prediction_type</c>. Stable Diffusion 2.x at 768 px predicts <c>v</c>.</remarks>
public enum PredictionType
{
    /// <summary>The noise that was added.</summary>
    Epsilon,

    /// <summary>The velocity <c>sqrt(abar) noise - sqrt(1 - abar) x0</c>.</summary>
    VPrediction,

    /// <summary>The clean sample itself.</summary>
    Sample,
}

/// <summary>
/// A scheduler's configuration, as a diffusion repository's <c>scheduler/scheduler_config.json</c>
/// states it.
/// </summary>
/// <remarks>
/// <para>
/// Each field changes which timesteps a run visits or what one step computes, and the model's
/// weights were trained expecting particular values. Stable Diffusion 1.x ships
/// <c>steps_offset: 1</c> and <c>set_alpha_to_one: false</c>: without the first, every step is one
/// timestep early; without the second, the last step lands on a slightly different image.
/// </para>
/// <para>
/// The schedulers built from a configuration follow diffusers' <c>DDIMScheduler</c>,
/// <c>EulerDiscreteScheduler</c> and <c>DDPMScheduler</c> formula for formula, and the tests pin
/// them to diffusers' own output.
/// </para>
/// </remarks>
public sealed record SchedulerConfig
{
    /// <summary>How many steps the model was trained over.</summary>
    public int TrainTimesteps { get; init; } = 1000;

    /// <summary>Beta at the first training timestep.</summary>
    public double BetaStart { get; init; } = 0.00085;

    /// <summary>Beta at the last training timestep.</summary>
    public double BetaEnd { get; init; } = 0.012;

    /// <summary>How the betas are spaced.</summary>
    public BetaSchedule BetaSchedule { get; init; } = BetaSchedule.ScaledLinear;

    /// <summary>How inference steps are spread over the training timesteps.</summary>
    public TimestepSpacing Spacing { get; init; } = TimestepSpacing.Leading;

    /// <summary>Added to every <see cref="TimestepSpacing.Leading"/> timestep.</summary>
    public int StepsOffset { get; init; }

    /// <summary>
    /// Whether DDIM's step past the first training timestep uses a cumulative alpha of exactly 1,
    /// rather than the first timestep's.
    /// </summary>
    public bool SetAlphaToOne { get; init; } = true;

    /// <summary>What the model predicts.</summary>
    public PredictionType Prediction { get; init; } = PredictionType.Epsilon;

    /// <summary>Whether the reconstructed clean sample is clamped to <c>[-range, range]</c>.</summary>
    public bool ClipSample { get; init; }

    /// <summary>The clamp's bound.</summary>
    public double ClipSampleRange { get; init; } = 1.0;

    /// <summary>The diffusers class the file was written for, such as <c>EulerDiscreteScheduler</c>.</summary>
    public string? ClassName { get; init; }

    /// <summary>The noise schedule these betas describe.</summary>
    public NoiseSchedule NoiseSchedule => new(TrainTimesteps, BetaStart, BetaEnd, BetaSchedule);

    /// <summary>Stable Diffusion 1.x's configuration.</summary>
    public static SchedulerConfig StableDiffusion => new() { StepsOffset = 1, SetAlphaToOne = false };

    /// <summary>Reads a diffusers <c>scheduler_config.json</c>.</summary>
    /// <param name="path">The file.</param>
    /// <exception cref="NotSupportedException">It names a beta schedule, spacing or prediction type this does not implement.</exception>
    public static SchedulerConfig Load(string path)
    {
        using var document = System.Text.Json.JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;

        string? Text(string name) => root.TryGetProperty(name, out var value) && value.ValueKind == System.Text.Json.JsonValueKind.String ? value.GetString() : null;
        double? Number(string name) => root.TryGetProperty(name, out var value) && value.ValueKind == System.Text.Json.JsonValueKind.Number ? value.GetDouble() : null;
        bool? Flag(string name) => root.TryGetProperty(name, out var value) && value.ValueKind is System.Text.Json.JsonValueKind.True or System.Text.Json.JsonValueKind.False ? value.GetBoolean() : null;

        var defaults = new SchedulerConfig();
        return new SchedulerConfig
        {
            TrainTimesteps = (int)(Number("num_train_timesteps") ?? defaults.TrainTimesteps),
            BetaStart = Number("beta_start") ?? defaults.BetaStart,
            BetaEnd = Number("beta_end") ?? defaults.BetaEnd,
            BetaSchedule = Text("beta_schedule") switch
            {
                null or "scaled_linear" => BetaSchedule.ScaledLinear,
                "linear" => BetaSchedule.Linear,
                "squaredcos_cap_v2" => BetaSchedule.SquaredCosine,
                var other => throw new NotSupportedException($"beta_schedule '{other}' is not supported; scaled_linear, linear and squaredcos_cap_v2 are."),
            },
            Spacing = Text("timestep_spacing") switch
            {
                null or "leading" => TimestepSpacing.Leading,
                "linspace" => TimestepSpacing.Linspace,
                "trailing" => TimestepSpacing.Trailing,
                var other => throw new NotSupportedException($"timestep_spacing '{other}' is not supported; leading, linspace and trailing are."),
            },
            StepsOffset = (int)(Number("steps_offset") ?? 0),
            SetAlphaToOne = Flag("set_alpha_to_one") ?? defaults.SetAlphaToOne,
            Prediction = Text("prediction_type") switch
            {
                null or "epsilon" => PredictionType.Epsilon,
                "v_prediction" => PredictionType.VPrediction,
                "sample" => PredictionType.Sample,
                var other => throw new NotSupportedException($"prediction_type '{other}' is not supported; epsilon, v_prediction and sample are."),
            },
            ClipSample = Flag("clip_sample") ?? defaults.ClipSample,
            ClipSampleRange = Number("clip_sample_range") ?? defaults.ClipSampleRange,
            ClassName = Text("_class_name"),
        };
    }

    /// <summary>The timesteps diffusers visits for <paramref name="steps"/> inference steps.</summary>
    /// <param name="steps">How many inference steps.</param>
    /// <param name="round">Whether they are rounded to integers, as DDIM and DDPM do; Euler keeps them fractional.</param>
    internal double[] TimestepsFor(int steps, bool round)
    {
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps), "At least one step is needed.");

        var t = TrainTimesteps;
        var result = new double[steps];

        switch (Spacing)
        {
            case TimestepSpacing.Leading:
            {
                var ratio = t / steps;
                for (var i = 0; i < steps; i++) result[i] = Math.Round((double)(steps - 1 - i) * ratio) + StepsOffset;
                break;
            }

            case TimestepSpacing.Linspace:
                for (var i = 0; i < steps; i++)
                {
                    // numpy's linspace in float32 for Euler, exactly as it is computed there.
                    var value = steps == 1 ? 0 : (double)(float)((t - 1) * (double)(steps - 1 - i) / (steps - 1));
                    result[i] = round ? Math.Round(value, MidpointRounding.ToEven) : value;
                }

                break;

            case TimestepSpacing.Trailing:
            {
                var ratio = (double)t / steps;
                for (var i = 0; i < steps; i++) result[i] = Math.Round(t - i * ratio, MidpointRounding.ToEven) - 1;
                break;
            }
        }

        return result;
    }

    /// <summary>The clean sample and the noise a model output implies, at cumulative alpha <paramref name="alphaBar"/>.</summary>
    internal (double Original, double Noise) Decompose(double output, double sample, double alphaBar)
    {
        var a = Math.Sqrt(alphaBar);
        var b = Math.Sqrt(1 - alphaBar);

        var (original, noise) = Prediction switch
        {
            PredictionType.Epsilon => ((sample - b * output) / a, output),
            PredictionType.VPrediction => (a * sample - b * output, a * output + b * sample),
            _ => (output, (sample - a * output) / b),
        };

        if (ClipSample) original = Math.Clamp(original, -ClipSampleRange, ClipSampleRange);
        return (original, noise);
    }
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

    /// <summary>
    /// The timestep handed to the model at a step. Usually <see cref="Timesteps"/>' entry; Euler
    /// with linspace spacing visits fractional timesteps, which are rounded only in that list.
    /// </summary>
    double ModelTimestep(int stepIndex) => Timesteps[stepIndex];

    /// <summary>The configuration's <c>steps_offset</c>, which image-to-image uses to pick its first step.</summary>
    int StepsOffset => 0;

    /// <summary>
    /// Noises a clean sample to the level of a step: the forward process image-to-image starts
    /// from.
    /// </summary>
    /// <exception cref="NotSupportedException">The scheduler has no forward process.</exception>
    NdArray AddNoise(NdArray original, NdArray noise, int stepIndex)
        => throw new NotSupportedException($"{GetType().Name} does not implement AddNoise, which image-to-image needs.");
}

/// <summary>
/// DDIM: a deterministic reverse process that can skip timesteps.
/// </summary>
/// <remarks>
/// <para>
/// The reason to reach for DDIM over DDPM is that it is <i>non-Markovian</i> - it reconstructs an
/// estimate of the clean sample at every step and re-noises it to the next level, so it stays
/// consistent when the trajectory visits 50 of the 1000 training timesteps rather than all of them.
/// </para>
/// <para>
/// Built from a <see cref="SchedulerConfig"/>, this is diffusers' <c>DDIMScheduler</c>: the previous
/// timestep is <c>t - T / steps</c>, and past the first training timestep the cumulative alpha is 1
/// or the first timestep's, as <c>set_alpha_to_one</c> says.
/// </para>
/// </remarks>
public sealed class DdimScheduler : IScheduler
{
    private readonly NoiseSchedule _schedule;
    private readonly SchedulerConfig _config;
    private readonly double _eta;
    private readonly GraviRandom _random;
    private int[] _timesteps = [];

    /// <summary>A DDIM over a noise schedule, stepping to the clean sample at the end.</summary>
    /// <param name="schedule">The schedule; Stable Diffusion's when null.</param>
    /// <param name="eta">0 for the deterministic DDIM; 1 adds DDPM's noise back.</param>
    /// <param name="seed">Seeds the noise <paramref name="eta"/> adds.</param>
    public DdimScheduler(NoiseSchedule? schedule = null, double eta = 0.0, int seed = 42)
        : this(schedule ?? NoiseSchedule.StableDiffusion, new SchedulerConfig(), eta, seed)
    {
    }

    /// <summary>diffusers' <c>DDIMScheduler</c> with a repository's configuration.</summary>
    /// <param name="config">Usually <see cref="SchedulerConfig.Load"/> of <c>scheduler_config.json</c>.</param>
    /// <param name="eta">0 for the deterministic DDIM.</param>
    /// <param name="seed">Seeds the noise <paramref name="eta"/> adds.</param>
    public DdimScheduler(SchedulerConfig config, double eta = 0.0, int seed = 42)
        : this(config.NoiseSchedule, config, eta, seed)
    {
    }

    private DdimScheduler(NoiseSchedule schedule, SchedulerConfig config, double eta, int seed)
    {
        _schedule = schedule;
        _config = config;
        _eta = eta;
        _random = new GraviRandom(seed);
    }

    /// <inheritdoc />
    public IReadOnlyList<int> Timesteps => _timesteps;

    /// <inheritdoc />
    public double InitialNoiseScale => 1.0;

    /// <summary>The noise schedule being walked.</summary>
    public NoiseSchedule Schedule => _schedule;

    /// <summary>The configuration.</summary>
    public SchedulerConfig Config => _config;

    /// <inheritdoc />
    public int StepsOffset => _config.StepsOffset;

    /// <inheritdoc />
    public IReadOnlyList<int> SetTimesteps(int steps)
    {
        _timesteps = [.. _config.TimestepsFor(steps, round: true).Select(t => (int)t)];
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
        var previous = timestep - _schedule.TrainTimesteps / _timesteps.Length;

        var alphaBar = _schedule.AlphaBar(timestep);
        var alphaBarPrevious = previous >= 0
            ? _schedule.AlphaBar(previous)
            : _config.SetAlphaToOne ? 1.0 : _schedule.AlphasCumulative[0];

        // eta > 0 adds noise back and makes the walk stochastic; eta = 0 is the deterministic DDIM.
        var variance = (1 - alphaBarPrevious) / (1 - alphaBar) * (1 - alphaBar / alphaBarPrevious);
        var deviation = _eta * Math.Sqrt(variance);
        var direction = Math.Sqrt(Math.Max(0, 1 - alphaBarPrevious - deviation * deviation));
        var result = NdArray.Zeros(sample.Shape.ToArray());

        for (var i = 0; i < sample.Size; i++)
        {
            var (original, noise) = _config.Decompose(modelOutput.At(i), sample.At(i), alphaBar);
            var value = Math.Sqrt(alphaBarPrevious) * original + direction * noise;
            if (deviation > 0) value += deviation * _random.Normal();
            result.SetAt(i, value);
        }

        return result;
    }

    /// <inheritdoc />
    public NdArray AddNoise(NdArray original, NdArray noise, int stepIndex)
        => ForwardNoise(_schedule, _timesteps[stepIndex], original, noise);

    internal static NdArray ForwardNoise(NoiseSchedule schedule, int timestep, NdArray original, NdArray noise)
    {
        var alphaBar = schedule.AlphaBar(timestep);
        var a = Math.Sqrt(alphaBar);
        var b = Math.Sqrt(1 - alphaBar);
        var result = NdArray.Zeros(original.Shape.ToArray());
        for (var i = 0; i < original.Size; i++) result.SetAt(i, a * original.At(i) + b * noise.At(i));
        return result;
    }

    /// <inheritdoc />
    public override string ToString() => $"DdimScheduler(eta={_eta}, {_timesteps.Length} steps)";
}

/// <summary>DDPM: the original, Markovian reverse process.</summary>
/// <remarks>
/// Built from a <see cref="SchedulerConfig"/>, diffusers' <c>DDPMScheduler</c> with the
/// <c>fixed_small</c> variance. Accurate at hundreds of steps and poor at tens, which is why a
/// pipeline defaults to DDIM or Euler.
/// </remarks>
public sealed class DdpmScheduler : IScheduler
{
    private readonly NoiseSchedule _schedule;
    private readonly SchedulerConfig _config;
    private readonly GraviRandom _random;
    private int[] _timesteps = [];

    /// <summary>A DDPM over a noise schedule, with the reconstruction clamped to <c>[-1, 1]</c>.</summary>
    /// <param name="schedule">The schedule; the DDPM paper's when null.</param>
    /// <param name="seed">Seeds the noise each step adds.</param>
    public DdpmScheduler(NoiseSchedule? schedule = null, int seed = 42)
        : this(schedule ?? NoiseSchedule.Ddpm, new SchedulerConfig { Spacing = TimestepSpacing.Trailing, ClipSample = true }, seed)
    {
    }

    /// <summary>diffusers' <c>DDPMScheduler</c> with a repository's configuration.</summary>
    public DdpmScheduler(SchedulerConfig config, int seed = 42)
        : this(config.NoiseSchedule, config, seed)
    {
    }

    private DdpmScheduler(NoiseSchedule schedule, SchedulerConfig config, int seed)
    {
        _schedule = schedule;
        _config = config;
        _random = new GraviRandom(seed);
    }

    /// <inheritdoc />
    public IReadOnlyList<int> Timesteps => _timesteps;

    /// <inheritdoc />
    public double InitialNoiseScale => 1.0;

    /// <summary>The noise schedule being walked.</summary>
    public NoiseSchedule Schedule => _schedule;

    /// <inheritdoc />
    public int StepsOffset => _config.StepsOffset;

    /// <inheritdoc />
    public IReadOnlyList<int> SetTimesteps(int steps)
    {
        _timesteps = [.. _config.TimestepsFor(Math.Min(steps, _schedule.TrainTimesteps), round: true).Select(t => (int)t)];
        return _timesteps;
    }

    /// <inheritdoc />
    public NdArray ScaleInput(NdArray sample, int stepIndex) => sample;

    /// <inheritdoc />
    public NdArray Step(NdArray modelOutput, int stepIndex, NdArray sample)
    {
        var timestep = _timesteps[stepIndex];
        var previous = timestep - _schedule.TrainTimesteps / _timesteps.Length;

        var alphaBar = _schedule.AlphaBar(timestep);
        var alphaBarPrevious = previous >= 0 ? _schedule.AlphaBar(previous) : 1.0;
        var currentAlpha = alphaBar / alphaBarPrevious;
        var currentBeta = 1 - currentAlpha;

        var originalCoefficient = Math.Sqrt(alphaBarPrevious) * currentBeta / (1 - alphaBar);
        var currentCoefficient = Math.Sqrt(currentAlpha) * (1 - alphaBarPrevious) / (1 - alphaBar);

        var variance = Math.Max(1e-20, (1 - alphaBarPrevious) / (1 - alphaBar) * currentBeta);
        var deviation = timestep > 0 ? Math.Sqrt(variance) : 0;

        var result = NdArray.Zeros(sample.Shape.ToArray());
        for (var i = 0; i < sample.Size; i++)
        {
            // Clamping the reconstruction to the data range is what the reference does by
            // default, and it matters: an unclamped estimate early in the trajectory can be far
            // outside [-1, 1] and drag the whole mean with it.
            var (original, _) = _config.Decompose(modelOutput.At(i), sample.At(i), alphaBar);
            var mean = originalCoefficient * original + currentCoefficient * sample.At(i);
            result.SetAt(i, deviation > 0 && previous >= 0 ? mean + deviation * _random.Normal() : mean);
        }

        return result;
    }

    /// <inheritdoc />
    public NdArray AddNoise(NdArray original, NdArray noise, int stepIndex)
        => DdimScheduler.ForwardNoise(_schedule, _timesteps[stepIndex], original, noise);

    /// <inheritdoc />
    public override string ToString() => $"DdpmScheduler({_timesteps.Length} steps)";
}

/// <summary>
/// Euler: the sigma-parameterised sampler, walking <c>x + d * (sigma_next - sigma)</c>.
/// </summary>
/// <remarks>
/// <para>
/// It works in the variance-exploding parameterisation, so the model input has to be scaled down
/// by <c>sqrt(sigma^2 + 1)</c> first - <see cref="ScaleInput"/> - and the initial noise scaled up.
/// Forgetting either gives an image that is a smear rather than a picture.
/// </para>
/// <para>
/// Built from a <see cref="SchedulerConfig"/>, this is diffusers' <c>EulerDiscreteScheduler</c>:
/// linspace timesteps stay fractional and their sigmas are interpolated, and with leading spacing
/// the initial noise is <c>sqrt(sigma_max^2 + 1)</c> rather than <c>sigma_max</c>.
/// </para>
/// </remarks>
public sealed class EulerScheduler : IScheduler
{
    private readonly NoiseSchedule _schedule;
    private readonly SchedulerConfig _config;
    private double[] _modelTimesteps = [];
    private int[] _timesteps = [];
    private double[] _sigmas = [];

    /// <summary>An Euler sampler over a noise schedule, with linspace timesteps.</summary>
    /// <param name="schedule">The schedule; Stable Diffusion's when null.</param>
    public EulerScheduler(NoiseSchedule? schedule = null)
        : this(schedule ?? NoiseSchedule.StableDiffusion, new SchedulerConfig { Spacing = TimestepSpacing.Linspace })
    {
    }

    /// <summary>diffusers' <c>EulerDiscreteScheduler</c> with a repository's configuration.</summary>
    public EulerScheduler(SchedulerConfig config)
        : this(config.NoiseSchedule, config)
    {
    }

    private EulerScheduler(NoiseSchedule schedule, SchedulerConfig config)
    {
        _schedule = schedule;
        _config = config;
    }

    /// <inheritdoc />
    public IReadOnlyList<int> Timesteps => _timesteps;

    /// <summary>The noise level at each step, ending in 0.</summary>
    public IReadOnlyList<double> Sigmas => _sigmas;

    /// <inheritdoc />
    public double InitialNoiseScale
    {
        get
        {
            if (_sigmas.Length == 0) return 1.0;
            var largest = _sigmas.Max();
            return _config.Spacing == TimestepSpacing.Leading ? Math.Sqrt(largest * largest + 1) : largest;
        }
    }

    /// <inheritdoc />
    public int StepsOffset => _config.StepsOffset;

    /// <inheritdoc />
    public double ModelTimestep(int stepIndex) => _modelTimesteps[stepIndex];

    /// <inheritdoc />
    public IReadOnlyList<int> SetTimesteps(int steps)
    {
        _modelTimesteps = _config.TimestepsFor(steps, round: false);
        _timesteps = [.. _modelTimesteps.Select(t => (int)Math.Round(t))];

        // diffusers interpolates the training sigmas at each (possibly fractional) timestep.
        var training = new double[_schedule.TrainTimesteps];
        for (var t = 0; t < training.Length; t++) training[t] = _schedule.Sigma(t);

        _sigmas = new double[steps + 1];
        for (var i = 0; i < steps; i++)
        {
            var t = Math.Clamp(_modelTimesteps[i], 0, training.Length - 1);
            var low = (int)Math.Floor(t);
            var high = Math.Min(low + 1, training.Length - 1);
            _sigmas[i] = training[low] + (training[high] - training[low]) * (t - low);
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
            var output = modelOutput.At(i);
            var original = _config.Prediction switch
            {
                PredictionType.VPrediction => output * (-sigma / Math.Sqrt(sigma * sigma + 1)) + sample.At(i) / (sigma * sigma + 1),
                PredictionType.Sample => output,
                _ => sample.At(i) - sigma * output,
            };

            // The derivative is the direction from the clean estimate back to the sample.
            var derivative = (sample.At(i) - original) / sigma;
            result.SetAt(i, sample.At(i) + derivative * (next - sigma));
        }

        return result;
    }

    /// <inheritdoc />
    public NdArray AddNoise(NdArray original, NdArray noise, int stepIndex)
    {
        var sigma = _sigmas[stepIndex];
        var result = NdArray.Zeros(original.Shape.ToArray());
        for (var i = 0; i < original.Size; i++) result.SetAt(i, original.At(i) + sigma * noise.At(i));
        return result;
    }

    /// <inheritdoc />
    public override string ToString() => $"EulerScheduler({_timesteps.Length} steps)";
}
