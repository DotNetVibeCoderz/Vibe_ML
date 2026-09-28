using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviPEFT;

/// <summary>How <see cref="PeftModel.Train"/> fits adapters and a head.</summary>
/// <remarks>
/// The defaults are the Hugging Face <c>Trainer</c>'s where it has one - AdamW with betas 0.9 and
/// 0.999, epsilon 1e-8, a linear schedule, gradient norm clipped at 1 - and a learning rate in the
/// range PEFT's own examples use for LoRA, which is roughly ten times what full fine-tuning uses.
/// A record class rather than a record struct, so that <c>new TrainingOptions()</c> means these
/// defaults and not a zeroed configuration.
/// </remarks>
public sealed record TrainingOptions
{
    /// <summary>Passes over the training data.</summary>
    public int Epochs { get; init; } = 3;

    /// <summary>Examples per micro-batch; the loss is their mean.</summary>
    public int BatchSize { get; init; } = 8;

    /// <summary>Micro-batches whose gradients are summed before each optimizer step.</summary>
    public int GradientAccumulation { get; init; } = 1;

    /// <summary>Peak learning rate, reached at the end of warm-up.</summary>
    public double LearningRate { get; init; } = 5e-4;

    /// <summary>Decoupled weight decay, AdamW-style. Applied to adapters and head weights, not to biases.</summary>
    public double WeightDecay { get; init; } = 0.0;

    /// <summary>Fraction of the optimizer steps spent ramping the learning rate up from zero.</summary>
    public double WarmupFraction { get; init; } = 0.0;

    /// <summary>Global gradient norm is clipped to this; 0 or less turns clipping off.</summary>
    public double MaxGradientNorm { get; init; } = 1.0;

    /// <summary>Truncation limit in tokens.</summary>
    public int MaxLength { get; init; } = 128;

    /// <summary>Seeds shuffling, the head's initialisation and adapter dropout.</summary>
    public int Seed { get; init; } = 42;

    /// <summary>Receives one report per optimizer step.</summary>
    public IProgress<TrainingProgress>? Progress { get; init; }
}

/// <summary>One extractive question answering example, in SQuAD's shape.</summary>
/// <param name="Question">What is asked.</param>
/// <param name="Context">The passage that answers it.</param>
/// <param name="Answer">The answer, exactly as it appears in <paramref name="Context"/>.</param>
/// <param name="AnswerStart">
/// Index of the answer's first character in the context, SQuAD's <c>answer_start</c>; -1 means its
/// first occurrence.
/// </param>
public sealed record AnswerExample(string Question, string Context, string Answer, int AnswerStart = -1);

/// <summary>Where training has got to.</summary>
/// <param name="Epoch">The epoch, from 1.</param>
/// <param name="Step">Optimizer steps taken so far.</param>
/// <param name="TotalSteps">Optimizer steps the run will take.</param>
/// <param name="Loss">Mean cross-entropy over the examples of this step.</param>
/// <param name="LearningRate">The rate this step was taken with.</param>
public readonly record struct TrainingProgress(int Epoch, int Step, int TotalSteps, double Loss, double LearningRate);

/// <summary>What a training run did.</summary>
/// <param name="StepLosses">Mean loss of every optimizer step, in order.</param>
/// <param name="EpochLosses">Mean loss of every epoch.</param>
/// <param name="Steps">Optimizer steps taken.</param>
/// <param name="Elapsed">Wall-clock time.</param>
public sealed record TrainingReport(
    IReadOnlyList<double> StepLosses, IReadOnlyList<double> EpochLosses, int Steps, TimeSpan Elapsed)
{
    /// <inheritdoc />
    public override string ToString()
        => $"{Steps} steps in {Elapsed.TotalSeconds:F1} s, loss "
            + string.Join(" -> ", EpochLosses.Select(l => l.ToString("F4")));
}

/// <summary>A trainable tensor and its accumulated gradient.</summary>
internal sealed class Parameter(NdArray value, double[] gradient, bool decay)
{
    internal NdArray Value { get; } = value;

    internal double[] Gradient { get; } = gradient;

    /// <summary>Whether weight decay applies. Biases are exempt, as in the Hugging Face Trainer.</summary>
    internal bool Decay { get; } = decay;
}

/// <summary>AdamW exactly as <c>torch.optim.AdamW</c> computes it.</summary>
/// <remarks>
/// Decay is decoupled - the weight shrinks by <c>lr * decay</c> before the Adam step rather than
/// the decay being added to the gradient - and bias correction is applied the way torch applies
/// it: the step size is divided by <c>1 - beta1^t</c> and the square root of the second moment by
/// <c>sqrt(1 - beta2^t)</c>, with epsilon added after. Moving epsilon inside the square root, as
/// several write-ups do, changes the first step's size noticeably when gradients are small.
/// </remarks>
internal sealed class AdamW
{
    private readonly Parameter[] _parameters;
    private readonly double[][] _first;
    private readonly double[][] _second;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;
    private readonly double _decay;

    internal AdamW(
        IReadOnlyList<Parameter> parameters, double weightDecay = 0.0,
        double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
    {
        _parameters = [.. parameters];
        _first = [.. _parameters.Select(p => new double[p.Gradient.Length])];
        _second = [.. _parameters.Select(p => new double[p.Gradient.Length])];
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _decay = weightDecay;
    }

    /// <summary>Optimizer steps taken.</summary>
    internal int Steps { get; private set; }

    /// <summary>Applies one step with the gradients currently accumulated, then leaves them in place.</summary>
    internal void Step(double learningRate)
    {
        Steps++;
        var correction1 = 1 - Math.Pow(_beta1, Steps);
        var correction2 = Math.Sqrt(1 - Math.Pow(_beta2, Steps));
        var stepSize = learningRate / correction1;

        for (var p = 0; p < _parameters.Length; p++)
        {
            var parameter = _parameters[p];
            var value = parameter.Value.AsSpan();
            var gradient = parameter.Gradient;
            var m = _first[p];
            var v = _second[p];
            var shrink = parameter.Decay ? 1 - learningRate * _decay : 1.0;

            for (var i = 0; i < gradient.Length; i++)
            {
                value[i] *= shrink;
                m[i] = _beta1 * m[i] + (1 - _beta1) * gradient[i];
                v[i] = _beta2 * v[i] + (1 - _beta2) * gradient[i] * gradient[i];
                value[i] -= stepSize * m[i] / (Math.Sqrt(v[i]) / correction2 + _epsilon);
            }
        }
    }
}

/// <summary>The learning-rate schedule of <c>get_linear_schedule_with_warmup</c>.</summary>
internal static class LinearSchedule
{
    /// <summary>The multiplier for the optimizer step that follows <paramref name="step"/> earlier ones.</summary>
    /// <remarks>
    /// Counted the way the reference counts, from zero - so with any warm-up at all the very first
    /// step is taken at a learning rate of exactly zero. That surprises people, but matching it is
    /// what makes a loss curve comparable with one from Python.
    /// </remarks>
    internal static double Factor(int step, int warmup, int total)
    {
        if (step < warmup) return (double)step / Math.Max(1, warmup);
        return Math.Max(0.0, (double)(total - step) / Math.Max(1, total - warmup));
    }
}

/// <summary>The training loop behind <see cref="PeftModel.Train"/>.</summary>
internal static class LoraTrainer
{
    /// <summary>Fits every adapter in <paramref name="encoder"/> and <paramref name="head"/>, in place.</summary>
    /// <param name="encoder">The encoder with its adapters in the loop.</param>
    /// <param name="head">The classifier to train alongside.</param>
    /// <param name="ids">Token ids per example, already truncated.</param>
    /// <param name="targets">Class index per example.</param>
    /// <param name="options">Epochs, batch size, learning rate and schedule.</param>
    internal static TrainingReport Fit(
        LoraEncoder encoder, ClassifierHead head, IReadOnlyList<int[]> ids, IReadOnlyList<int> targets,
        TrainingOptions options)
        => Fit(encoder, head, ids, [.. targets.Select(t => new[] { t })], options);

    /// <summary>Fits every adapter in <paramref name="encoder"/> and <paramref name="head"/>, in place.</summary>
    /// <param name="encoder">The encoder with its adapters in the loop.</param>
    /// <param name="head">The classifier to train alongside.</param>
    /// <param name="ids">Token ids per example, already truncated.</param>
    /// <param name="targets">
    /// Targets per example: one class for a sequence head, one per position for a token head, with
    /// <see cref="LinearHead.Ignored"/> where a position does not count.
    /// </param>
    /// <param name="options">Epochs, batch size, learning rate and schedule.</param>
    /// <param name="typeIds">Segment ids per example, for sentence pairs; <c>null</c> when every position is segment 0.</param>
    /// <remarks>
    /// A step's loss is the mean over every target it contains - over examples for a sequence
    /// head, over labelled tokens for a token head. That is what PyTorch's cross-entropy computes
    /// on a batch, and it keeps a long sentence from counting for more than its tokens.
    /// </remarks>
    internal static TrainingReport Fit(
        LoraEncoder encoder, LinearHead head, IReadOnlyList<int[]> ids, IReadOnlyList<int[]> targets,
        TrainingOptions options, IReadOnlyList<int[]>? typeIds = null)
    {
        var started = System.Diagnostics.Stopwatch.StartNew();
        var width = encoder.Hidden;
        var gradients = new LoraGradients();

        var parameters = new List<Parameter>();
        foreach (var adapter in encoder.Adapters)
        {
            var (gradA, gradB) = gradients.For(adapter);
            parameters.Add(new Parameter(adapter.A, gradA, decay: true));
            parameters.Add(new Parameter(adapter.B, gradB, decay: true));
        }

        parameters.Add(new Parameter(head.Weight, head.WeightGradient, decay: true));
        parameters.Add(new Parameter(head.Bias, head.BiasGradient, decay: false));

        var optimizer = new AdamW(parameters, options.WeightDecay);

        var perStep = options.BatchSize * options.GradientAccumulation;
        var stepsPerEpoch = (ids.Count + perStep - 1) / perStep;
        var total = stepsPerEpoch * options.Epochs;
        var warmup = (int)Math.Ceiling(options.WarmupFraction * total);

        var random = new Random(options.Seed);
        var order = Enumerable.Range(0, ids.Count).ToArray();
        var stepLosses = new List<double>();
        var epochLosses = new List<double>();

        for (var epoch = 1; epoch <= options.Epochs; epoch++)
        {
            random.Shuffle(order);
            var epochLoss = 0.0;

            var epochUnits = 0;

            for (var first = 0; first < order.Length; first += perStep)
            {
                // The loss of a step is the mean over its targets, however they split into
                // micro-batches; with full micro-batches that is what the reference computes.
                var count = Math.Min(perStep, order.Length - first);
                var units = 0;
                for (var e = first; e < first + count; e++) units += head.Units(targets[order[e]]);

                var weight = units == 0 ? 0.0 : 1.0 / units;
                var stepLoss = 0.0;

                for (var e = first; e < first + count; e++)
                {
                    var example = order[e];
                    var tape = new LoraEncoder.Tape();
                    var hidden = encoder.Forward(ids[example], tape, random, typeIds?[example]);

                    var (loss, dHidden) = head.Backward(hidden, ids[example].Length, width, targets[example], weight);
                    encoder.Backward(dHidden, tape, gradients);

                    stepLoss += loss * weight;
                }

                ClipGradients(parameters, options.MaxGradientNorm);

                var rate = options.LearningRate * LinearSchedule.Factor(optimizer.Steps, warmup, total);
                optimizer.Step(rate);

                gradients.Clear();
                head.ClearGradients();

                stepLosses.Add(stepLoss);
                epochLoss += stepLoss * units;
                epochUnits += units;
                options.Progress?.Report(new TrainingProgress(epoch, optimizer.Steps, total, stepLoss, rate));
            }

            epochLosses.Add(epochUnits == 0 ? 0.0 : epochLoss / epochUnits);
        }

        return new TrainingReport(stepLosses, epochLosses, optimizer.Steps, started.Elapsed);
    }

    /// <summary>Scales every gradient down so their joint norm is at most <paramref name="limit"/>.</summary>
    /// <remarks>The same rule as <c>torch.nn.utils.clip_grad_norm_</c>, including its 1e-6.</remarks>
    internal static void ClipGradients(IReadOnlyList<Parameter> parameters, double limit)
    {
        if (limit <= 0) return;

        var squares = 0.0;
        foreach (var parameter in parameters)
        {
            foreach (var g in parameter.Gradient) squares += g * g;
        }

        var coefficient = limit / (Math.Sqrt(squares) + 1e-6);
        if (coefficient >= 1) return;

        foreach (var parameter in parameters)
        {
            var gradient = parameter.Gradient;
            for (var i = 0; i < gradient.Length; i++) gradient[i] *= coefficient;
        }
    }
}

