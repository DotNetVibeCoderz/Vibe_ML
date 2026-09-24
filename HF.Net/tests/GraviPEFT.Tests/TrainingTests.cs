using Gravicode.HFNet.GraviPEFT;
using Gravicode.HFNet.GraviTransformers;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Transformers;
using Xunit;
using Encoder = Gravicode.Science.GraviText.Transformers.TransformerModel;

namespace Gravicode.HFNet.GraviPEFT.Tests;

/// <summary>
/// The adapter backward pass, pinned to numerical differentiation of the forward pass, and the
/// forward pass pinned to the inference encoder.
/// </summary>
/// <remarks>
/// The model is tiny and random, and deliberately not tidy: every bias and every norm is moved away
/// from its initial value. A pretrained BERT's attention biases are not zero, and they are exactly
/// what the foundation's autodiff encoder leaves out - a gradient check on a model whose biases
/// were zero could not tell the two apart.
/// </remarks>
public sealed class GradientTests
{
    private const int Hidden = 8;
    private const int Inner = 16;
    private const int Layers = 2;

    internal static (Encoder Encoder, PretrainedConfig Config) TinyModel(int seed, string activation = "gelu")
    {
        var encoder = new Encoder(new TransformerConfig(30, Hidden, Layers, 2, Inner, 16), seed);
        var random = new GraviRandom(seed + 1);

        void Shift(NdArray values, double centre, double scale)
        {
            for (var i = 0; i < values.Size; i++) values.SetAt(i, centre + scale * random.Normal(0, 1));
        }

        foreach (var layer in encoder.Layers)
        {
            foreach (var dense in (DenseLayer[])[
                layer.Attention.Query, layer.Attention.Key, layer.Attention.Value, layer.Attention.Output,
                layer.Intermediate, layer.OutputProjection])
            {
                Shift(dense.Bias, 0, 0.2);
            }

            Shift(layer.AttentionNorm.Gamma, 1, 0.2);
            Shift(layer.AttentionNorm.Beta, 0, 0.2);
            Shift(layer.OutputNorm.Gamma, 1, 0.2);
            Shift(layer.OutputNorm.Beta, 0, 0.2);
        }

        Shift(encoder.EmbeddingNorm.Gamma, 1, 0.2);
        Shift(encoder.EmbeddingNorm.Beta, 0, 0.2);

        var config = new PretrainedConfig
        {
            ModelType = "bert",
            HiddenSize = Hidden,
            Layers = Layers,
            Heads = 2,
            IntermediateSize = Inner,
            VocabularySize = 30,
            MaxPositions = 16,
            Activation = activation,
        };

        return (encoder, config);
    }

    /// <summary>An adapter on every projection of every layer, with B already moved off zero.</summary>
    internal static Dictionary<(int, Projection), LoraAdapter> EveryProjection(double dropout = 0.0)
    {
        var config = new LoraConfig(Rank: 2, Alpha: 4, Dropout: dropout);
        var adapters = new Dictionary<(int, Projection), LoraAdapter>();
        var random = new GraviRandom(99);

        for (var layer = 0; layer < Layers; layer++)
        {
            foreach (var projection in Enum.GetValues<Projection>())
            {
                var (inputs, outputs) = projection switch
                {
                    Projection.Intermediate => (Hidden, Inner),
                    Projection.Output => (Inner, Hidden),
                    _ => (Hidden, Hidden),
                };

                var adapter = new LoraAdapter(inputs, outputs, config, seed: 10 * layer + (int)projection);

                // With B at zero - as LoRA starts - every gradient with respect to A is exactly zero,
                // which would make half of this check vacuous.
                var b = adapter.B.AsSpan();
                for (var i = 0; i < b.Length; i++) b[i] = 0.3 * random.Normal(0, 1);

                adapters[(layer, projection)] = adapter;
            }
        }

        return adapters;
    }

    private static readonly int[] Sentence = [2, 17, 5, 29, 11, 3];

    private static double Loss(LoraEncoder encoder, ClassifierHead head, int target, int? dropoutSeed)
    {
        var tape = dropoutSeed is null ? null : new LoraEncoder.Tape();
        var random = dropoutSeed is null ? null : new Random(dropoutSeed.Value);
        var hidden = encoder.Forward(Sentence, tape, random);

        var probabilities = ClassifierHead.Softmax(head.Logits(ClassifierHead.Pool(hidden, Sentence.Length, Hidden)));
        return -Math.Log(probabilities[target]);
    }

    [Theory]
    [InlineData("gelu", 0.0)]
    [InlineData("gelu_new", 0.0)]
    [InlineData("gelu", 0.25)]
    public void Every_adapter_gradient_agrees_with_numerical_differentiation(string activation, double dropout)
    {
        // The v0.3 milestone, as PLAN.md words it: a numerically differentiated loss and the
        // backward pass agreeing to 1e-6 on a two-layer model. With dropout the same seed draws the
        // same masks for every evaluation, so the loss is still a fixed smooth function.
        var (source, config) = TinyModel(7, activation);
        var adapters = EveryProjection(dropout);
        var encoder = LoraEncoder.Build(source, config, (l, p) => adapters[(l, p)]);
        var head = new ClassifierHead(Hidden, 3, seed: 5);
        const int Target = 1;
        int? seed = dropout > 0 ? 1234 : null;

        var tape = new LoraEncoder.Tape();
        var hidden = encoder.Forward(Sentence, tape, seed is null ? null : new Random(seed.Value));
        var (loss, dHidden) = head.Backward(hidden, Sentence.Length, Hidden, Target, weight: 1.0);
        var gradients = new LoraGradients();
        encoder.Backward(dHidden, tape, gradients);

        Assert.True(Math.Abs(loss - Loss(encoder, head, Target, seed)) < 1e-14, "the recorded pass is the same function");

        const double H = 1e-5;
        var worst = 0.0;
        var largest = 0.0;

        void Check(Span<double> values, double[] analytic, string what)
        {
            for (var i = 0; i < values.Length; i++)
            {
                var original = values[i];
                values[i] = original + H;
                var up = Loss(encoder, head, Target, seed);
                values[i] = original - H;
                var down = Loss(encoder, head, Target, seed);
                values[i] = original;

                var numeric = (up - down) / (2 * H);
                var error = Math.Abs(numeric - analytic[i]);

                worst = Math.Max(worst, error);
                largest = Math.Max(largest, Math.Abs(numeric));
                Assert.True(error < 1e-6, $"{what}[{i}]: backward {analytic[i]:E6}, numerical {numeric:E6}");
            }
        }

        foreach (var ((layer, projection), adapter) in adapters)
        {
            var (gradA, gradB) = gradients.For(adapter);
            Check(adapter.A.AsSpan(), gradA, $"layer {layer} {projection} A");
            Check(adapter.B.AsSpan(), gradB, $"layer {layer} {projection} B");
        }

        Check(head.Weight.AsSpan(), head.WeightGradient, "head W");
        Check(head.Bias.AsSpan(), head.BiasGradient, "head b");

        // Agreement between two zeros proves nothing; the gradients here are of order 0.1.
        Assert.True(largest > 1e-2, $"largest gradient {largest}");
    }

    [Fact]
    public void Without_adapters_the_forward_pass_is_the_inference_encoder_s()
    {
        var (source, config) = TinyModel(3);
        var encoder = LoraEncoder.Build(source, config, (_, _) => null);
        var inference = CompiledEncoder.Build(source, config);

        var expected = inference.Forward(Sentence, [.. Sentence.Select(_ => 1)]).ToArray();
        var actual = encoder.Forward(Sentence);

        for (var i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-12, $"[{i}] {actual[i]} against {expected[i]}");
        }
    }

    [Fact]
    public void Adapters_in_the_loop_give_what_merging_them_gives()
    {
        // The unmerged path (base + scaling * B A x) against the merged one (the delta folded into
        // the weights, then run by the inference encoder). The merged weights are rounded to
        // float32 on the way into the kernels, so the two differ at the level of that rounding.
        var (source, config) = TinyModel(4);
        var adapters = EveryProjection();
        var encoder = LoraEncoder.Build(source, config, (l, p) => adapters[(l, p)]);
        var adapted = encoder.Forward(Sentence);

        foreach (var ((layer, projection), adapter) in adapters)
        {
            var block = source.Layers[layer];
            var dense = projection switch
            {
                Projection.Query => block.Attention.Query,
                Projection.Key => block.Attention.Key,
                Projection.Value => block.Attention.Value,
                Projection.AttentionOutput => block.Attention.Output,
                Projection.Intermediate => block.Intermediate,
                _ => block.OutputProjection,
            };

            adapter.MergeInto(dense.Weights);
        }

        var merged = CompiledEncoder.Build(source, config).Forward(Sentence, [.. Sentence.Select(_ => 1)]).ToArray();
        var baseline = LoraEncoder.Build(TinyModel(4).Encoder, config, (_, _) => null).Forward(Sentence);

        var moved = 0.0;
        for (var i = 0; i < merged.Length; i++)
        {
            Assert.True(Math.Abs(merged[i] - adapted[i]) < 1e-5, $"[{i}] {adapted[i]} against {merged[i]}");
            moved = Math.Max(moved, Math.Abs(adapted[i] - baseline[i]));
        }

        Assert.True(moved > 1e-2, "the adapters changed the output");
    }
}

/// <summary>AdamW, the schedule and clipping, each against the reference's own formula.</summary>
public sealed class OptimizerTests
{
    [Fact]
    public void AdamW_s_first_step_moves_each_weight_by_the_learning_rate()
    {
        // After bias correction the first Adam step is lr * g / (|g| + eps): a step of exactly the
        // learning rate in the direction against the gradient, whatever the gradient's size.
        var value = new NdArray([0.5, -1.0, 2.0], 3);
        var gradient = new[] { 0.2, -3.0, 1e-3 };

        new AdamW([new Parameter(value, gradient, decay: false)]).Step(0.1);

        Assert.True(Math.Abs(value[0] - (0.5 - 0.1 * 0.2 / (0.2 + 1e-8))) < 1e-15);
        Assert.True(Math.Abs(value[1] - (-1.0 + 0.1 * 3.0 / (3.0 + 1e-8))) < 1e-15);
        Assert.True(Math.Abs(value[2] - (2.0 - 0.1 * 1e-3 / (1e-3 + 1e-8))) < 1e-15);
    }

    [Fact]
    public void AdamW_matches_torch_s_algorithm_over_several_steps_with_decay()
    {
        // torch.optim.AdamW, written out from its documentation: decay first, then moments, then
        // the bias-corrected step with epsilon outside the square root.
        var value = new NdArray([0.7, -0.4], 2);
        var gradient = new double[2];
        var optimizer = new AdamW([new Parameter(value, gradient, decay: true)], weightDecay: 0.1);

        double[] p = [0.7, -0.4], m = [0, 0], v = [0, 0];
        double[][] gradients = [[0.3, -0.2], [0.1, 0.4], [-0.5, 0.05]];
        double[] rates = [0.01, 0.02, 0.005];

        for (var t = 1; t <= 3; t++)
        {
            gradients[t - 1].CopyTo(gradient, 0);
            optimizer.Step(rates[t - 1]);

            for (var i = 0; i < 2; i++)
            {
                var g = gradients[t - 1][i];
                p[i] -= rates[t - 1] * 0.1 * p[i];
                m[i] = 0.9 * m[i] + 0.1 * g;
                v[i] = 0.999 * v[i] + 0.001 * g * g;
                var mHat = m[i] / (1 - Math.Pow(0.9, t));
                var vHat = v[i] / (1 - Math.Pow(0.999, t));
                p[i] -= rates[t - 1] * mHat / (Math.Sqrt(vHat) + 1e-8);
            }

            for (var i = 0; i < 2; i++) Assert.True(Math.Abs(value[i] - p[i]) < 1e-15, $"step {t} [{i}]: {value[i]} against {p[i]}");
        }
    }

    [Theory]
    // get_linear_schedule_with_warmup(num_warmup_steps=2, num_training_steps=10).lr_lambda(step)
    [InlineData(0, 0.0)]
    [InlineData(1, 0.5)]
    [InlineData(2, 1.0)]
    [InlineData(6, 0.5)]
    [InlineData(9, 0.125)]
    [InlineData(10, 0.0)]
    public void The_schedule_is_the_reference_s_linear_warmup_and_decay(int step, double expected)
        => Assert.Equal(expected, LinearSchedule.Factor(step, 2, 10), 15);

    [Fact]
    public void Clipping_scales_the_joint_norm_down_as_torch_does()
    {
        double[] a = [3.0], b = [4.0];
        var parameters = new[]
        {
            new Parameter(NdArray.Zeros(1), a, decay: false),
            new Parameter(NdArray.Zeros(1), b, decay: false),
        };

        LoraTrainer.ClipGradients(parameters, 1.0);

        var coefficient = 1.0 / (5.0 + 1e-6);
        Assert.Equal(3.0 * coefficient, a[0], 15);
        Assert.Equal(4.0 * coefficient, b[0], 15);
    }

    [Fact]
    public void A_zero_head_would_starve_every_adapter_so_the_head_starts_small_and_random()
    {
        var head = new ClassifierHead(64, 2, seed: 1);
        var weights = head.Weight.ToArray();

        Assert.Contains(weights, w => w != 0);
        var std = Math.Sqrt(weights.Select(w => w * w).Average());
        Assert.True(std is > 0.01 and < 0.04, $"std {std}");
    }
}

/// <summary>End-to-end training on a task the base model cannot do.</summary>
public sealed class TrainerTests
{
    [Fact]
    public void Training_fits_a_task_and_moves_the_adapters()
    {
        // Label 1 when token 5 comes before token 9, 0 when after: the same bag of tokens either
        // way, so a head over the frozen, mean-pooled features gets no signal from which tokens are
        // present - only the adapted attention can tell the order apart.
        var (source, config) = GradientTests.TinyModel(11);
        var random = new Random(3);
        var ids = new List<int[]>();
        var targets = new List<int>();

        for (var n = 0; n < 48; n++)
        {
            var sequence = Enumerable.Range(0, 6).Select(_ => random.Next(10, 30)).ToArray();
            var first = random.Next(0, 3);
            var second = random.Next(first + 1, 6);
            var label = n % 2;

            sequence[first] = label == 1 ? 5 : 9;
            sequence[second] = label == 1 ? 9 : 5;
            ids.Add(sequence);
            targets.Add(label);
        }

        var lora = new LoraConfig(Rank: 4, Alpha: 8, TargetModules: ["query", "key", "value"]);
        var adapters = new Dictionary<(int, Projection), LoraAdapter>();
        for (var layer = 0; layer < 2; layer++)
        {
            foreach (var p in (Projection[])[Projection.Query, Projection.Key, Projection.Value])
            {
                adapters[(layer, p)] = new LoraAdapter(8, 8, lora, seed: layer * 3 + (int)p);
            }
        }

        var encoder = LoraEncoder.Build(source, config, (l, p) => adapters.GetValueOrDefault((l, p)));
        var head = new ClassifierHead(8, 2, seed: 1);
        var steps = new List<TrainingProgress>();

        var report = LoraTrainer.Fit(encoder, head, ids, targets, new TrainingOptions
        {
            Epochs = 60,
            BatchSize = 8,
            LearningRate = 2e-2,
            WarmupFraction = 0.05,
            Progress = new Collect(steps),
        });

        Assert.Equal(6 * 60, report.Steps);
        Assert.Equal(report.Steps, steps.Count);
        Assert.True(report.EpochLosses[^1] < 0.5 * report.EpochLosses[0], report.ToString());
        Assert.All(adapters.Values, a => Assert.Contains(a.B.ToArray(), v => v != 0));

        var correct = 0;
        for (var i = 0; i < ids.Count; i++)
        {
            var hidden = encoder.Forward(ids[i]);
            var logits = head.Logits(ClassifierHead.Pool(hidden, ids[i].Length, 8));
            if ((logits[1] > logits[0] ? 1 : 0) == targets[i]) correct++;
        }

        Assert.True(correct >= 40, $"{correct}/48 correct after training; {report}");
    }

    private sealed class Collect(List<TrainingProgress> into) : IProgress<TrainingProgress>
    {
        public void Report(TrainingProgress value) => into.Add(value);
    }
}

/// <summary>The names a saved adapter carries, which decide whether PEFT in Python can load it.</summary>
public sealed class AdapterNameTests
{
    [Fact]
    public void Adapters_take_the_checkpoint_s_own_module_paths()
    {
        // PEFT matches adapter tensors to modules by name and skips, silently, any it cannot place.
        // An adapter saved as layer.0.query loads in Python as no adapter at all.
        string[] bert =
        [
            "bert.embeddings.word_embeddings.weight",
            "bert.encoder.layer.0.attention.self.query.weight",
            "bert.encoder.layer.0.attention.self.query.bias",
            "bert.encoder.layer.0.attention.output.dense.weight",
            "bert.encoder.layer.0.attention.output.LayerNorm.weight",
            "bert.encoder.layer.0.intermediate.dense.weight",
            "bert.encoder.layer.0.output.dense.weight",
            "bert.encoder.layer.0.output.LayerNorm.weight",
            "bert.encoder.layer.1.attention.self.value.weight",
        ];

        var paths = PeftModel.ModulePaths(bert, layers: 2);

        Assert.Equal("bert.encoder.layer.0.attention.self.query", paths[(0, Projection.Query)]);
        Assert.Equal("bert.encoder.layer.0.attention.output.dense", paths[(0, Projection.AttentionOutput)]);
        Assert.Equal("bert.encoder.layer.0.intermediate.dense", paths[(0, Projection.Intermediate)]);
        Assert.Equal("bert.encoder.layer.0.output.dense", paths[(0, Projection.Output)]);
        Assert.Equal("bert.encoder.layer.1.attention.self.value", paths[(1, Projection.Value)]);
        Assert.Equal(5, paths.Count);
    }

    [Fact]
    public void DistilBERT_s_spelling_maps_to_the_same_projections()
    {
        string[] distil =
        [
            "distilbert.transformer.layer.0.attention.q_lin.weight",
            "distilbert.transformer.layer.0.attention.out_lin.weight",
            "distilbert.transformer.layer.0.ffn.lin1.weight",
            "distilbert.transformer.layer.0.ffn.lin2.weight",
        ];

        var paths = PeftModel.ModulePaths(distil, layers: 1);

        Assert.Equal("distilbert.transformer.layer.0.attention.q_lin", paths[(0, Projection.Query)]);
        Assert.Equal("distilbert.transformer.layer.0.attention.out_lin", paths[(0, Projection.AttentionOutput)]);
        Assert.Equal("distilbert.transformer.layer.0.ffn.lin1", paths[(0, Projection.Intermediate)]);
        Assert.Equal("distilbert.transformer.layer.0.ffn.lin2", paths[(0, Projection.Output)]);
    }

    [Fact]
    public void Adapter_seeds_do_not_change_from_one_process_to_the_next()
    {
        // FNV-1a of "query" then layer 3, computed by hand from the algorithm's definition.
        // HashCode.Combine, which this replaced, differs in every process.
        var hash = 2166136261u;
        foreach (var c in "query") hash = (hash ^ c) * 16777619u;
        hash = (hash ^ 3u) * 16777619u;

        Assert.Equal((int)(hash & 0x7FFFFFFF), PeftModel.StableSeed(3, "query"));
        Assert.NotEqual(PeftModel.StableSeed(0, "query"), PeftModel.StableSeed(0, "value"));
        Assert.NotEqual(PeftModel.StableSeed(0, "query"), PeftModel.StableSeed(1, "query"));
    }
}
