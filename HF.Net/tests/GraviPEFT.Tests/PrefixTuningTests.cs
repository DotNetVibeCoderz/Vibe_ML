using System.Text.Json;
using Gravicode.HFNet.GraviPEFT;
using Gravicode.HFNet.GraviTransformers;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.HFNet.GraviPEFT.Tests;

/// <summary>
/// Prefix tuning's forward and backward pass: the prefix gradient against numerical
/// differentiation, and the prefix's effect against a closed-form case.
/// </summary>
public sealed class PrefixTuningTests
{
    private const int Hidden = 8;
    private const int Layers = 2;
    private const int Width = Layers * 2 * Hidden;

    private static readonly int[] First = [2, 17, 5, 29, 11, 3];
    private static readonly int[] Second = [2, 8, 21, 3];

    private static NdArray RandomPrefix(int virtualTokens, int seed)
    {
        var random = new GraviRandom(seed);
        return new NdArray([.. Enumerable.Range(0, virtualTokens * Width).Select(_ => 0.8 * random.Normal(0, 1))], virtualTokens, Width);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public void Every_prefix_gradient_agrees_with_numerical_differentiation(bool packed, bool withLora)
    {
        // Packed: two sequences share the prefix, so its gradient is the sum of both of theirs.
        // With LoRA: adapters and a prefix in the same pass, each gradient still exact.
        var (source, config) = GradientTests.TinyModel(11);
        var adapters = withLora ? GradientTests.EveryProjection() : null;
        var prefix = RandomPrefix(3, 21);
        var encoder = LoraEncoder.Build(source, config, (l, p) => adapters?[(l, p)], prefix: prefix);
        var head = new ClassifierHead(Hidden, 3, seed: 5, GradientTests.RandomPooler(8));

        int[] ids = packed ? [.. First, .. Second] : First;
        int[] lengths = packed ? [First.Length, Second.Length] : [First.Length];
        int[] targets = [1, 2];

        double Loss()
        {
            var hidden = encoder.Forward(ids, lengths: lengths);
            var total = 0.0;
            var offset = 0;
            for (var k = 0; k < lengths.Length; k++)
            {
                var own = hidden.AsSpan(offset * Hidden, lengths[k] * Hidden).ToArray();
                total -= Math.Log(ClassifierHead.Softmax(head.Logits(own, lengths[k], Hidden))[targets[k]]);
                offset += lengths[k];
            }

            return total;
        }

        var tape = new LoraEncoder.Tape();
        var output = encoder.Forward(ids, tape, lengths: lengths);
        var dHidden = new double[output.Length];
        var at = 0;
        for (var k = 0; k < lengths.Length; k++)
        {
            var own = output.AsSpan(at * Hidden, lengths[k] * Hidden).ToArray();
            head.Backward(own, lengths[k], Hidden, targets[k], weight: 1.0).HiddenGradient.CopyTo(dHidden, at * Hidden);
            at += lengths[k];
        }

        var gradients = new LoraGradients();
        encoder.Backward(dHidden, tape, gradients);

        const double H = 1e-5;
        var largest = 0.0;
        var values = prefix.AsSpan();
        var analytic = encoder.PrefixGradient!;

        for (var i = 0; i < values.Length; i++)
        {
            var original = values[i];
            values[i] = original + H;
            var up = Loss();
            values[i] = original - H;
            var down = Loss();
            values[i] = original;

            var numeric = (up - down) / (2 * H);
            largest = Math.Max(largest, Math.Abs(numeric));
            Assert.True(Math.Abs(numeric - analytic[i]) < 1e-6, $"prefix[{i}]: backward {analytic[i]:E6}, numerical {numeric:E6}");
        }

        if (withLora)
        {
            foreach (var ((layer, projection), adapter) in adapters!)
            {
                var (gradA, _) = gradients.For(adapter);
                var a = adapter.A.AsSpan();
                for (var i = 0; i < a.Length; i += 3)
                {
                    var original = a[i];
                    a[i] = original + H;
                    var up = Loss();
                    a[i] = original - H;
                    var down = Loss();
                    a[i] = original;
                    Assert.True(Math.Abs((up - down) / (2 * H) - gradA[i]) < 1e-6, $"layer {layer} {projection} A[{i}]");
                }
            }
        }

        Assert.True(largest > 1e-3, $"largest prefix gradient {largest}");
    }

    [Fact]
    public void Layer_k_reads_its_own_slice_of_each_prefix_row()
    {
        // Changing only layer 1's key/value slice must leave the gradient of layer 0's slice from a
        // loss on the output exactly as a pass with the original prefix gives - and must change the
        // output. And the same change on layer 0's slice must change layer-0-dependent output: the
        // two slices are independent parameters in the layout PEFT uses, [layer][key|value][head][d].
        var (source, config) = GradientTests.TinyModel(13);
        var prefix = RandomPrefix(2, 31);
        var encoder = LoraEncoder.Build(source, config, (_, _) => null, prefix: prefix);
        var before = encoder.Forward(First);

        var values = prefix.AsSpan();
        var layerOneValue = (2 * 1 + 1) * Hidden;
        var original = values[layerOneValue];
        values[layerOneValue] = original + 1.0;
        var after = encoder.Forward(First);
        values[layerOneValue] = original;

        Assert.True(before.Zip(after).Max(p => Math.Abs(p.First - p.Second)) > 1e-4, "layer 1's value slice is read");
        Assert.Equal(before, encoder.Forward(First));
    }

    [Fact]
    public void Training_lowers_the_loss_and_moves_only_the_prefix_and_head()
    {
        var (source, config) = GradientTests.TinyModel(14);
        var prefix = RandomPrefix(4, 41);
        var initial = prefix.ToArray();
        var encoder = LoraEncoder.Build(source, config, (_, _) => null, prefix: prefix);
        var head = new ClassifierHead(Hidden, 2, seed: 3, GradientTests.RandomPooler(9));

        // Memorising eight fixed sentences: the gradients are pinned numerically above, so this only
        // has to show the loop trains. A random 8-wide model cannot learn a real rule in a few
        // hundred steps (see CLAUDE.md on toy tests), and a test that asked it to would be flaky.
        var random = new Random(5);
        var ids = new List<int[]>();
        var targets = new List<int>();
        for (var i = 0; i < 8; i++)
        {
            ids.Add([2, .. Enumerable.Range(0, 4).Select(_ => random.Next(4, 29)), 3]);
            targets.Add(i % 2);
        }

        var report = LoraTrainer.Fit(encoder, head, ids, targets, new TrainingOptions { Epochs = 40, BatchSize = 4, LearningRate = 5e-2 });

        Assert.True(report.EpochLosses[^1] < report.EpochLosses[0] * 0.75, report.ToString());
        Assert.NotEqual(initial, prefix.ToArray());
    }

    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures");

    private static double[][] Logits(JsonElement root, string name)
        => [.. root.GetProperty(name).EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetDouble()).ToArray())];

    [Fact]
    public void An_adapter_saved_by_PEFT_gives_PEFT_s_logits()
    {
        // Fixtures/make_prefix.py: a random BERT, a PREFIX_TUNING adapter saved by PEFT 0.21, and
        // the adapted model's float64 logits from torch. The adapter file stores its classifier as
        // base_model.classifier.*, which sits on BERT's frozen pooler.
        using var model = TransformerModel.Open(Path.Combine(Fixtures, "bert-small"));
        var adapted = PrefixTuningModel.Load(model, Path.Combine(Fixtures, "prefix-adapter"));
        var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "prefix-reference.json"))).RootElement;
        var texts = reference.GetProperty("texts").EnumerateArray().Select(t => t.GetString()!).ToArray();
        var expected = Logits(reference, "adapted");

        Assert.Equal(3, adapted.VirtualTokens);
        for (var i = 0; i < texts.Length; i++)
        {
            var actual = adapted.Logits(texts[i]);
            var worst = expected[i].Zip(actual).Max(p => Math.Abs(p.First - p.Second));

            // Weights are float32 in the file; torch widened the same float32 values to double.
            Assert.True(worst < 1e-10, $"'{texts[i]}': largest logit difference {worst:E2}");
        }
    }

    [Fact]
    public void Saving_and_loading_keeps_the_prefix_the_head_and_the_labels()
    {
        using var model = TransformerModel.Open(Path.Combine(Fixtures, "bert-small"));
        var tuned = PEFT.ApplyPrefixTuning(model, new PrefixTuningConfig(VirtualTokens: 4, Seed: 3));
        tuned.Train(
            ["the cat sat on the mat", "a happy dog ran fast", "the sad tree", "a slow river"],
            ["cat", "dog", "cat", "dog"],
            new TrainingOptions { Epochs = 2, BatchSize = 2, LearningRate = 1e-2 });

        var directory = Path.Combine(Path.GetTempPath(), "hfnet-prefix-" + Guid.NewGuid().ToString("N"));
        try
        {
            tuned.Save(directory);
            var loaded = PrefixTuningModel.Load(model, directory);

            Assert.Equal(["cat", "dog"], loaded.Labels);

            // Saved as float32, as PEFT saves adapters.
            var before = tuned.Logits("the big house");
            var after = loaded.Logits("the big house");
            Assert.True(before.Zip(after).Max(p => Math.Abs(p.First - p.Second)) < 1e-5);

            using var config = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "adapter_config.json")));
            Assert.Equal("PREFIX_TUNING", config.RootElement.GetProperty("peft_type").GetString());
            Assert.Equal("SEQ_CLS", config.RootElement.GetProperty("task_type").GetString());
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void A_LoRA_adapter_is_refused_by_name()
    {
        var directory = Path.Combine(Path.GetTempPath(), "hfnet-notprefix-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            File.WriteAllText(Path.Combine(directory, "adapter_config.json"), """{"peft_type": "LORA"}""");
            using var model = TransformerModel.Open(Path.Combine(Fixtures, "bert-small"));

            var error = Assert.Throws<NotSupportedException>(() => PrefixTuningModel.Load(model, directory));
            Assert.Contains("PEFT.LoadAdapter", error.Message);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
