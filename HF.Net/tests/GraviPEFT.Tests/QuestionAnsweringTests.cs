using System.Text.Json;
using Gravicode.HFNet.GraviHub.Io;
using Gravicode.HFNet.GraviPEFT;
using Gravicode.HFNet.GraviTokenizers;
using Gravicode.HFNet.GraviTransformers;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.HFNet.GraviPEFT.Tests;

/// <summary>Training adapters for extractive question answering.</summary>
public sealed class QuestionAnsweringTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "hfnet-qa", Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }

    private static readonly int[] Pair = [1, 17, 5, 2, 29, 11, 3, 8, 2];
    private static readonly int[] Segments = [0, 0, 0, 0, 1, 1, 1, 1, 1];

    private static NdArray Delta(int seed)
    {
        var random = new GraviRandom(seed);
        return new NdArray([.. Enumerable.Range(0, 8).Select(_ => (double)(float)(0.5 * random.Normal(0, 1)))], 8);
    }

    [Fact]
    public void A_pair_goes_through_the_encoder_as_inference_runs_it()
    {
        // Segment 1 adds the segment difference before the embedding norm, in the same order of
        // addition as the inference encoder, so the two agree to rounding.
        var (source, config) = GradientTests.TinyModel(41);
        var delta = Delta(42);
        var encoder = LoraEncoder.Build(source, config, (_, _) => null, delta);
        var inference = CompiledEncoder.Build(source, config);

        var expected = inference.Forward(Pair, [.. Pair.Select(_ => 1)], Segments, delta).ToArray();
        var actual = encoder.Forward(Pair, typeIds: Segments);
        var unpaired = encoder.Forward(Pair);

        for (var i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-12, $"[{i}] {actual[i]} against {expected[i]}");
        }

        Assert.True(actual.Zip(unpaired).Max(p => Math.Abs(p.First - p.Second)) > 1e-3, "the segment made a difference");
    }

    [Fact]
    public void The_span_head_s_gradient_agrees_with_numerical_differentiation()
    {
        var (source, config) = GradientTests.TinyModel(43);
        var adapters = GradientTests.EveryProjection();
        var encoder = LoraEncoder.Build(source, config, (l, p) => adapters[(l, p)], Delta(44));
        var head = new SpanHead(8, seed: 6);
        int[] targets = [5, 6];

        double Loss()
        {
            var (start, end) = head.Logits(encoder.Forward(Pair, typeIds: Segments), Pair.Length, 8);
            return -(Math.Log(LinearHead.Softmax(start)[5]) + Math.Log(LinearHead.Softmax(end)[6])) / 2;
        }

        var tape = new LoraEncoder.Tape();
        var hidden = encoder.Forward(Pair, tape, typeIds: Segments);
        var (loss, dHidden) = head.Backward(hidden, Pair.Length, 8, targets, weight: 1.0);
        var gradients = new LoraGradients();
        encoder.Backward(dHidden, tape, gradients);

        Assert.True(Math.Abs(loss - Loss()) < 1e-14);

        const double H = 1e-5;
        var largest = 0.0;

        void Check(Span<double> values, double[] analytic, string what)
        {
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
                Assert.True(Math.Abs(numeric - analytic[i]) < 1e-6, $"{what}[{i}]: backward {analytic[i]:E6}, numerical {numeric:E6}");
            }
        }

        foreach (var ((layer, projection), adapter) in adapters)
        {
            var (gradA, gradB) = gradients.For(adapter);
            Check(adapter.A.AsSpan(), gradA, $"layer {layer} {projection} A");
            Check(adapter.B.AsSpan(), gradB, $"layer {layer} {projection} B");
        }

        Check(head.Weight.AsSpan(), head.WeightGradient, "qa_outputs W");
        Check(head.Bias.AsSpan(), head.BiasGradient, "qa_outputs b");
        Assert.True(largest > 1e-2, $"largest gradient {largest}");
    }

    /// <summary>
    /// "who lives there?" / "Ani lives in Bandung.": [CLS] who lives there ? [SEP] ani lives in
    /// bandung . [SEP], with context offsets into the context.
    /// </summary>
    private static Encoding AniInBandung() => new(
        ids: [101, 1, 2, 3, 4, 102, 5, 2, 6, 7, 8, 102],
        tokens: ["[CLS]", "who", "lives", "there", "?", "[SEP]", "ani", "lives", "in", "bandung", ".", "[SEP]"],
        attentionMask: [.. Enumerable.Repeat(1, 12)],
        typeIds: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        specialTokensMask: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        offsets: [(0, 0), (0, 3), (4, 9), (10, 15), (15, 16), (0, 0), (0, 3), (4, 9), (10, 12), (13, 20), (20, 21), (0, 0)]);

    [Fact]
    public void An_answer_maps_to_its_first_and_last_context_token()
    {
        // "Ani" is 0..3 in the context, token 6; "in Bandung" is 10..20, tokens 8 to 9. The
        // question's own "lives" at 4..9 must not be chosen - only segment 1 counts.
        Assert.Equal((6, 6), TokenAlignment.AnswerTokens(AniInBandung(), 0, 3, 12));
        Assert.Equal((8, 9), TokenAlignment.AnswerTokens(AniInBandung(), 10, 20, 12));
        Assert.Equal((7, 7), TokenAlignment.AnswerTokens(AniInBandung(), 4, 9, 12));
    }

    [Fact]
    public void An_answer_cut_off_by_truncation_points_at_CLS()
    {
        // With only the first nine tokens kept, "bandung" is gone.
        Assert.Equal((0, 0), TokenAlignment.AnswerTokens(AniInBandung(), 13, 20, 9));
    }

    [Fact]
    public void Training_learns_where_the_answer_is()
    {
        // The answer is marker 7, wherever it falls in the context. This checks the training
        // wiring - the pair input, the span loss, segment ids per example - on a task a model this
        // small learns reliably; whether the gradients are right is the gradient check's job, and
        // whether it learns a real task is checked against bert-base and Python.
        var (source, config) = GradientTests.TinyModel(47);
        var random = new Random(5);
        var ids = new List<int[]>();
        var types = new List<int[]>();
        var targets = new List<int[]>();

        for (var n = 0; n < 40; n++)
        {
            var context = Enumerable.Range(0, 8).Select(_ => random.Next(10, 30)).ToArray();
            var at = random.Next(0, 8);
            context[at] = 7;

            ids.Add([1, 3, 4, 2, .. context, 2]);
            types.Add([0, 0, 0, 0, .. Enumerable.Repeat(1, 9)]);
            targets.Add([4 + at, 4 + at]);
        }

        var lora = new LoraConfig(Rank: 4, Alpha: 8);
        var adapters = new Dictionary<(int, Projection), LoraAdapter>();
        for (var layer = 0; layer < 2; layer++)
        {
            foreach (var p in Enum.GetValues<Projection>())
            {
                var (inputs, outputs) = p switch
                {
                    Projection.Intermediate => (8, 16),
                    Projection.Output => (16, 8),
                    _ => (8, 8),
                };

                adapters[(layer, p)] = new LoraAdapter(inputs, outputs, lora, seed: layer * 6 + (int)p);
            }
        }

        var encoder = LoraEncoder.Build(source, config, (l, p) => adapters[(l, p)], Delta(48));
        var head = new SpanHead(8, seed: 2);

        var report = LoraTrainer.Fit(encoder, head, ids, targets,
            new TrainingOptions { Epochs = 80, BatchSize = 8, LearningRate = 2e-2 }, types);

        Assert.True(report.EpochLosses[^1] < 0.3 * report.EpochLosses[0], report.ToString());

        var correct = 0;
        for (var i = 0; i < ids.Count; i++)
        {
            var (start, _) = head.Logits(encoder.Forward(ids[i], typeIds: types[i]), ids[i].Length, 8);
            if (Array.IndexOf(start, start.Max()) == targets[i][0]) correct++;
        }

        Assert.True(correct >= 34, $"{correct}/40 starts right; {report}");
    }

    [Fact]
    public void A_span_head_is_saved_as_PEFT_saves_QUESTION_ANS()
    {
        // PEFT 0.21, BertForQuestionAnswering, task_type=QUESTION_ANS: base_model.model.qa_outputs.*
        // and modules_to_save ["qa_outputs"].
        var head = new SpanHead(8, seed: 3);
        var adapters = new LoraAdapterSet(new Dictionary<string, LoraAdapter>
        {
            ["bert.encoder.layer.0.attention.self.query"] = new LoraAdapter(8, 8, new LoraConfig(Rank: 2)),
        }, new LoraConfig(Rank: 2));

        adapters.Save(_directory, "bert-base-uncased", new SavedHead(head, [], 384));

        var tensors = SafeTensors.ReadAll(Path.Combine(_directory, "adapter_model.safetensors"));
        Assert.Contains("base_model.model.qa_outputs.weight", tensors.Keys);
        Assert.DoesNotContain(tensors.Keys, k => k.Contains("classifier"));

        using var config = JsonDocument.Parse(File.ReadAllText(Path.Combine(_directory, "adapter_config.json")));
        Assert.Equal("QUESTION_ANS", config.RootElement.GetProperty("task_type").GetString());
        Assert.Equal(["qa_outputs"], config.RootElement.GetProperty("modules_to_save").EnumerateArray().Select(e => e.GetString()));
        Assert.Equal("question_answering", HeadDescription.Read(_directory)!.Task);

        var read = LoraAdapterSet.Load(Path.Combine(_directory, "adapter_model.safetensors"));
        Assert.Equal([2, 8], read.Others["qa_outputs.weight"].Shape.ToArray());
    }
}
