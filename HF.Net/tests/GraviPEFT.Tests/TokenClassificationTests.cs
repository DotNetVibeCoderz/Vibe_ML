using System.Text.Json;
using Gravicode.HFNet.GraviHub.Io;
using Gravicode.HFNet.GraviPEFT;
using Gravicode.HFNet.GraviTokenizers;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.HFNet.GraviPEFT.Tests;

/// <summary>Training adapters for named entities: the head, the word alignment and the saved form.</summary>
public sealed class TokenClassificationTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "hfnet-token", Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }

    /// <summary>
    /// "Kang Fadhil lives" as WordPiece would split it:
    /// [CLS] kang fad ##hil lives [SEP].
    /// </summary>
    private static Encoding KangFadhil() => new(
        ids: [101, 1, 2, 3, 4, 102],
        tokens: ["[CLS]", "kang", "fad", "##hil", "lives", "[SEP]"],
        attentionMask: [1, 1, 1, 1, 1, 1],
        typeIds: [0, 0, 0, 0, 0, 0],
        specialTokensMask: [1, 0, 0, 0, 0, 1],
        offsets: [(0, 0), (0, 4), (5, 8), (8, 11), (12, 17), (0, 0)]);

    [Fact]
    public void Joining_words_records_where_each_one_sits()
    {
        var (text, spans) = TokenAlignment.Join(["Kang", "Fadhil", "lives"]);

        Assert.Equal("Kang Fadhil lives", text);
        Assert.Equal([(0, 4), (5, 11), (12, 17)], spans);
    }

    [Fact]
    public void Only_the_first_piece_of_each_word_carries_its_tag()
    {
        // Transformers' label_all_tokens=False: [CLS], ##hil and [SEP] are -100.
        var (_, spans) = TokenAlignment.Join(["Kang", "Fadhil", "lives"]);

        var targets = TokenAlignment.Targets(KangFadhil(), spans, [1, 2, 0]);

        Assert.Equal([-100, 1, 2, -100, 0, -100], targets);
    }

    [Fact]
    public void A_continuing_piece_takes_its_word_s_first_piece_but_touching_punctuation_does_not()
    {
        // "Fadhil, Bandung": fad ##hil are one word; the comma touches it and is still its own.
        const string Text = "Fadhil, Bandung";
        var encoding = new Encoding(
            ids: [101, 1, 2, 3, 4, 102],
            tokens: ["[CLS]", "fad", "##hil", ",", "bandung", "[SEP]"],
            attentionMask: [1, 1, 1, 1, 1, 1],
            typeIds: [0, 0, 0, 0, 0, 0],
            specialTokensMask: [1, 0, 0, 0, 0, 1],
            offsets: [(0, 0), (0, 3), (3, 6), (6, 7), (8, 15), (0, 0)]);

        Assert.Equal([0, 1, 1, 3, 4, 5], TokenAlignment.WordStarts(Text, encoding));
    }

    /// <summary>A logit row that picks <paramref name="label"/> out of three.</summary>
    private static double[] Pick(int label) => [.. Enumerable.Range(0, 3).Select(c => c == label ? 5.0 : 0.0)];

    [Fact]
    public void A_word_is_decoded_whole_from_its_first_piece()
    {
        // Labels: 0 = O, 1 = B-PER, 2 = I-PER. ##hil's own row says O on purpose: it was never
        // trained, and it must neither end the span nor split the word.
        string[] labels = ["O", "B-PER", "I-PER"];
        double[][] logits = [Pick(0), Pick(1), Pick(2), Pick(0), Pick(0), Pick(0)];

        var entities = TokenAlignment.Decode("Kang Fadhil lives", KangFadhil(), logits, labels);

        var entity = Assert.Single(entities);
        Assert.Equal("Kang Fadhil", entity.Text);
        Assert.Equal("PER", entity.Label);
        Assert.Equal((0, 11), (entity.Start, entity.End));
    }

    [Fact]
    public void Two_adjacent_beginnings_are_two_entities()
    {
        string[] labels = ["O", "B-PER", "I-PER"];
        double[][] logits = [Pick(0), Pick(1), Pick(1), Pick(1), Pick(0), Pick(0)];

        var entities = TokenAlignment.Decode("Kang Fadhil lives", KangFadhil(), logits, labels);

        Assert.Equal(["Kang", "Fadhil"], entities.Select(e => e.Text));
    }

    [Fact]
    public void The_token_head_s_gradient_agrees_with_numerical_differentiation()
    {
        // The head over every row, with some positions ignored - the ignored ones must contribute
        // nothing, which a gradient check catches and a loss curve does not.
        var (source, config) = GradientTests.TinyModel(21);
        var adapters = GradientTests.EveryProjection();
        var encoder = LoraEncoder.Build(source, config, (l, p) => adapters[(l, p)]);
        var head = new TokenClassifierHead(8, 3, seed: 4);

        int[] ids = [2, 17, 5, 29, 11, 3];
        int[] targets = [LinearHead.Ignored, 2, 0, LinearHead.Ignored, 1, LinearHead.Ignored];
        var weight = 1.0 / head.Units(targets);
        Assert.Equal(3, head.Units(targets));

        double Loss()
        {
            var hidden = encoder.Forward(ids);
            var logits = head.Logits(hidden, ids.Length, 8);
            var total = 0.0;
            for (var r = 0; r < ids.Length; r++)
            {
                if (targets[r] == LinearHead.Ignored) continue;
                total -= Math.Log(LinearHead.Softmax(logits[r])[targets[r]]);
            }

            return total * weight;
        }

        var tape = new LoraEncoder.Tape();
        var forward = encoder.Forward(ids, tape);
        var (_, dHidden) = head.Backward(forward, ids.Length, 8, targets, weight);
        var gradients = new LoraGradients();
        encoder.Backward(dHidden, tape, gradients);

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

        Check(head.Weight.AsSpan(), head.WeightGradient, "classifier W");
        Check(head.Bias.AsSpan(), head.BiasGradient, "classifier b");
        Assert.True(largest > 1e-2, $"largest gradient {largest}");
    }

    [Fact]
    public void Training_learns_to_tag_a_token_by_its_neighbours()
    {
        // Token 5 is an entity when it follows token 7 and not otherwise: the label depends on
        // context, which only the adapted attention can supply.
        var (source, config) = GradientTests.TinyModel(31);
        var random = new Random(9);
        var ids = new List<int[]>();
        var targets = new List<int[]>();

        for (var n = 0; n < 40; n++)
        {
            var sequence = Enumerable.Range(0, 6).Select(_ => random.Next(10, 30)).ToArray();
            var at = random.Next(1, 6);
            var entity = n % 2 == 0;
            sequence[at] = 5;
            sequence[at - 1] = entity ? 7 : 8;

            ids.Add(sequence);
            targets.Add([.. Enumerable.Range(0, 6).Select(i => i == at && entity ? 1 : 0)]);
        }

        var lora = new LoraConfig(Rank: 4, Alpha: 8);
        var adapters = new Dictionary<(int, Projection), LoraAdapter>();
        for (var layer = 0; layer < 2; layer++)
        {
            foreach (var p in (Projection[])[Projection.Query, Projection.Key, Projection.Value])
            {
                adapters[(layer, p)] = new LoraAdapter(8, 8, lora, seed: layer * 3 + (int)p);
            }
        }

        var encoder = LoraEncoder.Build(source, config, (l, p) => adapters.GetValueOrDefault((l, p)));
        var head = new TokenClassifierHead(8, 2, seed: 1);

        var report = LoraTrainer.Fit(encoder, head, ids, targets, new TrainingOptions
        {
            Epochs = 80,
            BatchSize = 8,
            LearningRate = 2e-2,
        });

        Assert.True(report.EpochLosses[^1] < 0.3 * report.EpochLosses[0], report.ToString());

        var correct = 0;
        for (var i = 0; i < ids.Count; i++)
        {
            var logits = head.Logits(encoder.Forward(ids[i]), 6, 8);
            var at = Array.IndexOf(ids[i], 5);
            if ((logits[at][1] > logits[at][0] ? 1 : 0) == targets[i][at]) correct++;
        }

        Assert.True(correct >= 36, $"{correct}/40 entity positions right; {report}");
    }

    [Fact]
    public void A_token_head_is_saved_as_PEFT_saves_TOKEN_CLS()
    {
        var head = new TokenClassifierHead(8, 3, seed: 2);
        var adapters = new LoraAdapterSet(new Dictionary<string, LoraAdapter>
        {
            ["roberta.encoder.layer.0.attention.self.query"] = new LoraAdapter(8, 8, new LoraConfig(Rank: 2)),
        }, new LoraConfig(Rank: 2));

        adapters.Save(_directory, "roberta-base", new SavedHead(head, ["B-PER", "I-PER", "O"], 128));

        var tensors = SafeTensors.ReadAll(Path.Combine(_directory, "adapter_model.safetensors"));
        Assert.Contains("base_model.model.classifier.weight", tensors.Keys);

        using var config = JsonDocument.Parse(File.ReadAllText(Path.Combine(_directory, "adapter_config.json")));
        Assert.Equal("TOKEN_CLS", config.RootElement.GetProperty("task_type").GetString());
        Assert.Equal("TOKEN_CLS", HeadDescription.TaskType(_directory));

        var description = HeadDescription.Read(_directory)!;
        Assert.Equal("token", description.Task);
        Assert.Equal(["B-PER", "I-PER", "O"], description.Labels);
    }
}
