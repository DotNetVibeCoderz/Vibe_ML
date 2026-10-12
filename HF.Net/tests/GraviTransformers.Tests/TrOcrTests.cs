using System.Diagnostics;
using System.Text.Json;
using Gravicode.HFNet.GraviTokenizers;
using Gravicode.HFNet.GraviTransformers.Vision;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.HFNet.GraviTransformers.Tests;

/// <summary>
/// TrOCR against transformers in float64 on two tiny random checkpoints that between them cover every
/// configuration switch - on the fixtures <c>Fixtures/make_trocr.py</c> writes.
/// </summary>
/// <remarks>
/// trocr-tiny: distilled DeiT, relu, learned positions, scaled embeddings, embedding norm, its own
/// output layer. trocr-tiny-vit: ViT without q/k/v bias, gelu, sinusoidal positions, no scale or
/// embedding norm, output layer tied to the token table.
/// </remarks>
public sealed class TrOcrTests
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures");

    private static readonly JsonElement Reference =
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "trocr-reference.json"))).RootElement;

    private static double[] Values(JsonElement node) => [.. node.EnumerateArray().Select(v => v.GetDouble())];

    private static int[] Ids(JsonElement node) => [.. node.EnumerateArray().Select(v => v.GetInt32())];

    private static NdArray Pixels(string folder) => new(Values(Reference.GetProperty(folder).GetProperty("pixels")), 3, 32, 32);

    private static void Close(double[] expected, double[] actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        var worst = expected.Zip(actual).Max(p => Math.Abs(p.First - p.Second));
        Assert.True(worst < tolerance, $"{what}: largest difference {worst:E2}");
    }

    [Theory]
    [InlineData("trocr-tiny")]
    [InlineData("trocr-tiny-vit")]
    public void The_processor_reads_the_line_as_the_PIL_processor_does(string folder)
    {
        // Bicubic for the DeiT processor, bilinear for the ViT one, both squashing the wide line to a square.
        using var model = TrOcrModel.Open(Path.Combine(Fixtures, folder));
        var pixels = model.Processor.Read(Path.Combine(Fixtures, "trocr-line.png"));

        Close(Values(Reference.GetProperty(folder).GetProperty("pixels")), pixels.AsContiguous().ToArray(), 1e-12, "pixel_values");
    }

    [Theory]
    [InlineData("trocr-tiny", 18)]       // [CLS], the distillation token, 16 patches
    [InlineData("trocr-tiny-vit", 17)]
    public void The_encoder_matches_transformers(string folder, int tokens)
    {
        using var model = TrOcrModel.Open(Path.Combine(Fixtures, folder));
        var states = model.EncoderStates(Pixels(folder));

        Assert.Equal(tokens, states.Shape[0]);
        Close(Values(Reference.GetProperty(folder).GetProperty("encoder")), states.AsContiguous().ToArray(), 1e-12, "encoder last_hidden_state");
    }

    [Theory]
    [InlineData("trocr-tiny", 1e-12)]
    [InlineData("trocr-tiny-vit", 1e-6)]   // the sinusoids are float32 in transformers too, rounded by another libm
    public void Decoder_logits_match_transformers_at_every_position(string folder, double tolerance)
    {
        // The decoder input holds a pad token (1): sinusoidal positions skip it, learned ones do not.
        using var model = TrOcrModel.Open(Path.Combine(Fixtures, folder));
        var node = Reference.GetProperty(folder);

        Close(Values(node.GetProperty("logits")), model.Logits(Pixels(folder), Ids(node.GetProperty("decoder_ids"))), tolerance, "logits");
    }

    [Theory]
    [InlineData("trocr-tiny")]
    [InlineData("trocr-tiny-vit")]
    public void Greedy_and_beam_search_generate_transformers_ids(string folder)
    {
        using var model = TrOcrModel.Open(Path.Combine(Fixtures, folder));
        var node = Reference.GetProperty(folder);
        var pixels = Pixels(folder);

        Assert.Equal(Ids(node.GetProperty("greedy")), model.GenerateIds(pixels, new OcrSettings(MaxNewTokens: 12)));
        Assert.Equal(Ids(node.GetProperty("beam")), model.GenerateIds(pixels, new OcrSettings(MaxNewTokens: 8, Beams: 3)));
        Assert.Equal(Ids(node.GetProperty("beam_penalised")), model.GenerateIds(pixels, new OcrSettings(MaxNewTokens: 8, Beams: 4, LengthPenalty: 0.5)));
    }

    [Fact]
    public void The_text_is_transformers_batch_decode()
    {
        using var model = TrOcrModel.Open(Path.Combine(Fixtures, "trocr-tiny"));
        var node = Reference.GetProperty("trocr-tiny");

        Assert.Equal(node.GetProperty("greedy_text").GetString(), model.Recognize(Path.Combine(Fixtures, "trocr-line.png"), new OcrSettings(MaxNewTokens: 12)));
    }

    [Fact]
    public void A_batch_reads_each_image_as_it_reads_it_alone()
    {
        // Sequences that end at different steps leave the batch one by one; the rest must not notice.
        using var model = TrOcrModel.Open(Path.Combine(Fixtures, "trocr-tiny"));
        var first = Pixels("trocr-tiny");
        var second = new NdArray([.. first.AsContiguous().ToArray().Select(v => -v)], 3, 32, 32);
        var third = new NdArray([.. first.AsContiguous().ToArray().Select((v, i) => i % 7 == 0 ? 0.0 : v)], 3, 32, 32);
        var settings = new OcrSettings(MaxNewTokens: 12);

        var batch = model.GenerateIds([first, second, third], settings);

        Assert.Equal(model.GenerateIds(first, settings), batch[0]);
        Assert.Equal(model.GenerateIds(second, settings), batch[1]);
        Assert.Equal(model.GenerateIds(third, settings), batch[2]);
        Assert.Equal([.. batch.Select(model.Decode)], model.RecognizeBatch([first, second, third], settings));
    }

    [Fact]
    public void The_generation_config_sets_the_defaults()
    {
        using var model = TrOcrModel.Open(Path.Combine(Fixtures, "trocr-tiny"));

        Assert.Equal(2, model.Config.DecoderStartTokenId);
        Assert.Equal([2], model.Config.EndTokenIds);
        Assert.Equal(20, model.Config.MaxNewTokens);   // transformers' default when the checkpoint sets none
        Assert.Equal(1, model.Config.Beams);
        Assert.Equal(2, model.Encoder.LeadingTokens);
    }

    [Fact]
    public void A_sentencepiece_model_tokenizes_as_XLMRobertaTokenizer()
    {
        var tokenizer = HfTokenizer.FromSentencePieceModel(Path.Combine(Fixtures, "trocr-tiny", "sentencepiece.bpe.model"));
        var decoded = Reference.GetProperty("tokenizer_decoded");

        foreach (var entry in Reference.GetProperty("tokenizer").EnumerateObject())
        {
            var ids = tokenizer.Encode(entry.Name).Ids;
            Assert.Equal(Ids(entry.Value), ids);
            Assert.Equal(decoded.GetProperty(entry.Name).GetString(), tokenizer.Decode(ids));
        }

        // fairseq's numbering: <s> <pad> </s> <unk> first, <mask> last.
        Assert.Equal(0, tokenizer.TokenToId("<s>"));
        Assert.Equal(1, tokenizer.TokenToId("<pad>"));
        Assert.Equal(2, tokenizer.TokenToId("</s>"));
        Assert.Equal(3, tokenizer.TokenToId("<unk>"));
        Assert.Equal(tokenizer.VocabularySize - 1, tokenizer.TokenToId("<mask>"));
    }

    [Fact]
    public void A_folder_with_only_a_sentencepiece_model_loads_its_tokenizer()
    {
        var tokenizer = HfTokenizer.FromDirectory(Path.Combine(Fixtures, "trocr-tiny"));
        var first = Reference.GetProperty("tokenizer").EnumerateObject().First();

        Assert.Equal(Ids(first.Value), tokenizer.Encode(first.Name).Ids);
    }
}

/// <summary>
/// The published TrOCR checkpoints against transformers on rendered text lines. They download
/// 250 MB and 1.3 GB, so they run only when <c>HFNET_DOWNLOAD_TESTS=1</c>.
/// </summary>
/// <remarks>
/// References from <c>Fixtures/make_trocr.py --real</c>: transformers' float64 outputs, sampled
/// (every 997th encoder value, every 509th logit, the ten largest logits), and the text float32
/// generates, which is what a user of the Python model gets.
/// </remarks>
public sealed class TrOcrPublishedTests(ITestOutputHelper output)
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures");

    private static bool Enabled => Environment.GetEnvironmentVariable("HFNET_DOWNLOAD_TESTS") == "1";

    private static double[] Values(JsonElement node) => [.. node.EnumerateArray().Select(v => v.GetDouble())];

    private static int[] Ids(JsonElement node) => [.. node.EnumerateArray().Select(v => v.GetInt32())];

    private static double Worst(double[] expected, double[] actual) => expected.Zip(actual).Max(p => Math.Abs(p.First - p.Second));

    private static double[] Every(double[] values, int step) => [.. values.Where((_, i) => i % step == 0)];

    [Theory]
    [InlineData("microsoft/trocr-small-printed")]
    [InlineData("microsoft/trocr-base-printed")]
    public void A_published_checkpoint_reads_the_lines_as_transformers_does(string repo)
    {
        Assert.SkipUnless(Enabled, "Downloads the checkpoint; set HFNET_DOWNLOAD_TESTS=1 to run.");

        var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "trocr-printed-reference.json"))).RootElement;
        var lines = reference.GetProperty("lines").EnumerateArray().Select(l => l.GetString()!).ToArray();
        var entries = reference.GetProperty(repo).EnumerateArray().ToArray();

        var clock = Stopwatch.StartNew();
        using var model = TrOcrModel.Load(repo);
        output.WriteLine($"{repo}: loaded in {clock.Elapsed.TotalSeconds:F1} s - {model.Config}");

        for (var i = 0; i < entries.Length; i++)
        {
            var entry = entries[i];
            var path = Path.Combine(Fixtures, "trocr-lines", $"line{i}.png");

            var pixels = model.Processor.Read(path);
            var flat = pixels.AsContiguous().ToArray();
            Assert.True(Math.Abs(entry.GetProperty("pixel_sum").GetDouble() - flat.Sum()) < 1e-6, "pixel sum");
            Assert.True(Worst(Values(entry.GetProperty("pixel_samples")), Every(flat, 1009)) < 1e-12, "pixel samples");

            var encoder = Every(model.EncoderStates(pixels).AsContiguous().ToArray(), 997);
            var encoder64 = Worst(Values(entry.GetProperty("encoder_samples")), encoder);
            var encoder32 = Worst(Values(entry.GetProperty("encoder32_samples")), encoder);

            var logits = model.Logits(pixels, [model.Config.DecoderStartTokenId]);
            var logits64 = Worst(Values(entry.GetProperty("logit_samples")), Every(logits, 509));
            var logits32 = Worst(Values(entry.GetProperty("logit32_samples")), Every(logits, 509));
            var top = Ids(entry.GetProperty("top_ids"));
            var topLogits = Worst(Values(entry.GetProperty("top_logits")), [.. top.Select(t => logits[t])]);
            var ranked = Enumerable.Range(0, logits.Length).OrderByDescending(t => logits[t]).Take(top.Length).ToArray();

            clock.Restart();
            var ids = model.GenerateIds(pixels);
            var greedySeconds = clock.Elapsed.TotalSeconds;
            var text = model.Decode(ids);

            clock.Restart();
            var beam = model.GenerateIds(pixels, new OcrSettings(Beams: 4));
            var beamSeconds = clock.Elapsed.TotalSeconds;

            output.WriteLine(
                $"line {i} \"{lines[i]}\" -> \"{text}\" | encoder vs f64 {encoder64:E1}, vs f32 {encoder32:E1} "
                + $"| first-step logits vs f64 {logits64:E1}, vs f32 {logits32:E1}, top-10 {topLogits:E1} "
                + $"| greedy {greedySeconds:F2} s ({ids.Count - 1} tokens), beam 4 {beamSeconds:F2} s");

            Assert.True(encoder64 < 1e-6, $"encoder vs float64: {encoder64:E2}");
            Assert.True(encoder32 < 1e-2, $"encoder vs float32: {encoder32:E2}");
            Assert.True(logits64 < 1e-6, $"logits vs float64: {logits64:E2}");
            Assert.True(logits32 < 1e-2, $"logits vs float32: {logits32:E2}");
            Assert.Equal(top, ranked);

            Assert.Equal(Ids(entry.GetProperty("ids")), ids);
            Assert.Equal(entry.GetProperty("text").GetString(), text);
            Assert.Equal(Ids(entry.GetProperty("beam4_ids")), beam);
            Assert.Equal(entry.GetProperty("beam4_text").GetString(), model.Decode(beam));
            Assert.Equal(Ids(entry.GetProperty("tokenizer_ids")), model.Tokenizer!.Encode(lines[i]).Ids);
        }

        clock.Restart();
        var batch = model.RecognizeBatch(Enumerable.Range(0, entries.Length).Select(i => Path.Combine(Fixtures, "trocr-lines", $"line{i}.png")));
        output.WriteLine($"batch of {entries.Length}: {clock.Elapsed.TotalSeconds:F2} s");
        Assert.Equal([.. entries.Select(e => e.GetProperty("text").GetString()!)], batch);
    }
}
