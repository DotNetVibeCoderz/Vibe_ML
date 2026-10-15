using System.Text.Json;
using System.Text.RegularExpressions;
using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviTransformers;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.HFNet.GraviTransformers.Tests;

/// <summary>
/// ESM against transformers' <c>EsmForMaskedLM</c> in float64, on two tiny random checkpoints that
/// between them cover every switch: <c>esm-rotary</c> (ESM-2: rotary positions, token dropout, a
/// contact head) and <c>esm-absolute</c> (ESM-1b: learned positions, a norm before the first layer).
/// </summary>
/// <remarks>
/// References from <c>Fixtures/make_esm.py</c>, with the reference's rotary angles patched to double
/// (transformers computes them in float32 even in a float64 model).
/// </remarks>
public sealed class EsmTests
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures");

    private static readonly JsonElement Reference =
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "esm-reference.json"))).RootElement;

    private static double[] Values(JsonElement node) => [.. node.EnumerateArray().Select(v => v.GetDouble())];

    private static int[] Ints(JsonElement node) => [.. node.EnumerateArray().Select(v => v.GetInt32())];

    private static void Close(double[] expected, double[] actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        var worst = expected.Zip(actual, (e, a) => Math.Abs(e - a)).Max();
        Assert.True(worst < tolerance, $"{what}: largest difference from transformers is {worst:E2} (tolerance {tolerance:E0}).");
    }

    public static TheoryData<string> Layouts => new() { "rotary", "absolute" };

    [Theory]
    [MemberData(nameof(Layouts))]
    public void Hidden_states_match_transformers(string layout)
    {
        using var model = TransformerModel.Open(Path.Combine(Fixtures, $"esm-{layout}"));

        var hidden = model.Hidden(Reference.GetProperty("sequence").GetString()!);
        Close(Values(Reference.GetProperty(layout).GetProperty("hidden")), hidden.ToArray(), 1e-10, $"{layout} hidden states");
    }

    [Theory]
    [MemberData(nameof(Layouts))]
    public void Masked_language_logits_match_transformers(string layout)
    {
        using var model = TransformerModel.Open(Path.Combine(Fixtures, $"esm-{layout}"));
        var expected = Reference.GetProperty(layout);
        var ids = model.Tokenizer.Encode(Reference.GetProperty("masked").GetString()!).ToIdArray();
        var position = expected.GetProperty("mask_position").GetInt32();

        Assert.Equal(model.Tokenizer.MaskId, ids[position]);
        var logits = model.MaskedLanguageLogits(ids, position);
        Close(Values(expected.GetProperty("mask_logits")), logits, 1e-10, $"{layout} logits");

        // FillMask ranks the same logits through a softmax.
        var best = Values(expected.GetProperty("mask_logits")).Select((v, i) => (v, i)).MaxBy(p => p.v).i;
        var fill = model.FillMask(Reference.GetProperty("masked").GetString()!, topK: 1);
        Assert.Equal(model.Tokenizer.IdToToken(best), fill[0].Token);
    }

    [Fact]
    public void Token_dropout_counts_only_real_tokens_and_padding_rows_do_not_leak()
    {
        using var model = TransformerModel.Open(Path.Combine(Fixtures, "esm-rotary"));
        var padded = Reference.GetProperty("padded");
        var ids = Ints(padded.GetProperty("ids"));
        var mask = Ints(padded.GetProperty("mask"));

        var hidden = model.Forward(ids, new int[ids.Length], mask).ToArray();
        var expected = Values(padded.GetProperty("hidden"));
        var real = mask.Sum() * model.Config.HiddenSize;

        // Only the real rows are defined: transformers' pad rows are whatever its kernels leave there.
        Close(expected[..real], hidden[..real], 1e-10, "padded hidden states");
    }

    [Fact]
    public void Contacts_match_transformers_predict_contacts()
    {
        using var model = TransformerModel.Open(Path.Combine(Fixtures, "esm-rotary"));
        var contacts = Reference.GetProperty("contacts");

        Assert.True(model.HasContactHead);
        var predicted = model.PredictContacts(contacts.GetProperty("sequence").GetString()!);

        var size = contacts.GetProperty("size").GetInt32();
        Assert.Equal([size, size], predicted.Shape.ToArray());
        Close(Values(contacts.GetProperty("values")), predicted.ToArray(), 1e-10, "contact probabilities");
    }

    [Fact]
    public void The_checkpoint_loads_completely_whichever_spelling_its_norms_use()
    {
        // transformers 5 saved this one's block norms as gamma/beta and the final norm as weight/bias.
        using var model = TransformerModel.Open(Path.Combine(Fixtures, "esm-rotary"));

        Assert.True(model.Report.IsComplete);
        Assert.Equal("esm.", model.Report.Prefix);
        Assert.Contains("esm.encoder.layer.0.attention.LayerNorm.gamma", model.Report.Loaded);
        Assert.Contains("esm.encoder.emb_layer_norm_after.weight", model.Report.Loaded);
        Assert.Equal("rotary", model.Config.Esm!.PositionEmbeddingType);
        Assert.True(model.HasMaskedLanguageHead);
    }

    [Fact]
    public void ESM_has_no_foundation_encoder_and_says_why()
    {
        using var model = TransformerModel.Open(Path.Combine(Fixtures, "esm-rotary"));

        var error = Assert.Throws<NotSupportedException>(() => model.Encoder);
        Assert.Contains("pre-norm", error.Message);
        Assert.Throws<NotSupportedException>(() => model.RequireFoundationEncoder("LoRA"));

        model.WeightsChanged(); // nothing to rebuild, and no error
    }

    [Theory]
    [InlineData("add_bias_fnn", "false", "SwiGLU")]
    [InlineData("is_folding_model", "true", "ESMFold")]
    [InlineData("position_embedding_type", "\"relative_key\"", "relative_key")]
    public void Unsupported_ESM_variants_are_refused_by_name(string key, string value, string named)
    {
        var folder = CopyFixture("esm-rotary", includeTokenizer: true);
        try
        {
            var config = File.ReadAllText(Path.Combine(folder, "config.json"));
            config = Regex.Replace(config, $"\"{key}\": [^,\\n]+", $"\"{key}\": {value}");
            if (!config.Contains($"\"{key}\"")) config = config.Replace("{", $"{{\n  \"{key}\": {value},");
            File.WriteAllText(Path.Combine(folder, "config.json"), config);

            var error = Assert.Throws<NotSupportedException>(() => TransformerModel.Open(folder));
            Assert.Contains(named, error.Message);
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    [Fact]
    public void A_model_without_a_tokenizer_runs_from_ids()
    {
        var folder = CopyFixture("esm-rotary", includeTokenizer: false);
        try
        {
            using var bare = TransformerModel.Open(folder);
            using var full = TransformerModel.Open(Path.Combine(Fixtures, "esm-rotary"));
            var ids = full.Tokenizer.Encode(Reference.GetProperty("sequence").GetString()!).ToIdArray();

            Assert.False(bare.HasTokenizer);
            Assert.True(full.HasTokenizer);
            Assert.Equal(full.Hidden(ids).ToArray(), bare.Hidden(ids).ToArray());
            Close(Values(Reference.GetProperty("rotary").GetProperty("hidden")), bare.Hidden(ids).ToArray(), 1e-10, "hidden states from ids");

            var masked = full.Tokenizer.Encode(Reference.GetProperty("masked").GetString()!).ToIdArray();
            var position = Reference.GetProperty("rotary").GetProperty("mask_position").GetInt32();
            Close(Values(Reference.GetProperty("rotary").GetProperty("mask_logits")), bare.MaskedLanguageLogits(masked, position), 1e-10, "logits from ids");

            // Anything that needs text says what to do instead.
            var error = Assert.Throws<InvalidOperationException>(() => bare.Hidden("MKT"));
            Assert.Contains("no tokenizer", error.Message);
            Assert.Contains("Hidden(int[])", error.Message);
            Assert.Throws<InvalidOperationException>(() => bare.FillMask("MK<mask>"));
            Assert.Throws<InvalidOperationException>(() => bare.Tokenizer);
            Assert.EndsWith("(no tokenizer)", bare.ToString());
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    [Fact]
    public void A_BERT_model_without_a_tokenizer_opens_too()
    {
        // Geneformer's case: a BertForMaskedLM whose repository has no tokenizer files at all.
        var folder = CopyFixture("bert-small", includeTokenizer: false);
        try
        {
            using var bare = TransformerModel.Open(folder);
            using var full = TransformerModel.Open(Path.Combine(Fixtures, "bert-small"));
            int[] ids = [2, 5, 7, 3];

            Assert.False(bare.HasTokenizer);
            Assert.Equal(full.Hidden(ids).ToArray(), bare.Hidden(ids).ToArray());
            Assert.Throws<InvalidOperationException>(() => bare.Embed("the cat"));
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    [Fact]
    public void Contacts_on_a_model_without_the_head_are_refused()
    {
        using var model = TransformerModel.Open(Path.Combine(Fixtures, "bert-small"));

        Assert.False(model.HasContactHead);
        var error = Assert.Throws<InvalidOperationException>(() => model.PredictContacts([2, 5, 3]));
        Assert.Contains("contact", error.Message);
    }

    /// <summary>Copies a fixture model to a temporary folder, with or without its tokenizer files.</summary>
    private static string CopyFixture(string name, bool includeTokenizer)
    {
        var folder = Directory.CreateTempSubdirectory("hfnet-esm-").FullName;
        foreach (var file in Directory.GetFiles(Path.Combine(Fixtures, name)))
        {
            var fileName = Path.GetFileName(file);
            var isTokenizer = fileName is "vocab.txt" or "tokenizer.json" or "tokenizer_config.json" or "special_tokens_map.json";
            if (includeTokenizer || !isTokenizer) File.Copy(file, Path.Combine(folder, fileName));
        }

        return folder;
    }
}

/// <summary>
/// <c>facebook/esm2_t6_8M_UR50D</c> against transformers in float64 and against the float32 figures
/// Biotech.Net recorded. Downloads about 30 MB, so it runs only with <c>HFNET_DOWNLOAD_TESTS=1</c> or a
/// Hugging Face token available, and skips itself when the Hub cannot be reached.
/// </summary>
/// <remarks>
/// The token comes from <c>HF_TOKEN</c>, or from the file <c>HFNET_HF_TOKEN_FILE</c> names, or from the
/// maintainer's credential file outside the repository; only the <c>hf_...</c> token in it is used and it
/// is never printed. References from <c>Fixtures/make_esm.py --real</c>.
/// </remarks>
public sealed class EsmPublishedTests(ITestOutputHelper output)
{
    private const string Repo = "facebook/esm2_t6_8M_UR50D";

    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures");

    private static double[] Values(JsonElement node) => [.. node.EnumerateArray().Select(v => v.GetDouble())];

    private static string? Token()
    {
        if (Environment.GetEnvironmentVariable("HF_TOKEN") is { Length: > 0 } token) return token;

        var file = Environment.GetEnvironmentVariable("HFNET_HF_TOKEN_FILE") ?? @"C:\Users\mifma\Documents\CodeSandbox\HFToken.txt";
        if (!File.Exists(file)) return null;

        // The file reads "Hugging-Face Access Token: hf_..."; only the token itself is wanted.
        var match = Regex.Match(File.ReadAllText(file), "hf_[A-Za-z0-9]+");
        return match.Success ? match.Value : null;
    }

    private static TransformerModel LoadOrSkip()
    {
        var token = Token();
        var enabled = Environment.GetEnvironmentVariable("HFNET_DOWNLOAD_TESTS") == "1" || token is not null;
        Assert.SkipUnless(enabled, "Downloads facebook/esm2_t6_8M_UR50D; set HFNET_DOWNLOAD_TESTS=1 or HF_TOKEN to run.");

        if (token is not null) Hub.Login(token);

        try
        {
            return TransformerModel.Load(Repo);
        }
        catch (Exception error) when (error is HubException or HttpRequestException or TaskCanceledException or IOException)
        {
            Assert.Skip($"The Hub could not be reached: {error.Message}");
            throw;
        }
    }

    [Fact]
    public void ESM2_t6_matches_transformers_and_the_figures_Biotech_Net_recorded()
    {
        var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "esm2-t6-reference.json"))).RootElement;
        var sequence = reference.GetProperty("sequence").GetString()!;
        using var model = LoadOrSkip();

        Assert.Equal("esm", model.Config.ModelType);
        Assert.True(model.HasMaskedLanguageHead);
        Assert.True(model.HasContactHead);

        // Mean over the residues (not <cls>, not <eos>).
        var hidden = model.Hidden(sequence, maxLength: 1026);
        var rows = hidden.Shape[0];
        var width = hidden.Shape[1];
        Assert.Equal(sequence.Length + 2, rows);
        var pooled = new double[width];
        for (var i = 1; i < rows - 1; i++)
        {
            for (var d = 0; d < width; d++) pooled[d] += hidden[i, d];
        }

        for (var d = 0; d < width; d++) pooled[d] /= rows - 2;

        var expectedPooled = Values(reference.GetProperty("pooled"));
        var worst = expectedPooled.Zip(pooled, (e, a) => Math.Abs(e - a)).Max();
        output.WriteLine($"pooled[0] = {pooled[0]:F6} (transformers float64 {expectedPooled[0]:F6}, Biotech.Net float32 -0.18984); worst {worst:E2}");
        Assert.True(worst < 1e-9, $"Pooled embedding differs from transformers by {worst:E2}.");
        Assert.True(Math.Abs(pooled[0] - -0.18984) < 2e-3, $"pooled[0] = {pooled[0]}, Biotech.Net recorded -0.18984.");
        Assert.True(Math.Abs(pooled[4] - 0.30577) < 2e-3, $"pooled[4] = {pooled[4]}, Biotech.Net recorded 0.30577.");

        var sampled = hidden.ToArray().Where((_, i) => i % 97 == 0).ToArray();
        var hiddenWorst = Values(reference.GetProperty("hidden_every_97th")).Zip(sampled, (e, a) => Math.Abs(e - a)).Max();
        Assert.True(hiddenWorst < 1e-9, $"Hidden states differ from transformers by {hiddenWorst:E2}.");

        // K11P masked marginal: mask residue 11 (token 11, after <cls>), log P(P) - log P(K).
        var ids = model.Tokenizer.Encode(sequence).ToIdArray();
        ids[11] = model.Tokenizer.MaskId;
        var logits = model.MaskedLanguageLogits(ids, 11);
        var max = logits.Max();
        var logSum = max + Math.Log(logits.Sum(v => Math.Exp(v - max)));
        var score = logits[model.Tokenizer.TokenToId("P")] - logits[model.Tokenizer.TokenToId("K")];
        var expectedScore = reference.GetProperty("k11p").GetProperty("score").GetDouble();
        output.WriteLine($"K11P = {score:F6} (transformers {expectedScore:F6}, Biotech.Net -2.0963); log P(K) = {logits[model.Tokenizer.TokenToId("K")] - logSum:F5}");
        Assert.True(Math.Abs(score - expectedScore) < 1e-9, $"K11P = {score}, transformers {expectedScore}.");
        Assert.True(Math.Abs(score - -2.0963) < 2e-3, $"K11P = {score}, Biotech.Net recorded -2.0963.");

        // The first residue, masked, is a methionine - as it is in nearly every protein.
        var fill = model.FillMask("<mask>" + sequence[1..], topK: 5);
        var top = reference.GetProperty("first_residue_top5").EnumerateArray().ToArray();
        Assert.Equal("M", fill[0].Token);
        for (var k = 0; k < top.Length; k++)
        {
            Assert.Equal(top[k].GetProperty("token").GetString(), fill[k].Token);
            Assert.True(Math.Abs(top[k].GetProperty("score").GetDouble() - fill[k].Score) < 1e-9);
        }

        var contacts = model.PredictContacts(sequence);
        Assert.Equal(reference.GetProperty("contacts_size").GetInt32(), contacts.Shape[0]);
        var contactSample = contacts.ToArray().Where((_, i) => i % 31 == 0).ToArray();
        var contactWorst = Values(reference.GetProperty("contacts_every_31st")).Zip(contactSample, (e, a) => Math.Abs(e - a)).Max();
        Assert.True(contactWorst < 1e-9, $"Contacts differ from transformers by {contactWorst:E2}.");
    }
}
