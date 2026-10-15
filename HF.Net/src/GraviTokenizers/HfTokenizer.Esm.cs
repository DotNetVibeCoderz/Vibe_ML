using System.Text.Json;
using Gravicode.HFNet.GraviTokenizers.Components;

namespace Gravicode.HFNet.GraviTokenizers;

public sealed partial class HfTokenizer
{
    /// <summary>The special tokens of transformers' <c>EsmTokenizer</c>, in the order it declares them.</summary>
    private static readonly string[] EsmSpecials = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"];

    /// <summary>
    /// Builds transformers' <c>EsmTokenizer</c> from its <c>vocab.txt</c>: the tokenizer of ESM-1b,
    /// ESM-1v, ESM-2 and the Nucleotide Transformer.
    /// </summary>
    /// <param name="vocabPath">One token per line, the line number being the id.</param>
    /// <param name="appendEos">
    /// Whether <c>&lt;eos&gt;</c> closes every sequence. True for every ESM protein model; a
    /// <c>tokenizer_config.json</c> with <c>"eos_token": null</c> turns it off, and
    /// <see cref="FromDirectory"/> reads that.
    /// </param>
    /// <remarks>
    /// <para>
    /// Read as a BERT <c>vocab.txt</c> - which is what happened before 0.11 - a protein is one
    /// "word" with no <c>##</c> pieces and becomes a single <c>[UNK]</c>. Here every vocabulary entry
    /// is matched verbatim, longest first (<see cref="LongestMatchModel"/>), whitespace is ignored, and
    /// <c>&lt;cls&gt; $A &lt;eos&gt;</c> wraps the result: <c>"MKT"</c> is <c>0 20 15 11 2</c> on
    /// ESM-2, as in Python.
    /// </para>
    /// <para>
    /// <c>&lt;cls&gt; &lt;pad&gt; &lt;eos&gt; &lt;unk&gt; &lt;mask&gt;</c> are recognised anywhere in the
    /// text and are special, so decoding with <c>skipSpecialTokens</c> drops them - including an
    /// unknown residue, as the reference does. Decoded tokens are joined with spaces, again as the
    /// reference joins them.
    /// </para>
    /// </remarks>
    public static HfTokenizer FromEsmVocabulary(string vocabPath, bool appendEos = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(vocabPath);

        // transformers strips each line and keeps every one, so a blank line still takes an id.
        var lines = File.ReadAllLines(vocabPath).Select(l => l.Trim()).ToList();
        while (lines.Count > 0 && lines[^1].Length == 0) lines.RemoveAt(lines.Count - 1);

        if (lines.Count == 0) throw new InvalidDataException($"'{vocabPath}' is empty; an ESM vocabulary lists one token per line.");

        var model = new LongestMatchModel(lines);
        var ids = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < lines.Count; i++) ids.TryAdd(lines[i], i);

        var special = new List<AddedToken>();
        foreach (var token in EsmSpecials)
        {
            if (ids.TryGetValue(token, out var id)) special.Add(new AddedToken(id, token, Special: true));
        }

        if (!ids.TryGetValue("<cls>", out var cls))
        {
            throw new InvalidDataException($"'{vocabPath}' has no <cls>; it is not an ESM vocabulary. Use FromBertVocabulary for a BERT vocab.txt.");
        }

        var tokens = new Dictionary<string, int>(StringComparer.Ordinal) { ["<cls>"] = cls };
        PostProcessor post;
        if (appendEos && ids.TryGetValue("<eos>", out var eos))
        {
            tokens["<eos>"] = eos;
            post = new PostProcessor(["<cls>", "$A", "<eos>"], ["<cls>", "$A", "<eos>", "$B", "<eos>"], tokens);
        }
        else
        {
            // The reference refuses a pair without <eos>; with nothing to separate them, they abut.
            post = new PostProcessor(["<cls>", "$A"], ["<cls>", "$A", "$B"], tokens);
        }

        return new HfTokenizer(model, new WhitespacePreTokenizer(), normalizer: null, new WhitespaceDecoder(), post, special);
    }

    /// <summary>
    /// Whether a <c>vocab.txt</c> should be read as ESM's: the config names <c>EsmTokenizer</c>, or,
    /// with no class named, the file holds <c>&lt;cls&gt;</c> and <c>&lt;eos&gt;</c> and no <c>[CLS]</c>.
    /// </summary>
    internal static bool IsEsm(string? configPath, string vocabPath)
    {
        if (IsTokenizerClass(configPath, "EsmTokenizer")) return true;
        if (ReadTokenizerClass(configPath) is not null) return false;

        var lines = File.ReadLines(vocabPath).Take(64).Select(l => l.Trim()).ToHashSet(StringComparer.Ordinal);
        return lines.Contains("<cls>") && lines.Contains("<eos>") && !lines.Contains("[CLS]");
    }

    /// <summary>False when <c>tokenizer_config.json</c> sets <c>"eos_token": null</c>.</summary>
    internal static bool ReadAppendsEos(string? configPath)
    {
        if (configPath is null || !File.Exists(configPath)) return true;

        try
        {
            using var document = JsonDocument.Parse(File.ReadAllText(configPath));
            return !(document.RootElement.TryGetProperty("eos_token", out var eos) && eos.ValueKind == JsonValueKind.Null);
        }
        catch (JsonException)
        {
            return true;
        }
    }
}
