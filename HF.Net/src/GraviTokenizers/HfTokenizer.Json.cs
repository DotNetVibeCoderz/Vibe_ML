using System.Text.Json;
using Gravicode.HFNet.GraviTokenizers.Components;

namespace Gravicode.HFNet.GraviTokenizers;

/// <summary>
/// The <c>tokenizer.json</c> reader: each stage of the file maps onto one component of the
/// pipeline.
/// </summary>
/// <remarks>
/// Unknown component types are refused rather than skipped. A tokenizer missing its normalizer
/// still runs and still produces ids, and the ids are wrong in a way nothing downstream can see -
/// so the failure has to happen here, where the cause is still visible.
/// </remarks>
public sealed partial class HfTokenizer
{
    private static IReadOnlyList<AddedToken> ReadAddedTokens(JsonElement root)
    {
        if (!root.TryGetProperty("added_tokens", out var added) || added.ValueKind != JsonValueKind.Array)
        {
            return [];
        }

        var tokens = new List<AddedToken>();
        foreach (var entry in added.EnumerateArray())
        {
            tokens.Add(new AddedToken(
                entry.GetProperty("id").GetInt32(),
                entry.GetProperty("content").GetString() ?? "",
                entry.TryGetProperty("special", out var special) && special.ValueKind == JsonValueKind.True,
                entry.TryGetProperty("normalized", out var normalized) && normalized.ValueKind == JsonValueKind.True,
                entry.TryGetProperty("lstrip", out var lstrip) && lstrip.ValueKind == JsonValueKind.True,
                entry.TryGetProperty("rstrip", out var rstrip) && rstrip.ValueKind == JsonValueKind.True));
        }

        return tokens;
    }

    private static ITokenizerModel ReadModel(JsonElement model)
    {
        // Older tokenizer.json files - GPT-2's and XLM-RoBERTa's among them - leave out "type", and
        // the reference infers it from the fields: merges mean BPE, an array vocabulary means
        // Unigram. Defaulting to WordPiece instead ran GPT-2 as greedy longest-match, which gets
        // whole common words right and splits the rest wrongly ("emoji" as emo+ji, not em+oji).
        var type = model.TryGetProperty("type", out var kind) && kind.ValueKind == JsonValueKind.String
            ? kind.GetString()
            : model.TryGetProperty("merges", out _) ? "BPE"
            : model.TryGetProperty("vocab", out var vocab) && vocab.ValueKind == JsonValueKind.Array ? "Unigram"
            : "WordPiece";

        return type switch
        {
            "WordPiece" => new WordPieceModel(
                ReadVocabulary(model.GetProperty("vocab")),
                model.TryGetProperty("unk_token", out var unk) ? unk.GetString() ?? "[UNK]" : "[UNK]",
                model.TryGetProperty("continuing_subword_prefix", out var prefix)
                    ? prefix.GetString() ?? "##" : "##",
                model.TryGetProperty("max_input_chars_per_word", out var max) ? max.GetInt32() : 100),

            "BPE" => new BpeModel(
                ReadVocabulary(model.GetProperty("vocab")),
                ReadMerges(model),
                model.TryGetProperty("unk_token", out var bpeUnk) && bpeUnk.ValueKind == JsonValueKind.String
                    ? bpeUnk.GetString() : null,
                model.TryGetProperty("continuing_subword_prefix", out var bpePrefix)
                    && bpePrefix.ValueKind == JsonValueKind.String ? bpePrefix.GetString() ?? "" : "",
                model.TryGetProperty("end_of_word_suffix", out var suffix)
                    && suffix.ValueKind == JsonValueKind.String ? suffix.GetString() ?? "" : "",
                Flag(model, "byte_fallback"),
                Flag(model, "fuse_unk"),
                Flag(model, "ignore_merges")),

            "Unigram" => ReadUnigram(model),

            _ => throw new NotSupportedException(
                $"Tokenizer model '{type}' is not supported. WordPiece, BPE and Unigram are."),
        };
    }

    private static bool Flag(JsonElement node, string name)
        => node.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.True;

    private static Dictionary<string, int> ReadVocabulary(JsonElement vocabulary)
    {
        var result = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var entry in vocabulary.EnumerateObject()) result[entry.Name] = entry.Value.GetInt32();
        return result;
    }

    /// <summary>
    /// Reads the merge list, which appears either as <c>"a b"</c> strings or as two-element arrays.
    /// </summary>
    /// <remarks>
    /// The array spelling arrived with tokenizers 0.20. Both forms are in the wild, and a reader
    /// that handles only strings throws on every newly-exported GPT-2 style tokenizer.
    /// </remarks>
    private static List<(string, string)> ReadMerges(JsonElement model)
    {
        var merges = new List<(string, string)>();
        if (!model.TryGetProperty("merges", out var list) || list.ValueKind != JsonValueKind.Array) return merges;

        foreach (var entry in list.EnumerateArray())
        {
            if (entry.ValueKind == JsonValueKind.Array && entry.GetArrayLength() == 2)
            {
                merges.Add((entry[0].GetString() ?? "", entry[1].GetString() ?? ""));
                continue;
            }

            var text = entry.GetString() ?? "";
            var space = text.IndexOf(' ');
            if (space > 0) merges.Add((text[..space], text[(space + 1)..]));
        }

        return merges;
    }

    private static UnigramModel ReadUnigram(JsonElement model)
    {
        var pieces = new List<(string, double)>();

        foreach (var entry in model.GetProperty("vocab").EnumerateArray())
        {
            pieces.Add((entry[0].GetString() ?? "", entry[1].GetDouble()));
        }

        var unknownId = model.TryGetProperty("unk_id", out var unk) && unk.ValueKind == JsonValueKind.Number
            ? unk.GetInt32()
            : 0;

        return new UnigramModel(pieces, unknownId);
    }

    private static INormalizer? ReadNormalizer(JsonElement root, string property)
    {
        if (!root.TryGetProperty(property, out var node) || node.ValueKind != JsonValueKind.Object) return null;
        return ParseNormalizer(node);
    }

    private static INormalizer? ParseNormalizer(JsonElement node)
    {
        var type = node.TryGetProperty("type", out var kind) ? kind.GetString() : null;

        switch (type)
        {
            case "Sequence":
            {
                var steps = new List<INormalizer>();
                foreach (var child in node.GetProperty("normalizers").EnumerateArray())
                {
                    if (ParseNormalizer(child) is { } step) steps.Add(step);
                }
                return steps.Count == 0 ? null : new SequenceNormalizer(steps);
            }

            case "BertNormalizer":
                return new BertNormalizer(
                    !node.TryGetProperty("lowercase", out var lower) || lower.ValueKind != JsonValueKind.False,
                    node.TryGetProperty("strip_accents", out var strip) && strip.ValueKind != JsonValueKind.Null
                        ? strip.ValueKind == JsonValueKind.True
                        : null);

            case "Lowercase":
                return new LowercaseNormalizer();

            case "Strip":
                return new StripNormalizer(
                    !node.TryGetProperty("strip_left", out var left) || left.ValueKind != JsonValueKind.False,
                    !node.TryGetProperty("strip_right", out var right) || right.ValueKind != JsonValueKind.False);

            case "Replace":
                return ParseReplace(node);

            case "Prepend":
                return new PrependNormalizer(node.TryGetProperty("prepend", out var prepend) ? prepend.GetString() ?? "" : "");

            // SentencePiece models carry their normalization as a table, which is read exactly.
            case "Precompiled":
                return node.TryGetProperty("precompiled_charsmap", out var map)
                    && map.ValueKind == JsonValueKind.String && map.GetString() is { Length: > 0 } charsmap
                    ? new PrecompiledNormalizer(charsmap)
                    : null;

            // Unicode normalization from HF.Net's own tables: String.Normalize is a no-op under
            // InvariantGlobalization, which left "e" + U+0301 uncomposed where Qwen2's NFC composes it.
            case "NFC":
                return new Unicode.UnicodeNormalizer(Unicode.NormalizationForm.C);
            case "NFD":
                return new Unicode.UnicodeNormalizer(Unicode.NormalizationForm.D);
            case "NFKC":
                return new Unicode.UnicodeNormalizer(Unicode.NormalizationForm.KC);
            case "NFKD":
                return new Unicode.UnicodeNormalizer(Unicode.NormalizationForm.KD);

            // Nmt only drops control characters these inputs rarely have.
            case "Nmt":
                return null;

            case "StripAccents":
                return new BertNormalizer(Lowercase: false, StripAccents: true);

            case null:
                return null;

            default:
                throw new NotSupportedException($"Normalizer '{type}' is not supported.");
        }
    }

    private static INormalizer ParseReplace(JsonElement node)
    {
        var pattern = node.GetProperty("pattern");
        var replacement = node.TryGetProperty("content", out var content) ? content.GetString() ?? "" : "";

        if (pattern.TryGetProperty("Regex", out var regex))
        {
            return new ReplaceNormalizer(regex.GetString() ?? "", replacement, IsRegex: true);
        }

        return new ReplaceNormalizer(
            pattern.TryGetProperty("String", out var literal) ? literal.GetString() ?? "" : "",
            replacement);
    }

    private static IPreTokenizer ReadPreTokenizer(JsonElement root, string property)
    {
        if (!root.TryGetProperty(property, out var node) || node.ValueKind != JsonValueKind.Object)
        {
            // No pre-tokenizer means the model is applied to the whole input, as the reference does.
            // Llama 2 relies on it: its spaces are "▁" by then, and splitting on whitespace instead
            // lost newlines and runs of spaces the model encodes.
            return new WholeTextPreTokenizer();
        }

        return ParsePreTokenizer(node);
    }

    private static IPreTokenizer ParsePreTokenizer(JsonElement node)
    {
        var type = node.TryGetProperty("type", out var kind) ? kind.GetString() : null;

        switch (type)
        {
            case "Sequence":
            {
                var steps = new List<IPreTokenizer>();
                foreach (var child in node.GetProperty("pretokenizers").EnumerateArray())
                {
                    steps.Add(ParsePreTokenizer(child));
                }
                return new SequencePreTokenizer(steps);
            }

            case "BertPreTokenizer":
                return new BertPreTokenizer();

            case "Whitespace" or "WhitespaceSplit":
                return new WhitespacePreTokenizer();

            case "Punctuation":
                return new PunctuationPreTokenizer();

            case "ByteLevel":
                return new ByteLevelPreTokenizer(
                    !node.TryGetProperty("add_prefix_space", out var prefix)
                    || prefix.ValueKind != JsonValueKind.False,
                    !node.TryGetProperty("use_regex", out var useRegex) || useRegex.ValueKind != JsonValueKind.False);

            case "Metaspace":
                return new MetaspacePreTokenizer(
                    node.TryGetProperty("replacement", out var replacement)
                        && (replacement.GetString() ?? "▁").Length > 0
                        ? (replacement.GetString() ?? "▁")[0] : '▁',
                    // Older files say add_prefix_space; newer ones prepend_scheme ("always",
                    // "first", "never").
                    node.TryGetProperty("prepend_scheme", out var scheme) && scheme.ValueKind == JsonValueKind.String
                        ? scheme.GetString() != "never"
                        : !node.TryGetProperty("add_prefix_space", out var addPrefix) || addPrefix.ValueKind != JsonValueKind.False,
                    !node.TryGetProperty("split", out var split) || split.ValueKind != JsonValueKind.False);

            case "Split":
                return ParseSplit(node);

            // Digits are isolated, not removed: one piece per digit with individual_digits, one per run
            // otherwise.
            case "Digits":
                return new SplitPreTokenizer(
                    Flag(node, "individual_digits") ? @"\d" : @"\d+", Invert: false, Behavior: SplitBehavior.Isolated);

            case null:
                return new WhitespacePreTokenizer();

            default:
                throw new NotSupportedException($"Pre-tokenizer '{type}' is not supported.");
        }
    }

    private static IPreTokenizer ParseSplit(JsonElement node)
    {
        var pattern = node.GetProperty("pattern");
        var expression = pattern.TryGetProperty("Regex", out var regex)
            ? regex.GetString() ?? ""
            : System.Text.RegularExpressions.Regex.Escape(
                pattern.TryGetProperty("String", out var literal) ? literal.GetString() ?? "" : "");

        // "invert" says which spans are delimiters, "behavior" what happens to them. Both matter:
        // CLIP is Removed + invert, which keeps the words; reading behavior alone keeps the spaces.
        var invert = node.TryGetProperty("invert", out var flag) && flag.ValueKind == JsonValueKind.True;
        var behavior = (node.TryGetProperty("behavior", out var mode) ? mode.GetString() : "Removed") switch
        {
            "Isolated" => SplitBehavior.Isolated,
            "MergedWithPrevious" => SplitBehavior.MergedWithPrevious,
            "MergedWithNext" => SplitBehavior.MergedWithNext,
            "Contiguous" => SplitBehavior.Contiguous,
            "Removed" or null => SplitBehavior.Removed,
            var other => throw new NotSupportedException($"Split behavior '{other}' is not supported."),
        };

        return new SplitPreTokenizer(expression, invert, behavior);
    }

    private static IDecoder ReadDecoder(JsonElement root, string property)
    {
        if (!root.TryGetProperty(property, out var node) || node.ValueKind != JsonValueKind.Object)
        {
            return new SequenceDecoder([new WhitespaceStep()]);
        }

        return new SequenceDecoder(DecoderSteps(node));
    }

    /// <summary>The decoder as a chain of steps, a sequence flattened in order.</summary>
    private static List<IDecoderStep> DecoderSteps(JsonElement node)
    {
        var type = node.TryGetProperty("type", out var kind) ? kind.GetString() : null;

        switch (type)
        {
            case "Sequence":
                var steps = new List<IDecoderStep>();
                if (node.TryGetProperty("decoders", out var children) && children.ValueKind == JsonValueKind.Array)
                {
                    foreach (var child in children.EnumerateArray()) steps.AddRange(DecoderSteps(child));
                }

                return steps;

            case "WordPiece":
                return [new WordPieceStep(
                    node.TryGetProperty("prefix", out var prefix) ? prefix.GetString() ?? "##" : "##",
                    !node.TryGetProperty("cleanup", out var cleanup) || cleanup.ValueKind != JsonValueKind.False)];

            case "ByteLevel":
                return [new ByteLevelStep()];

            case "Metaspace":
                return [new MetaspaceStep(
                    node.TryGetProperty("replacement", out var replacement) && (replacement.GetString() ?? "▁").Length > 0
                        ? (replacement.GetString() ?? "▁")[0] : '▁',
                    node.TryGetProperty("prepend_scheme", out var scheme) && scheme.ValueKind == JsonValueKind.String
                        ? scheme.GetString() != "never"
                        : !node.TryGetProperty("add_prefix_space", out var addPrefix) || addPrefix.ValueKind != JsonValueKind.False)];

            case "Replace":
            {
                var content = node.TryGetProperty("content", out var c) ? c.GetString() ?? "" : "";
                var pattern = node.GetProperty("pattern");
                return pattern.TryGetProperty("Regex", out var regex)
                    ? [new ReplaceStep(regex.GetString() ?? "", content, IsRegex: true)]
                    : [new ReplaceStep(pattern.TryGetProperty("String", out var literal) ? literal.GetString() ?? "" : "", content)];
            }

            case "ByteFallback":
                return [new ByteFallbackStep()];

            case "Fuse":
                return [new FuseStep()];

            case "Strip":
                return [new StripStep(
                    node.TryGetProperty("content", out var strip) && (strip.GetString() ?? " ").Length > 0 ? (strip.GetString() ?? " ")[0] : ' ',
                    node.TryGetProperty("start", out var start) ? start.GetInt32() : 0,
                    node.TryGetProperty("stop", out var stop) ? stop.GetInt32() : 0)];

            default:
                throw new NotSupportedException(
                    $"Decoder '{type}' is not supported. WordPiece, ByteLevel, Metaspace, Replace, ByteFallback, "
                    + "Fuse, Strip and sequences of them are.");
        }
    }

    private static PostProcessor ReadPostProcessor(JsonElement root, string property)
    {
        if (!root.TryGetProperty(property, out var node) || node.ValueKind != JsonValueKind.Object)
        {
            return PostProcessor.None;
        }

        var type = node.TryGetProperty("type", out var kind) ? kind.GetString() : null;

        switch (type)
        {
            case "TemplateProcessing":
                return ReadTemplate(node);

            case "BertProcessing":
            {
                var (clsToken, clsId) = ReadTokenPair(node, "cls");
                var (sepToken, sepId) = ReadTokenPair(node, "sep");

                return new PostProcessor(
                    [clsToken, "$A", sepToken],
                    [clsToken, "$A", sepToken, "$B", sepToken],
                    new Dictionary<string, int>(StringComparer.Ordinal)
                    {
                        [clsToken] = clsId,
                        [sepToken] = sepId,
                    });
            }

            case "RobertaProcessing":
            {
                var (clsToken, clsId) = ReadTokenPair(node, "cls");
                var (sepToken, sepId) = ReadTokenPair(node, "sep");

                // RoBERTa doubles the separator between a pair. It is not decoration: the model was
                // trained with it, and a single separator shifts the second segment by one.
                return new PostProcessor(
                    [clsToken, "$A", sepToken],
                    [clsToken, "$A", sepToken, sepToken, "$B", sepToken],
                    new Dictionary<string, int>(StringComparer.Ordinal)
                    {
                        [clsToken] = clsId,
                        [sepToken] = sepId,
                    });
            }

            case "Sequence":
            {
                foreach (var child in node.GetProperty("processors").EnumerateArray())
                {
                    var childType = child.TryGetProperty("type", out var childKind) ? childKind.GetString() : null;
                    if (childType is "TemplateProcessing") return ReadTemplate(child);
                }
                return PostProcessor.None;
            }

            default:
                return PostProcessor.None;
        }
    }

    private static (string Token, int Id) ReadTokenPair(JsonElement node, string property)
    {
        var pair = node.GetProperty(property);
        return (pair[0].GetString() ?? "", pair[1].GetInt32());
    }

    private static PostProcessor ReadTemplate(JsonElement node)
    {
        var specials = new Dictionary<string, int>(StringComparer.Ordinal);

        if (node.TryGetProperty("special_tokens", out var tokens) && tokens.ValueKind == JsonValueKind.Object)
        {
            foreach (var entry in tokens.EnumerateObject())
            {
                if (entry.Value.TryGetProperty("ids", out var ids) && ids.GetArrayLength() > 0)
                {
                    specials[entry.Name] = ids[0].GetInt32();
                }
            }
        }

        return new PostProcessor(ReadSlots(node, "single"), ReadSlots(node, "pair"), specials);
    }

    private static IReadOnlyList<string> ReadSlots(JsonElement node, string property)
    {
        if (!node.TryGetProperty(property, out var list) || list.ValueKind != JsonValueKind.Array)
        {
            return property == "single" ? ["$A"] : ["$A", "$B"];
        }

        var slots = new List<string>();
        foreach (var entry in list.EnumerateArray())
        {
            if (entry.TryGetProperty("SpecialToken", out var special))
            {
                slots.Add(special.GetProperty("id").GetString() ?? "");
            }
            else if (entry.TryGetProperty("Sequence", out var sequence))
            {
                slots.Add("$" + (sequence.GetProperty("id").GetString() ?? "A"));
            }
        }

        return slots.Count == 0 ? (property == "single" ? ["$A"] : ["$A", "$B"]) : slots;
    }
}
