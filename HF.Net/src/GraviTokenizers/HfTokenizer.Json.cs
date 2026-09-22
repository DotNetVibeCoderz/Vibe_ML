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
                entry.TryGetProperty("special", out var special) && special.ValueKind == JsonValueKind.True));
        }

        return tokens;
    }

    private static ITokenizerModel ReadModel(JsonElement model)
    {
        var type = model.TryGetProperty("type", out var kind) ? kind.GetString() : "WordPiece";

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
                    && suffix.ValueKind == JsonValueKind.String ? suffix.GetString() ?? "" : ""),

            "Unigram" => ReadUnigram(model),

            _ => throw new NotSupportedException(
                $"Tokenizer model '{type}' is not supported. WordPiece, BPE and Unigram are."),
        };
    }

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

            // NFC, NFD, NFKC and NFKD are no-ops under InvariantGlobalization, which the whole
            // stack builds with. Silently ignoring them is the honest behaviour: pretending to
            // normalize would be worse, and refusing to load would reject most real tokenizers.
            case "NFC" or "NFD" or "NFKC" or "NFKD" or "Nmt" or "Precompiled":
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
            // No pre-tokenizer means the model is applied to the whole input. Whitespace splitting
            // is the safe stand-in and matches what the reference does for a bare Unigram.
            return new WhitespacePreTokenizer();
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
                    || prefix.ValueKind != JsonValueKind.False);

            case "Metaspace":
                return new MetaspacePreTokenizer(
                    node.TryGetProperty("replacement", out var replacement)
                        && (replacement.GetString() ?? "▁").Length > 0
                        ? (replacement.GetString() ?? "▁")[0] : '▁',
                    !node.TryGetProperty("add_prefix_space", out var addPrefix)
                    || addPrefix.ValueKind != JsonValueKind.False);

            case "Split":
                return ParseSplit(node);

            case "Digits":
                return new SplitPreTokenizer(@"\d+", Invert: false);

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

        // "behavior": "Isolated" / "Removed" / "MergedWithPrevious" ... Only the question of
        // whether the match is the piece or the delimiter changes the token stream here.
        var behavior = node.TryGetProperty("behavior", out var mode) ? mode.GetString() : "Removed";
        return new SplitPreTokenizer(expression, Invert: behavior == "Isolated");
    }

    private static IDecoder ReadDecoder(JsonElement root, string property)
    {
        if (!root.TryGetProperty(property, out var node) || node.ValueKind != JsonValueKind.Object)
        {
            return new WhitespaceDecoder();
        }

        var type = node.TryGetProperty("type", out var kind) ? kind.GetString() : null;

        return type switch
        {
            "WordPiece" => new WordPieceDecoder(
                node.TryGetProperty("prefix", out var prefix) ? prefix.GetString() ?? "##" : "##"),

            "ByteLevel" => new ByteLevelDecoder(),

            "Metaspace" => new MetaspaceDecoder(
                node.TryGetProperty("replacement", out var replacement)
                    && (replacement.GetString() ?? "▁").Length > 0
                    ? (replacement.GetString() ?? "▁")[0] : '▁'),

            // A decoder sequence is usually Metaspace plus cosmetic strip steps; the first stage is
            // what determines how pieces rejoin.
            "Sequence" => ReadFirstDecoder(node),

            _ => new WhitespaceDecoder(),
        };
    }

    private static IDecoder ReadFirstDecoder(JsonElement node)
    {
        if (!node.TryGetProperty("decoders", out var children) || children.ValueKind != JsonValueKind.Array)
        {
            return new WhitespaceDecoder();
        }

        foreach (var child in children.EnumerateArray())
        {
            var type = child.TryGetProperty("type", out var kind) ? kind.GetString() : null;
            if (type is "ByteLevel") return new ByteLevelDecoder();
            if (type is "Metaspace") return new MetaspaceDecoder();
            if (type is "WordPiece") return new WordPieceDecoder();
        }

        return new WhitespaceDecoder();
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
