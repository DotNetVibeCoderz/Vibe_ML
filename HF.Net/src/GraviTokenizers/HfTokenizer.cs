using System.Text.Json;
using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviTokenizers.Components;

namespace Gravicode.HFNet.GraviTokenizers;

/// <summary>
/// A tokenizer that reproduces what a Hugging Face model was trained with, loaded either from a
/// <c>tokenizer.json</c> or from the older vocabulary files.
/// </summary>
/// <remarks>
/// <para>
/// The pipeline is the same four stages the reference implementation uses: normalize,
/// pre-tokenize, apply the subword model, then post-process to add special tokens. Keeping the
/// stages separate is what lets one class serve BERT, RoBERTa and SentencePiece models rather than
/// needing one tokenizer class per family.
/// </para>
/// <para>
/// Normalization runs per pre-token rather than over the whole string, so character offsets stay
/// anchored to the untouched input. That is what makes <see cref="Encoding.Span(string, int, int)"/>
/// return a real substring rather than an approximation.
/// </para>
/// </remarks>
public sealed partial class HfTokenizer
{
    private readonly INormalizer? _normalizer;
    private readonly IPreTokenizer _preTokenizer;
    private readonly ITokenizerModel _model;
    private readonly IDecoder _decoder;
    private readonly PostProcessor _postProcessor;
    private readonly Dictionary<string, AddedToken> _addedTokens;
    private readonly HashSet<int> _specialIds;

    /// <summary>Assembles a tokenizer from its stages.</summary>
    /// <param name="model">The subword model.</param>
    /// <param name="preTokenizer">How text is split before the model sees it.</param>
    /// <param name="normalizer">Per-piece normalization, or null for none.</param>
    /// <param name="decoder">How pieces are rejoined, or null for whitespace joining.</param>
    /// <param name="postProcessor">Which special tokens are inserted, or null for none.</param>
    /// <param name="addedTokens">Tokens recognised verbatim ahead of the model.</param>
    public HfTokenizer(
        ITokenizerModel model,
        IPreTokenizer preTokenizer,
        INormalizer? normalizer = null,
        IDecoder? decoder = null,
        PostProcessor? postProcessor = null,
        IReadOnlyList<AddedToken>? addedTokens = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _preTokenizer = preTokenizer ?? throw new ArgumentNullException(nameof(preTokenizer));
        _normalizer = normalizer;
        _decoder = decoder ?? new WhitespaceDecoder();
        _postProcessor = postProcessor ?? PostProcessor.None;

        _addedTokens = (addedTokens ?? []).ToDictionary(t => t.Content, StringComparer.Ordinal);
        _specialIds = [.. (addedTokens ?? []).Where(t => t.Special).Select(t => t.Id)];

        foreach (var id in _postProcessor.SpecialTokens.Values) _specialIds.Add(id);

        PadId = ResolveId("[PAD]", "<pad>");
        UnknownId = ResolveId("[UNK]", "<unk>");
        ClassifierId = ResolveId("[CLS]", "<s>");
        SeparatorId = ResolveId("[SEP]", "</s>");
        MaskId = ResolveId("[MASK]", "<mask>");
    }

    /// <summary>Number of entries in the vocabulary.</summary>
    public int VocabularySize => _model.VocabularySize;

    /// <summary>The padding id, or -1 when the vocabulary has none.</summary>
    public int PadId { get; }

    /// <summary>The unknown id, or -1 when the vocabulary has none.</summary>
    public int UnknownId { get; }

    /// <summary>The sequence-start id, or -1 when the vocabulary has none.</summary>
    public int ClassifierId { get; }

    /// <summary>The separator id, or -1 when the vocabulary has none.</summary>
    public int SeparatorId { get; }

    /// <summary>The mask id, or -1 when the vocabulary has none.</summary>
    public int MaskId { get; }

    // ------------------------------------------------------------------ loading

    /// <summary>
    /// Downloads a repository's tokenizer from the Hub and loads it.
    /// </summary>
    /// <param name="repoId">A model id such as <c>bert-base-uncased</c>.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    /// <remarks>
    /// Three layouts are tried in turn, because the Hub has all three in active use:
    /// <c>tokenizer.json</c> (the fast format), then <c>vocab.json</c> with <c>merges.txt</c>
    /// (GPT-2 style), then a bare <c>vocab.txt</c> (original BERT). A loader that only understands
    /// the first cannot open a large share of the older and smaller models.
    /// </remarks>
    public static HfTokenizer FromPretrained(string repoId, string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var info = Hub.ModelInfo(repoId, revision);
        var available = info.Files.Select(f => f.Path).ToHashSet(StringComparer.Ordinal);

        if (available.Contains("tokenizer.json"))
        {
            return Load(Hub.DownloadFile(repoId, "tokenizer.json", revision));
        }

        if (available.Contains("vocab.json") && available.Contains("merges.txt"))
        {
            return FromGpt2Files(
                Hub.DownloadFile(repoId, "vocab.json", revision),
                Hub.DownloadFile(repoId, "merges.txt", revision));
        }

        if (available.Contains("vocab.txt"))
        {
            var lowercase = true;
            if (available.Contains("tokenizer_config.json"))
            {
                lowercase = ReadLowercaseFlag(Hub.DownloadFile(repoId, "tokenizer_config.json", revision));
            }

            return FromBertVocabulary(Hub.DownloadFile(repoId, "vocab.txt", revision), lowercase);
        }

        throw new HubException(
            $"'{repoId}' has no tokenizer files this loader recognises (tokenizer.json, vocab.json + merges.txt, or vocab.txt).")
        {
            RepoId = repoId,
        };
    }

    /// <summary>Loads a tokenizer from a <c>tokenizer.json</c> file.</summary>
    /// <param name="path">Path to the file.</param>
    public static HfTokenizer Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;

        var addedTokens = ReadAddedTokens(root);
        var model = ReadModel(root.GetProperty("model"));
        var normalizer = ReadNormalizer(root, "normalizer");
        var preTokenizer = ReadPreTokenizer(root, "pre_tokenizer");
        var decoder = ReadDecoder(root, "decoder");
        var postProcessor = ReadPostProcessor(root, "post_processor");

        return new HfTokenizer(model, preTokenizer, normalizer, decoder, postProcessor, addedTokens);
    }

    /// <summary>Builds a BERT-style WordPiece tokenizer from a <c>vocab.txt</c>.</summary>
    /// <param name="vocabPath">One token per line, the line number being the id.</param>
    /// <param name="lowercase">Whether the model was trained on lowercased text.</param>
    public static HfTokenizer FromBertVocabulary(string vocabPath, bool lowercase = true)
    {
        var vocabulary = new Dictionary<string, int>(StringComparer.Ordinal);
        var lines = File.ReadAllLines(vocabPath);

        for (var i = 0; i < lines.Length; i++)
        {
            var token = lines[i].TrimEnd('\r', '\n');
            if (token.Length > 0) vocabulary.TryAdd(token, i);
        }

        var model = new WordPieceModel(vocabulary);
        var added = new List<AddedToken>();

        foreach (var special in (string[])["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        {
            if (vocabulary.TryGetValue(special, out var id)) added.Add(new AddedToken(id, special, Special: true));
        }

        var post = vocabulary.TryGetValue("[CLS]", out var cls) && vocabulary.TryGetValue("[SEP]", out var sep)
            ? PostProcessor.Bert(cls, sep)
            : PostProcessor.None;

        return new HfTokenizer(
            model,
            new BertPreTokenizer(),
            new BertNormalizer(lowercase),
            new WordPieceDecoder(),
            post,
            added);
    }

    /// <summary>Builds a GPT-2 style byte-level BPE tokenizer from its two files.</summary>
    /// <param name="vocabPath">A JSON object mapping piece to id.</param>
    /// <param name="mergesPath">The merge list, one pair per line in priority order.</param>
    public static HfTokenizer FromGpt2Files(string vocabPath, string mergesPath)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(vocabPath));
        var vocabulary = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var entry in document.RootElement.EnumerateObject())
        {
            vocabulary[entry.Name] = entry.Value.GetInt32();
        }

        var merges = new List<(string, string)>();
        foreach (var line in File.ReadLines(mergesPath))
        {
            if (line.StartsWith("#version", StringComparison.Ordinal) || line.Length == 0) continue;

            var space = line.IndexOf(' ');
            if (space > 0) merges.Add((line[..space], line[(space + 1)..]));
        }

        return new HfTokenizer(
            new BpeModel(vocabulary, merges),
            new ByteLevelPreTokenizer(),
            normalizer: null,
            new ByteLevelDecoder(),
            PostProcessor.None);
    }

    // ------------------------------------------------------------------ encoding

    /// <summary>Encodes one text.</summary>
    /// <param name="text">The input.</param>
    /// <param name="addSpecialTokens">Whether the post-processor's tokens are inserted.</param>
    public Encoding Encode(string text, bool addSpecialTokens = true)
    {
        ArgumentNullException.ThrowIfNull(text);

        var pieces = EncodePieces(text);
        return Assemble(pieces, null, addSpecialTokens);
    }

    /// <summary>Encodes a pair of texts as one sequence, as a cross-encoder expects.</summary>
    /// <param name="first">The first sequence.</param>
    /// <param name="second">The second sequence.</param>
    /// <param name="addSpecialTokens">Whether the post-processor's tokens are inserted.</param>
    public Encoding Encode(string first, string second, bool addSpecialTokens = true)
    {
        ArgumentNullException.ThrowIfNull(first);
        ArgumentNullException.ThrowIfNull(second);

        return Assemble(EncodePieces(first), EncodePieces(second), addSpecialTokens);
    }

    /// <summary>Encodes many texts and pads them into one rectangular batch.</summary>
    /// <param name="texts">The inputs.</param>
    /// <param name="settings">Padding and truncation. Defaults to <see cref="BatchOptions.Default"/>.</param>
    /// <param name="addSpecialTokens">Whether the post-processor's tokens are inserted.</param>
    public EncodingBatch EncodeBatch(
        IReadOnlyList<string> texts,
        BatchOptions? settings = null,
        bool addSpecialTokens = true)
    {
        ArgumentNullException.ThrowIfNull(texts);

        // Nullable rather than a defaulted struct: `BatchOptions options = default` would zero the
        // padding strategy, and the caller who wrote nothing would get a ragged batch.
        var options = settings ?? BatchOptions.Default;

        var encodings = new Encoding[texts.Count];
        Parallel.For(0, texts.Count, i => encodings[i] = Encode(texts[i], addSpecialTokens));

        var width = options.Padding switch
        {
            PaddingStrategy.Fixed => options.MaxLength,
            PaddingStrategy.Longest => encodings.Length == 0 ? 0 : encodings.Max(e => e.Length),
            _ => 0,
        };

        if (options.Truncate && width > options.MaxLength) width = options.MaxLength;

        if (options.Padding == PaddingStrategy.None)
        {
            return new EncodingBatch(options.Truncate
                ? [.. encodings.Select(e => Fit(e, Math.Min(e.Length, options.MaxLength)))]
                : encodings);
        }

        return new EncodingBatch([.. encodings.Select(e => Fit(e, width))]);
    }

    /// <summary>Turns ids back into text.</summary>
    /// <param name="ids">The ids to decode.</param>
    /// <param name="skipSpecialTokens">Whether inserted tokens are dropped rather than printed.</param>
    public string Decode(IReadOnlyList<int> ids, bool skipSpecialTokens = true)
    {
        ArgumentNullException.ThrowIfNull(ids);

        var tokens = new List<string>(ids.Count);
        foreach (var id in ids)
        {
            if (skipSpecialTokens && _specialIds.Contains(id)) continue;

            var token = _model.TokenOf(id);
            if (token.Length > 0) tokens.Add(token);
        }

        return _decoder.Decode(tokens);
    }

    /// <summary>The id for a piece, or the unknown id.</summary>
    public int TokenToId(string token)
        => _addedTokens.TryGetValue(token, out var added) ? added.Id : _model.IdOf(token);

    /// <summary>The piece for an id.</summary>
    public string IdToToken(int id) => _model.TokenOf(id);

    // ------------------------------------------------------------------ internals

    /// <summary>Runs normalize, pre-tokenize and the subword model over one text.</summary>
    private List<(int Id, string Token, int Start, int End, bool Special)> EncodePieces(string text)
    {
        var result = new List<(int, string, int, int, bool)>();

        foreach (var segment in SplitOnAddedTokens(text))
        {
            if (segment.Added is { } added)
            {
                result.Add((added.Id, added.Content, segment.Start, segment.End, added.Special));
                continue;
            }

            foreach (var preToken in _preTokenizer.Split(segment.Text))
            {
                var word = _normalizer is null ? preToken.Word : _normalizer.Normalize(preToken.Word);
                if (word.Length == 0) continue;

                var pieces = _model.Tokenize(word);

                // Subword pieces get their own span where the arithmetic is sound: normalization
                // must have preserved length, and the pieces' surface forms must add back up to the
                // word. Lowercasing satisfies both; accent folding and the byte alphabet do not, and
                // there each piece honestly reports the whole word rather than a span that would be
                // off by a few characters in a way no caller could detect.
                var surfaces = new string[pieces.Count];
                var total = 0;
                for (var i = 0; i < pieces.Count; i++)
                {
                    surfaces[i] = Surface(pieces[i]);
                    total += surfaces[i].Length;
                }

                var aligned = word.Length == preToken.Word.Length && total == preToken.Word.Length;
                var cursor = segment.Start + preToken.Start;

                for (var i = 0; i < pieces.Count; i++)
                {
                    var start = aligned ? cursor : segment.Start + preToken.Start;
                    var end = aligned ? cursor + surfaces[i].Length : segment.Start + preToken.End;
                    cursor = end;

                    result.Add((_model.IdOf(pieces[i]), pieces[i], start, end, false));
                }
            }
        }

        return result;
    }

    /// <summary>The text a piece contributes, with any continuation marker removed.</summary>
    private string Surface(string piece) => _model switch
    {
        WordPieceModel wordPiece
            when wordPiece.ContinuingPrefix.Length > 0
                && piece.Length > wordPiece.ContinuingPrefix.Length
                && piece.StartsWith(wordPiece.ContinuingPrefix, StringComparison.Ordinal)
            => piece[wordPiece.ContinuingPrefix.Length..],

        _ => piece,
    };

    /// <summary>
    /// Carves the input around any added tokens, which must be matched verbatim.
    /// </summary>
    /// <remarks>
    /// An added token such as <c>&lt;|endoftext|&gt;</c> has to survive intact. Letting the
    /// pre-tokenizer at it splits it into punctuation and letters and it is never recovered, which
    /// is how a control token silently turns into five ordinary ones.
    /// </remarks>
    private List<(string Text, int Start, int End, AddedToken? Added)> SplitOnAddedTokens(string text)
    {
        if (_addedTokens.Count == 0) return [(text, 0, text.Length, null)];

        var segments = new List<(string, int, int, AddedToken?)>();
        var cursor = 0;

        while (cursor < text.Length)
        {
            var bestIndex = -1;
            AddedToken? bestToken = null;

            foreach (var added in _addedTokens.Values)
            {
                var at = text.IndexOf(added.Content, cursor, StringComparison.Ordinal);
                if (at < 0) continue;

                // Earliest wins; on a tie the longer token wins, so <s> cannot pre-empt <sep>.
                if (bestIndex < 0 || at < bestIndex
                    || (at == bestIndex && added.Content.Length > bestToken!.Value.Content.Length))
                {
                    bestIndex = at;
                    bestToken = added;
                }
            }

            if (bestIndex < 0)
            {
                segments.Add((text[cursor..], cursor, text.Length, null));
                break;
            }

            if (bestIndex > cursor) segments.Add((text[cursor..bestIndex], cursor, bestIndex, null));

            var end = bestIndex + bestToken!.Value.Content.Length;
            segments.Add((bestToken.Value.Content, bestIndex, end, bestToken));
            cursor = end;
        }

        return segments;
    }

    /// <summary>Applies the post-processor's template and builds the final encoding.</summary>
    private Encoding Assemble(
        List<(int Id, string Token, int Start, int End, bool Special)> first,
        List<(int Id, string Token, int Start, int End, bool Special)>? second,
        bool addSpecialTokens)
    {
        var ids = new List<int>();
        var tokens = new List<string>();
        var typeIds = new List<int>();
        var specialMask = new List<int>();
        var offsets = new List<(int, int)>();

        var template = !addSpecialTokens || _postProcessor.IsIdentity
            ? (second is null ? (IReadOnlyList<string>)["$A"] : ["$A", "$B"])
            : second is null ? _postProcessor.SingleTemplate : _postProcessor.PairTemplate;

        foreach (var slot in template)
        {
            switch (slot)
            {
                case "$A":
                    Emit(first, 0);
                    break;

                case "$B":
                    if (second is not null) Emit(second, 1);
                    break;

                default:
                    if (!_postProcessor.SpecialTokens.TryGetValue(slot, out var specialId)) break;

                    ids.Add(specialId);
                    tokens.Add(slot);

                    // A separator between two sequences belongs to the segment it closes, which is
                    // what BERT was trained with; the trailing one after $B is therefore type 1.
                    typeIds.Add(second is not null && ids.Count > first.Count + 2 ? 1 : 0);
                    specialMask.Add(1);
                    offsets.Add((0, 0));
                    break;
            }
        }

        return new Encoding(ids, tokens, [.. ids.Select(_ => 1)], typeIds, specialMask, offsets);

        void Emit(List<(int Id, string Token, int Start, int End, bool Special)> pieces, int typeId)
        {
            foreach (var piece in pieces)
            {
                ids.Add(piece.Id);
                tokens.Add(piece.Token);
                typeIds.Add(typeId);
                specialMask.Add(piece.Special ? 1 : 0);
                offsets.Add((piece.Start, piece.End));
            }
        }
    }

    /// <summary>Pads or truncates an encoding to a fixed width.</summary>
    private Encoding Fit(Encoding encoding, int width)
    {
        if (encoding.Length == width) return encoding;

        if (encoding.Length > width)
        {
            return new Encoding(
                [.. encoding.Ids.Take(width)],
                [.. encoding.Tokens.Take(width)],
                [.. encoding.AttentionMask.Take(width)],
                [.. encoding.TypeIds.Take(width)],
                [.. encoding.SpecialTokensMask.Take(width)],
                [.. encoding.Offsets.Take(width)]);
        }

        var padCount = width - encoding.Length;
        var padId = PadId >= 0 ? PadId : 0;
        var padToken = _model.TokenOf(padId);

        return new Encoding(
            [.. encoding.Ids, .. Enumerable.Repeat(padId, padCount)],
            [.. encoding.Tokens, .. Enumerable.Repeat(padToken, padCount)],
            [.. encoding.AttentionMask, .. Enumerable.Repeat(0, padCount)],
            [.. encoding.TypeIds, .. Enumerable.Repeat(0, padCount)],
            [.. encoding.SpecialTokensMask, .. Enumerable.Repeat(1, padCount)],
            [.. encoding.Offsets, .. Enumerable.Repeat((0, 0), padCount)]);
    }

    private int ResolveId(params string[] candidates)
    {
        foreach (var candidate in candidates)
        {
            if (_addedTokens.TryGetValue(candidate, out var added)) return added.Id;
            if (_postProcessor.SpecialTokens.TryGetValue(candidate, out var special)) return special;
        }

        // Falling through to the model would return the unknown id for a token that is simply
        // absent, and a padding id of "whatever unknown is" corrupts every batch quietly.
        return -1;
    }

    private static bool ReadLowercaseFlag(string configPath)
    {
        try
        {
            using var document = JsonDocument.Parse(File.ReadAllText(configPath));
            return !document.RootElement.TryGetProperty("do_lower_case", out var flag)
                || flag.ValueKind != JsonValueKind.False;
        }
        catch (JsonException)
        {
            return true;
        }
    }
}
