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
    private readonly Dictionary<int, string> _addedById;

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
        // The reference keeps a "normalized" added token under its normalized form - Llama 2's <s> is
        // "▁<s>" - and that form is both what decoding prints and what the special-token check looks
        // up. So "<s>" is not found, and Llama 2 keeps it in skip-special decoding where TinyLlama
        // (not normalized) and RoBERTa (no normalizer, so the form is unchanged) drop it.
        _addedById = new Dictionary<int, string>();
        _specialIds = [];
        foreach (var token in addedTokens ?? [])
        {
            var stored = token.Normalized && normalizer is not null ? normalizer.Normalize(token.Content) : token.Content;
            _addedById.TryAdd(token.Id, stored);
            if (token.Special && stored == token.Content) _specialIds.Add(token.Id);
        }

        // Template tokens count as special unless the tokenizer's own list says otherwise - a bare
        // vocab.txt has no list at all.
        foreach (var id in _postProcessor.SpecialTokens.Values)
        {
            if (!_addedById.ContainsKey(id)) _specialIds.Add(id);
        }

        PadId = ResolveId("[PAD]", "<pad>");
        UnknownId = ResolveId("[UNK]", "<unk>");
        ClassifierId = ResolveId("[CLS]", "<s>");
        SeparatorId = ResolveId("[SEP]", "</s>");
        MaskId = ResolveId("[MASK]", "<mask>");
    }

    /// <summary>Number of entries in the vocabulary.</summary>
    public int VocabularySize => _model.VocabularySize;

    /// <summary>The padding id, or -1 when the vocabulary has none.</summary>
    public int PadId { get; private set; }

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

        var configPath = available.Contains("tokenizer_config.json")
            ? Hub.DownloadFile(repoId, "tokenizer_config.json", revision)
            : null;

        if (available.Contains("tokenizer.json"))
        {
            return Load(Hub.DownloadFile(repoId, "tokenizer.json", revision)).WithConfiguredPadding(configPath);
        }

        if (available.Contains("vocab.json") && available.Contains("merges.txt"))
        {
            var vocab = Hub.DownloadFile(repoId, "vocab.json", revision);
            var merges = Hub.DownloadFile(repoId, "merges.txt", revision);

            return (IsClip(configPath, vocab) ? FromClipFiles(vocab, merges) : FromGpt2Files(vocab, merges))
                .WithConfiguredPadding(configPath);
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
        return FromJson(File.ReadAllText(path));
    }

    /// <summary>Builds a tokenizer from the text of a <c>tokenizer.json</c>.</summary>
    /// <param name="json">The file's contents.</param>
    public static HfTokenizer FromJson(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);

        using var document = JsonDocument.Parse(json);
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

    /// <summary>Builds CLIP's tokenizer from its <c>vocab.json</c> and <c>merges.txt</c>.</summary>
    /// <param name="vocabPath">A JSON object mapping piece to id.</param>
    /// <param name="mergesPath">The merge list, one pair per line in priority order.</param>
    /// <remarks>
    /// <para>
    /// CLIP's BPE is not GPT-2's, though the two ship the same pair of files. CLIP lowercases,
    /// collapses whitespace, splits words with its own pattern, marks the <i>last</i> piece of each
    /// word with <c>&lt;/w&gt;</c> instead of the first with a space, and wraps the sequence in
    /// <c>&lt;|startoftext|&gt;</c> and <c>&lt;|endoftext|&gt;</c>. Read with the GPT-2 rules,
    /// every prompt comes out as different ids, and a diffusion model conditioned on them draws
    /// something other than what was asked.
    /// </para>
    /// <para>
    /// The pipeline is the one Transformers' <c>CLIPTokenizerFast</c> saves as <c>tokenizer.json</c>,
    /// built here as that document and read by the same loader, so the two cannot drift apart.
    /// Stable Diffusion repositories ship only these two files. Their padding token is set in
    /// <c>tokenizer_config.json</c>; see <see cref="WithPadToken"/>.
    /// </para>
    /// </remarks>
    public static HfTokenizer FromClipFiles(string vocabPath, string mergesPath)
    {
        using var vocabulary = JsonDocument.Parse(File.ReadAllText(vocabPath));
        var ids = vocabulary.RootElement.EnumerateObject().ToDictionary(e => e.Name, e => e.Value.GetInt32(), StringComparer.Ordinal);

        const string Start = "<|startoftext|>";
        const string End = "<|endoftext|>";
        if (!ids.TryGetValue(Start, out var startId) || !ids.TryGetValue(End, out var endId))
        {
            throw new InvalidDataException(
                $"'{vocabPath}' has no {Start} or {End}; it is not a CLIP vocabulary. Use FromGpt2Files for GPT-2 style BPE.");
        }

        var merges = File.ReadLines(mergesPath)
            .Where(line => line.Length > 0 && !line.StartsWith("#version", StringComparison.Ordinal))
            .ToList();

        using var stream = new MemoryStream();
        using (var json = new Utf8JsonWriter(stream))
        {
            json.WriteStartObject();

            json.WriteStartArray("added_tokens");
            foreach (var (content, id) in new[] { (Start, startId), (End, endId) })
            {
                json.WriteStartObject();
                json.WriteNumber("id", id);
                json.WriteString("content", content);
                json.WriteBoolean("special", true);
                json.WriteEndObject();
            }

            json.WriteEndArray();

            json.WriteStartObject("normalizer");
            json.WriteString("type", "Sequence");
            json.WriteStartArray("normalizers");
            json.WriteStartObject(); json.WriteString("type", "NFC"); json.WriteEndObject();
            json.WriteStartObject();
            json.WriteString("type", "Replace");
            json.WriteStartObject("pattern"); json.WriteString("Regex", @"\s+"); json.WriteEndObject();
            json.WriteString("content", " ");
            json.WriteEndObject();
            json.WriteStartObject(); json.WriteString("type", "Lowercase"); json.WriteEndObject();
            json.WriteEndArray();
            json.WriteEndObject();

            json.WriteStartObject("pre_tokenizer");
            json.WriteString("type", "Sequence");
            json.WriteStartArray("pretokenizers");
            json.WriteStartObject();
            json.WriteString("type", "Split");
            json.WriteStartObject("pattern");
            json.WriteString("Regex", @"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+");
            json.WriteEndObject();
            json.WriteString("behavior", "Removed");
            json.WriteBoolean("invert", true);
            json.WriteEndObject();
            json.WriteStartObject();
            json.WriteString("type", "ByteLevel");
            json.WriteBoolean("add_prefix_space", false);
            json.WriteEndObject();
            json.WriteEndArray();
            json.WriteEndObject();

            json.WriteStartObject("post_processor");
            json.WriteString("type", "RobertaProcessing");
            json.WriteStartArray("sep"); json.WriteStringValue(End); json.WriteNumberValue(endId); json.WriteEndArray();
            json.WriteStartArray("cls"); json.WriteStringValue(Start); json.WriteNumberValue(startId); json.WriteEndArray();
            json.WriteEndObject();

            json.WriteStartObject("decoder");
            json.WriteString("type", "ByteLevel");
            json.WriteEndObject();

            json.WriteStartObject("model");
            json.WriteString("type", "BPE");
            json.WriteString("unk_token", End);
            json.WriteString("continuing_subword_prefix", "");
            json.WriteString("end_of_word_suffix", "</w>");
            json.WriteStartObject("vocab");
            foreach (var (piece, id) in ids) json.WriteNumber(piece, id);
            json.WriteEndObject();
            json.WriteStartArray("merges");
            foreach (var merge in merges) json.WriteStringValue(merge);
            json.WriteEndArray();
            json.WriteEndObject();

            json.WriteEndObject();
        }

        return FromJson(System.Text.Encoding.UTF8.GetString(stream.ToArray()));
    }

    /// <summary>Loads a tokenizer from a folder on disk, as <see cref="FromPretrained"/> loads one from the Hub.</summary>
    /// <param name="directory">A folder holding <c>tokenizer.json</c>, or <c>vocab.json</c> with <c>merges.txt</c>.</param>
    /// <remarks>
    /// The padding token is taken from <c>tokenizer_config.json</c>, or failing that from
    /// <c>special_tokens_map.json</c>; a CLIP vocabulary is recognised and read as CLIP's.
    /// </remarks>
    public static HfTokenizer FromDirectory(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);

        var config = Path.Combine(directory, "tokenizer_config.json");
        var special = Path.Combine(directory, "special_tokens_map.json");
        var configPath = File.Exists(config) ? config : null;

        HfTokenizer tokenizer;
        if (File.Exists(Path.Combine(directory, "tokenizer.json")))
        {
            tokenizer = Load(Path.Combine(directory, "tokenizer.json"));
        }
        else if (File.Exists(Path.Combine(directory, "vocab.json")) && File.Exists(Path.Combine(directory, "merges.txt")))
        {
            var vocab = Path.Combine(directory, "vocab.json");
            var merges = Path.Combine(directory, "merges.txt");
            tokenizer = IsClip(configPath, vocab) ? FromClipFiles(vocab, merges) : FromGpt2Files(vocab, merges);
        }
        else if (File.Exists(Path.Combine(directory, "vocab.txt")))
        {
            tokenizer = FromBertVocabulary(Path.Combine(directory, "vocab.txt"), configPath is null || ReadLowercaseFlag(configPath));
        }
        else
        {
            throw new FileNotFoundException(
                $"'{directory}' holds no tokenizer.json, vocab.json with merges.txt, or vocab.txt.");
        }

        return ReadPadToken(configPath) is not null
            ? tokenizer.WithConfiguredPadding(configPath)
            : tokenizer.WithConfiguredPadding(File.Exists(special) ? special : null);
    }

    /// <summary>A copy of this tokenizer that pads with <paramref name="token"/>.</summary>
    /// <param name="token">The padding token, for example <c>&lt;|endoftext|&gt;</c>.</param>
    /// <exception cref="ArgumentException">The token is not in the vocabulary.</exception>
    /// <remarks>
    /// Which token pads is set per model, not per tokenizer family: Stable Diffusion 1.x pads with
    /// <c>&lt;|endoftext|&gt;</c>, 2.x with <c>!</c>. The text encoder was trained on one of them,
    /// and padding with another changes every prompt's embedding.
    /// </remarks>
    public HfTokenizer WithPadToken(string token)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(token);

        var id = _addedTokens.TryGetValue(token, out var added) ? added.Id : _model.IdOf(token);
        if (id < 0 || _model.TokenOf(id) != token && !_addedTokens.ContainsKey(token))
        {
            throw new ArgumentException($"'{token}' is not in the vocabulary, so it cannot pad.", nameof(token));
        }

        var copy = (HfTokenizer)MemberwiseClone();
        copy.PadId = id;
        return copy;
    }

    /// <summary>Applies <c>tokenizer_config.json</c>'s <c>pad_token</c>, when it names one this vocabulary has.</summary>
    internal HfTokenizer WithConfiguredPadding(string? configPath)
    {
        var pad = ReadPadToken(configPath);
        if (pad is null) return this;

        try
        {
            return WithPadToken(pad);
        }
        catch (ArgumentException)
        {
            return this;
        }
    }

    /// <summary>The <c>pad_token</c> a <c>tokenizer_config.json</c> or <c>special_tokens_map.json</c> names.</summary>
    internal static string? ReadPadToken(string? configPath)
    {
        if (configPath is null || !File.Exists(configPath)) return null;

        try
        {
            using var document = JsonDocument.Parse(File.ReadAllText(configPath));
            if (!document.RootElement.TryGetProperty("pad_token", out var pad)) return null;

            // Either a bare string or an AddedToken object with a "content" field.
            return pad.ValueKind switch
            {
                JsonValueKind.String => pad.GetString(),
                JsonValueKind.Object when pad.TryGetProperty("content", out var content) => content.GetString(),
                _ => null,
            };
        }
        catch (JsonException)
        {
            return null;
        }
    }

    /// <summary>Whether a <c>vocab.json</c> + <c>merges.txt</c> pair is CLIP's rather than GPT-2's.</summary>
    internal static bool IsClip(string? configPath, string vocabPath)
    {
        if (configPath is not null && File.Exists(configPath))
        {
            try
            {
                using var document = JsonDocument.Parse(File.ReadAllText(configPath));
                if (document.RootElement.TryGetProperty("tokenizer_class", out var kind)
                    && kind.GetString() is { } name
                    && name.StartsWith("CLIPTokenizer", StringComparison.Ordinal))
                {
                    return true;
                }
            }
            catch (JsonException)
            {
            }
        }

        // Without a config, CLIP's vocabulary gives itself away with its start token.
        return File.ReadAllText(vocabPath).Contains("\"<|startoftext|>\"", StringComparison.Ordinal);
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

            // Added tokens first, as the reference looks them up: Pythia's runs of spaces are added
            // tokens outside the model's vocabulary, and looking only there dropped them.
            var token = _addedById.TryGetValue(id, out var added) ? added : _model.TokenOf(id);
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

        // The reference matches added tokens in two passes: those marked "normalized: false" in the
        // raw text, then - after normalizing what lies between them - those marked "normalized: true"
        // by their own normalized form. Llama 2's <s> is the second kind: its normalizer turns " <s>"
        // into "▁<s>", so the space before it belongs to the token and must not become a piece.
        // BERT's normalizer runs per word, after the split, as it always has.
        var normalizeFirst = _normalizer is not null and not BertNormalizer;
        var rawTokens = normalizeFirst
            ? _addedTokens.Values.Where(t => !t.Normalized).Select(t => (t.Content, t)).ToList()
            : [.. _addedTokens.Values.Select(t => (t.Content, t))];
        var normalizedTokens = normalizeFirst
            ? _addedTokens.Values.Where(t => t.Normalized).Select(t => (_normalizer!.Normalize(t.Content), t)).Where(p => p.Item1.Length > 0).ToList()
            : [];

        foreach (var segment in SplitOn(text, rawTokens))
        {
            if (segment.Added is { } added)
            {
                result.Add((added.Id, added.Content, segment.Start, segment.End, added.Special));
                continue;
            }

            // The reference normalizes the whole text before splitting it. Normalizing each piece
            // afterwards is equivalent for BERT's normalizer, whose pieces are plain words, but not
            // after a byte-level split: lowercasing the byte alphabet turns the byte 0xC3, written
            // 'Ã', into 'ã' - another byte - and every accented letter gets the wrong id.
            var (normalizedText, map) = normalizeFirst ? NormalizeAligned(segment.Text) : (segment.Text, null);

            foreach (var part in SplitOn(normalizedText, normalizedTokens))
            {
                if (part.Added is { } inner)
                {
                    var from = map is null ? part.Start : map[part.Start];
                    var to = map is null ? part.End : map[part.End];
                    result.Add((inner.Id, inner.Content, segment.Start + from, segment.Start + to, inner.Special));
                    continue;
                }

                foreach (var preToken in _preTokenizer.Split(part.Text))
                {
                    var word = normalizeFirst || _normalizer is null ? preToken.Word : _normalizer.Normalize(preToken.Word);
                    if (word.Length == 0) continue;

                    // Positions back in the caller's text: through the alignment when the text was
                    // normalized first, directly otherwise.
                    var startInSegment = part.Start + preToken.Start;
                    var endInSegment = part.Start + preToken.End;
                    var originalStart = map is null ? startInSegment : map[startInSegment];
                    var originalEnd = map is null ? endInSegment : map[endInSegment];
                    var originalWord = segment.Text[originalStart..originalEnd];

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

                    var aligned = word.Length == originalWord.Length && total == originalWord.Length;
                    var cursor = segment.Start + originalStart;

                    for (var i = 0; i < pieces.Count; i++)
                    {
                        var start = aligned ? cursor : segment.Start + originalStart;
                        var end = aligned ? cursor + surfaces[i].Length : segment.Start + originalEnd;
                        cursor = end;

                        result.Add((_model.IdOf(pieces[i]), pieces[i], start, end, false));
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Normalizes a whole text, with a map from each normalized position back to the original one.
    /// </summary>
    /// <remarks>
    /// Each character is normalized on its own and the pieces concatenated, which gives an exact
    /// map. Where that differs from normalizing the text as a whole - a whitespace run collapsed to
    /// one space, say - the whole-text result is used, because it is what the ids have to come
    /// from, and the map falls back to proportional positions: the ids stay right, and only the
    /// offsets of that text become approximate.
    /// </remarks>
    private (string Text, int[] Map) NormalizeAligned(string original)
    {
        var whole = _normalizer!.Normalize(original);
        var built = new System.Text.StringBuilder(whole.Length);
        var map = new List<int>(whole.Length + 1);

        var index = 0;
        foreach (var rune in original.EnumerateRunes())
        {
            var piece = _normalizer.Normalize(rune.ToString());
            built.Append(piece);
            for (var k = 0; k < piece.Length; k++) map.Add(index);
            index += rune.Utf16SequenceLength;
        }

        map.Add(original.Length);
        if (built.ToString() == whole) return (whole, [.. map]);

        var scaled = new int[whole.Length + 1];
        for (var j = 0; j <= whole.Length; j++)
        {
            scaled[j] = whole.Length == 0 ? 0 : (int)Math.Round((double)j * original.Length / whole.Length);
        }

        return (whole, scaled);
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
    private static List<(string Text, int Start, int End, AddedToken? Added)> SplitOn(
        string text, IReadOnlyList<(string Pattern, AddedToken Token)> tokens)
    {
        if (tokens.Count == 0) return text.Length == 0 ? [] : [(text, 0, text.Length, null)];

        var segments = new List<(string, int, int, AddedToken?)>();
        var cursor = 0;

        while (cursor < text.Length)
        {
            var bestIndex = -1;
            var bestPattern = "";
            AddedToken? bestToken = null;

            foreach (var (pattern, added) in tokens)
            {
                var at = text.IndexOf(pattern, cursor, StringComparison.Ordinal);
                if (at < 0) continue;

                // Earliest wins; on a tie the longer token wins, so <s> cannot pre-empt <sep>.
                if (bestIndex < 0 || at < bestIndex || (at == bestIndex && pattern.Length > bestPattern.Length))
                {
                    bestIndex = at;
                    bestPattern = pattern;
                    bestToken = added;
                }
            }

            if (bestIndex < 0)
            {
                segments.Add((text[cursor..], cursor, text.Length, null));
                break;
            }

            // lstrip and rstrip absorb the whitespace beside the token into it.
            var matchStart = bestIndex;
            var matchEnd = bestIndex + bestPattern.Length;
            if (bestToken!.Value.LStrip) while (matchStart > cursor && char.IsWhiteSpace(text[matchStart - 1])) matchStart--;
            if (bestToken.Value.RStrip) while (matchEnd < text.Length && char.IsWhiteSpace(text[matchEnd])) matchEnd++;

            if (matchStart > cursor) segments.Add((text[cursor..matchStart], cursor, matchStart, null));

            segments.Add((bestToken.Value.Content, matchStart, matchEnd, bestToken));
            cursor = matchEnd;
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
            // The reference truncates the text and then adds the special tokens, so a closing
            // [SEP] or <|endoftext|> survives. Cutting the finished sequence would drop it, and a
            // model that pools or reads at the end token then reads a word instead.
            var keepLast = width > 1 && encoding.SpecialTokensMask[^1] == 1;
            IEnumerable<int> Positions() => keepLast
                ? [.. Enumerable.Range(0, width - 1), encoding.Length - 1]
                : Enumerable.Range(0, width);

            return new Encoding(
                [.. Positions().Select(i => encoding.Ids[i])],
                [.. Positions().Select(i => encoding.Tokens[i])],
                [.. Positions().Select(i => encoding.AttentionMask[i])],
                [.. Positions().Select(i => encoding.TypeIds[i])],
                [.. Positions().Select(i => encoding.SpecialTokensMask[i])],
                [.. Positions().Select(i => encoding.Offsets[i])]);
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
