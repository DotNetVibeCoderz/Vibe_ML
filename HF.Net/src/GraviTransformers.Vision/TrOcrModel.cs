using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviTokenizers;
using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviTransformers.Vision;

/// <summary>How <see cref="TrOcrModel"/> generates a text; anything left <c>null</c> comes from the checkpoint's generation config.</summary>
/// <param name="MaxNewTokens">
/// The most tokens to generate after the start token. transformers' default, when the checkpoint sets
/// none, is 20 - enough for a short line, not for a long one; raise it for long lines.
/// </param>
/// <param name="Beams">Beam search with this many beams; 1 is greedy decoding.</param>
/// <param name="LengthPenalty">Beam search: a finished text's log-probability is divided by its length to this power.</param>
/// <param name="EarlyStopping">Beam search: stop once there are as many finished texts as beams.</param>
public sealed record OcrSettings(
    int? MaxNewTokens = null,
    int? Beams = null,
    double? LengthPenalty = null,
    bool? EarlyStopping = null)
{
    /// <summary>The checkpoint's own settings.</summary>
    public static OcrSettings Default { get; } = new();
}

/// <summary>
/// TrOCR: printed or handwritten text recognition, one line of text per image - a
/// <c>VisionEncoderDecoderModel</c> of a ViT or DeiT image encoder and a TrOCR text decoder.
/// </summary>
/// <remarks>
/// <para>
/// The image is resized to the encoder's square (384x384 for the published checkpoints, whatever the
/// line's own shape - that is how they were trained), encoded into one vector per patch, and the
/// decoder writes the text a token at a time while attending to those vectors. It starts from
/// <c>decoder_start_token_id</c> (<c>&lt;/s&gt;</c>, id 2) and stops at the end token, as
/// transformers' <c>generate</c> does: greedy by default, beam search on request, with the same
/// stopping rules, so the text is token for token what Python returns.
/// </para>
/// <para>
/// The encoder is <see cref="VisionTransformer"/>'s own pre-norm block; the decoder is post-norm,
/// and keeps a key/value cache for its self-attention plus each image's cross-attention keys and
/// values, computed once. Several images, or several beams, are decoded as rows of one matrix
/// product per step.
/// </para>
/// <para>
/// trocr-*-printed were fine-tuned on receipts (SROIE) and write capitals: "Net revenue" comes back
/// as "NET REVENUE". That is the checkpoint, not the port.
/// </para>
/// </remarks>
public sealed class TrOcrModel : IDisposable
{
    private readonly TrOcrDecoder _decoder;
    private readonly HashSet<int> _specialIds;
    private bool _disposed;

    private TrOcrModel(string id, TrOcrConfig config, VisionTransformer encoder, TrOcrDecoder decoder, HfTokenizer? tokenizer)
    {
        Id = id;
        Config = config;
        Encoder = encoder;
        _decoder = decoder;
        Tokenizer = tokenizer;

        // What skip_special_tokens drops: the generation ids, and the tokenizer's named specials.
        _specialIds = [config.DecoderStartTokenId, config.BosTokenId, config.PadTokenId, .. config.EndTokenIds];
        if (tokenizer is not null)
        {
            foreach (var token in (string[])["<s>", "</s>", "<pad>", "<unk>", "<mask>"])
            {
                var tokenId = tokenizer.TokenToId(token);
                if (tokenId >= 0 && tokenizer.Decode([tokenId], skipSpecialTokens: false) == token) _specialIds.Add(tokenId);
            }
        }
    }

    /// <summary>The repository or directory this came from.</summary>
    public string Id { get; }

    /// <summary>The configuration.</summary>
    public TrOcrConfig Config { get; }

    /// <summary>The image encoder.</summary>
    public VisionTransformer Encoder { get; }

    /// <summary>How images are prepared: the encoder's square, the checkpoint's resampler and statistics.</summary>
    public ImageProcessor Processor => Encoder.Processor;

    /// <summary>The tokenizer, or <c>null</c> for a directory that holds none.</summary>
    public HfTokenizer? Tokenizer { get; }

    /// <summary>Downloads a TrOCR checkpoint from the Hub and loads it.</summary>
    /// <param name="repoId">A model id such as <c>microsoft/trocr-small-printed</c> or <c>microsoft/trocr-base-printed</c>.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    /// <exception cref="NotSupportedException">The checkpoint is not a ViT/DeiT + TrOCR encoder-decoder.</exception>
    public static TrOcrModel Load(string repoId, string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var config = TrOcrConfig.FromPretrained(repoId, revision);
        var processor = ImageProcessor.FromPretrained(repoId, revision);
        var tokenizer = HfTokenizer.FromPretrained(repoId, revision);

        using var weights = WeightStore.FromPretrained(repoId, revision);
        return Build(repoId, config, processor, weights, tokenizer);
    }

    /// <summary>Loads a TrOCR checkpoint from a directory.</summary>
    /// <param name="directory">
    /// A folder holding <c>config.json</c> and the weights, and optionally <c>generation_config.json</c>,
    /// <c>preprocessor_config.json</c> and the tokenizer files.
    /// </param>
    public static TrOcrModel Open(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);

        var generation = Path.Combine(directory, "generation_config.json");
        var config = TrOcrConfig.Load(Path.Combine(directory, "config.json"), File.Exists(generation) ? generation : null, directory);

        var preprocessor = Path.Combine(directory, "preprocessor_config.json");
        var processor = File.Exists(preprocessor) ? ImageProcessor.Load(preprocessor) : ImageProcessor.ViT;

        HfTokenizer? tokenizer = null;
        if (new[] { "tokenizer.json", "vocab.json", "sentencepiece.bpe.model" }.Any(f => File.Exists(Path.Combine(directory, f))))
        {
            tokenizer = HfTokenizer.FromDirectory(directory);
        }

        using var weights = WeightStore.Open(directory);
        return Build(directory, config, processor, weights, tokenizer);
    }

    private static TrOcrModel Build(string id, TrOcrConfig config, ImageProcessor processor, WeightStore weights, HfTokenizer? tokenizer)
    {
        if (!weights.Contains("encoder.embeddings.cls_token") || !weights.Contains("decoder.model.decoder.embed_tokens.weight"))
        {
            throw new InvalidDataException(
                $"'{id}' does not hold a VisionEncoderDecoderModel's weights (encoder.embeddings.*, decoder.model.decoder.*).");
        }

        // The encoder runs at the size it was trained at; its processor is made to produce exactly that.
        var aligned = VisionTransformer.Align(processor, config.Encoder, imageSize: null);
        var encoder = VisionTransformer.Build(id, config.Encoder, aligned, weights, "encoder.");
        var decoder = new TrOcrDecoder(config.Decoder, weights, config.Encoder.HiddenSize);

        return new TrOcrModel(id, config, encoder, decoder, tokenizer);
    }

    // ------------------------------------------------------------------ text

    /// <summary>Reads the text in an image of one line.</summary>
    /// <param name="path">A PNG, JPEG, BMP, GIF or WebP file.</param>
    /// <param name="settings">Generation settings; the checkpoint's by default.</param>
    public string Recognize(string path, OcrSettings? settings = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ObjectDisposedException.ThrowIf(_disposed, this);

        return Recognize(Processor.Read(path), settings);
    }

    /// <summary>Reads the text in an image of one line, from a stream.</summary>
    public string Recognize(Stream image, OcrSettings? settings = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        ObjectDisposedException.ThrowIf(_disposed, this);

        return Recognize(Processor.Read(image), settings);
    }

    /// <summary>Reads the text in prepared pixels, <c>[3, size, size]</c> as <see cref="Processor"/> makes them.</summary>
    public string Recognize(NdArray pixels, OcrSettings? settings = null) => Decode(GenerateIds(pixels, settings));

    /// <summary>Reads several line images, decoding them together.</summary>
    /// <param name="paths">Image files, one line of text each.</param>
    /// <param name="settings">Generation settings; the checkpoint's by default.</param>
    /// <returns>One text per image, in order.</returns>
    public IReadOnlyList<string> RecognizeBatch(IEnumerable<string> paths, OcrSettings? settings = null)
    {
        ArgumentNullException.ThrowIfNull(paths);
        ObjectDisposedException.ThrowIf(_disposed, this);

        return RecognizeBatch([.. paths.Select(Processor.Read)], settings);
    }

    /// <summary>Reads several prepared images, decoding them together.</summary>
    public IReadOnlyList<string> RecognizeBatch(IReadOnlyList<NdArray> pixels, OcrSettings? settings = null)
        => [.. GenerateIds(pixels, settings).Select(Decode)];

    /// <summary>The generated ids, start token and end token included, as transformers' <c>generate</c> returns them.</summary>
    public IReadOnlyList<int> GenerateIds(NdArray pixels, OcrSettings? settings = null)
    {
        ArgumentNullException.ThrowIfNull(pixels);
        return GenerateIds([pixels], settings)[0];
    }

    /// <summary>The generated ids for several images.</summary>
    public IReadOnlyList<IReadOnlyList<int>> GenerateIds(IReadOnlyList<NdArray> pixels, OcrSettings? settings = null)
    {
        ArgumentNullException.ThrowIfNull(pixels);
        ObjectDisposedException.ThrowIf(_disposed, this);

        var options = settings ?? OcrSettings.Default;
        var beams = options.Beams ?? Config.Beams;
        var maxNew = Math.Min(options.MaxNewTokens ?? Config.MaxNewTokens, _decoder.MaxPositions - 1);
        if (beams < 1) throw new ArgumentOutOfRangeException(nameof(settings), beams, "Beam search needs at least one beam.");
        if (maxNew < 1) throw new ArgumentOutOfRangeException(nameof(settings), maxNew, "At least one token has to be generated.");

        var images = pixels.Select(p => _decoder.Prepare(EncoderStates(p).AsContiguous().ToArray(), EncoderRows(p))).ToArray();

        return beams == 1
            ? Greedy(images, maxNew)
            : [.. images.Select(image => BeamSearch(image, beams, maxNew, options.LengthPenalty ?? Config.LengthPenalty, options.EarlyStopping ?? Config.EarlyStopping))];
    }

    /// <summary>Turns generated ids into text, special tokens dropped, as <c>batch_decode(skip_special_tokens=True)</c> does.</summary>
    public string Decode(IReadOnlyList<int> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var tokenizer = Tokenizer ?? throw new InvalidOperationException(
            $"'{Id}' was opened without a tokenizer; use GenerateIds, or put the tokenizer files beside the weights.");

        return tokenizer.Decode([.. ids.Where(id => !_specialIds.Contains(id))], skipSpecialTokens: true);
    }

    // ------------------------------------------------------------------ the two halves

    /// <summary>The encoder's final hidden states for prepared pixels, <c>[tokens, hidden]</c> - what the decoder attends to.</summary>
    public NdArray EncoderStates(NdArray pixels)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return Encoder.Forward(pixels);
    }

    private int EncoderRows(NdArray pixels)
        => Encoder.LeadingTokens + (pixels.Shape[1] / Config.Encoder.PatchSize) * (pixels.Shape[2] / Config.Encoder.PatchSize);

    /// <summary>
    /// The decoder's logits at every position of <paramref name="decoderIds"/>, <c>[tokens, vocab]</c>
    /// row-major - what transformers' forward pass returns for those <c>decoder_input_ids</c>.
    /// </summary>
    /// <param name="pixels">Prepared pixels.</param>
    /// <param name="decoderIds">The decoder's input, normally starting with <see cref="TrOcrConfig.DecoderStartTokenId"/>.</param>
    public double[] Logits(NdArray pixels, IReadOnlyList<int> decoderIds)
    {
        ArgumentNullException.ThrowIfNull(decoderIds);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (decoderIds.Count == 0 || decoderIds.Count > _decoder.MaxPositions)
        {
            throw new ArgumentException($"The decoder takes 1 to {_decoder.MaxPositions} tokens; got {decoderIds.Count}.", nameof(decoderIds));
        }

        var image = _decoder.Prepare(EncoderStates(pixels).AsContiguous().ToArray(), EncoderRows(pixels));
        var sequence = _decoder.Start(image);
        var vocab = _decoder.VocabularySize;
        var logits = new double[decoderIds.Count * vocab];

        for (var t = 0; t < decoderIds.Count; t++)
        {
            Array.Copy(_decoder.Step([sequence], [decoderIds[t]]), 0, logits, t * vocab, vocab);
        }

        return logits;
    }

    // ------------------------------------------------------------------ generation

    /// <summary>Greedy decoding of every image at once: the most likely token, until the end token or the length limit.</summary>
    private List<IReadOnlyList<int>> Greedy(CrossAttentionStates[] images, int maxNew)
    {
        var sequences = images.Select(_decoder.Start).ToArray();
        var pending = Enumerable.Repeat(Config.DecoderStartTokenId, images.Length).ToArray();
        var results = new IReadOnlyList<int>[images.Length];
        var active = Enumerable.Range(0, images.Length).ToList();
        var ends = Config.EndTokenIds.ToHashSet();
        var vocab = _decoder.VocabularySize;

        while (active.Count > 0)
        {
            var logits = _decoder.Step([.. active.Select(i => sequences[i])], [.. active.Select(i => pending[i])]);
            var still = new List<int>(active.Count);

            for (var row = 0; row < active.Count; row++)
            {
                var i = active[row];
                var next = ArgMax(logits.AsSpan(row * vocab, vocab));
                var length = sequences[i].Tokens.Count + 1;

                if (ends.Contains(next) || length >= 1 + maxNew)
                {
                    results[i] = [.. sequences[i].Tokens, next];
                }
                else
                {
                    pending[i] = next;
                    still.Add(i);
                }
            }

            active = still;
        }

        return [.. results];
    }

    /// <summary>The first index of the largest value, as <c>torch.argmax</c> breaks ties.</summary>
    private static int ArgMax(ReadOnlySpan<double> values)
    {
        var best = 0;
        for (var i = 1; i < values.Length; i++) if (values[i] > values[best]) best = i;
        return best;
    }

    /// <summary>A live beam: the sequence fed so far, and the token chosen for it but not yet fed.</summary>
    private readonly record struct Beam(DecoderSequence Sequence, int Pending);

    /// <summary>
    /// Beam search for one image, transformers' <c>_beam_search</c> step for step: the top
    /// <c>2 * beams</c> continuations by summed log-probability, those that end kept as finished
    /// texts (scored by length to <paramref name="lengthPenalty"/>), the best that do not carried on,
    /// and the same stopping heuristic.
    /// </summary>
    private IReadOnlyList<int> BeamSearch(CrossAttentionStates image, int beams, int maxNew, double lengthPenalty, bool earlyStopping)
    {
        const double Penalty = -1.0e9;
        var maxLength = 1 + maxNew;
        var ends = Config.EndTokenIds.ToHashSet();
        var vocab = _decoder.VocabularySize;
        var keep = Math.Max(2, 1 + Config.EndTokenIds.Count) * beams;

        // Every beam starts as the start token; all but the first at -1e9, so the first step's
        // candidates all come from one of them.
        var running = Enumerable.Range(0, beams).Select(_ => new Beam(_decoder.Start(image), Config.DecoderStartTokenId)).ToArray();
        var runningScores = new double[beams];
        for (var b = 1; b < beams; b++) runningScores[b] = Penalty;

        var finished = new List<int>[beams];
        var finishedScores = Enumerable.Repeat(Penalty, beams).ToArray();
        var isFinished = new bool[beams];
        for (var b = 0; b < beams; b++) finished[b] = [Config.DecoderStartTokenId];

        var improvable = true;

        while (true)
        {
            var logits = _decoder.Step([.. running.Select(r => r.Sequence)], [.. running.Select(r => r.Pending)]);
            var length = running[0].Sequence.Tokens.Count;   // cur_len: the start token and everything fed after it

            // log_softmax per beam, plus the beam's running score.
            var scores = new double[beams * vocab];
            for (var b = 0; b < beams; b++)
            {
                var row = logits.AsSpan(b * vocab, vocab);
                var largest = double.NegativeInfinity;
                foreach (var value in row) if (value > largest) largest = value;

                var total = 0.0;
                foreach (var value in row) total += Math.Exp(value - largest);
                var shift = largest + Math.Log(total);

                for (var v = 0; v < vocab; v++) scores[b * vocab + v] = row[v] - shift + runningScores[b];
            }

            var top = TopK(scores, keep);
            var candidates = top.Select(index => (Beam: index / vocab, Token: index % vocab, Score: scores[index])).ToArray();
            var hits = candidates.Select(c => ends.Contains(c.Token) || length + 1 >= maxLength).ToArray();

            // The best continuations that have not stopped carry on.
            var next = TopK([.. candidates.Select((c, k) => c.Score + (hits[k] ? Penalty : 0))], beams);

            // Those in the top `beams` that did stop become finished texts, if they beat one.
            var allFinished = isFinished.All(f => f);
            var mergedScores = new List<double>(finishedScores);
            var mergedTexts = new List<List<int>>(finished);
            var mergedFinished = new List<bool>(isFinished);
            for (var k = 0; k < candidates.Length; k++)
            {
                var justFinished = hits[k] && k < beams;
                var score = candidates[k].Score / Math.Pow(length + 1 - 1, lengthPenalty);
                if (allFinished && earlyStopping) score += Penalty;
                if (!improvable) score += Penalty;
                if (!justFinished) score += Penalty;

                mergedScores.Add(score);
                mergedTexts.Add([.. running[candidates[k].Beam].Sequence.Tokens, candidates[k].Token]);
                mergedFinished.Add(justFinished);
            }

            var best = TopK([.. mergedScores], beams);
            finished = [.. best.Select(i => mergedTexts[i])];
            finishedScores = [.. best.Select(i => mergedScores[i])];
            isFinished = [.. best.Select(i => mergedFinished[i])];

            // Reorder the running beams. A parent chosen twice is copied; its first child takes it as is.
            // Nothing has been fed yet, so every copy is taken from the parent as it stood.
            var taken = new HashSet<int>();
            var reordered = new Beam[beams];
            for (var b = 0; b < beams; b++)
            {
                var (parent, token, score) = candidates[next[b]];
                var sequence = taken.Add(parent) ? running[parent].Sequence : running[parent].Sequence.Fork();
                reordered[b] = new Beam(sequence, token);
                runningScores[b] = score + (hits[next[b]] ? Penalty : 0);
            }

            running = reordered;
            length++;

            // transformers' heuristic: stop once the best running beam, at its present length, cannot
            // beat the worst finished text.
            var bestPossible = runningScores[0] / Math.Pow(length - 1, lengthPenalty);
            var worst = finishedScores.Min();
            improvable &= isFinished.Any(f => !f) || bestPossible > worst;

            var openBeam = !(isFinished.All(f => f) && earlyStopping);
            if (!improvable || !openBeam || hits.All(h => h)) break;
        }

        return finished[0];
    }

    /// <summary>The indices of the <paramref name="k"/> largest values, largest first; ties keep index order.</summary>
    /// <remarks>A partial insertion sort: the beam search calls it on beams x vocabulary scores for a handful of winners.</remarks>
    private static int[] TopK(double[] values, int k)
    {
        k = Math.Min(k, values.Length);
        var indices = new int[k];
        var count = 0;

        for (var i = 0; i < values.Length; i++)
        {
            var value = values[i];
            if (count == k && !(value > values[indices[k - 1]])) continue;

            var at = count < k ? count++ : k - 1;
            while (at > 0 && value > values[indices[at - 1]])
            {
                indices[at] = indices[at - 1];
                at--;
            }

            indices[at] = i;
        }

        return indices;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;

        Encoder.Dispose();
        _disposed = true;
    }

    /// <inheritdoc />
    public override string ToString() => $"{Id}: {Config}";
}
