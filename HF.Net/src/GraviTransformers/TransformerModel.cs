using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviTokenizers;
using Gravicode.Science.GraviNum;
using Encoder = Gravicode.Science.GraviText.Transformers.TransformerModel;

namespace Gravicode.HFNet.GraviTransformers;

/// <summary>One predicted class and its probability.</summary>
/// <param name="Label">The label name from the model's config, or the class index as text.</param>
/// <param name="Score">The softmax probability, in <c>[0,1]</c>.</param>
/// <param name="Index">The class index.</param>
public readonly record struct Prediction(string Label, double Score, int Index)
{
    /// <inheritdoc />
    public override string ToString() => $"{Label}: {Score:P2}";
}

/// <summary>One candidate for a masked position.</summary>
/// <param name="Token">The predicted piece.</param>
/// <param name="Score">Its probability.</param>
/// <param name="Sequence">The input with the mask filled in.</param>
public readonly record struct MaskFill(string Token, double Score, string Sequence)
{
    /// <inheritdoc />
    public override string ToString() => $"{Token} ({Score:P2})";
}

/// <summary>
/// A pretrained transformer, loaded from the Hub and ready to run.
/// </summary>
/// <remarks>
/// <para>
/// This is the entry point the blueprint asks for: <c>TransformerModel.Load("bert-base-uncased")</c>
/// followed by <c>Predict(text)</c>. It ties together the four things a working model needs - the
/// config, the tokenizer, the encoder weights and whatever task head the checkpoint carries - so
/// that none of them can be mismatched by accident.
/// </para>
/// <para>
/// Inference is CPU-bound and single-sequence here; <c>EmbedBatch</c> parallelises across inputs,
/// which is where the throughput is. There is no KV cache because this is an encoder: every
/// position attends to every other one anyway.
/// </para>
/// </remarks>
public sealed class TransformerModel : IDisposable
{
    private readonly WeightStore _weights;
    private readonly ClassificationHead? _classifier;
    private readonly MaskedLanguageHead? _maskedLanguage;
    private bool _disposed;

    private TransformerModel(
        string repoId,
        PretrainedConfig config,
        HfTokenizer tokenizer,
        Encoder encoder,
        WeightStore weights,
        LoadReport report)
    {
        RepoId = repoId;
        Config = config;
        Tokenizer = tokenizer;
        Encoder = encoder;
        Report = report;
        _weights = weights;

        _classifier = ClassificationHead.TryLoad(weights, config);
        _maskedLanguage = MaskedLanguageHead.TryLoad(weights, config, encoder);
    }

    /// <summary>The Hub id this model came from.</summary>
    public string RepoId { get; }

    /// <summary>The model's configuration.</summary>
    public PretrainedConfig Config { get; }

    /// <summary>The tokenizer that matches these weights.</summary>
    public HfTokenizer Tokenizer { get; }

    /// <summary>The encoder stack, from Gravicode.Science.GraviText.</summary>
    public Encoder Encoder { get; }

    /// <summary>What the checkpoint load found.</summary>
    public LoadReport Report { get; }

    /// <summary>Whether the checkpoint carries a sequence classification head.</summary>
    public bool HasClassificationHead => _classifier is not null;

    /// <summary>Whether the checkpoint carries a masked language modelling head.</summary>
    public bool HasMaskedLanguageHead => _maskedLanguage is not null;

    /// <summary>The label names, when the model has a classification head.</summary>
    public IReadOnlyList<string> Labels =>
        [.. Enumerable.Range(0, Config.LabelCount).Select(i => Config.IdToLabel.GetValueOrDefault(i, $"LABEL_{i}"))];

    // ------------------------------------------------------------------ loading

    /// <summary>Downloads a model from the Hub and loads it.</summary>
    /// <param name="repoId">A model id such as <c>bert-base-uncased</c>.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    /// <param name="progress">Receives download progress.</param>
    /// <exception cref="NotSupportedException">The architecture is not an encoder this can run.</exception>
    public static TransformerModel Load(
        string repoId, string revision = "main", IProgress<TransferProgress>? progress = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var config = PretrainedConfig.FromPretrained(repoId, revision);
        RequireSupported(config, repoId);

        var weights = WeightStore.FromPretrained(repoId, revision, progress);

        try
        {
            var (encoder, report) = CheckpointLoader.Load(config, weights);
            var tokenizer = HfTokenizer.FromPretrained(repoId, revision);

            return new TransformerModel(repoId, config, tokenizer, encoder, weights, report);
        }
        catch
        {
            weights.Dispose();
            throw;
        }
    }

    /// <summary>The encoder families this can run.</summary>
    /// <remarks>
    /// Decoder-only and encoder-decoder models are refused rather than half-loaded. Their blocks are
    /// shaped differently - causal masking, cross-attention, rotary positions - and filling an
    /// encoder from their weights produces a model that runs and returns nonsense.
    /// </remarks>
    public static IReadOnlySet<string> SupportedModelTypes { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        "bert", "roberta", "xlm-roberta", "distilbert", "electra", "camembert", "mpnet", "deberta",
    };

    private static void RequireSupported(PretrainedConfig config, string repoId)
    {
        if (SupportedModelTypes.Contains(config.ModelType)) return;

        throw new NotSupportedException(
            $"'{repoId}' is a '{config.ModelType}' model. GraviTransformers runs BERT-family encoders "
            + $"({string.Join(", ", SupportedModelTypes.Order())}); decoder-only and encoder-decoder "
            + "architectures need causal masking and cross-attention, which this encoder does not have.");
    }

    // ------------------------------------------------------------------ inference

    /// <summary>The final hidden states for a text, one row per token.</summary>
    /// <param name="text">The input.</param>
    /// <param name="maxLength">Truncation limit in tokens.</param>
    public NdArray Hidden(string text, int maxLength = 512)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var encoding = Tokenizer.Encode(text);
        var ids = encoding.ToIdArray();

        if (ids.Length > maxLength) ids = ids[..maxLength];
        var mask = Enumerable.Repeat(1, ids.Length).ToArray();

        return Encoder.Forward(ids, mask);
    }

    /// <summary>A single vector for a text, mean-pooled over its tokens.</summary>
    /// <param name="text">The input.</param>
    /// <param name="maxLength">Truncation limit in tokens.</param>
    /// <remarks>
    /// Mean pooling rather than the <c>[CLS]</c> vector. <c>[CLS]</c> only carries sentence meaning
    /// once a model has been fine-tuned to put it there - on a plain pretrained encoder it is close
    /// to constant, which makes every pair of sentences look similar.
    /// </remarks>
    public NdArray Embed(string text, int maxLength = 512)
    {
        var hidden = Hidden(text, maxLength);
        var width = hidden.Shape[1];
        var pooled = NdArray.Zeros(width);

        for (var i = 0; i < hidden.Shape[0]; i++)
        {
            for (var d = 0; d < width; d++) pooled[d] += hidden[i, d];
        }

        for (var d = 0; d < width; d++) pooled[d] /= hidden.Shape[0];
        return pooled;
    }

    /// <summary>Embeds many texts, one row per text.</summary>
    /// <param name="texts">The inputs.</param>
    /// <param name="maxLength">Truncation limit in tokens.</param>
    public NdArray EmbedBatch(IReadOnlyList<string> texts, int maxLength = 512)
    {
        var result = NdArray.Zeros(texts.Count, Config.HiddenSize);

        // Sequences are independent, so this is embarrassingly parallel and is where the throughput
        // comes from on a CPU.
        Parallel.For(0, texts.Count, i =>
        {
            var vector = Embed(texts[i], maxLength);
            for (var d = 0; d < Config.HiddenSize; d++) result[i, d] = vector.At(d);
        });

        return result;
    }

    /// <summary>Classifies a text, best class first.</summary>
    /// <param name="text">The input.</param>
    /// <param name="topK">How many classes to return.</param>
    /// <exception cref="InvalidOperationException">The checkpoint has no classification head.</exception>
    public IReadOnlyList<Prediction> Predict(string text, int topK = 0)
    {
        if (_classifier is null)
        {
            throw new InvalidOperationException(
                $"'{RepoId}' has no sequence classification head, so there is nothing to predict. "
                + "Use Embed() for a feature vector, or load a fine-tuned checkpoint "
                + "(for example distilbert-base-uncased-finetuned-sst-2-english).");
        }

        var hidden = Hidden(text);
        var logits = _classifier.Apply(hidden);
        var probabilities = Softmax(logits);

        var ranked = probabilities
            .Select((score, index) => new Prediction(
                Config.IdToLabel.GetValueOrDefault(index, $"LABEL_{index}"), score, index))
            .OrderByDescending(p => p.Score)
            .ToList();

        return topK > 0 ? ranked.Take(topK).ToList() : ranked;
    }

    /// <summary>Fills a masked position, best candidate first.</summary>
    /// <param name="text">Input containing the tokenizer's mask token, for example <c>[MASK]</c>.</param>
    /// <param name="topK">How many candidates to return.</param>
    /// <exception cref="InvalidOperationException">The checkpoint has no masked language head.</exception>
    /// <remarks>
    /// This is the sharpest check that an encoder loaded correctly. A model with a transposed
    /// weight or a shifted position embedding still produces plausible-looking vectors, but it does
    /// not answer "The capital of France is [MASK]." with <c>paris</c>.
    /// </remarks>
    public IReadOnlyList<MaskFill> FillMask(string text, int topK = 5)
    {
        if (_maskedLanguage is null)
        {
            throw new InvalidOperationException(
                $"'{RepoId}' has no masked language modelling head. A base checkpoint such as "
                + "bert-base-uncased has one; a fine-tuned classifier usually does not.");
        }

        if (Tokenizer.MaskId < 0)
        {
            throw new InvalidOperationException($"The tokenizer for '{RepoId}' has no mask token.");
        }

        var encoding = Tokenizer.Encode(text);
        var position = encoding.Ids.ToList().IndexOf(Tokenizer.MaskId);

        if (position < 0)
        {
            var maskToken = Tokenizer.IdToToken(Tokenizer.MaskId);
            throw new ArgumentException($"The input contains no mask token ('{maskToken}').", nameof(text));
        }

        var hidden = Encoder.Forward(encoding.ToIdArray(), encoding.ToMaskArray());
        var scores = _maskedLanguage.Apply(hidden.Row(position));
        var probabilities = Softmax(scores);

        return [.. probabilities
            .Select((score, id) => (score, id))
            .OrderByDescending(p => p.score)
            .Take(topK)
            .Select(p => new MaskFill(
                Tokenizer.IdToToken(p.id),
                p.score,
                text.Replace(Tokenizer.IdToToken(Tokenizer.MaskId), Tokenizer.IdToToken(p.id))))];
    }

    /// <summary>Cosine similarity between two texts' embeddings, in <c>[-1,1]</c>.</summary>
    public double Similarity(string first, string second)
    {
        var a = Embed(first);
        var b = Embed(second);

        double dot = 0, normA = 0, normB = 0;
        for (var i = 0; i < a.Size; i++)
        {
            dot += a.At(i) * b.At(i);
            normA += a.At(i) * a.At(i);
            normB += b.At(i) * b.At(i);
        }

        var denominator = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denominator == 0 ? 0 : dot / denominator;
    }

    /// <summary>Numerically stable softmax.</summary>
    /// <remarks>
    /// The maximum is subtracted first. Logits from a real model routinely reach the high tens, and
    /// <c>exp(90)</c> overflows a double - giving NaN probabilities from a model that is working.
    /// </remarks>
    internal static double[] Softmax(double[] logits)
    {
        var max = logits.Max();
        var exponentials = new double[logits.Length];
        var total = 0.0;

        for (var i = 0; i < logits.Length; i++)
        {
            exponentials[i] = Math.Exp(logits[i] - max);
            total += exponentials[i];
        }

        for (var i = 0; i < logits.Length; i++) exponentials[i] /= total;
        return exponentials;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _weights.Dispose();
    }

    /// <inheritdoc />
    public override string ToString()
        => $"{RepoId}: {Config}"
            + (HasClassificationHead ? " +classifier" : "")
            + (HasMaskedLanguageHead ? " +mlm" : "");
}
