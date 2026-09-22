using Gravicode.Science.GraviNum;
using Encoder = Gravicode.Science.GraviText.Transformers.TransformerModel;

namespace Gravicode.HFNet.GraviTransformers;

/// <summary>
/// The sequence classification head a fine-tuned checkpoint carries on top of its encoder.
/// </summary>
/// <remarks>
/// Three layouts cover nearly every fine-tuned encoder on the Hub, and they differ in both their
/// parameter names and their activation:
/// <list type="bullet">
/// <item>BERT: a pooler (dense + tanh over <c>[CLS]</c>) then <c>classifier</c>.</item>
/// <item>DistilBERT: <c>pre_classifier</c> + ReLU then <c>classifier</c>, with no pooler.</item>
/// <item>RoBERTa: <c>classifier.dense</c> + tanh then <c>classifier.out_proj</c>.</item>
/// </list>
/// Using the wrong activation is not cosmetic - tanh saturates and ReLU does not, so the logits it
/// produces are wrong in a way that still ranks classes plausibly.
/// </remarks>
internal sealed class ClassificationHead
{
    private readonly NdArray? _poolerWeight;
    private readonly NdArray? _poolerBias;
    private readonly NdArray _classifierWeight;
    private readonly NdArray _classifierBias;
    private readonly bool _useTanh;

    private ClassificationHead(
        NdArray? poolerWeight, NdArray? poolerBias,
        NdArray classifierWeight, NdArray classifierBias,
        bool useTanh)
    {
        _poolerWeight = poolerWeight;
        _poolerBias = poolerBias;
        _classifierWeight = classifierWeight;
        _classifierBias = classifierBias;
        _useTanh = useTanh;
    }

    /// <summary>Loads a classification head, or returns null when the checkpoint has none.</summary>
    internal static ClassificationHead? TryLoad(WeightStore weights, PretrainedConfig config)
    {
        if (config.LabelCount == 0) return null;

        // RoBERTa keeps both of its layers under classifier.*, so it has to be checked before the
        // plain BERT spelling - "classifier.weight" does not exist there, but "classifier.dense"
        // does, and a looser probe would match the wrong pair.
        if (weights.Contains("classifier.out_proj.weight"))
        {
            return new ClassificationHead(
                weights.TryRead("classifier.dense.weight", out var dense) ? dense : null,
                weights.TryRead("classifier.dense.bias", out var denseBias) ? denseBias : null,
                weights.Read("classifier.out_proj.weight"),
                weights.Read("classifier.out_proj.bias"),
                useTanh: true);
        }

        if (weights.Contains("pre_classifier.weight") && weights.Contains("classifier.weight"))
        {
            return new ClassificationHead(
                weights.Read("pre_classifier.weight"),
                weights.Read("pre_classifier.bias"),
                weights.Read("classifier.weight"),
                weights.Read("classifier.bias"),
                useTanh: false);
        }

        if (weights.Contains("classifier.weight"))
        {
            weights.TryReadAny(out var poolerWeight, "bert.pooler.dense.weight", "pooler.dense.weight");
            weights.TryReadAny(out var poolerBias, "bert.pooler.dense.bias", "pooler.dense.bias");

            return new ClassificationHead(
                poolerWeight, poolerBias,
                weights.Read("classifier.weight"),
                weights.Read("classifier.bias"),
                useTanh: true);
        }

        return null;
    }

    /// <summary>Turns the encoder's hidden states into class logits.</summary>
    /// <param name="hidden">The final hidden states, one row per token.</param>
    internal double[] Apply(NdArray hidden)
    {
        // Every one of these layouts reads position 0 - the [CLS] or <s> token - rather than
        // pooling. That is what the head was trained on.
        var vector = hidden.Row(0).ToArray();

        if (_poolerWeight is not null && _poolerBias is not null)
        {
            vector = Linear(vector, _poolerWeight, _poolerBias);
            for (var i = 0; i < vector.Length; i++)
            {
                vector[i] = _useTanh ? Math.Tanh(vector[i]) : Math.Max(0, vector[i]);
            }
        }

        return Linear(vector, _classifierWeight, _classifierBias);
    }

    /// <summary>Applies a PyTorch-style linear layer, whose weight is stored (outputs, inputs).</summary>
    internal static double[] Linear(double[] input, NdArray weight, NdArray bias)
    {
        var outputs = weight.Shape[0];
        var inputs = weight.Shape[1];

        if (inputs != input.Length)
        {
            throw new InvalidDataException(
                $"A head layer expects {inputs} inputs but was given {input.Length}.");
        }

        var result = new double[outputs];
        for (var o = 0; o < outputs; o++)
        {
            var sum = bias.At(o);
            for (var i = 0; i < inputs; i++) sum += weight[o, i] * input[i];
            result[o] = sum;
        }

        return result;
    }
}

/// <summary>
/// The masked language modelling head: a transform block, then a projection back to the vocabulary.
/// </summary>
/// <remarks>
/// The output projection is usually <b>tied</b> to the input word embeddings and therefore absent
/// from the checkpoint. Falling back to the embedding matrix is not a shortcut, it is the model:
/// treating the missing tensor as "no head" would make fill-mask unavailable on most base
/// checkpoints, which are exactly the ones that have it.
/// </remarks>
internal sealed class MaskedLanguageHead
{
    private readonly NdArray _transformWeight;
    private readonly NdArray _transformBias;
    private readonly NdArray _normScale;
    private readonly NdArray _normShift;
    private readonly NdArray _decoder;
    private readonly NdArray? _decoderBias;

    private MaskedLanguageHead(
        NdArray transformWeight, NdArray transformBias,
        NdArray normScale, NdArray normShift,
        NdArray decoder, NdArray? decoderBias)
    {
        _transformWeight = transformWeight;
        _transformBias = transformBias;
        _normScale = normScale;
        _normShift = normShift;
        _decoder = decoder;
        _decoderBias = decoderBias;
    }

    /// <summary>Loads a masked language head, or returns null when the checkpoint has none.</summary>
    internal static MaskedLanguageHead? TryLoad(WeightStore weights, PretrainedConfig config, Encoder encoder)
    {
        if (!weights.TryReadAny(out var transformWeight,
            "cls.predictions.transform.dense.weight",
            "lm_head.dense.weight",
            "vocab_transform.weight"))
        {
            return null;
        }

        if (!weights.TryReadAny(out var transformBias,
            "cls.predictions.transform.dense.bias", "lm_head.dense.bias", "vocab_transform.bias"))
        {
            return null;
        }

        if (!weights.TryReadAny(out var normScale,
            "cls.predictions.transform.LayerNorm.weight", "cls.predictions.transform.LayerNorm.gamma",
            "lm_head.layer_norm.weight", "vocab_layer_norm.weight"))
        {
            return null;
        }

        if (!weights.TryReadAny(out var normShift,
            "cls.predictions.transform.LayerNorm.bias", "cls.predictions.transform.LayerNorm.beta",
            "lm_head.layer_norm.bias", "vocab_layer_norm.bias"))
        {
            return null;
        }

        // The decoder is tied to the embeddings in most checkpoints and simply is not stored.
        if (!weights.TryReadAny(out var decoder,
            "cls.predictions.decoder.weight", "lm_head.decoder.weight", "vocab_projector.weight"))
        {
            decoder = encoder.TokenEmbeddings;
        }

        weights.TryReadAny(out var decoderBias,
            "cls.predictions.bias", "cls.predictions.decoder.bias", "lm_head.bias", "vocab_projector.bias");

        return new MaskedLanguageHead(
            transformWeight, transformBias, normScale, normShift, decoder,
            decoderBias is { Size: > 0 } && decoderBias.Size == config.VocabularySize ? decoderBias : null);
    }

    /// <summary>Scores every vocabulary entry for one position's hidden state.</summary>
    /// <param name="hidden">The hidden state at the masked position.</param>
    internal double[] Apply(NdArray hidden)
    {
        var vector = ClassificationHead.Linear(hidden.ToArray(), _transformWeight, _transformBias);

        for (var i = 0; i < vector.Length; i++) vector[i] = Gelu(vector[i]);
        LayerNormalize(vector, _normScale, _normShift);

        var vocabulary = _decoder.Shape[0];
        var width = _decoder.Shape[1];
        var logits = new double[vocabulary];

        for (var v = 0; v < vocabulary; v++)
        {
            var sum = _decoderBias?.At(v) ?? 0.0;
            for (var d = 0; d < width; d++) sum += _decoder[v, d] * vector[d];
            logits[v] = sum;
        }

        return logits;
    }

    /// <summary>The exact GELU, as BERT's head uses.</summary>
    private static double Gelu(double x) => 0.5 * x * (1 + Erf(x / Math.Sqrt(2)));

    /// <summary>Abramowitz and Stegun 7.1.26, accurate to about 1.5e-7.</summary>
    private static double Erf(double x)
    {
        var sign = Math.Sign(x);
        x = Math.Abs(x);

        var t = 1.0 / (1.0 + 0.3275911 * x);
        var y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592) * t * Math.Exp(-x * x);

        return sign * y;
    }

    private static void LayerNormalize(double[] vector, NdArray scale, NdArray shift)
    {
        var mean = vector.Average();
        var variance = vector.Sum(v => (v - mean) * (v - mean)) / vector.Length;
        var denominator = Math.Sqrt(variance + 1e-12);

        for (var i = 0; i < vector.Length; i++)
        {
            vector[i] = ((vector[i] - mean) / denominator * scale.At(i)) + shift.At(i);
        }
    }
}
