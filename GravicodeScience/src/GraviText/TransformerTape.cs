using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Autodiff;

namespace Gravicode.Science.GraviText.Transformers;

/// <summary>
/// The transformer encoder expressed on the autodiff tape, so it can be trained.
/// </summary>
/// <remarks>
/// <para>
/// The classes in <c>Transformers.cs</c> compute a forward pass only. That was enough for what
/// they were for — running a reference architecture, measuring inference cost, inspecting
/// attention — but it meant a freshly built model stayed randomly initialised for ever, because
/// nothing could adjust its weights. These functions are the same architecture written on the
/// tape, which makes the weights reachable by a gradient.
/// </para>
/// <para>
/// Every layer here composes from operations that already existed for the graph networks:
/// <c>MatMul</c>, <c>Sum</c> along an axis, <c>Exp</c>, <c>Sqrt</c>, <c>Tanh</c> and the column
/// slice/concat pair. Nothing needed a bespoke kernel — not layer normalisation, not the attention
/// softmax — so nothing here needed its own derivation either.
/// </para>
/// </remarks>
public static class TransformerTape
{
    /// <summary>
    /// Layer normalisation over the feature axis, then a learned scale and shift.
    /// </summary>
    /// <param name="x">One row per token.</param>
    /// <param name="gamma">Per-feature scale, length equal to the width.</param>
    /// <param name="beta">Per-feature shift.</param>
    /// <param name="epsilon">Guards the divide when a row is constant.</param>
    /// <remarks>
    /// Normalising per token rather than per batch is what makes a transformer indifferent to
    /// batch size. Written out as mean, centring, variance and rescale rather than as one kernel:
    /// each step is an operation the tape already differentiates, so the notoriously fiddly
    /// layer-norm backward pass never has to be written down.
    /// </remarks>
    public static Tensor LayerNorm(Tensor x, Tensor gamma, Tensor beta, double epsilon = 1e-12)
    {
        var width = x.Shape[1];
        var rows = x.Shape[0];

        var mean = x.Sum(axis: 1).Reshape(rows, 1) / Tensor.Constant((double)width);
        var centred = x - mean;

        var variance = (centred * centred).Sum(axis: 1).Reshape(rows, 1) / Tensor.Constant((double)width);
        var deviation = (variance + Tensor.Constant(epsilon)).Sqrt();

        return centred / deviation * gamma + beta;
    }

    /// <summary>A fully connected layer: <c>x W + b</c>.</summary>
    public static Tensor Dense(Tensor x, Tensor weights, Tensor bias) => x.MatMul(weights) + bias;

    /// <summary>
    /// The Gaussian error linear unit, in the tanh approximation BERT uses.
    /// </summary>
    /// <remarks>
    /// Composed rather than fused. <c>MathUtil.Tanh</c> routes through <c>Math.Exp</c> for speed on
    /// the scalar path; here the tape's own <c>Tanh</c> is used, whose derivative comes from the
    /// forward result.
    /// </remarks>
    public static Tensor Gelu(Tensor x)
    {
        var inner = Tensor.Constant(Math.Sqrt(2.0 / Math.PI))
                    * (x + Tensor.Constant(0.044715) * x.Pow(3));

        return Tensor.Constant(0.5) * x * (Tensor.Constant(1.0) + inner.Tanh());
    }

    /// <summary>
    /// Softmax across each row.
    /// </summary>
    /// <remarks>
    /// The row maximum is subtracted before exponentiating and enters as a <em>constant</em>.
    /// That is exact rather than an approximation: softmax is unchanged by a shift within a row,
    /// so the shift carries no gradient — the same argument as
    /// <c>GnnTape.SegmentSoftmax</c>, and the reason attention scores of a few hundred do not
    /// overflow here.
    /// </remarks>
    public static Tensor SoftmaxRows(Tensor scores)
    {
        var rows = scores.Shape[0];
        var columns = scores.Shape[1];

        var maxima = NdArray.Zeros(rows, 1);
        for (var i = 0; i < rows; i++)
        {
            var max = double.NegativeInfinity;
            for (var j = 0; j < columns; j++) max = Math.Max(max, scores.Value[i, j]);
            maxima[i, 0] = double.IsNegativeInfinity(max) ? 0.0 : max;
        }

        var weights = (scores - Tensor.Constant(maxima)).Exp();
        return weights / weights.Sum(axis: 1).Reshape(rows, 1);
    }

    /// <summary>
    /// Multi-head scaled dot-product self-attention.
    /// </summary>
    /// <param name="x">One row per token.</param>
    /// <param name="query">Query projection, width by width.</param>
    /// <param name="key">Key projection.</param>
    /// <param name="value">Value projection.</param>
    /// <param name="output">Projection applied to the concatenated heads.</param>
    /// <param name="heads">Number of attention heads.</param>
    /// <param name="attentionMask">Optional per-token mask; zero marks padding.</param>
    /// <remarks>
    /// The <c>sqrt(headSize)</c> divisor is not cosmetic. Without it the dot products grow with
    /// width, the softmax saturates, and the gradient through it goes to zero — the model would
    /// still run and simply refuse to learn.
    /// </remarks>
    public static Tensor MultiHeadAttention(Tensor x, Tensor query, Tensor key, Tensor value,
        Tensor output, int heads, int[]? attentionMask = null)
    {
        var length = x.Shape[0];
        var width = x.Shape[1];

        if (width % heads != 0)
            throw new ArgumentException($"Width {width} is not divisible by {heads} heads.");

        var headSize = width / heads;
        var scale = Tensor.Constant(1.0 / Math.Sqrt(headSize));

        // Masked positions get a large negative score, which the softmax then sends to zero. The
        // mask is additive rather than multiplicative so it survives the exponential.
        Tensor? maskBias = null;
        if (attentionMask is not null)
        {
            var bias = NdArray.Zeros(length, length);
            for (var i = 0; i < length; i++)
                for (var j = 0; j < length; j++)
                    if (attentionMask[j] == 0) bias[i, j] = -1e9;
            maskBias = Tensor.Constant(bias);
        }

        var q = x.MatMul(query);
        var k = x.MatMul(key);
        var v = x.MatMul(value);

        Tensor? combined = null;
        for (var head = 0; head < heads; head++)
        {
            var offset = head * headSize;
            var qh = TensorOps.SliceColumns(q, offset, headSize);
            var kh = TensorOps.SliceColumns(k, offset, headSize);
            var vh = TensorOps.SliceColumns(v, offset, headSize);

            var scores = qh.MatMul(kh.T()) * scale;
            if (maskBias is not null) scores += maskBias;

            var attended = SoftmaxRows(scores).MatMul(vh);
            combined = combined is null ? attended : TensorOps.ConcatColumns(combined, attended);
        }

        return combined!.MatMul(output);
    }

    /// <summary>Weights of one encoder block, as tape parameters.</summary>
    /// <remarks>
    /// A record of tensors rather than of arrays: the same objects are handed to the optimiser, so
    /// a step updates exactly what the next forward pass will read.
    /// </remarks>
    public sealed record EncoderWeights(
        Tensor Query, Tensor Key, Tensor Value, Tensor AttentionOutput,
        Tensor AttentionGamma, Tensor AttentionBeta,
        Tensor Intermediate, Tensor IntermediateBias,
        Tensor OutputProjection, Tensor OutputBias,
        Tensor OutputGamma, Tensor OutputBeta)
    {
        /// <summary>Every parameter, for handing to an optimiser.</summary>
        public IEnumerable<Tensor> Parameters =>
        [
            Query, Key, Value, AttentionOutput, AttentionGamma, AttentionBeta,
            Intermediate, IntermediateBias, OutputProjection, OutputBias, OutputGamma, OutputBeta,
        ];
    }

    /// <summary>
    /// One encoder block: self-attention then a feed-forward network, each inside a residual
    /// connection and layer normalisation.
    /// </summary>
    /// <remarks>
    /// The residual connections are what make depth trainable — each block learns a correction to
    /// its input rather than a fresh representation, so the gradient has a short path back to the
    /// embeddings however deep the stack is.
    /// </remarks>
    public static Tensor EncoderLayer(Tensor x, EncoderWeights w, int heads, int[]? attentionMask = null)
    {
        var attended = LayerNorm(
            x + MultiHeadAttention(x, w.Query, w.Key, w.Value, w.AttentionOutput, heads, attentionMask),
            w.AttentionGamma, w.AttentionBeta);

        var expanded = Dense(Gelu(Dense(attended, w.Intermediate, w.IntermediateBias)),
            w.OutputProjection, w.OutputBias);

        return LayerNorm(attended + expanded, w.OutputGamma, w.OutputBeta);
    }

    /// <summary>
    /// Looks up token embeddings and adds the position embeddings for the sequence.
    /// </summary>
    /// <remarks>
    /// Attention is order-blind on its own — permuting the tokens permutes the output and changes
    /// nothing else — so position has to be added to the representation rather than implied by it.
    /// The lookup is a <c>Gather</c>, which means a token appearing twice correctly accumulates
    /// both occurrences' gradient into its one embedding row.
    /// </remarks>
    public static Tensor Embed(Tensor tokenEmbeddings, Tensor positionEmbeddings, int[] tokenIds)
    {
        var positions = new int[tokenIds.Length];
        for (var i = 0; i < tokenIds.Length; i++) positions[i] = i;

        return TensorOps.Gather(tokenEmbeddings, tokenIds)
               + TensorOps.Gather(positionEmbeddings, positions);
    }

    /// <summary>
    /// Averages the token rows, honouring a padding mask.
    /// </summary>
    /// <remarks>
    /// Mean pooling over real tokens is the usual way to turn a sequence into one vector for
    /// classification. Padding has to be excluded explicitly: averaging it in would make the
    /// representation depend on how much padding a batch happened to need.
    /// </remarks>
    public static Tensor MeanPool(Tensor x, int[]? attentionMask = null)
    {
        var length = x.Shape[0];

        var weights = NdArray.Zeros(1, length);
        var count = 0;
        for (var i = 0; i < length; i++)
            if (attentionMask is null || attentionMask[i] != 0) { weights[0, i] = 1.0; count++; }

        if (count == 0) throw new ArgumentException("The attention mask leaves no tokens to pool.");

        for (var i = 0; i < length; i++) weights[0, i] /= count;
        return Tensor.Constant(weights).MatMul(x);
    }
}
