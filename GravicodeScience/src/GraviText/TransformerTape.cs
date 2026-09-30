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
    /// The exact Gaussian error linear unit, <c>0.5 x (1 + erf(x / sqrt 2))</c>.
    /// </summary>
    /// <remarks>
    /// What a Hugging Face config means by <c>hidden_act: "gelu"</c> - BERT's and ViT's. The tanh
    /// form <see cref="Gelu"/> differs from it by up to about 4e-4 per value, so a model trained with
    /// one and run with the other is a slightly different model, and so is its gradient.
    /// </remarks>
    public static Tensor GeluExact(Tensor x)
        => Tensor.Constant(0.5) * x * (Tensor.Constant(1.0) + (x * Tensor.Constant(1.0 / Math.Sqrt(2.0))).Erf());

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
        => MultiHeadAttention(x, query, key, value, output, heads, attentionMask, null, null, null, null);

    /// <summary>
    /// Multi-head scaled dot-product self-attention with a bias on each projection, as every
    /// pretrained BERT has.
    /// </summary>
    /// <param name="x">One row per token.</param>
    /// <param name="query">Query projection, width by width.</param>
    /// <param name="key">Key projection.</param>
    /// <param name="value">Value projection.</param>
    /// <param name="output">Projection applied to the concatenated heads.</param>
    /// <param name="heads">Number of attention heads.</param>
    /// <param name="attentionMask">Optional per-token mask; zero marks padding.</param>
    /// <param name="queryBias">Bias added after the query projection, or <c>null</c> for none.</param>
    /// <param name="keyBias">Bias added after the key projection, or <c>null</c>.</param>
    /// <param name="valueBias">Bias added after the value projection, or <c>null</c>.</param>
    /// <param name="outputBias">Bias added after the output projection, or <c>null</c>.</param>
    /// <remarks>
    /// Without the biases this is a slightly different model from the one a checkpoint describes,
    /// and a gradient taken through it is the gradient of that different model: it trains, it
    /// converges, and the adapters or weights it produces are quietly wrong for the real one. The
    /// query bias does matter to the scores - it adds a per-head term to every one of them - while
    /// the key bias adds a constant to each query's row, which the softmax cancels; both are kept
    /// because a checkpoint has both.
    /// </remarks>
    public static Tensor MultiHeadAttention(Tensor x, Tensor query, Tensor key, Tensor value,
        Tensor output, int heads, int[]? attentionMask,
        Tensor? queryBias, Tensor? keyBias, Tensor? valueBias, Tensor? outputBias)
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

        var q = queryBias is null ? x.MatMul(query) : x.MatMul(query) + queryBias;
        var k = keyBias is null ? x.MatMul(key) : x.MatMul(key) + keyBias;
        var v = valueBias is null ? x.MatMul(value) : x.MatMul(value) + valueBias;

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

        var projected = combined!.MatMul(output);
        return outputBias is null ? projected : projected + outputBias;
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
        /// <summary>Bias after the query projection; <c>null</c> for none, as a freshly built block has.</summary>
        public Tensor? QueryBias { get; init; }

        /// <summary>Bias after the key projection.</summary>
        public Tensor? KeyBias { get; init; }

        /// <summary>Bias after the value projection.</summary>
        public Tensor? ValueBias { get; init; }

        /// <summary>Bias after the attention output projection.</summary>
        public Tensor? AttentionOutputBias { get; init; }

        /// <summary>Every parameter, for handing to an optimiser - the biases too, when present.</summary>
        public IEnumerable<Tensor> Parameters =>
        [
            Query, Key, Value, AttentionOutput, AttentionGamma, AttentionBeta,
            Intermediate, IntermediateBias, OutputProjection, OutputBias, OutputGamma, OutputBeta,
            .. new[] { QueryBias, KeyBias, ValueBias, AttentionOutputBias }.OfType<Tensor>(),
        ];

        /// <summary>
        /// Copies a block's weights onto the tape, attention biases included - the pretrained
        /// weights a checkpoint loaded into the forward-only classes, made trainable.
        /// </summary>
        /// <param name="layer">The block to copy.</param>
        /// <remarks>
        /// The tensors are copies: training them leaves <paramref name="layer"/> as it was.
        /// </remarks>
        public static EncoderWeights From(TransformerEncoderLayer layer)
        {
            ArgumentNullException.ThrowIfNull(layer);
            static Tensor Copy(NdArray values) => Tensor.Parameter(values.Copy());

            var attention = layer.Attention;
            return new EncoderWeights(
                Copy(attention.Query.Weights), Copy(attention.Key.Weights), Copy(attention.Value.Weights),
                Copy(attention.Output.Weights),
                Copy(layer.AttentionNorm.Gamma), Copy(layer.AttentionNorm.Beta),
                Copy(layer.Intermediate.Weights), Copy(layer.Intermediate.Bias),
                Copy(layer.OutputProjection.Weights), Copy(layer.OutputProjection.Bias),
                Copy(layer.OutputNorm.Gamma), Copy(layer.OutputNorm.Beta))
            {
                QueryBias = Copy(attention.Query.Bias),
                KeyBias = Copy(attention.Key.Bias),
                ValueBias = Copy(attention.Value.Bias),
                AttentionOutputBias = Copy(attention.Output.Bias),
            };
        }
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
    /// <param name="x">One row per token.</param>
    /// <param name="w">The block's weights; its attention biases are applied when present.</param>
    /// <param name="heads">Number of attention heads.</param>
    /// <param name="attentionMask">Optional per-token mask; zero marks padding.</param>
    /// <param name="exactGelu">
    /// Whether the feed-forward uses the exact GELU - what BERT's and ViT's configs mean by "gelu" -
    /// rather than the tanh approximation the forward-only classes use.
    /// </param>
    /// <param name="epsilon">The layer norms' epsilon: 1e-12 for BERT, 1e-5 for RoBERTa.</param>
    public static Tensor EncoderLayer(Tensor x, EncoderWeights w, int heads, int[]? attentionMask = null,
        bool exactGelu = false, double epsilon = 1e-12)
    {
        var attended = LayerNorm(
            x + MultiHeadAttention(x, w.Query, w.Key, w.Value, w.AttentionOutput, heads, attentionMask,
                w.QueryBias, w.KeyBias, w.ValueBias, w.AttentionOutputBias),
            w.AttentionGamma, w.AttentionBeta, epsilon);

        var inner = Dense(attended, w.Intermediate, w.IntermediateBias);
        var expanded = Dense(exactGelu ? GeluExact(inner) : Gelu(inner), w.OutputProjection, w.OutputBias);

        return LayerNorm(attended + expanded, w.OutputGamma, w.OutputBeta, epsilon);
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
