using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Transformers;

namespace Gravicode.HFNet.GraviTransformers.Vision;

/// <summary>One label and its probability.</summary>
/// <param name="Label">The class name from the checkpoint's own <c>id2label</c>.</param>
/// <param name="Score">Probability in <c>[0,1]</c>.</param>
/// <param name="Index">The class index.</param>
public readonly record struct Classification(string Label, double Score, int Index)
{
    /// <inheritdoc />
    public override string ToString() => $"{Label}: {Score:P2}";
}

/// <summary>
/// A pretrained Vision Transformer: an image in, a label or an embedding out.
/// </summary>
/// <remarks>
/// <para>
/// ViT is the same encoder block as BERT over a different embedding. An image is cut into a grid of
/// fixed squares, each square is projected to one vector, a learned <c>[CLS]</c> vector is put in
/// front, position embeddings are added, and from there it is a sequence of 197 tokens like any
/// other.
/// </para>
/// <para>
/// One structural difference makes this its own type rather than a mode of
/// <see cref="TransformerModel"/>: ViT is <b>pre-norm</b> and BERT is post-norm. ViT normalises
/// before each sublayer and adds the residual after it; BERT adds the residual first and normalises
/// the sum. The parameters have the same shapes either way, so a checkpoint loaded into the wrong
/// one loads "successfully" and returns confident nonsense.
/// </para>
/// </remarks>
public sealed class VisionTransformer : IDisposable
{
    private readonly WeightStore _weights;
    private readonly Block[] _blocks;
    private readonly LayerNorm _finalNorm;
    private readonly double[] _patchProjection;  // [hidden, patchValues], as the checkpoint stores it
    private readonly NdArray _patchBias;         // [hidden]
    private readonly NdArray _classToken;        // [hidden]
    private readonly NdArray _positions;         // [1 + patches, hidden]
    private readonly NdArray? _classifierWeight; // [hidden, classes]
    private readonly NdArray? _classifierBias;   // [classes]
    private bool _disposed;

    private VisionTransformer(
        string id, VisionConfig config, ImageProcessor processor, WeightStore weights,
        Block[] blocks, LayerNorm finalNorm, double[] patchProjection, NdArray patchBias,
        NdArray classToken, NdArray positions, NdArray? classifierWeight, NdArray? classifierBias)
    {
        Id = id;
        Config = config;
        Processor = processor;
        _weights = weights;
        _blocks = blocks;
        _finalNorm = finalNorm;
        _patchProjection = patchProjection;
        _patchBias = patchBias;
        _classToken = classToken;
        _positions = positions;
        _classifierWeight = classifierWeight;
        _classifierBias = classifierBias;
    }

    /// <summary>The repository this came from.</summary>
    public string Id { get; }

    /// <summary>The model's configuration.</summary>
    public VisionConfig Config { get; }

    /// <summary>How images are prepared for it.</summary>
    public ImageProcessor Processor { get; }

    /// <summary>Whether the checkpoint carries a classification head.</summary>
    public bool HasClassificationHead => _classifierWeight is not null;

    /// <summary>The class names, in index order.</summary>
    public IReadOnlyList<string> Labels =>
        [.. Enumerable.Range(0, Config.LabelCount).Select(i => Config.Label(i))];

    /// <summary>Downloads a vision model from the Hub and loads it.</summary>
    /// <param name="repoId">A model id such as <c>google/vit-base-patch16-224</c>.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    /// <exception cref="NotSupportedException">The architecture is not a ViT-style encoder.</exception>
    public static VisionTransformer Load(string repoId, string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var config = VisionConfig.FromPretrained(repoId, revision);
        var processor = Align(ImageProcessor.FromPretrained(repoId, revision), config);
        var weights = WeightStore.FromPretrained(repoId, revision);

        try { return Build(repoId, config, processor, weights); }
        catch { weights.Dispose(); throw; }
    }

    /// <summary>Loads a vision model from a directory that already holds one.</summary>
    /// <param name="directory">
    /// A folder containing <c>config.json</c>, the weights, and optionally
    /// <c>preprocessor_config.json</c>.
    /// </param>
    /// <remarks>
    /// The path a model that was never on the Hub takes - a fine-tune of your own, or a cached copy
    /// being inspected offline.
    /// </remarks>
    public static VisionTransformer Open(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);

        var config = VisionConfig.Load(Path.Combine(directory, "config.json"));
        VisionConfig.Require(config, directory);

        var preprocessor = Path.Combine(directory, "preprocessor_config.json");
        var processor = Align(
            File.Exists(preprocessor) ? ImageProcessor.Load(preprocessor) : ImageProcessor.ViT, config);

        var weights = WeightStore.Open(directory);

        try { return Build(directory, config, processor, weights); }
        catch { weights.Dispose(); throw; }
    }

    /// <summary>Makes the processor produce the edge length the model's positions were learned at.</summary>
    /// <remarks>
    /// The two can disagree: a repository may ship no <c>preprocessor_config.json</c> at all, or one
    /// written for a pipeline that crops after resizing. The model's own <c>image_size</c> is the
    /// one that cannot be negotiated - it is baked into how many position embeddings the checkpoint
    /// has - so it wins, and the alternative is a shape error at the first forward pass.
    /// </remarks>
    private static ImageProcessor Align(ImageProcessor processor, VisionConfig config)
        => processor.Size == config.ImageSize ? processor : processor with { Size = config.ImageSize };

    private static VisionTransformer Build(
        string id, VisionConfig config, ImageProcessor processor, WeightStore weights)
    {
        var prefix = weights.Contains("vit.embeddings.cls_token") ? "vit."
            : weights.Contains("deit.embeddings.cls_token") ? "deit."
            : "";

        var hidden = config.HiddenSize;
        var patchValues = config.Channels * config.PatchSize * config.PatchSize;

        // The patch embedding is stored as a convolution, [hidden, channels, k, k], but with the
        // kernel equal to the stride every output touches each input pixel exactly once - so it is
        // a plain matrix over the flattened patch, and unrolling it here avoids a convolution that
        // would never overlap anything.
        var projection = weights.Read($"{prefix}embeddings.patch_embeddings.projection.weight");
        if (projection.Size != (long)hidden * patchValues)
        {
            throw new InvalidDataException(
                $"The patch projection holds {projection.Size} values but a "
                + $"{config.PatchSize}x{config.PatchSize} patch of {config.Channels} channels into "
                + $"{hidden} dimensions needs {(long)hidden * patchValues}.");
        }

        // Left in the checkpoint's own (outputs, inputs) order. Transposing it to the shape the
        // loop "wants" makes the inner stride 768 doubles, and that cache miss per multiply costs
        // more than the whole projection.
        var patchProjection = new double[(int)projection.Size];
        for (var i = 0; i < patchProjection.Length; i++) patchProjection[i] = projection.At(i);

        var patchBias = Flat(weights.Read($"{prefix}embeddings.patch_embeddings.projection.bias"));
        var classToken = Flat(weights.Read($"{prefix}embeddings.cls_token"));
        var positions = Rows(weights.Read($"{prefix}embeddings.position_embeddings"), hidden);

        var expected = 1 + config.Patches;
        if (positions.Shape[0] != expected)
        {
            throw new InvalidDataException(
                $"The checkpoint has {positions.Shape[0]} position embeddings but a "
                + $"{config.ImageSize}px image in {config.PatchSize}px patches needs {expected}. "
                + "Interpolating them is not implemented; use the resolution the model was trained at.");
        }

        var blocks = new Block[config.Layers];
        for (var i = 0; i < config.Layers; i++) blocks[i] = Block.Load(weights, prefix, i, config);

        var finalNorm = ReadNorm(weights, $"{prefix}layernorm", hidden, config.LayerNormEpsilon);

        NdArray? classifierWeight = null;
        NdArray? classifierBias = null;

        if (weights.TryRead("classifier.weight", out var head) && head.Rank == 2 && head.Shape[1] == hidden)
        {
            var classes = head.Shape[0];
            classifierWeight = NdArray.Zeros(hidden, classes);

            for (var c = 0; c < classes; c++)
            {
                for (var d = 0; d < hidden; d++) classifierWeight[d, c] = head[c, d];
            }

            classifierBias = weights.TryRead("classifier.bias", out var bias) ? Flat(bias) : null;
        }

        return new VisionTransformer(
            id, config, processor, weights, blocks, finalNorm,
            patchProjection, patchBias, classToken, positions, classifierWeight, classifierBias);
    }

    /// <summary>Classifies an image file.</summary>
    /// <param name="path">A path to an image.</param>
    /// <param name="topK">How many classes to return; 0 for all of them.</param>
    /// <exception cref="NotSupportedException">The checkpoint has no classification head.</exception>
    public IReadOnlyList<Classification> Classify(string path, int topK = 5)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_classifierWeight is null)
        {
            throw new NotSupportedException(
                $"'{Id}' has no classification head - it is a feature extractor. Use Embed instead.");
        }

        var hidden = Forward(Processor.Read(path));
        var pooled = hidden.Row(0);   // the [CLS] vector, which is what the head was trained on

        var classes = _classifierWeight.Shape[1];
        var logits = new double[classes];

        for (var c = 0; c < classes; c++)
        {
            var sum = _classifierBias?.At(c) ?? 0.0;
            for (var d = 0; d < Config.HiddenSize; d++) sum += pooled.At(d) * _classifierWeight[d, c];
            logits[c] = sum;
        }

        var scores = Softmax(logits);
        var ranked = Enumerable.Range(0, classes)
            .Select(i => new Classification(Config.Label(i), scores[i], i))
            .OrderByDescending(p => p.Score);

        return topK > 0 ? [.. ranked.Take(topK)] : [.. ranked];
    }

    /// <summary>A single vector for an image, taken from the <c>[CLS]</c> position.</summary>
    /// <param name="path">A path to an image.</param>
    /// <remarks>
    /// <c>[CLS]</c> rather than a mean over the patches. Unlike a plain text encoder, ViT is
    /// pretrained with a classification objective on exactly that position, so it already carries
    /// the whole-image summary that mean pooling would have to reconstruct.
    /// </remarks>
    public NdArray Embed(string path)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return Forward(Processor.Read(path)).Row(0);
    }

    /// <summary>Cosine similarity between two images' embeddings, in <c>[-1,1]</c>.</summary>
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

    /// <summary>Runs the encoder over a prepared image and returns every position's hidden state.</summary>
    /// <param name="pixels">Normalised pixels, shaped <c>[channels, size, size]</c>.</param>
    public NdArray Forward(NdArray pixels)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(pixels);

        var hidden = Embeddings(pixels);

        // Every position is real - an image has no padding - so the mask is all ones rather than
        // an optimisation the attention could skip.
        var mask = new int[hidden.Shape[0]];
        Array.Fill(mask, 1);

        foreach (var block in _blocks) hidden = block.Forward(hidden, mask);

        return _finalNorm.Forward(hidden);
    }

    /// <summary>Builds the token sequence: the class token, then one vector per patch.</summary>
    /// <remarks>
    /// Internal rather than private so the tests can check the patch flattening on its own. The
    /// channel-major order is not visible in any end-to-end output - a scrambled projection still
    /// produces a confident label - so it needs to be asserted where it happens.
    /// </remarks>
    internal NdArray Embeddings(NdArray pixels)
    {
        var size = Config.ImageSize;
        var patch = Config.PatchSize;
        var grid = Config.Grid;
        var hidden = Config.HiddenSize;

        if (pixels.Rank != 3 || pixels.Shape[1] != size || pixels.Shape[2] != size)
        {
            throw new ArgumentException(
                $"The model wants [{Config.Channels}x{size}x{size}] but got "
                + $"[{string.Join("x", pixels.Shape.ToArray())}].", nameof(pixels));
        }

        var sequence = NdArray.Zeros(1 + Config.Patches, hidden);

        for (var d = 0; d < hidden; d++) sequence[0, d] = _classToken.At(d) + _positions[0, d];

        var values = new double[Config.Channels * patch * patch];

        for (var py = 0; py < grid; py++)
        {
            for (var px = 0; px < grid; px++)
            {
                // The patch is flattened channel-major, matching the convolution weight's own
                // [channels, ky, kx] layout - any other order silently scrambles the projection.
                var v = 0;
                for (var c = 0; c < Config.Channels; c++)
                {
                    for (var ky = 0; ky < patch; ky++)
                    {
                        for (var kx = 0; kx < patch; kx++)
                        {
                            values[v++] = pixels[c, py * patch + ky, px * patch + kx];
                        }
                    }
                }

                var row = 1 + py * grid + px;
                for (var d = 0; d < hidden; d++)
                {
                    sequence[row, d] = _patchBias.At(d)
                        + Simd.Dot(values, _patchProjection.AsSpan(d * values.Length, values.Length))
                        + _positions[row, d];
                }
            }
        }

        return sequence;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;

        _weights.Dispose();
        _disposed = true;
    }

    /// <inheritdoc />
    public override string ToString()
        => $"{Id} ({Config.Layers} layers, {Config.HiddenSize} hidden, "
            + $"{Config.Patches} patches, {(HasClassificationHead ? $"{Config.LabelCount} classes" : "no head")})";

    // ------------------------------------------------------------------ helpers

    private static double[] Softmax(double[] logits)
    {
        var largest = logits.Max();
        var total = 0.0;
        var result = new double[logits.Length];

        for (var i = 0; i < logits.Length; i++)
        {
            result[i] = Math.Exp(logits[i] - largest);
            total += result[i];
        }

        for (var i = 0; i < logits.Length; i++) result[i] /= total;
        return result;
    }

    /// <summary>Flattens a tensor of any rank to a vector, which is how the 1x1xH ones arrive.</summary>
    private static NdArray Flat(NdArray tensor)
    {
        var result = NdArray.Zeros((int)tensor.Size);
        for (var i = 0; i < tensor.Size; i++) result.SetAt(i, tensor.At(i));

        return result;
    }

    /// <summary>Reshapes a <c>[1, n, hidden]</c> tensor to <c>[n, hidden]</c>.</summary>
    private static NdArray Rows(NdArray tensor, int hidden)
    {
        var count = (int)(tensor.Size / hidden);
        var result = NdArray.Zeros(count, hidden);

        for (var i = 0; i < count; i++)
        {
            for (var d = 0; d < hidden; d++) result[i, d] = tensor.At(i * hidden + d);
        }

        return result;
    }

    private static LayerNorm ReadNorm(WeightStore weights, string prefix, int size, double epsilon)
    {
        var norm = new LayerNorm(size, epsilon);
        var scale = weights.Read($"{prefix}.weight");
        var shift = weights.Read($"{prefix}.bias");

        for (var i = 0; i < size; i++)
        {
            norm.Gamma.SetAt(i, scale.At(i));
            norm.Beta.SetAt(i, shift.At(i));
        }

        return norm;
    }

    /// <summary>
    /// One pre-norm encoder block.
    /// </summary>
    /// <remarks>
    /// Written here rather than reusing the foundation's <c>TransformerEncoderLayer</c>, which is
    /// post-norm. The attention and the projections are the foundation's; only the order of the
    /// norms and the residuals differs, and that order is the whole difference between a working
    /// ViT and a confident wrong answer.
    /// </remarks>
    private sealed class Block
    {
        private readonly LayerNorm _beforeAttention;
        private readonly MultiHeadAttention _attention;
        private readonly LayerNorm _beforeFeedForward;
        private readonly Linear _intermediate;
        private readonly Linear _output;

        private Block(
            LayerNorm beforeAttention, MultiHeadAttention attention,
            LayerNorm beforeFeedForward, Linear intermediate, Linear output)
        {
            _beforeAttention = beforeAttention;
            _attention = attention;
            _beforeFeedForward = beforeFeedForward;
            _intermediate = intermediate;
            _output = output;
        }

        internal static Block Load(WeightStore weights, string prefix, int index, VisionConfig config)
        {
            var block = $"{prefix}encoder.layer.{index}";
            var random = new GraviRandom(0);

            var attention = new MultiHeadAttention(
                new TransformerConfig(
                    VocabularySize: 1,
                    HiddenSize: config.HiddenSize,
                    Layers: config.Layers,
                    Heads: config.Heads,
                    IntermediateSize: config.IntermediateSize,
                    MaxPositions: 1 + config.Patches),
                random);

            Fill(weights, attention.Query, $"{block}.attention.attention.query");
            Fill(weights, attention.Key, $"{block}.attention.attention.key");
            Fill(weights, attention.Value, $"{block}.attention.attention.value");
            Fill(weights, attention.Output, $"{block}.attention.output.dense");

            return new Block(
                ReadNorm(weights, $"{block}.layernorm_before", config.HiddenSize, config.LayerNormEpsilon),
                attention,
                ReadNorm(weights, $"{block}.layernorm_after", config.HiddenSize, config.LayerNormEpsilon),
                Linear.Load(weights, $"{block}.intermediate.dense", config.HiddenSize, config.IntermediateSize),
                Linear.Load(weights, $"{block}.output.dense", config.IntermediateSize, config.HiddenSize));
        }

        internal NdArray Forward(NdArray hidden, int[] mask)
        {
            var attended = _attention.Forward(_beforeAttention.Forward(hidden), mask);
            var residual = Add(hidden, attended);

            var normed = _beforeFeedForward.Forward(residual);
            var projected = _output.Apply(_intermediate.Apply(normed, Activations.Gelu), null);

            return Add(residual, projected);
        }

        private static NdArray Add(NdArray a, NdArray b)
        {
            var result = NdArray.Zeros(a.Shape[0], a.Shape[1]);
            for (var i = 0; i < a.Size; i++) result.SetAt(i, a.At(i) + b.At(i));

            return result;
        }

        /// <summary>Copies a checkpoint matrix and bias into one of attention's projections.</summary>
        /// <remarks>
        /// Hugging Face stores a linear layer as (outputs, inputs) and the foundation's
        /// <see cref="DenseLayer"/> holds (inputs, outputs), so every one of these is transposed.
        /// All four are square here and would accept either reading in silence - the convention is
        /// settled by the non-square feed-forward pair, which <see cref="Linear"/> keeps in the
        /// checkpoint's own order instead.
        /// </remarks>
        private static void Fill(WeightStore weights, DenseLayer target, string name)
        {
            CheckpointLoader.CopyMatrix(
                weights.Read($"{name}.weight"), target.Weights, transposed: true, name);

            var bias = weights.Read($"{name}.bias");
            for (var i = 0; i < target.Bias.Size; i++) target.Bias.SetAt(i, bias.At(i));
        }

    }
}

/// <summary>
/// A fully connected layer held in the checkpoint's own memory layout.
/// </summary>
/// <remarks>
/// The feed-forward pair is where a ViT forward pass spends most of its time - 197 positions
/// through 768x3072 and back, twelve times - so it is worth not going through a general array type
/// for it. Two things make the difference. The weights stay in Hugging Face's <c>(outputs,
/// inputs)</c> order, so the inner loop walks one output's weights contiguously instead of striding
/// a row-major array by 24 KB per step; and the rows are independent, so they run in parallel.
/// Transposing to <c>(inputs, outputs)</c> and iterating the obvious way measured <b>five times
/// slower</b> than the sequential version it was meant to replace - the cache, not the arithmetic,
/// is what this loop is bound by.
/// </remarks>
internal sealed class Linear
{
    private readonly double[] _weights;   // [outputs, inputs], as the checkpoint stores it
    private readonly double[] _bias;
    private readonly int _inputs;
    private readonly int _outputs;

    private Linear(double[] weights, double[] bias, int inputs, int outputs)
    {
        _weights = weights;
        _bias = bias;
        _inputs = inputs;
        _outputs = outputs;
    }

    internal static Linear Load(WeightStore weights, string name, int inputs, int outputs)
    {
        var matrix = weights.Read($"{name}.weight");
        if (matrix.Size != (long)inputs * outputs)
        {
            throw new InvalidDataException(
                $"'{name}.weight' holds {matrix.Size} values but the layer is {inputs}x{outputs}.");
        }

        var flat = new double[inputs * outputs];
        for (var i = 0; i < flat.Length; i++) flat[i] = matrix.At(i);

        var biasTensor = weights.Read($"{name}.bias");
        var bias = new double[outputs];
        for (var i = 0; i < outputs; i++) bias[i] = biasTensor.At(i);

        return new Linear(flat, bias, inputs, outputs);
    }

    internal NdArray Apply(NdArray input, Func<double, double>? activation)
    {
        var rows = input.Shape[0];
        var source = input.ToArray();
        var target = new double[rows * _outputs];

        Parallel.For(0, rows, row =>
        {
            var x = source.AsSpan(row * _inputs, _inputs);
            var y = row * _outputs;

            for (var o = 0; o < _outputs; o++)
            {
                var sum = _bias[o] + Simd.Dot(x, _weights.AsSpan(o * _inputs, _inputs));
                target[y + o] = activation is null ? sum : activation(sum);
            }
        });

        return new NdArray(target, [rows, _outputs]);
    }

}

/// <summary>Vectorised primitives the vision encoder's hot loops share.</summary>
internal static class Simd
{
    /// <summary>Dot product of two equal-length spans, widened to whatever SIMD is available.</summary>
    /// <remarks>
    /// Both operands have to be contiguous, which is the point of keeping every weight matrix in
    /// the checkpoint's own (outputs, inputs) order: a strided operand cannot be loaded into a
    /// vector register at all, so the layout decision and this loop are one decision.
    /// </remarks>
    internal static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var width = Vector<double>.Count;
        var lanes = Vector<double>.Zero;
        var i = 0;

        for (; i <= a.Length - width; i += width)
        {
            lanes += new Vector<double>(a[i..]) * new Vector<double>(b[i..]);
        }

        var sum = Vector.Sum(lanes);
        for (; i < a.Length; i++) sum += a[i] * b[i];

        return sum;
    }
}
