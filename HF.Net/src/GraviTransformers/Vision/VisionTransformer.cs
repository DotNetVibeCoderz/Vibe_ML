using System.Collections.Concurrent;
using Gravicode.Science.GraviNum;

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
    private readonly EncoderBlock[] _blocks;
    private readonly Norm _finalNorm;
    private readonly Linear _patchProjection;    // [hidden, patchValues], as the checkpoint stores it
    private readonly NdArray _classToken;        // [hidden]
    private readonly NdArray _positions;         // [1 + patches, hidden], at the trained grid
    private readonly ConcurrentDictionary<(int Rows, int Columns), NdArray> _resized = new();
    private readonly NdArray? _classifierWeight; // [hidden, classes]
    private readonly NdArray? _classifierBias;   // [classes]
    private bool _disposed;

    private VisionTransformer(
        string id, VisionConfig config, ImageProcessor processor, WeightStore weights,
        EncoderBlock[] blocks, Norm finalNorm, Linear patchProjection,
        NdArray classToken, NdArray positions, NdArray? classifierWeight, NdArray? classifierBias)
    {
        Id = id;
        Config = config;
        Processor = processor;
        _weights = weights;
        _blocks = blocks;
        _finalNorm = finalNorm;
        _patchProjection = patchProjection;
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
    /// <param name="imageSize">
    /// The edge length to run at, or <c>null</c> for the one the model was trained at. Any multiple
    /// of the patch size works; the position embeddings are interpolated to the new grid.
    /// </param>
    /// <exception cref="NotSupportedException">The architecture is not a ViT-style encoder.</exception>
    public static VisionTransformer Load(string repoId, string revision = "main", int? imageSize = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var config = VisionConfig.FromPretrained(repoId, revision);
        var processor = Align(ImageProcessor.FromPretrained(repoId, revision), config, imageSize);
        var weights = WeightStore.FromPretrained(repoId, revision);

        try { return Build(repoId, config, processor, weights); }
        catch { weights.Dispose(); throw; }
    }

    /// <summary>Loads a vision model from a directory that already holds one.</summary>
    /// <param name="directory">
    /// A folder containing <c>config.json</c>, the weights, and optionally
    /// <c>preprocessor_config.json</c>.
    /// </param>
    /// <param name="imageSize">
    /// The edge length to run at, or <c>null</c> for the one the model was trained at.
    /// </param>
    /// <remarks>
    /// The path a model that was never on the Hub takes - a fine-tune of your own, or a cached copy
    /// being inspected offline.
    /// </remarks>
    public static VisionTransformer Open(string directory, int? imageSize = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);

        var config = VisionConfig.Load(Path.Combine(directory, "config.json"));
        VisionConfig.Require(config, directory);

        var preprocessor = Path.Combine(directory, "preprocessor_config.json");
        var processor = Align(
            File.Exists(preprocessor) ? ImageProcessor.Load(preprocessor) : ImageProcessor.ViT,
            config,
            imageSize);

        var weights = WeightStore.Open(directory);

        try { return Build(directory, config, processor, weights); }
        catch { weights.Dispose(); throw; }
    }

    /// <summary>Makes the processor produce the edge length the model will run at.</summary>
    /// <remarks>
    /// The processor and the model can disagree: a repository may ship no
    /// <c>preprocessor_config.json</c> at all, or one written for a pipeline that crops after
    /// resizing. Unless the caller asked for a size, the model's own <c>image_size</c> wins - it is
    /// the grid the position embeddings were learned on, so it is the one resolution that needs no
    /// interpolation.
    /// </remarks>
    private static ImageProcessor Align(ImageProcessor processor, VisionConfig config, int? imageSize)
    {
        var size = imageSize ?? config.ImageSize;

        if (size <= 0 || size % config.PatchSize != 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(imageSize), size,
                $"The image size has to be a positive multiple of the {config.PatchSize}px patch - "
                + $"{config.ImageSize} for the resolution this model was trained at. "
                + "A remainder would be a strip of pixels no patch covers.");
        }

        return processor.Size == size ? processor : processor with { Size = size };
    }

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

        // In the checkpoint's own (outputs, inputs) order, which is also the linear kernel's.
        var patchProjection = Linear.Load(
            weights, $"{prefix}embeddings.patch_embeddings.projection", patchValues, hidden);

        var classToken = Flat(weights.Read($"{prefix}embeddings.cls_token"));
        var positions = Rows(weights.Read($"{prefix}embeddings.position_embeddings"), hidden);

        var expected = 1 + config.Patches;
        if (positions.Shape[0] != expected)
        {
            // Interpolation starts from the grid the config names, so a table of any other length
            // means the config and the weights describe two different models - and guessing which
            // one is right shifts every patch's position.
            throw new InvalidDataException(
                $"The checkpoint has {positions.Shape[0]} position embeddings but its config says "
                + $"{config.ImageSize}px images in {config.PatchSize}px patches, which needs {expected}. "
                + "The config and the weights disagree; check that both came from the same repository.");
        }

        // Resolved before any block is loaded, so an unknown activation is refused up front.
        Activation.For(config.Activation);

        var blocks = new EncoderBlock[config.Layers];
        for (var i = 0; i < config.Layers; i++) blocks[i] = LoadBlock(weights, prefix, i, config);

        var finalNorm = Norm.Load(weights, $"{prefix}layernorm", hidden, config.LayerNormEpsilon);

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
            patchProjection, classToken, positions, classifierWeight, classifierBias);
    }

    /// <summary>Classifies an image file.</summary>
    /// <param name="path">A path to an image.</param>
    /// <param name="topK">How many classes to return; 0 for all of them.</param>
    /// <exception cref="NotSupportedException">The checkpoint has no classification head.</exception>
    public IReadOnlyList<Classification> Classify(string path, int topK = 5)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        RequireHead();

        return Classify(Processor.Read(path), topK);
    }

    /// <summary>Classifies pixels that have already been prepared.</summary>
    /// <param name="pixels">Normalised pixels, shaped <c>[channels, height, width]</c>.</param>
    /// <param name="topK">How many classes to return; 0 for all of them.</param>
    /// <exception cref="NotSupportedException">The checkpoint has no classification head.</exception>
    public IReadOnlyList<Classification> Classify(NdArray pixels, int topK = 5)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        RequireHead();

        var hidden = Forward(pixels);
        var pooled = hidden.Row(0);   // the [CLS] vector, which is what the head was trained on

        var classes = _classifierWeight!.Shape[1];
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

    private void RequireHead()
    {
        if (_classifierWeight is null)
        {
            throw new NotSupportedException(
                $"'{Id}' has no classification head - it is a feature extractor. Use Embed instead.");
        }
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
    /// <param name="pixels">
    /// Normalised pixels, shaped <c>[channels, height, width]</c>. Height and width need not be the
    /// trained size, only multiples of the patch size.
    /// </param>
    public NdArray Forward(NdArray pixels)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(pixels);

        var sequence = Embeddings(pixels);
        var rows = sequence.Shape[0];
        var hidden = sequence.ToArray();

        // Every position is real - an image has no padding - so there is no mask.
        foreach (var block in _blocks) hidden = block.Forward(hidden, rows, mask: null);

        return new NdArray(_finalNorm.Apply(hidden, rows), [rows, Config.HiddenSize]);
    }

    /// <summary>Builds the token sequence: the class token, then one vector per patch.</summary>
    /// <remarks>
    /// Internal rather than private so the tests can check the patch flattening on its own. The
    /// channel-major order is not visible in any end-to-end output - a scrambled projection still
    /// produces a confident label - so it needs to be asserted where it happens.
    /// </remarks>
    internal NdArray Embeddings(NdArray pixels)
    {
        var patch = Config.PatchSize;
        var hidden = Config.HiddenSize;

        if (pixels.Rank != 3 || pixels.Shape[0] != Config.Channels
            || pixels.Shape[1] <= 0 || pixels.Shape[1] % patch != 0
            || pixels.Shape[2] <= 0 || pixels.Shape[2] % patch != 0)
        {
            throw new ArgumentException(
                $"The model wants [{Config.Channels} x height x width] with both sides a multiple of "
                + $"{patch}px ({Config.ImageSize}x{Config.ImageSize} is what it was trained at) but got "
                + $"[{string.Join("x", pixels.Shape.ToArray())}].", nameof(pixels));
        }

        var rows = pixels.Shape[1] / patch;
        var columns = pixels.Shape[2] / patch;
        var positions = Positions(rows, columns);

        var height = pixels.Shape[1];
        var width = pixels.Shape[2];
        var source = pixels.ToArray();          // [channels, height, width], contiguous
        var count = rows * columns;
        var values = _patchProjection.Inputs;   // channels * patch * patch

        // Every patch flattened into one row of a [patches, values] matrix, so the projection is a
        // single call into the linear kernel. The flattening is channel-major, matching the
        // convolution weight's own [channels, ky, kx] layout - any other order silently scrambles
        // the projection. Each run of `patch` pixels along x is contiguous in both, so it is one copy.
        var gathered = new double[count * values];

        Parallel.For(0, count, p =>
        {
            var top = (p / columns) * patch;
            var left = (p % columns) * patch;
            var target = p * values;

            for (var c = 0; c < Config.Channels; c++)
            {
                for (var ky = 0; ky < patch; ky++)
                {
                    Array.Copy(source, (c * height + top + ky) * width + left, gathered, target, patch);
                    target += patch;
                }
            }
        });

        var projected = _patchProjection.Apply(gathered, count);
        var table = positions.ToArray();
        var sequence = new double[(1 + count) * hidden];

        for (var d = 0; d < hidden; d++) sequence[d] = _classToken.At(d) + table[d];

        for (var i = 0; i < count * hidden; i++) sequence[hidden + i] = projected[i] + table[hidden + i];

        return new NdArray(sequence, [1 + count, hidden]);
    }

    /// <summary>The position table for a grid of patches, interpolated if it is not the trained one.</summary>
    /// <remarks>
    /// Does what <c>interpolate_pos_encoding=True</c> does in transformers: the class token's
    /// position is kept as it is, and the patch positions are treated as a <c>hidden</c>-channel
    /// image and resized bicubically. Cached per grid, since every image at one resolution needs
    /// the same table.
    /// </remarks>
    internal NdArray Positions(int rows, int columns)
    {
        var grid = Config.Grid;
        if (rows == grid && columns == grid) return _positions;

        return _resized.GetOrAdd((rows, columns), key =>
        {
            var hidden = Config.HiddenSize;
            var patches = Bicubic(_positions, 1, grid, hidden, key.Rows, key.Columns);

            var table = NdArray.Zeros(1 + key.Rows * key.Columns, hidden);
            for (var d = 0; d < hidden; d++) table[0, d] = _positions[0, d];
            for (var i = 0; i < patches.Length; i++) table.SetAt(hidden + i, patches[i]);

            return table;
        });
    }

    /// <summary>
    /// Resizes a square grid of vectors, stored row-major from row <paramref name="first"/> of
    /// <paramref name="table"/>, to <paramref name="rows"/> by <paramref name="columns"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Matches <c>torch.nn.functional.interpolate(mode="bicubic", align_corners=False)</c>, which
    /// is what the reference uses. Any other resampler gives a model that runs and has quietly moved
    /// every patch. Three details decide agreement: the source coordinate is
    /// <c>(i + 0.5) * in / out - 0.5</c>; the kernel is Keys' cubic convolution with
    /// <c>a = -0.75</c>, not the <c>-0.5</c> most image libraries use; and a tap that falls off the
    /// grid repeats the edge rather than reading zero.
    /// </para>
    /// <para>
    /// No antialiasing, so shrinking the grid samples it rather than averaging it. That is also
    /// what the reference does.
    /// </para>
    /// </remarks>
    internal static double[] Bicubic(NdArray table, int first, int side, int hidden, int rows, int columns)
    {
        var (rowTaps, rowWeights) = CubicTaps(side, rows);
        var (columnTaps, columnWeights) = CubicTaps(side, columns);

        var result = new double[rows * columns * hidden];

        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < columns; c++)
            {
                var target = (r * columns + c) * hidden;

                for (var i = 0; i < 4; i++)
                {
                    var sourceRow = first + rowTaps[r * 4 + i] * side;

                    for (var j = 0; j < 4; j++)
                    {
                        var weight = rowWeights[r * 4 + i] * columnWeights[c * 4 + j];
                        var source = sourceRow + columnTaps[c * 4 + j];

                        for (var d = 0; d < hidden; d++) result[target + d] += weight * table[source, d];
                    }
                }
            }
        }

        return result;
    }

    /// <summary>For each output index, the four source indices it reads and their weights.</summary>
    private static (int[] Taps, double[] Weights) CubicTaps(int input, int output)
    {
        const double A = -0.75;

        var taps = new int[output * 4];
        var weights = new double[output * 4];
        var scale = (double)input / output;

        for (var o = 0; o < output; o++)
        {
            // Deliberately not clamped at zero: torch clamps the source coordinate for the linear
            // modes only, and clamping it here shifts the first row of every upscaled grid.
            var real = scale * (o + 0.5) - 0.5;
            var floor = (int)Math.Floor(real);
            var t = real - floor;

            weights[o * 4 + 0] = Far(t + 1);
            weights[o * 4 + 1] = Near(t);
            weights[o * 4 + 2] = Near(1 - t);
            weights[o * 4 + 3] = Far(2 - t);

            for (var k = 0; k < 4; k++) taps[o * 4 + k] = Math.Clamp(floor - 1 + k, 0, input - 1);
        }

        return (taps, weights);

        static double Near(double x) => ((A + 2) * x - (A + 3)) * x * x + 1;         // |x| <= 1
        static double Far(double x) => ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;    // 1 < |x| < 2
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

    /// <summary>Loads one pre-norm encoder block.</summary>
    /// <remarks>
    /// Pre-norm is why ViT cannot share BERT's block. Only the order of the norms and the residuals
    /// differs, and that order is the whole difference between a working ViT and a confident wrong
    /// answer: every parameter has the same shape either way.
    /// </remarks>
    private static EncoderBlock LoadBlock(WeightStore weights, string prefix, int index, VisionConfig config)
    {
        var block = $"{prefix}encoder.layer.{index}";
        var hidden = config.HiddenSize;

        return new EncoderBlock(
            NormOrder.Pre,
            config.Heads,
            Linear.Load(weights, $"{block}.attention.attention.query", hidden, hidden),
            Linear.Load(weights, $"{block}.attention.attention.key", hidden, hidden),
            Linear.Load(weights, $"{block}.attention.attention.value", hidden, hidden),
            Linear.Load(weights, $"{block}.attention.output.dense", hidden, hidden),
            Norm.Load(weights, $"{block}.layernorm_before", hidden, config.LayerNormEpsilon),
            Linear.Load(weights, $"{block}.intermediate.dense", hidden, config.IntermediateSize),
            Linear.Load(weights, $"{block}.output.dense", config.IntermediateSize, hidden),
            Norm.Load(weights, $"{block}.layernorm_after", hidden, config.LayerNormEpsilon),
            Activation.For(config.Activation));
    }
}
