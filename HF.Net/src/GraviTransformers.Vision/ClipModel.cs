using System.Text.Json;
using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviTokenizers;
using Gravicode.Science.GraviNum;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gravicode.HFNet.GraviTransformers.Vision;

/// <summary>The dimensions of one CLIP tower.</summary>
/// <param name="HiddenSize">Width of every hidden state.</param>
/// <param name="IntermediateSize">Width of the feed-forward layer.</param>
/// <param name="Layers">How many encoder blocks.</param>
/// <param name="Heads">Attention heads per block.</param>
/// <param name="Activation">The feed-forward activation; <c>quick_gelu</c> for OpenAI's checkpoints.</param>
/// <param name="LayerNormEpsilon">Epsilon inside every layer norm.</param>
public sealed record ClipTowerConfig(int HiddenSize, int IntermediateSize, int Layers, int Heads, string Activation, double LayerNormEpsilon);

/// <summary>A CLIP checkpoint's <c>config.json</c>.</summary>
public sealed record ClipConfig
{
    /// <summary>The text tower.</summary>
    public required ClipTowerConfig Text { get; init; }

    /// <summary>The vision tower.</summary>
    public required ClipTowerConfig Vision { get; init; }

    /// <summary>Token vocabulary size.</summary>
    public required int VocabularySize { get; init; }

    /// <summary>How many tokens the text tower takes, start and end included.</summary>
    public required int MaxPositions { get; init; }

    /// <summary>
    /// The end-of-text token id the pooled text vector is read at, or 2 for the original checkpoints,
    /// which instead read it at the highest id in the sequence - see <see cref="ClipModel"/>.
    /// </summary>
    public required int EndTokenId { get; init; }

    /// <summary>Input image edge length, in pixels.</summary>
    public required int ImageSize { get; init; }

    /// <summary>Patch edge length, in pixels.</summary>
    public required int PatchSize { get; init; }

    /// <summary>Width of the shared embedding space.</summary>
    public required int ProjectionSize { get; init; }

    /// <summary>Reads a <c>config.json</c>, taking transformers' defaults for anything it leaves out.</summary>
    public static ClipConfig Load(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;

        var modelType = root.TryGetProperty("model_type", out var type) ? type.GetString() : null;
        if (modelType is not ("clip" or null))
        {
            throw new NotSupportedException(
                $"This config is a '{modelType}' model, not CLIP. SigLIP, ALIGN and the other contrastive "
                + "models have different towers or losses; export them to ONNX and use GraviOptimum.");
        }

        var text = root.TryGetProperty("text_config", out var t) ? t : default;
        var vision = root.TryGetProperty("vision_config", out var v) ? v : default;

        return new ClipConfig
        {
            Text = Tower(text, 512, 2048, 12, 8),
            Vision = Tower(vision, 768, 3072, 12, 12),
            VocabularySize = Int(text, "vocab_size", 49408),
            MaxPositions = Int(text, "max_position_embeddings", 77),
            EndTokenId = Int(text, "eos_token_id", 49407),
            ImageSize = Int(vision, "image_size", 224),
            PatchSize = Int(vision, "patch_size", 32),
            ProjectionSize = Int(root, "projection_dim", 512),
        };
    }

    private static ClipTowerConfig Tower(JsonElement element, int hidden, int intermediate, int layers, int heads) => new(
        Int(element, "hidden_size", hidden),
        Int(element, "intermediate_size", intermediate),
        Int(element, "num_hidden_layers", layers),
        Int(element, "num_attention_heads", heads),
        element.ValueKind == JsonValueKind.Object && element.TryGetProperty("hidden_act", out var act) ? act.GetString() ?? "quick_gelu" : "quick_gelu",
        element.ValueKind == JsonValueKind.Object && element.TryGetProperty("layer_norm_eps", out var eps) ? eps.GetDouble() : 1e-5);

    private static int Int(JsonElement element, string name, int fallback)
        => element.ValueKind == JsonValueKind.Object && element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number
            ? value.GetInt32()
            : fallback;
}

/// <summary>
/// CLIP: images and text in one embedding space, and zero-shot image classification from it.
/// </summary>
/// <remarks>
/// <para>
/// Two encoders trained so an image and a sentence describing it land close together. Classifying
/// an image then needs no training at all: embed <c>"a photo of a {label}."</c> for each candidate
/// label and take a softmax over the image's similarity to each, scaled by the checkpoint's learned
/// temperature - which is what the <c>zero-shot-image-classification</c> pipeline does.
/// </para>
/// <para>
/// The vision tower is ViT with one extra layer norm before the first block, and no bias on its
/// patch projection. The text tower is the same pre-norm block run <b>causally</b>: each token
/// attends only to itself and the tokens before it, so the vector at the end-of-text token
/// summarises the sentence. Which position that is has two answers. Checkpoints whose config says
/// <c>eos_token_id: 2</c> - OpenAI's own - predate the field and read it at the highest token id in
/// the sequence, which is where <c>&lt;|endoftext|&gt;</c> is; newer ones read it at the first end
/// token. Both follow transformers.
/// </para>
/// <para>
/// OpenAI's checkpoints use <c>quick_gelu</c>, <c>x * sigmoid(1.702 x)</c> - a third GELU, neither
/// the exact one nor the tanh one. The activation is taken from the config, as everywhere else.
/// </para>
/// </remarks>
public sealed class ClipModel : IDisposable
{
    private readonly float[] _tokenTable;          // [vocab, hidden]
    private readonly double[] _textPositions;      // [maxPositions, hidden]
    private readonly EncoderBlock[] _textBlocks;
    private readonly Norm _textFinalNorm;
    private readonly Linear _textProjection;

    private readonly Linear _patchProjection;
    private readonly double[] _classEmbedding;     // [hidden]
    private readonly double[] _visionPositions;    // [1 + patches, hidden]
    private readonly Norm _preNorm;
    private readonly EncoderBlock[] _visionBlocks;
    private readonly Norm _postNorm;
    private readonly Linear _visualProjection;

    private bool _disposed;

    private ClipModel(string id, ClipConfig config, HfTokenizer? tokenizer, ImageProcessor processor, WeightStore weights)
    {
        Id = id;
        Config = config;
        Tokenizer = tokenizer;
        Processor = processor;

        if (!weights.Contains("text_model.embeddings.token_embedding.weight") || !weights.Contains("vision_model.embeddings.class_embedding"))
        {
            throw new NotSupportedException(
                $"'{id}' is not a full CLIP checkpoint: it needs both text_model.* and vision_model.* weights. "
                + "A CLIPTextModel or CLIPVisionModel alone cannot compare images with text.");
        }

        var text = config.Text;
        var vision = config.Vision;
        Activation.For(text.Activation);
        Activation.For(vision.Activation);

        _tokenTable = Floats(weights.Read("text_model.embeddings.token_embedding.weight"));
        _textPositions = weights.Read("text_model.embeddings.position_embedding.weight").AsContiguous().ToArray();
        _textBlocks = [.. Enumerable.Range(0, text.Layers).Select(i => Block(weights, $"text_model.encoder.layers.{i}", text))];
        _textFinalNorm = Norm.Load(weights, "text_model.final_layer_norm", text.HiddenSize, text.LayerNormEpsilon);
        _textProjection = Linear.From(weights.Read("text_projection.weight"), null);

        var patchValues = 3 * config.PatchSize * config.PatchSize;
        var patch = weights.Read("vision_model.embeddings.patch_embedding.weight");
        _patchProjection = Linear.From(new NdArray(patch.AsContiguous().ToArray(), vision.HiddenSize, patchValues), null);
        _classEmbedding = weights.Read("vision_model.embeddings.class_embedding").AsContiguous().ToArray();
        _visionPositions = weights.Read("vision_model.embeddings.position_embedding.weight").AsContiguous().ToArray();
        _preNorm = Norm.Load(weights, "vision_model.pre_layrnorm", vision.HiddenSize, vision.LayerNormEpsilon);
        _visionBlocks = [.. Enumerable.Range(0, vision.Layers).Select(i => Block(weights, $"vision_model.encoder.layers.{i}", vision))];
        _postNorm = Norm.Load(weights, "vision_model.post_layernorm", vision.HiddenSize, vision.LayerNormEpsilon);
        _visualProjection = Linear.From(weights.Read("visual_projection.weight"), null);

        LogitScale = weights.Read("logit_scale").At(0);

        var patches = (config.ImageSize / config.PatchSize) * (config.ImageSize / config.PatchSize);
        if (_visionPositions.Length != (1 + patches) * vision.HiddenSize)
        {
            throw new InvalidDataException(
                $"The vision tower has {_visionPositions.Length / vision.HiddenSize} position embeddings, but "
                + $"{config.ImageSize}px images in {config.PatchSize}px patches need {1 + patches}. "
                + "The config and the weights disagree.");
        }
    }

    /// <summary>The repository or directory this came from.</summary>
    public string Id { get; }

    /// <summary>The configuration.</summary>
    public ClipConfig Config { get; }

    /// <summary>The text tokenizer, or <c>null</c> for a directory that holds none.</summary>
    public HfTokenizer? Tokenizer { get; }

    /// <summary>How images are prepared.</summary>
    public ImageProcessor Processor { get; }

    /// <summary>The learned log temperature; similarities are multiplied by its exponential.</summary>
    public double LogitScale { get; }

    /// <summary>Downloads a CLIP checkpoint from the Hub and loads it.</summary>
    /// <param name="repoId">A model id such as <c>openai/clip-vit-base-patch32</c>.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    public static ClipModel Load(string repoId, string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var config = ClipConfig.Load(Hub.DownloadFile(repoId, "config.json", revision));
        var processor = ImageProcessorFor(repoId, revision);
        var tokenizer = HfTokenizer.FromPretrained(repoId, revision);
        var weights = WeightStore.FromPretrained(repoId, revision);

        try { return new ClipModel(repoId, config, tokenizer, processor, weights); }
        finally { weights.Dispose(); }
    }

    /// <summary>Loads a CLIP checkpoint from a directory.</summary>
    /// <param name="directory">A folder holding <c>config.json</c>, the weights and the tokenizer files.</param>
    public static ClipModel Open(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);

        var config = ClipConfig.Load(Path.Combine(directory, "config.json"));
        var preprocessor = Path.Combine(directory, "preprocessor_config.json");
        var processor = File.Exists(preprocessor) ? ImageProcessor.Load(preprocessor) : ImageProcessor.Clip;
        var weights = WeightStore.Open(directory);

        // A directory without tokenizer files still embeds images and token ids.
        var tokenizer = File.Exists(Path.Combine(directory, "tokenizer.json")) || File.Exists(Path.Combine(directory, "vocab.json"))
            ? HfTokenizer.FromDirectory(directory)
            : null;

        try { return new ClipModel(directory, config, tokenizer, processor, weights); }
        finally { weights.Dispose(); }
    }

    private static ImageProcessor ImageProcessorFor(string repoId, string revision)
    {
        try { return ImageProcessor.Load(Hub.DownloadFile(repoId, "preprocessor_config.json", revision)); }
        catch (HubException) { return ImageProcessor.Clip; }
    }

    // ------------------------------------------------------------------ zero-shot

    /// <summary>Classifies an image among labels it was never trained on.</summary>
    /// <param name="path">A path to an image.</param>
    /// <param name="labels">The candidate labels, in words.</param>
    /// <param name="template">
    /// The sentence each label is put into - <c>{0}</c> marks the label. The default is transformers'
    /// pipeline default; CLIP was trained on captions, and a bare word scores noticeably worse.
    /// </param>
    /// <returns>Every label with its probability, highest first.</returns>
    public IReadOnlyList<Classification> ZeroShot(string path, IEnumerable<string> labels, string template = "This is a photo of {0}.")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return ZeroShot(Processor.Read(path), labels, template);
    }

    /// <summary>Classifies prepared pixels among labels; see <see cref="ZeroShot(string, IEnumerable{string}, string)"/>.</summary>
    /// <param name="pixels">Normalised pixels, <c>[3, size, size]</c>.</param>
    /// <param name="labels">The candidate labels, in words.</param>
    /// <param name="template">The sentence each label is put into.</param>
    public IReadOnlyList<Classification> ZeroShot(NdArray pixels, IEnumerable<string> labels, string template = "This is a photo of {0}.")
    {
        ArgumentNullException.ThrowIfNull(labels);
        var names = labels.ToArray();
        if (names.Length == 0) throw new ArgumentException("At least one candidate label is needed.", nameof(labels));

        var logits = Logits(pixels, [.. names.Select(n => string.Format(template, n))]);

        var largest = logits.Max();
        var weights = logits.Select(l => Math.Exp(l - largest)).ToArray();
        var total = weights.Sum();

        return [.. names.Select((name, i) => new Classification(name, weights[i] / total, i)).OrderByDescending(c => c.Score)];
    }

    /// <summary>
    /// The image's similarity to each text, as the model scores it: cosine similarity times
    /// <c>exp(LogitScale)</c> - transformers' <c>logits_per_image</c>.
    /// </summary>
    /// <param name="pixels">Normalised pixels, <c>[3, size, size]</c>.</param>
    /// <param name="texts">The texts to compare with.</param>
    public double[] Logits(NdArray pixels, IReadOnlyList<string> texts)
    {
        ArgumentNullException.ThrowIfNull(texts);

        var image = Normalised(EmbedImage(pixels).ToArray());
        var scale = Math.Exp(LogitScale);
        var result = new double[texts.Count];

        Parallel.For(0, texts.Count, i =>
        {
            var text = Normalised(EmbedText(texts[i]).ToArray());
            var dot = 0.0;
            for (var d = 0; d < image.Length; d++) dot += image[d] * text[d];
            result[i] = scale * dot;
        });

        return result;
    }

    /// <summary>Cosine similarity between an image and a text, in <c>[-1, 1]</c>.</summary>
    public double Similarity(string imagePath, string text)
    {
        var image = Normalised(EmbedImage(imagePath).ToArray());
        var sentence = Normalised(EmbedText(text).ToArray());

        var dot = 0.0;
        for (var d = 0; d < image.Length; d++) dot += image[d] * sentence[d];
        return dot;
    }

    // ------------------------------------------------------------------ text

    /// <summary>A text's vector in the shared space, projected but not normalised - <c>get_text_features</c>.</summary>
    /// <param name="text">The text. Anything past the model's token limit (77) is cut, keeping the end token.</param>
    public NdArray EmbedText(string text)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(text);

        if (Tokenizer is null)
        {
            throw new InvalidOperationException(
                $"'{Id}' has no tokenizer files (tokenizer.json, or vocab.json and merges.txt), so it cannot "
                + "read text. Add them, or pass token ids to EmbedTokens.");
        }

        var ids = Tokenizer.Encode(text).Ids.ToArray();
        if (ids.Length > Config.MaxPositions) ids = [.. ids[..(Config.MaxPositions - 1)], ids[^1]];

        return EmbedTokens(ids);
    }

    /// <summary>The projected text vector for token ids that already carry the start and end tokens.</summary>
    public NdArray EmbedTokens(IReadOnlyList<int> ids)
    {
        var hidden = TextHidden(ids);
        var width = Config.Text.HiddenSize;
        var pooled = hidden.AsSpan(EndPosition(ids) * width, width).ToArray();

        return new NdArray(_textProjection.Apply(pooled, 1), Config.ProjectionSize);
    }

    /// <summary>
    /// Where the pooled text vector is read: the highest id for the original checkpoints
    /// (<c>eos_token_id == 2</c>), otherwise the first end token.
    /// </summary>
    internal int EndPosition(IReadOnlyList<int> ids)
    {
        if (Config.EndTokenId == 2)
        {
            var best = 0;
            for (var i = 1; i < ids.Count; i++) if (ids[i] > ids[best]) best = i;
            return best;
        }

        for (var i = 0; i < ids.Count; i++) if (ids[i] == Config.EndTokenId) return i;
        return ids.Count - 1;
    }

    /// <summary>The text tower's final hidden states, <c>[tokens, hidden]</c> row-major - <c>last_hidden_state</c>.</summary>
    internal double[] TextHidden(IReadOnlyList<int> ids)
    {
        if (ids.Count == 0 || ids.Count > Config.MaxPositions)
        {
            throw new ArgumentException($"The text tower takes 1 to {Config.MaxPositions} tokens; got {ids.Count}.", nameof(ids));
        }

        var width = Config.Text.HiddenSize;
        var rows = ids.Count;
        var hidden = new double[rows * width];

        for (var t = 0; t < rows; t++)
        {
            var id = ids[t];
            if ((uint)id >= (uint)Config.VocabularySize) throw new ArgumentOutOfRangeException(nameof(ids), id, "Token id outside the vocabulary.");

            for (var d = 0; d < width; d++) hidden[t * width + d] = _tokenTable[id * width + d] + _textPositions[t * width + d];
        }

        foreach (var block in _textBlocks) hidden = block.Forward(hidden, rows, mask: null, causal: true);

        return _textFinalNorm.Apply(hidden, rows);
    }

    // ------------------------------------------------------------------ images

    /// <summary>An image's vector in the shared space, projected but not normalised - <c>get_image_features</c>.</summary>
    public NdArray EmbedImage(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return EmbedImage(Processor.Read(path));
    }

    /// <summary>An image's vector in the shared space.</summary>
    public NdArray EmbedImage(Image<Rgb24> image) => EmbedImage(Processor.Convert(image));

    /// <summary>The projected vector for prepared pixels, <c>[3, size, size]</c>.</summary>
    public NdArray EmbedImage(NdArray pixels)
    {
        var hidden = VisionHidden(pixels);
        var pooled = _postNorm.Apply(hidden.AsSpan(0, Config.Vision.HiddenSize).ToArray(), 1);

        return new NdArray(_visualProjection.Apply(pooled, 1), Config.ProjectionSize);
    }

    /// <summary>The vision tower's final hidden states, before the post-norm - <c>last_hidden_state</c>.</summary>
    internal double[] VisionHidden(NdArray pixels)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(pixels);

        var size = Config.ImageSize;
        if (pixels.Rank != 3 || pixels.Shape[0] != 3 || pixels.Shape[1] != size || pixels.Shape[2] != size)
        {
            throw new ArgumentException(
                $"CLIP's vision tower takes [3 x {size} x {size}] pixels; got [{string.Join("x", pixels.Shape.ToArray())}]. "
                + "Prepare the image with Processor, which resizes and crops it.", nameof(pixels));
        }

        var width = Config.Vision.HiddenSize;
        var patches = (size / Config.PatchSize) * (size / Config.PatchSize);
        var projected = _patchProjection.Apply(VisionTransformer.GatherPatches(pixels, Config.PatchSize), patches);

        var rows = 1 + patches;
        var hidden = new double[rows * width];
        for (var d = 0; d < width; d++) hidden[d] = _classEmbedding[d] + _visionPositions[d];
        for (var i = 0; i < patches * width; i++) hidden[width + i] = projected[i] + _visionPositions[width + i];

        hidden = _preNorm.Apply(hidden, rows);
        foreach (var block in _visionBlocks) hidden = block.Forward(hidden, rows, mask: null);

        return hidden;
    }

    // ------------------------------------------------------------------ loading helpers

    private static EncoderBlock Block(WeightStore weights, string prefix, ClipTowerConfig tower)
    {
        var hidden = tower.HiddenSize;
        return new EncoderBlock(
            NormOrder.Pre,
            tower.Heads,
            Linear.Load(weights, $"{prefix}.self_attn.q_proj", hidden, hidden),
            Linear.Load(weights, $"{prefix}.self_attn.k_proj", hidden, hidden),
            Linear.Load(weights, $"{prefix}.self_attn.v_proj", hidden, hidden),
            Linear.Load(weights, $"{prefix}.self_attn.out_proj", hidden, hidden),
            Norm.Load(weights, $"{prefix}.layer_norm1", hidden, tower.LayerNormEpsilon),
            Linear.Load(weights, $"{prefix}.mlp.fc1", hidden, tower.IntermediateSize),
            Linear.Load(weights, $"{prefix}.mlp.fc2", tower.IntermediateSize, hidden),
            Norm.Load(weights, $"{prefix}.layer_norm2", hidden, tower.LayerNormEpsilon),
            Activation.For(tower.Activation));
    }

    private static float[] Floats(NdArray tensor)
    {
        var values = new float[tensor.Size];
        for (var i = 0; i < values.Length; i++) values[i] = (float)tensor.At(i);
        return values;
    }

    private static double[] Normalised(double[] vector)
    {
        var norm = Math.Sqrt(vector.Sum(v => v * v));
        return norm == 0 ? vector : [.. vector.Select(v => v / norm)];
    }

    /// <inheritdoc />
    public void Dispose() => _disposed = true;

    /// <inheritdoc />
    public override string ToString()
        => $"{Id} (CLIP: text {Config.Text.Layers}L {Config.Text.HiddenSize}H, vision {Config.Vision.Layers}L "
            + $"{Config.Vision.HiddenSize}H, {Config.ImageSize}px/{Config.PatchSize}, {Config.ProjectionSize}-d space)";
}
