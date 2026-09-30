using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviTokenizers;
using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviTransformers;

/// <summary>
/// A BERT-family encoder run straight from its checkpoint files, one layer at a time, for models
/// larger than memory.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="TransformerModel.Load"/> holds every parameter as a <see cref="double"/> - eight bytes
/// each, whatever the file stored - plus a float32 copy for the kernels. That is fine for bert-base
/// (110M parameters, about 1.3 GB) and not for a model of ten billion.
/// </para>
/// <para>
/// This keeps nothing resident but the norms. Each forward pass reads the embedding rows its tokens
/// need, then each layer's weights in turn, straight from the memory-mapped safetensors to float32,
/// runs the layer and lets it go. The operating system pages the file in and out, so peak memory is
/// one layer and the activations, and a checkpoint of any size runs on a machine that can hold one
/// layer of it - slower, because every pass reads the weights again, but correctly.
/// </para>
/// <para>
/// The arithmetic is <see cref="TransformerModel"/>'s own, kernel for kernel, so a model small
/// enough for both gives the same hidden states from each. Sharded checkpoints
/// (<c>model.safetensors.index.json</c>) are read across their shards. A PyTorch pickle cannot be
/// read in parts and is refused; convert it to safetensors.
/// </para>
/// </remarks>
public sealed class StreamingEncoder : IDisposable
{
    private readonly WeightStore _weights;
    private readonly Naming _names;
    private readonly Norm _embeddingNorm;
    private readonly Norm[] _attentionNorms;
    private readonly Norm[] _outputNorms;
    private readonly Func<double, double> _activation;
    private readonly int _vocabulary;
    private readonly int _positions;
    private readonly double[]? _segmentZero;   // token_type_embeddings[0], folded into each word row
    private bool _disposed;

    private StreamingEncoder(string id, PretrainedConfig config, HfTokenizer tokenizer, WeightStore weights)
    {
        Id = id;
        Config = config;
        Tokenizer = tokenizer;
        _weights = weights;
        _names = Naming.Detect(weights, config);
        _activation = Activation.For(config.Activation);

        foreach (var name in (string[])[_names.WordEmbeddings, _names.PositionEmbeddings, _names.Query(0) + ".weight"])
        {
            if (!weights.Contains(name))
            {
                throw new InvalidDataException(
                    $"'{id}' has no '{name}'; it is not a BERT-family checkpoint this can stream (prefix detected: '{_names.Prefix}').");
            }

            if (!weights.IsMapped(name))
            {
                throw new NotSupportedException(
                    $"'{id}' is stored as a PyTorch pickle, which cannot be read a layer at a time. Streaming needs "
                    + "safetensors: convert it with transformers' save_pretrained(safe_serialization=True), or load it "
                    + "whole with TransformerModel.Load if it fits in memory.");
            }
        }

        var width = config.HiddenSize;
        _vocabulary = weights.ShapeOf(_names.WordEmbeddings)[0];
        // RoBERTa's family reserves the first rows of its position table (PositionOffset, 2 for
        // RoBERTa), exactly as the whole-model loader trims them.
        _positions = weights.ShapeOf(_names.PositionEmbeddings)[0] - config.PositionOffset;
        _segmentZero = weights.Contains(_names.TokenTypeEmbeddings) ? weights.ReadRange(_names.TokenTypeEmbeddings, 0, width) : null;

        _embeddingNorm = LoadNorm(_names.EmbeddingNorm);
        _attentionNorms = [.. Enumerable.Range(0, config.Layers).Select(i => LoadNorm(_names.AttentionNorm(i)))];
        _outputNorms = [.. Enumerable.Range(0, config.Layers).Select(i => LoadNorm(_names.OutputNorm(i)))];
    }

    /// <summary>The repository or directory this reads from.</summary>
    public string Id { get; }

    /// <summary>The configuration.</summary>
    public PretrainedConfig Config { get; }

    /// <summary>The tokenizer.</summary>
    public HfTokenizer Tokenizer { get; }

    /// <summary>
    /// Downloads a checkpoint's files (without loading them) and opens them for streaming.
    /// </summary>
    /// <param name="repoId">A model id such as <c>bert-large-uncased</c>.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    /// <param name="progress">Receives download progress.</param>
    public static StreamingEncoder Load(string repoId, string revision = "main", IProgress<TransferProgress>? progress = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var config = PretrainedConfig.FromPretrained(repoId, revision);
        var weights = WeightStore.FromPretrained(repoId, revision, progress);

        try { return new StreamingEncoder(repoId, config, HfTokenizer.FromPretrained(repoId, revision), weights); }
        catch { weights.Dispose(); throw; }
    }

    /// <summary>Opens a checkpoint in a directory for streaming.</summary>
    /// <param name="directory">A folder holding <c>config.json</c>, safetensors weights (one file or sharded) and the tokenizer.</param>
    public static StreamingEncoder Open(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);

        var config = PretrainedConfig.Load(Path.Combine(directory, "config.json"));
        var weights = WeightStore.Open(directory);

        try { return new StreamingEncoder(directory, config, HfTokenizer.FromDirectory(directory), weights); }
        catch { weights.Dispose(); throw; }
    }

    /// <summary>The final hidden states for a text, one row per token - what <see cref="TransformerModel.Hidden"/> returns.</summary>
    /// <param name="text">The input.</param>
    /// <param name="maxLength">Truncation limit in tokens.</param>
    public NdArray Hidden(string text, int maxLength = 512)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var ids = Tokenizer.Encode(text).ToIdArray();
        if (ids.Length > maxLength) ids = ids[..maxLength];

        return new NdArray(Forward(ids), ids.Length, Config.HiddenSize);
    }

    /// <summary>A mean-pooled vector for a text, as <see cref="TransformerModel.Embed"/> computes it.</summary>
    public NdArray Embed(string text, int maxLength = 512)
    {
        var hidden = Hidden(text, maxLength);
        var width = Config.HiddenSize;
        var rows = hidden.Shape[0];
        var pooled = NdArray.Zeros(width);

        for (var i = 0; i < rows; i++)
        {
            for (var d = 0; d < width; d++) pooled[d] += hidden[i, d];
        }

        for (var d = 0; d < width; d++) pooled[d] /= rows;
        return pooled;
    }

    /// <summary>Runs token ids through every layer, loading each in turn.</summary>
    internal double[] Forward(int[] ids)
    {
        var rows = ids.Length;
        var width = Config.HiddenSize;

        if (rows > _positions)
        {
            throw new ArgumentException($"The sequence is {rows} tokens but the model has {_positions} position embeddings.", nameof(ids));
        }

        // Only the rows this text uses, in the loader's order of addition: (word + segment 0) + position.
        var positions = _weights.ReadRange(_names.PositionEmbeddings, (long)Config.PositionOffset * width, rows * width);
        var hidden = new double[rows * width];
        for (var t = 0; t < rows; t++)
        {
            var id = Math.Clamp(ids[t], 0, _vocabulary - 1);
            var word = _weights.ReadRange(_names.WordEmbeddings, (long)id * width, width);

            for (var d = 0; d < width; d++)
            {
                var folded = _segmentZero is null ? word[d] : word[d] + _segmentZero[d];
                hidden[t * width + d] = folded + positions[t * width + d];
            }
        }

        hidden = _embeddingNorm.Apply(hidden, rows);

        var mask = Enumerable.Repeat(1, rows).ToArray();
        for (var layer = 0; layer < Config.Layers; layer++)
        {
            // Built, run and dropped: nothing of it outlives this iteration.
            hidden = Block(layer).Forward(hidden, rows, mask);
        }

        return hidden;
    }

    private EncoderBlock Block(int layer)
    {
        var hidden = Config.HiddenSize;
        var inner = Config.IntermediateSize;

        return new EncoderBlock(
            NormOrder.Post,
            Config.Heads,
            Dense(_names.Query(layer), hidden, hidden),
            Dense(_names.Key(layer), hidden, hidden),
            Dense(_names.Value(layer), hidden, hidden),
            Dense(_names.AttentionOut(layer), hidden, hidden),
            _attentionNorms[layer],
            Dense(_names.Intermediate(layer), hidden, inner),
            Dense(_names.Output(layer), inner, hidden),
            _outputNorms[layer],
            _activation);
    }

    private Linear Dense(string name, int inputs, int outputs)
        => Linear.FromFloats(
            _weights.ReadFloats($"{name}.weight"),
            _weights.ReadRange($"{name}.bias", 0, outputs),
            inputs,
            outputs,
            inputsFirst: !_names.Transposed);

    /// <summary>A norm under either spelling: <c>weight</c>/<c>bias</c>, or TensorFlow's <c>gamma</c>/<c>beta</c>.</summary>
    private Norm LoadNorm(string prefix)
    {
        var width = Config.HiddenSize;
        var scale = _weights.Contains($"{prefix}.weight") ? $"{prefix}.weight" : $"{prefix}.gamma";
        var shift = _weights.Contains($"{prefix}.bias") ? $"{prefix}.bias" : $"{prefix}.beta";

        return new Norm(_weights.ReadRange(scale, 0, width), _weights.ReadRange(shift, 0, width), Config.LayerNormEpsilon);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _weights.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"StreamingEncoder({Id}, {Config.Layers} layers, {_weights.Description})";
}
