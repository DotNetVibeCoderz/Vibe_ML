using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Transformers;
using Encoder = Gravicode.Science.GraviText.Transformers.TransformerModel;

namespace Gravicode.HFNet.GraviTransformers;

/// <summary>What a checkpoint load found and what it could not.</summary>
/// <param name="Loaded">Parameter names that were read and applied.</param>
/// <param name="Missing">Parameters the model wanted and the checkpoint did not have.</param>
/// <param name="Prefix">The parameter-name prefix that was detected, for example <c>bert.</c>.</param>
public sealed record LoadReport(IReadOnlyList<string> Loaded, IReadOnlyList<string> Missing, string Prefix)
{
    /// <summary>Whether every parameter the encoder needs was found.</summary>
    public bool IsComplete => Missing.Count == 0;

    /// <inheritdoc />
    public override string ToString()
        => IsComplete
            ? $"Loaded {Loaded.Count} tensors under '{Prefix}'."
            : $"Loaded {Loaded.Count} tensors under '{Prefix}', {Missing.Count} missing: "
                + string.Join(", ", Missing.Take(5)) + (Missing.Count > 5 ? ", ..." : "");
}

/// <summary>
/// Fills a <see cref="Encoder"/> encoder from a Hugging Face checkpoint.
/// </summary>
/// <remarks>
/// <para>
/// Three things vary between families and all three are detected rather than assumed: the prefix
/// parameters live under (<c>bert.</c>, <c>roberta.</c>, <c>distilbert.</c> or nothing at all), the
/// spelling of each block's parts, and whether weight matrices are stored transposed. The transpose
/// is checked against the feed-forward weight, which is the only non-square matrix in a block -
/// the attention projections are square and accept either reading silently.
/// </para>
/// <para>
/// <b>Token type embeddings are folded into the word embeddings.</b> The encoder underneath has no
/// segment input, but for a single-sequence input every position gets segment 0, so adding
/// <c>token_type_embeddings[0]</c> to every row of the word embedding matrix is exactly equivalent.
/// It is not equivalent for a sentence pair, and <see cref="SupportsPairs"/> reports that.
/// </para>
/// </remarks>
public static class CheckpointLoader
{
    /// <summary>Whether a model loaded this way reproduces sentence-pair inputs exactly.</summary>
    /// <remarks>
    /// False: the segment-1 embedding cannot be folded away the way segment 0 can. Pair inputs
    /// still run and still give sensible output, but they are not bit-identical to the reference.
    /// </remarks>
    public const bool SupportsPairs = false;

    /// <summary>Builds an encoder from a config and fills it from a checkpoint.</summary>
    /// <param name="config">The model's configuration.</param>
    /// <param name="weights">The checkpoint.</param>
    /// <param name="strict">Whether a missing parameter throws rather than being reported.</param>
    /// <returns>The filled model and a report of what was loaded.</returns>
    public static (Encoder Model, LoadReport Report) Load(
        PretrainedConfig config, WeightStore weights, bool strict = true)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);

        var model = new Encoder(new TransformerConfig(
            VocabularySize: config.VocabularySize,
            HiddenSize: config.HiddenSize,
            Layers: config.Layers,
            Heads: config.Heads,
            IntermediateSize: config.IntermediateSize,
            MaxPositions: config.MaxPositions));

        var names = Naming.Detect(weights, config);
        var loaded = new List<string>();
        var missing = new List<string>();

        // ---------------------------------------------------------- embeddings
        if (weights.TryRead(names.WordEmbeddings, out var wordEmbeddings))
        {
            var folded = FoldTokenType(weights, names, wordEmbeddings, config.HiddenSize);
            model.ReplaceTokenEmbeddings(folded);
            loaded.Add(names.WordEmbeddings);
        }
        else missing.Add(names.WordEmbeddings);

        if (weights.TryRead(names.PositionEmbeddings, out var positions))
        {
            model.ReplacePositionEmbeddings(TrimPositions(positions, config));
            loaded.Add(names.PositionEmbeddings);
        }
        else missing.Add(names.PositionEmbeddings);

        LoadNorm(weights, model.EmbeddingNorm, names.EmbeddingNorm, loaded, missing);

        // ---------------------------------------------------------- encoder blocks
        for (var layer = 0; layer < config.Layers; layer++)
        {
            var block = model.Layers[layer];

            LoadDense(weights, block.Attention.Query, names.Query(layer), names.Transposed, loaded, missing);
            LoadDense(weights, block.Attention.Key, names.Key(layer), names.Transposed, loaded, missing);
            LoadDense(weights, block.Attention.Value, names.Value(layer), names.Transposed, loaded, missing);
            LoadDense(weights, block.Attention.Output, names.AttentionOut(layer), names.Transposed, loaded, missing);

            LoadNorm(weights, block.AttentionNorm, names.AttentionNorm(layer), loaded, missing);

            LoadDense(weights, block.Intermediate, names.Intermediate(layer), names.Transposed, loaded, missing);
            LoadDense(weights, block.OutputProjection, names.Output(layer), names.Transposed, loaded, missing);

            LoadNorm(weights, block.OutputNorm, names.OutputNorm(layer), loaded, missing);
        }

        var report = new LoadReport(loaded, missing, names.Prefix);

        if (strict && !report.IsComplete)
        {
            throw new InvalidDataException(
                $"The checkpoint is missing {missing.Count} parameters the encoder needs: "
                + string.Join(", ", missing.Take(8)) + (missing.Count > 8 ? ", ..." : "")
                + $". Detected prefix '{names.Prefix}'. Available names begin: "
                + string.Join(", ", weights.Names.Take(5)) + ".");
        }

        if (report.IsComplete) model.MarkPretrained();
        return (model, report);
    }

    /// <summary>
    /// Adds the segment-0 embedding into every word embedding row.
    /// </summary>
    /// <remarks>
    /// Exact for single-sequence input, and the alternative - dropping the term - shifts every
    /// hidden state by a constant vector that the first layer norm does not remove, because it is
    /// added before the norm rather than after.
    /// </remarks>
    private static NdArray FoldTokenType(WeightStore weights, Naming names, NdArray wordEmbeddings, int hiddenSize)
    {
        if (!weights.TryRead(names.TokenTypeEmbeddings, out var tokenTypes)) return wordEmbeddings;
        if (tokenTypes.Shape[1] != hiddenSize) return wordEmbeddings;

        var result = wordEmbeddings.Copy();
        var rows = result.Shape[0];

        for (var row = 0; row < rows; row++)
        {
            for (var d = 0; d < hiddenSize; d++)
            {
                result[row, d] += tokenTypes[0, d];
            }
        }

        return result;
    }

    /// <summary>
    /// Drops RoBERTa's two reserved position slots so index 0 is the first real position.
    /// </summary>
    private static NdArray TrimPositions(NdArray positions, PretrainedConfig config)
    {
        if (config.PositionOffset == 0) return positions;

        var kept = positions.Shape[0] - config.PositionOffset;
        var result = NdArray.Zeros(kept, positions.Shape[1]);

        for (var i = 0; i < kept; i++)
        {
            for (var d = 0; d < positions.Shape[1]; d++)
            {
                result[i, d] = positions[i + config.PositionOffset, d];
            }
        }

        return result;
    }

    private static void LoadDense(
        WeightStore weights, DenseLayer target, string prefix, bool transposed,
        List<string> loaded, List<string> missing)
    {
        var weightName = prefix + ".weight";
        var biasName = prefix + ".bias";

        if (weights.TryRead(weightName, out var matrix))
        {
            CopyMatrix(matrix, target.Weights, transposed, weightName);
            loaded.Add(weightName);
        }
        else missing.Add(weightName);

        if (weights.TryRead(biasName, out var bias))
        {
            if (bias.Size != target.Bias.Size)
            {
                throw new InvalidDataException(
                    $"'{biasName}' has {bias.Size} values but the layer needs {target.Bias.Size}.");
            }

            for (var i = 0; i < bias.Size; i++) target.Bias.SetAt(i, bias.At(i));
            loaded.Add(biasName);
        }
        else missing.Add(biasName);
    }

    /// <summary>Loads a layer norm, accepting either spelling of its two parameters.</summary>
    /// <remarks>
    /// The original BERT checkpoints came from TensorFlow and name these <c>gamma</c> and
    /// <c>beta</c>; everything exported by a modern PyTorch names them <c>weight</c> and
    /// <c>bias</c>. <c>bert-base-uncased</c> itself is still published with the old spelling, so a
    /// loader that knows only the new one fails on the single most-downloaded model on the Hub.
    /// </remarks>
    private static void LoadNorm(
        WeightStore weights, LayerNorm target, string prefix,
        List<string> loaded, List<string> missing)
    {
        var scaleName = weights.Contains($"{prefix}.weight") ? $"{prefix}.weight" : $"{prefix}.gamma";
        var shiftName = weights.Contains($"{prefix}.bias") ? $"{prefix}.bias" : $"{prefix}.beta";

        foreach (var (name, destination) in ((string, NdArray)[])[(scaleName, target.Gamma), (shiftName, target.Beta)])
        {
            if (!weights.TryRead(name, out var tensor))
            {
                missing.Add(name);
                continue;
            }

            if (tensor.Size != destination.Size)
            {
                throw new InvalidDataException(
                    $"'{name}' has {tensor.Size} values but the norm needs {destination.Size}.");
            }

            for (var i = 0; i < destination.Size; i++) destination.SetAt(i, tensor.At(i));
            loaded.Add(name);
        }
    }

    /// <summary>Copies a checkpoint matrix into a layer, transposing if the checkpoint is stored that way.</summary>
    internal static void CopyMatrix(NdArray source, NdArray destination, bool transposed, string name)
    {
        var rows = destination.Shape[0];
        var columns = destination.Shape[1];

        var sourceRows = transposed ? source.Shape[1] : source.Shape[0];
        var sourceColumns = transposed ? source.Shape[0] : source.Shape[1];

        if (sourceRows != rows || sourceColumns != columns)
        {
            throw new InvalidDataException(
                $"'{name}' is [{string.Join("x", source.Shape.ToArray())}]"
                + $"{(transposed ? " (read transposed)" : "")} but the layer is [{rows}x{columns}].");
        }

        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < columns; j++)
            {
                destination[i, j] = transposed ? source[j, i] : source[i, j];
            }
        }
    }
}

/// <summary>
/// The parameter-name convention of one checkpoint family, resolved by looking at the file.
/// </summary>
/// <remarks>
/// Detection probes for a tensor that must exist rather than trusting <c>model_type</c> in the
/// config: fine-tuned checkpoints are routinely re-exported with a different prefix from the one
/// their architecture implies, and a wrong guess produces "missing parameter" for every tensor.
/// </remarks>
internal sealed record Naming
{
    internal required string Prefix { get; init; }

    /// <summary>Whether weight matrices are stored as (outputs, inputs).</summary>
    internal required bool Transposed { get; init; }

    /// <summary>True for the DistilBERT block layout, which names its parts differently.</summary>
    internal required bool IsDistil { get; init; }

    internal string WordEmbeddings => $"{Prefix}embeddings.word_embeddings.weight";
    internal string PositionEmbeddings => $"{Prefix}embeddings.position_embeddings.weight";
    internal string TokenTypeEmbeddings => $"{Prefix}embeddings.token_type_embeddings.weight";
    internal string EmbeddingNorm => $"{Prefix}embeddings.LayerNorm";

    private string Block(int layer) => IsDistil
        ? $"{Prefix}transformer.layer.{layer}"
        : $"{Prefix}encoder.layer.{layer}";

    internal string Query(int layer) => IsDistil ? $"{Block(layer)}.attention.q_lin" : $"{Block(layer)}.attention.self.query";
    internal string Key(int layer) => IsDistil ? $"{Block(layer)}.attention.k_lin" : $"{Block(layer)}.attention.self.key";
    internal string Value(int layer) => IsDistil ? $"{Block(layer)}.attention.v_lin" : $"{Block(layer)}.attention.self.value";
    internal string AttentionOut(int layer) => IsDistil ? $"{Block(layer)}.attention.out_lin" : $"{Block(layer)}.attention.output.dense";
    internal string AttentionNorm(int layer) => IsDistil ? $"{Block(layer)}.sa_layer_norm" : $"{Block(layer)}.attention.output.LayerNorm";
    internal string Intermediate(int layer) => IsDistil ? $"{Block(layer)}.ffn.lin1" : $"{Block(layer)}.intermediate.dense";
    internal string Output(int layer) => IsDistil ? $"{Block(layer)}.ffn.lin2" : $"{Block(layer)}.output.dense";
    internal string OutputNorm(int layer) => IsDistil ? $"{Block(layer)}.output_layer_norm" : $"{Block(layer)}.output.LayerNorm";

    /// <summary>Works out which convention a checkpoint follows.</summary>
    internal static Naming Detect(WeightStore weights, PretrainedConfig config)
    {
        string[] candidates =
        [
            "",
            config.ModelType + ".",
            "bert.",
            "roberta.",
            "distilbert.",
            "electra.",
            "model.",
            "encoder.",
        ];

        foreach (var prefix in candidates)
        {
            if (!weights.Contains($"{prefix}embeddings.word_embeddings.weight")) continue;

            var isDistil = weights.Contains($"{prefix}transformer.layer.0.attention.q_lin.weight");
            var naming = new Naming { Prefix = prefix, Transposed = true, IsDistil = isDistil };

            return naming with { Transposed = DetectTranspose(weights, naming, config) };
        }

        // Nothing matched. Returning the config's own guess gives the caller a load report naming
        // the parameters it looked for, which is more useful than a bare "unsupported".
        return new Naming { Prefix = config.ModelType + ".", Transposed = true, IsDistil = config.ModelType == "distilbert" };
    }

    /// <summary>
    /// Decides the orientation using the feed-forward weight, the one matrix whose dimensions differ.
    /// </summary>
    private static bool DetectTranspose(WeightStore weights, Naming naming, PretrainedConfig config)
    {
        if (!weights.TryRead(naming.Intermediate(0) + ".weight", out var matrix)) return true;
        if (matrix.Rank != 2) return true;

        // The layer is (hidden, intermediate). Stored as (intermediate, hidden) means transposed,
        // which is what PyTorch's nn.Linear does and therefore what almost every checkpoint holds.
        if (matrix.Shape[0] == config.IntermediateSize && matrix.Shape[1] == config.HiddenSize) return true;
        if (matrix.Shape[0] == config.HiddenSize && matrix.Shape[1] == config.IntermediateSize) return false;

        return true;
    }
}
