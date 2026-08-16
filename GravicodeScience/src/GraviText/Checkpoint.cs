using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Io;

namespace Gravicode.Science.GraviText.Transformers;

/// <summary>
/// Where each parameter lives in a checkpoint, as a set of name templates.
/// </summary>
/// <remarks>
/// <para>
/// Tensor names are a convention, not a standard. The same BERT exported by two toolchains lands
/// under two different sets of names, and there is no way to infer the mapping from the file — a
/// tensor of shape [768, 768] could be a query projection, a key projection or an output
/// projection, and nothing but the name says which. So the mapping is data, and this is it.
/// </para>
/// <para>
/// Layer templates take <c>{0}</c> for the layer index. Use <see cref="TransformerCheckpoint.Inspect"/>
/// on an unfamiliar file to see what it actually contains, then adjust.
/// </para>
/// </remarks>
public sealed record CheckpointNames
{
    /// <summary>Token embedding table.</summary>
    public string TokenEmbeddings { get; init; } = "bert.embeddings.word_embeddings.weight";

    /// <summary>Position embedding table.</summary>
    public string PositionEmbeddings { get; init; } = "bert.embeddings.position_embeddings.weight";

    /// <summary>Scale of the layer norm applied to the summed embeddings.</summary>
    public string EmbeddingNormScale { get; init; } = "bert.embeddings.LayerNorm.weight";

    /// <summary>Shift of the layer norm applied to the summed embeddings.</summary>
    public string EmbeddingNormShift { get; init; } = "bert.embeddings.LayerNorm.bias";

    /// <summary>Query projection weight, per layer.</summary>
    public string QueryWeight { get; init; } = "bert.encoder.layer.{0}.attention.self.query.weight";

    /// <summary>Query projection bias, per layer.</summary>
    public string QueryBias { get; init; } = "bert.encoder.layer.{0}.attention.self.query.bias";

    /// <summary>Key projection weight, per layer.</summary>
    public string KeyWeight { get; init; } = "bert.encoder.layer.{0}.attention.self.key.weight";

    /// <summary>Key projection bias, per layer.</summary>
    public string KeyBias { get; init; } = "bert.encoder.layer.{0}.attention.self.key.bias";

    /// <summary>Value projection weight, per layer.</summary>
    public string ValueWeight { get; init; } = "bert.encoder.layer.{0}.attention.self.value.weight";

    /// <summary>Value projection bias, per layer.</summary>
    public string ValueBias { get; init; } = "bert.encoder.layer.{0}.attention.self.value.bias";

    /// <summary>Attention output projection weight, per layer.</summary>
    public string AttentionOutputWeight { get; init; } = "bert.encoder.layer.{0}.attention.output.dense.weight";

    /// <summary>Attention output projection bias, per layer.</summary>
    public string AttentionOutputBias { get; init; } = "bert.encoder.layer.{0}.attention.output.dense.bias";

    /// <summary>Scale of the layer norm after attention, per layer.</summary>
    public string AttentionNormScale { get; init; } = "bert.encoder.layer.{0}.attention.output.LayerNorm.weight";

    /// <summary>Shift of the layer norm after attention, per layer.</summary>
    public string AttentionNormShift { get; init; } = "bert.encoder.layer.{0}.attention.output.LayerNorm.bias";

    /// <summary>Feed-forward expansion weight, per layer.</summary>
    public string IntermediateWeight { get; init; } = "bert.encoder.layer.{0}.intermediate.dense.weight";

    /// <summary>Feed-forward expansion bias, per layer.</summary>
    public string IntermediateBias { get; init; } = "bert.encoder.layer.{0}.intermediate.dense.bias";

    /// <summary>Feed-forward contraction weight, per layer.</summary>
    public string OutputWeight { get; init; } = "bert.encoder.layer.{0}.output.dense.weight";

    /// <summary>Feed-forward contraction bias, per layer.</summary>
    public string OutputBias { get; init; } = "bert.encoder.layer.{0}.output.dense.bias";

    /// <summary>Scale of the layer norm after the feed-forward network, per layer.</summary>
    public string OutputNormScale { get; init; } = "bert.encoder.layer.{0}.output.LayerNorm.weight";

    /// <summary>Shift of the layer norm after the feed-forward network, per layer.</summary>
    public string OutputNormShift { get; init; } = "bert.encoder.layer.{0}.output.LayerNorm.bias";

    /// <summary>
    /// Whether linear weights are stored as (outputs, inputs) and need transposing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>This is the setting that silently ruins a load.</b> PyTorch's <c>nn.Linear</c> stores its
    /// weight as (out_features, in_features) and computes <c>x Wᵀ</c>;
    /// <see cref="DenseLayer"/> stores (inputs, outputs) and computes <c>x W</c>. So a PyTorch
    /// checkpoint has to be transposed on the way in.
    /// </para>
    /// <para>
    /// It cannot be detected from the shape where it matters most. A BERT query projection is
    /// 768×768 — square, so both readings are consistent, and a wrong transpose loads without
    /// complaint and produces confident nonsense. The feed-forward weights are 768×3072 and would
    /// give it away, which is why <see cref="TransformerCheckpoint.Load"/> checks those and reports
    /// a mismatch rather than transposing on a guess.
    /// </para>
    /// </remarks>
    public bool Transposed { get; init; } = true;

    /// <summary>The HuggingFace BERT convention, which is the default.</summary>
    public static CheckpointNames HuggingFaceBert => new();

    /// <summary>The same layout without the <c>bert.</c> prefix, as some exports produce.</summary>
    public static CheckpointNames Unprefixed => Reprefixed("");

    /// <summary>The HuggingFace convention with a different top-level prefix.</summary>
    /// <remarks>
    /// A RoBERTa export uses <c>roberta.</c> and a DistilBERT one <c>distilbert.</c>; the rest of
    /// the path is usually identical, so replacing the prefix is often the whole adaptation needed.
    /// </remarks>
    public static CheckpointNames Reprefixed(string prefix)
    {
        var template = new CheckpointNames();
        string Swap(string name) => prefix + name["bert.".Length..];

        return template with
        {
            TokenEmbeddings = Swap(template.TokenEmbeddings),
            PositionEmbeddings = Swap(template.PositionEmbeddings),
            EmbeddingNormScale = Swap(template.EmbeddingNormScale),
            EmbeddingNormShift = Swap(template.EmbeddingNormShift),
            QueryWeight = Swap(template.QueryWeight),
            QueryBias = Swap(template.QueryBias),
            KeyWeight = Swap(template.KeyWeight),
            KeyBias = Swap(template.KeyBias),
            ValueWeight = Swap(template.ValueWeight),
            ValueBias = Swap(template.ValueBias),
            AttentionOutputWeight = Swap(template.AttentionOutputWeight),
            AttentionOutputBias = Swap(template.AttentionOutputBias),
            AttentionNormScale = Swap(template.AttentionNormScale),
            AttentionNormShift = Swap(template.AttentionNormShift),
            IntermediateWeight = Swap(template.IntermediateWeight),
            IntermediateBias = Swap(template.IntermediateBias),
            OutputWeight = Swap(template.OutputWeight),
            OutputBias = Swap(template.OutputBias),
            OutputNormScale = Swap(template.OutputNormScale),
            OutputNormShift = Swap(template.OutputNormShift),
        };
    }
}

/// <summary>What a load actually did, so the result can be checked rather than assumed.</summary>
/// <param name="Loaded">Parameter names that were found and applied.</param>
/// <param name="Missing">Parameters the model has and the checkpoint did not.</param>
/// <param name="Unused">Tensors in the checkpoint that nothing consumed.</param>
public sealed record CheckpointReport(
    IReadOnlyList<string> Loaded,
    IReadOnlyList<string> Missing,
    IReadOnlyList<string> Unused)
{
    /// <summary>True when every parameter the model has was filled from the file.</summary>
    public bool IsComplete => Missing.Count == 0;

    /// <inheritdoc />
    public override string ToString()
        => $"loaded {Loaded.Count}, missing {Missing.Count}, unused {Unused.Count}";
}

/// <summary>
/// Fills a <see cref="TransformerModel"/> from an ONNX checkpoint.
/// </summary>
/// <remarks>
/// <para>
/// The oldest gap in this library: the encoder is exact given the right numbers, and the numbers
/// were random. <c>LoadOnnxWeights</c> filled the embedding tables, which is enough to look up a
/// word vector and not enough to run the model — every attention and feed-forward weight stayed
/// randomly initialised, so the output of a "pretrained" model was still noise shaped like a
/// sentence.
/// </para>
/// <para>
/// This fills all of it: embeddings, embedding norm, and per layer the four attention projections,
/// both feed-forward layers and both layer norms. What it does not do is guess. Every parameter is
/// looked up by name, shapes are checked rather than trusted, and the result is a
/// <see cref="CheckpointReport"/> saying exactly what was filled — because a partial load that
/// reports success is worse than a failure.
/// </para>
/// </remarks>
public static class TransformerCheckpoint
{
    /// <summary>Lists the tensors in a checkpoint, with their shapes.</summary>
    /// <remarks>
    /// The first thing to run on an unfamiliar file. Names are a convention and the file is the
    /// only authority on which one it follows.
    /// </remarks>
    public static IReadOnlyList<(string Name, int[] Shape)> Inspect(string path)
        => [.. OnnxReader.ReadWeights(path).Select(t => (t.Name, t.Shape)).OrderBy(t => t.Name, StringComparer.Ordinal)];

    /// <summary>
    /// Loads every parameter it can find, and reports what it could not.
    /// </summary>
    /// <param name="model">The model to fill. Its configuration decides what is looked for.</param>
    /// <param name="path">An ONNX file holding the weights as initializers.</param>
    /// <param name="names">The naming convention. Defaults to HuggingFace BERT.</param>
    /// <param name="strict">
    /// When true, a missing parameter throws. When false, it is recorded in the report and the
    /// model keeps its random initialisation for that tensor.
    /// </param>
    /// <remarks>
    /// <b>Prefer <c>strict: true</c>.</b> A model with three of twelve layers loaded runs happily
    /// and produces output that is neither the pretrained model's nor a random one's, and nothing
    /// downstream can tell. The lenient mode exists for inspecting a partial export deliberately.
    /// </remarks>
    public static CheckpointReport Load(TransformerModel model, string path,
        CheckpointNames? names = null, bool strict = true)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        names ??= CheckpointNames.HuggingFaceBert;

        var weights = OnnxReader.ReadWeightsByName(path);
        var consumed = new HashSet<string>(StringComparer.Ordinal);
        var loaded = new List<string>();
        var missing = new List<string>();

        var config = model.Config;

        // The transpose convention is checked on a NON-square matrix before anything is written.
        // The attention projections are square and would accept either reading; the feed-forward
        // weights are not, so they are what can actually catch the mistake.
        VerifyOrientation(weights, names, config);

        // ------------------------------------------------------------ embeddings
        if (TryMatrix(weights, names.TokenEmbeddings, out var tokens))
        {
            Require(tokens.Shape[1] == config.HiddenSize, names.TokenEmbeddings,
                $"[{string.Join("x", tokens.Shape)}]", $"width {config.HiddenSize}");

            model.ReplaceTokenEmbeddings(tokens.ToNdArray());
            Consume(names.TokenEmbeddings);
        }
        else Miss(names.TokenEmbeddings);

        if (TryMatrix(weights, names.PositionEmbeddings, out var positions))
        {
            Require(positions.Shape[1] == config.HiddenSize, names.PositionEmbeddings,
                $"[{string.Join("x", positions.Shape)}]", $"width {config.HiddenSize}");

            model.ReplacePositionEmbeddings(positions.ToNdArray());
            Consume(names.PositionEmbeddings);
        }
        else Miss(names.PositionEmbeddings);

        LoadNorm(model.EmbeddingNorm, names.EmbeddingNormScale, names.EmbeddingNormShift);

        // ------------------------------------------------------------ encoder layers
        for (var layer = 0; layer < config.Layers; layer++)
        {
            var block = model.Layers[layer];

            LoadDense(block.Attention.Query, Name(names.QueryWeight, layer), Name(names.QueryBias, layer));
            LoadDense(block.Attention.Key, Name(names.KeyWeight, layer), Name(names.KeyBias, layer));
            LoadDense(block.Attention.Value, Name(names.ValueWeight, layer), Name(names.ValueBias, layer));
            LoadDense(block.Attention.Output,
                Name(names.AttentionOutputWeight, layer), Name(names.AttentionOutputBias, layer));

            LoadNorm(block.AttentionNorm,
                Name(names.AttentionNormScale, layer), Name(names.AttentionNormShift, layer));

            LoadDense(block.Intermediate,
                Name(names.IntermediateWeight, layer), Name(names.IntermediateBias, layer));
            LoadDense(block.OutputProjection,
                Name(names.OutputWeight, layer), Name(names.OutputBias, layer));

            LoadNorm(block.OutputNorm, Name(names.OutputNormScale, layer), Name(names.OutputNormShift, layer));
        }

        var unused = weights.Keys.Where(k => !consumed.Contains(k)).OrderBy(k => k, StringComparer.Ordinal).ToArray();
        var report = new CheckpointReport(loaded, missing, unused);

        if (strict && missing.Count > 0)
            throw new InvalidDataException(
                $"'{path}' is missing {missing.Count} parameter(s) this model needs, starting with "
                + string.Join(", ", missing.Take(5))
                + (missing.Count > 5 ? $" and {missing.Count - 5} more. " : ". ")
                + "Call Inspect to see what the file contains, or pass strict: false to load what is there.");

        if (report.IsComplete) model.MarkPretrained();
        return report;

        // -------------------------------------------------------------- local helpers

        string Name(string template, int layer) => string.Format(template, layer);

        void Consume(string name) { consumed.Add(name); loaded.Add(name); }

        void Miss(string name) => missing.Add(name);

        void LoadDense(DenseLayer target, string weightName, string biasName)
        {
            if (TryMatrix(weights, weightName, out var w))
            {
                // A transposed checkpoint stores (outputs, inputs); the target is (inputs, outputs).
                var expectedRows = names.Transposed ? target.Weights.Shape[1] : target.Weights.Shape[0];
                var expectedColumns = names.Transposed ? target.Weights.Shape[0] : target.Weights.Shape[1];

                Require(w.Shape[0] == expectedRows && w.Shape[1] == expectedColumns,
                    weightName, $"[{string.Join("x", w.Shape)}]",
                    $"[{expectedRows}x{expectedColumns}]");

                CopyInto(target.Weights, w.ToNdArray(), names.Transposed);
                Consume(weightName);
            }
            else Miss(weightName);

            if (weights.TryGetValue(biasName, out var b))
            {
                Require(b.Values.Length == target.Bias.Size, biasName,
                    $"{b.Values.Length} values", $"{target.Bias.Size}");

                for (var i = 0; i < target.Bias.Size; i++) target.Bias.SetAt(i, b.Values[i]);
                Consume(biasName);
            }
            else Miss(biasName);
        }

        void LoadNorm(LayerNorm target, string scaleName, string shiftName)
        {
            foreach (var (name, destination) in new[] { (scaleName, target.Gamma), (shiftName, target.Beta) })
            {
                if (!weights.TryGetValue(name, out var tensor)) { Miss(name); continue; }

                Require(tensor.Values.Length == destination.Size, name,
                    $"{tensor.Values.Length} values", $"{destination.Size}");

                for (var i = 0; i < destination.Size; i++) destination.SetAt(i, tensor.Values[i]);
                Consume(name);
            }
        }
    }

    /// <summary>
    /// Checks the transpose convention against a matrix whose two dimensions differ.
    /// </summary>
    /// <remarks>
    /// The feed-forward weight is hidden×intermediate, so its orientation is unambiguous. If it
    /// does not match the declared convention, every square attention projection would be loaded
    /// the wrong way round too — silently, because square matrices accept either reading.
    /// </remarks>
    private static void VerifyOrientation(
        IReadOnlyDictionary<string, OnnxTensor> weights, CheckpointNames names, TransformerConfig config)
    {
        if (config.HiddenSize == config.IntermediateSize) return;      // nothing to check against

        var probe = string.Format(names.IntermediateWeight, 0);
        if (!weights.TryGetValue(probe, out var tensor) || tensor.Shape.Length != 2) return;

        var (rows, columns) = (tensor.Shape[0], tensor.Shape[1]);

        // As stored: transposed means (out, in) = (intermediate, hidden).
        var expectedRows = names.Transposed ? config.IntermediateSize : config.HiddenSize;
        var expectedColumns = names.Transposed ? config.HiddenSize : config.IntermediateSize;

        if (rows == expectedRows && columns == expectedColumns) return;

        var suggestion = rows == expectedColumns && columns == expectedRows
            ? $" The shape matches the opposite convention, so try Transposed = {!names.Transposed}."
            : "";

        throw new InvalidDataException(
            $"'{probe}' is [{rows}x{columns}], but Transposed = {names.Transposed} expects "
            + $"[{expectedRows}x{expectedColumns}].{suggestion}");
    }

    private static bool TryMatrix(
        IReadOnlyDictionary<string, OnnxTensor> weights, string name, out OnnxTensor tensor)
        => weights.TryGetValue(name, out tensor!) && tensor.Shape.Length == 2;

    /// <summary>Copies a matrix in, transposing when the checkpoint's convention calls for it.</summary>
    private static void CopyInto(NdArray destination, NdArray source, bool transpose)
    {
        var rows = destination.Shape[0];
        var columns = destination.Shape[1];

        for (var i = 0; i < rows; i++)
            for (var j = 0; j < columns; j++)
                destination[i, j] = transpose ? source[j, i] : source[i, j];
    }

    private static void Require(bool condition, string name, string got, string expected)
    {
        if (condition) return;

        throw new InvalidDataException(
            $"'{name}' is {got}, but this model expects {expected}. "
            + "That usually means the checkpoint is a different architecture, not a different naming.");
    }
}
