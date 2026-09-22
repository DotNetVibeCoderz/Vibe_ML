using System.Text.RegularExpressions;
using Gravicode.HFNet.GraviTransformers;
using Gravicode.Science.GraviLearn.Linear;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Transformers;
using TransformerModel = Gravicode.HFNet.GraviTransformers.TransformerModel;

namespace Gravicode.HFNet.GraviPEFT;

/// <summary>
/// A pretrained encoder with LoRA adapters attached, plus a trainable task head.
/// </summary>
/// <remarks>
/// <para>
/// Two things this does and one it does not. It <b>applies</b> adapters - including adapters trained
/// in Python and published on the Hub - and merges them exactly into the frozen weights, which is
/// the operation needed to serve a LoRA-tuned model. It also trains a task <b>head</b> over the
/// adapted encoder's frozen features, which is the cheap and effective way to fit a classifier on a
/// few thousand labelled examples.
/// </para>
/// <para>
/// What it does not do is backpropagate into the adapter matrices themselves. The encoder
/// underneath computes a forward pass only, and the autodiff encoder available in the foundation
/// omits the attention biases that a pretrained BERT has - so a gradient taken through it would be
/// a gradient of a slightly different model. That is stated here rather than approximated:
/// <see cref="SupportsAdapterTraining"/> is false, and the honest path for now is to train adapters
/// with PEFT in Python and serve them here.
/// </para>
/// </remarks>
public sealed class PeftModel
{
    private static readonly Regex LayerIndex = new(@"layer[s]?\.(\d+)\.", RegexOptions.Compiled);

    private readonly TransformerModel _model;
    private LogisticRegression? _head;
    private double[] _headClasses = [];
    private string[] _headLabels = [];

    private PeftModel(TransformerModel model, LoraAdapterSet adapters, bool merged)
    {
        _model = model;
        Adapters = adapters;
        IsMerged = merged;
    }

    /// <summary>Whether adapter matrices themselves can be trained here.</summary>
    /// <remarks>
    /// False. See the remarks on <see cref="PeftModel"/> for why, and what to do instead.
    /// </remarks>
    public const bool SupportsAdapterTraining = false;

    /// <summary>The adapters attached to this model.</summary>
    public LoraAdapterSet Adapters { get; private set; }

    /// <summary>Whether the adapters have been folded into the base weights.</summary>
    public bool IsMerged { get; private set; }

    /// <summary>The underlying pretrained model.</summary>
    public TransformerModel Model => _model;

    /// <summary>The labels the trained head predicts, empty until <see cref="FitHead"/> runs.</summary>
    public IReadOnlyList<string> HeadLabels => _headLabels;

    // ------------------------------------------------------------------ construction

    /// <summary>
    /// Attaches freshly initialised LoRA adapters to a model.
    /// </summary>
    /// <param name="model">The pretrained model to adapt.</param>
    /// <param name="config">Rank, scaling and which projections to target.</param>
    /// <remarks>
    /// The returned model behaves <i>identically</i> to the input, because every adapter's B matrix
    /// is zero. That is the point: it is a starting position, not a change.
    /// </remarks>
    public static PeftModel ApplyLoRA(TransformerModel model, LoraConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        config ??= new LoraConfig();

        var adapters = new Dictionary<string, LoraAdapter>(StringComparer.Ordinal);
        var hidden = model.Config.HiddenSize;
        var intermediate = model.Config.IntermediateSize;

        for (var layer = 0; layer < model.Config.Layers; layer++)
        {
            foreach (var target in config.Targets)
            {
                var (inputs, outputs) = ShapeOf(target, hidden, intermediate);
                if (inputs == 0) continue;

                // Seeded per layer and target so the same configuration always initialises the
                // same way - an adapter set that differs run to run cannot be compared.
                var seed = HashCode.Combine(layer, target) & 0x7FFFFFFF;
                adapters[$"layer.{layer}.{target}"] = new LoraAdapter(inputs, outputs, config, seed);
            }
        }

        return new PeftModel(model, new LoraAdapterSet(adapters, config), merged: false);
    }

    /// <summary>Loads a PEFT adapter from the Hub and attaches it.</summary>
    /// <param name="model">The base model the adapter was trained against.</param>
    /// <param name="adapterRepoId">The adapter repository id.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    /// <param name="merge">Whether to fold the adapter into the weights immediately.</param>
    public static PeftModel FromPretrained(
        TransformerModel model, string adapterRepoId, string revision = "main", bool merge = true)
    {
        ArgumentNullException.ThrowIfNull(model);

        var adapters = LoraAdapterSet.FromPretrained(adapterRepoId, revision);
        var peft = new PeftModel(model, adapters, merged: false);

        return merge ? peft.Merge() : peft;
    }

    /// <summary>Attaches an already-loaded adapter set.</summary>
    public static PeftModel WithAdapters(TransformerModel model, LoraAdapterSet adapters, bool merge = true)
    {
        var peft = new PeftModel(model, adapters, merged: false);
        return merge ? peft.Merge() : peft;
    }

    private static (int Inputs, int Outputs) ShapeOf(string target, int hidden, int intermediate)
        => target.ToLowerInvariant() switch
        {
            "query" or "q_lin" or "key" or "k_lin" or "value" or "v_lin"
                or "attention.output.dense" or "out_lin" or "dense" => (hidden, hidden),

            "intermediate.dense" or "lin1" => (hidden, intermediate),
            "output.dense" or "lin2" => (intermediate, hidden),

            _ => (0, 0),
        };

    // ------------------------------------------------------------------ merging

    /// <summary>
    /// Folds every adapter into the base weights, in place, and returns this model.
    /// </summary>
    /// <remarks>
    /// After merging, inference costs exactly what the base model costs - there is no adapter left
    /// to evaluate. This is the right thing to do before serving and the wrong thing to do before
    /// swapping adapters, because the fold cannot be undone from the merged weights alone.
    /// </remarks>
    public PeftModel Merge()
    {
        if (IsMerged) return this;

        var applied = 0;

        foreach (var (path, adapter) in Adapters.Adapters)
        {
            var target = Resolve(path);
            if (target is null) continue;

            adapter.MergeInto(target.Weights);
            applied++;
        }

        if (applied == 0 && Adapters.Adapters.Count > 0)
        {
            throw new InvalidOperationException(
                $"None of the {Adapters.Adapters.Count} adapters matched a layer in this model. "
                + $"First adapter path: '{Adapters.Adapters.Keys.First()}'. "
                + "The adapter was probably trained against a different architecture.");
        }

        IsMerged = true;
        return this;
    }

    /// <summary>
    /// Finds the dense layer an adapter path refers to.
    /// </summary>
    /// <remarks>
    /// Matched on the layer index plus a suffix rather than on the full path, because the same
    /// adapter is published with several prefixes - <c>base_model.model.bert.encoder.layer.0...</c>
    /// from PEFT, and a bare <c>layer.0...</c> from this library's own <c>ApplyLoRA</c>.
    /// </remarks>
    private DenseLayer? Resolve(string path)
    {
        var match = LayerIndex.Match(path);
        if (!match.Success) return null;

        var index = int.Parse(match.Groups[1].Value);
        if (index < 0 || index >= _model.Encoder.Layers.Count) return null;

        var block = _model.Encoder.Layers[index];
        var lower = path.ToLowerInvariant();

        // The attention output projection has to be tested before the block output projection:
        // "attention.output.dense" ends with "output.dense" and would otherwise match it.
        if (lower.Contains("attention.output.dense") || lower.EndsWith("out_lin", StringComparison.Ordinal))
        {
            return block.Attention.Output;
        }

        if (lower.EndsWith("query", StringComparison.Ordinal) || lower.EndsWith("q_lin", StringComparison.Ordinal))
        {
            return block.Attention.Query;
        }

        if (lower.EndsWith("key", StringComparison.Ordinal) || lower.EndsWith("k_lin", StringComparison.Ordinal))
        {
            return block.Attention.Key;
        }

        if (lower.EndsWith("value", StringComparison.Ordinal) || lower.EndsWith("v_lin", StringComparison.Ordinal))
        {
            return block.Attention.Value;
        }

        if (lower.Contains("intermediate.dense") || lower.EndsWith("lin1", StringComparison.Ordinal))
        {
            return block.Intermediate;
        }

        if (lower.Contains("output.dense") || lower.EndsWith("lin2", StringComparison.Ordinal))
        {
            return block.OutputProjection;
        }

        return null;
    }

    // ------------------------------------------------------------------ head training

    /// <summary>
    /// Trains a classification head on the adapted encoder's frozen embeddings.
    /// </summary>
    /// <param name="texts">The training texts.</param>
    /// <param name="labels">One label per text.</param>
    /// <param name="learningRate">Gradient descent step size for the head.</param>
    /// <param name="iterations">Maximum head iterations.</param>
    /// <param name="l2Penalty">L2 strength on the head's coefficients.</param>
    /// <returns>This model, so the call can be chained.</returns>
    /// <remarks>
    /// The encoder is evaluated once per example and its output reused, which is what makes this
    /// cheap: the transformer forward pass dominates the cost, and training the head is then a
    /// logistic regression over a few hundred dimensions.
    /// </remarks>
    public PeftModel FitHead(
        IReadOnlyList<string> texts,
        IReadOnlyList<string> labels,
        double learningRate = 0.5,
        int iterations = 600,
        double l2Penalty = 0.01)
    {
        ArgumentNullException.ThrowIfNull(texts);
        ArgumentNullException.ThrowIfNull(labels);

        if (texts.Count != labels.Count)
        {
            throw new ArgumentException(
                $"{texts.Count} texts against {labels.Count} labels.", nameof(labels));
        }

        if (texts.Count == 0) throw new ArgumentException("Nothing to train on.", nameof(texts));

        _headLabels = [.. labels.Distinct().Order(StringComparer.Ordinal)];
        var indexOf = _headLabels.Select((label, i) => (label, i))
            .ToDictionary(p => p.label, p => (double)p.i, StringComparer.Ordinal);

        var features = _model.EmbedBatch(texts);
        var targets = new NdArray([.. labels.Select(l => indexOf[l])], labels.Count);

        _head = new LogisticRegression(learningRate, iterations, l2Penalty);
        _head.Fit(features, targets);
        _headClasses = [.. _head.Classes];

        return this;
    }

    /// <summary>Classifies a text with the trained head.</summary>
    /// <param name="text">The input.</param>
    /// <exception cref="InvalidOperationException"><see cref="FitHead"/> has not been called.</exception>
    public IReadOnlyList<Prediction> Predict(string text)
    {
        if (_head is null)
        {
            throw new InvalidOperationException(
                "No head has been trained. Call FitHead first, or use Model.Predict to use the "
                + "checkpoint's own classification head if it has one.");
        }

        var features = _model.Embed(text);
        var row = NdArray.Zeros(1, features.Size);
        for (var d = 0; d < features.Size; d++) row[0, d] = features.At(d);

        var probabilities = _head.PredictProbabilities(row);

        return [.. Enumerable.Range(0, probabilities.Shape[1])
            .Select(k => new Prediction(
                _headLabels[(int)_headClasses[k]],
                probabilities[0, k],
                (int)_headClasses[k]))
            .OrderByDescending(p => p.Score)];
    }

    /// <summary>Accuracy of the trained head on held-out data.</summary>
    /// <param name="texts">The evaluation texts.</param>
    /// <param name="labels">Their true labels.</param>
    public double Score(IReadOnlyList<string> texts, IReadOnlyList<string> labels)
    {
        if (_head is null) throw new InvalidOperationException("No head has been trained.");

        var correct = 0;
        for (var i = 0; i < texts.Count; i++)
        {
            if (Predict(texts[i])[0].Label == labels[i]) correct++;
        }

        return texts.Count == 0 ? 0 : (double)correct / texts.Count;
    }

    /// <summary>Writes the adapters in the Hugging Face PEFT layout.</summary>
    /// <param name="directory">Where to write.</param>
    public void SaveAdapter(string directory) => Adapters.Save(directory, _model.RepoId);

    /// <summary>
    /// How much smaller the adapter is than the encoder it adapts.
    /// </summary>
    /// <returns>Adapter parameters, encoder parameters, and the ratio between them.</returns>
    public (long Adapter, long Encoder, double Fraction) ParameterEfficiency()
    {
        var encoder = _model.Encoder.Config.ParameterCount;
        var adapter = Adapters.ParameterCount;
        return (adapter, encoder, encoder == 0 ? 0 : (double)adapter / encoder);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var (adapter, encoder, fraction) = ParameterEfficiency();
        return $"PeftModel({_model.RepoId}, {Adapters.Adapters.Count} adapters, "
            + $"{adapter:N0}/{encoder:N0} params = {fraction:P3}"
            + (IsMerged ? ", merged" : "") + ")";
    }
}

/// <summary>The blueprint's entry point: <c>PEFT.ApplyLoRA(model)</c>.</summary>
public static class PEFT
{
    /// <summary>Attaches freshly initialised LoRA adapters to a model.</summary>
    /// <param name="model">The pretrained model to adapt.</param>
    /// <param name="config">Rank, scaling and which projections to target.</param>
    public static PeftModel ApplyLoRA(TransformerModel model, LoraConfig? config = null)
        => PeftModel.ApplyLoRA(model, config);

    /// <summary>Loads a published PEFT adapter onto a base model.</summary>
    /// <param name="model">The base model.</param>
    /// <param name="adapterRepoId">The adapter repository id.</param>
    /// <param name="merge">Whether to fold it into the weights immediately.</param>
    public static PeftModel LoadAdapter(TransformerModel model, string adapterRepoId, bool merge = true)
        => PeftModel.FromPretrained(model, adapterRepoId, "main", merge);

    /// <summary>Reads an adapter file without attaching it to anything.</summary>
    public static LoraAdapterSet ReadAdapter(string path) => LoraAdapterSet.Load(path);
}
