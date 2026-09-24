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
/// Three things. It <b>applies</b> adapters - including adapters trained in Python and published on
/// the Hub - and merges them exactly into the frozen weights, which is the operation needed to
/// serve a LoRA-tuned model. It <b>trains</b> adapters and a classification head together with
/// <see cref="Train"/>, backpropagating through the pretrained encoder with its weights frozen. And
/// it fits a head alone over frozen features with <see cref="FitHead"/>, which is the cheap
/// baseline worth running first.
/// </para>
/// <para>
/// Training runs on its own backward pass rather than on the foundation's autodiff encoder, which
/// has no biases on its query, key and value projections - a gradient through it would be the
/// gradient of a slightly different model. The pass here is checked against numerical
/// differentiation to 1e-6 in the tests.
/// </para>
/// </remarks>
public sealed class PeftModel
{
    private static readonly Regex LayerIndex = new(@"layer[s]?\.(\d+)\.", RegexOptions.Compiled);

    private readonly TransformerModel _model;
    private LogisticRegression? _head;
    private ClassifierHead? _classifier;
    private LoraEncoder? _encoder;
    private double[] _headClasses = [];
    private string[] _headLabels = [];
    private int _maxLength = 512;

    private PeftModel(TransformerModel model, LoraAdapterSet adapters, bool merged)
    {
        _model = model;
        Adapters = adapters;
        IsMerged = merged;
    }

    /// <summary>Whether adapter matrices themselves can be trained here.</summary>
    /// <remarks>True since 0.3: see <see cref="Train"/>.</remarks>
    public const bool SupportsAdapterTraining = true;

    /// <summary>The adapters attached to this model.</summary>
    public LoraAdapterSet Adapters { get; private set; }

    /// <summary>Whether the adapters have been folded into the base weights.</summary>
    public bool IsMerged { get; private set; }

    /// <summary>The underlying pretrained model.</summary>
    public TransformerModel Model => _model;

    /// <summary>The labels the trained head predicts, empty until <see cref="FitHead"/> or <see cref="Train"/> runs.</summary>
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
        var layers = model.Config.Layers;

        // Each adapter is keyed by the checkpoint's own module path - bert.encoder.layer.0.
        // attention.self.query, not layer.0.query - because that is the name PEFT in Python looks
        // for when it loads a saved adapter. Under any other name it loads nothing, says nothing,
        // and runs the base model.
        var modules = ModulePaths(model.Report.Loaded, layers);

        for (var layer = 0; layer < layers; layer++)
        {
            foreach (var target in config.Targets)
            {
                if (Locate($"layer.{layer}.{target}", layers) is not var (_, projection))
                {
                    throw new NotSupportedException(
                        $"The LoRA target '{target}' names no projection this can adapt. Use query, key, "
                        + "value, attention.output.dense, intermediate.dense or output.dense - or q_lin, "
                        + "k_lin, v_lin, out_lin, lin1 and lin2 on DistilBERT.");
                }

                var (inputs, outputs) = ShapeOf(projection, model.Config.HiddenSize, model.Config.IntermediateSize);

                // Seeded per layer and target so the same configuration always initialises the
                // same way - an adapter set that differs run to run cannot be compared.
                var seed = StableSeed(layer, target);
                var key = modules.GetValueOrDefault((layer, projection), $"layer.{layer}.{target}");
                adapters[key] = new LoraAdapter(inputs, outputs, config, seed);
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

    /// <summary>A seed that depends only on its arguments.</summary>
    /// <remarks>
    /// Not <c>HashCode.Combine</c>, which .NET randomises per process: seeding from it gave every
    /// run of the same configuration different adapters, and so a different loss curve.
    /// </remarks>
    internal static int StableSeed(int layer, string target)
    {
        var hash = 2166136261u;   // FNV-1a
        foreach (var c in target)
        {
            hash = (hash ^ c) * 16777619u;
        }

        hash = (hash ^ (uint)layer) * 16777619u;
        return (int)(hash & 0x7FFFFFFF);
    }

    /// <summary>The module path of each adaptable projection, from a checkpoint's tensor names.</summary>
    internal static Dictionary<(int, Projection), string> ModulePaths(IEnumerable<string> tensorNames, int layers)
    {
        var modules = new Dictionary<(int, Projection), string>();
        foreach (var name in tensorNames)
        {
            if (!name.EndsWith(".weight", StringComparison.Ordinal)) continue;

            var module = name[..^".weight".Length];
            if (Locate(module, layers) is { } at) modules.TryAdd(at, module);
        }

        return modules;
    }

    private static (int Inputs, int Outputs) ShapeOf(Projection projection, int hidden, int intermediate)
        => projection switch
        {
            Projection.Intermediate => (hidden, intermediate),
            Projection.Output => (intermediate, hidden),
            _ => (hidden, hidden),
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

        // Inference runs on a compiled copy of the encoder's weights. Without this the merge would
        // change the model of record and leave every prediction exactly as it was.
        if (applied > 0) _model.WeightsChanged();

        // The training encoder holds a copy of the unmerged weights with the adapters beside them.
        _encoder = null;
        IsMerged = true;
        return this;
    }

    /// <summary>
    /// Finds the dense layer an adapter path refers to.
    /// </summary>
    private DenseLayer? Resolve(string path)
    {
        if (Locate(path, _model.Encoder.Layers.Count) is not var (index, projection)) return null;

        var block = _model.Encoder.Layers[index];
        return projection switch
        {
            Projection.Query => block.Attention.Query,
            Projection.Key => block.Attention.Key,
            Projection.Value => block.Attention.Value,
            Projection.AttentionOutput => block.Attention.Output,
            Projection.Intermediate => block.Intermediate,
            _ => block.OutputProjection,
        };
    }

    /// <summary>
    /// The layer and projection an adapter path refers to, or <c>null</c> when it names neither.
    /// </summary>
    /// <remarks>
    /// Matched on the layer index plus a suffix rather than on the full path, because the same
    /// adapter is published with several prefixes - <c>base_model.model.bert.encoder.layer.0...</c>
    /// from PEFT, and a bare <c>layer.0...</c> from this library's own <c>ApplyLoRA</c>.
    /// </remarks>
    private static (int Layer, Projection Projection)? Locate(string path, int layers)
    {
        var match = LayerIndex.Match(path);
        if (!match.Success) return null;

        var index = int.Parse(match.Groups[1].Value);
        if (index < 0 || index >= layers) return null;

        var lower = path.ToLowerInvariant();

        // The attention output projection has to be tested before the block output projection:
        // "attention.output.dense" ends with "output.dense" and would otherwise match it.
        if (lower.Contains("attention.output.dense") || lower.EndsWith("out_lin", StringComparison.Ordinal))
        {
            return (index, Projection.AttentionOutput);
        }

        if (lower.EndsWith("query", StringComparison.Ordinal) || lower.EndsWith("q_lin", StringComparison.Ordinal))
        {
            return (index, Projection.Query);
        }

        if (lower.EndsWith("key", StringComparison.Ordinal) || lower.EndsWith("k_lin", StringComparison.Ordinal))
        {
            return (index, Projection.Key);
        }

        if (lower.EndsWith("value", StringComparison.Ordinal) || lower.EndsWith("v_lin", StringComparison.Ordinal))
        {
            return (index, Projection.Value);
        }

        if (lower.Contains("intermediate.dense") || lower.EndsWith("lin1", StringComparison.Ordinal))
        {
            return (index, Projection.Intermediate);
        }

        if (lower.Contains("output.dense") || lower.EndsWith("lin2", StringComparison.Ordinal))
        {
            return (index, Projection.Output);
        }

        return null;
    }

    /// <summary>
    /// The encoder with this model's adapters in the loop, built on first use.
    /// </summary>
    /// <exception cref="InvalidOperationException">An adapter is shaped differently from its layer.</exception>
    internal LoraEncoder TrainingEncoder()
    {
        if (_encoder is not null) return _encoder;

        var placed = new Dictionary<(int, Projection), LoraAdapter>();

        foreach (var (path, adapter) in Adapters.Adapters)
        {
            if (Locate(path, _model.Encoder.Layers.Count) is not var (layer, projection)) continue;

            var (inputs, outputs) = ShapeOf(projection, _model.Config.HiddenSize, _model.Config.IntermediateSize);

            if (adapter.Inputs != inputs || adapter.Outputs != outputs)
            {
                throw new InvalidOperationException(
                    $"The adapter '{path}' is {adapter.Outputs}x{adapter.Inputs} but that projection is "
                    + $"{outputs}x{inputs}. It was trained against a different model.");
            }

            placed[(layer, projection)] = adapter;
        }

        _encoder = LoraEncoder.Build(
            _model.Encoder, _model.Config, (layer, projection) => placed.GetValueOrDefault((layer, projection)));

        return _encoder;
    }

    /// <summary>Whether any adapter would change the base model's output if applied.</summary>
    private bool AdaptersChangeOutput()
        => !IsMerged && Adapters.Adapters.Values.Any(a => a.B.ToArray().Any(v => v != 0));

    private int[] Tokenize(string text, int maxLength)
    {
        var ids = _model.Tokenizer.Encode(text).ToIdArray();
        return ids.Length > maxLength ? ids[..maxLength] : ids;
    }

    /// <summary>
    /// A text's mean-pooled features through the adapted encoder: the base model's own inference
    /// path when the adapters are merged or still zero, the adapter-in-the-loop path otherwise.
    /// </summary>
    private double[] Features(string text, int maxLength)
    {
        if (!AdaptersChangeOutput()) return _model.Embed(text, maxLength).ToArray();

        var ids = Tokenize(text, maxLength);
        var encoder = TrainingEncoder();
        return ClassifierHead.Pool(encoder.Forward(ids), ids.Length, encoder.Hidden);
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

        NdArray features;
        if (AdaptersChangeOutput())
        {
            features = NdArray.Zeros(texts.Count, _model.Config.HiddenSize);
            for (var i = 0; i < texts.Count; i++)
            {
                var row = Features(texts[i], _maxLength);
                for (var d = 0; d < row.Length; d++) features[i, d] = row[d];
            }
        }
        else
        {
            features = _model.EmbedBatch(texts, _maxLength);
        }

        var targets = new NdArray([.. labels.Select(l => indexOf[l])], labels.Count);

        _head = new LogisticRegression(learningRate, iterations, l2Penalty);
        _head.Fit(features, targets);
        _headClasses = [.. _head.Classes];
        _classifier = null;

        return this;
    }

    /// <summary>
    /// Trains the adapters and a classification head together, with the base weights frozen.
    /// </summary>
    /// <param name="texts">The training texts.</param>
    /// <param name="labels">One label per text.</param>
    /// <param name="options">Epochs, batch size, learning rate and schedule.</param>
    /// <returns>The loss curve and how long it took.</returns>
    /// <exception cref="InvalidOperationException">
    /// The adapters are already merged, or none of them attaches to this model.
    /// </exception>
    /// <remarks>
    /// <para>
    /// Each step backpropagates a mean cross-entropy through a mean-pooled linear head and the
    /// whole encoder into every adapter's A and B, then takes an AdamW step on the adapters and the
    /// head. The adapters are updated in place, so <see cref="SaveAdapter"/> and
    /// <see cref="Merge"/> afterwards see the trained values.
    /// </para>
    /// <para>
    /// Sequences are processed one at a time, unpadded, so no position is ever masked; the batch is
    /// a unit of averaging, not of vectorisation. On a CPU a step costs roughly three forward
    /// passes per example. For more than a few thousand examples, train with PEFT in Python and
    /// load the adapter here with <see cref="FromPretrained"/>.
    /// </para>
    /// </remarks>
    public TrainingReport Train(
        IReadOnlyList<string> texts, IReadOnlyList<string> labels, TrainingOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(texts);
        ArgumentNullException.ThrowIfNull(labels);
        options ??= new TrainingOptions();

        if (texts.Count != labels.Count)
        {
            throw new ArgumentException($"{texts.Count} texts against {labels.Count} labels.", nameof(labels));
        }

        if (texts.Count == 0) throw new ArgumentException("Nothing to train on.", nameof(texts));

        if (options.Epochs < 1 || options.BatchSize < 1 || options.GradientAccumulation < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options), "Epochs, BatchSize and GradientAccumulation must each be at least 1.");
        }

        if (IsMerged)
        {
            throw new InvalidOperationException(
                "The adapters are already merged into the base weights, so training them now would "
                + "count their update twice. Load the base model again and attach the adapter unmerged "
                + "(PeftModel.WithAdapters(model, adapters, merge: false)) to keep training it.");
        }

        var encoder = TrainingEncoder();
        if (!encoder.Adapters.Any())
        {
            throw new InvalidOperationException(
                $"None of the {Adapters.Adapters.Count} adapters attaches to a projection of this model, "
                + "so there is nothing to train. Target query, key, value, attention.output.dense, "
                + "intermediate.dense or output.dense.");
        }

        _maxLength = options.MaxLength;
        _headLabels = [.. labels.Distinct().Order(StringComparer.Ordinal)];
        var indexOf = _headLabels.Select((label, i) => (label, i))
            .ToDictionary(p => p.label, p => p.i, StringComparer.Ordinal);

        var ids = texts.Select(t => Tokenize(t, options.MaxLength)).ToArray();
        var targets = labels.Select(l => indexOf[l]).ToArray();

        var head = new ClassifierHead(encoder.Hidden, _headLabels.Length, options.Seed);
        var report = LoraTrainer.Fit(encoder, head, ids, targets, options);

        _classifier = head;
        _head = null;

        return report;
    }

    /// <summary>Classifies a text with the trained head.</summary>
    /// <param name="text">The input.</param>
    /// <exception cref="InvalidOperationException"><see cref="FitHead"/> has not been called.</exception>
    public IReadOnlyList<Prediction> Predict(string text)
    {
        if (_head is null && _classifier is null)
        {
            throw new InvalidOperationException(
                "No head has been trained. Call Train or FitHead first, or use Model.Predict to use "
                + "the checkpoint's own classification head if it has one.");
        }

        var features = Features(text, _maxLength);

        if (_classifier is not null)
        {
            var scores = ClassifierHead.Softmax(_classifier.Logits(features));
            return [.. scores
                .Select((score, k) => new Prediction(_headLabels[k], score, k))
                .OrderByDescending(p => p.Score)];
        }

        var row = NdArray.Zeros(1, features.Length);
        for (var d = 0; d < features.Length; d++) row[0, d] = features[d];

        var probabilities = _head!.PredictProbabilities(row);

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
        if (_head is null && _classifier is null) throw new InvalidOperationException("No head has been trained.");

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
