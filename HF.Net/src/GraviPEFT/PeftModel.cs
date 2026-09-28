using System.Text.RegularExpressions;
using Gravicode.HFNet.GraviHub.Io;
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
    private TokenClassifierHead? _tagger;
    private SpanHead? _answerer;
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
        return WithAdapters(model, adapters, merge);
    }

    /// <summary>Loads an adapter saved to a local directory - by <see cref="SaveAdapter"/>, or by PEFT.</summary>
    /// <param name="model">The base model the adapter was trained against.</param>
    /// <param name="directory">Holds <c>adapter_model.safetensors</c> (or <c>.bin</c>) and <c>adapter_config.json</c>.</param>
    /// <param name="merge">Whether to fold the adapter into the weights immediately.</param>
    /// <remarks>
    /// A classifier saved with the adapter comes back with it, so <see cref="Predict"/> works
    /// straight away. That includes one trained in Python with <c>task_type="SEQ_CLS"</c> on a BERT
    /// model; its labels are then <c>LABEL_0</c>, <c>LABEL_1</c> and so on, because PEFT does not
    /// save label names.
    /// </remarks>
    public static PeftModel Load(TransformerModel model, string directory, bool merge = true)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);

        if (!Directory.Exists(directory))
        {
            throw new DirectoryNotFoundException($"'{directory}' does not exist. SaveAdapter writes a directory; pass that.");
        }

        var weights = new[] { "adapter_model.safetensors", "adapter_model.bin" }
            .Select(name => Path.Combine(directory, name))
            .FirstOrDefault(File.Exists)
            ?? throw new FileNotFoundException(
                $"'{directory}' has no adapter_model.safetensors or adapter_model.bin; it is not a PEFT adapter.");

        var configPath = Path.Combine(directory, "adapter_config.json");
        var config = File.Exists(configPath) ? LoraAdapterSet.ReadConfig(configPath) : new LoraConfig();

        return WithAdapters(model, LoraAdapterSet.Load(weights, config), merge);
    }

    /// <summary>Attaches an already-loaded adapter set, and the classifier saved with it if any.</summary>
    public static PeftModel WithAdapters(TransformerModel model, LoraAdapterSet adapters, bool merge = true)
    {
        var peft = new PeftModel(model, adapters, merged: false);
        peft.AttachSavedHead();
        return merge ? peft.Merge() : peft;
    }

    /// <summary>Restores a classifier saved beside the adapters, when there is one.</summary>
    /// <exception cref="NotSupportedException">The head needs a BERT pooler this model does not have.</exception>
    private void AttachSavedHead()
    {
        var directory = Adapters.SourceDirectory;
        var description = directory is null ? null : HeadDescription.Read(directory);
        LinearHead? head = null;

        // The two classifiers have the same shape, so only the saved task can tell them apart:
        // hfnet_head.json when HF.Net wrote the adapter, task_type when PEFT did.
        var isToken = description is not null
            ? description.Task == "token"
            : directory is not null && HeadDescription.TaskType(directory) == "TOKEN_CLS";

        if (Adapters.Others.TryGetValue("qa_outputs.weight", out var spanWeight)
            && Adapters.Others.TryGetValue("qa_outputs.bias", out var spanBias))
        {
            head = SpanHead.FromWeights(spanWeight, spanBias);
        }
        else if (isToken
            && Adapters.Others.TryGetValue("classifier.weight", out var tokenWeight)
            && Adapters.Others.TryGetValue("classifier.bias", out var tokenBias))
        {
            head = TokenClassifierHead.FromWeights(tokenWeight, tokenBias);
        }
        else if (Adapters.Others.TryGetValue("classifier.weight", out var weight)
            && Adapters.Others.TryGetValue("classifier.bias", out var bias))
        {
            // Saved the way PEFT saves SEQ_CLS: BertForSequenceClassification's classifier, which
            // reads the pooler. Python-trained adapters arrive like this with no description.
            var pooler = PoolerOf(_model) ?? throw new NotSupportedException(
                $"The adapter carries a sequence classifier that reads BERT's pooler, and '{_model.RepoId}' "
                + "has no pooler.dense weights. Load it onto the BERT model it was trained against.");

            head = ClassifierHead.FromWeights(weight, bias, pooler);
        }
        else if (description is { Pooling: "mean" } && Adapters.SourceDirectory is { } source
            && File.Exists(Path.Combine(source, LoraAdapterSet.HeadWeightsFile)))
        {
            var tensors = SafeTensors.ReadAll(Path.Combine(source, LoraAdapterSet.HeadWeightsFile));
            head = ClassifierHead.FromWeights(tensors["classifier.weight"], tensors["classifier.bias"], pooler: null);
        }

        if (head is null) return;

        if (head.Weight.Shape[1] != _model.Config.HiddenSize)
        {
            throw new InvalidDataException(
                $"The saved classifier reads {head.Weight.Shape[1]} inputs but '{_model.RepoId}' is "
                + $"{_model.Config.HiddenSize} wide. It was trained against a different model.");
        }

        // A span head has no labels: its two outputs are the start and the end.
        IReadOnlyList<string> labels = head is SpanHead
            ? []
            : description?.Labels is { Count: > 0 } named
                ? named
                : [.. Enumerable.Range(0, head.Classes).Select(i => $"LABEL_{i}")];

        if (head is not SpanHead && labels.Count != head.Classes)
        {
            throw new InvalidDataException(
                $"{LoraAdapterSet.HeadConfigFile} names {labels.Count} labels but the classifier has {head.Classes} outputs.");
        }

        _classifier = head as ClassifierHead;
        _tagger = head as TokenClassifierHead;
        _answerer = head as SpanHead;
        _headLabels = [.. labels];
        _maxLength = description?.MaxLength ?? 512;
    }

    /// <summary>
    /// BERT's pretrained pooler, when <paramref name="model"/> is a BERT checkpoint that has one.
    /// </summary>
    /// <remarks>
    /// Only BERT. RoBERTa checkpoints often carry a pooler too, but
    /// <c>RobertaForSequenceClassification</c> never reads it, and DistilBERT's head adds a
    /// <c>pre_classifier</c> that PEFT does not save - a head built on either would not be the head
    /// Python loads.
    /// </remarks>
    private static Pooler? PoolerOf(TransformerModel model)
    {
        if (!model.Config.ModelType.Equals("bert", StringComparison.OrdinalIgnoreCase)) return null;

        var prefix = model.Report.Prefix;
        if (!model.TryReadTensor($"{prefix}pooler.dense.weight", out var weight)
            || !model.TryReadTensor($"{prefix}pooler.dense.bias", out var bias))
        {
            return null;
        }

        var width = model.Config.HiddenSize;
        if (weight.Size != (long)width * width || bias.Size != width) return null;

        return new Pooler(weight.ToArray(), bias.ToArray());
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
            _model.Encoder, _model.Config, (layer, projection) => placed.GetValueOrDefault((layer, projection)),
            _model.Report.SegmentDelta);

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
    /// A text's final hidden states through the adapted encoder: the base model's own inference
    /// path when the adapters are merged or still zero, the adapter-in-the-loop path otherwise.
    /// </summary>
    private (double[] Hidden, int Rows) Encode(string text, int maxLength)
    {
        if (!AdaptersChangeOutput())
        {
            var hidden = _model.Hidden(text, maxLength);
            return (hidden.ToArray(), hidden.Shape[0]);
        }

        var ids = Tokenize(text, maxLength);
        return (TrainingEncoder().Forward(ids), ids.Length);
    }

    /// <summary>A text's mean-pooled features, what <see cref="FitHead"/> fits on.</summary>
    private double[] Features(string text, int maxLength)
    {
        var (hidden, rows) = Encode(text, maxLength);
        return ClassifierHead.MeanPool(hidden, rows, _model.Config.HiddenSize);
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

        var encoder = PrepareTraining(options);

        _maxLength = options.MaxLength;
        _headLabels = [.. labels.Distinct().Order(StringComparer.Ordinal)];
        var indexOf = _headLabels.Select((label, i) => (label, i))
            .ToDictionary(p => p.label, p => p.i, StringComparer.Ordinal);

        var ids = texts.Select(t => Tokenize(t, options.MaxLength)).ToArray();
        var targets = labels.Select(l => indexOf[l]).ToArray();

        // On BERT the head is BertForSequenceClassification's own, so a saved adapter predicts
        // the same thing in Python; elsewhere it is a linear layer over the mean row.
        var head = new ClassifierHead(encoder.Hidden, _headLabels.Length, options.Seed, PoolerOf(_model));
        var report = LoraTrainer.Fit(encoder, head, ids, targets, options);

        _classifier = head;
        _tagger = null;
        _answerer = null;
        _head = null;

        return report;
    }

    /// <summary>
    /// Trains the adapters and a token classifier together, for named entities.
    /// </summary>
    /// <param name="words">Each sentence as its words - the form CoNLL-style datasets come in.</param>
    /// <param name="tags">One tag per word, such as <c>B-PER</c>, <c>I-PER</c> or <c>O</c>.</param>
    /// <param name="options">Epochs, batch size, learning rate and schedule.</param>
    /// <returns>The loss curve and how long it took.</returns>
    /// <remarks>
    /// <para>
    /// The head is <c>classifier</c> over every token, which is what each family's
    /// <c>ForTokenClassification</c> model has, so a saved adapter loads in Python on any of them.
    /// Only the first piece of each word is trained on, as Transformers' <c>label_all_tokens=False</c>
    /// does. The loss is the mean over those pieces across a step.
    /// </para>
    /// <para>
    /// Training a token head replaces a sequence head trained earlier, and the reverse: both live on
    /// the same adapters, and retraining the adapters for one leaves the other reading features it
    /// was not fitted to.
    /// </para>
    /// </remarks>
    public TrainingReport TrainTokenClassifier(
        IReadOnlyList<IReadOnlyList<string>> words, IReadOnlyList<IReadOnlyList<string>> tags, TrainingOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(words);
        ArgumentNullException.ThrowIfNull(tags);
        options ??= new TrainingOptions();

        if (words.Count != tags.Count)
        {
            throw new ArgumentException($"{words.Count} sentences against {tags.Count} tag sequences.", nameof(tags));
        }

        if (words.Count == 0) throw new ArgumentException("Nothing to train on.", nameof(words));

        for (var i = 0; i < words.Count; i++)
        {
            if (words[i].Count != tags[i].Count)
            {
                throw new ArgumentException(
                    $"Sentence {i} has {words[i].Count} words and {tags[i].Count} tags; tags are one per word.", nameof(tags));
            }
        }

        var encoder = PrepareTraining(options);

        _maxLength = options.MaxLength;
        _headLabels = [.. tags.SelectMany(t => t).Distinct().Order(StringComparer.Ordinal)];
        var indexOf = _headLabels.Select((label, i) => (label, i))
            .ToDictionary(p => p.label, p => p.i, StringComparer.Ordinal);

        var ids = new int[words.Count][];
        var targets = new int[words.Count][];

        for (var i = 0; i < words.Count; i++)
        {
            var (text, spans) = TokenAlignment.Join(words[i]);
            var encoding = _model.Tokenizer.Encode(text);
            var aligned = TokenAlignment.Targets(encoding, spans, [.. tags[i].Select(t => indexOf[t])]);
            var length = Math.Min(encoding.Length, options.MaxLength);

            ids[i] = encoding.ToIdArray()[..length];
            targets[i] = aligned[..length];
        }

        var head = new TokenClassifierHead(encoder.Hidden, _headLabels.Length, options.Seed);
        var report = LoraTrainer.Fit(encoder, head, ids, targets, options);

        _tagger = head;
        _classifier = null;
        _answerer = null;
        _head = null;

        return report;
    }

    /// <summary>
    /// Trains the adapters and an extractive question answering head together.
    /// </summary>
    /// <param name="examples">Questions, the passages that answer them, and the answers.</param>
    /// <param name="options">Epochs, batch size, learning rate and schedule.</param>
    /// <returns>The loss curve and how long it took.</returns>
    /// <remarks>
    /// <para>
    /// The head is <c>qa_outputs</c>, a start and an end score per token, trained with the loss
    /// Transformers uses: the mean of a cross-entropy over start positions and one over end
    /// positions. The question and the passage go in as a pair, so the passage is segment 1.
    /// </para>
    /// <para>
    /// A pair longer than <see cref="TrainingOptions.MaxLength"/> is cut at the end, which cuts the
    /// passage. An answer that falls past the cut is trained as pointing at <c>[CLS]</c>, which is
    /// how Transformers marks "not in this window". It does not split long passages into
    /// overlapping windows; keep passages within the limit.
    /// </para>
    /// </remarks>
    public TrainingReport TrainQuestionAnswering(IReadOnlyList<AnswerExample> examples, TrainingOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(examples);
        options ??= new TrainingOptions();

        if (examples.Count == 0) throw new ArgumentException("Nothing to train on.", nameof(examples));

        var encoder = PrepareTraining(options);

        var ids = new int[examples.Count][];
        var types = new int[examples.Count][];
        var targets = new int[examples.Count][];

        for (var i = 0; i < examples.Count; i++)
        {
            var example = examples[i];
            var start = example.AnswerStart >= 0
                ? example.AnswerStart
                : example.Context.IndexOf(example.Answer, StringComparison.Ordinal);

            if (start < 0 || start + example.Answer.Length > example.Context.Length
                || string.CompareOrdinal(example.Context, start, example.Answer, 0, example.Answer.Length) != 0)
            {
                throw new ArgumentException(
                    $"Example {i}: the answer '{example.Answer}' is not in its context"
                    + (example.AnswerStart >= 0 ? $" at {example.AnswerStart}" : "") + ". Answers are extracted spans.",
                    nameof(examples));
            }

            var encoding = TokenAlignment.Truncate(_model.Tokenizer.Encode(example.Question, example.Context), options.MaxLength);
            var (first, last) = TokenAlignment.AnswerTokens(encoding, start, start + example.Answer.Length, encoding.Length);

            ids[i] = encoding.ToIdArray();
            types[i] = [.. encoding.TypeIds];
            targets[i] = [first, last];
        }

        var head = new SpanHead(encoder.Hidden, options.Seed);
        var report = LoraTrainer.Fit(encoder, head, ids, targets, options, types);

        _maxLength = options.MaxLength;
        _headLabels = [];
        _answerer = head;
        _classifier = null;
        _tagger = null;
        _head = null;

        return report;
    }

    /// <summary>Answers a question from a passage with the head trained here or loaded.</summary>
    /// <param name="question">What to ask.</param>
    /// <param name="context">The passage the answer must come from.</param>
    /// <param name="topK">How many candidate spans to return, best first.</param>
    /// <exception cref="InvalidOperationException">No question answering head has been trained or loaded.</exception>
    /// <remarks>
    /// Decoded as <c>TransformerModel.Answer</c> decodes: only passage positions are eligible, the end
    /// never precedes the start, and the answer is a substring of <paramref name="context"/>.
    /// </remarks>
    public IReadOnlyList<Answer> Answer(string question, string context, int topK = 1)
    {
        ArgumentNullException.ThrowIfNull(question);
        ArgumentNullException.ThrowIfNull(context);

        if (_answerer is null)
        {
            throw new InvalidOperationException(
                "No question answering head has been trained or loaded. Call TrainQuestionAnswering, or "
                + "load an adapter saved with one (task_type QUESTION_ANS).");
        }

        var encoding = TokenAlignment.Truncate(_model.Tokenizer.Encode(question, context), _maxLength);
        double[] hidden;

        if (AdaptersChangeOutput())
        {
            hidden = TrainingEncoder().Forward(encoding.ToIdArray(), typeIds: [.. encoding.TypeIds]);
        }
        else
        {
            hidden = _model.Forward(encoding.ToIdArray(), [.. encoding.TypeIds], encoding.ToMaskArray()).ToArray();
        }

        var (start, end) = _answerer.Logits(hidden, encoding.Length, _model.Config.HiddenSize);
        return QuestionAnsweringHead.Decode(context, encoding, start, end, topK: topK);
    }

    /// <summary>Finds the named entities in a text with the token classifier trained here or loaded.</summary>
    /// <param name="text">The input.</param>
    /// <exception cref="InvalidOperationException">No token classifier has been trained or loaded.</exception>
    /// <remarks>
    /// Each word takes the label predicted for its first piece, the only piece that was trained -
    /// Transformers' <c>aggregation_strategy="first"</c>. Spans are substrings of
    /// <paramref name="text"/>, taken from the tokenizer's offsets.
    /// </remarks>
    public IReadOnlyList<Entity> FindEntities(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        if (_tagger is null)
        {
            throw new InvalidOperationException(
                "No token classifier has been trained or loaded. Call TrainTokenClassifier, or load an "
                + "adapter saved with one (task_type TOKEN_CLS).");
        }

        var encoding = _model.Tokenizer.Encode(text);
        var (hidden, rows) = Encode(text, _maxLength);
        var logits = _tagger.Logits(hidden, rows, _model.Config.HiddenSize);

        return TokenAlignment.Decode(text, encoding, logits, _headLabels);
    }

    /// <summary>The checks both kinds of training share, and the encoder they train through.</summary>
    private LoraEncoder PrepareTraining(TrainingOptions options)
    {
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

        return encoder;
    }

    /// <summary>Classifies a text with the trained head.</summary>
    /// <param name="text">The input.</param>
    /// <exception cref="InvalidOperationException"><see cref="FitHead"/> has not been called.</exception>
    public IReadOnlyList<Prediction> Predict(string text)
    {
        if (_head is null && _classifier is null)
        {
            throw new InvalidOperationException(
                _tagger is not null
                    ? "This model's head labels tokens, not texts. Use FindEntities."
                    : _answerer is not null
                    ? "This model's head extracts answers, not labels. Use Answer."
                    : "No head has been trained. Call Train or FitHead first, or use Model.Predict to use "
                        + "the checkpoint's own classification head if it has one.");
        }

        if (_classifier is not null)
        {
            var (hidden, rows) = Encode(text, _maxLength);
            var scores = ClassifierHead.Softmax(_classifier.Logits(hidden, rows, _model.Config.HiddenSize));
            return [.. scores
                .Select((score, k) => new Prediction(_headLabels[k], score, k))
                .OrderByDescending(p => p.Score)];
        }

        var features = Features(text, _maxLength);
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
    /// <remarks>
    /// A classifier trained with <see cref="Train"/> is saved with the adapters. On BERT it is saved
    /// as PEFT saves a <c>SEQ_CLS</c> head, so the directory loads in Python with
    /// <c>AutoModelForSequenceClassification</c> and <c>PeftModel.from_pretrained</c>; the label
    /// names go into <c>hfnet_head.json</c>. A head fitted with <see cref="FitHead"/> is a logistic
    /// regression and is not saved.
    /// </remarks>
    public void SaveAdapter(string directory)
    {
        LinearHead? trained = (LinearHead?)_classifier ?? (LinearHead?)_tagger ?? _answerer;
        Adapters.Save(directory, _model.RepoId, trained is null ? null : new SavedHead(trained, _headLabels, _maxLength));
    }

    /// <summary>
    /// Whether the trained classifier would load in Python too: always for a token classifier, and
    /// for a sequence classifier when it sits on BERT's own pooler, as
    /// <c>BertForSequenceClassification</c>'s does.
    /// </summary>
    public bool HeadLoadsInPython => _tagger is not null || _answerer is not null || _classifier?.Pooler is not null;

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

    /// <summary>Loads a PEFT adapter onto a base model, from the Hub or from a local directory.</summary>
    /// <param name="model">The base model.</param>
    /// <param name="adapter">An adapter repository id, or a directory written by <c>SaveAdapter</c> or by PEFT.</param>
    /// <param name="merge">Whether to fold it into the weights immediately.</param>
    public static PeftModel LoadAdapter(TransformerModel model, string adapter, bool merge = true)
        => Directory.Exists(adapter)
            ? PeftModel.Load(model, adapter, merge)
            : PeftModel.FromPretrained(model, adapter, "main", merge);

    /// <summary>Reads an adapter file without attaching it to anything.</summary>
    public static LoraAdapterSet ReadAdapter(string path) => LoraAdapterSet.Load(path);
}
