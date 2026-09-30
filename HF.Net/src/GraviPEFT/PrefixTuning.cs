using System.Text.Json;
using Gravicode.HFNet.GraviHub.Io;
using Gravicode.HFNet.GraviTransformers;
using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviPEFT;

/// <summary>How <see cref="PrefixTuningModel"/> sets up its prefix.</summary>
/// <param name="VirtualTokens">
/// How many learned positions every layer's attention sees ahead of the text. PEFT's examples use 20
/// to 30; each costs <c>layers * 2 * hidden</c> parameters - 36,864 on bert-base.
/// </param>
/// <param name="Seed">Seeds the prefix's initialisation and the head's.</param>
public sealed record PrefixTuningConfig(int VirtualTokens = 20, int Seed = 42);

/// <summary>
/// Prefix tuning: a frozen model steered by learned keys and values that every attention layer
/// reads ahead of the text.
/// </summary>
/// <remarks>
/// <para>
/// Where LoRA changes the weights a little, prefix tuning changes none of them. It adds a handful of
/// virtual positions to every layer - not tokens, which would be confined to the vocabulary's
/// embeddings, but a key and a value per layer and head, trained directly - and each real token can
/// attend to them as if they were more text. Only the prefix and the classifier are trained, usually
/// well under one percent of the model.
/// </para>
/// <para>
/// The format is PEFT's <c>PREFIX_TUNING</c>: <c>prompt_embeddings</c> of shape
/// <c>[virtual tokens, layers * 2 * hidden]</c>, each row holding every layer's key then value, and
/// on BERT the classifier saved as <c>base_model.classifier.*</c> over the frozen pooler - so an
/// adapter trained here loads in Python with <c>PeftModel.from_pretrained</c>, and one trained there
/// loads here. A prefix saved with <c>prefix_projection</c> holds the projection's output, which has
/// the same shape and is read the same way.
/// </para>
/// </remarks>
public sealed class PrefixTuningModel
{
    private const string HeadConfigFile = "hfnet_head.json";

    private readonly TransformerModel _model;
    private readonly NdArray _prefix;
    private readonly int _seed;
    private LoraEncoder? _encoder;
    private ClassifierHead? _head;
    private string[] _labels = [];
    private int _maxLength = 128;

    private PrefixTuningModel(TransformerModel model, NdArray prefix, int seed)
    {
        _model = model;
        _prefix = prefix;
        _seed = seed;
    }

    /// <summary>The frozen model.</summary>
    public TransformerModel Model => _model;

    /// <summary>The prefix, <c>[virtual tokens, layers * 2 * hidden]</c> - PEFT's <c>prompt_embeddings</c>.</summary>
    public NdArray Prefix => _prefix;

    /// <summary>How many virtual positions the prefix has.</summary>
    public int VirtualTokens => _prefix.Shape[0];

    /// <summary>The trained head's labels, in class order; empty before training.</summary>
    public IReadOnlyList<string> Labels => _labels;

    /// <summary>Trainable parameters - the prefix and the classifier - against the encoder's.</summary>
    public (long Trainable, long Encoder, double Fraction) ParameterEfficiency()
    {
        var trainable = _prefix.Size + (_head is null ? 0 : _head.Weight.Size + _head.Bias.Size);
        var encoder = _model.Encoder.Config.ParameterCount;
        return (trainable, encoder, encoder == 0 ? 0 : (double)trainable / encoder);
    }

    /// <summary>Attaches a freshly initialised prefix to a model.</summary>
    /// <param name="model">The pretrained model; its weights stay frozen.</param>
    /// <param name="config">The number of virtual tokens and the seed.</param>
    /// <remarks>The prefix starts as standard normal values, as PEFT's <c>nn.Embedding</c> does.</remarks>
    public static PrefixTuningModel Apply(TransformerModel model, PrefixTuningConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        config ??= new PrefixTuningConfig();
        if (config.VirtualTokens < 1) throw new ArgumentOutOfRangeException(nameof(config), "At least one virtual token is needed.");

        var width = Width(model);
        var random = new Random(config.Seed);
        var values = new double[config.VirtualTokens * width];
        for (var i = 0; i < values.Length; i += 2)
        {
            // Box-Muller: two standard normals from two uniforms.
            var radius = Math.Sqrt(-2 * Math.Log(1 - random.NextDouble()));
            var angle = 2 * Math.PI * random.NextDouble();
            values[i] = radius * Math.Cos(angle);
            if (i + 1 < values.Length) values[i + 1] = radius * Math.Sin(angle);
        }

        return new PrefixTuningModel(model, new NdArray(values, config.VirtualTokens, width), config.Seed);
    }

    /// <summary>Loads a prefix-tuning adapter saved by <see cref="Save"/> or by PEFT.</summary>
    /// <param name="model">The base model it was trained on.</param>
    /// <param name="directory">A folder holding <c>adapter_config.json</c> and <c>adapter_model.safetensors</c>.</param>
    /// <exception cref="NotSupportedException">The adapter is not a prefix-tuning adapter, or was made for another shape of model.</exception>
    public static PrefixTuningModel Load(TransformerModel model, string directory)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);

        using (var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "adapter_config.json"))))
        {
            var root = document.RootElement;
            var type = root.TryGetProperty("peft_type", out var t) ? t.GetString() : null;
            if (type != "PREFIX_TUNING")
            {
                throw new NotSupportedException(
                    $"'{directory}' is a {type ?? "unknown"} adapter, not PREFIX_TUNING. Load a LoRA adapter with PEFT.LoadAdapter.");
            }

            var layers = root.TryGetProperty("num_layers", out var l) ? l.GetInt32() : model.Config.Layers;
            var hidden = root.TryGetProperty("token_dim", out var d) ? d.GetInt32() : model.Config.HiddenSize;
            if (layers != model.Config.Layers || hidden != model.Config.HiddenSize)
            {
                throw new NotSupportedException(
                    $"The prefix was trained for {layers} layers of width {hidden}; '{model.RepoId}' has "
                    + $"{model.Config.Layers} of width {model.Config.HiddenSize}. Load the base model it was trained on.");
            }
        }

        var tensors = SafeTensors.ReadAll(Path.Combine(directory, "adapter_model.safetensors"));
        if (!tensors.TryGetValue("prompt_embeddings", out var prefix))
        {
            throw new InvalidDataException($"'{directory}' holds no prompt_embeddings tensor.");
        }

        var width = Width(model);
        if (prefix.Rank != 2 || prefix.Shape[1] != width)
        {
            throw new InvalidDataException(
                $"prompt_embeddings is [{string.Join(", ", prefix.Shape.ToArray())}]; this model needs [virtual tokens, {width}].");
        }

        var loaded = new PrefixTuningModel(model, prefix.AsContiguous(), 42);

        var weight = First(tensors, "base_model.classifier.weight", "base_model.model.classifier.weight", "classifier.weight");
        var bias = First(tensors, "base_model.classifier.bias", "base_model.model.classifier.bias", "classifier.bias");
        if (weight is not null && bias is not null)
        {
            loaded._head = ClassifierHead.FromWeights(weight, bias, PeftModel.PoolerOf(model));
            loaded._labels = ReadLabels(Path.Combine(directory, HeadConfigFile), weight.Shape[0], out loaded._maxLength);
        }

        return loaded;
    }

    // ------------------------------------------------------------------ training

    /// <summary>Trains the prefix and a classifier together, with every model weight frozen.</summary>
    /// <param name="texts">The training texts.</param>
    /// <param name="labels">One label per text.</param>
    /// <param name="options">Epochs, batch size, learning rate and schedule.</param>
    /// <remarks>
    /// The same loop as LoRA training: packed micro-batches, AdamW, a linear schedule, gradients
    /// clipped at norm 1, and the gradient taken through every layer by the hand-written backward
    /// pass, which carries each layer's share back into the prefix. Prefix tuning wants a larger
    /// learning rate than LoRA - PEFT's examples use around 1e-2.
    /// </remarks>
    public TrainingReport Train(IReadOnlyList<string> texts, IReadOnlyList<string> labels, TrainingOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(texts);
        ArgumentNullException.ThrowIfNull(labels);
        options ??= new TrainingOptions { LearningRate = 1e-2 };

        if (texts.Count != labels.Count) throw new ArgumentException($"{texts.Count} texts against {labels.Count} labels.", nameof(labels));
        if (texts.Count == 0) throw new ArgumentException("Nothing to train on.", nameof(texts));

        _maxLength = options.MaxLength;
        _labels = [.. labels.Distinct().Order(StringComparer.Ordinal)];
        var indexOf = _labels.Select((label, i) => (label, i)).ToDictionary(p => p.label, p => p.i, StringComparer.Ordinal);

        var ids = texts.Select(t => Tokenize(t, options.MaxLength)).ToArray();
        var targets = labels.Select(l => indexOf[l]).ToArray();

        var encoder = Encoder();
        var head = new ClassifierHead(encoder.Hidden, _labels.Length, options.Seed, PeftModel.PoolerOf(_model));
        var report = LoraTrainer.Fit(encoder, head, ids, targets, options);

        _head = head;
        return report;
    }

    /// <summary>Classifies a text with the trained head.</summary>
    public IReadOnlyList<Prediction> Predict(string text)
    {
        if (_head is null) throw new InvalidOperationException("No head has been trained or loaded. Call Train first.");

        var logits = Logits(text);
        var scores = ClassifierHead.Softmax(logits);
        return [.. scores.Select((score, k) => new Prediction(_labels[k], score, k)).OrderByDescending(p => p.Score)];
    }

    /// <summary>The head's raw scores for a text, before the softmax - what transformers returns as <c>logits</c>.</summary>
    public double[] Logits(string text)
    {
        if (_head is null) throw new InvalidOperationException("No head has been trained or loaded. Call Train first.");

        var ids = Tokenize(text, _maxLength);
        var hidden = Encoder().Forward(ids);
        return _head.Logits(hidden, ids.Length, _model.Config.HiddenSize);
    }

    /// <summary>Final hidden states for a text, with the prefix in every layer - <c>[tokens, hidden]</c>.</summary>
    public NdArray Hidden(string text)
    {
        var ids = Tokenize(text, _maxLength);
        return new NdArray(Encoder().Forward(ids), ids.Length, _model.Config.HiddenSize);
    }

    /// <summary>Accuracy of the trained head on held-out data.</summary>
    public double Score(IReadOnlyList<string> texts, IReadOnlyList<string> labels)
    {
        ArgumentNullException.ThrowIfNull(texts);
        ArgumentNullException.ThrowIfNull(labels);

        var correct = texts.Where((t, i) => Predict(t)[0].Label == labels[i]).Count();
        return texts.Count == 0 ? 0 : (double)correct / texts.Count;
    }

    // ------------------------------------------------------------------ saving

    /// <summary>Writes the prefix, and the head if one was trained, in PEFT's layout.</summary>
    /// <param name="directory">Where to write.</param>
    /// <remarks>
    /// On BERT the head reads the frozen pooler, as <c>BertForSequenceClassification</c>'s does, and is
    /// saved as PEFT saves it; the directory then loads in Python with
    /// <c>AutoModelForSequenceClassification</c> and <c>PeftModel.from_pretrained</c>.
    /// </remarks>
    public void Save(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        Directory.CreateDirectory(directory);

        var tensors = new Dictionary<string, NdArray>(StringComparer.Ordinal) { ["prompt_embeddings"] = _prefix };
        if (_head is not null)
        {
            tensors["base_model.classifier.weight"] = _head.Weight;
            tensors["base_model.classifier.bias"] = _head.Bias;
        }

        SafeTensors.Write(Path.Combine(directory, "adapter_model.safetensors"), tensors);

        var config = new Dictionary<string, object?>
        {
            ["peft_type"] = "PREFIX_TUNING",
            ["task_type"] = _head is null ? "FEATURE_EXTRACTION" : "SEQ_CLS",
            ["base_model_name_or_path"] = _model.RepoId,
            ["num_virtual_tokens"] = VirtualTokens,
            ["token_dim"] = _model.Config.HiddenSize,
            ["num_transformer_submodules"] = 1,
            ["num_attention_heads"] = _model.Config.Heads,
            ["num_layers"] = _model.Config.Layers,
            ["encoder_hidden_size"] = _model.Config.HiddenSize,
            ["prefix_projection"] = false,
            ["inference_mode"] = true,
        };

        if (_head is not null) config["modules_to_save"] = new[] { "classifier", "score" };

        var options = new JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(Path.Combine(directory, "adapter_config.json"), JsonSerializer.Serialize(config, options));

        var headConfig = Path.Combine(directory, HeadConfigFile);
        if (_head is null)
        {
            if (File.Exists(headConfig)) File.Delete(headConfig);
            return;
        }

        File.WriteAllText(headConfig, JsonSerializer.Serialize(new Dictionary<string, object?>
        {
            ["task"] = "sequence",
            ["id2label"] = _labels.Select((label, i) => (label, i)).ToDictionary(p => p.i.ToString(), p => p.label),
            ["max_length"] = _maxLength,
            ["pooling"] = _head.Pooler is null ? "mean" : "pooler",
        }, options));
    }

    // ------------------------------------------------------------------ helpers

    /// <summary>The training encoder with this prefix in the loop and no adapters, built once.</summary>
    internal LoraEncoder Encoder()
        => _encoder ??= LoraEncoder.Build(_model.Encoder, _model.Config, (_, _) => null, _model.Report.SegmentDelta, _prefix);

    private int[] Tokenize(string text, int maxLength)
    {
        var ids = _model.Tokenizer.Encode(text).ToIdArray();
        return ids.Length > maxLength ? ids[..maxLength] : ids;
    }

    private static int Width(TransformerModel model) => model.Config.Layers * 2 * model.Config.HiddenSize;

    private static NdArray? First(IReadOnlyDictionary<string, NdArray> tensors, params string[] names)
        => names.Select(n => tensors.TryGetValue(n, out var t) ? t : null).FirstOrDefault(t => t is not null);

    private static string[] ReadLabels(string path, int classes, out int maxLength)
    {
        maxLength = 128;
        var labels = Enumerable.Range(0, classes).Select(i => $"LABEL_{i}").ToArray();
        if (!File.Exists(path)) return labels;

        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;
        if (root.TryGetProperty("max_length", out var length)) maxLength = length.GetInt32();
        if (root.TryGetProperty("id2label", out var map))
        {
            foreach (var entry in map.EnumerateObject())
            {
                if (int.TryParse(entry.Name, out var index) && index < classes) labels[index] = entry.Value.GetString() ?? labels[index];
            }
        }

        return labels;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var (trainable, encoder, fraction) = ParameterEfficiency();
        return $"PrefixTuningModel({_model.RepoId}, {VirtualTokens} virtual tokens, {trainable:N0}/{encoder:N0} params = {fraction:P3})";
    }
}
