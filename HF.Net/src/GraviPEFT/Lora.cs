using System.Text.Json;
using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviHub.Io;
using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviPEFT;

/// <summary>How a LoRA adapter is shaped and where it attaches.</summary>
/// <param name="Rank">The bottleneck width. 4 to 16 covers most tasks.</param>
/// <param name="Alpha">Scaling numerator; the update is multiplied by <c>Alpha / Rank</c>.</param>
/// <param name="TargetModules">
/// Which projections to adapt, by name suffix - <c>query</c>, <c>value</c> and so on.
/// </param>
/// <param name="Dropout">Dropout applied to the adapter input during training.</param>
/// <remarks>
/// Adapting only the query and value projections is the original paper's recommendation and is the
/// default here. Adding key and output roughly doubles the trainable parameters for a change that
/// is usually within noise.
/// </remarks>
public sealed record LoraConfig(
    int Rank = 8,
    double Alpha = 16,
    IReadOnlyList<string>? TargetModules = null,
    double Dropout = 0.0)
{
    /// <summary>The projections adapted when none are named.</summary>
    public static IReadOnlyList<string> DefaultTargets { get; } = ["query", "value"];

    /// <summary>The projections this configuration adapts.</summary>
    public IReadOnlyList<string> Targets => TargetModules ?? DefaultTargets;

    /// <summary>The multiplier applied to the low-rank product.</summary>
    /// <remarks>
    /// Scaling by <c>Alpha / Rank</c> is what makes the rank a capacity knob rather than a learning
    /// rate knob: without it, doubling the rank doubles the size of the update at initialisation.
    /// </remarks>
    public double Scaling => Alpha / Rank;
}

/// <summary>
/// One low-rank adapter: a pair of matrices whose product is added to a frozen weight.
/// </summary>
/// <remarks>
/// <b>B starts at zero.</b> That is the defining property of LoRA, not an implementation detail -
/// it makes the adapted model identical to the base model at step zero, so training starts from the
/// pretrained behaviour rather than from a perturbation of it. Initialising both matrices randomly
/// trains, converges, and reaches a measurably worse place.
/// </remarks>
public sealed class LoraAdapter
{
    /// <summary>Creates an adapter for a weight of the given shape.</summary>
    /// <param name="inputs">Input width of the layer being adapted.</param>
    /// <param name="outputs">Output width of the layer being adapted.</param>
    /// <param name="config">Rank and scaling.</param>
    /// <param name="seed">Seed for A's initialisation.</param>
    public LoraAdapter(int inputs, int outputs, LoraConfig config, int seed = 42)
    {
        Config = config;
        Inputs = inputs;
        Outputs = outputs;

        var random = new GraviRandom(seed);
        A = NdArray.Zeros(config.Rank, inputs);
        B = NdArray.Zeros(outputs, config.Rank);

        // Kaiming-uniform on A, zeros on B.
        var bound = Math.Sqrt(1.0 / inputs);
        for (var r = 0; r < config.Rank; r++)
        {
            for (var i = 0; i < inputs; i++) A[r, i] = (random.NextDouble() * 2 - 1) * bound;
        }
    }

    private LoraAdapter(NdArray a, NdArray b, LoraConfig config)
    {
        A = a;
        B = b;
        Config = config;
        Inputs = a.Shape[1];
        Outputs = b.Shape[0];
    }

    /// <summary>The down-projection, <c>[rank, inputs]</c>.</summary>
    public NdArray A { get; }

    /// <summary>The up-projection, <c>[outputs, rank]</c>.</summary>
    public NdArray B { get; }

    /// <summary>The configuration this adapter was built with.</summary>
    public LoraConfig Config { get; }

    /// <summary>Input width of the adapted layer.</summary>
    public int Inputs { get; }

    /// <summary>Output width of the adapted layer.</summary>
    public int Outputs { get; }

    /// <summary>Number of trainable values, against the full weight's <c>inputs x outputs</c>.</summary>
    public long ParameterCount => (long)A.Size + B.Size;

    /// <summary>Builds an adapter from a stored pair of matrices.</summary>
    public static LoraAdapter FromMatrices(NdArray a, NdArray b, LoraConfig config) => new(a, b, config);

    /// <summary>The dense update this adapter represents, <c>scaling * B A</c>.</summary>
    /// <returns>A matrix shaped <c>[outputs, inputs]</c>, matching PyTorch's weight layout.</returns>
    public NdArray Delta()
    {
        var delta = NdArray.Zeros(Outputs, Inputs);
        var scaling = Config.Scaling;
        var rank = Config.Rank;

        for (var o = 0; o < Outputs; o++)
        {
            for (var i = 0; i < Inputs; i++)
            {
                var sum = 0.0;
                for (var r = 0; r < rank; r++) sum += B[o, r] * A[r, i];
                delta[o, i] = sum * scaling;
            }
        }

        return delta;
    }

    /// <summary>Adds this adapter's update into a weight matrix, in place.</summary>
    /// <param name="weight">
    /// The layer's weight. Accepts either orientation and matches on shape, because the encoder
    /// here stores <c>[inputs, outputs]</c> while the checkpoint format is <c>[outputs, inputs]</c>.
    /// </param>
    public void MergeInto(NdArray weight)
    {
        var scaling = Config.Scaling;
        var rank = Config.Rank;

        var transposed = weight.Shape[0] == Inputs && weight.Shape[1] == Outputs;

        if (!transposed && (weight.Shape[0] != Outputs || weight.Shape[1] != Inputs))
        {
            throw new ArgumentException(
                $"Adapter is {Outputs}x{Inputs} but the weight is {weight.Shape[0]}x{weight.Shape[1]}.",
                nameof(weight));
        }

        for (var o = 0; o < Outputs; o++)
        {
            for (var i = 0; i < Inputs; i++)
            {
                var sum = 0.0;
                for (var r = 0; r < rank; r++) sum += B[o, r] * A[r, i];

                if (transposed) weight[i, o] += sum * scaling;
                else weight[o, i] += sum * scaling;
            }
        }
    }

    /// <inheritdoc />
    public override string ToString()
        => $"LoRA r={Config.Rank} {Outputs}x{Inputs} ({ParameterCount:N0} params, "
            + $"{(double)ParameterCount / ((long)Outputs * Inputs):P2} of the full weight)";
}

/// <summary>
/// A set of LoRA adapters keyed by the parameter they attach to, in the Hugging Face PEFT layout.
/// </summary>
/// <remarks>
/// PEFT stores adapters as <c>adapter_model.safetensors</c> beside an <c>adapter_config.json</c>,
/// with names like
/// <c>base_model.model.bert.encoder.layer.0.attention.self.query.lora_A.weight</c>. Reading and
/// writing that exact layout is what makes an adapter trained in Python usable here, and one
/// produced here usable there.
/// </remarks>
public sealed class LoraAdapterSet
{
    private readonly Dictionary<string, LoraAdapter> _adapters;

    /// <summary>Creates a set from adapters keyed by target parameter path.</summary>
    /// <param name="adapters">Adapters, keyed by the base parameter they adapt.</param>
    /// <param name="config">The configuration they share.</param>
    public LoraAdapterSet(IReadOnlyDictionary<string, LoraAdapter> adapters, LoraConfig config)
        : this(adapters, config, new Dictionary<string, NdArray>(StringComparer.Ordinal))
    {
    }

    private LoraAdapterSet(
        IReadOnlyDictionary<string, LoraAdapter> adapters, LoraConfig config, IReadOnlyDictionary<string, NdArray> others)
    {
        _adapters = new Dictionary<string, LoraAdapter>(adapters, StringComparer.Ordinal);
        Config = config;
        Others = others;
    }

    /// <summary>The shared configuration.</summary>
    public LoraConfig Config { get; }

    /// <summary>The adapters, keyed by the base parameter path they adapt.</summary>
    public IReadOnlyDictionary<string, LoraAdapter> Adapters => _adapters;

    /// <summary>
    /// Tensors in the adapter file that are not LoRA pairs, by name with PEFT's prefixes removed -
    /// <c>classifier.weight</c> from a sequence classification adapter, for instance.
    /// </summary>
    internal IReadOnlyDictionary<string, NdArray> Others { get; }

    /// <summary>The directory the adapter file was read from, where a saved head sits beside it.</summary>
    internal string? SourceDirectory { get; private set; }

    /// <summary>Total trainable values across every adapter.</summary>
    public long ParameterCount => _adapters.Values.Sum(a => a.ParameterCount);

    /// <summary>Downloads a PEFT adapter from the Hub and reads it.</summary>
    /// <param name="repoId">An adapter repository id.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    public static LoraAdapterSet FromPretrained(string repoId, string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var info = Hub.ModelInfo(repoId, revision);
        var available = info.Files.Select(f => f.Path).ToHashSet(StringComparer.Ordinal);

        var config = available.Contains("adapter_config.json")
            ? ReadConfig(Hub.DownloadFile(repoId, "adapter_config.json", revision))
            : new LoraConfig();

        // The head's side files land in the same snapshot directory, where LoadHead finds them.
        foreach (var extra in (string[])[HeadConfigFile, HeadWeightsFile])
        {
            if (available.Contains(extra)) Hub.DownloadFile(repoId, extra, revision);
        }

        if (available.Contains("adapter_model.safetensors"))
        {
            return Load(Hub.DownloadFile(repoId, "adapter_model.safetensors", revision), config);
        }

        if (available.Contains("adapter_model.bin"))
        {
            return Load(Hub.DownloadFile(repoId, "adapter_model.bin", revision), config);
        }

        throw new HubException(
            $"'{repoId}' has no adapter_model.safetensors or adapter_model.bin; it is not a PEFT adapter.")
        {
            RepoId = repoId,
        };
    }

    /// <summary>Reads an adapter file.</summary>
    /// <param name="path">An <c>adapter_model.safetensors</c> or <c>.bin</c>.</param>
    /// <param name="config">The configuration, usually from <c>adapter_config.json</c>.</param>
    public static LoraAdapterSet Load(string path, LoraConfig? config = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        config ??= new LoraConfig();

        var tensors = Path.GetExtension(path).Equals(".safetensors", StringComparison.OrdinalIgnoreCase)
            ? SafeTensors.ReadAll(path)
            : PyTorchCheckpoint.ReadAll(path);

        // Pair each lora_A with its lora_B by the base path they share.
        var pairs = new Dictionary<string, (NdArray? A, NdArray? B)>(StringComparer.Ordinal);
        var others = new Dictionary<string, NdArray>(StringComparer.Ordinal);

        foreach (var (name, tensor) in tensors)
        {
            var isA = name.Contains(".lora_A", StringComparison.Ordinal);
            var isB = name.Contains(".lora_B", StringComparison.Ordinal);

            if (!isA && !isB)
            {
                // modules_to_save copies, such as a classifier. Older PEFT releases kept the
                // wrapper's own path segment in the name; either spelling means the same module.
                others[Normalize(name).Replace(".modules_to_save", "", StringComparison.Ordinal)] = tensor;
                continue;
            }

            var basePath = Normalize(name);
            pairs.TryGetValue(basePath, out var pair);
            pairs[basePath] = isA ? (tensor, pair.B) : (pair.A, tensor);
        }

        var adapters = new Dictionary<string, LoraAdapter>(StringComparer.Ordinal);
        foreach (var (basePath, (a, b)) in pairs)
        {
            // A half-pair means a truncated or hand-edited file. Applying it would add a
            // zero-rank update that looks like a working adapter doing nothing.
            if (a is null || b is null)
            {
                throw new InvalidDataException(
                    $"'{basePath}' has only its lora_{(a is null ? "B" : "A")} matrix; the adapter file is incomplete.");
            }

            adapters[basePath] = LoraAdapter.FromMatrices(a, b, config);
        }

        return new LoraAdapterSet(adapters, config, others)
        {
            SourceDirectory = Path.GetDirectoryName(Path.GetFullPath(path)),
        };
    }

    /// <summary>Writes the set in the PEFT layout: weights plus an <c>adapter_config.json</c>.</summary>
    /// <param name="directory">Directory to write into; created if missing.</param>
    /// <param name="baseModelId">The model the adapter was trained against, recorded in the config.</param>
    public void Save(string directory, string? baseModelId = null) => Save(directory, baseModelId, head: null);

    /// <summary>The file naming the labels and pooling of a saved head. HF.Net's own; PEFT ignores it.</summary>
    internal const string HeadConfigFile = "hfnet_head.json";

    /// <summary>A mean-pooled head's weights, which have no PEFT equivalent to be saved as.</summary>
    internal const string HeadWeightsFile = "head.safetensors";

    /// <summary>Writes the adapters and, when there is one, the classifier trained with them.</summary>
    /// <remarks>
    /// A head over BERT's pooler is <c>BertForSequenceClassification</c>'s <c>classifier</c>, so it is
    /// written the way PEFT writes a <c>modules_to_save</c> module - into the adapter file as
    /// <c>base_model.model.classifier.*</c>, with <c>task_type</c> <c>SEQ_CLS</c> - and loads in
    /// Python. A mean-pooled head has no counterpart there and goes into a file of its own. Either
    /// way the label names go into <c>hfnet_head.json</c> rather than <c>adapter_config.json</c>,
    /// where PEFT would answer an unknown key with advice to upgrade.
    /// </remarks>
    internal void Save(string directory, string? baseModelId, SavedHead? head)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        Directory.CreateDirectory(directory);

        var tensors = new Dictionary<string, NdArray>(StringComparer.Ordinal);
        foreach (var (basePath, adapter) in _adapters)
        {
            tensors[$"base_model.model.{basePath}.lora_A.weight"] = adapter.A;
            tensors[$"base_model.model.{basePath}.lora_B.weight"] = adapter.B;
        }

        // A token head is all Transformers' token classifiers have, on every family; a sequence head
        // matches BertForSequenceClassification only when it reads BERT's pooler.
        var isToken = head?.Head is TokenClassifierHead;
        var isSpan = head?.Head is SpanHead;
        var inPeftLayout = isToken || isSpan || (head?.Head as ClassifierHead)?.Pooler is not null;

        // PEFT saves a modules_to_save module under the module's own name: classifier for the two
        // classification tasks, qa_outputs for question answering.
        var module = isSpan ? "qa_outputs" : "classifier";
        if (inPeftLayout)
        {
            tensors[$"base_model.model.{module}.weight"] = head!.Head.Weight;
            tensors[$"base_model.model.{module}.bias"] = head.Head.Bias;
        }

        SafeTensors.Write(Path.Combine(directory, "adapter_model.safetensors"), tensors);

        // A previous save into the same directory must not leave a head behind that no longer
        // matches the adapters beside it.
        foreach (var stale in (string[])[HeadConfigFile, HeadWeightsFile])
        {
            var path = Path.Combine(directory, stale);
            if (File.Exists(path)) File.Delete(path);
        }

        if (head is not null)
        {
            if (!inPeftLayout)
            {
                SafeTensors.Write(Path.Combine(directory, HeadWeightsFile), new Dictionary<string, NdArray>
                {
                    ["classifier.weight"] = head.Head.Weight,
                    ["classifier.bias"] = head.Head.Bias,
                });
            }

            var description = new Dictionary<string, object?>
            {
                ["task"] = isSpan ? "question_answering" : isToken ? "token" : "sequence",
                ["id2label"] = head.Labels.Select((label, i) => (label, i)).ToDictionary(p => p.i.ToString(), p => p.label),
                ["max_length"] = head.MaxLength,
            };

            if (!isToken && !isSpan) description["pooling"] = inPeftLayout ? "pooler" : "mean";

            File.WriteAllText(
                Path.Combine(directory, HeadConfigFile),
                JsonSerializer.Serialize(description, new JsonSerializerOptions { WriteIndented = true }));
        }

        var config = new Dictionary<string, object?>
        {
            ["peft_type"] = "LORA",
            ["task_type"] = isSpan ? "QUESTION_ANS" : isToken ? "TOKEN_CLS" : inPeftLayout ? "SEQ_CLS" : "FEATURE_EXTRACTION",
            ["base_model_name_or_path"] = baseModelId,
            ["r"] = Config.Rank,
            ["lora_alpha"] = Config.Alpha,
            ["lora_dropout"] = Config.Dropout,
            ["target_modules"] = Config.Targets,
            ["bias"] = "none",
            ["inference_mode"] = true,
        };

        // What PEFT itself writes for each task: the head is trained and saved in full.
        if (inPeftLayout) config["modules_to_save"] = isSpan ? new[] { "qa_outputs" } : new[] { "classifier", "score" };

        File.WriteAllText(
            Path.Combine(directory, "adapter_config.json"),
            JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true }));
    }

    internal static LoraConfig ReadConfig(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;

        return new LoraConfig(
            Rank: root.TryGetProperty("r", out var r) ? r.GetInt32() : 8,
            Alpha: root.TryGetProperty("lora_alpha", out var alpha) ? alpha.GetDouble() : 16,
            TargetModules: root.TryGetProperty("target_modules", out var targets)
                && targets.ValueKind == JsonValueKind.Array
                ? [.. targets.EnumerateArray().Select(t => t.GetString() ?? "")]
                : null,
            Dropout: root.TryGetProperty("lora_dropout", out var dropout) ? dropout.GetDouble() : 0.0);
    }

    /// <summary>Strips PEFT's wrapper prefixes and the lora suffix to recover the base parameter path.</summary>
    private static string Normalize(string name)
    {
        var path = name;

        foreach (var prefix in (string[])["base_model.model.", "base_model."])
        {
            if (path.StartsWith(prefix, StringComparison.Ordinal))
            {
                path = path[prefix.Length..];
                break;
            }
        }

        var at = path.IndexOf(".lora_", StringComparison.Ordinal);
        return at > 0 ? path[..at] : path;
    }

    /// <inheritdoc />
    public override string ToString()
        => $"LoraAdapterSet(r={Config.Rank}, {_adapters.Count} adapters, {ParameterCount:N0} parameters)";
}

/// <summary>A trained classifier and what is needed to use it again.</summary>
/// <param name="Head">The classifier, and the pooler it reads through if any.</param>
/// <param name="Labels">Label names, by class index.</param>
/// <param name="MaxLength">The truncation it was trained with.</param>
internal sealed record SavedHead(LinearHead Head, IReadOnlyList<string> Labels, int MaxLength);

/// <summary>What <c>hfnet_head.json</c> says.</summary>
/// <param name="Task"><c>sequence</c> or <c>token</c>.</param>
/// <param name="Pooling">For a sequence head, <c>pooler</c> or <c>mean</c>.</param>
/// <param name="Labels">Label names, by class index.</param>
/// <param name="MaxLength">The truncation the head was trained with.</param>
internal sealed record HeadDescription(string Task, string Pooling, IReadOnlyList<string> Labels, int MaxLength)
{
    /// <summary>
    /// The <c>task_type</c> in an adapter's <c>adapter_config.json</c>, or <c>null</c> when it has none.
    /// </summary>
    /// <remarks>
    /// A sequence classifier and a token classifier store <c>classifier.weight</c> at the same shape,
    /// so an adapter trained in Python, which has no <c>hfnet_head.json</c>, can only be told apart
    /// by this.
    /// </remarks>
    internal static string? TaskType(string directory)
    {
        var path = Path.Combine(directory, "adapter_config.json");
        if (!File.Exists(path)) return null;

        using var document = JsonDocument.Parse(File.ReadAllText(path));
        return document.RootElement.TryGetProperty("task_type", out var task) && task.ValueKind == JsonValueKind.String
            ? task.GetString()
            : null;
    }

    /// <summary>Reads the description saved beside an adapter, or returns <c>null</c> when there is none.</summary>
    internal static HeadDescription? Read(string directory)
    {
        var path = Path.Combine(directory, LoraAdapterSet.HeadConfigFile);
        if (!File.Exists(path)) return null;

        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;

        var labels = new SortedDictionary<int, string>();
        if (root.TryGetProperty("id2label", out var map))
        {
            foreach (var entry in map.EnumerateObject()) labels[int.Parse(entry.Name)] = entry.Value.GetString() ?? entry.Name;
        }

        return new HeadDescription(
            root.TryGetProperty("task", out var task) ? task.GetString() ?? "sequence" : "sequence",
            root.TryGetProperty("pooling", out var pooling) && pooling.ValueKind == JsonValueKind.String
                ? pooling.GetString()!
                : "pooler",
            [.. labels.Values],
            root.TryGetProperty("max_length", out var max) ? max.GetInt32() : 512);
    }
}
