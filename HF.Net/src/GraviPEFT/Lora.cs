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
    {
        _adapters = new Dictionary<string, LoraAdapter>(adapters, StringComparer.Ordinal);
        Config = config;
    }

    /// <summary>The shared configuration.</summary>
    public LoraConfig Config { get; }

    /// <summary>The adapters, keyed by the base parameter path they adapt.</summary>
    public IReadOnlyDictionary<string, LoraAdapter> Adapters => _adapters;

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

        foreach (var (name, tensor) in tensors)
        {
            var isA = name.Contains(".lora_A", StringComparison.Ordinal);
            var isB = name.Contains(".lora_B", StringComparison.Ordinal);
            if (!isA && !isB) continue;

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

        return new LoraAdapterSet(adapters, config);
    }

    /// <summary>Writes the set in the PEFT layout: weights plus an <c>adapter_config.json</c>.</summary>
    /// <param name="directory">Directory to write into; created if missing.</param>
    /// <param name="baseModelId">The model the adapter was trained against, recorded in the config.</param>
    public void Save(string directory, string? baseModelId = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        Directory.CreateDirectory(directory);

        var tensors = new Dictionary<string, NdArray>(StringComparer.Ordinal);
        foreach (var (basePath, adapter) in _adapters)
        {
            tensors[$"base_model.model.{basePath}.lora_A.weight"] = adapter.A;
            tensors[$"base_model.model.{basePath}.lora_B.weight"] = adapter.B;
        }

        SafeTensors.Write(Path.Combine(directory, "adapter_model.safetensors"), tensors);

        var config = new Dictionary<string, object?>
        {
            ["peft_type"] = "LORA",
            ["task_type"] = "FEATURE_EXTRACTION",
            ["base_model_name_or_path"] = baseModelId,
            ["r"] = Config.Rank,
            ["lora_alpha"] = Config.Alpha,
            ["lora_dropout"] = Config.Dropout,
            ["target_modules"] = Config.Targets,
            ["bias"] = "none",
            ["inference_mode"] = true,
        };

        File.WriteAllText(
            Path.Combine(directory, "adapter_config.json"),
            JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static LoraConfig ReadConfig(string path)
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
