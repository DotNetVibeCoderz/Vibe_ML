using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviHub.Io;
using Gravicode.HFNet.GraviOptimum;
using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviDiffusers;

/// <summary>What loading a LoRA into a pipeline found.</summary>
/// <param name="UnetLayers">UNet layers the adapter changed.</param>
/// <param name="TextEncoderLayers">Text-encoder layers it changed.</param>
/// <param name="Unmatched">Adapter modules with no layer of that name in the ONNX graphs.</param>
public sealed record LoraReport(int UnetLayers, int TextEncoderLayers, IReadOnlyList<string> Unmatched)
{
    /// <inheritdoc />
    public override string ToString()
        => $"LoRA: {UnetLayers} UNet and {TextEncoderLayers} text-encoder layers"
            + (Unmatched.Count > 0 ? $", {Unmatched.Count} modules unmatched" : "");
}

/// <summary>
/// Merges Stable Diffusion LoRA adapters into ONNX weights.
/// </summary>
/// <remarks>
/// <para>
/// An ONNX export names its nodes after the PyTorch modules they came from -
/// <c>/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_q/MatMul</c> - but the weights of
/// its linear layers are anonymous (<c>onnx::MatMul_2567</c>). So an adapter is matched to its
/// layer through the node that consumes the weight, the delta <c>scale * up * down</c> is added to
/// the stored weight (transposed, since a MatMul holds <c>[in, out]</c>), and the result replaces
/// the original through <c>SessionOptions.AddInitializer</c>. The model file is never rewritten.
/// </para>
/// <para>
/// Three key layouts are read: kohya's (<c>lora_unet_..._to_q.lora_down.weight</c> with an
/// <c>alpha</c>), diffusers' and PEFT's (<c>unet....to_q.lora_A.weight</c>), and the older diffusers
/// attention-processor one (<c>...processor.to_q_lora.down.weight</c>). Each adapter's scale is
/// <c>alpha / rank</c> when an alpha is stored, 1 otherwise, times the scale passed in.
/// </para>
/// </remarks>
internal static class DiffusionLora
{
    private const string TextModel = "text_model.";

    /// <summary>One adapter: the module it changes and its two matrices.</summary>
    internal sealed record Adapter(string Component, string Module, NdArray Down, NdArray Up, double Scale);

    /// <summary>Resolves a file, a directory or a Hub repository to a weights file.</summary>
    internal static string Resolve(string source)
    {
        if (File.Exists(source)) return source;

        if (Directory.Exists(source))
        {
            var found = Directory.GetFiles(source, "*.safetensors").OrderBy(f => f.Contains("lora", StringComparison.OrdinalIgnoreCase) ? 0 : 1).FirstOrDefault();
            return found ?? throw new FileNotFoundException($"'{source}' holds no .safetensors file.");
        }

        var info = Hub.ModelInfo(source);
        var file = info.FilesWithExtension(".safetensors")
            .OrderBy(f => f.Path.Contains("lora", StringComparison.OrdinalIgnoreCase) ? 0 : 1)
            .FirstOrDefault();

        if (file.Path is null)
        {
            throw new HubException($"'{source}' publishes no .safetensors file, so there is no LoRA to load.") { RepoId = source };
        }

        return Hub.DownloadFile(source, file.Path);
    }

    /// <summary>Reads every adapter pair in a LoRA file.</summary>
    internal static IReadOnlyList<Adapter> Read(string path, double scale)
    {
        var tensors = SafeTensors.ReadAll(path);
        var pairs = new Dictionary<(string Component, string Module), (NdArray? Down, NdArray? Up, double? Alpha)>();

        foreach (var (name, tensor) in tensors)
        {
            if (Classify(name) is not var (component, module, role)) continue;

            pairs.TryGetValue((component, module), out var pair);
            pairs[(component, module)] = role switch
            {
                "down" => (tensor, pair.Up, pair.Alpha),
                "up" => (pair.Down, tensor, pair.Alpha),
                _ => (pair.Down, pair.Up, tensor.At(0)),
            };
        }

        var adapters = new List<Adapter>();
        foreach (var ((component, module), (down, up, alpha)) in pairs)
        {
            if (down is null || up is null)
            {
                throw new InvalidDataException(
                    $"'{module}' has only its {(down is null ? "up" : "down")} matrix; the LoRA file is incomplete.");
            }

            var rank = down.Shape[0];
            adapters.Add(new Adapter(component, module, down, up, scale * (alpha is { } a ? a / rank : 1.0)));
        }

        return adapters;
    }

    /// <summary>
    /// Which component, module and matrix a key belongs to, or <c>null</c> for a key that is not
    /// part of a LoRA pair.
    /// </summary>
    internal static (string Component, string Module, string Role)? Classify(string name)
    {
        // kohya: lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight
        foreach (var (prefix, component) in new[] { ("lora_unet_", "unet"), ("lora_te_", "text_encoder"), ("lora_te1_", "text_encoder") })
        {
            if (!name.StartsWith(prefix, StringComparison.Ordinal)) continue;

            var dot = name.IndexOf('.', StringComparison.Ordinal);
            if (dot < 0) return null;

            var module = name[prefix.Length..dot];
            var role = name[(dot + 1)..] switch
            {
                "lora_down.weight" => "down",
                "lora_up.weight" => "up",
                "alpha" => "alpha",
                _ => null,
            };

            // kohya keys use underscores for both separators; they are matched against the graph
            // with underscores too.
            return role is null ? null : (component, "kohya:" + module, role);
        }

        // diffusers / PEFT: unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight
        foreach (var (prefix, component) in new[] { ("unet.", "unet"), ("text_encoder.", "text_encoder"), ("base_model.model.", "unet") })
        {
            if (!name.StartsWith(prefix, StringComparison.Ordinal)) continue;

            var rest = name[prefix.Length..];
            foreach (var (marker, role) in new[]
            {
                (".lora_A.weight", "down"), (".lora_B.weight", "up"), (".lora.down.weight", "down"), (".lora.up.weight", "up"),
                (".alpha", "alpha"),
            })
            {
                if (rest.EndsWith(marker, StringComparison.Ordinal)) return (component, rest[..^marker.Length], role);
            }

            // Older attention processors: ...attn1.processor.to_q_lora.down.weight
            foreach (var (marker, role) in new[] { ("_lora.down.weight", "down"), ("_lora.up.weight", "up") })
            {
                if (!rest.EndsWith(marker, StringComparison.Ordinal)) continue;

                var module = rest[..^marker.Length].Replace(".processor.", ".", StringComparison.Ordinal);
                return (component, module.Replace(".to_out", ".to_out.0", StringComparison.Ordinal), role);
            }
        }

        return null;
    }

    /// <summary>
    /// The merged weights for one ONNX graph: every initializer an adapter changes, with the
    /// adapter's delta added.
    /// </summary>
    internal static Dictionary<string, InitializerOverride> Merge(
        OnnxModelFile model, IEnumerable<Adapter> adapters, List<string> unmatched, out int changedLayers)
    {
        var producers = model.Nodes.SelectMany(n => n.Outputs.Select(o => (o, n))).ToDictionary(p => p.o, p => p.n, StringComparer.Ordinal);

        // Module path -> (weight initializer, whether it is stored transposed as [in, out]).
        var layers = new Dictionary<string, (string Weight, bool Transposed)>(StringComparer.Ordinal);
        foreach (var node in model.Nodes)
        {
            if (node.OpType is not ("MatMul" or "Gemm" or "Conv") || node.Inputs.Count < 2) continue;

            var module = ModuleOf(node.Name);
            if (module.Length == 0) continue;

            var input = node.Inputs[1];
            if (model.Initializers.ContainsKey(input))
            {
                // A MatMul multiplies by [in, out]; Gemm (transB) and Conv hold PyTorch's [out, in].
                var transposed = node.OpType == "MatMul" || (node.OpType == "Gemm" && !node.Attributes.Contains("transB"));
                layers.TryAdd(module, (input, transposed));
            }
            else if (producers.TryGetValue(input, out var transpose) && transpose.OpType == "Transpose"
                && model.Initializers.ContainsKey(transpose.Inputs[0]))
            {
                // Some exports keep the PyTorch layout and transpose in the graph.
                layers.TryAdd(module, (transpose.Inputs[0], false));
            }
        }

        // transformers 5 dropped CLIPTextModel's text_model level, so a text encoder exported with it
        // names its layers encoder.layers.0... while nearly every LoRA file says
        // text_model.encoder.layers.0... Each layer answers to both spellings.
        foreach (var key in layers.Keys.ToList())
        {
            var alias = key.StartsWith(TextModel, StringComparison.Ordinal) ? key[TextModel.Length..] : TextModel + key;
            layers.TryAdd(alias, layers[key]);
        }

        var kohya = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var key in layers.Keys) kohya.TryAdd(key.Replace('.', '_'), key);
        var merged = new Dictionary<string, (float[] Values, OnnxInitializer Initializer)>(StringComparer.Ordinal);
        changedLayers = 0;

        foreach (var adapter in adapters)
        {
            var module = adapter.Module.StartsWith("kohya:", StringComparison.Ordinal)
                ? kohya.GetValueOrDefault(adapter.Module["kohya:".Length..])
                : layers.ContainsKey(adapter.Module) ? adapter.Module : null;

            if (module is null)
            {
                unmatched.Add($"{adapter.Component}:{adapter.Module}");
                continue;
            }

            var (weightName, transposed) = layers[module];
            var initializer = model.Initializers[weightName];

            if (!merged.TryGetValue(weightName, out var entry))
            {
                entry = (model.ReadFloats(initializer), initializer);
                merged[weightName] = entry;
            }

            AddDelta(entry.Values, initializer.Dimensions, adapter, transposed, module);
            changedLayers++;
        }

        return merged.ToDictionary(
            p => p.Key,
            p => new InitializerOverride(p.Value.Values, [.. p.Value.Initializer.Dimensions], p.Value.Initializer.DataType == 10),
            StringComparer.Ordinal);
    }

    /// <summary>
    /// The PyTorch module path an exporter recorded in a node's name:
    /// <c>/down_blocks.0/attentions.0/proj_in/Conv</c> becomes <c>down_blocks.0.attentions.0.proj_in</c>.
    /// </summary>
    internal static string ModuleOf(string nodeName)
    {
        var trimmed = nodeName.Trim('/');
        var last = trimmed.LastIndexOf('/');
        return last <= 0 ? "" : trimmed[..last].Replace('/', '.');
    }

    private static void AddDelta(float[] weight, IReadOnlyList<long> dims, Adapter adapter, bool transposed, string module)
    {
        var down = adapter.Down;
        var up = adapter.Up;
        var rank = down.Shape[0];
        var outputs = up.Shape[0];
        var inputSize = (int)(down.Size / rank);     // in, or in * kh * kw for a convolution

        if (up.Size / outputs != rank)
        {
            throw new InvalidDataException($"'{module}': the up matrix is [{string.Join(",", up.Shape.ToArray())}], which does not have rank {rank}.");
        }

        if (weight.Length != (long)outputs * inputSize)
        {
            throw new InvalidDataException(
                $"'{module}': the adapter makes a [{outputs} x {inputSize}] update but the ONNX weight is [{string.Join(" x ", dims)}]. "
                + "It was trained for a different model.");
        }

        var d = down.AsContiguous().ToArray();
        var u = up.AsContiguous().ToArray();

        for (var o = 0; o < outputs; o++)
        {
            for (var i = 0; i < inputSize; i++)
            {
                var sum = 0.0;
                for (var r = 0; r < rank; r++) sum += u[o * rank + r] * d[r * inputSize + i];

                var index = transposed ? i * outputs + o : o * inputSize + i;
                weight[index] += (float)(adapter.Scale * sum);
            }
        }
    }
}
