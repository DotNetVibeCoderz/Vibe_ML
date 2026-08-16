using System.Globalization;
using LocalGen.Core.Engines;
using LocalGen.Core.Inference;

namespace LocalGen.Core.Modelfiles;

/// <summary>
/// Parses the Modelfile format. Instructions are one per line (<c>FROM</c>, <c>SYSTEM</c>,
/// <c>PARAMETER</c>, …) and long values may be wrapped in triple quotes to span lines.
/// Lines starting with <c>#</c> are comments.
/// </summary>
public static class ModelfileParser
{
    private const string TripleQuote = "\"\"\"";

    public static Modelfile Parse(string content)
    {
        string? from = null;
        string? system = null;
        string? template = null;
        string? license = null;
        var isEmbedding = false;
        var adapters = new List<string>();
        var messages = new List<ChatMessage>();
        var stops = new List<string>();
        var extras = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        // Sampling and load settings accumulate across PARAMETER lines before being folded
        // into their respective option records at the end.
        var gen = new GenerationOptionsBuilder();
        var load = new LoadOptionsBuilder();

        var lines = content.Replace("\r\n", "\n").Split('\n');
        var index = 0;

        while (index < lines.Length)
        {
            var lineNumber = index + 1;
            var raw = lines[index];
            var trimmed = raw.Trim();
            index++;

            if (trimmed.Length == 0 || trimmed.StartsWith('#'))
            {
                continue;
            }

            var (instruction, rest) = SplitInstruction(trimmed);
            var value = ReadValue(rest, lines, ref index, lineNumber);

            switch (instruction.ToUpperInvariant())
            {
                case "FROM":
                    from = value;
                    break;

                case "SYSTEM":
                    system = value;
                    break;

                case "TEMPLATE":
                    template = value;
                    break;

                case "LICENSE":
                    license = value;
                    break;

                case "ADAPTER":
                    adapters.Add(value);
                    break;

                case "EMBEDDING":
                    isEmbedding = ParseBool(value, lineNumber);
                    break;

                case "MESSAGE":
                    messages.Add(ParseMessage(value, lineNumber));
                    break;

                case "PARAMETER":
                    ApplyParameter(value, lineNumber, gen, load, stops, extras);
                    break;

                default:
                    throw new ModelfileException($"unknown instruction '{instruction}'", lineNumber);
            }
        }

        if (string.IsNullOrWhiteSpace(from))
        {
            throw new ModelfileException("a FROM instruction is required", 1);
        }

        return new Modelfile
        {
            From = from,
            System = system,
            Template = template,
            License = license,
            IsEmbedding = isEmbedding,
            Adapters = adapters,
            Messages = messages,
            Extras = extras,
            Parameters = gen.Build(stops),
            LoadOptions = load.Build(isEmbedding)
        };
    }

    public static async Task<Modelfile> ParseFileAsync(
        string path,
        CancellationToken cancellationToken = default)
    {
        var text = await File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false);
        return Parse(text);
    }

    /// <summary>Renders a Modelfile back to text, used by <c>localgen show --modelfile</c>.</summary>
    public static string Render(Modelfile modelfile)
    {
        var sb = new System.Text.StringBuilder();
        sb.Append("FROM ").AppendLine(modelfile.From);

        foreach (var adapter in modelfile.Adapters)
        {
            sb.Append("ADAPTER ").AppendLine(adapter);
        }

        var p = modelfile.Parameters;
        AppendParameter(sb, "temperature", p.Temperature);
        AppendParameter(sb, "top_p", p.TopP);
        AppendParameter(sb, "top_k", p.TopK);
        AppendParameter(sb, "min_p", p.MinP);
        AppendParameter(sb, "repeat_penalty", p.RepeatPenalty);
        AppendParameter(sb, "repeat_last_n", p.RepeatLastN);
        AppendParameter(sb, "presence_penalty", p.PresencePenalty);
        AppendParameter(sb, "frequency_penalty", p.FrequencyPenalty);
        AppendParameter(sb, "num_predict", p.MaxTokens);
        AppendParameter(sb, "seed", p.Seed);
        AppendParameter(sb, "num_ctx", modelfile.LoadOptions.ContextSize);
        AppendParameter(sb, "num_gpu", modelfile.LoadOptions.GpuLayerCount);
        AppendParameter(sb, "num_thread", modelfile.LoadOptions.ThreadCount);
        AppendParameter(sb, "num_batch", modelfile.LoadOptions.BatchSize);

        foreach (var stop in p.StopSequences)
        {
            sb.Append("PARAMETER stop ").AppendLine(Quote(stop));
        }

        if (modelfile.LoadOptions.Device != DeviceKind.Auto)
        {
            sb.Append("PARAMETER device ").AppendLine(modelfile.LoadOptions.Device.ToString().ToLowerInvariant());
        }

        if (modelfile.IsEmbedding)
        {
            sb.AppendLine("EMBEDDING true");
        }

        AppendBlock(sb, "SYSTEM", modelfile.System);
        AppendBlock(sb, "TEMPLATE", modelfile.Template);
        AppendBlock(sb, "LICENSE", modelfile.License);

        foreach (var message in modelfile.Messages)
        {
            sb.Append("MESSAGE ")
              .Append(message.Role.ToString().ToLowerInvariant())
              .Append(' ')
              .AppendLine(Quote(message.Text));
        }

        return sb.ToString();
    }

    private static (string Instruction, string Remainder) SplitInstruction(string line)
    {
        var space = line.IndexOfAny([' ', '\t']);
        return space < 0
            ? (line, string.Empty)
            : (line[..space], line[(space + 1)..].Trim());
    }

    /// <summary>
    /// Reads an instruction's value, consuming extra lines when it opens a triple-quoted block.
    /// </summary>
    private static string ReadValue(string rest, string[] lines, ref int index, int startLine)
    {
        if (!rest.StartsWith(TripleQuote, StringComparison.Ordinal))
        {
            return Unquote(rest);
        }

        var body = rest[TripleQuote.Length..];

        // A block that opens and closes on the same line.
        var closing = body.IndexOf(TripleQuote, StringComparison.Ordinal);
        if (closing >= 0)
        {
            return body[..closing];
        }

        var sb = new System.Text.StringBuilder(body);
        while (index < lines.Length)
        {
            var line = lines[index];
            index++;

            var end = line.IndexOf(TripleQuote, StringComparison.Ordinal);
            if (end >= 0)
            {
                if (sb.Length > 0)
                {
                    sb.Append('\n');
                }

                sb.Append(line[..end]);
                return sb.ToString().Trim('\n');
            }

            if (sb.Length > 0)
            {
                sb.Append('\n');
            }

            sb.Append(line);
        }

        throw new ModelfileException("unterminated \"\"\" block", startLine);
    }

    private static void ApplyParameter(
        string value,
        int lineNumber,
        GenerationOptionsBuilder gen,
        LoadOptionsBuilder load,
        List<string> stops,
        Dictionary<string, string> extras)
    {
        var (key, arg) = SplitInstruction(value);
        if (string.IsNullOrWhiteSpace(arg))
        {
            throw new ModelfileException($"PARAMETER '{key}' is missing a value", lineNumber);
        }

        arg = Unquote(arg.Trim());

        switch (key.ToLowerInvariant())
        {
            // Sampling
            case "temperature": gen.Temperature = ParseFloat(arg, key, lineNumber); break;
            case "top_p": gen.TopP = ParseFloat(arg, key, lineNumber); break;
            case "top_k": gen.TopK = ParseInt(arg, key, lineNumber); break;
            case "min_p": gen.MinP = ParseFloat(arg, key, lineNumber); break;
            case "repeat_penalty": gen.RepeatPenalty = ParseFloat(arg, key, lineNumber); break;
            case "repeat_last_n": gen.RepeatLastN = ParseInt(arg, key, lineNumber); break;
            case "presence_penalty": gen.PresencePenalty = ParseFloat(arg, key, lineNumber); break;
            case "frequency_penalty": gen.FrequencyPenalty = ParseFloat(arg, key, lineNumber); break;
            case "num_predict" or "max_tokens": gen.MaxTokens = ParseInt(arg, key, lineNumber); break;
            case "seed": gen.Seed = (uint)ParseInt(arg, key, lineNumber); break;
            case "stop": stops.Add(arg); break;

            // Load-time
            case "num_ctx" or "context_length": load.ContextSize = ParseInt(arg, key, lineNumber); break;
            case "num_gpu" or "gpu_layers": load.GpuLayerCount = ParseInt(arg, key, lineNumber); break;
            case "num_thread" or "threads": load.ThreadCount = ParseInt(arg, key, lineNumber); break;
            case "num_batch" or "batch_size": load.BatchSize = ParseInt(arg, key, lineNumber); break;
            case "use_mmap": load.UseMemoryMap = ParseBool(arg, lineNumber); break;
            case "use_mlock": load.UseMemoryLock = ParseBool(arg, lineNumber); break;
            case "device": load.Device = ParseDevice(arg, lineNumber); break;
            case "tensor_split":
                load.TensorSplit = arg
                    .Split([',', ' '], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Select(x => ParseFloat(x, key, lineNumber))
                    .ToList();
                break;

            default:
                extras[key] = arg;
                break;
        }
    }

    private static ChatMessage ParseMessage(string value, int lineNumber)
    {
        var (role, text) = SplitInstruction(value);
        text = Unquote(text.Trim());

        return role.ToLowerInvariant() switch
        {
            "system" => ChatMessage.System(text),
            "user" => ChatMessage.User(text),
            "assistant" => ChatMessage.Assistant(text),
            _ => throw new ModelfileException($"unknown MESSAGE role '{role}'", lineNumber)
        };
    }

    private static DeviceKind ParseDevice(string value, int lineNumber) =>
        Enum.TryParse<DeviceKind>(value, ignoreCase: true, out var device)
            ? device
            : throw new ModelfileException($"unknown device '{value}'", lineNumber);

    private static float ParseFloat(string value, string key, int lineNumber) =>
        float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed)
            ? parsed
            : throw new ModelfileException($"'{key}' expects a number but got '{value}'", lineNumber);

    private static int ParseInt(string value, string key, int lineNumber) =>
        int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed)
            ? parsed
            : throw new ModelfileException($"'{key}' expects an integer but got '{value}'", lineNumber);

    private static bool ParseBool(string value, int lineNumber) => value.ToLowerInvariant() switch
    {
        "true" or "1" or "yes" or "on" => true,
        "false" or "0" or "no" or "off" => false,
        _ => throw new ModelfileException($"expected a boolean but got '{value}'", lineNumber)
    };

    private static string Unquote(string value)
    {
        if (value.Length >= 2 &&
            ((value[0] == '"' && value[^1] == '"') || (value[0] == '\'' && value[^1] == '\'')))
        {
            return value[1..^1].Replace("\\n", "\n").Replace("\\t", "\t").Replace("\\\"", "\"");
        }

        return value;
    }

    private static string Quote(string value) =>
        "\"" + value.Replace("\"", "\\\"").Replace("\n", "\\n") + "\"";

    private static void AppendParameter<T>(System.Text.StringBuilder sb, string key, T? value)
        where T : struct
    {
        if (value.HasValue)
        {
            sb.Append("PARAMETER ").Append(key).Append(' ')
              .AppendLine(Convert.ToString(value.Value, CultureInfo.InvariantCulture));
        }
    }

    private static void AppendBlock(System.Text.StringBuilder sb, string instruction, string? value)
    {
        if (!string.IsNullOrEmpty(value))
        {
            sb.Append(instruction).Append(' ').Append(TripleQuote)
              .Append(value).AppendLine(TripleQuote);
        }
    }

    private sealed class GenerationOptionsBuilder
    {
        public float? Temperature, TopP, MinP, RepeatPenalty, PresencePenalty, FrequencyPenalty;
        public int? TopK, RepeatLastN, MaxTokens;
        public uint? Seed;

        public GenerationOptions Build(List<string> stops) => new()
        {
            Temperature = Temperature,
            TopP = TopP,
            TopK = TopK,
            MinP = MinP,
            RepeatPenalty = RepeatPenalty,
            RepeatLastN = RepeatLastN,
            PresencePenalty = PresencePenalty,
            FrequencyPenalty = FrequencyPenalty,
            MaxTokens = MaxTokens,
            Seed = Seed,
            StopSequences = stops
        };
    }

    private sealed class LoadOptionsBuilder
    {
        public int? ContextSize, GpuLayerCount, ThreadCount, BatchSize;
        public bool UseMemoryMap = true;
        public bool UseMemoryLock;
        public DeviceKind Device = DeviceKind.Auto;
        public IReadOnlyList<float> TensorSplit = [];

        public ModelLoadOptions Build(bool embedding) => new()
        {
            ContextSize = ContextSize,
            GpuLayerCount = GpuLayerCount,
            ThreadCount = ThreadCount,
            BatchSize = BatchSize,
            UseMemoryMap = UseMemoryMap,
            UseMemoryLock = UseMemoryLock,
            Device = Device,
            TensorSplit = TensorSplit,
            EmbeddingMode = embedding
        };
    }
}
