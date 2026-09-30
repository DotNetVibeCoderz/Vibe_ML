using System.Runtime.CompilerServices;
using System.Text.Json;
using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviTokenizers;
using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviTransformers;

/// <summary>How a <see cref="CausalLanguageModel"/> picks each next token.</summary>
/// <param name="MaxNewTokens">The most tokens to add.</param>
/// <param name="Sample">
/// Draw each token from the distribution rather than take the most likely one. Off by default, which
/// is greedy decoding - deterministic, and token for token what transformers' <c>generate</c> gives.
/// </param>
/// <param name="Temperature">Divides the logits before sampling: below 1 sharpens, above 1 flattens.</param>
/// <param name="TopK">Sample only among the K most likely tokens; 0 for no limit. transformers' default is 50.</param>
/// <param name="TopP">Sample only among the smallest set of tokens whose probabilities reach P; 1 for no limit.</param>
/// <param name="RepetitionPenalty">
/// Above 1, makes every token already in the text less likely: a positive logit is divided by it and
/// a negative one multiplied, as transformers does. Applies to greedy decoding too.
/// </param>
/// <param name="Seed">Seeds the sampler, so a sampled run can be repeated. It does not reproduce torch's draws.</param>
/// <param name="StopAtEndToken">Stop when the model produces its end-of-text token.</param>
public sealed record GenerationSettings(
    int MaxNewTokens = 50,
    bool Sample = false,
    double Temperature = 1.0,
    int TopK = 50,
    double TopP = 1.0,
    double RepetitionPenalty = 1.0,
    int? Seed = null,
    bool StopAtEndToken = true)
{
    /// <summary>Greedy decoding of up to 50 tokens.</summary>
    public static GenerationSettings Greedy { get; } = new();
}

/// <summary>A GPT-2 checkpoint's <c>config.json</c>.</summary>
public sealed record CausalLanguageModelConfig
{
    /// <summary>The architecture family.</summary>
    public required string ModelType { get; init; }

    /// <summary>Token vocabulary size.</summary>
    public required int VocabularySize { get; init; }

    /// <summary>How many tokens the model can attend over, prompt and continuation together.</summary>
    public required int MaxPositions { get; init; }

    /// <summary>Width of every hidden state.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>How many blocks.</summary>
    public required int Layers { get; init; }

    /// <summary>Attention heads per block.</summary>
    public required int Heads { get; init; }

    /// <summary>Width of the feed-forward layer.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>The feed-forward activation; GPT-2's is <c>gelu_new</c>, the tanh approximation.</summary>
    public required string Activation { get; init; }

    /// <summary>Epsilon inside every layer norm.</summary>
    public required double LayerNormEpsilon { get; init; }

    /// <summary>The end-of-text token, or <c>null</c> when the config names none.</summary>
    public int? EndTokenId { get; init; }

    /// <summary>The model families this runs.</summary>
    public static IReadOnlySet<string> Supported { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "gpt2" };

    /// <summary>Reads a <c>config.json</c>.</summary>
    /// <exception cref="NotSupportedException">It describes a model this does not run.</exception>
    public static CausalLanguageModelConfig Load(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;

        var type = root.TryGetProperty("model_type", out var t) ? t.GetString() ?? "" : "";
        if (!Supported.Contains(type))
        {
            throw new NotSupportedException(
                $"This is a '{type}' model. CausalLanguageModel runs GPT-2 and its fine-tunes and distillations "
                + "(model_type gpt2: gpt2, distilgpt2, gpt2-medium and the like). Rotary-position decoders such as "
                + "Llama, Mistral or GPT-NeoX have a different block; export them to ONNX and use GraviOptimum.");
        }

        if (root.TryGetProperty("scale_attn_by_inverse_layer_idx", out var inverse) && inverse.GetBoolean())
        {
            throw new NotSupportedException("scale_attn_by_inverse_layer_idx is set; this runs GPT-2's standard attention scaling only.");
        }

        var hidden = Int(root, "n_embd");
        return new CausalLanguageModelConfig
        {
            ModelType = type,
            VocabularySize = Int(root, "vocab_size"),
            MaxPositions = Int(root, "n_positions"),
            HiddenSize = hidden,
            Layers = Int(root, "n_layer"),
            Heads = Int(root, "n_head"),
            IntermediateSize = root.TryGetProperty("n_inner", out var inner) && inner.ValueKind == JsonValueKind.Number ? inner.GetInt32() : 4 * hidden,
            Activation = root.TryGetProperty("activation_function", out var act) ? act.GetString() ?? "gelu_new" : "gelu_new",
            LayerNormEpsilon = root.TryGetProperty("layer_norm_epsilon", out var eps) ? eps.GetDouble() : 1e-5,
            EndTokenId = root.TryGetProperty("eos_token_id", out var eos) && eos.ValueKind == JsonValueKind.Number ? eos.GetInt32() : null,
        };
    }

    private static int Int(JsonElement root, string name)
        => root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number
            ? value.GetInt32()
            : throw new InvalidDataException($"The config has no '{name}'; guessing it would build a model of the wrong shape.");
}

/// <summary>
/// A decoder-only language model that continues text: GPT-2 and its descendants.
/// </summary>
/// <remarks>
/// <para>
/// The blueprint's GPT. The block is pre-norm, like ViT's, and attends <b>causally</b>, like CLIP's
/// text tower: each position sees only itself and the positions before it, which is what lets the
/// model be trained to predict the next token and then asked to.
/// </para>
/// <para>
/// Generation keeps a <b>key/value cache</b>. A token's keys and values never change once computed,
/// because nothing after it can affect them, so each new token costs one row through the network
/// rather than the whole text again - linear rather than quadratic in the length.
/// </para>
/// <para>
/// GPT-2 stores its projections as <c>Conv1D</c>, which is a linear layer with the matrix kept
/// <c>[inputs, outputs]</c> - the transpose of every other layer on the Hub. They are turned the
/// right way round on load; a model that skipped it would run and produce fluent-looking noise. The
/// output layer is the token embedding table itself.
/// </para>
/// </remarks>
public sealed class CausalLanguageModel : IDisposable
{
    private readonly float[] _tokenTable;       // [vocab, hidden], also the output layer's weights
    private readonly double[] _positions;       // [maxPositions, hidden]
    private readonly DecoderBlock[] _blocks;
    private readonly Norm _finalNorm;
    private readonly Linear _output;            // the token table as a linear layer, no bias
    private bool _disposed;

    private CausalLanguageModel(string id, CausalLanguageModelConfig config, HfTokenizer? tokenizer, WeightStore weights)
    {
        Id = id;
        Config = config;
        Tokenizer = tokenizer;

        var prefix = weights.Contains("transformer.wte.weight") ? "transformer." : "";
        Activation.For(config.Activation);

        var table = weights.Read($"{prefix}wte.weight");
        _tokenTable = new float[table.Size];
        for (var i = 0; i < _tokenTable.Length; i++) _tokenTable[i] = (float)table.At(i);

        // A checkpoint may carry its own output layer; GPT-2's is tied to the embeddings.
        _output = weights.TryRead("lm_head.weight", out var head) ? Linear.From(head, null) : Linear.From(table, null);

        _positions = weights.Read($"{prefix}wpe.weight").AsContiguous().ToArray();
        _blocks = [.. Enumerable.Range(0, config.Layers).Select(i => DecoderBlock.Load(weights, $"{prefix}h.{i}", config))];
        _finalNorm = Norm.Load(weights, $"{prefix}ln_f", config.HiddenSize, config.LayerNormEpsilon);
    }

    /// <summary>The repository or directory this came from.</summary>
    public string Id { get; }

    /// <summary>The configuration.</summary>
    public CausalLanguageModelConfig Config { get; }

    /// <summary>The tokenizer, or <c>null</c> for a directory that holds none.</summary>
    public HfTokenizer? Tokenizer { get; }

    /// <summary>Downloads a GPT-2 checkpoint from the Hub and loads it.</summary>
    /// <param name="repoId">A model id such as <c>gpt2</c> or <c>distilgpt2</c>.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    public static CausalLanguageModel Load(string repoId, string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var config = CausalLanguageModelConfig.Load(Hub.DownloadFile(repoId, "config.json", revision));
        var tokenizer = HfTokenizer.FromPretrained(repoId, revision);

        using var weights = WeightStore.FromPretrained(repoId, revision);
        return new CausalLanguageModel(repoId, config, tokenizer, weights);
    }

    /// <summary>Loads a GPT-2 checkpoint from a directory.</summary>
    /// <param name="directory">A folder holding <c>config.json</c>, the weights and, optionally, the tokenizer.</param>
    public static CausalLanguageModel Open(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);

        var config = CausalLanguageModelConfig.Load(Path.Combine(directory, "config.json"));
        var tokenizer = File.Exists(Path.Combine(directory, "tokenizer.json")) || File.Exists(Path.Combine(directory, "vocab.json"))
            ? HfTokenizer.FromDirectory(directory)
            : null;

        using var weights = WeightStore.Open(directory);
        return new CausalLanguageModel(directory, config, tokenizer, weights);
    }

    // ------------------------------------------------------------------ text

    /// <summary>Continues a prompt.</summary>
    /// <param name="prompt">The text to continue.</param>
    /// <param name="settings">How to choose tokens; greedy by default.</param>
    /// <returns>The continuation only, without the prompt.</returns>
    public string Generate(string prompt, GenerationSettings? settings = null)
    {
        var tokenizer = RequireTokenizer();
        var ids = tokenizer.Encode(prompt, addSpecialTokens: false).Ids;

        return tokenizer.Decode([.. GenerateIds(ids, settings)]);
    }

    /// <summary>Continues a prompt, yielding the text as each token arrives.</summary>
    /// <remarks>
    /// GPT-2's byte-level tokens can split a character across two of them, so each piece is what the
    /// whole continuation decodes to minus what was already yielded, and a piece that ends inside a
    /// character waits for the next token.
    /// </remarks>
    public IEnumerable<string> Stream(string prompt, GenerationSettings? settings = null)
    {
        var tokenizer = RequireTokenizer();
        var ids = tokenizer.Encode(prompt, addSpecialTokens: false).Ids;

        var generated = new List<int>();
        var shown = 0;
        foreach (var id in GenerateIds(ids, settings))
        {
            generated.Add(id);
            var text = tokenizer.Decode(generated);
            if (text.Length > shown && !text.EndsWith('�'))
            {
                yield return text[shown..];
                shown = text.Length;
            }
        }
    }

    // ------------------------------------------------------------------ tokens

    /// <summary>Generates token ids after <paramref name="prompt"/>, one at a time.</summary>
    /// <param name="prompt">The prompt's token ids; at least one.</param>
    /// <param name="settings">How to choose tokens; greedy by default.</param>
    public IEnumerable<int> GenerateIds(IReadOnlyList<int> prompt, GenerationSettings? settings = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(prompt);
        if (prompt.Count == 0)
        {
            throw new ArgumentException(
                "Generation needs at least one token to continue from. GPT-2 has no start token; "
                + "begin an empty text with its end-of-text token, as transformers does.", nameof(prompt));
        }

        var options = settings ?? GenerationSettings.Greedy;
        if (prompt.Count >= Config.MaxPositions)
        {
            throw new ArgumentException($"The prompt is {prompt.Count} tokens; the model attends over {Config.MaxPositions} in all.", nameof(prompt));
        }

        return Run(prompt, options);
    }

    private IEnumerable<int> Run(IReadOnlyList<int> prompt, GenerationSettings options)
    {
        var cache = new KeyValueCache(Config);
        var seen = new List<int>(prompt);
        var random = options.Seed is { } seed ? new Random(seed) : new Random();

        var logits = LastLogits(Step(prompt, cache), prompt.Count);

        for (var produced = 0; produced < options.MaxNewTokens; produced++)
        {
            var next = Choose(logits, seen, options, random);
            yield return next;

            if (options.StopAtEndToken && next == Config.EndTokenId) yield break;
            if (seen.Count + 1 >= Config.MaxPositions) yield break;   // the context is full

            seen.Add(next);
            logits = LastLogits(Step([next], cache), 1);
        }
    }

    /// <summary>Logits at every position of <paramref name="ids"/>, <c>[tokens, vocab]</c> row-major.</summary>
    /// <remarks>What transformers' forward pass returns as <c>logits</c>; the score of each next token.</remarks>
    public double[] Logits(IReadOnlyList<int> ids)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(ids);
        if (ids.Count == 0 || ids.Count > Config.MaxPositions)
        {
            throw new ArgumentException($"The model takes 1 to {Config.MaxPositions} tokens; got {ids.Count}.", nameof(ids));
        }

        var hidden = Step(ids, new KeyValueCache(Config));
        return _output.Apply(_finalNorm.Apply(hidden, ids.Count), ids.Count);
    }

    /// <summary>Runs new tokens through every block, extending the cache, and returns their hidden states.</summary>
    private double[] Step(IReadOnlyList<int> ids, KeyValueCache cache)
    {
        var width = Config.HiddenSize;
        var start = cache.Length;
        var hidden = new double[ids.Count * width];

        for (var t = 0; t < ids.Count; t++)
        {
            var id = ids[t];
            if ((uint)id >= (uint)Config.VocabularySize) throw new ArgumentOutOfRangeException(nameof(ids), id, "Token id outside the vocabulary.");

            for (var d = 0; d < width; d++) hidden[t * width + d] = _tokenTable[id * width + d] + _positions[(start + t) * width + d];
        }

        for (var layer = 0; layer < _blocks.Length; layer++) hidden = _blocks[layer].Forward(hidden, ids.Count, cache, layer);

        cache.Length += ids.Count;
        return hidden;
    }

    /// <summary>The output layer for the last row only - the one that predicts the next token.</summary>
    private double[] LastLogits(double[] hidden, int rows)
    {
        var width = Config.HiddenSize;
        var last = hidden.AsSpan((rows - 1) * width, width).ToArray();
        return _output.Apply(_finalNorm.Apply(last, 1), 1);
    }

    /// <summary>
    /// The next token, after transformers' logits processors in their order: repetition penalty,
    /// then - when sampling - temperature, top-k and top-p.
    /// </summary>
    internal static int Choose(double[] logits, IReadOnlyList<int> seen, GenerationSettings options, Random random)
    {
        var scores = (double[])logits.Clone();

        if (options.RepetitionPenalty != 1.0)
        {
            foreach (var id in seen.Distinct())
            {
                scores[id] = scores[id] < 0 ? scores[id] * options.RepetitionPenalty : scores[id] / options.RepetitionPenalty;
            }
        }

        if (!options.Sample)
        {
            var best = 0;
            for (var i = 1; i < scores.Length; i++) if (scores[i] > scores[best]) best = i;
            return best;
        }

        if (options.Temperature <= 0) throw new ArgumentOutOfRangeException(nameof(options), "Temperature must be positive; use greedy decoding for the limit.");
        for (var i = 0; i < scores.Length; i++) scores[i] /= options.Temperature;

        var order = Enumerable.Range(0, scores.Length).OrderByDescending(i => scores[i]).ToArray();
        var keep = options.TopK > 0 ? Math.Min(options.TopK, order.Length) : order.Length;

        var largest = scores[order[0]];
        var weights = new double[keep];
        var total = 0.0;
        for (var i = 0; i < keep; i++)
        {
            weights[i] = Math.Exp(scores[order[i]] - largest);
            total += weights[i];
        }

        if (options.TopP < 1.0)
        {
            // The smallest prefix whose probability reaches P; the token that crosses it is kept.
            var cumulative = 0.0;
            var cut = keep;
            for (var i = 0; i < keep; i++)
            {
                cumulative += weights[i] / total;
                if (cumulative >= options.TopP)
                {
                    cut = i + 1;
                    break;
                }
            }

            keep = cut;
            total = weights.Take(keep).Sum();
        }

        var draw = random.NextDouble() * total;
        for (var i = 0; i < keep; i++)
        {
            draw -= weights[i];
            if (draw <= 0) return order[i];
        }

        return order[keep - 1];
    }

    private HfTokenizer RequireTokenizer()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return Tokenizer ?? throw new InvalidOperationException(
            $"'{Id}' has no tokenizer files, so it cannot read text. Add them, or use GenerateIds with token ids.");
    }

    /// <inheritdoc />
    public void Dispose() => _disposed = true;

    /// <inheritdoc />
    public override string ToString()
        => $"{Id} ({Config.ModelType}: {Config.Layers} layers, {Config.HiddenSize} hidden, {Config.MaxPositions} positions)";
}

/// <summary>Every layer's keys and values for the tokens seen so far.</summary>
internal sealed class KeyValueCache
{
    internal KeyValueCache(CausalLanguageModelConfig config)
    {
        var size = config.HiddenSize / config.Heads;
        Keys = new double[config.Layers][][];
        Values = new double[config.Layers][][];

        for (var layer = 0; layer < config.Layers; layer++)
        {
            Keys[layer] = new double[config.Heads][];
            Values[layer] = new double[config.Heads][];
            for (var head = 0; head < config.Heads; head++)
            {
                Keys[layer][head] = new double[config.MaxPositions * size];
                Values[layer][head] = new double[config.MaxPositions * size];
            }
        }
    }

    /// <summary><c>[layer][head][position * headSize + d]</c>.</summary>
    internal double[][][] Keys { get; }

    /// <summary><c>[layer][head][position * headSize + d]</c>.</summary>
    internal double[][][] Values { get; }

    /// <summary>How many positions are filled.</summary>
    internal int Length { get; set; }
}

/// <summary>One GPT-2 block: pre-norm, causal attention over a cache, pre-norm feed-forward.</summary>
internal sealed class DecoderBlock
{
    private readonly Norm _first;
    private readonly Linear _queryKeyValue;   // c_attn, [q | k | v]
    private readonly Linear _projection;      // attn.c_proj
    private readonly Norm _second;
    private readonly Linear _up;              // mlp.c_fc
    private readonly Linear _down;            // mlp.c_proj
    private readonly Func<double, double> _activation;
    private readonly int _heads;

    private DecoderBlock(Norm first, Linear queryKeyValue, Linear projection, Norm second, Linear up, Linear down, Func<double, double> activation, int heads)
    {
        _first = first;
        _queryKeyValue = queryKeyValue;
        _projection = projection;
        _second = second;
        _up = up;
        _down = down;
        _activation = activation;
        _heads = heads;
    }

    internal static DecoderBlock Load(WeightStore weights, string prefix, CausalLanguageModelConfig config)
    {
        var hidden = config.HiddenSize;
        return new DecoderBlock(
            Norm.Load(weights, $"{prefix}.ln_1", hidden, config.LayerNormEpsilon),
            Conv1D(weights, $"{prefix}.attn.c_attn", hidden, 3 * hidden),
            Conv1D(weights, $"{prefix}.attn.c_proj", hidden, hidden),
            Norm.Load(weights, $"{prefix}.ln_2", hidden, config.LayerNormEpsilon),
            Conv1D(weights, $"{prefix}.mlp.c_fc", hidden, config.IntermediateSize),
            Conv1D(weights, $"{prefix}.mlp.c_proj", config.IntermediateSize, hidden),
            Activation.For(config.Activation),
            config.Heads);
    }

    /// <summary>A GPT-2 <c>Conv1D</c>: a linear layer stored <c>[inputs, outputs]</c>, turned to <c>[outputs, inputs]</c>.</summary>
    private static Linear Conv1D(WeightStore weights, string name, int inputs, int outputs)
    {
        var stored = weights.Read($"{name}.weight");
        if (stored.Rank != 2 || stored.Shape[0] != inputs || stored.Shape[1] != outputs)
        {
            throw new InvalidDataException(
                $"'{name}.weight' is [{string.Join(", ", stored.Shape.ToArray())}]; a GPT-2 Conv1D from {inputs} to {outputs} is [{inputs}, {outputs}].");
        }

        var source = stored.AsContiguous().ToArray();
        var turned = new double[source.Length];
        for (var i = 0; i < inputs; i++)
        {
            for (var o = 0; o < outputs; o++) turned[o * inputs + i] = source[i * outputs + o];
        }

        return Linear.From(new NdArray(turned, outputs, inputs), weights.Read($"{name}.bias"));
    }

    internal double[] Forward(double[] hidden, int rows, KeyValueCache cache, int layer)
    {
        var attended = Add(hidden, _projection.Apply(Attend(_first.Apply(hidden, rows), rows, cache, layer), rows));
        return Add(attended, _down.Apply(_up.Apply(_second.Apply(attended, rows), rows, _activation), rows));
    }

    /// <summary>
    /// Appends the new rows' keys and values to the cache, then lets each new row attend over every
    /// cached position up to and including its own.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private double[] Attend(double[] input, int rows, KeyValueCache cache, int layer)
    {
        var hidden = _projection.Outputs;
        var size = hidden / _heads;
        var stride = 3 * hidden;
        var start = cache.Length;
        var scale = 1.0 / Math.Sqrt(size);
        var projected = _queryKeyValue.Apply(input, rows);

        var keys = cache.Keys[layer];
        var values = cache.Values[layer];
        for (var head = 0; head < _heads; head++)
        {
            for (var r = 0; r < rows; r++)
            {
                Array.Copy(projected, r * stride + hidden + head * size, keys[head], (start + r) * size, size);
                Array.Copy(projected, r * stride + 2 * hidden + head * size, values[head], (start + r) * size, size);
            }
        }

        var result = new double[rows * hidden];
        Parallel.For(0, _heads * rows, item =>
        {
            var head = item / rows;
            var r = item % rows;
            var seen = start + r + 1;
            var query = projected.AsSpan(r * stride + head * size, size);
            var k = keys[head];
            var v = values[head];

            var weights = new double[seen];
            Simd.Scores(query, k, seen, size, scale, weights);

            var largest = double.NegativeInfinity;
            for (var j = 0; j < seen; j++) if (weights[j] > largest) largest = weights[j];

            var total = 0.0;
            for (var j = 0; j < seen; j++)
            {
                weights[j] = Math.Exp(weights[j] - largest);
                total += weights[j];
            }

            for (var j = 0; j < seen; j++) weights[j] /= total;
            Simd.Combine(weights, v, seen, size, result.AsSpan(r * hidden + head * size, size));
        });

        return result;
    }

    private static double[] Add(double[] a, double[] b)
    {
        var result = new double[a.Length];
        for (var i = 0; i < a.Length; i++) result[i] = a[i] + b[i];
        return result;
    }
}
