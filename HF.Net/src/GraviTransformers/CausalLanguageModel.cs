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

/// <summary>
/// A decoder-only language model that continues text: GPT-2, Llama, Mistral, Qwen2, Qwen3 and GPT-NeoX
/// (Pythia), with their fine-tunes and distillations.
/// </summary>
/// <remarks>
/// <para>
/// The blueprint's GPT. Every block is pre-norm and attends <b>causally</b>: each position sees
/// only itself and the positions before it, which is what lets the model be trained to predict the
/// next token and then asked to.
/// </para>
/// <para>
/// Three block families. GPT-2 adds learned position vectors to the tokens. Llama and its
/// descendants rotate each query and key by an angle that grows with position (rotary embeddings),
/// normalize with RMSNorm, gate their feed-forward (<c>down(silu(gate(x)) * up(x))</c>) and may share
/// one key/value head among several query heads (grouped-query attention); Mistral adds a sliding
/// window, Qwen2 query/key/value biases, Qwen3 a norm on each head. GPT-NeoX rotates part of each head
/// and runs attention and feed-forward side by side from the same input.
/// </para>
/// <para>
/// Generation keeps a <b>key/value cache</b>. A token's keys and values never change once computed,
/// because nothing after it can affect them, so each new token costs one row through the network
/// rather than the whole text again - linear rather than quadratic in the length.
/// </para>
/// <para>
/// GPT-2 stores its projections as <c>Conv1D</c>, which is a linear layer with the matrix kept
/// <c>[inputs, outputs]</c> - the transpose of every other layer on the Hub. They are turned the
/// right way round on load; a model that skipped it would run and produce fluent-looking noise.
/// NeoX fuses its query, key and value projections head by head, and they are regrouped on load.
/// </para>
/// </remarks>
public sealed class CausalLanguageModel : IDisposable
{
    private readonly float[] _tokenTable;       // [vocab, hidden]
    private readonly double[]? _positions;      // GPT-2's learned positions, [maxPositions, hidden]
    private readonly IDecoderLayer[] _blocks;
    private readonly Func<double[], int, double[]> _finalNorm;
    private readonly Linear _output;            // the output layer, often the token table itself
    private bool _disposed;

    private CausalLanguageModel(string id, CausalLanguageModelConfig config, HfTokenizer? tokenizer, WeightStore weights)
    {
        Id = id;
        Config = config;
        Tokenizer = tokenizer;
        Activation.For(config.Activation);

        var hidden = config.HiddenSize;
        switch (config.Family)
        {
            case DecoderFamily.Gpt2:
            {
                var prefix = weights.Contains("transformer.wte.weight") ? "transformer." : "";
                _tokenTable = weights.ReadFloats($"{prefix}wte.weight");
                _positions = weights.Read($"{prefix}wpe.weight").AsContiguous().ToArray();
                _blocks = [.. Enumerable.Range(0, config.Layers).Select(i => (IDecoderLayer)Gpt2Block.Load(weights, $"{prefix}h.{i}", config))];
                _finalNorm = Norm.Load(weights, $"{prefix}ln_f", hidden, config.LayerNormEpsilon).Apply;
                _output = OutputLayer(weights, "lm_head.weight", tied: true);
                break;
            }

            case DecoderFamily.Llama:
            {
                var prefix = weights.Contains("model.embed_tokens.weight") ? "model." : "";
                var rotary = new RotaryEmbedding(config);
                _tokenTable = weights.ReadFloats($"{prefix}embed_tokens.weight");
                _blocks = [.. Enumerable.Range(0, config.Layers).Select(i => (IDecoderLayer)LlamaBlock.Load(weights, $"{prefix}layers.{i}", config, rotary))];
                _finalNorm = new RmsNorm(weights.ReadRange($"{prefix}norm.weight", 0, hidden), config.LayerNormEpsilon).Apply;
                _output = OutputLayer(weights, "lm_head.weight", config.TieEmbeddings);
                break;
            }

            default:
            {
                var prefix = weights.Contains("gpt_neox.embed_in.weight") ? "gpt_neox." : "";
                var rotary = new RotaryEmbedding(config);
                _tokenTable = weights.ReadFloats($"{prefix}embed_in.weight");
                _blocks = [.. Enumerable.Range(0, config.Layers).Select(i => (IDecoderLayer)NeoXBlock.Load(weights, $"{prefix}layers.{i}", config, rotary))];
                _finalNorm = Norm.Load(weights, $"{prefix}final_layer_norm", hidden, config.LayerNormEpsilon).Apply;
                _output = OutputLayer(weights, "embed_out.weight", config.TieEmbeddings);
                break;
            }
        }

        if (_tokenTable.Length != (long)config.VocabularySize * hidden)
        {
            throw new InvalidDataException(
                $"The token table holds {_tokenTable.Length} values; {config.VocabularySize} tokens of width {hidden} need "
                + $"{(long)config.VocabularySize * hidden}. The config and the weights disagree.");
        }
    }

    /// <summary>The output layer: the checkpoint's own when it has one, else the token table.</summary>
    private Linear OutputLayer(WeightStore weights, string name, bool tied)
    {
        var hidden = Config.HiddenSize;
        if (weights.Contains(name))
        {
            var outputs = weights.ShapeOf(name)[0];
            return Linear.FromFloats(weights.ReadFloats(name), new double[outputs], hidden, outputs);
        }

        if (!tied)
        {
            throw new InvalidDataException($"The checkpoint has no '{name}' and does not tie its output layer to the embeddings.");
        }

        return Linear.FromFloats(_tokenTable, new double[Config.VocabularySize], hidden, Config.VocabularySize);
    }

    /// <summary>The repository or directory this came from.</summary>
    public string Id { get; }

    /// <summary>The configuration.</summary>
    public CausalLanguageModelConfig Config { get; }

    /// <summary>The tokenizer, or <c>null</c> for a directory that holds none.</summary>
    public HfTokenizer? Tokenizer { get; }

    /// <summary>Downloads a decoder checkpoint from the Hub and loads it.</summary>
    /// <param name="repoId">
    /// A model id such as <c>gpt2</c>, <c>HuggingFaceTB/SmolLM2-135M</c>, <c>Qwen/Qwen2.5-0.5B</c> or
    /// <c>EleutherAI/pythia-160m</c>.
    /// </param>
    /// <param name="revision">A branch, tag or commit.</param>
    public static CausalLanguageModel Load(string repoId, string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var config = CausalLanguageModelConfig.Load(Hub.DownloadFile(repoId, "config.json", revision));
        var tokenizer = HfTokenizer.FromPretrained(repoId, revision);

        using var weights = WeightStore.FromPretrained(repoId, revision);
        return new CausalLanguageModel(repoId, config, tokenizer, weights);
    }

    /// <summary>Loads a decoder checkpoint from a directory.</summary>
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
    /// <remarks>
    /// The prompt is encoded as transformers' <c>tokenizer(prompt)</c> encodes it, special tokens and
    /// all - Llama's begin-of-text token in front, nothing for GPT-2. Chat models expect their chat
    /// template around the prompt; apply it to the text before calling this.
    /// </remarks>
    public string Generate(string prompt, GenerationSettings? settings = null)
    {
        var tokenizer = RequireTokenizer();
        var ids = tokenizer.Encode(prompt).Ids;

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
        var ids = tokenizer.Encode(prompt).Ids;

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
        var cache = NewCache();
        var seen = new List<int>(prompt);
        var ends = Config.EndTokenIds.ToHashSet();
        var random = options.Seed is { } seed ? new Random(seed) : new Random();

        var logits = LastLogits(Step(prompt, cache), prompt.Count);

        for (var produced = 0; produced < options.MaxNewTokens; produced++)
        {
            var next = Choose(logits, seen, options, random);
            yield return next;

            if (options.StopAtEndToken && ends.Contains(next)) yield break;
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

        var hidden = Step(ids, NewCache());
        return _output.Apply(_finalNorm(hidden, ids.Count), ids.Count);
    }

    /// <summary>Runs new tokens through every block, extending the cache, and returns their hidden states.</summary>
    private double[] Step(IReadOnlyList<int> ids, KeyValueCache cache)
    {
        var width = Config.HiddenSize;
        var start = cache.Length;
        var hidden = new double[ids.Count * width];
        cache.Reserve(start + ids.Count);

        for (var t = 0; t < ids.Count; t++)
        {
            var id = ids[t];
            if ((uint)id >= (uint)Config.VocabularySize) throw new ArgumentOutOfRangeException(nameof(ids), id, "Token id outside the vocabulary.");

            for (var d = 0; d < width; d++)
            {
                hidden[t * width + d] = _positions is null
                    ? _tokenTable[(long)id * width + d]
                    : _tokenTable[(long)id * width + d] + _positions[(start + t) * width + d];
            }
        }

        for (var layer = 0; layer < _blocks.Length; layer++) hidden = _blocks[layer].Forward(hidden, ids.Count, cache, layer, start);

        cache.Length += ids.Count;
        return hidden;
    }

    /// <summary>The output layer for the last row only - the one that predicts the next token.</summary>
    private double[] LastLogits(double[] hidden, int rows)
    {
        var width = Config.HiddenSize;
        var last = hidden.AsSpan((rows - 1) * width, width).ToArray();
        return _output.Apply(_finalNorm(last, 1), 1);
    }

    private KeyValueCache NewCache() => new(Config.Layers, Config.KeyValueHeads, Config.HeadSize);

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
        => $"{Id} ({Config.ModelType}: {Config.Layers} layers, {Config.HiddenSize} hidden, {Config.Heads} heads"
            + (Config.KeyValueHeads != Config.Heads ? $" sharing {Config.KeyValueHeads} key/value heads" : "")
            + $", {Config.MaxPositions:N0} positions)";
}
