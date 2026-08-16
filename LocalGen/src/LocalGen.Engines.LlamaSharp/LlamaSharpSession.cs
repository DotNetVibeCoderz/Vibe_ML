using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using LLama;
using LLama.Common;
using LLama.Sampling;
using LocalGen.Core;
using LocalGen.Core.Engines;
using LocalGen.Core.Inference;
using LocalGen.Core.Models;
using LocalGen.Core.Prompting;
using Microsoft.Extensions.Logging;

namespace LocalGen.Engines.LlamaSharp;

/// <summary>
/// A loaded GGUF model. Generation is serialised per session: llama.cpp decodes one sequence at
/// a time against a context, and running several concurrently would corrupt the KV cache and
/// exhaust VRAM. The server therefore queues requests per model rather than parallelising them.
/// </summary>
internal sealed class LlamaSharpSession : IModelSession
{
    private readonly LLamaWeights _weights;
    private readonly ModelParams _parameters;
    private readonly StatelessExecutor _executor;
    private readonly ILogger<LlamaSharpSession> _logger;
    private readonly SemaphoreSlim _generationLock = new(1, 1);

    private LLamaEmbedder? _embedder;
    private bool _hasChatTemplate;
    private bool _templateProbed;

    public LlamaSharpSession(
        ModelDescriptor model,
        LLamaWeights weights,
        ModelParams parameters,
        DeviceKind device,
        ILogger<LlamaSharpSession> logger)
    {
        Model = model;
        Device = device;
        _weights = weights;
        _parameters = parameters;
        _logger = logger;

        _executor = new StatelessExecutor(weights, parameters, logger)
        {
            // LocalGen renders the chat template itself so that tool instructions and document
            // parts are folded in consistently across every backend.
            ApplyTemplate = false
        };
    }

    public ModelDescriptor Model { get; }

    public EngineKind Engine => EngineKind.LlamaSharp;

    public DeviceKind Device { get; }

    public int ContextSize => (int)(_parameters.ContextSize ?? (uint)_weights.ContextSize);

    public async IAsyncEnumerable<ChatStreamChunk> StreamAsync(
        ChatRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await _generationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var messages = ChatPrompt.Prepare(request);
            var (prompt, stopSequences) = RenderPrompt(messages, request.Options);

            var inferenceParams = BuildInferenceParams(request.Options, stopSequences);
            var promptTokens = CountTokens(prompt);
            var filter = new ToolCallStreamFilter();
            var completionText = new StringBuilder();
            var finishReason = FinishReason.Stop;
            var generatedTokens = 0;

            _logger.LogDebug(
                "Generating for {Model}: {PromptTokens} prompt tokens, max {MaxTokens}",
                Model.Id, promptTokens, inferenceParams.MaxTokens);

            await foreach (var token in _executor
                .InferAsync(prompt, inferenceParams, cancellationToken)
                .ConfigureAwait(false))
            {
                generatedTokens++;
                var visible = filter.Push(token);

                if (visible.Length > 0)
                {
                    completionText.Append(visible);
                    yield return new ChatStreamChunk { Delta = visible };
                }
            }

            var tail = filter.Flush();
            if (tail.Length > 0)
            {
                completionText.Append(tail);
                yield return new ChatStreamChunk { Delta = tail };
            }

            if (filter.HasCalls)
            {
                // The finish reason rides on the closing chunk alone. Setting it here as well
                // would put it on the wire twice, which the OpenAI contract does not do.
                finishReason = FinishReason.ToolCalls;
                yield return new ChatStreamChunk { ToolCalls = filter.Calls };
            }
            else if (inferenceParams.MaxTokens > 0 && generatedTokens >= inferenceParams.MaxTokens)
            {
                finishReason = FinishReason.Length;
            }

            yield return new ChatStreamChunk
            {
                FinishReason = finishReason,
                Usage = new TokenUsage
                {
                    PromptTokens = promptTokens,
                    CompletionTokens = generatedTokens
                }
            };
        }
        finally
        {
            _generationLock.Release();
        }
    }

    public async ValueTask<EmbeddingResponse> EmbedAsync(
        EmbeddingRequest request,
        CancellationToken cancellationToken = default)
    {
        var embedder = GetEmbedder();
        var vectors = new List<ReadOnlyMemory<float>>(request.Inputs.Count);
        var tokens = 0;

        await _generationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            foreach (var input in request.Inputs)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var result = await embedder.GetEmbeddings(input, cancellationToken).ConfigureAwait(false);
                if (result.Count == 0)
                {
                    throw new LocalGenException($"Model '{Model.Id}' returned no embedding for the input.");
                }

                // With mean pooling llama.cpp returns a single vector; take the first either way.
                vectors.Add(result[0]);
                tokens += CountTokens(input);
            }
        }
        finally
        {
            _generationLock.Release();
        }

        return new EmbeddingResponse
        {
            Model = Model.Id,
            Embeddings = vectors,
            Usage = new TokenUsage { PromptTokens = tokens }
        };
    }

    public int CountTokens(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return 0;
        }

        return _weights.Tokenize(text, add_bos: false, special: true, Encoding.UTF8).Length;
    }

    /// <summary>
    /// Renders the conversation with the template baked into the GGUF file when there is one,
    /// falling back to ChatML. The fallback also returns stop strings, which the model's own
    /// template does not need because llama.cpp stops on its EOS token.
    /// </summary>
    private (string Prompt, IReadOnlyList<string> StopSequences) RenderPrompt(
        IReadOnlyList<ChatMessage> messages,
        GenerationOptions options)
    {
        if (TryRenderWithModelTemplate(messages, out var templated))
        {
            return (templated, options.StopSequences);
        }

        var stops = options.StopSequences.Count > 0
            ? [.. options.StopSequences, .. ChatPrompt.ChatMlStopSequences]
            : ChatPrompt.ChatMlStopSequences;

        return (ChatPrompt.RenderChatMl(messages), stops);
    }

    private bool TryRenderWithModelTemplate(IReadOnlyList<ChatMessage> messages, out string prompt)
    {
        prompt = string.Empty;

        if (_templateProbed && !_hasChatTemplate)
        {
            return false;
        }

        try
        {
            var template = new LLamaTemplate(_weights) { AddAssistant = true };

            foreach (var message in messages)
            {
                template.Add(message.Role.ToString().ToLowerInvariant(), message.Text);
            }

            prompt = Encoding.UTF8.GetString(template.Apply());
            _hasChatTemplate = true;
            _templateProbed = true;
            return true;
        }
        catch (Exception ex)
        {
            if (!_templateProbed)
            {
                _logger.LogInformation(
                    "{Model} has no usable chat template ({Reason}); falling back to ChatML.",
                    Model.Id, ex.Message);
            }

            _hasChatTemplate = false;
            _templateProbed = true;
            return false;
        }
    }

    private InferenceParams BuildInferenceParams(
        GenerationOptions options,
        IReadOnlyList<string> stopSequences)
    {
        // The sampler's properties are init-only, so unspecified options fall back to a pristine
        // instance rather than to hard-coded constants that could drift from LLamaSharp's defaults.
        var defaults = new DefaultSamplingPipeline();
        var grammar = options.Grammar ?? (options.JsonMode ? GbnfGrammars.Json : null);

        var pipeline = new DefaultSamplingPipeline
        {
            Temperature = options.Temperature ?? defaults.Temperature,
            TopP = options.TopP ?? defaults.TopP,
            TopK = options.TopK ?? defaults.TopK,
            MinP = options.MinP ?? defaults.MinP,
            RepeatPenalty = options.RepeatPenalty ?? defaults.RepeatPenalty,
            PenaltyCount = options.RepeatLastN ?? defaults.PenaltyCount,
            PresencePenalty = options.PresencePenalty ?? defaults.PresencePenalty,
            FrequencyPenalty = options.FrequencyPenalty ?? defaults.FrequencyPenalty,
            Grammar = grammar is not null ? new Grammar(grammar, "root") : defaults.Grammar
        };

        if (options.Seed is { } seed)
        {
            pipeline.Seed = seed;
        }

        return new InferenceParams
        {
            MaxTokens = options.MaxTokens ?? -1,
            AntiPrompts = [.. stopSequences],
            SamplingPipeline = pipeline
        };
    }

    private LLamaEmbedder GetEmbedder()
    {
        if (_embedder is not null)
        {
            return _embedder;
        }

        // Embedding needs a context configured for pooled output, which is a different context
        // shape from generation — hence a dedicated embedder over the same weights.
        var embeddingParams = _parameters with
        {
            Embeddings = true,
            PoolingType = LLama.Native.LLamaPoolingType.Mean
        };

        return _embedder = new LLamaEmbedder(_weights, embeddingParams, _logger);
    }

    public ValueTask DisposeAsync()
    {
        _embedder?.Dispose();
        _weights.Dispose();
        _generationLock.Dispose();
        return ValueTask.CompletedTask;
    }
}

/// <summary>Grammars used to constrain decoding.</summary>
internal static class GbnfGrammars
{
    /// <summary>
    /// Restricts output to a single well-formed JSON value. Used for JSON mode so the model
    /// cannot emit prose around the object.
    /// </summary>
    public const string Json =
        """
        root   ::= object
        value  ::= object | array | string | number | ("true" | "false" | "null") ws
        object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
        array  ::= "[" ws ( value ("," ws value)* )? "]" ws
        string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt/] | "u" [0-9a-fA-F]{4}) )* "\"" ws
        number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
        ws     ::= | " " | "\n" [ \t]{0,20}
        """;
}
