using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using LLama;
using LLama.Common;
using LLama.Native;
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

    private readonly ToolDialect _toolDialect;
    private readonly MtmdWeights? _projector;

    private LLamaEmbedder? _embedder;
    private bool _hasChatTemplate;
    private bool _templateProbed;

    public LlamaSharpSession(
        ModelDescriptor model,
        LLamaWeights weights,
        MtmdWeights? projector,
        ModelParams parameters,
        DeviceKind device,
        ILogger<LlamaSharpSession> logger)
    {
        Model = model;
        Device = device;
        _weights = weights;
        _projector = projector;
        _parameters = parameters;
        _logger = logger;

        // Decided once at load: the template cannot change under a loaded model, and the dialect
        // has to be the same for the prompt that asks for a call and the filter that reads it.
        _toolDialect = ToolDialect.Detect(model.ChatTemplate);

        if (_toolDialect != ToolDialect.Hermes)
        {
            _logger.LogInformation(
                "{Model} uses the {Dialect} tool-call convention.", model.Id, _toolDialect.Name);
        }

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

        // Disposed at the end of the generation: a vision request runs on a context of its own,
        // and holding it past the request would keep its KV cache allocated for nothing.
        VisionScope? vision = null;

        try
        {
            var images = CollectImages(request);
            var useVision = images.Count > 0 && SupportsVision;

            if (images.Count > 0 && !useVision)
            {
                _logger.LogWarning(
                    "{Model} was sent {Count} image(s) but has no multimodal projector; they will be described rather than seen.",
                    Model.Id, images.Count);
            }

            var messages = ChatPrompt.Prepare(
                request,
                _toolDialect,
                useVision ? NativeApi.MtmdDefaultMarker() : null);

            var (prompt, stopSequences) = RenderPrompt(messages, request.Options);

            var inferenceParams = BuildInferenceParams(request.Options, stopSequences);
            var promptTokens = CountTokens(prompt);
            var filter = new ToolCallStreamFilter(_toolDialect);
            var completionText = new StringBuilder();
            var finishReason = FinishReason.Stop;
            var generatedTokens = 0;

            if (useVision)
            {
                vision = CreateVisionScope(images);
            }

            var tokens = vision is not null
                ? vision.Executor.InferAsync(prompt, inferenceParams, cancellationToken)
                : _executor.InferAsync(prompt, inferenceParams, cancellationToken);

            _logger.LogDebug(
                "Generating for {Model}: {PromptTokens} prompt tokens, max {MaxTokens}, images {Images}",
                Model.Id, promptTokens, inferenceParams.MaxTokens, images.Count);

            await foreach (var token in tokens.ConfigureAwait(false))
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
            vision?.Dispose();
            _generationLock.Release();
        }
    }

    /// <summary>Whether this session can actually see images.</summary>
    public bool SupportsVision => _projector is { SupportsVision: true };

    private static IReadOnlyList<ContentPart.Image> CollectImages(ChatRequest request) =>
        [.. request.Messages.SelectMany(static m => m.Content).OfType<ContentPart.Image>()];

    /// <summary>
    /// Builds the throwaway context and executor a vision request runs on.
    /// </summary>
    /// <remarks>
    /// The stateless executor the text path uses cannot carry a projector — LLamaSharp exposes
    /// multimodal input only on the stateful executors — so a vision request gets its own context
    /// and an <see cref="InteractiveExecutor"/> over it. Building one per request rather than
    /// keeping it for the session is deliberate: the stateful executor retains the KV cache
    /// between calls, and a second request against a warm one would be decoded on top of the
    /// first conversation rather than the prompt actually supplied.
    /// </remarks>
    private VisionScope CreateVisionScope(IReadOnlyList<ContentPart.Image> images)
    {
        var context = _weights.CreateContext(_parameters, _logger);

        try
        {
            var executor = new InteractiveExecutor(context, _projector!, _logger);

            // Order matters: the nth marker in the prompt takes the nth embed, so the images are
            // added in the order they appeared in the conversation.
            foreach (var image in images)
            {
                executor.Embeds.Add(_projector!.LoadMedia(image.Data.Span));
            }

            return new VisionScope(context, executor, _projector!);
        }
        catch
        {
            context.Dispose();
            throw;
        }
    }

    /// <summary>Owns the per-request context and the media loaded into the projector.</summary>
    private sealed class VisionScope(
        LLamaContext context,
        InteractiveExecutor executor,
        MtmdWeights projector) : IDisposable
    {
        public InteractiveExecutor Executor { get; } = executor;

        public void Dispose()
        {
            // The projector holds the decoded bitmaps until told otherwise; leaving them would
            // leak native memory for every image ever sent to this model.
            projector.ClearMedia();
            Executor.Embeds.Clear();
            context.Dispose();
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

        // Before the weights: the projector was built against them and holds a reference into
        // the loaded model.
        _projector?.Dispose();
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
