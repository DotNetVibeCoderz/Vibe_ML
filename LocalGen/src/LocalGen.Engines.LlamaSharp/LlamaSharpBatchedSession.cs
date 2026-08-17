using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Channels;
using LLama;
using LLama.Batched;
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
/// A loaded GGUF model that decodes several requests at once against a single context.
/// </summary>
/// <remarks>
/// The serialised session runs one sequence at a time because that is what a llama.cpp context
/// natively does. llama.cpp can also decode a batch spanning several sequences, which is what
/// this session uses: every request becomes a <see cref="Conversation"/> holding its own slice of
/// the KV cache, and one decode step advances all of them together. Where the serialised path
/// spends most of its time waiting on memory bandwidth for one token, this amortises that read
/// across every request in flight — the second concurrent request is very nearly free.
///
/// The cost is that the context window is now shared. Four conversations against an 8,192-token
/// context have about 2,048 tokens each, so batching is opt-in rather than the default.
///
/// Everything touching the executor happens on one pump loop. <c>BatchedExecutor</c> and its
/// conversations are not thread-safe, and the whole point of this class is that many threads want
/// to use them at once; requests are therefore posted to the pump and read back over channels.
/// </remarks>
internal sealed class LlamaSharpBatchedSession : IModelSession
{
    private readonly LLamaWeights _weights;
    private readonly ModelParams _parameters;
    private readonly BatchedExecutor _executor;
    private readonly ToolDialect _toolDialect;
    private readonly ILogger _logger;
    private readonly int _maxSequences;

    private readonly Channel<Slot> _incoming = Channel.CreateUnbounded<Slot>(
        new UnboundedChannelOptions { SingleReader = true });

    private readonly CancellationTokenSource _shutdown = new();
    private readonly Task _pump;

    private LLamaEmbedder? _embedder;
    private readonly SemaphoreSlim _embedLock = new(1, 1);
    private bool _hasChatTemplate;
    private bool _templateProbed;

    public LlamaSharpBatchedSession(
        ModelDescriptor model,
        LLamaWeights weights,
        ModelParams parameters,
        DeviceKind device,
        int maxSequences,
        ILogger logger)
    {
        Model = model;
        Device = device;
        _weights = weights;
        _parameters = parameters;
        _logger = logger;
        _maxSequences = Math.Max(1, maxSequences);
        _toolDialect = ToolDialect.Detect(model.ChatTemplate);

        _executor = new BatchedExecutor(weights, parameters);
        _pump = Task.Run(PumpAsync);

        _logger.LogInformation(
            "{Model} is serving batched, up to {Max} concurrent sequences over a {Context}-token context.",
            model.Id, _maxSequences, ContextSize);
    }

    public ModelDescriptor Model { get; }

    public EngineKind Engine => EngineKind.LlamaSharp;

    public DeviceKind Device { get; }

    public int ContextSize => (int)(_parameters.ContextSize ?? (uint)_weights.ContextSize);

    public async IAsyncEnumerable<ChatStreamChunk> StreamAsync(
        ChatRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var messages = ChatPrompt.Prepare(request, _toolDialect);
        var (prompt, stopSequences) = RenderPrompt(messages, request.Options);

        var slot = new Slot(
            prompt,
            BuildPipeline(request.Options),
            request.Options.MaxTokens ?? int.MaxValue,
            stopSequences,
            CountTokens(prompt),
            _weights,
            cancellationToken);

        if (!_incoming.Writer.TryWrite(slot))
        {
            throw new LocalGenException($"Model '{Model.Id}' is shutting down and cannot accept requests.");
        }

        var filter = new ToolCallStreamFilter(_toolDialect);
        var generated = 0;

        try
        {
            await foreach (var piece in slot.Output.Reader
                .ReadAllAsync(cancellationToken)
                .ConfigureAwait(false))
            {
                generated++;
                var visible = filter.Push(piece);

                if (visible.Length > 0)
                {
                    yield return new ChatStreamChunk { Delta = visible };
                }
            }
        }
        finally
        {
            // Tells the pump to drop this conversation even if the caller walked away mid-stream.
            slot.Abandon();
        }

        // Surfaces a decode failure as an exception rather than a silently short answer.
        slot.ThrowIfFaulted();

        var tail = filter.Flush();
        if (tail.Length > 0)
        {
            yield return new ChatStreamChunk { Delta = tail };
        }

        var finishReason = FinishReason.Stop;

        if (filter.HasCalls)
        {
            finishReason = FinishReason.ToolCalls;
            yield return new ChatStreamChunk { ToolCalls = filter.Calls };
        }
        else if (slot.HitTokenLimit)
        {
            finishReason = FinishReason.Length;
        }

        yield return new ChatStreamChunk
        {
            FinishReason = finishReason,
            Usage = new TokenUsage
            {
                PromptTokens = slot.PromptTokens,
                CompletionTokens = generated
            }
        };
    }

    /// <summary>
    /// The single thread that owns the executor: admits queued requests, runs one decode step
    /// across every live conversation, then samples one token for each.
    /// </summary>
    private async Task PumpAsync()
    {
        var active = new List<Slot>();
        var token = _shutdown.Token;

        try
        {
            while (!token.IsCancellationRequested)
            {
                Admit(active);

                if (active.Count == 0)
                {
                    // Nothing to decode. Waiting on the queue rather than spinning keeps an idle
                    // model off the CPU entirely.
                    if (!await _incoming.Reader.WaitToReadAsync(token).ConfigureAwait(false))
                    {
                        break;
                    }

                    continue;
                }

                var result = await _executor.Infer(token).ConfigureAwait(false);

                if (result != DecodeResult.Ok)
                {
                    HandleDecodeFailure(active, result);
                    continue;
                }

                SampleAll(active);
            }
        }
        catch (OperationCanceledException)
        {
            // Shutdown.
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "The batched decode loop for {Model} stopped.", Model.Id);
            FaultAll(active, ex);
        }
        finally
        {
            foreach (var slot in active)
            {
                slot.Complete();
                slot.Conversation?.Dispose();
            }
        }
    }

    /// <summary>Starts as many queued requests as there is room for.</summary>
    private void Admit(List<Slot> active)
    {
        while (active.Count < _maxSequences && _incoming.Reader.TryRead(out var slot))
        {
            if (slot.IsAbandoned)
            {
                slot.Complete();
                continue;
            }

            try
            {
                var conversation = _executor.Create();
                conversation.Prompt(slot.Prompt, addBos: true, special: true);

                slot.Conversation = conversation;
                active.Add(slot);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Could not start a batched conversation for {Model}.", Model.Id);
                slot.Fault(ex);
            }
        }
    }

    /// <summary>Takes one token for every conversation the last decode produced logits for.</summary>
    private void SampleAll(List<Slot> active)
    {
        for (var i = active.Count - 1; i >= 0; i--)
        {
            var slot = active[i];

            if (slot.IsAbandoned)
            {
                Retire(active, i, slot);
                continue;
            }

            if (slot.Conversation is not { RequiresSampling: true } conversation)
            {
                continue;
            }

            LLamaToken sampled;

            try
            {
                sampled = conversation.Sample(slot.Pipeline);
            }
            catch (Exception ex)
            {
                slot.Fault(ex);
                Retire(active, i, slot);
                continue;
            }

            if (IsEndOfGeneration(sampled))
            {
                Retire(active, i, slot);
                continue;
            }

            slot.Decoder.Add(sampled);
            var text = slot.Decoder.Read();

            if (text.Length > 0 && !slot.Emit(text))
            {
                // The reader is gone.
                Retire(active, i, slot);
                continue;
            }

            if (slot.ShouldStop(out var hitLimit))
            {
                slot.HitTokenLimit = hitLimit;
                Retire(active, i, slot);
                continue;
            }

            // Feeds the sampled token back so the next decode step continues this sequence.
            conversation.Prompt(sampled);
        }
    }

    /// <summary>
    /// Deals with a decode that could not run. <see cref="DecodeResult.NoKvSlot"/> means the
    /// shared context is full, which is the one failure batching introduces that serialising did
    /// not have — the newest conversation is dropped so the older ones can finish rather than
    /// every request stalling together.
    /// </summary>
    private void HandleDecodeFailure(List<Slot> active, DecodeResult result)
    {
        if (result == DecodeResult.NoKvSlot && active.Count > 1)
        {
            var victim = active[^1];

            _logger.LogWarning(
                "{Model} ran out of KV cache with {Count} sequences batched; the newest was dropped.",
                Model.Id, active.Count);

            victim.Fault(new LocalGenException(
                $"'{Model.Id}' ran out of context with {active.Count} requests batched together. " +
                "Lower Engine.MaxBatchedSequences or raise Engine.ContextSize."));

            Retire(active, active.Count - 1, victim);
            return;
        }

        var error = new LocalGenException($"Decoding failed for '{Model.Id}': {result}.");
        _logger.LogError("Batched decode for {Model} returned {Result}.", Model.Id, result);
        FaultAll(active, error);
        active.Clear();
    }

    private static void Retire(List<Slot> active, int index, Slot slot)
    {
        active.RemoveAt(index);
        slot.Complete();
        slot.Conversation?.Dispose();
        slot.Conversation = null;
    }

    private static void FaultAll(List<Slot> active, Exception error)
    {
        foreach (var slot in active)
        {
            slot.Fault(error);
            slot.Conversation?.Dispose();
            slot.Conversation = null;
        }
    }

    private bool IsEndOfGeneration(LLamaToken token)
    {
        var vocab = _weights.Vocab;
        return token == vocab.EOS || token == vocab.EOT;
    }

    public async ValueTask<EmbeddingResponse> EmbedAsync(
        EmbeddingRequest request,
        CancellationToken cancellationToken = default)
    {
        // Embedding needs a pooled context, which is a different shape from the batched generation
        // context, so it runs on a dedicated embedder rather than through the pump.
        var embedder = GetEmbedder();
        var vectors = new List<ReadOnlyMemory<float>>(request.Inputs.Count);
        var tokens = 0;

        await _embedLock.WaitAsync(cancellationToken).ConfigureAwait(false);
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

                vectors.Add(result[0]);
                tokens += CountTokens(input);
            }
        }
        finally
        {
            _embedLock.Release();
        }

        return new EmbeddingResponse
        {
            Model = Model.Id,
            Embeddings = vectors,
            Usage = new TokenUsage { PromptTokens = tokens }
        };
    }

    private LLamaEmbedder GetEmbedder()
    {
        if (_embedder is not null)
        {
            return _embedder;
        }

        var embeddingParams = _parameters with
        {
            Embeddings = true,
            PoolingType = LLamaPoolingType.Mean
        };

        return _embedder = new LLamaEmbedder(_weights, embeddingParams, _logger);
    }

    public int CountTokens(string text) =>
        string.IsNullOrEmpty(text)
            ? 0
            : _weights.Tokenize(text, add_bos: false, special: true, Encoding.UTF8).Length;

    /// <summary>Renders with the model's own template when it has one, else ChatML.</summary>
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

    private static ISamplingPipeline BuildPipeline(GenerationOptions options)
    {
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

        return pipeline;
    }

    public async ValueTask DisposeAsync()
    {
        await _shutdown.CancelAsync().ConfigureAwait(false);
        _incoming.Writer.TryComplete();

        try
        {
            await _pump.ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
        }

        _embedder?.Dispose();

        // After the pump has stopped: the executor owns the context every conversation decodes
        // against, and freeing it under a running decode loop takes the process down with it.
        _executor.Dispose();
        _weights.Dispose();
        _embedLock.Dispose();
        _shutdown.Dispose();
    }

    /// <summary>One in-flight request, and everything the pump needs to advance it.</summary>
    private sealed class Slot(
        string prompt,
        ISamplingPipeline pipeline,
        int maxTokens,
        IReadOnlyList<string> stopSequences,
        int promptTokens,
        LLamaWeights weights,
        CancellationToken cancellationToken)
    {
        private readonly StringBuilder _text = new();
        private int _generated;
        private Exception? _fault;
        private volatile bool _abandoned;

        public string Prompt { get; } = prompt;

        public ISamplingPipeline Pipeline { get; } = pipeline;

        public int PromptTokens { get; } = promptTokens;

        /// <summary>
        /// Per-request, because a multi-byte character can straddle two tokens: the decoder holds
        /// the incomplete bytes until the rest arrives, and sharing one across interleaved
        /// sequences would splice fragments of different replies into the same character.
        /// </summary>
        public StreamingTokenDecoder Decoder { get; } = new(Encoding.UTF8, weights);

        public Channel<string> Output { get; } = Channel.CreateUnbounded<string>(
            new UnboundedChannelOptions { SingleWriter = true });

        public Conversation? Conversation { get; set; }

        public bool HitTokenLimit { get; set; }

        public bool IsAbandoned => _abandoned || cancellationToken.IsCancellationRequested;

        public void Abandon() => _abandoned = true;

        public bool Emit(string piece)
        {
            _text.Append(piece);
            _generated++;
            return Output.Writer.TryWrite(piece);
        }

        /// <summary>Whether this request has finished on a stop string or its token budget.</summary>
        public bool ShouldStop(out bool hitTokenLimit)
        {
            hitTokenLimit = _generated >= maxTokens;

            if (hitTokenLimit)
            {
                return true;
            }

            foreach (var stop in stopSequences)
            {
                if (stop.Length > 0 && EndsWith(stop))
                {
                    return true;
                }
            }

            return false;
        }

        private bool EndsWith(string value)
        {
            if (_text.Length < value.Length)
            {
                return false;
            }

            var start = _text.Length - value.Length;

            for (var i = 0; i < value.Length; i++)
            {
                if (_text[start + i] != value[i])
                {
                    return false;
                }
            }

            return true;
        }

        public void Fault(Exception error)
        {
            _fault ??= error;
            Complete();
        }

        public void Complete() => Output.Writer.TryComplete();

        public void ThrowIfFaulted()
        {
            if (_fault is not null)
            {
                throw _fault;
            }
        }
    }
}
