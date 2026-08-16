using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace LocalGen.Kernel;

/// <summary>Something that happened while the agent was working, streamed to the UI.</summary>
public abstract record AgentEvent
{
    /// <summary>Newly generated assistant text.</summary>
    public sealed record TextDelta(string Text) : AgentEvent;

    /// <summary>The model asked to call a tool.</summary>
    public sealed record ToolCallStarted(string CallId, string Tool, string ArgumentsJson) : AgentEvent;

    /// <summary>A tool finished. <paramref name="Error"/> is set when it threw.</summary>
    public sealed record ToolCallCompleted(
        string CallId,
        string Tool,
        string Result,
        TimeSpan Duration,
        string? Error = null) : AgentEvent;

    /// <summary>The agent finished its turn.</summary>
    public sealed record Completed(int Iterations) : AgentEvent;

    /// <summary>The turn failed.</summary>
    public sealed record Failed(string Message) : AgentEvent;
}

public sealed class AgentOptions
{
    /// <summary>
    /// Cap on tool-calling rounds in a single turn. Without it, a model that keeps re-calling the
    /// same tool would loop until the context filled.
    /// </summary>
    public int MaxIterations { get; set; } = 8;

    /// <summary>Truncation applied to a tool result before it is fed back to the model.</summary>
    public int MaxToolResultLength { get; set; } = 8000;
}

/// <summary>
/// Drives a chat turn to completion, invoking kernel functions the model asks for and feeding
/// their results back until it answers in plain text.
/// </summary>
/// <remarks>
/// The loop is written here rather than relying on SK's auto-invocation because LocalGen's
/// backends express tool calls through a prompt convention rather than a native API. Owning the
/// loop also lets the Playground show each call and result as it happens, which is most of the
/// value of a local agent playground.
/// </remarks>
public sealed class LocalGenAgent
{
    private readonly Microsoft.SemanticKernel.Kernel _kernel;
    private readonly IChatCompletionService _chat;
    private readonly AgentOptions _options;
    private readonly ILogger<LocalGenAgent> _logger;

    public LocalGenAgent(
        Microsoft.SemanticKernel.Kernel kernel,
        AgentOptions? options = null,
        ILogger<LocalGenAgent>? logger = null)
    {
        _kernel = kernel;
        _chat = kernel.GetRequiredService<IChatCompletionService>();
        _options = options ?? new AgentOptions();
        _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger<LocalGenAgent>.Instance;
    }

    /// <summary>
    /// Runs one turn, streaming events. <paramref name="history"/> is appended to in place, so the
    /// caller ends up with the full conversation including tool calls and their results.
    /// </summary>
    public async IAsyncEnumerable<AgentEvent> StreamAsync(
        ChatHistory history,
        PromptExecutionSettings? settings = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        settings ??= new LocalGenPromptExecutionSettings
        {
            // Tools are advertised but not auto-invoked: this loop does the invoking.
            FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(autoInvoke: false)
        };

        for (var iteration = 1; iteration <= _options.MaxIterations; iteration++)
        {
            var text = new StringBuilder();
            var calls = new CallAccumulator();
            var failure = default(string);

            await using (var enumerator = _chat
                .GetStreamingChatMessageContentsAsync(history, settings, _kernel, cancellationToken)
                .GetAsyncEnumerator(cancellationToken))
            {
                while (true)
                {
                    StreamingChatMessageContent chunk;

                    try
                    {
                        if (!await enumerator.MoveNextAsync().ConfigureAwait(false))
                        {
                            break;
                        }

                        chunk = enumerator.Current;
                    }
                    catch (OperationCanceledException)
                    {
                        throw;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Agent generation failed on iteration {Iteration}", iteration);
                        failure = ex.Message;
                        break;
                    }

                    if (!string.IsNullOrEmpty(chunk.Content))
                    {
                        text.Append(chunk.Content);
                        yield return new AgentEvent.TextDelta(chunk.Content);
                    }

                    foreach (var item in chunk.Items.OfType<StreamingFunctionCallUpdateContent>())
                    {
                        calls.Add(item);
                    }
                }
            }

            if (failure is not null)
            {
                yield return new AgentEvent.Failed(failure);
                yield break;
            }

            var pending = calls.Build();

            // Plain text answer: the turn is done.
            if (pending.Count == 0)
            {
                history.AddAssistantMessage(text.ToString());
                yield return new AgentEvent.Completed(iteration);
                yield break;
            }

            history.Add(BuildAssistantMessage(text.ToString(), pending));

            foreach (var call in pending)
            {
                yield return new AgentEvent.ToolCallStarted(
                    call.Id ?? string.Empty,
                    QualifiedName(call),
                    JsonSerializer.Serialize(call.Arguments ?? []));

                var (result, error, duration) = await InvokeAsync(call, cancellationToken)
                    .ConfigureAwait(false);

                history.Add(new FunctionResultContent(call, result).ToChatMessage());

                yield return new AgentEvent.ToolCallCompleted(
                    call.Id ?? string.Empty,
                    QualifiedName(call),
                    result,
                    duration,
                    error);
            }
        }

        // The iteration budget ran out with tool calls still pending.
        _logger.LogWarning("Agent stopped after {Max} tool-calling iterations.", _options.MaxIterations);
        yield return new AgentEvent.Failed(
            $"Stopped after {_options.MaxIterations} tool-calling rounds without a final answer.");
    }

    /// <summary>Runs a turn to completion and returns the assistant's final text.</summary>
    public async Task<string> InvokeAsync(
        ChatHistory history,
        PromptExecutionSettings? settings = null,
        CancellationToken cancellationToken = default)
    {
        var text = new StringBuilder();

        await foreach (var evt in StreamAsync(history, settings, cancellationToken).ConfigureAwait(false))
        {
            switch (evt)
            {
                case AgentEvent.TextDelta delta:
                    text.Append(delta.Text);
                    break;

                case AgentEvent.Failed failed:
                    throw new KernelException(failed.Message);
            }
        }

        return text.ToString();
    }

    private async Task<(string Result, string? Error, TimeSpan Duration)> InvokeAsync(
        FunctionCallContent call,
        CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();

        try
        {
            var result = await call.InvokeAsync(_kernel, cancellationToken).ConfigureAwait(false);
            var text = result.Result?.ToString() ?? string.Empty;

            // A runaway tool result would eat the whole context window.
            if (text.Length > _options.MaxToolResultLength)
            {
                text = text[.._options.MaxToolResultLength] +
                       $"\n… [truncated, {text.Length:N0} characters total]";
            }

            return (text, null, stopwatch.Elapsed);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogWarning(ex, "Tool {Tool} failed", QualifiedName(call));

            // The error goes back to the model as the tool result so it can recover or explain.
            return ($"Error: {ex.Message}", ex.Message, stopwatch.Elapsed);
        }
    }

    private static ChatMessageContent BuildAssistantMessage(
        string text,
        IReadOnlyList<FunctionCallContent> calls)
    {
        var items = new ChatMessageContentItemCollection();

        if (!string.IsNullOrWhiteSpace(text))
        {
            items.Add(new TextContent(text));
        }

        foreach (var call in calls)
        {
            items.Add(call);
        }

        return new ChatMessageContent(AuthorRole.Assistant, items);
    }

    private static string QualifiedName(FunctionCallContent call) =>
        string.IsNullOrEmpty(call.PluginName) ? call.FunctionName : $"{call.PluginName}.{call.FunctionName}";

    /// <summary>
    /// Reassembles streamed function-call updates. LocalGen backends emit a call in one piece,
    /// but the OpenAI wire format splits arguments across chunks, so updates are joined by id.
    /// </summary>
    private sealed class CallAccumulator
    {
        private readonly Dictionary<string, (string Name, StringBuilder Arguments)> _calls = new();
        private readonly List<string> _order = [];

        public void Add(StreamingFunctionCallUpdateContent update)
        {
            var id = update.CallId ?? update.Name ?? $"call_{_order.Count}";

            if (!_calls.TryGetValue(id, out var entry))
            {
                entry = (update.Name ?? string.Empty, new StringBuilder());
                _calls[id] = entry;
                _order.Add(id);
            }
            else if (!string.IsNullOrEmpty(update.Name))
            {
                entry = (update.Name, entry.Arguments);
                _calls[id] = entry;
            }

            if (!string.IsNullOrEmpty(update.Arguments))
            {
                entry.Arguments.Append(update.Arguments);
            }
        }

        public IReadOnlyList<FunctionCallContent> Build()
        {
            var results = new List<FunctionCallContent>(_order.Count);

            foreach (var id in _order)
            {
                var (name, arguments) = _calls[id];
                var separator = name.IndexOf('-');

                results.Add(new FunctionCallContent(
                    separator > 0 ? name[(separator + 1)..] : name,
                    separator > 0 ? name[..separator] : null,
                    id,
                    ParseArguments(arguments.ToString())));
            }

            return results;
        }

        private static KernelArguments? ParseArguments(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
            {
                return null;
            }

            try
            {
                var parsed = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json);
                if (parsed is null)
                {
                    return null;
                }

                var arguments = new KernelArguments();
                foreach (var (key, value) in parsed)
                {
                    arguments[key] = value.ValueKind switch
                    {
                        JsonValueKind.String => value.GetString(),
                        JsonValueKind.Number => value.TryGetInt64(out var l) ? l : value.GetDouble(),
                        JsonValueKind.True => true,
                        JsonValueKind.False => false,
                        JsonValueKind.Null => null,
                        _ => value.GetRawText()
                    };
                }

                return arguments;
            }
            catch (JsonException)
            {
                return null;
            }
        }
    }
}
