using System.ComponentModel;
using System.Text;
using LocalGen.Core.Inference;
using Spectre.Console;
using Spectre.Console.Cli;

namespace LocalGen.Cli.Commands;

/// <summary>
/// Chats with a model — one-shot when a prompt is given, interactive otherwise.
/// </summary>
/// <remarks>
/// Tokens are written straight to the console rather than through a Spectre live display: a live
/// region re-renders the whole block on every token, which flickers and mangles wrapping for
/// long answers. Plain streaming writes are what a terminal chat should feel like.
/// </remarks>
public sealed class RunCommand : AsyncCommand<RunCommand.Settings>
{
    public sealed class Settings : GlobalSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("Model id or name")]
        public string Model { get; init; } = string.Empty;

        [CommandArgument(1, "[prompt]")]
        [Description("Prompt to send. Omit for an interactive session.")]
        public string? Prompt { get; init; }

        [CommandOption("-s|--system <TEXT>")]
        [Description("System prompt.")]
        public string? System { get; init; }

        [CommandOption("-t|--temperature <VALUE>")]
        [Description("Sampling temperature.")]
        public float? Temperature { get; init; }

        [CommandOption("-n|--max-tokens <COUNT>")]
        [Description("Maximum tokens to generate.")]
        public int? MaxTokens { get; init; }

        [CommandOption("--stats")]
        [Description("Print token counts and throughput after each reply.")]
        public bool Stats { get; init; }
    }

    protected override async Task<int> ExecuteAsync(CommandContext context, Settings settings, CancellationToken cancellationToken)
    {
        await using var bridge = await BridgeFactory.CreateAsync(settings.Host, settings.Verbose, cancellationToken);

        var model = await bridge.GetModelAsync(settings.Model, cancellationToken);
        if (model is null)
        {
            AnsiConsole.MarkupLine($"[red]No model named '{Markup.Escape(settings.Model)}'.[/]");
            AnsiConsole.MarkupLine($"Pull it first: [cyan]localgen pull {Markup.Escape(settings.Model)}[/]");
            return 1;
        }

        var options = new GenerationOptions
        {
            Temperature = settings.Temperature,
            MaxTokens = settings.MaxTokens
        };

        var history = new List<ChatMessage>();

        if (!string.IsNullOrWhiteSpace(settings.System))
        {
            history.Add(ChatMessage.System(settings.System));
        }

        // One-shot mode: send the prompt, print the reply, exit.
        if (!string.IsNullOrWhiteSpace(settings.Prompt))
        {
            history.Add(ChatMessage.User(settings.Prompt));
            await StreamReplyAsync(bridge, model.Id, history, options, settings.Stats, cancellationToken);
            return 0;
        }

        return await RunInteractiveAsync(bridge, model.Id, history, options, settings, cancellationToken);
    }

    private static async Task<int> RunInteractiveAsync(
        ILocalGenBridge bridge,
        string modelId,
        List<ChatMessage> history,
        GenerationOptions options,
        Settings settings,
        CancellationToken hostToken)
    {
        AnsiConsole.Write(new Rule($"[cyan]{Markup.Escape(modelId)}[/]").LeftJustified());
        AnsiConsole.MarkupLine($"[dim]{bridge.Mode}[/]");
        AnsiConsole.MarkupLine("[dim]Commands: /clear reset the conversation · /bye or Ctrl+C to exit[/]");
        AnsiConsole.WriteLine();

        // Ctrl+C cancels the current generation rather than killing the session outright.
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(hostToken);
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            cancellation.Cancel();
        };

        while (true)
        {
            if (cancellation.IsCancellationRequested)
            {
                break;
            }

            AnsiConsole.Markup("[bold green]>[/] ");
            var input = Console.ReadLine();

            if (input is null)
            {
                break;
            }

            input = input.Trim();

            if (input.Length == 0)
            {
                continue;
            }

            switch (input.ToLowerInvariant())
            {
                case "/bye" or "/exit" or "/quit":
                    return 0;

                case "/clear":
                    // The system prompt survives a clear; it is configuration, not conversation.
                    history.RemoveAll(static m => m.Role != ChatRole.System);
                    AnsiConsole.MarkupLine("[dim]Conversation cleared.[/]");
                    continue;

                case "/help":
                    AnsiConsole.MarkupLine("[dim]/clear · /bye · /help[/]");
                    continue;
            }

            history.Add(ChatMessage.User(input));

            using var turn = CancellationTokenSource.CreateLinkedTokenSource(cancellation.Token);

            try
            {
                await StreamReplyAsync(bridge, modelId, history, options, settings.Stats, turn.Token);
            }
            catch (OperationCanceledException)
            {
                AnsiConsole.MarkupLine("\n[yellow]Generation cancelled.[/]");

                // Reset so the next Ctrl+C is treated as a fresh cancellation.
                if (cancellation.IsCancellationRequested)
                {
                    return 0;
                }
            }
        }

        return 0;
    }

    private static async Task StreamReplyAsync(
        ILocalGenBridge bridge,
        string modelId,
        List<ChatMessage> history,
        GenerationOptions options,
        bool showStats,
        CancellationToken cancellationToken = default)
    {
        var request = new ChatRequest
        {
            Model = modelId,
            Messages = [.. history],
            Options = options
        };

        var reply = new StringBuilder();
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var firstToken = TimeSpan.Zero;
        TokenUsage? usage = null;

        await foreach (var chunk in bridge.StreamAsync(request, cancellationToken))
        {
            if (chunk.Delta.Length > 0)
            {
                if (firstToken == TimeSpan.Zero)
                {
                    firstToken = stopwatch.Elapsed;
                }

                reply.Append(chunk.Delta);

                // Written raw: the model's output is not Spectre markup and must not be parsed as it.
                Console.Write(chunk.Delta);
            }

            foreach (var call in chunk.ToolCalls)
            {
                AnsiConsole.MarkupLine(
                    $"\n[magenta]→ tool:[/] {Markup.Escape(call.Name)} {Markup.Escape(call.ArgumentsJson)}");
            }

            if (chunk.Usage is not null)
            {
                usage = chunk.Usage;
            }
        }

        Console.WriteLine();

        history.Add(ChatMessage.Assistant(reply.ToString()));

        if (showStats && usage is not null)
        {
            var generationTime = (stopwatch.Elapsed - firstToken).TotalSeconds;
            var throughput = generationTime > 0.001 ? usage.CompletionTokens / generationTime : 0;

            AnsiConsole.MarkupLine(
                $"[dim]{usage.PromptTokens} prompt + {usage.CompletionTokens} completion tokens · " +
                $"{firstToken.TotalMilliseconds:N0} ms to first token · {throughput:N1} tokens/s[/]");
        }

        AnsiConsole.WriteLine();
    }
}
