using System.ComponentModel;
using System.Diagnostics;
using LocalGen.Core.Inference;
using LocalGen.Core.Models;
using LocalGen.Core.Modelfiles;
using Spectre.Console;
using Spectre.Console.Cli;

namespace LocalGen.Cli.Commands;

/// <summary>Options every command shares.</summary>
public class GlobalSettings : CommandSettings
{
    [CommandOption("--host <URL>")]
    [Description("LocalGen server address. Defaults to $LOCALGEN_HOST or http://127.0.0.1:11434.")]
    public string? Host { get; init; }

    [CommandOption("-v|--verbose")]
    [Description("Show diagnostic logging.")]
    public bool Verbose { get; init; }
}

/// <summary>Lists the models installed locally.</summary>
public sealed class ListCommand : AsyncCommand<ListCommand.Settings>
{
    public sealed class Settings : GlobalSettings
    {
        [CommandOption("--json")]
        [Description("Emit JSON instead of a table.")]
        public bool Json { get; init; }
    }

    protected override async Task<int> ExecuteAsync(CommandContext context, Settings settings, CancellationToken cancellationToken)
    {
        await using var bridge = await BridgeFactory.CreateAsync(settings.Host, settings.Verbose, cancellationToken);
        var models = await bridge.ListModelsAsync(cancellationToken);

        if (settings.Json)
        {
            // Written straight to stdout: AnsiConsole word-wraps at the console width, which
            // inserts line breaks inside string literals and leaves the JSON unparseable — the
            // one thing this flag exists to avoid.
            Console.Out.WriteLine(System.Text.Json.JsonSerializer.Serialize(
                models, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
            return 0;
        }

        if (models.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No models installed.[/]");
            AnsiConsole.MarkupLine("Pull one to get started, for example:");
            AnsiConsole.MarkupLine("  [dim]localgen pull huggingface:bartowski/Qwen2.5-7B-Instruct-GGUF[/]");
            return 0;
        }

        var table = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Grey30)
            .AddColumn("[bold]Model[/]")
            .AddColumn("[bold]Quant[/]")
            .AddColumn("[bold]Size[/]", c => c.RightAligned())
            .AddColumn("[bold]Context[/]", c => c.RightAligned())
            .AddColumn("[bold]Capabilities[/]");

        foreach (var model in models)
        {
            table.AddRow(
                Markup.Escape(model.Id),
                Markup.Escape(model.Quantization.Name),
                FormatSize(model.SizeBytes),
                $"{model.ContextLength:N0}",
                Markup.Escape(string.Join(", ", model.Capabilities)));
        }

        AnsiConsole.Write(table);
        AnsiConsole.MarkupLine($"[dim]{models.Count} model(s) — {bridge.Mode}[/]");
        return 0;
    }

    internal static string FormatSize(long bytes) => bytes switch
    {
        <= 0 => "-",
        < 1024L * 1024 => $"{bytes / 1024.0:N0} KB",
        < 1024L * 1024 * 1024 => $"{bytes / (1024.0 * 1024):N0} MB",
        _ => $"{bytes / (1024.0 * 1024 * 1024):N2} GB"
    };
}

/// <summary>Downloads a model.</summary>
public sealed class PullCommand : AsyncCommand<PullCommand.Settings>
{
    public sealed class Settings : GlobalSettings
    {
        [CommandArgument(0, "<reference>")]
        [Description("Model reference, e.g. huggingface:owner/repo or huggingface:owner/repo/file.gguf")]
        public string Reference { get; init; } = string.Empty;
    }

    protected override async Task<int> ExecuteAsync(CommandContext context, Settings settings, CancellationToken cancellationToken)
    {
        await using var bridge = await BridgeFactory.CreateAsync(settings.Host, settings.Verbose, cancellationToken);

        AnsiConsole.MarkupLine($"Pulling [cyan]{Markup.Escape(settings.Reference)}[/]…");

        ModelDescriptor? model = null;
        Exception? failure = null;

        await AnsiConsole.Progress()
            .Columns(
                new TaskDescriptionColumn(),
                new ProgressBarColumn(),
                new PercentageColumn(),
                new DownloadedColumn(),
                new TransferSpeedColumn(),
                new RemainingTimeColumn())
            .StartAsync(async ctx =>
            {
                var task = ctx.AddTask("[green]Downloading[/]", maxValue: 100);

                var progress = new Progress<DownloadProgress>(update =>
                {
                    if (update.TotalBytes > 0)
                    {
                        // Spectre's byte columns read MaxValue/Value directly, so the task is
                        // scaled to real byte counts once the size is known.
                        task.MaxValue = update.TotalBytes;
                        task.Value = update.BytesDownloaded;
                    }

                    if (!string.IsNullOrEmpty(update.FileName))
                    {
                        task.Description = $"[green]{Markup.Escape(update.FileName)}[/]";
                    }
                });

                try
                {
                    model = await bridge.PullAsync(settings.Reference, progress, cancellationToken);
                    task.Value = task.MaxValue;
                }
                catch (Exception ex)
                {
                    failure = ex;
                }
            });

        if (failure is not null)
        {
            AnsiConsole.MarkupLine($"[red]Pull failed:[/] {Markup.Escape(failure.Message)}");
            return 1;
        }

        AnsiConsole.MarkupLine($"[green]✓[/] Installed [cyan]{Markup.Escape(model!.Id)}[/]");
        AnsiConsole.MarkupLine(
            $"[dim]{ListCommand.FormatSize(model.SizeBytes)} · {model.Quantization.Name} · " +
            $"{model.ContextLength:N0} token context[/]");
        AnsiConsole.MarkupLine($"Run it with: [cyan]localgen run {Markup.Escape(model.Id)}[/]");
        return 0;
    }
}

/// <summary>Removes an installed model.</summary>
public sealed class RemoveCommand : AsyncCommand<RemoveCommand.Settings>
{
    public sealed class Settings : GlobalSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("Model id or name")]
        public string Model { get; init; } = string.Empty;

        [CommandOption("-y|--yes")]
        [Description("Do not ask for confirmation.")]
        public bool Yes { get; init; }
    }

    protected override async Task<int> ExecuteAsync(CommandContext context, Settings settings, CancellationToken cancellationToken)
    {
        await using var bridge = await BridgeFactory.CreateAsync(settings.Host, settings.Verbose, cancellationToken);

        var model = await bridge.GetModelAsync(settings.Model, cancellationToken);
        if (model is null)
        {
            AnsiConsole.MarkupLine($"[red]No model named '{Markup.Escape(settings.Model)}'.[/]");
            return 1;
        }

        // Deleting weights means re-downloading gigabytes, so confirm unless told not to.
        if (!settings.Yes)
        {
            // Prompting where nothing can answer fails the command with an error about stdin,
            // which tells a script author nothing about how to fix it.
            if (!AnsiConsole.Profile.Capabilities.Interactive)
            {
                AnsiConsole.MarkupLine(
                    $"[red]Refusing to delete {Markup.Escape(model.Id)} without confirmation.[/]");
                AnsiConsole.MarkupLine("Pass [cyan]--yes[/] to delete it in a non-interactive session.");
                return 1;
            }

            if (!AnsiConsole.Confirm(
                    $"Delete [cyan]{Markup.Escape(model.Id)}[/] ({ListCommand.FormatSize(model.SizeBytes)})?",
                    defaultValue: false))
            {
                AnsiConsole.MarkupLine("[dim]Cancelled.[/]");
                return 0;
            }
        }

        var removed = await bridge.RemoveAsync(model.Id, cancellationToken);

        AnsiConsole.MarkupLine(removed
            ? $"[green]✓[/] Removed {Markup.Escape(model.Id)}"
            : $"[red]Could not remove {Markup.Escape(model.Id)}.[/]");

        return removed ? 0 : 1;
    }
}

/// <summary>Shows a model's details and its Modelfile.</summary>
public sealed class ShowCommand : AsyncCommand<ShowCommand.Settings>
{
    public sealed class Settings : GlobalSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("Model id or name")]
        public string Model { get; init; } = string.Empty;

        [CommandOption("--modelfile")]
        [Description("Print only the Modelfile.")]
        public bool ModelfileOnly { get; init; }
    }

    protected override async Task<int> ExecuteAsync(CommandContext context, Settings settings, CancellationToken cancellationToken)
    {
        await using var bridge = await BridgeFactory.CreateAsync(settings.Host, settings.Verbose, cancellationToken);

        var model = await bridge.GetModelAsync(settings.Model, cancellationToken);
        if (model is null)
        {
            AnsiConsole.MarkupLine($"[red]No model named '{Markup.Escape(settings.Model)}'.[/]");
            return 1;
        }

        if (settings.ModelfileOnly)
        {
            AnsiConsole.WriteLine(ModelfileParser.Render(new Modelfile
            {
                From = model.Path,
                Template = model.ChatTemplate,
                LoadOptions = new Core.Engines.ModelLoadOptions { ContextSize = model.ContextLength }
            }));

            return 0;
        }

        var grid = new Grid().AddColumn(new GridColumn().NoWrap().PadRight(2)).AddColumn();

        void Row(string label, string value) =>
            grid.AddRow($"[grey]{label}[/]", Markup.Escape(value));

        Row("Id", model.Id);
        Row("Name", model.Name);
        Row("Publisher", string.IsNullOrEmpty(model.Publisher) ? "-" : model.Publisher);
        Row("Format", model.Format.ToString());
        Row("Quantization", $"{model.Quantization.Name} ({model.Quantization.BitsPerWeight}-bit)");
        Row("Size", ListCommand.FormatSize(model.SizeBytes));
        Row("Parameters", model.ParameterCountB > 0 ? $"{model.ParameterCountB:N1}B" : "unknown");
        Row("Context", $"{model.ContextLength:N0} tokens");
        Row("Capabilities", string.Join(", ", model.Capabilities));
        Row("Engines", string.Join(", ", model.SupportedEngines));
        Row("Source", model.Source);
        Row("Path", model.Path);
        Row("Est. memory", ListCommand.FormatSize(model.EstimateMemoryBytes()));

        AnsiConsole.Write(new Panel(grid)
            .Header($"[bold cyan]{Markup.Escape(model.Id)}[/]")
            .Border(BoxBorder.Rounded)
            .BorderColor(Color.Grey30));

        if (!string.IsNullOrEmpty(model.Quantization.Notes))
        {
            AnsiConsole.MarkupLine($"[dim]{Markup.Escape(model.Quantization.Notes)}[/]");
        }

        return 0;
    }
}

/// <summary>Measures prompt processing and generation throughput.</summary>
public sealed class BenchmarkCommand : AsyncCommand<BenchmarkCommand.Settings>
{
    public sealed class Settings : GlobalSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("Model id or name")]
        public string Model { get; init; } = string.Empty;

        [CommandOption("-n|--runs <COUNT>")]
        [Description("Number of runs to average over.")]
        public int Runs { get; init; } = 3;

        [CommandOption("-t|--tokens <COUNT>")]
        [Description("Tokens to generate per run.")]
        public int Tokens { get; init; } = 128;

        [CommandOption("-p|--prompt <TEXT>")]
        [Description("Prompt to benchmark with.")]
        public string Prompt { get; init; } =
            "Write a detailed technical explanation of how transformer attention works.";
    }

    protected override async Task<int> ExecuteAsync(CommandContext context, Settings settings, CancellationToken cancellationToken)
    {
        await using var bridge = await BridgeFactory.CreateAsync(settings.Host, settings.Verbose, cancellationToken);

        var model = await bridge.GetModelAsync(settings.Model, cancellationToken);
        if (model is null)
        {
            AnsiConsole.MarkupLine($"[red]No model named '{Markup.Escape(settings.Model)}'.[/]");
            return 1;
        }

        AnsiConsole.MarkupLine($"Benchmarking [cyan]{Markup.Escape(model.Id)}[/] " +
                               $"— {settings.Runs} run(s) × {settings.Tokens} tokens");
        AnsiConsole.MarkupLine($"[dim]{bridge.Mode}[/]");
        AnsiConsole.WriteLine();

        var results = new List<(TimeSpan FirstToken, TimeSpan Total, int Tokens)>();

        await AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .StartAsync("Warming up…", async ctx =>
            {
                // The first run pays model load and page-in costs, which would skew the average.
                await RunOnceAsync(bridge, model.Id, settings.Prompt, 16, cancellationToken);

                for (var run = 1; run <= settings.Runs; run++)
                {
                    ctx.Status($"Run {run} of {settings.Runs}…");
                    results.Add(await RunOnceAsync(bridge, model.Id, settings.Prompt, settings.Tokens, cancellationToken));
                }
            });

        var table = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Grey30)
            .AddColumn("Run")
            .AddColumn("Time to first token", c => c.RightAligned())
            .AddColumn("Tokens", c => c.RightAligned())
            .AddColumn("Total", c => c.RightAligned())
            .AddColumn("Tokens/s", c => c.RightAligned());

        for (var i = 0; i < results.Count; i++)
        {
            var (firstToken, total, tokens) = results[i];
            table.AddRow(
                (i + 1).ToString(),
                $"{firstToken.TotalMilliseconds:N0} ms",
                tokens.ToString(),
                $"{total.TotalSeconds:N2} s",
                $"{Throughput(firstToken, total, tokens):N1}");
        }

        AnsiConsole.Write(table);

        var averageThroughput = results.Average(r => Throughput(r.FirstToken, r.Total, r.Tokens));
        var averageFirstToken = results.Average(r => r.FirstToken.TotalMilliseconds);

        AnsiConsole.MarkupLine(
            $"[bold green]Average:[/] {averageThroughput:N1} tokens/s, " +
            $"{averageFirstToken:N0} ms to first token");

        return 0;
    }

    /// <summary>Throughput excludes prompt processing, which is what "tokens/s" conventionally means.</summary>
    private static double Throughput(TimeSpan firstToken, TimeSpan total, int tokens)
    {
        var generationTime = (total - firstToken).TotalSeconds;
        return generationTime > 0.001 ? tokens / generationTime : 0;
    }

    private static async Task<(TimeSpan FirstToken, TimeSpan Total, int Tokens)> RunOnceAsync(
        ILocalGenBridge bridge,
        string modelId,
        string prompt,
        int maxTokens,
        CancellationToken cancellationToken)
    {
        var request = new ChatRequest
        {
            Model = modelId,
            Messages = [ChatMessage.User(prompt)],
            Options = new GenerationOptions { MaxTokens = maxTokens, Temperature = 0.7f }
        };

        var stopwatch = Stopwatch.StartNew();
        var firstToken = TimeSpan.Zero;
        var tokens = 0;

        await foreach (var chunk in bridge.StreamAsync(request, cancellationToken))
        {
            if (chunk.Delta.Length > 0)
            {
                if (firstToken == TimeSpan.Zero)
                {
                    firstToken = stopwatch.Elapsed;
                }

                tokens++;
            }

            if (chunk.Usage is { CompletionTokens: > 0 })
            {
                tokens = chunk.Usage.CompletionTokens;
            }
        }

        return (firstToken, stopwatch.Elapsed, tokens);
    }
}
