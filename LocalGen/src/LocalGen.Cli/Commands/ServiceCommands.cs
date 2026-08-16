using System.ComponentModel;
using LocalGen.Core.Modelfiles;
using LocalGen.Sdk;
using LocalGen.Server;
using Microsoft.Extensions.Configuration;
using Spectre.Console;
using Spectre.Console.Cli;

namespace LocalGen.Cli.Commands;

/// <summary>Starts the LocalGen web service in the foreground.</summary>
public sealed class ServeCommand : AsyncCommand<ServeCommand.Settings>
{
    public sealed class Settings : GlobalSettings
    {
        [CommandOption("-p|--port <PORT>")]
        [Description("Port to listen on.")]
        public int? Port { get; init; }

        [CommandOption("-b|--bind <ADDRESS>")]
        [Description("Address to bind. Use 0.0.0.0 to accept connections from other machines.")]
        public string? Bind { get; init; }

        [CommandOption("--api-key <KEY>")]
        [Description("Require this bearer token on every request.")]
        public string? ApiKey { get; init; }

        [CommandOption("--preload <MODEL>")]
        [Description("Load a model at startup so the first request is fast.")]
        public string? Preload { get; init; }

        [CommandOption("--offline")]
        [Description("Block all outbound network access.")]
        public bool Offline { get; init; }
    }

    protected override async Task<int> ExecuteAsync(CommandContext context, Settings settings, CancellationToken cancellationToken)
    {
        // Command-line flags are layered in as configuration so they follow the same binding
        // rules as appsettings.json and environment variables.
        var overrides = new Dictionary<string, string?>();

        if (settings.Port is { } port)
        {
            overrides["LocalGen:Server:Port"] = port.ToString();
        }

        if (!string.IsNullOrWhiteSpace(settings.Bind))
        {
            overrides["LocalGen:Server:Host"] = settings.Bind;
        }

        if (!string.IsNullOrWhiteSpace(settings.ApiKey))
        {
            overrides["LocalGen:Server:ApiKey"] = settings.ApiKey;
        }

        if (!string.IsNullOrWhiteSpace(settings.Preload))
        {
            overrides["LocalGen:Runtime:PreloadModel"] = settings.Preload;
        }

        if (settings.Offline)
        {
            overrides["LocalGen:Runtime:OfflineMode"] = "true";
        }

        var configuration = new ConfigurationBuilder()
            .AddJsonFile("appsettings.json", optional: true)
            .AddEnvironmentVariables()
            .AddInMemoryCollection(overrides)
            .Build();

        var host = new LocalGenServerHost();

        AnsiConsole.Write(new FigletText("LocalGen").Color(Color.Cyan1));
        AnsiConsole.MarkupLine("[dim]Local AI inference engine — Gravicode Studios, led by Kang Fadhil[/]");
        AnsiConsole.WriteLine();

        await host.StartAsync(configuration);

        AnsiConsole.MarkupLine($"[green]✓[/] Listening on [cyan]{host.BaseUrl}[/]");
        AnsiConsole.MarkupLine($"[dim]OpenAI-compatible endpoint: {host.BaseUrl}/v1[/]");

        if (!string.IsNullOrWhiteSpace(settings.Bind) && settings.Bind != "127.0.0.1" &&
            string.IsNullOrWhiteSpace(settings.ApiKey))
        {
            // Binding beyond loopback without a key exposes inference and the tool functions.
            AnsiConsole.MarkupLine(
                "[yellow]Warning:[/] the server is reachable from other machines and no API key is set. " +
                "Pass --api-key to require authentication.");
        }

        AnsiConsole.MarkupLine("[dim]Press Ctrl+C to stop.[/]");

        var stopped = new TaskCompletionSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            stopped.TrySetResult();
        };

        await stopped.Task;

        AnsiConsole.MarkupLine("[dim]Stopping…[/]");
        await host.StopAsync();
        return 0;
    }
}

/// <summary>Shows service status and which models are resident.</summary>
public sealed class StatusCommand : AsyncCommand<GlobalSettings>
{
    protected override async Task<int> ExecuteAsync(CommandContext context, GlobalSettings settings, CancellationToken cancellationToken)
    {
        var endpoint = settings.Host
                       ?? Environment.GetEnvironmentVariable("LOCALGEN_HOST")
                       ?? "http://127.0.0.1:11434";

        using var client = new LocalGenClient(endpoint);

        if (!await client.PingAsync())
        {
            AnsiConsole.MarkupLine($"[red]No LocalGen server is running at {Markup.Escape(endpoint)}.[/]");
            AnsiConsole.MarkupLine("Start one with: [cyan]localgen serve[/]");
            return 1;
        }

        var status = await client.Admin.GetStatusAsync();

        var grid = new Grid().AddColumn(new GridColumn().NoWrap().PadRight(2)).AddColumn();
        grid.AddRow("[grey]Status[/]", $"[green]{status.Status}[/]");
        grid.AddRow("[grey]Version[/]", status.Version);
        grid.AddRow("[grey]Address[/]", status.BaseUrl);
        grid.AddRow("[grey]Uptime[/]", FormatUptime(status.Uptime));
        grid.AddRow("[grey]Offline mode[/]", status.OfflineMode ? "on" : "off");
        grid.AddRow("[grey]Data[/]", Markup.Escape(status.DataDirectory));

        AnsiConsole.Write(new Panel(grid)
            .Header("[bold cyan]LocalGen[/]")
            .Border(BoxBorder.Rounded)
            .BorderColor(Color.Grey30));

        if (status.LoadedModels.Count == 0)
        {
            AnsiConsole.MarkupLine("[dim]No models are loaded.[/]");
            return 0;
        }

        var table = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Grey30)
            .AddColumn("Loaded model")
            .AddColumn("Engine")
            .AddColumn("Device")
            .AddColumn("Context", c => c.RightAligned())
            .AddColumn("Requests", c => c.RightAligned())
            .AddColumn("Idle");

        foreach (var model in status.LoadedModels)
        {
            table.AddRow(
                Markup.Escape(model.Id),
                model.Engine,
                model.Device,
                $"{model.ContextSize:N0}",
                model.RequestCount.ToString(),
                FormatUptime(DateTimeOffset.UtcNow - model.LastUsedAt));
        }

        AnsiConsole.Write(table);
        return 0;
    }

    private static string FormatUptime(TimeSpan value) => value switch
    {
        { TotalSeconds: < 60 } => $"{value.TotalSeconds:N0}s",
        { TotalMinutes: < 60 } => $"{value.TotalMinutes:N0}m",
        { TotalHours: < 24 } => $"{value.Hours}h {value.Minutes}m",
        _ => $"{value.Days}d {value.Hours}h"
    };
}

/// <summary>Lists the inference backends and what this machine can actually use.</summary>
public sealed class EnginesCommand : AsyncCommand<GlobalSettings>
{
    protected override async Task<int> ExecuteAsync(CommandContext context, GlobalSettings settings, CancellationToken cancellationToken)
    {
        var endpoint = settings.Host
                       ?? Environment.GetEnvironmentVariable("LOCALGEN_HOST")
                       ?? "http://127.0.0.1:11434";

        using var client = new LocalGenClient(endpoint);

        if (!await client.PingAsync())
        {
            AnsiConsole.MarkupLine($"[red]No LocalGen server is running at {Markup.Escape(endpoint)}.[/]");
            AnsiConsole.MarkupLine("Start one with: [cyan]localgen serve[/]");
            return 1;
        }

        var engines = await client.Admin.GetEnginesAsync();

        foreach (var engine in engines.Engines)
        {
            var state = engine.IsAvailable ? "[green]available[/]" : "[red]unavailable[/]";
            var isDefault = engine.IsDefault ? " [cyan](default)[/]" : string.Empty;

            var body = new Grid().AddColumn(new GridColumn().NoWrap().PadRight(2)).AddColumn();
            body.AddRow("[grey]State[/]", state + isDefault);

            if (!engine.IsAvailable && !string.IsNullOrEmpty(engine.UnavailableReason))
            {
                body.AddRow("[grey]Reason[/]", Markup.Escape(engine.UnavailableReason));
            }

            body.AddRow("[grey]Devices[/]", string.Join(", ", engine.Devices));
            body.AddRow("[grey]Formats[/]", string.Join(", ", engine.Formats));
            body.AddRow("[grey]Grammar[/]", engine.SupportsGrammar ? "yes" : "no");
            body.AddRow("[grey]Embeddings[/]", engine.SupportsEmbeddings ? "yes" : "no");
            body.AddRow("[grey]Multi-GPU[/]", engine.SupportsMultiGpu ? "yes" : "no");

            if (!string.IsNullOrEmpty(engine.Version))
            {
                body.AddRow("[grey]Build[/]", Markup.Escape(Truncate(engine.Version, 90)));
            }

            body.AddRow("[grey]When to use[/]", Markup.Escape(engine.Recommendation));

            AnsiConsole.Write(new Panel(body)
                .Header($"[bold]{Markup.Escape(engine.DisplayName)}[/]")
                .Border(BoxBorder.Rounded)
                .BorderColor(engine.IsAvailable ? Color.Green : Color.Grey30));
        }

        AnsiConsole.Write(new Panel(
                new Markup($"[bold]{engines.RecommendedEngine}[/] on [bold]{engines.RecommendedDevice}[/]\n" +
                           Markup.Escape(engines.Rationale)))
            .Header("[bold cyan]Recommended for this machine[/]")
            .Border(BoxBorder.Rounded)
            .BorderColor(Color.Cyan1));

        return 0;
    }

    private static string Truncate(string value, int length) =>
        value.Length <= length ? value : value[..length] + "…";
}

/// <summary>Creates a derived model from a Modelfile.</summary>
public sealed class CreateCommand : AsyncCommand<CreateCommand.Settings>
{
    public sealed class Settings : GlobalSettings
    {
        [CommandArgument(0, "<name>")]
        [Description("Name for the new model")]
        public string Name { get; init; } = string.Empty;

        [CommandOption("-f|--file <PATH>")]
        [Description("Path to the Modelfile.")]
        public string File { get; init; } = "Modelfile";
    }

    protected override async Task<int> ExecuteAsync(CommandContext context, Settings settings, CancellationToken cancellationToken)
    {
        if (!System.IO.File.Exists(settings.File))
        {
            AnsiConsole.MarkupLine($"[red]No Modelfile at '{Markup.Escape(settings.File)}'.[/]");
            return 1;
        }

        Modelfile modelfile;

        try
        {
            modelfile = await ModelfileParser.ParseFileAsync(settings.File);
        }
        catch (ModelfileException ex)
        {
            AnsiConsole.MarkupLine($"[red]{Markup.Escape(ex.Message)}[/]");
            return 1;
        }

        // Creating a derived model needs the store, which the remote bridge does not expose;
        // this runs in-process against the same data directory the server uses.
        await using var embedded = EmbeddedBridge.Create(settings.Verbose);

        AnsiConsole.MarkupLine($"Creating [cyan]{Markup.Escape(settings.Name)}[/] " +
                               $"from [dim]{Markup.Escape(modelfile.From)}[/]…");

        try
        {
            var store = (Core.Models.IModelStore)embedded.Services
                .GetService(typeof(Core.Models.IModelStore))!;

            var model = await store.CreateAsync(settings.Name, modelfile);

            AnsiConsole.MarkupLine($"[green]✓[/] Created [cyan]{Markup.Escape(model.Id)}[/]");
            AnsiConsole.MarkupLine($"Run it with: [cyan]localgen run {Markup.Escape(model.Id)}[/]");
            return 0;
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]{Markup.Escape(ex.Message)}[/]");
            return 1;
        }
    }
}
