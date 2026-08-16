using System.ComponentModel;
using LocalGen.Core.Models;
using Microsoft.Extensions.DependencyInjection;
using Spectre.Console;
using Spectre.Console.Cli;

namespace LocalGen.Cli.Commands;

/// <summary>
/// Converts a model to a smaller quantization.
/// </summary>
/// <remarks>
/// Runs in-process rather than against a server: quantization is a long CPU-bound job that reads
/// and writes whole model files, and holding an HTTP request open for minutes to do it would be
/// the wrong shape.
/// </remarks>
public sealed class QuantizeCommand : AsyncCommand<QuantizeCommand.Settings>
{
    public sealed class Settings : GlobalSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("Installed model id, or a path to a .gguf file")]
        public string Model { get; init; } = string.Empty;

        [CommandArgument(1, "<quantization>")]
        [Description("Target quantization, e.g. Q4_K_M")]
        public string Quantization { get; init; } = string.Empty;

        [CommandOption("-o|--output <PATH>")]
        [Description("Where to write the result. Defaults to the models directory.")]
        public string? Output { get; init; }

        [CommandOption("-t|--threads <COUNT>")]
        [Description("Threads to use. Omit to let llama.cpp decide.")]
        public int? Threads { get; init; }

        [CommandOption("--allow-requantize")]
        [Description("Permit quantizing a model that is already quantized. Quality suffers.")]
        public bool AllowRequantize { get; init; }

        [CommandOption("-y|--yes")]
        [Description("Overwrite an existing output file without asking.")]
        public bool Yes { get; init; }
    }

    protected override async Task<int> ExecuteAsync(
        CommandContext context,
        Settings settings,
        CancellationToken cancellationToken)
    {
        await using var embedded = EmbeddedBridge.Create(settings.Verbose);

        var quantizer = embedded.Services.GetService<IModelQuantizer>();

        if (quantizer is null)
        {
            AnsiConsole.MarkupLine(
                "[red]No installed backend can quantize models.[/] " +
                "Quantization needs the LlamaSharp backend.");
            return 1;
        }

        if (!quantizer.CanProduce(settings.Quantization))
        {
            AnsiConsole.MarkupLine($"[red]'{Markup.Escape(settings.Quantization)}' is not supported.[/]");
            AnsiConsole.WriteLine();

            var table = new Table()
                .Border(TableBorder.Rounded)
                .BorderColor(Color.Grey30)
                .AddColumn("Quantization")
                .AddColumn("Bits", c => c.RightAligned())
                .AddColumn("Notes");

            foreach (var quantization in quantizer.Supported)
            {
                table.AddRow(
                    Markup.Escape(quantization.Name),
                    quantization.BitsPerWeight > 0 ? quantization.BitsPerWeight.ToString() : "-",
                    Markup.Escape(quantization.Notes));
            }

            AnsiConsole.Write(table);
            return 1;
        }

        var store = embedded.Services.GetRequiredService<IModelStore>();

        // A model id is resolved through the store; anything else is taken as a path, so an
        // F16 file that was never installed can still be quantized.
        var source = await store.GetAsync(settings.Model, cancellationToken).ConfigureAwait(false);
        var sourcePath = source?.Path ?? settings.Model;

        if (!File.Exists(sourcePath))
        {
            AnsiConsole.MarkupLine(
                $"[red]No model '{Markup.Escape(settings.Model)}'.[/] " +
                "Give an installed model id or a path to a .gguf file.");
            return 1;
        }

        var targetPath = settings.Output
            ?? QuantizationNaming.BuildOutputPath(sourcePath, settings.Quantization);

        if (File.Exists(targetPath) && !settings.Yes)
        {
            var name = Markup.Escape(Path.GetFileName(targetPath));

            // Prompting where nothing can answer would fail the command outright, so a
            // non-interactive run is told what to pass instead.
            if (!AnsiConsole.Profile.Capabilities.Interactive)
            {
                AnsiConsole.MarkupLine($"[red]{name} already exists.[/]");
                AnsiConsole.MarkupLine("Pass [cyan]--yes[/] to overwrite it, or [cyan]--output[/] to write elsewhere.");
                return 1;
            }

            if (!AnsiConsole.Confirm($"[yellow]{name} already exists.[/] Overwrite?", false))
            {
                AnsiConsole.MarkupLine("[dim]Cancelled.[/]");
                return 0;
            }
        }

        var sourceSize = new FileInfo(sourcePath).Length;

        AnsiConsole.MarkupLine($"Source : [cyan]{Markup.Escape(Path.GetFileName(sourcePath))}[/] " +
                               $"({ListCommand.FormatSize(sourceSize)})");
        AnsiConsole.MarkupLine($"Target : [cyan]{Markup.Escape(Path.GetFileName(targetPath))}[/] " +
                               $"({Markup.Escape(settings.Quantization)})");
        AnsiConsole.WriteLine();

        QuantizationResult? result = null;
        Exception? failure = null;

        // llama.cpp reports progress per tensor through the log rather than as a percentage, so
        // a spinner is honest here where a progress bar would not be.
        await AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .StartAsync("Quantizing — this takes minutes on a large model…", async _ =>
            {
                try
                {
                    result = await quantizer.QuantizeAsync(new QuantizationRequest
                    {
                        SourcePath = sourcePath,
                        TargetPath = targetPath,
                        Quantization = settings.Quantization,
                        ThreadCount = settings.Threads,
                        AllowRequantize = settings.AllowRequantize,
                        QuantizeOutputTensor = true
                    }, cancellationToken).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    failure = ex;
                }
            });

        if (failure is not null)
        {
            AnsiConsole.MarkupLine($"[red]Quantization failed:[/] {Markup.Escape(failure.Message)}");
            return 1;
        }

        AnsiConsole.MarkupLine($"[green]✓[/] Wrote [cyan]{Markup.Escape(result!.TargetPath)}[/]");
        AnsiConsole.MarkupLine(
            $"[dim]{ListCommand.FormatSize(result.SourceBytes)} → {ListCommand.FormatSize(result.TargetBytes)} " +
            $"({result.Reduction:P0} smaller) in {result.Duration.TotalSeconds:N0}s[/]");

        // The models directory is rescanned on load, so a file written there is picked up on its
        // own — but saying so saves the user wondering.
        AnsiConsole.MarkupLine("[dim]Run 'localgen list' to see it; models in the data directory are found automatically.[/]");

        return 0;
    }
}
