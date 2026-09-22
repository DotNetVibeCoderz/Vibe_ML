using Avalonia;
using HFAppGen.Models;
using HFAppGen.Services;

namespace HFAppGen;

/// <summary>Entry point for the HFAppGen desktop application.</summary>
public static class Program
{
    /// <summary>Starts the Avalonia application, or runs a headless check.</summary>
    /// <param name="args">
    /// <c>--selftest</c> runs one round trip against the configured model and exits. Everything
    /// else starts the window.
    /// </param>
    [STAThread]
    public static int Main(string[] args)
    {
        if (args.Contains("--selftest", StringComparer.OrdinalIgnoreCase))
        {
            return SelfTest(args).GetAwaiter().GetResult();
        }

        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
        return 0;
    }

    /// <summary>Configures Avalonia. Referenced by the designer as well as <see cref="Main"/>.</summary>
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<App>()
        .UsePlatformDetect()
        .WithInterFont()
        .LogToTrace();

    /// <summary>
    /// Drives the assistant once, without a window, and reports what came back.
    /// </summary>
    /// <remarks>
    /// The LLM path is the one part of this app that cannot be covered by a unit test: it needs a
    /// real endpoint, a real key and a real model. A headless mode makes that check something that
    /// can be run from a terminal or from CI rather than by opening the window and typing.
    /// </remarks>
    private static async Task<int> SelfTest(string[] args)
    {
        var prompt = args.SkipWhile(a => !a.Equals("--selftest", StringComparison.OrdinalIgnoreCase))
            .Skip(1)
            .FirstOrDefault()
            ?? "In one sentence: what does GraviHub do in HF.Net? Call HFNetReference first.";

        var settings = new ConfigurationService().Load();

        Console.WriteLine($"provider : {settings.Provider}");
        Console.WriteLine($"model    : {settings.Model}");
        Console.WriteLine($"endpoint : {(string.IsNullOrWhiteSpace(settings.Endpoint) ? "(default)" : settings.Endpoint)}");
        Console.WriteLine($"key      : {(settings.ApiKey.Length > 0 ? $"set ({settings.ApiKey.Length} chars)" : "MISSING")}");
        Console.WriteLine();

        if (!settings.IsConfigured)
        {
            Console.Error.WriteLine("Not configured. Set llm.apiKey in app.config, or HFAPPGEN_APIKEY.");
            return 2;
        }

        var logs = new LogService();
        var projects = new ProjectService(logs);
        var assistant = new AssistantService(projects, logs);

        try
        {
            assistant.Configure(settings);
            assistant.ResetHistory(settings);
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"Could not build the kernel: {error.Message}");
            return 3;
        }

        Console.WriteLine($"> {prompt}\n");

        var tools = new List<string>();
        var length = 0;

        try
        {
            await foreach (var chunk in assistant.SendAsync(prompt, settings))
            {
                if (chunk.ToolName is { } tool && !tools.Contains(tool)) tools.Add(tool);

                Console.Write(chunk.Text);
                length += chunk.Text.Length;
            }
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"\n\nRequest failed: {error.Message}");
            return 4;
        }

        Console.WriteLine($"\n\n---\n{length} characters"
            + (tools.Count > 0 ? $", tools called: {string.Join(", ", tools)}" : ", no tools called"));

        return length > 0 ? 0 : 5;
    }
}
