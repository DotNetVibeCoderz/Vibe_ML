using CommunityToolkit.Mvvm.Input;
using LocalGen.Core.Configuration;
using Microsoft.Extensions.Options;

namespace LocalGen.Desktop.ViewModels;

/// <summary>One project in the solution, as listed on the About screen.</summary>
public sealed record Component(string Name, string Purpose);

/// <summary>Product identity, environment facts and where to find things on disk.</summary>
public sealed partial class AboutViewModel : ViewModelBase
{
    private readonly LocalGenOptions _options;

    public AboutViewModel(IOptions<LocalGenOptions> options) => _options = options.Value;

    public string Version =>
        typeof(AboutViewModel).Assembly.GetName().Version?.ToString(3) ?? "0.1.0";

    public string Runtime => System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription;

    public string Platform =>
        $"{System.Runtime.InteropServices.RuntimeInformation.OSDescription} " +
        $"({System.Runtime.InteropServices.RuntimeInformation.OSArchitecture})";

    public string Processors => $"{Environment.ProcessorCount} logical processors";

    public string DataDirectory => _options.DataDirectory;

    public string ModelsDirectory => _options.ModelsDirectory;

    public string SkillsDirectory => _options.SkillsDirectory;

    public string Endpoint => _options.Server.BaseUrl;

    public string Author => "Gravicode Studios";

    public string Lead => "Led by Kang Fadhil";

    public string Tagline => "A local AI inference engine for .NET.";

    // A record rather than a tuple: compiled bindings erase tuple element names at runtime.
    public IReadOnlyList<Component> Components { get; } =
    [
        new("LocalGen.Core", "Engine contracts, Modelfile format and the OpenAI wire protocol"),
        new("LocalGen.Runtime", "Model store, catalogues, engine selection and session lifetime"),
        new("LocalGen.Engines.LlamaSharp", "llama.cpp backend for GGUF models on CPU or GPU"),
        new("LocalGen.Engines.Onnx", "ONNX Runtime GenAI backend"),
        new("LocalGen.Engines.FoundryLocal", "Microsoft Foundry Local backend"),
        new("LocalGen.Server", "OpenAI-compatible HTTP API and management endpoints"),
        new("LocalGen.Kernel", "Semantic Kernel integration, built-in functions, skills and MCP"),
        new("LocalGen.Rag", "Document ingestion and vector search"),
        new("LocalGen.Sdk", "Client library for console, web, desktop and IoT applications"),
        new("LocalGen.Cli", "Command line interface")
    ];

    [RelayCommand]
    private void OpenDataDirectory() => OpenPath(DataDirectory);

    [RelayCommand]
    private void OpenModelsDirectory() => OpenPath(ModelsDirectory);

    [RelayCommand]
    private void OpenSkillsDirectory() => OpenPath(SkillsDirectory);

    /// <summary>
    /// Opens a folder in the platform's file manager. <c>UseShellExecute</c> is what lets the OS
    /// pick the handler rather than LocalGen guessing at explorer.exe, Finder or xdg-open.
    /// </summary>
    private void OpenPath(string path)
    {
        try
        {
            Directory.CreateDirectory(path);

            System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = path,
                UseShellExecute = true
            });
        }
        catch (Exception ex)
        {
            ErrorMessage = $"Could not open {path}: {ex.Message}";
        }
    }
}
