using System.Collections.ObjectModel;
using System.Diagnostics;
using Avalonia.Layout;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LocalGen.Core.Inference;
using LocalGen.Core.Models;
using LocalGen.Engines.LlamaSharp;
using LocalGen.Runtime;
using LocalGen.Runtime.Sessions;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace AvaloniaEmbedded;

public sealed partial class Turn(string speaker, bool isUser) : ObservableObject
{
    [ObservableProperty]
    private string _text = string.Empty;

    public string Speaker { get; } = speaker;

    public HorizontalAlignment Alignment { get; } =
        isUser ? HorizontalAlignment.Right : HorizontalAlignment.Left;
}

/// <summary>
/// A desktop chat client with no server.
/// </summary>
/// <remarks>
/// This is the pattern for a self-contained desktop AI application: the LocalGen runtime is
/// composed into the app's own container and models are loaded in-process, so the whole thing is
/// one executable with nothing to install or start alongside it.
///
/// The trade-off is that the model lives in this process — its memory is the app's memory, and
/// nothing else on the machine can share it. Talk to a server instead when several applications
/// need the same model.
/// </remarks>
public sealed partial class MainWindowViewModel : ObservableObject
{
    private readonly ServiceProvider _services;
    private readonly IModelStore _store;
    private readonly ModelSessionManager _sessions;

    private CancellationTokenSource? _generation;

    [ObservableProperty]
    private ModelDescriptor? _selectedModel;

    [ObservableProperty]
    private string _prompt = string.Empty;

    [ObservableProperty]
    private string _status = "starting…";

    [ObservableProperty]
    private string _statistics = string.Empty;

    [ObservableProperty]
    private bool _isGenerating;

    public MainWindowViewModel()
    {
        var configuration = new ConfigurationBuilder()
            .AddEnvironmentVariables("LOCALGEN_")
            .Build();

        var services = new ServiceCollection();

        services.AddLogging(logging => logging.SetMinimumLevel(LogLevel.Warning));
        services.AddLocalGenRuntime(configuration);
        services.AddLlamaSharpEngine();

        _services = services.BuildServiceProvider();
        _store = _services.GetRequiredService<IModelStore>();
        _sessions = _services.GetRequiredService<ModelSessionManager>();

        _ = LoadModelsAsync();
    }

    public ObservableCollection<ModelDescriptor> Models { get; } = [];

    public ObservableCollection<Turn> Turns { get; } = [];

    private async Task LoadModelsAsync()
    {
        // Reads the same data directory the CLI and server use, so a model pulled with
        // `localgen pull` shows up here with no extra step.
        var models = await _store.ListAsync();

        foreach (var model in models.Where(static m => m.Supports(ModelCapability.Chat)))
        {
            Models.Add(model);
        }

        if (Models.Count == 0)
        {
            Status = "No models installed. Pull one with: localgen pull <reference>";
            return;
        }

        SelectedModel = Models[0];
        Status = $"{Models.Count} model(s) available · loaded on first message";
    }

    [RelayCommand]
    private async Task SendAsync()
    {
        if (IsGenerating || string.IsNullOrWhiteSpace(Prompt) || SelectedModel is null)
        {
            return;
        }

        var input = Prompt.Trim();

        Prompt = string.Empty;
        IsGenerating = true;
        _generation = new CancellationTokenSource();

        Turns.Add(new Turn("You", isUser: true) { Text = input });

        var reply = new Turn(SelectedModel.Id, isUser: false);
        Turns.Add(reply);

        var stopwatch = Stopwatch.StartNew();
        var firstToken = TimeSpan.Zero;
        TokenUsage? usage = null;

        try
        {
            Status = "loading model…";

            // The lease is held for the whole generation, including the enumeration below —
            // releasing it early would let the session be evicted mid-decode.
            using var lease = await _sessions.AcquireAsync(SelectedModel.Id, null, _generation.Token);

            Status = $"{lease.Session.Engine} · {lease.Session.Device} · ctx {lease.Session.ContextSize:N0}";

            var request = new ChatRequest
            {
                Model = SelectedModel.Id,
                Options = new GenerationOptions { Temperature = 0.7f, MaxTokens = 1024 },
                Messages = [.. BuildHistory()]
            };

            await foreach (var chunk in lease.Session.StreamAsync(request, _generation.Token))
            {
                if (chunk.Delta.Length > 0)
                {
                    if (firstToken == TimeSpan.Zero)
                    {
                        firstToken = stopwatch.Elapsed;
                    }

                    // Generation runs off the UI thread; the binding needs the UI one.
                    await Dispatcher.UIThread.InvokeAsync(() => reply.Text += chunk.Delta);
                }

                if (chunk.Usage is not null)
                {
                    usage = chunk.Usage;
                }
            }

            if (usage is not null)
            {
                var generationTime = (stopwatch.Elapsed - firstToken).TotalSeconds;
                var throughput = generationTime > 0.001 ? usage.CompletionTokens / generationTime : 0;

                Statistics =
                    $"{usage.PromptTokens} prompt + {usage.CompletionTokens} completion · " +
                    $"{firstToken.TotalMilliseconds:N0} ms to first token · {throughput:N1} tok/s";
            }
        }
        catch (OperationCanceledException)
        {
            reply.Text += "\n\n[stopped]";
        }
        catch (Exception ex)
        {
            reply.Text = $"Error: {ex.Message}";
        }
        finally
        {
            IsGenerating = false;
        }
    }

    [RelayCommand]
    private void Stop() => _generation?.Cancel();

    [RelayCommand]
    private void Clear()
    {
        Turns.Clear();
        Statistics = string.Empty;
    }

    private IEnumerable<ChatMessage> BuildHistory() =>
        Turns
            .Where(static turn => !string.IsNullOrEmpty(turn.Text))
            .Select(static turn => turn.Alignment == HorizontalAlignment.Right
                ? ChatMessage.User(turn.Text)
                : ChatMessage.Assistant(turn.Text));

    /// <summary>Releases the loaded model before the process exits.</summary>
    public void Shutdown()
    {
        _generation?.Cancel();
        _sessions.DisposeAsync().AsTask().GetAwaiter().GetResult();
        _services.Dispose();
    }
}
