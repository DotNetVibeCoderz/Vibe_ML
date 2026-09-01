using System.Collections.ObjectModel;
using System.Text.Json;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LocalGen.Core.Protocol;
using LocalGen.Sdk;

namespace LocalGenDesktop;

/// <summary>Converters used by the chat transcript.</summary>
public static class TurnConverters
{
    /// <summary>Puts the user's own turns on the right, the way a conversation reads.</summary>
    public static readonly Avalonia.Data.Converters.IValueConverter Alignment =
        new Avalonia.Data.Converters.FuncValueConverter<bool, Avalonia.Layout.HorizontalAlignment>(
            static isUser => isUser
                ? Avalonia.Layout.HorizontalAlignment.Right
                : Avalonia.Layout.HorizontalAlignment.Left);
}

/// <summary>One turn in the transcript.</summary>
public sealed partial class ChatTurn : ObservableObject
{
    [ObservableProperty]
    private string _text = string.Empty;

    public required bool IsUser { get; init; }

    public string Speaker => IsUser ? "You" : "Assistant";
}

public sealed partial class MainWindowViewModel : ObservableObject
{
    private readonly LocalGenClient _client;
    private CancellationTokenSource? _generation;

    [ObservableProperty]
    private string _prompt = string.Empty;

    [ObservableProperty]
    private string _model = "LOCALGEN_MODEL_PLACEHOLDER";

    [ObservableProperty]
    private string _status = "connecting…";

    [ObservableProperty]
    private bool _isGenerating;

    public MainWindowViewModel()
    {
        _client = new LocalGenClient(new LocalGenClientOptions
        {
            Endpoint = "LOCALGEN_ENDPOINT_PLACEHOLDER",
            DefaultModel = Model
        });

        _ = ConnectAsync();
    }

    public ObservableCollection<ChatTurn> Turns { get; } = [];

    public ObservableCollection<string> Models { get; } = [];

    private async Task ConnectAsync()
    {
        if (!await _client.PingAsync())
        {
            Status = "LocalGen is not running — start it with: localgen serve";
            return;
        }

        foreach (var model in await _client.ListModelsAsync())
        {
            Models.Add(model.Id);
        }

        if (Models.Count == 0)
        {
            Status = "No models installed. Pull one with: localgen pull <reference>";
            return;
        }

        // Prefer the configured model, but fall back to whatever is actually installed.
        if (!Models.Contains(Model))
        {
            Model = Models[0];
        }

        Status = "ready";
    }

    [RelayCommand]
    private async Task SendAsync()
    {
        if (IsGenerating || string.IsNullOrWhiteSpace(Prompt))
        {
            return;
        }

        var input = Prompt.Trim();

        Prompt = string.Empty;
        IsGenerating = true;
        _generation = new CancellationTokenSource();

        Turns.Add(new ChatTurn { IsUser = true, Text = input });

        var reply = new ChatTurn { IsUser = false, Text = string.Empty };
        Turns.Add(reply);

        try
        {
            var request = new ChatCompletionRequest
            {
                Model = Model,
                Temperature = 0.7f,
                // The server holds no session state, so the conversation is replayed each turn.
                Messages = [.. BuildHistory()]
            };

            await foreach (var chunk in _client.StreamChatAsync(request, _generation.Token))
            {
                var delta = chunk.Choices.FirstOrDefault()?.Delta?.Content;

                if (!string.IsNullOrEmpty(delta))
                {
                    reply.Text += delta;
                }
            }
        }
        catch (OperationCanceledException)
        {
            reply.Text += "\n\n[stopped]";
        }
        catch (LocalGenClientException ex)
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
    private void Clear() => Turns.Clear();

    private IEnumerable<OpenAiMessage> BuildHistory() =>
        Turns
            .Where(static turn => !string.IsNullOrEmpty(turn.Text))
            .Select(static turn => new OpenAiMessage
            {
                Role = turn.IsUser ? "user" : "assistant",
                Content = JsonSerializer.SerializeToElement(turn.Text)
            });
}
