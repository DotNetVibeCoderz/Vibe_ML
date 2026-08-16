using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using LocalGen.Core.Protocol;
using LocalGen.Sdk;

namespace WpfChat;

/// <summary>
/// A WPF chat client for a local model.
/// </summary>
/// <remarks>
/// The SDK usage is identical to the console sample; what is WPF-specific is the marshalling.
/// Tokens arrive on a thread-pool thread, and WPF bindings must be updated from the UI thread, so
/// every mutation of a bound property goes through the dispatcher.
/// </remarks>
public partial class MainWindow : Window
{
    private readonly LocalGenClient _client;
    private readonly ObservableCollection<Turn> _turns = [];

    private CancellationTokenSource? _generation;
    private bool _isGenerating;

    public MainWindow()
    {
        InitializeComponent();

        Transcript.ItemsSource = _turns;

        _client = new LocalGenClient(new LocalGenClientOptions
        {
            Endpoint = Environment.GetEnvironmentVariable("LOCALGEN_HOST") ?? "http://127.0.0.1:11434",
            ApiKey = Environment.GetEnvironmentVariable("LOCALGEN_API_KEY")
        });

        Loaded += async (_, _) => await ConnectAsync();
        Closed += (_, _) =>
        {
            _generation?.Cancel();
            _client.Dispose();
        };
    }

    private async Task ConnectAsync()
    {
        if (!await _client.PingAsync())
        {
            StatusText.Text = "LocalGen is not running — start it with: localgen serve";
            SendButton.IsEnabled = false;
            return;
        }

        var models = await _client.ListModelsAsync();

        if (models.Count == 0)
        {
            StatusText.Text = "no models installed";
            SendButton.IsEnabled = false;
            return;
        }

        ModelPicker.ItemsSource = models.Select(m => m.Id).ToList();
        ModelPicker.SelectedIndex = 0;

        StatusText.Text = "ready";
    }

    private void OnPromptKeyDown(object sender, KeyEventArgs e)
    {
        // Enter sends, Shift+Enter adds a line — what people expect from a chat box.
        if (e.Key == Key.Enter && !Keyboard.Modifiers.HasFlag(ModifierKeys.Shift))
        {
            e.Handled = true;
            _ = SendAsync();
        }
    }

    private void OnSendClick(object sender, RoutedEventArgs e)
    {
        if (_isGenerating)
        {
            _generation?.Cancel();
            return;
        }

        _ = SendAsync();
    }

    private async Task SendAsync()
    {
        var input = PromptBox.Text.Trim();

        if (_isGenerating || input.Length == 0 || ModelPicker.SelectedItem is not string model)
        {
            return;
        }

        PromptBox.Clear();
        SetGenerating(true);

        _generation = new CancellationTokenSource();

        _turns.Add(new Turn("You", HorizontalAlignment.Right) { Text = input });

        var reply = new Turn("LocalGen", HorizontalAlignment.Left);
        _turns.Add(reply);

        ScrollToEnd();

        try
        {
            var request = new ChatCompletionRequest
            {
                Model = model,
                Temperature = 0.7f,
                // The server holds no session state, so the conversation is replayed each turn.
                Messages = [.. BuildHistory()]
            };

            await foreach (var chunk in _client.StreamChatAsync(request, _generation.Token))
            {
                var delta = chunk.Choices.FirstOrDefault()?.Delta?.Content;

                if (string.IsNullOrEmpty(delta))
                {
                    continue;
                }

                // StreamChatAsync resumes on a thread-pool thread; the binding needs the UI one.
                await Dispatcher.InvokeAsync(() =>
                {
                    reply.Text += delta;
                    ScrollToEnd();
                });
            }
        }
        catch (OperationCanceledException)
        {
            await Dispatcher.InvokeAsync(() => reply.Text += "\n\n[stopped]");
        }
        catch (LocalGenClientException ex)
        {
            await Dispatcher.InvokeAsync(() => reply.Text = $"Error: {ex.Message}");
        }
        finally
        {
            await Dispatcher.InvokeAsync(() => SetGenerating(false));
        }
    }

    private IEnumerable<OpenAiMessage> BuildHistory() =>
        _turns
            .Where(turn => !string.IsNullOrEmpty(turn.Text))
            .Select(turn => new OpenAiMessage
            {
                Role = turn.Speaker == "You" ? "user" : "assistant",
                Content = JsonSerializer.SerializeToElement(turn.Text)
            });

    private void SetGenerating(bool generating)
    {
        _isGenerating = generating;
        SendButton.Content = generating ? "Stop" : "Send";
        PromptBox.IsEnabled = !generating;
    }

    private void ScrollToEnd() => TranscriptScroller.ScrollToEnd();

    /// <summary>
    /// One turn. Implements <see cref="INotifyPropertyChanged"/> because the assistant's text
    /// grows token by token after the item is already bound.
    /// </summary>
    private sealed class Turn(string speaker, HorizontalAlignment alignment) : INotifyPropertyChanged
    {
        private string _text = string.Empty;

        public string Speaker { get; } = speaker;

        public HorizontalAlignment Alignment { get; } = alignment;

        public string Text
        {
            get => _text;
            set
            {
                _text = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Text)));
            }
        }

        public event PropertyChangedEventHandler? PropertyChanged;
    }
}
