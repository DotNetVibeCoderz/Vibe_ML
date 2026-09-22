using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using HFAppGen.ViewModels;

namespace HFAppGen.Views;

/// <summary>The assistant chat panel.</summary>
public partial class ChatView : UserControl
{
    /// <summary>Creates the view.</summary>
    public ChatView()
    {
        AvaloniaXamlLoader.Load(this);
        DataContextChanged += OnDataContextChanged;
    }

    private ChatViewModel? Model => DataContext as ChatViewModel;

    private void OnDataContextChanged(object? sender, EventArgs e)
    {
        if (Model is not null) Model.Messages.CollectionChanged += (_, _) => ScrollToEnd();
    }

    private void ScrollToEnd() => Dispatcher.UIThread.Post(
        () => this.FindControl<ScrollViewer>("Transcript")?.ScrollToEnd(),
        DispatcherPriority.Background);

    private async void OnSend(object? sender, RoutedEventArgs e) => await SendAsync();

    private async void OnComposerKeyDown(object? sender, KeyEventArgs e)
    {
        // Ctrl+Enter sends; plain Enter inserts a newline, because prompts are often multi-line.
        if (e.Key != Key.Enter || !e.KeyModifiers.HasFlag(KeyModifiers.Control)) return;

        e.Handled = true;
        await SendAsync();
    }

    private async Task SendAsync()
    {
        if (Model is null) return;

        await Model.SendAsync();
        ScrollToEnd();
    }

    private void OnClearThread(object? sender, RoutedEventArgs e) => Model?.ClearThread();

    private void OnRemoveAttachment(object? sender, RoutedEventArgs e)
    {
        if (sender is Button { Tag: string path }) Model?.RemoveAttachment(path);
    }

    private async void OnAttach(object? sender, RoutedEventArgs e)
    {
        var storage = TopLevel.GetTopLevel(this)?.StorageProvider;
        if (storage is null || Model is null) return;

        var files = await storage.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Attach images",
            AllowMultiple = true,
            FileTypeFilter = [FilePickerFileTypes.ImageAll],
        });

        Model.Attach(files.Select(f => f.Path.LocalPath));
    }
}
