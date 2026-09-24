using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using HFGallery.ViewModels;

namespace HFGallery.Views;

/// <summary>The gallery window.</summary>
public sealed partial class MainWindow : Window
{
    /// <summary>Builds the window.</summary>
    public MainWindow()
    {
        AvaloniaXamlLoader.Load(this);
        DataContextChanged += OnDataContextChanged;
    }

    private void OnDataContextChanged(object? sender, EventArgs e)
    {
        if (DataContext is MainViewModel model) model.PropertyChanged += OnModelChanged;
    }

    /// <summary>
    /// Brings the result into view once a run finishes.
    /// </summary>
    /// <remarks>
    /// A case whose input is eight lines long pushes its own answer below the fold, so pressing Run
    /// appears to do nothing for the twenty seconds the model takes and then still shows nothing.
    /// </remarks>
    private void OnModelChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(MainViewModel.Result)) return;
        if (sender is not MainViewModel { Result: not null }) return;

        // Queued behind the layout pass the new result triggers: the panel has no size yet.
        Dispatcher.UIThread.Post(
            () => this.FindControl<Border>("ResultPanel")?.BringIntoView(),
            DispatcherPriority.Loaded);

        // Late enough for the scroll and the charts' first frame to have happened.
        if (Program.CapturePath is { } path) DispatcherTimer.RunOnce(() => Capture(path), TimeSpan.FromSeconds(1.5));
    }

    /// <summary>Renders the window into a PNG and closes it, for <c>--capture</c>.</summary>
    private void Capture(string path)
    {
        var size = new PixelSize((int)Bounds.Width, (int)Bounds.Height);
        using (var bitmap = new RenderTargetBitmap(size, new Vector(96, 96)))
        {
            bitmap.Render(this);
            bitmap.Save(path, new PngBitmapEncoderOptions());
        }

        Close();
    }
}
