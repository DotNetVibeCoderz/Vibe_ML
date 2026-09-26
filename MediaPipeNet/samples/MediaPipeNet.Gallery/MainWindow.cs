using Avalonia;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using MediaPipeNet.Gallery.Controls;
using MediaPipeNet.Gallery.Services;
using MediaPipeNet.Gallery.Views;

namespace MediaPipeNet.Gallery;

/// <summary>A page hosted by the main window.</summary>
public interface IGalleryPage
{
    Control View { get; }

    /// <summary>Brings the page into a representative state for an automated screenshot.</summary>
    Task PrepareForScreenshotAsync();

    /// <summary>Called when the user navigates away (stop cameras, cancel work).</summary>
    void Deactivate();
}

public sealed class MainWindow : Window
{
    private readonly ListBox _nav = new ListBox().With("nav");
    private readonly ContentControl _host = new();
    private readonly List<(string Key, ListBoxItem Item)> _items = [];
    private IGalleryPage? _page;
    private string _currentKey = "home";

    public MainWindow()
    {
        Title = "MediaPipe.Net Gallery — Gravicode Studios";
        Width = 1440;
        Height = 900;
        MinWidth = 1080;
        MinHeight = 700;
        WindowStartupLocation = WindowStartupLocation.CenterScreen;
        Icon = new WindowIcon(AssetLoader("Assets/logo.png"));
        BuildShell();
        Navigate("home");
        AppSettings.Changed += () => Dispatcher.UIThread.Post(() =>
        {
            App.ApplyTheme();
            BuildShell();
            Navigate(_currentKey);
        });
        if (App.ScreenshotDirectory is not null) Opened += async (_, _) => await RunScreenshotsAsync(App.ScreenshotDirectory);
    }

    private static Bitmap AssetLoader(string path) =>
        new(Avalonia.Platform.AssetLoader.Open(new Uri($"avares://MediaPipeNet.Gallery/{path}")));

    private void BuildShell()
    {
        _items.Clear();
        _nav.Items.Clear();
        // Rebuilding (language/theme change) reuses the nav list and page host: detach them first.
        if (_nav.Parent is ScrollViewer oldScroll) oldScroll.Content = null;
        if (_host.Parent is Panel oldPanel) oldPanel.Children.Remove(_host);
        void Header(string key) => _nav.Items.Add(new ListBoxItem { Content = Loc.T(key) }.With("header"));
        void Item(string key, string label)
        {
            var item = new ListBoxItem { Content = label, Tag = key };
            _items.Add((key, item));
            _nav.Items.Add(item);
        }

        Item("home", Loc.T("nav.home"));
        foreach (var (category, header) in new[] { ("detect", "nav.detect"), ("landmarks", "nav.landmarks"), ("understand", "nav.understand") })
        {
            Header(header);
            foreach (var t in TaskCatalog.All.Where(t => t.Category == category)) Item(t.Id, t.Title);
        }
        Header("nav.pipelines");
        Item("live", Loc.T("nav.live"));
        Item("graph", Loc.T("nav.graph"));
        Item("benchmark", Loc.T("nav.benchmark"));
        Header("nav.app");
        Item("models", Loc.T("nav.models"));
        Item("settings", Loc.T("nav.settings"));
        Item("about", Loc.T("nav.about"));

        _nav.SelectionChanged -= OnNavSelection;
        _nav.SelectionChanged += OnNavSelection;

        var brand = Ui.Stack(Orientation.Horizontal, 10,
            new Image { Source = AssetLoader("Assets/logo.png"), Width = 34, Height = 34 },
            Ui.Stack(Orientation.Vertical, 0,
                new TextBlock { Text = "MediaPipe.Net", FontFamily = (FontFamily)Application.Current!.Resources["DisplayFont"]!, FontSize = 15 },
                Ui.Text("GALLERY · v0.1", "eyebrow")));
        brand.Margin = new Thickness(18, 20, 18, 10);

        var credit = new TextBlock { Text = Loc.T("footer.credit"), FontSize = 11, TextWrapping = TextWrapping.Wrap, Margin = new Thickness(18, 10, 18, 16) }.With("muted");
        var sidebar = new DockPanel { LastChildFill = true };
        DockPanel.SetDock(brand, Dock.Top);
        DockPanel.SetDock(credit, Dock.Bottom);
        sidebar.Children.Add(brand);
        sidebar.Children.Add(credit);
        sidebar.Children.Add(new ScrollViewer { Content = _nav, Padding = new Thickness(10, 0) });
        var sidebarBorder = new Border { Child = sidebar, Width = 240 };
        sidebarBorder.Bind(Border.BackgroundProperty, sidebarBorder.GetResourceObservable("SidebarBrush"));

        var root = new DockPanel();
        DockPanel.SetDock(sidebarBorder, Dock.Left);
        root.Children.Add(sidebarBorder);
        root.Children.Add(_host);
        Content = root;
    }

    private void OnNavSelection(object? sender, SelectionChangedEventArgs e)
    {
        if (_nav.SelectedItem is ListBoxItem { Tag: string key } && key != _currentKey) Navigate(key);
    }

    public IGalleryPage Navigate(string key)
    {
        _page?.Deactivate();
        _currentKey = key;
        _page = key switch
        {
            "home" => new HomeView(Navigate),
            "live" => new LiveView(),
            "graph" => new GraphView(),
            "benchmark" => new BenchmarkView(),
            "models" => new ModelsView(),
            "settings" => new SettingsView(),
            "about" => new AboutView(),
            _ => new TaskView(TaskCatalog.Get(key)),
        };
        _host.Content = _page.View;
        var item = _items.FirstOrDefault(i => i.Key == key).Item;
        if (item is not null && !ReferenceEquals(_nav.SelectedItem, item)) _nav.SelectedItem = item;
        return _page;
    }

    private async Task RunScreenshotsAsync(string directory)
    {
        Directory.CreateDirectory(directory);
        string[] pages = ["home", .. TaskCatalog.All.Select(t => t.Id), "live", "graph", "benchmark", "models", "settings", "about"];
        foreach (var key in pages) await CaptureAsync(key, Path.Combine(directory, $"gallery-{key}.png"));

        // A dark-theme, Indonesian-language variant shows theming and localization.
        AppSettings.Current.Theme = "Dark";
        AppSettings.Current.Language = "id";
        App.ApplyTheme();
        BuildShell();
        foreach (var key in new[] { "home", "hands", "settings" }) await CaptureAsync(key, Path.Combine(directory, $"gallery-{key}-dark-id.png"));
        Close();
    }

    private async Task CaptureAsync(string key, string path)
    {
        var page = Navigate(key);
        await Task.Delay(300);
        await page.PrepareForScreenshotAsync();
        await Task.Delay(500);
        double scale = RenderScaling;
        var size = new PixelSize((int)(Bounds.Width * scale), (int)(Bounds.Height * scale));
        using var rtb = new RenderTargetBitmap(size, new Vector(96 * scale, 96 * scale));
        rtb.Render(this);
        rtb.Save(path);
    }
}
