using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Layout;
using Avalonia.Media;
using MediaPipeNet.Gallery.Controls;
using MediaPipeNet.Gallery.Services;
using MediaPipeNet.Imaging;

namespace MediaPipeNet.Gallery.Views;

/// <summary>Overview: the thesis (a live holistic result on the stage) and a card per task.</summary>
public sealed class HomeView : IGalleryPage
{
    private readonly Func<string, IGalleryPage> _navigate;
    private readonly StageView _stage = new() { Message = Loc.T("task.running") };
    private bool _ready;

    public HomeView(Func<string, IGalleryPage> navigate)
    {
        _navigate = navigate;
        View = Build();
        _ = RenderHeroAsync();
    }

    public Control View { get; }

    private Control Build()
    {
        var intro = Ui.Stack(Orientation.Vertical, 10,
            Ui.Text(Loc.T("home.eyebrow"), "eyebrow"),
            new TextBlock { Text = Loc.T("home.title"), FontSize = 40, LineHeight = 50, TextWrapping = TextWrapping.Wrap }.With("display"),
            new TextBlock { Text = Loc.T("home.lede"), MaxWidth = 460, HorizontalAlignment = HorizontalAlignment.Left }.With("lede"),
            Ui.Stack(Orientation.Horizontal, 10,
                Ui.Button(Loc.Pick("Try face mesh", "Coba face mesh"), "primary", (_, _) => _navigate("face-mesh")),
                Ui.Button(Loc.Pick("Open the live camera", "Buka kamera langsung"), "quiet", (_, _) => _navigate("live"))));
        intro.VerticalAlignment = VerticalAlignment.Center;
        intro.Margin = new Thickness(0, 0, 32, 0);

        var hero = new Grid { ColumnDefinitions = new ColumnDefinitions("0.9*,1.1*"), Height = 440 };
        hero.Children.Add(intro);
        var stage = new Border { Child = _stage }.With("stage");
        Grid.SetColumn(stage, 1);
        hero.Children.Add(stage);

        var cards = new StackPanel { Spacing = 4, Margin = new Thickness(0, 30, 0, 0) };
        cards.Children.Add(Ui.Text(Loc.T("home.tasks"), "section"));
        var grid = new UniformGrid { Columns = 3 };
        foreach (var t in TaskCatalog.All) grid.Children.Add(Card(t.Title, t.Blurb, t.ModelLabel, t.Id));
        cards.Children.Add(grid);
        cards.Children.Add(Ui.Text(Loc.T("home.more"), "section"));
        var tools = new UniformGrid { Columns = 3 };
        tools.Children.Add(Card(Loc.T("nav.live"), Loc.T("live.lede"), "LiveStreamProcessor · OpenCvSharp", "live"));
        tools.Children.Add(Card(Loc.T("nav.graph"), Loc.Pick("Wire calculator nodes into your own pipeline, in code or .pbtxt.", "Rangkai node kalkulator menjadi pipeline sendiri, lewat kode atau .pbtxt."), "CalculatorGraph · Packet<T>", "graph"));
        tools.Children.Add(Card(Loc.T("nav.benchmark"), Loc.T("bench.lede"), "640×480 · per task", "benchmark"));
        cards.Children.Add(tools);

        var page = new StackPanel { Margin = new Thickness(32, 30, 20, 30) };
        page.Children.Add(hero);
        page.Children.Add(cards);
        return new ScrollViewer { Content = page };
    }

    private Button Card(string title, string blurb, string model, string key)
    {
        var content = Ui.Stack(Orientation.Vertical, 6,
            Ui.Text(title, "title"),
            new TextBlock { Text = blurb, MaxLines = 3, TextTrimming = TextTrimming.CharacterEllipsis, Height = 60 }.With("muted"),
            Ui.Text(model, "eyebrow"));
        var card = new Button { Content = content, Tag = key }.With("card");
        card.Click += (_, _) => _navigate(key);
        return card;
    }

    private async Task RenderHeroAsync()
    {
        try
        {
            var task = TaskCatalog.Get("holistic");
            var options = new OptionValues(task.Options);
            var path = TaskCatalog.SamplePath("pose.jpg");
            var (bitmapImage, output) = await Task.Run(() =>
            {
                var image = MPImage.Load(path);
                var o = task.Run(TaskEngine.Get(task, options), image, null, options);
                return (image, o);
            });
            _stage.Image = Bitmaps.ToBitmap(bitmapImage);
            bitmapImage.Dispose();
            _stage.Overlay = output.Overlay;
            _stage.Message = null;
            _stage.Readout = $"holistic  ·  pose + face mesh + hands  ·  {TaskEngine.ProviderLabel}";
        }
        catch (Exception e) when (e is MediaPipeException or IOException)
        {
            _stage.Message = e.Message;
        }
        _ready = true;
    }

    public async Task PrepareForScreenshotAsync()
    {
        while (!_ready) await Task.Delay(100);
    }

    public void Deactivate() { }
}
