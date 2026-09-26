using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media;
using SixLabors.ImageSharp.Processing;
using MediaPipeNet.Gallery.Controls;
using MediaPipeNet.Gallery.Services;
using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;

namespace MediaPipeNet.Gallery.Views;

/// <summary>Per-task latency on this machine at 640×480.</summary>
public sealed class BenchmarkView : IGalleryPage
{
    private readonly StackPanel _rows = new() { Spacing = 10 };
    private readonly NumericUpDown _iterations = new() { Minimum = 3, Maximum = 200, Value = 20, Width = 120, FormatString = "0" };
    private readonly TextBlock _status = new TextBlock().With("muted");
    private bool _done;

    public BenchmarkView() => View = Build();

    public Control View { get; }

    private Control Build()
    {
        var header = Ui.Header("640×480 · END TO END", Loc.T("bench.title"), Loc.T("bench.lede"));
        var run = Ui.Button(Loc.T("bench.run"), "primary", async (_, _) => await RunAsync((int)(_iterations.Value ?? 20)));
        var controls = Ui.Stack(Orientation.Horizontal, 12, Ui.Text(Loc.T("bench.iterations"), "muted"), _iterations, run, _status);
        foreach (var c in controls.Children) c.VerticalAlignment = VerticalAlignment.Center;
        controls.Margin = new Thickness(0, 18, 0, 18);
        var page = new StackPanel { Margin = new Thickness(32, 26, 32, 24) };
        page.Children.Add(header);
        page.Children.Add(controls);
        page.Children.Add(Ui.Panel(Ui.Stack(Orientation.Vertical, 8, Ui.Text(Loc.T("bench.target"), "eyebrow"), _rows)));
        return new ScrollViewer { Content = page };
    }

    private async Task RunAsync(int iterations)
    {
        _rows.Children.Clear();
        var results = new List<(string Title, double Mean, double P95)>();
        foreach (var task in TaskCatalog.All)
        {
            _status.Text = $"{task.Title}…";
            var (mean, p95) = await Task.Run(() =>
            {
                using var source = MPImage.Load(TaskCatalog.SamplePath(task.Samples[0]));
                using var sharp = source.ToImage();
                sharp.Mutate(x => x.Resize(new SixLabors.ImageSharp.Processing.ResizeOptions { Size = new SixLabors.ImageSharp.Size(640, 480), Mode = SixLabors.ImageSharp.Processing.ResizeMode.Pad }));
                using var image = MPImage.FromImage(sharp);
                var options = new OptionValues(task.Options);
                var instance = TaskEngine.Get(task, options);
                for (int i = 0; i < 2; i++) task.Run(instance, image, null, options);
                var times = new double[iterations];
                for (int i = 0; i < iterations; i++)
                {
                    long t0 = Stopwatch.GetTimestamp();
                    task.Run(instance, image, null, options);
                    times[i] = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                }
                Array.Sort(times);
                return (Mean: times.Average(), P95: times[(int)Math.Min(iterations - 1, iterations * 0.95)]);
            });
            results.Add((task.Title, mean, p95));
            Render(results);
        }
        _status.Text = $"{TaskEngine.ProviderLabel} · {Environment.ProcessorCount} logical cores";
        _done = true;
    }

    private void Render(List<(string Title, double Mean, double P95)> results)
    {
        _rows.Children.Clear();
        double max = Math.Max(60, results.Max(r => r.P95));
        foreach (var (title, mean, p95) in results)
        {
            var grid = new Grid { ColumnDefinitions = new ColumnDefinitions("200,*,250") };
            grid.Children.Add(Ui.Text(title));
            var bar = new Grid { Height = 14, VerticalAlignment = VerticalAlignment.Center };
            var fill = new Border { CornerRadius = new CornerRadius(3), HorizontalAlignment = HorizontalAlignment.Left };
            fill.Bind(Border.BackgroundProperty, fill.GetResourceObservable("CobaltBrush"));
            var target = new Border { Width = 1.5, HorizontalAlignment = HorizontalAlignment.Left, Background = new SolidColorBrush(Color.Parse("#FF5A36")) };
            bar.Children.Add(fill);
            bar.Children.Add(target);
            bar.SizeChanged += (_, e) =>
            {
                fill.Width = mean / max * e.NewSize.Width;
                target.Margin = new Thickness(50 / max * e.NewSize.Width, 0, 0, 0);
            };
            Grid.SetColumn(bar, 1);
            grid.Children.Add(bar);
            var value = Ui.Text($"{mean,6:0.0} ms · p95 {p95:0.0} · {1000 / mean:0} fps", "mono");
            value.HorizontalAlignment = HorizontalAlignment.Right;
            value.TextWrapping = TextWrapping.NoWrap;
            Grid.SetColumn(value, 2);
            grid.Children.Add(value);
            _rows.Children.Add(grid);
        }
    }

    public async Task PrepareForScreenshotAsync()
    {
        await RunAsync(8);
        while (!_done) await Task.Delay(50);
    }

    public void Deactivate() { }
}

/// <summary>Model catalog: status, checksums and downloads.</summary>
public sealed class ModelsView : IGalleryPage
{
    private readonly StackPanel _rows = new() { Spacing = 0 };
    private readonly ProgressBar _progress = new() { Width = 220, IsVisible = false };
    private readonly TextBlock _status = new TextBlock().With("muted");
    private readonly Dictionary<string, TextBlock> _state = new();

    public ModelsView()
    {
        View = Build();
        Refresh();
    }

    public Control View { get; }

    private static ModelStore Store => ModelStore.CreateDefault(AppSettings.Current.ModelDirectory);

    private Control Build()
    {
        var header = Ui.Header("GRAVICODE.MEDIAPIPENET.MODELS.* · APACHE-2.0", Loc.T("models.title"), Loc.T("models.lede"));
        var controls = Ui.Stack(Orientation.Horizontal, 10,
            Ui.Button(Loc.T("models.verify"), "primary", async (_, _) => await VerifyAsync()),
            Ui.Button(Loc.T("models.download"), "quiet", async (_, _) => await DownloadAsync()),
            _progress, _status);
        foreach (var c in controls.Children) c.VerticalAlignment = VerticalAlignment.Center;
        controls.Margin = new Thickness(0, 18, 0, 18);
        var page = new StackPanel { Margin = new Thickness(32, 26, 32, 24) };
        page.Children.Add(header);
        page.Children.Add(controls);
        page.Children.Add(Ui.Panel(_rows));
        return new ScrollViewer { Content = page };
    }

    private void Refresh()
    {
        _rows.Children.Clear();
        var store = Store;
        var head = Row("MODEL", "SIZE", "PACKAGE", "STATUS", header: true);
        _rows.Children.Add(head);
        foreach (var m in ModelCatalog.All)
        {
            var local = store.FindLocal(m);
            var row = Row(m.Title + "\n" + m.FileName, $"{m.SizeBytes / 1048576.0:0.0} MB", m.PackageId, local is null ? Loc.T("models.missing") : Loc.T("models.local"));
            _state[m.Id] = (TextBlock)row.Children[3];
            _rows.Children.Add(row);
        }
    }

    private static Grid Row(string a, string b, string c, string d, bool header = false)
    {
        var g = new Grid { ColumnDefinitions = new ColumnDefinitions("2*,90,2*,*"), Margin = new Thickness(0, header ? 0 : 8, 0, 0) };
        string cls = header ? "section" : "";
        var cells = new[] { Ui.Text(a, header ? cls : null), Ui.Text(b, header ? cls : "mono"), Ui.Text(c, header ? cls : "mono"), Ui.Text(d, header ? cls : "mono") };
        for (int i = 0; i < cells.Length; i++)
        {
            Grid.SetColumn(cells[i], i);
            g.Children.Add(cells[i]);
        }
        return g;
    }

    private async Task VerifyAsync()
    {
        var store = Store;
        int ok = 0;
        foreach (var m in ModelCatalog.All)
        {
            var path = store.FindLocal(m);
            if (path is null) continue;
            bool good = string.Equals(await ModelStore.ComputeSha256Async(path), m.Sha256, StringComparison.OrdinalIgnoreCase);
            if (good) ok++;
            _state[m.Id].Text = good ? "✓ sha-256 " + m.Sha256[..10] : "✗ checksum mismatch";
        }
        _status.Text = $"{ok}/{ModelCatalog.All.Count} verified";
    }

    private async Task DownloadAsync()
    {
        _progress.IsVisible = true;
        var progress = new Progress<ModelDownloadProgress>(p =>
        {
            _progress.Value = (p.Fraction ?? 0) * 100;
            _status.Text = $"{p.Model.Id}: {p.Stage}";
        });
        try
        {
            await Store.EnsureModelsAsync(ModelCatalog.All, progress);
            _status.Text = "✓";
        }
        catch (ModelNotFoundException e)
        {
            _status.Text = e.Message;
        }
        _progress.IsVisible = false;
        Refresh();
    }

    public Task PrepareForScreenshotAsync() => VerifyAsync();

    public void Deactivate() { }
}

/// <summary>Provider, threads, language, theme and model folder.</summary>
public sealed class SettingsView : IGalleryPage
{
    public SettingsView() => View = Build();

    public Control View { get; }

    private static Control Build()
    {
        var s = AppSettings.Current;
        var providers = new[] { ExecutionProvider.Auto, ExecutionProvider.Cpu, ExecutionProvider.DirectML, ExecutionProvider.Cuda, ExecutionProvider.CoreML };
        var provider = new ComboBox { ItemsSource = providers, SelectedItem = s.Provider, Width = 260 };
        var threads = new NumericUpDown { Minimum = 0, Maximum = Environment.ProcessorCount, Value = s.Threads, Width = 260, FormatString = "0" };
        var language = new ComboBox { ItemsSource = new[] { "English", "Bahasa Indonesia" }, SelectedIndex = s.Language == "id" ? 1 : 0, Width = 260 };
        var theme = new ComboBox { ItemsSource = new[] { "Light", "Dark" }, SelectedIndex = s.Theme == "Dark" ? 1 : 0, Width = 260 };
        var modelDir = new TextBox { Text = s.ModelDirectory, Width = 420, Watermark = ModelStore.DefaultCacheDirectory };
        var points = new ToggleSwitch { Content = Loc.T("settings.points"), IsChecked = s.ShowLandmarkPoints };
        var saved = new TextBlock { IsVisible = false, Text = Loc.T("settings.saved") }.With("muted");
        var save = Ui.Button(Loc.T("settings.save"), "primary", (_, _) =>
        {
            s.Provider = (ExecutionProvider)provider.SelectedItem!;
            s.Threads = (int)(threads.Value ?? 0);
            s.Language = language.SelectedIndex == 1 ? "id" : "en";
            s.Theme = theme.SelectedIndex == 1 ? "Dark" : "Light";
            s.ModelDirectory = string.IsNullOrWhiteSpace(modelDir.Text) ? null : modelDir.Text;
            s.ShowLandmarkPoints = points.IsChecked == true;
            s.Save();
        });

        Control Field(string label, Control editor, string? help = null)
        {
            var st = Ui.Stack(Orientation.Vertical, 5, Ui.Text(label, "title"), editor);
            if (help is not null) st.Children.Add(Ui.Text(help, "muted"));
            st.Margin = new Thickness(0, 0, 0, 16);
            return st;
        }

        var form = Ui.Stack(Orientation.Vertical, 0,
            Field(Loc.T("settings.provider"), provider, Loc.T("settings.provider.help")),
            Field(Loc.T("settings.threads"), threads),
            Field(Loc.T("settings.language"), language),
            Field(Loc.T("settings.theme"), theme),
            Field(Loc.T("settings.modeldir"), modelDir),
            points,
            Ui.Stack(Orientation.Horizontal, 12, save, saved));

        var runtime = Ui.Stack(Orientation.Vertical, 6,
            Ui.Text(Loc.T("settings.runtime"), "section"),
            Ui.Text($"ONNX Runtime {ExecutionProviderSelector.GetRuntimeVersion() ?? "—"}", "mono"),
            Ui.Text($"Providers: {string.Join(", ", ExecutionProviderSelector.GetAvailableProviders())}", "mono"),
            Ui.Text($".NET {Environment.Version} · {System.Runtime.InteropServices.RuntimeInformation.OSDescription}", "mono"),
            Ui.Text($"Model cache: {ModelStore.DefaultCacheDirectory}", "mono"),
            Ui.Text($"MediaPipe.NET {ModelStore.LibraryVersion}", "mono"));

        var grid = new Grid { ColumnDefinitions = new ColumnDefinitions("*,420"), Margin = new Thickness(0, 22, 0, 0) };
        grid.Children.Add(Ui.Panel(form));
        var rt = Ui.Panel(runtime);
        rt.Margin = new Thickness(16, 0, 0, 0);
        rt.VerticalAlignment = VerticalAlignment.Top;
        Grid.SetColumn(rt, 1);
        grid.Children.Add(rt);

        var page = new StackPanel { Margin = new Thickness(32, 26, 32, 24) };
        page.Children.Add(Ui.Header("PREFERENCES", Loc.T("settings.title"),
            Loc.Pick("Choose where models run and how the Gallery looks. Changes apply to every task after saving.",
                     "Pilih tempat model dijalankan dan tampilan Gallery. Perubahan berlaku untuk semua task setelah disimpan.")));
        page.Children.Add(grid);
        return new ScrollViewer { Content = page };
    }

    public Task PrepareForScreenshotAsync() => Task.CompletedTask;

    public void Deactivate() { }
}

/// <summary>Credits and licenses.</summary>
public sealed class AboutView : IGalleryPage
{
    public AboutView()
    {
        var logo = new Image { Source = new Avalonia.Media.Imaging.Bitmap(Avalonia.Platform.AssetLoader.Open(new Uri("avares://MediaPipeNet.Gallery/Assets/logo.png"))), Width = 96, Height = 96, HorizontalAlignment = HorizontalAlignment.Left };
        var body = Ui.Stack(Orientation.Vertical, 12,
            logo,
            Ui.Text("MediaPipe.NET", "display"),
            new TextBlock
            {
                Text = Loc.Pick(
                    "A native .NET 10 port of Google MediaPipe's vision tasks. Models are converted from the official TFLite releases to ONNX and cross-validated against the TFLite interpreter and the official MediaPipe Python package.",
                    "Port native .NET 10 untuk task vision Google MediaPipe. Model dikonversi dari rilis TFLite resmi ke ONNX dan divalidasi silang dengan interpreter TFLite serta paket resmi MediaPipe Python."),
                MaxWidth = 720, HorizontalAlignment = HorizontalAlignment.Left,
            }.With("lede"),
            Ui.Text(Loc.T("footer.credit"), "title"),
            Ui.Text("LICENSES", "section"),
            Ui.Text("MediaPipe.NET — Apache-2.0 · © 2026 Gravicode Studios", "mono"),
            Ui.Text("Model weights — Apache-2.0 · © Google LLC (MediaPipe)", "mono"),
            Ui.Text("ONNX Runtime — MIT · SixLabors.ImageSharp — Six Labors Split License · OpenCvSharp — Apache-2.0", "mono"),
            Ui.Text("Avalonia — MIT · Unbounded, JetBrains Mono, Inter — SIL Open Font License", "mono"));
        View = new ScrollViewer { Content = new Border { Child = body, Margin = new Thickness(32, 30) } };
    }

    public Control View { get; }

    public Task PrepareForScreenshotAsync() => Task.CompletedTask;

    public void Deactivate() { }
}
