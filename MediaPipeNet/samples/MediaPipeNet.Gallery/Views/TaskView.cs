using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using MediaPipeNet.Gallery.Controls;
using MediaPipeNet.Gallery.Services;
using MediaPipeNet.Imaging;

namespace MediaPipeNet.Gallery.Views;

/// <summary>The page shared by all nine tasks: stage + samples on the left, options and results on the right.</summary>
public sealed class TaskView : IGalleryPage
{
    private readonly GalleryTask _task;
    private readonly OptionValues _options;
    private readonly StageView _stage = new();
    private readonly StackPanel _results = new() { Spacing = 2 };
    private readonly CodeView _code = new();
    private readonly SelectableTextBlock _json = new SelectableTextBlock { TextWrapping = TextWrapping.NoWrap, Foreground = Brushes.Gainsboro }.With("mono");
    private readonly StackPanel _thumbs = new() { Orientation = Orientation.Horizontal };
    private MPImage? _image;
    private Bitmap? _bitmap;
    private string _imageName = "";
    private int _runId;

    public TaskView(GalleryTask task)
    {
        _task = task;
        _options = new OptionValues(task.Options);
        View = Build();
        _code.SetCode(task.Code(_options));
        _ = LoadAsync(TaskCatalog.SamplePath(task.Samples[0]));
    }

    public Control View { get; }

    private Control Build()
    {
        var header = Ui.Header($"{Loc.T("nav." + _task.Category)} ·{_task.ModelLabel.ToUpperInvariant()}", _task.Title, _task.Blurb);

        // Left: tabs (preview / code / json) + sample strip.
        var stageHost = new Border { Child = _stage, MinHeight = 420 }.With("stage");
        var codeHost = new Border
        {
            Child = new Grid
            {
                Children =
                {
                    new ScrollViewer { Content = _code, HorizontalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Auto },
                    CopyButton(() => _code.CodeText),
                },
            },
        }.With("code");
        var jsonHost = new Border
        {
            Child = new Grid
            {
                Children =
                {
                    new ScrollViewer { Content = _json, HorizontalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Auto },
                    CopyButton(() => _json.Text ?? ""),
                },
            },
        }.With("code");

        var tabs = new TabControl
        {
            ItemsSource = new[]
            {
                new TabItem { Header = Loc.T("task.preview"), Content = stageHost },
                new TabItem { Header = Loc.T("task.code"), Content = codeHost },
                new TabItem { Header = Loc.T("task.json"), Content = jsonHost },
            },
        }.With("switch");

        foreach (var sample in _task.Samples) _thumbs.Children.Add(Thumb(sample));
        var open = Ui.Button(Loc.T("task.open"), "quiet", async (_, _) => await OpenFileAsync());
        var rerun = Ui.Button(Loc.T("task.run"), "quiet", async (_, _) => await RunAsync());
        var strip = new DockPanel { Margin = new Thickness(0, 12, 0, 0) };
        var buttons = Ui.Stack(Orientation.Horizontal, 8, open, rerun);
        buttons.VerticalAlignment = VerticalAlignment.Center;
        DockPanel.SetDock(buttons, Dock.Right);
        strip.Children.Add(buttons);
        strip.Children.Add(Ui.Stack(Orientation.Horizontal, 10, Ui.Text(Loc.T("task.samples"), "section"), _thumbs));

        var left = new DockPanel();
        DockPanel.SetDock(strip, Dock.Bottom);
        left.Children.Add(strip);
        left.Children.Add(tabs);

        // Right: options and results.
        var options = new StackPanel { Spacing = 4, Margin = new Thickness(0, 0, 14, 0) };
        options.Children.Add(Ui.Text(Loc.T("task.options"), "section"));
        foreach (var spec in _task.Options) options.Children.Add(OptionEditor(spec));
        options.Children.Add(Ui.Text(Loc.T("task.results"), "section"));
        options.Children.Add(_results);
        var right = Ui.Panel(new ScrollViewer { Content = options });
        right.Width = 320;
        right.Margin = new Thickness(16, 44, 0, 0);

        var body = new DockPanel { Margin = new Thickness(0, 18, 0, 0) };
        DockPanel.SetDock(right, Dock.Right);
        body.Children.Add(right);
        body.Children.Add(left);

        var page = new DockPanel { Margin = new Thickness(32, 26, 32, 24) };
        DockPanel.SetDock(header, Dock.Top);
        page.Children.Add(header);
        page.Children.Add(body);
        return page;
    }

    private static Button CopyButton(Func<string> text)
    {
        var b = new Button { Content = Loc.T("task.copy"), HorizontalAlignment = HorizontalAlignment.Right, VerticalAlignment = VerticalAlignment.Top }.With("quiet");
        b.Foreground = Brushes.Gainsboro;
        b.Click += async (_, _) =>
        {
            if (TopLevel.GetTopLevel(b)?.Clipboard is { } clipboard) await clipboard.SetTextAsync(text());
            b.Content = Loc.T("task.copied");
            await Task.Delay(1200);
            b.Content = Loc.T("task.copy");
        };
        return b;
    }

    private Button Thumb(string name)
    {
        var path = TaskCatalog.SamplePath(name);
        Bitmap? bmp = null;
        try
        {
            using var s = File.OpenRead(path);
            bmp = Bitmap.DecodeToWidth(s, 120);
        }
        catch (IOException) { }
        var b = new Button
        {
            Content = new Border { Child = new Image { Source = bmp, Stretch = Stretch.UniformToFill, Width = 64, Height = 44 }, CornerRadius = new CornerRadius(6), ClipToBounds = true },
            Tag = name,
        }.With("thumb");
        ToolTip.SetTip(b, name);
        b.Click += async (_, _) => await LoadAsync(path);
        return b;
    }

    private Control OptionEditor(OptionSpec spec)
    {
        switch (spec.Kind)
        {
            case OptionKind.Toggle:
            {
                var t = new ToggleSwitch { Content = spec.Label, IsChecked = _options.B(spec.Key), OnContent = null, OffContent = null };
                t.IsCheckedChanged += async (_, _) => { _options[spec.Key] = t.IsChecked == true ? 1 : 0; await OnOptionsChangedAsync(); };
                return t;
            }
            case OptionKind.Choice:
            {
                var c = new ComboBox { ItemsSource = spec.Choices, SelectedIndex = (int)spec.Default, HorizontalAlignment = HorizontalAlignment.Stretch };
                c.SelectionChanged += async (_, _) => { _options[spec.Key] = c.SelectedIndex; await OnOptionsChangedAsync(); };
                return Ui.Stack(Orientation.Vertical, 4, Ui.Text(spec.Label), c);
            }
            default:
            {
                var value = Ui.Text(_options[spec.Key].ToString(spec.Format, System.Globalization.CultureInfo.InvariantCulture), "mono");
                value.HorizontalAlignment = HorizontalAlignment.Right;
                var label = new Grid { ColumnDefinitions = new ColumnDefinitions("*,Auto") };
                label.Children.Add(Ui.Text(spec.Label));
                Grid.SetColumn(value, 1);
                label.Children.Add(value);
                var slider = new Slider { Minimum = spec.Min, Maximum = spec.Max, Value = spec.Default, SmallChange = spec.Step, TickFrequency = spec.Step, IsSnapToTickEnabled = true };
                var debounce = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(250) };
                debounce.Tick += async (_, _) => { debounce.Stop(); await OnOptionsChangedAsync(); };
                slider.ValueChanged += (_, e) =>
                {
                    _options[spec.Key] = e.NewValue;
                    value.Text = e.NewValue.ToString(spec.Format, System.Globalization.CultureInfo.InvariantCulture);
                    debounce.Stop();
                    debounce.Start();
                };
                return Ui.Stack(Orientation.Vertical, 0, label, slider);
            }
        }
    }

    private async Task OnOptionsChangedAsync()
    {
        _code.SetCode(_task.Code(_options));
        await RunAsync();
    }

    private async Task OpenFileAsync()
    {
        if (TopLevel.GetTopLevel(View) is not { } top) return;
        var files = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = Loc.T("task.open"),
            AllowMultiple = false,
            FileTypeFilter = [new FilePickerFileType("Images") { Patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"] }],
        });
        if (files.Count > 0 && files[0].TryGetLocalPath() is { } path) await LoadAsync(path);
    }

    public async Task LoadAsync(string path)
    {
        try
        {
            var image = await Task.Run(() => MPImage.Load(path));
            _image?.Dispose();
            _image = image;
            _imageName = Path.GetFileName(path);
            _bitmap = Bitmaps.ToBitmap(image);
            _stage.Image = _bitmap;
            _stage.Overlay = null;
            foreach (var t in _thumbs.Children.OfType<Button>())
                t.Classes.Set("active", (string?)t.Tag == _imageName);
            await RunAsync();
        }
        catch (Exception e) when (e is IOException or SixLabors.ImageSharp.UnknownImageFormatException)
        {
            _stage.Message = e.Message;
        }
    }

    private async Task RunAsync()
    {
        if (_image is null) return;
        int id = ++_runId;
        _stage.Message = Loc.T("task.running");
        var image = _image;
        try
        {
            var (output, ms) = await Task.Run(() =>
            {
                var instance = TaskEngine.Get(_task, _options);
                if (TaskEngine.MarkWarm(instance)) _task.Run(instance, image, null, _options); // first call pays one-time session warm-up
                var sw = Stopwatch.StartNew();
                var o = _task.Run(instance, image, null, _options);
                return (o, sw.Elapsed.TotalMilliseconds);
            });
            if (id != _runId) return;
            if (output.ImageOverride is { } composed)
            {
                _stage.Image = Bitmaps.ToBitmap(composed);
                composed.Dispose();
            }
            else
            {
                _stage.Image = _bitmap;
            }
            _stage.Overlay = output.Overlay;
            _stage.Message = output.IsEmpty ? Loc.T("task.nothing") : null;
            _stage.Readout = $"{ms:0.0} ms  ·  {TaskEngine.ProviderLabel}  ·  {image.Width}×{image.Height}  ·  {output.Summary}";
            _results.Children.Clear();
            foreach (var row in output.Rows) _results.Children.Add(Ui.Meter(row));
            _json.Text = output.Json.Length > 60_000 ? output.Json[..60_000] + "\n…" : output.Json;
        }
        catch (Exception e) when (e is MediaPipeException or InvalidOperationException or IOException)
        {
            if (id == _runId) _stage.Message = Loc.T("task.error") + e.Message;
        }
    }

    public async Task PrepareForScreenshotAsync()
    {
        while (_stage.Message == Loc.T("task.running") || _stage.Overlay is null && _stage.Message is null)
            await Task.Delay(100);
    }

    public void Deactivate() => _runId++;
}
