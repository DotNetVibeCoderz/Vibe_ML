using Avalonia;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using MediaPipeNet.Gallery.Controls;
using MediaPipeNet.Gallery.Services;
using MediaPipeNet.Imaging;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Video.OpenCv;

namespace MediaPipeNet.Gallery.Views;

/// <summary>Webcam page: any task in video mode through <see cref="LiveStreamProcessor{TResult}"/>.</summary>
public sealed class LiveView : IGalleryPage
{
    private readonly StageView _stage = new() { Message = Loc.T("live.idle") };
    private readonly ComboBox _taskBox;
    private readonly NumericUpDown _camera = new() { Minimum = 0, Maximum = 8, Value = AppSettings.Current.CameraIndex, Width = 110, FormatString = "0" };
    private readonly ToggleSwitch _mirror = new() { Content = Loc.T("live.mirror"), IsChecked = AppSettings.Current.MirrorCamera };
    private readonly Button _start;
    private readonly StackPanel _results = new() { Spacing = 2 };
    private CancellationTokenSource? _cts;
    private WriteableBitmap? _bitmap;
    private MPImage? _display;
    private int _uiPending;

    public LiveView()
    {
        var tasks = TaskCatalog.All.Where(t => t.SupportsLive).ToList();
        _taskBox = new ComboBox { ItemsSource = tasks.Select(t => t.Title).ToList(), SelectedIndex = tasks.FindIndex(t => t.Id == "hands"), Width = 220, Tag = tasks };
        _start = Ui.Button(Loc.T("live.start"), "primary", async (_, _) => await ToggleAsync());
        View = Build();
    }

    public Control View { get; }

    private Control Build()
    {
        var header = Ui.Header("VIDEO MODE · LIVESTREAMPROCESSOR", Loc.T("live.title"), Loc.T("live.lede"));
        var controls = Ui.Stack(Orientation.Horizontal, 14,
            Ui.Stack(Orientation.Vertical, 4, Ui.Text(Loc.T("live.task"), "muted"), _taskBox),
            Ui.Stack(Orientation.Vertical, 4, Ui.Text(Loc.T("live.camera"), "muted"), _camera),
            _mirror, _start);
        controls.Margin = new Thickness(0, 16, 0, 12);
        foreach (var c in controls.Children) c.VerticalAlignment = VerticalAlignment.Bottom;

        var side = Ui.Panel(new ScrollViewer { Content = new Border { Child = Ui.Stack(Orientation.Vertical, 0, Ui.Text(Loc.T("task.results"), "section"), _results), Margin = new Thickness(0, 0, 14, 0) } });
        side.Width = 300;
        side.Margin = new Thickness(16, 0, 0, 0);
        var body = new DockPanel();
        DockPanel.SetDock(side, Dock.Right);
        body.Children.Add(side);
        body.Children.Add(new Border { Child = _stage }.With("stage"));

        var page = new DockPanel { Margin = new Thickness(32, 26, 32, 24) };
        DockPanel.SetDock(header, Dock.Top);
        DockPanel.SetDock(controls, Dock.Top);
        page.Children.Add(header);
        page.Children.Add(controls);
        page.Children.Add(body);
        return page;
    }

    private async Task ToggleAsync()
    {
        if (_cts is not null)
        {
            Stop();
            return;
        }
        var task = ((List<GalleryTask>)_taskBox.Tag!)[Math.Max(0, _taskBox.SelectedIndex)];
        int camera = (int)(_camera.Value ?? 0);
        AppSettings.Current.CameraIndex = camera;
        AppSettings.Current.MirrorCamera = _mirror.IsChecked == true;
        _cts = new CancellationTokenSource();
        _start.Content = Loc.T("live.stop");
        _stage.Message = Loc.T("task.running");
        var token = _cts.Token;
        try
        {
            await Task.Run(async () =>
            {
                var options = new OptionValues(task.Options);
                using var instance = task.Create(options, MediaPipeNet.RunningMode.Video, TaskEngine.BaseOptions);
                IFrameSource source;
                try { source = new WebcamFrameSource(camera); }
                catch (InvalidOperationException) { throw new IOException(Loc.T("live.nocamera")); }
                await using var processor = new LiveStreamProcessor<GalleryOutput>(source,
                    (frame, ts) => task.Run(instance, frame, ts, options),
                    new LiveStreamProcessorOptions { MirrorFrames = AppSettings.Current.MirrorCamera });
                processor.ResultReady += (_, r) => Present(r, task);
                await processor.RunAsync(token);
            }, token);
        }
        catch (OperationCanceledException) { }
        catch (Exception e) when (e is IOException or MediaPipeException or TypeInitializationException or DllNotFoundException)
        {
            _stage.Message = e.Message;
        }
        finally
        {
            Stop();
        }
    }

    private void Present(LiveFrameResult<GalleryOutput> r, GalleryTask task)
    {
        // Skip UI updates while the previous one is still pending; the model keeps running.
        if (Interlocked.Exchange(ref _uiPending, 1) == 1) return;
        var source = r.Result.ImageOverride ?? r.Frame;
        if (_display is null) _display = source.Clone();
        else _display.CopyFrom(source);
        r.Result.ImageOverride?.Dispose();
        var stats = r.Stats;
        Dispatcher.UIThread.Post(() =>
        {
            _bitmap = Bitmaps.ToBitmap(_display!, _bitmap);
            _stage.Image = null;
            _stage.Image = _bitmap;
            _stage.Overlay = r.Result.Overlay;
            _stage.Message = null;
            _stage.Readout = $"{stats.ProcessingFps,5:0.0} fps  ·  {r.Latency.TotalMilliseconds,5:0.0} ms  ·  camera {stats.CaptureFps:0} fps  ·  dropped {stats.FramesDropped}  ·  {task.ModelLabel}";
            _results.Children.Clear();
            foreach (var row in r.Result.Rows) _results.Children.Add(Ui.Meter(row));
            Interlocked.Exchange(ref _uiPending, 0);
        });
    }

    private void Stop()
    {
        _cts?.Cancel();
        _cts = null;
        _start.Content = Loc.T("live.start");
        if (_stage.Image is null) _stage.Message ??= Loc.T("live.idle");
    }

    public Task PrepareForScreenshotAsync() => Task.CompletedTask;

    public void Deactivate() => Stop();
}
