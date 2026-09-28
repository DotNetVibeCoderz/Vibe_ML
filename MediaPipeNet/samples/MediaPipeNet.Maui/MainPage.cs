using System.Diagnostics;
using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Tasks;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Visualization;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using Image = Microsoft.Maui.Controls.Image;

namespace MediaPipeNet.Maui;

/// <summary>Pick a photo (or use a bundled sample), choose a task and a provider, see the result and its latency.</summary>
public sealed class MainPage : ContentPage
{
    private static readonly string[] s_tasks = ["Face stylizer", "Face mesh", "Object detection"];
    private readonly Picker _task = new() { Title = "Task", ItemsSource = s_tasks, SelectedIndex = 0 };
    private readonly Picker _provider = new() { Title = "Execution provider", ItemsSource = ProviderNames(), SelectedIndex = 0 };
    private readonly Image _image = new() { Aspect = Aspect.AspectFit, HeightRequest = 420 };
    private readonly Label _status = new() { Text = "Pick a photo, or press Run for a bundled sample.", FontSize = 14 };
    private byte[]? _photo;

    public MainPage()
    {
        Title = "MediaPipe.NET";
        var pick = new Button { Text = "Pick photo" };
        pick.Clicked += async (_, _) => await PickAsync();
        var run = new Button { Text = "Run" };
        run.Clicked += async (_, _) => await RunAsync();
        Content = new ScrollView
        {
            Content = new VerticalStackLayout
            {
                Padding = 16,
                Spacing = 12,
                Children =
                {
                    new Label { Text = "Google MediaPipe tasks, native in .NET, on your phone.", FontSize = 18 },
                    _task,
                    _provider,
                    new HorizontalStackLayout { Spacing = 12, Children = { pick, run } },
                    _image,
                    _status,
                    new Label { Text = "Created by Gravicode Studios, led by Kang Fadhil.", FontSize = 11, Opacity = 0.6 },
                },
            },
        };
    }

    // NNAPI on Android and CoreML on iOS appear here when the native ONNX Runtime offers them.
    private static string[] ProviderNames() =>
        ["Auto", .. ExecutionProviderSelector.GetAvailableProviders().Select(p => p.ToString())];

    private async Task PickAsync()
    {
        var file = (await MediaPicker.Default.PickPhotosAsync(new MediaPickerOptions { SelectionLimit = 1 })).FirstOrDefault();
        if (file is null) return;
        await using var stream = await file.OpenReadAsync();
        using var buffer = new MemoryStream();
        await stream.CopyToAsync(buffer);
        _photo = buffer.ToArray();
        _status.Text = $"Picked {file.FileName}.";
    }

    private async Task RunAsync()
    {
        try
        {
            _status.Text = "Running...";
            var provider = Enum.TryParse<ExecutionProvider>((string)_provider.SelectedItem, out var p) ? p : ExecutionProvider.Auto;
            var options = await ModelInstaller.InstallAsync(provider);
            int task = _task.SelectedIndex;
            byte[] photo = _photo ?? await ModelInstaller.ReadSampleAsync(task == 2 ? "cats_and_dogs.jpg" : "portrait.jpg");
            var (png, summary) = await Task.Run(() => Process(task, options, photo));
            _image.Source = ImageSource.FromStream(() => new MemoryStream(png));
            _status.Text = summary;
        }
        catch (Exception e)
        {
            _status.Text = $"{e.GetType().Name}: {e.Message}";
        }
    }

    private static (byte[] Png, string Summary) Process(int task, BaseOptions options, byte[] photo)
    {
        using var image = MPImage.Load(photo);
        using var canvas = image.ToImage();
        var watch = new Stopwatch();
        string summary;
        switch (task)
        {
            case 0:
            {
                using var stylizer = FaceStylizer.Create(new() { BaseOptions = options });
                watch.Start();
                using var result = stylizer.Stylize(image);
                watch.Stop();
                using var composite = result.Composite(image);
                if (composite is not null)
                {
                    using var pasted = composite.ToImage();
                    canvas.Mutate(c => c.DrawImage(pasted, 1f));
                }
                summary = result.HasFace ? "Face stylized (color sketch)" : "No face found";
                break;
            }
            case 1:
            {
                using var landmarker = FaceLandmarker.Create(new() { BaseOptions = options });
                watch.Start();
                var result = landmarker.Detect(image);
                watch.Stop();
                ResultRenderer.Render(canvas, result);
                summary = $"{result.Faces.Count} face(s), {FaceLandmarker.LandmarkCount} landmarks each";
                break;
            }
            default:
            {
                using var detector = ObjectDetector.Create(new() { BaseOptions = options });
                watch.Start();
                var result = detector.Detect(image);
                watch.Stop();
                ResultRenderer.Render(canvas, result);
                summary = $"{result.Detections.Count} object(s): {string.Join(", ", result.Detections.Select(d => d.TopCategory.CategoryName))}";
                break;
            }
        }
        using var png = new MemoryStream();
        canvas.SaveAsPng(png);
        return (png.ToArray(), $"{summary} - {watch.Elapsed.TotalMilliseconds:F0} ms ({options.Inference.Provider})");
    }
}
