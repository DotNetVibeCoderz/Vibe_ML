using System.Collections.Concurrent;
using System.Text;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media;
using MediaPipeNet.Framework;
using MediaPipeNet.Framework.Config;
using MediaPipeNet.Framework.Nodes;
using MediaPipeNet.Gallery.Controls;
using MediaPipeNet.Gallery.Services;
using MediaPipeNet.Imaging;
using MediaPipeNet.Serialization;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Tasks.Vision.Graph;

namespace MediaPipeNet.Gallery.Views;

/// <summary>Graph API page: a live schematic of a custom graph, its output, and an editable .pbtxt config.</summary>
public sealed class GraphView : IGalleryPage
{
    private const string DefaultPbtxt = """
        # Two tasks side by side on one image stream.
        input_stream: "image"
        output_stream: "objects"
        output_stream: "labels"

        node {
          calculator: "ObjectDetectorCalculator"
          input_stream: "IMAGE:image"
          output_stream: "RESULT:objects"
          options { score_threshold: 0.4 }
        }
        node {
          calculator: "ImageClassifierCalculator"
          input_stream: "IMAGE:image"
          output_stream: "RESULT:labels"
          options { max_results: 3 }
        }
        """;

    private readonly GraphDiagram _diagram = new() { Height = 250 };
    private readonly StageView _stage = new();
    private readonly TextBox _pbtxt = new() { Text = DefaultPbtxt, AcceptsReturn = true, FontSize = 12, MinHeight = 300 };
    private readonly SelectableTextBlock _log = new SelectableTextBlock { TextWrapping = TextWrapping.Wrap }.With("mono");
    private bool _done;

    public GraphView()
    {
        _pbtxt.FontFamily = new FontFamily("avares://MediaPipeNet.Gallery/Assets/Fonts#JetBrains Mono");
        View = Build();
        _diagram.SetGraph(BuildDemoGraph(out _)); // never started: nothing to dispose
    }

    public Control View { get; }

    private Control Build()
    {
        var header = Ui.Header("MEDIAPIPENET.FRAMEWORK · CALCULATORGRAPH", Loc.T("graph.title"), Loc.T("graph.lede"), 900);
        var run = Ui.Button(Loc.T("graph.run"), "primary", async (_, _) => await RunDemoAsync());
        var left = new DockPanel();
        var top = Ui.Stack(Orientation.Vertical, 10, new Border { Child = _diagram, CornerRadius = new CornerRadius(10), ClipToBounds = true }, run);
        top.Margin = new Thickness(0, 0, 0, 12);
        DockPanel.SetDock(top, Dock.Top);
        left.Children.Add(top);
        left.Children.Add(new Border { Child = _stage, MinHeight = 180 }.With("stage"));

        var runConfig = Ui.Button(Loc.T("graph.runpbtxt"), "quiet", async (_, _) => await RunConfigAsync());
        var right = Ui.Panel(Ui.Stack(Orientation.Vertical, 8,
            Ui.Text(Loc.T("graph.pbtxt").ToUpperInvariant(), "section"),
            _pbtxt, runConfig,
            new ScrollViewer { Content = _log, MaxHeight = 220 }));
        right.Width = 420;
        right.Margin = new Thickness(16, 0, 0, 0);

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

    private static CalculatorGraph BuildDemoGraph(out GraphBuilder builder)
    {
        var b = TaskEngine.BaseOptions;
        builder = new GraphBuilder { Options = new GraphOptions { MaxInFlight = 2 } };
        builder.AddInputStream<MPImage>("image");
        builder.AddNode("face_detector", new VisionTaskNode<FaceDetector, FaceDetectionResult>(
                ct => FaceDetector.CreateAsync(new() { BaseOptions = b, RunningMode = RunningMode.Video }, ct), (t, i, ts) => t.DetectForVideo(i, ts)))
            .In("IMAGE", "image").Out("RESULT", "faces");
        builder.AddNode("hand_landmarker", new VisionTaskNode<HandLandmarker, HandLandmarkResult>(
                ct => HandLandmarker.CreateAsync(new() { BaseOptions = b, RunningMode = RunningMode.Video }, ct), (t, i, ts) => t.DetectForVideo(i, ts)))
            .In("IMAGE", "image").Out("RESULT", "hands");
        builder.AddNode("summary", new CombineNode<FaceDetectionResult, HandLandmarkResult, string>(
                (f, h) => $"{f?.Detections.Count ?? 0} face(s), {h?.Hands.Count ?? 0} hand(s)"))
            .In("A", "faces").In("B", "hands").Out("OUT", "summary");
        builder.AddNode("anonymizer", new PixelateFacesNode(16)).In("IMAGE", "image").In("FACES", "faces").Out("IMAGE", "anonymized");
        builder.AddOutputStream("summary").AddOutputStream("anonymized").AddOutputStream("hands");
        return builder.Build();
    }

    private async Task RunDemoAsync()
    {
        using var image = MPImage.Load(TaskCatalog.SamplePath("portrait.jpg"));
        await using var graph = BuildDemoGraph(out _);
        string summary = "";
        MPImage? anonymized = null;
        HandLandmarkResult? hands = null;
        graph.ObserveOutputStream<string>("summary", p => summary = p.Value);
        graph.ObserveOutputStream<MPImage>("anonymized", p => anonymized = p.Value);
        graph.ObserveOutputStream<HandLandmarkResult>("hands", p => hands = p.Value);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        await graph.StartAsync();
        graph.AddPacket("image", image, 0);
        await graph.CloseAsync();
        if (anonymized is not null)
        {
            _stage.Image = Bitmaps.ToBitmap(anonymized);
            anonymized.Dispose();
        }
        var overlay = new Overlay();
        foreach (var h in hands?.Hands ?? []) overlay.AddSkeleton(h.Landmarks, Connections.Hand, Overlay.Mint, Overlay.Paper, 2.4, 3);
        _stage.Overlay = overlay;
        _stage.Readout = $"{sw.Elapsed.TotalMilliseconds:0} ms incl. model load  ·  {summary}  ·  4 nodes, {graph.StreamNames.Count} streams";
        _diagram.SetActive(graph.NodeNames);
        _done = true;
    }

    private async Task RunConfigAsync()
    {
        var log = new StringBuilder();
        try
        {
            var config = GraphConfig.ParsePbtxt(_pbtxt.Text ?? "");
            var registry = new CalculatorRegistry().AddVisionCalculators(TaskEngine.BaseOptions);
            await using var graph = config.Build(registry);
            _diagram.SetGraph(graph);
            var outputs = new ConcurrentQueue<string>();
            foreach (var stream in config.OutputStreams)
                graph.ObserveOutputStream<object>(stream, p => outputs.Enqueue($"{stream} @ {p.Timestamp}:\n{MediaPipeJson.Serialize(p.Value)}"));
            await graph.StartAsync();
            using var image = MPImage.Load(TaskCatalog.SamplePath("cats_and_dogs.jpg"));
            _stage.Image = Bitmaps.ToBitmap(image);
            _stage.Overlay = null;
            foreach (var input in config.InputStreams) graph.AddPacket(input, image, 0);
            await graph.CloseAsync();
            _diagram.SetActive(graph.NodeNames);
            foreach (var o in outputs) log.AppendLine(o.Length > 600 ? o[..600] + " …" : o).AppendLine();
        }
        catch (Exception e) when (e is FormatException or MediaPipeException or ArgumentException or InvalidOperationException)
        {
            log.AppendLine(e.Message);
        }
        _log.Text = log.ToString();
    }

    public async Task PrepareForScreenshotAsync()
    {
        await RunDemoAsync();
        while (!_done) await Task.Delay(50);
    }

    public void Deactivate() { }

    /// <summary>A custom calculator: pixelates every detected face.</summary>
    private sealed class PixelateFacesNode(int block) : CalculatorNode
    {
        public override void GetContract(CalculatorContract contract) =>
            contract.AddInput<MPImage>("IMAGE").AddInput<FaceDetectionResult>("FACES").AddOutput<MPImage>("IMAGE");

        protected override void Process(CalculatorContext context)
        {
            var copy = context.GetInput<MPImage>("IMAGE").Clone();
            if (context.TryGetInput<FaceDetectionResult>("FACES", out var faces))
            {
                var px = copy.GetPixelSpan();
                foreach (var face in faces.Detections)
                {
                    var box = face.BoundingBox.Clamp(copy.Width, copy.Height);
                    for (int by = (int)box.Y; by < box.Bottom; by += block)
                        for (int bx = (int)box.X; bx < box.Right; bx += block)
                        {
                            var sample = px[Math.Min(by + block / 2, copy.Height - 1) * copy.Width + Math.Min(bx + block / 2, copy.Width - 1)];
                            for (int y = by; y < Math.Min(by + block, (int)box.Bottom); y++)
                                for (int x = bx; x < Math.Min(bx + block, (int)box.Right); x++)
                                    px[y * copy.Width + x] = sample;
                        }
                }
            }
            context.Send("IMAGE", copy);
        }
    }
}
