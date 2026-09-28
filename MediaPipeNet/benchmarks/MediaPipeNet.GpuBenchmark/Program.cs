// MediaPipe.NET GPU benchmark: median latency per task on the CPU and DirectML providers, in float32, float16 and
// with IoBinding. Prints a Markdown table (docs/en/performance.md).
//
//   dotnet run -c Release --project benchmarks/MediaPipeNet.GpuBenchmark [-- --iterations 30]
//
// Created by Gravicode Studios, led by Kang Fadhil.
using System.Diagnostics;
using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Tasks;
using MediaPipeNet.Tasks.Vision;

int iterations = args.SkipWhile(a => a != "--iterations").Skip(1).Select(int.Parse).FirstOrDefault(30);
string Img(string name) => Path.Combine(AppContext.BaseDirectory, "images", name);

var configs = new (string Name, InferenceOptions Options)[]
{
    ("CPU fp32", new() { Provider = ExecutionProvider.Cpu }),
    ("DirectML fp32", new() { Provider = ExecutionProvider.DirectML, FallbackToCpu = false }),
    ("DirectML fp16", new() { Provider = ExecutionProvider.DirectML, FallbackToCpu = false, Precision = ModelPrecision.Float16 }),
    ("DirectML fp32 + IoBinding", new() { Provider = ExecutionProvider.DirectML, FallbackToCpu = false, UseIoBinding = true }),
};

var tasks = new (string Name, string Image, Func<BaseOptions, Func<MPImage, object>> Create)[]
{
    ("FaceDetector", "portrait.jpg", b => { var t = FaceDetector.Create(new() { BaseOptions = b }); return i => t.Detect(i); }),
    ("FaceLandmarker", "portrait.jpg", b => { var t = FaceLandmarker.Create(new() { BaseOptions = b }); return i => t.Detect(i); }),
    ("HandLandmarker", "thumb_up.jpg", b => { var t = HandLandmarker.Create(new() { BaseOptions = b }); return i => t.Detect(i); }),
    ("PoseLandmarker (full)", "pose.jpg", b => { var t = PoseLandmarker.Create(new() { BaseOptions = b, Model = PoseModel.Full }); return i => t.Detect(i); }),
    ("ImageSegmenter (multiclass)", "portrait.jpg", b => { var t = ImageSegmenter.Create(new() { BaseOptions = b, Model = SegmenterModel.SelfieMulticlass }); return i => t.Segment(i); }),
    ("ObjectDetector", "cats_and_dogs.jpg", b => { var t = ObjectDetector.Create(new() { BaseOptions = b }); return i => t.Detect(i); }),
    ("ImageClassifier", "burger.jpg", b => { var t = ImageClassifier.Create(new() { BaseOptions = b }); return i => t.Classify(i); }),
    ("InteractiveSegmenter", "cats_and_dogs.jpg", b => { var t = InteractiveSegmenter.Create(new() { BaseOptions = b }); return i => t.Segment(i, RegionOfInterest.FromKeypoint(0.6f, 0.5f)); }),
    ("FaceStylizer", "portrait.jpg", b => { var t = FaceStylizer.Create(new() { BaseOptions = b }); return i => { using var r = t.Stylize(i); return r.HasFace; }; }),
};

Console.WriteLine($"ONNX Runtime {ExecutionProviderSelector.GetRuntimeVersion()}, providers: {string.Join(", ", ExecutionProviderSelector.GetAvailableProviders())}");
Console.WriteLine($"CPU: {Environment.ProcessorCount} logical cores; {iterations} iterations, median ms (image mode, pre/post-processing included)");
Console.WriteLine();
Console.WriteLine($"| Task | {string.Join(" | ", configs.Select(c => c.Name))} |");
Console.WriteLine($"|---|{string.Concat(configs.Select(_ => "---:|"))}");
foreach (var (name, image, create) in tasks)
{
    using var frame = MPImage.Load(Img(image));
    var cells = new List<string>();
    string? reference = null;
    foreach (var (_, options) in configs)
    {
        try
        {
            var run = create(new BaseOptions { Inference = options });
            object result = run(frame);
            for (int i = 0; i < 4; i++) run(frame);
            var times = new double[iterations];
            for (int i = 0; i < iterations; i++)
            {
                long start = Stopwatch.GetTimestamp();
                run(frame);
                times[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            }
            Array.Sort(times);
            // A configuration that finds something different from the CPU reference is flagged, not timed as a win.
            string found = Found(result);
            reference ??= found;
            string ms = times[iterations / 2].ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
            cells.Add(found == reference ? ms : $"{ms} ⚠ {found}");
        }
        catch (Exception e)
        {
            cells.Add($"n/a ({e.GetType().Name})");
        }
    }
    Console.WriteLine($"| {name} | {string.Join(" | ", cells)} |");
}

// What a result found, to catch providers that run fast but wrong.
static string Found(object result) => result switch
{
    FaceDetectionResult r => $"{r.Detections.Count} face(s)",
    FaceLandmarkResult r => $"{r.Faces.Count} face(s)",
    HandLandmarkResult r => $"{r.Hands.Count} hand(s)",
    PoseLandmarkResult r => $"{r.Poses.Count} pose(s)",
    ObjectDetectionResult r => $"{r.Detections.Count} object(s)",
    ClassificationResult r => r.Categories.FirstOrDefault()?.CategoryName ?? "none",
    bool b => b ? "stylized" : "no face",
    _ => "",
};
