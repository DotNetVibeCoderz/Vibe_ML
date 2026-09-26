// MediaPipe.NET benchmarks — created by Gravicode Studios, led by Kang Fadhil.
//
//   dotnet run -c Release --project benchmarks/MediaPipeNet.Benchmarks            (all)
//   dotnet run -c Release --project benchmarks/MediaPipeNet.Benchmarks -- --filter *Face*
//
// Every task runs on 640×480 frames (the resolution named in NFR-1) on the CPU provider.

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Tasks.Vision;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

BenchmarkSwitcher.FromAssembly(typeof(TaskBenchmarks).Assembly).Run(args);

[MemoryDiagnoser]
[MinIterationCount(15)]
[MaxIterationCount(40)]
public class TaskBenchmarks
{
    private static readonly BaseOptions Cpu = new() { Inference = new InferenceOptions { Provider = ExecutionProvider.Cpu } };
    private MPImage _portrait = null!, _hand = null!, _pose = null!, _pets = null!;
    private FaceDetector _faces = null!;
    private FaceLandmarker _mesh = null!;
    private HandLandmarker _hands = null!;
    private HandLandmarker _handsVideo = null!;
    private GestureRecognizer _gestures = null!;
    private PoseLandmarker _poseLandmarker = null!;
    private ImageSegmenter _segmenter = null!;
    private ObjectDetector _objects = null!;
    private ImageClassifier _classifier = null!;
    private long _ts;

    [GlobalSetup]
    public void Setup()
    {
        _portrait = Load("portrait.jpg");
        _hand = Load("victory.jpg");
        _pose = Load("pose.jpg");
        _pets = Load("cats_and_dogs.jpg");
        _faces = FaceDetector.Create(new() { BaseOptions = Cpu });
        _mesh = FaceLandmarker.Create(new() { BaseOptions = Cpu });
        _hands = HandLandmarker.Create(new() { BaseOptions = Cpu });
        _handsVideo = HandLandmarker.Create(new() { BaseOptions = Cpu, RunningMode = MediaPipeNet.RunningMode.Video });
        _gestures = GestureRecognizer.Create(new() { BaseOptions = Cpu });
        _poseLandmarker = PoseLandmarker.Create(new() { BaseOptions = Cpu });
        _segmenter = ImageSegmenter.Create(new() { BaseOptions = Cpu });
        _objects = ObjectDetector.Create(new() { BaseOptions = Cpu });
        _classifier = ImageClassifier.Create(new() { BaseOptions = Cpu });
    }

    private static MPImage Load(string name)
    {
        using var image = Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(Path.Combine(AppContext.BaseDirectory, "images", name));
        image.Mutate(x => x.Resize(new ResizeOptions { Size = new Size(640, 480), Mode = ResizeMode.Pad }));
        return MPImage.FromImage(image);
    }

    [Benchmark(Description = "FaceDetector (640x480)")]
    public int FaceDetection() => _faces.Detect(_portrait).Detections.Count;

    [Benchmark(Description = "FaceLandmarker (640x480)")]
    public int FaceMesh() => _mesh.Detect(_portrait).Faces.Count;

    [Benchmark(Description = "HandLandmarker image mode")]
    public int Hands() => _hands.Detect(_hand).Hands.Count;

    [Benchmark(Description = "HandLandmarker video mode (tracking)")]
    public int HandsTracking() => _handsVideo.DetectForVideo(_hand, _ts += 33).Hands.Count;

    [Benchmark(Description = "GestureRecognizer")]
    public int Gestures() => _gestures.Recognize(_hand).Hands.Count;

    [Benchmark(Description = "PoseLandmarker (lite)")]
    public int Pose() => _poseLandmarker.Detect(_pose).Poses.Count;

    [Benchmark(Description = "ImageSegmenter (selfie)")]
    public int Segmentation() => _segmenter.Segment(_portrait).ConfidenceMask.Width;

    [Benchmark(Description = "ObjectDetector (EfficientDet-Lite0)")]
    public int Objects() => _objects.Detect(_pets).Detections.Count;

    [Benchmark(Description = "ImageClassifier (EfficientNet-Lite0)")]
    public int Classification() => _classifier.Classify(_pets).Categories.Count;

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (var d in new IDisposable[] { _faces, _mesh, _hands, _handsVideo, _gestures, _poseLandmarker, _segmenter, _objects, _classifier, _portrait, _hand, _pose, _pets })
            d.Dispose();
    }
}

[MemoryDiagnoser]
public class ImageToTensorBenchmarks
{
    private MPImage _image = null!;
    private readonly float[] _tensor = new float[256 * 256 * 3];

    [GlobalSetup]
    public void Setup()
    {
        _image = MPImage.Create(1280, 720);
        _image.GetPixelSpan().Fill(new SixLabors.ImageSharp.PixelFormats.Rgba32(90, 120, 150, 255));
    }

    [Benchmark(Description = "720p -> 256x256 letterbox")]
    public void Letterbox() => ImageToTensor.Convert(_image, new ImageToTensorOptions(256, 256, 0, 1, KeepAspectRatio: true), _tensor);

    [Benchmark(Description = "720p rotated ROI -> 256x256")]
    public void RotatedRoi() => ImageToTensor.Convert(_image, new MediaPipeNet.NormalizedRect(0.5f, 0.5f, 0.4f, 0.7f, 0.6f), new ImageToTensorOptions(256, 256), _tensor);

    [GlobalCleanup]
    public void Cleanup() => _image.Dispose();
}
