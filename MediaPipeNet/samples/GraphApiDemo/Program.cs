// MediaPipe.NET — Graph API demo
// Created by Gravicode Studios, led by Kang Fadhil.
//
// 1. Builds a custom calculator graph in code: one image stream fans out to a face detector and a
//    hand landmarker running in parallel; a custom node joins both results per timestamp and a
//    second custom node anonymizes faces (pixelation).
// 2. Loads an equivalent graph from MediaPipe-style .pbtxt text.
// 3. Shows live-stream flow limiting (MaxInFlight) dropping frames while the graph is busy.

using System.Collections.Concurrent;
using System.Diagnostics;
using MediaPipeNet;
using MediaPipeNet.Framework;
using MediaPipeNet.Framework.Config;
using MediaPipeNet.Framework.Nodes;
using MediaPipeNet.Imaging;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Tasks.Vision.Graph;

var images = Path.Combine(AppContext.BaseDirectory, "images");
var output = Directory.CreateDirectory(Path.Combine(AppContext.BaseDirectory, "output")).FullName;
Console.WriteLine("MediaPipe.NET — Graph API demo (Gravicode Studios / Kang Fadhil)\n");

// ------------------------------------------------------------------ 1. graph built in code
var builder = new GraphBuilder { Options = new GraphOptions { MaxInFlight = 4 } };
builder.AddInputStream<MPImage>("image");
builder.AddNode("face_detector", new VisionTaskNode<FaceDetector, FaceDetectionResult>(
        ct => FaceDetector.CreateAsync(new() { RunningMode = RunningMode.Video }, ct),
        (task, img, ts) => task.DetectForVideo(img, ts)))
    .In("IMAGE", "image").Out("RESULT", "faces");
builder.AddNode("hand_landmarker", new VisionTaskNode<HandLandmarker, HandLandmarkResult>(
        ct => HandLandmarker.CreateAsync(new() { RunningMode = RunningMode.Video }, ct),
        (task, img, ts) => task.DetectForVideo(img, ts)))
    .In("IMAGE", "image").Out("RESULT", "hands");
builder.AddNode("summary", new CombineNode<FaceDetectionResult, HandLandmarkResult, string>(
        (faces, hands) => $"{faces?.Detections.Count ?? 0} face(s), {hands?.Hands.Count ?? 0} hand(s)"))
    .In("A", "faces").In("B", "hands").Out("OUT", "summary");
builder.AddNode("anonymizer", new PixelateFacesNode(blockSize: 14))
    .In("IMAGE", "image").In("FACES", "faces").Out("IMAGE", "anonymized");
builder.AddOutputStream("summary").AddOutputStream("anonymized");

await using (var graph = builder.Build())
{
    Console.WriteLine("Graph (Mermaid):\n" + graph.ToMermaid());
    graph.ObserveOutputStream<string>("summary", p => Console.WriteLine($"  t={p.Timestamp.Milliseconds,5:F0} ms  {p.Value}"));
    int saved = 0;
    graph.ObserveOutputStream<MPImage>("anonymized", p =>
    {
        p.Value.SaveAsJpeg(Path.Combine(output, $"anonymized-{saved++}.jpg"));
        p.Value.Dispose();
    });
    await graph.StartAsync();

    string[] files = ["portrait.jpg", "victory.jpg", "pose.jpg", "thumb_up.jpg"];
    var frames = files.Select(f => MPImage.Load(Path.Combine(images, f))).ToArray();
    for (int i = 0; i < frames.Length; i++) graph.AddPacket("image", frames[i], i * 40L);
    await graph.CloseAsync();
    foreach (var f in frames) f.Dispose();
    Console.WriteLine($"  -> {saved} anonymized images in {output}\n");
}

// ------------------------------------------------------------------ 2. graph from .pbtxt
const string Pbtxt = """
    # Same idea, declared like a MediaPipe CalculatorGraphConfig.
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
      options { max_results: 2 }
    }
    """;
var registry = CalculatorRegistry.Default.AddVisionCalculators();
await using (var graph = GraphConfig.ParsePbtxt(Pbtxt).Build(registry))
{
    graph.ObserveOutputStream<ObjectDetectionResult>("objects", p =>
        Console.WriteLine($"  objects: {string.Join(", ", p.Value.Detections.Select(d => d.TopCategory))}"));
    graph.ObserveOutputStream<ClassificationResult>("labels", p =>
        Console.WriteLine($"  labels : {string.Join(", ", p.Value.Categories)}"));
    await graph.StartAsync();
    Console.WriteLine("Graph from .pbtxt:");
    using var image = MPImage.Load(Path.Combine(images, "cats_and_dogs.jpg"));
    graph.AddPacket("image", image, 0);
    await graph.CloseAsync();
    Console.WriteLine();
}

// ------------------------------------------------------------------ 3. flow limiting
var limited = new GraphBuilder { Options = new GraphOptions { MaxInFlight = 1 } }
    .AddInputStream<int>("frames")
    .AddNode("slow_model", new LambdaNode<int, int>(x => { Thread.Sleep(50); return x; })).In("IN", "frames").Out("OUT", "done").Graph
    .AddOutputStream("done")
    .Build();
var processed = new ConcurrentQueue<int>();
limited.ObserveOutputStream<int>("done", p => processed.Enqueue(p.Value));
await limited.StartAsync();
var clock = Stopwatch.StartNew();
for (int i = 0; i < 60; i++)
{
    limited.AddPacket("frames", i, clock.ElapsedMilliseconds + i); // a 100 FPS "camera"
    await Task.Delay(10);
}
await limited.CloseAsync();
Console.WriteLine($"Flow limiting: 60 frames offered at 100 FPS to a 20 FPS model -> {processed.Count} processed, {limited.DroppedPackets} dropped");
Console.WriteLine($"Processed frames: {string.Join(" ", processed)}");
await limited.DisposeAsync();

/// <summary>A custom calculator: pixelates every detected face in the image.</summary>
internal sealed class PixelateFacesNode(int blockSize) : CalculatorNode
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
                for (int by = (int)box.Y; by < box.Bottom; by += blockSize)
                    for (int bx = (int)box.X; bx < box.Right; bx += blockSize)
                    {
                        var sample = px[Math.Min(by + blockSize / 2, copy.Height - 1) * copy.Width + Math.Min(bx + blockSize / 2, copy.Width - 1)];
                        for (int y = by; y < Math.Min(by + blockSize, (int)box.Bottom); y++)
                            for (int x = bx; x < Math.Min(bx + blockSize, (int)box.Right); x++)
                                px[y * copy.Width + x] = sample;
                    }
            }
        }
        context.Send("IMAGE", copy);
    }
}
