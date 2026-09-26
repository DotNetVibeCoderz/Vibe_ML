"""Builds notebooks/MediaPipeNet_QuickStart.ipynb (Polyglot Notebooks, C# kernel)."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def cs(s):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {"dotnet_interactive": {"language": "csharp"}, "polyglot_notebook": {"kernelName": "csharp"}},
            "outputs": [], "source": s.strip("\n").splitlines(True)}


cells = [
    md("""
# MediaPipe.NET — Quick Start

**EN** — Run Google MediaPipe's vision tasks natively in .NET: faces, face mesh, hands, gestures, pose, segmentation, objects and classification. Open this notebook in VS Code with the *Polyglot Notebooks* extension (C# kernel).

**ID** — Jalankan task vision Google MediaPipe secara native di .NET: wajah, face mesh, tangan, gestur, pose, segmentasi, objek, dan klasifikasi. Buka notebook ini di VS Code dengan ekstensi *Polyglot Notebooks* (kernel C#).

*Created by Gravicode Studios, led by Kang Fadhil.*
"""),
    md("""
## 1. Install / Instalasi

`Gravicode.MediaPipeNet` brings the CPU runtime; `Gravicode.MediaPipeNet.Models.All` brings every model (≈75 MB) so nothing is downloaded at run time.
If you build from source, run `dotnet pack -c Release` first and uncomment the `#i` line to use `artifacts/packages`.

`Gravicode.MediaPipeNet` membawa runtime CPU; `Gravicode.MediaPipeNet.Models.All` membawa semua model (≈75 MB). Jika membangun dari source, jalankan `dotnet pack -c Release` lalu aktifkan baris `#i`.
"""),
    cs("""
// #i "nuget: ../artifacts/packages"
#r "nuget: Gravicode.MediaPipeNet, 0.1.0"
#r "nuget: Gravicode.MediaPipeNet.Visualization, 0.1.0"
#r "nuget: Gravicode.MediaPipeNet.Models.All, 0.1.0"
"""),
    cs("""
using MediaPipeNet;
using MediaPipeNet.Imaging;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Visualization;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

// In a notebook, NuGet content files are not copied next to the kernel, so the model store is pointed
// at the model packages in the NuGet cache. Download from nuget.org stays as the fallback.
var nugetRoot = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".nuget", "packages");
var modelDirs = Directory.GetDirectories(nugetRoot, "gravicode.mediapipenet.models.*")
    .SelectMany(Directory.GetDirectories)
    .Select(v => Path.Combine(v, "contentFiles", "any", "any", "models"))
    .Where(Directory.Exists);
var store = new ModelStore(modelDirs.Select(d => (IModelProvider)new DirectoryModelProvider(d)).Concat(ModelStore.Default.Providers));
var baseOptions = new BaseOptions { ModelStore = store };

// Shows an ImageSharp image inline.
void Show(Image<Rgba32> image, int width = 520)
{
    using var ms = new MemoryStream();
    image.SaveAsPng(ms);
    display(HTML($"<img width='{width}' src='data:image/png;base64,{Convert.ToBase64String(ms.ToArray())}'/>"));
}

// MediaPipe's public test images.
async Task<MPImage> Download(string name)
{
    using var http = new HttpClient();
    return MPImage.Load(await http.GetByteArrayAsync($"https://storage.googleapis.com/mediapipe-assets/{name}"));
}
"""),
    md("## 2. Face detection & face mesh / Deteksi wajah & face mesh"),
    cs("""
var portrait = await Download("portrait.jpg");
var faces = FaceDetector.Create(new() { BaseOptions = baseOptions });
faces.Detect(portrait).Detections.Select(d => new { d.BoundingBox, d.Score })
"""),
    cs("""
var mesh = FaceLandmarker.Create(new() { BaseOptions = baseOptions, OutputFaceBlendshapes = true });
var face = mesh.Detect(portrait).Faces[0];
var canvas = portrait.ToImage();
LandmarkDrawer.DrawFace(canvas, face, drawAllPoints: true);
Show(canvas);
face.Blendshapes!.Where(c => c.CategoryName != "_neutral").OrderByDescending(c => c.Score).Take(5)
"""),
    md("## 3. Hands & gestures / Tangan & gestur"),
    cs("""
var thumb = await Download("thumb_up.jpg");
var gestures = GestureRecognizer.Create(new() { BaseOptions = baseOptions });
var result = gestures.Recognize(thumb);
var handCanvas = thumb.ToImage();
ResultRenderer.Render(handCanvas, result);
Show(handCanvas, 360);
result.Hands.Select(h => $"{h.Hand.Handedness.CategoryName}: {h.TopGesture}")
"""),
    md("## 4. Pose + segmentation / Pose + segmentasi"),
    cs("""
var yoga = await Download("pose.jpg");
var pose = PoseLandmarker.Create(new() { BaseOptions = baseOptions, OutputSegmentationMasks = true });
var person = pose.Detect(yoga).Poses[0];
var poseCanvas = yoga.ToImage();
SegmentationMaskOverlay.BlurBackground(poseCanvas, person.SegmentationMask!, 10);
LandmarkDrawer.DrawPose(poseCanvas, person);
Show(poseCanvas);
person[PoseLandmark.LeftWrist]
"""),
    md("## 5. Objects & classification / Objek & klasifikasi"),
    cs("""
var pets = await Download("cats_and_dogs.jpg");
var objects = ObjectDetector.Create(new() { BaseOptions = baseOptions });
var found = objects.Detect(pets);
var petCanvas = pets.ToImage();
ResultRenderer.Render(petCanvas, found);
Show(petCanvas);
var classifier = ImageClassifier.Create(new() { BaseOptions = baseOptions, MaxResults = 3 });
classifier.Classify(pets).Categories
"""),
    md("## 6. Results are JSON-ready / Hasil siap JSON"),
    cs("""
found.ToJson(indented: true)
"""),
    md("""
## 7. Graph API

Compose tasks and your own logic into a calculator graph — the same model MediaPipe uses internally.

Susun task dan logika sendiri menjadi calculator graph — model yang sama dengan internal MediaPipe.
"""),
    cs("""
using MediaPipeNet.Framework;
using MediaPipeNet.Framework.Nodes;
using MediaPipeNet.Tasks.Vision.Graph;

var builder = new GraphBuilder();
builder.AddInputStream<MPImage>("image");
builder.AddNode("objects", new VisionTaskNode<ObjectDetector, ObjectDetectionResult>(
        ct => ObjectDetector.CreateAsync(new() { BaseOptions = baseOptions, RunningMode = RunningMode.Video }, ct),
        (t, img, ts) => t.DetectForVideo(img, ts)))
    .In("IMAGE", "image").Out("RESULT", "detections");
builder.AddNode("count", new LambdaNode<ObjectDetectionResult, string>(r => $"{r.Detections.Count} objects"))
    .In("IN", "detections").Out("OUT", "summary");
builder.AddOutputStream("summary");

var graph = builder.Build();
graph.ObserveOutputStream<string>("summary", p => Console.WriteLine($"{p.Timestamp}: {p.Value}"));
await graph.StartAsync();
graph.AddPacket("image", pets, 0);
await graph.CloseAsync();
graph.ToMermaid()
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": ".NET (C#)", "language": "C#", "name": ".net-csharp"},
        "language_info": {"name": "polyglot-notebook"},
        "polyglot_notebook": {"kernelInfo": {"defaultKernelName": "csharp", "items": [{"aliases": [], "name": "csharp"}]}},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = ROOT / "notebooks" / "MediaPipeNet_QuickStart.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}")
