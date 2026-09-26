using System.Text.Json;
using MediaPipeNet.Imaging;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Tests;

namespace MediaPipeNet.Tasks.Tests;

/// <summary>Access to the golden reference produced by the official MediaPipe Python package.</summary>
internal static class Golden
{
    private static readonly Lazy<JsonElement> s_results = new(() =>
        JsonDocument.Parse(File.ReadAllText(TestPaths.Golden)).RootElement.GetProperty("results"));

    public static JsonElement For(string task, string image) => s_results.Value.GetProperty(task).GetProperty(image);

    public static NormalizedLandmark[] Landmarks(JsonElement list) =>
        list.EnumerateArray().Select(e => new NormalizedLandmark(e.GetProperty("x").GetSingle(), e.GetProperty("y").GetSingle(), e.GetProperty("z").GetSingle())).ToArray();

    public static RectF Box(JsonElement detection)
    {
        var b = detection.GetProperty("box");
        return new RectF(b.GetProperty("x").GetSingle(), b.GetProperty("y").GetSingle(), b.GetProperty("width").GetSingle(), b.GetProperty("height").GetSingle());
    }

    /// <summary>Mean 2-D distance (normalized units) between two landmark lists.</summary>
    public static float MeanError(IReadOnlyList<NormalizedLandmark> a, IReadOnlyList<NormalizedLandmark> b)
    {
        float sum = 0;
        for (int i = 0; i < a.Count; i++) sum += MathF.Sqrt((a[i].X - b[i].X) * (a[i].X - b[i].X) + (a[i].Y - b[i].Y) * (a[i].Y - b[i].Y));
        return sum / a.Count;
    }
}

/// <summary>Shared options pointing at the repository's model folder.</summary>
internal static class Fixtures
{
    public static BaseOptions Base { get; } = new() { ModelDirectory = TestPaths.Models };

    public static MPImage Load(string name) => MPImage.Load(TestPaths.Image(name));
}
