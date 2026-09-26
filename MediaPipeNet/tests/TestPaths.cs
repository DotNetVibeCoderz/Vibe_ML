namespace MediaPipeNet.Tests;

/// <summary>Locates repository assets (models, fixture images, golden data) from the test output folder.</summary>
internal static class TestPaths
{
    private static readonly Lazy<string> s_root = new(() =>
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "MediaPipeNet.slnx"))) dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("Repository root not found.");
    });

    public static string Root => s_root.Value;

    public static string Models => Path.Combine(Root, "models", "onnx");

    public static string Image(string name) => Path.Combine(Root, "tests", "assets", "images", name);

    public static string Golden => Path.Combine(Root, "tests", "assets", "golden", "mediapipe_python_reference.json");
}
