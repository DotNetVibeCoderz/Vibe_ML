using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Tasks;

namespace MediaPipeNet.Maui;

/// <summary>
/// Model files ship inside the APK / IPA as MAUI assets, which are not plain files on disk. On first launch they are
/// copied to the app data folder, and every task reads them from there through <see cref="BaseOptions.ModelPaths"/>.
/// The sample bundles the INT8 variants where they exist (a quarter of the size) and maps each catalog id to its file.
/// </summary>
public static class ModelInstaller
{
    private static readonly (ModelDescriptor Model, string File)[] s_models =
    [
        (ModelCatalog.FaceDetectionShortRange, "face_detection_short_range.onnx"),
        (ModelCatalog.FaceLandmarksDetector, "face_landmarks_detector.int8.onnx"),
        (ModelCatalog.FaceLandmarksDetector192, "face_landmarks_detector_192.int8.onnx"),
        (ModelCatalog.FaceStylizerColorSketch, "face_stylizer_color_sketch.int8.onnx"),
        (ModelCatalog.EfficientDetLite0, "efficientdet_lite0.int8.onnx"),
    ];

    public static async Task<BaseOptions> InstallAsync(ExecutionProvider provider)
    {
        string dir = Path.Combine(FileSystem.AppDataDirectory, "models");
        Directory.CreateDirectory(dir);
        foreach (var (_, file) in s_models)
        {
            string target = Path.Combine(dir, file);
            if (File.Exists(target)) continue;
            await using var source = await FileSystem.OpenAppPackageFileAsync($"models/{file}");
            await using var output = File.Create(target);
            await source.CopyToAsync(output);
        }
        return new BaseOptions
        {
            ModelDirectory = dir,
            ModelPaths = s_models.ToDictionary(m => m.Model.Id, m => Path.Combine(dir, m.File)),
            Inference = new InferenceOptions { Provider = provider },
        };
    }

    public static async Task<byte[]> ReadSampleAsync(string name)
    {
        await using var stream = await FileSystem.OpenAppPackageFileAsync($"samples/{name}");
        using var buffer = new MemoryStream();
        await stream.CopyToAsync(buffer);
        return buffer.ToArray();
    }
}
