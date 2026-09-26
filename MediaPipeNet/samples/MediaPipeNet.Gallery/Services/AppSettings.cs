using System.Text.Json;
using MediaPipeNet.Inference;

namespace MediaPipeNet.Gallery.Services;

/// <summary>User settings, persisted as JSON under the local application data folder.</summary>
public sealed class AppSettings
{
    private static readonly string FilePath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "MediaPipeNet", "gallery-settings.json");

    public static AppSettings Current { get; private set; } = Load();

    // CPU by default: it works everywhere; GPU providers are opt-in from the Settings page.
    public ExecutionProvider Provider { get; set; } = ExecutionProvider.Cpu;
    public int Threads { get; set; }
    public string Language { get; set; } = "en";
    public string Theme { get; set; } = "Light";
    public string? ModelDirectory { get; set; }
    public bool MirrorCamera { get; set; } = true;
    public int CameraIndex { get; set; }
    public bool ShowLandmarkPoints { get; set; } = true;

    /// <summary>Raised after settings that affect running tasks change.</summary>
    public static event Action? Changed;

    public static AppSettings Load()
    {
        try
        {
            if (File.Exists(FilePath)) return JsonSerializer.Deserialize<AppSettings>(File.ReadAllText(FilePath)) ?? new AppSettings();
        }
        catch (JsonException) { }
        catch (IOException) { }
        return new AppSettings();
    }

    public void Save()
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(FilePath)!);
            File.WriteAllText(FilePath, JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch (IOException) { }
        Changed?.Invoke();
    }

    /// <summary>Uses in-memory settings only (screenshot mode must not touch the user's file).</summary>
    public static void UseTransient(AppSettings settings) => Current = settings;
}
