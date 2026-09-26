using System.Globalization;
using MediaPipeNet.Gallery.Controls;
using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Tasks.Vision;

namespace MediaPipeNet.Gallery.Services;

public enum OptionKind { Slider, Toggle, Choice }

/// <summary>A user-tunable task option shown in the options panel.</summary>
public sealed record OptionSpec(string Key, string LabelEn, string LabelId, OptionKind Kind, double Default,
    double Min = 0, double Max = 1, double Step = 0.05, string Format = "0.00", string[]? Choices = null)
{
    public string Label => Loc.Pick(LabelEn, LabelId);
}

/// <summary>Current option values of a task page.</summary>
public sealed class OptionValues
{
    private readonly Dictionary<string, double> _values = new();

    public OptionValues(IEnumerable<OptionSpec> specs)
    {
        foreach (var s in specs) _values[s.Key] = s.Default;
    }

    public double this[string key] { get => _values[key]; set => _values[key] = value; }
    public float F(string key) => (float)_values[key];
    public int I(string key) => (int)Math.Round(_values[key]);
    public bool B(string key) => _values[key] > 0.5;
    public string Key => string.Join(";", _values.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key}={kv.Value:0.###}"));
    public string Inv(string key, string format = "0.00") => _values[key].ToString(format, CultureInfo.InvariantCulture);
}

/// <summary>One line in the results panel; <see cref="Fraction"/> draws a meter.</summary>
public sealed record ResultRow(string Label, string Value, double? Fraction = null);

/// <summary>What a task page shows after running.</summary>
public sealed record GalleryOutput(Overlay Overlay, IReadOnlyList<ResultRow> Rows, string Json, string Summary, MPImage? ImageOverride = null)
{
    public bool IsEmpty => Rows.Count == 0;
}

/// <summary>A task as presented by the Gallery.</summary>
public sealed class GalleryTask
{
    public required string Id { get; init; }
    public required string Category { get; init; }
    public required string TitleEn { get; init; }
    public required string TitleId { get; init; }
    public required string BlurbEn { get; init; }
    public required string BlurbId { get; init; }
    public required string ModelLabel { get; init; }
    public required string[] Samples { get; init; }
    public IReadOnlyList<OptionSpec> Options { get; init; } = [];
    public required Func<OptionValues, RunningMode, BaseOptions, IDisposable> Create { get; init; }
    public required Func<IDisposable, MPImage, long?, OptionValues, GalleryOutput> Run { get; init; }
    public required Func<OptionValues, string> Code { get; init; }
    public bool SupportsLive { get; init; } = true;

    public string Title => Loc.Pick(TitleEn, TitleId);
    public string Blurb => Loc.Pick(BlurbEn, BlurbId);
}

/// <summary>Creates and caches task instances according to the current settings.</summary>
public static class TaskEngine
{
    private static readonly Dictionary<string, IDisposable> Cache = new();
    private static readonly Lock Gate = new();

    static TaskEngine()
    {
        AppSettings.Changed += Reset;
    }

    public static BaseOptions BaseOptions => new()
    {
        ModelDirectory = string.IsNullOrWhiteSpace(AppSettings.Current.ModelDirectory) ? null : AppSettings.Current.ModelDirectory,
        Inference = new InferenceOptions { Provider = AppSettings.Current.Provider, IntraOpThreads = AppSettings.Current.Threads },
    };

    /// <summary>Returns a cached image-mode instance for these options (created on first use).</summary>
    public static IDisposable Get(GalleryTask task, OptionValues options)
    {
        var key = task.Id + "|" + options.Key;
        lock (Gate)
        {
            if (Cache.TryGetValue(key, out var existing)) return existing;
            foreach (var stale in Cache.Where(kv => kv.Key.StartsWith(task.Id + "|", StringComparison.Ordinal)).ToList())
            {
                stale.Value.Dispose();
                Cache.Remove(stale.Key);
            }
            var created = task.Create(options, RunningMode.Image, BaseOptions);
            Cache[key] = created;
            return created;
        }
    }

    /// <summary>Execution provider actually used by the most recently created model, for the readout.</summary>
    public static string ProviderLabel =>
        AppSettings.Current.Provider == ExecutionProvider.Auto
            ? "auto→" + (ExecutionProviderSelector.GetCandidates(ExecutionProvider.Auto, true).FirstOrDefault())
            : AppSettings.Current.Provider.ToString();

    private static readonly HashSet<IDisposable> Warm = [];

    /// <summary>Returns true the first time an instance is seen (the caller runs a warm-up pass).</summary>
    public static bool MarkWarm(IDisposable instance)
    {
        lock (Gate) return Warm.Add(instance);
    }

    public static void Reset()
    {
        lock (Gate)
        {
            Warm.Clear();
            foreach (var t in Cache.Values) t.Dispose();
            Cache.Clear();
        }
    }
}
