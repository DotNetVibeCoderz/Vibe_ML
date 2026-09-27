using System.Diagnostics;
using MediaPipeNet.Diagnostics;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Serialization;

namespace MediaPipeNet.Tasks.Audio;

/// <summary>How an audio task receives its input, mirroring MediaPipe's audio running modes.</summary>
public enum AudioRunningMode
{
    /// <summary>Independent, complete clips: each call returns the results of the whole clip.</summary>
    AudioClips = 0,

    /// <summary>
    /// A continuous stream delivered in chunks of any size; results are produced for every complete
    /// model window and delivered through the result callback.
    /// </summary>
    AudioStream = 1,
}

/// <summary>Result of one audio window.</summary>
/// <param name="Categories">Categories sorted by score.</param>
/// <param name="TimestampMs">Start of the window, in milliseconds from the start of the clip or stream.</param>
public sealed record AudioClassificationResult(IReadOnlyList<Category> Categories, long TimestampMs)
{
    /// <summary>The best category, or null when every category was filtered out.</summary>
    public Category? TopCategory => Categories.Count > 0 ? Categories[0] : null;

    /// <summary>Serializes the result to JSON.</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);
}

/// <summary>Options of <see cref="AudioClassifier"/>.</summary>
public sealed record AudioClassifierOptions
{
    /// <summary>Model and runtime options.</summary>
    public BaseOptions BaseOptions { get; init; } = BaseOptions.Default;

    /// <summary>Running mode. Default <see cref="AudioRunningMode.AudioClips"/>.</summary>
    public AudioRunningMode RunningMode { get; init; } = AudioRunningMode.AudioClips;

    /// <summary>Category filtering and ranking. Default: the top 5 categories.</summary>
    public ClassifierOptions Classifier { get; init; } = new() { MaxResults = 5 };

    /// <summary>Receives results in <see cref="AudioRunningMode.AudioStream"/> mode.</summary>
    public Action<AudioClassificationResult>? ResultCallback { get; init; }

    /// <summary>
    /// Path of your own ONNX audio model taking 15 600 samples of 16 kHz audio (e.g. a MediaPipe Model Maker export converted with
    /// <c>convert_models.py --custom</c>) used instead of the built-in one. Default null.
    /// </summary>
    public string? ModelPath { get; init; }

    /// <summary>
    /// Category names for a custom model, in output order. When null, <c>{model}.labels.txt</c> next to
    /// <see cref="ModelPath"/> is used if present, otherwise the built-in labels.
    /// </summary>
    public IReadOnlyList<string>? Labels { get; init; }
}

/// <summary>
/// Audio event classification with YAMNet: 521 AudioSet classes (speech, music, animals, vehicles,
/// alarms, ...) for every 0.975 s window of 16 kHz mono audio. Other sample rates are resampled.
/// </summary>
/// <example>
/// <code>
/// using var classifier = AudioClassifier.Create();
/// foreach (var window in classifier.Classify(AudioData.LoadWav("speech.wav")))
///     Console.WriteLine($"{window.TimestampMs} ms: {window.TopCategory}");
/// </code>
/// </example>
public sealed class AudioClassifier : IDisposable
{
    private readonly YamNet _yamnet;
    private readonly AudioStreamBuffer? _stream;
    private readonly KeyValuePair<string, object?>[] _tags = [new("task", nameof(AudioClassifier))];
    private readonly IReadOnlyList<string> _labels;

    private AudioClassifier(AudioClassifierOptions options, OnnxModel model)
    {
        if (options.RunningMode == AudioRunningMode.AudioStream && options.ResultCallback is null)
            throw new ArgumentException("A ResultCallback is required in AudioStream mode.", nameof(options));
        if (options.RunningMode == AudioRunningMode.AudioClips && options.ResultCallback is not null)
            throw new ArgumentException("ResultCallback is only allowed in AudioStream mode.", nameof(options));
        Options = options;
        _yamnet = new YamNet(model);
        _labels = CustomModels.ResolveLabels(options.Labels, options.ModelPath, YamNet.Labels);
        if (options.RunningMode == AudioRunningMode.AudioStream)
            _stream = new AudioStreamBuffer(YamNet.WindowSamples, YamNet.WindowSamples, YamNet.SampleRate);
    }

    /// <summary>The options the task was created with.</summary>
    public AudioClassifierOptions Options { get; }

    /// <summary>The 521 YAMNet category names, in model output order.</summary>
    public static IReadOnlyList<string> Labels => YamNet.Labels;

    /// <summary>Creates the classifier (resolving the model synchronously).</summary>
    public static AudioClassifier Create(AudioClassifierOptions? options = null)
    {
        options ??= new AudioClassifierOptions();
        return new AudioClassifier(options, CustomModels.Load(options.BaseOptions, ModelCatalog.YamNet, options.ModelPath));
    }

    /// <summary>Creates the classifier, downloading the model asynchronously when needed.</summary>
    public static async Task<AudioClassifier> CreateAsync(AudioClassifierOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new AudioClassifierOptions();
        return new AudioClassifier(options, await CustomModels.LoadAsync(options.BaseOptions, ModelCatalog.YamNet, options.ModelPath, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>
    /// Classifies a complete clip: one result per 0.975 s window (the last window is zero-padded), as
    /// MediaPipe does in audio-clips mode.
    /// </summary>
    public IReadOnlyList<AudioClassificationResult> Classify(AudioData audio)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (_stream is not null) throw new InvalidOperationException("Classify is only available in AudioClips mode; use ClassifyStream.");
        var samples = audio.Resample(YamNet.SampleRate).Samples;
        var results = new List<AudioClassificationResult>();
        var window = new float[YamNet.WindowSamples];
        for (int start = 0; start < samples.Length || start == 0; start += YamNet.WindowSamples)
        {
            Array.Clear(window);
            samples.AsSpan(start, Math.Min(YamNet.WindowSamples, samples.Length - start)).CopyTo(window);
            results.Add(ClassifyWindow(window, (long)start * 1000 / YamNet.SampleRate));
            if (samples.Length == 0) break;
        }
        return results;
    }

    /// <summary>Classifies a clip on a worker thread.</summary>
    public Task<IReadOnlyList<AudioClassificationResult>> ClassifyAsync(AudioData audio, CancellationToken cancellationToken = default) =>
        Task.Run(() => Classify(audio), cancellationToken);

    /// <summary>
    /// Appends a chunk of a stream (any length, any sample rate — resampled per chunk). Every completed
    /// window is classified and delivered to <see cref="AudioClassifierOptions.ResultCallback"/> before
    /// this method returns. <paramref name="timestampMs"/> is the chunk's start time; it must increase.
    /// </summary>
    public void ClassifyStream(AudioData chunk, long timestampMs)
    {
        ArgumentNullException.ThrowIfNull(chunk);
        if (_stream is null) throw new InvalidOperationException("ClassifyStream is only available in AudioStream mode; use Classify.");
        foreach (var (window, ts) in _stream.Append(chunk.Resample(YamNet.SampleRate).Samples, timestampMs))
            Options.ResultCallback!(ClassifyWindow(window, ts));
    }

    /// <summary>Forgets buffered stream samples.</summary>
    public void ResetStream() => _stream?.Reset();

    private AudioClassificationResult ClassifyWindow(ReadOnlySpan<float> window, long timestampMs)
    {
        long start = Stopwatch.GetTimestamp();
        var scores = _yamnet.Run(window);
        var result = new AudioClassificationResult(Options.Classifier.Select(scores, _labels), timestampMs);
        MediaPipeTelemetry.TaskDuration.Record(Stopwatch.GetElapsedTime(start).TotalMilliseconds, _tags);
        MediaPipeTelemetry.FramesProcessed.Add(1, _tags);
        return result;
    }

    /// <inheritdoc />
    public void Dispose() => _yamnet.Dispose();
}
