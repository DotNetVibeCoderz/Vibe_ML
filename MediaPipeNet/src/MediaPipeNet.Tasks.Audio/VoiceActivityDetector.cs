using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Serialization;

namespace MediaPipeNet.Tasks.Audio;

/// <summary>Speech probability of one analysis window.</summary>
/// <param name="TimestampMs">Start of the window.</param>
/// <param name="DurationMs">Window length (975 ms).</param>
/// <param name="SpeechProbability">Highest score among YAMNet's speech classes.</param>
/// <param name="IsSpeech">Whether the probability reached the threshold.</param>
public readonly record struct VoiceActivityFrame(long TimestampMs, long DurationMs, float SpeechProbability, bool IsSpeech);

/// <summary>A stretch of detected speech.</summary>
/// <param name="StartMs">Start time.</param>
/// <param name="EndMs">End time.</param>
/// <param name="MeanProbability">Mean speech probability of the windows in the segment.</param>
public readonly record struct SpeechSegment(long StartMs, long EndMs, float MeanProbability)
{
    /// <summary>Segment length.</summary>
    public long DurationMs => EndMs - StartMs;
}

/// <summary>Result of <see cref="VoiceActivityDetector.Detect"/>.</summary>
/// <param name="Frames">Per-window speech probabilities.</param>
/// <param name="Segments">Merged speech segments.</param>
public sealed record VoiceActivityResult(IReadOnlyList<VoiceActivityFrame> Frames, IReadOnlyList<SpeechSegment> Segments)
{
    /// <summary>Fraction of the analysed windows that contain speech.</summary>
    public float SpeechRatio => Frames.Count == 0 ? 0f : Frames.Count(f => f.IsSpeech) / (float)Frames.Count;

    /// <summary>Serializes the result to JSON.</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);
}

/// <summary>Options of <see cref="VoiceActivityDetector"/>.</summary>
public sealed record VoiceActivityDetectorOptions
{
    /// <summary>Model and runtime options.</summary>
    public BaseOptions BaseOptions { get; init; } = BaseOptions.Default;

    /// <summary>Running mode. Default <see cref="AudioRunningMode.AudioClips"/>.</summary>
    public AudioRunningMode RunningMode { get; init; } = AudioRunningMode.AudioClips;

    /// <summary>Speech probability at or above which a window counts as speech. Default 0.5.</summary>
    public float Threshold { get; init; } = 0.5f;

    /// <summary>Hop between analysis windows in milliseconds (1..975; smaller = finer timing, more compute). Default 487.</summary>
    public int HopMs { get; init; } = 487;

    /// <summary>Speech segments separated by less silence than this are merged. Default 500 ms.</summary>
    public int MinSilenceMs { get; init; } = 500;

    /// <summary>Speech segments shorter than this are dropped. Default 250 ms.</summary>
    public int MinSpeechMs { get; init; } = 250;

    /// <summary>Receives every analysed window in <see cref="AudioRunningMode.AudioStream"/> mode.</summary>
    public Action<VoiceActivityFrame>? FrameCallback { get; init; }
}

/// <summary>
/// Voice activity detection built on YAMNet: a window is speech when the highest score among the speech
/// classes (speech, conversation, narration, child speech, whispering, shouting, ...) reaches the
/// threshold. Overlapping windows give finer timing; adjacent speech windows are merged into segments.
/// </summary>
/// <example>
/// <code>
/// using var vad = VoiceActivityDetector.Create();
/// foreach (var s in vad.Detect(AudioData.LoadWav("meeting.wav")).Segments)
///     Console.WriteLine($"speech {s.StartMs}–{s.EndMs} ms");
/// </code>
/// </example>
public sealed class VoiceActivityDetector : IDisposable
{
    /// <summary>YAMNet classes treated as speech.</summary>
    public static IReadOnlyList<string> SpeechClasses { get; } =
    [
        "Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue", "Babbling",
        "Speech synthesizer", "Shout", "Yell", "Whispering",
    ];

    private readonly YamNet _yamnet;
    private readonly int[] _speechIndices;
    private readonly int _hopSamples;
    private readonly AudioStreamBuffer? _stream;

    private VoiceActivityDetector(VoiceActivityDetectorOptions options, OnnxModel model)
    {
        if (options.HopMs is < 1 or > 975) throw new ArgumentOutOfRangeException(nameof(options), "HopMs must be in 1..975.");
        if (options.RunningMode == AudioRunningMode.AudioStream && options.FrameCallback is null)
            throw new ArgumentException("A FrameCallback is required in AudioStream mode.", nameof(options));
        Options = options;
        _yamnet = new YamNet(model);
        _speechIndices = SpeechClasses.Select(c => YamNet.Labels.ToList().IndexOf(c)).Where(i => i >= 0).ToArray();
        _hopSamples = options.HopMs * YamNet.SampleRate / 1000;
        if (options.RunningMode == AudioRunningMode.AudioStream)
            _stream = new AudioStreamBuffer(YamNet.WindowSamples, _hopSamples, YamNet.SampleRate);
    }

    /// <summary>The options the detector was created with.</summary>
    public VoiceActivityDetectorOptions Options { get; }

    /// <summary>Creates the detector (resolving the model synchronously).</summary>
    public static VoiceActivityDetector Create(VoiceActivityDetectorOptions? options = null)
    {
        options ??= new VoiceActivityDetectorOptions();
        return new VoiceActivityDetector(options, ModelLoader.Load(options.BaseOptions, ModelCatalog.YamNet));
    }

    /// <summary>Creates the detector, downloading the model asynchronously when needed.</summary>
    public static async Task<VoiceActivityDetector> CreateAsync(VoiceActivityDetectorOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new VoiceActivityDetectorOptions();
        return new VoiceActivityDetector(options, await ModelLoader.LoadAsync(options.BaseOptions, ModelCatalog.YamNet, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Analyses a complete clip.</summary>
    public VoiceActivityResult Detect(AudioData audio)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (_stream is not null) throw new InvalidOperationException("Detect is only available in AudioClips mode; use DetectStream.");
        var samples = audio.Resample(YamNet.SampleRate).Samples;
        var frames = new List<VoiceActivityFrame>();
        var window = new float[YamNet.WindowSamples];
        for (int start = 0; start == 0 || start + YamNet.WindowSamples - _hopSamples < samples.Length; start += _hopSamples)
        {
            Array.Clear(window);
            int n = Math.Clamp(samples.Length - start, 0, YamNet.WindowSamples);
            samples.AsSpan(start, n).CopyTo(window);
            frames.Add(Analyse(window, (long)start * 1000 / YamNet.SampleRate));
        }
        return new VoiceActivityResult(frames, Merge(frames, Options.MinSilenceMs, Options.MinSpeechMs));
    }

    /// <summary>Analyses a clip on a worker thread.</summary>
    public Task<VoiceActivityResult> DetectAsync(AudioData audio, CancellationToken cancellationToken = default) =>
        Task.Run(() => Detect(audio), cancellationToken);

    /// <summary>
    /// Appends a chunk of a stream; every completed window is delivered to
    /// <see cref="VoiceActivityDetectorOptions.FrameCallback"/> before this method returns.
    /// </summary>
    public void DetectStream(AudioData chunk, long timestampMs)
    {
        ArgumentNullException.ThrowIfNull(chunk);
        if (_stream is null) throw new InvalidOperationException("DetectStream is only available in AudioStream mode; use Detect.");
        foreach (var (window, ts) in _stream.Append(chunk.Resample(YamNet.SampleRate).Samples, timestampMs))
            Options.FrameCallback!(Analyse(window, ts));
    }

    /// <summary>Forgets buffered stream samples.</summary>
    public void ResetStream() => _stream?.Reset();

    /// <summary>
    /// Merges speech windows into segments: windows closer than <paramref name="minSilenceMs"/> join the
    /// same segment, and segments shorter than <paramref name="minSpeechMs"/> are dropped.
    /// </summary>
    public static IReadOnlyList<SpeechSegment> Merge(IReadOnlyList<VoiceActivityFrame> frames, int minSilenceMs, int minSpeechMs)
    {
        ArgumentNullException.ThrowIfNull(frames);
        var segments = new List<SpeechSegment>();
        long start = -1, end = -1;
        float sum = 0;
        int count = 0;
        foreach (var f in frames)
        {
            if (!f.IsSpeech) continue;
            long fEnd = f.TimestampMs + f.DurationMs;
            if (start >= 0 && f.TimestampMs - end <= minSilenceMs)
            {
                end = Math.Max(end, fEnd);
            }
            else
            {
                Flush();
                start = f.TimestampMs;
                end = fEnd;
            }
            sum += f.SpeechProbability;
            count++;
        }
        Flush();
        return segments;

        void Flush()
        {
            if (start >= 0 && end - start >= minSpeechMs) segments.Add(new SpeechSegment(start, end, sum / count));
            start = end = -1;
            sum = 0;
            count = 0;
        }
    }

    private VoiceActivityFrame Analyse(ReadOnlySpan<float> window, long timestampMs)
    {
        var scores = _yamnet.Run(window);
        float p = 0;
        foreach (int i in _speechIndices) p = MathF.Max(p, scores[i]);
        return new VoiceActivityFrame(timestampMs, YamNet.WindowSamples * 1000L / YamNet.SampleRate, p, p >= Options.Threshold);
    }

    /// <inheritdoc />
    public void Dispose() => _yamnet.Dispose();
}
