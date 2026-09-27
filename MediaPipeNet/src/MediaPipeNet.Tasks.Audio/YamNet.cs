using MediaPipeNet.Inference;

namespace MediaPipeNet.Tasks.Audio;

/// <summary>Runs the YAMNet model on one 0.975 s window of 16 kHz audio.</summary>
internal sealed class YamNet(OnnxModel model) : IDisposable
{
    public const int SampleRate = 16_000;
    public const int WindowSamples = 15_600;

    private static readonly Lazy<string[]> s_labels = new(() =>
    {
        using var stream = typeof(YamNet).Assembly.GetManifestResourceStream("MediaPipeNet.Tasks.Audio.Resources.yamnet_labels.txt")
                           ?? throw new InvalidOperationException("Missing embedded YAMNet labels.");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd().Split('\n').Select(l => l.TrimEnd('\r')).Where(l => l.Length > 0).ToArray();
    });

    public static IReadOnlyList<string> Labels => s_labels.Value;

    /// <summary>Returns the 521 class scores (already sigmoid probabilities).</summary>
    public float[] Run(ReadOnlySpan<float> window)
    {
        using var ctx = model.RentContext();
        window.CopyTo(ctx.GetInput(0));
        ctx.Run();
        return ctx.GetOutput(0).ToArray();
    }

    public void Dispose() => model.Dispose();
}

/// <summary>Accumulates stream chunks and cuts them into fixed windows with a given hop.</summary>
internal sealed class AudioStreamBuffer(int windowSamples, int hopSamples, int sampleRate)
{
    private readonly Lock _gate = new();
    private float[] _buffer = new float[windowSamples * 2];
    private int _count;
    private long _bufferStartSample = -1;
    private long _originMs;
    private long _lastTimestampMs = long.MinValue;

    public List<(float[] Window, long TimestampMs)> Append(float[] samples, long timestampMs)
    {
        lock (_gate)
        {
            if (timestampMs <= _lastTimestampMs)
                throw new ArgumentException($"Timestamps must increase monotonically: {timestampMs} ms after {_lastTimestampMs} ms.", nameof(timestampMs));
            _lastTimestampMs = timestampMs;
            if (_bufferStartSample < 0)
            {
                _bufferStartSample = 0;
                _originMs = timestampMs;
            }
            if (_count + samples.Length > _buffer.Length) Array.Resize(ref _buffer, Math.Max(_buffer.Length * 2, _count + samples.Length));
            samples.CopyTo(_buffer, _count);
            _count += samples.Length;

            var windows = new List<(float[], long)>();
            int offset = 0;
            while (_count - offset >= windowSamples)
            {
                var window = _buffer.AsSpan(offset, windowSamples).ToArray();
                windows.Add((window, _originMs + (_bufferStartSample + offset) * 1000 / sampleRate));
                offset += hopSamples;
            }
            if (offset > 0)
            {
                Array.Copy(_buffer, offset, _buffer, 0, _count - offset);
                _count -= offset;
                _bufferStartSample += offset;
            }
            return windows;
        }
    }

    public void Reset()
    {
        lock (_gate)
        {
            _count = 0;
            _bufferStartSample = -1;
            _lastTimestampMs = long.MinValue;
        }
    }
}
