using System.Buffers.Binary;

namespace MediaPipeNet.Tasks.Audio;

/// <summary>
/// A block of mono audio: float samples in [−1, 1] at a known sample rate. Multi-channel input is
/// down-mixed by averaging the channels.
/// </summary>
public sealed class AudioData
{
    /// <summary>Creates audio from mono samples.</summary>
    public AudioData(float[] samples, int sampleRate)
    {
        ArgumentNullException.ThrowIfNull(samples);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        Samples = samples;
        SampleRate = sampleRate;
    }

    /// <summary>Mono samples in [−1, 1].</summary>
    public float[] Samples { get; }

    /// <summary>Samples per second.</summary>
    public int SampleRate { get; }

    /// <summary>Duration.</summary>
    public TimeSpan Duration => TimeSpan.FromSeconds((double)Samples.Length / SampleRate);

    /// <summary>Creates audio from interleaved float samples, down-mixing to mono.</summary>
    public static AudioData FromInterleaved(ReadOnlySpan<float> interleaved, int sampleRate, int channels = 1)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        var mono = new float[interleaved.Length / channels];
        for (int i = 0; i < mono.Length; i++)
        {
            float sum = 0;
            for (int c = 0; c < channels; c++) sum += interleaved[i * channels + c];
            mono[i] = sum / channels;
        }
        return new AudioData(mono, sampleRate);
    }

    /// <summary>Creates audio from interleaved 16-bit PCM samples, down-mixing to mono.</summary>
    public static AudioData FromPcm16(ReadOnlySpan<short> interleaved, int sampleRate, int channels = 1)
    {
        var floats = new float[interleaved.Length];
        for (int i = 0; i < floats.Length; i++) floats[i] = interleaved[i] / 32768f;
        return FromInterleaved(floats, sampleRate, channels);
    }

    /// <summary>Loads a RIFF/WAVE file (PCM 8/16/24/32-bit or IEEE float 32/64-bit, any channel count).</summary>
    public static AudioData LoadWav(string path) => LoadWav(File.ReadAllBytes(path));

    /// <summary>Loads a RIFF/WAVE stream.</summary>
    public static AudioData LoadWav(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        return LoadWav(ms.ToArray());
    }

    /// <summary>Parses RIFF/WAVE bytes.</summary>
    public static AudioData LoadWav(ReadOnlySpan<byte> wav)
    {
        if (wav.Length < 12 || !wav[..4].SequenceEqual("RIFF"u8) || !wav[8..12].SequenceEqual("WAVE"u8))
            throw new InvalidDataException("Not a RIFF/WAVE file.");
        int format = 0, channels = 0, sampleRate = 0, bits = 0;
        int pos = 12;
        while (pos + 8 <= wav.Length)
        {
            var id = wav.Slice(pos, 4);
            int size = BinaryPrimitives.ReadInt32LittleEndian(wav[(pos + 4)..]);
            var body = wav.Slice(pos + 8, Math.Min(size, wav.Length - pos - 8));
            if (id.SequenceEqual("fmt "u8))
            {
                format = BinaryPrimitives.ReadUInt16LittleEndian(body);
                channels = BinaryPrimitives.ReadUInt16LittleEndian(body[2..]);
                sampleRate = BinaryPrimitives.ReadInt32LittleEndian(body[4..]);
                bits = BinaryPrimitives.ReadUInt16LittleEndian(body[14..]);
                if (format == 0xFFFE && body.Length >= 26) format = BinaryPrimitives.ReadUInt16LittleEndian(body[24..]); // WAVE_FORMAT_EXTENSIBLE
            }
            else if (id.SequenceEqual("data"u8))
            {
                if (channels == 0) throw new InvalidDataException("WAVE data chunk before fmt chunk.");
                return FromInterleaved(Decode(body, format, bits), sampleRate, channels);
            }
            pos += 8 + size + (size & 1);
        }
        throw new InvalidDataException("WAVE file has no data chunk.");
    }

    private static float[] Decode(ReadOnlySpan<byte> data, int format, int bits)
    {
        int bytes = bits / 8;
        var result = new float[data.Length / bytes];
        for (int i = 0; i < result.Length; i++)
        {
            var s = data.Slice(i * bytes, bytes);
            result[i] = (format, bits) switch
            {
                (1, 8) => (s[0] - 128) / 128f,
                (1, 16) => BinaryPrimitives.ReadInt16LittleEndian(s) / 32768f,
                (1, 24) => ((s[0] | (s[1] << 8) | (s[2] << 16)) << 8 >> 8) / 8388608f,
                (1, 32) => BinaryPrimitives.ReadInt32LittleEndian(s) / 2147483648f,
                (3, 32) => BinaryPrimitives.ReadSingleLittleEndian(s),
                (3, 64) => (float)BinaryPrimitives.ReadDoubleLittleEndian(s),
                _ => throw new NotSupportedException($"Unsupported WAVE encoding (format {format}, {bits} bits)."),
            };
        }
        return result;
    }

    /// <summary>
    /// Returns this audio at <paramref name="targetSampleRate"/> using a Hann-windowed sinc resampler
    /// (band-limited, anti-aliased when downsampling). Returns this instance when the rate already matches.
    /// </summary>
    public AudioData Resample(int targetSampleRate)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetSampleRate);
        if (targetSampleRate == SampleRate) return this;
        const int zeroCrossings = 16;
        double ratio = (double)targetSampleRate / SampleRate;
        double cutoff = Math.Min(1.0, ratio); // normalized to the input Nyquist
        int outLength = (int)Math.Floor(Samples.Length * ratio);
        var output = new float[outLength];
        double halfWidth = zeroCrossings / cutoff;
        var input = Samples;
        Parallel.For(0, outLength, i =>
        {
            double center = i / ratio;
            int first = (int)Math.Ceiling(center - halfWidth), last = (int)Math.Floor(center + halfWidth);
            double sum = 0, weightSum = 0;
            for (int j = Math.Max(first, 0); j <= Math.Min(last, input.Length - 1); j++)
            {
                double x = (j - center) * cutoff;
                double sinc = x == 0 ? 1 : Math.Sin(Math.PI * x) / (Math.PI * x);
                double window = 0.5 + 0.5 * Math.Cos(Math.PI * (j - center) / halfWidth);
                double w = sinc * window;
                sum += input[j] * w;
                weightSum += w;
            }
            output[i] = weightSum != 0 ? (float)(sum / weightSum) : 0f;
        });
        return new AudioData(output, targetSampleRate);
    }
}
