using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.IO;
using Xunit;
using static Gravicode.Science.Tests.SciNet.IOAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// WAV tests. wav_*.wav were written by scipy 1.17.1 wavfile.write (wav_oddchunk.wav by hand:
/// an odd-sized unknown chunk before 'data'); scipy/test-*.wav come from SciPy's test suite
/// (BSD-3-Clause) and cover 12/20/24/36-bit PCM, RIFX, RF64, WAVE_FORMAT_EXTENSIBLE and µ-law.
/// Expected values are scipy 1.17.1's wavfile.read output. Files written by SciNet in every
/// format (including extensible and 24-bit) were read back by scipy during development and matched.
/// </summary>
public class WavFileTests
{
    private static string Scipy(string name) => Fixture(Path.Combine("scipy", name));

    private static void Check(WavData w, int rate, int[] shape, WavSampleFormat fmt, double[] first, double sum, double sumTol = 0)
    {
        Assert.Equal(rate, w.SampleRate);
        Assert.Equal(shape, w.Data.Shape.ToArray());
        Assert.Equal(fmt, w.Format);
        var all = w.Data.ToArray();
        Close(first, all[..first.Length], 0, "first samples");
        Close(sum, all.Sum(), sumTol, "sum");
    }

    [Fact]
    public void ReadsScipyWrittenFiles()
    {
        Check(WavFile.Read(Fixture("wav_u8.wav")), 8000, [50], WavSampleFormat.UInt8, [128, 160, 189, 212, 225, 227], 6890);
        Check(WavFile.Read(Fixture("wav_i16_stereo.wav")), 44100, [50, 2], WavSampleFormat.Int16, [0, -20000, 1986, -19796, 3894, -19189], -15004);
        Check(WavFile.Read(Fixture("wav_i32.wav")), 22050, [50], WavSampleFormat.Int32, [0, 494807918, 958851077, 1363277520, 1682941969, 1897969238], 83843833);
        Check(WavFile.Read(Fixture("wav_f32.wav")), 16000, [50], WavSampleFormat.Float32,
            [0.0, 0.1244221031665802, 0.24539601802825928, 0.35956916213035583, 0.4637773633003235, 0.5551326274871826], 6.227454906329513, 1e-12);
        Check(WavFile.Read(Fixture("wav_f64_3ch.wav")), 48000, [50, 3], WavSampleFormat.Float64,
            [0.0, 1.0, 0.0, 0.3271946967961522, 0.9449569463147377, 0.02], 27.943874974173184, 1e-12);
        Check(WavFile.Read(Fixture("wav_oddchunk.wav")), 11025, [5], WavSampleFormat.Int16, [0, 1000, -1000, 32767, -32768], -1);
    }

    [Fact]
    public void ReadsScipyTestSuiteFiles()
    {
        // 24-bit in 3-byte containers: left-justified int32, little- and big-endian (RIFX) and RF64.
        double[] s24 = [-2147483648, -2147483392, -512, -1073741824, -1073741568, -256];
        Check(WavFile.Read(Scipy("test-8000Hz-le-3ch-5S-24bit.wav")), 8000, [5, 3], WavSampleFormat.Int24, s24, -256);
        Check(WavFile.Read(Scipy("test-8000Hz-be-3ch-5S-24bit.wav")), 8000, [5, 3], WavSampleFormat.Int24, s24, -256);
        Check(WavFile.Read(Scipy("test-8000Hz-le-3ch-5S-24bit-rf64.wav")), 8000, [5, 3], WavSampleFormat.Int24, s24, -256);
        // 20-bit audio in 3-byte containers, plus an extra chunk.
        var w20 = WavFile.Read(Scipy("test-1234Hz-le-1ch-10S-20bit-extra.wav"));
        Check(w20, 1234, [10], WavSampleFormat.Int24, [2147479552, -2147479552, 1073739776, -1073739776, 536869888, -536869888], 0);
        Assert.Equal(20, w20.BitsPerSample);
        // 36-bit audio in 5-byte containers: left-justified int64.
        Check(WavFile.Read(Scipy("test-8000Hz-le-3ch-5S-36bit.wav")), 8000, [5, 3], WavSampleFormat.Int64,
            [-9223372036854775808.0, -9223372036586340352.0, -536870912, -4611686018427387904.0, -4611686018158952448.0, -268435456], -268435456);
        Check(WavFile.Read(Scipy("test-8000Hz-le-4ch-9S-12bit.wav")), 8000, [9, 4], WavSampleFormat.Int16, [0, 0, 0, 0, 23168, 32752], -128);
        Check(WavFile.Read(Scipy("test-8000Hz-le-2ch-1byteu.wav")), 8000, [800, 2], WavSampleFormat.UInt8, [136, 136, 189, 189, 217, 218], 204805);
        Check(WavFile.Read(Scipy("test-44100Hz-2ch-32bit-float-be.wav")), 44100, [441, 2], WavSampleFormat.Float32,
            [0.0, 0.0, 0.05011868476867676, 0.05011868476867676, 0.10004043579101562, 0.10004043579101562], 45.6856164932251, 1e-9);
        Check(WavFile.Read(Scipy("test-48000Hz-2ch-64bit-float-le-wavex.wav")), 48000, [480, 2], WavSampleFormat.Float64,
            [0.0, 0.0, 0.04605122283101082, 0.04605122283101082, 0.09194972366094589, 0.09194972366094589], 49.76995729841292, 1e-9);

        var e = Assert.Throws<NotSupportedException>(() => WavFile.Read(Scipy("test-8000Hz-le-1ch-1byte-ulaw.wav")));
        Assert.Contains("MULAW", e.Message);
    }

    [Fact]
    public void Normalize_ScalesToUnitRange()
    {
        var u8 = WavFile.Read(Fixture("wav_u8.wav"), normalize: true).Data.ToArray();
        Assert.Equal(0.0, u8[0]);
        Assert.Equal(32.0 / 128, u8[1]);
        var i16 = WavFile.Read(Fixture("wav_oddchunk.wav"), normalize: true).Data.ToArray();
        Close([0, 1000 / 32768.0, -1000 / 32768.0, 32767 / 32768.0, -1], i16, 0);
        var i24 = WavFile.Read(Scipy("test-8000Hz-le-3ch-5S-24bit.wav"), normalize: true).Data.ToArray();
        Assert.Equal(-1.0, i24[0]);
    }

    [Theory]
    [InlineData(WavSampleFormat.UInt8, false)]
    [InlineData(WavSampleFormat.Int16, false)]
    [InlineData(WavSampleFormat.Int24, false)]
    [InlineData(WavSampleFormat.Int32, false)]
    [InlineData(WavSampleFormat.Int64, false)]
    [InlineData(WavSampleFormat.Float32, false)]
    [InlineData(WavSampleFormat.Float64, false)]
    [InlineData(WavSampleFormat.Int24, true)]
    [InlineData(WavSampleFormat.Float32, true)]
    public void Write_RoundTrips(WavSampleFormat fmt, bool extensible)
    {
        const int frames = 7; // odd: 8-bit mono data needs a pad byte
        var rng = new Random(5);
        var mono = new double[frames];
        var stereo = new double[frames * 3];
        double Sample() => fmt switch
        {
            WavSampleFormat.UInt8 => rng.Next(256),
            WavSampleFormat.Int16 => rng.Next(-32768, 32768),
            WavSampleFormat.Int24 => rng.Next(-8388608, 8388608) * 256.0,
            WavSampleFormat.Int32 => rng.Next(int.MinValue, int.MaxValue),
            WavSampleFormat.Int64 => (double)rng.NextInt64(long.MinValue / 2048, long.MaxValue / 2048) * 2048,
            WavSampleFormat.Float32 => (float)(rng.NextDouble() * 2 - 1),
            _ => rng.NextDouble() * 2 - 1,
        };
        for (var i = 0; i < mono.Length; i++) mono[i] = Sample();
        for (var i = 0; i < stereo.Length; i++) stereo[i] = Sample();
        foreach (var data in new[] { new NdArray(mono, frames), new NdArray(stereo, frames, 3) })
        {
            var ms = new MemoryStream();
            WavFile.Write(ms, 22050, data, fmt, extensible: extensible);
            Assert.Equal(0, ms.Length % 2); // RIFF chunks are word aligned
            ms.Position = 0;
            var back = WavFile.Read(ms);
            Assert.Equal(22050, back.SampleRate);
            Assert.Equal(fmt, back.Format);
            Assert.Equal(data.Shape.ToArray(), back.Data.Shape.ToArray());
            Close(data.ToArray(), back.Data.ToArray(), 0, fmt.ToString());
        }
    }

    [Fact]
    public void Write_Normalized_AndRangeChecks()
    {
        var ms = new MemoryStream();
        WavFile.Write(ms, 8000, new NdArray([0.0, 0.5, -1.0, 1.0, 2.0], 5), WavSampleFormat.Int16, normalized: true);
        ms.Position = 0;
        Close([0, 16384, -32768, 32767, 32767], WavFile.Read(ms).Data.ToArray(), 0); // clipped at full scale

        ms = new MemoryStream();
        WavFile.Write(ms, 8000, new NdArray([0.0, -1.0, 0.5], 3), WavSampleFormat.Int24, normalized: true);
        ms.Position = 0;
        Close([0, -2147483648.0, 1073741824.0], WavFile.Read(ms).Data.ToArray(), 0);

        Assert.Throws<ArgumentException>(() => WavFile.Write(new MemoryStream(), 8000, new NdArray([40000.0], 1), WavSampleFormat.Int16));
        Assert.Throws<ArgumentException>(() => WavFile.Write(new MemoryStream(), 8000, new NdArray([0.5], 1), WavSampleFormat.Int16));
        Assert.Throws<ArgumentException>(() => WavFile.Write(new MemoryStream(), 8000, new NdArray([300.0], 1), WavSampleFormat.Int24)); // not left-justified
        Assert.Throws<InvalidDataException>(() => WavFile.Read(new MemoryStream("RIFX\0\0\0\0WAVX"u8.ToArray())));
        Assert.Throws<InvalidDataException>(() => WavFile.Read(new MemoryStream("OggS\0\0\0\0WAVE"u8.ToArray())));
    }
}
