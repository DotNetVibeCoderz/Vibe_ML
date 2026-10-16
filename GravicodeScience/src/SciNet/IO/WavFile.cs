using System.Buffers.Binary;
using System.Text;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.IO;

/// <summary>
/// Sample containers of a WAV file, and the integer type SciPy returns each one as. 1-byte
/// samples are unsigned; 3-byte samples come back left-justified in 32 bits and 5-7-byte samples
/// left-justified in 64 bits, exactly as <c>scipy.io.wavfile.read</c> does.
/// </summary>
public enum WavSampleFormat
{
    /// <summary>8-bit unsigned PCM (0..255, silence at 128).</summary>
    UInt8,
    /// <summary>16-bit signed PCM.</summary>
    Int16,
    /// <summary>24-bit signed PCM, read and written as left-justified 32-bit values (sample × 256).</summary>
    Int24,
    /// <summary>32-bit signed PCM.</summary>
    Int32,
    /// <summary>40- to 64-bit signed PCM, as left-justified 64-bit values.</summary>
    Int64,
    /// <summary>32-bit IEEE float.</summary>
    Float32,
    /// <summary>64-bit IEEE float.</summary>
    Float64,
}

/// <summary>Result of <see cref="WavFile.Read(string, bool)"/>.</summary>
/// <param name="SampleRate">Frames per second.</param>
/// <param name="Data">Samples: shape (n) for mono, (n, channels) otherwise.</param>
/// <param name="Format">The container the samples were stored in.</param>
/// <param name="BitsPerSample">Valid bits per sample as the header states (e.g. 20 for 20-bit audio in 24-bit containers).</param>
/// <param name="Channels">Number of channels.</param>
public sealed record WavData(int SampleRate, NdArray Data, WavSampleFormat Format, int BitsPerSample, int Channels);

/// <summary>
/// WAV audio — <c>scipy.io.wavfile.read</c> / <c>write</c>: PCM of any bit depth up to 64 and
/// IEEE float, plain <c>RIFF</c>, big-endian <c>RIFX</c> and <c>RF64</c> (files over 4 GiB),
/// <c>WAVE_FORMAT_EXTENSIBLE</c>, unknown chunks skipped, odd-sized chunks padded.
/// </summary>
/// <remarks>
/// Values keep SciPy's conventions: integers are the raw (left-justified) integer values held in
/// doubles, unsigned 8-bit stays 0..255. Pass <c>normalize: true</c> to get floats in [-1, 1)
/// instead. Compressed encodings (µ-law, A-law, ADPCM, ...) are refused, as SciPy refuses them.
/// </remarks>
public static class WavFile
{
    private const int FormatPcm = 1, FormatFloat = 3, FormatExtensible = 0xFFFE;

    // ------------------------------------------------------------------ reading

    /// <summary>Reads a WAV file.</summary>
    /// <param name="path">File path.</param>
    /// <param name="normalize">Scale integer samples to [-1, 1): signed values divided by 2^(bits-1) of their container type, unsigned 8-bit as (x - 128)/128.</param>
    public static WavData Read(string path, bool normalize = false)
    {
        using var f = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16);
        return Read(f, normalize);
    }

    /// <summary>Reads a WAV stream.</summary>
    /// <param name="stream">Source stream (read forward only).</param>
    /// <param name="normalize">See <see cref="Read(string, bool)"/>.</param>
    public static WavData Read(Stream stream, bool normalize = false)
    {
        var head = new byte[12];
        if (stream.ReadAtLeast(head, 12, throwOnEndOfStream: false) < 12) throw new InvalidDataException("Not a WAV file: shorter than the 12-byte RIFF header.");
        var id = Encoding.ASCII.GetString(head, 0, 4);
        var big = id == "RIFX";
        var rf64 = id == "RF64";
        if (id is not ("RIFF" or "RIFX" or "RF64")) throw new InvalidDataException($"Not a WAV file: file starts with '{id}'; only 'RIFF', 'RIFX' and 'RF64' are supported.");
        if (Encoding.ASCII.GetString(head, 8, 4) != "WAVE") throw new InvalidDataException("Not a WAV file: RIFF form type is not 'WAVE'.");

        long rf64DataSize = -1;
        int formatTag = 0, channels = 0, rate = 0, blockAlign = 0, bits = 0;
        var haveFmt = false;
        var hdr = new byte[8];
        while (true)
        {
            var got = stream.ReadAtLeast(hdr, 8, throwOnEndOfStream: false);
            if (got < 8) throw new InvalidDataException("WAV file has no 'data' chunk.");
            var chunk = Encoding.ASCII.GetString(hdr, 0, 4);
            long size = big ? BinaryPrimitives.ReadUInt32BigEndian(hdr.AsSpan(4)) : BinaryPrimitives.ReadUInt32LittleEndian(hdr.AsSpan(4));
            switch (chunk)
            {
                case "ds64" when rf64:
                {
                    var ds = ReadChunk(stream, size);
                    rf64DataSize = BinaryPrimitives.ReadInt64LittleEndian(ds.AsSpan(8));
                    break;
                }
                case "fmt ":
                {
                    var fmt = ReadChunk(stream, size);
                    if (fmt.Length < 16) throw new InvalidDataException("WAV 'fmt ' chunk is shorter than 16 bytes.");
                    int U16(int o) => big ? BinaryPrimitives.ReadUInt16BigEndian(fmt.AsSpan(o)) : BinaryPrimitives.ReadUInt16LittleEndian(fmt.AsSpan(o));
                    int I32(int o) => big ? BinaryPrimitives.ReadInt32BigEndian(fmt.AsSpan(o)) : BinaryPrimitives.ReadInt32LittleEndian(fmt.AsSpan(o));
                    formatTag = U16(0);
                    channels = U16(2);
                    rate = I32(4);
                    blockAlign = U16(12);
                    bits = U16(14);
                    if (formatTag == FormatExtensible && fmt.Length >= 26)
                        formatTag = U16(24); // first two bytes of the SubFormat GUID are the real format tag
                    haveFmt = true;
                    break;
                }
                case "data":
                {
                    if (!haveFmt) throw new InvalidDataException("WAV 'data' chunk comes before its 'fmt ' chunk.");
                    if (rf64 && size == 0xFFFFFFFF) size = rf64DataSize;
                    return ReadData(stream, size, formatTag, channels, rate, blockAlign, bits, big, normalize);
                }
                default:
                    Skip(stream, size + (size & 1)); // unknown chunk (LIST, fact, cue, junk...) — odd sizes carry a pad byte
                    continue;
            }
            if ((size & 1) == 1) Skip(stream, 1);
        }
    }

    private static byte[] ReadChunk(Stream s, long size)
    {
        if (size > 1 << 20) throw new InvalidDataException($"WAV header chunk of {size} bytes is implausibly large.");
        var b = new byte[size];
        s.ReadExactly(b);
        return b;
    }

    private static void Skip(Stream s, long n)
    {
        if (n <= 0) return;
        if (s.CanSeek) { s.Seek(n, SeekOrigin.Current); return; }
        var buf = new byte[Math.Min(n, 1 << 16)];
        while (n > 0)
        {
            var r = s.Read(buf, 0, (int)Math.Min(n, buf.Length));
            if (r == 0) return;
            n -= r;
        }
    }

    private static WavData ReadData(Stream s, long size, int formatTag, int channels, int rate, int blockAlign, int bits, bool big, bool normalize)
    {
        if (channels <= 0) throw new InvalidDataException("WAV header declares no channels.");
        if (blockAlign < channels || blockAlign % channels != 0) throw new InvalidDataException($"WAV block align {blockAlign} is not a multiple of {channels} channels.");
        var bps = blockAlign / channels;
        WavSampleFormat fmt;
        if (formatTag == FormatPcm)
        {
            if (bits is < 1 or > 64 || bps > 8) throw new NotSupportedException($"Unsupported bit depth: the WAV file has {bits}-bit integer data in {bps}-byte containers.");
            fmt = bits <= 8 ? WavSampleFormat.UInt8 : bps switch
            {
                1 => WavSampleFormat.UInt8,
                2 => WavSampleFormat.Int16,
                3 => WavSampleFormat.Int24,
                4 => WavSampleFormat.Int32,
                _ => WavSampleFormat.Int64,
            };
        }
        else if (formatTag == FormatFloat)
        {
            fmt = (bits, bps) switch
            {
                (32, 4) => WavSampleFormat.Float32,
                (64, 8) => WavSampleFormat.Float64,
                _ => throw new NotSupportedException($"Unsupported bit depth: the WAV file has {bits}-bit floating-point data."),
            };
        }
        else
        {
            var name = formatTag switch { 2 => "ADPCM", 6 => "ALAW", 7 => "MULAW", 0x11 => "IMA_ADPCM", 0x55 => "MPEGLAYER3", _ => $"0x{formatTag:X4}" };
            throw new NotSupportedException($"Unknown wave file format: {name}. Supported formats: PCM, IEEE_FLOAT.");
        }

        // A truncated file holds fewer bytes than the chunk promises; SciPy reads what is there.
        if (s.CanSeek) size = Math.Min(size, s.Length - s.Position);
        var frames = size / blockAlign;
        var total = checked((int)(frames * channels));
        var result = new double[total];
        var buf = new byte[Math.Min((long)blockAlign * frames, 1 << 20) / blockAlign * blockAlign + blockAlign];
        var done = 0;
        while (done < total)
        {
            var want = (int)Math.Min((long)(total - done) * bps, buf.Length / bps * bps);
            var got = s.ReadAtLeast(buf.AsSpan(0, want), want, throwOnEndOfStream: false);
            var n = got / bps;
            if (n == 0) break;
            Decode(buf.AsSpan(0, n * bps), result.AsSpan(done, n), fmt, bps, bits, big);
            done += n;
            if (got < want) break;
        }
        if (done < total) Array.Resize(ref result, done / channels * channels);

        if (normalize) Normalize(result, fmt);
        var data = channels == 1 ? new NdArray(result, result.Length) : new NdArray(result, result.Length / channels, channels);
        return new WavData(rate, data, fmt, bits, channels);
    }

    private static void Decode(ReadOnlySpan<byte> src, Span<double> dst, WavSampleFormat fmt, int bps, int bits, bool big)
    {
        switch (fmt)
        {
            case WavSampleFormat.UInt8 when bps == 1:
                for (var i = 0; i < dst.Length; i++) dst[i] = src[i];
                return;
            case WavSampleFormat.Int16:
                for (var i = 0; i < dst.Length; i++) dst[i] = big ? BinaryPrimitives.ReadInt16BigEndian(src[(2 * i)..]) : BinaryPrimitives.ReadInt16LittleEndian(src[(2 * i)..]);
                return;
            case WavSampleFormat.Int32:
                for (var i = 0; i < dst.Length; i++) dst[i] = big ? BinaryPrimitives.ReadInt32BigEndian(src[(4 * i)..]) : BinaryPrimitives.ReadInt32LittleEndian(src[(4 * i)..]);
                return;
            case WavSampleFormat.Float32:
                for (var i = 0; i < dst.Length; i++) dst[i] = big ? BinaryPrimitives.ReadSingleBigEndian(src[(4 * i)..]) : BinaryPrimitives.ReadSingleLittleEndian(src[(4 * i)..]);
                return;
            case WavSampleFormat.Float64:
                for (var i = 0; i < dst.Length; i++) dst[i] = big ? BinaryPrimitives.ReadDoubleBigEndian(src[(8 * i)..]) : BinaryPrimitives.ReadDoubleLittleEndian(src[(8 * i)..]);
                return;
        }
        // 3, 5, 6, 7 (and ≤8-bit data in wider containers): assemble the bytes at the most
        // significant end of the 32/64-bit result, i.e. left-justified, as SciPy does.
        var width = fmt == WavSampleFormat.Int24 ? 4 : fmt == WavSampleFormat.UInt8 ? bps : 8;
        for (var i = 0; i < dst.Length; i++)
        {
            var b = src.Slice(i * bps, bps);
            ulong v = 0;
            for (var k = 0; k < bps; k++)
            {
                var octet = big ? b[k] : b[bps - 1 - k]; // most significant first
                v = (v << 8) | octet;
            }
            if (fmt == WavSampleFormat.UInt8)
            {
                dst[i] = v; // ≤8-bit audio in a wider container: SciPy keeps it unsigned
                continue;
            }
            v <<= 8 * (width - bps);
            dst[i] = width == 4 ? (int)(uint)v : (long)v;
        }
    }

    private static void Normalize(double[] v, WavSampleFormat fmt)
    {
        switch (fmt)
        {
            case WavSampleFormat.UInt8: for (var i = 0; i < v.Length; i++) v[i] = (v[i] - 128) / 128; break;
            case WavSampleFormat.Int16: for (var i = 0; i < v.Length; i++) v[i] /= 32768.0; break;
            case WavSampleFormat.Int24:
            case WavSampleFormat.Int32: for (var i = 0; i < v.Length; i++) v[i] /= 2147483648.0; break;
            case WavSampleFormat.Int64: for (var i = 0; i < v.Length; i++) v[i] /= 9223372036854775808.0; break;
        }
    }

    // ------------------------------------------------------------------ writing

    /// <summary>Writes a WAV file (<c>scipy.io.wavfile.write</c>).</summary>
    /// <param name="path">Destination path.</param>
    /// <param name="rate">Sample rate in Hz.</param>
    /// <param name="data">Shape (n) for mono or (n, channels).</param>
    /// <param name="format">Sample container. Integer values must be exact and in range (24-bit takes the left-justified 32-bit values <see cref="Read(string, bool)"/> returns, multiples of 256).</param>
    /// <param name="normalized">Treat <paramref name="data"/> as floats in [-1, 1] and scale them to the integer format (clipped to its range).</param>
    /// <param name="extensible">Write a <c>WAVE_FORMAT_EXTENSIBLE</c> header (Microsoft's recommendation for more than two channels or more than 16 bits).</param>
    public static void Write(string path, int rate, NdArray data, WavSampleFormat format = WavSampleFormat.Int16, bool normalized = false, bool extensible = false)
    {
        using var f = File.Create(path);
        Write(f, rate, data, format, normalized, extensible);
    }

    /// <summary>Writes WAV data to a stream.</summary>
    /// <param name="stream">Destination.</param>
    /// <param name="rate">Sample rate in Hz.</param>
    /// <param name="data">Shape (n) or (n, channels).</param>
    /// <param name="format">Sample container.</param>
    /// <param name="normalized">Scale [-1, 1] floats to the integer format.</param>
    /// <param name="extensible">Write a <c>WAVE_FORMAT_EXTENSIBLE</c> header.</param>
    public static void Write(Stream stream, int rate, NdArray data, WavSampleFormat format = WavSampleFormat.Int16, bool normalized = false, bool extensible = false)
    {
        if (rate <= 0) throw new ArgumentOutOfRangeException(nameof(rate), "Sample rate must be positive.");
        if (data.Rank is not (1 or 2)) throw new ArgumentException("WAV data must be 1-D (mono) or 2-D (frames x channels).", nameof(data));
        var channels = data.Rank == 1 ? 1 : data.Shape[1];
        if (channels is < 1 or > 65535) throw new ArgumentException("Channel count must be 1..65535.", nameof(data));
        var frames = data.Rank == 1 ? data.Size : data.Shape[0];
        var bps = format switch
        {
            WavSampleFormat.UInt8 => 1,
            WavSampleFormat.Int16 => 2,
            WavSampleFormat.Int24 => 3,
            WavSampleFormat.Int32 or WavSampleFormat.Float32 => 4,
            _ => 8,
        };
        var isFloat = format is WavSampleFormat.Float32 or WavSampleFormat.Float64;
        var bits = bps * 8;
        var blockAlign = checked(channels * bps);
        var dataBytes = (long)frames * blockAlign;

        // fmt chunk: 16 bytes for PCM, 18 (cbSize = 0) for float, 40 for extensible.
        var fmt = new List<byte>();
        void U16(int v) { fmt.Add((byte)v); fmt.Add((byte)(v >> 8)); }
        void U32(uint v) { U16((int)(v & 0xFFFF)); U16((int)(v >> 16)); }
        var tag = isFloat ? FormatFloat : FormatPcm;
        U16(extensible ? FormatExtensible : tag);
        U16(channels);
        U32((uint)rate);
        U32((uint)Math.Min((long)rate * blockAlign, uint.MaxValue));
        U16(blockAlign);
        U16(bits);
        if (extensible)
        {
            U16(22);
            U16(bits); // valid bits
            U32(channels <= 18 ? (uint)((1L << channels) - 1) : 0); // speaker mask: the first N standard positions
            U16(tag);
            fmt.AddRange([0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71]); // KSDATAFORMAT_SUBTYPE GUID tail
        }
        else if (isFloat) U16(0);
        var needsFact = isFloat || extensible;

        var headerLen = 12 + 8 + fmt.Count + (needsFact ? 12 : 0) + 8;
        var pad = dataBytes & 1;
        var rf64 = headerLen - 8 + dataBytes + pad > uint.MaxValue;
        var h = new MemoryStream();
        void Id(string s) => h.Write(Encoding.ASCII.GetBytes(s));
        void W32(uint v) { Span<byte> b = stackalloc byte[4]; BinaryPrimitives.WriteUInt32LittleEndian(b, v); h.Write(b); }
        void W64(ulong v) { Span<byte> b = stackalloc byte[8]; BinaryPrimitives.WriteUInt64LittleEndian(b, v); h.Write(b); }
        if (rf64)
        {
            var riff = (ulong)(headerLen + 36 - 8 + dataBytes + pad);
            Id("RF64"); W32(0xFFFFFFFF); Id("WAVE");
            Id("ds64"); W32(28); W64(riff); W64((ulong)dataBytes); W64((ulong)frames); W32(0);
        }
        else
        {
            Id("RIFF"); W32((uint)(headerLen - 8 + dataBytes + pad)); Id("WAVE");
        }
        Id("fmt "); W32((uint)fmt.Count); h.Write(fmt.ToArray());
        if (needsFact) { Id("fact"); W32(4); W32((uint)Math.Min(frames, uint.MaxValue)); }
        Id("data"); W32(rf64 ? 0xFFFFFFFF : (uint)dataBytes);
        h.Position = 0;
        h.CopyTo(stream);

        var src = data.AsContiguous().AsSpan();
        var block = new byte[Math.Min(Math.Max(dataBytes, 1), 1 << 20) / bps * bps + bps];
        var perBlock = block.Length / bps;
        for (var start = 0; start < src.Length; start += perBlock)
        {
            var n = Math.Min(perBlock, src.Length - start);
            Encode(src.Slice(start, n), block.AsSpan(0, n * bps), format, normalized);
            stream.Write(block, 0, n * bps);
        }
        if (pad == 1) stream.WriteByte(0);
    }

    private static void Encode(ReadOnlySpan<double> src, Span<byte> dst, WavSampleFormat fmt, bool normalized)
    {
        for (var i = 0; i < src.Length; i++)
        {
            var x = src[i];
            switch (fmt)
            {
                case WavSampleFormat.Float32: BinaryPrimitives.WriteSingleLittleEndian(dst[(4 * i)..], (float)x); break;
                case WavSampleFormat.Float64: BinaryPrimitives.WriteDoubleLittleEndian(dst[(8 * i)..], x); break;
                case WavSampleFormat.UInt8:
                    dst[i] = (byte)ToInt(normalized ? x * 128 + 128 : x, 0, 255, normalized);
                    break;
                case WavSampleFormat.Int16:
                    BinaryPrimitives.WriteInt16LittleEndian(dst[(2 * i)..], (short)ToInt(normalized ? x * 32768 : x, short.MinValue, short.MaxValue, normalized));
                    break;
                case WavSampleFormat.Int24:
                {
                    long v;
                    if (normalized) v = ToInt(x * 8388608, -8388608, 8388607, true);
                    else
                    {
                        var raw = ToInt(x, int.MinValue, int.MaxValue, false);
                        if ((raw & 0xFF) != 0) throw new ArgumentException($"24-bit sample {x} is not a multiple of 256: 24-bit audio is given as left-justified 32-bit values, as WavFile.Read returns it (pass normalized: true for [-1, 1] floats).");
                        v = raw >> 8;
                    }
                    dst[3 * i] = (byte)v;
                    dst[3 * i + 1] = (byte)(v >> 8);
                    dst[3 * i + 2] = (byte)(v >> 16);
                    break;
                }
                case WavSampleFormat.Int32:
                    BinaryPrimitives.WriteInt32LittleEndian(dst[(4 * i)..], (int)ToInt(normalized ? x * 2147483648.0 : x, int.MinValue, int.MaxValue, normalized));
                    break;
                default:
                    if (normalized)
                    {
                        var y = Math.Clamp(Math.Round(x * 9223372036854775808.0), -9223372036854775808.0, 9223372036854774784.0);
                        BinaryPrimitives.WriteInt64LittleEndian(dst[(8 * i)..], (long)y);
                    }
                    else
                    {
                        if (x != Math.Floor(x) || x < -9223372036854775808.0 || x >= 9223372036854775808.0) throw new ArgumentException($"Sample {x} does not fit a 64-bit integer.");
                        BinaryPrimitives.WriteInt64LittleEndian(dst[(8 * i)..], (long)x);
                    }
                    break;
            }
        }
    }

    /// <summary>
    /// Raw values must be exact integers in range — a silently clipped or truncated sample is a
    /// plausible-looking wrong file. Normalized input is rounded and clipped, which is what scaling means.
    /// </summary>
    private static long ToInt(double x, long min, long max, bool clip)
    {
        if (double.IsNaN(x)) throw new ArgumentException("NaN cannot be written as an integer sample.");
        if (clip) return (long)Math.Clamp(Math.Round(x), min, max);
        if (x != Math.Floor(x) || x < min || x > max) throw new ArgumentException($"Sample {x} is not an integer in [{min}, {max}]; pass normalized: true for [-1, 1] data.");
        return (long)x;
    }
}
