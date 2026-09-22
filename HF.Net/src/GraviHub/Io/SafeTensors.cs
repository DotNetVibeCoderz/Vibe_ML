using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Text.Json;
using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviHub.Io;

/// <summary>The element types a safetensors file may declare.</summary>
public enum SafeTensorDType
{
    /// <summary>One byte per element, 0 or 1.</summary>
    Bool,
    /// <summary>Unsigned 8-bit integer.</summary>
    U8,
    /// <summary>Signed 8-bit integer.</summary>
    I8,
    /// <summary>8-bit float, 4-bit exponent, 3-bit mantissa, no infinities (<c>float8_e4m3fn</c>).</summary>
    F8E4M3,
    /// <summary>8-bit float, 5-bit exponent, 2-bit mantissa (<c>float8_e5m2</c>).</summary>
    F8E5M2,
    /// <summary>Unsigned 16-bit integer.</summary>
    U16,
    /// <summary>Signed 16-bit integer.</summary>
    I16,
    /// <summary>IEEE half precision.</summary>
    F16,
    /// <summary>Brain float: the top 16 bits of an IEEE single.</summary>
    BF16,
    /// <summary>Unsigned 32-bit integer.</summary>
    U32,
    /// <summary>Signed 32-bit integer.</summary>
    I32,
    /// <summary>IEEE single precision.</summary>
    F32,
    /// <summary>Unsigned 64-bit integer.</summary>
    U64,
    /// <summary>Signed 64-bit integer.</summary>
    I64,
    /// <summary>IEEE double precision.</summary>
    F64,
}

/// <summary>One tensor's entry in a safetensors header.</summary>
/// <param name="Name">The tensor's name, which is the checkpoint's parameter path.</param>
/// <param name="DType">The stored element type.</param>
/// <param name="Shape">The dimensions, outermost first. Row-major, like <see cref="NdArray"/>.</param>
/// <param name="Begin">Offset of the first byte, relative to the start of the data block.</param>
/// <param name="End">Offset one past the last byte, relative to the start of the data block.</param>
public readonly record struct SafeTensorInfo(
    string Name, SafeTensorDType DType, int[] Shape, long Begin, long End)
{
    /// <summary>Number of elements the shape implies.</summary>
    public long ElementCount
    {
        get
        {
            long n = 1;
            foreach (var d in Shape) n *= d;
            return n;
        }
    }

    /// <summary>Bytes the tensor occupies in the file.</summary>
    public long ByteCount => End - Begin;

    /// <inheritdoc />
    public override string ToString() => $"{Name} {DType} [{string.Join("x", Shape)}]";
}

/// <summary>
/// Reads and writes <c>.safetensors</c> files - the weight format the Hugging Face Hub serves.
/// </summary>
/// <remarks>
/// <para>
/// The layout is deliberately trivial: a little-endian <c>u64</c> header length, a JSON header
/// mapping each name to its dtype, shape and byte range, then one contiguous data block. Offsets
/// in the header are relative to the start of that block, <b>not</b> to the start of the file -
/// reading them as absolute produces tensors shifted by the header length and full of
/// plausible-looking numbers.
/// </para>
/// <para>
/// The file is memory-mapped rather than read into a byte array. A checkpoint is routinely larger
/// than RAM, and the common case - pulling twenty of four hundred tensors - should not pay for the
/// rest. <see cref="Open"/> gives that lazy access; <see cref="ReadAll"/> is the eager convenience.
/// </para>
/// <para>
/// Everything is widened to <see cref="double"/> on read, because that is what
/// <see cref="NdArray"/> holds. For an F32 checkpoint this doubles the memory; it is the same
/// trade the rest of the Gravicode stack makes.
/// </para>
/// </remarks>
public static class SafeTensors
{
    private const string MetadataKey = "__metadata__";

    /// <summary>The largest header this reader accepts, as a guard against a corrupt length.</summary>
    /// <remarks>
    /// A real header is kilobytes. The field is a <c>u64</c>, so a truncated or unrelated file
    /// typically asks for an absurd allocation rather than failing to parse.
    /// </remarks>
    public const long MaxHeaderBytes = 256L * 1024 * 1024;

    /// <summary>Lists the tensors in a file without reading any tensor data.</summary>
    /// <param name="path">Path to a <c>.safetensors</c> file.</param>
    /// <returns>The entries, ordered by name.</returns>
    /// <remarks>The first thing to run on an unfamiliar checkpoint.</remarks>
    public static IReadOnlyList<SafeTensorInfo> Inspect(string path)
    {
        using var reader = Open(path);
        return reader.Tensors;
    }

    /// <summary>Reads the free-form <c>__metadata__</c> map, which is often absent.</summary>
    /// <param name="path">Path to a <c>.safetensors</c> file.</param>
    public static IReadOnlyDictionary<string, string> ReadMetadata(string path)
    {
        using var reader = Open(path);
        return reader.Metadata;
    }

    /// <summary>Opens a file for lazy access, keeping it memory-mapped until disposed.</summary>
    /// <param name="path">Path to a <c>.safetensors</c> file.</param>
    public static SafeTensorsReader Open(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path)) throw new FileNotFoundException("Safetensors file not found.", path);
        return new SafeTensorsReader(path);
    }

    /// <summary>Reads every tensor in a file into memory.</summary>
    /// <param name="path">Path to a <c>.safetensors</c> file.</param>
    /// <remarks>
    /// Convenient and expensive: a 7B checkpoint in F16 is 14 GB on disk and 56 GB once widened to
    /// <see cref="double"/>. Prefer <see cref="Open"/> and pull what you need.
    /// </remarks>
    public static Dictionary<string, NdArray> ReadAll(string path)
    {
        using var reader = Open(path);
        var result = new Dictionary<string, NdArray>(StringComparer.Ordinal);
        foreach (var info in reader.Tensors) result[info.Name] = reader.Read(info.Name);
        return result;
    }

    /// <summary>
    /// Reads a sharded checkpoint's <c>model.safetensors.index.json</c>.
    /// </summary>
    /// <param name="indexPath">Path to the index file.</param>
    /// <returns>A map from tensor name to the absolute path of the shard that holds it.</returns>
    /// <remarks>
    /// Checkpoints above a few gigabytes ship as numbered shards with an index mapping every
    /// parameter to its shard. The paths in the index are relative to the index's own directory.
    /// </remarks>
    public static IReadOnlyDictionary<string, string> ReadShardIndex(string indexPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(indexPath);
        var directory = Path.GetDirectoryName(Path.GetFullPath(indexPath))!;

        using var document = JsonDocument.Parse(File.ReadAllText(indexPath));
        if (!document.RootElement.TryGetProperty("weight_map", out var map))
        {
            throw new InvalidDataException($"'{indexPath}' has no weight_map; it is not a shard index.");
        }

        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var entry in map.EnumerateObject())
        {
            result[entry.Name] = Path.Combine(directory, entry.Value.GetString()!);
        }
        return result;
    }

    /// <summary>Writes tensors to a <c>.safetensors</c> file.</summary>
    /// <param name="path">Destination path.</param>
    /// <param name="tensors">The tensors to write, keyed by name.</param>
    /// <param name="dtype">The element type to store. F32 halves the file against F64.</param>
    /// <param name="metadata">Optional free-form metadata for the <c>__metadata__</c> entry.</param>
    /// <remarks>
    /// Names are written in sorted order so the same tensors always produce the same bytes - a
    /// checkpoint that is diffable and hashable is worth more than one written in dictionary order.
    /// </remarks>
    public static void Write(
        string path,
        IReadOnlyDictionary<string, NdArray> tensors,
        SafeTensorDType dtype = SafeTensorDType.F32,
        IReadOnlyDictionary<string, string>? metadata = null)
        => Write(path, tensors, _ => dtype, metadata);

    /// <summary>Writes tensors, choosing the element type per tensor.</summary>
    /// <param name="path">Destination path.</param>
    /// <param name="tensors">The tensors to write, keyed by name.</param>
    /// <param name="dtypeOf">Picks the element type for each tensor, by name.</param>
    /// <param name="metadata">Optional free-form metadata for the <c>__metadata__</c> entry.</param>
    /// <remarks>
    /// The per-tensor choice exists because a checkpoint is not uniformly weights. A BERT
    /// checkpoint carries an integer <c>position_ids</c> buffer holding values up to 511, and
    /// bfloat16 has eight mantissa bits - so quantising it along with everything else rounds 511 to
    /// 512 and reports a maximum error of 1.0 that has nothing to do with the weights.
    /// </remarks>
    public static void Write(
        string path,
        IReadOnlyDictionary<string, NdArray> tensors,
        Func<string, SafeTensorDType> dtypeOf,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(tensors);
        ArgumentNullException.ThrowIfNull(dtypeOf);

        var names = tensors.Keys.OrderBy(n => n, StringComparer.Ordinal).ToArray();
        var dtypes = new Dictionary<string, SafeTensorDType>(StringComparer.Ordinal);

        foreach (var name in names)
        {
            var dtype = dtypeOf(name);

            if (dtype is not (SafeTensorDType.F32 or SafeTensorDType.F64
                or SafeTensorDType.F16 or SafeTensorDType.BF16))
            {
                throw new NotSupportedException(
                    $"Writing {dtype} (for '{name}') is not supported; NdArray holds double, so F16, "
                    + "BF16, F32 and F64 are the meaningful targets.");
            }

            dtypes[name] = dtype;
        }

        // The header is built first: every byte range has to be known before any data is written.
        var header = new Dictionary<string, object>(StringComparer.Ordinal);
        if (metadata is { Count: > 0 })
        {
            header[MetadataKey] = metadata.ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal);
        }

        long offset = 0;
        foreach (var name in names)
        {
            var tensor = tensors[name];
            var bytes = (long)tensor.Size * ElementSize(dtypes[name]);
            header[name] = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                ["dtype"] = DTypeName(dtypes[name]),
                ["shape"] = tensor.Shape.ToArray(),
                ["data_offsets"] = new[] { offset, offset + bytes },
            };
            offset += bytes;
        }

        var headerJson = JsonSerializer.SerializeToUtf8Bytes(header);

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20);
        Span<byte> length = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(length, (ulong)headerJson.Length);
        stream.Write(length);
        stream.Write(headerJson);

        var buffer = new byte[1 << 16];

        foreach (var name in names)
        {
            var dtype = dtypes[name];
            var width = ElementSize(dtype);
            var perBuffer = buffer.Length / width;
            var data = tensors[name].AsContiguous().ToArray();

            for (var i = 0; i < data.Length; i += perBuffer)
            {
                var count = Math.Min(perBuffer, data.Length - i);
                EncodeBlock(data.AsSpan(i, count), buffer.AsSpan(0, count * width), dtype);
                stream.Write(buffer, 0, count * width);
            }
        }
    }

    // ------------------------------------------------------------------ header parsing

    internal static (IReadOnlyList<SafeTensorInfo> Tensors, Dictionary<string, string> Metadata, long DataStart)
        ParseHeader(ReadOnlySpan<byte> prefix, long fileLength)
    {
        if (prefix.Length < 8) throw new InvalidDataException("File is too short to be safetensors.");

        var headerLength = (long)BinaryPrimitives.ReadUInt64LittleEndian(prefix);
        if (headerLength <= 0 || headerLength > MaxHeaderBytes || 8 + headerLength > fileLength)
        {
            throw new InvalidDataException(
                $"Safetensors header length {headerLength} is not plausible for a {fileLength} byte file.");
        }

        var json = prefix.Slice(8, (int)headerLength);
        var dataStart = 8 + headerLength;

        using var document = JsonDocument.Parse(json.ToArray());
        var tensors = new List<SafeTensorInfo>();
        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (var entry in document.RootElement.EnumerateObject())
        {
            if (entry.Name == MetadataKey)
            {
                foreach (var m in entry.Value.EnumerateObject())
                {
                    metadata[m.Name] = m.Value.ValueKind == JsonValueKind.String
                        ? m.Value.GetString() ?? ""
                        : m.Value.GetRawText();
                }
                continue;
            }

            var dtype = ParseDType(entry.Value.GetProperty("dtype").GetString()!);

            var shapeElement = entry.Value.GetProperty("shape");
            var shape = new int[shapeElement.GetArrayLength()];
            var axis = 0;
            foreach (var d in shapeElement.EnumerateArray()) shape[axis++] = d.GetInt32();

            // A zero-rank tensor is legal and means a scalar. NdArray has no rank 0, so it becomes
            // a one-element vector on read.
            if (shape.Length == 0) shape = [1];

            var offsets = entry.Value.GetProperty("data_offsets");
            var begin = offsets[0].GetInt64();
            var end = offsets[1].GetInt64();

            if (begin < 0 || end < begin || dataStart + end > fileLength)
            {
                throw new InvalidDataException(
                    $"Tensor '{entry.Name}' claims bytes [{begin}, {end}) which do not fit the file.");
            }

            var info = new SafeTensorInfo(entry.Name, dtype, shape, begin, end);

            var expected = info.ElementCount * ElementSize(dtype);
            if (expected != info.ByteCount)
            {
                throw new InvalidDataException(
                    $"Tensor '{entry.Name}' is [{string.Join("x", shape)}] of {dtype}, which needs {expected} bytes, but the header reserves {info.ByteCount}.");
            }

            tensors.Add(info);
        }

        tensors.Sort(static (a, b) => string.CompareOrdinal(a.Name, b.Name));
        return (tensors, metadata, dataStart);
    }

    // ------------------------------------------------------------------ dtype plumbing

    internal static int ElementSize(SafeTensorDType dtype) => dtype switch
    {
        SafeTensorDType.Bool or SafeTensorDType.U8 or SafeTensorDType.I8
            or SafeTensorDType.F8E4M3 or SafeTensorDType.F8E5M2 => 1,
        SafeTensorDType.U16 or SafeTensorDType.I16 or SafeTensorDType.F16 or SafeTensorDType.BF16 => 2,
        SafeTensorDType.U32 or SafeTensorDType.I32 or SafeTensorDType.F32 => 4,
        SafeTensorDType.U64 or SafeTensorDType.I64 or SafeTensorDType.F64 => 8,
        _ => throw new NotSupportedException($"Unknown dtype {dtype}."),
    };

    internal static SafeTensorDType ParseDType(string name) => name.ToUpperInvariant() switch
    {
        "BOOL" => SafeTensorDType.Bool,
        "U8" => SafeTensorDType.U8,
        "I8" => SafeTensorDType.I8,
        "F8_E4M3" => SafeTensorDType.F8E4M3,
        "F8_E5M2" => SafeTensorDType.F8E5M2,
        "U16" => SafeTensorDType.U16,
        "I16" => SafeTensorDType.I16,
        "F16" => SafeTensorDType.F16,
        "BF16" => SafeTensorDType.BF16,
        "U32" => SafeTensorDType.U32,
        "I32" => SafeTensorDType.I32,
        "F32" => SafeTensorDType.F32,
        "U64" => SafeTensorDType.U64,
        "I64" => SafeTensorDType.I64,
        "F64" => SafeTensorDType.F64,
        _ => throw new NotSupportedException($"Unsupported safetensors dtype '{name}'."),
    };

    internal static string DTypeName(SafeTensorDType dtype) => dtype switch
    {
        SafeTensorDType.Bool => "BOOL",
        SafeTensorDType.U8 => "U8",
        SafeTensorDType.I8 => "I8",
        SafeTensorDType.F8E4M3 => "F8_E4M3",
        SafeTensorDType.F8E5M2 => "F8_E5M2",
        SafeTensorDType.U16 => "U16",
        SafeTensorDType.I16 => "I16",
        SafeTensorDType.F16 => "F16",
        SafeTensorDType.BF16 => "BF16",
        SafeTensorDType.U32 => "U32",
        SafeTensorDType.I32 => "I32",
        SafeTensorDType.F32 => "F32",
        SafeTensorDType.U64 => "U64",
        SafeTensorDType.I64 => "I64",
        SafeTensorDType.F64 => "F64",
        _ => throw new NotSupportedException($"Unknown dtype {dtype}."),
    };

    /// <summary>Widens one stored element to <see cref="double"/>.</summary>
    internal static double Decode(ReadOnlySpan<byte> source, SafeTensorDType dtype) => dtype switch
    {
        SafeTensorDType.Bool => source[0] != 0 ? 1.0 : 0.0,
        SafeTensorDType.U8 => source[0],
        SafeTensorDType.I8 => (sbyte)source[0],
        SafeTensorDType.F8E4M3 => DecodeF8E4M3(source[0]),
        SafeTensorDType.F8E5M2 => DecodeF8E5M2(source[0]),
        SafeTensorDType.U16 => BinaryPrimitives.ReadUInt16LittleEndian(source),
        SafeTensorDType.I16 => BinaryPrimitives.ReadInt16LittleEndian(source),
        SafeTensorDType.F16 => (double)BitConverter.Int16BitsToHalf(BinaryPrimitives.ReadInt16LittleEndian(source)),
        SafeTensorDType.BF16 => BitConverter.UInt32BitsToSingle((uint)BinaryPrimitives.ReadUInt16LittleEndian(source) << 16),
        SafeTensorDType.U32 => BinaryPrimitives.ReadUInt32LittleEndian(source),
        SafeTensorDType.I32 => BinaryPrimitives.ReadInt32LittleEndian(source),
        SafeTensorDType.F32 => BinaryPrimitives.ReadSingleLittleEndian(source),
        SafeTensorDType.U64 => BinaryPrimitives.ReadUInt64LittleEndian(source),
        SafeTensorDType.I64 => BinaryPrimitives.ReadInt64LittleEndian(source),
        SafeTensorDType.F64 => BinaryPrimitives.ReadDoubleLittleEndian(source),
        _ => throw new NotSupportedException($"Unknown dtype {dtype}."),
    };

    /// <summary>Decodes a contiguous run of elements, taking the fast path where one exists.</summary>
    internal static void DecodeBlock(ReadOnlySpan<byte> source, Span<double> destination, SafeTensorDType dtype)
    {
        var width = ElementSize(dtype);

        switch (dtype)
        {
            // F64 and F32 are bit-for-bit reinterpretable on every platform .NET runs on, so the
            // per-element switch is skipped: this is the hot path for real checkpoints.
            case SafeTensorDType.F64 when BitConverter.IsLittleEndian:
                MemoryMarshal.Cast<byte, double>(source).CopyTo(destination);
                return;

            case SafeTensorDType.F32 when BitConverter.IsLittleEndian:
            {
                var floats = MemoryMarshal.Cast<byte, float>(source);
                for (var i = 0; i < floats.Length; i++) destination[i] = floats[i];
                return;
            }

            default:
                for (var i = 0; i < destination.Length; i++)
                {
                    destination[i] = Decode(source.Slice(i * width, width), dtype);
                }
                return;
        }
    }

    private static void EncodeBlock(ReadOnlySpan<double> source, Span<byte> destination, SafeTensorDType dtype)
    {
        var width = ElementSize(dtype);
        for (var i = 0; i < source.Length; i++)
        {
            var slot = destination.Slice(i * width, width);
            switch (dtype)
            {
                case SafeTensorDType.F64:
                    BinaryPrimitives.WriteDoubleLittleEndian(slot, source[i]);
                    break;
                case SafeTensorDType.F32:
                    BinaryPrimitives.WriteSingleLittleEndian(slot, (float)source[i]);
                    break;
                case SafeTensorDType.F16:
                    BinaryPrimitives.WriteInt16LittleEndian(slot, BitConverter.HalfToInt16Bits((Half)source[i]));
                    break;
                case SafeTensorDType.BF16:
                    BinaryPrimitives.WriteUInt16LittleEndian(slot, ToBFloat16((float)source[i]));
                    break;
                default:
                    throw new NotSupportedException($"Writing {dtype} is not supported.");
            }
        }
    }

    /// <summary>Narrows a single to bfloat16, rounding to nearest even.</summary>
    /// <remarks>
    /// Truncating the low 16 bits is the obvious implementation and it biases every weight towards
    /// zero. The bias is small per value and systematic across a whole checkpoint, which is exactly
    /// the kind of error that survives a round-trip test written with a loose tolerance.
    /// </remarks>
    private static ushort ToBFloat16(float value)
    {
        var bits = BitConverter.SingleToUInt32Bits(value);
        if (float.IsNaN(value)) return (ushort)((bits >> 16) | 0x0040);

        var rounding = 0x7FFFu + ((bits >> 16) & 1);
        return (ushort)((bits + rounding) >> 16);
    }

    /// <summary>Widens <c>float8_e4m3fn</c>: 1 sign, 4 exponent (bias 7), 3 mantissa, no infinities.</summary>
    /// <remarks>
    /// The <c>fn</c> suffix is "finite": the all-ones exponent holds normal values and only the
    /// all-ones mantissa beside it is NaN. Decoding it as IEEE yields infinities where the format
    /// stores 448, which then poisons every downstream sum.
    /// </remarks>
    private static double DecodeF8E4M3(byte raw)
    {
        var sign = (raw & 0x80) != 0 ? -1.0 : 1.0;
        var exponent = (raw >> 3) & 0x0F;
        var mantissa = raw & 0x07;

        if (exponent == 0x0F && mantissa == 0x07) return double.NaN;
        if (exponent == 0) return sign * Math.ScaleB(mantissa / 8.0, -6);
        return sign * Math.ScaleB(1.0 + mantissa / 8.0, exponent - 7);
    }

    /// <summary>Widens <c>float8_e5m2</c>: 1 sign, 5 exponent (bias 15), 2 mantissa, IEEE specials.</summary>
    private static double DecodeF8E5M2(byte raw)
    {
        var sign = (raw & 0x80) != 0 ? -1.0 : 1.0;
        var exponent = (raw >> 2) & 0x1F;
        var mantissa = raw & 0x03;

        if (exponent == 0x1F) return mantissa == 0 ? sign * double.PositiveInfinity : double.NaN;
        if (exponent == 0) return sign * Math.ScaleB(mantissa / 4.0, -14);
        return sign * Math.ScaleB(1.0 + mantissa / 4.0, exponent - 15);
    }
}

/// <summary>
/// A memory-mapped safetensors file. Holds the mapping open so tensors can be pulled one at a
/// time; dispose it when done.
/// </summary>
public sealed class SafeTensorsReader : IDisposable
{
    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view = null!;
    private readonly Dictionary<string, SafeTensorInfo> _byName;
    private readonly long _dataStart;
    private bool _disposed;

    internal SafeTensorsReader(string path)
    {
        Path = path;
        var length = new FileInfo(path).Length;

        _file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, mapName: null, 0, MemoryMappedFileAccess.Read);

        try
        {
            _view = _file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

            // The declared header length is read first, and then exactly that many bytes. Reading
            // a fixed MaxHeaderBytes prefix instead would allocate and copy 256 MB to parse a
            // header of about twenty kilobytes - which measured 429 ms against the 0.65 ms the
            // reference implementation takes, for no reason other than the size of the buffer.
            var prefixLength = (int)Math.Min(length, 8);
            var prefix = new byte[prefixLength];
            _view.ReadArray(0, prefix, 0, prefixLength);

            if (prefixLength == 8)
            {
                var declared = (long)BinaryPrimitives.ReadUInt64LittleEndian(prefix);
                if (declared > 0 && declared <= SafeTensors.MaxHeaderBytes && 8 + declared <= length)
                {
                    prefix = new byte[8 + declared];
                    _view.ReadArray(0, prefix, 0, prefix.Length);
                }
            }

            var (tensors, metadata, dataStart) = SafeTensors.ParseHeader(prefix, length);
            Tensors = tensors;
            Metadata = metadata;
            _dataStart = dataStart;
            _byName = tensors.ToDictionary(t => t.Name, StringComparer.Ordinal);
        }
        catch
        {
            // A malformed header throws after the mapping is open. Without this the file stays
            // locked for the life of the process, so the caller cannot even delete the bad file -
            // and on Windows the next attempt to open it fails with a sharing violation that points
            // nowhere near the real cause.
            _view?.Dispose();
            _file.Dispose();
            throw;
        }
    }

    /// <summary>The file this reader was opened on.</summary>
    public string Path { get; }

    /// <summary>Every tensor in the file, ordered by name.</summary>
    public IReadOnlyList<SafeTensorInfo> Tensors { get; }

    /// <summary>The file's <c>__metadata__</c> map, empty when it has none.</summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    /// <summary>Whether a tensor of that name is present.</summary>
    public bool Contains(string name) => _byName.ContainsKey(name);

    /// <summary>The header entry for a tensor, or <c>null</c> when it is absent.</summary>
    public SafeTensorInfo? Info(string name) => _byName.TryGetValue(name, out var info) ? info : null;

    /// <summary>Reads one tensor, widening it to <see cref="double"/>.</summary>
    /// <param name="name">The tensor's name.</param>
    /// <exception cref="KeyNotFoundException">The file has no tensor with that name.</exception>
    public NdArray Read(string name)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_byName.TryGetValue(name, out var info))
        {
            throw new KeyNotFoundException($"'{name}' is not in {System.IO.Path.GetFileName(Path)}.");
        }

        if (info.ElementCount > int.MaxValue)
        {
            throw new NotSupportedException(
                $"'{name}' has {info.ElementCount:N0} elements; NdArray is indexed by int and tops out at {int.MaxValue:N0}.");
        }

        var count = (int)info.ElementCount;
        var width = SafeTensors.ElementSize(info.DType);
        var data = new double[count];

        // Copied through a staging buffer rather than in one read: a 4 GB tensor would otherwise
        // need a 4 GB byte[] alongside the 8 GB double[] that is the actual result.
        const int ChunkElements = 1 << 16;
        var staging = new byte[ChunkElements * width];

        for (var i = 0; i < count; i += ChunkElements)
        {
            var take = Math.Min(ChunkElements, count - i);
            var bytes = take * width;
            _view.ReadArray(_dataStart + info.Begin + (long)i * width, staging, 0, bytes);
            SafeTensors.DecodeBlock(staging.AsSpan(0, bytes), data.AsSpan(i, take), info.DType);
        }

        return new NdArray(data, info.Shape);
    }

    /// <summary>Reads a tensor if present, without throwing when it is not.</summary>
    public bool TryRead(string name, out NdArray tensor)
    {
        if (_byName.ContainsKey(name))
        {
            tensor = Read(name);
            return true;
        }

        tensor = null!;
        return false;
    }

    /// <summary>Total size of the tensor data, excluding the header.</summary>
    public long DataBytes => Tensors.Count == 0 ? 0 : Tensors.Max(t => t.End);

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _view.Dispose();
        _file.Dispose();
    }

    /// <inheritdoc />
    public override string ToString()
        => $"{System.IO.Path.GetFileName(Path)}: {Tensors.Count} tensors, {DataBytes / (1024.0 * 1024):0.#} MB";
}
