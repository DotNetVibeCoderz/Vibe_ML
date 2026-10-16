using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.IO;

/// <summary>NetCDF classic external data types.</summary>
public enum NetCdfType
{
    /// <summary>Signed 8-bit integer.</summary>
    Byte = 1,
    /// <summary>8-bit character (text).</summary>
    Char = 2,
    /// <summary>Signed 16-bit integer.</summary>
    Short = 3,
    /// <summary>Signed 32-bit integer.</summary>
    Int = 4,
    /// <summary>32-bit IEEE float.</summary>
    Float = 5,
    /// <summary>64-bit IEEE float.</summary>
    Double = 6,
}

/// <summary>
/// NetCDF classic (CDF-1) and 64-bit-offset (CDF-2) files — <c>scipy.io.netcdf_file</c>, without
/// memory mapping: variables are read lazily from the stream on first access and cached.
/// </summary>
/// <remarks>
/// <para>Open an existing file with <see cref="Open(string, bool)"/>; build a new one with
/// <see cref="Create"/>, <see cref="CreateDimension"/>, <see cref="CreateVariable"/>, set each
/// variable's data, then <see cref="Save(string)"/>. NetCDF-4 (HDF5) and CDF-5 files are refused
/// with <see cref="NotSupportedException"/>.</para>
/// <para>Attribute values are <see cref="string"/> for text and, for numbers, a scalar
/// (<see cref="sbyte"/>, <see cref="short"/>, <see cref="int"/>, <see cref="float"/>,
/// <see cref="double"/>) when there is one value or an array of that type otherwise — SciPy's
/// convention. Variable data is always an <see cref="NdArray"/> of doubles, exact for every classic type.</para>
/// </remarks>
public sealed class NetCdfFile : IDisposable
{
    internal static readonly byte[] Magic = "CDF"u8.ToArray();
    private const int NcDimension = 0x0A, NcVariable = 0x0B, NcAttribute = 0x0C;

    private readonly Stream? _stream;
    private readonly bool _ownsStream;
    private readonly OrderedDictionary<string, int?> _dims = new(StringComparer.Ordinal);
    private readonly OrderedDictionary<string, NetCdfVariable> _vars = new(StringComparer.Ordinal);
    private int _numRecs;

    private NetCdfFile(int version, bool maskAndScale, Stream? stream, bool ownsStream)
    {
        if (version is not (1 or 2)) throw new ArgumentException("Version must be 1 (classic) or 2 (64-bit offset).", nameof(version));
        Version = version;
        MaskAndScale = maskAndScale;
        _stream = stream;
        _ownsStream = ownsStream;
    }

    /// <summary>1 for classic format, 2 for 64-bit offset format.</summary>
    public int Version { get; }

    /// <summary>
    /// When true, reading replaces values equal to <c>_FillValue</c> (or <c>missing_value</c>) by
    /// NaN and applies <c>scale_factor</c>/<c>add_offset</c>; writing reverses both (rounding to the
    /// nearest integer for integer types, where SciPy truncates).
    /// </summary>
    public bool MaskAndScale { get; }

    /// <summary>Dimensions by name; a null length marks the unlimited (record) dimension.</summary>
    public IReadOnlyDictionary<string, int?> Dimensions => _dims;

    /// <summary>Variables by name, in file order.</summary>
    public IReadOnlyDictionary<string, NetCdfVariable> Variables => _vars;

    /// <summary>Global attributes.</summary>
    public Dictionary<string, object> Attributes { get; } = new(StringComparer.Ordinal);

    /// <summary>Number of records along the unlimited dimension.</summary>
    public int RecordCount
    {
        get
        {
            if (_stream is not null) return _numRecs;
            var n = 0;
            foreach (var v in _vars.Values)
                if (v.IsRecord && v.PendingData is { } d) n = Math.Max(n, d.Rank == 0 ? 0 : d.Shape[0]);
            return n;
        }
    }

    /// <summary>Starts a new, empty dataset to be written with <see cref="Save(string)"/>.</summary>
    /// <param name="version">1 (classic, offsets below 2 GiB) or 2 (64-bit offsets).</param>
    /// <param name="maskAndScale">See <see cref="MaskAndScale"/>.</param>
    public static NetCdfFile Create(int version = 1, bool maskAndScale = false) => new(version, maskAndScale, null, false);

    /// <summary>Opens a NetCDF classic or 64-bit-offset file for reading.</summary>
    /// <param name="path">File path.</param>
    /// <param name="maskAndScale">See <see cref="MaskAndScale"/>.</param>
    public static NetCdfFile Open(string path, bool maskAndScale = false)
        => Open(new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16), maskAndScale, leaveOpen: false);

    /// <summary>Opens a NetCDF stream for reading; it must be seekable and stay open while data is read.</summary>
    /// <param name="stream">Seekable source.</param>
    /// <param name="maskAndScale">See <see cref="MaskAndScale"/>.</param>
    /// <param name="leaveOpen">Leave the stream open when the file is disposed.</param>
    public static NetCdfFile Open(Stream stream, bool maskAndScale = false, bool leaveOpen = false)
    {
        if (!stream.CanSeek) throw new ArgumentException("NetCDF reading needs a seekable stream.", nameof(stream));
        var head = new byte[8];
        stream.Position = 0;
        var n = stream.ReadAtLeast(head, 8, throwOnEndOfStream: false);
        if (n >= 4 && head[0] == 0x89 && head[1] == (byte)'H' && head[2] == (byte)'D' && head[3] == (byte)'F')
            throw new NotSupportedException("This is a NetCDF-4 / HDF5 file. SciNet reads NetCDF classic (CDF-1) and 64-bit offset (CDF-2) only; convert it with 'nccopy -k classic' or use an HDF5 library.");
        if (n < 4 || !head.AsSpan(0, 3).SequenceEqual(Magic)) throw new InvalidDataException("Not a NetCDF file: missing the 'CDF' signature.");
        if (head[3] == 5) throw new NotSupportedException("This is a CDF-5 (64-bit data) NetCDF file; SciNet reads CDF-1 and CDF-2 only.");
        if (head[3] is not (1 or 2)) throw new InvalidDataException($"Unknown NetCDF version byte {head[3]}.");
        var f = new NetCdfFile(head[3], maskAndScale, stream, !leaveOpen);
        try
        {
            stream.Position = 4;
            f.ReadHeader(new NetCdfReader(stream));
        }
        catch
        {
            f.Dispose();
            throw;
        }
        return f;
    }

    /// <summary>Adds a dimension; a null <paramref name="length"/> makes it the unlimited (record) dimension.</summary>
    public void CreateDimension(string name, int? length)
    {
        EnsureWritable();
        ValidateName(name);
        if (_dims.ContainsKey(name)) throw new ArgumentException($"Dimension '{name}' already exists.");
        if (length is null && _dims.Values.Any(d => d is null)) throw new ArgumentException("Only one unlimited dimension is allowed in a classic NetCDF file.");
        if (length is <= 0) throw new ArgumentException("Fixed dimensions must have a positive length.", nameof(length));
        _dims[name] = length;
    }

    /// <summary>Adds a variable over existing dimensions (none for a scalar). The unlimited dimension may only be the first.</summary>
    public NetCdfVariable CreateVariable(string name, NetCdfType type, params string[] dimensions)
    {
        EnsureWritable();
        ValidateName(name);
        if (_vars.ContainsKey(name)) throw new ArgumentException($"Variable '{name}' already exists.");
        for (var i = 0; i < dimensions.Length; i++)
        {
            if (!_dims.TryGetValue(dimensions[i], out var len)) throw new ArgumentException($"Unknown dimension '{dimensions[i]}'.");
            if (len is null && i != 0) throw new ArgumentException("The unlimited dimension must be a variable's first dimension.");
        }
        var v = new NetCdfVariable(this, name, type, dimensions, dimensions.Length > 0 && _dims[dimensions[0]] is null);
        _vars[name] = v;
        return v;
    }

    private void EnsureWritable()
    {
        if (_stream is not null) throw new InvalidOperationException("A NetCDF file opened for reading cannot be modified; create a new one with NetCdfFile.Create.");
    }

    private static void ValidateName(string name)
    {
        if (string.IsNullOrEmpty(name)) throw new ArgumentException("Names must not be empty.");
    }

    internal int[] ShapeOf(IReadOnlyList<string> dims, int records)
    {
        var s = new int[dims.Count];
        for (var i = 0; i < s.Length; i++) s[i] = _dims[dims[i]] ?? records;
        return s;
    }

    internal bool ReadMode => _stream is not null;

    /// <inheritdoc />
    public void Dispose()
    {
        if (_ownsStream) _stream?.Dispose();
    }

    // ------------------------------------------------------------------ reading

    internal sealed class NetCdfReader(Stream s)
    {
        private readonly byte[] _b8 = new byte[8];
        public Stream Stream { get; } = s;

        public int Int()
        {
            Stream.ReadExactly(_b8, 0, 4);
            return BinaryPrimitives.ReadInt32BigEndian(_b8);
        }

        public long Long()
        {
            Stream.ReadExactly(_b8, 0, 8);
            return BinaryPrimitives.ReadInt64BigEndian(_b8);
        }

        public byte[] Bytes(long n)
        {
            if (n < 0 || n > Stream.Length - Stream.Position) throw new InvalidDataException($"Malformed NetCDF header: {n} bytes requested at offset {Stream.Position}, file is {Stream.Length} bytes.");
            var b = new byte[n];
            Stream.ReadExactly(b);
            var pad = (4 - n % 4) % 4;
            if (pad > 0) Stream.ReadExactly(_b8, 0, (int)pad);
            return b;
        }

        public string Name() => Encoding.UTF8.GetString(Bytes(Int()));
    }

    private void ReadHeader(NetCdfReader r)
    {
        try
        {
            var numrecs = r.Int();
            var dimTag = r.Int();
            var nd = r.Int();
            if (dimTag != NcDimension && !(dimTag == 0 && nd == 0)) throw new InvalidDataException("Malformed NetCDF header: bad dimension list.");
            var dimNames = new string[nd];
            for (var i = 0; i < nd; i++)
            {
                var name = r.Name();
                var len = r.Int();
                dimNames[i] = name;
                _dims[name] = len == 0 ? null : len;
            }
            ReadAttributes(r, Attributes);
            var varTag = r.Int();
            var nv = r.Int();
            if (varTag != NcVariable && !(varTag == 0 && nv == 0)) throw new InvalidDataException("Malformed NetCDF header: bad variable list.");
            for (var i = 0; i < nv; i++)
            {
                var name = r.Name();
                var ndims = r.Int();
                var dims = new string[ndims];
                for (var k = 0; k < ndims; k++)
                {
                    var id = r.Int();
                    if ((uint)id >= (uint)nd) throw new InvalidDataException($"Malformed NetCDF header: variable '{name}' uses dimension id {id}.");
                    dims[k] = dimNames[id];
                }
                var v = new NetCdfVariable(this, name, (NetCdfType)0, dims, ndims > 0 && _dims[dims[0]] is null);
                ReadAttributes(r, v.Attributes);
                var type = r.Int();
                if (type is < 1 or > 6) throw new InvalidDataException($"Variable '{name}' has NetCDF type {type}, outside the classic types 1..6.");
                v.Type = (NetCdfType)type;
                r.Int(); // vsize: recomputed from the shape (it saturates for > 4 GiB variables)
                v.Begin = Version == 1 ? (uint)r.Int() : r.Long();
                _vars[name] = v;
            }
            // numrecs is 0xFFFFFFFF (STREAMING) when a writer did not go back to fill it in.
            if (numrecs == -1) numrecs = InferRecords();
            if (numrecs < 0) throw new InvalidDataException("Malformed NetCDF header: negative record count.");
            _numRecs = numrecs;
        }
        catch (EndOfStreamException e)
        {
            throw new InvalidDataException("Malformed NetCDF header: the file ends inside it.", e);
        }
    }

    private int InferRecords()
    {
        var rec = RecordVariables();
        if (rec.Count == 0) return 0;
        var size = RecordSize(rec);
        var start = rec.Min(v => v.Begin);
        return size == 0 ? 0 : (int)((_stream!.Length - start) / size);
    }

    internal List<NetCdfVariable> RecordVariables() => _vars.Values.Where(v => v.IsRecord).ToList();

    /// <summary>Bytes per record of one variable (all dimensions but the record one).</summary>
    internal long SlabBytes(NetCdfVariable v)
    {
        long n = TypeSize(v.Type);
        for (var i = v.IsRecord ? 1 : 0; i < v.Dimensions.Count; i++) n *= _dims[v.Dimensions[i]]!.Value;
        return n;
    }

    /// <summary>
    /// The record stride: every record variable's slab padded to 4 bytes — except that a file
    /// with a single record variable packs its records with no padding at all (the NetCDF
    /// spec's special case, which matters for byte/char/short).
    /// </summary>
    internal long RecordSize(List<NetCdfVariable> rec) => rec.Count == 1 ? SlabBytes(rec[0]) : rec.Sum(v => Pad4(SlabBytes(v)));

    private static void ReadAttributes(NetCdfReader r, Dictionary<string, object> into)
    {
        var tag = r.Int();
        var n = r.Int();
        if (tag != NcAttribute && !(tag == 0 && n == 0)) throw new InvalidDataException("Malformed NetCDF header: bad attribute list.");
        for (var i = 0; i < n; i++)
        {
            var name = r.Name();
            var type = r.Int();
            var count = r.Int();
            if (type is < 1 or > 6 || count < 0) throw new InvalidDataException($"Malformed NetCDF attribute '{name}'.");
            var raw = r.Bytes((long)count * TypeSize((NetCdfType)type));
            into[name] = DecodeAttribute((NetCdfType)type, raw, count);
        }
    }

    private static object DecodeAttribute(NetCdfType type, byte[] raw, int count)
    {
        switch (type)
        {
            case NetCdfType.Char:
                return Encoding.UTF8.GetString(raw).TrimEnd('\0');
            case NetCdfType.Byte:
            {
                var a = new sbyte[count];
                for (var i = 0; i < count; i++) a[i] = (sbyte)raw[i];
                return count == 1 ? a[0] : a;
            }
            case NetCdfType.Short:
            {
                var a = new short[count];
                for (var i = 0; i < count; i++) a[i] = BinaryPrimitives.ReadInt16BigEndian(raw.AsSpan(2 * i));
                return count == 1 ? a[0] : a;
            }
            case NetCdfType.Int:
            {
                var a = new int[count];
                for (var i = 0; i < count; i++) a[i] = BinaryPrimitives.ReadInt32BigEndian(raw.AsSpan(4 * i));
                return count == 1 ? a[0] : a;
            }
            case NetCdfType.Float:
            {
                var a = new float[count];
                for (var i = 0; i < count; i++) a[i] = BinaryPrimitives.ReadSingleBigEndian(raw.AsSpan(4 * i));
                return count == 1 ? a[0] : a;
            }
            default:
            {
                var a = new double[count];
                for (var i = 0; i < count; i++) a[i] = BinaryPrimitives.ReadDoubleBigEndian(raw.AsSpan(8 * i));
                return count == 1 ? a[0] : a;
            }
        }
    }

    internal static int TypeSize(NetCdfType t) => t switch
    {
        NetCdfType.Byte or NetCdfType.Char => 1,
        NetCdfType.Short => 2,
        NetCdfType.Int or NetCdfType.Float => 4,
        NetCdfType.Double => 8,
        _ => throw new InvalidDataException($"Unknown NetCDF type {(int)t}."),
    };

    internal static long Pad4(long n) => (n + 3) & ~3L;

    /// <summary>Reads a variable's raw values (as doubles, row-major) from the stream.</summary>
    internal double[] ReadRaw(NetCdfVariable v)
    {
        var s = _stream!;
        var shape = v.Shape;
        var total = MatLayout.Product(shape);
        var result = new double[total];
        var slab = SlabBytes(v);
        var size = TypeSize(v.Type);
        var perRecord = (int)(slab / size);
        var records = v.IsRecord ? _numRecs : 1;
        var stride = v.IsRecord ? RecordSize(RecordVariables()) : 0;
        var buf = new byte[slab];
        for (var rec = 0; rec < records; rec++)
        {
            var offset = v.Begin + rec * stride;
            if (offset + slab > s.Length) throw new InvalidDataException($"NetCDF variable '{v.Name}' runs past the end of the file (record {rec}).");
            s.Position = offset;
            s.ReadExactly(buf);
            Decode(v.Type, buf, result.AsSpan(rec * perRecord, perRecord));
        }
        return result;
    }

    private static void Decode(NetCdfType t, byte[] b, Span<double> dst)
    {
        switch (t)
        {
            case NetCdfType.Byte: for (var i = 0; i < dst.Length; i++) dst[i] = (sbyte)b[i]; break;
            case NetCdfType.Char: for (var i = 0; i < dst.Length; i++) dst[i] = b[i]; break;
            case NetCdfType.Short: for (var i = 0; i < dst.Length; i++) dst[i] = BinaryPrimitives.ReadInt16BigEndian(b.AsSpan(2 * i)); break;
            case NetCdfType.Int: for (var i = 0; i < dst.Length; i++) dst[i] = BinaryPrimitives.ReadInt32BigEndian(b.AsSpan(4 * i)); break;
            case NetCdfType.Float: for (var i = 0; i < dst.Length; i++) dst[i] = BinaryPrimitives.ReadSingleBigEndian(b.AsSpan(4 * i)); break;
            default: for (var i = 0; i < dst.Length; i++) dst[i] = BinaryPrimitives.ReadDoubleBigEndian(b.AsSpan(8 * i)); break;
        }
    }

    // ------------------------------------------------------------------ writing

    /// <summary>Writes the dataset to a file.</summary>
    public void Save(string path)
    {
        using var f = File.Create(path);
        Save(f);
    }

    /// <summary>Writes the dataset to a stream.</summary>
    public void Save(Stream output)
    {
        var numrecs = _stream is null ? RecordCount : _numRecs;
        var vars = _vars.Values.ToList();
        // Pass 1: the header's size does not depend on the begin offsets, so measure it with zeros.
        // (Offsets are kept local: a file opened for reading still needs its own to read from.)
        var begins = vars.ToDictionary(v => v, _ => 0L);
        var headerSize = WriteHeader(Stream.Null, numrecs, begins);
        long pos = headerSize;
        foreach (var v in vars.Where(v => !v.IsRecord))
        {
            begins[v] = pos;
            pos += Pad4(SlabBytes(v));
        }
        var rec = RecordVariables();
        foreach (var v in rec)
        {
            begins[v] = pos;
            pos += rec.Count == 1 ? SlabBytes(v) : Pad4(SlabBytes(v));
        }
        if (Version == 1 && begins.Values.Any(b => b > int.MaxValue))
            throw new InvalidOperationException("Variable offsets exceed 2 GiB; save with version 2 (64-bit offset).");

        WriteHeader(output, numrecs, begins);
        foreach (var v in vars.Where(v => !v.IsRecord))
        {
            var bytes = Encode(v, v.ValuesForWrite(numrecs));
            output.Write(bytes);
            var pad = Pad4(bytes.Length) - bytes.Length;
            if (pad > 0) output.Write(FillPad(v, (int)pad));
        }
        if (rec.Count > 0 && numrecs > 0)
        {
            var encoded = rec.Select(v => Encode(v, v.ValuesForWrite(numrecs))).ToList();
            for (var r = 0; r < numrecs; r++)
                for (var k = 0; k < rec.Count; k++)
                {
                    var slab = (int)SlabBytes(rec[k]);
                    output.Write(encoded[k], r * slab, slab);
                    if (rec.Count > 1)
                    {
                        var pad = (int)(Pad4(slab) - slab);
                        if (pad > 0) output.Write(FillPad(rec[k], pad));
                    }
                }
        }
    }

    private static byte[] FillPad(NetCdfVariable v, int n)
    {
        // Padding of a byte/char/short variable is filled with its fill value, per the spec.
        var fill = Encode(v, Enumerable.Repeat(v.FillValue, 4).ToArray());
        return fill.AsSpan(0, n).ToArray();
    }

    private long WriteHeader(Stream s, int numrecs, Dictionary<NetCdfVariable, long> begins)
    {
        var w = new NetCdfWriter(s);
        w.Raw(Magic);
        w.Raw([(byte)Version]);
        w.Int(numrecs);
        if (_dims.Count == 0) { w.Int(0); w.Int(0); }
        else
        {
            w.Int(NcDimension);
            w.Int(_dims.Count);
            foreach (var (name, len) in _dims) { w.Name(name); w.Int(len ?? 0); }
        }
        WriteAttributes(w, Attributes);
        if (_vars.Count == 0) { w.Int(0); w.Int(0); }
        else
        {
            w.Int(NcVariable);
            w.Int(_vars.Count);
            var dimIds = _dims.Keys.Select((k, i) => (k, i)).ToDictionary(t => t.k, t => t.i);
            foreach (var v in _vars.Values)
            {
                w.Name(v.Name);
                w.Int(v.Dimensions.Count);
                foreach (var d in v.Dimensions) w.Int(dimIds[d]);
                WriteAttributes(w, v.Attributes);
                w.Int((int)v.Type);
                var vsize = Pad4(SlabBytes(v));
                w.Int(vsize > uint.MaxValue - 4 ? -1 : (int)(uint)vsize);
                if (Version == 1) w.Int((int)begins[v]);
                else w.Long(begins[v]);
            }
        }
        return w.Count;
    }

    private sealed class NetCdfWriter(Stream s)
    {
        private readonly byte[] _b = new byte[8];
        public long Count { get; private set; }

        public void Raw(ReadOnlySpan<byte> b) { s.Write(b); Count += b.Length; }

        public void Int(int v) { BinaryPrimitives.WriteInt32BigEndian(_b, v); Raw(_b.AsSpan(0, 4)); }

        public void Long(long v) { BinaryPrimitives.WriteInt64BigEndian(_b, v); Raw(_b.AsSpan(0, 8)); }

        public void Padded(ReadOnlySpan<byte> b)
        {
            Raw(b);
            var pad = (int)(Pad4(b.Length) - b.Length);
            if (pad > 0) Raw(new byte[pad]);
        }

        public void Name(string name)
        {
            var b = Encoding.UTF8.GetBytes(name);
            Int(b.Length);
            Padded(b);
        }
    }

    private static void WriteAttributes(NetCdfWriter w, Dictionary<string, object> attrs)
    {
        if (attrs.Count == 0) { w.Int(0); w.Int(0); return; }
        w.Int(NcAttribute);
        w.Int(attrs.Count);
        foreach (var (name, value) in attrs)
        {
            w.Name(name);
            var (type, count, bytes) = EncodeAttribute(name, value);
            w.Int((int)type);
            w.Int(count);
            w.Padded(bytes);
        }
    }

    private static (NetCdfType, int, byte[]) EncodeAttribute(string name, object value)
    {
        switch (value)
        {
            case string s:
            {
                var b = Encoding.UTF8.GetBytes(s);
                return (NetCdfType.Char, b.Length, b);
            }
            case sbyte x: return EncodeAttribute(name, new[] { x });
            case short x: return EncodeAttribute(name, new[] { x });
            case int x: return EncodeAttribute(name, new[] { x });
            case float x: return EncodeAttribute(name, new[] { x });
            case double x: return EncodeAttribute(name, new[] { x });
            case sbyte[] a: return (NetCdfType.Byte, a.Length, a.Select(v => (byte)v).ToArray());
            case short[] a:
            {
                var b = new byte[2 * a.Length];
                for (var i = 0; i < a.Length; i++) BinaryPrimitives.WriteInt16BigEndian(b.AsSpan(2 * i), a[i]);
                return (NetCdfType.Short, a.Length, b);
            }
            case int[] a:
            {
                var b = new byte[4 * a.Length];
                for (var i = 0; i < a.Length; i++) BinaryPrimitives.WriteInt32BigEndian(b.AsSpan(4 * i), a[i]);
                return (NetCdfType.Int, a.Length, b);
            }
            case float[] a:
            {
                var b = new byte[4 * a.Length];
                for (var i = 0; i < a.Length; i++) BinaryPrimitives.WriteSingleBigEndian(b.AsSpan(4 * i), a[i]);
                return (NetCdfType.Float, a.Length, b);
            }
            case double[] a:
            {
                var b = new byte[8 * a.Length];
                for (var i = 0; i < a.Length; i++) BinaryPrimitives.WriteDoubleBigEndian(b.AsSpan(8 * i), a[i]);
                return (NetCdfType.Double, a.Length, b);
            }
            default:
                throw new ArgumentException($"Attribute '{name}' has type {value.GetType().Name}; NetCDF classic attributes are string, sbyte, short, int, float or double (scalars or arrays).");
        }
    }

    internal static byte[] Encode(NetCdfVariable v, ReadOnlySpan<double> values)
    {
        var size = TypeSize(v.Type);
        var b = new byte[values.Length * size];
        for (var i = 0; i < values.Length; i++)
        {
            var x = values[i];
            switch (v.Type)
            {
                case NetCdfType.Byte: b[i] = (byte)(sbyte)CheckInt(v, x, sbyte.MinValue, sbyte.MaxValue); break;
                case NetCdfType.Char: b[i] = (byte)CheckInt(v, x, 0, 255); break;
                case NetCdfType.Short: BinaryPrimitives.WriteInt16BigEndian(b.AsSpan(2 * i), (short)CheckInt(v, x, short.MinValue, short.MaxValue)); break;
                case NetCdfType.Int: BinaryPrimitives.WriteInt32BigEndian(b.AsSpan(4 * i), (int)CheckInt(v, x, int.MinValue, int.MaxValue)); break;
                case NetCdfType.Float: BinaryPrimitives.WriteSingleBigEndian(b.AsSpan(4 * i), (float)x); break;
                default: BinaryPrimitives.WriteDoubleBigEndian(b.AsSpan(8 * i), x); break;
            }
        }
        return b;
    }

    private static long CheckInt(NetCdfVariable v, double x, long min, long max)
    {
        if (x != Math.Floor(x) || x < min || x > max)
            throw new ArgumentException($"Value {x.ToString(CultureInfo.InvariantCulture)} of variable '{v.Name}' does not fit NetCDF type {v.Type}.");
        return (long)x;
    }
}

/// <summary>One variable of a <see cref="NetCdfFile"/>.</summary>
public sealed class NetCdfVariable
{
    private readonly NetCdfFile _file;
    private NdArray? _raw;

    internal NetCdfVariable(NetCdfFile file, string name, NetCdfType type, string[] dims, bool isRecord)
    {
        _file = file;
        Name = name;
        Type = type;
        Dimensions = dims;
        IsRecord = isRecord;
    }

    /// <summary>Variable name.</summary>
    public string Name { get; }

    /// <summary>External data type.</summary>
    public NetCdfType Type { get; internal set; }

    /// <summary>Dimension names, outermost first.</summary>
    public IReadOnlyList<string> Dimensions { get; }

    /// <summary>True when the first dimension is the unlimited (record) dimension.</summary>
    public bool IsRecord { get; }

    /// <summary>Per-variable attributes (<c>units</c>, <c>_FillValue</c>, <c>scale_factor</c>, ...).</summary>
    public Dictionary<string, object> Attributes { get; } = new(StringComparer.Ordinal);

    internal long Begin { get; set; }

    internal NdArray? PendingData { get; private set; }

    /// <summary>Current shape (the record dimension has the file's record count); empty for a scalar.</summary>
    public int[] Shape => _file.ShapeOf(Dimensions, _file.RecordCount);

    /// <summary>
    /// The data, row-major over <see cref="Shape"/> (a scalar has shape [1]). With
    /// <see cref="NetCdfFile.MaskAndScale"/>, fill/missing values read as NaN and
    /// <c>scale_factor</c>/<c>add_offset</c> are applied.
    /// </summary>
    public NdArray GetData()
    {
        var raw = Raw();
        if (!_file.MaskAndScale) return raw;
        var src = raw.AsContiguous().AsSpan();
        var r = new double[src.Length];
        var missing = MissingValue();
        var scale = AttributeDouble("scale_factor");
        var offset = AttributeDouble("add_offset");
        for (var i = 0; i < r.Length; i++)
        {
            var x = src[i];
            if (missing is { } m && (x == m || (double.IsNaN(m) && double.IsNaN(x)))) { r[i] = double.NaN; continue; }
            if (scale is { } s) x *= s;
            if (offset is { } o) x += o;
            r[i] = x;
        }
        return new NdArray(r, raw.Shape.ToArray());
    }

    private NdArray Raw()
    {
        if (PendingData is { } p) return p;
        if (_raw is not null) return _raw;
        var shape = Shape;
        if (!_file.ReadMode)
        {
            // A new variable with no data yet reads as its fill value, as in netCDF itself.
            var fill = new double[MatLayout.Product(shape)];
            Array.Fill(fill, FillValue);
            return new NdArray(fill, shape.Length == 0 ? [1] : shape);
        }
        _raw = new NdArray(_file.ReadRaw(this), shape.Length == 0 ? [1] : shape);
        return _raw;
    }

    /// <summary>
    /// Sets the data to write (new files only). The shape must match the dimensions, except along
    /// the record dimension, whose length becomes the record count. With
    /// <see cref="NetCdfFile.MaskAndScale"/>, NaN is stored as the missing value and values are packed
    /// with <c>scale_factor</c>/<c>add_offset</c>.
    /// </summary>
    public void SetData(NdArray data)
    {
        if (_file.ReadMode) throw new InvalidOperationException("Variables of a file opened for reading cannot be modified; copy them into a file made with NetCdfFile.Create.");
        var expected = _file.ShapeOf(Dimensions, IsRecord ? (data.Rank > 0 ? data.Shape[0] : 0) : 0);
        var got = data.Shape.ToArray();
        var scalarOk = expected.Length == 0 && data.Size == 1;
        if (!scalarOk && !expected.SequenceEqual(got))
            throw new ArgumentException($"Data shape ({string.Join(", ", got)}) does not match variable '{Name}' ({string.Join(", ", expected)}).");
        var values = data.AsContiguous().AsSpan().ToArray();
        if (_file.MaskAndScale)
        {
            var missing = MissingValue() ?? FillValue;
            var scale = AttributeDouble("scale_factor") ?? 1.0;
            var offset = AttributeDouble("add_offset") ?? 0.0;
            var integer = Type is NetCdfType.Byte or NetCdfType.Short or NetCdfType.Int or NetCdfType.Char;
            for (var i = 0; i < values.Length; i++)
            {
                if (double.IsNaN(values[i])) { values[i] = missing; continue; }
                var x = (values[i] - offset) / scale;
                values[i] = integer ? Math.Round(x, MidpointRounding.AwayFromZero) : x;
            }
        }
        NetCdfFile.Encode(this, values); // validates ranges now rather than at save time
        PendingData = new NdArray(values, scalarOk ? [1] : got);
    }

    /// <summary>Shorthand for <see cref="GetData"/> / <see cref="SetData"/>.</summary>
    public NdArray Data { get => GetData(); set => SetData(value); }

    /// <summary>The value of a scalar (or single-element) variable.</summary>
    public double GetValue()
    {
        var d = GetData();
        if (d.Size != 1) throw new InvalidOperationException($"Variable '{Name}' has {d.Size} values.");
        return d.AsContiguous().AsSpan()[0];
    }

    /// <summary>
    /// A character variable as strings along its last dimension, with trailing NULs removed
    /// (a <c>(x, strlen)</c> variable gives x strings).
    /// </summary>
    public string[] GetStrings()
    {
        if (Type != NetCdfType.Char) throw new InvalidOperationException($"Variable '{Name}' is {Type}, not Char.");
        var raw = Raw().AsContiguous().AsSpan();
        var shape = Shape;
        var len = shape.Length == 0 ? 1 : shape[^1];
        var n = len == 0 ? 0 : raw.Length / len;
        var r = new string[n];
        var bytes = new byte[len];
        for (var i = 0; i < n; i++)
        {
            for (var k = 0; k < len; k++) bytes[k] = (byte)raw[i * len + k];
            r[i] = Encoding.Latin1.GetString(bytes).TrimEnd('\0');
        }
        return r;
    }

    /// <summary>Sets a character variable from strings (one per row of its last dimension), NUL-padded.</summary>
    public void SetStrings(IReadOnlyList<string> rows)
    {
        if (Type != NetCdfType.Char) throw new InvalidOperationException($"Variable '{Name}' is {Type}, not Char.");
        if (Dimensions.Count == 0) throw new InvalidOperationException("A scalar char variable holds one character.");
        var len = _file.Dimensions[Dimensions[^1]] ?? throw new InvalidOperationException("The string length dimension cannot be unlimited.");
        var lead = _file.ShapeOf(Dimensions, IsRecord ? rows.Count : 0)[..^1];
        if (MatLayout.Product(lead) != rows.Count) throw new ArgumentException($"Variable '{Name}' needs {MatLayout.Product(lead)} strings, got {rows.Count}.");
        var v = new double[rows.Count * len];
        for (var i = 0; i < rows.Count; i++)
        {
            var b = Encoding.Latin1.GetBytes(rows[i]);
            if (b.Length > len) throw new ArgumentException($"String '{rows[i]}' is longer than the {len}-character dimension.");
            for (var k = 0; k < b.Length; k++) v[i * len + k] = b[k];
        }
        if (_file.ReadMode) throw new InvalidOperationException("Variables of a file opened for reading cannot be modified.");
        PendingData = new NdArray(v, [.. lead, len]); // text is never masked or scaled
    }

    private double? AttributeDouble(string name)
    {
        if (!Attributes.TryGetValue(name, out var v)) return null;
        return v switch
        {
            sbyte b => b, short s => s, int i => i, float f => f, double d => d,
            sbyte[] { Length: > 0 } a => a[0], short[] { Length: > 0 } a => a[0], int[] { Length: > 0 } a => a[0],
            float[] { Length: > 0 } a => a[0], double[] { Length: > 0 } a => a[0],
            string { Length: 1 } c => c[0], // a char variable's fill value is a one-character text attribute
            _ => null,
        };
    }

    /// <summary>SciPy's rule: <c>_FillValue</c> wins over <c>missing_value</c>; no masking when neither is present.</summary>
    private double? MissingValue() => AttributeDouble("_FillValue") ?? AttributeDouble("missing_value");

    /// <summary>The fill value written for unset data: <c>_FillValue</c>, else the type's NetCDF default.</summary>
    internal double FillValue => AttributeDouble("_FillValue") ?? Type switch
    {
        NetCdfType.Byte => -127,
        NetCdfType.Char => 0,
        NetCdfType.Short => -32767,
        NetCdfType.Int => -2147483647,
        NetCdfType.Float => 9.9692099683868690e+36f,
        _ => 9.9692099683868690e+36,
    };

    internal double[] ValuesForWrite(int numrecs)
    {
        var shape = _file.ShapeOf(Dimensions, numrecs);
        var n = MatLayout.Product(shape);
        var src = _file.ReadMode ? Raw() : PendingData;
        var r = new double[n];
        if (src is null) { Array.Fill(r, FillValue); return r; }
        var s = src.AsContiguous().AsSpan();
        s.CopyTo(r);
        if (s.Length < n) r.AsSpan(s.Length).Fill(FillValue); // a record variable with fewer records than the file
        return r;
    }
}
