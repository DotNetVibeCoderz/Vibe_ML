using System.Buffers.Binary;
using System.IO.Compression;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.IO;

/// <summary>
/// Builds a new HDF5 file in memory and writes it in one pass: groups, datasets and attributes,
/// with numeric, boolean, complex, string and object-reference data. Files open in h5py / HDF5
/// 1.8 and later, MATLAB (v7.3 MAT files are written through this) and netCDF-4.
/// </summary>
/// <remarks>
/// <para>Element types come from the .NET data: <c>double</c>, <c>float</c>, <see cref="Half"/>,
/// <c>sbyte</c> ... <c>ulong</c> as the matching IEEE/integer types (little-endian); <c>bool</c> as
/// h5py's boolean enum; <see cref="Complex"/> as h5py's <c>{r, i}</c> compound of float64;
/// <c>string</c> as variable-length UTF-8 (what h5py writes for <c>str</c>) or, on request,
/// fixed-length strings; <see cref="Hdf5WriterNode"/> as object references. Multi-dimensional .NET
/// arrays keep their shape; <see cref="NdArray"/> and <see cref="ComplexNdArray"/> are accepted.</para>
/// <para>Datasets are contiguous. A dataset with a compression level, or with an unlimited
/// maximum dimension, is chunked and stored as a single chunk covering the current extent
/// (deflate-compressed when asked); readers that slice it decompress that whole chunk.</para>
/// <para>Format: superblock version 2 with version 2 object headers, and groups that keep their
/// links in the object header (compact link storage). That is HDF5 1.8's format — the one
/// netCDF-4 writes — and it was chosen over the 1.6 format (superblock 0, version 1 headers)
/// because a 1.6 group is a v1 B-tree of symbol-table nodes over a local heap, whose node capacity
/// is fixed file-wide by the superblock's K values; a 1.8 group is a list of link messages. The
/// price is a Jenkins lookup3 checksum on the superblock and every object header. Link and
/// attribute creation order are tracked, so members and attributes read back in the order they
/// were added — netCDF-C iterates attributes by creation order. Not written: dense (fractal heap)
/// storage, so one attribute is limited to 64 KiB; soft and external links; filters other than
/// deflate; datasets above 4 GiB when chunked.</para>
/// </remarks>
public sealed class Hdf5Writer
{
    /// <summary>Starts an empty file.</summary>
    /// <param name="userBlockSize">Bytes reserved before the HDF5 data (0, or a power of two from 512 up); MATLAB v7.3 files use 512.</param>
    public Hdf5Writer(int userBlockSize = 0)
    {
        if (userBlockSize != 0 && (userBlockSize < 512 || (userBlockSize & (userBlockSize - 1)) != 0))
            throw new ArgumentException("The user block size must be 0 or a power of two of at least 512.", nameof(userBlockSize));
        UserBlockSize = userBlockSize;
        Root = new Hdf5WriterGroup(this, null, "");
    }

    /// <summary>Size of the user block in bytes.</summary>
    public int UserBlockSize { get; }

    /// <summary>Content of the user block (zero-padded to <see cref="UserBlockSize"/>); null for zeros.</summary>
    public byte[]? UserBlock { get; set; }

    /// <summary>The root group <c>/</c>.</summary>
    public Hdf5WriterGroup Root { get; }

    /// <summary>Writes the file.</summary>
    /// <param name="path">Destination path (overwritten).</param>
    public void Save(string path)
    {
        using var f = File.Create(path);
        Save(f);
    }

    /// <summary>Writes the file to a stream; only sequential writes are made, so it need not be seekable.</summary>
    /// <param name="stream">Destination stream.</param>
    public void Save(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (UserBlock is { } ub && ub.Length > UserBlockSize)
            throw new InvalidOperationException($"The user block content ({ub.Length} bytes) is larger than the user block ({UserBlockSize} bytes).");
        H5WSerializer.Write(this, stream);
    }
}

/// <summary>A group or dataset of an <see cref="Hdf5Writer"/>; it carries attributes and can be the target of an object reference.</summary>
public abstract class Hdf5WriterNode
{
    internal readonly List<KeyValuePair<string, H5WData>> Attrs = new();
    internal long Address = -1;
    internal int HeaderSize;

    private protected Hdf5WriterNode(Hdf5Writer file, Hdf5WriterGroup? parent, string name)
    {
        File = file;
        Parent = parent;
        Name = name;
    }

    /// <summary>The file being built.</summary>
    public Hdf5Writer File { get; }

    /// <summary>The containing group (null for the root).</summary>
    public Hdf5WriterGroup? Parent { get; }

    /// <summary>Link name ("" for the root).</summary>
    public string Name { get; }

    /// <summary>Absolute path.</summary>
    public string Path => Parent is null ? "/" : Parent.Parent is null ? "/" + Name : Parent.Path + "/" + Name;

    /// <summary>Attribute names in creation order.</summary>
    public IReadOnlyList<string> AttributeNames => Attrs.Select(a => a.Key).ToArray();

    /// <summary>
    /// Sets (or replaces) an attribute. A .NET scalar (<c>double</c>, <c>int</c>, <c>string</c>,
    /// <c>bool</c>, <see cref="Complex"/>, a node for an object reference, ...) gives a scalar
    /// attribute; an array gives an array attribute of its shape.
    /// </summary>
    /// <param name="name">Attribute name.</param>
    /// <param name="value">The value.</param>
    /// <param name="fixedLengthStrings">Store strings as fixed-length (NUL-terminated, sized to the longest) rather than variable-length UTF-8.</param>
    public void SetAttribute(string name, object value, bool fixedLengthStrings = false)
        => SetAttribute(name, H5WData.FromObject(value, null, fixedLengthStrings));

    internal void SetAttribute(string name, H5WData data)
    {
        if (string.IsNullOrEmpty(name)) throw new ArgumentException("Attribute names must not be empty.", nameof(name));
        if (data.TooLargeForAttribute)
            throw new ArgumentException($"Attribute '{name}' holds {data.ByteCount} bytes; attributes live in the object header here and are limited to 64 KiB, so store this as a dataset.", nameof(data));
        for (var i = 0; i < Attrs.Count; i++)
            if (Attrs[i].Key == name)
            {
                Attrs[i] = new(name, data);
                return;
            }
        Attrs.Add(new(name, data));
    }

    /// <inheritdoc />
    public override string ToString() => $"{GetType().Name} {Path}";
}

/// <summary>A group being written.</summary>
public sealed class Hdf5WriterGroup : Hdf5WriterNode
{
    internal readonly List<Hdf5WriterNode> Children = new();

    internal Hdf5WriterGroup(Hdf5Writer file, Hdf5WriterGroup? parent, string name) : base(file, parent, name) { }

    /// <summary>Members in creation order.</summary>
    public IReadOnlyList<Hdf5WriterNode> Members => Children;

    /// <summary>The member called <paramref name="name"/>, or null.</summary>
    /// <param name="name">Link name.</param>
    public Hdf5WriterNode? this[string name] => Children.FirstOrDefault(c => c.Name == name);

    /// <summary>Adds a sub-group.</summary>
    /// <param name="name">Link name (no '/').</param>
    public Hdf5WriterGroup CreateGroup(string name)
    {
        CheckName(name);
        var g = new Hdf5WriterGroup(File, this, name);
        Children.Add(g);
        return g;
    }

    /// <summary>
    /// Adds a dataset. <paramref name="data"/> is an array (its shape is kept), an
    /// <see cref="NdArray"/>/<see cref="ComplexNdArray"/>, or a scalar (a scalar dataspace).
    /// </summary>
    /// <param name="name">Link name (no '/').</param>
    /// <param name="data">The data.</param>
    /// <param name="shape">Reshapes the data (same element count); empty for a scalar.</param>
    /// <param name="compressionLevel">0 for none, 1..9 for deflate (the dataset becomes chunked).</param>
    /// <param name="maxShape">Maximum dimensions, -1 for unlimited (the dataset becomes chunked); null for fixed.</param>
    /// <param name="fixedLengthStrings">Store strings as fixed-length rather than variable-length UTF-8.</param>
    public Hdf5WriterDataset CreateDataset(string name, object data, int[]? shape = null, int compressionLevel = 0, long[]? maxShape = null, bool fixedLengthStrings = false)
        => CreateDataset(name, H5WData.FromObject(data, shape, fixedLengthStrings), compressionLevel, maxShape);

    internal Hdf5WriterDataset CreateDataset(string name, H5WData data, int compressionLevel = 0, long[]? maxShape = null, bool allocate = true, byte[]? fill = null)
    {
        CheckName(name);
        if (compressionLevel is < 0 or > 9) throw new ArgumentOutOfRangeException(nameof(compressionLevel), "Deflate levels are 0 (none) to 9.");
        if (compressionLevel > 0 && data.IsDeferred) throw new ArgumentException("Reference and variable-length data cannot be compressed by this writer.", nameof(compressionLevel));
        if (maxShape is not null)
        {
            if (maxShape.Length != data.Shape.Length) throw new ArgumentException("maxShape needs one entry per dimension.", nameof(maxShape));
            for (var i = 0; i < maxShape.Length; i++)
                if (maxShape[i] != -1 && maxShape[i] < data.Shape[i]) throw new ArgumentException($"maxShape[{i}] = {maxShape[i]} is below the current extent {data.Shape[i]}.", nameof(maxShape));
            if (maxShape.SequenceEqual(data.Shape)) maxShape = null;
        }
        if (fill is not null && fill.Length != data.Type.Size) throw new ArgumentException("The fill value must be one element.", nameof(fill));
        var d = new Hdf5WriterDataset(File, this, name, data)
        {
            MaxShape = maxShape?.ToArray(),
            Deflate = data.Count == 0 ? 0 : compressionLevel,
            Allocate = allocate,
            Fill = fill,
        };
        Children.Add(d);
        return d;
    }

    private void CheckName(string name)
    {
        if (string.IsNullOrEmpty(name) || name.Contains('/') || name == ".")
            throw new ArgumentException($"'{name}' is not a valid HDF5 link name (it must be non-empty, without '/').", nameof(name));
        if (Children.Any(c => c.Name == name)) throw new ArgumentException($"Group '{Path}' already has a member '{name}'.", nameof(name));
    }
}

/// <summary>A dataset being written.</summary>
public sealed class Hdf5WriterDataset : Hdf5WriterNode
{
    internal Hdf5WriterDataset(Hdf5Writer file, Hdf5WriterGroup parent, string name, H5WData data) : base(file, parent, name) => Data = data;

    internal H5WData Data { get; }
    internal long[]? MaxShape { get; init; }
    internal int Deflate { get; init; }
    internal bool Allocate { get; init; } = true;
    internal byte[]? Fill { get; init; }

    internal bool Chunked => Deflate > 0 || MaxShape is not null;
    internal bool HasStorage => Allocate && Data.Count > 0;

    // Layout, filled in by the serializer.
    internal long DataAddress = -1, StoredSize, BTreeAddress = -1;
    internal byte[]? Compressed;

    /// <summary>Dimensions (empty for a scalar).</summary>
    public IReadOnlyList<long> Shape => Data.Shape;
}

/// <summary>An HDF5 datatype as encoded in a datatype message.</summary>
internal sealed class H5WType
{
    private H5WType(int size, byte[] message, bool deferred = false)
    {
        Size = size;
        Message = message;
        Deferred = deferred;
    }

    public int Size { get; }
    public byte[] Message { get; }

    /// <summary>Element bytes depend on addresses (references, variable-length data).</summary>
    public bool Deferred { get; }

    private static byte[] Head(byte classAndVersion, byte b1, byte b2, byte b3, int size, int extra)
    {
        var m = new byte[8 + extra];
        m[0] = classAndVersion;
        m[1] = b1;
        m[2] = b2;
        m[3] = b3;
        BinaryPrimitives.WriteInt32LittleEndian(m.AsSpan(4), size);
        return m;
    }

    public static H5WType Integer(int size, bool signed)
    {
        var m = Head(0x10, (byte)(signed ? 0x08 : 0), 0, 0, size, 4);
        BinaryPrimitives.WriteUInt16LittleEndian(m.AsSpan(10), (ushort)(8 * size)); // bit offset 0, precision
        return new H5WType(size, m);
    }

    public static H5WType Float(int size)
    {
        var (sign, expLoc, expSize, mantSize, bias) = size switch
        {
            2 => (15, 10, 5, 10, 15),
            4 => (31, 23, 8, 23, 127),
            8 => (63, 52, 11, 52, 1023),
            _ => throw new ArgumentException("Floats are 2, 4 or 8 bytes."),
        };
        // Mantissa normalisation 2 (implied leading 1), little-endian, zero padding.
        var m = Head(0x11, 0x20, (byte)sign, 0, size, 12);
        BinaryPrimitives.WriteUInt16LittleEndian(m.AsSpan(10), (ushort)(8 * size));
        m[12] = (byte)expLoc;
        m[13] = (byte)expSize;
        m[14] = 0;
        m[15] = (byte)mantSize;
        BinaryPrimitives.WriteInt32LittleEndian(m.AsSpan(16), bias);
        return new H5WType(size, m);
    }

    public static readonly H5WType I8 = Integer(1, true), U8 = Integer(1, false), I16 = Integer(2, true), U16 = Integer(2, false),
        I32 = Integer(4, true), U32 = Integer(4, false), I64 = Integer(8, true), U64 = Integer(8, false),
        F16 = Float(2), F32 = Float(4), F64 = Float(8);

    /// <summary>Fixed-length string, NUL-terminated (MATLAB's and netCDF's choice; a string filling the field has no terminator).</summary>
    public static H5WType FixedString(int size, bool utf8) => new(size, Head(0x13, (byte)(utf8 ? 0x10 : 0x00), 0, 0, size, 0));

    /// <summary>Object reference (an object header address).</summary>
    public static readonly H5WType Reference = new(8, Head(0x17, 0, 0, 0, 8, 0), deferred: true);

    /// <summary>Variable-length string (heap descriptor: length, collection address, index), base type uint8 as HDF5 writes it.</summary>
    public static H5WType VlenString(bool utf8) => new(16, [.. Head(0x19, 0x01, (byte)(utf8 ? 1 : 0), 0, 16, 0), .. U8.Message], deferred: true);

    /// <summary>Variable-length sequence of <paramref name="b"/>.</summary>
    public static H5WType VlenSequence(H5WType b) => new(16, [.. Head(0x19, 0, 0, 0, 16, 0), .. b.Message], deferred: true);

    /// <summary>h5py's boolean: an int8 enum FALSE = 0, TRUE = 1 (version 1 encoding, names padded to 8).</summary>
    public static readonly H5WType Bool = new(1, [.. Head(0x18, 2, 0, 0, 1, 0), .. I8.Message, .. "FALSE\0\0\0"u8, .. "TRUE\0\0\0\0"u8, 0, 1]);

    /// <summary>A compound of the members laid end to end (version 1 encoding, as h5py writes complex numbers).</summary>
    public static H5WType Compound(params (string Name, H5WType Type)[] members)
    {
        var body = new List<byte>();
        var offset = 0;
        var deferred = false;
        foreach (var (name, type) in members)
        {
            var nb = Encoding.UTF8.GetBytes(name);
            var padded = (nb.Length + 1 + 7) / 8 * 8;
            var entry = new byte[padded + 32];
            nb.CopyTo(entry, 0);
            BinaryPrimitives.WriteInt32LittleEndian(entry.AsSpan(padded), offset);
            // dimensionality 0, reserved, permutation, reserved, four zero dimension sizes
            body.AddRange(entry);
            body.AddRange(type.Message);
            offset += type.Size;
            deferred |= type.Deferred;
        }
        var head = Head(0x16, (byte)(members.Length & 0xFF), (byte)(members.Length >> 8), 0, offset, 0);
        return new H5WType(offset, [.. head, .. body], deferred);
    }
}

/// <summary>A variable-length element's payload, placed in a global heap collection.</summary>
internal sealed class H5WHeapObject
{
    public byte[]? Bytes;
    public Hdf5WriterNode?[]? Refs;
    public int Elements;
    public long Collection = -1;
    public int Index;

    public int Length => Bytes?.Length ?? 8 * Refs!.Length;
}

/// <summary>Address resolution while encoding; during sizing every address is 0.</summary>
internal sealed class H5WContext
{
    public bool Sizing;
    public HashSet<Hdf5WriterNode>? Members;

    public long AddressOf(Hdf5WriterNode? n)
    {
        if (n is null) return 0;
        if (Members is not null && !Members.Contains(n))
            throw new InvalidOperationException($"A reference points to '{n.Path}', which is not part of this file.");
        return Sizing ? 0 : n.Address;
    }
}

/// <summary>Typed data (dataset contents or an attribute value) ready to be encoded.</summary>
internal sealed class H5WData
{
    private readonly Array? _blittable;
    private readonly byte[]? _bytes;
    private readonly Action<H5WContext, Span<byte>>? _deferred;

    private H5WData(H5WType type, long[] shape, Array? blittable, byte[]? bytes, Action<H5WContext, Span<byte>>? deferred, List<H5WHeapObject>? heap)
    {
        Type = type;
        Shape = shape;
        long n = 1;
        foreach (var d in shape) n = checked(n * d);
        Count = n;
        _blittable = blittable;
        _bytes = bytes;
        _deferred = deferred;
        Heap = heap ?? [];
    }

    public H5WType Type { get; }

    /// <summary>Dimensions; empty for a scalar.</summary>
    public long[] Shape { get; }

    public long Count { get; }

    public List<H5WHeapObject> Heap { get; }

    public bool IsDeferred => _deferred is not null;

    public long ByteCount => checked(Count * Type.Size);

    /// <summary>True when this cannot be an attribute (more than an object header message can hold).</summary>
    public bool TooLargeForAttribute => ByteCount > 60000;

    /// <summary>Raw bytes of a non-deferred value (the source array itself, no copy).</summary>
    public ReadOnlySpan<byte> RawSpan
    {
        get
        {
            if (_bytes is not null) return _bytes;
            if (_blittable is not null)
                return MemoryMarshal.CreateReadOnlySpan(ref MemoryMarshal.GetArrayDataReference(_blittable), checked((int)ByteCount));
            throw new InvalidOperationException("Deferred data has no raw form.");
        }
    }

    public byte[] Encode(H5WContext ctx)
    {
        if (_deferred is null) return RawSpan.ToArray();
        var b = new byte[ByteCount];
        _deferred(ctx, b);
        return b;
    }

    public void WriteTo(Stream s, H5WContext ctx)
    {
        if (_deferred is null) s.Write(RawSpan);
        else s.Write(Encode(ctx));
    }

    private static long[] ShapeOrScalar(int[]? shape, long count, long[] natural)
    {
        if (shape is null) return natural;
        long n = 1;
        foreach (var d in shape)
        {
            if (d < 0) throw new ArgumentException("Dimensions must be non-negative.", nameof(shape));
            n *= d;
        }
        if (n != count) throw new ArgumentException($"Shape ({string.Join(", ", shape)}) holds {n} elements; the data has {count}.", nameof(shape));
        return shape.Select(x => (long)x).ToArray();
    }

    // ---------------------------------------------------------------- factories

    public static H5WData Numeric(Array values, H5WType type, long[] shape)
    {
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("The HDF5 writer writes little-endian data and needs a little-endian machine.");
        return new H5WData(type, shape, values, null, null, null);
    }

    public static H5WData Raw(byte[] bytes, H5WType type, long[] shape) => new(type, shape, null, bytes, null, null);

    public static H5WData Scalar<T>(T value) where T : unmanaged => Numeric(new[] { value }, TypeOf(typeof(T)) ?? throw new ArgumentException($"{typeof(T)} is not numeric."), []);

    /// <summary>One fixed-length NUL-terminated string as a scalar (MATLAB_class, netCDF text attributes).</summary>
    public static H5WData FixedString(string s)
    {
        var b = Encoding.UTF8.GetBytes(s);
        var utf8 = b.Length != s.Length;
        if (b.Length == 0) b = [0];
        return Raw(b, H5WType.FixedString(b.Length, utf8), []);
    }

    public static H5WData FixedStrings(IReadOnlyList<string> strings, long[] shape)
    {
        var enc = strings.Select(s => Encoding.UTF8.GetBytes(s ?? "")).ToArray();
        var size = Math.Max(1, enc.Length == 0 ? 1 : enc.Max(e => e.Length));
        var utf8 = strings.Any(s => s is not null && Encoding.UTF8.GetByteCount(s) != s.Length);
        var b = new byte[(long)size * enc.Length];
        for (var i = 0; i < enc.Length; i++) enc[i].CopyTo(b, (long)i * size);
        return Raw(b, H5WType.FixedString(size, utf8), shape);
    }

    public static H5WData VlenStrings(IReadOnlyList<string?> strings, long[] shape)
    {
        var heap = new List<H5WHeapObject>();
        var objs = new H5WHeapObject?[strings.Count];
        for (var i = 0; i < strings.Count; i++)
        {
            if (string.IsNullOrEmpty(strings[i])) continue; // a zero descriptor, as HDF5 writes empty strings
            var b = Encoding.UTF8.GetBytes(strings[i]!);
            objs[i] = new H5WHeapObject { Bytes = b, Elements = b.Length };
            heap.Add(objs[i]!);
        }
        return new H5WData(H5WType.VlenString(utf8: true), shape, null, null, (_, dst) => Descriptors(objs, dst), heap);
    }

    /// <summary>A variable-length sequence per element, of fixed-size base elements given as bytes.</summary>
    public static H5WData VlenSequences(H5WType baseType, IReadOnlyList<byte[]> elements, long[] shape)
    {
        var heap = new List<H5WHeapObject>();
        var objs = new H5WHeapObject?[elements.Count];
        for (var i = 0; i < elements.Count; i++)
        {
            if (elements[i].Length == 0) continue;
            objs[i] = new H5WHeapObject { Bytes = elements[i], Elements = elements[i].Length / baseType.Size };
            heap.Add(objs[i]!);
        }
        return new H5WData(H5WType.VlenSequence(baseType), shape, null, null, (_, dst) => Descriptors(objs, dst), heap);
    }

    /// <summary>A variable-length sequence of object references per element (HDF5's DIMENSION_LIST).</summary>
    public static H5WData VlenReferences(IReadOnlyList<Hdf5WriterNode[]> elements, long[] shape)
    {
        var heap = new List<H5WHeapObject>();
        var objs = new H5WHeapObject?[elements.Count];
        for (var i = 0; i < elements.Count; i++)
        {
            if (elements[i].Length == 0) continue;
            objs[i] = new H5WHeapObject { Refs = elements[i], Elements = elements[i].Length };
            heap.Add(objs[i]!);
        }
        return new H5WData(H5WType.VlenSequence(H5WType.Reference), shape, null, null, (_, dst) => Descriptors(objs, dst), heap);
    }

    private static void Descriptors(H5WHeapObject?[] objs, Span<byte> dst)
    {
        for (var i = 0; i < objs.Length; i++)
        {
            var o = objs[i];
            if (o is null) continue; // zero length, null collection
            var d = dst.Slice(16 * i, 16);
            BinaryPrimitives.WriteInt32LittleEndian(d, o.Elements);
            BinaryPrimitives.WriteInt64LittleEndian(d[4..], o.Collection);
            BinaryPrimitives.WriteInt32LittleEndian(d[12..], o.Index);
        }
    }

    public static H5WData References(IReadOnlyList<Hdf5WriterNode?> nodes, long[] shape)
    {
        var copy = nodes.ToArray();
        return new H5WData(H5WType.Reference, shape, null, null, (ctx, dst) =>
        {
            for (var i = 0; i < copy.Length; i++) BinaryPrimitives.WriteInt64LittleEndian(dst[(8 * i)..], ctx.AddressOf(copy[i]));
        }, null);
    }

    /// <summary>Records of a compound type whose bytes depend on addresses (e.g. REFERENCE_LIST).</summary>
    public static H5WData DeferredRecords(H5WType type, long count, Action<H5WContext, Span<byte>> encode)
        => new(type, [count], null, null, encode, null);

    internal static H5WType? TypeOf(Type t)
    {
        if (t == typeof(double)) return H5WType.F64;
        if (t == typeof(float)) return H5WType.F32;
        if (t == typeof(Half)) return H5WType.F16;
        if (t == typeof(sbyte)) return H5WType.I8;
        if (t == typeof(byte)) return H5WType.U8;
        if (t == typeof(short)) return H5WType.I16;
        if (t == typeof(ushort)) return H5WType.U16;
        if (t == typeof(int)) return H5WType.I32;
        if (t == typeof(uint)) return H5WType.U32;
        if (t == typeof(long)) return H5WType.I64;
        if (t == typeof(ulong)) return H5WType.U64;
        return null;
    }

    private static readonly H5WType ComplexType = H5WType.Compound(("r", H5WType.F64), ("i", H5WType.F64));

    /// <summary>Converts a .NET value (see <see cref="Hdf5Writer"/>) into typed data.</summary>
    public static H5WData FromObject(object value, int[]? shape, bool fixedLengthStrings)
    {
        ArgumentNullException.ThrowIfNull(value);
        switch (value)
        {
            case NdArray a:
                return Numeric(a.ToArray(), H5WType.F64, ShapeOrScalar(shape, a.Size, a.Shape.ToArray().Select(x => (long)x).ToArray()));
            case ComplexNdArray c:
            {
                var z = c.ToArray();
                return FromArray(z, shape, fixedLengthStrings, c.Shape.Select(x => (long)x).ToArray());
            }
            case Array arr:
            {
                var natural = new long[arr.Rank];
                for (var i = 0; i < arr.Rank; i++) natural[i] = arr.GetLength(i);
                return FromArray(arr, shape, fixedLengthStrings, natural);
            }
            default:
            {
                // A scalar: wrap it in a one-element array of its own type.
                var one = System.Array.CreateInstance(value.GetType(), 1);
                one.SetValue(value, 0);
                return FromArray(one, shape, fixedLengthStrings, []);
            }
        }
    }

    private static H5WData FromArray(Array arr, int[]? shape, bool fixedStrings, long[] natural)
    {
        var et = arr.GetType().GetElementType()!;
        var count = arr.LongLength;
        var sh = ShapeOrScalar(shape, count, natural);
        if (TypeOf(et) is { } t)
        {
            // Multi-dimensional arrays are row-major and contiguous, so the bytes can be used as they are.
            return Numeric(arr, t, sh);
        }
        if (et == typeof(bool))
        {
            var b = new byte[count];
            var k = 0;
            foreach (bool x in arr) b[k++] = x ? (byte)1 : (byte)0;
            return Raw(b, H5WType.Bool, sh);
        }
        if (et == typeof(Complex))
        {
            var d = new double[2 * count];
            var k = 0;
            foreach (Complex z in arr)
            {
                d[k++] = z.Real;
                d[k++] = z.Imaginary;
            }
            return Numeric(d, ComplexType, sh);
        }
        if (et == typeof(string))
        {
            var s = arr.Cast<string?>().ToArray();
            return fixedStrings ? FixedStrings(s!, sh) : VlenStrings(s, sh);
        }
        if (typeof(Hdf5WriterNode).IsAssignableFrom(et))
            return References(arr.Cast<Hdf5WriterNode?>().ToArray(), sh);
        throw new ArgumentException($"Cannot write {et.Name} data to HDF5; supported element types are the .NET numeric types, bool, Complex, string and Hdf5WriterNode (object references).");
    }
}

/// <summary>Jenkins' lookup3 <c>hashlittle</c>, which HDF5 uses for its metadata checksums (initial value 0).</summary>
internal static class H5Checksum
{
    public static uint Lookup3(ReadOnlySpan<byte> k, uint initval = 0)
    {
        uint a, b, c;
        a = b = c = 0xdeadbeef + (uint)k.Length + initval;
        while (k.Length > 12)
        {
            a += BinaryPrimitives.ReadUInt32LittleEndian(k);
            b += BinaryPrimitives.ReadUInt32LittleEndian(k[4..]);
            c += BinaryPrimitives.ReadUInt32LittleEndian(k[8..]);
            a -= c; a ^= BitOperations.RotateLeft(c, 4); c += b;
            b -= a; b ^= BitOperations.RotateLeft(a, 6); a += c;
            c -= b; c ^= BitOperations.RotateLeft(b, 8); b += a;
            a -= c; a ^= BitOperations.RotateLeft(c, 16); c += b;
            b -= a; b ^= BitOperations.RotateLeft(a, 19); a += c;
            c -= b; c ^= BitOperations.RotateLeft(b, 4); b += a;
            k = k[12..];
        }
        if (k.Length == 0) return c;
        Span<byte> tail = stackalloc byte[12];
        tail.Clear();
        k.CopyTo(tail);
        // The byte-wise tail adds zero for missing bytes, so a zero-padded word read is the same.
        if (k.Length > 8) c += BinaryPrimitives.ReadUInt32LittleEndian(tail[8..]);
        if (k.Length > 4) b += BinaryPrimitives.ReadUInt32LittleEndian(tail[4..]);
        a += BinaryPrimitives.ReadUInt32LittleEndian(tail);
        c ^= b; c -= BitOperations.RotateLeft(b, 14);
        a ^= c; a -= BitOperations.RotateLeft(c, 11);
        b ^= a; b -= BitOperations.RotateLeft(a, 25);
        c ^= b; c -= BitOperations.RotateLeft(b, 16);
        a ^= c; a -= BitOperations.RotateLeft(c, 4);
        b ^= a; b -= BitOperations.RotateLeft(a, 14);
        c ^= b; c -= BitOperations.RotateLeft(b, 24);
        return c;
    }
}

/// <summary>Lays the file out (every size is known before any address is) and writes it front to back.</summary>
internal static class H5WSerializer
{
    private const int SuperblockSize = 48;
    private const int ChunkBTreeK = 32; // HDF5's default chunk B-tree K: a v1 B-tree node is always read at full size
    private const long Undefined = -1;

    private sealed class Collection
    {
        public long Address;
        public long Size;
        public readonly List<H5WHeapObject> Objects = new();
    }

    public static void Write(Hdf5Writer file, Stream s)
    {
        var nodes = new List<Hdf5WriterNode>();
        Collect(file.Root, nodes);
        var members = new HashSet<Hdf5WriterNode>(nodes);
        var sizing = new H5WContext { Sizing = true, Members = members };
        var real = new H5WContext { Members = members };

        // Global heap: every variable-length payload, packed into collections of at most 65535 objects.
        var collections = new List<Collection>();
        foreach (var n in nodes)
        {
            foreach (var (_, a) in n.Attrs) Place(a.Heap, collections);
            if (n is Hdf5WriterDataset d) Place(d.Data.Heap, collections);
        }
        foreach (var c in collections)
        {
            long used = 16;
            foreach (var o in c.Objects) used += 16 + Pad8(o.Length);
            c.Size = Math.Max(4096, used); // HDF5 reads a collection's first 4 KiB before its size
        }

        foreach (var n in nodes) n.HeaderSize = Header(n, sizing).Length;

        var datasets = nodes.OfType<Hdf5WriterDataset>().ToList();
        foreach (var d in datasets)
        {
            if (!d.HasStorage) continue;
            if (d.ByteCountOk() is { } err) throw new NotSupportedException(err);
            if (d.Deflate > 0)
            {
                var z = new MemoryStream();
                using (var zs = new ZLibStream(z, Level(d.Deflate), leaveOpen: true)) zs.Write(d.Data.RawSpan);
                d.Compressed = z.ToArray();
                d.StoredSize = d.Compressed.Length;
            }
            else d.StoredSize = d.Data.ByteCount;
        }

        // Addresses, relative to the superblock.
        long pos = SuperblockSize;
        foreach (var n in nodes)
        {
            n.Address = pos;
            pos += n.HeaderSize;
        }
        foreach (var d in datasets)
            if (d.Chunked && d.HasStorage)
            {
                d.BTreeAddress = pos;
                pos += BTreeNodeSize(d.Data.Shape.Length);
            }
        foreach (var c in collections)
        {
            c.Address = pos;
            pos += c.Size;
        }
        for (var ci = 0; ci < collections.Count; ci++)
            for (var i = 0; i < collections[ci].Objects.Count; i++)
            {
                collections[ci].Objects[i].Collection = collections[ci].Address;
                collections[ci].Objects[i].Index = i + 1;
            }
        foreach (var d in datasets)
            if (d.HasStorage)
            {
                pos = Pad8(pos);
                d.DataAddress = pos;
                pos += d.StoredSize;
            }
        var end = pos;

        // Writing, in address order.
        var w = new PositionedWriter(s);
        if (file.UserBlockSize > 0)
        {
            var ub = new byte[file.UserBlockSize];
            file.UserBlock?.CopyTo(ub, 0);
            w.Write(ub);
        }
        w.Write(Superblock(file.UserBlockSize, file.UserBlockSize + end, file.Root.Address));
        foreach (var n in nodes)
        {
            var h = Header(n, real);
            if (h.Length != n.HeaderSize) throw new InvalidOperationException($"Internal error: the header of '{n.Path}' changed size between layout and writing.");
            w.Write(h);
        }
        foreach (var d in datasets)
            if (d.Chunked && d.HasStorage) w.Write(BTreeNode(d));
        foreach (var c in collections) w.Write(CollectionBytes(c, real));
        foreach (var d in datasets)
        {
            if (!d.HasStorage) continue;
            w.PadTo(file.UserBlockSize + d.DataAddress);
            if (d.Compressed is not null) w.Write(d.Compressed);
            else w.WriteData(d.Data, real);
        }
        if (w.Position != file.UserBlockSize + end) throw new InvalidOperationException("Internal error: the file length disagrees with its layout.");
    }

    private static string? ByteCountOk(this Hdf5WriterDataset d)
        => d.Chunked && d.Data.ByteCount >= uint.MaxValue
            ? $"Dataset '{d.Path}' is {d.Data.ByteCount} bytes; chunked (compressed or extendible) datasets are written as one chunk and are limited to 4 GiB."
            : null;

    private static CompressionLevel Level(int deflate) => deflate switch
    {
        <= 3 => CompressionLevel.Fastest,
        >= 8 => CompressionLevel.SmallestSize,
        _ => CompressionLevel.Optimal,
    };

    private static void Collect(Hdf5WriterNode n, List<Hdf5WriterNode> into)
    {
        into.Add(n);
        if (n is Hdf5WriterGroup g)
            foreach (var c in g.Children) Collect(c, into);
    }

    private static void Place(List<H5WHeapObject> objs, List<Collection> collections)
    {
        foreach (var o in objs)
        {
            if (collections.Count == 0 || collections[^1].Objects.Count == 65535) collections.Add(new Collection());
            collections[^1].Objects.Add(o);
        }
    }

    internal static long Pad8(long n) => (n + 7) & ~7L;

    private sealed class PositionedWriter(Stream s)
    {
        public long Position { get; private set; }

        public void Write(ReadOnlySpan<byte> b)
        {
            s.Write(b);
            Position += b.Length;
        }

        public void WriteData(H5WData data, H5WContext ctx)
        {
            data.WriteTo(s, ctx);
            Position += data.ByteCount;
        }

        public void PadTo(long target)
        {
            if (target < Position) throw new InvalidOperationException("Internal error: layout overlap.");
            if (target > Position) Write(new byte[target - Position]);
        }
    }

    // ---------------------------------------------------------------- superblock

    private static byte[] Superblock(long baseAddress, long endOfFile, long root)
    {
        var b = new byte[SuperblockSize];
        // Not a u8 literal: "\x89" there is U+0089, which UTF-8 encodes as two bytes.
        ReadOnlySpan<byte> signature = [0x89, (byte)'H', (byte)'D', (byte)'F', 0x0D, 0x0A, 0x1A, 0x0A];
        signature.CopyTo(b);
        b[8] = 2;  // superblock version
        b[9] = 8;  // size of offsets
        b[10] = 8; // size of lengths
        b[11] = 0; // file consistency flags
        BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(12), baseAddress);
        BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(20), Undefined); // no superblock extension
        // The end-of-file address is the absolute file length (HDF5 compares it with the file size),
        // unlike every other address, which is relative to the base address (the user block size).
        BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(28), endOfFile);
        BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(36), root);
        BinaryPrimitives.WriteUInt32LittleEndian(b.AsSpan(44), H5Checksum.Lookup3(b.AsSpan(0, 44)));
        return b;
    }

    // ---------------------------------------------------------------- object headers

    private sealed class Messages
    {
        public readonly List<(byte Type, byte Flags, ushort Order, byte[] Body)> List = new();

        public void Add(int type, byte[] body, byte flags = 0, int order = 0)
        {
            if (body.Length > ushort.MaxValue) throw new NotSupportedException($"An HDF5 header message of {body.Length} bytes exceeds the 64 KiB limit of compact storage.");
            List.Add(((byte)type, flags, (ushort)order, body));
        }
    }

    private static byte[] Header(Hdf5WriterNode n, H5WContext ctx)
    {
        var m = new Messages();
        if (n is Hdf5WriterGroup g)
        {
            m.Add(H5Msg.LinkInfo, LinkInfo(g.Children.Count));
            m.Add(H5Msg.GroupInfo, [0, 0]);
            for (var i = 0; i < g.Children.Count; i++) m.Add(H5Msg.Link, Link(g.Children[i].Name, i, ctx.AddressOf(g.Children[i])));
        }
        else
        {
            var d = (Hdf5WriterDataset)n;
            m.Add(H5Msg.Dataspace, Dataspace(d.Data.Shape, d.MaxShape));
            m.Add(H5Msg.Datatype, d.Data.Type.Message, flags: 1);
            m.Add(H5Msg.FillValue, FillValue(d), flags: 1);
            m.Add(H5Msg.Layout, Layout(d));
            if (d.Deflate > 0) m.Add(H5Msg.FilterPipeline, Deflate(d.Deflate));
        }
        m.Add(H5Msg.AttributeInfo, AttributeInfo(n.Attrs.Count));
        for (var i = 0; i < n.Attrs.Count; i++)
            m.Add(H5Msg.Attribute, Attribute(n.Attrs[i].Key, n.Attrs[i].Value, ctx, n), order: i);

        var nattrs = n.Attrs.Count;
        var phase = nattrs > 8; // HDF5's default compact limit is 8; record a larger one rather than look inconsistent
        var bodySize = m.List.Sum(x => 6 + x.Body.Length);
        var size = 4 + 1 + 1 + (phase ? 4 : 0) + 4 + bodySize + 4;
        var b = new byte[size];
        "OHDR"u8.CopyTo(b);
        b[4] = 2;
        // Chunk #0 size in 4 bytes; attribute creation order tracked; phase change values stored when needed.
        b[5] = (byte)(0x02 | 0x04 | (phase ? 0x10 : 0));
        var p = 6;
        if (phase)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(b.AsSpan(p), (ushort)Math.Min(nattrs, ushort.MaxValue));
            BinaryPrimitives.WriteUInt16LittleEndian(b.AsSpan(p + 2), 6);
            p += 4;
        }
        BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(p), bodySize);
        p += 4;
        foreach (var (type, flags, order, body) in m.List)
        {
            b[p] = type;
            BinaryPrimitives.WriteUInt16LittleEndian(b.AsSpan(p + 1), (ushort)body.Length);
            b[p + 3] = flags;
            BinaryPrimitives.WriteUInt16LittleEndian(b.AsSpan(p + 4), order);
            body.CopyTo(b, p + 6);
            p += 6 + body.Length;
        }
        BinaryPrimitives.WriteUInt32LittleEndian(b.AsSpan(p), H5Checksum.Lookup3(b.AsSpan(0, p)));
        return b;
    }

    private static byte[] LinkInfo(int links)
    {
        // Version 0, creation order tracked (not indexed), max creation index, no dense storage.
        var b = new byte[2 + 8 + 8 + 8];
        b[1] = 0x01;
        BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(2), links);
        BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(10), Undefined);
        BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(18), Undefined);
        return b;
    }

    private static byte[] AttributeInfo(int attrs)
    {
        var b = new byte[2 + 2 + 8 + 8];
        b[1] = 0x01;
        BinaryPrimitives.WriteUInt16LittleEndian(b.AsSpan(2), (ushort)Math.Min(attrs, ushort.MaxValue));
        BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(4), Undefined);
        BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(12), Undefined);
        return b;
    }

    private static byte[] Link(string name, int order, long address)
    {
        var nb = Encoding.UTF8.GetBytes(name);
        var utf8 = nb.Length != name.Length;
        var lenSize = nb.Length < 256 ? 1 : nb.Length < 65536 ? 2 : 4;
        var b = new List<byte>
        {
            1,
            (byte)((lenSize == 1 ? 0 : lenSize == 2 ? 1 : 2) | 0x04 | (utf8 ? 0x10 : 0)),
        };
        Span<byte> tmp = stackalloc byte[8];
        BinaryPrimitives.WriteInt64LittleEndian(tmp, order);
        b.AddRange(tmp.ToArray());
        if (utf8) b.Add(1);
        BinaryPrimitives.WriteInt32LittleEndian(tmp, nb.Length);
        b.AddRange(tmp[..lenSize].ToArray());
        b.AddRange(nb);
        BinaryPrimitives.WriteInt64LittleEndian(tmp, address);
        b.AddRange(tmp.ToArray());
        return b.ToArray();
    }

    private static byte[] Dataspace(long[] dims, long[]? max)
    {
        var rank = dims.Length;
        var b = new byte[4 + 8 * rank * (max is null ? 1 : 2)];
        b[0] = 2;
        b[1] = (byte)rank;
        b[2] = (byte)(max is null ? 0 : 1);
        b[3] = (byte)(rank == 0 ? 0 : 1); // scalar or simple
        for (var i = 0; i < rank; i++) BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(4 + 8 * i), dims[i]);
        if (max is not null)
            for (var i = 0; i < rank; i++) BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(4 + 8 * (rank + i)), max[i] < 0 ? Undefined : max[i]);
        return b;
    }

    private static byte[] FillValue(Hdf5WriterDataset d)
    {
        // Version 3: allocation time late (contiguous) or incremental (chunked), fill written "if set".
        var flags = (byte)((d.Chunked ? 3 : 2) | (2 << 2));
        if (d.Fill is null) return [3, flags];
        var b = new byte[2 + 4 + d.Fill.Length];
        b[0] = 3;
        b[1] = (byte)(flags | 0x20);
        BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(2), d.Fill.Length);
        d.Fill.CopyTo(b, 6);
        return b;
    }

    private static int[] ChunkDims(Hdf5WriterDataset d) => d.Data.Shape.Select(x => (int)Math.Max(1, x)).ToArray();

    private static byte[] Layout(Hdf5WriterDataset d)
    {
        if (!d.Chunked)
        {
            var b = new byte[2 + 8 + 8];
            b[0] = 3;
            b[1] = 1; // contiguous
            BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(2), d.HasStorage ? d.DataAddress : Undefined);
            BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(10), d.Data.ByteCount);
            return b;
        }
        var rank = d.Data.Shape.Length;
        if (rank == 0) throw new NotSupportedException($"Scalar dataset '{d.Path}' cannot be chunked.");
        foreach (var x in d.Data.Shape)
            if (x > uint.MaxValue) throw new NotSupportedException($"Dataset '{d.Path}' has a dimension too long for one chunk.");
        var c = new byte[3 + 8 + 4 * (rank + 1)];
        c[0] = 3;
        c[1] = 2; // chunked, v1 B-tree index
        c[2] = (byte)(rank + 1);
        BinaryPrimitives.WriteInt64LittleEndian(c.AsSpan(3), d.HasStorage ? d.BTreeAddress : Undefined);
        var dims = ChunkDims(d);
        for (var i = 0; i < rank; i++) BinaryPrimitives.WriteInt32LittleEndian(c.AsSpan(11 + 4 * i), dims[i]);
        BinaryPrimitives.WriteInt32LittleEndian(c.AsSpan(11 + 4 * rank), d.Data.Type.Size);
        return c;
    }

    private static byte[] Deflate(int level)
    {
        // Filter pipeline version 2: one filter, id 1 (deflate), optional (as H5Pset_deflate marks it), one parameter.
        var b = new byte[2 + 6 + 4];
        b[0] = 2;
        b[1] = 1;
        BinaryPrimitives.WriteUInt16LittleEndian(b.AsSpan(2), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(b.AsSpan(4), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(b.AsSpan(6), 1);
        BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(8), level);
        return b;
    }

    private static byte[] Attribute(string name, H5WData data, H5WContext ctx, Hdf5WriterNode owner)
    {
        var nb = Encoding.UTF8.GetBytes(name);
        var utf8 = nb.Length != name.Length;
        var type = data.Type.Message;
        var space = Dataspace(data.Shape, null);
        var value = data.Encode(ctx);
        var size = 9 + nb.Length + 1 + type.Length + space.Length + value.Length;
        if (size > ushort.MaxValue)
            throw new NotSupportedException($"Attribute '{name}' of '{owner.Path}' needs {size} bytes; this writer stores attributes in the object header, which holds at most 64 KiB per attribute.");
        var b = new byte[size];
        b[0] = 3;
        b[1] = 0;
        BinaryPrimitives.WriteUInt16LittleEndian(b.AsSpan(2), (ushort)(nb.Length + 1));
        BinaryPrimitives.WriteUInt16LittleEndian(b.AsSpan(4), (ushort)type.Length);
        BinaryPrimitives.WriteUInt16LittleEndian(b.AsSpan(6), (ushort)space.Length);
        b[8] = (byte)(utf8 ? 1 : 0);
        var p = 9;
        nb.CopyTo(b, p);
        p += nb.Length + 1;
        type.CopyTo(b, p);
        p += type.Length;
        space.CopyTo(b, p);
        p += space.Length;
        value.CopyTo(b, p);
        return b;
    }

    // ---------------------------------------------------------------- chunk index and heap

    private static int KeySize(int rank) => 8 + 8 * (rank + 1);

    private static long BTreeNodeSize(int rank) => 24 + 2 * ChunkBTreeK * 8 + (2 * ChunkBTreeK + 1) * KeySize(rank);

    /// <summary>A v1 B-tree leaf (node type 1) with one chunk: left key at the origin, right key one chunk further.</summary>
    private static byte[] BTreeNode(Hdf5WriterDataset d)
    {
        var rank = d.Data.Shape.Length;
        var b = new byte[BTreeNodeSize(rank)];
        "TREE"u8.CopyTo(b);
        b[4] = 1; // raw data chunks
        b[5] = 0; // leaf
        BinaryPrimitives.WriteUInt16LittleEndian(b.AsSpan(6), 1);
        BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(8), Undefined);
        BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(16), Undefined);
        var p = 24;
        BinaryPrimitives.WriteUInt32LittleEndian(b.AsSpan(p), (uint)d.StoredSize); // chunk size; filter mask 0; offsets 0
        p += KeySize(rank);
        BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(p), d.DataAddress);
        p += 8;
        var dims = ChunkDims(d);
        for (var i = 0; i < rank; i++) BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(p + 8 + 8 * i), dims[i]);
        return b;
    }

    private static byte[] CollectionBytes(Collection c, H5WContext ctx)
    {
        var b = new byte[c.Size];
        "GCOL"u8.CopyTo(b);
        b[4] = 1;
        BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(8), c.Size);
        long p = 16;
        for (var i = 0; i < c.Objects.Count; i++)
        {
            var o = c.Objects[i];
            BinaryPrimitives.WriteUInt16LittleEndian(b.AsSpan((int)p), (ushort)(i + 1));
            BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan((int)p + 8), o.Length);
            var body = b.AsSpan((int)p + 16, o.Length);
            if (o.Bytes is not null) o.Bytes.CopyTo(body);
            else
                for (var r = 0; r < o.Refs!.Length; r++) BinaryPrimitives.WriteInt64LittleEndian(body[(8 * r)..], ctx.AddressOf(o.Refs[r]));
            p += 16 + Pad8(o.Length);
        }
        if (c.Size - p >= 16)
        {
            // Free space: object 0, whose size counts its own header.
            BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan((int)p + 8), c.Size - p);
        }
        return b;
    }
}
