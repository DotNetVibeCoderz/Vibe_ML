using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.IO;

/// <summary>Storage layout of an HDF5 dataset.</summary>
public enum Hdf5Layout
{
    /// <summary>Data held inside the object header (small datasets).</summary>
    Compact,
    /// <summary>One contiguous block in the file.</summary>
    Contiguous,
    /// <summary>Split into equally sized chunks, possibly compressed.</summary>
    Chunked,
}

/// <summary>An HDF5 object reference (the address of the referenced object's header).</summary>
/// <param name="Address">Object header address relative to the HDF5 base address; -1 for a null reference.</param>
public readonly record struct Hdf5Reference(long Address)
{
    /// <summary>True for a null (unset) reference.</summary>
    public bool IsNull => Address < 0;
}

/// <summary>A filter in a dataset's pipeline.</summary>
/// <param name="Id">HDF5 filter id (1 deflate, 2 shuffle, 3 fletcher32, 32015 zstd, ...).</param>
/// <param name="Name">Filter name.</param>
/// <param name="ClientData">Filter parameters (deflate: level; shuffle: element size).</param>
public sealed record Hdf5Filter(int Id, string Name, IReadOnlyList<uint> ClientData);

/// <summary>
/// A managed, read-only HDF5 reader — enough of the format to read what h5py, MATLAB (v7.3
/// MAT files) and netCDF-4 write, with only the BCL (<see cref="System.IO.Compression.ZLibStream"/>
/// for deflate).
/// </summary>
/// <remarks>
/// <para>Covered: superblock versions 0–3 (any user block, 4- or 8-byte offsets), object header
/// versions 1 and 2 with continuation blocks, old-style (symbol table) and new-style (link
/// messages, dense fractal-heap storage) groups, soft links, compact and dense attributes,
/// compact/contiguous/chunked layouts with every chunk index (v1 B-tree, single chunk, implicit,
/// fixed array, extensible array, v2 B-tree), the deflate, shuffle and fletcher32 filters
/// (checksums are not verified), and datatypes: integers and IEEE floats of either byte order,
/// fixed and variable-length strings, compound, enum, array, variable-length sequences, opaque,
/// and object references.</para>
/// <para>Not covered, each with a <see cref="NotSupportedException"/> naming it: other filters
/// (szip, zstd, blosc, lz4, n-bit, scale-offset), virtual and external datasets, region references,
/// shared-message (SOHM) heaps, external links. Checksums are not verified. Files are written with <see cref="Hdf5Writer"/>.
/// A single read is limited to 2 GiB.</para>
/// <para>Datasets come back row-major in HDF5's own dimension order (C order, as h5py shows them).
/// The underlying stream must stay open while the file is in use; reads are thread-safe.</para>
/// </remarks>
public sealed class Hdf5File : IDisposable
{
    private readonly Hdf5Core _core;
    private Hdf5Group? _root;

    private Hdf5File(Hdf5Core core) => _core = core;

    /// <summary>Opens an HDF5 file for reading.</summary>
    /// <param name="path">File path.</param>
    public static Hdf5File Open(string path)
        => Open(new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 12, FileOptions.RandomAccess), leaveOpen: false);

    /// <summary>Opens an HDF5 stream for reading; it must be seekable and stay open while the file is used.</summary>
    /// <param name="stream">Seekable source.</param>
    /// <param name="leaveOpen">Leave the stream open when the file is disposed.</param>
    public static Hdf5File Open(Stream stream, bool leaveOpen = false)
    {
        ArgumentNullException.ThrowIfNull(stream);
        return new Hdf5File(new Hdf5Core(stream, !leaveOpen));
    }

    /// <summary>True when the file starts with (or, after a user block, contains) the HDF5 signature.</summary>
    /// <param name="path">File path.</param>
    public static bool IsHdf5(string path)
    {
        using var f = File.OpenRead(path);
        return Hdf5Core.IsHdf5(f);
    }

    /// <summary>True when the seekable stream holds the HDF5 signature; its position is restored.</summary>
    /// <param name="stream">Seekable stream.</param>
    public static bool IsHdf5(Stream stream) => Hdf5Core.IsHdf5(stream);

    internal Hdf5Core Core => _core;

    /// <summary>The root group <c>/</c>.</summary>
    public Hdf5Group Root => _root ??= new Hdf5Group(this, _core.RootAddress, "/", "/");

    /// <summary>Superblock version (0 and 1 are HDF5 1.6-compatible files; 2 and 3 come from <c>libver='latest'</c>).</summary>
    public int SuperblockVersion => _core.SuperblockVersion;

    /// <summary>Size of the user block before the HDF5 data (512 for MATLAB v7.3 files).</summary>
    public long UserBlockSize => _core.BaseAddress;

    /// <summary>Object at an absolute or root-relative path (<c>"group/dataset"</c>).</summary>
    /// <param name="path">Slash-separated path.</param>
    public Hdf5Object this[string path] => Root[path];

    /// <summary>The object an object reference points to.</summary>
    /// <param name="reference">The reference (from <see cref="Hdf5Dataset.ReadReferences"/> or an attribute).</param>
    public Hdf5Object Dereference(Hdf5Reference reference)
    {
        if (reference.IsNull) throw new ArgumentException("Cannot dereference a null HDF5 reference.", nameof(reference));
        return Hdf5Object.Open(this, reference.Address, "", "");
    }

    /// <summary>Closes the file (and the stream, unless it was opened with <c>leaveOpen</c>).</summary>
    public void Dispose() => _core.Dispose();
}

/// <summary>A group, dataset or committed datatype in an <see cref="Hdf5File"/>.</summary>
public abstract class Hdf5Object
{
    private IReadOnlyDictionary<string, Hdf5Attribute>? _attrs;

    private protected Hdf5Object(Hdf5File file, long address, string name, string path)
    {
        File = file;
        Address = address;
        Name = name;
        Path = path;
    }

    /// <summary>The file this object belongs to.</summary>
    public Hdf5File File { get; }

    /// <summary>Object header address (identifies the object; two links to one object share it).</summary>
    public long Address { get; }

    /// <summary>The link name this object was reached by ("" for a dereferenced object).</summary>
    public string Name { get; }

    /// <summary>The path it was reached by ("" for a dereferenced object).</summary>
    public string Path { get; }

    internal H5ObjectHeader Header => File.Core.Header(Address);

    /// <summary>Attributes by name, in creation order when the object tracks it and in name order otherwise (as h5py lists them).</summary>
    public IReadOnlyDictionary<string, Hdf5Attribute> Attributes
    {
        get
        {
            if (_attrs is not null) return _attrs;
            var d = new OrderedDictionary<string, Hdf5Attribute>(StringComparer.Ordinal);
            foreach (var a in File.Core.Attributes(Header)) d[a.Name] = new Hdf5Attribute(File.Core, a);
            return _attrs = d;
        }
    }

    /// <summary>An attribute's value (see <see cref="Hdf5Attribute.Value"/>), or null when absent.</summary>
    /// <param name="name">Attribute name.</param>
    public object? GetAttribute(string name) => Attributes.TryGetValue(name, out var a) ? a.Value : null;

    internal static Hdf5Object Open(Hdf5File file, long address, string name, string path)
    {
        var h = file.Core.Header(address);
        if (h.Find(H5Msg.Layout) is not null && h.Find(H5Msg.Dataspace) is not null)
            return new Hdf5Dataset(file, address, name, path);
        if (h.Find(H5Msg.Datatype) is not null && h.Find(H5Msg.Dataspace) is null && h.Find(H5Msg.SymbolTable) is null
            && h.Find(H5Msg.Link) is null && h.Find(H5Msg.LinkInfo) is null)
            return new Hdf5NamedType(file, address, name, path);
        return new Hdf5Group(file, address, name, path);
    }

    /// <inheritdoc />
    public override string ToString() => $"{GetType().Name} {(Path.Length > 0 ? Path : "@" + Address)}";
}

/// <summary>A committed (named) datatype stored as its own object.</summary>
public sealed class Hdf5NamedType : Hdf5Object
{
    internal Hdf5NamedType(Hdf5File file, long address, string name, string path) : base(file, address, name, path) { }

    /// <summary>The datatype.</summary>
    public Hdf5DataType DataType => File.Core.DataType(Header);
}

/// <summary>An HDF5 group: named links to groups, datasets and committed datatypes.</summary>
public sealed class Hdf5Group : Hdf5Object
{
    private List<H5Link>? _links;

    internal Hdf5Group(Hdf5File file, long address, string name, string path) : base(file, address, name, path) { }

    private List<H5Link> Links => _links ??= File.Core.Links(Header);

    /// <summary>Member names, in name order (or creation order when the group tracks it).</summary>
    public IReadOnlyList<string> Keys => Links.Select(l => l.Name).ToArray();

    /// <summary>True when the group has a member (or the relative path resolves).</summary>
    /// <param name="path">Name or slash-separated path.</param>
    public bool Contains(string path) => TryGet(path, out _);

    /// <summary>Member objects in <see cref="Keys"/> order; external links are skipped.</summary>
    public IEnumerable<Hdf5Object> Members
    {
        get
        {
            foreach (var l in Links)
                if (l.Kind != H5Link.External && TryGet(l.Name, out var o)) yield return o!;
        }
    }

    /// <summary>Sub-groups by name.</summary>
    public IReadOnlyDictionary<string, Hdf5Group> Groups => Members.OfType<Hdf5Group>().ToDictionary(g => g.Name, StringComparer.Ordinal);

    /// <summary>Datasets by name.</summary>
    public IReadOnlyDictionary<string, Hdf5Dataset> Datasets => Members.OfType<Hdf5Dataset>().ToDictionary(g => g.Name, StringComparer.Ordinal);

    /// <summary>The object at a relative path, or an absolute one starting with <c>/</c>.</summary>
    /// <param name="path">Slash-separated path.</param>
    public Hdf5Object this[string path] => TryGet(path, out var o) ? o! : throw new KeyNotFoundException($"No HDF5 object '{path}' in group '{Path}'.");

    /// <summary>The group at <paramref name="path"/>.</summary>
    /// <param name="path">Slash-separated path.</param>
    public Hdf5Group Group(string path) => this[path] as Hdf5Group ?? throw new InvalidOperationException($"'{path}' is not a group.");

    /// <summary>The dataset at <paramref name="path"/>.</summary>
    /// <param name="path">Slash-separated path.</param>
    public Hdf5Dataset Dataset(string path) => this[path] as Hdf5Dataset ?? throw new InvalidOperationException($"'{path}' is not a dataset.");

    /// <summary>Resolves a path (following soft links); false when any component is missing.</summary>
    /// <param name="path">Slash-separated path.</param>
    /// <param name="value">The object found.</param>
    public bool TryGet(string path, out Hdf5Object? value) => TryResolve(path, 0, out value);

    private bool TryResolve(string path, int depth, out Hdf5Object? value)
    {
        value = null;
        if (depth > 32) throw new InvalidDataException("HDF5 soft links form a cycle.");
        Hdf5Group current = path.StartsWith('/') ? File.Root : this;
        var parts = path.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length == 0) { value = current; return true; }
        for (var i = 0; i < parts.Length; i++)
        {
            if (parts[i] == ".") { if (i == parts.Length - 1) value = current; continue; }
            var link = current.Links.FirstOrDefault(l => l.Name == parts[i]);
            if (link is null) return false;
            Hdf5Object? next;
            switch (link.Kind)
            {
                case H5Link.Hard:
                    var childPath = current.Path.EndsWith('/') ? current.Path + link.Name : current.Path + "/" + link.Name;
                    next = Open(File, link.Address, link.Name, current.Path.Length == 0 ? "" : childPath);
                    break;
                case H5Link.Soft:
                    if (!current.TryResolve(link.SoftTarget!, depth + 1, out next)) return false;
                    break;
                default:
                    throw new NotSupportedException($"'{link.Name}' is an external link ({link.SoftTarget}); external links are not followed.");
            }
            if (i == parts.Length - 1) { value = next; return true; }
            if (next is not Hdf5Group g) return false;
            current = g;
        }
        return value is not null;
    }

    /// <summary>Visits every object below this group, depth first (h5py's <c>visititems</c>); each object is visited once.</summary>
    /// <param name="action">Called with each object.</param>
    public void Visit(Action<Hdf5Object> action)
    {
        var seen = new HashSet<long> { Address };
        Walk(this, action, seen);

        static void Walk(Hdf5Group g, Action<Hdf5Object> action, HashSet<long> seen)
        {
            foreach (var m in g.Members)
            {
                if (!seen.Add(m.Address)) continue;
                action(m);
                if (m is Hdf5Group sub) Walk(sub, action, seen);
            }
        }
    }
}

/// <summary>An HDF5 dataset: an n-dimensional array of one datatype.</summary>
public sealed class Hdf5Dataset : Hdf5Object
{
    private H5Dataspace? _space;
    private Hdf5DataType? _type;
    private H5DatasetStorage? _storage;

    internal Hdf5Dataset(Hdf5File file, long address, string name, string path) : base(file, address, name, path) { }

    private H5Dataspace Space => _space ??= File.Core.Dataspace(Header);

    private H5DatasetStorage Storage => _storage ??= new H5DatasetStorage(File.Core, Header, DataType, Space);

    /// <summary>The element datatype.</summary>
    public Hdf5DataType DataType => _type ??= File.Core.DataType(Header);

    /// <summary>Dimensions in HDF5 (C) order; empty for a scalar dataset.</summary>
    public int[] Shape => Space.Dims.Select(d => checked((int)d)).ToArray();

    /// <summary>Maximum dimensions; -1 marks an unlimited dimension.</summary>
    public long[] MaxShape => (Space.MaxDims ?? Space.Dims).ToArray();

    /// <summary>Number of dimensions.</summary>
    public int Rank => Space.Dims.Length;

    /// <summary>Number of elements (1 for a scalar, 0 for a null dataspace).</summary>
    public long Count => Space.Count;

    /// <summary>True for a scalar dataspace.</summary>
    public bool IsScalar => Space.IsScalar;

    /// <summary>Storage layout.</summary>
    public Hdf5Layout Layout => Storage.Kind;

    /// <summary>Chunk dimensions, or null when not chunked.</summary>
    public int[]? ChunkShape => Storage.ChunkDims?.ToArray();

    /// <summary>The filter pipeline (empty when unfiltered).</summary>
    public IReadOnlyList<Hdf5Filter> Filters => Storage.Filters.Select(f => new Hdf5Filter(f.Id, f.Name, f.ClientData)).ToArray();

    /// <summary>Raw element bytes of the whole dataset, row-major, in the file's byte order.</summary>
    public byte[] ReadRaw() => ReadRawRegion(new long[Rank], Space.Dims);

    private unsafe byte[] ReadRawRegion(long[] start, long[] count)
    {
        if (Space.IsNull) return [];
        var n = checked((int)(count.Aggregate(1L, (p, c) => p * c) * DataType.Size));
        var r = GC.AllocateUninitializedArray<byte>(n);
        fixed (byte* p = r) Storage.Read(start, count, p);
        return r;
    }

    private void CheckRegion(int[] start, int[] count)
    {
        ArgumentNullException.ThrowIfNull(start);
        ArgumentNullException.ThrowIfNull(count);
        var dims = Space.Dims;
        if (start.Length != dims.Length || count.Length != dims.Length) throw new ArgumentException($"Dataset has rank {dims.Length}; start and count need one entry per dimension.");
        for (var d = 0; d < dims.Length; d++)
            if (start[d] < 0 || count[d] < 0 || start[d] + (long)count[d] > dims[d])
                throw new ArgumentOutOfRangeException(nameof(count), $"Selection [{start[d]}, {start[d] + count[d]}) is outside dimension {d} of length {dims[d]}.");
    }

    /// <summary>
    /// The whole dataset as doubles (integer, float, bit-field or enum data), row-major in HDF5
    /// order; a scalar has shape [1]. int64/uint64 values beyond 2^53 lose their low bits.
    /// </summary>
    public NdArray Read()
    {
        var dims = Space.Dims;
        return ReadSlice(new int[dims.Length], dims.Select(d => checked((int)d)).ToArray());
    }

    /// <summary>
    /// A hyperslab as doubles: <paramref name="count"/> elements per dimension starting at
    /// <paramref name="start"/>. Only the chunks that intersect the selection are read.
    /// </summary>
    /// <param name="start">First index per dimension.</param>
    /// <param name="count">Number of elements per dimension.</param>
    public unsafe NdArray ReadSlice(int[] start, int[] count)
    {
        CheckRegion(start, count);
        if (Space.IsNull) return new NdArray([], 0);
        var t = DataType;
        if (!t.IsNumeric) throw new InvalidOperationException($"Dataset '{Path}' has type {t}, which is not numeric; use ReadArray, ReadStrings or ReadCompound.");
        var n = count.Aggregate(1L, (p, c) => p * c);
        var shape = count.Length == 0 ? [1] : count;
        var result = new double[checked((int)n)];
        var bt = t.Class == Hdf5TypeClass.Enum ? t.BaseType! : t;
        var ls = start.Select(x => (long)x).ToArray();
        var lc = count.Select(x => (long)x).ToArray();
        if (bt.Class == Hdf5TypeClass.Float && bt.Size == 8 && !bt.IsBigEndian && BitConverter.IsLittleEndian)
        {
            // Little-endian doubles decode straight into the result: no intermediate byte buffer.
            fixed (double* p = result) Storage.Read(ls, lc, (byte*)p);
            return new NdArray(result, shape);
        }
        var raw = ReadRawRegion(ls, lc);
        Hdf5Convert.ToDouble(t, raw, result);
        return new NdArray(result, shape);
    }

    /// <summary>
    /// The whole dataset, flattened row-major, in its natural .NET element type: <c>sbyte[]</c> ...
    /// <c>ulong[]</c>, <c>Half[]</c>, <c>float[]</c>, <c>double[]</c>, <c>bool[]</c> (h5py booleans),
    /// <c>string[]</c>, <see cref="Hdf5Reference"/>[], <c>Array[]</c> (variable-length and array
    /// elements), <c>Dictionary&lt;string, object?&gt;[]</c> (compound records) or <c>byte[][]</c> (opaque).
    /// </summary>
    public Array ReadArray() => Hdf5Convert.ToNatural(File.Core, DataType, ReadRaw(), checked((int)Count));

    /// <summary>Numeric data converted to <typeparamref name="T"/> (exact copy when the stored type matches; values out of range throw).</summary>
    /// <typeparam name="T">Target numeric type.</typeparam>
    public T[] Read<T>() where T : unmanaged, INumberBase<T>
    {
        if (!DataType.IsNumeric) throw new InvalidOperationException($"Dataset '{Path}' has type {DataType}, which is not numeric.");
        return Hdf5Convert.ToNumber<T>(DataType, ReadRaw(), checked((int)Count));
    }

    /// <summary>Fixed- or variable-length string data (a NULL variable-length string reads as null).</summary>
    public string?[] ReadStrings()
    {
        var t = DataType;
        if (t.Class != Hdf5TypeClass.String && !t.IsVariableLengthString) throw new InvalidOperationException($"Dataset '{Path}' has type {t}, which is not a string type.");
        return (string?[])ReadArray();
    }

    /// <summary>Object references (MATLAB cell arrays and netCDF dimension lists are made of these).</summary>
    public Hdf5Reference[] ReadReferences()
    {
        if (DataType.Class != Hdf5TypeClass.Reference) throw new InvalidOperationException($"Dataset '{Path}' has type {DataType}, not a reference type.");
        return (Hdf5Reference[])ReadArray();
    }

    /// <summary>A compound dataset split into one natural array per member, in member order (columnar, like a NumPy structured array's fields).</summary>
    public IReadOnlyDictionary<string, Array> ReadCompound()
        => Hdf5Convert.Columns(File.Core, DataType, ReadRaw(), checked((int)Count));
}

/// <summary>An attribute attached to an HDF5 object; its value is decoded on construction.</summary>
public sealed class Hdf5Attribute
{
    private readonly Hdf5Core _core;
    private readonly byte[] _data;
    private object? _value;
    private bool _decoded;

    internal Hdf5Attribute(Hdf5Core core, H5AttributeMsg m)
    {
        _core = core;
        Name = m.Name;
        DataType = m.Type;
        IsNull = m.Space.IsNull;
        IsScalar = m.Space.IsScalar;
        Shape = m.Space.Dims.Select(d => checked((int)d)).ToArray();
        _data = m.Data;
    }

    /// <summary>Attribute name.</summary>
    public string Name { get; }

    /// <summary>Element datatype.</summary>
    public Hdf5DataType DataType { get; }

    /// <summary>Dimensions (empty for a scalar).</summary>
    public int[] Shape { get; }

    /// <summary>True for a scalar dataspace.</summary>
    public bool IsScalar { get; }

    /// <summary>True for a null dataspace (no value at all).</summary>
    public bool IsNull { get; }

    /// <summary>
    /// The value as h5py returns it: a single element for a scalar attribute (<c>string</c>,
    /// <c>int</c>, <c>double</c>, ...), a flattened natural array (see
    /// <see cref="Hdf5Dataset.ReadArray"/>) otherwise, null for a null dataspace.
    /// </summary>
    public object? Value
    {
        get
        {
            if (_decoded) return _value;
            if (!IsNull)
            {
                var n = Shape.Aggregate(1, (p, d) => p * d);
                var a = Hdf5Convert.ToNatural(_core, DataType, _data, n);
                _value = IsScalar ? a.GetValue(0) : a;
            }
            _decoded = true;
            return _value;
        }
    }

    /// <summary>The raw element bytes, in the file's byte order.</summary>
    public ReadOnlySpan<byte> RawData => _data;

    /// <summary>All elements as doubles (numeric attributes only).</summary>
    public double[] ToDoubles()
    {
        if (!DataType.IsNumeric) throw new InvalidOperationException($"Attribute '{Name}' has type {DataType}, which is not numeric.");
        var n = IsNull ? 0 : Shape.Aggregate(1, (p, d) => p * d);
        var r = new double[n];
        Hdf5Convert.ToDouble(DataType, _data, r);
        return r;
    }

    /// <summary>The value as text: a string attribute, or the first string of a string array.</summary>
    public string? AsString() => Value switch
    {
        string s => s,
        string?[] { Length: > 0 } a => a[0],
        _ => null,
    };

    /// <inheritdoc />
    public override string ToString() => $"{Name}: {DataType}";
}
