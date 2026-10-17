using System.Buffers.Binary;
using System.Text;

namespace Gravicode.Science.SciNet.IO;

/// <summary>
/// File-level state of an open HDF5 file: superblock fields, the byte source, and caches of
/// parsed object headers and heaps. All addresses held by the parser are relative to
/// <see cref="BaseAddress"/>, which is where the superblock was found (after any user block).
/// </summary>
internal sealed class Hdf5Core : IDisposable
{
    private static ReadOnlySpan<byte> Signature => [0x89, (byte)'H', (byte)'D', (byte)'F', 0x0D, 0x0A, 0x1A, 0x0A];

    private readonly Dictionary<long, H5ObjectHeader> _headers = new();
    private readonly Dictionary<long, (byte[] Data, Dictionary<int, (int Offset, int Length)> Objects)> _globalHeaps = new();
    private readonly Dictionary<long, H5FractalHeap> _fractalHeaps = new();
    private readonly object _gate = new();

    public Hdf5Source Source { get; }
    public int SizeOfOffsets { get; }
    public int SizeOfLengths { get; }
    public long BaseAddress { get; }
    public int SuperblockVersion { get; }
    public long RootAddress { get; }

    public Hdf5Core(Stream stream, bool owns)
    {
        Source = new Hdf5Source(stream, owns);
        try
        {
            var super = FindSuperblock(Source) ?? throw new InvalidDataException("Not an HDF5 file: no superblock signature at offset 0, 512, 1024, 2048, ...");
            BaseAddress = super;
            var head = Source.Read(super, (int)Math.Min(128, Source.Length - super));
            if (head.Length < 16) throw new InvalidDataException("Malformed HDF5 file: the superblock is truncated.");
            var ver = head[8];
            SuperblockVersion = ver;
            if (ver is 0 or 1)
            {
                SizeOfOffsets = head[13];
                SizeOfLengths = head[14];
                CheckSizes();
                var b = new H5Buf(head, SizeOfOffsets, SizeOfLengths, 24 + (ver == 1 ? 4 : 0));
                b.Offset(); // base address: HDF5 itself trusts where the signature was found
                b.Offset(); // free-space info
                b.Offset(); // end of file
                b.Offset(); // driver info
                b.Offset(); // root entry: link name offset
                RootAddress = b.Offset();
            }
            else if (ver is 2 or 3)
            {
                SizeOfOffsets = head[9];
                SizeOfLengths = head[10];
                CheckSizes();
                var b = new H5Buf(head, SizeOfOffsets, SizeOfLengths, 12);
                b.Offset(); // base address
                b.Offset(); // superblock extension (shared message tables, B-tree K values: not needed to read)
                b.Offset(); // end of file
                RootAddress = b.Offset();
            }
            else throw new NotSupportedException($"HDF5 superblock version {ver} is not supported (0–3 are).");
            if (RootAddress < 0) throw new InvalidDataException("Malformed HDF5 file: undefined root group address.");
        }
        catch
        {
            Source.Dispose();
            throw;
        }
    }

    private void CheckSizes()
    {
        if (SizeOfOffsets is not (2 or 4 or 8) || SizeOfLengths is not (2 or 4 or 8))
            throw new InvalidDataException($"Malformed HDF5 superblock: offset size {SizeOfOffsets}, length size {SizeOfLengths}.");
    }

    /// <summary>The superblock sits at 0 or at a power of two from 512 up (after a user block).</summary>
    internal static long? FindSuperblock(Hdf5Source s)
    {
        Span<byte> sig = stackalloc byte[8];
        for (long pos = 0; pos + 8 <= s.Length; pos = pos == 0 ? 512 : pos * 2)
        {
            s.Read(pos, sig);
            if (sig.SequenceEqual(Signature)) return pos;
        }
        return null;
    }

    /// <summary>True when the stream holds an HDF5 signature at a valid superblock position; the position is restored.</summary>
    internal static bool IsHdf5(Stream stream)
    {
        if (!stream.CanSeek) return false;
        var keep = stream.Position;
        try
        {
            Span<byte> sig = stackalloc byte[8];
            for (long pos = 0; pos + 8 <= stream.Length; pos = pos == 0 ? 512 : pos * 2)
            {
                stream.Position = pos;
                if (stream.ReadAtLeast(sig, 8, throwOnEndOfStream: false) < 8) return false;
                if (sig.SequenceEqual(Signature)) return true;
            }
            return false;
        }
        finally
        {
            stream.Position = keep;
        }
    }

    public void Dispose() => Source.Dispose();

    // ------------------------------------------------------------------ raw access

    public byte[] Read(long address, int count) => Source.Read(BaseAddress + address, count);

    /// <summary>Up to <paramref name="count"/> bytes (fewer at the end of the file).</summary>
    public byte[] ReadUpTo(long address, int count)
    {
        var avail = Source.Length - (BaseAddress + address);
        if (avail <= 0) throw new InvalidDataException($"Malformed HDF5 file: address {address} is past the end of the file.");
        return Source.Read(BaseAddress + address, (int)Math.Min(count, avail));
    }

    public H5Buf Buf(byte[] data, int pos = 0, int end = -1) => new(data, SizeOfOffsets, SizeOfLengths, pos, end);

    public H5Buf Buf(long address, int count) => Buf(Read(address, count));

    // ------------------------------------------------------------------ object headers

    public H5ObjectHeader Header(long address)
    {
        lock (_gate)
            if (_headers.TryGetValue(address, out var h)) return h;
        var parsed = ParseHeader(address);
        lock (_gate) _headers[address] = parsed;
        return parsed;
    }

    private H5ObjectHeader ParseHeader(long address)
    {
        if (address < 0) throw new InvalidDataException("Malformed HDF5 file: undefined object header address.");
        var first = ReadUpTo(address, 64);
        var messages = new List<H5Message>();
        var pending = new Queue<(long Addr, long Len)>();
        if (first[0] == (byte)'O' && first[1] == (byte)'H' && first[2] == (byte)'D' && first[3] == (byte)'R')
        {
            var b = Buf(first);
            b.Skip(4);
            var ver = b.U8();
            if (ver != 2) throw new NotSupportedException($"HDF5 object header version {ver} is not supported.");
            var flags = b.U8();
            if ((flags & 0x20) != 0) b.Skip(16);
            if ((flags & 0x10) != 0) b.Skip(4);
            var size = (long)b.UN(1 << (flags & 3));
            var prefix = b.Pos;
            var chunk = Read(address, checked((int)(prefix + size)));
            ParseV2Messages(chunk, prefix, chunk.Length, flags, messages, pending);
            while (pending.Count > 0)
            {
                var (a, len) = pending.Dequeue();
                var c = Read(a, checked((int)len));
                var cb = Buf(c);
                cb.Signature("OCHK");
                ParseV2Messages(c, 4, c.Length - 4, flags, messages, pending);
            }
            return new H5ObjectHeader { Version = 2, Messages = messages, TracksAttributeOrder = (flags & 0x04) != 0 };
        }
        if (first[0] != 1) throw new InvalidDataException($"Malformed HDF5 file: no object header at address {address} (version byte {first[0]}).");
        var hb = Buf(first);
        hb.Skip(4);
        hb.U32(); // reference count
        var hsize = hb.U32();
        pending.Enqueue((address + 16, hsize));
        while (pending.Count > 0)
        {
            var (a, len) = pending.Dequeue();
            if (len == 0) continue;
            var c = Read(a, checked((int)len));
            var b = Buf(c);
            while (b.Remaining >= 8)
            {
                int type = b.U16();
                int size = b.U16();
                var mflags = b.U8();
                b.Skip(3);
                if (size > b.Remaining) throw new InvalidDataException("Malformed HDF5 object header: message runs past its block.");
                var data = b.Bytes(size);
                if (type == H5Msg.Continuation)
                {
                    var cb = Buf(data);
                    pending.Enqueue((cb.Offset(), cb.Length()));
                }
                else if (type != H5Msg.Nil) messages.Add(new H5Message(type, mflags, data, -1));
            }
        }
        return new H5ObjectHeader { Version = 1, Messages = messages };
    }

    private void ParseV2Messages(byte[] chunk, int start, int end, byte hflags, List<H5Message> messages, Queue<(long, long)> pending)
    {
        var b = Buf(chunk, start, end);
        var headerSize = (hflags & 0x04) != 0 ? 6 : 4;
        while (b.Remaining >= headerSize)
        {
            int type = b.U8();
            int size = b.U16();
            var mflags = b.U8();
            var order = (hflags & 0x04) != 0 ? b.U16() : -1;
            if (size > b.Remaining) throw new InvalidDataException("Malformed HDF5 object header: message runs past its chunk.");
            var data = b.Bytes(size);
            if (type == H5Msg.Continuation)
            {
                var cb = Buf(data);
                pending.Enqueue((cb.Offset(), cb.Length()));
            }
            else if (type != H5Msg.Nil) messages.Add(new H5Message(type, mflags, data, order));
        }
    }

    /// <summary>The message body, following a "shared" message to the committed object that holds it.</summary>
    public byte[] Resolve(H5Message m) => (m.Flags & 0x02) != 0 ? SharedTarget(m.Data, m.Type) : m.Data;

    private byte[] SharedTarget(byte[] shared, int type)
    {
        var b = Buf(shared);
        var ver = b.U8();
        var kind = b.U8();
        long addr;
        if (ver == 1)
        {
            b.Skip(6);
            addr = b.Offset();
        }
        else if (ver == 2) addr = b.Offset();
        else if (ver == 3)
        {
            if (kind != 2) throw new NotSupportedException("This HDF5 file stores messages in a shared object header message (SOHM) heap, which is not supported.");
            addr = b.Offset();
        }
        else throw new NotSupportedException($"HDF5 shared message version {ver} is not supported.");
        var target = Header(addr).Find(type) ?? throw new InvalidDataException($"Malformed HDF5 file: shared message of type {type} not found in its committed object.");
        return Resolve(target);
    }

    public Hdf5DataType DataType(H5ObjectHeader h)
    {
        var m = h.Find(H5Msg.Datatype) ?? throw new InvalidDataException("Malformed HDF5 dataset: no datatype message.");
        return Hdf5DataType.Parse(Buf(Resolve(m)));
    }

    public H5Dataspace Dataspace(H5ObjectHeader h)
    {
        var m = h.Find(H5Msg.Dataspace) ?? throw new InvalidDataException("Malformed HDF5 dataset: no dataspace message.");
        return H5Dataspace.Parse(Buf(Resolve(m)));
    }

    // ------------------------------------------------------------------ attributes

    public List<H5AttributeMsg> Attributes(H5ObjectHeader h)
    {
        var list = new List<H5AttributeMsg>();
        foreach (var m in h.All(H5Msg.Attribute)) list.Add(ParseAttribute(Resolve(m), m.CreationOrder));
        var track = false;
        if (h.Find(H5Msg.AttributeInfo) is { } ai)
        {
            var info = H5DenseInfo.ParseAttributeInfo(Buf(Resolve(ai)));
            track = info.TrackOrder;
            if (info.HeapAddress >= 0 && info.NameIndexAddress >= 0)
            {
                var heap = FractalHeap(info.HeapAddress);
                var tree = new H5BTreeV2(this, info.NameIndexAddress);
                tree.ForEach(rec =>
                {
                    // Type 8 record: heap ID, message flags (1), creation order (4), name hash (4).
                    var idLen = rec.Length - 9;
                    var id = rec[..idLen];
                    var mflags = rec[idLen];
                    var order = (int)BinaryPrimitives.ReadUInt32LittleEndian(rec[(idLen + 1)..]);
                    var body = heap.Get(id);
                    if ((mflags & 0x02) != 0) body = SharedTarget(body, H5Msg.Attribute);
                    list.Add(ParseAttribute(body, order));
                });
            }
        }
        if (track && list.All(a => a.CreationOrder >= 0)) list.Sort((a, b) => a.CreationOrder.CompareTo(b.CreationOrder));
        else list.Sort((a, b) => string.CompareOrdinal(a.Name, b.Name));
        return list;
    }

    private H5AttributeMsg ParseAttribute(byte[] data, int order)
    {
        var b = Buf(data);
        var ver = b.U8();
        if (ver is < 1 or > 3) throw new NotSupportedException($"HDF5 attribute message version {ver} is not supported.");
        var flags = b.U8();
        if (ver == 1) flags = 0;
        int nameSize = b.U16();
        int typeSize = b.U16();
        int spaceSize = b.U16();
        if (ver == 3) b.U8();
        var start = b.Pos;
        var nameRaw = b.Span(nameSize);
        var z = nameRaw.IndexOf((byte)0);
        var name = Encoding.UTF8.GetString(z >= 0 ? nameRaw[..z] : nameRaw);
        if (ver == 1) b.Align8(start);
        start = b.Pos;
        var typeBytes = b.Bytes(typeSize);
        if (ver == 1) b.Align8(start);
        start = b.Pos;
        var spaceBytes = b.Bytes(spaceSize);
        if (ver == 1) b.Align8(start);
        if ((flags & 1) != 0) typeBytes = SharedTarget(typeBytes, H5Msg.Datatype);
        if ((flags & 2) != 0) spaceBytes = SharedTarget(spaceBytes, H5Msg.Dataspace);
        var type = Hdf5DataType.Parse(Buf(typeBytes));
        var space = H5Dataspace.Parse(Buf(spaceBytes));
        var bytes = checked((int)(space.Count * type.Size));
        if (bytes > b.Remaining) throw new InvalidDataException($"Malformed HDF5 attribute '{name}': data is truncated.");
        return new H5AttributeMsg(name, type, space, b.Bytes(bytes), order);
    }

    // ------------------------------------------------------------------ links

    /// <summary>The members of a group, in name order (or creation order when the group tracks it).</summary>
    public List<H5Link> Links(H5ObjectHeader h)
    {
        var list = new List<H5Link>();
        if (h.Find(H5Msg.SymbolTable) is { } st)
        {
            var b = Buf(Resolve(st));
            var btree = b.Offset();
            var heap = b.Offset();
            var names = LocalHeap(heap);
            SymbolTable(btree, names, list);
            return list; // the B-tree is keyed by name, so this is already name order
        }
        foreach (var m in h.All(H5Msg.Link)) list.Add(H5Link.Parse(Buf(Resolve(m))));
        var track = false;
        if (h.Find(H5Msg.LinkInfo) is { } li)
        {
            var info = H5DenseInfo.ParseLinkInfo(Buf(Resolve(li)));
            track = info.TrackOrder;
            if (info.HeapAddress >= 0 && info.NameIndexAddress >= 0)
            {
                var heap = FractalHeap(info.HeapAddress);
                var tree = new H5BTreeV2(this, info.NameIndexAddress);
                tree.ForEach(rec => list.Add(H5Link.Parse(Buf(heap.Get(rec[4..])))));
            }
        }
        if (track && list.All(l => l.CreationOrder >= 0)) list.Sort((a, b) => a.CreationOrder.CompareTo(b.CreationOrder));
        else list.Sort((a, b) => string.CompareOrdinal(a.Name, b.Name));
        return list;
    }

    private byte[] LocalHeap(long address)
    {
        var b = Buf(address, 8 + 2 * SizeOfLengths + SizeOfOffsets);
        b.Signature("HEAP");
        b.Skip(4);
        var size = b.Length();
        b.Length();
        var data = b.Offset();
        return Read(data, checked((int)size));
    }

    private static string HeapString(byte[] heap, long offset)
    {
        if (offset < 0 || offset >= heap.Length) throw new InvalidDataException("Malformed HDF5 local heap offset.");
        var s = heap.AsSpan((int)offset);
        var z = s.IndexOf((byte)0);
        return Encoding.UTF8.GetString(z >= 0 ? s[..z] : s);
    }

    /// <summary>Walks a version 1 group B-tree down to its symbol table nodes.</summary>
    private void SymbolTable(long address, byte[] heap, List<H5Link> into, int depth = 0)
    {
        if (depth > 64) throw new InvalidDataException("Malformed HDF5 group B-tree: too deep.");
        var o = SizeOfOffsets;
        var hb = Buf(address, 8 + 2 * o);
        hb.Signature("TREE");
        var type = hb.U8();
        if (type != 0) throw new InvalidDataException("Malformed HDF5 file: a group B-tree node has the chunk node type.");
        var level = hb.U8();
        int entries = hb.U16();
        var keySize = SizeOfLengths;
        var b = Buf(address, 8 + 2 * o + entries * (keySize + o) + keySize);
        b.Pos = 8 + 2 * o;
        for (var i = 0; i < entries; i++)
        {
            b.Skip(keySize);
            var child = b.Offset();
            if (level > 0) SymbolTable(child, heap, into, depth + 1);
            else SymbolNode(child, heap, into);
        }
    }

    private void SymbolNode(long address, byte[] heap, List<H5Link> into)
    {
        var o = SizeOfOffsets;
        var hb = Buf(address, 8);
        hb.Signature("SNOD");
        hb.Skip(2);
        int n = hb.U16();
        var entry = 2 * o + 24;
        var b = Buf(address, 8 + n * entry);
        b.Pos = 8;
        for (var i = 0; i < n; i++)
        {
            var nameOff = b.Offset();
            var obj = b.Offset();
            var cache = b.U32();
            b.Skip(4);
            var scratch = b.Span(16);
            var name = HeapString(heap, nameOff);
            if (cache == 2)
            {
                var target = HeapString(heap, BinaryPrimitives.ReadUInt32LittleEndian(scratch));
                into.Add(new H5Link(name, H5Link.Soft, -1, target, -1));
            }
            else into.Add(new H5Link(name, H5Link.Hard, obj, null, -1));
        }
    }

    // ------------------------------------------------------------------ heaps

    public H5FractalHeap FractalHeap(long address)
    {
        lock (_gate)
            if (_fractalHeaps.TryGetValue(address, out var h)) return h;
        var heap = new H5FractalHeap(this, address);
        lock (_gate) _fractalHeaps[address] = heap;
        return heap;
    }

    /// <summary>
    /// Bytes of a variable-length element (length, global heap collection address, object index);
    /// null for a NULL element (undefined collection address).
    /// </summary>
    public byte[]? ReadVlen(ReadOnlySpan<byte> descriptor, int elementSize)
    {
        var b = Buf(descriptor.ToArray());
        var count = b.U32();
        var collection = b.Offset();
        var index = (int)b.U32();
        if (collection <= 0 || (count == 0 && index == 0)) return collection <= 0 ? null : [];
        var (data, objects) = GlobalHeap(collection);
        if (!objects.TryGetValue(index, out var obj)) throw new InvalidDataException($"Malformed HDF5 file: global heap object {index} not found in collection {collection}.");
        var want = (long)count * elementSize;
        var len = (int)Math.Min(obj.Length, want);
        return data.AsSpan(obj.Offset, len).ToArray();
    }

    private (byte[] Data, Dictionary<int, (int Offset, int Length)> Objects) GlobalHeap(long address)
    {
        lock (_gate)
            if (_globalHeaps.TryGetValue(address, out var g)) return g;
        var hb = Buf(address, 8 + SizeOfLengths);
        hb.Signature("GCOL");
        hb.Skip(4);
        var size = hb.Length();
        if (size < 8 + SizeOfLengths || size > int.MaxValue) throw new InvalidDataException("Malformed HDF5 global heap collection size.");
        var data = Read(address, (int)size);
        var b = Buf(data, 8 + SizeOfLengths);
        var objects = new Dictionary<int, (int, int)>();
        while (b.Remaining >= 8 + SizeOfLengths)
        {
            int idx = b.U16();
            if (idx == 0) break; // free space runs to the end of the collection
            b.Skip(6);
            var len = b.Length();
            if (len < 0 || len > b.Remaining) throw new InvalidDataException("Malformed HDF5 global heap object.");
            objects[idx] = (b.Pos, (int)len);
            b.Skip((int)len);
            b.Pos = Math.Min(b.End, b.Pos + (int)((8 - len % 8) % 8));
        }
        var result = (data, objects);
        lock (_gate) _globalHeaps[address] = result;
        return result;
    }
}

/// <summary>
/// Fractal heap (dense link and attribute storage). Direct blocks are located once by walking the
/// doubling table; an object's heap ID then gives its offset in the heap's address space.
/// </summary>
internal sealed class H5FractalHeap
{
    private readonly Hdf5Core _core;
    private readonly List<(long Offset, long Address, long Size)> _blocks = new();
    private readonly int _idLen, _offBytes, _lenBytes, _width, _maxDirectRows;
    private readonly long _startBlock, _maxDirect, _hugeTree;
    private readonly bool _filtered;
    private Dictionary<ulong, (long Address, long Length)>? _huge;

    public H5FractalHeap(Hdf5Core core, long address)
    {
        _core = core;
        var b = core.Buf(core.ReadUpTo(address, 256));
        b.Signature("FRHP");
        b.U8();
        _idLen = b.U16();
        var ioLen = b.U16();
        _filtered = ioLen > 0;
        b.U8();
        var maxManaged = b.U32();
        b.Length();
        _hugeTree = b.Offset();
        b.Length(); b.Offset();
        b.Length(); b.Length(); b.Length(); b.Length();
        b.Length(); b.Length(); b.Length(); b.Length();
        _width = b.U16();
        _startBlock = b.Length();
        _maxDirect = b.Length();
        var maxHeapBits = b.U16();
        b.U16();
        var root = b.Offset();
        var curRows = b.U16();
        if (_width <= 0 || _startBlock <= 0 || _maxDirect < _startBlock) throw new InvalidDataException("Malformed HDF5 fractal heap header.");
        _offBytes = (maxHeapBits + 7) / 8;
        _lenBytes = Math.Min((Log2(_maxDirect) + 7) / 8, H5Buf.LimitEncSize(maxManaged));
        _maxDirectRows = Log2(_maxDirect) - Log2(_startBlock) + 2;
        if (root >= 0)
        {
            if (curRows == 0) _blocks.Add((0, root, _startBlock));
            else Indirect(root, curRows, 0);
        }
        _blocks.Sort((x, y) => x.Offset.CompareTo(y.Offset));
    }

    private static int Log2(long x) => 63 - System.Numerics.BitOperations.LeadingZeroCount((ulong)x);

    private long RowSize(int r) => r <= 1 ? _startBlock : _startBlock << (r - 1);

    private void Indirect(long address, int rows, long offset, int depth = 0)
    {
        if (depth > 32) throw new InvalidDataException("Malformed HDF5 fractal heap: indirect blocks nested too deep.");
        var o = _core.SizeOfOffsets;
        var direct = Math.Min(rows, _maxDirectRows);
        var entrySize = o + (_filtered ? _core.SizeOfLengths + 4 : 0);
        var size = 4 + 1 + o + _offBytes + direct * _width * entrySize + (rows - direct) * _width * o + 4;
        var b = _core.Buf(address, size);
        b.Signature("FHIB");
        b.U8();
        b.Offset();
        b.UN(_offBytes);
        var pos = offset;
        for (var r = 0; r < rows; r++)
        {
            var rs = RowSize(r);
            for (var c = 0; c < _width; c++)
            {
                var child = b.Offset();
                if (r < _maxDirectRows)
                {
                    if (_filtered) { b.Length(); b.U32(); }
                    if (child >= 0) _blocks.Add((pos, child, rs));
                }
                else if (child >= 0)
                {
                    var childRows = Log2(rs) - Log2(_startBlock * _width) + 1;
                    Indirect(child, childRows, pos, depth + 1);
                }
                pos += rs;
            }
        }
    }

    /// <summary>The object a heap ID refers to.</summary>
    public byte[] Get(ReadOnlySpan<byte> id)
    {
        if (id.Length < 1) throw new InvalidDataException("Malformed HDF5 heap ID.");
        var kind = (id[0] >> 4) & 3;
        if (kind == 2)
        {
            // Tiny: the object lives in the ID itself.
            if (_idLen <= 18)
            {
                var n = (id[0] & 0x0F) + 1;
                return id.Slice(1, n).ToArray();
            }
            var len = (((id[0] & 0x0F) << 8) | id[1]) + 1;
            return id.Slice(2, len).ToArray();
        }
        if (kind == 1)
        {
            // Huge objects (larger than the managed limit, e.g. a > 64 KiB attribute) live outside
            // the heap: the ID holds the address and length directly when they fit, otherwise a
            // key into the heap's huge-object v2 B-tree (record type 1: address, length, ID).
            var o = _core.SizeOfOffsets;
            var l = _core.SizeOfLengths;
            if (_filtered) throw new NotSupportedException("Huge objects of a filtered fractal heap are not supported.");
            if (_idLen - 1 >= o + l)
            {
                var hb = _core.Buf(id.ToArray(), 1);
                var addr = hb.Offset();
                var len = hb.Length();
                return _core.Read(addr, checked((int)len));
            }
            var key = _core.Buf(id.ToArray(), 1).UN(Math.Min(8, id.Length - 1));
            if (_huge is null)
            {
                var map = new Dictionary<ulong, (long, long)>();
                if (_hugeTree >= 0)
                {
                    var tree = new H5BTreeV2(_core, _hugeTree);
                    if (tree.Type != 1) throw new NotSupportedException($"Fractal heap huge-object B-tree type {tree.Type} is not supported.");
                    tree.ForEach(rec =>
                    {
                        var rb = _core.Buf(rec.ToArray());
                        var a = rb.Offset();
                        var n = rb.Length();
                        map[rb.UN(l)] = (a, n);
                    });
                }
                _huge = map;
            }
            if (!_huge.TryGetValue(key, out var obj)) throw new InvalidDataException($"Malformed HDF5 fractal heap: huge object {key} not found.");
            return _core.Read(obj.Address, checked((int)obj.Length));
        }
        if (kind != 0) throw new InvalidDataException("Malformed HDF5 heap ID type.");
        if (_filtered) throw new NotSupportedException("Filtered (compressed) fractal heaps are not supported.");
        var b = _core.Buf(id.ToArray(), 1);
        var off = (long)b.UN(_offBytes);
        var length = (int)b.UN(_lenBytes);
        int lo = 0, hi = _blocks.Count - 1, found = -1;
        while (lo <= hi)
        {
            var mid = (lo + hi) >> 1;
            if (_blocks[mid].Offset <= off) { found = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        if (found < 0 || off + length > _blocks[found].Offset + _blocks[found].Size)
            throw new InvalidDataException($"Malformed HDF5 fractal heap: object at offset {off} is in no direct block.");
        var blk = _blocks[found];
        return _core.Read(blk.Address + (off - blk.Offset), length);
    }
}

/// <summary>Version 2 B-tree: records are visited in key order (internal-node records between their children).</summary>
internal sealed class H5BTreeV2
{
    public delegate void RecordAction(ReadOnlySpan<byte> record);

    private readonly Hdf5Core _core;
    private readonly int _nodeSize, _recSize, _depth, _maxNrecSize;
    private readonly int[] _cumSize;
    private readonly long _root;
    private readonly int _rootRecords;

    public int Type { get; }

    public int RecordSize => _recSize;

    public H5BTreeV2(Hdf5Core core, long address)
    {
        _core = core;
        var b = core.Buf(core.ReadUpTo(address, 64));
        b.Signature("BTHD");
        b.U8();
        Type = b.U8();
        _nodeSize = (int)b.U32();
        _recSize = b.U16();
        _depth = b.U16();
        b.Skip(2);
        _root = b.Offset();
        _rootRecords = b.U16();
        if (_recSize <= 0 || _nodeSize <= 10) throw new InvalidDataException("Malformed HDF5 v2 B-tree header.");
        // Field widths of child pointers, as HDF5's H5B2__hdr_init derives them.
        var o = core.SizeOfOffsets;
        var maxNrec = new long[_depth + 1];
        var cumMax = new long[_depth + 1];
        _cumSize = new int[_depth + 1];
        maxNrec[0] = (_nodeSize - 10) / _recSize;
        cumMax[0] = maxNrec[0];
        _maxNrecSize = H5Buf.LimitEncSize((ulong)maxNrec[0]);
        for (var d = 1; d <= _depth; d++)
        {
            var ptr = o + _maxNrecSize + (d > 1 ? _cumSize[d - 1] : 0);
            maxNrec[d] = (_nodeSize - (10 + ptr)) / (_recSize + ptr);
            cumMax[d] = (maxNrec[d] + 1) * cumMax[d - 1] + maxNrec[d];
            _cumSize[d] = H5Buf.LimitEncSize((ulong)cumMax[d]);
        }
    }

    public void ForEach(RecordAction action)
    {
        if (_root < 0 || _rootRecords == 0) return;
        Visit(_root, _rootRecords, _depth, action);
    }

    private void Visit(long address, int nrec, int depth, RecordAction action)
    {
        var data = _core.ReadUpTo(address, _nodeSize);
        var b = _core.Buf(data);
        b.Signature(depth == 0 ? "BTLF" : "BTIN");
        b.Skip(2);
        var recStart = b.Pos;
        if (depth == 0)
        {
            for (var i = 0; i < nrec; i++) action(data.AsSpan(recStart + i * _recSize, _recSize));
            return;
        }
        b.Pos = recStart + nrec * _recSize;
        var children = new (long Addr, int N)[nrec + 1];
        for (var i = 0; i <= nrec; i++)
        {
            var a = b.Offset();
            var n = (int)b.UN(_maxNrecSize);
            if (depth > 1) b.UN(_cumSize[depth - 1]);
            children[i] = (a, n);
        }
        for (var i = 0; i <= nrec; i++)
        {
            Visit(children[i].Addr, children[i].N, depth - 1, action);
            if (i < nrec) action(data.AsSpan(recStart + i * _recSize, _recSize));
        }
    }
}
