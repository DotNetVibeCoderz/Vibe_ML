using System.Buffers.Binary;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace Gravicode.Science.SciNet.IO;

/// <summary>
/// Positional, thread-safe reads from the file. A <see cref="FileStream"/> is read through its
/// handle with <see cref="RandomAccess"/> (no shared position, so chunk reads can run in parallel);
/// an exposable <see cref="MemoryStream"/> is read from its buffer; anything else under a lock.
/// </summary>
internal sealed class Hdf5Source : IDisposable
{
    private readonly Stream _stream;
    private readonly bool _owns;
    private readonly SafeFileHandle? _handle;
    private readonly byte[]? _memory;
    private readonly object _gate = new();

    public Hdf5Source(Stream stream, bool owns)
    {
        if (!stream.CanRead || !stream.CanSeek) throw new ArgumentException("HDF5 reading needs a readable, seekable stream.", nameof(stream));
        _stream = stream;
        _owns = owns;
        Length = stream.Length;
        if (stream is FileStream fs) _handle = fs.SafeFileHandle;
        else if (stream is MemoryStream ms && ms.TryGetBuffer(out var seg) && seg.Offset == 0) _memory = seg.Array;
    }

    public long Length { get; }

    public void Read(long pos, Span<byte> dst)
    {
        if (pos < 0 || pos > Length - dst.Length)
            throw new InvalidDataException($"Malformed HDF5 file: read of {dst.Length} bytes at {pos} runs past the end of the file ({Length} bytes).");
        if (_memory is not null)
        {
            _memory.AsSpan((int)pos, dst.Length).CopyTo(dst);
            return;
        }
        if (_handle is not null)
        {
            var done = 0;
            while (done < dst.Length)
            {
                var r = RandomAccess.Read(_handle, dst[done..], pos + done);
                if (r == 0) throw new EndOfStreamException("HDF5 file ended unexpectedly.");
                done += r;
            }
            return;
        }
        lock (_gate)
        {
            _stream.Position = pos;
            _stream.ReadExactly(dst);
        }
    }

    public byte[] Read(long pos, int count)
    {
        var b = new byte[count];
        Read(pos, b);
        return b;
    }

    public void Dispose()
    {
        if (_owns) _stream.Dispose();
    }
}

/// <summary>Little-endian cursor over a metadata block, sized by the file's offset/length widths.</summary>
internal sealed class H5Buf
{
    public readonly byte[] B;
    public int Pos;
    public readonly int End;
    private readonly int _o, _l;

    public H5Buf(byte[] b, int sizeOfOffsets, int sizeOfLengths, int pos = 0, int end = -1)
    {
        B = b;
        Pos = pos;
        End = end < 0 ? b.Length : end;
        _o = sizeOfOffsets;
        _l = sizeOfLengths;
    }

    public int Remaining => End - Pos;

    private void Need(int n)
    {
        if (n < 0 || Pos + n > End) throw new InvalidDataException($"Malformed HDF5 file: a metadata structure needs {n} bytes at {Pos} but ends at {End}.");
    }

    public byte U8() { Need(1); return B[Pos++]; }

    public ushort U16() { Need(2); var v = BinaryPrimitives.ReadUInt16LittleEndian(B.AsSpan(Pos)); Pos += 2; return v; }

    public uint U32() { Need(4); var v = BinaryPrimitives.ReadUInt32LittleEndian(B.AsSpan(Pos)); Pos += 4; return v; }

    public ulong U64() { Need(8); var v = BinaryPrimitives.ReadUInt64LittleEndian(B.AsSpan(Pos)); Pos += 8; return v; }

    /// <summary>An unsigned little-endian integer of <paramref name="n"/> bytes (0..8).</summary>
    public ulong UN(int n)
    {
        Need(n);
        ulong v = 0;
        for (var i = n - 1; i >= 0; i--) v = (v << 8) | B[Pos + i];
        Pos += n;
        return v;
    }

    /// <summary>A file address; the all-ones "undefined address" comes back as -1.</summary>
    public long Offset()
    {
        Need(_o);
        var undefined = true;
        for (var i = 0; i < _o; i++) if (B[Pos + i] != 0xFF) { undefined = false; break; }
        var v = UN(_o);
        if (undefined) return -1;
        if (v > long.MaxValue) throw new InvalidDataException("Malformed HDF5 file: address out of range.");
        return (long)v;
    }

    /// <summary>A length field; all-ones (used for "unlimited") comes back as -1.</summary>
    public long Length()
    {
        Need(_l);
        var allOnes = true;
        for (var i = 0; i < _l; i++) if (B[Pos + i] != 0xFF) { allOnes = false; break; }
        var v = UN(_l);
        if (allOnes) return -1;
        if (v > long.MaxValue) throw new InvalidDataException("Malformed HDF5 file: length out of range.");
        return (long)v;
    }

    public void Skip(int n) { Need(n); Pos += n; }

    public byte[] Bytes(int n) { Need(n); var r = B.AsSpan(Pos, n).ToArray(); Pos += n; return r; }

    public ReadOnlySpan<byte> Span(int n) { Need(n); var s = B.AsSpan(Pos, n); Pos += n; return s; }

    /// <summary>A NUL-terminated string (UTF-8); the cursor ends after the terminator.</summary>
    public string CString()
    {
        var s = B.AsSpan(Pos, End - Pos);
        var z = s.IndexOf((byte)0);
        if (z < 0) throw new InvalidDataException("Malformed HDF5 file: unterminated name.");
        var r = Encoding.UTF8.GetString(s[..z]);
        Pos += z + 1;
        return r;
    }

    /// <summary>Advances to the next multiple of 8 counted from <paramref name="start"/>.</summary>
    public void Align8(int start)
    {
        var rel = Pos - start;
        var pad = (8 - rel % 8) % 8;
        Pos = Math.Min(End, Pos + pad);
    }

    public void Signature(string sig)
    {
        Need(4);
        for (var i = 0; i < 4; i++)
            if (B[Pos + i] != sig[i])
                throw new InvalidDataException($"Malformed HDF5 file: expected a '{sig}' block, found '{Encoding.Latin1.GetString(B, Pos, 4)}'.");
        Pos += 4;
    }

    /// <summary>Bytes needed to encode <paramref name="x"/> (HDF5's <c>H5VM_limit_enc_size</c>).</summary>
    public static int LimitEncSize(ulong x) => (x == 0 ? 0 : 63 - System.Numerics.BitOperations.LeadingZeroCount(x)) / 8 + 1;
}

/// <summary>Object header message type numbers (HDF5 file format specification, section IV.A.2).</summary>
internal static class H5Msg
{
    public const int Nil = 0x00, Dataspace = 0x01, LinkInfo = 0x02, Datatype = 0x03, FillValueOld = 0x04, FillValue = 0x05,
        Link = 0x06, ExternalFiles = 0x07, Layout = 0x08, GroupInfo = 0x0A, FilterPipeline = 0x0B, Attribute = 0x0C,
        Continuation = 0x10, SymbolTable = 0x11, AttributeInfo = 0x15;
}

/// <summary>One raw header message.</summary>
internal sealed record H5Message(int Type, byte Flags, byte[] Data, int CreationOrder);

/// <summary>A parsed object header: its messages in file order (continuations followed).</summary>
internal sealed class H5ObjectHeader
{
    public required int Version { get; init; }
    public required List<H5Message> Messages { get; init; }
    public bool TracksAttributeOrder { get; init; }

    public H5Message? Find(int type)
    {
        foreach (var m in Messages) if (m.Type == type) return m;
        return null;
    }

    public IEnumerable<H5Message> All(int type)
    {
        foreach (var m in Messages) if (m.Type == type) yield return m;
    }
}

/// <summary>Dataspace message: rank 0 with <see cref="IsNull"/> false is a scalar.</summary>
internal sealed record H5Dataspace(long[] Dims, long[]? MaxDims, bool IsNull)
{
    public bool IsScalar => !IsNull && Dims.Length == 0;

    public long Count
    {
        get
        {
            if (IsNull) return 0;
            long n = 1;
            foreach (var d in Dims) n = checked(n * d);
            return n;
        }
    }

    public static H5Dataspace Parse(H5Buf b)
    {
        var ver = b.U8();
        var rank = b.U8();
        var flags = b.U8();
        var isNull = false;
        if (ver == 1) b.Skip(5);
        else if (ver == 2)
        {
            var kind = b.U8();
            isNull = kind == 2;
        }
        else throw new NotSupportedException($"HDF5 dataspace message version {ver} is not supported.");
        var dims = new long[rank];
        for (var i = 0; i < rank; i++) dims[i] = b.Length();
        long[]? max = null;
        if ((flags & 1) != 0)
        {
            max = new long[rank];
            for (var i = 0; i < rank; i++) max[i] = b.Length();
        }
        foreach (var d in dims) if (d < 0) throw new InvalidDataException("Malformed HDF5 dataspace: negative dimension.");
        return new H5Dataspace(dims, max, isNull);
    }
}

/// <summary>How a dataset's chunks are indexed.</summary>
internal enum H5ChunkIndex { BTreeV1, SingleChunk, Implicit, FixedArray, ExtensibleArray, BTreeV2 }

/// <summary>Data layout message (versions 1–4).</summary>
internal sealed class H5LayoutInfo
{
    public Hdf5Layout Kind;
    public byte[]? CompactData;
    public long Address = -1;
    public long ContiguousSize = -1;
    public int[] ChunkDims = [];
    public H5ChunkIndex Index;
    public bool DontFilterPartialEdge;
    public long SingleFilteredSize = -1;
    public uint SingleFilterMask;
    public int FaPageBits;
    public int EaMaxBits, EaIndexElements, EaMinPointers, EaMinElements, EaPageBits;

    public static H5LayoutInfo Parse(H5Buf b, int rank)
    {
        var l = new H5LayoutInfo();
        var ver = b.U8();
        if (ver is 1 or 2)
        {
            var nd = b.U8();
            var cls = b.U8();
            b.Skip(5);
            if (cls != 0) l.Address = b.Offset();
            var dims = new long[nd];
            for (var i = 0; i < nd; i++) dims[i] = b.U32();
            switch (cls)
            {
                case 0:
                    l.Kind = Hdf5Layout.Compact;
                    l.CompactData = b.Bytes((int)b.U32());
                    break;
                case 1:
                    l.Kind = Hdf5Layout.Contiguous;
                    break;
                case 2:
                    l.Kind = Hdf5Layout.Chunked;
                    l.Index = H5ChunkIndex.BTreeV1;
                    l.ChunkDims = ToChunkDims(dims, rank);
                    break;
                default: throw new InvalidDataException($"Malformed HDF5 layout class {cls}.");
            }
            return l;
        }
        // Version 5 (HDF5 2.0, libver latest) encodes the same fields as version 4.
        if (ver is not (3 or 4 or 5)) throw new NotSupportedException($"HDF5 data layout message version {ver} is not supported.");
        var klass = b.U8();
        switch (klass)
        {
            case 0:
                l.Kind = Hdf5Layout.Compact;
                l.CompactData = b.Bytes(b.U16());
                return l;
            case 1:
                l.Kind = Hdf5Layout.Contiguous;
                l.Address = b.Offset();
                l.ContiguousSize = b.Length();
                return l;
            case 2:
                l.Kind = Hdf5Layout.Chunked;
                break;
            case 3:
                throw new NotSupportedException("HDF5 virtual datasets (VDS) are not supported.");
            default:
                throw new InvalidDataException($"Malformed HDF5 layout class {klass}.");
        }
        if (ver == 3)
        {
            var nd = b.U8();
            l.Address = b.Offset();
            var dims = new long[nd];
            for (var i = 0; i < nd; i++) dims[i] = b.U32();
            l.Index = H5ChunkIndex.BTreeV1;
            l.ChunkDims = ToChunkDims(dims, rank);
            return l;
        }
        var flags = b.U8();
        var ndims = b.U8();
        var enc = b.U8();
        var d4 = new long[ndims];
        for (var i = 0; i < ndims; i++) d4[i] = (long)b.UN(enc);
        l.ChunkDims = ToChunkDims(d4, rank);
        l.DontFilterPartialEdge = (flags & 1) != 0;
        var idx = b.U8();
        switch (idx)
        {
            case 1:
                l.Index = H5ChunkIndex.SingleChunk;
                if ((flags & 2) != 0)
                {
                    l.SingleFilteredSize = b.Length();
                    l.SingleFilterMask = b.U32();
                }
                break;
            case 2:
                l.Index = H5ChunkIndex.Implicit;
                break;
            case 3:
                l.Index = H5ChunkIndex.FixedArray;
                l.FaPageBits = b.U8();
                break;
            case 4:
                l.Index = H5ChunkIndex.ExtensibleArray;
                l.EaMaxBits = b.U8();
                l.EaIndexElements = b.U8();
                l.EaMinPointers = b.U8();
                l.EaMinElements = b.U8();
                l.EaPageBits = b.U8();
                break;
            case 5:
                l.Index = H5ChunkIndex.BTreeV2;
                b.Skip(6);
                break;
            default:
                throw new NotSupportedException($"HDF5 chunk index type {idx} is not supported.");
        }
        l.Address = b.Offset();
        return l;
    }

    /// <summary>The layout stores rank + 1 chunk dimensions, the last being the element size.</summary>
    private static int[] ToChunkDims(long[] dims, int rank)
    {
        var n = dims.Length == rank + 1 ? rank : dims.Length;
        var r = new int[n];
        for (var i = 0; i < n; i++)
        {
            if (dims[i] <= 0 || dims[i] > int.MaxValue) throw new InvalidDataException($"Malformed HDF5 layout: chunk dimension {dims[i]}.");
            r[i] = (int)dims[i];
        }
        return r;
    }
}

/// <summary>One filter of a pipeline.</summary>
internal sealed record H5FilterInfo(int Id, string Name, int Flags, uint[] ClientData);

internal static class H5Filters
{
    public static List<H5FilterInfo> Parse(H5Buf b)
    {
        var ver = b.U8();
        var n = b.U8();
        var list = new List<H5FilterInfo>(n);
        if (ver == 1) b.Skip(6);
        else if (ver != 2) throw new NotSupportedException($"HDF5 filter pipeline message version {ver} is not supported.");
        for (var i = 0; i < n; i++)
        {
            int id = b.U16();
            var nameLen = ver == 1 || id >= 256 ? b.U16() : 0;
            int flags = b.U16();
            int nvals = b.U16();
            var name = "";
            if (nameLen > 0)
            {
                var start = b.Pos;
                var raw = b.Span(nameLen);
                var z = raw.IndexOf((byte)0);
                name = Encoding.ASCII.GetString(z >= 0 ? raw[..z] : raw);
                if (ver == 1) b.Align8(start);
            }
            var cd = new uint[nvals];
            for (var k = 0; k < nvals; k++) cd[k] = b.U32();
            if (ver == 1 && nvals % 2 == 1) b.Skip(4);
            if (name.Length == 0) name = DefaultName(id);
            list.Add(new H5FilterInfo(id, name, flags, cd));
        }
        return list;
    }

    public static string DefaultName(int id) => id switch
    {
        1 => "deflate",
        2 => "shuffle",
        3 => "fletcher32",
        4 => "szip",
        5 => "nbit",
        6 => "scaleoffset",
        307 => "bzip2",
        32001 => "blosc",
        32004 => "lz4",
        32008 => "bitshuffle",
        32013 => "zfp",
        32015 => "zstd",
        _ => $"filter {id}",
    };
}

/// <summary>Attribute message, with its datatype and dataspace already resolved.</summary>
internal sealed record H5AttributeMsg(string Name, Hdf5DataType Type, H5Dataspace Space, byte[] Data, int CreationOrder);

/// <summary>Link message (hard, soft or external).</summary>
internal sealed record H5Link(string Name, int Kind, long Address, string? SoftTarget, long CreationOrder)
{
    public const int Hard = 0, Soft = 1, External = 64;

    public static H5Link Parse(H5Buf b)
    {
        var ver = b.U8();
        if (ver != 1) throw new NotSupportedException($"HDF5 link message version {ver} is not supported.");
        var flags = b.U8();
        var kind = (flags & 0x08) != 0 ? b.U8() : Hard;
        long order = -1;
        if ((flags & 0x04) != 0) order = (long)b.U64();
        if ((flags & 0x10) != 0) b.U8(); // charset: names are decoded as UTF-8 either way (ASCII is a subset)
        var lenSize = 1 << (flags & 3);
        var nameLen = (int)b.UN(lenSize);
        var name = Encoding.UTF8.GetString(b.Span(nameLen));
        switch (kind)
        {
            case Hard:
                return new H5Link(name, kind, b.Offset(), null, order);
            case Soft:
            {
                var n = b.U16();
                return new H5Link(name, kind, -1, Encoding.UTF8.GetString(b.Span(n)), order);
            }
            default:
            {
                var n = b.U16();
                var raw = b.Span(n);
                var text = n > 1 ? Encoding.UTF8.GetString(raw[1..]).Replace('\0', ':').TrimEnd(':') : "";
                return new H5Link(name, External, -1, text, order);
            }
        }
    }
}

/// <summary>Link info / attribute info message: where dense storage lives.</summary>
internal sealed record H5DenseInfo(bool TrackOrder, long HeapAddress, long NameIndexAddress)
{
    public static H5DenseInfo ParseLinkInfo(H5Buf b)
    {
        b.U8();
        var flags = b.U8();
        if ((flags & 1) != 0) b.Skip(8);
        var heap = b.Offset();
        var name = b.Offset();
        return new H5DenseInfo((flags & 1) != 0, heap, name);
    }

    public static H5DenseInfo ParseAttributeInfo(H5Buf b)
    {
        b.U8();
        var flags = b.U8();
        if ((flags & 1) != 0) b.Skip(2);
        var heap = b.Offset();
        var name = b.Offset();
        return new H5DenseInfo((flags & 1) != 0, heap, name);
    }
}
