using System.IO.Compression;

namespace Gravicode.Science.SciNet.IO;

/// <summary>One stored chunk: element offsets of its first element, file address, stored size and filter mask.</summary>
internal readonly record struct H5Chunk(long[] Origin, long Address, long Size, uint FilterMask);

/// <summary>
/// Reads a rectangular region of a dataset's raw element bytes (row-major, file byte order).
/// Handles compact, contiguous and chunked layouts, every chunk index HDF5 1.8–2.0 writes, and
/// the deflate / shuffle / fletcher32 filters. Chunks are decompressed in parallel.
/// </summary>
internal sealed class H5DatasetStorage
{
    private readonly Hdf5Core _core;
    private readonly long[] _dims;
    private readonly long[] _maxDims;
    private readonly H5LayoutInfo _layout;
    private readonly List<H5FilterInfo> _filters;
    private readonly int _es;
    private readonly byte[]? _fill;
    private H5FixedArrayIndex? _fa;
    private H5ExtensibleArrayIndex? _ea;

    public H5DatasetStorage(Hdf5Core core, H5ObjectHeader h, Hdf5DataType type, H5Dataspace space)
    {
        _core = core;
        _dims = space.Dims;
        _maxDims = space.MaxDims ?? space.Dims;
        _es = type.Size;
        var lm = h.Find(H5Msg.Layout) ?? throw new InvalidDataException("Malformed HDF5 dataset: no data layout message.");
        _layout = H5LayoutInfo.Parse(core.Buf(core.Resolve(lm)), _dims.Length);
        _filters = h.Find(H5Msg.FilterPipeline) is { } fp ? H5Filters.Parse(core.Buf(core.Resolve(fp))) : [];
        if (h.Find(H5Msg.ExternalFiles) is not null) throw new NotSupportedException("HDF5 datasets stored in external files are not supported.");
        _fill = ParseFill(core, h);
        if (_fill is not null && (_fill.Length != _es || _fill.All(x => x == 0))) _fill = null;
        if (_layout.Kind == Hdf5Layout.Chunked && _layout.ChunkDims.Length != _dims.Length)
            throw new InvalidDataException("Malformed HDF5 layout: chunk rank differs from the dataspace rank.");
    }

    public Hdf5Layout Kind => _layout.Kind;
    public int[]? ChunkDims => _layout.Kind == Hdf5Layout.Chunked ? _layout.ChunkDims : null;
    public IReadOnlyList<H5FilterInfo> Filters => _filters;
    public byte[]? Fill => _fill;

    private static byte[]? ParseFill(Hdf5Core core, H5ObjectHeader h)
    {
        if (h.Find(H5Msg.FillValue) is { } fm)
        {
            var b = core.Buf(core.Resolve(fm));
            var ver = b.U8();
            if (ver is 1 or 2)
            {
                b.U8();
                b.U8();
                var defined = b.U8();
                if (ver == 1 || defined != 0)
                {
                    if (b.Remaining < 4) return null;
                    var size = (int)b.U32();
                    return size > 0 ? b.Bytes(size) : null;
                }
                return null;
            }
            if (ver == 3)
            {
                var flags = b.U8();
                if ((flags & 0x20) == 0) return null;
                var size = (int)b.U32();
                return size > 0 ? b.Bytes(size) : null;
            }
            return null;
        }
        if (h.Find(H5Msg.FillValueOld) is { } old)
        {
            var b = core.Buf(core.Resolve(old));
            var size = (int)b.U32();
            return size > 0 ? b.Bytes(size) : null;
        }
        return null;
    }

    /// <summary>Copies the region [start, start + count) into <paramref name="dst"/> (row-major over count).</summary>
    public unsafe void Read(long[] start, long[] count, byte* dst)
    {
        var rank = _dims.Length;
        long total = _es;
        foreach (var c in count) total *= c;
        if (total == 0) return;
        var full = true;
        for (var d = 0; d < rank; d++) if (start[d] != 0 || count[d] != _dims[d]) full = false;

        switch (_layout.Kind)
        {
            case Hdf5Layout.Compact:
            {
                var data = _layout.CompactData!;
                if (data.Length < Product(_dims) * _es) throw new InvalidDataException("Malformed HDF5 compact dataset: data is shorter than its dataspace.");
                fixed (byte* p = data) CopyRegion(p, new long[rank], ToInt(_dims), start, count, dst);
                return;
            }
            case Hdf5Layout.Contiguous:
            {
                if (_layout.Address < 0) { FillAll(dst, total); return; }
                var baseAddr = _core.BaseAddress + _layout.Address;
                if (full)
                {
                    ReadInto(baseAddr, dst, total);
                    return;
                }
                ReadContiguousRegion(baseAddr, start, count, dst);
                return;
            }
        }

        var chunks = Chunks(start, count);
        var covered = chunks.Count == ChunkCount(start, count);
        if (!covered) FillAll(dst, total);
        var cdims = _layout.ChunkDims;
        var chunkBytes = Product(cdims) * _es;
        var outPtr = (IntPtr)dst;
        void One(H5Chunk ch)
        {
            var data = LoadChunk(ch, chunkBytes);
            fixed (byte* p = data) CopyRegion(p, ch.Origin, cdims, start, count, (byte*)outPtr);
        }
        if (chunks.Count > 1 && (_filters.Count > 0 || total >= 1 << 20) && Environment.ProcessorCount > 1)
        {
            try { Parallel.ForEach(chunks, One); }
            catch (AggregateException e) when (e.InnerExceptions.Count > 0)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
            }
        }
        else
            foreach (var ch in chunks) One(ch);
    }

    private unsafe void ReadInto(long fileOffset, byte* dst, long total)
    {
        const int Max = 1 << 30;
        for (long done = 0; done < total; done += Max)
        {
            var n = (int)Math.Min(Max, total - done);
            _core.Source.Read(fileOffset + done, new Span<byte>(dst + done, n));
        }
    }

    private unsafe void ReadContiguousRegion(long baseAddr, long[] start, long[] count, byte* dst)
    {
        var rank = _dims.Length;
        if (rank == 0) { ReadInto(baseAddr, dst, _es); return; }
        var run = count[rank - 1] * _es;
        var idx = new long[rank];
        Array.Copy(start, idx, rank);
        var strides = new long[rank];
        long s = 1;
        for (var d = rank - 1; d >= 0; d--) { strides[d] = s; s *= _dims[d]; }
        long outer = 1;
        for (var d = 0; d < rank - 1; d++) outer *= count[d];
        long o = 0;
        for (long k = 0; k < outer; k++)
        {
            long off = 0;
            for (var d = 0; d < rank; d++) off += idx[d] * strides[d];
            ReadInto(baseAddr + off * _es, dst + o, run);
            o += run;
            for (var d = rank - 2; d >= 0; d--)
            {
                if (++idx[d] < start[d] + count[d]) break;
                idx[d] = start[d];
            }
        }
    }

    private unsafe void FillAll(byte* dst, long total)
    {
        if (_fill is null) { new Span<byte>(dst, (int)Math.Min(total, int.MaxValue)).Clear(); return; }
        for (long i = 0; i < total; i += _es)
            for (var k = 0; k < _es; k++) dst[i + k] = _fill[k];
    }

    private static long Product(ReadOnlySpan<long> d) { long p = 1; foreach (var x in d) p *= x; return p; }
    private static long Product(ReadOnlySpan<int> d) { long p = 1; foreach (var x in d) p *= x; return p; }

    private static int[] ToInt(long[] d)
    {
        var r = new int[d.Length];
        for (var i = 0; i < d.Length; i++) r[i] = checked((int)d[i]);
        return r;
    }

    /// <summary>Copies the intersection of a chunk (origin, full chunk dims) with the selection into the output.</summary>
    private unsafe void CopyRegion(byte* src, long[] origin, int[] cdims, long[] start, long[] count, byte* dst)
    {
        var rank = cdims.Length;
        if (rank == 0) { Buffer.MemoryCopy(src, dst, _es, _es); return; }
        Span<long> lo = stackalloc long[rank];
        Span<long> hi = stackalloc long[rank];
        for (var d = 0; d < rank; d++)
        {
            lo[d] = Math.Max(origin[d], start[d]);
            hi[d] = Math.Min(Math.Min(origin[d] + cdims[d], start[d] + count[d]), _dims[d]);
            if (lo[d] >= hi[d]) return;
        }
        Span<long> cstride = stackalloc long[rank];
        Span<long> ostride = stackalloc long[rank];
        long cs = 1, os = 1;
        for (var d = rank - 1; d >= 0; d--)
        {
            cstride[d] = cs; cs *= cdims[d];
            ostride[d] = os; os *= count[d];
        }
        var last = rank - 1;
        var run = (hi[last] - lo[last]) * _es;
        Span<long> idx = stackalloc long[rank];
        lo.CopyTo(idx);
        while (true)
        {
            long so = 0, doff = 0;
            for (var d = 0; d < rank; d++)
            {
                so += (idx[d] - origin[d]) * cstride[d];
                doff += (idx[d] - start[d]) * ostride[d];
            }
            Buffer.MemoryCopy(src + so * _es, dst + doff * _es, run, run);
            var d2 = last - 1;
            for (; d2 >= 0; d2--)
            {
                if (++idx[d2] < hi[d2]) break;
                idx[d2] = lo[d2];
            }
            if (d2 < 0) return;
        }
    }

    // ------------------------------------------------------------------ chunks

    private long ChunkCount(long[] start, long[] count)
    {
        long n = 1;
        var c = _layout.ChunkDims;
        for (var d = 0; d < _dims.Length; d++)
        {
            var end = Math.Min(start[d] + count[d], _dims[d]);
            n *= (end - 1) / c[d] - start[d] / c[d] + 1;
        }
        return n;
    }

    private bool Overlaps(long[] origin, long[] start, long[] count)
    {
        var c = _layout.ChunkDims;
        for (var d = 0; d < origin.Length; d++)
            if (origin[d] >= start[d] + count[d] || origin[d] + c[d] <= start[d] || origin[d] >= _dims[d]) return false;
        return true;
    }

    /// <summary>Chunk coordinates (scaled) overlapping the selection, in row-major order.</summary>
    private IEnumerable<long[]> GridCells(long[] start, long[] count)
    {
        var rank = _dims.Length;
        var c = _layout.ChunkDims;
        var lo = new long[rank];
        var hi = new long[rank];
        for (var d = 0; d < rank; d++)
        {
            lo[d] = start[d] / c[d];
            hi[d] = (Math.Min(start[d] + count[d], _dims[d]) - 1) / c[d];
        }
        var idx = (long[])lo.Clone();
        while (true)
        {
            yield return (long[])idx.Clone();
            var d = rank - 1;
            for (; d >= 0; d--)
            {
                if (++idx[d] <= hi[d]) break;
                idx[d] = lo[d];
            }
            if (d < 0) yield break;
        }
    }

    /// <summary>Row-major linear index of a chunk over the chunk grid of the maximum dimensions.</summary>
    private long Linear(long[] scaled, int unlimitedFirst = -1)
    {
        var rank = scaled.Length;
        var c = _layout.ChunkDims;
        var order = new int[rank];
        if (unlimitedFirst > 0)
        {
            order[0] = unlimitedFirst;
            var k = 1;
            for (var d = 0; d < rank; d++) if (d != unlimitedFirst) order[k++] = d;
        }
        else for (var d = 0; d < rank; d++) order[d] = d;
        long idx = 0;
        for (var k = 0; k < rank; k++)
        {
            var d = order[k];
            var max = _maxDims[d] < 0 ? _dims[d] : Math.Max(_maxDims[d], _dims[d]);
            var n = (max + c[d] - 1) / c[d];
            idx = k == 0 ? scaled[d] : idx * n + scaled[d];
        }
        return idx;
    }

    private long[] Origin(long[] scaled)
    {
        var o = new long[scaled.Length];
        for (var d = 0; d < o.Length; d++) o[d] = scaled[d] * _layout.ChunkDims[d];
        return o;
    }

    private List<H5Chunk> Chunks(long[] start, long[] count)
    {
        var list = new List<H5Chunk>();
        var chunkBytes = Product(_layout.ChunkDims) * _es;
        if (_layout.Address < 0) return list; // nothing written yet: all fill value
        switch (_layout.Index)
        {
            case H5ChunkIndex.BTreeV1:
                BTreeV1(_layout.Address, start, count, list, 0);
                break;
            case H5ChunkIndex.SingleChunk:
            {
                var size = _layout.SingleFilteredSize >= 0 ? _layout.SingleFilteredSize : chunkBytes;
                list.Add(new H5Chunk(new long[_dims.Length], _layout.Address, size, _layout.SingleFilterMask));
                break;
            }
            case H5ChunkIndex.Implicit:
                foreach (var s in GridCells(start, count))
                    list.Add(new H5Chunk(Origin(s), _layout.Address + Linear(s) * chunkBytes, chunkBytes, 0));
                break;
            case H5ChunkIndex.FixedArray:
            {
                _fa ??= new H5FixedArrayIndex(_core, _layout.Address);
                foreach (var s in GridCells(start, count))
                    if (_fa.Get(Linear(s)) is { } e && e.Address >= 0)
                        list.Add(new H5Chunk(Origin(s), e.Address, e.Size < 0 ? chunkBytes : e.Size, e.Mask));
                break;
            }
            case H5ChunkIndex.ExtensibleArray:
            {
                _ea ??= new H5ExtensibleArrayIndex(_core, _layout);
                var unlim = Array.FindIndex(_maxDims, m => m < 0);
                foreach (var s in GridCells(start, count))
                    if (_ea.Get(Linear(s, unlim)) is { } e && e.Address >= 0)
                        list.Add(new H5Chunk(Origin(s), e.Address, e.Size < 0 ? chunkBytes : e.Size, e.Mask));
                break;
            }
            case H5ChunkIndex.BTreeV2:
            {
                var tree = new H5BTreeV2(_core, _layout.Address);
                var rank = _dims.Length;
                var o = _core.SizeOfOffsets;
                var filtered = tree.Type == 11;
                var sizeLen = filtered ? tree.RecordSize - o - 4 - 8 * rank : 0;
                tree.ForEach(rec =>
                {
                    var b = _core.Buf(rec.ToArray());
                    var addr = b.Offset();
                    long size = chunkBytes;
                    uint mask = 0;
                    if (filtered)
                    {
                        size = (long)b.UN(sizeLen);
                        mask = b.U32();
                    }
                    var scaled = new long[rank];
                    for (var d = 0; d < rank; d++) scaled[d] = (long)b.U64();
                    var origin = Origin(scaled);
                    if (addr >= 0 && Overlaps(origin, start, count)) list.Add(new H5Chunk(origin, addr, size, mask));
                });
                break;
            }
        }
        return list;
    }

    private void BTreeV1(long address, long[] start, long[] count, List<H5Chunk> list, int depth)
    {
        if (depth > 64) throw new InvalidDataException("Malformed HDF5 chunk B-tree: too deep.");
        var o = _core.SizeOfOffsets;
        var rank = _dims.Length;
        var head = _core.Buf(address, 8 + 2 * o);
        head.Signature("TREE");
        var type = head.U8();
        if (type != 1) throw new InvalidDataException("Malformed HDF5 file: a chunk B-tree node has the group node type.");
        var level = head.U8();
        int entries = head.U16();
        var keySize = 8 + 8 * (rank + 1);
        var b = _core.Buf(address, 8 + 2 * o + entries * (keySize + o) + keySize);
        b.Pos = 8 + 2 * o;
        for (var i = 0; i < entries; i++)
        {
            var size = b.U32();
            var mask = b.U32();
            var origin = new long[rank];
            for (var d = 0; d < rank; d++) origin[d] = (long)b.U64();
            b.Skip(8);
            var child = b.Offset();
            if (level > 0) BTreeV1(child, start, count, list, depth + 1);
            else if (child >= 0 && Overlaps(origin, start, count)) list.Add(new H5Chunk(origin, child, size, mask));
        }
    }

    // ------------------------------------------------------------------ filters

    private byte[] LoadChunk(H5Chunk ch, long chunkBytes)
    {
        if (ch.Size <= 0 || ch.Size > int.MaxValue) throw new InvalidDataException($"Malformed HDF5 chunk size {ch.Size}.");
        var data = _core.Read(ch.Address, (int)ch.Size);
        var skipAll = false;
        if (_layout.DontFilterPartialEdge)
            for (var d = 0; d < _dims.Length; d++)
                if (ch.Origin[d] + _layout.ChunkDims[d] > _dims[d]) skipAll = true;
        if (!skipAll)
        {
            for (var i = _filters.Count - 1; i >= 0; i--)
            {
                if ((ch.FilterMask & (1u << i)) != 0) continue;
                var f = _filters[i];
                data = f.Id switch
                {
                    1 => Inflate(data, chunkBytes),
                    2 => Unshuffle(data, f.ClientData.Length > 0 ? (int)f.ClientData[0] : _es),
                    3 => data.Length >= 4 ? data[..^4] : throw new InvalidDataException("Malformed HDF5 chunk: shorter than its Fletcher-32 checksum."),
                    _ => throw new NotSupportedException($"HDF5 filter '{f.Name}' (id {f.Id}) is not supported; SciNet decodes deflate (gzip), shuffle and fletcher32. Re-save the file with gzip compression (e.g. h5repack -f GZIP=4)."),
                };
            }
        }
        if (data.Length < chunkBytes) throw new InvalidDataException($"Malformed HDF5 chunk: decoded to {data.Length} bytes, expected {chunkBytes}.");
        return data;
    }

    private static byte[] Inflate(byte[] src, long expected)
    {
        try
        {
            using var z = new ZLibStream(new MemoryStream(src, writable: false), CompressionMode.Decompress);
            var buf = new byte[Math.Max(16, expected)];
            var got = z.ReadAtLeast(buf, buf.Length, throwOnEndOfStream: false);
            if (got < buf.Length) return buf[..got];
            // A filter after deflate (e.g. fletcher32 applied first on write) makes the output larger.
            using var more = new MemoryStream();
            z.CopyTo(more);
            if (more.Length == 0) return buf;
            var r = new byte[buf.Length + more.Length];
            buf.CopyTo(r, 0);
            more.ToArray().CopyTo(r, buf.Length);
            return r;
        }
        catch (InvalidDataException e)
        {
            throw new InvalidDataException("Malformed HDF5 chunk: deflate data is corrupt.", e);
        }
    }

    /// <summary>Reverses the shuffle filter: byte k of every element was stored contiguously.</summary>
    internal static byte[] Unshuffle(byte[] src, int es)
    {
        if (es <= 1) return src;
        var n = src.Length / es;
        var dst = new byte[src.Length];
        for (var b = 0; b < es; b++)
        {
            var s = src.AsSpan(b * n, n);
            for (var i = 0; i < n; i++) dst[i * es + b] = s[i];
        }
        src.AsSpan(n * es).CopyTo(dst.AsSpan(n * es));
        return dst;
    }
}

/// <summary>An entry of a fixed or extensible array chunk index; <see cref="Size"/> is -1 for unfiltered chunks.</summary>
internal readonly record struct H5IndexEntry(long Address, long Size, uint Mask);

internal static class H5IndexEntries
{
    public static H5IndexEntry Parse(H5Buf b, int clientId, int entrySize, int sizeOfOffsets)
    {
        var addr = b.Offset();
        if (clientId == 0) return new H5IndexEntry(addr, -1, 0);
        var size = (long)b.UN(entrySize - sizeOfOffsets - 4);
        var mask = b.U32();
        return new H5IndexEntry(addr, size, mask);
    }
}

/// <summary>Fixed array chunk index (layout v4, datasets without unlimited dimensions).</summary>
internal sealed class H5FixedArrayIndex
{
    private readonly H5IndexEntry[] _entries;

    public H5FixedArrayIndex(Hdf5Core core, long address)
    {
        var o = core.SizeOfOffsets;
        var b = core.Buf(core.ReadUpTo(address, 64));
        b.Signature("FAHD");
        b.U8();
        var client = b.U8();
        var entrySize = b.U8();
        var pageBits = b.U8();
        var max = b.Length();
        var dblk = b.Offset();
        if (max < 0 || max > int.MaxValue) throw new InvalidDataException("Malformed HDF5 fixed array header.");
        _entries = new H5IndexEntry[max];
        Array.Fill(_entries, new H5IndexEntry(-1, -1, 0));
        if (dblk < 0 || max == 0) return;
        var prefix = 4 + 1 + 1 + o;
        var pageSize = 1L << pageBits;
        if (max <= pageSize)
        {
            var db = core.Buf(dblk, prefix + (int)max * entrySize);
            db.Signature("FADB");
            db.Pos = prefix;
            for (var i = 0; i < max; i++) _entries[i] = H5IndexEntries.Parse(db, client, entrySize, o);
            return;
        }
        var npages = (int)((max + pageSize - 1) / pageSize);
        var bitmapBytes = (npages + 7) / 8;
        var hb = core.Buf(dblk, prefix + bitmapBytes);
        hb.Signature("FADB");
        hb.Pos = prefix;
        var bitmap = hb.Span(bitmapBytes).ToArray();
        var pagesStart = dblk + prefix + bitmapBytes + 4;
        var pageBytes = pageSize * entrySize + 4;
        for (var p = 0; p < npages; p++)
        {
            if ((bitmap[p / 8] & (0x80 >> (p % 8))) == 0) continue;
            var n = (int)Math.Min(pageSize, max - p * pageSize);
            var pb = core.Buf(pagesStart + p * pageBytes, n * entrySize);
            for (var i = 0; i < n; i++) _entries[p * pageSize + i] = H5IndexEntries.Parse(pb, client, entrySize, o);
        }
    }

    public H5IndexEntry? Get(long index) => index >= 0 && index < _entries.Length ? _entries[index] : null;
}

/// <summary>
/// Extensible array chunk index (layout v4, one unlimited dimension): an index block holding the
/// first elements and data block pointers, then super blocks of doubling data blocks
/// (HDF5's H5EA; the block geometry below mirrors <c>H5EA__hdr_init</c>).
/// </summary>
internal sealed class H5ExtensibleArrayIndex
{
    private readonly Hdf5Core _core;
    private readonly int _client, _elemSize, _arrOff, _idxElmts, _dataMin, _iblockSblks, _pageNelmts;
    private readonly (long Ndblks, long DblkNelmts, long StartIdx, long StartDblk)[] _sblk;
    private readonly H5IndexEntry[] _idxEntries;
    private readonly long[] _dblkAddrs;
    private readonly long[] _sblkAddrs;
    private readonly Dictionary<long, (long[] Dblks, byte[]? Bitmaps, int PageInitSize)> _superBlocks = new();
    private readonly object _gate = new();

    public H5ExtensibleArrayIndex(Hdf5Core core, H5LayoutInfo layout)
    {
        _core = core;
        var o = core.SizeOfOffsets;
        var b = core.Buf(core.ReadUpTo(layout.Address, 128));
        b.Signature("EAHD");
        b.U8();
        _client = b.U8();
        _elemSize = b.U8();
        var maxBits = b.U8();
        _idxElmts = b.U8();
        _dataMin = b.U8();
        var supMinPtrs = b.U8();
        var pageBits = b.U8();
        for (var i = 0; i < 6; i++) b.Length();
        var iblock = b.Offset();
        if (_dataMin <= 0 || supMinPtrs <= 0) throw new InvalidDataException("Malformed HDF5 extensible array header.");
        _arrOff = (maxBits + 7) / 8;
        _pageNelmts = 1 << pageBits;
        var nsblks = 1 + (maxBits - Log2(_dataMin));
        _sblk = new (long, long, long, long)[nsblks];
        long startIdx = 0, startDblk = 0;
        for (var u = 0; u < nsblks; u++)
        {
            var nd = 1L << (u / 2);
            var ne = (1L << ((u + 1) / 2)) * _dataMin;
            _sblk[u] = (nd, ne, startIdx, startDblk);
            startIdx += nd * ne;
            startDblk += nd;
        }
        _iblockSblks = 2 * Log2(supMinPtrs);
        var ndblk = 2 * (supMinPtrs - 1);
        var nsblkAddrs = Math.Max(0, nsblks - _iblockSblks);
        _idxEntries = new H5IndexEntry[_idxElmts];
        _dblkAddrs = new long[ndblk];
        _sblkAddrs = new long[nsblkAddrs];
        Array.Fill(_idxEntries, new H5IndexEntry(-1, -1, 0));
        Array.Fill(_dblkAddrs, -1L);
        Array.Fill(_sblkAddrs, -1L);
        if (iblock < 0) return;
        var ib = core.Buf(iblock, 4 + 1 + 1 + o + _idxElmts * _elemSize + (ndblk + nsblkAddrs) * o);
        ib.Signature("EAIB");
        ib.Skip(2);
        ib.Offset();
        for (var i = 0; i < _idxElmts; i++) _idxEntries[i] = H5IndexEntries.Parse(ib, _client, _elemSize, o);
        for (var i = 0; i < ndblk; i++) _dblkAddrs[i] = ib.Offset();
        for (var i = 0; i < nsblkAddrs; i++) _sblkAddrs[i] = ib.Offset();
    }

    private static int Log2(long x) => 63 - System.Numerics.BitOperations.LeadingZeroCount((ulong)x);

    public H5IndexEntry? Get(long idx)
    {
        if (idx < 0) return null;
        if (idx < _idxElmts) return _idxEntries[idx];
        idx -= _idxElmts;
        var s = Log2(idx / _dataMin + 1);
        if (s >= _sblk.Length) return null;
        var info = _sblk[s];
        var rel = idx - info.StartIdx;
        var dblkInSblk = rel / info.DblkNelmts;
        var elem = rel % info.DblkNelmts;
        long dblk;
        var paged = info.DblkNelmts > _pageNelmts;
        if (s < _iblockSblks)
        {
            var di = info.StartDblk + dblkInSblk;
            if (di >= _dblkAddrs.Length) return null;
            dblk = _dblkAddrs[di];
        }
        else
        {
            var sa = _sblkAddrs[s - _iblockSblks];
            if (sa < 0) return null;
            var sb = SuperBlock(sa, info.Ndblks, info.DblkNelmts);
            dblk = sb.Dblks[dblkInSblk];
            if (dblk >= 0 && paged && sb.Bitmaps is { } bm)
            {
                var page = elem / _pageNelmts;
                var bit = dblkInSblk * sb.PageInitSize * 8 + page;
                if ((bm[bit / 8] & (0x80 >> (int)(bit % 8))) == 0) return null;
            }
        }
        if (dblk < 0) return null;
        var o = _core.SizeOfOffsets;
        var prefix = 4 + 1 + 1 + o + _arrOff;
        long at;
        if (paged)
        {
            var page = elem / _pageNelmts;
            at = dblk + prefix + 4 + page * ((long)_pageNelmts * _elemSize + 4) + (elem % _pageNelmts) * _elemSize;
        }
        else at = dblk + prefix + elem * _elemSize;
        return H5IndexEntries.Parse(_core.Buf(at, _elemSize), _client, _elemSize, o);
    }

    private (long[] Dblks, byte[]? Bitmaps, int PageInitSize) SuperBlock(long address, long ndblks, long dblkNelmts)
    {
        lock (_gate)
            if (_superBlocks.TryGetValue(address, out var c)) return c;
        var o = _core.SizeOfOffsets;
        var paged = dblkNelmts > _pageNelmts;
        var pageInit = paged ? (int)((dblkNelmts / _pageNelmts + 7) / 8) : 0;
        var bitmapBytes = (int)(ndblks * pageInit);
        var b = _core.Buf(address, 4 + 1 + 1 + o + _arrOff + bitmapBytes + (int)ndblks * o);
        b.Signature("EASB");
        b.Skip(2);
        b.Offset();
        b.UN(_arrOff);
        var bitmaps = paged ? b.Bytes(bitmapBytes) : null;
        var d = new long[ndblks];
        for (var i = 0; i < ndblks; i++) d[i] = b.Offset();
        var r = (d, bitmaps, pageInit);
        lock (_gate) _superBlocks[address] = r;
        return r;
    }
}
