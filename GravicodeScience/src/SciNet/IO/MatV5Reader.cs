using System.Buffers.Binary;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Sparse;

namespace Gravicode.Science.SciNet.IO;

/// <summary>MAT-file level 5 data types (<c>miINT8</c> ... <c>miUTF32</c>).</summary>
internal static class MatDataType
{
    public const int Int8 = 1, UInt8 = 2, Int16 = 3, UInt16 = 4, Int32 = 5, UInt32 = 6, Single = 7, Double = 9,
        Int64 = 12, UInt64 = 13, Matrix = 14, Compressed = 15, Utf8 = 16, Utf16 = 17, Utf32 = 18;

    public static int Size(int type) => type switch
    {
        Int8 or UInt8 or Utf8 => 1,
        Int16 or UInt16 or Utf16 => 2,
        Int32 or UInt32 or Single or Utf32 => 4,
        Double or Int64 or UInt64 => 8,
        _ => throw new InvalidDataException($"Malformed MAT file: data type {type} is not a numeric type."),
    };
}

/// <summary>
/// Level 5 MAT-file reader (MATLAB v5, v6 and v7 — everything before the HDF5-based v7.3).
/// The file is parsed from one in-memory buffer; compressed variables are inflated one at a time.
/// </summary>
internal sealed class MatV5Reader
{
    private readonly bool _swap;

    private MatV5Reader(bool swap) => _swap = swap;

    private readonly record struct Element(int Type, int Offset, int Length);

    /// <summary>Reads the 128-byte header; returns the reader configured for the file's byte order.</summary>
    public static MatV5Reader Open(byte[] file)
    {
        if (file.Length < 128) throw new InvalidDataException("Not a MAT file: shorter than the 128-byte level 5 header.");
        var swap = file[126] == (byte)'M' && file[127] == (byte)'I';
        if (!swap && !(file[126] == (byte)'I' && file[127] == (byte)'M'))
            throw new InvalidDataException("Not a level 5 MAT file: missing the 'IM'/'MI' endian indicator.");
        return new MatV5Reader(swap);
    }

    /// <summary>All (or the selected) variables in file order.</summary>
    public OrderedDictionary<string, MatValue> ReadAll(byte[] file, HashSet<string>? wanted)
    {
        var result = new OrderedDictionary<string, MatValue>(StringComparer.Ordinal);
        foreach (var (buffer, el) in TopLevel(file))
        {
            if (wanted is not null && wanted.Count == result.Count) break;
            byte[] buf;
            Element m;
            if (el.Type == MatDataType.Compressed)
            {
                if (wanted is not null)
                {
                    var head = Inflate(buffer, el, 4096, out _);
                    var probe = new Element(MatDataType.Matrix, 8, head.Length - 8);
                    if (!wanted.Contains(PeekName(head, probe))) continue;
                }
                buf = Inflate(buffer, el, int.MaxValue, out var inner);
                m = inner;
            }
            else
            {
                buf = buffer;
                m = el;
                if (wanted is not null && !wanted.Contains(PeekName(buf, m))) continue;
            }
            var (value, name) = ReadMatrix(buf, m);
            if (name.Length == 0) name = "__function_workspace__";
            if (wanted is null || wanted.Contains(name)) result[name] = value;
        }
        return result;
    }

    /// <summary>Variable headers (name, MATLAB dims, SciPy-style class) without reading the data.</summary>
    public List<MatVariableInfo> List(byte[] file)
    {
        var list = new List<MatVariableInfo>();
        foreach (var (buffer, el) in TopLevel(file))
        {
            byte[] buf = buffer;
            var m = el;
            if (el.Type == MatDataType.Compressed)
            {
                buf = Inflate(buffer, el, 4096, out _);
                m = new Element(MatDataType.Matrix, 8, buf.Length - 8);
            }
            list.Add(Header(buf, m));
        }
        return list;
    }

    private IEnumerable<(byte[] Buffer, Element El)> TopLevel(byte[] file)
    {
        var pos = 128;
        while (pos + 8 <= file.Length)
        {
            var type = (int)U32(file, pos);
            var len = U32(file, pos + 4);
            var data = pos + 8;
            if (len > (uint)(file.Length - data)) throw new InvalidDataException($"Malformed MAT file: element at byte {pos} claims {len} bytes but only {file.Length - data} remain.");
            pos = data + (int)len;
            if (type == MatDataType.Matrix || type == MatDataType.Compressed)
                yield return (file, new Element(type, data, (int)len));
            else
                throw new InvalidDataException($"Malformed MAT file: expected a matrix or compressed element at byte {data - 8}, found type {type}.");
        }
    }

    /// <summary>Inflates a miCOMPRESSED element; with a cap, returns only the leading bytes (for headers).</summary>
    private byte[] Inflate(byte[] file, Element el, int cap, out Element inner)
    {
        using var z = new ZLibStream(new MemoryStream(file, el.Offset, el.Length, writable: false), CompressionMode.Decompress);
        var tag = new byte[8];
        try { z.ReadExactly(tag); }
        catch (Exception e) when (e is EndOfStreamException or InvalidDataException)
        {
            throw new InvalidDataException("Malformed MAT file: compressed variable is truncated or corrupt.", e);
        }
        var type = (int)U32(tag, 0);
        var len = U32(tag, 4);
        if (type != MatDataType.Matrix) throw new InvalidDataException($"Malformed MAT file: compressed element holds type {type}, not a matrix.");
        if (len > int.MaxValue - 8) throw new InvalidDataException("Variable larger than 2 GB.");
        var want = (int)Math.Min(len, (uint)Math.Max(0, cap - 8));
        var buf = new byte[8 + want];
        tag.CopyTo(buf, 0);
        var got = 0;
        try
        {
            while (got < want)
            {
                var r = z.Read(buf, 8 + got, want - got);
                if (r == 0) break;
                got += r;
            }
        }
        catch (InvalidDataException e)
        {
            throw new InvalidDataException("Malformed MAT file: zlib data is corrupt.", e);
        }
        if (got < want)
        {
            if (cap == int.MaxValue) throw new InvalidDataException($"Malformed MAT file: compressed variable inflated to {got} of {len} bytes.");
            Array.Resize(ref buf, 8 + got);
        }
        inner = new Element(type, 8, buf.Length - 8);
        return buf;
    }

    // ------------------------------------------------------------------ primitives

    private uint U32(byte[] b, int p) => _swap ? BinaryPrimitives.ReadUInt32BigEndian(b.AsSpan(p, 4)) : BinaryPrimitives.ReadUInt32LittleEndian(b.AsSpan(p, 4));

    /// <summary>Reads a sub-element tag at <paramref name="pos"/> (either form) and advances past its 8-byte-aligned data.</summary>
    private Element Next(byte[] b, ref int pos, int end)
    {
        if (pos + 8 > end) throw new InvalidDataException($"Malformed MAT file: element header at byte {pos} runs past the end of its parent.");
        var tag = U32(b, pos);
        if ((tag >> 16) != 0)
        {
            // Small data element: type and byte count share the first word, data sits in the second.
            var t = (int)(tag & 0xFFFF);
            var n = (int)(tag >> 16);
            if (n > 4) throw new InvalidDataException($"Malformed MAT file: small element at byte {pos} claims {n} bytes.");
            var e = new Element(t, pos + 4, n);
            pos += 8;
            return e;
        }
        var len = U32(b, pos + 4);
        if (len > (uint)(end - pos - 8)) throw new InvalidDataException($"Malformed MAT file: element at byte {pos} claims {len} bytes but its parent ends at {end}.");
        var el = new Element((int)tag, pos + 8, (int)len);
        pos += 8 + (int)((len + 7) & ~7u);
        if (pos > end) pos = end; // the last element's padding may be omitted
        return el;
    }

    private int[] ReadInts(byte[] b, Element e, int? count = null)
    {
        var size = MatDataType.Size(e.Type);
        var n = e.Length / size;
        if (count is { } c)
        {
            if (n < c) throw new InvalidDataException($"Malformed MAT file: expected {c} integers, found {n}.");
            n = c;
        }
        var r = new int[n];
        var s = b.AsSpan(e.Offset, n * size);
        for (var i = 0; i < n; i++)
        {
            var v = ReadOne(s.Slice(i * size, size), e.Type);
            if (v < int.MinValue || v > int.MaxValue || v != Math.Floor(v)) throw new InvalidDataException($"Malformed MAT file: index value {v} is not a 32-bit integer.");
            r[i] = (int)v;
        }
        return r;
    }

    private double ReadOne(ReadOnlySpan<byte> s, int type) => type switch
    {
        MatDataType.Int8 => (sbyte)s[0],
        MatDataType.UInt8 or MatDataType.Utf8 => s[0],
        MatDataType.Int16 => _swap ? BinaryPrimitives.ReadInt16BigEndian(s) : BinaryPrimitives.ReadInt16LittleEndian(s),
        MatDataType.UInt16 or MatDataType.Utf16 => _swap ? BinaryPrimitives.ReadUInt16BigEndian(s) : BinaryPrimitives.ReadUInt16LittleEndian(s),
        MatDataType.Int32 => _swap ? BinaryPrimitives.ReadInt32BigEndian(s) : BinaryPrimitives.ReadInt32LittleEndian(s),
        MatDataType.UInt32 or MatDataType.Utf32 => _swap ? BinaryPrimitives.ReadUInt32BigEndian(s) : BinaryPrimitives.ReadUInt32LittleEndian(s),
        MatDataType.Single => _swap ? BinaryPrimitives.ReadSingleBigEndian(s) : BinaryPrimitives.ReadSingleLittleEndian(s),
        MatDataType.Double => _swap ? BinaryPrimitives.ReadDoubleBigEndian(s) : BinaryPrimitives.ReadDoubleLittleEndian(s),
        MatDataType.Int64 => _swap ? BinaryPrimitives.ReadInt64BigEndian(s) : BinaryPrimitives.ReadInt64LittleEndian(s),
        MatDataType.UInt64 => _swap ? BinaryPrimitives.ReadUInt64BigEndian(s) : BinaryPrimitives.ReadUInt64LittleEndian(s),
        _ => throw new InvalidDataException($"Malformed MAT file: data type {type} is not numeric."),
    };

    /// <summary>
    /// Numeric data in file (column-major) order, widened to double. MATLAB itself stores
    /// doubles in the smallest type that holds them exactly (integer-valued doubles as miUINT8,
    /// ...), so the storage type is independent of the array's class.
    /// </summary>
    private double[] ReadNumeric(byte[] b, Element e, int count)
    {
        var size = MatDataType.Size(e.Type);
        if (e.Length / size < count) throw new InvalidDataException($"Malformed MAT file: expected {count} values, element holds {e.Length / size}.");
        var r = new double[count];
        var s = b.AsSpan(e.Offset, count * size);
        if (!_swap)
        {
            switch (e.Type)
            {
                case MatDataType.Double: MemoryMarshal.Cast<byte, double>(s).CopyTo(r); return r;
                case MatDataType.Single: { var f = MemoryMarshal.Cast<byte, float>(s); for (var i = 0; i < count; i++) r[i] = f[i]; return r; }
                case MatDataType.Int8: for (var i = 0; i < count; i++) r[i] = (sbyte)s[i]; return r;
                case MatDataType.UInt8: case MatDataType.Utf8: for (var i = 0; i < count; i++) r[i] = s[i]; return r;
                case MatDataType.Int16: { var v = MemoryMarshal.Cast<byte, short>(s); for (var i = 0; i < count; i++) r[i] = v[i]; return r; }
                case MatDataType.UInt16: case MatDataType.Utf16: { var v = MemoryMarshal.Cast<byte, ushort>(s); for (var i = 0; i < count; i++) r[i] = v[i]; return r; }
                case MatDataType.Int32: { var v = MemoryMarshal.Cast<byte, int>(s); for (var i = 0; i < count; i++) r[i] = v[i]; return r; }
                case MatDataType.UInt32: case MatDataType.Utf32: { var v = MemoryMarshal.Cast<byte, uint>(s); for (var i = 0; i < count; i++) r[i] = v[i]; return r; }
                case MatDataType.Int64: { var v = MemoryMarshal.Cast<byte, long>(s); for (var i = 0; i < count; i++) r[i] = v[i]; return r; }
                case MatDataType.UInt64: { var v = MemoryMarshal.Cast<byte, ulong>(s); for (var i = 0; i < count; i++) r[i] = v[i]; return r; }
            }
        }
        for (var i = 0; i < count; i++) r[i] = ReadOne(s.Slice(i * size, size), e.Type);
        return r;
    }

    /// <summary>Real numeric data converted straight to row-major; doubles in native order skip the intermediate copy.</summary>
    private unsafe double[] ReadRowMajor(byte[] b, Element e, int[] dims)
    {
        var count = MatLayout.Product(dims);
        if (!_swap && e.Type == MatDataType.Double && !MatLayout.OrderIsTrivial(dims))
        {
            if (e.Length / 8 < count) throw new InvalidDataException($"Malformed MAT file: expected {count} values, element holds {e.Length / 8}.");
            var r = new double[count];
            if (count == 0) return r;
            // Element offsets are 8-byte aligned within the file buffer, but the buffer itself
            // need not be; x64 handles the unaligned loads.
            fixed (byte* p = &b[e.Offset])
            fixed (double* d = r)
                MatLayout.ReverseAxes((double*)p, d, Enumerable.Reverse(dims).ToArray());
            return r;
        }
        return MatLayout.ColumnToRowMajor(ReadNumeric(b, e, count), dims);
    }

    private static string DecodeName(byte[] b, Element e)
    {
        var s = b.AsSpan(e.Offset, e.Length);
        var z = s.IndexOf((byte)0);
        if (z >= 0) s = s[..z];
        return e.Type == MatDataType.Utf8 ? Encoding.UTF8.GetString(s) : Encoding.Latin1.GetString(s);
    }

    private char[] DecodeChars(byte[] b, Element e, int count)
    {
        var s = b.AsSpan(e.Offset, e.Length);
        string text;
        switch (e.Type)
        {
            case MatDataType.Utf8:
                text = Encoding.UTF8.GetString(s);
                break;
            case MatDataType.UInt8:
            case MatDataType.Int8:
                text = Encoding.Latin1.GetString(s);
                break;
            case MatDataType.Utf16:
            case MatDataType.UInt16:
            case MatDataType.Int16:
            {
                var c = new char[e.Length / 2];
                for (var i = 0; i < c.Length; i++) c[i] = (char)(ushort)ReadOne(s.Slice(2 * i, 2), MatDataType.UInt16);
                text = new string(c);
                break;
            }
            case MatDataType.Utf32:
            case MatDataType.UInt32:
            case MatDataType.Int32:
            {
                var sb = new StringBuilder(e.Length / 4);
                for (var i = 0; i < e.Length / 4; i++) sb.Append(char.ConvertFromUtf32((int)ReadOne(s.Slice(4 * i, 4), MatDataType.UInt32)));
                text = sb.ToString();
                break;
            }
            default:
            {
                var v = ReadNumeric(b, e, e.Length / MatDataType.Size(e.Type));
                text = new string(v.Select(x => (char)(int)x).ToArray());
                break;
            }
        }
        if (text.Length < count) throw new InvalidDataException($"Malformed MAT file: character data holds {text.Length} characters, dimensions need {count}.");
        return text.AsSpan(0, count).ToArray();
    }

    // ------------------------------------------------------------------ matrices

    private string PeekName(byte[] b, Element m)
    {
        if (m.Length == 0) return "__function_workspace__";
        var pos = m.Offset;
        var end = m.Offset + m.Length;
        var flags = Next(b, ref pos, end);
        var cls = (MatClass)(U32(b, flags.Offset) & 0xFF);
        if (cls != MatClass.Opaque) Next(b, ref pos, end);
        var name = DecodeName(b, Next(b, ref pos, end));
        return name.Length == 0 ? "__function_workspace__" : name;
    }

    private MatVariableInfo Header(byte[] b, Element m)
    {
        if (m.Length == 0) return new MatVariableInfo("", [0, 0], "double");
        var pos = m.Offset;
        var end = m.Offset + m.Length;
        var flags = Next(b, ref pos, end);
        var word = U32(b, flags.Offset);
        var cls = (MatClass)(word & 0xFF);
        int[] dims = [];
        if (cls != MatClass.Opaque) dims = ReadInts(b, Next(b, ref pos, end));
        var name = DecodeName(b, Next(b, ref pos, end));
        if (name.Length == 0) name = "__function_workspace__";
        var logical = (word & 0x200) != 0;
        return new MatVariableInfo(name, dims, logical ? "logical" : MatLayout.ClassName(cls));
    }

    /// <summary>Decodes one miMATRIX element (the tag already consumed) into a value and its name.</summary>
    private (MatValue Value, string Name) ReadMatrix(byte[] b, Element m)
    {
        // MATLAB writes an empty miMATRIX (no sub-elements at all) for some empty cell entries.
        if (m.Length == 0) return (new MatArray([0, 0], NdArray.Zeros(0, 0), null, MatClass.Double), "");
        var pos = m.Offset;
        var end = m.Offset + m.Length;
        var flagsEl = Next(b, ref pos, end);
        if (flagsEl.Length < 8) throw new InvalidDataException("Malformed MAT file: array flags sub-element is shorter than 8 bytes.");
        var word = U32(b, flagsEl.Offset);
        var cls = (MatClass)(word & 0xFF);
        var isComplex = (word & 0x800) != 0;
        var isGlobal = (word & 0x400) != 0;
        var isLogical = (word & 0x200) != 0;

        if (cls == MatClass.Opaque)
        {
            var oname = DecodeName(b, Next(b, ref pos, end));
            return (new MatUnsupported([1, 1], cls, $"Variable '{oname}' is an opaque MATLAB object (classdef, string, table, ...); its MCOS serialisation is undocumented and is skipped, as SciPy does."), oname);
        }

        var dims = ReadInts(b, Next(b, ref pos, end));
        if (dims.Length < 2) throw new InvalidDataException($"Malformed MAT file: {dims.Length} dimensions (MATLAB arrays have at least 2).");
        foreach (var d in dims) if (d < 0) throw new InvalidDataException($"Malformed MAT file: negative dimension {d}.");
        var name = DecodeName(b, Next(b, ref pos, end));
        MatValue value;
        switch (cls)
        {
            case MatClass.Function:
                value = new MatUnsupported(dims, cls, $"Variable '{name}' is a MATLAB function handle; its workspace encoding is undocumented and it is skipped (SciPy reads it only as an opaque record).");
                break;
            case >= MatClass.Double and <= MatClass.UInt64:
            {
                var re = ReadRowMajor(b, Next(b, ref pos, end), dims);
                var shape = MatLayout.ShapeOf(dims);
                if (isLogical)
                {
                    value = new MatLogical(dims, new NdArray(re, shape));
                    break;
                }
                NdArray? im = null;
                if (isComplex) im = new NdArray(ReadRowMajor(b, Next(b, ref pos, end), dims), shape);
                value = new MatArray(dims, new NdArray(re, shape), im, cls);
                break;
            }
            case MatClass.Char:
            {
                var count = MatLayout.Product(dims);
                char[] chars = count == 0 ? [] : DecodeChars(b, Next(b, ref pos, end), count);
                value = new MatChar(dims, MatLayout.ColumnToRowMajor(chars, dims));
                break;
            }
            case MatClass.Sparse:
                value = ReadSparse(b, ref pos, end, dims, isComplex, isLogical);
                break;
            case MatClass.Cell:
            {
                var n = MatLayout.Product(dims);
                var items = new MatValue[n];
                for (var i = 0; i < n; i++) items[i] = ReadChild(b, ref pos, end);
                value = new MatCell(dims, MatLayout.ColumnToRowMajor(items, dims));
                break;
            }
            case MatClass.Struct:
            case MatClass.Object:
            {
                string? className = null;
                if (cls == MatClass.Object) className = DecodeName(b, Next(b, ref pos, end));
                value = ReadStruct(b, ref pos, end, dims, className);
                break;
            }
            default:
                throw new InvalidDataException($"Malformed MAT file: unknown array class {(int)cls} for variable '{name}'.");
        }
        value.IsGlobal = isGlobal;
        return (value, name);
    }

    private MatValue ReadChild(byte[] b, ref int pos, int end)
    {
        var el = Next(b, ref pos, end);
        if (el.Type != MatDataType.Matrix) throw new InvalidDataException($"Malformed MAT file: expected a matrix element inside a cell/struct, found type {el.Type}.");
        return ReadMatrix(b, el).Value;
    }

    private MatValue ReadStruct(byte[] b, ref int pos, int end, int[] dims, string? className)
    {
        var lenEl = Next(b, ref pos, end);
        var fieldLen = ReadInts(b, lenEl, 1)[0];
        var namesEl = Next(b, ref pos, end);
        if (fieldLen <= 0 && namesEl.Length > 0) throw new InvalidDataException("Malformed MAT file: field name length is not positive.");
        var nfields = fieldLen <= 0 ? 0 : namesEl.Length / fieldLen;
        var names = new string[nfields];
        for (var f = 0; f < nfields; f++) names[f] = DecodeName(b, new Element(namesEl.Type, namesEl.Offset + f * fieldLen, fieldLen));
        // MATLAB forbids duplicate field names but broken writers produce them; keep the data reachable.
        for (var f = 0; f < nfields; f++)
            for (var g = 0; g < f; g++)
                if (names[g] == names[f]) names[f] += $"_{f}";
        var n = MatLayout.Product(dims);
        var elements = new IReadOnlyList<MatValue>[n];
        for (var i = 0; i < n; i++)
        {
            var row = new MatValue[nfields];
            for (var f = 0; f < nfields; f++) row[f] = ReadChild(b, ref pos, end);
            elements[i] = row;
        }
        return new MatStruct(dims, names, MatLayout.ColumnToRowMajor(elements, dims), className);
    }

    private MatValue ReadSparse(byte[] b, ref int pos, int end, int[] dims, bool isComplex, bool isLogical)
    {
        if (dims.Length != 2) throw new InvalidDataException("Malformed MAT file: sparse matrix is not 2-D.");
        int rows = dims[0], cols = dims[1];
        var irEl = Next(b, ref pos, end);
        var jcEl = Next(b, ref pos, end);
        var jc = ReadInts(b, jcEl, cols + 1);
        var nnz = jc[cols];
        if (nnz < 0) throw new InvalidDataException("Malformed MAT file: negative non-zero count.");
        var ir = ReadInts(b, irEl, nnz);
        double[] pr;
        var prEl = pos < end ? Next(b, ref pos, end) : new Element(MatDataType.Double, pos, 0);
        if (isLogical && prEl.Length < (long)nnz * MatDataType.Size(prEl.Type) && prEl.Length >= nnz)
            // MATLAB tags logical sparse values miDOUBLE but writes one byte per value; the
            // byte count is the only way to tell.
            pr = ReadNumeric(b, prEl with { Type = MatDataType.UInt8 }, nnz);
        else
            pr = ReadNumeric(b, prEl, nnz);
        double[]? pi = null;
        if (isComplex) pi = ReadNumeric(b, Next(b, ref pos, end), nnz);
        var csc = new CscMatrix(pr, ir, jc, rows, cols, copy: false);
        if (!csc.HasCanonicalFormat)
        {
            if (pi is not null)
                throw new InvalidDataException("Malformed MAT file: complex sparse matrix with unsorted or duplicate row indices.");
            csc = MatConvert.Canonical(csc);
        }
        return new MatSparse(csc, pi, isLogical);
    }
}
