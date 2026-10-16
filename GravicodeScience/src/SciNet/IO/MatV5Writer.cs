using System.Buffers.Binary;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.IO;

/// <summary>Level 5 MAT-file writer (little-endian, optionally zlib-compressed per variable).</summary>
internal static class MatV5Writer
{
    public static void Write(Stream output, IEnumerable<KeyValuePair<string, MatValue>> variables, bool compress)
    {
        var header = new byte[128];
        var text = $"MATLAB 5.0 MAT-file Platform: {(OperatingSystem.IsWindows() ? "nt" : "posix")}, Created on: {DateTime.Now.ToString("ddd MMM d HH:mm:ss yyyy", System.Globalization.CultureInfo.InvariantCulture)}";
        Encoding.ASCII.GetBytes(text.PadRight(116)[..116]).CopyTo(header, 0);
        // bytes 116..123: subsystem data offset, all zero = none
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(124), 0x0100);
        header[126] = (byte)'I';
        header[127] = (byte)'M';
        output.Write(header);

        Span<byte> tag = stackalloc byte[8];
        foreach (var (name, value) in variables)
        {
            MatFile.ValidateName(name);
            var ms = new MemoryStream();
            WriteMatrix(ms, value, name, top: true);
            if (!compress)
            {
                ms.Position = 0;
                ms.CopyTo(output);
                continue;
            }
            var z = new MemoryStream();
            using (var zs = new ZLibStream(z, CompressionLevel.Optimal, leaveOpen: true))
            {
                ms.Position = 0;
                ms.CopyTo(zs);
            }
            BinaryPrimitives.WriteUInt32LittleEndian(tag, MatDataType.Compressed);
            BinaryPrimitives.WriteUInt32LittleEndian(tag[4..], checked((uint)z.Length));
            output.Write(tag);
            z.Position = 0;
            z.CopyTo(output);
        }
    }

    private static void Pad(Stream s)
    {
        var r = (int)(s.Position % 8);
        if (r != 0) s.Write(new byte[8 - r]);
    }

    /// <summary>Writes a data sub-element, using the 8-byte small form when the payload fits in 4 bytes.</summary>
    private static void Element(Stream s, int type, ReadOnlySpan<byte> data)
    {
        Span<byte> tag = stackalloc byte[8];
        if (data.Length <= 4 && data.Length > 0)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(tag, (uint)(type | (data.Length << 16)));
            tag[4..].Clear();
            data.CopyTo(tag[4..]);
            s.Write(tag);
            return;
        }
        BinaryPrimitives.WriteUInt32LittleEndian(tag, (uint)type);
        BinaryPrimitives.WriteUInt32LittleEndian(tag[4..], (uint)data.Length);
        s.Write(tag);
        s.Write(data);
        Pad(s);
    }

    private static void Ints(Stream s, ReadOnlySpan<int> v) => Element(s, MatDataType.Int32, MemoryMarshal.AsBytes(v));

    private static void WriteMatrix(Stream s, MatValue value, string name, bool top)
    {
        var start = s.Position;
        Span<byte> tag = stackalloc byte[8];
        BinaryPrimitives.WriteUInt32LittleEndian(tag, MatDataType.Matrix);
        s.Write(tag); // length patched below

        var dims = value.Dimensions.ToArray();
        if (dims.Length == 0) dims = [1, 1];
        else if (dims.Length == 1) dims = [1, dims[0]];

        MatClass cls;
        uint flags = 0;
        var nzmax = 0;
        switch (value)
        {
            case MatArray a: cls = a.Class; if (a.IsComplex) flags |= 0x800; break;
            case MatLogical: cls = MatClass.UInt8; flags |= 0x200; break;
            case MatChar: cls = MatClass.Char; break;
            case MatSparse sp:
                cls = MatClass.Sparse;
                if (sp.IsComplex) flags |= 0x800;
                if (sp.IsLogical) flags |= 0x200;
                nzmax = Math.Max(1, sp.Matrix.Nnz);
                dims = [sp.Matrix.Rows, sp.Matrix.Columns];
                break;
            case MatCell: cls = MatClass.Cell; break;
            case MatStruct st: cls = st.IsObject ? MatClass.Object : MatClass.Struct; break;
            case MatUnsupported u: throw new NotSupportedException($"Cannot write variable '{name}': {u.Message}");
            default: throw new NotSupportedException($"Cannot write a {value.GetType().Name}.");
        }
        if (top && value.IsGlobal) flags |= 0x400;
        Span<uint> fl = [flags | (uint)cls, (uint)nzmax];
        Element(s, MatDataType.UInt32, MemoryMarshal.AsBytes(fl));
        Ints(s, dims);
        var nameBytes = Encoding.Latin1.GetBytes(name);
        if (nameBytes.Length == 0)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(tag, MatDataType.Int8);
            BinaryPrimitives.WriteUInt32LittleEndian(tag[4..], 0);
            s.Write(tag);
        }
        else Element(s, MatDataType.Int8, nameBytes);

        switch (value)
        {
            case MatArray a:
                Numeric(s, a.Real, dims, a.Class);
                if (a.Imaginary is { } im) Numeric(s, im, dims, a.Class);
                break;
            case MatLogical l:
            {
                var cm = MatLayout.RowToColumnMajor(l.Values.AsContiguous().AsSpan(), dims);
                var bytes = new byte[cm.Length];
                for (var i = 0; i < cm.Length; i++) bytes[i] = cm[i] != 0 ? (byte)1 : (byte)0;
                Element(s, MatDataType.UInt8, bytes);
                break;
            }
            case MatChar c:
            {
                var cm = MatLayout.RowToColumnMajor(c.Chars.ToArray(), dims);
                // UTF-16 code units, as MATLAB's own char storage. Tagged miUTF16 rather than
                // miUINT16: both are the same bytes to MATLAB, but SciPy decodes miUINT16 text as
                // ASCII and turns every non-ASCII character into U+FFFD.
                Element(s, MatDataType.Utf16, MemoryMarshal.AsBytes(cm.AsSpan()));
                break;
            }
            case MatSparse sp:
            {
                var m = sp.Matrix;
                Ints(s, m.Indices);
                Ints(s, m.Indptr);
                // Logical values go out one byte each as miUINT8 (what SciPy writes, and what it
                // needs to come back as bool rather than float).
                if (sp.IsLogical) Element(s, MatDataType.UInt8, m.Data.ToArray().Select(v => v != 0 ? (byte)1 : (byte)0).ToArray());
                else Element(s, MatDataType.Double, MemoryMarshal.AsBytes(m.Data));
                if (sp.IsComplex) Element(s, MatDataType.Double, MemoryMarshal.AsBytes(sp.Imaginary));
                break;
            }
            case MatCell cell:
                foreach (var item in MatLayout.RowToColumnMajor(cell.ToArray(), dims)) WriteMatrix(s, item, "", top: false);
                break;
            case MatStruct st:
            {
                if (st.IsObject) Element(s, MatDataType.Int8, Encoding.Latin1.GetBytes(st.ObjectClassName!));
                var fields = st.FieldNames;
                foreach (var f in fields)
                    if (Encoding.Latin1.GetByteCount(f) > 63) throw new ArgumentException($"Field name '{f}' is longer than MATLAB's 63-character limit.");
                var len = fields.Count == 0 ? 1 : fields.Max(f => Encoding.Latin1.GetByteCount(f)) + 1;
                Ints(s, [len]);
                var names = new byte[fields.Count * len];
                for (var f = 0; f < fields.Count; f++) Encoding.Latin1.GetBytes(fields[f], names.AsSpan(f * len));
                if (names.Length == 0)
                {
                    BinaryPrimitives.WriteUInt32LittleEndian(tag, MatDataType.Int8);
                    BinaryPrimitives.WriteUInt32LittleEndian(tag[4..], 0);
                    s.Write(tag);
                }
                else Element(s, MatDataType.Int8, names);
                var order = MatLayout.ColumnMajorOrder(dims);
                foreach (var e in order)
                {
                    var row = st.RawElement(e);
                    for (var f = 0; f < fields.Count; f++) WriteMatrix(s, row[f], "", top: false);
                }
                break;
            }
        }

        var end = s.Position;
        s.Position = start + 4;
        Span<byte> len4 = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(len4, checked((uint)(end - start - 8)));
        s.Write(len4);
        s.Position = end;
    }

    /// <summary>Writes numeric data in its own class's storage type, column-major.</summary>
    private static void Numeric(Stream s, NdArray data, int[] dims, MatClass cls)
    {
        var cm = MatLayout.RowToColumnMajor(data.AsContiguous().AsSpan(), dims);
        switch (cls)
        {
            case MatClass.Double: Element(s, MatDataType.Double, MemoryMarshal.AsBytes(cm.AsSpan())); return;
            case MatClass.Single: Element(s, MatDataType.Single, MemoryMarshal.AsBytes(cm.Select(v => (float)v).ToArray().AsSpan())); return;
            case MatClass.Int8: Element(s, MatDataType.Int8, MemoryMarshal.AsBytes(Convert(cm, sbyte.MinValue, sbyte.MaxValue, v => (sbyte)v).AsSpan())); return;
            case MatClass.UInt8: Element(s, MatDataType.UInt8, Convert(cm, byte.MinValue, byte.MaxValue, v => (byte)v)); return;
            case MatClass.Int16: Element(s, MatDataType.Int16, MemoryMarshal.AsBytes(Convert(cm, short.MinValue, short.MaxValue, v => (short)v).AsSpan())); return;
            case MatClass.UInt16: Element(s, MatDataType.UInt16, MemoryMarshal.AsBytes(Convert(cm, ushort.MinValue, ushort.MaxValue, v => (ushort)v).AsSpan())); return;
            case MatClass.Int32: Element(s, MatDataType.Int32, MemoryMarshal.AsBytes(Convert(cm, int.MinValue, int.MaxValue, v => (int)v).AsSpan())); return;
            case MatClass.UInt32: Element(s, MatDataType.UInt32, MemoryMarshal.AsBytes(Convert(cm, uint.MinValue, uint.MaxValue, v => (uint)v).AsSpan())); return;
            case MatClass.Int64: Element(s, MatDataType.Int64, MemoryMarshal.AsBytes(Convert(cm, -9223372036854775808.0, 9223372036854775807.0, v => (long)v, upperExclusive: true).AsSpan())); return;
            case MatClass.UInt64: Element(s, MatDataType.UInt64, MemoryMarshal.AsBytes(Convert(cm, 0, 18446744073709551615.0, v => (ulong)v, upperExclusive: true).AsSpan())); return;
            default: throw new NotSupportedException($"{cls} is not numeric.");
        }
    }

    /// <summary>
    /// Narrows doubles to an integer class, refusing values that would not survive: a silent
    /// wrap-around or truncation writes a plausible file with the wrong numbers.
    /// </summary>
    private static T[] Convert<T>(double[] v, double min, double max, Func<double, T> cast, bool upperExclusive = false)
    {
        var r = new T[v.Length];
        for (var i = 0; i < v.Length; i++)
        {
            var x = v[i];
            if (x != Math.Floor(x) || x < min || (upperExclusive ? x >= max : x > max))
                throw new ArgumentException($"Value {x} cannot be stored exactly in the integer class (range {min}..{max}).");
            r[i] = cast(x);
        }
        return r;
    }
}
