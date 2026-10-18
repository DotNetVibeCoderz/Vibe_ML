using System.Runtime.InteropServices;
using System.Text;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.IO;

/// <summary>
/// MATLAB v7.3 MAT-file writer: an HDF5 file (written by <see cref="Hdf5Writer"/>) behind a
/// 512-byte user block holding the MAT header, in the layout <see cref="MatV73Reader"/> documents
/// and MATLAB itself writes and reads.
/// </summary>
/// <remarks>
/// Every variable carries <c>MATLAB_class</c> (a NUL-terminated fixed string sized to the name, as
/// MATLAB writes it). Data goes out with the dimensions reversed, which is MATLAB's column-major
/// order; char as UTF-16 code units in uint16 with <c>MATLAB_int_decode</c> = 2; logical as uint8;
/// complex as a <c>{real, imag}</c> compound of the class's type; an empty array as a uint64 vector
/// of its dimensions with <c>MATLAB_empty</c> = 1; sparse as a group (<c>data</c>, <c>ir</c>,
/// <c>jc</c>, <c>MATLAB_sparse</c> = rows); a struct as a group with <c>MATLAB_fields</c> (a
/// variable-length array of one-character strings per field) holding one member per field, or —
/// for a struct array — one object-reference dataset per field; a cell as an object-reference
/// dataset. Cell and struct-array elements live in <c>/#refs#</c>. With compression, numeric, char
/// and logical datasets of at least 4 KiB are deflated (smaller ones would grow: a chunk index
/// node alone is about 3 KiB).
/// </remarks>
internal static class MatV73Writer
{
    private const int CompressThreshold = 4096;

    public static void Write(Stream output, IEnumerable<KeyValuePair<string, MatValue>> variables, bool compress)
    {
        var w = new Hdf5Writer(userBlockSize: 512) { UserBlock = Header() };
        var st = new State(w, compress);
        foreach (var (name, value) in variables)
        {
            MatFile.ValidateName(name);
            var node = st.Put(w.Root, name, value);
            if (value.IsGlobal) node.SetAttribute("MATLAB_global", H5WData.Scalar<byte>(1));
        }
        w.Save(output);
    }

    /// <summary>The 128-byte MAT header at the start of the user block (version 0x0200, little-endian marker).</summary>
    private static byte[] Header()
    {
        var b = new byte[128];
        var text = $"MATLAB 7.3 MAT-file, Platform: {(OperatingSystem.IsWindows() ? "PCWIN64" : "GLNXA64")}, Created on: {DateTime.Now.ToString("ddd MMM d HH:mm:ss yyyy", System.Globalization.CultureInfo.InvariantCulture)} HDF5 schema 1.00 .";
        Encoding.ASCII.GetBytes(text.PadRight(116)[..116]).CopyTo(b, 0);
        // bytes 116..123: subsystem data offset (none)
        b[124] = 0x00;
        b[125] = 0x02;
        b[126] = (byte)'I';
        b[127] = (byte)'M';
        return b;
    }

    private sealed class State(Hdf5Writer w, bool compress)
    {
        private Hdf5WriterGroup? _refs;
        private int _next;

        private Hdf5WriterGroup Refs => _refs ??= w.Root.CreateGroup("#refs#");

        /// <summary>a, b, ..., z, A, ..., Z, ba, bb, ...: MATLAB's naming of /#refs# members.</summary>
        private string NextName()
        {
            const string digits = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
            var n = _next++;
            var sb = new StringBuilder();
            do
            {
                sb.Insert(0, digits[n % digits.Length]);
                n /= digits.Length;
            } while (n > 0);
            return sb.ToString();
        }

        private Hdf5WriterNode Element(MatValue v) => Put(Refs, NextName(), v);

        private static int[] Dims(MatValue v)
        {
            var d = v.Dimensions.ToArray();
            return d.Length switch { 0 => [1, 1], 1 => [1, d[0]], _ => d };
        }

        private static long[] Reversed(int[] dims) => dims.Reverse().Select(x => (long)x).ToArray();

        private static void Class(Hdf5WriterNode n, string cls) => n.SetAttribute("MATLAB_class", H5WData.FixedString(cls));

        private int Level(H5WData d) => compress && d.ByteCount >= CompressThreshold && d.Shape.Length > 0 ? 3 : 0;

        private Hdf5WriterDataset Dataset(Hdf5WriterGroup parent, string name, H5WData data)
            => parent.CreateDataset(name, data, compressionLevel: Level(data));

        public Hdf5WriterNode Put(Hdf5WriterGroup parent, string name, MatValue value)
        {
            var dims = Dims(value);
            switch (value)
            {
                case MatUnsupported u:
                    throw new NotSupportedException($"Cannot write variable '{name}': {u.Message}");
                case MatSparse sp:
                    return Sparse(parent, name, sp);
            }
            if (value.Count == 0) return Empty(parent, name, dims, value);
            switch (value)
            {
                case MatArray a:
                {
                    var t = TypeOf(a.Class);
                    var re = Typed(MatLayout.RowToColumnMajor(a.Real.AsContiguous().AsSpan(), dims), a.Class);
                    H5WData data;
                    if (a.Imaginary is { } im)
                    {
                        var ia = Typed(MatLayout.RowToColumnMajor(im.AsContiguous().AsSpan(), dims), a.Class);
                        data = H5WData.Raw(Interleave(re, ia, t.Size), H5WType.Compound(("real", t), ("imag", t)), Reversed(dims));
                    }
                    else data = H5WData.Numeric(re, t, Reversed(dims));
                    var d = Dataset(parent, name, data);
                    Class(d, a.ClassName);
                    return d;
                }
                case MatLogical l:
                {
                    var cm = MatLayout.RowToColumnMajor(l.Values.AsContiguous().AsSpan(), dims);
                    var b = new byte[cm.Length];
                    for (var i = 0; i < b.Length; i++) b[i] = cm[i] != 0 ? (byte)1 : (byte)0;
                    var d = Dataset(parent, name, H5WData.Numeric(b, H5WType.U8, Reversed(dims)));
                    Class(d, "logical");
                    return d;
                }
                case MatChar c:
                {
                    var cm = MatLayout.RowToColumnMajor(c.Chars.ToArray(), dims);
                    var u = MemoryMarshal.Cast<char, ushort>(cm.AsSpan()).ToArray();
                    var d = Dataset(parent, name, H5WData.Numeric(u, H5WType.U16, Reversed(dims)));
                    Class(d, "char");
                    d.SetAttribute("MATLAB_int_decode", H5WData.Scalar(2));
                    return d;
                }
                case MatCell cell:
                {
                    var items = MatLayout.RowToColumnMajor(cell.ToArray(), dims);
                    var refs = items.Select(Element).ToArray();
                    var d = parent.CreateDataset(name, H5WData.References(refs, Reversed(dims)));
                    Class(d, "cell");
                    return d;
                }
                case MatStruct s:
                {
                    var g = parent.CreateGroup(name);
                    Class(g, s.ClassName);
                    Fields(g, s.FieldNames);
                    if (s.Count == 1)
                    {
                        var row = s.RawElement(0);
                        for (var f = 0; f < s.FieldNames.Count; f++) Put(g, s.FieldNames[f], row[f]);
                        return g;
                    }
                    // A struct array: one reference dataset per field (no MATLAB_class of its own), column-major over the elements.
                    var order = MatLayout.ColumnMajorOrder(dims);
                    for (var f = 0; f < s.FieldNames.Count; f++)
                    {
                        var refs = order.Select(e => Element(s.RawElement(e)[f])).ToArray();
                        g.CreateDataset(s.FieldNames[f], H5WData.References(refs, Reversed(dims)));
                    }
                    return g;
                }
                default:
                    throw new NotSupportedException($"Cannot write a {value.GetType().Name}.");
            }
        }

        private static void Fields(Hdf5WriterNode n, IReadOnlyList<string> fields)
        {
            foreach (var f in fields)
                if (f.Length > 63) throw new ArgumentException($"Field name '{f}' is longer than MATLAB's 63-character limit.");
            var seqs = fields.Select(f => Encoding.Latin1.GetBytes(f)).ToArray();
            n.SetAttribute("MATLAB_fields", H5WData.VlenSequences(H5WType.FixedString(1, false), seqs, [seqs.Length]));
        }

        private Hdf5WriterNode Empty(Hdf5WriterGroup parent, string name, int[] dims, MatValue value)
        {
            var d = parent.CreateDataset(name, H5WData.Numeric(dims.Select(x => (ulong)x).ToArray(), H5WType.U64, [dims.Length]));
            Class(d, value.ClassName);
            d.SetAttribute("MATLAB_empty", H5WData.Scalar<byte>(1));
            if (value is MatStruct s) Fields(d, s.FieldNames);
            return d;
        }

        private Hdf5WriterNode Sparse(Hdf5WriterGroup parent, string name, MatSparse sp)
        {
            var m = sp.Matrix;
            var g = parent.CreateGroup(name);
            Class(g, sp.IsLogical ? "logical" : "double");
            g.SetAttribute("MATLAB_sparse", H5WData.Scalar((ulong)m.Rows));
            var nnz = m.Indptr[m.Columns];
            if (nnz > 0)
            {
                H5WData data;
                if (sp.IsLogical)
                    data = H5WData.Numeric(m.Data[..nnz].ToArray().Select(v => v != 0 ? (byte)1 : (byte)0).ToArray(), H5WType.U8, [nnz]);
                else if (sp.IsComplex)
                    data = H5WData.Raw(Interleave(m.Data[..nnz].ToArray(), sp.Imaginary[..nnz].ToArray(), 8), H5WType.Compound(("real", H5WType.F64), ("imag", H5WType.F64)), [nnz]);
                else data = H5WData.Numeric(m.Data[..nnz].ToArray(), H5WType.F64, [nnz]);
                Dataset(g, "data", data);
                Dataset(g, "ir", H5WData.Numeric(m.Indices[..nnz].ToArray().Select(x => (ulong)x).ToArray(), H5WType.U64, [nnz]));
            }
            Dataset(g, "jc", H5WData.Numeric(m.Indptr.ToArray().Select(x => (ulong)x).ToArray(), H5WType.U64, [m.Columns + 1]));
            return g;
        }
    }

    private static H5WType TypeOf(MatClass c) => c switch
    {
        MatClass.Double => H5WType.F64,
        MatClass.Single => H5WType.F32,
        MatClass.Int8 => H5WType.I8,
        MatClass.UInt8 => H5WType.U8,
        MatClass.Int16 => H5WType.I16,
        MatClass.UInt16 => H5WType.U16,
        MatClass.Int32 => H5WType.I32,
        MatClass.UInt32 => H5WType.U32,
        MatClass.Int64 => H5WType.I64,
        MatClass.UInt64 => H5WType.U64,
        _ => throw new NotSupportedException($"{c} is not numeric."),
    };

    /// <summary>Column-major doubles in the class's storage type; integer classes refuse values they cannot hold exactly.</summary>
    private static Array Typed(double[] cm, MatClass cls) => cls switch
    {
        MatClass.Double => cm,
        MatClass.Single => cm.Select(v => (float)v).ToArray(),
        MatClass.Int8 => MatV5Writer.Convert(cm, sbyte.MinValue, sbyte.MaxValue, v => (sbyte)v),
        MatClass.UInt8 => MatV5Writer.Convert(cm, byte.MinValue, byte.MaxValue, v => (byte)v),
        MatClass.Int16 => MatV5Writer.Convert(cm, short.MinValue, short.MaxValue, v => (short)v),
        MatClass.UInt16 => MatV5Writer.Convert(cm, ushort.MinValue, ushort.MaxValue, v => (ushort)v),
        MatClass.Int32 => MatV5Writer.Convert(cm, int.MinValue, int.MaxValue, v => (int)v),
        MatClass.UInt32 => MatV5Writer.Convert(cm, uint.MinValue, uint.MaxValue, v => (uint)v),
        MatClass.Int64 => MatV5Writer.Convert(cm, -9223372036854775808.0, 9223372036854775807.0, v => (long)v, upperExclusive: true),
        MatClass.UInt64 => MatV5Writer.Convert(cm, 0, 18446744073709551615.0, v => (ulong)v, upperExclusive: true),
        _ => throw new NotSupportedException($"{cls} is not numeric."),
    };

    /// <summary>Real and imaginary arrays of one element type, interleaved record by record.</summary>
    private static byte[] Interleave(Array re, Array im, int size)
    {
        var n = re.Length;
        var a = MemoryMarshal.CreateReadOnlySpan(ref MemoryMarshal.GetArrayDataReference(re), n * size);
        var b = MemoryMarshal.CreateReadOnlySpan(ref MemoryMarshal.GetArrayDataReference(im), n * size);
        var r = new byte[2 * n * size];
        for (var i = 0; i < n; i++)
        {
            a.Slice(i * size, size).CopyTo(r.AsSpan(2 * i * size));
            b.Slice(i * size, size).CopyTo(r.AsSpan((2 * i + 1) * size));
        }
        return r;
    }
}
