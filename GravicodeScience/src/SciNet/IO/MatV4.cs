using System.Buffers.Binary;
using System.Text;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Sparse;

namespace Gravicode.Science.SciNet.IO;

/// <summary>
/// Level 4 MAT files (MATLAB 4, still written by Octave and <c>save -v4</c>): a flat sequence of
/// 2-D matrices, each with a 20-byte header whose <c>MOPT</c> word encodes byte order, storage
/// precision and kind (full / text / sparse).
/// </summary>
internal static class MatV4
{
    private static readonly MatClass[] PrecisionClass = [MatClass.Double, MatClass.Single, MatClass.Int32, MatClass.Int16, MatClass.UInt16, MatClass.UInt8];
    private static readonly int[] PrecisionSize = [8, 4, 4, 2, 2, 1];

    private readonly record struct Header(bool BigEndian, int P, int T, int Rows, int Cols, bool Imag, string Name, int DataOffset, int End);

    private static Header ReadHeader(byte[] b, int pos)
    {
        if (pos + 20 > b.Length) throw new InvalidDataException($"Malformed v4 MAT file: truncated header at byte {pos}.");
        var le = BinaryPrimitives.ReadInt32LittleEndian(b.AsSpan(pos));
        bool big;
        int mopt;
        if (le is >= 0 and < 5000) { mopt = le; big = mopt / 1000 == 1; }
        else { mopt = BinaryPrimitives.ReadInt32BigEndian(b.AsSpan(pos)); big = true; }
        var m = mopt / 1000;
        var o = mopt / 100 % 10;
        var p = mopt / 10 % 10;
        var t = mopt % 10;
        if (mopt is < 0 or >= 5000 || o != 0 || p > 5 || t > 2) throw new InvalidDataException($"Malformed v4 MAT file: MOPT {mopt} at byte {pos}.");
        if (m > 1) throw new NotSupportedException($"v4 MAT file uses machine format {m} (VAX/Cray); only IEEE little/big-endian is supported.");
        if (big != (m == 1)) throw new InvalidDataException($"Malformed v4 MAT file: MOPT {mopt} disagrees with the byte order it was stored in.");
        int I(int k) => big ? BinaryPrimitives.ReadInt32BigEndian(b.AsSpan(pos + 4 * k)) : BinaryPrimitives.ReadInt32LittleEndian(b.AsSpan(pos + 4 * k));
        int rows = I(1), cols = I(2), imagf = I(3), namlen = I(4);
        if (rows < 0 || cols < 0 || namlen < 0 || pos + 20 + namlen > b.Length) throw new InvalidDataException($"Malformed v4 MAT file: bad header at byte {pos}.");
        var nameSpan = b.AsSpan(pos + 20, namlen);
        var z = nameSpan.IndexOf((byte)0);
        var name = Encoding.Latin1.GetString(z >= 0 ? nameSpan[..z] : nameSpan);
        var data = pos + 20 + namlen;
        var count = (long)rows * cols * (imagf != 0 ? 2 : 1);
        var end = data + count * PrecisionSize[p];
        if (end > b.Length) throw new InvalidDataException($"Malformed v4 MAT file: variable '{name}' runs past the end of the file.");
        return new Header(big, p, t, rows, cols, imagf != 0, name, data, (int)end);
    }

    private static double[] Values(byte[] b, Header h, int offset, int count)
    {
        var r = new double[count];
        var size = PrecisionSize[h.P];
        var s = b.AsSpan(offset, count * size);
        for (var i = 0; i < count; i++)
        {
            var x = s.Slice(i * size, size);
            r[i] = h.P switch
            {
                0 => h.BigEndian ? BinaryPrimitives.ReadDoubleBigEndian(x) : BinaryPrimitives.ReadDoubleLittleEndian(x),
                1 => h.BigEndian ? BinaryPrimitives.ReadSingleBigEndian(x) : BinaryPrimitives.ReadSingleLittleEndian(x),
                2 => h.BigEndian ? BinaryPrimitives.ReadInt32BigEndian(x) : BinaryPrimitives.ReadInt32LittleEndian(x),
                3 => h.BigEndian ? BinaryPrimitives.ReadInt16BigEndian(x) : BinaryPrimitives.ReadInt16LittleEndian(x),
                4 => h.BigEndian ? BinaryPrimitives.ReadUInt16BigEndian(x) : BinaryPrimitives.ReadUInt16LittleEndian(x),
                _ => x[0],
            };
        }
        return r;
    }

    public static OrderedDictionary<string, MatValue> Read(byte[] b, HashSet<string>? wanted)
    {
        var result = new OrderedDictionary<string, MatValue>(StringComparer.Ordinal);
        var pos = 0;
        while (pos < b.Length)
        {
            var h = ReadHeader(b, pos);
            pos = h.End;
            if (wanted is not null && !wanted.Contains(h.Name)) continue;
            result[h.Name] = Decode(b, h);
        }
        return result;
    }

    public static List<MatVariableInfo> List(byte[] b)
    {
        var list = new List<MatVariableInfo>();
        var pos = 0;
        while (pos < b.Length)
        {
            var h = ReadHeader(b, pos);
            pos = h.End;
            var cls = h.T switch { 1 => "char", 2 => "sparse", _ => MatLayout.ClassName(PrecisionClass[h.P]) };
            int[] shape = [h.Rows, h.Cols];
            if (h.T == 2)
            {
                var v = Decode(b, h);
                shape = [v.Dimensions[0], v.Dimensions[1]];
            }
            list.Add(new MatVariableInfo(h.Name, shape, cls));
        }
        return list;
    }

    private static MatValue Decode(byte[] b, Header h)
    {
        var n = h.Rows * h.Cols;
        int[] dims = [h.Rows, h.Cols];
        var re = Values(b, h, h.DataOffset, n);
        double[]? im = h.Imag ? Values(b, h, h.DataOffset + n * PrecisionSize[h.P], n) : null;
        switch (h.T)
        {
            case 1:
                return new MatChar(dims, MatLayout.ColumnToRowMajor(re.Select(v => (char)(int)v).ToArray(), dims));
            case 2:
            {
                // (nnz + 1) x 3 or 4 matrix of 1-based [row, col, re, (im)]; the last row carries the shape.
                if (h.Cols < 3 || h.Rows < 1) throw new InvalidDataException($"Malformed v4 MAT file: sparse '{h.Name}' is not an N x 3/4 triplet matrix.");
                var nnz = h.Rows - 1;
                var r = h.Rows;
                var rows = (int)re[nnz];
                var cols = (int)re[r + nnz];
                var ri = new int[nnz];
                var ci = new int[nnz];
                var v = new double[nnz];
                var vi = h.Cols == 4 || im is not null ? new double[nnz] : null;
                for (var k = 0; k < nnz; k++)
                {
                    ri[k] = (int)re[k] - 1;
                    ci[k] = (int)re[r + k] - 1;
                    v[k] = re[2 * r + k];
                    if (vi is not null) vi[k] = h.Cols == 4 ? re[3 * r + k] : im![2 * r + k];
                }
                if (vi is null) return new MatSparse(CscMatrix.FromTriplets(rows, cols, ri, ci, v));
                var (csc, imag) = MatConvert.ComplexCsc(rows, cols, ri, ci, v, vi);
                return new MatSparse(csc, imag);
            }
            default:
            {
                var cls = PrecisionClass[h.P];
                var real = new NdArray(MatLayout.ColumnToRowMajor(re, dims), dims);
                var imag = im is null ? null : new NdArray(MatLayout.ColumnToRowMajor(im, dims), dims);
                return new MatArray(dims, real, imag, cls);
            }
        }
    }

    public static void Write(Stream s, IEnumerable<KeyValuePair<string, MatValue>> variables)
    {
        foreach (var (name, value) in variables)
        {
            MatFile.ValidateName(name);
            switch (value)
            {
                case MatArray a:
                    Matrix(s, name, 0, To2D(a.Dimensions), a.Real.AsContiguous().AsSpan(), a.Imaginary?.AsContiguous().AsSpan().ToArray(), a.Dimensions.ToArray());
                    break;
                case MatLogical l:
                    Matrix(s, name, 0, To2D(l.Dimensions), l.Values.AsContiguous().AsSpan(), null, l.Dimensions.ToArray());
                    break;
                case MatChar c:
                {
                    var chars = c.Chars.ToArray().Select(ch => (double)ch).ToArray();
                    Matrix(s, name, 1, To2D(c.Dimensions), chars, null, c.Dimensions.ToArray());
                    break;
                }
                case MatSparse sp:
                {
                    var m = sp.Matrix;
                    var nnz = m.Nnz;
                    var cols = sp.IsComplex ? 4 : 3;
                    var r = nnz + 1;
                    var cm = new double[r * cols];
                    var k = 0;
                    var ptr = m.Indptr;
                    var idx = m.Indices;
                    var data = m.Data;
                    var imag = sp.Imaginary;
                    for (var j = 0; j < m.Columns; j++)
                        for (var q = ptr[j]; q < ptr[j + 1]; q++, k++)
                        {
                            cm[k] = idx[q] + 1;
                            cm[r + k] = j + 1;
                            cm[2 * r + k] = data[q];
                            if (sp.IsComplex) cm[3 * r + k] = imag[q];
                        }
                    cm[nnz] = m.Rows;
                    cm[r + nnz] = m.Columns;
                    WriteRaw(s, name, 2, r, cols, cm, null);
                    break;
                }
                default:
                    throw new NotSupportedException($"Variable '{name}' is a MATLAB {value.ClassName}; the v4 format holds only 2-D numeric, char and sparse matrices. Use format \"5\".");
            }
        }
    }

    /// <summary>v4 is 2-D only: higher dimensions fold into the columns, which keeps MATLAB's column-major element order.</summary>
    private static (int Rows, int Cols) To2D(IReadOnlyList<int> dims) => dims.Count switch
    {
        0 => (1, 1),
        1 => (1, dims[0]),
        _ => (dims[0], MatLayout.Product(dims.Skip(1).ToArray())),
    };

    private static void Matrix(Stream s, string name, int t, (int Rows, int Cols) shape, ReadOnlySpan<double> rowMajor, double[]? imagRowMajor, int[] dims)
    {
        if (dims.Length == 0) dims = [1, 1];
        var re = MatLayout.RowToColumnMajor(rowMajor, dims);
        var im = imagRowMajor is null ? null : MatLayout.RowToColumnMajor(imagRowMajor, dims);
        WriteRaw(s, name, t, shape.Rows, shape.Cols, re, im);
    }

    private static void WriteRaw(Stream s, string name, int t, int rows, int cols, double[] colMajor, double[]? imag)
    {
        var nameBytes = Encoding.Latin1.GetBytes(name + "\0");
        Span<byte> h = stackalloc byte[20];
        BinaryPrimitives.WriteInt32LittleEndian(h, t); // M=0 (little-endian IEEE), O=0, P=0 (double)
        BinaryPrimitives.WriteInt32LittleEndian(h[4..], rows);
        BinaryPrimitives.WriteInt32LittleEndian(h[8..], cols);
        BinaryPrimitives.WriteInt32LittleEndian(h[12..], imag is null ? 0 : 1);
        BinaryPrimitives.WriteInt32LittleEndian(h[16..], nameBytes.Length);
        s.Write(h);
        s.Write(nameBytes);
        s.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(colMajor.AsSpan()));
        if (imag is not null) s.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(imag.AsSpan()));
    }
}
