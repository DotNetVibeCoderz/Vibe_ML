using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Sparse;

namespace Gravicode.Science.SciNet.IO;

/// <summary>
/// MATLAB v7.3 MAT files: an HDF5 file behind a 512-byte user block holding the MAT header.
/// Every variable is a root-level dataset or group carrying a <c>MATLAB_class</c> attribute;
/// this maps that layout onto the same <see cref="MatValue"/> model the level 5 reader produces.
/// </summary>
/// <remarks>
/// MATLAB's layout, as written by MATLAB 7.3 onwards: data is stored with the dimensions reversed
/// (HDF5 is row-major, so this *is* MATLAB's column-major order); <c>char</c> is UTF-16 in uint16
/// (<c>MATLAB_int_decode</c> = 2); <c>logical</c> is uint8; complex is a compound of
/// <c>real</c>/<c>imag</c>; an empty array is a uint64 vector of its dimensions with
/// <c>MATLAB_empty</c> = 1; a cell array is a dataset of object references into <c>/#refs#</c>;
/// a struct is a group with a <c>MATLAB_fields</c> attribute — one child per field for a scalar
/// struct, one reference dataset per field for a struct array; a sparse matrix is a group
/// (<c>data</c>, <c>ir</c>, <c>jc</c>) with <c>MATLAB_sparse</c> = the row count. Function handles and
/// classdef objects (<c>MATLAB_object_decode</c>, kept in the undocumented <c>#subsystem#</c>) are
/// returned as <see cref="MatUnsupported"/>, as with level 5 files.
/// </remarks>
internal static class MatV73Reader
{
    public static OrderedDictionary<string, MatValue> ReadAll(byte[] file, HashSet<string>? wanted)
    {
        using var h5 = Hdf5File.Open(new MemoryStream(file, 0, file.Length, writable: false, publiclyVisible: true));
        var result = new OrderedDictionary<string, MatValue>(StringComparer.Ordinal);
        foreach (var name in h5.Root.Keys)
        {
            if (name.StartsWith('#')) continue; // #refs# (cell/struct-array elements), #subsystem# (classdef data)
            if (wanted is not null && !wanted.Contains(name)) continue;
            var obj = h5.Root[name];
            var v = Read(h5, obj, name);
            v.IsGlobal = Flag(obj, "MATLAB_global");
            result[name] = v;
        }
        return result;
    }

    public static List<MatVariableInfo> List(byte[] file)
    {
        using var h5 = Hdf5File.Open(new MemoryStream(file, 0, file.Length, writable: false, publiclyVisible: true));
        var list = new List<MatVariableInfo>();
        foreach (var name in h5.Root.Keys)
        {
            if (name.StartsWith('#')) continue;
            var obj = h5.Root[name];
            var cls = ClassOf(obj);
            int[] dims;
            if (obj is Hdf5Dataset d) dims = Flag(d, "MATLAB_empty") ? EmptyDims(d) : DimsOf(d);
            else if (obj.GetAttribute("MATLAB_sparse") is not null)
            {
                var g = (Hdf5Group)obj;
                dims = [checked((int)ToLong(obj.GetAttribute("MATLAB_sparse"))), g.Dataset("jc").Shape.Aggregate(1, (p, x) => p * x) - 1];
                cls = "sparse";
            }
            else dims = StructDims((Hdf5Group)obj);
            list.Add(new MatVariableInfo(name, dims, cls));
        }
        return list;
    }

    private static string ClassOf(Hdf5Object o) => o.Attributes.TryGetValue("MATLAB_class", out var a) ? a.AsString() ?? "" : "";

    private static bool Flag(Hdf5Object o, string name)
        => o.Attributes.TryGetValue(name, out var a) && !a.IsNull && a.DataType.IsNumeric && a.ToDoubles().FirstOrDefault() != 0;

    private static long ToLong(object? v) => v switch
    {
        null => 0,
        Array a => a.Length == 0 ? 0 : Convert.ToInt64(a.GetValue(0)),
        _ => Convert.ToInt64(v),
    };

    /// <summary>MATLAB dimensions: the HDF5 shape reversed, padded to at least two.</summary>
    private static int[] DimsOf(Hdf5Dataset d)
    {
        var s = d.Shape;
        var r = new int[Math.Max(2, s.Length)];
        Array.Fill(r, 1);
        for (var i = 0; i < s.Length; i++) r[i] = s[s.Length - 1 - i];
        return r;
    }

    /// <summary>An empty array stores its MATLAB dimensions as its data.</summary>
    private static int[] EmptyDims(Hdf5Dataset d)
    {
        var raw = d.Read<long>();
        var r = raw.Select(x => checked((int)x)).ToArray();
        return r.Length >= 2 ? r : [.. r, .. Enumerable.Repeat(r.Length == 0 ? 0 : 1, 2 - r.Length)];
    }

    private static MatClass NumericClass(string cls) => cls switch
    {
        "double" => MatClass.Double,
        "single" => MatClass.Single,
        "int8" => MatClass.Int8,
        "uint8" => MatClass.UInt8,
        "int16" => MatClass.Int16,
        "uint16" => MatClass.UInt16,
        "int32" => MatClass.Int32,
        "uint32" => MatClass.UInt32,
        "int64" => MatClass.Int64,
        "uint64" => MatClass.UInt64,
        _ => 0,
    };

    private static MatValue Read(Hdf5File h5, Hdf5Object obj, string name)
    {
        var cls = ClassOf(obj);
        if (obj is Hdf5Group g)
        {
            if (g.GetAttribute("MATLAB_sparse") is { } rows) return ReadSparse(g, checked((int)ToLong(rows)), cls == "logical");
            if (cls is "struct" or "") return ReadStruct(h5, g, null);
            if (cls == "function_handle")
                return new MatUnsupported([1, 1], MatClass.Function, $"Variable '{name}' is a MATLAB function handle; its v7.3 encoding is undocumented and it is skipped.");
            if (g.GetAttribute("MATLAB_fields") is not null && g.GetAttribute("MATLAB_object_decode") is null)
                return ReadStruct(h5, g, cls); // old-style object: a struct with a class name
            return new MatUnsupported([1, 1], MatClass.Opaque, $"Variable '{name}' is a MATLAB '{cls}' object; its v7.3 serialisation is undocumented and it is skipped.");
        }
        var d = (Hdf5Dataset)obj;
        if (d.GetAttribute("MATLAB_object_decode") is not null)
            return new MatUnsupported(DimsOf(d), MatClass.Opaque, $"Variable '{name}' is a MATLAB '{cls}' object (classdef, string, table, ...); its serialisation is undocumented and it is skipped, as SciPy does.");
        if (Flag(d, "MATLAB_empty"))
        {
            var ed = EmptyDims(d);
            return cls switch
            {
                "char" => new MatChar(ed, []),
                "logical" => new MatLogical(ed, new NdArray(new double[MatLayout.Product(ed)], MatLayout.ShapeOf(ed))),
                "cell" => new MatCell(ed, []),
                "struct" => new MatStruct(ed, Fields(d), []),
                _ => new MatArray(ed, new NdArray(new double[MatLayout.Product(ed)], MatLayout.ShapeOf(ed)), null, NumericClass(cls) == 0 ? MatClass.Double : NumericClass(cls)),
            };
        }
        var dims = DimsOf(d);
        var shape = MatLayout.ShapeOf(dims);
        switch (cls)
        {
            case "cell":
            case "" when d.DataType.Class == Hdf5TypeClass.Reference:
            {
                var refs = d.ReadReferences();
                var items = new MatValue[refs.Length];
                for (var i = 0; i < refs.Length; i++) items[i] = Element(h5, refs[i], name);
                return new MatCell(dims, MatLayout.ColumnToRowMajor(items, dims));
            }
            case "char":
            {
                var codes = d.Read<long>();
                var decode = d.GetAttribute("MATLAB_int_decode") is { } dec ? ToLong(dec) : 2;
                var chars = new char[codes.Length];
                for (var i = 0; i < codes.Length; i++)
                {
                    var c = codes[i];
                    // MATLAB_int_decode 4 (UTF-32) cannot keep one char per element beyond the BMP; such code points become U+FFFD.
                    chars[i] = decode == 4 && c > 0xFFFF ? '�' : (char)c;
                }
                return new MatChar(dims, MatLayout.ColumnToRowMajor(chars, dims));
            }
            case "logical":
                return new MatLogical(dims, new NdArray(MatLayout.ColumnToRowMajor(d.Read().ToArray(), dims), shape));
            default:
            {
                var mc = NumericClass(cls);
                if (mc == 0)
                    return new MatUnsupported(dims, MatClass.Opaque, $"Variable '{name}' has MATLAB class '{cls}', which SciNet does not decode.");
                if (d.DataType.Class == Hdf5TypeClass.Compound)
                {
                    var (re, im) = ComplexParts(d.ReadCompound(), d.Path);
                    return new MatArray(dims, new NdArray(MatLayout.ColumnToRowMajor(re, dims), shape), new NdArray(MatLayout.ColumnToRowMajor(im, dims), shape), mc);
                }
                return new MatArray(dims, new NdArray(MatLayout.ColumnToRowMajor(d.Read().ToArray(), dims), shape), null, mc);
            }
        }
    }

    private static MatValue Element(Hdf5File h5, Hdf5Reference r, string name)
    {
        if (r.IsNull) return new MatArray([0, 0], NdArray.Zeros(0, 0), null, MatClass.Double);
        return Read(h5, h5.Dereference(r), name);
    }

    private static (double[] Re, double[] Im) ComplexParts(IReadOnlyDictionary<string, Array> cols, string path)
    {
        if (!cols.TryGetValue("real", out var re) || !cols.TryGetValue("imag", out var im))
            throw new InvalidDataException($"Malformed v7.3 MAT file: compound variable '{path}' lacks real/imag members.");
        return (ToDoubles(re), ToDoubles(im));
    }

    private static double[] ToDoubles(Array a)
    {
        if (a is double[] d) return d;
        var r = new double[a.Length];
        for (var i = 0; i < r.Length; i++) r[i] = Convert.ToDouble(a.GetValue(i));
        return r;
    }

    private static string[] Fields(Hdf5Object o)
    {
        if (!o.Attributes.TryGetValue("MATLAB_fields", out var a) || a.IsNull) return [];
        // A variable-length sequence of single characters per field name.
        return a.Value switch
        {
            Array[] seqs => seqs.Select(s => string.Concat(s.Cast<object>().Select(c => c switch
            {
                string str => str,
                byte b => ((char)b).ToString(),
                sbyte sb => ((char)(byte)sb).ToString(),
                _ => c.ToString(),
            }))).ToArray(),
            string[] strs => strs,
            string s => [s],
            _ => [],
        };
    }

    private static int[] StructDims(Hdf5Group g)
    {
        var fields = Fields(g);
        foreach (var f in fields)
            if (g.TryGet(f, out var o) && o is Hdf5Dataset d && IsStructArrayField(d))
                return DimsOf(d);
        return [1, 1];
    }

    /// <summary>A struct array keeps each field as a reference dataset with no MATLAB_class of its own.</summary>
    private static bool IsStructArrayField(Hdf5Dataset d)
        => d.DataType.Class == Hdf5TypeClass.Reference && !d.Attributes.ContainsKey("MATLAB_class");

    private static MatValue ReadStruct(Hdf5File h5, Hdf5Group g, string? className)
    {
        var fields = Fields(g);
        if (fields.Length == 0) fields = g.Keys.ToArray();
        var children = new Hdf5Object[fields.Length];
        for (var f = 0; f < fields.Length; f++)
            children[f] = g.TryGet(fields[f], out var o) ? o! : throw new InvalidDataException($"Malformed v7.3 MAT file: struct '{g.Path}' has no member for field '{fields[f]}'.");
        var arrayField = children.OfType<Hdf5Dataset>().FirstOrDefault(IsStructArrayField);
        if (arrayField is null)
        {
            var values = new MatValue[fields.Length];
            for (var f = 0; f < fields.Length; f++) values[f] = Read(h5, children[f], fields[f]);
            return new MatStruct([1, 1], fields, [values], className);
        }
        var dims = DimsOf(arrayField);
        var n = MatLayout.Product(dims);
        var columns = new Hdf5Reference[fields.Length][];
        for (var f = 0; f < fields.Length; f++)
        {
            if (children[f] is not Hdf5Dataset fd || !IsStructArrayField(fd) || fd.Count != n)
                throw new InvalidDataException($"Malformed v7.3 MAT file: field '{fields[f]}' of struct array '{g.Path}' is not a matching reference array.");
            columns[f] = fd.ReadReferences();
        }
        var elements = new IReadOnlyList<MatValue>[n];
        for (var e = 0; e < n; e++)
        {
            var row = new MatValue[fields.Length];
            for (var f = 0; f < fields.Length; f++) row[f] = Element(h5, columns[f][e], fields[f]);
            elements[e] = row;
        }
        return new MatStruct(dims, fields, MatLayout.ColumnToRowMajor(elements, dims), className);
    }

    private static MatValue ReadSparse(Hdf5Group g, int rows, bool logical)
    {
        var jc = g.Dataset("jc").Read<long>().Select(x => checked((int)x)).ToArray();
        if (jc.Length == 0) throw new InvalidDataException($"Malformed v7.3 MAT file: sparse '{g.Path}' has an empty jc.");
        var cols = jc.Length - 1;
        var nnz = jc[^1];
        var ir = nnz == 0 || !g.Contains("ir") ? [] : g.Dataset("ir").Read<long>().Select(x => checked((int)x)).ToArray();
        double[] data;
        double[]? imag = null;
        if (nnz == 0 || !g.Contains("data")) data = [];
        else
        {
            var ds = g.Dataset("data");
            if (ds.DataType.Class == Hdf5TypeClass.Compound) (data, imag) = ComplexParts(ds.ReadCompound(), ds.Path);
            else data = ds.Read().ToArray();
        }
        if (ir.Length < nnz || data.Length < nnz) throw new InvalidDataException($"Malformed v7.3 MAT file: sparse '{g.Path}' has fewer entries than jc promises.");
        var csc = new CscMatrix(data[..nnz], ir[..nnz], jc, rows, cols, copy: false);
        if (!csc.HasCanonicalFormat)
        {
            if (imag is not null) throw new InvalidDataException("Malformed v7.3 MAT file: complex sparse matrix with unsorted or duplicate row indices.");
            csc = MatConvert.Canonical(csc);
        }
        return new MatSparse(csc, imag?[..nnz], logical);
    }
}
