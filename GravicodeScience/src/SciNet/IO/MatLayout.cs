using System.Collections;
using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Sparse;

namespace Gravicode.Science.SciNet.IO;

/// <summary>Shape and memory-order helpers shared by the MAT reader and writer.</summary>
internal static class MatLayout
{
    public static int Product(ReadOnlySpan<int> dims)
    {
        long p = 1;
        foreach (var d in dims) p *= d;
        if (p > int.MaxValue) throw new InvalidDataException($"Array with {p} elements exceeds the 2^31 element limit.");
        return (int)p;
    }

    /// <summary>NdArray shape for MATLAB dims; a squeezed scalar (no dims) becomes shape [1].</summary>
    public static int[] ShapeOf(int[] dims) => dims.Length == 0 ? [1] : dims;

    public static bool IsNumeric(MatClass c) => c is >= MatClass.Double and <= MatClass.UInt64;

    public static string ClassName(MatClass c) => c switch
    {
        MatClass.Double => "double",
        MatClass.Single => "single",
        MatClass.Int8 => "int8",
        MatClass.UInt8 => "uint8",
        MatClass.Int16 => "int16",
        MatClass.UInt16 => "uint16",
        MatClass.Int32 => "int32",
        MatClass.UInt32 => "uint32",
        MatClass.Int64 => "int64",
        MatClass.UInt64 => "uint64",
        MatClass.Cell => "cell",
        MatClass.Struct => "struct",
        MatClass.Object => "object",
        MatClass.Char => "char",
        MatClass.Sparse => "sparse",
        MatClass.Function => "function",
        MatClass.Opaque => "opaque",
        _ => "unknown",
    };

    public static int RowMajorIndex(int[] dims, int[] index)
    {
        if (index.Length != dims.Length) throw new ArgumentException($"Expected {dims.Length} subscripts, got {index.Length}.");
        var k = 0;
        for (var i = 0; i < dims.Length; i++)
        {
            if ((uint)index[i] >= (uint)dims[i]) throw new IndexOutOfRangeException($"Subscript {index[i]} out of range for dimension {i} of length {dims[i]}.");
            k = k * dims[i] + index[i];
        }
        return k;
    }

    /// <summary>True when column-major and row-major order coincide (at most one axis longer than 1).</summary>
    public static bool OrderIsTrivial(ReadOnlySpan<int> dims)
    {
        var big = 0;
        foreach (var d in dims) if (d > 1) big++;
        return big <= 1;
    }

    /// <summary>
    /// Reverses the axes of a row-major array: <paramref name="src"/> has shape
    /// <paramref name="shape"/>, the result has the reversed shape, also row-major. Column-major
    /// data with dims D *is* the row-major array of shape reverse(D), so this one routine converts
    /// MATLAB → GraviNum (call with reverse(D)) and back (call with D).
    /// </summary>
    public static unsafe void ReverseAxes(double* src, double* dst, int[] shape)
    {
        var n = Product(shape);
        if (OrderIsTrivial(shape))
        {
            Buffer.MemoryCopy(src, dst, (long)n * 8, (long)n * 8);
            return;
        }
        if (shape.Length == 2)
        {
            Transpose(src, dst, shape[0], shape[1]);
            return;
        }
        // n-D: walk the destination in order, tracking the source offset with an odometer.
        var rank = shape.Length;
        var srcStride = new int[rank];
        var acc = 1;
        for (var i = rank - 1; i >= 0; i--) { srcStride[i] = acc; acc *= shape[i]; }
        // destination axis k is source axis rank-1-k; the innermost destination axis is source axis 0.
        var inner = shape[0];
        var innerStride = srcStride[0];
        var counter = new int[rank]; // over source axes 1..rank-1 (destination axes rank-2..0)
        long baseOff = 0;
        var d = 0L;
        var outer = n / Math.Max(inner, 1);
        for (var o = 0; o < outer; o++)
        {
            var s = src + baseOff;
            for (var i = 0; i < inner; i++) dst[d++] = s[(long)i * innerStride];
            // increment destination odometer: destination axis rank-2 is source axis 1, etc.
            for (var ax = 1; ax < rank; ax++)
            {
                counter[ax]++;
                baseOff += srcStride[ax];
                if (counter[ax] < shape[ax]) break;
                baseOff -= (long)srcStride[ax] * shape[ax];
                counter[ax] = 0;
            }
        }
    }

    /// <summary>Cache-blocked transpose of a rows x cols row-major matrix; parallel above ~1M elements.</summary>
    public static unsafe void Transpose(double* src, double* dst, int rows, int cols)
    {
        const int B = 32;
        var rowBlocks = (rows + B - 1) / B;
        if ((long)rows * cols >= 1 << 20 && Environment.ProcessorCount > 1)
        {
            var s = (IntPtr)src;
            var t = (IntPtr)dst;
            Parallel.For(0, rowBlocks, rb => Block((double*)s, (double*)t, rows, cols, rb * B, B));
        }
        else
        {
            for (var rb = 0; rb < rowBlocks; rb++) Block(src, dst, rows, cols, rb * B, B);
        }

        static void Block(double* src, double* dst, int rows, int cols, int i0, int b)
        {
            var i1 = Math.Min(i0 + b, rows);
            for (var j0 = 0; j0 < cols; j0 += b)
            {
                var j1 = Math.Min(j0 + b, cols);
                for (var i = i0; i < i1; i++)
                {
                    var srow = src + (long)i * cols;
                    for (var j = j0; j < j1; j++) dst[(long)j * rows + i] = srow[j];
                }
            }
        }
    }

    public static double[] ColumnToRowMajor(double[] colMajor, int[] dims)
    {
        if (OrderIsTrivial(dims)) return colMajor;
        var r = new double[colMajor.Length];
        unsafe
        {
            fixed (double* s = colMajor)
            fixed (double* d = r)
                ReverseAxes(s, d, Enumerable.Reverse(dims).ToArray());
        }
        return r;
    }

    public static double[] RowToColumnMajor(ReadOnlySpan<double> rowMajor, int[] dims)
    {
        var r = new double[rowMajor.Length];
        if (OrderIsTrivial(dims)) { rowMajor.CopyTo(r); return r; }
        unsafe
        {
            fixed (double* s = rowMajor)
            fixed (double* d = r)
                ReverseAxes(s, d, dims);
        }
        return r;
    }

    /// <summary>perm[k] = row-major index of the k-th element in column-major order.</summary>
    public static int[] ColumnMajorOrder(int[] dims)
    {
        var n = Product(dims);
        var idx = new double[n];
        for (var i = 0; i < n; i++) idx[i] = i;
        var cm = RowToColumnMajor(idx, dims);
        var perm = new int[n];
        for (var i = 0; i < n; i++) perm[i] = (int)cm[i];
        return perm;
    }

    public static T[] ColumnToRowMajor<T>(T[] colMajor, int[] dims)
    {
        if (OrderIsTrivial(dims)) return colMajor;
        var perm = ColumnMajorOrder(dims);
        var r = new T[colMajor.Length];
        for (var k = 0; k < perm.Length; k++) r[perm[k]] = colMajor[k];
        return r;
    }

    public static T[] RowToColumnMajor<T>(IReadOnlyList<T> rowMajor, int[] dims)
    {
        var perm = ColumnMajorOrder(dims);
        var r = new T[perm.Length];
        for (var k = 0; k < perm.Length; k++) r[k] = rowMajor[perm[k]];
        return r;
    }
}

/// <summary>.NET object → <see cref="MatValue"/> conversion used by the writer.</summary>
internal static class MatConvert
{
    public static int[] DimsOf(NdArray a) => DimsOf(a.Shape.ToArray());

    public static int[] DimsOf(IReadOnlyList<int> shape) => shape.Count switch
    {
        0 => [1, 1],
        1 => [1, shape[0]],
        _ => shape.ToArray(),
    };

    public static CscMatrix Canonical(CscMatrix m) => m.SumDuplicates().SortIndices();

    /// <summary>Canonical CSC from complex triplets, with the imaginary parts aligned to the stored entries (duplicates summed).</summary>
    public static (CscMatrix Real, double[] Imag) ComplexCsc(int rows, int cols, int[] ri, int[] ci, double[] re, double[] im)
    {
        var order = Enumerable.Range(0, ri.Length).ToArray();
        Array.Sort(order, (a, b) => ci[a] != ci[b] ? ci[a].CompareTo(ci[b]) : ri[a].CompareTo(ri[b]));
        var ptr = new int[cols + 1];
        var idx = new List<int>(ri.Length);
        var vr = new List<double>(ri.Length);
        var vi = new List<double>(ri.Length);
        for (var q = 0; q < order.Length; q++)
        {
            var k = order[q];
            if ((uint)ri[k] >= (uint)rows || (uint)ci[k] >= (uint)cols) throw new ArgumentException($"Entry ({ri[k]}, {ci[k]}) outside a {rows} x {cols} matrix.");
            if (q > 0 && ci[order[q - 1]] == ci[k] && ri[order[q - 1]] == ri[k])
            {
                vr[^1] += re[k];
                vi[^1] += im[k];
                continue;
            }
            idx.Add(ri[k]);
            vr.Add(re[k]);
            vi.Add(im[k]);
            ptr[ci[k] + 1]++;
        }
        for (var j = 0; j < cols; j++) ptr[j + 1] += ptr[j];
        return (new CscMatrix(vr.ToArray(), idx.ToArray(), ptr, rows, cols, copy: false), vi.ToArray());
    }

    public static char[] CharMatrix(IReadOnlyList<string> rows, out int[] dims)
    {
        var width = rows.Count == 0 ? 0 : rows.Max(r => r.Length);
        dims = [rows.Count, width];
        var c = new char[rows.Count * width];
        for (var i = 0; i < rows.Count; i++)
        {
            var r = rows[i];
            r.CopyTo(0, c, i * width, r.Length);
            for (var j = r.Length; j < width; j++) c[i * width + j] = ' ';
        }
        return c;
    }

    public static MatValue[] ScalarStruct(IEnumerable<KeyValuePair<string, object?>> fields, out string[] names)
    {
        var list = fields.ToList();
        names = list.Select(p => p.Key).ToArray();
        return list.Select(p => FromObject(p.Value, false)).ToArray();
    }

    private static int[] Oned(int n, bool column) => column ? [n, 1] : [1, n];

    public static MatValue FromObject(object? value, bool onedAsColumn)
    {
        switch (value)
        {
            case null:
                return new MatArray([0, 0], NdArray.Zeros(0, 0), null, MatClass.Double);
            case MatValue mv:
                return mv;
            case NdArray nd:
            {
                var dims = nd.Rank == 1 ? Oned(nd.Size, onedAsColumn) : DimsOf(nd);
                return new MatArray(dims, nd.AsContiguous().Reshape(dims), null, MatClass.Double);
            }
            case ComplexNdArray c:
            {
                var dims = c.Rank == 1 ? Oned(c.Size, onedAsColumn) : DimsOf(c.Shape);
                return new MatArray(dims, c.Real().Reshape(dims), c.Imaginary().Reshape(dims), MatClass.Double);
            }
            case Complex z:
                return new MatArray([1, 1], new NdArray([z.Real], 1, 1), new NdArray([z.Imaginary], 1, 1), MatClass.Double);
            case Complex[] zs:
            {
                var dims = Oned(zs.Length, onedAsColumn);
                return new MatArray(dims, new NdArray(zs.Select(z => z.Real).ToArray(), dims), new NdArray(zs.Select(z => z.Imaginary).ToArray(), dims), MatClass.Double);
            }
            case string s:
                return new MatChar(s);
            case string[] ss:
                return new MatChar(ss);
            case char[] cs:
                return new MatChar(new string(cs));
            case char ch:
                return new MatChar(ch.ToString());
            case bool b:
                return new MatLogical([1, 1], new NdArray([b ? 1.0 : 0.0], 1, 1));
            case SparseMatrixBase sm:
                return new MatSparse(sm);
            case IEnumerable<KeyValuePair<string, object?>> dict:
                return new MatStruct(dict);
            case IDictionary idict:
            {
                var pairs = new List<KeyValuePair<string, object?>>();
                foreach (DictionaryEntry e in idict) pairs.Add(new(e.Key.ToString()!, e.Value));
                return new MatStruct(pairs);
            }
            case Array arr when arr.GetType().GetElementType() is { } et && (et.IsPrimitive || et == typeof(decimal)) && et != typeof(char) && et != typeof(IntPtr):
                return FromPrimitiveArray(arr, et, onedAsColumn);
            case Array objs when objs.GetType().GetElementType() == typeof(object):
            {
                var dims = objs.Rank == 1 ? Oned(objs.Length, onedAsColumn) : Enumerable.Range(0, objs.Rank).Select(objs.GetLength).ToArray();
                var items = new List<MatValue>(objs.Length);
                foreach (var o in objs) items.Add(FromObject(o, onedAsColumn)); // C# arrays enumerate row-major
                return new MatCell(dims, items);
            }
            case IList list:
            {
                var items = new List<MatValue>(list.Count);
                foreach (var o in list) items.Add(FromObject(o, onedAsColumn));
                return new MatCell(Oned(items.Count, onedAsColumn), items);
            }
        }
        if (value.GetType().IsPrimitive || value is decimal)
        {
            var (cls, isBool) = ClassOf(value.GetType());
            var v = Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
            return isBool ? new MatLogical([1, 1], new NdArray([v], 1, 1)) : new MatArray([1, 1], new NdArray([v], 1, 1), null, cls);
        }
        throw new ArgumentException($"Cannot convert a {value.GetType().Name} to a MATLAB value.");
    }

    private static (MatClass Class, bool IsBool) ClassOf(Type t)
    {
        if (t == typeof(double) || t == typeof(decimal)) return (MatClass.Double, false);
        if (t == typeof(float)) return (MatClass.Single, false);
        if (t == typeof(sbyte)) return (MatClass.Int8, false);
        if (t == typeof(byte)) return (MatClass.UInt8, false);
        if (t == typeof(short)) return (MatClass.Int16, false);
        if (t == typeof(ushort)) return (MatClass.UInt16, false);
        if (t == typeof(int)) return (MatClass.Int32, false);
        if (t == typeof(uint)) return (MatClass.UInt32, false);
        if (t == typeof(long)) return (MatClass.Int64, false);
        if (t == typeof(ulong)) return (MatClass.UInt64, false);
        if (t == typeof(bool)) return (MatClass.UInt8, true);
        throw new ArgumentException($"Unsupported element type {t.Name}.");
    }

    private static MatValue FromPrimitiveArray(Array arr, Type et, bool onedAsColumn)
    {
        var dims = arr.Rank == 1 ? Oned(arr.Length, onedAsColumn) : Enumerable.Range(0, arr.Rank).Select(arr.GetLength).ToArray();
        var (cls, isBool) = ClassOf(et);
        var data = new double[arr.Length];
        var i = 0;
        if (arr is double[] d1) data = (double[])d1.Clone();
        else
            foreach (var o in arr) data[i++] = Convert.ToDouble(o, System.Globalization.CultureInfo.InvariantCulture);
        var nd = new NdArray(data, dims);
        return isBool ? new MatLogical(dims, nd) : new MatArray(dims, nd, null, cls);
    }
}
