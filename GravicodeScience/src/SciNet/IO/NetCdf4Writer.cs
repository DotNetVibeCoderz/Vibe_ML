using System.Runtime.InteropServices;
using System.Text;

namespace Gravicode.Science.SciNet.IO;

/// <summary>
/// Writes a <see cref="NetCdfFile"/> created with version 4 as a NetCDF-4 file: an HDF5 file
/// (through <see cref="Hdf5Writer"/>) following the conventions <see cref="NetCdf4Reader"/>
/// documents, as netCDF-C writes them.
/// </summary>
/// <remarks>
/// Each dimension becomes an HDF5 dimension scale: the coordinate variable's dataset when the
/// dimension has one (a 1-D variable of the same name), otherwise an unallocated float32 dataset
/// whose <c>NAME</c> is netCDF's "This is a netCDF dimension but not a netCDF variable" followed by
/// the length in ten columns. Scales carry <c>CLASS</c> = <c>DIMENSION_SCALE</c>,
/// <c>_Netcdf4Dimid</c> (the dimension's index, which gives netCDF its order) and
/// <c>REFERENCE_LIST</c> (every (variable, axis) attached); each other variable carries
/// <c>DIMENSION_LIST</c>, one object reference per axis. A record variable is chunked with an
/// unlimited first dimension and padded to the file's record count with its fill value, as in a
/// classic file. The HDF5 fill value is the variable's <c>_FillValue</c> (converted to its type, as
/// netCDF requires) or netCDF's default. Text attributes are NUL-terminated fixed strings (NC_CHAR),
/// string arrays variable-length strings (NC_STRING).
/// </remarks>
internal static class NetCdf4Writer
{
    private const string NotAVariable = "This is a netCDF dimension but not a netCDF variable.";

    public static void Write(NetCdfFile f, Stream output)
    {
        var w = new Hdf5Writer();
        var root = w.Root;
        foreach (var (k, v) in f.Attributes) root.SetAttribute(k, Attribute(k, v, null));

        var numrecs = f.RecordCount;
        var dims = f.Dimensions.Keys.ToList();
        var vars = f.Variables.Values.ToList();
        bool IsCoordinate(NetCdfVariable v) => v.Dimensions.Count == 1 && v.Dimensions[0] == v.Name && f.Dimensions.ContainsKey(v.Name);
        var scales = new Dictionary<string, Hdf5WriterDataset>(StringComparer.Ordinal);

        // Dimensions without a coordinate variable first, as netCDF-C defines them before the variables.
        foreach (var d in dims)
        {
            if (vars.Any(v => v.Name == d && IsCoordinate(v))) continue;
            if (vars.Any(v => v.Name == d))
                throw new NotSupportedException($"Variable '{d}' has the name of a dimension it is not the coordinate variable of; netCDF-4 would need to rename the dimension's scale, which this writer does not do.");
            var len = f.Dimensions[d] ?? numrecs;
            var data = H5WData.Numeric(Array.Empty<float>(), H5WType.F32, [len]);
            var scale = root.CreateDataset(d, data, maxShape: f.Dimensions[d] is null ? [-1] : null, allocate: false);
            scale.SetAttribute("NAME", H5WData.FixedString($"{NotAVariable}{len,10}\0"));
            scales[d] = scale;
        }

        var datasets = new List<(NetCdfVariable V, Hdf5WriterDataset D)>();
        foreach (var v in vars)
        {
            var shape = f.ShapeOf(v.Dimensions, numrecs).Select(x => (long)x).ToArray();
            var (data, fill) = Data(v, shape, numrecs);
            var maxShape = v.IsRecord ? shape.Select((x, i) => i == 0 ? -1 : x).ToArray() : null;
            var ds = root.CreateDataset(v.Name, data, maxShape: maxShape, fill: fill);
            foreach (var (k, a) in v.Attributes) ds.SetAttribute(k, Attribute(k, a, k == "_FillValue" ? v.Type : null));
            if (IsCoordinate(v))
            {
                ds.SetAttribute("NAME", H5WData.FixedString(v.Name + "\0"));
                scales[v.Name] = ds;
            }
            datasets.Add((v, ds));
        }

        // Dimension scale bookkeeping: CLASS and _Netcdf4Dimid on scales, DIMENSION_LIST on variables, REFERENCE_LIST back.
        var attached = dims.ToDictionary(d => d, _ => new List<(Hdf5WriterDataset D, int Axis)>(), StringComparer.Ordinal);
        foreach (var (v, ds) in datasets)
        {
            if (IsCoordinate(v) || v.Dimensions.Count == 0) continue;
            var lists = new Hdf5WriterNode[v.Dimensions.Count][];
            for (var i = 0; i < lists.Length; i++)
            {
                lists[i] = [scales[v.Dimensions[i]]];
                attached[v.Dimensions[i]].Add((ds, i));
            }
            ds.SetAttribute("DIMENSION_LIST", H5WData.VlenReferences(lists, [lists.Length]));
        }
        var refList = H5WType.Compound(("dataset", H5WType.Reference), ("dimension", H5WType.U32));
        for (var id = 0; id < dims.Count; id++)
        {
            var scale = scales[dims[id]];
            scale.SetAttribute("CLASS", H5WData.FixedString("DIMENSION_SCALE\0"));
            scale.SetAttribute("_Netcdf4Dimid", H5WData.Scalar(id));
            var refs = attached[dims[id]];
            if (refs.Count == 0) continue;
            var copy = refs.ToArray();
            scale.SetAttribute("REFERENCE_LIST", H5WData.DeferredRecords(refList, copy.Length, (ctx, dst) =>
            {
                for (var i = 0; i < copy.Length; i++)
                {
                    MemoryMarshal.Write(dst[(12 * i)..], ctx.AddressOf(copy[i].D));
                    MemoryMarshal.Write(dst[(12 * i + 8)..], (uint)copy[i].Axis);
                }
            }));
        }
        w.Save(output);
    }

    /// <summary>The variable's data in its own type (padded to the record count), and the encoded HDF5 fill value.</summary>
    private static (H5WData Data, byte[]? Fill) Data(NetCdfVariable v, long[] shape, int numrecs)
    {
        if (v.Type == NetCdfType.String)
        {
            var strings = v.GetStrings();
            return (H5WData.VlenStrings(strings, shape), null);
        }
        var values = v.ValuesForWrite(numrecs);
        var fillValue = v.FillValue;
        var hasFillAttr = v.Attributes.ContainsKey("_FillValue");
        // Elements past the data that was set are fill (ValuesForWrite pads with it). netCDF's 64-bit
        // default fills are not doubles, so those positions get the exact values; matching by value
        // instead would also rewrite a real element that happens to round to the same double.
        var set = v.PendingData?.Size ?? 0;
        bool Unset(int i) => !hasFillAttr && i >= set;
        Array typed = v.Type switch
        {
            NetCdfType.Byte => values.Select(x => (sbyte)x).ToArray(),
            NetCdfType.Char => values.Select(x => (byte)x).ToArray(),
            NetCdfType.Short => values.Select(x => (short)x).ToArray(),
            NetCdfType.Int => values.Select(x => (int)x).ToArray(),
            NetCdfType.Float => values.Select(x => (float)x).ToArray(),
            NetCdfType.Double => values,
            NetCdfType.UByte => values.Select(x => (byte)x).ToArray(),
            NetCdfType.UShort => values.Select(x => (ushort)x).ToArray(),
            NetCdfType.UInt => values.Select(x => (uint)x).ToArray(),
            NetCdfType.Int64 => values.Select((x, i) => Unset(i) ? -9223372036854775806L : (long)x).ToArray(),
            NetCdfType.UInt64 => values.Select((x, i) => Unset(i) ? 18446744073709551614UL : (ulong)x).ToArray(),
            _ => throw new NotSupportedException($"NetCDF type {v.Type} cannot be written."),
        };
        var type = v.Type == NetCdfType.Char ? H5WType.FixedString(1, false) : H5WData.TypeOf(typed.GetType().GetElementType()!)!;
        var fill = One(v.Type, fillValue, hasFillAttr);
        return (v.Type == NetCdfType.Char ? H5WData.Raw((byte[])typed, type, shape) : H5WData.Numeric(typed, type, shape), fill);
    }

    private static byte[] One(NetCdfType t, double x, bool explicitFill) => t switch
    {
        NetCdfType.Byte => [(byte)(sbyte)x],
        NetCdfType.Char or NetCdfType.UByte => [(byte)x],
        NetCdfType.Short => BitConverter.GetBytes((short)x),
        NetCdfType.UShort => BitConverter.GetBytes((ushort)x),
        NetCdfType.Int => BitConverter.GetBytes((int)x),
        NetCdfType.UInt => BitConverter.GetBytes((uint)x),
        NetCdfType.Int64 => BitConverter.GetBytes(explicitFill ? (long)x : -9223372036854775806L),
        NetCdfType.UInt64 => BitConverter.GetBytes(explicitFill ? (ulong)x : 18446744073709551614UL),
        NetCdfType.Float => BitConverter.GetBytes((float)x),
        _ => BitConverter.GetBytes(x),
    };

    /// <summary>
    /// A netCDF attribute value as HDF5 data: text as a NUL-terminated fixed string (NC_CHAR),
    /// string arrays as variable-length strings (NC_STRING), numbers as a 1-D array of their type.
    /// <c>_FillValue</c> is converted to the variable's type, which netCDF insists on.
    /// </summary>
    private static H5WData Attribute(string name, object value, NetCdfType? coerce)
    {
        switch (value)
        {
            case string s:
                return H5WData.FixedString(s);
            case string[] ss:
                return H5WData.VlenStrings(ss, [ss.Length]);
            case char c:
                return H5WData.FixedString(c.ToString());
        }
        if (coerce is { } t and not (NetCdfType.Char or NetCdfType.String))
        {
            var d = value switch
            {
                Array a when a.Length > 0 => System.Convert.ToDouble(a.GetValue(0)),
                IConvertible c => c.ToDouble(System.Globalization.CultureInfo.InvariantCulture),
                _ => throw new ArgumentException($"Attribute '{name}' must be a number."),
            };
            var bytes = One(t, d, explicitFill: true);
            var type = t switch
            {
                NetCdfType.Byte => H5WType.I8,
                NetCdfType.UByte => H5WType.U8,
                NetCdfType.Short => H5WType.I16,
                NetCdfType.UShort => H5WType.U16,
                NetCdfType.Int => H5WType.I32,
                NetCdfType.UInt => H5WType.U32,
                NetCdfType.Int64 => H5WType.I64,
                NetCdfType.UInt64 => H5WType.U64,
                NetCdfType.Float => H5WType.F32,
                _ => H5WType.F64,
            };
            return H5WData.Raw(bytes, type, [1]);
        }
        var arr = value is Array x ? x : CreateOne(value);
        var et = arr.GetType().GetElementType()!;
        var ht = H5WData.TypeOf(et) ?? throw new ArgumentException($"Attribute '{name}' has type {value.GetType().Name}; NetCDF-4 attributes are strings or numbers (scalars or 1-D arrays).");
        if (arr.Rank != 1) throw new ArgumentException($"Attribute '{name}' must be one-dimensional.");
        if (et == typeof(Half)) throw new ArgumentException($"Attribute '{name}': NetCDF has no 16-bit float type.");
        return H5WData.Numeric(arr, ht, [arr.Length]);
    }

    private static Array CreateOne(object value)
    {
        var a = Array.CreateInstance(value.GetType(), 1);
        a.SetValue(value, 0);
        return a;
    }
}
