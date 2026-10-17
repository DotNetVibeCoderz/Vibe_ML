namespace Gravicode.Science.SciNet.IO;

/// <summary>
/// Maps a NetCDF-4 file (an HDF5 file following the netCDF-4 conventions) onto
/// <see cref="NetCdfFile"/>/<see cref="NetCdfVariable"/>.
/// </summary>
/// <remarks>
/// The conventions, as netCDF-C writes them: a dimension is an HDF5 dimension scale (attribute
/// <c>CLASS</c> = <c>DIMENSION_SCALE</c>); one that is not also a coordinate variable has a
/// <c>NAME</c> starting "This is a netCDF dimension but not a netCDF variable" and its length is the
/// scale's current size. <c>_Netcdf4Dimid</c> numbers dimensions file-wide (and so gives their order);
/// a variable's dimensions are its <c>DIMENSION_LIST</c> (one object reference to a scale per axis),
/// except for a coordinate variable, which is its own scale (<c>_Netcdf4Coordinates</c> lists the ids
/// of a multi-dimensional one). Bookkeeping attributes are hidden, as netCDF hides them.
/// </remarks>
internal static class NetCdf4Reader
{
    private const string NotAVariable = "This is a netCDF dimension but not a netCDF variable";

    private static readonly HashSet<string> Hidden = new(StringComparer.Ordinal)
    {
        "CLASS", "NAME", "DIMENSION_LIST", "REFERENCE_LIST", "_Netcdf4Dimid", "_Netcdf4Coordinates",
        "_nc3_strict", "_NCProperties", "_IsNetcdf4", "_SuperblockVersion",
    };

    private sealed class State
    {
        public required Hdf5File H5 { get; init; }
        public required Stream Stream { get; init; }
        public required bool Mask { get; init; }
        public required string Model { get; init; }
        public Dictionary<long, string> DimByAddress { get; } = new();
        public Dictionary<int, string> DimById { get; } = new();
        public int Phony;
    }

    public static NetCdfFile Open(Stream stream, bool maskAndScale, bool ownsStream)
    {
        Hdf5File h5;
        try
        {
            h5 = Hdf5File.Open(stream, leaveOpen: true);
        }
        catch
        {
            if (ownsStream) stream.Dispose();
            throw;
        }
        var model = h5.Root.Attributes.ContainsKey("_nc3_strict") ? "NETCDF4_CLASSIC" : "NETCDF4";
        var root = NetCdfFile.ForNetCdf4(stream, ownsStream, maskAndScale, model);
        try
        {
            var st = new State { H5 = h5, Stream = stream, Mask = maskAndScale, Model = model };
            Build(st, h5.Root, root);
        }
        catch
        {
            root.Dispose();
            throw;
        }
        return root;
    }

    private static string? Text(Hdf5Object o, string name) => o.Attributes.TryGetValue(name, out var a) ? a.AsString() : null;

    private static bool IsScale(Hdf5Dataset d) => Text(d, "CLASS") == "DIMENSION_SCALE";

    private static void Build(State st, Hdf5Group g, NetCdfFile nc)
    {
        var members = g.Members.ToList();
        var datasets = members.OfType<Hdf5Dataset>().ToList();

        // Dimensions, in _Netcdf4Dimid order (link order for files without it).
        var scales = new List<(int Id, int Order, Hdf5Dataset D)>();
        for (var i = 0; i < datasets.Count; i++)
        {
            var d = datasets[i];
            if (!IsScale(d) || d.Rank == 0) continue;
            var id = d.Attributes.TryGetValue("_Netcdf4Dimid", out var a) && a.DataType.IsNumeric ? (int)a.ToDoubles()[0] : int.MaxValue;
            scales.Add((id, i, d));
        }
        scales.Sort((x, y) => x.Id != y.Id ? x.Id.CompareTo(y.Id) : x.Order.CompareTo(y.Order));
        string? recordDim = null;
        foreach (var (id, _, d) in scales)
        {
            var unlimited = d.MaxShape[0] < 0;
            nc.AddNetCdf4Dimension(d.Name, unlimited ? null : d.Shape[0]);
            if (unlimited) recordDim ??= d.Name;
            st.DimByAddress[d.Address] = d.Name;
            if (id != int.MaxValue) st.DimById[id] = d.Name;
        }

        foreach (var a in g.Attributes.Values)
            if (!Hidden.Contains(a.Name) && Convert(a) is { } v) nc.Attributes[a.Name] = v;

        foreach (var d in datasets)
        {
            if (IsScale(d) && (Text(d, "NAME") ?? "").StartsWith(NotAVariable, StringComparison.Ordinal)) continue;
            if (TypeOf(d.DataType) is not { } type) continue; // user-defined types: not modelled
            var dims = Dimensions(st, d, nc);
            var isRecord = dims.Length > 0 && d.MaxShape[0] < 0;
            var v = new NetCdfVariable(nc, d.Name, type, dims, isRecord) { H5 = d };
            foreach (var a in d.Attributes.Values)
                if (!Hidden.Contains(a.Name) && Convert(a) is { } value) v.Attributes[a.Name] = value;
            nc.AddNetCdf4Variable(v);
        }

        foreach (var sub in members.OfType<Hdf5Group>())
        {
            var child = NetCdfFile.ForNetCdf4(st.Stream, false, st.Mask, st.Model);
            Build(st, sub, child);
            nc.AddNetCdf4Group(sub.Name, child);
        }

        // An unlimited dimension's scale is not extended when data is written (unless it is also a
        // coordinate variable); netCDF reports the longest extent of any variable that uses it.
        if (recordDim is not null)
        {
            var scale = scales.First(x => x.D.Name == recordDim).D;
            nc.NetCdf4Records = Math.Max(scale.Shape[0], Longest(nc, recordDim));
        }
    }

    private static int Longest(NetCdfFile nc, string dim)
    {
        var n = 0;
        foreach (var v in nc.Variables.Values)
            for (var i = 0; i < v.Dimensions.Count; i++)
                if (v.Dimensions[i] == dim) n = Math.Max(n, v.Shape[i]);
        foreach (var g in nc.Groups.Values)
            if (!g.Dimensions.ContainsKey(dim)) n = Math.Max(n, Longest(g, dim)); // a nested group can shadow the name
        return n;
    }

    private static string[] Dimensions(State st, Hdf5Dataset d, NetCdfFile nc)
    {
        var rank = d.Rank;
        if (rank == 0) return [];
        if (d.Attributes.TryGetValue("DIMENSION_LIST", out var dl) && dl.Value is Array[] lists && lists.Length == rank)
        {
            var names = new string[rank];
            for (var i = 0; i < rank; i++)
            {
                if (lists[i] is not Hdf5Reference[] { Length: > 0 } refs || !st.DimByAddress.TryGetValue(refs[0].Address, out var n))
                    throw new InvalidDataException($"Malformed NetCDF-4 file: dimension {i} of '{d.Path}' does not refer to a known dimension scale.");
                names[i] = n;
            }
            return names;
        }
        if (IsScale(d))
        {
            if (d.Attributes.TryGetValue("_Netcdf4Coordinates", out var co) && co.DataType.IsNumeric)
            {
                var ids = co.ToDoubles();
                if (ids.Length == rank && ids.All(x => st.DimById.ContainsKey((int)x)))
                    return ids.Select(x => st.DimById[(int)x]).ToArray();
            }
            if (rank == 1) return [d.Name];
        }
        // Plain HDF5 data without dimension scales: netCDF-C invents "phony" dimensions too.
        var phony = new string[rank];
        for (var i = 0; i < rank; i++)
        {
            phony[i] = $"phony_dim_{st.Phony++}";
            nc.AddNetCdf4Dimension(phony[i], d.Shape[i]);
        }
        return phony;
    }

    private static NetCdfType? TypeOf(Hdf5DataType t) => t.Class switch
    {
        Hdf5TypeClass.Integer => (t.Size, t.IsSigned) switch
        {
            (1, true) => NetCdfType.Byte,
            (1, false) => NetCdfType.UByte,
            (2, true) => NetCdfType.Short,
            (2, false) => NetCdfType.UShort,
            (4, true) => NetCdfType.Int,
            (4, false) => NetCdfType.UInt,
            (8, true) => NetCdfType.Int64,
            (8, false) => NetCdfType.UInt64,
            _ => null,
        },
        Hdf5TypeClass.Float => t.Size switch { 4 => NetCdfType.Float, 8 => NetCdfType.Double, _ => null },
        Hdf5TypeClass.String => t.Size == 1 ? NetCdfType.Char : NetCdfType.String,
        Hdf5TypeClass.VariableLength when t.IsVariableLengthString => NetCdfType.String,
        _ => null,
    };

    /// <summary>SciPy's attribute convention: text as a string, one number as a scalar, several as an array.</summary>
    private static object? Convert(Hdf5Attribute a)
    {
        if (a.IsNull) return null;
        object? v;
        try
        {
            v = a.Value;
        }
        catch (NotSupportedException)
        {
            return null; // attributes of types netCDF cannot express are not part of the dataset
        }
        return v switch
        {
            null => null,
            string s => s,
            string?[] { Length: 1 } one => one[0] ?? "",
            string?[] many => many.Select(x => x ?? "").ToArray(),
            Array { Length: 1 } arr when arr.GetType().GetElementType()!.IsPrimitive => arr.GetValue(0),
            Array arr when arr.GetType().GetElementType()!.IsPrimitive => arr,
            _ when v.GetType().IsPrimitive => v,
            _ => null,
        };
    }
}
