using System.Globalization;
using Gravicode.Science.GraviFrame;
using NetTopologySuite.Geometries;
using NetTopologySuite.Geometries.Utilities;
using NetTopologySuite.Index.Strtree;
using NetTopologySuite.IO;
using NetTopologySuite.Operation.Buffer;
using NetTopologySuite.Operation.Union;
using NetTopologySuite.Simplify;

namespace Gravicode.Science.DaskNet.Geo;

/// <summary>How <see cref="GeometryFrame.Overlay"/> combines two layers (geopandas' <c>overlay(how=...)</c>).</summary>
public enum OverlayKind
{
    /// <summary>One row per overlapping (left, right) pair, its geometry the shared part.</summary>
    Intersection,
    /// <summary>Each left row minus every right geometry it overlaps; rows left empty are dropped.</summary>
    Difference,
}

/// <summary>
/// Geometry operations on WKT columns (dask-geopandas' constructive methods), backed by
/// NetTopologySuite: buffer, simplify, measures, dissolve (union by key) and overlay. All are
/// <see cref="RemoteTaskAttribute"/> kernels and run on a cluster as well as locally.
/// </summary>
/// <remarks>
/// <para>
/// Without a <see cref="GeoCrs"/> the arithmetic is planar, in the units of the coordinates: for
/// longitude/latitude data a buffer of 0.01 is 0.01 degrees and areas are square degrees. Given
/// one (<c>crs: GeoCrs.Wgs84</c>), buffer distances, simplify tolerances, areas and lengths are
/// metres on the ground: each geometry is projected into a transverse Mercator centred on its
/// own centroid, worked on there, and projected back, so results stay in the column's system.
/// That is exact at the centroid and off by (distance / Earth radius)^2 / 2 away from it - a part
/// in a million 9 km out, a part in ten thousand at 90 km - which suits anything up to a large
/// region; a continent wants an equal-area projection. Dissolve and overlay are topological and
/// need no system. Geometries crossing the antimeridian are not split.
/// </para>
/// <para>
/// Invalid input (a self-touching ring, a bow-tie) is repaired with NTS's <c>GeometryFixer</c>
/// before any operation, as shapely's <c>make_valid</c> would; a missing or empty WKT gives a
/// missing result. Output WKT is written with full double precision.
/// </para>
/// </remarks>
public static class GeometryFrame
{
    /// <summary>
    /// Replaces each geometry by its buffer of <paramref name="distance"/> (negative shrinks a
    /// polygon), into <paramref name="output"/> or in place. <paramref name="quadrantSegments"/>
    /// is the number of segments approximating a quarter circle.
    /// </summary>
    /// <remarks>With <paramref name="crs"/>, <paramref name="distance"/> is in metres on the ground.</remarks>
    public static DaskFrame Buffer(this DaskFrame frame, string wktColumn, double distance, int quadrantSegments = 8, string? output = null,
        GeoCrs? crs = null)
    {
        Check(frame, wktColumn);
        ArgumentOutOfRangeException.ThrowIfLessThan(quadrantSegments, 1);
        if (!double.IsFinite(distance)) throw new ArgumentOutOfRangeException(nameof(distance));
        return Map(frame, wktColumn, output, "buffer", [distance, (double)quadrantSegments], crs);
    }

    /// <summary>Simplifies each geometry within <paramref name="tolerance"/>, never changing its topology
    /// (Douglas-Peucker that keeps rings valid and holes inside).</summary>
    /// <remarks>With <paramref name="crs"/>, <paramref name="tolerance"/> is in metres on the ground.</remarks>
    public static DaskFrame Simplify(this DaskFrame frame, string wktColumn, double tolerance, string? output = null, GeoCrs? crs = null)
    {
        Check(frame, wktColumn);
        if (!(tolerance >= 0)) throw new ArgumentOutOfRangeException(nameof(tolerance));
        return Map(frame, wktColumn, output, "simplify", [tolerance], crs);
    }

    /// <summary>Replaces each geometry by its centroid (a POINT).</summary>
    /// <remarks>With <paramref name="crs"/> the centroid is taken in the local plane rather than in
    /// degrees, which matters for a large or high-latitude polygon.</remarks>
    public static DaskFrame Centroid(this DaskFrame frame, string wktColumn, string? output = null, GeoCrs? crs = null)
    {
        Check(frame, wktColumn);
        return Map(frame, wktColumn, output, "centroid", [], crs);
    }

    /// <summary>Adds the area and length (perimeter, for polygons) of each geometry: planar in the
    /// coordinates' units, or with <paramref name="crs"/> square metres and metres on the ground.</summary>
    public static DaskFrame WithMeasures(this DaskFrame frame, string wktColumn, string area = "area", string length = "length", GeoCrs? crs = null)
    {
        Check(frame, wktColumn);
        var parts = frame.ToDelayed().Select(p => Dask.Delayed(GeometryKernels.Measures, p,
            Dask.Value(new[] { wktColumn, area, length, crs?.ToString() ?? "" }), "measures")).ToArray();
        return DaskFrame.FromDelayed(parts, [.. frame.Columns.Where(c => c != area && c != length), area, length]);
    }

    /// <summary>Converts each geometry from <paramref name="from"/> to <paramref name="to"/>
    /// (geopandas' <c>to_crs</c>), into <paramref name="output"/> or in place.</summary>
    public static DaskFrame ToCrs(this DaskFrame frame, string wktColumn, GeoCrs from, GeoCrs to, string? output = null)
    {
        Check(frame, wktColumn);
        ArgumentNullException.ThrowIfNull(from);
        ArgumentNullException.ThrowIfNull(to);
        var target = output ?? wktColumn;
        var parts = frame.ToDelayed().Select(p => Dask.Delayed(GeometryKernels.Map, p,
            Dask.Value(new[] { wktColumn, target, "to-crs", from.ToString(), to.ToString() }), Dask.Value(Array.Empty<double>()), "to-crs")).ToArray();
        return DaskFrame.FromDelayed(parts, target == wktColumn ? frame.Columns : [.. frame.Columns.Where(c => c != target), target]);
    }

    /// <summary>
    /// Unions the geometries of each group of <paramref name="by"/> (geopandas' <c>dissolve</c>),
    /// with a row count per group. Each partition unions its own groups first and the partial
    /// unions are merged pairwise in a tree, so no task holds every input geometry. Result: the
    /// key columns, the WKT column and <paramref name="count"/>, in one partition. With no keys,
    /// everything is unioned into one row.
    /// </summary>
    public static DaskFrame Dissolve(this DaskFrame frame, string wktColumn, string[]? by = null, string count = "count")
    {
        Check(frame, wktColumn);
        by ??= [];
        foreach (var k in by) Check(frame, k);
        var spec = Dask.Value<object?[]>([wktColumn, by, count], "dissolve-spec");
        var level = frame.ToDelayed().Select(p => Dask.Delayed(GeometryKernels.PartialDissolve, p, spec, "dissolve-partial")).ToList();
        while (level.Count > 1)
            level = level.Chunk(2).Select(pair => pair.Length == 2
                ? Dask.Delayed(GeometryKernels.MergeDissolve, pair[0], pair[1], spec, "dissolve-merge")
                : pair[0]).ToList();
        return DaskFrame.FromDelayed([level[0]], [.. by, wktColumn, count]);
    }

    /// <summary>
    /// Overlays two polygon layers (geopandas' <c>overlay</c>). The right layer is broadcast to
    /// every left partition and indexed with an STR-tree, so each left geometry is intersected only
    /// with right geometries whose envelopes overlap it. <see cref="OverlayKind.Intersection"/>
    /// keeps both sides' columns (right clashes get <paramref name="suffix"/>, the right WKT column
    /// is dropped) and puts the shared part in <paramref name="leftWkt"/>; pairs that only touch
    /// (a zero-area intersection) are dropped. <see cref="OverlayKind.Difference"/> keeps the left
    /// columns.
    /// </summary>
    public static DaskFrame Overlay(this DaskFrame left, DaskFrame right, string leftWkt, string rightWkt,
        OverlayKind how = OverlayKind.Intersection, string suffix = "_right")
    {
        Check(left, leftWkt);
        Check(right, rightWkt);
        var broadcast = right.Repartition(1).ToDelayed()[0];
        var spec = Dask.Value<object?[]>([leftWkt, rightWkt, how.ToString(), suffix], "overlay-spec");
        var parts = left.ToDelayed().Select(p => Dask.Delayed(GeometryKernels.Overlay, p, broadcast, spec, "overlay")).ToArray();
        string[] columns = how == OverlayKind.Intersection
            ? GeoFrame.JoinedColumns(left.Columns, [.. right.Columns.Where(c => c != rightWkt)], suffix)
            : [.. left.Columns];
        return DaskFrame.FromDelayed(parts, columns);
    }

    private static DaskFrame Map(DaskFrame frame, string wktColumn, string? output, string op, double[] args, GeoCrs? crs)
    {
        var target = output ?? wktColumn;
        var parts = frame.ToDelayed().Select(p => Dask.Delayed(GeometryKernels.Map, p,
            Dask.Value(new[] { wktColumn, target, op, crs?.ToString() ?? "" }), Dask.Value(args), op)).ToArray();
        return DaskFrame.FromDelayed(parts, target == wktColumn ? frame.Columns : [.. frame.Columns.Where(c => c != target), target]);
    }

    private static void Check(DaskFrame frame, string column)
    {
        if (!frame.Columns.Contains(column)) throw new ArgumentException($"No column '{column}'. Columns: {string.Join(", ", frame.Columns)}.");
    }
}

/// <summary>The per-partition kernels behind <see cref="GeometryFrame"/>.</summary>
public static class GeometryKernels
{
    private static readonly GeometryFactory Factory = new(new PrecisionModel(), 0);

    internal static Geometry? Read(string? wkt)
    {
        if (string.IsNullOrWhiteSpace(wkt)) return null;
        var g = new WKTReader(Factory.GeometryServices).Read(wkt);
        if (g.IsEmpty) return null;
        return g.IsValid ? g : GeometryFixer.Fix(g);
    }

    internal static string? Write(Geometry? g) => g is null || g.IsEmpty ? null : new WKTWriter { MaxCoordinatesPerLine = int.MaxValue }.Write(g);

    private static Geometry?[] Geometries(DataFrame frame, string column)
    {
        var s = frame[column];
        var result = new Geometry?[frame.RowCount];
        for (var i = 0; i < result.Length; i++) result[i] = s.IsMissing(i) ? null : Read(s.GetValue(i)?.ToString());
        return result;
    }

    /// <summary>A geometry-to-geometry operation; <paramref name="names"/> is (input, output, op,
    /// crs) - an empty crs for planar - or (input, output, "to-crs", from, to).</summary>
    [RemoteTask]
    public static DataFrame Map(DataFrame frame, string[] names, double[] args)
    {
        var op = names[2];
        var crs = names.Length > 3 && names[3].Length > 0 ? GeoCrs.Parse(names[3]) : null;
        var to = op == "to-crs" ? GeoCrs.Parse(names[4]) : null;
        var input = Geometries(frame, names[0]);
        var result = new string?[input.Length];
        for (var i = 0; i < input.Length; i++)
        {
            if (input[i] is not { } g) continue;
            if (op == "to-crs")
            {
                result[i] = Write(crs!.Equals(to) ? g : GeoCrs.Transform(g, (x, y) =>
                {
                    var (lon, lat) = crs.ToGeographic(x, y);
                    return to!.FromGeographic(lon, lat);
                }));
                continue;
            }
            Geometry Apply(Geometry x) => op switch
            {
                "buffer" => x.Buffer(args[0], new BufferParameters((int)args[1])),
                "simplify" => TopologyPreservingSimplifier.Simplify(x, args[0]),
                "centroid" => x.Centroid,
                _ => throw new NotSupportedException(op),
            };
            if (crs is null) result[i] = Write(Apply(g));
            else
            {
                var local = new LocalProjection(g, crs, out var projected);
                var done = Apply(projected);
                result[i] = done.IsEmpty ? null : Write(local.Back(done));
            }
        }
        return Replace(frame, names[1], new TextSeries(names[1], result));
    }

    /// <summary>Area and length per row; <paramref name="names"/> is (wkt, area, length, crs).</summary>
    [RemoteTask]
    public static DataFrame Measures(DataFrame frame, string[] names)
    {
        var crs = names.Length > 3 && names[3].Length > 0 ? GeoCrs.Parse(names[3]) : null;
        var g = Geometries(frame, names[0]);
        if (crs is not null)
            g = g.Select(x =>
            {
                if (x is null) return null;
                _ = new LocalProjection(x, crs, out var projected);
                return projected;
            }).ToArray();
        var area = g.Select(x => x?.Area ?? double.NaN).ToArray();
        var length = g.Select(x => x?.Length ?? double.NaN).ToArray();
        return Replace(Replace(frame, names[1], new NumericSeries(names[1], area)), names[2], new NumericSeries(names[2], length));
    }

    /// <summary>Unions each group of one partition. <paramref name="spec"/> is (wkt, keys, count).</summary>
    [RemoteTask]
    public static DataFrame PartialDissolve(DataFrame frame, object?[] spec)
        => DissolveRows(frame, (string)spec[0]!, (string[])spec[1]!, (string)spec[2]!, counts: null);

    /// <summary>Merges two partial dissolves: their groups are unioned again and counts summed.</summary>
    [RemoteTask]
    public static DataFrame MergeDissolve(DataFrame a, DataFrame b, object?[] spec)
    {
        var count = (string)spec[2]!;
        var both = DataFrame.Concat([a, b]);
        return DissolveRows(both, (string)spec[0]!, (string[])spec[1]!, count, both.Numeric(count).Values.ToArray());
    }

    private static DataFrame DissolveRows(DataFrame frame, string wkt, string[] keys, string count, double[]? counts)
    {
        var geometries = Geometries(frame, wkt);
        var groups = new Dictionary<string, (int First, List<Geometry> Parts, double Count)>(StringComparer.Ordinal);
        var order = new List<string>();
        for (var i = 0; i < frame.RowCount; i++)
        {
            var key = string.Join("\u001f", keys.Select(k => frame[k].IsMissing(i) ? "\u0000" : Convert.ToString(frame[k].GetValue(i), CultureInfo.InvariantCulture)));
            if (!groups.TryGetValue(key, out var group))
            {
                group = (i, [], 0);
                order.Add(key);
            }
            if (geometries[i] is { } g) group.Parts.Add(g);
            group.Count += counts?[i] ?? 1;
            groups[key] = group;
        }
        var first = order.Select(k => groups[k].First).ToArray();
        var columns = keys.Select(k => frame[k].Take(first)).ToList();
        columns.Add(new TextSeries(wkt, order.Select(k => Write(Union(groups[k].Parts))).ToArray()));
        columns.Add(new NumericSeries(count, order.Select(k => groups[k].Count).ToArray()));
        return new DataFrame(columns);
    }

    private static Geometry? Union(List<Geometry> parts) => parts.Count switch
    {
        0 => null,
        1 => parts[0],
        _ => UnaryUnionOp.Union(parts),
    };

    /// <summary>Overlays one left partition with the broadcast right layer; <paramref name="spec"/>
    /// is (left wkt, right wkt, how, suffix).</summary>
    [RemoteTask]
    public static DataFrame Overlay(DataFrame left, DataFrame right, object?[] spec)
    {
        var (leftWkt, rightWkt, how, suffix) = ((string)spec[0]!, (string)spec[1]!, Enum.Parse<OverlayKind>((string)spec[2]!), (string)spec[3]!);
        var lg = Geometries(left, leftWkt);
        var rg = Geometries(right, rightWkt);
        var index = new STRtree<int>();
        for (var j = 0; j < rg.Length; j++)
            if (rg[j] is { } g) index.Insert(g.EnvelopeInternal, j);
        index.Build();

        var li = new List<int>();
        var ri = new List<int>();
        var shapes = new List<string?>();
        for (var i = 0; i < lg.Length; i++)
        {
            if (lg[i] is not { } a)
            {
                if (how == OverlayKind.Difference) { li.Add(i); shapes.Add(null); }
                continue;
            }
            var candidates = index.Query(a.EnvelopeInternal).Where(j => rg[j]!.Intersects(a)).Order().ToList();
            if (how == OverlayKind.Intersection)
            {
                foreach (var j in candidates)
                {
                    var shared = a.Intersection(rg[j]!);
                    if (shared.IsEmpty || (a.Dimension == Dimension.Surface && shared.Area == 0)) continue;
                    li.Add(i);
                    ri.Add(j);
                    shapes.Add(Write(Polygonal(shared, a.Dimension)));
                }
            }
            else
            {
                var rest = candidates.Count == 0 ? a : a.Difference(Union([.. candidates.Select(j => rg[j]!)])!);
                rest = Polygonal(rest, a.Dimension);
                if (rest.IsEmpty) continue;
                li.Add(i);
                shapes.Add(Write(rest));
            }
        }

        if (how == OverlayKind.Difference) return Replace(left.Take(li), leftWkt, new TextSeries(leftWkt, [.. shapes]));
        var rightColumns = right.ColumnNames.Where(c => c != rightWkt).ToArray();
        var joined = GeoKernels.Combine(left, li, right, rightColumns, ri, suffix);
        return Replace(joined, leftWkt, new TextSeries(leftWkt, [.. shapes]));
    }

    /// <summary>The parts of an overlay result of the input's own dimension: two polygons can share
    /// an edge as well as an area, and a GEOMETRYCOLLECTION of both is not what a polygon layer wants.</summary>
    private static Geometry Polygonal(Geometry g, Dimension dimension)
    {
        if (g is not GeometryCollection c || g is MultiPolygon or MultiLineString or MultiPoint) return g;
        var kept = Enumerable.Range(0, c.NumGeometries).Select(c.GetGeometryN).Where(x => x.Dimension == dimension).ToArray();
        return kept.Length == 1 ? kept[0] : Factory.BuildGeometry(kept);
    }

    private static DataFrame Replace(DataFrame frame, string name, Series column)
        => frame.ColumnNames.Contains(name)
            ? new DataFrame(frame.ColumnNames.Select(c => c == name ? column : frame[c]))
            : frame.WithColumn(column);
}
