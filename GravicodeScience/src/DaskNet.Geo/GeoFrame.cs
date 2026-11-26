using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet.Geo;

/// <summary>
/// Spatial operations on <see cref="DaskFrame"/>s (dask-geopandas' role). Points are a latitude
/// and a longitude column in degrees; polygons are a WKT text column (x = longitude, y = latitude).
/// Every operation is a <see cref="RemoteTaskAttribute"/> kernel per partition, so it runs on a
/// cluster as well as locally - a cluster's workers need this assembly loaded.
/// </summary>
/// <remarks>
/// Both joins broadcast the right-hand frame to every left partition: make the right side the
/// smaller one (the polygons, the store list). Inside a task the right side is indexed - a grid
/// of polygon bounding boxes, or a 3-D grid of points on the sphere - so each left row is tested
/// only against nearby candidates, and every candidate is then checked exactly.
/// </remarks>
public static class GeoFrame
{
    /// <summary>Adds a geohash column computed from latitude/longitude. Rows with a missing
    /// coordinate get a missing hash.</summary>
    public static DaskFrame WithGeohash(this DaskFrame frame, string latitude, string longitude, int precision = 7, string name = "geohash")
    {
        Check(frame, latitude, longitude);
        if (precision is < 1 or > 12) throw new ArgumentOutOfRangeException(nameof(precision));
        var parts = frame.ToDelayed().Select(p => Dask.Delayed(GeoKernels.AddGeohash, p,
            Dask.Value(new[] { latitude, longitude, name }), Dask.Value(precision), "geohash")).ToArray();
        return DaskFrame.FromDelayed(parts, [.. frame.Columns.Where(c => c != name), name]);
    }

    /// <summary>
    /// Re-partitions so nearby points share a partition: rows are range-partitioned by geohash,
    /// which follows a Z-order curve. Adds the geohash column (keep it - it is the sort key).
    /// </summary>
    public static DaskFrame SpatialPartition(this DaskFrame frame, string latitude, string longitude, int partitions, int precision = 7, string name = "geohash")
        => frame.WithGeohash(latitude, longitude, precision, name).SortBy(name, partitions: partitions);

    /// <summary>
    /// Pairs every left row with every right row within <paramref name="radiusKm"/> kilometres
    /// (great-circle), adding a distance column. Right columns that clash with left ones get
    /// <paramref name="suffix"/>. The right frame is broadcast; rows with a missing coordinate
    /// never match.
    /// </summary>
    public static DaskFrame DistanceJoin(this DaskFrame left, DaskFrame right, double radiusKm,
        (string Latitude, string Longitude) leftOn, (string Latitude, string Longitude) rightOn,
        string distanceColumn = "distance_km", string suffix = "_right")
    {
        Check(left, leftOn.Latitude, leftOn.Longitude);
        Check(right, rightOn.Latitude, rightOn.Longitude);
        if (!(radiusKm > 0)) throw new ArgumentOutOfRangeException(nameof(radiusKm), "The radius must be positive.");
        var broadcast = right.Repartition(1).ToDelayed()[0];
        var columns = new[] { leftOn.Latitude, leftOn.Longitude, rightOn.Latitude, rightOn.Longitude, distanceColumn, suffix };
        var parts = left.ToDelayed().Select(p => Dask.Delayed(GeoKernels.DistanceJoin, p, broadcast,
            Dask.Value(columns), Dask.Value(radiusKm), "distance-join")).ToArray();
        return DaskFrame.FromDelayed(parts, [.. JoinedColumns(left.Columns, right.Columns, suffix), distanceColumn]);
    }

    /// <summary>
    /// Attaches to each point the columns of every polygon containing it (points on a boundary
    /// count as inside). <paramref name="keepUnmatched"/> keeps points in no polygon, with missing
    /// polygon columns (a left join); otherwise they are dropped and a point in two overlapping
    /// polygons appears twice. The WKT column itself is not copied to the output.
    /// </summary>
    public static DaskFrame PointInPolygonJoin(this DaskFrame points, DaskFrame polygons,
        (string Latitude, string Longitude) pointOn, string wktColumn, bool keepUnmatched = false, string suffix = "_right")
    {
        Check(points, pointOn.Latitude, pointOn.Longitude);
        if (!polygons.Columns.Contains(wktColumn)) throw new ArgumentException($"No column '{wktColumn}' in the polygon frame.", nameof(wktColumn));
        var broadcast = polygons.Repartition(1).ToDelayed()[0];
        var columns = new[] { pointOn.Latitude, pointOn.Longitude, wktColumn, suffix };
        var parts = points.ToDelayed().Select(p => Dask.Delayed(GeoKernels.PointInPolygon, p, broadcast,
            Dask.Value(columns), Dask.Value(keepUnmatched), "point-in-polygon")).ToArray();
        return DaskFrame.FromDelayed(parts, JoinedColumns(points.Columns, [.. polygons.Columns.Where(c => c != wktColumn)], suffix));
    }

    /// <summary>
    /// Keeps rows inside a polygon (or on its boundary). The polygon travels with each task as WKT.
    /// </summary>
    public static DaskFrame WithinPolygon(this DaskFrame frame, string latitude, string longitude, Polygon polygon)
    {
        Check(frame, latitude, longitude);
        var parts = frame.ToDelayed().Select(p => Dask.Delayed(GeoKernels.Within, p,
            Dask.Value(new[] { latitude, longitude }), Dask.Value(polygon.ToWkt()), "within")).ToArray();
        return DaskFrame.FromDelayed(parts, frame.Columns);
    }

    internal static string[] JoinedColumns(IReadOnlyList<string> left, IReadOnlyList<string> right, string suffix)
        => [.. left, .. right.Select(c => left.Contains(c) ? c + suffix : c)];

    private static void Check(DaskFrame frame, string latitude, string longitude)
    {
        foreach (var c in new[] { latitude, longitude })
            if (!frame.Columns.Contains(c)) throw new ArgumentException($"No column '{c}'. Columns: {string.Join(", ", frame.Columns)}.");
    }
}

/// <summary>The per-partition kernels behind <see cref="GeoFrame"/>.</summary>
public static class GeoKernels
{
    /// <summary>Adds a geohash column; <paramref name="names"/> is (latitude, longitude, output).</summary>
    [RemoteTask]
    public static DataFrame AddGeohash(DataFrame frame, string[] names, int precision)
    {
        var lat = frame.Numeric(names[0]).Values;
        var lon = frame.Numeric(names[1]).Values;
        var hashes = new string?[frame.RowCount];
        for (var i = 0; i < hashes.Length; i++)
            if (double.IsFinite(lat[i]) && double.IsFinite(lon[i])) hashes[i] = Geohash.Encode(lat[i], lon[i], precision);
        var kept = frame.ColumnNames.Where(c => c != names[2]).ToArray();
        return new DataFrame([.. kept.Select(c => frame[c]), new TextSeries(names[2], hashes)]);
    }

    /// <summary>Keeps rows inside <paramref name="wkt"/>; <paramref name="names"/> is (latitude, longitude).</summary>
    [RemoteTask]
    public static DataFrame Within(DataFrame frame, string[] names, string wkt)
    {
        var polygon = Polygon.Parse(wkt);
        var lat = frame.Numeric(names[0]).Values;
        var lon = frame.Numeric(names[1]).Values;
        var mask = new bool[frame.RowCount];
        for (var i = 0; i < mask.Length; i++) mask[i] = double.IsFinite(lat[i]) && double.IsFinite(lon[i]) && polygon.Contains(lon[i], lat[i]);
        return frame.Filter(mask);
    }

    /// <summary>
    /// Distance join of one left partition against the broadcast right frame. <paramref name="names"/>
    /// is (left lat, left lon, right lat, right lon, distance column, suffix).
    /// </summary>
    [RemoteTask]
    public static DataFrame DistanceJoin(DataFrame left, DataFrame right, string[] names, double radiusKm)
    {
        var (lLat, lLon) = (left.Numeric(names[0]).Values, left.Numeric(names[1]).Values);
        var (rLat, rLon) = (right.Numeric(names[2]).Values, right.Numeric(names[3]).Values);

        // Points on a sphere of radius R, bucketed into cubes as wide as the chord of the radius:
        // anything within the radius is in the same or an adjacent cube. No pole or antimeridian
        // special cases, unlike a latitude/longitude grid.
        var chord = 2 * GeoDistance.EarthRadiusKm * Math.Sin(Math.Min(radiusKm / (2 * GeoDistance.EarthRadiusKm), Math.PI / 2));
        var cell = Math.Max(chord, 1e-9);
        var grid = new Dictionary<(long, long, long), List<int>>();
        for (var j = 0; j < right.RowCount; j++)
        {
            if (!double.IsFinite(rLat[j]) || !double.IsFinite(rLon[j])) continue;
            var key = Cell(Xyz(rLat[j], rLon[j]), cell);
            if (!grid.TryGetValue(key, out var list)) grid[key] = list = [];
            list.Add(j);
        }

        var li = new List<int>();
        var ri = new List<int>();
        var distances = new List<double>();
        var candidates = new List<int>();
        for (var i = 0; i < left.RowCount; i++)
        {
            if (!double.IsFinite(lLat[i]) || !double.IsFinite(lLon[i])) continue;
            var (cx, cy, cz) = Cell(Xyz(lLat[i], lLon[i]), cell);
            candidates.Clear();
            for (var dx = -1; dx <= 1; dx++)
                for (var dy = -1; dy <= 1; dy++)
                    for (var dz = -1; dz <= 1; dz++)
                        if (grid.TryGetValue((cx + dx, cy + dy, cz + dz), out var list)) candidates.AddRange(list);
            candidates.Sort();                           // right rows in their original order
            foreach (var j in candidates)
            {
                var d = GeoDistance.Haversine(lLat[i], lLon[i], rLat[j], rLon[j]);
                if (d > radiusKm) continue;
                li.Add(i);
                ri.Add(j);
                distances.Add(d);
            }
        }
        var joined = Combine(left, li, right, [.. right.ColumnNames], ri, names[5]);
        return joined.WithColumn(new NumericSeries(names[4], [.. distances]));
    }

    /// <summary>
    /// Point-in-polygon join of one point partition against the broadcast polygon frame.
    /// <paramref name="names"/> is (latitude, longitude, WKT column, suffix).
    /// </summary>
    [RemoteTask]
    public static DataFrame PointInPolygon(DataFrame points, DataFrame polygons, string[] names, bool keepUnmatched)
    {
        var lat = points.Numeric(names[0]).Values;
        var lon = points.Numeric(names[1]).Values;
        var index = new PolygonIndex(polygons.Text(names[2]).Values.ToArray());

        var li = new List<int>();
        var ri = new List<int>();
        var hits = new List<int>();
        for (var i = 0; i < points.RowCount; i++)
        {
            hits.Clear();
            if (double.IsFinite(lat[i]) && double.IsFinite(lon[i])) index.Query(lon[i], lat[i], hits);
            if (hits.Count == 0 && keepUnmatched) { li.Add(i); ri.Add(-1); }
            foreach (var j in hits) { li.Add(i); ri.Add(j); }
        }
        return Combine(points, li, polygons, [.. polygons.ColumnNames.Where(c => c != names[2])], ri, names[3]);
    }

    private static (double, double, double) Xyz(double lat, double lon)
    {
        const double rad = Math.PI / 180;
        var r = GeoDistance.EarthRadiusKm;
        return (r * Math.Cos(lat * rad) * Math.Cos(lon * rad), r * Math.Cos(lat * rad) * Math.Sin(lon * rad), r * Math.Sin(lat * rad));
    }

    private static (long, long, long) Cell((double X, double Y, double Z) p, double size)
        => ((long)Math.Floor(p.X / size), (long)Math.Floor(p.Y / size), (long)Math.Floor(p.Z / size));

    /// <summary>The left rows at <paramref name="li"/> beside the right rows at <paramref name="ri"/>
    /// (-1 = missing), right names suffixed where they clash.</summary>
    internal static DataFrame Combine(DataFrame left, List<int> li, DataFrame right, string[] rightColumns, List<int> ri, string suffix)
    {
        var columns = new List<Series>();
        foreach (var name in left.ColumnNames) columns.Add(left[name].Take(li));
        var names = GeoFrame.JoinedColumns(left.ColumnNames, rightColumns, suffix);
        for (var c = 0; c < rightColumns.Length; c++)
            columns.Add(TakeOrMissing(right[rightColumns[c]], ri, names[left.ColumnCount + c]));
        return new DataFrame(columns);
    }

    private static Series TakeOrMissing(Series s, List<int> rows, string name)
    {
        if (!rows.Contains(-1)) return s.Take(rows).Rename(name);
        switch (s)
        {
            case NumericSeries n:
                return new NumericSeries(name, rows.Select(r => r < 0 ? double.NaN : n.Values[r]).ToArray());
            case BooleanSeries b:
                return new BooleanSeries(name, rows.Select(r => r < 0 ? null : (bool?)b.GetValue(r)).ToArray());
            case DateTimeSeries d:
                return new DateTimeSeries(name, rows.Select(r => r < 0 ? null : (DateTime?)d.GetValue(r)).ToArray());
            default:
                return new TextSeries(name, rows.Select(r => r < 0 || s.IsMissing(r) ? null : s.GetValue(r)?.ToString()).ToArray());
        }
    }
}

/// <summary>A uniform grid over polygon bounding boxes; each query tests only the polygons whose
/// box covers the query's cell, and then tests those exactly.</summary>
internal sealed class PolygonIndex
{
    private readonly Polygon?[] _polygons;
    private readonly List<int>[,] _cells;
    private readonly Bounds _extent;
    private readonly int _n;

    public PolygonIndex(IReadOnlyList<string?> wkt)
    {
        _polygons = new Polygon?[wkt.Count];
        for (var i = 0; i < wkt.Count; i++)
            if (wkt[i] is { } text) _polygons[i] = Polygon.Parse(text);    // malformed WKT is an error, not a silent miss
        var live = _polygons.Where(p => p is not null).Select(p => p!.Bounds).ToArray();
        _extent = live.Length == 0 ? new Bounds(0, 0, 0, 0)
            : new Bounds(live.Min(b => b.MinX), live.Min(b => b.MinY), live.Max(b => b.MaxX), live.Max(b => b.MaxY));
        _n = Math.Clamp((int)Math.Ceiling(Math.Sqrt(Math.Max(1, live.Length))), 1, 256);
        _cells = new List<int>[_n, _n];
        for (var i = 0; i < _polygons.Length; i++)
        {
            if (_polygons[i] is not { } p) continue;
            var (x0, y0) = CellOf(p.Bounds.MinX, p.Bounds.MinY);
            var (x1, y1) = CellOf(p.Bounds.MaxX, p.Bounds.MaxY);
            for (var x = x0; x <= x1; x++)
                for (var y = y0; y <= y1; y++)
                    (_cells[x, y] ??= []).Add(i);
        }
    }

    private (int, int) CellOf(double x, double y)
    {
        var w = _extent.MaxX - _extent.MinX;
        var h = _extent.MaxY - _extent.MinY;
        var cx = w > 0 ? (int)((x - _extent.MinX) / w * _n) : 0;
        var cy = h > 0 ? (int)((y - _extent.MinY) / h * _n) : 0;
        return (Math.Clamp(cx, 0, _n - 1), Math.Clamp(cy, 0, _n - 1));
    }

    public void Query(double x, double y, List<int> hits)
    {
        if (!_extent.Contains(x, y)) return;
        var (cx, cy) = CellOf(x, y);
        if (_cells[cx, cy] is not { } list) return;
        foreach (var i in list)
            if (_polygons[i]!.Contains(x, y)) hits.Add(i);
    }
}
