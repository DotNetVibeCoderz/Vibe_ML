using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.DaskNet.Geo;
using Gravicode.Science.GraviFrame;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet.Geo;

/// <summary>Geohash against published values; every join against a brute-force double loop.</summary>
public sealed class GeoTests
{
    [Theory]
    [InlineData(42.6, -5.6, 5, "ezs42")]                    // the example on Wikipedia's Geohash page
    [InlineData(57.64911, 10.40744, 11, "u4pruydqqvj")]     // geohash.org's reference point
    [InlineData(-25.382708, -49.265506, 9, "6gkzwgjzn")]
    public void Geohash_matches_published_values(double lat, double lon, int precision, string expected)
    {
        Assert.Equal(expected, Geohash.Encode(lat, lon, precision));
        var cell = Geohash.Decode(expected);
        Assert.InRange(lat, cell.LatitudeMin, cell.LatitudeMax);
        Assert.InRange(lon, cell.LongitudeMin, cell.LongitudeMax);
        var neighbours = Geohash.Neighbors(expected);
        Assert.Equal(8, neighbours.Length);
        Assert.All(neighbours, n => Assert.Equal(expected.Length, n.Length));
        Assert.DoesNotContain(expected, neighbours);
    }

    [Fact]
    public void Haversine_gives_known_distances()
    {
        Assert.InRange(GeoDistance.Haversine(51.5074, -0.1278, 48.8566, 2.3522), 342.5, 344.5);    // London-Paris
        Assert.Equal(Math.PI * GeoDistance.EarthRadiusKm, GeoDistance.Haversine(0, 0, 0, 180), 6);
        Assert.Equal(0, GeoDistance.Haversine(10, 179.9, 10, 179.9));
        Assert.True(GeoDistance.Haversine(0, 179.9, 0, -179.9) < 23);                              // across the antimeridian
    }

    [Fact]
    public void Polygons_contain_with_holes_boundaries_and_parts()
    {
        var donut = Polygon.Parse("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (4 4, 6 4, 6 6, 4 6, 4 4))");
        Assert.True(donut.Contains(1, 1));
        Assert.False(donut.Contains(5, 5));              // in the hole
        Assert.True(donut.Contains(4, 5));               // on the hole's edge
        Assert.True(donut.Contains(10, 3));              // on the shell
        Assert.True(donut.Contains(0, 0));               // a vertex
        Assert.False(donut.Contains(11, 5));
        Assert.Equal(96, donut.Area(), 12);

        var multi = Polygon.Parse("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), ((5 5, 7 5, 7 7, 5 7, 5 5)))");
        Assert.True(multi.Contains(6, 6));
        Assert.True(multi.Contains(0.9, 0.1));
        Assert.False(multi.Contains(3, 3));
        Assert.Equal(new Bounds(0, 0, 7, 7), multi.Bounds);
        Assert.Equal(4.5, multi.Area(), 12);

        Assert.True(Polygon.Parse(multi.ToWkt()).Contains(6, 6));
        Assert.Equal(donut.ToWkt(), Polygon.Parse(donut.ToWkt()).ToWkt());
        Assert.True(Polygon.Parse("POLYGON Z ((0 0 1, 2 0 1, 2 2 1, 0 0 1))").Contains(1.5, 0.5));
        Assert.True(Polygon.Parse("polygon((0 0,1 0,1 1))").Contains(0.9, 0.5));   // closed for you
        Assert.Throws<FormatException>(() => Polygon.Parse("POINT (1 2)"));
        Assert.Throws<FormatException>(() => Polygon.Parse("POLYGON ((0 0, 1 1, 0 0))"));
        Assert.Throws<FormatException>(() => Polygon.Parse("POLYGON ((0 0, 1 0, 1 1, 0 0)) extra"));
        Assert.False(Polygon.TryParse("POLYGON ((0 0, x 0, 1 1, 0 0))", out _));
    }

    [Fact]
    public void Containment_agrees_with_a_brute_force_crossing_count_on_a_star()
    {
        // A concave 10-point star: ray casting must handle the reflex vertices.
        var points = Enumerable.Range(0, 10).Select(k =>
        {
            var r = k % 2 == 0 ? 10.0 : 4.0;
            var a = Math.PI * 2 * k / 10;
            return (X: r * Math.Cos(a), Y: r * Math.Sin(a));
        }).ToArray();
        var star = Polygon.Parse("POLYGON ((" + string.Join(", ", points.Select(p => FormattableString.Invariant($"{p.X} {p.Y}"))) + "))");
        var random = new Random(3);
        for (var i = 0; i < 5000; i++)
        {
            double x = random.NextDouble() * 24 - 12, y = random.NextDouble() * 24 - 12;
            Assert.Equal(WindingNumber(points, x, y) != 0, star.Contains(x, y));
        }
    }

    private static int WindingNumber((double X, double Y)[] ring, double x, double y)
    {
        var w = 0;
        for (var i = 0; i < ring.Length; i++)
        {
            var (a, b) = (ring[i], ring[(i + 1) % ring.Length]);
            var cross = (b.X - a.X) * (y - a.Y) - (x - a.X) * (b.Y - a.Y);
            if (a.Y <= y) { if (b.Y > y && cross > 0) w++; }
            else if (b.Y <= y && cross < 0) w--;
        }
        return w;
    }

    private static DataFrame Points(int n, int seed, string prefix)
    {
        var random = new Random(seed);
        var lat = new double[n];
        var lon = new double[n];
        for (var i = 0; i < n; i++)
        {
            // A third near the antimeridian, a few near a pole, the rest anywhere.
            lat[i] = i % 3 == 0 ? random.NextDouble() * 20 - 10 : i % 17 == 0 ? 85 + random.NextDouble() * 5 : random.NextDouble() * 160 - 80;
            lon[i] = i % 3 == 0 ? (random.NextDouble() < 0.5 ? 178 + random.NextDouble() * 2 : -180 + random.NextDouble() * 2) : random.NextDouble() * 360 - 180;
        }
        lat[n - 1] = double.NaN;                         // a missing coordinate never matches
        return new DataFrame([
            new NumericSeries(prefix + "id", Enumerable.Range(0, n).Select(i => (double)i).ToArray()),
            new NumericSeries("lat", lat), new NumericSeries("lon", lon),
            new TextSeries("name", Enumerable.Range(0, n).Select(i => prefix + i).ToArray())]);
    }

    [Fact]
    public async Task Distance_join_equals_the_brute_force_pairs_locally_and_on_a_cluster()
    {
        var left = Points(900, 1, "a");
        var right = Points(300, 2, "b");
        const double radius = 400;
        var expected = new List<(double, double, double)>();
        for (var i = 0; i < left.RowCount; i++)
            for (var j = 0; j < right.RowCount; j++)
            {
                var d = GeoDistance.Haversine(left.Numeric("lat")[i], left.Numeric("lon")[i], right.Numeric("lat")[j], right.Numeric("lon")[j]);
                if (d <= radius) expected.Add((i, j, d));
            }
        Assert.True(expected.Count > 200, "the fixture should produce many pairs");

        var joined = DaskFrame.FromFrame(left, 5).DistanceJoin(DaskFrame.FromFrame(right, 3), radius, ("lat", "lon"), ("lat", "lon"));
        Assert.Equal(["aid", "lat", "lon", "name", "bid", "lat_right", "lon_right", "name_right", "distance_km"], joined.Columns);

        await using var cluster = await LocalCluster.StartAsync(workers: 2, threadsPerWorker: 2);
        foreach (var result in new[] { joined.Compute(), joined.Compute(cluster.Client) })
        {
            Assert.Equal(joined.Columns, result.ColumnNames);
            var actual = Enumerable.Range(0, result.RowCount)
                .Select(r => (result.Numeric("aid")[r], result.Numeric("bid")[r], result.Numeric("distance_km")[r]))
                .OrderBy(t => t.Item1).ThenBy(t => t.Item2).ToList();
            Assert.Equal(expected.Count, actual.Count);
            for (var k = 0; k < actual.Count; k++)
            {
                Assert.Equal((expected[k].Item1, expected[k].Item2), (actual[k].Item1, actual[k].Item2));
                Assert.Equal(expected[k].Item3, actual[k].Item3, 9);
            }
            Assert.Equal("b" + (int)actual[0].Item2, result.Text("name_right")[result.Numeric("bid").Values.ToArray().ToList().IndexOf(actual[0].Item2)]);
        }
    }

    [Fact]
    public void A_tiny_radius_still_finds_exact_duplicates()
    {
        var a = Points(50, 4, "a");
        var joined = DaskFrame.FromFrame(a, 2).DistanceJoin(DaskFrame.FromFrame(a, 1), 1e-6, ("lat", "lon"), ("lat", "lon"), suffix: "_b").Compute();
        Assert.Equal(49, joined.RowCount);                    // every row with itself, the NaN row with nothing
        Assert.All(Enumerable.Range(0, 49), r => Assert.Equal(joined.Numeric("aid")[r], joined.Numeric("aid_b")[r]));
    }

    private static DataFrame Regions()
    {
        var wkt = new List<string?>();
        var names = new List<string?>();
        var random = new Random(9);
        for (var i = 0; i < 40; i++)
        {
            var x = random.NextDouble() * 300 - 150;
            var y = random.NextDouble() * 120 - 60;
            var w = 2 + random.NextDouble() * 30;
            var h = 2 + random.NextDouble() * 20;
            wkt.Add(i % 4 == 0
                ? FormattableString.Invariant($"POLYGON (({x} {y}, {x + w} {y}, {x + w / 2} {y + h}, {x} {y}))")
                : i % 4 == 1
                    ? FormattableString.Invariant($"POLYGON (({x} {y}, {x + w} {y}, {x + w} {y + h}, {x} {y + h}, {x} {y}), ({x + w / 4} {y + h / 4}, {x + w * 3 / 4} {y + h / 4}, {x + w * 3 / 4} {y + h * 3 / 4}, {x + w / 4} {y + h * 3 / 4}, {x + w / 4} {y + h / 4}))")
                    : Polygon.Box(x, y, x + w, y + h).ToWkt());
            names.Add("region" + i);
        }
        wkt.Add(null);                                     // a missing geometry matches nothing
        names.Add("nowhere");
        return new DataFrame([new TextSeries("region", [.. names]), new TextSeries("geometry", [.. wkt]),
            new NumericSeries("lat", Enumerable.Range(0, names.Count).Select(i => (double)i).ToArray())]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Point_in_polygon_join_equals_brute_force(bool keepUnmatched)
    {
        var points = Points(1200, 5, "p");
        var regions = Regions();
        var polygons = regions.Text("geometry").Values.ToArray().Select(w => w is null ? null : Polygon.Parse(w)).ToArray();
        var expected = new List<(double, string?)>();
        for (var i = 0; i < points.RowCount; i++)
        {
            var any = false;
            for (var j = 0; j < polygons.Length; j++)
                if (polygons[j] is { } p && !double.IsNaN(points.Numeric("lat")[i]) && p.Contains(points.Numeric("lon")[i], points.Numeric("lat")[i]))
                {
                    expected.Add((i, regions.Text("region")[j]));
                    any = true;
                }
            if (!any && keepUnmatched) expected.Add((i, null));
        }

        var joined = DaskFrame.FromFrame(points, 4).PointInPolygonJoin(DaskFrame.FromFrame(regions, 2), ("lat", "lon"), "geometry", keepUnmatched);
        Assert.Equal(["pid", "lat", "lon", "name", "region", "lat_right"], joined.Columns);
        await using var cluster = await LocalCluster.StartAsync(workers: 2, threadsPerWorker: 1);
        foreach (var result in new[] { joined.Compute(), joined.Compute(cluster.Client) })
        {
            var actual = Enumerable.Range(0, result.RowCount)
                .Select(r => (result.Numeric("pid")[r], result.Text("region")[r])).ToList();
            Assert.Equal(expected.OrderBy(e => e.Item1).ThenBy(e => e.Item2, StringComparer.Ordinal),
                actual.OrderBy(e => e.Item1).ThenBy(e => e.Item2, StringComparer.Ordinal));
            if (keepUnmatched) Assert.Contains(actual, a => a.Item2 is null);
        }
        Assert.Contains(expected, e => e.Item2 is not null);
    }

    [Fact]
    public void Spatial_partitioning_keeps_every_row_and_orders_partitions_by_geohash()
    {
        var points = Points(2000, 6, "p");
        var spatial = DaskFrame.FromFrame(points, 7).SpatialPartition("lat", "lon", partitions: 4, precision: 6);
        var parts = spatial.ComputePartitionsAsync().GetAwaiter().GetResult();
        Assert.Equal(points.RowCount, parts.Sum(p => p.RowCount));
        string? previousMax = null;
        foreach (var part in parts.Where(p => p.RowCount > 0))
        {
            var hashes = part.Text("geohash").Values.ToArray().Where(h => h is not null).Select(h => h!).ToArray();
            if (hashes.Length == 0) continue;
            var min = hashes.Min(StringComparer.Ordinal)!;
            if (previousMax is not null) Assert.True(string.CompareOrdinal(previousMax, min) <= 0, "partitions overlap in geohash");
            previousMax = hashes.Max(StringComparer.Ordinal);
            Assert.All(Enumerable.Range(0, part.RowCount).Where(r => part.Text("geohash")[r] is not null),
                r => Assert.Equal(Geohash.Encode(part.Numeric("lat")[r], part.Numeric("lon")[r], 6), part.Text("geohash")[r]));
        }
    }

    [Fact]
    public void Within_polygon_filters_like_contains()
    {
        var points = Points(800, 7, "p");
        var area = Polygon.Parse("POLYGON ((-30 -30, 60 -30, 60 50, -30 50, -30 -30), (0 0, 20 0, 20 20, 0 20, 0 0))");
        var kept = DaskFrame.FromFrame(points, 3).WithinPolygon("lat", "lon", area).Compute();
        var expected = Enumerable.Range(0, points.RowCount)
            .Where(i => !double.IsNaN(points.Numeric("lat")[i]) && area.Contains(points.Numeric("lon")[i], points.Numeric("lat")[i]))
            .Select(i => (double)i);
        Assert.Equal(expected, kept.Numeric("pid").Values.ToArray());
        Assert.Throws<ArgumentException>(() => DaskFrame.FromFrame(points, 1).WithinPolygon("latitude", "lon", area));
    }
}
