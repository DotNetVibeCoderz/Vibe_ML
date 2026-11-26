using System.Globalization;
using System.Text.Json;
using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.DaskNet.Geo;
using Gravicode.Science.GraviFrame;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet.Geo;

/// <summary>
/// Metric geometry pinned against PROJ (the same local transverse Mercator, Web Mercator, UTM)
/// and against geodesic areas and lengths on the ellipsoid (GeographicLib), both through pyproj
/// in <c>tools/verify/make_crs_fixtures.py</c>.
/// </summary>
public sealed class CrsTests
{
    private static readonly JsonElement Fixture = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "crs.json"))).RootElement;
    private static JsonElement[] Shapes => Fixture.GetProperty("shapes").EnumerateArray().ToArray();

    private static DataFrame Frame() => new([
        new NumericSeries("id", Shapes.Select((_, i) => (double)i).ToArray()),
        new TextSeries("wkt", Shapes.Select(s => s.GetProperty("wkt").GetString()).ToArray()),
    ]);

    private static void Close(double expected, double actual, double relative, string what)
        => Assert.True(Math.Abs(expected - actual) <= relative * Math.Max(1, Math.Abs(expected)), $"{what}: expected {expected}, got {actual} (rel {Math.Abs(expected - actual) / Math.Abs(expected):E2})");

    private static double[] Point(object? wkt)
        => ((string)wkt!)["POINT (".Length..^1].Split(' ').Select(v => double.Parse(v, CultureInfo.InvariantCulture)).ToArray();

    [Fact]
    public async Task Areas_and_lengths_are_ground_metres_locally_and_on_a_cluster()
    {
        var frame = DaskFrame.FromFrame(Frame(), 3);
        await using var cluster = await LocalCluster.StartAsync(workers: 2, threadsPerWorker: 2);
        foreach (var scheduler in new IScheduler?[] { null, cluster.Client })
        {
            var measured = frame.WithMeasures("wkt", crs: GeoCrs.Wgs84).Compute(scheduler);
            for (var i = 0; i < Shapes.Length; i++)
            {
                var s = Shapes[i];
                var name = s.GetProperty("name").GetString()!;
                var area = measured.Numeric("area")[i];
                var length = measured.Numeric("length")[i];
                // The same projection as PROJ's tmerc: nanometre agreement, so 1e-9 relative.
                Close(s.GetProperty("area").GetDouble(), area, 1e-9, $"{name} area vs PROJ");
                Close(s.GetProperty("length").GetDouble(), length, 1e-9, $"{name} length vs PROJ");
                // Against the ellipsoid: the local plane's scale error grows as (extent / R)^2 / 2,
                // under 3e-7 for the city shapes and 3.5e-5 for the 90 km region.
                var geodesicArea = s.GetProperty("geodesic_area").GetDouble();
                var bound = name == "bandung_region" ? 5e-5 : 1e-6;
                if (geodesicArea > 0) Close(geodesicArea, area, bound, $"{name} area vs geodesic");
                if (name != "london_ring")                      // pyproj's perimeter leaves out holes
                    Close(s.GetProperty("geodesic_length").GetDouble(), length, bound, $"{name} length vs geodesic");
            }
            // Without a system the same column is measured in square degrees.
            var planar = frame.WithMeasures("wkt").Compute(scheduler);
            Assert.True(planar.Numeric("area")[0] < 1e-2);
        }
    }

    [Fact]
    public void Buffer_simplify_and_centroid_work_in_metres()
    {
        var frame = DaskFrame.FromFrame(Frame(), 2);
        // GEOS and NTS approximate round joins differently: they may differ by at most the area a
        // 32-gon misses of its circle (see GeometryTests). Everything else is the projection.
        const double r = 500;
        var deficit = r * r * (Math.PI - 16 * Math.Sin(Math.PI / 16));
        var buffered = frame.Buffer("wkt", r, crs: GeoCrs.Wgs84).WithMeasures("wkt", crs: GeoCrs.Wgs84).Compute();
        var simplified = frame.Simplify("wkt", 150, crs: GeoCrs.Wgs84).WithMeasures("wkt", crs: GeoCrs.Wgs84).Compute();
        var centroids = frame.Centroid("wkt", output: "c", crs: GeoCrs.Wgs84).Compute();
        for (var i = 0; i < Shapes.Length; i++)
        {
            var s = Shapes[i];
            var name = s.GetProperty("name").GetString()!;
            var expected = s.GetProperty("buffer_500").GetProperty("area").GetDouble();
            var got = buffered.Numeric("area")[i];
            Assert.True(Math.Abs(expected - got) <= deficit + 1e-7 * expected, $"{name} buffer: GEOS {expected}, NTS {got}");
            // A metre buffer in degrees would have added about 0 square metres; this must add roughly perimeter * r.
            Assert.True(got - s.GetProperty("area").GetDouble() > 0.9 * r * s.GetProperty("length").GetDouble(), $"{name} buffer grew too little");

            Close(s.GetProperty("simplify_150").GetProperty("area").GetDouble(), simplified.Numeric("area")[i], 1e-8, $"{name} simplified area");
            Close(s.GetProperty("simplify_150").GetProperty("length").GetDouble(), simplified.Numeric("length")[i], 1e-8, $"{name} simplified length");

            var c = Point(centroids["c"].GetValue(i));
            var e = s.GetProperty("centroid");
            Assert.True(Math.Abs(c[0] - e[0].GetDouble()) < 1e-9 && Math.Abs(c[1] - e[1].GetDouble()) < 1e-9,
                $"{name} centroid ({c[0]}, {c[1]}) vs ({e[0]}, {e[1]})");
        }
    }

    [Fact]
    public void To_crs_matches_proj_for_utm_and_web_mercator_and_round_trips()
    {
        var conversions = Fixture.GetProperty("conversions").EnumerateArray().ToArray();
        var points = new DataFrame([new TextSeries("wkt", conversions.Select(c =>
            string.Create(CultureInfo.InvariantCulture, $"POINT ({c.GetProperty("lon").GetDouble():R} {c.GetProperty("lat").GetDouble():R})")).ToArray())]);
        var frame = DaskFrame.FromFrame(points, 2);

        var web = frame.ToCrs("wkt", GeoCrs.Wgs84, GeoCrs.WebMercator, output: "web").Compute();
        var back = frame.ToCrs("wkt", GeoCrs.Wgs84, GeoCrs.WebMercator).ToCrs("wkt", GeoCrs.WebMercator, GeoCrs.Wgs84).Compute();
        for (var i = 0; i < conversions.Length; i++)
        {
            var c = conversions[i];
            var expected = c.GetProperty("web");
            var got = Point(web["web"].GetValue(i));
            Assert.True(Math.Abs(got[0] - expected[0].GetDouble()) < 1e-6 && Math.Abs(got[1] - expected[1].GetDouble()) < 1e-6, $"web mercator of point {i}");

            var zone = GeoCrs.FromEpsg(c.GetProperty("utm_epsg").GetInt32());
            Assert.Equal(zone, GeoCrs.UtmFor(c.GetProperty("lon").GetDouble(), c.GetProperty("lat").GetDouble()));
            var one = DaskFrame.FromFrame(new DataFrame([new TextSeries("wkt", [(string)points["wkt"].GetValue(i)!])]), 1);
            var utm = Point(one.ToCrs("wkt", GeoCrs.Wgs84, zone).Compute()["wkt"].GetValue(0));
            var u = c.GetProperty("utm");
            Assert.True(Math.Abs(utm[0] - u[0].GetDouble()) < 1e-6 && Math.Abs(utm[1] - u[1].GetDouble()) < 1e-6,
                $"UTM {zone} of point {i}: PROJ ({u[0]}, {u[1]}), got ({utm[0]}, {utm[1]})");
            var home = Point(one.ToCrs("wkt", GeoCrs.Wgs84, zone).ToCrs("wkt", zone, GeoCrs.Wgs84).Compute()["wkt"].GetValue(0));
            Assert.True(Math.Abs(home[0] - c.GetProperty("lon").GetDouble()) < 1e-9 && Math.Abs(home[1] - c.GetProperty("lat").GetDouble()) < 1e-9, $"UTM round trip of point {i}");

            var round = Point(back["wkt"].GetValue(i));
            Assert.True(Math.Abs(round[0] - c.GetProperty("lon").GetDouble()) < 1e-9 && Math.Abs(round[1] - c.GetProperty("lat").GetDouble()) < 1e-9, $"web round trip of point {i}");
        }
    }

    [Fact]
    public void Projected_columns_measure_the_same_ground()
    {
        // The Jakarta shape in UTM 48S and in Web Mercator must measure what it measures in lon/lat:
        // the operation goes back to the ground, whatever system the column is in.
        var frame = DaskFrame.FromFrame(new DataFrame([new TextSeries("wkt", [Shapes[0].GetProperty("wkt").GetString()])]), 1);
        var expected = Shapes[0].GetProperty("area").GetDouble();
        var utm = GeoCrs.Utm(48, north: false);
        foreach (var crs in new[] { utm, GeoCrs.WebMercator })
        {
            var area = frame.ToCrs("wkt", GeoCrs.Wgs84, crs).WithMeasures("wkt", crs: crs).Compute().Numeric("area")[0];
            Close(expected, area, 1e-9, $"area through {crs}");
            var buffered = frame.ToCrs("wkt", GeoCrs.Wgs84, crs).Buffer("wkt", 200, crs: crs).ToCrs("wkt", crs, GeoCrs.Wgs84)
                .WithMeasures("wkt", crs: GeoCrs.Wgs84).Compute().Numeric("area")[0];
            var direct = frame.Buffer("wkt", 200, crs: GeoCrs.Wgs84).WithMeasures("wkt", crs: GeoCrs.Wgs84).Compute().Numeric("area")[0];
            Close(direct, buffered, 1e-8, $"buffer through {crs}");
        }
        // Planar UTM area differs from the ground by the zone's scale factor, about k0^2 here.
        var planar = frame.ToCrs("wkt", GeoCrs.Wgs84, utm).WithMeasures("wkt").Compute().Numeric("area")[0];
        Assert.True(Math.Abs(planar / expected - 1) > 1e-4, "UTM planar area should carry the zone's scale error");
    }

    [Fact]
    public void Systems_parse_and_refuse_what_they_cannot_be()
    {
        Assert.Equal(GeoCrs.Wgs84, GeoCrs.Parse("EPSG:4326"));
        Assert.Equal(GeoCrs.WebMercator, GeoCrs.Parse("epsg:3857"));
        Assert.Equal(GeoCrs.WebMercator, GeoCrs.FromEpsg(900913));
        Assert.Equal(32748, GeoCrs.Parse("32748").Epsg);
        Assert.Equal("EPSG:32633", GeoCrs.Utm(33).ToString());
        Assert.True(GeoCrs.Wgs84.IsGeographic);
        Assert.False(GeoCrs.Utm(1, north: false).IsGeographic);
        Assert.Throws<NotSupportedException>(() => GeoCrs.FromEpsg(2154));
        Assert.Throws<FormatException>(() => GeoCrs.Parse("lambert"));
        Assert.Throws<ArgumentOutOfRangeException>(() => GeoCrs.Utm(61));
        Assert.Equal(GeoCrs.Utm(1), GeoCrs.UtmFor(180.5, 10));       // longitudes wrap

        // Metres in a column that is not longitude/latitude is a declaration error, not a silent answer.
        var wrong = DaskFrame.FromFrame(new DataFrame([new TextSeries("wkt", ["POLYGON ((0 0, 500 0, 500 500, 0 500, 0 0))"])]), 1);
        var error = Assert.ThrowsAny<Exception>(() => wrong.WithMeasures("wkt", crs: GeoCrs.Wgs84).Compute());
        Assert.Contains("outside [-90, 90]", error.ToString());

        var withMissing = DaskFrame.FromFrame(new DataFrame([new TextSeries("wkt", [null, Shapes[0].GetProperty("wkt").GetString()])]), 1);
        var m = withMissing.WithMeasures("wkt", crs: GeoCrs.Wgs84).Compute();
        Assert.True(double.IsNaN(m.Numeric("area")[0]));
        Assert.Null(withMissing.Buffer("wkt", 10, crs: GeoCrs.Wgs84).Compute()["wkt"].GetValue(0));
    }
}
