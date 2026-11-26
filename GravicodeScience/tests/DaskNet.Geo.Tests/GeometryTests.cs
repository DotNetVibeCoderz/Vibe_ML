using System.Globalization;
using System.Text.Json;
using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.DaskNet.Geo;
using Gravicode.Science.GraviFrame;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet.Geo;

/// <summary>
/// Buffer, simplify, dissolve and overlay pinned against shapely/GEOS
/// (<c>tools/verify/make_geometry_fixtures.py</c>), and against closed forms.
/// </summary>
public sealed class GeometryTests
{
    private static readonly JsonElement Fixture = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "geometry.json"))).RootElement;

    private static DataFrame Left()
    {
        var rows = Fixture.GetProperty("left").EnumerateArray().ToArray();
        return new DataFrame([
            new NumericSeries("id", rows.Select(r => r.GetProperty("id").GetDouble()).ToArray()),
            new TextSeries("zone", rows.Select(r => r.GetProperty("zone").GetString()).ToArray()),
            new TextSeries("wkt", rows.Select(r => r.GetProperty("wkt").GetString()).ToArray()),
        ]);
    }

    private static DataFrame Right()
    {
        var rows = Fixture.GetProperty("right").EnumerateArray().ToArray();
        return new DataFrame([
            new NumericSeries("rid", rows.Select(r => r.GetProperty("rid").GetDouble()).ToArray()),
            new TextSeries("wkt", rows.Select(r => r.GetProperty("wkt").GetString()).ToArray()),
        ]);
    }

    private static double[] Expected(string name) => Fixture.GetProperty(name).EnumerateArray().Select(e => e.GetDouble()).ToArray();

    private static void Close(double expected, double actual, string what, double relative = 1e-9)
        => Assert.True(Math.Abs(expected - actual) <= relative * Math.Max(1, Math.Abs(expected)), $"{what}: GEOS {expected}, NTS {actual}");

    private static Dictionary<double, double> AreaById(DataFrame f)
    {
        var ids = f.Numeric("id").Values;
        var area = f.Numeric("area").Values;
        return Enumerable.Range(0, f.RowCount).ToDictionary(i => ids[i], i => double.IsNaN(area[i]) ? 0 : area[i]);
    }

    [Fact]
    public void Buffer_shrink_and_simplify_match_geos()
    {
        var frame = DaskFrame.FromFrame(Left(), 4);
        // GEOS and NTS approximate a round join differently (both with 8 segments a quadrant), so
        // buffered areas agree only to the arc-approximation error: at most the area a 32-gon
        // misses of its circle, pi r^2 - 16 r^2 sin(pi / 16). Measured worst: 1.8e-4 relative for
        // r = 0.3, 3.3e-5 for -0.2. Simplify is the same algorithm in both and agrees to 1e-9.
        static double Deficit(double r) => r * r * (Math.PI - 16 * Math.Sin(Math.PI / 16));
        foreach (var (name, op, tolerance) in new (string, Func<DaskFrame, DaskFrame>, double)[]
        {
            ("buffer", f => f.Buffer("wkt", 0.3), Deficit(0.3)),
            ("shrink", f => f.Buffer("wkt", -0.2), Deficit(0.2)),
            ("simplify", f => f.Simplify("wkt", 0.25), 0),
        })
        {
            var areas = AreaById(op(frame).WithMeasures("wkt").Compute());
            var expected = Expected(name);
            for (var i = 0; i < expected.Length; i++)
                if (tolerance > 0) Assert.True(Math.Abs(expected[i] - areas[i]) <= tolerance, $"{name} area of {i}: GEOS {expected[i]}, NTS {areas[i]}");
                else Close(expected[i], areas[i], $"{name} area of {i}");
        }

        var centroids = frame.Centroid("wkt", output: "c").Compute();
        var expectedCentroids = Fixture.GetProperty("centroid").EnumerateArray().ToArray();
        for (var i = 0; i < centroids.RowCount; i++)
        {
            var xy = ((string)centroids["c"].GetValue(i)!)["POINT (".Length..^1].Split(' ').Select(v => double.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            Close(expectedCentroids[i][0].GetDouble(), xy[0], $"centroid x of {i}");
            Close(expectedCentroids[i][1].GetDouble(), xy[1], $"centroid y of {i}");
        }
        Assert.Equal(["id", "zone", "wkt", "c"], centroids.ColumnNames);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(16)]
    public void Dissolve_matches_geos_whatever_the_partitioning(int partitions)
    {
        var frame = DaskFrame.FromFrame(Left(), partitions);
        var dissolved = frame.Dissolve("wkt", by: ["zone"]).WithMeasures("wkt").Compute();
        Assert.Equal(["zone", "wkt", "count", "area", "length"], dissolved.ColumnNames);
        Assert.Equal(3, dissolved.RowCount);
        var expected = Fixture.GetProperty("dissolve");
        for (var i = 0; i < dissolved.RowCount; i++)
        {
            var zone = (string)dissolved["zone"].GetValue(i)!;
            Close(expected.GetProperty(zone).GetProperty("area").GetDouble(), dissolved.Numeric("area")[i], $"area of {zone}");
            Close(expected.GetProperty(zone).GetProperty("length").GetDouble(), dissolved.Numeric("length")[i], $"length of {zone}");
            Assert.Equal(expected.GetProperty(zone).GetProperty("count").GetDouble(), dissolved.Numeric("count")[i]);
        }

        var all = frame.Dissolve("wkt").WithMeasures("wkt").Compute();
        Assert.Equal(1, all.RowCount);
        Close(Fixture.GetProperty("union_area").GetDouble(), all.Numeric("area")[0], "area of the whole union");
        Assert.Equal(60, all.Numeric("count")[0]);
    }

    [Fact]
    public async Task Overlay_matches_geos_locally_and_on_a_cluster()
    {
        var left = DaskFrame.FromFrame(Left(), 6);
        var right = DaskFrame.FromFrame(Right(), 2);
        var expected = Fixture.GetProperty("intersection").EnumerateArray()
            .ToDictionary(p => (p.GetProperty("id").GetDouble(), p.GetProperty("rid").GetDouble()), p => p.GetProperty("area").GetDouble());

        await using var cluster = await LocalCluster.StartAsync(workers: 2, threadsPerWorker: 2);
        foreach (var scheduler in new IScheduler?[] { null, cluster.Client })
        {
            var shared = left.Overlay(right, "wkt", "wkt").WithMeasures("wkt").Compute(scheduler);
            Assert.Equal(["id", "zone", "wkt", "rid", "area", "length"], shared.ColumnNames);
            var got = Enumerable.Range(0, shared.RowCount)
                .ToDictionary(i => (shared.Numeric("id")[i], shared.Numeric("rid")[i]), i => shared.Numeric("area")[i]);
            Assert.Equal(expected.Keys.Order(), got.Keys.Order());
            foreach (var (pair, area) in expected) Close(area, got[pair], $"intersection {pair}");

            var rest = left.Overlay(right, "wkt", "wkt", OverlayKind.Difference).WithMeasures("wkt").Compute(scheduler);
            var restArea = AreaById(rest);
            foreach (var d in Fixture.GetProperty("difference").EnumerateArray())
            {
                var id = d.GetProperty("id").GetDouble();
                if (d.GetProperty("empty").GetBoolean()) Assert.False(restArea.ContainsKey(id), $"{id} should be gone");
                else Close(d.GetProperty("area").GetDouble(), restArea[id], $"difference of {id}");
            }
        }
    }

    [Fact]
    public void Closed_forms_and_edge_cases()
    {
        // Two unit squares overlapping by half; a third only touching the first along an edge.
        var a = new DataFrame([new TextSeries("name", ["a", "b"]), new TextSeries("g", ["POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))", null])]);
        var b = new DataFrame([new NumericSeries("k", [1, 2]), new TextSeries("g", ["POLYGON ((1 0, 3 0, 3 2, 1 2, 1 0))", "POLYGON ((-1 0, 0 0, 0 2, -1 2, -1 0))"])]);
        var shared = DaskFrame.FromFrame(a, 1).Overlay(DaskFrame.FromFrame(b, 1), "g", "g").WithMeasures("g").Compute();
        Assert.Equal(1, shared.RowCount);                  // the touching square shares no area
        Assert.Equal(2.0, shared.Numeric("area")[0], 12);
        var rest = DaskFrame.FromFrame(a, 1).Overlay(DaskFrame.FromFrame(b, 1), "g", "g", OverlayKind.Difference).WithMeasures("g").Compute();
        Assert.Equal(2, rest.RowCount);                    // a missing geometry stays, missing
        Assert.Equal(2.0, rest.Numeric("area")[0], 12);
        Assert.True(rest["g"].IsMissing(1));

        // A buffered point is a 32-gon: area n r^2 sin(2 pi / n) / 2 for n = 4 x quadrant segments.
        var point = DaskFrame.FromFrame(new DataFrame([new TextSeries("g", ["POINT (5 5)"])]), 1);
        var circle = point.Buffer("g", 2.0).WithMeasures("g").Compute();
        Assert.Equal(32 * 4.0 * Math.Sin(2 * Math.PI / 32) / 2, circle.Numeric("area")[0], 10);

        // A bow-tie is invalid; it is repaired into its two triangles (area 0.5 each), not refused.
        var bowtie = DaskFrame.FromFrame(new DataFrame([new TextSeries("g", ["POLYGON ((0 0, 1 1, 1 0, 0 1, 0 0))"])]), 1);
        Assert.Equal(0.5, bowtie.WithMeasures("g").Compute().Numeric("area")[0], 12);
        Assert.Throws<ArgumentException>(() => point.Buffer("nope", 1));
    }
}
