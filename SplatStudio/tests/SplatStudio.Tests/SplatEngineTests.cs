using SplatStudio.Domain.Enums;
using SplatStudio.Infrastructure.Splatting;

namespace SplatStudio.Tests;

public class SplatEngineTests
{
    private const int MaxPoints = 40_000;

    [Fact]
    public async Task LocalHeuristic_produces_a_well_formed_splat_file()
    {
        var engine = new LocalHeuristicSplatEngine();
        using var stream = new MemoryStream(SplatTestData.CreateTestJpeg());

        var result = await engine.GenerateAsync(stream, MaxPoints);

        Assert.True(result.Success, result.ErrorMessage);
        Assert.NotNull(result.SplatFileBytes);
        Assert.Equal(SplatEngineType.LocalHeuristic, engine.EngineType);

        var points = SplatTestData.Decode(result.SplatFileBytes!);
        Assert.Equal(result.PointCount, points.Count);
        Assert.Equal(points.Count * SplatTestData.BytesPerPoint, result.SplatFileBytes!.Length);

        // The point budget is a ceiling, and a budget this size should be nearly filled.
        Assert.InRange(points.Count, MaxPoints / 2, MaxPoints);
    }

    [Fact]
    public async Task Generated_points_stay_inside_the_viewer_world_box()
    {
        var engine = new LocalHeuristicSplatEngine();
        using var stream = new MemoryStream(SplatTestData.CreateTestJpeg());
        var result = await engine.GenerateAsync(stream, MaxPoints);

        var points = SplatTestData.Decode(result.SplatFileBytes!);

        // X/Y are normalised to [-1,1]; Z is the pseudo-depth mapped to +/- half of the
        // 0.6-unit depth range. A regression in the depth maths shows up here first.
        Assert.All(points, p =>
        {
            Assert.InRange(p.X, -1.001f, 1.001f);
            Assert.InRange(p.Y, -1.001f, 1.001f);
            Assert.InRange(p.Z, -0.301f, 0.301f);
            Assert.True(p.ScaleX > 0 && p.ScaleY > 0 && p.ScaleZ > 0, "Scales must be positive.");
        });

        // Identity rotation, in the format's 0..255 encoding.
        Assert.All(points, p =>
        {
            Assert.Equal(128, p.RotX);
            Assert.Equal(128, p.RotY);
            Assert.Equal(128, p.RotZ);
            Assert.Equal(255, p.RotW);
        });
    }

    [Fact]
    public async Task Depth_varies_across_the_image_rather_than_being_flat()
    {
        var engine = new LocalHeuristicSplatEngine();
        using var stream = new MemoryStream(SplatTestData.CreateTestJpeg());
        var result = await engine.GenerateAsync(stream, MaxPoints);

        var points = SplatTestData.Decode(result.SplatFileBytes!);
        var spread = points.Max(p => p.Z) - points.Min(p => p.Z);

        // A flat cloud would mean the heuristic silently degenerated into a billboard.
        Assert.True(spread > 0.1f, $"Expected real depth variation, got a spread of {spread:F4}.");
    }

    [Fact]
    public async Task Fully_transparent_pixels_are_skipped()
    {
        var engine = new LocalHeuristicSplatEngine();
        using var stream = new MemoryStream(SplatTestData.CreateTransparentBorderPng());
        var result = await engine.GenerateAsync(stream, MaxPoints);

        Assert.True(result.Success, result.ErrorMessage);
        var points = SplatTestData.Decode(result.SplatFileBytes!);

        Assert.All(points, p => Assert.True(p.A >= 8, "Transparent pixels must not become splats."));
        // The opaque square covers the middle half in each axis, i.e. a quarter of the area.
        Assert.True(points.Count > 0);
        Assert.True(points.Count < MaxPoints / 2,
            $"Expected the transparent border to be dropped, but got {points.Count} points.");
    }

    [Fact]
    public async Task Invalid_image_data_fails_without_throwing()
    {
        var engine = new LocalHeuristicSplatEngine();
        using var stream = new MemoryStream(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 });

        var result = await engine.GenerateAsync(stream, MaxPoints);

        // The background worker relies on failures arriving as a result, not an exception.
        Assert.False(result.Success);
        Assert.NotNull(result.ErrorMessage);
        Assert.Equal(0, result.PointCount);
    }

    [Fact]
    public async Task Output_is_deterministic_for_the_same_input()
    {
        var engine = new LocalHeuristicSplatEngine();
        var jpeg = SplatTestData.CreateTestJpeg();

        using var first = new MemoryStream(jpeg);
        using var second = new MemoryStream(jpeg);
        var a = await engine.GenerateAsync(first, MaxPoints);
        var b = await engine.GenerateAsync(second, MaxPoints);

        Assert.Equal(a.SplatFileBytes, b.SplatFileBytes);
    }
}
