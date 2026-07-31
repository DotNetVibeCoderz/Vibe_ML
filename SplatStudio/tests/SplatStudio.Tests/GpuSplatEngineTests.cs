using System.Diagnostics;
using Microsoft.Extensions.Logging.Abstractions;
using SplatStudio.Application.Abstractions;
using SplatStudio.Domain.Enums;
using SplatStudio.Infrastructure.Splatting;
using Xunit.Abstractions;

namespace SplatStudio.Tests;

/// <summary>
/// Exercises the ILGPU path on real hardware. The GPU engine must stay behaviourally
/// interchangeable with the CPU engine — same heuristic, same output geometry — so most of
/// these tests assert equivalence rather than re-testing the maths.
/// </summary>
public class GpuSplatEngineTests
{
    private const int MaxPoints = 40_000;
    private readonly ITestOutputHelper _output;

    public GpuSplatEngineTests(ITestOutputHelper output) => _output = output;

    private static GpuSplatEngine CreateEngine() => new(NullLogger<GpuSplatEngine>.Instance);

    [Fact]
    public void Probe_reports_which_device_the_suite_will_use()
    {
        // Always runs, so the test log records what hardware was (or was not) found.
        _output.WriteLine($"GPU available: {GpuProbe.IsAvailable}");
        _output.WriteLine($"Device: {GpuProbe.Detail}");
    }

    [GpuFact]
    public void Engine_initialises_and_reports_its_device()
    {
        using var engine = CreateEngine();

        Assert.True(engine.IsAvailable);
        Assert.Equal(SplatEngineType.Gpu, engine.EngineType);
        Assert.NotEqual("unavailable", engine.DeviceName);
        _output.WriteLine($"Engine device: {engine.DeviceName}");
    }

    [GpuFact]
    public async Task Produces_a_well_formed_splat_file()
    {
        using var engine = CreateEngine();
        using var stream = new MemoryStream(SplatTestData.CreateTestJpeg());

        var result = await engine.GenerateAsync(stream, MaxPoints);

        Assert.True(result.Success, result.ErrorMessage);
        var points = SplatTestData.Decode(result.SplatFileBytes!);

        Assert.Equal(result.PointCount, points.Count);
        Assert.InRange(points.Count, MaxPoints / 2, MaxPoints);
        Assert.All(points, p =>
        {
            Assert.InRange(p.X, -1.001f, 1.001f);
            Assert.InRange(p.Y, -1.001f, 1.001f);
            Assert.InRange(p.Z, -0.301f, 0.301f);
        });
    }

    [GpuFact]
    public async Task Matches_the_cpu_engine_point_for_point()
    {
        var jpeg = SplatTestData.CreateTestJpeg();

        using var cpuStream = new MemoryStream(jpeg);
        var cpuResult = await new LocalHeuristicSplatEngine().GenerateAsync(cpuStream, MaxPoints);

        using var engine = CreateEngine();
        using var gpuStream = new MemoryStream(jpeg);
        var gpuResult = await engine.GenerateAsync(gpuStream, MaxPoints);

        Assert.True(cpuResult.Success && gpuResult.Success);
        Assert.Equal(cpuResult.PointCount, gpuResult.PointCount);

        var cpuPoints = SplatTestData.Decode(cpuResult.SplatFileBytes!);
        var gpuPoints = SplatTestData.Decode(gpuResult.SplatFileBytes!);

        // The GPU compacts its output with an atomic counter, so emission order differs from
        // the CPU's row-major order. Key by position to compare the clouds as sets.
        static (int, int) Key(DecodedSplat p) =>
            ((int)MathF.Round(p.X * 10_000f), (int)MathF.Round(p.Y * 10_000f));

        var cpuByPosition = cpuPoints.ToDictionary(Key);
        Assert.Equal(cpuPoints.Count, cpuByPosition.Count);

        foreach (var gpuPoint in gpuPoints)
        {
            Assert.True(cpuByPosition.TryGetValue(Key(gpuPoint), out var cpuPoint),
                $"GPU emitted a splat at ({gpuPoint.X}, {gpuPoint.Y}) that the CPU engine did not.");

            // Same colour exactly — colour is copied, not computed.
            Assert.Equal(cpuPoint.R, gpuPoint.R);
            Assert.Equal(cpuPoint.G, gpuPoint.G);
            Assert.Equal(cpuPoint.B, gpuPoint.B);
            Assert.Equal(cpuPoint.A, gpuPoint.A);

            // Depth is computed in floats on two different units, so allow a small epsilon.
            Assert.True(MathF.Abs(cpuPoint.Z - gpuPoint.Z) < 1e-4f,
                $"Depth mismatch at ({gpuPoint.X}, {gpuPoint.Y}): CPU {cpuPoint.Z} vs GPU {gpuPoint.Z}.");
            Assert.True(MathF.Abs(cpuPoint.ScaleX - gpuPoint.ScaleX) < 1e-6f);
        }
    }

    [GpuFact]
    public async Task Skips_fully_transparent_pixels_like_the_cpu_engine()
    {
        var png = SplatTestData.CreateTransparentBorderPng();

        using var cpuStream = new MemoryStream(png);
        var cpuResult = await new LocalHeuristicSplatEngine().GenerateAsync(cpuStream, MaxPoints);

        using var engine = CreateEngine();
        using var gpuStream = new MemoryStream(png);
        var gpuResult = await engine.GenerateAsync(gpuStream, MaxPoints);

        Assert.True(gpuResult.Success, gpuResult.ErrorMessage);
        Assert.Equal(cpuResult.PointCount, gpuResult.PointCount);
        Assert.All(SplatTestData.Decode(gpuResult.SplatFileBytes!), p => Assert.True(p.A >= 8));
    }

    [GpuFact]
    public async Task Handles_a_dense_point_budget_the_cpu_engine_would_struggle_with()
    {
        using var engine = CreateEngine();
        using var stream = new MemoryStream(SplatTestData.CreateTestJpeg(1024, 1024));

        const int denseBudget = 1_000_000;
        var sw = Stopwatch.StartNew();
        var result = await engine.GenerateAsync(stream, denseBudget);
        sw.Stop();

        Assert.True(result.Success, result.ErrorMessage);
        Assert.True(result.PointCount > 500_000,
            $"Expected a dense cloud, got {result.PointCount:N0} points.");
        _output.WriteLine($"{result.PointCount:N0} splats in {sw.ElapsedMilliseconds} ms " +
                          $"({result.SplatFileBytes!.Length / (1024.0 * 1024.0):F1} MB)");
    }

    [GpuFact]
    public async Task Invalid_image_data_fails_without_throwing()
    {
        using var engine = CreateEngine();
        using var stream = new MemoryStream(new byte[] { 9, 8, 7, 6 });

        var result = await engine.GenerateAsync(stream, MaxPoints);

        Assert.False(result.Success);
        Assert.NotNull(result.ErrorMessage);
    }

    /// <summary>
    /// Times both engines across the useful part of the budget range. Note that both share
    /// the same CPU-side JPEG decode and Lanczos resize, which is a fixed floor neither can
    /// beat — the GPU only accelerates the per-pixel depth/emit stage, so its advantage is
    /// invisible at small budgets and only shows up once point count dominates.
    /// </summary>
    [GpuFact]
    public async Task Benchmark_gpu_against_cpu()
    {
        var jpeg = SplatTestData.CreateTestJpeg(1024, 1024);
        using var engine = CreateEngine();
        var cpuEngine = new LocalHeuristicSplatEngine();

        // Warm up: the first call pays ILGPU's PTX JIT, which would otherwise dominate.
        using (var warm = new MemoryStream(jpeg))
            await engine.GenerateAsync(warm, MaxPoints);

        const int iterations = 5;
        // 262,144 is the CPU engine's hard ceiling (it clamps its working image to 512x512).
        int[] budgets = { 10_000, 40_000, 100_000, 262_144 };

        _output.WriteLine($"Device: {engine.DeviceName}, {iterations} iterations per budget");
        _output.WriteLine("");
        _output.WriteLine("  budget     points      CPU ms/img   GPU ms/img   speedup");
        _output.WriteLine("  ---------  ----------  -----------  -----------  -------");

        foreach (var budget in budgets)
        {
            var (cpuMs, cpuPoints) = await TimeAsync(cpuEngine, jpeg, budget, iterations);
            var (gpuMs, gpuPoints) = await TimeAsync(engine, jpeg, budget, iterations);

            Assert.Equal(cpuPoints, gpuPoints);
            _output.WriteLine($"  {budget,9:N0}  {gpuPoints,10:N0}  {cpuMs,11:F1}  {gpuMs,11:F1}  {cpuMs / gpuMs,6:F2}x");
        }

        // Beyond the CPU engine's ceiling the GPU is the only option at all.
        using var dense = new MemoryStream(jpeg);
        var denseSw = Stopwatch.StartNew();
        var denseResult = await engine.GenerateAsync(dense, 1_000_000);
        denseSw.Stop();
        _output.WriteLine("");
        _output.WriteLine($"  GPU-only:  {denseResult.PointCount,10:N0}  " +
                          $"{"n/a",11}  {denseSw.ElapsedMilliseconds,11:F1}  (CPU caps at 262,144)");
    }

    private static async Task<(double MsPerImage, int Points)> TimeAsync(
        IGaussianSplatEngine engine, byte[] jpeg, int budget, int iterations)
    {
        // One untimed pass so allocation/cache effects don't land on the first measurement.
        using (var warm = new MemoryStream(jpeg))
            await engine.GenerateAsync(warm, budget);

        var sw = Stopwatch.StartNew();
        int points = 0;
        for (int i = 0; i < iterations; i++)
        {
            using var s = new MemoryStream(jpeg);
            points = (await engine.GenerateAsync(s, budget)).PointCount;
        }
        sw.Stop();

        return (sw.Elapsed.TotalMilliseconds / iterations, points);
    }
}
