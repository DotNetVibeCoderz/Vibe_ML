using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using SplatStudio.Application.Abstractions;
using SplatStudio.Domain.Enums;
using SplatStudio.Infrastructure.Splatting;

namespace SplatStudio.Tests;

/// <summary>
/// Covers the mode-selection layer: the catalogue the upload page reads, the availability
/// gate that keeps unconfigured modes from accepting work, and the response parsing the
/// hosted client depends on.
/// </summary>
public class ConversionModeTests
{
    private static HostedEnginesOptions UnconfiguredHosted() => new();

    private static IConversionEngineCatalog BuildCatalog(HostedEnginesOptions hosted)
    {
        var splatting = new SplattingOptions();
        var httpFactory = new StubHttpClientFactory();

        return new ConversionEngineCatalog(new IConversionEngine[]
        {
            new HeuristicConversionEngine(new LocalHeuristicSplatEngine(), splatting),
            new HostedPhotorealConversionEngine(hosted, httpFactory, NullLogger<HostedPhotorealConversionEngine>.Instance),
            new HostedMeshConversionEngine(hosted, httpFactory, NullLogger<HostedMeshConversionEngine>.Instance)
        });
    }

    [Fact]
    public void Catalog_exposes_all_three_modes_in_a_stable_order()
    {
        var catalog = BuildCatalog(UnconfiguredHosted());

        Assert.Equal(
            new[] { ConversionMode.HeuristicSplat, ConversionMode.PhotorealSplat, ConversionMode.Mesh },
            catalog.Engines.Select(e => e.Mode));
    }

    [Fact]
    public void Catalog_resolves_each_mode_to_its_engine()
    {
        var catalog = BuildCatalog(UnconfiguredHosted());

        foreach (var mode in Enum.GetValues<ConversionMode>())
            Assert.Equal(mode, catalog.Resolve(mode)?.Mode);
    }

    [Fact]
    public void Local_mode_is_always_available()
    {
        var engine = BuildCatalog(UnconfiguredHosted()).Resolve(ConversionMode.HeuristicSplat)!;

        // This is the guarantee that makes the app usable with no credentials at all.
        Assert.True(engine.IsAvailable);
        Assert.Null(engine.UnavailableReason);
    }

    [Theory]
    [InlineData(ConversionMode.PhotorealSplat)]
    [InlineData(ConversionMode.Mesh)]
    public void Hosted_modes_are_unavailable_until_configured_and_say_why(ConversionMode mode)
    {
        var engine = BuildCatalog(UnconfiguredHosted()).Resolve(mode)!;

        Assert.False(engine.IsAvailable);
        Assert.False(string.IsNullOrWhiteSpace(engine.UnavailableReason));
        // The reason has to name the setting, or it is not actionable.
        Assert.Contains("Splatting:Hosted", engine.UnavailableReason);
    }

    [Theory]
    [InlineData(ConversionMode.PhotorealSplat)]
    [InlineData(ConversionMode.Mesh)]
    public async Task Unconfigured_hosted_mode_fails_without_calling_out(ConversionMode mode)
    {
        var engine = BuildCatalog(UnconfiguredHosted()).Resolve(mode)!;
        using var stream = new MemoryStream(SplatTestData.CreateTestJpeg(64, 64));

        var result = await engine.ConvertAsync(stream);

        Assert.False(result.Success);
        Assert.Equal(engine.UnavailableReason, result.ErrorMessage);
    }

    [Fact]
    public void Hosted_modes_become_available_once_url_and_key_are_set()
    {
        var hosted = new HostedEnginesOptions();
        hosted.Photoreal.BaseUrl = "https://api.example.com";
        hosted.Photoreal.ApiKey = "secret";

        var catalog = BuildCatalog(hosted);

        Assert.True(catalog.Resolve(ConversionMode.PhotorealSplat)!.IsAvailable);
        // Mesh is configured separately and must not be switched on by the other one.
        Assert.False(catalog.Resolve(ConversionMode.Mesh)!.IsAvailable);
    }

    [Fact]
    public async Task Local_mode_reports_a_splat_artifact_the_pipeline_can_store()
    {
        var engine = BuildCatalog(UnconfiguredHosted()).Resolve(ConversionMode.HeuristicSplat)!;
        using var stream = new MemoryStream(SplatTestData.CreateTestJpeg());

        var result = await engine.ConvertAsync(stream);

        Assert.True(result.Success, result.ErrorMessage);
        Assert.Equal(ConversionArtifactKind.Splat, result.Kind);
        Assert.Equal(".splat", result.FileExtension);
        Assert.Equal("application/octet-stream", result.ContentType);
        Assert.NotNull(result.Content);
        Assert.Equal(result.PointCount * 32, result.Content!.Length);
    }

    // ---- GLB sniffing --------------------------------------------------------

    [Fact]
    public void Glb_detection_accepts_the_gltf_magic_and_rejects_anything_else()
    {
        // "glTF" followed by version and length words.
        var glb = new byte[] { 0x67, 0x6C, 0x54, 0x46, 2, 0, 0, 0, 0, 0, 0, 0 };
        Assert.True(HostedMeshConversionEngine.LooksLikeGlb(glb));

        // A provider misconfigured to return .obj/.ply text, or an error page, must not be
        // stored as a model — the viewer would fail with no explanation.
        Assert.False(HostedMeshConversionEngine.LooksLikeGlb("<!doctype html>"u8.ToArray()));
        Assert.False(HostedMeshConversionEngine.LooksLikeGlb(new byte[] { 0x67, 0x6C }));
        Assert.False(HostedMeshConversionEngine.LooksLikeGlb(Array.Empty<byte>()));
    }

    // ---- JSON path resolution ------------------------------------------------

    [Theory]
    [InlineData("job_id", "abc123")]
    [InlineData("data.job.id", "nested")]
    [InlineData("results.0.url", "https://cdn.example.com/a.glb")]
    [InlineData("count", "42")]
    [InlineData("done", "true")]
    public void Json_paths_resolve_against_provider_responses(string path, string expected)
    {
        const string body = """
        {
          "job_id": "abc123",
          "data": { "job": { "id": "nested" } },
          "results": [ { "url": "https://cdn.example.com/a.glb" } ],
          "count": 42,
          "done": true
        }
        """;

        using var doc = JsonDocument.Parse(body);
        Assert.Equal(expected, HostedGenerationClient.ReadPath(doc.RootElement, path));
    }

    [Theory]
    [InlineData("missing")]
    [InlineData("data.nope")]
    [InlineData("results.9.url")]
    [InlineData("")]
    public void Missing_json_paths_return_null_rather_than_throwing(string path)
    {
        const string body = """{ "data": { "job": { "id": "x" } }, "results": [] }""";

        using var doc = JsonDocument.Parse(body);
        // The client turns null into a clear "check your path configuration" error; an
        // exception here would surface as an unhandled worker crash instead.
        Assert.Null(HostedGenerationClient.ReadPath(doc.RootElement, path));
    }

    private sealed class StubHttpClientFactory : IHttpClientFactory
    {
        public HttpClient CreateClient(string name) => new();
    }
}
