using Microsoft.Extensions.Logging;
using SplatStudio.Application.Abstractions;
using SplatStudio.Domain.Enums;

namespace SplatStudio.Infrastructure.Splatting;

/// <summary>
/// Mode 1 — the offline depth heuristic, wrapping whichever <see cref="IGaussianSplatEngine"/>
/// configuration selected (CPU or GPU). Always available: it is the reason the app works with
/// no credentials and no network.
/// </summary>
public class HeuristicConversionEngine : IConversionEngine
{
    private readonly IGaussianSplatEngine _engine;
    private readonly SplattingOptions _options;

    public HeuristicConversionEngine(IGaussianSplatEngine engine, SplattingOptions options)
    {
        _engine = engine;
        _options = options;
    }

    public ConversionMode Mode => ConversionMode.HeuristicSplat;
    public string DisplayName => "Depth-estimated splat";

    public string Description =>
        "Runs here, finishes in milliseconds, costs nothing. Estimates depth from the single " +
        "image, so it is an approximation rather than reconstruction — best with one clear subject.";

    public bool IsAvailable => true;
    public string? UnavailableReason => null;

    public async Task<ConversionOutput> ConvertAsync(Stream imageStream, CancellationToken ct = default)
    {
        var budget = _engine.EngineType == SplatEngineType.Gpu ? _options.GpuMaxPoints : _options.MaxPoints;
        var result = await _engine.GenerateAsync(imageStream, budget, ct);

        if (!result.Success || result.SplatFileBytes is null)
            return ConversionOutput.Failed(result.ErrorMessage ?? "Conversion failed.", _engine.EngineType);

        return new ConversionOutput(
            true, result.SplatFileBytes, ConversionArtifactKind.Splat,
            ".splat", "application/octet-stream", result.PointCount, _engine.EngineType, null);
    }
}

/// <summary>
/// Mode 2 — photorealistic Gaussian splatting through a hosted service. Produces a .splat, so
/// the existing point-cloud viewer renders it unchanged; the difference is upstream, where a
/// real reconstruction pipeline replaces the depth guess.
/// </summary>
public class HostedPhotorealConversionEngine : IConversionEngine
{
    private readonly HostedGenerationOptions _options;
    private readonly IHttpClientFactory _httpFactory;
    private readonly ILogger<HostedPhotorealConversionEngine> _logger;

    public HostedPhotorealConversionEngine(
        HostedEnginesOptions options,
        IHttpClientFactory httpFactory,
        ILogger<HostedPhotorealConversionEngine> logger)
    {
        _options = options.Photoreal;
        _httpFactory = httpFactory;
        _logger = logger;
    }

    public ConversionMode Mode => ConversionMode.PhotorealSplat;
    public string DisplayName => "Photorealistic splat";

    public string Description =>
        "Sends the image to a hosted 3D Gaussian splatting service for a real reconstruction. " +
        "Much higher fidelity than the depth heuristic, and takes minutes rather than milliseconds.";

    public bool IsAvailable => _options.IsConfigured;

    public string? UnavailableReason => _options.IsConfigured
        ? null
        : "Needs Splatting:Hosted:Photoreal:BaseUrl and :ApiKey to be set.";

    public async Task<ConversionOutput> ConvertAsync(Stream imageStream, CancellationToken ct = default)
    {
        if (!IsAvailable)
            return ConversionOutput.Failed(UnavailableReason!, SplatEngineType.HostedPhotoreal);

        try
        {
            var client = new HostedGenerationClient(_httpFactory.CreateClient(nameof(HostedGenerationClient)), _logger);
            var bytes = await client.GenerateAsync(imageStream, "upload.jpg", _options, ct);

            // The viewer parses fixed-size records, so a truncated or unexpected payload would
            // otherwise render as a silently wrong cloud.
            if (bytes.Length % 32 != 0)
                return ConversionOutput.Failed(
                    $"{_options.ProviderName} returned {bytes.Length} bytes, which is not a whole " +
                    "number of 32-byte splat records. Check that it is configured to emit .splat.",
                    SplatEngineType.HostedPhotoreal);

            return new ConversionOutput(
                true, bytes, ConversionArtifactKind.Splat,
                ".splat", "application/octet-stream", bytes.Length / 32, SplatEngineType.HostedPhotoreal, null);
        }
        catch (HostedGenerationException ex)
        {
            _logger.LogWarning(ex, "Photorealistic splat generation failed.");
            return ConversionOutput.Failed(ex.Message, SplatEngineType.HostedPhotoreal);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Photorealistic splat generation failed unexpectedly.");
            return ConversionOutput.Failed(
                "The photorealistic service could not be reached.", SplatEngineType.HostedPhotoreal);
        }
    }
}

/// <summary>
/// Mode 3 — image-to-3D-object generation (TRELLIS, Hunyuan3D, Rodin and similar) through a
/// hosted service. Unlike the other two modes this produces a textured mesh, not a point
/// cloud, so it is stored under a different prefix and rendered by the glTF viewer.
/// </summary>
public class HostedMeshConversionEngine : IConversionEngine
{
    private readonly HostedGenerationOptions _options;
    private readonly IHttpClientFactory _httpFactory;
    private readonly ILogger<HostedMeshConversionEngine> _logger;

    public HostedMeshConversionEngine(
        HostedEnginesOptions options,
        IHttpClientFactory httpFactory,
        ILogger<HostedMeshConversionEngine> logger)
    {
        _options = options.Mesh;
        _httpFactory = httpFactory;
        _logger = logger;
    }

    public ConversionMode Mode => ConversionMode.Mesh;
    public string DisplayName => "3D object (mesh)";

    public string Description =>
        "Generates a watertight textured mesh with a model such as TRELLIS, Hunyuan3D or Rodin. " +
        "Gives you geometry you can export and edit, rather than a point cloud.";

    public bool IsAvailable => _options.IsConfigured;

    public string? UnavailableReason => _options.IsConfigured
        ? null
        : "Needs Splatting:Hosted:Mesh:BaseUrl and :ApiKey to be set.";

    public async Task<ConversionOutput> ConvertAsync(Stream imageStream, CancellationToken ct = default)
    {
        if (!IsAvailable)
            return ConversionOutput.Failed(UnavailableReason!, SplatEngineType.HostedMesh);

        try
        {
            var client = new HostedGenerationClient(_httpFactory.CreateClient(nameof(HostedGenerationClient)), _logger);
            var bytes = await client.GenerateAsync(imageStream, "upload.jpg", _options, ct);

            if (!LooksLikeGlb(bytes))
                return ConversionOutput.Failed(
                    $"{_options.ProviderName} returned data that is not a binary glTF (.glb). " +
                    "Configure it to output GLB — that is the only mesh format this viewer loads.",
                    SplatEngineType.HostedMesh);

            // PointCount stays 0: a triangle count is a different measure and the UI labels
            // mesh scenes by file size instead.
            return new ConversionOutput(
                true, bytes, ConversionArtifactKind.Mesh,
                ".glb", "model/gltf-binary", 0, SplatEngineType.HostedMesh, null);
        }
        catch (HostedGenerationException ex)
        {
            _logger.LogWarning(ex, "Mesh generation failed.");
            return ConversionOutput.Failed(ex.Message, SplatEngineType.HostedMesh);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Mesh generation failed unexpectedly.");
            return ConversionOutput.Failed(
                "The 3D object service could not be reached.", SplatEngineType.HostedMesh);
        }
    }

    /// <summary>A .glb starts with the ASCII magic "glTF" followed by a version word.</summary>
    internal static bool LooksLikeGlb(byte[] bytes) =>
        bytes.Length >= 12 &&
        bytes[0] == 0x67 && bytes[1] == 0x6C && bytes[2] == 0x54 && bytes[3] == 0x46;
}

/// <summary>
/// Holds every registered mode. Ordered deliberately — the always-available local mode first,
/// so the upload page's default is the one that cannot fail.
/// </summary>
public class ConversionEngineCatalog : IConversionEngineCatalog
{
    public ConversionEngineCatalog(IEnumerable<IConversionEngine> engines)
    {
        Engines = engines.OrderBy(e => (int)e.Mode).ToList();
    }

    public IReadOnlyList<IConversionEngine> Engines { get; }

    public IConversionEngine? Resolve(ConversionMode mode) =>
        Engines.FirstOrDefault(e => e.Mode == mode);
}
