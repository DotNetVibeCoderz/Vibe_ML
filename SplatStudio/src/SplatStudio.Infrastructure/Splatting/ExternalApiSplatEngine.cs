using SplatStudio.Application.Abstractions;
using SplatStudio.Domain.Enums;

namespace SplatStudio.Infrastructure.Splatting;

public class ExternalSplatEngineOptions
{
    public const string SectionName = "Splatting:ExternalApi";

    /// <summary>POST endpoint that accepts an image and returns a binary .splat file.</summary>
    public string Endpoint { get; set; } = string.Empty;
    public string? ApiKey { get; set; }
    public int TimeoutSeconds { get; set; } = 120;
}

/// <summary>
/// Integration point for a real, multi-view-capable 3D Gaussian Splatting
/// backend (e.g. a self-hosted nerfstudio/gsplat training service, or a
/// hosted API such as Luma AI / KIRI Engine). Posts the source image and
/// expects the raw 32-byte-per-point ".splat" payload back; swap the body
/// of <see cref="GenerateAsync"/> for whatever contract your chosen
/// provider actually exposes (most return a job id you poll, rather than
/// a synchronous response — this class is a starting point, not a
/// finished SDK).
///
/// To activate: set "Splatting:Engine": "ExternalApi" in appsettings.json
/// and fill in "Splatting:ExternalApi:Endpoint" / ApiKey.
/// </summary>
public class ExternalApiSplatEngine : IGaussianSplatEngine
{
    private readonly HttpClient _httpClient;
    private readonly ExternalSplatEngineOptions _options;

    public SplatEngineType EngineType => SplatEngineType.ExternalApi;

    public ExternalApiSplatEngine(HttpClient httpClient, ExternalSplatEngineOptions options)
    {
        _httpClient = httpClient;
        _options = options;
        _httpClient.Timeout = TimeSpan.FromSeconds(Math.Max(10, options.TimeoutSeconds));
    }

    public async Task<GaussianSplatGenerationResult> GenerateAsync(
        Stream imageStream,
        int maxOutputPoints,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(_options.Endpoint))
        {
            return new GaussianSplatGenerationResult(false, null, 0,
                "ExternalApi engine selected but Splatting:ExternalApi:Endpoint is not configured.");
        }

        try
        {
            using var content = new MultipartFormDataContent();
            using var imageContent = new StreamContent(imageStream);
            imageContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("image/jpeg");
            content.Add(imageContent, "image", "source.jpg");
            content.Add(new StringContent(maxOutputPoints.ToString()), "maxPoints");

            using var request = new HttpRequestMessage(HttpMethod.Post, _options.Endpoint) { Content = content };
            if (!string.IsNullOrWhiteSpace(_options.ApiKey))
                request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _options.ApiKey);

            using var response = await _httpClient.SendAsync(request, ct);
            response.EnsureSuccessStatusCode();

            var bytes = await response.Content.ReadAsByteArrayAsync(ct);
            var pointCount = bytes.Length / SplatFileWriter.BytesPerPoint;

            return new GaussianSplatGenerationResult(true, bytes, pointCount, null);
        }
        catch (Exception ex)
        {
            return new GaussianSplatGenerationResult(false, null, 0, $"External splat engine call failed: {ex.Message}");
        }
    }
}
