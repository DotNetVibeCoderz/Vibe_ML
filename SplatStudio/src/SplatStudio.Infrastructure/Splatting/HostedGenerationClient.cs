using System.Net.Http.Headers;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace SplatStudio.Infrastructure.Splatting;

/// <summary>Raised when a hosted provider rejects a job or never finishes it.</summary>
public class HostedGenerationException : Exception
{
    public HostedGenerationException(string message) : base(message) { }
    public HostedGenerationException(string message, Exception inner) : base(message, inner) { }
}

/// <summary>
/// Drives the submit → poll → download cycle every hosted image-to-3D provider uses, with the
/// vendor-specific naming supplied by <see cref="HostedGenerationOptions"/>.
///
/// These jobs take minutes, which is exactly why the app already has a background worker: the
/// uploader's HTTP request returns immediately and this runs off the queue. Nothing here
/// blocks a Blazor circuit.
/// </summary>
public class HostedGenerationClient
{
    private readonly HttpClient _http;
    private readonly ILogger _logger;

    public HostedGenerationClient(HttpClient http, ILogger logger)
    {
        _http = http;
        _logger = logger;
    }

    public async Task<byte[]> GenerateAsync(
        Stream imageStream,
        string imageFileName,
        HostedGenerationOptions options,
        CancellationToken ct)
    {
        if (!options.IsConfigured)
            throw new HostedGenerationException($"{options.ProviderName} is not configured.");

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutCts.CancelAfter(TimeSpan.FromSeconds(Math.Max(30, options.TimeoutSeconds)));
        var token = timeoutCts.Token;

        try
        {
            var jobId = await SubmitAsync(imageStream, imageFileName, options, token);
            _logger.LogInformation("Hosted generation: submitted job {JobId} to {Provider}.", jobId, options.ProviderName);

            var resultUrl = await PollAsync(jobId, options, token);
            _logger.LogInformation("Hosted generation: job {JobId} finished, downloading asset.", jobId);

            return await DownloadAsync(resultUrl, options, token);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            // Our own deadline fired rather than the app shutting down.
            throw new HostedGenerationException(
                $"{options.ProviderName} did not finish within {options.TimeoutSeconds}s.");
        }
    }

    private async Task<string> SubmitAsync(
        Stream imageStream, string imageFileName, HostedGenerationOptions options, CancellationToken ct)
    {
        using var content = new MultipartFormDataContent();
        var imageContent = new StreamContent(imageStream);
        imageContent.Headers.ContentType = new MediaTypeHeaderValue(GuessContentType(imageFileName));
        content.Add(imageContent, options.ImageFieldName, imageFileName);

        foreach (var (key, value) in options.SubmitFields)
            content.Add(new StringContent(value), key);

        using var request = new HttpRequestMessage(HttpMethod.Post, Combine(options.BaseUrl, options.SubmitPath))
        {
            Content = content
        };
        ApplyAuth(request, options);

        using var response = await _http.SendAsync(request, ct);
        var body = await response.Content.ReadAsStringAsync(ct);

        if (!response.IsSuccessStatusCode)
            throw new HostedGenerationException(
                $"{options.ProviderName} rejected the job ({(int)response.StatusCode}): {Truncate(body)}");

        using var json = ParseJson(body, options.ProviderName);
        var jobId = ReadPath(json.RootElement, options.JobIdPath);

        if (string.IsNullOrWhiteSpace(jobId))
            throw new HostedGenerationException(
                $"{options.ProviderName} did not return a job id at \"{options.JobIdPath}\". " +
                $"Check Splatting:Hosted JobIdPath against the provider's response shape.");

        return jobId;
    }

    private async Task<string> PollAsync(string jobId, HostedGenerationOptions options, CancellationToken ct)
    {
        var interval = TimeSpan.FromSeconds(Math.Clamp(options.PollIntervalSeconds, 1, 60));
        var statusUrl = Combine(options.BaseUrl, options.StatusPath.Replace("{jobId}", Uri.EscapeDataString(jobId)));

        while (true)
        {
            ct.ThrowIfCancellationRequested();

            using var request = new HttpRequestMessage(HttpMethod.Get, statusUrl);
            ApplyAuth(request, options);

            using var response = await _http.SendAsync(request, ct);
            var body = await response.Content.ReadAsStringAsync(ct);

            if (!response.IsSuccessStatusCode)
                throw new HostedGenerationException(
                    $"{options.ProviderName} status check failed ({(int)response.StatusCode}): {Truncate(body)}");

            using var json = ParseJson(body, options.ProviderName);
            var state = ReadPath(json.RootElement, options.StatusFieldPath) ?? string.Empty;

            if (options.FailureStates.Any(s => string.Equals(s, state, StringComparison.OrdinalIgnoreCase)))
            {
                var providerError = ReadPath(json.RootElement, options.ErrorMessagePath);
                throw new HostedGenerationException(
                    $"{options.ProviderName} reported '{state}'" +
                    (string.IsNullOrWhiteSpace(providerError) ? "." : $": {providerError}"));
            }

            if (options.SuccessStates.Any(s => string.Equals(s, state, StringComparison.OrdinalIgnoreCase)))
            {
                var url = ReadPath(json.RootElement, options.ResultUrlPath);
                if (string.IsNullOrWhiteSpace(url))
                    throw new HostedGenerationException(
                        $"{options.ProviderName} finished but returned no asset URL at \"{options.ResultUrlPath}\".");
                return url;
            }

            // Any other value means still running. Providers use their own vocabulary here
            // (queued/processing/running/generating), so anything unrecognised is treated as
            // "not done yet" rather than an error.
            await Task.Delay(interval, ct);
        }
    }

    private async Task<byte[]> DownloadAsync(string url, HostedGenerationOptions options, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, url);

        // Result URLs are usually pre-signed and on a different host (S3, CDN). Only attach
        // the API credential when the download stays on the provider's own API host, so the
        // key is never leaked to a third-party bucket.
        if (Uri.TryCreate(url, UriKind.Absolute, out var resultUri) &&
            Uri.TryCreate(options.BaseUrl, UriKind.Absolute, out var baseUri) &&
            string.Equals(resultUri.Host, baseUri.Host, StringComparison.OrdinalIgnoreCase))
        {
            ApplyAuth(request, options);
        }

        using var response = await _http.SendAsync(request, ct);
        if (!response.IsSuccessStatusCode)
            throw new HostedGenerationException(
                $"Downloading the generated asset failed ({(int)response.StatusCode}).");

        var bytes = await response.Content.ReadAsByteArrayAsync(ct);
        if (bytes.Length == 0)
            throw new HostedGenerationException($"{options.ProviderName} returned an empty asset.");

        return bytes;
    }

    private static void ApplyAuth(HttpRequestMessage request, HostedGenerationOptions options)
    {
        var value = string.IsNullOrWhiteSpace(options.AuthScheme)
            ? options.ApiKey
            : $"{options.AuthScheme} {options.ApiKey}";

        request.Headers.TryAddWithoutValidation(options.AuthHeaderName, value);
    }

    private static JsonDocument ParseJson(string body, string providerName)
    {
        try
        {
            return JsonDocument.Parse(body);
        }
        catch (JsonException ex)
        {
            throw new HostedGenerationException(
                $"{providerName} returned a response that is not JSON: {Truncate(body)}", ex);
        }
    }

    /// <summary>
    /// Resolves a dotted path like "data.result.url" against a JSON document. Numbers and
    /// booleans are returned as text so a job id can be either without extra configuration.
    /// </summary>
    internal static string? ReadPath(JsonElement root, string path)
    {
        if (string.IsNullOrWhiteSpace(path)) return null;

        var current = root;
        foreach (var segment in path.Split('.', StringSplitOptions.RemoveEmptyEntries))
        {
            // Bare integers index into arrays, e.g. "results.0.url".
            if (current.ValueKind == JsonValueKind.Array && int.TryParse(segment, out var index))
            {
                if (index < 0 || index >= current.GetArrayLength()) return null;
                current = current[index];
                continue;
            }

            if (current.ValueKind != JsonValueKind.Object) return null;
            if (!current.TryGetProperty(segment, out var next)) return null;
            current = next;
        }

        return current.ValueKind switch
        {
            JsonValueKind.String => current.GetString(),
            JsonValueKind.Number => current.GetRawText(),
            JsonValueKind.True => "true",
            JsonValueKind.False => "false",
            _ => null
        };
    }

    private static string GuessContentType(string fileName) =>
        Path.GetExtension(fileName).ToLowerInvariant() switch
        {
            ".png" => "image/png",
            ".webp" => "image/webp",
            _ => "image/jpeg"
        };

    private static string Combine(string baseUrl, string path) =>
        $"{baseUrl.TrimEnd('/')}/{path.TrimStart('/')}";

    private static string Truncate(string value) =>
        value.Length <= 300 ? value : value[..300] + "…";
}
