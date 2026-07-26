using System.Diagnostics;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using BlazorML.Web.Endpoints;

namespace BlazorML.Web.Services;

/// <summary>What a trial request came back with.</summary>
public sealed record ScoreAttempt(int Status, string Body, long ElapsedMs, bool Ok, string? Failure = null)
{
    public static ScoreAttempt Failed(string failure, long elapsedMs) =>
        new(0, string.Empty, elapsedMs, false, failure);
}

/// <summary>
/// Sends a trial request to a published endpoint.
/// <para>
/// Over real HTTP, against the app's own route, with the key in the header a caller would use.
/// Calling the scoring code directly would be faster and would prove nothing: the parts most
/// likely to be wrong are the key, the header name and whether the endpoint is live, and all three
/// live in front of the model, not behind it.
/// </para>
/// </summary>
public sealed class EndpointTester(IHttpClientFactory clients)
{
    public const string ClientName = "endpoint-trial";

    public async Task<ScoreAttempt> ScoreAsync(string baseUri, string slug, string apiKey,
        string payloadJson, CancellationToken ct = default)
    {
        // Checked here rather than sent, so an obvious mistake is answered immediately instead of
        // coming back as a 400 the user has to interpret.
        if (!IsJsonObject(payloadJson, out var complaint))
        {
            return ScoreAttempt.Failed(complaint, 0);
        }

        var clock = Stopwatch.StartNew();

        try
        {
            var http = clients.CreateClient(ClientName);

            using var request = new HttpRequestMessage(HttpMethod.Post,
                $"{baseUri.TrimEnd('/')}/api/v1/score/{slug}")
            {
                Content = new StringContent(payloadJson, Encoding.UTF8, "application/json")
            };

            request.Headers.TryAddWithoutValidation(ScoringApi.KeyHeader, apiKey);
            request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));

            using var response = await http.SendAsync(request, ct);
            var body = await response.Content.ReadAsStringAsync(ct);

            clock.Stop();

            return new ScoreAttempt((int)response.StatusCode, Prettify(body),
                clock.ElapsedMilliseconds, response.IsSuccessStatusCode);
        }
        catch (TaskCanceledException) when (!ct.IsCancellationRequested)
        {
            clock.Stop();
            return ScoreAttempt.Failed(
                "Permintaan habis waktu. Endpoint-nya mungkin sedang memuat model yang besar.",
                clock.ElapsedMilliseconds);
        }
        catch (HttpRequestException e)
        {
            clock.Stop();
            return ScoreAttempt.Failed($"Tidak bisa menghubungi endpoint: {e.Message}", clock.ElapsedMilliseconds);
        }
    }

    private static bool IsJsonObject(string json, out string complaint)
    {
        complaint = string.Empty;

        if (string.IsNullOrWhiteSpace(json))
        {
            complaint = "Isi dulu badan permintaannya.";
            return false;
        }

        try
        {
            using var document = JsonDocument.Parse(json);

            if (document.RootElement.ValueKind != JsonValueKind.Object)
            {
                complaint = "Badan permintaan harus objek JSON dengan properti \"rows\".";
                return false;
            }

            if (!document.RootElement.TryGetProperty("rows", out var rows)
                || rows.ValueKind != JsonValueKind.Array)
            {
                complaint = "Tambahkan properti \"rows\" berisi array baris yang mau diskor.";
                return false;
            }

            if (rows.GetArrayLength() == 0)
            {
                complaint = "Array \"rows\" masih kosong — isi minimal satu baris.";
                return false;
            }

            return true;
        }
        catch (JsonException e)
        {
            complaint = $"JSON-nya belum valid: {e.Message}";
            return false;
        }
    }

    /// <summary>
    /// Re-indents the response so it can be read. Left as-is when it is not JSON — an error page
    /// or a proxy's HTML is exactly the sort of thing worth seeing verbatim.
    /// </summary>
    private static string Prettify(string body)
    {
        if (string.IsNullOrWhiteSpace(body))
        {
            return string.Empty;
        }

        try
        {
            using var document = JsonDocument.Parse(body);
            return JsonSerializer.Serialize(document.RootElement,
                new JsonSerializerOptions { WriteIndented = true });
        }
        catch (JsonException)
        {
            return body;
        }
    }
}
