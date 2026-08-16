using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using LocalGen.Core;
using Microsoft.Extensions.Logging;

namespace LocalGen.Rag.Stores;

/// <summary>
/// A vector index over a Chroma server, spoken directly through its HTTP API.
/// </summary>
/// <remarks>
/// Chroma has no current <c>Microsoft.Extensions.VectorData</c> connector, and the legacy one is
/// alpha-only. Its REST surface is small enough that talking to it directly is more dependable
/// than taking a pre-release dependency, and it keeps Chroma a first-class option as the product
/// requires. Everything above <see cref="IVectorIndex"/> is unaware of the difference.
/// </remarks>
public sealed class ChromaVectorIndex : IVectorIndex
{
    private readonly HttpClient _http;
    private readonly string _collectionName;
    private readonly ILogger _logger;
    private readonly SemaphoreSlim _initLock = new(1, 1);

    /// <summary>Chroma's own collection id, resolved on first use.</summary>
    private string? _collectionId;

    public ChromaVectorIndex(
        HttpClient http,
        string endpoint,
        string collectionName,
        ILogger logger)
    {
        _http = http;
        _collectionName = collectionName;
        _logger = logger;

        _http.BaseAddress ??= new Uri(endpoint.TrimEnd('/') + "/");
    }

    public string Provider => "chroma";

    public async ValueTask EnsureCreatedAsync(CancellationToken cancellationToken = default)
    {
        if (_collectionId is not null)
        {
            return;
        }

        await _initLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_collectionId is not null)
            {
                return;
            }

            // get_or_create makes this idempotent, which matters because every operation calls it.
            using var response = await _http.PostAsJsonAsync(
                "api/v1/collections",
                new
                {
                    name = _collectionName,
                    get_or_create = true,
                    metadata = new Dictionary<string, string> { ["hnsw:space"] = "cosine" }
                },
                cancellationToken).ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
            {
                var body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
                throw new LocalGenException(
                    $"Chroma rejected the collection request ({(int)response.StatusCode}): {body}");
            }

            using var document = JsonDocument.Parse(
                await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false));

            _collectionId = document.RootElement.TryGetProperty("id", out var id)
                ? id.GetString()
                : throw new LocalGenException("Chroma did not return a collection id.");

            _logger.LogInformation("Using Chroma collection '{Collection}' ({Id})", _collectionName, _collectionId);
        }
        catch (HttpRequestException ex)
        {
            throw new LocalGenException(
                $"Could not reach the Chroma server at {_http.BaseAddress}. Is it running?", ex);
        }
        finally
        {
            _initLock.Release();
        }
    }

    public async ValueTask UpsertAsync(
        IReadOnlyList<DocumentRecord> records,
        CancellationToken cancellationToken = default)
    {
        if (records.Count == 0)
        {
            return;
        }

        await EnsureCreatedAsync(cancellationToken).ConfigureAwait(false);

        var payload = new
        {
            ids = records.Select(static r => r.Id).ToArray(),
            embeddings = records.Select(static r => r.Embedding.ToArray()).ToArray(),
            documents = records.Select(static r => r.Text).ToArray(),
            metadatas = records.Select(static r => new Dictionary<string, object>
            {
                ["collection"] = r.Collection,
                ["source"] = r.Source,
                ["chunkIndex"] = r.ChunkIndex,
                ["metadata"] = r.Metadata,
                ["ingestedAt"] = r.IngestedAt
            }).ToArray()
        };

        using var response = await _http.PostAsJsonAsync(
            $"api/v1/collections/{_collectionId}/upsert", payload, cancellationToken).ConfigureAwait(false);

        await EnsureSuccessAsync(response, "upsert", cancellationToken).ConfigureAwait(false);
    }

    public async ValueTask<IReadOnlyList<SearchHit>> SearchAsync(
        ReadOnlyMemory<float> queryVector,
        int top,
        string? collection = null,
        CancellationToken cancellationToken = default)
    {
        await EnsureCreatedAsync(cancellationToken).ConfigureAwait(false);

        var payload = new Dictionary<string, object>
        {
            ["query_embeddings"] = new[] { queryVector.ToArray() },
            ["n_results"] = top,
            ["include"] = new[] { "documents", "metadatas", "distances" }
        };

        if (!string.IsNullOrEmpty(collection))
        {
            payload["where"] = new Dictionary<string, object> { ["collection"] = collection };
        }

        using var response = await _http.PostAsJsonAsync(
            $"api/v1/collections/{_collectionId}/query", payload, cancellationToken).ConfigureAwait(false);

        await EnsureSuccessAsync(response, "query", cancellationToken).ConfigureAwait(false);

        using var document = JsonDocument.Parse(
            await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false));

        return ReadHits(document.RootElement);
    }

    /// <summary>
    /// Chroma returns parallel arrays nested one level per query. Only one query is ever sent,
    /// so the first row of each array is read.
    /// </summary>
    private static IReadOnlyList<SearchHit> ReadHits(JsonElement root)
    {
        var ids = FirstRow(root, "ids");
        var documents = FirstRow(root, "documents");
        var metadatas = FirstRow(root, "metadatas");
        var distances = FirstRow(root, "distances");

        var hits = new List<SearchHit>();

        for (var i = 0; i < ids.Count; i++)
        {
            var metadata = i < metadatas.Count ? metadatas[i] : default;

            hits.Add(new SearchHit
            {
                // Chroma reports cosine distance; the rest of LocalGen works in similarity.
                Score = i < distances.Count && distances[i].TryGetDouble(out var distance)
                    ? 1 - distance
                    : 0,
                Record = new DocumentRecord
                {
                    Id = ids[i].GetString() ?? string.Empty,
                    Text = i < documents.Count ? documents[i].GetString() ?? string.Empty : string.Empty,
                    Collection = ReadMetadata(metadata, "collection"),
                    Source = ReadMetadata(metadata, "source"),
                    Metadata = ReadMetadata(metadata, "metadata"),
                    IngestedAt = ReadMetadata(metadata, "ingestedAt"),
                    ChunkIndex = int.TryParse(ReadMetadata(metadata, "chunkIndex"), out var index) ? index : 0
                }
            });
        }

        return hits;
    }

    private static List<JsonElement> FirstRow(JsonElement root, string property)
    {
        if (!root.TryGetProperty(property, out var outer) ||
            outer.ValueKind != JsonValueKind.Array ||
            outer.GetArrayLength() == 0)
        {
            return [];
        }

        var inner = outer[0];
        return inner.ValueKind == JsonValueKind.Array ? [.. inner.EnumerateArray()] : [];
    }

    private static string ReadMetadata(JsonElement metadata, string key) =>
        metadata.ValueKind == JsonValueKind.Object && metadata.TryGetProperty(key, out var value)
            ? value.ValueKind == JsonValueKind.String ? value.GetString() ?? string.Empty : value.ToString()
            : string.Empty;

    public async ValueTask<int> DeleteBySourceAsync(
        string source,
        CancellationToken cancellationToken = default)
    {
        await EnsureCreatedAsync(cancellationToken).ConfigureAwait(false);

        using var response = await _http.PostAsJsonAsync(
            $"api/v1/collections/{_collectionId}/delete",
            new { where = new Dictionary<string, object> { ["source"] = source } },
            cancellationToken).ConfigureAwait(false);

        await EnsureSuccessAsync(response, "delete", cancellationToken).ConfigureAwait(false);

        // Chroma's delete response does not reliably report a count, so this reports "unknown"
        // as zero rather than inventing a number.
        return 0;
    }

    public async ValueTask ClearAsync(CancellationToken cancellationToken = default)
    {
        using var response = await _http
            .DeleteAsync($"api/v1/collections/{_collectionName}", cancellationToken)
            .ConfigureAwait(false);

        if (response.StatusCode != HttpStatusCode.NotFound)
        {
            await EnsureSuccessAsync(response, "delete collection", cancellationToken).ConfigureAwait(false);
        }

        _collectionId = null;
    }

    private static async Task EnsureSuccessAsync(
        HttpResponseMessage response,
        string operation,
        CancellationToken cancellationToken)
    {
        if (response.IsSuccessStatusCode)
        {
            return;
        }

        var body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        throw new LocalGenException($"Chroma {operation} failed ({(int)response.StatusCode}): {body}");
    }

    public ValueTask DisposeAsync()
    {
        _initLock.Dispose();
        return ValueTask.CompletedTask;
    }
}
