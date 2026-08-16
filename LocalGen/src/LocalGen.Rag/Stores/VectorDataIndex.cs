using Microsoft.Extensions.Logging;
using Microsoft.Extensions.VectorData;

namespace LocalGen.Rag.Stores;

/// <summary>
/// A vector index backed by any <c>Microsoft.Extensions.VectorData</c> store — SQLite, Qdrant,
/// Azure AI Search or the in-memory store.
/// </summary>
/// <remarks>
/// The collection is created from a runtime <see cref="VectorStoreCollectionDefinition"/> rather
/// than from the attributes on <see cref="DocumentRecord"/>, because the vector dimension depends
/// on whichever embedding model the user configured and is not known at compile time.
/// </remarks>
public sealed class VectorDataIndex : IVectorIndex
{
    private readonly VectorStore _store;
    private readonly VectorStoreCollection<string, DocumentRecord> _collection;
    private readonly ILogger _logger;

    public VectorDataIndex(
        VectorStore store,
        string provider,
        string collectionName,
        int embeddingDimensions,
        ILogger logger)
    {
        _store = store;
        Provider = provider;
        _logger = logger;

        _collection = store.GetCollection<string, DocumentRecord>(
            collectionName,
            BuildDefinition(embeddingDimensions));
    }

    public string Provider { get; }

    /// <summary>
    /// Describes the record shape with the embedding dimension the configured model actually
    /// produces. A mismatch here surfaces as an opaque backend error at upsert time.
    /// </summary>
    private static VectorStoreCollectionDefinition BuildDefinition(int dimensions) => new()
    {
        Properties =
        [
            new VectorStoreKeyProperty(nameof(DocumentRecord.Id), typeof(string)),
            new VectorStoreDataProperty(nameof(DocumentRecord.Collection), typeof(string))
            {
                IsIndexed = true
            },
            new VectorStoreDataProperty(nameof(DocumentRecord.Text), typeof(string))
            {
                IsFullTextIndexed = true
            },
            new VectorStoreDataProperty(nameof(DocumentRecord.Source), typeof(string))
            {
                IsIndexed = true
            },
            new VectorStoreDataProperty(nameof(DocumentRecord.ChunkIndex), typeof(int)),
            new VectorStoreDataProperty(nameof(DocumentRecord.Metadata), typeof(string)),
            new VectorStoreDataProperty(nameof(DocumentRecord.IngestedAt), typeof(string)),
            new VectorStoreVectorProperty(
                nameof(DocumentRecord.Embedding),
                typeof(ReadOnlyMemory<float>),
                dimensions)
            {
                DistanceFunction = DistanceFunction.CosineSimilarity
            }
        ]
    };

    public async ValueTask EnsureCreatedAsync(CancellationToken cancellationToken = default) =>
        await _collection.EnsureCollectionExistsAsync(cancellationToken).ConfigureAwait(false);

    public async ValueTask UpsertAsync(
        IReadOnlyList<DocumentRecord> records,
        CancellationToken cancellationToken = default)
    {
        if (records.Count == 0)
        {
            return;
        }

        await _collection.UpsertAsync(records, cancellationToken).ConfigureAwait(false);
        _logger.LogDebug("Upserted {Count} chunk(s) into {Provider}", records.Count, Provider);
    }

    public async ValueTask<IReadOnlyList<SearchHit>> SearchAsync(
        ReadOnlyMemory<float> queryVector,
        int top,
        string? collection = null,
        CancellationToken cancellationToken = default)
    {
        var options = new VectorSearchOptions<DocumentRecord>();

        if (!string.IsNullOrEmpty(collection))
        {
            options.Filter = record => record.Collection == collection;
        }

        var hits = new List<SearchHit>(top);

        await foreach (var result in _collection
            .SearchAsync(queryVector, top, options, cancellationToken)
            .ConfigureAwait(false))
        {
            hits.Add(new SearchHit
            {
                Record = result.Record,
                Score = result.Score ?? 0
            });
        }

        return hits;
    }

    public async ValueTask<int> DeleteBySourceAsync(
        string source,
        CancellationToken cancellationToken = default)
    {
        // Deletion is by key, so the matching records are located first. The cap keeps a stray
        // call from enumerating an entire large collection.
        var keys = new List<string>();

        await foreach (var record in _collection
            .GetAsync(r => r.Source == source, top: 10_000, cancellationToken: cancellationToken)
            .ConfigureAwait(false))
        {
            keys.Add(record.Id);
        }

        if (keys.Count == 0)
        {
            return 0;
        }

        await _collection.DeleteAsync(keys, cancellationToken).ConfigureAwait(false);
        _logger.LogInformation("Deleted {Count} chunk(s) from source {Source}", keys.Count, source);

        return keys.Count;
    }

    public async ValueTask ClearAsync(CancellationToken cancellationToken = default) =>
        await _collection.EnsureCollectionDeletedAsync(cancellationToken).ConfigureAwait(false);

    public ValueTask DisposeAsync()
    {
        _collection.Dispose();
        _store.Dispose();
        return ValueTask.CompletedTask;
    }
}
