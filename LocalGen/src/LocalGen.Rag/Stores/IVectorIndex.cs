using Microsoft.Extensions.VectorData;

namespace LocalGen.Rag.Stores;

/// <summary>One indexed chunk of a document.</summary>
public sealed class DocumentRecord
{
    /// <summary>Stable id derived from the source and chunk index, so re-ingesting overwrites.</summary>
    [VectorStoreKey]
    public string Id { get; set; } = string.Empty;

    /// <summary>Logical grouping, e.g. a project or knowledge base name.</summary>
    [VectorStoreData(IsIndexed = true)]
    public string Collection { get; set; } = string.Empty;

    [VectorStoreData(IsFullTextIndexed = true)]
    public string Text { get; set; } = string.Empty;

    /// <summary>File name or URL the chunk came from.</summary>
    [VectorStoreData(IsIndexed = true)]
    public string Source { get; set; } = string.Empty;

    [VectorStoreData]
    public int ChunkIndex { get; set; }

    /// <summary>Free-form metadata serialised as JSON.</summary>
    [VectorStoreData]
    public string Metadata { get; set; } = "{}";

    [VectorStoreData]
    public string IngestedAt { get; set; } = string.Empty;

    /// <summary>
    /// The embedding. The attribute's dimension is only a default — the real value comes from the
    /// embedding model and is supplied through a collection definition at runtime.
    /// </summary>
    [VectorStoreVector(768, DistanceFunction = DistanceFunction.CosineSimilarity)]
    public ReadOnlyMemory<float> Embedding { get; set; }
}

/// <summary>A chunk that matched a query, with its similarity score.</summary>
public sealed record SearchHit
{
    public required DocumentRecord Record { get; init; }

    /// <summary>Cosine similarity in 0..1; higher is more relevant.</summary>
    public double Score { get; init; }
}

/// <summary>
/// LocalGen's vector storage contract.
/// </summary>
/// <remarks>
/// Most backends are reached through <c>Microsoft.Extensions.VectorData</c>, but not every store
/// the product supports has a connector for it. This interface is the seam that lets those be
/// implemented directly without leaking two different storage APIs into the RAG service.
/// </remarks>
public interface IVectorIndex : IAsyncDisposable
{
    /// <summary>Backend name, as configured: <c>sqlite</c>, <c>qdrant</c>, …</summary>
    string Provider { get; }

    /// <summary>Creates the backing collection if it does not exist yet.</summary>
    ValueTask EnsureCreatedAsync(CancellationToken cancellationToken = default);

    ValueTask UpsertAsync(IReadOnlyList<DocumentRecord> records, CancellationToken cancellationToken = default);

    /// <summary>Finds the chunks closest to a query vector.</summary>
    ValueTask<IReadOnlyList<SearchHit>> SearchAsync(
        ReadOnlyMemory<float> queryVector,
        int top,
        string? collection = null,
        CancellationToken cancellationToken = default);

    /// <summary>Removes every chunk that came from a given source.</summary>
    ValueTask<int> DeleteBySourceAsync(string source, CancellationToken cancellationToken = default);

    /// <summary>Deletes the whole collection.</summary>
    ValueTask ClearAsync(CancellationToken cancellationToken = default);
}
