using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using LocalGen.Core;
using LocalGen.Core.Configuration;
using LocalGen.Core.Inference;
using LocalGen.Rag.Documents;
using LocalGen.Rag.Stores;
using LocalGen.Runtime.Sessions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace LocalGen.Rag;

/// <summary>Outcome of ingesting one document.</summary>
public sealed record IngestionResult
{
    public required string Source { get; init; }

    public int ChunkCount { get; init; }

    public int CharacterCount { get; init; }

    public TimeSpan Duration { get; init; }

    public string Kind { get; init; } = string.Empty;
}

/// <summary>
/// Ingests documents and retrieves passages for a query.
/// </summary>
/// <remarks>
/// Embeddings are produced by a local model through <see cref="ModelSessionManager"/>, so RAG
/// works in offline mode with no external embedding service. The index is created lazily because
/// its vector dimension is only known once the embedding model has been loaded and asked.
/// </remarks>
public sealed class RagService : IAsyncDisposable
{
    private readonly ModelSessionManager _sessions;
    private readonly VectorIndexFactory _indexFactory;
    private readonly DocumentExtractor _extractor;
    private readonly LocalGenOptions _options;
    private readonly ILogger<RagService> _logger;
    private readonly SemaphoreSlim _initLock = new(1, 1);

    private IVectorIndex? _index;
    private int _embeddingDimensions;

    public RagService(
        ModelSessionManager sessions,
        VectorIndexFactory indexFactory,
        DocumentExtractor extractor,
        IOptions<LocalGenOptions> options,
        ILogger<RagService> logger)
    {
        _sessions = sessions;
        _indexFactory = indexFactory;
        _extractor = extractor;
        _options = options.Value;
        _logger = logger;
    }

    /// <summary>Name of the collection documents land in unless another is given.</summary>
    public const string DefaultCollection = "default";

    /// <summary>Ingests a single file.</summary>
    public async Task<IngestionResult> IngestFileAsync(
        string path,
        string collection = DefaultCollection,
        CancellationToken cancellationToken = default)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var document = await _extractor.ExtractAsync(path, cancellationToken).ConfigureAwait(false);

        if (string.IsNullOrWhiteSpace(document.Text))
        {
            return new IngestionResult
            {
                Source = document.FileName,
                Kind = document.Kind,
                Duration = stopwatch.Elapsed
            };
        }

        var chunks = document.Kind == "markdown"
            ? TextChunker.SplitMarkdown(document.Text, _options.Rag.ChunkSize, _options.Rag.ChunkOverlap)
            : TextChunker.Split(document.Text, _options.Rag.ChunkSize, _options.Rag.ChunkOverlap);

        await IngestChunksAsync(chunks, document.FileName, collection, document.Metadata, cancellationToken)
            .ConfigureAwait(false);

        _logger.LogInformation(
            "Ingested {File}: {Chunks} chunk(s) in {Elapsed:N1}s",
            document.FileName, chunks.Count, stopwatch.Elapsed.TotalSeconds);

        return new IngestionResult
        {
            Source = document.FileName,
            Kind = document.Kind,
            ChunkCount = chunks.Count,
            CharacterCount = document.Text.Length,
            Duration = stopwatch.Elapsed
        };
    }

    /// <summary>Ingests every supported file in a directory.</summary>
    public async Task<IReadOnlyList<IngestionResult>> IngestDirectoryAsync(
        string directory,
        string collection = DefaultCollection,
        bool recursive = true,
        CancellationToken cancellationToken = default)
    {
        if (!Directory.Exists(directory))
        {
            throw new LocalGenException($"No directory at '{directory}'.");
        }

        var results = new List<IngestionResult>();

        var files = Directory
            .EnumerateFiles(directory, "*", recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly)
            .Where(_extractor.CanExtract);

        foreach (var file in files)
        {
            cancellationToken.ThrowIfCancellationRequested();

            try
            {
                results.Add(await IngestFileAsync(file, collection, cancellationToken).ConfigureAwait(false));
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                // One unreadable file should not abandon a bulk ingest.
                _logger.LogWarning(ex, "Skipping {File}", file);
            }
        }

        return results;
    }

    /// <summary>Ingests raw text that did not come from a file, such as a scraped page.</summary>
    public async Task<IngestionResult> IngestTextAsync(
        string text,
        string source,
        string collection = DefaultCollection,
        CancellationToken cancellationToken = default)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        var chunks = TextChunker.Split(text, _options.Rag.ChunkSize, _options.Rag.ChunkOverlap);

        await IngestChunksAsync(chunks, source, collection, null, cancellationToken).ConfigureAwait(false);

        return new IngestionResult
        {
            Source = source,
            Kind = "text",
            ChunkCount = chunks.Count,
            CharacterCount = text.Length,
            Duration = stopwatch.Elapsed
        };
    }

    private async Task IngestChunksAsync(
        IReadOnlyList<TextChunk> chunks,
        string source,
        string collection,
        IReadOnlyDictionary<string, string>? metadata,
        CancellationToken cancellationToken)
    {
        if (chunks.Count == 0)
        {
            return;
        }

        var index = await GetIndexAsync(cancellationToken).ConfigureAwait(false);

        // Re-ingesting a source replaces it, so editing and re-adding a file does not leave the
        // old chunks behind to be retrieved alongside the new ones.
        await index.DeleteBySourceAsync(source, cancellationToken).ConfigureAwait(false);

        var embeddings = await EmbedAsync(
            [.. chunks.Select(static c => c.Text)], cancellationToken).ConfigureAwait(false);

        var metadataJson = metadata is null or { Count: 0 }
            ? "{}"
            : JsonSerializer.Serialize(metadata);

        var timestamp = DateTimeOffset.UtcNow.ToString("O");

        var records = chunks.Select((chunk, i) => new DocumentRecord
        {
            Id = BuildId(collection, source, chunk.Index),
            Collection = collection,
            Source = source,
            Text = chunk.Text,
            ChunkIndex = chunk.Index,
            Metadata = metadataJson,
            IngestedAt = timestamp,
            Embedding = embeddings[i]
        }).ToList();

        await index.UpsertAsync(records, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Retrieves the passages most relevant to a query.</summary>
    public async Task<IReadOnlyList<SearchHit>> SearchAsync(
        string query,
        int? top = null,
        string? collection = DefaultCollection,
        CancellationToken cancellationToken = default)
    {
        var index = await GetIndexAsync(cancellationToken).ConfigureAwait(false);

        var vectors = await EmbedAsync([query], cancellationToken).ConfigureAwait(false);

        var hits = await index
            .SearchAsync(vectors[0], top ?? _options.Rag.TopK, collection, cancellationToken)
            .ConfigureAwait(false);

        // Below the relevance floor a "match" is usually noise that would mislead the model.
        return [.. hits.Where(hit => hit.Score >= _options.Rag.MinRelevance)];
    }

    /// <summary>
    /// Formats retrieved passages as a context block to prepend to a prompt, with sources so the
    /// model can cite them.
    /// </summary>
    public static string FormatContext(IReadOnlyList<SearchHit> hits)
    {
        if (hits.Count == 0)
        {
            return string.Empty;
        }

        var builder = new StringBuilder();
        builder.AppendLine("Use the following retrieved context to answer. Cite sources by name.");
        builder.AppendLine();

        for (var i = 0; i < hits.Count; i++)
        {
            var hit = hits[i];
            builder.Append("[").Append(i + 1).Append("] ").Append(hit.Record.Source);

            if (hit.Record.ChunkIndex > 0)
            {
                builder.Append(" (chunk ").Append(hit.Record.ChunkIndex).Append(')');
            }

            builder.Append(" — relevance ").Append(hit.Score.ToString("P0")).AppendLine();
            builder.AppendLine(hit.Record.Text);
            builder.AppendLine();
        }

        return builder.ToString();
    }

    public async Task<int> RemoveSourceAsync(string source, CancellationToken cancellationToken = default)
    {
        var index = await GetIndexAsync(cancellationToken).ConfigureAwait(false);
        return await index.DeleteBySourceAsync(source, cancellationToken).ConfigureAwait(false);
    }

    public async Task ClearAsync(CancellationToken cancellationToken = default)
    {
        var index = await GetIndexAsync(cancellationToken).ConfigureAwait(false);
        await index.ClearAsync(cancellationToken).ConfigureAwait(false);

        // The collection is gone; drop the handle so the next call recreates it.
        await index.DisposeAsync().ConfigureAwait(false);
        _index = null;
    }

    /// <summary>Embeds text with the configured embedding model.</summary>
    private async Task<IReadOnlyList<ReadOnlyMemory<float>>> EmbedAsync(
        IReadOnlyList<string> inputs,
        CancellationToken cancellationToken)
    {
        using var lease = await _sessions
            .AcquireAsync(_options.Rag.EmbeddingModel, null, cancellationToken)
            .ConfigureAwait(false);

        var response = await lease.Session.EmbedAsync(
            new EmbeddingRequest { Model = _options.Rag.EmbeddingModel, Inputs = inputs },
            cancellationToken).ConfigureAwait(false);

        return response.Embeddings;
    }

    /// <summary>
    /// Creates the index on first use. The embedding model is asked for one vector first because
    /// the store needs the dimension up front and only the model can say what it is.
    /// </summary>
    private async ValueTask<IVectorIndex> GetIndexAsync(CancellationToken cancellationToken)
    {
        if (_index is not null)
        {
            return _index;
        }

        await _initLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_index is not null)
            {
                return _index;
            }

            if (_embeddingDimensions == 0)
            {
                var probe = await EmbedAsync(["dimension probe"], cancellationToken).ConfigureAwait(false);
                _embeddingDimensions = probe[0].Length;

                _logger.LogInformation(
                    "Embedding model '{Model}' produces {Dimensions}-dimensional vectors.",
                    _options.Rag.EmbeddingModel, _embeddingDimensions);
            }

            var index = _indexFactory.Create("localgen_documents", _embeddingDimensions);
            await index.EnsureCreatedAsync(cancellationToken).ConfigureAwait(false);

            return _index = index;
        }
        finally
        {
            _initLock.Release();
        }
    }

    /// <summary>
    /// Builds a deterministic id so re-ingesting the same chunk updates it rather than
    /// duplicating it. The source is hashed to keep long paths out of the key.
    /// </summary>
    private static string BuildId(string collection, string source, int chunkIndex)
    {
        var hash = Convert.ToHexString(
            SHA256.HashData(Encoding.UTF8.GetBytes($"{collection}|{source}")))[..16];

        return $"{hash}_{chunkIndex}";
    }

    public async ValueTask DisposeAsync()
    {
        if (_index is not null)
        {
            await _index.DisposeAsync().ConfigureAwait(false);
        }

        _initLock.Dispose();
    }
}
