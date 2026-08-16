using LocalGen.Core;
using LocalGen.Core.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.InMemory;
using Microsoft.SemanticKernel.Connectors.SqliteVec;

namespace LocalGen.Rag.Stores;

/// <summary>
/// Builds the configured vector index.
/// </summary>
/// <remarks>
/// SQLite is the default because it needs no server: a laptop deployment gets working RAG with
/// nothing to install. The other backends exist for deployments that already run them.
/// </remarks>
public sealed class VectorIndexFactory
{
    private readonly LocalGenOptions _options;
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly ILoggerFactory _loggerFactory;

    public VectorIndexFactory(
        IOptions<LocalGenOptions> options,
        IHttpClientFactory httpClientFactory,
        ILoggerFactory loggerFactory)
    {
        _options = options.Value;
        _httpClientFactory = httpClientFactory;
        _loggerFactory = loggerFactory;
    }

    /// <summary>
    /// Creates an index for a collection. <paramref name="embeddingDimensions"/> must match the
    /// configured embedding model — the backend cannot infer it.
    /// </summary>
    public IVectorIndex Create(string collectionName, int embeddingDimensions)
    {
        var provider = _options.Rag.Provider.Trim().ToLowerInvariant();
        var logger = _loggerFactory.CreateLogger($"LocalGen.Rag.{provider}");

        return provider switch
        {
            "sqlite" => new VectorDataIndex(
                new SqliteVectorStore(ResolveSqliteConnectionString()),
                provider,
                collectionName,
                embeddingDimensions,
                logger),

            "inmemory" or "memory" => new VectorDataIndex(
                new InMemoryVectorStore(),
                provider,
                collectionName,
                embeddingDimensions,
                logger),

            "chroma" => new ChromaVectorIndex(
                _httpClientFactory.CreateClient("chroma"),
                string.IsNullOrWhiteSpace(_options.Rag.ConnectionString)
                    ? "http://localhost:8000"
                    : _options.Rag.ConnectionString,
                collectionName,
                logger),

            "qdrant" => CreateQdrant(collectionName, embeddingDimensions, logger),

            "azureaisearch" or "azure" => CreateAzureAiSearch(collectionName, embeddingDimensions, logger),

            _ => throw new LocalGenException(
                $"Unknown RAG provider '{_options.Rag.Provider}'. " +
                "Use sqlite, qdrant, chroma, azureaisearch or inmemory.")
        };
    }

    /// <summary>Defaults the database to the LocalGen data directory when none is configured.</summary>
    private string ResolveSqliteConnectionString()
    {
        if (!string.IsNullOrWhiteSpace(_options.Rag.ConnectionString))
        {
            return _options.Rag.ConnectionString;
        }

        var path = Path.Combine(_options.DataDirectory, "rag.localgen.db");
        Directory.CreateDirectory(_options.DataDirectory);

        return $"Data Source={path}";
    }

    private IVectorIndex CreateQdrant(string collectionName, int dimensions, ILogger logger)
    {
        var endpoint = string.IsNullOrWhiteSpace(_options.Rag.ConnectionString)
            ? "http://localhost:6334"
            : _options.Rag.ConnectionString;

        if (!Uri.TryCreate(endpoint, UriKind.Absolute, out var uri))
        {
            throw new LocalGenException(
                $"'{endpoint}' is not a valid Qdrant endpoint. Expected something like http://localhost:6334.");
        }

        var client = new Qdrant.Client.QdrantClient(uri);

        return new VectorDataIndex(
            new Microsoft.SemanticKernel.Connectors.Qdrant.QdrantVectorStore(client, ownsClient: true),
            "qdrant",
            collectionName,
            dimensions,
            logger);
    }

    private IVectorIndex CreateAzureAiSearch(string collectionName, int dimensions, ILogger logger)
    {
        var connection = _options.Rag.ConnectionString;

        if (string.IsNullOrWhiteSpace(connection))
        {
            throw new LocalGenException(
                "Azure AI Search needs LocalGen:Rag:ConnectionString set to " +
                "\"Endpoint=https://<service>.search.windows.net;ApiKey=<key>\".");
        }

        var (endpoint, apiKey) = ParseAzureConnection(connection);

        var client = new Azure.Search.Documents.Indexes.SearchIndexClient(
            new Uri(endpoint),
            new Azure.AzureKeyCredential(apiKey));

        return new VectorDataIndex(
            new Microsoft.SemanticKernel.Connectors.AzureAISearch.AzureAISearchVectorStore(client),
            "azureaisearch",
            collectionName,
            dimensions,
            logger);
    }

    /// <summary>Parses the <c>Endpoint=…;ApiKey=…</c> form used for Azure AI Search.</summary>
    private static (string Endpoint, string ApiKey) ParseAzureConnection(string connection)
    {
        string? endpoint = null;
        string? apiKey = null;

        foreach (var part in connection.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            var separator = part.IndexOf('=');
            if (separator <= 0)
            {
                continue;
            }

            var key = part[..separator].Trim();
            var value = part[(separator + 1)..].Trim();

            if (key.Equals("Endpoint", StringComparison.OrdinalIgnoreCase))
            {
                endpoint = value;
            }
            else if (key.Equals("ApiKey", StringComparison.OrdinalIgnoreCase))
            {
                apiKey = value;
            }
        }

        if (string.IsNullOrWhiteSpace(endpoint) || string.IsNullOrWhiteSpace(apiKey))
        {
            throw new LocalGenException(
                "The Azure AI Search connection string must contain both Endpoint and ApiKey.");
        }

        return (endpoint, apiKey);
    }
}
