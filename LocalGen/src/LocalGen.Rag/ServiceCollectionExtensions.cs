using LocalGen.Rag.Documents;
using LocalGen.Rag.Stores;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace LocalGen.Rag;

public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Registers document ingestion and vector search. Assumes <c>AddLocalGenRuntime</c> has
    /// already run, since embeddings are produced by a locally loaded model.
    /// </summary>
    public static IServiceCollection AddLocalGenRag(this IServiceCollection services)
    {
        services.AddHttpClient("chroma", client => client.Timeout = TimeSpan.FromSeconds(60));

        services.TryAddSingleton<DocumentExtractor>();
        services.TryAddSingleton<VectorIndexFactory>();
        services.TryAddSingleton<RagService>();

        return services;
    }
}
