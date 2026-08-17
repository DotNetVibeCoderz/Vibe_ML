using ElBruno.MarkItDotNet;
using ElBruno.MarkItDotNet.Excel;
using ElBruno.MarkItDotNet.PowerPoint;
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

        // Document conversion. Excel and PowerPoint have to be registered explicitly — their
        // converters are not discovered from the loaded assemblies, so leaving either call out
        // means .xlsx and .pptx come back as "format not supported" at ingestion time.
        //
        // The size limit is set to LocalGen's own so that an oversized file is refused once, with
        // LocalGen's message naming the actual limit, rather than by whichever guard fires first.
        services.AddMarkItDotNet(options => options.MaxFileSizeBytes = DocumentExtractor.MaxFileSizeBytes);
        services.AddMarkItDotNetExcel();
        services.AddMarkItDotNetPowerPoint();

        services.TryAddSingleton<DocumentExtractor>();
        services.TryAddSingleton<VectorIndexFactory>();
        services.TryAddSingleton<RagService>();

        return services;
    }
}
