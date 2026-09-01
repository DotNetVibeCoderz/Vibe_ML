using LocalGen.Core.Configuration;
using LocalGen.Core.Diagnostics;
using LocalGen.Core.Models;
using LocalGen.Runtime.Catalogs;
using LocalGen.Runtime.Diagnostics;
using LocalGen.Runtime.Downloads;
using LocalGen.Runtime.Engines;
using LocalGen.Runtime.Models;
using LocalGen.Runtime.Sessions;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace LocalGen.Runtime;

public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Registers model management, engine selection and session lifetime. Backends are added
    /// separately (<c>AddLlamaSharpEngine()</c>, <c>AddOnnxEngine()</c>) so that a deployment only
    /// pulls in the native dependencies it actually needs.
    /// </summary>
    public static IServiceCollection AddLocalGenRuntime(
        this IServiceCollection services,
        IConfiguration configuration)
    {
        services.Configure<LocalGenOptions>(configuration.GetSection(LocalGenOptions.SectionName));

        // Model downloads are long-lived streams of multi-gigabyte files, so the default 100s
        // HttpClient timeout is removed and cancellation is left to the caller's token.
        services.AddHttpClient(HuggingFaceDownloader.HttpClientName, client =>
        {
            client.Timeout = Timeout.InfiniteTimeSpan;
            client.DefaultRequestHeaders.UserAgent.ParseAdd("LocalGen/0.1 (+https://github.com/gravicode/LocalGen)");
        });

        services.TryAddSingleton<IModelDownloader, HuggingFaceDownloader>();
        services.TryAddSingleton<IModelStore, FileModelStore>();
        services.TryAddEnumerable(ServiceDescriptor.Singleton<IModelCatalog, HuggingFaceCatalog>());

        services.TryAddSingleton<Files.FileStore>();
        services.TryAddSingleton<EngineRegistry>();
        services.TryAddSingleton<ModelSessionManager>();
        services.TryAddSingleton<SystemMonitor>();
        services.TryAddSingleton<InferenceMetrics>();

        return services;
    }
}
