using LocalGen.Core.Engines;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace LocalGen.Engines.FoundryLocal;

public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Registers the Foundry Local backend. Probing is lazy, so adding it on a machine without
    /// Foundry Local installed costs nothing beyond one failed probe.
    /// </summary>
    public static IServiceCollection AddFoundryLocalEngine(this IServiceCollection services)
    {
        services.TryAddEnumerable(ServiceDescriptor.Singleton<IInferenceEngine, FoundryLocalEngine>());
        return services;
    }
}
