using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;

namespace LocalGen.Sdk;

public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Registers <see cref="LocalGenClient"/> against a LocalGen server. The underlying
    /// <see cref="HttpClient"/> comes from <c>IHttpClientFactory</c> so socket handling and
    /// DNS refresh follow the usual .NET conventions.
    /// </summary>
    public static IServiceCollection AddLocalGenClient(
        this IServiceCollection services,
        Action<LocalGenClientOptions>? configure = null)
    {
        services.Configure(configure ?? (static _ => { }));

        services.AddHttpClient(nameof(LocalGenClient), (provider, http) =>
        {
            var options = provider.GetRequiredService<IOptions<LocalGenClientOptions>>().Value;

            http.BaseAddress = new Uri(options.Endpoint.TrimEnd('/') + "/");
            http.Timeout = options.Timeout;

            if (!string.IsNullOrEmpty(options.ApiKey))
            {
                http.DefaultRequestHeaders.Authorization =
                    new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", options.ApiKey);
            }
        });

        services.AddSingleton(provider =>
        {
            var factory = provider.GetRequiredService<IHttpClientFactory>();
            var options = provider.GetRequiredService<IOptions<LocalGenClientOptions>>().Value;

            return new LocalGenClient(factory.CreateClient(nameof(LocalGenClient)), options);
        });

        return services;
    }

    /// <summary>
    /// Registers LocalGen as the application's <see cref="IChatClient"/>, so anything built on
    /// <c>Microsoft.Extensions.AI</c> — including Semantic Kernel — resolves it automatically.
    /// </summary>
    public static IServiceCollection AddLocalGenChatClient(
        this IServiceCollection services,
        string modelId,
        Action<LocalGenClientOptions>? configure = null)
    {
        services.AddLocalGenClient(configure);

        services.AddSingleton<IChatClient>(provider =>
            new LocalGenChatClient(provider.GetRequiredService<LocalGenClient>(), modelId));

        return services;
    }
}
