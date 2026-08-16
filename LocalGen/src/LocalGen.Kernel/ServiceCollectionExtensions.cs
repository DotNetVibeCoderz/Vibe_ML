using LocalGen.Core.Configuration;
using LocalGen.Kernel.Mcp;
using LocalGen.Kernel.Plugins;
using LocalGen.Kernel.Skills;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;

namespace LocalGen.Kernel;

public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Registers the Semantic Kernel layer: the kernel factory, skills, MCP and the shared path
    /// guard. Assumes <c>AddLocalGenRuntime</c> has already run.
    /// </summary>
    public static IServiceCollection AddLocalGenKernel(this IServiceCollection services)
    {
        // Web-touching tools get their own client with a bounded timeout — unlike model
        // downloads, a page fetch that hangs should fail rather than stall a conversation.
        foreach (var name in (string[])["WebScrape", "Download", nameof(SearchPlugin), nameof(SkillStore)])
        {
            services.AddHttpClient(name, client =>
            {
                client.Timeout = TimeSpan.FromSeconds(60);
                client.DefaultRequestHeaders.UserAgent.ParseAdd(
                    "LocalGen/0.1 (+https://github.com/gravicode/LocalGen)");
            });
        }

        services.TryAddSingleton(provider =>
            new PathGuard(provider.GetRequiredService<IOptions<LocalGenOptions>>().Value));

        services.TryAddSingleton<SkillStore>();
        services.TryAddSingleton<McpManager>();
        services.TryAddSingleton<LocalGenKernelFactory>();

        return services;
    }
}
