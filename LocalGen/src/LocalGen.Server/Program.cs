namespace LocalGen.Server;

/// <summary>
/// Entry point for the standalone service. The Avalonia Admin Control and the CLI's
/// <c>serve</c> command drive <see cref="LocalGenServerHost"/> directly instead of spawning
/// this executable.
/// </summary>
/// <remarks>
/// Written as an explicit, namespaced class rather than top-level statements: this assembly is
/// referenced by the CLI, and a global <c>Program</c> type would collide with the CLI's own.
/// </remarks>
internal static class ServerEntryPoint
{
    public static async Task Main(string[] args)
    {
        var host = new LocalGenServerHost();
        var app = host.Build();

        await app.RunAsync();
    }
}
