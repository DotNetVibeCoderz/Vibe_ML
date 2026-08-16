using LocalGen.Core.Configuration;
using LocalGen.Kernel.Mcp;
using LocalGen.Kernel.Plugins;
using LocalGen.Kernel.Skills;
using LocalGen.Runtime.Sessions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace LocalGen.Kernel;

/// <summary>
/// Which built-in kernel functions a conversation may use.
/// </summary>
/// <remarks>
/// Every function is on by default, matching the Playground's checklist. Turning one off removes
/// it from the kernel entirely rather than instructing the model not to use it — a prompt is not
/// a permission boundary.
/// </remarks>
public sealed record ToolSelection
{
    public bool Math { get; init; } = true;

    public bool InternetSearch { get; init; } = true;

    public bool Download { get; init; } = true;

    public bool WebScrape { get; init; } = true;

    public bool CodeExecution { get; init; } = true;

    public bool TimeAndDate { get; init; } = true;

    public bool FileSystem { get; init; } = true;

    public bool Skills { get; init; } = true;

    /// <summary>MCP servers to attach, by configured name. Empty attaches none.</summary>
    public IReadOnlyList<string> McpServers { get; init; } = [];

    public static readonly ToolSelection All = new();

    public static readonly ToolSelection None = new()
    {
        Math = false,
        InternetSearch = false,
        Download = false,
        WebScrape = false,
        CodeExecution = false,
        TimeAndDate = false,
        FileSystem = false,
        Skills = false
    };

    /// <summary>Narrows a selection by what the operator has allowed in configuration.</summary>
    public ToolSelection RestrictTo(ToolOptions options) => new()
    {
        Math = Math && options.Math,
        InternetSearch = InternetSearch && options.InternetSearch,
        Download = Download && options.Download,
        WebScrape = WebScrape && options.WebScrape,
        CodeExecution = CodeExecution && options.CodeExecution,
        TimeAndDate = TimeAndDate && options.TimeAndDate,
        FileSystem = FileSystem && options.FileSystem,
        Skills = Skills,
        McpServers = McpServers
    };
}

/// <summary>
/// Builds a configured <see cref="Microsoft.SemanticKernel.Kernel"/>: a local model as the chat
/// service, plus whichever built-in functions, skills and MCP servers the caller selected.
/// </summary>
public sealed class LocalGenKernelFactory
{
    private readonly IServiceProvider _services;
    private readonly ModelSessionManager _sessions;
    private readonly LocalGenOptions _options;
    private readonly McpManager _mcp;
    private readonly ILoggerFactory _loggerFactory;

    public LocalGenKernelFactory(
        IServiceProvider services,
        ModelSessionManager sessions,
        IOptions<LocalGenOptions> options,
        McpManager mcp,
        ILoggerFactory loggerFactory)
    {
        _services = services;
        _sessions = sessions;
        _options = options.Value;
        _mcp = mcp;
        _loggerFactory = loggerFactory;
    }

    /// <summary>
    /// Creates a kernel bound to <paramref name="modelId"/>. Connecting MCP servers can be slow
    /// (a stdio server is a child process), so it is done here rather than lazily mid-conversation.
    /// </summary>
    public async Task<Microsoft.SemanticKernel.Kernel> CreateAsync(
        string modelId,
        ToolSelection? tools = null,
        CancellationToken cancellationToken = default)
    {
        var selection = (tools ?? ToolSelection.All).RestrictTo(_options.Tools);

        var builder = Microsoft.SemanticKernel.Kernel.CreateBuilder();
        builder.Services.AddSingleton(_loggerFactory);

        builder.Services.AddSingleton<IChatCompletionService>(
            new LocalGenChatCompletionService(_sessions, modelId));

        var kernel = builder.Build();

        AddBuiltInPlugins(kernel, selection);
        await AddMcpPluginsAsync(kernel, selection, cancellationToken).ConfigureAwait(false);

        return kernel;
    }

    /// <summary>Creates an agent over a freshly built kernel — the usual entry point for the Playground.</summary>
    public async Task<LocalGenAgent> CreateAgentAsync(
        string modelId,
        ToolSelection? tools = null,
        AgentOptions? agentOptions = null,
        CancellationToken cancellationToken = default)
    {
        var kernel = await CreateAsync(modelId, tools, cancellationToken).ConfigureAwait(false);

        return new LocalGenAgent(
            kernel,
            agentOptions,
            _loggerFactory.CreateLogger<LocalGenAgent>());
    }

    private void AddBuiltInPlugins(Microsoft.SemanticKernel.Kernel kernel, ToolSelection selection)
    {
        var guard = _services.GetRequiredService<PathGuard>();
        var httpClientFactory = _services.GetRequiredService<IHttpClientFactory>();

        if (selection.Math)
        {
            kernel.Plugins.AddFromObject(new MathPlugin(), "Math");
        }

        if (selection.TimeAndDate)
        {
            kernel.Plugins.AddFromObject(new TimePlugin(), "Time");
        }

        if (selection.FileSystem)
        {
            kernel.Plugins.AddFromObject(new FileSystemPlugin(guard), "Files");
        }

        if (selection.WebScrape)
        {
            kernel.Plugins.AddFromObject(
                new WebScrapePlugin(httpClientFactory, _options, _loggerFactory.CreateLogger<WebScrapePlugin>()),
                "Web");
        }

        if (selection.InternetSearch)
        {
            kernel.Plugins.AddFromObject(
                new SearchPlugin(httpClientFactory, _options, _loggerFactory.CreateLogger<SearchPlugin>()),
                "Search");
        }

        if (selection.Download)
        {
            kernel.Plugins.AddFromObject(
                new DownloadPlugin(httpClientFactory, guard, _options),
                "Downloads");
        }

        // The skills plugin runs bundled scripts through the code execution tool, so it needs an
        // instance even when the model itself may not execute arbitrary code.
        var codeExecution = new CodeExecutionPlugin(
            guard, _options, _loggerFactory.CreateLogger<CodeExecutionPlugin>());

        if (selection.CodeExecution)
        {
            kernel.Plugins.AddFromObject(codeExecution, "Code");
        }

        if (selection.Skills)
        {
            kernel.Plugins.AddFromObject(
                new SkillsPlugin(_services.GetRequiredService<SkillStore>(), codeExecution),
                "Skills");
        }
    }

    private async Task AddMcpPluginsAsync(
        Microsoft.SemanticKernel.Kernel kernel,
        ToolSelection selection,
        CancellationToken cancellationToken)
    {
        if (selection.McpServers.Count == 0)
        {
            return;
        }

        var definitions = _mcp.ListDefinitions()
            .Where(d => selection.McpServers.Contains(d.Name, StringComparer.OrdinalIgnoreCase))
            .ToList();

        foreach (var definition in definitions)
        {
            try
            {
                await _mcp.ConnectAsync(definition, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _loggerFactory
                    .CreateLogger<LocalGenKernelFactory>()
                    .LogWarning(ex, "MCP server '{Server}' is unavailable for this kernel.", definition.Name);
            }
        }

        _mcp.RegisterPlugins(kernel);
    }
}
