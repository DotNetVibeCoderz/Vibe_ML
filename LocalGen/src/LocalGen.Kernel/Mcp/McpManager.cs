using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.Json.Serialization;
using LocalGen.Core;
using LocalGen.Core.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.SemanticKernel;
using ModelContextProtocol.Client;

namespace LocalGen.Kernel.Mcp;

/// <summary>How to reach one MCP server.</summary>
public sealed record McpServerDefinition
{
    public required string Name { get; init; }

    public string Description { get; init; } = string.Empty;

    /// <summary>Executable for a stdio server, e.g. <c>npx</c>. Empty for HTTP servers.</summary>
    public string Command { get; init; } = string.Empty;

    public IReadOnlyList<string> Arguments { get; init; } = [];

    /// <summary>Endpoint for an HTTP server. Empty for stdio servers.</summary>
    public string Url { get; init; } = string.Empty;

    public bool Enabled { get; init; } = true;

    [JsonIgnore]
    public bool IsHttp => !string.IsNullOrWhiteSpace(Url);
}

/// <summary>A connected server and the tools it exposes.</summary>
public sealed record McpConnection
{
    public required McpServerDefinition Definition { get; init; }

    public required McpClient Client { get; init; }

    public required IReadOnlyList<McpClientTool> Tools { get; init; }

    public string ServerName => Client.ServerInfo?.Name ?? Definition.Name;

    public string? Instructions => Client.ServerInstructions;
}

/// <summary>
/// Connects to Model Context Protocol servers and exposes their tools to the kernel.
/// </summary>
/// <remarks>
/// MCP is how LocalGen reaches capabilities it does not implement itself — a database, an issue
/// tracker, a company's internal API. Connections are long-lived because a stdio server is a
/// child process, so they are cached per definition and only torn down on explicit disconnect.
/// </remarks>
public sealed class McpManager : IAsyncDisposable
{
    private readonly LocalGenOptions _options;
    private readonly ILoggerFactory _loggerFactory;
    private readonly ILogger<McpManager> _logger;
    private readonly ConcurrentDictionary<string, McpConnection> _connections = new(StringComparer.OrdinalIgnoreCase);

    public McpManager(
        IOptions<LocalGenOptions> options,
        ILoggerFactory loggerFactory)
    {
        _options = options.Value;
        _loggerFactory = loggerFactory;
        _logger = loggerFactory.CreateLogger<McpManager>();
    }

    /// <summary>Where the configured servers are persisted.</summary>
    public string ConfigurationPath => Path.Combine(_options.DataDirectory, "mcp.json");

    public IReadOnlyList<McpConnection> Connections => [.. _connections.Values];

    /// <summary>Servers the user has configured, whether connected or not.</summary>
    public IReadOnlyList<McpServerDefinition> ListDefinitions()
    {
        if (!File.Exists(ConfigurationPath))
        {
            return [];
        }

        try
        {
            var json = File.ReadAllText(ConfigurationPath);
            return JsonSerializer.Deserialize<List<McpServerDefinition>>(json, SerializerOptions) ?? [];
        }
        catch (JsonException ex)
        {
            _logger.LogWarning(ex, "mcp.json is not valid JSON; treating it as empty.");
            return [];
        }
    }

    public void SaveDefinitions(IReadOnlyList<McpServerDefinition> definitions)
    {
        Directory.CreateDirectory(_options.DataDirectory);
        File.WriteAllText(
            ConfigurationPath,
            JsonSerializer.Serialize(definitions, SerializerOptions));
    }

    public void AddDefinition(McpServerDefinition definition)
    {
        var definitions = ListDefinitions()
            .Where(d => !string.Equals(d.Name, definition.Name, StringComparison.OrdinalIgnoreCase))
            .Append(definition)
            .ToList();

        SaveDefinitions(definitions);
    }

    public async ValueTask<bool> RemoveDefinitionAsync(string name)
    {
        var definitions = ListDefinitions();
        var remaining = definitions
            .Where(d => !string.Equals(d.Name, name, StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (remaining.Count == definitions.Count)
        {
            return false;
        }

        await DisconnectAsync(name).ConfigureAwait(false);
        SaveDefinitions(remaining);
        return true;
    }

    /// <summary>Connects to a server and caches the session. Reconnecting returns the cached one.</summary>
    public async ValueTask<McpConnection> ConnectAsync(
        McpServerDefinition definition,
        CancellationToken cancellationToken = default)
    {
        if (_connections.TryGetValue(definition.Name, out var existing))
        {
            return existing;
        }

        if (definition.IsHttp && _options.Runtime.OfflineMode)
        {
            throw new OfflineModeException($"connect to MCP server '{definition.Name}'");
        }

        IClientTransport transport = definition.IsHttp
            ? new HttpClientTransport(
                new HttpClientTransportOptions
                {
                    Endpoint = new Uri(definition.Url),
                    Name = definition.Name
                },
                _loggerFactory)
            : new StdioClientTransport(
                new StdioClientTransportOptions
                {
                    Command = definition.Command,
                    Arguments = [.. definition.Arguments],
                    Name = definition.Name
                },
                _loggerFactory);

        _logger.LogInformation("Connecting to MCP server '{Server}'…", definition.Name);

        var client = await McpClient
            .CreateAsync(transport, loggerFactory: _loggerFactory, cancellationToken: cancellationToken)
            .ConfigureAwait(false);

        var tools = await client.ListToolsAsync(cancellationToken: cancellationToken).ConfigureAwait(false);

        var connection = new McpConnection
        {
            Definition = definition,
            Client = client,
            Tools = [.. tools]
        };

        _connections[definition.Name] = connection;

        _logger.LogInformation(
            "Connected to '{Server}' ({ToolCount} tools).",
            connection.ServerName, connection.Tools.Count);

        return connection;
    }

    /// <summary>Connects every enabled server, skipping any that fail so one bad entry cannot block the rest.</summary>
    public async ValueTask<IReadOnlyList<McpConnection>> ConnectAllAsync(
        CancellationToken cancellationToken = default)
    {
        foreach (var definition in ListDefinitions().Where(static d => d.Enabled))
        {
            try
            {
                await ConnectAsync(definition, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _logger.LogWarning(
                    ex, "Could not connect to MCP server '{Server}'; skipping it.", definition.Name);
            }
        }

        return Connections;
    }

    public async ValueTask<bool> DisconnectAsync(string name)
    {
        if (!_connections.TryRemove(name, out var connection))
        {
            return false;
        }

        await connection.Client.DisposeAsync().ConfigureAwait(false);
        _logger.LogInformation("Disconnected from MCP server '{Server}'.", name);
        return true;
    }

    /// <summary>
    /// Registers each connected server's tools as a kernel plugin, one plugin per server so that
    /// two servers exposing a "search" tool do not collide.
    /// </summary>
    public void RegisterPlugins(Microsoft.SemanticKernel.Kernel kernel)
    {
        foreach (var connection in _connections.Values)
        {
            if (connection.Tools.Count == 0)
            {
                continue;
            }

            var pluginName = SanitizePluginName(connection.Definition.Name);

            if (kernel.Plugins.Contains(pluginName))
            {
                kernel.Plugins.Remove(kernel.Plugins[pluginName]);
            }

            kernel.Plugins.AddFromFunctions(
                pluginName,
                connection.Tools.Select(static tool => tool.AsKernelFunction()));
        }
    }

    /// <summary>
    /// Well-known servers offered in the MCP gallery. Shipping a curated starting set is what
    /// makes the gallery useful before a user knows any server names.
    /// </summary>
    public static IReadOnlyList<McpServerDefinition> Gallery { get; } =
    [
        new McpServerDefinition
        {
            Name = "filesystem",
            Description = "Read and write files in a directory you nominate.",
            Command = "npx",
            Arguments = ["-y", "@modelcontextprotocol/server-filesystem", "."]
        },
        new McpServerDefinition
        {
            Name = "git",
            Description = "Inspect a Git repository: log, diff, blame and status.",
            Command = "uvx",
            Arguments = ["mcp-server-git", "--repository", "."]
        },
        new McpServerDefinition
        {
            Name = "sqlite",
            Description = "Query a SQLite database.",
            Command = "uvx",
            Arguments = ["mcp-server-sqlite", "--db-path", "database.db"]
        },
        new McpServerDefinition
        {
            Name = "fetch",
            Description = "Fetch a URL and convert it to markdown.",
            Command = "uvx",
            Arguments = ["mcp-server-fetch"]
        },
        new McpServerDefinition
        {
            Name = "memory",
            Description = "A knowledge graph the model can persist facts into across sessions.",
            Command = "npx",
            Arguments = ["-y", "@modelcontextprotocol/server-memory"]
        },
        new McpServerDefinition
        {
            Name = "github",
            Description = "Search repositories, issues and pull requests on GitHub.",
            Command = "npx",
            Arguments = ["-y", "@modelcontextprotocol/server-github"]
        }
    ];

    /// <summary>Kernel plugin names must be identifier-like, but server names are free-form.</summary>
    private static string SanitizePluginName(string name)
    {
        var sanitized = new string([.. name.Select(static c => char.IsLetterOrDigit(c) ? c : '_')]);
        return char.IsDigit(sanitized.FirstOrDefault()) ? "mcp_" + sanitized : sanitized;
    }

    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    public async ValueTask DisposeAsync()
    {
        foreach (var connection in _connections.Values)
        {
            await connection.Client.DisposeAsync().ConfigureAwait(false);
        }

        _connections.Clear();
    }
}
