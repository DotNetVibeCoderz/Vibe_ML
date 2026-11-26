using System.Collections;
using System.Net;

namespace Gravicode.Science.DaskNet.Distributed;

/// <summary>Finds worker addresses (<c>host:port</c>) for a <see cref="Client"/>.</summary>
/// <remarks>Discovery may return addresses that are not listening yet - a pod that is starting, a
/// stale DNS record. The client skips those and tries them again on its next refresh.</remarks>
public interface IWorkerDiscovery
{
    /// <summary>The addresses known now.</summary>
    Task<IReadOnlyList<string>> DiscoverAsync(CancellationToken cancellationToken = default);

    /// <summary>What is being asked, for error messages.</summary>
    string Describe();
}

/// <summary>
/// Ready-made <see cref="IWorkerDiscovery"/> sources. None needs a package: Kubernetes is DNS on a
/// headless Service, Aspire is the environment variables its service discovery injects.
/// </summary>
public static class WorkerDiscovery
{
    /// <summary>A fixed list.</summary>
    public static IWorkerDiscovery Static(params string[] addresses)
        => new Source($"static [{string.Join(", ", addresses)}]", _ => Task.FromResult<IReadOnlyList<string>>(addresses.ToArray()));

    /// <summary>
    /// Every address <paramref name="host"/> resolves to, each with <paramref name="port"/>. For
    /// Kubernetes, give the workers a headless Service (<c>clusterIP: None</c>): its name resolves to
    /// one record per ready pod, so scaling the Deployment changes what this returns.
    /// </summary>
    public static IWorkerDiscovery Dns(string host, int port = 8786)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(host);
        return new Source($"DNS {host}:{port}", async ct =>
        {
            var addresses = await System.Net.Dns.GetHostAddressesAsync(host, ct).ConfigureAwait(false);
            return addresses.Distinct().Select(a => WireConnection.FormatAddress(new IPEndPoint(a, port))).ToArray();
        });
    }

    /// <summary>
    /// .NET Aspire service discovery: an AppHost that references a resource named
    /// <paramref name="service"/> injects <c>services__{service}__{endpoint}__{index}</c> variables
    /// holding URLs such as <c>tcp://localhost:51234</c>. With no <paramref name="endpoint"/>, every
    /// endpoint of the service is used. A trailing <c>*</c> matches several resources
    /// (<c>dask-worker-*</c> for <c>dask-worker-0</c>, <c>dask-worker-1</c>, ...): replicas made with
    /// <c>WithReplicas</c> share one proxied address, which would look like a single worker, so give
    /// each worker its own resource.
    /// </summary>
    /// <param name="service">The resource name in the AppHost, optionally ending in <c>*</c>.</param>
    /// <param name="endpoint">The endpoint name (<c>WithEndpoint(name: ...)</c>), or null for all.</param>
    /// <param name="variables">Variables to read instead of the process environment (for tests).</param>
    public static IWorkerDiscovery Aspire(string service, string? endpoint = null, IReadOnlyDictionary<string, string>? variables = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(service);
        var wildcard = service.EndsWith('*');
        var stem = wildcard ? service[..^1] : service;
        return new Source($"Aspire service '{service}'{(endpoint is null ? "" : $" endpoint '{endpoint}'")}", _ =>
        {
            var found = new List<(string Service, string Endpoint, int Index, string Address)>();
            foreach (var (name, value) in variables ?? EnvironmentVariables())
            {
                if (!name.StartsWith("services__", StringComparison.OrdinalIgnoreCase)) continue;
                var parts = name["services__".Length..].Split("__");   // service, endpoint, index
                if (parts.Length != 3 || !int.TryParse(parts[2], out var index)) continue;
                var matches = wildcard
                    ? parts[0].StartsWith(stem, StringComparison.OrdinalIgnoreCase)
                    : parts[0].Equals(stem, StringComparison.OrdinalIgnoreCase);
                if (!matches) continue;
                if (endpoint is not null && !parts[1].Equals(endpoint, StringComparison.OrdinalIgnoreCase)) continue;
                found.Add((parts[0], parts[1], index, ToAddress(value)));
            }
            return Task.FromResult<IReadOnlyList<string>>(found
                .OrderBy(f => f.Service, StringComparer.OrdinalIgnoreCase).ThenBy(f => f.Endpoint, StringComparer.OrdinalIgnoreCase)
                .ThenBy(f => f.Index).Select(f => f.Address).Distinct().ToArray());
        });
    }

    /// <summary>A comma- or semicolon-separated list in an environment variable (<c>DASKNET_WORKERS</c>
    /// by default), read on every refresh.</summary>
    public static IWorkerDiscovery EnvironmentVariable(string name = "DASKNET_WORKERS")
        => new Source($"environment variable {name}", _ => Task.FromResult<IReadOnlyList<string>>(
            (Environment.GetEnvironmentVariable(name) ?? "").Split([',', ';'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)));

    /// <summary>Any lookup you write - a service registry, a cloud API.</summary>
    public static IWorkerDiscovery From(Func<CancellationToken, Task<IReadOnlyList<string>>> lookup, string description = "custom")
        => new Source(description, lookup);

    /// <summary>Everything the sources find, without duplicates. One source failing does not hide the others.</summary>
    public static IWorkerDiscovery Combine(params IWorkerDiscovery[] sources)
        => new Source(string.Join(" + ", sources.Select(s => s.Describe())), async ct =>
        {
            var results = await Task.WhenAll(sources.Select(async s =>
            {
                try { return await s.DiscoverAsync(ct).ConfigureAwait(false); }
                catch (Exception e) when (e is not OperationCanceledException) { return Array.Empty<string>(); }
            })).ConfigureAwait(false);
            return results.SelectMany(r => r).Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
        });

    /// <summary><c>tcp://host:port</c>, <c>http://[::1]:80</c> or a bare <c>host:port</c> to <c>host:port</c>.</summary>
    internal static string ToAddress(string value)
    {
        value = value.Trim();
        if (!value.Contains("://", StringComparison.Ordinal)) return value;
        var uri = new Uri(value);
        return $"{uri.Host}:{uri.Port}";   // Host keeps an IPv6 address's brackets
    }

    private static IEnumerable<KeyValuePair<string, string>> EnvironmentVariables()
    {
        foreach (DictionaryEntry e in Environment.GetEnvironmentVariables())
            if (e.Key is string k && e.Value is string v) yield return new(k, v);
    }

    private sealed class Source(string description, Func<CancellationToken, Task<IReadOnlyList<string>>> lookup) : IWorkerDiscovery
    {
        public Task<IReadOnlyList<string>> DiscoverAsync(CancellationToken cancellationToken = default) => lookup(cancellationToken);
        public string Describe() => description;
    }
}
