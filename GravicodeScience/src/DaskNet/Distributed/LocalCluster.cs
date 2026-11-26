using System.Diagnostics;
using System.Security.Cryptography;

namespace Gravicode.Science.DaskNet.Distributed;

/// <summary>
/// Workers on this machine plus a connected <see cref="Client"/> (Dask's <c>LocalCluster</c>):
/// in this process for speed and debugging, or as child processes for isolation.
/// </summary>
/// <remarks>
/// <para>
/// In-process workers still talk TCP over loopback and serialise everything, so a graph that runs
/// here runs on a real cluster - the point of using one over a <see cref="ThreadScheduler"/>.
/// </para>
/// <para>
/// Process workers run the <c>DaskNet.Worker</c> host (or any program that calls
/// <see cref="Worker.RunCommandLineAsync"/>). The token is generated here and handed over in the
/// child's environment; each child stops when its standard input closes, so a crashed parent
/// does not leave workers behind.
/// </para>
/// </remarks>
public sealed class LocalCluster : IAsyncDisposable
{
    private readonly List<Worker> _inProcess = [];
    private readonly List<Process> _processes = [];
    private string? _pfxPath;

    private LocalCluster(string token, TlsOptions? tls)
    {
        Token = token;
        Tls = tls;
    }

    /// <summary>The cluster's shared secret.</summary>
    public string Token { get; }

    /// <summary>The TLS settings every node uses, or null when traffic is plain TCP. Give these to
    /// <see cref="Client.ConnectAsync(IEnumerable{string}, string, TlsOptions?, CancellationToken)"/>
    /// to connect a second client.</summary>
    public TlsOptions? Tls { get; }

    /// <summary>The connected client; use it as the scheduler.</summary>
    public Client Client { get; private set; } = null!;

    /// <summary>Worker addresses.</summary>
    public IReadOnlyList<string> Addresses { get; private set; } = [];

    /// <summary>The in-process workers (empty for a process cluster).</summary>
    public IReadOnlyList<Worker> InProcessWorkers => _inProcess;

    /// <summary>The worker processes (empty for an in-process cluster).</summary>
    public IReadOnlyList<Process> Processes => _processes;

    /// <summary>Starts workers in this process. With <paramref name="tls"/>, every connection is
    /// encrypted with a fresh self-signed cluster certificate (<see cref="TlsOptions.SelfSigned"/>).
    /// <paramref name="memoryLimit"/> is each worker's <see cref="WorkerOptions.MemoryLimit"/>.</summary>
    public static async Task<LocalCluster> StartAsync(int workers = 2, int threadsPerWorker = 2, RetryPolicy? retry = null,
        bool tls = false, long? memoryLimit = null)
    {
        if (workers <= 0) throw new ArgumentOutOfRangeException(nameof(workers));
        var cluster = new LocalCluster(NewToken(), tls ? TlsOptions.SelfSigned() : null);
        try
        {
            for (var i = 0; i < workers; i++)
                cluster._inProcess.Add(await Worker.StartAsync(new WorkerOptions
                {
                    Token = cluster.Token, Threads = threadsPerWorker, Name = $"worker-{i}", Tls = cluster.Tls,
                    MemoryLimit = memoryLimit,
                }).ConfigureAwait(false));
            cluster.Addresses = cluster._inProcess.Select(w => w.Address).ToArray();
            cluster.Client = await ConnectClient(cluster, retry).ConfigureAwait(false);
            return cluster;
        }
        catch
        {
            await cluster.DisposeAsync().ConfigureAwait(false);
            throw;
        }
    }

    /// <summary>
    /// Starts workers as child processes running <paramref name="workerHost"/> (a
    /// <c>DaskNet.Worker.dll</c>, or an executable). Assemblies in <paramref name="preload"/> are
    /// loaded by every worker so their <see cref="RemoteTaskAttribute"/> methods can run. With
    /// <paramref name="tls"/>, the cluster certificate is handed to the children as a PKCS#12 file
    /// readable only through a password in their environment, and deleted when the cluster stops.
    /// </summary>
    public static async Task<LocalCluster> StartProcessesAsync(string workerHost, int workers = 2, int threadsPerWorker = 2,
        IEnumerable<string>? preload = null, RetryPolicy? retry = null, TimeSpan? startTimeout = null, bool tls = false,
        long? memoryLimit = null)
    {
        if (!File.Exists(workerHost)) throw new FileNotFoundException("Worker host not found.", workerHost);
        var cluster = new LocalCluster(NewToken(), tls ? TlsOptions.SelfSigned() : null);
        string? pfxPassword = null;
        try
        {
            if (cluster.Tls is not null)
            {
                pfxPassword = NewToken();
                cluster._pfxPath = Path.Combine(Path.GetTempPath(), $"dasknet-{Guid.NewGuid():N}.pfx");
                await File.WriteAllBytesAsync(cluster._pfxPath,
                    cluster.Tls.Certificate!.Export(System.Security.Cryptography.X509Certificates.X509ContentType.Pkcs12, pfxPassword)).ConfigureAwait(false);
            }
            var addresses = new List<string>();
            for (var i = 0; i < workers; i++)
            {
                var isDll = workerHost.EndsWith(".dll", StringComparison.OrdinalIgnoreCase);
                var start = new ProcessStartInfo(isDll ? DotnetHost() : workerHost)
                {
                    RedirectStandardInput = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                if (isDll) start.ArgumentList.Add(workerHost);
                foreach (var a in new[] { "--threads", threadsPerWorker.ToString(System.Globalization.CultureInfo.InvariantCulture), "--name", $"process-{i}" })
                    start.ArgumentList.Add(a);
                if (memoryLimit is { } limit)
                {
                    start.ArgumentList.Add("--memory-limit");
                    start.ArgumentList.Add(limit.ToString(System.Globalization.CultureInfo.InvariantCulture));
                }
                foreach (var p in preload ?? [])
                {
                    start.ArgumentList.Add("--preload");
                    start.ArgumentList.Add(Path.GetFullPath(p));
                }
                start.Environment["DASKNET_TOKEN"] = cluster.Token;
                if (cluster._pfxPath is not null)
                {
                    start.ArgumentList.Add("--tls-pfx");
                    start.ArgumentList.Add(cluster._pfxPath);
                    start.Environment["DASKNET_TLS_PASSWORD"] = pfxPassword;
                }

                var process = Process.Start(start) ?? throw new InvalidOperationException("Could not start a worker process.");
                cluster._processes.Add(process);
                addresses.Add(await ReadAddressAsync(process, startTimeout ?? TimeSpan.FromSeconds(30)).ConfigureAwait(false));
            }
            cluster.Addresses = addresses;
            cluster.Client = await ConnectClient(cluster, retry).ConfigureAwait(false);
            return cluster;
        }
        catch
        {
            await cluster.DisposeAsync().ConfigureAwait(false);
            throw;
        }
    }

    private static async Task<Client> ConnectClient(LocalCluster cluster, RetryPolicy? retry)
    {
        var client = await Client.ConnectAsync(cluster.Addresses, cluster.Token, cluster.Tls).ConfigureAwait(false);
        if (retry is not null) client.Retry = retry;
        return client;
    }

    private static async Task<string> ReadAddressAsync(Process process, TimeSpan timeout)
    {
        using var cts = new CancellationTokenSource(timeout);
        while (!cts.IsCancellationRequested)
        {
            var line = await process.StandardOutput.ReadLineAsync(cts.Token).ConfigureAwait(false);
            if (line is null)
            {
                var error = await process.StandardError.ReadToEndAsync(CancellationToken.None).ConfigureAwait(false);
                throw new InvalidOperationException($"Worker process exited before listening: {error}");
            }
            const string marker = "DASKNET_WORKER_LISTENING ";
            if (line.StartsWith(marker, StringComparison.Ordinal)) return line[marker.Length..].Trim();
        }
        throw new TimeoutException("A worker process did not start listening in time.");
    }

    private static string DotnetHost()
    {
        var root = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH");
        if (!string.IsNullOrEmpty(root) && File.Exists(root)) return root;
        var current = Process.GetCurrentProcess().MainModule?.FileName;
        if (current is not null && Path.GetFileNameWithoutExtension(current).Equals("dotnet", StringComparison.OrdinalIgnoreCase))
            return current;
        return "dotnet";
    }

    private static string NewToken() => Convert.ToHexString(RandomNumberGenerator.GetBytes(24));

    /// <summary>Kills worker <paramref name="index"/> abruptly, as a crash would. For testing recovery.</summary>
    public void KillWorker(int index)
    {
        if (_inProcess.Count > 0) _inProcess[index].Kill();
        else _processes[index].Kill(entireProcessTree: true);
    }

    /// <summary>Disconnects the client and stops every worker.</summary>
    public async ValueTask DisposeAsync()
    {
        if (Client is not null) await Client.DisposeAsync().ConfigureAwait(false);
        foreach (var w in _inProcess) await w.DisposeAsync().ConfigureAwait(false);
        foreach (var p in _processes)
        {
            try
            {
                p.StandardInput.Close();
                if (!p.WaitForExit(3000)) p.Kill(entireProcessTree: true);
            }
            catch (InvalidOperationException) { }
            p.Dispose();
        }
        if (_pfxPath is not null)
        {
            try { File.Delete(_pfxPath); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }
}
