using System.Net;
using System.Net.Sockets;
using System.Security.Authentication;
using System.Text;
using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.GraviFrame;
using Xunit;
using static Gravicode.Science.DaskNet.Expr;

namespace Gravicode.Science.Tests.DaskNet;

public sealed class TlsTests
{
    private static string Token => new('t', 32);

    /// <summary>
    /// The claim TLS makes, checked on the wire: a relay between client and worker records every
    /// byte, and a marker sent as a task argument (and returned upper-cased as its result) must be
    /// readable in the capture without TLS and absent from it with TLS.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Task_data_is_unreadable_on_the_wire_only_with_tls(bool encrypted)
    {
        var tls = encrypted ? TlsOptions.SelfSigned() : null;
        await using var worker = await Worker.StartAsync(new WorkerOptions { Token = Token, Threads = 1, Tls = tls });
        await using var relay = await Relay.StartAsync(worker.Address);
        await using var client = await Client.ConnectAsync([relay.Address], Token, tls);

        const string marker = "plaintext-marker-7f3a9c";
        var shouted = Dask.Delayed(RemoteFunctions.Shout, Dask.Value(marker)).Compute(client);
        Assert.Equal(marker.ToUpperInvariant(), shouted);

        var wire = Encoding.UTF8.GetString(relay.Captured());
        Assert.True(relay.Captured().Length > 100, "the relay should have seen the traffic");
        Assert.Equal(!encrypted, wire.Contains(marker, StringComparison.Ordinal));
        Assert.Equal(!encrypted, wire.Contains(marker.ToUpperInvariant(), StringComparison.Ordinal));
        Assert.Equal(!encrypted, wire.Contains("DASKNET1", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Tls_clusters_give_the_same_answers_including_peer_transfers()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 3, threadsPerWorker: 2, tls: true);
        Assert.NotNull(cluster.Tls);
        var sales = SalesData.Make(800);
        // A shuffle join: partitions move between workers, so peer connections must use TLS too.
        var left = DaskFrame.FromFrame(sales, 4);
        var right = DaskFrame.FromFrame(sales.Head(300), 3).Select("id", "qty").Rename("qty", "qty2");
        var joined = left.Join(right, "id", broadcast: false).GroupBy("region").Agg(("qty2", "sum"));
        Assert.Equal(SalesData.Rows(joined.Compute()), SalesData.Rows(joined.Compute(cluster.Client)));
        Assert.True(cluster.Client.LastRun!.TasksPerWorker.Count >= 2);

        var a = DaskArray.Random([200, 200], [50, 50]);
        Assert.Equal(a.Sum().Compute(), a.Sum().Compute(cluster.Client));
    }

    [Fact]
    public async Task Mismatched_settings_fail_with_a_reason_instead_of_hanging()
    {
        var tls = TlsOptions.SelfSigned();
        await using var secure = await Worker.StartAsync(new WorkerOptions { Token = Token, Threads = 1, Tls = tls });
        await using var plain = await Worker.StartAsync(new WorkerOptions { Token = Token, Threads = 1 });

        // Plain client, TLS worker: no greeting arrives in plain text.
        var e1 = await Assert.ThrowsAsync<TimeoutException>(() => Client.ConnectAsync([secure.Address], Token));
        Assert.Contains("TLS", e1.Message);

        // TLS client, plain worker: the greeting is not a TLS handshake.
        await Assert.ThrowsAsync<AuthenticationException>(() => Client.ConnectAsync([plain.Address], Token, tls));

        // A different cluster's certificate is not trusted, in either direction.
        await Assert.ThrowsAsync<AuthenticationException>(() => Client.ConnectAsync([secure.Address], Token, TlsOptions.SelfSigned()));

        // Mutual TLS: pinning the worker without presenting a certificate is refused by the worker.
        var noClientCert = new TlsOptions { TrustedCertificates = tls.TrustedCertificates };
        await Assert.ThrowsAsync<AuthenticationException>(() => Client.ConnectAsync([secure.Address], Token, noClientCert));

        // TLS does not replace the token.
        await Assert.ThrowsAsync<UnauthorizedAccessException>(() => Client.ConnectAsync([secure.Address], new string('x', 32), tls));

        // And the right settings work.
        await using var ok = await Client.ConnectAsync([secure.Address], Token, tls);
        Assert.Single(await ok.GetWorkerInfoAsync());
    }

    [Fact]
    public void A_tls_worker_needs_a_certificate_with_its_key()
    {
        Assert.Throws<ArgumentException>(() =>
            Worker.StartAsync(new WorkerOptions { Token = Token, Tls = new TlsOptions() }).GetAwaiter().GetResult());
        var publicOnly = System.Security.Cryptography.X509Certificates.X509CertificateLoader.LoadCertificate(
            TlsOptions.SelfSigned().Certificate!.RawData);
        Assert.Throws<ArgumentException>(() => TlsOptions.FromCertificate(publicOnly));
    }

    [Fact]
    public async Task Process_workers_run_over_tls_and_the_key_file_is_removed()
    {
        var host = Path.Combine(AppContext.BaseDirectory, "DaskNet.Worker.dll");
        string[] before = Directory.GetFiles(Path.GetTempPath(), "dasknet-*.pfx");
        await using (var cluster = await LocalCluster.StartProcessesAsync(host, workers: 2, threadsPerWorker: 1,
                         preload: [typeof(RemoteFunctions).Assembly.Location], tls: true))
        {
            Assert.Equal("ABC", Dask.Delayed(RemoteFunctions.Shout, Dask.Value("abc")).Compute(cluster.Client));
            var x = DaskArray.Random([60, 60], [20, 20]);
            Assert.Equal(x.Sum().Compute(), x.Sum().Compute(cluster.Client));
        }
        Assert.Equal(before, Directory.GetFiles(Path.GetTempPath(), "dasknet-*.pfx"));
    }

    /// <summary>A TCP relay that forwards both ways and keeps a copy of everything.</summary>
    private sealed class Relay : IAsyncDisposable
    {
        private readonly TcpListener _listener = new(IPAddress.Loopback, 0);
        private readonly MemoryStream _captured = new();
        private readonly List<TcpClient> _sockets = [];
        private readonly CancellationTokenSource _stop = new();

        public string Address { get; private set; } = "";

        public static Task<Relay> StartAsync(string target)
        {
            var relay = new Relay();
            relay._listener.Start();
            relay.Address = $"127.0.0.1:{((IPEndPoint)relay._listener.LocalEndpoint).Port}";
            _ = relay.AcceptAsync(target);
            return Task.FromResult(relay);
        }

        public byte[] Captured() { lock (_captured) return _captured.ToArray(); }

        private async Task AcceptAsync(string target)
        {
            while (!_stop.IsCancellationRequested)
            {
                TcpClient inbound;
                try { inbound = await _listener.AcceptTcpClientAsync(_stop.Token); }
                catch (Exception) { return; }
                var colon = target.LastIndexOf(':');
                var outbound = new TcpClient();
                await outbound.ConnectAsync(target[..colon], int.Parse(target[(colon + 1)..]));
                lock (_sockets) _sockets.AddRange([inbound, outbound]);
                _ = PumpAsync(inbound.GetStream(), outbound.GetStream());
                _ = PumpAsync(outbound.GetStream(), inbound.GetStream());
            }
        }

        private async Task PumpAsync(Stream from, Stream to)
        {
            var buffer = new byte[16384];
            try
            {
                int n;
                while ((n = await from.ReadAsync(buffer, _stop.Token)) > 0)
                {
                    lock (_captured) _captured.Write(buffer, 0, n);
                    await to.WriteAsync(buffer.AsMemory(0, n), _stop.Token);
                }
            }
            catch (Exception) { }
        }

        public ValueTask DisposeAsync()
        {
            _stop.Cancel();
            _listener.Stop();
            lock (_sockets) foreach (var s in _sockets) s.Dispose();
            return ValueTask.CompletedTask;
        }
    }
}
