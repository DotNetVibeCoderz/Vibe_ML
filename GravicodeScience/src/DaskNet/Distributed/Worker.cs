using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Reflection;

namespace Gravicode.Science.DaskNet.Distributed;

/// <summary>How a worker listens and runs.</summary>
public sealed record WorkerOptions
{
    /// <summary>The address to listen on. Loopback by default: exposing a worker is a decision.</summary>
    public string Host { get; init; } = "127.0.0.1";

    /// <summary>The port; 0 picks a free one.</summary>
    public int Port { get; init; }

    /// <summary>Tasks run at once.</summary>
    public int Threads { get; init; } = Environment.ProcessorCount;

    /// <summary>The shared secret clients and peers must prove they know (at least 16 characters).</summary>
    public required string Token { get; init; }

    /// <summary>A name for logs and statistics; defaults to the address.</summary>
    public string? Name { get; init; }
}

/// <summary>
/// A process that runs tasks for a <see cref="Client"/> and keeps their results until told to
/// release them (Dask's <c>dask-worker</c>).
/// </summary>
/// <remarks>
/// <para>
/// Results stay where they were computed. A task's inputs are fetched from whichever worker holds
/// them - peer to peer, not through the client - and the fetched copy is kept and reported, so the
/// client can send later readers of the same data to where it already is.
/// </para>
/// <para>
/// Only <see cref="RemoteTaskAttribute"/> methods run, and only for callers that pass the token
/// handshake. A worker hosts any assembly it has loaded; the <c>DaskNet.Worker</c> host loads more
/// with <c>--preload</c>.
/// </para>
/// </remarks>
public sealed class Worker : IAsyncDisposable
{
    private readonly WorkerOptions _options;
    private readonly TcpListener _listener;
    private readonly ConcurrentDictionary<TaskKey, object?> _data = new();
    private readonly ConcurrentDictionary<string, Lazy<Task<RpcClient>>> _peers = new(StringComparer.Ordinal);
    private readonly ConcurrentBag<WireConnection> _connections = [];
    private readonly SemaphoreSlim _slots;
    private readonly CancellationTokenSource _stop = new();
    private long _tasksRun;

    private Worker(WorkerOptions options, TcpListener listener)
    {
        _options = options;
        _listener = listener;
        _slots = new SemaphoreSlim(options.Threads, options.Threads);
        Address = WireConnection.FormatAddress((IPEndPoint)listener.LocalEndpoint);
        Name = options.Name ?? Address;
    }

    /// <summary>Where clients connect: <c>host:port</c>.</summary>
    public string Address { get; }

    /// <summary>The worker's name.</summary>
    public string Name { get; }

    /// <summary>Tasks run so far.</summary>
    public long TasksRun => Interlocked.Read(ref _tasksRun);

    /// <summary>Results held now.</summary>
    public int StoredKeys => _data.Count;

    /// <summary>Starts listening.</summary>
    public static Task<Worker> StartAsync(WorkerOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (string.IsNullOrEmpty(options.Token) || options.Token.Length < 16)
            throw new ArgumentException("A worker needs a token of at least 16 characters.", nameof(options));
        if (options.Threads <= 0) throw new ArgumentOutOfRangeException(nameof(options), "Threads must be positive.");

        var address = options.Host is "localhost" ? IPAddress.Loopback
            : options.Host is "*" or "0.0.0.0" ? IPAddress.Any : IPAddress.Parse(options.Host);
        var listener = new TcpListener(address, options.Port);
        listener.Start();
        var worker = new Worker(options, listener);
        _ = worker.AcceptLoopAsync();
        return Task.FromResult(worker);
    }

    private async Task AcceptLoopAsync()
    {
        while (!_stop.IsCancellationRequested)
        {
            TcpClient client;
            try { client = await _listener.AcceptTcpClientAsync(_stop.Token).ConfigureAwait(false); }
            catch (Exception) { break; }
            _ = Task.Run(() => ServeAsync(client));
        }
    }

    private async Task ServeAsync(TcpClient client)
    {
        var connection = await WireConnection.AcceptAsync(client, _options.Token, _stop.Token).ConfigureAwait(false);
        if (connection is null) return;
        _connections.Add(connection);
        try
        {
            while (!_stop.IsCancellationRequested)
            {
                var message = await connection.ReceiveAsync(_stop.Token).ConfigureAwait(false);
                switch (message.Kind)
                {
                    case MessageKind.Run:
                        _ = Task.Run(() => RunAsync(connection, message));
                        break;
                    case MessageKind.Get:
                        await ReplyGetAsync(connection, message).ConfigureAwait(false);
                        break;
                    case MessageKind.Release:
                    {
                        using var r = message.Reader();
                        var count = r.ReadInt32();
                        for (var i = 0; i < count; i++) _data.TryRemove(WireCodec.ReadKey(r), out _);
                        await connection.SendAsync(MessageKind.Ok, message.Id).ConfigureAwait(false);
                        break;
                    }
                    case MessageKind.Ping:
                        await connection.SendAsync(MessageKind.Pong, message.Id, w =>
                        {
                            w.Write(Name);
                            w.Write(_options.Threads);
                            w.Write(_data.Count);
                            w.Write(TasksRun);
                            w.Write(Environment.ProcessId);
                        }).ConfigureAwait(false);
                        break;
                }
            }
        }
        catch (Exception) { /* the peer went away */ }
        finally { await connection.DisposeAsync().ConfigureAwait(false); }
    }

    private async Task ReplyGetAsync(WireConnection connection, Message message)
    {
        using var r = message.Reader();
        var key = WireCodec.ReadKey(r);
        if (_data.TryGetValue(key, out var value))
        {
            var bytes = WireCodec.Serialize(value);
            await connection.SendAsync(MessageKind.Data, message.Id, w => { w.Write(bytes.Length); w.Write(bytes); }).ConfigureAwait(false);
        }
        else
        {
            await connection.SendAsync(MessageKind.Error, message.Id, w =>
            {
                w.Write(nameof(KeyNotFoundException));
                w.Write($"{Name} does not hold {key}.");
                w.Write(true);
                WireCodec.WriteKey(w, key);
            }).ConfigureAwait(false);
        }
    }

    private async Task RunAsync(WireConnection connection, Message message)
    {
        TaskKey key = default;
        RemoteCall call;
        List<(TaskKey Key, string Address)> deps;
        try
        {
            using var r = message.Reader();
            key = WireCodec.ReadKey(r);
            call = (RemoteCall)WireCodec.Read(r)!;
            var count = r.ReadInt32();
            deps = new List<(TaskKey, string)>(count);
            for (var i = 0; i < count; i++) deps.Add((WireCodec.ReadKey(r), r.ReadString()));
        }
        catch (Exception e)
        {
            await SendErrorAsync(connection, message.Id, e, null).ConfigureAwait(false);
            return;
        }

        await _slots.WaitAsync(_stop.Token).ConfigureAwait(false);
        try
        {
            var inputs = new Dictionary<TaskKey, object?>();
            var fetched = new List<TaskKey>();
            foreach (var (dep, holder) in deps)
            {
                if (_data.TryGetValue(dep, out var local)) { inputs[dep] = local; continue; }
                try
                {
                    var value = await FetchAsync(dep, holder).ConfigureAwait(false);
                    _data[dep] = value;
                    inputs[dep] = value;
                    fetched.Add(dep);
                }
                catch (Exception e)
                {
                    await SendErrorAsync(connection, message.Id, e, dep).ConfigureAwait(false);
                    return;
                }
            }

            var clock = Stopwatch.StartNew();
            object? result;
            using (var activity = DaskTelemetry.Source.StartActivity(call.Method.Name))
            {
                activity?.SetTag("dask.key", key.ToString());
                activity?.SetTag("dask.worker", Name);
                try { result = call.Invoke(k => inputs[k]); }
                catch (Exception e)
                {
                    activity?.SetStatus(ActivityStatusCode.Error, e.Message);
                    await SendErrorAsync(connection, message.Id, e, null).ConfigureAwait(false);
                    return;
                }
            }
            _data[key] = result;
            Interlocked.Increment(ref _tasksRun);
            var elapsed = clock.Elapsed.TotalMilliseconds;
            await connection.SendAsync(MessageKind.Done, message.Id, w =>
            {
                WireCodec.WriteKey(w, key);
                w.Write(elapsed);
                w.Write(fetched.Count);
                foreach (var f in fetched) WireCodec.WriteKey(w, f);
            }).ConfigureAwait(false);
        }
        catch (Exception) { /* the client went away mid-task */ }
        finally { _slots.Release(); }
    }

    private async Task<object?> FetchAsync(TaskKey key, string holder)
    {
        if (string.IsNullOrEmpty(holder) || holder == Address)
            throw new KeyNotFoundException($"{Name} was told it holds {key}, and does not.");
        var peer = await _peers.GetOrAdd(holder, a => new Lazy<Task<RpcClient>>(() => RpcClient.ConnectAsync(a, _options.Token))).Value
            .ConfigureAwait(false);
        if (peer.IsClosed)
        {
            _peers.TryRemove(holder, out _);
            throw new WorkerLostException(holder);
        }
        var reply = await peer.RequestAsync(MessageKind.Get, w => WireCodec.WriteKey(w, key)).ConfigureAwait(false);
        using var r = reply.Reader();
        if (reply.Kind == MessageKind.Error) throw new KeyNotFoundException($"{holder} does not hold {key}.");
        return WireCodec.Deserialize(r.ReadBytes(r.ReadInt32()));
    }

    private static Task SendErrorAsync(WireConnection connection, long id, Exception e, TaskKey? missing)
        => connection.SendAsync(MessageKind.Error, id, w =>
        {
            w.Write(e.GetType().Name);
            w.Write(e.Message);
            w.Write(missing is not null);
            if (missing is { } m) WireCodec.WriteKey(w, m);
        });

    /// <summary>
    /// Drops every connection and all data at once, as a crash would. For testing recovery; a
    /// normal shutdown is <see cref="DisposeAsync"/>.
    /// </summary>
    public void Kill()
    {
        _stop.Cancel();
        _listener.Stop();
        foreach (var c in _connections) c.Abort();
        _data.Clear();
    }

    /// <summary>Stops listening and closes every connection.</summary>
    public async ValueTask DisposeAsync()
    {
        if (!_stop.IsCancellationRequested) _stop.Cancel();
        _listener.Stop();
        foreach (var c in _connections) await c.DisposeAsync().ConfigureAwait(false);
        foreach (var p in _peers.Values)
            if (p.IsValueCreated && p.Value.IsCompletedSuccessfully) await p.Value.Result.DisposeAsync().ConfigureAwait(false);
        _data.Clear();
    }

    /// <summary>
    /// Runs a worker from command-line arguments until the process is stopped or standard input
    /// closes: <c>--host</c>, <c>--port</c>, <c>--threads</c>, <c>--name</c> and any number of
    /// <c>--preload path.dll</c>. The token comes from the <c>DASKNET_TOKEN</c> environment
    /// variable, not the command line, where other users could read it from the process list.
    /// Prints <c>DASKNET_WORKER_LISTENING host:port</c> once ready.
    /// </summary>
    public static async Task<int> RunCommandLineAsync(string[] args, TextWriter? output = null)
    {
        output ??= Console.Out;
        string host = "127.0.0.1", name = "";
        int port = 0, threads = Environment.ProcessorCount;
        var preload = new List<string>();
        for (var i = 0; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");
            switch (args[i])
            {
                case "--host": host = Next(); break;
                case "--port": port = int.Parse(Next(), System.Globalization.CultureInfo.InvariantCulture); break;
                case "--threads": threads = int.Parse(Next(), System.Globalization.CultureInfo.InvariantCulture); break;
                case "--name": name = Next(); break;
                case "--preload": preload.Add(Next()); break;
                default: throw new ArgumentException($"Unknown option {args[i]}.");
            }
        }

        var token = Environment.GetEnvironmentVariable("DASKNET_TOKEN");
        if (string.IsNullOrEmpty(token))
        {
            await output.WriteLineAsync("DASKNET_TOKEN is not set.").ConfigureAwait(false);
            return 2;
        }

        Preload(preload);
        await using var worker = await StartAsync(new WorkerOptions
        {
            Host = host, Port = port, Threads = threads, Token = token, Name = name.Length > 0 ? name : null,
        }).ConfigureAwait(false);
        await output.WriteLineAsync($"DASKNET_WORKER_LISTENING {worker.Address}").ConfigureAwait(false);
        await output.FlushAsync().ConfigureAwait(false);

        // Standard input closing is the parent's way of saying "stop" that also works when the
        // parent itself dies; Ctrl+C is the interactive way.
        var stdinClosed = Task.Run(() => { try { while (Console.In.Read() != -1) { } } catch (IOException) { } });
        var cancel = new TaskCompletionSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancel.TrySetResult(); };
        await Task.WhenAny(stdinClosed, cancel.Task).ConfigureAwait(false);
        return 0;
    }

    /// <summary>Loads assemblies so their <see cref="RemoteTaskAttribute"/> methods can be called,
    /// resolving their dependencies from the same folders.</summary>
    public static void Preload(IEnumerable<string> paths)
    {
        var folders = new List<string>();
        foreach (var path in paths)
        {
            var full = Path.GetFullPath(path);
            folders.Add(Path.GetDirectoryName(full)!);
            Assembly.LoadFrom(full);
        }
        if (folders.Count == 0) return;
        AppDomain.CurrentDomain.AssemblyResolve += (_, e) =>
        {
            var file = new AssemblyName(e.Name).Name + ".dll";
            foreach (var folder in folders)
            {
                var candidate = Path.Combine(folder, file);
                if (File.Exists(candidate)) return Assembly.LoadFrom(candidate);
            }
            return null;
        };
    }
}
