using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;

namespace Gravicode.Science.DaskNet.Distributed;

/// <summary>A worker went away while something still depended on it.</summary>
public sealed class WorkerLostException(string address, Exception? inner = null)
    : Exception($"Lost the connection to worker {address}.", inner)
{
    /// <summary>The worker's address.</summary>
    public string Address { get; } = address;
}

internal enum MessageKind : byte
{
    Run = 1, Get = 2, Release = 3, Ping = 4,
    Done = 10, Error = 11, Data = 12, Pong = 13, Ok = 14,
}

internal readonly record struct Message(MessageKind Kind, long Id, byte[] Body)
{
    public BinaryReader Reader() => new(new MemoryStream(Body), Encoding.UTF8);
}

/// <summary>
/// One authenticated TCP connection: length-prefixed frames of <c>[kind][request id][body]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Authentication is a challenge-response on a shared token: the listener sends a random nonce,
/// the caller answers with HMAC-SHA256(token, nonce), and the listener compares in constant time.
/// The token itself never crosses the wire, and a recorded answer is useless against a new nonce.
/// </para>
/// <para>
/// The traffic is not encrypted. Keep clusters on a private network, or tunnel them; TLS is on
/// the roadmap. The token stops strangers from running tasks - which, since tasks run code, is
/// the property that matters most - not eavesdroppers from reading data.
/// </para>
/// </remarks>
internal sealed class WireConnection : IAsyncDisposable
{
    private static readonly byte[] Magic = "DASKNET1"u8.ToArray();
    private readonly TcpClient _client;
    private readonly NetworkStream _stream;
    private readonly SemaphoreSlim _writeLock = new(1, 1);

    private WireConnection(TcpClient client, string remote)
    {
        _client = client;
        _client.NoDelay = true;
        _stream = client.GetStream();
        Remote = remote;
    }

    public string Remote { get; }

    public static async Task<WireConnection> ConnectAsync(string address, string token, CancellationToken cancellationToken = default)
    {
        var (host, port) = ParseAddress(address);
        var client = new TcpClient();
        try
        {
            await client.ConnectAsync(host, port, cancellationToken).ConfigureAwait(false);
            var connection = new WireConnection(client, address);
            var hello = await connection.ReadExactAsync(Magic.Length + 32, cancellationToken).ConfigureAwait(false);
            if (!hello.AsSpan(0, Magic.Length).SequenceEqual(Magic))
                throw new InvalidDataException($"{address} is not a Dask.Net worker.");
            var answer = HMACSHA256.HashData(Encoding.UTF8.GetBytes(token), hello.AsSpan(Magic.Length));
            await connection._stream.WriteAsync(answer, cancellationToken).ConfigureAwait(false);
            var verdict = await connection.ReadExactAsync(1, cancellationToken).ConfigureAwait(false);
            if (verdict[0] != 1) throw new UnauthorizedAccessException($"Worker {address} rejected the token.");
            return connection;
        }
        catch
        {
            client.Dispose();
            throw;
        }
    }

    /// <summary>Server side of the handshake; returns null (and closes) if the caller fails it.</summary>
    public static async Task<WireConnection?> AcceptAsync(TcpClient client, string token, CancellationToken cancellationToken)
    {
        var connection = new WireConnection(client, client.Client.RemoteEndPoint?.ToString() ?? "?");
        try
        {
            var nonce = RandomNumberGenerator.GetBytes(32);
            await connection._stream.WriteAsync(Magic.Concat(nonce).ToArray(), cancellationToken).ConfigureAwait(false);
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeout.CancelAfter(TimeSpan.FromSeconds(10));
            var answer = await connection.ReadExactAsync(32, timeout.Token).ConfigureAwait(false);
            var expected = HMACSHA256.HashData(Encoding.UTF8.GetBytes(token), nonce);
            var ok = CryptographicOperations.FixedTimeEquals(answer, expected);
            await connection._stream.WriteAsync(new[] { ok ? (byte)1 : (byte)0 }, cancellationToken).ConfigureAwait(false);
            if (ok) return connection;
        }
        catch (Exception e) when (e is IOException or OperationCanceledException or SocketException) { }
        await connection.DisposeAsync().ConfigureAwait(false);
        return null;
    }

    public async Task SendAsync(MessageKind kind, long id, Action<BinaryWriter>? body = null, CancellationToken cancellationToken = default)
    {
        using var ms = new MemoryStream();
        ms.Write(new byte[4]);
        using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
        {
            w.Write((byte)kind);
            w.Write(id);
            body?.Invoke(w);
        }
        var frame = ms.GetBuffer();
        BinaryPrimitives.WriteInt32LittleEndian(frame, (int)ms.Length - 4);

        await _writeLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try { await _stream.WriteAsync(frame.AsMemory(0, (int)ms.Length), cancellationToken).ConfigureAwait(false); }
        finally { _writeLock.Release(); }
    }

    public async Task<Message> ReceiveAsync(CancellationToken cancellationToken = default)
    {
        var header = await ReadExactAsync(4, cancellationToken).ConfigureAwait(false);
        var length = BinaryPrimitives.ReadInt32LittleEndian(header);
        if (length < 9) throw new InvalidDataException("Malformed frame.");
        var frame = await ReadExactAsync(length, cancellationToken).ConfigureAwait(false);
        return new Message((MessageKind)frame[0], BinaryPrimitives.ReadInt64LittleEndian(frame.AsSpan(1)), frame[9..]);
    }

    private async Task<byte[]> ReadExactAsync(int count, CancellationToken cancellationToken)
    {
        var buffer = new byte[count];
        await _stream.ReadExactlyAsync(buffer, cancellationToken).ConfigureAwait(false);
        return buffer;
    }

    public void Abort()
    {
        // A connection that already closed has no socket left (Client is null after Dispose).
        try { _client.Client?.LingerState = new LingerOption(true, 0); } catch (SocketException) { } catch (ObjectDisposedException) { }
        _client.Dispose();
    }

    public ValueTask DisposeAsync()
    {
        _client.Dispose();
        _writeLock.Dispose();
        return ValueTask.CompletedTask;
    }

    public static (string Host, int Port) ParseAddress(string address)
    {
        var colon = address.LastIndexOf(':');
        if (colon <= 0 || !int.TryParse(address[(colon + 1)..], out var port))
            throw new FormatException($"'{address}' is not host:port.");
        return (address[..colon].Trim('[', ']'), port);
    }

    public static string FormatAddress(IPEndPoint endpoint)
        => endpoint.AddressFamily == AddressFamily.InterNetworkV6 ? $"[{endpoint.Address}]:{endpoint.Port}" : $"{endpoint.Address}:{endpoint.Port}";
}

/// <summary>
/// The calling side of a connection: requests get an id, a reader loop routes each reply to the
/// request that is waiting for it, and a dropped connection fails every waiter at once.
/// </summary>
internal sealed class RpcClient : IAsyncDisposable
{
    private readonly WireConnection _connection;
    private readonly ConcurrentDictionary<long, TaskCompletionSource<Message>> _pending = new();
    private readonly CancellationTokenSource _stop = new();
    private readonly TaskCompletionSource<Exception> _closed = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private long _nextId;

    private RpcClient(WireConnection connection, string address)
    {
        _connection = connection;
        Address = address;
        _ = ReadLoopAsync();
    }

    public string Address { get; }

    /// <summary>Completes, with the cause, when the connection drops.</summary>
    public Task<Exception> Closed => _closed.Task;

    public bool IsClosed => _closed.Task.IsCompleted;

    public static async Task<RpcClient> ConnectAsync(string address, string token, CancellationToken cancellationToken = default)
        => new(await WireConnection.ConnectAsync(address, token, cancellationToken).ConfigureAwait(false), address);

    public async Task<Message> RequestAsync(MessageKind kind, Action<BinaryWriter>? body = null, CancellationToken cancellationToken = default)
    {
        var id = Interlocked.Increment(ref _nextId);
        var tcs = new TaskCompletionSource<Message>(TaskCreationOptions.RunContinuationsAsynchronously);
        _pending[id] = tcs;
        if (IsClosed) { _pending.TryRemove(id, out _); throw new WorkerLostException(Address, _closed.Task.Result); }
        try
        {
            await _connection.SendAsync(kind, id, body, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception e) when (e is IOException or SocketException or ObjectDisposedException)
        {
            _pending.TryRemove(id, out _);
            Fail(e);
            throw new WorkerLostException(Address, e);
        }
        using (cancellationToken.Register(() => tcs.TrySetCanceled(cancellationToken)))
            return await tcs.Task.ConfigureAwait(false);
    }

    private async Task ReadLoopAsync()
    {
        try
        {
            while (!_stop.IsCancellationRequested)
            {
                var message = await _connection.ReceiveAsync(_stop.Token).ConfigureAwait(false);
                if (_pending.TryRemove(message.Id, out var tcs)) tcs.TrySetResult(message);
            }
        }
        catch (Exception e)
        {
            Fail(e);
        }
    }

    private void Fail(Exception cause)
    {
        if (!_closed.TrySetResult(cause)) return;
        foreach (var (id, tcs) in _pending)
            if (_pending.TryRemove(id, out _)) tcs.TrySetException(new WorkerLostException(Address, cause));
    }

    public async ValueTask DisposeAsync()
    {
        _stop.Cancel();
        Fail(new ObjectDisposedException(nameof(RpcClient)));
        await _connection.DisposeAsync().ConfigureAwait(false);
        _stop.Dispose();
    }
}
