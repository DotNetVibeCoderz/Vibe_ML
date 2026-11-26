using System.Collections.Concurrent;
using System.IO.MemoryMappedFiles;
using System.Numerics;
using System.Runtime.Versioning;

namespace Gravicode.Science.DaskNet.Distributed;

/// <summary>
/// Hands a large result to a process on the same machine through shared memory instead of the
/// socket: the holder encodes the value straight into a mapped segment and replies with its name,
/// and the reader decodes straight out of its own mapping of it. A transfer is then two memory
/// copies (into the segment, out into the new value) where loopback TCP made five.
/// </summary>
/// <remarks>
/// <para>
/// Mappings must stay open to pay off. Mapping a fresh view per transfer page-faults through the
/// whole segment on both sides, and measured slower than the socket (0.87x at 64 MiB); so the
/// holder keeps a small pool of segments with their views mapped and reuses them once a reader
/// says it is done, and a reader keeps the last few segments it opened mapped too.
/// </para>
/// <para>
/// On Windows a segment is a named, page-file-backed mapping; elsewhere it is a file in
/// <c>/dev/shm</c> (RAM-backed) or, where that does not exist, the temp folder. Co-location is
/// claimed by host id and proved by opening the segment: a reader that cannot open it asks again
/// over the socket and stops trying shared memory with that holder.
/// </para>
/// </remarks>
internal static class SharedMemory
{
    /// <summary>Identifies this machine (and, on Windows, this login session, since segment names
    /// are session-local).</summary>
    public static readonly string HostId = Environment.MachineName + "|" + Environment.UserName + "|"
        + (OperatingSystem.IsWindows() ? "win-local" : Directory.Exists("/dev/shm") ? "/dev/shm" : Path.GetTempPath());

    private static readonly string? FileRoot = OperatingSystem.IsWindows() ? null
        : Directory.Exists("/dev/shm") ? "/dev/shm" : Path.GetTempPath();

    /// <summary>A mapping with its view mapped for as long as it lives.</summary>
    internal sealed unsafe class Mapping : IDisposable
    {
        private readonly MemoryMappedFile _map;
        private readonly MemoryMappedViewAccessor _view;
        private readonly string? _path;
        private readonly byte* _pointer;
        private int _disposed;

        private Mapping(string locator, MemoryMappedFile map, long capacity, bool writable, string? path)
        {
            Locator = locator;
            _map = map;
            _path = path;
            _view = map.CreateViewAccessor(0, capacity, writable ? MemoryMappedFileAccess.ReadWrite : MemoryMappedFileAccess.Read);
            Capacity = capacity > 0 ? capacity : _view.Capacity;      // 0: the whole segment, as opened
            byte* p = null;
            _view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
            _pointer = p + _view.PointerOffset;
        }

        public string Locator { get; }
        public long Capacity { get; }

        /// <summary>A new segment of <paramref name="capacity"/> bytes, owned by the caller.</summary>
        public static Mapping Create(long capacity)
        {
            var name = "dasknet-" + Guid.NewGuid().ToString("N");
            if (OperatingSystem.IsWindows()) return CreateNamed(name, capacity);
            var path = Path.Combine(FileRoot!, name);
            var map = MemoryMappedFile.CreateFromFile(path, FileMode.CreateNew, null, capacity, MemoryMappedFileAccess.ReadWrite);
            return new Mapping("file:" + path, map, capacity, writable: true, path);
        }

        [SupportedOSPlatform("windows")]
        private static Mapping CreateNamed(string name, long capacity)
            => new("name:" + name, MemoryMappedFile.CreateNew(name, capacity, MemoryMappedFileAccess.ReadWrite), capacity, writable: true, null);

        /// <summary>Opens a segment another process created, read-only.</summary>
        public static Mapping Open(string locator)
        {
            if (locator.StartsWith("name:", StringComparison.Ordinal))
            {
                if (!OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("Named segments exist on Windows only.");
                return OpenNamed(locator);
            }
            if (!locator.StartsWith("file:", StringComparison.Ordinal)) throw new InvalidDataException($"Unknown segment '{locator}'.");
            var path = locator[5..];
            var map = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            return new Mapping(locator, map, 0, writable: false, null);
        }

        [SupportedOSPlatform("windows")]
        private static Mapping OpenNamed(string locator)
            => new(locator, MemoryMappedFile.OpenExisting(locator[5..], MemoryMappedFileRights.Read), 0, writable: false, null);

        public UnmanagedMemoryStream Stream(long length, bool write)
            => new(_pointer, length, length, write ? FileAccess.Write : FileAccess.Read);

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _view.Dispose();
            _map.Dispose();
            if (_path is not null)
                try { File.Delete(_path); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }

    /// <summary>Encodes <paramref name="value"/> (of measured <paramref name="size"/>) into
    /// <paramref name="mapping"/>.</summary>
    public static void Write(Mapping mapping, object? value, long size)
    {
        using var s = mapping.Stream(size, write: true);
        using var w = new BinaryWriter(s, System.Text.Encoding.UTF8);
        WireCodec.Write(w, value);
    }

    /// <summary>Copies already-encoded bytes (a spilled result) into <paramref name="mapping"/>.</summary>
    public static void Write(Mapping mapping, byte[] bytes)
    {
        using var s = mapping.Stream(bytes.LongLength, write: true);
        s.Write(bytes);
    }

    // ------------------------------------------------------------------ reader side

    private const int ReaderCacheSize = 4;
    private static readonly LinkedList<Mapping> ReaderCache = new();

    /// <summary>Decodes a value from a segment another process wrote, through a cached mapping.</summary>
    public static object? Read(string locator, long length)
    {
        Mapping? mapping;
        lock (ReaderCache)
        {
            mapping = ReaderCache.FirstOrDefault(m => m.Locator == locator && m.Capacity >= length);
            if (mapping is not null) ReaderCache.Remove(mapping);
        }
        mapping ??= Mapping.Open(locator);
        try
        {
            using var s = mapping.Stream(length, write: false);
            using var r = new BinaryReader(s, System.Text.Encoding.UTF8);
            return WireCodec.Read(r);
        }
        finally
        {
            // Back to the front of the cache; the oldest falls out. A cached mapping keeps a segment
            // the holder has dropped alive, which is why there are only a few.
            Mapping? evicted = null;
            lock (ReaderCache)
            {
                ReaderCache.AddFirst(mapping);
                if (ReaderCache.Count > ReaderCacheSize)
                {
                    evicted = ReaderCache.Last!.Value;
                    ReaderCache.RemoveLast();
                }
            }
            evicted?.Dispose();
        }
    }

    /// <summary>
    /// Fetches <paramref name="key"/> from <paramref name="holder"/>, through shared memory when
    /// <paramref name="useShared"/> and the holder agrees (same host, value at least its
    /// threshold). Returns false when the holder does not have the key.
    /// </summary>
    public static async Task<(bool Found, object? Value)> GetAsync(RpcClient holder, TaskKey key, bool useShared,
        Action? sharedFailed, CancellationToken cancellationToken)
    {
        if (useShared)
        {
            var reply = await holder.RequestAsync(MessageKind.GetShared, w => { WireCodec.WriteKey(w, key); w.Write(HostId); }, cancellationToken)
                .ConfigureAwait(false);
            if (reply.Kind != MessageKind.SharedData) return Decode(reply);
            string locator;
            long length;
            long segment;
            using (var r = reply.Reader())
            {
                segment = r.ReadInt64();
                locator = r.ReadString();
                length = r.ReadInt64();
            }
            object? value = null;
            var ok = false;
            try
            {
                value = Read(locator, length);
                ok = true;
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException or PlatformNotSupportedException)
            {
                sharedFailed?.Invoke();          // not the same machine after all, or no access
            }
            finally
            {
                await holder.RequestAsync(MessageKind.SharedDone, w => w.Write(segment), cancellationToken).ConfigureAwait(false);
            }
            if (ok) return (true, value);
        }
        return Decode(await holder.RequestAsync(MessageKind.Get, w => WireCodec.WriteKey(w, key), cancellationToken).ConfigureAwait(false));
    }

    private static (bool, object?) Decode(Message reply)
    {
        if (reply.Kind != MessageKind.Data) return (false, null);
        using var r = reply.Reader();
        r.ReadInt32();                                   // length; decode in place, no copy
        return (true, WireCodec.Read(r));
    }
}

/// <summary>
/// The holder's segments: lent to one reader at a time, back in the pool on the reader's word.
/// A lent segment whose reader never answers is dropped after a minute rather than reused, since
/// the reader may still be reading it.
/// </summary>
internal sealed class SharedSegments : IDisposable
{
    private static readonly TimeSpan Lifetime = TimeSpan.FromMinutes(1);
    private const long MinimumCapacity = 1 << 20;
    private readonly ConcurrentDictionary<long, (SharedMemory.Mapping Mapping, DateTime Lent)> _lent = new();
    private readonly List<SharedMemory.Mapping> _free = [];
    private readonly long _poolBytes;
    private long _next;
    private bool _disposed;

    /// <param name="poolBytes">Free segments kept mapped for reuse, at most this many bytes in total.</param>
    public SharedSegments(long poolBytes = 256L << 20) => _poolBytes = poolBytes;

    /// <summary>A segment of at least <paramref name="size"/> bytes, lent under the returned id.</summary>
    public (long Id, SharedMemory.Mapping Mapping) Lend(long size)
    {
        var now = DateTime.UtcNow;
        foreach (var (id, l) in _lent)
            if (now - l.Lent > Lifetime && _lent.TryRemove(id, out var stale)) stale.Mapping.Dispose();

        SharedMemory.Mapping? mapping = null;
        lock (_free)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            // The smallest free segment that fits, if it is not more than twice too big.
            var fit = _free.Where(m => m.Capacity >= size && m.Capacity <= 2 * Math.Max(size, MinimumCapacity)).MinBy(m => m.Capacity);
            if (fit is not null)
            {
                _free.Remove(fit);
                mapping = fit;
            }
        }
        mapping ??= SharedMemory.Mapping.Create(Math.Max(MinimumCapacity, (long)BitOperations.RoundUpToPowerOf2((ulong)size)));
        var key = Interlocked.Increment(ref _next);
        _lent[key] = (mapping, now);
        return (key, mapping);
    }

    /// <summary>The reader is done: the segment goes back to the pool, or is dropped if the pool is full.</summary>
    public void Release(long id)
    {
        if (!_lent.TryRemove(id, out var l)) return;
        lock (_free)
        {
            if (!_disposed && _free.Sum(m => m.Capacity) + l.Mapping.Capacity <= _poolBytes)
            {
                _free.Add(l.Mapping);
                return;
            }
        }
        l.Mapping.Dispose();
    }

    /// <summary>Segments lent and not yet released.</summary>
    public int Count => _lent.Count;

    public void Dispose()
    {
        lock (_free)
        {
            _disposed = true;
            foreach (var m in _free) m.Dispose();
            _free.Clear();
        }
        foreach (var id in _lent.Keys)
            if (_lent.TryRemove(id, out var l)) l.Mapping.Dispose();
    }
}
