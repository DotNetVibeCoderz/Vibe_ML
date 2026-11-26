using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.DaskNet.Distributed;

/// <summary>What a worker's result store holds now, and how much it has moved to disk.</summary>
/// <param name="Keys">Results held, in memory and on disk.</param>
/// <param name="MemoryBytes">Estimated size of the results in memory.</param>
/// <param name="SpilledKeys">Results on disk now.</param>
/// <param name="SpilledBytes">Bytes on disk now.</param>
/// <param name="Spills">Results written to disk so far.</param>
/// <param name="Unspills">Results read back from disk so far.</param>
public sealed record StoreStatistics(int Keys, long MemoryBytes, int SpilledKeys, long SpilledBytes, long Spills, long Unspills);

/// <summary>
/// A worker's results: in memory while they fit, spilled to disk, least recently used first, once
/// the estimated size passes <c>target × limit</c> (Dask's <c>distributed.worker.memory.target</c>).
/// </summary>
/// <remarks>
/// <para>
/// Sizes are estimates (<see cref="SizeOf"/>): 8 bytes a number, UTF-16 for text, a fixed
/// overhead for everything else. They are what the limit is measured in - not the process's
/// working set, which the garbage collector decides.
/// </para>
/// <para>
/// A spilled result is its wire encoding, so a peer asking for it is sent the file's bytes as
/// they are. Read back for a task, it returns to memory as the most recently used result - unless
/// it alone is larger than the target, in which case it is read through and stays on disk, rather
/// than evicting everything else only to be spilled again. A value the wire codec cannot encode
/// stays in memory whatever its size.
/// </para>
/// </remarks>
internal sealed class WorkerStore(long? memoryLimit, double target, string? spillDirectory, string name) : IDisposable
{
    private sealed class Entry
    {
        public object? Value;
        public long Size;
        public string? Path;          // set while spilled
        public long DiskBytes;
        public LinkedListNode<TaskKey>? Node;   // set while in memory
        public bool Unspillable;
    }

    private readonly object _gate = new();
    private readonly Dictionary<TaskKey, Entry> _entries = [];
    private readonly LinkedList<TaskKey> _lru = new();   // first = least recently used, in memory only
    private readonly long _budget = memoryLimit is { } l ? (long)(l * target) : long.MaxValue;
    private string? _directory;
    private long _memory, _disk, _spills, _unspills;
    private int _spilledKeys;

    public int Count { get { lock (_gate) return _entries.Count; } }

    public StoreStatistics Statistics()
    {
        lock (_gate) return new StoreStatistics(_entries.Count, _memory, _spilledKeys, _disk, _spills, _unspills);
    }

    public bool Contains(TaskKey key) { lock (_gate) return _entries.ContainsKey(key); }

    public void Set(TaskKey key, object? value)
    {
        lock (_gate)
        {
            RemoveLocked(key);
            var entry = new Entry { Value = value, Size = SizeOf.Estimate(value) };
            entry.Node = _lru.AddLast(key);
            _entries[key] = entry;
            _memory += entry.Size;
            Enforce(protect: null);
        }
    }

    public bool TryGet(TaskKey key, out object? value)
    {
        lock (_gate)
        {
            if (!_entries.TryGetValue(key, out var entry)) { value = null; return false; }
            if (entry.Path is null)
            {
                _lru.Remove(entry.Node!);
                _lru.AddLast(entry.Node!);
                value = entry.Value;
                return true;
            }

            value = WireCodec.Deserialize(File.ReadAllBytes(entry.Path));
            _unspills++;
            if (entry.Size > _budget) return true;   // read through; it stays on disk
            File.Delete(entry.Path);
            entry.Path = null;
            _disk -= entry.DiskBytes;
            _spilledKeys--;
            entry.Value = value;
            entry.Node = _lru.AddLast(key);
            _memory += entry.Size;
            Enforce(protect: key);
            return true;
        }
    }

    /// <summary>The wire encoding of a result, for a peer: a spilled one is read from disk as is.</summary>
    /// <summary>A result for sending: its encoded bytes if it is spilled (already encoded), or the
    /// value itself, so a caller can encode it straight into its destination.</summary>
    public bool TryGetForTransfer(TaskKey key, out object? value, out byte[]? spilled)
    {
        lock (_gate)
        {
            value = null;
            spilled = null;
            if (!_entries.TryGetValue(key, out var entry)) return false;
            if (entry.Path is not null) spilled = File.ReadAllBytes(entry.Path);
            else value = entry.Value;
            return true;
        }
    }

    public bool TryGetSerialized(TaskKey key, out byte[] bytes)
    {
        object? value;
        lock (_gate)
        {
            if (!_entries.TryGetValue(key, out var entry)) { bytes = []; return false; }
            if (entry.Path is not null)
            {
                bytes = File.ReadAllBytes(entry.Path);
                return true;
            }
            value = entry.Value;
        }
        bytes = WireCodec.Serialize(value);
        return true;
    }

    public bool Remove(TaskKey key)
    {
        lock (_gate) return RemoveLocked(key);
    }

    public void Clear()
    {
        lock (_gate)
        {
            foreach (var key in _entries.Keys.ToList()) RemoveLocked(key);
        }
    }

    private bool RemoveLocked(TaskKey key)
    {
        if (!_entries.Remove(key, out var entry)) return false;
        if (entry.Node is not null)
        {
            _lru.Remove(entry.Node);
            _memory -= entry.Size;
        }
        if (entry.Path is not null)
        {
            TryDelete(entry.Path);
            _disk -= entry.DiskBytes;
            _spilledKeys--;
        }
        return true;
    }

    /// <summary>Spills least recently used results until the estimate is under the budget.</summary>
    private void Enforce(TaskKey? protect)
    {
        var node = _lru.First;
        while (_memory > _budget && node is not null)
        {
            var next = node.Next;
            var key = node.Value;
            var entry = _entries[key];
            if (!entry.Unspillable && !(protect is { } p && p.Equals(key))) Spill(key, entry);
            node = next;
        }
    }

    private void Spill(TaskKey key, Entry entry)
    {
        byte[] bytes;
        try { bytes = WireCodec.Serialize(entry.Value); }
        catch (Exception)
        {
            entry.Unspillable = true;
            return;
        }
        _directory ??= Directory.CreateDirectory(Path.Combine(spillDirectory ?? Path.Combine(Path.GetTempPath(), "dasknet-spill"),
            $"{Sanitise(name)}-{Environment.ProcessId}-{Guid.NewGuid():N}")).FullName;
        var path = Path.Combine(_directory, $"{_spills}-{Sanitise(key.ToString())}.bin");
        File.WriteAllBytes(path, bytes);
        _lru.Remove(entry.Node!);
        entry.Node = null;
        entry.Value = null;
        entry.Path = path;
        entry.DiskBytes = bytes.Length;
        _memory -= entry.Size;
        _disk += bytes.Length;
        _spilledKeys++;
        _spills++;
    }

    private static string Sanitise(string s)
    {
        var chars = s.Select(c => char.IsLetterOrDigit(c) || c is '-' or '_' ? c : '_').ToArray();
        return new string(chars, 0, Math.Min(chars.Length, 60));
    }

    private static void TryDelete(string path)
    {
        try { File.Delete(path); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    /// <summary>Deletes every spilled file and the spill directory.</summary>
    public void Dispose()
    {
        lock (_gate)
        {
            Clear();
            if (_directory is not null)
            {
                try { Directory.Delete(_directory, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
                _directory = null;
            }
        }
    }
}

/// <summary>Estimated in-memory size of a task result, in bytes.</summary>
public static class SizeOf
{
    private const long Overhead = 32;

    /// <summary>8 bytes per number, two per character plus a header per string, recursing into
    /// frames and arrays; <see cref="Overhead"/> for anything else.</summary>
    public static long Estimate(object? value) => value switch
    {
        null => 8,
        string s => Overhead + 2L * s.Length,
        NdArray n => Overhead + 8L * n.Size,
        DataFrame f => Overhead + f.Columns.Sum(Series),
        double[] a => Overhead + 8L * a.Length,
        long[] a => Overhead + 8L * a.Length,
        int[] a => Overhead + 4L * a.Length,
        bool[] a => Overhead + a.Length,
        bool?[] a => Overhead + 2L * a.Length,
        byte[] a => Overhead + a.Length,
        string?[] a => Overhead + a.Sum(s => 8 + (s is null ? 0 : Overhead + 2L * s.Length)),
        double[][] j => Overhead + j.Sum(a => Overhead + 8L * a.Length),
        System.Collections.ICollection items => Overhead + Sampled(items),
        _ => Overhead,
    };

    /// <summary>A Bag partition can hold millions of items: measure the first thousand and scale.</summary>
    private static long Sampled(System.Collections.ICollection items)
    {
        const int sample = 1000;
        long total = 0;
        var seen = 0;
        foreach (var x in items)
        {
            if (seen == sample) break;
            total += 8 + Estimate(x);
            seen++;
        }
        return seen == 0 ? 0 : total * items.Count / seen;
    }

    private static long Series(Series s)
    {
        if (s is not TextSeries) return Overhead + 8L * s.Length;
        var total = Overhead;
        for (var i = 0; i < s.Length; i++)
            total += 8 + (s.GetValue(i) is string v ? Overhead + 2L * v.Length : 0);
        return total;
    }
}
