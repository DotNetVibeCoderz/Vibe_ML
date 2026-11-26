using System.Buffers;
using System.Buffers.Binary;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.DaskNet.Distributed;

/// <summary>
/// An array result read in place: when it came from a worker on the same machine through shared
/// memory, <see cref="Span"/> and <see cref="Memory"/> point straight into that worker's segment,
/// so nothing is copied into this process at all. Dispose it to hand the segment back.
/// </summary>
/// <remarks>
/// <para>
/// From another machine, below the worker's shared-memory threshold, or with
/// <see cref="Client.SharedMemory"/> off, the view is over an ordinary decoded copy and
/// <see cref="IsShared"/> is false; the API is the same either way.
/// </para>
/// <para>
/// The data is read-only, and valid only until <see cref="Dispose"/>: a span taken before then
/// must not outlive it (<see cref="Memory"/> throws once the view is disposed). A view held for
/// more than a minute stays valid - the worker gives up on the segment and drops it, but never
/// reuses it, and the mapping this view holds keeps its pages alive.
/// </para>
/// </remarks>
public sealed unsafe class SharedArrayView : IDisposable
{
    private readonly double* _data;
    private readonly double[]? _copy;
    private readonly Action? _release;
    private readonly Manager? _manager;
    private int _disposed;

    private SharedArrayView(int[] shape, double* data, int length, Action release)
    {
        Shape = shape;
        _data = data;
        Length = length;
        _release = release;
        _manager = new Manager(this);
    }

    private SharedArrayView(int[] shape, double[] copy)
    {
        Shape = shape;
        _copy = copy;
        Length = copy.Length;
    }

    /// <summary>The array's shape (one dimension for a <c>double[]</c> result).</summary>
    public IReadOnlyList<int> Shape { get; }

    /// <summary>Elements, in C (row-major) order.</summary>
    public int Length { get; }

    /// <summary>True when the data is the worker's shared-memory segment rather than a copy.</summary>
    public bool IsShared => _manager is not null;

    /// <summary>The elements, in place.</summary>
    public ReadOnlySpan<double> Span
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            return _copy is not null ? _copy : new ReadOnlySpan<double>(_data, Length);
        }
    }

    /// <summary>The elements as memory, for APIs that take it (it fails once the view is disposed).</summary>
    public ReadOnlyMemory<double> Memory
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            return _copy is not null ? _copy : _manager!.Memory;
        }
    }

    /// <summary>One element by flat index.</summary>
    public double this[int index] => Span[index];

    /// <summary>A managed copy, as an <see cref="NdArray"/> that outlives the view.</summary>
    public NdArray ToNdArray() => new(Span.ToArray(), [.. Shape]);

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _release?.Invoke();
    }

    /// <summary>A view over an encoded <c>NdArray</c> or <c>double[]</c> at the start of
    /// <paramref name="mapping"/>; null (and the mapping untouched) for anything else.</summary>
    internal static SharedArrayView? Over(SharedMemory.Mapping mapping, long length, Action release)
    {
        var p = mapping.Pointer;
        var bytes = new ReadOnlySpan<byte>(p, checked((int)Math.Min(length, int.MaxValue)));
        if (bytes.Length < 5) return null;
        int offset;
        int[] shape;
        if (bytes[0] == WireCodec.NdArrayTag)
        {
            var rank = BinaryPrimitives.ReadInt32LittleEndian(bytes[1..]);
            shape = new int[rank];
            for (var i = 0; i < rank; i++) shape[i] = BinaryPrimitives.ReadInt32LittleEndian(bytes[(5 + 4 * i)..]);
            offset = 5 + 4 * rank;
        }
        else if (bytes[0] == WireCodec.DoubleArrayTag)
        {
            shape = [];
            offset = 1;
        }
        else return null;
        var count = BinaryPrimitives.ReadInt32LittleEndian(bytes[offset..]);
        offset += 4;
        if (offset + 8L * count > length) throw new InvalidDataException("A shared array segment is shorter than its header says.");
        if (shape.Length == 0) shape = [count];
        // Unaligned in general (the header is 5 + 4 * rank bytes): x64 and Arm64 read doubles at
        // any address, and the span is only ever read.
        return new SharedArrayView(shape, (double*)(p + offset), count, release);
    }

    /// <summary>A view over a decoded value, which must be an array.</summary>
    internal static SharedArrayView Copy(object? value, TaskKey key) => value switch
    {
        NdArray n => new SharedArrayView([.. n.Shape], n.ToArray()),
        double[] a => new SharedArrayView([a.Length], a),
        _ => throw new InvalidCastException($"{key} is a {value?.GetType().Name ?? "null"}, not an NdArray or double[]; only arrays can be viewed."),
    };

    /// <summary>Lets <see cref="Memory"/> point at the segment.</summary>
    private sealed class Manager(SharedArrayView owner) : MemoryManager<double>
    {
        public override Span<double> GetSpan()
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref owner._disposed) != 0, owner);
            return new Span<double>(owner._data, owner.Length);
        }

        public override MemoryHandle Pin(int elementIndex = 0)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref owner._disposed) != 0, owner);
            return new MemoryHandle(owner._data + elementIndex);
        }

        public override void Unpin() { }

        protected override void Dispose(bool disposing) { }
    }
}
