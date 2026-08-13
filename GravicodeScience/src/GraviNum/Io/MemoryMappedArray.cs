using System.IO.MemoryMappedFiles;

namespace Gravicode.Science.GraviNum.Io;

/// <summary>
/// A <see cref="double"/> array backed directly by a file on disk.
/// </summary>
/// <remarks>
/// The point is datasets larger than RAM: the operating system pages in only the regions actually
/// touched, so a 40 GB matrix can be streamed through a machine with 16 GB of memory and writes
/// go straight back to the file. Random access costs a page fault rather than a parse, which is
/// why the CSV-vs-memory-mapping benchmark in <c>benchmarks/GraviFrame.Benchmark</c> uses this type.
/// </remarks>
public sealed class MemoryMappedArray : IDisposable
{
    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _accessor;
    private bool _disposed;

    private MemoryMappedArray(MemoryMappedFile file, MemoryMappedViewAccessor accessor, long length, int[] shape, string path)
    {
        _file = file;
        _accessor = accessor;
        Length = length;
        Shape = shape;
        Path = path;
    }

    /// <summary>Number of elements.</summary>
    public long Length { get; }

    /// <summary>Logical shape of the mapped data.</summary>
    public int[] Shape { get; }

    /// <summary>Path of the backing file.</summary>
    public string Path { get; }

    /// <summary>Size of the backing file in bytes, excluding the header.</summary>
    public long ByteLength => Length * sizeof(double);

    private const int HeaderBytes = 64;
    private const uint Magic = 0x314D4D47; // "GMM1"

    /// <summary>
    /// Creates (or truncates) a file large enough for <paramref name="shape"/> and maps it.
    /// </summary>
    public static MemoryMappedArray Create(string path, params int[] shape)
    {
        var length = (long)Shapes.Size(shape);
        var total = HeaderBytes + length * sizeof(double);

        using (var stream = new FileStream(path, FileMode.Create, FileAccess.ReadWrite, FileShare.None))
        {
            stream.SetLength(total);
            using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
            writer.Write(Magic);
            writer.Write(shape.Length);
            foreach (var d in shape) writer.Write(d);
        }

        return Open(path, writable: true);
    }

    /// <summary>Maps an existing file created by <see cref="Create"/>.</summary>
    public static MemoryMappedArray Open(string path, bool writable = false)
    {
        int[] shape;
        using (var stream = File.OpenRead(path))
        using (var reader = new BinaryReader(stream))
        {
            if (reader.ReadUInt32() != Magic)
                throw new InvalidDataException($"'{path}' is not a memory-mapped Gravicode array.");
            var rank = reader.ReadInt32();
            shape = new int[rank];
            for (var i = 0; i < rank; i++) shape[i] = reader.ReadInt32();
        }

        var length = (long)Shapes.Size(shape);
        var access = writable ? MemoryMappedFileAccess.ReadWrite : MemoryMappedFileAccess.Read;
        var file = MemoryMappedFile.CreateFromFile(path, writable ? FileMode.Open : FileMode.Open,
            mapName: null, capacity: 0, access);
        var accessor = file.CreateViewAccessor(HeaderBytes, length * sizeof(double), access);
        return new MemoryMappedArray(file, accessor, length, shape, path);
    }

    /// <summary>Reads or writes a single element without loading the rest of the file.</summary>
    public double this[long index]
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _accessor.ReadDouble(index * sizeof(double));
        }
        set
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _accessor.Write(index * sizeof(double), value);
        }
    }

    /// <summary>Reads an element of a 2-D mapped array.</summary>
    public double Get(int row, int column) => this[(long)row * Shape[^1] + column];

    /// <summary>Writes an element of a 2-D mapped array.</summary>
    public void Set(int row, int column, double value) => this[(long)row * Shape[^1] + column] = value;

    /// <summary>Copies a contiguous run of elements into managed memory.</summary>
    public double[] ReadRange(long start, int count)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var buffer = new double[count];
        _accessor.ReadArray(start * sizeof(double), buffer, 0, count);
        return buffer;
    }

    /// <summary>Writes a contiguous run of elements.</summary>
    public void WriteRange(long start, double[] values)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _accessor.WriteArray(start * sizeof(double), values, 0, values.Length);
    }

    /// <summary>Copies one row of a 2-D mapped array into an <see cref="NdArray"/>.</summary>
    public NdArray ReadRow(int row)
    {
        var columns = Shape[^1];
        return new NdArray(ReadRange((long)row * columns, columns), columns);
    }

    /// <summary>
    /// Materialises the whole mapping as an in-memory array. Only sensible when the data fits;
    /// the point of this type is usually to avoid exactly this call.
    /// </summary>
    public NdArray ToNdArray()
    {
        if (Length > int.MaxValue)
            throw new InvalidOperationException($"{Length} elements exceed the in-memory array limit.");
        return new NdArray(ReadRange(0, (int)Length), Shape);
    }

    /// <summary>Streams the mapping in fixed-size chunks, which is how large scans should be written.</summary>
    public IEnumerable<double[]> Chunks(int chunkSize = 1 << 16)
    {
        for (long start = 0; start < Length; start += chunkSize)
        {
            var count = (int)Math.Min(chunkSize, Length - start);
            yield return ReadRange(start, count);
        }
    }

    /// <summary>Writes an in-memory array to a new memory-mapped file.</summary>
    public static MemoryMappedArray Persist(NdArray array, string path)
    {
        var mapped = Create(path, array.Shape.ToArray());
        mapped.WriteRange(0, array.ToArray());
        mapped.Flush();
        return mapped;
    }

    /// <summary>Flushes pending writes to disk.</summary>
    public void Flush() => _accessor.Flush();

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _accessor.Flush();
        _accessor.Dispose();
        _file.Dispose();
    }
}
