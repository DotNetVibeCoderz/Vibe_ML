using System.Buffers.Binary;
using System.Runtime.InteropServices;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.IO;

/// <summary>Thrown when a Fortran file ends cleanly between records (SciPy's <c>FortranEOFError</c>).</summary>
public sealed class FortranEofException : EndOfStreamException
{
    /// <summary>Creates the exception.</summary>
    public FortranEofException(string message) : base(message) { }
}

/// <summary>Thrown when a Fortran record is malformed or truncated (SciPy's <c>FortranFormattingError</c>).</summary>
public sealed class FortranFormattingException : IOException
{
    /// <summary>Creates the exception.</summary>
    public FortranFormattingException(string message) : base(message) { }
}

/// <summary>
/// The payload of one record, read item by item — what SciPy's <c>read_record(dtype1, dtype2, ...)</c>
/// does for mixed records (e.g. <c>write(1) a, b</c> with a real array and an integer array).
/// </summary>
public sealed class FortranRecord
{
    private readonly byte[] _data;
    private readonly bool _bigEndian;
    private int _pos;

    internal FortranRecord(byte[] data, bool bigEndian)
    {
        _data = data;
        _bigEndian = bigEndian;
    }

    /// <summary>Record length in bytes.</summary>
    public int Length => _data.Length;

    /// <summary>Bytes not yet consumed.</summary>
    public int Remaining => _data.Length - _pos;

    /// <summary>The raw bytes.</summary>
    public ReadOnlySpan<byte> Bytes => _data;

    /// <summary>Reads <paramref name="count"/> values (all remaining when null) of an unmanaged primitive type.</summary>
    public T[] Read<T>(int? count = null) where T : unmanaged
    {
        var size = Marshal.SizeOf<T>();
        var n = count ?? Remaining / size;
        if (count is null && Remaining % size != 0)
            throw new FortranFormattingException($"Record remainder of {Remaining} bytes is not a multiple of {typeof(T).Name} ({size} bytes).");
        if (n < 0 || (long)n * size > Remaining)
            throw new FortranFormattingException($"Record has {Remaining} bytes left, {n} x {typeof(T).Name} needs {(long)n * size}.");
        var r = MemoryMarshal.Cast<byte, T>(_data.AsSpan(_pos, n * size)).ToArray();
        if (_bigEndian) FortranFile.Swap(r);
        _pos += n * size;
        return r;
    }

    /// <summary>
    /// Reads a Fortran array of the given (column-major) shape as a row-major <see cref="NdArray"/>
    /// with the same shape: Fortran's <c>a(i, j)</c> is <c>result[i-1, j-1]</c>.
    /// </summary>
    public NdArray ReadArray<T>(params int[] fortranShape) where T : unmanaged
    {
        var n = MatLayout.Product(fortranShape);
        var raw = Read<T>(n);
        var d = new double[n];
        for (var i = 0; i < n; i++) d[i] = Convert.ToDouble(raw[i], System.Globalization.CultureInfo.InvariantCulture);
        var shape = fortranShape.Length == 0 ? [n] : fortranShape;
        return new NdArray(MatLayout.ColumnToRowMajor(d, shape), shape);
    }
}

/// <summary>
/// Unformatted sequential Fortran files — <c>scipy.io.FortranFile</c>. Each record is framed by
/// its byte length before and after (4-byte markers by default, as gfortran and ifort write;
/// 8 bytes for some older 64-bit compilers).
/// </summary>
/// <remarks>
/// Records longer than 2 GiB that gfortran splits into sub-records (negative markers) are not
/// supported. Data are little-endian unless <c>bigEndian</c> is set (for files written with
/// <c>-fconvert=big-endian</c> or on big-endian machines).
/// </remarks>
public sealed class FortranFile : IDisposable
{
    private readonly Stream _s;
    private readonly bool _leaveOpen;
    private readonly int _header;
    private readonly bool _big;
    private readonly byte[] _marker = new byte[8];

    /// <summary>Opens a file for reading (<paramref name="mode"/> 'r') or creates one for writing ('w').</summary>
    /// <param name="path">File path.</param>
    /// <param name="mode">'r' or 'w'.</param>
    /// <param name="headerSize">Record marker size in bytes: 4 (default) or 8.</param>
    /// <param name="bigEndian">Markers and data are big-endian.</param>
    public FortranFile(string path, char mode = 'r', int headerSize = 4, bool bigEndian = false)
        : this(mode switch
        {
            'r' => new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16),
            'w' => new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16),
            _ => throw new ArgumentException("mode must be 'r' or 'w'.", nameof(mode)),
        }, headerSize, bigEndian, leaveOpen: false)
    { }

    /// <summary>Wraps a stream.</summary>
    /// <param name="stream">Readable and/or writable stream.</param>
    /// <param name="headerSize">Record marker size in bytes: 4 or 8.</param>
    /// <param name="bigEndian">Markers and data are big-endian.</param>
    /// <param name="leaveOpen">Leave the stream open on dispose.</param>
    public FortranFile(Stream stream, int headerSize = 4, bool bigEndian = false, bool leaveOpen = false)
    {
        if (headerSize is not (4 or 8)) throw new ArgumentException("headerSize must be 4 or 8.", nameof(headerSize));
        _s = stream;
        _header = headerSize;
        _big = bigEndian;
        _leaveOpen = leaveOpen;
    }

    /// <summary>True when a seekable stream is positioned at its end.</summary>
    public bool EndOfFile => _s.CanSeek && _s.Position >= _s.Length;

    private long ReadMarker(bool eofOk)
    {
        var n = _s.ReadAtLeast(_marker.AsSpan(0, _header), _header, throwOnEndOfStream: false);
        if (n == 0 && eofOk) throw new FortranEofException("End of file occurred at end of record.");
        if (n < _header) throw new FortranFormattingException("End of file in the middle of the record size.");
        var v = _header == 4
            ? (long)(_big ? BinaryPrimitives.ReadUInt32BigEndian(_marker) : BinaryPrimitives.ReadUInt32LittleEndian(_marker))
            : (_big ? BinaryPrimitives.ReadInt64BigEndian(_marker) : BinaryPrimitives.ReadInt64LittleEndian(_marker));
        if (v < 0) throw new FortranFormattingException("Negative record size (gfortran sub-records for > 2 GiB records are not supported).");
        return v;
    }

    /// <summary>Reads the next record's payload for item-by-item reading (mixed-type records).</summary>
    /// <exception cref="FortranEofException">The file ended cleanly before the record.</exception>
    public FortranRecord ReadRecord()
    {
        var size = ReadMarker(eofOk: true);
        if (_s.CanSeek && size > _s.Length - _s.Position)
            throw new FortranFormattingException($"Record claims {size} bytes but only {_s.Length - _s.Position} remain — wrong header size or byte order?");
        if (size > int.MaxValue) throw new FortranFormattingException($"Record of {size} bytes exceeds 2 GiB.");
        var data = new byte[size];
        var got = _s.ReadAtLeast(data, data.Length, throwOnEndOfStream: false);
        if (got < size) throw new FortranFormattingException("End of file in the middle of a record.");
        var footer = ReadMarker(eofOk: false);
        if (footer != size) throw new FortranFormattingException($"Sizes do not agree in the header ({size}) and footer ({footer}) for this record - check header size.");
        return new FortranRecord(data, _big);
    }

    /// <summary>Reads a whole record as an array of one primitive type (<c>read_record(dtype)</c>).</summary>
    public T[] ReadRecord<T>() where T : unmanaged
    {
        var r = ReadRecord();
        if (r.Length % Marshal.SizeOf<T>() != 0)
            throw new FortranFormattingException($"Size obtained ({r.Length}) is not a multiple of {typeof(T).Name} ({Marshal.SizeOf<T>()} bytes).");
        return r.Read<T>();
    }

    /// <summary>Reads a record of 32-bit integers (<c>read_ints()</c>).</summary>
    public int[] ReadInts() => ReadRecord<int>();

    /// <summary>Reads a record of 64-bit reals (<c>read_reals()</c>).</summary>
    public double[] ReadReals() => ReadRecord<double>();

    /// <summary>Writes one record holding all <paramref name="items"/> back to back (<c>write_record(*items)</c>).</summary>
    /// <param name="items">Arrays of primitive types (any rank, written in memory order) or <see cref="NdArray"/> (written as float64, row-major).</param>
    public void WriteRecord(params object[] items)
    {
        var parts = new List<byte[]>();
        foreach (var item in items)
        {
            switch (item)
            {
                case NdArray nd:
                    parts.Add(Bytes(nd.AsContiguous().AsSpan().ToArray()));
                    break;
                case Array a when a.GetType().GetElementType() is { IsPrimitive: true } && a.GetType().GetElementType() != typeof(bool) && a.GetType().GetElementType() != typeof(char):
                {
                    var b = new byte[Buffer.ByteLength(a)];
                    Buffer.BlockCopy(a, 0, b, 0, b.Length);
                    if (_big) SwapBytes(b, Marshal.SizeOf(a.GetType().GetElementType()!));
                    parts.Add(b);
                    break;
                }
                default:
                    throw new ArgumentException($"Cannot write a {item?.GetType().Name ?? "null"} to a Fortran record; use arrays of numeric primitives or NdArray.");
            }
        }
        long total = parts.Sum(p => (long)p.Length);
        if (_header == 4 && total > uint.MaxValue) throw new ArgumentException("Record exceeds 4 GiB; use headerSize 8.");
        WriteMarker(total);
        foreach (var p in parts) _s.Write(p);
        WriteMarker(total);
    }

    /// <summary>Writes an <see cref="NdArray"/> as one float64 record; with <paramref name="fortranOrder"/> its elements go out column-major, so that Fortran reading <c>a(m, n)</c> sees <c>a(i, j) = array[i-1, j-1]</c>.</summary>
    public void WriteRecord(NdArray array, bool fortranOrder)
    {
        var v = fortranOrder ? MatLayout.RowToColumnMajor(array.AsContiguous().AsSpan(), array.Shape.ToArray()) : array.AsContiguous().AsSpan().ToArray();
        WriteRecord((object)v);
    }

    private byte[] Bytes(double[] v)
    {
        var b = MemoryMarshal.AsBytes(v.AsSpan()).ToArray();
        if (_big) SwapBytes(b, 8);
        return b;
    }

    private void WriteMarker(long n)
    {
        if (_header == 4)
        {
            if (_big) BinaryPrimitives.WriteUInt32BigEndian(_marker, (uint)n);
            else BinaryPrimitives.WriteUInt32LittleEndian(_marker, (uint)n);
        }
        else if (_big) BinaryPrimitives.WriteInt64BigEndian(_marker, n);
        else BinaryPrimitives.WriteInt64LittleEndian(_marker, n);
        _s.Write(_marker, 0, _header);
    }

    private static void SwapBytes(byte[] b, int size)
    {
        if (size == 1) return;
        for (var i = 0; i + size <= b.Length; i += size) b.AsSpan(i, size).Reverse();
    }

    internal static void Swap<T>(T[] values) where T : unmanaged
    {
        var bytes = MemoryMarshal.AsBytes(values.AsSpan());
        var size = Marshal.SizeOf<T>();
        if (size == 1) return;
        // Complex and other compound types swap each 8-byte component.
        var unit = typeof(T) == typeof(System.Numerics.Complex) ? 8 : size;
        for (var i = 0; i + unit <= bytes.Length; i += unit) bytes.Slice(i, unit).Reverse();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_leaveOpen) _s.Dispose();
        else _s.Flush();
    }
}
