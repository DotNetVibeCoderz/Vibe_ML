using System.Numerics;

namespace Gravicode.Science.GraviNum.Generic;

/// <summary>
/// An n-dimensional array over any IEEE floating-point element type.
/// </summary>
/// <typeparam name="T">The element type — <see cref="float"/> or <see cref="double"/> in practice.</typeparam>
/// <remarks>
/// <para>
/// The generic counterpart to <see cref="NdArray"/>, which is fixed to <c>double</c>. Same design:
/// an instance is a <em>view</em> — a shared buffer plus a shape, strides and an offset — so
/// reshaping, transposing and slicing cost a header allocation and nothing else. Only
/// <see cref="Copy"/> and <see cref="AsContiguous"/> move data.
/// </para>
/// <para>
/// This exists because the single-precision prototype measured a real gain — roughly <b>2.1x</b> on
/// element-wise arithmetic and <b>1.8x</b> on the matrix product — from twice the SIMD lanes and
/// half the bytes per value. Generic math (<c>IFloatingPointIeee754&lt;T&gt;</c>) is what makes one
/// implementation serve both widths, rather than the second numeric stack that copying the whole
/// file would have created.
/// </para>
/// <para>
/// <b>The six libraries still speak <see cref="NdArray"/>.</b> Migrating them so that
/// <c>NdArray</c> becomes <c>NdArray&lt;double&gt;</c> is a separate change and a large one; what
/// is here is the core it needs, proven against the double path. Convert at the boundary with
/// <see cref="NdArrayConvert"/>.
/// </para>
/// </remarks>
public sealed class NdArray<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly T[] _buffer;
    private readonly int[] _shape;
    private readonly int[] _strides;
    private readonly int _offset;

    internal NdArray(T[] buffer, int[] shape, int[] strides, int offset)
    {
        _buffer = buffer;
        _shape = shape;
        _strides = strides;
        _offset = offset;
        Size = Shapes.Size(shape);
    }

    /// <summary>Wraps <paramref name="data"/> without copying.</summary>
    public NdArray(T[] data, params int[] shape)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (shape.Length == 0) shape = [data.Length];

        var size = Shapes.Size(shape);
        if (size != data.Length)
            throw new ArgumentException(
                $"Shape ({Shapes.Describe(shape)}) needs {size} elements but {data.Length} were given.");

        _buffer = data;
        _shape = (int[])shape.Clone();
        _strides = Shapes.ContiguousStrides(_shape);
        _offset = 0;
        Size = size;
    }

    /// <summary>Number of axes.</summary>
    public int Rank => _shape.Length;

    /// <summary>Total number of elements.</summary>
    public int Size { get; }

    /// <summary>Length of each axis.</summary>
    public ReadOnlySpan<int> Shape => _shape;

    /// <summary>Stride of each axis, in elements.</summary>
    public ReadOnlySpan<int> Strides => _strides;

    /// <summary>The shared backing store.</summary>
    internal T[] Buffer => _buffer;

    /// <summary>Offset of element zero inside the buffer.</summary>
    internal int Offset => _offset;

    /// <summary>True when elements sit back to back in row-major order from the offset.</summary>
    public bool IsContiguous
    {
        get
        {
            var expected = 1;
            for (var axis = _shape.Length - 1; axis >= 0; axis--)
            {
                if (_shape[axis] == 1) continue;
                if (_strides[axis] != expected) return false;
                expected *= _shape[axis];
            }
            return true;
        }
    }

    // ---------------------------------------------------------------- construction

    /// <summary>An array of zeros.</summary>
    public static NdArray<T> Zeros(params int[] shape) => new(new T[Shapes.Size(shape)], shape);

    /// <summary>An array of ones.</summary>
    public static NdArray<T> Ones(params int[] shape) => Full(T.One, shape);

    /// <summary>An array filled with <paramref name="value"/>.</summary>
    public static NdArray<T> Full(T value, params int[] shape)
    {
        var data = new T[Shapes.Size(shape)];
        Array.Fill(data, value);
        return new NdArray<T>(data, shape);
    }

    /// <summary>The <paramref name="n"/>-square identity matrix.</summary>
    public static NdArray<T> Eye(int n)
    {
        var a = Zeros(n, n);
        for (var i = 0; i < n; i++) a._buffer[i * n + i] = T.One;
        return a;
    }

    /// <summary>A one-dimensional array of the given values.</summary>
    public static NdArray<T> FromValues(IEnumerable<T> values)
    {
        var data = values.ToArray();
        return new NdArray<T>(data, data.Length);
    }

    // ---------------------------------------------------------------- element access

    /// <summary>The element at a flat index, walking strides when the view is not contiguous.</summary>
    public T At(int flatIndex)
    {
        if (IsContiguous) return _buffer[_offset + flatIndex];

        var remaining = flatIndex;
        var position = _offset;
        for (var axis = _shape.Length - 1; axis >= 0; axis--)
        {
            position += remaining % _shape[axis] * _strides[axis];
            remaining /= _shape[axis];
        }
        return _buffer[position];
    }

    /// <summary>Sets the element at a flat index.</summary>
    public void SetAt(int flatIndex, T value)
    {
        if (IsContiguous) { _buffer[_offset + flatIndex] = value; return; }

        var remaining = flatIndex;
        var position = _offset;
        for (var axis = _shape.Length - 1; axis >= 0; axis--)
        {
            position += remaining % _shape[axis] * _strides[axis];
            remaining /= _shape[axis];
        }
        _buffer[position] = value;
    }

    /// <summary>Indexer for a rank-2 array.</summary>
    public T this[int row, int column]
    {
        get => _buffer[_offset + row * _strides[0] + column * _strides[1]];
        set => _buffer[_offset + row * _strides[0] + column * _strides[1]] = value;
    }

    // ---------------------------------------------------------------- views

    /// <summary>A view with a different shape; the data is not moved.</summary>
    public NdArray<T> Reshape(params int[] shape)
    {
        var resolved = Shapes.ResolveReshape(shape, Size);
        if (!IsContiguous) return AsContiguous().Reshape(resolved);
        return new NdArray<T>(_buffer, resolved, Shapes.ContiguousStrides(resolved), _offset);
    }

    /// <summary>A view with the axes reversed, or permuted by <paramref name="axes"/>.</summary>
    public NdArray<T> Transpose(params int[] axes)
    {
        if (axes.Length == 0)
        {
            axes = new int[Rank];
            for (var i = 0; i < Rank; i++) axes[i] = Rank - 1 - i;
        }

        var shape = new int[Rank];
        var strides = new int[Rank];
        for (var i = 0; i < Rank; i++)
        {
            shape[i] = _shape[axes[i]];
            strides[i] = _strides[axes[i]];
        }
        return new NdArray<T>(_buffer, shape, strides, _offset);
    }

    /// <summary>The transpose of a matrix.</summary>
    public NdArray<T> T2 => Transpose();

    /// <summary>A deep copy with fresh, contiguous storage.</summary>
    public NdArray<T> Copy() => new(ToArray(), _shape);

    /// <summary>This array when it is already contiguous, otherwise a compacted copy.</summary>
    public NdArray<T> AsContiguous() => IsContiguous ? this : Copy();

    /// <summary>Every element in row-major order, as a new array.</summary>
    public T[] ToArray()
    {
        var result = new T[Size];
        if (IsContiguous)
        {
            Array.Copy(_buffer, _offset, result, 0, Size);
            return result;
        }

        for (var i = 0; i < Size; i++) result[i] = At(i);
        return result;
    }

    /// <summary>The contiguous span backing this array.</summary>
    public Span<T> AsSpan() => IsContiguous
        ? _buffer.AsSpan(_offset, Size)
        : throw new InvalidOperationException("AsSpan requires a contiguous array; call AsContiguous() first.");

    /// <inheritdoc />
    public override string ToString() => $"NdArray<{typeof(T).Name}>({Shapes.Describe(_shape)})";
}
