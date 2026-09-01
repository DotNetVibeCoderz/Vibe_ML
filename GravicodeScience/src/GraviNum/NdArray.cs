using System.Runtime.CompilerServices;
using System.Text;

// The Slice method below would otherwise hide the Slice type inside this class body.
using Sel = Gravicode.Science.GraviNum.Slice;

namespace Gravicode.Science.GraviNum;

/// <summary>
/// The N-dimensional, double precision array at the heart of Gravicode.Science.
/// </summary>
/// <remarks>
/// An <see cref="NdArray"/> is a view: a shared <see cref="double"/> buffer plus a shape, a set of
/// strides and an offset. Reshaping, transposing and slicing therefore cost nothing but a small
/// header allocation, and only <see cref="Copy"/> / <see cref="AsContiguous"/> ever move data.
/// Element-wise work runs through <see cref="UFunc"/>, which takes a SIMD fast path whenever the
/// operands are contiguous.
/// </remarks>
public sealed partial class NdArray
{
    private readonly double[] _buffer;
    private readonly int[] _shape;
    private readonly int[] _strides;
    private readonly int _offset;

    internal NdArray(double[] buffer, int[] shape, int[] strides, int offset)
    {
        _buffer = buffer;
        _shape = shape;
        _strides = strides;
        _offset = offset;
        Size = Shapes.Size(shape);
    }

    /// <summary>Wraps <paramref name="data"/> without copying; the array takes the given shape.</summary>
    public NdArray(double[] data, params int[] shape)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (shape.Length == 0) shape = [data.Length];
        var size = Shapes.Size(shape);
        if (size != data.Length)
            throw new ArgumentException($"Shape ({Shapes.Describe(shape)}) needs {size} elements but {data.Length} were given.");
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

    /// <summary>Stride of each axis, measured in elements.</summary>
    public ReadOnlySpan<int> Strides => _strides;

    /// <summary>Offset of element zero inside the shared buffer.</summary>
    internal int Offset => _offset;

    /// <summary>The shared backing store. Mutating it mutates every view over it.</summary>
    internal double[] Buffer => _buffer;

    /// <summary>Rows of a 2-D array; the length of axis 0 otherwise.</summary>
    public int Rows => _shape.Length > 0 ? _shape[0] : 1;

    /// <summary>Columns of a 2-D array.</summary>
    public int Columns => _shape.Length > 1 ? _shape[1] : 1;

    /// <summary>
    /// True when the elements sit back-to-back in row-major order starting at the offset,
    /// which is what unlocks the vectorised kernels.
    /// </summary>
    public bool IsContiguous
    {
        get
        {
            var expected = 1;
            for (var i = _shape.Length - 1; i >= 0; i--)
            {
                if (_shape[i] == 1) continue;
                if (_strides[i] != expected) return false;
                expected *= _shape[i];
            }
            return true;
        }
    }

    // ---------------------------------------------------------------- creation

    /// <summary>An array of zeros.</summary>
    public static NdArray Zeros(params int[] shape) => new(new double[Shapes.Size(shape)], shape);

    /// <summary>An array of ones.</summary>
    public static NdArray Ones(params int[] shape) => Full(1.0, shape);

    /// <summary>An array filled with <paramref name="value"/>.</summary>
    public static NdArray Full(double value, params int[] shape)
    {
        var data = new double[Shapes.Size(shape)];
        Array.Fill(data, value);
        return new NdArray(data, shape);
    }

    /// <summary>An uninitialised array; contents are unspecified.</summary>
    public static NdArray Empty(params int[] shape) => Zeros(shape);

    /// <summary>An array shaped like <paramref name="other"/>, filled with zeros.</summary>
    public static NdArray ZerosLike(NdArray other) => Zeros(other._shape);

    /// <summary>A 1-D array of evenly spaced values over <c>[start, stop)</c>.</summary>
    public static NdArray Arange(double start, double stop, double step = 1.0)
    {
        if (step == 0) throw new ArgumentException("Step cannot be zero.", nameof(step));
        var count = (int)Math.Ceiling((stop - start) / step);
        if (count < 0) count = 0;
        var data = new double[count];
        for (var i = 0; i < count; i++) data[i] = start + i * step;
        return new NdArray(data, count);
    }

    /// <summary>A 1-D array <c>[0, 1, ..., stop-1]</c>.</summary>
    public static NdArray Arange(int stop) => Arange(0, stop);

    /// <summary>A 1-D array of <paramref name="count"/> values evenly spaced over <c>[start, stop]</c>.</summary>
    public static NdArray Linspace(double start, double stop, int count, bool endpoint = true)
    {
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count));
        var data = new double[count];
        if (count == 1) { data[0] = start; return new NdArray(data, 1); }
        var divisor = endpoint ? count - 1 : count;
        for (var i = 0; i < count; i++) data[i] = start + (stop - start) * i / divisor;
        return new NdArray(data, count);
    }

    /// <summary>The <paramref name="n"/>×<paramref name="n"/> identity matrix.</summary>
    public static NdArray Eye(int n)
    {
        var a = Zeros(n, n);
        for (var i = 0; i < n; i++) a.Buffer[i * n + i] = 1.0;
        return a;
    }

    /// <summary>A square matrix with <paramref name="diagonal"/> on the main diagonal.</summary>
    public static NdArray Diag(IReadOnlyList<double> diagonal)
    {
        var n = diagonal.Count;
        var a = Zeros(n, n);
        for (var i = 0; i < n; i++) a.Buffer[i * n + i] = diagonal[i];
        return a;
    }

    /// <summary>A rank-0 style 1-element array holding <paramref name="value"/>.</summary>
    public static NdArray Scalar(double value) => new([value], 1);

    /// <summary>Builds a 1-D array from a sequence.</summary>
    public static NdArray FromValues(IEnumerable<double> values)
    {
        var data = values as double[] ?? values.ToArray();
        return new NdArray(data, data.Length);
    }

    /// <summary>Builds a 2-D array from a jagged array; every row must be the same length.</summary>
    public static NdArray FromRows(IReadOnlyList<double[]> rows)
    {
        if (rows.Count == 0) return Zeros(0, 0);
        var cols = rows[0].Length;
        var data = new double[rows.Count * cols];
        for (var i = 0; i < rows.Count; i++)
        {
            if (rows[i].Length != cols)
                throw new ArgumentException($"Row {i} has {rows[i].Length} values, expected {cols}.");
            Array.Copy(rows[i], 0, data, i * cols, cols);
        }
        return new NdArray(data, rows.Count, cols);
    }

    /// <summary>Builds a 2-D array from a rectangular array.</summary>
    public static NdArray FromArray(double[,] values)
    {
        var rows = values.GetLength(0);
        var cols = values.GetLength(1);
        var data = new double[rows * cols];
        var k = 0;
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                data[k++] = values[i, j];
        return new NdArray(data, rows, cols);
    }

    // ---------------------------------------------------------------- indexing

    /// <summary>Reads or writes a single element by full index.</summary>
    public double this[params int[] indices]
    {
        get => _buffer[LinearIndex(indices)];
        set => _buffer[LinearIndex(indices)] = value;
    }

    /// <summary>Reads or writes an element of a 2-D array.</summary>
    public double this[int i, int j]
    {
        get
        {
            if (Rank != 2) throw new InvalidOperationException($"Two indices given for a rank {Rank} array.");
            return _buffer[_offset + i * _strides[0] + j * _strides[1]];
        }
        set
        {
            if (Rank != 2) throw new InvalidOperationException($"Two indices given for a rank {Rank} array.");
            _buffer[_offset + i * _strides[0] + j * _strides[1]] = value;
        }
    }

    private int LinearIndex(ReadOnlySpan<int> indices)
    {
        if (indices.Length != _shape.Length)
            throw new ArgumentException($"Expected {_shape.Length} indices, got {indices.Length}.");
        var linear = _offset;
        for (var i = 0; i < indices.Length; i++)
        {
            var idx = indices[i] < 0 ? indices[i] + _shape[i] : indices[i];
            if (idx < 0 || idx >= _shape[i])
                throw new IndexOutOfRangeException($"Index {indices[i]} out of range for axis {i} of length {_shape[i]}.");
            linear += idx * _strides[i];
        }
        return linear;
    }

    /// <summary>Reads the <paramref name="flatIndex"/>-th element in row-major order.</summary>
    public double At(int flatIndex)
    {
        if (flatIndex < 0 || flatIndex >= Size) throw new IndexOutOfRangeException();
        return _buffer[FlatToLinear(flatIndex)];
    }

    /// <summary>Writes the <paramref name="flatIndex"/>-th element in row-major order.</summary>
    public void SetAt(int flatIndex, double value)
    {
        if (flatIndex < 0 || flatIndex >= Size) throw new IndexOutOfRangeException();
        _buffer[FlatToLinear(flatIndex)] = value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal int FlatToLinear(int flatIndex)
    {
        if (IsContiguous) return _offset + flatIndex;
        var linear = _offset;
        for (var axis = _shape.Length - 1; axis >= 0; axis--)
        {
            var dim = _shape[axis];
            if (dim == 0) continue;
            linear += (flatIndex % dim) * _strides[axis];
            flatIndex /= dim;
        }
        return linear;
    }

    /// <summary>A zero-copy view of sub-array <paramref name="index"/> along axis 0.</summary>
    public NdArray Row(int index)
    {
        if (Rank == 0) throw new InvalidOperationException("Cannot take a row of a rank 0 array.");
        var i = index < 0 ? index + _shape[0] : index;
        if (i < 0 || i >= _shape[0]) throw new IndexOutOfRangeException($"Row {index} out of range ({_shape[0]} rows).");
        if (Rank == 1) return new NdArray(_buffer, [1], [_strides[0]], _offset + i * _strides[0]);
        return new NdArray(_buffer, _shape[1..], _strides[1..], _offset + i * _strides[0]);
    }

    /// <summary>A zero-copy view of column <paramref name="index"/> of a 2-D array.</summary>
    public NdArray Column(int index)
    {
        if (Rank != 2) throw new InvalidOperationException("Column requires a rank 2 array.");
        var j = index < 0 ? index + _shape[1] : index;
        if (j < 0 || j >= _shape[1]) throw new IndexOutOfRangeException($"Column {index} out of range ({_shape[1]} columns).");
        return new NdArray(_buffer, [_shape[0]], [_strides[0]], _offset + j * _strides[1]);
    }

    /// <summary>
    /// A zero-copy strided view. Index selectors drop their axis, range selectors keep it,
    /// and omitted trailing axes are taken whole.
    /// </summary>
    public NdArray Slice(params Sel[] slices)
    {
        if (slices.Length > Rank)
            throw new ArgumentException($"Got {slices.Length} selectors for a rank {Rank} array.");

        var shape = new List<int>();
        var strides = new List<int>();
        var offset = _offset;

        for (var axis = 0; axis < Rank; axis++)
        {
            if (axis >= slices.Length)
            {
                shape.Add(_shape[axis]);
                strides.Add(_strides[axis]);
                continue;
            }

            var (start, count, step) = slices[axis].Resolve(_shape[axis]);
            offset += start * _strides[axis];
            if (slices[axis].IsIndex) continue;
            shape.Add(count);
            strides.Add(_strides[axis] * step);
        }

        return new NdArray(_buffer, shape.ToArray(), strides.ToArray(), offset);
    }

    /// <summary>Selects rows (axis 0 entries) by index, copying them into a new array.</summary>
    public NdArray Take(IReadOnlyList<int> indices)
    {
        if (Rank == 0) throw new InvalidOperationException("Cannot take from a rank 0 array.");
        var shape = _shape.ToArray();
        shape[0] = indices.Count;
        var result = Zeros(shape);
        var rowSize = Size / Math.Max(_shape[0], 1);
        for (var i = 0; i < indices.Count; i++)
        {
            var src = Row(indices[i]);
            var dst = result.Row(i);
            for (var k = 0; k < rowSize; k++) dst.SetAt(k, src.At(k));
        }
        return result;
    }

    /// <summary>Selects elements where <paramref name="mask"/> is true, as a 1-D array.</summary>
    public NdArray BooleanMask(bool[] mask)
    {
        if (mask.Length != Size)
            throw new ArgumentException($"Mask has {mask.Length} entries but the array has {Size} elements.");
        var kept = new List<double>();
        for (var i = 0; i < Size; i++) if (mask[i]) kept.Add(At(i));
        return FromValues(kept);
    }

    /// <summary>Elements satisfying <paramref name="predicate"/>, as a 1-D array.</summary>
    public NdArray Where(Func<double, bool> predicate)
    {
        var kept = new List<double>();
        for (var i = 0; i < Size; i++)
        {
            var v = At(i);
            if (predicate(v)) kept.Add(v);
        }
        return FromValues(kept);
    }

    /// <summary>Element-wise mask over the flattened array.</summary>
    public bool[] Mask(Func<double, bool> predicate)
    {
        var mask = new bool[Size];
        for (var i = 0; i < Size; i++) mask[i] = predicate(At(i));
        return mask;
    }

    // ---------------------------------------------------------------- reshaping

    /// <summary>
    /// A view with a new shape. One dimension may be <c>-1</c> and is inferred.
    /// Non-contiguous arrays are compacted first.
    /// </summary>
    public NdArray Reshape(params int[] shape)
    {
        var resolved = Shapes.ResolveReshape(shape, Size);
        var source = IsContiguous ? this : Copy();
        return new NdArray(source._buffer, resolved, Shapes.ContiguousStrides(resolved), source._offset);
    }

    /// <summary>A 1-D view (or copy, when strided) of every element.</summary>
    public NdArray Ravel() => Reshape(Size);

    /// <summary>A 1-D copy of every element.</summary>
    public NdArray Flatten() => new(ToArray(), Size);

    /// <summary>A view with the axes permuted; with no arguments the axis order is reversed.</summary>
    public NdArray Transpose(params int[] axes)
    {
        if (axes.Length == 0)
        {
            axes = new int[Rank];
            for (var i = 0; i < Rank; i++) axes[i] = Rank - 1 - i;
        }
        if (axes.Length != Rank) throw new ArgumentException($"Expected {Rank} axes, got {axes.Length}.");

        var seen = new bool[Rank];
        var shape = new int[Rank];
        var strides = new int[Rank];
        for (var i = 0; i < Rank; i++)
        {
            var a = Shapes.NormalizeAxis(axes[i], Rank);
            if (seen[a]) throw new ArgumentException($"Axis {a} listed twice.");
            seen[a] = true;
            shape[i] = _shape[a];
            strides[i] = _strides[a];
        }
        return new NdArray(_buffer, shape, strides, _offset);
    }

    /// <summary>The matrix transpose of a 2-D array (or the identity for 1-D).</summary>
    public NdArray T => Rank <= 1 ? this : Transpose();

    /// <summary>Inserts a length-1 axis at <paramref name="axis"/>.</summary>
    public NdArray ExpandDims(int axis)
    {
        var a = axis < 0 ? axis + Rank + 1 : axis;
        if (a < 0 || a > Rank) throw new ArgumentOutOfRangeException(nameof(axis));
        var shape = new List<int>(_shape);
        shape.Insert(a, 1);
        return Reshape(shape.ToArray());
    }

    /// <summary>Removes every length-1 axis.</summary>
    public NdArray Squeeze()
    {
        var shape = _shape.Where(d => d != 1).ToArray();
        if (shape.Length == 0) shape = [1];
        return Reshape(shape);
    }

    /// <summary>A view stretched to <paramref name="shape"/> using broadcast rules; no data is copied.</summary>
    public NdArray BroadcastTo(params int[] shape)
    {
        var strides = Shapes.BroadcastStrides(_shape, _strides, shape);
        return new NdArray(_buffer, (int[])shape.Clone(), strides, _offset);
    }

    /// <summary>A deep copy with fresh, contiguous storage.</summary>
    public NdArray Copy() => new(ToArray(), _shape);

    /// <summary>This array if it is already contiguous, otherwise a compacted copy.</summary>
    public NdArray AsContiguous() => IsContiguous ? this : Copy();

    /// <summary>Copies every element into a new row-major <see cref="double"/> array.</summary>
    public double[] ToArray()
    {
        var result = new double[Size];
        CopyTo(result);
        return result;
    }

    /// <summary>Copies every element into <paramref name="destination"/> in row-major order.</summary>
    public void CopyTo(Span<double> destination)
    {
        if (destination.Length < Size)
            throw new ArgumentException($"Destination holds {destination.Length} elements, need {Size}.");

        if (IsContiguous)
        {
            _buffer.AsSpan(_offset, Size).CopyTo(destination);
            return;
        }

        var counter = new int[Rank];
        var linear = _offset;
        for (var i = 0; i < Size; i++)
        {
            destination[i] = _buffer[linear];
            for (var axis = Rank - 1; axis >= 0; axis--)
            {
                counter[axis]++;
                linear += _strides[axis];
                if (counter[axis] < _shape[axis]) break;
                linear -= _strides[axis] * _shape[axis];
                counter[axis] = 0;
            }
        }
    }

    /// <summary>A 2-D copy as a rectangular array.</summary>
    public double[,] To2DArray()
    {
        if (Rank != 2) throw new InvalidOperationException($"To2DArray requires rank 2, got {Rank}.");
        var result = new double[_shape[0], _shape[1]];
        for (var i = 0; i < _shape[0]; i++)
            for (var j = 0; j < _shape[1]; j++)
                result[i, j] = this[i, j];
        return result;
    }

    /// <summary>The contiguous span backing this array. Throws when the view is strided.</summary>
    public Span<double> AsSpan()
    {
        if (!IsContiguous) throw new InvalidOperationException("AsSpan requires a contiguous array; call AsContiguous() first.");
        return _buffer.AsSpan(_offset, Size);
    }

    /// <summary>Overwrites every element with <paramref name="value"/>.</summary>
    public void Fill(double value)
    {
        if (IsContiguous) { _buffer.AsSpan(_offset, Size).Fill(value); return; }
        for (var i = 0; i < Size; i++) SetAt(i, value);
    }

    /// <summary>Copies <paramref name="source"/> into this array, broadcasting if needed.</summary>
    public void Assign(NdArray source)
    {
        var view = source.Size == Size && source.Rank == Rank ? source : source.BroadcastTo(_shape);
        for (var i = 0; i < Size; i++) SetAt(i, view.At(i));
    }

    // ---------------------------------------------------------------- joining

    /// <summary>Concatenates arrays along an existing axis.</summary>
    public static NdArray Concatenate(IReadOnlyList<NdArray> arrays, int axis = 0)
    {
        if (arrays.Count == 0) throw new ArgumentException("Nothing to concatenate.", nameof(arrays));
        var rank = arrays[0].Rank;
        var a = Shapes.NormalizeAxis(axis, rank);

        var shape = arrays[0]._shape.ToArray();
        shape[a] = arrays.Sum(x => x._shape[a]);
        for (var i = 1; i < arrays.Count; i++)
        {
            if (arrays[i].Rank != rank) throw new ArgumentException("All arrays must have the same rank.");
            for (var d = 0; d < rank; d++)
                if (d != a && arrays[i]._shape[d] != shape[d])
                    throw new ArgumentException($"Axis {d} differs: {arrays[i]._shape[d]} vs {shape[d]}.");
        }

        var result = Zeros(shape);
        var written = 0;
        foreach (var arr in arrays)
        {
            var selectors = new Sel[rank];
            for (var d = 0; d < rank; d++) selectors[d] = Sel.All;
            selectors[a] = Sel.Range(written, written + arr._shape[a]);
            result.Slice(selectors).Assign(arr);
            written += arr._shape[a];
        }
        return result;
    }

    /// <summary>Stacks arrays of identical shape along a new leading axis.</summary>
    public static NdArray Stack(IReadOnlyList<NdArray> arrays)
        => Concatenate(arrays.Select(a => a.ExpandDims(0)).ToList());

    /// <summary>Stacks 2-D arrays vertically (along axis 0).</summary>
    public static NdArray VStack(params NdArray[] arrays)
        => Concatenate(arrays.Select(a => a.Rank == 1 ? a.Reshape(1, a.Size) : a).ToList(), 0);

    /// <summary>Stacks 2-D arrays horizontally (along axis 1).</summary>
    public static NdArray HStack(params NdArray[] arrays)
        => Concatenate(arrays.Select(a => a.Rank == 1 ? a.Reshape(a.Size, 1) : a).ToList(), 1);

    // ---------------------------------------------------------------- rendering

    /// <inheritdoc />
    public override string ToString() => ToString(6);

    /// <summary>Renders the array with a fixed number of significant digits.</summary>
    public string ToString(int precision)
    {
        var sb = new StringBuilder();
        sb.Append("NdArray(shape=[").Append(Shapes.Describe(_shape)).Append("])");
        if (Size == 0) return sb.Append(" []").ToString();
        sb.AppendLine();
        Render(sb, [], precision);
        return sb.ToString();
    }

    private void Render(StringBuilder sb, int[] prefix, int precision)
    {
        var axis = prefix.Length;
        var indent = new string(' ', axis * 2);

        if (axis == Rank - 1)
        {
            sb.Append(indent).Append('[');
            var limit = Math.Min(_shape[axis], 12);
            for (var i = 0; i < limit; i++)
            {
                if (i > 0) sb.Append(", ");
                var idx = prefix.Append(i).ToArray();
                sb.Append(Math.Round(this[idx], precision).ToString($"0.{new string('#', precision)}"));
            }
            if (limit < _shape[axis]) sb.Append(", ...");
            sb.Append(']');
            sb.AppendLine();
            return;
        }

        sb.Append(indent).AppendLine("[");
        var outer = Math.Min(_shape[axis], 8);
        for (var i = 0; i < outer; i++) Render(sb, prefix.Append(i).ToArray(), precision);
        if (outer < _shape[axis]) sb.Append(indent).AppendLine("  ...");
        sb.Append(indent).AppendLine("]");
    }
}
