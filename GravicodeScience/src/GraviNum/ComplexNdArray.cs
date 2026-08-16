using System.Numerics;
using Gravicode.Science.GraviNum.Signal;

namespace Gravicode.Science.GraviNum;

/// <summary>
/// An n-dimensional array of complex numbers.
/// </summary>
/// <remarks>
/// <para>
/// The companion to <see cref="NdArray"/> for the work that is naturally complex: spectra, transfer
/// functions, the eigenvalues of a non-symmetric matrix, anything phase-carrying. Splitting a
/// complex problem into two real arrays works and is miserable to read — every multiplication
/// becomes four, and the sign on one of them is the bug everyone writes at least once.
/// </para>
/// <para>
/// Storage is a flat <see cref="Complex"/> buffer in row-major order. <see cref="Complex"/> is a
/// struct of two doubles, so the array is contiguous interleaved real/imaginary pairs — the same
/// layout FFTW and NumPy use, which is what lets the buffer be handed to
/// <see cref="Fft"/> without a repack.
/// </para>
/// <para>
/// Unlike <see cref="NdArray"/>, this does not implement strided views: reshaping and transposing
/// copy. Views exist on the real array because element-wise kernels there are bandwidth-bound and a
/// copy is most of the cost; complex arithmetic does six flops per element, so the copy is no
/// longer what dominates, and a plain contiguous buffer keeps every kernel here simple.
/// </para>
/// </remarks>
public sealed class ComplexNdArray
{
    private readonly Complex[] _data;
    private readonly int[] _shape;

    /// <summary>Wraps a buffer in the given shape without copying.</summary>
    public ComplexNdArray(Complex[] data, params int[] shape)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(shape);

        var size = 1;
        foreach (var dimension in shape)
        {
            if (dimension < 0) throw new ArgumentException("Shape dimensions cannot be negative.", nameof(shape));
            size *= dimension;
        }

        if (size != data.Length)
            throw new ArgumentException(
                $"Shape [{string.Join(", ", shape)}] needs {size} elements but {data.Length} were given.");

        _data = data;
        _shape = shape.Length == 0 ? [data.Length] : shape;
    }

    /// <summary>An array of zeros.</summary>
    public static ComplexNdArray Zeros(params int[] shape)
    {
        var size = 1;
        foreach (var dimension in shape) size *= dimension;
        return new ComplexNdArray(new Complex[size], shape);
    }

    /// <summary>Builds from separate real and imaginary parts, which must have the same shape.</summary>
    public static ComplexNdArray FromParts(NdArray real, NdArray? imaginary = null)
    {
        ArgumentNullException.ThrowIfNull(real);

        if (imaginary is not null && !real.Shape.SequenceEqual(imaginary.Shape))
            throw new ArgumentException("The real and imaginary parts must have the same shape.");

        var data = new Complex[real.Size];
        for (var i = 0; i < real.Size; i++)
            data[i] = new Complex(real.At(i), imaginary?.At(i) ?? 0.0);

        return new ComplexNdArray(data, [.. real.Shape]);
    }

    /// <summary>Builds a flat array from values.</summary>
    public static ComplexNdArray FromValues(params Complex[] values)
        => new((Complex[])values.Clone(), values.Length);

    /// <summary>Element count.</summary>
    public int Size => _data.Length;

    /// <summary>Number of dimensions.</summary>
    public int Rank => _shape.Length;

    /// <summary>The extent of each axis.</summary>
    public IReadOnlyList<int> Shape => _shape;

    /// <summary>The underlying buffer, in row-major order.</summary>
    public Span<Complex> Data => _data;

    /// <summary>Reads or writes by flat index.</summary>
    public Complex this[int index]
    {
        get => _data[index];
        set => _data[index] = value;
    }

    /// <summary>Reads or writes an element of a rank 2 array.</summary>
    public Complex this[int row, int column]
    {
        get => _data[Offset(row, column)];
        set => _data[Offset(row, column)] = value;
    }

    private int Offset(int row, int column)
    {
        if (Rank != 2) throw new InvalidOperationException($"This is a rank {Rank} array, not a matrix.");
        if ((uint)row >= (uint)_shape[0]) throw new IndexOutOfRangeException($"Row {row} is outside 0..{_shape[0] - 1}.");
        if ((uint)column >= (uint)_shape[1]) throw new IndexOutOfRangeException($"Column {column} is outside 0..{_shape[1] - 1}.");
        return row * _shape[1] + column;
    }

    /// <summary>An independent copy.</summary>
    public ComplexNdArray Copy() => new((Complex[])_data.Clone(), [.. _shape]);

    /// <summary>The same data seen under a new shape.</summary>
    /// <remarks>Copies, so the result does not alias the original. A single <c>-1</c> is inferred.</remarks>
    public ComplexNdArray Reshape(params int[] shape)
    {
        var resolved = (int[])shape.Clone();
        var inferred = Array.IndexOf(resolved, -1);

        if (inferred >= 0)
        {
            var known = 1;
            for (var i = 0; i < resolved.Length; i++) if (i != inferred) known *= resolved[i];
            if (known == 0 || Size % known != 0)
                throw new ArgumentException($"Cannot reshape {Size} elements to [{string.Join(", ", shape)}].");
            resolved[inferred] = Size / known;
        }

        return new ComplexNdArray((Complex[])_data.Clone(), resolved);
    }

    // ------------------------------------------------------------------ parts

    /// <summary>The real parts.</summary>
    public NdArray Real()
    {
        var result = NdArray.Zeros([.. _shape]);
        for (var i = 0; i < Size; i++) result.SetAt(i, _data[i].Real);
        return result;
    }

    /// <summary>The imaginary parts.</summary>
    public NdArray Imaginary()
    {
        var result = NdArray.Zeros([.. _shape]);
        for (var i = 0; i < Size; i++) result.SetAt(i, _data[i].Imaginary);
        return result;
    }

    /// <summary>Element-wise magnitude.</summary>
    /// <remarks>
    /// Goes through <see cref="Complex.Abs"/>, which scales by the larger component before squaring.
    /// The obvious <c>sqrt(re² + im²)</c> overflows for magnitudes above about 1.3e154 even when the
    /// answer is perfectly representable.
    /// </remarks>
    public NdArray Magnitude()
    {
        var result = NdArray.Zeros([.. _shape]);
        for (var i = 0; i < Size; i++) result.SetAt(i, Complex.Abs(_data[i]));
        return result;
    }

    /// <summary>Element-wise argument, in radians on (-π, π].</summary>
    public NdArray Phase()
    {
        var result = NdArray.Zeros([.. _shape]);
        for (var i = 0; i < Size; i++) result.SetAt(i, _data[i].Phase);
        return result;
    }

    /// <summary>Element-wise squared magnitude — the power spectrum, when this is a spectrum.</summary>
    /// <remarks>
    /// Not <c>Magnitude()²</c>: skipping the square root avoids both the rounding it introduces and
    /// the cost of computing it only to undo it.
    /// </remarks>
    public NdArray Power()
    {
        var result = NdArray.Zeros([.. _shape]);
        for (var i = 0; i < Size; i++)
        {
            var value = _data[i];
            result.SetAt(i, value.Real * value.Real + value.Imaginary * value.Imaginary);
        }
        return result;
    }

    /// <summary>Element-wise complex conjugate.</summary>
    public ComplexNdArray Conjugate()
    {
        var result = new Complex[Size];
        for (var i = 0; i < Size; i++) result[i] = Complex.Conjugate(_data[i]);
        return new ComplexNdArray(result, [.. _shape]);
    }

    // ------------------------------------------------------------- arithmetic

    /// <summary>Element-wise addition.</summary>
    public static ComplexNdArray operator +(ComplexNdArray a, ComplexNdArray b) => Zip(a, b, static (x, y) => x + y);

    /// <summary>Element-wise subtraction.</summary>
    public static ComplexNdArray operator -(ComplexNdArray a, ComplexNdArray b) => Zip(a, b, static (x, y) => x - y);

    /// <summary>Element-wise multiplication — not a matrix product; see <see cref="Dot"/>.</summary>
    public static ComplexNdArray operator *(ComplexNdArray a, ComplexNdArray b) => Zip(a, b, static (x, y) => x * y);

    /// <summary>Element-wise division.</summary>
    public static ComplexNdArray operator /(ComplexNdArray a, ComplexNdArray b) => Zip(a, b, static (x, y) => x / y);

    /// <summary>Scales every element.</summary>
    public static ComplexNdArray operator *(ComplexNdArray a, Complex scalar) => a.Map(v => v * scalar);

    /// <summary>Scales every element.</summary>
    public static ComplexNdArray operator *(Complex scalar, ComplexNdArray a) => a * scalar;

    /// <summary>Negates every element.</summary>
    public static ComplexNdArray operator -(ComplexNdArray a) => a.Map(static v => -v);

    /// <summary>Applies a function to every element.</summary>
    public ComplexNdArray Map(Func<Complex, Complex> f)
    {
        var result = new Complex[Size];
        for (var i = 0; i < Size; i++) result[i] = f(_data[i]);
        return new ComplexNdArray(result, [.. _shape]);
    }

    private static ComplexNdArray Zip(ComplexNdArray a, ComplexNdArray b, Func<Complex, Complex, Complex> f)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        if (!a._shape.SequenceEqual(b._shape))
            throw new ArgumentException(
                $"Shapes [{string.Join(", ", a._shape)}] and [{string.Join(", ", b._shape)}] do not match.");

        var result = new Complex[a.Size];
        for (var i = 0; i < a.Size; i++) result[i] = f(a._data[i], b._data[i]);
        return new ComplexNdArray(result, [.. a._shape]);
    }

    /// <summary>The sum of every element.</summary>
    public Complex Sum()
    {
        var total = Complex.Zero;
        foreach (var value in _data) total += value;
        return total;
    }

    // ------------------------------------------------------------ linear algebra

    /// <summary>Ordinary transpose, without conjugating.</summary>
    /// <remarks>
    /// Rarely what you want for complex matrices — see <see cref="ConjugateTranspose"/>. It is here
    /// because the two are easy to confuse, and having only one of them under the name "transpose"
    /// is how the wrong one gets used.
    /// </remarks>
    public ComplexNdArray Transpose()
    {
        if (Rank != 2) throw new InvalidOperationException($"This is a rank {Rank} array, not a matrix.");

        var (rows, columns) = (_shape[0], _shape[1]);
        var result = new Complex[Size];

        for (var i = 0; i < rows; i++)
            for (var j = 0; j < columns; j++)
                result[j * rows + i] = _data[i * columns + j];

        return new ComplexNdArray(result, columns, rows);
    }

    /// <summary>
    /// The conjugate (Hermitian) transpose, <c>Aᴴ</c>.
    /// </summary>
    /// <remarks>
    /// This is the complex analogue of a real transpose, and the one nearly every formula means.
    /// <c>AᴴA</c> is positive semi-definite with real diagonal; <c>AᵀA</c> is neither, so using a
    /// plain transpose gives a matrix that looks plausible and has complex "variances" on its
    /// diagonal.
    /// </remarks>
    public ComplexNdArray ConjugateTranspose()
    {
        if (Rank != 2) throw new InvalidOperationException($"This is a rank {Rank} array, not a matrix.");

        var (rows, columns) = (_shape[0], _shape[1]);
        var result = new Complex[Size];

        for (var i = 0; i < rows; i++)
            for (var j = 0; j < columns; j++)
                result[j * rows + i] = Complex.Conjugate(_data[i * columns + j]);

        return new ComplexNdArray(result, columns, rows);
    }

    /// <summary>Matrix product.</summary>
    /// <remarks>
    /// Straightforward triple loop with the accumulation hoisted out of the inner index arithmetic.
    /// The register-blocked kernel <see cref="LinAlg.Dot"/> uses is not replicated here: it is tuned
    /// against a <c>double</c> buffer, and a <see cref="Complex"/> multiply is six flops on a
    /// 16-byte struct, which moves the bottleneck from memory to arithmetic and makes the blocking
    /// far less valuable.
    /// </remarks>
    public static ComplexNdArray Dot(ComplexNdArray a, ComplexNdArray b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        if (a.Rank != 2 || b.Rank != 2) throw new ArgumentException("Both operands must be matrices.");
        if (a._shape[1] != b._shape[0])
            throw new ArgumentException(
                $"Cannot multiply [{a._shape[0]}, {a._shape[1]}] by [{b._shape[0]}, {b._shape[1]}].");

        var (rows, inner, columns) = (a._shape[0], a._shape[1], b._shape[1]);
        var result = new Complex[rows * columns];

        for (var i = 0; i < rows; i++)
            for (var k = 0; k < inner; k++)
            {
                var left = a._data[i * inner + k];
                if (left == Complex.Zero) continue;

                var rowOffset = i * columns;
                var bOffset = k * columns;
                for (var j = 0; j < columns; j++) result[rowOffset + j] += left * b._data[bOffset + j];
            }

        return new ComplexNdArray(result, rows, columns);
    }

    /// <summary>
    /// The Hermitian inner product <c>Σ conj(aᵢ) bᵢ</c>.
    /// </summary>
    /// <remarks>
    /// Conjugating the first operand is what makes <c>⟨a, a⟩</c> a real, non-negative number equal
    /// to the squared norm. Without it, the "norm" of <c>[i]</c> comes out as -1.
    /// </remarks>
    public static Complex Inner(ComplexNdArray a, ComplexNdArray b)
    {
        if (a.Size != b.Size) throw new ArgumentException("Both operands must have the same length.");

        var total = Complex.Zero;
        for (var i = 0; i < a.Size; i++) total += Complex.Conjugate(a._data[i]) * b._data[i];
        return total;
    }

    /// <summary>The Euclidean norm, <c>sqrt(⟨a, a⟩)</c>.</summary>
    public double Norm()
    {
        var total = 0.0;
        foreach (var value in _data) total += value.Real * value.Real + value.Imaginary * value.Imaginary;
        return Math.Sqrt(total);
    }

    // ------------------------------------------------------------------ signal

    /// <summary>The discrete Fourier transform of a flat array.</summary>
    public ComplexNdArray Fft()
    {
        if (Rank != 1) throw new InvalidOperationException("Use Fft2 for a matrix.");
        return new ComplexNdArray(Signal.Fft.Forward(_data.ToArray()), Size);
    }

    /// <summary>The inverse transform of a flat array.</summary>
    public ComplexNdArray Ifft()
    {
        if (Rank != 1) throw new InvalidOperationException("Use Ifft2 for a matrix.");
        return new ComplexNdArray(Signal.Fft.Inverse(_data.ToArray()), Size);
    }

    /// <summary>
    /// The two-dimensional transform: rows first, then columns.
    /// </summary>
    /// <remarks>
    /// The 2-D DFT is separable, so it factors into 1-D transforms along each axis — which turns an
    /// O(n⁴) double sum into O(n² log n) and is the only reason image-sized transforms are
    /// practical. The order of the two passes does not affect the result.
    /// </remarks>
    public ComplexNdArray Fft2() => Transform2(inverse: false);

    /// <summary>The inverse two-dimensional transform.</summary>
    public ComplexNdArray Ifft2() => Transform2(inverse: true);

    private ComplexNdArray Transform2(bool inverse)
    {
        if (Rank != 2) throw new InvalidOperationException($"This is a rank {Rank} array, not a matrix.");

        var (rows, columns) = (_shape[0], _shape[1]);
        var result = (Complex[])_data.Clone();

        var buffer = new Complex[columns];
        for (var i = 0; i < rows; i++)
        {
            Array.Copy(result, i * columns, buffer, 0, columns);
            var transformed = inverse ? Signal.Fft.Inverse(buffer) : Signal.Fft.Forward(buffer);
            Array.Copy(transformed, 0, result, i * columns, columns);
        }

        var column = new Complex[rows];
        for (var j = 0; j < columns; j++)
        {
            for (var i = 0; i < rows; i++) column[i] = result[i * columns + j];
            var transformed = inverse ? Signal.Fft.Inverse(column) : Signal.Fft.Forward(column);
            for (var i = 0; i < rows; i++) result[i * columns + j] = transformed[i];
        }

        return new ComplexNdArray(result, rows, columns);
    }

    // ------------------------------------------------------------------ misc

    /// <summary>True when every element agrees with <paramref name="other"/> to <paramref name="tolerance"/>.</summary>
    public bool AllClose(ComplexNdArray other, double tolerance = 1e-9)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (!_shape.SequenceEqual(other._shape)) return false;

        for (var i = 0; i < Size; i++)
            if (Complex.Abs(_data[i] - other._data[i]) > tolerance) return false;

        return true;
    }

    /// <summary>A copy of the buffer.</summary>
    public Complex[] ToArray() => (Complex[])_data.Clone();

    /// <inheritdoc />
    public override string ToString()
    {
        var preview = string.Join(", ", _data.Take(6).Select(Format));
        if (Size > 6) preview += ", ...";
        return $"ComplexNdArray[{string.Join(", ", _shape)}] ({preview})";
    }

    private static string Format(Complex value)
        => value.Imaginary >= 0
            ? $"{value.Real:G6}+{value.Imaginary:G6}i"
            : $"{value.Real:G6}-{Math.Abs(value.Imaginary):G6}i";
}
