using System.Collections;
using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Sparse;

namespace Gravicode.Science.SciNet.IO;

/// <summary>
/// MATLAB array classes as numbered in the MAT-file format (<c>mxCLASS_ID</c>). Logical arrays
/// are not a class of their own on disk: they are <see cref="UInt8"/> (or <see cref="Sparse"/>)
/// with the logical flag set, which is how <see cref="MatLogical"/> and
/// <see cref="MatSparse.IsLogical"/> report them.
/// </summary>
public enum MatClass
{
    /// <summary>Cell array.</summary>
    Cell = 1,
    /// <summary>Structure (or structure array).</summary>
    Struct = 2,
    /// <summary>Old-style (pre-classdef) MATLAB object: a structure with a class name.</summary>
    Object = 3,
    /// <summary>Character array.</summary>
    Char = 4,
    /// <summary>Sparse matrix (double or logical).</summary>
    Sparse = 5,
    /// <summary>64-bit float.</summary>
    Double = 6,
    /// <summary>32-bit float.</summary>
    Single = 7,
    /// <summary>Signed 8-bit integer.</summary>
    Int8 = 8,
    /// <summary>Unsigned 8-bit integer.</summary>
    UInt8 = 9,
    /// <summary>Signed 16-bit integer.</summary>
    Int16 = 10,
    /// <summary>Unsigned 16-bit integer.</summary>
    UInt16 = 11,
    /// <summary>Signed 32-bit integer.</summary>
    Int32 = 12,
    /// <summary>Unsigned 32-bit integer.</summary>
    UInt32 = 13,
    /// <summary>Signed 64-bit integer.</summary>
    Int64 = 14,
    /// <summary>Unsigned 64-bit integer.</summary>
    UInt64 = 15,
    /// <summary>Function handle (not decodable; read as <see cref="MatUnsupported"/>).</summary>
    Function = 16,
    /// <summary>Opaque classdef object, string, table, ... (not decodable; read as <see cref="MatUnsupported"/>).</summary>
    Opaque = 17,
}

/// <summary>
/// One variable (or one cell / field element) of a MAT file. A small typed model in place of
/// SciPy's mix of ndarrays, <c>mat_struct</c> and object arrays: every subclass carries the MATLAB
/// <see cref="Dimensions"/> and <see cref="ClassName"/>, and the payload is in the natural .NET form
/// (<see cref="NdArray"/>, <see cref="ComplexNdArray"/>, <see cref="CscMatrix"/>, strings).
/// </summary>
/// <remarks>
/// MATLAB stores arrays column-major; every payload here is already converted to the row-major
/// order GraviNum uses, with the same shape — MATLAB's <c>A(i,j,k)</c> is <c>array[i-1, j-1, k-1]</c>.
/// Cell and struct elements are likewise held in row-major order of <see cref="Dimensions"/>.
/// </remarks>
public abstract class MatValue
{
    private protected int[] Dims;

    private protected MatValue(int[] dimensions)
    {
        ArgumentNullException.ThrowIfNull(dimensions);
        foreach (var d in dimensions)
            if (d < 0) throw new ArgumentException("Dimensions must be non-negative.", nameof(dimensions));
        Dims = (int[])dimensions.Clone();
    }

    /// <summary>MATLAB dimensions (at least two unless the value was squeezed).</summary>
    public IReadOnlyList<int> Dimensions => Dims;

    /// <summary>Number of elements (the product of <see cref="Dimensions"/>; 1 for a squeezed scalar).</summary>
    public int Count => MatLayout.Product(Dims);

    /// <summary>True when the value has no elements.</summary>
    public bool IsEmpty => Count == 0;

    /// <summary>What MATLAB's <c>class()</c> would return: <c>double</c>, <c>int16</c>, <c>logical</c>, <c>char</c>, <c>cell</c>, <c>struct</c>, an object's class name, ...</summary>
    public abstract string ClassName { get; }

    /// <summary>True when the variable was saved as a MATLAB global.</summary>
    public bool IsGlobal { get; internal set; }

    /// <summary>The numeric payload; throws unless this is a real numeric, logical or sparse value.</summary>
    public virtual NdArray ToNdArray() => throw new InvalidOperationException($"A MATLAB '{ClassName}' value has no numeric array form.");

    /// <summary>The single element of a 1-element real numeric or logical value.</summary>
    public double Scalar
    {
        get
        {
            var a = ToNdArray();
            if (a.Size != 1) throw new InvalidOperationException($"Value has {a.Size} elements, not 1.");
            return a.AsContiguous().AsSpan()[0];
        }
    }

    /// <summary>The text of a single-row character value; throws for anything else.</summary>
    public virtual string Text => throw new InvalidOperationException($"A MATLAB '{ClassName}' value is not text.");

    /// <summary>Returns the same value with every length-1 dimension removed (SciPy's <c>squeeze_me</c>); nested cells and fields are squeezed too.</summary>
    public MatValue Squeeze()
    {
        var kept = Dims.Where(d => d != 1).ToArray();
        var v = WithDimensions(kept);
        v.IsGlobal = IsGlobal;
        return v;
    }

    /// <summary>Re-labels the dimensions without moving data (removing unit axes keeps the row-major order).</summary>
    private protected abstract MatValue WithDimensions(int[] dims);

    /// <summary>
    /// Converts a .NET value into the MAT model, the way <see cref="MatFile.Save(string, IReadOnlyDictionary{string, object?}, bool, string, string)"/> does:
    /// <see cref="MatValue"/> as is; <see cref="NdArray"/>/<see cref="ComplexNdArray"/> as double;
    /// primitive scalars and (multi-dimensional) arrays keep their MATLAB class (<c>int</c> → int32,
    /// <c>float</c> → single, <c>bool</c> → logical, ...); <c>string</c>/<c>string[]</c>/<c>char[]</c> → char;
    /// <see cref="SparseMatrixBase"/> → sparse; dictionaries → scalar struct; <c>object[]</c>/<c>IList</c>
    /// → cell row; <c>object[,]</c> → 2-D cell; <c>null</c> → empty 0x0 double.
    /// </summary>
    /// <param name="value">The value to convert.</param>
    /// <param name="onedAsColumn">Write 1-D data as N x 1 (SciPy's <c>oned_as='column'</c>) rather than 1 x N.</param>
    public static MatValue FromObject(object? value, bool onedAsColumn = false) => MatConvert.FromObject(value, onedAsColumn);
}

/// <summary>Real or complex numeric array of any MATLAB numeric class (double, single, int8 ... uint64).</summary>
/// <remarks>
/// Integer and single data are widened to <c>double</c> on read (the original class is kept in
/// <see cref="Class"/> and is used again on <see cref="MatFile.Save(string, IReadOnlyDictionary{string, object?}, bool, string, string)"/>).
/// int64/uint64 values beyond 2^53 lose their low bits in that widening.
/// </remarks>
public sealed class MatArray : MatValue
{
    /// <summary>Wraps real data; the array's shape becomes the MATLAB dimensions (a rank-1 array becomes 1 x N).</summary>
    public MatArray(NdArray real, MatClass matClass = MatClass.Double)
        : this(MatConvert.DimsOf(real), real, null, matClass) { }

    /// <summary>Wraps complex data; the array's shape becomes the MATLAB dimensions.</summary>
    public MatArray(ComplexNdArray value, MatClass matClass = MatClass.Double)
        : this(MatConvert.DimsOf(value.Shape), value.Real(), value.Imaginary(), matClass) { }

    internal MatArray(int[] dims, NdArray real, NdArray? imaginary, MatClass matClass) : base(dims)
    {
        if (!MatLayout.IsNumeric(matClass)) throw new ArgumentException($"{matClass} is not a numeric MATLAB class.", nameof(matClass));
        if (real.Size != MatLayout.Product(dims)) throw new ArgumentException("Data size does not match dimensions.");
        if (imaginary is not null && imaginary.Size != real.Size) throw new ArgumentException("Real and imaginary parts differ in size.");
        Real = real;
        Imaginary = imaginary;
        Class = matClass;
    }

    /// <summary>The storage class in the file (kept so a save writes the same class back).</summary>
    public MatClass Class { get; }

    /// <inheritdoc />
    public override string ClassName => MatLayout.ClassName(Class);

    /// <summary>Real part, row-major, shaped like <see cref="MatValue.Dimensions"/> (shape [1] for a squeezed scalar).</summary>
    public NdArray Real { get; }

    /// <summary>Imaginary part, or null for real data.</summary>
    public NdArray? Imaginary { get; }

    /// <summary>True when the array has an imaginary part.</summary>
    public bool IsComplex => Imaginary is not null;

    /// <summary>The data; throws for complex arrays so the imaginary part cannot be dropped silently.</summary>
    public override NdArray ToNdArray() => IsComplex
        ? throw new InvalidOperationException("The array is complex; use ToComplex() or Real.")
        : Real;

    /// <summary>The data as a complex array (zero imaginary part for real data).</summary>
    public ComplexNdArray ToComplex() => ComplexNdArray.FromParts(Real, Imaginary);

    private protected override MatValue WithDimensions(int[] dims)
        => new MatArray(dims, Real.Reshape(MatLayout.ShapeOf(dims)), Imaginary?.Reshape(MatLayout.ShapeOf(dims)), Class);
}

/// <summary>MATLAB logical array (stored as 0/1 doubles here).</summary>
public sealed class MatLogical : MatValue
{
    /// <summary>Builds from booleans laid out row-major over <paramref name="dimensions"/>.</summary>
    public MatLogical(bool[] values, params int[] dimensions)
        : this(dimensions.Length == 0 ? [1, values.Length] : dimensions, new NdArray(values.Select(b => b ? 1.0 : 0.0).ToArray(), MatLayout.ShapeOf(dimensions.Length == 0 ? [1, values.Length] : dimensions))) { }

    internal MatLogical(int[] dims, NdArray values) : base(dims)
    {
        if (values.Size != MatLayout.Product(dims)) throw new ArgumentException("Data size does not match dimensions.");
        Values = values;
    }

    /// <summary>0/1 values, row-major.</summary>
    public NdArray Values { get; }

    /// <inheritdoc />
    public override string ClassName => "logical";

    /// <inheritdoc />
    public override NdArray ToNdArray() => Values;

    /// <summary>The values as booleans, row-major.</summary>
    public bool[] ToBoolArray()
    {
        var s = Values.AsContiguous().AsSpan();
        var r = new bool[s.Length];
        for (var i = 0; i < s.Length; i++) r[i] = s[i] != 0;
        return r;
    }

    private protected override MatValue WithDimensions(int[] dims) => new MatLogical(dims, Values.Reshape(MatLayout.ShapeOf(dims)));
}

/// <summary>MATLAB character array. Characters are UTF-16 code units, as in MATLAB.</summary>
public sealed class MatChar : MatValue
{
    private readonly char[] _chars;

    /// <summary>A 1 x N character row (an empty string becomes 0 x 0, as in MATLAB).</summary>
    public MatChar(string text) : this(text.Length == 0 ? [0, 0] : [1, text.Length], text.ToCharArray()) { }

    /// <summary>A character matrix, one row per string; shorter rows are padded with spaces (MATLAB's <c>char()</c> behaviour).</summary>
    public MatChar(IReadOnlyList<string> rows) : this(MatConvert.CharMatrix(rows, out var dims), dims) { }

    private MatChar(char[] chars, int[] dims) : this(dims, chars) { }

    internal MatChar(int[] dims, char[] chars) : base(dims)
    {
        if (chars.Length != MatLayout.Product(dims)) throw new ArgumentException("Character count does not match dimensions.");
        _chars = chars;
    }

    /// <summary>Characters in row-major order.</summary>
    public ReadOnlySpan<char> Chars => _chars;

    /// <inheritdoc />
    public override string ClassName => "char";

    /// <summary>
    /// One string per row along the last dimension (SciPy's <c>chars_as_strings</c>): a 2 x 3 char
    /// matrix gives two 3-character strings. Trailing padding is kept, as SciPy does.
    /// </summary>
    public string[] Strings
    {
        get
        {
            if (Dims.Length == 0) return [new string(_chars)];
            var len = Dims[^1];
            if (len == 0) return Enumerable.Repeat("", MatLayout.Product(Dims[..^1])).ToArray();
            var n = _chars.Length / len;
            var r = new string[n];
            for (var i = 0; i < n; i++) r[i] = new string(_chars, i * len, len);
            return r;
        }
    }

    /// <summary>The text of an empty or single-row character array; throws for a multi-row matrix (use <see cref="Strings"/>).</summary>
    public override string Text
    {
        get
        {
            if (_chars.Length == 0) return "";
            var s = Strings;
            if (s.Length != 1) throw new InvalidOperationException($"Character array has {s.Length} rows; use Strings.");
            return s[0];
        }
    }

    private protected override MatValue WithDimensions(int[] dims) => new MatChar(dims, _chars);
}

/// <summary>MATLAB sparse matrix: real, complex or logical.</summary>
/// <remarks>
/// <see cref="CscMatrix"/> is real, so a complex sparse matrix keeps its imaginary part in
/// <see cref="Imaginary"/>, aligned entry for entry with <c>Matrix.Data</c> (same CSC structure).
/// </remarks>
public sealed class MatSparse : MatValue
{
    private readonly double[]? _imag;

    /// <summary>Wraps a real (or, with <paramref name="isLogical"/>, logical) sparse matrix.</summary>
    public MatSparse(SparseMatrixBase matrix, bool isLogical = false)
        : this(MatConvert.Canonical(matrix.ToCsc()), null, isLogical) { }

    /// <summary>A complex sparse matrix: <paramref name="imaginary"/> is aligned with <c>real.Data</c> in canonical CSC order.</summary>
    public MatSparse(CscMatrix real, double[] imaginary) : this(real, imaginary, false)
    {
        if (imaginary.Length != real.Nnz) throw new ArgumentException("Imaginary data must have one entry per stored value.");
    }

    internal MatSparse(CscMatrix matrix, double[]? imaginary, bool isLogical) : base([matrix.Rows, matrix.Columns])
    {
        Matrix = matrix;
        _imag = imaginary;
        IsLogical = isLogical;
    }

    /// <summary>The (real part of the) matrix in compressed sparse column form.</summary>
    public CscMatrix Matrix { get; }

    /// <summary>Imaginary parts aligned with <c>Matrix.Data</c>, or null for real/logical matrices.</summary>
    public ReadOnlySpan<double> Imaginary => _imag;

    /// <summary>True for a complex sparse matrix.</summary>
    public bool IsComplex => _imag is not null;

    /// <summary>True for a logical sparse matrix (values are 0/1).</summary>
    public bool IsLogical { get; }

    /// <inheritdoc />
    public override string ClassName => IsLogical ? "logical" : "double";

    /// <summary>Dense real form; throws for complex matrices (use <see cref="ToComplexDense"/>).</summary>
    public override NdArray ToNdArray() => IsComplex
        ? throw new InvalidOperationException("The sparse matrix is complex; use ToComplexDense().")
        : Matrix.ToDense();

    /// <summary>Dense complex form.</summary>
    public ComplexNdArray ToComplexDense()
    {
        var rows = Matrix.Rows;
        var cols = Matrix.Columns;
        var c = new Complex[rows * cols];
        var ptr = Matrix.Indptr;
        var idx = Matrix.Indices;
        var re = Matrix.Data;
        for (var j = 0; j < cols; j++)
            for (var k = ptr[j]; k < ptr[j + 1]; k++)
                c[idx[k] * cols + j] += new Complex(re[k], _imag is null ? 0 : _imag[k]);
        return new ComplexNdArray(c, rows, cols);
    }

    private protected override MatValue WithDimensions(int[] dims) => this; // sparse matrices are always 2-D, as in SciPy
}

/// <summary>
/// MATLAB structure or structure array (and old-style objects, which are structures with a class
/// name). A scalar structure reads like a dictionary: <c>s["field"]</c>.
/// </summary>
public sealed class MatStruct : MatValue, IReadOnlyDictionary<string, MatValue>
{
    private readonly string[] _fields;
    private readonly MatValue[][] _elements; // [element][field]

    /// <summary>A 1 x 1 structure from field/value pairs (values converted with <see cref="MatValue.FromObject"/>).</summary>
    public MatStruct(IEnumerable<KeyValuePair<string, object?>> fields, string? objectClassName = null)
        : this(MatConvert.ScalarStruct(fields, out var names), names, objectClassName) { }

    private MatStruct(MatValue[] values, string[] names, string? objectClassName)
        : this([1, 1], names, [values], objectClassName) { }

    /// <summary>A structure array: <paramref name="elements"/> are in row-major order of <paramref name="dimensions"/>, each holding one value per field.</summary>
    public MatStruct(int[] dimensions, IReadOnlyList<string> fieldNames, IReadOnlyList<IReadOnlyList<MatValue>> elements, string? objectClassName = null)
        : base(dimensions)
    {
        _fields = fieldNames.ToArray();
        if (_fields.Distinct(StringComparer.Ordinal).Count() != _fields.Length) throw new ArgumentException("Field names must be unique.");
        if (elements.Count != MatLayout.Product(dimensions)) throw new ArgumentException("Element count does not match dimensions.");
        _elements = new MatValue[elements.Count][];
        for (var e = 0; e < elements.Count; e++)
        {
            if (elements[e].Count != _fields.Length) throw new ArgumentException($"Element {e} has {elements[e].Count} values for {_fields.Length} fields.");
            _elements[e] = elements[e].ToArray();
        }
        ObjectClassName = objectClassName;
    }

    /// <summary>Field names in file order.</summary>
    public IReadOnlyList<string> FieldNames => _fields;

    /// <summary>The class name of an old-style MATLAB object, or null for a plain structure.</summary>
    public string? ObjectClassName { get; }

    /// <summary>True when this is an old-style MATLAB object rather than a plain structure.</summary>
    public bool IsObject => ObjectClassName is not null;

    /// <inheritdoc />
    public override string ClassName => ObjectClassName ?? "struct";

    /// <summary>Field value of element <paramref name="index"/> (row-major element order).</summary>
    public MatValue this[int index, string field] => _elements[index][FieldIndex(field)];

    /// <summary>Field value of a scalar (1-element) structure.</summary>
    public MatValue this[string key]
    {
        get
        {
            if (_elements.Length != 1) throw new InvalidOperationException($"Structure array has {_elements.Length} elements; use [index, field].");
            return _elements[0][FieldIndex(key)];
        }
    }

    /// <summary>Element <paramref name="index"/> as a field → value dictionary.</summary>
    public IReadOnlyDictionary<string, MatValue> Element(int index)
    {
        var d = new Dictionary<string, MatValue>(StringComparer.Ordinal);
        for (var f = 0; f < _fields.Length; f++) d[_fields[f]] = _elements[index][f];
        return d;
    }

    internal MatValue[] RawElement(int index) => _elements[index];

    private int FieldIndex(string field)
    {
        var i = Array.IndexOf(_fields, field);
        if (i < 0) throw new KeyNotFoundException($"No field '{field}'. Fields: {string.Join(", ", _fields)}.");
        return i;
    }

    /// <inheritdoc />
    public IEnumerable<string> Keys => _fields;

    /// <inheritdoc />
    public IEnumerable<MatValue> Values => _fields.Select(f => this[f]);

    int IReadOnlyCollection<KeyValuePair<string, MatValue>>.Count => _fields.Length;

    /// <inheritdoc />
    public bool ContainsKey(string key) => Array.IndexOf(_fields, key) >= 0;

    /// <inheritdoc />
    public bool TryGetValue(string key, [System.Diagnostics.CodeAnalysis.MaybeNullWhen(false)] out MatValue value)
    {
        var i = Array.IndexOf(_fields, key);
        if (i < 0 || _elements.Length != 1) { value = null; return false; }
        value = _elements[0][i];
        return true;
    }

    /// <inheritdoc />
    public IEnumerator<KeyValuePair<string, MatValue>> GetEnumerator()
    {
        if (_elements.Length != 1) throw new InvalidOperationException($"Structure array has {_elements.Length} elements; enumerate Element(i) instead.");
        for (var f = 0; f < _fields.Length; f++) yield return new(_fields[f], _elements[0][f]);
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    private protected override MatValue WithDimensions(int[] dims)
        => new MatStruct(dims, _fields, _elements.Select(e => (IReadOnlyList<MatValue>)e.Select(v => v.Squeeze()).ToArray()).ToArray(), ObjectClassName);
}

/// <summary>MATLAB cell array; elements are in row-major order of <see cref="MatValue.Dimensions"/>.</summary>
public sealed class MatCell : MatValue, IReadOnlyList<MatValue>
{
    private readonly MatValue[] _items;

    /// <summary>A 1 x N cell row of converted values.</summary>
    public MatCell(params object?[] items) : this([1, items.Length], items.Select(i => MatValue.FromObject(i)).ToArray()) { }

    /// <summary>A cell array with the given dimensions; <paramref name="items"/> are row-major.</summary>
    public MatCell(int[] dimensions, IReadOnlyList<MatValue> items) : base(dimensions)
    {
        if (items.Count != MatLayout.Product(dimensions)) throw new ArgumentException("Item count does not match dimensions.");
        _items = items.ToArray();
    }

    /// <inheritdoc />
    public override string ClassName => "cell";

    /// <summary>Element by row-major linear index.</summary>
    public MatValue this[int index] => _items[index];

    /// <summary>Element by subscripts (zero-based, one per dimension).</summary>
    public MatValue this[params int[] index] => _items[MatLayout.RowMajorIndex(Dims, index)];

    int IReadOnlyCollection<MatValue>.Count => _items.Length;

    /// <inheritdoc />
    public IEnumerator<MatValue> GetEnumerator() => ((IEnumerable<MatValue>)_items).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    private protected override MatValue WithDimensions(int[] dims) => new MatCell(dims, _items.Select(v => v.Squeeze()).ToArray());
}

/// <summary>
/// A variable that SciNet cannot decode — function handles and opaque classdef objects (MATLAB
/// <c>string</c>, <c>table</c>, <c>datetime</c>, ...). They are skipped with a reason rather than
/// failing the whole load, exactly as SciPy keeps going past them.
/// </summary>
public sealed class MatUnsupported : MatValue
{
    internal MatUnsupported(int[] dims, MatClass matClass, string message) : base(dims)
    {
        Class = matClass;
        Message = message;
    }

    /// <summary>The class found in the file.</summary>
    public MatClass Class { get; }

    /// <summary>Why the value was not decoded.</summary>
    public string Message { get; }

    /// <inheritdoc />
    public override string ClassName => Class switch { MatClass.Function => "function_handle", MatClass.Opaque => "opaque", _ => "unknown" };

    private protected override MatValue WithDimensions(int[] dims) => this;
}
