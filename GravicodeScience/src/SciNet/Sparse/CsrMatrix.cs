using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Sparse;

/// <summary>
/// Compressed sparse row matrix — <c>scipy.sparse.csr_matrix</c>. The workhorse format: fast
/// row slicing, fast <c>A x</c>, fast sparse products.
/// </summary>
/// <remarks>
/// Row <c>i</c>'s entries are <c>Data[Indptr[i]..Indptr[i+1]]</c> at columns
/// <c>Indices[Indptr[i]..Indptr[i+1]]</c>. Indices need not be sorted and duplicates are allowed
/// (they add up), exactly as in SciPy; <see cref="HasCanonicalFormat"/> says whether either
/// happened, and every operation that needs canonical form canonicalises a copy first.
/// The matrix is immutable; see <see cref="SparseMatrixBase"/>.
/// </remarks>
public sealed class CsrMatrix : SparseMatrixBase
{
    private readonly int[] _indptr;
    private readonly int[] _indices;
    private readonly double[] _data;
    private int _canonical = -1; // -1 unknown, 0 no, 1 yes
    private int _sorted = -1;

    internal CsrMatrix(int rows, int columns, int[] indptr, int[] indices, double[] data, bool? canonical = null)
        : base(rows, columns)
    {
        _indptr = indptr;
        _indices = indices;
        _data = data;
        if (canonical == true) { _canonical = 1; _sorted = 1; }
    }

    internal CsrMatrix(in CompressedArrays a, bool? canonical = null) : this(a.Major, a.Minor, a.Ptr, a.Idx, a.Val, canonical) { }

    /// <summary>
    /// Builds a CSR matrix from its three arrays (SciPy's <c>csr_matrix((data, indices, indptr), shape)</c>).
    /// The arrays are validated and, unless <paramref name="copy"/> is false, copied.
    /// </summary>
    /// <param name="data">Stored values.</param>
    /// <param name="indices">Column of each stored value.</param>
    /// <param name="indptr">Row start offsets, length <c>rows + 1</c>, non-decreasing, ending at <c>data.Length</c>.</param>
    /// <param name="rows">Number of rows.</param>
    /// <param name="columns">Number of columns.</param>
    /// <param name="copy">Copy the arrays (default) or adopt them; adopted arrays must not be mutated afterwards.</param>
    public CsrMatrix(double[] data, int[] indices, int[] indptr, int rows, int columns, bool copy = true)
        : base(rows, columns)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(indptr);
        Validate(data, indices, indptr, rows, columns);
        _data = copy ? (double[])data.Clone() : data;
        _indices = copy ? (int[])indices.Clone() : indices;
        _indptr = copy ? (int[])indptr.Clone() : indptr;
    }

    internal static void Validate(double[] data, int[] indices, int[] indptr, int major, int minor)
    {
        if (indptr.Length != major + 1) throw new ArgumentException($"indptr must have {major + 1} entries, has {indptr.Length}.");
        if (indptr[0] != 0) throw new ArgumentException("indptr must start at 0.");
        if (data.Length != indices.Length) throw new ArgumentException("data and indices must have equal length.");
        if (indptr[major] != data.Length) throw new ArgumentException($"indptr ends at {indptr[major]} but there are {data.Length} stored values.");
        for (var i = 0; i < major; i++)
            if (indptr[i + 1] < indptr[i]) throw new ArgumentException("indptr must be non-decreasing.");
        foreach (var j in indices)
            if ((uint)j >= (uint)minor) throw new ArgumentException($"Index {j} out of range [0, {minor}).");
    }

    /// <summary>An all-zero matrix.</summary>
    public static CsrMatrix Zeros(int rows, int columns) => new(rows, columns, new int[rows + 1], [], [], canonical: true);

    /// <summary>Builds from (row, column, value) coordinates; duplicates are summed.</summary>
    public static CsrMatrix FromTriplets(int rows, int columns, ReadOnlySpan<int> rowIndices, ReadOnlySpan<int> columnIndices, ReadOnlySpan<double> values)
    {
        if (rowIndices.Length != columnIndices.Length || rowIndices.Length != values.Length)
            throw new ArgumentException("Row, column and value arrays must have equal length.");
        if (rows < 0 || columns < 0) throw new ArgumentException($"Invalid shape ({rows}, {columns}).");
        return new CsrMatrix(SparseKernels.FromTriplets(rows, columns, rowIndices, columnIndices, values), canonical: true);
    }

    /// <summary>Builds from a dense rank 2 array, keeping entries with <c>|v| &gt; threshold</c>.</summary>
    public static CsrMatrix FromDense(NdArray dense, double threshold = 0.0)
    {
        if (dense.Rank != 2) throw new ArgumentException("FromDense expects a rank 2 array.");
        int m = dense.Shape[0], n = dense.Shape[1];
        var src = dense.AsContiguous().AsSpan();
        var ptr = new int[m + 1];
        var count = 0;
        for (var k = 0; k < src.Length; k++) if (Math.Abs(src[k]) > threshold || double.IsNaN(src[k])) count++;
        var idx = new int[count];
        var val = new double[count];
        var w = 0;
        for (var i = 0; i < m; i++)
        {
            var row = src.Slice(i * n, n);
            for (var j = 0; j < n; j++)
            {
                var v = row[j];
                if (Math.Abs(v) > threshold || double.IsNaN(v)) { idx[w] = j; val[w++] = v; }
            }
            ptr[i + 1] = w;
        }
        return new CsrMatrix(m, n, ptr, idx, val, canonical: true);
    }

    /// <summary>Builds from a dense 2-D C# array.</summary>
    public static CsrMatrix FromDense(double[,] dense) => FromDense(NdArray.FromArray(dense));

    /// <summary>Converts from GraviNum's CSR <see cref="SparseMatrix"/> (a copy of its three arrays).</summary>
    public static CsrMatrix FromGraviNum(SparseMatrix matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        var a = new CompressedArrays(matrix.Rows, matrix.Columns, matrix.RowPointers.ToArray(), matrix.ColumnIndices.ToArray(), matrix.Values.ToArray());
        return new CsrMatrix(SparseKernels.Canonicalize(a), canonical: true);
    }

    /// <summary>Converts any sparse format to CSR.</summary>
    public static CsrMatrix From(SparseMatrixBase matrix) => matrix.ToCsr();

    internal CompressedArrays Arrays => new(Rows, Columns, _indptr, _indices, _data);

    /// <summary>Stored values, row by row.</summary>
    public ReadOnlySpan<double> Data => _data;

    /// <summary>Column index of each stored value.</summary>
    public ReadOnlySpan<int> Indices => _indices;

    /// <summary>Row start offsets into <see cref="Data"/>; length <c>Rows + 1</c>.</summary>
    public ReadOnlySpan<int> Indptr => _indptr;

    internal int[] IndptrArray => _indptr;
    internal int[] IndicesArray => _indices;
    internal double[] DataArray => _data;

    /// <inheritdoc />
    public override int Nnz => _indptr[Rows];

    /// <inheritdoc />
    public override string Format => "csr";

    /// <summary>True when each row's column indices are non-decreasing.</summary>
    public bool HasSortedIndices
    {
        get
        {
            if (_sorted < 0) _sorted = SparseKernels.IsSorted(Rows, _indptr, _indices) ? 1 : 0;
            return _sorted == 1;
        }
    }

    /// <summary>True when indices are sorted and there are no duplicates (SciPy's <c>has_canonical_format</c>).</summary>
    public bool HasCanonicalFormat
    {
        get
        {
            if (_canonical < 0) _canonical = SparseKernels.IsCanonical(Rows, _indptr, _indices) ? 1 : 0;
            return _canonical == 1;
        }
    }

    /// <inheritdoc />
    public override CsrMatrix ToCsr() => this;

    /// <inheritdoc />
    public override CscMatrix ToCsc()
    {
        // The transpose kernel emits sorted indices with duplicates still separate; canonicalise
        // the source first so the CSC result is canonical too.
        var t = SparseKernels.Transpose(Canonical().Arrays);
        return new CscMatrix(new CsrMatrix(t, canonical: true));
    }

    /// <inheritdoc />
    public override CooMatrix ToCoo()
    {
        var r = new int[Nnz];
        for (var i = 0; i < Rows; i++)
            for (var k = _indptr[i]; k < _indptr[i + 1]; k++) r[k] = i;
        return new CooMatrix(Rows, Columns, r, (int[])_indices.Clone(), (double[])_data.Clone(), HasCanonicalFormat);
    }

    /// <inheritdoc />
    public override NdArray ToDense()
    {
        var buf = new double[Rows * Columns];
        for (var i = 0; i < Rows; i++)
        {
            var off = i * Columns;
            for (var k = _indptr[i]; k < _indptr[i + 1]; k++) buf[off + _indices[k]] += _data[k];
        }
        return new NdArray(buf, Rows, Columns);
    }

    /// <inheritdoc />
    public override double this[int row, int column]
    {
        get
        {
            if ((uint)row >= (uint)Rows || (uint)column >= (uint)Columns)
                throw new IndexOutOfRangeException($"Index ({row}, {column}) out of range for shape ({Rows}, {Columns}).");
            int s = _indptr[row], e = _indptr[row + 1];
            if (HasCanonicalFormat)
            {
                var k = Array.BinarySearch(_indices, s, e - s, column);
                return k >= 0 ? _data[k] : 0.0;
            }
            var sum = 0.0;
            for (var k = s; k < e; k++) if (_indices[k] == column) sum += _data[k];
            return sum;
        }
    }

    /// <inheritdoc />
    public override IEnumerable<(int Row, int Column, double Value)> EnumerateEntries()
    {
        for (var i = 0; i < Rows; i++)
            for (var k = _indptr[i]; k < _indptr[i + 1]; k++)
                yield return (i, _indices[k], _data[k]);
    }

    /// <inheritdoc />
    public override void MatVec(ReadOnlySpan<double> x, Span<double> y)
    {
        if (x.Length != Columns) throw new ArgumentException($"Vector has {x.Length} entries, expected {Columns}.");
        if (y.Length != Rows) throw new ArgumentException($"Output has {y.Length} entries, expected {Rows}.");
        SparseKernels.SpMV(Arrays, x, y);
    }

    /// <inheritdoc />
    public override void RMatVec(ReadOnlySpan<double> x, Span<double> y)
    {
        if (x.Length != Rows) throw new ArgumentException($"Vector has {x.Length} entries, expected {Rows}.");
        if (y.Length != Columns) throw new ArgumentException($"Output has {y.Length} entries, expected {Columns}.");
        SparseKernels.SpMVTransposed(Arrays, x, y);
    }

    /// <summary>
    /// <c>A x</c> by the plain scalar loop — the reference the unrolled, parallel <see cref="MatVec"/>
    /// is tested against. Use <see cref="SparseMatrixBase.Dot(double[])"/> for real work.
    /// </summary>
    public double[] DotReference(ReadOnlySpan<double> x)
    {
        if (x.Length != Columns) throw new ArgumentException($"Vector has {x.Length} entries, expected {Columns}.");
        var y = new double[Rows];
        SparseKernels.SpMVReference(Arrays, x, y);
        return y;
    }

    /// <summary>The transpose as CSC over the same arrays — O(1), nothing is copied.</summary>
    public new CscMatrix Transpose() => new(new CsrMatrix(Rows, Columns, _indptr, _indices, _data, HasCanonicalFormat ? true : null));

    /// <inheritdoc />
    protected override SparseMatrixBase TransposeCore() => Transpose();

    // ------------------------------------------------------------------ products

    /// <summary><c>A B</c> for a dense vector (rank 1) or dense matrix (rank 2).</summary>
    public new NdArray Dot(NdArray b)
    {
        if (b.Rank == 1)
            return new NdArray(Dot(b.AsContiguous().AsSpan()), Rows);
        if (b.Rank != 2 || b.Shape[0] != Columns)
            throw new ArgumentException($"Cannot multiply a {Rows}x{Columns} sparse matrix by an array of shape ({string.Join(", ", b.Shape.ToArray())}).");
        var k = b.Shape[1];
        var c = new double[Rows * k];
        SparseKernels.SpMM(Arrays, b.AsContiguous().ToArray(), k, c);
        return new NdArray(c, Rows, k);
    }

    /// <summary>Sparse–sparse product <c>A B</c> by two-pass parallel Gustavson SpGEMM. The result is canonical.</summary>
    public CsrMatrix MatMul(CsrMatrix b)
    {
        if (Columns != b.Rows) throw new ArgumentException($"Cannot multiply {Rows}x{Columns} by {b.Rows}x{b.Columns}.");
        return new CsrMatrix(SparseKernels.SpGemm(Arrays, b.Arrays), canonical: true);
    }

    // ------------------------------------------------------------------ canonical form and cleanup

    /// <summary>This matrix if canonical, otherwise a sorted, duplicate-summed copy.</summary>
    internal CsrMatrix Canonical() => HasCanonicalFormat ? this : new CsrMatrix(SparseKernels.Canonicalize(Arrays), canonical: true);

    /// <summary>Sorts and sums duplicates (SciPy's <c>sum_duplicates</c>); returns this instance when already canonical.</summary>
    public CsrMatrix SumDuplicates() => Canonical();

    /// <summary>Sorts the column indices of each row (duplicates are kept separate).</summary>
    public CsrMatrix SortIndices()
    {
        if (HasSortedIndices) return this;
        var idx = (int[])_indices.Clone();
        var val = (double[])_data.Clone();
        for (var i = 0; i < Rows; i++)
        {
            int s = _indptr[i], len = _indptr[i + 1] - s;
            if (len > 1) Array.Sort(idx, val, s, len);
        }
        return new CsrMatrix(Rows, Columns, _indptr, idx, val);
    }

    /// <summary>Removes explicitly stored zeros.</summary>
    public CsrMatrix EliminateZeros()
    {
        var any = false;
        foreach (var v in _data) if (v == 0.0) { any = true; break; }
        if (!any) return this;
        return new CsrMatrix(SparseKernels.Filter(Arrays, static (_, _, v) => v == 0.0), HasCanonicalFormat ? true : null);
    }

    /// <summary>Removes stored entries with <c>|v| ≤ tolerance</c>.</summary>
    public CsrMatrix DropSmall(double tolerance)
        => new(SparseKernels.Filter(Arrays, (_, _, v) => Math.Abs(v) <= tolerance), HasCanonicalFormat ? true : null);

    /// <summary>
    /// SciPy's <c>prune</c> trims arrays longer than <c>nnz</c>; arrays here are always exact, so
    /// this returns the same matrix. Kept for API parity.
    /// </summary>
    public CsrMatrix Prune() => this;

    /// <summary>A deep copy (independent arrays).</summary>
    public CsrMatrix Copy() => new(Rows, Columns, (int[])_indptr.Clone(), (int[])_indices.Clone(), (double[])_data.Clone(), HasCanonicalFormat ? true : null);

    // ------------------------------------------------------------------ element-wise

    internal CsrMatrix AddCore(SparseMatrixBase b) => MergeWith<SparseAddOp>(b, union: true);
    internal CsrMatrix SubtractCore(SparseMatrixBase b) => MergeWith<SparseSubOp>(b, union: true);
    internal CsrMatrix MultiplyCore(SparseMatrixBase b) => MergeWith<SparseMulOp>(b, union: false);
    internal CsrMatrix MaximumCore(SparseMatrixBase b) => MergeWith<SparseMaxOp>(b, union: true);
    internal CsrMatrix MinimumCore(SparseMatrixBase b) => MergeWith<SparseMinOp>(b, union: true);

    private CsrMatrix MergeWith<TOp>(SparseMatrixBase other, bool union) where TOp : ISparseBinaryOp
    {
        if (other.Rows != Rows || other.Columns != Columns)
            throw new ArgumentException($"Shape mismatch: ({Rows}, {Columns}) vs ({other.Rows}, {other.Columns}).");
        var b = other.ToCsr().Canonical();
        return new CsrMatrix(SparseKernels.Merge<TOp>(Canonical().Arrays, b.Arrays, union), canonical: true);
    }

    internal CsrMatrix ScaleCore(double alpha)
    {
        var v = new double[_data.Length];
        for (var k = 0; k < v.Length; k++) v[k] = _data[k] * alpha;
        return new CsrMatrix(Rows, Columns, _indptr, _indices, v, HasCanonicalFormat ? true : null);
    }

    internal CsrMatrix MultiplyDense(NdArray dense)
    {
        double[] buf;
        bool rowVector;
        if (dense.Rank == 2 && dense.Shape[0] == Rows && dense.Shape[1] == Columns) { buf = dense.AsContiguous().ToArray(); rowVector = false; }
        else if (dense.Rank == 1 && dense.Shape[0] == Columns) { buf = dense.ToArray(); rowVector = true; }
        else throw new ArgumentException($"Cannot broadcast shape ({string.Join(", ", dense.Shape.ToArray())}) against ({Rows}, {Columns}).");
        var c = Canonical();
        var v = new double[c._data.Length];
        for (var i = 0; i < Rows; i++)
            for (var k = c._indptr[i]; k < c._indptr[i + 1]; k++)
                v[k] = c._data[k] * (rowVector ? buf[c._indices[k]] : buf[i * Columns + c._indices[k]]);
        return new CsrMatrix(Rows, Columns, c._indptr, c._indices, v, canonical: true).EliminateZeros();
    }

    internal NdArray AddDense(NdArray dense)
    {
        if (dense.Rank != 2 || dense.Shape[0] != Rows || dense.Shape[1] != Columns)
            throw new ArgumentException($"Cannot add an array of shape ({string.Join(", ", dense.Shape.ToArray())}) to a ({Rows}, {Columns}) sparse matrix.");
        var buf = dense.AsContiguous().ToArray();
        for (var i = 0; i < Rows; i++)
            for (var k = _indptr[i]; k < _indptr[i + 1]; k++) buf[i * Columns + _indices[k]] += _data[k];
        return new NdArray(buf, Rows, Columns);
    }

    /// <summary>Applies <paramref name="f"/> to every stored value; <paramref name="f"/> must map zero to zero, or the result would be dense.</summary>
    public CsrMatrix Map(Func<double, double> f)
    {
        var c = Canonical();
        var v = new double[c._data.Length];
        for (var k = 0; k < v.Length; k++) v[k] = f(c._data[k]);
        return new CsrMatrix(Rows, Columns, c._indptr, c._indices, v, canonical: true);
    }

    /// <summary>Element-wise power. Non-positive exponents are refused: <c>0^p</c> would densify or be infinite (as in SciPy).</summary>
    public CsrMatrix Power(double exponent)
    {
        if (exponent <= 0) throw new ArgumentException("Power requires a positive exponent; zero or negative powers would densify the matrix.");
        if (exponent == 2.0) return Map(static v => v * v);
        return Map(v => Math.Pow(v, exponent));
    }

    /// <summary>Element-wise absolute value.</summary>
    public CsrMatrix Abs() => Map(Math.Abs);

    /// <summary>Element-wise square root.</summary>
    public CsrMatrix Sqrt() => Map(Math.Sqrt);

    /// <summary>Element-wise sign.</summary>
    public CsrMatrix Sign() => Map(static v => Math.Sign(v));

    // ------------------------------------------------------------------ reductions

    internal double SumAll()
    {
        var s = 0.0;
        foreach (var v in _data) s += v;
        return s;
    }

    internal double[] SumAxis(int axis)
    {
        if (axis == 1)
        {
            var r = new double[Rows];
            for (var i = 0; i < Rows; i++)
            {
                var s = 0.0;
                for (var k = _indptr[i]; k < _indptr[i + 1]; k++) s += _data[k];
                r[i] = s;
            }
            return r;
        }
        if (axis == 0)
        {
            var c = new double[Columns];
            for (var k = 0; k < Nnz; k++) c[_indices[k]] += _data[k];
            return c;
        }
        throw new ArgumentException("axis must be 0 or 1.");
    }

    internal double Extreme(bool max)
    {
        if (Rows == 0 || Columns == 0) throw new InvalidOperationException("Max/Min of an empty matrix.");
        var c = Canonical();
        var best = c.Nnz < (long)Rows * Columns ? 0.0 : (max ? double.NegativeInfinity : double.PositiveInfinity);
        foreach (var v in c._data) best = max ? Math.Max(best, v) : Math.Min(best, v);
        return best;
    }

    internal double[] ExtremeAxis(int axis, bool max)
    {
        if (axis != 0 && axis != 1) throw new ArgumentException("axis must be 0 or 1.");
        var c = axis == 1 ? Canonical() : ToCsc().Inner;
        var major = c.Rows;
        var minor = c.Columns;
        var r = new double[major];
        for (var i = 0; i < major; i++)
        {
            int s = c._indptr[i], e = c._indptr[i + 1];
            var best = e - s < minor ? 0.0 : (max ? double.NegativeInfinity : double.PositiveInfinity);
            for (var k = s; k < e; k++) best = max ? Math.Max(best, c._data[k]) : Math.Min(best, c._data[k]);
            r[i] = minor == 0 ? double.NaN : best;
        }
        return r;
    }

    internal double[] DiagonalCore(int k)
    {
        var len = k >= 0 ? Math.Min(Rows, Columns - k) : Math.Min(Rows + k, Columns);
        if (len <= 0) return [];
        var d = new double[len];
        var r0 = Math.Max(0, -k);
        for (var t = 0; t < len; t++)
        {
            var i = r0 + t;
            var j = i + k;
            for (var p = _indptr[i]; p < _indptr[i + 1]; p++) if (_indices[p] == j) d[t] += _data[p];
        }
        return d;
    }

    /// <summary>
    /// A copy with the <paramref name="k"/>-th diagonal replaced by <paramref name="values"/>
    /// (SciPy's <c>setdiag</c>). A shorter array sets only that many leading entries.
    /// </summary>
    public CsrMatrix SetDiag(ReadOnlySpan<double> values, int k = 0)
    {
        var len = k >= 0 ? Math.Min(Rows, Columns - k) : Math.Min(Rows + k, Columns);
        if (len <= 0) throw new ArgumentException($"Diagonal {k} is out of bounds for shape ({Rows}, {Columns}).");
        len = Math.Min(len, values.Length);
        var r0 = Math.Max(0, -k);
        var kept = SparseKernels.Filter(Canonical().Arrays, (i, j, _) => j - i == k && i - r0 < len && i >= r0);
        var rows = new int[len];
        var cols = new int[len];
        var vals = new double[len];
        for (var t = 0; t < len; t++) { rows[t] = r0 + t; cols[t] = r0 + t + k; vals[t] = values[t]; }
        // Union with SparseAddOp would drop explicit zeros the caller asked for; build by concatenation instead.
        var coo = new CsrMatrix(kept, canonical: true).ToCoo();
        var r = new int[coo.Nnz + len];
        var c = new int[coo.Nnz + len];
        var v = new double[coo.Nnz + len];
        coo.RowArray.CopyTo(r, 0); coo.ColArray.CopyTo(c, 0); coo.DataArray.CopyTo(v, 0);
        rows.CopyTo(r, coo.Nnz); cols.CopyTo(c, coo.Nnz); vals.CopyTo(v, coo.Nnz);
        return FromTriplets(Rows, Columns, r, c, v);
    }

    /// <summary>A copy with every entry of the <paramref name="k"/>-th diagonal set to <paramref name="value"/>.</summary>
    public CsrMatrix SetDiag(double value, int k = 0)
    {
        var len = k >= 0 ? Math.Min(Rows, Columns - k) : Math.Min(Rows + k, Columns);
        if (len <= 0) throw new ArgumentException($"Diagonal {k} is out of bounds for shape ({Rows}, {Columns}).");
        var v = new double[len];
        Array.Fill(v, value);
        return SetDiag(v, k);
    }

    internal double NormCore(SparseNormKind kind)
    {
        switch (kind)
        {
            case SparseNormKind.Frobenius:
            {
                // Scaled sum of squares so huge or tiny entries neither overflow nor underflow.
                var c = Canonical();
                double scale = 0, ssq = 1;
                foreach (var x in c._data)
                {
                    if (x == 0) continue;
                    var ax = Math.Abs(x);
                    if (scale < ax) { ssq = 1 + ssq * (scale / ax) * (scale / ax); scale = ax; }
                    else ssq += (ax / scale) * (ax / scale);
                }
                return scale * Math.Sqrt(ssq);
            }
            case SparseNormKind.One:
            case SparseNormKind.MinusOne:
            {
                var s = Abs().SumAxis(0);
                return s.Length == 0 ? 0 : kind == SparseNormKind.One ? s.Max() : s.Min();
            }
            case SparseNormKind.Infinity:
            case SparseNormKind.MinusInfinity:
            {
                var s = Abs().SumAxis(1);
                return s.Length == 0 ? 0 : kind == SparseNormKind.Infinity ? s.Max() : s.Min();
            }
            default: throw new ArgumentOutOfRangeException(nameof(kind));
        }
    }

    // ------------------------------------------------------------------ structure: triangles and slicing

    /// <summary>A copy without the stored entries for which <paramref name="drop"/>(row, column, value) is true.</summary>
    internal CsrMatrix Filter(Func<int, int, double, bool> drop) => new(SparseKernels.Filter(Arrays, drop), HasCanonicalFormat ? true : null);

    /// <summary>Lower triangle: entries with <c>j − i ≤ k</c>.</summary>
    public CsrMatrix Tril(int k = 0) => new(SparseKernels.Filter(Arrays, (i, j, _) => j - i > k), HasCanonicalFormat ? true : null);

    /// <summary>Upper triangle: entries with <c>j − i ≥ k</c>.</summary>
    public CsrMatrix Triu(int k = 0) => new(SparseKernels.Filter(Arrays, (i, j, _) => j - i < k), HasCanonicalFormat ? true : null);

    /// <summary>Row <paramref name="i"/> as a 1×n CSR matrix.</summary>
    public CsrMatrix GetRow(int i) => GetRows([i]);

    /// <summary>Column <paramref name="j"/> as an m×1 CSR matrix.</summary>
    public CsrMatrix GetColumn(int j) => GetColumns([j]);

    /// <summary>Fancy row indexing: the rows listed (repeats allowed), in that order.</summary>
    public CsrMatrix GetRows(ReadOnlySpan<int> rows)
    {
        var ptr = new int[rows.Length + 1];
        for (var t = 0; t < rows.Length; t++)
        {
            var i = rows[t];
            if ((uint)i >= (uint)Rows) throw new IndexOutOfRangeException($"Row {i} out of range for {Rows} rows.");
            ptr[t + 1] = ptr[t] + _indptr[i + 1] - _indptr[i];
        }
        var idx = new int[ptr[rows.Length]];
        var val = new double[ptr[rows.Length]];
        for (var t = 0; t < rows.Length; t++)
        {
            var i = rows[t];
            Array.Copy(_indices, _indptr[i], idx, ptr[t], ptr[t + 1] - ptr[t]);
            Array.Copy(_data, _indptr[i], val, ptr[t], ptr[t + 1] - ptr[t]);
        }
        return new CsrMatrix(rows.Length, Columns, ptr, idx, val, HasCanonicalFormat ? true : null);
    }

    /// <summary>Fancy column indexing: the columns listed (repeats allowed), in that order.</summary>
    public CsrMatrix GetColumns(ReadOnlySpan<int> columns)
    {
        // Column j may be requested several times: map each source column to its output slots.
        var first = new int[Columns];
        Array.Fill(first, -1);
        var nextSlot = new int[columns.Length];
        Array.Fill(nextSlot, -1);
        for (var t = columns.Length - 1; t >= 0; t--)
        {
            var j = columns[t];
            if ((uint)j >= (uint)Columns) throw new IndexOutOfRangeException($"Column {j} out of range for {Columns} columns.");
            nextSlot[t] = first[j];
            first[j] = t;
        }
        var ptr = new int[Rows + 1];
        for (var i = 0; i < Rows; i++)
        {
            var count = 0;
            for (var k = _indptr[i]; k < _indptr[i + 1]; k++)
                for (var t = first[_indices[k]]; t >= 0; t = nextSlot[t]) count++;
            ptr[i + 1] = ptr[i] + count;
        }
        var idx = new int[ptr[Rows]];
        var val = new double[ptr[Rows]];
        for (var i = 0; i < Rows; i++)
        {
            var w = ptr[i];
            for (var k = _indptr[i]; k < _indptr[i + 1]; k++)
                for (var t = first[_indices[k]]; t >= 0; t = nextSlot[t]) { idx[w] = t; val[w++] = _data[k]; }
        }
        return new CsrMatrix(SparseKernels.Canonicalize(new CompressedArrays(Rows, columns.Length, ptr, idx, val)), canonical: true);
    }

    /// <summary>Rows <c>[rowStart, rowStop)</c> × columns <c>[colStart, colStop)</c>, like <c>A[r0:r1, c0:c1]</c>.</summary>
    public CsrMatrix Slice(int rowStart, int rowStop, int colStart, int colStop)
    {
        if (rowStart < 0 || rowStop > Rows || rowStart > rowStop || colStart < 0 || colStop > Columns || colStart > colStop)
            throw new ArgumentOutOfRangeException(nameof(rowStart), $"Slice [{rowStart}:{rowStop}, {colStart}:{colStop}] out of range for ({Rows}, {Columns}).");
        var m = rowStop - rowStart;
        var ptr = new int[m + 1];
        for (var t = 0; t < m; t++)
        {
            var i = rowStart + t;
            var c = 0;
            for (var k = _indptr[i]; k < _indptr[i + 1]; k++) if (_indices[k] >= colStart && _indices[k] < colStop) c++;
            ptr[t + 1] = ptr[t] + c;
        }
        var idx = new int[ptr[m]];
        var val = new double[ptr[m]];
        for (var t = 0; t < m; t++)
        {
            var i = rowStart + t;
            var w = ptr[t];
            for (var k = _indptr[i]; k < _indptr[i + 1]; k++)
                if (_indices[k] >= colStart && _indices[k] < colStop) { idx[w] = _indices[k] - colStart; val[w++] = _data[k]; }
        }
        return new CsrMatrix(m, colStop - colStart, ptr, idx, val, HasCanonicalFormat ? true : null);
    }

    /// <summary>The submatrix <c>A[np.ix_(rows, columns)]</c>.</summary>
    public CsrMatrix Submatrix(ReadOnlySpan<int> rows, ReadOnlySpan<int> columns) => GetRows(rows).GetColumns(columns);

    /// <summary>Row and column indices plus values of the non-zero entries (SciPy's <c>find</c>), explicit zeros excluded.</summary>
    public (int[] Rows, int[] Columns, double[] Values) Find()
    {
        var c = Canonical();
        var r = new List<int>(); var col = new List<int>(); var v = new List<double>();
        for (var i = 0; i < Rows; i++)
            for (var k = c._indptr[i]; k < c._indptr[i + 1]; k++)
                if (c._data[k] != 0.0) { r.Add(i); col.Add(c._indices[k]); v.Add(c._data[k]); }
        return (r.ToArray(), col.ToArray(), v.ToArray());
    }

    /// <summary>The non-zero count of each row.</summary>
    public int[] RowNnz()
    {
        var r = new int[Rows];
        for (var i = 0; i < Rows; i++) r[i] = _indptr[i + 1] - _indptr[i];
        return r;
    }
}
