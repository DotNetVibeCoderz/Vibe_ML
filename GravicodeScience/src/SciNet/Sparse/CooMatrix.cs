using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Sparse;

/// <summary>
/// Coordinate-format matrix — <c>scipy.sparse.coo_matrix</c>: three parallel arrays of row, column
/// and value. The interchange format: cheap to build, duplicates allowed (they add up on
/// conversion), but slow to index. Convert to CSR or CSC for arithmetic.
/// </summary>
public sealed class CooMatrix : SparseMatrixBase
{
    private readonly int[] _row;
    private readonly int[] _col;
    private readonly double[] _data;
    private int _canonical;

    internal CooMatrix(int rows, int columns, int[] row, int[] col, double[] data, bool canonical) : base(rows, columns)
    {
        _row = row;
        _col = col;
        _data = data;
        _canonical = canonical ? 1 : -1;
    }

    /// <summary>Builds from coordinate arrays (copied). Duplicates are allowed and kept until conversion.</summary>
    public CooMatrix(double[] data, int[] row, int[] col, int rows, int columns) : base(rows, columns)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(row);
        ArgumentNullException.ThrowIfNull(col);
        if (data.Length != row.Length || data.Length != col.Length) throw new ArgumentException("row, col and data must have equal length.");
        for (var k = 0; k < data.Length; k++)
        {
            if ((uint)row[k] >= (uint)rows) throw new ArgumentException($"Row index {row[k]} out of range for {rows} rows.");
            if ((uint)col[k] >= (uint)columns) throw new ArgumentException($"Column index {col[k]} out of range for {columns} columns.");
        }
        _row = (int[])row.Clone();
        _col = (int[])col.Clone();
        _data = (double[])data.Clone();
        _canonical = -1;
    }

    /// <summary>Builds from a dense rank 2 array.</summary>
    public static CooMatrix FromDense(NdArray dense, double threshold = 0.0) => CsrMatrix.FromDense(dense, threshold).ToCoo();

    /// <summary>Row index of each stored value.</summary>
    public ReadOnlySpan<int> Row => _row;

    /// <summary>Column index of each stored value.</summary>
    public ReadOnlySpan<int> Col => _col;

    /// <summary>Stored values.</summary>
    public ReadOnlySpan<double> Data => _data;

    internal int[] RowArray => _row;
    internal int[] ColArray => _col;
    internal double[] DataArray => _data;

    /// <inheritdoc />
    public override int Nnz => _data.Length;

    /// <inheritdoc />
    public override string Format => "coo";

    /// <summary>True when entries are in row-major order with no duplicate coordinates.</summary>
    public bool HasCanonicalFormat
    {
        get
        {
            if (_canonical < 0)
            {
                var ok = true;
                for (var k = 1; k < _data.Length && ok; k++)
                    ok = _row[k] > _row[k - 1] || (_row[k] == _row[k - 1] && _col[k] > _col[k - 1]);
                _canonical = ok ? 1 : 0;
            }
            return _canonical == 1;
        }
    }

    /// <inheritdoc />
    public override CsrMatrix ToCsr() => new(SparseKernels.FromTriplets(Rows, Columns, _row, _col, _data), canonical: true);

    /// <inheritdoc />
    public override CscMatrix ToCsc() => new(new CsrMatrix(SparseKernels.FromTriplets(Columns, Rows, _col, _row, _data), canonical: true));

    /// <inheritdoc />
    public override CooMatrix ToCoo() => this;

    /// <summary>Row-major ordered copy with duplicates summed.</summary>
    public CooMatrix SumDuplicates() => HasCanonicalFormat ? this : ToCsr().ToCoo();

    /// <inheritdoc />
    public override NdArray ToDense()
    {
        var buf = new double[Rows * Columns];
        for (var k = 0; k < _data.Length; k++) buf[_row[k] * Columns + _col[k]] += _data[k];
        return new NdArray(buf, Rows, Columns);
    }

    /// <inheritdoc />
    public override double this[int row, int column]
    {
        get
        {
            var s = 0.0;
            for (var k = 0; k < _data.Length; k++) if (_row[k] == row && _col[k] == column) s += _data[k];
            return s;
        }
    }

    /// <inheritdoc />
    public override IEnumerable<(int Row, int Column, double Value)> EnumerateEntries()
    {
        for (var k = 0; k < _data.Length; k++) yield return (_row[k], _col[k], _data[k]);
    }

    /// <inheritdoc />
    public override void MatVec(ReadOnlySpan<double> x, Span<double> y)
    {
        if (x.Length != Columns || y.Length != Rows) throw new ArgumentException("Vector length does not match the matrix shape.");
        y.Clear();
        for (var k = 0; k < _data.Length; k++) y[_row[k]] += _data[k] * x[_col[k]];
    }

    /// <inheritdoc />
    public override void RMatVec(ReadOnlySpan<double> x, Span<double> y)
    {
        if (x.Length != Rows || y.Length != Columns) throw new ArgumentException("Vector length does not match the matrix shape.");
        y.Clear();
        for (var k = 0; k < _data.Length; k++) y[_col[k]] += _data[k] * x[_row[k]];
    }

    /// <summary>The transpose: the row and column arrays swap places (O(1), shared arrays).</summary>
    public new CooMatrix Transpose() => new(Columns, Rows, _col, _row, _data, false);

    /// <inheritdoc />
    protected override SparseMatrixBase TransposeCore() => Transpose();

    /// <summary>Removes explicitly stored zeros.</summary>
    public CooMatrix EliminateZeros()
    {
        var keep = 0;
        foreach (var v in _data) if (v != 0.0) keep++;
        var r = new int[keep]; var c = new int[keep]; var d = new double[keep];
        var w = 0;
        for (var k = 0; k < _data.Length; k++)
            if (_data[k] != 0.0) { r[w] = _row[k]; c[w] = _col[k]; d[w++] = _data[k]; }
        return new CooMatrix(Rows, Columns, r, c, d, _canonical == 1);
    }
}
