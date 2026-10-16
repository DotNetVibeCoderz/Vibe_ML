using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Sparse;

/// <summary>
/// Compressed sparse column matrix — <c>scipy.sparse.csc_matrix</c>. Fast column slicing and
/// the natural input of the sparse LU (<see cref="SparseLinalg.Splu"/>), which works column by column.
/// </summary>
/// <remarks>
/// A CSC matrix stores exactly the arrays of the CSR form of its transpose, so it is implemented as
/// one: <see cref="Transpose"/> and <see cref="CsrMatrix.Transpose"/> are O(1) reinterpretations.
/// Column <c>j</c>'s entries are <c>Data[Indptr[j]..Indptr[j+1]]</c> at rows <c>Indices[...]</c>.
/// </remarks>
public sealed class CscMatrix : SparseMatrixBase
{
    /// <summary>The CSR structure of the transpose — the same three arrays.</summary>
    internal CsrMatrix Inner { get; }

    internal CscMatrix(CsrMatrix transposeAsCsr) : base(transposeAsCsr.Columns, transposeAsCsr.Rows)
    {
        Inner = transposeAsCsr;
    }

    /// <summary>
    /// Builds a CSC matrix from its three arrays (SciPy's <c>csc_matrix((data, indices, indptr), shape)</c>).
    /// </summary>
    /// <param name="data">Stored values, column by column.</param>
    /// <param name="indices">Row of each stored value.</param>
    /// <param name="indptr">Column start offsets, length <c>columns + 1</c>.</param>
    /// <param name="rows">Number of rows.</param>
    /// <param name="columns">Number of columns.</param>
    /// <param name="copy">Copy the arrays (default) or adopt them.</param>
    public CscMatrix(double[] data, int[] indices, int[] indptr, int rows, int columns, bool copy = true)
        : this(new CsrMatrix(data, indices, indptr, columns, rows, copy))
    {
    }

    /// <summary>Builds from (row, column, value) coordinates; duplicates are summed.</summary>
    public static CscMatrix FromTriplets(int rows, int columns, ReadOnlySpan<int> rowIndices, ReadOnlySpan<int> columnIndices, ReadOnlySpan<double> values)
        => new(CsrMatrix.FromTriplets(columns, rows, columnIndices, rowIndices, values));

    /// <summary>Builds from a dense rank 2 array.</summary>
    public static CscMatrix FromDense(NdArray dense, double threshold = 0.0) => CsrMatrix.FromDense(dense, threshold).ToCsc();

    /// <summary>Converts from GraviNum's CSR <see cref="SparseMatrix"/>.</summary>
    public static CscMatrix FromGraviNum(SparseMatrix matrix) => CsrMatrix.FromGraviNum(matrix).ToCsc();

    /// <summary>Stored values, column by column.</summary>
    public ReadOnlySpan<double> Data => Inner.Data;

    /// <summary>Row index of each stored value.</summary>
    public ReadOnlySpan<int> Indices => Inner.Indices;

    /// <summary>Column start offsets into <see cref="Data"/>; length <c>Columns + 1</c>.</summary>
    public ReadOnlySpan<int> Indptr => Inner.Indptr;

    /// <inheritdoc />
    public override int Nnz => Inner.Nnz;

    /// <inheritdoc />
    public override string Format => "csc";

    /// <summary>True when each column's row indices are non-decreasing.</summary>
    public bool HasSortedIndices => Inner.HasSortedIndices;

    /// <summary>True when indices are sorted with no duplicates.</summary>
    public bool HasCanonicalFormat => Inner.HasCanonicalFormat;

    /// <inheritdoc />
    public override CsrMatrix ToCsr() => Inner.ToCsc().Inner;

    /// <inheritdoc />
    public override CscMatrix ToCsc() => this;

    /// <inheritdoc />
    public override CooMatrix ToCoo()
    {
        var t = Inner.ToCoo();
        return new CooMatrix(Rows, Columns, t.ColArray, t.RowArray, t.DataArray, false);
    }

    /// <inheritdoc />
    public override NdArray ToDense() => ToCsr().ToDense();

    /// <inheritdoc />
    public override double this[int row, int column] => Inner[column, row];

    /// <inheritdoc />
    public override IEnumerable<(int Row, int Column, double Value)> EnumerateEntries()
    {
        foreach (var (j, i, v) in Inner.EnumerateEntries()) yield return (i, j, v);
    }

    /// <inheritdoc />
    public override void MatVec(ReadOnlySpan<double> x, Span<double> y) => Inner.RMatVec(x, y);

    /// <inheritdoc />
    public override void RMatVec(ReadOnlySpan<double> x, Span<double> y) => Inner.MatVec(x, y);

    /// <summary>The transpose as CSR over the same arrays — O(1).</summary>
    public new CsrMatrix Transpose() => Inner;

    /// <inheritdoc />
    protected override SparseMatrixBase TransposeCore() => Inner;

    /// <summary>Sorts and sums duplicates.</summary>
    public CscMatrix SumDuplicates() => Inner.HasCanonicalFormat ? this : new CscMatrix(Inner.SumDuplicates());

    /// <summary>Sorts each column's row indices.</summary>
    public CscMatrix SortIndices() => Inner.HasSortedIndices ? this : new CscMatrix(Inner.SortIndices());

    /// <summary>Removes explicitly stored zeros.</summary>
    public CscMatrix EliminateZeros() => new(Inner.EliminateZeros());

    /// <summary>A deep copy.</summary>
    public CscMatrix Copy() => new(Inner.Copy());

    /// <summary>Column <paramref name="j"/> as an m×1 CSC matrix.</summary>
    public CscMatrix GetColumn(int j) => new(Inner.GetRows([j]));

    /// <summary>Fancy column indexing.</summary>
    public CscMatrix GetColumns(ReadOnlySpan<int> columns) => new(Inner.GetRows(columns));

    /// <summary>Fancy row indexing.</summary>
    public CscMatrix GetRows(ReadOnlySpan<int> rows) => new(Inner.GetColumns(rows));

    /// <summary>Rows <c>[rowStart, rowStop)</c> × columns <c>[colStart, colStop)</c>.</summary>
    public CscMatrix Slice(int rowStart, int rowStop, int colStart, int colStop) => new(Inner.Slice(colStart, colStop, rowStart, rowStop));
}
