using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Sparse;

/// <summary>Which matrix norm <see cref="SparseMatrixBase.Norm"/> computes.</summary>
public enum SparseNormKind
{
    /// <summary>Frobenius norm: square root of the sum of squared entries.</summary>
    Frobenius,

    /// <summary>Induced 1-norm: largest absolute column sum.</summary>
    One,

    /// <summary>Induced ∞-norm: largest absolute row sum.</summary>
    Infinity,

    /// <summary>Smallest absolute column sum (SciPy's <c>ord=-1</c>).</summary>
    MinusOne,

    /// <summary>Smallest absolute row sum (SciPy's <c>ord=-inf</c>).</summary>
    MinusInfinity,
}

/// <summary>
/// The common surface of every sparse format: shape, conversions, products and the arithmetic that
/// SciPy's <c>spmatrix</c> exposes. Operations not specific to a format go through CSR.
/// </summary>
/// <remarks>
/// <para>CSR, CSC, COO and DIA matrices are immutable: every operation returns a new matrix, and
/// matrices may share their index arrays. That makes them safe to read from many threads, and is
/// why <c>SortIndices</c>, <c>SumDuplicates</c> and friends return a result rather than working in
/// place as SciPy's do. <see cref="DokMatrix"/> and <see cref="LilMatrix"/> are the mutable builders.</para>
/// <para><c>*</c> between two sparse matrices, or a sparse matrix and a vector, is the matrix
/// product (SciPy's <c>spmatrix</c> semantics, <c>@</c>); the element-wise product is
/// <see cref="Multiply(SparseMatrixBase)"/>.</para>
/// </remarks>
public abstract class SparseMatrixBase : ILinearOperator
{
    /// <summary>Initialises the shape.</summary>
    protected SparseMatrixBase(int rows, int columns)
    {
        if (rows < 0 || columns < 0) throw new ArgumentException($"Invalid shape ({rows}, {columns}).");
        Rows = rows;
        Columns = columns;
    }

    /// <summary>Number of rows.</summary>
    public int Rows { get; }

    /// <summary>Number of columns.</summary>
    public int Columns { get; }

    /// <summary>The shape as a (rows, columns) pair.</summary>
    public (int Rows, int Columns) Shape => (Rows, Columns);

    /// <summary>Number of stored entries, including explicit zeros and (for COO) duplicates.</summary>
    public abstract int Nnz { get; }

    /// <summary>SciPy's format name: <c>"csr"</c>, <c>"csc"</c>, <c>"coo"</c>, <c>"dok"</c>, <c>"lil"</c> or <c>"dia"</c>.</summary>
    public abstract string Format { get; }

    /// <summary>Converts to CSR (returns the same instance for a CSR matrix).</summary>
    public abstract CsrMatrix ToCsr();

    /// <summary>Converts to CSC.</summary>
    public virtual CscMatrix ToCsc() => ToCsr().ToCsc();

    /// <summary>Converts to COO.</summary>
    public virtual CooMatrix ToCoo() => ToCsr().ToCoo();

    /// <summary>Converts to a dictionary-of-keys builder (a mutable copy).</summary>
    public DokMatrix ToDok() => DokMatrix.FromCsr(ToCsr());

    /// <summary>Converts to a list-of-lists builder (a mutable copy).</summary>
    public LilMatrix ToLil() => LilMatrix.FromCsr(ToCsr());

    /// <summary>Converts to diagonal storage. Wasteful unless the non-zeros sit on a few diagonals.</summary>
    public DiaMatrix ToDia() => DiaMatrix.FromCsr(ToCsr());

    /// <summary>Expands to a dense rank 2 <see cref="NdArray"/>.</summary>
    public virtual NdArray ToDense() => ToCsr().ToDense();

    /// <summary>Converts to GraviNum's CSR <see cref="SparseMatrix"/> (explicit zeros are dropped, as its builder does).</summary>
    public SparseMatrix ToGraviNum()
    {
        var csr = ToCsr();
        return SparseMatrix.FromTriplets(Rows, Columns, csr.EnumerateEntries());
    }

    /// <summary>Converts to GraviNum's CSR <see cref="SparseMatrix"/>; same as <see cref="ToGraviNum"/>.</summary>
    public SparseMatrix ToSparseMatrix() => ToGraviNum();

    /// <inheritdoc />
    public abstract void MatVec(ReadOnlySpan<double> x, Span<double> y);

    /// <inheritdoc />
    public abstract void RMatVec(ReadOnlySpan<double> x, Span<double> y);

    /// <summary>The transpose. CSR↔CSC transposes are O(1): they reinterpret the same arrays.</summary>
    public SparseMatrixBase Transpose() => TransposeCore();

    /// <summary>The transpose (see <see cref="Transpose"/>).</summary>
    public SparseMatrixBase T => TransposeCore();

    /// <summary>Format-specific transpose.</summary>
    protected abstract SparseMatrixBase TransposeCore();

    /// <summary>Reads one entry; absent entries read as zero.</summary>
    public virtual double this[int row, int column] => ToCsr()[row, column];

    /// <summary>Iterates the stored entries as (row, column, value) in format order.</summary>
    public virtual IEnumerable<(int Row, int Column, double Value)> EnumerateEntries() => ToCsr().EnumerateEntries();

    // ------------------------------------------------------------------ products

    /// <summary><c>A x</c> for a dense vector.</summary>
    public double[] Dot(ReadOnlySpan<double> x)
    {
        if (x.Length != Columns) throw new ArgumentException($"Vector has {x.Length} entries, expected {Columns}.");
        var y = new double[Rows];
        MatVec(x, y);
        return y;
    }

    /// <summary><c>A x</c> for a dense vector.</summary>
    public double[] Dot(double[] x) => Dot((ReadOnlySpan<double>)x);

    /// <summary><c>A B</c> for a dense vector (rank 1) or matrix (rank 2); the result is dense.</summary>
    public NdArray Dot(NdArray b) => ToCsr().Dot(b);

    /// <summary>Sparse–sparse matrix product <c>A B</c> (Gustavson SpGEMM), as CSR.</summary>
    public CsrMatrix Dot(SparseMatrixBase b) => ToCsr().MatMul(b.ToCsr());

    // ------------------------------------------------------------------ element-wise algebra (via CSR)

    /// <summary><c>A + B</c>.</summary>
    public CsrMatrix Add(SparseMatrixBase b) => ToCsr().AddCore(b);

    /// <summary><c>A − B</c>.</summary>
    public CsrMatrix Subtract(SparseMatrixBase b) => ToCsr().SubtractCore(b);

    /// <summary>Element-wise (Hadamard) product with another sparse matrix; only shared positions survive.</summary>
    public CsrMatrix Multiply(SparseMatrixBase b) => ToCsr().MultiplyCore(b);

    /// <summary>Element-wise product with a dense array (broadcast rank 1 as a row, like NumPy); sparse result.</summary>
    public CsrMatrix Multiply(NdArray dense) => ToCsr().MultiplyDense(dense);

    /// <summary>Element-wise maximum, implicit zeros included.</summary>
    public CsrMatrix Maximum(SparseMatrixBase b) => ToCsr().MaximumCore(b);

    /// <summary>Element-wise minimum, implicit zeros included.</summary>
    public CsrMatrix Minimum(SparseMatrixBase b) => ToCsr().MinimumCore(b);

    /// <summary>Multiplies every entry by <paramref name="alpha"/>.</summary>
    public CsrMatrix Scale(double alpha) => ToCsr().ScaleCore(alpha);

    /// <summary>Sparse plus sparse.</summary>
    public static CsrMatrix operator +(SparseMatrixBase a, SparseMatrixBase b) => a.Add(b);

    /// <summary>Sparse minus sparse.</summary>
    public static CsrMatrix operator -(SparseMatrixBase a, SparseMatrixBase b) => a.Subtract(b);

    /// <summary>Negation.</summary>
    public static CsrMatrix operator -(SparseMatrixBase a) => a.Scale(-1.0);

    /// <summary>Scaling.</summary>
    public static CsrMatrix operator *(double alpha, SparseMatrixBase a) => a.Scale(alpha);

    /// <summary>Scaling.</summary>
    public static CsrMatrix operator *(SparseMatrixBase a, double alpha) => a.Scale(alpha);

    /// <summary>Division by a scalar.</summary>
    public static CsrMatrix operator /(SparseMatrixBase a, double alpha) => a.Scale(1.0 / alpha);

    /// <summary>Matrix product (not element-wise; see <see cref="Multiply(SparseMatrixBase)"/>).</summary>
    public static CsrMatrix operator *(SparseMatrixBase a, SparseMatrixBase b) => a.Dot(b);

    /// <summary>Matrix–vector product.</summary>
    public static double[] operator *(SparseMatrixBase a, double[] x) => a.Dot(x);

    /// <summary>Matrix times a dense vector or matrix.</summary>
    public static NdArray operator *(SparseMatrixBase a, NdArray b) => a.Dot(b);

    /// <summary>Sparse plus dense is dense, as in SciPy.</summary>
    public static NdArray operator +(SparseMatrixBase a, NdArray b) => a.ToCsr().AddDense(b);

    /// <summary>Dense plus sparse is dense.</summary>
    public static NdArray operator +(NdArray b, SparseMatrixBase a) => a.ToCsr().AddDense(b);

    // ------------------------------------------------------------------ reductions and queries (via CSR)

    /// <summary>Sum of all entries.</summary>
    public double Sum() => ToCsr().SumAll();

    /// <summary>Sum along an axis: 0 gives one value per column, 1 one value per row.</summary>
    public double[] Sum(int axis) => ToCsr().SumAxis(axis);

    /// <summary>Mean of all <c>rows × columns</c> entries (implicit zeros count).</summary>
    public double Mean() => Rows * (long)Columns == 0 ? double.NaN : Sum() / ((double)Rows * Columns);

    /// <summary>Mean along an axis (implicit zeros count).</summary>
    public double[] Mean(int axis)
    {
        var s = Sum(axis);
        var len = axis == 0 ? Rows : Columns;
        for (var i = 0; i < s.Length; i++) s[i] /= len;
        return s;
    }

    /// <summary>Largest entry, implicit zeros included (SciPy's <c>max()</c>).</summary>
    public double Max() => ToCsr().Extreme(max: true);

    /// <summary>Smallest entry, implicit zeros included.</summary>
    public double Min() => ToCsr().Extreme(max: false);

    /// <summary>Largest entry along an axis, implicit zeros included.</summary>
    public double[] Max(int axis) => ToCsr().ExtremeAxis(axis, max: true);

    /// <summary>Smallest entry along an axis, implicit zeros included.</summary>
    public double[] Min(int axis) => ToCsr().ExtremeAxis(axis, max: false);

    /// <summary>The <paramref name="k"/>-th diagonal (k &gt; 0 above the main diagonal).</summary>
    public double[] Diagonal(int k = 0) => ToCsr().DiagonalCore(k);

    /// <summary>Sum of the <paramref name="offset"/>-th diagonal.</summary>
    public double Trace(int offset = 0)
    {
        var d = Diagonal(offset);
        var s = 0.0;
        foreach (var v in d) s += v;
        return s;
    }

    /// <summary>Number of stored entries that are actually non-zero (SciPy's <c>count_nonzero</c>).</summary>
    public int CountNonzero()
    {
        var csr = ToCsr();
        var c = 0;
        foreach (var v in csr.Data) if (v != 0.0) c++;
        return c;
    }

    /// <summary>Row and column indices of the non-zero entries (explicit zeros excluded), row-major order.</summary>
    public (int[] Rows, int[] Columns) Nonzero()
    {
        var csr = ToCsr().SumDuplicates();
        var r = new List<int>();
        var c = new List<int>();
        foreach (var (i, j, v) in csr.EnumerateEntries())
            if (v != 0.0) { r.Add(i); c.Add(j); }
        return (r.ToArray(), c.ToArray());
    }

    /// <summary>Matrix norm; see <see cref="SparseNormKind"/>.</summary>
    public double Norm(SparseNormKind kind = SparseNormKind.Frobenius) => ToCsr().NormCore(kind);

    /// <summary>
    /// Value equality within tolerance: <c>|a − b| ≤ atol + rtol·|b|</c> at every position (both
    /// patterns visited), regardless of format or stored zeros.
    /// </summary>
    public bool AllClose(SparseMatrixBase other, double rtol = 1e-5, double atol = 1e-8)
    {
        if (Rows != other.Rows || Columns != other.Columns) return false;
        var a = ToCsr().SumDuplicates();
        var b = other.ToCsr().SumDuplicates();
        foreach (var (i, j, v) in a.EnumerateEntries())
            if (!(Math.Abs(v - b[i, j]) <= atol + rtol * Math.Abs(b[i, j]))) return false;
        foreach (var (i, j, v) in b.EnumerateEntries())
            if (!(Math.Abs(a[i, j] - v) <= atol + rtol * Math.Abs(v))) return false;
        return true;
    }

    /// <summary>Exact value equality (same shape, same values everywhere; stored zeros don't matter).</summary>
    public bool ValueEquals(SparseMatrixBase other) => AllClose(other, 0.0, 0.0);

    /// <inheritdoc />
    public override string ToString() => $"<{Rows}x{Columns} sparse matrix with {Nnz} stored elements in {Format.ToUpperInvariant()} format>";
}
