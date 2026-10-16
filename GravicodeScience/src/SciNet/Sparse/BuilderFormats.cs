using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Sparse;

/// <summary>
/// Dictionary-of-keys matrix — <c>scipy.sparse.dok_matrix</c>. A mutable builder with O(1)
/// random reads and writes; convert to CSR/CSC once building is finished.
/// </summary>
/// <remarks>Assigning zero removes the entry, as in SciPy, so a DOK never stores explicit zeros.</remarks>
public sealed class DokMatrix : SparseMatrixBase
{
    private readonly Dictionary<long, double> _entries = [];

    /// <summary>An empty matrix of the given shape.</summary>
    public DokMatrix(int rows, int columns) : base(rows, columns) { }

    internal static DokMatrix FromCsr(CsrMatrix csr)
    {
        var d = new DokMatrix(csr.Rows, csr.Columns);
        foreach (var (i, j, v) in csr.EnumerateEntries()) d[i, j] += v;
        return d;
    }

    private long Key(int row, int column)
    {
        if ((uint)row >= (uint)Rows || (uint)column >= (uint)Columns)
            throw new IndexOutOfRangeException($"Index ({row}, {column}) out of range for shape ({Rows}, {Columns}).");
        return (long)row * Columns + column;
    }

    /// <summary>Reads or writes one entry; writing zero deletes it.</summary>
    public new double this[int row, int column]
    {
        get => _entries.TryGetValue(Key(row, column), out var v) ? v : 0.0;
        set
        {
            var key = Key(row, column);
            if (value == 0.0) _entries.Remove(key);
            else _entries[key] = value;
        }
    }

    /// <summary>True when an entry is stored at (row, column).</summary>
    public bool ContainsKey(int row, int column) => _entries.ContainsKey(Key(row, column));

    /// <summary>Removes every entry.</summary>
    public void Clear() => _entries.Clear();

    /// <inheritdoc />
    public override int Nnz => _entries.Count;

    /// <inheritdoc />
    public override string Format => "dok";

    /// <inheritdoc />
    public override CsrMatrix ToCsr()
    {
        var n = _entries.Count;
        var r = new int[n]; var c = new int[n]; var v = new double[n];
        var k = 0;
        foreach (var (key, value) in _entries)
        {
            r[k] = (int)(key / Columns);
            c[k] = (int)(key % Columns);
            v[k++] = value;
        }
        return CsrMatrix.FromTriplets(Rows, Columns, r, c, v);
    }

    /// <inheritdoc />
    public override IEnumerable<(int Row, int Column, double Value)> EnumerateEntries()
    {
        foreach (var (key, value) in _entries) yield return ((int)(key / Columns), (int)(key % Columns), value);
    }

    /// <inheritdoc />
    public override void MatVec(ReadOnlySpan<double> x, Span<double> y)
    {
        if (x.Length != Columns || y.Length != Rows) throw new ArgumentException("Vector length does not match the matrix shape.");
        y.Clear();
        foreach (var (key, v) in _entries) y[(int)(key / Columns)] += v * x[(int)(key % Columns)];
    }

    /// <inheritdoc />
    public override void RMatVec(ReadOnlySpan<double> x, Span<double> y)
    {
        if (x.Length != Rows || y.Length != Columns) throw new ArgumentException("Vector length does not match the matrix shape.");
        y.Clear();
        foreach (var (key, v) in _entries) y[(int)(key % Columns)] += v * x[(int)(key / Columns)];
    }

    /// <inheritdoc />
    protected override SparseMatrixBase TransposeCore()
    {
        var t = new DokMatrix(Columns, Rows);
        foreach (var (i, j, v) in EnumerateEntries()) t[j, i] = v;
        return t;
    }
}

/// <summary>
/// List-of-lists matrix — <c>scipy.sparse.lil_matrix</c>: one sorted (column, value) list per row.
/// A mutable builder that is efficient when filled row by row or column-sorted within rows.
/// </summary>
public sealed class LilMatrix : SparseMatrixBase
{
    private readonly List<int>[] _cols;
    private readonly List<double>[] _vals;

    /// <summary>An empty matrix of the given shape.</summary>
    public LilMatrix(int rows, int columns) : base(rows, columns)
    {
        _cols = new List<int>[rows];
        _vals = new List<double>[rows];
        for (var i = 0; i < rows; i++) { _cols[i] = []; _vals[i] = []; }
    }

    internal static LilMatrix FromCsr(CsrMatrix csr)
    {
        var c = csr.SumDuplicates();
        var l = new LilMatrix(c.Rows, c.Columns);
        for (var i = 0; i < c.Rows; i++)
            for (var k = c.Indptr[i]; k < c.Indptr[i + 1]; k++)
            {
                l._cols[i].Add(c.Indices[k]);
                l._vals[i].Add(c.Data[k]);
            }
        return l;
    }

    /// <summary>Reads or writes one entry; writing zero deletes it.</summary>
    public new double this[int row, int column]
    {
        get
        {
            Check(row, column);
            var k = _cols[row].BinarySearch(column);
            return k >= 0 ? _vals[row][k] : 0.0;
        }
        set
        {
            Check(row, column);
            var k = _cols[row].BinarySearch(column);
            if (k >= 0)
            {
                if (value == 0.0) { _cols[row].RemoveAt(k); _vals[row].RemoveAt(k); }
                else _vals[row][k] = value;
            }
            else if (value != 0.0)
            {
                _cols[row].Insert(~k, column);
                _vals[row].Insert(~k, value);
            }
        }
    }

    private void Check(int row, int column)
    {
        if ((uint)row >= (uint)Rows || (uint)column >= (uint)Columns)
            throw new IndexOutOfRangeException($"Index ({row}, {column}) out of range for shape ({Rows}, {Columns}).");
    }

    /// <summary>The sorted column indices stored in row <paramref name="row"/> (SciPy's <c>rows[i]</c>).</summary>
    public IReadOnlyList<int> RowIndices(int row) => _cols[row];

    /// <summary>The values stored in row <paramref name="row"/> (SciPy's <c>data[i]</c>).</summary>
    public IReadOnlyList<double> RowValues(int row) => _vals[row];

    /// <inheritdoc />
    public override int Nnz
    {
        get
        {
            var n = 0;
            foreach (var c in _cols) n += c.Count;
            return n;
        }
    }

    /// <inheritdoc />
    public override string Format => "lil";

    /// <inheritdoc />
    public override CsrMatrix ToCsr()
    {
        var ptr = new int[Rows + 1];
        for (var i = 0; i < Rows; i++) ptr[i + 1] = ptr[i] + _cols[i].Count;
        var idx = new int[ptr[Rows]];
        var val = new double[ptr[Rows]];
        for (var i = 0; i < Rows; i++)
        {
            _cols[i].CopyTo(idx, ptr[i]);
            _vals[i].CopyTo(val, ptr[i]);
        }
        return new CsrMatrix(Rows, Columns, ptr, idx, val, canonical: true);
    }

    /// <inheritdoc />
    public override void MatVec(ReadOnlySpan<double> x, Span<double> y)
    {
        if (x.Length != Columns || y.Length != Rows) throw new ArgumentException("Vector length does not match the matrix shape.");
        for (var i = 0; i < Rows; i++)
        {
            var s = 0.0;
            var c = _cols[i]; var v = _vals[i];
            for (var k = 0; k < c.Count; k++) s += v[k] * x[c[k]];
            y[i] = s;
        }
    }

    /// <inheritdoc />
    public override void RMatVec(ReadOnlySpan<double> x, Span<double> y)
    {
        if (x.Length != Rows || y.Length != Columns) throw new ArgumentException("Vector length does not match the matrix shape.");
        y.Clear();
        for (var i = 0; i < Rows; i++)
        {
            var c = _cols[i]; var v = _vals[i];
            for (var k = 0; k < c.Count; k++) y[c[k]] += v[k] * x[i];
        }
    }

    /// <inheritdoc />
    protected override SparseMatrixBase TransposeCore() => FromCsr(ToCsr().ToCsc().Inner);
}

/// <summary>
/// Diagonal-storage matrix — <c>scipy.sparse.dia_matrix</c>. Stores whole diagonals, which suits
/// banded matrices from finite differences.
/// </summary>
/// <remarks>
/// SciPy's convention: <c>Data[d, j]</c> holds <c>A[j − Offsets[d], j]</c>, i.e. data is indexed by
/// <em>column</em>, so the first <c>k</c> entries of a super-diagonal row (offset <c>k</c>) and the
/// last entries of a sub-diagonal row are padding and ignored.
/// </remarks>
public sealed class DiaMatrix : SparseMatrixBase
{
    private readonly double[] _data; // ndiag × width, row-major
    private readonly int[] _offsets;
    private readonly int _width;

    /// <summary>Builds from a (number of diagonals × width) data array and the diagonal offsets.</summary>
    public DiaMatrix(NdArray data, int[] offsets, int rows, int columns) : base(rows, columns)
    {
        ArgumentNullException.ThrowIfNull(offsets);
        if (data.Rank == 1) data = data.Reshape(1, data.Size);
        if (data.Rank != 2 || data.Shape[0] != offsets.Length)
            throw new ArgumentException("data must have one row per offset.");
        if (offsets.Distinct().Count() != offsets.Length) throw new ArgumentException("Offsets must be distinct.");
        _data = data.AsContiguous().ToArray();
        _offsets = (int[])offsets.Clone();
        _width = data.Shape[1];
    }

    internal static DiaMatrix FromCsr(CsrMatrix csr)
    {
        var offs = new SortedSet<int>();
        foreach (var (i, j, _) in csr.EnumerateEntries()) offs.Add(j - i);
        var offsets = offs.ToArray();
        var pos = new Dictionary<int, int>();
        for (var d = 0; d < offsets.Length; d++) pos[offsets[d]] = d;
        var width = csr.Columns;
        var buf = new double[offsets.Length * width];
        foreach (var (i, j, v) in csr.EnumerateEntries()) buf[pos[j - i] * width + j] += v;
        return new DiaMatrix(new NdArray(buf, offsets.Length, width), offsets, csr.Rows, csr.Columns);
    }

    /// <summary>Diagonal offsets (k &gt; 0 above the main diagonal).</summary>
    public ReadOnlySpan<int> Offsets => _offsets;

    /// <summary>The (number of diagonals × width) data array, a copy.</summary>
    public NdArray Data => new((double[])_data.Clone(), _offsets.Length, _width);

    /// <inheritdoc />
    public override int Nnz
    {
        get
        {
            var n = 0;
            foreach (var k in _offsets)
            {
                int jStart = Math.Max(0, k), jEnd = Math.Min(Columns, Math.Min(_width, Rows + k));
                n += Math.Max(0, jEnd - jStart);
            }
            return n;
        }
    }

    /// <inheritdoc />
    public override string Format => "dia";

    /// <inheritdoc />
    public override CsrMatrix ToCsr()
    {
        var r = new List<int>(); var c = new List<int>(); var v = new List<double>();
        for (var d = 0; d < _offsets.Length; d++)
        {
            var k = _offsets[d];
            int jStart = Math.Max(0, k), jEnd = Math.Min(Columns, Math.Min(_width, Rows + k));
            for (var j = jStart; j < jEnd; j++)
            {
                var val = _data[d * _width + j];
                if (val == 0.0) continue; // padding and in-band zeros are not entries (SciPy's tocoo masks them too)
                r.Add(j - k); c.Add(j); v.Add(val);
            }
        }
        return CsrMatrix.FromTriplets(Rows, Columns, r.ToArray(), c.ToArray(), v.ToArray());
    }

    /// <inheritdoc />
    public override void MatVec(ReadOnlySpan<double> x, Span<double> y)
    {
        if (x.Length != Columns || y.Length != Rows) throw new ArgumentException("Vector length does not match the matrix shape.");
        y.Clear();
        for (var d = 0; d < _offsets.Length; d++)
        {
            var k = _offsets[d];
            int jStart = Math.Max(0, k), jEnd = Math.Min(Columns, Math.Min(_width, Rows + k));
            var row = _data.AsSpan(d * _width, _width);
            for (var j = jStart; j < jEnd; j++) y[j - k] += row[j] * x[j];
        }
    }

    /// <inheritdoc />
    public override void RMatVec(ReadOnlySpan<double> x, Span<double> y)
    {
        if (x.Length != Rows || y.Length != Columns) throw new ArgumentException("Vector length does not match the matrix shape.");
        y.Clear();
        for (var d = 0; d < _offsets.Length; d++)
        {
            var k = _offsets[d];
            int jStart = Math.Max(0, k), jEnd = Math.Min(Columns, Math.Min(_width, Rows + k));
            var row = _data.AsSpan(d * _width, _width);
            for (var j = jStart; j < jEnd; j++) y[j] += row[j] * x[j - k];
        }
    }

    /// <inheritdoc />
    protected override SparseMatrixBase TransposeCore() => FromCsr(ToCsr().ToCsc().Inner);
}
