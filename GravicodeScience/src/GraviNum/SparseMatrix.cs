namespace Gravicode.Science.GraviNum;

/// <summary>
/// A sparse matrix in compressed sparse row (CSR) form.
/// </summary>
/// <remarks>
/// CSR stores only the non-zeros plus one row pointer per row, so a matrix that is 99% zeros
/// costs about 1% of the memory and its products cost work proportional to the non-zero count
/// rather than to rows×columns. This is what makes the 100k-node graphs in GraviGraph and the
/// bag-of-words matrices in GraviText tractable. Build one with <see cref="FromDense"/>,
/// <see cref="FromTriplets"/> or the <see cref="SparseBuilder"/>.
/// </remarks>
public sealed class SparseMatrix
{
    private readonly double[] _values;
    private readonly int[] _columnIndices;
    private readonly int[] _rowPointers;

    private SparseMatrix(int rows, int columns, double[] values, int[] columnIndices, int[] rowPointers)
    {
        Rows = rows;
        Columns = columns;
        _values = values;
        _columnIndices = columnIndices;
        _rowPointers = rowPointers;
    }

    /// <summary>Number of rows.</summary>
    public int Rows { get; }

    /// <summary>Number of columns.</summary>
    public int Columns { get; }

    /// <summary>Number of stored non-zero entries.</summary>
    public int NonZeroCount => _values.Length;

    /// <summary>Fraction of entries that are stored.</summary>
    public double Density => Rows * (long)Columns == 0 ? 0 : (double)NonZeroCount / ((long)Rows * Columns);

    /// <summary>Stored values, ordered row by row.</summary>
    public ReadOnlySpan<double> Values => _values;

    /// <summary>Column index of each stored value.</summary>
    public ReadOnlySpan<int> ColumnIndices => _columnIndices;

    /// <summary>Index into <see cref="Values"/> where each row starts; length is <c>Rows + 1</c>.</summary>
    public ReadOnlySpan<int> RowPointers => _rowPointers;

    /// <summary>Reads an entry; missing entries read as zero.</summary>
    public double this[int row, int column]
    {
        get
        {
            for (var k = _rowPointers[row]; k < _rowPointers[row + 1]; k++)
                if (_columnIndices[k] == column) return _values[k];
            return 0.0;
        }
    }

    /// <summary>Builds a CSR matrix from a dense array, dropping values at or below <paramref name="threshold"/>.</summary>
    public static SparseMatrix FromDense(NdArray dense, double threshold = 0.0)
    {
        if (dense.Rank != 2) throw new ArgumentException("FromDense expects a rank 2 array.");
        var rows = dense.Shape[0];
        var columns = dense.Shape[1];

        var values = new List<double>();
        var cols = new List<int>();
        var pointers = new int[rows + 1];

        for (var i = 0; i < rows; i++)
        {
            pointers[i] = values.Count;
            for (var j = 0; j < columns; j++)
            {
                var v = dense[i, j];
                if (Math.Abs(v) > threshold)
                {
                    values.Add(v);
                    cols.Add(j);
                }
            }
        }
        pointers[rows] = values.Count;
        return new SparseMatrix(rows, columns, values.ToArray(), cols.ToArray(), pointers);
    }

    /// <summary>Builds a CSR matrix from (row, column, value) triplets. Duplicates are summed.</summary>
    public static SparseMatrix FromTriplets(int rows, int columns, IEnumerable<(int Row, int Column, double Value)> triplets)
    {
        var perRow = new Dictionary<int, Dictionary<int, double>>();
        foreach (var (r, c, v) in triplets)
        {
            if (r < 0 || r >= rows) throw new ArgumentOutOfRangeException(nameof(triplets), $"Row {r} out of range.");
            if (c < 0 || c >= columns) throw new ArgumentOutOfRangeException(nameof(triplets), $"Column {c} out of range.");
            if (!perRow.TryGetValue(r, out var row)) perRow[r] = row = [];
            row.TryGetValue(c, out var existing);
            row[c] = existing + v;
        }

        var values = new List<double>();
        var cols = new List<int>();
        var pointers = new int[rows + 1];
        for (var i = 0; i < rows; i++)
        {
            pointers[i] = values.Count;
            if (!perRow.TryGetValue(i, out var row)) continue;
            foreach (var (c, v) in row.OrderBy(kv => kv.Key))
            {
                if (v == 0.0) continue;
                values.Add(v);
                cols.Add(c);
            }
        }
        pointers[rows] = values.Count;
        return new SparseMatrix(rows, columns, values.ToArray(), cols.ToArray(), pointers);
    }

    /// <summary>The identity matrix in sparse form.</summary>
    public static SparseMatrix Identity(int n)
        => FromTriplets(n, n, Enumerable.Range(0, n).Select(i => (i, i, 1.0)));

    /// <summary>Expands back to a dense array.</summary>
    public NdArray ToDense()
    {
        var dense = NdArray.Zeros(Rows, Columns);
        for (var i = 0; i < Rows; i++)
            for (var k = _rowPointers[i]; k < _rowPointers[i + 1]; k++)
                dense[i, _columnIndices[k]] = _values[k];
        return dense;
    }

    /// <summary>Sparse matrix times dense vector.</summary>
    public NdArray Multiply(NdArray vector)
    {
        if (vector.Size != Columns)
            throw new InvalidOperationException($"Vector has {vector.Size} entries, expected {Columns}.");

        var x = vector.ToArray();
        var result = NdArray.Zeros(Rows);
        var y = result.Buffer;

        if (Rows >= 512)
        {
            Parallel.For(0, Rows, i =>
            {
                var acc = 0.0;
                for (var k = _rowPointers[i]; k < _rowPointers[i + 1]; k++) acc += _values[k] * x[_columnIndices[k]];
                y[i] = acc;
            });
        }
        else
        {
            for (var i = 0; i < Rows; i++)
            {
                var acc = 0.0;
                for (var k = _rowPointers[i]; k < _rowPointers[i + 1]; k++) acc += _values[k] * x[_columnIndices[k]];
                y[i] = acc;
            }
        }
        return result;
    }

    /// <summary>Sparse matrix times dense matrix.</summary>
    public NdArray Multiply(NdArray dense, bool denseIsMatrix)
    {
        if (!denseIsMatrix) return Multiply(dense);
        if (dense.Rank != 2 || dense.Shape[0] != Columns)
            throw new InvalidOperationException(
                $"Cannot multiply a {Rows}x{Columns} sparse matrix by ({Shapes.Describe(dense.Shape)}).");

        var n = dense.Shape[1];
        var b = dense.AsContiguous();
        var bBuf = b.Buffer;
        var bOff = b.Offset;
        var result = NdArray.Zeros(Rows, n);
        var c = result.Buffer;

        Parallel.For(0, Rows, i =>
        {
            var row = c.AsSpan(i * n, n);
            for (var k = _rowPointers[i]; k < _rowPointers[i + 1]; k++)
                LinAlg.AxpySpan(_values[k], bBuf.AsSpan(bOff + _columnIndices[k] * n, n), row);
        });
        return result;
    }

    /// <summary>Transposes the matrix, returning a new CSR matrix.</summary>
    public SparseMatrix Transpose()
    {
        var triplets = new List<(int, int, double)>(NonZeroCount);
        for (var i = 0; i < Rows; i++)
            for (var k = _rowPointers[i]; k < _rowPointers[i + 1]; k++)
                triplets.Add((_columnIndices[k], i, _values[k]));
        return FromTriplets(Columns, Rows, triplets);
    }

    /// <summary>Scales every stored value.</summary>
    public SparseMatrix Scale(double factor)
    {
        var values = new double[_values.Length];
        for (var i = 0; i < values.Length; i++) values[i] = _values[i] * factor;
        return new SparseMatrix(Rows, Columns, values, _columnIndices, _rowPointers);
    }

    /// <summary>Applies a function to every stored value; the function must map zero to zero.</summary>
    public SparseMatrix MapValues(Func<double, double> f)
    {
        var values = new double[_values.Length];
        for (var i = 0; i < values.Length; i++) values[i] = f(_values[i]);
        return new SparseMatrix(Rows, Columns, values, _columnIndices, _rowPointers);
    }

    /// <summary>Sum of each row.</summary>
    public NdArray RowSums()
    {
        var sums = NdArray.Zeros(Rows);
        for (var i = 0; i < Rows; i++)
        {
            var acc = 0.0;
            for (var k = _rowPointers[i]; k < _rowPointers[i + 1]; k++) acc += _values[k];
            sums.SetAt(i, acc);
        }
        return sums;
    }

    /// <summary>Non-zero entries of a row as (column, value) pairs.</summary>
    public IEnumerable<(int Column, double Value)> Row(int row)
    {
        for (var k = _rowPointers[row]; k < _rowPointers[row + 1]; k++)
            yield return (_columnIndices[k], _values[k]);
    }

    /// <summary>Every stored entry as (row, column, value).</summary>
    public IEnumerable<(int Row, int Column, double Value)> Entries()
    {
        for (var i = 0; i < Rows; i++)
            for (var k = _rowPointers[i]; k < _rowPointers[i + 1]; k++)
                yield return (i, _columnIndices[k], _values[k]);
    }

    /// <inheritdoc />
    public override string ToString()
        => $"SparseMatrix({Rows}x{Columns}, nnz={NonZeroCount}, density={Density:P2})";
}

/// <summary>Incremental builder for <see cref="SparseMatrix"/>, useful when the entries arrive one at a time.</summary>
public sealed class SparseBuilder(int rows, int columns)
{
    private readonly List<(int Row, int Column, double Value)> _entries = [];

    /// <summary>Number of rows in the matrix being built.</summary>
    public int Rows { get; } = rows;

    /// <summary>Number of columns in the matrix being built.</summary>
    public int Columns { get; } = columns;

    /// <summary>Records an entry; repeated coordinates accumulate.</summary>
    public SparseBuilder Add(int row, int column, double value)
    {
        _entries.Add((row, column, value));
        return this;
    }

    /// <summary>Materialises the CSR matrix.</summary>
    public SparseMatrix Build() => SparseMatrix.FromTriplets(Rows, Columns, _entries);
}
