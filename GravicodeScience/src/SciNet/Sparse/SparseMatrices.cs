using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Sparse;

/// <summary>
/// Constructors for sparse matrices — the module-level functions of <c>scipy.sparse</c>:
/// <c>eye</c>, <c>identity</c>, <c>diags</c>, <c>spdiags</c>, <c>kron</c>, <c>kronsum</c>,
/// <c>hstack</c>, <c>vstack</c>, <c>block_diag</c>, <c>bmat</c>, <c>random</c>, <c>tril</c>,
/// <c>triu</c> and <c>find</c>. Results are CSR unless the name says otherwise.
/// </summary>
public static class SparseMatrices
{
    /// <summary>Ones on the <paramref name="k"/>-th diagonal of an m×n matrix (SciPy's <c>eye</c>).</summary>
    public static CsrMatrix Eye(int m, int? n = null, int k = 0)
    {
        var cols = n ?? m;
        var r0 = Math.Max(0, -k);
        var len = k >= 0 ? Math.Min(m, cols - k) : Math.Min(m + k, cols);
        len = Math.Max(0, len);
        var r = new int[len]; var c = new int[len]; var v = new double[len];
        for (var t = 0; t < len; t++) { r[t] = r0 + t; c[t] = r0 + t + k; v[t] = 1.0; }
        return CsrMatrix.FromTriplets(m, cols, r, c, v);
    }

    /// <summary>The n×n identity.</summary>
    public static CsrMatrix Identity(int n) => Eye(n);

    /// <summary>
    /// Builds a matrix from diagonals (SciPy's <c>diags</c>). Each diagonal fills
    /// <c>A[i, i + offset]</c> from its top-left end; a one-element diagonal is broadcast, and a
    /// longer one is truncated. Without <paramref name="shape"/> the matrix is square with side
    /// <c>len(diagonals[0]) + |offsets[0]|</c>.
    /// </summary>
    public static CsrMatrix Diags(IReadOnlyList<double[]> diagonals, IReadOnlyList<int> offsets, (int Rows, int Columns)? shape = null)
    {
        if (diagonals.Count != offsets.Count) throw new ArgumentException("Need one offset per diagonal.");
        if (diagonals.Count == 0) throw new ArgumentException("At least one diagonal is required.");
        if (offsets.Distinct().Count() != offsets.Count) throw new ArgumentException("Offsets must be distinct.");
        var (m, n) = shape ?? (diagonals[0].Length + Math.Abs(offsets[0]), diagonals[0].Length + Math.Abs(offsets[0]));
        var r = new List<int>(); var c = new List<int>(); var v = new List<double>();
        var kMin = Math.Min(m, n);
        for (var d = 0; d < diagonals.Count; d++)
        {
            var off = offsets[d];
            var length = Math.Min(Math.Min(m + off, n - off), kMin);
            if (length < 0) throw new ArgumentException($"Offset {off} (index {d}) out of bounds.");
            var diag = diagonals[d];
            if (diag.Length != 1 && diag.Length < length)
                throw new ArgumentException($"Diagonal length (index {d}: {diag.Length} at offset {off}) does not agree with matrix size ({m}, {n}).");
            var i0 = Math.Max(0, -off);
            for (var t = 0; t < length; t++)
            {
                r.Add(i0 + t); c.Add(i0 + t + off); v.Add(diag.Length == 1 ? diag[0] : diag[t]);
            }
        }
        return CsrMatrix.FromTriplets(m, n, r.ToArray(), c.ToArray(), v.ToArray());
    }

    /// <summary>Constant diagonals: <c>values[d]</c> repeated along offset <c>offsets[d]</c> of an m×n matrix.</summary>
    public static CsrMatrix Diags(IReadOnlyList<double> values, IReadOnlyList<int> offsets, int m, int? n = null)
        => Diags(values.Select(x => new[] { x }).ToArray(), offsets, (m, n ?? m));

    /// <summary>A single diagonal at <paramref name="offset"/>.</summary>
    public static CsrMatrix Diags(double[] diagonal, int offset = 0) => Diags([diagonal], [offset]);

    /// <summary>
    /// SciPy's <c>spdiags</c>: <c>data[d, j]</c> is placed at <c>A[j − diags[d], j]</c> (column-indexed,
    /// unlike <see cref="Diags(IReadOnlyList{double[]}, IReadOnlyList{int}, ValueTuple{int, int}?)"/>).
    /// </summary>
    public static DiaMatrix SpDiags(NdArray data, int[] diags, int m, int n) => new(data, diags, m, n);

    /// <summary>Kronecker product <c>A ⊗ B</c>.</summary>
    public static CsrMatrix Kron(SparseMatrixBase a, SparseMatrixBase b)
    {
        var ca = a.ToCoo();
        var cb = b.ToCoo();
        int r = b.Rows, s = b.Columns;
        var nnz = (long)ca.Nnz * cb.Nnz;
        if (nnz > int.MaxValue) throw new ArgumentException("Kronecker product has too many non-zeros.");
        var rows = new int[nnz]; var cols = new int[nnz]; var vals = new double[nnz];
        var w = 0;
        for (var p = 0; p < ca.Nnz; p++)
        {
            int i = ca.RowArray[p], j = ca.ColArray[p];
            var av = ca.DataArray[p];
            for (var q = 0; q < cb.Nnz; q++)
            {
                rows[w] = i * r + cb.RowArray[q];
                cols[w] = j * s + cb.ColArray[q];
                vals[w++] = av * cb.DataArray[q];
            }
        }
        return CsrMatrix.FromTriplets(a.Rows * r, a.Columns * s, rows, cols, vals);
    }

    /// <summary>Kronecker sum <c>kron(I_n, A) + kron(B, I_m)</c> for square A (m×m) and B (n×n).</summary>
    public static CsrMatrix Kronsum(SparseMatrixBase a, SparseMatrixBase b)
    {
        if (a.Rows != a.Columns || b.Rows != b.Columns) throw new ArgumentException("Kronsum needs square matrices.");
        return Kron(Identity(b.Rows), a).Add(Kron(b, Identity(a.Rows)));
    }

    /// <summary>Side-by-side concatenation (equal row counts).</summary>
    public static CsrMatrix HStack(params SparseMatrixBase[] blocks)
    {
        var grid = new SparseMatrixBase?[1, blocks.Length];
        for (var j = 0; j < blocks.Length; j++) grid[0, j] = blocks[j];
        return Bmat(grid);
    }

    /// <summary>Vertical concatenation (equal column counts).</summary>
    public static CsrMatrix VStack(params SparseMatrixBase[] blocks)
    {
        var grid = new SparseMatrixBase?[blocks.Length, 1];
        for (var i = 0; i < blocks.Length; i++) grid[i, 0] = blocks[i];
        return Bmat(grid);
    }

    /// <summary>Block-diagonal matrix of the given blocks.</summary>
    public static CsrMatrix BlockDiag(params SparseMatrixBase[] blocks)
    {
        var rows = new List<int>(); var cols = new List<int>(); var vals = new List<double>();
        int r0 = 0, c0 = 0;
        foreach (var b in blocks)
        {
            foreach (var (i, j, v) in b.ToCsr().EnumerateEntries()) { rows.Add(r0 + i); cols.Add(c0 + j); vals.Add(v); }
            r0 += b.Rows;
            c0 += b.Columns;
        }
        return CsrMatrix.FromTriplets(r0, c0, rows.ToArray(), cols.ToArray(), vals.ToArray());
    }

    /// <summary>
    /// Assembles a matrix from a grid of blocks (SciPy's <c>bmat</c>); null blocks are zero. Every
    /// block row and block column needs at least one non-null block to fix its size.
    /// </summary>
    public static CsrMatrix Bmat(SparseMatrixBase?[,] blocks)
    {
        int br = blocks.GetLength(0), bc = blocks.GetLength(1);
        var heights = new int[br];
        var widths = new int[bc];
        Array.Fill(heights, -1);
        Array.Fill(widths, -1);
        for (var i = 0; i < br; i++)
            for (var j = 0; j < bc; j++)
            {
                var b = blocks[i, j];
                if (b is null) continue;
                if (heights[i] >= 0 && heights[i] != b.Rows)
                    throw new ArgumentException($"blocks[{i},:] has incompatible row dimensions ({heights[i]} vs {b.Rows}).");
                if (widths[j] >= 0 && widths[j] != b.Columns)
                    throw new ArgumentException($"blocks[:,{j}] has incompatible column dimensions ({widths[j]} vs {b.Columns}).");
                heights[i] = b.Rows;
                widths[j] = b.Columns;
            }
        for (var i = 0; i < br; i++) if (heights[i] < 0) throw new ArgumentException($"blocks[{i},:] is all null; its height is unknown.");
        for (var j = 0; j < bc; j++) if (widths[j] < 0) throw new ArgumentException($"blocks[:,{j}] is all null; its width is unknown.");

        var rows = new List<int>(); var cols = new List<int>(); var vals = new List<double>();
        var rOff = 0;
        for (var i = 0; i < br; i++)
        {
            var cOff = 0;
            for (var j = 0; j < bc; j++)
            {
                var b = blocks[i, j];
                if (b is not null)
                    foreach (var (r, c, v) in b.ToCsr().EnumerateEntries()) { rows.Add(rOff + r); cols.Add(cOff + c); vals.Add(v); }
                cOff += widths[j];
            }
            rOff += heights[i];
        }
        return CsrMatrix.FromTriplets(rOff, widths.Sum(), rows.ToArray(), cols.ToArray(), vals.ToArray());
    }

    /// <summary>
    /// A random m×n matrix with exactly <c>round(density·m·n)</c> distinct non-zeros at uniformly
    /// chosen positions, values uniform on [0, 1) (SciPy's <c>random</c>; the stream differs from NumPy's).
    /// </summary>
    public static CsrMatrix Random(int m, int n, double density = 0.01, int? seed = null)
        => RandomArray(m, n, density, seed, null);

    /// <summary>
    /// Like <see cref="Random"/> with a custom value sampler (SciPy's <c>random_array</c> with
    /// <c>data_sampler</c>). Positions are drawn with Floyd's algorithm, so the cost is O(nnz), not O(m·n).
    /// </summary>
    public static CsrMatrix RandomArray(int m, int n, double density, int? seed, Func<GraviRandom, double>? dataSampler)
    {
        if (density < 0 || density > 1) throw new ArgumentException("density must be in [0, 1].");
        var rng = seed.HasValue ? new GraviRandom(seed.Value) : new GraviRandom((int)Environment.TickCount64);
        var total = (long)m * n;
        var k = (long)Math.Round(density * total);
        if (k > int.MaxValue) throw new ArgumentException("Too many non-zeros requested.");
        var chosen = new HashSet<long>();
        for (var j = total - k; j < total; j++)
        {
            var t = (long)(rng.NextDouble() * (j + 1));
            if (t > j) t = j;
            if (!chosen.Add(t)) chosen.Add(j);
        }
        var keys = chosen.ToArray();
        Array.Sort(keys);
        var r = new int[keys.Length]; var c = new int[keys.Length]; var v = new double[keys.Length];
        for (var i = 0; i < keys.Length; i++)
        {
            r[i] = (int)(keys[i] / n);
            c[i] = (int)(keys[i] % n);
            v[i] = dataSampler is null ? rng.NextDouble() : dataSampler(rng);
        }
        return CsrMatrix.FromTriplets(m, n, r, c, v);
    }

    /// <summary>Lower triangle (entries with <c>j − i ≤ k</c>).</summary>
    public static CsrMatrix Tril(SparseMatrixBase a, int k = 0) => a.ToCsr().Tril(k);

    /// <summary>Upper triangle (entries with <c>j − i ≥ k</c>).</summary>
    public static CsrMatrix Triu(SparseMatrixBase a, int k = 0) => a.ToCsr().Triu(k);

    /// <summary>Rows, columns and values of the non-zero entries (SciPy's <c>find</c>).</summary>
    public static (int[] Rows, int[] Columns, double[] Values) Find(SparseMatrixBase a) => a.ToCsr().Find();
}
