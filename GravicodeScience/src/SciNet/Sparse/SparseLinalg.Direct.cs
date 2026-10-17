using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Sparse;

/// <summary>Fill-reducing column ordering for <see cref="SparseLinalg.Splu"/> (SciPy's <c>permc_spec</c>).</summary>
public enum ColumnOrdering
{
    /// <summary>No reordering.</summary>
    Natural,

    /// <summary>Approximate minimum degree on the pattern of <c>AᵀA</c> (SuperLU's <c>MMD_ATA</c>).</summary>
    MmdAtA,

    /// <summary>Approximate minimum degree on the pattern of <c>Aᵀ + A</c> (SuperLU's <c>MMD_AT_PLUS_A</c>); best for nearly symmetric patterns.</summary>
    MmdAtPlusA,

    /// <summary>
    /// SciPy's default. COLAMD itself is not ported: this uses the minimum-degree ordering of
    /// <c>AᵀA</c> (the ordering COLAMD approximates without forming <c>AᵀA</c>), or of
    /// <c>A + Aᵀ</c> when the pattern is symmetric, where the AᵀA graph gives far more fill.
    /// Fill is comparable, not identical, to SuperLU's.
    /// </summary>
    Colamd,
}

/// <summary>
/// A sparse LU factorisation <c>Pr A Pc = L U</c> from <see cref="SparseLinalg.Splu"/>, with
/// SciPy's <c>SuperLU</c> conventions for the permutation vectors.
/// </summary>
/// <remarks>
/// <c>PermR[i]</c> is the position original row <c>i</c> moves to and <c>PermC[j]</c> the position of
/// original column <c>j</c>; so <c>Pr[PermR[i], i] = 1</c>, <c>Pc[j, PermC[j]] = 1</c> and
/// <c>A = Prᵀ L U Pcᵀ</c> — exactly the reconstruction recipe in SciPy's documentation.
/// <c>L</c> is unit lower triangular, <c>U</c> upper triangular, both CSC with sorted indices.
/// </remarks>
public sealed class SparseLuFactorization : ILinearOperator
{
    // Either CSC factors (the left-looking reference path) or a supernodal factorisation.
    private int[] _lp = [], _li = [], _up = [], _ui = [];
    private double[] _lx = [], _ux = [];
    private int[] _pinv = []; // original row -> pivot position
    private int[] _q = [];    // pivot position -> original column
    private int[] _permC = [];
    private readonly SupernodalLu? _sn;
    private readonly object _exportLock = new();
    private bool _exported;

    internal SparseLuFactorization(int n, int[] lp, int[] li, double[] lx, int[] up, int[] ui, double[] ux, int[] pinv, int[] q)
    {
        N = n;
        _lp = lp; _li = li; _lx = lx; _up = up; _ui = ui; _ux = ux; _pinv = pinv; _q = q;
        var permC = new int[n];
        for (var k = 0; k < n; k++) permC[q[k]] = k;
        _permC = permC;
        _exported = true;
    }

    internal SparseLuFactorization(SupernodalLu sn)
    {
        N = sn.N;
        _sn = sn;
    }

    /// <summary>The supernodal factorisation behind this object, when <see cref="SparseLinalg.Splu"/> produced one.</summary>
    internal SupernodalLu? Supernodal => _sn;

    /// <summary>Order of the factorised matrix.</summary>
    public int N { get; }

    int ILinearOperator.Rows => N;
    int ILinearOperator.Columns => N;

    /// <summary>Converts supernodal storage to CSC factors the first time they are asked for.</summary>
    private void EnsureCsc()
    {
        if (_exported) return;
        lock (_exportLock)
        {
            if (_exported) return;
            var (permR, permC, l, u) = _sn!.Export();
            _pinv = permR;
            _permC = permC;
            _q = new int[N];
            for (var j = 0; j < N; j++) _q[permC[j]] = j;
            _lp = l.Indptr.ToArray(); _li = l.Indices.ToArray(); _lx = l.Data.ToArray();
            _up = u.Indptr.ToArray(); _ui = u.Indices.ToArray(); _ux = u.Data.ToArray();
            _exported = true;
        }
    }

    /// <summary>Unit lower-triangular factor (CSC).</summary>
    public CscMatrix L { get { EnsureCsc(); return new(_lx, _li, _lp, N, N, copy: true); } }

    /// <summary>Upper-triangular factor (CSC).</summary>
    public CscMatrix U { get { EnsureCsc(); return new(_ux, _ui, _up, N, N, copy: true); } }

    /// <summary>Row permutation: original row <c>i</c> becomes row <c>PermR[i]</c>.</summary>
    public int[] PermR { get { EnsureCsc(); return (int[])_pinv.Clone(); } }

    /// <summary>Column permutation: original column <c>j</c> becomes column <c>PermC[j]</c>.</summary>
    public int[] PermC { get { EnsureCsc(); return (int[])_permC.Clone(); } }

    /// <summary>
    /// Entries in L plus U (the fill the ordering is trying to keep down): the non-zeros of
    /// <see cref="L"/> and <see cref="U"/>. The supernodal storage also holds explicit zeros from
    /// supernode amalgamation; those are not counted.
    /// </summary>
    public int Nnz => _sn is not null ? (int)Math.Min(int.MaxValue, _sn.NonZeros) : _lp[N] + _up[N];

    /// <summary>Solves <c>A x = b</c> (or <c>Aᵀ x = b</c> with <paramref name="transpose"/>).</summary>
    public double[] Solve(ReadOnlySpan<double> b, bool transpose = false)
    {
        if (b.Length != N) throw new ArgumentException($"b has {b.Length} entries, expected {N}.");
        var x = new double[N];
        if (transpose) SolveTransposeInto(b, x); else SolveInto(b, x);
        return x;
    }

    /// <summary>Solves <c>A X = B</c> column by column for a dense rank 2 (or rank 1) <paramref name="b"/>.</summary>
    public NdArray Solve(NdArray b, bool transpose = false)
    {
        if (b.Rank == 1) return new NdArray(Solve(b.AsContiguous().AsSpan(), transpose), N);
        if (b.Rank != 2 || b.Shape[0] != N) throw new ArgumentException($"B must have {N} rows.");
        var k = b.Shape[1];
        var src = b.AsContiguous().ToArray();
        var outBuf = new double[N * k];
        var col = new double[N];
        var res = new double[N];
        for (var j = 0; j < k; j++)
        {
            for (var i = 0; i < N; i++) col[i] = src[i * k + j];
            if (transpose) SolveTransposeInto(col, res); else SolveInto(col, res);
            for (var i = 0; i < N; i++) outBuf[i * k + j] = res[i];
        }
        return new NdArray(outBuf, N, k);
    }

    internal void SolveInto(ReadOnlySpan<double> b, Span<double> x)
    {
        if (_sn is not null) { _sn.Solve(b, x); return; }
        var c = new double[N];
        for (var i = 0; i < N; i++) c[_pinv[i]] = b[i];
        for (var j = 0; j < N; j++)
        {
            var cj = c[j];
            if (cj == 0) continue;
            for (var p = _lp[j] + 1; p < _lp[j + 1]; p++) c[_li[p]] -= _lx[p] * cj;
        }
        for (var j = N - 1; j >= 0; j--)
        {
            var d = _up[j + 1] - 1;
            c[j] /= _ux[d];
            var cj = c[j];
            if (cj == 0) continue;
            for (var p = _up[j]; p < d; p++) c[_ui[p]] -= _ux[p] * cj;
        }
        for (var k = 0; k < N; k++) x[_q[k]] = c[k];
    }

    internal void SolveTransposeInto(ReadOnlySpan<double> b, Span<double> x)
    {
        if (_sn is not null) { _sn.SolveTranspose(b, x); return; }
        var w = new double[N];
        for (var k = 0; k < N; k++) w[k] = b[_q[k]];
        // Uᵀ w' = w: column j of U is row j of Uᵀ.
        for (var j = 0; j < N; j++)
        {
            var d = _up[j + 1] - 1;
            var acc = w[j];
            for (var p = _up[j]; p < d; p++) acc -= _ux[p] * w[_ui[p]];
            w[j] = acc / _ux[d];
        }
        // Lᵀ v = w', backwards.
        for (var j = N - 1; j >= 0; j--)
        {
            var acc = w[j];
            for (var p = _lp[j] + 1; p < _lp[j + 1]; p++) acc -= _lx[p] * w[_li[p]];
            w[j] = acc;
        }
        for (var i = 0; i < N; i++) x[i] = w[_pinv[i]];
    }

    /// <summary>Applies <c>A⁻¹</c>: lets the factorisation act as an exact preconditioner or shift-invert operator.</summary>
    public void MatVec(ReadOnlySpan<double> x, Span<double> y) => SolveInto(x, y);

    /// <summary>Applies <c>A⁻ᵀ</c>.</summary>
    public void RMatVec(ReadOnlySpan<double> x, Span<double> y) => SolveTransposeInto(x, y);
}

/// <summary>
/// An incomplete LU factorisation (<see cref="SparseLinalg.Spilu"/>, <see cref="SparseLinalg.Ilu0"/>).
/// Its <c>MatVec</c> applies <c>(L U)⁻¹</c>, so it can be passed directly as the preconditioner
/// <c>M</c> of any Krylov solver.
/// </summary>
public sealed class IncompleteLu : ILinearOperator
{
    private readonly CsrMatrix _l; // strictly lower, unit diagonal implied
    private readonly CsrMatrix _u; // upper including diagonal

    internal IncompleteLu(CsrMatrix l, CsrMatrix u)
    {
        _l = l;
        _u = u;
    }

    /// <summary>Order of the factorised matrix.</summary>
    public int Rows => _u.Rows;

    /// <summary>Order of the factorised matrix.</summary>
    public int Columns => _u.Rows;

    /// <summary>Unit lower factor (the unit diagonal is stored explicitly here).</summary>
    public CsrMatrix L => _l.Add(SparseMatrices.Identity(Rows));

    /// <summary>Upper factor including the diagonal.</summary>
    public CsrMatrix U => _u;

    /// <summary>Stored entries of both factors (excluding L's unit diagonal).</summary>
    public int Nnz => _l.Nnz + _u.Nnz;

    /// <summary>Solves <c>L U x = b</c>.</summary>
    public double[] Solve(ReadOnlySpan<double> b)
    {
        var x = new double[Rows];
        MatVec(b, x);
        return x;
    }

    /// <inheritdoc />
    public void MatVec(ReadOnlySpan<double> x, Span<double> y)
    {
        var n = Rows;
        var lp = _l.IndptrArray; var li = _l.IndicesArray; var lx = _l.DataArray;
        var up = _u.IndptrArray; var ui = _u.IndicesArray; var ux = _u.DataArray;
        for (var i = 0; i < n; i++)
        {
            var acc = x[i];
            for (var p = lp[i]; p < lp[i + 1]; p++) acc -= lx[p] * y[li[p]];
            y[i] = acc;
        }
        for (var i = n - 1; i >= 0; i--)
        {
            var acc = y[i];
            var diag = 0.0;
            for (var p = up[i]; p < up[i + 1]; p++)
            {
                var j = ui[p];
                if (j == i) diag = ux[p];
                else acc -= ux[p] * y[j];
            }
            y[i] = acc / diag;
        }
    }

    /// <inheritdoc />
    public void RMatVec(ReadOnlySpan<double> x, Span<double> y)
    {
        // (LU)ᵀ = Uᵀ Lᵀ: solve Uᵀ w = x forwards (scatter), then Lᵀ y = w backwards.
        var n = Rows;
        var lp = _l.IndptrArray; var li = _l.IndicesArray; var lx = _l.DataArray;
        var up = _u.IndptrArray; var ui = _u.IndicesArray; var ux = _u.DataArray;
        var w = x.ToArray();
        for (var i = 0; i < n; i++)
        {
            var diag = 0.0;
            for (var p = up[i]; p < up[i + 1]; p++) if (ui[p] == i) diag = ux[p];
            w[i] /= diag;
            for (var p = up[i]; p < up[i + 1]; p++) if (ui[p] != i) w[ui[p]] -= ux[p] * w[i];
        }
        for (var i = n - 1; i >= 0; i--)
            for (var p = lp[i]; p < lp[i + 1]; p++) w[li[p]] -= lx[p] * w[i];
        w.CopyTo(y);
    }
}

public static partial class SparseLinalg
{
    // ------------------------------------------------------------------ orderings

    /// <summary>
    /// A fill-reducing column permutation <c>q</c> (<c>q[k]</c> = original column eliminated k-th).
    /// </summary>
    /// <remarks>
    /// Approximate minimum degree (Amestoy, Davis &amp; Duff 1996) with aggressive absorption,
    /// supervariables and dense-row removal, on the pattern the option names: <c>A + Aᵀ</c>, or
    /// <c>AᵀA</c> (rows denser than <c>10√n</c> dropped first, as COLAMD does). COLAMD itself is
    /// not ported; <see cref="ColumnOrdering.Colamd"/> orders <c>AᵀA</c>, or <c>A + Aᵀ</c> when the
    /// pattern is symmetric. Deterministic.
    /// </remarks>
    public static int[] ColumnPermutation(SparseMatrixBase a, ColumnOrdering ordering)
    {
        var n = a.Columns;
        if (ordering == ColumnOrdering.Natural) return Identity(n);
        var csr = a.ToCsr().SumDuplicates();
        // Colamd on a structurally symmetric matrix: order A + Aᵀ instead. Measured on the
        // 100×100 Poisson matrix, the AᵀA ordering gave 2.6× the fill of the A + Aᵀ one
        // (1.39M vs 0.54M entries in L + U; SuperLU's COLAMD gives 0.65M) — squaring the pattern
        // is the wrong graph when the matrix is already symmetric.
        if (ordering == ColumnOrdering.Colamd && a.Rows == a.Columns && IsPatternSymmetric(csr)) ordering = ColumnOrdering.MmdAtPlusA;
        if (ordering == ColumnOrdering.MmdAtPlusA)
        {
            if (a.Rows != a.Columns) throw new ArgumentException("MmdAtPlusA needs a square matrix.");
            var (sp, si) = SparseAmd.SymmetricPattern(n, csr.Indptr, csr.Indices);
            return SparseAmd.Order(n, sp, si);
        }
        var (tp, ti) = SparseAmd.AtAPattern(a.Rows, n, csr.Indptr, csr.Indices);
        return SparseAmd.Order(n, tp, ti);
    }

    private static int[] Identity(int n)
    {
        var id = new int[n];
        for (var i = 0; i < n; i++) id[i] = i;
        return id;
    }

    private static bool IsPatternSymmetric(CsrMatrix a)
    {
        var c = a.SumDuplicates();
        var t = c.ToCsc().Inner; // Aᵀ as CSR, canonical
        if (c.Nnz != t.Nnz) return false;
        return c.Indptr.SequenceEqual(t.Indptr) && c.Indices.SequenceEqual(t.Indices);
    }

    /// <summary>
    /// A maximum transversal (Duff's MC21: depth-first augmenting paths after a cheap-assignment
    /// pass; the structure of CSparse <c>cs_maxtrans</c>) on the non-zero entries. Returns
    /// <c>rowOf[j]</c>, the row matched to column j, or null when the matrix is structurally singular.
    /// </summary>
    internal static int[]? MaxTransversal(int n, int[] ap, int[] ai, double[] ax)
    {
        var jmatch = new int[n]; // row -> column
        Array.Fill(jmatch, -1);
        var cheap = new int[n];
        Array.Copy(ap, cheap, n);
        var visited = new int[n];
        Array.Fill(visited, -1);
        var js = new int[n]; var isr = new int[n]; var ps = new int[n];
        for (var k = 0; k < n; k++)
        {
            var head = 0;
            js[0] = k;
            var found = false;
            while (head >= 0)
            {
                var j = js[head];
                if (visited[j] != k)
                {
                    visited[j] = k;
                    var p = cheap[j];
                    for (; p < ap[j + 1]; p++)
                    {
                        var i = ai[p];
                        if (ax[p] != 0 && jmatch[i] == -1) { found = true; isr[head] = i; break; }
                    }
                    cheap[j] = found ? p + 1 : p;
                    if (found) break;
                    ps[head] = ap[j];
                }
                var q = ps[head];
                for (; q < ap[j + 1]; q++)
                {
                    var i = ai[q];
                    if (ax[q] == 0 || jmatch[i] < 0 || visited[jmatch[i]] == k) continue;
                    ps[head] = q + 1;
                    isr[head] = i;
                    js[++head] = jmatch[i];
                    break;
                }
                if (q == ap[j + 1]) head--;
            }
            if (!found) return null;
            for (var p = head; p >= 0; p--) jmatch[isr[p]] = js[p];
        }
        var rowOf = new int[n];
        for (var i = 0; i < n; i++) rowOf[jmatch[i]] = i;
        return rowOf;
    }

    // ------------------------------------------------------------------ sparse LU

    /// <summary>
    /// Sparse LU factorisation <c>Pr A Pc = L U</c> — <c>scipy.sparse.linalg.splu</c>.
    /// </summary>
    /// <param name="a">Square sparse matrix.</param>
    /// <param name="permcSpec">Column ordering (default <see cref="ColumnOrdering.Colamd"/>, as SciPy).</param>
    /// <param name="diagPivotThresh">
    /// Threshold partial pivoting: the diagonal entry is kept as pivot when
    /// <c>|a_diag| ≥ thresh · max|column|</c>. 1.0 (default) is plain partial pivoting with ties
    /// going to the diagonal; smaller values preserve sparsity at some cost in stability; 0 always
    /// takes the diagonal when it is non-zero. Every multiplier in L is bounded by <c>1/thresh</c>.
    /// </param>
    /// <remarks>
    /// <para>Multifrontal (Duff &amp; Reid 1983; the unsymmetric scheme of MUMPS): the symbolic
    /// analysis runs on <c>A + Aᵀ</c> under the fill-reducing order — elimination tree, postorder,
    /// column counts, supernodes with relaxed amalgamation — and each supernode is factorised as a
    /// dense front with blocked GEMM updates. Pivots that fail the threshold test against their
    /// whole front column are delayed to the parent front, so the pivot sequence is still chosen
    /// numerically. A zero-free diagonal is first established by a maximum transversal when the
    /// diagonal has zeros.</para>
    /// <para><see cref="ColumnOrdering.MmdAtPlusA"/> and <see cref="ColumnOrdering.Colamd"/> both
    /// order <c>A + Aᵀ</c> (after the transversal) here, because that is the graph whose fill a
    /// multifrontal method pays for; <see cref="ColumnOrdering.MmdAtA"/> orders <c>AᵀA</c>. Fill is
    /// comparable to SuperLU's, not identical. Throws if the matrix is structurally or numerically
    /// singular. The left-looking Gilbert–Peierls factorisation this replaced is kept internally as
    /// the reference the tests compare against.</para>
    /// </remarks>
    public static SparseLuFactorization Splu(SparseMatrixBase a, ColumnOrdering permcSpec = ColumnOrdering.Colamd, double diagPivotThresh = 1.0)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Rows != a.Columns) throw new ArgumentException($"Splu needs a square matrix, got {a.Rows}x{a.Columns}.");
        if (diagPivotThresh < 0 || diagPivotThresh > 1) throw new ArgumentException("diagPivotThresh must be in [0, 1].");
        var n = a.Rows;
        var csc = a.ToCsc().SumDuplicates();
        int[] ap = csc.Indptr.ToArray(), ai = csc.Indices.ToArray();
        var ax = csc.Data.ToArray();

        // Zero-free diagonal: the symmetric analysis must see every pivot's own entry.
        var rowOf = Identity(n);
        var zeroDiag = false;
        for (var j = 0; j < n && !zeroDiag; j++)
        {
            var ok = false;
            for (var p = ap[j]; p < ap[j + 1]; p++) if (ai[p] == j && ax[p] != 0) { ok = true; break; }
            zeroDiag = !ok;
        }
        if (zeroDiag)
            rowOf = MaxTransversal(n, ap, ai, ax) ?? throw new InvalidOperationException("Splu: matrix is structurally singular.");

        // Pattern of the row-permuted matrix (row rowOf[j] moves to position j).
        var posOfRow = new int[n];
        for (var j = 0; j < n; j++) posOfRow[rowOf[j]] = j;
        var a1i = new int[ai.Length];
        for (var p = 0; p < ai.Length; p++) a1i[p] = posOfRow[ai[p]];
        var (sp, si) = SparseAmd.SymmetricPattern(n, ap, a1i);
        int[] order;
        switch (permcSpec)
        {
            case ColumnOrdering.Natural: order = Identity(n); break;
            case ColumnOrdering.MmdAtA:
            {
                var csr = csc.ToCsr();
                var (tp, ti) = SparseAmd.AtAPattern(n, n, csr.Indptr, csr.Indices);
                order = SparseAmd.Order(n, tp, ti);
                break;
            }
            default: order = SparseAmd.Order(n, sp, si); break;
        }
        var sym = SparseSymbolic.Analyze(n, sp, si, order);
        return new SparseLuFactorization(SupernodalLu.Factor(n, ap, ai, ax, rowOf, sym, diagPivotThresh));
    }

    /// <summary>
    /// The left-looking Gilbert–Peierls LU that <see cref="Splu"/> used before the multifrontal
    /// rewrite — kept as an independent reference: same contract, entirely different algorithm.
    /// </summary>
    /// <remarks>
    /// Column <c>k</c> of <c>L</c> and <c>U</c> comes from one sparse triangular solve
    /// <c>L x = A(:, q[k])</c>, whose non-zero pattern is found first by a depth-first search of
    /// <c>L</c>'s graph, so the work is proportional to flops, not to n (Gilbert &amp; Peierls 1988;
    /// the structure follows Davis's CSparse <c>cs_lu</c>). No supernodes: several times slower.
    /// </remarks>
    internal static SparseLuFactorization SpluGilbertPeierls(SparseMatrixBase a, ColumnOrdering permcSpec = ColumnOrdering.Colamd, double diagPivotThresh = 1.0)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Rows != a.Columns) throw new ArgumentException($"Splu needs a square matrix, got {a.Rows}x{a.Columns}.");
        if (diagPivotThresh < 0 || diagPivotThresh > 1) throw new ArgumentException("diagPivotThresh must be in [0, 1].");
        var n = a.Rows;
        var csc = a.ToCsc().SumDuplicates();
        int[] ap = csc.Indptr.ToArray(), ai = csc.Indices.ToArray();
        var ax = csc.Data.ToArray();
        var q = ColumnPermutation(csc, permcSpec);

        var cap = Math.Max(4 * ap[n] + n, 16);
        var lp = new int[n + 1]; var li = new int[cap]; var lx = new double[cap];
        var up = new int[n + 1]; var ui = new int[cap]; var ux = new double[cap];
        int lnz = 0, unz = 0;
        var pinv = new int[n];
        Array.Fill(pinv, -1);
        var x = new double[n];
        var xi = new int[n];
        var pstack = new int[n];
        var marked = new int[n];
        Array.Fill(marked, -1);

        for (var k = 0; k < n; k++)
        {
            lp[k] = lnz;
            up[k] = unz;
            if (lnz + n > li.Length) { Array.Resize(ref li, 2 * li.Length + n); Array.Resize(ref lx, li.Length); }
            if (unz + n > ui.Length) { Array.Resize(ref ui, 2 * ui.Length + n); Array.Resize(ref ux, ui.Length); }
            var col = q[k];

            // ---- symbolic: reach of A(:, col) in the graph of L, in topological order xi[top..n).
            var top = n;
            for (var p = ap[col]; p < ap[col + 1]; p++)
            {
                var start = ai[p];
                if (marked[start] == k) continue;
                // Non-recursive DFS (CSparse cs_dfs) — no recursion, so no stack overflow on long chains.
                var head = 0;
                xi[0] = start;
                while (head >= 0)
                {
                    var j = xi[head];
                    var jnew = pinv[j];
                    if (marked[j] != k)
                    {
                        marked[j] = k;
                        pstack[head] = jnew < 0 ? 0 : lp[jnew] + 1; // +1 skips L's own diagonal (row j)
                    }
                    var done = true;
                    var p2 = jnew < 0 ? 0 : (jnew < k ? lp[jnew + 1] : lnz);
                    for (var pp = pstack[head]; pp < p2; pp++)
                    {
                        var i = li[pp];
                        if (marked[i] == k) continue;
                        pstack[head] = pp;
                        xi[++head] = i;
                        done = false;
                        break;
                    }
                    if (done)
                    {
                        head--;
                        xi[--top] = j;
                    }
                }
            }

            // ---- numeric: x = L \ A(:, col) restricted to the reach.
            for (var p = top; p < n; p++) x[xi[p]] = 0;
            for (var p = ap[col]; p < ap[col + 1]; p++) x[ai[p]] = ax[p];
            for (var px = top; px < n; px++)
            {
                var j = xi[px];
                var jj = pinv[j];
                if (jj < 0) continue;
                var xj = x[j];
                if (xj == 0) continue;
                for (var p = lp[jj] + 1; p < lp[jj + 1]; p++) x[li[p]] -= lx[p] * xj;
            }

            // ---- pivot: largest |x_i| among rows not yet pivotal, diagonal preferred within the threshold.
            var ipiv = -1;
            var amax = -1.0;
            for (var p = top; p < n; p++)
            {
                var i = xi[p];
                if (pinv[i] < 0)
                {
                    var t = Math.Abs(x[i]);
                    if (t > amax) { amax = t; ipiv = i; }
                }
                else
                {
                    ui[unz] = pinv[i];
                    ux[unz++] = x[i];
                }
            }
            if (ipiv == -1 || amax <= 0)
                throw new InvalidOperationException($"Splu: matrix is singular (no usable pivot in column {col}).");
            if (pinv[col] < 0 && marked[col] == k && Math.Abs(x[col]) >= amax * diagPivotThresh && x[col] != 0) ipiv = col;
            var pivot = x[ipiv];
            ui[unz] = k;
            ux[unz++] = pivot;
            pinv[ipiv] = k;
            li[lnz] = ipiv;
            lx[lnz++] = 1.0;
            for (var p = top; p < n; p++)
            {
                var i = xi[p];
                if (pinv[i] < 0)
                {
                    li[lnz] = i;
                    lx[lnz++] = x[i] / pivot;
                }
                x[i] = 0;
            }
        }
        lp[n] = lnz;
        up[n] = unz;
        for (var p = 0; p < lnz; p++) li[p] = pinv[li[p]];
        Array.Resize(ref li, lnz); Array.Resize(ref lx, lnz);
        Array.Resize(ref ui, unz); Array.Resize(ref ux, unz);
        // Sort every column: L's diagonal then comes first and U's last, which the solves rely on.
        for (var j = 0; j < n; j++)
        {
            Array.Sort(li, lx, lp[j], lp[j + 1] - lp[j]);
            Array.Sort(ui, ux, up[j], up[j + 1] - up[j]);
        }
        return new SparseLuFactorization(n, lp, li, lx, up, ui, ux, pinv, q);
    }

    /// <summary>Solves <c>A x = b</c> by sparse LU (SciPy's <c>spsolve</c>).</summary>
    public static double[] Spsolve(SparseMatrixBase a, double[] b, ColumnOrdering permcSpec = ColumnOrdering.Colamd)
        => Splu(a, permcSpec).Solve(b);

    /// <summary>Solves <c>A X = B</c> for a dense right-hand side (vector or matrix).</summary>
    public static NdArray Spsolve(SparseMatrixBase a, NdArray b, ColumnOrdering permcSpec = ColumnOrdering.Colamd)
        => Splu(a, permcSpec).Solve(b);

    /// <summary>Solves <c>A X = B</c> for a sparse right-hand side; the result is sparse (CSC), exact zeros dropped.</summary>
    public static CscMatrix Spsolve(SparseMatrixBase a, SparseMatrixBase b, ColumnOrdering permcSpec = ColumnOrdering.Colamd)
    {
        var lu = Splu(a, permcSpec);
        return SolveColumns(lu, b.ToCsc(), a.Rows);
    }

    private static CscMatrix SolveColumns(SparseLuFactorization lu, CscMatrix b, int n)
    {
        var ptr = new List<int> { 0 };
        var idx = new List<int>();
        var val = new List<double>();
        var col = new double[n];
        var res = new double[n];
        for (var j = 0; j < b.Columns; j++)
        {
            Array.Clear(col);
            for (var p = b.Indptr[j]; p < b.Indptr[j + 1]; p++) col[b.Indices[p]] += b.Data[p];
            lu.SolveInto(col, res);
            for (var i = 0; i < n; i++) if (res[i] != 0) { idx.Add(i); val.Add(res[i]); }
            ptr.Add(idx.Count);
        }
        return new CscMatrix(val.ToArray(), idx.ToArray(), ptr.ToArray(), n, b.Columns, copy: false);
    }

    /// <summary>Returns a solver closure for repeated solves with the same matrix (SciPy's <c>factorized</c>).</summary>
    public static Func<double[], double[]> Factorized(SparseMatrixBase a)
    {
        var lu = Splu(a);
        return b => lu.Solve(b);
    }

    /// <summary>The sparse inverse, column by column from one LU (dense-ish in general — SciPy warns the same).</summary>
    public static CscMatrix Inv(SparseMatrixBase a)
    {
        if (a.Rows != a.Columns) throw new ArgumentException("Inv needs a square matrix.");
        return SolveColumns(Splu(a), SparseMatrices.Identity(a.Rows).ToCsc(), a.Rows);
    }

    // ------------------------------------------------------------------ triangular solves

    /// <summary>
    /// Solves <c>A x = b</c> for triangular sparse <c>A</c> (SciPy's <c>spsolve_triangular</c>).
    /// Entries outside the named triangle are ignored, as SciPy does.
    /// </summary>
    /// <param name="a">Square triangular matrix.</param>
    /// <param name="b">Right-hand side.</param>
    /// <param name="lower">Lower (forward substitution) or upper (back substitution).</param>
    /// <param name="unitDiagonal">Assume ones on the diagonal and ignore the stored diagonal.</param>
    public static double[] SpsolveTriangular(SparseMatrixBase a, double[] b, bool lower = true, bool unitDiagonal = false)
    {
        if (a.Rows != a.Columns) throw new ArgumentException("SpsolveTriangular needs a square matrix.");
        var n = a.Rows;
        if (b.Length != n) throw new ArgumentException($"b has {b.Length} entries, expected {n}.");
        var c = a.ToCsr().SumDuplicates();
        var x = (double[])b.Clone();
        var ptr = c.IndptrArray; var idx = c.IndicesArray; var val = c.DataArray;
        for (var t = 0; t < n; t++)
        {
            var i = lower ? t : n - 1 - t;
            var acc = x[i];
            var diag = 0.0;
            for (var p = ptr[i]; p < ptr[i + 1]; p++)
            {
                var j = idx[p];
                if (j == i) diag = val[p];
                else if (lower ? j < i : j > i) acc -= val[p] * x[j];
            }
            if (unitDiagonal) diag = 1.0;
            if (diag == 0) throw new InvalidOperationException($"A is singular: diagonal {i} is zero.");
            x[i] = acc / diag;
        }
        return x;
    }

    /// <summary>Triangular solve for a dense matrix right-hand side, one column at a time.</summary>
    public static NdArray SpsolveTriangular(SparseMatrixBase a, NdArray b, bool lower = true, bool unitDiagonal = false)
    {
        if (b.Rank == 1) return new NdArray(SpsolveTriangular(a, b.ToArray(), lower, unitDiagonal), b.Size);
        int n = b.Shape[0], k = b.Shape[1];
        var src = b.AsContiguous().ToArray();
        var outBuf = new double[n * k];
        var col = new double[n];
        for (var j = 0; j < k; j++)
        {
            for (var i = 0; i < n; i++) col[i] = src[i * k + j];
            var r = SpsolveTriangular(a, col, lower, unitDiagonal);
            for (var i = 0; i < n; i++) outBuf[i * k + j] = r[i];
        }
        return new NdArray(outBuf, n, k);
    }

    // ------------------------------------------------------------------ incomplete factorisations

    /// <summary>
    /// Threshold incomplete LU, ILUT(τ, p) (Saad 1994) — the role of SciPy's <c>spilu</c>.
    /// </summary>
    /// <param name="a">Square sparse matrix.</param>
    /// <param name="dropTol">Entries smaller than <c>dropTol · ‖row of A‖₂</c> are dropped.</param>
    /// <param name="fillFactor">Each row of L and of U keeps at most <c>fillFactor · nnz(row of A) / 2</c> entries (the largest).</param>
    /// <remarks>
    /// Row-wise IKJ elimination with no pivoting, so it can break down where SuperLU's ILUTP
    /// (which pivots) would not; a zero pivot is replaced by <c>(dropTol + 1e-4)·‖row‖</c> rather
    /// than failing, as SuperLU's ILU does with its <c>ILU_FillTol</c>. Results therefore differ
    /// from SciPy's in detail but serve the same purpose.
    /// </remarks>
    public static IncompleteLu Spilu(SparseMatrixBase a, double dropTol = 1e-4, double fillFactor = 10)
        => IncompleteFactor(a, dropTol, fillFactor, patternOnly: false);

    /// <summary>ILU(0): incomplete LU restricted to the sparsity pattern of <c>A</c>. Exact for tridiagonal matrices.</summary>
    public static IncompleteLu Ilu0(SparseMatrixBase a) => IncompleteFactor(a, 0, double.PositiveInfinity, patternOnly: true);

    private static IncompleteLu IncompleteFactor(SparseMatrixBase a, double dropTol, double fillFactor, bool patternOnly)
    {
        if (a.Rows != a.Columns) throw new ArgumentException("Incomplete LU needs a square matrix.");
        var n = a.Rows;
        var c = a.ToCsr().SumDuplicates();
        var ap = c.IndptrArray; var ai = c.IndicesArray; var av = c.DataArray;
        var lRows = new (int[] Idx, double[] Val)[n];
        var uRows = new (int[] Idx, double[] Val)[n];
        var uDiag = new double[n];
        var w = new double[n];
        var inRow = new int[n];
        Array.Fill(inRow, -1);
        var nz = new List<int>();
        var heap = new PriorityQueue<int, int>();
        for (var i = 0; i < n; i++)
        {
            nz.Clear();
            var rowNorm = 0.0;
            for (var p = ap[i]; p < ap[i + 1]; p++)
            {
                var j = ai[p];
                w[j] = av[p];
                inRow[j] = i;
                nz.Add(j);
                rowNorm += av[p] * av[p];
                if (j < i) heap.Enqueue(j, j);
            }
            rowNorm = Math.Sqrt(rowNorm);
            if (rowNorm == 0) rowNorm = 1;
            var tau = dropTol * rowNorm;
            var lastK = -1;
            while (heap.TryDequeue(out var k, out _))
            {
                if (k == lastK) continue;
                lastK = k;
                var wk = w[k] / uDiag[k];
                if (!patternOnly && Math.Abs(wk) < tau) { w[k] = 0; continue; }
                w[k] = wk;
                if (wk == 0) continue;
                var (uIdx, uVal) = uRows[k];
                for (var t = 0; t < uIdx.Length; t++)
                {
                    var j = uIdx[t];
                    if (inRow[j] != i)
                    {
                        if (patternOnly) continue;
                        inRow[j] = i;
                        w[j] = 0;
                        nz.Add(j);
                        if (j < i) heap.Enqueue(j, j);
                    }
                    w[j] -= wk * uVal[t];
                }
            }
            var keep = patternOnly ? int.MaxValue : (int)Math.Max(1, Math.Ceiling(fillFactor * (ap[i + 1] - ap[i]) / 2.0));
            var lower = new List<(int, double)>();
            var upper = new List<(int, double)>();
            var diag = 0.0;
            foreach (var j in nz)
            {
                var v = w[j];
                if (j == i) diag = v;
                else if (!patternOnly && Math.Abs(v) < tau) { }
                else if (v == 0 && !patternOnly) { }
                else if (j < i) lower.Add((j, v));
                else upper.Add((j, v));
                w[j] = 0;
            }
            if (lower.Count > keep) lower = lower.OrderByDescending(e => Math.Abs(e.Item2)).Take(keep).ToList();
            if (upper.Count > keep) upper = upper.OrderByDescending(e => Math.Abs(e.Item2)).Take(keep).ToList();
            lower.Sort((x, y) => x.Item1.CompareTo(y.Item1));
            upper.Sort((x, y) => x.Item1.CompareTo(y.Item1));
            if (diag == 0 || Math.Abs(diag) < Eps * rowNorm) diag = (diag < 0 ? -1 : 1) * (dropTol + 1e-4) * rowNorm;
            uDiag[i] = diag;
            lRows[i] = (lower.Select(e => e.Item1).ToArray(), lower.Select(e => e.Item2).ToArray());
            uRows[i] = (upper.Select(e => e.Item1).ToArray(), upper.Select(e => e.Item2).ToArray());
        }
        // Assemble CSR factors; U stores its diagonal first in each row.
        var lp = new int[n + 1]; var up = new int[n + 1];
        for (var i = 0; i < n; i++) { lp[i + 1] = lp[i] + lRows[i].Idx.Length; up[i + 1] = up[i] + uRows[i].Idx.Length + 1; }
        var li = new int[lp[n]]; var lx = new double[lp[n]]; var ui = new int[up[n]]; var ux = new double[up[n]];
        for (var i = 0; i < n; i++)
        {
            lRows[i].Idx.CopyTo(li, lp[i]); lRows[i].Val.CopyTo(lx, lp[i]);
            ui[up[i]] = i; ux[up[i]] = uDiag[i];
            uRows[i].Idx.CopyTo(ui, up[i] + 1); uRows[i].Val.CopyTo(ux, up[i] + 1);
        }
        return new IncompleteLu(new CsrMatrix(n, n, lp, li, lx, canonical: true), new CsrMatrix(n, n, up, ui, ux, canonical: true));
    }

    /// <summary>The Jacobi (diagonal) preconditioner <c>M = diag(A)⁻¹</c>. Zero diagonal entries are left as 1.</summary>
    public static LinearOperator JacobiPreconditioner(SparseMatrixBase a)
    {
        if (a.Rows != a.Columns) throw new ArgumentException("Jacobi preconditioner needs a square matrix.");
        var d = a.Diagonal();
        var inv = new double[d.Length];
        for (var i = 0; i < d.Length; i++) inv[i] = d[i] == 0 ? 1.0 : 1.0 / d[i];
        MatVecFunction f = (x, y) => { for (var i = 0; i < inv.Length; i++) y[i] = inv[i] * x[i]; };
        return new LinearOperator(d.Length, d.Length, f, f);
    }
}
