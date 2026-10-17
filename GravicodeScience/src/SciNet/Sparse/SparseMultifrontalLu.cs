namespace Gravicode.Science.SciNet.Sparse;

/// <summary>
/// Multifrontal LU with threshold partial pivoting and delayed pivots (the unsymmetric scheme of
/// MUMPS: Amestoy, Duff, L'Excellent &amp; Koster 2001), on the symmetrised pattern <c>A + Aᵀ</c>.
/// </summary>
/// <remarks>
/// <para>Each supernode is a dense front: its own (fully summed) rows and columns plus the border
/// rows/columns that later fronts eliminate. The front is assembled from the original entries and
/// the children's contribution blocks (extend-add), its fully summed block is factorised with
/// row interchanges restricted to fully summed rows, and the Schur complement is passed up.</para>
/// <para>A pivot is accepted only if it is at least <c>u</c> times the largest entry of its whole
/// front column, border rows included — so <c>u = 1</c> is genuine partial pivoting and every
/// multiplier is bounded by <c>1/u</c>, as with SuperLU. A column with no acceptable pivot is
/// <em>delayed</em>: it and an unused row are handed to the parent front as extra fully summed
/// variables. At a root everything is fully summed, so a column that still fails there means the
/// matrix is singular.</para>
/// <para>The fully summed block is factorised by blocks of <see cref="Nb"/> columns, left-looking
/// inside a block (a column is updated into a scratch copy, so a delayed column leaves the front
/// untouched) and right-looking across blocks: one triangular solve and one GEMM per block, which
/// is where the time goes.</para>
/// </remarks>
internal sealed unsafe class SupernodalLu
{
    private const int Nb = 32;

    internal int N;
    internal int Supernodes;
    // Per supernode: front size m, pivots p, offsets into the index and value stores.
    internal int[] FrontM = [], FrontP = [], IdxOff = [];
    internal long[] LOff = [], UOff = [];
    internal int[] RowIdx = [], ColIdx = []; // original row / column ids, final front order
    internal double[] Values = [];
    internal int Delayed; // total delayed pivots, for diagnostics

    /// <summary>Dense entries stored (L incl. unit diagonal + U incl. diagonal), amalgamation zeros included.</summary>
    internal long Entries
    {
        get
        {
            long t = 0;
            for (var s = 0; s < Supernodes; s++)
            {
                long m = FrontM[s], p = FrontP[s];
                t += m * p + p * (m - p);
            }
            return t;
        }
    }

    private long _nonZeros = -1;

    /// <summary>
    /// Non-zero entries of L (unit diagonal included) and U (diagonal included) — the count
    /// <see cref="Export"/> produces; explicit zeros from supernode amalgamation are not fill.
    /// </summary>
    internal long NonZeros
    {
        get
        {
            if (_nonZeros >= 0) return _nonZeros;
            long t = 0;
            for (var s = 0; s < Supernodes; s++)
            {
                int m = FrontM[s], p = FrontP[s];
                var lo = LOff[s];
                for (var i = 0; i < m; i++)
                    for (var c = 0; c < p; c++)
                        if (Values[lo + (long)i * p + c] != 0 || i == c) t++;
                var uo = UOff[s];
                var len = (long)p * (m - p);
                for (long q = 0; q < len; q++) if (Values[uo + q] != 0) t++;
                t += p; // L's unit diagonal (U's diagonal was counted in the L block)
            }
            return _nonZeros = t;
        }
    }

    /// <summary>
    /// Factorises <c>A</c> given as canonical CSC (<paramref name="ap"/>, <paramref name="ai"/>, <paramref name="ax"/>).
    /// <paramref name="rowOf"/>[j] is the original row placed at position j by the transversal (identity if none);
    /// the analysis is of that row-permuted matrix.
    /// </summary>
    internal static SupernodalLu Factor(int n, int[] ap, int[] ai, double[] ax, int[] rowOf, SparseSymbolic sym, double u)
    {
        var order = sym.Order;
        var vinv = sym.OrderInv;            // analysed index -> variable
        var posOfRow = new int[n];           // original row -> analysed index
        for (var j = 0; j < n; j++) posOfRow[rowOf[j]] = j;

        // Permuted matrix B = P (A with rows permuted) Pᵀ in variable space, as CSC and CSR.
        var nnz = ap[n];
        var bcp = new int[n + 1]; var bci = new int[nnz]; var bcx = new double[nnz];
        var brp = new int[n + 1]; var brj = new int[nnz]; var brx = new double[nnz];
        for (var j = 0; j < n; j++)
        {
            var cv = vinv[j];
            bcp[cv + 1] = ap[j + 1] - ap[j];
            for (var p = ap[j]; p < ap[j + 1]; p++) brp[vinv[posOfRow[ai[p]]] + 1]++;
        }
        for (var v = 0; v < n; v++) { bcp[v + 1] += bcp[v]; brp[v + 1] += brp[v]; }
        {
            var rpos = (int[])brp.Clone();
            for (var j = 0; j < n; j++)
            {
                var cv = vinv[j];
                var q = bcp[cv];
                for (var p = ap[j]; p < ap[j + 1]; p++)
                {
                    var rv = vinv[posOfRow[ai[p]]];
                    bci[q] = rv; bcx[q++] = ax[p];
                    var r = rpos[rv]++;
                    brj[r] = cv; brx[r] = ax[p];
                }
            }
        }

        var ns = sym.Count;
        var f = new SupernodalLu { N = n, Supernodes = ns };
        f.FrontM = new int[ns]; f.FrontP = new int[ns]; f.IdxOff = new int[ns + 1];
        f.LOff = new long[ns]; f.UOff = new long[ns];
        f._delayed = new int[ns];
        var rowIdx = new int[Math.Max(16, 2 * (n + sym.BorderIdx.Length))];
        var colIdx = new int[rowIdx.Length];
        var values = new double[Math.Max(16, 2 * sym.FactorEntries + 16)];
        long valTop = 0;
        var idxTop = 0;

        var rowLoc = new int[n]; var colLoc = new int[n];
        Array.Fill(rowLoc, -1); Array.Fill(colLoc, -1);
        double[] front = new double[1024];
        var cbStack = new double[1024];
        long cbTop = 0;
        var cbs = new Stack<(int Sn, long Off)>();
        var scratch = new double[2 * n + 2 * Nb + 64];
        var rmap = new int[n]; var cmap = new int[n];

        for (var s = 0; s < ns; s++)
        {
            int f0 = sym.SnStart[s], f1 = sym.SnStart[s + 1], nc = f1 - f0;
            int bp0 = sym.BorderPtr[s], bl = sym.BorderPtr[s + 1] - bp0;
            var nch = sym.SnChildren[s];
            // The children's contribution blocks are the top nch stack entries.
            var kids = new (int Sn, long Off)[nch];
            var delay = 0;
            for (var c = nch - 1; c >= 0; c--)
            {
                kids[c] = cbs.Pop();
                var cs = kids[c].Sn;
                // A child's delayed variables sit first in its contribution block.
                delay += f.DelayedOf(cs);
            }
            var nfs = nc + delay;
            var m = nfs + bl;
            if (idxTop + m > rowIdx.Length) { Array.Resize(ref rowIdx, 2 * rowIdx.Length + m); Array.Resize(ref colIdx, rowIdx.Length); }
            var io = idxTop;
            f.IdxOff[s] = io;
            var q = io;
            for (var v = f0; v < f1; v++) { rowIdx[q] = v; colIdx[q++] = v; }
            foreach (var (cs, _) in kids)
            {
                int co = f.IdxOff[cs], cp = f.FrontP[cs], dk = f.DelayedOf(cs);
                for (var t = 0; t < dk; t++) { rowIdx[q] = rowIdx[co + cp + t]; colIdx[q++] = colIdx[co + cp + t]; }
            }
            // Delayed rows/cols must be filled before the border so the counts line up.
            for (var t = 0; t < bl; t++) { rowIdx[q] = sym.BorderIdx[bp0 + t]; colIdx[q++] = sym.BorderIdx[bp0 + t]; }
            idxTop = q;
            for (var t = 0; t < m; t++) { rowLoc[rowIdx[io + t]] = t; colLoc[colIdx[io + t]] = t; }

            var size = (long)m * m;
            if (front.LongLength < size) front = new double[Math.Max(size, front.LongLength * 2)];
            Array.Clear(front, 0, (int)size);
            fixed (double* F = front)
            {
                // ---- assemble original entries
                for (var v = f0; v < f1; v++)
                {
                    var cl = colLoc[v];
                    for (var p = bcp[v]; p < bcp[v + 1]; p++)
                    {
                        var r = bci[p];
                        if (r >= f0) F[(long)rowLoc[r] * m + cl] += bcx[p];
                    }
                    var rl = (long)rowLoc[v] * m;
                    for (var p = brp[v]; p < brp[v + 1]; p++)
                    {
                        var c = brj[p];
                        if (c >= f1) F[rl + colLoc[c]] += brx[p];
                    }
                }
                // ---- extend-add the children's contribution blocks
                fixed (double* stk = cbStack)
                    foreach (var (cs, off) in kids)
                    {
                        int co = f.IdxOff[cs], cp = f.FrontP[cs], mc = f.FrontM[cs] - cp;
                        for (var t = 0; t < mc; t++) { rmap[t] = rowLoc[rowIdx[co + cp + t]]; cmap[t] = colLoc[colIdx[co + cp + t]]; }
                        var src = stk + off;
                        for (var i = 0; i < mc; i++)
                        {
                            var dst = F + (long)rmap[i] * m;
                            var row = src + (long)i * mc;
                            for (var t = 0; t < mc; t++) dst[cmap[t]] += row[t];
                        }
                    }
                if (nch > 0) cbTop = kids[0].Off;

                // ---- partial factorisation
                int p0;
                fixed (int* ri = &rowIdx[io], ci = &colIdx[io])
                fixed (double* sc = scratch)
                    p0 = PartialFactor(F, m, nfs, ri, ci, rowLoc, u, sc);
                f.FrontM[s] = m;
                f.FrontP[s] = p0;
                f.Delayed += nfs - p0;
                f._delayed[s] = nfs - p0;
                if (p0 < nfs && sym.SnParent[s] < 0)
                    throw new InvalidOperationException($"Splu: matrix is singular ({nfs - p0} column(s) without a usable pivot).");

                // ---- store L (m×p) and U12 (p×(m−p))
                var need = (long)m * p0 + (long)p0 * (m - p0);
                if (valTop + need > values.LongLength) Array.Resize(ref values, (int)Math.Min(Array.MaxLength, Math.Max(values.LongLength * 2, valTop + need)));
                f.LOff[s] = valTop;
                fixed (double* vs = values)
                {
                    var dst = vs + valTop;
                    for (var i = 0; i < m; i++)
                    {
                        var srow = F + (long)i * m;
                        for (var t = 0; t < p0; t++) dst[t] = srow[t];
                        dst += p0;
                    }
                    f.UOff[s] = valTop + (long)m * p0;
                    for (var i = 0; i < p0; i++)
                    {
                        var srow = F + (long)i * m + p0;
                        for (var t = 0; t < m - p0; t++) dst[t] = srow[t];
                        dst += m - p0;
                    }
                }
                valTop += need;

                // ---- push the contribution block
                var mcb = m - p0;
                if (mcb > 0)
                {
                    var cbSize = (long)mcb * mcb;
                    if (cbTop + cbSize > cbStack.LongLength) Array.Resize(ref cbStack, (int)Math.Max(cbStack.LongLength * 2, cbTop + cbSize));
                    fixed (double* stk = cbStack)
                    {
                        var dst = stk + cbTop;
                        for (var i = 0; i < mcb; i++)
                        {
                            var srow = F + (long)(p0 + i) * m + p0;
                            for (var t = 0; t < mcb; t++) dst[t] = srow[t];
                            dst += mcb;
                        }
                    }
                    cbs.Push((s, cbTop));
                    cbTop += cbSize;
                }
            }
            for (var t = 0; t < m; t++) { rowLoc[rowIdx[io + t]] = -1; colLoc[colIdx[io + t]] = -1; }
        }
        f.IdxOff[ns] = idxTop;
        // Variables -> original ids.
        for (var t = 0; t < idxTop; t++)
        {
            rowIdx[t] = rowOf[order[rowIdx[t]]];
            colIdx[t] = order[colIdx[t]];
        }
        Array.Resize(ref rowIdx, idxTop);
        Array.Resize(ref colIdx, idxTop);
        Array.Resize(ref values, (int)valTop);
        f.RowIdx = rowIdx; f.ColIdx = colIdx; f.Values = values;
        return f;
    }

    private int DelayedOf(int s) => _delayed is null ? 0 : _delayed[s];

    private int[]? _delayed;

    /// <summary>
    /// Partial LU of the m×m front F (row-major) with <paramref name="nfs"/> fully summed
    /// rows/columns. Returns the number of pivots p; rows/cols p..nfs−1 are delayed.
    /// </summary>
    private static int PartialFactor(double* F, int m, int nfs, int* rowIdx, int* colIdx, int[] rowLoc, double u, double* tmp)
    {
        var p = 0;
        var nfsEnd = nfs;
        while (p < nfsEnd)
        {
            var bs = p;
            while (p < nfsEnd && p - bs < Nb)
            {
                var j = p;
                // Left-looking update of column j by this block's pivots, into tmp[i − bs].
                var len = m - bs;
                for (var i = 0; i < len; i++) tmp[i] = F[(long)(bs + i) * m + j];
                var kb = j - bs;
                if (kb > 0)
                {
                    for (var r = 1; r < kb; r++)
                    {
                        var row = F + (long)(bs + r) * m + bs;
                        tmp[r] -= SparseDenseKernels.Dot(row, tmp, r);
                    }
                    for (var i = kb; i < len; i++)
                        tmp[i] -= SparseDenseKernels.Dot(F + (long)(bs + i) * m + bs, tmp, kb);
                }
                // Pivot search: candidates are the fully summed rows j..nfs−1; the threshold is
                // against the whole column, border rows included.
                var colMax = 0.0;
                for (var i = kb; i < len; i++) { var a = Math.Abs(tmp[i]); if (a > colMax) colMax = a; }
                var best = -1;
                var bestVal = 0.0;
                for (var i = j; i < nfs; i++) { var a = Math.Abs(tmp[i - bs]); if (a > bestVal) { bestVal = a; best = i; } }
                var diag = rowLoc[colIdx[j]];
                var piv = -1;
                if (diag >= j && diag < nfs && tmp[diag - bs] != 0 && Math.Abs(tmp[diag - bs]) >= u * colMax) piv = diag;
                else if (best >= 0 && bestVal > 0 && bestVal >= u * colMax) piv = best;
                if (piv < 0)
                {
                    // Delay: swap column j with the last undelayed fully summed column. The front
                    // itself was not modified (the update went to tmp), so it stays consistent.
                    var e = nfsEnd - 1;
                    if (e != j)
                    {
                        for (var i = 0; i < m; i++)
                        {
                            var row = F + (long)i * m;
                            (row[j], row[e]) = (row[e], row[j]);
                        }
                        (colIdx[j], colIdx[e]) = (colIdx[e], colIdx[j]);
                    }
                    nfsEnd--;
                    continue;
                }
                if (piv != j)
                {
                    var ra = F + (long)j * m;
                    var rb = F + (long)piv * m;
                    for (var t = 0; t < m; t++) (ra[t], rb[t]) = (rb[t], ra[t]);
                    (tmp[j - bs], tmp[piv - bs]) = (tmp[piv - bs], tmp[j - bs]);
                    (rowIdx[j], rowIdx[piv]) = (rowIdx[piv], rowIdx[j]);
                    rowLoc[rowIdx[j]] = j;
                    rowLoc[rowIdx[piv]] = piv;
                }
                var pv = tmp[kb];
                var inv = 1.0 / pv;
                for (var i = 0; i <= kb; i++) F[(long)(bs + i) * m + j] = tmp[i];
                for (var i = kb + 1; i < len; i++) F[(long)(bs + i) * m + j] = tmp[i] * inv;
                p++;
            }
            var be = p;
            var nb = be - bs;
            if (nb > 0 && be < m)
            {
                // U rows of the block for the remaining columns, then the trailing update.
                var w = m - be;
                for (var r = 1; r < nb; r++)
                {
                    var dst = F + (long)(bs + r) * m + be;
                    var lrow = F + (long)(bs + r) * m + bs;
                    for (var t = 0; t < r; t++)
                    {
                        var l = lrow[t];
                        if (l != 0) SparseDenseKernels.AxpySub(l, F + (long)(bs + t) * m + be, dst, w);
                    }
                }
                SparseDenseKernels.GemmSub(w, w, nb, F + (long)be * m + bs, m, F + (long)bs * m + be, m, F + (long)be * m + be, m);
            }
        }
        return p;
    }

    // ------------------------------------------------------------------ solves

    /// <summary>Solves <c>A x = b</c>; <paramref name="b"/> and <paramref name="x"/> may not alias.</summary>
    internal void Solve(ReadOnlySpan<double> b, Span<double> x)
    {
        var y = b.ToArray(); // indexed by original row
        var tmp = new double[MaxFront() + 1];
        fixed (double* vs = Values)
        fixed (double* yp = y)
        fixed (double* tp = tmp)
        {
            // Forward: L y = Pr b, front by front.
            for (var s = 0; s < Supernodes; s++)
            {
                int m = FrontM[s], p = FrontP[s], io = IdxOff[s];
                if (p == 0) continue;
                var L = vs + LOff[s];
                for (var r = 0; r < p; r++)
                {
                    var acc = yp[RowIdx[io + r]] - SparseDenseKernels.Dot(L + (long)r * p, tp, r);
                    tp[r] = acc;
                    yp[RowIdx[io + r]] = acc;
                }
                for (var i = p; i < m; i++)
                    yp[RowIdx[io + i]] -= SparseDenseKernels.Dot(L + (long)i * p, tp, p);
            }
            // Backward: U x = y, fronts in reverse.
            var xs = new double[N];
            fixed (double* xp = xs)
                for (var s = Supernodes - 1; s >= 0; s--)
                {
                    int m = FrontM[s], p = FrontP[s], io = IdxOff[s];
                    if (p == 0) continue;
                    var L = vs + LOff[s];
                    var U = vs + UOff[s];
                    var w = m - p;
                    for (var t = 0; t < w; t++) tp[t] = xp[ColIdx[io + p + t]];
                    for (var r = p - 1; r >= 0; r--)
                    {
                        var acc = yp[RowIdx[io + r]] - SparseDenseKernels.Dot(U + (long)r * w, tp, w);
                        var lr = L + (long)r * p;
                        for (var c = r + 1; c < p; c++) acc -= lr[c] * xp[ColIdx[io + c]];
                        xp[ColIdx[io + r]] = acc / lr[r];
                    }
                }
            xs.CopyTo(x);
        }
    }

    /// <summary>Solves <c>Aᵀ x = b</c>.</summary>
    internal void SolveTranspose(ReadOnlySpan<double> b, Span<double> x)
    {
        var w = b.ToArray(); // indexed by original column
        var tmp = new double[MaxFront() + 1];
        var res = new double[N];
        fixed (double* vs = Values)
        fixed (double* wp = w)
        fixed (double* tp = tmp)
        fixed (double* rp = res)
        {
            // Uᵀ z = b (forward over fronts): pivot r's unknown is z at column ColIdx[r].
            for (var s = 0; s < Supernodes; s++)
            {
                int m = FrontM[s], p = FrontP[s], io = IdxOff[s];
                if (p == 0) continue;
                var L = vs + LOff[s];
                var U = vs + UOff[s];
                var wd = m - p;
                for (var r = 0; r < p; r++)
                {
                    var acc = wp[ColIdx[io + r]];
                    for (var c = 0; c < r; c++) acc -= L[(long)c * p + r] * tp[c];
                    tp[r] = acc / L[(long)r * p + r];
                }
                for (var r = 0; r < p; r++)
                {
                    var z = tp[r];
                    wp[ColIdx[io + r]] = z;
                    if (z == 0) continue;
                    var urow = U + (long)r * wd;
                    for (var t = 0; t < wd; t++) wp[ColIdx[io + p + t]] -= urow[t] * z;
                }
            }
            // Lᵀ v = z (backward): unknown of pivot r lives at row RowIdx[r].
            for (var s = Supernodes - 1; s >= 0; s--)
            {
                int m = FrontM[s], p = FrontP[s], io = IdxOff[s];
                if (p == 0) continue;
                var L = vs + LOff[s];
                for (var t = 0; t < p; t++) tp[t] = wp[ColIdx[io + t]];
                for (var i = p; i < m; i++)
                {
                    var xi = rp[RowIdx[io + i]];
                    if (xi == 0) continue;
                    SparseDenseKernels.AxpySub(xi, L + (long)i * p, tp, p);
                }
                for (var r = p - 1; r >= 0; r--)
                {
                    var acc = tp[r];
                    for (var i = r + 1; i < p; i++) acc -= L[(long)i * p + r] * rp[RowIdx[io + i]];
                    rp[RowIdx[io + r]] = acc;
                }
            }
        }
        res.CopyTo(x);
    }

    private int _maxFront = -1;

    private int MaxFront()
    {
        if (_maxFront < 0)
        {
            var mx = 0;
            for (var s = 0; s < Supernodes; s++) mx = Math.Max(mx, FrontM[s]);
            _maxFront = mx;
        }
        return _maxFront;
    }

    /// <summary>
    /// The factors in SciPy's convention: <c>PermR</c> (original row → pivot position),
    /// <c>PermC</c> (original column → pivot position) and CSC <c>L</c> (unit lower) and <c>U</c>.
    /// </summary>
    internal (int[] PermR, int[] PermC, CscMatrix L, CscMatrix U) Export()
    {
        var permR = new int[N];
        var permC = new int[N];
        var k = 0;
        for (var s = 0; s < Supernodes; s++)
        {
            int p = FrontP[s], io = IdxOff[s];
            for (var r = 0; r < p; r++) { permR[RowIdx[io + r]] = k; permC[ColIdx[io + r]] = k; k++; }
        }
        var lr = new List<int>(); var lc = new List<int>(); var lv = new List<double>();
        var ur = new List<int>(); var uc = new List<int>(); var uv = new List<double>();
        k = 0;
        for (var s = 0; s < Supernodes; s++)
        {
            int m = FrontM[s], p = FrontP[s], io = IdxOff[s];
            var lo = LOff[s];
            var uo = UOff[s];
            for (var r = 0; r < p; r++)
            {
                var kk = k + r;
                lr.Add(kk); lc.Add(kk); lv.Add(1.0);
                for (var i = r + 1; i < m; i++)
                {
                    var v = Values[lo + (long)i * p + r];
                    if (v != 0) { lr.Add(permR[RowIdx[io + i]]); lc.Add(kk); lv.Add(v); }
                }
                for (var c = r; c < p; c++)
                {
                    var v = Values[lo + (long)r * p + c];
                    if (v != 0 || c == r) { ur.Add(kk); uc.Add(permC[ColIdx[io + c]]); uv.Add(v); }
                }
                for (var c = p; c < m; c++)
                {
                    var v = Values[uo + (long)r * (m - p) + (c - p)];
                    if (v != 0) { ur.Add(kk); uc.Add(permC[ColIdx[io + c]]); uv.Add(v); }
                }
            }
            k += p;
        }
        var l = CsrMatrix.FromTriplets(N, N, lr.ToArray(), lc.ToArray(), lv.ToArray()).ToCsc();
        var u = CsrMatrix.FromTriplets(N, N, ur.ToArray(), uc.ToArray(), uv.ToArray()).ToCsc();
        return (permR, permC, l, u);
    }
}
