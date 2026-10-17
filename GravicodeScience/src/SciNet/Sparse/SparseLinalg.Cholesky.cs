using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Sparse;

/// <summary>
/// A sparse Cholesky factorisation <c>P A Pᵀ = L Lᵀ</c> from <see cref="SparseLinalg.Cholesky"/>.
/// </summary>
/// <remarks>
/// <c>Perm[k]</c> is the original index placed at position k, so <c>A[Perm][:, Perm] = L Lᵀ</c>
/// (CHOLMOD's and scikit-sparse's convention). Its <c>MatVec</c> applies <c>A⁻¹</c>, so it can be
/// used directly as an exact preconditioner or a shift-invert operator.
/// </remarks>
public sealed class SparseCholeskyFactorization : ILinearOperator
{
    private readonly SupernodalCholesky _f;

    internal SparseCholeskyFactorization(SupernodalCholesky f) => _f = f;

    /// <summary>Order of the factorised matrix.</summary>
    public int N => _f.N;

    int ILinearOperator.Rows => N;
    int ILinearOperator.Columns => N;

    /// <summary>The fill-reducing permutation: position k holds original index <c>Perm[k]</c>.</summary>
    public int[] Perm => (int[])_f.Order.Clone();

    /// <summary>Lower-triangular factor (CSC) of the permuted matrix; explicit zeros dropped.</summary>
    public CscMatrix L => _f.ExportL();

    /// <summary>Non-zeros of <see cref="L"/> (supernode amalgamation zeros not counted).</summary>
    public int Nnz => (int)Math.Min(int.MaxValue, _f.NonZeros);

    /// <summary><c>log det A = 2 Σ log Lⱼⱼ</c> — finite where the determinant itself would overflow.</summary>
    public double LogDeterminant() => _f.LogDeterminant();

    /// <summary>Solves <c>A x = b</c>.</summary>
    public double[] Solve(ReadOnlySpan<double> b)
    {
        if (b.Length != N) throw new ArgumentException($"b has {b.Length} entries, expected {N}.");
        var x = new double[N];
        _f.Solve(b, x);
        return x;
    }

    /// <summary>Solves <c>A X = B</c> column by column for a dense rank 2 (or rank 1) <paramref name="b"/>.</summary>
    public NdArray Solve(NdArray b)
    {
        if (b.Rank == 1) return new NdArray(Solve(b.AsContiguous().AsSpan()), N);
        if (b.Rank != 2 || b.Shape[0] != N) throw new ArgumentException($"B must have {N} rows.");
        var k = b.Shape[1];
        var src = b.AsContiguous().ToArray();
        var outBuf = new double[N * k];
        var col = new double[N];
        var res = new double[N];
        for (var j = 0; j < k; j++)
        {
            for (var i = 0; i < N; i++) col[i] = src[i * k + j];
            _f.Solve(col, res);
            for (var i = 0; i < N; i++) outBuf[i * k + j] = res[i];
        }
        return new NdArray(outBuf, N, k);
    }

    /// <summary>Applies <c>A⁻¹</c>.</summary>
    public void MatVec(ReadOnlySpan<double> x, Span<double> y) => _f.Solve(x, y);

    /// <summary>Applies <c>A⁻ᵀ = A⁻¹</c>.</summary>
    public void RMatVec(ReadOnlySpan<double> x, Span<double> y) => _f.Solve(x, y);
}

/// <summary>Thrown by <see cref="SparseLinalg.Cholesky"/> when the matrix is not positive definite.</summary>
public sealed class SparseNotPositiveDefiniteException(string message, int column) : InvalidOperationException(message)
{
    /// <summary>The (permuted) column at which a non-positive pivot appeared.</summary>
    public int Column { get; } = column;
}

/// <summary>
/// Supernodal multifrontal Cholesky (Liu 1992): the same symbolic analysis as the LU, no
/// pivoting, lower triangles only, so about half the flops and memory of LU on an SPD matrix.
/// </summary>
internal sealed unsafe class SupernodalCholesky
{
    private const int Nb = 48;

    internal int N;
    internal int[] Order = [];
    internal SparseSymbolic Sym = null!;
    internal long[] LOff = [];
    internal double[] Values = [];

    internal static SupernodalCholesky Factor(int n, int[] ap, int[] ai, double[] ax, SparseSymbolic sym)
    {
        var vinv = sym.OrderInv;
        // Lower triangle of P A Pᵀ in variable space, CSC.
        var cp = new int[n + 1];
        for (var j = 0; j < n; j++)
            for (var p = ap[j]; p < ap[j + 1]; p++)
                if (vinv[ai[p]] >= vinv[j]) cp[vinv[j] + 1]++;
        for (var v = 0; v < n; v++) cp[v + 1] += cp[v];
        var ci = new int[cp[n]]; var cx = new double[cp[n]];
        var pos = (int[])cp.Clone();
        for (var j = 0; j < n; j++)
        {
            var vj = vinv[j];
            for (var p = ap[j]; p < ap[j + 1]; p++)
            {
                var vi = vinv[ai[p]];
                if (vi < vj) continue;
                var q = pos[vj]++;
                ci[q] = vi; cx[q] = ax[p];
            }
        }

        var ns = sym.Count;
        var f = new SupernodalCholesky { N = n, Order = sym.Order, Sym = sym, LOff = new long[ns + 1] };
        long total = 0;
        for (var s = 0; s < ns; s++)
        {
            f.LOff[s] = total;
            long nc = sym.SnStart[s + 1] - sym.SnStart[s];
            total += (nc + sym.BorderPtr[s + 1] - sym.BorderPtr[s]) * nc;
        }
        f.LOff[ns] = total;
        var values = new double[total];
        var loc = new int[n];
        var map = new int[n];
        var front = new double[1024];
        var cbStack = new double[1024];
        long cbTop = 0;
        var cbs = new Stack<(int Sn, long Off)>();
        var tmp = new double[n + 64];
        var bt = new double[1024];

        for (var s = 0; s < ns; s++)
        {
            int f0 = sym.SnStart[s], f1 = sym.SnStart[s + 1], nc = f1 - f0;
            int bp0 = sym.BorderPtr[s], bl = sym.BorderPtr[s + 1] - bp0;
            var m = nc + bl;
            for (var t = 0; t < nc; t++) loc[f0 + t] = t;
            for (var t = 0; t < bl; t++) loc[sym.BorderIdx[bp0 + t]] = nc + t;
            var size = (long)m * m;
            if (front.LongLength < size) front = new double[Math.Max(size, front.LongLength * 2)];
            // Only the lower triangle is ever read.
            fixed (double* F = front)
            {
                for (var i = 0; i < m; i++) new Span<double>(F + (long)i * m, i + 1).Clear();
                for (var v = f0; v < f1; v++)
                {
                    var cl = v - f0;
                    for (var p = cp[v]; p < cp[v + 1]; p++) F[(long)loc[ci[p]] * m + cl] += cx[p];
                }
                var nch = sym.SnChildren[s];
                if (nch > 0)
                {
                    var kids = new (int Sn, long Off)[nch];
                    for (var c = nch - 1; c >= 0; c--) kids[c] = cbs.Pop();
                    fixed (double* stk = cbStack)
                        foreach (var (cs, off) in kids)
                        {
                            int cb0 = sym.BorderPtr[cs], mc = sym.BorderPtr[cs + 1] - cb0;
                            for (var t = 0; t < mc; t++) map[t] = loc[sym.BorderIdx[cb0 + t]];
                            var src = stk + off;
                            for (var i = 0; i < mc; i++)
                            {
                                var dst = F + (long)map[i] * m;
                                var row = src + (long)i * mc;
                                for (var t = 0; t <= i; t++) dst[map[t]] += row[t];
                            }
                        }
                    cbTop = kids[0].Off;
                }

                // ---- partial Cholesky of the nc fully summed columns
                for (var bs = 0; bs < nc; bs += Nb)
                {
                    var be = Math.Min(nc, bs + Nb);
                    fixed (double* tp = tmp)
                        for (var j = bs; j < be; j++)
                        {
                            var kb = j - bs;
                            var lj = F + (long)j * m + bs;
                            for (var i = j; i < m; i++)
                            {
                                var v = F[(long)i * m + j];
                                if (kb > 0) v -= SparseDenseKernels.Dot(F + (long)i * m + bs, lj, kb);
                                tp[i - j] = v;
                            }
                            var d = tp[0];
                            if (!(d > 0))
                                throw new SparseNotPositiveDefiniteException($"Cholesky: the matrix is not positive definite (pivot {d:G3} at position {f0 + j}).", f0 + j);
                            var ljj = Math.Sqrt(d);
                            var inv = 1.0 / ljj;
                            F[(long)j * m + j] = ljj;
                            for (var i = j + 1; i < m; i++) F[(long)i * m + j] = tp[i - j] * inv;
                        }
                    var kblk = be - bs;
                    var w = m - be;
                    if (w <= 0) continue;
                    // Trailing lower update: C −= L₂ L₂ᵀ with L₂ = F[be:m, bs:be], via a packed L₂ᵀ.
                    if (bt.Length < (long)kblk * w) bt = new double[Math.Max((long)kblk * w, bt.LongLength * 2)];
                    fixed (double* B = bt)
                    {
                        for (var c = 0; c < w; c++)
                        {
                            var srow = F + (long)(be + c) * m + bs;
                            for (var t = 0; t < kblk; t++) B[(long)t * w + c] = srow[t];
                        }
                        SyrkLower(w, kblk, F + (long)be * m + bs, m, B, w, F + (long)be * m + be, m);
                    }
                }

                // ---- store L (m × nc) and push the contribution block (lower triangle, packed full rows)
                fixed (double* vs = values)
                {
                    var dst = vs + f.LOff[s];
                    for (var i = 0; i < m; i++)
                    {
                        var srow = F + (long)i * m;
                        var cnt = Math.Min(i + 1, nc);
                        for (var t = 0; t < cnt; t++) dst[t] = srow[t];
                        for (var t = cnt; t < nc; t++) dst[t] = 0;
                        dst += nc;
                    }
                }
                if (bl > 0)
                {
                    var cbSize = (long)bl * bl;
                    if (cbTop + cbSize > cbStack.LongLength) Array.Resize(ref cbStack, (int)Math.Max(cbStack.LongLength * 2, cbTop + cbSize));
                    fixed (double* stk = cbStack)
                    {
                        var dst = stk + cbTop;
                        for (var i = 0; i < bl; i++)
                        {
                            var srow = F + (long)(nc + i) * m + nc;
                            for (var t = 0; t <= i; t++) dst[t] = srow[t];
                            dst += bl;
                        }
                    }
                    cbs.Push((s, cbTop));
                    cbTop += cbSize;
                }
            }
        }
        f.Values = values;
        return f;
    }

    /// <summary>Lower triangle of <c>C −= A B</c> (B = Aᵀ packed), in row strips so little of the upper half is computed.</summary>
    private static void SyrkLower(int w, int k, double* a, int lda, double* b, int ldb, double* c, int ldc)
    {
        const int Strip = 64;
        var strips = (w + Strip - 1) / Strip;
        var work = (long)w * w * k / 2;
        if (work >= SparseDenseKernels.ParallelGemmThreshold && strips > 1 && Environment.ProcessorCount > 1)
        {
            nint pa = (nint)a, pb = (nint)b, pc = (nint)c;
            // Later strips are longer: hand them out from the end so the big ones start first.
            Parallel.For(0, strips, t =>
            {
                var s = strips - 1 - t;
                var r0 = s * Strip;
                var r1 = Math.Min(w, r0 + Strip);
                SparseDenseKernels.GemmSub(r1 - r0, r1, k, (double*)pa + (long)r0 * lda, lda, (double*)pb, ldb, (double*)pc + (long)r0 * ldc, ldc, allowParallel: false);
            });
            return;
        }
        for (var r0 = 0; r0 < w; r0 += Strip)
        {
            var r1 = Math.Min(w, r0 + Strip);
            SparseDenseKernels.GemmSub(r1 - r0, r1, k, a + (long)r0 * lda, lda, b, ldb, c + (long)r0 * ldc, ldc, allowParallel: false);
        }
    }

    internal void Solve(ReadOnlySpan<double> b, Span<double> x)
    {
        var sym = Sym;
        var y = new double[N];
        for (var v = 0; v < N; v++) y[v] = b[Order[v]];
        var tmp = new double[N + 1];
        fixed (double* vs = Values)
        fixed (double* yp = y)
        fixed (double* tp = tmp)
        {
            var ns = sym.Count;
            for (var s = 0; s < ns; s++)
            {
                int f0 = sym.SnStart[s], nc = sym.SnStart[s + 1] - f0, bp0 = sym.BorderPtr[s], bl = sym.BorderPtr[s + 1] - bp0;
                var L = vs + LOff[s];
                for (var r = 0; r < nc; r++)
                {
                    var lr = L + (long)r * nc;
                    yp[f0 + r] = (yp[f0 + r] - SparseDenseKernels.Dot(lr, yp + f0, r)) / lr[r];
                }
                for (var i = 0; i < bl; i++)
                    yp[sym.BorderIdx[bp0 + i]] -= SparseDenseKernels.Dot(L + (long)(nc + i) * nc, yp + f0, nc);
            }
            for (var s = ns - 1; s >= 0; s--)
            {
                int f0 = sym.SnStart[s], nc = sym.SnStart[s + 1] - f0, bp0 = sym.BorderPtr[s], bl = sym.BorderPtr[s + 1] - bp0;
                var L = vs + LOff[s];
                for (var t = 0; t < nc; t++) tp[t] = yp[f0 + t];
                for (var i = 0; i < bl; i++)
                {
                    var xi = yp[sym.BorderIdx[bp0 + i]];
                    if (xi != 0) SparseDenseKernels.AxpySub(xi, L + (long)(nc + i) * nc, tp, nc);
                }
                for (var r = nc - 1; r >= 0; r--)
                {
                    var acc = tp[r];
                    for (var i = r + 1; i < nc; i++) acc -= L[(long)i * nc + r] * yp[f0 + i];
                    yp[f0 + r] = acc / L[(long)r * nc + r];
                }
            }
        }
        for (var v = 0; v < N; v++) x[Order[v]] = y[v];
    }

    internal double LogDeterminant()
    {
        var s = 0.0;
        for (var sn = 0; sn < Sym.Count; sn++)
        {
            int nc = Sym.SnStart[sn + 1] - Sym.SnStart[sn];
            for (var r = 0; r < nc; r++) s += Math.Log(Values[LOff[sn] + (long)r * nc + r]);
        }
        return 2 * s;
    }

    private long _nonZeros = -1;

    internal long NonZeros
    {
        get
        {
            if (_nonZeros >= 0) return _nonZeros;
            long t = 0;
            for (var sn = 0; sn < Sym.Count; sn++)
                for (var q = LOff[sn]; q < LOff[sn + 1]; q++) if (Values[q] != 0) t++;
            return _nonZeros = t;
        }
    }

    internal CscMatrix ExportL()
    {
        var rows = new List<int>(); var cols = new List<int>(); var vals = new List<double>();
        for (var s = 0; s < Sym.Count; s++)
        {
            int f0 = Sym.SnStart[s], nc = Sym.SnStart[s + 1] - f0, bp0 = Sym.BorderPtr[s], bl = Sym.BorderPtr[s + 1] - bp0;
            var o = LOff[s];
            for (var i = 0; i < nc + bl; i++)
            {
                var row = i < nc ? f0 + i : Sym.BorderIdx[bp0 + i - nc];
                for (var c = 0; c < Math.Min(i + 1, nc); c++)
                {
                    var v = Values[o + (long)i * nc + c];
                    if (v != 0) { rows.Add(row); cols.Add(f0 + c); vals.Add(v); }
                }
            }
        }
        return CsrMatrix.FromTriplets(N, N, rows.ToArray(), cols.ToArray(), vals.ToArray()).ToCsc();
    }
}

public static partial class SparseLinalg
{
    /// <summary>
    /// Sparse Cholesky factorisation <c>P A Pᵀ = L Lᵀ</c> of a symmetric positive definite matrix
    /// (the role of CHOLMOD / <c>scikit-sparse</c>'s <c>cholesky</c>; SciPy itself has none).
    /// </summary>
    /// <param name="a">Symmetric positive definite matrix; both triangles must be stored.</param>
    /// <param name="ordering">
    /// Fill-reducing order: <see cref="ColumnOrdering.Natural"/>, or approximate minimum degree on
    /// the pattern (any other value).
    /// </param>
    /// <remarks>
    /// Supernodal multifrontal: the analysis of <see cref="Splu"/> (AMD, elimination tree,
    /// relaxed supernodes) with dense Cholesky on each front and SYRK-shaped updates. About half
    /// the work and storage of LU and no pivoting, so prefer it whenever the matrix is SPD. Throws
    /// <see cref="ArgumentException"/> if <paramref name="a"/> is not symmetric (to 1e-12 relative)
    /// and <see cref="SparseNotPositiveDefiniteException"/> on a non-positive pivot.
    /// </remarks>
    public static SparseCholeskyFactorization Cholesky(SparseMatrixBase a, ColumnOrdering ordering = ColumnOrdering.MmdAtPlusA)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Rows != a.Columns) throw new ArgumentException($"Cholesky needs a square matrix, got {a.Rows}x{a.Columns}.");
        var n = a.Rows;
        var csc = a.ToCsc().SumDuplicates();
        int[] ap = csc.Indptr.ToArray(), ai = csc.Indices.ToArray();
        var ax = csc.Data.ToArray();
        if (!IsNumericallySymmetric(csc, 1e-12)) throw new ArgumentException("Cholesky needs a symmetric matrix (both triangles stored).");
        var (sp, si) = SparseAmd.SymmetricPattern(n, ap, ai);
        var order = ordering == ColumnOrdering.Natural ? Identity(n) : SparseAmd.Order(n, sp, si);
        var sym = SparseSymbolic.Analyze(n, sp, si, order);
        return new SparseCholeskyFactorization(SupernodalCholesky.Factor(n, ap, ai, ax, sym));
    }

    /// <summary>True when <c>|a_ij − a_ji| ≤ rtol · max|a|</c> for every stored pair.</summary>
    internal static bool IsNumericallySymmetric(CscMatrix c, double rtol)
    {
        var n = c.Columns;
        var max = 0.0;
        foreach (var v in c.Data) max = Math.Max(max, Math.Abs(v));
        var tol = rtol * max;
        // A as CSR against Aᵀ as CSR (the CSC arrays of A read row-wise), merged entry by entry.
        var a = c.ToCsr();
        var at = c.Inner;
        var ap = a.Indptr; var ai = a.Indices; var ax = a.Data;
        var bp = at.Indptr; var bi = at.Indices; var bx = at.Data;
        for (var i = 0; i < n; i++)
        {
            int p = ap[i], q = bp[i], pe = ap[i + 1], qe = bp[i + 1];
            while (p < pe || q < qe)
            {
                var ja = p < pe ? ai[p] : int.MaxValue;
                var jb = q < qe ? bi[q] : int.MaxValue;
                double va = 0, vb = 0;
                if (ja <= jb) va = ax[p++];
                if (jb <= ja) vb = bx[q++];
                if (Math.Abs(va - vb) > tol) return false;
            }
        }
        return true;
    }
}
