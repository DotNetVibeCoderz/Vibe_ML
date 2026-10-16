using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Sparse;

public static partial class SparseLinalg
{
    // ------------------------------------------------------------------ norms and powers

    /// <summary>Matrix norm of a sparse matrix (SciPy's <c>scipy.sparse.linalg.norm</c>; Frobenius, 1, ∞ and their minus variants).</summary>
    public static double Norm(SparseMatrixBase a, SparseNormKind kind = SparseNormKind.Frobenius) => a.Norm(kind);

    /// <summary>The spectral norm <c>‖A‖₂</c> (largest singular value), via <see cref="Svds"/>; SciPy's <c>norm(A, 2)</c>.</summary>
    public static double SpectralNorm(ILinearOperator a)
    {
        if (Math.Min(a.Rows, a.Columns) <= 2)
        {
            var dense = a.ToDenseArray();
            return Decomposition.SingularValues(dense).At(0);
        }
        return Svds(a, 1).S[0];
    }

    /// <summary><c>Aᵖ</c> by repeated squaring with sparse products (SciPy's <c>matrix_power</c>); <c>A⁰ = I</c>.</summary>
    public static CsrMatrix MatrixPower(SparseMatrixBase a, int power)
    {
        if (a.Rows != a.Columns) throw new ArgumentException("MatrixPower needs a square matrix.");
        if (power < 0) throw new ArgumentException("Negative powers are not supported (they need an inverse).");
        var result = SparseMatrices.Identity(a.Rows);
        var b = a.ToCsr();
        var first = true;
        while (power > 0)
        {
            if ((power & 1) == 1) { result = first ? b : result.MatMul(b); first = false; }
            power >>= 1;
            if (power > 0) b = b.MatMul(b);
        }
        return result;
    }

    // ------------------------------------------------------------------ 1-norm estimation

    /// <summary>
    /// Estimates <c>‖A‖₁</c> from products with <c>A</c> and <c>Aᵀ</c> only — SciPy's <c>onenormest</c>.
    /// </summary>
    /// <param name="a">Square operator (needs <c>RMatVec</c>).</param>
    /// <param name="t">Number of columns in the block (2 is SciPy's default); larger is more reliable.</param>
    /// <param name="itmax">Maximum iterations (at least 2).</param>
    /// <remarks>
    /// The block algorithm of Higham &amp; Tisseur (2000, Algorithm 2.4). The estimate is always a
    /// lower bound and in practice almost always exact; it is computed exactly (n products) when
    /// n ≤ 8 or n ≤ t. The random ±1 columns use a fixed seed, so the estimate is reproducible.
    /// </remarks>
    public static double Onenormest(ILinearOperator a, int t = 2, int itmax = 5)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Rows != a.Columns) throw new ArgumentException("Onenormest needs a square operator.");
        var n = a.Rows;
        if (t < 1) throw new ArgumentException("t must be at least 1.");
        if (itmax < 2) throw new ArgumentException("itmax must be at least 2.");
        if (n == 0) return 0;
        if (n <= Math.Max(8, t)) return ExactOneNorm(a);
        var rng = new GraviRandom(1234);
        var x = new double[t][];
        for (var j = 0; j < t; j++)
        {
            x[j] = new double[n];
            for (var i = 0; i < n; i++) x[j][i] = j == 0 ? 1.0 : (rng.NextDouble() < 0.5 ? -1.0 : 1.0);
        }
        ResampleParallelColumns(x, null, rng);
        foreach (var col in x) ScaleInPlace(col, 1.0 / n);
        var y = new double[t][];
        var s = new double[t][];
        double[][]? sOld = null;
        for (var j = 0; j < t; j++) { y[j] = new double[n]; s[j] = new double[n]; }
        var z = new double[n];
        var h = new double[n];
        var history = new HashSet<int>();
        double estOld = 0, est = 0;
        var indBest = -1;
        int[]? ind = null;
        for (var k = 1; ; k++)
        {
            for (var j = 0; j < t; j++) a.MatVec(x[j], y[j]);
            var jBest = 0;
            est = 0;
            for (var j = 0; j < t; j++)
            {
                var c = 0.0;
                foreach (var v in y[j]) c += Math.Abs(v);
                if (c > est) { est = c; jBest = j; }
            }
            if (est > estOld || k == 2) indBest = ind is null ? jBest : ind[jBest];
            if (k >= 2 && est <= estOld) { est = estOld; break; }
            estOld = est;
            if (k > itmax) break;
            for (var j = 0; j < t; j++)
                for (var i = 0; i < n; i++) s[j][i] = y[j][i] >= 0 ? 1.0 : -1.0;
            if (sOld is not null && AllParallel(s, sOld)) break;
            if (t > 1) ResampleParallelColumns(s, sOld, rng);
            sOld = s.Select(c => (double[])c.Clone()).ToArray();
            Array.Clear(h);
            for (var j = 0; j < t; j++)
            {
                a.RMatVec(s[j], z);
                for (var i = 0; i < n; i++) h[i] = Math.Max(h[i], Math.Abs(z[i]));
            }
            if (k >= 2 && indBest >= 0 && h.Max() == h[indBest]) break;
            var order = Enumerable.Range(0, n).OrderByDescending(i => h[i]).ToArray();
            if (t > 1 && order.Take(t).All(history.Contains)) break;
            var chosen = order.Where(i => !history.Contains(i)).Take(t).ToArray();
            if (chosen.Length == 0) break;
            ind = new int[t];
            for (var j = 0; j < t; j++)
            {
                Array.Clear(x[j]);
                var col = chosen[Math.Min(j, chosen.Length - 1)];
                ind[j] = col;
                x[j][col] = 1.0;
                history.Add(col);
            }
        }
        return est;
    }

    private static double ExactOneNorm(ILinearOperator a)
    {
        var n = a.Columns;
        var e = new double[n];
        var col = new double[a.Rows];
        var best = 0.0;
        for (var j = 0; j < n; j++)
        {
            e[j] = 1;
            a.MatVec(e, col);
            e[j] = 0;
            var s = 0.0;
            foreach (var v in col) s += Math.Abs(v);
            best = Math.Max(best, s);
        }
        return best;
    }

    private static bool AllParallel(double[][] s, double[][] sOld)
    {
        foreach (var c in s)
        {
            var found = false;
            foreach (var o in sOld)
                if (Math.Abs(Dot(c, o)) == c.Length) { found = true; break; }
            if (!found) return false;
        }
        return true;
    }

    private static void ResampleParallelColumns(double[][] s, double[][]? sOld, GraviRandom rng)
    {
        var n = s[0].Length;
        for (var j = 0; j < s.Length; j++)
        {
            for (var attempt = 0; attempt < 10; attempt++)
            {
                var parallel = false;
                for (var i = 0; i < j && !parallel; i++) parallel = Math.Abs(Dot(s[j], s[i])) == n;
                if (sOld is not null) foreach (var o in sOld) parallel |= Math.Abs(Dot(s[j], o)) == n;
                if (!parallel) break;
                var scale = Math.Abs(s[j][0]);
                for (var i = 0; i < n; i++) s[j][i] = rng.NextDouble() < 0.5 ? -scale : scale;
            }
        }
    }

    // ------------------------------------------------------------------ expm_multiply

    private static readonly SortedDictionary<int, double> Theta = new()
    {
        [1] = 2.29e-16, [2] = 2.58e-8, [3] = 1.39e-5, [4] = 3.40e-4, [5] = 2.40e-3, [6] = 9.07e-3, [7] = 2.38e-2,
        [8] = 5.00e-2, [9] = 8.96e-2, [10] = 1.44e-1, [11] = 2.14e-1, [12] = 3.00e-1, [13] = 4.00e-1, [14] = 5.14e-1,
        [15] = 6.41e-1, [16] = 7.81e-1, [17] = 9.31e-1, [18] = 1.09, [19] = 1.26, [20] = 1.44, [21] = 1.62, [22] = 1.82,
        [23] = 2.01, [24] = 2.22, [25] = 2.43, [26] = 2.64, [27] = 2.86, [28] = 3.08, [29] = 3.31, [30] = 3.54,
        [35] = 4.7, [40] = 6.0, [45] = 7.2, [50] = 8.5, [55] = 9.9,
    };

    /// <summary>
    /// <c>exp(t A) b</c> without forming <c>exp(t A)</c> — <c>scipy.sparse.linalg.expm_multiply</c>.
    /// </summary>
    /// <param name="a">Square sparse matrix.</param>
    /// <param name="b">Vector to act on.</param>
    /// <param name="t">Time scaling (default 1).</param>
    /// <remarks>
    /// Al-Mohy &amp; Higham (2011), Algorithm 3.2 as SciPy implements it: shift by <c>μ = tr(A)/n</c>,
    /// choose the Taylor degree m* and scaling s from the θ table and 1-norm estimates of powers of
    /// A (fragment 3.1), then take s truncated-Taylor steps with early termination. Only products
    /// <c>A x</c> are needed, so the cost is about <c>s·m*</c> sparse products.
    /// </remarks>
    public static double[] ExpmMultiply(SparseMatrixBase a, double[] b, double t = 1.0)
    {
        if (a.Rows != a.Columns) throw new ArgumentException("ExpmMultiply needs a square matrix.");
        if (b.Length != a.Rows) throw new ArgumentException($"b has {b.Length} entries, expected {a.Rows}.");
        var r = ExpmMultiplyBlock(a, a.Trace(), [(double[])b.Clone()], t);
        return r[0];
    }

    /// <summary><c>exp(t A) B</c> for a dense vector or n×p matrix <paramref name="b"/>.</summary>
    public static NdArray ExpmMultiply(SparseMatrixBase a, NdArray b, double t = 1.0)
    {
        if (b.Rank == 1) return new NdArray(ExpmMultiply(a, b.ToArray(), t), b.Size);
        if (b.Rank != 2 || b.Shape[0] != a.Rows) throw new ArgumentException($"B must have {a.Rows} rows.");
        int n = b.Shape[0], p = b.Shape[1];
        var src = b.AsContiguous().ToArray();
        var cols = new double[p][];
        for (var j = 0; j < p; j++) { cols[j] = new double[n]; for (var i = 0; i < n; i++) cols[j][i] = src[i * p + j]; }
        var r = ExpmMultiplyBlock(a, a.Trace(), cols, t);
        var outBuf = new double[n * p];
        for (var j = 0; j < p; j++) for (var i = 0; i < n; i++) outBuf[i * p + j] = r[j][i];
        return new NdArray(outBuf, n, p);
    }

    /// <summary>
    /// <c>exp(t A) b</c> for a general operator. <paramref name="traceA"/> enables the shift that
    /// helps when the spectrum is far from zero; without it no shift is applied (still correct).
    /// </summary>
    public static double[] ExpmMultiply(ILinearOperator a, double[] b, double t = 1.0, double? traceA = null)
    {
        if (a.Rows != a.Columns) throw new ArgumentException("ExpmMultiply needs a square operator.");
        if (b.Length != a.Rows) throw new ArgumentException($"b has {b.Length} entries, expected {a.Rows}.");
        if (a is SparseMatrixBase s && traceA is null) traceA = s.Trace();
        return ExpmMultiplyBlock(a, traceA ?? 0.0, [(double[])b.Clone()], t)[0];
    }

    /// <summary>
    /// <c>exp(tₖ A) b</c> at <paramref name="num"/> evenly spaced times in [start, stop] (SciPy's
    /// interval form). Returns a (num × n) array, one row per time.
    /// </summary>
    /// <remarks>
    /// Steps from each time to the next with the single-time algorithm (<c>X(tₖ₊₁) = exp(h A) X(tₖ)</c>),
    /// rather than porting SciPy's interval-specific bookkeeping; each step is controlled to unit
    /// roundoff, so results agree with SciPy's to rounding, not bit for bit.
    /// </remarks>
    public static NdArray ExpmMultiply(SparseMatrixBase a, double[] b, double start, double stop, int num, bool endpoint = true)
    {
        if (num < 1) throw new ArgumentException("num must be positive.");
        var n = b.Length;
        var h = num == 1 ? 0 : (stop - start) / (endpoint ? num - 1 : num);
        var outBuf = new double[num * n];
        var trace = a.Trace();
        var x = start == 0 ? (double[])b.Clone() : ExpmMultiplyBlock(a, trace, [(double[])b.Clone()], start)[0];
        x.CopyTo(outBuf, 0);
        for (var q = 1; q < num; q++)
        {
            x = ExpmMultiplyBlock(a, trace, [x], h)[0];
            x.CopyTo(outBuf, q * n);
        }
        return new NdArray(outBuf, num, n);
    }

    private static double[][] ExpmMultiplyBlock(ILinearOperator a0, double traceA, double[][] b, double t)
    {
        var n = a0.Rows;
        if (n == 0) return b;
        var n0 = b.Length;
        const double tol = 1.1102230246251565e-16; // 2^-53
        var mu = traceA / n;
        // A − μI as an operator (and exactly as a sparse matrix when we have one, for the exact 1-norm).
        ILinearOperator a;
        double a1;
        if (a0 is SparseMatrixBase sm)
        {
            var shifted = mu == 0 ? sm.ToCsr() : sm.ToCsr().Subtract(SparseMatrices.Identity(n).Scale(mu));
            a = shifted;
            a1 = shifted.Norm(SparseNormKind.One);
        }
        else
        {
            a = mu == 0 ? a0 : LinearOperator.Sum(a0, LinearOperator.Scaled(LinearOperator.Identity(n), -mu));
            a1 = Onenormest(a);
        }
        int mStar, s;
        if (t * a1 == 0) { mStar = 0; s = 1; }
        else (mStar, s) = Fragment31(a, t, t * a1, n0, ell: 2);

        var f = b.Select(c => (double[])c.Clone()).ToArray();
        var bb = b.Select(c => (double[])c.Clone()).ToArray();
        var eta = Math.Exp(t * mu / s);
        var tmp = new double[n];
        for (var i = 0; i < s; i++)
        {
            var c1 = InfNorm(bb);
            for (var j = 0; j < mStar; j++)
            {
                var coeff = t / (s * (double)(j + 1));
                for (var c = 0; c < n0; c++)
                {
                    a.MatVec(bb[c], tmp);
                    for (var r = 0; r < n; r++) bb[c][r] = coeff * tmp[r];
                    Axpy(1.0, bb[c], f[c]);
                }
                var c2 = InfNorm(bb);
                if (c1 + c2 <= tol * InfNorm(f)) break;
                c1 = c2;
            }
            foreach (var c in f) ScaleInPlace(c, eta);
            for (var c = 0; c < n0; c++) f[c].CopyTo(bb[c], 0);
        }
        return f;
    }

    private static double InfNorm(double[][] cols)
    {
        var n = cols[0].Length;
        var best = 0.0;
        for (var i = 0; i < n; i++)
        {
            var s = 0.0;
            foreach (var c in cols) s += Math.Abs(c[i]);
            best = Math.Max(best, s);
        }
        return best;
    }

    private static (int M, int S) Fragment31(ILinearOperator a, double t, double tA1, int n0, int ell)
    {
        const int mMax = 55;
        var pMax = ComputePMax(mMax);
        int bestM = -1;
        long bestS = -1;
        if (tA1 <= 2.0 * ell * pMax * (pMax + 3) * (Theta[mMax] / (n0 * (double)mMax)))
        {
            foreach (var (m, theta) in Theta)
            {
                var s = (long)Math.Ceiling(tA1 / theta);
                if (bestM < 0 || m * s < bestM * bestS) { bestM = m; bestS = s; }
            }
        }
        else
        {
            var d = new Dictionary<int, double>();
            double D(int p)
            {
                if (!d.TryGetValue(p, out var v))
                {
                    var power = LinearOperator.Power(LinearOperator.Scaled(a, t), p);
                    v = Math.Pow(Onenormest(power, ell), 1.0 / p);
                    d[p] = v;
                }
                return v;
            }
            for (var p = 2; p <= pMax; p++)
                for (var m = p * (p - 1) - 1; m <= mMax; m++)
                {
                    if (!Theta.TryGetValue(m, out var theta)) continue;
                    var alpha = Math.Max(D(p), D(p + 1));
                    var s = (long)Math.Ceiling(alpha / theta);
                    if (bestM < 0 || m * s < bestM * bestS) { bestM = m; bestS = s; }
                }
            bestS = Math.Max(bestS, 1);
        }
        if (bestS > int.MaxValue) throw new ArgumentException("ExpmMultiply: ‖tA‖ is too large.");
        return (bestM, (int)bestS);
    }

    private static int ComputePMax(int mMax)
    {
        var sq = Math.Sqrt(mMax);
        int lo = (int)Math.Floor(sq), hi = (int)Math.Ceiling(sq + 1);
        var best = 0;
        for (var p = lo; p <= hi; p++) if (p * (p - 1) <= mMax + 1) best = p;
        return best;
    }

    // ------------------------------------------------------------------ expm

    private static readonly double[] Pade3 = [120, 60, 12, 1];
    private static readonly double[] Pade5 = [30240, 15120, 3360, 420, 30, 1];
    private static readonly double[] Pade7 = [17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1];
    private static readonly double[] Pade9 = [17643225600, 8821612800, 2075673600, 302702400, 30270240, 2162160, 110880, 3960, 90, 1];
    private static readonly double[] Pade13 =
    [
        64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000,
        10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1,
    ];

    /// <summary>
    /// The matrix exponential of a sparse matrix — <c>scipy.sparse.linalg.expm</c>.
    /// </summary>
    /// <remarks>
    /// Scaling and squaring with a [m/m] Padé approximant, m ∈ {3, 5, 7, 9, 13}, chosen from the
    /// exact 1-norm by Higham's (2005) θ thresholds; the products are sparse and the final solve
    /// <c>(V − U)⁻¹(V + U)</c> uses <see cref="Splu"/>. SciPy uses the refined 2009 variant (which
    /// can pick a lower degree via norm estimates of powers), so the two may differ in the last
    /// digits. <c>exp(A)</c> is dense whenever A's graph is connected; this is only a win for
    /// reducible or block-structured matrices. To apply <c>exp(A)</c> to vectors use
    /// <see cref="ExpmMultiply(SparseMatrixBase, double[], double)"/>.
    /// </remarks>
    public static CscMatrix Expm(SparseMatrixBase a)
    {
        if (a.Rows != a.Columns) throw new ArgumentException("Expm needs a square matrix.");
        var n = a.Rows;
        var A = a.ToCsr();
        var id = SparseMatrices.Identity(n);
        var norm1 = A.Norm(SparseNormKind.One);
        CsrMatrix u, v;
        var s = 0;
        double[] b;
        if (norm1 <= 1.495585217958292e-2) b = Pade3;
        else if (norm1 <= 2.539398330063230e-1) b = Pade5;
        else if (norm1 <= 9.504178996162932e-1) b = Pade7;
        else if (norm1 <= 2.097847961257068) b = Pade9;
        else b = Pade13;
        if (b.Length < 14)
        {
            var a2 = A.MatMul(A);
            var powers = new List<CsrMatrix> { id, a2 };
            for (var p = 2; p * 2 < b.Length; p++) powers.Add(powers[^1].MatMul(a2));
            var uu = CsrMatrix.Zeros(n, n);
            var vv = CsrMatrix.Zeros(n, n);
            for (var p = 0; p < powers.Count; p++)
            {
                vv = vv.Add(powers[p].Scale(b[2 * p]));
                if (2 * p + 1 < b.Length) uu = uu.Add(powers[p].Scale(b[2 * p + 1]));
            }
            u = A.MatMul(uu);
            v = vv;
        }
        else
        {
            s = Math.Max(0, (int)Math.Ceiling(Math.Log2(norm1 / 5.371920351148152)));
            if (s > 0) A = A.Scale(Math.Pow(2, -s));
            var a2 = A.MatMul(A);
            var a4 = a2.MatMul(a2);
            var a6 = a4.MatMul(a2);
            var inner = a6.Scale(b[13]).Add(a4.Scale(b[11])).Add(a2.Scale(b[9]));
            u = A.MatMul(a6.MatMul(inner).Add(a6.Scale(b[7])).Add(a4.Scale(b[5])).Add(a2.Scale(b[3])).Add(id.Scale(b[1])));
            var innerV = a6.Scale(b[12]).Add(a4.Scale(b[10])).Add(a2.Scale(b[8]));
            v = a6.MatMul(innerV).Add(a6.Scale(b[6])).Add(a4.Scale(b[4])).Add(a2.Scale(b[2])).Add(id.Scale(b[0]));
        }
        var r = Spsolve(v.Subtract(u), v.Add(u)).ToCsr();
        for (var i = 0; i < s; i++) r = r.MatMul(r);
        return r.ToCsc();
    }
}
