using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Solvers for <see cref="Optimize.QuadraticAssignment(Gravicode.Science.GraviNum.NdArray, Gravicode.Science.GraviNum.NdArray, QuadraticAssignmentMethod, QuadraticAssignmentOptions?)"/>.</summary>
public enum QuadraticAssignmentMethod
{
    /// <summary>Fast Approximate QAP (Vogelstein et al. 2015): Frank–Wolfe on the doubly-stochastic relaxation, then projection.</summary>
    Faq,
    /// <summary>Pairwise-exchange local search from a random (or guessed) permutation; slow but often better.</summary>
    TwoOpt,
}

/// <summary>Options for <see cref="Optimize.QuadraticAssignment(Gravicode.Science.GraviNum.NdArray, Gravicode.Science.GraviNum.NdArray, QuadraticAssignmentMethod, QuadraticAssignmentOptions?)"/>, mirroring SciPy's option dictionary.</summary>
public sealed record QuadraticAssignmentOptions
{
    /// <summary>Maximise the objective instead of minimising it (graph matching maximises).</summary>
    public bool Maximize { get; init; }

    /// <summary>Seeds: pairs (row of A, row of B) that are fixed in the answer.</summary>
    public (int A, int B)[]? PartialMatch { get; init; }

    /// <summary>Random source for <see cref="RandomizedP0"/>, <see cref="ShuffleInput"/> and 2-opt's initial permutation.</summary>
    public GraviRandom? Rng { get; init; }

    /// <summary>FAQ: the initial doubly-stochastic matrix over the unseeded nodes; null is the barycentre J/n.</summary>
    public NdArray? P0 { get; init; }

    /// <summary>FAQ: start from (J + K)/2 with K a random doubly-stochastic matrix (SciPy's <c>P0='randomized'</c>).</summary>
    public bool RandomizedP0 { get; init; }

    /// <summary>FAQ: randomly permute the unseeded nodes of B first, to break ties differently between runs.</summary>
    public bool ShuffleInput { get; init; }

    /// <summary>FAQ: maximum Frank–Wolfe iterations.</summary>
    public int MaxIter { get; init; } = 30;

    /// <summary>FAQ: stop when ‖Pᵢ − Pᵢ₊₁‖_F / √n′ falls below this.</summary>
    public double Tol { get; init; } = 0.03;

    /// <summary>2-opt: an initial guess for some pairs (row of A, row of B), refined by the search.</summary>
    public (int A, int B)[]? PartialGuess { get; init; }
}

/// <summary>Result of <see cref="Optimize.QuadraticAssignment(Gravicode.Science.GraviNum.NdArray, Gravicode.Science.GraviNum.NdArray, QuadraticAssignmentMethod, QuadraticAssignmentOptions?)"/>.</summary>
/// <param name="ColInd">The permutation: node i of A is matched to node ColInd[i] of B.</param>
/// <param name="Fun">Objective value <c>Σᵢⱼ A[i,j]·B[p(i),p(j)]</c>.</param>
/// <param name="Nit">Frank–Wolfe iterations (FAQ) or candidate swaps examined (2-opt).</param>
public sealed record QuadraticAssignmentResult(int[] ColInd, double Fun, int Nit);

public static partial class Optimize
{
    /// <summary>
    /// Approximately solves the quadratic assignment problem
    /// <c>min over permutations P of trace(Aᵀ P B Pᵀ)</c> (<c>scipy.optimize.quadratic_assignment</c>).
    /// QAP is NP-hard; both methods are heuristics and the answer is a local optimum.
    /// </summary>
    /// <param name="a">Square (n, n) matrix, e.g. flows or a graph's adjacency.</param>
    /// <param name="b">Square (n, n) matrix of the same size, e.g. distances or a second adjacency.</param>
    /// <param name="method">FAQ (default) or 2-opt.</param>
    /// <param name="options">Method options.</param>
    public static QuadraticAssignmentResult QuadraticAssignment(NdArray a, NdArray b, QuadraticAssignmentMethod method = QuadraticAssignmentMethod.Faq,
        QuadraticAssignmentOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        options ??= new QuadraticAssignmentOptions();
        if (a.Rank != 2 || b.Rank != 2) throw new ArgumentException("`A` and `B` must have exactly two dimensions");
        if (a.Shape[0] != a.Shape[1]) throw new ArgumentException("`A` must be square");
        if (b.Shape[0] != b.Shape[1]) throw new ArgumentException("`B` must be square");
        if (a.Shape[0] != b.Shape[0]) throw new ArgumentException("`A` and `B` matrices must be of equal size");
        var n = a.Shape[0];
        var am = a.AsContiguous().ToArray();
        var bm = b.AsContiguous().ToArray();
        var seeds = OptimizeQap.ValidatePairs(options.PartialMatch, n, "partial_match");
        var rng = options.Rng ?? new GraviRandom();
        return method switch
        {
            QuadraticAssignmentMethod.Faq => OptimizeQap.Faq(am, bm, n, seeds, options, rng),
            QuadraticAssignmentMethod.TwoOpt => OptimizeQap.TwoOpt(am, bm, n, seeds, OptimizeQap.ValidatePairs(options.PartialGuess, n, "partial_guess"), options.Maximize, rng),
            _ => throw new ArgumentOutOfRangeException(nameof(method)),
        };
    }

    /// <inheritdoc cref="QuadraticAssignment(NdArray, NdArray, QuadraticAssignmentMethod, QuadraticAssignmentOptions?)"/>
    public static QuadraticAssignmentResult QuadraticAssignment(double[,] a, double[,] b, QuadraticAssignmentMethod method = QuadraticAssignmentMethod.Faq,
        QuadraticAssignmentOptions? options = null)
        => QuadraticAssignment(NdArray.FromArray(a), NdArray.FromArray(b), method, options);
}

internal static class OptimizeQap
{
    public static (int A, int B)[] ValidatePairs((int A, int B)[]? pairs, int n, string name)
    {
        if (pairs is null) return [];
        if (pairs.Length > n) throw new ArgumentException($"`{name}` can have only as many entries as there are nodes");
        var sa = new HashSet<int>(); var sb = new HashSet<int>();
        foreach (var (pa, pb) in pairs)
        {
            if (pa < 0 || pb < 0) throw new ArgumentException($"`{name}` must contain only positive indices");
            if (pa >= n || pb >= n) throw new ArgumentException($"`{name}` entries must be less than number of nodes");
            if (!sa.Add(pa) || !sb.Add(pb)) throw new ArgumentException($"`{name}` column entries must be unique");
        }
        return pairs;
    }

    /// <summary>Σᵢⱼ A[i,j] B[p(i), p(j)].</summary>
    public static double Score(double[] a, double[] b, int n, int[] perm)
    {
        double s = 0;
        for (var i = 0; i < n; i++)
        {
            var pi = perm[i] * n;
            for (var j = 0; j < n; j++) s += a[i * n + j] * b[pi + perm[j]];
        }
        return s;
    }

    private static double[] Sub(double[] x, int n, int[] rows, int[] cols)
    {
        var r = new double[rows.Length * cols.Length];
        for (var i = 0; i < rows.Length; i++)
            for (var j = 0; j < cols.Length; j++) r[i * cols.Length + j] = x[rows[i] * n + cols[j]];
        return r;
    }

    private static double[] MatMul(double[] x, int m, int k, double[] y, int n) => OptimizeDense.MatMul(x, m, k, y, n);

    private static double[] T(double[] x, int m, int n) => OptimizeDense.Transpose(x, m, n);

    public static QuadraticAssignmentResult Faq(double[] a, double[] b, int n, (int A, int B)[] seeds, QuadraticAssignmentOptions o, GraviRandom rng)
    {
        if (o.MaxIter <= 0) throw new ArgumentException("'maxiter' must be a positive integer");
        if (o.Tol <= 0) throw new ArgumentException("'tol' must be a positive float");
        var ns = seeds.Length;
        var nu = n - ns;
        double[] p;
        if (o.P0 is { } p0)
        {
            if (o.P0.Rank != 2 || p0.Shape[0] != nu || p0.Shape[1] != nu) throw new ArgumentException("`P0` matrix must have shape m' x m', where m'=n-m");
            p = p0.AsContiguous().ToArray();
            for (var i = 0; i < nu; i++)
            {
                double rs = 0, cs = 0;
                for (var j = 0; j < nu; j++)
                {
                    if (p[i * nu + j] < 0) throw new ArgumentException("`P0` matrix must be doubly stochastic");
                    rs += p[i * nu + j]; cs += p[j * nu + i];
                }
                if (Math.Abs(rs - 1) > 1e-5 || Math.Abs(cs - 1) > 1e-5) throw new ArgumentException("`P0` matrix must be doubly stochastic");
            }
        }
        else
        {
            p = new double[nu * nu];
            Array.Fill(p, nu > 0 ? 1.0 / nu : 0);
            if (o.RandomizedP0 && nu > 0)
            {
                var k = new double[nu * nu];
                for (var i = 0; i < k.Length; i++) k[i] = rng.NextDouble();
                k = DoublyStochastic(k, nu);
                for (var i = 0; i < p.Length; i++) p[i] = (p[i] + k[i]) / 2;
            }
        }
        if (n == 0 || ns == n)
        {
            var permS = new int[n];
            foreach (var (sa, sb) in seeds) permS[sa] = sb;
            return new QuadraticAssignmentResult(permS, Score(a, b, n, permS), 0);
        }
        var objScalar = o.Maximize ? -1.0 : 1.0;
        var seedB = new HashSet<int>(seeds.Select(s => s.B));
        var seedA = new HashSet<int>(seeds.Select(s => s.A));
        var nonseedB = Enumerable.Range(0, n).Where(i => !seedB.Contains(i)).ToArray();
        if (o.ShuffleInput)
            for (var i = nonseedB.Length - 1; i > 0; i--) { var j = rng.Next(i + 1); (nonseedB[i], nonseedB[j]) = (nonseedB[j], nonseedB[i]); }
        var nonseedA = Enumerable.Range(0, n).Where(i => !seedA.Contains(i)).ToArray();
        int[] permA = [.. seeds.Select(s => s.A), .. nonseedA];
        int[] permB = [.. seeds.Select(s => s.B), .. nonseedB];
        int[] sIdx = [.. Enumerable.Range(0, ns)];
        int[] uIdx = [.. Enumerable.Range(ns, nu)];
        var ap = Sub(a, n, permA, permA);
        var bp = Sub(b, n, permB, permB);
        var a12 = Sub(ap, n, sIdx, uIdx); var a21 = Sub(ap, n, uIdx, sIdx); var a22 = Sub(ap, n, uIdx, uIdx);
        var b12 = Sub(bp, n, sIdx, uIdx); var b21 = Sub(bp, n, uIdx, sIdx); var b22 = Sub(bp, n, uIdx, uIdx);
        // const_sum = A21 B21ᵀ + A12ᵀ B12 (nu × nu)
        var constSum = MatMul(a21, nu, ns, T(b21, nu, ns), nu);
        var t2 = MatMul(T(a12, ns, nu), nu, ns, b12, nu);
        for (var i = 0; i < constSum.Length; i++) constSum[i] += t2[i];
        var a22T = T(a22, nu, nu);
        var b22T = T(b22, nu, nu);

        var nIter = 0;
        for (nIter = 1; nIter <= o.MaxIter; nIter++)
        {
            // ∇f(P) = const + A22 P B22ᵀ + A22ᵀ P B22
            var g1 = MatMul(MatMul(a22, nu, nu, p, nu), nu, nu, b22T, nu);
            var g2 = MatMul(MatMul(a22T, nu, nu, p, nu), nu, nu, b22, nu);
            var grad = new double[nu * nu];
            for (var i = 0; i < grad.Length; i++) grad[i] = constSum[i] + g1[i] + g2[i];
            var (_, cols) = OptimizeLsa.Solve(grad, nu, nu, o.Maximize);
            // Q = I[cols]; R = P − Q
            var r = (double[])p.Clone();
            for (var i = 0; i < nu; i++) r[i * nu + cols[i]] -= 1;
            var rT = T(r, nu, nu);
            // Exact line search: f(P + t(Q − P)) is quadratic in t (Vogelstein et al. eq. 5).
            double b21s = 0, b12s = 0;
            if (ns > 0)
            {
                var ra21 = MatMul(rT, nu, nu, a21, ns);
                for (var i = 0; i < ra21.Length; i++) b21s += ra21[i] * b21[i];
                var ra12 = MatMul(rT, nu, nu, T(a12, ns, nu), ns);
                var b12T = T(b12, ns, nu);
                for (var i = 0; i < ra12.Length; i++) b12s += ra12[i] * b12T[i];
            }
            var ar22 = MatMul(a22T, nu, nu, r, nu);
            var br22 = MatMul(b22, nu, nu, rT, nu);
            double b22a = 0, b22b = 0, aa = 0;
            for (var i = 0; i < nu; i++)
                for (var j = 0; j < nu; j++)
                {
                    b22a += ar22[i * nu + j] * b22T[cols[i] * nu + j];
                    b22b += a22[i * nu + j] * br22[cols[i] * nu + j];
                    aa += ar22[j * nu + i] * br22[i * nu + j];
                }
            var bb = b21s + b12s + b22a + b22b;
            double alpha;
            if (aa * objScalar > 0 && 0 <= -bb / (2 * aa) && -bb / (2 * aa) <= 1) alpha = -bb / (2 * aa);
            else alpha = (bb + aa) * objScalar < 0 ? 1 : 0;
            double diff = 0;
            var pNext = new double[nu * nu];
            for (var i = 0; i < nu; i++)
                for (var j = 0; j < nu; j++)
                {
                    var q = cols[i] == j ? 1.0 : 0.0;
                    var v = alpha * p[i * nu + j] + (1 - alpha) * q;
                    diff += (p[i * nu + j] - v) * (p[i * nu + j] - v);
                    pNext[i * nu + j] = v;
                }
            p = pNext;
            if (Math.Sqrt(diff) / Math.Sqrt(nu) < o.Tol) break;
        }
        if (nIter > o.MaxIter) nIter = o.MaxIter;
        var (_, col) = OptimizeLsa.Solve(p, nu, nu, true);
        var perm = new int[n];
        for (var i = 0; i < ns; i++) perm[i] = i;
        for (var i = 0; i < nu; i++) perm[ns + i] = col[i] + ns;
        var unshuffled = new int[n];
        for (var i = 0; i < n; i++) unshuffled[permA[i]] = permB[perm[i]];
        return new QuadraticAssignmentResult(unshuffled, Score(a, b, n, unshuffled), nIter);
    }

    /// <summary>Sinkhorn–Knopp balancing to row and column sums within 1e-3 (SciPy's <c>_doubly_stochastic</c>).</summary>
    private static double[] DoublyStochastic(double[] p, int n, double tol = 1e-3)
    {
        var c = new double[n]; var r = new double[n];
        for (var j = 0; j < n; j++) { double s = 0; for (var i = 0; i < n; i++) s += p[i * n + j]; c[j] = 1 / s; }
        for (var i = 0; i < n; i++) { double s = 0; for (var j = 0; j < n; j++) s += p[i * n + j] * c[j]; r[i] = 1 / s; }
        var pe = p;
        for (var it = 0; it < 1000; it++)
        {
            var ok = true;
            for (var i = 0; i < n && ok; i++) { double s = 0; for (var j = 0; j < n; j++) s += pe[i * n + j]; if (Math.Abs(s - 1) >= tol) ok = false; }
            for (var j = 0; j < n && ok; j++) { double s = 0; for (var i = 0; i < n; i++) s += pe[i * n + j]; if (Math.Abs(s - 1) >= tol) ok = false; }
            if (ok) break;
            for (var j = 0; j < n; j++) { double s = 0; for (var i = 0; i < n; i++) s += r[i] * p[i * n + j]; c[j] = 1 / s; }
            for (var i = 0; i < n; i++) { double s = 0; for (var j = 0; j < n; j++) s += p[i * n + j] * c[j]; r[i] = 1 / s; }
            pe = new double[n * n];
            for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) pe[i * n + j] = r[i] * p[i * n + j] * c[j];
        }
        return pe;
    }

    public static QuadraticAssignmentResult TwoOpt(double[] a, double[] b, int n, (int A, int B)[] seeds, (int A, int B)[] guess, bool maximize, GraviRandom rng)
    {
        if (n == 0 || seeds.Length == n)
        {
            var permS = new int[n];
            foreach (var (sa, sb) in seeds) permS[sa] = sb;
            return new QuadraticAssignmentResult(permS, Score(a, b, n, permS), 0);
        }
        var perm = new int[n];
        var fixedRows = new bool[n];
        var usedRows = new bool[n]; var usedCols = new bool[n];
        foreach (var (ga, gb) in guess) { perm[ga] = gb; usedRows[ga] = true; usedCols[gb] = true; }
        foreach (var (sa, sb) in seeds)
        {
            if (usedRows[sa] && perm[sa] != sb) usedCols[perm[sa]] = false;
            perm[sa] = sb; fixedRows[sa] = true; usedRows[sa] = true; usedCols[sb] = true;
        }
        var freeCols = Enumerable.Range(0, n).Where(j => !usedCols[j]).ToArray();
        for (var i = freeCols.Length - 1; i > 0; i--) { var j = rng.Next(i + 1); (freeCols[i], freeCols[j]) = (freeCols[j], freeCols[i]); }
        var k = 0;
        for (var i = 0; i < n; i++) if (!usedRows[i]) perm[i] = freeCols[k++];
        if (k != freeCols.Length) throw new ArgumentException("`partial_guess` conflicts with `partial_match`.");

        var best = Score(a, b, n, perm);
        var free = Enumerable.Range(0, n).Where(i => !fixedRows[i]).ToArray();
        var nIter = 0;
        var done = false;
        while (!done)
        {
            done = true;
            for (var ii = 0; ii < free.Length && done; ii++)
                for (var jj = ii; jj < free.Length; jj++)
                {
                    nIter++;
                    int i = free[ii], j = free[jj];
                    (perm[i], perm[j]) = (perm[j], perm[i]);
                    var s = Score(a, b, n, perm);
                    if (maximize ? s > best : s < best)
                    {
                        best = s;
                        done = false;
                        break;
                    }
                    (perm[i], perm[j]) = (perm[j], perm[i]);
                }
        }
        return new QuadraticAssignmentResult(perm, best, nIter);
    }
}
