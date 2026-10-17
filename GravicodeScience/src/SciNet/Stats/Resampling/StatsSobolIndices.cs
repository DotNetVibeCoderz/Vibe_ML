using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Result of <c>Stats.SobolIndices</c> (scipy's <c>SobolResult</c>).</summary>
public sealed class SobolIndicesResult
{
    private readonly double[][] _fA, _fB;
    private readonly double[][][] _fAB;

    internal SobolIndicesResult(double[][] fA, double[][] fB, double[][][] fAB, Func<double[][], double[][], double[][][], (double[][] S, double[][] St)> method)
    {
        _fA = fA;
        _fB = fB;
        _fAB = fAB;
        var (s, st) = method(fA, fB, fAB);
        FirstOrderByOutput = s;
        TotalOrderByOutput = st;
    }

    /// <summary>First-order indices of the (first) output, one per input variable.</summary>
    public double[] FirstOrder => FirstOrderByOutput[0];

    /// <summary>Total-order indices of the (first) output.</summary>
    public double[] TotalOrder => TotalOrderByOutput[0];

    /// <summary>First-order indices, [output][variable].</summary>
    public double[][] FirstOrderByOutput { get; }

    /// <summary>Total-order indices, [output][variable].</summary>
    public double[][] TotalOrderByOutput { get; }

    /// <summary>The model evaluated on sample A, [output][point] (centred by the overall mean, as scipy).</summary>
    public double[][] FA => _fA;

    /// <summary>The model evaluated on sample B, [output][point] (centred).</summary>
    public double[][] FB => _fB;

    /// <summary>The model evaluated on the hybrid samples A_B^(i), [variable][output][point] (centred).</summary>
    public double[][][] FAB => _fAB;

    /// <summary>
    /// BCa bootstrap confidence intervals of the indices of output <paramref name="output"/>, scipy's
    /// <c>SobolResult.bootstrap</c>: the evaluation points are resampled (no new model evaluations) and
    /// the estimator is recomputed; every index shares the same resamples.
    /// </summary>
    /// <remarks>
    /// The Saltelli/Jansen estimators are ratios of sums over points, so each resample costs O(n·d) and the
    /// jackknife that BCa needs for its acceleration is done by removing one point from the sums — O(n·d)
    /// in total rather than the O(n²·d) of re-running the estimator n times (9 s → milliseconds at n = 4096).
    /// </remarks>
    public (BootstrapResult[] FirstOrder, BootstrapResult[] TotalOrder) Bootstrap(double confidenceLevel = 0.95, int nResamples = 999, int output = 0, int? seed = null)
    {
        if (output < 0 || output >= _fA.Length) throw new ArgumentOutOfRangeException(nameof(output));
        if (!(confidenceLevel > 0 && confidenceLevel < 1)) throw new ArgumentException("confidenceLevel must lie in (0, 1).", nameof(confidenceLevel));
        if (nResamples < 1) throw new ArgumentException("nResamples must be positive.", nameof(nResamples));
        var fa = _fA[output];
        var fb = _fB[output];
        var n = fa.Length;
        var d = _fAB.Length;
        var fab = new double[d][];
        for (var i = 0; i < d; i++) fab[i] = _fAB[i][output];
        // Per-point terms: a, b, a², b², b(ab_i − a), (a − ab_i)².
        var t1 = new double[d][];
        var t2 = new double[d][];
        for (var i = 0; i < d; i++)
        {
            t1[i] = new double[n];
            t2[i] = new double[n];
            for (var j = 0; j < n; j++)
            {
                t1[i][j] = fb[j] * (fab[i][j] - fa[j]);
                var u = fa[j] - fab[i][j];
                t2[i][j] = u * u;
            }
        }
        void Indices(double sa, double sb, double saa, double sbb, double[] s1, double[] s2, double m, double[] outS, double[] outT)
        {
            var mu = (sa + sb) / (2 * m);
            var v = (saa + sbb) / (2 * m) - mu * mu;
            for (var i = 0; i < d; i++)
            {
                outS[i] = Clean(s1[i] / m / v);
                outT[i] = Clean(0.5 * s2[i] / m / v);
            }
        }
        double SA = 0, SB = 0, SAA = 0, SBB = 0;
        var S1 = new double[d];
        var S2 = new double[d];
        for (var j = 0; j < n; j++)
        {
            SA += fa[j]; SB += fb[j]; SAA += fa[j] * fa[j]; SBB += fb[j] * fb[j];
            for (var i = 0; i < d; i++) { S1[i] += t1[i][j]; S2[i] += t2[i][j]; }
        }
        var hatS = new double[d];
        var hatT = new double[d];
        Indices(SA, SB, SAA, SBB, S1, S2, n, hatS, hatT);
        // Resamples.
        var master = new GraviRandom(seed ?? Random.Shared.Next());
        var bootS = new double[d][];
        var bootT = new double[d][];
        for (var i = 0; i < d; i++) { bootS[i] = new double[nResamples]; bootT[i] = new double[nResamples]; }
        var r1 = new double[d];
        var r2 = new double[d];
        var oS = new double[d];
        var oT = new double[d];
        for (var b = 0; b < nResamples; b++)
        {
            var rng = new GraviRandom(master.NextUInt64());
            double sa = 0, sb = 0, saa = 0, sbb = 0;
            Array.Clear(r1);
            Array.Clear(r2);
            for (var t = 0; t < n; t++)
            {
                var j = rng.Next(n);
                sa += fa[j]; sb += fb[j]; saa += fa[j] * fa[j]; sbb += fb[j] * fb[j];
                for (var i = 0; i < d; i++) { r1[i] += t1[i][j]; r2[i] += t2[i][j]; }
            }
            Indices(sa, sb, saa, sbb, r1, r2, n, oS, oT);
            for (var i = 0; i < d; i++) { bootS[i][b] = oS[i]; bootT[i][b] = oT[i]; }
        }
        // Jackknife by removal from the sums.
        var jackS = new double[d][];
        var jackT = new double[d][];
        for (var i = 0; i < d; i++) { jackS[i] = new double[n]; jackT[i] = new double[n]; }
        for (var j = 0; j < n; j++)
        {
            for (var i = 0; i < d; i++) { r1[i] = S1[i] - t1[i][j]; r2[i] = S2[i] - t2[i][j]; }
            Indices(SA - fa[j], SB - fb[j], SAA - fa[j] * fa[j], SBB - fb[j] * fb[j], r1, r2, n - 1, oS, oT);
            for (var i = 0; i < d; i++) { jackS[i][j] = oS[i]; jackT[i][j] = oT[i]; }
        }
        var alpha = (1 - confidenceLevel) / 2;
        var first = new BootstrapResult[d];
        var total = new BootstrapResult[d];
        for (var i = 0; i < d; i++)
        {
            first[i] = Bca(hatS[i], bootS[i], jackS[i], alpha);
            total[i] = Bca(hatT[i], bootT[i], jackT[i], alpha);
        }
        return (first, total);
    }

    /// <summary>scipy's BCa interval from the bootstrap distribution and the jackknife values (Efron 1987).</summary>
    private static BootstrapResult Bca(double thetaHat, double[] thetaB, double[] jack, double alpha)
    {
        double below = 0, belowEq = 0;
        foreach (var t in thetaB)
        {
            if (t < thetaHat) below++;
            if (t <= thetaHat) belowEq++;
        }
        var z0 = Special.Ndtri((below + belowEq) / (2.0 * thetaB.Length));
        double n = jack.Length;
        var mean = jack.Average();
        double s3 = 0, s2 = 0;
        foreach (var t in jack)
        {
            var u = (n - 1) * (mean - t);
            s3 += u * u * u;
            s2 += u * u;
        }
        var a = 1.0 / 6 * (s3 / (n * n * n)) / Math.Pow(s2 / (n * n), 1.5);
        var za = Special.Ndtri(alpha);
        var n1 = z0 + za;
        var a1 = Special.Ndtr(z0 + n1 / (1 - a * n1));
        var n2 = z0 - za;
        var a2 = Special.Ndtr(z0 + n2 / (1 - a * n2));
        var lo = double.IsNaN(a1) ? double.NaN : Stats.Quantile(thetaB, a1);
        var hi = double.IsNaN(a2) ? double.NaN : Stats.Quantile(thetaB, a2);
        var m = thetaB.Average();
        var ss = 0.0;
        foreach (var t in thetaB) ss += (t - m) * (t - m);
        return new BootstrapResult(new ConfidenceInterval(lo, hi), thetaB, Math.Sqrt(ss / (thetaB.Length - 1)));
    }

    internal static double Clean(double v) => double.IsFinite(v) ? v : 0.0;
}

public static partial class Stats
{
    /// <summary>
    /// Sobol' sensitivity indices of a scalar model, scipy's <c>sobol_indices</c>: the model is called
    /// once per point with a vector of the d inputs (drawn from <paramref name="dists"/>), n(d + 2) calls in all.
    /// </summary>
    public static SobolIndicesResult SobolIndices(Func<double[], double> func, int n, IUnivariateDistribution[] dists, int? seed = null, bool parallel = false)
    {
        ArgumentNullException.ThrowIfNull(func);
        return SobolIndices(x =>
        {
            int d = x.Shape[0], m = x.Shape[1];
            var s = x.AsContiguous().ToArray();
            var r = new double[m];
            void One(int j)
            {
                var p = new double[d];
                for (var i = 0; i < d; i++) p[i] = s[i * m + j];
                r[j] = func(p);
            }
            if (parallel) Parallel.For(0, m, One); else for (var j = 0; j < m; j++) One(j);
            return new NdArray(r, 1, m);
        }, n, dists, seed);
    }

    /// <summary>
    /// Sobol' sensitivity indices, scipy's <c>sobol_indices</c> with Saltelli et al. (2010): first-order
    /// S_i = mean(f_B (f_{A_B^i} − f_A))/Var and Jansen's total order S_Ti = ½ mean((f_A − f_{A_B^i})²)/Var.
    /// </summary>
    /// <param name="func">Vectorised model: takes a (d, m) array of points (one column per point) and returns (s, m) or (m) outputs.</param>
    /// <param name="n">Base sample size; a power of two, for the balance of the Sobol' points.</param>
    /// <param name="dists">One marginal per input variable; points are mapped through their <c>Ppf</c>.</param>
    /// <param name="seed">Seed of the scrambled Sobol' sequence (the scrambling differs from scipy's for the same seed).</param>
    /// <remarks>A and B are the two halves of one scrambled 2d-dimensional Sobol' sample, as scipy. Outputs with
    /// zero variance give zero indices rather than NaN.</remarks>
    public static SobolIndicesResult SobolIndices(Func<NdArray, NdArray> func, int n, IUnivariateDistribution[] dists, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(func);
        ArgumentNullException.ThrowIfNull(dists);
        if (n < 1 || (n & (n - 1)) != 0) throw new ArgumentException("The balance properties of Sobol' points require 'n' to be a power of 2.", nameof(n));
        var d = dists.Length;
        if (d == 0) throw new ArgumentException("dists must not be empty.", nameof(dists));
        var u = new SobolEngine(2 * d, scramble: true, seed: seed).Random(n).AsContiguous().ToArray();
        var a = new double[d * n];
        var b = new double[d * n];
        for (var j = 0; j < n; j++)
            for (var i = 0; i < d; i++)
            {
                a[i * n + j] = dists[i].Ppf(u[j * 2 * d + i]);
                b[i * n + j] = dists[i].Ppf(u[j * 2 * d + d + i]);
            }
        double[][] Call(double[] x, int m)
        {
            var r = func(new NdArray(x, d, m));
            var flat = r.AsContiguous().ToArray();
            var s = r.Shape.Length == 1 ? 1 : r.Shape[0];
            if (flat.Length != s * m) throw new ArgumentException("func output should have a shape (s, m) with s the number of outputs.", nameof(func));
            var o = new double[s][];
            for (var k = 0; k < s; k++) o[k] = flat[(k * m)..((k + 1) * m)];
            return o;
        }
        var fA = Call(a, n);
        var fB = Call(b, n);
        // All d hybrid samples in one call: column block i is A with row i taken from B.
        var ab = new double[d * n * d];
        for (var i = 0; i < d; i++)
            for (var r = 0; r < d; r++)
                for (var j = 0; j < n; j++)
                    ab[r * (n * d) + i * n + j] = (r == i ? b : a)[r * n + j];
        var fAll = Call(ab, n * d);
        var fAB = new double[d][][];
        for (var i = 0; i < d; i++)
        {
            fAB[i] = new double[fA.Length][];
            for (var k = 0; k < fA.Length; k++) fAB[i][k] = fAll[k][(i * n)..((i + 1) * n)];
        }
        return SobolIndices(fA, fB, fAB);
    }

    /// <summary>
    /// Sobol' indices from model evaluations made elsewhere (scipy's dict form of <c>func</c>):
    /// <paramref name="fA"/> and <paramref name="fB"/> are [output][point], <paramref name="fAB"/> is
    /// [variable][output][point]. The arrays are copied and centred by the overall mean (Sobol' &amp; Levitan 1999).
    /// </summary>
    public static SobolIndicesResult SobolIndices(double[][] fA, double[][] fB, double[][][] fAB)
    {
        ArgumentNullException.ThrowIfNull(fA);
        ArgumentNullException.ThrowIfNull(fB);
        ArgumentNullException.ThrowIfNull(fAB);
        var s = fA.Length;
        if (s == 0 || fB.Length != s || fAB.Length == 0) throw new ArgumentException("fA and fB must have shape (s, n) and fAB (d, s, n).");
        var n = fA[0].Length;
        foreach (var r in fA.Concat(fB)) if (r.Length != n) throw new ArgumentException("fA and fB must have shape (s, n).");
        foreach (var r in fAB) { if (r.Length != s) throw new ArgumentException("fAB must have shape (d, s, n)."); foreach (var q in r) if (q.Length != n) throw new ArgumentException("fAB must have shape (d, s, n)."); }
        var a = fA.Select(r => (double[])r.Clone()).ToArray();
        var b = fB.Select(r => (double[])r.Clone()).ToArray();
        var ab = fAB.Select(v => v.Select(r => (double[])r.Clone()).ToArray()).ToArray();
        for (var k = 0; k < s; k++)
        {
            var mean = (a[k].Sum() + b[k].Sum()) / (2.0 * n);
            for (var j = 0; j < n; j++) { a[k][j] -= mean; b[k][j] -= mean; }
            foreach (var v in ab) for (var j = 0; j < n; j++) v[k][j] -= mean;
        }
        return new SobolIndicesResult(a, b, ab, Saltelli2010);
    }

    /// <summary>Saltelli et al. (2010) Table 2 (b) first order and (f) Jansen total order, per output.</summary>
    private static (double[][] S, double[][] St) Saltelli2010(double[][] fA, double[][] fB, double[][][] fAB)
    {
        var s = fA.Length;
        var d = fAB.Length;
        var n = fA[0].Length;
        var first = new double[s][];
        var total = new double[s][];
        for (var k = 0; k < s; k++)
        {
            // Variance of the pooled A and B evaluations (numpy var over both, ddof 0).
            double m = 0;
            for (var j = 0; j < n; j++) m += fA[k][j] + fB[k][j];
            m /= 2.0 * n;
            double v = 0;
            for (var j = 0; j < n; j++) v += (fA[k][j] - m) * (fA[k][j] - m) + (fB[k][j] - m) * (fB[k][j] - m);
            v /= 2.0 * n;
            first[k] = new double[d];
            total[k] = new double[d];
            for (var i = 0; i < d; i++)
            {
                double f1 = 0, ft = 0;
                var ab = fAB[i][k];
                for (var j = 0; j < n; j++)
                {
                    f1 += fB[k][j] * (ab[j] - fA[k][j]);
                    var t = fA[k][j] - ab[j];
                    ft += t * t;
                }
                first[k][i] = SobolIndicesResult.Clean(f1 / n / v);
                total[k][i] = SobolIndicesResult.Clean(0.5 * ft / n / v);
            }
        }
        return (first, total);
    }
}
