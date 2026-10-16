using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// scipy.stats.qmc helpers: discrepancy measures, scaling to bounds and small number-theory utilities
/// shared by the engines.
/// </summary>
public static class Qmc
{
    // ------------------------------------------------------------------ discrepancy

    /// <summary>
    /// Discrepancy of a sample in [0, 1]^d, scipy's <c>qmc.discrepancy</c>: <c>"CD"</c> (centred L2,
    /// Hickernell 1998), <c>"WD"</c> (wrap-around), <c>"MD"</c> (mixture) or <c>"L2-star"</c>. With
    /// <paramref name="iterative"/> the normalisation uses n + 1 points, so the value can be extended by
    /// <see cref="UpdateDiscrepancy"/>. The double sums are O(n² d) and run in parallel over rows.
    /// </summary>
    public static double Discrepancy(NdArray sample, bool iterative = false, string method = "CD")
    {
        ArgumentNullException.ThrowIfNull(sample);
        if (sample.Rank != 2) throw new ArgumentException("Sample is not a 2D array", nameof(sample));
        var n = sample.Shape[0];
        var d = sample.Shape[1];
        var x = sample.AsContiguous().ToArray();
        foreach (var v in x) if (!(v >= 0 && v <= 1)) throw new ArgumentException("Sample is not in unit hypercube", nameof(sample));
        double nn = iterative ? n + 1 : n;
        switch (method)
        {
            case "CD":
            {
                var d1 = RowSum(n, i =>
                {
                    var p = 1.0;
                    for (var k = 0; k < d; k++)
                    {
                        var z = Math.Abs(x[i * d + k] - 0.5);
                        p *= 1 + 0.5 * z - 0.5 * z * z;
                    }
                    return p;
                });
                var d2 = PairSum(n, i =>
                {
                    var row = 0.0;
                    for (var j = 0; j < n; j++)
                    {
                        var p = 1.0;
                        for (var k = 0; k < d; k++)
                        {
                            double a = x[i * d + k], b = x[j * d + k];
                            p *= 1 + 0.5 * Math.Abs(a - 0.5) + 0.5 * Math.Abs(b - 0.5) - 0.5 * Math.Abs(a - b);
                        }
                        row += p;
                    }
                    return row;
                });
                return Math.Pow(13.0 / 12.0, d) - 2.0 / nn * d1 + 1.0 / (nn * nn) * d2;
            }
            case "WD":
            {
                var d2 = PairSum(n, i =>
                {
                    var row = 0.0;
                    for (var j = 0; j < n; j++)
                    {
                        var p = 1.0;
                        for (var k = 0; k < d; k++)
                        {
                            var t = Math.Abs(x[i * d + k] - x[j * d + k]);
                            p *= 1.5 - t * (1 - t);
                        }
                        row += p;
                    }
                    return row;
                });
                return -Math.Pow(4.0 / 3.0, d) + 1.0 / (nn * nn) * d2;
            }
            case "MD":
            {
                var d1 = RowSum(n, i =>
                {
                    var p = 1.0;
                    for (var k = 0; k < d; k++)
                    {
                        var z = Math.Abs(x[i * d + k] - 0.5);
                        p *= 5.0 / 3.0 - 0.25 * z - 0.25 * z * z;
                    }
                    return p;
                });
                var d2 = PairSum(n, i =>
                {
                    var row = 0.0;
                    for (var j = 0; j < n; j++)
                    {
                        var p = 1.0;
                        for (var k = 0; k < d; k++)
                        {
                            double a = x[i * d + k], b = x[j * d + k];
                            var t = Math.Abs(a - b);
                            p *= 15.0 / 8.0 - 0.25 * Math.Abs(a - 0.5) - 0.25 * Math.Abs(b - 0.5) - 0.75 * t + 0.5 * t * t;
                        }
                        row += p;
                    }
                    return row;
                });
                return Math.Pow(19.0 / 12.0, d) - 2.0 / nn * d1 + 1.0 / (nn * nn) * d2;
            }
            case "L2-star":
            {
                var d1 = RowSum(n, i =>
                {
                    var p = 1.0;
                    for (var k = 0; k < d; k++) p *= 1 - x[i * d + k] * x[i * d + k];
                    return p;
                });
                var d2 = PairSum(n, i =>
                {
                    var row = 0.0;
                    for (var j = 0; j < n; j++)
                    {
                        var p = 1.0;
                        for (var k = 0; k < d; k++) p *= 1 - Math.Max(x[i * d + k], x[j * d + k]);
                        row += p;
                    }
                    return row;
                });
                return Math.Sqrt(Math.Pow(3, -d) - Math.Pow(2, 1 - d) / nn * d1 + 1.0 / (nn * nn) * d2);
            }
            default:
                throw new ArgumentException($"'{method}' is not a valid method. It must be one of CD, WD, MD, L2-star", nameof(method));
        }
    }

    private static double RowSum(int n, Func<int, double> f)
    {
        var s = 0.0;
        for (var i = 0; i < n; i++) s += f(i);
        return s;
    }

    private static double PairSum(int n, Func<int, double> rowSum)
    {
        // Full double sum (both orders and the diagonal) as scipy; rows are independent, so parallelise.
        var rows = new double[n];
        void Row(int i) => rows[i] = rowSum(i);
        if ((long)n * n >= 1 << 14) Parallel.For(0, n, Row);
        else for (var i = 0; i < n; i++) Row(i);
        var t = 0.0;
        foreach (var v in rows) t += v;
        return t;
    }

    /// <summary>
    /// Centred discrepancy after appending <paramref name="xNew"/> to <paramref name="sample"/>, in O(n d),
    /// given <paramref name="initialDisc"/> = <c>Discrepancy(sample, iterative: true)</c>; scipy's
    /// <c>qmc.update_discrepancy</c>.
    /// </summary>
    public static double UpdateDiscrepancy(double[] xNew, NdArray sample, double initialDisc)
    {
        ArgumentNullException.ThrowIfNull(xNew);
        ArgumentNullException.ThrowIfNull(sample);
        var n = sample.Shape[0] + 1;
        var d = sample.Shape[1];
        if (xNew.Length != d) throw new ArgumentException("x_new and sample must be broadcastable", nameof(xNew));
        foreach (var v in xNew) if (!(v >= 0 && v <= 1)) throw new ArgumentException("x_new is not in unit hypercube", nameof(xNew));
        var x = sample.AsContiguous().ToArray();
        var prod = 1.0;
        for (var k = 0; k < d; k++)
        {
            var z = Math.Abs(xNew[k] - 0.5);
            prod *= 1 + 0.5 * z - 0.5 * z * z;
        }
        var disc1 = -2.0 / n * prod;
        var disc2 = 0.0;
        for (var i = 0; i < n - 1; i++)
        {
            prod = 1;
            for (var k = 0; k < d; k++)
                prod *= 1 + 0.5 * Math.Abs(xNew[k] - 0.5) + 0.5 * Math.Abs(x[i * d + k] - 0.5) - 0.5 * Math.Abs(xNew[k] - x[i * d + k]);
            disc2 += prod;
        }
        disc2 *= 2.0 / ((double)n * n);
        prod = 1;
        for (var k = 0; k < d; k++) prod *= 1 + Math.Abs(xNew[k] - 0.5);
        var disc3 = 1.0 / ((double)n * n) * prod;
        return initialDisc + disc1 + disc2 + disc3;
    }

    // ------------------------------------------------------------------ scaling

    /// <summary>
    /// Maps a sample from [0, 1]^d to the box [l, u] (or back with <paramref name="reverse"/>),
    /// scipy's <c>qmc.scale</c>. Points outside the source box are rejected rather than extrapolated.
    /// </summary>
    public static NdArray Scale(NdArray sample, double[] lBounds, double[] uBounds, bool reverse = false)
    {
        ArgumentNullException.ThrowIfNull(sample);
        ArgumentNullException.ThrowIfNull(lBounds);
        ArgumentNullException.ThrowIfNull(uBounds);
        if (sample.Rank != 2) throw new ArgumentException("Sample is not a 2D array", nameof(sample));
        var n = sample.Shape[0];
        var d = sample.Shape[1];
        if (lBounds.Length == 1 && d > 1) lBounds = Enumerable.Repeat(lBounds[0], d).ToArray();
        if (uBounds.Length == 1 && d > 1) uBounds = Enumerable.Repeat(uBounds[0], d).ToArray();
        if (lBounds.Length != d || uBounds.Length != d) throw new ArgumentException("Sample dimension is different than bounds dimension");
        for (var k = 0; k < d; k++) if (!(lBounds[k] < uBounds[k])) throw new ArgumentException("Bounds are not consistent 'l_bounds' < 'u_bounds'");
        var x = sample.AsContiguous().ToArray();
        var r = new double[x.Length];
        for (var i = 0; i < n; i++)
            for (var k = 0; k < d; k++)
            {
                var v = x[i * d + k];
                if (!reverse)
                {
                    if (!(v >= 0 && v <= 1)) throw new ArgumentException("Sample is not in unit hypercube", nameof(sample));
                    r[i * d + k] = v * (uBounds[k] - lBounds[k]) + lBounds[k];
                }
                else
                {
                    if (!(v >= lBounds[k] && v <= uBounds[k])) throw new ArgumentException("Sample is out of bounds", nameof(sample));
                    r[i * d + k] = (v - lBounds[k]) / (uBounds[k] - lBounds[k]);
                }
            }
        return new NdArray(r, n, d);
    }

    // ------------------------------------------------------------------ helpers for the engines

    /// <summary>The first <paramref name="count"/> primes (scipy's <c>n_primes</c>).</summary>
    public static int[] FirstPrimes(int count)
    {
        var r = new List<int>(count);
        for (var c = 2; r.Count < count; c++) if (IsPrime(c)) r.Add(c);
        return r.ToArray();
    }

    /// <summary>Trial-division primality test.</summary>
    public static bool IsPrime(int n)
    {
        if (n < 2) return false;
        if (n % 2 == 0) return n == 2;
        for (var f = 3; (long)f * f <= n; f += 2) if (n % f == 0) return false;
        return true;
    }

    /// <summary>
    /// scipy's <c>_random_cd</c>: random coordinate swaps between two rows of one column, kept when the
    /// centred discrepancy decreases; stops after <paramref name="nNoChange"/> failures in a row. Swaps keep
    /// each column a permutation, so a Latin hypercube stays one. The change in discrepancy is computed
    /// exactly in O(n d) per trial from the rows the swap touches.
    /// </summary>
    internal static void RandomCdOptimize(double[] x, int n, int d, GraviRandom rng, int nIters = 100_000, int nNoChange = 100)
    {
        if (d <= 1 || n <= 1) return;
        // Row terms of the CD: r(i) = Π (1 + ½|zᵢ| − ½zᵢ²) and pair terms p(i,j) = Π (1 + ½|zᵢ| + ½|zⱼ| − ½|xᵢ − xⱼ|).
        double RowTerm(int i)
        {
            var p = 1.0;
            for (var k = 0; k < d; k++) { var z = Math.Abs(x[i * d + k] - 0.5); p *= 1 + 0.5 * z - 0.5 * z * z; }
            return p;
        }
        double PairTerm(int i, int j)
        {
            var p = 1.0;
            for (var k = 0; k < d; k++)
            {
                double a = x[i * d + k], b = x[j * d + k];
                p *= 1 + 0.5 * Math.Abs(a - 0.5) + 0.5 * Math.Abs(b - 0.5) - 0.5 * Math.Abs(a - b);
            }
            return p;
        }
        // Contribution of rows i1, i2 to −(2/n)Σ r + (1/n²)ΣΣ p.
        double Partial(int i1, int i2)
        {
            var s = -2.0 / n * (RowTerm(i1) + RowTerm(i2));
            var pairs = 0.0;
            for (var j = 0; j < n; j++)
            {
                if (j == i1 || j == i2) continue;
                pairs += 2 * (PairTerm(i1, j) + PairTerm(i2, j));
            }
            pairs += PairTerm(i1, i1) + PairTerm(i2, i2) + 2 * PairTerm(i1, i2);
            return s + pairs / ((double)n * n);
        }
        var noChange = 0;
        for (var it = 0; it < nIters && noChange < nNoChange; it++)
        {
            var col = rng.Next(d);
            var r1 = rng.Next(n);
            var r2 = rng.Next(n);
            if (r1 == r2) { noChange++; continue; }
            var before = Partial(r1, r2);
            (x[r1 * d + col], x[r2 * d + col]) = (x[r2 * d + col], x[r1 * d + col]);
            var after = Partial(r1, r2);
            if (after < before) noChange = 0;
            else
            {
                (x[r1 * d + col], x[r2 * d + col]) = (x[r2 * d + col], x[r1 * d + col]);
                noChange++;
            }
        }
    }
}

/// <summary>
/// Quasi-Monte Carlo draws from a multivariate normal (scipy's <c>qmc.MultivariateNormalQMC</c>): QMC
/// points mapped through Φ⁻¹ (or Box–Muller) and correlated by a Cholesky (or eigen) root of the covariance.
/// </summary>
public sealed class MultivariateNormalQmc
{
    private readonly double[] _mean;
    private readonly double[]? _root; // d×d, row-major: sample = z · root + mean
    private readonly bool _invTransform;

    /// <summary>Creates the sampler.</summary>
    /// <param name="mean">Mean vector.</param>
    /// <param name="cov">Covariance (symmetric PSD); null for the identity.</param>
    /// <param name="invTransform">Inverse-CDF transform (default) or Box–Muller.</param>
    /// <param name="engine">The underlying engine (default a scrambled Sobol' of the needed dimension).</param>
    /// <param name="seed">Seed of the default engine.</param>
    public MultivariateNormalQmc(double[] mean, NdArray? cov = null, bool invTransform = true, QmcEngine? engine = null, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(mean);
        _mean = (double[])mean.Clone();
        var d = mean.Length;
        if (cov is not null)
        {
            var c = cov.AsContiguous().ToArray();
            if (c.Length != d * d) throw new ArgumentException("Dimension mismatch between mean and covariance.", nameof(cov));
            for (var i = 0; i < d; i++)
                for (var j = 0; j < d; j++)
                    if (Math.Abs(c[i * d + j] - c[j * d + i]) > 1e-8 + 1e-5 * Math.Abs(c[j * d + i])) throw new ArgumentException("Covariance matrix is not symmetric.", nameof(cov));
            _root = CholeskyUpper(c, d) ?? EigenRoot(c, d);
        }
        _invTransform = invTransform;
        var engineDim = invTransform ? d : 2 * ((d + 1) / 2);
        Engine = engine ?? new SobolEngine(engineDim, scramble: true, bits: 30, seed: seed);
        if (Engine.D != engineDim) throw new ArgumentException($"Dimension of `engine` must be consistent with dimensions of mean and covariance. If `inv_transform` is False, it must be an even number.", nameof(engine));
        D = d;
    }

    /// <summary>Dimension.</summary>
    public int D { get; }

    /// <summary>The underlying QMC engine.</summary>
    public QmcEngine Engine { get; }

    /// <summary>Draws n points, n×d.</summary>
    public NdArray Random(int n = 1)
    {
        var u = Engine.Random(n).AsContiguous().ToArray();
        var ed = Engine.D;
        var z = new double[n * D];
        if (_invTransform)
            for (var i = 0; i < u.Length; i++) z[i] = Special.Ndtri(0.5 + (1 - 1e-10) * (u[i] - 0.5));
        else
            for (var i = 0; i < n; i++)
                for (var k = 0; 2 * k < ed; k++)
                {
                    var r = Math.Sqrt(-2 * Math.Log(u[i * ed + 2 * k]));
                    var th = 2 * Math.PI * u[i * ed + 2 * k + 1];
                    if (2 * k < D) z[i * D + 2 * k] = r * Math.Cos(th);
                    if (2 * k + 1 < D) z[i * D + 2 * k + 1] = r * Math.Sin(th);
                }
        var outp = new double[n * D];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < D; j++)
            {
                double s;
                if (_root is null) s = z[i * D + j];
                else
                {
                    s = 0;
                    for (var k = 0; k < D; k++) s += z[i * D + k] * _root[k * D + j];
                }
                outp[i * D + j] = s + _mean[j];
            }
        return new NdArray(outp, n, D);
    }

    /// <summary>Upper-triangular Lᵀ of the Cholesky factor, or null when not positive definite.</summary>
    private static double[]? CholeskyUpper(double[] a, int n)
    {
        var l = new double[n * n];
        for (var j = 0; j < n; j++)
        {
            var s = a[j * n + j];
            for (var k = 0; k < j; k++) s -= l[j * n + k] * l[j * n + k];
            if (!(s > 0)) return null;
            l[j * n + j] = Math.Sqrt(s);
            for (var i = j + 1; i < n; i++)
            {
                var t = a[i * n + j];
                for (var k = 0; k < j; k++) t -= l[i * n + k] * l[j * n + k];
                l[i * n + j] = t / l[j * n + j];
            }
        }
        var u = new double[n * n];
        for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) u[j * n + i] = l[i * n + j];
        return u;
    }

    private static double[] EigenRoot(double[] a, int n)
    {
        var (vals, vecs) = JacobiEigen(a, n);
        if (vals.Any(v => v < -1e-8)) throw new ArgumentException("Covariance matrix not PSD.");
        // root = (V · diag(√λ))ᵀ so that rootᵀ root = V Λ Vᵀ.
        var r = new double[n * n];
        for (var k = 0; k < n; k++)
        {
            var s = Math.Sqrt(Math.Max(vals[k], 0));
            for (var j = 0; j < n; j++) r[k * n + j] = vecs[j * n + k] * s;
        }
        return r;
    }

    /// <summary>Cyclic Jacobi eigen-decomposition of a small symmetric matrix, to full precision.</summary>
    private static (double[] Values, double[] Vectors) JacobiEigen(double[] a0, int n)
    {
        var a = (double[])a0.Clone();
        var v = new double[n * n];
        for (var i = 0; i < n; i++) v[i * n + i] = 1;
        for (var sweep = 0; sweep < 100; sweep++)
        {
            var off = 0.0;
            for (var p = 0; p < n; p++) for (var q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
            if (off < 1e-300) break;
            for (var p = 0; p < n; p++)
                for (var q = p + 1; q < n; q++)
                {
                    var apq = a[p * n + q];
                    if (apq == 0) continue;
                    var theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
                    var t = Math.Sign(theta == 0 ? 1 : theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
                    var c = 1 / Math.Sqrt(t * t + 1);
                    var s = t * c;
                    for (var k = 0; k < n; k++)
                    {
                        double akp = a[k * n + p], akq = a[k * n + q];
                        a[k * n + p] = c * akp - s * akq;
                        a[k * n + q] = s * akp + c * akq;
                    }
                    for (var k = 0; k < n; k++)
                    {
                        double apk = a[p * n + k], aqk = a[q * n + k];
                        a[p * n + k] = c * apk - s * aqk;
                        a[q * n + k] = s * apk + c * aqk;
                    }
                    for (var k = 0; k < n; k++)
                    {
                        double vkp = v[k * n + p], vkq = v[k * n + q];
                        v[k * n + p] = c * vkp - s * vkq;
                        v[k * n + q] = s * vkp + c * vkq;
                    }
                }
        }
        var vals = new double[n];
        for (var i = 0; i < n; i++) vals[i] = a[i * n + i];
        return (vals, v);
    }
}

/// <summary>
/// Quasi-Monte Carlo multinomial draws (scipy's <c>qmc.MultinomialQMC</c>): each draw categorises
/// <c>nTrials</c> one-dimensional QMC points by the cumulative probabilities.
/// </summary>
public sealed class MultinomialQmc
{
    private readonly double[] _cum;

    /// <summary>Creates the sampler.</summary>
    public MultinomialQmc(double[] pvals, int nTrials, QmcEngine? engine = null, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(pvals);
        if (pvals.Any(p => p < 0)) throw new ArgumentException("Elements of pvals must be non-negative.", nameof(pvals));
        if (Math.Abs(pvals.Sum() - 1) > 1e-8) throw new ArgumentException("Elements of pvals must sum to 1.", nameof(pvals));
        Pvals = (double[])pvals.Clone();
        NTrials = nTrials;
        Engine = engine ?? new SobolEngine(1, scramble: true, bits: 30, seed: seed);
        if (Engine.D != 1) throw new ArgumentException("Dimension of `engine` must be 1.", nameof(engine));
        _cum = new double[pvals.Length];
        var acc = 0.0;
        for (var i = 0; i < pvals.Length; i++) { acc += pvals[i]; _cum[i] = acc; }
    }

    /// <summary>Category probabilities.</summary>
    public double[] Pvals { get; }

    /// <summary>Trials per draw.</summary>
    public int NTrials { get; }

    /// <summary>The underlying engine.</summary>
    public QmcEngine Engine { get; }

    /// <summary>Draws n count vectors, n × k.</summary>
    public NdArray Random(int n = 1)
    {
        var k = Pvals.Length;
        var r = new double[n * k];
        for (var i = 0; i < n; i++)
        {
            var draws = Engine.Random(NTrials).AsContiguous().ToArray();
            foreach (var u in draws)
            {
                var j = 0;
                while (j < k - 1 && u > _cum[j]) j++;
                r[i * k + j]++;
            }
        }
        return new NdArray(r, n, k);
    }
}
