using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.StatsDistMath;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Eigen-decomposition of a covariance matrix for densities, scipy's <c>_PSD</c>: eigenvalues
/// below 1e6·eps·max|λ| count as zero, which gives the pseudo-determinant and a whitening map that
/// also work for singular matrices.
/// </summary>
internal sealed class StatsPsd
{
    public StatsPsd(NdArray cov, bool allowSingular, string name = "cov")
    {
        if (cov.Shape.Length != 2 || cov.Shape[0] != cov.Shape[1]) throw new ArgumentException($"{name} must be a square matrix.", name);
        Dim = cov.Shape[0];
        var c = cov.AsContiguous().AsSpan();
        for (var i = 0; i < Dim; i++)
            for (var j = 0; j < i; j++)
                if (Math.Abs(c[i * Dim + j] - c[j * Dim + i]) > 1e-10 * (Math.Abs(c[i * Dim + j]) + Math.Abs(c[j * Dim + i]) + 1e-300))
                    throw new ArgumentException($"{name} must be symmetric.", name);
        var eig = Linalg.Eigh(cov);
        var s = eig.Values;
        var v = eig.Vectors.AsContiguous().AsSpan();
        var maxAbs = s.Max(Math.Abs);
        var eps = 1e6 * 2.220446049250313e-16 * maxAbs;
        if (s.Min() < -eps) throw new ArgumentException($"{name} is not positive semi-definite.", name);
        var keep = new List<int>();
        for (var i = 0; i < Dim; i++) if (s[i] > eps) keep.Add(i);
        Rank = keep.Count;
        if (Rank < Dim && !allowSingular) throw new ArgumentException($"{name} is singular; pass allowSingular: true.", name);
        LogPdet = keep.Sum(i => Math.Log(s[i]));
        Whiten = new double[Dim * Rank];
        SqrtFactor = new double[Dim * Dim];
        Null = new double[Dim * (Dim - Rank)];
        for (var r = 0; r < Dim; r++)
        {
            for (var k = 0; k < Rank; k++) Whiten[r * Rank + k] = v[r * Dim + keep[k]] / Math.Sqrt(s[keep[k]]);
            for (var k = 0; k < Dim; k++) SqrtFactor[r * Dim + k] = v[r * Dim + k] * Math.Sqrt(Math.Max(s[k], 0));
            var nk = 0;
            for (var k = 0; k < Dim; k++) if (!(s[k] > eps)) Null[r * (Dim - Rank) + nk++] = v[r * Dim + k];
        }
    }

    public int Dim { get; }
    public int Rank { get; }
    public double LogPdet { get; }
    /// <summary>U with Σ⁺ = U Uᵀ, row-major Dim × Rank.</summary>
    public double[] Whiten { get; }
    /// <summary>V diag(√λ), so V diag(√λ) z has covariance Σ (singular Σ included).</summary>
    public double[] SqrtFactor { get; }
    /// <summary>Basis of the null space, Dim × (Dim − Rank).</summary>
    public double[] Null { get; }

    /// <summary>‖Uᵀ d‖², the Mahalanobis distance through the pseudo-inverse.</summary>
    public double Mahalanobis(ReadOnlySpan<double> d)
    {
        var total = 0.0;
        for (var k = 0; k < Rank; k++)
        {
            var t = 0.0;
            for (var r = 0; r < Dim; r++) t += d[r] * Whiten[r * Rank + k];
            total += t * t;
        }
        return total;
    }

    /// <summary>True when d lies (numerically) in the range of Σ — the support of a singular normal.</summary>
    public bool InSupport(ReadOnlySpan<double> d)
    {
        var nn = Dim - Rank;
        if (nn == 0) return true;
        var scale = 0.0;
        foreach (var x in d) scale = Math.Max(scale, Math.Abs(x));
        for (var k = 0; k < nn; k++)
        {
            var t = 0.0;
            for (var r = 0; r < Dim; r++) t += d[r] * Null[r * nn + k];
            if (Math.Abs(t) > 1e-8 * Math.Max(1.0, scale)) return false;
        }
        return true;
    }
}

/// <summary>The multivariate normal distribution, scipy's <c>multivariate_normal</c>.</summary>
/// <remarks>
/// <para>The density goes through the eigen-decomposition of Σ (as scipy's <c>_PSD</c>), so a
/// singular covariance with <c>allowSingular</c> gives the density on its supporting subspace and
/// −∞ off it.</para>
/// <para><see cref="Cdf"/> is Genz's (1992) separation-of-variables integral on a randomised
/// Richtmyer lattice with variables ordered by Genz–Bretz prioritisation — the same algorithm family
/// as scipy's <c>mvndst</c>, not a port of it: both are randomised quasi-Monte Carlo with an error
/// target of 1e-5, so results agree to that level rather than to the last digit.</para>
/// </remarks>
public sealed class MultivariateNormalDistribution
{
    private readonly StatsPsd? _psd;
    private readonly Covariance? _covObj;
    private readonly double[] _cov;

    /// <summary>Multivariate normal with the given mean and covariance.</summary>
    public MultivariateNormalDistribution(double[] mean, NdArray cov, bool allowSingular = false)
    {
        ArgumentNullException.ThrowIfNull(mean);
        ArgumentNullException.ThrowIfNull(cov);
        _psd = new StatsPsd(cov, allowSingular);
        if (mean.Length != _psd.Dim) throw new ArgumentException("mean and cov dimensions differ.", nameof(mean));
        MeanVector = (double[])mean.Clone();
        _cov = cov.AsContiguous().ToArray();
    }

    /// <summary>
    /// Multivariate normal with the covariance given as a <see cref="SciNet.Covariance"/> object (scipy's
    /// <c>cov=Covariance</c>): densities and draws use its whitening/colouring directly, so a precision,
    /// diagonal, Cholesky or eigen form is never re-factorised. A singular form (zero diagonal entries or
    /// eigenvalues) gives the density on its support, as <c>allowSingular</c> does.
    /// </summary>
    public MultivariateNormalDistribution(double[] mean, Covariance cov)
    {
        ArgumentNullException.ThrowIfNull(mean);
        ArgumentNullException.ThrowIfNull(cov);
        if (mean.Length != cov.Dim) throw new ArgumentException("mean and cov dimensions differ.", nameof(mean));
        _covObj = cov;
        MeanVector = (double[])mean.Clone();
        _cov = cov.CovarianceMatrix.AsContiguous().ToArray();
    }

    /// <summary>The covariance object this distribution was built from (null when built from a matrix).</summary>
    public Covariance? CovarianceObject => _covObj;

    /// <summary>The dimension.</summary>
    public int Dim => _psd?.Dim ?? _covObj!.Dim;
    /// <summary>The mean vector.</summary>
    public double[] MeanVector { get; }
    /// <summary>The covariance matrix (a copy).</summary>
    public NdArray Cov => new((double[])_cov.Clone(), Dim, Dim);
    /// <summary>Rank of the covariance.</summary>
    public int Rank => _psd?.Rank ?? _covObj!.Rank;

    /// <summary>Log density at one point.</summary>
    public double LogPdf(ReadOnlySpan<double> x)
    {
        if (x.Length != Dim) throw new ArgumentException("x has the wrong dimension.", nameof(x));
        Span<double> d = Dim <= 64 ? stackalloc double[Dim] : new double[Dim];
        for (var i = 0; i < Dim; i++) d[i] = x[i] - MeanVector[i];
        if (_covObj != null)
        {
            if (!_covObj.InSupport(d)) return double.NegativeInfinity;
            var w = _covObj.Whiten(d.ToArray());
            var m = 0.0;
            foreach (var v in w) m += v * v;
            return -0.5 * (_covObj.Rank * Math.Log(2 * Math.PI) + _covObj.LogPdet + m);
        }
        if (!_psd!.InSupport(d)) return double.NegativeInfinity;
        return -0.5 * (_psd.Rank * Math.Log(2 * Math.PI) + _psd.LogPdet + _psd.Mahalanobis(d));
    }

    /// <summary>Density at one point.</summary>
    public double Pdf(ReadOnlySpan<double> x) => Math.Exp(LogPdf(x));

    /// <summary>Log densities of the rows of an (n, dim) array.</summary>
    public double[] LogPdf(NdArray x) => StatsMultivariate.Rows(x, Dim, r => LogPdf(r));

    /// <summary>Densities of the rows of an (n, dim) array.</summary>
    public double[] Pdf(NdArray x) => StatsMultivariate.Rows(x, Dim, r => Pdf(r));

    /// <summary>Differential entropy, ½(r(1 + log 2π) + log pdet Σ).</summary>
    public double Entropy() => 0.5 * (Rank * (1 + Math.Log(2 * Math.PI)) + (_psd?.LogPdet ?? _covObj!.LogPdet));

    /// <summary>
    /// P(lower ≤ X ≤ x), scipy's <c>cdf</c> (with <c>lower_limit</c>). Randomised quasi-Monte Carlo;
    /// iterates until the error estimate is below max(abseps, releps·P) or <paramref name="maxpts"/>
    /// integrand evaluations (default 1,000,000·dim) are used. The seed fixes the random shifts.
    /// </summary>
    public double Cdf(double[] x, double[]? lowerLimit = null, int? maxpts = null, double abseps = 1e-5, double releps = 1e-5, int? seed = null)
        => StatsGenz.Mvn(x, lowerLimit, MeanVector, _cov, Dim, maxpts ?? 1_000_000 * Dim, abseps, releps, seed).Value;

    /// <summary>log of <see cref="Cdf"/>.</summary>
    public double LogCdf(double[] x, double[]? lowerLimit = null, int? maxpts = null, double abseps = 1e-5, double releps = 1e-5, int? seed = null)
        => Math.Log(Cdf(x, lowerLimit, maxpts, abseps, releps, seed));

    /// <summary><paramref name="size"/> draws as an (size, dim) array: mean + V·diag(√λ)·z.</summary>
    public NdArray Rvs(int size, GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        var n = Dim;
        var r = new double[size * n];
        var z = new double[n];
        if (_covObj != null)
        {
            for (var s = 0; s < size; s++)
            {
                for (var k = 0; k < n; k++) z[k] = StatsRandom.Normal(rng);
                var c = _covObj.Colorize(z);
                for (var i = 0; i < n; i++) r[s * n + i] = MeanVector[i] + c[i];
            }
            return new NdArray(r, size, n);
        }
        var f = _psd!.SqrtFactor;
        for (var s = 0; s < size; s++)
        {
            for (var k = 0; k < n; k++) z[k] = StatsRandom.Normal(rng);
            for (var i = 0; i < n; i++)
            {
                var t = MeanVector[i];
                for (var k = 0; k < n; k++) t += f[i * n + k] * z[k];
                r[s * n + i] = t;
            }
        }
        return new NdArray(r, size, n);
    }

    /// <summary>Draws with a seed (null: fresh).</summary>
    public NdArray Rvs(int size, int? seed = null) => Rvs(size, StatsDistributions.MakeRng(seed));
}

/// <summary>Shared plumbing for the multivariate types.</summary>
internal static class StatsMultivariate
{
    public static double[] Rows(NdArray x, int dim, Func<double[], double> f)
    {
        ArgumentNullException.ThrowIfNull(x);
        var c = x.AsContiguous().ToArray();
        if (x.Shape.Length == 1)
        {
            if (x.Shape[0] != dim) throw new ArgumentException("x has the wrong dimension.", nameof(x));
            return [f(c)];
        }
        if (x.Shape.Length != 2 || x.Shape[1] != dim) throw new ArgumentException("x must have shape (n, dim).", nameof(x));
        var n = x.Shape[0];
        var r = new double[n];
        for (var i = 0; i < n; i++) r[i] = f(c.AsSpan(i * dim, dim).ToArray());
        return r;
    }

    /// <summary>log Γ_p(a), the multivariate gamma function.</summary>
    public static double MultiGammaLn(double a, int p)
    {
        var s = p * (p - 1) / 4.0 * Math.Log(Math.PI);
        for (var j = 1; j <= p; j++) s += Special.GammaLn(a + (1 - j) / 2.0);
        return s;
    }

    /// <summary>Σ ψ(a + (1 − j)/2), the derivative of log Γ_p.</summary>
    public static double MultiDigamma(double a, int p)
    {
        var s = 0.0;
        for (var j = 1; j <= p; j++) s += Special.Digamma(a + (1 - j) / 2.0);
        return s;
    }

    /// <summary>Lower Cholesky factor of a symmetric positive-definite matrix (row-major); throws when not SPD.</summary>
    public static double[] CholeskyLower(double[] a, int n, string name)
    {
        var l = new double[n * n];
        for (var j = 0; j < n; j++)
        {
            var d = a[j * n + j];
            for (var k = 0; k < j; k++) d -= l[j * n + k] * l[j * n + k];
            if (!(d > 0)) throw new ArgumentException($"{name} must be symmetric positive definite.", name);
            var ljj = Math.Sqrt(d);
            l[j * n + j] = ljj;
            for (var i = j + 1; i < n; i++)
            {
                var s = a[i * n + j];
                for (var k = 0; k < j; k++) s -= l[i * n + k] * l[j * n + k];
                l[i * n + j] = s / ljj;
            }
        }
        return l;
    }

    /// <summary>log det from a Cholesky factor.</summary>
    public static double LogDetFromCholesky(double[] l, int n)
    {
        var s = 0.0;
        for (var i = 0; i < n; i++) s += Math.Log(l[i * n + i]);
        return 2 * s;
    }

    /// <summary>Solves L Lᵀ X = B for X (B row-major n × m), returning X.</summary>
    public static double[] CholeskySolve(double[] l, int n, double[] b, int m)
    {
        var x = (double[])b.Clone();
        for (var c = 0; c < m; c++)
        {
            for (var i = 0; i < n; i++)
            {
                var s = x[i * m + c];
                for (var k = 0; k < i; k++) s -= l[i * n + k] * x[k * m + c];
                x[i * m + c] = s / l[i * n + i];
            }
            for (var i = n - 1; i >= 0; i--)
            {
                var s = x[i * m + c];
                for (var k = i + 1; k < n; k++) s -= l[k * n + i] * x[k * m + c];
                x[i * m + c] = s / l[i * n + i];
            }
        }
        return x;
    }
}

/// <summary>
/// Genz's (1992) algorithm for multivariate normal probabilities over a box: Cholesky separation of
/// variables, Genz–Bretz variable prioritisation (narrowest expected interval first) and a
/// randomised Richtmyer lattice, repeated over random shifts for an error estimate.
/// </summary>
internal static class StatsGenz
{
    private static readonly double[] Primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313];

    public static (double Value, double Error) Mvn(double[] upper, double[]? lower, double[] mean, double[] cov, int n, int maxpts, double abseps, double releps, int? seed)
    {
        ArgumentNullException.ThrowIfNull(upper);
        if (upper.Length != n || (lower != null && lower.Length != n)) throw new ArgumentException("Limits have the wrong dimension.");
        var a = new double[n];
        var b = new double[n];
        for (var i = 0; i < n; i++)
        {
            a[i] = (lower?[i] ?? double.NegativeInfinity) - mean[i];
            b[i] = upper[i] - mean[i];
            if (a[i] > b[i]) return (0.0, 0.0);
        }
        if (n == 1)
        {
            var sd = Math.Sqrt(cov[0]);
            return (Math.Max(0, Phi(b[0] / sd) - Phi(a[0] / sd)), 0.0);
        }
        // Cholesky with prioritisation: at each step pick the variable with the smallest expected
        // conditional probability, permuting the covariance as we go (Genz & Bretz 2002, §4.1.3).
        var c = (double[])cov.Clone();
        var l = new double[n * n];
        var y = new double[n];
        for (var i = 0; i < n; i++)
        {
            var best = i;
            var bestP = double.PositiveInfinity;
            for (var j = i; j < n; j++)
            {
                var s = c[j * n + j];
                for (var k = 0; k < i; k++) s -= l[j * n + k] * l[j * n + k];
                if (s <= 1e-14 * Math.Max(1e-300, c[j * n + j])) continue;
                var sd = Math.Sqrt(s);
                var mu = 0.0;
                for (var k = 0; k < i; k++) mu += l[j * n + k] * y[k];
                var p = Phi((b[j] - mu) / sd) - Phi((a[j] - mu) / sd);
                if (p < bestP) { bestP = p; best = j; }
            }
            if (best != i)
            {
                Swap(a, i, best); Swap(b, i, best);
                for (var k = 0; k < n; k++) { (c[i * n + k], c[best * n + k]) = (c[best * n + k], c[i * n + k]); }
                for (var k = 0; k < n; k++) { (c[k * n + i], c[k * n + best]) = (c[k * n + best], c[k * n + i]); }
                for (var k = 0; k < i; k++) { (l[i * n + k], l[best * n + k]) = (l[best * n + k], l[i * n + k]); }
            }
            var dii = c[i * n + i];
            for (var k = 0; k < i; k++) dii -= l[i * n + k] * l[i * n + k];
            var lii = dii > 1e-14 * Math.Max(1e-300, c[i * n + i]) ? Math.Sqrt(dii) : 0.0;
            l[i * n + i] = lii;
            for (var j = i + 1; j < n; j++)
            {
                var s = c[j * n + i];
                for (var k = 0; k < i; k++) s -= l[j * n + k] * l[i * n + k];
                l[j * n + i] = lii > 0 ? s / lii : 0.0;
            }
            // Expected value of the i-th variable under its truncated normal, for the next choice.
            if (lii > 0)
            {
                var mu = 0.0;
                for (var k = 0; k < i; k++) mu += l[i * n + k] * y[k];
                var lo = (a[i] - mu) / lii;
                var hi = (b[i] - mu) / lii;
                var pm = Phi(hi) - Phi(lo);
                var phiLo = double.IsInfinity(lo) ? 0 : Math.Exp(-0.5 * lo * lo) / Sqrt2Pi;
                var phiHi = double.IsInfinity(hi) ? 0 : Math.Exp(-0.5 * hi * hi) / Sqrt2Pi;
                y[i] = pm > 1e-300 ? (phiLo - phiHi) / pm : (double.IsInfinity(lo) ? hi : lo);
            }
            else y[i] = 0;
        }

        var rng = StatsDistributions.MakeRng(seed ?? 12345);
        var q = new double[n - 1];
        for (var i = 0; i < n - 1; i++) q[i] = Math.Sqrt(Primes[i % Primes.Length]) % 1.0;
        const int shifts = 12;
        var points = 64;
        double est = 0, err = double.PositiveInfinity;
        var used = 0;
        var w = new double[n];
        var yy = new double[n];
        var shift = new double[n - 1];
        var sums = new double[shifts];
        while (true)
        {
            for (var sIdx = 0; sIdx < shifts; sIdx++)
            {
                for (var i = 0; i < n - 1; i++) shift[i] = rng.NextDouble();
                var total = 0.0;
                for (var p = 1; p <= points; p++)
                {
                    // Antithetic pair of baker-transformed lattice points.
                    for (var anti = 0; anti < 2; anti++)
                    {
                        for (var i = 0; i < n - 1; i++)
                        {
                            var t = (p * q[i] + shift[i]) % 1.0;
                            t = Math.Abs(2 * t - 1);
                            w[i] = anti == 0 ? t : 1 - t;
                        }
                        total += Integrand(a, b, l, n, w, yy);
                    }
                }
                sums[sIdx] = total / (2 * points);
            }
            used += shifts * points * 2;
            var roundMean = sums.Average();
            var v = sums.Sum(s => (s - roundMean) * (s - roundMean)) / (shifts * (shifts - 1));
            // Combine with earlier rounds weighted by their variance.
            var e = 3.5 * Math.Sqrt(v);
            if (double.IsPositiveInfinity(err)) { est = roundMean; err = e; }
            else
            {
                var w1 = 1 / Math.Max(err * err, 1e-300);
                var w2 = 1 / Math.Max(e * e, 1e-300);
                est = (est * w1 + roundMean * w2) / (w1 + w2);
                err = 1 / Math.Sqrt(w1 + w2);
            }
            if (err <= Math.Max(abseps, releps * Math.Abs(est)) || used >= maxpts) break;
            points = (int)(points * 1.5);
        }
        return (Math.Clamp(est, 0, 1), err);
    }

    private static double Integrand(double[] a, double[] b, double[] l, int n, double[] w, double[] y)
    {
        var f = 1.0;
        for (var i = 0; i < n; i++)
        {
            var s = 0.0;
            for (var k = 0; k < i; k++) s += l[i * n + k] * y[k];
            var lii = l[i * n + i];
            double d, e;
            if (lii > 0)
            {
                d = Phi((a[i] - s) / lii);
                e = Phi((b[i] - s) / lii);
            }
            else
            {
                // A degenerate direction: the constraint is either met or not.
                if (a[i] - s <= 0 && b[i] - s >= 0) { d = 0; e = 1; }
                else return 0.0;
            }
            var de = e - d;
            if (de <= 0) return 0.0;
            f *= de;
            if (i < n - 1 && lii > 0)
            {
                var u = Math.Clamp(d + w[i] * de, 1e-300, 1 - 1e-16);
                y[i] = Special.Ndtri(u);
            }
            else y[i] = 0;
        }
        return f;
    }

    private static void Swap(double[] v, int i, int j) => (v[i], v[j]) = (v[j], v[i]);
}

/// <summary>The multivariate Student t distribution, scipy's <c>multivariate_t</c>.</summary>
public sealed class MultivariateTDistribution
{
    private readonly StatsPsd _psd;

    /// <summary>Multivariate t with location, shape matrix and degrees of freedom (df = ∞ is the normal).</summary>
    public MultivariateTDistribution(double[] loc, NdArray shape, double df = 1.0, bool allowSingular = false)
    {
        ArgumentNullException.ThrowIfNull(loc);
        _psd = new StatsPsd(shape, allowSingular, nameof(shape));
        if (loc.Length != _psd.Dim) throw new ArgumentException("loc and shape dimensions differ.", nameof(loc));
        if (!(df > 0)) throw new ArgumentException("df must be positive.", nameof(df));
        Loc = (double[])loc.Clone();
        Df = df;
    }

    /// <summary>Dimension.</summary>
    public int Dim => _psd.Dim;
    /// <summary>Location vector.</summary>
    public double[] Loc { get; }
    /// <summary>Degrees of freedom.</summary>
    public double Df { get; }

    /// <summary>Log density at a point.</summary>
    public double LogPdf(ReadOnlySpan<double> x)
    {
        if (x.Length != Dim) throw new ArgumentException("x has the wrong dimension.", nameof(x));
        var d = new double[Dim];
        for (var i = 0; i < Dim; i++) d[i] = x[i] - Loc[i];
        if (!_psd.InSupport(d)) return double.NegativeInfinity;
        var maha = _psd.Mahalanobis(d);
        double p = _psd.Rank, v = Df;
        if (double.IsPositiveInfinity(v)) return -0.5 * (p * Math.Log(2 * Math.PI) + _psd.LogPdet + maha);
        return Special.GammaLn((v + p) / 2) - Special.GammaLn(v / 2) - p / 2 * Math.Log(v * Math.PI) - 0.5 * _psd.LogPdet
               - (v + p) / 2 * Special.Log1p(maha / v);
    }

    /// <summary>Density at a point.</summary>
    public double Pdf(ReadOnlySpan<double> x) => Math.Exp(LogPdf(x));

    /// <summary>Log densities of the rows of an (n, dim) array.</summary>
    public double[] LogPdf(NdArray x) => StatsMultivariate.Rows(x, Dim, r => LogPdf(r));

    /// <summary>Differential entropy.</summary>
    public double Entropy()
    {
        double p = _psd.Rank, v = Df;
        if (double.IsPositiveInfinity(v)) return 0.5 * (p * (1 + Math.Log(2 * Math.PI)) + _psd.LogPdet);
        return 0.5 * _psd.LogPdet + p / 2 * Math.Log(v * Math.PI) + Special.GammaLn(v / 2) - Special.GammaLn((v + p) / 2)
               + (v + p) / 2 * (Special.Digamma((v + p) / 2) - Special.Digamma(v / 2));
    }

    /// <summary><paramref name="size"/> draws as a (size, dim) array.</summary>
    public NdArray Rvs(int size, GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        var n = Dim;
        var r = new double[size * n];
        var z = new double[n];
        var f = _psd.SqrtFactor;
        for (var s = 0; s < size; s++)
        {
            var w = double.IsPositiveInfinity(Df) ? 1.0 : Math.Sqrt(Df / (2 * StatsRandom.Gamma(rng, Df / 2)));
            for (var k = 0; k < n; k++) z[k] = StatsRandom.Normal(rng);
            for (var i = 0; i < n; i++)
            {
                var t = 0.0;
                for (var k = 0; k < n; k++) t += f[i * n + k] * z[k];
                r[s * n + i] = Loc[i] + w * t;
            }
        }
        return new NdArray(r, size, n);
    }

    /// <summary>Draws with a seed (null: fresh).</summary>
    public NdArray Rvs(int size, int? seed = null) => Rvs(size, StatsDistributions.MakeRng(seed));
}

/// <summary>The Dirichlet distribution on the simplex, scipy's <c>dirichlet</c>.</summary>
public sealed class DirichletDistribution
{
    /// <summary>Dirichlet with concentration vector <paramref name="alpha"/> (all positive).</summary>
    public DirichletDistribution(double[] alpha)
    {
        ArgumentNullException.ThrowIfNull(alpha);
        if (alpha.Length < 2 || alpha.Any(a => !(a > 0) || double.IsInfinity(a))) throw new ArgumentException("alpha needs at least two positive entries.", nameof(alpha));
        Alpha = (double[])alpha.Clone();
    }

    /// <summary>Concentration parameters.</summary>
    public double[] Alpha { get; }
    private double A0 => Alpha.Sum();

    private double LogNorm()
    {
        var s = -Special.GammaLn(A0);
        foreach (var a in Alpha) s += Special.GammaLn(a);
        return s;
    }

    private double[] Complete(ReadOnlySpan<double> x)
    {
        var k = Alpha.Length;
        if (x.Length == k - 1)
        {
            var r = new double[k];
            var s = 0.0;
            for (var i = 0; i < k - 1; i++) { r[i] = x[i]; s += x[i]; }
            r[k - 1] = 1 - s;
            return r;
        }
        if (x.Length != k) throw new ArgumentException("x must have K or K−1 components.", nameof(x));
        var t = x.ToArray();
        if (Math.Abs(t.Sum() - 1) > 1e-9) throw new ArgumentException("x must sum to 1.", nameof(x));
        return t;
    }

    /// <summary>Log density at a point of the simplex (K components, or K−1 with the last implied).</summary>
    public double LogPdf(ReadOnlySpan<double> x)
    {
        var p = Complete(x);
        var s = -LogNorm();
        for (var i = 0; i < p.Length; i++)
        {
            if (p[i] < 0 || p[i] > 1) return double.NegativeInfinity;
            s += Special.Xlogy(Alpha[i] - 1, p[i]);
        }
        return s;
    }

    /// <summary>Density at a point of the simplex.</summary>
    public double Pdf(ReadOnlySpan<double> x) => Math.Exp(LogPdf(x));

    /// <summary>Mean vector α/α₀.</summary>
    public double[] Mean() => Alpha.Select(a => a / A0).ToArray();

    /// <summary>Marginal variances α(α₀ − α)/(α₀²(α₀ + 1)).</summary>
    public double[] Var() => Alpha.Select(a => a * (A0 - a) / (A0 * A0 * (A0 + 1))).ToArray();

    /// <summary>Covariance matrix.</summary>
    public NdArray Cov()
    {
        var k = Alpha.Length;
        var a0 = A0;
        var r = new double[k * k];
        for (var i = 0; i < k; i++)
            for (var j = 0; j < k; j++)
                r[i * k + j] = (i == j ? Alpha[i] * a0 : 0) / (a0 * a0 * (a0 + 1)) - Alpha[i] * Alpha[j] / (a0 * a0 * (a0 + 1));
        return new NdArray(r, k, k);
    }

    /// <summary>Differential entropy log B(α) + (α₀ − K)ψ(α₀) − Σ(αᵢ − 1)ψ(αᵢ).</summary>
    public double Entropy()
    {
        var s = LogNorm() + (A0 - Alpha.Length) * Special.Digamma(A0);
        foreach (var a in Alpha) s -= (a - 1) * Special.Digamma(a);
        return s;
    }

    /// <summary><paramref name="size"/> draws as a (size, K) array (normalised gammas).</summary>
    public NdArray Rvs(int size, GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        var k = Alpha.Length;
        var r = new double[size * k];
        for (var s = 0; s < size; s++)
        {
            var t = 0.0;
            for (var i = 0; i < k; i++) { var g = StatsRandom.Gamma(rng, Alpha[i]); r[s * k + i] = g; t += g; }
            for (var i = 0; i < k; i++) r[s * k + i] /= t;
        }
        return new NdArray(r, size, k);
    }

    /// <summary>Draws with a seed (null: fresh).</summary>
    public NdArray Rvs(int size, int? seed = null) => Rvs(size, StatsDistributions.MakeRng(seed));
}

/// <summary>The multinomial distribution, scipy's <c>multinomial</c>.</summary>
public sealed class MultinomialDistribution
{
    /// <summary>Multinomial with <paramref name="n"/> trials and category probabilities <paramref name="p"/> (summing to 1).</summary>
    public MultinomialDistribution(int n, double[] p)
    {
        ArgumentNullException.ThrowIfNull(p);
        if (n < 0) throw new ArgumentException("n must be non-negative.", nameof(n));
        if (p.Length < 1 || p.Any(v => !(v >= 0 && v <= 1)) || Math.Abs(p.Sum() - 1) > 1e-9) throw new ArgumentException("p must be probabilities summing to 1.", nameof(p));
        N = n;
        P = (double[])p.Clone();
    }

    /// <summary>Number of trials.</summary>
    public int N { get; }
    /// <summary>Category probabilities.</summary>
    public double[] P { get; }

    /// <summary>log P(X = x).</summary>
    public double LogPmf(ReadOnlySpan<double> x)
    {
        if (x.Length != P.Length) throw new ArgumentException("x has the wrong length.", nameof(x));
        var total = 0.0;
        var s = Special.GammaLn(N + 1);
        for (var i = 0; i < x.Length; i++)
        {
            if (x[i] < 0 || x[i] != Math.Floor(x[i])) return double.NegativeInfinity;
            total += x[i];
            s += Special.Xlogy(x[i], P[i]) - Special.GammaLn(x[i] + 1);
        }
        return total == N ? s : double.NegativeInfinity;
    }

    /// <summary>P(X = x).</summary>
    public double Pmf(ReadOnlySpan<double> x) => Math.Exp(LogPmf(x));

    /// <summary>Mean n·p.</summary>
    public double[] Mean() => P.Select(v => N * v).ToArray();

    /// <summary>Covariance n(diag p − p pᵀ).</summary>
    public NdArray Cov()
    {
        var k = P.Length;
        var r = new double[k * k];
        for (var i = 0; i < k; i++)
            for (var j = 0; j < k; j++)
                r[i * k + j] = N * ((i == j ? P[i] : 0) - P[i] * P[j]);
        return new NdArray(r, k, k);
    }

    /// <summary>Entropy −log n! − nΣp log p + Σᵢ Σₓ Binom(x; n, pᵢ) log x! (scipy's formula).</summary>
    public double Entropy()
    {
        var s = -Special.GammaLn(N + 1);
        foreach (var p in P) s -= N * Special.Xlogy(p, p);
        foreach (var p in P)
        {
            if (p == 0 || p == 1) continue;
            for (var x = 2; x <= N; x++)
                s += Math.Exp(StatsLoader.LogDbinom(x, N, p, 1 - p)) * Special.GammaLn(x + 1);
        }
        return s;
    }

    /// <summary><paramref name="size"/> draws as a (size, K) array, by sequential conditional binomials.</summary>
    public NdArray Rvs(int size, GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        var k = P.Length;
        var r = new double[size * k];
        for (var s = 0; s < size; s++)
        {
            var left = (double)N;
            var rem = 1.0;
            for (var i = 0; i < k - 1 && left > 0; i++)
            {
                var pi = rem > 0 ? Math.Clamp(P[i] / rem, 0, 1) : 0;
                var draw = StatsRandom.Binomial(rng, left, pi, new BinomialDistribution(left, pi));
                r[s * k + i] = draw;
                left -= draw;
                rem -= P[i];
            }
            r[s * k + k - 1] += left;
        }
        return new NdArray(r, size, k);
    }

    /// <summary>Draws with a seed (null: fresh).</summary>
    public NdArray Rvs(int size, int? seed = null) => Rvs(size, StatsDistributions.MakeRng(seed));
}

/// <summary>The Wishart distribution, scipy's <c>wishart</c>.</summary>
public sealed class WishartDistribution
{
    private readonly double[] _scale;
    private readonly double[] _chol;

    /// <summary>Wishart with <paramref name="df"/> ≥ p degrees of freedom and positive-definite scale matrix.</summary>
    public WishartDistribution(double df, NdArray scale)
    {
        ArgumentNullException.ThrowIfNull(scale);
        if (scale.Shape.Length != 2 || scale.Shape[0] != scale.Shape[1]) throw new ArgumentException("scale must be square.", nameof(scale));
        Dim = scale.Shape[0];
        if (!(df > Dim - 1)) throw new ArgumentException("df must exceed dim − 1.", nameof(df));
        Df = df;
        _scale = scale.AsContiguous().ToArray();
        _chol = StatsMultivariate.CholeskyLower(_scale, Dim, nameof(scale));
    }

    /// <summary>Dimension p.</summary>
    public int Dim { get; }
    /// <summary>Degrees of freedom.</summary>
    public double Df { get; }

    /// <summary>log density at the positive-definite matrix X.</summary>
    public double LogPdf(NdArray x)
    {
        var p = Dim;
        var xa = x.AsContiguous().ToArray();
        double[] lx;
        try { lx = StatsMultivariate.CholeskyLower(xa, p, nameof(x)); }
        catch (ArgumentException) { return double.NegativeInfinity; }
        var logdetX = StatsMultivariate.LogDetFromCholesky(lx, p);
        var logdetS = StatsMultivariate.LogDetFromCholesky(_chol, p);
        var sinvX = StatsMultivariate.CholeskySolve(_chol, p, xa, p);
        var tr = 0.0;
        for (var i = 0; i < p; i++) tr += sinvX[i * p + i];
        return 0.5 * (Df - p - 1) * logdetX - 0.5 * tr - 0.5 * Df * p * Math.Log(2) - 0.5 * Df * logdetS
               - StatsMultivariate.MultiGammaLn(Df / 2, p);
    }

    /// <summary>Density at X.</summary>
    public double Pdf(NdArray x) => Math.Exp(LogPdf(x));

    /// <summary>Mean df·S.</summary>
    public NdArray Mean() => new(_scale.Select(v => Df * v).ToArray(), Dim, Dim);

    /// <summary>Element-wise variances df(S_ij² + S_ii S_jj).</summary>
    public NdArray Var()
    {
        var p = Dim;
        var r = new double[p * p];
        for (var i = 0; i < p; i++)
            for (var j = 0; j < p; j++)
                r[i * p + j] = Df * (_scale[i * p + j] * _scale[i * p + j] + _scale[i * p + i] * _scale[j * p + j]);
        return new NdArray(r, p, p);
    }

    /// <summary>Differential entropy.</summary>
    public double Entropy()
    {
        var p = Dim;
        var logdetS = StatsMultivariate.LogDetFromCholesky(_chol, p);
        return 0.5 * (p + 1) * logdetS + 0.5 * p * (p + 1) * Math.Log(2) + StatsMultivariate.MultiGammaLn(Df / 2, p)
               - 0.5 * (Df - p - 1) * StatsMultivariate.MultiDigamma(Df / 2, p) + 0.5 * Df * p;
    }

    /// <summary>One draw by the Bartlett decomposition: L A Aᵀ Lᵀ with A lower triangular, χ on the diagonal.</summary>
    internal static double[] Bartlett(GraviRandom rng, double df, double[] chol, int p)
    {
        var a = new double[p * p];
        for (var i = 0; i < p; i++)
        {
            a[i * p + i] = Math.Sqrt(2 * StatsRandom.Gamma(rng, (df - i) / 2));
            for (var j = 0; j < i; j++) a[i * p + j] = StatsRandom.Normal(rng);
        }
        var la = new double[p * p];
        for (var i = 0; i < p; i++)
            for (var j = 0; j <= i; j++)
            {
                var s = 0.0;
                for (var k = j; k <= i; k++) s += chol[i * p + k] * a[k * p + j];
                la[i * p + j] = s;
            }
        var w = new double[p * p];
        for (var i = 0; i < p; i++)
            for (var j = 0; j <= i; j++)
            {
                var s = 0.0;
                for (var k = 0; k <= j; k++) s += la[i * p + k] * la[j * p + k];
                w[i * p + j] = s;
                w[j * p + i] = s;
            }
        return w;
    }

    /// <summary><paramref name="size"/> draws, each a (dim, dim) array.</summary>
    public NdArray[] Rvs(int size, GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        var r = new NdArray[size];
        for (var s = 0; s < size; s++) r[s] = new NdArray(Bartlett(rng, Df, _chol, Dim), Dim, Dim);
        return r;
    }

    /// <summary>Draws with a seed (null: fresh).</summary>
    public NdArray[] Rvs(int size, int? seed = null) => Rvs(size, StatsDistributions.MakeRng(seed));
}

/// <summary>The inverse Wishart distribution, scipy's <c>invwishart</c>.</summary>
public sealed class InvWishartDistribution
{
    private readonly double[] _scale;
    private readonly double[] _chol;
    private readonly double[] _cholInv; // Cholesky factor of S⁻¹, for sampling

    /// <summary>Inverse Wishart with <paramref name="df"/> &gt; p − 1 and positive-definite scale.</summary>
    public InvWishartDistribution(double df, NdArray scale)
    {
        ArgumentNullException.ThrowIfNull(scale);
        if (scale.Shape.Length != 2 || scale.Shape[0] != scale.Shape[1]) throw new ArgumentException("scale must be square.", nameof(scale));
        Dim = scale.Shape[0];
        if (!(df > Dim - 1)) throw new ArgumentException("df must exceed dim − 1.", nameof(df));
        Df = df;
        _scale = scale.AsContiguous().ToArray();
        _chol = StatsMultivariate.CholeskyLower(_scale, Dim, nameof(scale));
        var id = new double[Dim * Dim];
        for (var i = 0; i < Dim; i++) id[i * Dim + i] = 1;
        var inv = StatsMultivariate.CholeskySolve(_chol, Dim, id, Dim);
        _cholInv = StatsMultivariate.CholeskyLower(inv, Dim, nameof(scale));
    }

    /// <summary>Dimension p.</summary>
    public int Dim { get; }
    /// <summary>Degrees of freedom.</summary>
    public double Df { get; }

    /// <summary>log density at the positive-definite matrix X.</summary>
    public double LogPdf(NdArray x)
    {
        var p = Dim;
        var xa = x.AsContiguous().ToArray();
        double[] lx;
        try { lx = StatsMultivariate.CholeskyLower(xa, p, nameof(x)); }
        catch (ArgumentException) { return double.NegativeInfinity; }
        var logdetX = StatsMultivariate.LogDetFromCholesky(lx, p);
        var logdetS = StatsMultivariate.LogDetFromCholesky(_chol, p);
        var xinvS = StatsMultivariate.CholeskySolve(lx, p, _scale, p);
        var tr = 0.0;
        for (var i = 0; i < p; i++) tr += xinvS[i * p + i];
        return 0.5 * Df * logdetS - 0.5 * (Df + p + 1) * logdetX - 0.5 * tr - 0.5 * Df * p * Math.Log(2)
               - StatsMultivariate.MultiGammaLn(Df / 2, p);
    }

    /// <summary>Density at X.</summary>
    public double Pdf(NdArray x) => Math.Exp(LogPdf(x));

    /// <summary>Mean S/(df − p − 1), defined for df &gt; p + 1.</summary>
    public NdArray Mean()
    {
        var d = Df - Dim - 1;
        return new(_scale.Select(v => d > 0 ? v / d : double.NaN).ToArray(), Dim, Dim);
    }

    /// <summary><paramref name="size"/> draws, each the inverse of a Wishart(df, S⁻¹) draw.</summary>
    public NdArray[] Rvs(int size, GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        var p = Dim;
        var id = new double[p * p];
        for (var i = 0; i < p; i++) id[i * p + i] = 1;
        var r = new NdArray[size];
        for (var s = 0; s < size; s++)
        {
            var w = WishartDistribution.Bartlett(rng, Df, _cholInv, p);
            var lw = StatsMultivariate.CholeskyLower(w, p, "draw");
            r[s] = new NdArray(StatsMultivariate.CholeskySolve(lw, p, id, p), p, p);
        }
        return r;
    }

    /// <summary>Draws with a seed (null: fresh).</summary>
    public NdArray[] Rvs(int size, int? seed = null) => Rvs(size, StatsDistributions.MakeRng(seed));
}

/// <summary>
/// Haar-random orthogonal and rotation matrices, scipy's <c>ortho_group</c> and <c>special_ortho_group</c>:
/// QR of a Gaussian matrix with the signs of R's diagonal folded into Q (Mezzadri 2007) — without that
/// correction the result is not Haar distributed.
/// </summary>
public static class RandomMatrixGroups
{
    /// <summary>A Haar-distributed orthogonal matrix of order <paramref name="dim"/>.</summary>
    public static NdArray OrthoGroup(int dim, GraviRandom rng) => new(Haar(dim, rng, special: false), dim, dim);

    /// <summary>A Haar-distributed rotation (determinant +1) of order <paramref name="dim"/>.</summary>
    public static NdArray SpecialOrthoGroup(int dim, GraviRandom rng) => new(Haar(dim, rng, special: true), dim, dim);

    private static double[] Haar(int n, GraviRandom rng, bool special)
    {
        ArgumentNullException.ThrowIfNull(rng);
        if (n < 1) throw new ArgumentException("dim must be at least 1.", nameof(n));
        // Householder QR of a Gaussian matrix, accumulating Q explicitly.
        var a = new double[n * n];
        for (var i = 0; i < a.Length; i++) a[i] = StatsRandom.Normal(rng);
        var q = new double[n * n];
        for (var i = 0; i < n; i++) q[i * n + i] = 1;
        var v = new double[n];
        var sign = new double[n];
        for (var k = 0; k < n; k++)
        {
            var norm = 0.0;
            for (var i = k; i < n; i++) norm += a[i * n + k] * a[i * n + k];
            norm = Math.Sqrt(norm);
            var alpha = a[k * n + k] > 0 ? -norm : norm;
            // R's diagonal entry is alpha; Q's column k picks up the reflection.
            sign[k] = alpha < 0 ? -1 : 1;
            for (var i = 0; i < n; i++) v[i] = i < k ? 0 : a[i * n + k];
            v[k] -= alpha;
            var vn = 0.0;
            for (var i = k; i < n; i++) vn += v[i] * v[i];
            if (vn == 0) { sign[k] = 1; continue; }
            for (var j = k; j < n; j++)
            {
                var s = 0.0;
                for (var i = k; i < n; i++) s += v[i] * a[i * n + j];
                s *= 2 / vn;
                for (var i = k; i < n; i++) a[i * n + j] -= s * v[i];
            }
            for (var r = 0; r < n; r++)
            {
                var s = 0.0;
                for (var i = k; i < n; i++) s += q[r * n + i] * v[i];
                s *= 2 / vn;
                for (var i = k; i < n; i++) q[r * n + i] -= s * v[i];
            }
        }
        // Make R's diagonal positive: Q ← Q·diag(sign(R_kk)).
        for (var r = 0; r < n; r++)
            for (var k = 0; k < n; k++) q[r * n + k] *= sign[k];
        if (special && Det(q, n) < 0)
            for (var r = 0; r < n; r++) q[r * n] = -q[r * n];
        return q;
    }

    private static double Det(double[] m, int n)
    {
        var a = (double[])m.Clone();
        var det = 1.0;
        for (var c = 0; c < n; c++)
        {
            var p = c;
            for (var r = c + 1; r < n; r++) if (Math.Abs(a[r * n + c]) > Math.Abs(a[p * n + c])) p = r;
            if (a[p * n + c] == 0) return 0;
            if (p != c) { for (var k = 0; k < n; k++) (a[p * n + k], a[c * n + k]) = (a[c * n + k], a[p * n + k]); det = -det; }
            det *= a[c * n + c];
            for (var r = c + 1; r < n; r++)
            {
                var f = a[r * n + c] / a[c * n + c];
                for (var k = c; k < n; k++) a[r * n + k] -= f * a[c * n + k];
            }
        }
        return det;
    }
}
