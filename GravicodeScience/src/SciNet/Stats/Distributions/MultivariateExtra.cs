using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>The multivariate hypergeometric distribution, scipy's <c>multivariate_hypergeom</c>: n draws without replacement from an urn with m_i objects of type i.</summary>
public sealed class MultivariateHypergeomDistribution
{
    /// <summary>Urn with <paramref name="m"/>[i] objects of type i, <paramref name="n"/> draws (0 ≤ n ≤ Σm).</summary>
    public MultivariateHypergeomDistribution(int[] m, int n)
    {
        ArgumentNullException.ThrowIfNull(m);
        if (m.Length == 0 || m.Any(v => v < 0)) throw new ArgumentException("m must be non-negative counts.", nameof(m));
        Total = m.Sum();
        if (n < 0 || n > Total) throw new ArgumentException("n must lie in [0, Σm].", nameof(n));
        M = (int[])m.Clone();
        N = n;
    }

    /// <summary>Objects of each type.</summary>
    public int[] M { get; }
    /// <summary>Number of draws.</summary>
    public int N { get; }
    /// <summary>Total population Σm.</summary>
    public int Total { get; }

    /// <summary>log P(X = x) = Σ log C(m_i, x_i) − log C(M, n).</summary>
    public double LogPmf(ReadOnlySpan<double> x)
    {
        if (x.Length != M.Length) throw new ArgumentException("x has the wrong length.", nameof(x));
        var s = 0.0;
        var total = 0.0;
        for (var i = 0; i < x.Length; i++)
        {
            if (x[i] < 0 || x[i] > M[i] || x[i] != Math.Floor(x[i])) return double.NegativeInfinity;
            total += x[i];
            s += LogBinom(M[i], x[i]);
        }
        return total == N ? s - LogBinom(Total, N) : double.NegativeInfinity;
    }

    private static double LogBinom(double n, double k) => Special.GammaLn(n + 1) - Special.GammaLn(k + 1) - Special.GammaLn(n - k + 1);

    /// <summary>P(X = x).</summary>
    public double Pmf(ReadOnlySpan<double> x) => Math.Exp(LogPmf(x));

    /// <summary>Mean n·m_i/M.</summary>
    public double[] Mean() => M.Select(v => Total == 0 ? 0.0 : (double)N * v / Total).ToArray();

    /// <summary>Variances n (m_i/M)(1 − m_i/M)(M − n)/(M − 1).</summary>
    public double[] Var()
    {
        var c = Cov().AsContiguous().ToArray();
        return Enumerable.Range(0, M.Length).Select(i => c[i * M.Length + i]).ToArray();
    }

    /// <summary>Covariance n (M − n)/(M − 1) · (m_i/M)(δ_ij − m_j/M).</summary>
    public NdArray Cov()
    {
        var k = M.Length;
        var r = new double[k * k];
        if (Total > 1)
        {
            double f = N * (double)(Total - N) / (Total - 1);
            for (var i = 0; i < k; i++)
                for (var j = 0; j < k; j++)
                {
                    double pi = (double)M[i] / Total, pj = (double)M[j] / Total;
                    r[i * k + j] = f * pi * ((i == j ? 1 : 0) - pj);
                }
        }
        return new NdArray(r, k, k);
    }

    /// <summary><paramref name="size"/> draws as a (size, K) array, by sequential univariate hypergeometric draws.</summary>
    public NdArray Rvs(int size, GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        var k = M.Length;
        var r = new double[size * k];
        for (var s = 0; s < size; s++)
        {
            var left = N;
            var pop = Total;
            for (var i = 0; i < k - 1 && left > 0; i++)
            {
                var draw = (int)new HypergeometricDistribution(pop, M[i], left).Rvs(rng);
                r[s * k + i] = draw;
                left -= draw;
                pop -= M[i];
            }
            r[s * k + k - 1] += left;
        }
        return new NdArray(r, size, k);
    }

    /// <summary>Draws with a seed (null: fresh).</summary>
    public NdArray Rvs(int size, int? seed = null) => Rvs(size, StatsDistributions.MakeRng(seed));
}

/// <summary>The Dirichlet-multinomial distribution, scipy's <c>dirichlet_multinomial</c>: multinomial counts whose probabilities are Dirichlet(α).</summary>
public sealed class DirichletMultinomialDistribution
{
    /// <summary>Concentrations <paramref name="alpha"/> (positive) and <paramref name="n"/> trials.</summary>
    public DirichletMultinomialDistribution(double[] alpha, int n)
    {
        ArgumentNullException.ThrowIfNull(alpha);
        if (alpha.Length == 0 || alpha.Any(a => !(a > 0) || double.IsInfinity(a))) throw new ArgumentException("alpha must be positive and finite.", nameof(alpha));
        if (n < 0) throw new ArgumentException("n must be non-negative.", nameof(n));
        Alpha = (double[])alpha.Clone();
        N = n;
        A0 = Alpha.Sum();
    }

    /// <summary>Concentrations.</summary>
    public double[] Alpha { get; }
    /// <summary>Number of trials.</summary>
    public int N { get; }
    private double A0 { get; }

    /// <summary>log P(X = x) = log Γ(α₀) + log Γ(n+1) − log Γ(n+α₀) + Σ[log Γ(x_i+α_i) − log Γ(α_i) − log Γ(x_i+1)].</summary>
    public double LogPmf(ReadOnlySpan<double> x)
    {
        if (x.Length != Alpha.Length) throw new ArgumentException("x has the wrong length.", nameof(x));
        var s = Special.GammaLn(A0) + Special.GammaLn(N + 1) - Special.GammaLn(N + A0);
        var total = 0.0;
        for (var i = 0; i < x.Length; i++)
        {
            if (x[i] < 0 || x[i] != Math.Floor(x[i])) return double.NegativeInfinity;
            total += x[i];
            s += Special.GammaLn(x[i] + Alpha[i]) - Special.GammaLn(Alpha[i]) - Special.GammaLn(x[i] + 1);
        }
        return total == N ? s : double.NegativeInfinity;
    }

    /// <summary>P(X = x).</summary>
    public double Pmf(ReadOnlySpan<double> x) => Math.Exp(LogPmf(x));

    /// <summary>Mean n·α/α₀.</summary>
    public double[] Mean() => Alpha.Select(a => N * a / A0).ToArray();

    /// <summary>Variances n p_i(1 − p_i)(n + α₀)/(1 + α₀).</summary>
    public double[] Var() => Alpha.Select(a => { var p = a / A0; return N * p * (1 - p) * (N + A0) / (1 + A0); }).ToArray();

    /// <summary>Covariance n (n + α₀)/(1 + α₀) · p_i(δ_ij − p_j).</summary>
    public NdArray Cov()
    {
        var k = Alpha.Length;
        var r = new double[k * k];
        var f = N * (N + A0) / (1 + A0);
        for (var i = 0; i < k; i++)
            for (var j = 0; j < k; j++)
                r[i * k + j] = f * Alpha[i] / A0 * ((i == j ? 1 : 0) - Alpha[j] / A0);
        return new NdArray(r, k, k);
    }

    /// <summary><paramref name="size"/> draws: a Dirichlet probability vector, then multinomial counts.</summary>
    public NdArray Rvs(int size, GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        var k = Alpha.Length;
        var dir = new DirichletDistribution(k == 1 ? [Alpha[0], 1] : Alpha);
        var r = new double[size * k];
        for (var s = 0; s < size; s++)
        {
            if (k == 1) { r[s] = N; continue; }
            var p = dir.Rvs(1, rng).AsContiguous().ToArray();
            var left = (double)N;
            var rem = 1.0;
            for (var i = 0; i < k - 1 && left > 0; i++)
            {
                var pi = rem > 0 ? Math.Clamp(p[i] / rem, 0, 1) : 0;
                var draw = StatsRandom.Binomial(rng, left, pi, new BinomialDistribution(left, pi));
                r[s * k + i] = draw;
                left -= draw;
                rem -= p[i];
            }
            r[s * k + k - 1] += left;
        }
        return new NdArray(r, size, k);
    }

    /// <summary>Draws with a seed (null: fresh).</summary>
    public NdArray Rvs(int size, int? seed = null) => Rvs(size, StatsDistributions.MakeRng(seed));
}

/// <summary>
/// The von Mises–Fisher distribution on the unit sphere S^(p−1), scipy's <c>vonmises_fisher</c>:
/// f(x) = C_p(κ) exp(κ μᵀx) with C_p(κ) = κ^(p/2−1)/((2π)^(p/2) I_(p/2−1)(κ)).
/// </summary>
/// <remarks>The normaliser uses the exponentially scaled Bessel function, log I_v(κ) = log Ive(v, κ) + κ,
/// so large concentrations do not overflow. Sampling is Wood's (1994) rejection method for the
/// component along μ followed by a Householder reflection of e₁ onto μ.</remarks>
public sealed class VonMisesFisherDistribution
{
    private readonly double _logC;

    /// <summary>Mean direction <paramref name="mu"/> (unit vector, p ≥ 2) and concentration κ ≥ 0.</summary>
    public VonMisesFisherDistribution(double[] mu, double kappa = 1.0)
    {
        ArgumentNullException.ThrowIfNull(mu);
        if (mu.Length < 2) throw new ArgumentException("mu must have at least two components.", nameof(mu));
        var norm = Math.Sqrt(mu.Sum(v => v * v));
        if (Math.Abs(norm - 1) > 1e-5) throw new ArgumentException("mu must be a unit vector.", nameof(mu));
        if (!(kappa >= 0) || double.IsInfinity(kappa)) throw new ArgumentException("kappa must be non-negative and finite.", nameof(kappa));
        Mu = mu.Select(v => v / norm).ToArray();
        Kappa = kappa;
        _logC = LogNormFactor(Mu.Length, kappa);
    }

    /// <summary>Mean direction.</summary>
    public double[] Mu { get; }
    /// <summary>Concentration κ.</summary>
    public double Kappa { get; }
    /// <summary>Dimension p of the ambient space.</summary>
    public int Dim => Mu.Length;

    internal static double LogNormFactor(int p, double kappa)
    {
        var h = p / 2.0;
        if (kappa == 0) return Special.GammaLn(h) - Math.Log(2) - h * Math.Log(Math.PI);
        return (h - 1) * Math.Log(kappa) - h * Math.Log(2 * Math.PI) - Math.Log(Special.Ive(h - 1, kappa)) - kappa;
    }

    /// <summary>A_p(κ) = I_(p/2)(κ)/I_(p/2−1)(κ), the mean resultant length.</summary>
    internal static double MeanResultant(int p, double kappa)
        => kappa == 0 ? 0.0 : Special.Ive(p / 2.0, kappa) / Special.Ive(p / 2.0 - 1, kappa);

    /// <summary>log density at a unit vector (no check that ‖x‖ = 1, as scipy).</summary>
    public double LogPdf(ReadOnlySpan<double> x)
    {
        if (x.Length != Dim) throw new ArgumentException("x has the wrong dimension.", nameof(x));
        var dot = 0.0;
        for (var i = 0; i < Dim; i++) dot += Mu[i] * x[i];
        return _logC + Kappa * dot;
    }

    /// <summary>Density.</summary>
    public double Pdf(ReadOnlySpan<double> x) => Math.Exp(LogPdf(x));

    /// <summary>Log densities of the rows of an (n, p) array.</summary>
    public double[] LogPdf(NdArray x) => StatsMultivariate.Rows(x, Dim, r => LogPdf(r));

    /// <summary>Differential entropy −log C_p(κ) − κ A_p(κ).</summary>
    public double Entropy() => -_logC - Kappa * MeanResultant(Dim, Kappa);

    /// <summary><paramref name="size"/> draws as a (size, p) array.</summary>
    public NdArray Rvs(int size, GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        var p = Dim;
        var r = new double[size * p];
        double k = Kappa, pm1 = p - 1;
        var b = k == 0 ? 1.0 : pm1 / (2 * k + Math.Sqrt(4 * k * k + pm1 * pm1));
        var x0 = (1 - b) / (1 + b);
        var c = k * x0 + pm1 * Math.Log(1 - x0 * x0);
        // Householder vector reflecting e1 onto mu.
        var h = (double[])Mu.Clone();
        h[0] -= 1;
        var hn = h.Sum(v => v * v);
        var y = new double[p];
        for (var s = 0; s < size; s++)
        {
            double w;
            if (k == 0) w = SampleUniformW(rng, p);
            else
                while (true)
                {
                    var z = StatsRandom.Beta(rng, pm1 / 2, pm1 / 2);
                    w = (1 - (1 + b) * z) / (1 - (1 - b) * z);
                    var u = StatsRandom.U(rng);
                    if (k * w + pm1 * Math.Log(1 - x0 * w) - c >= Math.Log(u)) break;
                }
            // Uniform direction orthogonal to e1.
            var vn = 0.0;
            for (var i = 1; i < p; i++) { y[i] = StatsRandom.Normal(rng); vn += y[i] * y[i]; }
            vn = Math.Sqrt(vn);
            var sw = Math.Sqrt(Math.Max(0, 1 - w * w));
            y[0] = w;
            for (var i = 1; i < p; i++) y[i] = sw * y[i] / vn;
            // Reflect: x = y − 2 h (hᵀy)/‖h‖² (identity when mu = e1).
            var dot = 0.0;
            if (hn > 1e-300) for (var i = 0; i < p; i++) dot += h[i] * y[i];
            for (var i = 0; i < p; i++) r[s * p + i] = hn > 1e-300 ? y[i] - 2 * h[i] * dot / hn : y[i];
        }
        return new NdArray(r, size, p);
    }

    private static double SampleUniformW(GraviRandom rng, int p)
    {
        // The first coordinate of a uniform point on S^(p−1): 2·Beta((p−1)/2, (p−1)/2) − 1.
        return 2 * StatsRandom.Beta(rng, (p - 1) / 2.0, (p - 1) / 2.0) - 1;
    }

    /// <summary>Draws with a seed (null: fresh).</summary>
    public NdArray Rvs(int size, int? seed = null) => Rvs(size, StatsDistributions.MakeRng(seed));

    /// <summary>
    /// Maximum-likelihood fit to unit vectors (rows of <paramref name="x"/>), scipy's <c>vonmises_fisher.fit</c>:
    /// μ is the mean resultant direction and κ solves A_p(κ) = R̄ (bracketed root finding).
    /// </summary>
    public static (double[] Mu, double Kappa) Fit(NdArray x)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Shape.Length != 2 || x.Shape[1] < 2 || x.Shape[0] == 0) throw new ArgumentException("x must have shape (n, p) with p ≥ 2.", nameof(x));
        int n = x.Shape[0], p = x.Shape[1];
        var s = x.AsContiguous().ToArray();
        var m = new double[p];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < p; j++) m[j] += s[i * p + j];
        var norm = Math.Sqrt(m.Sum(v => v * v));
        if (norm == 0) throw new ArgumentException("The mean resultant is zero: the direction is undefined.", nameof(x));
        var mu = m.Select(v => v / norm).ToArray();
        var rbar = norm / n;
        if (rbar >= 1) return (mu, double.PositiveInfinity);
        double F(double k) => MeanResultant(p, k) - rbar;
        double lo = 1e-12, hi = 1.0;
        while (F(hi) < 0) { lo = hi; hi *= 2; if (hi > 1e12) return (mu, hi); }
        var kappa = StatsDistMath.Brent(F, lo, hi, F(lo), F(hi));
        return (mu, kappa);
    }
}

/// <summary>
/// The normal-inverse-gamma distribution of (x, σ²), scipy's <c>normal_inverse_gamma</c>: σ² ~ InvGamma(a, scale b)
/// and x | σ² ~ N(μ, σ²/λ) — the conjugate prior of a normal with unknown mean and variance.
/// </summary>
public sealed class NormalInverseGammaDistribution
{
    /// <summary>Parameters μ, λ &gt; 0, a &gt; 0, b &gt; 0.</summary>
    public NormalInverseGammaDistribution(double mu = 0, double lmbda = 1, double a = 1, double b = 1)
    {
        Mu = StatsDistributions.Finite(mu, nameof(mu));
        Lambda = StatsDistributions.Positive(lmbda, nameof(lmbda));
        A = StatsDistributions.Positive(a, nameof(a));
        B = StatsDistributions.Positive(b, nameof(b));
    }

    /// <summary>Location μ.</summary>
    public double Mu { get; }
    /// <summary>Precision multiplier λ.</summary>
    public double Lambda { get; }
    /// <summary>Shape a.</summary>
    public double A { get; }
    /// <summary>Scale b.</summary>
    public double B { get; }

    /// <summary>log f(x, σ²) = ½ log λ − ½ log 2πσ² + a log b − log Γ(a) − (a+1) log σ² − (2b + λ(x − μ)²)/(2σ²).</summary>
    public double LogPdf(double x, double s2)
    {
        if (!(s2 > 0)) return double.NegativeInfinity;
        var d = x - Mu;
        return 0.5 * Math.Log(Lambda) - 0.5 * Math.Log(2 * Math.PI * s2) + A * Math.Log(B) - Special.GammaLn(A)
               - (A + 1) * Math.Log(s2) - (2 * B + Lambda * d * d) / (2 * s2);
    }

    /// <summary>Density.</summary>
    public double Pdf(double x, double s2) => Math.Exp(LogPdf(x, s2));

    /// <summary>Means (μ, b/(a − 1)); the second is +∞ for a ≤ 1.</summary>
    public (double X, double S2) Mean() => (Mu, A > 1 ? B / (A - 1) : double.PositiveInfinity);

    /// <summary>Variances (b/((a − 1)λ), b²/((a − 1)²(a − 2))); +∞ where they do not exist.</summary>
    public (double X, double S2) Var() => (A > 1 ? B / ((A - 1) * Lambda) : double.PositiveInfinity,
        A > 2 ? B * B / ((A - 1) * (A - 1) * (A - 2)) : double.PositiveInfinity);

    /// <summary><paramref name="size"/> draws: σ² from the inverse gamma, then x from the conditional normal.</summary>
    public (double[] X, double[] S2) Rvs(int size, GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        var x = new double[size];
        var s2 = new double[size];
        for (var i = 0; i < size; i++)
        {
            s2[i] = B / StatsRandom.Gamma(rng, A);
            x[i] = Mu + Math.Sqrt(s2[i] / Lambda) * StatsRandom.Normal(rng);
        }
        return (x, s2);
    }

    /// <summary>Draws with a seed (null: fresh).</summary>
    public (double[] X, double[] S2) Rvs(int size, int? seed = null) => Rvs(size, StatsDistributions.MakeRng(seed));
}

public static partial class Stats
{
    /// <summary>Multivariate hypergeometric, scipy's <c>multivariate_hypergeom</c>.</summary>
    public static MultivariateHypergeomDistribution MultivariateHypergeom(int[] m, int n) => new(m, n);

    /// <summary>Dirichlet-multinomial, scipy's <c>dirichlet_multinomial</c>.</summary>
    public static DirichletMultinomialDistribution DirichletMultinomial(double[] alpha, int n) => new(alpha, n);

    /// <summary>von Mises–Fisher, scipy's <c>vonmises_fisher</c>.</summary>
    public static VonMisesFisherDistribution VonMisesFisher(double[] mu, double kappa = 1.0) => new(mu, kappa);

    /// <summary>Normal-inverse-gamma, scipy's <c>normal_inverse_gamma</c>.</summary>
    public static NormalInverseGammaDistribution NormalInverseGamma(double mu = 0, double lmbda = 1, double a = 1, double b = 1) => new(mu, lmbda, a, b);

    /// <summary>
    /// A Haar-random unitary matrix, scipy's <c>unitary_group.rvs</c>: the QR factor of a complex Ginibre
    /// matrix with R's diagonal made positive real (Mezzadri 2007). Gram–Schmidt with a second
    /// orthogonalisation pass gives exactly that factor, and keeps ‖QᴴQ − I‖ at rounding level.
    /// </summary>
    public static ComplexNdArray UnitaryGroup(int dim, int? seed = null) => UnitaryGroup(dim, StatsDistributions.MakeRng(seed));

    /// <summary><see cref="UnitaryGroup(int, int?)"/> drawing from <paramref name="rng"/>.</summary>
    public static ComplexNdArray UnitaryGroup(int dim, GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        if (dim < 1) throw new ArgumentException("dim must be at least 1.", nameof(dim));
        var n = dim;
        var q = new Complex[n * n]; // column-major work: q[col * n + row]
        var s = Math.Sqrt(0.5);
        for (var i = 0; i < q.Length; i++) q[i] = new Complex(s * StatsRandom.Normal(rng), s * StatsRandom.Normal(rng));
        for (var c = 0; c < n; c++)
        {
            for (var pass = 0; pass < 2; pass++)
                for (var k = 0; k < c; k++)
                {
                    var dot = Complex.Zero; // <q_k, q_c> = q_kᴴ q_c
                    for (var r = 0; r < n; r++) dot += Complex.Conjugate(q[k * n + r]) * q[c * n + r];
                    for (var r = 0; r < n; r++) q[c * n + r] -= dot * q[k * n + r];
                }
            var norm = 0.0;
            for (var r = 0; r < n; r++) norm += q[c * n + r].Real * q[c * n + r].Real + q[c * n + r].Imaginary * q[c * n + r].Imaginary;
            norm = Math.Sqrt(norm);
            for (var r = 0; r < n; r++) q[c * n + r] /= norm;
        }
        var outp = new Complex[n * n];
        for (var r = 0; r < n; r++)
            for (var c = 0; c < n; c++) outp[r * n + c] = q[c * n + r];
        return new ComplexNdArray(outp, n, n);
    }
}
