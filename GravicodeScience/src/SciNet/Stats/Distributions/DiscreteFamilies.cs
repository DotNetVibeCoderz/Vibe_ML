using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.StatsDistMath;

namespace Gravicode.Science.SciNet;

/// <summary>The Bernoulli distribution, scipy's <c>bernoulli</c>.</summary>
public sealed class BernoulliDistribution : DiscreteDistribution
{
    /// <summary>Bernoulli with success probability <paramref name="p"/>.</summary>
    public BernoulliDistribution(double p, double loc = 0.0) : base(loc) => P = StatsDistributions.InRange(p, 0, 1, nameof(p));

    /// <summary>Success probability.</summary>
    public double P { get; }

    /// <inheritdoc />
    public override string Name => "bernoulli";
    /// <inheritdoc />
    public override string[] ShapeNames => ["p"];
    /// <inheritdoc />
    public override double[] Shapes => [P];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new BernoulliDistribution(shapes[0], loc);
    /// <inheritdoc />
    protected override double Upper0 => 1;
    /// <inheritdoc />
    protected override double Pmf0(double k) => k == 0 ? 1 - P : P;
    /// <inheritdoc />
    protected override double LogPmf0(double k) => k == 0 ? Special.Log1p(-P) : Math.Log(P);
    /// <inheritdoc />
    protected override double Cdf0(double k) => 1 - P;
    /// <inheritdoc />
    protected override double Sf0(double k) => P;
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var q = 1 - P;
        var pq = P * q;
        return new(P, pq, (1 - 2 * P) / Math.Sqrt(pq), (1 - 6 * pq) / pq);
    }
    /// <inheritdoc />
    protected override double Munp0(int n) => P;
    /// <inheritdoc />
    protected override double Entropy0() => -Special.Xlogy(P, P) - Special.Xlog1py(1 - P, -P);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => rng.NextDouble() < P ? 1 : 0;
}

/// <summary>The binomial distribution, scipy's <c>binom</c>.</summary>
/// <remarks>The mass function uses Loader's saddle-point form, accurate to a few ulps for any n;
/// CDF and survival function are the regularised incomplete beta functions (Bdtr/Bdtrc).</remarks>
public sealed class BinomialDistribution : DiscreteDistribution
{
    /// <summary>Binomial with <paramref name="n"/> trials and success probability <paramref name="p"/>.</summary>
    public BinomialDistribution(double n, double p, double loc = 0.0) : base(loc)
    {
        if (!(n >= 0) || n != Math.Floor(n) || double.IsInfinity(n)) throw new ArgumentException("n must be a non-negative integer.", nameof(n));
        N = n;
        P = StatsDistributions.InRange(p, 0, 1, nameof(p));
    }

    /// <summary>Number of trials.</summary>
    public double N { get; }
    /// <summary>Success probability.</summary>
    public double P { get; }

    /// <inheritdoc />
    public override string Name => "binom";
    /// <inheritdoc />
    public override string[] ShapeNames => ["n", "p"];
    /// <inheritdoc />
    public override double[] Shapes => [N, P];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new BinomialDistribution(shapes[0], shapes[1], loc);
    /// <inheritdoc />
    protected override double Upper0 => N;
    /// <inheritdoc />
    protected override double Pmf0(double k) => Math.Exp(LogPmf0(k));
    /// <inheritdoc />
    protected override double LogPmf0(double k) => StatsLoader.LogDbinom(k, N, P, 1 - P);
    /// <inheritdoc />
    protected override double Cdf0(double k) => Special.Bdtr(k, N, P);
    /// <inheritdoc />
    protected override double Sf0(double k) => Special.Bdtrc(k, N, P);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var q = 1 - P;
        var v = N * P * q;
        return new(N * P, v, (q - P) / Math.Sqrt(v), (1 - 6 * P * q) / v);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => StatsRandom.Binomial(rng, N, P, this);
}

/// <summary>The Poisson distribution, scipy's <c>poisson</c>.</summary>
public sealed class PoissonDistribution : DiscreteDistribution
{
    /// <summary>Poisson with mean <paramref name="mu"/>.</summary>
    public PoissonDistribution(double mu, double loc = 0.0) : base(loc)
    {
        if (!(mu >= 0) || double.IsInfinity(mu)) throw new ArgumentException("mu must be non-negative and finite.", nameof(mu));
        Mu = mu;
    }

    /// <summary>The mean.</summary>
    public double Mu { get; }

    /// <inheritdoc />
    public override string Name => "poisson";
    /// <inheritdoc />
    public override string[] ShapeNames => ["mu"];
    /// <inheritdoc />
    public override double[] Shapes => [Mu];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new PoissonDistribution(shapes[0], loc);
    /// <inheritdoc />
    protected override double Pmf0(double k) => Math.Exp(LogPmf0(k));
    /// <inheritdoc />
    protected override double LogPmf0(double k) => StatsLoader.LogDpois(k, Mu);
    /// <inheritdoc />
    protected override double Cdf0(double k) => Special.GammaIncC(k + 1, Mu);
    /// <inheritdoc />
    protected override double Sf0(double k) => Special.GammaInc(k + 1, Mu);
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(Mu, Mu, 1 / Math.Sqrt(Mu), 1 / Mu);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => StatsRandom.Poisson(rng, Mu);

    /// <summary>The MLE of the mean is the sample mean.</summary>
    public static PoissonDistribution Fit(double[] data)
    {
        StatsDistributions.CheckData(data);
        return new PoissonDistribution(data.Average());
    }
}

/// <summary>The geometric distribution on {1, 2, …} (trials to first success), scipy's <c>geom</c>.</summary>
public sealed class GeometricDistribution : DiscreteDistribution
{
    /// <summary>Geometric with success probability <paramref name="p"/>.</summary>
    public GeometricDistribution(double p, double loc = 0.0) : base(loc)
    {
        if (!(p > 0 && p <= 1)) throw new ArgumentException("p must lie in (0, 1].", nameof(p));
        P = p;
    }

    /// <summary>Success probability.</summary>
    public double P { get; }

    /// <inheritdoc />
    public override string Name => "geom";
    /// <inheritdoc />
    public override string[] ShapeNames => ["p"];
    /// <inheritdoc />
    public override double[] Shapes => [P];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new GeometricDistribution(shapes[0], loc);
    /// <inheritdoc />
    protected override double Lower0 => 1;
    /// <inheritdoc />
    protected override double Pmf0(double k) => Math.Exp(LogPmf0(k));
    /// <inheritdoc />
    protected override double LogPmf0(double k) => Special.Xlog1py(k - 1, -P) + Math.Log(P);
    /// <inheritdoc />
    protected override double Cdf0(double k) => -Special.Expm1(k * Special.Log1p(-P));
    /// <inheritdoc />
    protected override double Sf0(double k) => Math.Exp(k * Special.Log1p(-P));
    /// <inheritdoc />
    protected override double LogSf0(double k) => k * Special.Log1p(-P);
    /// <inheritdoc />
    protected override double LogCdf0(double k) => Log1mExp(k * Special.Log1p(-P));
    /// <inheritdoc />
    protected override double PpfGuess0(double q) => Math.Max(1, Math.Ceiling(Special.Log1p(-q) / Special.Log1p(-P)));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var p = P;
        return new(1 / p, (1 - p) / (p * p), (2 - p) / Math.Sqrt(1 - p), 6 + p * p / (1 - p));
    }
    /// <inheritdoc />
    protected override double Entropy0() => -Math.Log(P) - Special.Log1p(-P) * (1 - P) / P;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
        => P == 1 ? 1 : Math.Ceiling(Math.Log(StatsRandom.U(rng)) / Special.Log1p(-P));
}

/// <summary>The negative binomial distribution (failures before the n-th success), scipy's <c>nbinom</c>.</summary>
public sealed class NegativeBinomialDistribution : DiscreteDistribution
{
    /// <summary>Negative binomial with <paramref name="n"/> successes (real n &gt; 0 allowed) and probability <paramref name="p"/>.</summary>
    public NegativeBinomialDistribution(double n, double p, double loc = 0.0) : base(loc)
    {
        N = StatsDistributions.Positive(n, nameof(n));
        if (!(p > 0 && p <= 1)) throw new ArgumentException("p must lie in (0, 1].", nameof(p));
        P = p;
    }

    /// <summary>Number of successes.</summary>
    public double N { get; }
    /// <summary>Success probability.</summary>
    public double P { get; }

    /// <inheritdoc />
    public override string Name => "nbinom";
    /// <inheritdoc />
    public override string[] ShapeNames => ["n", "p"];
    /// <inheritdoc />
    public override double[] Shapes => [N, P];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new NegativeBinomialDistribution(shapes[0], shapes[1], loc);
    /// <inheritdoc />
    protected override double Pmf0(double k) => Math.Exp(LogPmf0(k));
    /// <inheritdoc />
    protected override double LogPmf0(double k)
    {
        // R's dnbinom: n/(n+k) · dbinom(n; n+k, p), exact for real n via the saddle-point form.
        if (k == 0) return N * Math.Log(P);
        return Math.Log(N / (N + k)) + StatsLoader.LogDbinom(N, N + k, P, 1 - P);
    }
    /// <inheritdoc />
    protected override double Cdf0(double k) => Special.BetaInc(N, k + 1, P);
    /// <inheritdoc />
    protected override double Sf0(double k) => Special.BetaIncC(N, k + 1, P);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        double n = N, p = P, q = 1 - p;
        return new(n * q / p, n * q / (p * p), (2 - p) / Math.Sqrt(n * q), 6 / n + p * p / (n * q));
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => StatsRandom.Poisson(rng, StatsRandom.Gamma(rng, N) * (1 - P) / P);
}

/// <summary>The hypergeometric distribution, scipy's <c>hypergeom</c>: successes in N draws from M objects of which n are successes.</summary>
public sealed class HypergeometricDistribution : DiscreteDistribution
{
    /// <summary>Hypergeometric with population <paramref name="M"/>, <paramref name="n"/> successes and <paramref name="N"/> draws (scipy's names).</summary>
    public HypergeometricDistribution(int M, int n, int N, double loc = 0.0) : base(loc)
    {
        if (M < 0 || n < 0 || N < 0 || n > M || N > M) throw new ArgumentException("Require 0 ≤ n ≤ M and 0 ≤ N ≤ M.");
        TotalM = M;
        SuccessN = n;
        DrawsN = N;
    }

    /// <summary>Population size M.</summary>
    public int TotalM { get; }
    /// <summary>Number of success states n.</summary>
    public int SuccessN { get; }
    /// <summary>Number of draws N.</summary>
    public int DrawsN { get; }

    /// <inheritdoc />
    public override string Name => "hypergeom";
    /// <inheritdoc />
    public override string[] ShapeNames => ["M", "n", "N"];
    /// <inheritdoc />
    public override double[] Shapes => [TotalM, SuccessN, DrawsN];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new HypergeometricDistribution((int)shapes[0], (int)shapes[1], (int)shapes[2], loc);
    /// <inheritdoc />
    protected override double Lower0 => Math.Max(0, DrawsN - (TotalM - SuccessN));
    /// <inheritdoc />
    protected override double Upper0 => Math.Min(SuccessN, DrawsN);
    /// <inheritdoc />
    protected override double Pmf0(double k) => Math.Exp(LogPmf0(k));
    /// <inheritdoc />
    protected override double LogPmf0(double k)
    {
        // R's dhyper: three saddle-point binomials with p = N/M.
        double M = TotalM, n = SuccessN, N = DrawsN;
        var p = N / M;
        var q = (M - N) / M;
        return StatsLoader.LogDbinom(k, n, p, q) + StatsLoader.LogDbinom(N - k, M - n, p, q) - StatsLoader.LogDbinom(N, M, p, q);
    }
    /// <summary>Sums the mass from the nearer end so the smaller tail is computed directly.</summary>
    private double SumRange(double from, double to)
    {
        var s = 0.0;
        for (var j = from; j <= to; j++) s += Pmf0(j);
        return s;
    }
    private double Mode => Math.Floor((DrawsN + 1.0) * (SuccessN + 1.0) / (TotalM + 2.0));
    /// <inheritdoc />
    protected override double Cdf0(double k) => k <= Mode ? Math.Min(1, SumRange(Lower0, k)) : Math.Max(0, 1 - SumRange(k + 1, Upper0));
    /// <inheritdoc />
    protected override double Sf0(double k) => k >= Mode ? Math.Min(1, SumRange(k + 1, Upper0)) : Math.Max(0, 1 - SumRange(Lower0, k));
    /// <inheritdoc />
    protected override double LogCdf0(double k) => k <= Mode ? Math.Log(Cdf0(k)) : Special.Log1p(-SumRange(k + 1, Upper0));
    /// <inheritdoc />
    protected override double LogSf0(double k) => k >= Mode ? Math.Log(Sf0(k)) : Special.Log1p(-SumRange(Lower0, k));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        double M = TotalM, n = SuccessN, N = DrawsN;
        var m = M - n;
        var p = n / M;
        var mu = N * p;
        var v = m * n * N * (M - N) / (M * M * (M - 1));
        var g1 = (m - n) * (M - 2 * N) / (M - 2.0) * Math.Sqrt((M - 1.0) / (m * n * N * (M - N)));
        var g2 = M * (M + 1) - 6.0 * N * (M - N) - 6.0 * n * m;
        g2 *= (M - 1) * M * M;
        g2 += 6.0 * n * N * (M - N) * m * (5.0 * M - 6);
        g2 /= n * N * (M - N) * m * (M - 2.0) * (M - 3.0);
        return new(mu, v, g1, g2);
    }
}

/// <summary>The discrete uniform distribution on {low, …, high − 1}, scipy's <c>randint</c>.</summary>
public sealed class RandIntDistribution : DiscreteDistribution
{
    /// <summary>Uniform on the integers low ≤ k &lt; high.</summary>
    public RandIntDistribution(double low, double high, double loc = 0.0) : base(loc)
    {
        if (low != Math.Floor(low) || high != Math.Floor(high) || !(high > low)) throw new ArgumentException("Require integers low < high.");
        Low = low;
        High = high;
    }

    /// <summary>Smallest value.</summary>
    public double Low { get; }
    /// <summary>One past the largest value.</summary>
    public double High { get; }

    /// <inheritdoc />
    public override string Name => "randint";
    /// <inheritdoc />
    public override string[] ShapeNames => ["low", "high"];
    /// <inheritdoc />
    public override double[] Shapes => [Low, High];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new RandIntDistribution(shapes[0], shapes[1], loc);
    /// <inheritdoc />
    protected override double Lower0 => Low;
    /// <inheritdoc />
    protected override double Upper0 => High - 1;
    private double Count => High - Low;
    /// <inheritdoc />
    protected override double Pmf0(double k) => 1 / Count;
    /// <inheritdoc />
    protected override double Cdf0(double k) => (k - Low + 1) / Count;
    /// <inheritdoc />
    protected override double Sf0(double k) => (High - 1 - k) / Count;
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Max(Low, Math.Ceiling(q * Count + Low - 1));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var d = Count;
        var d2 = d * d;
        return new((Low + High - 1) / 2, (d2 - 1) / 12, 0, d2 == 1 ? double.NaN : -6.0 / 5.0 * (d2 + 1) / (d2 - 1));
    }
    /// <inheritdoc />
    protected override double Entropy0() => Math.Log(Count);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Low + Math.Floor(rng.NextDouble() * Count);
}

/// <summary>The Zipf (zeta) distribution on {1, 2, …}, scipy's <c>zipf</c>.</summary>
public sealed class ZipfDistribution : DiscreteDistribution
{
    private readonly double _zeta;

    /// <summary>Zipf with exponent <paramref name="a"/> &gt; 1.</summary>
    public ZipfDistribution(double a, double loc = 0.0) : base(loc)
    {
        if (!(a > 1) || double.IsInfinity(a)) throw new ArgumentException("a must exceed 1.", nameof(a));
        A = a;
        _zeta = Special.Zeta(a, 1.0);
    }

    /// <summary>The exponent.</summary>
    public double A { get; }

    /// <inheritdoc />
    public override string Name => "zipf";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a"];
    /// <inheritdoc />
    public override double[] Shapes => [A];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new ZipfDistribution(shapes[0], loc);
    /// <inheritdoc />
    protected override double Lower0 => 1;
    /// <inheritdoc />
    protected override double Pmf0(double k) => Math.Exp(LogPmf0(k));
    /// <inheritdoc />
    protected override double LogPmf0(double k) => -A * Math.Log(k) - Math.Log(_zeta);
    /// <inheritdoc />
    protected override double Sf0(double k) => Special.Zeta(A, k + 1) / _zeta;
    /// <inheritdoc />
    protected override double Cdf0(double k)
    {
        if (k < 1000)
        {
            var s = 0.0;
            for (var j = 1.0; j <= k; j++) s += Math.Pow(j, -A);
            return Math.Min(1, s / _zeta);
        }
        return 1 - Sf0(k);
    }
    /// <inheritdoc />
    protected override double PpfGuess0(double q) => 1;
    /// <inheritdoc />
    protected override double Entropy0()
    {
        // H = log ζ(a) + a·Σ log k·k^{−a}/ζ(a). The tail is heavy, so the sum runs to N = 1000 and the
        // rest is Euler–Maclaurin: ∫_N^∞ f + f(N)/2 − f'(N)/12 with f(x) = log x·x^{−a}.
        const int n = 1000;
        var s = 0.0;
        for (var k = 2; k < n; k++) s += Math.Log(k) * Math.Pow(k, -A);
        double lnN = Math.Log(n), am1 = A - 1;
        s += Math.Pow(n, -am1) * (lnN / am1 + 1 / (am1 * am1));
        s += 0.5 * lnN * Math.Pow(n, -A);
        s -= Math.Pow(n, -A - 1) * (1 - A * lnN) / 12;
        return Math.Log(_zeta) + A * s / _zeta;
    }
    /// <inheritdoc />
    protected override double Munp0(int n) => A > n + 1 ? Special.Zeta(A - n, 1.0) / _zeta : double.PositiveInfinity;
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var m = base.Moments0();
        return new(A > 2 ? m.Mean : double.PositiveInfinity, A > 3 ? m.Variance : double.PositiveInfinity, A > 4 ? m.Skewness : double.NaN, A > 5 ? m.Kurtosis : double.NaN);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        // Devroye's rejection sampler (Non-Uniform Random Variate Generation, X.6.1).
        var am1 = A - 1;
        var b = Math.Pow(2, am1);
        while (true)
        {
            var u = 1 - StatsRandom.U(rng);
            var v = StatsRandom.U(rng);
            var x = Math.Floor(Math.Pow(u, -1 / am1));
            if (x > 1e18 || x < 1) continue;
            var t = Math.Pow(1 + 1 / x, am1);
            if (v * x * (t - 1) / (b - 1) <= t / b) return x;
        }
    }
}

/// <summary>The Zipfian (truncated Zipf) distribution on {1, …, n}, scipy's <c>zipfian</c>.</summary>
public sealed class ZipfianDistribution : DiscreteDistribution
{
    private readonly double[] _cum; // _cum[k] = Σ_{j≤k} j^{-a}
    private readonly double[] _tail; // _tail[k] = Σ_{j>k} j^{-a}

    /// <summary>Zipfian with exponent <paramref name="a"/> ≥ 0 on {1, …, <paramref name="n"/>}.</summary>
    public ZipfianDistribution(double a, int n, double loc = 0.0) : base(loc)
    {
        if (!(a >= 0) || double.IsInfinity(a)) throw new ArgumentException("a must be non-negative.", nameof(a));
        if (n < 1 || n > 50_000_000) throw new ArgumentException("n must lie in [1, 5e7].", nameof(n));
        A = a;
        N = n;
        _cum = new double[n + 1];
        _tail = new double[n + 1];
        for (var j = 1; j <= n; j++) _cum[j] = _cum[j - 1] + Math.Pow(j, -a);
        for (var j = n - 1; j >= 0; j--) _tail[j] = _tail[j + 1] + Math.Pow(j + 1, -a);
    }

    /// <summary>The exponent.</summary>
    public double A { get; }
    /// <summary>The largest value.</summary>
    public int N { get; }

    /// <inheritdoc />
    public override string Name => "zipfian";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a", "n"];
    /// <inheritdoc />
    public override double[] Shapes => [A, N];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new ZipfianDistribution(shapes[0], (int)shapes[1], loc);
    /// <inheritdoc />
    protected override double Lower0 => 1;
    /// <inheritdoc />
    protected override double Upper0 => N;
    private double H => _cum[N];
    /// <inheritdoc />
    protected override double Pmf0(double k) => Math.Pow(k, -A) / H;
    /// <inheritdoc />
    protected override double LogPmf0(double k) => -A * Math.Log(k) - Math.Log(H);
    /// <inheritdoc />
    protected override double Cdf0(double k) => _cum[(int)k] / H;
    /// <inheritdoc />
    protected override double Sf0(double k) => _tail[(int)k] / H;
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        var s = 0.0;
        for (var j = 1; j <= N; j++) s += Math.Pow(j, n - A);
        return s / H;
    }
}

/// <summary>The logarithmic series distribution on {1, 2, …}, scipy's <c>logser</c>.</summary>
public sealed class LogSeriesDistribution : DiscreteDistribution
{
    /// <summary>Logarithmic series with parameter 0 &lt; <paramref name="p"/> &lt; 1.</summary>
    public LogSeriesDistribution(double p, double loc = 0.0) : base(loc)
    {
        if (!(p > 0 && p < 1)) throw new ArgumentException("p must lie in (0, 1).", nameof(p));
        P = p;
    }

    /// <summary>The shape parameter.</summary>
    public double P { get; }

    /// <inheritdoc />
    public override string Name => "logser";
    /// <inheritdoc />
    public override string[] ShapeNames => ["p"];
    /// <inheritdoc />
    public override double[] Shapes => [P];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new LogSeriesDistribution(shapes[0], loc);
    /// <inheritdoc />
    protected override double Lower0 => 1;
    /// <inheritdoc />
    protected override double Pmf0(double k) => Math.Exp(LogPmf0(k));
    /// <inheritdoc />
    protected override double LogPmf0(double k) => k * Math.Log(P) - Math.Log(k) - Math.Log(-Special.Log1p(-P));
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        var p = P;
        var r = Special.Log1p(-p);
        return n switch
        {
            1 => p / (p - 1) / r,
            2 => -p / r / ((p - 1) * (p - 1)),
            3 => -p / r * (1 + p) / Math.Pow(1 - p, 3),
            4 => -p / r * (1 / ((p - 1) * (p - 1)) - 6 * p / Math.Pow(p - 1, 3) + 6 * p * p / Math.Pow(p - 1, 4)),
            _ => base.Munp0(n),
        };
    }
    /// <inheritdoc />
    protected override double PpfGuess0(double q) => 1;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        // Kemp's LK algorithm.
        var r = Special.Log1p(-P);
        var v = StatsRandom.U(rng);
        if (v >= P) return 1;
        var q = -Special.Expm1(r * StatsRandom.U(rng));
        if (v <= q * q) return Math.Floor(1 + Math.Log(v) / Math.Log(q));
        return v <= q ? 2 : 1;
    }
}

/// <summary>The beta-binomial distribution, scipy's <c>betabinom</c>.</summary>
public sealed class BetaBinomialDistribution : DiscreteDistribution
{
    /// <summary>Beta-binomial with <paramref name="n"/> trials and Beta(<paramref name="a"/>, <paramref name="b"/>) success probability.</summary>
    public BetaBinomialDistribution(double n, double a, double b, double loc = 0.0) : base(loc)
    {
        if (!(n >= 0) || n != Math.Floor(n)) throw new ArgumentException("n must be a non-negative integer.", nameof(n));
        N = n;
        A = StatsDistributions.Positive(a, nameof(a));
        B = StatsDistributions.Positive(b, nameof(b));
    }

    /// <summary>Number of trials.</summary>
    public double N { get; }
    /// <summary>First beta shape.</summary>
    public double A { get; }
    /// <summary>Second beta shape.</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "betabinom";
    /// <inheritdoc />
    public override string[] ShapeNames => ["n", "a", "b"];
    /// <inheritdoc />
    public override double[] Shapes => [N, A, B];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new BetaBinomialDistribution(shapes[0], shapes[1], shapes[2], loc);
    /// <inheritdoc />
    protected override double Upper0 => N;
    /// <inheritdoc />
    protected override double Pmf0(double k) => Math.Exp(LogPmf0(k));
    /// <inheritdoc />
    protected override double LogPmf0(double k)
        => -Math.Log(N + 1) - Special.BetaLn(N - k + 1, k + 1) + Special.BetaLn(k + A, N - k + B) - Special.BetaLn(A, B);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        double n = N, a = A, b = B;
        var ep = a / (a + b);
        var eq = 1 - ep;
        var mu = n * ep;
        var v = n * (a + b + n) * ep * eq / (a + b + 1);
        var m = base.Moments0();
        return new(mu, v, m.Skewness, m.Kurtosis);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => StatsRandom.Binomial(rng, N, StatsRandom.Beta(rng, A, B), new BinomialDistribution(N, 0.5));
}

/// <summary>The Skellam distribution (difference of two Poissons), scipy's <c>skellam</c>.</summary>
public sealed class SkellamDistribution : DiscreteDistribution
{
    /// <summary>Skellam with means <paramref name="mu1"/>, <paramref name="mu2"/>.</summary>
    public SkellamDistribution(double mu1, double mu2, double loc = 0.0) : base(loc)
    {
        Mu1 = StatsDistributions.Positive(mu1, nameof(mu1));
        Mu2 = StatsDistributions.Positive(mu2, nameof(mu2));
    }

    /// <summary>Mean of the first Poisson.</summary>
    public double Mu1 { get; }
    /// <summary>Mean of the second Poisson.</summary>
    public double Mu2 { get; }

    /// <inheritdoc />
    public override string Name => "skellam";
    /// <inheritdoc />
    public override string[] ShapeNames => ["mu1", "mu2"];
    /// <inheritdoc />
    public override double[] Shapes => [Mu1, Mu2];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new SkellamDistribution(shapes[0], shapes[1], loc);
    /// <inheritdoc />
    protected override double Lower0 => double.NegativeInfinity;
    /// <inheritdoc />
    protected override double Pmf0(double k) => Math.Exp(LogPmf0(k));
    /// <inheritdoc />
    protected override double LogPmf0(double k)
    {
        var s1 = Math.Sqrt(Mu1);
        var s2 = Math.Sqrt(Mu2);
        return -(s1 - s2) * (s1 - s2) + k / 2 * Math.Log(Mu1 / Mu2) + Math.Log(Special.Ive(Math.Abs(k), 2 * s1 * s2));
    }
    // Via the noncentral chi-squared, as scipy: P(K ≤ k) = ncx2.cdf(2μ2; df −2k, nc 2μ1) for k < 0 and
    // 1 − ncx2.cdf(2μ1; df 2(k+1), nc 2μ2) for k ≥ 0; the complement is a separate Poisson mixture.
    /// <inheritdoc />
    protected override double Cdf0(double k)
        => k < 0 ? new NonCentralChi2Distribution(-2 * k, 2 * Mu1).Cdf(2 * Mu2) : new NonCentralChi2Distribution(2 * (k + 1), 2 * Mu2).Sf(2 * Mu1);
    /// <inheritdoc />
    protected override double Sf0(double k)
        => k < 0 ? new NonCentralChi2Distribution(-2 * k, 2 * Mu1).Sf(2 * Mu2) : new NonCentralChi2Distribution(2 * (k + 1), 2 * Mu2).Cdf(2 * Mu1);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var v = Mu1 + Mu2;
        return new(Mu1 - Mu2, v, (Mu1 - Mu2) / Math.Pow(v, 1.5), 1 / v);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => StatsRandom.Poisson(rng, Mu1) - StatsRandom.Poisson(rng, Mu2);
}

/// <summary>The discrete Laplace distribution on the integers, scipy's <c>dlaplace</c>.</summary>
public sealed class DLaplaceDistribution : DiscreteDistribution
{
    /// <summary>Discrete Laplace with decay <paramref name="a"/>.</summary>
    public DLaplaceDistribution(double a, double loc = 0.0) : base(loc) => A = StatsDistributions.Positive(a, nameof(a));

    /// <summary>The decay parameter.</summary>
    public double A { get; }

    /// <inheritdoc />
    public override string Name => "dlaplace";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a"];
    /// <inheritdoc />
    public override double[] Shapes => [A];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new DLaplaceDistribution(shapes[0], loc);
    /// <inheritdoc />
    protected override double Lower0 => double.NegativeInfinity;
    /// <inheritdoc />
    protected override double Pmf0(double k) => Math.Tanh(A / 2) * Math.Exp(-A * Math.Abs(k));
    /// <inheritdoc />
    protected override double LogPmf0(double k) => Math.Log(Math.Tanh(A / 2)) - A * Math.Abs(k);
    /// <summary>P(K ≥ m) for m ≥ 1: e^{−a m}/(1 + e^{−a}).</summary>
    private double UpperFrom(double m) => Math.Exp(-A * m) / (1 + Math.Exp(-A));
    /// <inheritdoc />
    protected override double Cdf0(double k) => k >= 0 ? 1 - UpperFrom(k + 1) : UpperFrom(-k);
    /// <inheritdoc />
    protected override double Sf0(double k) => k >= 0 ? UpperFrom(k + 1) : 1 - UpperFrom(-k);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var ea = Math.Exp(A);
        var em1 = Special.Expm1(A);
        var mu2 = 2 * ea / (em1 * em1);
        var mu4 = 2 * ea * (ea * ea + 10 * ea + 1) / Math.Pow(em1, 4);
        return new(0, mu2, 0, mu4 / (mu2 * mu2) - 3);
    }
    /// <inheritdoc />
    protected override double Entropy0() => A / Math.Sinh(A) - Math.Log(Math.Tanh(A / 2));
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        // The difference of two geometrics on {0, 1, …} with P(G ≥ m) = e^{−a m}.
        double G() => Math.Floor(Math.Log(StatsRandom.U(rng)) / -A);
        return G() - G();
    }
}

/// <summary>The truncated discrete exponential (Boltzmann) distribution on {0, …, N−1}, scipy's <c>boltzmann</c>.</summary>
public sealed class BoltzmannDistribution : DiscreteDistribution
{
    /// <summary>Boltzmann with rate <paramref name="lambda"/> and <paramref name="n"/> states.</summary>
    public BoltzmannDistribution(double lambda, int n, double loc = 0.0) : base(loc)
    {
        Lambda = StatsDistributions.Positive(lambda, nameof(lambda));
        if (n < 1) throw new ArgumentException("N must be at least 1.", nameof(n));
        N = n;
    }

    /// <summary>The rate.</summary>
    public double Lambda { get; }
    /// <summary>The number of states.</summary>
    public int N { get; }

    /// <inheritdoc />
    public override string Name => "boltzmann";
    /// <inheritdoc />
    public override string[] ShapeNames => ["lambda_", "N"];
    /// <inheritdoc />
    public override double[] Shapes => [Lambda, N];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new BoltzmannDistribution(shapes[0], (int)shapes[1], loc);
    /// <inheritdoc />
    protected override double Upper0 => N - 1;
    private double Norm => -Special.Expm1(-Lambda * N);
    /// <inheritdoc />
    protected override double Pmf0(double k) => -Special.Expm1(-Lambda) * Math.Exp(-Lambda * k) / Norm;
    /// <inheritdoc />
    protected override double Cdf0(double k) => -Special.Expm1(-Lambda * (k + 1)) / Norm;
    /// <inheritdoc />
    protected override double Sf0(double k) => Math.Exp(-Lambda * (k + 1)) * -Special.Expm1(-Lambda * (N - k - 1)) / Norm;
}

/// <summary>The Planck (discrete exponential) distribution on {0, 1, …}, scipy's <c>planck</c>.</summary>
public sealed class PlanckDistribution : DiscreteDistribution
{
    /// <summary>Planck with rate <paramref name="lambda"/>.</summary>
    public PlanckDistribution(double lambda, double loc = 0.0) : base(loc) => Lambda = StatsDistributions.Positive(lambda, nameof(lambda));

    /// <summary>The rate.</summary>
    public double Lambda { get; }

    /// <inheritdoc />
    public override string Name => "planck";
    /// <inheritdoc />
    public override string[] ShapeNames => ["lambda_"];
    /// <inheritdoc />
    public override double[] Shapes => [Lambda];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new PlanckDistribution(shapes[0], loc);
    /// <inheritdoc />
    protected override double Pmf0(double k) => -Special.Expm1(-Lambda) * Math.Exp(-Lambda * k);
    /// <inheritdoc />
    protected override double LogPmf0(double k) => Math.Log(-Special.Expm1(-Lambda)) - Lambda * k;
    /// <inheritdoc />
    protected override double Cdf0(double k) => -Special.Expm1(-Lambda * (k + 1));
    /// <inheritdoc />
    protected override double Sf0(double k) => Math.Exp(-Lambda * (k + 1));
    /// <inheritdoc />
    protected override double LogSf0(double k) => -Lambda * (k + 1);
    /// <inheritdoc />
    protected override double PpfGuess0(double q) => Math.Max(0, Math.Ceiling(-Special.Log1p(-q) / Lambda - 1));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var em1 = Special.Expm1(Lambda);
        return new(1 / em1, Math.Exp(Lambda) / (em1 * em1), 2 * Math.Cosh(Lambda / 2), 4 + 2 * Math.Cosh(Lambda));
    }
    /// <inheritdoc />
    protected override double Entropy0()
    {
        var c = -Special.Expm1(-Lambda);
        return Lambda * Math.Exp(-Lambda) / c - Math.Log(c);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Floor(StatsRandom.Exponential(rng) / Lambda);
}

/// <summary>The negative hypergeometric distribution, scipy's <c>nhypergeom</c>: successes drawn before the r-th failure.</summary>
public sealed class NegativeHypergeometricDistribution : DiscreteDistribution
{
    /// <summary>Negative hypergeometric with population <paramref name="M"/>, <paramref name="n"/> successes and stopping count <paramref name="r"/>.</summary>
    public NegativeHypergeometricDistribution(int M, int n, int r, double loc = 0.0) : base(loc)
    {
        if (M < 0 || n < 0 || r < 0 || n > M || r > M - n) throw new ArgumentException("Require 0 ≤ n ≤ M and 0 ≤ r ≤ M − n.");
        TotalM = M;
        SuccessN = n;
        R = r;
    }

    /// <summary>Population size.</summary>
    public int TotalM { get; }
    /// <summary>Number of successes in the population.</summary>
    public int SuccessN { get; }
    /// <summary>Number of failures at which sampling stops.</summary>
    public int R { get; }

    /// <inheritdoc />
    public override string Name => "nhypergeom";
    /// <inheritdoc />
    public override string[] ShapeNames => ["M", "n", "r"];
    /// <inheritdoc />
    public override double[] Shapes => [TotalM, SuccessN, R];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new NegativeHypergeometricDistribution((int)shapes[0], (int)shapes[1], (int)shapes[2], loc);
    /// <inheritdoc />
    protected override double Upper0 => SuccessN;
    private static double LnChoose(double a, double b) => Special.GammaLn(a + 1) - Special.GammaLn(b + 1) - Special.GammaLn(a - b + 1);
    /// <inheritdoc />
    protected override double Pmf0(double k) => Math.Exp(LogPmf0(k));
    /// <inheritdoc />
    protected override double LogPmf0(double k)
        => LnChoose(k + R - 1, k) + LnChoose(TotalM - R - k, SuccessN - k) - LnChoose(TotalM, SuccessN);
    /// <inheritdoc />
    protected override double Cdf0(double k)
    {
        var s = 0.0;
        for (var j = 0.0; j <= k; j++) s += Pmf0(j);
        return Math.Min(1, s);
    }
    /// <inheritdoc />
    protected override double Sf0(double k)
    {
        var s = 0.0;
        for (var j = k + 1; j <= SuccessN; j++) s += Pmf0(j);
        return Math.Min(1, s);
    }
}

/// <summary>The Yule–Simon distribution on {1, 2, …}, scipy's <c>yulesimon</c>.</summary>
public sealed class YuleSimonDistribution : DiscreteDistribution
{
    /// <summary>Yule–Simon with shape <paramref name="alpha"/>.</summary>
    public YuleSimonDistribution(double alpha, double loc = 0.0) : base(loc) => Alpha = StatsDistributions.Positive(alpha, nameof(alpha));

    /// <summary>The shape parameter.</summary>
    public double Alpha { get; }

    /// <inheritdoc />
    public override string Name => "yulesimon";
    /// <inheritdoc />
    public override string[] ShapeNames => ["alpha"];
    /// <inheritdoc />
    public override double[] Shapes => [Alpha];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new YuleSimonDistribution(shapes[0], loc);
    /// <inheritdoc />
    protected override double Lower0 => 1;
    /// <inheritdoc />
    protected override double Pmf0(double k) => Math.Exp(LogPmf0(k));
    /// <inheritdoc />
    protected override double LogPmf0(double k) => Math.Log(Alpha) + Special.BetaLn(k, Alpha + 1);
    /// <inheritdoc />
    protected override double LogSf0(double k) => Math.Log(k) + Special.BetaLn(k, Alpha + 1);
    /// <inheritdoc />
    protected override double Sf0(double k) => Math.Exp(LogSf0(k));
    /// <inheritdoc />
    protected override double Cdf0(double k) => -Special.Expm1(LogSf0(k));
    /// <inheritdoc />
    protected override double LogCdf0(double k) => Log1mExp(LogSf0(k));
    /// <inheritdoc />
    protected override double PpfGuess0(double q) => 1;
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var a = Alpha;
        // scipy's yulesimon._stats, including where each moment is infinite rather than undefined.
        var mu = a > 1 ? a / (a - 1) : double.PositiveInfinity;
        var v = a > 2 ? a * a / ((a - 1) * (a - 1) * (a - 2)) : a > 1 ? double.PositiveInfinity : double.NaN;
        var s = a > 3 ? Math.Sqrt(a - 2) * (a + 1) * (a + 1) / (a * (a - 3)) : a > 2 ? double.PositiveInfinity : double.NaN;
        var k = a > 4 ? a + 3 + (11 * a * a * a - 49 * a - 22) / (a * (a - 4) * (a - 3)) : a > 2 ? double.PositiveInfinity : double.NaN;
        return new(mu, v, s, k);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        // A geometric whose success probability is e^{−E/α} with E ~ Exp(1) (Devroye).
        var e1 = StatsRandom.Exponential(rng);
        var e2 = StatsRandom.Exponential(rng);
        var p = Math.Exp(-e2 / Alpha);
        return Math.Ceiling(-e1 / Special.Log1p(-p));
    }
}
