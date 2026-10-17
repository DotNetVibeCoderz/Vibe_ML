using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.StatsDistMath;

namespace Gravicode.Science.SciNet;

/// <summary>
/// The whole mass function of a finite-support discrete law, tabulated once: prefix sums give the
/// CDF and suffix sums (accumulated from the far end) the survival function, so neither tail is a
/// difference of numbers near one.
/// </summary>
internal sealed class StatsPmfTable
{
    private readonly double[] _logPmf, _pmf, _cdf, _sf;

    /// <summary>Builds the table from unnormalised log masses for k = lo, lo + 1, ….</summary>
    public StatsPmfTable(int lo, double[] logW)
    {
        Lo = lo;
        var n = logW.Length;
        var m = double.NegativeInfinity;
        foreach (var v in logW) m = Math.Max(m, v);
        var s = 0.0;
        foreach (var v in logW) s += Math.Exp(v - m);
        var logZ = m + Math.Log(s);
        _logPmf = new double[n];
        _pmf = new double[n];
        for (var i = 0; i < n; i++) { _logPmf[i] = logW[i] - logZ; _pmf[i] = Math.Exp(_logPmf[i]); }
        _cdf = new double[n];
        _sf = new double[n];
        var acc = 0.0;
        for (var i = 0; i < n; i++) { acc += _pmf[i]; _cdf[i] = Math.Min(acc, 1.0); }
        acc = 0.0;
        for (var i = n - 1; i >= 0; i--) { _sf[i] = Math.Min(acc, 1.0); acc += _pmf[i]; }
    }

    /// <summary>Smallest support point.</summary>
    public int Lo { get; }
    /// <summary>Largest support point.</summary>
    public int Hi => Lo + _pmf.Length - 1;

    private int Index(double k) => (int)(k - Lo);
    public double Pmf(double k) => _pmf[Index(k)];
    public double LogPmf(double k) => _logPmf[Index(k)];
    public double Cdf(double k) => _cdf[Index(k)];
    public double Sf(double k) => _sf[Index(k)];

    /// <summary>Mean, variance, skewness, excess kurtosis by summing centred powers.</summary>
    public DistributionMoments Moments()
    {
        double mean = 0;
        for (var i = 0; i < _pmf.Length; i++) mean += (Lo + i) * _pmf[i];
        double m2 = 0, m3 = 0, m4 = 0;
        for (var i = 0; i < _pmf.Length; i++)
        {
            var d = Lo + i - mean;
            var d2 = d * d;
            m2 += d2 * _pmf[i];
            m3 += d2 * d * _pmf[i];
            m4 += d2 * d2 * _pmf[i];
        }
        return new(mean, m2, m3 / Math.Pow(m2, 1.5), m4 / (m2 * m2) - 3);
    }

    /// <summary>Shannon entropy.</summary>
    public double Entropy()
    {
        var h = 0.0;
        for (var i = 0; i < _pmf.Length; i++) if (_pmf[i] > 0) h -= _pmf[i] * _logPmf[i];
        return h;
    }

    /// <summary>Raw moment E[Kⁿ].</summary>
    public double Raw(int n)
    {
        var s = 0.0;
        for (var i = 0; i < _pmf.Length; i++) s += Math.Pow(Lo + i, n) * _pmf[i];
        return s;
    }

    /// <summary>A variate by inversion: binary search of a uniform in the CDF table.</summary>
    public double Sample(GraviRandom rng)
    {
        var u = StatsRandom.U(rng);
        int a = 0, b = _cdf.Length - 1;
        while (a < b)
        {
            var mid = (a + b) >> 1;
            if (_cdf[mid] >= u) b = mid; else a = mid + 1;
        }
        return Lo + a;
    }
}

/// <summary>The beta negative binomial distribution, scipy's <c>betanbinom</c>: NB(n, p) with p ~ Beta(a, b).</summary>
/// <remarks>
/// Its tail decays like k^(−a−1), so the generic tail sums would need millions of terms for small a;
/// the survival function sums 200 terms and closes the tail by Euler–Maclaurin. scipy's entropy
/// sum does not converge for heavy tails (1.90 instead of 3.00 at n = 10, a = 4.5, b = 2.5).
/// </remarks>
public sealed class BetaNegativeBinomialDistribution : DiscreteDistribution
{
    private readonly double _logP0;

    /// <summary>Beta negative binomial with <paramref name="n"/> ≥ 0 (integer) successes and Beta(<paramref name="a"/>, <paramref name="b"/>) success probability.</summary>
    public BetaNegativeBinomialDistribution(double n, double a, double b, double loc = 0.0) : base(loc)
    {
        if (!(n >= 0) || n != Math.Floor(n) || double.IsInfinity(n)) throw new ArgumentException("n must be a non-negative integer.", nameof(n));
        N = n;
        A = StatsDistributions.Positive(a, nameof(a));
        B = StatsDistributions.Positive(b, nameof(b));
        _logP0 = Special.BetaLn(a + n, b) - Special.BetaLn(a, b);
    }

    /// <summary>Number of successes n.</summary>
    public double N { get; }
    /// <summary>First beta shape a.</summary>
    public double A { get; }
    /// <summary>Second beta shape b.</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "betanbinom";
    /// <inheritdoc />
    public override string[] ShapeNames => ["n", "a", "b"];
    /// <inheritdoc />
    public override double[] Shapes => [N, A, B];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new BetaNegativeBinomialDistribution(shapes[0], shapes[1], shapes[2], loc);
    /// <inheritdoc />
    protected override double Upper0 => N == 0 ? 0.0 : double.PositiveInfinity;
    /// <inheritdoc />
    protected override double Pmf0(double k) => Math.Exp(LogPmf0(k));
    /// <inheritdoc />
    protected override double LogPmf0(double k)
    {
        if (N == 0) return k == 0 ? 0.0 : double.NegativeInfinity;
        return -Math.Log(N + k) - Special.BetaLn(N, k + 1) + Special.BetaLn(A + N, B + k) - Special.BetaLn(A, B);
    }
    /// <summary>pmf(k + 1)/pmf(k).</summary>
    private double Ratio(double k) => (N + k) * (B + k) / ((k + 1) * (A + N + B + k));
    /// <summary>The CDF: summed directly up to k = 2000 (cheap, and accurate wherever it is small), 1 − SF beyond.</summary>
    protected override double Cdf0(double k)
    {
        if (k >= 2000) return 1 - Sf0(k);
        var p = Math.Exp(_logP0);
        var s = p;
        for (var j = 0.0; j < k; j++) { p *= Ratio(j); s += p; }
        return Math.Min(s, 1.0);
    }
    /// <summary>
    /// The survival function: 200 terms summed directly, then the rest of the tail by Euler–Maclaurin
    /// on the mass as a smooth function of k (∫ + f/2 − f′/12 + f‴/720), which is accurate to about
    /// 1e-13 there because the derivatives of a power law fall like 1/k.
    /// </summary>
    protected override double Sf0(double k)
    {
        if (N == 0) return 0.0;
        var p = Pmf0(k + 1);
        var s = p;
        var j = k + 1;
        for (var i = 0; i < 200; i++, j++)
        {
            p *= Ratio(j);
            s += p;
            if (p <= 1e-17 * s) return Math.Min(s, 1.0);
        }
        // s now holds pmf(k + 1) … pmf(j); the tail starts at K = j + 1.
        var kk = j + 1;
        double LogF(double x) => Special.GammaLn(N + x) - Special.GammaLn(x + 1) + Special.GammaLn(B + x) - Special.GammaLn(A + N + B + x)
                                 + Special.GammaLn(A + N) - Special.GammaLn(N) - Special.GammaLn(B) + Special.GammaLn(A + B) - Special.GammaLn(A);
        var g1 = Special.Digamma(N + kk) - Special.Digamma(kk + 1) + Special.Digamma(B + kk) - Special.Digamma(A + N + B + kk);
        var g2 = Special.Polygamma(1, N + kk) - Special.Polygamma(1, kk + 1) + Special.Polygamma(1, B + kk) - Special.Polygamma(1, A + N + B + kk);
        var g3 = Special.Polygamma(2, N + kk) - Special.Polygamma(2, kk + 1) + Special.Polygamma(2, B + kk) - Special.Polygamma(2, A + N + B + kk);
        var fk = Math.Exp(LogF(kk));
        var integral = StatsDistMoreMath.QuadRel(x => Math.Exp(LogF(x)), kk, double.PositiveInfinity, null, 1e-13);
        var tail = integral + fk / 2 - fk * g1 / 12 + fk * (g3 + 3 * g1 * g2 + g1 * g1 * g1) / 720;
        return Math.Min(s + tail, 1.0);
    }
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        double n = N, a = A, b = B;
        var mu = a > 1 ? n * b / (a - 1) : double.PositiveInfinity;
        var v = a > 2 ? n * b * (n + a - 1) * (a + b - 1) / ((a - 2) * (a - 1) * (a - 1)) : double.PositiveInfinity;
        var g1 = a > 3 ? (2 * n + a - 1) * (2 * b + a - 1) / (a - 3) / Math.Sqrt(n * b * (n + a - 1) * (b + a - 1) / (a - 2)) : double.PositiveInfinity;
        double g2;
        if (a > 4)
        {
            var common = (a + 5) * b * b + (a + 5) * (a - 1) * b + 2 * (a - 1) * (a - 1);
            var term2 = (a - 1) * (a - 1) * (a * a + a * (6 * b - 1) + 6 * (b - 1) * b) + 3 * n * n * common + 3 * (a - 1) * n * common;
            g2 = (a - 2) * term2 / ((a - 4) * (a - 3) * b * n * (a + b - 1) * (a + n - 1)) - 3;
        }
        else g2 = double.PositiveInfinity;
        return new(mu, v, g1, g2);
    }
    /// <summary>Entropy by summing until the remaining mass is below 1e-16 (at most two million terms).</summary>
    protected override double Entropy0()
    {
        var p = Math.Exp(_logP0);
        double h = 0, mass = 0;
        for (var k = 0.0; k < 2_000_000; k++)
        {
            if (p > 0) h -= p * Math.Log(p);
            mass += p;
            if (1 - mass < 1e-16 && k > 10) break;
            p *= Ratio(k);
        }
        return h;
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        if (N == 0) return 0;
        var p = StatsRandom.Beta(rng, A, B);
        return StatsRandom.Poisson(rng, StatsRandom.Gamma(rng, N) * (1 - p) / p);
    }
}

/// <summary>Fisher's noncentral hypergeometric distribution, scipy's <c>nchypergeom_fisher</c>.</summary>
/// <remarks>
/// scipy's shapes: M balls in total, n of them red, N drawn, odds ratio ω. The mass ∝ C(n, k)C(M − n, N − k)ωᵏ
/// is tabulated exactly over the (finite) support, so CDF, survival function, moments and entropy are sums
/// (scipy's BiasedUrn works to 1e-12 relative).
/// </remarks>
public sealed class NcHypergeomFisherDistribution : DiscreteDistribution
{
    private readonly StatsPmfTable _t;

    /// <summary>Fisher's noncentral hypergeometric with scipy's <paramref name="M"/>, <paramref name="n"/>, <paramref name="N"/> and <paramref name="odds"/> &gt; 0.</summary>
    public NcHypergeomFisherDistribution(int M, int n, int N, double odds, double loc = 0.0) : base(loc)
    {
        StatsNcHypergeom.Check(M, n, N, odds);
        this.M = M; this.n = n; this.N = N; Odds = odds;
        var lo = Math.Max(0, N - (M - n));
        var hi = Math.Min(N, n);
        var w = new double[hi - lo + 1];
        var lw = Math.Log(odds);
        for (var k = lo; k <= hi; k++)
            w[k - lo] = StatsNcHypergeom.LogComb(n, k) + StatsNcHypergeom.LogComb(M - n, N - k) + k * lw;
        _t = new StatsPmfTable(lo, w);
    }

    /// <summary>Total number of balls.</summary>
    public int M { get; }
    /// <summary>Number of red balls.</summary>
    public int n { get; }
    /// <summary>Number drawn.</summary>
    public int N { get; }
    /// <summary>Odds ratio ω of red to white.</summary>
    public double Odds { get; }

    /// <inheritdoc />
    public override string Name => "nchypergeom_fisher";
    /// <inheritdoc />
    public override string[] ShapeNames => ["M", "n", "N", "odds"];
    /// <inheritdoc />
    public override double[] Shapes => [M, n, N, Odds];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc)
        => new NcHypergeomFisherDistribution(StatsNcHypergeom.Int(shapes[0]), StatsNcHypergeom.Int(shapes[1]), StatsNcHypergeom.Int(shapes[2]), shapes[3], loc);
    /// <inheritdoc />
    protected override double Lower0 => _t.Lo;
    /// <inheritdoc />
    protected override double Upper0 => _t.Hi;
    /// <inheritdoc />
    protected override double Pmf0(double k) => _t.Pmf(k);
    /// <inheritdoc />
    protected override double LogPmf0(double k) => _t.LogPmf(k);
    /// <inheritdoc />
    protected override double Cdf0(double k) => _t.Cdf(k);
    /// <inheritdoc />
    protected override double Sf0(double k) => _t.Sf(k);
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => _t.Moments();
    /// <inheritdoc />
    protected override double Munp0(int k) => _t.Raw(k);
    /// <inheritdoc />
    protected override double Entropy0() => _t.Entropy();
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => _t.Sample(rng);
}

/// <summary>Wallenius' noncentral hypergeometric distribution, scipy's <c>nchypergeom_wallenius</c>.</summary>
/// <remarks>
/// P(k) = C(n, k)C(M − n, N − k)∫₀¹(1 − t^(ω/D))ᵏ(1 − t^(1/D))^(N−k) dt with D = ω(n − k) + (M − n − N + k).
/// Substituting t = e^(−Dv) turns the integrand, which otherwise lives within e^(−D) of t = 0, into a
/// log-concave bump; it is integrated in log space around its maximum (Fog 2008 uses a similar
/// transformation). Masses are tabulated over the support and normalised.
/// </remarks>
public sealed class NcHypergeomWalleniusDistribution : DiscreteDistribution
{
    private readonly StatsPmfTable _t;

    /// <summary>Wallenius' noncentral hypergeometric with scipy's <paramref name="M"/>, <paramref name="n"/>, <paramref name="N"/> and <paramref name="odds"/> &gt; 0.</summary>
    public NcHypergeomWalleniusDistribution(int M, int n, int N, double odds, double loc = 0.0) : base(loc)
    {
        StatsNcHypergeom.Check(M, n, N, odds);
        this.M = M; this.n = n; this.N = N; Odds = odds;
        var lo = Math.Max(0, N - (M - n));
        var hi = Math.Min(N, n);
        var w = new double[hi - lo + 1];
        for (var k = lo; k <= hi; k++)
            w[k - lo] = StatsNcHypergeom.LogComb(n, k) + StatsNcHypergeom.LogComb(M - n, N - k) + LogIntegral(k, N - k, n - k, M - n - (N - k), odds);
        _t = new StatsPmfTable(lo, w);
    }

    /// <summary>log ∫₀¹(1 − t^(ω/D))ˣ(1 − t^(1/D))ʸ dt with D = ω·r1 + r2 (r1, r2 the balls left).</summary>
    private static double LogIntegral(int x, int y, int r1, int r2, double omega)
    {
        var d = omega * r1 + r2;
        if (d == 0) return 0.0; // everything drawn: a single support point
        // ∫ = D ∫₀^∞ (1 − e^(−ωv))ˣ (1 − e^(−v))ʸ e^(−Dv) dv.
        double L(double v)
        {
            if (v <= 0) return x + y > 0 ? double.NegativeInfinity : 0.0;
            var s = -d * v;
            if (x > 0) s += x * Math.Log(-Special.Expm1(-omega * v));
            if (y > 0) s += y * Math.Log(-Special.Expm1(-v));
            return s;
        }
        double dL(double v) => -d + (x > 0 ? x * omega / Special.Expm1(omega * v) : 0) + (y > 0 ? y / Special.Expm1(v) : 0);
        // The log integrand is concave: its maximum is the root of dL (or v = 0 when x = y = 0).
        double peak;
        if (x + y == 0) peak = 0.0;
        else
        {
            double a = 1e-300, b = 1.0;
            while (dL(b) > 0) b *= 2;
            for (var i = 0; i < 200; i++)
            {
                var m = 0.5 * (a + b);
                if (m <= a || m >= b) break;
                if (dL(m) > 0) a = m; else b = m;
            }
            peak = 0.5 * (a + b);
        }
        var lmax = L(peak);
        // Break points where the integrand has fallen by e^(−k) on either side of the peak.
        var pts = new List<double> { peak };
        foreach (var drop in new[] { 1.0, 5.0, 20.0, 50.0 })
        {
            if (peak > 0)
            {
                double a = 0, b = peak;
                for (var i = 0; i < 100; i++) { var m = 0.5 * (a + b); if (L(m) < lmax - drop) a = m; else b = m; }
                pts.Add(0.5 * (a + b));
            }
            double lo = peak, hi = Math.Max(2 * peak, 1 / d);
            while (L(hi) > lmax - drop) hi *= 2;
            for (var i = 0; i < 100; i++) { var m = 0.5 * (lo + hi); if (L(m) > lmax - drop) lo = m; else hi = m; }
            pts.Add(0.5 * (lo + hi));
        }
        double upper = pts.Max() * 4 + 1 / d * 60;
        var val = StatsDistMoreMath.QuadRel(v => { var l = L(v); return double.IsNegativeInfinity(l) ? 0.0 : Math.Exp(l - lmax); }, 0.0, upper, pts, 1e-13);
        return Math.Log(d) + lmax + Math.Log(val);
    }

    /// <summary>Total number of balls.</summary>
    public int M { get; }
    /// <summary>Number of red balls.</summary>
    public int n { get; }
    /// <summary>Number drawn.</summary>
    public int N { get; }
    /// <summary>Weight ratio ω of red to white.</summary>
    public double Odds { get; }

    /// <inheritdoc />
    public override string Name => "nchypergeom_wallenius";
    /// <inheritdoc />
    public override string[] ShapeNames => ["M", "n", "N", "odds"];
    /// <inheritdoc />
    public override double[] Shapes => [M, n, N, Odds];
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc)
        => new NcHypergeomWalleniusDistribution(StatsNcHypergeom.Int(shapes[0]), StatsNcHypergeom.Int(shapes[1]), StatsNcHypergeom.Int(shapes[2]), shapes[3], loc);
    /// <inheritdoc />
    protected override double Lower0 => _t.Lo;
    /// <inheritdoc />
    protected override double Upper0 => _t.Hi;
    /// <inheritdoc />
    protected override double Pmf0(double k) => _t.Pmf(k);
    /// <inheritdoc />
    protected override double LogPmf0(double k) => _t.LogPmf(k);
    /// <inheritdoc />
    protected override double Cdf0(double k) => _t.Cdf(k);
    /// <inheritdoc />
    protected override double Sf0(double k) => _t.Sf(k);
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => _t.Moments();
    /// <inheritdoc />
    protected override double Munp0(int k) => _t.Raw(k);
    /// <inheritdoc />
    protected override double Entropy0() => _t.Entropy();
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => _t.Sample(rng);
}

/// <summary>Shared validation for the noncentral hypergeometric families.</summary>
internal static class StatsNcHypergeom
{
    public static void Check(int M, int n, int N, double odds)
    {
        if (M < 0 || n < 0 || N < 0) throw new ArgumentException("M, n and N must be non-negative.");
        if (n > M || N > M) throw new ArgumentException("n and N cannot exceed M.");
        if (!(odds > 0) || double.IsInfinity(odds)) throw new ArgumentException("odds must be positive and finite.", nameof(odds));
    }

    public static int Int(double v)
    {
        if (v != Math.Floor(v) || v < 0 || v > int.MaxValue) throw new ArgumentException("Expected a non-negative integer.");
        return (int)v;
    }

    public static double LogComb(int n, int k) => Special.GammaLn(n + 1.0) - Special.GammaLn(k + 1.0) - Special.GammaLn(n - k + 1.0);
}

/// <summary>The Poisson binomial distribution, scipy's <c>poisson_binom</c>: the number of successes in independent trials with probabilities p.</summary>
/// <remarks>
/// The mass function is built by the O(n²) convolution recurrence, which only forms convex
/// combinations and is therefore stable (the DFT formula is faster but cancels in the tails).
/// Mean, variance, skewness and kurtosis are the closed cumulant sums.
/// </remarks>
public sealed class PoissonBinomialDistribution : DiscreteDistribution
{
    private readonly double[] _p;
    private readonly StatsPmfTable _t;

    /// <summary>Poisson binomial over success probabilities <paramref name="p"/>, each in [0, 1].</summary>
    public PoissonBinomialDistribution(double[] p, double loc = 0.0) : base(loc)
    {
        ArgumentNullException.ThrowIfNull(p);
        foreach (var v in p)
            if (!(v >= 0 && v <= 1)) throw new ArgumentException("Every probability must lie in [0, 1].", nameof(p));
        _p = (double[])p.Clone();
        var pmf = new double[p.Length + 1];
        pmf[0] = 1;
        for (var i = 0; i < p.Length; i++)
        {
            var pi = p[i];
            for (var k = i + 1; k >= 1; k--) pmf[k] = pmf[k] * (1 - pi) + pmf[k - 1] * pi;
            pmf[0] *= 1 - pi;
        }
        var lw = new double[pmf.Length];
        for (var k = 0; k < pmf.Length; k++) lw[k] = Math.Log(pmf[k]);
        _t = new StatsPmfTable(0, lw);
    }

    /// <summary>The success probabilities (a copy).</summary>
    public double[] P => (double[])_p.Clone();

    /// <inheritdoc />
    public override string Name => "poisson_binom";
    /// <inheritdoc />
    public override string[] ShapeNames => Enumerable.Range(0, _p.Length).Select(i => $"p{i}").ToArray();
    /// <inheritdoc />
    public override double[] Shapes => P;
    /// <inheritdoc />
    public override DiscreteDistribution WithParameters(double[] shapes, double loc) => new PoissonBinomialDistribution(shapes, loc);
    /// <inheritdoc />
    protected override double Upper0 => _p.Length;
    /// <inheritdoc />
    protected override double Pmf0(double k) => _t.Pmf(k);
    /// <inheritdoc />
    protected override double LogPmf0(double k) => _t.LogPmf(k);
    /// <inheritdoc />
    protected override double Cdf0(double k) => _t.Cdf(k);
    /// <inheritdoc />
    protected override double Sf0(double k) => _t.Sf(k);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        double m = 0, v = 0, k3 = 0, k4 = 0;
        foreach (var p in _p)
        {
            var q = p * (1 - p);
            m += p;
            v += q;
            k3 += q * (1 - 2 * p);
            k4 += q * (1 - 6 * q);
        }
        return new(m, v, k3 / Math.Pow(v, 1.5), k4 / (v * v));
    }
    /// <inheritdoc />
    protected override double Munp0(int k) => _t.Raw(k);
    /// <inheritdoc />
    protected override double Entropy0() => _t.Entropy();
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        var s = 0;
        foreach (var p in _p) if (StatsRandom.U(rng) < p) s++;
        return s;
    }
}
