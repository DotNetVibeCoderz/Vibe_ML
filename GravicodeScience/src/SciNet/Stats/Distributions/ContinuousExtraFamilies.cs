using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Tukey's lambda distribution, scipy's <c>tukeylambda</c>, defined by its quantile function
/// Q(q) = (q^λ − (1 − q)^λ)/λ (the logistic quantile log(q/(1 − q)) at λ = 0).
/// </summary>
/// <remarks>
/// There is no closed-form CDF: it is the root of Q(q) = z, found by bisection on log q in the lower
/// half (and by symmetry in the upper half), so tail probabilities keep their relative accuracy. The
/// density follows as 1/Q′(q). λ &gt; 0 gives the bounded support [−1/λ, 1/λ]. The quantile is written
/// with expm1 so that small |λ| does not cancel. It is the default family of <see cref="Stats.PpccMax"/>.
/// </remarks>
public sealed class TukeyLambdaDistribution : ContinuousDistribution
{
    /// <summary>Tukey lambda with shape <paramref name="lam"/>.</summary>
    public TukeyLambdaDistribution(double lam, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => Lambda = StatsDistributions.Finite(lam, nameof(lam));

    /// <summary>The shape λ.</summary>
    public double Lambda { get; }

    /// <inheritdoc />
    public override string Name => "tukeylambda";
    /// <inheritdoc />
    public override string[] ShapeNames => ["lam"];
    /// <inheritdoc />
    public override double[] Shapes => [Lambda];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new TukeyLambdaDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => Lambda > 0 ? -1 / Lambda : double.NegativeInfinity;
    /// <inheritdoc />
    protected override double Upper0 => Lambda > 0 ? 1 / Lambda : double.PositiveInfinity;

    /// <summary>Q(q) from log q and log(1 − q).</summary>
    private double Quantile(double logQ, double log1mQ)
        => Lambda == 0 ? logQ - log1mQ : (Special.Expm1(Lambda * logQ) - Special.Expm1(Lambda * log1mQ)) / Lambda;

    /// <inheritdoc />
    protected override double Ppf0(double q) => Quantile(Math.Log(q), Special.Log1p(-q));
    /// <inheritdoc />
    protected override double Isf0(double q) => -Ppf0(q);

    /// <summary>The lower-half probability q ≤ ½ with Q(q) = z ≤ 0, by bisection on log q.</summary>
    private double LowerQ(double z)
    {
        if (z >= 0) return 0.5;
        if (Lambda > 0 && z <= -1 / Lambda) return 0.0;
        // Bracket log q: Q is increasing in q.
        double hi = Math.Log(0.5), lo = -1.0;
        while (Quantile(lo, Special.Log1p(-Math.Exp(lo))) > z)
        {
            hi = lo;
            lo *= 2;
            if (lo < -1e5) return 0.0;
        }
        for (var i = 0; i < 200; i++)
        {
            var mid = 0.5 * (lo + hi);
            if (mid == lo || mid == hi) break;
            if (Quantile(mid, Special.Log1p(-Math.Exp(mid))) > z) hi = mid; else lo = mid;
        }
        return Math.Exp(0.5 * (lo + hi));
    }

    /// <inheritdoc />
    protected override double Cdf0(double z) => z <= 0 ? LowerQ(z) : 1 - LowerQ(-z);
    /// <inheritdoc />
    protected override double Sf0(double z) => Cdf0(-z);
    /// <inheritdoc />
    protected override double Pdf0(double z)
    {
        var q = z <= 0 ? LowerQ(z) : LowerQ(-z); // symmetric density
        if (q == 0) return 0.0;
        return 1.0 / (Math.Pow(q, Lambda - 1) + Math.Pow(1 - q, Lambda - 1));
    }
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(Pdf0(z));

    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var l = Lambda;
        if (l <= -0.5) return new(l > -1 ? 0 : double.NaN, double.PositiveInfinity, double.NaN, double.NaN);
        if (l == 0) return new(0, Math.PI * Math.PI / 3, 0, 1.2);
        // E[Q²] = (2/λ²)(1/(1+2λ) − B(λ+1, λ+1)); kurtosis from E[Q⁴] (Joiner & Rosenblatt 1971), λ > −1/4.
        var v = 2 / (l * l) * (1 / (1 + 2 * l) - Special.Beta(l + 1, l + 1));
        double k;
        if (l > -0.25)
        {
            var m4 = 2 / Math.Pow(l, 4) * (1 / (1 + 4 * l) - 4 * Special.Beta(3 * l + 1, l + 1) + 3 * Special.Beta(2 * l + 1, 2 * l + 1));
            k = m4 / (v * v) - 3;
        }
        else k = double.NaN;
        return new(0, v, 0, k);
    }

    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));
}

/// <summary>
/// The generalised gamma distribution, scipy's <c>gengamma</c>: pdf |c| x^(ca−1) e^(−x^c)/Γ(a) on x &gt; 0,
/// i.e. X = Y^(1/c) with Y ~ Gamma(a). A negative c gives the inverse family (c = −2 is the posterior
/// of a normal standard deviation, which is why <see cref="Stats.Mvsdist"/> needs it).
/// </summary>
public sealed class GenGammaDistribution : ContinuousDistribution
{
    /// <summary>Generalised gamma with shapes a &gt; 0 and c ≠ 0.</summary>
    public GenGammaDistribution(double a, double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        A = StatsDistributions.Positive(a, nameof(a));
        if (c == 0 || !double.IsFinite(c)) throw new ArgumentException("c must be finite and non-zero.", nameof(c));
        C = c;
    }

    /// <summary>The gamma shape a.</summary>
    public double A { get; }
    /// <summary>The power c.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "gengamma";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a", "c"];
    /// <inheritdoc />
    public override double[] Shapes => [A, C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new GenGammaDistribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
    {
        if (z == 0) return C * A < 1 ? double.PositiveInfinity : C * A == 1 && C > 0 ? Math.Log(Math.Abs(C)) - Special.GammaLn(A) : double.NegativeInfinity;
        return Math.Log(Math.Abs(C)) + Special.Xlogy(C * A - 1, z) - Math.Pow(z, C) - Special.GammaLn(A);
    }
    /// <inheritdoc />
    protected override double Cdf0(double z) { var t = Math.Pow(z, C); return C > 0 ? Special.GammaInc(A, t) : Special.GammaIncC(A, t); }
    /// <inheritdoc />
    protected override double Sf0(double z) { var t = Math.Pow(z, C); return C > 0 ? Special.GammaIncC(A, t) : Special.GammaInc(A, t); }
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Pow(C > 0 ? Special.GammaIncInv(A, q) : Special.GammaIncCInv(A, q), 1 / C);
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Pow(C > 0 ? Special.GammaIncCInv(A, q) : Special.GammaIncInv(A, q), 1 / C);
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        var s = A + n / C;
        return s > 0 ? Math.Exp(Special.GammaLn(s) - Special.GammaLn(A)) : double.PositiveInfinity;
    }
    /// <inheritdoc />
    protected override double Entropy0()
    {
        // a(1 − ψ(a)) + ψ(a)/c + log Γ(a) − log|c|, as scipy (whose large-a asymptotic branch agrees).
        var a = A;
        var psi = Special.Digamma(a);
        return a * (1 - psi) + psi / C + Special.GammaLn(a) - Math.Log(Math.Abs(C));
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Pow(StatsRandom.Gamma(rng, A), 1 / C);
}

public static partial class Stats
{
    /// <summary>Tukey lambda, scipy's <c>tukeylambda</c>.</summary>
    public static TukeyLambdaDistribution TukeyLambda(double lam, double loc = 0.0, double scale = 1.0) => new(lam, loc, scale);

    /// <summary>Generalised gamma, scipy's <c>gengamma</c>.</summary>
    public static GenGammaDistribution GenGamma(double a, double c, double loc = 0.0, double scale = 1.0) => new(a, c, loc, scale);
}
