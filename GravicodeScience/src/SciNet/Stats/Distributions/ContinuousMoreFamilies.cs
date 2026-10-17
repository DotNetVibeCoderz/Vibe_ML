using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.StatsDistMath;

namespace Gravicode.Science.SciNet;

/// <summary>The alpha distribution, scipy's <c>alpha</c>: 1/X is a normal truncated to (−∞, a).</summary>
/// <remarks>
/// The survival function is P(a − 1/z &lt; Z &lt; a)/Φ(a) evaluated as a normal interval probability,
/// not 1 − CDF, so the upper tail keeps its digits. Every moment is infinite.
/// </remarks>
public sealed class AlphaDistribution : ContinuousDistribution
{
    private readonly double _logPhiA;

    /// <summary>Alpha with shape <paramref name="a"/> &gt; 0.</summary>
    public AlphaDistribution(double a, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        A = StatsDistributions.Positive(a, nameof(a));
        _logPhiA = Special.LogNdtr(A);
    }

    /// <summary>The shape a.</summary>
    public double A { get; }

    /// <inheritdoc />
    public override string Name => "alpha";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a"];
    /// <inheritdoc />
    public override double[] Shapes => [A];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new AlphaDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => z <= 0 ? 0.0 : Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => z <= 0 ? double.NegativeInfinity : -2 * Math.Log(z) + LogPhi(A - 1 / z) - _logPhiA;
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Exp(LogCdf0(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Special.LogNdtr(A - 1 / z) - _logPhiA;
    /// <inheritdoc />
    protected override double Sf0(double z) => StatsDistMoreMath.NormInterval(A - 1 / z, A) / Math.Exp(_logPhiA);
    /// <inheritdoc />
    protected override double Ppf0(double q) => q <= 0.5 ? 1 / (A - NdtriExp(Math.Log(q) + _logPhiA)) : base.Ppf0(q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(double.PositiveInfinity, double.PositiveInfinity, double.NaN, double.NaN);
    /// <inheritdoc />
    protected override double Munp0(int n) => double.PositiveInfinity;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => 1 / (A - NdtriExp(Math.Log(StatsRandom.U(rng)) + _logPhiA));
}

/// <summary>The beta prime (inverted beta) distribution, scipy's <c>betaprime</c>: X/(1 − X) for X ~ Beta(a, b).</summary>
/// <remarks>Each tail goes through whichever regularised incomplete beta keeps it small, as scipy does.</remarks>
public sealed class BetaPrimeDistribution : ContinuousDistribution
{
    /// <summary>Beta prime with shapes <paramref name="a"/>, <paramref name="b"/> &gt; 0.</summary>
    public BetaPrimeDistribution(double a, double b, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        A = StatsDistributions.Positive(a, nameof(a));
        B = StatsDistributions.Positive(b, nameof(b));
    }

    /// <summary>The first shape a.</summary>
    public double A { get; }
    /// <summary>The second shape b.</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "betaprime";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a", "b"];
    /// <inheritdoc />
    public override double[] Shapes => [A, B];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new BetaPrimeDistribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Special.Xlogy(A - 1, z) - Special.Xlog1py(A + B, z) - Special.BetaLn(A, B);
    /// <inheritdoc />
    protected override double Cdf0(double z) => z > 1 ? Special.BetaIncC(B, A, 1 / (1 + z)) : Special.BetaInc(A, B, z / (1 + z));
    /// <inheritdoc />
    protected override double Sf0(double z) => z > 1 ? Special.BetaInc(B, A, 1 / (1 + z)) : Special.BetaIncC(A, B, z / (1 + z));
    /// <inheritdoc />
    protected override double Ppf0(double q)
    {
        var r = Special.BetaIncInv(A, B, q);
        if (r <= 0.5) return r / (1 - r);
        return 1 / Special.BetaIncCInv(B, A, q) - 1; // 1/(1 + X) ~ Beta(b, a): its upper quantile, away from 1
    }
    /// <inheritdoc />
    protected override double Isf0(double q)
    {
        var s = Special.BetaIncInv(B, A, q);
        if (s <= 0.5) return 1 / s - 1;
        var r = Special.BetaIncCInv(A, B, q);
        return r / (1 - r);
    }
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        if (!(B > n)) return double.PositiveInfinity;
        var p = 1.0;
        for (var i = 1; i <= n; i++) p *= (A + i - 1) / (B - i);
        return p;
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => StatsRandom.Gamma(rng, A) / StatsRandom.Gamma(rng, B);
}

/// <summary>Formulas shared by <see cref="BurrDistribution"/> and <see cref="FiskDistribution"/>.</summary>
internal static class StatsBurrMath
{
    /// <summary>log(1 + z^(−c)), without overflowing z^(−c) for small z.</summary>
    public static double Log1pZmc(double z, double c)
        => z >= 1 ? Special.Log1p(Math.Pow(z, -c)) : -c * Math.Log(z) + Special.Log1p(Math.Pow(z, c));

    public static double LogPdf(double z, double c, double d)
        => z == 0 ? (c * d < 1 ? double.PositiveInfinity : c * d == 1 ? Math.Log(c * d) : double.NegativeInfinity)
                  : Math.Log(c) + Math.Log(d) - (c + 1) * Math.Log(z) - (d + 1) * Log1pZmc(z, c);

    public static double LogCdf(double z, double c, double d) => -d * Log1pZmc(z, c);

    public static double Ppf(double q, double c, double d) => Math.Pow(Special.Expm1(-Math.Log(q) / d), -1 / c);

    public static double Isf(double q, double c, double d) => Math.Pow(Special.Expm1(-Special.Log1p(-q) / d), -1 / c);

    public static double Munp(int n, double c, double d)
        => c > n ? d * Math.Exp(Special.BetaLn(1 - (double)n / c, d + (double)n / c)) : double.NaN;
}

/// <summary>The Burr type III distribution, scipy's <c>burr</c>: CDF (1 + x^(−c))^(−d).</summary>
public sealed class BurrDistribution : ContinuousDistribution
{
    /// <summary>Burr III with shapes <paramref name="c"/>, <paramref name="d"/> &gt; 0.</summary>
    public BurrDistribution(double c, double d, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        C = StatsDistributions.Positive(c, nameof(c));
        D = StatsDistributions.Positive(d, nameof(d));
    }

    /// <summary>The shape c.</summary>
    public double C { get; }
    /// <summary>The shape d.</summary>
    public double D { get; }

    /// <inheritdoc />
    public override string Name => "burr";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c", "d"];
    /// <inheritdoc />
    public override double[] Shapes => [C, D];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new BurrDistribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => StatsBurrMath.LogPdf(z, C, D);
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Exp(LogCdf0(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => StatsBurrMath.LogCdf(z, C, D);
    /// <inheritdoc />
    protected override double Sf0(double z) => -Special.Expm1(LogCdf0(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => Log1mExp(LogCdf0(z));
    /// <inheritdoc />
    protected override double Ppf0(double q) => StatsBurrMath.Ppf(q, C, D);
    /// <inheritdoc />
    protected override double Isf0(double q) => StatsBurrMath.Isf(q, C, D);
    /// <inheritdoc />
    protected override double Munp0(int n) => StatsBurrMath.Munp(n, C, D);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => StatsBurrMath.Ppf(StatsRandom.U(rng), C, D);
}

/// <summary>The Fisk (log-logistic) distribution, scipy's <c>fisk</c>: Burr III with d = 1.</summary>
public sealed class FiskDistribution : ContinuousDistribution
{
    /// <summary>Fisk with shape <paramref name="c"/> &gt; 0.</summary>
    public FiskDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Positive(c, nameof(c));

    /// <summary>The shape c.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "fisk";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new FiskDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => StatsBurrMath.LogPdf(z, C, 1.0);
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Exp(LogCdf0(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => StatsBurrMath.LogCdf(z, C, 1.0);
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(LogSf0(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => StatsBurrMath.LogCdf(1 / z, C, 1.0); // the log-logistic is symmetric in log z
    /// <inheritdoc />
    protected override double Ppf0(double q) => StatsBurrMath.Ppf(q, C, 1.0);
    /// <inheritdoc />
    protected override double Isf0(double q) => 1 / StatsBurrMath.Ppf(q, C, 1.0);
    /// <inheritdoc />
    protected override double Munp0(int n) => StatsBurrMath.Munp(n, C, 1.0);
    /// <inheritdoc />
    protected override double Entropy0() => 2 - Math.Log(C);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => StatsBurrMath.Ppf(StatsRandom.U(rng), C, 1.0);
}

/// <summary>The double gamma distribution, scipy's <c>dgamma</c>: a gamma variate with a random sign.</summary>
public sealed class DGammaDistribution : ContinuousDistribution
{
    /// <summary>Double gamma with shape <paramref name="a"/> &gt; 0.</summary>
    public DGammaDistribution(double a, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => A = StatsDistributions.Positive(a, nameof(a));

    /// <summary>The shape a.</summary>
    public double A { get; }

    /// <inheritdoc />
    public override string Name => "dgamma";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a"];
    /// <inheritdoc />
    public override double[] Shapes => [A];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new DGammaDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) { var a = Math.Abs(z); return Special.Xlogy(A - 1, a) - a - Ln2 - Special.GammaLn(A); }
    /// <inheritdoc />
    protected override double Cdf0(double z) => z > 0 ? 1 - 0.5 * Special.GammaIncC(A, z) : 0.5 * Special.GammaIncC(A, -z);
    /// <inheritdoc />
    protected override double Sf0(double z) => Cdf0(-z);
    /// <inheritdoc />
    protected override double LogCdf0(double z) => z > 0 ? Special.Log1p(-0.5 * Special.GammaIncC(A, z)) : Math.Log(0.5 * Special.GammaIncC(A, -z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => LogCdf0(-z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => q <= 0.5 ? -Special.GammaIncCInv(A, 2 * q) : Special.GammaIncCInv(A, 2 * (1 - q));
    /// <inheritdoc />
    protected override double Isf0(double q) => -Ppf0(q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var mu2 = A * (A + 1);
        return new(0.0, mu2, 0.0, (A + 2) * (A + 3) / mu2 - 3);
    }
    /// <inheritdoc />
    protected override double Munp0(int n) => n % 2 == 1 ? 0.0 : Math.Exp(Special.GammaLn(A + n) - Special.GammaLn(A));
    /// <inheritdoc />
    protected override double Entropy0() => A + Special.GammaLn(A) + (1 - A) * Special.Digamma(A) + Ln2;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        var g = StatsRandom.Gamma(rng, A);
        return (rng.NextUInt64() & 1) == 0 ? g : -g;
    }
}

/// <summary>The exponential power distribution, scipy's <c>exponpow</c>: survival exp(1 − exp(x^b)).</summary>
public sealed class ExponPowDistribution : ContinuousDistribution
{
    /// <summary>Exponential power with shape <paramref name="b"/> &gt; 0.</summary>
    public ExponPowDistribution(double b, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => B = StatsDistributions.Positive(b, nameof(b));

    /// <summary>The shape b.</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "exponpow";
    /// <inheritdoc />
    public override string[] ShapeNames => ["b"];
    /// <inheritdoc />
    public override double[] Shapes => [B];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new ExponPowDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) { var xb = Math.Pow(z, B); return 1 + Math.Log(B) + Special.Xlogy(B - 1, z) + xb - Math.Exp(xb); }
    /// <inheritdoc />
    protected override double LogSf0(double z) => -Special.Expm1(Math.Pow(z, B));
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(LogSf0(z));
    /// <inheritdoc />
    protected override double Cdf0(double z) => -Special.Expm1(LogSf0(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Log1mExp(LogSf0(z));
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Pow(Special.Log1p(-Special.Log1p(-q)), 1 / B);
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Pow(Special.Log1p(-Math.Log(q)), 1 / B);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Isf0(StatsRandom.U(rng));
}

/// <summary>The exponentiated Weibull distribution, scipy's <c>exponweib</c>: CDF (1 − exp(−x^c))^a.</summary>
public sealed class ExponWeibDistribution : ContinuousDistribution
{
    /// <summary>Exponentiated Weibull with shapes <paramref name="a"/>, <paramref name="c"/> &gt; 0.</summary>
    public ExponWeibDistribution(double a, double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        A = StatsDistributions.Positive(a, nameof(a));
        C = StatsDistributions.Positive(c, nameof(c));
    }

    /// <summary>The exponent a.</summary>
    public double A { get; }
    /// <summary>The Weibull shape c.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "exponweib";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a", "c"];
    /// <inheritdoc />
    public override double[] Shapes => [A, C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new ExponWeibDistribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
    {
        var negxc = -Math.Pow(z, C);
        return Math.Log(A) + Math.Log(C) + Special.Xlogy(A - 1, -Special.Expm1(negxc)) + negxc + Special.Xlogy(C - 1, z);
    }
    /// <inheritdoc />
    protected override double LogCdf0(double z) => A * Log1mExp(-Math.Pow(z, C));
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Exp(LogCdf0(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => -Special.Expm1(LogCdf0(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => Log1mExp(LogCdf0(z));
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Pow(-Log1mExp(Math.Log(q) / A), 1 / C);
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Pow(-Math.Log(-Special.Expm1(Special.Log1p(-q) / A)), 1 / C);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));
}

/// <summary>The fatigue-life (Birnbaum–Saunders) distribution, scipy's <c>fatiguelife</c>.</summary>
/// <remarks>The quantile 0.25(t + √(t² + 4))² is rewritten as 4/(√(t² + 4) − t)²/4 for t &lt; 0, where the sum cancels.</remarks>
public sealed class FatigueLifeDistribution : ContinuousDistribution
{
    /// <summary>Birnbaum–Saunders with shape <paramref name="c"/> &gt; 0.</summary>
    public FatigueLifeDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Positive(c, nameof(c));

    /// <summary>The shape c.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "fatiguelife";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new FatigueLifeDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    private double Arg(double z) => (Math.Sqrt(z) - 1 / Math.Sqrt(z)) / C;
    /// <inheritdoc />
    protected override double Pdf0(double z) => z <= 0 ? 0.0 : Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
        => z <= 0 ? double.NegativeInfinity : Special.Log1p(z) - (z - 1) * (z - 1) / (2 * z * C * C) - Math.Log(2 * C) - 0.5 * (Math.Log(2 * Math.PI) + 3 * Math.Log(z));
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.Ndtr(Arg(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.Ndtr(-Arg(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Special.LogNdtr(Arg(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => Special.LogNdtr(-Arg(z));
    private static double FromT(double t)
    {
        var r = Math.Sqrt(t * t + 4);
        var s = t >= 0 ? t + r : 4 / (r - t);
        return 0.25 * s * s;
    }
    /// <inheritdoc />
    protected override double Ppf0(double q) => FromT(C * Special.Ndtri(q));
    /// <inheritdoc />
    protected override double Isf0(double q) => FromT(-C * Special.Ndtri(q));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var c2 = C * C;
        var den = 5 * c2 + 4;
        return new(c2 / 2 + 1, c2 * den / 4, 4 * C * (11 * c2 + 6) / Math.Pow(den, 1.5), 6 * c2 * (93 * c2 + 40) / (den * den));
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => FromT(C * StatsRandom.Normal(rng));
}

/// <summary>The folded Cauchy distribution, scipy's <c>foldcauchy</c>: |Y| for Y ~ Cauchy(c, 1).</summary>
/// <remarks>atan(z − c) + atan(z + c) is evaluated as one atan2, which keeps the small-z CDF exact.</remarks>
public sealed class FoldCauchyDistribution : ContinuousDistribution
{
    /// <summary>Folded Cauchy with shape <paramref name="c"/> ≥ 0.</summary>
    public FoldCauchyDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.InRange(c, 0.0, double.MaxValue, nameof(c));

    /// <summary>The shape c.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "foldcauchy";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new FoldCauchyDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => (1 / (1 + (z - C) * (z - C)) + 1 / (1 + (z + C) * (z + C))) / Math.PI;
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Atan2(2 * z, 1 + C * C - z * z) / Math.PI;
    /// <inheritdoc />
    protected override double Sf0(double z) => (Math.Atan2(1, z - C) + Math.Atan2(1, z + C)) / Math.PI;
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(double.PositiveInfinity, double.PositiveInfinity, double.NaN, double.NaN);
    /// <inheritdoc />
    protected override double Munp0(int n) => double.PositiveInfinity;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Abs(C + Math.Tan(Math.PI * (StatsRandom.U(rng) - 0.5)));
}

/// <summary>The generalised exponential distribution, scipy's <c>genexpon</c>.</summary>
/// <remarks>
/// Survival exp(−(a + b)x + b(1 − e^(−cx))/c). The quantile has a Lambert-W closed form, used for the
/// upper tail; near zero its two terms cancel, so lower quantiles invert the CDF instead.
/// </remarks>
public sealed class GenExponDistribution : ContinuousDistribution
{
    /// <summary>Generalised exponential with shapes <paramref name="a"/>, <paramref name="b"/>, <paramref name="c"/> &gt; 0.</summary>
    public GenExponDistribution(double a, double b, double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        A = StatsDistributions.Positive(a, nameof(a));
        B = StatsDistributions.Positive(b, nameof(b));
        C = StatsDistributions.Positive(c, nameof(c));
    }

    /// <summary>The shape a.</summary>
    public double A { get; }
    /// <summary>The shape b.</summary>
    public double B { get; }
    /// <summary>The shape c.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "genexpon";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a", "b", "c"];
    /// <inheritdoc />
    public override double[] Shapes => [A, B, C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new GenExponDistribution(shapes[0], shapes[1], shapes[2], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double LogSf0(double z) => -(A + B) * z + B * -Special.Expm1(-C * z) / C;
    /// <inheritdoc />
    protected override double Pdf0(double z) => (A + B * -Special.Expm1(-C * z)) * Math.Exp(LogSf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(A + B * -Special.Expm1(-C * z)) + LogSf0(z);
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(LogSf0(z));
    /// <inheritdoc />
    protected override double Cdf0(double z) => -Special.Expm1(LogSf0(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Log1mExp(LogSf0(z));
    private double IsfClosed(double logQ)
    {
        var s = A + B;
        var t = (B - C * logQ) / s;
        return (t + Special.LambertW(-B / s * Math.Exp(-t))) / C;
    }
    /// <inheritdoc />
    protected override double Isf0(double q) => q <= 0.5 ? IsfClosed(Math.Log(q)) : base.Isf0(q);
    /// <inheritdoc />
    protected override double Ppf0(double q) => q >= 0.5 ? IsfClosed(Special.Log1p(-q)) : base.Ppf0(q);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => IsfClosed(Math.Log(StatsRandom.U(rng)));
}

/// <summary>The generalised half-logistic distribution, scipy's <c>genhalflogistic</c>, on [0, 1/c].</summary>
public sealed class GenHalfLogisticDistribution : ContinuousDistribution
{
    /// <summary>Generalised half-logistic with shape <paramref name="c"/> &gt; 0.</summary>
    public GenHalfLogisticDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Positive(c, nameof(c));

    /// <summary>The shape c.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "genhalflogistic";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new GenHalfLogisticDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Upper0 => 1 / C;
    /// <summary>log t with t = (1 − cz)^(1/c).</summary>
    private double LogT(double z) => Special.Log1p(-C * z) / C;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
    {
        var lt = LogT(z);
        return Ln2 + (1 / C - 1) * Special.Log1p(-C * z) - 2 * StatsDistMoreMath.Log1pExp(lt);
    }
    /// <inheritdoc />
    protected override double Cdf0(double z) { var lt = LogT(z); return -Special.Expm1(lt) / (1 + Math.Exp(lt)); }
    /// <inheritdoc />
    protected override double Sf0(double z) { var t = Math.Exp(LogT(z)); return 2 * t / (1 + t); }
    /// <inheritdoc />
    protected override double Ppf0(double q) => -Special.Expm1(C * (Special.Log1p(-q) - Special.Log1p(q))) / C;
    /// <inheritdoc />
    protected override double Isf0(double q) => -Special.Expm1(C * (Math.Log(q) - Math.Log(2 - q))) / C;
    /// <inheritdoc />
    protected override double Entropy0() => 2 - (2 * C + 1) * Ln2;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));
}

/// <summary>The generalised normal (exponential power) distribution, scipy's <c>gennorm</c>: density ∝ exp(−|x|^β).</summary>
public sealed class GenNormDistribution : ContinuousDistribution
{
    /// <summary>Generalised normal with shape <paramref name="beta"/> &gt; 0.</summary>
    public GenNormDistribution(double beta, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => Beta = StatsDistributions.Positive(beta, nameof(beta));

    /// <summary>The shape β (2 is a normal with variance ½, 1 a Laplace).</summary>
    public double Beta { get; }

    /// <inheritdoc />
    public override string Name => "gennorm";
    /// <inheritdoc />
    public override string[] ShapeNames => ["beta"];
    /// <inheritdoc />
    public override double[] Shapes => [Beta];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new GenNormDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(0.5 * Beta) - Special.GammaLn(1 / Beta) - Math.Pow(Math.Abs(z), Beta);
    /// <inheritdoc />
    protected override double Cdf0(double z)
    {
        var tail = 0.5 * Special.GammaIncC(1 / Beta, Math.Pow(Math.Abs(z), Beta));
        return z < 0 ? tail : 1 - tail;
    }
    /// <inheritdoc />
    protected override double Sf0(double z) => Cdf0(-z);
    /// <inheritdoc />
    protected override double Ppf0(double q)
        => q <= 0.5 ? -Math.Pow(Special.GammaIncCInv(1 / Beta, 2 * q), 1 / Beta) : Math.Pow(Special.GammaIncCInv(1 / Beta, 2 * (1 - q)), 1 / Beta);
    /// <inheritdoc />
    protected override double Isf0(double q) => -Ppf0(q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var c1 = Special.GammaLn(1 / Beta);
        var c3 = Special.GammaLn(3 / Beta);
        var c5 = Special.GammaLn(5 / Beta);
        return new(0.0, Math.Exp(c3 - c1), 0.0, Math.Exp(c5 + c1 - 2 * c3) - 3);
    }
    /// <inheritdoc />
    protected override double Munp0(int n) => n % 2 == 1 ? 0.0 : Math.Exp(Special.GammaLn((n + 1.0) / Beta) - Special.GammaLn(1 / Beta));
    /// <inheritdoc />
    protected override double Entropy0() => 1 / Beta - Math.Log(0.5 * Beta) + Special.GammaLn(1 / Beta);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        var y = Math.Pow(StatsRandom.Gamma(rng, 1 / Beta), 1 / Beta);
        return (rng.NextUInt64() & 1) == 0 ? y : -y;
    }
}

/// <summary>The half generalised normal distribution, scipy's <c>halfgennorm</c>: density ∝ exp(−x^β) on x ≥ 0.</summary>
public sealed class HalfGenNormDistribution : ContinuousDistribution
{
    /// <summary>Half generalised normal with shape <paramref name="beta"/> &gt; 0.</summary>
    public HalfGenNormDistribution(double beta, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => Beta = StatsDistributions.Positive(beta, nameof(beta));

    /// <summary>The shape β.</summary>
    public double Beta { get; }

    /// <inheritdoc />
    public override string Name => "halfgennorm";
    /// <inheritdoc />
    public override string[] ShapeNames => ["beta"];
    /// <inheritdoc />
    public override double[] Shapes => [Beta];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new HalfGenNormDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(Beta) - Special.GammaLn(1 / Beta) - Math.Pow(z, Beta);
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.GammaInc(1 / Beta, Math.Pow(z, Beta));
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.GammaIncC(1 / Beta, Math.Pow(z, Beta));
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Pow(Special.GammaIncInv(1 / Beta, q), 1 / Beta);
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Pow(Special.GammaIncCInv(1 / Beta, q), 1 / Beta);
    /// <inheritdoc />
    protected override double Munp0(int n) => Math.Exp(Special.GammaLn((n + 1.0) / Beta) - Special.GammaLn(1 / Beta));
    /// <inheritdoc />
    protected override double Entropy0() => 1 / Beta - Math.Log(Beta) + Special.GammaLn(1 / Beta);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Pow(StatsRandom.Gamma(rng, 1 / Beta), 1 / Beta);
}

/// <summary>Gibrat's distribution, scipy's <c>gibrat</c>: the log-normal with s = 1.</summary>
public sealed class GibratDistribution : ContinuousDistribution
{
    /// <summary>Gibrat's distribution.</summary>
    public GibratDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "gibrat";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new GibratDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => z <= 0 ? 0.0 : Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) { if (z <= 0) return double.NegativeInfinity; var l = Math.Log(z); return -l - 0.5 * l * l - LnSqrt2Pi; }
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.Ndtr(Math.Log(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.Ndtr(-Math.Log(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Special.LogNdtr(Math.Log(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => Special.LogNdtr(-Math.Log(z));
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Exp(Special.Ndtri(q));
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Exp(-Special.Ndtri(q));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var p = Math.E;
        return new(Math.Sqrt(p), p * (p - 1), Math.Sqrt(p - 1) * (2 + p), p * p * p * p + 2 * p * p * p + 3 * p * p - 6);
    }
    /// <inheritdoc />
    protected override double Munp0(int n) => Math.Exp(0.5 * n * n);
    /// <inheritdoc />
    protected override double Entropy0() => LnSqrt2Pi + 0.5;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Exp(StatsRandom.Normal(rng));
}

/// <summary>The Gompertz (truncated Gumbel) distribution, scipy's <c>gompertz</c>.</summary>
public sealed class GompertzDistribution : ContinuousDistribution
{
    /// <summary>Gompertz with shape <paramref name="c"/> &gt; 0.</summary>
    public GompertzDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Positive(c, nameof(c));

    /// <summary>The shape c.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "gompertz";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new GompertzDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(C) + z - C * Special.Expm1(z);
    /// <inheritdoc />
    protected override double LogSf0(double z) => -C * Special.Expm1(z);
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(LogSf0(z));
    /// <inheritdoc />
    protected override double Cdf0(double z) => -Special.Expm1(LogSf0(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Log1mExp(LogSf0(z));
    /// <inheritdoc />
    protected override double Ppf0(double q) => Special.Log1p(-Special.Log1p(-q) / C);
    /// <inheritdoc />
    protected override double Isf0(double q) => Special.Log1p(-Math.Log(q) / C);
    /// <inheritdoc />
    protected override double Entropy0() => 1 - Math.Log(C) - StatsDistMoreMath.ScaledExp1(C);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Special.Log1p(StatsRandom.Exponential(rng) / C);
}

/// <summary>The half-logistic distribution, scipy's <c>halflogistic</c>.</summary>
public sealed class HalfLogisticDistribution : ContinuousDistribution
{
    /// <summary>Half-logistic.</summary>
    public HalfLogisticDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "halflogistic";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new HalfLogisticDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Ln2 - z - 2 * Special.Log1p(Math.Exp(-z));
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Tanh(z / 2);
    /// <inheritdoc />
    protected override double Sf0(double z) => 2 / (1 + Math.Exp(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => Ln2 - StatsDistMoreMath.Log1pExp(z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => 2 * Math.Atanh(q);
    /// <inheritdoc />
    protected override double Isf0(double q) => q < 0.5 ? Math.Log(2 / q - 1) : 2 * Math.Atanh(1 - q);
    /// <inheritdoc />
    protected override double Munp0(int n) => n switch
    {
        1 => 2 * Ln2,
        2 => Math.PI * Math.PI / 3,
        3 => 9 * 1.2020569031595942853997,
        4 => 7 * Math.Pow(Math.PI, 4) / 15,
        _ => 2 * (1 - Math.Pow(2, 1 - n)) * Special.Gamma(n + 1) * Special.Zeta(n),
    };
    /// <inheritdoc />
    protected override double Entropy0() => 2 - Ln2;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));
}

/// <summary>The hyperbolic secant distribution, scipy's <c>hypsecant</c>.</summary>
public sealed class HypSecantDistribution : ContinuousDistribution
{
    /// <summary>Hyperbolic secant.</summary>
    public HypSecantDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "hypsecant";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new HypSecantDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) { var a = Math.Abs(z); return -Math.Log(Math.PI) - a - Special.Log1p(Math.Exp(-2 * a)) + Ln2; }
    /// <inheritdoc />
    protected override double Cdf0(double z) => 2 / Math.PI * Math.Atan(Math.Exp(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => 2 / Math.PI * Math.Atan(Math.Exp(-z));
    /// <inheritdoc />
    protected override double Ppf0(double q) => q <= 0.5 ? Math.Log(Math.Tan(Math.PI * q / 2)) : -Math.Log(Math.Tan(Math.PI * (1 - q) / 2));
    /// <inheritdoc />
    protected override double Isf0(double q) => -Ppf0(q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(0.0, Math.PI * Math.PI / 4, 0.0, 2.0);
    /// <inheritdoc />
    protected override double Entropy0() => Math.Log(2 * Math.PI);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));
}

/// <summary>The inverted Weibull (Fréchet) distribution, scipy's <c>invweibull</c>.</summary>
public sealed class InvWeibullDistribution : ContinuousDistribution
{
    /// <summary>Inverted Weibull with shape <paramref name="c"/> &gt; 0.</summary>
    public InvWeibullDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Positive(c, nameof(c));

    /// <summary>The shape c.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "invweibull";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new InvWeibullDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => z <= 0 ? 0.0 : Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => z <= 0 ? double.NegativeInfinity : Math.Log(C) - (C + 1) * Math.Log(z) - Math.Pow(z, -C);
    /// <inheritdoc />
    protected override double LogCdf0(double z) => -Math.Pow(z, -C);
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Exp(-Math.Pow(z, -C));
    /// <inheritdoc />
    protected override double Sf0(double z) => -Special.Expm1(-Math.Pow(z, -C));
    /// <inheritdoc />
    protected override double LogSf0(double z) => Log1mExp(-Math.Pow(z, -C));
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Pow(-Math.Log(q), -1 / C);
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Pow(-Special.Log1p(-q), -1 / C);
    /// <summary>Moments that exist are closed-form; the rest are reported as infinite (mean, variance) or undefined.</summary>
    protected override DistributionMoments Moments0()
    {
        if (C > 4) return base.Moments0();
        var m1 = Munp0(1);
        var m2 = Munp0(2);
        var r = StatsDistMoreMath.FromRaw(m1, m2, Munp0(3), Munp0(4));
        return new(m1, C > 2 ? m2 - m1 * m1 : double.PositiveInfinity, C > 3 ? r.Skewness : double.NaN, double.NaN);
    }
    /// <summary>Γ(1 − n/c) for n &lt; c; infinite otherwise (scipy evaluates Γ at a negative argument there and reports a finite, wrong value).</summary>
    protected override double Munp0(int n) => C > n ? Special.Gamma(1 - n / C) : double.PositiveInfinity;
    /// <inheritdoc />
    protected override double Entropy0() => 1 + EulerGamma + EulerGamma / C - Math.Log(C);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Pow(StatsRandom.Exponential(rng), -1 / C);
}

/// <summary>The asymmetric Laplace distribution, scipy's <c>laplace_asymmetric</c>.</summary>
public sealed class LaplaceAsymmetricDistribution : ContinuousDistribution
{
    /// <summary>Asymmetric Laplace with shape <paramref name="kappa"/> &gt; 0 (1 is the Laplace).</summary>
    public LaplaceAsymmetricDistribution(double kappa, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => Kappa = StatsDistributions.Positive(kappa, nameof(kappa));

    /// <summary>The asymmetry κ.</summary>
    public double Kappa { get; }

    /// <inheritdoc />
    public override string Name => "laplace_asymmetric";
    /// <inheritdoc />
    public override string[] ShapeNames => ["kappa"];
    /// <inheritdoc />
    public override double[] Shapes => [Kappa];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new LaplaceAsymmetricDistribution(shapes[0], loc, scale);
    private double K2 => Kappa * Kappa;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => (z >= 0 ? -Kappa * z : z / Kappa) - Math.Log(Kappa + 1 / Kappa);
    /// <inheritdoc />
    protected override double LogCdf0(double z) => z >= 0 ? Special.Log1p(-Math.Exp(-Kappa * z) / (1 + K2)) : z / Kappa + Math.Log(K2 / (1 + K2));
    /// <inheritdoc />
    protected override double LogSf0(double z) => z >= 0 ? -Kappa * z - Special.Log1p(K2) : Special.Log1p(-Math.Exp(z / Kappa) * K2 / (1 + K2));
    /// <inheritdoc />
    protected override double Cdf0(double z) => z >= 0 ? 1 - Math.Exp(-Kappa * z) / (1 + K2) : Math.Exp(z / Kappa) * K2 / (1 + K2);
    /// <inheritdoc />
    protected override double Sf0(double z) => z >= 0 ? Math.Exp(-Kappa * z) / (1 + K2) : 1 - Math.Exp(z / Kappa) * K2 / (1 + K2);
    /// <inheritdoc />
    protected override double Ppf0(double q)
        => q < K2 / (1 + K2) ? Kappa * Math.Log(q * (1 + K2) / K2) : -Math.Log((1 - q) * (1 + K2)) / Kappa;
    /// <inheritdoc />
    protected override double Isf0(double q)
        => q <= 1 / (1 + K2) ? -Math.Log(q * (1 + K2)) / Kappa : Kappa * Math.Log((1 - q) * (1 + K2) / K2);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var k = Kappa;
        var k4 = k * k * k * k;
        return new(1 / k - k, 1 / (k * k) + k * k, 2 * (1 - Math.Pow(k, 6)) / Math.Pow(1 + k4, 1.5), 6 * (1 + k4 * k4) / ((1 + k4) * (1 + k4)));
    }
    /// <inheritdoc />
    protected override double Entropy0() => 1 + Math.Log(Kappa + 1 / Kappa);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));
}

/// <summary>The left-skewed Lévy distribution, scipy's <c>levy_l</c>: the reflection of <see cref="LevyDistribution"/>.</summary>
public sealed class LevyLDistribution : ContinuousDistribution
{
    /// <summary>Left Lévy.</summary>
    public LevyLDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "levy_l";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new LevyLDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Upper0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => z >= 0 ? 0.0 : Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) { var a = -z; return z >= 0 ? double.NegativeInfinity : -LnSqrt2Pi - 1.5 * Math.Log(a) - 1 / (2 * a); }
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.Erf(1 / Math.Sqrt(-2 * z));
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.Erfc(1 / Math.Sqrt(-2 * z));
    /// <inheritdoc />
    protected override double Ppf0(double q) { var e = Special.ErfInv(q); return -1 / (2 * e * e); }
    /// <inheritdoc />
    protected override double Isf0(double q) { var e = Special.ErfcInv(q); return -1 / (2 * e * e); }
    /// <summary>The mean is −∞ (scipy reports +∞); the higher moments do not exist.</summary>
    protected override DistributionMoments Moments0() => new(double.NegativeInfinity, double.PositiveInfinity, double.NaN, double.NaN);
    /// <inheritdoc />
    protected override double Munp0(int n) => n % 2 == 1 ? double.NegativeInfinity : double.PositiveInfinity;
    /// <inheritdoc />
    protected override double Entropy0() => new LevyDistribution().Entropy();
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) { var n = StatsRandom.Normal(rng); return -1 / (n * n); }
}

/// <summary>The log-gamma distribution, scipy's <c>loggamma</c>: log of a Gamma(c) variate.</summary>
public sealed class LogGammaDistribution : ContinuousDistribution
{
    private const double LogXMin = -708.3964185322641;

    /// <summary>Log-gamma with shape <paramref name="c"/> &gt; 0.</summary>
    public LogGammaDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Positive(c, nameof(c));

    /// <summary>The shape c.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "loggamma";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new LogGammaDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => C * z - Math.Exp(z) - Special.GammaLn(C);
    /// <inheritdoc />
    protected override double Cdf0(double z) => z < LogXMin ? Math.Exp(C * z - Special.GammaLn(C + 1)) : Special.GammaInc(C, Math.Exp(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => z < LogXMin ? -Special.Expm1(C * z - Special.GammaLn(C + 1)) : Special.GammaIncC(C, Math.Exp(z));
    /// <inheritdoc />
    protected override double Ppf0(double q)
    {
        var g = Special.GammaIncInv(C, q);
        return g < 2.2250738585072014e-308 ? (Math.Log(q) + Special.GammaLn(C + 1)) / C : Math.Log(g);
    }
    /// <inheritdoc />
    protected override double Isf0(double q)
    {
        var g = Special.GammaIncCInv(C, q);
        return g < 2.2250738585072014e-308 ? (Special.Log1p(-q) + Special.GammaLn(C + 1)) / C : Math.Log(g);
    }
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var v = Special.Polygamma(1, C);
        return new(Special.Digamma(C), v, Special.Polygamma(2, C) / Math.Pow(v, 1.5), Special.Polygamma(3, C) / (v * v));
    }
    /// <inheritdoc />
    protected override double Entropy0()
    {
        if (C < 45) return Special.GammaLn(C) - C * Special.Digamma(C) + C;
        // Asymptotic series; the direct form cancels three large terms here.
        return LnSqrt2Pi + 0.5 - 0.5 * Math.Log(C) + 1 / (6 * C) - Math.Pow(C, -3) / 90 + Math.Pow(C, -5) / 210;
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Log(StatsRandom.Gamma(rng, C + 1)) + Math.Log(StatsRandom.U(rng)) / C;
}

/// <summary>The Mielke Beta-Kappa / Dagum distribution, scipy's <c>mielke</c>.</summary>
public sealed class MielkeDistribution : ContinuousDistribution
{
    /// <summary>Mielke with shapes <paramref name="k"/>, <paramref name="s"/> &gt; 0.</summary>
    public MielkeDistribution(double k, double s, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        K = StatsDistributions.Positive(k, nameof(k));
        S = StatsDistributions.Positive(s, nameof(s));
    }

    /// <summary>The shape k.</summary>
    public double K { get; }
    /// <summary>The shape s.</summary>
    public double S { get; }

    /// <inheritdoc />
    public override string Name => "mielke";
    /// <inheritdoc />
    public override string[] ShapeNames => ["k", "s"];
    /// <inheritdoc />
    public override double[] Shapes => [K, S];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new MielkeDistribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(K) + Special.Xlogy(K - 1, z) - (1 + K / S) * Special.Log1p(Math.Pow(z, S));
    /// <inheritdoc />
    protected override double LogCdf0(double z)
        => z >= 1 ? -(K / S) * Special.Log1p(Math.Pow(z, -S)) : K * Math.Log(z) - (K / S) * Special.Log1p(Math.Pow(z, S));
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Exp(LogCdf0(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => -Special.Expm1(LogCdf0(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => Log1mExp(LogCdf0(z));
    private double FromLogQ(double logQ)
    {
        var qsk = Math.Exp(S / K * logQ);
        return Math.Pow(qsk / -Special.Expm1(S / K * logQ), 1 / S);
    }
    /// <inheritdoc />
    protected override double Ppf0(double q) => FromLogQ(Math.Log(q));
    /// <inheritdoc />
    protected override double Isf0(double q) => FromLogQ(Special.Log1p(-q));
    /// <inheritdoc />
    protected override double Munp0(int n)
        => n < S ? Math.Exp(Special.GammaLn((K + n) / S) + Special.GammaLn(1 - n / S) - Special.GammaLn(K / S)) : double.PositiveInfinity;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => FromLogQ(Math.Log(StatsRandom.U(rng)));
}

/// <summary>The Moyal distribution, scipy's <c>moyal</c>.</summary>
public sealed class MoyalDistribution : ContinuousDistribution
{
    private const double Zeta3 = 1.2020569031595942853997;

    /// <summary>Moyal.</summary>
    public MoyalDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "moyal";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new MoyalDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => -0.5 * (z + Math.Exp(-z)) - LnSqrt2Pi;
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.Erfc(Math.Exp(-0.5 * z) / Sqrt2);
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.Erf(Math.Exp(-0.5 * z) / Sqrt2);
    /// <inheritdoc />
    protected override double Ppf0(double q) => -Ln2 - 2 * Math.Log(Special.ErfcInv(q));
    /// <summary>−log(2·erfinv(q)²), in logs so a tail probability whose erfinv squares to zero still inverts.</summary>
    protected override double Isf0(double q) => -Ln2 - 2 * Math.Log(Special.ErfInv(q));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
        => new(Ln2 + EulerGamma, Math.PI * Math.PI / 2, 28 * Sqrt2 * Zeta3 / Math.Pow(Math.PI, 3), 4.0);
    /// <summary>Closed form: with Y = e^(−Z) ~ χ²₁, H = ½ log 2π + ½(E[Z] + E[Y]) = ½(log 4π + γ + 1).</summary>
    protected override double Entropy0() => LnSqrt2Pi + 0.5 * (Ln2 + EulerGamma + 1);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => -Math.Log(2 * StatsRandom.Gamma(rng, 0.5));
}
