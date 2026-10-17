using System.Numerics;
using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.StatsDistMath;

namespace Gravicode.Science.SciNet;

/// <summary>The power log-normal distribution, scipy's <c>powerlognorm</c>: survival Φ(−log(x)/s)^c.</summary>
/// <remarks>Quantiles go through the inverse of log Φ, so q^(1/c) close to 1 does not round away the tail.</remarks>
public sealed class PowerLogNormDistribution : ContinuousDistribution
{
    /// <summary>Power log-normal with shapes <paramref name="c"/>, <paramref name="s"/> &gt; 0.</summary>
    public PowerLogNormDistribution(double c, double s, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        C = StatsDistributions.Positive(c, nameof(c));
        S = StatsDistributions.Positive(s, nameof(s));
    }

    /// <summary>The power c.</summary>
    public double C { get; }
    /// <summary>The log-scale s.</summary>
    public double S { get; }

    /// <inheritdoc />
    public override string Name => "powerlognorm";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c", "s"];
    /// <inheritdoc />
    public override double[] Shapes => [C, S];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new PowerLogNormDistribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => z <= 0 ? 0.0 : Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
    {
        if (z <= 0) return double.NegativeInfinity;
        var l = Math.Log(z) / S;
        return Math.Log(C) - Math.Log(z) - Math.Log(S) + LogPhi(l) + (C - 1) * Special.LogNdtr(-l);
    }
    /// <inheritdoc />
    protected override double LogSf0(double z) => C * Special.LogNdtr(-Math.Log(z) / S);
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(LogSf0(z));
    /// <inheritdoc />
    protected override double Cdf0(double z) => -Special.Expm1(LogSf0(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Log1mExp(LogSf0(z));
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Exp(-S * NdtriExp(Math.Log(q) / C));
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Exp(-S * NdtriExp(Special.Log1p(-q) / C));
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Isf0(StatsRandom.U(rng));
}

/// <summary>The power normal distribution, scipy's <c>powernorm</c>: survival Φ(−x)^c.</summary>
public sealed class PowerNormDistribution : ContinuousDistribution
{
    /// <summary>Power normal with shape <paramref name="c"/> &gt; 0.</summary>
    public PowerNormDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Positive(c, nameof(c));

    /// <summary>The power c.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "powernorm";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new PowerNormDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(C) + LogPhi(z) + (C - 1) * Special.LogNdtr(-z);
    /// <inheritdoc />
    protected override double LogSf0(double z) => C * Special.LogNdtr(-z);
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(LogSf0(z));
    /// <inheritdoc />
    protected override double Cdf0(double z) => -Special.Expm1(LogSf0(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Log1mExp(LogSf0(z));
    /// <inheritdoc />
    protected override double Ppf0(double q) => -NdtriExp(Special.Log1p(-q) / C);
    /// <inheritdoc />
    protected override double Isf0(double q) => -NdtriExp(Math.Log(q) / C);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Isf0(StatsRandom.U(rng));
}

/// <summary>The R distribution, scipy's <c>rdist</c>: 2B − 1 for B ~ Beta(c/2, c/2), on [−1, 1].</summary>
public sealed class RDistribution : ContinuousDistribution
{
    /// <summary>R distribution with shape <paramref name="c"/> &gt; 0.</summary>
    public RDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Positive(c, nameof(c));

    /// <summary>The shape c.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "rdist";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new RDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => -1.0;
    /// <inheritdoc />
    protected override double Upper0 => 1.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
    {
        var h = C / 2;
        return -Ln2 + Special.Xlog1py(h - 1, z) + Special.Xlog1py(h - 1, -z) - 2 * (h - 1) * Ln2 - Special.BetaLn(h, h);
    }
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.BetaInc(C / 2, C / 2, (1 + z) / 2);
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.BetaInc(C / 2, C / 2, (1 - z) / 2);
    /// <inheritdoc />
    protected override double Ppf0(double q) => q <= 0.5 ? 2 * Special.BetaIncInv(C / 2, C / 2, q) - 1 : 1 - 2 * Special.BetaIncInv(C / 2, C / 2, 1 - q);
    /// <inheritdoc />
    protected override double Isf0(double q) => -Ppf0(q);
    /// <inheritdoc />
    protected override double Munp0(int n)
        => n % 2 == 1 ? 0.0 : Math.Exp(Special.BetaLn((n + 1.0) / 2, C / 2) - Special.BetaLn(0.5, C / 2));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var m2 = Munp0(2);
        return new(0.0, m2, 0.0, Munp0(4) / (m2 * m2) - 3);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => 2 * StatsRandom.Beta(rng, C / 2, C / 2) - 1;
}

/// <summary>The skewed Cauchy distribution, scipy's <c>skewcauchy</c>, with |a| &lt; 1.</summary>
/// <remarks>
/// Each tail is written as an atan2 of the two scale factors rather than ½ ± atan(…)/π, and each
/// tail quantile as a cotangent of the tail probability, so both tails keep relative accuracy.
/// </remarks>
public sealed class SkewCauchyDistribution : ContinuousDistribution
{
    /// <summary>Skewed Cauchy with skewness <paramref name="a"/> in (−1, 1).</summary>
    public SkewCauchyDistribution(double a, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        if (!(Math.Abs(a) < 1)) throw new ArgumentException("a must lie in (-1, 1).", nameof(a));
        A = a;
    }

    /// <summary>The skewness a.</summary>
    public double A { get; }

    /// <inheritdoc />
    public override string Name => "skewcauchy";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a"];
    /// <inheritdoc />
    public override double[] Shapes => [A];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new SkewCauchyDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double z)
    {
        var d = A * Math.Sign(z) + 1;
        return 1 / (Math.PI * (z * z / (d * d) + 1));
    }
    /// <inheritdoc />
    protected override double Cdf0(double z) => z <= 0 ? (1 - A) / Math.PI * Math.Atan2(1 - A, -z) : 1 - Sf0(z);
    /// <inheritdoc />
    protected override double Sf0(double z) => z > 0 ? (1 + A) / Math.PI * Math.Atan2(1 + A, z) : 1 - Cdf0(z);
    /// <inheritdoc />
    protected override double Ppf0(double q)
        => q <= (1 - A) / 2 ? -(1 - A) / Math.Tan(Math.PI * q / (1 - A)) : (1 + A) / Math.Tan(Math.PI * (1 - q) / (1 + A));
    /// <inheritdoc />
    protected override double Isf0(double q)
        => q <= (1 + A) / 2 ? (1 + A) / Math.Tan(Math.PI * q / (1 + A)) : -(1 - A) / Math.Tan(Math.PI * (1 - q) / (1 - A));
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(double.NaN, double.NaN, double.NaN, double.NaN);
    /// <inheritdoc />
    protected override double Munp0(int n) => double.NaN;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));
}

/// <summary>The trapezoidal distribution, scipy's <c>trapezoid</c>: rising on [0, c], flat on [c, d], falling on [d, 1].</summary>
public sealed class TrapezoidDistribution : ContinuousDistribution
{
    /// <summary>Trapezoid with 0 ≤ <paramref name="c"/> ≤ <paramref name="d"/> ≤ 1.</summary>
    public TrapezoidDistribution(double c, double d, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        if (!(c >= 0 && c <= 1 && d >= 0 && d <= 1 && d >= c)) throw new ArgumentException("Need 0 <= c <= d <= 1.");
        C = c;
        D = d;
    }

    /// <summary>Start of the flat top.</summary>
    public double C { get; }
    /// <summary>End of the flat top.</summary>
    public double D { get; }

    /// <inheritdoc />
    public override string Name => "trapezoid";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c", "d"];
    /// <inheritdoc />
    public override double[] Shapes => [C, D];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new TrapezoidDistribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Upper0 => 1.0;
    private double W => D - C + 1;
    /// <inheritdoc />
    protected override double Pdf0(double z)
    {
        var u = 2 / W;
        if (z < C) return u * z / C;
        if (z <= D) return u;
        return u * (1 - z) / (1 - D);
    }
    /// <inheritdoc />
    protected override double Cdf0(double z)
    {
        if (z < C) return z * z / C / W;
        if (z <= D) return (C + 2 * (z - C)) / W;
        return 1 - (1 - z) * (1 - z) / W / (1 - D);
    }
    /// <inheritdoc />
    protected override double Sf0(double z)
    {
        if (z > D) return (1 - z) * (1 - z) / W / (1 - D);
        if (z >= C) return (D + 1 - 2 * z) / W;
        return 1 - z * z / C / W;
    }
    /// <inheritdoc />
    protected override double Ppf0(double q)
    {
        var qc = C / W;
        var qd = (2 * D - C) / W;
        if (q < qc) return Math.Sqrt(q * C * W);
        if (q <= qd) return 0.5 * q * W + 0.5 * C;
        return 1 - Math.Sqrt((1 - q) * W * (1 - D));
    }
    /// <inheritdoc />
    protected override double Isf0(double q)
    {
        var pd = (1 - D) / W; // Sf(d)
        var pc = (1 + D - 2 * C) / W; // Sf(c)
        if (q < pd) return 1 - Math.Sqrt(q * W * (1 - D));
        if (q <= pc) return (D + 1 - q * W) / 2;
        return Math.Sqrt((1 - q) * C * W);
    }
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        double dc = D == 0 ? 1.0 : D == 1 ? n + 2 : Special.Expm1((n + 2) * Math.Log(D)) / (D - 1);
        return 2.0 / (1 + D - C) * (dc - Math.Pow(C, n + 1)) / ((n + 1) * (n + 2));
    }
    /// <inheritdoc />
    protected override double Entropy0() => 0.5 * (1 - D + C) / (1 + D - C) + Math.Log(0.5 * (1 + D - C));
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));
}

/// <summary>The upper-truncated Pareto distribution, scipy's <c>truncpareto</c>, on [1, c].</summary>
/// <remarks>b may be negative (an increasing power density), as in scipy; only b = 0 is excluded.</remarks>
public sealed class TruncParetoDistribution : ContinuousDistribution
{
    private readonly double _logC;
    private readonly double _em; // expm1(−b log c) = c^(−b) − 1, negative for b > 0

    /// <summary>Truncated Pareto with exponent <paramref name="b"/> ≠ 0 and upper end <paramref name="c"/> &gt; 1.</summary>
    public TruncParetoDistribution(double b, double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        if (!(b != 0) || !double.IsFinite(b)) throw new ArgumentException("b must be finite and non-zero.", nameof(b));
        if (!(c > 1) || !double.IsFinite(c)) throw new ArgumentException("c must exceed 1.", nameof(c));
        B = b;
        C = c;
        _logC = Math.Log(c);
        _em = Special.Expm1(-b * _logC);
    }

    /// <summary>The exponent b.</summary>
    public double B { get; }
    /// <summary>The upper end c.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "truncpareto";
    /// <inheritdoc />
    public override string[] ShapeNames => ["b", "c"];
    /// <inheritdoc />
    public override double[] Shapes => [B, C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new TruncParetoDistribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 1.0;
    /// <inheritdoc />
    protected override double Upper0 => C;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(B / -_em) - (B + 1) * Math.Log(z);
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.Expm1(-B * Math.Log(z)) / _em;
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(-B * Math.Log(z)) * Special.Expm1(-B * Math.Log(C / z)) / _em;
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Exp(-Special.Log1p(q * _em) / B);
    /// <inheritdoc />
    protected override double Isf0(double q) => C * Math.Exp(-Special.Log1p(q * Special.Expm1(B * _logC)) / B);
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        if (n == B) return B * _logC / -_em;
        // b/(b − n)·(cᵇ − cⁿ)/(cᵇ − 1), written with c^(−b) so a large cᵇ does not overflow.
        return B / (B - n) * Special.Expm1((n - B) * _logC) / _em;
    }
    /// <inheritdoc />
    protected override double Entropy0() => -(Math.Log(B / -_em) + (B + 1) * (_logC / Special.Expm1(B * _logC) - 1 / B));
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));
}

/// <summary>The doubly truncated Weibull minimum distribution, scipy's <c>truncweibull_min</c>, on [a, b].</summary>
/// <remarks>
/// The truncation points are in standardised units, as in scipy. With A = aᶜ, B = bᶜ every probability
/// is a ratio of expm1 terms, so neither a narrow window nor a window far in the tail loses digits.
/// </remarks>
public sealed class TruncWeibullMinDistribution : ContinuousDistribution
{
    private readonly double _ac, _bc, _logDen; // log(e^(−A) − e^(−B))

    /// <summary>Truncated Weibull with shape <paramref name="c"/> &gt; 0 on [<paramref name="a"/>, <paramref name="b"/>], 0 ≤ a &lt; b.</summary>
    public TruncWeibullMinDistribution(double c, double a, double b, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        C = StatsDistributions.Positive(c, nameof(c));
        if (!(a >= 0) || !(b > a) || !double.IsFinite(a)) throw new ArgumentException("Need 0 <= a < b.");
        A = a;
        B = b;
        _ac = Math.Pow(a, c);
        _bc = Math.Pow(b, c);
        _logDen = -_ac + Log1mExp(_ac - _bc);
    }

    /// <summary>The Weibull shape c.</summary>
    public double C { get; }
    /// <summary>The lower truncation point a.</summary>
    public double A { get; }
    /// <summary>The upper truncation point b.</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "truncweibull_min";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c", "a", "b"];
    /// <inheritdoc />
    public override double[] Shapes => [C, A, B];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new TruncWeibullMinDistribution(shapes[0], shapes[1], shapes[2], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => A;
    /// <inheritdoc />
    protected override double Upper0 => B;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(C) + Special.Xlogy(C - 1, z) - Math.Pow(z, C) - _logDen;
    /// <inheritdoc />
    protected override double LogCdf0(double z) => -_ac + Log1mExp(_ac - Math.Pow(z, C)) - _logDen;
    /// <inheritdoc />
    protected override double LogSf0(double z) { var zc = Math.Pow(z, C); return -zc + Log1mExp(zc - _bc) - _logDen; }
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Exp(LogCdf0(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(LogSf0(z));
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Pow(_ac - Special.Log1p(-q * -Special.Expm1(_ac - _bc)), 1 / C);
    /// <inheritdoc />
    protected override double Isf0(double q)
    {
        var zc = _bc - _ac > 700 ? _ac - Math.Log(q + (1 - q) * Math.Exp(_ac - _bc)) : _bc - Special.Log1p(q * Special.Expm1(_bc - _ac));
        return Math.Pow(zc, 1 / C);
    }
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        var s = n / C + 1;
        var diff = _ac > s ? Special.GammaIncC(s, _ac) - Special.GammaIncC(s, _bc) : Special.GammaInc(s, _bc) - Special.GammaInc(s, _ac);
        return Math.Exp(Special.GammaLn(s) - _logDen) * diff;
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));
}

/// <summary>The wrapped Cauchy distribution, scipy's <c>wrapcauchy</c>, on [0, 2π].</summary>
public sealed class WrapCauchyDistribution : ContinuousDistribution
{
    /// <summary>Wrapped Cauchy with concentration <paramref name="c"/> in (0, 1).</summary>
    public WrapCauchyDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        if (!(c > 0 && c < 1)) throw new ArgumentException("c must lie in (0, 1).", nameof(c));
        C = c;
    }

    /// <summary>The concentration c.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "wrapcauchy";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new WrapCauchyDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Upper0 => 2 * Math.PI;
    private double Cr => (1 + C) / (1 - C);
    /// <inheritdoc />
    protected override double Pdf0(double z) => (1 - C * C) / (2 * Math.PI * (1 + C * C - 2 * C * Math.Cos(z)));
    /// <inheritdoc />
    protected override double Cdf0(double z) => z < Math.PI ? Math.Atan(Cr * Math.Tan(z / 2)) / Math.PI : 1 - Sf0(z);
    /// <inheritdoc />
    protected override double Sf0(double z) => z > Math.PI ? Math.Atan(Cr * Math.Tan((2 * Math.PI - z) / 2)) / Math.PI : 1 - Cdf0(z);
    /// <inheritdoc />
    protected override double Ppf0(double q)
        => q < 0.5 ? 2 * Math.Atan(Math.Tan(Math.PI * q) / Cr) : 2 * Math.PI - 2 * Math.Atan(Math.Tan(Math.PI * (1 - q)) / Cr);
    /// <inheritdoc />
    protected override double Isf0(double q) => 2 * Math.PI - Ppf0(q);
    /// <inheritdoc />
    protected override double Entropy0() => Math.Log(2 * Math.PI * (1 - C * C));
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));
}

/// <summary>The Kappa 3 distribution, scipy's <c>kappa3</c>: CDF x(a + xᵃ)^(−1/a).</summary>
public sealed class Kappa3Distribution : ContinuousDistribution
{
    /// <summary>Kappa 3 with shape <paramref name="a"/> &gt; 0.</summary>
    public Kappa3Distribution(double a, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => A = StatsDistributions.Positive(a, nameof(a));

    /// <summary>The shape a.</summary>
    public double A { get; }

    /// <inheritdoc />
    public override string Name => "kappa3";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a"];
    /// <inheritdoc />
    public override double[] Shapes => [A];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new Kappa3Distribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(A) - (1 / A + 1) * Math.Log(A + Math.Pow(z, A));
    /// <summary>log CDF = −log1p(a·z^(−a))/a, the same expression at both ends of the support.</summary>
    protected override double LogCdf0(double z) => -Special.Log1p(A * Math.Exp(-A * Math.Log(z))) / A;
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Exp(LogCdf0(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => -Special.Expm1(LogCdf0(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => Log1mExp(LogCdf0(z));
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Pow(A / Special.Expm1(-A * Math.Log(q)), 1 / A);
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Pow(A / Special.Expm1(-A * Special.Log1p(-q)), 1 / A);
    /// <summary>Raw moments by quadrature; they exist only below order a (NaN otherwise, as scipy).</summary>
    protected override double Munp0(int n) => n < A ? base.Munp0(n) : double.NaN;
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var m = new double[5];
        for (var i = 1; i <= 4; i++) m[i] = Munp0(i);
        var r = StatsDistMoreMath.FromRaw(m[1], m[2], m[3], m[4]);
        // scipy reports each moment it can compute; a NaN fourth moment must not poison the mean.
        return new(m[1], double.IsNaN(m[2]) ? double.NaN : r.Variance, double.IsNaN(m[3]) ? double.NaN : r.Skewness, double.IsNaN(m[4]) ? double.NaN : r.Kurtosis);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));
}

/// <summary>The four-parameter kappa distribution of Hosking (1994), scipy's <c>kappa4</c>.</summary>
/// <remarks>
/// Covers the GEV (h = 0), generalised logistic (h = −1), generalised Pareto (h = 1), Gumbel
/// (h = k = 0) and more. Quantiles are closed-form for both tails; moments integrate the quantile
/// function, as scipy does, and exist only below the order its shapes allow.
/// </remarks>
public sealed class Kappa4Distribution : ContinuousDistribution
{
    /// <summary>Kappa 4 with shapes <paramref name="h"/> and <paramref name="k"/> (any reals).</summary>
    public Kappa4Distribution(double h, double k, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        H = StatsDistributions.Finite(h, nameof(h));
        K = StatsDistributions.Finite(k, nameof(k));
    }

    /// <summary>The shape h.</summary>
    public double H { get; }
    /// <summary>The shape k.</summary>
    public double K { get; }

    /// <inheritdoc />
    public override string Name => "kappa4";
    /// <inheritdoc />
    public override string[] ShapeNames => ["h", "k"];
    /// <inheritdoc />
    public override double[] Shapes => [H, K];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new Kappa4Distribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0
        => H > 0 ? (K == 0 ? Math.Log(H) : -Special.Expm1(-K * Math.Log(H)) / K) : (K < 0 ? 1 / K : double.NegativeInfinity);
    /// <inheritdoc />
    protected override double Upper0 => K > 0 ? 1 / K : double.PositiveInfinity;
    /// <summary>log y with y = (1 − kz)^(1/k), or e^(−z) when k = 0.</summary>
    private double LogY(double z) => K == 0 ? -z : Special.Log1p(-K * z) / K;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
    {
        var ly = LogY(z);
        var first = K == 0 ? -z : Special.Xlog1py(1 / K - 1, -K * z);
        return H == 0 ? first - Math.Exp(ly) : first + Special.Xlog1py(1 / H - 1, -H * Math.Exp(ly));
    }
    /// <inheritdoc />
    protected override double LogCdf0(double z)
    {
        var y = Math.Exp(LogY(z));
        return H == 0 ? -y : Special.Log1p(-H * y) / H;
    }
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Exp(LogCdf0(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => -Special.Expm1(LogCdf0(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => Log1mExp(LogCdf0(z));
    /// <summary>The quantile from log F: T = (1 − Fʰ)/h (−log F at h = 0), x = (1 − Tᵏ)/k (−log T at k = 0).</summary>
    private double FromLogF(double logF)
    {
        var t = H == 0 ? -logF : -Special.Expm1(H * logF) / H;
        return K == 0 ? -Math.Log(t) : -Special.Expm1(K * Math.Log(t)) / K;
    }
    /// <inheritdoc />
    protected override double Ppf0(double q) => FromLogF(Math.Log(q));
    /// <inheritdoc />
    protected override double Isf0(double q) => FromLogF(Special.Log1p(-q));
    private int MaxOrder => H < 0 && K >= 0 ? (int)(-K / H) : K < 0 ? (int)(-1 / K) : 5;
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        if (n >= MaxOrder) return double.NaN;
        return StatsDistMoreMath.QuadRel(q => Math.Pow(Ppf0(q), n), 0.0, 1.0, [0.5], 1e-12);
    }
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var m = new double[5];
        for (var i = 1; i <= 4; i++) m[i] = Munp0(i);
        var r = StatsDistMoreMath.FromRaw(m[1], m[2], m[3], m[4]);
        return new(m[1], double.IsNaN(m[2]) ? double.NaN : r.Variance, double.IsNaN(m[3]) ? double.NaN : r.Skewness, double.IsNaN(m[4]) ? double.NaN : r.Kurtosis);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));
}

/// <summary>The Crystal Ball distribution, scipy's <c>crystalball</c>: a normal core with a power-law left tail.</summary>
public sealed class CrystalBallDistribution : ContinuousDistribution
{
    private readonly double _n, _c, _pBeta;

    /// <summary>Crystal Ball with transition point <paramref name="beta"/> &gt; 0 and tail power <paramref name="m"/> &gt; 1.</summary>
    public CrystalBallDistribution(double beta, double m, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        Beta = StatsDistributions.Positive(beta, nameof(beta));
        if (!(m > 1) || !double.IsFinite(m)) throw new ArgumentException("m must exceed 1.", nameof(m));
        M = m;
        _c = m / beta / (m - 1) * Math.Exp(-beta * beta / 2);
        _n = 1 / (_c + Sqrt2Pi * Special.Ndtr(beta));
        _pBeta = _n * _c;
    }

    /// <summary>Where the power-law tail starts (at −β).</summary>
    public double Beta { get; }
    /// <summary>The power of the tail.</summary>
    public double M { get; }

    /// <inheritdoc />
    public override string Name => "crystalball";
    /// <inheritdoc />
    public override string[] ShapeNames => ["beta", "m"];
    /// <inheritdoc />
    public override double[] Shapes => [Beta, M];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new CrystalBallDistribution(shapes[0], shapes[1], loc, scale);
    private double Bm => M / Beta - Beta;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
        => Math.Log(_n) + (z > -Beta ? -z * z / 2 : M * Math.Log(M / Beta) - Beta * Beta / 2 - M * Math.Log(Bm - z));
    /// <inheritdoc />
    protected override double Cdf0(double z)
    {
        if (z <= -Beta) return _n * Math.Exp(M * Math.Log(M / Beta) - Beta * Beta / 2 + (1 - M) * Math.Log(Bm - z)) / (M - 1);
        return z > 0 ? 1 - Sf0(z) : _n * (_c + Sqrt2Pi * StatsDistMoreMath.NormInterval(-Beta, z));
    }
    /// <inheritdoc />
    protected override double Sf0(double z) => z > -Beta ? _n * Sqrt2Pi * Special.Ndtr(-z) : 1 - Cdf0(z);
    /// <inheritdoc />
    protected override double LogSf0(double z) => z > -Beta ? Math.Log(_n * Sqrt2Pi) + Special.LogNdtr(-z) : Special.Log1p(-Cdf0(z));
    /// <inheritdoc />
    protected override double Ppf0(double q)
    {
        if (q < _pBeta)
            return Bm - Math.Pow((M - 1) * Math.Pow(M / Beta, -M) * Math.Exp(Beta * Beta / 2) * q / _n, 1 / (1 - M));
        if (q > 0.5) return Isf0(1 - q);
        return Special.Ndtri(Special.Ndtr(-Beta) + (q / _n - _c) / Sqrt2Pi);
    }
    /// <inheritdoc />
    protected override double Isf0(double q) => q < 1 - _pBeta ? -Special.Ndtri(q / (_n * Sqrt2Pi)) : Ppf0(1 - q);
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        if (!(n + 1 < M)) return double.PositiveInfinity;
        var a = Math.Pow(M / Beta, M) * Math.Exp(-Beta * Beta / 2);
        var b = Bm;
        var rhs = Math.Pow(2, (n - 1) / 2.0) * Special.Gamma((n + 1) / 2.0) * (1 + (n % 2 == 0 ? 1 : -1) * Special.GammaInc((n + 1) / 2.0, Beta * Beta / 2));
        var lhs = 0.0;
        for (var k = 0; k <= n; k++)
            lhs += Special.Binom(n, k) * Math.Pow(b, n - k) * (k % 2 == 0 ? 1 : -1) / (M - k - 1) * Math.Pow(M / Beta, -M + k + 1);
        return _n * (a * lhs + rhs);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));
}

/// <summary>The ARGUS distribution, scipy's <c>argus</c>, on [0, 1].</summary>
/// <remarks>
/// Ψ(χ) = P(3/2, χ²/2)/2. The survival function Ψ(χ√(1 − x²))/Ψ(χ) is exact; the CDF near 0 is
/// a short quadrature of the density instead of 1 − SF. Variates use scipy's three regimes
/// (Baumgarten 2017 rejection for small χ, a gamma ratio for large χ).
/// </remarks>
public sealed class ArgusDistribution : ContinuousDistribution
{
    private readonly double _logA;

    /// <summary>ARGUS with curvature <paramref name="chi"/> &gt; 0.</summary>
    public ArgusDistribution(double chi, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        Chi = StatsDistributions.Positive(chi, nameof(chi));
        _logA = 3 * Math.Log(Chi) - LnSqrt2Pi - Math.Log(Psi(Chi));
    }

    /// <summary>The curvature χ.</summary>
    public double Chi { get; }

    private static double Psi(double x) => Special.GammaInc(1.5, x * x / 2) / 2;

    /// <inheritdoc />
    public override string Name => "argus";
    /// <inheritdoc />
    public override string[] ShapeNames => ["chi"];
    /// <inheritdoc />
    public override double[] Shapes => [Chi];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new ArgusDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Upper0 => 1.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => z <= 0 || z >= 1 ? 0.0 : Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
        => _logA + Math.Log(z) + 0.5 * Special.Log1p(-z * z) - Chi * Chi * (1 - z) * (1 + z) / 2;
    /// <inheritdoc />
    protected override double Sf0(double z) => Psi(Chi * Math.Sqrt((1 - z) * (1 + z))) / Psi(Chi);
    /// <inheritdoc />
    protected override double Cdf0(double z)
    {
        var sf = Sf0(z);
        return sf < 0.5 ? 1 - sf : StatsDistMoreMath.QuadRel(Pdf0, 0.0, z);
    }
    /// <summary>
    /// Small quantiles start from CDF ≈ A e^(−χ²/2) x²/2 and are polished by Newton: bracketing on
    /// [0, 1] cannot reach x ~ 1e-150 within any sensible iteration budget.
    /// </summary>
    protected override double Ppf0(double q)
    {
        if (q >= 1e-4) return base.Ppf0(q);
        var x = Math.Sqrt(2 * q * Math.Exp(Chi * Chi / 2 - _logA));
        for (var i = 0; i < 20; i++)
        {
            var step = (Math.Log(Cdf0(x)) - Math.Log(q)) * Cdf0(x) / Pdf0(x);
            x -= step;
            if (Math.Abs(step) <= 1e-15 * x) break;
        }
        return x;
    }
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var phi = Psi(Chi);
        var m = Math.Sqrt(Math.PI / 8) * Chi * Special.Ive(1, Chi * Chi / 4) / phi;
        double mu2;
        if (Chi > 0.1) mu2 = 1 - 3 / (Chi * Chi) + Chi * Math.Exp(LogPhi(Chi)) / phi;
        else mu2 = StatsDistMath.PolyEval([0.4, 0, 6 / 175.0, 0, 2 / 2625.0, 0, -94 / 1010625.0, 0, -358 / 65690625.0], Chi);
        var r = StatsDistMoreMath.FromRaw(m, mu2, Munp0(3), Munp0(4));
        return new(m, mu2 - m * m, r.Skewness, r.Kurtosis);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        var chi2 = Chi * Chi;
        if (Chi <= 0.5)
        {
            var d = -chi2 / 2;
            while (true)
            {
                var u = StatsRandom.U(rng);
                var z = Math.Pow(StatsRandom.U(rng), 2.0 / 3.0);
                if (Math.Log(u) <= d * z) return Math.Sqrt(1 - z);
            }
        }
        if (Chi <= 1.8)
        {
            var echi = Math.Exp(-chi2 / 2);
            while (true)
            {
                var u = StatsRandom.U(rng);
                var v = StatsRandom.U(rng);
                var z = 2 * Math.Log(echi * (1 - v) + v) / chi2;
                if (u * u + z <= 0) return Math.Sqrt(1 + z);
            }
        }
        while (true)
        {
            var g = StatsRandom.Gamma(rng, 1.5);
            if (g <= chi2 / 2) return Math.Sqrt(1 - 2 * g / chi2);
        }
    }
}

/// <summary>The reciprocal inverse Gaussian distribution, scipy's <c>recipinvgauss</c>: 1/Y for Y ~ invgauss(μ).</summary>
/// <remarks>Every function is the inverse-Gaussian one at 1/x, with the tails swapped, so both inherit its accuracy.</remarks>
public sealed class RecipInvGaussDistribution : ContinuousDistribution
{
    private readonly InvGaussDistribution _ig;

    /// <summary>Reciprocal inverse Gaussian with shape <paramref name="mu"/> &gt; 0.</summary>
    public RecipInvGaussDistribution(double mu, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        Mu = StatsDistributions.Positive(mu, nameof(mu));
        _ig = new InvGaussDistribution(mu);
    }

    /// <summary>The shape μ.</summary>
    public double Mu { get; }

    /// <inheritdoc />
    public override string Name => "recipinvgauss";
    /// <inheritdoc />
    public override string[] ShapeNames => ["mu"];
    /// <inheritdoc />
    public override double[] Shapes => [Mu];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new RecipInvGaussDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => z <= 0 ? 0.0 : Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
        => z <= 0 ? double.NegativeInfinity : -(1 - Mu * z) * (1 - Mu * z) / (2 * z * Mu * Mu) - 0.5 * Math.Log(2 * Math.PI * z);
    /// <inheritdoc />
    protected override double Cdf0(double z) => _ig.Sf(1 / z);
    /// <inheritdoc />
    protected override double Sf0(double z) => _ig.Cdf(1 / z);
    /// <inheritdoc />
    protected override double LogCdf0(double z) => _ig.LogSf(1 / z);
    /// <inheritdoc />
    protected override double LogSf0(double z) => _ig.LogCdf(1 / z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => 1 / _ig.Isf(q);
    /// <inheritdoc />
    protected override double Isf0(double q) => 1 / _ig.Ppf(q);
    /// <summary>E[Y^(−n)] = E[Y^(n+1)]/μ^(2n+1) for the inverse Gaussian with λ = 1.</summary>
    protected override double Munp0(int n) => StatsDistMoreMath.InvGaussRawMoment(Mu, n + 1) / Math.Pow(Mu, 2 * n + 1);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => 1 / _ig.Rvs(rng);
}

/// <summary>The relativistic Breit–Wigner distribution, scipy's <c>rel_breitwigner</c>.</summary>
/// <remarks>
/// The CDF is scipy's complex arctangent closed form; the survival function uses the same form with
/// atan(1/(xw)) in place of π/2 − atan(xw), which is exact in the upper tail (scipy's 1 − CDF is
/// not: at x = 10⁴, ρ = 0.7 it is wrong in the 4th digit).
/// </remarks>
public sealed class RelBreitWignerDistribution : ContinuousDistribution
{
    private readonly double _cPdf, _cCdf;
    private readonly Complex _s, _w;

    /// <summary>Relativistic Breit–Wigner with shape <paramref name="rho"/> &gt; 0 (mass over width).</summary>
    public RelBreitWignerDistribution(double rho, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        Rho = StatsDistributions.Positive(rho, nameof(rho));
        var r = Math.Sqrt(1 + 1 / (rho * rho));
        _cPdf = Math.Sqrt(2 * (1 + 1 / (rho * rho)) / (1 + r)) * 2 / Math.PI;
        _cCdf = Math.Sqrt(2 / (1 + r)) / Math.PI;
        _s = Complex.Sqrt(new Complex(-1, 1 / rho));
        _w = Complex.Sqrt(-rho * new Complex(rho, 1));
    }

    /// <summary>The shape ρ.</summary>
    public double Rho { get; }

    /// <inheritdoc />
    public override string Name => "rel_breitwigner";
    /// <inheritdoc />
    public override string[] ShapeNames => ["rho"];
    /// <inheritdoc />
    public override double[] Shapes => [Rho];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new RelBreitWignerDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z)
    {
        var t = (z - Rho) * (z + Rho) / Rho;
        return _cPdf / (t * t + 1);
    }
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Min(1.0, 2 * _cCdf * (_s * Complex.Atan(z / _w)).Imaginary);
    /// <inheritdoc />
    protected override double Sf0(double z) => z == 0 ? 1.0 : Math.Max(0.0, 2 * _cCdf * (_s * Complex.Atan(_w / z)).Imaginary);
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        if (n == 1)
            return Math.Sqrt(2 * (1 + 1 / (Rho * Rho)) / (1 + Math.Sqrt(1 + 1 / (Rho * Rho)))) / Math.PI * Rho * (Math.PI / 2 + Math.Atan(Rho));
        if (n == 2)
        {
            var c = Math.Sqrt((1 + 1 / (Rho * Rho)) / (2 * (1 + Math.Sqrt(1 + 1 / (Rho * Rho))))) * Rho;
            return 2 * c * (new Complex(1, -Rho) / Complex.Sqrt(new Complex(-1, -1 / Rho))).Real;
        }
        return double.PositiveInfinity;
    }
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var m = Munp0(1);
        return new(m, Munp0(2) - m * m, double.NaN, double.NaN);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));
}

/// <summary>Jones and Faddy's skew-t distribution, scipy's <c>jf_skew_t</c>.</summary>
/// <remarks>
/// With r = √(a + b + x²), the factors 1 ± x/r are formed as (r ± x)/r with the cancelling one
/// rewritten as (a + b)/(r(r ∓ x)), so the density and both tails stay accurate for large |x|.
/// </remarks>
public sealed class JFSkewTDistribution : ContinuousDistribution
{
    private readonly double _logC;

    /// <summary>Jones–Faddy skew-t with shapes <paramref name="a"/>, <paramref name="b"/> &gt; 0 (a = b is Student's t with 2a df, rescaled).</summary>
    public JFSkewTDistribution(double a, double b, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        A = StatsDistributions.Positive(a, nameof(a));
        B = StatsDistributions.Positive(b, nameof(b));
        _logC = (a + b - 1) * Ln2 + Special.BetaLn(a, b) + 0.5 * Math.Log(a + b);
    }

    /// <summary>The shape a.</summary>
    public double A { get; }
    /// <summary>The shape b.</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "jf_skew_t";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a", "b"];
    /// <inheritdoc />
    public override double[] Shapes => [A, B];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new JFSkewTDistribution(shapes[0], shapes[1], loc, scale);
    /// <summary>(r + x)/(2r) and (r − x)/(2r), both without cancellation.</summary>
    private (double Y, double OneMinusY) Halves(double x)
    {
        var s = A + B;
        var r = Math.Sqrt(s + x * x);
        double plus, minus;
        if (x >= 0) { plus = r + x; minus = s / plus; }
        else { minus = r - x; plus = s / minus; }
        return (plus / (2 * r), minus / (2 * r));
    }
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
    {
        var (y, y1) = Halves(z);
        return (A + 0.5) * Math.Log(2 * y) + (B + 0.5) * Math.Log(2 * y1) - _logC;
    }
    /// <inheritdoc />
    protected override double Cdf0(double z) { var (y, y1) = Halves(z); return y <= 0.5 ? Special.BetaInc(A, B, y) : Special.BetaIncC(B, A, y1); }
    /// <inheritdoc />
    protected override double Sf0(double z) { var (y, y1) = Halves(z); return y1 <= 0.5 ? Special.BetaInc(B, A, y1) : Special.BetaIncC(A, B, y); }
    private double FromBeta(double d, double e) => (d - e) * Math.Sqrt(A + B) / (2 * Math.Sqrt(d * e));
    /// <inheritdoc />
    protected override double Ppf0(double q)
    {
        if (q <= 0.5) { var d = Special.BetaIncInv(A, B, q); return FromBeta(d, 1 - d); }
        var e = Special.BetaIncInv(B, A, 1 - q);
        return FromBeta(1 - e, e);
    }
    /// <inheritdoc />
    protected override double Isf0(double q)
    {
        if (q <= 0.5) { var e = Special.BetaIncInv(B, A, q); return FromBeta(1 - e, e); }
        var d = Special.BetaIncInv(A, B, 1 - q);
        return FromBeta(d, 1 - d);
    }
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        if (!(A > 0.5 * n && B > 0.5 * n)) return double.NaN;
        var sum = 0.0;
        for (var i = 0; i <= n; i++)
            sum += Special.Binom(n, i) * (i % 2 == 1 ? -1 : 1) * Math.Exp(Special.BetaLn(A + 0.5 * n - i, B - 0.5 * n + i));
        return Math.Pow(A + B, 0.5 * n) / Math.Pow(2, n) * sum / Math.Exp(Special.BetaLn(A, B));
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        var d = StatsRandom.Beta(rng, A, B);
        return FromBeta(d, 1 - d);
    }
}

/// <summary>
/// The von Mises distribution on the line, scipy's <c>vonmises_line</c>: support [−π, π] after loc and
/// scale (identical to <see cref="VonMisesDistribution"/> here, which already uses that support).
/// </summary>
public sealed class VonMisesLineDistribution : ContinuousDistribution
{
    private readonly VonMisesDistribution _vm;

    /// <summary>von Mises on the line with concentration <paramref name="kappa"/> &gt; 0.</summary>
    public VonMisesLineDistribution(double kappa, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        _vm = new VonMisesDistribution(kappa);
        Kappa = kappa;
    }

    /// <summary>The concentration κ.</summary>
    public double Kappa { get; }

    /// <inheritdoc />
    public override string Name => "vonmises_line";
    /// <inheritdoc />
    public override string[] ShapeNames => ["kappa"];
    /// <inheritdoc />
    public override double[] Shapes => [Kappa];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new VonMisesLineDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => -Math.PI;
    /// <inheritdoc />
    protected override double Upper0 => Math.PI;
    /// <inheritdoc />
    protected override double Pdf0(double z) => _vm.Pdf(z);
    /// <inheritdoc />
    protected override double LogPdf0(double z) => _vm.LogPdf(z);
    /// <inheritdoc />
    protected override double Cdf0(double z) => _vm.Cdf(z);
    /// <inheritdoc />
    protected override double Sf0(double z) => _vm.Sf(z);
    /// <inheritdoc />
    protected override double LogCdf0(double z) => _vm.LogCdf(z);
    /// <inheritdoc />
    protected override double LogSf0(double z) => _vm.LogSf(z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => _vm.Ppf(q);
    /// <inheritdoc />
    protected override double Isf0(double q) => _vm.Isf(q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => _vm.Stats();
    /// <inheritdoc />
    protected override double Entropy0() => _vm.Entropy();
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => _vm.Rvs(rng);
}
