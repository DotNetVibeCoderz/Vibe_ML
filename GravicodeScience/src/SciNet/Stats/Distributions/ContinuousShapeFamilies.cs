using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.StatsDistMath;

namespace Gravicode.Science.SciNet;

/// <summary>The Weibull distribution for minima, scipy's <c>weibull_min</c>.</summary>
public sealed class WeibullMinDistribution : ContinuousDistribution
{
    /// <summary>Weibull with shape <paramref name="c"/>.</summary>
    public WeibullMinDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Positive(c, nameof(c));

    /// <summary>The shape parameter.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "weibull_min";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new WeibullMinDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(C) + Special.Xlogy(C - 1, z) - Math.Pow(z, C);
    /// <inheritdoc />
    protected override double Cdf0(double z) => -Special.Expm1(-Math.Pow(z, C));
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(-Math.Pow(z, C));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Log1mExp(-Math.Pow(z, C));
    /// <inheritdoc />
    protected override double LogSf0(double z) => -Math.Pow(z, C);
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Pow(-Special.Log1p(-q), 1 / C);
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Pow(-Math.Log(q), 1 / C);
    /// <inheritdoc />
    protected override double Munp0(int n) => Special.Gamma(1 + n / C);
    /// <inheritdoc />
    protected override double Entropy0() => -Math.Log(C) + EulerGamma * (1 - 1 / C) + 1;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Pow(StatsRandom.Exponential(rng), 1 / C);

    /// <summary>
    /// MLE. With the location fixed the shape solves the profile equation
    /// 1/c + mean(log y) − Σyᶜ log y / Σyᶜ = 0 and scale = (mean yᶜ)^{1/c}; otherwise numerical.
    /// </summary>
    public static WeibullMinDistribution Fit(double[] data, double? fc = null, double? floc = null, double? fscale = null)
    {
        StatsDistributions.CheckData(data);
        if (floc is double loc && fscale is null)
        {
            var y = data.Select(x => x - loc).ToArray();
            if (y.Any(v => v <= 0)) throw new ArgumentException("All data must exceed floc.", nameof(floc));
            var ly = y.Select(v => Math.Log(v)).ToArray();
            var meanLog = ly.Average();
            var maxLog = ly.Max();
            double Profile(double c)
            {
                double s = 0, sl = 0;
                for (var i = 0; i < y.Length; i++)
                {
                    var w = Math.Exp(c * (ly[i] - maxLog)); // scaled yᶜ, safe from overflow
                    s += w;
                    sl += w * ly[i];
                }
                return 1 / c + meanLog - sl / s;
            }
            var c = fc ?? StatsFitHelpers.PositiveRoot(Profile, 1.0);
            var mean = 0.0;
            foreach (var l in ly) mean += Math.Exp(c * (l - maxLog));
            mean /= y.Length;
            return new WeibullMinDistribution(c, loc, Math.Exp(maxLog + Math.Log(mean) / c));
        }
        return (WeibullMinDistribution)new WeibullMinDistribution(1.0).FitNumerically(data, [fc], floc, fscale);
    }

    /// <inheritdoc />
    public override ContinuousDistribution FitTo(double[] data, double?[]? fixedShapes = null, double? floc = null, double? fscale = null)
        => Fit(data, StatsDistributions.Fixed(fixedShapes, 0), floc, fscale);
}

/// <summary>The Weibull distribution for maxima, scipy's <c>weibull_max</c> (support x ≤ loc).</summary>
public sealed class WeibullMaxDistribution : ContinuousDistribution
{
    /// <summary>Reflected Weibull with shape <paramref name="c"/>.</summary>
    public WeibullMaxDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Positive(c, nameof(c));

    /// <summary>The shape parameter.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "weibull_max";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new WeibullMaxDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Upper0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(C) + Special.Xlogy(C - 1, -z) - Math.Pow(-z, C);
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Exp(-Math.Pow(-z, C));
    /// <inheritdoc />
    protected override double Sf0(double z) => -Special.Expm1(-Math.Pow(-z, C));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => -Math.Pow(-z, C);
    /// <inheritdoc />
    protected override double LogSf0(double z) => Log1mExp(-Math.Pow(-z, C));
    /// <inheritdoc />
    protected override double Ppf0(double q) => -Math.Pow(-Math.Log(q), 1 / C);
    /// <inheritdoc />
    protected override double Isf0(double q) => -Math.Pow(-Special.Log1p(-q), 1 / C);
    /// <inheritdoc />
    protected override double Munp0(int n) => (n % 2 == 0 ? 1 : -1) * Special.Gamma(1 + n / C);
    /// <inheritdoc />
    protected override double Entropy0() => -Math.Log(C) + EulerGamma * (1 - 1 / C) + 1;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => -Math.Pow(StatsRandom.Exponential(rng), 1 / C);
}

/// <summary>The double (two-sided) Weibull distribution, scipy's <c>dweibull</c>.</summary>
public sealed class DWeibullDistribution : ContinuousDistribution
{
    /// <summary>Double Weibull with shape <paramref name="c"/>.</summary>
    public DWeibullDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Positive(c, nameof(c));

    /// <summary>The shape parameter.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "dweibull";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new DWeibullDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) { var a = Math.Abs(z); return Math.Log(C / 2) + Special.Xlogy(C - 1, a) - Math.Pow(a, C); }
    /// <inheritdoc />
    protected override double Cdf0(double z) => z <= 0 ? 0.5 * Math.Exp(-Math.Pow(-z, C)) : 1 - 0.5 * Math.Exp(-Math.Pow(z, C));
    /// <inheritdoc />
    protected override double Sf0(double z) => Cdf0(-z);
    /// <inheritdoc />
    protected override double LogCdf0(double z) => z <= 0 ? -Ln2 - Math.Pow(-z, C) : Special.Log1p(-0.5 * Math.Exp(-Math.Pow(z, C)));
    /// <inheritdoc />
    protected override double LogSf0(double z) => LogCdf0(-z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => q <= 0.5 ? -Math.Pow(-Math.Log(2 * q), 1 / C) : Math.Pow(-Math.Log(2 * (1 - q)), 1 / C);
    /// <inheritdoc />
    protected override double Isf0(double q) => -Ppf0(q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var g2 = Special.Gamma(1 + 2 / C);
        return new(0, g2, 0, Special.Gamma(1 + 4 / C) / (g2 * g2) - 3);
    }
    /// <inheritdoc />
    protected override double Munp0(int n) => n % 2 == 1 ? 0.0 : Special.Gamma(1 + n / C);
    /// <inheritdoc />
    protected override double Entropy0() => -Math.Log(C) + EulerGamma * (1 - 1 / C) + 1 + Ln2;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        var w = Math.Pow(StatsRandom.Exponential(rng), 1 / C);
        return (rng.NextUInt64() & 1) == 0 ? w : -w;
    }
}

/// <summary>The Pareto distribution, scipy's <c>pareto</c> (support x ≥ loc + scale).</summary>
public sealed class ParetoDistribution : ContinuousDistribution
{
    /// <summary>Pareto with tail index <paramref name="b"/>.</summary>
    public ParetoDistribution(double b, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => B = StatsDistributions.Positive(b, nameof(b));

    /// <summary>The tail index.</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "pareto";
    /// <inheritdoc />
    public override string[] ShapeNames => ["b"];
    /// <inheritdoc />
    public override double[] Shapes => [B];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new ParetoDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 1.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => B * Math.Pow(z, -B - 1);
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(B) - (B + 1) * Math.Log(z);
    /// <inheritdoc />
    protected override double Cdf0(double z) => -Special.Expm1(-B * Math.Log(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(-B * Math.Log(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Log1mExp(-B * Math.Log(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => -B * Math.Log(z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Exp(-Special.Log1p(-q) / B);
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Exp(-Math.Log(q) / B);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var b = B;
        var m = b > 1 ? b / (b - 1) : double.PositiveInfinity;
        var v = b > 2 ? b / ((b - 1) * (b - 1) * (b - 2)) : double.PositiveInfinity;
        var s = b > 3 ? 2 * (1 + b) / (b - 3) * Math.Sqrt((b - 2) / b) : double.NaN;
        var k = b > 4 ? 6 * (b * b * b + b * b - 6 * b - 2) / (b * (b - 3) * (b - 4)) : double.NaN;
        return new(m, v, s, k);
    }
    /// <inheritdoc />
    protected override double Munp0(int n) => B > n ? B / (B - n) : double.PositiveInfinity;
    /// <inheritdoc />
    protected override double Entropy0() => 1 / B + 1 - Math.Log(B);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Exp(StatsRandom.Exponential(rng) / B);

    /// <summary>
    /// MLE with the location fixed (closed form, as scipy): scale = min(x) − loc, b = n / Σ log((x − loc)/scale).
    /// Without a fixed location the problem is fitted numerically.
    /// </summary>
    public static ParetoDistribution Fit(double[] data, double? fb = null, double? floc = null, double? fscale = null)
    {
        StatsDistributions.CheckData(data);
        if (floc is double loc)
        {
            var scale = fscale ?? data.Min() - loc;
            if (!(scale > 0)) throw new ArgumentException("All data must exceed floc.", nameof(floc));
            var b = fb ?? data.Length / data.Sum(x => Math.Log((x - loc) / scale));
            return new ParetoDistribution(b, loc, scale);
        }
        return (ParetoDistribution)new ParetoDistribution(1.0).FitNumerically(data, [fb], null, fscale);
    }

    /// <inheritdoc />
    public override ContinuousDistribution FitTo(double[] data, double?[]? fixedShapes = null, double? floc = null, double? fscale = null)
        => Fit(data, StatsDistributions.Fixed(fixedShapes, 0), floc, fscale);
}

/// <summary>The Lomax (Pareto type II) distribution, scipy's <c>lomax</c>.</summary>
public sealed class LomaxDistribution : ContinuousDistribution
{
    /// <summary>Lomax with shape <paramref name="c"/>.</summary>
    public LomaxDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Positive(c, nameof(c));

    /// <summary>The shape parameter.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "lomax";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new LomaxDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(C) - (C + 1) * Special.Log1p(z);
    /// <inheritdoc />
    protected override double Cdf0(double z) => -Special.Expm1(-C * Special.Log1p(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(-C * Special.Log1p(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Log1mExp(-C * Special.Log1p(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => -C * Special.Log1p(z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => Special.Expm1(-Special.Log1p(-q) / C);
    /// <inheritdoc />
    protected override double Isf0(double q) => Special.Expm1(-Math.Log(q) / C);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var c = C;
        var m = c > 1 ? 1 / (c - 1) : double.PositiveInfinity;
        var v = c > 2 ? c / ((c - 1) * (c - 1) * (c - 2)) : double.PositiveInfinity;
        var s = c > 3 ? 2 * (1 + c) / (c - 3) * Math.Sqrt((c - 2) / c) : double.NaN;
        var k = c > 4 ? 6 * (c * c * c + c * c - 6 * c - 2) / (c * (c - 3) * (c - 4)) : double.NaN;
        return new(m, v, s, k);
    }
    /// <inheritdoc />
    protected override double Entropy0() => 1 + 1 / C - Math.Log(C);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Special.Expm1(StatsRandom.Exponential(rng) / C);
}

/// <summary>The triangular distribution on [0, 1] with mode c, scipy's <c>triang</c>.</summary>
public sealed class TriangDistribution : ContinuousDistribution
{
    /// <summary>Triangular with mode at fraction <paramref name="c"/> of the support.</summary>
    public TriangDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.InRange(c, 0, 1, nameof(c));

    /// <summary>The mode, as a fraction of the support.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "triang";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new TriangDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Upper0 => 1.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => z < C ? 2 * z / C : z == C ? 2.0 : 2 * (1 - z) / (1 - C);
    /// <inheritdoc />
    protected override double Cdf0(double z) => z <= C ? (C == 0 ? 0 : z * z / C) : 1 - (1 - z) * (1 - z) / (1 - C);
    /// <inheritdoc />
    protected override double Sf0(double z) => z <= C ? 1 - (C == 0 ? 0 : z * z / C) : (1 - z) * (1 - z) / (1 - C);
    /// <inheritdoc />
    protected override double Ppf0(double q) => q < C ? Math.Sqrt(q * C) : 1 - Math.Sqrt((1 - q) * (1 - C));
    /// <inheritdoc />
    protected override double Isf0(double q) => q <= 1 - C ? 1 - Math.Sqrt(q * (1 - C)) : Math.Sqrt((1 - q) * C);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var c = C;
        var w = 1 - c + c * c;
        return new((1 + c) / 3, w / 18, Math.Sqrt(2) * (2 * c - 1) * (c + 1) * (c - 2) / (5 * Math.Pow(w, 1.5)), -0.6);
    }
    /// <inheritdoc />
    protected override double Entropy0() => 0.5 - Math.Log(2);
}

/// <summary>The normal distribution truncated to [a, b] (standardised bounds), scipy's <c>truncnorm</c>.</summary>
/// <remarks>
/// Everything is done in log space with the tail on the side the interval lies in: for a &gt; 0
/// the mass is Φ̄(a) − Φ̄(b), not Φ(b) − Φ(a), which would be 1 − 1 = 0 for a = 40. Quantiles invert
/// log Φ with <c>ndtri_exp</c>, so truncnorm(40, 41) has the correct distribution rather than a
/// spike at the bound — the same design as scipy's log-space algorithm.
/// </remarks>
public sealed class TruncNormDistribution : ContinuousDistribution
{
    private readonly double _logZ;
    private readonly bool _flip;

    /// <summary>Normal truncated to [a, b] in standardised units (a &lt; b; either may be infinite).</summary>
    public TruncNormDistribution(double a, double b, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        if (double.IsNaN(a) || double.IsNaN(b) || !(a < b)) throw new ArgumentException("Require a < b.", nameof(a));
        A = a;
        B = b;
        _flip = a > 0; // the interval sits in the right tail: work with upper tails
        _logZ = LogDelta(a, b);
    }

    /// <summary>Lower truncation point (standardised).</summary>
    public double A { get; }
    /// <summary>Upper truncation point (standardised).</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "truncnorm";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a", "b"];
    /// <inheritdoc />
    public override double[] Shapes => [A, B];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new TruncNormDistribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => A;
    /// <inheritdoc />
    protected override double Upper0 => B;

    /// <summary>log(Φ(hi) − Φ(lo)), taking the difference of whichever tails are small.</summary>
    private static double LogDelta(double lo, double hi)
    {
        if (lo > 0) return LogSubExp(Special.LogNdtr(-lo), Special.LogNdtr(-hi));
        if (hi < 0) return LogSubExp(Special.LogNdtr(hi), Special.LogNdtr(lo));
        // The interval straddles 0: 1 − Φ(lo) − Φ(−hi) loses nothing.
        return Special.Log1p(-Phi(lo) - Phi(-hi));
    }

    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => LogPhi(z) - _logZ;
    /// <inheritdoc />
    protected override double LogCdf0(double z)
    {
        var l = Math.Min(0, LogDelta(A, z) - _logZ);
        // Near 1 the difference of two nearly equal logs has no digits left; use the other tail.
        return l > -Ln2 ? Special.Log1p(-Math.Exp(Math.Min(0, LogDelta(z, B) - _logZ))) : l;
    }
    /// <inheritdoc />
    protected override double LogSf0(double z)
    {
        var l = Math.Min(0, LogDelta(z, B) - _logZ);
        return l > -Ln2 ? Special.Log1p(-Math.Exp(Math.Min(0, LogDelta(A, z) - _logZ))) : l;
    }
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Exp(LogCdf0(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(LogSf0(z));

    /// <summary>Solves for z given P(Z ≤ z) = p and P(Z &gt; z) = 1 − p = s, using whichever is smaller.</summary>
    private double Solve(double logP, double logS, bool useP)
    {
        double x;
        if (!_flip)
        {
            // Lower-tail form: log Φ(z) = log(Φ(a) + p·Z), or upper: log Φ̄(z) = log(Φ̄(b) + s·Z).
            x = useP
                ? NdtriExp(LogAddExp(Special.LogNdtr(A), logP + _logZ))
                : -NdtriExp(LogAddExp(Special.LogNdtr(-B), logS + _logZ));
        }
        else
        {
            // Right-tail interval: Φ̄(z) = Φ̄(a) − p·Z or Φ̄(z) = Φ̄(b) + s·Z.
            x = useP
                ? -NdtriExp(LogSubExp(Special.LogNdtr(-A), logP + _logZ))
                : -NdtriExp(LogAddExp(Special.LogNdtr(-B), logS + _logZ));
        }
        return Math.Clamp(x, A, B);
    }

    /// <inheritdoc />
    protected override double Ppf0(double q)
    {
        // When the interval straddles 0 the lower-tail form is accurate for q ≤ ½; in the right
        // tail the Φ̄(a) − pZ subtraction is benign for q ≤ ½ as well (it removes at most half).
        return q <= 0.5 ? Solve(Math.Log(q), Special.Log1p(-q), true) : Solve(Special.Log1p(-q), Special.Log1p(-q), false);
    }

    /// <inheritdoc />
    protected override double Isf0(double q)
        => q <= 0.5 ? Solve(Special.Log1p(-q), Math.Log(q), false) : Solve(Special.Log1p(-q), 0, true);

    /// <summary>
    /// φ(x)/Z without large logarithms: Z is written as the tail mass at the reference end r times a
    /// fraction, and the tail mass itself through erfcx, so φ(x)/Z = e^{−(x−r)(x+r)/2}/(√2π·erfcx-term·fraction).
    /// Going through log φ − log Z instead costs ~1e-13 relative at |x| = 38, which the variance of
    /// a far-tail interval then amplifies a thousandfold.
    /// </summary>
    private double PhiRatio(double x)
    {
        if (double.IsInfinity(x)) return 0.0;
        double r, tail, frac;
        if (A > 0)
        {
            r = A;
            tail = 0.5 * Special.Erfcx(A / Sqrt2);
            frac = -Special.Expm1(Special.LogNdtr(-B) - Special.LogNdtr(-A));
        }
        else if (B < 0)
        {
            r = B;
            tail = 0.5 * Special.Erfcx(-B / Sqrt2);
            frac = -Special.Expm1(Special.LogNdtr(A) - Special.LogNdtr(B));
        }
        else return Math.Exp(LogPhi(x) - _logZ);
        return Math.Exp(-0.5 * (x - r) * (x + r)) / (Sqrt2Pi * tail * frac);
    }

    private double PowPhi(double x, int k) => double.IsInfinity(x) ? 0.0 : Math.Pow(x, k) * PhiRatio(x);

    /// <summary>Boundary term ((a−s)^k φ(a) − (b−s)^k φ(b))/Z of the shifted moment recursion.</summary>
    private double Boundary(double s, int k)
    {
        var ta = double.IsInfinity(A) ? 0.0 : Math.Pow(A - s, k) * PhiRatio(A);
        var tb = double.IsInfinity(B) ? 0.0 : Math.Pow(B - s, k) * PhiRatio(B);
        return ta - tb;
    }

    /// <summary>
    /// E[(Z − s)^k] for k = 0..n by the recursion I_{k+1} = k I_{k−1} − s I_k + boundary_k, which
    /// follows from φ'(z) = −zφ(z). Taken about the mean it avoids the catastrophic cancellation of
    /// raw moments when the interval sits far in a tail: truncnorm(−40, −38) has mean −38.03 and
    /// variance 7e-4, and raw moments lose six digits of the variance before the kurtosis is touched.
    /// </summary>
    private double[] ShiftedMoments(double s, int n)
    {
        var I = new double[n + 1];
        I[0] = 1;
        if (n >= 1) I[1] = -s + Boundary(s, 0);
        for (var k = 1; k < n; k++) I[k + 1] = k * I[k - 1] - s * I[k] + Boundary(s, k);
        return I;
    }

    /// <inheritdoc />
    protected override double Munp0(int n) => ShiftedMoments(0, n)[n];

    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var mean = PhiRatio(A) - PhiRatio(B);
        if (A > 3 || B < -3)
        {
            // Far in a tail even the shifted recursion loses ~3 digits per order (the boundary terms
            // nearly cancel s·I_k), so the third and fourth central moments come from quadrature of
            // the smooth density instead. The variance stays on the recursion, which is exact enough.
            var hi = double.IsInfinity(B) ? A + 40 : B;
            var lo = double.IsInfinity(A) ? B - 40 : A;
            double Central(int k) => StatsDistMath.Quad(z => Math.Pow(z - mean, k) * PhiRatio(z), lo, hi, 0, 1e-13);
            var c2 = Central(2);
            return new(mean, c2, Central(3) / Math.Pow(c2, 1.5), Central(4) / (c2 * c2) - 3);
        }
        var I = ShiftedMoments(mean, 4);
        // I[1] is the rounding residue of the mean; fold it in so the central moments are exact.
        var d = I[1];
        var v = I[2] - d * d;
        var mu3 = I[3] - 3 * d * I[2] + 2 * d * d * d;
        var mu4 = I[4] - 4 * d * I[3] + 6 * d * d * I[2] - 3 * d * d * d * d;
        return new(mean + d, v, mu3 / Math.Pow(v, 1.5), mu4 / (v * v) - 3);
    }
    /// <inheritdoc />
    protected override double Entropy0() => LnSqrt2Pi + 0.5 + _logZ + 0.5 * (PowPhi(A, 1) - PowPhi(B, 1));
}

/// <summary>The exponential truncated to [0, b], scipy's <c>truncexpon</c>.</summary>
public sealed class TruncExponDistribution : ContinuousDistribution
{
    /// <summary>Exponential truncated at <paramref name="b"/>.</summary>
    public TruncExponDistribution(double b, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => B = StatsDistributions.Positive(b, nameof(b));

    /// <summary>The truncation point.</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "truncexpon";
    /// <inheritdoc />
    public override string[] ShapeNames => ["b"];
    /// <inheritdoc />
    public override double[] Shapes => [B];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new TruncExponDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Upper0 => B;
    private double Mass => -Special.Expm1(-B);
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(-z) / Mass;
    /// <inheritdoc />
    protected override double LogPdf0(double z) => -z - Math.Log(Mass);
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.Expm1(-z) / Special.Expm1(-B);
    /// <inheritdoc />
    protected override double Sf0(double z) => -Math.Exp(-z) * Special.Expm1(z - B) / Mass;
    /// <inheritdoc />
    protected override double Ppf0(double q) => -Special.Log1p(q * Special.Expm1(-B));
    /// <inheritdoc />
    protected override double Isf0(double q) => -Math.Log(Math.Exp(-B) + q * Mass);
    /// <inheritdoc />
    protected override double Munp0(int n) => Special.Gamma(n + 1) * Special.GammaInc(n + 1, B) / Mass;
    /// <inheritdoc />
    protected override double Entropy0()
    {
        var eB = Math.Exp(B);
        return Math.Log(Special.Expm1(B)) + (1 + eB * (B - 1)) / (1 - eB);
    }
}

/// <summary>The generalised extreme value distribution, scipy's <c>genextreme</c> (scipy's sign convention for c).</summary>
public sealed class GenExtremeDistribution : ContinuousDistribution
{
    /// <summary>GEV with shape <paramref name="c"/> (c &gt; 0: bounded above; c &lt; 0: heavy right tail; 0: Gumbel).</summary>
    public GenExtremeDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Finite(c, nameof(c));

    /// <summary>The shape parameter.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "genextreme";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new GenExtremeDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => C < 0 ? 1 / C : double.NegativeInfinity;
    /// <inheritdoc />
    protected override double Upper0 => C > 0 ? 1 / C : double.PositiveInfinity;

    /// <summary>log(−log F(z)).</summary>
    private double LogLogCdf(double z) => C == 0 ? -z : Special.Log1p(-C * z) / C;

    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
    {
        var cx = C * z;
        if (cx == 1) return C == 1 ? 0.0 : double.NegativeInfinity;
        var l = LogLogCdf(z);
        return -Math.Exp(l) + l - Special.Log1p(-cx);
    }
    /// <inheritdoc />
    protected override double LogCdf0(double z) => -Math.Exp(LogLogCdf(z));
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Exp(LogCdf0(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => -Special.Expm1(LogCdf0(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => Log1mExp(LogCdf0(z));
    private double FromGumbel(double x) => C == 0 ? x : -Special.Expm1(-C * x) / C;
    /// <inheritdoc />
    protected override double Ppf0(double q) => FromGumbel(-Math.Log(-Math.Log(q)));
    /// <inheritdoc />
    protected override double Isf0(double q) => FromGumbel(-Math.Log(-Special.Log1p(-q)));
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        if (Math.Abs(C) < 1e-8) return base.Munp0(n);
        if (n * C <= -1) return double.PositiveInfinity;
        // Z = (1 − Wᶜ)/c with W ~ Exp(1), E[W^{kc}] = Γ(1 + kc).
        var s = 0.0;
        for (var k = 0; k <= n; k++) s += Special.Binom(n, k) * (k % 2 == 0 ? 1 : -1) * Special.Gamma(1 + k * C);
        return s / Math.Pow(C, n);
    }
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var c = C;
        if (Math.Abs(c) < 1e-5)
        {
            var g = new GumbelRDistribution();
            return g.Stats();
        }
        double g1 = Special.Gamma(1 + c), g2 = Special.Gamma(1 + 2 * c), g3 = Special.Gamma(1 + 3 * c), g4 = Special.Gamma(1 + 4 * c);
        var mean = c > -1 ? (1 - g1) / c : double.PositiveInfinity;
        var v = c > -0.5 ? (g2 - g1 * g1) / (c * c) : double.PositiveInfinity;
        var sk = c > -1.0 / 3 ? Math.Sign(c) * (-g3 + 3 * g1 * g2 - 2 * g1 * g1 * g1) / Math.Pow(g2 - g1 * g1, 1.5) : double.NaN;
        var ku = c > -0.25 ? (g4 - 4 * g3 * g1 - 3 * g2 * g2 + 12 * g2 * g1 * g1 - 6 * g1 * g1 * g1 * g1) / ((g2 - g1 * g1) * (g2 - g1 * g1)) : double.NaN;
        return new(mean, v, sk, ku);
    }
    /// <inheritdoc />
    protected override double Entropy0() => EulerGamma * (1 - C) + 1;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => FromGumbel(-Math.Log(StatsRandom.Exponential(rng)));
}

/// <summary>The generalised Pareto distribution, scipy's <c>genpareto</c>.</summary>
public sealed class GenParetoDistribution : ContinuousDistribution
{
    /// <summary>Generalised Pareto with shape <paramref name="c"/> (c &lt; 0 bounds the support at −1/c).</summary>
    public GenParetoDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Finite(c, nameof(c));

    /// <summary>The shape parameter.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "genpareto";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new GenParetoDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Upper0 => C < 0 ? -1 / C : double.PositiveInfinity;
    /// <summary>log(1 + c z)/c, the negative log survival function (z when c = 0).</summary>
    private double L(double z) => C == 0 ? z : Special.Log1p(C * z) / C;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
    {
        if (C < 0 && z == -1 / C) return C < -1 ? double.PositiveInfinity : C == -1 ? 0.0 : double.NegativeInfinity;
        return C == 0 ? -z : -(C + 1) * Special.Log1p(C * z) / C;
    }
    /// <inheritdoc />
    protected override double Cdf0(double z) => -Special.Expm1(-L(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(-L(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Log1mExp(-L(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => -L(z);
    private double FromExp(double e) => C == 0 ? e : Special.Expm1(C * e) / C;
    /// <inheritdoc />
    protected override double Ppf0(double q) => FromExp(-Special.Log1p(-q));
    /// <inheritdoc />
    protected override double Isf0(double q) => FromExp(-Math.Log(q));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var c = C;
        var m = c < 1 ? 1 / (1 - c) : double.PositiveInfinity;
        var v = c < 0.5 ? 1 / ((1 - c) * (1 - c) * (1 - 2 * c)) : double.PositiveInfinity;
        var s = c < 1.0 / 3 ? 2 * (1 + c) * Math.Sqrt(1 - 2 * c) / (1 - 3 * c) : double.NaN;
        var k = c < 0.25 ? 3 * (1 - 2 * c) * (2 * c * c + c + 3) / ((1 - 3 * c) * (1 - 4 * c)) - 3 : double.NaN;
        return new(m, v, s, k);
    }
    /// <inheritdoc />
    protected override double Entropy0() => C + 1;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => FromExp(StatsRandom.Exponential(rng));
}

/// <summary>The skew-normal distribution, scipy's <c>skewnorm</c>.</summary>
/// <remarks>The CDF is Φ(z) − 2T(z, a) with Owen's T. That difference cancels catastrophically in the
/// tail opposite the skew, so — exactly as scipy does — once it drops below 1e-6 the tail is
/// integrated from the density instead.</remarks>
public sealed class SkewNormDistribution : ContinuousDistribution
{
    /// <summary>Skew-normal with shape <paramref name="a"/>.</summary>
    public SkewNormDistribution(double a, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => A = StatsDistributions.Finite(a, nameof(a));

    /// <summary>The shape (skewness) parameter.</summary>
    public double A { get; }

    /// <inheritdoc />
    public override string Name => "skewnorm";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a"];
    /// <inheritdoc />
    public override double[] Shapes => [A];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new SkewNormDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Ln2 + LogPhi(z) + Special.LogNdtr(A * z);

    private static double LowerTail(double a, double z)
    {
        // P(Z ≤ z) for skewnorm(a); accurate when a ≤ 0 or the result is not tiny.
        var c = Special.Ndtr(z) - 2 * Special.OwensT(z, a);
        if (a > 0 && c < 1e-6)
            c = StatsDistMath.Quad(t => Math.Exp(Ln2 + LogPhi(t) + Special.LogNdtr(a * t)), double.NegativeInfinity, z, 0, 1e-13);
        return Math.Clamp(c, 0, 1);
    }
    /// <inheritdoc />
    protected override double Cdf0(double z) => LowerTail(A, z);
    /// <inheritdoc />
    protected override double Sf0(double z) => LowerTail(-A, -z);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var delta = A / Math.Sqrt(1 + A * A);
        var mean = delta * Math.Sqrt(2 / Math.PI);
        var v = 1 - mean * mean;
        var skew = (4 - Math.PI) / 2 * mean * mean * mean / Math.Pow(v, 1.5);
        var kurt = 2 * (Math.PI - 3) * mean * mean * mean * mean / (v * v);
        return new(mean, v, skew, kurt);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        var delta = A / Math.Sqrt(1 + A * A);
        var u0 = StatsRandom.Normal(rng);
        var v = StatsRandom.Normal(rng);
        return delta * Math.Abs(u0) + Math.Sqrt(1 - delta * delta) * v;
    }
}

/// <summary>The power-function distribution on [0, 1], scipy's <c>powerlaw</c>.</summary>
public sealed class PowerLawDistribution : ContinuousDistribution
{
    /// <summary>Power law with exponent <paramref name="a"/> (pdf a·x^(a−1)).</summary>
    public PowerLawDistribution(double a, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => A = StatsDistributions.Positive(a, nameof(a));

    /// <summary>The shape parameter.</summary>
    public double A { get; }

    /// <inheritdoc />
    public override string Name => "powerlaw";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a"];
    /// <inheritdoc />
    public override double[] Shapes => [A];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new PowerLawDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Upper0 => 1.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(A) + Special.Xlogy(A - 1, z);
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Pow(z, A);
    /// <inheritdoc />
    protected override double Sf0(double z) => -Special.Expm1(A * Math.Log(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => A * Math.Log(z);
    /// <inheritdoc />
    protected override double LogSf0(double z) => Log1mExp(A * Math.Log(z));
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Pow(q, 1 / A);
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Exp(Special.Log1p(-q) / A);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var a = A;
        return new(a / (a + 1), a / ((a + 2) * (a + 1) * (a + 1)),
            -2 * ((a - 1) / (a + 3)) * Math.Sqrt((a + 2) / a),
            6 * (a * a * a - a * a - 6 * a + 2) / (a * (a + 3) * (a + 4)));
    }
    /// <inheritdoc />
    protected override double Munp0(int n) => A / (A + n);
    /// <inheritdoc />
    protected override double Entropy0() => 1 - 1 / A - Math.Log(A);
}

/// <summary>The log-Laplace distribution, scipy's <c>loglaplace</c>.</summary>
public sealed class LogLaplaceDistribution : ContinuousDistribution
{
    /// <summary>Log-Laplace with shape <paramref name="c"/>.</summary>
    public LogLaplaceDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Positive(c, nameof(c));

    /// <summary>The shape parameter.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "loglaplace";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new LogLaplaceDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(C / 2) + (z < 1 ? Special.Xlogy(C - 1, z) : (-C - 1) * Math.Log(z));
    /// <inheritdoc />
    protected override double Cdf0(double z) => z < 1 ? 0.5 * Math.Pow(z, C) : 1 - 0.5 * Math.Pow(z, -C);
    /// <inheritdoc />
    protected override double Sf0(double z) => z < 1 ? 1 - 0.5 * Math.Pow(z, C) : 0.5 * Math.Pow(z, -C);
    /// <inheritdoc />
    protected override double LogCdf0(double z) => z < 1 ? C * Math.Log(z) - Ln2 : Special.Log1p(-0.5 * Math.Pow(z, -C));
    /// <inheritdoc />
    protected override double LogSf0(double z) => z < 1 ? Special.Log1p(-0.5 * Math.Pow(z, C)) : -C * Math.Log(z) - Ln2;
    /// <inheritdoc />
    protected override double Ppf0(double q) => q < 0.5 ? Math.Pow(2 * q, 1 / C) : Math.Pow(2 * (1 - q), -1 / C);
    /// <inheritdoc />
    protected override double Isf0(double q) => q < 0.5 ? Math.Pow(2 * q, -1 / C) : Math.Pow(2 * (1 - q), 1 / C);
    /// <inheritdoc />
    protected override double Munp0(int n) => C > n ? C * C / (C * C - n * n) : double.PositiveInfinity;
    /// <inheritdoc />
    protected override double Entropy0() => Math.Log(2 / C) + 1;
}

/// <summary>The Burr type XII distribution, scipy's <c>burr12</c>.</summary>
public sealed class Burr12Distribution : ContinuousDistribution
{
    /// <summary>Burr XII with shapes <paramref name="c"/>, <paramref name="d"/>.</summary>
    public Burr12Distribution(double c, double d, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        C = StatsDistributions.Positive(c, nameof(c));
        D = StatsDistributions.Positive(d, nameof(d));
    }

    /// <summary>First shape parameter.</summary>
    public double C { get; }
    /// <summary>Second shape parameter.</summary>
    public double D { get; }

    /// <inheritdoc />
    public override string Name => "burr12";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c", "d"];
    /// <inheritdoc />
    public override double[] Shapes => [C, D];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new Burr12Distribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(C * D) + Special.Xlogy(C - 1, z) - (D + 1) * Special.Log1p(Math.Pow(z, C));
    /// <inheritdoc />
    protected override double LogSf0(double z) => -D * Special.Log1p(Math.Pow(z, C));
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(LogSf0(z));
    /// <inheritdoc />
    protected override double Cdf0(double z) => -Special.Expm1(LogSf0(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Log1mExp(LogSf0(z));
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Pow(Special.Expm1(-Special.Log1p(-q) / D), 1 / C);
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Pow(Special.Expm1(-Math.Log(q) / D), 1 / C);
    /// <inheritdoc />
    protected override double Munp0(int n)
        => C * D > n ? D * Math.Exp(Special.BetaLn(D - n / C, 1 + n / C)) : double.PositiveInfinity;
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var d = base.Moments0();
        var cd = C * D;
        return new(cd > 1 ? d.Mean : double.PositiveInfinity, cd > 2 ? d.Variance : double.PositiveInfinity,
            cd > 3 ? d.Skewness : double.NaN, cd > 4 ? d.Kurtosis : double.NaN);
    }
}

/// <summary>The exponentially modified normal distribution, scipy's <c>exponnorm</c>.</summary>
public sealed class ExponNormDistribution : ContinuousDistribution
{
    /// <summary>Exponentially modified normal with shape <paramref name="k"/> (exponential mean / σ).</summary>
    public ExponNormDistribution(double k, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => K = StatsDistributions.Positive(k, nameof(k));

    /// <summary>The shape parameter K.</summary>
    public double K { get; }

    /// <inheritdoc />
    public override string Name => "exponnorm";
    /// <inheritdoc />
    public override string[] ShapeNames => ["K"];
    /// <inheritdoc />
    public override double[] Shapes => [K];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new ExponNormDistribution(shapes[0], loc, scale);
    private double LogProd(double z) { var ik = 1 / K; return ik * (0.5 * ik - z) + Special.LogNdtr(z - ik); }
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => LogProd(z) - Math.Log(K);
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.Ndtr(z) - Math.Exp(LogProd(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.Ndtr(-z) + Math.Exp(LogProd(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => LogAddExp(Special.LogNdtr(-z), LogProd(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => z < 0 ? LogSubExp(Special.LogNdtr(z), LogProd(z)) : Special.Log1p(-Sf0(z));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var k2 = K * K;
        return new(K, 1 + k2, 2 * K * k2 * Math.Pow(1 + k2, -1.5), 6 * k2 * k2 / ((1 + k2) * (1 + k2)));
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => StatsRandom.Normal(rng) + K * StatsRandom.Exponential(rng);
}

/// <summary>The log-uniform (reciprocal) distribution on [a, b], scipy's <c>loguniform</c> / <c>reciprocal</c>.</summary>
public sealed class LogUniformDistribution : ContinuousDistribution
{
    /// <summary>Log-uniform on [a, b], 0 &lt; a &lt; b.</summary>
    public LogUniformDistribution(double a, double b, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        A = StatsDistributions.Positive(a, nameof(a));
        B = StatsDistributions.Positive(b, nameof(b));
        if (!(b > a)) throw new ArgumentException("Require a < b.", nameof(b));
    }

    /// <summary>Lower bound.</summary>
    public double A { get; }
    /// <summary>Upper bound.</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "loguniform";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a", "b"];
    /// <inheritdoc />
    public override double[] Shapes => [A, B];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new LogUniformDistribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => A;
    /// <inheritdoc />
    protected override double Upper0 => B;
    private double LogRatio => Math.Log(B / A);
    /// <inheritdoc />
    protected override double Pdf0(double z) => 1 / (z * LogRatio);
    /// <inheritdoc />
    protected override double LogPdf0(double z) => -Math.Log(z) - Math.Log(LogRatio);
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Log(z / A) / LogRatio;
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Log(B / z) / LogRatio;
    /// <inheritdoc />
    protected override double Ppf0(double q) => A * Math.Exp(q * LogRatio);
    /// <inheritdoc />
    protected override double Isf0(double q) => B * Math.Exp(-q * LogRatio);
    /// <inheritdoc />
    protected override double Munp0(int n) => (Math.Pow(B, n) - Math.Pow(A, n)) / (n * LogRatio);
    /// <inheritdoc />
    protected override double Entropy0() => 0.5 * Math.Log(A * B) + Math.Log(LogRatio);
}

/// <summary>Johnson's S_U distribution, scipy's <c>johnsonsu</c>.</summary>
public sealed class JohnsonSUDistribution : ContinuousDistribution
{
    /// <summary>Johnson S_U with shapes <paramref name="a"/>, <paramref name="b"/> (b &gt; 0).</summary>
    public JohnsonSUDistribution(double a, double b, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        A = StatsDistributions.Finite(a, nameof(a));
        B = StatsDistributions.Positive(b, nameof(b));
    }

    /// <summary>First shape parameter.</summary>
    public double A { get; }
    /// <summary>Second shape parameter.</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "johnsonsu";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a", "b"];
    /// <inheritdoc />
    public override double[] Shapes => [A, B];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new JohnsonSUDistribution(shapes[0], shapes[1], loc, scale);
    private double T(double z) => A + B * Math.Asinh(z);
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(B) - 0.5 * Math.Log(z * z + 1) + LogPhi(T(z));
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.Ndtr(T(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.Ndtr(-T(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Special.LogNdtr(T(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => Special.LogNdtr(-T(z));
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Sinh((Special.Ndtri(q) - A) / B);
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Sinh((-Special.Ndtri(q) - A) / B);
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        // E[sinh^n Y], Y = (N − a)/b, expanded into E[e^{kY}] = exp(−k a/b + k²/(2b²)).
        var s = 0.0;
        for (var j = 0; j <= n; j++)
        {
            var k = n - 2 * j;
            s += Special.Binom(n, j) * (j % 2 == 0 ? 1 : -1) * Math.Exp(-k * A / B + k * k / (2 * B * B));
        }
        return s / Math.Pow(2, n);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Sinh((StatsRandom.Normal(rng) - A) / B);
}

/// <summary>Johnson's S_B distribution on [0, 1], scipy's <c>johnsonsb</c>.</summary>
public sealed class JohnsonSBDistribution : ContinuousDistribution
{
    /// <summary>Johnson S_B with shapes <paramref name="a"/>, <paramref name="b"/> (b &gt; 0).</summary>
    public JohnsonSBDistribution(double a, double b, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        A = StatsDistributions.Finite(a, nameof(a));
        B = StatsDistributions.Positive(b, nameof(b));
    }

    /// <summary>First shape parameter.</summary>
    public double A { get; }
    /// <summary>Second shape parameter.</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "johnsonsb";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a", "b"];
    /// <inheritdoc />
    public override double[] Shapes => [A, B];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new JohnsonSBDistribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Upper0 => 1.0;
    private double T(double z) => A + B * (Math.Log(z) - Special.Log1p(-z));
    /// <inheritdoc />
    protected override double Pdf0(double z) => z <= 0 || z >= 1 ? 0.0 : Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
        => z <= 0 || z >= 1 ? double.NegativeInfinity : Math.Log(B) - Math.Log(z) - Special.Log1p(-z) + LogPhi(T(z));
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.Ndtr(T(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.Ndtr(-T(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Special.LogNdtr(T(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => Special.LogNdtr(-T(z));
    /// <inheritdoc />
    protected override double Ppf0(double q) => Special.Expit((Special.Ndtri(q) - A) / B);
    /// <inheritdoc />
    protected override double Isf0(double q) => Special.Expit((-Special.Ndtri(q) - A) / B);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Special.Expit((StatsRandom.Normal(rng) - A) / B);
}

/// <summary>The folded normal distribution |N(c, 1)|, scipy's <c>foldnorm</c>.</summary>
public sealed class FoldNormDistribution : ContinuousDistribution
{
    /// <summary>Folded normal with shape <paramref name="c"/> ≥ 0.</summary>
    public FoldNormDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        if (!(c >= 0) || double.IsInfinity(c)) throw new ArgumentException("c must be non-negative and finite.", nameof(c));
        C = c;
    }

    /// <summary>The shape parameter.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "foldnorm";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new FoldNormDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => (Math.Exp(-0.5 * (z - C) * (z - C)) + Math.Exp(-0.5 * (z + C) * (z + C))) / Sqrt2Pi;
    /// <inheritdoc />
    protected override double LogPdf0(double z) => LogAddExp(LogPhi(z - C), LogPhi(z + C));
    /// <inheritdoc />
    protected override double Cdf0(double z)
    {
        // Φ(z − c) − Φ(−z − c) cancels for small z; integrate the smooth density there instead.
        if (z < 0.5) return StatsDistMath.Quad(Pdf0, 0, z, 0, 2e-14);
        return Special.Ndtr(z - C) - Special.Ndtr(-z - C);
    }
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.Ndtr(C - z) + Special.Ndtr(-z - C);
    /// <inheritdoc />
    protected override double LogSf0(double z) => LogAddExp(Special.LogNdtr(C - z), Special.LogNdtr(-z - C));
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        var c = C;
        var e = 1 - 2 * Special.Ndtr(-c);
        var phi = Math.Exp(-0.5 * c * c) / Sqrt2Pi;
        return n switch
        {
            1 => c * e + 2 * phi,
            2 => c * c + 1,
            3 => (c * c * c + 3 * c) * e + 2 * (c * c + 2) * phi,
            4 => c * c * c * c + 6 * c * c + 3,
            _ => base.Munp0(n),
        };
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Abs(StatsRandom.Normal(rng) + C);
}

/// <summary>The Bradford distribution on [0, 1], scipy's <c>bradford</c>.</summary>
public sealed class BradfordDistribution : ContinuousDistribution
{
    /// <summary>Bradford with shape <paramref name="c"/>.</summary>
    public BradfordDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Positive(c, nameof(c));

    /// <summary>The shape parameter.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "bradford";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new BradfordDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Upper0 => 1.0;
    private double K => Special.Log1p(C);
    /// <inheritdoc />
    protected override double Pdf0(double z) => C / (K * (1 + C * z));
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.Log1p(C * z) / K;
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.Log1p(C * (1 - z) / (1 + C * z)) / K;
    /// <inheritdoc />
    protected override double Ppf0(double q) => Special.Expm1(q * K) / C;
    /// <inheritdoc />
    protected override double Isf0(double q) => Special.Expm1((1 - q) * K) / C;
    /// <inheritdoc />
    protected override double Entropy0() => K / 2 - Math.Log(C / K);
}

/// <summary>The generalised logistic (type I) distribution, scipy's <c>genlogistic</c>.</summary>
public sealed class GenLogisticDistribution : ContinuousDistribution
{
    /// <summary>Generalised logistic with shape <paramref name="c"/>.</summary>
    public GenLogisticDistribution(double c, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => C = StatsDistributions.Positive(c, nameof(c));

    /// <summary>The shape parameter.</summary>
    public double C { get; }

    /// <inheritdoc />
    public override string Name => "genlogistic";
    /// <inheritdoc />
    public override string[] ShapeNames => ["c"];
    /// <inheritdoc />
    public override double[] Shapes => [C];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new GenLogisticDistribution(shapes[0], loc, scale);
    /// <summary>log(1 + e^{−z}) without overflow.</summary>
    private static double Softplus(double z) => z > 0 ? Special.Log1p(Math.Exp(-z)) : -z + Special.Log1p(Math.Exp(z));
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(C) - z - (C + 1) * Softplus(z);
    /// <inheritdoc />
    protected override double LogCdf0(double z) => -C * Softplus(z);
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Exp(LogCdf0(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => -Special.Expm1(LogCdf0(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => Log1mExp(LogCdf0(z));
    /// <inheritdoc />
    protected override double Ppf0(double q) => NegLogExpm1(-Math.Log(q) / C);
    /// <inheritdoc />
    protected override double Isf0(double q) => NegLogExpm1(-Special.Log1p(-q) / C);
    /// <summary>−log(e^t − 1), written as −t − log1p(−e^{−t}) once e^t would overflow.</summary>
    private static double NegLogExpm1(double t) => t > 30 ? -t - Special.Log1p(-Math.Exp(-t)) : -Math.Log(Special.Expm1(t));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var c = C;
        var mu = Special.Digamma(c) - Special.Digamma(1);
        var v = Math.PI * Math.PI / 6 + Special.Polygamma(1, c);
        var g1 = (Special.Polygamma(2, c) + 2 * 1.2020569031595942854) / Math.Pow(v, 1.5);
        var g2 = (Math.Pow(Math.PI, 4) / 15 + Special.Polygamma(3, c)) / (v * v);
        return new(mu, v, g1, g2);
    }
}
