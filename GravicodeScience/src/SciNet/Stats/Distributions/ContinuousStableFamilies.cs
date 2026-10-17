using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.StatsDistMath;

namespace Gravicode.Science.SciNet;

/// <summary>Parameterisation of <see cref="LevyStableDistribution"/>, as scipy's <c>levy_stable.parameterization</c>.</summary>
public enum LevyStableParameterization
{
    /// <summary>Nolan's S0: continuous in α and β; the location is the mode region's centre. Not scipy's default.</summary>
    S0,
    /// <summary>Nolan's S1 (Samorodnitsky–Taqqu): the location is the mean when α &gt; 1. scipy's default.</summary>
    S1,
}

/// <summary>
/// Densities and distribution functions of the standard stable law in Nolan's S0 form by his
/// integral representations (Nolan 1997, "Numerical calculation of stable densities and distribution
/// functions"), the method scipy's <c>levy_stable</c> uses by default.
/// </summary>
/// <remarks>
/// With ζ = −β tan(πα/2) and ξ = arctan(−ζ)/α, for x &gt; ζ the density is c₂∫ g e^(−g) dθ and the
/// CDF c₁ + c₃∫ e^(−g) dθ over θ ∈ (−ξ, π/2), g monotone in θ. g is evaluated in log space and the
/// quadrature is split where log g crosses a ladder of levels (found by bisection), which is where
/// the integrand's mass sits however far into a tail x is. The tail that would be 1 − (something
/// close to 1) is integrated directly as ∫(1 − e^(−g)), so both tails keep relative accuracy.
/// </remarks>
internal static class StatsStableMath
{
    private static readonly double[] Levels = [-30, -12, -5, -2, -1, 0, 1, 2, 3, 4.6, 7];

    /// <summary>log g(θ) for α ≠ 1, x0 &gt; ζ.</summary>
    private static double LogG(double theta, double alpha, double xi, double logDx)
    {
        var e = alpha / (alpha - 1);
        var cosT = Math.Sin(Math.PI / 2 - theta);
        var sinA = Math.Sin(alpha * (xi + theta));
        return e * logDx + Math.Log(Math.Cos(alpha * xi)) / (alpha - 1) + e * (Math.Log(cosT) - Math.Log(sinA))
               + Math.Log(Math.Cos(alpha * xi + (alpha - 1) * theta)) - Math.Log(cosT);
    }

    /// <summary>log g(θ) for α = 1, β &gt; 0.</summary>
    private static double LogG1(double theta, double beta, double x0)
    {
        var cosT = Math.Sin(Math.PI / 2 - theta);
        var w = Math.PI / 2 + beta * theta;
        return -Math.PI * x0 / (2 * beta) + Math.Log(2 / Math.PI) + Math.Log(w / cosT) + w * Math.Tan(theta) / beta;
    }

    /// <summary>The quadrature break points: where log g crosses each level.</summary>
    private static List<double> Breaks(Func<double, double> logg, double lo, double hi)
    {
        var pts = new List<double>();
        var span = hi - lo;
        var a = logg(lo + 1e-12 * span);
        var b = logg(hi - 1e-12 * span);
        var increasing = !(a > b);
        foreach (var level in Levels)
        {
            double l = lo, h = hi;
            for (var i = 0; i < 80; i++)
            {
                var m = 0.5 * (l + h);
                if (m <= l || m >= h) break;
                var v = logg(m);
                var below = double.IsNaN(v) ? !increasing : v < level;
                if (below == increasing) l = m; else h = m;
            }
            var p = 0.5 * (l + h);
            // A level the integrand never reaches converges onto an end point; a cut a few ulps from
            // the end makes a degenerate sub-interval that QUADPACK integrates badly (1e-4 errors seen).
            if (p > lo + 1e-9 * span && p < hi - 1e-9 * span && (pts.Count == 0 || Math.Abs(p - pts[^1]) > 1e-12 * span)) pts.Add(p);
        }
        return pts;
    }

    private static double Integrate(Func<double, double> f, double lo, double hi, List<double> pts)
        => hi > lo ? StatsDistMoreMath.QuadRel(f, lo, hi, pts, 1e-13) : 0.0;

    private static double GeG(double lg)
    {
        if (double.IsNaN(lg)) return 0.0;
        return Math.Exp(lg - Math.Exp(lg));
    }

    /// <summary>Density of the standard S0 stable law at x0.</summary>
    public static double Pdf(double x0, double alpha, double beta)
    {
        if (alpha == 2) return Math.Exp(-x0 * x0 / 4) / (2 * Math.Sqrt(Math.PI));
        if (alpha == 1 && beta == 0) return 1 / (Math.PI * (1 + x0 * x0));
        if (alpha == 0.5 && beta == 1)
        {
            var y = x0 + 1;
            return y <= 0 ? 0.0 : Math.Exp(-LnSqrt2Pi - 1.5 * Math.Log(y) - 1 / (2 * y));
        }
        if (alpha == 0.5 && beta == -1) return Pdf(-x0, alpha, 1);
        if (alpha == 1)
        {
            if (beta < 0) return Pdf(-x0, 1, -beta);
            Func<double, double> lg1 = t => LogG1(t, beta, x0);
            var pts1 = Breaks(lg1, -Math.PI / 2, Math.PI / 2);
            return Integrate(t => GeG(lg1(t)), -Math.PI / 2, Math.PI / 2, pts1) / (2 * beta);
        }
        var zeta = -beta * Math.Tan(Math.PI * alpha / 2);
        if (x0 < zeta) return Pdf(-x0, alpha, -beta);
        var xi = Math.Atan(-zeta) / alpha;
        if (x0 == zeta)
            return Math.Exp(Special.GammaLn(1 + 1 / alpha)) * Math.Cos(xi) / Math.PI / Math.Pow(1 + zeta * zeta, 1 / (2 * alpha));
        var logDx = Math.Log(x0 - zeta);
        Func<double, double> lg = t => LogG(t, alpha, xi, logDx);
        var pts = Breaks(lg, -xi, Math.PI / 2);
        var c2 = alpha / (Math.PI * Math.Abs(alpha - 1) * (x0 - zeta));
        return c2 * Integrate(t => GeG(lg(t)), -xi, Math.PI / 2, pts);
    }

    /// <summary>CDF and survival function of the standard S0 stable law at x0, each with relative accuracy.</summary>
    public static (double Cdf, double Sf) Cdf(double x0, double alpha, double beta)
    {
        if (alpha == 2) return (Special.Ndtr(x0 / Sqrt2), Special.Ndtr(-x0 / Sqrt2));
        if (alpha == 1 && beta == 0) return (Math.Atan2(1, -x0) / Math.PI, Math.Atan2(1, x0) / Math.PI);
        if (alpha == 0.5 && beta == 1)
        {
            var y = x0 + 1;
            if (y <= 0) return (0.0, 1.0);
            var r = Math.Sqrt(0.5 / y);
            return (Special.Erfc(r), Special.Erf(r));
        }
        if (alpha == 1)
        {
            if (beta < 0) { var (c, s) = Cdf(-x0, 1, -beta); return (s, c); }
            Func<double, double> lg1 = t => LogG1(t, beta, x0);
            var pts1 = Breaks(lg1, -Math.PI / 2, Math.PI / 2);
            var lower = Integrate(t => Math.Exp(-Math.Exp(lg1(t))), -Math.PI / 2, Math.PI / 2, pts1) / Math.PI;
            var upper = Integrate(t => -Special.Expm1(-Math.Exp(lg1(t))), -Math.PI / 2, Math.PI / 2, pts1) / Math.PI;
            return (lower, upper);
        }
        var zeta = -beta * Math.Tan(Math.PI * alpha / 2);
        if (x0 < zeta) { var (c, s) = Cdf(-x0, alpha, -beta); return (s, c); }
        var xi = Math.Atan(-zeta) / alpha;
        if (x0 == zeta) return ((Math.PI / 2 - xi) / Math.PI, (Math.PI / 2 + xi) / Math.PI);
        var logDx = Math.Log(x0 - zeta);
        Func<double, double> lg = t => LogG(t, alpha, xi, logDx);
        var lo = -xi;
        var hi = Math.PI / 2;
        if (!(hi > lo)) return (1.0, 0.0);
        var pts = Breaks(lg, lo, hi);
        var eInt = Integrate(t => { var v = lg(t); return double.IsNaN(v) ? 0.0 : Math.Exp(-Math.Exp(v)); }, lo, hi, pts);
        if (alpha > 1) return (1 - eInt / Math.PI, eInt / Math.PI);
        var oneMinus = Integrate(t => { var v = lg(t); return double.IsNaN(v) ? 0.0 : -Special.Expm1(-Math.Exp(v)); }, lo, hi, pts);
        return ((Math.PI / 2 - xi + eInt) / Math.PI, oneMinus / Math.PI);
    }

    /// <summary>One standard S1 variate by Chambers, Mallows and Stuck (1976), as scipy's <c>_rvs_Z1</c>.</summary>
    public static double RvsS1(GraviRandom rng, double alpha, double beta)
    {
        var th = Math.PI * (StatsRandom.U(rng) - 0.5);
        var w = StatsRandom.Exponential(rng);
        var ath = alpha * th;
        var bth = beta * th;
        var cosTh = Math.Cos(th);
        var tanTh = Math.Tan(th);
        if (alpha == 1)
            return 2 / Math.PI * ((Math.PI / 2 + bth) * tanTh - beta * Math.Log(Math.PI / 2 * w * cosTh / (Math.PI / 2 + bth)));
        if (beta == 0)
            return w / (cosTh / Math.Tan(ath) + Math.Sin(th)) * Math.Pow((Math.Cos(ath) + Math.Sin(ath) * tanTh) / w, 1 / alpha);
        var val0 = beta * Math.Tan(Math.PI * alpha / 2);
        var th0 = Math.Atan(val0) / alpha;
        var val3 = w / (cosTh / Math.Tan(alpha * (th0 + th)) + Math.Sin(th));
        return val3 * Math.Pow((Math.Cos(ath) + Math.Sin(ath) * tanTh - val0 * (Math.Sin(ath) - Math.Cos(ath) * tanTh)) / w, 1 / alpha);
    }
}

/// <summary>The Lévy alpha-stable distribution, scipy's <c>levy_stable</c>.</summary>
/// <remarks>
/// <para>
/// Density and CDF use Nolan's integral representations, scipy's default <c>'piecewise'</c> method,
/// with relative tolerance 1e-13 on each piece; against scipy 1.17.1 they agree to about 1e-10
/// away from the points discussed next, and both tails keep relative accuracy (the CDF far in a tail
/// is not formed as 1 − SF). Quantiles invert the CDF numerically; variates use Chambers–Mallows–Stuck.
/// </para>
/// <para>
/// Unlike scipy, the integrals are not short-circuited near x = ζ (scipy snaps |x − ζ| &lt; 0.005·α^(1/α)
/// to ζ, an error of order 1e-3 in the density there) or near α = 1 (scipy rounds |α − 1| &lt; 0.005 to
/// α = 1, which in S1 is discontinuous). Near α = 1 and β ≠ 0 the S1 location shift ζ is large and the
/// integrands steep; results there are slower but still converge.
/// </para>
/// <para>
/// In S1 with α = 1 the location applied is δ + 2βγ log(γ)/π (scipy's convention); <see cref="ContinuousDistribution.Loc"/>
/// reports that applied value and <see cref="NominalLoc"/> the δ passed in. The mean is the S1
/// location for α &gt; 1 (−β tan(πα/2)·γ + δ in S0) and undefined otherwise; the variance is 2γ² at
/// α = 2 and infinite below. There is no custom <c>fit</c> (scipy's uses McCulloch's quantile start
/// followed by the generic MLE, which is very slow with an integral per density evaluation).
/// </para>
/// </remarks>
public sealed class LevyStableDistribution : ContinuousDistribution
{
    private readonly double _zeta;

    /// <summary>Stable law with index <paramref name="alpha"/> in (0, 2] and skewness <paramref name="beta"/> in [−1, 1].</summary>
    public LevyStableDistribution(double alpha, double beta, double loc = 0.0, double scale = 1.0, LevyStableParameterization parameterization = LevyStableParameterization.S1)
        : base(EffectiveLoc(alpha, beta, loc, scale, parameterization), scale)
    {
        Alpha = alpha;
        Beta = beta;
        NominalLoc = loc;
        Parameterization = parameterization;
        _zeta = alpha == 1 ? 0.0 : -beta * Math.Tan(Math.PI * alpha / 2);
    }

    private static double EffectiveLoc(double alpha, double beta, double loc, double scale, LevyStableParameterization pz)
    {
        if (!(alpha > 0 && alpha <= 2)) throw new ArgumentException("alpha must lie in (0, 2].", nameof(alpha));
        if (!(beta >= -1 && beta <= 1)) throw new ArgumentException("beta must lie in [-1, 1].", nameof(beta));
        if (!(scale > 0) || !double.IsFinite(scale)) throw new ArgumentException("scale must be positive and finite.", nameof(scale));
        return pz == LevyStableParameterization.S1 && alpha == 1 ? loc + 2 * beta * scale * Math.Log(scale) / Math.PI : loc;
    }

    /// <summary>The stability index α.</summary>
    public double Alpha { get; }
    /// <summary>The skewness β.</summary>
    public double Beta { get; }
    /// <summary>The location δ as given (see the remarks for α = 1 in S1).</summary>
    public double NominalLoc { get; }
    /// <summary>The parameterisation, S1 by default as in scipy.</summary>
    public LevyStableParameterization Parameterization { get; }

    /// <inheritdoc />
    public override string Name => "levy_stable";
    /// <inheritdoc />
    public override string[] ShapeNames => ["alpha", "beta"];
    /// <inheritdoc />
    public override double[] Shapes => [Alpha, Beta];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new LevyStableDistribution(shapes[0], shapes[1], loc, scale, Parameterization);

    /// <summary>The S0 argument for a standardised value.</summary>
    private double X0(double z) => Parameterization == LevyStableParameterization.S1 ? z + _zeta : z;
    /// <inheritdoc />
    protected override double Lower0 => Alpha < 1 && Beta == 1 ? (Parameterization == LevyStableParameterization.S1 ? 0.0 : _zeta) : double.NegativeInfinity;
    /// <inheritdoc />
    protected override double Upper0 => Alpha < 1 && Beta == -1 ? (Parameterization == LevyStableParameterization.S1 ? 0.0 : _zeta) : double.PositiveInfinity;
    /// <inheritdoc />
    protected override double Pdf0(double z) => StatsStableMath.Pdf(X0(z), Alpha, Beta);
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(Pdf0(z));
    /// <inheritdoc />
    protected override double Cdf0(double z) => StatsStableMath.Cdf(X0(z), Alpha, Beta).Cdf;
    /// <inheritdoc />
    protected override double Sf0(double z) => StatsStableMath.Cdf(X0(z), Alpha, Beta).Sf;
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var mean = Alpha > 1 ? (Parameterization == LevyStableParameterization.S1 ? 0.0 : _zeta) : double.NaN;
        return new(mean, Alpha == 2 ? 2.0 : double.PositiveInfinity, Alpha == 2 ? 0.0 : double.NaN, Alpha == 2 ? 0.0 : double.NaN);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        var x = StatsStableMath.RvsS1(rng, Alpha, Beta);
        return Parameterization == LevyStableParameterization.S1 || Alpha == 1 ? x : x + _zeta;
    }
}

/// <summary>The Landau distribution, scipy's <c>landau</c>: the stable law with α = 1, β = 1 (S0, unit scale).</summary>
/// <remarks>
/// scipy evaluates it with Boost's rational approximations; here it is the Nolan integral of
/// <see cref="LevyStableDistribution"/>, which agrees with them to about 1e-12 relative and keeps
/// the doubly-exponential left tail and the 2/(πx) right tail accurate. Moments do not exist; the
/// entropy is scipy's constant.
/// </remarks>
public sealed class LandauDistribution : ContinuousDistribution
{
    /// <summary>Landau.</summary>
    public LandauDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "landau";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new LandauDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double z) => StatsStableMath.Pdf(z, 1, 1);
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(Pdf0(z));
    /// <inheritdoc />
    protected override double Cdf0(double z) => StatsStableMath.Cdf(z, 1, 1).Cdf;
    /// <inheritdoc />
    protected override double Sf0(double z) => StatsStableMath.Cdf(z, 1, 1).Sf;
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(double.NaN, double.NaN, double.NaN, double.NaN);
    /// <inheritdoc />
    protected override double Munp0(int n) => double.NaN;
    /// <inheritdoc />
    protected override double Entropy0() => 2.3726364400044817;
    /// <summary>Location and scale start from the median and half the interquartile range, as scipy.</summary>
    protected override double[] FitStart(double[] data)
    {
        var s = (double[])data.Clone();
        Array.Sort(s);
        double Pct(double p)
        {
            var h = (s.Length - 1) * p;
            var i = (int)Math.Floor(h);
            return i + 1 < s.Length ? s[i] + (h - i) * (s[i + 1] - s[i]) : s[i];
        }
        var iqr = Pct(0.75) - Pct(0.25);
        return [Pct(0.5), iqr > 0 ? iqr / 2 : 1.0];
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        const double pi2 = Math.PI / 2;
        var u = Math.PI * (StatsRandom.U(rng) - 0.5);
        var w = StatsRandom.Exponential(rng);
        return 2 / Math.PI * ((pi2 + u) * Math.Tan(u) - Math.Log(pi2 * w * Math.Cos(u) / (pi2 + u)));
    }
}
