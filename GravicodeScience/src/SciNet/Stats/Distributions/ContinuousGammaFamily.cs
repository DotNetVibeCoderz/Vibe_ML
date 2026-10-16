using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.StatsDistMath;

namespace Gravicode.Science.SciNet;

/// <summary>The gamma distribution with shape a, scipy's <c>gamma</c>.</summary>
/// <remarks>CDF, survival function and both quantiles are the regularised incomplete gamma
/// functions and their inverses, so neither tail is computed as a complement.</remarks>
public class GammaDistribution : ContinuousDistribution
{
    /// <summary>Gamma with shape <paramref name="a"/>.</summary>
    public GammaDistribution(double a, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => A = StatsDistributions.Positive(a, nameof(a));

    /// <summary>The shape parameter.</summary>
    public double A { get; }

    /// <inheritdoc />
    public override string Name => "gamma";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a"];
    /// <inheritdoc />
    public override double[] Shapes => [A];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new GammaDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Special.Xlogy(A - 1, z) - z - Special.GammaLn(A);
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.GammaInc(A, z);
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.GammaIncC(A, z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => Special.GammaIncInv(A, q);
    /// <inheritdoc />
    protected override double Isf0(double q) => Special.GammaIncCInv(A, q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(A, A, 2 / Math.Sqrt(A), 6 / A);
    /// <inheritdoc />
    protected override double Munp0(int n) => Math.Exp(Special.GammaLn(A + n) - Special.GammaLn(A));
    /// <inheritdoc />
    protected override double Entropy0() => A + Special.GammaLn(A) + (1 - A) * Special.Digamma(A);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => StatsRandom.Gamma(rng, A);

    /// <inheritdoc />
    protected override double[] FitStart(double[] data)
    {
        // scipy's _fitstart: shape from the sample skewness, a = 4/skew².
        var sk = StatsFitHelpers.Skewness(data);
        var a = Math.Clamp(4 / (sk * sk + 1e-12), 1e-3, 1e6);
        var std = new GammaDistribution(a);
        var (loc, scale) = std.FitLocScaleSupport(data);
        return [a, loc, scale];
    }

    /// <summary>
    /// MLE. With the location fixed the shape solves log a − ψ(a) = log mean(y) − mean(log y),
    /// y = x − loc (the closed reduction scipy uses), and scale = mean(y)/a; otherwise all three
    /// parameters are fitted numerically.
    /// </summary>
    public static GammaDistribution Fit(double[] data, double? fa = null, double? floc = null, double? fscale = null)
    {
        StatsDistributions.CheckData(data);
        if (floc is double loc)
        {
            var y = data.Select(x => x - loc).ToArray();
            if (y.Any(v => v <= 0)) throw new ArgumentException("All data must exceed floc.", nameof(floc));
            var my = y.Average();
            if (fa is double a0) return new GammaDistribution(a0, loc, fscale ?? my / a0);
            if (fscale is double s0)
            {
                // ψ(a) = mean(log y) − log s.
                var target = y.Average(v => Math.Log(v)) - Math.Log(s0);
                var a1 = StatsFitHelpers.PositiveRoot(a => Special.Digamma(a) - target, 1.0);
                return new GammaDistribution(a1, loc, s0);
            }
            var s = Math.Log(my) - y.Average(v => Math.Log(v));
            var guess = (3 - s + Math.Sqrt((s - 3) * (s - 3) + 24 * s)) / (12 * s);
            var a = StatsFitHelpers.PositiveRoot(v => Math.Log(v) - Special.Digamma(v) - s, guess);
            return new GammaDistribution(a, loc, my / a);
        }
        return (GammaDistribution)new GammaDistribution(1.0).FitNumerically(data, [fa], null, fscale);
    }

    /// <inheritdoc />
    public override ContinuousDistribution FitTo(double[] data, double?[]? fixedShapes = null, double? floc = null, double? fscale = null)
        => floc.HasValue ? Fit(data, StatsDistributions.Fixed(fixedShapes, 0), floc, fscale) : FitNumerically(data, fixedShapes, floc, fscale);
}

/// <summary>The Erlang distribution, scipy's <c>erlang</c>: a gamma with integer shape.</summary>
public sealed class ErlangDistribution : GammaDistribution
{
    /// <summary>Erlang with shape <paramref name="a"/> (scipy accepts, and so does this, a non-integer shape).</summary>
    public ErlangDistribution(double a, double loc = 0.0, double scale = 1.0) : base(a, loc, scale) { }

    /// <inheritdoc />
    public override string Name => "erlang";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new ErlangDistribution(shapes[0], loc, scale);
}

/// <summary>The beta distribution on [0, 1], scipy's <c>beta</c>.</summary>
public sealed class BetaDistribution : ContinuousDistribution
{
    /// <summary>Beta with shapes <paramref name="a"/>, <paramref name="b"/>.</summary>
    public BetaDistribution(double a, double b, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        A = StatsDistributions.Positive(a, nameof(a));
        B = StatsDistributions.Positive(b, nameof(b));
    }

    /// <summary>First shape parameter.</summary>
    public double A { get; }
    /// <summary>Second shape parameter.</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "beta";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a", "b"];
    /// <inheritdoc />
    public override double[] Shapes => [A, B];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new BetaDistribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Upper0 => 1.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Special.Xlogy(A - 1, z) + Special.Xlog1py(B - 1, -z) - Special.BetaLn(A, B);
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.BetaInc(A, B, z);
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.BetaIncC(A, B, z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => Special.BetaIncInv(A, B, q);
    /// <inheritdoc />
    protected override double Isf0(double q) => Special.BetaIncCInv(A, B, q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        double a = A, b = B, s = a + b;
        var mean = a / s;
        var v = a * b / (s * s * (s + 1));
        var skew = 2 * (b - a) * Math.Sqrt(s + 1) / ((s + 2) * Math.Sqrt(a * b));
        var kurt = 6 * ((a - b) * (a - b) * (s + 1) - a * b * (s + 2)) / (a * b * (s + 2) * (s + 3));
        return new(mean, v, skew, kurt);
    }
    /// <inheritdoc />
    protected override double Munp0(int n) => Math.Exp(Special.BetaLn(A + n, B) - Special.BetaLn(A, B));
    /// <inheritdoc />
    protected override double Entropy0()
        => Special.BetaLn(A, B) - (A - 1) * Special.Digamma(A) - (B - 1) * Special.Digamma(B) + (A + B - 2) * Special.Digamma(A + B);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => StatsRandom.Beta(rng, A, B);

    /// <inheritdoc />
    protected override double[] FitStart(double[] data)
    {
        // Method of moments on the data rescaled into (0, 1).
        var min = data.Min();
        var max = data.Max();
        var w = max - min;
        var loc = min - 0.01 * w - 1e-9;
        var scale = w * 1.02 + 2e-9;
        var y = data.Select(x => (x - loc) / scale).ToArray();
        var m = y.Average();
        var v = y.Average(t => (t - m) * (t - m));
        var c = m * (1 - m) / Math.Max(v, 1e-12) - 1;
        var a = c > 0 ? m * c : 1.0;
        var b = c > 0 ? (1 - m) * c : 1.0;
        return [a, b, loc, scale];
    }
}

/// <summary>Student's t distribution, scipy's <c>t</c>.</summary>
public sealed class StudentTDistribution : ContinuousDistribution
{
    /// <summary>Student's t with <paramref name="df"/> degrees of freedom.</summary>
    public StudentTDistribution(double df, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        if (!(df > 0)) throw new ArgumentException("df must be positive.", nameof(df));
        Df = df;
    }

    /// <summary>Degrees of freedom (may be +∞, the normal limit).</summary>
    public double Df { get; }

    /// <inheritdoc />
    public override string Name => "t";
    /// <inheritdoc />
    public override string[] ShapeNames => ["df"];
    /// <inheritdoc />
    public override double[] Shapes => [Df];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new StudentTDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
    {
        if (double.IsPositiveInfinity(Df) || Df > 1e12) return LogPhi(z);
        var d = Df;
        return Special.GammaLn((d + 1) / 2) - Special.GammaLn(d / 2) - 0.5 * Math.Log(d * Math.PI) - (d + 1) / 2 * Special.Log1p(z * z / d);
    }
    /// <inheritdoc />
    protected override double Cdf0(double z) => double.IsPositiveInfinity(Df) ? Special.Ndtr(z) : Special.Stdtr(Df, z);
    /// <inheritdoc />
    protected override double Sf0(double z) => Cdf0(-z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => double.IsPositiveInfinity(Df) ? Special.Ndtri(q) : Special.Stdtrit(Df, q);
    /// <inheritdoc />
    protected override double Isf0(double q) => -Ppf0(q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var d = Df;
        var mean = d > 1 ? 0.0 : double.PositiveInfinity;
        var v = d > 2 ? (double.IsPositiveInfinity(d) ? 1.0 : d / (d - 2)) : d > 1 ? double.PositiveInfinity : double.NaN;
        var s = d > 3 ? 0.0 : double.NaN;
        var k = d > 4 ? (double.IsPositiveInfinity(d) ? 0.0 : 6 / (d - 4)) : d > 2 ? double.PositiveInfinity : double.NaN;
        return new(mean, v, s, k);
    }
    /// <inheritdoc />
    protected override double Entropy0()
    {
        var d = Df;
        if (double.IsPositiveInfinity(d)) return 0.5 * (Math.Log(2 * Math.PI) + 1);
        var h = (d + 1) / 2;
        return h * (Special.Digamma(h) - Special.Digamma(d / 2)) + Math.Log(Math.Sqrt(d)) + Special.BetaLn(d / 2, 0.5);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        var z = StatsRandom.Normal(rng);
        if (double.IsPositiveInfinity(Df)) return z;
        return z / Math.Sqrt(2 * StatsRandom.Gamma(rng, Df / 2) / Df);
    }
    /// <inheritdoc />
    protected override double[] FitStart(double[] data)
    {
        var m = data.Average();
        var sd = Math.Sqrt(data.Average(x => (x - m) * (x - m)));
        return [Df, StatsFitHelpers.Median(data), sd > 0 ? sd : 1];
    }
}

/// <summary>The chi-squared distribution, scipy's <c>chi2</c>.</summary>
public sealed class ChiSquaredDistribution : ContinuousDistribution
{
    /// <summary>Chi-squared with <paramref name="df"/> degrees of freedom.</summary>
    public ChiSquaredDistribution(double df, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => Df = StatsDistributions.Positive(df, nameof(df));

    /// <summary>Degrees of freedom.</summary>
    public double Df { get; }

    /// <inheritdoc />
    public override string Name => "chi2";
    /// <inheritdoc />
    public override string[] ShapeNames => ["df"];
    /// <inheritdoc />
    public override double[] Shapes => [Df];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new ChiSquaredDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Special.Xlogy(Df / 2 - 1, z) - z / 2 - Special.GammaLn(Df / 2) - Ln2 * Df / 2;
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.GammaInc(Df / 2, z / 2);
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.GammaIncC(Df / 2, z / 2);
    /// <inheritdoc />
    protected override double Ppf0(double q) => 2 * Special.GammaIncInv(Df / 2, q);
    /// <inheritdoc />
    protected override double Isf0(double q) => 2 * Special.GammaIncCInv(Df / 2, q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(Df, 2 * Df, Math.Sqrt(8 / Df), 12 / Df);
    /// <inheritdoc />
    protected override double Munp0(int n) => Math.Exp(n * Ln2 + Special.GammaLn(Df / 2 + n) - Special.GammaLn(Df / 2));
    /// <inheritdoc />
    protected override double Entropy0()
    {
        var h = Df / 2;
        return h + Math.Log(2) + Special.GammaLn(h) + (1 - h) * Special.Digamma(h);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => 2 * StatsRandom.Gamma(rng, Df / 2);
}

/// <summary>The chi distribution (square root of a chi-squared), scipy's <c>chi</c>.</summary>
public sealed class ChiDistribution : ContinuousDistribution
{
    /// <summary>Chi with <paramref name="df"/> degrees of freedom.</summary>
    public ChiDistribution(double df, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => Df = StatsDistributions.Positive(df, nameof(df));

    /// <summary>Degrees of freedom.</summary>
    public double Df { get; }

    /// <inheritdoc />
    public override string Name => "chi";
    /// <inheritdoc />
    public override string[] ShapeNames => ["df"];
    /// <inheritdoc />
    public override double[] Shapes => [Df];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new ChiDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Special.Xlogy(Df - 1, z) - z * z / 2 - (Df / 2 - 1) * Ln2 - Special.GammaLn(Df / 2);
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.GammaInc(Df / 2, z * z / 2);
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.GammaIncC(Df / 2, z * z / 2);
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Sqrt(2 * Special.GammaIncInv(Df / 2, q));
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Sqrt(2 * Special.GammaIncCInv(Df / 2, q));
    /// <inheritdoc />
    protected override double Munp0(int n) => Math.Exp(n / 2.0 * Ln2 + Special.GammaLn((Df + n) / 2) - Special.GammaLn(Df / 2));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var mu = Munp0(1);
        var v = Df - mu * mu;
        var g1 = (2 * mu * mu * mu + mu * (1 - 2 * Df)) / Math.Pow(v, 1.5);
        var g2 = 2 * Df * (1 - Df) - 6 * mu * mu * mu * mu + 4 * mu * mu * (2 * Df - 1);
        return new(mu, v, g1, g2 / (v * v));
    }
    /// <inheritdoc />
    protected override double Entropy0()
    {
        var h = Df / 2;
        return Special.GammaLn(h) + 0.5 * (Df - Ln2 - (Df - 1) * Special.Digamma(h));
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Sqrt(2 * StatsRandom.Gamma(rng, Df / 2));
}

/// <summary>Fisher's F distribution, scipy's <c>f</c>.</summary>
/// <remarks>Quantiles invert whichever incomplete-beta form keeps the small quantity small: the
/// upper tail goes through I_{dfd/2, dfn/2}, so z = dfd(1 − y)/(dfn·y) never subtracts from 1 near 1.</remarks>
public sealed class FDistribution : ContinuousDistribution
{
    /// <summary>F with numerator and denominator degrees of freedom.</summary>
    public FDistribution(double dfn, double dfd, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        Dfn = StatsDistributions.Positive(dfn, nameof(dfn));
        Dfd = StatsDistributions.Positive(dfd, nameof(dfd));
    }

    /// <summary>Numerator degrees of freedom.</summary>
    public double Dfn { get; }
    /// <summary>Denominator degrees of freedom.</summary>
    public double Dfd { get; }

    /// <inheritdoc />
    public override string Name => "f";
    /// <inheritdoc />
    public override string[] ShapeNames => ["dfn", "dfd"];
    /// <inheritdoc />
    public override double[] Shapes => [Dfn, Dfd];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new FDistribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
    {
        double m = Dfn, n = Dfd;
        // log(n + m z) written as log n + log1p(m z / n) keeps small z exact.
        return m / 2 * Math.Log(m / n) + Special.Xlogy(m / 2 - 1, z) - (m + n) / 2 * Special.Log1p(m * z / n) - Special.BetaLn(m / 2, n / 2);
    }
    /// <inheritdoc />
    protected override double Cdf0(double z)
    {
        var mz = Dfn * z;
        return mz <= Dfd ? Special.BetaInc(Dfn / 2, Dfd / 2, mz / (mz + Dfd)) : Special.BetaIncC(Dfd / 2, Dfn / 2, Dfd / (mz + Dfd));
    }
    /// <inheritdoc />
    protected override double Sf0(double z)
    {
        var mz = Dfn * z;
        return mz <= Dfd ? Special.BetaIncC(Dfn / 2, Dfd / 2, mz / (mz + Dfd)) : Special.BetaInc(Dfd / 2, Dfn / 2, Dfd / (mz + Dfd));
    }
    /// <inheritdoc />
    protected override double Ppf0(double q)
    {
        if (q <= 0.5)
        {
            var x = Special.BetaIncInv(Dfn / 2, Dfd / 2, q);
            return Dfd * x / (Dfn * (1 - x));
        }
        var y = Special.BetaIncInv(Dfd / 2, Dfn / 2, 1 - q);
        return Dfd * (1 - y) / (Dfn * y);
    }
    /// <inheritdoc />
    protected override double Isf0(double q)
    {
        if (q <= 0.5)
        {
            var y = Special.BetaIncInv(Dfd / 2, Dfn / 2, q);
            return Dfd * (1 - y) / (Dfn * y);
        }
        var x = Special.BetaIncInv(Dfn / 2, Dfd / 2, 1 - q);
        return Dfd * x / (Dfn * (1 - x));
    }
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        double m = Dfn, n = Dfd;
        var mean = n > 2 ? n / (n - 2) : double.PositiveInfinity;
        var v = n > 4 ? 2 * n * n * (m + n - 2) / (m * (n - 2) * (n - 2) * (n - 4)) : n > 2 ? double.PositiveInfinity : double.NaN;
        var s = n > 6 ? (2 * m + n - 2) * Math.Sqrt(8 * (n - 4)) / ((n - 6) * Math.Sqrt(m * (m + n - 2))) : double.NaN;
        var k = n > 8 ? 12 * (m * (5 * n - 22) * (m + n - 2) + (n - 4) * (n - 2) * (n - 2)) / (m * (n - 6) * (n - 8) * (m + n - 2)) : double.NaN;
        return new(mean, v, s, k);
    }
    /// <inheritdoc />
    protected override double Entropy0()
    {
        double hm = Dfn / 2, hn = Dfd / 2, hs = (Dfn + Dfd) / 2;
        return Math.Log(Dfd) - Math.Log(Dfn) + Special.BetaLn(hm, hn) + (1 - hm) * Special.Digamma(hm)
               - (1 + hn) * Special.Digamma(hn) + hs * Special.Digamma(hs);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
        => StatsRandom.Gamma(rng, Dfn / 2) / Dfn / (StatsRandom.Gamma(rng, Dfd / 2) / Dfd);
}

/// <summary>The log-normal distribution with shape s (σ of log X), scipy's <c>lognorm</c>; scale = e^μ.</summary>
public sealed class LogNormalDistribution : ContinuousDistribution
{
    /// <summary>Log-normal with shape <paramref name="s"/>.</summary>
    public LogNormalDistribution(double s, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => S = StatsDistributions.Positive(s, nameof(s));

    /// <summary>Shape: the standard deviation of log X.</summary>
    public double S { get; }

    /// <inheritdoc />
    public override string Name => "lognorm";
    /// <inheritdoc />
    public override string[] ShapeNames => ["s"];
    /// <inheritdoc />
    public override double[] Shapes => [S];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new LogNormalDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => z == 0 ? 0.0 : Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
    {
        if (z == 0) return double.NegativeInfinity;
        var l = Math.Log(z) / S;
        return -0.5 * l * l - LnSqrt2Pi - Math.Log(S * z);
    }
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.Ndtr(Math.Log(z) / S);
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.Ndtr(-Math.Log(z) / S);
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Special.LogNdtr(Math.Log(z) / S);
    /// <inheritdoc />
    protected override double LogSf0(double z) => Special.LogNdtr(-Math.Log(z) / S);
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Exp(S * Special.Ndtri(q));
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Exp(-S * Special.Ndtri(q));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var p = Math.Exp(S * S);
        var pm1 = Special.Expm1(S * S);
        return new(Math.Sqrt(p), p * pm1, (p + 2) * Math.Sqrt(pm1), p * p * p * p + 2 * p * p * p + 3 * p * p - 6);
    }
    /// <inheritdoc />
    protected override double Munp0(int n) => Math.Exp(0.5 * n * n * S * S);
    /// <inheritdoc />
    protected override double Entropy0() => 0.5 * (1 + Math.Log(2 * Math.PI) + 2 * Math.Log(S));
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Exp(S * StatsRandom.Normal(rng));

    /// <summary>
    /// MLE. With the location fixed it is closed form — the mean and (ddof = 0) standard deviation of
    /// log(x − loc), as scipy; otherwise numerical in all three parameters.
    /// </summary>
    public static LogNormalDistribution Fit(double[] data, double? fs = null, double? floc = null, double? fscale = null)
    {
        StatsDistributions.CheckData(data);
        if (floc is double loc)
        {
            var l = data.Select(x => x - loc).ToArray();
            if (l.Any(v => v <= 0)) throw new ArgumentException("All data must exceed floc.", nameof(floc));
            var logs = l.Select(v => Math.Log(v)).ToArray();
            var mu = fscale is double sc ? Math.Log(sc) : logs.Average();
            var s = fs ?? Math.Sqrt(logs.Average(v => (v - mu) * (v - mu)));
            return new LogNormalDistribution(s, loc, Math.Exp(mu));
        }
        return (LogNormalDistribution)new LogNormalDistribution(1.0).FitNumerically(data, [fs], null, fscale);
    }

    /// <inheritdoc />
    protected override double[] FitStart(double[] data)
    {
        var min = data.Min();
        var max = data.Max();
        var loc = min - 0.1 * (max - min) - 1e-6 * Math.Max(1, Math.Abs(min));
        var logs = data.Select(x => Math.Log(x - loc)).ToArray();
        var mu = logs.Average();
        var s = Math.Sqrt(logs.Average(v => (v - mu) * (v - mu)));
        return [s > 0 ? s : 1, loc, Math.Exp(mu)];
    }

    /// <inheritdoc />
    public override ContinuousDistribution FitTo(double[] data, double?[]? fixedShapes = null, double? floc = null, double? fscale = null)
        => floc.HasValue ? Fit(data, StatsDistributions.Fixed(fixedShapes, 0), floc, fscale) : FitNumerically(data, fixedShapes, floc, fscale);
}

/// <summary>The inverse gamma distribution, scipy's <c>invgamma</c>.</summary>
public sealed class InvGammaDistribution : ContinuousDistribution
{
    /// <summary>Inverse gamma with shape <paramref name="a"/>.</summary>
    public InvGammaDistribution(double a, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => A = StatsDistributions.Positive(a, nameof(a));

    /// <summary>The shape parameter.</summary>
    public double A { get; }

    /// <inheritdoc />
    public override string Name => "invgamma";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a"];
    /// <inheritdoc />
    public override double[] Shapes => [A];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new InvGammaDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => z == 0 ? 0.0 : Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => z == 0 ? double.NegativeInfinity : -(A + 1) * Math.Log(z) - Special.GammaLn(A) - 1 / z;
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.GammaIncC(A, 1 / z);
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.GammaInc(A, 1 / z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => 1 / Special.GammaIncCInv(A, q);
    /// <inheritdoc />
    protected override double Isf0(double q) => 1 / Special.GammaIncInv(A, q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var a = A;
        var m = a > 1 ? 1 / (a - 1) : double.PositiveInfinity;
        var v = a > 2 ? 1 / ((a - 1) * (a - 1) * (a - 2)) : double.PositiveInfinity;
        var s = a > 3 ? 4 * Math.Sqrt(a - 2) / (a - 3) : double.NaN;
        var k = a > 4 ? 6 * (5 * a - 11) / ((a - 3) * (a - 4)) : double.NaN;
        return new(m, v, s, k);
    }
    /// <inheritdoc />
    protected override double Munp0(int n) => A > n ? Math.Exp(Special.GammaLn(A - n) - Special.GammaLn(A)) : double.PositiveInfinity;
    /// <inheritdoc />
    protected override double Entropy0() => A - (A + 1) * Special.Digamma(A) + Special.GammaLn(A);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => 1 / StatsRandom.Gamma(rng, A);
}

/// <summary>The Nakagami-m distribution, scipy's <c>nakagami</c>: √(Gamma(ν)/ν).</summary>
public sealed class NakagamiDistribution : ContinuousDistribution
{
    /// <summary>Nakagami with shape <paramref name="nu"/>.</summary>
    public NakagamiDistribution(double nu, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => Nu = StatsDistributions.Positive(nu, nameof(nu));

    /// <summary>The shape parameter ν (Nakagami's m).</summary>
    public double Nu { get; }

    /// <inheritdoc />
    public override string Name => "nakagami";
    /// <inheritdoc />
    public override string[] ShapeNames => ["nu"];
    /// <inheritdoc />
    public override double[] Shapes => [Nu];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new NakagamiDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Ln2 + Nu * Math.Log(Nu) - Special.GammaLn(Nu) + Special.Xlogy(2 * Nu - 1, z) - Nu * z * z;
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.GammaInc(Nu, Nu * z * z);
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.GammaIncC(Nu, Nu * z * z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Sqrt(Special.GammaIncInv(Nu, q) / Nu);
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Sqrt(Special.GammaIncCInv(Nu, q) / Nu);
    /// <inheritdoc />
    protected override double Munp0(int n) => Math.Exp(Special.GammaLn(Nu + n / 2.0) - Special.GammaLn(Nu) - n / 2.0 * Math.Log(Nu));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var nu = Nu;
        var mu = Munp0(1);
        var mu2 = 1 - mu * mu;
        var g1 = mu * (1 - 4 * nu * mu2) / 2 / nu / Math.Pow(mu2, 1.5);
        var g2 = (-6 * Math.Pow(mu, 4) * nu + (8 * nu - 2) * mu * mu - 2 * nu + 1) / (nu * mu2 * mu2);
        return new(mu, mu2, g1, g2);
    }
    /// <inheritdoc />
    protected override double Entropy0()
        => Special.GammaLn(Nu) + Nu + (0.5 - Nu) * Special.Digamma(Nu) - Ln2 - 0.5 * Math.Log(Nu);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Sqrt(StatsRandom.Gamma(rng, Nu) / Nu);
}

/// <summary>The Pearson type III distribution parameterised by its skewness, scipy's <c>pearson3</c>.</summary>
/// <remarks>A shifted, scaled gamma with mean 0 and variance 1; skew = 0 is the normal.
/// For negative skew the gamma is reflected, so the tails swap roles.</remarks>
public sealed class Pearson3Distribution : ContinuousDistribution
{
    private readonly double _alpha, _beta, _zeta;

    /// <summary>Pearson III with the given skewness.</summary>
    public Pearson3Distribution(double skew, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        SkewParameter = StatsDistributions.Finite(skew, nameof(skew));
        if (Math.Abs(skew) >= 1.6e-5)
        {
            _alpha = 4 / (skew * skew);
            _beta = 2 / skew;
            _zeta = -_alpha / _beta;
        }
    }

    /// <summary>The skewness shape parameter.</summary>
    public double SkewParameter { get; }

    private bool IsNormal => Math.Abs(SkewParameter) < 1.6e-5;

    /// <inheritdoc />
    public override string Name => "pearson3";
    /// <inheritdoc />
    public override string[] ShapeNames => ["skew"];
    /// <inheritdoc />
    public override double[] Shapes => [SkewParameter];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new Pearson3Distribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => IsNormal || SkewParameter < 0 ? double.NegativeInfinity : _zeta;
    /// <inheritdoc />
    protected override double Upper0 => IsNormal || SkewParameter > 0 ? double.PositiveInfinity : _zeta;
    private double Y(double z) => _beta * (z - _zeta);
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
    {
        if (IsNormal) return LogPhi(z);
        var y = Y(z);
        return Math.Log(Math.Abs(_beta)) + Special.Xlogy(_alpha - 1, y) - y - Special.GammaLn(_alpha);
    }
    /// <inheritdoc />
    protected override double Cdf0(double z)
        => IsNormal ? Special.Ndtr(z) : _beta > 0 ? Special.GammaInc(_alpha, Y(z)) : Special.GammaIncC(_alpha, Y(z));
    /// <inheritdoc />
    protected override double Sf0(double z)
        => IsNormal ? Special.Ndtr(-z) : _beta > 0 ? Special.GammaIncC(_alpha, Y(z)) : Special.GammaInc(_alpha, Y(z));
    /// <inheritdoc />
    protected override double Ppf0(double q)
        => IsNormal ? Special.Ndtri(q) : _zeta + (_beta > 0 ? Special.GammaIncInv(_alpha, q) : Special.GammaIncCInv(_alpha, q)) / _beta;
    /// <inheritdoc />
    protected override double Isf0(double q)
        => IsNormal ? -Special.Ndtri(q) : _zeta + (_beta > 0 ? Special.GammaIncCInv(_alpha, q) : Special.GammaIncInv(_alpha, q)) / _beta;
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(0, 1, SkewParameter, 1.5 * SkewParameter * SkewParameter);
    /// <inheritdoc />
    protected override double Entropy0()
        => IsNormal ? 0.5 * (Math.Log(2 * Math.PI) + 1)
            : _alpha + Special.GammaLn(_alpha) + (1 - _alpha) * Special.Digamma(_alpha) - Math.Log(Math.Abs(_beta));
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
        => IsNormal ? StatsRandom.Normal(rng) : _zeta + StatsRandom.Gamma(rng, _alpha) / _beta;
}

/// <summary>The inverse Gaussian distribution with mean μ, scipy's <c>invgauss</c>.</summary>
/// <remarks>The CDF is Φ(a) + e^{2/μ}Φ(b); both terms are combined in log space, because for small μ
/// the factor e^{2/μ} overflows while Φ(b) underflows and their product is perfectly ordinary.</remarks>
public class InvGaussDistribution : ContinuousDistribution
{
    /// <summary>Inverse Gaussian with mean parameter <paramref name="mu"/>.</summary>
    public InvGaussDistribution(double mu, double loc = 0.0, double scale = 1.0) : base(loc, scale)
        => Mu = StatsDistributions.Positive(mu, nameof(mu));

    /// <summary>The mean parameter μ.</summary>
    public double Mu { get; }

    /// <inheritdoc />
    public override string Name => "invgauss";
    /// <inheritdoc />
    public override string[] ShapeNames => ["mu"];
    /// <inheritdoc />
    public override double[] Shapes => [Mu];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new InvGaussDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => z == 0 ? 0.0 : Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
        => z == 0 ? double.NegativeInfinity : -LnSqrt2Pi - 1.5 * Math.Log(z) - (z - Mu) * (z - Mu) / (2 * z * Mu * Mu);
    private (double A, double B) Args(double z)
    {
        var f = 1 / Math.Sqrt(z);
        return (f * (z - Mu) / Mu, -f * (z + Mu) / Mu);
    }
    /// <inheritdoc />
    protected override double LogCdf0(double z)
    {
        if (z <= 0) return double.NegativeInfinity;
        var (a, b) = Args(z);
        return LogAddExp(Special.LogNdtr(a), 2 / Mu + Special.LogNdtr(b));
    }
    /// <inheritdoc />
    protected override double LogSf0(double z)
    {
        if (z <= 0) return 0.0;
        var (a, b) = Args(z);
        if (a > 2)
        {
            // Φ(−a) − e^{2/μ}Φ(b) with both tails written through erfcx: since b² − a² = 4/μ exactly,
            // the common factor e^{−a²/2} comes out and no overflow or rounding inversion can occur.
            var d = Special.Erfcx(a / Sqrt2) - Special.Erfcx(-b / Sqrt2);
            return -0.5 * a * a + Math.Log(0.5 * d);
        }
        return LogSubExp(Special.LogNdtr(-a), 2 / Mu + Special.LogNdtr(b));
    }
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Exp(LogCdf0(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(LogSf0(z));
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(Mu, Mu * Mu * Mu, 3 * Math.Sqrt(Mu), 15 * Mu);
    /// <inheritdoc />
    protected override double Ppf0(double q)
        => q <= 0.5
            ? InvertMonotone(LogCdf0, Math.Log(q), 0, double.PositiveInfinity, true, Mu)
            : InvertMonotone(LogSf0, Special.Log1p(-q), 0, double.PositiveInfinity, false, Mu);
    /// <inheritdoc />
    protected override double Isf0(double q)
        => q <= 0.5
            ? InvertMonotone(LogSf0, Math.Log(q), 0, double.PositiveInfinity, false, Mu)
            : InvertMonotone(LogCdf0, Special.Log1p(-q), 0, double.PositiveInfinity, true, Mu);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        // Michael, Schucany & Haas (1976).
        var n = StatsRandom.Normal(rng);
        var y = n * n;
        var mu = Mu;
        var x = mu + mu * mu * y / 2 - mu / 2 * Math.Sqrt(4 * mu * y + mu * mu * y * y);
        return StatsRandom.U(rng) <= mu / (mu + x) ? x : mu * mu / x;
    }
}

/// <summary>The Wald distribution, scipy's <c>wald</c>: the inverse Gaussian with μ = 1.</summary>
public sealed class WaldDistribution : InvGaussDistribution
{
    /// <summary>Wald with the given location and scale.</summary>
    public WaldDistribution(double loc = 0.0, double scale = 1.0) : base(1.0, loc, scale) { }

    /// <inheritdoc />
    public override string Name => "wald";
    /// <inheritdoc />
    public override string[] ShapeNames => [];
    /// <inheritdoc />
    public override double[] Shapes => [];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new WaldDistribution(loc, scale);
}
