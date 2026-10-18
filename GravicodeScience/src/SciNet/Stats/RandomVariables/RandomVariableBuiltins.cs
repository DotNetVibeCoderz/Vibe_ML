using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// The normal distribution with mean <see cref="Mu"/> and standard deviation <see cref="Sigma"/>
/// (scipy's <c>stats.Normal</c>). Every function has a formula: the CDFs through
/// <c>ndtr</c>/<c>log_ndtr</c>, the quantiles through
/// <c>ndtri</c> and <c>ndtri_exp</c>.
/// </summary>
public class Normal : ContinuousRandomVariable
{
    /// <summary>The normal with mean <paramref name="mu"/> and standard deviation <paramref name="sigma"/> &gt; 0.</summary>
    public Normal(double mu = 0.0, double sigma = 1.0)
    {
        if (!double.IsFinite(mu)) throw new ArgumentException("mu must be finite.", nameof(mu));
        if (!(sigma > 0) || !double.IsFinite(sigma)) throw new ArgumentException("sigma must be positive and finite.", nameof(sigma));
        Mu = mu;
        Sigma = sigma;
    }

    /// <summary>The mean μ.</summary>
    public double Mu { get; }

    /// <summary>The standard deviation σ.</summary>
    public double Sigma { get; }

    private double Z(double x) => (x - Mu) / Sigma;

    /// <inheritdoc />
    public override string Name => "Normal";
    /// <inheritdoc />
    protected internal override (double Lower, double Upper) SupportCore() => (double.NegativeInfinity, double.PositiveInfinity);
    /// <inheritdoc />
    protected override double LogPdfFormula(double x) => StandardNormal.LogPdf0(Z(x)) - Math.Log(Sigma);
    /// <inheritdoc />
    protected override double PdfFormula(double x) => StandardNormal.Pdf0(Z(x)) / Sigma;
    /// <inheritdoc />
    protected override double LogCdfFormula(double x) => Special.LogNdtr(Z(x));
    /// <inheritdoc />
    protected override double CdfFormula(double x) => Special.Ndtr(Z(x));
    /// <inheritdoc />
    protected override double LogCcdfFormula(double x) => Special.LogNdtr(-Z(x));
    /// <inheritdoc />
    protected override double CcdfFormula(double x) => Special.Ndtr(-Z(x));
    /// <inheritdoc />
    protected override double IcdfFormula(double p) => Special.Ndtri(p) * Sigma + Mu;
    /// <inheritdoc />
    protected override double ILogCdfFormula(double logp) => StatsDistMath.NdtriExp(logp) * Sigma + Mu;
    /// <inheritdoc />
    protected override double IccdfFormula(double p) => -Special.Ndtri(p) * Sigma + Mu;
    /// <inheritdoc />
    protected override double ILogCcdfFormula(double logp) => -StatsDistMath.NdtriExp(logp) * Sigma + Mu;
    /// <inheritdoc />
    protected override double EntropyFormula() => StandardNormal.Entropy0 + Math.Log(Math.Abs(Sigma));
    /// <inheritdoc />
    protected override Complex LogEntropyFormula() => RandomVariableNumerics.LogReal(EntropyFormula());
    /// <inheritdoc />
    protected override double MedianFormula() => Mu;
    /// <inheritdoc />
    protected override double ModeFormula() => Mu;

    /// <inheritdoc />
    protected override double? MomentRawFormula(int order) => order switch { 0 => 1.0, 1 => Mu, _ => null };

    /// <inheritdoc />
    protected override double? MomentCentralFormula(int order)
    {
        if (order == 0) return 1.0;
        if (order % 2 == 1) return 0.0;
        return Math.Pow(Sigma, order) * Special.Factorial2(order - 1);
    }

    /// <inheritdoc />
    protected override void SampleFormula(Span<double> destination, GraviRandom rng)
    {
        for (var i = 0; i < destination.Length; i++) destination[i] = Mu + Sigma * StatsRandom.Normal(rng);
    }

    /// <inheritdoc />
    public override string ToString() => $"Normal(mu={PyFormat.F(Mu)}, sigma={PyFormat.F(Sigma)})";
}

/// <summary>The standard normal (scipy's <c>StandardNormal</c>, what <c>stats.Normal()</c> returns).</summary>
public sealed class StandardNormal : Normal
{
    internal const double LogNormalization = 0.91893853320467274178; // log(2π)/2
    internal const double Normalization = 0.39894228040143267794;    // 1/√(2π)
    internal static readonly double Entropy0 = (1 + Math.Log(2 * Math.PI)) / 2;

    /// <summary>The standard normal, μ = 0, σ = 1.</summary>
    public StandardNormal() : base(0.0, 1.0) { }

    internal static double LogPdf0(double x) => -(LogNormalization + x * x / 2);
    internal static double Pdf0(double x) => Normalization * Math.Exp(-x * x / 2);

    /// <inheritdoc />
    public override string Name => "StandardNormal";
    /// <inheritdoc />
    protected override double LogPdfFormula(double x) => LogPdf0(x);
    /// <inheritdoc />
    protected override double PdfFormula(double x) => Pdf0(x);
    /// <inheritdoc />
    protected override double LogCdfFormula(double x) => Special.LogNdtr(x);
    /// <inheritdoc />
    protected override double CdfFormula(double x) => Special.Ndtr(x);
    /// <inheritdoc />
    protected override double LogCcdfFormula(double x) => Special.LogNdtr(-x);
    /// <inheritdoc />
    protected override double CcdfFormula(double x) => Special.Ndtr(-x);
    /// <inheritdoc />
    protected override double IcdfFormula(double p) => Special.Ndtri(p);
    /// <inheritdoc />
    protected override double ILogCdfFormula(double logp) => StatsDistMath.NdtriExp(logp);
    /// <inheritdoc />
    protected override double IccdfFormula(double p) => -Special.Ndtri(p);
    /// <inheritdoc />
    protected override double ILogCcdfFormula(double logp) => -StatsDistMath.NdtriExp(logp);
    /// <inheritdoc />
    protected override double EntropyFormula() => Entropy0;
    /// <inheritdoc />
    protected override Complex LogEntropyFormula() => new(Special.Log1p(Math.Log(2 * Math.PI)) - Math.Log(2), 0);
    /// <inheritdoc />
    protected override double MedianFormula() => 0.0;
    /// <inheritdoc />
    protected override double ModeFormula() => 0.0;

    private static double? Raw(int order) => order switch { 0 => 1, 1 => 0, 2 => 1, 3 => 0, 4 => 3, 5 => 0, _ => null };
    /// <inheritdoc />
    protected override double? MomentRawFormula(int order) => Raw(order);
    /// <inheritdoc />
    protected override double? MomentCentralFormula(int order) => Raw(order);
    /// <inheritdoc />
    protected override double? MomentStandardizedFormula(int order) => Raw(order);

    /// <inheritdoc />
    protected override void SampleFormula(Span<double> destination, GraviRandom rng)
    {
        for (var i = 0; i < destination.Length; i++) destination[i] = StatsRandom.Normal(rng);
    }

    /// <inheritdoc />
    public override string ToString() => "StandardNormal()";
}

/// <summary>The uniform distribution on [a, b] (scipy's <c>stats.Uniform</c>).</summary>
public sealed class Uniform : ContinuousRandomVariable
{
    private readonly double _ab;

    /// <summary>Uniform on [<paramref name="a"/>, <paramref name="b"/>], a &lt; b.</summary>
    public Uniform(double a = 0.0, double b = 1.0)
    {
        if (!double.IsFinite(a) || !double.IsFinite(b) || !(a < b)) throw new ArgumentException("Uniform requires finite a < b.");
        A = a;
        B = b;
        _ab = b - a;
    }

    /// <summary>The lower end.</summary>
    public double A { get; }
    /// <summary>The upper end.</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "Uniform";
    /// <inheritdoc />
    protected internal override (double Lower, double Upper) SupportCore() => (A, B);
    /// <inheritdoc />
    protected internal override (bool Lower, bool Upper) SupportInclusive => (true, true);
    /// <inheritdoc />
    protected override double LogPdfFormula(double x) => double.IsNaN(x) ? double.NaN : -Math.Log(_ab);
    /// <inheritdoc />
    protected override double PdfFormula(double x) => double.IsNaN(x) ? double.NaN : 1 / _ab;
    /// <inheritdoc />
    protected override double LogCdfFormula(double x) => Math.Log(x - A) - Math.Log(_ab);
    /// <inheritdoc />
    protected override double CdfFormula(double x) => (x - A) / _ab;
    /// <inheritdoc />
    protected override double LogCcdfFormula(double x) => Math.Log(B - x) - Math.Log(_ab);
    /// <inheritdoc />
    protected override double CcdfFormula(double x) => (B - x) / _ab;
    /// <inheritdoc />
    protected override double IcdfFormula(double p) => A + _ab * p;
    /// <inheritdoc />
    protected override double IccdfFormula(double p) => B - _ab * p;
    /// <inheritdoc />
    protected override double EntropyFormula() => Math.Log(_ab);
    /// <inheritdoc />
    protected override double ModeFormula() => A + 0.5 * _ab;
    /// <inheritdoc />
    protected override double MedianFormula() => A + 0.5 * _ab;
    /// <inheritdoc />
    protected override double? MomentRawFormula(int order)
    {
        var np1 = order + 1;
        return (Math.Pow(B, np1) - Math.Pow(A, np1)) / (np1 * _ab);
    }
    /// <inheritdoc />
    protected override double? MomentCentralFormula(int order) => order == 2 ? _ab * _ab / 12 : null;

    /// <inheritdoc />
    protected override void SampleFormula(Span<double> destination, GraviRandom rng)
    {
        for (var i = 0; i < destination.Length; i++) destination[i] = A + _ab * rng.NextDouble();
    }

    /// <inheritdoc />
    public override string ToString() => $"Uniform(a={PyFormat.F(A)}, b={PyFormat.F(B)})";
}

/// <summary>The standard logistic distribution (scipy's <c>stats.Logistic</c>, which has no parameters;
/// shift and scale it with arithmetic).</summary>
public sealed class Logistic : ContinuousRandomVariable
{
    private static readonly double s_scale = Math.PI / Math.Sqrt(3);

    /// <inheritdoc />
    public override string Name => "Logistic";
    /// <inheritdoc />
    protected internal override (double Lower, double Upper) SupportCore() => (double.NegativeInfinity, double.PositiveInfinity);
    /// <inheritdoc />
    protected override double LogPdfFormula(double x)
    {
        var y = -Math.Abs(x);
        return y - 2 * Special.Log1p(Math.Exp(y));
    }
    /// <inheritdoc />
    protected override double PdfFormula(double x)
    {
        var c = 0.5 / Math.Cosh(x / 2);
        return c * c;
    }
    /// <inheritdoc />
    protected override double LogCdfFormula(double x) => Special.LogExpit(x);
    /// <inheritdoc />
    protected override double CdfFormula(double x) => Special.Expit(x);
    /// <inheritdoc />
    protected override double LogCcdfFormula(double x) => Special.LogExpit(-x);
    /// <inheritdoc />
    protected override double CcdfFormula(double x) => Special.Expit(-x);
    /// <inheritdoc />
    protected override double IcdfFormula(double p) => Special.Logit(p);
    /// <inheritdoc />
    protected override double IccdfFormula(double p) => -Special.Logit(p);
    /// <inheritdoc />
    protected override double EntropyFormula() => 2.0;
    /// <inheritdoc />
    protected override Complex LogEntropyFormula() => new(Math.Log(2), 0);
    /// <inheritdoc />
    protected override double MedianFormula() => 0.0;
    /// <inheritdoc />
    protected override double ModeFormula() => 0.0;

    /// <summary>E[Xⁿ] = πⁿ |(2ⁿ − 2) Bₙ| for even n (Bₙ the Bernoulli numbers), 0 for odd n.</summary>
    protected override double? MomentRawFormula(int order)
    {
        if (order % 2 == 1) return 0.0;
        if (order == 0) return 1.0;
        return Math.Pow(Math.PI, order) * Math.Abs((Math.Pow(2, order) - 2) * BernoulliEven(order));
    }

    /// <inheritdoc />
    protected override double? MomentCentralFormula(int order) => MomentRawFormula(order);
    /// <inheritdoc />
    protected override double? MomentStandardizedFormula(int order) => MomentRawFormula(order) / Math.Pow(s_scale, order);

    /// <summary>|B_n| for even n ≥ 2: 2 n! ζ(n) / (2π)ⁿ.</summary>
    private static double BernoulliEven(int n)
    {
        var logFact = Special.GammaLn(n + 1);
        return 2 * Math.Exp(logFact - n * Math.Log(2 * Math.PI)) * Special.Zeta(n);
    }

    /// <inheritdoc />
    protected override void SampleFormula(Span<double> destination, GraviRandom rng)
    {
        for (var i = 0; i < destination.Length; i++) destination[i] = Special.Logit(StatsDistMath.OpenUniform(rng));
    }

    /// <inheritdoc />
    public override string ToString() => "Logistic()";
}

/// <summary>
/// The binomial distribution with <see cref="N"/> trials and success probability <see cref="P"/>
/// (scipy's <c>stats.Binomial</c>). The mass function, CDFs and quantiles are those of SciNet's
/// <see cref="BinomialDistribution"/>; the log mass uses scipy's log-gamma formula.
/// </summary>
public sealed class Binomial : DiscreteRandomVariable
{
    private readonly BinomialDistribution _d;

    /// <summary>Binomial with <paramref name="n"/> ≥ 1 trials and success probability 0 &lt; <paramref name="p"/> &lt; 1.</summary>
    public Binomial(int n, double p)
    {
        if (n < 1) throw new ArgumentException("n must be a positive integer.", nameof(n));
        if (!(p > 0 && p < 1)) throw new ArgumentException("p must lie strictly between 0 and 1.", nameof(p));
        N = n;
        P = p;
        _d = new BinomialDistribution(n, p);
    }

    /// <summary>Number of trials.</summary>
    public int N { get; }
    /// <summary>Success probability.</summary>
    public double P { get; }

    /// <inheritdoc />
    public override string Name => "Binomial";
    /// <inheritdoc />
    protected internal override (double Lower, double Upper) SupportCore() => (0.0, N);
    /// <inheritdoc />
    protected override double PmfFormula(double x) => _d.Pmf(x);
    /// <inheritdoc />
    protected override double LogPmfFormula(double x)
    {
        var combiln = Special.GammaLn(N + 1) - (Special.GammaLn(x + 1) + Special.GammaLn(N - x + 1));
        return combiln + Special.Xlogy(x, P) + Special.Xlog1py(N - x, -P);
    }
    /// <inheritdoc />
    protected override double CdfFormula(double x) => _d.Cdf(Math.Floor(x));
    /// <inheritdoc />
    protected override double LogCdfFormula(double x)
        => x < IcdfFormula(0.5) ? Math.Log(_d.Cdf(Math.Floor(x))) : Special.Log1p(-_d.Sf(Math.Floor(x)));
    /// <inheritdoc />
    protected override double CcdfFormula(double x) => _d.Sf(Math.Floor(x));
    /// <inheritdoc />
    protected override double LogCcdfFormula(double x)
        => x < IcdfFormula(0.5) ? Special.Log1p(-_d.Cdf(Math.Floor(x))) : Math.Log(_d.Sf(Math.Floor(x)));
    /// <inheritdoc />
    protected override double IcdfFormula(double p) => _d.Ppf(p);
    /// <inheritdoc />
    protected override double IccdfFormula(double p) => _d.Isf(p);
    /// <inheritdoc />
    protected override double ModeFormula()
    {
        var mode = Math.Floor((N + 1) * P);
        return P == 1 ? mode - 1 : mode;
    }
    /// <inheritdoc />
    protected override double? MomentRawFormula(int order) => order switch
    {
        1 => N * P,
        2 => N * P * (1 - P + N * P),
        _ => null,
    };
    /// <inheritdoc />
    protected override double? MomentCentralFormula(int order) => order switch
    {
        1 => 0.0,
        2 => N * P * (1 - P),
        3 => N * P * (1 - P) * (1 - 2 * P),
        4 => N * P * (1 - P) * (1 + (3.0 * N - 6) * P * (1 - P)),
        _ => null,
    };

    /// <inheritdoc />
    public override string ToString() => $"Binomial(n={PyFormat.F(N)}, p={PyFormat.F(P)})";
}
