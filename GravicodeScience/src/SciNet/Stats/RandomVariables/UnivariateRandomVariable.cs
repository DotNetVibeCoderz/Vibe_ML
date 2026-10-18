using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// How a quantity of a random variable is computed — scipy's <c>method=</c> strings
/// (<c>'formula'</c>, <c>'log/exp'</c>, <c>'complement'</c>, <c>'quadrature'</c>, …).
/// </summary>
/// <remarks>Passing a method that does not apply to a function (for example
/// <see cref="Inversion"/> to <c>Pdf</c>) throws <see cref="ArgumentException"/>; asking for a
/// formula the distribution does not have throws <see cref="NotSupportedException"/>.</remarks>
public enum RandomVariableMethod
{
    /// <summary>The distribution's own closed form (<c>'formula'</c>).</summary>
    Formula,
    /// <summary>The exponential or logarithm of the log/linear counterpart (<c>'log/exp'</c>).</summary>
    LogExp,
    /// <summary>One minus the complementary function, or its log1mexp (<c>'complement'</c>).</summary>
    Complement,
    /// <summary>Numerical integration (tanh-sinh) or summation over the support (<c>'quadrature'</c>).</summary>
    Quadrature,
    /// <summary>Root finding on the forward function (<c>'inversion'</c>).</summary>
    Inversion,
    /// <summary>Difference of two one-argument CDFs, for the two-argument CDF (<c>'subtraction'</c>).</summary>
    Subtraction,
    /// <summary>Sum of CDF and CCDF, for the two-argument CCDF (<c>'addition'</c>).</summary>
    Addition,
    /// <summary>The median as the 0.5 quantile (<c>'icdf'</c>).</summary>
    Icdf,
    /// <summary>The mode by maximising the density (<c>'optimization'</c>).</summary>
    Optimization,
    /// <summary>A moment already computed (<c>'cache'</c>).</summary>
    Cache,
    /// <summary>A moment from moments of another kind (<c>'transform'</c>).</summary>
    Transform,
    /// <summary>A central moment from a standardized one and the variance, or vice versa (<c>'normalize'</c>).</summary>
    Normalize,
    /// <summary>Moments that are the same for every distribution (orders 0, 1, 2) (<c>'general'</c>).</summary>
    General,
    /// <summary>A moment by integrating the inverse CDF over (0, 1) (<c>'quadrature_icdf'</c>).</summary>
    QuadratureIcdf,
    /// <summary>Sampling by applying the inverse CDF to uniforms (<c>'inverse_transform'</c>).</summary>
    InverseTransform,
}

/// <summary>Which moment <see cref="UnivariateRandomVariable.Moment"/> returns (scipy's <c>kind=</c>).</summary>
public enum MomentKind
{
    /// <summary>E[Xⁿ], about the origin.</summary>
    Raw,
    /// <summary>E[(X − μ)ⁿ], about the mean.</summary>
    Central,
    /// <summary>E[(X − μ)ⁿ]/σⁿ.</summary>
    Standardized,
}

/// <summary>Kurtosis convention (scipy's <c>convention=</c>; the new framework defaults to non-excess).</summary>
public enum KurtosisConvention
{
    /// <summary>E[(X − μ)⁴]/σ⁴, 3 for the normal (scipy's default here).</summary>
    NonExcess,
    /// <summary>Non-excess minus 3, zero for the normal.</summary>
    Excess,
}

/// <summary>
/// The overridable "formula" hooks of a random variable. A hook counts as available when the
/// subclass overrides the corresponding <c>…Formula</c> method — the test scipy makes with
/// <c>_overrides</c>, and the input to every default-method decision.
/// </summary>
public enum RandomVariableFormula
{
    /// <summary><c>PdfFormula</c>.</summary>
    Pdf,
    /// <summary><c>LogPdfFormula</c>.</summary>
    LogPdf,
    /// <summary><c>PmfFormula</c>.</summary>
    Pmf,
    /// <summary><c>LogPmfFormula</c>.</summary>
    LogPmf,
    /// <summary><c>CdfFormula</c>.</summary>
    Cdf,
    /// <summary><c>LogCdfFormula</c>.</summary>
    LogCdf,
    /// <summary><c>CcdfFormula</c>.</summary>
    Ccdf,
    /// <summary><c>LogCcdfFormula</c>.</summary>
    LogCcdf,
    /// <summary><c>Cdf2Formula</c> (two-argument CDF).</summary>
    Cdf2,
    /// <summary><c>LogCdf2Formula</c>.</summary>
    LogCdf2,
    /// <summary><c>Ccdf2Formula</c>.</summary>
    Ccdf2,
    /// <summary><c>LogCcdf2Formula</c>.</summary>
    LogCcdf2,
    /// <summary><c>IcdfFormula</c>.</summary>
    Icdf,
    /// <summary><c>IccdfFormula</c>.</summary>
    Iccdf,
    /// <summary><c>ILogCdfFormula</c>.</summary>
    ILogCdf,
    /// <summary><c>ILogCcdfFormula</c>.</summary>
    ILogCcdf,
    /// <summary><c>EntropyFormula</c>.</summary>
    Entropy,
    /// <summary><c>LogEntropyFormula</c>.</summary>
    LogEntropy,
    /// <summary><c>MedianFormula</c>.</summary>
    Median,
    /// <summary><c>ModeFormula</c>.</summary>
    Mode,
    /// <summary><c>SampleFormula</c>.</summary>
    Sample,
}

/// <summary>Where variates come from: a pseudo-random generator, or uniforms already drawn from a QMC engine.</summary>
internal sealed class SampleSource
{
    public SampleSource(GraviRandom rng) { Rng = rng; }
    public SampleSource(double[] uniforms) { Uniforms = uniforms; }
    public GraviRandom? Rng { get; }
    public double[]? Uniforms { get; }
    public bool IsQmc => Uniforms is not null;
}

[Flags]
internal enum MomentMethods
{
    None = 0,
    Cache = 1,
    Formula = 2,
    Transform = 4,
    Normalize = 8,
    General = 16,
    Quadrature = 32,
    QuadratureIcdf = 64,
    /// <summary>Ignore cached values that came from quadrature (used when rebuilding one kind from another).</summary>
    ExactOnly = 128,
    All = Cache | Formula | Transform | Normalize | General | Quadrature,
}

/// <summary>
/// A univariate random variable in scipy's new distribution framework (scipy 1.15+,
/// <c>scipy.stats.ContinuousDistribution</c>/<c>DiscreteDistribution</c>, here
/// <see cref="ContinuousRandomVariable"/> and <see cref="DiscreteRandomVariable"/>).
/// </summary>
/// <remarks>
/// <para>
/// Every public function (<see cref="Pdf(double, RandomVariableMethod?)"/>, <see cref="Cdf(double, RandomVariableMethod?)"/>,
/// <see cref="Icdf(double, RandomVariableMethod?)"/>, <see cref="Entropy"/>, <see cref="Moment"/>, …)
/// chooses how to compute its value the way scipy does: a formula if the subclass overrides the
/// matching <c>…Formula</c> hook; otherwise the log/exp of a formula for the logarithm (or the
/// reverse); otherwise the complement of the complementary function, with scipy's "safe"
/// fall-back to quadrature where the subtraction would lose every digit; otherwise numerical
/// integration of the density, root finding on the CDF, or optimisation of the density. The
/// optional <c>method</c> argument overrides that choice, as scipy's <c>method=</c> does.
/// </para>
/// <para>
/// Differences from scipy: parameters are scalars (one distribution per object, no broadcasting
/// over parameter arrays), and invalid parameters throw on construction instead of producing NaN
/// results. Arguments outside the support are handled as scipy's <c>_set_invalid_nan</c> does —
/// the density is 0, the CDF 0 or 1, the inverse CDF at 0 and 1 is the end of the support.
/// </para>
/// <para>
/// The class also implements <see cref="IUnivariateDistribution"/> (explicitly), so a random
/// variable can be passed to the goodness-of-fit tests and resampling functions. Through that
/// interface the kurtosis is the excess kurtosis, as for the frozen <c>rv_continuous</c> types.
/// </para>
/// </remarks>
public abstract class UnivariateRandomVariable : IUnivariateDistribution
{
    private static readonly ConcurrentDictionary<Type, bool[]> s_declared = new();
    private readonly RandomVariableMethod?[] _methodCache = new RandomVariableMethod?[(int)Fn.Count];
    private readonly Dictionary<int, double> _rawCache = [];
    private readonly Dictionary<int, double> _centralCache = [];
    private readonly Dictionary<int, double> _standardizedCache = [];
    // Orders whose cached moment came from quadrature. scipy's "transform" methods rebuild a moment
    // of one kind from cached moments of the other, so with quadrature results in the cache the
    // answer depends on call order: a truncated normal's 6th central moment, rebuilt from raw
    // moments of size 2e4, kept only seven digits. Those entries are not used for transforms here.
    private readonly HashSet<int> _rawNumeric = [];
    private readonly HashSet<int> _centralNumeric = [];

    internal enum Fn
    {
        Pdf, LogPdf, Pmf, LogPmf, Cdf, LogCdf, Ccdf, LogCcdf, Cdf2, LogCdf2, Ccdf2, LogCcdf2,
        Icdf, Iccdf, ILogCdf, ILogCcdf, Entropy, LogEntropy, Median, Mode, Count,
    }

    /// <summary>
    /// Relative tolerance of the numerical methods (scipy's <c>tol</c>): passed to the quadrature
    /// and the root finder, and the threshold of the "safe" complement checks. Null means each
    /// algorithm's default (eps^0.75 for quadrature, √eps for the complement checks).
    /// </summary>
    public double? Tol
    {
        get => _tol;
        set
        {
            if (value is { } t && !(t >= 0)) throw new ArgumentException("tol must be a non-negative number.", nameof(value));
            _tol = value;
            Array.Clear(_methodCache);
        }
    }
    private double? _tol;

    /// <summary>True for a distribution on the integers.</summary>
    public abstract bool IsDiscrete { get; }

    /// <summary>The distribution's name as scipy prints it (e.g. <c>"Normal"</c>).</summary>
    public virtual string Name => GetType().Name;

    // ================================================================== support

    /// <summary>The support [lower, upper]; either end may be infinite.</summary>
    public (double Lower, double Upper) Support() => SupportCore();

    /// <summary>The support of this distribution (scipy's <c>_support</c>).</summary>
    protected internal abstract (double Lower, double Upper) SupportCore();

    /// <summary>
    /// Whether the ends of the support belong to it. It decides only whether the density is
    /// evaluated (rather than set to zero) exactly at an end, as scipy's domain <c>inclusive</c> flags do.
    /// </summary>
    protected internal virtual (bool Lower, bool Upper) SupportInclusive => (false, false);

    // ================================================================== formula hooks

    /// <summary>Whether the distribution has its own formula for <paramref name="formula"/>.
    /// Default: whether the subclass overrides the corresponding <c>…Formula</c> method.</summary>
    internal virtual bool Overrides(RandomVariableFormula formula) => Declared(GetType())[(int)formula];

    private static bool[] Declared(Type type) => s_declared.GetOrAdd(type, static t =>
    {
        var names = Enum.GetNames<RandomVariableFormula>();
        var result = new bool[names.Length];
        for (var i = 0; i < names.Length; i++)
        {
            var m = t.GetMethod(names[i] + "Formula", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            result[i] = m is not null && m.DeclaringType != typeof(UnivariateRandomVariable);
        }
        return result;
    });

    private NotSupportedException NoFormula(string what) => new($"{Name} has no formula for {what}.");

    /// <summary>Probability density at x.</summary>
    protected virtual double PdfFormula(double x) => throw NoFormula("pdf");
    /// <summary>Log of the probability density at x.</summary>
    protected virtual double LogPdfFormula(double x) => throw NoFormula("logpdf");
    /// <summary>Probability mass at x.</summary>
    protected virtual double PmfFormula(double x) => throw NoFormula("pmf");
    /// <summary>Log of the probability mass at x.</summary>
    protected virtual double LogPmfFormula(double x) => throw NoFormula("logpmf");
    /// <summary>P(X ≤ x).</summary>
    protected virtual double CdfFormula(double x) => throw NoFormula("cdf");
    /// <summary>log P(X ≤ x).</summary>
    protected virtual double LogCdfFormula(double x) => throw NoFormula("logcdf");
    /// <summary>P(X &gt; x).</summary>
    protected virtual double CcdfFormula(double x) => throw NoFormula("ccdf");
    /// <summary>log P(X &gt; x).</summary>
    protected virtual double LogCcdfFormula(double x) => throw NoFormula("logccdf");
    /// <summary>P(x ≤ X ≤ y) for x ≤ y.</summary>
    protected virtual double Cdf2Formula(double x, double y) => throw NoFormula("two-argument cdf");
    /// <summary>log P(x ≤ X ≤ y) for x ≤ y.</summary>
    protected virtual double LogCdf2Formula(double x, double y) => throw NoFormula("two-argument logcdf");
    /// <summary>P(X &lt; x or X &gt; y) for x ≤ y.</summary>
    protected virtual double Ccdf2Formula(double x, double y) => throw NoFormula("two-argument ccdf");
    /// <summary>log P(X &lt; x or X &gt; y) for x ≤ y.</summary>
    protected virtual double LogCcdf2Formula(double x, double y) => throw NoFormula("two-argument logccdf");
    /// <summary>Inverse CDF at p in (0, 1).</summary>
    protected virtual double IcdfFormula(double p) => throw NoFormula("icdf");
    /// <summary>Inverse CCDF at p in (0, 1).</summary>
    protected virtual double IccdfFormula(double p) => throw NoFormula("iccdf");
    /// <summary>The x with log CDF(x) = logp.</summary>
    protected virtual double ILogCdfFormula(double logp) => throw NoFormula("ilogcdf");
    /// <summary>The x with log CCDF(x) = logp.</summary>
    protected virtual double ILogCcdfFormula(double logp) => throw NoFormula("ilogccdf");
    /// <summary>Differential (or Shannon) entropy in nats.</summary>
    protected virtual double EntropyFormula() => throw NoFormula("entropy");
    /// <summary>Log of the entropy (complex when the entropy is negative).</summary>
    protected virtual Complex LogEntropyFormula() => throw NoFormula("logentropy");
    /// <summary>The median.</summary>
    protected virtual double MedianFormula() => throw NoFormula("median");
    /// <summary>The mode.</summary>
    protected virtual double ModeFormula() => throw NoFormula("mode");
    /// <summary>Fills <paramref name="destination"/> with variates.</summary>
    protected virtual void SampleFormula(Span<double> destination, GraviRandom rng) => throw NoFormula("sampling");
    /// <summary>The raw moment of the given order, or null if there is no formula for it.</summary>
    protected virtual double? MomentRawFormula(int order) => null;
    /// <summary>The central moment of the given order, or null.</summary>
    protected virtual double? MomentCentralFormula(int order) => null;
    /// <summary>The standardized moment of the given order, or null.</summary>
    protected virtual double? MomentStandardizedFormula(int order) => null;

    // ================================================================== dispatch helpers

    private RandomVariableMethod Choose(Fn fn, RandomVariableMethod? method, Func<RandomVariableMethod> pick)
    {
        if (method is { } m) return m;
        return _methodCache[(int)fn] ??= pick();
    }

    private static ArgumentException BadMethod(RandomVariableMethod m, string function)
        => new($"Method {m} is not available for {function}.");

    private double SolveTol => _tol ?? 0;
    private double SafeTol => _tol ?? Math.Sqrt(RandomVariableNumerics.Eps);

    internal double Quadrature(Func<double, double> f, double a, double b)
        => IsDiscrete ? RandomVariableNumerics.Sum(f, a, b) : RandomVariableNumerics.Quad(f, a, b, _tol);

    internal double LogQuadrature(Func<double, double> logf, double a, double b)
        => IsDiscrete ? RandomVariableNumerics.LogSum(logf, a, b) : RandomVariableNumerics.LogQuad(logf, a, b, _tol);

    /// <summary>pdf for continuous, pmf for discrete distributions (scipy's <c>_pxf_dispatch</c>).</summary>
    internal abstract double PxfDispatch(double x, RandomVariableMethod? method = null);

    /// <summary>log pdf or log pmf.</summary>
    internal abstract double LogPxfDispatch(double x, RandomVariableMethod? method = null);

    // ------------------------------------------------------------------ densities

    internal virtual double LogPdfDispatch(double x, RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.LogPdf, method, () => Overrides(RandomVariableFormula.LogPdf) ? RandomVariableMethod.Formula : RequireLinear(RandomVariableFormula.Pdf, RandomVariableMethod.LogExp));
        return m switch
        {
            RandomVariableMethod.Formula => LogPdfFormula(x),
            RandomVariableMethod.LogExp => Math.Log(PdfDispatch(x)),
            _ => throw BadMethod(m, "logpdf"),
        };
    }

    internal virtual double PdfDispatch(double x, RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.Pdf, method, () => Overrides(RandomVariableFormula.Pdf) ? RandomVariableMethod.Formula : RequireLinear(RandomVariableFormula.LogPdf, RandomVariableMethod.LogExp));
        return m switch
        {
            RandomVariableMethod.Formula => PdfFormula(x),
            RandomVariableMethod.LogExp => Math.Exp(LogPdfDispatch(x)),
            _ => throw BadMethod(m, "pdf"),
        };
    }

    internal virtual double LogPmfDispatch(double x, RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.LogPmf, method, () => Overrides(RandomVariableFormula.LogPmf) ? RandomVariableMethod.Formula : RequireLinear(RandomVariableFormula.Pmf, RandomVariableMethod.LogExp));
        return m switch
        {
            RandomVariableMethod.Formula => LogPmfFormula(x),
            RandomVariableMethod.LogExp => Math.Log(PmfDispatch(x)),
            _ => throw BadMethod(m, "logpmf"),
        };
    }

    internal virtual double PmfDispatch(double x, RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.Pmf, method, () => Overrides(RandomVariableFormula.Pmf) ? RandomVariableMethod.Formula : RequireLinear(RandomVariableFormula.LogPmf, RandomVariableMethod.LogExp));
        return m switch
        {
            RandomVariableMethod.Formula => PmfFormula(x),
            RandomVariableMethod.LogExp => Math.Exp(LogPmfDispatch(x)),
            _ => throw BadMethod(m, "pmf"),
        };
    }

    // pdf ⇄ logpdf would recurse forever when neither is defined; fail loudly instead.
    private RandomVariableMethod RequireLinear(RandomVariableFormula other, RandomVariableMethod result)
        => Overrides(other) || IsTransformed ? result : throw new NotSupportedException($"{Name} defines neither {other} nor its logarithm.");

    /// <summary>True for distributions that override the dispatch layer themselves (transforms).</summary>
    internal virtual bool IsTransformed => false;

    // ------------------------------------------------------------------ two-argument CDFs

    internal virtual double LogCdf2Dispatch(double x, double y, RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.LogCdf2, method, () =>
            Overrides(RandomVariableFormula.LogCdf2) ? RandomVariableMethod.Formula
            : Overrides(RandomVariableFormula.LogCdf) || Overrides(RandomVariableFormula.LogCcdf) ? RandomVariableMethod.Subtraction
            : Overrides(RandomVariableFormula.Cdf) || Overrides(RandomVariableFormula.Ccdf) ? (RandomVariableMethod)SafeLogExp
            : RandomVariableMethod.Quadrature);
        return m switch
        {
            RandomVariableMethod.Formula => LogCdf2Formula(x, y),
            RandomVariableMethod.Subtraction => LogCdf2Subtraction(x, y),
            RandomVariableMethod.LogExp => Math.Log(Math.Abs(Cdf2Dispatch(x, y))),
            SafeLogExp => LogCdf2LogExpSafe(x, y),
            RandomVariableMethod.Quadrature => LogQuadrature(t => LogPxfDispatch(t), x, y),
            _ => throw BadMethod(m, "logcdf(x, y)"),
        };
    }

    // Private method codes for scipy's "_safe" variants (not user-selectable).
    private const RandomVariableMethod SafeLogExp = (RandomVariableMethod)1001;
    private const RandomVariableMethod SafeComplement = (RandomVariableMethod)1002;
    private const RandomVariableMethod SafeSubtraction = (RandomVariableMethod)1003;

    private double LogCdf2Subtraction(double x, double y)
    {
        if (x > y) (x, y) = (y, x);
        var lcx = LogCdfDispatch(x);
        var lcy = LogCdfDispatch(y);
        var lccx = LogCcdfDispatch(x);
        var lccy = LogCcdfDispatch(y);
        if (lcx < -1 && lcy < -1) return RandomVariableNumerics.LogExpXmExpY(lcy, lcx).Real;
        if (lccx < -1 && lccy < -1) return RandomVariableNumerics.LogExpXmExpY(lccx, lccy).Real;
        return StatsDistMath.Log1mExp(StatsDistMath.LogAddExp(lcx, lccy));
    }

    private double LogCdf2LogExpSafe(double x, double y)
    {
        var r = Math.Log(Math.Abs(Cdf2Dispatch(x, y)));
        return double.IsInfinity(r) ? LogQuadrature(t => LogPxfDispatch(t), x, y) : r;
    }

    internal virtual double Cdf2Dispatch(double x, double y, RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.Cdf2, method, () =>
            Overrides(RandomVariableFormula.Cdf2) ? RandomVariableMethod.Formula
            : Overrides(RandomVariableFormula.LogCdf) || Overrides(RandomVariableFormula.LogCcdf) ? RandomVariableMethod.LogExp
            : Overrides(RandomVariableFormula.Cdf) || Overrides(RandomVariableFormula.Ccdf) ? SafeSubtraction
            : RandomVariableMethod.Quadrature);
        return m switch
        {
            RandomVariableMethod.Formula => Cdf2Formula(x, y),
            RandomVariableMethod.LogExp => Math.Exp(LogCdf2Dispatch(x, y)),
            RandomVariableMethod.Subtraction => Cdf2Subtraction(x, y, safe: false),
            SafeSubtraction => Cdf2Subtraction(x, y, safe: true),
            RandomVariableMethod.Quadrature => Quadrature(t => PxfDispatch(t), x, y),
            _ => throw BadMethod(m, "cdf(x, y)"),
        };
    }

    private double Cdf2Subtraction(double x, double y, bool safe)
    {
        var cx = CdfDispatch(x);
        var cy = CdfDispatch(y);
        var ccx = CcdfDispatch(x);
        var ccy = CcdfDispatch(y);
        var upper = ccx < 0.5 && ccy < 0.5;
        var r = upper ? ccx - ccy : cy - cx;
        if (!safe) return r;
        var spacing = RandomVariableNumerics.Spacing(upper ? Math.Max(ccx, ccy) : Math.Max(cx, cy));
        if (Math.Abs(SafeTol * r) < spacing) return Quadrature(t => PxfDispatch(t), x, y);
        return r;
    }

    internal virtual double LogCcdf2Dispatch(double x, double y, RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.LogCcdf2, method, () => Overrides(RandomVariableFormula.LogCcdf2) ? RandomVariableMethod.Formula : RandomVariableMethod.Addition);
        return m switch
        {
            RandomVariableMethod.Formula => LogCcdf2Formula(x, y),
            RandomVariableMethod.Addition => StatsDistMath.LogAddExp(LogCdfDispatch(x), LogCcdfDispatch(y)),
            _ => throw BadMethod(m, "logccdf(x, y)"),
        };
    }

    internal virtual double Ccdf2Dispatch(double x, double y, RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.Ccdf2, method, () => Overrides(RandomVariableFormula.Ccdf2) ? RandomVariableMethod.Formula : RandomVariableMethod.Addition);
        return m switch
        {
            RandomVariableMethod.Formula => Ccdf2Formula(x, y),
            RandomVariableMethod.Addition => CdfDispatch(x) + CcdfDispatch(y),
            _ => throw BadMethod(m, "ccdf(x, y)"),
        };
    }

    // ------------------------------------------------------------------ one-argument CDFs

    internal virtual double LogCdfDispatch(double x, RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.LogCdf, method, () =>
            Overrides(RandomVariableFormula.LogCdf) ? RandomVariableMethod.Formula
            : Overrides(RandomVariableFormula.LogCcdf) ? RandomVariableMethod.Complement
            : Overrides(RandomVariableFormula.Cdf) ? SafeLogExp
            : RandomVariableMethod.Quadrature);
        return m switch
        {
            RandomVariableMethod.Formula => LogCdfFormula(x),
            RandomVariableMethod.Complement => StatsDistMath.Log1mExp(LogCcdfDispatch(x)),
            RandomVariableMethod.LogExp => Math.Log(CdfDispatch(x)),
            SafeLogExp => LogExpSafe(Math.Log(CdfDispatch(x)), () => LogCdfQuadrature(x)),
            RandomVariableMethod.Quadrature => LogCdfQuadrature(x),
            _ => throw BadMethod(m, "logcdf"),
        };
    }

    private static double LogExpSafe(double value, Func<double> fallback) => double.IsInfinity(value) ? fallback() : value;

    internal virtual double LogCdfQuadrature(double x) => LogQuadrature(t => LogPxfDispatch(t), SupportCore().Lower, x);
    internal virtual double LogCcdfQuadrature(double x) => LogQuadrature(t => LogPxfDispatch(t), x, SupportCore().Upper);
    internal virtual double CdfQuadrature(double x) => Quadrature(t => PxfDispatch(t), SupportCore().Lower, x);
    internal virtual double CcdfQuadrature(double x) => Quadrature(t => PxfDispatch(t), x, SupportCore().Upper);

    internal virtual double CdfDispatch(double x, RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.Cdf, method, () =>
            Overrides(RandomVariableFormula.Cdf) ? RandomVariableMethod.Formula
            : Overrides(RandomVariableFormula.LogCdf) ? RandomVariableMethod.LogExp
            : Overrides(RandomVariableFormula.Ccdf) ? SafeComplement
            : RandomVariableMethod.Quadrature);
        return m switch
        {
            RandomVariableMethod.Formula => CdfFormula(x),
            RandomVariableMethod.LogExp => Math.Exp(LogCdfDispatch(x)),
            RandomVariableMethod.Complement => 1 - CcdfDispatch(x),
            SafeComplement => ComplementSafe(CcdfDispatch(x), () => CdfQuadrature(x)),
            RandomVariableMethod.Quadrature => CdfQuadrature(x),
            _ => throw BadMethod(m, "cdf"),
        };
    }

    private double ComplementSafe(double other, Func<double> fallback)
    {
        var r = 1 - other;
        return SafeTol * r < RandomVariableNumerics.Spacing(other) ? fallback() : r;
    }

    internal virtual double LogCcdfDispatch(double x, RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.LogCcdf, method, () =>
            Overrides(RandomVariableFormula.LogCcdf) ? RandomVariableMethod.Formula
            : Overrides(RandomVariableFormula.LogCdf) ? RandomVariableMethod.Complement
            : Overrides(RandomVariableFormula.Ccdf) ? SafeLogExp
            : RandomVariableMethod.Quadrature);
        return m switch
        {
            RandomVariableMethod.Formula => LogCcdfFormula(x),
            RandomVariableMethod.Complement => StatsDistMath.Log1mExp(LogCdfDispatch(x)),
            RandomVariableMethod.LogExp => Math.Log(CcdfDispatch(x)),
            SafeLogExp => LogExpSafe(Math.Log(CcdfDispatch(x)), () => LogCcdfQuadrature(x)),
            RandomVariableMethod.Quadrature => LogCcdfQuadrature(x),
            _ => throw BadMethod(m, "logccdf"),
        };
    }

    internal virtual double CcdfDispatch(double x, RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.Ccdf, method, () =>
            Overrides(RandomVariableFormula.Ccdf) ? RandomVariableMethod.Formula
            : Overrides(RandomVariableFormula.LogCcdf) ? RandomVariableMethod.LogExp
            : Overrides(RandomVariableFormula.Cdf) ? SafeComplement
            : RandomVariableMethod.Quadrature);
        return m switch
        {
            RandomVariableMethod.Formula => CcdfFormula(x),
            RandomVariableMethod.LogExp => Math.Exp(LogCcdfDispatch(x)),
            RandomVariableMethod.Complement => 1 - CdfDispatch(x),
            SafeComplement => ComplementSafe(CdfDispatch(x), () => CcdfQuadrature(x)),
            RandomVariableMethod.Quadrature => CcdfQuadrature(x),
            _ => throw BadMethod(m, "ccdf"),
        };
    }

    // ------------------------------------------------------------------ inverse functions

    internal virtual double ILogCdfDispatch(double logp, RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.ILogCdf, method, () =>
            Overrides(RandomVariableFormula.ILogCdf) ? RandomVariableMethod.Formula
            : Overrides(RandomVariableFormula.ILogCcdf) ? RandomVariableMethod.Complement
            : RandomVariableMethod.Inversion);
        return m switch
        {
            RandomVariableMethod.Formula => ILogCdfFormula(logp),
            RandomVariableMethod.Complement => ILogCcdfDispatch(StatsDistMath.Log1mExp(logp)),
            RandomVariableMethod.Inversion => InvertLogCdf(logp),
            _ => throw BadMethod(m, "ilogcdf"),
        };
    }

    internal virtual double IcdfDispatch(double p, RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.Icdf, method, () =>
            Overrides(RandomVariableFormula.Icdf) ? RandomVariableMethod.Formula
            : Overrides(RandomVariableFormula.Iccdf) ? SafeComplement
            : RandomVariableMethod.Inversion);
        return m switch
        {
            RandomVariableMethod.Formula => IcdfFormula(p),
            RandomVariableMethod.Complement => IccdfDispatch(1 - p),
            SafeComplement => SafeTol * p < RandomVariableNumerics.Spacing(1 - p) ? InvertCdf(p) : IccdfDispatch(1 - p),
            RandomVariableMethod.Inversion => InvertCdf(p),
            _ => throw BadMethod(m, "icdf"),
        };
    }

    internal virtual double ILogCcdfDispatch(double logp, RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.ILogCcdf, method, () =>
            Overrides(RandomVariableFormula.ILogCcdf) ? RandomVariableMethod.Formula
            : Overrides(RandomVariableFormula.ILogCdf) ? RandomVariableMethod.Complement
            : RandomVariableMethod.Inversion);
        return m switch
        {
            RandomVariableMethod.Formula => ILogCcdfFormula(logp),
            RandomVariableMethod.Complement => ILogCdfDispatch(StatsDistMath.Log1mExp(logp)),
            RandomVariableMethod.Inversion => InvertLogCcdf(logp),
            _ => throw BadMethod(m, "ilogccdf"),
        };
    }

    internal virtual double IccdfDispatch(double p, RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.Iccdf, method, () =>
            Overrides(RandomVariableFormula.Iccdf) ? RandomVariableMethod.Formula
            : Overrides(RandomVariableFormula.Icdf) ? SafeComplement
            : RandomVariableMethod.Inversion);
        return m switch
        {
            RandomVariableMethod.Formula => IccdfFormula(p),
            RandomVariableMethod.Complement => IcdfDispatch(1 - p),
            SafeComplement => SafeTol * p < RandomVariableNumerics.Spacing(1 - p) ? InvertCcdf(p) : IcdfDispatch(1 - p),
            RandomVariableMethod.Inversion => InvertCcdf(p),
            _ => throw BadMethod(m, "iccdf"),
        };
    }

    /// <summary>Inverse of the CDF by root finding (discrete: the smallest k with CDF(k) ≥ p).</summary>
    internal virtual double InvertCdf(double p)
    {
        var (a, b) = SupportCore();
        return RandomVariableNumerics.Solve(x => CdfDispatch(x), p, a, b, increasing: true);
    }

    internal virtual double InvertCcdf(double p)
    {
        var (a, b) = SupportCore();
        return RandomVariableNumerics.Solve(x => CcdfDispatch(x), p, a, b, increasing: false);
    }

    internal virtual double InvertLogCdf(double logp)
    {
        var (a, b) = SupportCore();
        return RandomVariableNumerics.Solve(x => LogCdfDispatch(x), logp, a, b, increasing: true);
    }

    internal virtual double InvertLogCcdf(double logp)
    {
        var (a, b) = SupportCore();
        return RandomVariableNumerics.Solve(x => LogCcdfDispatch(x), logp, a, b, increasing: false);
    }

    // ------------------------------------------------------------------ properties

    internal virtual Complex LogEntropyDispatch(RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.LogEntropy, method, () =>
            Overrides(RandomVariableFormula.LogEntropy) ? RandomVariableMethod.Formula
            : Overrides(RandomVariableFormula.Entropy) ? SafeLogExp
            : RandomVariableMethod.Quadrature);
        return m switch
        {
            RandomVariableMethod.Formula => LogEntropyFormula(),
            RandomVariableMethod.LogExp => RandomVariableNumerics.LogReal(EntropyDispatch()),
            SafeLogExp => LogEntropySafe(),
            RandomVariableMethod.Quadrature => RandomVariableNumerics.LogReal(EntropyQuadrature()),
            _ => throw BadMethod(m, "logentropy"),
        };
    }

    private Complex LogEntropySafe()
    {
        var r = RandomVariableNumerics.LogReal(EntropyDispatch());
        return double.IsInfinity(r.Real) ? RandomVariableNumerics.LogReal(EntropyQuadrature()) : r;
    }

    internal virtual double EntropyDispatch(RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.Entropy, method, () =>
            Overrides(RandomVariableFormula.Entropy) ? RandomVariableMethod.Formula
            : Overrides(RandomVariableFormula.LogEntropy) ? RandomVariableMethod.LogExp
            : RandomVariableMethod.Quadrature);
        return m switch
        {
            RandomVariableMethod.Formula => EntropyFormula(),
            RandomVariableMethod.LogExp => Complex.Exp(LogEntropyDispatch()).Real,
            RandomVariableMethod.Quadrature => EntropyQuadrature(),
            _ => throw BadMethod(m, "entropy"),
        };
    }

    /// <summary>
    /// Points inside the support where an integrand over the whole support should be split: the
    /// median by default. scipy integrates over the support in one piece, which misses an
    /// integrable singularity or a jump in the interior (X**3 has both at 0, its median) by 3%.
    /// </summary>
    internal virtual IEnumerable<double> QuadratureBreakpoints()
    {
        double m;
        try { m = MedianDispatch(); }
        catch (NotSupportedException) { yield break; }
        catch (ArgumentException) { yield break; }
        yield return m;
    }

    /// <summary>Quadrature (or summation) of f over the whole support, split at <see cref="QuadratureBreakpoints"/>.</summary>
    internal double SupportQuadrature(Func<double, double> f)
    {
        var (a, b) = SupportCore();
        if (IsDiscrete) return Quadrature(f, a, b);
        return RandomVariableNumerics.QuadSplit(f, a, b, QuadratureBreakpoints(), _tol);
    }

    private double EntropyQuadrature()
    {
        return SupportQuadrature(x =>
        {
            var p = PxfDispatch(x);
            if (p == 0 || double.IsNaN(p)) return 0.0;
            return -p * LogPxfDispatch(x);
        });
    }

    internal virtual double MedianDispatch(RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.Median, method, () => Overrides(RandomVariableFormula.Median) ? RandomVariableMethod.Formula : RandomVariableMethod.Icdf);
        return m switch
        {
            RandomVariableMethod.Formula => MedianFormula(),
            RandomVariableMethod.Icdf => IcdfDispatch(0.5),
            _ => throw BadMethod(m, "median"),
        };
    }

    internal virtual double ModeDispatch(RandomVariableMethod? method = null)
    {
        var m = Choose(Fn.Mode, method, () => Overrides(RandomVariableFormula.Mode) ? RandomVariableMethod.Formula : RandomVariableMethod.Optimization);
        return m switch
        {
            RandomVariableMethod.Formula => ModeFormula(),
            RandomVariableMethod.Optimization => ModeOptimization(),
            _ => throw BadMethod(m, "mode"),
        };
    }

    /// <summary>The mode by maximising the density from the median outwards (scipy's <c>_mode_optimization</c>).</summary>
    internal virtual double ModeOptimization()
    {
        var (a, b) = SupportCore();
        var med = MedianDispatch();
        return RandomVariableNumerics.Minimize(x => -PxfDispatch(x), med, a, b);
    }

    // ------------------------------------------------------------------ sampling

    internal virtual void SampleDispatch(Span<double> destination, SampleSource source, RandomVariableMethod? method = null)
    {
        var m = method ?? (Overrides(RandomVariableFormula.Sample) && !source.IsQmc ? RandomVariableMethod.Formula : RandomVariableMethod.InverseTransform);
        switch (m)
        {
            case RandomVariableMethod.Formula:
                if (source.Rng is null) throw new ArgumentException("A formula sampler needs a pseudo-random generator, not a QMC engine.");
                SampleFormula(destination, source.Rng);
                break;
            case RandomVariableMethod.InverseTransform:
                for (var i = 0; i < destination.Length; i++)
                {
                    var u = source.Uniforms is { } us ? us[i] : StatsDistMath.OpenUniform(source.Rng!);
                    destination[i] = IcdfDispatch(u);
                }
                break;
            default:
                throw BadMethod(m, "sample");
        }
    }

    // ================================================================== moments

    private static MomentMethods ToMomentMethods(RandomVariableMethod? method) => method switch
    {
        null => MomentMethods.All,
        RandomVariableMethod.Cache => MomentMethods.Cache,
        RandomVariableMethod.Formula => MomentMethods.Formula,
        RandomVariableMethod.Transform => MomentMethods.Transform,
        RandomVariableMethod.Normalize => MomentMethods.Normalize,
        RandomVariableMethod.General => MomentMethods.General,
        RandomVariableMethod.Quadrature => MomentMethods.Quadrature,
        RandomVariableMethod.QuadratureIcdf => MomentMethods.QuadratureIcdf,
        _ => throw BadMethod(method.Value, "moment"),
    };

    private const MomentMethods ExactOnly = MomentMethods.ExactOnly;

    internal virtual double? MomentRawDispatch(int order, MomentMethods methods)
    {
        double? moment = null;
        if (methods.HasFlag(MomentMethods.Cache) && _rawCache.TryGetValue(order, out var c)
            && !(methods.HasFlag(ExactOnly) && _rawNumeric.Contains(order))) moment = c;
        if (moment is null && methods.HasFlag(MomentMethods.Formula)) moment = MomentRawFormula(order);
        if (moment is null && methods.HasFlag(MomentMethods.Transform) && order > 1) moment = MomentRawTransform(order);
        if (moment is null && methods.HasFlag(MomentMethods.General)) moment = order == 0 ? 1.0 : null;
        var numeric = false;
        if (moment is null && methods.HasFlag(MomentMethods.Quadrature)) { moment = MomentFromPxf(order, 0.0); numeric = true; }
        if (moment is null && methods.HasFlag(MomentMethods.QuadratureIcdf)) { moment = MomentIntegrateIcdf(order, 0.0); numeric = true; }
        if (moment is { } v && !_rawCache.ContainsKey(order)) { _rawCache[order] = v; if (numeric) _rawNumeric.Add(order); }
        return moment;
    }

    private double? MomentRawTransform(int order)
    {
        var central = new double[order + 1];
        for (var i = 0; i <= order; i++)
        {
            var mi = MomentCentralDispatch(i, ExactOnly | MomentMethods.Cache | MomentMethods.Formula | MomentMethods.Normalize | MomentMethods.General);
            if (mi is null) return null;
            central[i] = mi.Value;
        }
        var mean = MomentRawDispatch(1, MomentMethods.Cache | MomentMethods.Formula | MomentMethods.Quadrature);
        if (mean is null) return null;
        return TransformCenter(order, central, mean.Value, 0.0);
    }

    internal virtual double? MomentCentralDispatch(int order, MomentMethods methods)
    {
        double? moment = null;
        if (methods.HasFlag(MomentMethods.Cache) && _centralCache.TryGetValue(order, out var c)
            && !(methods.HasFlag(ExactOnly) && _centralNumeric.Contains(order))) moment = c;
        if (moment is null && methods.HasFlag(MomentMethods.Formula)) moment = MomentCentralFormula(order);
        if (moment is null && methods.HasFlag(MomentMethods.Transform)) moment = MomentCentralTransform(order);
        if (moment is null && methods.HasFlag(MomentMethods.Normalize) && order > 2) moment = MomentCentralNormalize(order);
        if (moment is null && methods.HasFlag(MomentMethods.General)) moment = order switch { 0 => 1.0, 1 => 0.0, _ => null };
        var numeric = false;
        if (moment is null && methods.HasFlag(MomentMethods.Quadrature))
        {
            var mean = MomentRawDispatch(1, MomentMethods.All);
            moment = mean is { } mu ? MomentFromPxf(order, mu) : null;
            numeric = true;
        }
        if (moment is null && methods.HasFlag(MomentMethods.QuadratureIcdf))
        {
            var mean = MomentRawDispatch(1, MomentMethods.All);
            moment = mean is { } mu ? MomentIntegrateIcdf(order, mu) : null;
            numeric = true;
        }
        if (moment is { } v && !_centralCache.ContainsKey(order)) { _centralCache[order] = v; if (numeric) _centralNumeric.Add(order); }
        return moment;
    }

    private double? MomentCentralTransform(int order)
    {
        var raw = new double[order + 1];
        for (var i = 0; i <= order; i++)
        {
            var mi = MomentRawDispatch(i, ExactOnly | MomentMethods.Cache | MomentMethods.Formula | MomentMethods.General);
            if (mi is null) return null;
            raw[i] = mi.Value;
        }
        var mean = MomentRawDispatch(1, MomentMethods.All);
        if (mean is null) return null;
        return TransformCenter(order, raw, 0.0, mean.Value);
    }

    private double? MomentCentralNormalize(int order)
    {
        var standard = MomentStandardizedDispatch(order, MomentMethods.Cache | MomentMethods.Formula | MomentMethods.General);
        if (standard is null) return null;
        var variance = MomentCentralDispatch(2, MomentMethods.All);
        if (variance is null) return null;
        return standard.Value * Math.Pow(variance.Value, order / 2.0);
    }

    internal virtual double? MomentStandardizedDispatch(int order, MomentMethods methods)
    {
        double? moment = null;
        if (methods.HasFlag(MomentMethods.Cache) && _standardizedCache.TryGetValue(order, out var c)) moment = c;
        if (moment is null && methods.HasFlag(MomentMethods.Formula)) moment = MomentStandardizedFormula(order);
        if (moment is null && methods.HasFlag(MomentMethods.Normalize)) moment = MomentStandardizedNormalize(order, false);
        if (moment is null && methods.HasFlag(MomentMethods.General)) moment = order switch { 0 => 1.0, 1 => 0.0, 2 => 1.0, _ => null };
        if (moment is null && methods.HasFlag(MomentMethods.Normalize)) moment = MomentStandardizedNormalize(order, true);
        if (moment is { } v) _standardizedCache[order] = v;
        return moment;
    }

    private double? MomentStandardizedNormalize(int order, bool useQuadrature)
    {
        var methods = useQuadrature ? MomentMethods.Quadrature : MomentMethods.Cache | MomentMethods.Formula | MomentMethods.Transform;
        var central = MomentCentralDispatch(order, methods);
        if (central is null) return null;
        var variance = MomentCentralDispatch(2, MomentMethods.All);
        if (variance is null) return null;
        return central.Value / Math.Pow(variance.Value, order / 2.0);
    }

    private double MomentFromPxf(int order, double center)
        => SupportQuadrature(x =>
        {
            var p = PxfDispatch(x);
            return p == 0 ? 0.0 : p * Math.Pow(x - center, order);
        });

    private double MomentIntegrateIcdf(int order, double center)
        => RandomVariableNumerics.Quad(u => Math.Pow(IcdfDispatch(u) - center, order), 0.0, 1.0, _tol);

    /// <summary>Moments about b from the moments <paramref name="momentsAboutA"/> (orders 0..n) about a:
    /// Σᵢ C(n, i) mᵢ (a − b)ⁿ⁻ⁱ (scipy's <c>_moment_transform_center</c>).</summary>
    internal static double TransformCenter(int order, double[] momentsAboutA, double a, double b)
    {
        var total = 0.0;
        for (var i = 0; i <= order; i++) total += Special.Binom(order, i) * momentsAboutA[i] * Math.Pow(a - b, order - i);
        return total;
    }

    // ================================================================== public API

    private bool IsInSupportForDensity(double x)
    {
        var (lo, hi) = SupportCore();
        var (li, ri) = SupportInclusive;
        var low = li ? x < lo : x <= lo;
        var high = ri ? x > hi : x >= hi;
        return !(low || high);
    }

    /// <summary>Probability density function (for a discrete distribution: +∞ at the support points, as scipy).</summary>
    public double Pdf(double x, RandomVariableMethod? method = null)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (!IsInSupportForDensity(x)) return 0.0;
        if (IsDiscrete && x != Math.Floor(x)) return 0.0;
        return PdfDispatch(x, method);
    }

    /// <summary>Log of the probability density function.</summary>
    public double LogPdf(double x, RandomVariableMethod? method = null)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (!IsInSupportForDensity(x)) return double.NegativeInfinity;
        if (IsDiscrete && x != Math.Floor(x)) return double.NegativeInfinity;
        return LogPdfDispatch(x, method);
    }

    /// <summary>Probability mass function (zero everywhere for a continuous distribution, as scipy).</summary>
    public double Pmf(double x, RandomVariableMethod? method = null)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (!IsInSupportForDensity(x)) return 0.0;
        if (IsDiscrete && x != Math.Floor(x)) return 0.0;
        return PmfDispatch(x, method);
    }

    /// <summary>Log of the probability mass function.</summary>
    public double LogPmf(double x, RandomVariableMethod? method = null)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (!IsInSupportForDensity(x)) return double.NegativeInfinity;
        if (IsDiscrete && x != Math.Floor(x)) return double.NegativeInfinity;
        return LogPmfDispatch(x, method);
    }

    /// <summary>Cumulative distribution function P(X ≤ x).</summary>
    public double Cdf(double x, RandomVariableMethod? method = null)
    {
        if (double.IsNaN(x)) return double.NaN;
        var (lo, hi) = SupportCore();
        if (IsDiscrete ? x < lo : x <= lo) return 0.0;
        if (x >= hi) return 1.0;
        return Math.Clamp(CdfDispatch(x, method), 0.0, 1.0);
    }

    /// <summary>Log of the CDF.</summary>
    public double LogCdf(double x, RandomVariableMethod? method = null)
    {
        if (double.IsNaN(x)) return double.NaN;
        var (lo, hi) = SupportCore();
        if (IsDiscrete ? x < lo : x <= lo) return double.NegativeInfinity;
        if (x >= hi) return 0.0;
        return Math.Min(LogCdfDispatch(x, method), 0.0);
    }

    /// <summary>Complementary CDF P(X &gt; x) (scipy's <c>ccdf</c>, the survival function).</summary>
    public double Ccdf(double x, RandomVariableMethod? method = null)
    {
        if (double.IsNaN(x)) return double.NaN;
        var (lo, hi) = SupportCore();
        if (IsDiscrete ? x < lo : x <= lo) return 1.0;
        if (x >= hi) return 0.0;
        return Math.Clamp(CcdfDispatch(x, method), 0.0, 1.0);
    }

    /// <summary>Log of the complementary CDF.</summary>
    public double LogCcdf(double x, RandomVariableMethod? method = null)
    {
        if (double.IsNaN(x)) return double.NaN;
        var (lo, hi) = SupportCore();
        if (IsDiscrete ? x < lo : x <= lo) return 0.0;
        if (x >= hi) return double.NegativeInfinity;
        return Math.Min(LogCcdfDispatch(x, method), 0.0);
    }

    private (double X, double Y, bool Swapped) Cdf2Arguments(double x, double y)
    {
        if (IsDiscrete) throw new NotSupportedException("Two argument cdf functions are currently only supported for continuous distributions.");
        var (lo, hi) = SupportCore();
        var swapped = y < x;
        if (swapped) (x, y) = (y, x);
        x = Math.Clamp(x, lo, hi);
        y = Math.Clamp(y, lo, hi);
        return (x, y, swapped);
    }

    /// <summary>P(x ≤ X ≤ y); negative when y &lt; x, as scipy's two-argument <c>cdf</c>.</summary>
    public double Cdf(double x, double y, RandomVariableMethod? method = null)
    {
        if (double.IsNaN(x) || double.IsNaN(y)) return double.NaN;
        var (a, b, swapped) = Cdf2Arguments(x, y);
        var r = Math.Clamp(Cdf2Dispatch(a, b, method), 0.0, 1.0);
        return swapped ? -r : r;
    }

    /// <summary>log P(x ≤ X ≤ y); complex (imaginary part π) when y &lt; x, where the "probability" is negative.</summary>
    public Complex LogCdf(double x, double y, RandomVariableMethod? method = null)
    {
        if (double.IsNaN(x) || double.IsNaN(y)) return new Complex(double.NaN, 0);
        var (a, b, swapped) = Cdf2Arguments(x, y);
        var r = Math.Min(LogCdf2Dispatch(a, b, method), 0.0);
        return swapped ? new Complex(r, Math.PI) : new Complex(r, 0);
    }

    /// <summary>1 − P(x ≤ X ≤ y), the mass outside [x, y] (2 − that when y &lt; x, as scipy).</summary>
    public double Ccdf(double x, double y, RandomVariableMethod? method = null)
    {
        if (double.IsNaN(x) || double.IsNaN(y)) return double.NaN;
        var (a, b, swapped) = Cdf2Arguments(x, y);
        var r = Math.Clamp(Ccdf2Dispatch(a, b, method), 0.0, 1.0);
        return swapped ? 2 - r : r;
    }

    /// <summary>Log of the two-argument complementary CDF.</summary>
    public double LogCcdf(double x, double y, RandomVariableMethod? method = null)
    {
        if (double.IsNaN(x) || double.IsNaN(y)) return double.NaN;
        var (a, b, swapped) = Cdf2Arguments(x, y);
        var r = Math.Min(LogCcdf2Dispatch(a, b, method), 0.0);
        return swapped ? RandomVariableNumerics.LogExpXmExpY(Math.Log(2), r).Real : r;
    }

    /// <summary>Inverse of the CDF (quantile function); 0 and 1 map to the ends of the support.</summary>
    public double Icdf(double p, RandomVariableMethod? method = null)
    {
        if (!(p >= 0 && p <= 1)) return double.NaN;
        var (lo, hi) = SupportCore();
        if (p == 0) return lo;
        if (p == 1) return hi;
        return IcdfDispatch(p, method);
    }

    /// <summary>Inverse of the complementary CDF.</summary>
    public double Iccdf(double p, RandomVariableMethod? method = null)
    {
        if (!(p >= 0 && p <= 1)) return double.NaN;
        var (lo, hi) = SupportCore();
        if (p == 0) return hi;
        if (p == 1) return lo;
        return IccdfDispatch(p, method);
    }

    /// <summary>Inverse of the log CDF: the x with log CDF(x) = <paramref name="logp"/> (≤ 0).</summary>
    public double ILogCdf(double logp, RandomVariableMethod? method = null)
    {
        if (!(logp <= 0)) return double.NaN;
        var (lo, hi) = SupportCore();
        if (double.IsNegativeInfinity(logp)) return lo;
        if (logp == 0) return hi;
        return ILogCdfDispatch(logp, method);
    }

    /// <summary>Inverse of the log complementary CDF.</summary>
    public double ILogCcdf(double logp, RandomVariableMethod? method = null)
    {
        if (!(logp <= 0)) return double.NaN;
        var (lo, hi) = SupportCore();
        if (double.IsNegativeInfinity(logp)) return hi;
        if (logp == 0) return lo;
        return ILogCcdfDispatch(logp, method);
    }

    /// <summary>Differential entropy (continuous) or Shannon entropy (discrete), in nats.</summary>
    public double Entropy(RandomVariableMethod? method = null) => EntropyDispatch(method);

    /// <summary>Log of the entropy; complex with imaginary part π when the entropy is negative.</summary>
    public Complex LogEntropy(RandomVariableMethod? method = null) => LogEntropyDispatch(method);

    /// <summary>The median.</summary>
    public double Median(RandomVariableMethod? method = null) => MedianDispatch(method);

    /// <summary>The mode (the location of the density's maximum).</summary>
    public double Mode(RandomVariableMethod? method = null) => ModeDispatch(method);

    /// <summary>The mean, <c>Moment(1, Raw)</c>.</summary>
    public double Mean(RandomVariableMethod? method = null) => Moment(1, MomentKind.Raw, method);

    /// <summary>The variance, <c>Moment(2, Central)</c>.</summary>
    public double Variance(RandomVariableMethod? method = null) => Moment(2, MomentKind.Central, method);

    /// <summary>The standard deviation.</summary>
    public double StandardDeviation(RandomVariableMethod? method = null) => Math.Sqrt(Variance(method));

    /// <summary>The skewness, <c>Moment(3, Standardized)</c>.</summary>
    public double Skewness(RandomVariableMethod? method = null) => Moment(3, MomentKind.Standardized, method);

    /// <summary>The kurtosis, non-excess by default as in scipy's new framework (3 for the normal).</summary>
    public double Kurtosis(KurtosisConvention convention = KurtosisConvention.NonExcess, RandomVariableMethod? method = null)
    {
        var k = Moment(4, MomentKind.Standardized, method);
        return convention == KurtosisConvention.Excess ? k - 3 : k;
    }

    /// <summary>
    /// A moment of the given order and kind. The methods are tried in scipy's order — cache,
    /// formula, transform from another kind, normalisation, the general values for orders 0–2,
    /// and quadrature — and the first that applies is used; <paramref name="method"/> restricts the
    /// search to one of them.
    /// </summary>
    public double Moment(int order = 1, MomentKind kind = MomentKind.Raw, RandomVariableMethod? method = null)
    {
        if (order < 0) throw new ArgumentException("Argument `order` of `moment` must be a finite, positive integer.", nameof(order));
        var methods = ToMomentMethods(method);
        var r = kind switch
        {
            MomentKind.Raw => MomentRawDispatch(order, methods),
            MomentKind.Central => MomentCentralDispatch(order, methods),
            MomentKind.Standardized => MomentStandardizedDispatch(order, methods),
            _ => throw new ArgumentException("Unknown moment kind.", nameof(kind)),
        };
        return r ?? throw new NotSupportedException($"The requested moment of {Name} is not available with the given method.");
    }

    /// <summary>One variate.</summary>
    public double Sample(GraviRandom? rng = null, RandomVariableMethod? method = null)
    {
        Span<double> one = stackalloc double[1];
        SampleDispatch(one, new SampleSource(rng ?? StatsDistributions.MakeRng(null)), method);
        return one[0];
    }

    /// <summary><paramref name="n"/> variates.</summary>
    public double[] Sample(int n, GraviRandom? rng = null, RandomVariableMethod? method = null)
    {
        if (n < 0) throw new ArgumentException("n must be non-negative.", nameof(n));
        var r = new double[n];
        SampleDispatch(r, new SampleSource(rng ?? StatsDistributions.MakeRng(null)), method);
        return r;
    }

    /// <summary>Variates in an array of the given shape (scipy's <c>sample(shape)</c>).</summary>
    public NdArray Sample(int[] shape, GraviRandom? rng = null, RandomVariableMethod? method = null)
    {
        ArgumentNullException.ThrowIfNull(shape);
        var n = 1;
        foreach (var d in shape) { if (d < 0) throw new ArgumentException("Negative dimension.", nameof(shape)); n *= d; }
        return new NdArray(Sample(n, rng, method), (int[])shape.Clone());
    }

    /// <summary>
    /// Quasi-random variates by inverse transform of a low-discrepancy sequence (scipy's
    /// <c>sample(shape, rng=qmc_engine)</c>). Each column of the result is one low-discrepancy
    /// sequence along axis 0, so the engine's dimension must equal the product of
    /// <c>shape[1..]</c> (1 for a one-dimensional shape).
    /// </summary>
    public NdArray Sample(int[] shape, QmcEngine qrng)
    {
        ArgumentNullException.ThrowIfNull(shape); ArgumentNullException.ThrowIfNull(qrng);
        if (shape.Length == 0) shape = [1];
        var rows = shape[0];
        var cols = 1;
        for (var i = 1; i < shape.Length; i++) cols *= shape[i];
        if (qrng.D != cols) throw new ArgumentException($"The QMC engine must have dimension {cols} (the product of shape[1..]), got {qrng.D}.", nameof(qrng));
        var u = qrng.Random(rows).AsContiguous().ToArray();
        var r = new double[u.Length];
        SampleDispatch(r, new SampleSource(u), null);
        return new NdArray(r, (int[])shape.Clone());
    }

    // ------------------------------------------------------------------ vectorised conveniences

    /// <summary>Element-wise <see cref="Pdf(double, RandomVariableMethod?)"/>.</summary>
    public double[] Pdf(double[] x, RandomVariableMethod? method = null) => Map(x, v => Pdf(v, method));
    /// <summary>Element-wise <see cref="LogPdf(double, RandomVariableMethod?)"/>.</summary>
    public double[] LogPdf(double[] x, RandomVariableMethod? method = null) => Map(x, v => LogPdf(v, method));
    /// <summary>Element-wise <see cref="Pmf(double, RandomVariableMethod?)"/>.</summary>
    public double[] Pmf(double[] x, RandomVariableMethod? method = null) => Map(x, v => Pmf(v, method));
    /// <summary>Element-wise <see cref="Cdf(double, RandomVariableMethod?)"/>.</summary>
    public double[] Cdf(double[] x, RandomVariableMethod? method = null) => Map(x, v => Cdf(v, method));
    /// <summary>Element-wise <see cref="LogCdf(double, RandomVariableMethod?)"/>.</summary>
    public double[] LogCdf(double[] x, RandomVariableMethod? method = null) => Map(x, v => LogCdf(v, method));
    /// <summary>Element-wise <see cref="Ccdf(double, RandomVariableMethod?)"/>.</summary>
    public double[] Ccdf(double[] x, RandomVariableMethod? method = null) => Map(x, v => Ccdf(v, method));
    /// <summary>Element-wise <see cref="LogCcdf(double, RandomVariableMethod?)"/>.</summary>
    public double[] LogCcdf(double[] x, RandomVariableMethod? method = null) => Map(x, v => LogCcdf(v, method));
    /// <summary>Element-wise <see cref="Icdf(double, RandomVariableMethod?)"/>.</summary>
    public double[] Icdf(double[] p, RandomVariableMethod? method = null) => Map(p, v => Icdf(v, method));
    /// <summary>Element-wise <see cref="Iccdf(double, RandomVariableMethod?)"/>.</summary>
    public double[] Iccdf(double[] p, RandomVariableMethod? method = null) => Map(p, v => Iccdf(v, method));

    private static double[] Map(double[] x, Func<double, double> f)
    {
        ArgumentNullException.ThrowIfNull(x);
        var r = new double[x.Length];
        for (var i = 0; i < x.Length; i++) r[i] = f(x[i]);
        return r;
    }

    // ================================================================== IUnivariateDistribution

    string IUnivariateDistribution.Name => Name;
    double IUnivariateDistribution.Density(double x) => IsDiscrete ? Pmf(x) : Pdf(x);
    double IUnivariateDistribution.LogDensity(double x) => IsDiscrete ? LogPmf(x) : LogPdf(x);
    double IUnivariateDistribution.Cdf(double x) => Cdf(x);
    double IUnivariateDistribution.LogCdf(double x) => LogCdf(x);
    double IUnivariateDistribution.Sf(double x) => Ccdf(x);
    double IUnivariateDistribution.LogSf(double x) => LogCcdf(x);
    double IUnivariateDistribution.Ppf(double q) => Icdf(q);
    double IUnivariateDistribution.Isf(double q) => Iccdf(q);
    double IUnivariateDistribution.Mean() => Mean();
    double IUnivariateDistribution.Var() => Variance();
    double IUnivariateDistribution.Std() => StandardDeviation();
    double IUnivariateDistribution.Median() => Median();
    double IUnivariateDistribution.Skew() => Skewness();
    double IUnivariateDistribution.Kurtosis() => Kurtosis(KurtosisConvention.Excess);
    DistributionMoments IUnivariateDistribution.Stats() => new(Mean(), Variance(), Skewness(), Kurtosis(KurtosisConvention.Excess));
    double IUnivariateDistribution.Moment(int n) => Moment(n);
    double IUnivariateDistribution.Entropy() => Entropy();
    (double Lower, double Upper) IUnivariateDistribution.Interval(double confidence)
    {
        if (!(confidence >= 0 && confidence <= 1)) throw new ArgumentException("confidence must lie in [0, 1].", nameof(confidence));
        var a = 0.5 * (1 - confidence);
        return (Icdf(a), Iccdf(a));
    }
    double IUnivariateDistribution.Rvs(GraviRandom rng) => Sample(rng);
    double[] IUnivariateDistribution.Rvs(int size, GraviRandom rng) => Sample(size, rng);
}
