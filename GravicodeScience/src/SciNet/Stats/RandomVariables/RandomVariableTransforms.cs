using System.Globalization;
using System.Numerics;

namespace Gravicode.Science.SciNet;

/// <summary>Formatting of parameters the way Python prints floats (<c>2.0</c>, not <c>2</c>).</summary>
internal static class PyFormat
{
    public static string F(double x)
    {
        if (double.IsNaN(x)) return "nan";
        if (double.IsPositiveInfinity(x)) return "inf";
        if (double.IsNegativeInfinity(x)) return "-inf";
        if (x == Math.Floor(x) && Math.Abs(x) < 1e16) return x.ToString("F1", CultureInfo.InvariantCulture);
        return x.ToString("R", CultureInfo.InvariantCulture);
    }
}

/// <summary>
/// A distribution defined by transforming another one (scipy's <c>TransformedDistribution</c>).
/// It overrides the dispatch layer directly, so the inner distribution's own method choices
/// (formula, quadrature, …) carry through the transformation.
/// </summary>
public abstract class TransformedDistribution : ContinuousRandomVariable
{
    /// <summary>Wraps the distribution being transformed.</summary>
    protected TransformedDistribution(ContinuousRandomVariable inner)
    {
        ArgumentNullException.ThrowIfNull(inner);
        Inner = inner;
    }

    /// <summary>The distribution being transformed.</summary>
    public ContinuousRandomVariable Inner { get; }

    internal override bool IsTransformed => true;

    internal override bool Overrides(RandomVariableFormula formula) => Inner.Overrides(formula) || base.Overrides(formula);

    /// <inheritdoc />
    protected internal override (bool Lower, bool Upper) SupportInclusive => Inner.SupportInclusive;
}

/// <summary>
/// loc + scale·X (scipy's <c>ShiftedScaledDistribution</c>), what <c>X + c</c>, <c>c * X</c>,
/// <c>X / c</c> and <c>-X</c> produce. A negative scale swaps CDF and CCDF; shifting or scaling
/// it again folds into the same loc and scale rather than nesting.
/// </summary>
public sealed class ShiftedScaledDistribution : TransformedDistribution
{
    private readonly bool _sign;

    /// <summary>The distribution of <paramref name="loc"/> + <paramref name="scale"/>·X.</summary>
    public ShiftedScaledDistribution(ContinuousRandomVariable x, double loc = 0.0, double scale = 1.0) : base(x)
    {
        if (!double.IsFinite(loc)) throw new ArgumentException("loc must be finite.", nameof(loc));
        if (!double.IsFinite(scale) || scale == 0) throw new ArgumentException("scale must be finite and non-zero.", nameof(scale));
        Loc = loc;
        Scale = scale;
        _sign = scale > 0;
    }

    /// <summary>The shift.</summary>
    public double Loc { get; }

    /// <summary>The scale (negative values reflect).</summary>
    public double Scale { get; }

    private double T(double x) => (x - Loc) / Scale;
    private double IT(double x) => x * Scale + Loc;

    internal override ContinuousRandomVariable AddLoc(double loc) => new ShiftedScaledDistribution(Inner, Loc + loc, Scale);
    internal override ContinuousRandomVariable SubtractLoc(double loc) => new ShiftedScaledDistribution(Inner, Loc - loc, Scale);
    internal override ContinuousRandomVariable MultiplyScale(double scale) => new ShiftedScaledDistribution(Inner, Loc * scale, Scale * scale);
    internal override ContinuousRandomVariable DivideScale(double scale) => new ShiftedScaledDistribution(Inner, Loc / scale, Scale / scale);

    /// <inheritdoc />
    protected internal override (double Lower, double Upper) SupportCore()
    {
        var (a, b) = Inner.SupportCore();
        var (ta, tb) = (IT(a), IT(b));
        return _sign ? (ta, tb) : (tb, ta);
    }

    /// <inheritdoc />
    protected internal override (bool Lower, bool Upper) SupportInclusive
    {
        get { var (l, r) = Inner.SupportInclusive; return _sign ? (l, r) : (r, l); }
    }

    internal override double EntropyDispatch(RandomVariableMethod? method = null) => Inner.EntropyDispatch(method) + Math.Log(Math.Abs(Scale));

    internal override Complex LogEntropyDispatch(RandomVariableMethod? method = null)
    {
        var h0 = Complex.Exp(Inner.LogEntropyDispatch(method)).Real;
        return RandomVariableNumerics.LogReal(h0 + Math.Log(Math.Abs(Scale)));
    }

    internal override double MedianDispatch(RandomVariableMethod? method = null) => IT(Inner.MedianDispatch(method));
    internal override double ModeDispatch(RandomVariableMethod? method = null) => IT(Inner.ModeDispatch(method));

    internal override double LogPdfDispatch(double x, RandomVariableMethod? method = null) => Inner.LogPdfDispatch(T(x), method) - Math.Log(Math.Abs(Scale));
    internal override double PdfDispatch(double x, RandomVariableMethod? method = null) => Inner.PdfDispatch(T(x), method) / Math.Abs(Scale);

    internal override double LogCdfDispatch(double x, RandomVariableMethod? method = null) => _sign ? Inner.LogCdfDispatch(T(x), method) : Inner.LogCcdfDispatch(T(x), method);
    internal override double CdfDispatch(double x, RandomVariableMethod? method = null) => _sign ? Inner.CdfDispatch(T(x), method) : Inner.CcdfDispatch(T(x), method);
    internal override double LogCcdfDispatch(double x, RandomVariableMethod? method = null) => _sign ? Inner.LogCcdfDispatch(T(x), method) : Inner.LogCdfDispatch(T(x), method);
    internal override double CcdfDispatch(double x, RandomVariableMethod? method = null) => _sign ? Inner.CcdfDispatch(T(x), method) : Inner.CdfDispatch(T(x), method);

    internal override double LogCdf2Dispatch(double x, double y, RandomVariableMethod? method = null) => _sign ? Inner.LogCdf2Dispatch(T(x), T(y), method) : Inner.LogCdf2Dispatch(T(y), T(x), method);
    internal override double Cdf2Dispatch(double x, double y, RandomVariableMethod? method = null) => _sign ? Inner.Cdf2Dispatch(T(x), T(y), method) : Inner.Cdf2Dispatch(T(y), T(x), method);
    internal override double LogCcdf2Dispatch(double x, double y, RandomVariableMethod? method = null) => _sign ? Inner.LogCcdf2Dispatch(T(x), T(y), method) : Inner.LogCcdf2Dispatch(T(y), T(x), method);
    internal override double Ccdf2Dispatch(double x, double y, RandomVariableMethod? method = null) => _sign ? Inner.Ccdf2Dispatch(T(x), T(y), method) : Inner.Ccdf2Dispatch(T(y), T(x), method);

    internal override double ILogCdfDispatch(double logp, RandomVariableMethod? method = null) => IT(_sign ? Inner.ILogCdfDispatch(logp, method) : Inner.ILogCcdfDispatch(logp, method));
    internal override double IcdfDispatch(double p, RandomVariableMethod? method = null) => IT(_sign ? Inner.IcdfDispatch(p, method) : Inner.IccdfDispatch(p, method));
    internal override double ILogCcdfDispatch(double logp, RandomVariableMethod? method = null) => IT(_sign ? Inner.ILogCcdfDispatch(logp, method) : Inner.ILogCdfDispatch(logp, method));
    internal override double IccdfDispatch(double p, RandomVariableMethod? method = null) => IT(_sign ? Inner.IccdfDispatch(p, method) : Inner.IcdfDispatch(p, method));

    internal override double? MomentStandardizedDispatch(int order, MomentMethods methods)
    {
        var r = Inner.MomentStandardizedDispatch(order, methods);
        return r is { } v ? v * Math.Pow(Math.Sign(Scale), order) : null;
    }

    internal override double? MomentCentralDispatch(int order, MomentMethods methods)
    {
        var r = Inner.MomentCentralDispatch(order, methods);
        return r is { } v ? v * Math.Pow(Scale, order) : null;
    }

    internal override double? MomentRawDispatch(int order, MomentMethods methods)
    {
        var raw = new double[order + 1];
        for (var i = 0; i <= order; i++)
        {
            var r = Inner.MomentRawDispatch(i, i < order ? MomentMethods.All : methods);
            if (r is null) return null;
            raw[i] = r.Value * Math.Pow(Scale, i);
        }
        return TransformCenter(order, raw, Loc, 0.0);
    }

    internal override void SampleDispatch(Span<double> destination, SampleSource source, RandomVariableMethod? method = null)
    {
        Inner.SampleDispatch(destination, source, method);
        for (var i = 0; i < destination.Length; i++) destination[i] = IT(destination[i]);
    }

    /// <inheritdoc />
    public override string Name => Inner.Name;

    /// <inheritdoc />
    public override string ToString()
    {
        var s = $"{PyFormat.F(Scale)}*{Inner}";
        if (Loc < 0) s += $" - {PyFormat.F(-Loc)}";
        else if (Loc != 0) s += $" + {PyFormat.F(Loc)}";
        return s;
    }
}

/// <summary>
/// X restricted to [lb, ub] and renormalised (scipy's <c>TruncatedDistribution</c>, made by
/// <c>stats.truncate</c>). The log density and log CDFs are those of X shifted by the log mass
/// of the interval; as in scipy, the linear CDF and the moments have no formula here and come
/// from quadrature of the density.
/// </summary>
public sealed class TruncatedDistribution : TransformedDistribution
{
    private readonly double _a, _b, _logMass;

    /// <summary>X truncated to [<paramref name="lb"/>, <paramref name="ub"/>].</summary>
    public TruncatedDistribution(ContinuousRandomVariable x, double lb = double.NegativeInfinity, double ub = double.PositiveInfinity) : base(x)
    {
        if (double.IsNaN(lb) || double.IsNaN(ub) || !(lb < ub) || double.IsPositiveInfinity(lb) || double.IsNegativeInfinity(ub))
            throw new ArgumentException("Truncation bounds must satisfy lb < ub (lb may be -inf, ub +inf).");
        Lb = lb;
        Ub = ub;
        var (a, b) = x.SupportCore();
        _a = Math.Max(a, lb);
        _b = Math.Min(b, ub);
        if (!(_a < _b)) throw new ArgumentException("The truncation interval does not overlap the support.");
        _logMass = x.LogCdf2Dispatch(_a, _b);
    }

    /// <summary>The lower truncation point.</summary>
    public double Lb { get; }

    /// <summary>The upper truncation point.</summary>
    public double Ub { get; }

    // scipy's _overrides is False for every function here, so its linear CDF, CCDF and two-argument
    // CDF fall back to quadrature of the density. The log functions overridden below are exact, so
    // they are reported as formulas instead and the linear ones become their exponentials.
    internal override bool Overrides(RandomVariableFormula formula) => formula is RandomVariableFormula.Pmf or RandomVariableFormula.LogPmf or RandomVariableFormula.LogPdf or RandomVariableFormula.LogCdf or RandomVariableFormula.LogCcdf or RandomVariableFormula.LogCdf2 or RandomVariableFormula.ILogCdf or RandomVariableFormula.ILogCcdf or RandomVariableFormula.Icdf or RandomVariableFormula.Iccdf;

    /// <inheritdoc />
    protected internal override (double Lower, double Upper) SupportCore() => (_a, _b);

    internal override double LogPdfDispatch(double x, RandomVariableMethod? method = null) => Inner.LogPdfDispatch(x, method) - _logMass;
    internal override double LogCdfDispatch(double x, RandomVariableMethod? method = null) => Inner.LogCdf2Dispatch(_a, x, method) - _logMass;
    internal override double LogCcdfDispatch(double x, RandomVariableMethod? method = null) => Inner.LogCdf2Dispatch(x, _b, method) - _logMass;
    internal override double LogCdf2Dispatch(double x, double y, RandomVariableMethod? method = null) => Inner.LogCdf2Dispatch(x, y, method) - _logMass;

    // scipy inverts F(a) + p·M with the inner icdf. When a sits in the upper tail, F(a) is 1 − 3e-7
    // (say) and the sum keeps only nine digits of the tail; the same point is S(b) + (1 − p)·M in
    // terms of the survival function, which keeps them all. Each inverse uses whichever side of the
    // inner distribution the kept interval is in.
    private bool UpperTail => Inner.LogCdfDispatch(_a) > -StatsDistMath.Ln2;
    private bool LowerTail => Inner.LogCcdfDispatch(_b) > -StatsDistMath.Ln2;

    internal override double ILogCdfDispatch(double logp, RandomVariableMethod? method = null)
    {
        if (UpperTail)
            return Inner.ILogCcdfDispatch(StatsDistMath.LogAddExp(Inner.LogCcdfDispatch(_b), StatsDistMath.Log1mExp(logp) + _logMass), method);
        return Inner.ILogCdfDispatch(StatsDistMath.LogAddExp(Inner.LogCdfDispatch(_a), logp + _logMass), method);
    }

    internal override double ILogCcdfDispatch(double logp, RandomVariableMethod? method = null)
    {
        if (LowerTail)
            return Inner.ILogCdfDispatch(StatsDistMath.LogAddExp(Inner.LogCdfDispatch(_a), StatsDistMath.Log1mExp(logp) + _logMass), method);
        return Inner.ILogCcdfDispatch(StatsDistMath.LogAddExp(Inner.LogCcdfDispatch(_b), logp + _logMass), method);
    }

    internal override double IcdfDispatch(double p, RandomVariableMethod? method = null)
    {
        if (UpperTail) return Inner.IccdfDispatch(Inner.CcdfDispatch(_b) + (1 - p) * Math.Exp(_logMass), method);
        return Inner.IcdfDispatch(Inner.CdfDispatch(_a) + p * Math.Exp(_logMass), method);
    }

    internal override double IccdfDispatch(double p, RandomVariableMethod? method = null)
    {
        if (LowerTail) return Inner.IcdfDispatch(Inner.CdfDispatch(_a) + (1 - p) * Math.Exp(_logMass), method);
        return Inner.IccdfDispatch(Inner.CcdfDispatch(_b) + p * Math.Exp(_logMass), method);
    }

    /// <inheritdoc />
    public override string ToString() => $"truncate({Inner}, lb={PyFormat.F(Lb)}, ub={PyFormat.F(Ub)})";
}

/// <summary>
/// The r-th order statistic of n independent draws of X (scipy's
/// <c>OrderStatisticDistribution</c>): density n!/((r−1)!(n−r)!) f F^(r−1) (1−F)^(n−r), CDF the
/// regularised incomplete beta function of F, quantiles X's quantile of the beta quantile.
/// </summary>
public sealed class OrderStatisticDistribution : TransformedDistribution
{
    private readonly double _logBeta;

    /// <summary>The <paramref name="r"/>-th smallest of <paramref name="n"/> draws of X (1 ≤ r ≤ n).</summary>
    public OrderStatisticDistribution(ContinuousRandomVariable x, int r, int n) : base(x)
    {
        if (n < 1) throw new ArgumentException("`n` must be a positive integer.", nameof(n));
        if (r < 1 || r > n) throw new ArgumentException("`r` must be an integer between 1 and n.", nameof(r));
        R = r;
        N = n;
        _logBeta = Special.BetaLn(r, n - r + 1);
    }

    /// <summary>The rank.</summary>
    public int R { get; }

    /// <summary>The sample size.</summary>
    public int N { get; }

    internal override bool Overrides(RandomVariableFormula formula) => formula is RandomVariableFormula.Pmf or RandomVariableFormula.LogPmf or RandomVariableFormula.LogPdf or RandomVariableFormula.Pdf
        or RandomVariableFormula.Cdf or RandomVariableFormula.Ccdf or RandomVariableFormula.Icdf or RandomVariableFormula.Iccdf;

    /// <inheritdoc />
    protected internal override (double Lower, double Upper) SupportCore() => Inner.SupportCore();

    /// <inheritdoc />
    protected override double LogPdfFormula(double x)
    {
        var lf = Inner.LogPdfDispatch(x);
        var lF = Inner.LogCdfDispatch(x);
        var lcF = Inner.LogCcdfDispatch(x);
        var a = R - 1 == 0 && double.IsNegativeInfinity(lF) ? 0.0 : (R - 1) * lF;
        var b = N - R == 0 && double.IsNegativeInfinity(lcF) ? 0.0 : (N - R) * lcF;
        return lf + a + b - _logBeta;
    }

    /// <inheritdoc />
    protected override double PdfFormula(double x)
    {
        var f = Inner.PdfDispatch(x);
        var fx = Inner.CdfDispatch(x);
        var cfx = Inner.CcdfDispatch(x);
        return f * Math.Pow(fx, R - 1) * Math.Pow(cfx, N - R) / Special.Beta(R, N - R + 1);
    }

    /// <inheritdoc />
    protected override double CdfFormula(double x) => Special.BetaInc(R, N - R + 1, Inner.CdfDispatch(x));

    /// <inheritdoc />
    protected override double CcdfFormula(double x) => Special.BetaIncC(R, N - R + 1, Inner.CdfDispatch(x));

    /// <inheritdoc />
    protected override double IcdfFormula(double p) => Inner.IcdfDispatch(Special.BetaIncInv(R, N - R + 1, p));

    /// <inheritdoc />
    protected override double IccdfFormula(double p) => Inner.IcdfDispatch(Special.BetaIncCInv(R, N - R + 1, p));

    /// <inheritdoc />
    public override string ToString() => $"order_statistic({Inner}, r={R}, n={N})";
}

/// <summary>
/// Y = g(X) for a strictly monotonic g with inverse h and |h'| = dh (scipy's
/// <c>MonotonicTransformedDistribution</c>): the density is f(h(y))·|h'(y)|, the CDF X's CDF (or
/// CCDF, for a decreasing g) at h(y), and the quantiles g of X's quantiles.
/// </summary>
public sealed class MonotonicTransformedDistribution : TransformedDistribution
{
    private readonly Func<double, double> _g, _h, _dh, _logdh;
    private readonly bool _increasing;
    private readonly string _repr;

    /// <summary>The distribution of g(X).</summary>
    /// <param name="x">The random variable X.</param>
    /// <param name="g">The strictly monotonic function.</param>
    /// <param name="h">Its inverse.</param>
    /// <param name="dh">|h'(u)|.</param>
    /// <param name="logdh">log |h'(u)| (default log(dh(u)); supply it to avoid under/overflow).</param>
    /// <param name="increasing">Whether g is increasing.</param>
    /// <param name="repr">What <see cref="ToString"/> prints (default <c>g(X)</c>).</param>
    public MonotonicTransformedDistribution(ContinuousRandomVariable x, Func<double, double> g, Func<double, double> h, Func<double, double> dh,
        Func<double, double>? logdh = null, bool increasing = true, string? repr = null) : base(x)
    {
        ArgumentNullException.ThrowIfNull(g); ArgumentNullException.ThrowIfNull(h); ArgumentNullException.ThrowIfNull(dh);
        _g = g; _h = h; _dh = dh;
        _logdh = logdh ?? (u => Math.Log(dh(u)));
        _increasing = increasing;
        _repr = repr ?? $"g({x})";
    }

    // Reported as formulas (scipy says False): what is overridden below is exact, and the two-argument
    // CDF built from it by subtraction is right where scipy's quadrature of the density is not —
    // for X**3 the density has an integrable singularity inside the interval.
    internal override bool Overrides(RandomVariableFormula formula) => formula is RandomVariableFormula.Pmf or RandomVariableFormula.LogPmf or RandomVariableFormula.LogPdf or RandomVariableFormula.Pdf or RandomVariableFormula.LogCdf or RandomVariableFormula.Cdf or RandomVariableFormula.LogCcdf or RandomVariableFormula.Ccdf or RandomVariableFormula.ILogCdf or RandomVariableFormula.Icdf or RandomVariableFormula.ILogCcdf or RandomVariableFormula.Iccdf;

    /// <inheritdoc />
    protected internal override (double Lower, double Upper) SupportCore()
    {
        var (a, b) = Inner.SupportCore();
        if (b == 0) b = -0.0;
        return _increasing ? (_g(a), _g(b)) : (_g(b), _g(a));
    }

    internal override double LogPdfDispatch(double x, RandomVariableMethod? method = null) => Inner.LogPdfDispatch(_h(x), method) + _logdh(x);
    internal override double PdfDispatch(double x, RandomVariableMethod? method = null) => Inner.PdfDispatch(_h(x), method) * _dh(x);
    internal override double LogCdfDispatch(double x, RandomVariableMethod? method = null) => _increasing ? Inner.LogCdfDispatch(_h(x), method) : Inner.LogCcdfDispatch(_h(x), method);
    internal override double CdfDispatch(double x, RandomVariableMethod? method = null) => _increasing ? Inner.CdfDispatch(_h(x), method) : Inner.CcdfDispatch(_h(x), method);
    internal override double LogCcdfDispatch(double x, RandomVariableMethod? method = null) => _increasing ? Inner.LogCcdfDispatch(_h(x), method) : Inner.LogCdfDispatch(_h(x), method);
    internal override double CcdfDispatch(double x, RandomVariableMethod? method = null) => _increasing ? Inner.CcdfDispatch(_h(x), method) : Inner.CdfDispatch(_h(x), method);
    internal override double ILogCdfDispatch(double logp, RandomVariableMethod? method = null) => _g(_increasing ? Inner.ILogCdfDispatch(logp, method) : Inner.ILogCcdfDispatch(logp, method));
    internal override double IcdfDispatch(double p, RandomVariableMethod? method = null) => _g(_increasing ? Inner.IcdfDispatch(p, method) : Inner.IccdfDispatch(p, method));
    internal override double ILogCcdfDispatch(double logp, RandomVariableMethod? method = null) => _g(_increasing ? Inner.ILogCcdfDispatch(logp, method) : Inner.ILogCdfDispatch(logp, method));
    internal override double IccdfDispatch(double p, RandomVariableMethod? method = null) => _g(_increasing ? Inner.IccdfDispatch(p, method) : Inner.IcdfDispatch(p, method));

    internal override void SampleDispatch(Span<double> destination, SampleSource source, RandomVariableMethod? method = null)
    {
        Inner.SampleDispatch(destination, source, method);
        for (var i = 0; i < destination.Length; i++) destination[i] = _g(destination[i]);
    }

    /// <inheritdoc />
    public override string ToString() => _repr;
}

/// <summary>
/// |X| (scipy's <c>FoldedDistribution</c>, made by <c>stats.abs</c>): the density at y is
/// f(y) + f(−y) within X's support, and the CDF is the mass X puts on [−y, y].
/// </summary>
public sealed class FoldedDistribution : TransformedDistribution
{
    /// <summary>The distribution of |X|.</summary>
    public FoldedDistribution(ContinuousRandomVariable x) : base(x) { }

    // As for the monotonic transform: the overridden functions are reported as formulas.
    internal override bool Overrides(RandomVariableFormula formula) => formula is RandomVariableFormula.Pmf or RandomVariableFormula.LogPmf or RandomVariableFormula.LogPdf or RandomVariableFormula.Pdf or RandomVariableFormula.LogCdf or RandomVariableFormula.Cdf or RandomVariableFormula.LogCcdf or RandomVariableFormula.Ccdf;

    /// <inheritdoc />
    protected internal override (bool Lower, bool Upper) SupportInclusive => (true, Inner.SupportInclusive.Upper);

    /// <inheritdoc />
    protected internal override (double Lower, double Upper) SupportCore()
    {
        var (a, b) = Inner.SupportCore();
        double a1 = Math.Abs(a), b1 = Math.Abs(b);
        var lo = Math.Min(a1, b1);
        var hi = Math.Max(a1, b1);
        if (a < 0 && b > 0) lo = 0;
        return (lo, hi);
    }

    internal override IEnumerable<double> QuadratureBreakpoints()
    {
        var (a, b) = Inner.SupportCore();
        return base.QuadratureBreakpoints().Concat([Math.Abs(a), Math.Abs(b)]);
    }

    internal override double LogPdfDispatch(double x, RandomVariableMethod? method = null)
    {
        x = Math.Abs(x);
        var (a, b) = Inner.SupportCore();
        var right = x > b ? double.NegativeInfinity : Inner.LogPdfDispatch(x, method);
        var left = -x < a ? double.NegativeInfinity : Inner.LogPdfDispatch(-x, method);
        return StatsDistMath.LogAddExp(left, right);
    }

    internal override double PdfDispatch(double x, RandomVariableMethod? method = null)
    {
        x = Math.Abs(x);
        var (a, b) = Inner.SupportCore();
        var right = x > b ? 0.0 : Inner.PdfDispatch(x, method);
        var left = -x < a ? 0.0 : Inner.PdfDispatch(-x, method);
        return left + right;
    }

    private (double L, double R) Bounds(double x)
    {
        x = Math.Abs(x);
        var (a, b) = Inner.SupportCore();
        return (Math.Max(-x, a), Math.Min(x, b));
    }

    internal override double LogCdfDispatch(double x, RandomVariableMethod? method = null) { var (l, r) = Bounds(x); return Inner.LogCdf2Dispatch(l, r, method); }
    internal override double CdfDispatch(double x, RandomVariableMethod? method = null) { var (l, r) = Bounds(x); return Inner.Cdf2Dispatch(l, r); }
    internal override double LogCcdfDispatch(double x, RandomVariableMethod? method = null) { var (l, r) = Bounds(x); return Inner.LogCcdf2Dispatch(l, r, method); }
    internal override double CcdfDispatch(double x, RandomVariableMethod? method = null) { var (l, r) = Bounds(x); return Inner.Ccdf2Dispatch(l, r, method); }

    internal override void SampleDispatch(Span<double> destination, SampleSource source, RandomVariableMethod? method = null)
    {
        Inner.SampleDispatch(destination, source, method);
        for (var i = 0; i < destination.Length; i++) destination[i] = Math.Abs(destination[i]);
    }

    /// <inheritdoc />
    public override string ToString() => $"abs({Inner})";
}
