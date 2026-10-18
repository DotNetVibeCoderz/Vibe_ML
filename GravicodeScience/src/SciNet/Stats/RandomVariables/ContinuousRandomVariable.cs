using System.Numerics;

namespace Gravicode.Science.SciNet;

/// <summary>
/// A continuous random variable in scipy's new framework (<c>scipy.stats.ContinuousDistribution</c>).
/// </summary>
/// <remarks>
/// <para>Subclass it and override the <c>…Formula</c> hooks you have closed forms for — at least
/// <c>PdfFormula</c> or <c>LogPdfFormula</c> and <see cref="UnivariateRandomVariable.SupportCore"/>; every
/// other quantity then has a numerical fallback (see <see cref="UnivariateRandomVariable"/>). Or build one
/// without subclassing through <see cref="Stats.MakeDistribution(ContinuousDistributionDefinition)"/>.</para>
/// <para>Arithmetic follows scipy: <c>X + c</c>, <c>X - c</c>, <c>c * X</c>, <c>X / c</c> and <c>-X</c>
/// shift and scale (<see cref="ShiftedScaledDistribution"/>); <c>c / X</c>, <see cref="Pow(int)"/>,
/// <see cref="Pow(double, ContinuousRandomVariable)"/>, <see cref="Exp()"/>, <see cref="Log()"/> and
/// <see cref="Abs()"/> are monotonic or folding transforms.</para>
/// </remarks>
public abstract class ContinuousRandomVariable : UnivariateRandomVariable
{
    /// <inheritdoc />
    public sealed override bool IsDiscrete => false;

    /// <summary>A continuous distribution has zero mass everywhere (scipy defines <c>pmf</c> that way).</summary>
    protected override double PmfFormula(double x) => 0.0;

    /// <summary>Log of the zero mass: −∞.</summary>
    protected override double LogPmfFormula(double x) => double.NegativeInfinity;

    internal override double PxfDispatch(double x, RandomVariableMethod? method = null) => PdfDispatch(x, method);
    internal override double LogPxfDispatch(double x, RandomVariableMethod? method = null) => LogPdfDispatch(x, method);

    // ------------------------------------------------------------------ arithmetic (overridden by ShiftedScaled to fold)

    internal virtual ContinuousRandomVariable AddLoc(double loc) => new ShiftedScaledDistribution(this, loc, 1.0);
    internal virtual ContinuousRandomVariable SubtractLoc(double loc) => new ShiftedScaledDistribution(this, -loc, 1.0);
    internal virtual ContinuousRandomVariable MultiplyScale(double scale) => new ShiftedScaledDistribution(this, 0.0, scale);
    internal virtual ContinuousRandomVariable DivideScale(double scale) => new ShiftedScaledDistribution(this, 0.0, 1 / scale);

    /// <summary>The distribution of X + c.</summary>
    public static ContinuousRandomVariable operator +(ContinuousRandomVariable x, double c) { ArgumentNullException.ThrowIfNull(x); return x.AddLoc(c); }
    /// <summary>The distribution of c + X.</summary>
    public static ContinuousRandomVariable operator +(double c, ContinuousRandomVariable x) { ArgumentNullException.ThrowIfNull(x); return x.AddLoc(c); }
    /// <summary>The distribution of X − c.</summary>
    public static ContinuousRandomVariable operator -(ContinuousRandomVariable x, double c) { ArgumentNullException.ThrowIfNull(x); return x.SubtractLoc(c); }
    /// <summary>The distribution of c − X, i.e. (−X) + c.</summary>
    public static ContinuousRandomVariable operator -(double c, ContinuousRandomVariable x) { ArgumentNullException.ThrowIfNull(x); return (-x).AddLoc(c); }
    /// <summary>The distribution of c·X.</summary>
    public static ContinuousRandomVariable operator *(ContinuousRandomVariable x, double c) { ArgumentNullException.ThrowIfNull(x); return x.MultiplyScale(c); }
    /// <summary>The distribution of c·X.</summary>
    public static ContinuousRandomVariable operator *(double c, ContinuousRandomVariable x) { ArgumentNullException.ThrowIfNull(x); return x.MultiplyScale(c); }
    /// <summary>The distribution of X / c.</summary>
    public static ContinuousRandomVariable operator /(ContinuousRandomVariable x, double c) { ArgumentNullException.ThrowIfNull(x); return x.DivideScale(c); }
    /// <summary>The distribution of −X.</summary>
    public static ContinuousRandomVariable operator -(ContinuousRandomVariable x) { ArgumentNullException.ThrowIfNull(x); return x.MultiplyScale(-1.0); }

    /// <summary>
    /// The distribution of c / X (scipy's <c>__rtruediv__</c>): 1/X as a decreasing monotonic transform,
    /// then scaled by c. Only defined when the support of X is non-negative or non-positive.
    /// </summary>
    public static ContinuousRandomVariable operator /(double c, ContinuousRandomVariable x)
    {
        ArgumentNullException.ThrowIfNull(x);
        var (a, b) = x.Support();
        if (!(a >= 0 || b <= 0))
            throw new NotSupportedException("Division by a random variable is only implemented when the support is either non-negative or non-positive.");
        var name = $"1/({x})";
        ContinuousRandomVariable r = new MonotonicTransformedDistribution(x, u => 1 / u, u => 1 / u, u => 1 / (u * u), null, increasing: false, name);
        return c == 1 ? r : r * c;
    }

    /// <summary>
    /// The distribution of Xⁿ for a positive integer n (scipy's <c>X**n</c>): an increasing
    /// transform of X for odd n and of |X| for even n.
    /// </summary>
    public ContinuousRandomVariable Pow(int n)
    {
        if (n <= 0) throw new NotSupportedException("Raising a random variable to the power of an argument is only implemented when the argument is a positive integer.");
        var name = $"({this})**{n}";
        var x = n % 2 == 0 ? Abs() : this;
        var inv = 1.0 / n;
        return new MonotonicTransformedDistribution(x,
            u => Math.Pow(u, n),
            u => Math.Sign(u) * Math.Pow(Math.Abs(u), inv),
            u => inv * Math.Pow(Math.Abs(u), inv - 1),
            null, increasing: true, name);
    }

    /// <summary>
    /// The distribution of b^X for a positive base b ≠ 1 (scipy's <c>b**X</c>).
    /// </summary>
    public static ContinuousRandomVariable Pow(double @base, ContinuousRandomVariable x)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (!(@base > 0) || @base == 1 || double.IsInfinity(@base))
            throw new NotSupportedException("Raising an argument to the power of a random variable is only implemented when the argument is a positive scalar other than 1.");
        var lb = Math.Log(@base);
        return new MonotonicTransformedDistribution(x,
            u => Math.Pow(@base, u),
            u => Math.Log(u) / lb,
            u => 1 / Math.Abs(u * lb),
            null, increasing: @base > 1, $"{@base}**({x})");
    }

    /// <summary>The distribution of |X| (scipy's <c>abs(X)</c>), folding the negative half onto the positive.</summary>
    public ContinuousRandomVariable Abs() => new FoldedDistribution(this);

    /// <summary>The distribution of exp(X) (scipy's <c>stats.exp</c>).</summary>
    public ContinuousRandomVariable Exp()
        => new MonotonicTransformedDistribution(this, Math.Exp, Math.Log, u => 1 / u, u => -Math.Log(u), increasing: true, $"exp({this})");

    /// <summary>The distribution of log(X) for X with non-negative support (scipy's <c>stats.log</c>).</summary>
    public ContinuousRandomVariable Log()
    {
        if (Support().Lower < 0)
            throw new NotSupportedException("The logarithm of a random variable is only implemented when the support is non-negative.");
        return new MonotonicTransformedDistribution(this, Math.Log, Math.Exp, Math.Exp, u => u, increasing: true, $"log({this})");
    }

    /// <summary>X restricted to [lb, ub] and renormalised (scipy's <c>stats.truncate</c>).</summary>
    public ContinuousRandomVariable Truncate(double lb = double.NegativeInfinity, double ub = double.PositiveInfinity)
        => new TruncatedDistribution(this, lb, ub);

    /// <summary>The distribution of the r-th smallest of n independent draws of X (scipy's <c>stats.order_statistic</c>).</summary>
    public ContinuousRandomVariable OrderStatistic(int r, int n) => new OrderStatisticDistribution(this, r, n);
}

/// <summary>
/// A discrete random variable on the integers in scipy's new framework
/// (<c>scipy.stats.DiscreteDistribution</c>).
/// </summary>
/// <remarks>Override <c>PmfFormula</c> or <c>LogPmfFormula</c> and the support; the CDF falls back to
/// summation over the support, quantiles to an integer search for the smallest k with
/// CDF(k) ≥ p (scipy's predicate). Two-argument CDFs and transformations are not available, as
/// in scipy.</remarks>
public abstract class DiscreteRandomVariable : UnivariateRandomVariable
{
    /// <inheritdoc />
    public sealed override bool IsDiscrete => true;

    /// <summary>The supports' ends are integers and belong to it.</summary>
    protected internal override (bool Lower, bool Upper) SupportInclusive => (true, true);

    /// <summary>A discrete distribution's density is a sum of Dirac deltas: +∞ at the support points (scipy's convention).</summary>
    protected override double PdfFormula(double x) => double.PositiveInfinity;

    /// <summary>+∞, the log of the Dirac density.</summary>
    protected override double LogPdfFormula(double x) => double.PositiveInfinity;

    internal override double PxfDispatch(double x, RandomVariableMethod? method = null) => PmfDispatch(x, method);
    internal override double LogPxfDispatch(double x, RandomVariableMethod? method = null) => LogPmfDispatch(x, method);

    internal override double CdfQuadrature(double x) => base.CdfQuadrature(Math.Floor(x));
    internal override double LogCdfQuadrature(double x) => base.LogCdfQuadrature(Math.Floor(x));
    internal override double CcdfQuadrature(double x) => base.CcdfQuadrature(Math.Floor(x + 1));
    internal override double LogCcdfQuadrature(double x) => base.LogCcdfQuadrature(Math.Floor(x + 1));

    internal override double InvertCdf(double p)
    {
        var (a, b) = SupportCore();
        return RandomVariableNumerics.SmallestInteger(k => CdfDispatch(k) >= p, a, b);
    }

    internal override double InvertLogCdf(double logp)
    {
        var (a, b) = SupportCore();
        return RandomVariableNumerics.SmallestInteger(k => LogCdfDispatch(k) >= logp, a, b);
    }

    internal override double InvertCcdf(double p)
    {
        var (a, b) = SupportCore();
        return RandomVariableNumerics.SmallestInteger(k => CcdfDispatch(k) <= p, a, b);
    }

    internal override double InvertLogCcdf(double logp)
    {
        var (a, b) = SupportCore();
        return RandomVariableNumerics.SmallestInteger(k => LogCcdfDispatch(k) <= logp, a, b);
    }

    /// <summary>
    /// The mode by climbing the mass function from the median to its peak. scipy optimises the
    /// pmf as a function of a real argument and rounds; walking the integers gives the same
    /// answer for unimodal mass functions without needing the pmf between the integers.
    /// </summary>
    internal override double ModeOptimization()
    {
        var (a, b) = SupportCore();
        var k = Math.Clamp(Math.Floor(MedianDispatch()), a, b);
        var pk = PmfDispatch(k);
        while (k + 1 <= b) { var p = PmfDispatch(k + 1); if (p > pk) { k++; pk = p; } else break; }
        while (k - 1 >= a) { var p = PmfDispatch(k - 1); if (p > pk) { k--; pk = p; } else break; }
        return k;
    }
}
