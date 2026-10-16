using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// A frozen continuous distribution, scipy's <c>rv_continuous</c> with its parameters bound.
/// </summary>
/// <remarks>
/// <para>
/// Each family implements the <em>standard form</em> (loc = 0, scale = 1) through the protected
/// <c>…0</c> hooks; this class applies location and scale exactly as scipy does,
/// <c>pdf(x) = pdf₀((x − loc)/scale)/scale</c>, and handles the support, NaN and the edge
/// probabilities so no family has to.
/// </para>
/// <para>
/// Every hook has a generic fallback — CDF by quadrature of the pdf, quantiles by bracketing and
/// Brent on the CDF (or on the survival function above the median, which is what keeps upper-tail
/// quantiles accurate), moments and entropy by quadrature. Families override them with closed
/// forms wherever those exist; the fallbacks are there so a new family is correct before it is fast.
/// </para>
/// </remarks>
public abstract class ContinuousDistribution : IUnivariateDistribution
{
    private DistributionMoments? _moments;

    /// <summary>Validates and stores location and scale.</summary>
    protected ContinuousDistribution(double loc, double scale)
    {
        if (!double.IsFinite(loc)) throw new ArgumentException("loc must be finite.", nameof(loc));
        if (!(scale > 0) || !double.IsFinite(scale)) throw new ArgumentException("scale must be positive and finite.", nameof(scale));
        Loc = loc;
        Scale = scale;
    }

    /// <summary>The location parameter.</summary>
    public double Loc { get; }

    /// <summary>The scale parameter.</summary>
    public double Scale { get; }

    /// <inheritdoc />
    public abstract string Name { get; }

    /// <inheritdoc />
    public bool IsDiscrete => false;

    /// <summary>Names of the shape parameters in scipy's order (empty for a pure location–scale family).</summary>
    public virtual string[] ShapeNames => [];

    /// <summary>The shape parameter values, in the order of <see cref="ShapeNames"/>.</summary>
    public virtual double[] Shapes => [];

    /// <summary>
    /// A distribution of the same family with other parameters — what <see cref="FitTo"/> and generic
    /// code use to build candidates. Throws <see cref="ArgumentException"/> for invalid parameters.
    /// </summary>
    public abstract ContinuousDistribution WithParameters(double[] shapes, double loc, double scale);

    // ------------------------------------------------------------------ standard-form hooks

    /// <summary>Lower end of the standard support.</summary>
    protected virtual double Lower0 => double.NegativeInfinity;

    /// <summary>Upper end of the standard support.</summary>
    protected virtual double Upper0 => double.PositiveInfinity;

    /// <summary>Standard density at z (z inside the closed support).</summary>
    protected abstract double Pdf0(double z);

    /// <summary>Standard log density.</summary>
    protected virtual double LogPdf0(double z) => Math.Log(Pdf0(z));

    /// <summary>Standard CDF at z strictly inside the support. Fallback: quadrature of the pdf.</summary>
    protected virtual double Cdf0(double z) => Math.Clamp(StatsDistMath.Quad(Pdf0, Lower0, z), 0.0, 1.0);

    /// <summary>Standard survival function. Fallback: quadrature of the pdf over the upper tail.</summary>
    protected virtual double Sf0(double z) => Math.Clamp(StatsDistMath.Quad(Pdf0, z, Upper0), 0.0, 1.0);

    /// <summary>Standard log CDF. Fallback: log Cdf₀, or log1p(−Sf₀) once the CDF passes ½ (where
    /// log of a number near 1 would keep none of the digits of the small tail).</summary>
    protected virtual double LogCdf0(double z)
    {
        var c = Cdf0(z);
        return c > 0.5 ? Special.Log1p(-Sf0(z)) : Math.Log(c);
    }

    /// <summary>Standard log survival function, with the same switch as <see cref="LogCdf0"/>.</summary>
    protected virtual double LogSf0(double z)
    {
        var s = Sf0(z);
        return s > 0.5 ? Special.Log1p(-Cdf0(z)) : Math.Log(s);
    }

    /// <summary>
    /// Standard quantile for 0 &lt; q &lt; 1. Fallback: invert the CDF below the median and the survival
    /// function above it (1 − q is exact there), so neither tail loses digits to cancellation.
    /// </summary>
    protected virtual double Ppf0(double q)
        => q <= 0.5
            ? StatsDistMath.InvertMonotone(Cdf0, q, Lower0, Upper0, increasing: true, start: FiniteStart())
            : StatsDistMath.InvertMonotone(Sf0, 1.0 - q, Lower0, Upper0, increasing: false, start: FiniteStart());

    /// <summary>Standard inverse survival function for 0 &lt; q &lt; 1.</summary>
    protected virtual double Isf0(double q)
        => q <= 0.5
            ? StatsDistMath.InvertMonotone(Sf0, q, Lower0, Upper0, increasing: false, start: FiniteStart())
            : StatsDistMath.InvertMonotone(Cdf0, 1.0 - q, Lower0, Upper0, increasing: true, start: FiniteStart());

    private double FiniteStart()
    {
        if (double.IsFinite(Lower0) && double.IsFinite(Upper0)) return 0.5 * (Lower0 + Upper0);
        if (double.IsFinite(Lower0)) return Lower0 + 1.0;
        if (double.IsFinite(Upper0)) return Upper0 - 1.0;
        return 0.0;
    }

    /// <summary>
    /// Mean, variance, skewness and excess kurtosis of the standard form. Fallback: from the raw
    /// moments <see cref="Munp0"/>.
    /// </summary>
    protected virtual DistributionMoments Moments0()
    {
        var m1 = Munp0(1);
        var m2 = Munp0(2);
        var m3 = Munp0(3);
        var m4 = Munp0(4);
        var v = m2 - m1 * m1;
        var mu3 = m3 - 3 * m1 * m2 + 2 * m1 * m1 * m1;
        var mu4 = m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1 * m1 * m1 * m1;
        return new DistributionMoments(m1, v, mu3 / Math.Pow(v, 1.5), mu4 / (v * v) - 3.0);
    }

    /// <summary>Standard raw moment E[Zⁿ]. Fallback: quadrature (split at the median).</summary>
    protected virtual double Munp0(int n)
    {
        var med = Ppf0(0.5);
        double F(double z)
        {
            var p = Pdf0(z);
            return p == 0 ? 0.0 : Math.Pow(z, n) * p;
        }
        return StatsDistMath.Quad(F, Lower0, med) + StatsDistMath.Quad(F, med, Upper0);
    }

    /// <summary>Differential entropy of the standard form. Fallback: quadrature of −p log p.</summary>
    protected virtual double Entropy0()
    {
        var med = Ppf0(0.5);
        double F(double z)
        {
            var lp = LogPdf0(z);
            return double.IsNegativeInfinity(lp) || double.IsNaN(lp) ? 0.0 : -Math.Exp(lp) * lp;
        }
        return StatsDistMath.Quad(F, Lower0, med) + StatsDistMath.Quad(F, med, Upper0);
    }

    /// <summary>One standard variate. Fallback: inversion of an open-interval uniform.</summary>
    protected virtual double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));

    /// <summary>Fills <paramref name="dest"/> with standard variates.</summary>
    protected virtual void Rvs0(GraviRandom rng, Span<double> dest)
    {
        for (var i = 0; i < dest.Length; i++) dest[i] = Rvs0(rng);
    }

    // ------------------------------------------------------------------ public scalar API

    /// <inheritdoc />
    public (double Lower, double Upper) Support() => (Loc + Scale * Lower0, Loc + Scale * Upper0);

    /// <summary>Probability density function.</summary>
    public double Pdf(double x)
    {
        var z = (x - Loc) / Scale;
        if (double.IsNaN(z)) return double.NaN;
        if (z < Lower0 || z > Upper0) return 0.0;
        return Pdf0(z) / Scale;
    }

    /// <summary>Log of the probability density function.</summary>
    public double LogPdf(double x)
    {
        var z = (x - Loc) / Scale;
        if (double.IsNaN(z)) return double.NaN;
        if (z < Lower0 || z > Upper0) return double.NegativeInfinity;
        return LogPdf0(z) - Math.Log(Scale);
    }

    /// <inheritdoc />
    public double Density(double x) => Pdf(x);

    /// <inheritdoc />
    public double LogDensity(double x) => LogPdf(x);

    /// <inheritdoc />
    public double Cdf(double x)
    {
        var z = (x - Loc) / Scale;
        if (double.IsNaN(z)) return double.NaN;
        if (z <= Lower0) return 0.0;
        if (z >= Upper0) return 1.0;
        return Cdf0(z);
    }

    /// <inheritdoc />
    public double LogCdf(double x)
    {
        var z = (x - Loc) / Scale;
        if (double.IsNaN(z)) return double.NaN;
        if (z <= Lower0) return double.NegativeInfinity;
        if (z >= Upper0) return 0.0;
        return LogCdf0(z);
    }

    /// <inheritdoc />
    public double Sf(double x)
    {
        var z = (x - Loc) / Scale;
        if (double.IsNaN(z)) return double.NaN;
        if (z <= Lower0) return 1.0;
        if (z >= Upper0) return 0.0;
        return Sf0(z);
    }

    /// <inheritdoc />
    public double LogSf(double x)
    {
        var z = (x - Loc) / Scale;
        if (double.IsNaN(z)) return double.NaN;
        if (z <= Lower0) return 0.0;
        if (z >= Upper0) return double.NegativeInfinity;
        return LogSf0(z);
    }

    /// <inheritdoc />
    public double Ppf(double q)
    {
        if (!(q >= 0 && q <= 1)) return double.NaN;
        if (q == 0) return Loc + Scale * Lower0;
        if (q == 1) return Loc + Scale * Upper0;
        return Loc + Scale * Ppf0(q);
    }

    /// <inheritdoc />
    public double Isf(double q)
    {
        if (!(q >= 0 && q <= 1)) return double.NaN;
        if (q == 0) return Loc + Scale * Upper0;
        if (q == 1) return Loc + Scale * Lower0;
        return Loc + Scale * Isf0(q);
    }

    /// <inheritdoc />
    public DistributionMoments Stats()
    {
        var m = _moments ??= Moments0();
        return new DistributionMoments(Loc + Scale * m.Mean, Scale * Scale * m.Variance, m.Skewness, m.Kurtosis);
    }

    /// <inheritdoc />
    public double Mean() => Stats().Mean;

    /// <inheritdoc />
    public double Var() => Stats().Variance;

    /// <inheritdoc />
    public double Std() => Math.Sqrt(Var());

    /// <inheritdoc />
    public double Skew() => Stats().Skewness;

    /// <inheritdoc />
    public double Kurtosis() => Stats().Kurtosis;

    /// <inheritdoc />
    public double Median() => Ppf(0.5);

    /// <inheritdoc />
    public double Entropy() => Entropy0() + Math.Log(Scale);

    /// <summary>
    /// The n-th non-central moment E[Xⁿ], expanded binomially from the standard moments. Orders up to
    /// four come from the closed-form <see cref="Stats"/> when the family has one.
    /// </summary>
    public double Moment(int n)
    {
        if (n < 0) throw new ArgumentException("Moment order must be non-negative.", nameof(n));
        if (n == 0) return 1.0;
        var raw = new double[n + 1];
        raw[0] = 1.0;
        if (n <= 4)
        {
            var s = _moments ??= Moments0();
            var m = s.Mean;
            var v = s.Variance;
            raw[1] = m;
            if (n >= 2) raw[2] = v + m * m;
            if (n >= 3)
            {
                var mu3 = s.Skewness * Math.Pow(v, 1.5);
                raw[3] = mu3 + 3 * m * v + m * m * m;
                if (n >= 4)
                {
                    var mu4 = (s.Kurtosis + 3.0) * v * v;
                    raw[4] = mu4 + 4 * m * mu3 + 6 * m * m * v + m * m * m * m;
                }
            }
        }
        else
        {
            for (var k = 1; k <= n; k++) raw[k] = Munp0(k);
        }
        if (Loc == 0) return Math.Pow(Scale, n) * raw[n];
        var total = 0.0;
        for (var k = 0; k <= n; k++)
            total += Special.Binom(n, k) * Math.Pow(Loc, n - k) * Math.Pow(Scale, k) * raw[k];
        return total;
    }

    /// <inheritdoc />
    public (double Lower, double Upper) Interval(double confidence)
    {
        if (!(confidence >= 0 && confidence <= 1)) throw new ArgumentException("confidence must lie in [0, 1].", nameof(confidence));
        var a = 0.5 * (1.0 - confidence);
        return (Ppf(a), Isf(a));
    }

    /// <summary>
    /// E[f(X)] over [lb, ub] by adaptive quadrature, scipy's <c>expect</c>. With
    /// <paramref name="conditional"/> the result is divided by P(lb ≤ X ≤ ub). The range is split
    /// at the median so a narrow peak on an infinite range is not stepped over.
    /// </summary>
    public double Expect(Func<double, double>? func = null, double? lb = null, double? ub = null, bool conditional = false)
    {
        var (lo, hi) = Support();
        var a = Math.Max(lb ?? lo, lo);
        var b = Math.Min(ub ?? hi, hi);
        if (!(a < b)) return 0.0;
        func ??= x => x;
        double F(double x)
        {
            var p = Pdf(x);
            return p == 0 ? 0.0 : func(x) * p;
        }
        var med = Median();
        double value;
        if (med > a && med < b) value = StatsDistMath.Quad(F, a, med) + StatsDistMath.Quad(F, med, b);
        else value = StatsDistMath.Quad(F, a, b);
        if (conditional)
        {
            var mass = a > med ? Sf(a) - Sf(b) : Cdf(b) - Cdf(a);
            value /= mass;
        }
        return value;
    }

    /// <summary>One random variate.</summary>
    public double Rvs(GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        return Loc + Scale * Rvs0(rng);
    }

    /// <summary><paramref name="size"/> random variates drawn from <paramref name="rng"/>.</summary>
    public double[] Rvs(int size, GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        if (size < 0) throw new ArgumentException("size must be non-negative.", nameof(size));
        var result = new double[size];
        Rvs0(rng, result);
        if (Loc != 0 || Scale != 1)
            for (var i = 0; i < result.Length; i++) result[i] = Loc + Scale * result[i];
        return result;
    }

    /// <summary><paramref name="size"/> random variates; a null seed draws a fresh one.</summary>
    public double[] Rvs(int size, int? seed = null) => Rvs(size, StatsDistributions.MakeRng(seed));

    /// <summary>Random variates in an array of the given shape.</summary>
    public NdArray Rvs(int[] shape, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(shape);
        var n = 1;
        foreach (var d in shape) n *= d;
        return new NdArray(Rvs(n, StatsDistributions.MakeRng(seed)), (int[])shape.Clone());
    }

    // ------------------------------------------------------------------ vectorised API

    /// <summary>Element-wise <see cref="Pdf(double)"/>; parallel above <see cref="Special.ParallelThreshold"/>.</summary>
    public double[] Pdf(double[] x) => StatsDistributions.Map(x, Pdf);
    /// <summary>Element-wise <see cref="LogPdf(double)"/>.</summary>
    public double[] LogPdf(double[] x) => StatsDistributions.Map(x, LogPdf);
    /// <summary>Element-wise <see cref="Cdf(double)"/>.</summary>
    public double[] Cdf(double[] x) => StatsDistributions.Map(x, Cdf);
    /// <summary>Element-wise <see cref="LogCdf(double)"/>.</summary>
    public double[] LogCdf(double[] x) => StatsDistributions.Map(x, LogCdf);
    /// <summary>Element-wise <see cref="Sf(double)"/>.</summary>
    public double[] Sf(double[] x) => StatsDistributions.Map(x, Sf);
    /// <summary>Element-wise <see cref="LogSf(double)"/>.</summary>
    public double[] LogSf(double[] x) => StatsDistributions.Map(x, LogSf);
    /// <summary>Element-wise <see cref="Ppf(double)"/>.</summary>
    public double[] Ppf(double[] q) => StatsDistributions.Map(q, Ppf);
    /// <summary>Element-wise <see cref="Isf(double)"/>.</summary>
    public double[] Isf(double[] q) => StatsDistributions.Map(q, Isf);

    /// <summary>Element-wise <see cref="Pdf(double)"/> over an array of any shape.</summary>
    public NdArray Pdf(NdArray x) => Special.Apply(x, Pdf);
    /// <summary>Element-wise <see cref="LogPdf(double)"/>.</summary>
    public NdArray LogPdf(NdArray x) => Special.Apply(x, LogPdf);
    /// <summary>Element-wise <see cref="Cdf(double)"/>.</summary>
    public NdArray Cdf(NdArray x) => Special.Apply(x, Cdf);
    /// <summary>Element-wise <see cref="LogCdf(double)"/>.</summary>
    public NdArray LogCdf(NdArray x) => Special.Apply(x, LogCdf);
    /// <summary>Element-wise <see cref="Sf(double)"/>.</summary>
    public NdArray Sf(NdArray x) => Special.Apply(x, Sf);
    /// <summary>Element-wise <see cref="LogSf(double)"/>.</summary>
    public NdArray LogSf(NdArray x) => Special.Apply(x, LogSf);
    /// <summary>Element-wise <see cref="Ppf(double)"/>.</summary>
    public NdArray Ppf(NdArray q) => Special.Apply(q, Ppf);
    /// <summary>Element-wise <see cref="Isf(double)"/>.</summary>
    public NdArray Isf(NdArray q) => Special.Apply(q, Isf);

    // ------------------------------------------------------------------ fitting

    /// <summary>
    /// The negative log-likelihood of <paramref name="data"/>, scipy's <c>nnlf</c>; +∞ when a point
    /// lies outside the support.
    /// </summary>
    public double NegativeLogLikelihood(ReadOnlySpan<double> data)
    {
        var total = 0.0;
        foreach (var x in data)
        {
            var lp = LogPdf(x);
            if (double.IsNegativeInfinity(lp) || double.IsNaN(lp)) return double.PositiveInfinity;
            total -= lp;
        }
        return total;
    }

    /// <summary>
    /// Maximum-likelihood fit of this family to <paramref name="data"/>, scipy's <c>dist.fit</c>.
    /// Parameters given as non-null in <paramref name="fixedShapes"/> (scipy's <c>f0, f1, …</c>),
    /// <paramref name="floc"/> or <paramref name="fscale"/> are held fixed. Families with closed-form
    /// or one-dimensional MLEs override this; the rest use <see cref="FitNumerically"/>, starting
    /// from this instance's shapes. Returns a new frozen distribution.
    /// </summary>
    public virtual ContinuousDistribution FitTo(double[] data, double?[]? fixedShapes = null, double? floc = null, double? fscale = null)
        => FitNumerically(data, fixedShapes, floc, fscale);

    /// <summary>
    /// Generic MLE: Nelder–Mead on the penalised negative log-likelihood (scipy's
    /// <c>_penalized_nnlf</c> with <c>optimize.fmin</c>), restarted from its own answer until it
    /// stops improving because a collapsed simplex is the method's usual failure. The start is this
    /// instance's shapes with location and scale from scipy's <c>_fit_loc_scale_support</c>.
    /// </summary>
    public ContinuousDistribution FitNumerically(double[] data, double?[]? fixedShapes = null, double? floc = null, double? fscale = null)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (data.Length == 0) throw new ArgumentException("data must not be empty.", nameof(data));
        foreach (var x in data)
            if (!double.IsFinite(x)) throw new ArgumentException("data must be finite.", nameof(data));
        var k = Shapes.Length;
        if (fixedShapes != null && fixedShapes.Length > k) throw new ArgumentException($"{Name} has {k} shape parameter(s).", nameof(fixedShapes));
        var full = new double[k + 2];
        var start = FitStart(data);
        Array.Copy(start, full, k + 2);
        var isFixed = new bool[k + 2];
        for (var i = 0; i < k; i++)
            if (fixedShapes != null && i < fixedShapes.Length && fixedShapes[i] is double f) { full[i] = f; isFixed[i] = true; }
        if (floc is double fl) { full[k] = fl; isFixed[k] = true; }
        if (fscale is double fs) { full[k + 1] = fs; isFixed[k + 1] = true; }
        if (floc.HasValue || fixedShapes != null)
        {
            // Re-derive the free location/scale guess under the fixed values.
            var redo = WithParametersOrNull(full[..k], 0.0, 1.0);
            if (redo != null)
            {
                var (l, s) = redo.FitLocScaleSupport(data);
                if (!isFixed[k]) full[k] = l;
                if (!isFixed[k + 1]) full[k + 1] = fscale ?? (floc.HasValue ? FitScaleGivenLoc(redo, data, full[k]) : s);
            }
        }
        var free = new List<int>();
        for (var i = 0; i < k + 2; i++) if (!isFixed[i]) free.Add(i);
        if (free.Count == 0) return WithParameters(full[..k], full[k], full[k + 1]);

        double Objective(double[] p)
        {
            var t = (double[])full.Clone();
            for (var i = 0; i < free.Count; i++) t[free[i]] = p[i];
            return PenalizedNnlf(t, k, data);
        }

        var x0 = new double[free.Count];
        for (var i = 0; i < free.Count; i++) x0[i] = full[free[i]];
        var best = x0;
        var bestF = Objective(x0);
        for (var round = 0; round < 6; round++)
        {
            var res = Optimize.Minimize(Objective, best, MinimizeMethod.NelderMead,
                options: new NelderMeadOptions { Xatol = 1e-10, Fatol = 1e-13, MaxIter = 20000, MaxFev = 40000 });
            if (!(res.Fun < bestF - 1e-12 * Math.Max(1.0, Math.Abs(bestF))))
            {
                if (res.Fun <= bestF) { best = res.X; bestF = res.Fun; }
                break;
            }
            best = res.X;
            bestF = res.Fun;
        }
        for (var i = 0; i < free.Count; i++) full[free[i]] = best[i];
        return WithParameters(full[..k], full[k], full[k + 1]);
    }

    private static double FitScaleGivenLoc(ContinuousDistribution standard, double[] data, double loc)
    {
        // Moment match the spread about the fixed location; any positive value is a valid start.
        var s = 0.0;
        foreach (var x in data) s += Math.Abs(x - loc);
        s /= data.Length;
        var m = Math.Abs(standard.Stats().Mean);
        var div = double.IsFinite(m) && m > 0 ? m : 1.0;
        var v = s / div;
        return v > 0 && double.IsFinite(v) ? v : 1.0;
    }

    private ContinuousDistribution? WithParametersOrNull(double[] shapes, double loc, double scale)
    {
        try { return WithParameters(shapes, loc, scale); }
        catch (ArgumentException) { return null; }
    }

    private double PenalizedNnlf(double[] p, int k, double[] data)
    {
        if (!(p[k + 1] > 0) || !double.IsFinite(p[k])) return double.PositiveInfinity;
        var d = WithParametersOrNull(p[..k], p[k], p[k + 1]);
        if (d == null) return double.PositiveInfinity;
        const double penalty = 70978.0; // 100·log(DBL_MAX), as scipy's _LOGXMIN-based penalty
        var total = 0.0;
        var bad = 0;
        foreach (var x in data)
        {
            var lp = d.LogPdf(x);
            // An infinite density (a point on a pole of the pdf) is as unusable as a point outside
            // the support: either way the likelihood is degenerate there.
            if (double.IsFinite(lp)) total -= lp;
            else bad++;
        }
        return total + bad * penalty;
    }

    /// <summary>
    /// Starting values [shapes…, loc, scale] for the numerical fit. The default uses this instance's
    /// shapes and scipy's support-aware moment match for location and scale.
    /// </summary>
    protected virtual double[] FitStart(double[] data)
    {
        var shapes = Shapes;
        var standard = WithParameters(shapes, 0.0, 1.0);
        var (loc, scale) = standard.FitLocScaleSupport(data);
        var r = new double[shapes.Length + 2];
        shapes.CopyTo(r, 0);
        r[^2] = loc;
        r[^1] = scale;
        return r;
    }

    /// <summary>scipy's <c>_fit_loc_scale_support</c>: moment-matched loc/scale nudged so the data lie inside the support.</summary>
    internal (double Loc, double Scale) FitLocScaleSupport(double[] data)
    {
        var n = data.Length;
        double mean = 0, min = double.PositiveInfinity, max = double.NegativeInfinity;
        foreach (var x in data) { mean += x; min = Math.Min(min, x); max = Math.Max(max, x); }
        mean /= n;
        var v = 0.0;
        foreach (var x in data) v += (x - mean) * (x - mean);
        v /= n;
        double locHat, scaleHat;
        try
        {
            var s = Moments0();
            scaleHat = Math.Sqrt(v / s.Variance);
            locHat = mean - scaleHat * s.Mean;
        }
        catch (ArithmeticException) { locHat = double.NaN; scaleHat = double.NaN; }
        if (!double.IsFinite(locHat)) locHat = 0;
        if (!double.IsFinite(scaleHat) || scaleHat <= 0) scaleHat = 1;
        double a = Lower0, b = Upper0;
        if (!double.IsFinite(a) && !double.IsFinite(b)) return (locHat, scaleHat);
        var aHat = locHat + a * scaleHat;
        var bHat = locHat + b * scaleHat;
        if (aHat < min && max < bHat) return (locHat, scaleHat);
        var width = max - min;
        var margin = 0.1 * width;
        if (double.IsFinite(a) && double.IsFinite(b))
        {
            var sc = (width + 2 * margin) / (b - a);
            if (!(sc > 0)) sc = 1;
            return (min - margin - a * sc, sc);
        }
        if (double.IsFinite(a)) return (min - a - margin - (margin == 0 ? 1e-3 * Math.Max(1, Math.Abs(min)) : 0), 1.0);
        return (max - b + margin + (margin == 0 ? 1e-3 * Math.Max(1, Math.Abs(max)) : 0), 1.0);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var parts = new List<string>();
        var names = ShapeNames;
        var vals = Shapes;
        for (var i = 0; i < vals.Length; i++) parts.Add($"{names[i]}={vals[i]:R}");
        parts.Add($"loc={Loc:R}");
        parts.Add($"scale={Scale:R}");
        return $"{Name}({string.Join(", ", parts)})";
    }
}
