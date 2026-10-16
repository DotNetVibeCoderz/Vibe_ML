using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// A frozen discrete distribution on the integers, scipy's <c>rv_discrete</c> with its parameters
/// bound. <c>pmf(k) = pmf₀(k − loc)</c>.
/// </summary>
/// <remarks>
/// <para>
/// Quantiles follow scipy exactly: <c>Ppf(q)</c> is the smallest k with <c>Cdf(k) ≥ q</c> and
/// <c>Isf(q)</c> the smallest k with <c>Sf(k) ≤ q</c>; <c>Ppf(0)</c> is one below the support. The
/// search starts from a family's continuous approximation, gallops to bracket and then bisects, so
/// it is exact by construction rather than by trusting a rounded continuous inverse. The predicates
/// are used literally — <c>Ppf(Cdf(k)) = k</c> and <c>Isf(Sf(k)) = k</c> hold bit for bit — so
/// upper-tail quantiles should be asked of <c>Isf</c>, whose Sf does not round to 1.
/// </para>
/// <para>
/// Generic fallbacks sum the mass function; families override them with closed forms.
/// </para>
/// </remarks>
public abstract class DiscreteDistribution : IUnivariateDistribution
{
    private DistributionMoments? _moments;

    /// <summary>Validates and stores the location.</summary>
    protected DiscreteDistribution(double loc)
    {
        if (!double.IsFinite(loc)) throw new ArgumentException("loc must be finite.", nameof(loc));
        Loc = loc;
    }

    /// <summary>The location shift (normally an integer).</summary>
    public double Loc { get; }

    /// <inheritdoc />
    public abstract string Name { get; }

    /// <inheritdoc />
    public bool IsDiscrete => true;

    /// <summary>Names of the shape parameters in scipy's order.</summary>
    public virtual string[] ShapeNames => [];

    /// <summary>The shape parameter values.</summary>
    public virtual double[] Shapes => [];

    /// <summary>The same family with other parameters; throws <see cref="ArgumentException"/> when invalid.</summary>
    public abstract DiscreteDistribution WithParameters(double[] shapes, double loc);

    // ------------------------------------------------------------------ standard-form hooks

    /// <summary>Smallest integer in the standard support.</summary>
    protected virtual double Lower0 => 0.0;

    /// <summary>Largest integer in the standard support (may be +∞).</summary>
    protected virtual double Upper0 => double.PositiveInfinity;

    /// <summary>Standard mass at the integer k inside the support.</summary>
    protected abstract double Pmf0(double k);

    /// <summary>Standard log mass.</summary>
    protected virtual double LogPmf0(double k) => Math.Log(Pmf0(k));

    /// <summary>Standard CDF at the integer k, Lower0 ≤ k &lt; Upper0. Fallback: summation.</summary>
    protected virtual double Cdf0(double k)
    {
        var s = 0.0;
        for (var j = Lower0; j <= k; j++) s += Pmf0(j);
        return Math.Min(s, 1.0);
    }

    /// <summary>
    /// Standard survival function P(X &gt; k). Fallback: sums the upper tail directly until it stops
    /// changing (so it does not cancel), or 1 − Cdf₀ when the tail is the larger part.
    /// </summary>
    protected virtual double Sf0(double k)
    {
        var s = 0.0;
        for (var j = k + 1; j <= Upper0; j++)
        {
            var p = Pmf0(j);
            s += p;
            if (p <= s * 1e-17 && (j - k) > 10) break;
            if (j - k > 1e7) return Math.Max(0.0, 1.0 - Cdf0(k));
        }
        return Math.Min(s, 1.0);
    }

    /// <summary>Standard log CDF: log Cdf₀, or log1p(−Sf₀) once the CDF passes ½.</summary>
    protected virtual double LogCdf0(double k)
    {
        var c = Cdf0(k);
        return c > 0.5 ? Special.Log1p(-Sf0(k)) : Math.Log(c);
    }

    /// <summary>Standard log survival function, with the same switch as <see cref="LogCdf0"/>.</summary>
    protected virtual double LogSf0(double k)
    {
        var s = Sf0(k);
        return s > 0.5 ? Special.Log1p(-Cdf0(k)) : Math.Log(s);
    }

    /// <summary>A real-valued guess for the q-quantile (a continuous approximation); the exact search starts there.</summary>
    protected virtual double PpfGuess0(double q)
    {
        var m = _moments ??= Moments0();
        if (double.IsFinite(m.Mean) && double.IsFinite(m.Variance))
            return m.Mean + Math.Sqrt(m.Variance) * Special.Ndtri(q);
        return double.IsFinite(Lower0) ? Lower0 : 0.0;
    }

    /// <summary>Standard quantile for 0 &lt; q &lt; 1: smallest k with Cdf₀(k) ≥ q.</summary>
    protected virtual double Ppf0(double q)
    {
        // Exactly scipy's predicate. Testing Sf(k) ≤ 1 − q above the median is the same thing in exact
        // arithmetic but not in floating point: Ppf(Cdf(k)) must return k, and only Cdf can promise that.
        return SmallestSatisfying(k => Cdf0Safe(k) >= q, PpfGuess0(q));
    }

    /// <summary>Standard inverse survival function: smallest k with Sf₀(k) ≤ q.</summary>
    protected virtual double Isf0(double q)
    {
        return SmallestSatisfying(k => Sf0Safe(k) <= q, PpfGuess0(1.0 - q));
    }

    private double Cdf0Safe(double k) => k < Lower0 ? 0.0 : k >= Upper0 ? 1.0 : Cdf0(k);

    private double Sf0Safe(double k) => k < Lower0 ? 1.0 : k >= Upper0 ? 0.0 : Sf0(k);

    /// <summary>Smallest support integer satisfying a monotone predicate, by galloping from a guess then bisecting.</summary>
    private double SmallestSatisfying(Func<double, bool> ok, double guess)
    {
        double k = double.IsFinite(guess) ? Math.Round(guess) : (double.IsFinite(Lower0) ? Lower0 : 0.0);
        k = Math.Clamp(k, Lower0, Upper0);
        if (double.IsInfinity(k)) k = 0;
        double lo, hi; // invariant: ok(hi) is true, ok(lo) false (lo may be Lower0 - 1)
        var step = 1.0;
        if (ok(k))
        {
            hi = k;
            while (true)
            {
                lo = k - step;
                if (lo < Lower0) { lo = Lower0 - 1; break; }
                if (!ok(lo)) break;
                hi = lo;
                step *= 2;
            }
        }
        else
        {
            lo = k;
            while (true)
            {
                hi = k + step;
                if (hi >= Upper0) { hi = Upper0; break; }
                if (ok(hi)) break;
                lo = hi;
                step *= 2;
                if (step > 1e300) return double.PositiveInfinity;
            }
        }
        while (hi - lo > 1)
        {
            var mid = Math.Floor(lo + 0.5 * (hi - lo));
            if (ok(mid)) hi = mid; else lo = mid;
        }
        return hi;
    }

    /// <summary>Lower index at which generic sums over a support unbounded below start.</summary>
    private double SumLow() => double.IsFinite(Lower0) ? Lower0 : Ppf0(1e-20) - 10;

    /// <summary>Upper index at which generic sums over an infinite support stop.</summary>
    private double SumLimit()
    {
        if (double.IsFinite(Upper0)) return Upper0;
        return Isf0(1e-20) + 10;
    }

    /// <summary>Standard mean, variance, skewness, excess kurtosis. Fallback: from <see cref="Munp0"/>.</summary>
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

    /// <summary>Standard raw moment by summation of kⁿ pmf(k).</summary>
    protected virtual double Munp0(int n)
    {
        var hi = SumLimit();
        var s = 0.0;
        for (var k = SumLow(); k <= hi; k++) s += Math.Pow(k, n) * Pmf0(k);
        return s;
    }

    /// <summary>Shannon entropy of the standard form, by summation.</summary>
    protected virtual double Entropy0()
    {
        var hi = SumLimit();
        var s = 0.0;
        for (var k = SumLow(); k <= hi; k++)
        {
            var lp = LogPmf0(k);
            if (double.IsFinite(lp)) s -= Math.Exp(lp) * lp;
        }
        return s;
    }

    /// <summary>One standard variate. Fallback: exact inversion.</summary>
    protected virtual double Rvs0(GraviRandom rng) => Ppf0(StatsRandom.U(rng));

    // ------------------------------------------------------------------ public API

    /// <inheritdoc />
    public (double Lower, double Upper) Support() => (Lower0 + Loc, Upper0 + Loc);

    /// <summary>Probability mass function; 0 off the integers and outside the support.</summary>
    public double Pmf(double x)
    {
        var k = x - Loc;
        if (double.IsNaN(k)) return double.NaN;
        if (k != Math.Floor(k) || k < Lower0 || k > Upper0) return 0.0;
        return Pmf0(k);
    }

    /// <summary>Log of the probability mass function.</summary>
    public double LogPmf(double x)
    {
        var k = x - Loc;
        if (double.IsNaN(k)) return double.NaN;
        if (k != Math.Floor(k) || k < Lower0 || k > Upper0) return double.NegativeInfinity;
        return LogPmf0(k);
    }

    /// <inheritdoc />
    public double Density(double x) => Pmf(x);

    /// <inheritdoc />
    public double LogDensity(double x) => LogPmf(x);

    /// <inheritdoc />
    public double Cdf(double x)
    {
        var k = Math.Floor(x - Loc);
        if (double.IsNaN(k)) return double.NaN;
        if (k < Lower0) return 0.0;
        if (k >= Upper0) return 1.0;
        return Cdf0(k);
    }

    /// <inheritdoc />
    public double LogCdf(double x)
    {
        var k = Math.Floor(x - Loc);
        if (double.IsNaN(k)) return double.NaN;
        if (k < Lower0) return double.NegativeInfinity;
        if (k >= Upper0) return 0.0;
        return LogCdf0(k);
    }

    /// <inheritdoc />
    public double Sf(double x)
    {
        var k = Math.Floor(x - Loc);
        if (double.IsNaN(k)) return double.NaN;
        if (k < Lower0) return 1.0;
        if (k >= Upper0) return 0.0;
        return Sf0(k);
    }

    /// <inheritdoc />
    public double LogSf(double x)
    {
        var k = Math.Floor(x - Loc);
        if (double.IsNaN(k)) return double.NaN;
        if (k < Lower0) return 0.0;
        if (k >= Upper0) return double.NegativeInfinity;
        return LogSf0(k);
    }

    /// <inheritdoc />
    public double Ppf(double q)
    {
        if (!(q >= 0 && q <= 1)) return double.NaN;
        if (q == 0) return Lower0 - 1 + Loc;
        if (q == 1) return Upper0 + Loc;
        return Ppf0(q) + Loc;
    }

    /// <inheritdoc />
    public double Isf(double q)
    {
        if (!(q >= 0 && q <= 1)) return double.NaN;
        if (q == 0) return Upper0 + Loc;
        if (q == 1) return Lower0 - 1 + Loc;
        return Isf0(q) + Loc;
    }

    /// <inheritdoc />
    public DistributionMoments Stats()
    {
        var m = _moments ??= Moments0();
        return m with { Mean = m.Mean + Loc };
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
    public double Entropy() => Entropy0();

    /// <inheritdoc />
    public double Moment(int n)
    {
        if (n < 0) throw new ArgumentException("Moment order must be non-negative.", nameof(n));
        if (n == 0) return 1.0;
        var raw = new double[n + 1];
        raw[0] = 1;
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
                if (n >= 4) raw[4] = (s.Kurtosis + 3) * v * v + 4 * m * mu3 + 6 * m * m * v + m * m * m * m;
            }
        }
        else for (var k = 1; k <= n; k++) raw[k] = Munp0(k);
        if (Loc == 0) return raw[n];
        var total = 0.0;
        for (var k = 0; k <= n; k++) total += Special.Binom(n, k) * Math.Pow(Loc, n - k) * raw[k];
        return total;
    }

    /// <inheritdoc />
    public (double Lower, double Upper) Interval(double confidence)
    {
        if (!(confidence >= 0 && confidence <= 1)) throw new ArgumentException("confidence must lie in [0, 1].", nameof(confidence));
        var a = 0.5 * (1.0 - confidence);
        return (Ppf(a), Ppf(1.0 - a));
    }

    /// <summary>
    /// E[f(X)] over lb ≤ X ≤ ub by summation, scipy's <c>expect</c>; infinite ranges are summed until
    /// the remaining tail mass is below 1e-20. With <paramref name="conditional"/>, divided by P(lb ≤ X ≤ ub).
    /// </summary>
    public double Expect(Func<double, double>? func = null, double? lb = null, double? ub = null, bool conditional = false)
    {
        func ??= x => x;
        var lo = Math.Max(Math.Ceiling((lb ?? double.NegativeInfinity) - Loc), Lower0);
        var hi = Math.Min(Math.Floor((ub ?? double.PositiveInfinity) - Loc), Upper0);
        if (double.IsInfinity(lo)) lo = SumLow();
        if (double.IsInfinity(hi)) hi = Math.Max(lo, SumLimit());
        var s = 0.0;
        var mass = 0.0;
        for (var k = lo; k <= hi; k++)
        {
            var p = Pmf0(k);
            if (p == 0) continue;
            s += func(k + Loc) * p;
            mass += p;
        }
        return conditional ? s / mass : s;
    }

    /// <inheritdoc />
    public double Rvs(GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        return Rvs0(rng) + Loc;
    }

    /// <inheritdoc />
    public double[] Rvs(int size, GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        if (size < 0) throw new ArgumentException("size must be non-negative.", nameof(size));
        var r = new double[size];
        for (var i = 0; i < size; i++) r[i] = Rvs0(rng) + Loc;
        return r;
    }

    /// <summary><paramref name="size"/> random variates; a null seed draws a fresh one.</summary>
    public double[] Rvs(int size, int? seed = null) => Rvs(size, StatsDistributions.MakeRng(seed));

    // ------------------------------------------------------------------ vectorised API

    /// <summary>Element-wise <see cref="Pmf(double)"/>.</summary>
    public double[] Pmf(double[] x) => StatsDistributions.Map(x, Pmf);
    /// <summary>Element-wise <see cref="LogPmf(double)"/>.</summary>
    public double[] LogPmf(double[] x) => StatsDistributions.Map(x, LogPmf);
    /// <summary>Element-wise <see cref="Cdf(double)"/>.</summary>
    public double[] Cdf(double[] x) => StatsDistributions.Map(x, Cdf);
    /// <summary>Element-wise <see cref="Sf(double)"/>.</summary>
    public double[] Sf(double[] x) => StatsDistributions.Map(x, Sf);
    /// <summary>Element-wise <see cref="Ppf(double)"/>.</summary>
    public double[] Ppf(double[] q) => StatsDistributions.Map(q, Ppf);
    /// <summary>Element-wise <see cref="Isf(double)"/>.</summary>
    public double[] Isf(double[] q) => StatsDistributions.Map(q, Isf);
    /// <summary>Element-wise <see cref="Pmf(double)"/> over an array of any shape.</summary>
    public NdArray Pmf(NdArray x) => Special.Apply(x, Pmf);
    /// <summary>Element-wise <see cref="Cdf(double)"/> over an array of any shape.</summary>
    public NdArray Cdf(NdArray x) => Special.Apply(x, Cdf);

    /// <summary>The negative log-likelihood of integer data; +∞ when a point lies outside the support.</summary>
    public double NegativeLogLikelihood(ReadOnlySpan<double> data)
    {
        var total = 0.0;
        foreach (var x in data)
        {
            var lp = LogPmf(x);
            if (!double.IsFinite(lp)) return double.PositiveInfinity;
            total -= lp;
        }
        return total;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var parts = new List<string>();
        var names = ShapeNames;
        var vals = Shapes;
        for (var i = 0; i < vals.Length; i++) parts.Add($"{names[i]}={vals[i]:R}");
        parts.Add($"loc={Loc:R}");
        return $"{Name}({string.Join(", ", parts)})";
    }
}
