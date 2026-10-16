using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.StatsDistMath;

namespace Gravicode.Science.SciNet;

/// <summary>Poisson-mixture sums shared by the noncentral families.</summary>
internal static class StatsNoncentral
{
    /// <summary>
    /// Σ_j Pois(j; λ)·term(j), summed outward from the Poisson mode in both directions until the
    /// weights stop mattering (Ding's scheme). Every term is non-negative, so each tail of a
    /// noncentral distribution is computed directly and never as a complement.
    /// </summary>
    public static double PoissonMixture(double lambda, Func<int, double> term)
    {
        if (lambda == 0) return term(0);
        var mode = (int)Math.Floor(lambda);
        var logLam = Math.Log(lambda);
        double W(int j) => Math.Exp(-lambda + j * logLam - Special.GammaLn(j + 1));
        var sum = 0.0;
        var wsum = 0.0;
        for (var j = mode; ; j++)
        {
            var w = W(j);
            var t = w * term(j);
            sum += t;
            wsum += w;
            if ((w < 1e-17 * wsum && t <= 1e-17 * sum) || j - mode > 100000) break;
        }
        for (var j = mode - 1; j >= 0; j--)
        {
            var w = W(j);
            var t = w * term(j);
            sum += t;
            if (w < 1e-17 * wsum && t <= 1e-17 * sum) break;
        }
        return sum;
    }

    /// <summary>Raw moments E[Xᵏ], k ≤ 4, of a noncentral chi-squared from its cumulants 2^{r−1}(r−1)!(df + r·nc).</summary>
    public static double Ncx2Raw(double df, double nc, int k)
    {
        double k1 = df + nc, k2 = 2 * (df + 2 * nc), k3 = 8 * (df + 3 * nc), k4 = 48 * (df + 4 * nc);
        return k switch
        {
            0 => 1,
            1 => k1,
            2 => k2 + k1 * k1,
            3 => k3 + 3 * k2 * k1 + k1 * k1 * k1,
            4 => k4 + 4 * k3 * k1 + 3 * k2 * k2 + 6 * k2 * k1 * k1 + k1 * k1 * k1 * k1,
            _ => throw new ArgumentOutOfRangeException(nameof(k)),
        };
    }
}

/// <summary>The noncentral chi-squared distribution, scipy's <c>ncx2</c>.</summary>
/// <remarks>The density uses the exponentially scaled Bessel function, ½e^{−(√x−√λ)²/2}(x/λ)^{(k−2)/4}Ive(k/2−1, √(λx)),
/// so it neither overflows nor underflows for large arguments; CDF and survival function are
/// separate Poisson mixtures of incomplete gamma functions.</remarks>
public sealed class NonCentralChi2Distribution : ContinuousDistribution
{
    /// <summary>Noncentral chi-squared with <paramref name="df"/> degrees of freedom and noncentrality <paramref name="nc"/>.</summary>
    public NonCentralChi2Distribution(double df, double nc, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        Df = StatsDistributions.Positive(df, nameof(df));
        if (!(nc >= 0) || double.IsInfinity(nc)) throw new ArgumentException("nc must be non-negative and finite.", nameof(nc));
        Nc = nc;
    }

    /// <summary>Degrees of freedom.</summary>
    public double Df { get; }
    /// <summary>Noncentrality parameter λ.</summary>
    public double Nc { get; }

    /// <inheritdoc />
    public override string Name => "ncx2";
    /// <inheritdoc />
    public override string[] ShapeNames => ["df", "nc"];
    /// <inheritdoc />
    public override double[] Shapes => [Df, Nc];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new NonCentralChi2Distribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z)
    {
        if (Nc == 0 || z == 0)
        {
            var chi = Special.Xlogy(Df / 2 - 1, z) - z / 2 - Special.GammaLn(Df / 2) - Ln2 * Df / 2;
            return Nc == 0 ? chi : chi - Nc / 2;
        }
        var d = Math.Sqrt(z) - Math.Sqrt(Nc);
        return -Ln2 - 0.5 * d * d + (Df / 4 - 0.5) * Math.Log(z / Nc) + Math.Log(Special.Ive(Df / 2 - 1, Math.Sqrt(Nc * z)));
    }
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Min(1, StatsNoncentral.PoissonMixture(Nc / 2, j => Special.GammaInc(Df / 2 + j, z / 2)));
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Min(1, StatsNoncentral.PoissonMixture(Nc / 2, j => Special.GammaIncC(Df / 2 + j, z / 2)));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        double k = Df, l = Nc;
        return new(k + l, 2 * (k + 2 * l), Math.Pow(2, 1.5) * (k + 3 * l) / Math.Pow(k + 2 * l, 1.5), 12 * (k + 4 * l) / ((k + 2 * l) * (k + 2 * l)));
    }
    /// <inheritdoc />
    protected override double Munp0(int n) => n <= 4 ? StatsNoncentral.Ncx2Raw(Df, Nc, n) : base.Munp0(n);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        // A Poisson mixture of central chi-squares: exact for every df, including df < 1.
        var j = StatsRandom.Poisson(rng, Nc / 2);
        return 2 * StatsRandom.Gamma(rng, Df / 2 + j);
    }
}

/// <summary>The noncentral F distribution, scipy's <c>ncf</c>.</summary>
/// <remarks>Every function is a Poisson(nc/2) mixture over beta functions in y = dfn·x/(dfn·x + dfd).</remarks>
public sealed class NonCentralFDistribution : ContinuousDistribution
{
    /// <summary>Noncentral F.</summary>
    public NonCentralFDistribution(double dfn, double dfd, double nc, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        Dfn = StatsDistributions.Positive(dfn, nameof(dfn));
        Dfd = StatsDistributions.Positive(dfd, nameof(dfd));
        if (!(nc >= 0) || double.IsInfinity(nc)) throw new ArgumentException("nc must be non-negative and finite.", nameof(nc));
        Nc = nc;
    }

    /// <summary>Numerator degrees of freedom.</summary>
    public double Dfn { get; }
    /// <summary>Denominator degrees of freedom.</summary>
    public double Dfd { get; }
    /// <summary>Noncentrality parameter.</summary>
    public double Nc { get; }

    /// <inheritdoc />
    public override string Name => "ncf";
    /// <inheritdoc />
    public override string[] ShapeNames => ["dfn", "dfd", "nc"];
    /// <inheritdoc />
    public override double[] Shapes => [Dfn, Dfd, Nc];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new NonCentralFDistribution(shapes[0], shapes[1], shapes[2], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z)
    {
        var den = Dfn * z + Dfd;
        var y = Dfn * z / den;
        var ly = Math.Log(Dfn * z) - Math.Log(den);
        var l1y = Math.Log(Dfd) - Math.Log(den);
        var jac = Math.Log(Dfn) + Math.Log(Dfd) - 2 * Math.Log(den);
        if (z == 0) return Dfn < 2 ? double.PositiveInfinity : Dfn == 2 ? Math.Exp(-Nc / 2) : 0.0;
        _ = y;
        return StatsNoncentral.PoissonMixture(Nc / 2, j =>
        {
            var a = Dfn / 2 + j;
            return Math.Exp((a - 1) * ly + (Dfd / 2 - 1) * l1y - Special.BetaLn(a, Dfd / 2) + jac);
        });
    }
    private double Y(double z) => Dfn * z / (Dfn * z + Dfd);
    private double Y1(double z) => Dfd / (Dfn * z + Dfd);
    /// <inheritdoc />
    protected override double Cdf0(double z)
    {
        double y = Y(z), y1 = Y1(z);
        return Math.Min(1, StatsNoncentral.PoissonMixture(Nc / 2, j => y <= 0.5 ? Special.BetaInc(Dfn / 2 + j, Dfd / 2, y) : Special.BetaIncC(Dfd / 2, Dfn / 2 + j, y1)));
    }
    /// <inheritdoc />
    protected override double Sf0(double z)
    {
        double y = Y(z), y1 = Y1(z);
        return Math.Min(1, StatsNoncentral.PoissonMixture(Nc / 2, j => y <= 0.5 ? Special.BetaIncC(Dfn / 2 + j, Dfd / 2, y) : Special.BetaInc(Dfd / 2, Dfn / 2 + j, y1)));
    }
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        if (n > 4) return base.Munp0(n);
        if (Dfd <= 2 * n) return double.PositiveInfinity;
        // F = (X/dfn)/(Y/dfd): E[Fⁿ] = (dfd/dfn)ⁿ E[Xⁿ] E[Y⁻ⁿ], E[Y⁻ⁿ] = Γ(dfd/2 − n)/(2ⁿΓ(dfd/2)).
        var inv = Math.Exp(Special.GammaLn(Dfd / 2 - n) - Special.GammaLn(Dfd / 2)) / Math.Pow(2, n);
        return Math.Pow(Dfd / Dfn, n) * StatsNoncentral.Ncx2Raw(Dfn, Nc, n) * inv;
    }
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var m = base.Moments0();
        return new(Dfd > 2 ? m.Mean : double.PositiveInfinity, Dfd > 4 ? m.Variance : double.PositiveInfinity,
            Dfd > 6 ? m.Skewness : double.NaN, Dfd > 8 ? m.Kurtosis : double.NaN);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        var j = StatsRandom.Poisson(rng, Nc / 2);
        var x = 2 * StatsRandom.Gamma(rng, Dfn / 2 + j);
        var y = 2 * StatsRandom.Gamma(rng, Dfd / 2);
        return x / Dfn / (y / Dfd);
    }
}

/// <summary>The noncentral Student t distribution, scipy's <c>nct</c>.</summary>
/// <remarks>
/// The CDF is Lenth's AS 243 series (with the AS R80 correction), the density the Poisson/hypergeometric
/// series scipy used before it switched to Boost. Accurate to ~1e-12 in the body; in the far upper
/// tail the CDF is a complement of the series and loses relative accuracy (Boost does better there).
/// </remarks>
public sealed class NonCentralTDistribution : ContinuousDistribution
{
    /// <summary>Noncentral t.</summary>
    public NonCentralTDistribution(double df, double nc, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        Df = StatsDistributions.Positive(df, nameof(df));
        Nc = StatsDistributions.Finite(nc, nameof(nc));
    }

    /// <summary>Degrees of freedom.</summary>
    public double Df { get; }
    /// <summary>Noncentrality parameter.</summary>
    public double Nc { get; }

    /// <inheritdoc />
    public override string Name => "nct";
    /// <inheritdoc />
    public override string[] ShapeNames => ["df", "nc"];
    /// <inheritdoc />
    public override double[] Shapes => [Df, Nc];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new NonCentralTDistribution(shapes[0], shapes[1], loc, scale);

    /// <inheritdoc />
    protected override double Pdf0(double x)
    {
        // f(x) = df/x · [F(x√(1+2/df); df+2, nc) − F(x; df, nc)] is exact but cancels near 0;
        // the series form below is scipy's pre-Boost _pdf.
        double n = Df, nc = Nc;
        var x2 = x * x;
        var fac1 = n + x2;
        var trm = n / 2 * Math.Log(n) + Special.GammaLn(n + 1) - (n * Ln2 + nc * nc / 2 + n / 2 * Math.Log(fac1) + Special.GammaLn(n / 2));
        var px = Math.Exp(trm);
        var valF = nc * nc * x2 / (2 * fac1);
        var t1 = Math.Sqrt(2) * nc * x * Special.Hyp1f1(n / 2 + 1, 1.5, valF) / (fac1 * Special.Gamma((n + 1) / 2));
        var t2 = Special.Hyp1f1((n + 1) / 2, 0.5, valF) / (Math.Sqrt(fac1) * Special.Gamma(n / 2 + 1));
        return Math.Max(0, px * (t1 + t2));
    }

    /// <summary>AS 243: P(T ≤ t) for t ≥ 0 with noncentrality delta (either sign).</summary>
    private static double Tnc(double t, double df, double delta)
    {
        const double errmax = 1e-14;
        const int itrmax = 5000;
        var negdel = false;
        double tt = t, del = delta;
        if (t < 0) { negdel = true; tt = -t; del = -delta; }
        var x = tt * tt / (tt * tt + df);
        double tnc = 0;
        if (x > 0)
        {
            var lambda = del * del;
            var p = 0.5 * Math.Exp(-0.5 * lambda);
            var q = Math.Sqrt(2 / Math.PI) * p * del;
            var s = -0.5 * Special.Expm1(-0.5 * lambda); // 0.5 − p without cancellation
            var a = 0.5;
            var b = 0.5 * df;
            var rxb = Math.Pow(1 - x, b);
            var albeta = 0.5 * Math.Log(Math.PI) + Special.GammaLn(b) - Special.GammaLn(0.5 + b);
            var xodd = Special.BetaInc(a, b, x);
            var godd = 2 * rxb * Math.Exp(a * Math.Log(x) - albeta);
            var bx = b * x;
            var xeven = bx < 2.2e-16 ? bx : -Special.Expm1(b * Special.Log1p(-x));
            var geven = bx * rxb;
            tnc = p * xodd + q * xeven;
            for (var en = 1.0; en <= itrmax; en++)
            {
                a += 1;
                xodd -= godd;
                xeven -= geven;
                godd *= x * (a + b - 1) / a;
                geven *= x * (a + b - 0.5) / (a + 0.5);
                p *= lambda / (2 * en);
                q *= lambda / (2 * en + 1);
                s -= p;
                tnc += p * xodd + q * xeven;
                var errbd = 2 * s * (xodd - godd);
                if (Math.Abs(errbd) <= errmax && en > lambda / 2) break;
            }
        }
        tnc += Special.Ndtr(-del);
        if (negdel) tnc = 1 - tnc;
        return Math.Clamp(tnc, 0, 1);
    }

    /// <inheritdoc />
    protected override double Cdf0(double z) => z >= 0 ? Tnc(z, Df, Nc) : 1 - Tnc(-z, Df, -Nc);
    /// <inheritdoc />
    protected override double Sf0(double z) => z >= 0 ? 1 - Tnc(z, Df, Nc) : Tnc(-z, Df, -Nc);
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        if (n > 4) return base.Munp0(n);
        if (Df <= n) return double.PositiveInfinity;
        double nc = Nc;
        var zn = n switch { 1 => nc, 2 => nc * nc + 1, 3 => nc * nc * nc + 3 * nc, _ => nc * nc * nc * nc + 6 * nc * nc + 3 };
        return Math.Pow(Df / 2, n / 2.0) * Math.Exp(Special.GammaLn((Df - n) / 2) - Special.GammaLn(Df / 2)) * zn;
    }
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var m = base.Moments0();
        return new(Df > 1 ? m.Mean : double.NaN, Df > 2 ? m.Variance : double.NaN, Df > 3 ? m.Skewness : double.NaN, Df > 4 ? m.Kurtosis : double.NaN);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
        => (StatsRandom.Normal(rng) + Nc) / Math.Sqrt(2 * StatsRandom.Gamma(rng, Df / 2) / Df);
}

/// <summary>The Rice distribution, scipy's <c>rice</c>: the length of a 2-D normal vector with offset b.</summary>
public sealed class RiceDistribution : ContinuousDistribution
{
    private readonly NonCentralChi2Distribution _ncx2;

    /// <summary>Rice with shape <paramref name="b"/> ≥ 0.</summary>
    public RiceDistribution(double b, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        if (!(b >= 0) || double.IsInfinity(b)) throw new ArgumentException("b must be non-negative and finite.", nameof(b));
        B = b;
        _ncx2 = new NonCentralChi2Distribution(2, b * b);
    }

    /// <summary>The shape parameter.</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "rice";
    /// <inheritdoc />
    public override string[] ShapeNames => ["b"];
    /// <inheritdoc />
    public override double[] Shapes => [B];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new RiceDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => z * Math.Exp(-0.5 * (z - B) * (z - B)) * Special.I0e(z * B);
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(z) - 0.5 * (z - B) * (z - B) + Math.Log(Special.I0e(z * B));
    /// <inheritdoc />
    protected override double Cdf0(double z) => _ncx2.Cdf(z * z);
    /// <inheritdoc />
    protected override double Sf0(double z) => _ncx2.Sf(z * z);
    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        var b2 = B * B / 2;
        return Math.Pow(2, n / 2.0) * Math.Exp(-b2) * Special.Gamma(1 + n / 2.0) * Special.Hyp1f1(1 + n / 2.0, 1, b2);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        var a = StatsRandom.Normal(rng) + B;
        var c = StatsRandom.Normal(rng);
        return Math.Sqrt(a * a + c * c);
    }
}

/// <summary>The von Mises (circular normal) distribution on [−π, π], scipy's <c>vonmises</c>.</summary>
/// <remarks>The CDF is the Fourier series ½ + x/2π + (1/π)Σ (Iₙ(κ)/I₀(κ)) sin(nx)/n with the Bessel
/// ratios from Miller's backward recurrence, which is stable where the forward one is not.
/// Like scipy's, it is accurate in absolute rather than relative terms near ±π.</remarks>
public sealed class VonMisesDistribution : ContinuousDistribution
{
    private readonly double[] _ratios;

    /// <summary>von Mises with concentration <paramref name="kappa"/>.</summary>
    public VonMisesDistribution(double kappa, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        Kappa = StatsDistributions.Positive(kappa, nameof(kappa));
        var n = (int)Math.Ceiling(50 + 9 * Math.Sqrt(kappa));
        _ratios = new double[n + 1]; // _ratios[k] = I_k(κ)/I_0(κ)
        var r = new double[n + 2];
        var m = n + 30;
        double rr = 0;
        for (var k = m; k >= 1; k--)
        {
            rr = 1 / (2 * k / kappa + rr); // I_k/I_{k−1}
            if (k <= n + 1) r[k] = rr;
        }
        _ratios[0] = 1;
        for (var k = 1; k <= n; k++) _ratios[k] = _ratios[k - 1] * r[k];
    }

    /// <summary>The concentration parameter κ.</summary>
    public double Kappa { get; }

    /// <inheritdoc />
    public override string Name => "vonmises";
    /// <inheritdoc />
    public override string[] ShapeNames => ["kappa"];
    /// <inheritdoc />
    public override double[] Shapes => [Kappa];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new VonMisesDistribution(shapes[0], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => -Math.PI;
    /// <inheritdoc />
    protected override double Upper0 => Math.PI;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(Kappa * (Math.Cos(z) - 1)) / (2 * Math.PI * Special.I0e(Kappa));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Kappa * (Math.Cos(z) - 1) - Math.Log(2 * Math.PI * Special.I0e(Kappa));
    /// <inheritdoc />
    protected override double Cdf0(double z)
    {
        var s = 0.0;
        for (var k = _ratios.Length - 1; k >= 1; k--) s += _ratios[k] * Math.Sin(k * z) / k;
        return Math.Clamp(0.5 + z / (2 * Math.PI) + s / Math.PI, 0, 1);
    }
    /// <inheritdoc />
    protected override double Sf0(double z) => Cdf0(-z);
    /// <inheritdoc />
    protected override double Entropy0()
    {
        var r = Special.I1e(Kappa) / Special.I0e(Kappa);
        return -Kappa * r + Math.Log(2 * Math.PI * Special.I0e(Kappa)) + Kappa;
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        // Best & Fisher (1979).
        var k = Kappa;
        var tau = 1 + Math.Sqrt(1 + 4 * k * k);
        var rho = (tau - Math.Sqrt(2 * tau)) / (2 * k);
        var r = (1 + rho * rho) / (2 * rho);
        while (true)
        {
            var u1 = StatsRandom.U(rng);
            var zz = Math.Cos(Math.PI * u1);
            var f = (1 + r * zz) / (r + zz);
            var c = k * (r - f);
            var u2 = StatsRandom.U(rng);
            if (c * (2 - c) - u2 > 0 || Math.Log(c / u2) + 1 - c >= 0)
            {
                var theta = Math.Acos(Math.Clamp(f, -1, 1));
                return StatsRandom.U(rng) < 0.5 ? -theta : theta;
            }
        }
    }
}

/// <summary>
/// The limiting distribution of √n·Dₙ, the scaled two-sided Kolmogorov–Smirnov statistic, scipy's
/// <c>kstwobign</c>.
/// </summary>
/// <remarks>The survival function is <see cref="Special.Kolmogorov"/>; the CDF below 1 is the Jacobi
/// series √(2π)/x Σ e^{−(2k−1)²π²/(8x²)} directly, so tiny CDF values keep their relative accuracy.</remarks>
public sealed class KSTwoBignDistribution : ContinuousDistribution
{
    /// <summary>Kolmogorov's limiting distribution.</summary>
    public KSTwoBignDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "kstwobign";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new KSTwoBignDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;

    /// <inheritdoc />
    protected override double Pdf0(double x)
    {
        if (x <= 0) return 0.0;
        if (x < 1)
        {
            var s = 0.0;
            for (var k = 1; k < 20; k++)
            {
                var c = (2 * k - 1) * (2 * k - 1) * Math.PI * Math.PI / 8;
                var e = Math.Exp(-c / (x * x));
                if (e == 0) break;
                s += e * (2 * c / (x * x * x * x) - 1 / (x * x));
            }
            return Sqrt2Pi * s;
        }
        var t = 0.0;
        for (var k = 1; k < 100; k++)
        {
            var term = k * k * Math.Exp(-2 * k * k * x * x);
            t += (k % 2 == 1 ? 1 : -1) * term;
            if (term < 1e-20 * Math.Abs(t)) break;
        }
        return 8 * x * t;
    }
    /// <inheritdoc />
    protected override double Cdf0(double x)
    {
        if (x >= 1) return 1 - Special.Kolmogorov(x);
        var s = 0.0;
        for (var k = 1; k < 20; k++)
        {
            var e = Math.Exp(-(2 * k - 1) * (2 * k - 1) * Math.PI * Math.PI / (8 * x * x));
            if (e == 0) break;
            s += e;
        }
        return Sqrt2Pi / x * s;
    }
    /// <inheritdoc />
    protected override double Sf0(double x) => Special.Kolmogorov(x);
    /// <inheritdoc />
    protected override double Isf0(double q) => Special.KolmogorovInverse(q);
    /// <inheritdoc />
    protected override double Ppf0(double q) => q > 0.5 ? Special.KolmogorovInverse(1 - q) : base.Ppf0(q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        // Mean √(π/2)·ln 2 and E[X²] = π²/12 are closed; skewness and kurtosis by quadrature.
        var b = base.Moments0();
        var mean = Math.Sqrt(Math.PI / 2) * Math.Log(2);
        return new(mean, Math.PI * Math.PI / 12 - mean * mean, b.Skewness, b.Kurtosis);
    }
}

/// <summary>
/// The exact distribution of the two-sided Kolmogorov–Smirnov statistic Dₙ for sample size n,
/// scipy's <c>kstwo</c>.
/// </summary>
/// <remarks>
/// <para>Exact closed forms at the ends (d ≤ 1/(2n): 0; d ≤ 1/n: n!(2d − 1/n)ⁿ; d ≥ 1 − 1/n: survival
/// 2(1 − d)ⁿ); for d ≥ ½ the two one-sided events are disjoint so the survival function is exactly
/// 2·<see cref="Special.Smirnov"/>. In between, the CDF is the Durbin matrix method of Marsaglia,
/// Tsang &amp; Wang (2003), O(k³ log n) with k = ⌈nd⌉. Survival probabilities below 1e-12 in that middle
/// range, where 1 − CDF has no digits left, fall back to 2·Smirnov, which over-counts only the
/// probability of both one-sided deviations at once. The density is a Richardson-extrapolated
/// numerical derivative (~1e-7 relative). This is not a port of scipy's <c>_kolmogn</c>.</para>
/// </remarks>
public sealed class KSTwoDistribution : ContinuousDistribution
{
    /// <summary>Distribution of Dₙ for <paramref name="n"/> observations.</summary>
    public KSTwoDistribution(int n, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        if (n < 1) throw new ArgumentException("n must be at least 1.", nameof(n));
        N = n;
    }

    /// <summary>The sample size.</summary>
    public int N { get; }

    /// <inheritdoc />
    public override string Name => "kstwo";
    /// <inheritdoc />
    public override string[] ShapeNames => ["n"];
    /// <inheritdoc />
    public override double[] Shapes => [N];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale)
    {
        if (shapes[0] != Math.Floor(shapes[0])) throw new ArgumentException("n must be an integer.", nameof(shapes));
        return new KSTwoDistribution((int)shapes[0], loc, scale);
    }
    /// <inheritdoc />
    protected override double Lower0 => 0.5 / N;
    /// <inheritdoc />
    protected override double Upper0 => 1.0;

    /// <inheritdoc />
    protected override double Cdf0(double d)
    {
        var n = N;
        if (d <= 0.5 / n) return 0.0;
        if (d <= 1.0 / n) return Math.Exp(Special.GammaLn(n + 1) + n * Math.Log(2 * d - 1.0 / n));
        if (d >= 0.5) return 1 - Sf0(d);
        var c = Durbin(n, d);
        return c;
    }

    /// <inheritdoc />
    protected override double Sf0(double d)
    {
        var n = N;
        if (d >= 1 - 1.0 / n) return 2 * Math.Pow(1 - d, n);
        if (d >= 0.5) return 2 * Special.Smirnov(n, d);
        if (d <= 1.0 / n) return 1 - Cdf0(d);
        var c = Durbin(n, d);
        var s = 1 - c;
        if (s < 1e-12) return 2 * Special.Smirnov(n, d);
        return s;
    }

    /// <inheritdoc />
    protected override double Pdf0(double d)
    {
        // The matrix method has no closed derivative: Richardson-extrapolated central differences
        // (fourth order) of whichever tail function is small at d, clipped to the support.
        var h = 1e-3 * Math.Min(d - Lower0, Upper0 - d);
        if (!(h > 0)) return 0.0;
        Func<double, double> f = d > 0.5 ? x => -Sf0(x) : Cdf0;
        double D(double step) => (f(d + step) - f(d - step)) / (2 * step);
        return Math.Max(0, (4 * D(h / 2) - D(h)) / 3);
    }

    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        // E[Dⁿ] = l₀ⁿ + ∫ n xⁿ⁻¹ Sf(x) dx over the support: one quadrature of the survival function
        // instead of differencing the CDF into a density first.
        var lo = Lower0;
        return Math.Pow(lo, n) + StatsDistMath.Quad(x => n * Math.Pow(x, n - 1) * Sf0(x), lo, 1.0, 1e-14, 1e-10);
    }

    /// <summary>Marsaglia–Tsang–Wang (2003) P(Dₙ &lt; d) by the Durbin matrix power with exponent tracking.</summary>
    private static double Durbin(int n, double d)
    {
        var k = (int)(n * d) + 1;
        var m = 2 * k - 1;
        var h = k - n * d;
        var H = new double[m * m];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < m; j++)
                H[i * m + j] = i - j + 1 < 0 ? 0 : 1;
        for (var i = 0; i < m; i++)
        {
            H[i * m] -= Math.Pow(h, i + 1);
            H[(m - 1) * m + i] -= Math.Pow(h, m - i);
        }
        H[(m - 1) * m] += 2 * h - 1 > 0 ? Math.Pow(2 * h - 1, m) : 0;
        for (var i = 0; i < m; i++)
            for (var j = 0; j < m; j++)
                if (i - j + 1 > 0)
                    for (var g = 1; g <= i - j + 1; g++) H[i * m + j] /= g;
        var (q, eq) = MatPow(H, m, n);
        var s = q[(k - 1) * m + k - 1];
        var es = eq;
        for (var i = 1; i <= n; i++)
        {
            s = s * i / n;
            if (s < 1e-140) { s *= 1e140; es -= 140; }
        }
        return s * Math.Pow(10.0, es);
    }

    private static (double[] M, int Exp) MatPow(double[] a, int m, int n)
    {
        if (n == 1) return ((double[])a.Clone(), 0);
        var (half, e) = MatPow(a, m, n / 2);
        var b = MatMul(half, half, m);
        var eb = 2 * e;
        if (n % 2 == 1) b = MatMul(a, b, m);
        if (b[(m / 2) * m + m / 2] > 1e140)
        {
            for (var i = 0; i < b.Length; i++) b[i] *= 1e-140;
            eb += 140;
        }
        return (b, eb);
    }

    private static double[] MatMul(double[] x, double[] y, int m)
    {
        var r = new double[m * m];
        for (var i = 0; i < m; i++)
            for (var l = 0; l < m; l++)
            {
                var xv = x[i * m + l];
                if (xv == 0) continue;
                for (var j = 0; j < m; j++) r[i * m + j] += xv * y[l * m + j];
            }
        return r;
    }
}

/// <summary>
/// The studentized range distribution (range of k standard normals over an independent
/// chi/√df), scipy's <c>studentized_range</c> — the reference distribution of Tukey's HSD test.
/// </summary>
/// <remarks>
/// As scipy, by numerical integration: P(q) = ∫₀^∞ f_s(s) · k∫ φ(z)[Φ(z) − Φ(z − qs)]^{k−1} dz ds, where
/// f_s is the density of √(χ²_df/df). Each CDF value costs a nested adaptive quadrature (~1 ms);
/// df = ∞ drops the outer integral. Quantiles invert the CDF by Brent.
/// </remarks>
public sealed class StudentizedRangeDistribution : ContinuousDistribution
{
    /// <summary>Studentized range for <paramref name="k"/> groups and <paramref name="df"/> degrees of freedom (may be +∞).</summary>
    public StudentizedRangeDistribution(double k, double df, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        if (!(k > 1) || double.IsInfinity(k)) throw new ArgumentException("k must exceed 1.", nameof(k));
        if (!(df > 0)) throw new ArgumentException("df must be positive.", nameof(df));
        K = k;
        Df = df;
    }

    /// <summary>Number of groups.</summary>
    public double K { get; }
    /// <summary>Degrees of freedom of the variance estimate.</summary>
    public double Df { get; }

    /// <inheritdoc />
    public override string Name => "studentized_range";
    /// <inheritdoc />
    public override string[] ShapeNames => ["k", "df"];
    /// <inheritdoc />
    public override double[] Shapes => [K, Df];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new StudentizedRangeDistribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;

    /// <summary>P(range of k normals ≤ w).</summary>
    private double RangeCdf(double w)
    {
        if (w <= 0) return 0;
        var k = K;
        double F(double z)
        {
            var d = Special.Ndtr(z) - Special.Ndtr(z - w);
            return d <= 0 ? 0 : Math.Exp(LogPhi(z) + (k - 1) * Math.Log(d));
        }
        return Math.Min(1, k * (Quad(F, double.NegativeInfinity, w / 2, 1e-16, 1e-11) + Quad(F, w / 2, double.PositiveInfinity, 1e-16, 1e-11)));
    }

    /// <summary>
    /// P(range of k normals &gt; w), integrated directly so the upper tail is not 1 − P:
    /// since k∫φ(z)Φ(z)^{k−1}dz = 1, P(R &gt; w) = k∫φ(z)·Φ(z)^{k−1}·[1 − (1 − Φ(z−w)/Φ(z))^{k−1}] dz, and the
    /// bracket is −expm1((k−1)·log1p(−Φ(z−w)/Φ(z))), positive term by term.
    /// </summary>
    private double RangeSf(double w)
    {
        if (w <= 0) return 1;
        var k = K;
        double F(double z)
        {
            var lA = Special.LogNdtr(z);
            var ratio = Math.Exp(Special.LogNdtr(z - w) - lA);
            var bracket = -Special.Expm1((k - 1) * Special.Log1p(-ratio));
            return bracket <= 0 ? 0 : Math.Exp(LogPhi(z) + (k - 1) * lA) * bracket;
        }
        return Math.Min(1, k * (Quad(F, double.NegativeInfinity, w / 2, 1e-300, 1e-11) + Quad(F, w / 2, double.PositiveInfinity, 1e-300, 1e-11)));
    }

    private double ChiSqrtLogPdf(double s)
    {
        var v = Df;
        return v / 2 * Math.Log(v) - Special.GammaLn(v / 2) - (v / 2 - 1) * Ln2 + (v - 1) * Math.Log(s) - v * s * s / 2;
    }

    private double Outer(Func<double, double> inner)
    {
        if (double.IsPositiveInfinity(Df) || Df > 1e8) return inner(1.0);
        double G(double s) => s <= 0 ? 0 : Math.Exp(ChiSqrtLogPdf(s)) * inner(s);
        // The chi density peaks at ~1 with width ~1/√(2df); split there so quad sees the peak.
        var w = 1 / Math.Sqrt(2 * Df);
        return Quad(G, 0, Math.Max(1e-3, 1 - 3 * w), 1e-16, 1e-10) + Quad(G, Math.Max(1e-3, 1 - 3 * w), 1 + 6 * w, 1e-16, 1e-10)
               + Quad(G, 1 + 6 * w, double.PositiveInfinity, 1e-16, 1e-10);
    }

    /// <inheritdoc />
    protected override double Cdf0(double q) => Math.Clamp(Outer(s => RangeCdf(q * s)), 0, 1);
    /// <inheritdoc />
    protected override double Sf0(double q) => Math.Clamp(Outer(s => RangeSf(q * s)), 0, 1);
    /// <inheritdoc />
    protected override double Pdf0(double q)
    {
        var k = K;
        double RangePdf(double w)
        {
            if (w <= 0) return 0;
            double F(double z)
            {
                var d = Special.Ndtr(z) - Special.Ndtr(z - w);
                return d <= 0 ? 0 : Math.Exp(LogPhi(z) + LogPhi(z - w) + (k - 2) * Math.Log(d));
            }
            return k * (k - 1) * (Quad(F, double.NegativeInfinity, w / 2, 1e-300, 1e-11) + Quad(F, w / 2, double.PositiveInfinity, 1e-300, 1e-11));
        }
        return Math.Max(0, Outer(s => s * RangePdf(q * s)));
    }
}
