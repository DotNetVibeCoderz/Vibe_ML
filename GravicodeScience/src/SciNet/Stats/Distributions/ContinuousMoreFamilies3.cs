using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.StatsDistMath;

namespace Gravicode.Science.SciNet;

/// <summary>
/// The double Pareto log-normal distribution of Reed and Jorgensen (2004), scipy's
/// <c>dpareto_lognorm</c>: exp(N(u, s²) + E₁/a − E₂/b) with independent standard exponentials.
/// </summary>
/// <remarks>
/// Both tails are written as Φ-tail ∓ φ·(Mills-ratio terms), each bracket provably non-negative,
/// and summed in log space; nothing is formed as 1 − CDF.
/// </remarks>
public sealed class DParetoLogNormDistribution : ContinuousDistribution
{
    private readonly double _logA, _logB; // log(a/(a+b)), log(b/(a+b))

    /// <summary>Double Pareto log-normal with log-location <paramref name="u"/>, log-scale <paramref name="s"/> &gt; 0 and tail indices <paramref name="a"/>, <paramref name="b"/> &gt; 0.</summary>
    public DParetoLogNormDistribution(double u, double s, double a, double b, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        U = StatsDistributions.Finite(u, nameof(u));
        S = StatsDistributions.Positive(s, nameof(s));
        A = StatsDistributions.Positive(a, nameof(a));
        B = StatsDistributions.Positive(b, nameof(b));
        _logA = Math.Log(a / (a + b));
        _logB = Math.Log(b / (a + b));
    }

    /// <summary>Location of the log-normal core.</summary>
    public double U { get; }
    /// <summary>Scale of the log-normal core.</summary>
    public double S { get; }
    /// <summary>Upper (Pareto) tail index.</summary>
    public double A { get; }
    /// <summary>Lower tail index.</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "dpareto_lognorm";
    /// <inheritdoc />
    public override string[] ShapeNames => ["u", "s", "a", "b"];
    /// <inheritdoc />
    public override double[] Shapes => [U, S, A, B];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new DParetoLogNormDistribution(shapes[0], shapes[1], shapes[2], shapes[3], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <summary>log of the Mills ratio Φc(t)/φ(t).</summary>
    private static double LogR(double t) => Special.LogNdtr(-t) - LogPhi(t);
    /// <inheritdoc />
    protected override double Pdf0(double y) => y <= 0 ? 0.0 : Math.Exp(LogPdf0(y));
    /// <inheritdoc />
    protected override double LogPdf0(double y)
    {
        if (y <= 0 || double.IsPositiveInfinity(y)) return double.NegativeInfinity;
        var ly = Math.Log(y);
        var z = (ly - U) / S;
        return Math.Log(A) + Math.Log(B) - Math.Log(A + B) - ly + LogPhi(z) + LogAddExp(LogR(A * S - z), LogR(B * S + z));
    }
    private static double SafeSub(double a, double b) => b >= a ? double.NegativeInfinity : LogSubExp(a, b);
    /// <inheritdoc />
    protected override double LogCdf0(double y)
    {
        var z = (Math.Log(y) - U) / S;
        var lp = LogPhi(z);
        return LogAddExp(SafeSub(Special.LogNdtr(z), lp + _logB + LogR(A * S - z)), lp + _logA + LogR(B * S + z));
    }
    /// <inheritdoc />
    protected override double LogSf0(double y)
    {
        var z = (Math.Log(y) - U) / S;
        var lp = LogPhi(z);
        return LogAddExp(SafeSub(Special.LogNdtr(-z), lp + _logA + LogR(B * S + z)), lp + _logB + LogR(A * S - z));
    }
    /// <inheritdoc />
    protected override double Cdf0(double y) => Math.Exp(LogCdf0(y));
    /// <inheritdoc />
    protected override double Sf0(double y) => Math.Exp(LogSf0(y));
    /// <inheritdoc />
    protected override double Munp0(int n)
        => A > n ? A * B / ((A - n) * (B + n)) * Math.Exp(n * U + n * n * S * S / 2) : double.NaN;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
        => Math.Exp(U + S * StatsRandom.Normal(rng) + StatsRandom.Exponential(rng) / A - StatsRandom.Exponential(rng) / B);
}

/// <summary>The Gauss hypergeometric distribution, scipy's <c>gausshyper</c>, on [0, 1].</summary>
/// <remarks>
/// Density ∝ x^(a−1)(1 − x)^(b−1)(1 + zx)^(−c). The CDF has no closed form and is integrated, as in
/// scipy; variates are beta proposals accepted with probability (1 + zx)^(−c)/max.
/// </remarks>
public sealed class GaussHyperDistribution : ContinuousDistribution
{
    private readonly double _logNorm;

    /// <summary>Gauss hypergeometric with <paramref name="a"/>, <paramref name="b"/> &gt; 0, any <paramref name="c"/> and <paramref name="z"/> &gt; −1.</summary>
    public GaussHyperDistribution(double a, double b, double c, double z, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        A = StatsDistributions.Positive(a, nameof(a));
        B = StatsDistributions.Positive(b, nameof(b));
        C = StatsDistributions.Finite(c, nameof(c));
        if (!(z > -1) || !double.IsFinite(z)) throw new ArgumentException("z must exceed -1.", nameof(z));
        Z = z;
        _logNorm = Special.BetaLn(a, b) + Math.Log(Special.Hyp2f1(c, a, a + b, -z));
    }

    /// <summary>The shape a.</summary>
    public double A { get; }
    /// <summary>The shape b.</summary>
    public double B { get; }
    /// <summary>The shape c.</summary>
    public double C { get; }
    /// <summary>The shape z.</summary>
    public double Z { get; }

    /// <inheritdoc />
    public override string Name => "gausshyper";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a", "b", "c", "z"];
    /// <inheritdoc />
    public override double[] Shapes => [A, B, C, Z];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new GaussHyperDistribution(shapes[0], shapes[1], shapes[2], shapes[3], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Upper0 => 1.0;
    /// <inheritdoc />
    protected override double Pdf0(double x) => Math.Exp(LogPdf0(x));
    /// <inheritdoc />
    protected override double LogPdf0(double x)
        => Special.Xlogy(A - 1, x) + Special.Xlog1py(B - 1, -x) - C * Special.Log1p(Z * x) - _logNorm;
    /// <inheritdoc />
    protected override double Cdf0(double x) => Math.Clamp(StatsDistMoreMath.QuadRel(Pdf0, 0.0, x, null, 1e-12), 0.0, 1.0);
    /// <inheritdoc />
    protected override double Sf0(double x) => Math.Clamp(StatsDistMoreMath.QuadRel(Pdf0, x, 1.0, null, 1e-12), 0.0, 1.0);
    /// <inheritdoc />
    protected override double Munp0(int n)
        => Math.Exp(Special.BetaLn(n + A, B) - Special.BetaLn(A, B)) * Special.Hyp2f1(C, A + n, A + B + n, -Z) / Special.Hyp2f1(C, A, A + B, -Z);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        // (1 + zx)^(−c) is monotone on [0, 1]; its maximum is at one end.
        var logMax = Math.Max(0.0, -C * Special.Log1p(Z));
        while (true)
        {
            var x = StatsRandom.Beta(rng, A, B);
            if (Math.Log(StatsRandom.U(rng)) <= -C * Special.Log1p(Z * x) - logMax) return x;
        }
    }
}

/// <summary>The generalised inverse Gaussian distribution, scipy's <c>geninvgauss</c>: density ∝ x^(p−1) exp(−b(x + 1/x)/2).</summary>
/// <remarks>
/// The normaliser uses the exponentially scaled Bessel function K̃ₚ(b) = eᵇKₚ(b) so large b does not
/// underflow. The CDF is integrated (as in scipy). Variates follow scipy's port of Hörmann and
/// Leydold (2014): ratio-of-uniforms with or without mode shift, or their rejection method for small
/// b and p &lt; 1.
/// </remarks>
public sealed class GenInvGaussDistribution : ContinuousDistribution
{
    private readonly double _logNorm; // log(2 K_p(b))

    /// <summary>Generalised inverse Gaussian with shapes <paramref name="p"/> (any real) and <paramref name="b"/> &gt; 0.</summary>
    public GenInvGaussDistribution(double p, double b, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        P = StatsDistributions.Finite(p, nameof(p));
        B = StatsDistributions.Positive(b, nameof(b));
        _logNorm = Ln2 + Math.Log(Special.Kve(p, b)) - b;
    }

    /// <summary>The shape p.</summary>
    public double P { get; }
    /// <summary>The shape b.</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "geninvgauss";
    /// <inheritdoc />
    public override string[] ShapeNames => ["p", "b"];
    /// <inheritdoc />
    public override double[] Shapes => [P, B];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new GenInvGaussDistribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double x) => x <= 0 ? 0.0 : Math.Exp(LogPdf0(x));
    /// <inheritdoc />
    protected override double LogPdf0(double x)
        => x <= 0 ? double.NegativeInfinity : (P - 1) * Math.Log(x) - B * (x + 1 / x) / 2 - _logNorm;
    private double Mode => Mode0(P, B);
    private static double Mode0(double p, double b)
        => p < 1 ? b / (Math.Sqrt((p - 1) * (p - 1) + b * b) + 1 - p) : (Math.Sqrt((1 - p) * (1 - p) + b * b) - (1 - p)) / b;
    /// <inheritdoc />
    protected override double Cdf0(double x)
    {
        var m = Mode;
        if (x <= m) return Math.Clamp(StatsDistMoreMath.QuadRel(Pdf0, 0.0, x, null, 1e-12), 0.0, 1.0);
        return Math.Clamp(1 - Sf0(x), 0.0, 1.0);
    }
    /// <inheritdoc />
    protected override double Sf0(double x)
    {
        var m = Mode;
        if (x >= m) return Math.Clamp(StatsDistMoreMath.QuadRel(Pdf0, x, double.PositiveInfinity, null, 1e-12), 0.0, 1.0);
        return Math.Clamp(1 - Cdf0(x), 0.0, 1.0);
    }
    /// <inheritdoc />
    protected override double Munp0(int n) => Special.Kve(P + n, B) / Special.Kve(P, B);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Sample(rng, P, B);

    private static double LogQuasi(double x, double p, double b) => x > 0 ? (p - 1) * Math.Log(x) - b * (x + 1 / x) / 2 : double.NegativeInfinity;

    /// <summary>One standard GIG(p, b) variate (scipy's <c>_rvs_scalar</c>).</summary>
    internal static double Sample(GraviRandom rng, double p, double b)
    {
        var invert = false;
        if (p < 0) { p = -p; invert = true; }
        var m = Mode0(p, b);
        double x;
        if (p >= 1 || b > 1 || b >= Math.Min(0.5, 2 * Math.Sqrt(1 - p) / 3))
        {
            double umax, vmin, vmax, c, lm;
            if (p >= 1 || b > 1)
            {
                // Ratio-of-uniforms with mode shift; the bounding rectangle's v-extent comes from the
                // two real roots of a cubic (trigonometric form).
                var a2 = -2 * (p + 1) / b - m;
                var a1 = 2 * m * (p - 1) / b - 1;
                var p1 = a1 - a2 * a2 / 3;
                var q1 = 2 * a2 * a2 * a2 / 27 - a2 * a1 / 3 + m;
                var phi = Math.Acos(-q1 * Math.Sqrt(-27 / (p1 * p1 * p1)) / 2);
                var s1 = -Math.Sqrt(-4 * p1 / 3);
                var root1 = s1 * Math.Cos(phi / 3 + Math.PI / 3) - a2 / 3;
                var root2 = -s1 * Math.Cos(phi / 3) - a2 / 3;
                lm = LogQuasi(m, p, b);
                vmin = (root1 - m) * Math.Exp(0.5 * (LogQuasi(root1, p, b) - lm));
                vmax = (root2 - m) * Math.Exp(0.5 * (LogQuasi(root2, p, b) - lm));
                umax = 1;
                c = m;
            }
            else
            {
                lm = 0;
                umax = Math.Exp(0.5 * LogQuasi(m, p, b));
                var xplus = (1 + p + Math.Sqrt((1 + p) * (1 + p) + b * b)) / b;
                vmin = 0;
                vmax = xplus * Math.Exp(0.5 * LogQuasi(xplus, p, b));
                c = 0;
            }
            while (true)
            {
                var u = umax * StatsRandom.U(rng);
                var v = vmin + (vmax - vmin) * StatsRandom.U(rng);
                x = v / u + c;
                if (2 * Math.Log(u) <= LogQuasi(x, p, b) - lm) break;
            }
        }
        else
        {
            // Hörmann–Leydold's rejection from a three-part hat, for p < 1 and small b.
            var x0 = b / (1 - p);
            var xs = Math.Max(x0, 2 / b);
            var k1 = Math.Exp(LogQuasi(m, p, b));
            var a1 = k1 * x0;
            double k2 = 0, a2 = 0;
            if (x0 < 2 / b)
            {
                k2 = Math.Exp(-b);
                a2 = p > 0 ? k2 * (Math.Pow(2 / b, p) - Math.Pow(x0, p)) / p : k2 * Math.Log(2 / (b * b));
            }
            var k3 = Math.Pow(xs, p - 1);
            var a3 = 2 * k3 * Math.Exp(-xs * b / 2) / b;
            var total = a1 + a2 + a3;
            while (true)
            {
                var u = StatsRandom.U(rng);
                var v = total * StatsRandom.U(rng);
                double h;
                if (v <= a1) { x = x0 * v / a1; h = k1; }
                else if (v <= a1 + a2)
                {
                    x = p > 0 ? Math.Pow(Math.Pow(x0, p) + (v - a1) * p / k2, 1 / p) : b * Math.Exp((v - a1) * Math.Exp(b));
                    h = k2 * Math.Pow(x, p - 1);
                }
                else
                {
                    var zz = Math.Exp(-xs * b / 2) - b * (v - a1 - a2) / (2 * k3);
                    x = -2 / b * Math.Log(zz);
                    h = k3 * Math.Exp(-x * b / 2);
                }
                if (Math.Log(u * h) <= LogQuasi(x, p, b)) break;
            }
        }
        return invert ? 1 / x : x;
    }
}

/// <summary>The generalised hyperbolic distribution, scipy's <c>genhyperbolic</c> (δ = 1, μ = 0 standard form).</summary>
/// <remarks>
/// Densities use K̃ (exponentially scaled Bessel K) throughout. The CDF is integrated, split at the
/// mean, as in scipy. Variates are a normal variance–mean mixture over a GIG(p, γ)/γ mixing variable.
/// </remarks>
public sealed class GenHyperbolicDistribution : ContinuousDistribution
{
    private readonly double _gamma, _logNorm, _mean;

    /// <summary>Generalised hyperbolic with tail <paramref name="p"/>, shape <paramref name="a"/> and skewness <paramref name="b"/> (|b| &lt; a, or |b| ≤ a when p &lt; 0).</summary>
    public GenHyperbolicDistribution(double p, double a, double b, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        P = StatsDistributions.Finite(p, nameof(p));
        A = StatsDistributions.Positive(a, nameof(a));
        B = StatsDistributions.Finite(b, nameof(b));
        if (!(Math.Abs(b) < a || (p < 0 && Math.Abs(b) <= a))) throw new ArgumentException("Need |b| < a (or |b| <= a when p < 0).", nameof(b));
        _gamma = Math.Sqrt((a + b) * (a - b));
        // log(γᵖ/Kₚ(γ)); at γ = 0 (p < 0) its limit 2^(p+1)/Γ(−p).
        var lk = _gamma > 0 ? P * Math.Log(_gamma) - (Math.Log(Special.Kve(P, _gamma)) - _gamma) : (P + 1) * Ln2 - Special.GammaLn(-P);
        _logNorm = lk - LnSqrt2Pi - (P - 0.5) * Math.Log(A);
        _mean = _gamma > 0 ? B / _gamma * Special.Kve(P + 1, _gamma) / Special.Kve(P, _gamma) : double.NaN;
    }

    /// <summary>The tail parameter p (λ).</summary>
    public double P { get; }
    /// <summary>The shape a (α).</summary>
    public double A { get; }
    /// <summary>The skewness b (β).</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "genhyperbolic";
    /// <inheritdoc />
    public override string[] ShapeNames => ["p", "a", "b"];
    /// <inheritdoc />
    public override double[] Shapes => [P, A, B];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new GenHyperbolicDistribution(shapes[0], shapes[1], shapes[2], loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double x) => Math.Exp(LogPdf0(x));
    /// <inheritdoc />
    protected override double LogPdf0(double x)
    {
        var s = Math.Sqrt(1 + x * x);
        return _logNorm + (P - 0.5) / 2 * Math.Log(1 + x * x) + Math.Log(Special.Kve(P - 0.5, A * s)) - A * s + B * x;
    }
    private double Split => double.IsFinite(_mean) ? _mean : 0.0;
    /// <inheritdoc />
    protected override double Cdf0(double x)
        => x <= Split ? Math.Clamp(StatsDistMoreMath.QuadRel(Pdf0, double.NegativeInfinity, x, null, 1e-12), 0.0, 1.0) : Math.Clamp(1 - Sf0(x), 0.0, 1.0);
    /// <inheritdoc />
    protected override double Sf0(double x)
        => x >= Split ? Math.Clamp(StatsDistMoreMath.QuadRel(Pdf0, x, double.PositiveInfinity, null, 1e-12), 0.0, 1.0) : Math.Clamp(1 - Cdf0(x), 0.0, 1.0);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var t1 = _gamma;
        var t2 = 1 / t1;
        var b0 = Special.Kve(P, t1);
        double b1 = Special.Kve(P + 1, t1), b2 = Special.Kve(P + 2, t1), b3 = Special.Kve(P + 3, t1), b4 = Special.Kve(P + 4, t1);
        double r1 = b1 / b0, r2 = b2 / b0, r3 = b3 / b0, r4 = b4 / b0;
        var b = B;
        var m = b * t2 * r1;
        var v = t2 * r1 + b * b * t2 * t2 * (r2 - r1 * r1);
        var m3e = b * b * b * t2 * t2 * t2 * (r3 - 3 * b2 * b1 / (b0 * b0) + 2 * r1 * r1 * r1) + 3 * b * t2 * t2 * (r2 - r1 * r1);
        var m4e = Math.Pow(b * t2, 4) * (r4 - 4 * b3 * b1 / (b0 * b0) + 6 * b2 * b1 * b1 / (b0 * b0 * b0) - 3 * Math.Pow(r1, 4))
                  + b * b * t2 * t2 * t2 * (6 * r3 - 12 * b2 * b1 / (b0 * b0) + 6 * r1 * r1 * r1) + 3 * t2 * t2 * r2;
        return new(m, v, m3e / Math.Pow(v, 1.5), m4e / (v * v) - 3);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        var g = GenInvGaussDistribution.Sample(rng, P, _gamma) / _gamma;
        return B * g + Math.Sqrt(g) * StatsRandom.Normal(rng);
    }
}

/// <summary>The normal-inverse Gaussian distribution, scipy's <c>norminvgauss</c>.</summary>
/// <remarks>The density uses K̃₁ so a large a does not underflow; the CDF is integrated from the nearer tail, split at the mean.</remarks>
public sealed class NormInvGaussDistribution : ContinuousDistribution
{
    private readonly double _gamma;

    /// <summary>Normal-inverse Gaussian with tail heaviness <paramref name="a"/> &gt; 0 and asymmetry |<paramref name="b"/>| &lt; a.</summary>
    public NormInvGaussDistribution(double a, double b, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        A = StatsDistributions.Positive(a, nameof(a));
        B = StatsDistributions.Finite(b, nameof(b));
        if (!(Math.Abs(b) < a)) throw new ArgumentException("Need |b| < a.", nameof(b));
        _gamma = Math.Sqrt((a + b) * (a - b));
    }

    /// <summary>The tail heaviness a (α).</summary>
    public double A { get; }
    /// <summary>The asymmetry b (β).</summary>
    public double B { get; }

    /// <inheritdoc />
    public override string Name => "norminvgauss";
    /// <inheritdoc />
    public override string[] ShapeNames => ["a", "b"];
    /// <inheritdoc />
    public override double[] Shapes => [A, B];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new NormInvGaussDistribution(shapes[0], shapes[1], loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double x) => Math.Exp(LogPdf0(x));
    /// <inheritdoc />
    protected override double LogPdf0(double x)
    {
        var sq = Math.Sqrt(1 + x * x);
        return Math.Log(A / Math.PI) + Math.Log(Special.K1e(A * sq)) + B * x - A * sq + _gamma - Math.Log(sq);
    }
    private double MeanValue => B / _gamma;
    /// <inheritdoc />
    protected override double Cdf0(double x)
        => x <= MeanValue ? Math.Clamp(StatsDistMoreMath.QuadRel(Pdf0, double.NegativeInfinity, x, null, 1e-12), 0.0, 1.0) : Math.Clamp(1 - Sf0(x), 0.0, 1.0);
    /// <inheritdoc />
    protected override double Sf0(double x)
        => x >= MeanValue ? Math.Clamp(StatsDistMoreMath.QuadRel(Pdf0, x, double.PositiveInfinity, null, 1e-12), 0.0, 1.0) : Math.Clamp(1 - Cdf0(x), 0.0, 1.0);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var g = _gamma;
        return new(B / g, A * A / (g * g * g), 3 * B / (A * Math.Sqrt(g)), 3 * (1 + 4 * B * B / (A * A)) / g);
    }
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        // Inverse Gaussian with mean 1/γ and shape 1 (Michael, Schucany and Haas 1976).
        var mu = 1 / _gamma;
        var y = StatsRandom.Normal(rng);
        y *= y;
        var w = mu + 0.5 * mu * mu * y - 0.5 * mu * Math.Sqrt(4 * mu * y + mu * mu * y * y);
        if (StatsRandom.U(rng) > mu / (mu + w)) w = mu * mu / w;
        return B * w + Math.Sqrt(w) * StatsRandom.Normal(rng);
    }
}

/// <summary>The Irwin–Hall distribution, scipy's <c>irwinhall</c>: the sum of n independent uniforms on [0, 1].</summary>
/// <remarks>
/// The density is the cardinal B-spline Mₙ, evaluated by the de Boor–Cox recursion (convex
/// combinations only — the textbook alternating sum Σ(−1)ᵏC(n,k)(x − k)ⁿ⁻¹ loses every digit by
/// n ≈ 30). The CDF is Σⱼ Mₙ₊₁(x − j), which telescopes to ∫Mₙ and is again a sum of positive terms;
/// the upper half uses the symmetry about n/2.
/// </remarks>
public sealed class IrwinHallDistribution : ContinuousDistribution
{
    /// <summary>Irwin–Hall for the sum of <paramref name="n"/> ≥ 1 uniforms.</summary>
    public IrwinHallDistribution(int n, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        if (n < 1) throw new ArgumentException("n must be a positive integer.", nameof(n));
        N = n;
    }

    /// <summary>The number of summed uniforms.</summary>
    public int N { get; }

    /// <inheritdoc />
    public override string Name => "irwinhall";
    /// <inheritdoc />
    public override string[] ShapeNames => ["n"];
    /// <inheritdoc />
    public override double[] Shapes => [N];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale)
    {
        if (shapes[0] != Math.Floor(shapes[0])) throw new ArgumentException("n must be an integer.");
        return new IrwinHallDistribution((int)shapes[0], loc, scale);
    }
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Upper0 => N;

    /// <summary>(Mₙ(t), Σⱼ Mₙ₊₁(t − j)) for 0 ≤ t ≤ n.</summary>
    private (double Pdf, double Cdf) Spline(double t)
    {
        var n = N;
        var len = (int)Math.Floor(t) + n + 3;
        var v = new double[len];
        for (var j = 0; j < len; j++) { var s = t - j; v[j] = s >= 0 && s < 1 ? 1.0 : 0.0; }
        var pdf = n == 1 ? v[0] : 0.0;
        for (var k = 2; k <= n + 1; k++)
        {
            for (var j = 0; j + k < len; j++)
            {
                var s = t - j;
                v[j] = (s * v[j] + (k - s) * v[j + 1]) / (k - 1);
            }
            if (k == n) pdf = v[0];
        }
        var cdf = 0.0;
        for (var j = 0; j <= (int)Math.Floor(t) && j + n + 1 < len; j++) cdf += v[j];
        return (pdf, cdf);
    }
    /// <inheritdoc />
    protected override double Pdf0(double z)
    {
        if (z < 0 || z > N) return 0.0;
        return Spline(z <= 0.5 * N ? z : N - z).Pdf;
    }
    /// <inheritdoc />
    protected override double Cdf0(double z) => z <= 0.5 * N ? Spline(z).Cdf : 1 - Spline(N - z).Cdf;
    /// <inheritdoc />
    protected override double Sf0(double z) => Cdf0(N - z);
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(N / 2.0, N / 12.0, 0.0, -6.0 / (5.0 * N));
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        var s = 0.0;
        for (var i = 0; i < N; i++) s += StatsRandom.U(rng);
        return s;
    }
}

/// <summary>The one-sided Kolmogorov–Smirnov statistic D⁺ₙ, scipy's <c>ksone</c>.</summary>
/// <remarks>
/// By Abel's identity the Birnbaum–Tingey terms over j = 0…n sum to one: the terms with
/// 1 − d − j/n ≥ 0 give the survival function, the remaining ones (alternating in sign) the CDF.
/// Each probability, and the density from the differentiated terms, is taken from whichever set
/// has the smaller absolute sum, i.e. the one with the smaller rounding error.
/// </remarks>
public sealed class KSOneDistribution : ContinuousDistribution
{
    /// <summary>D⁺ₙ for sample size <paramref name="n"/> ≥ 1.</summary>
    public KSOneDistribution(int n, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        if (n < 1) throw new ArgumentException("n must be a positive integer.", nameof(n));
        N = n;
    }

    /// <summary>The sample size.</summary>
    public int N { get; }

    /// <inheritdoc />
    public override string Name => "ksone";
    /// <inheritdoc />
    public override string[] ShapeNames => ["n"];
    /// <inheritdoc />
    public override double[] Shapes => [N];
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale)
    {
        if (shapes[0] != Math.Floor(shapes[0])) throw new ArgumentException("n must be an integer.");
        return new KSOneDistribution((int)shapes[0], loc, scale);
    }
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Upper0 => 1.0;

    /// <summary>Survival function, CDF and density at d.</summary>
    private (double Sf, double Cdf, double Pdf) Terms(double d)
    {
        var n = N;
        double sLow = 0, sUp = 0, aUp = 0, dLow = 0, adLow = 0, dUp = 0, adUp = 0;
        var lgn = Special.GammaLn(n + 1.0);
        for (var j = 0; j <= n; j++)
        {
            var rest = (double)(n - j) / n - d;
            var lc = lgn - Special.GammaLn(j + 1.0) - Special.GammaLn(n - j + 1.0);
            var plus = d + (double)j / n;
            double t, dt;
            if (rest == 0)
            {
                t = 0;
                dt = n - j == 1 ? -Math.Exp(lc + Math.Log(d) + (j - 1) * Math.Log(plus)) : 0.0;
            }
            else
            {
                var sign = rest < 0 && (n - j) % 2 == 1 ? -1.0 : 1.0;
                t = sign * Math.Exp(lc + Math.Log(d) + (j - 1) * Math.Log(plus) + (n - j) * Math.Log(Math.Abs(rest)));
                dt = t * (1 / d + (j - 1) / plus - (n - j) / rest);
            }
            // A term whose base is exactly zero belongs to the upper set: the density is right-continuous at d = 1/n.
            if (rest > 0) { sLow += t; dLow += dt; adLow += Math.Abs(dt); }
            else { sUp += t; aUp += Math.Abs(t); dUp += dt; adUp += Math.Abs(dt); }
        }
        var sf = Math.Clamp(sLow, 0.0, 1.0);
        var cdf = aUp < 1 ? Math.Clamp(sUp, 0.0, 1.0) : 1 - sf;
        var pdf = adUp < adLow ? dUp : -dLow;
        return (sf, cdf, Math.Max(0.0, pdf));
    }
    /// <inheritdoc />
    protected override double Pdf0(double z) => Terms(Math.Clamp(z, 1e-300, 1.0)).Pdf;
    /// <inheritdoc />
    protected override double Cdf0(double z) => Terms(z).Cdf;
    /// <inheritdoc />
    protected override double Sf0(double z) => Terms(z).Sf;
    /// <inheritdoc />
    protected override double Isf0(double q) => Special.SmirnovInverse(N, q);
    /// <summary>The density has kinks at multiples of 1/n; moments integrate between them.</summary>
    protected override double Munp0(int k)
    {
        var s = 0.0;
        for (var j = 0; j < N; j++) s += StatsDistMoreMath.QuadRel(z => Math.Pow(z, k) * Pdf0(z), (double)j / N, (j + 1.0) / N, null, 1e-12);
        return s;
    }
    /// <inheritdoc />
    protected override double Entropy0()
    {
        var s = 0.0;
        for (var j = 0; j < N; j++)
            s += StatsDistMoreMath.QuadRel(z => { var p = Pdf0(z); return p > 0 ? -p * Math.Log(p) : 0.0; }, (double)j / N, (j + 1.0) / N, null, 1e-12);
        return s;
    }
}
