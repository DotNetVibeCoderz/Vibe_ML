namespace Gravicode.Science.SciNet;

public static partial class Stats
{
    // ------------------------------------------------------------------ continuous (scipy 1.17.1 families added in 1.3)

    /// <summary>Alpha, scipy's <c>alpha</c>.</summary>
    public static AlphaDistribution Alpha(double a, double loc = 0.0, double scale = 1.0) => new(a, loc, scale);
    /// <summary>ARGUS, scipy's <c>argus</c>.</summary>
    public static ArgusDistribution Argus(double chi, double loc = 0.0, double scale = 1.0) => new(chi, loc, scale);
    /// <summary>Beta prime, scipy's <c>betaprime</c>.</summary>
    public static BetaPrimeDistribution BetaPrime(double a, double b, double loc = 0.0, double scale = 1.0) => new(a, b, loc, scale);
    /// <summary>Burr type III, scipy's <c>burr</c>.</summary>
    public static BurrDistribution Burr(double c, double d, double loc = 0.0, double scale = 1.0) => new(c, d, loc, scale);
    /// <summary>Crystal Ball, scipy's <c>crystalball</c>.</summary>
    public static CrystalBallDistribution CrystalBall(double beta, double m, double loc = 0.0, double scale = 1.0) => new(beta, m, loc, scale);
    /// <summary>Double gamma, scipy's <c>dgamma</c>.</summary>
    public static DGammaDistribution DGamma(double a, double loc = 0.0, double scale = 1.0) => new(a, loc, scale);
    /// <summary>Double Pareto log-normal, scipy's <c>dpareto_lognorm</c>.</summary>
    public static DParetoLogNormDistribution DParetoLogNorm(double u, double s, double a, double b, double loc = 0.0, double scale = 1.0) => new(u, s, a, b, loc, scale);
    /// <summary>Exponential power, scipy's <c>exponpow</c>.</summary>
    public static ExponPowDistribution ExponPow(double b, double loc = 0.0, double scale = 1.0) => new(b, loc, scale);
    /// <summary>Exponentiated Weibull, scipy's <c>exponweib</c>.</summary>
    public static ExponWeibDistribution ExponWeib(double a, double c, double loc = 0.0, double scale = 1.0) => new(a, c, loc, scale);
    /// <summary>Fatigue-life (Birnbaum–Saunders), scipy's <c>fatiguelife</c>.</summary>
    public static FatigueLifeDistribution FatigueLife(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Fisk (log-logistic), scipy's <c>fisk</c>.</summary>
    public static FiskDistribution Fisk(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Folded Cauchy, scipy's <c>foldcauchy</c>.</summary>
    public static FoldCauchyDistribution FoldCauchy(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Gauss hypergeometric, scipy's <c>gausshyper</c>.</summary>
    public static GaussHyperDistribution GaussHyper(double a, double b, double c, double z, double loc = 0.0, double scale = 1.0) => new(a, b, c, z, loc, scale);
    /// <summary>Generalised exponential, scipy's <c>genexpon</c>.</summary>
    public static GenExponDistribution GenExpon(double a, double b, double c, double loc = 0.0, double scale = 1.0) => new(a, b, c, loc, scale);
    /// <summary>Generalised half-logistic, scipy's <c>genhalflogistic</c>.</summary>
    public static GenHalfLogisticDistribution GenHalfLogistic(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Generalised hyperbolic, scipy's <c>genhyperbolic</c>.</summary>
    public static GenHyperbolicDistribution GenHyperbolic(double p, double a, double b, double loc = 0.0, double scale = 1.0) => new(p, a, b, loc, scale);
    /// <summary>Generalised inverse Gaussian, scipy's <c>geninvgauss</c>.</summary>
    public static GenInvGaussDistribution GenInvGauss(double p, double b, double loc = 0.0, double scale = 1.0) => new(p, b, loc, scale);
    /// <summary>Generalised normal, scipy's <c>gennorm</c>.</summary>
    public static GenNormDistribution GenNorm(double beta, double loc = 0.0, double scale = 1.0) => new(beta, loc, scale);
    /// <summary>Gibrat, scipy's <c>gibrat</c>.</summary>
    public static GibratDistribution Gibrat(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Gompertz, scipy's <c>gompertz</c>.</summary>
    public static GompertzDistribution Gompertz(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Half generalised normal, scipy's <c>halfgennorm</c>.</summary>
    public static HalfGenNormDistribution HalfGenNorm(double beta, double loc = 0.0, double scale = 1.0) => new(beta, loc, scale);
    /// <summary>Half-logistic, scipy's <c>halflogistic</c>.</summary>
    public static HalfLogisticDistribution HalfLogistic(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Hyperbolic secant, scipy's <c>hypsecant</c>.</summary>
    public static HypSecantDistribution HypSecant(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Inverted Weibull, scipy's <c>invweibull</c>.</summary>
    public static InvWeibullDistribution InvWeibull(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Irwin–Hall, scipy's <c>irwinhall</c>.</summary>
    public static IrwinHallDistribution IrwinHall(int n, double loc = 0.0, double scale = 1.0) => new(n, loc, scale);
    /// <summary>Jones–Faddy skew-t, scipy's <c>jf_skew_t</c>.</summary>
    public static JFSkewTDistribution JFSkewT(double a, double b, double loc = 0.0, double scale = 1.0) => new(a, b, loc, scale);
    /// <summary>Kappa 3, scipy's <c>kappa3</c>.</summary>
    public static Kappa3Distribution Kappa3(double a, double loc = 0.0, double scale = 1.0) => new(a, loc, scale);
    /// <summary>Kappa 4, scipy's <c>kappa4</c>.</summary>
    public static Kappa4Distribution Kappa4(double h, double k, double loc = 0.0, double scale = 1.0) => new(h, k, loc, scale);
    /// <summary>One-sided Kolmogorov–Smirnov statistic, scipy's <c>ksone</c>.</summary>
    public static KSOneDistribution KSOne(int n, double loc = 0.0, double scale = 1.0) => new(n, loc, scale);
    /// <summary>Landau, scipy's <c>landau</c>.</summary>
    public static LandauDistribution Landau(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Asymmetric Laplace, scipy's <c>laplace_asymmetric</c>.</summary>
    public static LaplaceAsymmetricDistribution LaplaceAsymmetric(double kappa, double loc = 0.0, double scale = 1.0) => new(kappa, loc, scale);
    /// <summary>Left-skewed Lévy, scipy's <c>levy_l</c>.</summary>
    public static LevyLDistribution LevyL(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Lévy alpha-stable, scipy's <c>levy_stable</c> (S1 by default, as scipy).</summary>
    public static LevyStableDistribution LevyStable(double alpha, double beta, double loc = 0.0, double scale = 1.0, LevyStableParameterization parameterization = LevyStableParameterization.S1)
        => new(alpha, beta, loc, scale, parameterization);
    /// <summary>Log-gamma, scipy's <c>loggamma</c>.</summary>
    public static LogGammaDistribution LogGamma(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Mielke Beta-Kappa, scipy's <c>mielke</c>.</summary>
    public static MielkeDistribution Mielke(double k, double s, double loc = 0.0, double scale = 1.0) => new(k, s, loc, scale);
    /// <summary>Moyal, scipy's <c>moyal</c>.</summary>
    public static MoyalDistribution Moyal(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Normal-inverse Gaussian, scipy's <c>norminvgauss</c>.</summary>
    public static NormInvGaussDistribution NormInvGauss(double a, double b, double loc = 0.0, double scale = 1.0) => new(a, b, loc, scale);
    /// <summary>Power log-normal, scipy's <c>powerlognorm</c>.</summary>
    public static PowerLogNormDistribution PowerLogNorm(double c, double s, double loc = 0.0, double scale = 1.0) => new(c, s, loc, scale);
    /// <summary>Power normal, scipy's <c>powernorm</c>.</summary>
    public static PowerNormDistribution PowerNorm(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>R distribution, scipy's <c>rdist</c>.</summary>
    public static RDistribution RDist(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Reciprocal inverse Gaussian, scipy's <c>recipinvgauss</c>.</summary>
    public static RecipInvGaussDistribution RecipInvGauss(double mu, double loc = 0.0, double scale = 1.0) => new(mu, loc, scale);
    /// <summary>Relativistic Breit–Wigner, scipy's <c>rel_breitwigner</c>.</summary>
    public static RelBreitWignerDistribution RelBreitWigner(double rho, double loc = 0.0, double scale = 1.0) => new(rho, loc, scale);
    /// <summary>Skewed Cauchy, scipy's <c>skewcauchy</c>.</summary>
    public static SkewCauchyDistribution SkewCauchy(double a, double loc = 0.0, double scale = 1.0) => new(a, loc, scale);
    /// <summary>Trapezoidal, scipy's <c>trapezoid</c>.</summary>
    public static TrapezoidDistribution Trapezoid(double c, double d, double loc = 0.0, double scale = 1.0) => new(c, d, loc, scale);
    /// <summary>Upper-truncated Pareto, scipy's <c>truncpareto</c>.</summary>
    public static TruncParetoDistribution TruncPareto(double b, double c, double loc = 0.0, double scale = 1.0) => new(b, c, loc, scale);
    /// <summary>Doubly truncated Weibull minimum, scipy's <c>truncweibull_min</c>.</summary>
    public static TruncWeibullMinDistribution TruncWeibullMin(double c, double a, double b, double loc = 0.0, double scale = 1.0) => new(c, a, b, loc, scale);
    /// <summary>von Mises on the line, scipy's <c>vonmises_line</c>.</summary>
    public static VonMisesLineDistribution VonMisesLine(double kappa, double loc = 0.0, double scale = 1.0) => new(kappa, loc, scale);
    /// <summary>Wrapped Cauchy, scipy's <c>wrapcauchy</c>.</summary>
    public static WrapCauchyDistribution WrapCauchy(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);

    // ------------------------------------------------------------------ discrete

    /// <summary>Beta negative binomial, scipy's <c>betanbinom</c>.</summary>
    public static BetaNegativeBinomialDistribution BetaNBinom(double n, double a, double b, double loc = 0.0) => new(n, a, b, loc);
    /// <summary>Fisher's noncentral hypergeometric (scipy's M, n, N, odds), scipy's <c>nchypergeom_fisher</c>.</summary>
    public static NcHypergeomFisherDistribution NcHypergeomFisher(int M, int n, int N, double odds, double loc = 0.0) => new(M, n, N, odds, loc);
    /// <summary>Wallenius' noncentral hypergeometric (scipy's M, n, N, odds), scipy's <c>nchypergeom_wallenius</c>.</summary>
    public static NcHypergeomWalleniusDistribution NcHypergeomWallenius(int M, int n, int N, double odds, double loc = 0.0) => new(M, n, N, odds, loc);
    /// <summary>Poisson binomial, scipy's <c>poisson_binom</c>.</summary>
    public static PoissonBinomialDistribution PoissonBinom(double[] p, double loc = 0.0) => new(p, loc);
}

/// <summary>
/// A fit that failed or would produce an invalid distribution, scipy's <c>FitError</c>: thrown by the
/// numerical MLE when the optimiser converges to parameters outside the family's domain, or to a
/// fit under which some observation has zero density.
/// </summary>
public sealed class FitError : InvalidOperationException
{
    /// <summary>A fit error with a message.</summary>
    public FitError(string message) : base(message) { }

    /// <summary>A fit error wrapping the exception that exposed it.</summary>
    public FitError(string message, Exception inner) : base(message, inner) { }
}
