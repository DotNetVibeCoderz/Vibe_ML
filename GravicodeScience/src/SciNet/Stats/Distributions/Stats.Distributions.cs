using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// scipy.stats: probability distributions under scipy's names. Every factory returns a frozen
/// distribution (<see cref="ContinuousDistribution"/>, <see cref="DiscreteDistribution"/> or a
/// multivariate type) whose methods mirror scipy's (<c>Pdf</c>, <c>Cdf</c>, <c>Sf</c>, <c>Ppf</c>,
/// <c>Isf</c>, <c>Stats</c>, <c>Entropy</c>, <c>Rvs</c>, <c>FitTo</c>, …).
/// </summary>
/// <example><code>
/// var t = Stats.T(df: 7);
/// double crit = t.Isf(0.025);              // two-sided 5% critical value
/// var fitted = Stats.Gamma(1.0).FitTo(data); // MLE of shape, loc and scale
/// </code></example>
public static partial class Stats
{
    // ------------------------------------------------------------------ continuous

    /// <summary>Normal, scipy's <c>norm</c>.</summary>
    public static NormalDistribution Norm(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Uniform on [loc, loc + scale], scipy's <c>uniform</c>.</summary>
    public static UniformDistribution Uniform(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Exponential, scipy's <c>expon</c>.</summary>
    public static ExponentialDistribution Expon(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Gamma, scipy's <c>gamma</c>.</summary>
    public static GammaDistribution Gamma(double a, double loc = 0.0, double scale = 1.0) => new(a, loc, scale);
    /// <summary>Erlang, scipy's <c>erlang</c>.</summary>
    public static ErlangDistribution Erlang(double a, double loc = 0.0, double scale = 1.0) => new(a, loc, scale);
    /// <summary>Beta, scipy's <c>beta</c>.</summary>
    public static BetaDistribution Beta(double a, double b, double loc = 0.0, double scale = 1.0) => new(a, b, loc, scale);
    /// <summary>Student's t, scipy's <c>t</c>.</summary>
    public static StudentTDistribution T(double df, double loc = 0.0, double scale = 1.0) => new(df, loc, scale);
    /// <summary>Chi-squared, scipy's <c>chi2</c>.</summary>
    public static ChiSquaredDistribution Chi2(double df, double loc = 0.0, double scale = 1.0) => new(df, loc, scale);
    /// <summary>Chi, scipy's <c>chi</c>.</summary>
    public static ChiDistribution Chi(double df, double loc = 0.0, double scale = 1.0) => new(df, loc, scale);
    /// <summary>Fisher's F, scipy's <c>f</c>.</summary>
    public static FDistribution F(double dfn, double dfd, double loc = 0.0, double scale = 1.0) => new(dfn, dfd, loc, scale);
    /// <summary>Log-normal, scipy's <c>lognorm</c>.</summary>
    public static LogNormalDistribution LogNorm(double s, double loc = 0.0, double scale = 1.0) => new(s, loc, scale);
    /// <summary>Weibull (minima), scipy's <c>weibull_min</c>.</summary>
    public static WeibullMinDistribution WeibullMin(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Weibull (maxima), scipy's <c>weibull_max</c>.</summary>
    public static WeibullMaxDistribution WeibullMax(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Double Weibull, scipy's <c>dweibull</c>.</summary>
    public static DWeibullDistribution DWeibull(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Cauchy, scipy's <c>cauchy</c>.</summary>
    public static CauchyDistribution Cauchy(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Laplace, scipy's <c>laplace</c>.</summary>
    public static LaplaceDistribution Laplace(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Logistic, scipy's <c>logistic</c>.</summary>
    public static LogisticDistribution Logistic(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Pareto, scipy's <c>pareto</c>.</summary>
    public static ParetoDistribution Pareto(double b, double loc = 0.0, double scale = 1.0) => new(b, loc, scale);
    /// <summary>Lomax, scipy's <c>lomax</c>.</summary>
    public static LomaxDistribution Lomax(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Gumbel (maxima), scipy's <c>gumbel_r</c>.</summary>
    public static GumbelRDistribution GumbelR(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Gumbel (minima), scipy's <c>gumbel_l</c>.</summary>
    public static GumbelLDistribution GumbelL(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Rayleigh, scipy's <c>rayleigh</c>.</summary>
    public static RayleighDistribution Rayleigh(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Rice, scipy's <c>rice</c>.</summary>
    public static RiceDistribution Rice(double b, double loc = 0.0, double scale = 1.0) => new(b, loc, scale);
    /// <summary>von Mises, scipy's <c>vonmises</c>.</summary>
    public static VonMisesDistribution VonMises(double kappa, double loc = 0.0, double scale = 1.0) => new(kappa, loc, scale);
    /// <summary>Triangular, scipy's <c>triang</c>.</summary>
    public static TriangDistribution Triang(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Truncated normal with standardised bounds, scipy's <c>truncnorm</c>.</summary>
    public static TruncNormDistribution TruncNorm(double a, double b, double loc = 0.0, double scale = 1.0) => new(a, b, loc, scale);
    /// <summary>Truncated exponential, scipy's <c>truncexpon</c>.</summary>
    public static TruncExponDistribution TruncExpon(double b, double loc = 0.0, double scale = 1.0) => new(b, loc, scale);
    /// <summary>Half-normal, scipy's <c>halfnorm</c>.</summary>
    public static HalfNormalDistribution HalfNorm(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Half-Cauchy, scipy's <c>halfcauchy</c>.</summary>
    public static HalfCauchyDistribution HalfCauchy(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Inverse gamma, scipy's <c>invgamma</c>.</summary>
    public static InvGammaDistribution InvGamma(double a, double loc = 0.0, double scale = 1.0) => new(a, loc, scale);
    /// <summary>Inverse Gaussian, scipy's <c>invgauss</c>.</summary>
    public static InvGaussDistribution InvGauss(double mu, double loc = 0.0, double scale = 1.0) => new(mu, loc, scale);
    /// <summary>Wald, scipy's <c>wald</c>.</summary>
    public static WaldDistribution Wald(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Nakagami, scipy's <c>nakagami</c>.</summary>
    public static NakagamiDistribution Nakagami(double nu, double loc = 0.0, double scale = 1.0) => new(nu, loc, scale);
    /// <summary>Maxwell, scipy's <c>maxwell</c>.</summary>
    public static MaxwellDistribution Maxwell(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Generalised extreme value, scipy's <c>genextreme</c>.</summary>
    public static GenExtremeDistribution GenExtreme(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Generalised Pareto, scipy's <c>genpareto</c>.</summary>
    public static GenParetoDistribution GenPareto(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Generalised logistic, scipy's <c>genlogistic</c>.</summary>
    public static GenLogisticDistribution GenLogistic(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Skew-normal, scipy's <c>skewnorm</c>.</summary>
    public static SkewNormDistribution SkewNorm(double a, double loc = 0.0, double scale = 1.0) => new(a, loc, scale);
    /// <summary>Arcsine, scipy's <c>arcsine</c>.</summary>
    public static ArcsineDistribution Arcsine(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Power-function, scipy's <c>powerlaw</c>.</summary>
    public static PowerLawDistribution PowerLaw(double a, double loc = 0.0, double scale = 1.0) => new(a, loc, scale);
    /// <summary>Log-Laplace, scipy's <c>loglaplace</c>.</summary>
    public static LogLaplaceDistribution LogLaplace(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Lévy, scipy's <c>levy</c>.</summary>
    public static LevyDistribution Levy(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Burr type XII, scipy's <c>burr12</c>.</summary>
    public static Burr12Distribution Burr12(double c, double d, double loc = 0.0, double scale = 1.0) => new(c, d, loc, scale);
    /// <summary>Exponentially modified normal, scipy's <c>exponnorm</c>.</summary>
    public static ExponNormDistribution ExponNorm(double k, double loc = 0.0, double scale = 1.0) => new(k, loc, scale);
    /// <summary>Log-uniform, scipy's <c>loguniform</c>.</summary>
    public static LogUniformDistribution LogUniform(double a, double b, double loc = 0.0, double scale = 1.0) => new(a, b, loc, scale);
    /// <summary>Reciprocal (the same as <see cref="LogUniform"/>), scipy's <c>reciprocal</c>.</summary>
    public static LogUniformDistribution Reciprocal(double a, double b, double loc = 0.0, double scale = 1.0) => new(a, b, loc, scale);
    /// <summary>Johnson S_U, scipy's <c>johnsonsu</c>.</summary>
    public static JohnsonSUDistribution JohnsonSU(double a, double b, double loc = 0.0, double scale = 1.0) => new(a, b, loc, scale);
    /// <summary>Johnson S_B, scipy's <c>johnsonsb</c>.</summary>
    public static JohnsonSBDistribution JohnsonSB(double a, double b, double loc = 0.0, double scale = 1.0) => new(a, b, loc, scale);
    /// <summary>Folded normal, scipy's <c>foldnorm</c>.</summary>
    public static FoldNormDistribution FoldNorm(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Semicircular, scipy's <c>semicircular</c>.</summary>
    public static SemicircularDistribution Semicircular(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Raised cosine, scipy's <c>cosine</c>.</summary>
    public static CosineDistribution Cosine(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Anglit, scipy's <c>anglit</c>.</summary>
    public static AnglitDistribution Anglit(double loc = 0.0, double scale = 1.0) => new(loc, scale);
    /// <summary>Bradford, scipy's <c>bradford</c>.</summary>
    public static BradfordDistribution Bradford(double c, double loc = 0.0, double scale = 1.0) => new(c, loc, scale);
    /// <summary>Pearson type III, scipy's <c>pearson3</c>.</summary>
    public static Pearson3Distribution Pearson3(double skew, double loc = 0.0, double scale = 1.0) => new(skew, loc, scale);
    /// <summary>Noncentral chi-squared, scipy's <c>ncx2</c>.</summary>
    public static NonCentralChi2Distribution Ncx2(double df, double nc, double loc = 0.0, double scale = 1.0) => new(df, nc, loc, scale);
    /// <summary>Noncentral t, scipy's <c>nct</c>.</summary>
    public static NonCentralTDistribution Nct(double df, double nc, double loc = 0.0, double scale = 1.0) => new(df, nc, loc, scale);
    /// <summary>Noncentral F, scipy's <c>ncf</c>.</summary>
    public static NonCentralFDistribution Ncf(double dfn, double dfd, double nc, double loc = 0.0, double scale = 1.0) => new(dfn, dfd, nc, loc, scale);
    /// <summary>Studentized range, scipy's <c>studentized_range</c>.</summary>
    public static StudentizedRangeDistribution StudentizedRange(double k, double df, double loc = 0.0, double scale = 1.0) => new(k, df, loc, scale);
    /// <summary>Exact two-sided Kolmogorov–Smirnov statistic, scipy's <c>kstwo</c>.</summary>
    public static KSTwoDistribution KSTwo(int n, double loc = 0.0, double scale = 1.0) => new(n, loc, scale);
    /// <summary>Limiting Kolmogorov distribution, scipy's <c>kstwobign</c>.</summary>
    public static KSTwoBignDistribution KSTwoBign(double loc = 0.0, double scale = 1.0) => new(loc, scale);

    // ------------------------------------------------------------------ discrete

    /// <summary>Bernoulli, scipy's <c>bernoulli</c>.</summary>
    public static BernoulliDistribution Bernoulli(double p, double loc = 0.0) => new(p, loc);
    /// <summary>Binomial, scipy's <c>binom</c>.</summary>
    public static BinomialDistribution Binom(double n, double p, double loc = 0.0) => new(n, p, loc);
    /// <summary>Poisson, scipy's <c>poisson</c>.</summary>
    public static PoissonDistribution Poisson(double mu, double loc = 0.0) => new(mu, loc);
    /// <summary>Geometric on {1, 2, …}, scipy's <c>geom</c>.</summary>
    public static GeometricDistribution Geom(double p, double loc = 0.0) => new(p, loc);
    /// <summary>Negative binomial, scipy's <c>nbinom</c>.</summary>
    public static NegativeBinomialDistribution NBinom(double n, double p, double loc = 0.0) => new(n, p, loc);
    /// <summary>Hypergeometric (scipy's M, n, N), scipy's <c>hypergeom</c>.</summary>
    public static HypergeometricDistribution Hypergeom(int M, int n, int N, double loc = 0.0) => new(M, n, N, loc);
    /// <summary>Negative hypergeometric, scipy's <c>nhypergeom</c>.</summary>
    public static NegativeHypergeometricDistribution NHypergeom(int M, int n, int r, double loc = 0.0) => new(M, n, r, loc);
    /// <summary>Discrete uniform on {low, …, high−1}, scipy's <c>randint</c>.</summary>
    public static RandIntDistribution RandInt(double low, double high, double loc = 0.0) => new(low, high, loc);
    /// <summary>Zipf, scipy's <c>zipf</c>.</summary>
    public static ZipfDistribution Zipf(double a, double loc = 0.0) => new(a, loc);
    /// <summary>Truncated Zipf, scipy's <c>zipfian</c>.</summary>
    public static ZipfianDistribution Zipfian(double a, int n, double loc = 0.0) => new(a, n, loc);
    /// <summary>Logarithmic series, scipy's <c>logser</c>.</summary>
    public static LogSeriesDistribution Logser(double p, double loc = 0.0) => new(p, loc);
    /// <summary>Beta-binomial, scipy's <c>betabinom</c>.</summary>
    public static BetaBinomialDistribution BetaBinom(double n, double a, double b, double loc = 0.0) => new(n, a, b, loc);
    /// <summary>Skellam, scipy's <c>skellam</c>.</summary>
    public static SkellamDistribution Skellam(double mu1, double mu2, double loc = 0.0) => new(mu1, mu2, loc);
    /// <summary>Discrete Laplace, scipy's <c>dlaplace</c>.</summary>
    public static DLaplaceDistribution DLaplace(double a, double loc = 0.0) => new(a, loc);
    /// <summary>Boltzmann, scipy's <c>boltzmann</c>.</summary>
    public static BoltzmannDistribution Boltzmann(double lambda, int n, double loc = 0.0) => new(lambda, n, loc);
    /// <summary>Planck, scipy's <c>planck</c>.</summary>
    public static PlanckDistribution Planck(double lambda, double loc = 0.0) => new(lambda, loc);
    /// <summary>Yule–Simon, scipy's <c>yulesimon</c>.</summary>
    public static YuleSimonDistribution YuleSimon(double alpha, double loc = 0.0) => new(alpha, loc);

    // ------------------------------------------------------------------ multivariate

    /// <summary>Multivariate normal, scipy's <c>multivariate_normal</c>.</summary>
    public static MultivariateNormalDistribution MultivariateNormal(double[] mean, NdArray cov, bool allowSingular = false) => new(mean, cov, allowSingular);
    /// <summary>Multivariate normal with a factored <see cref="Covariance"/>, scipy's <c>multivariate_normal(mean, cov=Covariance)</c>.</summary>
    public static MultivariateNormalDistribution MultivariateNormal(double[] mean, Covariance cov) => new(mean, cov);
    /// <summary>Multivariate t, scipy's <c>multivariate_t</c>.</summary>
    public static MultivariateTDistribution MultivariateT(double[] loc, NdArray shape, double df = 1.0, bool allowSingular = false) => new(loc, shape, df, allowSingular);
    /// <summary>Dirichlet, scipy's <c>dirichlet</c>.</summary>
    public static DirichletDistribution Dirichlet(double[] alpha) => new(alpha);
    /// <summary>Multinomial, scipy's <c>multinomial</c>.</summary>
    public static MultinomialDistribution Multinomial(int n, double[] p) => new(n, p);
    /// <summary>Wishart, scipy's <c>wishart</c>.</summary>
    public static WishartDistribution Wishart(double df, NdArray scale) => new(df, scale);
    /// <summary>Inverse Wishart, scipy's <c>invwishart</c>.</summary>
    public static InvWishartDistribution InvWishart(double df, NdArray scale) => new(df, scale);

    /// <summary>A Haar-random orthogonal matrix, scipy's <c>ortho_group.rvs</c>.</summary>
    public static NdArray OrthoGroup(int dim, int? seed = null) => RandomMatrixGroups.OrthoGroup(dim, StatsDistributions.MakeRng(seed));
    /// <summary>A Haar-random rotation matrix, scipy's <c>special_ortho_group.rvs</c>.</summary>
    public static NdArray SpecialOrthoGroup(int dim, int? seed = null) => RandomMatrixGroups.SpecialOrthoGroup(dim, StatsDistributions.MakeRng(seed));

    /// <summary>
    /// Maximum-likelihood fit of any continuous family, scipy's <c>dist.fit</c>: the prototype names
    /// the family and its shapes are the starting point. Closed forms are used where the family has
    /// them; see <see cref="ContinuousDistribution.FitTo(double[], double?[], double?, double?)"/>.
    /// </summary>
    public static ContinuousDistribution Fit(ContinuousDistribution family, double[] data, double?[]? fixedShapes = null, double? floc = null, double? fscale = null)
    {
        ArgumentNullException.ThrowIfNull(family);
        return family.FitTo(data, fixedShapes, floc, fscale);
    }
}
