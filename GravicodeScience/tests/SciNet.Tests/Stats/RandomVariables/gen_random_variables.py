"""Generates RandomVariableReference.Generated.cs from scipy 1.17's new distribution framework.

Run from this folder:  python gen_random_variables.py
Each case pairs a scipy expression with the C# expression that should build the same random
variable. For every function the script records scipy's value and whether scipy computed it by a
formula or numerically (quadrature / inversion / optimisation, read from the method cache), so
the C# test can use a tight tolerance for formulas and a looser one for numerics.
"""
import math
import warnings
import numpy as np
import scipy
from scipy import stats
from scipy.stats._distribution_infrastructure import DiscreteDistribution

warnings.simplefilter("ignore")

Normal, Uniform, Logistic, Binomial, Mixture = stats.Normal, stats.Uniform, stats.Logistic, stats.Binomial, stats.Mixture
truncate, order_statistic = stats.truncate, stats.order_statistic
make_distribution = stats.make_distribution


class MyTriangle:
    """scipy custom-distribution protocol: density 2x on [0, 1] with its CDF."""
    __make_distribution_version__ = "1.16.0"

    @property
    def parameters(self):
        # scipy 1.17.1 fails on a custom distribution with no parameters ('list' object has no
        # attribute 'matches'), so each carries a dummy one that the functions ignore.
        return {"k": {"endpoints": (0, np.inf)}}

    @property
    def support(self):
        return {"endpoints": (0.0, 1.0), "inclusive": (True, True)}

    def pdf(self, x, k):
        return 2 * x

    def cdf(self, x, k):
        return x * x


class MyHalfNormalPdfOnly:
    """Only a density: everything else is numerical in scipy."""
    __make_distribution_version__ = "1.16.0"

    @property
    def parameters(self):
        # scipy 1.17.1 fails on a custom distribution with no parameters ('list' object has no
        # attribute 'matches'), so each carries a dummy one that the functions ignore.
        return {"k": {"endpoints": (0, np.inf)}}

    @property
    def support(self):
        return {"endpoints": (0.0, np.inf)}

    def pdf(self, x, k):
        return np.sqrt(2 / np.pi) * np.exp(-x * x / 2)


Gamma = make_distribution(stats.gamma)
Rayleigh = make_distribution(stats.rayleigh)
Beta = make_distribution(stats.beta)
Poisson = make_distribution(stats.poisson)
Triangle = make_distribution(MyTriangle())
HalfNormalPdf = make_distribution(MyHalfNormalPdfOnly())

TRIANGLE_CS = ("Stats.MakeDistribution(new ContinuousDistributionDefinition { Name = \"MyTriangle\", Support = (0.0, 1.0), "
               "SupportInclusive = (true, true), Pdf = x => 2 * x, Cdf = x => x * x })")
HALFNORMAL_CS = ("Stats.MakeDistribution(new ContinuousDistributionDefinition { Name = \"MyHalfNormalPdfOnly\", "
                 "Support = (0.0, double.PositiveInfinity), Pdf = x => Math.Sqrt(2 / Math.PI) * Math.Exp(-x * x / 2) })")

P = [1e-300, 1e-100, 1e-15, 1e-5, 0.01, 0.3, 0.5, 0.77, 0.999, 1 - 1e-10]
P_SHORT = [1e-10, 0.01, 0.3, 0.5, 0.77, 0.999]
LOGP = [-700.0, -50.0, -5.0, -1.0, -0.3, -1e-5]

# (label, scipy expression, C# expression, x points, p points, log-p points, pairs, base tolerance)
CASES = [
    ("Normal(1.5, 2)", "Normal(mu=1.5, sigma=2.0)", "new Normal(mu: 1.5, sigma: 2.0)",
     [-80, -20, -3, 0, 1.5, 4, 25, 75], P, LOGP, [(-1, 2), (2, -1), (30, 40), (-40, -30), (-3, 50)], 1e-13),
    ("StandardNormal", "Normal()", "new StandardNormal()",
     [-38, -10, -1, 0, 0.5, 9, 38], P, LOGP + [-1e4], [(-1, 1), (5, 6), (-6, -5), (0, 0)], 1e-13),
    ("Uniform(-1, 3)", "Uniform(a=-1.0, b=3.0)", "new Uniform(a: -1.0, b: 3.0)",
     [-2, -1, -0.999, 0, 1.2, 2.9999, 3, 4], P, LOGP, [(-1, 3), (0, 1), (2, 0.5)], 1e-13),
    ("Logistic", "Logistic()", "new Logistic()",
     [-800, -40, -2, 0, 1, 30, 700], P, LOGP, [(-1, 1), (10, 20)], 1e-13),
    ("Binomial(20, 0.3)", "Binomial(n=20, p=0.3)", "new Binomial(n: 20, p: 0.3)",
     [-1, 0, 1, 2.5, 6, 13, 19, 20, 21], P_SHORT, [-20.0, -3.0, -0.5, -1e-3], [], 1e-12),
    # shift / scale arithmetic
    ("2*Normal()+1", "2.0*Normal() + 1.0", "2.0 * new StandardNormal() + 1.0",
     [-30, -1, 0, 1, 3, 30], P, LOGP, [(0, 2), (-40, -35)], 1e-13),
    ("-Normal(1,2)", "-Normal(mu=1.0, sigma=2.0)", "-new Normal(mu: 1.0, sigma: 2.0)",
     [-20, -1, 0, 3, 40], P, LOGP, [(-3, 1), (30, 40)], 1e-13),
    ("Normal()/4-3", "Normal()/4.0 - 3.0", "new StandardNormal() / 4.0 - 3.0",
     [-5, -3, -2.9, 0], P_SHORT, LOGP, [(-3.5, -2.5)], 1e-13),
    ("3*Logistic()-2", "3.0*Logistic() - 2.0", "3.0 * new Logistic() - 2.0",
     [-50, -2, 0, 10], P_SHORT, LOGP, [(-5, 1)], 1e-13),
    # truncation
    ("truncate(Normal(),-1,2)", "truncate(Normal(), -1.0, 2.0)", "Stats.Truncate(new StandardNormal(), -1.0, 2.0)",
     [-2, -1, -0.5, 0, 1, 1.999, 2, 3], P_SHORT, [-10.0, -1.0, -0.1], [(-0.5, 0.5)], 1e-12),
    ("truncate(2N+1, lb=0)", "truncate(2.0*Normal() + 1.0, lb=0.0)", "Stats.Truncate(2.0 * new StandardNormal() + 1.0, lb: 0.0)",
     [-1, 0, 0.1, 1, 5, 20], P_SHORT, [-10.0, -1.0, -0.1], [(0.5, 3)], 1e-12),
    ("truncate(Normal(), lb=5)", "truncate(Normal(), lb=5.0)", "Stats.Truncate(new StandardNormal(), lb: 5.0)",
     [5, 5.1, 6, 8], P_SHORT, [-10.0, -1.0, -0.1], [(5.5, 6.5)], 1e-11),
    # order statistics
    ("order_statistic(Normal(), r=2, n=5)", "order_statistic(Normal(), r=2, n=5)", "Stats.OrderStatistic(new StandardNormal(), r: 2, n: 5)",
     [-5, -1, 0, 0.7, 4], P_SHORT, [-20.0, -2.0, -0.2], [(-1, 0.5)], 1e-12),
    ("order_statistic(Uniform(0,1), r=3, n=4)", "order_statistic(Uniform(a=0.0, b=1.0), r=3, n=4)", "Stats.OrderStatistic(new Uniform(0.0, 1.0), r: 3, n: 4)",
     [0, 0.1, 0.5, 0.9, 1], P_SHORT, [-20.0, -2.0, -0.2], [(0.2, 0.6)], 1e-12),
    # monotonic and folding transforms
    ("abs(Normal(0.5,1))", "stats.abs(Normal(mu=0.5, sigma=1.0))", "Stats.Abs(new Normal(mu: 0.5, sigma: 1.0))",
     [-1, 0, 0.2, 1, 3, 9], P_SHORT, [-10.0, -1.0, -0.1], [(0.5, 2)], 1e-12),
    ("exp(Normal())", "stats.exp(Normal())", "Stats.Exp(new StandardNormal())",
     [0, 1e-5, 0.3, 1, 4, 300], P, LOGP, [(0.5, 2)], 1e-12),
    ("log(Uniform(1,3))", "stats.log(Uniform(a=1.0, b=3.0))", "Stats.Log(new Uniform(1.0, 3.0))",
     [-1, 0, 0.3, 1.0, 1.0986122886681098, 2], P_SHORT, [-10.0, -1.0, -0.1], [(0.2, 0.8)], 1e-12),
    ("Normal()**2", "Normal()**2", "new StandardNormal().Pow(2)",
     [-1, 0.01, 0.5, 1, 4, 30], P_SHORT, [-10.0, -1.0, -0.1], [(0.5, 2)], 1e-12),
    ("Normal()**3", "Normal()**3", "new StandardNormal().Pow(3)",
     [-8, -0.1, 0.2, 1, 27], P_SHORT, [-10.0, -1.0, -0.1], [(-1, 2)], 1e-12),
    ("1/Uniform(1,2)", "1/Uniform(a=1.0, b=2.0)", "1.0 / new Uniform(1.0, 2.0)",
     [0.4, 0.5, 0.7, 1, 1.2], P_SHORT, [-10.0, -1.0, -0.1], [(0.6, 0.9)], 1e-12),
    ("3/Uniform(1,2)", "3.0/Uniform(a=1.0, b=2.0)", "3.0 / new Uniform(1.0, 2.0)",
     [1.5, 2, 2.9], P_SHORT, [-10.0, -1.0, -0.1], [(1.8, 2.7)], 1e-12),
    ("2**Uniform(0,1)", "2.0**Uniform(a=0.0, b=1.0)", "ContinuousRandomVariable.Pow(2.0, new Uniform(0.0, 1.0))",
     [0.5, 1, 1.5, 2, 3], P_SHORT, [-10.0, -1.0, -0.1], [(1.2, 1.8)], 1e-12),
    ("0.5**Normal()", "0.5**Normal()", "ContinuousRandomVariable.Pow(0.5, new StandardNormal())",
     [0.01, 0.5, 1, 3, 100], P_SHORT, [-10.0, -1.0, -0.1], [(0.8, 1.5)], 1e-12),
    # make_distribution from rv_continuous / rv_discrete and from a user definition
    ("Gamma(a=2.5)", "Gamma(a=2.5)", "Stats.MakeDistribution(Stats.Gamma(2.5))",
     [0, 1e-3, 1, 2.5, 10, 60], P, LOGP, [(1, 3), (40, 50)], 1e-12),
    ("Rayleigh()", "Rayleigh()", "Stats.MakeDistribution(Stats.Rayleigh())",
     [0, 0.5, 1, 3, 30], P_SHORT, LOGP, [(0.5, 1.5)], 1e-12),
    ("3*Rayleigh()+1", "3.0*Rayleigh() + 1.0", "3.0 * Stats.MakeDistribution(Stats.Rayleigh()) + 1.0",
     [1, 2, 4, 30], P_SHORT, LOGP, [(2, 5)], 1e-12),
    ("Beta(a=2, b=3)", "Beta(a=2.0, b=3.0)", "Stats.MakeDistribution(Stats.Beta(2.0, 3.0))",
     [0, 0.1, 0.5, 0.99, 1], P_SHORT, LOGP, [(0.2, 0.4)], 1e-12),
    ("Poisson(mu=3.5)", "Poisson(mu=3.5)", "Stats.MakeDistribution(Stats.Poisson(3.5))",
     [-1, 0, 1, 2.5, 4, 12, 40], P_SHORT, [-20.0, -3.0, -0.5, -1e-3], [], 1e-12),
    ("MyTriangle", "Triangle(k=1.0)", TRIANGLE_CS,
     [-0.5, 0, 0.25, 0.5, 0.999, 1, 2], P_SHORT, [-10.0, -1.0, -0.1], [(0.2, 0.6)], 1e-12),
    ("MyHalfNormalPdfOnly", "HalfNormalPdf(k=1.0)", HALFNORMAL_CS,
     [0.0, 0.1, 1, 3, 8], [0.01, 0.3, 0.5, 0.77, 0.999], [-5.0, -1.0, -0.1], [(0.5, 1.5)], 1e-10),
    # mixtures
    ("Mixture(N(-1,0.5), N(2,1); 0.3/0.7)", "Mixture([Normal(mu=-1.0, sigma=0.5), Normal(mu=2.0, sigma=1.0)], weights=[0.3, 0.7])",
     "new Mixture([new Normal(-1.0, 0.5), new Normal(2.0, 1.0)], [0.3, 0.7])",
     [-10, -1, 0, 0.5, 2, 9], P_SHORT, [-10.0, -1.0, -0.1], [(-1, 2)], 1e-12),
    ("Mixture(Uniform(0,1), Normal())", "Mixture([Uniform(a=0.0, b=1.0), Normal()])", "new Mixture([new Uniform(0.0, 1.0), new StandardNormal()])",
     [-3, 0, 0.5, 1, 3], P_SHORT, [-10.0, -1.0, -0.1], [(0.25, 2)], 1e-12),
]


# ---------------------------------------------------------------------------------------------
# Where scipy is wrong. These values replace scipy's in the generated table; each comes from mpmath
# at 40 digits or from a closed form, and the comment says what scipy got and why.
import mpmath as mp
mp.mp.dps = 40


def moments_from_raw(raw):
    """Exact central and standardized moments (orders 0..len-1) from exact raw moments."""
    mean = raw[1]
    central = [sum(mp.binomial(n, i) * raw[i] * (-mean) ** (n - i) for i in range(n + 1)) for n in range(len(raw))]
    sd = mp.sqrt(central[2])
    std = [central[n] / sd ** n for n in range(len(raw))]
    return central, std


def exact_moment_overrides(raw):
    central, std = moments_from_raw(raw)
    o = {}
    for n in range(len(raw)):
        o[("moment_raw", n)] = raw[n]
        o[("moment_central", n)] = central[n]
        o[("moment_standardized", n)] = std[n]
    o[("mean", 0)] = raw[1]
    o[("variance", 0)] = central[2]
    o[("standard_deviation", 0)] = mp.sqrt(central[2])
    o[("skewness", 0)] = std[3]
    o[("kurtosis", 0)] = std[4]
    return o


def _truncated_normal_tail():
    s5 = mp.erfc(5 / mp.sqrt(2)) / 2
    def sfinv(q):
        return mp.findroot(lambda x: mp.log(mp.erfc(x / mp.sqrt(2)) / 2) - mp.log(q), 5.5)
    o = {("icdf", p): sfinv((1 - mp.mpf(p)) * s5) for p in [1e-10, 0.01, 0.3, 0.5, 0.77, 0.999]}
    o[("median", 0)] = o[("icdf", 0.5)]
    pdf = lambda x: mp.npdf(x) / s5
    raw = [mp.quad(lambda x: x ** k * pdf(x), [5, 6, 8, mp.inf]) for k in range(7)]
    o.update(exact_moment_overrides(raw))
    return o


def _reciprocal_uniform(c):
    # Y = c/U, U ~ Uniform(1, 2): density c/y^2 on [c/2, c], E[Y^k] = c^k ∫_1^2 u^-k du.
    raw = [mp.mpf(c) ** k * mp.quad(lambda u: u ** -k, [1, 2]) for k in range(7)]
    return exact_moment_overrides(raw)


def _normal_power(n):
    # E[Z^(n k)] = (nk - 1)!! for even nk, 0 for odd.
    raw = [mp.mpf(1) if k == 0 else (mp.fac2(n * k - 1) if (n * k) % 2 == 0 else mp.mpf(0)) for k in range(7)]
    return exact_moment_overrides(raw)


def _logistic():
    # E[X^n] = 2 n! eta(n) for even n.
    raw = [mp.mpf(1)] + [mp.mpf(0) if k % 2 else 2 * mp.factorial(k) * mp.altzeta(k) for k in range(1, 7)]
    return exact_moment_overrides(raw)


Z3_CDF2 = mp.ncdf(mp.cbrt(2)) - mp.ncdf(-1)
Z3_H = (1 + mp.log(2 * mp.pi)) / 2 + mp.log(3) - (mp.euler + mp.log(2))   # H(Z) + E log|3 Z^2|
Z2_H = mp.mpf(1) / 2 + mp.log(2 * mp.gamma(0.5)) + mp.mpf(1) / 2 * mp.digamma(0.5)  # chi2(1)
_mix_f = lambda x: 0.5 * (1 if 0 <= x <= 1 else 0) + 0.5 * mp.npdf(x)
MIX_UN_H = mp.quad(lambda x: -_mix_f(x) * mp.log(_mix_f(x)), [-mp.inf, 0, 1, mp.inf])

OVERRIDES = {
    # scipy inverts F(5) + p (1 - F(5)) with ndtri: F(5) = 1 - 2.9e-7 keeps nine digits of the tail
    # (icdf(0.999) off by 1e-8); its tanh-sinh 6th central moment is off by 2e-7.
    "truncate(Normal(), lb=5)": _truncated_normal_tail(),
    # scipy's quadrature of the 6th moment of c/U is off by 2e-9.
    "1/Uniform(1,2)": _reciprocal_uniform(1),
    "3/Uniform(1,2)": _reciprocal_uniform(3),
    # scipy takes log(sf) = log(1 - 9.5e-9): eight digits; and logaddexp of rounded log CDFs.
    "Gamma(a=2.5)": {("logccdf", 0.001): mp.log(mp.gammainc(2.5, 0.001, mp.inf, regularized=True)),
                     ("logccdf2", 40): mp.log(1 - mp.gammainc(2.5, 40, 50, regularized=True))},
    "Beta(a=2, b=3)": {("logcdf", 0.99): mp.log(mp.betainc(2, 3, 0, 0.99, regularized=True))},
    # The density of Z^3 has an integrable singularity at 0, inside the interval: scipy's one-piece
    # quadrature gives cdf(-1, 2) = 0.693 (exact 0.7375), entropy 1.584 (exact 1.2472), NaN moments
    # and a NaN mode. The density is unbounded at 0, so the mode is 0.
    "Normal()**3": {**_normal_power(3), **{("cdf2", -1): Z3_CDF2, ("logcdf2", -1): mp.log(Z3_CDF2),
                                             ("entropy", 0): Z3_H, ("logentropy", 0): mp.log(Z3_H), ("mode", 0): mp.mpf(0)}},
    # Z^2 is chi2(1): scipy's entropy is off by 2e-7, its moments by 1e-8, and its mode is NaN.
    "Normal()**2": {**_normal_power(2), **{("entropy", 0): Z2_H, ("logentropy", 0): mp.log(Z2_H), ("mode", 0): mp.mpf(0)}},
    # scipy's Binomial.cdf(2.5) is boost's incomplete-beta CDF at a non-integer argument, 0.0644,
    # between P(X <= 2) = 0.0355 and P(X <= 3) = 0.107. A discrete CDF is a step function: P(X <= 2).
    "Binomial(20, 0.3)": {("cdf", 2.5): mp.betainc(18, 3, 0, 0.7, regularized=True),
                          ("ccdf", 2.5): mp.betainc(3, 18, 0, 0.3, regularized=True),
                          ("logcdf", 2.5): mp.log(mp.betainc(18, 3, 0, 0.7, regularized=True)),
                          ("logccdf", 2.5): mp.log(mp.betainc(3, 18, 0, 0.3, regularized=True))},
    # The entropy of the density 2x on [0, 1] is 1/2 - log 2 < 0; scipy's complex log-space
    # quadrature gives log|H| = -1.6422, exact -1.64430.
    "MyTriangle": {("logentropy", 0): mp.log(mp.log(2) - mp.mpf(1) / 2)},
    # scipy's Bernoulli-number formula is off by 2e-12 in the 4th standardized moment.
    "Logistic": _logistic(),
    # The uniform component makes the density jump at 0 and 1; scipy's one-piece quadrature is 1e-5 off.
    "Mixture(Uniform(0,1), Normal())": {("entropy", 0): MIX_UN_H, ("logentropy", 0): mp.log(MIX_UN_H)},
}
# ---------------------------------------------------------------------------------------------

NUMERIC_TAGS = ("quadrature", "inversion", "optimization")


def classify(X, name):
    """True when scipy computed `name` numerically, looking through transformed distributions."""
    if isinstance(X, Mixture):
        return True  # every Mixture property is computed from component quantities or by quadrature
    d = X
    while d is not None:
        m = getattr(d, "_method_cache", {}).get(name)
        if m is not None:
            n = getattr(m, "__name__", str(m))
            return any(t in n for t in NUMERIC_TAGS)
        d = getattr(d, "_dist", None)
    return False


def lit(v):
    v = float(v)
    if math.isnan(v):
        return "double.NaN"
    if math.isinf(v):
        return "double.PositiveInfinity" if v > 0 else "double.NegativeInfinity"
    r = repr(v)
    return r if ("." in r or "e" in r or "inf" in r) else r + ".0"


def arr(xs):
    return "[" + ", ".join(lit(x) for x in xs) + "]"


def evaluate(X, fn, args, dispatch_name):
    out, numeric = [], False
    for a in args:
        try:
            if isinstance(a, tuple):
                v = getattr(X, fn)(*a)
            else:
                v = getattr(X, fn)(a)
            v = complex(np.asarray(v)[()])
            out.append(v)
        except Exception:
            out.append(None)
        numeric = numeric or classify(X, dispatch_name)
    return out, numeric


def main():
    lines = []
    lines.append(f"// <auto-generated> Reference values generated from scipy {scipy.__version__} (numpy {np.__version__}) by gen_random_variables.py. Do not edit. </auto-generated>")
    lines.append("using Gravicode.Science.SciNet;")
    lines.append("namespace Gravicode.Science.Tests.SciNet;")
    lines.append("internal static partial class RandomVariableReference")
    lines.append("{")
    lines.append("    public static readonly RvReferenceCase[] Cases =")
    lines.append("    [")
    current_label = [None]
    for label, py, cs, xs, ps, lps, pairs, tol in CASES:
        current_label[0] = label
        ov = OVERRIDES.get(label, {})
        X = eval(py)
        X.reset_cache() if hasattr(X, "reset_cache") else None
        discrete = isinstance(X, DiscreteDistribution)
        checks = []

        def add(fn, args, dispatch, values=None):
            vals, numeric = evaluate(X, fn, args, dispatch) if values is None else values
            ov = OVERRIDES.get(current_label[0], {})
            vals = [complex(float(ov[(fn, a[0] if isinstance(a, tuple) else a)]), (v.imag if v is not None else 0.0)) if (fn, a[0] if isinstance(a, tuple) else a) in ov else v
                    for a, v in zip(args, vals)]
            keep = [(a, v) for a, v in zip(args, vals) if v is not None]
            if not keep:
                return
            reals = [(a, v.real) for a, v in keep]
            if fn in ("logcdf2",):
                imag = [(a, v.imag) for a, v in keep]
            if isinstance(keep[0][0], tuple):
                a1 = [a[0] for a, _ in reals]
                a2 = [a[1] for a, _ in reals]
                checks.append(f"new(\"{fn}\", {arr(a1)}, {arr([v for _, v in reals])}, {str(numeric).lower()}, {arr(a2)})")
                if fn == "logcdf2":
                    checks.append(f"new(\"logcdf2imag\", {arr(a1)}, {arr([v for _, v in imag])}, false, {arr(a2)})")
            else:
                checks.append(f"new(\"{fn}\", {arr([a for a, _ in reals])}, {arr([v for _, v in reals])}, {str(numeric).lower()})")

        dens = "pmf" if discrete else "pdf"
        add(dens, xs, f"_{dens}_dispatch")
        add("log" + dens, xs, f"_log{dens}_dispatch")
        add("cdf", xs, "_cdf_dispatch")
        add("ccdf", xs, "_ccdf_dispatch")
        add("logcdf", xs, "_logcdf_dispatch")
        add("logccdf", xs, "_logccdf_dispatch")
        add("icdf", ps, "_icdf_dispatch")
        add("iccdf", ps, "_iccdf_dispatch")
        add("ilogcdf", lps, "_ilogcdf_dispatch")
        add("ilogccdf", lps, "_ilogccdf_dispatch")
        if pairs and not discrete:
            add("cdf2", pairs, "_cdf2_dispatch", values=evaluate_pairs(X, "cdf", pairs, "_cdf2_dispatch"))
            add("ccdf2", pairs, "_ccdf2_dispatch", values=evaluate_pairs(X, "ccdf", pairs, "_ccdf2_dispatch"))
            add("logcdf2", pairs, "_logcdf2_dispatch", values=evaluate_pairs(X, "logcdf", pairs, "_logcdf2_dispatch"))
            add("logccdf2", pairs, "_logccdf2_dispatch", values=evaluate_pairs(X, "logccdf", pairs, "_logccdf2_dispatch"))
        props = []
        for name, f, dispatch in [("entropy", lambda: X.entropy(), "_entropy_dispatch"),
                                  ("logentropy", lambda: X.logentropy().real, "_logentropy_dispatch"),
                                  ("median", lambda: X.median(), "_median_dispatch"),
                                  ("mode", lambda: X.mode(), "_mode_dispatch"),
                                  ("mean", lambda: X.mean(), None),
                                  ("variance", lambda: X.variance(), None),
                                  ("standard_deviation", lambda: X.standard_deviation(), None),
                                  ("skewness", lambda: X.skewness(), None),
                                  ("kurtosis", lambda: X.kurtosis(), None)]:
            try:
                v = float(np.real(f()))
            except Exception:
                continue
            if (name, 0) in ov:
                v = float(ov[(name, 0)])
            numeric = True if dispatch is None else classify(X, dispatch)
            if dispatch is None and type(X) in (stats.Normal, stats.Uniform, stats.Logistic, stats.Binomial) or label == "StandardNormal":
                numeric = False
            props.append(f"new(\"{name}\", [0.0], [{lit(v)}], {str(numeric).lower()})")
        for kind in ("raw", "central", "standardized"):
            orders, vals = [], []
            for order in range(0, 7):
                if isinstance(X, Mixture) and order > 4:
                    continue
                try:
                    v = float(np.real(X.moment(order, kind)))
                except Exception:
                    continue
                if (f"moment_{kind}", order) in ov:
                    v = float(ov[(f"moment_{kind}", order)])
                orders.append(order)
                vals.append(v)
            if orders:
                props.append(f"new(\"moment_{kind}\", {arr(orders)}, {arr(vals)}, true)")
        checks.extend(props)
        lines.append(f"        new(\"{label}\", () => {cs}, {lit(tol)},")
        lines.append("        [")
        for c in checks:
            lines.append(f"            {c},")
        lines.append("        ]),")
    lines.append("    ];")
    lines.append("}")
    with open("RandomVariableReference.Generated.cs", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def evaluate_pairs(X, fn, pairs, dispatch):
    out, numeric = [], False
    for x, y in pairs:
        try:
            v = complex(np.asarray(getattr(X, fn)(x, y))[()])
            out.append(v)
        except Exception:
            out.append(None)
        numeric = numeric or classify(X, dispatch)
    return out, numeric


if __name__ == "__main__":
    main()
