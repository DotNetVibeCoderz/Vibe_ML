using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Numerical helpers for the families added in ContinuousMoreFamilies*.cs.</summary>
internal static class StatsDistMoreMath
{
    private static readonly double[] GlNodes;
    private static readonly double[] GlWeights;

    static StatsDistMoreMath()
    {
        // 20-point Gauss–Legendre on [-1, 1] by Newton on P₂₀; computed once, to full precision.
        const int n = 20;
        GlNodes = new double[n];
        GlWeights = new double[n];
        for (var i = 0; i < n; i++)
        {
            var x = Math.Cos(Math.PI * (i + 0.75) / (n + 0.5));
            double dp = 0;
            for (var it = 0; it < 100; it++)
            {
                double p0 = 1, p1 = x;
                for (var k = 2; k <= n; k++)
                {
                    var p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k;
                    p0 = p1;
                    p1 = p2;
                }
                dp = n * (x * p1 - p0) / (x * x - 1);
                var dx = p1 / dp;
                x -= dx;
                if (Math.Abs(dx) < 1e-16) break;
            }
            GlNodes[i] = x;
            GlWeights[i] = 2 / ((1 - x * x) * dp * dp);
        }
    }

    /// <summary>∫ f over [a, b] by fixed 20-point Gauss–Legendre: exact enough for a smooth f on a short interval.</summary>
    public static double GaussLegendre20(Func<double, double> f, double a, double b)
    {
        var h = 0.5 * (b - a);
        var m = 0.5 * (a + b);
        var s = 0.0;
        for (var i = 0; i < GlNodes.Length; i++) s += GlWeights[i] * f(m + h * GlNodes[i]);
        return s * h;
    }

    /// <summary>
    /// P(lo &lt; Z &lt; hi) for a standard normal Z with relative accuracy: a short interval is
    /// integrated directly (differencing two nearly equal CDFs keeps none of its digits), a
    /// one-sided one as Φ(hi)·(1 − Φ(lo)/Φ(hi)) in log space.
    /// </summary>
    public static double NormInterval(double lo, double hi)
    {
        if (!(hi > lo)) return 0.0;
        if (double.IsFinite(lo) && double.IsFinite(hi) && (hi - lo) * Math.Max(1.0, Math.Max(Math.Abs(lo), Math.Abs(hi))) < 0.5)
            return GaussLegendre20(t => Math.Exp(-0.5 * t * t), lo, hi) / StatsDistMath.Sqrt2Pi;
        if (hi <= 0)
        {
            var lh = Special.LogNdtr(hi);
            return Math.Exp(lh) * -Special.Expm1(Special.LogNdtr(lo) - lh);
        }
        if (lo >= 0)
        {
            var ll = Special.LogNdtr(-lo);
            return Math.Exp(ll) * -Special.Expm1(Special.LogNdtr(-hi) - ll);
        }
        return 1.0 - StatsDistMath.Phi(lo) - StatsDistMath.Phi(-hi);
    }

    /// <summary>log(1 + exp(x)) without overflow.</summary>
    public static double Log1pExp(double x) => x > 0 ? x + Special.Log1p(Math.Exp(-x)) : Special.Log1p(Math.Exp(x));

    /// <summary>exp(x)·E₁(x), the scaled exponential integral, finite for every x &gt; 0.</summary>
    public static double ScaledExp1(double x)
    {
        if (x < 50) return Special.Exp1(x) * Math.Exp(x);
        // Asymptotic series Σ (−1)ᵏ k!/x^{k+1}; at x ≥ 50 its smallest term is far below 1e-17.
        double term = 1 / x, s = 0;
        for (var k = 0; k < 40; k++)
        {
            s += term;
            term *= -(k + 1) / x;
            if (Math.Abs(term) < 1e-18 * Math.Abs(s)) break;
        }
        return s;
    }

    /// <summary>
    /// ∫ f over [a, b] adaptively, after splitting at <paramref name="points"/> inside the range, with
    /// a purely relative tolerance: densities far in a tail are tiny and an absolute floor would
    /// return zero for them.
    /// </summary>
    public static double QuadRel(Func<double, double> f, double a, double b, IEnumerable<double>? points = null, double epsrel = 1e-13)
    {
        var cuts = new List<double> { a };
        if (points != null)
            foreach (var p in points)
                if (p > a && p < b && double.IsFinite(p)) cuts.Add(p);
        cuts.Add(b);
        cuts.Sort();
        // Cuts a few ulps apart make degenerate pieces that QUADPACK integrates badly; merge them.
        var minGap = double.IsFinite(a) && double.IsFinite(b) ? 1e-12 * (b - a) : 0.0;
        for (var i = cuts.Count - 2; i >= 1; i--)
            if (cuts[i] - cuts[i - 1] <= minGap || cuts[i + 1] - cuts[i] <= minGap) cuts.RemoveAt(i);
        var total = 0.0;
        for (var i = 0; i + 1 < cuts.Count; i++)
        {
            if (!(cuts[i + 1] > cuts[i])) continue;
            total += Integrate.Quad(f, cuts[i], cuts[i + 1], epsabs: 0.0, epsrel: epsrel, limit: 200).Value;
        }
        return total;
    }

    /// <summary>Raw moment E[Yᵏ] of scipy's standard <c>invgauss(μ)</c> (λ = 1).</summary>
    public static double InvGaussRawMoment(double mu, int k)
    {
        if (k == 0) return 1.0;
        var s = 0.0;
        for (var i = 0; i < k; i++)
            s += Math.Exp(Special.GammaLn(k + i) - Special.GammaLn(i + 1) - Special.GammaLn(k - i)) * Math.Pow(mu / 2, i);
        return Math.Pow(mu, k) * s;
    }

    /// <summary>Mean, variance, skewness and excess kurtosis from the first four raw moments.</summary>
    public static DistributionMoments FromRaw(double m1, double m2, double m3, double m4)
    {
        var v = m2 - m1 * m1;
        var mu3 = m3 - 3 * m1 * m2 + 2 * m1 * m1 * m1;
        var mu4 = m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1 * m1 * m1 * m1;
        return new DistributionMoments(m1, v, mu3 / Math.Pow(v, 1.5), mu4 / (v * v) - 3.0);
    }
}
