namespace Gravicode.Science.SciNet;

/// <summary>
/// Catherine Loader's saddle-point algorithm (2000) for binomial and Poisson probabilities, as used
/// by R's <c>dbinom</c>/<c>dpois</c>. Writing the mass function through Stirling remainders and the
/// deviance bd0 avoids the cancellation of lnΓ(n+1) − lnΓ(k+1) − lnΓ(n−k+1), which costs ~log₁₀(n log n)
/// digits for large n.
/// </summary>
internal static class StatsLoader
{
    private static readonly double[] SferrHalves =
    [
        0.0, 0.1534264097200273452913848, 0.0810614667953272582196702, 0.0548141210519176538961390,
        0.0413406959554092940938221, 0.03316287351993628748511048, 0.02767792568499833914878929,
        0.02374616365629749597132920, 0.02079067210376509311152277, 0.01848845053267318523077934,
        0.01664469118982119216319487, 0.01513497322191737887351255, 0.01387612882307074799874573,
        0.01281046524292022692424986, 0.01189670994589177009505572, 0.01110455975820691732662991,
        0.010411265261972096497478567, 0.009799416126158803298389475, 0.009255462182712732917728637,
        0.008768700134139385462952823, 0.008330563433362871256469318, 0.007934114564314020547248100,
        0.007573675487951840794972024, 0.007244554301320383179543912, 0.006942840107209529865664152,
        0.006665247032707682442354394, 0.006408994188004207068439631, 0.006171712263039457647532867,
        0.005951370112758847735624416, 0.005746216513010115682023589, 0.005554733551962801371038690,
    ];

    /// <summary>Stirling's remainder lnΓ(n+1) − [(n+½)ln n − n + ln√(2π)]; exact table on half-integers up to 15.</summary>
    public static double StirlErr(double n)
    {
        const double s0 = 0.083333333333333333333, s1 = 0.00277777777777777777778, s2 = 0.00079365079365079365079365,
            s3 = 0.000595238095238095238095238, s4 = 0.0008417508417508417508417508;
        if (n <= 15.0)
        {
            var nn = n + n;
            if (nn == Math.Floor(nn)) return SferrHalves[(int)nn];
            return Special.GammaLn(n + 1) - (n + 0.5) * Math.Log(n) + n - StatsDistMath.LnSqrt2Pi;
        }
        var nn2 = n * n;
        if (n > 500) return (s0 - s1 / nn2) / n;
        if (n > 80) return (s0 - (s1 - s2 / nn2) / nn2) / n;
        if (n > 35) return (s0 - (s1 - (s2 - s3 / nn2) / nn2) / nn2) / n;
        return (s0 - (s1 - (s2 - (s3 - s4 / nn2) / nn2) / nn2) / nn2) / n;
    }

    /// <summary>The deviance term x log(x/np) + np − x, summed as a series when x ≈ np.</summary>
    public static double Bd0(double x, double np)
    {
        if (Math.Abs(x - np) < 0.1 * (x + np))
        {
            var v = (x - np) / (x + np);
            var s = (x - np) * v;
            var ej = 2 * x * v;
            v *= v;
            for (var j = 1; j < 1000; j++)
            {
                ej *= v;
                var s1 = s + ej / (2 * j + 1);
                if (s1 == s) return s1;
                s = s1;
            }
            return s;
        }
        return x * Math.Log(x / np) + np - x;
    }

    /// <summary>log P(X = x) for X ~ Binomial(n, p), q = 1 − p passed separately for accuracy.</summary>
    public static double LogDbinom(double x, double n, double p, double q)
    {
        if (p == 0) return x == 0 ? 0 : double.NegativeInfinity;
        if (q == 0) return x == n ? 0 : double.NegativeInfinity;
        if (x == 0) return n == 0 ? 0 : (p < 0.1 ? -Bd0(n, n * q) - n * p : n * Math.Log(q));
        if (x == n) return q < 0.1 ? -Bd0(n, n * p) - n * q : n * Math.Log(p);
        if (x < 0 || x > n) return double.NegativeInfinity;
        var lc = StirlErr(n) - StirlErr(x) - StirlErr(n - x) - Bd0(x, n * p) - Bd0(n - x, n * q);
        var lf = StatsDistMath.Ln2 + Math.Log(Math.PI) + Math.Log(x) + Special.Log1p(-x / n);
        return lc - 0.5 * lf;
    }

    /// <summary>log P(X = x) for X ~ Poisson(λ).</summary>
    public static double LogDpois(double x, double lambda)
    {
        if (lambda == 0) return x == 0 ? 0 : double.NegativeInfinity;
        if (x < 0) return double.NegativeInfinity;
        if (x == 0) return -lambda;
        return -StirlErr(x) - Bd0(x, lambda) - 0.5 * Math.Log(2 * Math.PI * x);
    }
}
