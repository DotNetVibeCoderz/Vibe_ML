namespace Gravicode.Science.SciNet;

/// <summary>Closed-form and one-dimensional maximum-likelihood solvers shared by the families' <c>Fit</c> methods.</summary>
internal static class StatsFitHelpers
{
    /// <summary>NumPy's median (mean of the two middle values for even n).</summary>
    public static double Median(double[] data)
    {
        var s = (double[])data.Clone();
        Array.Sort(s);
        var n = s.Length;
        return n % 2 == 1 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
    }

    /// <summary>Biased sample skewness m₃/m₂^1.5 (scipy.stats.skew's default).</summary>
    public static double Skewness(double[] data)
    {
        var m = data.Average();
        double m2 = 0, m3 = 0;
        foreach (var x in data) { var d = x - m; m2 += d * d; m3 += d * d * d; }
        m2 /= data.Length;
        m3 /= data.Length;
        return m2 > 0 ? m3 / Math.Pow(m2, 1.5) : 0.0;
    }

    /// <summary>Finds the root of a monotone function by expanding a bracket around <paramref name="guess"/> (positive domain).</summary>
    public static double PositiveRoot(Func<double, double> f, double guess)
    {
        var lo = guess;
        var hi = guess;
        var flo = f(lo);
        var fhi = flo;
        for (var i = 0; i < 200 && Math.Sign(flo) == Math.Sign(fhi); i++)
        {
            lo /= 2; hi *= 2;
            flo = f(lo); fhi = f(hi);
        }
        if (Math.Sign(flo) == Math.Sign(fhi)) throw new ArithmeticException("Could not bracket the likelihood equation.");
        return StatsDistMath.Brent(f, lo, hi, flo, fhi);
    }

    /// <summary>Gumbel (maxima) MLE; see <see cref="GumbelRDistribution.Fit"/>.</summary>
    public static (double Loc, double Scale) GumbelR(double[] data, double? floc, double? fscale)
    {
        var n = data.Length;
        var mean = data.Average();
        var max = data.Max();
        var min = data.Min();
        if (floc is double l && fscale is double s0) return (l, s0);
        if (fscale is double s)
        {
            // loc = −s log mean exp(−x/s), shifted by the minimum for overflow safety.
            var sum = 0.0;
            foreach (var x in data) sum += Math.Exp(-(x - min) / s);
            return (min - s * Math.Log(sum / n), s);
        }
        if (floc is double loc)
        {
            // d/ds of the log-likelihood with loc fixed: n·s = Σ(x−loc) − Σ(x−loc)e^{−(x−loc)/s}... solve numerically.
            double G(double sc)
            {
                double a = 0, b = 0;
                foreach (var x in data)
                {
                    var z = (x - loc) / sc;
                    a += z;
                    b += z * Math.Exp(-z);
                }
                return a - b - n;
            }
            var sd0 = Math.Sqrt(data.Sum(x => (x - mean) * (x - mean)) / n) * Math.Sqrt(6) / Math.PI;
            return (loc, PositiveRoot(G, sd0 > 0 ? sd0 : 1.0));
        }
        double F(double sc)
        {
            double sw = 0, sxw = 0;
            foreach (var x in data)
            {
                var w = Math.Exp(-(x - max) / sc);
                sw += w;
                sxw += x * w;
            }
            return mean - sxw / sw - sc;
        }
        var sd = Math.Sqrt(data.Sum(x => (x - mean) * (x - mean)) / n) * Math.Sqrt(6) / Math.PI;
        var scale = PositiveRoot(F, sd > 0 ? sd : 1.0);
        var t = 0.0;
        foreach (var x in data) t += Math.Exp(-(x - min) / scale);
        return (min - scale * Math.Log(t / n), scale);
    }
}
