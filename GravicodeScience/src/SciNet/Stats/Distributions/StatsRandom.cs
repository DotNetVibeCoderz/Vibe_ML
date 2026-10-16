using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Variate generators behind the distributions' <c>Rvs</c>: 256-layer ziggurats for the normal and
/// exponential (Marsaglia &amp; Tsang 2000), Marsaglia–Tsang gamma, and Hörmann's PTRS Poisson.
/// All draw from a <see cref="GraviRandom"/>; reproducing NumPy's streams is not a goal.
/// </summary>
internal static class StatsRandom
{
    private const int Layers = 256;
    private const double NormR = 3.6541528853610088;
    private const double NormV = 0.00492867323399;
    private const double ExpR = 7.69711747013104972;
    private const double ExpV = 0.0039496598225815571993;
    private const double Two53Inv = 1.0 / 9007199254740992.0;

    private static readonly double[] Nx = new double[Layers + 1];
    private static readonly double[] Nf = new double[Layers + 1];
    private static readonly double[] Ex = new double[Layers + 1];
    private static readonly double[] Ef = new double[Layers + 1];

    static StatsRandom()
    {
        // Layer i spans [0, x[i]]; layer 0 is the base strip whose "width" v/f(r) includes the tail.
        Nx[0] = NormV / Math.Exp(-0.5 * NormR * NormR);
        Nx[1] = NormR;
        for (var i = 1; i < Layers - 1; i++)
            Nx[i + 1] = Math.Sqrt(-2.0 * Math.Log(NormV / Nx[i] + Math.Exp(-0.5 * Nx[i] * Nx[i])));
        Nx[Layers] = 0.0;
        for (var i = 0; i <= Layers; i++) Nf[i] = Math.Exp(-0.5 * Nx[i] * Nx[i]);

        Ex[0] = ExpV / Math.Exp(-ExpR);
        Ex[1] = ExpR;
        for (var i = 1; i < Layers - 1; i++)
            Ex[i + 1] = -Math.Log(ExpV / Ex[i] + Math.Exp(-Ex[i]));
        Ex[Layers] = 0.0;
        for (var i = 0; i <= Layers; i++) Ef[i] = Math.Exp(-Ex[i]);
    }

    /// <summary>Uniform on the open interval (0, 1).</summary>
    public static double U(GraviRandom rng) => ((rng.NextUInt64() >> 11) + 0.5) * Two53Inv;

    /// <summary>Standard normal by the ziggurat: ~99% of draws cost one 64-bit word and a compare.</summary>
    public static double Normal(GraviRandom rng)
    {
        while (true)
        {
            var bits = rng.NextUInt64();
            var i = (int)(bits & 0xFF);
            // Bits 11..63 give the magnitude, bit 8 the sign: disjoint from the layer index bits.
            var u = (bits >> 11) * Two53Inv;
            var sign = (bits & 0x100) != 0 ? -1.0 : 1.0;
            var x = u * Nx[i];
            if (x < Nx[i + 1]) return sign * x;
            if (i == 0)
            {
                // Tail beyond r (Marsaglia 1964).
                double a, b;
                do
                {
                    a = -Math.Log(U(rng)) / NormR;
                    b = -Math.Log(U(rng));
                } while (2.0 * b < a * a);
                return sign * (NormR + a);
            }
            var y = Nf[i] + U(rng) * (Nf[i + 1] - Nf[i]);
            if (y < Math.Exp(-0.5 * x * x)) return sign * x;
        }
    }

    /// <summary>Standard exponential by the ziggurat.</summary>
    public static double Exponential(GraviRandom rng)
    {
        while (true)
        {
            var bits = rng.NextUInt64();
            var i = (int)(bits & 0xFF);
            var x = (bits >> 11) * Two53Inv * Ex[i];
            if (x < Ex[i + 1]) return x;
            if (i == 0) return ExpR - Math.Log(U(rng));
            var y = Ef[i] + U(rng) * (Ef[i + 1] - Ef[i]);
            if (y < Math.Exp(-x)) return x;
        }
    }

    /// <summary>Gamma(shape, 1) by Marsaglia–Tsang; shapes below 1 use the U^(1/a) boost in log space.</summary>
    public static double Gamma(GraviRandom rng, double shape)
    {
        if (shape < 1.0)
        {
            // Computed in log space so that tiny shapes underflow gracefully to 0 instead of NaN.
            var g = Gamma(rng, shape + 1.0);
            return Math.Exp(Math.Log(g) + Math.Log(U(rng)) / shape);
        }
        if (shape == 1.0) return Exponential(rng);
        var d = shape - 1.0 / 3.0;
        var c = 1.0 / Math.Sqrt(9.0 * d);
        while (true)
        {
            double x, v;
            do
            {
                x = Normal(rng);
                v = 1.0 + c * x;
            } while (v <= 0);
            v = v * v * v;
            var u = U(rng);
            var x2 = x * x;
            if (u < 1.0 - 0.0331 * x2 * x2) return d * v;
            if (Math.Log(u) < 0.5 * x2 + d * (1.0 - v + Math.Log(v))) return d * v;
        }
    }

    /// <summary>Beta(a, b) from two gammas.</summary>
    public static double Beta(GraviRandom rng, double a, double b)
    {
        if (a < 1 && b < 1)
        {
            // Jöhnk's method keeps tiny shapes from producing 0/0 out of two underflowed gammas.
            while (true)
            {
                var lu = Math.Log(U(rng)) / a;
                var lv = Math.Log(U(rng)) / b;
                var ls = StatsDistMath.LogAddExp(lu, lv);
                if (ls <= 0) return Math.Exp(lu - ls);
            }
        }
        var x = Gamma(rng, a);
        var y = Gamma(rng, b);
        return x / (x + y);
    }

    /// <summary>Poisson: inversion for small means, Hörmann's PTRS (1993) transformed rejection above 10.</summary>
    public static double Poisson(GraviRandom rng, double lam)
    {
        if (lam == 0) return 0;
        if (lam < 10)
        {
            var p = Math.Exp(-lam);
            var k = 0;
            var s = p;
            var u = U(rng);
            while (u > s && k < 1000)
            {
                k++;
                p *= lam / k;
                s += p;
            }
            return k;
        }
        var slam = Math.Sqrt(lam);
        var loglam = Math.Log(lam);
        var b = 0.931 + 2.53 * slam;
        var a = -0.059 + 0.02483 * b;
        var invalpha = 1.1239 + 1.1328 / (b - 3.4);
        var vr = 0.9277 - 3.6224 / (b - 2);
        while (true)
        {
            var u = U(rng) - 0.5;
            var v = U(rng);
            var us = 0.5 - Math.Abs(u);
            var k = Math.Floor((2 * a / us + b) * u + lam + 0.43);
            if (us >= 0.07 && v <= vr) return k;
            if (k < 0 || (us < 0.013 && v > us)) continue;
            if (Math.Log(v) + Math.Log(invalpha) - Math.Log(a / (us * us) + b) <= -lam + k * loglam - Special.GammaLn(k + 1))
                return k;
        }
    }

    /// <summary>Binomial(n, p): sequential inversion for a small mean, otherwise inversion of the exact CDF via <paramref name="exact"/>.</summary>
    public static double Binomial(GraviRandom rng, double n, double p, DiscreteDistribution exact)
    {
        if (p == 0 || n == 0) return 0;
        if (p == 1) return n;
        var flip = p > 0.5;
        var pp = flip ? 1 - p : p;
        double k;
        if (n * pp < 30)
        {
            // Sequential inversion from 0; expected n·p steps.
            var q = 1 - pp;
            var r = pp / q;
            var f = Math.Exp(n * Special.Log1p(-pp));
            var u = U(rng);
            k = 0;
            while (u > f && k < n)
            {
                u -= f;
                f *= r * (n - k) / (k + 1);
                k++;
            }
            return flip ? n - k : k;
        }
        // Large mean: exact inversion of the CDF, which starts from a normal guess and steps.
        return exact.Ppf(U(rng));
    }
}
