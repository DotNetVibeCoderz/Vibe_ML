using System.Numerics;

namespace Gravicode.Science.SciNet;

public static partial class Special
{
    private const double EiRootHi = 0.3725074107813666;
    private const double EiRootLo = 1.3140183414386028e-17;

    /// <summary>B_n/(n+1)!, the coefficients of Li₂ as a series in u = -log(1 - y) (mpmath).</summary>
    private static readonly double[] DilogBernoulli =
    [
        1.0, -2.5E-1, 2.7777777777777777778E-2, 0.0, -2.7777777777777777778E-4, 0.0, 4.7241118669690098262E-6, 0.0,
        -9.1857730746619635509E-8, 0.0, 1.8978869988970999072E-9, 0.0, -4.0647616451442255268E-11, 0.0,
        8.9216910204564525552E-13, 0.0, -1.9939295860721075687E-14, 0.0, 4.5189800296199181917E-16, 0.0,
        -1.0356517612181247014E-17, 0.0, 2.3952186210261867457E-19, 0.0, -5.5817858743250093363E-21,
    ];

    /// <summary>The exponential integral E₁(x) = ∫₁^∞ e^(-xt)/t dt for x &gt; 0 (scipy.special.exp1).</summary>
    /// <remarks>Power series for x ≤ 1, the Lentz-evaluated continued fraction beyond (Numerical Recipes expint).</remarks>
    public static double Exp1(double x) => Expn(1, x);

    /// <summary>The generalised exponential integral E_n(x) (scipy.special.expn).</summary>
    /// <remarks>
    /// Series with the digamma correction for x ≤ 1 and the continued fraction (DLMF 8.19.17)
    /// otherwise; E₀(x) = e^-x/x and E_n(0) = 1/(n - 1). NaN for n &lt; 0 or x &lt; 0.
    /// </remarks>
    public static double Expn(int n, double x)
    {
        if (double.IsNaN(x) || n < 0 || x < 0) return double.NaN;
        if (double.IsPositiveInfinity(x)) return 0.0;
        if (n == 0) return x == 0 ? double.PositiveInfinity : Math.Exp(-x) / x;
        if (x == 0) return n == 1 ? double.PositiveInfinity : 1.0 / (n - 1);
        var nm1 = n - 1;
        if (x > 1.0)
        {
            var b = x + n;
            var c = 1e300;
            var d = 1.0 / b;
            var h = d;
            for (var i = 1; i < 100000; i++)
            {
                var an = -(double)i * (nm1 + i);
                b += 2.0;
                d = 1.0 / (an * d + b);
                c = b + an / c;
                var del = c * d;
                h *= del;
                if (Math.Abs(del - 1.0) < 3e-16) break;
            }
            return h * Math.Exp(-x);
        }
        var ans = nm1 != 0 ? 1.0 / nm1 : -Math.Log(x) - EulerGamma;
        var fact = 1.0;
        for (var i = 1; i < 1000; i++)
        {
            fact *= -x / i;
            double del;
            if (i != nm1) del = -fact / (i - nm1);
            else
            {
                var psi = -EulerGamma;
                for (var ii = 1; ii <= nm1; ii++) psi += 1.0 / ii;
                del = fact * (-Math.Log(x) + psi);
            }
            ans += del;
            if (Math.Abs(del) < Math.Abs(ans) * 1e-17) break;
        }
        return ans;
    }

    /// <summary>The exponential integral Ei(x) (scipy.special.expi); Ei(x) = -E₁(-x) for x &lt; 0.</summary>
    /// <remarks>
    /// Positive-term power series up to 50 and the asymptotic series beyond. Near the root
    /// x₀ = 0.37250741... the series cancels, so there Ei(x) = ∫_{x₀}^{x} eᵗ/t dt is integrated by
    /// 20-point Gauss–Legendre, which keeps relative accuracy through the zero.
    /// </remarks>
    public static double Expi(double x)
    {
        if (double.IsNaN(x)) return x;
        if (x < 0) return -Exp1(-x);
        if (x == 0) return double.NegativeInfinity;
        if (double.IsPositiveInfinity(x)) return x;
        var dx = (x - EiRootHi) - EiRootLo;
        if (Math.Abs(dx) < 0.15)
        {
            var (nodes, weights) = GaussLegendre20Rule.Value;
            var mid = EiRootHi + 0.5 * dx;
            var half = 0.5 * dx;
            var s = 0.0;
            for (var i = 0; i < nodes.Length; i++)
            {
                var t = mid + half * nodes[i];
                s += weights[i] * Math.Exp(t) / t;
            }
            return s * half;
        }
        if (x < 50)
        {
            double sum = 0, term = 1;
            for (var k = 1; k < 500; k++)
            {
                term *= x / k;
                var t = term / k;
                sum += t;
                if (t < 1e-17 * sum) break;
            }
            return sum + EulerGamma + Math.Log(x);
        }
        {
            double sum = 1, term = 1, prev = double.PositiveInfinity;
            for (var k = 1; k < 200; k++)
            {
                var next = term * k / x;
                if (next >= prev) break;
                prev = next;
                term = next;
                sum += term;
                if (term < 1e-17 * sum) break;
            }
            if (x > 709) return Math.Exp(x - Math.Log(x)) * sum;
            return Math.Exp(x) / x * sum;
        }
    }

    /// <summary>Sine and cosine integrals (Si(x), Ci(x)) (scipy.special.sici).</summary>
    /// <remarks>
    /// Series for |x| &lt; 2, the complex continued fraction for E₁(ix) beyond (Numerical Recipes cisi).
    /// For negative x, Si is odd and Ci is reported as Ci(|x|), the real part, as SciPy does.
    /// </remarks>
    public static (double Si, double Ci) Sici(double x)
    {
        if (double.IsNaN(x)) return (x, x);
        var t = Math.Abs(x);
        double si, ci;
        if (t == 0) return (x, double.NegativeInfinity);
        if (double.IsPositiveInfinity(t)) return (x > 0 ? Math.PI / 2 : -Math.PI / 2, 0.0);
        if (t > 2.0)
        {
            var b = new Complex(1.0, t);
            var c = new Complex(1e300, 0);
            var d = 1.0 / b;
            var h = d;
            for (var i = 2; i < 100000; i++)
            {
                var a = -(double)(i - 1) * (i - 1);
                b += 2.0;
                d = 1.0 / (a * d + b);
                c = b + a / c;
                var del = c * d;
                h *= del;
                if (Math.Abs(del.Real - 1.0) + Math.Abs(del.Imaginary) < 3e-16) break;
            }
            h = new Complex(Math.Cos(t), -Math.Sin(t)) * h;
            ci = -h.Real;
            si = Math.PI / 2 + h.Imaginary;
        }
        else
        {
            double sum = 0, sums = 0, sumc = 0, sign = 1, fact = 1;
            var odd = true;
            for (var k = 1; k < 200; k++)
            {
                fact *= t / k;
                var term = fact / k;
                sum += sign * term;
                var err = term / Math.Abs(sum);
                if (odd) { sign = -sign; sums = sum; sum = sumc; }
                else { sumc = sum; sum = sums; }
                if (err < 1e-17) break;
                odd = !odd;
            }
            si = sums;
            ci = sumc + Math.Log(t) + EulerGamma;
        }
        return (x < 0 ? -si : si, ci);
    }

    /// <summary>Hyperbolic sine and cosine integrals (Shi(x), Chi(x)) (scipy.special.shichi).</summary>
    /// <remarks>
    /// Power series (Shi odd, Chi = γ + log|x| + even series) up to |x| = 8; beyond that
    /// Shi = (Ei + E₁)/2 and Chi = (Ei - E₁)/2. For negative x, Chi is the real part, as in SciPy.
    /// </remarks>
    public static (double Shi, double Chi) Shichi(double x)
    {
        if (double.IsNaN(x)) return (x, x);
        var t = Math.Abs(x);
        if (t == 0) return (x, double.NegativeInfinity);
        double shi, chi;
        if (t <= 8)
        {
            double sumS = 0, sumC = 0, term = 1;
            for (var k = 1; k < 300; k++)
            {
                term *= t / k;
                var v = term / k;
                if (k % 2 == 1) sumS += v; else sumC += v;
                if (v < 1e-17 * Math.Min(sumS, k > 1 ? sumC : sumS) && k > 2) break;
            }
            shi = sumS;
            chi = EulerGamma + Math.Log(t) + sumC;
        }
        else
        {
            var ei = Expi(t);
            var e1 = t > 745 ? 0.0 : Exp1(t);
            shi = 0.5 * (ei + e1);
            chi = 0.5 * (ei - e1);
        }
        return (x < 0 ? -shi : shi, chi);
    }

    /// <summary>
    /// Spence's function in SciPy's convention, spence(x) = ∫₁ˣ log t/(1 - t) dt = Li₂(1 - x), for
    /// x ≥ 0 (scipy.special.spence). NaN for x &lt; 0.
    /// </summary>
    /// <remarks>
    /// The dilogarithm is summed as its Bernoulli series in u = -log(1 - y), |u| ≤ log 2, after the
    /// reflection and inversion formulas map y = 1 - x there; u is formed from x directly, so
    /// spence keeps relative accuracy at its zero x = 1.
    /// </remarks>
    public static double Spence(double x)
    {
        if (double.IsNaN(x) || x < 0) return double.NaN;
        const double pi2over6 = 1.6449340668482264365;
        if (x == 0) return pi2over6;
        if (x == 1) return 0.0;
        if (double.IsPositiveInfinity(x)) return double.NegativeInfinity;
        if (x >= 0.5 && x <= 2.0) return DilogSeries(-Math.Log(x)); // y = 1 - x in [-1, 1/2]
        if (x < 0.5)
        {
            // Li₂(1 - x) = π²/6 - log(1 - x) log x - Li₂(x)
            return pi2over6 - Log1p(-x) * Math.Log(x) - DilogSeries(-Log1p(-x));
        }
        // y = 1 - x < -1: Li₂(y) = -π²/6 - ½ log²(-y) - Li₂(1/y), with 1/y = -1/(x-1)
        var l = Math.Log(x - 1);
        return -pi2over6 - 0.5 * l * l - DilogSeries(-Log1p(1.0 / (x - 1)));
    }

    private static double DilogSeries(double u)
    {
        var c = DilogBernoulli;
        var u2 = u * u;
        // u - u²/4 + Σ_k B_2k u^(2k+1)/(2k+1)!
        var odd = 0.0;
        for (var k = c.Length - 1; k >= 2; k -= 2) odd = odd * u2 + c[k];
        return c[0] * u + c[1] * u2 + odd * u * u2;
    }

    /// <summary>The real branches of the Lambert W function, w eʷ = x (scipy.special.lambertw, real part).</summary>
    /// <param name="x">Argument.</param>
    /// <param name="k">Branch: 0 (principal, x ≥ -1/e) or -1 (-1/e ≤ x &lt; 0).</param>
    /// <remarks>
    /// Halley iteration from the branch-point series in p = √(2(ex + 1)) (with x + 1/e formed from
    /// a two-word 1/e) or the asymptotic log form. Arguments off the real branches give NaN; use
    /// <see cref="LambertW(Complex, int)"/> for complex values.
    /// </remarks>
    public static double LambertW(double x, int k = 0)
    {
        if (double.IsNaN(x)) return x;
        const double em1Hi = 0.36787944117144233, em1Lo = -1.2428753672788363e-17;
        if (k != 0 && k != -1) return double.NaN;
        var d = (x + em1Hi) + em1Lo; // x + 1/e
        if (d < 0) return d > -1e-15 ? -1.0 : double.NaN;
        if (k == -1)
        {
            if (x >= 0) return x == 0 ? double.NegativeInfinity : double.NaN;
        }
        else
        {
            if (x == 0) return x;
            if (double.IsPositiveInfinity(x)) return x;
        }
        var p = Math.Sqrt(2 * Math.E * d);
        if (k == -1) p = -p;
        double w;
        if (Math.Abs(p) < 1e-3)
            return -1 + p * (1 + p * (-1.0 / 3 + p * (11.0 / 72 + p * (-43.0 / 540 + p * (769.0 / 17280 + p * (-221.0 / 8505))))));
        if (Math.Abs(p) < 0.6)
            w = -1 + p * (1 + p * (-1.0 / 3 + p * (11.0 / 72 + p * (-43.0 / 540 + p * (769.0 / 17280)))));
        else if (k == 0)
        {
            if (x < 3) w = 0.8 * Math.Log(1 + x);
            else
            {
                var l1 = Math.Log(x);
                var l2 = Math.Log(l1);
                w = l1 - l2 + l2 / l1;
            }
        }
        else
        {
            var l1 = Math.Log(-x);
            var l2 = Math.Log(-l1);
            w = l1 - l2 + l2 / l1;
        }
        for (var i = 0; i < 100; i++)
        {
            var ew = Math.Exp(w);
            var f = w * ew - x;
            var wp1 = w + 1;
            if (wp1 == 0) break;
            var dw = f / (ew * wp1 - (w + 2) * f / (2 * wp1));
            w -= dw;
            if (Math.Abs(dw) <= 1e-16 * (1 + Math.Abs(w))) break;
        }
        return w;
    }

    /// <summary>Lambert W on any branch k for complex z (scipy.special.lambertw).</summary>
    /// <remarks>
    /// Halley iteration from SciPy's starting values: the branch-point series near -1/e, a Padé
    /// approximant near the origin on the principal branch, the asymptotic expansion
    /// log z + 2πik - log(log z + 2πik) elsewhere.
    /// </remarks>
    public static Complex LambertW(Complex z, int k)
    {
        if (double.IsNaN(z.Real) || double.IsNaN(z.Imaginary)) return new Complex(double.NaN, double.NaN);
        if (z == Complex.Zero) return k == 0 ? Complex.Zero : new Complex(double.NegativeInfinity, 0);
        if (z.Imaginary == 0 && (k == 0 || k == -1))
        {
            var r = LambertW(z.Real, k);
            if (!double.IsNaN(r)) return new Complex(r, 0);
        }
        Complex w;
        var em1 = 0.36787944117144233;
        if (k == 0 && Complex.Abs(z + em1) < 0.3)
        {
            var p = Complex.Sqrt(2 * (Math.E * z + 1));
            w = -1 + p - p * p / 3 + 11.0 / 72 * p * p * p;
        }
        else if (k == -1 && Complex.Abs(z + em1) < 0.3 && z.Imaginary <= 0)
        {
            var p = -Complex.Sqrt(2 * (Math.E * z + 1));
            w = -1 + p - p * p / 3 + 11.0 / 72 * p * p * p;
        }
        else if (k == 0 && Complex.Abs(z - 0.5) < 1.5)
        {
            w = z * (3 + 6 * z + z * z) / (3 + 9 * z + 5 * z * z);
        }
        else
        {
            var l = Complex.Log(z) + new Complex(0, 2 * Math.PI * k);
            w = l - Complex.Log(l);
        }
        for (var i = 0; i < 100; i++)
        {
            var ew = Complex.Exp(w);
            var f = w * ew - z;
            var wp1 = w + 1;
            var dw = f / (ew * wp1 - (w + 2) * f / (2 * wp1));
            w -= dw;
            if (Complex.Abs(dw) <= 1e-16 * (1 + Complex.Abs(w))) break;
        }
        return w;
    }

    private static class GaussLegendre20Rule
    {
        internal static readonly (double[] Nodes, double[] Weights) Value = Build();

        private static (double[], double[]) Build()
        {
            var rule = RootsLegendre(20);
            return (rule.Nodes, rule.Weights);
        }
    }
}
