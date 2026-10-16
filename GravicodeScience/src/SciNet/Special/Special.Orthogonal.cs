using System.Numerics;

namespace Gravicode.Science.SciNet;

/// <summary>
/// A Gauss quadrature rule: ∫ w(x) f(x) dx ≈ Σ Weights[i] f(Nodes[i]), exact for polynomials of
/// degree below 2n. Nodes are ascending. <see cref="Mu"/> is the total mass of the weight
/// function (SciPy's <c>mu</c>), which the weights sum to.
/// </summary>
public sealed record GaussQuadratureRule(double[] Nodes, double[] Weights, double Mu)
{
    /// <summary>Number of nodes.</summary>
    public int Count => Nodes.Length;

    /// <summary>Applies the rule to <paramref name="f"/>.</summary>
    public double Integrate(Func<double, double> f)
    {
        ArgumentNullException.ThrowIfNull(f);
        var s = 0.0;
        for (var i = 0; i < Nodes.Length; i++) s += Weights[i] * f(Nodes[i]);
        return s;
    }

    /// <summary>Deconstructs into (nodes, weights) like SciPy's two-value return.</summary>
    public void Deconstruct(out double[] nodes, out double[] weights)
    {
        nodes = Nodes;
        weights = Weights;
    }
}

public static partial class Special
{
    // ---------------------------------------------------------------- evaluation

    /// <summary>Legendre polynomial P_n(x) (scipy.special.eval_legendre) by the three-term recurrence.</summary>
    public static double EvalLegendre(int n, double x)
    {
        if (n < 0) n = -n - 1; // P_{-n-1} = P_n
        if (n == 0) return 1.0;
        double p0 = 1.0, p1 = x;
        for (var k = 1; k < n; k++)
        {
            var p2 = ((2 * k + 1) * x * p1 - k * p0) / (k + 1);
            p0 = p1;
            p1 = p2;
        }
        return p1;
    }

    /// <summary>
    /// Legendre function P_v(x) for real degree v (scipy.special.eval_legendre with a non-integer n):
    /// ₂F₁(-v, v + 1; 1; (1 - x)/2). Integer v uses the recurrence.
    /// </summary>
    public static double EvalLegendre(double v, double x)
    {
        if (double.IsNaN(v) || double.IsNaN(x)) return double.NaN;
        if (v == Math.Floor(v) && Math.Abs(v) < int.MaxValue) return EvalLegendre((int)v, x);
        return Hyp2f1(-v, v + 1, 1, 0.5 * (1 - x));
    }

    /// <summary>Chebyshev polynomial of the first kind T_n(x) (scipy.special.eval_chebyt).</summary>
    public static double EvalChebyT(int n, double x)
    {
        n = Math.Abs(n);
        if (n == 0) return 1.0;
        double t0 = 1.0, t1 = x;
        for (var k = 1; k < n; k++)
        {
            var t2 = 2 * x * t1 - t0;
            t0 = t1;
            t1 = t2;
        }
        return t1;
    }

    /// <summary>Chebyshev polynomial of the second kind U_n(x) (scipy.special.eval_chebyu).</summary>
    public static double EvalChebyU(int n, double x)
    {
        if (n == -1) return 0.0;
        if (n < -1) return -EvalChebyU(-n - 2, x);
        if (n == 0) return 1.0;
        double u0 = 1.0, u1 = 2 * x;
        for (var k = 1; k < n; k++)
        {
            var u2 = 2 * x * u1 - u0;
            u0 = u1;
            u1 = u2;
        }
        return u1;
    }

    /// <summary>Physicists' Hermite polynomial H_n(x) (scipy.special.eval_hermite).</summary>
    public static double EvalHermite(int n, double x)
    {
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n), "Hermite degree must be non-negative.");
        if (n == 0) return 1.0;
        double h0 = 1.0, h1 = 2 * x;
        for (var k = 1; k < n; k++)
        {
            var h2 = 2 * x * h1 - 2.0 * k * h0;
            h0 = h1;
            h1 = h2;
        }
        return h1;
    }

    /// <summary>Probabilists' Hermite polynomial He_n(x) (scipy.special.eval_hermitenorm).</summary>
    public static double EvalHermiteE(int n, double x)
    {
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n), "Hermite degree must be non-negative.");
        if (n == 0) return 1.0;
        double h0 = 1.0, h1 = x;
        for (var k = 1; k < n; k++)
        {
            var h2 = x * h1 - k * h0;
            h0 = h1;
            h1 = h2;
        }
        return h1;
    }

    /// <summary>Laguerre polynomial L_n(x) (scipy.special.eval_laguerre).</summary>
    public static double EvalLaguerre(int n, double x) => EvalGenLaguerre(n, 0.0, x);

    /// <summary>Generalised Laguerre polynomial L_n^(α)(x) (scipy.special.eval_genlaguerre), α &gt; -1.</summary>
    public static double EvalGenLaguerre(int n, double alpha, double x)
    {
        if (n < 0) return 0.0;
        if (alpha <= -1) return double.NaN;
        if (n == 0) return 1.0;
        double l0 = 1.0, l1 = 1 + alpha - x;
        for (var k = 1; k < n; k++)
        {
            var l2 = ((2 * k + 1 + alpha - x) * l1 - (k + alpha) * l0) / (k + 1);
            l0 = l1;
            l1 = l2;
        }
        return l1;
    }

    /// <summary>Gegenbauer (ultraspherical) polynomial C_n^(α)(x) (scipy.special.eval_gegenbauer).</summary>
    /// <remarks>α = 0 gives 2/n T_n(x), the limit SciPy uses; α &lt; -1/2 is outside the classical family and gives NaN.</remarks>
    public static double EvalGegenbauer(int n, double alpha, double x)
    {
        if (n < 0) return 0.0;
        if (double.IsNaN(alpha) || double.IsNaN(x) || alpha < -0.5) return double.NaN;
        if (n == 0) return 1.0;
        if (alpha == 0) return 2.0 / n * EvalChebyT(n, x);
        double c0 = 1.0, c1 = 2 * alpha * x;
        for (var k = 1; k < n; k++)
        {
            var c2 = (2 * x * (k + alpha) * c1 - (k + 2 * alpha - 1) * c0) / (k + 1);
            c0 = c1;
            c1 = c2;
        }
        return c1;
    }

    /// <summary>Jacobi polynomial P_n^(α,β)(x) (scipy.special.eval_jacobi).</summary>
    /// <remarks>
    /// Three-term recurrence (DLMF 18.9.2), whose first step is written out so α + β = 0 is no
    /// special case; only when a later denominator vanishes (α + β a negative integer) is the
    /// terminating hypergeometric sum used, which cancels badly for x near -1.
    /// </remarks>
    public static double EvalJacobi(int n, double alpha, double beta, double x)
    {
        if (n < 0) return 0.0;
        if (n == 0) return 1.0;
        var ab = alpha + beta;
        var degenerate = false;
        for (var k = 1; k < n; k++) if (2 * k + ab == 0 || k + ab + 1 == 0) { degenerate = true; break; }
        if (degenerate)
        {
            // binom(n + α, n) Σ (-n)_k (n + α + β + 1)_k / ((α + 1)_k k!) ((1 - x)/2)^k
            var z = 0.5 * (1 - x);
            double term = 1, sum = 1;
            for (var k = 0; k < n; k++)
            {
                term *= (k - n) * (n + ab + 1 + k) / ((alpha + 1 + k) * (k + 1)) * z;
                sum += term;
            }
            return Binom(n + alpha, n) * sum;
        }
        double p0 = 1.0, p1 = (alpha + 1) + (ab + 2) * (x - 1) / 2;
        for (var k = 1; k < n; k++)
        {
            var c = 2 * k + ab;
            var a1 = 2.0 * (k + 1) * (k + ab + 1) * c;
            var a2 = (c + 1) * (alpha * alpha - beta * beta);
            var a3 = c * (c + 1) * (c + 2);
            var a4 = 2.0 * (k + alpha) * (k + beta) * (c + 2);
            var p2 = ((a2 + a3 * x) * p1 - a4 * p0) / a1;
            p0 = p1;
            p1 = p2;
        }
        return p1;
    }

    /// <summary>
    /// Associated Legendre function P_v^m(x) of integer order m and real degree v, for |x| ≤ 1,
    /// including the Condon–Shortley phase (-1)^m (scipy.special.lpmv).
    /// </summary>
    /// <remarks>
    /// Integer degree: the stable upward recurrence in the degree from P_m^m. Real degree: the
    /// hypergeometric representation (DLMF 14.3.1 with the phase), whose series converges
    /// slowly as x → -1. Negative m uses DLMF 14.9.3.
    /// </remarks>
    public static double Lpmv(double m, double v, double x)
    {
        if (double.IsNaN(m) || double.IsNaN(v) || double.IsNaN(x)) return double.NaN;
        if (m != Math.Floor(m)) return double.NaN;
        if (Math.Abs(x) > 1) return double.NaN;
        var mi = (int)m;
        if (v < 0) v = -v - 1;
        if (mi < 0)
        {
            var mm = -mi;
            var pos = Lpmv(mm, v, x);
            var sgn = (mm % 2 == 0) ? 1.0 : -1.0;
            if (v == Math.Floor(v) && mm > v) return double.NaN; // Γ(v - m + 1) at a pole, as in SciPy
            return sgn * Math.Exp(GammaLn(v - mm + 1) - GammaLn(v + mm + 1)) * GammaSgn(v - mm + 1) * pos;
        }
        if (v == Math.Floor(v))
        {
            var n = (int)v;
            if (mi > n) return 0.0;
            // P_m^m = (-1)^m (2m-1)!! (1 - x^2)^(m/2)
            var pmm = 1.0;
            var s = Math.Sqrt((1 - x) * (1 + x));
            for (var k = 1; k <= mi; k++) pmm *= -(2 * k - 1) * s;
            if (n == mi) return pmm;
            var pm1 = x * (2 * mi + 1) * pmm;
            for (var l = mi + 2; l <= n; l++)
            {
                var pl = ((2 * l - 1) * x * pm1 - (l + mi - 1) * pmm) / (l - mi);
                pmm = pm1;
                pm1 = pl;
            }
            return pm1;
        }
        if (v > 2)
        {
            // The ₂F₁ series cancels as the degree grows (1e-11 by v = 10), so start from the
            // fractional degree and its successor and recur upward, which is stable for |x| ≤ 1:
            // (ν - m + 1) P_(ν+1) = (2ν + 1) x P_ν - (ν + m) P_(ν-1).
            var d = v - Math.Floor(v);
            var p0 = Lpmv(mi, d, x);
            var p1 = Lpmv(mi, d + 1, x);
            for (var nu = d + 1; nu < v - 0.5; nu += 1.0)
            {
                var p2 = ((2 * nu + 1) * x * p1 - (nu + mi) * p0) / (nu - mi + 1);
                p0 = p1;
                p1 = p2;
            }
            return p1;
        }
        // Real degree: (-1)^m Γ(v+m+1)/(2^m m! Γ(v-m+1)) (1-x²)^(m/2) ₂F₁(m - v, m + v + 1; m + 1; (1 - x)/2)
        var lg = GammaLn(v + mi + 1) - GammaLn(v - mi + 1) - mi * Math.Log(2.0) - GammaLn(mi + 1.0);
        var sign = ((mi % 2 == 0) ? 1.0 : -1.0) * GammaSgn(v + mi + 1) * GammaSgn(v - mi + 1);
        var f = Hyp2f1(mi - v, mi + v + 1, mi + 1, 0.5 * (1 - x));
        return sign * Math.Exp(lg) * Math.Pow((1 - x) * (1 + x), 0.5 * mi) * f;
    }

    /// <summary>
    /// Spherical harmonic Y_n^m(θ, φ) with θ the polar (colatitude) angle and φ the azimuth,
    /// following scipy.special.sph_harm_y(n, m, theta, phi) — the SciPy 1.15+ function. Note that
    /// the deprecated sph_harm(m, n, θ, φ) took the order first and used θ for the azimuth.
    /// </summary>
    /// <remarks>
    /// Orthonormal with the Condon–Shortley phase. The normalised associated Legendre function is
    /// built by its own stable recurrence, so no factorial ratio is formed and degrees in the
    /// thousands neither overflow nor lose accuracy. |m| &gt; n gives zero.
    /// </remarks>
    public static Complex SphHarmY(int n, int m, double theta, double phi)
    {
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n), "Degree must be non-negative.");
        var am = Math.Abs(m);
        if (am > n) return Complex.Zero;
        var x = Math.Cos(theta);
        var s = Math.Sin(theta);
        // P̄_m^m = (-1)^m sqrt((2m+1)/(4π) (2m-1)!!/(2m)!!) sin^m θ
        var p = Math.Sqrt(1.0 / (4 * Math.PI));
        for (var k = 1; k <= am; k++) p *= -Math.Sqrt((2.0 * k + 1) / (2.0 * k)) * s;
        double pl = p;
        if (n > am)
        {
            var pm1 = Math.Sqrt(2.0 * am + 3) * x * p;
            var pm2 = p;
            for (var l = am + 2; l <= n; l++)
            {
                var a = Math.Sqrt((4.0 * l * l - 1) / ((double)l * l - (double)am * am));
                var b = Math.Sqrt(((l - 1.0) * (l - 1.0) - (double)am * am) / (4.0 * (l - 1.0) * (l - 1.0) - 1));
                var pn = a * (x * pm1 - b * pm2);
                pm2 = pm1;
                pm1 = pn;
            }
            pl = pm1;
        }
        var y = pl * Complex.Exp(new Complex(0, am * phi));
        if (m < 0)
        {
            y = Complex.Conjugate(y);
            if (am % 2 == 1) y = -y;
        }
        return y;
    }

    // ---------------------------------------------------------------- quadrature

    /// <summary>Gauss–Legendre rule on [-1, 1], weight 1 (scipy.special.roots_legendre).</summary>
    /// <remarks>
    /// All Roots* functions: Golub–Welsch eigenvalues of the Jacobi matrix by implicit QL, each node
    /// then polished by Newton's method on the orthonormal three-term recurrence (which also gives
    /// the weight as 1/Σ p̂_k², rescaled to avoid overflow). This is O(n²): n = 1000 Legendre takes
    /// a few milliseconds and agrees with SciPy to ~1e-15. Symmetric families are symmetrised.
    /// </remarks>
    public static GaussQuadratureRule RootsLegendre(int n)
        => GolubWelsch(n, _ => 0.0, k => k * (double)k / (4.0 * k * k - 1), 2.0, true);

    /// <summary>Gauss–Chebyshev (first kind) rule, weight 1/√(1 - x²) (scipy.special.roots_chebyt). Closed form.</summary>
    public static GaussQuadratureRule RootsChebyT(int n)
    {
        CheckOrder(n);
        var x = new double[n];
        var w = new double[n];
        for (var i = 0; i < n; i++)
        {
            x[i] = -CosPi((2.0 * i + 1) / (2.0 * n));
            w[i] = Math.PI / n;
        }
        return new GaussQuadratureRule(x, w, Math.PI);
    }

    /// <summary>Gauss–Chebyshev (second kind) rule, weight √(1 - x²) (scipy.special.roots_chebyu). Closed form.</summary>
    public static GaussQuadratureRule RootsChebyU(int n)
    {
        CheckOrder(n);
        var x = new double[n];
        var w = new double[n];
        for (var i = 0; i < n; i++)
        {
            var t = (i + 1.0) / (n + 1);
            x[i] = -CosPi(t);
            var s = SinPi(t);
            w[i] = Math.PI / (n + 1) * s * s;
        }
        return new GaussQuadratureRule(x, w, Math.PI / 2);
    }

    /// <summary>Gauss–Hermite rule, weight exp(-x²) (scipy.special.roots_hermite).</summary>
    public static GaussQuadratureRule RootsHermite(int n)
        => GolubWelsch(n, _ => 0.0, k => k / 2.0, SqrtPi, true);

    /// <summary>Gauss–Hermite rule for the probabilists' weight exp(-x²/2) (scipy.special.roots_hermitenorm).</summary>
    public static GaussQuadratureRule RootsHermiteE(int n)
        => GolubWelsch(n, _ => 0.0, k => k, Math.Sqrt(2 * Math.PI), true);

    /// <summary>Gauss–Laguerre rule on [0, ∞), weight exp(-x) (scipy.special.roots_laguerre).</summary>
    public static GaussQuadratureRule RootsLaguerre(int n) => RootsGenLaguerre(n, 0.0);

    /// <summary>Generalised Gauss–Laguerre rule, weight x^α exp(-x), α &gt; -1 (scipy.special.roots_genlaguerre).</summary>
    public static GaussQuadratureRule RootsGenLaguerre(int n, double alpha)
    {
        if (!(alpha > -1)) throw new ArgumentOutOfRangeException(nameof(alpha), "alpha must be greater than -1.");
        return GolubWelsch(n, k => 2 * k + alpha + 1, k => k * (k + alpha), Gamma(alpha + 1), false);
    }

    /// <summary>Gauss–Jacobi rule on [-1, 1], weight (1 - x)^α (1 + x)^β, α, β &gt; -1 (scipy.special.roots_jacobi).</summary>
    public static GaussQuadratureRule RootsJacobi(int n, double alpha, double beta)
    {
        if (!(alpha > -1) || !(beta > -1)) throw new ArgumentOutOfRangeException(nameof(alpha), "alpha and beta must be greater than -1.");
        var ab = alpha + beta;
        var mu = Math.Exp((ab + 1) * Math.Log(2.0) + BetaLn(alpha + 1, beta + 1));
        double A(int k)
        {
            if (k == 0) return (beta - alpha) / (ab + 2);
            var c = 2.0 * k + ab;
            return (beta * beta - alpha * alpha) / (c * (c + 2));
        }
        double B(int k)
        {
            var c = 2.0 * k + ab;
            if (k == 1) return 4.0 * (1 + alpha) * (1 + beta) / ((2 + ab) * (2 + ab) * (3 + ab));
            return 4.0 * k * (k + alpha) * (k + beta) * (k + ab) / (c * c * (c + 1) * (c - 1));
        }
        return GolubWelsch(n, A, B, mu, alpha == beta);
    }

    /// <summary>Gauss–Gegenbauer rule, weight (1 - x²)^(α - 1/2), α &gt; -1/2 (scipy.special.roots_gegenbauer).</summary>
    public static GaussQuadratureRule RootsGegenbauer(int n, double alpha)
    {
        if (!(alpha > -0.5)) throw new ArgumentOutOfRangeException(nameof(alpha), "alpha must be greater than -1/2.");
        if (alpha == 0) return RootsChebyT(n);
        return RootsJacobi(n, alpha - 0.5, alpha - 0.5);
    }

    private static void CheckOrder(int n)
    {
        if (n < 1) throw new ArgumentOutOfRangeException(nameof(n), "The number of nodes must be at least 1.");
    }

    /// <summary>Golub–Welsch with Newton polishing; alpha(k), k ≥ 0, beta(k), k ≥ 1, of the monic recurrence.</summary>
    private static GaussQuadratureRule GolubWelsch(int n, Func<int, double> alpha, Func<int, double> beta, double mu0, bool symmetric)
    {
        CheckOrder(n);
        var a = new double[n];
        var b = new double[n + 1]; // b[k] = sqrt(beta(k)), k = 1..n
        for (var k = 0; k < n; k++) a[k] = alpha(k);
        for (var k = 1; k <= n; k++) b[k] = Math.Sqrt(beta(k));
        var d = (double[])a.Clone();
        var e = new double[n];
        for (var i = 0; i < n - 1; i++) e[i] = b[i + 1];
        TridiagonalEigenvalues(d, e);
        Array.Sort(d);
        var w = new double[n];
        var p0 = 1.0 / Math.Sqrt(mu0);
        for (var i = 0; i < n; i++)
        {
            var x = d[i];
            for (var it = 0; it < 10; it++)
            {
                var (f, fp) = OrthonormalValue(x, n, a, b, p0);
                if (fp == 0 || !double.IsFinite(fp)) break;
                var dx = f / fp;
                x -= dx;
                if (Math.Abs(dx) <= 1e-17 * Math.Abs(x)) break;
            }
            d[i] = x;
            w[i] = ChristoffelWeight(x, n, a, b, p0, mu0);
        }
        if (symmetric)
        {
            for (var i = 0; i < n / 2; i++)
            {
                var j = n - 1 - i;
                var xm = 0.5 * (d[j] - d[i]);
                d[i] = -xm;
                d[j] = xm;
                var wm = 0.5 * (w[i] + w[j]);
                w[i] = wm;
                w[j] = wm;
            }
            if (n % 2 == 1) d[n / 2] = 0.0;
        }
        return new GaussQuadratureRule(d, w, mu0);
    }

    /// <summary>Value (up to a positive scale) and derivative of the degree-n orthonormal polynomial.</summary>
    private static (double F, double Fp) OrthonormalValue(double x, int n, double[] a, double[] b, double p0)
    {
        double pm = 0, p = p0, dpm = 0, dp = 0;
        for (var k = 0; k < n; k++)
        {
            // b[k+1] p_{k+1} = (x - a_k) p_k - b_k p_{k-1}
            var pn = ((x - a[k]) * p - b[k] * pm) / b[k + 1];
            var dpn = ((x - a[k]) * dp + p - b[k] * dpm) / b[k + 1];
            pm = p; p = pn; dpm = dp; dp = dpn;
            if (Math.Abs(p) > 1e150)
            {
                p *= 1e-150; pm *= 1e-150; dp *= 1e-150; dpm *= 1e-150;
            }
        }
        return (p, dp);
    }

    /// <summary>Christoffel weight 1/Σ_{k&lt;n} p̂_k(x)², with the sum carried in scaled form.</summary>
    private static double ChristoffelWeight(double x, int n, double[] a, double[] b, double p0, double mu0)
    {
        double pm = 0, p = p0, sum = 1.0 / mu0, logScale = 0; // p̂_0² = 1/μ0 exactly
        for (var k = 0; k < n - 1; k++)
        {
            var pn = ((x - a[k]) * p - b[k] * pm) / b[k + 1];
            pm = p;
            p = pn;
            sum += p * p;
            if (Math.Abs(p) > 1e100)
            {
                p *= 1e-100; pm *= 1e-100; sum *= 1e-200;
                logScale += 200 * Math.Log(10.0);
            }
        }
        return Math.Exp(-logScale) / sum;
    }

    /// <summary>
    /// Eigenvalues of a symmetric tridiagonal matrix by the implicit QL method with Wilkinson
    /// shifts (tqli without eigenvectors). d holds the diagonal, e[i] the element (i, i+1); both
    /// are overwritten, d with the eigenvalues in no particular order.
    /// </summary>
    internal static void TridiagonalEigenvalues(double[] d, double[] e)
    {
        var n = d.Length;
        if (n > 0) e[n - 1] = 0;
        for (var l = 0; l < n; l++)
        {
            var iter = 0;
            int m;
            do
            {
                for (m = l; m < n - 1; m++)
                {
                    var dd = Math.Abs(d[m]) + Math.Abs(d[m + 1]);
                    if (Math.Abs(e[m]) <= MachEp * dd) break;
                }
                if (m != l)
                {
                    if (iter++ == 60) throw new InvalidOperationException("Tridiagonal QL did not converge.");
                    var g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                    var r = Hypot(g, 1.0);
                    g = d[m] - d[l] + e[l] / (g + (g >= 0 ? Math.Abs(r) : -Math.Abs(r)));
                    double s = 1, c = 1, p = 0;
                    int i;
                    var deflated = false;
                    for (i = m - 1; i >= l; i--)
                    {
                        var f = s * e[i];
                        var bb = c * e[i];
                        r = Hypot(f, g);
                        e[i + 1] = r;
                        if (r == 0)
                        {
                            d[i + 1] -= p;
                            e[m] = 0;
                            deflated = true;
                            break;
                        }
                        s = f / r;
                        c = g / r;
                        g = d[i + 1] - p;
                        r = (d[i] - g) * s + 2.0 * c * bb;
                        p = s * r;
                        d[i + 1] = g + p;
                        g = c * r - bb;
                    }
                    if (deflated) continue;
                    d[l] -= p;
                    e[l] = g;
                    e[m] = 0;
                }
            } while (m != l);
        }
    }

    private static double Hypot(double a, double b)
    {
        a = Math.Abs(a);
        b = Math.Abs(b);
        if (a < b) (a, b) = (b, a);
        if (a == 0) return 0;
        var r = b / a;
        return a * Math.Sqrt(1 + r * r);
    }
}
