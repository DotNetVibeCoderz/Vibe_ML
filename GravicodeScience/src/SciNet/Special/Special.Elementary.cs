namespace Gravicode.Science.SciNet;

public static partial class Special
{
    /// <summary>log(1 + x), accurate for small x (scipy.special.log1p).</summary>
    /// <remarks>
    /// Kahan's trick: with u = 1 + x rounded, log(u)·x/(u - 1) cancels the rounding of u. .NET's
    /// generic <c>double.LogP1</c> is the naive log(x + 1) and loses every digit below 1e-16.
    /// </remarks>
    public static double Log1p(double x)
    {
        if (double.IsNaN(x)) return x;
        var u = 1.0 + x;
        if (u == 1.0) return x;
        if (double.IsPositiveInfinity(u)) return u;
        return Math.Log(u) * (x / (u - 1.0));
    }

    /// <summary>exp(x) - 1, accurate for small x (scipy.special.expm1).</summary>
    /// <remarks>Kahan's trick (u - 1)·x/log(u) for |x| &lt; 1; the cancellation-free series for |x| &lt; 1e-5.</remarks>
    public static double Expm1(double x)
    {
        if (double.IsNaN(x)) return x;
        if (Math.Abs(x) < 1e-5) return x + 0.5 * x * x + x * x * x / 6.0;
        if (Math.Abs(x) >= 1.0) return Math.Exp(x) - 1.0;
        var u = Math.Exp(x);
        return (u - 1.0) * x / Math.Log(u);
    }

    /// <summary>cos(x) - 1 without cancellation near zero (scipy.special.cosm1), as -2 sin²(x/2).</summary>
    public static double Cosm1(double x)
    {
        if (Math.Abs(x) > Math.PI / 4) return Math.Cos(x) - 1.0;
        var s = Math.Sin(0.5 * x);
        return -2.0 * s * s;
    }

    /// <summary>Real cube root (scipy.special.cbrt).</summary>
    /// <remarks>
    /// <c>Math.Cbrt</c> is the C library's, and neither common one is correctly rounded: on Linux
    /// <c>Math.Cbrt(-27)</c> is -3.0000000000000004, and on Windows <c>Math.Cbrt(1.5237e192)</c> is
    /// 2–3 ulps off. One Newton step on y³ = x, with the residual formed through FMA, makes perfect
    /// cubes exact and gives the same, correctly rounded answer on both.
    /// </remarks>
    public static double Cbrt(double x)
    {
        var y = Math.Cbrt(x);
        if (y == 0 || !double.IsFinite(y)) return y;
        var y2 = y * y;
        var y2Error = Math.FusedMultiplyAdd(y, y, -y2);              // y² = y2 + y2Error exactly
        var residual = Math.FusedMultiplyAdd(y2, y, -x) + y2Error * y; // y³ - x
        return y - residual / (3.0 * y2);
    }

    /// <summary>2^x (scipy.special.exp2).</summary>
    public static double Exp2(double x) => Math.Pow(2.0, x);

    /// <summary>10^x (scipy.special.exp10).</summary>
    public static double Exp10(double x) => Math.Pow(10.0, x);

    /// <summary>(eˣ - 1)/x, 1 at x = 0 (scipy.special.exprel).</summary>
    public static double ExpRel(double x)
    {
        if (Math.Abs(x) < 1e-16) return 1.0;
        if (x > 717) return double.PositiveInfinity;
        return Expm1(x) / x;
    }

    /// <summary>The normalised sinc function sin(πx)/(πx), 1 at 0 (numpy.sinc convention).</summary>
    public static double Sinc(double x)
    {
        if (x == 0) return 1.0;
        if (double.IsInfinity(x)) return 0.0;
        return SinPi(x) / (Math.PI * x);
    }

    /// <summary>The logistic sigmoid 1/(1 + e^-x) (scipy.special.expit), without overflow for either sign.</summary>
    public static double Expit(double x)
    {
        if (x >= 0) return 1.0 / (1.0 + Math.Exp(-x));
        var e = Math.Exp(x);
        return e / (1.0 + e);
    }

    /// <summary>log(p/(1 - p)) (scipy.special.logit); near p = 1/2 via log1p((2p - 1)/(1 - p)), where 2p - 1 is exact.</summary>
    public static double Logit(double p)
    {
        if (double.IsNaN(p) || p < 0 || p > 1) return double.NaN;
        if (p == 0) return double.NegativeInfinity;
        if (p == 1) return double.PositiveInfinity;
        if (p > 0.3 && p < 0.7) return Log1p((2 * p - 1) / (1 - p));
        return Math.Log(p) - Log1p(-p);
    }

    /// <summary>log(expit(x)) without underflow for large negative x (scipy.special.log_expit).</summary>
    public static double LogExpit(double x)
    {
        if (x >= 0) return -Log1p(Math.Exp(-x));
        return x - Log1p(Math.Exp(x));
    }

    /// <summary>
    /// log Σ b_i exp(a_i) (scipy.special.logsumexp). The weights default to one; a negative total
    /// gives NaN (use <see cref="LogSumExpWithSign"/> to obtain the sign instead).
    /// </summary>
    public static double LogSumExp(ReadOnlySpan<double> a, ReadOnlySpan<double> b = default)
    {
        var (v, s) = LogSumExpWithSign(a, b);
        return s < 0 ? double.NaN : v;
    }

    /// <summary>log|Σ b_i exp(a_i)| and the sign of the sum (scipy.special.logsumexp with return_sign=True).</summary>
    public static (double Value, double Sign) LogSumExpWithSign(ReadOnlySpan<double> a, ReadOnlySpan<double> b = default)
    {
        if (!b.IsEmpty && b.Length != a.Length) throw new ArgumentException("a and b must have the same length.", nameof(b));
        if (a.IsEmpty) return (double.NegativeInfinity, 0.0);
        var m = double.NegativeInfinity;
        for (var i = 0; i < a.Length; i++)
        {
            if (double.IsNaN(a[i])) return (double.NaN, double.NaN);
            if (!b.IsEmpty && b[i] == 0) continue;
            if (a[i] > m) m = a[i];
        }
        if (double.IsNegativeInfinity(m)) return (double.NegativeInfinity, 0.0);
        if (double.IsPositiveInfinity(m))
        {
            var sg = 0.0;
            for (var i = 0; i < a.Length; i++)
                if (double.IsPositiveInfinity(a[i])) sg += b.IsEmpty ? 1.0 : Math.Sign(b[i]);
            return sg == 0 ? (double.NaN, double.NaN) : (double.PositiveInfinity, Math.Sign(sg));
        }
        var s = 0.0;
        for (var i = 0; i < a.Length; i++)
        {
            var e = Math.Exp(a[i] - m);
            s += b.IsEmpty ? e : b[i] * e;
        }
        if (s == 0) return (double.NegativeInfinity, 0.0);
        return (m + Math.Log(Math.Abs(s)), Math.Sign(s));
    }

    /// <summary>The softmax exp(x_i - logsumexp(x)) (scipy.special.softmax), stable for large inputs.</summary>
    public static double[] Softmax(ReadOnlySpan<double> x)
    {
        var r = new double[x.Length];
        if (x.IsEmpty) return r;
        var m = double.NegativeInfinity;
        foreach (var v in x) if (v > m) m = v;
        var s = 0.0;
        for (var i = 0; i < x.Length; i++) { r[i] = Math.Exp(x[i] - m); s += r[i]; }
        for (var i = 0; i < r.Length; i++) r[i] /= s;
        return r;
    }

    /// <summary>log softmax x_i - logsumexp(x) (scipy.special.log_softmax).</summary>
    public static double[] LogSoftmax(ReadOnlySpan<double> x)
    {
        var r = new double[x.Length];
        if (x.IsEmpty) return r;
        // (x_i - max) - log Σ exp(x_j - max): subtracting the maximum first keeps the result
        // accurate to an ulp of itself; x_i - logsumexp(x) is only accurate to an ulp of x_i.
        var m = double.NegativeInfinity;
        foreach (var v in x) if (v > m) m = v;
        if (double.IsInfinity(m))
        {
            var lse = LogSumExp(x);
            for (var i = 0; i < r.Length; i++) r[i] = x[i] - lse;
            return r;
        }
        var s = 0.0;
        for (var i = 0; i < x.Length; i++) s += Math.Exp(x[i] - m);
        var ls = Math.Log(s);
        for (var i = 0; i < x.Length; i++) r[i] = (x[i] - m) - ls;
        return r;
    }

    /// <summary>x log(y), with 0 when x = 0 and y is not NaN (scipy.special.xlogy).</summary>
    public static double Xlogy(double x, double y)
    {
        if (x == 0 && !double.IsNaN(y)) return 0.0;
        return x * Math.Log(y);
    }

    /// <summary>x log1p(y), with 0 when x = 0 and y is not NaN (scipy.special.xlog1py).</summary>
    public static double Xlog1py(double x, double y)
    {
        if (x == 0 && !double.IsNaN(y)) return 0.0;
        return x * Log1p(y);
    }

    /// <summary>Elementwise entropy -x log x, 0 at 0, -∞ for x &lt; 0 (scipy.special.entr).</summary>
    public static double Entr(double x)
    {
        if (double.IsNaN(x)) return x;
        if (x > 0) return -x * Math.Log(x);
        return x == 0 ? 0.0 : double.NegativeInfinity;
    }

    /// <summary>Relative entropy x log(x/y) (scipy.special.rel_entr).</summary>
    public static double RelEntr(double x, double y)
    {
        if (double.IsNaN(x) || double.IsNaN(y)) return double.NaN;
        if (x > 0 && y > 0) return x * Math.Log(x / y);
        if (x == 0 && y >= 0) return 0.0;
        return double.PositiveInfinity;
    }

    /// <summary>Kullback–Leibler term x log(x/y) - x + y (scipy.special.kl_div).</summary>
    public static double KlDiv(double x, double y)
    {
        if (double.IsNaN(x) || double.IsNaN(y)) return double.NaN;
        if (x > 0 && y > 0) return x * Math.Log(x / y) - x + y;
        if (x == 0 && y >= 0) return y;
        return double.PositiveInfinity;
    }

    /// <summary>Huber loss: r²/2 for |r| ≤ δ, else δ(|r| - δ/2) (scipy.special.huber).</summary>
    public static double Huber(double delta, double r)
    {
        if (double.IsNaN(delta) || double.IsNaN(r)) return double.NaN;
        if (delta < 0) return double.PositiveInfinity;
        var ar = Math.Abs(r);
        return ar <= delta ? 0.5 * r * r : delta * (ar - 0.5 * delta);
    }

    /// <summary>Pseudo-Huber loss δ²(√(1 + (r/δ)²) - 1) (scipy.special.pseudo_huber), via expm1/log1p.</summary>
    public static double PseudoHuber(double delta, double r)
    {
        if (double.IsNaN(delta) || double.IsNaN(r)) return double.NaN;
        if (delta < 0) return double.PositiveInfinity;
        if (delta == 0 || r == 0) return 0.0;
        var u = r / delta;
        return delta * delta * Expm1(0.5 * Log1p(u * u));
    }

    /// <summary>Box–Cox transform (x^λ - 1)/λ, log x at λ = 0 (scipy.special.boxcox).</summary>
    public static double Boxcox(double x, double lmbda)
    {
        if (Math.Abs(lmbda) < 1e-19) return Math.Log(x);
        return Expm1(lmbda * Math.Log(x)) / lmbda;
    }

    /// <summary>Inverse Box–Cox transform (scipy.special.inv_boxcox).</summary>
    public static double InvBoxcox(double y, double lmbda)
    {
        if (lmbda == 0) return Math.Exp(y);
        return Math.Exp(Log1p(lmbda * y) / lmbda);
    }

    /// <summary>Box–Cox transform of 1 + x, accurate for small x (scipy.special.boxcox1p).</summary>
    public static double Boxcox1p(double x, double lmbda)
    {
        var lgx = Log1p(x);
        if (Math.Abs(lmbda) < 1e-19 || (Math.Abs(lgx) < 1e-289 && Math.Abs(lmbda) < 1e273)) return lgx;
        return Expm1(lmbda * lgx) / lmbda;
    }

    /// <summary>Inverse of <see cref="Boxcox1p"/> (scipy.special.inv_boxcox1p).</summary>
    public static double InvBoxcox1p(double y, double lmbda)
    {
        if (lmbda == 0) return Expm1(y);
        return Expm1(Log1p(lmbda * y) / lmbda);
    }
}
