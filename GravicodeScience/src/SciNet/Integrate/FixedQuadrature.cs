using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Result of <see cref="Integrate.Romberg"/>.</summary>
public sealed record RombergResult(double Value, double AbsErr, int Neval, bool Converged);

public static partial class Integrate
{
    /// <summary>scipy.integrate.fixed_quad: n-point Gauss-Legendre on [a, b].</summary>
    public static double FixedQuad(Func<double, double> f, double a, double b, int n = 5)
    {
        ArgumentNullException.ThrowIfNull(f);
        if (double.IsInfinity(a) || double.IsInfinity(b)) throw new ArgumentException("Gaussian quadrature is only available for finite limits.");
        var (x, w) = IntegrateRules.GaussLegendre(n);
        var c = 0.5 * (a + b);
        var h = 0.5 * (b - a);
        var s = 0.0;
        for (var i = 0; i < n; i++) s += w[i] * f(c + h * x[i]);
        return h * s;
    }

    /// <summary>
    /// Romberg integration of a callable, the algorithm of scipy's (removed in 1.15) <c>romberg</c>:
    /// trapezoid sums with 2^i panels and Richardson extrapolation, stopping when two successive
    /// diagonal entries agree to <paramref name="tol"/> or <paramref name="rtol"/>.
    /// </summary>
    public static RombergResult Romberg(Func<double, double> f, double a, double b, double tol = 1.48e-8, double rtol = 1.48e-8, int divmax = 10)
    {
        ArgumentNullException.ThrowIfNull(f);
        if (double.IsInfinity(a) || double.IsInfinity(b)) throw new ArgumentException("Romberg integration only available for finite limits.");
        var n = 1;
        var range = b - a;
        var ordsum = 0.5 * (f(a) + f(b));
        var neval = 2;
        var result = range * ordsum;
        var last = new List<double> { result };
        var err = double.PositiveInfinity;
        for (var i = 1; i <= divmax; i++)
        {
            n *= 2;
            // Sum f at the new midpoints of the n/2 previous panels.
            var numtosum = n / 2;
            var h = range / numtosum;
            var lox = a + 0.5 * h;
            var s = 0.0;
            for (var k = 0; k < numtosum; k++) s += f(lox + h * k);
            neval += numtosum;
            ordsum += s;
            var row = new List<double> { range * ordsum / n };
            for (var k = 0; k < i; k++)
            {
                var t = Math.Pow(4.0, k + 1);
                row.Add((t * row[k] - last[k]) / (t - 1.0));
            }
            result = row[i];
            var lastresult = last[i - 1];
            err = Math.Abs(result - lastresult);
            if (err < tol || err < rtol * Math.Abs(result)) return new RombergResult(result, err, neval, true);
            last = row;
        }
        return new RombergResult(result, err, neval, false);
    }

    /// <summary>scipy.integrate.romb: Romberg on 2^k + 1 equally spaced samples.</summary>
    public static double Romb(ReadOnlySpan<double> y, double dx = 1.0)
    {
        var nsamps = y.Length;
        var ninterv = nsamps - 1;
        var n = 1;
        var k = 0;
        while (n < ninterv) { n <<= 1; k++; }
        if (n != ninterv) throw new ArgumentException("Number of samples must be one plus a non-negative power of 2.");
        var r = new double[k + 1, k + 1];
        var h = ninterv * dx;
        r[0, 0] = (y[0] + y[nsamps - 1]) / 2.0 * h;
        int start = ninterv, step = ninterv;
        for (var i = 1; i <= k; i++)
        {
            start >>= 1;
            var s = 0.0;
            for (var j = start; j < ninterv; j += step) s += y[j];
            step >>= 1;
            r[i, 0] = 0.5 * (r[i - 1, 0] + h * s);
            for (var j = 1; j <= i; j++)
            {
                var prev = r[i, j - 1];
                r[i, j] = prev + (prev - r[i - 1, j - 1]) / ((1 << (2 * j)) - 1);
            }
            h /= 2.0;
        }
        return r[k, k];
    }

    /// <summary>scipy.integrate.trapezoid for samples at spacing dx.</summary>
    public static double Trapezoid(ReadOnlySpan<double> y, double dx = 1.0)
    {
        if (y.Length < 2) return 0.0;
        var s = 0.5 * (y[0] + y[^1]);
        for (var i = 1; i < y.Length - 1; i++) s += y[i];
        return s * dx;
    }

    /// <summary>scipy.integrate.trapezoid for samples at (possibly uneven) abscissae x.</summary>
    public static double Trapezoid(ReadOnlySpan<double> y, ReadOnlySpan<double> x)
    {
        if (x.Length != y.Length) throw new ArgumentException("x and y must have the same length.");
        var s = 0.0;
        for (var i = 1; i < y.Length; i++) s += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) * 0.5;
        return s;
    }

    /// <summary>Trapezoid along an axis of an n-dimensional array (x optional, 1-D of the axis length).</summary>
    public static NdArray Trapezoid(NdArray y, double[]? x = null, double dx = 1.0, int axis = -1)
        => AlongAxis(y, axis, (line, dst) => dst[0] = x is null ? Trapezoid(line, dx) : Trapezoid(line, x), _ => 1, collapse: true);

    /// <summary>
    /// scipy.integrate.simpson (scipy ≥ 1.11 semantics). With an odd number of samples this is
    /// composite Simpson (the uneven-spacing form when x is given). With an even number the
    /// first N-1 samples get Simpson and the last interval Cartwright's correction, which is
    /// exact for quadratics; N == 2 falls back to the trapezoid.
    /// </summary>
    public static double Simpson(ReadOnlySpan<double> y, ReadOnlySpan<double> x = default, double dx = 1.0)
    {
        var n = y.Length;
        var hasX = !x.IsEmpty;
        if (hasX && x.Length != n) throw new ArgumentException("x and y must have the same length.");
        if (n == 0) return 0.0;
        if (n == 1) return 0.0;
        if (n % 2 == 1) return BasicSimpson(y, 0, n - 2, x, dx);
        if (n == 2)
        {
            var lastDx = hasX ? x[1] - x[0] : dx;
            return 0.5 * lastDx * (y[1] + y[0]);
        }
        var result = BasicSimpson(y, 0, n - 3, x, dx);
        double h0 = dx, h1 = dx;
        if (hasX) { h0 = x[n - 2] - x[n - 3]; h1 = x[n - 1] - x[n - 2]; }
        var den = 6 * (h1 + h0);
        var alpha = den != 0 ? (2 * h1 * h1 + 3 * h0 * h1) / den : 0;
        den = 6 * h0;
        var beta = den != 0 ? (h1 * h1 + 3.0 * h0 * h1) / den : 0;
        den = 6 * h0 * (h0 + h1);
        var eta = den != 0 ? h1 * h1 * h1 / den : 0;
        return result + alpha * y[n - 1] + beta * y[n - 2] - eta * y[n - 3];
    }

    /// <summary>Simpson along an axis of an n-dimensional array.</summary>
    public static NdArray Simpson(NdArray y, double[]? x = null, double dx = 1.0, int axis = -1)
        => AlongAxis(y, axis, (line, dst) => dst[0] = Simpson(line, x ?? default, dx), _ => 1, collapse: true);

    private static double BasicSimpson(ReadOnlySpan<double> y, int start, int stop, ReadOnlySpan<double> x, double dx)
    {
        // Triples (i, i+1, i+2) for i = start, start+2, ... < stop.
        var s = 0.0;
        if (x.IsEmpty)
        {
            for (var i = start; i < stop; i += 2) s += y[i] + 4 * y[i + 1] + y[i + 2];
            return s * dx / 3.0;
        }
        for (var i = start; i < stop; i += 2)
        {
            var h0 = x[i + 1] - x[i];
            var h1 = x[i + 2] - x[i + 1];
            var hsum = h0 + h1;
            var hprod = h0 * h1;
            var h0divh1 = h1 != 0 ? h0 / h1 : 0;
            var inv = h0divh1 != 0 ? 1.0 / h0divh1 : 0;
            var mid = hprod != 0 ? hsum / hprod : 0;
            s += hsum / 6.0 * (y[i] * (2.0 - inv) + y[i + 1] * (hsum * mid) + y[i + 2] * (2.0 - h0divh1));
        }
        return s;
    }

    /// <summary>
    /// scipy.integrate.cumulative_trapezoid. Length N-1, or N with <paramref name="initial"/>
    /// prepended (scipy 1.12+ accepts only 0 there; so does this).
    /// </summary>
    public static double[] CumulativeTrapezoid(ReadOnlySpan<double> y, ReadOnlySpan<double> x = default, double dx = 1.0, double? initial = null)
    {
        if (initial is double v && v != 0) throw new ArgumentException("`initial` must be 0 or null (scipy >= 1.12).", nameof(initial));
        var n = y.Length;
        var hasX = !x.IsEmpty;
        if (hasX && x.Length != n) throw new ArgumentException("x and y must have the same length.");
        var off = initial.HasValue ? 1 : 0;
        var res = new double[Math.Max(n - 1, 0) + off];
        var s = 0.0;
        for (var i = 1; i < n; i++)
        {
            var d = hasX ? x[i] - x[i - 1] : dx;
            s += d * (y[i] + y[i - 1]) / 2.0;
            res[i - 1 + off] = s;
        }
        return res;
    }

    /// <summary>
    /// scipy.integrate.cumulative_simpson (scipy 1.12+). Each panel integral comes from the
    /// quadratic through three neighbouring samples; panel i uses the triple starting at i when i
    /// is even and the triple ending at i+1 otherwise (and the last panel always the latter), so the
    /// running sum at every odd sample agrees with composite Simpson. x must be strictly increasing.
    /// Unlike cumulative_trapezoid, any <paramref name="initial"/> is accepted and added to every value.
    /// </summary>
    public static double[] CumulativeSimpson(ReadOnlySpan<double> y, ReadOnlySpan<double> x = default, double dx = 1.0, double? initial = null)
    {
        var n = y.Length;
        var hasX = !x.IsEmpty;
        if (hasX && x.Length != n) throw new ArgumentException("x and y must have the same length.");
        double[] res;
        if (n < 3)
        {
            res = CumulativeTrapezoid(y, x, dx);
        }
        else
        {
            if (hasX)
                for (var i = 1; i < n; i++)
                    if (!(x[i] - x[i - 1] > 0)) throw new ArgumentException("Input x must be strictly increasing.");
            var sub = new double[n - 1];
            var ya = y.ToArray();
            var xa = hasX ? x.ToArray() : null;
            double H(int i) => xa is not null ? xa[i + 1] - xa[i] : dx;
            // Integral over [x_i, x_{i+1}] using the triple (i, i+1, i+2) (first panel of it).
            double First(int i) => Panel(ya[i], ya[i + 1], ya[i + 2], H(i), H(i + 1));
            // Integral over [x_{i+1}, x_{i+2}] using the same triple, i.e. the reversed formula.
            double Second(int i) => Panel(ya[i + 2], ya[i + 1], ya[i], H(i + 1), H(i));
            for (var i = 0; i < n - 1; i += 2) if (i <= n - 3) sub[i] = First(i);
            for (var i = 1; i < n - 1; i += 2) sub[i] = Second(i - 1);
            sub[n - 2] = Second(n - 3);
            res = new double[n - 1];
            var s = 0.0;
            for (var i = 0; i < n - 1; i++) { s += sub[i]; res[i] = s; }
        }
        if (initial is double v)
        {
            var o = new double[res.Length + 1];
            o[0] = v;
            for (var i = 0; i < res.Length; i++) o[i + 1] = res[i] + v;
            return o;
        }
        return res;

        static double Panel(double f1, double f2, double f3, double x21, double x32)
        {
            if (x21 == x32)
                return x21 / 3 * (5 * f1 / 4 + 2 * f2 - f3 / 4);
            var x31 = x21 + x32;
            var x21x31 = x21 / x31;
            var x21x32 = x21 / x32;
            var q = x21x31 * x21x32;
            return x21 / 6 * ((3 - x21x31) * f1 + (3 + q + x21x31) * f2 - q * f3);
        }
    }

    /// <summary>
    /// scipy.integrate.newton_cotes for N + 1 equally spaced points: (weights, error coefficient),
    /// where the rule is dx * Σ w_i f(x_i) and the error is B (dx)^(p+1) f^(p)(ξ), with
    /// p = N + 2 for even N and N + 1 for odd N. Computed in exact rational arithmetic, so these are
    /// scipy's tabulated fractions for N ≤ 14 and exact (rather than Vandermonde-inverted) beyond.
    /// </summary>
    public static (double[] Weights, double ErrorCoefficient) NewtonCotes(int n)
    {
        if (n < 1) throw new ArgumentException("N must be at least 1.", nameof(n));
        var rn = new double[n + 1];
        for (var i = 0; i <= n; i++) rn[i] = i;
        return NewtonCotesCore(rn, n, true);
    }

    /// <summary>
    /// newton_cotes for sample positions <paramref name="rn"/> (must start at 0 and end at N = len-1).
    /// Integer equally spaced positions (or <paramref name="equal"/>) use the exact rational path;
    /// otherwise the weights come from the moment equations solved in double precision.
    /// </summary>
    public static (double[] Weights, double ErrorCoefficient) NewtonCotes(double[] rn, bool equal = false)
    {
        ArgumentNullException.ThrowIfNull(rn);
        var n = rn.Length - 1;
        if (n < 1) throw new ArgumentException("At least two sample positions are required.", nameof(rn));
        var isEqual = equal;
        if (!isEqual)
        {
            isEqual = true;
            for (var i = 1; i <= n; i++) if (rn[i] - rn[i - 1] != 1) { isEqual = false; break; }
        }
        if (isEqual) return NewtonCotes(n);
        if (rn[0] != 0 || rn[n] != n) throw new ArgumentException("The sample positions must start at 0 and end at N.");
        return NewtonCotesCore(rn, n, false);
    }

    private static (double[] Weights, double ErrorCoefficient) NewtonCotesCore(double[] rn, int n, bool equal)
    {
        var p = n % 2 == 0 && equal ? n + 2 : n + 1;
        if (equal)
        {
            // Lagrange basis integrals ∫_0^N l_i(t) dt in rationals: the rule is dx * Σ w_i f_i.
            var w = new IntegrateRational[n + 1];
            for (var i = 0; i <= n; i++)
            {
                // Polynomial Π_{j≠i} (t - j) with rational coefficients, ascending powers.
                var poly = new IntegrateRational[n + 1];
                poly[0] = IntegrateRational.One;
                for (var k = 1; k <= n; k++) poly[k] = IntegrateRational.Zero;
                var deg = 0;
                BigInteger denom = BigInteger.One;
                for (var j = 0; j <= n; j++)
                {
                    if (j == i) continue;
                    for (var k = deg + 1; k >= 1; k--) poly[k] = poly[k - 1] - poly[k] * new IntegrateRational(j);
                    poly[0] = -poly[0] * new IntegrateRational(j);
                    deg++;
                    denom *= i - j;
                }
                var integral = IntegrateRational.Zero;
                BigInteger nPow = n;
                for (var k = 0; k <= n; k++)
                {
                    integral += poly[k] * new IntegrateRational(nPow, k + 1);
                    nPow *= n;
                }
                w[i] = integral / new IntegrateRational(denom);
            }
            // B = (N^(p+1)/(p+1) - Σ w_i i^p) / p!
            var exact = new IntegrateRational(BigInteger.Pow(n, p + 1), p + 1);
            var sum = IntegrateRational.Zero;
            for (var i = 0; i <= n; i++) sum += w[i] * new IntegrateRational(BigInteger.Pow(i, p));
            var fact = BigInteger.One;
            for (var k = 2; k <= p; k++) fact *= k;
            var bErr = (exact - sum) / new IntegrateRational(fact);
            return (w.Select(r => r.ToDouble()).ToArray(), bErr.ToDouble());
        }
        else
        {
            // Moment equations Σ a_i t_i^k = ∫_{-1}^{1} t^k dt on ti = 2 yi - 1, then scale by N/2,
            // as scipy does; we solve with LU plus one step of iterative refinement.
            var m = n + 1;
            var yi = rn.Select(r => r / n).ToArray();
            var ti = yi.Select(v => 2 * v - 1).ToArray();
            var c = new double[m * m];
            for (var k = 0; k < m; k++)
                for (var i = 0; i < m; i++) c[k * m + i] = Math.Pow(ti[i], k);
            var rhs = new double[m];
            for (var k = 0; k < m; k++) rhs[k] = k % 2 == 0 ? 2.0 / (k + 1) : 0.0;
            var a = IntegrateLinear.SolveDense(c, rhs, m);
            var r = new double[m];
            for (var k = 0; k < m; k++)
            {
                var s = rhs[k];
                for (var i = 0; i < m; i++) s -= c[k * m + i] * a[i];
                r[k] = s;
            }
            var corr = IntegrateLinear.SolveDense(c, r, m);
            for (var i = 0; i < m; i++) a[i] = (a[i] + corr[i]) * (n / 2.0);
            var bn = n / (p + 1.0);
            for (var i = 0; i < m; i++) bn -= Math.Pow(yi[i], p) * a[i];
            var lg = p * Math.Log(n) - LogGamma(p + 1);
            return (a, bn * Math.Exp(lg));
        }
    }

    private static double LogGamma(int n)
    {
        var s = 0.0;
        for (var k = 2; k < n; k++) s += Math.Log(k);
        return s;
    }

    /// <summary>Applies a 1-D kernel along one axis of an n-d array (copying lines out, results back).</summary>
    private static NdArray AlongAxis(NdArray y, int axis, Action<double[], double[]> kernel, Func<int, int> outLen, bool collapse)
    {
        ArgumentNullException.ThrowIfNull(y);
        var rank = y.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentException("axis out of range.", nameof(axis));
        var shape = y.Shape.ToArray();
        var len = shape[axis];
        var olen = outLen(len);
        var src = y.AsContiguous().AsSpan().ToArray();
        int outer = 1, inner = 1;
        for (var i = 0; i < axis; i++) outer *= shape[i];
        for (var i = axis + 1; i < rank; i++) inner *= shape[i];
        var dst = new double[outer * olen * inner];
        var line = new double[len];
        var res = new double[olen];
        for (var o = 0; o < outer; o++)
            for (var i = 0; i < inner; i++)
            {
                for (var k = 0; k < len; k++) line[k] = src[(o * len + k) * inner + i];
                kernel(line, res);
                for (var k = 0; k < olen; k++) dst[(o * olen + k) * inner + i] = res[k];
            }
        var newShape = shape.ToList();
        if (collapse && olen == 1) newShape.RemoveAt(axis); else newShape[axis] = olen;
        if (newShape.Count == 0) newShape.Add(1);
        return new NdArray(dst, newShape.ToArray());
    }
}

/// <summary>Minimal exact rational on BigInteger, for Newton-Cotes weights.</summary>
internal readonly struct IntegrateRational
{
    internal readonly BigInteger Num, Den;
    internal static readonly IntegrateRational Zero = new(0);
    internal static readonly IntegrateRational One = new(1);

    internal IntegrateRational(BigInteger num, BigInteger? den = null)
    {
        var d = den ?? BigInteger.One;
        if (d.IsZero) throw new DivideByZeroException();
        if (d.Sign < 0) { num = -num; d = -d; }
        var g = BigInteger.GreatestCommonDivisor(num, d);
        if (!g.IsZero && !g.IsOne) { num /= g; d /= g; }
        Num = num; Den = d;
    }

    public static IntegrateRational operator +(IntegrateRational a, IntegrateRational b) => new(a.Num * b.Den + b.Num * a.Den, a.Den * b.Den);
    public static IntegrateRational operator -(IntegrateRational a, IntegrateRational b) => new(a.Num * b.Den - b.Num * a.Den, a.Den * b.Den);
    public static IntegrateRational operator -(IntegrateRational a) => new(-a.Num, a.Den);
    public static IntegrateRational operator *(IntegrateRational a, IntegrateRational b) => new(a.Num * b.Num, a.Den * b.Den);
    public static IntegrateRational operator /(IntegrateRational a, IntegrateRational b) => new(a.Num * b.Den, a.Den * b.Num);

    internal double ToDouble()
    {
        // Scale so the quotient keeps 64+ significant bits before converting.
        if (Num.IsZero) return 0.0;
        // Both exactly representable: one IEEE division is correctly rounded.
        if (BigInteger.Abs(Num).GetBitLength() <= 53 && Den.GetBitLength() <= 53) return (double)Num / (double)Den;
        var shift = (int)(Den.GetBitLength() - BigInteger.Abs(Num).GetBitLength()) + 70;
        var q = shift >= 0 ? (Num << shift) / Den : Num / (Den << -shift);
        return (double)q * Math.Pow(2, -shift);
    }
}
