using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Signal
{
    /// <summary>
    /// Coefficients (highest power first) of the monic polynomial with the given roots, as
    /// <c>numpy.poly</c>. The roots must come in conjugate pairs so the result is real.
    /// </summary>
    /// <exception cref="ArgumentException">The roots are not closed under conjugation.</exception>
    public static double[] Poly(ReadOnlySpan<Complex> roots)
    {
        var c = PolyComplex(roots);
        var scale = 0.0;
        foreach (var v in c) scale = Math.Max(scale, Complex.Abs(v));
        var result = new double[c.Length];
        for (var i = 0; i < c.Length; i++)
        {
            if (Math.Abs(c[i].Imaginary) > 1e-8 * Math.Max(scale, 1e-300))
                throw new ArgumentException("Roots are not closed under complex conjugation; the polynomial would be complex. Use PolyComplex.");
            result[i] = c[i].Real;
        }
        return result;
    }

    /// <summary>Coefficients of the polynomial with real roots.</summary>
    public static double[] Poly(ReadOnlySpan<double> roots)
    {
        var a = new double[roots.Length + 1];
        a[0] = 1.0;
        for (var j = 0; j < roots.Length; j++)
        {
            var r = roots[j];
            for (var i = j + 1; i >= 1; i--) a[i] -= r * a[i - 1];
        }
        return a;
    }

    /// <summary>Complex coefficients of the monic polynomial with the given roots.</summary>
    public static Complex[] PolyComplex(ReadOnlySpan<Complex> roots)
    {
        var a = new Complex[roots.Length + 1];
        a[0] = Complex.One;
        for (var j = 0; j < roots.Length; j++)
        {
            var r = roots[j];
            for (var i = j + 1; i >= 1; i--) a[i] -= r * a[i - 1];
        }
        return a;
    }

    /// <summary>Evaluates a real polynomial (highest power first) at a complex point by Horner's rule.</summary>
    public static Complex Polyval(ReadOnlySpan<double> p, Complex x)
    {
        var y = Complex.Zero;
        foreach (var c in p) y = y * x + c;
        return y;
    }

    /// <summary>Evaluates a real polynomial (highest power first) at a real point.</summary>
    public static double Polyval(ReadOnlySpan<double> p, double x)
    {
        var y = 0.0;
        foreach (var c in p) y = y * x + c;
        return y;
    }

    /// <summary>Evaluates a complex polynomial at a complex point.</summary>
    public static Complex Polyval(ReadOnlySpan<Complex> p, Complex x)
    {
        var y = Complex.Zero;
        foreach (var c in p) y = y * x + c;
        return y;
    }

    /// <summary>Product of two polynomials (their coefficient convolution), leading zeros trimmed as <c>numpy.polymul</c>.</summary>
    public static double[] Polymul(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        a = TrimLeadingZeros(a);
        b = TrimLeadingZeros(b);
        if (a.Length == 0) a = [0.0];
        if (b.Length == 0) b = [0.0];
        var r = new double[a.Length + b.Length - 1];
        for (var i = 0; i < a.Length; i++)
            for (var j = 0; j < b.Length; j++)
                r[i + j] += a[i] * b[j];
        return r;
    }

    /// <summary>Sum of two polynomials, aligned at the constant term.</summary>
    public static double[] Polyadd(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var n = Math.Max(a.Length, b.Length);
        var r = new double[n];
        for (var i = 0; i < a.Length; i++) r[n - a.Length + i] += a[i];
        for (var i = 0; i < b.Length; i++) r[n - b.Length + i] += b[i];
        return r;
    }

    /// <summary>Difference <c>a − b</c> of two polynomials, aligned at the constant term.</summary>
    public static double[] Polysub(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var n = Math.Max(a.Length, b.Length);
        var r = new double[n];
        for (var i = 0; i < a.Length; i++) r[n - a.Length + i] += a[i];
        for (var i = 0; i < b.Length; i++) r[n - b.Length + i] -= b[i];
        return r;
    }

    private static ReadOnlySpan<double> TrimLeadingZeros(ReadOnlySpan<double> a)
    {
        var i = 0;
        while (i < a.Length && a[i] == 0.0) i++;
        return a[i..];
    }

    /// <summary>
    /// Roots of a real polynomial (highest power first), as <c>numpy.roots</c>: leading zeros are
    /// stripped, trailing zeros become roots at the origin, and the rest are eigenvalues of the
    /// companion matrix.
    /// </summary>
    /// <remarks>
    /// The companion matrix is <b>balanced</b> before the QR iteration (Parlett &amp; Reinsch 1969),
    /// as LAPACK's <c>dgeev</c> does for numpy — the coefficients of a high-order filter span many
    /// decades and an unbalanced companion loses most of its digits. Each root is then polished by
    /// Newton's method on the original polynomial, accepting a step only if it reduces the residual,
    /// so conjugate pairs stay exact conjugates. Multiple roots remain limited by conditioning
    /// (an N-fold root is only determined to about eps^(1/N)) — that is the problem, not the method.
    /// </remarks>
    public static Complex[] Roots(ReadOnlySpan<double> p)
    {
        var start = 0;
        while (start < p.Length && p[start] == 0.0) start++;
        var end = p.Length;
        while (end > start && p[end - 1] == 0.0) end--;
        var trailing = p.Length - end;
        if (start >= end) return [];
        var c = p[start..end].ToArray();
        var n = c.Length - 1;
        var roots = new Complex[n + trailing];
        if (n >= 1)
        {
            var found = CompanionRoots(c);
            Array.Copy(found, roots, n);
        }
        return roots;
    }

    private static Complex[] CompanionRoots(double[] c)
    {
        var n = c.Length - 1;
        if (n == 1) return [new Complex(-c[1] / c[0], 0)];

        // numpy's companion: ones on the sub-diagonal, -c[1:]/c[0] in the first row.
        var a = new double[n * n];
        for (var j = 0; j < n; j++) a[j] = -c[j + 1] / c[0];
        for (var i = 1; i < n; i++) a[i * n + i - 1] = 1.0;
        Balance(a, n);

        // A balanced companion is generically non-symmetric; GraviNum's general eigenvalue
        // routine (Hessenberg + Francis QR) does the rest.
        var (re, im) = Decomposition.Eigenvalues(new NdArray(a, n, n));
        var reS = re.ToArray();
        var imS = im.ToArray();
        var roots = new Complex[n];
        for (var i = 0; i < n; i++) roots[i] = new Complex(reS[i], imS[i]);

        // Polish: real roots stay real; for each complex pair polish the upper member and mirror.
        var used = new bool[n];
        for (var i = 0; i < n; i++)
        {
            if (used[i]) continue;
            used[i] = true;
            if (roots[i].Imaginary == 0)
            {
                roots[i] = new Complex(PolishReal(c, roots[i].Real), 0);
                continue;
            }
            var partner = -1;
            var best = double.PositiveInfinity;
            for (var j = 0; j < n; j++)
            {
                if (used[j] || roots[j].Imaginary == 0) continue;
                var d = Complex.Abs(roots[j] - Complex.Conjugate(roots[i]));
                if (d < best) { best = d; partner = j; }
            }
            var z = PolishComplex(c, roots[i].Imaginary > 0 ? roots[i] : Complex.Conjugate(roots[i]));
            if (partner >= 0)
            {
                used[partner] = true;
                if (roots[i].Imaginary > 0) { roots[i] = z; roots[partner] = Complex.Conjugate(z); }
                else { roots[i] = Complex.Conjugate(z); roots[partner] = z; }
            }
            else roots[i] = roots[i].Imaginary > 0 ? z : Complex.Conjugate(z);
        }
        return roots;
    }

    private static double PolishReal(double[] c, double x)
    {
        var fx = Math.Abs(Polyval(c, x));
        for (var it = 0; it < 4 && fx > 0; it++)
        {
            double y = 0, dy = 0;
            foreach (var a in c) { dy = dy * x + y; y = y * x + a; }
            if (dy == 0) break;
            var xn = x - y / dy;
            var fn = Math.Abs(Polyval(c, xn));
            if (!(fn < fx)) break;
            x = xn; fx = fn;
        }
        return x;
    }

    private static Complex PolishComplex(double[] c, Complex x)
    {
        var fx = Complex.Abs(Polyval(c, x));
        for (var it = 0; it < 4 && fx > 0; it++)
        {
            Complex y = 0, dy = 0;
            foreach (var a in c) { dy = dy * x + y; y = y * x + a; }
            if (dy == Complex.Zero) break;
            var xn = x - y / dy;
            var fn = Complex.Abs(Polyval(c, xn));
            if (!(fn < fx)) break;
            x = xn; fx = fn;
        }
        return x;
    }

    /// <summary>Diagonal similarity scaling by powers of two (EISPACK <c>balanc</c>).</summary>
    private static void Balance(double[] a, int n)
    {
        const double radix = 2.0;
        const double sqrdx = radix * radix;
        var done = false;
        var guard = 0;
        while (!done && guard++ < 1000)
        {
            done = true;
            for (var i = 0; i < n; i++)
            {
                double r = 0, c = 0;
                for (var j = 0; j < n; j++)
                {
                    if (j == i) continue;
                    c += Math.Abs(a[j * n + i]);
                    r += Math.Abs(a[i * n + j]);
                }
                if (c == 0 || r == 0) continue;
                var g = r / radix;
                var f = 1.0;
                var s = c + r;
                while (c < g) { f *= radix; c *= sqrdx; }
                g = r * radix;
                while (c > g) { f /= radix; c /= sqrdx; }
                if ((c + r) / f < 0.95 * s)
                {
                    done = false;
                    g = 1.0 / f;
                    for (var j = 0; j < n; j++) a[i * n + j] *= g;
                    for (var j = 0; j < n; j++) a[j * n + i] *= f;
                }
            }
        }
    }
}
