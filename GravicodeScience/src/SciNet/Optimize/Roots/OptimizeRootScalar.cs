namespace Gravicode.Science.SciNet;

/// <summary>Methods of <see cref="Optimize.RootScalar"/>.</summary>
public enum RootScalarMethod
{
    /// <summary>Bisection (needs a bracket).</summary>
    Bisect,
    /// <summary>Brent's method with inverse quadratic extrapolation (needs a bracket).</summary>
    Brentq,
    /// <summary>Brent's method with hyperbolic extrapolation (needs a bracket).</summary>
    Brenth,
    /// <summary>Ridders' method (needs a bracket).</summary>
    Ridder,
    /// <summary>Alefeld–Potra–Shi TOMS 748 (needs a bracket).</summary>
    Toms748,
    /// <summary>Newton–Raphson (needs x0 and fprime).</summary>
    Newton,
    /// <summary>Secant method (needs x0, optionally x1).</summary>
    Secant,
    /// <summary>Halley's method (needs x0, fprime and fprime2).</summary>
    Halley,
}

/// <summary>Result of a scalar root find, mirroring <c>scipy.optimize.RootResults</c>.</summary>
/// <param name="Root">Estimated root.</param>
/// <param name="Iterations">Iterations performed.</param>
/// <param name="FunctionCalls">Function (and derivative) evaluations.</param>
/// <param name="Converged">True when the tolerance was met.</param>
/// <param name="Flag">"converged", "convergence error", "sign error", ...</param>
/// <param name="Method">The method used.</param>
public sealed record RootResults(double Root, int Iterations, int FunctionCalls, bool Converged, string Flag, RootScalarMethod Method);

public static partial class Optimize
{
    private const double RootXtol = 2e-12;
    private const double RootRtol = 4 * 2.220446049250313e-16;

    /// <summary>
    /// Bisection on <c>[a, b]</c> (<c>scipy.optimize.bisect</c>); f(a) and f(b) must differ in sign.
    /// Throws <see cref="InvalidOperationException"/> when it does not converge and
    /// <paramref name="throwOnFailure"/> is true.
    /// </summary>
    public static RootResults Bisect(Func<double, double> f, double a, double b, double xtol = RootXtol, double rtol = RootRtol, int maxiter = 100, bool throwOnFailure = true)
        => OptimizeRootBracket.Run(RootScalarMethod.Bisect, f, a, b, xtol, rtol, maxiter, throwOnFailure);

    /// <summary>Brent's method (<c>scipy.optimize.brentq</c>): the usual first choice for a bracketed root.</summary>
    public static RootResults Brentq(Func<double, double> f, double a, double b, double xtol = RootXtol, double rtol = RootRtol, int maxiter = 100, bool throwOnFailure = true)
        => OptimizeRootBracket.Run(RootScalarMethod.Brentq, f, a, b, xtol, rtol, maxiter, throwOnFailure);

    /// <summary>Brent's method with hyperbolic extrapolation (<c>scipy.optimize.brenth</c>).</summary>
    public static RootResults Brenth(Func<double, double> f, double a, double b, double xtol = RootXtol, double rtol = RootRtol, int maxiter = 100, bool throwOnFailure = true)
        => OptimizeRootBracket.Run(RootScalarMethod.Brenth, f, a, b, xtol, rtol, maxiter, throwOnFailure);

    /// <summary>Ridders' method (<c>scipy.optimize.ridder</c>).</summary>
    public static RootResults Ridder(Func<double, double> f, double a, double b, double xtol = RootXtol, double rtol = RootRtol, int maxiter = 100, bool throwOnFailure = true)
        => OptimizeRootBracket.Run(RootScalarMethod.Ridder, f, a, b, xtol, rtol, maxiter, throwOnFailure);

    /// <summary>
    /// Alefeld, Potra &amp; Shi's TOMS Algorithm 748 (<c>scipy.optimize.toms748</c>); <paramref name="k"/>
    /// Newton-quadratic steps per iteration (1 or 2 are sensible).
    /// </summary>
    public static RootResults Toms748(Func<double, double> f, double a, double b, int k = 1, double xtol = RootXtol, double rtol = RootRtol, int maxiter = 100, bool throwOnFailure = true)
        => OptimizeToms748.Solve(f, a, b, k, xtol, rtol, maxiter, throwOnFailure);

    /// <summary>
    /// Newton–Raphson, secant (no <paramref name="fprime"/>) or Halley (with <paramref name="fprime2"/>),
    /// exactly as <c>scipy.optimize.newton</c>: convergence when |x_{k+1} − x_k| ≤ tol + rtol·|x_k|.
    /// </summary>
    public static RootResults Newton(Func<double, double> f, double x0, Func<double, double>? fprime = null, double tol = 1.48e-8, int maxiter = 50,
        Func<double, double>? fprime2 = null, double? x1 = null, double rtol = 0.0, bool throwOnFailure = true)
        => OptimizeRootNewton.Scalar(f, x0, fprime, tol, maxiter, fprime2, x1, rtol, throwOnFailure);

    /// <summary>
    /// Element-wise Newton/secant/Halley over many independent starting points (SciPy's
    /// <c>_array_newton</c>): <paramref name="f"/> maps the whole vector at once. Converged entries
    /// stop moving; throws only if every entry fails.
    /// </summary>
    /// <returns>The roots, and which entries converged.</returns>
    public static (double[] Root, bool[] Converged) Newton(Func<double[], double[]> f, double[] x0, Func<double[], double[]>? fprime = null,
        double tol = 1.48e-8, int maxiter = 50, Func<double[], double[]>? fprime2 = null)
        => OptimizeRootNewton.Array(f, x0, fprime, tol, maxiter, fprime2);

    /// <summary>
    /// Fixed point of <paramref name="func"/> (<c>scipy.optimize.fixed_point</c>) with Steffensen's
    /// Δ² acceleration (<paramref name="useDel2"/>, default) or plain iteration. Works element-wise on vectors.
    /// </summary>
    public static double[] FixedPoint(Func<double[], double[]> func, double[] x0, double xtol = 1e-8, int maxiter = 500, bool useDel2 = true)
    {
        ArgumentNullException.ThrowIfNull(func);
        var p0 = (double[])x0.Clone();
        double[] p = p0;
        for (var i = 0; i < maxiter; i++)
        {
            var p1 = func((double[])p0.Clone());
            if (useDel2)
            {
                var p2 = func((double[])p1.Clone());
                p = new double[p0.Length];
                for (var k = 0; k < p.Length; k++)
                {
                    var d = p2[k] - 2.0 * p1[k] + p0[k];
                    p[k] = d != 0 ? p0[k] - (p1[k] - p0[k]) * (p1[k] - p0[k]) / d : p2[k];
                }
            }
            else p = p1;
            var ok = true;
            for (var k = 0; k < p.Length; k++)
            {
                var rel = p0[k] != 0 ? (p[k] - p0[k]) / p0[k] : p[k];
                if (!(Math.Abs(rel) < xtol)) { ok = false; break; }
            }
            if (ok) return p;
            p0 = p;
        }
        throw new InvalidOperationException($"Failed to converge after {maxiter} iterations, value is [{string.Join(", ", p)}]");
    }

    /// <summary>Scalar <see cref="FixedPoint(Func{double[], double[]}, double[], double, int, bool)"/>.</summary>
    public static double FixedPoint(Func<double, double> func, double x0, double xtol = 1e-8, int maxiter = 500, bool useDel2 = true)
        => FixedPoint(v => [func(v[0])], [x0], xtol, maxiter, useDel2)[0];

    /// <summary>
    /// Finds a root of a scalar function (<c>scipy.optimize.root_scalar</c>). Without
    /// <paramref name="method"/>: Brentq when a bracket is given, else Halley/Newton/secant according to
    /// which derivatives and starting points are supplied. Never throws for non-convergence; check
    /// <see cref="RootResults.Converged"/>.
    /// </summary>
    public static RootResults RootScalar(Func<double, double> f, RootScalarMethod? method = null, (double A, double B)? bracket = null,
        Func<double, double>? fprime = null, Func<double, double>? fprime2 = null, double? x0 = null, double? x1 = null,
        double? xtol = null, double? rtol = null, int? maxiter = null)
    {
        ArgumentNullException.ThrowIfNull(f);
        RootScalarMethod m;
        if (method is { } mm) m = mm;
        else if (bracket is not null) m = RootScalarMethod.Brentq;
        else if (x0 is not null)
            m = fprime is not null ? (fprime2 is not null ? RootScalarMethod.Halley : RootScalarMethod.Newton)
                : x1 is not null ? RootScalarMethod.Secant : RootScalarMethod.Newton;
        else throw new ArgumentException("Unable to select a solver as neither bracket nor starting point provided.");

        switch (m)
        {
            case RootScalarMethod.Bisect or RootScalarMethod.Brentq or RootScalarMethod.Brenth or RootScalarMethod.Ridder or RootScalarMethod.Toms748:
                if (bracket is null) throw new ArgumentException($"Method {m} requires a bracket.");
                var (a, b) = bracket.Value;
                if (m == RootScalarMethod.Toms748)
                    return Toms748(f, a, b, 1, xtol ?? RootXtol, rtol ?? RootRtol, maxiter ?? 100, throwOnFailure: false);
                return OptimizeRootBracket.Run(m, f, a, b, xtol ?? RootXtol, rtol ?? RootRtol, maxiter ?? 100, false);
            case RootScalarMethod.Newton:
                if (x0 is null) throw new ArgumentException("Newton requires x0.");
                // SciPy: Newton without a derivative runs the secant method.
                return OptimizeRootNewton.Scalar(f, x0.Value, fprime, xtol ?? 1.48e-8, maxiter ?? 50, null, x1, rtol ?? 0.0, false) with { Method = fprime is null ? RootScalarMethod.Secant : RootScalarMethod.Newton };
            case RootScalarMethod.Secant:
                if (x0 is null) throw new ArgumentException("Secant requires x0.");
                return OptimizeRootNewton.Scalar(f, x0.Value, null, xtol ?? 1.48e-8, maxiter ?? 50, null, x1, rtol ?? 0.0, false);
            case RootScalarMethod.Halley:
                if (x0 is null || fprime is null || fprime2 is null) throw new ArgumentException("Halley requires x0, fprime and fprime2.");
                return OptimizeRootNewton.Scalar(f, x0.Value, fprime, xtol ?? 1.48e-8, maxiter ?? 50, fprime2, null, rtol ?? 0.0, false);
            default:
                throw new ArgumentOutOfRangeException(nameof(method));
        }
    }
}

/// <summary>Translations of SciPy's C bracketing solvers (<c>bisect.c</c>, <c>brentq.c</c>, <c>brenth.c</c>, <c>ridder.c</c>).</summary>
internal static class OptimizeRootBracket
{
    public static RootResults Run(RootScalarMethod method, Func<double, double> f0, double xa, double xb, double xtol, double rtol, int maxiter, bool throwOnFailure)
    {
        ArgumentNullException.ThrowIfNull(f0);
        if (!(xtol > 0)) throw new ArgumentException($"xtol too small ({xtol} <= 0)");
        if (rtol < 4 * 2.220446049250313e-16) throw new ArgumentException($"rtol too small ({rtol} < {4 * 2.220446049250313e-16})");
        if (maxiter < 0) throw new ArgumentException("maxiter must be >= 0");
        double F(double x)
        {
            var v = f0(x);
            if (double.IsNaN(v)) throw new ArgumentException($"The function value at x={x} is NaN; solver cannot continue.");
            return v;
        }
        var funcalls = 2;
        var iterations = 0;
        var fa = F(xa); var fb = F(xb);
        if (fa * fb > 0) throw new ArgumentException("f(a) and f(b) must have different signs");
        RootResults Done(double x, bool ok)
        {
            if (!ok && throwOnFailure) throw new InvalidOperationException($"Failed to converge after {iterations} iterations, value is {x}.");
            return new RootResults(x, iterations, funcalls, ok, ok ? "converged" : "convergence error", method);
        }
        if (fa == 0) return Done(xa, true);
        if (fb == 0) return Done(xb, true);

        switch (method)
        {
            case RootScalarMethod.Bisect:
            {
                var dm = xb - xa;
                for (var i = 0; i < maxiter; i++)
                {
                    iterations++;
                    dm *= 0.5;
                    var xm = xa + dm;
                    var fm = F(xm); funcalls++;
                    if (fm * fa >= 0) xa = xm;
                    if (fm == 0 || Math.Abs(dm) < xtol + rtol * Math.Abs(xm)) return Done(xm, true);
                }
                return Done(xa, false);
            }
            case RootScalarMethod.Ridder:
            {
                var tol = xtol + rtol * Math.Min(Math.Abs(xa), Math.Abs(xb));
                double xn = 0;
                for (var i = 0; i < maxiter; i++)
                {
                    iterations++;
                    var dm = 0.5 * (xb - xa);
                    var xm = xa + dm;
                    var fm = F(xm); funcalls++;
                    var dn = (fb - fa > 0 ? 1.0 : -1.0) * dm * fm / Math.Sqrt(fm * fm - fa * fb);
                    xn = xm - (dn > 0 ? 1.0 : -1.0) * Math.Min(Math.Abs(dn), Math.Abs(dm) - 0.5 * tol);
                    var fn = F(xn); funcalls++;
                    if (double.IsNegative(fn) != double.IsNegative(fm)) { xa = xn; fa = fn; xb = xm; fb = fm; }
                    else if (double.IsNegative(fn) != double.IsNegative(fa)) { xb = xn; fb = fn; }
                    else { xa = xn; fa = fn; }
                    tol = xtol + rtol * xn;
                    if (fn == 0.0 || Math.Abs(xb - xa) < tol) return Done(xn, true);
                }
                return Done(xn, false);
            }
            default:
            {
                var hyperbolic = method == RootScalarMethod.Brenth;
                double xpre = xa, xcur = xb, fpre = fa, fcur = fb;
                double xblk = 0, fblk = 0, spre = 0, scur = 0;
                for (var i = 0; i < maxiter; i++)
                {
                    iterations++;
                    if (fpre != 0 && fcur != 0 && double.IsNegative(fpre) != double.IsNegative(fcur))
                    {
                        xblk = xpre; fblk = fpre;
                        spre = scur = xcur - xpre;
                    }
                    if (Math.Abs(fblk) < Math.Abs(fcur))
                    {
                        xpre = xcur; xcur = xblk; xblk = xpre;
                        fpre = fcur; fcur = fblk; fblk = fpre;
                    }
                    var delta = (xtol + rtol * Math.Abs(xcur)) / 2;
                    var sbis = (xblk - xcur) / 2;
                    if (fcur == 0 || Math.Abs(sbis) < delta) return Done(xcur, true);
                    if (Math.Abs(spre) > delta && Math.Abs(fcur) < Math.Abs(fpre))
                    {
                        double stry;
                        if (xpre == xblk) stry = -fcur * (xcur - xpre) / (fcur - fpre);
                        else
                        {
                            var dpre = (fpre - fcur) / (xpre - xcur);
                            var dblk = (fblk - fcur) / (xblk - xcur);
                            stry = hyperbolic
                                ? -fcur * (fblk - fpre) / (fblk * dpre - fpre * dblk)
                                : -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
                        }
                        if (2 * Math.Abs(stry) < Math.Min(Math.Abs(spre), 3 * Math.Abs(sbis) - delta)) { spre = scur; scur = stry; }
                        else { spre = sbis; scur = sbis; }
                    }
                    else { spre = sbis; scur = sbis; }
                    xpre = xcur; fpre = fcur;
                    if (Math.Abs(scur) > delta) xcur += scur;
                    else xcur += sbis > 0 ? delta : -delta;
                    fcur = F(xcur); funcalls++;
                }
                return Done(xcur, false);
            }
        }
    }
}

/// <summary>SciPy's <c>TOMS748Solver</c>, line for line.</summary>
internal static class OptimizeToms748
{
    private const double Eps = 2.220446049250313e-16;

    private static bool IsClose(double a, double b, double rtol, double atol) => Math.Abs(a - b) <= atol + rtol * Math.Abs(b);

    public static RootResults Solve(Func<double, double> f0, double a, double b, int k, double xtol, double rtol, int maxiter, bool throwOnFailure)
    {
        if (!(xtol > 0)) throw new ArgumentException($"xtol too small ({xtol} <= 0)");
        if (rtol < Eps) throw new ArgumentException($"rtol too small ({rtol} < {Eps})");
        if (maxiter < 1) throw new ArgumentException("maxiter must be greater than 0");
        if (!double.IsFinite(a) || !double.IsFinite(b)) throw new ArgumentException("a and b must be finite.");
        if (a >= b) throw new ArgumentException($"a and b are not an interval [{a}, {b}]");
        if (k < 1) throw new ArgumentException($"k too small ({k} < 1)");
        k = Math.Min(k, 100);
        var calls = 0; var iterations = 0;
        double F(double x)
        {
            var v = f0(x); calls++;
            if (!double.IsFinite(v)) throw new ArgumentException($"Invalid function value: f({x}) -> {v}");
            return v;
        }
        RootResults Res(double x, bool ok = true) => new(x, iterations, calls, ok, ok ? "converged" : "convergence error", RootScalarMethod.Toms748);

        double[] ab = [a, b];
        var fa = F(a);
        if (fa == 0) return Res(a);
        var fb = F(b);
        if (fb == 0) return Res(b);
        if (Math.Sign(fa) * Math.Sign(fb) > 0) throw new ArgumentException($"f(a) and f(b) must have different signs, but f({a})={fa}, f({b})={fb}");
        double[] fab = [fa, fb];

        double c = Secant(ab, fab);
        if (!(ab[0] < c && c < ab[1])) c = (ab[0] + ab[1]) / 2.0;
        var fc = F(c);
        if (fc == 0) return Res(c);
        var (d, fd) = Update(ab, fab, c, fc);
        double? e = null, fe = null;
        iterations++;

        while (true)
        {
            iterations++;
            var abWidth = ab[1] - ab[0];
            double? cc = null;
            double? dd = d, fdd = fd;
            for (var nsteps = 2; nsteps < k + 2; nsteps++)
            {
                if (NotClose([fab[0], fab[1], fdd, fe], 32 * Eps))
                {
                    var c0 = InversePolyZero(ab[0], ab[1], dd!.Value, e!.Value, fab[0], fab[1], fdd!.Value, fe!.Value);
                    if (ab[0] < c0 && c0 < ab[1]) cc = c0;
                }
                cc ??= NewtonQuadratic(ab, fab, dd!.Value, fdd!.Value, nsteps);
                fc = F(cc.Value);
                if (fc == 0) return Res(cc.Value);
                e = dd; fe = fdd;
                (var nd, var nfd) = Update(ab, fab, cc.Value, fc);
                dd = nd; fdd = nfd;
            }
            var uix = Math.Abs(fab[0]) < Math.Abs(fab[1]) ? 0 : 1;
            double u = ab[uix], fu = fab[uix];
            var A = (fab[1] - fab[0]) / (ab[1] - ab[0]);
            c = u - 2 * fu / A;
            if (Math.Abs(c - u) > 0.5 * (ab[1] - ab[0])) c = (ab[0] + ab[1]) / 2.0;
            else if (IsClose(c, u, Eps, 0))
            {
                var e0 = Frexp(fab[uix]); var e1 = Frexp(fab[1 - uix]);
                if (e0 < e1 - 50) c = (31 * ab[uix] + ab[1 - uix]) / 32;
                else
                {
                    var mm = uix == 0 ? 1 : -1;
                    c = u + mm * Math.Abs(c) * rtol + mm * xtol;
                }
                if (!(ab[0] < c && c < ab[1])) c = (ab[0] + ab[1]) / 2.0;
            }
            fc = F(c);
            if (fc == 0) return Res(c);
            e = dd; fe = fdd;
            (d, fd) = Update(ab, fab, c, fc);
            if (ab[1] - ab[0] > 0.5 * abWidth)
            {
                e = d; fe = fd;
                var z = (ab[0] + ab[1]) / 2.0;
                var fz = F(z);
                if (fz == 0) return Res(z);
                (d, fd) = Update(ab, fab, z, fz);
            }
            if (IsClose(ab[0], ab[1], rtol, xtol)) return Res((ab[0] + ab[1]) / 2.0);
            if (iterations >= maxiter)
            {
                if (throwOnFailure) throw new InvalidOperationException($"Failed to converge after {iterations + 1} iterations, bracket is [{ab[0]}, {ab[1]}]");
                return Res((ab[0] + ab[1]) / 2.0, false);
            }
        }
    }

    // np.frexp exponent: x = m 2^e with 0.5 <= |m| < 1.
    private static int Frexp(double x) => x == 0 ? 0 : Math.ILogB(x) + 1;

    private static double Secant(double[] x, double[] f)
    {
        if (f[0] == f[1]) return double.NaN;
        return Math.Abs(f[1]) > Math.Abs(f[0])
            ? (-f[0] / f[1] * x[1] + x[0]) / (1 - f[0] / f[1])
            : (-f[1] / f[0] * x[0] + x[1]) / (1 - f[1] / f[0]);
    }

    private static (double, double) Update(double[] ab, double[] fab, double c, double fc)
    {
        var idx = Math.Sign(fab[0]) * Math.Sign(fc) > 0 ? 0 : 1;
        var rx = ab[idx]; var rfx = fab[idx];
        fab[idx] = fc; ab[idx] = c;
        return (rx, rfx);
    }

    private static bool NotClose(double?[] fs, double atol)
    {
        foreach (var v in fs) if (v is null || v.Value == 0 || !double.IsFinite(v.Value)) return false;
        for (var i = 0; i < fs.Length; i++)
            for (var j = i + 1; j < fs.Length; j++)
                if (IsClose(fs[i]!.Value, fs[j]!.Value, 0, atol)) return false;
        return true;
    }

    private static double InversePolyZero(double a, double b, double c, double d, double fa, double fb, double fc, double fd)
    {
        // Neville-style interpolation of x as a polynomial in f, evaluated at f = 0.
        double[] xv = [fa, fb, fc, fd];
        double[] fv = [a, b, c, d];
        const int n = 4;
        var q = new double[n, n]; var dm = new double[n, n];
        for (var i = 0; i < n; i++) { q[i, 0] = fv[i]; dm[i, 0] = fv[i]; }
        for (var k = 1; k < n; k++)
            for (var i = k; i < n; i++)
            {
                var alpha = dm[i, k - 1] - q[i - 1, k - 1];
                var diffik = xv[i - k] - xv[i];
                q[i, k] = (xv[i] - 0) / diffik * alpha;
                dm[i, k] = (xv[i - k] - 0) / diffik * alpha;
            }
        double s = 0;
        for (var k = 1; k < n; k++) s += q[n - 1, k];
        return s + q[n - 1, 0];
    }

    private static double NewtonQuadratic(double[] ab, double[] fab, double d, double fd, int k)
    {
        double a = ab[0], b = ab[1], fa = fab[0], fb = fab[1];
        var B = (fb - fa) / (b - a);
        var fdb = (fd - fb) / (d - b);
        var A = (fdb - B) / (d - a);
        double P(double x) => (A * (x - b) + B) * (x - a) + fa;
        if (A == 0) return a - fa / B;
        var r = Math.Sign(A) * Math.Sign(fa) > 0 ? a : b;
        for (var i = 0; i < k; i++)
        {
            var r1 = r - P(r) / (B + A * (2 * r - a - b));
            if (!(ab[0] < r1 && r1 < ab[1]))
            {
                if (ab[0] < r && r < ab[1]) return r;
                r = (ab[0] + ab[1]) / 2.0;
                break;
            }
            r = r1;
        }
        return r;
    }
}

/// <summary>SciPy's <c>newton</c> (scalar) and <c>_array_newton</c>.</summary>
internal static class OptimizeRootNewton
{
    private static bool IsClose(double a, double b, double rtol, double atol) => Math.Abs(a - b) <= atol + rtol * Math.Abs(b);

    public static RootResults Scalar(Func<double, double> f, double x0, Func<double, double>? fprime, double tol, int maxiter,
        Func<double, double>? fprime2, double? x1, double rtol, bool throwOnFailure)
    {
        ArgumentNullException.ThrowIfNull(f);
        if (!(tol > 0)) throw new ArgumentException($"tol too small ({tol} <= 0)");
        if (maxiter < 1) throw new ArgumentException("maxiter must be greater than 0");
        var funcalls = 0;
        var p0 = x0;
        double p = p0;
        int itr;
        RootResults Out(double x, int it, bool ok, RootScalarMethod mth, string? flag = null)
        {
            if (!ok && throwOnFailure) throw new InvalidOperationException($"{flag ?? "Failed to converge"} after {it} iterations, value is {x}.");
            return new RootResults(x, it, funcalls, ok, ok ? "converged" : "convergence error", mth);
        }
        if (fprime is not null)
        {
            var mth = fprime2 is null ? RootScalarMethod.Newton : RootScalarMethod.Halley;
            for (itr = 0; itr < maxiter; itr++)
            {
                var fval = f(p0); funcalls++;
                if (fval == 0) return Out(p0, itr, true, mth);
                var fder = fprime(p0); funcalls++;
                if (fder == 0) return Out(p0, itr + 1, false, mth, "Derivative was zero. Failed to converge");
                var step = fval / fder;
                if (fprime2 is not null)
                {
                    var fder2 = fprime2(p0); funcalls++;
                    var adj = step * fder2 / fder / 2;
                    if (Math.Abs(adj) < 1) step /= 1.0 - adj;
                }
                p = p0 - step;
                if (IsClose(p, p0, rtol, tol)) return Out(p, itr + 1, true, mth);
                p0 = p;
            }
            return Out(p, itr, false, mth);
        }

        double p1;
        if (x1 is { } xv1)
        {
            if (xv1 == x0) throw new ArgumentException("x1 and x0 must be different");
            p1 = xv1;
        }
        else
        {
            const double eps = 1e-4;
            p1 = x0 * (1 + eps);
            p1 += p1 >= 0 ? eps : -eps;
        }
        var q0 = f(p0); funcalls++;
        var q1 = f(p1); funcalls++;
        if (Math.Abs(q1) < Math.Abs(q0)) { (p0, p1) = (p1, p0); (q0, q1) = (q1, q0); }
        for (itr = 0; itr < maxiter; itr++)
        {
            if (q1 == q0)
            {
                p = (p1 + p0) / 2.0;
                return Out(p, itr + 1, false, RootScalarMethod.Secant, $"Tolerance of {p1 - p0} reached. Failed to converge");
            }
            p = Math.Abs(q1) > Math.Abs(q0) ? (-q0 / q1 * p1 + p0) / (1 - q0 / q1) : (-q1 / q0 * p0 + p1) / (1 - q1 / q0);
            if (IsClose(p, p1, rtol, tol)) return Out(p, itr + 1, true, RootScalarMethod.Secant);
            p0 = p1; q0 = q1;
            p1 = p;
            q1 = f(p1); funcalls++;
        }
        return Out(p, itr, false, RootScalarMethod.Secant);
    }

    public static (double[] Root, bool[] Converged) Array(Func<double[], double[]> func, double[] x0, Func<double[], double[]>? fprime,
        double tol, int maxiter, Func<double[], double[]>? fprime2)
    {
        ArgumentNullException.ThrowIfNull(func);
        var n = x0.Length;
        var p = (double[])x0.Clone();
        var failures = Enumerable.Repeat(true, n).ToArray();
        var nzDer = Enumerable.Repeat(true, n).ToArray();
        if (fprime is not null)
        {
            for (var it = 0; it < maxiter; it++)
            {
                var fval = func((double[])p.Clone());
                if (fval.All(v => v == 0)) { for (var i = 0; i < n; i++) failures[i] = false; break; }
                var fder = fprime((double[])p.Clone());
                for (var i = 0; i < n; i++) nzDer[i] = fder[i] != 0;
                if (!nzDer.Any(v => v)) break;
                var fder2 = fprime2?.Invoke((double[])p.Clone());
                var anyFail = false;
                for (var i = 0; i < n; i++)
                {
                    if (!nzDer[i]) continue;
                    var dp = fval[i] / fder[i];
                    if (fder2 is not null) dp /= 1.0 - 0.5 * dp * fder2[i] / fder[i];
                    p[i] -= dp;
                    failures[i] = Math.Abs(dp) >= tol;
                    anyFail |= failures[i];
                }
                if (!anyFail) break;
            }
        }
        else
        {
            var dx = Math.Pow(2.220446049250313e-16, 0.33);
            var p1 = new double[n];
            for (var i = 0; i < n; i++) p1[i] = p[i] * (1 + dx) + (p[i] >= 0 ? dx : -dx);
            var q0 = func((double[])p.Clone());
            var q1 = func((double[])p1.Clone());
            var active = Enumerable.Repeat(true, n).ToArray();
            for (var it = 0; it < maxiter; it++)
            {
                for (var i = 0; i < n; i++) nzDer[i] = q1[i] != q0[i];
                if (!nzDer.Any(v => v)) { for (var i = 0; i < n; i++) p[i] = (p1[i] + p[i]) / 2.0; break; }
                var anyFail = false;
                var pn = (double[])p.Clone();
                for (var i = 0; i < n; i++)
                {
                    if (nzDer[i])
                    {
                        var dp = q1[i] * (p1[i] - p[i]) / (q1[i] - q0[i]);
                        pn[i] = p1[i] - dp;
                        failures[i] = Math.Abs(dp) >= tol;
                        anyFail |= failures[i];
                    }
                    else if (active[i]) pn[i] = (p1[i] + p[i]) / 2.0;
                    active[i] &= nzDer[i];
                }
                p = pn;
                if (!anyFail) break;
                (p1, p) = (p, p1);
                q0 = q1;
                q1 = func((double[])p1.Clone());
            }
        }
        var conv = failures.Select(v => !v).ToArray();
        var zeroDer = new bool[n];
        for (var i = 0; i < n; i++) zeroDer[i] = !nzDer[i] && failures[i];
        if (!zeroDer.Any(v => v) && failures.All(v => v))
            throw new InvalidOperationException($"all failed to converge after {maxiter} iterations");
        return (p, conv);
    }
}
