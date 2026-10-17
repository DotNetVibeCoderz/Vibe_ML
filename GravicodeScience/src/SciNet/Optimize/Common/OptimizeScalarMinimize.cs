namespace Gravicode.Science.SciNet;

/// <summary>Methods of <see cref="Optimize.MinimizeScalar"/>.</summary>
public enum MinimizeScalarMethod
{
    /// <summary>Brent's parabolic interpolation with golden-section safeguards (needs a bracket, not bounds).</summary>
    Brent,
    /// <summary>Brent's method confined to an interval (<c>fminbound</c>).</summary>
    Bounded,
    /// <summary>Plain golden-section search; slow but very predictable.</summary>
    Golden,
}

/// <summary>Result of a one-dimensional minimisation.</summary>
public sealed class MinimizeScalarResult
{
    /// <summary>The minimiser.</summary>
    public double X { get; init; }
    /// <summary>f at <see cref="X"/>.</summary>
    public double Fun { get; init; }
    /// <summary>Iterations.</summary>
    public int Nit { get; init; }
    /// <summary>Function evaluations.</summary>
    public int Nfev { get; init; }
    /// <summary>True when the method's own convergence test was met.</summary>
    public bool Success { get; init; }
    /// <summary>0 success, 1 evaluation/iteration limit, 2 NaN encountered.</summary>
    public int Status { get; init; }
    /// <summary>Termination reason.</summary>
    public string Message { get; init; } = "";

    /// <inheritdoc />
    public override string ToString() => $"{Message} x={X:G12} fun={Fun:G12} nit={Nit} nfev={Nfev}";
}

/// <summary>Result of <see cref="Optimize.Bracket"/>: points with <c>f(Xb) &lt; f(Xa)</c> and <c>f(Xb) &lt; f(Xc)</c>.</summary>
public sealed record BracketResult(double Xa, double Xb, double Xc, double Fa, double Fb, double Fc, int Nfev);

public static partial class Optimize
{
    /// <summary>
    /// Finds a downhill bracket <c>(xa, xb, xc)</c> around a minimum by golden-ratio expansion with
    /// parabolic extrapolation (Numerical Recipes <c>mnbrak</c>, as <c>scipy.optimize.bracket</c>).
    /// </summary>
    /// <exception cref="InvalidOperationException">No valid bracket was found (for example f is monotone).</exception>
    public static BracketResult Bracket(Func<double, double> f, double xa = 0.0, double xb = 1.0, double growLimit = 110.0, int maxiter = 1000)
    {
        ArgumentNullException.ThrowIfNull(f);
        var r = BracketCore(f, xa, xb, growLimit, maxiter, out var valid);
        if (!valid) throw new InvalidOperationException("The algorithm terminated without finding a valid bracket. Consider trying different initial points.");
        return r;
    }

    /// <summary>The bracket search; <paramref name="valid"/> is false when the final triple is not a bracket.</summary>
    internal static BracketResult BracketCore(Func<double, double> f, double xa, double xb, double growLimit, int maxiter, out bool valid)
    {
        const double gold = 1.618034;
        const double verySmall = 1e-21;
        var fa = f(xa);
        var fb = f(xb);
        if (fa < fb) { (xa, xb) = (xb, xa); (fa, fb) = (fb, fa); }
        var xc = xb + gold * (xb - xa);
        var fc = f(xc);
        var funcalls = 3;
        var iter = 0;
        while (fc < fb)
        {
            var tmp1 = (xb - xa) * (fb - fc);
            var tmp2 = (xb - xc) * (fb - fa);
            var val = tmp2 - tmp1;
            var denom = Math.Abs(val) < verySmall ? 2.0 * verySmall : 2.0 * val;
            var w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
            var wlim = xb + growLimit * (xc - xb);
            if (iter > maxiter) throw new InvalidOperationException("Bracket: too many iterations.");
            iter++;
            double fw;
            if ((w - xc) * (xb - w) > 0.0)
            {
                fw = f(w); funcalls++;
                if (fw < fc) { xa = xb; xb = w; fa = fb; fb = fw; break; }
                if (fw > fb) { xc = w; fc = fw; break; }
                w = xc + gold * (xc - xb);
                fw = f(w); funcalls++;
            }
            else if ((w - wlim) * (wlim - xc) >= 0.0)
            {
                w = wlim; fw = f(w); funcalls++;
            }
            else if ((w - wlim) * (xc - w) > 0.0)
            {
                fw = f(w); funcalls++;
                if (fw < fc)
                {
                    xb = xc; xc = w; w = xc + gold * (xc - xb);
                    fb = fc; fc = fw; fw = f(w); funcalls++;
                }
            }
            else
            {
                w = xc + gold * (xc - xb);
                fw = f(w); funcalls++;
            }
            xa = xb; xb = xc; xc = w;
            fa = fb; fb = fc; fc = fw;
        }
        var cond1 = (fb < fc && fb <= fa) || (fb < fa && fb <= fc);
        var cond2 = (xa < xb && xb < xc) || (xc < xb && xb < xa);
        var cond3 = double.IsFinite(xa) && double.IsFinite(xb) && double.IsFinite(xc);
        valid = cond1 && cond2 && cond3;
        return new BracketResult(xa, xb, xc, fa, fb, fc, funcalls);
    }

    /// <summary>
    /// Minimises a function of one variable (<c>scipy.optimize.minimize_scalar</c>).
    /// </summary>
    /// <param name="f">The function.</param>
    /// <param name="bracket">For Brent/Golden: two points to start the bracket search, or three
    /// <c>(a, b, c)</c> with <c>f(b)</c> below both ends. Null starts the search from (0, 1).</param>
    /// <param name="bounds">For Bounded: the interval <c>(lo, hi)</c>; supplying it selects Bounded by default.</param>
    /// <param name="method">Defaults to Bounded when bounds are given, otherwise Brent.</param>
    /// <param name="tol">Relative x tolerance for Brent/Golden; absolute (<c>xatol</c>) for Bounded.</param>
    /// <param name="maxiter">Iteration limit (Bounded: evaluation limit). Defaults 500 / 5000 (Golden).</param>
    public static MinimizeScalarResult MinimizeScalar(Func<double, double> f, double[]? bracket = null, (double Lo, double Hi)? bounds = null,
        MinimizeScalarMethod? method = null, double? tol = null, int? maxiter = null)
    {
        ArgumentNullException.ThrowIfNull(f);
        var m = method ?? (bounds is null ? MinimizeScalarMethod.Brent : MinimizeScalarMethod.Bounded);
        switch (m)
        {
            case MinimizeScalarMethod.Bounded:
                if (bounds is null) throw new ArgumentException("The Bounded method requires bounds.");
                return Fminbound(f, bounds.Value.Lo, bounds.Value.Hi, tol ?? 1e-5, maxiter ?? 500);
            case MinimizeScalarMethod.Brent:
                if (bounds is not null) throw new ArgumentException("Use method Bounded with bounds; Brent takes a bracket.");
                return BrentScalar(f, bracket, tol ?? 1.48e-8, maxiter ?? 500);
            case MinimizeScalarMethod.Golden:
                if (bounds is not null) throw new ArgumentException("Use method Bounded with bounds; Golden takes a bracket.");
                return GoldenScalar(f, bracket, tol ?? 1.4901161193847656e-08, maxiter ?? 5000);
            default:
                throw new ArgumentOutOfRangeException(nameof(method));
        }
    }

    private static BracketResult BracketInfo(Func<double, double> f, double[]? brack)
    {
        if (brack is null) return Bracket(f, 0.0, 1.0);
        if (brack.Length == 2) return Bracket(f, brack[0], brack[1]);
        if (brack.Length == 3)
        {
            double xa = brack[0], xb = brack[1], xc = brack[2];
            if (xa > xc) (xa, xc) = (xc, xa);
            if (!(xa < xb && xb < xc)) throw new ArgumentException("Bracketing values (xa, xb, xc) do not fulfil xa < xb < xc.");
            var fa = f(xa); var fb = f(xb); var fc = f(xc);
            if (!(fb < fa && fb < fc)) throw new ArgumentException("Bracketing values do not fulfil f(xb) < f(xa) and f(xb) < f(xc).");
            return new BracketResult(xa, xb, xc, fa, fb, fc, 3);
        }
        throw new ArgumentException("Bracketing interval must be length 2 or 3.");
    }

    /// <summary>SciPy's <c>Brent.optimize</c>, line for line.</summary>
    /// <param name="f">Function.</param>
    /// <param name="brack">Bracket specification.</param>
    /// <param name="tol">Relative tolerance.</param>
    /// <param name="maxiter">Iteration limit.</param>
    /// <param name="recover">When the bracket search fails, return the best point seen instead of
    /// throwing (SciPy's <c>_recover_from_bracket_error</c>, used by Powell's line search).</param>
    internal static MinimizeScalarResult BrentScalar(Func<double, double> f, double[]? brack, double tol, int maxiter, bool recover = false)
    {
        if (tol < 0) throw new ArgumentException("tol must be >= 0.");
        const double mintol = 1.0e-11, cg = 0.3819660;
        BracketResult br;
        if (recover && (brack is null || brack.Length == 2))
        {
            br = BracketCore(f, brack?[0] ?? 0.0, brack?[1] ?? 1.0, 110.0, 1000, out var valid);
            if (!valid)
            {
                double bx = br.Xa, bf = br.Fa;
                if (br.Fb < bf) { bx = br.Xb; bf = br.Fb; }
                if (br.Fc < bf) { bx = br.Xc; bf = br.Fc; }
                return new MinimizeScalarResult { X = bx, Fun = bf, Nfev = br.Nfev, Nit = 0, Success = false, Status = 1,
                    Message = "The algorithm terminated without finding a valid bracket." };
            }
        }
        else br = BracketInfo(f, brack);
        var funcalls = br.Nfev;
        double x, w, v, fx, fw, fv;
        x = w = v = br.Xb;
        fw = fv = fx = br.Fb;
        double a, b;
        if (br.Xa < br.Xc) { a = br.Xa; b = br.Xc; } else { a = br.Xc; b = br.Xa; }
        double deltax = 0, rat = 0;
        var iter = 0;
        while (iter < maxiter)
        {
            var tol1 = tol * Math.Abs(x) + mintol;
            var tol2 = 2.0 * tol1;
            var xmid = 0.5 * (a + b);
            if (Math.Abs(x - xmid) < (tol2 - 0.5 * (b - a))) break;
            if (Math.Abs(deltax) <= tol1)
            {
                deltax = x >= xmid ? a - x : b - x;
                rat = cg * deltax;
            }
            else
            {
                var tmp1 = (x - w) * (fx - fv);
                var tmp2 = (x - v) * (fx - fw);
                var p = (x - v) * tmp2 - (x - w) * tmp1;
                tmp2 = 2.0 * (tmp2 - tmp1);
                if (tmp2 > 0.0) p = -p;
                tmp2 = Math.Abs(tmp2);
                var dxTemp = deltax;
                deltax = rat;
                if (p > tmp2 * (a - x) && p < tmp2 * (b - x) && Math.Abs(p) < Math.Abs(0.5 * tmp2 * dxTemp))
                {
                    rat = p * 1.0 / tmp2;
                    var u0 = x + rat;
                    if ((u0 - a) < tol2 || (b - u0) < tol2) rat = xmid - x >= 0 ? tol1 : -tol1;
                }
                else
                {
                    deltax = x >= xmid ? a - x : b - x;
                    rat = cg * deltax;
                }
            }
            var u = Math.Abs(rat) < tol1 ? (rat >= 0 ? x + tol1 : x - tol1) : x + rat;
            var fu = f(u);
            funcalls++;
            if (fu > fx)
            {
                if (u < x) a = u; else b = u;
                if (fu <= fw || w == x) { v = w; w = u; fv = fw; fw = fu; }
                else if (fu <= fv || v == x || v == w) { v = u; fv = fu; }
            }
            else
            {
                if (u >= x) a = x; else b = x;
                v = w; w = x; x = u;
                fv = fw; fw = fx; fx = fu;
            }
            iter++;
        }
        var nan = double.IsNaN(x) || double.IsNaN(fx);
        var ok = iter < maxiter && !nan;
        return new MinimizeScalarResult
        {
            X = x, Fun = fx, Nit = iter, Nfev = funcalls, Success = ok,
            Status = nan ? 2 : ok ? 0 : 1,
            Message = nan ? "NaN result encountered." : ok ? "Optimization terminated successfully;\nThe returned value satisfies the termination criteria\n(using xtol = " + tol + " )" : "Maximum number of iterations exceeded",
        };
    }

    /// <summary>
    /// Bounded scalar minimisation (<c>scipy.optimize.fminbound</c>, the engine of
    /// <c>minimize_scalar(method='bounded')</c>): Brent's golden-section/parabolic <c>fmin</c> on [x1, x2].
    /// </summary>
    /// <param name="f">Function of one variable.</param>
    /// <param name="x1">Lower end (finite).</param>
    /// <param name="x2">Upper end (finite).</param>
    /// <param name="xatol">Absolute tolerance on x (SciPy's <c>xtol</c>).</param>
    /// <param name="maxfun">Maximum function evaluations.</param>
    public static MinimizeScalarResult Fminbound(Func<double, double> f, double x1, double x2, double xatol = 1e-5, int maxfun = 500)
    {
        if (!double.IsFinite(x1) || !double.IsFinite(x2)) throw new ArgumentException("Optimization bounds must be finite scalars.");
        if (x1 > x2) throw new ArgumentException("The lower bound exceeds the upper bound.");
        var flag = 0;
        var sqrtEps = Math.Sqrt(2.2e-16);
        var goldenMean = 0.5 * (3.0 - Math.Sqrt(5.0));
        double a = x1, b = x2;
        var fulc = a + goldenMean * (b - a);
        double nfc = fulc, xf = fulc;
        double rat = 0, e = 0;
        var x = xf;
        var fx = f(x);
        var num = 1;
        var fu = double.PositiveInfinity;
        double ffulc = fx, fnfc = fx;
        var xm = 0.5 * (a + b);
        var tol1 = sqrtEps * Math.Abs(xf) + xatol / 3.0;
        var tol2 = 2.0 * tol1;
        while (Math.Abs(xf - xm) > (tol2 - 0.5 * (b - a)))
        {
            var golden = true;
            if (Math.Abs(e) > tol1)
            {
                golden = false;
                var r = (xf - nfc) * (fx - ffulc);
                var q = (xf - fulc) * (fx - fnfc);
                var p = (xf - fulc) * q - (xf - nfc) * r;
                q = 2.0 * (q - r);
                if (q > 0.0) p = -p;
                q = Math.Abs(q);
                r = e;
                e = rat;
                if (Math.Abs(p) < Math.Abs(0.5 * q * r) && p > q * (a - xf) && p < q * (b - xf))
                {
                    rat = (p + 0.0) / q;
                    x = xf + rat;
                    if ((x - a) < tol2 || (b - x) < tol2)
                    {
                        var si0 = Math.Sign(xm - xf) + ((xm - xf) == 0 ? 1 : 0);
                        rat = tol1 * si0;
                    }
                }
                else golden = true;
            }
            if (golden)
            {
                e = xf >= xm ? a - xf : b - xf;
                rat = goldenMean * e;
            }
            var si = Math.Sign(rat) + (rat == 0 ? 1 : 0);
            x = xf + si * Math.Max(Math.Abs(rat), tol1);
            fu = f(x);
            num++;
            if (fu <= fx)
            {
                if (x >= xf) a = xf; else b = xf;
                fulc = nfc; ffulc = fnfc;
                nfc = xf; fnfc = fx;
                xf = x; fx = fu;
            }
            else
            {
                if (x < xf) a = x; else b = x;
                if (fu <= fnfc || nfc == xf) { fulc = nfc; ffulc = fnfc; nfc = x; fnfc = fu; }
                else if (fu <= ffulc || fulc == xf || fulc == nfc) { fulc = x; ffulc = fu; }
            }
            xm = 0.5 * (a + b);
            tol1 = sqrtEps * Math.Abs(xf) + xatol / 3.0;
            tol2 = 2.0 * tol1;
            if (num >= maxfun) { flag = 1; break; }
        }
        if (double.IsNaN(xf) || double.IsNaN(fx) || double.IsNaN(fu)) flag = 2;
        return new MinimizeScalarResult
        {
            X = xf, Fun = fx, Nit = num, Nfev = num, Success = flag == 0, Status = flag,
            Message = flag switch { 0 => "Solution found.", 1 => "Maximum number of function calls reached.", _ => "NaN result encountered." },
        };
    }

    /// <summary>SciPy's <c>_minimize_scalar_golden</c>.</summary>
    internal static MinimizeScalarResult GoldenScalar(Func<double, double> f, double[]? brack, double tol, int maxiter)
    {
        const double gR = 0.61803399, gC = 1.0 - gR;
        var br = BracketInfo(f, brack);
        var funcalls = br.Nfev;
        var x3 = br.Xc;
        var x0 = br.Xa;
        double x1, x2;
        if (Math.Abs(br.Xc - br.Xb) > Math.Abs(br.Xb - br.Xa)) { x1 = br.Xb; x2 = br.Xb + gC * (br.Xc - br.Xb); }
        else { x2 = br.Xb; x1 = br.Xb - gC * (br.Xb - br.Xa); }
        var f1 = f(x1);
        var f2 = f(x2);
        funcalls += 2;
        var nit = 0;
        for (var i = 0; i < maxiter; i++)
        {
            if (Math.Abs(x3 - x0) <= tol * (Math.Abs(x1) + Math.Abs(x2))) break;
            if (f2 < f1) { x0 = x1; x1 = x2; x2 = gR * x1 + gC * x3; f1 = f2; f2 = f(x2); }
            else { x3 = x2; x2 = x1; x1 = gR * x2 + gC * x0; f2 = f1; f1 = f(x1); }
            funcalls++;
            nit++;
        }
        double xmin, fval;
        if (f1 < f2) { xmin = x1; fval = f1; } else { xmin = x2; fval = f2; }
        var ok = nit < maxiter && !(double.IsNaN(fval) || double.IsNaN(xmin));
        return new MinimizeScalarResult
        {
            X = xmin, Fun = fval, Nit = nit, Nfev = funcalls, Success = ok, Status = ok ? 0 : (nit >= maxiter ? 1 : 2),
            Message = ok ? "Optimization terminated successfully;\nThe returned value satisfies the termination criteria\n(using xtol = " + tol + " )" : nit >= maxiter ? "Maximum number of iterations exceeded" : "NaN result encountered.",
        };
    }
}
