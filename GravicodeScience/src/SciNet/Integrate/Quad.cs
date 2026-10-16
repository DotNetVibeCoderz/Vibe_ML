namespace Gravicode.Science.SciNet;

/// <summary>The weight functions <see cref="Integrate.Quad(Func{double, double}, double, double, double, double, int, double[], QuadWeight, double[], int, int)"/> accepts (scipy's <c>weight=</c> strings).</summary>
public enum QuadWeight
{
    /// <summary>No weight: QAGS / QAGI / QAGP.</summary>
    None,
    /// <summary>cos(wvar x): QAWO, or QAWF when an end is infinite.</summary>
    Cos,
    /// <summary>sin(wvar x): QAWO, or QAWF when an end is infinite.</summary>
    Sin,
    /// <summary>(x-a)^alpha (b-x)^beta, wvar = (alpha, beta): QAWS.</summary>
    Alg,
    /// <summary>(x-a)^alpha (b-x)^beta log(x-a): QAWS.</summary>
    AlgLogA,
    /// <summary>(x-a)^alpha (b-x)^beta log(b-x): QAWS.</summary>
    AlgLogB,
    /// <summary>(x-a)^alpha (b-x)^beta log(x-a) log(b-x): QAWS.</summary>
    AlgLog,
    /// <summary>1/(x - wvar): Cauchy principal value, QAWC.</summary>
    Cauchy,
}

/// <summary>Result of <see cref="Integrate.Quad(Func{double, double}, double, double, double, double, int, double[], QuadWeight, double[], int, int)"/>, scipy's (y, abserr, infodict, message).</summary>
public sealed record QuadResult
{
    /// <summary>The integral estimate.</summary>
    public double Value { get; init; }
    /// <summary>QUADPACK's estimate of the absolute error.</summary>
    public double AbsErr { get; init; }
    /// <summary>Number of integrand evaluations.</summary>
    public int Neval { get; init; }
    /// <summary>QUADPACK's ier: 0 success; 1 subdivision limit; 2 roundoff; 3 bad integrand; 4 no convergence; 5 divergent; 6 invalid input; 7 (QAWF) cycle trouble.</summary>
    public int Ier { get; init; }
    /// <summary>Human-readable explanation of <see cref="Ier"/>.</summary>
    public string Message { get; init; } = "";
    /// <summary>Number of subintervals used (scipy's infodict['last']).</summary>
    public int Last { get; init; }
    /// <summary>Left ends of the subintervals (infodict['alist']), first <see cref="Last"/> entries.</summary>
    public double[] Alist { get; init; } = [];
    /// <summary>Right ends of the subintervals (infodict['blist']).</summary>
    public double[] Blist { get; init; } = [];
    /// <summary>Integral over each subinterval (infodict['rlist']).</summary>
    public double[] Rlist { get; init; } = [];
    /// <summary>Error estimate of each subinterval (infodict['elist']).</summary>
    public double[] Elist { get; init; } = [];
    /// <summary>QAWF only: integral over each cycle (infodict['rslst']).</summary>
    public double[]? CycleResults { get; init; }
    /// <summary>QAWF only: error of each cycle (infodict['erlst']).</summary>
    public double[]? CycleErrors { get; init; }
    /// <summary>True when <see cref="Ier"/> is 0.</summary>
    public bool Success => Ier == 0;
}

/// <summary>Norm used by <see cref="Integrate.QuadVec"/> to reduce a vector error to a scalar.</summary>
public enum QuadVecNorm
{
    /// <summary>Euclidean norm (scipy's '2', the default).</summary>
    Two,
    /// <summary>Max-abs norm (scipy's 'max').</summary>
    Max,
}

/// <summary>Result of <see cref="Integrate.QuadVec"/>.</summary>
public sealed record QuadVecResult(double[] Value, double AbsErr, int Neval, int Intervals, int Status, string Message)
{
    /// <summary>True when the tolerance was met (status 0).</summary>
    public bool Success => Status == 0;
}

/// <summary>scipy.integrate: quadrature, cubature, ODE initial- and boundary-value solvers.</summary>
public static partial class Integrate
{
    /// <summary>
    /// scipy.integrate.quad: adaptive quadrature via QUADPACK. Dispatch follows scipy exactly:
    /// QAGS for a finite range, QAGI for an infinite one, QAGP with <paramref name="points"/>,
    /// QAWO/QAWF for cos/sin weights, QAWS for algebraic-logarithmic end singularities and QAWC
    /// for a Cauchy principal value. Reversed limits integrate the flipped range and negate,
    /// as scipy does. A non-zero <see cref="QuadResult.Ier"/> is reported, not thrown, because
    /// the value is usually still the best available (scipy warns in the same situation).
    /// </summary>
    /// <param name="f">Integrand.</param>
    /// <param name="a">Lower limit (may be -infinity).</param>
    /// <param name="b">Upper limit (may be +infinity).</param>
    /// <param name="epsabs">Absolute tolerance.</param>
    /// <param name="epsrel">Relative tolerance.</param>
    /// <param name="limit">Maximum number of subintervals.</param>
    /// <param name="points">Break points inside (a, b) (finite ranges, no weight).</param>
    /// <param name="weight">Weight function kind.</param>
    /// <param name="wvar">Weight parameters: omega for cos/sin, (alpha, beta) for algebraic, c for Cauchy.</param>
    /// <param name="maxp1">Maximum number of Chebyshev moment levels (QAWO/QAWF).</param>
    /// <param name="limlst">Maximum number of cycles (QAWF).</param>
    public static QuadResult Quad(Func<double, double> f, double a, double b, double epsabs = 1.49e-8, double epsrel = 1.49e-8,
        int limit = 50, double[]? points = null, QuadWeight weight = QuadWeight.None, double[]? wvar = null, int maxp1 = 50, int limlst = 50)
    {
        ArgumentNullException.ThrowIfNull(f);
        if (double.IsNaN(a) || double.IsNaN(b)) throw new ArgumentException("Integration limits must not be NaN.");
        if (limit < 1) throw new ArgumentException("limit must be at least 1.", nameof(limit));
        var flip = b < a;
        if (flip) (a, b) = (b, a);
        var r = weight == QuadWeight.None ? QuadPlain(f, a, b, epsabs, epsrel, limit, points) : QuadWeighted(f, a, b, epsabs, epsrel, limit, weight, wvar, maxp1, limlst);
        return flip ? r with { Value = -r.Value } : r;
    }

    private static QuadResult QuadPlain(Func<double, double> f, double a, double b, double epsabs, double epsrel, int limit, double[]? points)
    {
        var aInf = double.IsInfinity(a);
        var bInf = double.IsInfinity(b);
        if (points is { Length: > 0 })
        {
            if (aInf || bInf) throw new ArgumentException("Infinity inputs cannot be used with break points.");
            var inner = points.Where(p => p > a && p < b).Distinct().OrderBy(p => p).ToArray();
            var lim = Math.Max(limit, inner.Length + 2);
            var w = new Quadpack.Work(lim);
            var o = Quadpack.Qagpe(f, a, b, inner, epsabs, epsrel, lim, w);
            return Pack(o, w);
        }
        if (!aInf && !bInf)
        {
            var w = new Quadpack.Work(limit);
            return Pack(Quadpack.Qagse(f, a, b, epsabs, epsrel, limit, w), w);
        }
        {
            var w = new Quadpack.Work(limit);
            Quadpack.Output o;
            if (aInf && bInf) o = Quadpack.Qagie(f, 0.0, 2, epsabs, epsrel, limit, w);
            else if (bInf) o = Quadpack.Qagie(f, a, 1, epsabs, epsrel, limit, w);
            else o = Quadpack.Qagie(f, b, -1, epsabs, epsrel, limit, w);
            return Pack(o, w);
        }
    }

    private static QuadResult QuadWeighted(Func<double, double> f, double a, double b, double epsabs, double epsrel, int limit,
        QuadWeight weight, double[]? wvar, int maxp1, int limlst)
    {
        if (wvar is null || wvar.Length == 0) throw new ArgumentException("wvar is required for a weighted integral.", nameof(wvar));
        var aInf = double.IsInfinity(a);
        var bInf = double.IsInfinity(b);
        switch (weight)
        {
            case QuadWeight.Cos:
            case QuadWeight.Sin:
            {
                var integr = weight == QuadWeight.Cos ? 1 : 2;
                var omega = wvar[0];
                if (!aInf && !bInf)
                {
                    var w = new Quadpack.Work(limit);
                    var mc = new Quadpack.MomentCache(maxp1);
                    return Pack(Quadpack.Qawoe(f, a, b, omega, integr, epsabs, epsrel, limit, 1, mc, w), w);
                }
                if (aInf && bInf) throw new ArgumentException("Cannot integrate with a cos/sin weight from -inf to +inf; split the range.");
                // QAWF works on [a, inf). For (-inf, b] substitute x = -t: cos is even, sin odd.
                var g = bInf ? f : (x => f(-x));
                var start = bInf ? a : -b;
                var fo = Quadpack.Qawfe(g, start, omega, integr, epsabs, limlst, limit, maxp1);
                var value = fo.Main.Result;
                if (!bInf && integr == 2) value = -value;
                return new QuadResult
                {
                    Value = value, AbsErr = fo.Main.AbsErr, Neval = fo.Main.Neval, Ier = fo.Main.Ier,
                    Message = MessageFor(fo.Main.Ier, limit), Last = fo.Lst,
                    CycleResults = fo.Rslst.AsSpan(1, Math.Max(fo.Lst, 0)).ToArray(),
                    CycleErrors = fo.Erlst.AsSpan(1, Math.Max(fo.Lst, 0)).ToArray(),
                };
            }
            case QuadWeight.Alg:
            case QuadWeight.AlgLogA:
            case QuadWeight.AlgLogB:
            case QuadWeight.AlgLog:
            {
                if (aInf || bInf) throw new ArgumentException("Algebraic weights need finite limits.");
                if (wvar.Length < 2) throw new ArgumentException("wvar must be (alpha, beta) for algebraic weights.", nameof(wvar));
                var integr = weight switch { QuadWeight.Alg => 1, QuadWeight.AlgLogA => 2, QuadWeight.AlgLogB => 3, _ => 4 };
                var w = new Quadpack.Work(Math.Max(limit, 2));
                return Pack(Quadpack.Qawse(f, a, b, wvar[0], wvar[1], integr, epsabs, epsrel, Math.Max(limit, 2), w), w);
            }
            case QuadWeight.Cauchy:
            {
                if (aInf || bInf) throw new ArgumentException("Cauchy weight needs finite limits.");
                var w = new Quadpack.Work(limit);
                return Pack(Quadpack.Qawce(f, a, b, wvar[0], epsabs, epsrel, limit, w), w);
            }
            default:
                throw new ArgumentException($"Unknown weight {weight}.", nameof(weight));
        }
    }

    private static QuadResult Pack(Quadpack.Output o, Quadpack.Work w)
    {
        var last = Math.Max(w.Last, 0);
        var al = new double[last]; var bl = new double[last]; var rl = new double[last]; var el = new double[last];
        for (var i = 0; i < last; i++)
        {
            al[i] = w.Alist[i + 1]; bl[i] = w.Blist[i + 1]; rl[i] = w.Rlist[i + 1]; el[i] = w.Elist[i + 1];
        }
        return new QuadResult
        {
            Value = o.Result, AbsErr = o.AbsErr, Neval = o.Neval, Ier = o.Ier, Message = MessageFor(o.Ier, w.Alist.Length - 2),
            Last = last, Alist = al, Blist = bl, Rlist = rl, Elist = el,
        };
    }

    private static string MessageFor(int ier, int limit) => ier switch
    {
        0 => "The integral was computed to the requested accuracy.",
        1 => $"The maximum number of subdivisions ({limit}) has been achieved. If increasing the limit yields no improvement it is advised to analyze the integrand in order to determine the difficulties.",
        2 => "The occurrence of roundoff error is detected, which prevents the requested tolerance from being achieved. The error may be underestimated.",
        3 => "Extremely bad integrand behavior occurs at some points of the integration interval.",
        4 => "The algorithm does not converge. Roundoff error is detected in the extrapolation table. It is assumed that the requested tolerance cannot be achieved.",
        5 => "The integral is probably divergent, or slowly convergent.",
        6 => "The input is invalid.",
        7 => "Bad integrand behavior occurs within one or more of the cycles. Location and type of the difficulty involved can be determined from the cycle error flags.",
        _ => "Unknown error.",
    };

    // ------------------------------------------------------------------ multiple integrals

    /// <summary>
    /// scipy.integrate.dblquad: integral of f(y, x) for x in [a, b], y in [gfun(x), hfun(x)].
    /// Note the argument order f(y, x): the inner variable comes first, as in scipy.
    /// </summary>
    public static (double Value, double AbsErr) Dblquad(Func<double, double, double> f, double a, double b,
        Func<double, double> gfun, Func<double, double> hfun, double epsabs = 1.49e-8, double epsrel = 1.49e-8)
    {
        var r = Quad(x => Quad(y => f(y, x), gfun(x), hfun(x), epsabs, epsrel).Value, a, b, epsabs, epsrel);
        return (r.Value, r.AbsErr);
    }

    /// <summary>Dblquad with constant inner limits.</summary>
    public static (double Value, double AbsErr) Dblquad(Func<double, double, double> f, double a, double b, double c, double d,
        double epsabs = 1.49e-8, double epsrel = 1.49e-8) => Dblquad(f, a, b, _ => c, _ => d, epsabs, epsrel);

    /// <summary>
    /// scipy.integrate.tplquad: integral of f(z, y, x) for x in [a, b], y in [gfun(x), hfun(x)],
    /// z in [qfun(x, y), rfun(x, y)].
    /// </summary>
    public static (double Value, double AbsErr) Tplquad(Func<double, double, double, double> f, double a, double b,
        Func<double, double> gfun, Func<double, double> hfun, Func<double, double, double> qfun, Func<double, double, double> rfun,
        double epsabs = 1.49e-8, double epsrel = 1.49e-8)
    {
        var r = Quad(x => Quad(y => Quad(z => f(z, y, x), qfun(x, y), rfun(x, y), epsabs, epsrel).Value,
            gfun(x), hfun(x), epsabs, epsrel).Value, a, b, epsabs, epsrel);
        return (r.Value, r.AbsErr);
    }

    /// <summary>
    /// scipy.integrate.nquad. <paramref name="ranges"/>[i] gives the limits of x_i as a function of the
    /// outer variables (x_{i+1}, ..., x_{n-1}); ranges[0] is the innermost integral, matching scipy.
    /// The returned error is the outermost QUADPACK estimate, as in scipy.
    /// </summary>
    public static (double Value, double AbsErr) Nquad(Func<double[], double> f, IReadOnlyList<Func<double[], (double Lo, double Hi)>> ranges,
        double epsabs = 1.49e-8, double epsrel = 1.49e-8, int limit = 50)
    {
        ArgumentNullException.ThrowIfNull(f);
        var n = ranges.Count;
        if (n == 0) throw new ArgumentException("At least one range is required.", nameof(ranges));
        var x = new double[n];
        QuadResult Level(int i)
        {
            var outer = new double[n - 1 - i];
            Array.Copy(x, i + 1, outer, 0, outer.Length);
            var (lo, hi) = ranges[i](outer);
            return Quad(t =>
            {
                x[i] = t;
                return i == 0 ? f((double[])x.Clone()) : Level(i - 1).Value;
            }, lo, hi, epsabs, epsrel, limit);
        }
        var r = Level(n - 1);
        return (r.Value, r.AbsErr);
    }

    /// <summary>Nquad over a fixed hyper-rectangle; ranges[0] belongs to the first argument.</summary>
    public static (double Value, double AbsErr) Nquad(Func<double[], double> f, IReadOnlyList<(double Lo, double Hi)> ranges,
        double epsabs = 1.49e-8, double epsrel = 1.49e-8, int limit = 50)
        => Nquad(f, ranges.Select(r => (Func<double[], (double, double)>)(_ => r)).ToArray(), epsabs, epsrel, limit);

    // ------------------------------------------------------------------ quad_vec

    /// <summary>
    /// scipy.integrate.quad_vec: adaptive Gauss-Kronrod for a vector-valued integrand, bisecting
    /// the worst interval from a global heap so that every component shares one mesh. Infinite
    /// limits are mapped to finite ones with scipy's substitutions. The error per interval is
    /// QUADPACK's rescaled |K - G| under the chosen norm; convergence is declared, as in scipy,
    /// when the summed error falls below tol/8.
    /// </summary>
    public static QuadVecResult QuadVec(Func<double, double[]> f, double a, double b, double epsabs = 1e-200, double epsrel = 1e-8,
        QuadVecNorm norm = QuadVecNorm.Two, int limit = 10000, double[]? points = null, bool useGk15 = false)
    {
        ArgumentNullException.ThrowIfNull(f);
        var sign = 1.0;
        if (b < a) { (a, b) = (b, a); sign = -1.0; }
        Func<double, double[]> g = f;
        double lo = a, hi = b;
        double[]? pts = points;
        if (double.IsNegativeInfinity(a) && double.IsPositiveInfinity(b))
        {
            g = t => Scale(f(t / (1 - t * t)), (1 + t * t) / ((1 - t * t) * (1 - t * t)));
            lo = -1; hi = 1;
            pts = points?.Select(p => p == 0 ? 0 : (-1 + Math.Sqrt(1 + 4 * p * p)) / (2 * p)).ToArray();
        }
        else if (double.IsPositiveInfinity(b))
        {
            g = t => Scale(f(a + t / (1 - t)), 1.0 / ((1 - t) * (1 - t)));
            lo = 0; hi = 1;
            pts = points?.Select(p => (p - a) / (p - a + 1)).ToArray();
        }
        else if (double.IsNegativeInfinity(a))
        {
            g = t => Scale(f(b - t / (1 - t)), 1.0 / ((1 - t) * (1 - t)));
            lo = 0; hi = 1;
            pts = points?.Select(p => (b - p) / (b - p + 1)).ToArray();
        }
        var breaks = new List<double> { lo };
        if (pts != null) breaks.AddRange(pts.Where(p => p > lo && p < hi).Distinct().OrderBy(p => p));
        breaks.Add(hi);

        var heap = new PriorityQueue<(double A, double B, double[] V, double E), double>();
        double[]? total = null;
        double errSum = 0, roundSum = 0;
        var neval = 0;
        for (var i = 0; i + 1 < breaks.Count; i++)
        {
            var (v, e, rnd, ne) = GkVec(g, breaks[i], breaks[i + 1], norm, useGk15);
            neval += ne;
            total = total is null ? (double[])v.Clone() : Add(total, v, 1);
            errSum += e; roundSum += rnd;
            heap.Enqueue((breaks[i], breaks[i + 1], v, e), -e);
        }
        var status = 1;
        var minIntervals = 2 * (breaks.Count - 1);
        while (heap.Count > 0)
        {
            var tol = Math.Max(epsabs, epsrel * Norm(total!, norm));
            if (heap.Count >= minIntervals || heap.Count >= limit)
            {
                if (errSum < tol / 8) { status = 0; break; }
                if (errSum < roundSum) { status = 2; break; }
            }
            if (heap.Count >= limit) { status = 1; break; }
            var (ia, ib, iv, ie) = heap.Dequeue();
            var mid = 0.5 * (ia + ib);
            var (v1, e1, r1, n1) = GkVec(g, ia, mid, norm, useGk15);
            var (v2, e2, r2, n2) = GkVec(g, mid, ib, norm, useGk15);
            neval += n1 + n2;
            for (var k = 0; k < total!.Length; k++) total[k] += v1[k] + v2[k] - iv[k];
            errSum += e1 + e2 - ie;
            roundSum += r1 + r2;
            heap.Enqueue((ia, mid, v1, e1), -e1);
            heap.Enqueue((mid, ib, v2, e2), -e2);
        }
        // Re-sum from the intervals: the running total accumulates cancellation error.
        var res = new double[total!.Length];
        double err = 0;
        foreach (var (_, _, v, e) in heap.UnorderedItems.Select(u => u.Element))
        {
            for (var k = 0; k < res.Length; k++) res[k] += v[k];
            err += e;
        }
        for (var k = 0; k < res.Length; k++) res[k] *= sign;
        var msg = status switch
        {
            0 => "Target precision reached.",
            1 => "Maximum number of subdivisions reached.",
            _ => "Rounding error dominates; the requested tolerance cannot be met.",
        };
        return new QuadVecResult(res, err, neval, heap.Count, status, msg);
    }

    private static double[] Scale(double[] v, double s)
    {
        var r = new double[v.Length];
        for (var i = 0; i < v.Length; i++) r[i] = double.IsFinite(s) ? v[i] * s : 0.0;
        return r;
    }

    private static double[] Add(double[] a, double[] b, double s)
    {
        for (var i = 0; i < a.Length; i++) a[i] += s * b[i];
        return a;
    }

    private static double Norm(double[] v, QuadVecNorm norm)
    {
        double s = 0;
        if (norm == QuadVecNorm.Max)
        {
            foreach (var x in v) s = Math.Max(s, Math.Abs(x));
            return s;
        }
        foreach (var x in v) s += x * x;
        return Math.Sqrt(s);
    }

    private static (double[] V, double Err, double Round, int Neval) GkVec(Func<double, double[]> f, double a, double b, QuadVecNorm norm, bool gk15)
    {
        var xk = gk15 ? IntegrateRules.Xgk15 : IntegrateRules.Xgk21;
        var wk = gk15 ? IntegrateRules.Wgk15 : IntegrateRules.Wgk21;
        var nh = xk.Length - 1; // nodes per side
        var c = 0.5 * (a + b);
        var h = 0.5 * (b - a);
        var total = 2 * nh + 1;
        var fv = new double[total][];
        // Node order: -x0..-x_{nh-1}, centre, +x_{nh-1}..+x0 is irrelevant; keep side arrays.
        var fc = Endpoint(f(c));
        var m = fc.Length;
        var sk = new double[m]; var sg = new double[m]; var skAbs = new double[m];
        for (var k = 0; k < m; k++) { sk[k] = wk[nh] * fc[k]; skAbs[k] = Math.Abs(sk[k]); }
        if (gk15) for (var k = 0; k < m; k++) sg[k] = IntegrateRules.Wg7[3] * fc[k];
        fv[2 * nh] = fc;
        for (var j = 0; j < nh; j++)
        {
            var f1 = Endpoint(f(c - h * xk[j]));
            var f2 = Endpoint(f(c + h * xk[j]));
            if (f1.Length != m || f2.Length != m) throw new ArgumentException("The integrand must return vectors of a fixed length.");
            fv[2 * j] = f1; fv[2 * j + 1] = f2;
            var gw = j % 2 == 1 ? (gk15 ? IntegrateRules.Wg7[j / 2] : IntegrateRules.Wg10[j / 2]) : 0.0;
            for (var k = 0; k < m; k++)
            {
                var s = f1[k] + f2[k];
                sk[k] += wk[j] * s;
                skAbs[k] += wk[j] * (Math.Abs(f1[k]) + Math.Abs(f2[k]));
                sg[k] += gw * s;
            }
        }
        var dabsV = new double[m];
        var diff = new double[m];
        var rnd = new double[m];
        for (var k = 0; k < m; k++)
        {
            var y0 = sk[k] / 2;
            var s = wk[nh] * Math.Abs(fc[k] - y0);
            for (var j = 0; j < nh; j++) s += wk[j] * (Math.Abs(fv[2 * j][k] - y0) + Math.Abs(fv[2 * j + 1][k] - y0));
            dabsV[k] = s * Math.Abs(h);
            diff[k] = (sk[k] - sg[k]) * h;
            rnd[k] = 50 * Quadpack.Epmach * Math.Abs(h) * skAbs[k];
        }
        var err = Norm(diff, norm);
        var dabs = Norm(dabsV, norm);
        if (dabs != 0 && err != 0) err = dabs * Math.Min(1.0, Math.Pow(200 * err / dabs, 1.5));
        var round = Norm(rnd, norm);
        if (round > double.Epsilon) err = Math.Max(err, round);
        var val = new double[m];
        for (var k = 0; k < m; k++) val[k] = h * sk[k];
        return (val, err, round, total);
    }

    private static double[] Endpoint(double[] v) => v ?? throw new ArgumentException("The integrand returned null.");
}
