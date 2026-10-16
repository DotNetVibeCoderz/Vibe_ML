using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Result of <see cref="Integrate.TanhSinh"/> (scipy's _RichResult fields).</summary>
public sealed record TanhSinhResult(double Integral, double Error, int Nfev, int MaxLevel, int Status)
{
    /// <summary>True when the tolerance was met (status 0). -2: level limit; -3: non-finite value.</summary>
    public bool Success => Status == 0;
}

/// <summary>Cubature rule for <see cref="Integrate.Cubature"/>.</summary>
public enum CubatureRule
{
    /// <summary>Tensor-product 21-point Gauss-Kronrod with the 10-point Gauss product as error estimate (scipy 'gk21').</summary>
    GaussKronrod21,
    /// <summary>Tensor-product 15-point Gauss-Kronrod (scipy 'gk15').</summary>
    GaussKronrod15,
    /// <summary>Genz-Malik degree-7 rule with embedded degree 5 (scipy 'genz-malik'), 2^n + 2n^2 + 2n + 1 points; n ≥ 2.</summary>
    GenzMalik,
}

/// <summary>Result of <see cref="Integrate.Cubature"/>.</summary>
public sealed record CubatureResult(double[] Estimate, double[] Error, bool Converged, int Subdivisions, int Regions, int Nfev)
{
    /// <summary>scipy's status string: "converged" or "not_converged".</summary>
    public string Status => Converged ? "converged" : "not_converged";
}

/// <summary>Result of <see cref="Integrate.QmcQuad"/>.</summary>
public sealed record QmcQuadResult(double Integral, double StandardError);

public static partial class Integrate
{
    /// <summary>
    /// scipy.integrate.tanhsinh: double-exponential quadrature (Takahasi-Mori, error estimate after
    /// Bailey). Nodes crowd doubly-exponentially towards the ends, so integrable endpoint
    /// singularities (log, x^-1/2, ...) converge as fast as smooth integrands. Abscissae near an end
    /// are formed from their distance to it (the complement), not as tanh(...) close to ±1, which is
    /// what lets the rule sample 1e-300 away from a singular endpoint. Infinite limits are mapped
    /// to (-1, 1) by rational substitutions. Not a port of scipy's code: the level scheme and
    /// Bailey's estimate are the same, but values differ in the last digits.
    /// </summary>
    /// <param name="f">Integrand.</param>
    /// <param name="a">Lower limit (may be -infinity).</param>
    /// <param name="b">Upper limit (may be +infinity).</param>
    /// <param name="atol">Absolute tolerance (default 0).</param>
    /// <param name="rtol">Relative tolerance (default eps^0.75, as scipy).</param>
    /// <param name="maxlevel">Finest level (step 2^-maxlevel); default 10 as scipy.</param>
    /// <param name="minlevel">Level at which the error test may first stop (default 2).</param>
    public static TanhSinhResult TanhSinh(Func<double, double> f, double a, double b, double atol = 0.0, double? rtol = null, int maxlevel = 10, int minlevel = 2)
    {
        ArgumentNullException.ThrowIfNull(f);
        if (double.IsNaN(a) || double.IsNaN(b)) throw new ArgumentException("Limits must not be NaN.");
        if (a == b) return new TanhSinhResult(0, 0, 0, 0, 0);
        var sign = 1.0;
        if (b < a) { (a, b) = (b, a); sign = -1.0; }
        var rt = rtol ?? Math.Pow(2.220446049250313e-16, 0.75);
        // g(side, delta): f(x(u)) dx/du at u = side * (1 - delta); side 0 means u = 0.
        Func<int, double, double> g;
        var aInf = double.IsNegativeInfinity(a);
        var bInf = double.IsPositiveInfinity(b);
        if (!aInf && !bInf)
        {
            var c = 0.5 * (a + b); var h = 0.5 * (b - a);
            g = (side, d) => side == 0 ? h * f(c) : h * f(side > 0 ? b - h * d : a + h * d);
        }
        else if (!aInf)
        {
            g = (side, d) => side switch
            {
                0 => 2 * f(a + 1),
                > 0 => 2 / (d * d) * f(a + (2 - d) / d),
                _ => 2 / ((2 - d) * (2 - d)) * f(a + d / (2 - d)),
            };
        }
        else if (!bInf)
        {
            g = (side, d) => side switch
            {
                0 => 2 * f(b - 1),
                > 0 => 2 / (d * d) * f(b - (2 - d) / d),
                _ => 2 / ((2 - d) * (2 - d)) * f(b - d / (2 - d)),
            };
        }
        else
        {
            g = (side, d) =>
            {
                if (side == 0) return f(0);
                var u = 1 - d;
                var omu2 = d * (2 - d);
                var x = u / omu2;
                return (1 + u * u) / (omu2 * omu2) * f(side > 0 ? x : -x);
            };
        }

        const double tmax = 6.5;
        var nfev = 0;
        double maxTerm = 0, edgeTerm = 0;
        var status = 0;
        double Term(double t, int side)
        {
            var s = Math.PI / 2 * Math.Sinh(t);
            var cs = Math.Cosh(s);
            var w = Math.PI / 2 * Math.Cosh(t) / (cs * cs);
            var delta = 2.0 / (Math.Exp(2 * s) + 1.0); // 1 - tanh(s)
            if (w == 0 || delta == 0) return 0;
            var v = g(side, delta);
            nfev++;
            var term = w * v;
            if (!double.IsFinite(term))
            {
                // A non-finite value right at the end is the singularity itself; anywhere else it is an error.
                if (delta < 1e-12) return 0;
                status = -3;
                return 0;
            }
            var at = Math.Abs(term);
            if (at > maxTerm) maxTerm = at;
            return term;
        }

        var levels = new List<double>();
        var h0 = 1.0;
        // Level 0: t = 0, ±1, ±2, ... up to tmax.
        var sum = Term(0, 0);
        for (var k = 1; k * h0 <= tmax; k++) sum += Term(k, +1) + Term(k, -1);
        var est = sum * h0;
        levels.Add(est);
        var err = double.NaN;
        var level = 0;
        for (level = 1; level <= maxlevel; level++)
        {
            var hl = Math.Pow(2, -level);
            var add = 0.0;
            edgeTerm = 0;
            var kmax = (int)(tmax / hl);
            for (var k = 1; k <= kmax; k += 2)
            {
                var t = k * hl;
                var tp = Term(t, +1);
                var tm = Term(t, -1);
                add += tp + tm;
                if (k + 2 > kmax) edgeTerm = Math.Max(Math.Abs(tp), Math.Abs(tm));
            }
            sum += add;
            est = sum * hl;
            levels.Add(est);
            if (status != 0) break;
            if (levels.Count >= 3)
            {
                // Bailey's estimate, in relative terms so it does not depend on the integral's scale:
                // with d_i = log10 of the relative changes over one and two levels, the error is taken
                // as 10^max(d1^2/d2, 2 d1, d3, d4), d3 the rounding floor and d4 the size of the
                // outermost terms (a truncation floor). It presumes quadratic convergence, so while
                // the change over one level is still large the plain change e1 is reported instead.
                var e1 = Math.Abs(est - levels[^2]);
                var e2 = Math.Abs(est - levels[^3]);
                const double eps = 2.220446049250313e-16;
                var scale = Math.Abs(est) > 0 ? Math.Abs(est) : 1.0;
                if (e1 == 0) err = eps * Math.Max(Math.Abs(est), maxTerm);
                else
                {
                    var d1 = Math.Log10(e1 / scale);
                    var d2 = e2 == 0 ? d1 : Math.Log10(e2 / scale);
                    var d3 = Math.Log10(eps * maxTerm / scale + double.Epsilon);
                    var d4 = Math.Log10(edgeTerm / scale + double.Epsilon);
                    if (d1 >= -1 || d2 >= 0) err = e1;
                    else
                    {
                        var de = Math.Max(Math.Max(d1 * d1 / d2, 2 * d1), Math.Max(d3, d4));
                        err = Math.Pow(10, de) * scale;
                    }
                }
                if (level >= minlevel && err <= Math.Max(atol, rt * Math.Abs(est))) break;
            }
        }
        if (status == 0 && level > maxlevel) { status = -2; level = maxlevel; }
        return new TanhSinhResult(sign * est, err, nfev, level, status);
    }

    /// <summary>
    /// scipy.integrate.cubature: globally adaptive cubature of a vector-valued integrand over a
    /// hyper-rectangle. The region with the largest error is halved in every dimension (2^n
    /// children) until each component satisfies error ≤ atol + rtol·|estimate|. Infinite limits
    /// are mapped per dimension (x = t/(1-t²) on (-1, 1), x = a + t/(1-t) on [0, 1)).
    /// </summary>
    public static CubatureResult Cubature(Func<double[], double[]> f, double[] a, double[] b, CubatureRule rule = CubatureRule.GaussKronrod21,
        double rtol = 1e-8, double atol = 0.0, int maxSubdivisions = 10000)
    {
        ArgumentNullException.ThrowIfNull(f);
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        var n = a.Length;
        if (b.Length != n || n == 0) throw new ArgumentException("a and b must be non-empty and of equal length.");
        if (rule == CubatureRule.GenzMalik && n < 2) throw new ArgumentException("Genz-Malik cubature needs at least 2 dimensions.");
        // Per-dimension maps from the working box to the original variables.
        var lo = new double[n]; var hi = new double[n];
        var kind = new int[n]; var sgn = new double[n];
        var offs = new double[n];
        for (var i = 0; i < n; i++)
        {
            double ai = a[i], bi = b[i];
            sgn[i] = 1;
            if (bi < ai) { (ai, bi) = (bi, ai); sgn[i] = -1; }
            if (double.IsNegativeInfinity(ai) && double.IsPositiveInfinity(bi)) { kind[i] = 1; lo[i] = -1; hi[i] = 1; }
            else if (double.IsPositiveInfinity(bi)) { kind[i] = 2; offs[i] = ai; lo[i] = 0; hi[i] = 1; }
            else if (double.IsNegativeInfinity(ai)) { kind[i] = 3; offs[i] = bi; lo[i] = 0; hi[i] = 1; }
            else { lo[i] = ai; hi[i] = bi; }
        }
        var totalSign = sgn.Aggregate(1.0, (p, s) => p * s);
        var xbuf = new double[n];
        var nfev = 0;
        double[] G(double[] t)
        {
            var jac = 1.0;
            for (var i = 0; i < n; i++)
            {
                var ti = t[i];
                switch (kind[i])
                {
                    case 1: { var d = 1 - ti * ti; xbuf[i] = ti / d; jac *= (1 + ti * ti) / (d * d); break; }
                    case 2: { var d = 1 - ti; xbuf[i] = offs[i] + ti / d; jac *= 1 / (d * d); break; }
                    case 3: { var d = 1 - ti; xbuf[i] = offs[i] - ti / d; jac *= 1 / (d * d); break; }
                    default: xbuf[i] = ti; break;
                }
            }
            nfev++;
            var v = f((double[])xbuf.Clone());
            if (jac == 1.0) return v;
            var r = new double[v.Length];
            for (var k = 0; k < v.Length; k++) r[k] = v[k] * jac;
            return r;
        }

        var heap = new PriorityQueue<(double[] Lo, double[] Hi, double[] Est, double[] Err), double>();
        var (e0, r0) = CubatureRegion(G, lo, hi, rule);
        var est = (double[])e0.Clone();
        var err = (double[])r0.Clone();
        heap.Enqueue((lo, hi, e0, r0), -r0.Max());
        var subdivisions = 0;
        var converged = false;
        while (true)
        {
            var ok = true;
            for (var k = 0; k < est.Length; k++)
                if (!(err[k] <= atol + rtol * Math.Abs(est[k]))) { ok = false; break; }
            if (ok) { converged = true; break; }
            if (subdivisions >= maxSubdivisions) break;
            var (rl, rh, re, rr) = heap.Dequeue();
            for (var k = 0; k < est.Length; k++) { est[k] -= re[k]; err[k] -= rr[k]; }
            var mid = new double[n];
            for (var i = 0; i < n; i++) mid[i] = 0.5 * (rl[i] + rh[i]);
            for (var mask = 0; mask < 1 << n; mask++)
            {
                var cl = new double[n]; var ch = new double[n];
                for (var i = 0; i < n; i++)
                {
                    if ((mask >> i & 1) == 0) { cl[i] = rl[i]; ch[i] = mid[i]; }
                    else { cl[i] = mid[i]; ch[i] = rh[i]; }
                }
                var (ce, cr) = CubatureRegion(G, cl, ch, rule);
                for (var k = 0; k < est.Length; k++) { est[k] += ce[k]; err[k] += cr[k]; }
                heap.Enqueue((cl, ch, ce, cr), -cr.Max());
            }
            subdivisions++;
        }
        // Re-sum to shed the running-sum cancellation.
        var fe = new double[est.Length]; var fr = new double[est.Length];
        foreach (var (rl, rh, re, rr) in heap.UnorderedItems.Select(u => u.Element))
            for (var k = 0; k < fe.Length; k++) { fe[k] += re[k]; fr[k] += rr[k]; }
        for (var k = 0; k < fe.Length; k++) fe[k] *= totalSign;
        return new CubatureResult(fe, fr, converged, subdivisions, heap.Count, nfev);
    }

    private static (double[] Est, double[] Err) CubatureRegion(Func<double[], double[]> g, double[] lo, double[] hi, CubatureRule rule)
    {
        var n = lo.Length;
        if (rule == CubatureRule.GenzMalik) return GenzMalik(g, lo, hi);
        var xk = rule == CubatureRule.GaussKronrod21 ? IntegrateRules.Xgk21 : IntegrateRules.Xgk15;
        var wk = rule == CubatureRule.GaussKronrod21 ? IntegrateRules.Wgk21 : IntegrateRules.Wgk15;
        // Full 1-D node list (ascending) with Kronrod and Gauss weights.
        var nh = xk.Length - 1;
        var m = 2 * nh + 1;
        var nodes = new double[m]; var wK = new double[m]; var wG = new double[m];
        for (var j = 0; j < nh; j++)
        {
            nodes[j] = -xk[j]; nodes[m - 1 - j] = xk[j];
            wK[j] = wK[m - 1 - j] = wk[j];
            double gw = 0;
            if (j % 2 == 1) gw = rule == CubatureRule.GaussKronrod21 ? IntegrateRules.Wg10[j / 2] : IntegrateRules.Wg7[j / 2];
            wG[j] = wG[m - 1 - j] = gw;
        }
        nodes[nh] = 0; wK[nh] = wk[nh];
        wG[nh] = rule == CubatureRule.GaussKronrod15 ? IntegrateRules.Wg7[3] : 0.0;
        var c = new double[n]; var h = new double[n];
        var vol = 1.0;
        for (var i = 0; i < n; i++) { c[i] = 0.5 * (lo[i] + hi[i]); h[i] = 0.5 * (hi[i] - lo[i]); vol *= h[i]; }
        double[]? sk = null, sg = null;
        var idx = new int[n];
        var x = new double[n];
        while (true)
        {
            double wkp = 1, wgp = 1;
            for (var i = 0; i < n; i++) { x[i] = c[i] + h[i] * nodes[idx[i]]; wkp *= wK[idx[i]]; wgp *= wG[idx[i]]; }
            var v = g(x);
            sk ??= new double[v.Length];
            sg ??= new double[v.Length];
            for (var k = 0; k < v.Length; k++) { sk[k] += wkp * v[k]; sg[k] += wgp * v[k]; }
            var d = 0;
            while (d < n && ++idx[d] == m) { idx[d] = 0; d++; }
            if (d == n) break;
        }
        var est = new double[sk!.Length]; var err = new double[sk.Length];
        for (var k = 0; k < est.Length; k++) { est[k] = vol * sk[k]; err[k] = Math.Abs(vol * (sk[k] - sg![k])); }
        return (est, err);
    }

    private static (double[] Est, double[] Err) GenzMalik(Func<double[], double[]> g, double[] lo, double[] hi)
    {
        var n = lo.Length;
        double l2 = Math.Sqrt(9.0 / 70), l3 = Math.Sqrt(9.0 / 10), l4 = Math.Sqrt(9.0 / 10), l5 = Math.Sqrt(9.0 / 19);
        double w1 = (12824 - 9120.0 * n + 400.0 * n * n) / 19683, w2 = 980.0 / 6561, w3 = (1820 - 400.0 * n) / 19683,
            w4 = 200.0 / 19683, w5 = 6859.0 / 19683 / Math.Pow(2, n);
        double v1 = (729 - 950.0 * n + 50.0 * n * n) / 729, v2 = 245.0 / 486, v3 = (265 - 100.0 * n) / 1458, v4 = 25.0 / 729;
        var c = new double[n]; var h = new double[n];
        var vol = 1.0;
        for (var i = 0; i < n; i++) { c[i] = 0.5 * (lo[i] + hi[i]); h[i] = 0.5 * (hi[i] - lo[i]); vol *= hi[i] - lo[i]; }
        var x = new double[n];
        double[] Eval(Action<double[]> set)
        {
            Array.Copy(c, x, n);
            set(x);
            return g(x);
        }
        var fc = Eval(_ => { });
        var m = fc.Length;
        var s2 = new double[m]; var s3 = new double[m]; var s4 = new double[m]; var s5 = new double[m];
        for (var i = 0; i < n; i++)
        {
            var ii = i;
            var p2 = Eval(z => z[ii] += h[ii] * l2); var m2 = Eval(z => z[ii] -= h[ii] * l2);
            var p3 = Eval(z => z[ii] += h[ii] * l3); var m3 = Eval(z => z[ii] -= h[ii] * l3);
            for (var k = 0; k < m; k++) { s2[k] += p2[k] + m2[k]; s3[k] += p3[k] + m3[k]; }
        }
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
                for (var sa = -1; sa <= 1; sa += 2)
                    for (var sb = -1; sb <= 1; sb += 2)
                    {
                        int ii = i, jj = j, aa = sa, bb = sb;
                        var v = Eval(z => { z[ii] += aa * h[ii] * l4; z[jj] += bb * h[jj] * l4; });
                        for (var k = 0; k < m; k++) s4[k] += v[k];
                    }
        for (var mask = 0; mask < 1 << n; mask++)
        {
            var mm = mask;
            var v = Eval(z => { for (var i = 0; i < n; i++) z[i] += ((mm >> i & 1) == 0 ? -1 : 1) * h[i] * l5; });
            for (var k = 0; k < m; k++) s5[k] += v[k];
        }
        var est = new double[m]; var err = new double[m];
        for (var k = 0; k < m; k++)
        {
            var i7 = w1 * fc[k] + w2 * s2[k] + w3 * s3[k] + w4 * s4[k] + w5 * s5[k];
            var i5 = v1 * fc[k] + v2 * s2[k] + v3 * s3[k] + v4 * s4[k];
            est[k] = vol * i7;
            err[k] = Math.Abs(vol * (i7 - i5));
        }
        return (est, err);
    }

    /// <summary>
    /// scipy.integrate.qmc_quad: quasi-Monte Carlo integration over a box with a randomised
    /// (digit-permuted) Halton sequence. Each of <paramref name="nEstimates"/> independent
    /// randomisations gives volume × mean(f); the result is their mean and standard error
    /// (std with ddof 1 over sqrt(nEstimates)). The sequence is internal and not scipy's bit stream,
    /// so values agree with scipy statistically, not digit for digit.
    /// </summary>
    public static QmcQuadResult QmcQuad(Func<double[], double> f, double[] a, double[] b, int nPoints = 1024, int nEstimates = 8, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(f);
        var d = a.Length;
        if (b.Length != d || d == 0) throw new ArgumentException("a and b must be non-empty and of equal length.");
        if (nEstimates < 2) throw new ArgumentException("nEstimates must be at least 2 for a standard error.", nameof(nEstimates));
        var rng = seed.HasValue ? new GraviRandom(seed.Value) : new GraviRandom((ulong)Environment.TickCount64);
        var vol = 1.0;
        for (var i = 0; i < d; i++) vol *= b[i] - a[i];
        var estimates = new double[nEstimates];
        var halton = new IntegrateScrambledHalton(d);
        var u = new double[d];
        var x = new double[d];
        for (var e = 0; e < nEstimates; e++)
        {
            halton.Randomize(rng);
            var s = 0.0;
            for (var k = 0; k < nPoints; k++)
            {
                halton.Point(k, u);
                for (var i = 0; i < d; i++) x[i] = a[i] + (b[i] - a[i]) * u[i];
                s += f(x);
            }
            estimates[e] = vol * s / nPoints;
        }
        var mean = estimates.Average();
        var ss = estimates.Sum(v => (v - mean) * (v - mean));
        var se = Math.Sqrt(ss / (nEstimates - 1)) / Math.Sqrt(nEstimates);
        return new QmcQuadResult(mean, se);
    }
}

/// <summary>
/// Halton sequence with an independent random digit permutation per (dimension, digit position),
/// truncated to enough digits to exhaust double precision. Permuting every digit position (not
/// only the leading ones) is what makes each randomisation uniformly distributed, which is the
/// property the standard error of <see cref="Integrate.QmcQuad"/> relies on.
/// </summary>
internal sealed class IntegrateScrambledHalton
{
    private readonly int _d;
    private readonly int[] _bases;
    private readonly int[] _digits;
    private readonly int[][][] _perm;

    internal IntegrateScrambledHalton(int d)
    {
        _d = d;
        _bases = new int[d];
        var p = 2;
        for (var i = 0; i < d; i++)
        {
            while (!IsPrime(p)) p++;
            _bases[i] = p++;
        }
        _digits = _bases.Select(bb => (int)Math.Ceiling(53 * Math.Log(2) / Math.Log(bb)) + 1).ToArray();
        _perm = new int[d][][];
        for (var i = 0; i < d; i++)
        {
            _perm[i] = new int[_digits[i]][];
            for (var k = 0; k < _digits[i]; k++) _perm[i][k] = Enumerable.Range(0, _bases[i]).ToArray();
        }
    }

    private static bool IsPrime(int n)
    {
        if (n < 2) return false;
        for (var q = 2; q * q <= n; q++) if (n % q == 0) return false;
        return true;
    }

    internal void Randomize(GraviRandom rng)
    {
        foreach (var dim in _perm)
            foreach (var pm in dim)
                for (var i = pm.Length - 1; i > 0; i--)
                {
                    var j = rng.Next(i + 1);
                    (pm[i], pm[j]) = (pm[j], pm[i]);
                }
    }

    internal void Point(int index, Span<double> u)
    {
        for (var i = 0; i < _d; i++)
        {
            var bb = _bases[i];
            var inv = 1.0 / bb;
            var f = inv;
            var k = index;
            var v = 0.0;
            for (var digit = 0; digit < _digits[i]; digit++)
            {
                v += _perm[i][digit][k % bb] * f;
                k /= bb;
                f *= inv;
            }
            u[i] = Math.Min(v, 1.0 - 1e-16);
        }
    }
}
