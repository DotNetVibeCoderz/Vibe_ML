namespace Gravicode.Science.SciNet;

/// <summary>Outcome of a FITPACK-style curve fit.</summary>
internal sealed class FitpackCurveResult
{
    public required double[] T;
    /// <summary>Coefficients (n, idim) row-major, padded with zero rows to n = len(T).</summary>
    public required double[] C;
    public required int K;
    public required int Idim;
    public double Fp;
    public int Ier;
}

/// <summary>
/// A port of the univariate part of Dierckx's FITPACK (<c>fpcurf</c>/<c>fppara</c> with
/// <c>fpbspl</c>, <c>fpgivs</c>, <c>fprota</c>, <c>fpback</c>, <c>fpdisc</c>,
/// <c>fpknot</c>, <c>fprati</c>), written 1-based internally so it can be checked line by
/// line against the Fortran.
/// </summary>
/// <remarks>
/// <para>The knot search adds knots where the residual sum per knot interval is largest, at
/// the middle data point of that interval; the number of knots added per round is
/// extrapolated from the last decrease of the residual. Once the knots are fixed, the
/// smoothing parameter p of <c>fp(p) = s</c> is found by Dierckx's rational interpolation
/// (<c>fprati</c>), rotating the k-th derivative jump rows, weighted 1/p, into the
/// triangularised observation matrix with Givens rotations.</para>
/// <para><c>reproMode</c> reproduces scipy's pure-Python <c>make_splrep</c>, which refits the
/// least-squares spline between knot insertions (FITPACK only splits the interval sums) and
/// carries a different bracket update when p is too small. The two usually give identical
/// knots, but not always.</para>
/// </remarks>
internal static class InterpolateFitpack
{
    public const double Tol = 0.001;
    public const int MaxIt = 20;

    public static string Message(int ier) => ier switch
    {
        0 => "The spline has a residual sum of squares fp such that abs(fp-s)/s<=0.001",
        -1 => "The spline is an interpolating spline (fp=0)",
        -2 => "The spline is weighted least-squares polynomial of degree k. fp gives the upper bound fp0 for the smoothing factor s",
        1 => "The required storage space exceeds the available storage space. Probable causes: data (x,y) size is too small or smoothing parameter s is too small (fp>s).",
        2 => "A theoretically impossible result when finding a smoothing spline with fp = s. Probable cause: s too small. (abs(fp-s)/s>0.001)",
        3 => "The maximal number of iterations (20) allowed for finding smoothing spline with fp=s has been reached. Probable cause: s too small. (abs(fp-s)/s>0.001)",
        10 => "Error on input data",
        _ => "An error occurred",
    };

    // ------------------------------------------------------------- primitives

    /// <summary>fpgivs: Givens parameters annihilating piv against ww; ww becomes the norm.</summary>
    public static void Fpgivs(double piv, ref double ww, out double cos, out double sin)
    {
        var store = Math.Abs(piv);
        double dd;
        if (store >= ww) dd = store * Math.Sqrt(1.0 + (ww / piv) * (ww / piv));
        else dd = ww * Math.Sqrt(1.0 + (piv / ww) * (piv / ww));
        cos = ww / dd;
        sin = piv / dd;
        ww = dd;
    }

    /// <summary>fprota: applies a Givens rotation to (a, b).</summary>
    public static void Fprota(double cos, double sin, ref double a, ref double b)
    {
        var stor1 = a;
        var stor2 = b;
        b = cos * stor2 + sin * stor1;
        a = cos * stor1 - sin * stor2;
    }

    /// <summary>fprati: the zero of the rational interpolant through (p1,f1),(p2,f2),(p3,f3);
    /// p3 &lt;= 0 stands for infinity. Updates the bracket as FITPACK does.</summary>
    public static double Fprati(ref double p1, ref double f1, double p2, double f2, ref double p3, ref double f3)
    {
        double p;
        if (p3 > 0)
        {
            var h1 = f1 * (f2 - f3);
            var h2 = f2 * (f3 - f1);
            var h3 = f3 * (f1 - f2);
            p = -(p1 * p2 * h3 + p2 * p3 * h1 + p3 * p1 * h2) / (p1 * h1 + p2 * h2 + p3 * h3);
        }
        else p = (p1 * (f1 - f3) * f2 - p2 * (f2 - f3) * f1) / ((f1 - f2) * f3);
        if (f2 < 0) { p3 = p2; f3 = f2; }
        else { p1 = p2; f1 = f2; }
        return p;
    }

    /// <summary>fpbspl, 1-based: B-splines at x on interval t(l) &lt;= x &lt; t(l+1); h(1..k+1).</summary>
    private static void Fpbspl(double[] t, int k, double x, int l, double[] h, double[] hh)
    {
        h[1] = 1.0;
        for (var j = 1; j <= k; j++)
        {
            for (var i = 1; i <= j; i++) hh[i] = h[i];
            h[1] = 0.0;
            for (var i = 1; i <= j; i++)
            {
                var li = l + i;
                var lj = li - j;
                if (t[li] == t[lj]) { h[i + 1] = 0.0; continue; }
                var f = hh[i] / (t[li] - t[lj]);
                h[i] += f * (t[li] - x);
                h[i + 1] = f * (x - t[lj]);
            }
        }
    }

    /// <summary>fpback, 1-based: back substitution with the upper triangular band matrix
    /// a(1..n, 1..k) on rhs z (column d of idim), into c.</summary>
    private static void Fpback(double[,] a, double[] z, int n, int k, double[] c, int idim, int d, int nest)
    {
        var k1 = k - 1;
        c[(n - 1) * idim + d] = z[(n - 1) * idim + d] / a[n, 1];
        var i = n - 1;
        if (i == 0) return;
        for (var j = 2; j <= n; j++)
        {
            var store = z[(i - 1) * idim + d];
            var i1 = k1;
            if (j <= k1) i1 = j - 1;
            var m = i;
            for (var l = 1; l <= i1; l++)
            {
                m++;
                store -= c[(m - 1) * idim + d] * a[i, l + 1];
            }
            c[(i - 1) * idim + d] = store / a[i, 1];
            i--;
        }
    }

    /// <summary>fpdisc, 1-based: k-th derivative jumps of the B-splines at the interior knots,
    /// scaled as FITPACK does, into b(1..n-2k-2, 1..k+2).</summary>
    private static void Fpdisc(double[] t, int n, int k2, double[,] b)
    {
        var k1 = k2 - 1;
        var k = k1 - 1;
        var nk1 = n - k1;
        var nrint = nk1 - k;
        var an = (double)nrint;
        var fac = an / (t[nk1 + 1] - t[k1]);
        var h = new double[2 * k2 + 2];
        for (var l = k2; l <= nk1; l++)
        {
            var lmk = l - k1;
            for (var j = 1; j <= k1; j++)
            {
                var ik = j + k1;
                var lj = l + j;
                var lk = lj - k2;
                h[j] = t[l] - t[lk];
                h[ik] = t[l] - t[lj];
            }
            var lp = lmk;
            for (var j = 1; j <= k2; j++)
            {
                var jk = j;
                var prod = h[j];
                for (var i = 1; i <= k; i++)
                {
                    jk++;
                    prod = prod * h[jk] * fac;
                }
                var lk = lp + k1;
                b[lmk, j] = (t[lk] - t[lp]) / prod;
                lp++;
            }
        }
    }

    /// <summary>fpknot, 1-based: adds one knot inside the interval with the largest residual
    /// sum that still contains data points.</summary>
    private static void Fpknot(double[] x, double[] t, ref int n, double[] fpint, int[] nrdata, ref int nrint, int istart)
    {
        var k = (n - nrint - 1) / 2;
        var fpmax = 0.0;
        var jbegin = istart;
        int number = 0, maxpt = 0, maxbeg = 0;
        for (var j = 1; j <= nrint; j++)
        {
            var jpoint = nrdata[j];
            if (!(fpmax >= fpint[j] || jpoint == 0))
            {
                fpmax = fpint[j];
                number = j;
                maxpt = jpoint;
                maxbeg = jbegin;
            }
            jbegin = jbegin + jpoint + 1;
        }
        if (number == 0) throw new InvalidOperationException("FITPACK fpknot: no knot interval contains data points.");
        var ihalf = maxpt / 2 + 1;
        var nrx = maxbeg + ihalf;
        var next = number + 1;
        if (next <= nrint)
        {
            for (var j = next; j <= nrint; j++)
            {
                var jj = next + nrint - j;
                fpint[jj + 1] = fpint[jj];
                nrdata[jj + 1] = nrdata[jj];
                var jk = jj + k;
                t[jk + 1] = t[jk];
            }
        }
        nrdata[number] = ihalf - 1;
        nrdata[next] = maxpt - ihalf;
        var am = (double)maxpt;
        var an = (double)nrdata[number];
        fpint[number] = fpmax * an / am;
        an = nrdata[next];
        fpint[next] = fpmax * an / am;
        var jkk = next + k;
        t[jkk] = x[nrx];
        n++;
        nrint++;
    }

    // ------------------------------------------------------------- fixed-knot least squares

    /// <summary>
    /// Weighted least-squares spline with fixed knots t (0-based, length n) through Givens QR
    /// of the banded observation matrix (the first half of an fpcurf iteration). y is (m, idim).
    /// Returns coefficients (n, idim) padded with zeros and the residual sum fp.
    /// </summary>
    public static double[] LeastSquares(double[] x, double[] y, int idim, double[] w, double[] t0, int k, out double fp)
    {
        int m = x.Length, n = t0.Length;
        var k1 = k + 1;
        var nk1 = n - k1;
        var t = OneBased(t0);
        var xs = OneBased(x);
        var a = new double[nk1 + 1, k1 + 1];
        var z = new double[nk1 * idim];
        var h = new double[k1 + 2];
        var hh = new double[k1 + 2];
        var yi = new double[idim];
        fp = 0;
        var l = k1;
        for (var it = 1; it <= m; it++)
        {
            var xi = xs[it];
            var wi = w[it - 1];
            for (var d = 0; d < idim; d++) yi[d] = y[(it - 1) * idim + d] * wi;
            while (!(xi < t[l + 1] || l == nk1)) l++;
            Fpbspl(t, k, xi, l, h, hh);
            for (var i = 1; i <= k1; i++) h[i] *= wi;
            RotateRow(a, z, h, yi, l - k1, k1, idim, nk1);
            for (var d = 0; d < idim; d++) fp += yi[d] * yi[d];
        }
        var c = new double[n * idim];
        for (var d = 0; d < idim; d++)
        {
            CheckDiagonal(a, nk1);
            Fpback(a, z, nk1, k1, c, idim, d, n);
        }
        return c;
    }

    private static void CheckDiagonal(double[,] a, int nk1)
    {
        for (var i = 1; i <= nk1; i++)
            if (a[i, 1] == 0)
                throw new ArgumentException("The least-squares system is singular: the knots violate the Schoenberg-Whitney conditions for these data.");
    }

    /// <summary>Rotates one observation row h(1..k1) starting at column j0+1 into the triangle.</summary>
    private static void RotateRow(double[,] a, double[] z, double[] h, double[] yi, int j0, int k1, int idim, int nk1)
    {
        var j = j0;
        for (var i = 1; i <= k1; i++)
        {
            j++;
            var piv = h[i];
            if (piv == 0) continue;
            var ajj = a[j, 1];
            Fpgivs(piv, ref ajj, out var cos, out var sin);
            a[j, 1] = ajj;
            for (var d = 0; d < idim; d++)
            {
                var zz = z[(j - 1) * idim + d];
                Fprota(cos, sin, ref yi[d], ref zz);
                z[(j - 1) * idim + d] = zz;
            }
            if (i == k1) break;
            var i2 = 1;
            for (var i1 = i + 1; i1 <= k1; i1++)
            {
                i2++;
                var aj = a[j, i2];
                Fprota(cos, sin, ref h[i1], ref aj);
                a[j, i2] = aj;
            }
        }
    }

    private static double[] OneBased(double[] v)
    {
        var r = new double[v.Length + 1];
        Array.Copy(v, 0, r, 1, v.Length);
        return r;
    }

    // ------------------------------------------------------------- fpcurf / fppara

    /// <summary>
    /// FITPACK curfit/parcur (iopt = 0, non-periodic) for y of shape (m, idim) and smoothing
    /// factor s &gt;= 0. <paramref name="nest"/> bounds the number of knots.
    /// </summary>
    public static FitpackCurveResult Curfit(double[] x0, double[] y0, int idim, double[] w0, double xb, double xe,
        int k, double s, int nest, bool reproMode = false, double[]? fixedKnots = null)
    {
        var m = x0.Length;
        if (k < 1 || k > 5) throw new ArgumentException($"Given degree of the spline (k={k}) is not supported. (1<=k<=5)");
        if (m <= k) throw new ArgumentException("m > k must hold");
        if (w0.Length != m || y0.Length != m * idim) throw new ArgumentException("Lengths of x, y and w must agree.");
        if (s < 0) throw new ArgumentException("s must be non-negative.");
        foreach (var wv in w0) if (!(wv > 0)) throw new ArgumentException("Weights must be strictly positive.");
        if (!(xb <= x0[0]) || !(x0[m - 1] <= xe)) throw new ArgumentException("xb <= x[0] and x[-1] <= xe must hold.");
        for (var i = 1; i < m; i++) if (x0[i] < x0[i - 1]) throw new ArgumentException("x must be non-decreasing.");

        var k1 = k + 1;
        var k2 = k1 + 1;
        var nmin = 2 * k1;
        var nmax = m + k1;
        if (nest < nmin) throw new ArgumentException($"nest too small: {nest} < {nmin}.");
        if (s == 0 && nest < nmax) throw new ArgumentException("For interpolation nest must be at least m+k+1.");
        const double con1 = 0.1, con9 = 0.9, con4 = 0.04, half = 0.5;
        var acc = Tol * s;

        var x = OneBased(x0);
        var t = new double[nest + 2];
        var fpint = new double[nest + 2];
        var nrdata = new int[nest + 2];
        var a = new double[nest + 1, k1 + 1];
        var b = new double[nest + 1, k2 + 1];
        var g = new double[nest + 1, k2 + 1];
        var q = new double[m + 1, k1 + 1];
        var z = new double[nest * idim];
        var c = new double[nest * idim];
        var h = new double[k2 + 2];
        var hh = new double[k2 + 2];
        var yi = new double[idim];

        int n, nplus = 0, ier = 0, nrint, nk1 = 0;
        double fp = 0, fpold = 0, fp0 = 0, fpms = 0;

        void InterpolationKnots()
        {
            var mk1 = m - k1;
            if (mk1 == 0) return;
            var k3 = k / 2;
            var i = k2;
            var j = k3 + 2;
            if (k3 * 2 != k)
                for (var l = 1; l <= mk1; l++) { t[i] = x[j]; i++; j++; }
            else
                for (var l = 1; l <= mk1; l++) { t[i] = (x[j] + x[j - 1]) * half; i++; j++; }
        }

        if (s > 0)
        {
            n = nmin;
            fpold = 0;
            nplus = 0;
            nrdata[1] = m - 2;
        }
        else
        {
            n = nmax;
            InterpolationKnots();
        }

        // Residual sums per knot interval, needed only by the reproduction mode's refits.
        void ComputeFpint()
        {
            var fpart = 0.0;
            var i = 1;
            var l = k2;
            var isNew = 0;
            for (var it = 1; it <= m; it++)
            {
                if (!(x[it] < t[l] || l > nk1)) { isNew = 1; l++; }
                var term = 0.0;
                for (var d = 0; d < idim; d++)
                {
                    var tt = 0.0;
                    var l0 = l - k2;
                    for (var j = 1; j <= k1; j++) { l0++; tt += c[(l0 - 1) * idim + d] * q[it, j]; }
                    tt = w0[it - 1] * (tt - y0[(it - 1) * idim + d]);
                    term += tt * tt;
                }
                fpart += term;
                if (isNew == 0) continue;
                var store = term * half;
                fpint[i] = fpart - store;
                i++;
                fpart = store;
                isNew = 0;
            }
            fpint[nrint] = fpart;
        }

        // One least-squares fit with the current knots: sets a, z, q, c, fp.
        void LsqCurrent()
        {
            nk1 = n - k1;
            var ii = n;
            if (fixedKnots is null)
                for (var j = 1; j <= k1; j++) { t[j] = xb; t[ii] = xe; ii--; }
            fp = 0;
            for (var i = 1; i <= nk1; i++)
            {
                for (var d = 0; d < idim; d++) z[(i - 1) * idim + d] = 0;
                for (var j = 1; j <= k1; j++) a[i, j] = 0;
            }
            var l = k1;
            for (var it = 1; it <= m; it++)
            {
                var xi = x[it];
                var wi = w0[it - 1];
                for (var d = 0; d < idim; d++) yi[d] = y0[(it - 1) * idim + d] * wi;
                while (!(xi < t[l + 1] || l == nk1)) l++;
                Fpbspl(t, k, xi, l, h, hh);
                for (var i = 1; i <= k1; i++) { q[it, i] = h[i]; h[i] *= wi; }
                RotateRow(a, z, h, yi, l - k1, k1, idim, nk1);
                for (var d = 0; d < idim; d++) fp += yi[d] * yi[d];
            }
            for (var d = 0; d < idim; d++) Fpback(a, z, nk1, k1, c, idim, d, nest);
        }

        var goPart2 = false;
        var done = false;
        if (fixedKnots is not null)
        {
            // scipy make_splrep with user knots: no knot search, straight to fp(p) = s.
            n = fixedKnots.Length;
            if (n > nest) throw new ArgumentException("Too many knots for nest.");
            Array.Copy(fixedKnots, 0, t, 1, n);
            var tp = new double[nmin];
            for (var j = 0; j < k1; j++) { tp[j] = xb; tp[nmin - 1 - j] = xe; }
            LeastSquares(x0, y0, idim, w0, tp, k, out fp0);
            nrint = n - nmin + 1;
            LsqCurrent();
            fpms = fp - s;
            if (n == nmin || Math.Abs(fpms) < acc) done = true;
            else goPart2 = true;
            if (n == nmin) ier = -2;
        }
        for (var iter = 1; iter <= m && !done && !goPart2; iter++)
        {
            if (n == nmin) ier = -2;
            nrint = n - nmin + 1;
            LsqCurrent();
            if (ier == -2) fp0 = fp;
            fpms = fp - s;
            if (Math.Abs(fpms) < acc) { done = true; break; }
            if (fpms < 0) { goPart2 = true; break; }
            if (n == nmax) { ier = -1; done = true; break; }
            if (n == nest) { ier = 1; done = true; break; }
            if (ier == 0)
            {
                var npl1 = nplus * 2;
                var rn = (double)nplus;
                if (fpold - fp > acc) npl1 = (int)(rn * fpms / (fpold - fp));
                nplus = Math.Min(nplus * 2, Math.Max(npl1, Math.Max(nplus / 2, 1)));
            }
            else
            {
                nplus = 1;
                ier = 0;
            }
            fpold = fp;
            ComputeFpint();
            for (var l = 1; l <= nplus; l++)
            {
                Fpknot(x, t, ref n, fpint, nrdata, ref nrint, 1);
                if (n == nmax)
                {
                    InterpolationKnots();
                    break;
                }
                if (n == nest) break;
                if (reproMode && l < nplus)
                {
                    // scipy's make_splrep refits before placing the next knot.
                    LsqCurrent();
                    nrint = n - nmin + 1;
                    ComputeFpint();
                }
            }
        }

        if (!done && ier == -2) done = true;
        if (!done)
        {
            // ---------------- part 2: smoothing spline with fp(p) = s
            Fpdisc(t, n, k2, b);
            double p1 = 0, f1 = fp0 - s, p3 = -1, f3 = fpms;
            var p = 0.0;
            for (var i = 1; i <= nk1; i++) p += a[i, 1];
            var rnk = (double)nk1;
            p = rnk / p;
            int ich1 = 0, ich3 = 0;
            var n8 = n - nmin;
            var cz = new double[nest * idim];
            var hb = new double[k2 + 2];
            var converged = false;
            for (var iter = 1; iter <= MaxIt; iter++)
            {
                var pinv = 1.0 / p;
                for (var i = 1; i <= nk1; i++)
                {
                    for (var d = 0; d < idim; d++) cz[(i - 1) * idim + d] = z[(i - 1) * idim + d];
                    g[i, k2] = 0;
                    for (var j = 1; j <= k1; j++) g[i, j] = a[i, j];
                }
                for (var it = 1; it <= n8; it++)
                {
                    for (var i = 1; i <= k2; i++) hb[i] = b[it, i] * pinv;
                    for (var d = 0; d < idim; d++) yi[d] = 0;
                    for (var j = it; j <= nk1; j++)
                    {
                        var piv = hb[1];
                        var gj = g[j, 1];
                        Fpgivs(piv, ref gj, out var cos, out var sin);
                        g[j, 1] = gj;
                        for (var d = 0; d < idim; d++)
                        {
                            var cc = cz[(j - 1) * idim + d];
                            Fprota(cos, sin, ref yi[d], ref cc);
                            cz[(j - 1) * idim + d] = cc;
                        }
                        if (j == nk1) break;
                        var i2 = k1;
                        if (j > n8) i2 = nk1 - j;
                        for (var i = 1; i <= i2; i++)
                        {
                            var i1 = i + 1;
                            var gv = g[j, i1];
                            Fprota(cos, sin, ref hb[i1], ref gv);
                            g[j, i1] = gv;
                            hb[i] = hb[i1];
                        }
                        hb[i2 + 1] = 0;
                    }
                }
                for (var d = 0; d < idim; d++) Fpback(g, cz, nk1, k2, c, idim, d, nest);
                fp = 0;
                var l = k2;
                for (var it = 1; it <= m; it++)
                {
                    if (!(x[it] < t[l] || l > nk1)) l++;
                    for (var d = 0; d < idim; d++)
                    {
                        var l0 = l - k2;
                        var term = 0.0;
                        for (var j = 1; j <= k1; j++) { l0++; term += c[(l0 - 1) * idim + d] * q[it, j]; }
                        var r = w0[it - 1] * (term - y0[(it - 1) * idim + d]);
                        fp += r * r;
                    }
                }
                fpms = fp - s;
                if (Math.Abs(fpms) < acc) { converged = true; break; }
                if (iter == MaxIt) { ier = 3; break; }
                var p2 = p;
                var f2 = fpms;
                if (ich3 == 0)
                {
                    if (!(f2 - f3 > acc))
                    {
                        p3 = p2; f3 = f2;
                        p *= con4;
                        if (p <= p1) p = p1 * con9 + p2 * con1;
                        continue;
                    }
                    if (f2 < 0) ich3 = 1;
                }
                if (ich1 == 0)
                {
                    if (!(f1 - f2 > acc))
                    {
                        p1 = p2; f1 = f2;
                        p /= con4;
                        if (p3 < 0) continue;
                        if (reproMode ? p <= p3 : p >= p3) p = p2 * con1 + p3 * con9;
                        continue;
                    }
                    if (f2 > 0) ich1 = 1;
                }
                if (f2 >= f1 || f2 <= f3) { ier = 2; break; }
                p = Fprati(ref p1, ref f1, p2, f2, ref p3, ref f3);
            }
            if (converged) ier = 0;
        }

        var tOut = new double[n];
        Array.Copy(t, 1, tOut, 0, n);
        var cOut = new double[n * idim];
        Array.Copy(c, 0, cOut, 0, (n - k1) * idim);
        return new FitpackCurveResult { T = tOut, C = cOut, K = k, Idim = idim, Fp = fp, Ier = ier };
    }
}
