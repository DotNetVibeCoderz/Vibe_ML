using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.InterpolateSupport;

namespace Gravicode.Science.SciNet;

/// <summary>Full result of <see cref="Interpolate.SplrepFull"/>: the spline plus FITPACK's
/// residual and status (scipy's <c>full_output=1</c>).</summary>
/// <param name="T">Knots.</param>
/// <param name="C">Coefficients, padded with zeros to len(T) as FITPACK returns them.</param>
/// <param name="K">Degree.</param>
/// <param name="Fp">Weighted sum of squared residuals.</param>
/// <param name="Ier">FITPACK status: 0 converged, -1 interpolating, -2 least-squares
/// polynomial, 1–3 warnings (storage, theoretically impossible, max iterations).</param>
/// <param name="Message">FITPACK's text for <paramref name="Ier"/>.</param>
public sealed record SplrepResult(double[] T, double[] C, int K, double Fp, int Ier, string Message);

/// <summary>Result of <see cref="Interpolate.Splprep"/>: knots, one coefficient array per
/// dimension, degree, the curve parameter values and FITPACK's residual and status.</summary>
/// <param name="T">Knots.</param>
/// <param name="C">Coefficients, one array per curve dimension (length len(T) - K - 1).</param>
/// <param name="K">Degree.</param>
/// <param name="U">Parameter value of each data point.</param>
/// <param name="Fp">Weighted residual sum over all dimensions.</param>
/// <param name="Ier">FITPACK status code.</param>
public sealed record SplprepResult(double[] T, double[][] C, int K, double[] U, double Fp, int Ier);

/// <summary>What FITPACK-style evaluation does outside the base interval (scipy's
/// <c>ext</c>).</summary>
public enum SplineExtMode
{
    /// <summary>ext=0: extrapolate with the end polynomial pieces.</summary>
    Extrapolate = 0,
    /// <summary>ext=1: return 0.</summary>
    Zeros = 1,
    /// <summary>ext=2: throw.</summary>
    Raise = 2,
    /// <summary>ext=3: return the boundary value.</summary>
    Const = 3,
}

public static partial class Interpolate
{
    // ================================================================ splrep

    /// <summary>
    /// B-spline representation of 1-D data (scipy.interpolate.splrep, FITPACK curfit). With
    /// s &gt; 0 knots are added until the weighted residual sum fp ≤ s, then the smoothing
    /// parameter is solved so that fp = s within 0.1%. s defaults to 0 without weights and to
    /// m - √(2m) with weights. Interior knots <paramref name="t"/> give a weighted
    /// least-squares spline instead. Periodic fitting (<c>per</c>) is not implemented.
    /// </summary>
    /// <returns>(t, c, k); c is padded with zeros to len(t), as scipy returns it.</returns>
    public static (double[] T, double[] C, int K) Splrep(double[] x, double[] y, double[]? w = null, double? xb = null, double? xe = null,
        int k = 3, double? s = null, double[]? t = null)
    {
        var r = SplrepFull(x, y, w, xb, xe, k, s, t);
        return (r.T, r.C, r.K);
    }

    /// <summary><see cref="Splrep"/> with FITPACK's residual and status code.</summary>
    public static SplrepResult SplrepFull(double[] x, double[] y, double[]? w = null, double? xb = null, double? xe = null,
        int k = 3, double? s = null, double[]? t = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        var m = x.Length;
        if (y.Length != m) throw new ArgumentException("Lengths of the first three arguments (x,y,w) must be equal");
        double sv;
        double[] ww;
        if (w is null) { ww = Enumerable.Repeat(1.0, m).ToArray(); sv = s ?? 0.0; }
        else
        {
            if (w.Length != m) throw new ArgumentException($"len(w)={w.Length} is not equal to m={m}");
            ww = (double[])w.Clone();
            sv = s ?? m - Math.Sqrt(2.0 * m);
        }
        if (k < 1 || k > 5) throw new ArgumentException($"Given degree of the spline (k={k}) is not supported. (1<=k<=5)");
        if (m <= k) throw new ArgumentException("m > k must hold");
        var b = xb ?? x[0];
        var e = xe ?? x[^1];
        if (t is not null)
        {
            var full = new double[t.Length + 2 * k + 2];
            for (var i = 0; i <= k; i++) { full[i] = b; full[^(i + 1)] = e; }
            Array.Copy(t, 0, full, k + 1, t.Length);
            CheckKnots(x, full, k);
            var c = InterpolateFitpack.LeastSquares(x, y, 1, ww, full, k, out var fpl);
            return new SplrepResult(full, c, k, fpl, 0, InterpolateFitpack.Message(0));
        }
        var nest = Math.Max(m + k + 1, 2 * k + 3);
        var r = InterpolateFitpack.Curfit(x, y, 1, ww, b, e, k, sv, nest);
        return new SplrepResult(r.T, r.C, k, r.Fp, r.Ier, InterpolateFitpack.Message(r.Ier));
    }

    /// <summary>FITPACK's <c>fpchec</c> (scipy's <c>fpcheck</c>): knot count and order, data
    /// range, and the Schoenberg–Whitney conditions.</summary>
    internal static void CheckKnots(double[] x, double[] t, int k)
    {
        int m = x.Length, n = t.Length, nk1 = n - k - 1;
        if (!(k + 1 <= nk1 && nk1 <= m)) throw new ArgumentException($"Need k+1 <= n-k-1 <= m. Got m={m}, n={n} and k={k}.");
        for (var i = 0; i < k + 1; i++) if (t[i] > t[i + 1]) throw new ArgumentException("First k knots must be ordered.");
        for (var i = nk1; i < n; i++) if (t[i] < t[i - 1]) throw new ArgumentException("Last k knots must be ordered.");
        for (var i = k + 1; i < n - k; i++) if (t[i] <= t[i - 1]) throw new ArgumentException("Internal knots must be distinct.");
        if (x[0] < t[k] || x[^1] > t[n - k - 1]) throw new ArgumentException("Out of bounds: x outside the knot range.");
        const string mesg = "Schoenberg-Whitney condition is violated by the knots and data.";
        if (x[0] >= t[k + 1] || x[^1] <= t[n - k - 2]) throw new ArgumentException(mesg);
        var l = k + 1;
        var nk3 = n - k - 3;
        if (nk3 < 2) return;
        for (var j = 1; j <= nk3; j++)
        {
            var tj = t[j];
            l++;
            var tl = t[l];
            var i = 0;
            while (i < m && !(x[i] > tj)) i++;
            if (i >= m - 1) throw new ArgumentException(mesg);
            if (x[i] >= tl) throw new ArgumentException(mesg);
        }
    }

    // ================================================================ splev & co.

    /// <summary>Evaluates a spline or its derivative (scipy.interpolate.splev).</summary>
    public static double[] Splev(double[] x, (double[] T, double[] C, int K) tck, int der = 0, SplineExtMode ext = SplineExtMode.Extrapolate)
    {
        ArgumentNullException.ThrowIfNull(x);
        var (t, c, k) = tck;
        if (der < 0 || der > k) throw new ArgumentException($"0<=der={der}<=k={k} must hold");
        var n = t.Length - k - 1;
        var res = new double[x.Length];
        var h = new double[k + 1];
        var wk = new double[k + 1];
        var hint = k;
        double lo = t[k], hi = t[n];
        Span<double> o = stackalloc double[1];
        for (var i = 0; i < x.Length; i++)
        {
            var v = x[i];
            if (v < lo || v > hi)
            {
                switch (ext)
                {
                    case SplineExtMode.Zeros: res[i] = 0; continue;
                    case SplineExtMode.Raise: throw new ArgumentException("Found x value not in the domain");
                    case SplineExtMode.Const: v = v < lo ? lo : hi; break;
                }
            }
            EvaluateSplinePoint(t, c, 1, k, v, der, true, ref hint, o, h, wk);
            res[i] = o[0];
        }
        return res;
    }

    /// <summary>Scalar overload of <see cref="Splev(double[], ValueTuple{double[], double[], int}, int, SplineExtMode)"/>.</summary>
    public static double Splev(double x, (double[] T, double[] C, int K) tck, int der = 0, SplineExtMode ext = SplineExtMode.Extrapolate) =>
        Splev([x], tck, der, ext)[0];

    /// <summary>Definite integral of a spline over [a, b] (scipy.interpolate.splint); the range
    /// is clipped to the base interval, as FITPACK does.</summary>
    public static double Splint(double a, double b, (double[] T, double[] C, int K) tck)
    {
        var (t, c, k) = tck;
        var spl = new BSpline(t, c, k, InterpolateExtrapolate.None);
        return spl.Integrate(a, b);
    }

    /// <summary>Roots of a cubic spline inside its base interval (scipy.interpolate.sproot).
    /// Only k = 3 is supported, as in FITPACK; at most <paramref name="mest"/> are returned.</summary>
    public static double[] Sproot((double[] T, double[] C, int K) tck, int mest = 10)
    {
        var (t, c, k) = tck;
        if (k != 3) throw new ArgumentException("sproot works only for cubic (k=3) splines");
        if (t.Length < 8) throw new ArgumentException($"The number of knots {t.Length}>=8");
        var roots = BaseIntervalPPoly(t, c, k).Roots(discontinuity: false, extrapolate: false);
        var list = roots.Where(r => !double.IsNaN(r)).ToArray();
        return list.Length > mest ? list[..mest] : list;
    }

    /// <summary>The spline as a PPoly over its distinct knots inside [t[k], t[n]].</summary>
    internal static PPoly BaseIntervalPPoly(double[] t, double[] c, int k)
    {
        var n = t.Length - k - 1;
        var bp = new List<double> { t[k] };
        for (var i = k + 1; i <= n; i++) if (t[i] > bp[^1]) bp.Add(t[i]);
        var m = bp.Count - 1;
        var cv = new double[(k + 1) * m];
        var h = new double[k + 1];
        var wk = new double[k + 1];
        Span<double> o = stackalloc double[1];
        for (var i = 0; i < m; i++)
        {
            var hint = k;
            for (var mm = 0; mm <= k; mm++)
            {
                EvaluateSplinePoint(t, c, 1, k, bp[i], mm, true, ref hint, o, h, wk);
                cv[(k - mm) * m + i] = o[0] / Factorial(mm);
            }
        }
        return PPoly.ConstructFast(cv, [.. bp], null, InterpolateExtrapolate.None);
    }

    /// <summary>All derivatives 0..k of a spline at x (scipy.interpolate.spalde); x must lie in
    /// the base interval.</summary>
    public static double[] Spalde(double x, (double[] T, double[] C, int K) tck)
    {
        var (t, c, k) = tck;
        var n = t.Length - k - 1;
        if (x < t[k] || x > t[n]) throw new ArgumentException("Invalid input data. t(k)<=x<=t(n-k+1) must hold.");
        var res = new double[k + 1];
        var h = new double[k + 1];
        var wk = new double[k + 1];
        Span<double> o = stackalloc double[1];
        for (var d = 0; d <= k; d++)
        {
            var hint = k;
            EvaluateSplinePoint(t, c, 1, k, x, d, false, ref hint, o, h, wk);
            res[d] = o[0];
        }
        return res;
    }

    /// <summary>Derivative spline (scipy.interpolate.splder).</summary>
    public static (double[] T, double[] C, int K) Splder((double[] T, double[] C, int K) tck, int n = 1)
    {
        var (t, c, k) = tck;
        return BSpline.SplderCore(t, PadToKnots(t, c), 1, k, n);
    }

    /// <summary>Antiderivative spline, zero at t[k] (scipy.interpolate.splantider).</summary>
    public static (double[] T, double[] C, int K) Splantider((double[] T, double[] C, int K) tck, int n = 1)
    {
        var (t, c, k) = tck;
        return BSpline.SplantiderCore(t, PadToKnots(t, c), 1, k, n);
    }

    private static double[] PadToKnots(double[] t, double[] c)
    {
        if (c.Length >= t.Length) return (double[])c.Clone();
        var r = new double[t.Length];
        Array.Copy(c, r, c.Length);
        return r;
    }

    /// <summary>Inserts a knot m times without changing the spline (scipy.interpolate.insert,
    /// FITPACK insert); <paramref name="per"/> keeps a periodic spline's boundary constraints.</summary>
    public static (double[] T, double[] C, int K) Insert(double x, (double[] T, double[] C, int K) tck, int m = 1, bool per = false)
    {
        var (t, c, k) = tck;
        var tt = (double[])t.Clone();
        var cc = PadToKnots(t, c);
        var rows = cc.Length;
        for (var i = 0; i < m; i++) (tt, cc, rows) = BSpline.InsertOnce(x, tt, cc, rows, 1, k, per);
        return (tt, cc, k);
    }

    // ================================================================ splprep

    /// <summary>
    /// B-spline representation of an N-D curve (scipy.interpolate.splprep, FITPACK parcur).
    /// <paramref name="x"/> holds one array per dimension. Without <paramref name="u"/> the
    /// parameter is the normalised cumulative chord length. s defaults to m - √(2m).
    /// </summary>
    public static SplprepResult Splprep(double[][] x, double[]? w = null, double[]? u = null, double? ub = null, double? ue = null,
        int k = 3, double? s = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var idim = x.Length;
        if (idim < 1 || idim > 10) throw new ArgumentException("0 < idim < 11 must hold");
        var m = x[0].Length;
        foreach (var r in x) if (r.Length != m) throw new ArgumentException("Mismatch of input dimensions");
        if (k < 1 || k > 5) throw new ArgumentException($"1 <= k= {k} <=5 must hold");
        if (m <= k) throw new ArgumentException("m > k must hold");
        var ww = w is null ? Enumerable.Repeat(1.0, m).ToArray() : (double[])w.Clone();
        if (ww.Length != m) throw new ArgumentException("Mismatch of input dimensions");
        var sv = s ?? m - Math.Sqrt(2.0 * m);
        double[] uu;
        double b, e;
        if (u is null)
        {
            uu = new double[m];
            for (var i = 1; i < m; i++)
            {
                var d2 = 0.0;
                for (var j = 0; j < idim; j++) { var d = x[j][i] - x[j][i - 1]; d2 += d * d; }
                uu[i] = uu[i - 1] + Math.Sqrt(d2);
            }
            if (!(uu[m - 1] > 0)) throw new ArgumentException("Error on input data");
            for (var i = 1; i < m; i++) uu[i] /= uu[m - 1];
            b = 0; e = 1;
        }
        else
        {
            if (u.Length != m) throw new ArgumentException("Mismatch of input dimensions");
            uu = (double[])u.Clone();
            b = ub ?? uu[0];
            e = ue ?? uu[^1];
        }
        for (var i = 1; i < m; i++) if (!(uu[i] > uu[i - 1])) throw new ArgumentException("Error on input data: u must be strictly increasing.");
        var y = new double[m * idim];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < idim; j++) y[i * idim + j] = x[j][i];
        var nest = sv == 0 ? m + k + 1 : m + 2 * k;
        nest = Math.Max(nest, 2 * k + 3);
        var res = InterpolateFitpack.Curfit(uu, y, idim, ww, b, e, k, sv, nest);
        var n = res.T.Length;
        var cs = new double[idim][];
        for (var j = 0; j < idim; j++)
        {
            cs[j] = new double[n - k - 1];
            for (var i = 0; i < n - k - 1; i++) cs[j][i] = res.C[i * idim + j];
        }
        return new SplprepResult(res.T, cs, k, uu, res.Fp, res.Ier);
    }

    // ================================================================ make_splrep / make_splprep

    /// <summary>
    /// scipy's <c>make_splrep</c> (1.15+): FITPACK's smoothing-spline algorithm returning a
    /// <see cref="BSpline"/>. This reproduces scipy's Python re-implementation, including its
    /// refit between knot insertions, so knots can differ from <see cref="Splrep"/> when more
    /// than one knot is added per round. s = 0 delegates to <see cref="MakeInterpSpline(double[], double[], int, double[], SplineBoundary?)"/>.
    /// </summary>
    public static BSpline MakeSplrep(double[] x, double[] y, double[]? w = null, double? xb = null, double? xe = null,
        int k = 3, double s = 0, double[]? t = null, int? nest = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (s == 0)
        {
            if (t is not null || w is not null || nest is not null) throw new ArgumentException("s==0 is for interpolation only");
            return MakeInterpSpline(x, y, k);
        }
        if (s < 0) throw new ArgumentException($"`s` must be non-negative. Got s={s}");
        var m = x.Length;
        var ww = w is null ? Enumerable.Repeat(1.0, m).ToArray() : (double[])w.Clone();
        var b = xb ?? x.Min();
        var e = xe ?? x.Max();
        var r = FitRepro(x, y, 1, ww, b, e, k, s, t, nest);
        var nc = r.T.Length - k - 1;
        var c = new double[nc];
        Array.Copy(r.C, c, nc);
        return BSpline.Raw(r.T, c, nc, [], k, InterpolateExtrapolate.Extrapolate, 0);
    }

    private static FitpackCurveResult FitRepro(double[] x, double[] y, int idim, double[] w, double b, double e, int k, double s, double[]? t, int? nest)
    {
        var m = x.Length;
        if (t is not null)
        {
            if (nest is not null) throw new ArgumentException("Either supply `t` or `nest`.");
            CheckKnots(x, t, k);
            return InterpolateFitpack.Curfit(x, y, idim, w, b, e, k, s, t.Length, reproMode: true, fixedKnots: (double[])t.Clone());
        }
        var ns = nest ?? Math.Max(m + k + 1, 2 * k + 3);
        if (ns < 2 * (k + 1)) throw new ArgumentException($"`nest` too small: nest={ns} < 2*(k+1) = {2 * (k + 1)}.");
        return InterpolateFitpack.Curfit(x, y, idim, w, b, e, k, s, ns, reproMode: true);
    }

    /// <summary>
    /// scipy's <c>make_splprep</c>: a parametric smoothing spline. Returns a vector-valued
    /// <see cref="BSpline"/> with axis 1 — evaluating it at parameter values u gives shape
    /// (ndim, len(u)) — and the parameter values of the data.
    /// </summary>
    public static (BSpline Spline, double[] U) MakeSplprep(double[][] x, double[]? w = null, double[]? u = null, double? ub = null, double? ue = null,
        int k = 3, double s = 0, double[]? t = null, int? nest = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var idim = x.Length;
        var m = x[0].Length;
        double[] uu;
        if (u is null)
        {
            uu = new double[m];
            for (var i = 1; i < m; i++)
            {
                var d2 = 0.0;
                for (var j = 0; j < idim; j++) { var d = x[j][i] - x[j][i - 1]; d2 += d * d; }
                uu[i] = uu[i - 1] + Math.Sqrt(d2);
            }
            var last = uu[m - 1];
            for (var i = 0; i < m; i++) uu[i] /= last;
        }
        else uu = (double[])u.Clone();
        var y = new double[m * idim];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < idim; j++) y[i * idim + j] = x[j][i];
        if (s == 0)
        {
            if (t is not null || w is not null || nest is not null) throw new ArgumentException("s==0 is for interpolation only");
            var spl = MakeInterpSpline(uu, new NdArray(y, m, idim), k);
            return (BSpline.Raw(spl.KnotsRaw, spl.CoefficientsRaw, spl.CoefficientRows, [idim], k, InterpolateExtrapolate.Extrapolate, 1), uu);
        }
        var ww = w is null ? Enumerable.Repeat(1.0, m).ToArray() : (double[])w.Clone();
        var r = FitRepro(uu, y, idim, ww, ub ?? uu.Min(), ue ?? uu.Max(), k, s, t, nest);
        var nc = r.T.Length - k - 1;
        var c = new double[nc * idim];
        Array.Copy(r.C, c, nc * idim);
        return (BSpline.Raw(r.T, c, nc, [idim], k, InterpolateExtrapolate.Extrapolate, 1), uu);
    }
}

/// <summary>
/// One-dimensional smoothing spline (scipy.interpolate.UnivariateSpline, FITPACK curfit). The
/// number of knots grows until the weighted residual sum is at most s (default: the number of
/// points), then the fit is smoothed so that it equals s.
/// </summary>
/// <remarks><see cref="SetSmoothingFactor"/> refits from scratch; FITPACK (iopt=1) continues
/// from the current knots, so the knots it then picks can differ from scipy's.</remarks>
public class UnivariateSpline
{
    private protected double[] _t = [];
    private protected double[] _c = [];
    private protected int _k;
    private protected double[] _x = [], _y = [], _w = [];
    private protected double _xb, _xe;

    /// <summary>FITPACK's status code of the fit.</summary>
    public int Ier { get; private protected set; }

    /// <summary>Weighted sum of squared residuals (scipy's <c>get_residual()</c>).</summary>
    public double Residual { get; private protected set; }

    /// <summary>Out-of-range behaviour of <see cref="Evaluate(double, int, SplineExtMode?)"/>.</summary>
    public SplineExtMode Ext { get; set; }

    /// <summary>Fits a smoothing spline of degree k (1..5) to (x, y) with weights w over the
    /// interval [xb, xe] (default: the data range).</summary>
    public UnivariateSpline(double[] x, double[] y, double[]? w = null, double? xb = null, double? xe = null, int k = 3,
        double? s = null, SplineExtMode ext = SplineExtMode.Extrapolate)
    {
        Init(x, y, w, xb, xe, k);
        Ext = ext;
        Fit(s ?? _w.Length);
    }

    private protected UnivariateSpline() { }

    private protected void Init(double[] x, double[] y, double[]? w, double? xb, double? xe, int k)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (x.Length != y.Length) throw new ArgumentException("x and y must have the same length.");
        _x = CopyFinite(x, "x");
        _y = CopyFinite(y, "y");
        for (var i = 1; i < _x.Length; i++) if (!(_x[i] >= _x[i - 1])) throw new ArgumentException("x must be increasing");
        _w = w is null ? Enumerable.Repeat(1.0, x.Length).ToArray() : CopyFinite(w, "w");
        if (_w.Length != _x.Length) throw new ArgumentException("x, y, and w should have a same length");
        _xb = xb ?? _x[0];
        _xe = xe ?? _x[^1];
        _k = k;
        _t = []; _c = [];
    }

    private protected void Fit(double s)
    {
        var m = _x.Length;
        var nest = Math.Max(m + _k + 1, 2 * _k + 3);
        var r = InterpolateFitpack.Curfit(_x, _y, 1, _w, _xb, _xe, _k, s, nest);
        _t = r.T; _c = r.C; Residual = r.Fp; Ier = r.Ier;
    }

    private protected static UnivariateSpline FromTck(double[] t, double[] c, int k, SplineExtMode ext)
    {
        var s = new UnivariateSpline { _t = t, _c = c, _k = k, Ext = ext };
        return s;
    }

    /// <summary>Degree.</summary>
    public int K => _k;

    /// <summary>Knots inside and at the ends of the base interval (scipy's <c>get_knots()</c>).</summary>
    public double[] GetKnots() => _t[_k..(_t.Length - _k)];

    /// <summary>B-spline coefficients (scipy's <c>get_coeffs()</c>).</summary>
    public double[] GetCoeffs() => _c[..(_t.Length - _k - 1)];

    /// <summary>The underlying (t, c, k).</summary>
    public (double[] T, double[] C, int K) Tck => ((double[])_t.Clone(), (double[])_c.Clone(), _k);

    /// <summary>Weighted sum of squared residuals.</summary>
    public double GetResidual() => Residual;

    /// <summary>Value (or nu-th derivative) at x.</summary>
    public double Evaluate(double x, int nu = 0, SplineExtMode? ext = null) => Interpolate.Splev(x, Tck, nu, ext ?? Ext);

    /// <summary>Values (or nu-th derivatives) at every x.</summary>
    public double[] Evaluate(double[] x, int nu = 0, SplineExtMode? ext = null) => Interpolate.Splev(x, Tck, nu, ext ?? Ext);

    /// <summary>Definite integral over [a, b] (clipped to the base interval).</summary>
    public double Integral(double a, double b) => Interpolate.Splint(a, b, Tck);

    /// <summary>Zeros of a cubic spline (scipy supports only k = 3).</summary>
    public double[] Roots()
    {
        if (_k != 3) throw new NotSupportedException("finding roots unsupported for non-cubic splines");
        return Interpolate.Sproot(Tck);
    }

    /// <summary>All derivatives 0..k at x.</summary>
    public double[] Derivatives(double x) => Interpolate.Spalde(x, Tck);

    /// <summary>The n-th derivative as a new spline.</summary>
    public UnivariateSpline Derivative(int n = 1)
    {
        var (t, c, k) = Interpolate.Splder(Tck, n);
        return FromTck(t, c, k, Ext);
    }

    /// <summary>The n-th antiderivative as a new spline.</summary>
    public UnivariateSpline Antiderivative(int n = 1)
    {
        var (t, c, k) = Interpolate.Splantider(Tck, n);
        return FromTck(t, c, k, Ext);
    }

    /// <summary>Refits with a new smoothing factor (from scratch; see the class remarks).</summary>
    public void SetSmoothingFactor(double s)
    {
        if (_x.Length == 0) throw new InvalidOperationException("This spline was not fitted to data.");
        Fit(s);
    }
}

/// <summary>Interpolating spline through every point (scipy's
/// <c>InterpolatedUnivariateSpline</c>; s = 0).</summary>
public class InterpolatedUnivariateSpline : UnivariateSpline
{
    /// <summary>Interpolates (x, y) with a spline of degree k.</summary>
    public InterpolatedUnivariateSpline(double[] x, double[] y, double[]? w = null, double? xb = null, double? xe = null, int k = 3,
        SplineExtMode ext = SplineExtMode.Extrapolate)
        : base(x, y, w, xb, xe, k, 0.0, ext) { }
}

/// <summary>Weighted least-squares spline with given interior knots (scipy's
/// <c>LSQUnivariateSpline</c>); the knots must satisfy the Schoenberg–Whitney conditions.</summary>
public class LSQUnivariateSpline : UnivariateSpline
{
    /// <summary>Fits with interior knots <paramref name="t"/>.</summary>
    public LSQUnivariateSpline(double[] x, double[] y, double[] t, double[]? w = null, double? xb = null, double? xe = null, int k = 3,
        SplineExtMode ext = SplineExtMode.Extrapolate)
    {
        Init(x, y, w, xb, xe, k);
        Ext = ext;
        ArgumentNullException.ThrowIfNull(t);
        var full = new double[t.Length + 2 * k + 2];
        for (var i = 0; i <= k; i++) { full[i] = _xb; full[^(i + 1)] = _xe; }
        Array.Copy(t, 0, full, k + 1, t.Length);
        try { Interpolate.CheckKnots(_x, full, k); }
        catch (ArgumentException ex) { throw new ArgumentException("Interior knots t must satisfy Schoenberg-Whitney conditions", ex); }
        _t = full;
        _c = InterpolateFitpack.LeastSquares(_x, _y, 1, _w, full, k, out var fp);
        Residual = fp;
        Ier = 0;
    }
}
