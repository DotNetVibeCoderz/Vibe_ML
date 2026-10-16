using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Polynomial interpolation through (xi, yi) in barycentric form
/// (scipy.interpolate.BarycentricInterpolator; Berrut &amp; Trefethen 2004). Scalar values.
/// </summary>
/// <remarks>
/// <para>Weights are products of scaled node differences; the scale 4/(max - min) (the
/// inverse "capacity" of the interval) keeps the products from overflowing for many nodes.
/// scipy multiplies the factors in a random order; this class uses index order — the weights
/// then differ in the last bits, which cancels in the barycentric quotient.</para>
/// <para>Derivatives use the differentiation matrix at the nodes and recurse on a
/// barycentric interpolant of the node derivatives, as scipy does.</para>
/// </remarks>
public sealed class BarycentricInterpolator
{
    private double[] _xi;
    private double[] _yi;
    private double[] _wi;
    private readonly double _invCapacity;
    private double[]? _diff;          // n×n differentiation matrix
    private BarycentricInterpolator? _diffInterp;

    /// <summary>Creates the interpolant; <paramref name="wi"/> overrides the weights.</summary>
    public BarycentricInterpolator(double[] xi, double[] yi, double[]? wi = null)
    {
        ArgumentNullException.ThrowIfNull(xi);
        ArgumentNullException.ThrowIfNull(yi);
        if (xi.Length != yi.Length) throw new ArgumentException("xi and yi must have the same length.");
        if (xi.Length == 0) throw new ArgumentException("At least one node is needed.");
        _xi = (double[])xi.Clone();
        _yi = (double[])yi.Clone();
        var range = _xi.Max() - _xi.Min();
        _invCapacity = range > 0 ? 4.0 / range : 1.0;
        if (wi is not null)
        {
            if (wi.Length != xi.Length) throw new ArgumentException("wi must have the length of xi.");
            _wi = (double[])wi.Clone();
        }
        else
        {
            var n = _xi.Length;
            _wi = new double[n];
            for (var i = 0; i < n; i++)
            {
                var prod = 1.0;
                for (var j = 0; j < n; j++) if (j != i) prod *= _invCapacity * (_xi[i] - _xi[j]);
                if (prod == 0.0) throw new ArgumentException("Interpolation points xi must be distinct.");
                _wi[i] = 1.0 / prod;
            }
        }
    }

    /// <summary>Nodes.</summary>
    public double[] Xi => (double[])_xi.Clone();

    /// <summary>Barycentric weights.</summary>
    public double[] Weights => (double[])_wi.Clone();

    /// <summary>Replaces the values (weights depend only on the nodes).</summary>
    public void SetYi(double[] yi)
    {
        if (yi.Length != _xi.Length) throw new ArgumentException("yi must have the length of xi.");
        _yi = (double[])yi.Clone();
        _diffInterp = null;
    }

    /// <summary>Adds nodes and values, updating the weights in O(n·k) (scipy's <c>add_xi</c>).</summary>
    public void AddXi(double[] xi, double[] yi)
    {
        if (xi.Length != yi.Length) throw new ArgumentException("xi and yi must have the same length.");
        // scipy 1.17.1's add_xi multiplies by (x_new - x_old) for the old weights and by
        // (x_old - x_new) for the new one, so the signs can disagree and the interpolant comes
        // out wrong (adding 3 nodes to 4 does); this uses consistent signs.
        var oldN = _xi.Length;
        var nx = _xi.Concat(xi).ToArray();
        var n = nx.Length;
        var w = new double[n];
        for (var i = 0; i < oldN; i++) w[i] = 1.0 / _wi[i];
        for (var j = oldN; j < n; j++)
        {
            for (var i = 0; i < j; i++) w[i] *= _invCapacity * (nx[i] - nx[j]);
            var prod = 1.0;
            for (var i = 0; i < j; i++) prod *= _invCapacity * (nx[j] - nx[i]);
            w[j] = prod;
        }
        for (var i = 0; i < n; i++) w[i] = 1.0 / w[i];
        _xi = nx; _wi = w; _yi = _yi.Concat(yi).ToArray();
        _diff = null; _diffInterp = null;
    }

    /// <summary>Value at x.</summary>
    public double Evaluate(double x)
    {
        double num = 0, den = 0;
        for (var j = 0; j < _xi.Length; j++)
        {
            var d = x - _xi[j];
            if (d == 0) return _yi[j];
            var c = _wi[j] / d;
            num += c * _yi[j];
            den += c;
        }
        return num / den;
    }

    /// <summary>Values at every x.</summary>
    public double[] Evaluate(double[] x)
    {
        var r = new double[x.Length];
        InterpolateSupport.ForChunks(x.Length, 4L * _xi.Length, (s, e) => { for (var i = s; i < e; i++) r[i] = Evaluate(x[i]); });
        return r;
    }

    /// <summary>The <paramref name="der"/>-th derivative at every x.</summary>
    public double[] Derivative(double[] x, int der = 1)
    {
        if (der < 0) throw new ArgumentException("der must be non-negative.");
        if (der == 0) return Evaluate(x);
        if (der >= _xi.Length) return new double[x.Length];
        return DiffInterpolant().Derivative(x, der - 1);
    }

    /// <summary>Derivatives 0..der-1 at every x: result[d][i] (scipy's <c>derivatives</c>).</summary>
    public double[][] Derivatives(double[] x, int? der = null)
    {
        var count = der ?? _xi.Length;
        var res = new double[count][];
        for (var d = 0; d < count; d++) res[d] = Derivative(x, d);
        return res;
    }

    private BarycentricInterpolator DiffInterpolant()
    {
        if (_diffInterp is not null) return _diffInterp;
        var n = _xi.Length;
        if (_diff is null)
        {
            var c = new double[n * n];
            for (var i = 0; i < n; i++)
            {
                var rowSum = 0.0;
                for (var j = 0; j < n; j++)
                {
                    if (i == j) continue;
                    var v = _wi[j] / ((_xi[i] - _xi[j]) * _wi[i]);
                    c[i * n + j] = v;
                    rowSum += v;
                }
                c[i * n + i] = -rowSum;
            }
            _diff = c;
        }
        var dy = new double[n];
        for (var i = 0; i < n; i++)
        {
            var s = 0.0;
            for (var j = 0; j < n; j++) s += _diff[i * n + j] * _yi[j];
            dy[i] = s;
        }
        _diffInterp = new BarycentricInterpolator(_xi, dy, _wi) { _diff = _diff };
        return _diffInterp;
    }
}

/// <summary>
/// Hermite–Newton polynomial interpolation (scipy.interpolate.KroghInterpolator): a node
/// repeated r times takes the value and r-1 derivatives in consecutive yi entries. Scalar
/// values. Above about 30 nodes the divided differences lose accuracy, as scipy warns.
/// </summary>
public sealed class KroghInterpolator
{
    private readonly double[] _xi;
    private readonly double[] _c;
    private readonly int _n;

    /// <summary>Builds the Newton coefficients.</summary>
    public KroghInterpolator(double[] xi, double[] yi)
    {
        ArgumentNullException.ThrowIfNull(xi);
        ArgumentNullException.ThrowIfNull(yi);
        if (xi.Length != yi.Length) throw new ArgumentException("xi and yi must have the same length.");
        _xi = (double[])xi.Clone();
        _n = xi.Length;
        var n = _n;
        var c = new double[n + 1];
        c[0] = yi[0];
        var vk = new double[n];
        for (var k = 1; k < n; k++)
        {
            var s = 0;
            while (s <= k && xi[k - s] == xi[k]) s++;
            s--;
            vk[0] = yi[k] / InterpolateSupport.Factorial(s);
            for (var i = 0; i < k - s; i++)
            {
                if (xi[i] == xi[k]) throw new ArgumentException("Elements of `xi` can't be equal.");
                if (s == 0) vk[i + 1] = (c[i] - vk[i]) / (xi[i] - xi[k]);
                else vk[i + 1] = (vk[i + 1] - vk[i]) / (xi[i] - xi[k]);
            }
            c[k] = vk[k - s];
        }
        _c = c;
    }

    /// <summary>Value at x.</summary>
    public double Evaluate(double x)
    {
        var pi = 1.0;
        var p = _c[0];
        for (var k = 1; k < _n; k++)
        {
            pi *= x - _xi[k - 1];
            p += pi * _c[k];
        }
        return p;
    }

    /// <summary>Values at every x.</summary>
    public double[] Evaluate(double[] x) => x.Select(Evaluate).ToArray();

    /// <summary>Derivatives 0..der-1 at x (default: all n), scipy's <c>derivatives</c>.</summary>
    public double[] Derivatives(double x, int? der = null)
    {
        var n = _n;
        var d = der ?? n;
        var pi = new double[n];
        var w = new double[n];
        pi[0] = 1;
        var p = _c[0];
        for (var k = 1; k < n; k++)
        {
            w[k - 1] = x - _xi[k - 1];
            pi[k] = w[k - 1] * pi[k - 1];
            p += pi[k] * _c[k];
        }
        var cn = new double[Math.Max(d, n + 1)];
        for (var k = 0; k <= n; k++) cn[k] += _c[k];
        cn[0] = p;
        for (var k = 1; k < n; k++)
        {
            for (var i = 1; i < n - k + 1; i++)
            {
                pi[i] = w[k + i - 1] * pi[i - 1] + pi[i];
                cn[k] += pi[i] * cn[k + i];
            }
            cn[k] *= InterpolateSupport.Factorial(k);
        }
        cn[n] = 0;
        return cn[..d];
    }

    /// <summary>The <paramref name="der"/>-th derivative at x.</summary>
    public double Derivative(double x, int der = 1) => Derivatives(x, der + 1)[der];
}

public static partial class Interpolate
{
    /// <summary>Barycentric interpolation (scipy's <c>barycentric_interpolate</c>).</summary>
    public static double[] BarycentricInterpolate(double[] xi, double[] yi, double[] x, int der = 0) =>
        new BarycentricInterpolator(xi, yi).Derivative(x, der);

    /// <summary>Krogh interpolation (scipy's <c>krogh_interpolate</c>).</summary>
    public static double[] KroghInterpolate(double[] xi, double[] yi, double[] x, int der = 0)
    {
        var k = new KroghInterpolator(xi, yi);
        return x.Select(v => der == 0 ? k.Evaluate(v) : k.Derivative(v, der)).ToArray();
    }

    /// <summary>
    /// Lagrange interpolating polynomial (scipy.interpolate.lagrange), coefficients highest
    /// power first (numpy <c>poly1d</c> order). Numerically unstable beyond ~20 points, as
    /// scipy warns; prefer <see cref="BarycentricInterpolator"/>.
    /// </summary>
    public static double[] Lagrange(double[] x, double[] w)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(w);
        if (x.Length != w.Length) throw new ArgumentException("x and w must have the same length.");
        var m = x.Length;
        var p = new double[Math.Max(m, 1)];
        for (var j = 0; j < m; j++)
        {
            // pt = w[j] * Π_{k≠j} (X - x_k)/(x_j - x_k), ascending coefficients
            var pt = new double[m];
            pt[0] = w[j];
            var deg = 0;
            for (var k = 0; k < m; k++)
            {
                if (k == j) continue;
                var fac = x[j] - x[k];
                for (var q = deg + 1; q >= 1; q--) pt[q] = (pt[q - 1] - x[k] * pt[q]) / fac;
                pt[0] = -x[k] * pt[0] / fac;
                deg++;
            }
            for (var q = 0; q < m; q++) p[q] += pt[q];
        }
        // trim leading zeros (poly1d does), return highest first
        var hi = p.Length - 1;
        while (hi > 0 && p[hi] == 0) hi--;
        var res = new double[hi + 1];
        for (var q = 0; q <= hi; q++) res[q] = p[hi - q];
        return res;
    }

    /// <summary>
    /// Taylor polynomial of f at x estimated by interpolating at Chebyshev points of radius
    /// <paramref name="scale"/> (scipy.interpolate.approximate_taylor_polynomial).
    /// Coefficients of the polynomial in (t - x), highest power first.
    /// </summary>
    public static double[] ApproximateTaylorPolynomial(Func<double, double> f, double x, int degree, double scale, int? order = null)
    {
        ArgumentNullException.ThrowIfNull(f);
        var ord = order ?? degree;
        var n = ord + 1;
        // numpy: linspace(0, pi, n, endpoint=n % 1) → endpoint=False
        var xs = new double[n];
        for (var i = 0; i < n; i++) xs[i] = scale * Math.Cos(Math.PI * i / n) + x;
        var p = new KroghInterpolator(xs, xs.Select(f).ToArray());
        var d = p.Derivatives(x, degree + 1);
        var res = new double[degree + 1];
        for (var i = 0; i <= degree; i++) res[degree - i] = d[i] / InterpolateSupport.Factorial(i);
        return res;
    }

    /// <summary>
    /// Padé approximant p/q of a power series (scipy.interpolate.pade): q of order m, p of
    /// order n (default len(an) - 1 - m). Coefficients highest power first, q normalised to
    /// q(0) = 1.
    /// </summary>
    public static (double[] P, double[] Q) Pade(double[] an, int m, int? n = null)
    {
        ArgumentNullException.ThrowIfNull(an);
        var nn = n ?? an.Length - 1 - m;
        if (n is null && nn < 0) throw new ArgumentException("Order of q <m> must be smaller than len(an)-1.");
        if (nn < 0) throw new ArgumentException("Order of p <n> must be greater than 0.");
        var N = m + nn;
        if (N > an.Length - 1) throw new ArgumentException("Order of q+p <m+n> must be smaller than len(an).");
        var size = N + 1;
        var c = new double[size * size];
        for (var r = 0; r <= N; r++) if (r <= nn) c[r * size + r] = 1;
        for (var row = 1; row <= m; row++)
            for (var col = 0; col < row; col++) c[row * size + nn + 1 + col] = -an[row - 1 - col];
        for (var row = m + 1; row <= N; row++)
            for (var col = 0; col < m; col++) c[row * size + nn + 1 + col] = -an[row - 1 - col];
        var rhs = an[..(N + 1)];
        var pq = InterpolateSupport.DenseSolve(c, size, rhs, 1);
        var p = new double[nn + 1];
        for (var i = 0; i <= nn; i++) p[nn - i] = pq[i];
        var q = new double[m + 1];
        q[m] = 1.0;
        for (var i = 0; i < m; i++) q[m - 1 - i] = pq[nn + 1 + i];
        return (p, q);
    }
}

/// <summary>Shared evaluation, poles, residues and zeros of barycentric rational
/// interpolants (scipy's <c>_BarycentricRational</c>). Scalar values.</summary>
public abstract class BarycentricRationalBase
{
    private protected double[] _zj = [];
    private protected double[] _fj = [];
    private protected double[] _wj = [];

    /// <summary>Support points.</summary>
    public double[] SupportPoints => (double[])_zj.Clone();

    /// <summary>Values at the support points.</summary>
    public double[] SupportValues => (double[])_fj.Clone();

    /// <summary>Barycentric weights.</summary>
    public double[] Weights => (double[])_wj.Clone();

    /// <summary>r(z) = Σ w f/(z - z_j) / Σ w/(z - z_j); exact at the support points.</summary>
    public double Evaluate(double z)
    {
        if (double.IsInfinity(z))
        {
            double a = 0, b = 0;
            for (var j = 0; j < _zj.Length; j++) { a += _wj[j] * _fj[j]; b += _wj[j]; }
            return a / b;
        }
        double num = 0, den = 0;
        for (var j = 0; j < _zj.Length; j++)
        {
            var d = z - _zj[j];
            if (d == 0) return _fj[j];
            var c = 1.0 / d;
            num += c * _wj[j] * _fj[j];
            den += c * _wj[j];
        }
        return num / den;
    }

    /// <summary>Values at every z.</summary>
    public double[] Evaluate(double[] z) => z.Select(Evaluate).ToArray();

    /// <summary>
    /// Finite poles: zeros of Σ w_j/(z - z_j). scipy solves the arrowhead pencil
    /// (E, B); here the pencil is deflated to an ordinary m×m eigenproblem
    /// (I - 1wᵀ/wᵀ1)(Z - cI) whose spectrum is the poles minus c plus one spurious zero, with
    /// the shift c placed outside the support so the spurious eigenvalue is recognisable.
    /// </summary>
    public Complex[] Poles() => ZerosOf(_wj);

    /// <summary>Zeros of the rational function: zeros of Σ w_j f_j/(z - z_j).</summary>
    public Complex[] Roots() => ZerosOf(_wj.Zip(_fj, (w, f) => w * f).ToArray());

    /// <summary>Residues at <see cref="Poles"/>.</summary>
    public Complex[] Residues()
    {
        var poles = Poles();
        var res = new Complex[poles.Length];
        for (var p = 0; p < poles.Length; p++)
        {
            Complex n = 0, dd = 0;
            for (var j = 0; j < _zj.Length; j++)
            {
                var c = 1.0 / (poles[p] - _zj[j]);
                n += c * _fj[j] * _wj[j];
                dd -= c * c * _wj[j];
            }
            res[p] = n / dd;
        }
        return res;
    }

    private Complex[] ZerosOf(double[] u)
    {
        var m = _zj.Length;
        if (m < 2) return [];
        var su = u.Sum();
        if (su == 0) throw new InvalidOperationException("Degenerate weights: the denominator has a pole at infinity.");
        var c = 2 * _zj.Select(Math.Abs).Max() + 1;
        var mat = new double[m * m];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < m; j++)
            {
                var proj = (i == j ? 1.0 : 0.0) - u[j] / su;
                mat[i * m + j] = proj * (_zj[j] - c);
            }
        var ev = Linalg.Eigvals(new NdArray(mat, m, m));
        // drop the spurious eigenvalue (the one nearest 0, i.e. z = c)
        var drop = 0;
        for (var i = 1; i < ev.Length; i++) if (ev[i].Magnitude < ev[drop].Magnitude) drop = i;
        var res = new List<Complex>();
        for (var i = 0; i < ev.Length; i++)
        {
            if (i == drop) continue;
            // The deflated matrix is not well conditioned; polish on q(z) = Σ u_j/(z - z_j).
            var z = ev[i] + c;
            for (var it = 0; it < 4; it++)
            {
                Complex q = 0, dq = 0;
                var hit = false;
                for (var j = 0; j < m; j++)
                {
                    var d = z - _zj[j];
                    if (d == Complex.Zero) { hit = true; break; }
                    var inv = 1.0 / d;
                    q += u[j] * inv;
                    dq -= u[j] * inv * inv;
                }
                if (hit || dq == Complex.Zero) break;
                var step = q / dq;
                if (!double.IsFinite(step.Real) || !double.IsFinite(step.Imaginary)) break;
                z -= step;
            }
            res.Add(z);
        }
        return [.. res];
    }
}

/// <summary>
/// Floater–Hormann barycentric rational interpolation of blending degree d
/// (scipy.interpolate.FloaterHormannInterpolator): no real poles, convergence O(h^(d+1)).
/// </summary>
public sealed class FloaterHormannInterpolator : BarycentricRationalBase
{
    /// <summary>Interpolates; non-finite values and duplicate x are dropped as in scipy.</summary>
    public FloaterHormannInterpolator(double[] points, double[] values, int d = 3)
    {
        ArgumentNullException.ThrowIfNull(points);
        ArgumentNullException.ThrowIfNull(values);
        if (points.Length != values.Length) throw new ArgumentException("points and values must have the same length.");
        if (d < 0 || d >= points.Length) throw new ArgumentException("`d` must satisfy 0 <= d < n");
        foreach (var v in points) if (!double.IsFinite(v)) throw new ArgumentException("`x` must be finite.");
        (_zj, _fj) = Clean(points, values);
        var n = _zj.Length;
        var z = _zj;
        var w = new double[n];
        for (var k = 0; k < n; k++)
        {
            for (var i = Math.Max(k - d, 0); i < Math.Min(k + 1, n - d); i++)
            {
                var prod = 1.0;
                for (var j = i; j <= i + d; j++) if (j != k) prod *= Math.Abs(z[k] - z[j]);
                w[k] += 1 / prod;
            }
            if (((k - d) % 2 + 2) % 2 == 1) w[k] = -w[k];
        }
        _wj = w;
    }

    internal static (double[] Z, double[] F) Clean(double[] x, double[] y)
    {
        var pairs = new SortedDictionary<double, double>();
        for (var i = 0; i < x.Length; i++)
            if (double.IsFinite(y[i]) && !pairs.ContainsKey(x[i])) pairs[x[i]] = y[i];
        return (pairs.Keys.ToArray(), pairs.Values.ToArray());
    }
}

/// <summary>
/// AAA rational approximation (scipy.interpolate.AAA; Nakatsukasa, Sète &amp; Trefethen 2018)
/// for real data: greedy support-point selection with weights from the smallest right
/// singular vector of the Loewner matrix, then Froissart-doublet clean-up.
/// </summary>
/// <remarks>Weights are defined up to a common sign (and scale), so they may differ from
/// scipy's by a factor of -1; values, poles and residues do not.</remarks>
public sealed class AAA : BarycentricRationalBase
{
    /// <summary>Maximum error after each step.</summary>
    public double[] Errors { get; }

    private readonly double[] _points, _values;

    /// <summary>Builds the approximant to relative tolerance <paramref name="rtol"/> (default
    /// eps^0.75) with at most <paramref name="maxTerms"/> support points.</summary>
    public AAA(double[] x, double[] y, double? rtol = null, int maxTerms = 100, bool cleanUp = true, double cleanUpTol = 1e-13)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (maxTerms < 1) throw new ArgumentException("`max_terms` must be an integer value greater than or equal to one.");
        if (x.Length != y.Length) throw new ArgumentException("x and y must have the same length.");
        foreach (var v in x) if (!double.IsFinite(v)) throw new ArgumentException("`x` must be finite.");
        var (z, f) = FloaterHormannInterpolator.Clean(x, y);
        _points = z; _values = f;
        var M = z.Length;
        var tol = rtol ?? Math.Pow(2.220446049250313e-16, 0.75);
        var atol = tol * f.Select(Math.Abs).DefaultIfEmpty(0).Max();
        var mask = Enumerable.Repeat(true, M).ToArray();
        var zj = new List<double>();
        var fj = new List<double>();
        var C = new double[M, maxTerms];
        var A = new double[M, maxTerms];
        var errors = new List<double>();
        var mean = f.Average();
        var R = Enumerable.Repeat(mean, M).ToArray();
        double[] wj = [];
        var illTol = 1 / (3 * 2.220446049250313e-16);
        var ill = false;
        for (var m = 0; m < maxTerms; m++)
        {
            var jj = -1;
            var best = -1.0;
            for (var i = 0; i < M; i++)
                if (mask[i] && Math.Abs(f[i] - R[i]) > best) { best = Math.Abs(f[i] - R[i]); jj = i; }
            zj.Add(z[jj]); fj.Add(f[jj]);
            for (var i = 0; i < M; i++)
            {
                C[i, m] = 1 / (z[i] - z[jj]);
                A[i, m] = (f[i] - f[jj]) * C[i, m];
            }
            mask[jj] = false;
            var rowsIdx = Enumerable.Range(0, M).Where(i => mask[i]).ToArray();
            var rows = rowsIdx.Length;
            var cols = m + 1;
            if (rows >= cols)
            {
                var sub = new double[rows * cols];
                for (var r = 0; r < rows; r++)
                    for (var q = 0; q < cols; q++) sub[r * cols + q] = A[rowsIdx[r], q];
                var colNorm = new double[cols];
                LinalgSvdResult svd;
                if (!ill)
                {
                    svd = Linalg.Svd(new NdArray(sub, rows, cols), fullMatrices: false);
                    if (svd.S[0] / svd.S[^1] > illTol) ill = true;
                }
                else svd = null!;
                if (ill)
                {
                    for (var q = 0; q < cols; q++)
                    {
                        var s2 = 0.0;
                        for (var r = 0; r < rows; r++) s2 += sub[r * cols + q] * sub[r * cols + q];
                        colNorm[q] = Math.Sqrt(s2);
                    }
                    var scaled = new double[sub.Length];
                    for (var r = 0; r < rows; r++)
                        for (var q = 0; q < cols; q++) scaled[r * cols + q] = sub[r * cols + q] / colNorm[q];
                    svd = Linalg.Svd(new NdArray(scaled, rows, cols), fullMatrices: false);
                }
                var vh = svd.Vh.ToArray();
                var last = svd.S.Length - 1;
                wj = new double[cols];
                for (var q = 0; q < cols; q++) wj[q] = vh[last * cols + q];
                if (ill) for (var q = 0; q < cols; q++) wj[q] /= colNorm[q];
            }
            else
            {
                var sub = new double[Math.Max(rows, 1) * cols];
                for (var r = 0; r < rows; r++)
                    for (var q = 0; q < cols; q++) sub[r * cols + q] = A[rowsIdx[r], q];
                var ns = Linalg.NullSpace(new NdArray(sub, Math.Max(rows, 1), cols));
                var nm = ns.Shape[1];
                var nsa = ns.ToArray();
                wj = new double[cols];
                for (var q = 0; q < cols; q++)
                {
                    var s = 0.0;
                    for (var r = 0; r < nm; r++) s += nsa[q * nm + r];
                    wj[q] = s / Math.Sqrt(nm);
                }
            }
            var maxErr = 0.0;
            for (var i = 0; i < M; i++)
            {
                double N = 0, D = 0;
                for (var q = 0; q < cols; q++)
                {
                    if (wj[q] == 0) continue;
                    N += C[i, q] * wj[q] * fj[q];
                    D += C[i, q] * wj[q];
                }
                R[i] = double.IsInfinity(D) || double.IsNaN(D) ? f[i] : N / D;
                maxErr = Math.Max(maxErr, Math.Abs(f[i] - R[i]));
            }
            errors.Add(maxErr);
            if (maxErr <= atol) break;
        }
        Errors = [.. errors];
        var keep = Enumerable.Range(0, wj.Length).Where(q => wj[q] != 0).ToArray();
        _zj = keep.Select(q => zj[q]).ToArray();
        _fj = keep.Select(q => fj[q]).ToArray();
        _wj = keep.Select(q => wj[q]).ToArray();
        if (cleanUp) CleanUp(cleanUpTol);
    }

    /// <summary>Removes Froissart doublets (spurious pole–zero pairs with tiny residues) and
    /// recomputes the weights by least squares; returns how many were removed.</summary>
    public int CleanUp(double cleanUpTol = 1e-13)
    {
        var logSum = 0.0;
        foreach (var v in _values) logSum += Math.Log(Math.Abs(v));
        var gmean = Math.Exp(logSum / _values.Length);
        var poles = Poles();
        var res = Residues();
        var bad = new List<int>();
        for (var p = 0; p < poles.Length; p++)
        {
            var dist = _points.Min(z => (poles[p] - z).Magnitude);
            if (res[p].Magnitude / dist < cleanUpTol * gmean) bad.Add(p);
        }
        if (bad.Count == 0) return 0;
        var remove = new HashSet<int>();
        foreach (var p in bad)
        {
            var closest = 0;
            for (var j = 1; j < _zj.Length; j++)
                if ((_zj[j] - poles[p]).Magnitude < (_zj[closest] - poles[p]).Magnitude) closest = j;
            remove.Add(closest);
        }
        _zj = _zj.Where((_, j) => !remove.Contains(j)).ToArray();
        _fj = _fj.Where((_, j) => !remove.Contains(j)).ToArray();
        var m = _zj.Length;
        var idx = Enumerable.Range(0, _points.Length).Where(i => !_zj.Contains(_points[i])).ToArray();
        var a = new double[idx.Length * m];
        for (var r = 0; r < idx.Length; r++)
            for (var j = 0; j < m; j++)
            {
                var c = 1 / (_points[idx[r]] - _zj[j]);
                a[r * m + j] = _values[idx[r]] * c - c * _fj[j];
            }
        var svd = Linalg.Svd(new NdArray(a, idx.Length, m), fullMatrices: true);
        var vh = svd.Vh.ToArray();
        _wj = new double[m];
        for (var j = 0; j < m; j++) _wj[j] = vh[(m - 1) * m + j];
        return bad.Count;
    }
}
