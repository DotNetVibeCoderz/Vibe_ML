using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;
using static Gravicode.Science.SciNet.InterpolateSupport;

namespace Gravicode.Science.SciNet;

/// <summary>Radial basis function of <see cref="RBFInterpolator"/> (scipy's kernel names),
/// with r = ε‖x - y‖.</summary>
public enum RbfKernel
{
    /// <summary>-r.</summary>
    Linear,
    /// <summary>r² log r (default).</summary>
    ThinPlateSpline,
    /// <summary>r³.</summary>
    Cubic,
    /// <summary>-r⁵.</summary>
    Quintic,
    /// <summary>-√(1 + r²).</summary>
    Multiquadric,
    /// <summary>1/√(1 + r²).</summary>
    InverseMultiquadric,
    /// <summary>1/(1 + r²).</summary>
    InverseQuadratic,
    /// <summary>exp(-r²).</summary>
    Gaussian,
}

/// <summary>
/// Radial basis function interpolation in N dimensions (scipy.interpolate.RBFInterpolator):
/// <c>f(x) = Σ a_i φ(ε‖x - y_i‖) + Σ b_j p_j(x)</c> with a polynomial tail of the given degree,
/// solving the symmetric saddle-point system exactly as scipy builds it — kernel matrix on
/// ε·y, monomials on the data box shifted and scaled to [-1, 1], smoothing on the diagonal.
/// </summary>
/// <remarks>
/// <para>Global mode solves one dense (p + r) system with partially pivoted LU; with
/// <c>neighbors</c>, each query uses only its k nearest data points (KD-tree), and queries that
/// share the same neighbour set share one solve, as in scipy.</para>
/// <para>Evaluation is threaded and allocation-free per point (no p × q evaluation matrix is
/// formed).</para>
/// </remarks>
public sealed class RBFInterpolator
{
    private readonly double[] _y;     // (p, ndim)
    private readonly double[] _d;     // (p, nt)
    private readonly double[] _smoothing;
    private readonly int _p, _ndim, _nt;
    private readonly int[] _dShape;
    private readonly int[] _powers;   // (r, ndim)
    private readonly int _r;
    private readonly KDTree? _tree;
    private readonly int? _neighbors;
    private readonly double[]? _shift, _scale, _coeffs; // global mode

    /// <summary>Kernel.</summary>
    public RbfKernel Kernel { get; }

    /// <summary>Shape parameter ε.</summary>
    public double Epsilon { get; }

    /// <summary>Degree of the polynomial tail (-1: none).</summary>
    public int Degree { get; }

    private static readonly RbfKernel[] ScaleInvariant = [RbfKernel.Linear, RbfKernel.ThinPlateSpline, RbfKernel.Cubic, RbfKernel.Quintic];

    private static int MinDegree(RbfKernel k) => k switch
    {
        RbfKernel.Multiquadric or RbfKernel.Linear => 0,
        RbfKernel.ThinPlateSpline or RbfKernel.Cubic => 1,
        RbfKernel.Quintic => 2,
        _ => -1,
    };

    /// <summary>
    /// Builds the interpolant through data y (p, ndim) with values d (p, …).
    /// <paramref name="epsilon"/> is required unless the kernel is scale invariant (linear,
    /// thin-plate, cubic, quintic; default 1). <paramref name="degree"/> defaults to the
    /// kernel's minimum (0 if none); <paramref name="smoothing"/> is a scalar or one value per
    /// point.
    /// </summary>
    public RBFInterpolator(NdArray y, NdArray d, int? neighbors = null, double[]? smoothing = null,
        RbfKernel kernel = RbfKernel.ThinPlateSpline, double? epsilon = null, int? degree = null)
    {
        ArgumentNullException.ThrowIfNull(y);
        ArgumentNullException.ThrowIfNull(d);
        if (y.Rank != 2) throw new ArgumentException("`y` must be a 2-dimensional array.");
        _p = y.Shape[0];
        _ndim = y.Shape[1];
        if (d.Shape[0] != _p) throw new ArgumentException($"Expected the first axis of `d` to have length {_p}.");
        _y = CopyFinite(y.ToArray(), "y");
        _d = d.ToArray();
        _dShape = d.Shape[1..].ToArray();
        _nt = Product(_dShape);
        if (smoothing is null) _smoothing = new double[_p];
        else if (smoothing.Length == 1) _smoothing = Enumerable.Repeat(smoothing[0], _p).ToArray();
        else if (smoothing.Length == _p) _smoothing = (double[])smoothing.Clone();
        else throw new ArgumentException($"Expected `smoothing` to be a scalar or have shape ({_p},).");
        Kernel = kernel;
        if (epsilon is null)
        {
            if (Array.IndexOf(ScaleInvariant, kernel) < 0)
                throw new ArgumentException("`epsilon` must be specified if `kernel` is not one of linear, thin_plate_spline, cubic, quintic.");
            Epsilon = 1.0;
        }
        else Epsilon = epsilon.Value;
        var minDeg = MinDegree(kernel);
        Degree = degree ?? Math.Max(minDeg, 0);
        if (Degree < -1) throw new ArgumentException("`degree` must be at least -1.");
        _powers = MonomialPowers(_ndim, Degree);
        _r = _powers.Length / Math.Max(_ndim, 1);
        int nobs;
        if (neighbors is null) nobs = _p;
        else
        {
            _neighbors = Math.Min(neighbors.Value, _p);
            if (_neighbors < 1) throw new ArgumentException("neighbors must be positive.");
            nobs = _neighbors.Value;
        }
        if (_r > nobs) throw new ArgumentException($"At least {_r} data points are required when `degree` is {Degree} and the number of dimensions is {_ndim}.");
        if (_neighbors is null)
        {
            var all = Enumerable.Range(0, _p).ToArray();
            (_shift, _scale, _coeffs) = BuildAndSolve(all);
        }
        else _tree = new KDTree(new NdArray((double[])_y.Clone(), _p, _ndim));
    }

    /// <summary>Scalar-valued convenience constructor with a scalar smoothing.</summary>
    public RBFInterpolator(double[,] y, double[] d, int? neighbors = null, double smoothing = 0.0,
        RbfKernel kernel = RbfKernel.ThinPlateSpline, double? epsilon = null, int? degree = null)
        : this(NdArray.FromArray(y), new NdArray((double[])d.Clone(), d.Length), neighbors, [smoothing], kernel, epsilon, degree) { }

    /// <summary>scipy's <c>_monomial_powers</c>: exponents of all monomials of total degree ≤
    /// degree, in combinations-with-replacement order.</summary>
    internal static int[] MonomialPowers(int ndim, int degree)
    {
        var res = new List<int[]>();
        for (var deg = 0; deg <= degree; deg++)
        {
            var combo = new int[deg];
            void Rec(int pos, int start)
            {
                if (pos == deg)
                {
                    var pw = new int[ndim];
                    foreach (var v in combo) pw[v]++;
                    res.Add(pw);
                    return;
                }
                for (var v = start; v < ndim; v++) { combo[pos] = v; Rec(pos + 1, v); }
            }
            Rec(0, 0);
        }
        var flat = new int[res.Count * ndim];
        for (var i = 0; i < res.Count; i++) Array.Copy(res[i], 0, flat, i * ndim, ndim);
        return flat;
    }

    internal static double Phi(RbfKernel k, double r) => k switch
    {
        RbfKernel.Linear => -r,
        RbfKernel.ThinPlateSpline => r == 0 ? 0.0 : r * r * Math.Log(r),
        RbfKernel.Cubic => r * r * r,
        RbfKernel.Quintic => -(r * r * r * r * r),
        RbfKernel.Multiquadric => -Math.Sqrt(r * r + 1),
        RbfKernel.InverseMultiquadric => 1 / Math.Sqrt(r * r + 1),
        RbfKernel.InverseQuadratic => 1 / (r * r + 1),
        _ => Math.Exp(-r * r),
    };

    private double Monomial(ReadOnlySpan<double> xhat, int j)
    {
        var v = 1.0;
        for (var a = 0; a < _ndim; a++)
        {
            var pw = _powers[j * _ndim + a];
            for (var e = 0; e < pw; e++) v *= xhat[a];
        }
        return v;
    }

    private (double[] Shift, double[] Scale, double[] Coeffs) BuildAndSolve(int[] idx)
    {
        int p = idx.Length, r = _r, nd = _ndim, nt = _nt;
        var shift = new double[nd];
        var scale = new double[nd];
        for (var a = 0; a < nd; a++)
        {
            double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
            foreach (var i in idx) { lo = Math.Min(lo, _y[i * nd + a]); hi = Math.Max(hi, _y[i * nd + a]); }
            shift[a] = (hi + lo) / 2;
            scale[a] = (hi - lo) / 2;
            if (scale[a] == 0.0) scale[a] = 1.0;
        }
        var n = p + r;
        var lhs = new double[n * n];
        var eps = Epsilon;
        for (var i = 0; i < p; i++)
            for (var j = 0; j <= i; j++)
            {
                var s = 0.0;
                for (var a = 0; a < nd; a++)
                {
                    var df = _y[idx[i] * nd + a] * eps - _y[idx[j] * nd + a] * eps;
                    s += df * df;
                }
                var v = Phi(Kernel, Math.Sqrt(s));
                lhs[i * n + j] = v;
                lhs[j * n + i] = v;
            }
        Span<double> yhat = stackalloc double[nd];
        for (var i = 0; i < p; i++)
        {
            for (var a = 0; a < nd; a++) yhat[a] = (_y[idx[i] * nd + a] - shift[a]) / scale[a];
            for (var j = 0; j < r; j++)
            {
                var v = Monomial(yhat, j);
                lhs[i * n + p + j] = v;
                lhs[(p + j) * n + i] = v;
            }
            lhs[i * n + i] += _smoothing[idx[i]];
        }
        var rhs = new double[n * nt];
        for (var i = 0; i < p; i++) Array.Copy(_d, idx[i] * nt, rhs, i * nt, nt);
        double[] coeffs;
        try
        {
            coeffs = DenseSolve(lhs, n, rhs, nt);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            var msg = "Singular matrix.";
            if (r > 0)
            {
                var pm = new double[p * r];
                for (var i = 0; i < p; i++)
                {
                    for (var a = 0; a < nd; a++) yhat[a] = (_y[idx[i] * nd + a] - shift[a]) / scale[a];
                    for (var j = 0; j < r; j++) pm[i * r + j] = Monomial(yhat, j);
                }
                var sv = Linalg.Svdvals(new NdArray(pm, p, r));
                var rank = sv.Count(s => s > sv[0] * Math.Max(p, r) * 2.220446049250313e-16);
                if (rank < r) msg = $"Singular matrix. The matrix of monomials evaluated at the data point coordinates does not have full column rank ({rank}/{r}).";
            }
            throw new InvalidOperationException(msg, ex);
        }
        return (shift, scale, coeffs);
    }

    /// <summary>Evaluates at points x (q, ndim); result shape (q, …value shape).</summary>
    public NdArray Evaluate(NdArray x)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Rank != 2) throw new ArgumentException("`x` must be a 2-dimensional array.");
        if (x.Shape[1] != _ndim) throw new ArgumentException($"Expected the second axis of `x` to have length {_ndim}.");
        var q = x.Shape[0];
        var xs = x.ToArray();
        var res = new double[q * _nt];
        if (_neighbors is null)
        {
            var all = Enumerable.Range(0, _p).ToArray();
            ForChunks(q, 30L * (_p + _r) * Math.Max(1, _nt / 4 + 1), (s, e) => EvalRange(xs, s, e, all, _shift!, _scale!, _coeffs!, res));
        }
        else
        {
            var k = _neighbors.Value;
            var (_, ind) = _tree!.Query(new NdArray(xs, q, _ndim), k);
            var groups = new Dictionary<string, (int[] Y, List<int> X)>();
            var row = new int[k];
            for (var i = 0; i < q; i++)
            {
                for (var j = 0; j < k; j++) row[j] = ind[i, j];
                Array.Sort(row);
                var key = string.Join(",", row);
                if (!groups.TryGetValue(key, out var g)) groups[key] = g = ((int[])row.Clone(), new List<int>());
                g.X.Add(i);
            }
            var list = groups.Values.ToArray();
            Parallel.For(0, list.Length, gi =>
            {
                var (yIdx, xIdx) = list[gi];
                var (shift, scale, coeffs) = BuildAndSolve(yIdx);
                foreach (var xi in xIdx) EvalRange(xs, xi, xi + 1, yIdx, shift, scale, coeffs, res);
            });
        }
        return new NdArray(res, [q, .. _dShape]);
    }

    /// <summary>Scalar-valued evaluation at the rows of an (q, ndim) array.</summary>
    public double[] Evaluate(double[,] x)
    {
        if (_nt != 1) throw new InvalidOperationException("Vector-valued data: use the NdArray overload.");
        return Evaluate(NdArray.FromArray(x)).ToArray();
    }

    private void EvalRange(double[] xs, int s, int e, int[] idx, double[] shift, double[] scale, double[] coeffs, double[] res)
    {
        int nd = _ndim, nt = _nt, p = idx.Length;
        var eps = Epsilon;
        Span<double> xe = stackalloc double[nd];
        Span<double> xhat = stackalloc double[nd];
        for (var i = s; i < e; i++)
        {
            for (var a = 0; a < nd; a++)
            {
                xe[a] = xs[i * nd + a] * eps;
                xhat[a] = (xs[i * nd + a] - shift[a]) / scale[a];
            }
            var o = res.AsSpan(i * nt, nt);
            o.Clear();
            for (var j = 0; j < p; j++)
            {
                var yi = idx[j] * nd;
                var s2 = 0.0;
                for (var a = 0; a < nd; a++)
                {
                    var df = xe[a] - _y[yi + a] * eps;
                    s2 += df * df;
                }
                var v = Phi(Kernel, Math.Sqrt(s2));
                if (nt == 1) o[0] += v * coeffs[j];
                else for (var q = 0; q < nt; q++) o[q] += v * coeffs[j * nt + q];
            }
            for (var j = 0; j < _r; j++)
            {
                var v = Monomial(xhat, j);
                for (var q = 0; q < nt; q++) o[q] += v * coeffs[(p + j) * nt + q];
            }
        }
    }
}
