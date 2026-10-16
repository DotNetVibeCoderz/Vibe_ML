using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Result of <see cref="Differentiate.Derivative"/>, mirroring scipy's result object.</summary>
/// <param name="Df">The derivative estimate.</param>
/// <param name="Error">Estimated error: |difference between the last two estimates|.</param>
/// <param name="Success">True when <see cref="Status"/> is 0.</param>
/// <param name="Status">0 converged; −1 the error estimate increased (round-off took over);
/// −2 maximum iterations reached; −3 a non-finite value was encountered.</param>
/// <param name="Nit">Iterations performed.</param>
/// <param name="Nfev">Function evaluations performed.</param>
/// <param name="X">The abscissa.</param>
public sealed record DerivativeResult(double Df, double Error, bool Success, int Status, int Nit, int Nfev, double X);

/// <summary>Result of <see cref="Differentiate.Jacobian"/>, <see cref="Differentiate.Gradient"/> or
/// <see cref="Differentiate.Hessian"/>.</summary>
/// <param name="Df">m×n Jacobian (for <c>Gradient</c>, a length-n vector; for <c>Hessian</c>, n×n).</param>
/// <param name="Error">Element-wise error estimates, same shape as <paramref name="Df"/>.</param>
/// <param name="Success">True when every element converged.</param>
/// <param name="Status">Element-wise status codes (see <see cref="DerivativeResult.Status"/>), row-major.</param>
/// <param name="Nit">Largest iteration count over the elements.</param>
/// <param name="Nfev">Total function evaluations.</param>
public sealed record JacobianResult(NdArray Df, NdArray Error, bool Success, int[] Status, int Nit, int Nfev);

/// <summary>
/// Numerical differentiation with error estimates, scipy.differentiate (new in SciPy 1.15).
/// </summary>
/// <remarks>
/// <para>
/// Each iteration applies a finite-difference stencil of the requested order with step
/// <c>h = initialStep · stepFactor^(−i)</c>; the error estimate is the change from the previous
/// iteration, and iteration stops once it is below <c>atol + rtol·|df|</c>. The stencil nodes are
/// <c>h·stepFactor^(−k)</c>, so consecutive iterations share all but the innermost pair of
/// evaluations: after the first iteration each one costs two evaluations (one when one-sided).
/// Weights come from Fornberg's algorithm, which stays accurate for the geometric node spacing
/// where solving the Vandermonde system directly would not.
/// </para>
/// <para>
/// Shrinking h trades truncation error for round-off. When the error estimate jumps by more than
/// 10× (scipy's heuristic) the round-off regime has been reached; scipy then reports the latest
/// estimate, whereas this returns the <em>previous</em> estimate and error, which is the better
/// one — status is still −1.
/// </para>
/// </remarks>
public static class Differentiate
{
    private const double DefaultAtol = 2.2250738585072014e-308; // smallest normal double

    /// <summary>
    /// First derivative of a scalar function at x, scipy's <c>derivative</c>.
    /// </summary>
    /// <param name="f">Function to differentiate.</param>
    /// <param name="x">Abscissa.</param>
    /// <param name="atol">Absolute tolerance; default the smallest normal double (as scipy).</param>
    /// <param name="rtol">Relative tolerance; default √eps (as scipy).</param>
    /// <param name="maxIter">Maximum iterations (default 10).</param>
    /// <param name="order">Accuracy order of the stencil (default 8); odd orders are rounded up to even
    /// for central differences.</param>
    /// <param name="initialStep">First step size (default 0.5).</param>
    /// <param name="stepFactor">Factor the step shrinks by per iteration (default 2).</param>
    /// <param name="stepDirection">0 central; +1 forward (only <c>x ≥ x0</c> is evaluated); −1 backward —
    /// for use at a domain boundary.</param>
    public static DerivativeResult Derivative(
        Func<double, double> f, double x, double? atol = null, double? rtol = null, int maxIter = 10,
        int order = 8, double initialStep = 0.5, double stepFactor = 2.0, int stepDirection = 0)
    {
        ArgumentNullException.ThrowIfNull(f);
        var r = Core(t => [f(t)], x, 1, atol ?? DefaultAtol, rtol ?? Math.Sqrt(LinalgKernels.Eps),
            maxIter, order, initialStep, stepFactor, stepDirection);
        return new DerivativeResult(r.Df[0], r.Error[0], r.Status[0] == 0, r.Status[0], r.Nit[0], r.Nfev, x);
    }

    /// <summary>
    /// Jacobian of a vector function, scipy's <c>jacobian</c>: <c>J[i, j] = ∂f_i/∂x_j</c>, each column
    /// by an adaptive <see cref="Derivative"/> along coordinate j.
    /// </summary>
    public static JacobianResult Jacobian(
        Func<double[], double[]> f, double[] x, double? atol = null, double? rtol = null, int maxIter = 10,
        int order = 8, double initialStep = 0.5, double stepFactor = 2.0, int stepDirection = 0)
    {
        ArgumentNullException.ThrowIfNull(f);
        ArgumentNullException.ThrowIfNull(x);
        var n = x.Length;
        var m = f((double[])x.Clone()).Length;
        var df = new double[m * n];
        var err = new double[m * n];
        var status = new int[m * n];
        var nit = 0;
        var nfev = 1;
        for (var j = 0; j < n; j++)
        {
            var jj = j;
            var r = Core(t =>
            {
                var xp = (double[])x.Clone();
                xp[jj] = t;
                var y = f(xp);
                if (y.Length != m) throw new ArgumentException("Jacobian: f returned vectors of different lengths.");
                return y;
            }, x[j], m, atol ?? DefaultAtol, rtol ?? Math.Sqrt(LinalgKernels.Eps), maxIter, order, initialStep, stepFactor, stepDirection);
            for (var i = 0; i < m; i++)
            {
                df[i * n + j] = r.Df[i];
                err[i * n + j] = r.Error[i];
                status[i * n + j] = r.Status[i];
                nit = Math.Max(nit, r.Nit[i]);
            }
            nfev += r.Nfev;
        }
        return new JacobianResult(new NdArray(df, m, n), new NdArray(err, m, n), status.All(s => s == 0), status, nit, nfev);
    }

    /// <summary>Gradient of a scalar function of a vector (a 1×n Jacobian returned as a length-n vector).</summary>
    public static JacobianResult Gradient(
        Func<double[], double> f, double[] x, double? atol = null, double? rtol = null, int maxIter = 10,
        int order = 8, double initialStep = 0.5, double stepFactor = 2.0, int stepDirection = 0)
    {
        ArgumentNullException.ThrowIfNull(f);
        var j = Jacobian(v => [f(v)], x, atol, rtol, maxIter, order, initialStep, stepFactor, stepDirection);
        var n = x.Length;
        return j with { Df = j.Df.Reshape(n), Error = j.Error.Reshape(n) };
    }

    /// <summary>
    /// Hessian of a scalar function, scipy's <c>hessian</c>: the Jacobian of the gradient, with the
    /// inner gradients run at <c>rtol/100</c> so their error does not dominate the outer differences.
    /// </summary>
    /// <remarks>
    /// Nested adaptive differencing costs on the order of (evaluations per derivative)² · n², and
    /// the result is not exactly symmetric; it is returned as computed, as scipy does.
    /// </remarks>
    public static JacobianResult Hessian(
        Func<double[], double> f, double[] x, double? atol = null, double? rtol = null, int maxIter = 10,
        int order = 8, double initialStep = 0.5, double stepFactor = 2.0)
    {
        ArgumentNullException.ThrowIfNull(f);
        var outerRtol = rtol ?? Math.Sqrt(LinalgKernels.Eps);
        outerRtol = Math.Max(outerRtol, 100 * LinalgKernels.Eps);
        var innerNfev = 0;
        var res = Jacobian(v =>
        {
            var g = Gradient(f, v, atol, outerRtol / 100.0, maxIter, order, initialStep, stepFactor);
            innerNfev += g.Nfev;
            return g.Df.ToArray();
        }, x, atol, outerRtol, maxIter, order, initialStep, stepFactor);
        return res with { Nfev = innerNfev };
    }

    // ================================================================ engine

    private sealed record CoreResult(double[] Df, double[] Error, int[] Status, int[] Nit, int Nfev);

    private static CoreResult Core(Func<double, double[]> f, double x, int m, double atol, double rtol,
        int maxIter, int order, double h0, double c, int direction)
    {
        if (order < 1) throw new ArgumentException("Derivative: order must be a positive integer.");
        if (!(h0 > 0)) throw new ArgumentException("Derivative: initialStep must be positive.");
        if (!(c > 1)) throw new ArgumentException("Derivative: stepFactor must exceed 1.");
        if (maxIter < 1) throw new ArgumentException("Derivative: maxIter must be at least 1.");
        if (direction is < -1 or > 1) throw new ArgumentException("Derivative: stepDirection must be -1, 0 or 1.");
        if (atol < 0 || rtol < 0) throw new ArgumentException("Derivative: tolerances must be non-negative.");

        var central = direction == 0;
        // Relative nodes t_k = c^-k; central uses pairs (+-t_k), one-sided t_k plus the centre.
        var count = central ? (order + 1) / 2 : order;
        var t = new double[count];
        for (var k = 0; k < count; k++) t[k] = Math.Pow(c, -k);
        double[] weights;          // central: weight of (f(+t)-f(-t)); one-sided: weight of f(dir*t), centre last
        if (central)
        {
            var nodes = new double[2 * count];
            for (var k = 0; k < count; k++) { nodes[2 * k] = t[k]; nodes[2 * k + 1] = -t[k]; }
            var w = FirstDerivativeWeights(nodes);
            weights = new double[count];
            for (var k = 0; k < count; k++) weights[k] = w[2 * k]; // antisymmetric: w(-t) = -w(t)
        }
        else
        {
            var nodes = new double[count + 1];
            for (var k = 0; k < count; k++) nodes[k] = direction * t[k];
            nodes[count] = 0.0;
            weights = FirstDerivativeWeights(nodes);
        }

        // Cache of evaluations by level j: offset h0 * c^-j (both signs for central).
        var plus = new Dictionary<int, double[]>();
        var minus = new Dictionary<int, double[]>();
        double[]? centre = null;
        var nfev = 0;
        double[] Eval(double at) { nfev++; var y = f(at); if (y.Length != m) throw new ArgumentException("Derivative: f changed its output length."); return y; }

        var df = new double[m];
        var err = new double[m];
        var status = new int[m];
        var nit = new int[m];
        var stopped = new bool[m];
        var prevDf = new double[m];
        var prevErr = new double[m];
        Array.Fill(prevDf, double.NaN);
        Array.Fill(prevErr, double.PositiveInfinity);
        Array.Fill(err, double.NaN);
        Array.Fill(status, -2);

        for (var i = 0; i < maxIter; i++)
        {
            var h = h0 * Math.Pow(c, -i);
            var est = new double[m];
            for (var k = 0; k < count; k++)
            {
                var level = i + k;
                var off = h0 * Math.Pow(c, -level);
                if (!plus.TryGetValue(level, out var fp))
                {
                    fp = Eval(x + (central ? off : direction * off));
                    plus[level] = fp;
                }
                if (central)
                {
                    if (!minus.TryGetValue(level, out var fm)) { fm = Eval(x - off); minus[level] = fm; }
                    for (var q = 0; q < m; q++) est[q] += weights[k] * (fp[q] - fm[q]);
                }
                else
                {
                    for (var q = 0; q < m; q++) est[q] += weights[k] * fp[q];
                }
            }
            if (!central)
            {
                centre ??= Eval(x);
                for (var q = 0; q < m; q++) est[q] += weights[count] * centre[q];
            }
            for (var q = 0; q < m; q++) est[q] /= h;

            var allStopped = true;
            for (var q = 0; q < m; q++)
            {
                if (stopped[q]) continue;
                nit[q] = i + 1;
                var e = i == 0 ? double.NaN : Math.Abs(est[q] - prevDf[q]);
                if (!double.IsFinite(est[q]))
                {
                    df[q] = double.NaN; err[q] = double.NaN; status[q] = -3; stopped[q] = true;
                    continue;
                }
                if (i > 0 && e < atol + rtol * Math.Abs(est[q]))
                {
                    df[q] = est[q]; err[q] = e; status[q] = 0; stopped[q] = true;
                    continue;
                }
                if (i > 1 && e > 10 * prevErr[q])
                {
                    // Round-off has taken over: the previous estimate is the better one.
                    df[q] = prevDf[q]; err[q] = prevErr[q]; status[q] = -1; stopped[q] = true;
                    continue;
                }
                df[q] = est[q]; err[q] = e;
                prevDf[q] = est[q];
                prevErr[q] = i == 0 ? double.PositiveInfinity : e;
                allStopped = false;
            }
            if (allStopped) break;
        }
        return new CoreResult(df, err, status, nit, nfev);
    }

    /// <summary>
    /// Finite-difference weights for the first derivative at 0 from arbitrary distinct offsets
    /// (Fornberg 1988): <c>f'(x) ≈ Σ w_i f(x + h·offsets[i]) / h</c>.
    /// </summary>
    /// <remarks>Fornberg's recurrence, not a Vandermonde solve, which loses digits on geometric spacings.</remarks>
    public static double[] FirstDerivativeWeights(double[] x)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Length < 2) throw new ArgumentException("FirstDerivativeWeights: need at least two nodes.");
        const int mOrder = 1;
        var n = x.Length;
        var cmat = new double[n, mOrder + 1];
        var c1 = 1.0;
        var c4 = x[0];
        cmat[0, 0] = 1.0;
        for (var i = 1; i < n; i++)
        {
            var mn = Math.Min(i, mOrder);
            var c2 = 1.0;
            var c5 = c4;
            c4 = x[i];
            for (var j = 0; j < i; j++)
            {
                var c3 = x[i] - x[j];
                c2 *= c3;
                if (j == i - 1)
                {
                    for (var k = mn; k >= 1; k--)
                        cmat[i, k] = c1 * (k * cmat[i - 1, k - 1] - c5 * cmat[i - 1, k]) / c2;
                    cmat[i, 0] = -c1 * c5 * cmat[i - 1, 0] / c2;
                }
                for (var k = mn; k >= 1; k--)
                    cmat[j, k] = (c4 * cmat[j, k] - k * cmat[j, k - 1]) / c3;
                cmat[j, 0] = c4 * cmat[j, 0] / c3;
            }
            c1 = c2;
        }
        var w = new double[n];
        for (var i = 0; i < n; i++) w[i] = cmat[i, 1];
        return w;
    }
}
