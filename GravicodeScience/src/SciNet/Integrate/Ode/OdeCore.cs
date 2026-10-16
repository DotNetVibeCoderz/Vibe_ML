using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Allocation-free ODE right-hand side: write f(t, y) into <paramref name="dydt"/>.</summary>
public delegate void OdeFunction(double t, ReadOnlySpan<double> y, Span<double> dydt);

/// <summary>ODE Jacobian df/dy written row-major (n x n) into <paramref name="jacRowMajor"/>.</summary>
public delegate void OdeJacobian(double t, ReadOnlySpan<double> y, Span<double> jacRowMajor);

/// <summary>Integration method of Integrate.SolveIvp.</summary>
public enum OdeMethod
{
    /// <summary>Bogacki-Shampine 3(2), explicit.</summary>
    Rk23,
    /// <summary>Dormand-Prince 5(4), explicit (scipy's default).</summary>
    Rk45,
    /// <summary>Hairer's DOP853, explicit order 8.</summary>
    Dop853,
    /// <summary>Radau IIA order 5, implicit.</summary>
    Radau,
    /// <summary>Variable-order NDF/BDF 1-5, implicit.</summary>
    Bdf,
    /// <summary>Automatic stiff/non-stiff switching (a reimplementation, not ODEPACK; see <see cref="LsodaSolver"/>).</summary>
    Lsoda,
}

/// <summary>State of a step-wise solver.</summary>
public enum OdeSolverStatus
{
    /// <summary>Still integrating.</summary>
    Running,
    /// <summary>Reached t_bound.</summary>
    Finished,
    /// <summary>A step failed (step size underflow or a non-convergent implicit solve).</summary>
    Failed,
}

/// <summary>
/// An event g(t, y) for solve_ivp: a root of g is located on each step's dense output with Brent's
/// method. <see cref="Terminal"/> is scipy's terminal attribute as a count (0: never stop; k: stop at
/// the k-th occurrence); <see cref="Direction"/> restricts to rising (+1) or falling (-1) zero crossings.
/// </summary>
public sealed class OdeEvent
{
    /// <summary>Creates an event.</summary>
    public OdeEvent(Func<double, ReadOnlySpan<double>, double> function, int terminal = 0, double direction = 0)
    {
        Function = function ?? throw new ArgumentNullException(nameof(function));
        if (terminal < 0) throw new ArgumentException("terminal must be a non-negative count.", nameof(terminal));
        Terminal = terminal;
        Direction = direction;
    }

    /// <summary>The event function.</summary>
    public Func<double, ReadOnlySpan<double>, double> Function { get; }
    /// <summary>Number of occurrences after which integration stops (0 = not terminal).</summary>
    public int Terminal { get; }
    /// <summary>Crossing direction: positive, negative or 0 for both.</summary>
    public double Direction { get; }
}

/// <summary>Result of solve_ivp, with scipy's attribute names.</summary>
public sealed class OdeResult
{
    /// <summary>Time points.</summary>
    public required double[] T { get; init; }
    /// <summary>Solution values, shape (n, len(T)) like scipy.</summary>
    public required NdArray Y { get; init; }
    /// <summary>Continuous solution, when dense output was requested.</summary>
    public OdeSolution? Sol { get; init; }
    /// <summary>Event times per event, or null without events.</summary>
    public double[][]? TEvents { get; init; }
    /// <summary>Solution at each event, per event, each of length n.</summary>
    public double[][][]? YEvents { get; init; }
    /// <summary>Right-hand side evaluations (finite-difference Jacobian evaluations are not counted, as in scipy).</summary>
    public int Nfev { get; init; }
    /// <summary>Jacobian evaluations.</summary>
    public int Njev { get; init; }
    /// <summary>LU decompositions.</summary>
    public int Nlu { get; init; }
    /// <summary>-1 failed, 0 reached the end, 1 a terminal event occurred.</summary>
    public int Status { get; init; }
    /// <summary>Termination reason.</summary>
    public string Message { get; init; } = "";
    /// <summary>Status ≥ 0.</summary>
    public bool Success => Status >= 0;
}

/// <summary>Interpolant over one solver step, valid for t between <see cref="TOld"/> and <see cref="T"/>.</summary>
public abstract class OdeDenseOutput
{
    /// <summary>Creates the interpolant.</summary>
    protected OdeDenseOutput(double tOld, double t, int n)
    {
        TOld = tOld; T = t; N = n;
        TMin = Math.Min(tOld, t); TMax = Math.Max(tOld, t);
    }

    /// <summary>Start of the step.</summary>
    public double TOld { get; }
    /// <summary>End of the step.</summary>
    public double T { get; }
    /// <summary>Lower end of the step.</summary>
    public double TMin { get; }
    /// <summary>Upper end of the step.</summary>
    public double TMax { get; }
    /// <summary>System dimension.</summary>
    public int N { get; }

    /// <summary>Evaluates into a caller buffer (no allocation).</summary>
    public abstract void Evaluate(double t, Span<double> y);

    /// <summary>Evaluates at one time.</summary>
    public double[] Evaluate(double t)
    {
        var y = new double[N];
        Evaluate(t, y);
        return y;
    }
}

/// <summary>
/// scipy's OdeSolution: piecewise interpolant over all steps. Extrapolates outside the range with
/// the first/last segment. At a step boundary the earlier segment is used, except for BDF and LSODA
/// where (as in scipy's alt_segment) the later one is.
/// </summary>
public sealed class OdeSolution
{
    private readonly double[] _ts;
    private readonly double[] _tsSorted;
    private readonly List<OdeDenseOutput> _interpolants;
    private readonly bool _ascending;
    private readonly bool _right;

    internal OdeSolution(double[] ts, List<OdeDenseOutput> interpolants, bool altSegment)
    {
        if (ts.Length != interpolants.Count + 1) throw new ArgumentException("Numbers of time stamps and interpolants don't match.");
        _ts = ts;
        _interpolants = interpolants;
        _ascending = ts[^1] >= ts[0];
        _tsSorted = _ascending ? ts : ts.Reverse().ToArray();
        _right = _ascending ? altSegment : !altSegment;
        N = interpolants.Count > 0 ? interpolants[0].N : 0;
    }

    /// <summary>Step boundaries.</summary>
    public IReadOnlyList<double> Ts => _ts;
    /// <summary>Number of segments.</summary>
    public int NSegments => _interpolants.Count;
    /// <summary>Lowest time covered.</summary>
    public double TMin => _tsSorted[0];
    /// <summary>Highest time covered.</summary>
    public double TMax => _tsSorted[^1];
    /// <summary>System dimension.</summary>
    public int N { get; }

    private OdeDenseOutput Segment(double t)
    {
        // searchsorted(ts_sorted, t, side)
        int lo = 0, hi = _tsSorted.Length;
        while (lo < hi)
        {
            var mid = (lo + hi) >> 1;
            var go = _right ? _tsSorted[mid] <= t : _tsSorted[mid] < t;
            if (go) lo = mid + 1; else hi = mid;
        }
        var seg = Math.Min(Math.Max(lo - 1, 0), _interpolants.Count - 1);
        if (!_ascending) seg = _interpolants.Count - 1 - seg;
        return _interpolants[seg];
    }

    /// <summary>Evaluates into a caller buffer.</summary>
    public void Evaluate(double t, Span<double> y) => Segment(t).Evaluate(t, y);

    /// <summary>Solution at one time.</summary>
    public double[] Evaluate(double t) => Segment(t).Evaluate(t);

    /// <summary>Solution at many times, shape (n, len(ts)).</summary>
    public NdArray Evaluate(IReadOnlyList<double> ts)
    {
        var m = ts.Count;
        var data = new double[N * m];
        var tmp = new double[N];
        for (var j = 0; j < m; j++)
        {
            Evaluate(ts[j], tmp);
            for (var i = 0; i < N; i++) data[i * m + j] = tmp[i];
        }
        return new NdArray(data, N, m);
    }
}

/// <summary>
/// Base of the step-wise solvers (scipy's OdeSolver). One <see cref="Step"/> advances one accepted
/// step; <see cref="DenseOutput"/> returns an interpolant for the last step.
/// </summary>
public abstract class OdeSolverBase
{
    internal const double Eps = 2.220446049250313e-16;

    private readonly OdeFunction _fun;
    /// <summary>Current state buffer.</summary>
    protected double[] YBuf;

    /// <summary>Initialises common state and validates tolerances like scipy's validate_tol.</summary>
    protected OdeSolverBase(OdeFunction fun, double t0, double[] y0, double tBound, double rtol, double atol, double[]? atolVector)
    {
        _fun = fun ?? throw new ArgumentNullException(nameof(fun));
        ArgumentNullException.ThrowIfNull(y0);
        if (!double.IsFinite(t0)) throw new ArgumentException("t0 must be finite.");
        N = y0.Length;
        T = t0;
        TBound = tBound;
        Direction = tBound != t0 ? Math.Sign(tBound - t0) : 1;
        YBuf = (double[])y0.Clone();
        if (rtol < 100 * Eps) rtol = 100 * Eps; // scipy warns and clips
        Rtol = rtol;
        Atol = new double[N];
        if (atolVector is not null)
        {
            if (atolVector.Length != N) throw new ArgumentException("atol must be a scalar or have length n.");
            Array.Copy(atolVector, Atol, N);
        }
        else Array.Fill(Atol, atol);
        foreach (var a in Atol) if (a < 0) throw new ArgumentException("atol must be non-negative.");
    }

    /// <summary>Number of equations.</summary>
    public int N { get; }
    /// <summary>Current time.</summary>
    public double T { get; protected set; }
    /// <summary>Time before the last step (NaN before the first).</summary>
    public double TOld { get; protected set; } = double.NaN;
    /// <summary>Integration bound.</summary>
    public double TBound { get; }
    /// <summary>+1 forwards, -1 backwards.</summary>
    public int Direction { get; }
    /// <summary>Current state (no copy).</summary>
    public ReadOnlySpan<double> Y => YBuf;
    /// <summary>Solver status.</summary>
    public OdeSolverStatus Status { get; private set; } = OdeSolverStatus.Running;
    /// <summary>Relative tolerance after validation.</summary>
    public double Rtol { get; }
    /// <summary>Per-component absolute tolerance.</summary>
    public double[] Atol { get; }
    /// <summary>Counted right-hand side evaluations.</summary>
    public int Nfev { get; protected set; }
    /// <summary>Jacobian evaluations.</summary>
    public int Njev { get; protected set; }
    /// <summary>LU decompositions.</summary>
    public int Nlu { get; protected set; }
    /// <summary>Size of the last step, or NaN.</summary>
    public double StepSize => double.IsNaN(TOld) ? double.NaN : Math.Abs(T - TOld);

    /// <summary>Copy of the current state.</summary>
    public double[] GetY() => (double[])YBuf.Clone();

    /// <summary>Counted evaluation of the right-hand side.</summary>
    protected void Fun(double t, ReadOnlySpan<double> y, Span<double> dydt)
    {
        Nfev++;
        _fun(t, y, dydt);
    }

    /// <summary>Uncounted evaluation (finite-difference Jacobians, as scipy's fun_vectorized).</summary>
    protected void FunUncounted(double t, ReadOnlySpan<double> y, Span<double> dydt) => _fun(t, y, dydt);

    internal OdeFunction RawFunction => _fun;

    /// <summary>Advances one step. Returns a message on failure, null otherwise.</summary>
    public string? Step()
    {
        if (Status != OdeSolverStatus.Running) throw new InvalidOperationException("Attempt to step on a failed or finished solver.");
        if (N == 0 || T == TBound)
        {
            TOld = T;
            T = TBound;
            Status = OdeSolverStatus.Finished;
            ZeroStep = true;
            return null;
        }
        ZeroStep = false;
        var t = T;
        var ok = StepImpl(out var message);
        InterpolationReady = false;
        if (!ok)
        {
            Status = OdeSolverStatus.Failed;
            return message;
        }
        TOld = t;
        if (Direction * (T - TBound) >= 0) Status = OdeSolverStatus.Finished;
        return message;
    }

    /// <summary>True when the last <see cref="Step"/> was the degenerate zero-length one.</summary>
    protected bool ZeroStep { get; private set; }

    /// <summary>Whether interpolation data for the last step has been prepared.</summary>
    protected bool InterpolationReady { get; set; }

    /// <summary>Implementation of one step; sets T and the state.</summary>
    protected abstract bool StepImpl(out string? message);

    /// <summary>Interpolant over the last step.</summary>
    public OdeDenseOutput DenseOutput()
    {
        if (double.IsNaN(TOld)) throw new InvalidOperationException("Dense output is available after a successful step was made.");
        if (N == 0 || ZeroStep) return new OdeConstantDenseOutput(TOld, T, YBuf);
        return DenseOutputImpl();
    }

    /// <summary>Builds the interpolant snapshot.</summary>
    protected abstract OdeDenseOutput DenseOutputImpl();

    /// <summary>
    /// Evaluates the last step's interpolant without allocating a snapshot; used by solve_ivp for
    /// t_eval and event location. Defaults to the snapshot.
    /// </summary>
    internal virtual void InterpolateLastStep(double t, Span<double> y)
    {
        if (N == 0 || ZeroStep) { YBuf.AsSpan().CopyTo(y); return; }
        DenseOutputImpl().Evaluate(t, y);
    }

    /// <summary>Whether the solve_ivp OdeSolution should prefer the later segment at boundaries.</summary>
    internal virtual bool AltSegment => false;

    // ------------------------------------------------------------- shared numerics

    /// <summary>RMS norm, scipy's norm(x) = ||x|| / sqrt(n).</summary>
    internal static double RmsNorm(ReadOnlySpan<double> x)
    {
        if (x.Length == 0) return 0;
        var s = 0.0;
        foreach (var v in x) s += v * v;
        return Math.Sqrt(s / x.Length);
    }

    /// <summary>scipy's select_initial_step (Hairer, Nørsett and Wanner, II.4).</summary>
    protected double SelectInitialStep(ReadOnlySpan<double> f0, int order, double maxStep)
    {
        if (N == 0) return double.PositiveInfinity;
        var interval = Math.Abs(TBound - T);
        if (interval == 0) return 0;
        var y0 = YBuf;
        Span<double> tmp = N <= 256 ? stackalloc double[N] : new double[N];
        Span<double> y1 = N <= 256 ? stackalloc double[N] : new double[N];
        double s0 = 0, s1 = 0;
        for (var i = 0; i < N; i++)
        {
            var sc = Atol[i] + Math.Abs(y0[i]) * Rtol;
            s0 += (y0[i] / sc) * (y0[i] / sc);
            s1 += (f0[i] / sc) * (f0[i] / sc);
        }
        var d0 = Math.Sqrt(s0 / N);
        var d1 = Math.Sqrt(s1 / N);
        var h0 = d0 < 1e-5 || d1 < 1e-5 ? 1e-6 : 0.01 * d0 / d1;
        h0 = Math.Min(h0, interval);
        for (var i = 0; i < N; i++) y1[i] = y0[i] + h0 * Direction * f0[i];
        Fun(T + h0 * Direction, y1, tmp);
        double s2 = 0;
        for (var i = 0; i < N; i++)
        {
            var sc = Atol[i] + Math.Abs(y0[i]) * Rtol;
            var v = (tmp[i] - f0[i]) / sc;
            s2 += v * v;
        }
        var d2 = Math.Sqrt(s2 / N) / h0;
        var h1 = d1 <= 1e-15 && d2 <= 1e-15 ? Math.Max(1e-6, h0 * 1e-3) : Math.Pow(0.01 / Math.Max(d1, d2), 1.0 / (order + 1));
        return Math.Min(Math.Min(100 * h0, h1), Math.Min(interval, maxStep));
    }

    /// <summary>Validates max_step (positive).</summary>
    protected static double ValidateMaxStep(double maxStep)
    {
        if (!(maxStep > 0)) throw new ArgumentException("max_step must be positive.");
        return maxStep;
    }

    /// <summary>Validates first_step (positive, not beyond the bounds).</summary>
    protected double ValidateFirstStep(double firstStep)
    {
        if (!(firstStep > 0)) throw new ArgumentException("first_step must be positive.");
        if (firstStep > Math.Abs(TBound - T)) throw new ArgumentException("first_step exceeds bounds.");
        return firstStep;
    }

    /// <summary>10 ulps of t in the direction of integration: scipy's minimum step.</summary>
    protected double MinStep(double t) => 10 * Math.Abs((Direction > 0 ? Math.BitIncrement(t) : Math.BitDecrement(t)) - t);
}

/// <summary>Interpolant of a zero-length step.</summary>
internal sealed class OdeConstantDenseOutput : OdeDenseOutput
{
    private readonly double[] _y;
    internal OdeConstantDenseOutput(double tOld, double t, double[] y) : base(tOld, t, y.Length) => _y = (double[])y.Clone();
    public override void Evaluate(double t, Span<double> y) => _y.AsSpan().CopyTo(y);
}

/// <summary>
/// Finite-difference Jacobian, scipy's num_jac: one-sided differences whose step factor adapts per
/// column between calls (kept in <see cref="Factor"/>), the step directed the way f points, and a
/// retry with a 10x larger step when the difference drowns in rounding. With a sparsity pattern,
/// structurally orthogonal columns are perturbed together (Curtis-Powell-Reid grouping).
/// </summary>
internal sealed class OdeNumJac
{
    private static readonly double NumJacDiffReject = Math.Pow(OdeSolverBase.Eps, 0.875);
    private static readonly double NumJacDiffSmall = Math.Pow(OdeSolverBase.Eps, 0.75);
    private static readonly double NumJacDiffBig = Math.Pow(OdeSolverBase.Eps, 0.25);
    private static readonly double NumJacMinFactor = 1e3 * OdeSolverBase.Eps;

    private readonly int _n;
    internal double[]? Factor;
    private readonly int[][]? _groups;
    private readonly bool[,]? _sparsity;
    private readonly double[] _yp, _fnew, _fnew2, _h;

    internal OdeNumJac(int n, bool[,]? sparsity)
    {
        _n = n;
        _yp = new double[n]; _fnew = new double[n]; _fnew2 = new double[n]; _h = new double[n];
        if (sparsity is not null)
        {
            if (sparsity.GetLength(0) != n || sparsity.GetLength(1) != n) throw new ArgumentException("jac_sparsity must be n x n.");
            _sparsity = sparsity;
            _groups = GroupColumns(sparsity, n);
        }
    }

    private static int[][] GroupColumns(bool[,] s, int n)
    {
        var groups = new List<List<int>>();
        var rowsUsed = new List<bool[]>();
        for (var j = 0; j < n; j++)
        {
            var placed = false;
            for (var g = 0; g < groups.Count && !placed; g++)
            {
                var used = rowsUsed[g];
                var clash = false;
                for (var i = 0; i < n; i++) if (s[i, j] && used[i]) { clash = true; break; }
                if (clash) continue;
                groups[g].Add(j);
                for (var i = 0; i < n; i++) if (s[i, j]) used[i] = true;
                placed = true;
            }
            if (!placed)
            {
                groups.Add([j]);
                var used = new bool[n];
                for (var i = 0; i < n; i++) if (s[i, j]) used[i] = true;
                rowsUsed.Add(used);
            }
        }
        return groups.Select(g => g.ToArray()).ToArray();
    }

    /// <summary>Fills J (row-major) at (t, y) given f = f(t, y). threshold is atol per component.</summary>
    internal void Compute(OdeFunction fun, double t, ReadOnlySpan<double> y, ReadOnlySpan<double> f, double[] threshold, Span<double> jac)
    {
        var n = _n;
        if (Factor is null) { Factor = new double[n]; Array.Fill(Factor, Math.Sqrt(OdeSolverBase.Eps)); }
        var factor = Factor;
        Span<double> yScale = n <= 512 ? stackalloc double[n] : new double[n];
        for (var i = 0; i < n; i++)
        {
            var sgn = f[i] >= 0 ? 1.0 : -1.0;
            yScale[i] = sgn * Math.Max(threshold[i], Math.Abs(y[i]));
            _h[i] = (y[i] + factor[i] * yScale[i]) - y[i];
            while (_h[i] == 0)
            {
                factor[i] *= 10;
                _h[i] = (y[i] + factor[i] * yScale[i]) - y[i];
            }
        }
        if (_groups is not null) { ComputeSparse(fun, t, y, f, yScale, jac); return; }
        y.CopyTo(_yp);
        for (var j = 0; j < n; j++)
        {
            _yp[j] = y[j] + _h[j];
            fun(t, _yp, _fnew);
            _yp[j] = y[j];
            var maxInd = 0;
            var maxDiff = -1.0;
            for (var i = 0; i < n; i++)
            {
                var d = _fnew[i] - f[i];
                jac[i * n + j] = d;
                if (Math.Abs(d) > maxDiff) { maxDiff = Math.Abs(d); maxInd = i; }
            }
            var scale = Math.Max(Math.Abs(f[maxInd]), Math.Abs(_fnew[maxInd]));
            var h = _h[j];
            if (maxDiff < NumJacDiffReject * scale)
            {
                var newFactor = 10 * factor[j];
                var hNew = (y[j] + newFactor * yScale[j]) - y[j];
                _yp[j] = y[j] + hNew;
                fun(t, _yp, _fnew2);
                _yp[j] = y[j];
                var maxInd2 = 0;
                var maxDiff2 = -1.0;
                for (var i = 0; i < n; i++)
                {
                    var d = Math.Abs(_fnew2[i] - f[i]);
                    if (d > maxDiff2) { maxDiff2 = d; maxInd2 = i; }
                }
                var scale2 = Math.Max(Math.Abs(f[maxInd2]), Math.Abs(_fnew2[maxInd2]));
                if (maxDiff * scale2 < maxDiff2 * scale)
                {
                    factor[j] = newFactor;
                    h = hNew;
                    for (var i = 0; i < n; i++) jac[i * n + j] = _fnew2[i] - f[i];
                    scale = scale2;
                    maxDiff = maxDiff2;
                }
            }
            for (var i = 0; i < n; i++) jac[i * n + j] /= h;
            if (maxDiff < NumJacDiffSmall * scale) factor[j] *= 10;
            else if (maxDiff > NumJacDiffBig * scale) factor[j] *= 0.1;
            factor[j] = Math.Max(factor[j], NumJacMinFactor);
        }
    }

    private void ComputeSparse(OdeFunction fun, double t, ReadOnlySpan<double> y, ReadOnlySpan<double> f, ReadOnlySpan<double> yScale, Span<double> jac)
    {
        var n = _n;
        var s = _sparsity!;
        jac.Clear();
        var factor = Factor!;
        foreach (var g in _groups!)
        {
            y.CopyTo(_yp);
            foreach (var j in g) _yp[j] = y[j] + _h[j];
            fun(t, _yp, _fnew);
            foreach (var j in g)
            {
                double maxDiff = 0, scale = 0;
                for (var i = 0; i < n; i++)
                {
                    if (!s[i, j]) continue;
                    var d = _fnew[i] - f[i];
                    jac[i * n + j] = d / _h[j];
                    if (Math.Abs(d) >= maxDiff)
                    {
                        maxDiff = Math.Abs(d);
                        scale = Math.Max(Math.Abs(f[i]), Math.Abs(_fnew[i]));
                    }
                }
                if (maxDiff < NumJacDiffSmall * scale) factor[j] *= 10;
                else if (maxDiff > NumJacDiffBig * scale) factor[j] *= 0.1;
                factor[j] = Math.Max(factor[j], NumJacMinFactor);
            }
        }
    }
}
