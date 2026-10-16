namespace Gravicode.Science.SciNet;

/// <summary>
/// Automatic stiff/non-stiff switching in the spirit of LSODA. This is NOT ODEPACK: scipy's LSODA
/// wraps Hindmarsh and Petzold's Fortran (Adams-Moulton orders 1-12 vs BDF 1-5 in a Nordsieck
/// history); this class is a reimplementation built from the solvers in this module, so step
/// sequences and nfev differ from scipy's while the solutions agree to tolerance. Precisely:
/// <list type="bullet">
/// <item>Non-stiff phase: Dormand-Prince RK45 (<see cref="Rk45Solver"/>). After every accepted step
/// the dominant eigenvalue is estimated for free from the last two stages, which share the abscissa
/// t+h: ρ ≈ ‖k7 − k6‖ / ‖y_new − y6‖ (Hairer's DOPRI5 stiffness test). If h·ρ &gt; 3.25 (the edge of
/// RK45's stability region) on 15 accepted steps, without 6 consecutive non-stiff steps resetting
/// the count, the problem is declared stiff.</item>
/// <item>Stiff phase: variable-order BDF (<see cref="BdfSolver"/>), started from the current point
/// with the last RK step size. After every accepted step ρ(J) is estimated by power iteration on the
/// solver's current Jacobian; if h_BDF·ρ &lt; 1.5 (comfortably inside RK45's region, so an explicit
/// method would not be stability-limited at the step BDF is taking) on 15 consecutive steps, it
/// switches back to RK45 with that step size.</item>
/// </list>
/// Each switch restarts the target method (one or two extra evaluations for its initial derivative).
/// </summary>
public sealed class LsodaSolver : OdeSolverBase
{
    private const double StiffThreshold = 3.25, NonStiffThreshold = 1.5;
    private const int StiffCountLimit = 15, NonStiffReset = 6, BackCountLimit = 15;

    private readonly double _maxStep;
    private readonly OdeJacobian? _jac;
    private readonly bool[,]? _jacSparsity;
    private OdeSolverBase _inner;
    private int _stiffCount, _nonStiffRun, _backCount;
    private int _nfevDone, _njevDone, _nluDone;
    private readonly double[] _pv, _pw;

    /// <summary>Creates the solver (scipy defaults rtol 1e-3, atol 1e-6).</summary>
    public LsodaSolver(OdeFunction fun, double t0, double[] y0, double tBound, double maxStep = double.PositiveInfinity, double rtol = 1e-3,
        double atol = 1e-6, double[]? atolVector = null, double? firstStep = null, OdeJacobian? jac = null, bool[,]? jacSparsity = null)
        : base(fun, t0, y0, tBound, rtol, atol, atolVector)
    {
        _maxStep = ValidateMaxStep(maxStep);
        _jac = jac;
        _jacSparsity = jacSparsity;
        _inner = new Rk45Solver(fun, t0, y0, tBound, maxStep, Rtol, 0, Atol, firstStep);
        _pv = new double[N];
        _pw = new double[N];
        Sync();
    }

    /// <summary>True while the stiff (BDF) method is in use.</summary>
    public bool IsStiff => _inner is BdfSolver;

    /// <summary>Method used on the last step: 1 = non-stiff (RK45), 2 = stiff (BDF), as ODEPACK's mused codes.</summary>
    public int MethodUsed => IsStiff ? 2 : 1;

    /// <summary>Times at which the method switched.</summary>
    public List<double> SwitchTimes { get; } = [];

    /// <summary>Number of accepted steps.</summary>
    public int Nsteps { get; private set; }

    private void Sync()
    {
        Nfev = _nfevDone + _inner.Nfev;
        Njev = _njevDone + _inner.Njev;
        Nlu = _nluDone + _inner.Nlu;
    }

    /// <inheritdoc />
    protected override bool StepImpl(out string? message)
    {
        message = _inner.Step();
        Sync();
        if (_inner.Status == OdeSolverStatus.Failed) return false;
        Nsteps++;
        T = _inner.T;
        _inner.Y.CopyTo(YBuf);
        if (_inner.Status == OdeSolverStatus.Finished) return true;
        if (_inner is Rk45Solver rk)
        {
            var hr = rk.StiffnessRatio();
            if (hr > StiffThreshold)
            {
                _nonStiffRun = 0;
                if (++_stiffCount >= StiffCountLimit) SwitchTo(stiff: true, rk.StepSize);
            }
            else if (++_nonStiffRun >= NonStiffReset) _stiffCount = 0;
        }
        else if (_inner is BdfSolver bdf)
        {
            var rho = SpectralRadius(bdf.Jacobian);
            if (rho * bdf.StepSize < NonStiffThreshold)
            {
                if (++_backCount >= BackCountLimit) SwitchTo(stiff: false, bdf.StepSize);
            }
            else _backCount = 0;
        }
        return true;
    }

    private void SwitchTo(bool stiff, double h)
    {
        _nfevDone += _inner.Nfev; _njevDone += _inner.Njev; _nluDone += _inner.Nlu;
        var first = Math.Min(h, Math.Abs(TBound - T));
        double? fs = first > 0 && double.IsFinite(first) ? first : null;
        _inner = stiff
            ? new BdfSolver(RawFunction, T, YBuf, TBound, _maxStep, Rtol, 0, Atol, fs, _jac, _jacSparsity)
            : new Rk45Solver(RawFunction, T, YBuf, TBound, _maxStep, Rtol, 0, Atol, fs);
        _stiffCount = 0; _nonStiffRun = 0; _backCount = 0;
        SwitchTimes.Add(T);
        Sync();
    }

    /// <summary>Power iteration (30 steps) for |λ_max| of a row-major n x n matrix.</summary>
    private double SpectralRadius(ReadOnlySpan<double> j)
    {
        var n = N;
        for (var i = 0; i < n; i++) _pv[i] = 1.0 + 0.1 * i / Math.Max(1, n);
        var lambda = 0.0;
        for (var it = 0; it < 30; it++)
        {
            double norm = 0;
            for (var i = 0; i < n; i++)
            {
                var s = 0.0;
                for (var k = 0; k < n; k++) s += j[i * n + k] * _pv[k];
                _pw[i] = s;
                norm += s * s;
            }
            norm = Math.Sqrt(norm);
            if (norm == 0) return 0;
            double vn = 0;
            for (var i = 0; i < n; i++) vn += _pv[i] * _pv[i];
            lambda = norm / Math.Sqrt(vn);
            for (var i = 0; i < n; i++) _pv[i] = _pw[i] / norm;
        }
        return lambda;
    }

    /// <inheritdoc />
    protected override OdeDenseOutput DenseOutputImpl() => _inner.DenseOutput();

    internal override void InterpolateLastStep(double t, Span<double> y)
    {
        if (N == 0 || ZeroStep) { YBuf.AsSpan().CopyTo(y); return; }
        _inner.InterpolateLastStep(t, y);
    }

    internal override bool AltSegment => true;
}
