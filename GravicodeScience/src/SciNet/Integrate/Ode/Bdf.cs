namespace Gravicode.Science.SciNet;

/// <summary>
/// Variable-order (1-5) BDF in Klopfenstein-Shampine NDF form, quasi-constant step size, as scipy's
/// BDF (Shampine and Reichelt, "The MATLAB ODE Suite"): the history is a backward-difference array D
/// rescaled on every step-size change, the Newton matrix I - h/α J is factored only when the step or
/// the Jacobian changes, and the order moves by at most one after order+1 steps at constant size.
/// </summary>
public sealed class BdfSolver : OdeSolverBase
{
    private const int MaxOrder = 5, NewtonMaxIter = 4;
    private const double MinFactor = 0.2, MaxFactor = 10;
    private static readonly double[] Kappa = [0, -0.1850, -1.0 / 9, -0.0823, -0.0415, 0];
    private static readonly double[] Gamma = BuildGamma();
    private static readonly double[] Alpha = Enumerable.Range(0, 6).Select(i => (1 - Kappa[i]) * Gamma[i]).ToArray();
    private static readonly double[] ErrorConst = Enumerable.Range(0, 6).Select(i => Kappa[i] * Gamma[i] + 1.0 / (i + 1)).ToArray();

    private static double[] BuildGamma()
    {
        var g = new double[6];
        for (var i = 1; i <= 5; i++) g[i] = g[i - 1] + 1.0 / i;
        return g;
    }

    private readonly double _maxStep;
    private double _hAbs;
    private readonly double _newtonTol;
    private readonly OdeJacobian? _jacUser;
    private readonly OdeNumJac? _numJac;
    private readonly double[] _j, _lu;
    private readonly int[] _piv;
    private bool _luValid;
    private readonly double[] _d; // (MaxOrder + 3) x n
    private int _order = 1;
    private int _nEqualSteps;
    private readonly double[] _yPredict, _psi, _scale, _dd, _yNew, _fb, _rhs, _tmp;
    private readonly double[] _rBuf = new double[36], _uBuf = new double[36], _ruBuf = new double[36];
    private readonly double[] _denseD;

    /// <summary>Creates the solver (scipy defaults rtol 1e-3, atol 1e-6).</summary>
    public BdfSolver(OdeFunction fun, double t0, double[] y0, double tBound, double maxStep = double.PositiveInfinity, double rtol = 1e-3,
        double atol = 1e-6, double[]? atolVector = null, double? firstStep = null, OdeJacobian? jac = null, bool[,]? jacSparsity = null)
        : base(fun, t0, y0, tBound, rtol, atol, atolVector)
    {
        var n = N;
        _maxStep = ValidateMaxStep(maxStep);
        var f = new double[n];
        Fun(T, YBuf, f);
        _hAbs = firstStep is double fs ? ValidateFirstStep(fs) : SelectInitialStep(f, 1, _maxStep);
        _newtonTol = Math.Max(10 * Eps / Rtol, Math.Min(0.03, Math.Sqrt(Rtol)));
        _jacUser = jac;
        if (jac is null) _numJac = new OdeNumJac(n, jacSparsity);
        _j = new double[n * n];
        _lu = new double[n * n];
        _piv = new int[n];
        _yPredict = new double[n]; _psi = new double[n]; _scale = new double[n]; _dd = new double[n];
        _yNew = new double[n]; _fb = new double[n]; _rhs = new double[n]; _tmp = new double[n];
        EvaluateJacobian(T, YBuf);
        _d = new double[(MaxOrder + 3) * n];
        for (var i = 0; i < n; i++)
        {
            _d[i] = YBuf[i];
            _d[n + i] = f[i] * _hAbs * Direction;
        }
        _denseD = new double[(MaxOrder + 1) * n];
    }

    /// <summary>Current order (1-5).</summary>
    public int Order => _order;

    /// <summary>The Jacobian currently in use (row-major), possibly from an earlier step.</summary>
    internal ReadOnlySpan<double> Jacobian => _j;

    private void EvaluateJacobian(double t, double[] y)
    {
        Njev++;
        if (_jacUser is not null) { _jacUser(t, y, _j); return; }
        FunUncounted(t, y, _tmp);
        _numJac!.Compute(RawFunction, t, y, _tmp, Atol, _j);
    }

    /// <summary>R(order, factor) of scipy's compute_R, into a 6x6 row-major buffer.</summary>
    private static void ComputeR(int order, double factor, double[] m)
    {
        var s = order + 1;
        Array.Clear(m);
        for (var j = 0; j < s; j++) m[j] = 1;
        for (var i = 1; i < s; i++)
            for (var j = 1; j < s; j++)
                m[i * 6 + j] = (i - 1 - factor * j) / i;
        // cumulative product down the columns
        for (var i = 1; i < s; i++)
            for (var j = 0; j < s; j++)
                m[i * 6 + j] *= m[(i - 1) * 6 + j];
    }

    private void ChangeD(int order, double factor)
    {
        ComputeR(order, factor, _rBuf);
        ComputeR(order, 1, _uBuf);
        var s = order + 1;
        // RU = R U
        for (var i = 0; i < s; i++)
            for (var j = 0; j < s; j++)
            {
                var acc = 0.0;
                for (var k = 0; k < s; k++) acc += _rBuf[i * 6 + k] * _uBuf[k * 6 + j];
                _ruBuf[i * 6 + j] = acc;
            }
        // D[:s] = RU^T D[:s]
        var n = N;
        Span<double> col = stackalloc double[6];
        for (var c = 0; c < n; c++)
        {
            for (var k = 0; k < s; k++) col[k] = _d[k * n + c];
            for (var i = 0; i < s; i++)
            {
                var acc = 0.0;
                for (var k = 0; k < s; k++) acc += _ruBuf[k * 6 + i] * col[k];
                _d[i * n + c] = acc;
            }
        }
    }

    private bool SolveBdfSystem(double tNew, double c, out int nIter)
    {
        var n = N;
        Array.Clear(_dd);
        Array.Copy(_yPredict, _yNew, n);
        double? dyNormOld = null;
        var converged = false;
        var k = 0;
        for (k = 0; k < NewtonMaxIter; k++)
        {
            Fun(tNew, _yNew, _fb);
            var finite = true;
            for (var i = 0; i < n; i++) if (!double.IsFinite(_fb[i])) { finite = false; break; }
            if (!finite) break;
            for (var i = 0; i < n; i++) _rhs[i] = c * _fb[i] - _psi[i] - _dd[i];
            IntegrateLinear.LuSolve(_lu, n, _piv, _rhs);
            var s = 0.0;
            for (var i = 0; i < n; i++) { var v = _rhs[i] / _scale[i]; s += v * v; }
            var dyNorm = Math.Sqrt(s / n);
            double? rate = dyNormOld is double old ? dyNorm / old : null;
            if (rate is double r && (r >= 1 || Math.Pow(r, NewtonMaxIter - k) / (1 - r) * dyNorm > _newtonTol)) break;
            for (var i = 0; i < n; i++) { _yNew[i] += _rhs[i]; _dd[i] += _rhs[i]; }
            if (dyNorm == 0 || (rate is double r2 && r2 / (1 - r2) * dyNorm < _newtonTol)) { converged = true; break; }
            dyNormOld = dyNorm;
        }
        nIter = Math.Min(k, NewtonMaxIter - 1) + 1;
        return converged;
    }

    /// <inheritdoc />
    protected override bool StepImpl(out string? message)
    {
        message = null;
        var n = N;
        var t = T;
        var minStep = MinStep(t);
        double hAbs;
        if (_hAbs > _maxStep) { hAbs = _maxStep; ChangeD(_order, _maxStep / _hAbs); _nEqualSteps = 0; }
        else if (_hAbs < minStep) { hAbs = minStep; ChangeD(_order, minStep / _hAbs); _nEqualSteps = 0; }
        else hAbs = _hAbs;
        var order = _order;
        var currentJac = false;
        double errorNorm;
        int nIter;
        double tNew;
        while (true)
        {
            if (hAbs < minStep)
            {
                message = "Required step size is less than spacing between numbers.";
                return false;
            }
            var h = hAbs * Direction;
            tNew = t + h;
            if (Direction * (tNew - TBound) > 0)
            {
                tNew = TBound;
                ChangeD(order, Math.Abs(tNew - t) / hAbs);
                _nEqualSteps = 0;
                _luValid = false;
            }
            h = tNew - t;
            hAbs = Math.Abs(h);
            for (var i = 0; i < n; i++)
            {
                var s = 0.0;
                for (var k = 0; k <= order; k++) s += _d[k * n + i];
                _yPredict[i] = s;
                _scale[i] = Atol[i] + Rtol * Math.Abs(s);
                var p = 0.0;
                for (var k = 1; k <= order; k++) p += _d[k * n + i] * Gamma[k];
                _psi[i] = p / Alpha[order];
            }
            var c = h / Alpha[order];
            bool converged;
            while (true)
            {
                if (!_luValid)
                {
                    for (var i = 0; i < n; i++)
                        for (var k = 0; k < n; k++)
                            _lu[i * n + k] = (i == k ? 1.0 : 0.0) - c * _j[i * n + k];
                    IntegrateLinear.LuFactor(_lu, n, _piv);
                    Nlu++;
                    _luValid = true;
                }
                converged = SolveBdfSystem(tNew, c, out nIter);
                if (converged || currentJac) break;
                EvaluateJacobian(tNew, _yPredict);
                _luValid = false;
                currentJac = true;
            }
            if (!converged)
            {
                hAbs *= 0.5;
                ChangeD(order, 0.5);
                _nEqualSteps = 0;
                _luValid = false;
                continue;
            }
            var safety = 0.9 * (2 * NewtonMaxIter + 1) / (2 * NewtonMaxIter + nIter);
            var s2 = 0.0;
            for (var i = 0; i < n; i++)
            {
                _scale[i] = Atol[i] + Rtol * Math.Abs(_yNew[i]);
                var v = ErrorConst[order] * _dd[i] / _scale[i];
                s2 += v * v;
            }
            errorNorm = Math.Sqrt(s2 / n);
            if (errorNorm > 1)
            {
                var factor = Math.Max(MinFactor, safety * Math.Pow(errorNorm, -1.0 / (order + 1)));
                hAbs *= factor;
                ChangeD(order, factor);
                _nEqualSteps = 0;
                continue;
            }
            _nEqualSteps++;
            T = tNew;
            Array.Copy(_yNew, YBuf, n);
            _hAbs = hAbs;
            for (var i = 0; i < n; i++)
            {
                _d[(order + 2) * n + i] = _dd[i] - _d[(order + 1) * n + i];
                _d[(order + 1) * n + i] = _dd[i];
            }
            for (var k = order; k >= 0; k--)
                for (var i = 0; i < n; i++) _d[k * n + i] += _d[(k + 1) * n + i];
            if (_nEqualSteps < order + 1) return true;

            var errorMNorm = double.PositiveInfinity;
            if (order > 1)
            {
                var s = 0.0;
                for (var i = 0; i < n; i++) { var v = ErrorConst[order - 1] * _d[order * n + i] / _scale[i]; s += v * v; }
                errorMNorm = Math.Sqrt(s / n);
            }
            var errorPNorm = double.PositiveInfinity;
            if (order < MaxOrder)
            {
                var s = 0.0;
                for (var i = 0; i < n; i++) { var v = ErrorConst[order + 1] * _d[(order + 2) * n + i] / _scale[i]; s += v * v; }
                errorPNorm = Math.Sqrt(s / n);
            }
            Span<double> norms = [errorMNorm, errorNorm, errorPNorm];
            var best = 0;
            var bestF = double.NegativeInfinity;
            for (var q = 0; q < 3; q++)
            {
                var fq = Math.Pow(norms[q], -1.0 / (order + q));
                if (fq > bestF) { bestF = fq; best = q; }
            }
            order += best - 1;
            _order = order;
            var fac = Math.Min(MaxFactor, safety * bestF);
            _hAbs *= fac;
            ChangeD(order, fac);
            _nEqualSteps = 0;
            _luValid = false;
            return true;
        }
    }

    private void PrepareDense()
    {
        var n = N;
        Array.Copy(_d, _denseD, (_order + 1) * n);
    }

    /// <inheritdoc />
    protected override OdeDenseOutput DenseOutputImpl()
    {
        PrepareDense();
        return new OdeBdfDenseOutput(TOld, T, _hAbs * Direction, _order, _denseD.AsSpan(0, (_order + 1) * N).ToArray(), N);
    }

    internal override void InterpolateLastStep(double t, Span<double> y)
    {
        if (N == 0 || ZeroStep) { YBuf.AsSpan().CopyTo(y); return; }
        OdeBdfDenseOutput.Eval(t, T, _hAbs * Direction, _order, _d, N, y);
    }

    internal override bool AltSegment => true;
}

/// <summary>BDF interpolant: Newton form on the backward differences, y = D0 + Σ_k D_k Π_{j&lt;k} (t - t_j)/((j+1) h).</summary>
internal sealed class OdeBdfDenseOutput : OdeDenseOutput
{
    private readonly double _h;
    private readonly int _order;
    private readonly double[] _d;

    internal OdeBdfDenseOutput(double tOld, double t, double h, int order, double[] d, int n) : base(tOld, t, n)
    {
        _h = h; _order = order; _d = d;
    }

    public override void Evaluate(double t, Span<double> y) => Eval(t, T, _h, _order, _d, N, y);

    internal static void Eval(double t, double tEnd, double h, int order, ReadOnlySpan<double> d, int n, Span<double> y)
    {
        Span<double> p = stackalloc double[6];
        var acc = 1.0;
        for (var k = 0; k < order; k++)
        {
            acc *= (t - (tEnd - h * k)) / (h * (1 + k));
            p[k] = acc;
        }
        for (var i = 0; i < n; i++)
        {
            var s = d[i];
            for (var k = 0; k < order; k++) s += d[(k + 1) * n + i] * p[k];
            y[i] = s;
        }
    }
}
