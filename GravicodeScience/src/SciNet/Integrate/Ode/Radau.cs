using System.Numerics;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Implicit Radau IIA of order 5 (Hairer and Wanner's RADAU5 as scipy implements it). The
/// collocation system is decoupled by the eigendecomposition of the Radau matrix into one real and
/// one complex n x n system, solved by simplified Newton with LU factors kept across steps; the
/// Jacobian is re-evaluated only when Newton converges slowly, and the factors only when the step
/// size changes by more than 20%. Error estimate and step prediction (Gustafsson's two-step
/// controller) follow scipy line by line.
/// </summary>
public sealed class RadauSolver : OdeSolverBase
{
    private static readonly double S6 = Math.Sqrt(6.0);
    private static readonly double[] Cc = [(4 - S6) / 10, (4 + S6) / 10, 1];
    private static readonly double[] Ee = [(-13 - 7 * S6) / 3, (-13 + 7 * S6) / 3, -1.0 / 3];
    private static readonly double MuReal = 3 + Math.Pow(3, 2.0 / 3) - Math.Pow(3, 1.0 / 3);
    private static readonly Complex MuComplex = new(3 + 0.5 * (Math.Pow(3, 1.0 / 3) - Math.Pow(3, 2.0 / 3)), -0.5 * (Math.Pow(3, 5.0 / 6) + Math.Pow(3, 7.0 / 6)));
    private static readonly double[] Tm =
    [
        0.09443876248897524, -0.14125529502095421, 0.03002919410514742,
        0.25021312296533332, 0.20412935229379994, -0.38294211275726192,
        1, 1, 0,
    ];
    private static readonly double[] Ti =
    [
        4.17871859155190428, 0.32768282076106237, 0.52337644549944951,
        -4.17871859155190428, -0.32768282076106237, 0.47662355450055044,
        0.50287263494578682, -2.57192694985560522, 0.59603920482822492,
    ];
    private static readonly double[] Pm =
    [
        13.0 / 3 + 7 * S6 / 3, -23.0 / 3 - 22 * S6 / 3, 10.0 / 3 + 5 * S6,
        13.0 / 3 - 7 * S6 / 3, -23.0 / 3 + 22 * S6 / 3, 10.0 / 3 - 5 * S6,
        1.0 / 3, -8.0 / 3, 10.0 / 3,
    ];
    private const int NewtonMaxIter = 6;
    private const double MinFactor = 0.2, MaxFactor = 10;

    private readonly double _maxStep;
    private double _hAbs;
    private double? _hAbsOld, _errorNormOld;
    private readonly double _newtonTol;
    private readonly OdeJacobian? _jacUser;
    private readonly OdeNumJac? _numJac;
    private readonly double[] _j;
    private bool _currentJac = true;
    private readonly double[] _luReal;
    private readonly Complex[] _luComplex;
    private readonly int[] _pivReal, _pivComplex;
    private bool _luValid;
    private double[] _yOld, _yNew;
    private readonly double[] _f, _z, _w, _dw, _fstage, _tmp, _err, _scale, _zPrev, _qPrev;
    private readonly Complex[] _cbuf;
    private bool _hasSol;
    private double _solTOld, _solT;
    private readonly double[] _q; // n x 3 dense coefficients for the last step

    /// <summary>Creates the solver (scipy defaults rtol 1e-3, atol 1e-6).</summary>
    public RadauSolver(OdeFunction fun, double t0, double[] y0, double tBound, double maxStep = double.PositiveInfinity, double rtol = 1e-3,
        double atol = 1e-6, double[]? atolVector = null, double? firstStep = null, OdeJacobian? jac = null, bool[,]? jacSparsity = null)
        : base(fun, t0, y0, tBound, rtol, atol, atolVector)
    {
        var n = N;
        _maxStep = ValidateMaxStep(maxStep);
        _f = new double[n];
        Fun(T, YBuf, _f);
        _hAbs = firstStep is double fs ? ValidateFirstStep(fs) : SelectInitialStep(_f, 3, _maxStep);
        _newtonTol = Math.Max(10 * Eps / Rtol, Math.Min(0.03, Math.Sqrt(Rtol)));
        _jacUser = jac;
        if (jac is null) _numJac = new OdeNumJac(n, jacSparsity);
        _j = new double[n * n];
        EvaluateJacobian(T, YBuf, _f);
        _luReal = new double[n * n];
        _luComplex = new Complex[n * n];
        _pivReal = new int[n];
        _pivComplex = new int[n];
        _yOld = new double[n]; _yNew = new double[n];
        _z = new double[3 * n]; _w = new double[3 * n]; _dw = new double[3 * n];
        _fstage = new double[3 * n]; _tmp = new double[n]; _err = new double[n]; _scale = new double[n];
        _zPrev = new double[3 * n]; _qPrev = new double[3 * n];
        _cbuf = new Complex[n];
        _q = new double[3 * n];
    }

    private void EvaluateJacobian(double t, double[] y, double[] f)
    {
        Njev++;
        if (_jacUser is not null) _jacUser(t, y, _j);
        else _numJac!.Compute(RawFunction, t, y, f, Atol, _j);
    }

    private void FactorLu(double h)
    {
        var n = N;
        var mr = MuReal / h;
        var mc = MuComplex / h;
        for (var i = 0; i < n; i++)
            for (var k = 0; k < n; k++)
            {
                var jv = _j[i * n + k];
                _luReal[i * n + k] = (i == k ? mr : 0) - jv;
                _luComplex[i * n + k] = (i == k ? mc : Complex.Zero) - jv;
            }
        IntegrateLinear.LuFactor(_luReal, n, _pivReal);
        IntegrateLinear.LuFactor(_luComplex, n, _pivComplex);
        Nlu += 2;
        _luValid = true;
    }

    private bool SolveCollocation(double t, double h, out int nIter, out double rate)
    {
        var n = N;
        var y = YBuf;
        var mr = MuReal / h;
        var mc = MuComplex / h;
        // W = TI Z0
        for (var r = 0; r < 3; r++)
            for (var i = 0; i < n; i++)
                _w[r * n + i] = Ti[r * 3] * _z[i] + Ti[r * 3 + 1] * _z[n + i] + Ti[r * 3 + 2] * _z[2 * n + i];
        double? dwNormOld = null;
        double? rateN = null;
        var converged = false;
        var k = 0;
        for (k = 0; k < NewtonMaxIter; k++)
        {
            var finite = true;
            for (var s = 0; s < 3; s++)
            {
                for (var i = 0; i < n; i++) _tmp[i] = y[i] + _z[s * n + i];
                Fun(t + Cc[s] * h, _tmp, _fstage.AsSpan(s * n, n));
                for (var i = 0; i < n; i++) if (!double.IsFinite(_fstage[s * n + i])) finite = false;
            }
            if (!finite) break;
            for (var i = 0; i < n; i++)
            {
                double f0 = _fstage[i], f1 = _fstage[n + i], f2 = _fstage[2 * n + i];
                _dw[i] = f0 * Ti[0] + f1 * Ti[1] + f2 * Ti[2] - mr * _w[i];
                var cre = f0 * Ti[3] + f1 * Ti[4] + f2 * Ti[5];
                var cim = f0 * Ti[6] + f1 * Ti[7] + f2 * Ti[8];
                _cbuf[i] = new Complex(cre, cim) - mc * new Complex(_w[n + i], _w[2 * n + i]);
            }
            IntegrateLinear.LuSolve(_luReal, n, _pivReal, _dw.AsSpan(0, n));
            IntegrateLinear.LuSolve(_luComplex, n, _pivComplex, _cbuf);
            double s2 = 0;
            for (var i = 0; i < n; i++)
            {
                _dw[n + i] = _cbuf[i].Real;
                _dw[2 * n + i] = _cbuf[i].Imaginary;
                var sc = _scale[i];
                var a = _dw[i] / sc; var b = _dw[n + i] / sc; var c = _dw[2 * n + i] / sc;
                s2 += a * a + b * b + c * c;
            }
            var dwNorm = Math.Sqrt(s2 / (3 * n));
            if (dwNormOld is double old) rateN = dwNorm / old;
            if (rateN is double rr && (rr >= 1 || Math.Pow(rr, NewtonMaxIter - k) / (1 - rr) * dwNorm > _newtonTol)) break;
            for (var i = 0; i < 3 * n; i++) _w[i] += _dw[i];
            for (var r = 0; r < 3; r++)
                for (var i = 0; i < n; i++)
                    _z[r * n + i] = Tm[r * 3] * _w[i] + Tm[r * 3 + 1] * _w[n + i] + Tm[r * 3 + 2] * _w[2 * n + i];
            if (dwNorm == 0 || (rateN is double r2 && r2 / (1 - r2) * dwNorm < _newtonTol)) { converged = true; break; }
            dwNormOld = dwNorm;
        }
        nIter = Math.Min(k, NewtonMaxIter - 1) + 1;
        rate = rateN ?? double.NaN;
        return converged;
    }

    private static double PredictFactor(double hAbs, double? hAbsOld, double errorNorm, double? errorNormOld)
    {
        double multiplier;
        if (errorNormOld is null || hAbsOld is null || errorNorm == 0) multiplier = 1;
        else multiplier = hAbs / hAbsOld.Value * Math.Pow(errorNormOld.Value / errorNorm, 0.25);
        return Math.Min(1, multiplier) * Math.Pow(errorNorm, -0.25);
    }

    /// <inheritdoc />
    protected override bool StepImpl(out string? message)
    {
        message = null;
        var n = N;
        var t = T;
        var y = YBuf;
        var minStep = MinStep(t);
        double hAbs;
        double? hAbsOld, errorNormOld;
        if (_hAbs > _maxStep) { hAbs = _maxStep; hAbsOld = null; errorNormOld = null; }
        else if (_hAbs < minStep) { hAbs = minStep; hAbsOld = null; errorNormOld = null; }
        else { hAbs = _hAbs; hAbsOld = _hAbsOld; errorNormOld = _errorNormOld; }
        var currentJac = _currentJac;
        var rejected = false;
        double errorNorm;
        int nIter;
        double rate;
        double tNew;
        double h;
        while (true)
        {
            if (hAbs < minStep)
            {
                message = "Required step size is less than spacing between numbers.";
                return false;
            }
            h = hAbs * Direction;
            tNew = t + h;
            if (Direction * (tNew - TBound) > 0) tNew = TBound;
            h = tNew - t;
            hAbs = Math.Abs(h);

            // Z0 from the previous step's interpolant extrapolated over the new step.
            if (!_hasSol) Array.Clear(_z);
            else
            {
                for (var s = 0; s < 3; s++)
                {
                    var x = (t + h * Cc[s] - _solTOld) / (_solT - _solTOld);
                    for (var i = 0; i < n; i++)
                    {
                        var acc = 0.0;
                        for (var j = 2; j >= 0; j--) acc = (acc + _qPrev[i * 3 + j]) * x;
                        _z[s * n + i] = _yOld[i] + acc - y[i];
                    }
                }
            }
            for (var i = 0; i < n; i++) _scale[i] = Atol[i] + Math.Abs(y[i]) * Rtol;

            bool converged;
            Array.Copy(_z, _zPrev, 3 * n);
            while (true)
            {
                if (!_luValid) FactorLu(h);
                Array.Copy(_zPrev, _z, 3 * n);
                converged = SolveCollocation(t, h, out nIter, out rate);
                if (converged || currentJac) break;
                EvaluateJacobian(t, y, _f);
                currentJac = true;
                _luValid = false;
            }
            if (!converged)
            {
                hAbs *= 0.5;
                _luValid = false;
                continue;
            }
            for (var i = 0; i < n; i++) _yNew[i] = y[i] + _z[2 * n + i];
            for (var i = 0; i < n; i++)
                _err[i] = _f[i] + (_z[i] * Ee[0] + _z[n + i] * Ee[1] + _z[2 * n + i] * Ee[2]) / h;
            Array.Copy(_err, _tmp, n);
            IntegrateLinear.LuSolve(_luReal, n, _pivReal, _err);
            for (var i = 0; i < n; i++) _scale[i] = Atol[i] + Math.Max(Math.Abs(y[i]), Math.Abs(_yNew[i])) * Rtol;
            errorNorm = ScaledNorm(_err);
            var safety = 0.9 * (2 * NewtonMaxIter + 1) / (2 * NewtonMaxIter + nIter);
            if (rejected && errorNorm > 1)
            {
                // ZE = tmp - f; error = solve(LU, f(t, y + error) + ZE)
                var fy = new double[n];
                var yp = new double[n];
                for (var i = 0; i < n; i++) yp[i] = y[i] + _err[i];
                Fun(t, yp, fy);
                for (var i = 0; i < n; i++) _err[i] = fy[i] + (_tmp[i] - _f[i]);
                IntegrateLinear.LuSolve(_luReal, n, _pivReal, _err);
                errorNorm = ScaledNorm(_err);
            }
            if (errorNorm > 1)
            {
                var factor = PredictFactor(hAbs, hAbsOld, errorNorm, errorNormOld);
                hAbs *= Math.Max(MinFactor, safety * factor);
                _luValid = false;
                rejected = true;
                continue;
            }
            var recomputeJac = nIter > 2 && rate > 1e-3;
            var f2 = PredictFactor(hAbs, hAbsOld, errorNorm, errorNormOld);
            f2 = Math.Min(MaxFactor, safety * f2);
            if (!recomputeJac && f2 < 1.2) f2 = 1;
            else _luValid = false;

            (_yOld, YBuf, _yNew) = (YBuf, _yNew, _yOld);
            Fun(tNew, YBuf, _f);
            if (recomputeJac)
            {
                EvaluateJacobian(tNew, YBuf, _f);
                currentJac = true;
            }
            else currentJac = false;
            _hAbsOld = _hAbs;
            _errorNormOld = errorNorm;
            _hAbs = hAbs * f2;
            T = tNew;
            _currentJac = currentJac;
            // Dense output Q = Z^T P (n x 3), no factor h.
            for (var i = 0; i < n; i++)
                for (var c = 0; c < 3; c++)
                    _q[i * 3 + c] = _z[i] * Pm[c] + _z[n + i] * Pm[3 + c] + _z[2 * n + i] * Pm[6 + c];
            Array.Copy(_q, _qPrev, 3 * n);
            _solTOld = t;
            _solT = tNew;
            _hasSol = true;
            return true;
        }
    }

    private double ScaledNorm(ReadOnlySpan<double> v)
    {
        var s = 0.0;
        for (var i = 0; i < N; i++) { var x = v[i] / _scale[i]; s += x * x; }
        return Math.Sqrt(s / N);
    }

    /// <inheritdoc />
    protected override OdeDenseOutput DenseOutputImpl() => new OdeRadauDenseOutput(_solTOld, _solT, _yOld, (double[])_q.Clone());

    internal override void InterpolateLastStep(double t, Span<double> y)
    {
        if (N == 0 || ZeroStep) { YBuf.AsSpan().CopyTo(y); return; }
        OdeRadauDenseOutput.Eval(t, _solTOld, _solT - _solTOld, _yOld, _q, y);
    }
}

/// <summary>Radau's cubic interpolant y_old + Q [x, x², x³] (no factor h, unlike the RK forms).</summary>
internal sealed class OdeRadauDenseOutput : OdeDenseOutput
{
    private readonly double[] _yOld, _q;

    internal OdeRadauDenseOutput(double tOld, double t, double[] yOld, double[] q) : base(tOld, t, yOld.Length)
    {
        _yOld = (double[])yOld.Clone(); _q = q;
    }

    public override void Evaluate(double t, Span<double> y) => Eval(t, TOld, T - TOld, _yOld, _q, y);

    internal static void Eval(double t, double tOld, double h, ReadOnlySpan<double> yOld, ReadOnlySpan<double> q, Span<double> y)
    {
        var x = (t - tOld) / h;
        for (var i = 0; i < yOld.Length; i++)
        {
            var acc = 0.0;
            for (var j = 2; j >= 0; j--) acc = (acc + q[i * 3 + j]) * x;
            y[i] = yOld[i] + acc;
        }
    }
}
