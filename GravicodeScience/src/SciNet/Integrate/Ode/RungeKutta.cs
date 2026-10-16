namespace Gravicode.Science.SciNet;

/// <summary>
/// Explicit embedded Runge-Kutta with scipy's step control: error norm in the RMS of
/// err / (atol + max(|y|, |y_new|) rtol), new step h * min(10, 0.9 err^(-1/(q+1))) after an accepted
/// step (capped at 1 straight after a rejection), at least 0.2x after a rejected one. All stage
/// vectors live in buffers allocated once, so a step allocates nothing.
/// </summary>
public abstract class RungeKuttaSolver : OdeSolverBase
{
    private const double Safety = 0.9, MinFactor = 0.2, MaxFactor = 10;

    private readonly int _stages;
    private readonly double[] _c, _a, _b, _e;
    private readonly int _aStride;
    private readonly double[] _p; // (stages+1) x (order+1), row-major, or empty
    private readonly int _pCols;
    private readonly double _errorExponent;

    /// <summary>Stage derivatives, (stages + 1) x n, row-major; the last row is f(t+h, y_new).</summary>
    protected readonly double[] K;
    /// <summary>State before the last step.</summary>
    protected double[] YOld;
    private double[] _yNew;
    private readonly double[] _ytmp;
    /// <summary>f at the current point.</summary>
    protected readonly double[] FCur;
    private readonly double[] _scale;
    private double _hAbs;
    /// <summary>Signed size of the last accepted step.</summary>
    protected double HPrevious;
    private readonly double _maxStep;
    private readonly double[] _q; // dense output coefficients, n x pCols

    private protected RungeKuttaSolver(OdeFunction fun, double t0, double[] y0, double tBound, double maxStep, double rtol, double atol,
        double[]? atolVector, double? firstStep, int stages, int errorEstimatorOrder, double[] c, double[] a, int aStride, double[] b, double[] e,
        double[] p, int pCols)
        : base(fun, t0, y0, tBound, rtol, atol, atolVector)
    {
        _stages = stages; _c = c; _a = a; _aStride = aStride; _b = b; _e = e; _p = p; _pCols = pCols;
        _maxStep = ValidateMaxStep(maxStep);
        K = new double[(stages + 1) * N];
        YOld = new double[N];
        _yNew = new double[N];
        _ytmp = new double[N];
        FCur = new double[N];
        _scale = new double[N];
        _q = new double[N * Math.Max(pCols, 1)];
        _errorExponent = -1.0 / (errorEstimatorOrder + 1);
        Fun(T, YBuf, FCur);
        _hAbs = firstStep is double fs ? ValidateFirstStep(fs) : SelectInitialStep(FCur, errorEstimatorOrder, _maxStep);
    }

    /// <summary>Current step-size proposal.</summary>
    public double HAbs => _hAbs;

    /// <summary>Input state of the last stage of the last step (y6 for Dormand-Prince).</summary>
    private protected ReadOnlySpan<double> LastStageInput => _ytmp;

    /// <summary>Error norm of the step just computed (stages in <see cref="K"/>).</summary>
    protected virtual double EstimateErrorNorm(double h, ReadOnlySpan<double> scale)
    {
        var n = N;
        var s = 0.0;
        for (var i = 0; i < n; i++)
        {
            var err = 0.0;
            for (var j = 0; j <= _stages; j++) err += K[j * n + i] * _e[j];
            var v = err * h / scale[i];
            s += v * v;
        }
        return Math.Sqrt(s / n);
    }

    /// <inheritdoc />
    protected override bool StepImpl(out string? message)
    {
        message = null;
        var t = T;
        var n = N;
        var minStep = MinStep(t);
        double hAbs = _hAbs > _maxStep ? _maxStep : _hAbs < minStep ? minStep : _hAbs;
        var rejected = false;
        while (true)
        {
            if (hAbs < minStep)
            {
                message = "Required step size is less than spacing between numbers.";
                return false;
            }
            var h = hAbs * Direction;
            var tNew = t + h;
            if (Direction * (tNew - TBound) > 0) tNew = TBound;
            h = tNew - t;
            hAbs = Math.Abs(h);

            RkStep(t, h);
            for (var i = 0; i < n; i++) _scale[i] = Atol[i] + Math.Max(Math.Abs(YBuf[i]), Math.Abs(_yNew[i])) * Rtol;
            var errorNorm = EstimateErrorNorm(h, _scale);
            if (errorNorm < 1)
            {
                var factor = errorNorm == 0 ? MaxFactor : Math.Min(MaxFactor, Safety * Math.Pow(errorNorm, _errorExponent));
                if (rejected) factor = Math.Min(1, factor);
                hAbs *= factor;
                HPrevious = h;
                (YOld, YBuf, _yNew) = (YBuf, _yNew, YOld);
                T = tNew;
                _hAbs = hAbs;
                Array.Copy(K, _stages * n, FCur, 0, n);
                return true;
            }
            hAbs *= Math.Max(MinFactor, Safety * Math.Pow(errorNorm, _errorExponent));
            rejected = true;
        }
    }

    private void RkStep(double t, double h)
    {
        var n = N;
        var y = YBuf;
        Array.Copy(FCur, 0, K, 0, n);
        for (var s = 1; s < _stages; s++)
        {
            var arow = s * _aStride;
            for (var i = 0; i < n; i++)
            {
                var dy = 0.0;
                for (var j = 0; j < s; j++) dy += K[j * n + i] * _a[arow + j];
                _ytmp[i] = y[i] + dy * h;
            }
            Fun(t + _c[s] * h, _ytmp, K.AsSpan(s * n, n));
        }
        for (var i = 0; i < n; i++)
        {
            var dy = 0.0;
            for (var j = 0; j < _stages; j++) dy += K[j * n + i] * _b[j];
            _yNew[i] = y[i] + h * dy;
        }
        Fun(t + h, _yNew, K.AsSpan(_stages * n, n));
    }

    private void PrepareQ()
    {
        if (InterpolationReady) return;
        var n = N;
        for (var i = 0; i < n; i++)
            for (var c = 0; c < _pCols; c++)
            {
                var s = 0.0;
                for (var j = 0; j <= _stages; j++) s += K[j * n + i] * _p[j * _pCols + c];
                _q[i * _pCols + c] = s;
            }
        InterpolationReady = true;
    }

    /// <inheritdoc />
    protected override OdeDenseOutput DenseOutputImpl()
    {
        PrepareQ();
        return new OdeRkDenseOutput(TOld, T, YOld, (double[])_q.Clone(), _pCols);
    }

    internal override void InterpolateLastStep(double t, Span<double> y)
    {
        if (N == 0 || ZeroStep) { YBuf.AsSpan().CopyTo(y); return; }
        PrepareQ();
        OdeRkDenseOutput.Eval(t, TOld, T - TOld, YOld, _q, _pCols, y);
    }
}

/// <summary>Polynomial interpolant y_old + h Σ_j Q[:, j] x^(j+1), x = (t - t_old)/h.</summary>
internal sealed class OdeRkDenseOutput : OdeDenseOutput
{
    private readonly double[] _yOld, _q;
    private readonly int _cols;
    private readonly double _h;

    internal OdeRkDenseOutput(double tOld, double t, double[] yOld, double[] q, int cols) : base(tOld, t, yOld.Length)
    {
        _yOld = (double[])yOld.Clone(); _q = q; _cols = cols; _h = t - tOld;
    }

    public override void Evaluate(double t, Span<double> y) => Eval(t, TOld, _h, _yOld, _q, _cols, y);

    internal static void Eval(double t, double tOld, double h, ReadOnlySpan<double> yOld, ReadOnlySpan<double> q, int cols, Span<double> y)
    {
        var x = (t - tOld) / h;
        var n = yOld.Length;
        for (var i = 0; i < n; i++)
        {
            // Horner on Σ_j q_j x^(j+1).
            var acc = 0.0;
            for (var j = cols - 1; j >= 0; j--) acc = (acc + q[i * cols + j]) * x;
            y[i] = yOld[i] + h * acc;
        }
    }
}

/// <summary>Bogacki-Shampine 3(2) with its cubic Hermite-type interpolant (scipy's RK23).</summary>
public sealed class Rk23Solver : RungeKuttaSolver
{
    private static readonly double[] C3 = [0, 0.5, 0.75];
    private static readonly double[] A3 = [0, 0, 0, 0.5, 0, 0, 0, 0.75, 0];
    private static readonly double[] B3 = [2.0 / 9, 1.0 / 3, 4.0 / 9];
    private static readonly double[] E3 = [5.0 / 72, -1.0 / 12, -1.0 / 9, 1.0 / 8];
    private static readonly double[] P3 = [1, -4.0 / 3, 5.0 / 9, 0, 1, -2.0 / 3, 0, 4.0 / 3, -8.0 / 9, 0, -1, 1];

    /// <summary>Creates the solver (scipy defaults rtol 1e-3, atol 1e-6).</summary>
    public Rk23Solver(OdeFunction fun, double t0, double[] y0, double tBound, double maxStep = double.PositiveInfinity, double rtol = 1e-3,
        double atol = 1e-6, double[]? atolVector = null, double? firstStep = null)
        : base(fun, t0, y0, tBound, maxStep, rtol, atol, atolVector, firstStep, 3, 2, C3, A3, 3, B3, E3, P3, 3) { }
}

/// <summary>Dormand-Prince 5(4) with Shampine's quartic dense output (scipy's RK45, the default).</summary>
public sealed class Rk45Solver : RungeKuttaSolver
{
    private static readonly double[] C5 = [0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1];
    private static readonly double[] A5 =
    [
        0, 0, 0, 0, 0,
        1.0 / 5, 0, 0, 0, 0,
        3.0 / 40, 9.0 / 40, 0, 0, 0,
        44.0 / 45, -56.0 / 15, 32.0 / 9, 0, 0,
        19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729, 0,
        9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656,
    ];
    private static readonly double[] B5 = [35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84];
    private static readonly double[] E5 = [-71.0 / 57600, 0, 71.0 / 16695, -71.0 / 1920, 17253.0 / 339200, -22.0 / 525, 1.0 / 40];
    private static readonly double[] P5 =
    [
        1, -8048581381.0 / 2820520608, 8663915743.0 / 2820520608, -12715105075.0 / 11282082432,
        0, 0, 0, 0,
        0, 131558114200.0 / 32700410799, -68118460800.0 / 10900136933, 87487479700.0 / 32700410799,
        0, -1754552775.0 / 470086768, 14199869525.0 / 1410260304, -10690763975.0 / 1880347072,
        0, 127303824393.0 / 49829197408, -318862633887.0 / 49829197408, 701980252875.0 / 199316789632,
        0, -282668133.0 / 205662961, 2019193451.0 / 616988883, -1453857185.0 / 822651844,
        0, 40617522.0 / 29380423, -110615467.0 / 29380423, 69997945.0 / 29380423,
    ];

    /// <summary>Creates the solver (scipy defaults rtol 1e-3, atol 1e-6).</summary>
    public Rk45Solver(OdeFunction fun, double t0, double[] y0, double tBound, double maxStep = double.PositiveInfinity, double rtol = 1e-3,
        double atol = 1e-6, double[]? atolVector = null, double? firstStep = null)
        : base(fun, t0, y0, tBound, maxStep, rtol, atol, atolVector, firstStep, 6, 4, C5, A5, 5, B5, E5, P5, 4) { }

    /// <summary>The stiffness probe the LSODA driver uses: h·|k7 - k6| / |y7 - y6| (Hairer's DOPRI5 test).</summary>
    internal double StiffnessRatio()
    {
        var n = N;
        var y6 = LastStageInput;
        double num = 0, den = 0;
        for (var i = 0; i < n; i++)
        {
            var dk = K[6 * n + i] - K[5 * n + i];
            var dy = YBuf[i] - y6[i];
            num += dk * dk;
            den += dy * dy;
        }
        return den > 0 ? Math.Abs(HPrevious) * Math.Sqrt(num / den) : 0;
    }
}

/// <summary>
/// Hairer's DOP853: order 8 with the 5th/3rd-order combined error estimate exactly as scipy and
/// dop853.f compute it, and the 7th-order dense output that needs three extra stages per step
/// (evaluated only when an interpolant is asked for, and counted in nfev).
/// </summary>
public sealed class Dop853Solver : RungeKuttaSolver
{
    private readonly double[] _kExt;
    private readonly double[] _f7; // INTERPOLATOR_POWER x n
    private readonly double[] _tmp;

    /// <summary>Creates the solver (scipy defaults rtol 1e-3, atol 1e-6).</summary>
    public Dop853Solver(OdeFunction fun, double t0, double[] y0, double tBound, double maxStep = double.PositiveInfinity, double rtol = 1e-3,
        double atol = 1e-6, double[]? atolVector = null, double? firstStep = null)
        : base(fun, t0, y0, tBound, maxStep, rtol, atol, atolVector, firstStep, IntegrateDop853Tableau.Stages, 7,
            IntegrateDop853Tableau.C, IntegrateDop853Tableau.A, 16, IntegrateDop853Tableau.B, IntegrateDop853Tableau.E3, [], 0)
    {
        _kExt = new double[3 * N];
        _f7 = new double[IntegrateDop853Tableau.InterpolatorPower * N];
        _tmp = new double[N];
    }

    /// <inheritdoc />
    protected override double EstimateErrorNorm(double h, ReadOnlySpan<double> scale)
    {
        var n = N;
        var e3 = IntegrateDop853Tableau.E3;
        var e5 = IntegrateDop853Tableau.E5;
        double n5 = 0, n3 = 0;
        for (var i = 0; i < n; i++)
        {
            double s5 = 0, s3 = 0;
            for (var j = 0; j <= 12; j++)
            {
                var k = K[j * n + i];
                s5 += k * e5[j];
                s3 += k * e3[j];
            }
            s5 /= scale[i];
            s3 /= scale[i];
            n5 += s5 * s5;
            n3 += s3 * s3;
        }
        if (n5 == 0 && n3 == 0) return 0.0;
        var denom = n5 + 0.01 * n3;
        return Math.Abs(h) * n5 / Math.Sqrt(denom * n);
    }

    private double KAt(int stage, int i) => stage <= 12 ? K[stage * N + i] : _kExt[(stage - 13) * N + i];

    private void PrepareF()
    {
        if (InterpolationReady) return;
        var n = N;
        var h = HPrevious;
        var a = IntegrateDop853Tableau.A;
        var c = IntegrateDop853Tableau.C;
        for (var s = 13; s < 16; s++)
        {
            for (var i = 0; i < n; i++)
            {
                var dy = 0.0;
                for (var j = 0; j < s; j++) dy += KAt(j, i) * a[s * 16 + j];
                _tmp[i] = YOld[i] + dy * h;
            }
            Fun(TOld + c[s] * h, _tmp, _kExt.AsSpan((s - 13) * n, n));
        }
        var d = IntegrateDop853Tableau.D;
        for (var i = 0; i < n; i++)
        {
            var fOld = K[i];
            var delta = YBuf[i] - YOld[i];
            _f7[i] = delta;
            _f7[n + i] = h * fOld - delta;
            _f7[2 * n + i] = 2 * delta - h * (FCur[i] + fOld);
            for (var r = 0; r < 4; r++)
            {
                var s = 0.0;
                for (var j = 0; j < 16; j++) s += d[r * 16 + j] * KAt(j, i);
                _f7[(3 + r) * n + i] = h * s;
            }
        }
        InterpolationReady = true;
    }

    /// <inheritdoc />
    protected override OdeDenseOutput DenseOutputImpl()
    {
        PrepareF();
        return new OdeDop853DenseOutput(TOld, T, YOld, (double[])_f7.Clone());
    }

    internal override void InterpolateLastStep(double t, Span<double> y)
    {
        if (N == 0 || ZeroStep) { YBuf.AsSpan().CopyTo(y); return; }
        PrepareF();
        OdeDop853DenseOutput.Eval(t, TOld, T - TOld, YOld, _f7, y);
    }
}

/// <summary>DOP853's alternating x / (1 - x) nested interpolant.</summary>
internal sealed class OdeDop853DenseOutput : OdeDenseOutput
{
    private readonly double[] _yOld, _f;
    private readonly double _h;

    internal OdeDop853DenseOutput(double tOld, double t, double[] yOld, double[] f) : base(tOld, t, yOld.Length)
    {
        _yOld = (double[])yOld.Clone(); _f = f; _h = t - tOld;
    }

    public override void Evaluate(double t, Span<double> y) => Eval(t, TOld, _h, _yOld, _f, y);

    internal static void Eval(double t, double tOld, double h, ReadOnlySpan<double> yOld, ReadOnlySpan<double> f, Span<double> y)
    {
        var x = (t - tOld) / h;
        var n = yOld.Length;
        const int p = IntegrateDop853Tableau.InterpolatorPower;
        for (var i = 0; i < n; i++)
        {
            var acc = 0.0;
            for (var r = 0; r < p; r++)
            {
                acc += f[(p - 1 - r) * n + i];
                acc *= r % 2 == 0 ? x : 1 - x;
            }
            y[i] = acc + yOld[i];
        }
    }
}
