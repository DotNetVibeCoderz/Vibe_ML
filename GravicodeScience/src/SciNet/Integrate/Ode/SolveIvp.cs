using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Diagnostics of <see cref="Integrate.Odeint(OdeFunction, double[], double[], double, double, double, double, int)"/>, after scipy's odeint infodict.</summary>
public sealed class OdeintInfo
{
    /// <summary>Step size in use at each output time (hu).</summary>
    public required double[] Hu { get; init; }
    /// <summary>Solver time reached at each output time (tcur).</summary>
    public required double[] Tcur { get; init; }
    /// <summary>Cumulative steps at each output time (nst).</summary>
    public required int[] Nst { get; init; }
    /// <summary>Cumulative f evaluations at each output time (nfe).</summary>
    public required int[] Nfe { get; init; }
    /// <summary>Cumulative Jacobian evaluations at each output time (nje).</summary>
    public required int[] Nje { get; init; }
    /// <summary>Method at each output time: 1 non-stiff, 2 stiff (mused).</summary>
    public required int[] Mused { get; init; }
    /// <summary>Times of method switches (tsw holds the last one in ODEPACK; all are kept here).</summary>
    public required double[] SwitchTimes { get; init; }
    /// <summary>Outcome message.</summary>
    public string Message { get; init; } = "";
    /// <summary>True when every output time was reached.</summary>
    public bool Success { get; init; }
}

public static partial class Integrate
{
    /// <summary>
    /// scipy.integrate.solve_ivp. Integrates y' = f(t, y) from t0 to tf (tf &lt; t0 integrates
    /// backwards) with the chosen method. Output follows scipy: without <paramref name="tEval"/> every
    /// accepted step is reported; with it, values come from each step's interpolant. Events are
    /// located by Brent's method on the dense output (xtol = rtol = 4 eps as scipy).
    /// </summary>
    /// <param name="fun">Allocation-free right-hand side.</param>
    /// <param name="t0">Initial time.</param>
    /// <param name="tf">Final time.</param>
    /// <param name="y0">Initial state.</param>
    /// <param name="method">Integrator.</param>
    /// <param name="tEval">Times at which to store the solution (sorted in the direction of integration).</param>
    /// <param name="denseOutput">Build a continuous <see cref="OdeResult.Sol"/>.</param>
    /// <param name="events">Event functions.</param>
    /// <param name="firstStep">Initial step size (selected automatically by default).</param>
    /// <param name="maxStep">Maximum step size.</param>
    /// <param name="rtol">Relative tolerance.</param>
    /// <param name="atol">Absolute tolerance (scalar).</param>
    /// <param name="atolVector">Per-component absolute tolerance (overrides <paramref name="atol"/>).</param>
    /// <param name="jac">Jacobian for Radau/BDF/LSODA (finite differences otherwise).</param>
    /// <param name="jacSparsity">Sparsity pattern for the finite-difference Jacobian.</param>
    public static OdeResult SolveIvp(OdeFunction fun, double t0, double tf, double[] y0, OdeMethod method = OdeMethod.Rk45,
        double[]? tEval = null, bool denseOutput = false, IEnumerable<OdeEvent>? events = null, double? firstStep = null,
        double maxStep = double.PositiveInfinity, double rtol = 1e-3, double atol = 1e-6, double[]? atolVector = null,
        OdeJacobian? jac = null, bool[,]? jacSparsity = null)
    {
        ArgumentNullException.ThrowIfNull(fun);
        ArgumentNullException.ThrowIfNull(y0);
        if (!double.IsFinite(t0) || !double.IsFinite(tf)) throw new ArgumentException("t_span must be finite.");
        double[]? te = null;
        var teIndex = 0;
        if (tEval is not null)
        {
            var lo = Math.Min(t0, tf); var hi = Math.Max(t0, tf);
            foreach (var v in tEval) if (v < lo || v > hi) throw new ArgumentException("Values in `t_eval` are not within `t_span`.");
            for (var i = 1; i < tEval.Length; i++)
            {
                var d = tEval[i] - tEval[i - 1];
                if ((tf > t0 && d <= 0) || (tf < t0 && d >= 0)) throw new ArgumentException("Values in `t_eval` are not properly sorted.");
            }
            te = tEval;
        }
        var solver = CreateOdeSolver(method, fun, t0, y0, tf, maxStep, rtol, atol, atolVector, firstStep, jac, jacSparsity);
        var n = y0.Length;
        var ts = new List<double>();
        var ys = new List<double>();
        if (te is null) { ts.Add(t0); ys.AddRange(y0); }
        var interpolants = denseOutput ? new List<OdeDenseOutput>() : null;
        var ti = denseOutput && te is not null ? new List<double> { t0 } : null;

        OdeEvent[]? ev = events?.ToArray();
        double[]? g = null;
        int[]? eventCount = null;
        List<double>[]? tEvents = null;
        List<double[]>[]? yEvents = null;
        var scratch = new double[n];
        if (ev is { Length: > 0 })
        {
            g = new double[ev.Length];
            for (var k = 0; k < ev.Length; k++) g[k] = ev[k].Function(t0, y0);
            eventCount = new int[ev.Length];
            tEvents = ev.Select(_ => new List<double>()).ToArray();
            yEvents = ev.Select(_ => new List<double[]>()).ToArray();
        }
        else ev = null;

        int? status = null;
        string? message = null;
        var gNew = ev is null ? null : new double[ev.Length];
        var yCur = new double[n];
        while (status is null)
        {
            message = solver.Step();
            if (solver.Status == OdeSolverStatus.Finished) status = 0;
            else if (solver.Status == OdeSolverStatus.Failed) { status = -1; break; }
            var tOld = solver.TOld;
            var t = solver.T;
            solver.Y.CopyTo(yCur);
            OdeDenseOutput? sol = null;
            if (interpolants is not null)
            {
                sol = solver.DenseOutput();
                interpolants.Add(sol);
            }
            if (ev is not null)
            {
                for (var k = 0; k < ev.Length; k++) gNew![k] = ev[k].Function(t, yCur);
                var active = new List<int>();
                for (var k = 0; k < ev.Length; k++)
                {
                    var up = g![k] <= 0 && gNew![k] >= 0;
                    var down = g[k] >= 0 && gNew![k] <= 0;
                    var dir = ev[k].Direction;
                    if ((up && dir > 0) || (down && dir < 0) || ((up || down) && dir == 0)) active.Add(k);
                }
                if (active.Count > 0)
                {
                    foreach (var k in active) eventCount![k]++;
                    var roots = new double[active.Count];
                    for (var q = 0; q < active.Count; q++)
                    {
                        var e = ev[active[q]];
                        roots[q] = OdeBrentq(tt =>
                        {
                            solver.InterpolateLastStep(tt, scratch);
                            return e.Function(tt, scratch);
                        }, tOld, t, 4 * OdeSolverBase.Eps, 4 * OdeSolverBase.Eps);
                    }
                    var order = Enumerable.Range(0, active.Count).ToArray();
                    var terminate = false;
                    if (active.Any(k => ev[k].Terminal > 0 && eventCount![k] >= ev[k].Terminal))
                    {
                        Array.Sort(order, (p, q) => t > tOld ? roots[p].CompareTo(roots[q]) : roots[q].CompareTo(roots[p]));
                        var cut = Array.FindIndex(order, o => ev[active[o]].Terminal > 0 && eventCount![active[o]] >= ev[active[o]].Terminal);
                        order = order.Take(cut + 1).ToArray();
                        terminate = true;
                    }
                    foreach (var o in order)
                    {
                        var k = active[o];
                        tEvents![k].Add(roots[o]);
                        var yv = new double[n];
                        solver.InterpolateLastStep(roots[o], yv);
                        yEvents![k].Add(yv);
                    }
                    if (terminate)
                    {
                        status = 1;
                        t = roots[order[^1]];
                        solver.InterpolateLastStep(t, yCur);
                    }
                }
                (g, gNew) = (gNew, g);
            }
            if (te is null)
            {
                var doNotAppend = ts.Count > 1 && ts[^1] == t && denseOutput;
                if (!doNotAppend) { ts.Add(t); ys.AddRange(yCur); }
                else if (interpolants!.Count > 0) interpolants.RemoveAt(interpolants.Count - 1);
            }
            else
            {
                while (teIndex < te.Length && (solver.Direction > 0 ? te[teIndex] <= t : te[teIndex] >= t))
                {
                    solver.InterpolateLastStep(te[teIndex], scratch);
                    ts.Add(te[teIndex]);
                    ys.AddRange(scratch);
                    teIndex++;
                }
            }
            ti?.Add(t);
        }
        message = status switch
        {
            0 => "The solver successfully reached the end of the integration interval.",
            1 => "A termination event occurred.",
            _ => message ?? "Integration step failed.",
        };
        var m = ts.Count;
        var yData = new double[n * m];
        for (var j = 0; j < m; j++)
            for (var i = 0; i < n; i++) yData[i * m + j] = ys[j * n + i];
        OdeSolution? odeSol = null;
        if (interpolants is not null && interpolants.Count > 0)
            odeSol = new OdeSolution((ti ?? ts).ToArray(), interpolants, solver.AltSegment);
        return new OdeResult
        {
            T = ts.ToArray(),
            Y = new NdArray(yData, n, m),
            Sol = odeSol,
            TEvents = tEvents?.Select(l => l.ToArray()).ToArray(),
            YEvents = yEvents?.Select(l => l.ToArray()).ToArray(),
            Nfev = solver.Nfev,
            Njev = solver.Njev,
            Nlu = solver.Nlu,
            Status = status ?? -1,
            Message = message,
        };
    }

    /// <summary>solve_ivp with an allocating right-hand side f(t, y) → dy/dt.</summary>
    public static OdeResult SolveIvp(Func<double, double[], double[]> fun, double t0, double tf, double[] y0, OdeMethod method = OdeMethod.Rk45,
        double[]? tEval = null, bool denseOutput = false, IEnumerable<OdeEvent>? events = null, double? firstStep = null,
        double maxStep = double.PositiveInfinity, double rtol = 1e-3, double atol = 1e-6, double[]? atolVector = null,
        Func<double, double[], double[,]>? jac = null)
    {
        ArgumentNullException.ThrowIfNull(fun);
        return SolveIvp(Wrap(fun), t0, tf, y0, method, tEval, denseOutput, events, firstStep, maxStep, rtol, atol, atolVector, WrapJac(jac));
    }

    /// <summary>Adapts an allocating right-hand side to <see cref="OdeFunction"/>.</summary>
    public static OdeFunction Wrap(Func<double, double[], double[]> fun) => (t, y, dydt) =>
    {
        var r = fun(t, y.ToArray());
        if (r.Length != dydt.Length) throw new ArgumentException("The right-hand side returned a vector of the wrong length.");
        r.CopyTo(dydt);
    };

    private static OdeJacobian? WrapJac(Func<double, double[], double[,]>? jac) => jac is null ? null : (t, y, j) =>
    {
        var m = jac(t, y.ToArray());
        var n = y.Length;
        for (var i = 0; i < n; i++)
            for (var k = 0; k < n; k++) j[i * n + k] = m[i, k];
    };

    /// <summary>Constructs the step-wise solver for a method, the way solve_ivp does.</summary>
    public static OdeSolverBase CreateOdeSolver(OdeMethod method, OdeFunction fun, double t0, double[] y0, double tBound,
        double maxStep = double.PositiveInfinity, double rtol = 1e-3, double atol = 1e-6, double[]? atolVector = null, double? firstStep = null,
        OdeJacobian? jac = null, bool[,]? jacSparsity = null) => method switch
    {
        OdeMethod.Rk23 => new Rk23Solver(fun, t0, y0, tBound, maxStep, rtol, atol, atolVector, firstStep),
        OdeMethod.Rk45 => new Rk45Solver(fun, t0, y0, tBound, maxStep, rtol, atol, atolVector, firstStep),
        OdeMethod.Dop853 => new Dop853Solver(fun, t0, y0, tBound, maxStep, rtol, atol, atolVector, firstStep),
        OdeMethod.Radau => new RadauSolver(fun, t0, y0, tBound, maxStep, rtol, atol, atolVector, firstStep, jac, jacSparsity),
        OdeMethod.Bdf => new BdfSolver(fun, t0, y0, tBound, maxStep, rtol, atol, atolVector, firstStep, jac, jacSparsity),
        OdeMethod.Lsoda => new LsodaSolver(fun, t0, y0, tBound, maxStep, rtol, atol, atolVector, firstStep, jac, jacSparsity),
        _ => throw new ArgumentException($"Unknown method {method}.", nameof(method)),
    };

    /// <summary>
    /// scipy.integrate.odeint: LSODA-style integration (see <see cref="LsodaSolver"/>; a reimplementation,
    /// not ODEPACK) reporting the solution at every time in <paramref name="t"/>, which must be monotone and
    /// starts at the initial time. Returns shape (len(t), n), rows in time, as scipy. Note scipy's
    /// default is f(y, t); this overload is the t-first form (scipy's tfirst=True).
    /// </summary>
    public static NdArray Odeint(OdeFunction func, double[] y0, double[] t, double rtol = 1.49012e-8, double atol = 1.49012e-8,
        double h0 = 0.0, double hmax = 0.0, int mxstep = 500)
        => OdeintFull(func, y0, t, out _, rtol, atol, h0, hmax, mxstep);

    /// <summary>odeint with scipy's default argument order func(y, t).</summary>
    public static NdArray Odeint(Func<double[], double, double[]> func, double[] y0, double[] t, double rtol = 1.49012e-8, double atol = 1.49012e-8,
        double h0 = 0.0, double hmax = 0.0, int mxstep = 500)
    {
        ArgumentNullException.ThrowIfNull(func);
        return OdeintFull(Wrap((tt, y) => func(y, tt)), y0, t, out _, rtol, atol, h0, hmax, mxstep);
    }

    /// <summary>odeint with full_output: also returns the diagnostics.</summary>
    public static NdArray OdeintFull(OdeFunction func, double[] y0, double[] t, out OdeintInfo info, double rtol = 1.49012e-8, double atol = 1.49012e-8,
        double h0 = 0.0, double hmax = 0.0, int mxstep = 500)
    {
        ArgumentNullException.ThrowIfNull(func);
        ArgumentNullException.ThrowIfNull(y0);
        ArgumentNullException.ThrowIfNull(t);
        var n = y0.Length;
        var m = t.Length;
        if (m == 0) throw new ArgumentException("t must not be empty.");
        var dir = m > 1 ? Math.Sign(t[^1] - t[0]) : 1;
        for (var i = 1; i < m; i++)
            if (dir * (t[i] - t[i - 1]) < 0) throw new ArgumentException("t must be monotonic.");
        var outp = new double[m * n];
        Array.Copy(y0, outp, n);
        var hu = new double[m]; var tcur = new double[m];
        var nst = new int[m]; var nfe = new int[m]; var nje = new int[m]; var mused = new int[m];
        tcur[0] = t[0];
        var ok = true;
        var msg = "Integration successful.";
        LsodaSolver? s = null;
        if (m > 1 && t[^1] != t[0])
        {
            s = new LsodaSolver(func, t[0], y0, t[^1], hmax > 0 ? hmax : double.PositiveInfinity, rtol, atol, null, h0 > 0 ? h0 : null);
            var buf = new double[n];
            for (var i = 1; i < m; i++)
            {
                var steps = 0;
                while (s.Status == OdeSolverStatus.Running && dir * (s.T - t[i]) < 0)
                {
                    if (steps++ >= mxstep) { ok = false; msg = $"Excess work done on this call (more than mxstep = {mxstep} steps)."; break; }
                    var r = s.Step();
                    if (s.Status == OdeSolverStatus.Failed) { ok = false; msg = r ?? "Integration step failed."; break; }
                }
                if (!ok)
                {
                    for (var k = i; k < m; k++) for (var j = 0; j < n; j++) outp[k * n + j] = double.NaN;
                    break;
                }
                if (t[i] == s.T) s.Y.CopyTo(buf);
                else if (double.IsNaN(s.TOld)) Array.Copy(y0, buf, n);
                else s.InterpolateLastStep(t[i], buf);
                Array.Copy(buf, 0, outp, i * n, n);
                hu[i] = s.StepSize; tcur[i] = s.T; nst[i] = s.Nsteps; nfe[i] = s.Nfev; nje[i] = s.Njev; mused[i] = s.MethodUsed;
            }
        }
        else
            for (var i = 1; i < m; i++) Array.Copy(y0, 0, outp, i * n, n);
        info = new OdeintInfo
        {
            Hu = hu, Tcur = tcur, Nst = nst, Nfe = nfe, Nje = nje, Mused = mused,
            SwitchTimes = s?.SwitchTimes.ToArray() ?? [], Message = msg, Success = ok,
        };
        return new NdArray(outp, m, n);
    }

    /// <summary>scipy's brentq (the C implementation's iteration), private to event location.</summary>
    internal static double OdeBrentq(Func<double, double> f, double xa, double xb, double xtol, double rtol, int maxIter = 100)
    {
        double xpre = xa, xcur = xb, xblk = 0, fblk = 0, spre = 0, scur = 0;
        var fpre = f(xpre);
        var fcur = f(xcur);
        if (fpre == 0) return xpre;
        if (fcur == 0) return xcur;
        if (Math.Sign(fpre) == Math.Sign(fcur)) throw new InvalidOperationException("Event root is not bracketed.");
        for (var i = 0; i < maxIter; i++)
        {
            if (fpre != 0 && fcur != 0 && Math.Sign(fpre) != Math.Sign(fcur))
            {
                xblk = xpre; fblk = fpre;
                spre = scur = xcur - xpre;
            }
            if (Math.Abs(fblk) < Math.Abs(fcur))
            {
                xpre = xcur; xcur = xblk; xblk = xpre;
                fpre = fcur; fcur = fblk; fblk = fpre;
            }
            var delta = (xtol + rtol * Math.Abs(xcur)) / 2;
            var sbis = (xblk - xcur) / 2;
            if (fcur == 0 || Math.Abs(sbis) < delta) return xcur;
            if (Math.Abs(spre) > delta && Math.Abs(fcur) < Math.Abs(fpre))
            {
                double stry;
                if (xpre == xblk) stry = -fcur * (xcur - xpre) / (fcur - fpre);
                else
                {
                    var dpre = (fpre - fcur) / (xpre - xcur);
                    var dblk = (fblk - fcur) / (xblk - xcur);
                    stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
                }
                if (2 * Math.Abs(stry) < Math.Min(Math.Abs(spre), 3 * Math.Abs(sbis) - delta)) { spre = scur; scur = stry; }
                else { spre = sbis; scur = sbis; }
            }
            else { spre = sbis; scur = sbis; }
            xpre = xcur; fpre = fcur;
            xcur += Math.Abs(scur) > delta ? scur : (sbis > 0 ? delta : -delta);
            fcur = f(xcur);
        }
        return xcur;
    }
}
