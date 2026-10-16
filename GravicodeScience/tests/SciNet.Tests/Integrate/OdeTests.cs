using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class OdeTests
{
    private static void Decay(double t, ReadOnlySpan<double> y, Span<double> d)
    {
        for (var i = 0; i < y.Length; i++) d[i] = -0.5 * y[i];
    }

    private static void Robertson(double t, ReadOnlySpan<double> y, Span<double> d)
    {
        d[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
        d[1] = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1];
        d[2] = 3e7 * y[1] * y[1];
    }

    private static void VanDerPol1000(double t, ReadOnlySpan<double> y, Span<double> d)
    {
        d[0] = y[1];
        d[1] = 1000 * (1 - y[0] * y[0]) * y[1] - y[0];
    }

    // scipy 1.17.1 solve_ivp(lambda t, y: -0.5*y, [0, 10], [2, 4, 8], method=m):
    // (len(t), nfev, njev, nlu, t[1], y[0,-1], y[2,-1])
    [Theory]
    [InlineData(OdeMethod.Rk23, 16, 47, 0, 0, 0.02714681459685608, 0.013278896040337239, 0.053115584161348955)]
    [InlineData(OdeMethod.Rk45, 8, 44, 0, 0, 0.11487653462140478, 0.013507806569964347, 0.05403122627985739)]
    [InlineData(OdeMethod.Dop853, 5, 50, 0, 0, 0.25860956205565166, 0.013476952636505508, 0.053907810546022034)]
    [InlineData(OdeMethod.Radau, 11, 78, 2, 8, 0.06687890558661595, 0.013476538849906293, 0.05390615539962517)]
    [InlineData(OdeMethod.Bdf, 27, 54, 1, 8, 0.004472788012463491, 0.013440114444804389, 0.053760457779217556)]
    public void ExponentialDecay_ReproducesScipyStepSequence(OdeMethod m, int nt, int nfev, int njev, int nlu, double t1, double y0End, double y2End)
    {
        var r = Integrate.SolveIvp(Decay, 0, 10, [2, 4, 8], m);
        Assert.True(r.Success, r.Message);
        Assert.Equal(nt, r.T.Length);
        Assert.Equal(nfev, r.Nfev);
        Assert.Equal(njev, r.Njev);
        Assert.Equal(nlu, r.Nlu);
        IntegrateAssert.Rel(t1, r.T[1], 1e-12, "t[1]");
        IntegrateAssert.Rel(y0End, r.Y[0, nt - 1], 1e-10, "y0 end");
        IntegrateAssert.Rel(y2End, r.Y[2, nt - 1], 1e-10, "y2 end");
        IntegrateAssert.Rel(2 * Math.Exp(-5), r.Y[0, nt - 1], 0.03, "vs exact");
    }

    [Theory]
    [InlineData(OdeMethod.Rk23, 3)]
    [InlineData(OdeMethod.Rk45, 5)]
    [InlineData(OdeMethod.Dop853, 8)]
    [InlineData(OdeMethod.Radau, 5)]
    public void FixedStep_OrderOfConvergence(OdeMethod m, int order)
    {
        // Integrate y' = y cos t (y = exp(sin t)) with constant steps (huge tolerance, first_step = max_step = h)
        // and check the error ratio when h halves is 2^order.
        OdeFunction f = (t, y, d) => d[0] = y[0] * Math.Cos(t);
        double Err(double h)
        {
            var s = Integrate.CreateOdeSolver(m, f, 0, [1.0], 2.0, maxStep: h, rtol: 1e3, atol: 1e3, firstStep: h);
            while (s.Status == OdeSolverStatus.Running) s.Step();
            return Math.Abs(s.Y[0] - Math.Exp(Math.Sin(2.0)));
        }
        var h0 = m == OdeMethod.Dop853 ? 0.25 : 0.1;
        var e1 = Err(h0);
        var e2 = Err(h0 / 2);
        var observed = Math.Log2(e1 / e2);
        Assert.True(Math.Abs(observed - order) < 0.35, $"{m}: observed order {observed:F3}, expected {order}");
    }

    [Fact]
    public void Bdf_ConvergesAtExpectedRate()
    {
        // Variable order: just require the error to fall faster than first order with tolerance.
        OdeFunction f = (t, y, d) => d[0] = y[0] * Math.Cos(t);
        double Err(double tol)
        {
            var r = Integrate.SolveIvp(f, 0, 2, [1.0], OdeMethod.Bdf, rtol: tol, atol: tol * 1e-3);
            return Math.Abs(r.Y[0, r.T.Length - 1] - Math.Exp(Math.Sin(2.0)));
        }
        Assert.True(Err(1e-8) < 1e-6, "BDF accuracy at rtol 1e-8");
        Assert.True(Err(1e-8) < Err(1e-4) / 100, "BDF error must shrink with tolerance");
    }

    [Theory]
    [InlineData(OdeMethod.Radau, 0.7158270702916719, 9.18553482426642e-06, 0.28416374417350393)]
    [InlineData(OdeMethod.Bdf, 0.7158275895905584, 9.185554920103688e-06, 0.28416322485452156)]
    [InlineData(OdeMethod.Lsoda, 0.7158269127821871, 9.185528702026845e-06, 0.2841639016891104)]
    public void Robertson_MatchesScipy(OdeMethod m, double a, double b, double c)
    {
        // scipy 1.17.1, rtol 1e-6, atol 1e-10, t in [0, 40]. LSODA here is our reimplementation, so it is
        // held to the solution accuracy rather than to ODEPACK's digits.
        var r = Integrate.SolveIvp(Robertson, 0, 40, [1, 0, 0], m, rtol: 1e-6, atol: 1e-10);
        Assert.True(r.Success, r.Message);
        var k = r.T.Length - 1;
        var tol = m == OdeMethod.Lsoda ? 2e-6 : 1e-9;
        IntegrateAssert.Rel(a, r.Y[0, k], tol, "y0");
        IntegrateAssert.Rel(b, r.Y[1, k], m == OdeMethod.Lsoda ? 1e-5 : 1e-8, "y1");
        IntegrateAssert.Rel(c, r.Y[2, k], tol * 3, "y2");
        IntegrateAssert.Close(1.0, r.Y[0, k] + r.Y[1, k] + r.Y[2, k], 1e-9, "mass");
    }

    [Theory]
    [InlineData(OdeMethod.Radau, -1.5105962316351569, 2865, 1e-3)]
    [InlineData(OdeMethod.Bdf, -1.4899038286092663, 1888, 3e-2)]
    public void VanDerPolStiff_MatchesScipy(OdeMethod m, double y0End, int nfev, double methodError)
    {
        // The true y0(3000) is -1.5106069367598518 (scipy Radau, rtol = atol = 1e-12). At the default
        // rtol 1e-3, BDF lands 1.4% away from it - in scipy and here alike - so that, not scipy's
        // digits, is what the solver can promise. The end value is also sensitive to the last bit of
        // exp/log: Linux's libm moved the BDF answer by 2.3e-3 against Windows. So it is pinned to the
        // truth within the method's own error, and to scipy only within what libm differences allow.
        const double truth = -1.5106069367598518;
        var r = Integrate.SolveIvp(VanDerPol1000, 0, 3000, [2, 0], m);
        Assert.True(r.Success, r.Message);
        var end = r.Y[0, r.T.Length - 1];
        IntegrateAssert.Close(truth, end, methodError, "y0(3000) against the true value");
        IntegrateAssert.Close(y0End, end, 5e-3, "y0(3000) against scipy");
        Assert.True(Math.Abs(r.Nfev - nfev) <= nfev / 20, $"nfev {r.Nfev} vs scipy {nfev}");
    }

    [Fact]
    public void Lsoda_SwitchesToStiffOnVanDerPol()
    {
        var s = new LsodaSolver(VanDerPol1000, 0, [2, 0], 3000);
        while (s.Status == OdeSolverStatus.Running) s.Step();
        Assert.Equal(OdeSolverStatus.Finished, s.Status);
        Assert.NotEmpty(s.SwitchTimes);
        // Exact behaviour differs from ODEPACK; the solution sits near scipy's ~ -1.50 at t = 3000.
        Assert.True(Math.Abs(s.Y[0] + 1.50) < 0.05, $"y0 {s.Y[0]}");
        Assert.True(s.Nfev < 6000, $"nfev {s.Nfev} - the stiff phase must be used");
    }

    [Fact]
    public void TEval_And_Backward()
    {
        var r = Integrate.SolveIvp(Decay, 0, 10, [2, 4, 8], tEval: [1, 2, 5, 10]);
        double[] e = [1.2130536941836096, 0.7353402132367963, 0.1643330880146928, 0.013507806569964333];
        for (var i = 0; i < 4; i++) IntegrateAssert.Rel(e[i], r.Y[0, i], 1e-12, $"t_eval[{i}]");
        var b = Integrate.SolveIvp(Decay, 10, 0, [2, 4, 8]);
        Assert.Equal(7, b.T.Length);
        Assert.Equal(38, b.Nfev);
        IntegrateAssert.Rel(296.80050293445504, b.Y[0, b.T.Length - 1], 1e-12, "backward");
        foreach (var m in new[] { OdeMethod.Radau, OdeMethod.Bdf, OdeMethod.Dop853, OdeMethod.Lsoda })
        {
            var bb = Integrate.SolveIvp(Decay, 10, 0, [1.0], m, rtol: 1e-8, atol: 1e-10, tEval: [10, 7, 3, 0]);
            Assert.Equal(4, bb.T.Length);
            IntegrateAssert.Rel(Math.Exp(5), bb.Y[0, 3], 1e-6, $"{m} backward");
        }
    }

    [Fact]
    public void BouncingBall_Events_MatchScipy()
    {
        var hit = new OdeEvent((t, y) => y[0], terminal: 1, direction: -1);
        var apex = new OdeEvent((t, y) => y[1]);
        var r = Integrate.SolveIvp((t, y, d) => { d[0] = y[1]; d[1] = -9.8; }, 0, 100, [0, 10], events: [hit, apex], denseOutput: true);
        Assert.Equal(1, r.Status);
        Assert.Single(r.TEvents![0]);
        IntegrateAssert.Close(2.0408163265306105, r.TEvents[0][0], 1e-13, "landing");
        IntegrateAssert.Close(1.020408163265306, r.TEvents[1][0], 1e-13, "apex time");
        IntegrateAssert.Close(5.1020408163265385, r.YEvents![1][0][0], 1e-12, "apex height");
        Assert.Equal(7, r.T.Length);
        Assert.Equal(38, r.Nfev);
        IntegrateAssert.Close(r.TEvents[0][0], r.T[^1], 0, "ends at the event");
    }

    [Theory]
    [InlineData(OdeMethod.Rk23)]
    [InlineData(OdeMethod.Rk45)]
    [InlineData(OdeMethod.Dop853)]
    [InlineData(OdeMethod.Radau)]
    [InlineData(OdeMethod.Bdf)]
    [InlineData(OdeMethod.Lsoda)]
    public void DenseOutput_IsContinuousAndAccurate(OdeMethod m)
    {
        // Harmonic oscillator: the dense solution must match cos/sin between steps, be continuous
        // at the step boundaries, and conserve energy to tolerance.
        var r = Integrate.SolveIvp((t, y, d) => { d[0] = y[1]; d[1] = -y[0]; }, 0, 10, [1, 0], m, denseOutput: true, rtol: 1e-9, atol: 1e-12);
        var sol = r.Sol!;
        var tol = m is OdeMethod.Rk23 or OdeMethod.Bdf or OdeMethod.Lsoda ? 1e-5 : 1e-7;
        for (var k = 0; k <= 200; k++)
        {
            var t = 10.0 * k / 200;
            var y = sol.Evaluate(t);
            IntegrateAssert.Close(Math.Cos(t), y[0], tol, $"{m} cos at {t}");
            IntegrateAssert.Close(1.0, y[0] * y[0] + y[1] * y[1], 10 * tol, $"{m} energy at {t}");
        }
        for (var s = 1; s < sol.Ts.Count - 1; s++)
        {
            var tb = sol.Ts[s];
            var eps = 1e-9 * Math.Max(1, Math.Abs(tb));
            var a = sol.Evaluate(tb - eps);
            var b = sol.Evaluate(tb + eps);
            IntegrateAssert.Close(a[0], b[0], 1e-6, $"{m} continuity at {tb}");
        }
    }

    [Fact]
    public void Odeint_ExponentialAndAllocatingOverload()
    {
        double[] t = [0, 0.5, 1, 2, 4];
        var y = Integrate.Odeint((yy, tt) => [-2 * yy[0], yy[0]], [1.0, 0.0], t);
        Assert.Equal(5, y.Rows);
        Assert.Equal(2, y.Columns);
        for (var i = 0; i < t.Length; i++)
        {
            IntegrateAssert.Close(Math.Exp(-2 * t[i]), y[i, 0], 1e-7, $"odeint y0 at {t[i]}");
            IntegrateAssert.Close((1 - Math.Exp(-2 * t[i])) / 2, y[i, 1], 1e-7, $"odeint y1 at {t[i]}");
        }
        var stiff = Integrate.OdeintFull(Robertson, [1, 0, 0], [0, 1, 10, 40], out var info, rtol: 1e-6, atol: 1e-10);
        Assert.True(info.Success, info.Message);
        IntegrateAssert.Rel(0.7158270702916719, stiff[3, 0], 1e-5, "odeint robertson");
        Assert.Contains(2, info.Mused);
    }

    [Fact]
    public void StepwiseSolver_And_PerComponentAtol()
    {
        var s = new Rk45Solver(Decay, 0, [1.0, 1.0], 1.0, atolVector: [1e-8, 1e-3], rtol: 1e-6);
        var steps = 0;
        while (s.Status == OdeSolverStatus.Running) { s.Step(); steps++; }
        Assert.Equal(1.0, s.T);
        IntegrateAssert.Close(Math.Exp(-0.5), s.Y[0], 1e-6, "y");
        var d = s.DenseOutput();
        IntegrateAssert.Close(s.Y[0], d.Evaluate(1.0)[0], 1e-15, "dense at end");
        Assert.Throws<InvalidOperationException>(() => s.Step());
        Assert.True(steps > 1);
    }

    [Fact]
    public void AnalyticJacobian_IsUsed()
    {
        OdeJacobian jac = (t, y, j) =>
        {
            j[0] = -0.04; j[1] = 1e4 * y[2]; j[2] = 1e4 * y[1];
            j[3] = 0.04; j[4] = -1e4 * y[2] - 6e7 * y[1]; j[5] = -1e4 * y[1];
            j[6] = 0; j[7] = 6e7 * y[1]; j[8] = 0;
        };
        foreach (var m in new[] { OdeMethod.Radau, OdeMethod.Bdf })
        {
            var r = Integrate.SolveIvp(Robertson, 0, 40, [1, 0, 0], m, rtol: 1e-6, atol: 1e-10, jac: jac);
            IntegrateAssert.Rel(0.71582707, r.Y[0, r.T.Length - 1], 2e-6, $"{m} analytic jac");
            var sp = new bool[3, 3] { { true, true, true }, { true, true, true }, { false, true, false } };
            var r2 = Integrate.SolveIvp(Robertson, 0, 40, [1, 0, 0], m, rtol: 1e-6, atol: 1e-10, jacSparsity: sp);
            IntegrateAssert.Rel(0.71582707, r2.Y[0, r2.T.Length - 1], 2e-6, $"{m} sparse fd jac");
        }
    }
}
