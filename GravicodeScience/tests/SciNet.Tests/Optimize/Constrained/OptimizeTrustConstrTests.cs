using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class OptimizeTrustConstrTests
{
    private static readonly double Inf = double.PositiveInfinity;

    // The scipy.optimize tutorial's trust-constr example: Rosenbrock with bounds, a two-row linear
    // constraint and two nonlinear inequalities.
    private static Bounds TutorialBounds => new([0, -0.5], [1.0, 2.0]);
    private static LinearConstraint TutorialLinear => new(new NdArray([1.0, 2, 2, 1], 2, 2), [-Inf, 1], [1, 1]);
    private static double[] TutorialConsF(double[] x) => [x[0] * x[0] + x[1], x[0] * x[0] - x[1]];
    private static NdArray TutorialConsJ(double[] x) => new(new[] { 2 * x[0], 1, 2 * x[0], -1 }, 2, 2);
    private static NdArray TutorialConsH(double[] x, double[] v) => new(new[] { 2 * v[0] + 2 * v[1], 0, 0, 0 }, 2, 2);

    // scipy 1.17.1: x = [0.4149453134150196, 0.17010937316996083], fun = 0.3427175756422305, nit 12.
    private static readonly double[] TutorialX = [0.4149453134150196, 0.17010937316996083];
    private const double TutorialF = 0.3427175756422305;

    [Fact]
    public void Tutorial_ExactHessians()
    {
        var nc = new NonlinearConstraint(TutorialConsF, [-Inf, -Inf], [1, 1], TutorialConsJ) { Hess = TutorialConsH };
        var r = Optimize.Minimize(Optimize.Rosen, [0.5, 0], MinimizeMethod.TrustConstr, jac: Optimize.RosenDer, hess: Optimize.RosenHess,
            bounds: TutorialBounds, constraints: [TutorialLinear, nc]);
        Assert.True(r.Success, r.ToString());
        Assert.Equal(1, r.Status);
        Assert.Equal("tr_interior_point", r.Method);
        OptimizeAssert.Close(TutorialX, r.X, 1e-7, "x");
        OptimizeAssert.Close(TutorialF, r.Fun, 1e-9, "fun");
        Assert.True(r.ConstrViolation <= 1e-8, $"violation {r.ConstrViolation}");
        // The linear constraint's equality row is active: scipy reports v = [0.000654, 0.41281].
        OptimizeAssert.Close(0.41280921898317696, r.V![0][1], 1e-4, "v_lin[1]");
        OptimizeAssert.Close(0, r.Optimality!.Value, 1e-7, "optimality");
    }

    [Fact]
    public void Tutorial_QuasiNewtonAndFiniteDifferences()
    {
        var nc = new NonlinearConstraint(TutorialConsF, [-Inf, -Inf], [1, 1]);
        var r = Optimize.Minimize(Optimize.Rosen, [0.5, 0], MinimizeMethod.TrustConstr,
            bounds: TutorialBounds, constraints: [TutorialLinear, nc]);
        Assert.True(r.Success, r.ToString());
        // scipy 1.17.1 (jac='2-point', hess=BFGS()): [0.41494531470752777, 0.17010937058494444]
        OptimizeAssert.Close(TutorialX, r.X, 1e-6, "x");
        OptimizeAssert.Close(TutorialF, r.Fun, 1e-8, "fun");
    }

    [Fact]
    public void Tutorial_HessianVectorProduct()
    {
        var nc = new NonlinearConstraint(TutorialConsF, [-Inf, -Inf], [1, 1], TutorialConsJ) { Hess = TutorialConsH };
        var r = Optimize.Minimize(Optimize.Rosen, [0.5, 0], MinimizeMethod.TrustConstr, jac: Optimize.RosenDer, hessp: Optimize.RosenHessProd,
            bounds: TutorialBounds, constraints: [TutorialLinear, nc]);
        Assert.True(r.Success, r.ToString());
        OptimizeAssert.Close(TutorialX, r.X, 1e-7, "x");
    }

    [Fact]
    public void Tutorial_Sr1Objective()
    {
        var nc = new NonlinearConstraint(TutorialConsF, [-Inf, -Inf], [1, 1], TutorialConsJ) { Hess = TutorialConsH };
        var r = Optimize.Minimize(Optimize.Rosen, [0.5, 0], MinimizeMethod.TrustConstr, jac: Optimize.RosenDer,
            bounds: TutorialBounds, constraints: [TutorialLinear, nc],
            options: new TrustConstrOptions { HessianUpdate = new Sr1HessianUpdate() });
        Assert.True(r.Success, r.ToString());
        // scipy 1.17.1 (hess=SR1()): [0.4149453135006441, 0.17010937299871187]
        OptimizeAssert.Close(TutorialX, r.X, 1e-7, "x");
    }

    [Fact]
    public void HockSchittkowski71()
    {
        // HS71: min x0 x3 (x0+x1+x2) + x2, x0x1x2x3 ≥ 25, Σx² = 40, 1 ≤ x ≤ 5. Published f* = 17.0140173.
        var c1 = new NonlinearConstraint(x => [x[0] * x[1] * x[2] * x[3]], [25], [Inf]);
        var c2 = new NonlinearConstraint(x => [x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]], [40], [40]);
        var r = Optimize.Minimize(x => x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2], [1, 5, 5, 1], MinimizeMethod.TrustConstr,
            bounds: Bounds.Uniform(4, 1, 5), constraints: [c1, c2]);
        Assert.True(r.Success, r.ToString());
        // scipy 1.17.1: [1.0000000377, 4.7429996305, 3.8211500019, 1.3794082399], fun 17.014017362557382
        OptimizeAssert.Close([1.0, 4.742999630508693, 3.8211500019158056, 1.3794082399401983], r.X, 2e-5, "x");
        OptimizeAssert.Close(17.014017362557382, r.Fun, 1e-6, "fun");
        Assert.True(r.ConstrViolation <= 1e-7, $"violation {r.ConstrViolation}");
    }

    [Fact]
    public void HockSchittkowski35_QuadraticWithLinearConstraint()
    {
        // HS35: f* = 1/9 at (4/3, 7/9, 4/9).
        var r = Optimize.Minimize(x => 9 - 8 * x[0] - 6 * x[1] - 4 * x[2] + 2 * x[0] * x[0] + 2 * x[1] * x[1] + x[2] * x[2] + 2 * x[0] * x[1] + 2 * x[0] * x[2],
            [0.5, 0.5, 0.5], MinimizeMethod.TrustConstr, bounds: Bounds.Uniform(3, 0, Inf),
            constraints: [new LinearConstraint(new NdArray([1.0, 1, 2], 1, 3), [-Inf], [3])]);
        Assert.True(r.Success, r.ToString());
        OptimizeAssert.Close([4.0 / 3, 7.0 / 9, 4.0 / 9], r.X, 2e-6, "x");
        OptimizeAssert.Close(1.0 / 9, r.Fun, 1e-7, "fun");
        // The Lagrange multiplier of x0 + x1 + 2x2 ≤ 3 is 2/9 (KKT by hand).
        OptimizeAssert.Close(2.0 / 9, r.V![0][0], 1e-5, "v");
    }

    [Fact]
    public void EqualityOnly_LinearUsesSqpBranch()
    {
        var r = Optimize.Minimize(x => x[0] * x[0] + x[1] * x[1], [3, -1], MinimizeMethod.TrustConstr,
            constraints: [new LinearConstraint(new NdArray([1.0, 1], 1, 2), [1], [1])]);
        Assert.True(r.Success, r.ToString());
        Assert.Equal("equality_constrained_sqp", r.Method);
        OptimizeAssert.Close([0.5, 0.5], r.X, 1e-8, "x");
        // KKT: 2x + v = 0 → v = −1 (scipy: -1.000000014901161).
        OptimizeAssert.Close(-1, r.V![0][0], 1e-6, "v");
    }

    [Fact]
    public void EqualityOnly_NonlinearHs6()
    {
        // HS6: min (1 − x0)² s.t. 10(x1 − x0²) = 0 → (1, 1).
        var r = Optimize.Minimize(x => (1 - x[0]) * (1 - x[0]), [-1.2, 1], MinimizeMethod.TrustConstr,
            constraints: [new NonlinearConstraint(x => [10 * (x[1] - x[0] * x[0])], [0], [0])]);
        Assert.True(r.Success, r.ToString());
        OptimizeAssert.Close([1, 1], r.X, 1e-6, "x");
        Assert.True(r.Fun < 1e-12, $"fun {r.Fun}");
    }

    [Fact]
    public void EqualityOnly_Circle()
    {
        // min x0 + x1 on x0² + x1² = 2 → (−1, −1), multiplier ½.
        var r = Optimize.Minimize(x => x[0] + x[1], [0.5, -1.5], MinimizeMethod.TrustConstr,
            constraints: [new NonlinearConstraint(x => x[0] * x[0] + x[1] * x[1], 2, 2, x => [2 * x[0], 2 * x[1]])]);
        Assert.True(r.Success, r.ToString());
        OptimizeAssert.Close([-1, -1], r.X, 1e-7, "x");
        OptimizeAssert.Close(0.5, r.V![0][0], 1e-6, "v");
    }

    [Fact]
    public void BoundsOnly_InfeasibleStartIsNotClipped()
    {
        // x0[1] = 1 violates lb = 1.5. SciPy does not clip x0 for trust-constr; the barrier slacks
        // absorb the violation, and following the same path reproduces scipy to ~1e-12. (This
        // solution stops short of the active bound by ~1e-5 because gtol is met first — so does scipy's.)
        var r = Optimize.Minimize(Optimize.Rosen, [-1.2, 1, 0.5], MinimizeMethod.TrustConstr, jac: Optimize.RosenDer, hess: Optimize.RosenHess,
            bounds: new Bounds([-Inf, 1.5, -Inf], [Inf, Inf, 0.8]));
        Assert.True(r.Success, r.ToString());
        // scipy 1.17.1: [-1.221028121326935, 1.5000045828457886, 0.7999862072382274], fun 215.44922929566587
        OptimizeAssert.Close([-1.221028121326935, 1.5000045828457886, 0.7999862072382274], r.X, 1e-8, "x");
        OptimizeAssert.Close(215.44922929566587, r.Fun, 1e-7, "fun");
        Assert.InRange(r.Nit, 9, 13); // scipy 1.17.1: nit 11 (exact here; a range because the barrier uses Math.Log)
    }

    [Fact]
    public void Unconstrained_Rosenbrock()
    {
        var r = Optimize.Minimize(Optimize.Rosen, [-1.2, 1], MinimizeMethod.TrustConstr, jac: Optimize.RosenDer, hess: Optimize.RosenHess);
        Assert.True(r.Success, r.ToString());
        OptimizeAssert.Close([1, 1], r.X, 1e-8, "x");
    }

    [Fact]
    public void LegacyDictionaryConstraints()
    {
        IMinimizeConstraint[] cons =
        [
            OptimizeConstraint.Ineq(x => x[0] - 2 * x[1] + 2),
            OptimizeConstraint.Ineq(x => -x[0] - 2 * x[1] + 6),
            OptimizeConstraint.Ineq(x => -x[0] + 2 * x[1] + 2),
        ];
        var r = Optimize.Minimize(x => (x[0] - 1) * (x[0] - 1) + (x[1] - 2.5) * (x[1] - 2.5), [2, 0], MinimizeMethod.TrustConstr,
            bounds: Bounds.FromPairs((0, null), (0, null)), constraints: cons);
        Assert.True(r.Success, r.ToString());
        // scipy 1.17.1: [1.3999999744, 1.6999999808], v[0] = -0.8
        OptimizeAssert.Close([1.4, 1.7], r.X, 1e-6, "x");
        OptimizeAssert.Close(-0.8, r.V![0][0], 1e-6, "v");
    }

    [Fact]
    public void KeepFeasibleBoundsNeverEvaluateOutside()
    {
        // x log x is NaN for x < 0: keep_feasible must hold every evaluation (including finite
        // differences) inside the box. Minimum at 1/e.
        var outside = 0;
        var r = Optimize.Minimize(x =>
        {
            if (x[0] < 0 || x[1] < 0) outside++;
            return x[0] * Math.Log(x[0]) + x[1] * Math.Log(x[1]);
        }, [0.5, 2.0], MinimizeMethod.TrustConstr, bounds: new Bounds([0, 0], [Inf, Inf], keepFeasible: true));
        Assert.True(r.Success, r.ToString());
        Assert.Equal(0, outside);
        OptimizeAssert.Close([1 / Math.E, 1 / Math.E], r.X, 1e-5, "x");
    }

    [Fact]
    public void TolMapsOntoAllThreeTolerances()
    {
        var loose = Optimize.Minimize(Optimize.Rosen, [-1.2, 1], MinimizeMethod.TrustConstr, jac: Optimize.RosenDer, hess: Optimize.RosenHess, tol: 1e-3);
        var tight = Optimize.Minimize(Optimize.Rosen, [-1.2, 1], MinimizeMethod.TrustConstr, jac: Optimize.RosenDer, hess: Optimize.RosenHess);
        Assert.True(loose.Nit < tight.Nit, $"{loose.Nit} vs {tight.Nit}");
        Assert.True(loose.Optimality < 1e-3);
    }

    [Fact]
    public void MaxIterReportsStatusZero()
    {
        var r = Optimize.Minimize(Optimize.Rosen, [-1.2, 1], MinimizeMethod.TrustConstr, jac: Optimize.RosenDer,
            options: new TrustConstrOptions { MaxIter = 5 });
        Assert.False(r.Success);
        Assert.Equal(0, r.Status);
        Assert.Equal(5, r.Nit);
    }

    [Fact]
    public void CallbackStops()
    {
        var calls = 0;
        var r = Optimize.Minimize(Optimize.Rosen, [-1.2, 1], MinimizeMethod.TrustConstr, jac: Optimize.RosenDer,
            callback: (_, _) => ++calls >= 3);
        Assert.Equal(99, r.Status);
        Assert.False(r.Success);
        Assert.Equal(3, calls);
    }

    [Fact]
    public void SharedHessianStrategyInstanceIsRefused()
    {
        var shared = new BfgsHessianUpdate();
        var nc = new NonlinearConstraint(TutorialConsF, [-Inf, -Inf], [1, 1]) { HessianUpdate = shared };
        Assert.Throws<ArgumentException>(() => Optimize.Minimize(Optimize.Rosen, [0.5, 0], MinimizeMethod.TrustConstr,
            constraints: [nc], options: new TrustConstrOptions { HessianUpdate = shared }));
    }

    [Fact]
    public void InfeasibleProblemReportsConstraintViolation()
    {
        // x0 + x1 ≥ 3 and x0 + x1 ≤ 1 cannot both hold.
        var r = Optimize.Minimize(x => x[0] * x[0] + x[1] * x[1], [0, 0], MinimizeMethod.TrustConstr,
            constraints: [new LinearConstraint(new NdArray([1.0, 1, 1, 1], 2, 2), [3, -Inf], [Inf, 1])],
            options: new TrustConstrOptions { MaxIter = 300 });
        Assert.False(r.Success);
        Assert.True(r.ConstrViolation > 0.5, r.ToString());
    }
}
