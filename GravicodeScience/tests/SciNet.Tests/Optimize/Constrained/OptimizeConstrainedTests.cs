using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class OptimizeConstrainedTests
{
    public static TheoryData<MinimizeMethod> Methods => new() { MinimizeMethod.Slsqp, MinimizeMethod.Cobyla };

    // scipy tutorial: min (x0-1)² + (x1-2.5)² with three linear inequalities and x ≥ 0 → (1.4, 1.7).
    private static readonly IMinimizeConstraint[] TutorialCons =
    [
        OptimizeConstraint.Ineq(x => x[0] - 2 * x[1] + 2),
        OptimizeConstraint.Ineq(x => -x[0] - 2 * x[1] + 6),
        OptimizeConstraint.Ineq(x => -x[0] + 2 * x[1] + 2),
    ];

    [Theory]
    [MemberData(nameof(Methods))]
    public void TutorialProblem(MinimizeMethod method)
    {
        var r = Optimize.Minimize(x => (x[0] - 1) * (x[0] - 1) + (x[1] - 2.5) * (x[1] - 2.5), [2, 0], method,
            bounds: Bounds.FromPairs((0, null), (0, null)), constraints: TutorialCons);
        Assert.True(r.Success, r.ToString());
        // scipy 1.17.1: SLSQP (1.400000003383427, 1.7000000009466523), COBYLA (1.39999392, 1.69999696)
        OptimizeAssert.Close([1.4, 1.7], r.X, method == MinimizeMethod.Slsqp ? 1e-6 : 1e-4, method.ToString());
        OptimizeAssert.Close(0.8, r.Fun, method == MinimizeMethod.Slsqp ? 1e-6 : 1e-4, "fun");
        if (method == MinimizeMethod.Slsqp)
        {
            // Active constraint 0 with multiplier 0.8 (scipy reports [0.8, 0, 0]).
            OptimizeAssert.Close([0.8, 0, 0], r.Multipliers!, 1e-5, "multipliers");
        }
    }

    [Fact]
    public void Slsqp_ScipyTutorialRosenbrockWithLinearAndNonlinearConstraints()
    {
        var lc = new LinearConstraint(new NdArray([1.0, 2, 2, 1], 2, 2), [double.NegativeInfinity, 1], [1, 1]);
        var nc = new NonlinearConstraint(x => [x[0] * x[0] + x[1], x[0] * x[0] - x[1]], [double.NegativeInfinity, double.NegativeInfinity], [1, 1]);
        var r = Optimize.Minimize(Optimize.Rosen, [0.5, 0], MinimizeMethod.Slsqp, jac: Optimize.RosenDer,
            constraints: [lc, nc], bounds: new Bounds([0, -0.5], [1, 2]));
        Assert.True(r.Success, r.ToString());
        // scipy 1.17.1: x = (0.4149447491702523, 0.17011050165949543), fun 0.3427175749941983
        OptimizeAssert.Close([0.4149447491702523, 0.17011050165949543], r.X, 1e-6, "x");
        OptimizeAssert.Close(0.3427175749941983, r.Fun, 1e-7, "fun");
        var c = Optimize.Minimize(Optimize.Rosen, [0.5, 0], MinimizeMethod.Cobyla, constraints: [lc, nc], bounds: new Bounds([0, -0.5], [1, 2]),
            options: new CobylaOptions { Tol = 1e-7, MaxIter = 5000 });
        Assert.True(c.Success, c.ToString());
        OptimizeAssert.Close([0.4149447491702523, 0.17011050165949543], c.X, 1e-4, "cobyla x");
    }

    private static double Hs71(double[] x) => x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];

    [Theory]
    [MemberData(nameof(Methods))]
    public void HockSchittkowski71(MinimizeMethod method)
    {
        IMinimizeConstraint[] cons =
        [
            OptimizeConstraint.Ineq(x => x[0] * x[1] * x[2] * x[3] - 25),
            OptimizeConstraint.Eq(x => x.Sum(v => v * v) - 40),
        ];
        var r = Optimize.Minimize(Hs71, [1, 5, 5, 1], method, bounds: Bounds.Uniform(4, 1, 5), constraints: cons,
            options: method == MinimizeMethod.Cobyla ? new CobylaOptions { Tol = 1e-7, MaxIter = 5000 } : null);
        Assert.True(r.Success, r.ToString());
        OptimizeAssert.Close(17.0140173, r.Fun, method == MinimizeMethod.Slsqp ? 1e-6 : 1e-4, "f*");
        OptimizeAssert.Close([1.0, 4.742999643, 3.821149979, 1.379408293], r.X, method == MinimizeMethod.Slsqp ? 1e-4 : 2e-3, "x*");
    }

    [Theory]
    [MemberData(nameof(Methods))]
    public void HockSchittkowski35_21_28(MinimizeMethod method)
    {
        double hs35(double[] x) => 9 - 8 * x[0] - 6 * x[1] - 4 * x[2] + 2 * x[0] * x[0] + 2 * x[1] * x[1] + x[2] * x[2] + 2 * x[0] * x[1] + 2 * x[0] * x[2];
        var r35 = Optimize.Minimize(hs35, [0.5, 0.5, 0.5], method, bounds: Bounds.Uniform(3, 0, double.PositiveInfinity),
            constraints: [OptimizeConstraint.Ineq(x => 3 - x[0] - x[1] - 2 * x[2])], options: Opt(method));
        OptimizeAssert.Close(1.0 / 9, r35.Fun, 1e-6, "hs35");
        OptimizeAssert.Close([4.0 / 3, 7.0 / 9, 4.0 / 9], r35.X, 1e-3, "hs35 x");

        var r21 = Optimize.Minimize(x => 0.01 * x[0] * x[0] + x[1] * x[1] - 100, [-1, -1], method,
            bounds: new Bounds([2, -50], [50, 50]), constraints: [OptimizeConstraint.Ineq(x => 10 * x[0] - x[1] - 10)], options: Opt(method));
        OptimizeAssert.Close(-99.96, r21.Fun, 1e-6, "hs21");

        var r28 = Optimize.Minimize(x => (x[0] + x[1]) * (x[0] + x[1]) + (x[1] + x[2]) * (x[1] + x[2]), [-4, 1, 1], method,
            constraints: [new LinearConstraint(new NdArray([1.0, 2, 3], 1, 3), 1, 1)], options: Opt(method));
        OptimizeAssert.Close(0, r28.Fun, 1e-6, "hs28");
        OptimizeAssert.Close([0.5, -0.5, 0.5], r28.X, method == MinimizeMethod.Slsqp ? 1e-4 : 2e-3, "hs28 x");
    }

    private static MinimizeOptions? Opt(MinimizeMethod m) => m == MinimizeMethod.Cobyla ? new CobylaOptions { Tol = 1e-8, MaxIter = 5000 } : null;

    [Fact]
    public void Slsqp_KktHoldsAtTheSolution()
    {
        // min x0 + x1 on the unit circle: x* = -(1/√2, 1/√2), multiplier λ with ∇f = λ ∇c → λ = −1/√2 · ... checked by stationarity.
        var r = Optimize.Minimize(x => x[0] + x[1], [0.5, -0.2], MinimizeMethod.Slsqp, jac: _ => [1.0, 1.0],
            constraints: [OptimizeConstraint.Eq(x => x[0] * x[0] + x[1] * x[1] - 1, x => [2 * x[0], 2 * x[1]])]);
        Assert.True(r.Success, r.ToString());
        var s = Math.Sqrt(0.5);
        OptimizeAssert.Close([-s, -s], r.X, 2e-5, "x");
        var lam = r.Multipliers![0];
        // ∇f − λ∇c = 0
        OptimizeAssert.Close(1.0, lam * 2 * r.X[0], 1e-4, "stationarity");
    }

    [Fact]
    public void Slsqp_InfeasibleStartAndIncompatibleConstraints()
    {
        // Start violating everything; the relaxed QP still makes progress.
        var r = Optimize.Minimize(x => (x[0] - 1) * (x[0] - 1) + (x[1] - 2.5) * (x[1] - 2.5), [10, -10], MinimizeMethod.Slsqp,
            bounds: Bounds.FromPairs((0, null), (0, null)), constraints: TutorialCons);
        Assert.True(r.Success, r.ToString());
        OptimizeAssert.Close([1.4, 1.7], r.X, 1e-6, "from infeasible start");
        // x ≥ 2 and x ≤ 1 cannot both hold.
        var bad = Optimize.Minimize(x => x[0] * x[0], [0.0], MinimizeMethod.Slsqp,
            constraints: [OptimizeConstraint.Ineq(x => x[0] - 2), OptimizeConstraint.Ineq(x => 1 - x[0])]);
        Assert.False(bad.Success);
    }

    [Fact]
    public void Cobyla_ReportsConstraintViolation()
    {
        var r = Optimize.Minimize(x => x[0] * x[0] + x[1] * x[1], [1, 1], MinimizeMethod.Cobyla,
            constraints: [OptimizeConstraint.Ineq(x => x[0] + x[1] - 1)], options: new CobylaOptions { Tol = 1e-8 });
        Assert.True(r.Success, r.ToString());
        OptimizeAssert.Close([0.5, 0.5], r.X, 1e-5, "x");
        Assert.True(r.ConstrViolation <= 1e-7);
        var lim = Optimize.Minimize(x => x[0] * x[0] + x[1] * x[1], [1, 1], MinimizeMethod.Cobyla,
            constraints: [OptimizeConstraint.Ineq(x => x[0] + x[1] - 1)], options: new CobylaOptions { MaxIter = 5 });
        Assert.Equal(3, lim.Status);
        Assert.Equal(5, lim.Nfev);
    }
}
