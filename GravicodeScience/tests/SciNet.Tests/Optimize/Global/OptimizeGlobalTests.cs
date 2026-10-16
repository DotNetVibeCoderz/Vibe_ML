using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class OptimizeGlobalTests
{
    private static double Rastrigin(double[] x) => 10 * x.Length + x.Sum(v => v * v - 10 * Math.Cos(2 * Math.PI * v));

    private static double Ackley(double[] x)
    {
        var n = x.Length;
        var s1 = x.Sum(v => v * v) / n;
        var s2 = x.Sum(v => Math.Cos(2 * Math.PI * v)) / n;
        return -20 * Math.Exp(-0.2 * Math.Sqrt(s1)) - Math.Exp(s2) + 20 + Math.E;
    }

    private static double Eggholder(double[] x) =>
        -(x[1] + 47) * Math.Sin(Math.Sqrt(Math.Abs(x[0] / 2 + (x[1] + 47)))) - x[0] * Math.Sin(Math.Sqrt(Math.Abs(x[0] - (x[1] + 47))));

    private static double Himmelblau(double[] x) => Math.Pow(x[0] * x[0] + x[1] - 11, 2) + Math.Pow(x[0] + x[1] * x[1] - 7, 2);

    [Fact]
    public void DifferentialEvolution_FindsGlobalMinima()
    {
        var r = Optimize.DifferentialEvolution(Rastrigin, Bounds.Uniform(3, -5.12, 5.12), seed: 1);
        Assert.True(r.Success, r.ToString());
        Assert.True(r.Fun < 1e-8, r.ToString());
        var a = Optimize.DifferentialEvolution(Ackley, Bounds.Uniform(2, -5, 5), seed: 2);
        Assert.True(a.Fun < 1e-8, a.ToString());
        // Eggholder is deceptive: with the default tol = 0.01 many seeds stop in a local basin (SciPy's too).
        var e = Optimize.DifferentialEvolution(Eggholder, Bounds.Uniform(2, -512, 512), seed: 8);
        OptimizeAssert.Close(-959.6406627, e.Fun, 1e-4, "eggholder");
        OptimizeAssert.Close([512, 404.2319], e.X, 1e-3, "eggholder x");
        Assert.Equal(e.Population!.Shape[0], e.PopulationEnergies!.Length);
    }

    [Theory]
    [InlineData(DifferentialEvolutionStrategy.Rand1Bin, DifferentialEvolutionInit.Sobol)]
    [InlineData(DifferentialEvolutionStrategy.Best2Exp, DifferentialEvolutionInit.Halton)]
    [InlineData(DifferentialEvolutionStrategy.CurrentToBest1Bin, DifferentialEvolutionInit.Random)]
    [InlineData(DifferentialEvolutionStrategy.RandToBest1Exp, DifferentialEvolutionInit.LatinHypercube)]
    public void DifferentialEvolution_StrategiesAndInits(DifferentialEvolutionStrategy strategy, DifferentialEvolutionInit init)
    {
        var r = Optimize.DifferentialEvolution(Himmelblau, Bounds.Uniform(2, -5, 5), strategy: strategy, init: init, seed: 4, tol: 1e-10);
        Assert.True(r.Fun < 1e-10, $"{strategy}/{init}: {r}");
    }

    [Fact]
    public void DifferentialEvolution_DeferredWorkersMatchSerial()
    {
        var s = Optimize.DifferentialEvolution(Rastrigin, Bounds.Uniform(2, -5.12, 5.12), seed: 5, updating: DifferentialEvolutionUpdating.Deferred, polish: false);
        var p = Optimize.DifferentialEvolution(Rastrigin, Bounds.Uniform(2, -5.12, 5.12), seed: 5, workers: 4, polish: false);
        Assert.Equal(s.X, p.X);
        Assert.Equal(s.Nfev, p.Nfev);
    }

    [Fact]
    public void DifferentialEvolution_ConstraintsAndIntegrality()
    {
        // min x0 + x1 with x0² + x1² ≥ 1 in [0, 2]²: optimum on the circle at (1, 0)/(0, 1), value 1.
        var r = Optimize.DifferentialEvolution(x => x[0] + x[1], Bounds.Uniform(2, 0, 2), seed: 6,
            constraints: [new NonlinearConstraint(x => x[0] * x[0] + x[1] * x[1], 1, double.PositiveInfinity)]);
        OptimizeAssert.Close(1, r.Fun, 1e-5, "constrained");
        var i = Optimize.DifferentialEvolution(x => Math.Pow(x[0] - 2.4, 2) + Math.Pow(x[1] + 1.6, 2), Bounds.Uniform(2, -5, 5), seed: 7,
            integrality: [true, false]);
        Assert.Equal(2.0, i.X[0]);
        OptimizeAssert.Close(-1.6, i.X[1], 1e-4, "continuous part");
    }

    [Fact]
    public void DualAnnealing_FindsGlobalMinima()
    {
        var r = Optimize.DualAnnealing(Rastrigin, Bounds.Uniform(4, -5.12, 5.12), seed: 42, maxiter: 300);
        Assert.True(r.Fun < 1e-8, r.ToString());
        var e = Optimize.DualAnnealing(Eggholder, Bounds.Uniform(2, -512, 512), seed: 1, maxiter: 300);
        OptimizeAssert.Close(-959.6406627, e.Fun, 1e-4, "eggholder");
        var n = Optimize.DualAnnealing(Himmelblau, Bounds.Uniform(2, -5, 5), seed: 3, noLocalSearch: true, maxiter: 500);
        Assert.True(n.Fun < 1e-2, n.ToString());
    }

    [Fact]
    public void BasinHopping_FindsGlobalMinimum()
    {
        // A 1-D double well with a shallower local minimum near x = 1.1 and the global one near −1.3.
        double F(double[] x) => Math.Cos(14.5 * x[0] - 0.3) + (x[0] + 0.2) * x[0];
        var r = Optimize.BasinHopping(F, [1.0], niter: 200, seed: 1);
        // scipy docs: x ≈ -0.1951, f ≈ -1.0009
        OptimizeAssert.Close(-0.195, r.X[0], 1e-3, "x");
        OptimizeAssert.Close(-1.0009, r.Fun, 1e-4, "fun");
        var a = Optimize.BasinHopping(Rastrigin, [2.5, -2.1], niter: 200, seed: 2);
        Assert.True(a.Fun < 1e-6, a.ToString());
        Assert.NotNull(a.LowestOptimizationResult);
    }

    [Fact]
    public void Direct_FindsMinimaDeterministically()
    {
        // scipy 1.17.1 direct on Himmelblau over [-4, 4]²: (3, 2), f ≈ 2.6e-11, len_tol termination.
        var r = Optimize.Direct(Himmelblau, Bounds.Uniform(2, -4, 4));
        Assert.True(r.Success, r.ToString());
        Assert.True(r.Fun < 1e-6, r.ToString());
        var r2 = Optimize.Direct(Himmelblau, Bounds.Uniform(2, -4, 4));
        Assert.Equal(r.X, r2.X);
        Assert.Equal(r.Nfev, r2.Nfev);
        var g = Optimize.Direct(Rastrigin, Bounds.Uniform(2, -5.12, 4.6), locallyBiased: false, maxfun: 4000);
        Assert.True(g.Fun < 1e-3, g.ToString());
        var t = Optimize.Direct(Rastrigin, Bounds.Uniform(2, -5.12, 4.6), fGlobal: 0, fGlobalRelTol: 1e-3);
        Assert.True(t.Status is 1 or 3 or 4 or 5);
    }

    [Fact]
    public void Brute_MatchesScipyGrid()
    {
        // scipy 1.17.1 brute over ((-2,2),(-2,2)), Ns=9, finish=None: x = (1.5, -0.5), f = 0.05
        var r = Optimize.Brute(x => Math.Pow(x[0] - 1.3, 2) + Math.Pow(x[1] + 0.4, 2), [(-2, 2), (-2, 2)], ns: 9, finish: false);
        OptimizeAssert.Close([1.5, -0.5], r.X0, 1e-15, "x");
        OptimizeAssert.Close(0.05, r.Fval, 1e-15, "f");
        Assert.Equal([9, 9], r.Jout.Shape.ToArray());
        var f = Optimize.Brute(x => Math.Pow(x[0] - 1.3, 2) + Math.Pow(x[1] + 0.4, 2), [(-2, 2), (-2, 2)], ns: 9, workers: 2);
        OptimizeAssert.Close([1.3, -0.4], f.X0, 1e-4, "polished");
        var s = Optimize.Brute(x => x[0] * x[0], [(-1.0, 1.0, 0.25)], finish: false);
        Assert.Equal(8, s.Jout.Size);
    }
}
