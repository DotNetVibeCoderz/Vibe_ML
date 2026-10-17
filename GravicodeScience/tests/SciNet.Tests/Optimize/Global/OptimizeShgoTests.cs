using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class OptimizeShgoTests
{
    private static double Eggholder(double[] x)
        => -(x[1] + 47.0) * Math.Sin(Math.Sqrt(Math.Abs(x[0] / 2.0 + (x[1] + 47.0)))) - x[0] * Math.Sin(Math.Sqrt(Math.Abs(x[0] - (x[1] + 47.0))));

    private static double SinCos(double[] x) => Math.Sin(3 * x[0]) + Math.Cos(2 * x[1]) + 0.1 * (x[0] * x[0] + x[1] * x[1]);

    [Fact]
    public void Rosenbrock5d_DocExample()
    {
        var r = Optimize.Shgo(Optimize.Rosen, Bounds.Uniform(5, 0, 2));
        Assert.True(r.Success, r.Message);
        OptimizeAssert.Close([1, 1, 1, 1, 1], r.X, 1e-8, "x");
        Assert.True(r.Fun < 1e-14, $"fun {r.Fun}");
        Assert.Equal(1, r.Nit);
        // One cube: 2⁵ corners + centroid sampled, exactly as scipy 1.17.1 (nfev 50 − nlfev 17 = 33).
        Assert.Equal(33, r.Nfev - r.Nlfev);
        Assert.Single(r.Funl!);
    }

    [Fact]
    public void Eggholder_SobolFindsTheKnownMinima()
    {
        // scipy docs: shgo(eggholder, bounds, n=64, iters=3, sampling_method='sobol').
        var r = Optimize.Shgo(Eggholder, Bounds.Uniform(2, -512, 512), n: 64, iters: 3, samplingMethod: ShgoSamplingMethod.Sobol);
        Assert.True(r.Success, r.Message);
        // Global minimum of the Eggholder function: f(512, 404.2319) = −959.6407.
        OptimizeAssert.Close(-959.6406627208371, r.Fun, 1e-7, "fun");
        OptimizeAssert.Close([512, 404.23180856774957], r.X, 1e-5, "x");
        Assert.Equal(3, r.Nit);
        // The six lowest local minima scipy 1.17.1 reports are all found (Qhull and this triangulation
        // break cospherical ties differently, so the full lists differ in their tails).
        foreach (var expected in new[] { -959.6406627208371, -894.5789003905788, -821.1964610662833, -786.5259940785238, -718.1674596221036, -716.6715003419414 })
            Assert.Contains(r.Funl!, v => Math.Abs(v - expected) < 1e-6);
        AssertSortedAndConsistent(r);
    }

    [Fact]
    public void Eggholder_CompleteSimplicialRefinementMatchesScipyExactly()
    {
        // n = None refines every sub-cube completely, which does not depend on set iteration order:
        // the same 145 sampling points and the same 23 local minima as scipy 1.17.1.
        var r = Optimize.Shgo(Eggholder, Bounds.Uniform(2, -512, 512), n: null, iters: 4);
        Assert.Equal(145, r.Nfev - r.Nlfev);
        double[] scipyFunl = [-935.337952, -935.337952, -894.5789, -821.196461, -718.16746, -716.6715, -715.980114, -704.806596, -633.842302, -633.842302,
            -629.633681, -559.786857, -557.368687, -540.426295, -507.873859, -469.722928, -426.487997, -425.301506, -419.31195, -343.155591,
            -306.720468, -306.064079, -206.697406];
        OptimizeAssert.Close(scipyFunl, r.Funl!, 1e-5, "funl");
        OptimizeAssert.Close([439.481000749549, 453.9774373541639], r.X, 1e-4, "x");
    }

    [Fact]
    public void OneDimensional_MatchesScipy()
    {
        var r = Optimize.Shgo(x => (x[0] - 0.3) * (x[0] - 0.3) * (x[0] + 0.5) * (x[0] + 0.5) * (x[0] - 1.2) * (x[0] - 1.2) + 0.1 * x[0],
            Bounds.Uniform(1, -1, 2), iters: 5);
        // scipy 1.17.1: funl [-0.0512377, -0.0512377, 0.0248752, 0.0248752, 0.1188415], nfev 83, nlfev 68
        OptimizeAssert.Close([-0.05123772432044155, -0.05123772432044123, 0.02487519328765323, 0.024875193287653658, 0.11884149103359302], r.Funl!, 1e-10, "funl");
        OptimizeAssert.Close(-0.5237813870162068, r.X[0], 1e-7, "x");
        Assert.Equal(15, r.Nfev - r.Nlfev);
    }

    [Fact]
    public void ConstrainedDocExample_CattleFeed()
    {
        IMinimizeConstraint[] cons =
        [
            OptimizeConstraint.Ineq(x => 2.3 * x[0] + 5.6 * x[1] + 11.1 * x[2] + 1.3 * x[3] - 5),
            OptimizeConstraint.Ineq(x => 12 * x[0] + 11.9 * x[1] + 41.8 * x[2] + 52.1 * x[3] - 21
                - 1.645 * Math.Sqrt(0.28 * x[0] * x[0] + 0.19 * x[1] * x[1] + 20.5 * x[2] * x[2] + 0.62 * x[3] * x[3])),
            OptimizeConstraint.Eq(x => x[0] + x[1] + x[2] + x[3] - 1),
        ];
        var r = Optimize.Shgo(x => 24.55 * x[0] + 26.75 * x[1] + 39 * x[2] + 40.50 * x[3], Bounds.Uniform(4, 0, 1), cons, n: 150);
        Assert.True(r.Success, r.Message);
        // scipy 1.17.1: fun 29.89437811824064, x [0.63552157, 1.02e-13, 0.31270188, 0.05177655]
        OptimizeAssert.Close(29.894378159142136, r.Fun, 1e-6, "fun");
        OptimizeAssert.Close([0.6355215712243022, 0, 0.3127018804868929, 0.05177654828870282], r.X, 1e-6, "x");
        OptimizeAssert.Close(1, r.X.Sum(), 1e-9, "equality");
    }

    [Fact]
    public void SinCos_SimplicialAndHaltonAgreeOnTheGlobalMinimum()
    {
        // Two symmetric global minima at (−0.51221, ±1.49573), f = −1.738210901945599 (scipy 1.17.1).
        var simp = Optimize.Shgo(SinCos, Bounds.Uniform(2, -3, 3), iters: 4);
        OptimizeAssert.Close(-1.738210901945599, simp.Fun, 1e-10, "simplicial");
        OptimizeAssert.Close(-0.5122140355623124, simp.X[0], 1e-6, "x0");
        OptimizeAssert.Close(1.495728225387953, Math.Abs(simp.X[1]), 1e-6, "|x1|");
        var halton = Optimize.Shgo(SinCos, Bounds.Uniform(2, -3, 3), n: 40, iters: 2, samplingMethod: ShgoSamplingMethod.Halton);
        OptimizeAssert.Close(-1.738210901945599, halton.Fun, 1e-9, "halton");
        AssertSortedAndConsistent(halton);
        // The three basins (−1.7382, −1.5237, −1.0946) are all represented.
        foreach (var e in new[] { -1.738210901945599, -1.523658831061529, -1.0945767083840698 })
            Assert.Contains(simp.Funl!, v => Math.Abs(v - e) < 1e-8);
    }

    [Fact]
    public void LocalMinimizerCanBeLbfgsb()
    {
        var r = Optimize.Shgo(Eggholder, Bounds.Uniform(2, -512, 512), n: null, iters: 4,
            minimizerKwargs: new ShgoMinimizerKwargs { Method = MinimizeMethod.LBfgsB });
        // scipy 1.17.1 with L-BFGS-B: fun −935.3379515605109, 23 local minima
        OptimizeAssert.Close(-935.3379515605109, r.Fun, 1e-6, "fun");
        Assert.Equal(23, r.Funl!.Length);
    }

    [Fact]
    public void FMinStopsEarly()
    {
        var r = Optimize.Shgo(SinCos, Bounds.Uniform(2, -3, 3), options: new ShgoOptions { FMin = -1.7382109, FTol = 1e-4 });
        OptimizeAssert.Close(-1.738210901945599, r.Fun, 1e-8, "fun");
        Assert.True(r.Nit <= 5, $"nit {r.Nit}");
    }

    [Fact]
    public void InfeasibleEverywhereFails()
    {
        // −1 − x0² ≥ 0 never holds: scipy reports "Failed to find a feasible minimizer point."
        var r = Optimize.Shgo(x => x[0] * x[0] + x[1] * x[1], Bounds.Uniform(2, -1, 1),
            [OptimizeConstraint.Ineq(x => -1 - x[0] * x[0])]);
        Assert.False(r.Success);
        Assert.StartsWith("Failed to find a feasible minimizer point.", r.Message);
        Assert.Equal(0, r.Nfev);
    }

    [Fact]
    public void LegacyBoundsOverloadAndLocalIter()
    {
        var all = Optimize.Shgo(SinCos, [(-3.0, 3.0), (-3.0, 3.0)], iters: 3);
        var one = Optimize.Shgo(SinCos, [(-3.0, 3.0), (-3.0, 3.0)], iters: 3, options: new ShgoOptions { LocalIter = 1 });
        Assert.True(one.Funl!.Length < all.Funl!.Length, $"{one.Funl.Length} vs {all.Funl.Length}");
        Assert.True(one.Nlfev < all.Nlfev);
    }

    private static void AssertSortedAndConsistent(OptimizeResult r)
    {
        var funl = r.Funl!;
        for (var i = 1; i < funl.Length; i++) Assert.True(funl[i - 1] <= funl[i], "funl not sorted");
        Assert.Equal(funl.Length, r.Xl!.Shape[0]);
        Assert.Equal(r.Fun, funl[0]);
        var xl0 = r.Xl.AsContiguous().ToArray().AsSpan(0, r.X.Length).ToArray();
        Assert.Equal(r.X, xl0);
    }
}
