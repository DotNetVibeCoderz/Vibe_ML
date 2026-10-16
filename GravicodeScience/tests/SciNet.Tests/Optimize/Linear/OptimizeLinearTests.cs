using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class OptimizeLinearTests
{
    public static TheoryData<LinprogMethod> Methods => new() { LinprogMethod.Simplex, LinprogMethod.InteriorPoint };

    [Theory]
    [MemberData(nameof(Methods))]
    public void Linprog_DocsExampleWithMarginals(LinprogMethod method)
    {
        // scipy 1.17.1 (HiGHS): x = (10, -3), fun = -22, ineqlin marginals (0, -1), lower marginals (0, 6)
        var r = Optimize.Linprog([-1, 4], new NdArray([-3.0, 1, 1, 2], 2, 2), [6, 4], bounds: Bounds.FromPairs((null, null), (-3, null)), method: method);
        Assert.True(r.Success, r.Message);
        OptimizeAssert.Close([10, -3], r.X, 1e-7, "x");
        OptimizeAssert.Close(-22, r.Fun, 1e-7, "fun");
        OptimizeAssert.Close([39, 0], r.Slack, 1e-6, "slack");
        OptimizeAssert.Close([0, -1], r.IneqlinMarginals, 1e-7, "ineqlin");
        OptimizeAssert.Close([0, 6], r.LowerMarginals, 1e-7, "lower");
        OptimizeAssert.Close([0, 0], r.UpperMarginals, 1e-7, "upper");
    }

    [Theory]
    [MemberData(nameof(Methods))]
    public void Linprog_EqualityProblemMatchesScipy(LinprogMethod method)
    {
        var r = Optimize.Linprog([1, 2, 3, 0], aEq: new NdArray([1.0, 1, 1, 1, 1, -1, 0, 2], 2, 4), bEq: [10, 2], method: method);
        OptimizeAssert.Close([0, 6, 0, 4], r.X, 1e-7, "x");
        OptimizeAssert.Close(12, r.Fun, 1e-7, "fun");
        OptimizeAssert.Close([1.3333333333333333, -0.6666666666666666], r.EqlinMarginals, 1e-7, "eqlin");
        OptimizeAssert.Close([0.33333333333333337, 0.0, 1.6666666666666667, 0.0], r.LowerMarginals, 1e-7, "lower");
        OptimizeAssert.Close([0, 0], r.Con, 1e-8, "con");
    }

    [Theory]
    [MemberData(nameof(Methods))]
    public void Linprog_DetectsInfeasibleAndUnbounded(LinprogMethod method)
    {
        Assert.Equal(2, Optimize.Linprog([1, 1], new NdArray([1.0, 1], 1, 2), [-1], method: method).Status);
        Assert.Equal(3, Optimize.Linprog([-1, -1], new NdArray([1.0, -1], 1, 2), [1], method: method).Status);
        // Contradictory equalities.
        Assert.Equal(2, Optimize.Linprog([1, 1], aEq: new NdArray([1.0, 1, 1, 1], 2, 2), bEq: [1, 2], method: method).Status);
    }

    [Fact]
    public void Linprog_DegenerateAndCyclingProneProblems()
    {
        // Beale's example cycles under the textbook rule; scipy 1.17.1: x = (0.04, 0, 1, 0), fun = -0.05.
        var beale = Optimize.Linprog([-0.75, 150, -0.02, 6], new NdArray([0.25, -60, -0.04, 9, 0.5, -90, -0.02, 3, 0, 0, 1, 0], 3, 4), [0, 0, 1]);
        OptimizeAssert.Close([0.04, 0, 1, 0], beale.X, 1e-9, "beale");
        OptimizeAssert.Close(-0.05, beale.Fun, 1e-12, "beale fun");
        var bland = Optimize.Linprog([-0.75, 150, -0.02, 6], new NdArray([0.25, -60, -0.04, 9, 0.5, -90, -0.02, 3, 0, 0, 1, 0], 3, 4), [0, 0, 1],
            options: new LinprogOptions { Bland = true });
        OptimizeAssert.Close(-0.05, bland.Fun, 1e-12, "bland");
        // Klee–Minty cube, n = 3: optimum (0, 0, 125).
        var km = Optimize.Linprog([-4, -2, -1], new NdArray([1.0, 0, 0, 4, 1, 0, 8, 4, 1], 3, 3), [5, 25, 125]);
        OptimizeAssert.Close([0, 0, 125], km.X, 1e-9, "klee-minty");
        // Redundant equality rows are fine.
        var red = Optimize.Linprog([1, 1], aEq: new NdArray([1.0, 1, 2, 2], 2, 2), bEq: [1, 2]);
        Assert.True(red.Success);
        OptimizeAssert.Close(1, red.Fun, 1e-12, "redundant");
    }

    [Theory]
    [MemberData(nameof(Methods))]
    public void Linprog_RandomProblemMatchesScipy(LinprogMethod method)
    {
        double[] a =
        [
            0.6369616873214543, 0.2697867137638703, 0.04097352393619469, 0.016527635528529094, 0.8132702392002724,
            0.9127555772777217, 0.6066357757671799, 0.7294965609839984, 0.5436249914654229, 0.9350724237877682,
            0.8158535541215322, 0.002738500170148095, 0.8574042765875693, 0.033585575305464355, 0.7296554464299441,
            0.17565562060255901, 0.8631789223498866, 0.5414612202490917, 0.2997118905373848, 0.42268722119765845,
            0.028319671145462966, 0.12428327649956394, 0.6706244146936303, 0.6471895115742501, 0.6153851114812539,
            0.38367755426188344, 0.997209935789211, 0.9808353387762301, 0.6855419844806947, 0.6504592762678163,
        ];
        double[] b = [1.68844673057094, 1.3889214239791037, 1.1350965050224113, 1.7214883401940817, 1.5253543224757258, 1.3102418755589555];
        double[] c = [-0.4858353588317891, -0.8894878343490003, -0.9340435159562497, -0.35779519670907023, -0.5715298307297609];
        var r = Optimize.Linprog(c, new NdArray(a, 6, 5), b, method: method);
        OptimizeAssert.Close([0.7526671337553897, 0.42795774011954746, 0.6063165308697661, 0, 0], r.X, 1e-7, "x");
        OptimizeAssert.Close(-1.312661534736765, r.Fun, 1e-8, "fun");
        OptimizeAssert.Close([0, -0.15777105556293974, -0.044700469166081835, 0, 0, -0.7958764016861078], r.IneqlinMarginals, 1e-7, "duals");
    }

    // ----- MILP vs brute force -----

    [Fact]
    public void Milp_KnapsackMatchesBruteForce()
    {
        var rng = new GraviRandom(11);
        for (var trial = 0; trial < 5; trial++)
        {
            const int n = 10;
            var w = Enumerable.Range(0, n).Select(_ => (double)rng.Next(1, 20)).ToArray();
            var v = Enumerable.Range(0, n).Select(_ => (double)rng.Next(1, 30)).ToArray();
            var cap = w.Sum() / 2;
            var best = 0.0;
            for (var mask = 0; mask < 1 << n; mask++)
            {
                double ww = 0, vv = 0;
                for (var j = 0; j < n; j++) if ((mask >> j & 1) == 1) { ww += w[j]; vv += v[j]; }
                if (ww <= cap) best = Math.Max(best, vv);
            }
            var r = Optimize.Milp(v.Select(t => -t).ToArray(), Enumerable.Repeat(1, n).ToArray(), Bounds.Uniform(n, 0, 1),
                new LinearConstraint(new NdArray(w, 1, n), double.NegativeInfinity, cap), new MilpOptions { MipRelGap = 0 });
            Assert.True(r.Success, r.Message);
            OptimizeAssert.Close(-best, r.Fun!.Value, 1e-9, $"knapsack {trial}");
            Assert.All(r.X!, x => Assert.True(x == 0 || x == 1));
        }
    }

    [Fact]
    public void Milp_AssignmentAndInfeasible()
    {
        // 4x4 assignment as a MILP must agree with LinearSumAssignment.
        double[] cost = [9, 2, 7, 8, 6, 4, 3, 7, 5, 8, 1, 8, 7, 6, 9, 4];
        var rows = new List<double>();
        for (var i = 0; i < 4; i++) { var r = new double[16]; for (var j = 0; j < 4; j++) r[i * 4 + j] = 1; rows.AddRange(r); }
        for (var j = 0; j < 4; j++) { var r = new double[16]; for (var i = 0; i < 4; i++) r[i * 4 + j] = 1; rows.AddRange(r); }
        var res = Optimize.Milp(cost, Enumerable.Repeat(1, 16).ToArray(), Bounds.Uniform(16, 0, 1),
            [new LinearConstraint(new NdArray(rows.ToArray(), 8, 16), 1, 1)]);
        var (ri, ci) = Optimize.LinearSumAssignment(new NdArray(cost, 4, 4));
        var lsa = ri.Select((r, k) => cost[r * 4 + ci[k]]).Sum();
        OptimizeAssert.Close(lsa, res.Fun!.Value, 1e-9, "assignment");
        Assert.Equal(13.0, lsa);
        // x integer in [0, 1] with 2x = 1 has no solution.
        var inf = Optimize.Milp([1.0], [1], Bounds.Uniform(1, 0, 1), new LinearConstraint(new NdArray([2.0], 1, 1), 1, 1));
        Assert.Equal(2, inf.Status);
    }

    // ----- linear sum assignment -----

    [Fact]
    public void LinearSumAssignment_MatchesScipy()
    {
        var (r1, c1) = Optimize.LinearSumAssignment(new double[,] { { 4, 1, 3 }, { 2, 0, 5 }, { 3, 2, 2 } });
        Assert.Equal([0, 1, 2], r1); Assert.Equal([1, 0, 2], c1);
        var c2 = new double[,] { { 4, 1, 3, 7 }, { 2, 0, 5, 1 }, { 3, 2, 2, 9 } };
        var (r2, cc2) = Optimize.LinearSumAssignment(c2);
        Assert.Equal([0, 1, 2], r2); Assert.Equal([1, 3, 2], cc2);
        var (_, cm) = Optimize.LinearSumAssignment(c2, maximize: true);
        Assert.Equal([0, 2, 3], cm);
        var t = NdArray.FromArray(c2).T.Copy();
        var (rt, ct) = Optimize.LinearSumAssignment(t);
        Assert.Equal([1, 2, 3], rt); Assert.Equal([0, 2, 1], ct);
        var inf = double.PositiveInfinity;
        var (_, ci) = Optimize.LinearSumAssignment(new double[,] { { inf, 1, 3 }, { 2, inf, 5 }, { 3, 2, inf } });
        Assert.Equal([2, 0, 1], ci);
        Assert.Throws<ArgumentException>(() => Optimize.LinearSumAssignment(new double[,] { { inf, inf }, { 1, 2 } }));
        Assert.Throws<ArgumentException>(() => Optimize.LinearSumAssignment(new double[,] { { double.NaN, 1 }, { 1, 2 } }));
    }

    [Fact]
    public void LinearSumAssignment_AgreesWithBruteForce()
    {
        var rng = new GraviRandom(3);
        for (var trial = 0; trial < 30; trial++)
        {
            int nr = 2 + trial % 4, nc = 2 + (trial / 4) % 4;
            var c = new double[nr * nc];
            for (var i = 0; i < c.Length; i++) c[i] = Math.Round(rng.Uniform(0, 10), 1);
            var (ri, ci) = Optimize.LinearSumAssignment(new NdArray(c, nr, nc));
            var got = ri.Select((r, k) => c[r * nc + ci[k]]).Sum();
            Assert.Equal(Math.Min(nr, nc), ri.Length);
            Assert.True(ri.SequenceEqual(ri.Order()));
            Assert.Equal(BruteForce(c, nr, nc), got, 9);
        }
    }

    private static double BruteForce(double[] c, int nr, int nc)
    {
        var best = double.PositiveInfinity;
        var k = Math.Min(nr, nc);
        void Rec(int r, bool[] usedC, bool[] usedR, int placed, double s)
        {
            if (placed == k) { best = Math.Min(best, s); return; }
            if (nr <= nc)
            {
                for (var j = 0; j < nc; j++) if (!usedC[j]) { usedC[j] = true; Rec(r + 1, usedC, usedR, placed + 1, s + c[r * nc + j]); usedC[j] = false; }
            }
            else
            {
                var col = placed;
                for (var i = 0; i < nr; i++) if (!usedR[i]) { usedR[i] = true; Rec(r, usedC, usedR, placed + 1, s + c[i * nc + col]); usedR[i] = false; }
            }
        }
        Rec(0, new bool[nc], new bool[nr], 0, 0);
        return best;
    }
}
