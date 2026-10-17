using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class OptimizeAssignmentTests
{
    // scipy.optimize.quadratic_assignment docstring example (flows × distances).
    private static readonly double[,] DocA = { { 0, 80, 150, 170 }, { 80, 0, 130, 100 }, { 150, 130, 0, 120 }, { 170, 100, 120, 0 } };
    private static readonly double[,] DocB = { { 0, 5, 2, 7 }, { 0, 0, 3, 8 }, { 0, 0, 0, 3 }, { 0, 0, 0, 0 } };

    private static double BruteForceOptimum(double[,] a, double[,] b, bool maximize)
    {
        var n = a.GetLength(0);
        var best = maximize ? double.NegativeInfinity : double.PositiveInfinity;
        var perm = Enumerable.Range(0, n).ToArray();
        void Rec(int k)
        {
            if (k == n)
            {
                double s = 0;
                for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) s += a[i, j] * b[perm[i], perm[j]];
                best = maximize ? Math.Max(best, s) : Math.Min(best, s);
                return;
            }
            for (var i = k; i < n; i++)
            {
                (perm[k], perm[i]) = (perm[i], perm[k]);
                Rec(k + 1);
                (perm[k], perm[i]) = (perm[i], perm[k]);
            }
        }
        Rec(0);
        return best;
    }

    [Fact]
    public void Faq_DocExampleFindsTheOptimum()
    {
        var r = Optimize.QuadraticAssignment(DocA, DocB);
        // scipy 1.17.1: col_ind [0 3 2 1], fun 3260, nit 9
        Assert.Equal([0, 3, 2, 1], r.ColInd);
        Assert.Equal(3260, r.Fun);
        Assert.Equal(BruteForceOptimum(DocA, DocB, false), r.Fun);
    }

    [Fact]
    public void TwoOpt_WithPartialGuessFindsTheOptimum()
    {
        var r = Optimize.QuadraticAssignment(DocA, DocB, QuadraticAssignmentMethod.TwoOpt,
            new QuadraticAssignmentOptions { PartialGuess = [(0, 3), (1, 0)], Rng = new GraviRandom(1) });
        Assert.Equal(3260, r.Fun);
    }

    [Fact]
    public void TwoOpt_IsALocalOptimumUnderEverySwap()
    {
        var (a, b) = Structured();
        var r = Optimize.QuadraticAssignment(a, b, QuadraticAssignmentMethod.TwoOpt, new QuadraticAssignmentOptions { Rng = new GraviRandom(7) });
        var p = r.ColInd;
        for (var i = 0; i < 12; i++)
            for (var j = i + 1; j < 12; j++)
            {
                (p[i], p[j]) = (p[j], p[i]);
                Assert.True(Score(a, b, p) >= r.Fun, $"swap ({i},{j}) improves");
                (p[i], p[j]) = (p[j], p[i]);
            }
        Assert.Equal(Score(a, b, r.ColInd), r.Fun);
    }

    private static double Score(double[,] a, double[,] b, int[] p)
    {
        double s = 0;
        for (var i = 0; i < p.Length; i++) for (var j = 0; j < p.Length; j++) s += a[i, j] * b[p[i], p[j]];
        return s;
    }

    private static (double[,] A, double[,] B) Structured()
    {
        var a = new double[12, 12]; var b = new double[12, 12];
        for (var i = 0; i < 12; i++)
            for (var j = 0; j < 12; j++)
            {
                a[i, j] = (i * 7 + j * 3) % 11;
                b[i, j] = (i * 5 + j * 2 + 1) % 13;
            }
        return (a, b);
    }

    [Fact]
    public void Faq_MatchesScipyOnADeterministicTwelveNodeProblem()
    {
        // Barycentre start and no shuffling make FAQ deterministic; integer data keeps it exact.
        var (a, b) = Structured();
        var r = Optimize.QuadraticAssignment(a, b);
        // scipy 1.17.1: [8, 10, 6, 0, 2, 4, 7, 1, 11, 5, 9, 3] 3628.0 nit 5
        Assert.Equal([8, 10, 6, 0, 2, 4, 7, 1, 11, 5, 9, 3], r.ColInd);
        Assert.Equal(3628, r.Fun);
        Assert.Equal(5, r.Nit);

        var mx = Optimize.QuadraticAssignment(a, b, options: new QuadraticAssignmentOptions { Maximize = true });
        // scipy 1.17.1: [0, 7, 5, 10, 1, 2, 11, 9, 8, 6, 4, 3] 5010.0 nit 6
        Assert.Equal([0, 7, 5, 10, 1, 2, 11, 9, 8, 6, 4, 3], mx.ColInd);
        Assert.Equal(5010, mx.Fun);

        var seeded = Optimize.QuadraticAssignment(a, b, options: new QuadraticAssignmentOptions { PartialMatch = [(0, 5), (3, 1)] });
        // scipy 1.17.1: [5, 8, 4, 1, 0, 9, 11, 7, 2, 3, 10, 6] 3661.0 nit 20
        Assert.Equal([5, 8, 4, 1, 0, 9, 11, 7, 2, 3, 10, 6], seeded.ColInd);
        Assert.Equal(3661, seeded.Fun);
    }

    [Fact]
    public void Faq_RandomizedStartIsAValidPermutation()
    {
        var (a, b) = Structured();
        var r = Optimize.QuadraticAssignment(a, b, options: new QuadraticAssignmentOptions { RandomizedP0 = true, ShuffleInput = true, Rng = new GraviRandom(3) });
        Assert.Equal(Enumerable.Range(0, 12), r.ColInd.Order());
        Assert.Equal(Score(a, b, r.ColInd), r.Fun);
    }

    [Fact]
    public void QuadraticAssignment_RejectsBadInput()
    {
        Assert.Throws<ArgumentException>(() => Optimize.QuadraticAssignment(new double[2, 3], new double[2, 3]));
        Assert.Throws<ArgumentException>(() => Optimize.QuadraticAssignment(DocA, DocB, options: new QuadraticAssignmentOptions { PartialMatch = [(0, 1), (1, 1)] }));
    }

    [Fact]
    public void Isotonic_DocExample()
    {
        double[] y = [1.5, 1.0, 4.0, 6.0, 5.7, 5.0, 7.8, 9.0, 7.5, 9.5, 9.0];
        var r = Optimize.IsotonicRegression(y);
        OptimizeAssert.Close([1.25, 1.25, 4, 5.566666666666666, 5.566666666666666, 5.566666666666666, 7.8, 8.25, 8.25, 9.25, 9.25], r.X, 1e-12, "x");
        Assert.Equal([0, 2, 3, 6, 7, 9, 11], r.Blocks);
        OptimizeAssert.Close([2, 1, 3, 1, 2, 2], r.Weights, 0, "w");
    }

    [Fact]
    public void Isotonic_WeightedRaggedSeriesBothDirections()
    {
        // Ragged data where pooling must cascade backwards several blocks at once.
        double[] y = [1.259, 0.449, -0.574, 1.251, -1.517, -1.284, -0.261, -2.494, -5.46, -1.492, 0.18, 2.617, -2.074, -2.69, -1.253];
        double[] w = [1.061, 0.636, 1.491, 1.897, 0.811, 1.445, 0.947, 1.613, 1.583, 0.828, 1.745, 1.486, 1.524, 1.73, 1.143];
        var up = Optimize.IsotonicRegression(y, w);
        // scipy 1.17.1
        Assert.Equal([0, 10, 15], up.Blocks);
        OptimizeAssert.Close(-1.1448615984405457, up.X[0], 1e-13, "x0");
        OptimizeAssert.Close(-0.6612077871001573, up.X[14], 1e-13, "x14");
        OptimizeAssert.Close([12.312, 7.628], up.Weights, 1e-12, "w");

        var down = Optimize.IsotonicRegression(y, w, increasing: false);
        Assert.Equal([0, 1, 2, 4, 7, 12, 13, 15], down.Blocks);
        OptimizeAssert.Close([1.259, 0.449, 0.447849173553719, 0.447849173553719, -1.0405351233218858, -1.0405351233218858, -1.0405351233218858,
            -1.3367906271536867, -1.3367906271536867, -1.3367906271536867, -1.3367906271536867, -1.3367906271536867, -2.074, -2.118301079011486, -2.118301079011486],
            down.X, 1e-13, "down");
        OptimizeAssert.Close([1.061, 0.636, 3.388, 3.203, 7.255, 1.524, 2.873], down.Weights, 1e-12, "wdown");
    }

    [Fact]
    public void Isotonic_IsTheConstrainedLeastSquaresOptimum()
    {
        // Against an independent solver: SLSQP on the same QP with x[i+1] − x[i] ≥ 0.
        double[] y = [3, 1, 2, 5, 4, 4, 0, 6];
        var r = Optimize.IsotonicRegression(y);
        var cons = Enumerable.Range(0, y.Length - 1).Select(i => (IMinimizeConstraint)OptimizeConstraint.Ineq(x => x[i + 1] - x[i])).ToArray();
        var qp = Optimize.Minimize(x => x.Select((v, i) => (v - y[i]) * (v - y[i])).Sum(), y, MinimizeMethod.Slsqp, constraints: cons,
            options: new SlsqpOptions { Ftol = 1e-12, MaxIter = 500 });
        OptimizeAssert.Close(qp.X, r.X, 1e-5, "vs SLSQP");
        Assert.Equal([0, 3, 7, 8], r.Blocks); // (3,1,2) → 2 and (5,4,4,0) → 3.25
    }

    [Fact]
    public void Isotonic_TiesArePooledAndEdgeCases()
    {
        Assert.Equal([0, 1, 3, 4], Optimize.IsotonicRegression([1.0, 2, 2, 3]).Blocks);
        var dec = Optimize.IsotonicRegression([1.0, 2, 2, 3], increasing: false);
        Assert.Equal([0, 4], dec.Blocks);
        OptimizeAssert.Close([2, 2, 2, 2], dec.X, 0, "pooled");
        Assert.Empty(Optimize.IsotonicRegression([]).X);
        Assert.Throws<ArgumentException>(() => Optimize.IsotonicRegression([1.0, 2], [1.0, 0]));
    }
}
