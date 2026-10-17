using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpatialAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// Trees large enough to be built in parallel (subtrees of 32k+ points split across threads) and
/// queried through the stack-allocated small-k path and the heap path, against brute force.
/// </summary>
public class KDTreeLargeTests
{
    private static (double D, int I)[] Brute(double[] x, int n, double[] q, int j, int d, int k)
    {
        var best = new (double D, int I)[n];
        for (var i = 0; i < n; i++)
        {
            var s = 0.0;
            for (var t = 0; t < d; t++) { var v = x[i * d + t] - q[j * d + t]; s += v * v; }
            best[i] = (Math.Sqrt(s), i);
        }
        return best.OrderBy(b => b.D).ThenBy(b => b.I).Take(k).ToArray();
    }

    [Theory]
    [InlineData(true, 6)]
    [InlineData(false, 6)]
    [InlineData(true, 40)] // above the small-k limit: binary-heap path
    public void ParallelBuild_QueriesMatchBruteForce(bool balanced, int k)
    {
        const int n = 70_000, d = 3, q = 60;
        var x = Lcg(n, d, 7);
        var qs = Lcg(q, d, 8);
        var tree = new KDTree(new NdArray(x, n, d), balancedTree: balanced);

        // Every point is in exactly one leaf and the index is a permutation.
        Assert.Equal(Enumerable.Range(0, n), tree.Indices.OrderBy(i => i));

        var (dist, idx) = tree.Query(new NdArray(qs, q, d), k);
        var dv = dist.ToArray();
        for (var j = 0; j < q; j++)
        {
            var brute = Brute(x, n, qs, j, d, k);
            for (var r = 0; r < k; r++)
            {
                Close(brute[r].D, dv[j * k + r], 1e-12, $"distance q={j} r={r}");
                Assert.Equal(brute[r].I, idx[j, r]);
            }
        }
    }

    [Fact]
    public void ParallelBuild_SameTreeWhateverTheBatch()
    {
        // Node numbering is preorder in both the parallel build and a small sequential one, so a
        // subset query through the large tree must agree point for point with workers = 1.
        const int n = 40_000, d = 2;
        var x = LcgNd(n, d, 11);
        var tree = new KDTree(x, leafSize: 4);
        var qs = LcgNd(2_000, d, 12);
        var (d1, i1) = tree.Query(qs, 5, workers: 1);
        var (dAll, iAll) = tree.Query(qs, 5, workers: -1);
        Assert.Equal(d1.ToArray(), dAll.ToArray());
        Assert.Equal(i1.Cast<int>(), iAll.Cast<int>());
    }

    [Fact]
    public void UpperBoundAndApproximation_StillHonoured()
    {
        const int n = 50_000, d = 3;
        var x = Lcg(n, d, 21);
        var tree = new KDTree(new NdArray(x, n, d));
        var qs = Lcg(50, d, 22);
        var (dist, idx) = tree.Query(new NdArray(qs, 50, d), 4, distanceUpperBound: 0.01);
        var dv = dist.ToArray();
        for (var j = 0; j < 50; j++)
        {
            var brute = Brute(x, n, qs, j, d, 4).Where(b => b.D < 0.01).ToArray();
            for (var r = 0; r < 4; r++)
            {
                if (r < brute.Length) { Assert.Equal(brute[r].I, idx[j, r]); }
                else { Assert.True(double.IsPositiveInfinity(dv[j * 4 + r])); Assert.Equal(n, idx[j, r]); }
            }
        }
        // eps > 0: the k-th returned distance is within (1 + eps) of the true k-th.
        var (de, _) = tree.Query(new NdArray(qs, 50, d), 3, eps: 0.5);
        var dev = de.ToArray();
        for (var j = 0; j < 50; j++)
            Assert.True(dev[j * 3 + 2] <= 1.5 * Brute(x, n, qs, j, d, 3)[2].D + 1e-15, $"eps bound q={j}");
    }
}
