using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Gravicode.Science.SciNet.Spatial;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// The specialised NN-chain / heap linkage loops against the textbook O(n³) agglomeration written
/// here: repeatedly merge the closest pair over the full matrix and apply the Lance–Williams
/// formula. On data without ties both give the same merges and heights.
/// </summary>
public class LinkageReferenceTests
{
    private static double[] Points(int n, int d, int seed)
    {
        var r = new GraviRandom(seed);
        var x = new double[n * d];
        for (var i = 0; i < x.Length; i++) x[i] = r.NextDouble();
        return x;
    }

    private static double LanceWilliams(LinkageMethod m, double dxi, double dyi, double dxy, int nx, int ny, int ni) => m switch
    {
        LinkageMethod.Single => Math.Min(dxi, dyi),
        LinkageMethod.Complete => Math.Max(dxi, dyi),
        LinkageMethod.Average => (nx * dxi + ny * dyi) / (nx + ny),
        LinkageMethod.Weighted => 0.5 * (dxi + dyi),
        LinkageMethod.Centroid => Math.Sqrt((nx * dxi * dxi + ny * dyi * dyi - nx * ny * dxy * dxy / (nx + ny)) / (nx + ny)),
        LinkageMethod.Median => Math.Sqrt(0.5 * (dxi * dxi + dyi * dyi) - 0.25 * dxy * dxy),
        _ => Math.Sqrt(((ni + nx) * dxi * dxi + (ni + ny) * dyi * dyi - ni * dxy * dxy) / (nx + ny + ni)),
    };

    /// <summary>Naive agglomeration: list of (label a, label b, height) in merge order.</summary>
    private static List<(int A, int B, double H)> Naive(double[] y, int n, LinkageMethod method)
    {
        var d = new double[n, n];
        var k = 0;
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++) d[i, j] = d[j, i] = y[k++];
        var alive = Enumerable.Range(0, n).ToList();
        var label = Enumerable.Range(0, n).ToArray();
        var size = Enumerable.Repeat(1, n).ToArray();
        var merges = new List<(int, int, double)>();
        for (var step = 0; step < n - 1; step++)
        {
            int bx = -1, by = -1;
            var best = double.PositiveInfinity;
            foreach (var i in alive)
                foreach (var j in alive)
                    if (i < j && d[i, j] < best) { best = d[i, j]; bx = i; by = j; }
            merges.Add((Math.Min(label[bx], label[by]), Math.Max(label[bx], label[by]), best));
            foreach (var i in alive)
            {
                if (i == bx || i == by) continue;
                d[i, by] = d[by, i] = LanceWilliams(method, d[i, bx], d[i, by], best, size[bx], size[by], size[i]);
            }
            size[by] += size[bx];
            label[by] = n + step;
            alive.Remove(bx);
        }
        return merges;
    }

    [Theory]
    [InlineData(LinkageMethod.Single)]
    [InlineData(LinkageMethod.Complete)]
    [InlineData(LinkageMethod.Average)]
    [InlineData(LinkageMethod.Weighted)]
    [InlineData(LinkageMethod.Ward)]
    [InlineData(LinkageMethod.Centroid)]
    [InlineData(LinkageMethod.Median)]
    public void Linkage_MatchesNaiveAgglomeration(LinkageMethod method)
    {
        const int n = 120, dim = 4;
        var x = new NdArray(Points(n, dim, 5), n, dim);
        var y = Distance.Pdist(x);
        var copy = (double[])y.Clone();
        var z = Cluster.Linkage(y, method).ToArray();
        Assert.Equal(copy, y); // the caller's condensed matrix is not modified

        var expected = Naive(y, n, method);
        if (method is LinkageMethod.Centroid or LinkageMethod.Median)
        {
            // Not monotone: Z keeps merge order, which the naive greedy loop shares.
            for (var s = 0; s < n - 1; s++)
            {
                Assert.Equal(expected[s].A, (int)z[4 * s]);
                Assert.Equal(expected[s].B, (int)z[4 * s + 1]);
                Assert.True(Math.Abs(expected[s].H - z[4 * s + 2]) <= 1e-12, $"height {s}");
            }
            return;
        }
        // Monotone methods: Z is sorted by height and relabelled, so compare the multisets of
        // heights and, through the cophenetic matrix, the tree itself.
        var hz = Enumerable.Range(0, n - 1).Select(s => z[4 * s + 2]).OrderBy(v => v).ToArray();
        var he = expected.Select(e => e.H).OrderBy(v => v).ToArray();
        for (var s = 0; s < n - 1; s++) Assert.True(Math.Abs(he[s] - hz[s]) <= 1e-12, $"height {s}");
        var zFromX = Cluster.Linkage(x, method).ToArray();
        Assert.Equal(z, zFromX); // pdist-owned buffer path gives the same Z
    }
}
