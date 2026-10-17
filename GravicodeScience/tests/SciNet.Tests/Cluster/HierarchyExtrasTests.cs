using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpatialAssert;

namespace Gravicode.Science.Tests.SciNet;

public class HierarchyExtrasTests
{
    // scipy 1.17.1: Z = linkage(lcg_points(12, 2, 4), 'average')
    private static NdArray Z() => new(new double[]
    {
        3, 7, 0.045454443551690454, 2, 2, 6, 0.05722921577975698, 2, 10, 12, 0.08691354957177005, 3,
        0, 8, 0.10577301836448699, 2, 11, 15, 0.18456893810052855, 3, 13, 14, 0.24558195686420559, 5,
        5, 9, 0.26314965178331634, 2, 1, 17, 0.39482183296003537, 6, 4, 19, 0.40789643633660866, 7,
        16, 20, 0.46941157336413253, 10, 18, 21, 0.670748876706871, 12,
    }, 11, 4);

    [Fact]
    public void LinkageFixture_MatchesOurLinkage()
    {
        var z = Cluster.Linkage(LcgNd(12, 2, 4), LinkageMethod.Average).ToArray();
        Close(Z().ToArray(), z, 1e-15, "Z");
    }

    [Theory]
    // scipy 1.17.1: T = fcluster(Z, k, 'maxclust'); L, M = leaders(Z, T)
    [InlineData(1, new[] { 22 }, new[] { 1 })]
    [InlineData(3, new[] { 16, 20, 18 }, new[] { 2, 3, 1 })]
    [InlineData(5, new[] { 1, 17, 4, 16, 18 }, new[] { 4, 3, 5, 2, 1 })]
    [InlineData(12, new[] { 5, 9, 0, 8, 11, 2, 6, 3, 7, 10, 1, 4 }, new[] { 6, 10, 1, 9, 12, 3, 7, 4, 8, 11, 2, 5 })]
    public void Leaders_MatchScipy(int k, int[] l, int[] m)
    {
        var t = Cluster.Fcluster(Z(), k, FclusterCriterion.Maxclust);
        var (L, M) = Cluster.Leaders(Z(), t);
        Assert.Equal(l, L);
        Assert.Equal(m, M);
    }

    [Fact]
    public void Leaders_RejectsNonSubtreeAssignment()
    {
        // Observations 3 and 7 merge first; giving them different labels while 3 shares a label
        // with a far-away point is not a union of subtrees.
        var t = new int[12];
        for (var i = 0; i < 12; i++) t[i] = i + 1;
        t[3] = t[5];
        Assert.Throws<ArgumentException>(() => Cluster.Leaders(Z(), t));
    }

    [Fact]
    public void IsIsomorphic_IsABijectionTest()
    {
        Assert.True(Cluster.IsIsomorphic([1, 1, 2, 3], [7, 7, 5, 9]));
        Assert.False(Cluster.IsIsomorphic([1, 1, 2, 3], [7, 7, 7, 9]));
        Assert.False(Cluster.IsIsomorphic([1, 2, 2, 3], [7, 7, 5, 9]));
        var a = Cluster.Fcluster(Z(), 4, FclusterCriterion.Maxclust);
        var b = a.Select(x => 10 - x).ToArray();
        Assert.True(Cluster.IsIsomorphic(a, b));
    }

    [Fact]
    public void MlabLinkage_RoundTrips()
    {
        var mlab = Cluster.ToMlabLinkage(Z());
        // scipy 1.17.1: to_mlab_linkage(Z)[:2]
        Close([4, 8, 0.045454443551690454, 3, 7, 0.05722921577975698], mlab.ToArray()[..6], 0, "mlab");
        Close(Z().ToArray(), Cluster.FromMlabLinkage(mlab).ToArray(), 0, "round trip");
    }

    [Fact]
    public void IsValidIm_ChecksShapeAndSigns()
    {
        var r = Cluster.Inconsistent(Z());
        Assert.True(Cluster.IsValidIm(r));
        var bad = r.ToArray(); bad[1] = -1;
        Assert.False(Cluster.IsValidIm(new NdArray(bad, r.Shape[0], 4)));
        Assert.Throws<ArgumentException>(() => Cluster.IsValidIm(new NdArray(bad, r.Shape[0], 4), throwOnError: true));
    }

    [Fact]
    public void ClusterNodes_CompareByDistance()
    {
        var root = Cluster.ToTree(Z(), out var nodes);
        Assert.True(nodes[12].CompareTo(nodes[22]) < 0);
        Assert.Equal(root, nodes.Max());
    }

    [Fact]
    public void DisjointSet_FollowsScipyRootRule()
    {
        // scipy 1.17.1: ds = DisjointSet([1, 2, 3, 'a', 'b']) style usage with ints.
        var ds = new ClusterDisjointSet<int>([1, 2, 3, 4, 5]);
        Assert.Equal(5, ds.NSubsets);
        Assert.True(ds.Merge(1, 2));
        Assert.True(ds.Merge(3, 4));
        Assert.False(ds.Merge(2, 1));
        Assert.Equal(1, ds[2]);   // equal sizes: the earlier root wins
        Assert.True(ds.Merge(5, 3)); // {3,4} is larger: its root 3 wins
        Assert.Equal(3, ds[5]);
        Assert.True(ds.Merge(4, 1)); // equal sizes 3 vs 2? {3,4,5} larger -> root 3
        Assert.Equal(3, ds[1]);
        Assert.Equal(1, ds.NSubsets);
        Assert.Equal(5, ds.SubsetSize(2));
        Assert.True(ds.Connected(1, 5));
        ds.Add(9);
        Assert.Equal(2, ds.Subsets().Count);
        Assert.Throws<KeyNotFoundException>(() => ds[42]);
    }
}
