using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet;

/// <summary>Chunked broadcasting is checked against GraviNum broadcasting the whole arrays.</summary>
public sealed class BroadcastTests
{
    private static void Same(NdArray expected, NdArray actual)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var e = expected.ToArray();
        var a = actual.ToArray();
        for (var i = 0; i < e.Length; i++) Assert.True(Math.Abs(e[i] - a[i]) <= 1e-12 * (1 + Math.Abs(e[i])), $"element {i}: {e[i]} vs {a[i]}");
    }

    [Fact]
    public void Centring_columns_broadcasts_a_reduced_axis()
    {
        var x = DaskArray.RandomNormal([120, 80], [30, 20], seed: 3);
        var whole = x.Compute();
        var centred = x - x.Mean(axis: 0);
        Assert.Equal([120, 80], centred.Shape);
        Same(whole - Statistics.Mean(whole, axis: 0), centred.Compute());
        // Each column of the result has mean zero.
        Assert.All(centred.Mean(axis: 0).Compute().ToArray(), m => Assert.True(Math.Abs(m) < 1e-12));
    }

    [Fact]
    public void Rows_normalise_through_expand_dims()
    {
        var x = DaskArray.Random([90, 40], [25, 15], seed: 4) + 0.1;
        var rowSums = x.Sum(axis: 1).ExpandDims(1);
        Assert.Equal([90, 1], rowSums.Shape);
        var normalised = x / rowSums;
        var whole = x.Compute();
        Same(whole / Statistics.Sum(whole, axis: 1).Reshape(90, 1), normalised.Compute());
        Assert.All(normalised.Sum(axis: 1).Compute().ToArray(), s => Assert.True(Math.Abs(s - 1) < 1e-12));
    }

    [Fact]
    public void Operands_with_different_chunkings_and_both_sides_stretching()
    {
        var a = DaskArray.FromArray(NdArray.Arange(12).Reshape(3, 1, 4), 2, 1, 3);
        var b = DaskArray.FromArray(NdArray.Arange(8).Reshape(2, 4) * 10, 1, 4);
        var sum = a + b;
        Assert.Equal([3, 2, 4], sum.Shape);
        Same(NdArray.Arange(12).Reshape(3, 1, 4) + NdArray.Arange(8).Reshape(2, 4) * 10, sum.Compute());

        // The same column vector in a chunking unrelated to the matrix's.
        var m = DaskArray.RandomNormal([60, 50], [20, 25], seed: 9);
        var v = DaskArray.FromArray(NdArray.Arange(50), 7);
        Same(m.Compute() * NdArray.Arange(50), (m * v).Compute());
        Same(NdArray.Arange(50) - m.Compute(), (v - m).Compute());
    }

    [Fact]
    public void Incompatible_shapes_are_refused()
    {
        var e = Assert.Throws<ArgumentException>(() => DaskArray.Ones([4, 3], 2) + DaskArray.Ones([4], 2));
        Assert.Contains("cannot be broadcast", e.Message);
    }

    [Fact]
    public async Task Broadcast_graphs_stay_remotable()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 2);
        var x = DaskArray.RandomNormal([80, 60], [20, 20], seed: 1);
        var z = (x - x.Mean(axis: 0)) / x.Std(axis: 0, ddof: 1);
        Same(z.Compute(), z.Compute(cluster.Client));
    }
}
