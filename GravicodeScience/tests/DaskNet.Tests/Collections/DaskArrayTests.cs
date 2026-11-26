using Gravicode.Science.DaskNet;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet;

public sealed class DaskArrayTests
{
    private static NdArray Sample(params int[] shape)
    {
        var rng = new GraviRandom(7);
        var data = new double[Shapes.Size(shape)];
        for (var i = 0; i < data.Length; i++) data[i] = rng.Normal(3, 2);
        return new NdArray(data, shape);
    }

    private static void Close(NdArray expected, NdArray actual, double tol = 1e-10)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var e = expected.ToArray();
        var a = actual.ToArray();
        for (var i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= tol * Math.Max(1, Math.Abs(e[i])), $"[{i}] {e[i]} vs {a[i]}");
    }

    [Fact]
    public void Chunks_cover_uneven_shapes()
    {
        var x = DaskArray.FromArray(Sample(10, 7), 4, 3);
        Assert.Equal(new[] { 4, 4, 2 }, x.Chunks[0]);
        Assert.Equal(new[] { 3, 3, 1 }, x.Chunks[1]);
        Assert.Equal(9, x.BlockCount);
        Close(Sample(10, 7), x.Compute(), 0);
    }

    [Fact]
    public void Elementwise_arithmetic_matches_numpy_semantics()
    {
        var a = Sample(9, 11);
        var b = Sample(9, 11) * 0.5 + 1.0;
        var da = DaskArray.FromArray(a, 4);
        var db = DaskArray.FromArray(b, 3, 5); // different chunks: rechunked to match
        Close(a + b, (da + db).Compute());
        Close(a * b - a / b, (da * db - da / db).Compute());
        Close(2.0 - a, (2.0 - da).Compute());
        Close(UFunc.Exp(a * 0.1), (da * 0.1).Exp().Compute());
        Close(UFunc.Sqrt(UFunc.Abs(a)), da.Abs().Sqrt().Compute());
        Close(UFunc.Maximum(a, b), da.Maximum(db).Compute());
    }

    [Fact]
    public void Full_reductions_match_ndarray()
    {
        var a = Sample(37, 23);
        var da = DaskArray.FromArray(a, 5, 4);
        Assert.True(Math.Abs(a.Sum() - da.Sum().Compute()) < 1e-9);
        Assert.Equal(a.Min(), da.Min().Compute());
        Assert.Equal(a.Max(), da.Max().Compute());
        Assert.True(Math.Abs(a.Mean() - da.Mean().Compute()) < 1e-12);
        Assert.True(Math.Abs(a.Var() - da.Var().Compute()) < 1e-10);
        Assert.True(Math.Abs(a.Std(1) - da.Std(ddof: 1).Compute()) < 1e-10);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Axis_reductions_match_ndarray(int axis)
    {
        var a = Sample(6, 9, 5);
        var da = DaskArray.FromArray(a, 4, 2, 3);
        Close(Statistics.Sum(a, axis), da.Sum(axis).Compute());
        Close(Statistics.Mean(a, axis), da.Mean(axis).Compute());
        Close(Statistics.Min(a, axis), da.Min(axis).Compute(), 0);
        Close(Statistics.Max(a, axis), da.Max(axis).Compute(), 0);
        Close(Statistics.Var(a, axis: axis), da.Var(axis, 0).Compute());
        Close(Statistics.Std(a, axis: axis, ddof: 1), da.Std(axis, ddof: 1).Compute());
    }

    [Fact]
    public void Reduction_trees_are_deep_when_blocks_are_many()
    {
        var da = DaskArray.Ones([1000], 3);
        Assert.Equal(1000.0, da.WithSplitEvery(2).Sum().Compute());
        Assert.Equal(1000.0, da.WithSplitEvery(3).Sum(axis: 0).Compute().At(0));
        Assert.True(Dask.Graph(da.WithSplitEvery(2).Sum()).Count > Dask.Graph(da.Sum()).Count);
    }

    [Fact]
    public void Blocked_matmul_matches_linalg()
    {
        var a = Sample(17, 13);
        var b = Sample(13, 9);
        var product = DaskArray.FromArray(a, 5, 4).Dot(DaskArray.FromArray(b, 6, 4)); // k chunks differ
        Close(LinAlg.Dot(a, b), product.Compute(), 1e-9);
    }

    [Fact]
    public void Transpose_and_rechunk_preserve_values()
    {
        var a = Sample(8, 6, 3);
        var da = DaskArray.FromArray(a, 3, 4, 2);
        Close(a.Transpose(), da.T.Compute(), 0);
        var re = da.Rechunk(5, 1, 3);
        Assert.Equal(new[] { 5, 3 }, re.Chunks[0]);
        Close(a, re.Compute(), 0);
    }

    [Fact]
    public void Random_blocks_do_not_depend_on_the_scheduler()
    {
        var r = DaskArray.RandomNormal([200, 50], [64, 16], seed: 11);
        var threads = r.Compute(new ThreadScheduler(4));
        var sync = r.Compute(new SynchronousScheduler());
        Close(threads, sync, 0);
        Assert.True(Math.Abs(threads.Mean()) < 0.05);
        Assert.True(Math.Abs(threads.Std() - 1) < 0.05);
        // Different blocks are different streams.
        Assert.NotEqual(threads.At(0), threads.At(16));
    }

    [Fact]
    public void Map_blocks_runs_locally_and_checks_shape()
    {
        var da = DaskArray.Arange(50, 7);
        Assert.Equal(Enumerable.Range(0, 50).Sum(i => i * 3.0), da.Map(v => v * 3).Sum().Compute());
        Assert.Throws<DaskTaskException>(() => da.MapBlocks(b => b.Reshape(b.Size, 1).Slice(Slice.Range(0, 1))).Compute());
    }

    [Fact]
    public void Array_graphs_are_remotable()
    {
        var a = DaskArray.Random([40, 40], [10, 10]);
        Assert.True(Dask.Graph(a.Dot(a.T).Sum(axis: 1)).IsRemotable);
        Assert.True(Dask.Graph(a.Std()).IsRemotable);
        Assert.True(Dask.Graph(a.Std(axis: 0, ddof: 0)).IsRemotable);
    }
}
