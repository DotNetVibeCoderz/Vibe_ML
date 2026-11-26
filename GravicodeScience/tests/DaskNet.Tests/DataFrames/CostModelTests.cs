using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.GraviFrame;
using Xunit;
using static Gravicode.Science.DaskNet.Expr;

namespace Gravicode.Science.Tests.DaskNet;

/// <summary>
/// Size estimates and the join choices made from them. Thresholds are passed to
/// <c>CostModel.ChooseJoins</c> directly rather than through the global setting, which tests in
/// other classes read concurrently.
/// </summary>
public sealed class CostModelTests
{
    private static readonly DataFrame Sales = SalesData.Make(3_000, seed: 21);

    private static readonly DataFrame Products = new([
        new TextSeries("product", Enumerable.Range(0, 12).Select(i => (string?)$"p{i:D2}").ToArray()),
        new NumericSeries("weight", Enumerable.Range(0, 12).Select(i => i * 0.5).ToArray()),
        new NumericSeries("qty", Enumerable.Range(0, 12).Select(i => 100.0 + i).ToArray()),   // clashes with sales.qty
    ]);

    [Fact]
    public void Estimates_follow_the_data_and_the_rules_of_thumb()
    {
        var df = DaskFrame.FromFrame(Sales, 4);
        var all = CostModel.Bytes(df.Plan)!.Value;
        Assert.Equal(SizeOf.Estimate(Sales), all, (double)all * 0.01);   // partitions sum to the whole, up to headers

        var two = CostModel.Bytes(df.Select("id", "qty").Plan)!.Value;
        Assert.Equal(all * 2 / 7, two, (double)all * 0.01);              // 2 of 7 columns
        Assert.Equal(all / 2, CostModel.Bytes(df.Filter(Col("qty") > 3).Plan)!.Value, (double)all * 0.01);
        Assert.Equal(1024, CostModel.Bytes(df.Aggregate(new Aggregation("n", "size", null)).Plan));
        Assert.Equal(all, CostModel.Bytes(df.GroupBy("region").Sum("qty").Plan));   // grouped: bounded by the input only

        var opaque = DaskFrame.FromDelayed([Dask.Value(Sales)], Sales.ColumnNames.ToArray());
        Assert.Null(CostModel.Bytes(opaque.Plan));
        Assert.Null(CostModel.Bytes(df.Join(opaque, "id").Plan));

        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, "a,b\n" + string.Concat(Enumerable.Range(0, 5000).Select(i => $"{i},{i * 2}\n")));
            Assert.Equal(new FileInfo(path).Length, CostModel.Bytes(DaskFrame.ReadCsv(path, blockSize: 4096).Plan)!.Value, 64.0);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void A_small_right_side_in_several_partitions_is_broadcast()
    {
        var left = DaskFrame.FromFrame(Sales, 4);
        var right = DaskFrame.FromFrame(Products, 3);
        var join = (JoinNode)left.Join(right, "product").Plan;
        Assert.False(join.Broadcast);

        var chosen = CostModel.ChooseJoins(join, threshold: 1L << 20);
        Assert.True(((JoinNode)chosen).Broadcast);
        Assert.Contains("Repartition 3 -> 1", chosen.Explain());
        Assert.Same(join, CostModel.ChooseJoins(join, threshold: 100));       // too big for this threshold
        Assert.Same(join, CostModel.ChooseJoins(join, threshold: 0));         // switched off

        // Right and outer joins must see every right row in one place; they always shuffle.
        var outer = left.Join(right, "product", JoinKind.Outer).Plan;
        Assert.Same(outer, CostModel.ChooseJoins(outer, threshold: 1L << 20));

        // The caller's choice wins.
        var pinned = left.Join(right, "product", broadcast: false).Plan;
        Assert.Same(pinned, CostModel.ChooseJoins(pinned, threshold: 1L << 20));
    }

    /// <summary>
    /// A small left side of an inner join is turned around, and the result must be the same
    /// frame: the same column names, in the same order - including the suffixed clash
    /// (<c>qty</c>/<c>qty_right</c>) and a key both sides share - and the same rows.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void A_small_left_side_is_turned_around_without_changing_the_result(bool twoKeys)
    {
        var small = DaskFrame.FromFrame(Products, 1);
        var big = DaskFrame.FromFrame(Sales, 5).WithColumn("weight", Col("price") / 10);   // a second shared name
        DaskFrame joined;
        DataFrame expected;
        if (twoKeys)
        {
            joined = small.Merge(big, ["product", "weight"], ["product", "weight"]);
            expected = Joins.MergeOn(Products, big.Compute(), ["product", "weight"]);
        }
        else
        {
            joined = small.Join(big, "product");
            expected = Joins.Merge(Products, big.Compute(), "product", "product");
        }

        var smallBytes = CostModel.Bytes(small.Plan)!.Value;
        var chosen = CostModel.ChooseJoins(joined.Plan, threshold: smallBytes + 1);
        Assert.IsType<ProjectNode>(chosen);
        var turned = (JoinNode)chosen.Children[0];
        Assert.True(turned.Broadcast);
        Assert.Same(big.Plan, turned.Left);

        var actual = new DaskFrame(chosen).Compute();
        Assert.Equal(joined.Columns, actual.ColumnNames);
        Assert.Equal(expected.ColumnNames, actual.ColumnNames);
        Assert.Equal(SalesData.Rows(expected, sorted: true), SalesData.Rows(actual, sorted: true));
        if (!twoKeys) Assert.Contains("qty_right", actual.ColumnNames);
    }

    [Fact]
    public void Explain_shows_the_choice_and_results_do_not_change()
    {
        var left = DaskFrame.FromFrame(Sales, 4).Filter(Col("price") > 30);
        var right = DaskFrame.FromFrame(Products, 3);
        var byDefault = left.Join(right, "product");               // 16 MB default: the right side is tiny
        var shuffled = left.Join(right, "product", broadcast: false);
        var forced = left.Join(right, "product", broadcast: true);
        Assert.Contains("broadcast right", byDefault.Explain());
        Assert.Contains("hash shuffle", shuffled.Explain());
        Assert.Contains("broadcast right", forced.Explain());
        var reference = SalesData.Rows(shuffled.Compute(), sorted: true);
        Assert.Equal(reference, SalesData.Rows(byDefault.Compute(), sorted: true));
        Assert.Equal(reference, SalesData.Rows(forced.Compute(), sorted: true));

        Assert.Throws<ArgumentException>(() => left.Join(right, "product", JoinKind.Right, broadcast: true));
        Assert.Throws<ArgumentException>(() => left.Join(right, "product", partitions: 4, broadcast: true));
    }
}
