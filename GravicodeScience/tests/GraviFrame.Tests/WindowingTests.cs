using Gravicode.Science.GraviFrame;
using Xunit;

namespace Gravicode.Science.Tests.GraviFrame;

/// <summary>
/// Tests for partitioned window functions and the as-of join.
/// </summary>
/// <remarks>
/// The interesting cases are all about boundaries. A window function that ignores its partitions
/// produces numbers that look reasonable and mix one group's history into another's, so most of
/// these check what happens at the edge between groups rather than in the middle of one.
/// </remarks>
public class WindowingTests
{
    /// <summary>Two customers, interleaved, so a partition bug cannot hide.</summary>
    private static DataFrame Sales() => new(
    [
        new TextSeries("customer", ["a", "b", "a", "b", "a", "b"]),
        new NumericSeries("day", [1, 1, 2, 2, 3, 3]),
        new NumericSeries("amount", [10, 100, 20, 200, 30, 300]),
    ]);

    [Fact]
    public void CumulativeSumRestartsAtEachPartition()
    {
        var result = Windowing.CumulativeSum(Sales(), ["customer"], "amount");

        // a: 10, 30, 60   b: 100, 300, 600 — interleaved in the frame's row order.
        Assert.Equal([10, 100, 30, 300, 60, 600], result.Values.ToArray());
    }

    [Fact]
    public void LagDoesNotReachAcrossAPartitionBoundary()
    {
        // The leak that matters: without partitioning, customer b's first row would lag onto
        // customer a's last, and a model trained on it would look better than it is.
        var result = Windowing.Lag(Sales(), ["customer"], "amount");

        Assert.True(double.IsNaN(result[0]));    // a, first row
        Assert.True(double.IsNaN(result[1]));    // b, first row
        Assert.Equal(10.0, result[2]);           // a sees a's previous
        Assert.Equal(100.0, result[3]);          // b sees b's previous
    }

    [Fact]
    public void LeadIsTheMirrorOfLag()
    {
        var result = Windowing.Lead(Sales(), ["customer"], "amount");

        Assert.Equal(20.0, result[0]);
        Assert.Equal(200.0, result[1]);
        Assert.True(double.IsNaN(result[4]));    // a, last row
        Assert.True(double.IsNaN(result[5]));    // b, last row
    }

    [Fact]
    public void RollingMeanLeavesIncompleteWindowsMissing()
    {
        var result = Windowing.RollingMean(Sales(), ["customer"], "amount", window: 2);

        // Filling a partial window would give the first rows far more variance than the rest,
        // with nothing in the output to say so.
        Assert.True(double.IsNaN(result[0]));
        Assert.True(double.IsNaN(result[1]));
        Assert.Equal(15.0, result[2], 10);       // (10 + 20) / 2
        Assert.Equal(150.0, result[3], 10);      // (100 + 200) / 2
        Assert.Equal(25.0, result[4], 10);       // (20 + 30) / 2
    }

    [Fact]
    public void RankOrdersWithinEachPartition()
    {
        var ascending = Windowing.Rank(Sales(), ["customer"], "amount");
        Assert.Equal([1, 1, 2, 2, 3, 3], ascending.Values.ToArray());

        var descending = Windowing.Rank(Sales(), ["customer"], "amount", descending: true);
        Assert.Equal([3, 3, 2, 2, 1, 1], descending.Values.ToArray());
    }

    [Fact]
    public void TiesAreRankedAccordingToTheDenseFlag()
    {
        var frame = new DataFrame([new NumericSeries("score", [10, 20, 20, 30])]);

        // Standard ranking leaves a gap after the tie; dense ranking does not.
        Assert.Equal([1, 2, 2, 4], Windowing.Rank(frame, [], "score").Values.ToArray());
        Assert.Equal([1, 2, 2, 3], Windowing.Rank(frame, [], "score", dense: true).Values.ToArray());
    }

    [Fact]
    public void AnEmptyPartitionListTreatsTheWholeFrameAsOneGroup()
    {
        // What SQL does, and worth pinning so the degenerate case is not an error.
        var result = Windowing.CumulativeSum(Sales(), [], "amount");
        Assert.Equal([10, 110, 130, 330, 360, 660], result.Values.ToArray());
    }

    // ---------------------------------------------------------------- as-of join

    [Fact]
    public void AsOfJoinTakesTheMostRecentEarlierRow()
    {
        var trades = new DataFrame(
        [
            new NumericSeries("time", [10, 25, 40]),
            new NumericSeries("size", [1, 2, 3]),
        ]);

        var quotes = new DataFrame(
        [
            new NumericSeries("time", [5, 20, 30, 50]),
            new NumericSeries("price", [100, 200, 300, 400]),
        ]);

        var joined = Windowing.AsOfJoin(trades, quotes, "time");

        Assert.Equal(3, joined.RowCount);
        Assert.Equal(100.0, joined.Numeric("price")[0]);   // t=10 sees the quote from t=5
        Assert.Equal(200.0, joined.Numeric("price")[1]);   // t=25 sees t=20
        Assert.Equal(300.0, joined.Numeric("price")[2]);   // t=40 sees t=30, not t=50
    }

    [Fact]
    public void AsOfJoinNeverLooksForward()
    {
        // The whole point of the backward-only rule. A left row before every right row must come
        // back missing, not matched to the first future one.
        var left = new DataFrame([new NumericSeries("time", [1])]);
        var right = new DataFrame(
        [
            new NumericSeries("time", [5, 10]),
            new NumericSeries("price", [50, 100]),
        ]);

        var joined = Windowing.AsOfJoin(left, right, "time");
        Assert.True(double.IsNaN(joined.Numeric("price")[0]));
    }

    [Fact]
    public void AsOfJoinHonoursATolerance()
    {
        var left = new DataFrame([new NumericSeries("time", [100])]);
        var right = new DataFrame(
        [
            new NumericSeries("time", [10]),
            new NumericSeries("price", [42]),
        ]);

        // A quote ninety units stale is worse than no quote, if the caller says so.
        Assert.Equal(42.0, Windowing.AsOfJoin(left, right, "time").Numeric("price")[0]);
        Assert.True(double.IsNaN(
            Windowing.AsOfJoin(left, right, "time", tolerance: 30).Numeric("price")[0]));
    }

    [Fact]
    public void AsOfJoinSortsTheRightFrameItself()
    {
        // Demanding sorted input is an easy way to get silently wrong answers, so the join sorts.
        var left = new DataFrame([new NumericSeries("time", [25])]);
        var right = new DataFrame(
        [
            new NumericSeries("time", [30, 10, 20]),      // deliberately unordered
            new NumericSeries("price", [300, 100, 200]),
        ]);

        Assert.Equal(200.0, Windowing.AsOfJoin(left, right, "time").Numeric("price")[0]);
    }

    [Fact]
    public void AsOfJoinCarriesTextColumnsAndSuffixesCollisions()
    {
        var left = new DataFrame(
        [
            new NumericSeries("time", [10]),
            new NumericSeries("value", [1]),
        ]);

        var right = new DataFrame(
        [
            new NumericSeries("time", [5]),
            new NumericSeries("value", [99]),
            new TextSeries("label", ["open"]),
        ]);

        var joined = Windowing.AsOfJoin(left, right, "time");

        Assert.Equal(1.0, joined.Numeric("value")[0]);
        Assert.Equal(99.0, joined.Numeric("value_right")[0]);
        Assert.Equal("open", ((TextSeries)joined["label"])[0]);
    }
}
