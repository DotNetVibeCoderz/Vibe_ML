using Gravicode.Science.GraviFrame;
using Xunit;

namespace Gravicode.Science.Tests.GraviFrame;

public class GroupByTests
{
    private static DataFrame Sales() => DataFrame.ParseCsv("""
        Category,Region,Sales
        A,North,10
        A,North,20
        A,South,30
        B,North,40
        B,South,50
        B,South,60
        """);

    [Fact]
    public void GroupBy_CountsGroupsInFirstSeenOrder()
    {
        var grouped = Sales().GroupBy("Category");
        Assert.Equal(2, grouped.GroupCount);
        Assert.Equal("A", grouped.Groups.First().Key[0]);
    }

    [Fact]
    public void Mean_AveragesWithinEachGroup()
    {
        var result = Sales().GroupBy("Category").Mean("Sales");

        Assert.Equal(2, result.RowCount);
        Assert.Equal("A", result.Text("Category")[0]);
        Assert.Equal(20.0, result.Numeric("Sales")[0], 9);
        Assert.Equal(50.0, result.Numeric("Sales")[1], 9);
    }

    [Fact]
    public void SumMinMaxAndMedian_AgreeWithHandComputedValues()
    {
        var grouped = Sales().GroupBy("Category");
        Assert.Equal(60.0, grouped.Sum("Sales").Numeric("Sales")[0], 9);
        Assert.Equal(10.0, grouped.Min("Sales").Numeric("Sales")[0], 9);
        Assert.Equal(30.0, grouped.Max("Sales").Numeric("Sales")[0], 9);
        Assert.Equal(20.0, grouped.Median("Sales").Numeric("Sales")[0], 9);
    }

    [Fact]
    public void Count_ReportsRowsPerGroup()
    {
        var counts = Sales().GroupBy("Category").Count();
        Assert.Equal(3.0, counts.Numeric("count")[0]);
        Assert.Equal(3.0, counts.Numeric("count")[1]);
    }

    [Fact]
    public void GroupBy_SupportsCompositeKeys()
    {
        var result = Sales().GroupBy("Category", "Region").Sum("Sales");
        Assert.Equal(4, result.RowCount);

        // A/North is the first group and holds 10 + 20.
        Assert.Equal("A", result.Text("Category")[0]);
        Assert.Equal("North", result.Text("Region")[0]);
        Assert.Equal(30.0, result.Numeric("Sales")[0], 9);
    }

    [Fact]
    public void AggregateMany_ProducesOneColumnPerRequestedStatistic()
    {
        var result = Sales().GroupBy("Category").AggregateMany([("Sales", "sum"), ("Sales", "mean")]);
        Assert.Contains("Sales_sum", result.ColumnNames);
        Assert.Contains("Sales_mean", result.ColumnNames);
        Assert.Equal(60.0, result.Numeric("Sales_sum")[0], 9);
        Assert.Equal(20.0, result.Numeric("Sales_mean")[0], 9);
    }

    [Fact]
    public void CustomReducer_IsApplied()
    {
        var range = Sales().GroupBy("Category")
            .Aggregate("Sales", v => v.Max() - v.Min(), "SalesRange");
        Assert.Equal(20.0, range.Numeric("SalesRange")[0], 9);
    }

    [Fact]
    public void UnknownAggregate_FailsWithAHelpfulMessage()
    {
        var ex = Assert.Throws<ArgumentException>(() => Sales().GroupBy("Category").Aggregate("nonsense", "Sales"));
        Assert.Contains("mean", ex.Message);
    }
}

public class ReshapingTests
{
    private static DataFrame Long() => DataFrame.ParseCsv("""
        Date,Category,Sales
        2024-01-01,A,10
        2024-01-01,B,20
        2024-01-02,A,30
        2024-01-02,B,40
        """);

    [Fact]
    public void Pivot_TurnsLongIntoWide()
    {
        var wide = Long().Pivot("Date", "Category", "Sales");

        Assert.Equal(2, wide.RowCount);
        Assert.Equal(3, wide.ColumnCount);
        Assert.Contains("A", wide.ColumnNames);
        Assert.Contains("B", wide.ColumnNames);
        Assert.Equal(10.0, wide.Numeric("A")[0], 9);
        Assert.Equal(40.0, wide.Numeric("B")[1], 9);
    }

    [Fact]
    public void Pivot_LeavesMissingCombinationsAsNaN()
    {
        var ragged = DataFrame.ParseCsv("""
            Date,Category,Sales
            2024-01-01,A,10
            2024-01-02,B,40
            """);

        var wide = ragged.Pivot("Date", "Category", "Sales");
        Assert.True(wide["B"].IsMissing(0));
        Assert.True(wide["A"].IsMissing(1));
    }

    [Fact]
    public void Pivot_AggregatesDuplicateCells()
    {
        var duplicated = DataFrame.ParseCsv("""
            Date,Category,Sales
            2024-01-01,A,10
            2024-01-01,A,30
            """);

        Assert.Equal(20.0, duplicated.Pivot("Date", "Category", "Sales").Numeric("A")[0], 9);
        Assert.Equal(40.0, duplicated.Pivot("Date", "Category", "Sales", "sum").Numeric("A")[0], 9);
    }

    [Fact]
    public void Melt_TurnsWideBackIntoLong()
    {
        var wide = Long().Pivot("Date", "Category", "Sales");
        var melted = wide.Melt(["Date"], ["A", "B"], "Category", "Sales");

        Assert.Equal(4, melted.RowCount);
        Assert.Equal(["Date", "Category", "Sales"], melted.ColumnNames);
    }

    [Fact]
    public void OneHot_ExpandsACategoricalColumn()
    {
        var encoded = Reshaping.OneHot(Long(), "Category");
        Assert.Contains("Category_A", encoded.ColumnNames);
        Assert.Contains("Category_B", encoded.ColumnNames);
        Assert.DoesNotContain("Category", encoded.ColumnNames);
        Assert.Equal(1.0, encoded.Numeric("Category_A")[0]);
        Assert.Equal(0.0, encoded.Numeric("Category_B")[0]);
    }
}

public class JoinTests
{
    private static DataFrame Left() => DataFrame.ParseCsv("""
        id,name
        1,Ana
        2,Budi
        3,Citra
        """);

    private static DataFrame Right() => DataFrame.ParseCsv("""
        id,score
        2,88
        3,91
        4,70
        """);

    [Fact]
    public void InnerJoin_KeepsOnlyMatchingKeys()
    {
        var joined = Left().Join(Right(), "id");
        Assert.Equal(2, joined.RowCount);
        Assert.Equal(["id", "name", "score"], joined.ColumnNames);
        Assert.Equal("Budi", joined.Text("name")[0]);
        Assert.Equal(88.0, joined.Numeric("score")[0]);
    }

    [Fact]
    public void LeftJoin_KeepsEveryLeftRow()
    {
        var joined = Left().Join(Right(), "id", JoinKind.Left);
        Assert.Equal(3, joined.RowCount);
        Assert.True(joined["score"].IsMissing(0));
    }

    [Fact]
    public void RightJoin_KeepsEveryRightRow()
    {
        var joined = Left().Join(Right(), "id", JoinKind.Right);
        Assert.Equal(3, joined.RowCount);
        Assert.Equal(1, joined["name"].MissingCount);
    }

    [Fact]
    public void OuterJoin_KeepsEverything()
    {
        var joined = Left().Join(Right(), "id", JoinKind.Outer);
        Assert.Equal(4, joined.RowCount);
    }

    [Fact]
    public void DuplicateKeysOnTheRightFanOut()
    {
        var right = DataFrame.ParseCsv("""
            id,score
            2,88
            2,95
            """);
        var joined = Left().Join(right, "id");
        Assert.Equal(2, joined.RowCount);
    }

    [Fact]
    public void CollidingColumnNamesTakeTheSuffix()
    {
        var right = DataFrame.ParseCsv("""
            id,name
            2,Bee
            """);
        var joined = Left().Join(right, "id");
        Assert.Contains("name_right", joined.ColumnNames);
        Assert.Equal("Bee", joined.Text("name_right")[0]);
    }
}

public class TimeSeriesTests
{
    private static NumericSeries Prices() =>
        new("price", [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);

    [Fact]
    public void Shift_MovesValuesAndPadsWithMissing()
    {
        var shifted = Prices().Shift(2);
        Assert.True(double.IsNaN(shifted[0]));
        Assert.True(double.IsNaN(shifted[1]));
        Assert.Equal(10.0, shifted[2]);

        var back = Prices().Shift(-1);
        Assert.Equal(11.0, back[0]);
        Assert.True(double.IsNaN(back[5]));
    }

    [Fact]
    public void Diff_And_PercentChange_MatchDefinitions()
    {
        var diff = Prices().Diff();
        Assert.True(double.IsNaN(diff[0]));
        Assert.Equal(1.0, diff[1], 9);

        var pct = Prices().PercentChange();
        Assert.Equal(0.1, pct[1], 9);
    }

    [Fact]
    public void RollingMean_UsesTheRequestedWindow()
    {
        var rolling = Prices().Rolling(3).Mean();

        Assert.True(double.IsNaN(rolling[0]));
        Assert.True(double.IsNaN(rolling[1]));
        Assert.Equal(11.0, rolling[2], 9);
        Assert.Equal(14.0, rolling[5], 9);
    }

    [Fact]
    public void RollingSum_MatchesADirectComputation()
    {
        var rolling = Prices().Rolling(2).Sum();
        Assert.Equal(21.0, rolling[1], 9);
        Assert.Equal(29.0, rolling[5], 9);
    }

    [Fact]
    public void RollingMinMaxAndMedian_Work()
    {
        var series = new NumericSeries("v", [5.0, 1.0, 9.0, 3.0]);
        Assert.Equal(1.0, series.Rolling(3).Min()[2], 9);
        Assert.Equal(9.0, series.Rolling(3).Max()[2], 9);
        Assert.Equal(5.0, series.Rolling(3).Median()[2], 9);
    }

    [Fact]
    public void RollingHandlesMissingValues()
    {
        var series = new NumericSeries("v", [1.0, double.NaN, 3.0, 5.0]);
        var rolling = series.Rolling(2, minPeriods: 1).Mean();

        Assert.Equal(1.0, rolling[0], 9);
        Assert.Equal(1.0, rolling[1], 9);   // only the present value counts
        Assert.Equal(3.0, rolling[2], 9);
        Assert.Equal(4.0, rolling[3], 9);
    }

    [Fact]
    public void Expanding_GrowsFromTheStart()
    {
        var expanding = Prices().Expanding().Mean();
        Assert.Equal(10.0, expanding[0], 9);
        Assert.Equal(10.5, expanding[1], 9);
        Assert.Equal(12.5, expanding[5], 9);
    }

    [Fact]
    public void ExponentialMovingAverage_StartsAtTheFirstObservation()
    {
        var ema = Prices().ExponentialMovingAverage(0.5);
        Assert.Equal(10.0, ema[0], 9);
        Assert.Equal(10.5, ema[1], 9);
        Assert.Equal(11.25, ema[2], 9);
    }

    [Fact]
    public void Interpolate_FillsGapsLinearly()
    {
        var series = new NumericSeries("v", [1.0, double.NaN, double.NaN, 4.0]);
        var filled = series.Interpolate();
        Assert.Equal(2.0, filled[1], 9);
        Assert.Equal(3.0, filled[2], 9);
    }

    [Fact]
    public void ForwardAndBackwardFill_CarryValuesOverGaps()
    {
        var series = new NumericSeries("v", [double.NaN, 2.0, double.NaN, 4.0]);
        Assert.Equal(2.0, series.ForwardFill()[2], 9);
        Assert.Equal(2.0, series.BackwardFill()[0], 9);
    }

    [Fact]
    public void Resample_BucketsByCalendarMonth()
    {
        var df = DataFrame.ParseCsv("""
            date,value
            2024-01-05,10
            2024-01-20,20
            2024-02-03,40
            2024-02-10,60
            """);

        var monthly = Resampling.Resample(df, "date", ResampleFrequency.Monthly);
        Assert.Equal(2, monthly.RowCount);
        Assert.Equal(15.0, monthly.Numeric("value")[0], 9);
        Assert.Equal(50.0, monthly.Numeric("value")[1], 9);
        Assert.Equal(new DateTime(2024, 1, 1), monthly.DateTimes("date")[0]);
    }

    [Fact]
    public void Resample_SupportsOtherFrequencies()
    {
        Assert.Equal(new DateTime(2024, 4, 1), Resampling.Floor(new DateTime(2024, 5, 17), ResampleFrequency.Quarterly));
        Assert.Equal(new DateTime(2024, 1, 1), Resampling.Floor(new DateTime(2024, 5, 17), ResampleFrequency.Yearly));
        Assert.Equal(new DateTime(2024, 5, 17), Resampling.Floor(new DateTime(2024, 5, 17, 13, 45, 0), ResampleFrequency.Daily));
    }
}

public class NumericSeriesTests
{
    [Fact]
    public void Statistics_IgnoreMissingValues()
    {
        var series = new NumericSeries("v", [1.0, double.NaN, 3.0, 5.0]);

        Assert.Equal(3, series.Count);
        Assert.Equal(9.0, series.Sum(), 9);
        Assert.Equal(3.0, series.Mean(), 9);
        Assert.Equal(1.0, series.Min(), 9);
        Assert.Equal(5.0, series.Max(), 9);
        Assert.Equal(3.0, series.Median(), 9);
    }

    [Fact]
    public void Arithmetic_WorksBetweenColumnsAndScalars()
    {
        var a = new NumericSeries("a", [1.0, 2.0, 3.0]);
        var b = new NumericSeries("b", [10.0, 20.0, 30.0]);

        Assert.Equal(11.0, (a + b)[0], 9);
        Assert.Equal(40.0, (a * b)[1], 9);
        Assert.Equal(3.0, (a * 3.0)[0], 9);
    }

    [Fact]
    public void Standardize_ProducesZeroMeanUnitDeviation()
    {
        var z = new NumericSeries("v", [2.0, 4.0, 6.0, 8.0]).Standardize();
        Assert.Equal(0.0, z.Mean(), 9);
        Assert.Equal(1.0, z.Std(), 9);
    }

    [Fact]
    public void Rank_AveragesTies()
    {
        var ranks = new NumericSeries("v", [10.0, 20.0, 20.0, 30.0]).Rank();
        Assert.Equal(1.0, ranks[0], 9);
        Assert.Equal(2.5, ranks[1], 9);
        Assert.Equal(2.5, ranks[2], 9);
        Assert.Equal(4.0, ranks[3], 9);
    }

    [Fact]
    public void Correlation_UsesOnlyRowsPresentInBoth()
    {
        var a = new NumericSeries("a", [1.0, 2.0, double.NaN, 4.0]);
        var b = new NumericSeries("b", [2.0, 4.0, 100.0, 8.0]);
        Assert.Equal(1.0, a.Correlation(b), 9);
    }

    [Fact]
    public void Factorize_EncodesCategoriesAsCodes()
    {
        var text = new TextSeries("c", ["x", "y", "x", null]);
        var (codes, categories) = text.Factorize();

        Assert.Equal(["x", "y"], categories);
        Assert.Equal(0.0, codes[0]);
        Assert.Equal(1.0, codes[1]);
        Assert.Equal(0.0, codes[2]);
        Assert.True(double.IsNaN(codes[3]));
    }
}
