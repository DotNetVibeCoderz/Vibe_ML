using Gravicode.Science.GraviFrame;
using Xunit;

namespace Gravicode.Science.Tests.GraviFrame;

/// <summary>
/// Tests for the dictionary-encoded categorical column.
/// </summary>
/// <remarks>
/// Two things separate a categorical from a text column, and both are checked directly: the
/// dictionary is part of the column's type rather than a summary of its rows, and an ordered
/// categorical sorts by declared rank rather than alphabetically.
/// </remarks>
public class CategoricalSeriesTests
{
    private static CategoricalSeries Sizes() => CategoricalSeries.FromValues(
        "size", ["medium", "low", "high", "low", null, "medium"],
        categories: ["low", "medium", "high"], ordered: true);

    [Fact]
    public void ValuesRoundTripThroughTheirCodes()
    {
        var series = Sizes();

        Assert.Equal("medium", series[0]);
        Assert.Equal("low", series[1]);
        Assert.Equal("high", series[2]);
        Assert.Null(series[4]);
        Assert.True(series.IsMissing(4));

        Assert.Equal([1, 0, 2, 0, -1, 1], series.Codes.ToArray());
    }

    [Fact]
    public void FirstSeenOrderIsUsedWhenNoCategoriesAreGiven()
    {
        var series = CategoricalSeries.FromValues("colour", ["red", "blue", "red", "green"]);

        Assert.Equal(["red", "blue", "green"], series.Categories);
        Assert.Equal([0, 1, 0, 2], series.Codes.ToArray());
    }

    [Fact]
    public void AnExplicitCategoryListActsAsAWhitelist()
    {
        // Values outside the declared set are missing data, not new categories. Inventing a
        // category here would make the column's type depend on the rows that happened to arrive.
        var series = CategoricalSeries.FromValues(
            "grade", ["A", "B", "Z", "A"], categories: ["A", "B", "C"]);

        Assert.Equal(["A", "B", "C"], series.Categories);
        Assert.True(series.IsMissing(2));
        Assert.Equal(1, series.MissingCount);
    }

    [Fact]
    public void OrderedCategoricalsSortByRankRatherThanAlphabet()
    {
        // The wrong answer a text column gives: alphabetically "high" < "low" < "medium", which
        // reverses the actual ordering of the data.
        var order = Sizes().ArgSort();

        // low, low, medium, medium, high, then the missing value last.
        Assert.Equal(["low", "low", "medium", "medium", "high"],
            order.Take(5).Select(i => Sizes()[i]!).ToArray());
        Assert.Null(Sizes()[order[5]]);
    }

    [Fact]
    public void MissingSortsLastInBothDirections()
    {
        // Missing is absent, not extreme; putting it at one end would make it the minimum or the
        // maximum depending on the sort direction.
        var series = Sizes();

        Assert.Null(series[series.ArgSort()[^1]]);
        Assert.Null(series[series.ArgSort(descending: true)[^1]]);
    }

    [Fact]
    public void DescendingSortReversesTheRanking()
    {
        var series = Sizes();
        var order = series.ArgSort(descending: true);

        Assert.Equal(["high", "medium", "medium", "low", "low"],
            order.Take(5).Select(i => series[i]!).ToArray());
    }

    [Fact]
    public void TakeKeepsEveryCategoryEvenWhenNoRowsUseIt()
    {
        // Filtering rows must not change the column's type. If "high" vanished from the categories
        // here, a later group-by would stop reporting it as an empty group.
        var filtered = (CategoricalSeries)Sizes().Take([1, 3]);

        Assert.Equal(["low", "medium", "high"], filtered.Categories);
        Assert.Equal(2, filtered.Length);
        Assert.Equal(0, filtered.CategoryCounts().Single(c => c.Category == "high").Count);
    }

    [Fact]
    public void RemoveUnusedCategoriesIsTheExplicitWayToDropThem()
    {
        var trimmed = ((CategoricalSeries)Sizes().Take([1, 3])).RemoveUnusedCategories();

        Assert.Equal(["low"], trimmed.Categories);
        Assert.Equal("low", trimmed[0]);
        Assert.Equal("low", trimmed[1]);
    }

    [Fact]
    public void CategoryCountsReportsUnusedCategoriesAsZero()
    {
        var counts = CategoricalSeries.FromValues(
            "grade", ["A", "A", "B"], categories: ["A", "B", "C"]).CategoryCounts();

        Assert.Equal([("A", 2), ("B", 1), ("C", 0)], counts.ToArray());
    }

    [Fact]
    public void ReorderingCategoriesRemapsTheCodesRatherThanRelabelling()
    {
        // The bug this guards against: swapping the category array without touching the codes,
        // which silently renames every value in the column.
        var series = CategoricalSeries.FromValues("size", ["low", "high", "medium"]);
        var reordered = series.ReorderCategories(["low", "medium", "high"]);

        Assert.Equal("low", reordered[0]);
        Assert.Equal("high", reordered[1]);
        Assert.Equal("medium", reordered[2]);
        Assert.True(reordered.IsOrdered);

        // And now it sorts by rank.
        Assert.Equal(["low", "medium", "high"],
            reordered.ArgSort().Select(i => reordered[i]!).ToArray());
    }

    [Fact]
    public void ReorderingMustNameEveryExistingCategory()
    {
        // Omitting one would turn its rows into missing data without saying so.
        var series = CategoricalSeries.FromValues("size", ["low", "medium", "high"]);

        var error = Assert.Throws<ArgumentException>(
            () => series.ReorderCategories(["low", "medium"]));
        Assert.Contains("high", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RenamingCategoriesKeepsTheCodesAndOrder()
    {
        var renamed = Sizes().RenameCategories(
            new Dictionary<string, string> { ["low"] = "S", ["medium"] = "M", ["high"] = "L" });

        Assert.Equal(["S", "M", "L"], renamed.Categories);
        Assert.Equal("M", renamed[0]);
        Assert.Equal(Sizes().Codes.ToArray(), renamed.Codes.ToArray());
    }

    [Fact]
    public void WritingAValueOutsideTheCategorySetThrows()
    {
        // Widening the dictionary on assignment would make the column's type depend on the order
        // writes happened in.
        var series = Sizes();

        var error = Assert.Throws<ArgumentException>(() => series[0] = "extreme");
        Assert.Contains("AddCategories", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AddCategoriesWidensTheDictionaryWithoutDisturbingExistingCodes()
    {
        var widened = Sizes().AddCategories("extreme");

        Assert.Equal(["low", "medium", "high", "extreme"], widened.Categories);
        Assert.Equal("medium", widened[0]);

        widened[0] = "extreme";
        Assert.Equal("extreme", widened[0]);
    }

    [Fact]
    public void WritingNullMakesTheValueMissing()
    {
        var series = Sizes();
        series[0] = null;

        Assert.True(series.IsMissing(0));
        Assert.Equal(CategoricalSeries.MissingCode, series.Codes[0]);
    }

    [Fact]
    public void OneHotProducesOneColumnPerCategory()
    {
        var columns = CategoricalSeries.FromValues("colour", ["red", "blue", "red"]).OneHot();

        Assert.Equal(2, columns.Count);
        Assert.Equal("colour_red", columns[0].Name);
        Assert.Equal([1, 0, 1], columns[0].Values.ToArray());
        Assert.Equal([0, 1, 0], columns[1].Values.ToArray());
    }

    [Fact]
    public void DropFirstLeavesABaselineCategory()
    {
        // Keeping every column alongside an intercept makes the design matrix rank-deficient,
        // because the indicator columns sum to a constant.
        var columns = CategoricalSeries.FromValues("colour", ["red", "blue", "green"]).OneHot(dropFirst: true);

        Assert.Equal(2, columns.Count);
        Assert.DoesNotContain(columns, c => c.Name == "colour_red");
    }

    [Fact]
    public void AMissingValueIsZeroInEveryIndicatorColumn()
    {
        var columns = CategoricalSeries.FromValues("colour", ["red", null]).OneHot();

        foreach (var column in columns)
            Assert.Equal(0.0, column.Values[1]);
    }

    [Fact]
    public void CodesExposeMissingAsNaNRatherThanMinusOne()
    {
        // -1 would read as a category ranked below every other one, which is exactly the wrong
        // thing to hand a model.
        var codes = Sizes().ToCodes();

        Assert.Equal(1.0, codes.Values[0]);
        Assert.True(double.IsNaN(codes.Values[4]));
    }

    [Fact]
    public void RoundTripsThroughText()
    {
        var text = Sizes().ToText();

        Assert.Equal("medium", text[0]);
        Assert.Null(text[4]);

        var back = CategoricalSeries.FromValues("size", text.Values.ToArray(),
            categories: ["low", "medium", "high"], ordered: true);
        Assert.Equal(Sizes().Codes.ToArray(), back.Codes.ToArray());
    }

    [Fact]
    public void ACategoricalWorksAsAFrameColumn()
    {
        var frame = new DataFrame(
        [
            new NumericSeries("amount", [10, 20, 30]),
            CategoricalSeries.FromValues("size", ["low", "high", "low"],
                categories: ["low", "medium", "high"], ordered: true),
        ]);

        Assert.Equal(3, frame.RowCount);
        Assert.Equal(DataType.Categorical, frame["size"].DataType);
        Assert.Equal("high", frame["size"].GetValue(1));
    }

    [Fact]
    public void MalformedInputsAreRejected()
    {
        // A code outside the dictionary would be an out-of-range read on every access.
        Assert.Throws<ArgumentException>(() => new CategoricalSeries("x", [0, 5], ["a", "b"]));

        // A duplicated category makes the name-to-code lookup ambiguous.
        Assert.Throws<ArgumentException>(() => new CategoricalSeries("x", [0], ["a", "a"]));

        // Adding a category that already exists silently would create the same ambiguity.
        Assert.Throws<ArgumentException>(() => Sizes().AddCategories("low"));
    }
}
