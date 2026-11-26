using System.Globalization;
using Gravicode.Science.DaskNet;
using Gravicode.Science.GraviFrame;
using Xunit;
using static Gravicode.Science.DaskNet.Expr;

namespace Gravicode.Science.Tests.DaskNet;

/// <summary>Sales-like data with missing values in every column type.</summary>
internal static class SalesData
{
    public static readonly string[] Regions = ["north", "south", "east", "west", "central"];

    public static DataFrame Make(int rows = 2_000, int seed = 3)
    {
        var rng = new Random(seed);
        var region = new string?[rows];
        var product = new string?[rows];
        var price = new double[rows];
        var qty = new double[rows];
        var date = new DateTime?[rows];
        var promo = new bool?[rows];
        for (var i = 0; i < rows; i++)
        {
            region[i] = rng.Next(40) == 0 ? null : Regions[rng.Next(Regions.Length)];
            product[i] = $"p{rng.Next(12):D2}";
            price[i] = rng.Next(30) == 0 ? double.NaN : Math.Round(5 + rng.NextDouble() * 95, 2);
            qty[i] = rng.Next(1, 20);
            date[i] = rng.Next(50) == 0 ? null : new DateTime(2024, 1, 1).AddDays(rng.Next(365));
            promo[i] = rng.Next(25) == 0 ? null : rng.Next(3) == 0;
        }
        return new DataFrame([
            new NumericSeries("id", Enumerable.Range(0, rows).Select(i => (double)i).ToArray()),
            new TextSeries("region", region),
            new TextSeries("product", product),
            new NumericSeries("price", price),
            new NumericSeries("qty", qty),
            new DateTimeSeries("date", date),
            new BooleanSeries("promo", promo),
        ]);
    }

    /// <summary>Rows as comparable strings, in order.</summary>
    public static List<string> Rows(DataFrame f, bool sorted = false)
    {
        var rows = Enumerable.Range(0, f.RowCount).Select(i => string.Join("|", f.Columns.Select(c => Format(c.GetValue(i))))).ToList();
        if (sorted) rows.Sort(StringComparer.Ordinal);
        return rows;
    }

    private static string Format(object? v) => v switch
    {
        null => "∅",
        double d => Math.Round(d, 9).ToString("R", CultureInfo.InvariantCulture),
        DateTime t => t.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
        _ => v.ToString()!,
    };
}

public sealed class DaskFrameTests
{
    private static readonly DataFrame Sales = SalesData.Make();

    [Fact]
    public void FromFrame_round_trips()
    {
        var df = DaskFrame.FromFrame(Sales, 7);
        Assert.Equal(7, df.PartitionCount);
        Assert.Equal(SalesData.Rows(Sales), SalesData.Rows(df.Compute()));
        Assert.Equal(Sales.RowCount, df.Count().Compute());
    }

    [Fact]
    public void Filter_select_and_with_column_match_graviframe()
    {
        var df = DaskFrame.FromFrame(Sales, 5)
            .Filter((Col("price") > 50) & Col("region").Ne("north"))
            .WithColumn("revenue", Col("price") * Col("qty"))
            .Select("id", "region", "revenue");

        var mask = Enumerable.Range(0, Sales.RowCount)
            .Select(i => Sales.Numeric("price")[i] > 50 && Sales.Text("region")[i] is { } r && r != "north").ToArray();
        var expected = Sales.Filter(mask);
        expected = expected.WithColumn(new NumericSeries("revenue",
            Enumerable.Range(0, expected.RowCount).Select(i => expected.Numeric("price")[i] * expected.Numeric("qty")[i]).ToArray()));
        Assert.Equal(SalesData.Rows(expected.SelectColumns("id", "region", "revenue")), SalesData.Rows(df.Compute()));
    }

    [Fact]
    public void Three_valued_logic_follows_sql()
    {
        var f = new DataFrame([
            new NumericSeries("a", [1, double.NaN, 3, double.NaN]),
            new BooleanSeries("b", new bool?[] { null, false, true, null }),
        ]);
        var d = DaskFrame.FromFrame(f, 2);
        // NULL > 2 is unknown, unknown AND false is false, unknown OR true is true.
        Assert.Equal(1, d.Filter((Col("a") > 2) | Col("b")).Count().Compute());
        Assert.Equal(2, d.Filter(!((Col("a") > 2) & Col("b"))).Count().Compute()); // rows 0 and 1
        Assert.Equal(2, d.Filter(Col("a").IsNull()).Count().Compute());
        Assert.Equal(3, d.Filter(Col("a").FillNull(0) < 3).Count().Compute());
    }

    [Fact]
    public void Expressions_round_trip_through_bytes()
    {
        var e = If(Col("x").In(1, 2, 3) & !Col("s").Like("a%"), Col("y").Round(2), Null()).As("z");
        var back = FromBytes(e.ToBytes());
        Assert.Equal(e.ToString(), back.ToString());
        Assert.Equal("z", back.OutputName);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(6)]
    public void GroupBy_matches_graviframe(int splitEvery)
    {
        var old = DaskFrame.DefaultSplitEvery;
        DaskFrame.DefaultSplitEvery = Math.Max(2, splitEvery);
        try
        {
            var result = DaskFrame.FromFrame(Sales, 9).GroupBy("region")
                .Agg(new Aggregation("total", "sum", "price"), new Aggregation("avg", "mean", "price"),
                    new Aggregation("n", "count", "price"), new Aggregation("rows", "size", null),
                    new Aggregation("lo", "min", "price"), new Aggregation("hi", "max", "price"),
                    new Aggregation("sd", "std", "price"), new Aggregation("v", "var", "qty"))
                .Compute();

            var groups = Sales.GroupBy("region");
            Assert.Equal(groups.GroupCount, result.RowCount);
            foreach (var (key, rows) in groups.Groups)
            {
                var r = Enumerable.Range(0, result.RowCount).Single(i => Equals(result["region"].GetValue(i), key[0]));
                var prices = rows.Select(i => Sales.Numeric("price")[i]).Where(x => !double.IsNaN(x)).ToArray();
                var qtys = rows.Select(i => Sales.Numeric("qty")[i]).ToArray();
                Assert.True(Math.Abs(prices.Sum() - result.Numeric("total")[r]) < 1e-9);
                Assert.True(Math.Abs(prices.Average() - result.Numeric("avg")[r]) < 1e-9);
                Assert.Equal(prices.Length, result.Numeric("n")[r]);
                Assert.Equal(rows.Count, result.Numeric("rows")[r]);
                Assert.Equal(prices.Min(), result.Numeric("lo")[r]);
                Assert.Equal(prices.Max(), result.Numeric("hi")[r]);
                Assert.True(Math.Abs(GroupedDataFrame_Std(prices) - result.Numeric("sd")[r]) < 1e-9);
                Assert.True(Math.Abs(GroupedDataFrame_Var(qtys) - result.Numeric("v")[r]) < 1e-9);
            }

            // Sorted by key, missing last.
            Assert.Null(result["region"].GetValue(result.RowCount - 1));
            Assert.Equal("central", result["region"].GetValue(0));
        }
        finally { DaskFrame.DefaultSplitEvery = old; }
    }

    private static double GroupedDataFrame_Var(double[] v)
    {
        var m = v.Average();
        return v.Sum(x => (x - m) * (x - m)) / (v.Length - 1);
    }

    private static double GroupedDataFrame_Std(double[] v) => Math.Sqrt(GroupedDataFrame_Var(v));

    [Fact]
    public void Multi_key_groupby_and_shortcuts()
    {
        var result = DaskFrame.FromFrame(Sales, 4).GroupBy("region", "product").Sum("qty").Compute();
        Assert.Equal(Sales.GroupBy("region", "product").GroupCount, result.RowCount);
        Assert.Equal(Sales.Numeric("qty").Sum(), result.Numeric("qty").Sum());
        var sizes = DaskFrame.FromFrame(Sales, 4).GroupBy("product").Size().Compute();
        Assert.Equal(Sales.RowCount, sizes.Numeric("size").Sum());
    }

    [Fact]
    public void Scalar_aggregates_and_describe()
    {
        var df = DaskFrame.FromFrame(Sales, 6);
        var price = Sales.Numeric("price");
        Assert.True(Math.Abs(price.Sum() - df.Sum("price").Compute()) < 1e-6);
        Assert.True(Math.Abs(price.Mean() - df.Mean("price").Compute()) < 1e-9);
        Assert.True(Math.Abs(price.Std() - df.Std("price").Compute()) < 1e-9);
        Assert.Equal(price.Min(), df.Min("price").Compute());
        Assert.Equal(price.Max(), df.Max("price").Compute());

        var describe = df.Describe();
        Assert.Equal(new[] { "statistic", "id", "price", "qty" }, describe.ColumnNames);
        Assert.Equal(price.Count, describe.Numeric("price")[0]);

        // A global aggregate over nothing still returns a row.
        var none = df.Filter(Col("price") < 0).Aggregate(new Aggregation("n", "count", "price"), new Aggregation("m", "mean", "price")).Compute();
        Assert.Equal(1, none.RowCount);
        Assert.Equal(0, none.Numeric("n")[0]);
        Assert.True(double.IsNaN(none.Numeric("m")[0]));
    }

    /// <summary>Every kind, with the right side in one or several partitions, under each broadcast
    /// setting - so the shuffle path stays covered now that small joins broadcast by size.</summary>
    public static TheoryData<JoinKind, int, bool?> JoinCases()
    {
        var data = new TheoryData<JoinKind, int, bool?>();
        foreach (var kind in new[] { JoinKind.Inner, JoinKind.Left, JoinKind.Right, JoinKind.Outer })
            foreach (var partitions in new[] { 1, 3 })
                foreach (var broadcast in new bool?[] { null, false, true })
                    if (broadcast != true || kind is JoinKind.Inner or JoinKind.Left)
                        data.Add(kind, partitions, broadcast);
        return data;
    }

    [Theory]
    [MemberData(nameof(JoinCases))]
    public void Joins_match_graviframe(JoinKind kind, int rightPartitions, bool? broadcast)
    {
        var products = new DataFrame([
            new TextSeries("product", Enumerable.Range(0, 15).Select(i => (string?)$"p{i:D2}").ToArray()),
            new NumericSeries("weight", Enumerable.Range(0, 15).Select(i => i * 1.5).ToArray()),
            new NumericSeries("price", Enumerable.Range(0, 15).Select(i => 100.0 + i).ToArray()),
        ]);
        var left = Sales.Rows(0, 300);
        var expected = Joins.Merge(left, products, "product", "product", kind);
        var joined = DaskFrame.FromFrame(left, 5).Join(DaskFrame.FromFrame(products, rightPartitions), "product", kind, broadcast: broadcast);
        var explain = joined.Explain();
        if (broadcast == false) Assert.Contains("hash shuffle", explain);
        if (broadcast == true || (broadcast is null && kind is JoinKind.Inner or JoinKind.Left)) Assert.Contains("broadcast right", explain);
        var actual = joined.Compute();
        Assert.Equal(expected.ColumnNames, actual.ColumnNames);
        Assert.Equal(SalesData.Rows(expected, sorted: true), SalesData.Rows(actual, sorted: true));
    }

    [Fact]
    public void Sort_gives_a_globally_ordered_frame()
    {
        var df = DaskFrame.FromFrame(Sales, 6);
        var ascending = df.SortBy("price").Compute();
        var prices = ascending.Numeric("price").Values;
        var present = prices.TakeWhile(p => !double.IsNaN(p)).ToArray();
        Assert.Equal(present.Order(), present);
        Assert.All(prices.Skip(present.Length), p => Assert.True(double.IsNaN(p)));
        Assert.Equal(Sales.RowCount, ascending.RowCount);

        var desc = df.SortBy([("region", true), ("price", false)], partitions: 3).Compute();
        var regions = Enumerable.Range(0, desc.RowCount).Select(i => desc.Text("region")[i]).ToArray();
        var nonNull = regions.Where(r => r is not null).ToArray();
        Assert.Equal(nonNull.Order(StringComparer.Ordinal), nonNull);
        Assert.All(regions.Skip(nonNull.Length), Assert.Null);
        for (var i = 1; i < desc.RowCount; i++)
            if (regions[i] == regions[i - 1] && regions[i] is not null
                && !double.IsNaN(desc.Numeric("price")[i]) && !double.IsNaN(desc.Numeric("price")[i - 1]))
                Assert.True(desc.Numeric("price")[i] <= desc.Numeric("price")[i - 1]);

        var top = df.SortBy("qty", ascending: false).Limit(3).Compute();
        Assert.Equal(3, top.RowCount);
        Assert.Equal(Sales.Numeric("qty").Max(), top.Numeric("qty")[0]);
    }

    [Fact]
    public void Repartition_limit_head_and_value_counts()
    {
        var df = DaskFrame.FromFrame(Sales, 4);
        Assert.Equal(SalesData.Rows(Sales), SalesData.Rows(df.Repartition(11).Compute()));
        Assert.Equal(11, df.Repartition(11).PartitionCount);
        Assert.Equal(SalesData.Rows(Sales), SalesData.Rows(df.Repartition(2).Compute()));
        Assert.Equal(SalesData.Rows(Sales.Head(7)), SalesData.Rows(df.Limit(7).Compute()));
        Assert.Equal(SalesData.Rows(Sales.Head(900)), SalesData.Rows(df.Head(900)));

        var counts = df.ValueCounts("product").Compute();
        Assert.Equal(12, counts.RowCount);
        var c = counts.Numeric("count").Values;
        Assert.Equal(c.OrderDescending(), c);
        Assert.Equal(Sales.RowCount, c.Sum());
    }

    [Fact]
    public void Concat_and_drop_missing()
    {
        var a = DaskFrame.FromFrame(Sales.Rows(0, 100), 2);
        var b = DaskFrame.FromFrame(Sales.Rows(100, 50), 3);
        var both = DaskFrame.Concat(a, b);
        Assert.Equal(5, both.PartitionCount);
        Assert.Equal(SalesData.Rows(Sales.Head(150)), SalesData.Rows(both.Compute()));
        Assert.Equal(Sales.DropMissing().RowCount, DaskFrame.FromFrame(Sales, 3).DropMissing().Count().Compute());
        Assert.Equal(Sales.RowCount - Sales["price"].MissingCount,
            DaskFrame.FromFrame(Sales, 3).DropMissing("price").Count().Compute());
    }

    [Fact]
    public void Optimised_and_unoptimised_plans_agree()
    {
        var products = DaskFrame.FromFrame(new DataFrame([
            new TextSeries("product", Enumerable.Range(0, 12).Select(i => (string?)$"p{i:D2}").ToArray()),
            new TextSeries("category", Enumerable.Range(0, 12).Select(i => (string?)(i % 2 == 0 ? "even" : "odd")).ToArray()),
        ]), 2);

        DaskFrame Pipeline() => DaskFrame.FromFrame(Sales, 6)
            .WithColumn("revenue", Col("price") * Col("qty"))
            .Join(products, "product", JoinKind.Left)
            .Filter((Col("region").Eq("east") | Col("region").Eq("west")) & (Col("category").Eq("odd")) & (Col("qty") > 3))
            .Rename("revenue", "rev")
            .Filter(Col("product").Ne("p03"))
            .GroupBy("region", "category").Agg(("rev", "sum"), ("qty", "mean"))
            .Filter(Col("region").Ne("west") | Col("qty_mean") > 0);

        var optimised = Pipeline().Compute();
        DaskFrame.OptimizePlans = false;
        try
        {
            var plain = Pipeline().Compute();
            Assert.Equal(SalesData.Rows(plain), SalesData.Rows(optimised));
        }
        finally { DaskFrame.OptimizePlans = true; }

        var explain = Pipeline().Explain();
        // qty > 3 and the region filter reached the left scan; category stayed above the left join.
        Assert.Contains("FromFrame", explain);
        Assert.Matches("FromFrame.*filter=.*qty", explain);
        var planLines = explain.Split('\n');
        var categoryLine = Array.FindIndex(planLines, l => l.Contains("Filter (category = 'odd')"));
        var joinLine = Array.FindIndex(planLines, l => l.TrimStart().StartsWith("Join Left"));
        Assert.True(categoryLine >= 0 && categoryLine < joinLine, explain);
        Assert.DoesNotContain("date", planLines.First(l => l.Contains("2000 rows")));
    }

    [Fact]
    public void Inner_join_key_filters_reach_both_sides()
    {
        var left = DaskFrame.FromFrame(Sales, 3);
        var right = DaskFrame.FromFrame(new DataFrame([
            new TextSeries("product", ["p01", "p02"]), new NumericSeries("w", [1, 2])]), 1);
        var explain = left.Join(right, "product").Filter(Col("product").Eq("p01")).Explain();
        Assert.Equal(2, explain.Split('\n').Count(l => l.Contains("filter=(product = 'p01')")));
    }

    [Fact]
    public void Projection_pushdown_reads_only_needed_columns()
    {
        var df = DaskFrame.FromFrame(Sales, 3).Filter(Col("qty") > 10).GroupBy("region").Mean("price");
        var explain = df.Explain();
        Assert.Contains("columns=[region, price, qty]", explain);
        Assert.Contains("filter=(qty > 10)", explain);
        Assert.DoesNotContain("Filter (qty", explain.Replace("filter=(qty", ""));
    }

    [Fact]
    public void Row_wise_steps_fuse_into_one_task_per_partition()
    {
        var df = DaskFrame.FromFrame(Sales, 8)
            .Filter(Col("qty") > 2).WithColumn("r", Col("price") * 2).Select("id", "r").Rename("r", "double_price");
        var graph = df.TaskGraphFor();
        Assert.Equal(8, graph.Count);
        Assert.True(graph.IsRemotable);
    }

    [Fact]
    public void MapPartitions_is_opaque_but_works()
    {
        var df = DaskFrame.FromFrame(Sales, 4).MapPartitions(f => f.Head(1), label: "first-row");
        Assert.False(df.IsRemotable);
        Assert.Equal(4, df.Compute().RowCount);
        Assert.Contains("opaque", df.Explain());
    }

    [Fact]
    public void Unknown_columns_fail_at_build_time()
    {
        var df = DaskFrame.FromFrame(Sales, 2);
        Assert.Throws<KeyNotFoundException>(() => df.Filter(Col("nope") > 1));
        Assert.Throws<KeyNotFoundException>(() => df.GroupBy("nope"));
        Assert.Throws<ArgumentException>(() => df.GroupBy("region").Agg(new Aggregation("x", "median", "price")));
    }
}
