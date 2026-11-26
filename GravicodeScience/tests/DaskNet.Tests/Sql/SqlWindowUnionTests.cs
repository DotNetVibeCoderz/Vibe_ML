using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.DaskNet.Sql;
using Gravicode.Science.GraviFrame;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet;

/// <summary>Aggregate DISTINCT, UNION and window functions, checked against LINQ over the raw rows.</summary>
public sealed class SqlWindowUnionTests
{
    private static readonly DataFrame Sales = SalesData.Make(1_500, seed: 11);

    private sealed record Row(double Id, string? Region, string Product, double? Price, double Qty);

    private static readonly Row[] Rows = Enumerable.Range(0, Sales.RowCount).Select(i => new Row(
        ((NumericSeries)Sales["id"])[i],
        (string?)Sales["region"].GetValue(i),
        (string)Sales["product"].GetValue(i)!,
        double.IsNaN(((NumericSeries)Sales["price"])[i]) ? null : ((NumericSeries)Sales["price"])[i],
        ((NumericSeries)Sales["qty"])[i])).ToArray();

    private static SqlContext Context(int partitions = 6) => new SqlContext().Register("sales", DaskFrame.FromFrame(Sales, partitions));

    private static double[] Numbers(DataFrame f, string column) => ((NumericSeries)f[column]).Values.ToArray();

    /// <summary>A result column keyed by another (usually id), so row order does not matter.</summary>
    private static Dictionary<string, object?> By(DataFrame f, string key, string column)
        => Enumerable.Range(0, f.RowCount).ToDictionary(i => f[key].GetValue(i)?.ToString() ?? "<null>", i => f[column].GetValue(i));

    // ------------------------------------------------------------------ aggregate DISTINCT

    [Fact]
    public void Count_distinct_per_group_beside_ordinary_aggregates()
    {
        var r = Context().Execute("""
            SELECT region, COUNT(*) AS n, COUNT(DISTINCT product) AS products, COUNT(DISTINCT price) AS prices,
                   SUM(DISTINCT qty) AS qtys, SUM(qty) AS q
            FROM sales GROUP BY region
            """);
        var expected = Rows.GroupBy(x => x.Region).ToDictionary(g => g.Key ?? "<null>");
        Assert.Equal(expected.Count, r.RowCount);   // the missing-region group survives the join
        var n = By(r, "region", "n");
        var products = By(r, "region", "products");
        var prices = By(r, "region", "prices");
        var qtys = By(r, "region", "qtys");
        var q = By(r, "region", "q");
        foreach (var (key, g) in expected)
        {
            Assert.Equal((double)g.Count(), n[key]);
            Assert.Equal((double)g.Select(x => x.Product).Distinct().Count(), products[key]);
            Assert.Equal((double)g.Where(x => x.Price.HasValue).Select(x => x.Price).Distinct().Count(), prices[key]);   // NULL is not a value
            Assert.Equal(g.Select(x => x.Qty).Distinct().Sum(), qtys[key]);
            Assert.Equal(g.Sum(x => x.Qty), q[key]);
        }
    }

    [Fact]
    public void Count_distinct_without_group_by_and_with_several_keys()
    {
        var r = Context().Execute("SELECT COUNT(DISTINCT region) AS regions, COUNT(DISTINCT product) AS products, COUNT(*) AS n FROM sales");
        Assert.Equal(1, r.RowCount);
        Assert.Equal(Rows.Where(x => x.Region is not null).Select(x => x.Region).Distinct().Count(), Numbers(r, "regions")[0]);
        Assert.Equal(12, Numbers(r, "products")[0]);
        Assert.Equal(Rows.Length, Numbers(r, "n")[0]);

        // Two group keys, the multi-key join path, with an aggregate DISTINCT in HAVING.
        var two = Context().Execute("""
            SELECT region, product, COUNT(DISTINCT qty) AS k FROM sales
            WHERE region IS NOT NULL GROUP BY region, product HAVING COUNT(DISTINCT qty) >= 10
            """);
        var expected = Rows.Where(x => x.Region is not null).GroupBy(x => (x.Region, x.Product))
            .Select(g => (g.Key, K: g.Select(x => x.Qty).Distinct().Count())).Where(g => g.K >= 10)
            .ToDictionary(g => $"{g.Key.Region}/{g.Key.Product}", g => (double)g.K);
        Assert.NotEmpty(expected);
        var actual = Enumerable.Range(0, two.RowCount).ToDictionary(i => $"{two["region"].GetValue(i)}/{two["product"].GetValue(i)}", i => Numbers(two, "k")[i]);
        Assert.Equal(expected.OrderBy(p => p.Key), actual.OrderBy(p => p.Key));
    }

    /// <summary>Pruning once cut the scan to zero columns when no column was named, and a frame
    /// with no columns has no rows: COUNT(*) alone returned 0.</summary>
    [Fact]
    public void Queries_naming_no_column_still_count_rows()
    {
        Assert.Equal(Rows.Length, Numbers(Context().Execute("SELECT COUNT(*) AS n FROM sales"), "n")[0]);
        Assert.Equal(Rows.Length, Context().Execute("SELECT 1 AS one FROM sales").RowCount);
        var size = DaskFrame.FromFrame(Sales, 3).Aggregate(new Aggregation("n", "size", null)).Compute();
        Assert.Equal([(double)Rows.Length], Numbers(size, "n"));
    }

    // ------------------------------------------------------------------ UNION

    [Fact]
    public void Union_all_keeps_duplicates_and_union_removes_them()
    {
        const string north = "SELECT product, qty FROM sales WHERE region = 'north'";
        const string cheap = "SELECT product, qty FROM sales WHERE price < 20";
        var all = Context().Execute($"{north} UNION ALL {cheap}");
        var nRows = Rows.Where(x => x.Region == "north").Select(x => (x.Product, x.Qty)).ToList();
        var cRows = Rows.Where(x => x.Price < 20).Select(x => (x.Product, x.Qty)).ToList();
        Assert.Equal(nRows.Count + cRows.Count, all.RowCount);

        var distinct = Context().Execute($"{north} UNION {cheap} ORDER BY product, qty DESC LIMIT 25");
        var expected = nRows.Concat(cRows).Distinct().OrderBy(x => x.Product, StringComparer.Ordinal).ThenByDescending(x => x.Qty).Take(25).ToList();
        Assert.Equal(expected.Select(x => x.Product), Enumerable.Range(0, distinct.RowCount).Select(i => (string)distinct["product"].GetValue(i)!));
        Assert.Equal(expected.Select(x => x.Qty), Numbers(distinct, "qty"));
    }

    [Fact]
    public void Union_matches_columns_by_position_and_takes_the_first_names()
    {
        var r = Context().Execute("""
            SELECT region AS name, COUNT(*) AS n FROM sales WHERE region IS NOT NULL GROUP BY region
            UNION ALL
            SELECT product, COUNT(*) FROM sales GROUP BY product
            ORDER BY 2 DESC, name
            """);
        Assert.Equal(["name", "n"], r.ColumnNames);
        Assert.Equal(5 + 12, r.RowCount);
        var counts = Numbers(r, "n");
        Assert.True(counts.Zip(counts.Skip(1)).All(p => p.First >= p.Second));
        Assert.Equal((double)Rows.Count(x => x.Region == "south"), By(r, "name", "n")["south"]);

        // Left to right: UNION ALL then UNION deduplicates everything so far.
        var folded = Context().Execute("""
            SELECT product FROM sales UNION ALL SELECT product FROM sales UNION SELECT product FROM sales WHERE qty > 100
            """);
        Assert.Equal(12, folded.RowCount);
    }

    // ------------------------------------------------------------------ window functions

    [Fact]
    public void Ranking_functions_follow_sql_ties()
    {
        var r = Context().Execute("""
            SELECT id,
                   ROW_NUMBER() OVER (PARTITION BY region ORDER BY qty DESC, id) AS rn,
                   RANK() OVER (PARTITION BY region ORDER BY qty DESC) AS rk,
                   DENSE_RANK() OVER (PARTITION BY region ORDER BY qty DESC) AS drk,
                   ROW_NUMBER() OVER (ORDER BY id DESC) AS everywhere
            FROM sales
            """);
        Assert.Equal(Rows.Length, r.RowCount);
        var rn = By(r, "id", "rn");
        var rk = By(r, "id", "rk");
        var drk = By(r, "id", "drk");
        var everywhere = By(r, "id", "everywhere");
        foreach (var g in Rows.GroupBy(x => x.Region))
        {
            var ordered = g.OrderByDescending(x => x.Qty).ThenBy(x => x.Id).ToList();
            for (var i = 0; i < ordered.Count; i++)
            {
                var x = ordered[i];
                var id = x.Id.ToString();
                Assert.Equal((double)(i + 1), rn[id]);
                Assert.Equal((double)(1 + g.Count(y => y.Qty > x.Qty)), rk[id]);
                Assert.Equal((double)(1 + g.Where(y => y.Qty > x.Qty).Select(y => y.Qty).Distinct().Count()), drk[id]);
            }
        }
        Assert.Equal((double)Rows.Length, everywhere["0"]);
        Assert.Equal(1.0, everywhere[(Rows.Length - 1).ToString()]);
    }

    [Fact]
    public void Aggregates_over_windows_run_with_ties_or_cover_the_partition()
    {
        var r = Context().Execute("""
            SELECT id, qty,
                   SUM(qty) OVER (PARTITION BY region) AS total,
                   SUM(qty) OVER (PARTITION BY region ORDER BY id) AS running,
                   SUM(qty) OVER (PARTITION BY region ORDER BY qty) AS by_qty,
                   COUNT(price) OVER (PARTITION BY product ORDER BY id) AS priced,
                   AVG(price) OVER (PARTITION BY product) AS mean_price,
                   qty * 1.0 / SUM(qty) OVER (PARTITION BY region) AS share
            FROM sales
            """);
        var total = By(r, "id", "total");
        var running = By(r, "id", "running");
        var byQty = By(r, "id", "by_qty");
        var priced = By(r, "id", "priced");
        var meanPrice = By(r, "id", "mean_price");
        var share = By(r, "id", "share");
        foreach (var x in Rows)
        {
            var id = x.Id.ToString();
            var region = Rows.Where(y => y.Region == x.Region).ToList();
            var product = Rows.Where(y => y.Product == x.Product).ToList();
            Assert.Equal(region.Sum(y => y.Qty), total[id]);
            Assert.Equal(region.Where(y => y.Id <= x.Id).Sum(y => y.Qty), running[id]);
            Assert.Equal(region.Where(y => y.Qty <= x.Qty).Sum(y => y.Qty), byQty[id]);   // ties share the running value
            Assert.Equal((double)product.Count(y => y.Id <= x.Id && y.Price.HasValue), priced[id]);
            Assert.True(Math.Abs(product.Where(y => y.Price.HasValue).Average(y => y.Price!.Value) - (double)meanPrice[id]!) < 1e-9);
            Assert.True(Math.Abs(x.Qty / region.Sum(y => y.Qty) - (double)share[id]!) < 1e-12);
        }
    }

    [Fact]
    public void Lag_and_lead_look_along_the_window_order()
    {
        var r = Context().Execute("""
            SELECT id, LAG(qty) OVER (PARTITION BY product ORDER BY id) AS prev,
                   LEAD(region, 2) OVER (PARTITION BY product ORDER BY id) AS later
            FROM sales
            """);
        var prev = By(r, "id", "prev");
        var later = By(r, "id", "later");
        foreach (var g in Rows.GroupBy(x => x.Product))
        {
            var ordered = g.OrderBy(x => x.Id).ToList();
            for (var i = 0; i < ordered.Count; i++)
            {
                var id = ordered[i].Id.ToString();
                if (i == 0) Assert.True(prev[id] is null || prev[id] is double.NaN, "nothing before the first row");
                else Assert.Equal(ordered[i - 1].Qty, prev[id]);
                Assert.Equal(i + 2 < ordered.Count ? ordered[i + 2].Region : null, (string?)later[id]);
            }
        }
    }

    [Fact]
    public void Windows_over_a_grouped_query_rank_the_groups()
    {
        var r = Context().Execute("""
            SELECT region, SUM(qty) AS q, RANK() OVER (ORDER BY SUM(qty) DESC) AS place,
                   SUM(qty) * 100.0 / SUM(SUM(qty)) OVER () AS pct
            FROM sales WHERE region IS NOT NULL GROUP BY region ORDER BY place
            """);
        var expected = Rows.Where(x => x.Region is not null).GroupBy(x => x.Region!)
            .Select(g => (Region: g.Key, Q: g.Sum(x => x.Qty))).OrderByDescending(g => g.Q).ToList();
        var grand = expected.Sum(g => g.Q);
        Assert.Equal(expected.Select(g => g.Region), Enumerable.Range(0, r.RowCount).Select(i => (string)r["region"].GetValue(i)!));
        Assert.Equal(Enumerable.Range(1, expected.Count).Select(i => (double)i), Numbers(r, "place"));
        for (var i = 0; i < expected.Count; i++)
            Assert.True(Math.Abs(expected[i].Q * 100.0 / grand - Numbers(r, "pct")[i]) < 1e-9);
    }

    [Fact]
    public void Window_api_and_filters_are_not_pushed_below_it()
    {
        // ROW_NUMBER then filter: the filter must see every row's number, so it stays above.
        var frame = DaskFrame.FromFrame(Sales, 4)
            .Window(["region"], [("id", true)], new WindowFunction("rn", "row_number"))
            .Filter(Expr.Col("rn") <= 3 & Expr.Col("product").Eq(Expr.Lit("p01")));
        var result = frame.Compute();
        var expected = Rows.GroupBy(x => x.Region).SelectMany(g => g.OrderBy(x => x.Id).Take(3)).Where(x => x.Product == "p01").Select(x => x.Id).Order();
        Assert.Equal(expected, Numbers(result, "id").Order());
        Assert.Throws<ArgumentException>(() => DaskFrame.FromFrame(Sales, 2).Window([], [], new WindowFunction("r", "rank")));
        Assert.Throws<ArgumentException>(() => DaskFrame.FromFrame(Sales, 2).Window([], [], new WindowFunction("x", "median", "qty")));
    }

    [Fact]
    public async Task Everything_new_runs_on_a_cluster()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 2);
        var ctx = Context(4);
        foreach (var sql in new[]
        {
            "SELECT region, COUNT(DISTINCT product) AS k FROM sales GROUP BY region",
            "SELECT product FROM sales WHERE qty > 15 UNION SELECT product FROM sales WHERE price < 10",
            "SELECT id, RANK() OVER (PARTITION BY region ORDER BY qty) AS r, SUM(qty) OVER (PARTITION BY product ORDER BY id) AS s FROM sales",
        })
        {
            var local = ctx.Execute(sql);
            var remote = ctx.Execute(sql, cluster.Client);
            Assert.Equal(SalesData.Rows(local, sorted: true), SalesData.Rows(remote, sorted: true));
        }
    }

    [Theory]
    [InlineData("SELECT id FROM sales WHERE ROW_NUMBER() OVER (ORDER BY id) = 1", "only allowed in the select list")]
    [InlineData("SELECT RANK() OVER (PARTITION BY region) FROM sales", "needs ORDER BY")]
    [InlineData("SELECT SUM(qty) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM sales", "window frames")]
    [InlineData("SELECT STDDEV(qty) OVER () FROM sales", "not supported as a window function")]
    [InlineData("SELECT LAG(qty, 1, 0) OVER (ORDER BY id) FROM sales", "COALESCE")]
    [InlineData("SELECT id FROM sales ORDER BY id UNION SELECT id FROM sales", "last query of a UNION")]
    [InlineData("SELECT id, qty FROM sales UNION SELECT id FROM sales", "same number of columns")]
    [InlineData("SELECT id FROM sales UNION SELECT id FROM sales ORDER BY qty", "output column or position")]
    public void Errors_explain_themselves(string sql, string fragment)
    {
        var e = Assert.Throws<SqlException>(() => Context().Sql(sql));
        Assert.Contains(fragment, e.Message);
    }
}
