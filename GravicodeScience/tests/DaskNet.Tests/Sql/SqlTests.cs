using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Sql;
using Gravicode.Science.GraviFrame;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet;

/// <summary>SQL results are checked against LINQ over the raw rows, not against DaskFrame, so a
/// shared mistake in the frame layer cannot make both sides agree.</summary>
public sealed class SqlTests
{
    private static readonly DataFrame Sales = SalesData.Make(2_000);

    private static readonly DataFrame Products = new([
        // p10 and p11 have no product row, so a left join leaves their weight missing.
        new TextSeries("product", Enumerable.Range(0, 10).Select(i => (string?)$"p{i:D2}").ToArray()),
        new NumericSeries("weight", Enumerable.Range(0, 10).Select(i => i * 0.5).ToArray()),
        new TextSeries("family", Enumerable.Range(0, 10).Select(i => (string?)(i % 2 == 0 ? "even" : "odd")).ToArray()),
    ]);

    private sealed record Row(double Id, string? Region, string Product, double? Price, double Qty, DateTime? Date, bool? Promo);

    private static readonly Row[] Rows = Enumerable.Range(0, Sales.RowCount).Select(i => new Row(
        ((NumericSeries)Sales["id"])[i],
        (string?)Sales["region"].GetValue(i),
        (string)Sales["product"].GetValue(i)!,
        double.IsNaN(((NumericSeries)Sales["price"])[i]) ? null : ((NumericSeries)Sales["price"])[i],
        ((NumericSeries)Sales["qty"])[i],
        (DateTime?)Sales["date"].GetValue(i),
        (bool?)Sales["promo"].GetValue(i))).ToArray();

    private static SqlContext Context() => new SqlContext()
        .Register("sales", DaskFrame.FromFrame(Sales, 5))
        .Register("products", Products);

    private static double[] Numbers(DataFrame f, string column) => ((NumericSeries)f[column]).Values.ToArray();
    private static string?[] Texts(DataFrame f, string column) => Enumerable.Range(0, f.RowCount).Select(i => f[column].GetValue(i)?.ToString()).ToArray();

    [Fact]
    public void Where_and_computed_columns()
    {
        var r = Context().Execute("SELECT id, price * qty AS revenue FROM sales WHERE region = 'north' AND price > 50 ORDER BY id");
        var expected = Rows.Where(x => x.Region == "north" && x.Price > 50).OrderBy(x => x.Id).ToArray();
        Assert.Equal(["id", "revenue"], r.ColumnNames);
        Assert.Equal(expected.Select(x => x.Id), Numbers(r, "id"));
        Assert.Equal(expected.Select(x => x.Price!.Value * x.Qty), Numbers(r, "revenue"));
    }

    [Fact]
    public void Group_by_with_having_and_order()
    {
        var r = Context().Execute("""
            SELECT region, COUNT(*) AS n, SUM(qty) AS q, AVG(price) AS p, MAX(price) - MIN(price) AS spread
            FROM sales
            WHERE region IS NOT NULL AND promo = TRUE
            GROUP BY region
            HAVING COUNT(*) > 100
            ORDER BY q DESC
            """);
        var expected = Rows.Where(x => x.Region is not null && x.Promo == true).GroupBy(x => x.Region)
            .Where(g => g.Count() > 100)
            .Select(g => (Region: g.Key, N: g.Count(), Q: g.Sum(x => x.Qty), P: g.Where(x => x.Price.HasValue).Average(x => x.Price!.Value),
                Spread: g.Max(x => x.Price ?? double.MinValue) - g.Where(x => x.Price.HasValue).Min(x => x.Price!.Value)))
            .OrderByDescending(g => g.Q).ToArray();
        Assert.NotEmpty(expected);
        Assert.Equal(expected.Select(g => g.Region), Texts(r, "region"));
        Assert.Equal(expected.Select(g => (double)g.N), Numbers(r, "n"));
        Assert.Equal(expected.Select(g => g.Q), Numbers(r, "q"));
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i].P - Numbers(r, "p")[i]) < 1e-9, "AVG skips missing prices");
            Assert.True(Math.Abs(expected[i].Spread - Numbers(r, "spread")[i]) < 1e-9);
        }
    }

    [Fact]
    public void Global_aggregates_count_missing_values_the_sql_way()
    {
        var r = Context().Execute("SELECT COUNT(*), COUNT(price) AS priced, STDDEV(qty) AS sd, VAR_POP(qty) AS vp FROM sales");
        Assert.Equal(1, r.RowCount);
        Assert.Equal(Rows.Length, Numbers(r, "COUNT(*)")[0]);
        Assert.Equal(Rows.Count(x => x.Price.HasValue), Numbers(r, "priced")[0]);
        var mean = Rows.Average(x => x.Qty);
        var ss = Rows.Sum(x => (x.Qty - mean) * (x.Qty - mean));
        Assert.True(Math.Abs(Math.Sqrt(ss / (Rows.Length - 1)) - Numbers(r, "sd")[0]) < 1e-12);
        Assert.True(Math.Abs(ss / Rows.Length - Numbers(r, "vp")[0]) < 1e-12);
    }

    [Fact]
    public void Group_by_an_expression_by_position()
    {
        var r = Context().Execute("""
            SELECT CASE WHEN qty >= 10 THEN 'big' ELSE 'small' END AS size, COUNT(*) AS n, SUM(qty) / COUNT(*) AS avg_qty
            FROM sales GROUP BY 1 ORDER BY size
            """);
        Assert.Equal(new string?[] { "big", "small" }, Texts(r, "size"));
        Assert.Equal([Rows.Count(x => x.Qty >= 10), Rows.Count(x => x.Qty < 10)], Numbers(r, "n").Select(n => (int)n));
        Assert.True(Math.Abs(Rows.Where(x => x.Qty >= 10).Average(x => x.Qty) - Numbers(r, "avg_qty")[0]) < 1e-12);
    }

    [Fact]
    public void Inner_join_with_aliases_and_a_residual_condition()
    {
        var r = Context().Execute("""
            SELECT s.id, p.weight, s.qty * p.weight AS load
            FROM sales s JOIN products p ON s.product = p.product AND p.weight > s.qty / 10
            WHERE p.family = 'odd'
            ORDER BY s.id
            """);
        var weights = Enumerable.Range(0, 10).ToDictionary(i => $"p{i:D2}", i => i * 0.5);
        var expected = Rows.Where(x => weights.TryGetValue(x.Product, out var w) && w > x.Qty / 10 && (int)(w * 2) % 2 == 1)
            .OrderBy(x => x.Id).ToArray();
        Assert.Equal(["id", "weight", "load"], r.ColumnNames);
        Assert.Equal(expected.Select(x => x.Id), Numbers(r, "id"));
        Assert.Equal(expected.Select(x => x.Qty * weights[x.Product]), Numbers(r, "load"));
    }

    [Fact]
    public void Left_join_leaves_the_right_key_missing()
    {
        var r = Context().Execute("""
            SELECT s.product, p.product, COUNT(*) AS n
            FROM sales AS s LEFT JOIN products AS p ON s.product = p.product
            GROUP BY s.product, p.product
            ORDER BY 1
            """);
        Assert.Equal(["s.product", "p.product", "n"], r.ColumnNames);
        var left = Texts(r, "s.product");
        var right = Texts(r, "p.product");
        Assert.Equal(12, r.RowCount);
        for (var i = 0; i < 12; i++) Assert.Equal(i < 10 ? left[i] : null, right[i]);
        Assert.Equal(Rows.GroupBy(x => x.Product).OrderBy(g => g.Key).Select(g => (double)g.Count()), Numbers(r, "n"));
    }

    [Fact]
    public void Joins_on_two_keys_and_self_joins()
    {
        var ctx = Context();
        var r = ctx.Execute("""
            SELECT a.id, b.id AS other
            FROM sales a JOIN sales b ON a.region = b.region AND a.product = b.product
            WHERE a.id < 40 AND b.id < 40 AND a.id < b.id
            ORDER BY a.id, other
            """);
        var small = Rows.Where(x => x.Id < 40).ToArray();
        var expected = (from a in small from b in small
                        where a.Region is not null && a.Region == b.Region && a.Product == b.Product && a.Id < b.Id
                        orderby a.Id, b.Id select (a.Id, b.Id)).ToArray();
        Assert.Equal(expected.Select(e => e.Item1), Numbers(r, "id"));
        Assert.Equal(expected.Select(e => e.Item2), Numbers(r, "other"));
    }

    [Fact]
    public void Subqueries_distinct_and_limit()
    {
        var r = Context().Execute("""
            SELECT DISTINCT region FROM (
                SELECT region, qty FROM sales WHERE qty > 15 AND region IN ('north', 'east', 'west')
            ) AS big
            ORDER BY region DESC LIMIT 2
            """);
        Assert.Equal(new string?[] { "west", "north" }, Texts(r, "region"));
    }

    [Fact]
    public void Order_by_an_expression_that_is_not_selected()
    {
        var r = Context().Execute("SELECT id FROM sales WHERE price IS NOT NULL ORDER BY price * qty DESC, id LIMIT 7");
        Assert.Equal(["id"], r.ColumnNames);
        Assert.Equal(Rows.Where(x => x.Price.HasValue).OrderByDescending(x => x.Price!.Value * x.Qty).ThenBy(x => x.Id).Take(7).Select(x => x.Id),
            Numbers(r, "id"));
    }

    [Fact]
    public void Scalar_functions_and_three_valued_logic()
    {
        var r = Context().Execute("""
            SELECT id, UPPER(region) || '-' || product AS label, ROUND(price, 1) AS p, COALESCE(region, 'none') AS reg,
                   CAST(qty / 3 AS INT) AS third, MONTH(date) AS m
            FROM sales
            WHERE NOT (promo = TRUE) AND product LIKE 'p0%' AND qty BETWEEN 3 AND 6 AND id NOT IN (1, 2, 3)
            ORDER BY id
            """);
        // NOT (promo = TRUE) is unknown, so dropped, where promo is missing.
        var expected = Rows.Where(x => x.Promo == false && x.Product.StartsWith("p0") && x.Qty is >= 3 and <= 6 && x.Id is not (1 or 2 or 3))
            .OrderBy(x => x.Id).ToArray();
        Assert.Equal(expected.Select(x => x.Id), Numbers(r, "id"));
        Assert.Equal(expected.Select(x => x.Region is null ? null : $"{x.Region.ToUpperInvariant()}-{x.Product}"), Texts(r, "label"));
        Assert.Equal(expected.Select(x => x.Region ?? "none"), Texts(r, "reg"));
        Assert.Equal(expected.Select(x => Math.Truncate(x.Qty / 3)), Numbers(r, "third"));
        Assert.Equal(expected.Select(x => x.Date?.Month ?? double.NaN), Numbers(r, "m"));
        Assert.Equal(expected.Select(x => x.Price.HasValue ? Math.Round(x.Price.Value, 1, MidpointRounding.AwayFromZero) : double.NaN), Numbers(r, "p"));
    }

    [Fact]
    public void Select_star_qualifies_only_clashing_names()
    {
        var r = Context().Execute("SELECT * FROM sales s JOIN products p ON s.product = p.product LIMIT 3");
        Assert.Equal(["id", "region", "s.product", "price", "qty", "date", "promo", "p.product", "weight", "family"], r.ColumnNames);
        var star = Context().Execute("SELECT p.* FROM products p ORDER BY weight DESC LIMIT 1");
        Assert.Equal(["product", "weight", "family"], star.ColumnNames);
        Assert.Equal("p09", star["product"].GetValue(0));
    }

    [Fact]
    public void Queries_compile_to_optimisable_remotable_plans()
    {
        var ctx = new SqlContext().Register("sales", DaskFrame.FromFrame(Sales, 4)).Register("products", Products);
        var frame = ctx.Sql("SELECT s.region, SUM(p.weight) AS w FROM sales s JOIN products p ON s.product = p.product WHERE s.qty > 5 GROUP BY s.region");
        Assert.True(frame.IsRemotable);
        var plan = ctx.Explain("SELECT s.region, SUM(p.weight) AS w FROM sales s JOIN products p ON s.product = p.product WHERE s.qty > 5 GROUP BY s.region");
        var lines = plan.Split('\n');
        var join = Array.FindIndex(lines, l => l.Contains("Join"));
        var filter = Array.FindIndex(lines, l => l.Contains("qty") && l.Contains('>'));
        Assert.True(join >= 0 && filter > join, "the WHERE should be pushed below the join:\n" + plan);
    }

    [Theory]
    [InlineData("SELECT nope FROM sales", "Unknown column 'nope'")]
    [InlineData("SELECT * FROM missing", "Unknown table 'missing'")]
    [InlineData("SELECT product FROM sales s JOIN products p ON s.product = p.product", "ambiguous")]
    [InlineData("SELECT region, qty FROM sales GROUP BY region", "must appear in GROUP BY")]
    [InlineData("SELECT id FROM sales WHERE SUM(qty) > 3", "not allowed in WHERE")]
    [InlineData("SELECT id FROM sales LIMIT 3 OFFSET 2", "OFFSET")]
    [InlineData("SELECT id FROM sales s JOIN products p ON s.qty > p.weight", "equality")]
    [InlineData("SELECT id FROM sales WHERE", "Expected an expression")]
    [InlineData("SELECT id FROM sales WHERE region = 'north", "Unterminated")]
    public void Errors_explain_themselves(string sql, string fragment)
    {
        var e = Assert.Throws<SqlException>(() => Context().Sql(sql));
        Assert.Contains(fragment, e.Message);
    }

    [Fact]
    public void Error_positions_point_at_the_problem()
    {
        const string sql = "SELECT id, bogus FROM sales";
        var e = Assert.Throws<SqlException>(() => Context().Sql(sql));
        Assert.Equal(sql.IndexOf("bogus", StringComparison.Ordinal), e.Position);
    }
}
