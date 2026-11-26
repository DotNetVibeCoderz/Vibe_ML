using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Sql;
using Gravicode.Science.GraviFrame;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet;

/// <summary>
/// Multi-way join reordering. Every query is run both ways (<see cref="SqlContext.ReorderJoins"/>
/// on and off) and must give the same rows with the same columns; the plan is then checked for
/// the order the estimates call for.
/// </summary>
public sealed class JoinOrderTests
{
    // orders and shipments are big (one shipment per order); customers and regions are small.
    private static readonly DataFrame Orders = new([
        new NumericSeries("id", Enumerable.Range(0, 6_000).Select(i => (double)i).ToArray()),
        new NumericSeries("cust", Enumerable.Range(0, 6_000).Select(i => (double)(i * 7 % 300)).ToArray()),
        new NumericSeries("amount", Enumerable.Range(0, 6_000).Select(i => (i * 37 % 1000) / 10.0).ToArray()),
    ]);

    private static readonly DataFrame Shipments = new([
        new NumericSeries("order_id", Enumerable.Range(0, 6_000).Select(i => (double)(5_999 - i)).ToArray()),
        new NumericSeries("days", Enumerable.Range(0, 6_000).Select(i => (double)(i % 9)).ToArray()),
        new TextSeries("carrier", Enumerable.Range(0, 6_000).Select(i => (string?)$"carrier-{i % 4}").ToArray()),
    ]);

    private static readonly DataFrame Customers = new([
        new NumericSeries("cid", Enumerable.Range(0, 300).Select(i => (double)i).ToArray()),
        new TextSeries("region", Enumerable.Range(0, 300).Select(i => (string?)$"r{i % 5}").ToArray()),
        new TextSeries("tier", Enumerable.Range(0, 300).Select(i => (string?)(i % 10 == 0 ? "gold" : "basic")).ToArray()),
    ]);

    private static readonly DataFrame Regions = new([
        new TextSeries("code", Enumerable.Range(0, 5).Select(i => (string?)$"r{i}").ToArray()),
        new TextSeries("name", ["North", "South", "East", "West", "Central"]),
    ]);

    private static SqlContext Context(bool reorder) => new SqlContext { ReorderJoins = reorder }
        .Register("orders", DaskFrame.FromFrame(Orders, 4))
        .Register("shipments", DaskFrame.FromFrame(Shipments, 4))
        .Register("customers", DaskFrame.FromFrame(Customers, 2))
        .Register("regions", Regions);

    private static void SameBothWays(string sql)
    {
        var written = Context(reorder: false).Sql(sql).Compute();
        var reordered = Context(reorder: true).Sql(sql).Compute();
        Assert.Equal(written.ColumnNames, reordered.ColumnNames);
        Assert.Equal(written.Columns.Select(c => c.DataType), reordered.Columns.Select(c => c.DataType));
        Assert.Equal(SalesData.Rows(written, sorted: true), SalesData.Rows(reordered, sorted: true));
    }

    /// <summary>The innermost join of a plan: the tables (by alias prefix) on each of its sides.</summary>
    private static (string[] Left, string[] Right) FirstJoin(PlanNode node)
    {
        var joins = new List<JoinNode>();
        void Walk(PlanNode n)
        {
            if (n is JoinNode j) joins.Add(j);
            foreach (var c in n.Children) Walk(c);
        }
        Walk(node);
        var first = joins.Last(j => !Contains(j.Left) && !Contains(j.Right));
        return (Aliases(first.Left), Aliases(first.Right));

        static bool Contains(PlanNode n) => n is JoinNode || n.Children.Any(Contains);
        static string[] Aliases(PlanNode n) => n.Schema!.Where(c => c.Contains('.')).Select(c => c[..c.IndexOf('.')]).Distinct().ToArray();
    }

    [Fact]
    public void A_small_filtering_table_is_joined_before_a_second_big_one()
    {
        const string sql = """
            SELECT o.id, o.amount, s.days, c.tier
            FROM orders o JOIN shipments s ON s.order_id = o.id JOIN customers c ON o.cust = c.cid
            WHERE c.tier = 'gold'
            """;
        SameBothWays(sql);

        // Written: orders x shipments first (both big, shuffled). Reordered: orders x customers.
        Assert.Equal(["s"], FirstJoin(Context(false).Sql(sql).Plan).Right);
        var (left, right) = FirstJoin(Context(true).Sql(sql).Plan);
        Assert.Equal(["o"], left);
        Assert.Equal(["c"], right);

        var rows = Context(true).Sql(sql).Compute();
        Assert.Equal(Enumerable.Range(0, 6_000).Count(i => i * 7 % 300 % 10 == 0), rows.RowCount);
    }

    [Fact]
    public void Select_star_keeps_the_written_column_order_and_chains_through_a_dimension()
    {
        // regions joins customers, not orders: the two small tables are joined first (a bushy
        // plan) and then broadcast together to orders.
        const string sql = """
            SELECT * FROM orders o JOIN customers c ON o.cust = c.cid
                                   JOIN regions r ON c.region = r.code
            """;
        SameBothWays(sql);
        var frame = Context(true).Sql(sql);
        Assert.Equal(["id", "cust", "amount", "cid", "region", "tier", "code", "name"], frame.Columns);
        var (left, right) = FirstJoin(frame.Plan);
        Assert.Equal(["c"], left);
        Assert.Equal(["r"], right);
        Assert.Equal(6_000, frame.Compute().RowCount);
    }

    [Fact]
    public void Residual_conditions_multi_key_edges_and_aggregates_survive_reordering()
    {
        SameBothWays("""
            SELECT r.name, COUNT(*) AS n, SUM(o.amount) AS total
            FROM customers c JOIN orders o ON o.cust = c.cid AND o.amount > 50
                             JOIN regions r ON r.code = c.region
                             JOIN shipments s ON s.order_id = o.id AND s.days < 3
            GROUP BY r.name ORDER BY r.name
            """);
        // A cycle: the third equality joins two tables that are already in when its turn comes.
        SameBothWays("""
            SELECT o.id FROM shipments s JOIN orders o ON s.order_id = o.id
                                         JOIN customers c ON o.cust = c.cid AND c.cid = o.cust
            """);
    }

    [Fact]
    public void Outer_joins_and_unknown_sizes_keep_the_written_order()
    {
        const string outer = """
            SELECT o.id, s.days, c.tier FROM orders o JOIN shipments s ON s.order_id = o.id
                                                     LEFT JOIN customers c ON o.cust = c.cid
            """;
        SameBothWays(outer);
        Assert.Equal(["s"], FirstJoin(Context(true).Sql(outer).Plan).Right);

        // FromDelayed partitions have no estimate.
        var opaque = DaskFrame.FromDelayed([Dask.Value(Orders)], Orders.ColumnNames.ToArray());
        var context = Context(true).Register("orders", opaque);
        const string sql = "SELECT o.id FROM orders o JOIN shipments s ON s.order_id = o.id JOIN customers c ON o.cust = c.cid";
        Assert.Equal(["s"], FirstJoin(context.Sql(sql).Plan).Right);
        Assert.Equal(6_000, context.Sql(sql).Compute().RowCount);
    }
}
