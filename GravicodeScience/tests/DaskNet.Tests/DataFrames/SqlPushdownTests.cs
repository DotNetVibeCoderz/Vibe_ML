using Gravicode.Science.DaskNet;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;
using Microsoft.Data.Sqlite;
using Xunit;
using static Gravicode.Science.DaskNet.Expr;

namespace Gravicode.Science.Tests.DaskNet;

/// <summary>
/// Filters pushed into a table read's WHERE. Every query is checked against the same read with
/// pushdown off, and the translation is checked for what it sends and what it must not send.
/// </summary>
public sealed class SqlPushdownTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dasknet-pushdown-" + Guid.NewGuid().ToString("N"));
    private readonly string _cs;

    public SqlPushdownTests()
    {
        Directory.CreateDirectory(_dir);
        _cs = $"Data Source={Path.Combine(_dir, "shop.sqlite")}";
        using var c = new SqliteConnection(_cs);
        c.Open();
        using var cmd = c.CreateCommand();
        // region compares case-insensitively in the database, as SQL Server's default collation does.
        cmd.CommandText = "CREATE TABLE orders (id INTEGER, region TEXT COLLATE NOCASE, qty REAL, day TEXT)";
        cmd.ExecuteNonQuery();
        using var tx = c.BeginTransaction();
        cmd.Transaction = tx;
        cmd.CommandText = "INSERT INTO orders VALUES ($id, $region, $qty, $day)";
        var id = cmd.Parameters.Add("$id", SqliteType.Integer);
        var region = cmd.Parameters.Add("$region", SqliteType.Text);
        var qty = cmd.Parameters.Add("$qty", SqliteType.Real);
        var day = cmd.Parameters.Add("$day", SqliteType.Text);
        string?[] regions = ["North", "north", "South", "NORTH ", null];
        for (var i = 0; i < 2_000; i++)
        {
            id.Value = i;
            region.Value = regions[i % 5] ?? (object)DBNull.Value;
            qty.Value = i % 11 == 0 ? DBNull.Value : i % 37;
            day.Value = $"2024-03-{1 + i % 28:D2}";
            cmd.ExecuteNonQuery();
        }
        tx.Commit();
    }

    public void Dispose()
    {
        SqliteConnection.ClearAllPools();
        Directory.Delete(_dir, recursive: true);
    }

    private DaskFrame Orders(bool push) => DaskFrame.ReadSql(SqliteFactory.Instance, _cs, "orders", "id", partitions: 3, pushFilters: push);

    /// <summary>What the optimised plan's scan would send as WHERE, with its parameters.</summary>
    private static (string Sql, object?[] Values)? Sent(DaskFrame frame)
    {
        var plan = Optimizer.Optimize(frame.Plan);
        while (plan is not ScanNode) plan = plan.Children[0];
        var scan = (ScanNode)plan;
        return scan.Predicate is null ? null : ((SqlSource)scan.Source).Where([scan.Predicate]);
    }

    public static TheoryData<string> Filters => new()
    {
        "numeric", "text-equality", "in", "not", "or", "null", "untranslatable", "mixed",
    };

    private static Expr Filter(string name) => name switch
    {
        "numeric" => Col("qty") >= 20 & Col("id") < 1500,
        "text-equality" => Col("region").Eq("North"),
        "in" => Col("region").In("North", "South") & Col("qty").In(1, 2, 3),
        "not" => !(Col("qty") > 10),
        "or" => Col("qty") < 3 | Col("id") >= 1990,
        "null" => Col("region").IsNull() | Col("qty").IsNull(),
        "untranslatable" => Col("region").StartsWith("No"),
        _ => (Col("qty") * 2 > 30) & (Col("region").Eq("north")) & !(Col("region").Eq("South")),
    };

    [Theory]
    [MemberData(nameof(Filters))]
    public void Pushed_filters_give_the_same_rows_as_filtering_after_the_read(string name)
    {
        var expected = Orders(push: false).Filter(Filter(name)).Compute();
        var actual = Orders(push: true).Filter(Filter(name)).Compute();
        Assert.Equal(SalesData.Rows(expected, sorted: true), SalesData.Rows(actual, sorted: true));
        Assert.True(expected.RowCount > 0, name);
    }

    [Fact]
    public void A_case_insensitive_collation_only_widens_what_the_database_returns()
    {
        // NOCASE matches North and north for = 'North' (not "NORTH ": SQLite keeps trailing
        // spaces); the frame keeps only North.
        var frame = Orders(push: true).Filter(Col("region").Eq("North"));
        var sent = Sent(frame)!.Value;
        Assert.Equal("region = @f0", sent.Sql);
        Assert.Equal(["North"], sent.Values);
        Assert.Equal(400, frame.Compute().RowCount);
        Assert.All(Enumerable.Range(0, 400), i => Assert.Equal("North", frame.Compute()["region"].GetValue(i)));

        using var c = new SqliteConnection(_cs);
        Assert.Equal(800, SqlReader.Read(c, "SELECT * FROM orders WHERE region = 'North'").RowCount);
    }

    [Fact]
    public void Only_conditions_that_can_widen_are_sent()
    {
        var sent = Sent(Orders(true).Filter(Col("qty") >= 20 & Col("region").StartsWith("No") & (5 > Col("id"))))!.Value;
        Assert.Equal("qty >= @f0 AND id < @f1", sent.Sql);              // the literal-first comparison is flipped
        Assert.Equal([20.0, 5.0], sent.Values);

        // An exact condition may be negated; text equality may not (NOT would narrow it).
        Assert.Equal("NOT qty > @f0", Sent(Orders(true).Filter(!(Col("qty") > 10)))!.Value.Sql);
        Assert.Null(Sent(Orders(true).Filter(!(Col("region").Eq("North")))));
        // Text ordering, dates stored as text, arithmetic: kept on the client.
        Assert.Null(Sent(Orders(true).Filter(Col("region") > "M")));
        Assert.Null(Sent(Orders(true).Filter(Col("qty") * 2 > 30)));
        // One untranslatable side of an OR sinks the OR; of an AND, only itself.
        Assert.Null(Sent(Orders(true).Filter(Col("qty") < 3 | Col("region").StartsWith("No"))));
        Assert.Equal("(qty < @f0 OR id >= @f1)", Sent(Orders(true).Filter(Col("qty") < 3 | Col("id") >= 1990))!.Value.Sql);
        Assert.Equal("region IS NULL", Sent(Orders(true).Filter(Col("region").IsNull()))!.Value.Sql);
        Assert.Equal("qty IN (@f0, @f1)", Sent(Orders(true).Filter(Col("qty").In(1, 2)))!.Value.Sql);
    }

    [Fact]
    public void Filters_above_a_projection_or_rename_are_pushed_only_when_they_reach_the_scan()
    {
        // Renamed columns: the optimiser pushes the filter below the rename, back to the table's names.
        var renamed = Orders(true).Rename("qty", "quantity").Filter(Col("quantity") > 30);
        Assert.Equal(Orders(false).Rename("qty", "quantity").Filter(Col("quantity") > 30).Compute().RowCount, renamed.Compute().RowCount);

        // A computed column is not a table column, so nothing is sent for it.
        var computed = Orders(true).WithColumns((Col("qty") * 2).As("double_qty")).Filter(Col("double_qty") > 30);
        Assert.Equal(Orders(false).WithColumns((Col("qty") * 2).As("double_qty")).Filter(Col("double_qty") > 30).Compute().RowCount,
            computed.Compute().RowCount);
    }
}
