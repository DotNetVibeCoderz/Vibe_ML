using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;
using Microsoft.Data.Sqlite;
using Xunit;
using static Gravicode.Science.DaskNet.Expr;

namespace Gravicode.Science.Tests.DaskNet;

public sealed class ConnectorTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dasknet-connectors-" + Guid.NewGuid().ToString("N"));

    public ConnectorTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        SqliteConnection.ClearAllPools();   // pooled connections keep the database file open
        Directory.Delete(_dir, recursive: true);
    }

    // ------------------------------------------------------------------ JSON

    [Fact]
    public void Json_lines_round_trip_every_column_type_and_missing_values()
    {
        var sales = SalesData.Make(1_200, seed: 8);
        var paths = DaskFrame.FromFrame(sales, 3).ToJson(Path.Combine(_dir, "out"));
        Assert.Equal(3, paths.Length);

        foreach (var block in new long[] { 2_000, 64L << 20 })
        {
            var back = DaskFrame.ReadJson(Path.Combine(_dir, "out", "*.jsonl"), blockSize: block);
            if (block == 2_000) Assert.True(back.PartitionCount > 10);
            var frame = back.Compute();
            Assert.Equal(sales.ColumnNames, frame.ColumnNames);
            Assert.Equal(sales.Columns.Select(c => c.DataType), frame.Columns.Select(c => c.DataType));
            Assert.Equal(SalesData.Rows(sales, sorted: true), SalesData.Rows(frame, sorted: true));
        }
    }

    [Fact]
    public void Json_values_are_read_to_the_schema_of_the_sample()
    {
        var path = Path.Combine(_dir, "mixed.jsonl");
        File.WriteAllLines(path,
        [
            """{"id": 1, "name": "a", "ok": true, "when": "2024-03-01", "tags": ["x", "y"]}""",
            """{"id": 2, "name": null, "ok": false, "when": "2024-03-02T10:30:00", "tags": {"k": 1}}""",
            "",
            """{"id": "3", "ok": "true", "when": "not a date"}""",
            """{"id": "three", "name": 42, "late": "ignored"}""",
        ]);
        var frame = DaskFrame.ReadJson(path, inferenceRows: 2).Compute();
        Assert.Equal(["id", "name", "ok", "when", "tags"], frame.ColumnNames);
        Assert.Equal([1.0, 2.0, 3.0, double.NaN], ((NumericSeries)frame["id"]).Values.ToArray());   // "3" parsed, "three" missing
        Assert.Equal(["a", null, null, "42"], Enumerable.Range(0, 4).Select(i => (string?)frame["name"].GetValue(i)));
        Assert.Equal([true, false, true, null], Enumerable.Range(0, 4).Select(i => (bool?)frame["ok"].GetValue(i)));
        Assert.Equal(new DateTime(2024, 3, 2, 10, 30, 0), frame["when"].GetValue(1));
        Assert.Null(frame["when"].GetValue(2));
        Assert.Equal("""["x", "y"]""", frame["tags"].GetValue(0));                                    // nested kept as JSON

        File.AppendAllText(path, "{\"id\": 5, oops}\n");
        var e = Assert.ThrowsAny<Exception>(() => DaskFrame.ReadJson(path).Compute());
        Assert.Contains("malformed JSON", e.ToString());
    }

    [Fact]
    public async Task Json_arrays_read_one_file_per_partition_and_run_on_a_cluster()
    {
        for (var f = 0; f < 3; f++)
            File.WriteAllText(Path.Combine(_dir, $"part{f}.json"),
                "[" + string.Join(",", Enumerable.Range(0, 50).Select(i => $$"""{"k": "g{{i % 4}}", "v": {{f * 100 + i}}}""")) + "]");
        var df = DaskFrame.ReadJson(Path.Combine(_dir, "part*.json"), lines: false);
        Assert.Equal(3, df.PartitionCount);
        Assert.True(df.IsRemotable);
        var totals = df.Filter(Col("v") >= 10).GroupBy("k").Sum("v");
        await using var cluster = await LocalCluster.StartAsync(workers: 2);
        Assert.Equal(SalesData.Rows(totals.Compute(), sorted: true), SalesData.Rows(totals.Compute(cluster.Client), sorted: true));
        var expected = Enumerable.Range(0, 3).SelectMany(f => Enumerable.Range(0, 50).Select(i => f * 100 + i)).Where(v => v >= 10).Sum();
        Assert.Equal(expected, ((NumericSeries)totals.Compute()["v"]).Values.ToArray().Sum());
    }

    // ------------------------------------------------------------------ SQL

    private string Database(int rows, Func<int, object?>? qty = null)
    {
        var path = Path.Combine(_dir, $"db-{Guid.NewGuid():N}.sqlite");
        var cs = $"Data Source={path}";
        using var c = new SqliteConnection(cs);
        c.Open();
        using (var create = c.CreateCommand())
        {
            create.CommandText = "CREATE TABLE orders (id INTEGER, region TEXT, qty REAL)";
            create.ExecuteNonQuery();
        }
        using var tx = c.BeginTransaction();
        using var insert = c.CreateCommand();
        insert.CommandText = "INSERT INTO orders VALUES ($id, $region, $qty)";
        var id = insert.Parameters.Add("$id", SqliteType.Integer);
        var region = insert.Parameters.Add("$region", SqliteType.Text);
        var q = insert.Parameters.Add("$qty", SqliteType.Real);
        for (var i = 0; i < rows; i++)
        {
            id.Value = i % 97 == 0 ? DBNull.Value : i;                    // some rows have no index
            region.Value = i % 13 == 0 ? DBNull.Value : SalesData.Regions[i % 5];
            q.Value = (qty ?? (i => i * 0.5))(i) ?? DBNull.Value;
            insert.ExecuteNonQuery();
        }
        tx.Commit();
        return cs;
    }

    private static DataFrame Whole(string cs, string sql)
    {
        using var c = new SqliteConnection(cs);
        return SqlReader.Read(c, sql);
    }

    [Fact]
    public void Sql_tables_split_into_ranges_read_every_row_once()
    {
        var cs = Database(1_000);
        var expected = Whole(cs, "SELECT * FROM orders");
        foreach (var (partitions, bounds) in new (int, (double, double)?)[] { (1, null), (4, null), (7, null), (5, (200, 300)) })
        {
            var df = DaskFrame.ReadSql(SqliteFactory.Instance, cs, "orders", "id", partitions, bounds);
            Assert.Equal(partitions, df.PartitionCount);
            Assert.Equal(SalesData.Rows(expected, sorted: true), SalesData.Rows(df.Compute(), sorted: true));   // NULL ids included
        }
    }

    [Fact]
    public void Sql_reads_select_only_needed_columns()
    {
        var cs = Database(400, i => i >= 300 ? null : i * 2.0);
        var df = DaskFrame.ReadSql(SqliteFactory.Instance, cs, "orders", "id", partitions: 4);
        var byRegion = df.Filter(Col("region").IsNotNull()).GroupBy("region").Sum("qty");
        Assert.Contains("columns=[region, qty]", byRegion.Explain());

        var parts = df.Select("id", "qty").ToDelayed();
        var last = parts[^1].Compute();
        Assert.IsType<NumericSeries>(last["qty"]);
        // The last range also holds the rows with no id (which have quantities); the rest have none.
        var ids = ((NumericSeries)last["id"]).Values.ToArray();
        var qtys = ((NumericSeries)last["qty"]).Values.ToArray();
        Assert.True(ids.Count(id => !double.IsNaN(id)) > 50);
        Assert.All(Enumerable.Range(0, ids.Length).Where(i => !double.IsNaN(ids[i])), i => Assert.True(double.IsNaN(qtys[i])));

        var expected = Whole(cs, "SELECT region, SUM(qty) AS s FROM orders WHERE region IS NOT NULL GROUP BY region");
        var actual = byRegion.Compute();
        for (var i = 0; i < expected.RowCount; i++)
        {
            var region = (string)expected["region"].GetValue(i)!;
            var row = Enumerable.Range(0, actual.RowCount).Single(r => (string?)actual["region"].GetValue(r) == region);
            Assert.Equal((double)expected["s"].GetValue(i)!, ((NumericSeries)actual["qty"])[row], 9);
        }
    }

    /// <summary>
    /// A column with no declared type is typed by its values in SQLite, so a range where it is all
    /// NULL reads back as text while the others read numbers - and concatenating them would mix
    /// types. Every partition is converted to the types the schema probe found.
    /// </summary>
    [Fact]
    public void Sql_partitions_are_pinned_to_one_schema()
    {
        var path = Path.Combine(_dir, "loose.sqlite");
        var cs = $"Data Source={path}";
        using (var c = new SqliteConnection(cs))
        {
            c.Open();
            using var cmd = c.CreateCommand();
            cmd.CommandText = "CREATE TABLE loose (id INTEGER, v); " + string.Join(" ",
                Enumerable.Range(0, 40).Select(i => $"INSERT INTO loose VALUES ({i}, {(i < 30 ? (i * 1.5).ToString(System.Globalization.CultureInfo.InvariantCulture) : "NULL")});"));
            cmd.ExecuteNonQuery();
        }
        // The probe of an empty result cannot see values; pin to numbers by asking a range with some.
        var df = DaskFrame.ReadSql(SqliteFactory.Instance, cs, [new SqlPartition("SELECT id, v FROM loose WHERE id < 30"),
            new SqlPartition("SELECT id, v FROM loose WHERE id >= 30")]);
        var parts = df.ToDelayed().Select(p => p.Compute()).ToArray();
        Assert.All(parts, p => Assert.IsType<NumericSeries>(p["v"]));
        Assert.All(((NumericSeries)parts[1]["v"]).Values.ToArray(), v => Assert.True(double.IsNaN(v)));
        Assert.Equal(Enumerable.Range(0, 30).Sum(i => i * 1.5), ((NumericSeries)df.Compute()["v"]).Values.ToArray().Where(v => !double.IsNaN(v)).Sum(), 9);
    }

    [Fact]
    public async Task Sql_queries_are_partitions_and_run_on_a_cluster()
    {
        var cs = Database(600);
        var queries = SalesData.Regions.Select(r => new SqlPartition(
            "SELECT id, region, qty FROM orders WHERE region = @r", new Dictionary<string, object?> { ["r"] = r })).ToList();
        var df = DaskFrame.ReadSql(SqliteFactory.Instance, cs, queries);
        Assert.Equal(5, df.PartitionCount);
        Assert.True(df.IsRemotable);
        var expected = Whole(cs, "SELECT id, region, qty FROM orders WHERE region IS NOT NULL");
        Assert.Equal(SalesData.Rows(expected, sorted: true), SalesData.Rows(df.Compute(), sorted: true));

        await using var cluster = await LocalCluster.StartAsync(workers: 2);
        var report = df.GroupBy("region").Agg(("qty", "mean"), ("id", "count"));
        Assert.Equal(SalesData.Rows(report.Compute(), sorted: true), SalesData.Rows(report.Compute(cluster.Client), sorted: true));
    }

    [Fact]
    public void Sql_refuses_identifiers_that_could_carry_sql_and_handles_empty_tables()
    {
        var cs = Database(0);
        Assert.Throws<ArgumentException>(() => DaskFrame.ReadSql(SqliteFactory.Instance, cs, "orders; DROP TABLE orders", "id", 2));
        Assert.Throws<ArgumentException>(() => DaskFrame.ReadSql(SqliteFactory.Instance, cs, "orders", "id) OR (1=1", 2));
        Assert.Throws<ArgumentException>(() => DaskFrame.ReadSql(SqliteFactory.Instance, cs, "orders", "nope", 2));

        var empty = DaskFrame.ReadSql(SqliteFactory.Instance, cs, "orders", "id", partitions: 4);
        Assert.Equal(1, empty.PartitionCount);
        var frame = empty.Compute();
        Assert.Equal(0, frame.RowCount);
        Assert.Equal(["id", "region", "qty"], frame.ColumnNames);
    }
}
