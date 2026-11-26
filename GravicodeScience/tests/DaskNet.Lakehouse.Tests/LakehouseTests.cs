using System.Globalization;
using System.Text.Json;
using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.DaskNet.Lakehouse;
using Gravicode.Science.DaskNet.Sql;
using Gravicode.Science.GraviFrame;
using Xunit;
using static Gravicode.Science.DaskNet.Expr;

namespace Gravicode.Science.Tests.DaskNet.Lakehouse;

/// <summary>
/// Every version of every fixture table is read and compared with the rows the reference
/// implementation (delta-rs, pyiceberg) returned for the same version.
/// </summary>
public sealed class LakehouseTests
{
    internal static string Fixture(params string[] parts) => Path.Combine([AppContext.BaseDirectory, "Fixtures", .. parts]);

    /// <summary>One canonical line per row: columns sorted, numbers round-tripped, dates ISO.</summary>
    internal static List<string> Canonical(DataFrame frame)
    {
        var columns = frame.ColumnNames.Order(StringComparer.Ordinal).ToArray();
        return Enumerable.Range(0, frame.RowCount).Select(i => string.Join("|", columns.Select(c => c + "=" + frame[c].GetValue(i) switch
        {
            null => "null",
            double d when double.IsNaN(d) => "null",
            double d => d.ToString("R", CultureInfo.InvariantCulture),
            DateTime t => t.ToString("yyyy-MM-ddTHH:mm:ss.FFFFFFF", CultureInfo.InvariantCulture),
            bool b => b ? "true" : "false",
            var o => o.ToString(),
        }))).Order(StringComparer.Ordinal).ToList();
    }

    private static List<string> Expected(string file)
        => File.ReadLines(file).Where(l => l.Length > 0).Select(line =>
        {
            using var doc = JsonDocument.Parse(line);
            return string.Join("|", doc.RootElement.EnumerateObject().OrderBy(p => p.Name, StringComparer.Ordinal).Select(p => p.Name + "=" + p.Value.ValueKind switch
            {
                JsonValueKind.Null => "null",
                JsonValueKind.Number => p.Value.GetDouble().ToString("R", CultureInfo.InvariantCulture),
                JsonValueKind.True => "true",
                JsonValueKind.False => "false",
                _ when DateTime.TryParse(p.Value.GetString(), CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out var t)
                    && p.Name == "ts" => t.ToString("yyyy-MM-ddTHH:mm:ss.FFFFFFF", CultureInfo.InvariantCulture),
                _ => p.Value.GetString(),
            }));
        }).Order(StringComparer.Ordinal).ToList();

    internal static void Same(string expectedFile, DataFrame actual)
    {
        var expected = Expected(expectedFile);
        var got = Canonical(actual);
        Assert.Equal(expected.Count, got.Count);
        Assert.Equal(expected, got);
    }

    // ------------------------------------------------------------------ Delta

    public static TheoryData<string, int> DeltaVersions => new()
    {
        { "sales", 0 }, { "sales", 1 }, { "sales", 2 }, { "sales", 3 }, { "sales", 4 }, { "sales", 5 }, { "sales", 6 },
        { "plain", 0 }, { "plain", 1 },
    };

    [Theory]
    [MemberData(nameof(DeltaVersions))]
    public void Every_delta_version_reads_as_delta_rs_reads_it(string table, int version)
        => Same(Fixture("delta", table + "_expected", $"{version}.jsonl"), DeltaLake.Read(Fixture("delta", table), version).Compute());

    [Fact]
    public void Delta_latest_version_comes_from_the_checkpoint_and_the_commits_after_it()
    {
        Assert.Equal(6, DeltaLake.LatestVersion(Fixture("delta", "sales")));
        Same(Fixture("delta", "sales_expected", "6.jsonl"), DeltaLake.Read(Fixture("delta", "sales")).Compute());

        // Replaying v4 from the v3 checkpoint must equal replaying it from commit 0. Copy the
        // table without its checkpoint and compare.
        var copy = Path.Combine(Path.GetTempPath(), "delta-nocheckpoint-" + Guid.NewGuid().ToString("N"));
        try
        {
            CopyDirectory(Fixture("delta", "sales"), copy);
            foreach (var f in Directory.EnumerateFiles(Path.Combine(copy, "_delta_log"), "*checkpoint*")) File.Delete(f);
            Assert.Equal(Canonical(DeltaLake.Read(Fixture("delta", "sales"), 4).Compute()), Canonical(DeltaLake.Read(copy, 4).Compute()));

            // Without the early commits a checkpoint is the only way to v3 or later; v2 is gone.
            CopyDirectory(Fixture("delta", "sales"), copy);
            for (var v = 0; v <= 2; v++) File.Delete(Path.Combine(copy, "_delta_log", $"{v:D20}.json"));
            Same(Fixture("delta", "sales_expected", "4.jsonl"), DeltaLake.Read(copy, 4).Compute());
            Assert.Throws<InvalidDataException>(() => DeltaLake.Read(copy, 2));
        }
        finally { Directory.Delete(copy, recursive: true); }
    }

    [Fact]
    public void Delta_history_and_time_travel_by_timestamp()
    {
        var history = DeltaLake.History(Fixture("delta", "sales"));
        Assert.Equal(Enumerable.Range(0, 7).Select(v => (long)v), history.Select(h => h.Version));
        Assert.Contains(history, h => h.Operation == "DELETE");
        Assert.All(history, h => Assert.NotNull(h.Timestamp));
        var at = history[3].Timestamp!.Value;
        // Commits can share a millisecond; time travel picks the newest version at that instant.
        var expected = history.Where(h => h.Timestamp <= at).Max(h => h.Version);
        Same(Fixture("delta", "sales_expected", $"{expected}.jsonl"), DeltaLake.Read(Fixture("delta", "sales"), at).Compute());
    }

    [Fact]
    public void Delta_partition_filters_prune_files_and_columns_are_projected()
    {
        var table = DeltaLake.Read(Fixture("delta", "sales"), 4);
        var north = table.Filter(Col("region").Eq("north") & (Col("amount") > 20)).Select("id", "amount");
        var all = table.Compute();
        var expected = Enumerable.Range(0, all.RowCount)
            .Where(i => (string?)all["region"].GetValue(i) == "north" && all["amount"].GetValue(i) is double a && a > 20)
            .Select(i => (double)all["id"].GetValue(i)!).Order().ToArray();
        var got = north.Compute();
        Assert.Equal(["id", "amount"], got.ColumnNames);
        Assert.Equal(expected, ((NumericSeries)got["id"]).Values.ToArray().Order().ToArray());

        // Files of the other two regions are never opened: their tasks have no path.
        var scan = Scan(north);
        var read = scan.Build("t", ["id", "amount", "region"], [new FilterOp(Col("region").Eq("north"))]);
        Assert.Equal(scan.Files.Count(f => f.PartitionValues["region"] == "north"), read.Count(t => t.Remote!.Arguments[0] is string { Length: > 0 }));
        Assert.Equal(scan.Files.Count, read.Length);
    }

    [Fact]
    public async Task Lake_tables_run_on_a_cluster_and_through_sql()
    {
        var sales = DeltaLake.Read(Fixture("delta", "sales"));
        Assert.True(sales.IsRemotable);
        var report = sales.GroupBy("region").Agg(("amount", "sum"), ("id", "count"));
        await using var cluster = await LocalCluster.StartAsync(workers: 2);
        Assert.Equal(Canonical(report.Compute()), Canonical(report.Compute(cluster.Client)));

        // Version 4 predates the overwrite, so its ids overlap the Iceberg table's.
        var sql = new SqlContext().Register("delta_sales", DeltaLake.Read(Fixture("delta", "sales"), 4)).Register("ice_sales", Iceberg.Read(Fixture("iceberg", "db", "sales")));
        var both = sql.Sql("SELECT d.region, COUNT(*) AS n FROM delta_sales d JOIN ice_sales i ON d.id = i.id GROUP BY d.region ORDER BY d.region").Compute();
        Assert.Equal(["east", "north", "south"], Enumerable.Range(0, both.RowCount).Select(i => (string?)both["region"].GetValue(i)));
    }

    [Fact]
    public void Delta_refuses_what_it_cannot_read_correctly()
    {
        var copy = Path.Combine(Path.GetTempPath(), "delta-refuse-" + Guid.NewGuid().ToString("N"));
        try
        {
            CopyDirectory(Fixture("delta", "plain"), copy);
            var log = Path.Combine(copy, "_delta_log");
            File.AppendAllText(Path.Combine(log, $"{2:D20}.json"),
                """{"metaData":{"id":"x","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[{\"name\":\"id\",\"type\":\"long\",\"nullable\":true,\"metadata\":{}}]}","partitionColumns":[],"configuration":{"delta.columnMapping.mode":"name"}}}""" + "\n");
            Assert.Contains("no physical name", Assert.Throws<InvalidDataException>(() => DeltaLake.Read(copy).Compute()).Message);

            File.WriteAllText(Path.Combine(log, $"{3:D20}.json"), """{"protocol":{"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":["variantType"]}}""" + "\n");
            Assert.Contains("variantType", Assert.Throws<NotSupportedException>(() => DeltaLake.Read(copy)).Message);
        }
        finally { Directory.Delete(copy, recursive: true); }
        Assert.Throws<DirectoryNotFoundException>(() => DeltaLake.Read(Fixture("iceberg")));
    }

    // ------------------------------------------------------------------ Iceberg

    public static TheoryData<string, int> IcebergSnapshotIndexes => new()
    {
        { "sales", 0 }, { "sales", 1 }, { "sales", 2 }, { "sales", 3 }, { "plain", 0 }, { "plain", 1 }, { "plain", 2 },
        { "renamed", 0 }, { "renamed", 1 },
    };

    private static long[] SnapshotIds(string table)
        => JsonSerializer.Deserialize<long[]>(File.ReadAllText(Fixture("iceberg", table + "_expected", "snapshots.json")))!;

    [Theory]
    [MemberData(nameof(IcebergSnapshotIndexes))]
    public void Every_iceberg_snapshot_reads_as_pyiceberg_reads_it(string table, int index)
        => Same(Fixture("iceberg", table + "_expected", $"{index}.jsonl"),
            Iceberg.Read(Fixture("iceberg", "db", table), SnapshotIds(table)[index]).Compute());

    [Fact]
    public void Iceberg_current_snapshot_schema_evolution_and_snapshot_list()
    {
        var path = Fixture("iceberg", "db", "sales");
        Same(Fixture("iceberg", "sales_expected", "3.jsonl"), Iceberg.Read(path).Compute());
        Assert.Equal(SnapshotIds("sales"), Iceberg.Snapshots(path).Select(s => s.Id));
        Assert.DoesNotContain("score", Iceberg.Read(path, SnapshotIds("sales")[0]).Columns);   // the schema of then
        Assert.Contains("score", Iceberg.Read(path).Columns);

        // A metadata file can be given directly; snapshots by time pick the newest at or before.
        var last = Directory.EnumerateFiles(Path.Combine(path, "metadata"), "*.metadata.json").Order(StringComparer.Ordinal).Last();
        Same(Fixture("iceberg", "sales_expected", "3.jsonl"), Iceberg.Read(last).Compute());
        var second = Iceberg.Snapshots(path)[1];
        Assert.Equal(Canonical(Iceberg.Read(path, second.Id).Compute()), Canonical(Iceberg.Read(path, second.Timestamp).Compute()));
    }

    [Fact]
    public void Iceberg_tables_read_after_being_moved_and_prune_identity_partitions()
    {
        // Every path in the metadata is absolute; the copy must still find its own files.
        var copy = Path.Combine(Path.GetTempPath(), "iceberg-moved-" + Guid.NewGuid().ToString("N"));
        try
        {
            CopyDirectory(Fixture("iceberg", "db", "sales"), copy);
            var moved = Iceberg.Read(copy);
            Same(Fixture("iceberg", "sales_expected", "3.jsonl"), moved.Compute());
            var scan = Scan(moved);
            Assert.All(scan.Files, f => Assert.StartsWith(Path.GetFullPath(copy), f.Path, StringComparison.OrdinalIgnoreCase));

            var read = scan.Build("t", null, [new FilterOp(Col("region").Eq("south"))]);
            Assert.Equal(scan.Files.Count(f => f.PartitionValues["region"] == "south"), read.Count(t => t.Remote!.Arguments[0] is string { Length: > 0 }));
            var south = moved.Filter(Col("region").Eq("south")).Compute();
            Assert.All(Enumerable.Range(0, south.RowCount), i => Assert.Equal("south", south["region"].GetValue(i)));
        }
        finally { Directory.Delete(copy, recursive: true); }
    }

    [Fact]
    public void Avro_reader_decodes_iceberg_manifests()
    {
        var list = Directory.EnumerateFiles(Fixture("iceberg", "db", "sales", "metadata"), "snap-*.avro").First();
        var (metadata, records) = Avro.Read(list);
        Assert.Contains("avro.schema", metadata.Keys);
        Assert.All(records.Cast<Dictionary<string, object?>>(), r => Assert.EndsWith(".avro", (string)r["manifest_path"]!));
    }

    internal static LakeSource Scan(DaskFrame frame)
    {
        var node = Optimizer.Optimize(frame.Plan);
        while (node is not ScanNode) node = node.Children[0];
        return (LakeSource)((ScanNode)node).Source;
    }

    internal static void CopyDirectory(string from, string to)
    {
        if (Directory.Exists(to)) Directory.Delete(to, recursive: true);
        foreach (var dir in Directory.EnumerateDirectories(from, "*", SearchOption.AllDirectories))
            Directory.CreateDirectory(Path.Combine(to, Path.GetRelativePath(from, dir)));
        Directory.CreateDirectory(to);
        foreach (var file in Directory.EnumerateFiles(from, "*", SearchOption.AllDirectories))
            File.Copy(file, Path.Combine(to, Path.GetRelativePath(from, file)));
    }
}
