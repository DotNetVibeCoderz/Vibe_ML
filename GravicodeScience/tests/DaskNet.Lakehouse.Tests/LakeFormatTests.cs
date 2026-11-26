using System.Text.Json;
using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.DaskNet.Lakehouse;
using Gravicode.Science.GraviFrame;
using Xunit;
using static Gravicode.Science.DaskNet.Expr;
using static Gravicode.Science.Tests.DaskNet.Lakehouse.LakehouseTests;

namespace Gravicode.Science.Tests.DaskNet.Lakehouse;

/// <summary>
/// Delta deletion vectors and column mapping, and Hudi tables - all written by Spark 3.5 with
/// Delta Lake 3.3 and Hudi 1.0 (tools/verify/make_lake_fixtures_spark.py), each version compared
/// with what Spark itself read back.
/// </summary>
public sealed class LakeFormatTests
{
    // ------------------------------------------------------------------ Delta deletion vectors

    public static TheoryData<string, int> SparkDeltaVersions => new()
    {
        { "dv", 0 }, { "dv", 1 }, { "dv", 2 }, { "dv", 3 }, { "dv", 4 }, { "dv", 5 },
        { "mapped", 0 }, { "mapped", 1 }, { "mapped", 2 }, { "mapped", 3 }, { "mapped", 4 }, { "mapped", 5 },
        { "mapped_id", 0 }, { "mapped_id", 1 },
    };

    [Theory]
    [MemberData(nameof(SparkDeltaVersions))]
    public void Every_spark_delta_version_reads_as_spark_reads_it(string table, int version)
        => Same(Fixture("delta", table + "_expected", $"{version}.jsonl"), DeltaLake.Read(Fixture("delta", table), version).Compute());

    [Fact]
    public void Z85_decodes_the_reference_vector()
    {
        // From the ZeroMQ Z85 specification (rfc.zeromq.org/spec/32).
        Assert.Equal(new byte[] { 0x86, 0x4F, 0xD2, 0x6F, 0xB5, 0x59, 0xF7, 0x5B }, DeletionVectors.Z85Decode("HelloWorld"));
        Assert.Throws<InvalidDataException>(() => DeletionVectors.Z85Decode("Hello"[..4]));
    }

    [Fact]
    public void Every_deletion_vector_in_the_log_decodes_to_its_recorded_cardinality()
    {
        // The writer records how many rows each vector deletes; decoding must agree on every one,
        // and the positions must be distinct, ascending row numbers.
        var root = Fixture("delta", "dv");
        var seen = 0;
        foreach (var file in Directory.EnumerateFiles(Path.Combine(root, "_delta_log"), "*.json"))
            foreach (var line in File.ReadLines(file).Where(l => l.Length > 0))
            {
                using var doc = JsonDocument.Parse(line);
                if (!doc.RootElement.TryGetProperty("add", out var add) || DeletionVectors.FromJson(add, root) is not { } descriptor) continue;
                var rows = DeletionVectors.Read(descriptor);
                Assert.Equal(add.GetProperty("deletionVector").GetProperty("cardinality").GetInt64(), rows.Length);
                Assert.Equal(rows.Order().Distinct(), rows);
                seen++;
            }
        Assert.True(seen >= 5, $"only {seen} deletion vectors in the fixture");
    }

    [Fact]
    public void Deletion_vectors_survive_the_checkpoint_and_apply_before_filters()
    {
        // Version 4 is rebuilt from the v3 checkpoint, whose add actions carry the vectors.
        Assert.Contains(Directory.EnumerateFiles(Fixture("delta", "dv", "_delta_log")), f => f.EndsWith(".checkpoint.parquet", StringComparison.Ordinal));
        var table = DeltaLake.Read(Fixture("delta", "dv"), 4);
        Assert.Contains(Scan(table).Files, f => f.DeletionVector is not null);

        var copy = Path.Combine(Path.GetTempPath(), "delta-dv-nocheckpoint-" + Guid.NewGuid().ToString("N"));
        try
        {
            CopyDirectory(Fixture("delta", "dv"), copy);
            foreach (var f in Directory.EnumerateFiles(Path.Combine(copy, "_delta_log"), "*checkpoint*")) File.Delete(f);
            Assert.Equal(Canonical(table.Compute()), Canonical(DeltaLake.Read(copy, 4).Compute()));
        }
        finally { Directory.Delete(copy, recursive: true); }

        // Row positions are positions in the file, so a filter must not run before the vector.
        var north = table.Filter(Col("region").Eq("north") & (Col("amount") > 10)).Compute();
        var all = table.Compute();
        var expected = Enumerable.Range(0, all.RowCount)
            .Where(i => (string?)all["region"].GetValue(i) == "north" && all["amount"].GetValue(i) is double a && a > 10)
            .Select(i => (double)all["id"].GetValue(i)!).Order();
        Assert.Equal(expected, ((NumericSeries)north["id"]).Values.ToArray().Order());
    }

    [Fact]
    public void Column_mapping_reads_renamed_and_dropped_columns_by_their_table_names()
    {
        var path = Fixture("delta", "mapped");
        var v0 = DeltaLake.Read(path, 0);
        var latest = DeltaLake.Read(path);
        Assert.Equal(["id", "region", "amount", "name", "flag", "ts"], v0.Columns);
        Assert.Equal(["id", "area", "amount", "label", "ts"], latest.Columns);

        // The files are named by physical column; the partition directories too.
        Assert.All(Scan(latest).Files, f => Assert.Contains("col-", f.Path, StringComparison.Ordinal));
        var east = latest.Filter(Col("area").Eq("east"));
        var read = Scan(east).Build("t", null, [new FilterOp(Col("area").Eq("east"))]);
        Assert.Equal(Scan(latest).Files.Count(f => f.PartitionValues["area"] == "east"), read.Count(t => t.Remote!.Arguments[0] is string { Length: > 0 }));
        var rows = east.Compute();
        Assert.True(rows.RowCount > 0);
        Assert.All(Enumerable.Range(0, rows.RowCount), i => Assert.Equal("east", rows["area"].GetValue(i)));

        // Id mode matches by Parquet field id: renaming the column changes nothing in the files.
        Assert.Contains("price", DeltaLake.Read(Fixture("delta", "mapped_id")).Columns);
        Assert.Contains(Scan(DeltaLake.Read(Fixture("delta", "mapped_id"))).Files, f => f.FieldIds.Count > 0);
    }

    // ------------------------------------------------------------------ Hudi

    private static string[] HudiInstants(string table)
        => JsonSerializer.Deserialize<string[]>(File.ReadAllText(Fixture("hudi", table + "_expected", "instants.json")))!;

    public static TheoryData<string, int> HudiInstantIndexes => new()
    {
        { "cow", 0 }, { "cow", 1 }, { "cow", 2 }, { "cow", 3 }, { "flat", 0 }, { "flat", 1 }, { "mor", 0 }, { "mor", 1 },
    };

    [Theory]
    [MemberData(nameof(HudiInstantIndexes))]
    public void Every_hudi_instant_reads_as_spark_reads_it(string table, int index)
        => Same(Fixture("hudi", table + "_expected", $"{index}.jsonl"),
            Hudi.Read(Fixture("hudi", table), HudiInstants(table)[index], readOptimized: table == "mor").Compute());

    [Fact]
    public void Hudi_timeline_replacecommit_and_latest_read()
    {
        var path = Fixture("hudi", "cow");
        var instants = Hudi.Instants(path);
        Assert.Equal(HudiInstants("cow"), instants.Select(i => i.Time));
        Assert.Equal(["commit", "commit", "commit", "replacecommit"], instants.Select(i => i.Action));
        Assert.All(instants, i => Assert.NotNull(i.CompletionTime));
        Same(Fixture("hudi", "cow_expected", "3.jsonl"), Hudi.Read(path).Compute());

        // insert_overwrite replaced every north file group: only the new one remains.
        var latest = Scan(Hudi.Read(path));
        Assert.Single(latest.Files, f => f.PartitionValues["region"] == "north");
        Assert.True(Scan(Hudi.Read(path, instants[2].Time)).Files.Count(f => f.PartitionValues["region"] == "north") > 1);

        // A time between two instants reads the earlier one.
        var between = Hudi.ParseInstant(instants[1].Time).AddMilliseconds(1);
        Same(Fixture("hudi", "cow_expected", "1.jsonl"), Hudi.Read(path, new DateTimeOffset(between, TimeZoneInfo.Local.GetUtcOffset(between))).Compute());
        Assert.Throws<ArgumentException>(() => Hudi.Read(path, "19990101000000000"));
    }

    [Fact]
    public void Hudi_partition_pruning_meta_columns_and_a_cluster()
    {
        var path = Fixture("hudi", "cow");
        var table = Hudi.Read(path);
        Assert.DoesNotContain("_hoodie_record_key", table.Columns);
        var scan = Scan(table);
        var read = scan.Build("t", null, [new FilterOp(Col("region").Eq("south"))]);
        Assert.Equal(scan.Files.Count(f => f.PartitionValues["region"] == "south"), read.Count(t => t.Remote!.Arguments[0] is string { Length: > 0 }));

        var meta = Hudi.Read(path, includeMetaColumns: true).Compute();
        Assert.All(Enumerable.Range(0, meta.RowCount), i =>
            Assert.Equal(((double)meta["id"].GetValue(i)!).ToString(System.Globalization.CultureInfo.InvariantCulture), meta["_hoodie_record_key"].GetValue(i)));
        Assert.Equal("COPY_ON_WRITE", Hudi.Properties(path)["hoodie.table.type"]);
    }

    [Fact]
    public async Task Hudi_and_deletion_vector_tables_run_on_a_cluster()
    {
        var hudi = Hudi.Read(Fixture("hudi", "cow")).GroupBy("region").Agg(("amount", "sum"), ("id", "count"));
        var delta = DeltaLake.Read(Fixture("delta", "dv")).GroupBy("region").Agg(("amount", "sum"), ("id", "count"));
        await using var cluster = await LocalCluster.StartAsync(workers: 2);
        Assert.Equal(Canonical(hudi.Compute()), Canonical(hudi.Compute(cluster.Client)));
        Assert.Equal(Canonical(delta.Compute()), Canonical(delta.Compute(cluster.Client)));
    }

    [Fact]
    public void Hudi_merge_on_read_refuses_a_snapshot_with_pending_logs()
    {
        var path = Fixture("hudi", "mor");
        var error = Assert.Throws<NotSupportedException>(() => Hudi.Read(path));
        Assert.Contains("readOptimized", error.Message);
        // Before the upsert there were no logs, so a snapshot read is exact.
        Same(Fixture("hudi", "mor_expected", "0.jsonl"), Hudi.Read(path, HudiInstants("mor")[0]).Compute());
        Assert.Throws<DirectoryNotFoundException>(() => Hudi.Read(Fixture("delta", "dv")));
    }
}
