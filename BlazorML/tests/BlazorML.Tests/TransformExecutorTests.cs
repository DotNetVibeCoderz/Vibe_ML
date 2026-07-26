using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.ML.Execution.Executors;

namespace BlazorML.Tests;

public class TransformExecutorTests
{
    private readonly TransformExecutor _executor = new();

    // ------------------------------------------------------------------ split

    [Fact]
    public void SplitData_divides_rows_by_the_requested_fraction()
    {
        var input = TestContext.Classified(50, "a", "b");
        var context = TestContext.For("tf.splitData",
            [("fraction", "0.8"), ("stratifyColumn", null)], input);

        var results = _executor.ExecuteAsync(context).GetAwaiter().GetResult();
        var left = (TabularData)results[0]!;
        var right = (TabularData)results[1]!;

        Assert.Equal(80, left.RowCount);
        Assert.Equal(20, right.RowCount);
        Assert.Equal(input.RowCount, left.RowCount + right.RowCount);
    }

    [Fact]
    public void SplitData_stratified_keeps_the_class_balance_in_both_halves()
    {
        var input = TestContext.Classified(50, "a", "b");
        var context = TestContext.For("tf.splitData",
            [("fraction", "0.8"), ("stratifyColumn", "kelas")], input);

        var results = _executor.ExecuteAsync(context).GetAwaiter().GetResult();
        var left = (TabularData)results[0]!;
        var right = (TabularData)results[1]!;

        Assert.Equal(40, left.Rows.Count(r => Convert.ToString(r[1]) == "a"));
        Assert.Equal(40, left.Rows.Count(r => Convert.ToString(r[1]) == "b"));
        Assert.Equal(10, right.Rows.Count(r => Convert.ToString(r[1]) == "a"));
        Assert.Equal(10, right.Rows.Count(r => Convert.ToString(r[1]) == "b"));
    }

    /// <summary>
    /// Regression test for the worst bug found during the build. Stratifying on a continuous
    /// column made almost every value its own group of one, every group rounded entirely into
    /// the first output, and the second came back with a single row. The run still reported
    /// success, with regression metrics computed from that one row. It must refuse instead.
    /// </summary>
    [Fact]
    public void SplitData_refuses_to_stratify_by_a_continuous_column()
    {
        var input = TabularData.WithColumns("harga");

        for (var i = 0; i < 400; i++)
        {
            table_add(input, 1000 + i * 3.7);
        }

        var context = TestContext.For("tf.splitData",
            [("fraction", "0.8"), ("stratifyColumn", "harga")], input);

        var error = Assert.Throws<InvalidOperationException>(() =>
            _executor.ExecuteAsync(context).GetAwaiter().GetResult());

        Assert.Contains("cannot be used to stratify", error.Message);
        Assert.Contains("harga", error.Message);

        static void table_add(TabularData t, double value) => t.AddRow(value);
    }

    [Fact]
    public void SplitData_is_reproducible_for_a_fixed_seed()
    {
        var input = TestContext.Classified(30, "a", "b");

        static TabularData Run(TabularData data)
        {
            var context = TestContext.For("tf.splitData",
                [("fraction", "0.7"), ("randomSeed", "1234"), ("stratifyColumn", null)], data);

            return (TabularData)new TransformExecutor()
                .ExecuteAsync(context).GetAwaiter().GetResult()[0]!;
        }

        Assert.Equal(Run(input).ToCsv(), Run(input).ToCsv());
    }

    // ------------------------------------------------------------------ clean

    [Fact]
    public void CleanMissing_fills_gaps_with_the_column_mean()
    {
        var input = TabularData.WithColumns("nilai");
        input.AddRow(10);
        input.AddRow((object?)null);
        input.AddRow(20);

        var context = TestContext.For("tf.cleanMissing", [("strategy", "mean")], input);
        var output = _executor.Output<TabularData>(context);

        Assert.Equal(15d, TabularData.ToNumber(output.Value(1, "nilai")));
    }

    [Fact]
    public void CleanMissing_falls_back_to_the_most_frequent_value_for_a_text_column()
    {
        // A text column has no mean; silently writing NaN there would poison the column.
        var input = TabularData.WithColumns("kota");
        input.AddRow("Bandung");
        input.AddRow("Bandung");
        input.AddRow((object?)null);

        var context = TestContext.For("tf.cleanMissing", [("strategy", "mean")], input);
        var output = _executor.Output<TabularData>(context);

        Assert.Equal("Bandung", output.Text(2, "kota"));
    }

    [Fact]
    public void CleanMissing_dropRow_removes_only_rows_with_gaps()
    {
        var input = TabularData.WithColumns("a", "b");
        input.AddRow(1, 2);
        input.AddRow(3, null);
        input.AddRow(5, 6);

        var context = TestContext.For("tf.cleanMissing", [("strategy", "dropRow")], input);
        var output = _executor.Output<TabularData>(context);

        Assert.Equal(2, output.RowCount);
    }

    // ------------------------------------------------------------------- misc

    [Fact]
    public void SelectColumns_keeps_or_drops_as_asked()
    {
        var input = TabularData.WithColumns("a", "b", "c");
        input.AddRow(1, 2, 3);

        var kept = _executor.Output<TabularData>(
            TestContext.For("tf.selectColumns", [("columns", "a, c"), ("mode", "keep")], input));

        var dropped = _executor.Output<TabularData>(
            TestContext.For("tf.selectColumns", [("columns", "a, c"), ("mode", "drop")], input));

        Assert.Equal(["a", "c"], kept.Columns.Select(c => c.Name));
        Assert.Equal(["b"], dropped.Columns.Select(c => c.Name));
    }

    [Fact]
    public void FilterRows_compares_numerically_when_both_sides_are_numbers()
    {
        var input = TabularData.WithColumns("umur");
        foreach (var age in (int[])[10, 25, 40, 5])
        {
            input.AddRow(age);
        }

        var context = TestContext.For("tf.filterRows",
            [("column", "umur"), ("op", "gt"), ("value", "20")], input);

        var output = _executor.Output<TabularData>(context);

        Assert.Equal(2, output.RowCount);
    }

    [Fact]
    public void RemoveDuplicates_collapses_rows_sharing_the_key_columns()
    {
        var input = TabularData.WithColumns("id", "nilai");
        input.AddRow("x", 1);
        input.AddRow("x", 2);
        input.AddRow("y", 3);

        var context = TestContext.For("tf.removeDuplicates",
            [("keyColumns", "id"), ("keep", "first")], input);

        var output = _executor.Output<TabularData>(context);

        Assert.Equal(2, output.RowCount);
        Assert.Equal("1", output.Text(0, "nilai"));
    }

    [Fact]
    public void Join_matches_on_the_key_and_brings_the_right_columns_across()
    {
        var left = TabularData.WithColumns("id", "nama");
        left.AddRow("1", "Ani");
        left.AddRow("2", "Budi");

        var right = TabularData.WithColumns("id_pel", "kota");
        right.AddRow("1", "Bandung");
        right.AddRow("2", "Jakarta");

        var context = TestContext.For("tf.join",
            [("leftKey", "id"), ("rightKey", "id_pel"), ("kind", "inner")], left, right);

        var output = _executor.Output<TabularData>(context);

        Assert.Equal(2, output.RowCount);
        Assert.Equal("Bandung", output.Text(0, "kota"));
        Assert.Equal("Jakarta", output.Text(1, "kota"));
    }

    [Fact]
    public void Join_left_keeps_unmatched_rows_with_empty_right_columns()
    {
        var left = TabularData.WithColumns("id");
        left.AddRow("1");
        left.AddRow("hilang");

        var right = TabularData.WithColumns("id_pel", "kota");
        right.AddRow("1", "Bandung");

        var context = TestContext.For("tf.join",
            [("leftKey", "id"), ("rightKey", "id_pel"), ("kind", "left")], left, right);

        var output = _executor.Output<TabularData>(context);

        Assert.Equal(2, output.RowCount);
        Assert.Null(output.Value(1, "kota"));
    }

    [Fact]
    public void GroupBy_averages_within_each_group()
    {
        var input = TabularData.WithColumns("kota", "nilai");
        input.AddRow("Bandung", 10);
        input.AddRow("Bandung", 20);
        input.AddRow("Jakarta", 40);

        var context = TestContext.For("tf.groupBy",
            [("groupBy", "kota"), ("aggregateColumn", "nilai"), ("aggregate", "mean")], input);

        var output = _executor.Output<TabularData>(context);

        Assert.Equal(2, output.RowCount);
        Assert.Contains(output.Rows, r => Convert.ToString(r[0]) == "Bandung"
                                          && TabularData.ToNumber(r[1]) == 15);
    }

    [Fact]
    public void Normalize_minmax_maps_the_column_onto_zero_to_one()
    {
        var input = TabularData.WithColumns("nilai");
        foreach (var value in (int[])[10, 20, 30])
        {
            input.AddRow(value);
        }

        var context = TestContext.For("tf.normalize",
            [("columns", "nilai"), ("method", "minmax")], input);

        var output = _executor.Output<TabularData>(context);

        Assert.Equal(0d, TabularData.ToNumber(output.Value(0, "nilai")));
        Assert.Equal(0.5d, TabularData.ToNumber(output.Value(1, "nilai")));
        Assert.Equal(1d, TabularData.ToNumber(output.Value(2, "nilai")));
    }

    [Fact]
    public void EncodeCategories_one_hot_adds_one_column_per_value_and_drops_the_source()
    {
        var input = TabularData.WithColumns("warna");
        foreach (var colour in (string[])["merah", "biru", "merah"])
        {
            input.AddRow(colour);
        }

        var context = TestContext.For("tf.oneHot",
            [("columns", "warna"), ("method", "oneHot")], input);

        var output = _executor.Output<TabularData>(context);

        Assert.False(output.Has("warna"));
        Assert.True(output.Has("warna_merah"));
        Assert.True(output.Has("warna_biru"));
        Assert.Equal(1d, TabularData.ToNumber(output.Value(0, "warna_merah")));
        Assert.Equal(0d, TabularData.ToNumber(output.Value(0, "warna_biru")));
    }

    /// <summary>
    /// One-hot on a high-cardinality column would silently add thousands of columns and make
    /// the next step unusable. Refusing with a suggestion beats producing that quietly.
    /// </summary>
    [Fact]
    public void EncodeCategories_refuses_one_hot_on_a_high_cardinality_column()
    {
        var input = TabularData.WithColumns("id");

        for (var i = 0; i < 250; i++)
        {
            input.AddRow($"id-{i}");
        }

        var context = TestContext.For("tf.oneHot",
            [("columns", "id"), ("method", "oneHot")], input);

        var error = Assert.Throws<InvalidOperationException>(() =>
            _executor.ExecuteAsync(context).GetAwaiter().GetResult());

        Assert.Contains("distinct values", error.Message);
        Assert.Contains("hashing", error.Message);
    }

    [Fact]
    public void EditMetadata_renames_and_converts_the_stored_values_too()
    {
        var input = TabularData.WithColumns("angka_teks");
        input.AddRow("42");

        var context = TestContext.For("tf.editMetadata",
            [("column", "angka_teks"), ("newName", "angka"), ("dataType", "Numeric")], input);

        var output = _executor.Output<TabularData>(context);

        Assert.True(output.Has("angka"));
        Assert.Equal(ColumnDataType.Numeric, output.Column("angka")!.Value.DataType);
        Assert.Equal(42d, output.Value(0, "angka"));
    }

    [Fact]
    public void A_transform_never_mutates_the_table_it_was_given()
    {
        // Node outputs are shared with downstream nodes; mutating an input would corrupt a
        // sibling branch of the graph.
        var input = TabularData.WithColumns("a", "b");
        input.AddRow(1, 2);

        var before = input.ToCsv();

        _executor.Output<TabularData>(
            TestContext.For("tf.selectColumns", [("columns", "a"), ("mode", "keep")], input));

        Assert.Equal(before, input.ToCsv());
    }
}
