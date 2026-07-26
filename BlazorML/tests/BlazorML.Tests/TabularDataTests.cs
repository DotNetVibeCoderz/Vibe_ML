using BlazorML.Core.Data;
using BlazorML.Core.Domain;

namespace BlazorML.Tests;

public class TabularDataTests
{
    [Fact]
    public void InferTypes_reads_numeric_categorical_and_text_apart()
    {
        // Enough rows for the "mostly repeated values" signal to separate a category from free
        // text. On four rows three cities genuinely do look like free text, and the inference
        // is right to say so.
        var table = TabularData.WithColumns("harga", "kota", "ulasan");
        string[] cities = ["Bandung", "Jakarta", "Surabaya"];

        for (var i = 0; i < 30; i++)
        {
            table.AddRow(
                (1000 + i * 17).ToString(),
                cities[i % cities.Length],
                $"Ulasan nomor {i} yang panjang dan berbeda isinya dari ulasan yang lain sama sekali.");
        }

        table.InferTypes();

        Assert.Equal(ColumnDataType.Numeric, table.Column("harga")!.Value.DataType);
        Assert.Equal(ColumnDataType.Categorical, table.Column("kota")!.Value.DataType);
        Assert.Equal(ColumnDataType.Text, table.Column("ulasan")!.Value.DataType);
    }

    /// <summary>
    /// Regression test. "ya"/"tidak" reads as boolean to a human, but ML.NET's TextLoader cannot
    /// parse those words when a column is declared Boolean — declaring it that way blew the
    /// loader up mid-run and failed two of the sample experiments. They must stay categorical.
    /// </summary>
    [Theory]
    [InlineData("ya", "tidak")]
    [InlineData("yes", "no")]
    [InlineData("Y", "N")]
    public void InferTypes_does_not_call_localised_yes_no_a_boolean(string yes, string no)
    {
        var table = TabularData.WithColumns("pakai_internet");

        for (var i = 0; i < 10; i++)
        {
            table.AddRow(i % 2 == 0 ? yes : no);
        }

        table.InferTypes();

        Assert.NotEqual(ColumnDataType.Boolean, table.Column("pakai_internet")!.Value.DataType);
    }

    [Fact]
    public void InferTypes_still_calls_real_boolean_literals_boolean()
    {
        var table = TabularData.WithColumns("aktif");

        foreach (var value in (string[])["true", "false", "true", "false", "true"])
        {
            table.AddRow(value);
        }

        table.InferTypes();

        Assert.Equal(ColumnDataType.Boolean, table.Column("aktif")!.Value.DataType);
    }

    [Fact]
    public void AddColumn_keeps_every_row_rectangular()
    {
        var table = TabularData.WithColumns("a");
        table.AddRow("1");
        table.AddRow("2");

        table.AddColumn("b");
        table.AddColumn("c");

        Assert.All(table.Rows, row => Assert.Equal(3, row.Length));
    }

    [Fact]
    public void RemoveColumn_drops_the_right_values_not_just_the_header()
    {
        var table = TabularData.WithColumns("a", "b", "c");
        table.AddRow("a1", "b1", "c1");
        table.AddRow("a2", "b2", "c2");

        table.RemoveColumn("b");

        Assert.Equal(["a", "c"], table.Columns.Select(c => c.Name));
        Assert.Equal("a1", table.Text(0, "a"));
        Assert.Equal("c1", table.Text(0, "c"));
        Assert.Equal("c2", table.Text(1, "c"));
    }

    [Fact]
    public void AddColumn_disambiguates_a_name_that_is_already_taken()
    {
        var table = TabularData.WithColumns("Score");
        table.AddColumn("Score");
        table.AddColumn("Score");

        Assert.Equal(["Score", "Score_2", "Score_3"], table.Columns.Select(c => c.Name));
    }

    [Fact]
    public void ToCsv_escapes_delimiters_quotes_and_newlines()
    {
        var table = TabularData.WithColumns("teks", "n");
        table.AddRow("berisi, koma", 1);
        table.AddRow("berisi \"kutip\"", 2);
        table.AddRow("berisi\nbaris baru", 3);

        var csv = table.ToCsv();

        Assert.Contains("\"berisi, koma\"", csv);
        Assert.Contains("\"berisi \"\"kutip\"\"\"", csv);
        Assert.Contains("\"berisi\nbaris baru\"", csv);
    }

    [Fact]
    public void Clone_does_not_share_row_storage_with_the_original()
    {
        var table = TabularData.WithColumns("a");
        table.AddRow("original");

        var copy = table.Clone();
        copy.Rows[0][0] = "changed";

        Assert.Equal("original", table.Text(0, "a"));
    }

    [Fact]
    public void Profile_reports_missing_values_and_numeric_range()
    {
        var table = TabularData.WithColumns("nilai");
        table.AddRow(10);
        table.AddRow(20);
        table.AddRow((object?)null);
        table.AddRow(30);
        table.AddRow("");

        var profile = table.Profile();
        var column = profile.Column("nilai")!;

        Assert.Equal(5, profile.RowCount);
        Assert.Equal(2, column.MissingCount);
        Assert.Equal(10, column.Min);
        Assert.Equal(30, column.Max);
        Assert.Equal(20, column.Mean);
    }

    [Fact]
    public void FromDictionaries_unions_keys_across_rows_with_different_shapes()
    {
        var rows = new List<Dictionary<string, object?>>
        {
            new() { ["a"] = 1 },
            new() { ["a"] = 2, ["b"] = "x" },
            new() { ["c"] = true }
        };

        var table = TabularData.FromDictionaries(rows);

        Assert.Equal(["a", "b", "c"], table.Columns.Select(c => c.Name));
        Assert.Equal(3, table.RowCount);
        Assert.Equal("x", table.Text(1, "b"));
        Assert.Null(table.Value(0, "b"));
    }

    [Theory]
    [InlineData("1.5", 1.5)]
    [InlineData("-3", -3d)]
    [InlineData("bukan angka", null)]
    [InlineData("", null)]
    public void ToNumber_parses_invariantly_and_returns_null_when_it_cannot(string input, double? expected)
    {
        Assert.Equal(expected, TabularData.ToNumber(input));
    }
}
