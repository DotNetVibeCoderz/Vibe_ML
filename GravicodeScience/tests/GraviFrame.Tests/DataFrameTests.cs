using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;
using Xunit;

namespace Gravicode.Science.Tests.GraviFrame;

public class DataFrameTests
{
    private const string SalesCsv = """
        Date,Category,Region,Sales,Units
        2024-01-01,Electronics,North,1200.5,10
        2024-01-01,Furniture,North,800,4
        2024-01-02,Electronics,South,950.25,8
        2024-01-02,Furniture,South,,3
        2024-01-03,Electronics,North,1100,9
        2024-01-03,Furniture,South,760.75,5
        """;

    private static DataFrame Sales() => DataFrame.ParseCsv(SalesCsv);

    [Fact]
    public void ParseCsv_InfersColumnTypes()
    {
        var df = Sales();

        Assert.Equal(6, df.RowCount);
        Assert.Equal(5, df.ColumnCount);
        Assert.Equal(DataType.DateTime, df["Date"].DataType);
        Assert.Equal(DataType.Text, df["Category"].DataType);
        Assert.Equal(DataType.Numeric, df["Sales"].DataType);
        Assert.Equal(DataType.Numeric, df["Units"].DataType);
    }

    [Fact]
    public void ParseCsv_TreatsEmptyFieldsAsMissing()
    {
        var df = Sales();
        Assert.True(df["Sales"].IsMissing(3));
        Assert.Equal(1, df["Sales"].MissingCount);
        Assert.Equal(5, df["Sales"].Count);
    }

    [Fact]
    public void SplitLine_HonoursQuotedFieldsAndEscapedQuotes()
    {
        // a , "b,c" , "say ""hi""" , d
        var fields = CsvReader.SplitLine("a,\"b,c\",\"say \"\"hi\"\"\",d", ",");
        Assert.Equal(4, fields.Length);
        Assert.Equal("b,c", fields[1]);
        Assert.Equal("say \"hi\"", fields[2]);
    }

    [Fact]
    public void ColumnAccess_FailsClearlyOnTheWrongType()
    {
        var df = Sales();
        Assert.Throws<InvalidOperationException>(() => df.Numeric("Category"));
        Assert.Throws<KeyNotFoundException>(() => df["Nope"]);
    }

    [Fact]
    public void HeadAndTail_SliceRows()
    {
        var df = Sales();
        Assert.Equal(2, df.Head(2).RowCount);
        Assert.Equal(2, df.Tail(2).RowCount);
        Assert.Equal(760.75, df.Tail(1).Numeric("Sales")[0]);
    }

    [Fact]
    public void Filter_KeepsMatchingRows()
    {
        var df = Sales();
        var north = df.Filter(row => row.String("Region") == "North");
        Assert.Equal(3, north.RowCount);

        var big = df.FilterBy("Sales", v => v > 1000);
        Assert.Equal(2, big.RowCount);
    }

    [Fact]
    public void SortBy_OrdersAscendingAndDescending()
    {
        var df = Sales().DropMissing(subset: ["Sales"]);

        var ascending = df.SortBy("Sales");
        Assert.Equal(760.75, ascending.Numeric("Sales")[0]);

        var descending = df.SortBy("Sales", ascending: false);
        Assert.Equal(1200.5, descending.Numeric("Sales")[0]);
    }

    [Fact]
    public void SortBy_MultipleKeysIsLexicographic()
    {
        var df = Sales().SortBy([("Category", true), ("Sales", false)]);
        Assert.Equal("Electronics", df.Text("Category")[0]);
        Assert.Equal(1200.5, df.Numeric("Sales")[0]);
    }

    [Fact]
    public void WithColumn_AddsAndReplaces()
    {
        var df = Sales();
        var withPrice = df.WithColumn("UnitPrice", i => df.Numeric("Sales")[i] / df.Numeric("Units")[i]);

        Assert.Equal(6, withPrice.ColumnCount);
        Assert.Equal(120.05, withPrice.Numeric("UnitPrice")[0], 6);

        var replaced = withPrice.WithColumn(new NumericSeries("UnitPrice", new double[6]));
        Assert.Equal(6, replaced.ColumnCount);
        Assert.Equal(0.0, replaced.Numeric("UnitPrice")[0]);
    }

    [Fact]
    public void DropAndSelect_ChooseColumns()
    {
        var df = Sales();
        Assert.Equal(3, df.Drop("Date", "Region").ColumnCount);

        var selected = df.SelectColumns("Sales", "Category");
        Assert.Equal(["Sales", "Category"], selected.ColumnNames);
    }

    [Fact]
    public void DropMissing_RemovesIncompleteRows()
    {
        var df = Sales();
        Assert.Equal(5, df.DropMissing().RowCount);
    }

    [Fact]
    public void FillMissing_ReplacesNaNInNumericColumns()
    {
        var filled = Sales().FillMissing(0.0);
        Assert.Equal(0.0, filled.Numeric("Sales")[3]);
        Assert.Equal(0, filled["Sales"].MissingCount);
    }

    [Fact]
    public void Describe_ReportsPerColumnStatistics()
    {
        var described = Sales().Describe();
        Assert.Equal(8, described.RowCount);
        Assert.Contains("Sales", described.ColumnNames);

        // Row 0 is "count"; Sales has five present values.
        Assert.Equal(5.0, described.Numeric("Sales")[0]);
    }

    [Fact]
    public void Concat_StacksFramesWithMatchingColumns()
    {
        var df = Sales();
        var doubled = DataFrame.Concat([df, df]);
        Assert.Equal(12, doubled.RowCount);
        Assert.Equal(df.ColumnCount, doubled.ColumnCount);
    }

    [Fact]
    public void ToNdArray_ExportsNumericColumns()
    {
        var matrix = Sales().SelectColumns("Sales", "Units").FillMissing(0).ToNdArray();
        Assert.Equal(6, matrix.Shape[0]);
        Assert.Equal(2, matrix.Shape[1]);
        Assert.Equal(1200.5, matrix[0, 0]);
    }

    [Fact]
    public void CsvRoundTrip_PreservesValues()
    {
        var df = Sales();
        var csv = CsvWriter.ToCsv(df);
        var reparsed = DataFrame.ParseCsv(csv);

        Assert.Equal(df.RowCount, reparsed.RowCount);
        Assert.Equal(df.ColumnCount, reparsed.ColumnCount);
        Assert.Equal(1200.5, reparsed.Numeric("Sales")[0]);
        Assert.True(reparsed["Sales"].IsMissing(3));
    }

    [Fact]
    public void MemoryMappedRead_MatchesTheStreamingRead()
    {
        var path = Path.Combine(Path.GetTempPath(), $"gravi-{Guid.NewGuid():N}.csv");
        try
        {
            File.WriteAllText(path, SalesCsv);
            var streamed = DataFrame.ReadCsv(path);
            var mapped = DataFrame.ReadCsvMemoryMapped(path);

            Assert.Equal(streamed.RowCount, mapped.RowCount);
            Assert.Equal(streamed.ColumnNames, mapped.ColumnNames);
            for (var i = 0; i < streamed.RowCount; i++)
                Assert.Equal(streamed.Numeric("Units")[i], mapped.Numeric("Units")[i]);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void MemoryMappedRead_DoesNotInventARowFromPagePadding()
    {
        // A memory mapping is rounded up to the system page size. Unless the view is limited to
        // the real file length, the trailing NUL bytes decode as an extra line - so this checks
        // a file whose length is deliberately not a multiple of 4096.
        var path = Path.Combine(Path.GetTempPath(), $"gravi-{Guid.NewGuid():N}.csv");
        try
        {
            var rows = Enumerable.Range(0, 900).Select(i => $"{i},{i * 1.5}");
            File.WriteAllText(path, "id,value\n" + string.Join('\n', rows) + "\n");

            var streamed = DataFrame.ReadCsv(path);
            var mapped = DataFrame.ReadCsvMemoryMapped(path);

            Assert.Equal(900, streamed.RowCount);
            Assert.Equal(streamed.RowCount, mapped.RowCount);
            Assert.Equal(0, mapped["id"].MissingCount);
            Assert.Equal(899 * 1.5, mapped.Numeric("value")[899], 9);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void ParquetRoundTrip_PreservesTypesAndValues()
    {
        var path = Path.Combine(Path.GetTempPath(), $"gravi-{Guid.NewGuid():N}.parquet");
        try
        {
            var df = Sales();
            ParquetIO.Write(df, path);
            var loaded = ParquetIO.Read(path);

            Assert.Equal(df.RowCount, loaded.RowCount);
            Assert.Equal(df.ColumnCount, loaded.ColumnCount);
            Assert.Equal(1200.5, loaded.Numeric("Sales")[0], 6);
            Assert.True(loaded["Sales"].IsMissing(3));
            Assert.Equal("Electronics", loaded.Text("Category")[0]);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }
}
