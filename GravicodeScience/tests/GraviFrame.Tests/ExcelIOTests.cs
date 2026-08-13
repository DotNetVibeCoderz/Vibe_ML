using System.IO.Compression;
using System.Text;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;
using Xunit;

namespace Gravicode.Science.Tests.GraviFrame;

/// <summary>
/// Tests for the xlsx reader and writer.
/// </summary>
/// <remarks>
/// A write-then-read round trip only proves the two halves agree with each other, so most of these
/// read a workbook assembled here by hand from the OPC spec — shared strings, absent cells, a date
/// serial with a style, columns past Z. Those are the cases a reader written from intuition gets
/// wrong, and a round trip cannot catch any of them.
/// </remarks>
public class ExcelIOTests
{
    /// <summary>Assembles an xlsx from raw parts, exactly as a spreadsheet application would.</summary>
    private static MemoryStream Workbook(string sheetXml, string? sharedStrings = null,
        string? styles = null, string sheetName = "Data")
    {
        var stream = new MemoryStream();

        using (var archive = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: true))
        {
            void Add(string path, string content)
            {
                var entry = archive.CreateEntry(path);
                using var writer = new StreamWriter(entry.Open(), new UTF8Encoding(false));
                writer.Write(content);
            }

            Add("[Content_Types].xml",
                """<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>""");

            Add("xl/workbook.xml",
                $"""
                <?xml version="1.0"?>
                <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                  <sheets><sheet name="{sheetName}" sheetId="1" r:id="rId1"/></sheets>
                </workbook>
                """);

            Add("xl/_rels/workbook.xml.rels",
                """
                <?xml version="1.0"?>
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
                </Relationships>
                """);

            Add("xl/worksheets/sheet1.xml",
                $"""
                <?xml version="1.0"?>
                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                  <sheetData>{sheetXml}</sheetData>
                </worksheet>
                """);

            if (sharedStrings is not null) Add("xl/sharedStrings.xml", sharedStrings);
            if (styles is not null) Add("xl/styles.xml", styles);
        }

        stream.Position = 0;
        return stream;
    }

    private const string SharedStrings =
        """
        <?xml version="1.0"?>
        <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="5" uniqueCount="5">
          <si><t>name</t></si>
          <si><t>score</t></si>
          <si><t>Ada</t></si>
          <si><t>Linus</t></si>
          <si><t>city</t></si>
        </sst>
        """;

    [Fact]
    public void SharedStringsAreResolvedToTheirText()
    {
        // Almost all text in a real workbook lives in the shared table and the cell holds only an
        // index, so a reader that ignores it produces a sheet full of small integers.
        using var stream = Workbook(
            """
            <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
            <row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>92.5</v></c></row>
            <row r="3"><c r="A3" t="s"><v>3</v></c><c r="B3"><v>81</v></c></row>
            """,
            SharedStrings);

        var frame = ExcelReader.Read(stream);

        Assert.Equal(["name", "score"], frame.ColumnNames);
        Assert.Equal(2, frame.RowCount);
        Assert.Equal("Ada", ((TextSeries)frame["name"])[0]);
        Assert.Equal(92.5, frame.Numeric("score")[0]);
    }

    [Fact]
    public void AbsentCellsLeaveGapsRatherThanShiftingLaterColumns()
    {
        // The single most damaging bug a naive reader has. Row 2 writes only A and C; reading the
        // cells positionally would put "gamma" under column B and quietly shift the whole row.
        using var stream = Workbook(
            """
            <row r="1"><c r="A1" t="inlineStr"><is><t>a</t></is></c><c r="B1" t="inlineStr"><is><t>b</t></is></c><c r="C1" t="inlineStr"><is><t>c</t></is></c></row>
            <row r="2"><c r="A2"><v>1</v></c><c r="C2"><v>3</v></c></row>
            <row r="3"><c r="A3"><v>4</v></c><c r="B3"><v>5</v></c><c r="C3"><v>6</v></c></row>
            """);

        var frame = ExcelReader.Read(stream);

        Assert.Equal(["a", "b", "c"], frame.ColumnNames);
        Assert.Equal(1.0, frame.Numeric("a")[0]);
        Assert.True(double.IsNaN(frame.Numeric("b")[0]));    // genuinely empty, not shifted
        Assert.Equal(3.0, frame.Numeric("c")[0]);
        Assert.Equal(5.0, frame.Numeric("b")[1]);
    }

    [Fact]
    public void DateSerialsWithADateStyleBecomeDates()
    {
        // There is no date cell type: a date is a plain number whose style points at a date format.
        // Missing that turns a date column into five-digit integers.
        const string styles =
            """
            <?xml version="1.0"?>
            <styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <cellXfs count="2">
                <xf numFmtId="0"/>
                <xf numFmtId="14"/>
              </cellXfs>
            </styleSheet>
            """;

        using var stream = Workbook(
            """
            <row r="1"><c r="A1" t="inlineStr"><is><t>when</t></is></c></row>
            <row r="2"><c r="A2" s="1"><v>43831</v></c></row>
            <row r="3"><c r="A3" s="1"><v>44562</v></c></row>
            """,
            styles: styles);

        var frame = ExcelReader.Read(stream);

        Assert.Equal(DataType.DateTime, frame["when"].DataType);

        // Serial 43831 is 2020-01-01 and 44562 is 2022-01-01 — the values Excel itself shows.
        // Getting these right is what proves the epoch handles the 1900 leap-year bug.
        Assert.Equal(new DateTime(2020, 1, 1), ((DateTimeSeries)frame["when"])[0]);
        Assert.Equal(new DateTime(2022, 1, 1), ((DateTimeSeries)frame["when"])[1]);
    }

    [Fact]
    public void ANumberWithoutADateStyleStaysANumber()
    {
        // The other half of the same rule: the style is the only thing that makes 43831 a date.
        using var stream = Workbook(
            """
            <row r="1"><c r="A1" t="inlineStr"><is><t>n</t></is></c></row>
            <row r="2"><c r="A2"><v>43831</v></c></row>
            """);

        var frame = ExcelReader.Read(stream);

        Assert.Equal(DataType.Numeric, frame["n"].DataType);
        Assert.Equal(43831.0, frame.Numeric("n")[0]);
    }

    [Fact]
    public void ColumnReferencesPastZAreDecodedInBaseTwentySix()
    {
        // A is 1 and AA is 27: base 26 with no zero digit. Treating it as ordinary base 26 puts
        // every column past Z one place out.
        using var stream = Workbook(
            """
            <row r="1"><c r="A1" t="inlineStr"><is><t>first</t></is></c><c r="AA1" t="inlineStr"><is><t>last</t></is></c></row>
            <row r="2"><c r="A2"><v>1</v></c><c r="AA2"><v>27</v></c></row>
            """);

        var frame = ExcelReader.Read(stream);

        // A is column 0 and AA is column 26, so the sheet is 27 wide.
        Assert.Equal(27, frame.ColumnNames.Count);
        Assert.Equal("first", frame.ColumnNames[0]);
        Assert.Equal("last", frame.ColumnNames[26]);
        Assert.Equal(27.0, frame.Numeric("last")[0]);
    }

    [Fact]
    public void BooleansAndErrorsAreRecognised()
    {
        // Errors are missing data: #DIV/0! is not the string "#DIV/0!".
        using var stream = Workbook(
            """
            <row r="1"><c r="A1" t="inlineStr"><is><t>flag</t></is></c><c r="B1" t="inlineStr"><is><t>calc</t></is></c></row>
            <row r="2"><c r="A2" t="b"><v>1</v></c><c r="B2" t="e"><v>#DIV/0!</v></c></row>
            <row r="3"><c r="A3" t="b"><v>0</v></c><c r="B3"><v>2</v></c></row>
            """);

        var frame = ExcelReader.Read(stream);

        Assert.Equal(DataType.Boolean, frame["flag"].DataType);
        Assert.Equal(true, ((BooleanSeries)frame["flag"])[0]);
        Assert.Equal(false, ((BooleanSeries)frame["flag"])[1]);
        Assert.True(frame["calc"].IsMissing(0));
    }

    [Fact]
    public void AFormulaCellYieldsItsCachedResult()
    {
        // No formula evaluation, and none needed: the file records the last computed value, which
        // is what the caller wanted.
        using var stream = Workbook(
            """
            <row r="1"><c r="A1" t="inlineStr"><is><t>total</t></is></c></row>
            <row r="2"><c r="A2"><f>SUM(B1:B9)</f><v>42</v></c></row>
            """);

        Assert.Equal(42.0, ExcelReader.Read(stream).Numeric("total")[0]);
    }

    [Fact]
    public void ASheetCanBeChosenByName()
    {
        using var stream = Workbook("<row r=\"1\"><c r=\"A1\" t=\"inlineStr\"><is><t>x</t></is></c></row>",
            sheetName: "Results");

        var frame = ExcelReader.Read(stream, new ExcelOptions { SheetName = "Results" });
        Assert.Equal(["x"], frame.ColumnNames);
    }

    [Fact]
    public void AskingForASheetThatIsNotThereListsTheOnesThatAre()
    {
        using var stream = Workbook("<row r=\"1\"/>", sheetName: "Results");

        var error = Assert.Throws<ArgumentException>(
            () => ExcelReader.Read(stream, new ExcelOptions { SheetName = "Missing" }));

        Assert.Contains("Results", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void HeaderlessSheetsGetPositionalNames()
    {
        using var stream = Workbook(
            """
            <row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row>
            <row r="2"><c r="A2"><v>3</v></c><c r="B2"><v>4</v></c></row>
            """);

        var frame = ExcelReader.Read(stream, new ExcelOptions { HasHeader = false });

        Assert.Equal(["column_0", "column_1"], frame.ColumnNames);
        Assert.Equal(2, frame.RowCount);
        Assert.Equal(1.0, frame.Numeric("column_0")[0]);
    }

    [Fact]
    public void MaxRowsStopsEarly()
    {
        using var stream = Workbook(
            """
            <row r="1"><c r="A1" t="inlineStr"><is><t>n</t></is></c></row>
            <row r="2"><c r="A2"><v>1</v></c></row>
            <row r="3"><c r="A3"><v>2</v></c></row>
            <row r="4"><c r="A4"><v>3</v></c></row>
            """);

        Assert.Equal(2, ExcelReader.Read(stream, new ExcelOptions { MaxRows = 2 }).RowCount);
    }

    [Fact]
    public void SheetNamesAreListedInWorkbookOrder()
    {
        var path = Path.Combine(Path.GetTempPath(), $"gravi_{Guid.NewGuid():N}.xlsx");
        try
        {
            using (var source = Workbook("<row r=\"1\"/>", sheetName: "Alpha"))
            using (var file = File.Create(path)) source.CopyTo(file);

            Assert.Equal(["Alpha"], ExcelReader.SheetNames(path));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void SomethingThatIsNotAWorkbookIsRejected()
    {
        using var stream = new MemoryStream();
        using (var archive = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: true))
            archive.CreateEntry("readme.txt");
        stream.Position = 0;

        Assert.Throws<InvalidDataException>(() => ExcelReader.Read(stream));
    }

    // ------------------------------------------------------------------ writing

    [Fact]
    public void AWrittenWorkbookReadsBackWithEveryTypeIntact()
    {
        var frame = new DataFrame(
        [
            new TextSeries("label", ["alpha", "beta", null]),
            new NumericSeries("value", [1.5, -2.25, double.NaN]),
            new BooleanSeries("ok", [true, false, null]),
            new DateTimeSeries("when", [new DateTime(2024, 3, 1), new DateTime(1999, 12, 31), null]),
        ]);

        using var stream = new MemoryStream();
        ExcelWriter.Write(frame, stream);
        stream.Position = 0;

        var back = ExcelReader.Read(stream);

        Assert.Equal(3, back.RowCount);
        Assert.Equal(["label", "value", "ok", "when"], back.ColumnNames);

        Assert.Equal("alpha", ((TextSeries)back["label"])[0]);
        Assert.Equal(1.5, back.Numeric("value")[0]);
        Assert.Equal(-2.25, back.Numeric("value")[1]);
        Assert.Equal(true, ((BooleanSeries)back["ok"])[0]);
        Assert.Equal(new DateTime(2024, 3, 1), ((DateTimeSeries)back["when"])[0]);
        Assert.Equal(new DateTime(1999, 12, 31), ((DateTimeSeries)back["when"])[1]);

        // And every missing value came back missing rather than as a zero or an empty string.
        foreach (var name in back.ColumnNames) Assert.True(back[name].IsMissing(2));
    }

    [Fact]
    public void AWrittenWorkbookHasThePartsTheFormatRequires()
    {
        // Checked against the spec rather than against the reader, so a file this writer produces
        // will open in Excel and not merely in its own round trip.
        var frame = new DataFrame([new NumericSeries("a", [1])]);

        using var stream = new MemoryStream();
        ExcelWriter.Write(frame, stream);
        stream.Position = 0;

        using var archive = new ZipArchive(stream, ZipArchiveMode.Read);
        var names = archive.Entries.Select(e => e.FullName).ToArray();

        Assert.Contains("[Content_Types].xml", names);
        Assert.Contains("_rels/.rels", names);
        Assert.Contains("xl/workbook.xml", names);
        Assert.Contains("xl/_rels/workbook.xml.rels", names);
        Assert.Contains("xl/worksheets/sheet1.xml", names);
        Assert.Contains("xl/styles.xml", names);
    }

    [Fact]
    public void TextIsXmlEscapedOnTheWayOut()
    {
        // An ampersand written raw makes the whole part unparseable, which no round trip through a
        // lenient reader would reveal.
        var frame = new DataFrame([new TextSeries("t", ["a & b", "<tag>", "quote \" here"])]);

        using var stream = new MemoryStream();
        ExcelWriter.Write(frame, stream);
        stream.Position = 0;

        var back = ExcelReader.Read(stream);
        Assert.Equal("a & b", ((TextSeries)back["t"])[0]);
        Assert.Equal("<tag>", ((TextSeries)back["t"])[1]);
        Assert.Equal("quote \" here", ((TextSeries)back["t"])[2]);
    }

    [Fact]
    public void AWideFrameGetsCorrectReferencesPastColumnZ()
    {
        // The writer's half of the base-26 rule, checked by reading the references back.
        var columns = Enumerable.Range(0, 30)
            .Select(i => (Series)new NumericSeries($"c{i}", [i]))
            .ToList();

        using var stream = new MemoryStream();
        ExcelWriter.Write(new DataFrame(columns), stream);
        stream.Position = 0;

        var back = ExcelReader.Read(stream);

        Assert.Equal(30, back.ColumnNames.Count);
        Assert.Equal("c29", back.ColumnNames[29]);
        Assert.Equal(29.0, back.Numeric("c29")[0]);
    }

    [Fact]
    public void RoundTripsThroughAFileOnDisk()
    {
        var path = Path.Combine(Path.GetTempPath(), $"gravi_{Guid.NewGuid():N}.xlsx");
        try
        {
            var frame = new DataFrame(
            [
                new TextSeries("city", ["Bandung", "Jakarta"]),
                new NumericSeries("population", [2.5e6, 10.6e6]),
            ]);

            ExcelWriter.Write(frame, path, sheetName: "Cities");

            Assert.Equal(["Cities"], ExcelReader.SheetNames(path));

            var back = ExcelReader.Read(path);
            Assert.Equal("Bandung", ((TextSeries)back["city"])[0]);
            Assert.Equal(10.6e6, back.Numeric("population")[1]);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
