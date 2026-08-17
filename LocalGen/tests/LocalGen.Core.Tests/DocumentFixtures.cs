using System.IO.Compression;
using ClosedXML.Excel;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using UglyToad.PdfPig.Content;
using UglyToad.PdfPig.Core;
using UglyToad.PdfPig.Fonts.Standard14Fonts;
using UglyToad.PdfPig.Writer;
using D = DocumentFormat.OpenXml.Drawing;
using P = DocumentFormat.OpenXml.Presentation;
using W = DocumentFormat.OpenXml.Wordprocessing;

namespace LocalGen.Core.Tests;

/// <summary>
/// Builds the document files the extractor tests run against, in a directory of their own.
/// </summary>
/// <remarks>
/// Real files rather than stubs, because the thing under test is whether a converter can read
/// what a word processor writes. A hand-rolled XML fragment would pass tests that a genuine
/// .docx would fail, which is the wrong way round.
/// </remarks>
public sealed class DocumentFixtures : IDisposable
{
    public DocumentFixtures()
    {
        Directory = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "localgen-doc-tests-" + Guid.NewGuid().ToString("N")[..8]);
        System.IO.Directory.CreateDirectory(Directory);
    }

    public string Directory { get; }

    public void Dispose()
    {
        try
        {
            System.IO.Directory.Delete(Directory, recursive: true);
        }
        catch (IOException)
        {
            // A locked file must not fail an otherwise passing test run.
        }
    }

    public string At(string name) => System.IO.Path.Combine(Directory, name);

    public string Text(string name, string content)
    {
        var path = At(name);
        File.WriteAllText(path, content);
        return path;
    }

    /// <summary>A Word document with two heading levels, prose and a table.</summary>
    public string Word(string name = "report.docx")
    {
        var path = At(name);

        using var document = WordprocessingDocument.Create(path, WordprocessingDocumentType.Document);
        var main = document.AddMainDocumentPart();
        main.Document = new W.Document(new W.Body());
        var body = main.Document.Body!;

        static W.Paragraph Styled(string text, string style) =>
            new(new W.ParagraphProperties(new W.ParagraphStyleId { Val = style }),
                new W.Run(new W.Text(text)));

        body.Append(Styled("Quarterly Report", "Heading1"));
        body.Append(new W.Paragraph(new W.Run(new W.Text("Revenue grew by 18% against the prior quarter."))));
        body.Append(Styled("Regional breakdown", "Heading2"));
        body.Append(new W.Paragraph(new W.Run(new W.Text("Asia Pacific led growth, followed by Europe."))));

        var table = new W.Table(new W.TableProperties(new W.TableStyle { Val = "TableGrid" }));
        foreach (var row in new[] { new[] { "Region", "Revenue" }, new[] { "APAC", "4.2M" } })
        {
            var tableRow = new W.TableRow();
            foreach (var cell in row)
            {
                tableRow.Append(new W.TableCell(new W.Paragraph(new W.Run(new W.Text(cell)))));
            }

            table.Append(tableRow);
        }

        body.Append(table);
        return path;
    }

    /// <summary>A workbook with two sheets, so the per-sheet split is observable.</summary>
    public string Workbook(string name = "sales.xlsx")
    {
        var path = At(name);

        using var workbook = new XLWorkbook();
        var sales = workbook.AddWorksheet("Sales");
        sales.Cell(1, 1).Value = "Region";
        sales.Cell(1, 2).Value = "Revenue";
        sales.Cell(2, 1).Value = "APAC";
        sales.Cell(2, 2).Value = 4200000;

        workbook.AddWorksheet("Notes").Cell(1, 1).Value = "Figures are unaudited.";
        workbook.SaveAs(path);

        return path;
    }

    /// <summary>A one-slide deck with a title and two bullets.</summary>
    public string Presentation(string name = "roadmap.pptx")
    {
        var path = At(name);

        using var presentation = PresentationDocument.Create(path, PresentationDocumentType.Presentation);
        var presentationPart = presentation.AddPresentationPart();
        presentationPart.Presentation = new P.Presentation();

        var slidePart = presentationPart.AddNewPart<SlidePart>();
        slidePart.Slide = new P.Slide(
            new P.CommonSlideData(
                new P.ShapeTree(
                    new P.NonVisualGroupShapeProperties(
                        new P.NonVisualDrawingProperties { Id = 1, Name = "" },
                        new P.NonVisualGroupShapeDrawingProperties(),
                        new P.ApplicationNonVisualDrawingProperties()),
                    new P.GroupShapeProperties(),
                    new P.Shape(
                        new P.NonVisualShapeProperties(
                            new P.NonVisualDrawingProperties { Id = 2, Name = "Title" },
                            new P.NonVisualShapeDrawingProperties(),
                            new P.ApplicationNonVisualDrawingProperties()),
                        new P.ShapeProperties(),
                        new P.TextBody(
                            new D.BodyProperties(),
                            new D.ListStyle(),
                            new D.Paragraph(new D.Run(new D.Text("Roadmap"))),
                            new D.Paragraph(new D.Run(new D.Text("Ship the installers"))),
                            new D.Paragraph(new D.Run(new D.Text("Verify multi-GPU"))))))),
            new P.ColorMapOverride(new D.MasterColorMapping()));

        presentationPart.Presentation.Append(
            new P.SlideIdList(new P.SlideId { Id = 256U, RelationshipId = presentationPart.GetIdOfPart(slidePart) }),
            new P.SlideSize { Cx = 9144000, Cy = 6858000 },
            new P.NotesSize { Cx = 6858000, Cy = 9144000 });

        return path;
    }

    /// <summary>A minimal but valid EPUB 2 book with one chapter.</summary>
    /// <remarks>
    /// Assembled by hand because there is no writer library in the solution, and the parts are
    /// fixed: the <c>mimetype</c> entry has to come first and be stored uncompressed, which is
    /// the one rule an EPUB reader checks before anything else.
    /// </remarks>
    public string Book(string name = "handbook.epub")
    {
        var path = At(name);

        using var stream = File.Create(path);
        using var archive = new ZipArchive(stream, ZipArchiveMode.Create);

        void Write(string entryName, string content, CompressionLevel level = CompressionLevel.Optimal)
        {
            var entry = archive.CreateEntry(entryName, level);
            using var writer = new StreamWriter(entry.Open());
            writer.Write(content);
        }

        Write("mimetype", "application/epub+zip", CompressionLevel.NoCompression);

        Write("META-INF/container.xml",
            """
            <?xml version="1.0"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles>
                <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
              </rootfiles>
            </container>
            """);

        Write("OEBPS/content.opf",
            """
            <?xml version="1.0" encoding="utf-8"?>
            <package xmlns="http://www.idpf.org/2007/opf" unique-identifier="bookid" version="2.0">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:title>Running Models Locally</dc:title>
                <dc:creator>Gravicode Studios</dc:creator>
                <dc:identifier id="bookid">urn:uuid:6b2f0d1e-3a4c-4f21-9c7d-8e5a1b2c3d4e</dc:identifier>
                <dc:language>en</dc:language>
              </metadata>
              <manifest>
                <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
                <item id="ch1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine toc="ncx">
                <itemref idref="ch1"/>
              </spine>
            </package>
            """);

        Write("OEBPS/toc.ncx",
            """
            <?xml version="1.0" encoding="utf-8"?>
            <ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
              <head><meta name="dtb:uid" content="urn:uuid:6b2f0d1e-3a4c-4f21-9c7d-8e5a1b2c3d4e"/></head>
              <docTitle><text>Running Models Locally</text></docTitle>
              <navMap>
                <navPoint id="ch1" playOrder="1">
                  <navLabel><text>Choosing a quantization</text></navLabel>
                  <content src="chapter1.xhtml"/>
                </navPoint>
              </navMap>
            </ncx>
            """);

        Write("OEBPS/chapter1.xhtml",
            """
            <?xml version="1.0" encoding="utf-8"?>
            <html xmlns="http://www.w3.org/1999/xhtml">
              <head><title>Choosing a quantization</title></head>
              <body>
                <h1>Choosing a quantization</h1>
                <p>Q4_K_M is the usual trade between size and quality.</p>
              </body>
            </html>
            """);

        return path;
    }

    /// <summary>A two-page PDF, one distinguishable line per page.</summary>
    public string Pdf(string name = "manual.pdf")
    {
        var path = At(name);

        var builder = new PdfDocumentBuilder();
        var font = builder.AddStandard14Font(Standard14Font.Helvetica);

        foreach (var (number, line) in new[] { (1, "Loading a model"), (2, "Splitting across GPUs") })
        {
            var page = builder.AddPage(PageSize.A4);
            page.AddText(line, 12, new PdfPoint(25, 700), font);
            _ = number;
        }

        File.WriteAllBytes(path, builder.Build());
        return path;
    }
}
