using FluentAssertions;
using LocalGen.Core;
using LocalGen.Rag;
using LocalGen.Rag.Documents;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace LocalGen.Core.Tests;

/// <summary>
/// Ingestion across the document formats LocalGen accepts.
/// </summary>
/// <remarks>
/// The extractor is resolved from <see cref="ServiceCollectionExtensions.AddLocalGenRag"/> rather
/// than constructed directly, because half of what these tests protect is the registration: the
/// Excel and PowerPoint converters are separate packages that have to be added explicitly, and
/// forgetting either produces a working build whose only symptom is "format not supported" when
/// someone finally ingests a spreadsheet.
/// </remarks>
public class DocumentExtractorTests : IClassFixture<DocumentFixtures>
{
    private readonly DocumentFixtures _files;
    private readonly DocumentExtractor _extractor;

    public DocumentExtractorTests(DocumentFixtures files)
    {
        _files = files;

        var services = new ServiceCollection();
        services.AddLogging();
        services.AddLocalGenRag();

        _extractor = services.BuildServiceProvider().GetRequiredService<DocumentExtractor>();
    }

    [Fact]
    public async Task A_word_document_keeps_its_headings_and_its_table()
    {
        var document = await _extractor.ExtractAsync(_files.Word());

        document.Kind.Should().Be("docx");
        document.IsMarkdown.Should().BeTrue();

        document.Text.Should().Contain("# Quarterly Report");
        document.Text.Should().Contain("## Regional breakdown");
        document.Text.Should().Contain("| Region | Revenue |");
        document.Text.Should().Contain("| APAC | 4.2M |");
    }

    [Fact]
    public async Task A_workbook_becomes_one_table_per_sheet()
    {
        var document = await _extractor.ExtractAsync(_files.Workbook());

        document.Kind.Should().Be("xlsx");
        document.Text.Should().Contain("Sales");
        document.Text.Should().Contain("| APAC | 4200000 |");

        // The second sheet matters: a converter that stops at the first one loses data silently.
        document.Text.Should().Contain("Figures are unaudited.");
    }

    [Fact]
    public async Task A_presentation_keeps_its_slide_text()
    {
        var document = await _extractor.ExtractAsync(_files.Presentation());

        document.Kind.Should().Be("pptx");
        document.Text.Should().Contain("Slide 1");
        document.Text.Should().Contain("Ship the installers");
    }

    [Fact]
    public async Task An_epub_is_read_chapter_and_all()
    {
        var document = await _extractor.ExtractAsync(_files.Book());

        document.Kind.Should().Be("epub");
        document.IsMarkdown.Should().BeTrue();

        document.Text.Should().Contain("# Running Models Locally");
        document.Text.Should().Contain("Choosing a quantization");
        document.Text.Should().Contain("is the usual trade between size and quality");
    }

    [Fact]
    public async Task Markdown_metacharacters_come_back_escaped()
    {
        // Recorded rather than corrected. The HTML-derived converters escape characters that
        // would otherwise be markdown syntax, so an identifier like Q4_K_M is stored as Q4\_K\_M.
        // It costs nothing at retrieval — an embedding is unbothered — but a passage quoted back
        // to a model carries the backslashes, so it is a property to know about rather than a
        // surprise to discover in an answer.
        var path = _files.Text(
            "quantization.html",
            "<html><body><p>Prefer Q4_K_M over Q3_K_S.</p></body></html>");

        var document = await _extractor.ExtractAsync(path);

        document.Text.Should().Contain(@"Q4\_K\_M");
    }

    [Fact]
    public async Task Rich_text_is_read_as_markdown()
    {
        var path = _files.Text(
            "notes.rtf",
            @"{\rtf1\ansi\deff0 {\b Release notes}\par Multi-GPU splitting is configurable.\par }");

        var document = await _extractor.ExtractAsync(path);

        document.Kind.Should().Be("rtf");
        document.Text.Should().Contain("Release notes");
        document.Text.Should().Contain("Multi-GPU splitting is configurable.");
    }

    [Fact]
    public async Task A_csv_becomes_a_markdown_table()
    {
        var path = _files.Text("sales.csv", "region,revenue\nAPAC,4200000\nEMEA,3100000\n");

        var document = await _extractor.ExtractAsync(path);

        document.Kind.Should().Be("csv");
        document.Text.Should().Contain("| region | revenue |");
        document.Text.Should().Contain("| EMEA | 3100000 |");
    }

    [Fact]
    public async Task Html_keeps_its_structure_and_drops_its_scripts()
    {
        var path = _files.Text(
            "engine.html",
            """
            <html><head><title>Engine notes</title>
            <style>body { color: red }</style>
            <script>trackVisitor("engine")</script></head>
            <body><h1>Engine notes</h1><p>Layer splitting buys <b>capacity</b>, not speed.</p>
            <table><tr><th>Mode</th><th>Use</th></tr><tr><td>layer</td><td>default</td></tr></table>
            </body></html>
            """);

        var document = await _extractor.ExtractAsync(path);

        document.Kind.Should().Be("html");
        document.IsMarkdown.Should().BeTrue();

        document.Text.Should().Contain("# Engine notes");
        document.Text.Should().Contain("| Mode | Use |");

        // Script and style bodies are not content, and embedding them wastes the chunk on markup.
        document.Text.Should().NotContain("trackVisitor");
        document.Text.Should().NotContain("color: red");
    }

    [Fact]
    public async Task A_pdf_still_carries_its_page_markers()
    {
        // LocalGen reads PDFs itself rather than through the converter, for exactly this: the
        // page number is what makes a citation from a 300-page manual worth anything, and a
        // general-purpose converter emits a horizontal rule between pages instead.
        var document = await _extractor.ExtractAsync(_files.Pdf());

        document.Kind.Should().Be("pdf");
        document.IsMarkdown.Should().BeFalse();
        document.PageCount.Should().Be(2);

        document.Text.Should().Contain("[page 1]");
        document.Text.Should().Contain("[page 2]");
        document.Text.Should().Contain("Splitting across GPUs");
    }

    [Fact]
    public async Task Source_code_is_ingested_verbatim()
    {
        const string source = "public sealed class Engine\n{\n    // no conversion wanted here\n}\n";
        var path = _files.Text("Engine.cs", source);

        var document = await _extractor.ExtractAsync(path);

        document.Kind.Should().Be("text");
        document.IsMarkdown.Should().BeFalse();
        document.Text.Should().Be(source);
    }

    [Fact]
    public async Task Json_is_left_alone_rather_than_reformatted()
    {
        // A fenced code block would re-print the file without making any of it easier to find.
        const string json = """{"region":"APAC","revenue":4200000}""";
        var path = _files.Text("row.json", json);

        var document = await _extractor.ExtractAsync(path);

        document.Kind.Should().Be("text");
        document.Text.Should().Be(json);
    }

    [Fact]
    public async Task A_markdown_file_is_marked_for_heading_aware_chunking()
    {
        var path = _files.Text("guide.md", "# Guide\n\nSome prose.\n");

        var document = await _extractor.ExtractAsync(path);

        document.Kind.Should().Be("markdown");
        document.IsMarkdown.Should().BeTrue();
    }

    [Fact]
    public async Task An_image_is_recorded_for_a_vision_model_rather_than_read()
    {
        var path = _files.Text("diagram.png", "not really a png");

        var document = await _extractor.ExtractAsync(path);

        document.Kind.Should().Be("image");
        document.Text.Should().Contain("vision-capable model");
    }

    [Fact]
    public async Task An_unsupported_format_is_refused_with_the_list_of_supported_ones()
    {
        var path = _files.Text("archive.zzz", "binary-ish");

        var act = async () => await _extractor.ExtractAsync(path);

        (await act.Should().ThrowAsync<LocalGenException>())
            .Which.Message.Should().Contain(".docx").And.Contain(".pdf");
    }

    [Fact]
    public async Task A_corrupt_document_names_the_file_and_the_reason()
    {
        // A bulk ingest tells "skip this file" from "abandon the run" by catching exceptions, so
        // a conversion failure has to surface as one rather than as empty text.
        var path = _files.Text("broken.docx", "this is not a zip archive");

        var act = async () => await _extractor.ExtractAsync(path);

        (await act.Should().ThrowAsync<LocalGenException>())
            .Which.Message.Should().Contain("broken.docx");
    }

    [Fact]
    public async Task A_converted_document_chunks_on_the_headings_the_conversion_recovered()
    {
        // This is the point of routing converted documents through the markdown chunker: a
        // passage retrieved from the middle of a report arrives labelled with the section it
        // came from, instead of as an anonymous run of prose.
        var document = await _extractor.ExtractAsync(_files.Word());

        var chunks = TextChunker.SplitMarkdown(document.Text, chunkSize: 200, overlap: 40);

        chunks.Should().NotBeEmpty();

        // Every chunk either opens with its own heading or has its section's prepended, so no
        // passage reaches the model as unattributed prose.
        chunks.Should().OnlyContain(c => c.Text.TrimStart().StartsWith('#'));
        chunks.Should().Contain(c => c.Text.Contains("Regional breakdown"));
    }

    [Theory]
    [InlineData("report.docx")]
    [InlineData("sales.xlsx")]
    [InlineData("deck.pptx")]
    [InlineData("book.epub")]
    [InlineData("notes.rtf")]
    [InlineData("data.csv")]
    [InlineData("page.html")]
    public void The_new_formats_are_offered_to_a_directory_scan(string name)
    {
        // IngestDirectoryAsync filters on CanExtract, so a format missing from it is invisible to
        // a folder ingest even though ingesting the file directly would work.
        _extractor.CanExtract(name).Should().BeTrue();
    }
}
