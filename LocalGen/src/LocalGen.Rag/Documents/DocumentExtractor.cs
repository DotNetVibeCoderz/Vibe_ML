using System.Text;
using ElBruno.MarkItDotNet;
using LocalGen.Core;
using Microsoft.Extensions.Logging;
using UglyToad.PdfPig;

namespace LocalGen.Rag.Documents;

/// <summary>Text pulled out of a file, ready to be chunked.</summary>
public sealed record ExtractedDocument
{
    public required string FileName { get; init; }

    public required string Text { get; init; }

    /// <summary>
    /// The source format: <c>pdf</c>, <c>docx</c>, <c>xlsx</c>, <c>pptx</c>, <c>epub</c>,
    /// <c>rtf</c>, <c>csv</c>, <c>html</c>, <c>markdown</c>, <c>text</c> or <c>image</c>.
    /// </summary>
    public required string Kind { get; init; }

    /// <summary>
    /// Whether <see cref="Text"/> is markdown, and so can be split on its headings rather than
    /// on blank lines.
    /// </summary>
    /// <remarks>
    /// Separate from <see cref="Kind"/> because the two answer different questions: a converted
    /// Word document has <c>Kind</c> <c>docx</c> — which is what a citation should say — but its
    /// text is markdown, and chunking it as prose would throw away the heading structure the
    /// conversion just recovered.
    /// </remarks>
    public bool IsMarkdown { get; init; }

    /// <summary>Page count for paginated formats; zero otherwise.</summary>
    public int PageCount { get; init; }

    public IReadOnlyDictionary<string, string> Metadata { get; init; } =
        new Dictionary<string, string>();
}

/// <summary>
/// Turns files into text for ingestion.
/// </summary>
/// <remarks>
/// <para>
/// Two extraction paths, chosen per format rather than per convenience. Anything with structure
/// worth keeping — Word, Excel, PowerPoint, EPUB, RTF, HTML, CSV — goes through MarkItDotNet and
/// comes back as markdown, so headings survive into the chunker and tables stay tables. Formats
/// LocalGen already reads better than a general converter keep their own path: a PDF for its page
/// markers, plain text and source code because there is nothing to convert, and an image because
/// what matters about it is not text at all.
/// </para>
/// <para>
/// Images are deliberately not OCR'd: shipping an OCR engine would be a large dependency for a
/// feature most users would not reach for, so an image is recorded with its metadata and left for
/// a vision-capable model to interpret at query time.
/// </para>
/// </remarks>
public sealed class DocumentExtractor(MarkdownService markdown, ILogger<DocumentExtractor> logger)
{
    /// <summary>Extensions treated as plain text, including source code.</summary>
    /// <remarks>
    /// JSON, XML and YAML stay here rather than being converted. MarkItDotNet would wrap them in a
    /// fenced code block, which re-formats the file without making any of it easier to retrieve —
    /// the criterion for taking the conversion path is that it recovers structure, not that a
    /// converter exists.
    /// </remarks>
    private static readonly HashSet<string> TextExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".txt", ".log", ".json", ".xml", ".yaml", ".yml", ".toml", ".ini",
        ".cs", ".fs", ".vb", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go",
        ".rs", ".rb", ".php", ".c", ".h", ".cpp", ".hpp", ".sql", ".sh", ".ps1", ".bat"
    };

    /// <summary>Formats converted to markdown, mapped to the kind recorded against the chunks.</summary>
    private static readonly Dictionary<string, string> ConvertedExtensions =
        new(StringComparer.OrdinalIgnoreCase)
        {
            [".docx"] = "docx",
            [".xlsx"] = "xlsx",
            [".pptx"] = "pptx",
            [".epub"] = "epub",
            [".rtf"] = "rtf",
            [".csv"] = "csv",
            [".tsv"] = "csv",
            [".html"] = "html",
            [".htm"] = "html"
        };

    private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".svg"
    };

    /// <summary>Guards against loading an enormous file into memory during a bulk ingest.</summary>
    public const long MaxFileSizeBytes = 100L * 1024 * 1024;

    /// <summary>Every extension that can be ingested, for messages and directory scans.</summary>
    public static IReadOnlyCollection<string> SupportedExtensions { get; } =
        [.. new[] { ".pdf", ".md", ".markdown" }
            .Concat(ConvertedExtensions.Keys)
            .Concat(TextExtensions)
            .Concat(ImageExtensions)
            .Order(StringComparer.Ordinal)];

    public bool CanExtract(string path)
    {
        var extension = Path.GetExtension(path);

        return extension.Equals(".pdf", StringComparison.OrdinalIgnoreCase)
               || extension is ".md" or ".markdown"
               || ConvertedExtensions.ContainsKey(extension)
               || TextExtensions.Contains(extension)
               || ImageExtensions.Contains(extension);
    }

    public async Task<ExtractedDocument> ExtractAsync(
        string path,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(path))
        {
            throw new LocalGenException($"No file at '{path}'.");
        }

        var info = new FileInfo(path);
        if (info.Length > MaxFileSizeBytes)
        {
            throw new LocalGenException(
                $"'{info.Name}' is {info.Length / 1024 / 1024} MB, over the " +
                $"{MaxFileSizeBytes / 1024 / 1024} MB ingestion limit.");
        }

        var extension = Path.GetExtension(path).ToLowerInvariant();

        return extension switch
        {
            ".pdf" => await ExtractPdfAsync(path, cancellationToken).ConfigureAwait(false),
            ".md" or ".markdown" => await ExtractTextAsync(path, "markdown", isMarkdown: true, cancellationToken).ConfigureAwait(false),
            _ when ConvertedExtensions.TryGetValue(extension, out var kind) =>
                await ConvertAsync(path, kind, cancellationToken).ConfigureAwait(false),
            _ when ImageExtensions.Contains(extension) => DescribeImage(path, info),
            _ when TextExtensions.Contains(extension) =>
                await ExtractTextAsync(path, "text", isMarkdown: false, cancellationToken).ConfigureAwait(false),
            _ => throw new LocalGenException(
                $"'{extension}' files cannot be ingested. Supported: " +
                $"{string.Join(", ", SupportedExtensions)}.")
        };
    }

    /// <summary>
    /// Converts a structured document to markdown through MarkItDotNet.
    /// </summary>
    /// <remarks>
    /// The converter reports failure in its result rather than by throwing, and a bulk ingest
    /// distinguishes files it should skip from a run it should abandon by catching exceptions —
    /// so a failed conversion is turned back into one here, carrying the converter's own reason.
    /// </remarks>
    private async Task<ExtractedDocument> ConvertAsync(
        string path,
        string kind,
        CancellationToken cancellationToken)
    {
        var result = await markdown.ConvertAsync(path, cancellationToken).ConfigureAwait(false);
        var fileName = Path.GetFileName(path);

        if (!result.Success)
        {
            throw new LocalGenException(
                $"'{fileName}' could not be read as {kind}: {result.ErrorMessage}");
        }

        var text = result.Markdown ?? string.Empty;
        var metadata = new Dictionary<string, string> { ["format"] = kind };

        if (result.Metadata?.WordCount is { } words)
        {
            metadata["wordCount"] = words.ToString();
        }

        logger.LogInformation(
            "Converted {File} to {Chars:N0} characters of markdown", fileName, text.Length);

        return new ExtractedDocument
        {
            FileName = fileName,
            Text = text.Trim(),
            Kind = kind,
            IsMarkdown = true,
            PageCount = result.Metadata?.PageCount ?? 0,
            Metadata = metadata
        };
    }

    private Task<ExtractedDocument> ExtractPdfAsync(string path, CancellationToken cancellationToken) =>
        Task.Run(() =>
        {
            using var document = PdfDocument.Open(path);
            var builder = new StringBuilder();
            var pageCount = 0;

            foreach (var page in document.GetPages())
            {
                cancellationToken.ThrowIfCancellationRequested();
                pageCount++;

                // The page marker keeps citations meaningful after chunking.
                builder.Append("\n\n[page ").Append(page.Number).AppendLine("]");
                builder.AppendLine(page.Text);
            }

            logger.LogInformation("Extracted {Pages} page(s) from {File}", pageCount, Path.GetFileName(path));

            var metadata = new Dictionary<string, string>();

            if (!string.IsNullOrWhiteSpace(document.Information.Title))
            {
                metadata["title"] = document.Information.Title;
            }

            if (!string.IsNullOrWhiteSpace(document.Information.Author))
            {
                metadata["author"] = document.Information.Author;
            }

            return new ExtractedDocument
            {
                FileName = Path.GetFileName(path),
                Text = builder.ToString().Trim(),
                Kind = "pdf",
                PageCount = pageCount,
                Metadata = metadata
            };
        }, cancellationToken);

    private static async Task<ExtractedDocument> ExtractTextAsync(
        string path,
        string kind,
        bool isMarkdown,
        CancellationToken cancellationToken) => new()
        {
            FileName = Path.GetFileName(path),
            Text = await File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false),
            Kind = kind,
            IsMarkdown = isMarkdown
        };

    /// <summary>
    /// Records an image as a text placeholder. There is nothing to embed, but the entry means a
    /// search can still surface "there is a diagram called X in this folder".
    /// </summary>
    private static ExtractedDocument DescribeImage(string path, FileInfo info) => new()
    {
        FileName = info.Name,
        Kind = "image",
        Text = $"Image file: {info.Name} ({info.Length / 1024:N0} KB, {info.Extension.TrimStart('.').ToUpperInvariant()}). " +
               "Attach this image to a conversation with a vision-capable model to have its contents described.",
        Metadata = new Dictionary<string, string>
        {
            ["path"] = info.FullName,
            ["sizeBytes"] = info.Length.ToString()
        }
    };
}
