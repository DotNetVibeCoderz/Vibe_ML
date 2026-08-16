using System.Text;
using AngleSharp;
using LocalGen.Core;
using Microsoft.Extensions.Logging;
using UglyToad.PdfPig;

namespace LocalGen.Rag.Documents;

/// <summary>Text pulled out of a file, ready to be chunked.</summary>
public sealed record ExtractedDocument
{
    public required string FileName { get; init; }

    public required string Text { get; init; }

    /// <summary>Detected kind: <c>pdf</c>, <c>markdown</c>, <c>html</c>, <c>text</c>, <c>image</c>.</summary>
    public required string Kind { get; init; }

    /// <summary>Page count for paginated formats; zero otherwise.</summary>
    public int PageCount { get; init; }

    public IReadOnlyDictionary<string, string> Metadata { get; init; } =
        new Dictionary<string, string>();
}

/// <summary>
/// Turns files into plain text for ingestion.
/// </summary>
/// <remarks>
/// Only formats that can be read without an external converter are handled here. Images are
/// deliberately not OCR'd: shipping an OCR engine would be a large dependency for a feature most
/// users would not reach for, so an image is recorded with its metadata and left for a
/// vision-capable model to interpret at query time.
/// </remarks>
public sealed class DocumentExtractor(ILogger<DocumentExtractor> logger)
{
    /// <summary>Extensions treated as plain text, including source code.</summary>
    private static readonly HashSet<string> TextExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".txt", ".log", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml", ".toml", ".ini",
        ".cs", ".fs", ".vb", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go",
        ".rs", ".rb", ".php", ".c", ".h", ".cpp", ".hpp", ".sql", ".sh", ".ps1", ".bat"
    };

    private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"
    };

    /// <summary>Guards against loading an enormous file into memory during a bulk ingest.</summary>
    private const long MaxFileSizeBytes = 100L * 1024 * 1024;

    public bool CanExtract(string path)
    {
        var extension = Path.GetExtension(path);

        return extension.Equals(".pdf", StringComparison.OrdinalIgnoreCase)
               || extension is ".md" or ".markdown"
               || extension is ".html" or ".htm"
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
            ".md" or ".markdown" => await ExtractTextAsync(path, "markdown", cancellationToken).ConfigureAwait(false),
            ".html" or ".htm" => await ExtractHtmlAsync(path, cancellationToken).ConfigureAwait(false),
            _ when ImageExtensions.Contains(extension) => DescribeImage(path, info),
            _ when TextExtensions.Contains(extension) => await ExtractTextAsync(path, "text", cancellationToken).ConfigureAwait(false),
            _ => throw new LocalGenException(
                $"'{extension}' files cannot be ingested. Supported: PDF, markdown, HTML, images " +
                "and plain-text formats including source code.")
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
        CancellationToken cancellationToken) => new()
        {
            FileName = Path.GetFileName(path),
            Text = await File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false),
            Kind = kind
        };

    private static async Task<ExtractedDocument> ExtractHtmlAsync(
        string path,
        CancellationToken cancellationToken)
    {
        var html = await File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false);

        var context = BrowsingContext.New(Configuration.Default);
        using var document = await context
            .OpenAsync(request => request.Content(html), cancellationToken)
            .ConfigureAwait(false);

        foreach (var selector in (string[])["script", "style", "noscript"])
        {
            foreach (var element in document.QuerySelectorAll(selector).ToList())
            {
                element.Remove();
            }
        }

        return new ExtractedDocument
        {
            FileName = Path.GetFileName(path),
            Text = document.Body?.TextContent.Trim() ?? string.Empty,
            Kind = "html",
            Metadata = string.IsNullOrWhiteSpace(document.Title)
                ? new Dictionary<string, string>()
                : new Dictionary<string, string> { ["title"] = document.Title }
        };
    }

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
