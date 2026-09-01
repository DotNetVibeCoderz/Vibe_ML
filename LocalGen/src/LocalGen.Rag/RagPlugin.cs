using System.ComponentModel;
using System.Text;
using Microsoft.SemanticKernel;

namespace LocalGen.Rag;

/// <summary>
/// Exposes the document index to the model as kernel functions.
/// </summary>
/// <remarks>
/// Retrieval is offered as a tool rather than being stuffed into every prompt, so the model only
/// pays for context when the question actually calls for it — and can search again with a better
/// query if the first attempt misses.
/// </remarks>
public sealed class RagPlugin(RagService rag)
{
    [KernelFunction("search_documents")]
    [Description(
        "Searches the ingested documents and returns the most relevant passages with their sources. " +
        "Use this whenever a question might be answered by the user's own documents.")]
    public async Task<string> SearchAsync(
        [Description("What to search for, phrased as a question or keywords")] string query,
        [Description("How many passages to return")] int limit = 5,
        [Description("Collection to search; leave empty for the default")] string? collection = null,
        CancellationToken cancellationToken = default)
    {
        var hits = await rag
            .SearchAsync(query, limit, string.IsNullOrWhiteSpace(collection) ? RagService.DefaultCollection : collection, cancellationToken)
            .ConfigureAwait(false);

        if (hits.Count == 0)
        {
            return "No relevant passages were found. The documents may not cover this, or nothing has been ingested yet.";
        }

        var builder = new StringBuilder();

        for (var i = 0; i < hits.Count; i++)
        {
            var hit = hits[i];
            builder.Append('[').Append(i + 1).Append("] ")
                   .Append(hit.Record.Source)
                   .Append(" — relevance ").Append(hit.Score.ToString("P0")).AppendLine();
            builder.AppendLine(hit.Record.Text).AppendLine();
        }

        return builder.ToString();
    }

    [KernelFunction("ingest_file")]
    [Description("Reads a file (PDF, markdown, HTML, text or code) and adds it to the searchable document index.")]
    public async Task<string> IngestFileAsync(
        [Description("Path to the file")] string path,
        [Description("Collection to add it to; leave empty for the default")] string? collection = null,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var result = await rag
                .IngestFileAsync(
                    path,
                    string.IsNullOrWhiteSpace(collection) ? RagService.DefaultCollection : collection,
                    cancellationToken)
                .ConfigureAwait(false);

            return result.ChunkCount == 0
                ? $"'{result.Source}' contained no extractable text."
                : $"Indexed '{result.Source}' as {result.ChunkCount} chunk(s) " +
                  $"({result.CharacterCount:N0} characters, {result.Duration.TotalSeconds:N1}s).";
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            return $"Error: {ex.Message}";
        }
    }

    [KernelFunction("ingest_directory")]
    [Description("Adds every supported file in a directory to the searchable document index.")]
    public async Task<string> IngestDirectoryAsync(
        [Description("Path to the directory")] string path,
        [Description("Include subdirectories")] bool recursive = true,
        [Description("Collection to add them to; leave empty for the default")] string? collection = null,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var results = await rag
                .IngestDirectoryAsync(
                    path,
                    string.IsNullOrWhiteSpace(collection) ? RagService.DefaultCollection : collection,
                    recursive,
                    cancellationToken)
                .ConfigureAwait(false);

            if (results.Count == 0)
            {
                return $"No supported files were found under '{path}'.";
            }

            var chunks = results.Sum(static r => r.ChunkCount);
            return $"Indexed {results.Count} file(s) as {chunks} chunk(s).";
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            return $"Error: {ex.Message}";
        }
    }

    [KernelFunction("forget_source")]
    [Description("Removes a previously ingested document from the index.")]
    public async Task<string> ForgetAsync(
        [Description("File name of the document to remove")] string source,
        CancellationToken cancellationToken = default)
    {
        var removed = await rag.RemoveSourceAsync(source, cancellationToken).ConfigureAwait(false);

        return removed > 0
            ? $"Removed {removed} chunk(s) that came from '{source}'."
            : $"Removed '{source}' from the index.";
    }
}
