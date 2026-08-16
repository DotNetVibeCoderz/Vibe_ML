using System.Text;

namespace LocalGen.Rag.Documents;

/// <summary>A slice of a document, sized to be embedded and retrieved on its own.</summary>
public sealed record TextChunk
{
    public required string Text { get; init; }

    /// <summary>Zero-based position of this chunk within its document.</summary>
    public int Index { get; init; }

    /// <summary>Character offset of the chunk in the original document.</summary>
    public int Offset { get; init; }
}

/// <summary>
/// Splits documents into overlapping chunks.
/// </summary>
/// <remarks>
/// Chunking is what determines whether retrieval returns something useful. Splitting on a fixed
/// character count alone cuts sentences in half and strands the answer across two chunks, so this
/// prefers the largest natural boundary that fits — paragraph, then sentence, then word — and
/// carries an overlap forward so a fact spanning a boundary still appears whole in one chunk.
/// </remarks>
public static class TextChunker
{
    public static IReadOnlyList<TextChunk> Split(string text, int chunkSize = 1000, int overlap = 200)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return [];
        }

        if (chunkSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(chunkSize), "Chunk size must be positive.");
        }

        // Overlap is capped at half the chunk. Beyond that each chunk barely advances past the
        // last, so a large document turns into thousands of near-duplicate chunks — expensive to
        // embed and worse to retrieve from. Half is also the most overlap that is ever useful.
        overlap = Math.Clamp(overlap, 0, chunkSize / 2);

        var normalized = text.Replace("\r\n", "\n").Trim();

        if (normalized.Length <= chunkSize)
        {
            return [new TextChunk { Text = normalized, Index = 0, Offset = 0 }];
        }

        var chunks = new List<TextChunk>();
        var position = 0;
        var index = 0;

        while (position < normalized.Length)
        {
            var remaining = normalized.Length - position;
            var length = Math.Min(chunkSize, remaining);
            var end = position + length;

            // Only look for a boundary when the chunk was actually truncated.
            if (end < normalized.Length)
            {
                var boundary = FindBoundary(normalized, position, end);
                if (boundary > position)
                {
                    end = boundary;
                }
            }

            var slice = normalized[position..end].Trim();

            if (slice.Length > 0)
            {
                chunks.Add(new TextChunk { Text = slice, Index = index++, Offset = position });
            }

            if (end >= normalized.Length)
            {
                break;
            }

            // Step forward by the chunk minus the overlap, never by less than one character.
            position = Math.Max(position + 1, end - overlap);
        }

        return chunks;
    }

    /// <summary>
    /// Finds the best split point within the last quarter of the window: a paragraph break if
    /// there is one, else a sentence end, else a word boundary.
    /// </summary>
    private static int FindBoundary(string text, int start, int end)
    {
        var searchFrom = start + (end - start) * 3 / 4;

        var paragraph = text.LastIndexOf("\n\n", end - 1, end - searchFrom, StringComparison.Ordinal);
        if (paragraph > searchFrom)
        {
            return paragraph + 2;
        }

        for (var i = end - 1; i > searchFrom; i--)
        {
            // A sentence ends with terminal punctuation followed by whitespace.
            if (text[i] is '.' or '!' or '?' or '\n' &&
                i + 1 < text.Length && char.IsWhiteSpace(text[i + 1]))
            {
                return i + 1;
            }
        }

        for (var i = end - 1; i > searchFrom; i--)
        {
            if (char.IsWhiteSpace(text[i]))
            {
                return i + 1;
            }
        }

        return end;
    }

    /// <summary>
    /// Splits markdown so each chunk stays under one heading, which keeps retrieved passages
    /// self-describing instead of arriving without their section title.
    /// </summary>
    public static IReadOnlyList<TextChunk> SplitMarkdown(
        string markdown,
        int chunkSize = 1000,
        int overlap = 200)
    {
        var sections = new List<(string Heading, StringBuilder Body, int Offset)>();
        var currentHeading = string.Empty;
        var current = new StringBuilder();
        var sectionOffset = 0;
        var offset = 0;

        foreach (var line in markdown.Replace("\r\n", "\n").Split('\n'))
        {
            if (line.StartsWith('#'))
            {
                if (current.Length > 0)
                {
                    sections.Add((currentHeading, current, sectionOffset));
                    current = new StringBuilder();
                }

                currentHeading = line.TrimStart('#').Trim();
                sectionOffset = offset;
            }

            current.AppendLine(line);
            offset += line.Length + 1;
        }

        if (current.Length > 0)
        {
            sections.Add((currentHeading, current, sectionOffset));
        }

        var chunks = new List<TextChunk>();
        var index = 0;

        foreach (var (heading, body, sectionStart) in sections)
        {
            foreach (var chunk in Split(body.ToString(), chunkSize, overlap))
            {
                // The heading is prepended so an isolated chunk still says what it is about.
                var text = string.IsNullOrEmpty(heading) || chunk.Text.StartsWith('#')
                    ? chunk.Text
                    : $"## {heading}\n\n{chunk.Text}";

                chunks.Add(new TextChunk
                {
                    Text = text,
                    Index = index++,
                    Offset = sectionStart + chunk.Offset
                });
            }
        }

        return chunks;
    }
}
