using System.Text;

namespace Gravicode.Science.DaskNet;

/// <summary>Graph shapes several collections share.</summary>
internal static class GraphBuilders
{
    /// <summary>
    /// A reduction tree over <paramref name="leaves"/>: groups of <paramref name="splitEvery"/>
    /// combined level by level until one task remains (Dask's <c>split_every</c>).
    /// </summary>
    /// <remarks>
    /// Groups are contiguous and in input order, so the combine order - and with floating point
    /// the result, to the last bit - is fixed by the graph, not by which task finishes first.
    /// </remarks>
    public static (List<TaskSpec> Tasks, TaskKey Root) TreeReduce(string name, IReadOnlyList<TaskKey> leaves,
        int splitEvery, Func<TaskKey, IReadOnlyList<TaskKey>, TaskSpec> combine, Func<TaskKey>? newKey = null)
    {
        if (leaves.Count == 0) throw new ArgumentException("Nothing to reduce.", nameof(leaves));
        if (splitEvery < 2) throw new ArgumentOutOfRangeException(nameof(splitEvery), "split_every must be at least 2.");

        var tasks = new List<TaskSpec>();
        var level = leaves.ToList();
        var index = 0;

        // Even a single leaf gets one combine: the combine is what finalises the result.
        do
        {
            var next = new List<TaskKey>();
            for (var start = 0; start < level.Count; start += splitEvery)
            {
                var group = level.Skip(start).Take(splitEvery).ToArray();
                var key = newKey?.Invoke() ?? new TaskKey(name, index++);
                tasks.Add(combine(key, group));
                next.Add(key);
            }
            level = next;
        } while (level.Count > 1);

        return (tasks, level[0]);
    }

    /// <summary>Paths matching a file, a directory, or a glob in the file-name part (<c>data/*.csv</c>).</summary>
    public static string[] ExpandPaths(string pattern)
    {
        if (File.Exists(pattern)) return [Path.GetFullPath(pattern)];
        if (Directory.Exists(pattern))
            return Directory.GetFiles(pattern).Order(StringComparer.Ordinal).ToArray();

        var directory = Path.GetDirectoryName(pattern);
        if (string.IsNullOrEmpty(directory)) directory = ".";
        var filePattern = Path.GetFileName(pattern);
        if (!Directory.Exists(directory)) throw new DirectoryNotFoundException(directory);

        var files = Directory.GetFiles(directory, filePattern).Order(StringComparer.Ordinal).ToArray();
        return files.Length > 0 ? files : throw new FileNotFoundException($"No files match '{pattern}'.");
    }

    /// <summary>Paths for several patterns, each expanded and the whole list de-duplicated.</summary>
    public static string[] ExpandPaths(IEnumerable<string> patterns)
        => patterns.SelectMany(ExpandPaths).Distinct(StringComparer.Ordinal).ToArray();

    /// <summary>
    /// Splits a file into byte ranges of about <paramref name="blockSize"/>. Ranges are not yet
    /// line-aligned; <see cref="TextBlocks.ReadLines"/> aligns them when reading.
    /// </summary>
    public static List<(long Start, long End)> ByteRanges(long length, long blockSize, long firstByte = 0)
    {
        var ranges = new List<(long, long)>();
        if (blockSize <= 0 || length - firstByte <= blockSize)
        {
            ranges.Add((firstByte, length));
            return ranges;
        }
        for (var start = firstByte; start < length; start += blockSize)
            ranges.Add((start, Math.Min(length, start + blockSize)));
        return ranges;
    }
}

/// <summary>Reads the lines of one byte range of a text file.</summary>
public static class TextBlocks
{
    /// <summary>
    /// The lines that <i>start</i> inside <c>[start, end)</c>, without their terminators.
    /// </summary>
    /// <remarks>
    /// The rule that makes adjacent blocks tile the file exactly: a line belongs to the block its
    /// first byte falls in. A block that does not begin at 0 therefore skips the partial line it
    /// lands in (owned by the previous block), and the last line it owns may run past
    /// <paramref name="end"/>. Splitting on a byte count is safe for UTF-8: '\n' (0x0A) never
    /// occurs inside a multi-byte sequence.
    /// </remarks>
    public static List<string> ReadLines(string path, long start, long end)
    {
        var lines = new List<string>();
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16);
        var position = start;

        if (start > 0)
        {
            // Back up one byte: if the previous byte is '\n', a line starts exactly at `start`.
            stream.Position = start - 1;
            int b;
            while ((b = stream.ReadByte()) != -1)
            {
                position++;
                if (b == '\n') break;
            }
            position = stream.Position;
        }
        else
        {
            stream.Position = 0;
        }

        var buffer = new MemoryStream();
        var bytes = new byte[1 << 16];
        var lineStart = position;
        var pending = false;

        while (lineStart < end || pending)
        {
            var read = stream.Read(bytes, 0, bytes.Length);
            if (read == 0) break;
            var offset = 0;
            for (var i = 0; i < read; i++)
            {
                if (bytes[i] != (byte)'\n') continue;
                buffer.Write(bytes, offset, i - offset);
                if (lineStart < end) lines.Add(Decode(buffer));
                buffer.SetLength(0);
                lineStart = position + i + 1;
                offset = i + 1;
                pending = false;
                if (lineStart >= end) break;
            }
            if (lineStart >= end) break;
            buffer.Write(bytes, offset, read - offset);
            pending = buffer.Length > 0;
            position += read;
        }

        if (buffer.Length > 0 && lineStart < end) lines.Add(Decode(buffer));
        return lines;
    }

    private static string Decode(MemoryStream buffer)
    {
        var span = buffer.GetBuffer().AsSpan(0, (int)buffer.Length);
        if (span.Length > 0 && span[^1] == '\r') span = span[..^1];
        // A UTF-8 byte-order mark only ever starts the file.
        if (span.Length >= 3 && span[0] == 0xEF && span[1] == 0xBB && span[2] == 0xBF) span = span[3..];
        return Encoding.UTF8.GetString(span);
    }
}
