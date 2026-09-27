using System.Text.RegularExpressions;

namespace MediaPipeNet.Tasks.Text.Tokenizers;

/// <summary>
/// The regex tokenizer of MediaPipe's <c>RegexPreprocessorCalculator</c> (average word-embedding models):
/// text is split on a delimiter pattern (ASCII <c>\w</c> semantics, as in RE2, no lower-casing), each token
/// is mapped through the vocabulary (<c>&lt;UNKNOWN&gt;</c> when missing), <c>&lt;START&gt;</c> is prepended
/// and the sequence is padded with <c>&lt;PAD&gt;</c>.
/// </summary>
public sealed class RegexTokenizer
{
    private readonly Regex _delimiter;
    private readonly Dictionary<string, int> _vocab;

    /// <summary>Creates a tokenizer.</summary>
    /// <param name="delimiterPattern">Delimiter regex (MediaPipe's default for these models is <c>[^\w\']+</c>).</param>
    /// <param name="vocabulary">Lines of "token id" pairs (the model's vocab.txt).</param>
    public RegexTokenizer(string delimiterPattern, IEnumerable<string> vocabulary)
    {
        ArgumentNullException.ThrowIfNull(delimiterPattern);
        ArgumentNullException.ThrowIfNull(vocabulary);
        _delimiter = new Regex(delimiterPattern, RegexOptions.ECMAScript | RegexOptions.CultureInvariant | RegexOptions.Compiled);
        _vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var line in vocabulary)
        {
            int space = line.LastIndexOf(' ');
            if (space <= 0 || !int.TryParse(line.AsSpan(space + 1), out var id)) continue;
            _vocab.TryAdd(line[..space], id);
        }
        StartId = _vocab.TryGetValue("<START>", out var s) ? s : null;
        PadId = _vocab.GetValueOrDefault("<PAD>");
        UnknownId = _vocab.GetValueOrDefault("<UNKNOWN>");
    }

    /// <summary>Id of <c>&lt;START&gt;</c>, when the vocabulary has one.</summary>
    public int? StartId { get; }

    /// <summary>Id of <c>&lt;PAD&gt;</c>.</summary>
    public int PadId { get; }

    /// <summary>Id of <c>&lt;UNKNOWN&gt;</c>.</summary>
    public int UnknownId { get; }

    /// <summary>Splits text into tokens.</summary>
    public string[] Tokenize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return _delimiter.Split(text).Where(t => t.Length > 0).ToArray();
    }

    /// <summary>Encodes text as <paramref name="maxSequenceLength"/> token ids.</summary>
    public int[] Encode(string text, int maxSequenceLength)
    {
        var ids = new int[maxSequenceLength];
        Array.Fill(ids, PadId);
        int index = 0;
        if (StartId is { } start && maxSequenceLength > 0) ids[index++] = start;
        foreach (var token in Tokenize(text))
        {
            if (index >= maxSequenceLength) break;
            ids[index++] = _vocab.TryGetValue(token, out var id) ? id : UnknownId;
        }
        return ids;
    }
}
