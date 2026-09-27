using System.Buffers.Binary;
using System.Text;

namespace MediaPipeNet.Tasks.Text.Tokenizers;

/// <summary>
/// Character n-gram hashing of MediaPipe's language detector (the TFLite custom op <c>NGramHash</c>):
/// the text is lower-cased, every non-letter becomes a space, the characters are wrapped in <c>^</c> …
/// <c>$</c>, and every n-gram (n = 1..4) starting at each character is hashed with 64-bit MurmurHash into
/// a vocabulary id in [1, vocabSize].
/// </summary>
public sealed class NGramHasher
{
    private const ulong Mul = 0xc6a4a7935bd1e995UL;

    /// <summary>Creates a hasher (defaults are the parameters of MediaPipe's language_detector.tflite).</summary>
    public NGramHasher(ulong seed = 10911, IReadOnlyList<int>? ngramLengths = null, IReadOnlyList<int>? vocabSizes = null, int maxSplits = 128)
    {
        Seed = seed;
        NGramLengths = ngramLengths ?? [1, 2, 3, 4];
        VocabSizes = vocabSizes ?? [5500, 5500, 13000, 13000];
        if (NGramLengths.Count != VocabSizes.Count) throw new ArgumentException("ngramLengths and vocabSizes must have the same length.");
        MaxSplits = maxSplits;
    }

    /// <summary>Hash seed.</summary>
    public ulong Seed { get; }

    /// <summary>The n-gram lengths.</summary>
    public IReadOnlyList<int> NGramLengths { get; }

    /// <summary>Vocabulary size per n-gram length.</summary>
    public IReadOnlyList<int> VocabSizes { get; }

    /// <summary>Maximum number of character tokens (including the ^ and $ markers).</summary>
    public int MaxSplits { get; }

    /// <summary>
    /// Returns the ids as a row-major [n-gram lengths × tokens] matrix, and the number of tokens.
    /// </summary>
    public (int[] Ids, int TokenCount) Hash(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var (bytes, tokens) = Tokenize(text);
        int count = tokens.Count;
        var ids = new int[NGramLengths.Count * count];
        for (int g = 0; g < NGramLengths.Count; g++)
        {
            int n = NGramLengths[g];
            ulong vocab = (ulong)VocabSizes[g];
            for (int start = 0; start < count; start++)
            {
                int length = 0;
                for (int i = start; i < count && i < start + n; i++) length += tokens[i].Length;
                ulong hash = MurmurHash64(bytes.AsSpan(tokens[start].Offset, length), Seed);
                ids[g * count + start] = (int)(hash % vocab) + 1;
            }
        }
        return (ids, count);
    }

    private (byte[] Bytes, List<(int Offset, int Length)> Tokens) Tokenize(string text)
    {
        // LowercaseUnicodeStr: letters are lower-cased; the tokenizer then reads as many bytes as the
        // original UTF-8 input had (a quirk kept for parity when lower-casing changes the length).
        var original = Encoding.UTF8.GetByteCount(text);
        var lower = new StringBuilder(text.Length);
        foreach (var rune in text.EnumerateRunes()) lower.Append((Rune.IsLetter(rune) ? Rune.ToLowerInvariant(rune) : rune).ToString());
        var input = Encoding.UTF8.GetBytes(lower.ToString());
        int len = Math.Min(original, input.Length);

        var output = new List<byte>(len + 2);
        var tokens = new List<(int, int)>(len + 2);
        output.Add((byte)'^');
        tokens.Add((0, 1));
        int i = 0;
        while (i < len && tokens.Count + 1 < MaxSplits)
        {
            var status = Rune.DecodeFromUtf8(input.AsSpan(i, len - i), out var rune, out int read);
            if (read == 0) break;
            if (status != System.Buffers.OperationStatus.Done || !Rune.IsLetter(rune))
            {
                tokens.Add((output.Count, 1));
                output.Add((byte)' ');
            }
            else
            {
                tokens.Add((output.Count, read));
                for (int k = 0; k < read; k++) output.Add(input[i + k]);
            }
            i += read;
        }
        tokens.Add((output.Count, 1));
        output.Add((byte)'$');
        return (output.ToArray(), tokens);
    }

    /// <summary>MurmurHash64 with seed, as used by MediaPipe's language detector custom ops.</summary>
    public static ulong MurmurHash64(ReadOnlySpan<byte> data, ulong seed)
    {
        int aligned = data.Length & ~7;
        ulong hash = seed ^ ((ulong)data.Length * Mul);
        for (int p = 0; p < aligned; p += 8)
        {
            ulong k = BinaryPrimitives.ReadUInt64LittleEndian(data[p..]);
            hash ^= ShiftMix(k * Mul) * Mul;
            hash *= Mul;
        }
        int rest = data.Length & 7;
        if (rest != 0)
        {
            ulong k = 0;
            for (int i = 0; i < rest; i++) k |= (ulong)data[aligned + i] << (8 * i);
            hash ^= k;
            hash *= Mul;
        }
        hash = ShiftMix(hash) * Mul;
        return ShiftMix(hash);
    }

    private static ulong ShiftMix(ulong v) => v ^ (v >> 47);
}
