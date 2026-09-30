using System.Text;
using Gravicode.HFNet.GraviTokenizers.Components;

namespace Gravicode.HFNet.GraviTokenizers.Unicode;

/// <summary>The four Unicode normalization forms.</summary>
public enum NormalizationForm
{
    /// <summary>Canonical decomposition, then canonical composition.</summary>
    C,

    /// <summary>Canonical decomposition.</summary>
    D,

    /// <summary>Compatibility decomposition, then canonical composition.</summary>
    KC,

    /// <summary>Compatibility decomposition.</summary>
    KD,
}

/// <summary>
/// Unicode normalization from HF.Net's own tables, independent of the platform.
/// </summary>
/// <remarks>
/// <para>
/// .NET normalizes through the operating system or ICU, and under <c>InvariantGlobalization</c> -
/// which these libraries run with - <c>String.Normalize</c> returns its input unchanged. A tokenizer
/// whose <c>tokenizer.json</c> says NFC (Qwen2, Pythia, CLIP) then saw "e" plus a combining acute as
/// two characters where the reference sees "é", and produced different ids.
/// </para>
/// <para>
/// The tables are generated from Python's <c>unicodedata</c> by <c>make_tables.py</c>; the algorithm
/// is the standard one - full decomposition, canonical ordering by combining class, then canonical
/// composition for the C forms - with Hangul syllables computed rather than tabled.
/// </para>
/// </remarks>
public static class UnicodeNormalization
{
    private const int SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7;
    private const int LCount = 19, VCount = 21, TCount = 28, NCount = VCount * TCount, SCount = LCount * NCount;

    private static readonly Lazy<Tables> Data = new(Load);

    /// <summary>The Unicode version the tables were generated from.</summary>
    public static string UnicodeVersion => UnicodeTables.Version;

    /// <summary>Normalizes <paramref name="text"/> to <paramref name="form"/>.</summary>
    public static string Normalize(string text, NormalizationForm form)
    {
        ArgumentNullException.ThrowIfNull(text);

        // Pure ASCII is already in every form - the common case, and it costs one scan.
        var ascii = true;
        foreach (var c in text)
        {
            if (c >= 0x80)
            {
                ascii = false;
                break;
            }
        }

        if (ascii) return text;

        var tables = Data.Value;
        var compatibility = form is NormalizationForm.KC or NormalizationForm.KD;

        var decomposed = new List<int>(text.Length);
        foreach (var rune in text.EnumerateRunes()) Decompose(rune.Value, compatibility, tables, decomposed);

        Reorder(decomposed, tables);

        if (form is NormalizationForm.C or NormalizationForm.KC) Compose(decomposed, tables);

        var builder = new StringBuilder(decomposed.Count);
        foreach (var code in decomposed) builder.Append(char.ConvertFromUtf32(code));
        return builder.ToString();
    }

    private static void Decompose(int code, bool compatibility, Tables tables, List<int> output)
    {
        if (code >= SBase && code < SBase + SCount)
        {
            var index = code - SBase;
            output.Add(LBase + index / NCount);
            output.Add(VBase + index % NCount / TCount);
            if (index % TCount != 0) output.Add(TBase + index % TCount);
            return;
        }

        if (tables.Decompositions.TryGetValue(code, out var entry) && (compatibility || !entry.Compatibility))
        {
            foreach (var part in entry.Mapping) Decompose(part, compatibility, tables, output);
            return;
        }

        output.Add(code);
    }

    /// <summary>Canonical ordering: a stable sort by combining class within each run of non-starters.</summary>
    private static void Reorder(List<int> codes, Tables tables)
    {
        for (var i = 1; i < codes.Count; i++)
        {
            var current = Class(codes[i], tables);
            if (current == 0) continue;

            var j = i;
            while (j > 0)
            {
                var previous = Class(codes[j - 1], tables);
                if (previous <= current) break;
                (codes[j - 1], codes[j]) = (codes[j], codes[j - 1]);
                j--;
            }
        }
    }

    /// <summary>
    /// Canonical composition, UAX #15's algorithm: each character joins the last starter when a
    /// composite exists and nothing between them blocks it - no character of equal or higher class.
    /// </summary>
    private static void Compose(List<int> codes, Tables tables)
    {
        if (codes.Count == 0) return;

        var starterIndex = 0;
        var starter = codes[0];
        var lastClass = Class(starter, tables);
        if (lastClass != 0) lastClass = 256;   // a leading non-starter composes with nothing

        var write = 1;
        for (var read = 1; read < codes.Count; read++)
        {
            var code = codes[read];
            var cls = Class(code, tables);

            if ((lastClass < cls || lastClass == 0) && TryCompose(starter, code, tables, out var composite))
            {
                codes[starterIndex] = composite;
                starter = composite;
                continue;
            }

            if (cls == 0)
            {
                starterIndex = write;
                starter = code;
            }

            lastClass = cls;
            codes[write++] = code;
        }

        codes.RemoveRange(write, codes.Count - write);
    }

    private static bool TryCompose(int first, int second, Tables tables, out int composite)
    {
        // Hangul L + V, and LV + T.
        if (first >= LBase && first < LBase + LCount && second >= VBase && second < VBase + VCount)
        {
            composite = SBase + ((first - LBase) * VCount + (second - VBase)) * TCount;
            return true;
        }

        if (first >= SBase && first < SBase + SCount && (first - SBase) % TCount == 0
            && second > TBase && second < TBase + TCount)
        {
            composite = first + (second - TBase);
            return true;
        }

        return tables.Compositions.TryGetValue(((long)first << 21) | (uint)second, out composite);
    }

    private static int Class(int code, Tables tables) => tables.Combining.TryGetValue(code, out var cls) ? cls : 0;

    private sealed record Tables(
        Dictionary<int, (bool Compatibility, int[] Mapping)> Decompositions,
        Dictionary<int, int> Combining,
        Dictionary<long, int> Compositions);

    private static Tables Load()
    {
        var decompositions = new Dictionary<int, (bool, int[])>();
        foreach (var entry in UnicodeTables.Decompositions.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            var separator = entry.IndexOfAny([':', '!']);
            var code = Convert.ToInt32(entry[..separator], 16);
            var mapping = entry[(separator + 1)..].Split(' ').Select(h => Convert.ToInt32(h, 16)).ToArray();
            decompositions[code] = (entry[separator] == '!', mapping);
        }

        var combining = new Dictionary<int, int>();
        foreach (var entry in UnicodeTables.CombiningClasses.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            var parts = entry.Split(':');
            combining[Convert.ToInt32(parts[0], 16)] = int.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);
        }

        var compositions = new Dictionary<long, int>();
        foreach (var entry in UnicodeTables.Compositions.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            var parts = entry.Split(':');
            var pair = parts[0].Split(' ');
            var key = ((long)Convert.ToInt32(pair[0], 16) << 21) | (uint)Convert.ToInt32(pair[1], 16);
            compositions[key] = Convert.ToInt32(parts[1], 16);
        }

        return new Tables(decompositions, combining, compositions);
    }
}

/// <summary>A <c>tokenizer.json</c> NFC, NFD, NFKC or NFKD normalizer.</summary>
/// <param name="Form">The form.</param>
public sealed record UnicodeNormalizer(NormalizationForm Form) : INormalizer
{
    /// <inheritdoc />
    public string Normalize(string text) => UnicodeNormalization.Normalize(text, Form);
}
