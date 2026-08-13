using System.Text;

namespace Gravicode.Science.GraviText.Linguistics;

/// <summary>
/// Stop word lists for English and Bahasa Indonesia.
/// </summary>
/// <remarks>
/// Both languages are shipped because the ecosystem is documented and used bilingually, and a
/// stop word list is language-specific by definition - filtering Indonesian text with an English
/// list removes nothing useful and leaves every high-frequency Indonesian function word behind.
/// </remarks>
public static class StopWords
{
    /// <summary>Common English function words.</summary>
    public static IReadOnlySet<string> English { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
        "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
        "can", "cannot", "could", "did", "do", "does", "doing", "down", "during",
        "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
        "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
        "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
        "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
        "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
        "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
        "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
        "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves",
    };

    /// <summary>Common Indonesian function words.</summary>
    public static IReadOnlySet<string> Indonesian { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        "ada", "adalah", "adanya", "agar", "akan", "aku", "akhir", "antara", "apa", "apabila", "atau", "atas",
        "bagi", "bagaimana", "bahkan", "bahwa", "banyak", "barangkali", "belum", "berada", "berbagai", "beberapa",
        "begitu", "belakang", "berikut", "bersama", "bila", "bisa", "boleh", "buat", "bukan",
        "dan", "dapat", "dari", "dalam", "dengan", "demikian", "di", "dia", "dulu", "dua",
        "hal", "hanya", "harus", "hingga", "ia", "ialah", "ini", "itu",
        "jadi", "jangan", "jika", "juga", "kalau", "kami", "kamu", "kan", "karena", "ke", "kemudian", "kepada",
        "ketika", "kita", "lagi", "lain", "lalu", "lebih", "maka", "masih", "mau", "melakukan", "melalui",
        "memang", "mempunyai", "mendapat", "mengenai", "mereka", "merupakan", "meski", "mungkin",
        "namun", "nanti", "nya", "oleh", "pada", "paling", "para", "pula", "pun",
        "saat", "saja", "sama", "sampai", "sangat", "saya", "sebagai", "sebuah", "sedang", "sehingga",
        "sejak", "sekitar", "selain", "selama", "semua", "seperti", "serta", "sesuatu", "setelah", "siapa",
        "sudah", "supaya", "tak", "tanpa", "tapi", "telah", "tentang", "terhadap", "termasuk", "tersebut",
        "tetapi", "tidak", "untuk", "walaupun", "yaitu", "yang",
    };

    /// <summary>Both lists combined, for mixed-language corpora.</summary>
    public static IReadOnlySet<string> Bilingual { get; } =
        new HashSet<string>(English.Concat(Indonesian), StringComparer.OrdinalIgnoreCase);

    /// <summary>Drops stop words from a token sequence.</summary>
    public static IReadOnlyList<string> Remove(IReadOnlyList<string> tokens, IReadOnlySet<string>? list = null)
    {
        var stop = list ?? English;
        return tokens.Where(t => !stop.Contains(t)).ToList();
    }
}

/// <summary>
/// The Porter stemming algorithm for English.
/// </summary>
/// <remarks>
/// Stemming trades linguistic correctness for recall: "connection", "connections" and "connected"
/// all collapse to "connect", so a search or a bag-of-words model treats them as one feature. The
/// output is not always a real word, which is the accepted cost - a lemmatizer is the option when
/// real words matter.
/// </remarks>
public static class PorterStemmer
{
    /// <summary>Reduces an English word to its stem.</summary>
    public static string Stem(string word)
    {
        if (word.Length <= 2) return word;
        var b = word.ToLowerInvariant().ToCharArray();
        var k = b.Length - 1;

        k = Step1Ab(b, k);
        k = Step1C(b, k);
        k = Step2(b, k);
        k = Step3(b, k);
        k = Step4(b, k);
        k = Step5(b, k);

        return new string(b, 0, k + 1);
    }

    private static bool IsConsonant(char[] b, int i)
    {
        return b[i] switch
        {
            'a' or 'e' or 'i' or 'o' or 'u' => false,
            'y' => i == 0 || !IsConsonant(b, i - 1),
            _ => true,
        };
    }

    /// <summary>Counts vowel-consonant sequences, Porter's measure <c>m</c>.</summary>
    private static int Measure(char[] b, int k)
    {
        var n = 0;
        var i = 0;
        while (true)
        {
            if (i > k) return n;
            if (!IsConsonant(b, i)) break;
            i++;
        }
        i++;
        while (true)
        {
            while (true)
            {
                if (i > k) return n;
                if (IsConsonant(b, i)) break;
                i++;
            }
            i++;
            n++;
            while (true)
            {
                if (i > k) return n;
                if (!IsConsonant(b, i)) break;
                i++;
            }
            i++;
        }
    }

    private static bool ContainsVowel(char[] b, int k)
    {
        for (var i = 0; i <= k; i++) if (!IsConsonant(b, i)) return true;
        return false;
    }

    private static bool DoubleConsonant(char[] b, int j)
        => j >= 1 && b[j] == b[j - 1] && IsConsonant(b, j);

    private static bool CvcEndsWithShort(char[] b, int i)
    {
        if (i < 2 || !IsConsonant(b, i) || IsConsonant(b, i - 1) || !IsConsonant(b, i - 2)) return false;
        return b[i] is not ('w' or 'x' or 'y');
    }

    private static bool EndsWith(char[] b, int k, string suffix, out int stemEnd)
    {
        stemEnd = k;
        if (suffix.Length > k + 1) return false;
        for (var i = 0; i < suffix.Length; i++)
            if (b[k - suffix.Length + 1 + i] != suffix[i]) return false;
        stemEnd = k - suffix.Length;
        return true;
    }

    private static int SetTo(char[] b, int stemEnd, string suffix)
    {
        for (var i = 0; i < suffix.Length; i++) b[stemEnd + 1 + i] = suffix[i];
        return stemEnd + suffix.Length;
    }

    private static int ReplaceIfMeasureGreaterThanZero(char[] b, int k, int stemEnd, string suffix)
        => Measure(b, stemEnd) > 0 ? SetTo(b, stemEnd, suffix) : k;

    private static int Step1Ab(char[] b, int k)
    {
        if (b[k] == 's')
        {
            if (EndsWith(b, k, "sses", out var e1)) k = e1 + 2;
            else if (EndsWith(b, k, "ies", out var e2)) k = e2 + 1;
            else if (b[k - 1] != 's') k--;
        }

        if (EndsWith(b, k, "eed", out var eed)) { if (Measure(b, eed) > 0) k--; }
        else if ((EndsWith(b, k, "ed", out var ed) && ContainsVowel(b, ed))
              || (EndsWith(b, k, "ing", out var ing) && ContainsVowel(b, ing)))
        {
            k = EndsWith(b, k, "ed", out var e) ? e : EndsWith(b, k, "ing", out var g) ? g : k;

            if (EndsWith(b, k, "at", out var at)) k = SetTo(b, at, "ate");
            else if (EndsWith(b, k, "bl", out var bl)) k = SetTo(b, bl, "ble");
            else if (EndsWith(b, k, "iz", out var iz)) k = SetTo(b, iz, "ize");
            else if (DoubleConsonant(b, k) && b[k] is not ('l' or 's' or 'z')) k--;
            else if (Measure(b, k) == 1 && CvcEndsWithShort(b, k)) k = SetTo(b, k, "e");
        }
        return k;
    }

    private static int Step1C(char[] b, int k)
    {
        if (EndsWith(b, k, "y", out var y) && ContainsVowel(b, y)) b[k] = 'i';
        return k;
    }

    private static readonly (string From, string To)[] Step2Pairs =
    [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"),
        ("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
        ("ization", "ize"), ("ation", "ate"), ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
        ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
    ];

    private static int Step2(char[] b, int k)
    {
        foreach (var (from, to) in Step2Pairs)
            if (EndsWith(b, k, from, out var stem)) return ReplaceIfMeasureGreaterThanZero(b, k, stem, to);
        return k;
    }

    private static readonly (string From, string To)[] Step3Pairs =
    [
        ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
        ("ical", "ic"), ("ful", ""), ("ness", ""),
    ];

    private static int Step3(char[] b, int k)
    {
        foreach (var (from, to) in Step3Pairs)
            if (EndsWith(b, k, from, out var stem)) return ReplaceIfMeasureGreaterThanZero(b, k, stem, to);
        return k;
    }

    private static readonly string[] Step4Suffixes =
    [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent",
        "ou", "ism", "ate", "iti", "ous", "ive", "ize",
    ];

    private static int Step4(char[] b, int k)
    {
        if (EndsWith(b, k, "ion", out var ion))
        {
            if (ion >= 0 && (b[ion] == 's' || b[ion] == 't') && Measure(b, ion) > 1) return ion;
            return k;
        }

        foreach (var suffix in Step4Suffixes)
            if (EndsWith(b, k, suffix, out var stem) && Measure(b, stem) > 1) return stem;
        return k;
    }

    private static int Step5(char[] b, int k)
    {
        if (b[k] == 'e')
        {
            var m = Measure(b, k - 1);
            if (m > 1 || (m == 1 && !CvcEndsWithShort(b, k - 1))) k--;
        }
        if (b[k] == 'l' && DoubleConsonant(b, k) && Measure(b, k - 1) > 1) k--;
        return k;
    }

    /// <summary>Stems every token in a sequence.</summary>
    public static IReadOnlyList<string> StemAll(IReadOnlyList<string> tokens)
        => tokens.Select(Stem).ToList();
}

/// <summary>
/// A rule-based stemmer for Bahasa Indonesia, stripping the standard affixes.
/// </summary>
/// <remarks>
/// Indonesian is agglutinative and its morphology is regular enough that affix stripping works
/// well without a dictionary: prefixes (<c>me-</c>, <c>di-</c>, <c>ber-</c>, <c>ter-</c>,
/// <c>pe-</c>, <c>ke-</c>) and suffixes (<c>-kan</c>, <c>-an</c>, <c>-i</c>, plus the possessive
/// and particle clitics) are peeled off in the order the language builds them, which is
/// suffixes first.
/// </remarks>
public static class IndonesianStemmer
{
    private static readonly string[] Particles = ["kah", "lah", "pun"];
    private static readonly string[] Possessives = ["ku", "mu", "nya"];
    private static readonly string[] Suffixes = ["kan", "an", "i"];

    /// <summary>
    /// Prefix and suffix pairs Indonesian morphology does not produce. Nazief-Adriani uses this
    /// table to stop over-stemming: "berlari" looks like <c>ber- + lar + -i</c>, but <c>be-</c>
    /// never combines with <c>-i</c>, so the trailing i belongs to the root and must be kept.
    /// </summary>
    private static readonly Dictionary<string, string[]> ForbiddenCombinations = new(StringComparer.Ordinal)
    {
        ["be"] = ["i"],
        ["di"] = ["an"],
        ["ke"] = ["i", "kan"],
        ["me"] = ["an"],
        ["se"] = ["i", "kan"],
        ["te"] = ["an"],
    };

    /// <summary>Reduces an Indonesian word to its root form.</summary>
    public static string Stem(string word)
    {
        var value = word.ToLowerInvariant();
        if (value.Length <= 3) return value;

        // Inflectional clitics come off first: they attach outside the derivational affixes.
        value = StripFrom(value, Particles);
        value = StripFrom(value, Possessives);

        // The suffix can only be stripped if it is compatible with whatever prefix is present.
        var prefixFamily = DetectPrefixFamily(value);
        var allowedSuffixes = prefixFamily is not null && ForbiddenCombinations.TryGetValue(prefixFamily, out var forbidden)
            ? Suffixes.Where(s => !forbidden.Contains(s)).ToArray()
            : Suffixes;

        value = StripFrom(value, allowedSuffixes);
        value = StripPrefix(value);
        return value;
    }

    /// <summary>The two-letter prefix family of a word, or <c>null</c> when it has no prefix.</summary>
    private static string? DetectPrefixFamily(string word)
    {
        foreach (var prefix in new[] { "ber", "be", "di", "ke", "meng", "meny", "mem", "men", "me",
                     "peng", "peny", "pem", "pen", "pe", "ter", "te", "se" })
        {
            if (!word.StartsWith(prefix, StringComparison.Ordinal)) continue;
            if (word.Length - prefix.Length < 3) continue;
            return prefix[..2];
        }
        return null;
    }

    private static string StripFrom(string word, string[] suffixes)
    {
        foreach (var suffix in suffixes)
            if (word.Length - suffix.Length >= 3 && word.EndsWith(suffix, StringComparison.Ordinal))
                return word[..^suffix.Length];
        return word;
    }

    private static string StripPrefix(string word)
    {
        if (word.Length <= 3) return word;

        // Simple prefixes strip directly.
        foreach (var prefix in new[] { "di", "ke", "se" })
            if (word.Length - prefix.Length >= 3 && word.StartsWith(prefix, StringComparison.Ordinal))
                return word[prefix.Length..];

        foreach (var prefix in new[] { "ber", "ter", "per" })
            if (word.Length - prefix.Length >= 3 && word.StartsWith(prefix, StringComparison.Ordinal))
                return word[prefix.Length..];

        // me- and pe- assimilate to the following consonant, which has to be undone.
        foreach (var prefix in new[] { "meng", "meny", "mem", "men", "me", "peng", "peny", "pem", "pen", "pe" })
        {
            if (word.Length - prefix.Length < 3 || !word.StartsWith(prefix, StringComparison.Ordinal)) continue;
            var rest = word[prefix.Length..];

            return prefix switch
            {
                // meny + <vowel> came from a root starting with s; likewise mem+p, men+t, meng+k.
                "meny" or "peny" => "s" + rest,
                "mem" or "pem" when rest.Length > 0 && "aiueo".Contains(rest[0]) => "p" + rest,
                "men" or "pen" when rest.Length > 0 && "aiueo".Contains(rest[0]) => "t" + rest,
                "meng" or "peng" when rest.Length > 0 && "aiueo".Contains(rest[0]) => "k" + rest,
                _ => rest,
            };
        }
        return word;
    }

    /// <summary>Stems every token in a sequence.</summary>
    public static IReadOnlyList<string> StemAll(IReadOnlyList<string> tokens)
        => tokens.Select(Stem).ToList();
}

/// <summary>
/// A small dictionary-backed lemmatizer for English irregular forms, with a rule fallback.
/// </summary>
public static class Lemmatizer
{
    private static readonly Dictionary<string, string> Irregular = new(StringComparer.OrdinalIgnoreCase)
    {
        ["am"] = "be", ["is"] = "be", ["are"] = "be", ["was"] = "be", ["were"] = "be", ["been"] = "be",
        ["has"] = "have", ["had"] = "have", ["having"] = "have",
        ["does"] = "do", ["did"] = "do", ["done"] = "do", ["doing"] = "do",
        ["went"] = "go", ["gone"] = "go", ["going"] = "go",
        ["better"] = "good", ["best"] = "good", ["worse"] = "bad", ["worst"] = "bad",
        ["children"] = "child", ["men"] = "man", ["women"] = "woman", ["people"] = "person",
        ["mice"] = "mouse", ["feet"] = "foot", ["teeth"] = "tooth", ["geese"] = "goose",
        ["said"] = "say", ["made"] = "make", ["took"] = "take", ["saw"] = "see", ["came"] = "come",
    };

    /// <summary>Maps a word to its dictionary form.</summary>
    public static string Lemmatize(string word)
    {
        if (Irregular.TryGetValue(word, out var lemma)) return lemma;

        var lower = word.ToLowerInvariant();
        if (lower.EndsWith("ies", StringComparison.Ordinal) && lower.Length > 4) return lower[..^3] + "y";
        if (lower.EndsWith("ses", StringComparison.Ordinal) && lower.Length > 4) return lower[..^2];
        if (lower.EndsWith("s", StringComparison.Ordinal) && !lower.EndsWith("ss", StringComparison.Ordinal)
            && lower.Length > 3) return lower[..^1];
        return lower;
    }

    /// <summary>Lemmatizes every token in a sequence.</summary>
    public static IReadOnlyList<string> LemmatizeAll(IReadOnlyList<string> tokens)
        => tokens.Select(Lemmatize).ToList();
}

/// <summary>Contiguous token sequences, for models that need a little word order.</summary>
public static class NGrams
{
    /// <summary>All n-grams of size <paramref name="n"/>, joined with spaces.</summary>
    public static IReadOnlyList<string> Extract(IReadOnlyList<string> tokens, int n)
    {
        if (n < 1) throw new ArgumentOutOfRangeException(nameof(n));
        var result = new List<string>();
        var sb = new StringBuilder();
        for (var i = 0; i + n <= tokens.Count; i++)
        {
            sb.Clear();
            for (var k = 0; k < n; k++)
            {
                if (k > 0) sb.Append(' ');
                sb.Append(tokens[i + k]);
            }
            result.Add(sb.ToString());
        }
        return result;
    }

    /// <summary>All n-grams from <paramref name="min"/> to <paramref name="max"/> inclusive.</summary>
    public static IReadOnlyList<string> Range(IReadOnlyList<string> tokens, int min, int max)
    {
        var result = new List<string>();
        for (var n = min; n <= max; n++) result.AddRange(Extract(tokens, n));
        return result;
    }
}
