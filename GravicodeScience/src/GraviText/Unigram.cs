using System.Text;

namespace Gravicode.Science.GraviText.Tokenization;

/// <summary>
/// The unigram language-model tokenizer, as used by SentencePiece.
/// </summary>
/// <remarks>
/// <para>
/// A different idea from <see cref="BpeTokenizer"/>, not a variant of it. BPE builds a vocabulary
/// upwards by merging and then segments greedily by replaying those merges; unigram starts from a
/// large candidate vocabulary, assigns every piece a probability, and <em>prunes downwards</em> —
/// repeatedly dropping the pieces whose removal costs the corpus likelihood least. Segmentation is
/// then a shortest-path problem: the split whose pieces have the highest total log probability, found
/// exactly by Viterbi rather than approximated greedily.
/// </para>
/// <para>
/// Two consequences follow, and both are why the method is worth having alongside BPE. The
/// segmentation is globally optimal under the model rather than a greedy artefact, so it does not
/// depend on the order rules happened to be learned. And because every piece carries a probability,
/// the tokenizer can sample alternative segmentations of the same text — the basis of subword
/// regularisation, which trains a model on several segmentations of each sentence and makes it
/// markedly more robust to the tokenizer's arbitrary choices.
/// </para>
/// <para>
/// <b>Whitespace is part of the input, not a delimiter.</b> Following SentencePiece, spaces are
/// replaced by a visible marker and the text is treated as one raw stream. That is what makes the
/// scheme fully reversible — decoding is concatenation with the marker mapped back to a space — and
/// what lets it handle languages that do not put spaces between words at all, where a
/// whitespace pre-tokenizer has nothing to work with.
/// </para>
/// </remarks>
public sealed class UnigramTokenizer : ITokenizer
{
    /// <summary>Stands in for a space, so segmentation is fully reversible.</summary>
    /// <remarks>U+2581 LOWER ONE EIGHTH BLOCK, which is what SentencePiece uses.</remarks>
    public const char SpaceMarker = '▁';

    private readonly Dictionary<string, double> _logProbabilities;
    private readonly int _longestPiece;

    /// <summary>Creates a tokenizer from pieces and their log probabilities.</summary>
    public UnigramTokenizer(IReadOnlyDictionary<string, double> logProbabilities)
    {
        ArgumentNullException.ThrowIfNull(logProbabilities);
        if (logProbabilities.Count == 0) throw new ArgumentException("The vocabulary is empty.");

        _logProbabilities = new Dictionary<string, double>(logProbabilities, StringComparer.Ordinal);
        _longestPiece = _logProbabilities.Keys.Max(p => p.Length);

        Vocabulary = new Vocabulary();
        foreach (var piece in _logProbabilities.Keys.OrderByDescending(p => _logProbabilities[p]))
            Vocabulary.Add(piece);
    }

    /// <summary>The piece vocabulary, most probable first.</summary>
    public Vocabulary Vocabulary { get; }

    /// <summary>Number of pieces.</summary>
    public int PieceCount => _logProbabilities.Count;

    /// <summary>The log probability of a piece, or negative infinity when it is not in the vocabulary.</summary>
    public double LogProbability(string piece)
        => _logProbabilities.TryGetValue(piece, out var value) ? value : double.NegativeInfinity;

    /// <summary>
    /// Learns a vocabulary by expectation-maximisation with iterative pruning.
    /// </summary>
    /// <param name="corpus">Documents to learn from.</param>
    /// <param name="vocabularySize">Target number of pieces.</param>
    /// <param name="seedSize">How many candidate substrings to start from before pruning.</param>
    /// <param name="shrinkFactor">Fraction of the vocabulary kept at each pruning round.</param>
    /// <param name="emIterations">EM steps between prunes.</param>
    /// <remarks>
    /// <para>
    /// The loop is: estimate each piece's probability by EM given the current vocabulary, score
    /// every piece by how much the corpus likelihood would fall without it, drop the worst, repeat.
    /// Single characters are never dropped, because a vocabulary that cannot spell a character
    /// cannot segment text containing it at all.
    /// </para>
    /// <para>
    /// The E step is where the method differs from counting: the expected count of a piece is
    /// summed over <em>all</em> segmentations weighted by their probability, not just the best one.
    /// Doing it with the Viterbi path alone is a recognised approximation, and it makes rare pieces
    /// look worse than they are because they never win a single best path.
    /// </para>
    /// </remarks>
    public static UnigramTokenizer Train(IEnumerable<string> corpus, int vocabularySize = 1000,
        int seedSize = 10000, double shrinkFactor = 0.75, int emIterations = 2)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        if (vocabularySize < 2) throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        if (shrinkFactor is <= 0 or >= 1) throw new ArgumentOutOfRangeException(nameof(shrinkFactor));

        var sentences = corpus.Select(Normalise).Where(s => s.Length > 0).ToArray();
        if (sentences.Length == 0) throw new ArgumentException("The corpus is empty.", nameof(corpus));

        var (candidates, required) = SeedVocabulary(sentences, seedSize);

        while (true)
        {
            var probabilities = RunEm(sentences, candidates, emIterations);

            if (probabilities.Count <= vocabularySize)
                return new UnigramTokenizer(probabilities);

            // Drop the least useful pieces, but never one that is the only way to spell a character.
            var target = Math.Max(vocabularySize, (int)(probabilities.Count * shrinkFactor));

            var ranked = probabilities
                .Where(kv => !required.Contains(kv.Key))
                .OrderByDescending(kv => kv.Value)
                .ThenBy(kv => kv.Key, StringComparer.Ordinal)
                .ToList();

            var keep = new HashSet<string>(required, StringComparer.Ordinal);
            foreach (var (piece, _) in ranked.Take(Math.Max(0, target - required.Count))) keep.Add(piece);

            if (keep.Count >= candidates.Count) return new UnigramTokenizer(probabilities);
            candidates = keep;
        }
    }

    /// <inheritdoc />
    public IReadOnlyList<string> Tokenize(string text) => Encode(text);

    /// <summary>
    /// The most probable segmentation, found by Viterbi over the piece lattice.
    /// </summary>
    /// <remarks>
    /// Exact, not greedy: <c>best[i]</c> is the score of the best segmentation of the first
    /// <c>i</c> characters, so every prefix is solved once and the whole optimum falls out. A greedy
    /// longest-match scan is what BPE-style tokenizers do and it can be arbitrarily worse — taking a
    /// long, rare piece early can force the remainder into several improbable ones.
    /// </remarks>
    public IReadOnlyList<string> Encode(string text)
    {
        var normalised = Normalise(text);
        if (normalised.Length == 0) return [];

        var n = normalised.Length;
        var best = new double[n + 1];
        var from = new int[n + 1];
        var piece = new string[n + 1];

        Array.Fill(best, double.NegativeInfinity);
        best[0] = 0;

        for (var end = 1; end <= n; end++)
        {
            var earliest = Math.Max(0, end - _longestPiece);

            for (var start = earliest; start < end; start++)
            {
                if (double.IsNegativeInfinity(best[start])) continue;

                var candidate = normalised[start..end];
                if (!_logProbabilities.TryGetValue(candidate, out var logProbability)) continue;

                var score = best[start] + logProbability;
                if (score <= best[end]) continue;

                best[end] = score;
                from[end] = start;
                piece[end] = candidate;
            }

            // An unrepresentable character would leave the lattice disconnected and every later
            // position unreachable. Emitting it as its own piece keeps segmentation total.
            if (!double.IsNegativeInfinity(best[end])) continue;

            best[end] = best[end - 1] - 1e6;
            from[end] = end - 1;
            piece[end] = normalised[(end - 1)..end];
        }

        var result = new List<string>();
        for (var position = n; position > 0; position = from[position]) result.Add(piece[position]);
        result.Reverse();
        return result;
    }

    /// <summary>Encodes text as vocabulary ids.</summary>
    public int[] EncodeIds(string text) => Vocabulary.Encode(Encode(text));

    /// <summary>
    /// Samples a segmentation rather than taking the best one.
    /// </summary>
    /// <param name="text">The text to segment.</param>
    /// <param name="rng">Source of randomness.</param>
    /// <param name="alpha">
    /// Smoothing on the sampling distribution. Small values approach uniform over segmentations;
    /// large values approach the Viterbi path.
    /// </param>
    /// <remarks>
    /// The basis of subword regularisation. Training a model on several segmentations of the same
    /// sentence stops it depending on the tokenizer's arbitrary choices, and measurably improves
    /// robustness — particularly on text unlike the tokenizer's training corpus. This is the one
    /// thing a BPE tokenizer cannot do without extra machinery, because it has no probabilities to
    /// sample from.
    /// </remarks>
    public IReadOnlyList<string> SampleEncoding(string text, GraviNum.GraviRandom rng, double alpha = 0.2)
    {
        ArgumentNullException.ThrowIfNull(rng);

        var normalised = Normalise(text);
        if (normalised.Length == 0) return [];

        // Forward pass in log space: the total probability of reaching each position, so a choice
        // can be sampled in proportion to how much probability mass flows through it.
        var n = normalised.Length;
        var forward = new double[n + 1];
        Array.Fill(forward, double.NegativeInfinity);
        forward[0] = 0;

        for (var end = 1; end <= n; end++)
        {
            var earliest = Math.Max(0, end - _longestPiece);
            for (var start = earliest; start < end; start++)
            {
                if (double.IsNegativeInfinity(forward[start])) continue;
                if (!_logProbabilities.TryGetValue(normalised[start..end], out var logProbability)) continue;

                forward[end] = LogAdd(forward[end], forward[start] + alpha * logProbability);
            }

            if (double.IsNegativeInfinity(forward[end])) forward[end] = forward[end - 1] - 1e6;
        }

        // Backward sampling: walk from the end, choosing each predecessor in proportion to its share.
        var result = new List<string>();
        var position = n;

        while (position > 0)
        {
            var earliest = Math.Max(0, position - _longestPiece);
            var options = new List<(int Start, string Piece, double Score)>();

            for (var start = earliest; start < position; start++)
            {
                if (double.IsNegativeInfinity(forward[start])) continue;

                var candidate = normalised[start..position];
                if (!_logProbabilities.TryGetValue(candidate, out var logProbability)) continue;

                options.Add((start, candidate, forward[start] + alpha * logProbability));
            }

            if (options.Count == 0)
            {
                result.Add(normalised[(position - 1)..position]);
                position--;
                continue;
            }

            var max = options.Max(o => o.Score);
            var total = options.Sum(o => Math.Exp(o.Score - max));
            var draw = rng.NextDouble() * total;

            var chosen = options[^1];
            var running = 0.0;

            foreach (var option in options)
            {
                running += Math.Exp(option.Score - max);
                if (running < draw) continue;
                chosen = option;
                break;
            }

            result.Add(chosen.Piece);
            position = chosen.Start;
        }

        result.Reverse();
        return result;
    }

    /// <summary>Reassembles the original text from its pieces.</summary>
    /// <remarks>
    /// Plain concatenation with the marker mapped back to a space. Because whitespace was encoded
    /// rather than discarded, this is exactly reversible — unlike a whitespace-delimited scheme,
    /// where the original spacing has to be guessed.
    /// </remarks>
    public string Decode(IReadOnlyList<string> pieces)
        => string.Concat(pieces).Replace(SpaceMarker, ' ').TrimStart();

    /// <summary>Reassembles text from vocabulary ids.</summary>
    public string DecodeIds(IReadOnlyList<int> ids)
        => Decode([.. ids.Select(id => Vocabulary[id])]);

    /// <summary>Writes each piece and its log probability, one per line.</summary>
    public void Save(string path)
        => File.WriteAllLines(path, _logProbabilities
            .OrderByDescending(kv => kv.Value)
            .Select(kv => $"{kv.Key}\t{kv.Value.ToString("R", System.Globalization.CultureInfo.InvariantCulture)}"));

    /// <summary>Reads a model written by <see cref="Save"/>.</summary>
    public static UnigramTokenizer Load(string path)
    {
        var pieces = new Dictionary<string, double>(StringComparer.Ordinal);

        foreach (var line in File.ReadLines(path))
        {
            var tab = line.LastIndexOf('\t');
            if (tab <= 0) continue;

            pieces[line[..tab]] = double.Parse(
                line[(tab + 1)..], System.Globalization.CultureInfo.InvariantCulture);
        }

        return new UnigramTokenizer(pieces);
    }

    // ------------------------------------------------------------------ training

    /// <summary>Replaces spaces with the marker and prefixes one, so word starts are visible.</summary>
    private static string Normalise(string text)
    {
        var trimmed = text.Trim();
        if (trimmed.Length == 0) return "";
        return (SpaceMarker + trimmed).Replace(' ', SpaceMarker);
    }

    /// <summary>
    /// The candidate pieces to start pruning from, and the ones that may never be pruned.
    /// </summary>
    /// <remarks>
    /// Every substring up to a bounded length, kept by frequency times length — the product is what
    /// SentencePiece scores by, and it favours pieces that actually save space over merely frequent
    /// short ones. Single characters are separated out as required: dropping one would make some
    /// text unsegmentable.
    /// </remarks>
    private static (HashSet<string> Candidates, HashSet<string> Required) SeedVocabulary(
        string[] sentences, int seedSize)
    {
        const int maxPieceLength = 16;

        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        var required = new HashSet<string>(StringComparer.Ordinal);

        foreach (var sentence in sentences)
        {
            for (var start = 0; start < sentence.Length; start++)
            {
                required.Add(sentence[start..(start + 1)]);

                var limit = Math.Min(maxPieceLength, sentence.Length - start);
                for (var length = 2; length <= limit; length++)
                {
                    var piece = sentence.Substring(start, length);
                    counts.TryGetValue(piece, out var existing);
                    counts[piece] = existing + 1;
                }
            }
        }

        var candidates = new HashSet<string>(required, StringComparer.Ordinal);

        foreach (var (piece, _) in counts
            .Where(kv => kv.Value > 1)
            .OrderByDescending(kv => (long)kv.Value * kv.Key.Length)
            .ThenBy(kv => kv.Key, StringComparer.Ordinal)
            .Take(Math.Max(0, seedSize - required.Count)))
            candidates.Add(piece);

        return (candidates, required);
    }

    /// <summary>
    /// Estimates piece probabilities by expectation-maximisation.
    /// </summary>
    /// <remarks>
    /// The E step accumulates each piece's expected count over all segmentations — forward and
    /// backward passes give the probability mass flowing through every lattice edge — and the M
    /// step normalises those counts into probabilities. Repeating monotonically increases the
    /// corpus likelihood.
    /// </remarks>
    private static Dictionary<string, double> RunEm(string[] sentences, HashSet<string> candidates,
        int iterations)
    {
        var longest = candidates.Max(p => p.Length);

        // Start uniform: with no information, every candidate is equally likely.
        var logProbabilities = candidates.ToDictionary(
            p => p, _ => -Math.Log(candidates.Count), StringComparer.Ordinal);

        for (var iteration = 0; iteration < iterations; iteration++)
        {
            var expected = new Dictionary<string, double>(StringComparer.Ordinal);

            foreach (var sentence in sentences)
                Accumulate(sentence, logProbabilities, longest, expected);

            var total = expected.Values.Sum();
            if (total <= 0) break;

            var updated = new Dictionary<string, double>(StringComparer.Ordinal);
            foreach (var piece in candidates)
            {
                // A floor rather than zero: a piece with no expected count this round would
                // otherwise be unrecoverable, and EM has not finished converging.
                var count = expected.GetValueOrDefault(piece, 0.0) + 1e-10;
                updated[piece] = Math.Log(count / (total + 1e-10 * candidates.Count));
            }

            logProbabilities = updated;
        }

        return logProbabilities;
    }

    /// <summary>Adds one sentence's expected piece counts, by the forward-backward algorithm.</summary>
    private static void Accumulate(string sentence, Dictionary<string, double> logProbabilities,
        int longest, Dictionary<string, double> expected)
    {
        var n = sentence.Length;

        var forward = new double[n + 1];
        var backward = new double[n + 1];
        Array.Fill(forward, double.NegativeInfinity);
        Array.Fill(backward, double.NegativeInfinity);

        forward[0] = 0;
        backward[n] = 0;

        for (var end = 1; end <= n; end++)
            for (var start = Math.Max(0, end - longest); start < end; start++)
            {
                if (double.IsNegativeInfinity(forward[start])) continue;
                if (!logProbabilities.TryGetValue(sentence[start..end], out var logProbability)) continue;

                forward[end] = LogAdd(forward[end], forward[start] + logProbability);
            }

        for (var start = n - 1; start >= 0; start--)
            for (var end = start + 1; end <= Math.Min(n, start + longest); end++)
            {
                if (double.IsNegativeInfinity(backward[end])) continue;
                if (!logProbabilities.TryGetValue(sentence[start..end], out var logProbability)) continue;

                backward[start] = LogAdd(backward[start], backward[end] + logProbability);
            }

        var evidence = forward[n];
        if (double.IsNegativeInfinity(evidence)) return;

        // Each edge's share of the total probability mass is its expected count.
        for (var start = 0; start < n; start++)
            for (var end = start + 1; end <= Math.Min(n, start + longest); end++)
            {
                if (double.IsNegativeInfinity(forward[start]) || double.IsNegativeInfinity(backward[end])) continue;

                var piece = sentence[start..end];
                if (!logProbabilities.TryGetValue(piece, out var logProbability)) continue;

                var posterior = Math.Exp(forward[start] + logProbability + backward[end] - evidence);

                expected.TryGetValue(piece, out var existing);
                expected[piece] = existing + posterior;
            }
    }

    /// <summary>log(e^a + e^b), computed without leaving log space.</summary>
    private static double LogAdd(double a, double b)
    {
        if (double.IsNegativeInfinity(a)) return b;
        if (double.IsNegativeInfinity(b)) return a;

        var max = Math.Max(a, b);
        return max + Math.Log(Math.Exp(a - max) + Math.Exp(b - max));
    }
}
