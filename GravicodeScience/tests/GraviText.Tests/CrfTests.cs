using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Sequence;
using Xunit;

namespace Gravicode.Science.Tests.GraviText;

/// <summary>
/// Tests for the linear-chain CRF.
/// </summary>
/// <remarks>
/// A CRF's forward algorithm and its Viterbi decoder both look right when they are wrong, so the
/// pins here are exhaustive: on short sequences every possible labelling is enumerated inside the
/// test, and the partition function, the best path and the marginals are all checked against that
/// brute-force enumeration rather than against the implementation's own recursions.
/// </remarks>
public class CrfTests
{
    private static NdArray Emissions(params double[][] rows)
    {
        var result = NdArray.Zeros(rows.Length, rows[0].Length);
        for (var t = 0; t < rows.Length; t++)
            for (var label = 0; label < rows[0].Length; label++)
                result[t, label] = rows[t][label];
        return result;
    }

    /// <summary>Every labelling of a sequence of the given length.</summary>
    private static IEnumerable<int[]> AllLabellings(int steps, int labelCount)
    {
        var current = new int[steps];

        IEnumerable<int[]> Extend(int position)
        {
            if (position == steps) { yield return (int[])current.Clone(); yield break; }

            for (var label = 0; label < labelCount; label++)
            {
                current[position] = label;
                foreach (var result in Extend(position + 1)) yield return result;
            }
        }

        return Extend(0);
    }

    private static LinearChainCrf WithTransitions(int labels, params (int From, int To, double Score)[] scores)
    {
        var crf = new LinearChainCrf(labels);
        foreach (var (from, to, score) in scores) crf.SetTransition(from, to, score);
        return crf;
    }

    [Fact]
    public void ThePartitionFunctionMatchesBruteForceEnumeration()
    {
        // The exact check: the forward algorithm sums over k^n sequences in O(n k²), and the only
        // way to know it is right is to actually enumerate them on a small case.
        var crf = new LinearChainCrf(3);
        crf.Fit([Emissions([0.1, 0.2, 0.3], [0.4, 0.1, 0.2])], [[0, 1]], epochs: 5);

        var emissions = Emissions([0.5, -0.2, 0.1], [0.0, 0.7, -0.3], [-0.1, 0.2, 0.6]);

        var brute = double.NegativeInfinity;
        foreach (var labelling in AllLabellings(3, 3))
        {
            var score = crf.Score(emissions, labelling);
            brute = brute == double.NegativeInfinity
                ? score
                : Math.Max(brute, score) + Math.Log(1 + Math.Exp(-Math.Abs(brute - score)));
        }

        Assert.Equal(brute, crf.LogPartition(emissions), 9);
    }

    [Fact]
    public void ViterbiFindsTheHighestScoringSequence()
    {
        var crf = new LinearChainCrf(3);
        crf.Fit([Emissions([0.3, 0.1, 0.2], [0.1, 0.5, 0.1])], [[1, 0]], epochs: 10);

        var emissions = Emissions([0.5, -0.2, 0.1], [0.0, 0.7, -0.3], [-0.1, 0.2, 0.6]);

        var best = AllLabellings(3, 3).OrderByDescending(l => crf.Score(emissions, l)).First();
        Assert.Equal(best, crf.Decode(emissions));
    }

    [Fact]
    public void TheBestSequenceIsNotTheSequenceOfBestTokens()
    {
        // The reason to use a CRF at all. Transitions can make a locally worse label the right
        // choice, and independent per-token argmax cannot see that.
        var crf = new LinearChainCrf(2);

        // Staying in the same state is cheap; switching is expensive.
        crf.SetTransition(0, 0, 5.0);
        crf.SetTransition(1, 1, 5.0);
        crf.SetTransition(0, 1, -5.0);
        crf.SetTransition(1, 0, -5.0);

        // The middle token individually prefers label 1, but only just.
        var emissions = Emissions([2.0, 0.0], [0.0, 0.5], [2.0, 0.0]);

        var independent = new[] { 0, 1, 0 };
        var decoded = crf.Decode(emissions);

        Assert.Equal([0, 0, 0], decoded);
        Assert.NotEqual(independent, decoded);
        Assert.True(crf.Score(emissions, decoded) > crf.Score(emissions, independent));
    }

    [Fact]
    public void MarginalsMatchBruteForceEnumeration()
    {
        var crf = new LinearChainCrf(2);
        crf.SetTransition(0, 1, 0.8);
        crf.SetTransition(1, 0, -0.4);

        var emissions = Emissions([0.5, -0.2], [0.0, 0.7], [-0.1, 0.2]);
        var marginals = crf.Marginals(emissions);

        var partition = crf.LogPartition(emissions);

        for (var t = 0; t < 3; t++)
            for (var label = 0; label < 2; label++)
            {
                var mass = 0.0;
                foreach (var labelling in AllLabellings(3, 2))
                    if (labelling[t] == label) mass += Math.Exp(crf.Score(emissions, labelling) - partition);

                Assert.Equal(mass, marginals[t, label], 9);
            }
    }

    [Fact]
    public void MarginalsSumToOneAtEveryPosition()
    {
        var crf = new LinearChainCrf(3);
        var emissions = Emissions([0.5, -0.2, 0.1], [0.0, 0.7, -0.3], [-0.1, 0.2, 0.6], [0.4, 0.1, 0.0]);

        var marginals = crf.Marginals(emissions);

        for (var t = 0; t < 4; t++)
        {
            var total = 0.0;
            for (var label = 0; label < 3; label++) total += marginals[t, label];
            Assert.Equal(1.0, total, 9);
        }
    }

    [Fact]
    public void TheLogLikelihoodIsNeverPositive()
    {
        // A probability cannot exceed one. If the partition function is understated this fails
        // immediately, which no amount of plausible-looking decoding would reveal.
        var crf = new LinearChainCrf(3);
        var emissions = Emissions([0.5, -0.2, 0.1], [0.0, 0.7, -0.3], [-0.1, 0.2, 0.6]);

        foreach (var labelling in AllLabellings(3, 3))
            Assert.True(crf.LogLikelihood(emissions, labelling) <= 1e-12,
                $"labelling [{string.Join(", ", labelling)}] had log likelihood " +
                $"{crf.LogLikelihood(emissions, labelling)}");
    }

    [Fact]
    public void ForbiddenTransitionsNeverAppearInTheDecodedPath()
    {
        var crf = new LinearChainCrf(2);
        crf.Forbid(0, 1);

        // Emissions that would otherwise force exactly that transition.
        var emissions = Emissions([10.0, 0.0], [0.0, 10.0]);
        var decoded = crf.Decode(emissions);

        Assert.False(decoded[0] == 0 && decoded[1] == 1, "a forbidden transition was decoded");
    }

    [Fact]
    public void BioConstraintsRuleOutInvalidTagSequences()
    {
        // The practical case: an I-PER may not follow an O, and may not follow a B-LOC either —
        // checking only the prefix and not the entity type misses the second.
        string[] labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"];

        var crf = new LinearChainCrf(labels.Length);
        crf.ApplyBioConstraints(labels);

        // Emissions that strongly want O then I-PER.
        var emissions = NdArray.Full(-10.0, 2, labels.Length);
        emissions[0, 0] = 10;      // O
        emissions[1, 2] = 10;      // I-PER

        var decoded = crf.Decode(emissions);
        Assert.False(labels[decoded[0]] == "O" && labels[decoded[1]] == "I-PER",
            "I-PER was decoded directly after O");

        // And a mismatched type is refused too.
        var mismatched = NdArray.Full(-10.0, 2, labels.Length);
        mismatched[0, 3] = 10;     // B-LOC
        mismatched[1, 2] = 10;     // I-PER

        var second = crf.Decode(mismatched);
        Assert.False(labels[second[0]] == "B-LOC" && labels[second[1]] == "I-PER",
            "I-PER was decoded after B-LOC");
    }

    [Fact]
    public void ASequenceCannotOpenWithAContinuationTag()
    {
        string[] labels = ["O", "B-PER", "I-PER"];

        var crf = new LinearChainCrf(labels.Length);
        crf.ApplyBioConstraints(labels);

        var emissions = NdArray.Full(-10.0, 1, labels.Length);
        emissions[0, 2] = 10;      // I-PER, strongly

        Assert.NotEqual(2, crf.Decode(emissions)[0]);
    }

    [Fact]
    public void TrainingRaisesTheLikelihoodOfTheObservedSequences()
    {
        // The gradient is observed counts minus expected counts, so if the sign is right the
        // training data must become more likely.
        var sequences = new List<NdArray>();
        var labels = new List<int[]>();

        var rng = new GraviRandom(3);
        for (var s = 0; s < 40; s++)
        {
            // A pattern the transitions can learn: label 1 always follows label 0.
            var emissions = NdArray.Zeros(4, 2);
            for (var t = 0; t < 4; t++)
                for (var label = 0; label < 2; label++)
                    emissions[t, label] = rng.Normal() * 0.1;

            sequences.Add(emissions);
            labels.Add([0, 1, 0, 1]);
        }

        var crf = new LinearChainCrf(2);

        var before = sequences.Zip(labels).Sum(p => crf.LogLikelihood(p.First, p.Second));
        crf.Fit(sequences, labels, epochs: 100, learningRate: 0.5);
        var after = sequences.Zip(labels).Sum(p => crf.LogLikelihood(p.First, p.Second));

        Assert.True(after > before, $"training moved the likelihood from {before:F4} to {after:F4}");
    }

    [Fact]
    public void TrainingLearnsTheTransitionStructureInTheData()
    {
        // Alternating labels, with emissions carrying no signal at all — so anything the model gets
        // right must have come from the transitions.
        var sequences = new List<NdArray>();
        var labels = new List<int[]>();

        for (var s = 0; s < 50; s++)
        {
            sequences.Add(NdArray.Zeros(6, 2));
            labels.Add([0, 1, 0, 1, 0, 1]);
        }

        var crf = new LinearChainCrf(2).Fit(sequences, labels, epochs: 200, learningRate: 0.5);

        Assert.True(crf.Transition(0, 1) > crf.Transition(0, 0),
            $"0→1 scored {crf.Transition(0, 1):F4} against 0→0's {crf.Transition(0, 0):F4}");
        Assert.True(crf.Transition(1, 0) > crf.Transition(1, 1),
            $"1→0 scored {crf.Transition(1, 0):F4} against 1→1's {crf.Transition(1, 1):F4}");

        // And it decodes the pattern from featureless emissions.
        Assert.Equal([0, 1, 0, 1, 0, 1], crf.Decode(NdArray.Zeros(6, 2)));
    }

    [Fact]
    public void TrainingLeavesForbiddenTransitionsForbidden()
    {
        // A negative-infinity score has no gradient and must not acquire one, or the constraints
        // dissolve over training.
        var crf = new LinearChainCrf(2);
        crf.Forbid(0, 1);

        var sequences = Enumerable.Range(0, 20).Select(_ => NdArray.Zeros(3, 2)).ToList();
        var labels = Enumerable.Range(0, 20).Select(_ => new[] { 0, 1, 1 }).ToList();

        crf.Fit(sequences, labels, epochs: 100, learningRate: 1.0);

        Assert.True(double.IsNegativeInfinity(crf.Transition(0, 1)),
            $"the forbidden transition drifted to {crf.Transition(0, 1)}");
    }

    [Fact]
    public void AnEmptySequenceIsHandled()
    {
        var crf = new LinearChainCrf(3);
        Assert.Empty(crf.Decode(NdArray.Zeros(0, 3)));
        Assert.Equal(0.0, crf.LogPartition(NdArray.Zeros(0, 3)));
    }

    [Fact]
    public void MalformedInputsAreRejected()
    {
        var crf = new LinearChainCrf(3);

        Assert.Throws<ArgumentException>(() => crf.Decode(NdArray.Zeros(4, 5)));
        Assert.Throws<ArgumentException>(() => crf.Decode(NdArray.Zeros(4)));
        Assert.Throws<ArgumentException>(() => crf.Score(NdArray.Zeros(4, 3), [0, 1]));
        Assert.Throws<ArgumentException>(() => crf.ApplyBioConstraints(["O", "B-PER"]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LinearChainCrf(0));
    }
}
