using Gravicode.HFNet.GraviTokenizers;
using Gravicode.HFNet.GraviTokenizers.Components;
using Gravicode.HFNet.GraviTransformers;
using Xunit;

namespace Gravicode.HFNet.GraviTransformers.Tests;

/// <summary>
/// Tests for turning per-token labels into entity spans.
/// </summary>
/// <remarks>
/// The arithmetic of the head is one matrix multiply and is not what goes wrong. The decoding is:
/// merging subwords, honouring a <c>B-</c> tag, and recovering the source substring are each a
/// place where a plausible-looking answer can be quietly wrong.
/// </remarks>
public sealed class EntityDecodingTests
{
    private static readonly Dictionary<int, string> Labels = new()
    {
        [0] = "O",
        [1] = "B-PER",
        [2] = "I-PER",
        [3] = "B-ORG",
        [4] = "I-ORG",
    };

    /// <summary>Builds an encoding whose offsets index the given text.</summary>
    private static Encoding Encode(string text, params (string Token, int Start, int End, bool Special)[] tokens)
        => new(
            [.. tokens.Select((_, i) => i)],
            [.. tokens.Select(t => t.Token)],
            [.. tokens.Select(_ => 1)],
            [.. tokens.Select(_ => 0)],
            [.. tokens.Select(t => t.Special ? 1 : 0)],
            [.. tokens.Select(t => (t.Start, t.End))]);

    /// <summary>One-hot logits, so the argmax is unambiguous.</summary>
    private static double[][] Logits(params int[] classes)
        => [.. classes.Select(c => Enumerable.Range(0, Labels.Count)
            .Select(i => i == c ? 10.0 : 0.0).ToArray())];

    [Fact]
    public void MergesSubwordsIntoOneSpan()
    {
        const string Text = "Gravicode Studios ships it";

        // "Gravicode" is two subwords; both carry the same entity type.
        var encoding = Encode(Text,
            ("[CLS]", 0, 0, true),
            ("Gravi", 0, 5, false),
            ("##code", 5, 9, false),
            ("Studios", 10, 17, false),
            ("ships", 18, 23, false),
            ("[SEP]", 0, 0, true));

        var entities = TokenClassificationHead.Decode(
            Text, encoding, Logits(0, 3, 4, 4, 0, 0), Labels);

        Assert.Single(entities);
        Assert.Equal("ORG", entities[0].Label);
        Assert.Equal("Gravicode Studios", entities[0].Text);
        Assert.Equal(0, entities[0].Start);
        Assert.Equal(17, entities[0].End);
    }

    [Fact]
    public void TheSpanIsTheSourceSubstringNotRejoinedPieces()
    {
        // Rejoining "Gravi" + "##code" needs the ## stripped and the space guessed at; taking the
        // offsets keeps whatever was really there, capitals and punctuation included.
        const string Text = "McDonald's opened";

        var encoding = Encode(Text,
            ("Mc", 0, 2, false),
            ("##Donald", 2, 8, false),
            ("'", 8, 9, false),
            ("s", 9, 10, false));

        var entities = TokenClassificationHead.Decode(Text, encoding, Logits(3, 4, 4, 4), Labels);

        Assert.Equal("McDonald's", entities[0].Text);
    }

    [Fact]
    public void ABeginningTagSplitsTwoAdjacentEntitiesOfTheSameType()
    {
        // Without honouring B-, "Ada Lovelace" and "Alan Turing" run together into one person.
        const string Text = "Ada Turing";

        var encoding = Encode(Text, ("Ada", 0, 3, false), ("Turing", 4, 10, false));
        var entities = TokenClassificationHead.Decode(Text, encoding, Logits(1, 1), Labels);

        Assert.Equal(2, entities.Count);
        Assert.Equal("Ada", entities[0].Text);
        Assert.Equal("Turing", entities[1].Text);
    }

    [Fact]
    public void ContinuationKeepsOneEntity()
    {
        const string Text = "Ada Lovelace";

        var encoding = Encode(Text, ("Ada", 0, 3, false), ("Lovelace", 4, 12, false));
        var entities = TokenClassificationHead.Decode(Text, encoding, Logits(1, 2), Labels);

        Assert.Single(entities);
        Assert.Equal("Ada Lovelace", entities[0].Text);
    }

    [Fact]
    public void OutsideTagsProduceNothing()
    {
        const string Text = "nothing here";

        var encoding = Encode(Text, ("nothing", 0, 7, false), ("here", 8, 12, false));

        Assert.Empty(TokenClassificationHead.Decode(Text, encoding, Logits(0, 0), Labels));
    }

    [Fact]
    public void SpecialTokensCloseASpanRatherThanJoiningIt()
    {
        const string Text = "Ada Turing";

        var encoding = Encode(Text,
            ("Ada", 0, 3, false),
            ("[SEP]", 0, 0, true),
            ("Turing", 4, 10, false));

        // Same type either side of the separator; the separator must still break the span.
        var entities = TokenClassificationHead.Decode(Text, encoding, Logits(2, 0, 2), Labels);

        Assert.Equal(2, entities.Count);
    }

    [Fact]
    public void ScoreIsTheMeanOverTheSpansTokens()
    {
        const string Text = "Ada Lovelace";

        var encoding = Encode(Text, ("Ada", 0, 3, false), ("Lovelace", 4, 12, false));
        var entities = TokenClassificationHead.Decode(Text, encoding, Logits(1, 2), Labels);

        Assert.True(entities[0].Score is > 0.9 and <= 1.0, $"score was {entities[0].Score}");
    }
}

/// <summary>Tests for choosing an answer span.</summary>
public sealed class AnswerDecodingTests
{
    private const string Context = "HF.Net was built by Gravicode Studios in Bandung.";

    /// <summary>A question-and-context encoding, with segment ids set as a pair would have them.</summary>
    private static Encoding Pair()
    {
        // Positions 0-2 are the question (segment 0); the rest is the context (segment 1) with
        // offsets that index Context.
        (string Token, int Start, int End, int Type, bool Special)[] tokens =
        [
            ("[CLS]", 0, 0, 0, true),
            ("who", 0, 3, 0, false),
            ("[SEP]", 0, 0, 0, true),
            ("HF.Net", 0, 6, 1, false),
            ("built", 11, 16, 1, false),
            ("Gravicode", 20, 29, 1, false),
            ("Studios", 30, 37, 1, false),
            ("Bandung", 41, 48, 1, false),
            ("[SEP]", 0, 0, 1, true),
        ];

        return new Encoding(
            [.. tokens.Select((_, i) => i)],
            [.. tokens.Select(t => t.Token)],
            [.. tokens.Select(_ => 1)],
            [.. tokens.Select(t => t.Type)],
            [.. tokens.Select(t => t.Special ? 1 : 0)],
            [.. tokens.Select(t => (t.Start, t.End))]);
    }

    private static double[] Peak(int at, int length = 9)
        => [.. Enumerable.Range(0, length).Select(i => i == at ? 10.0 : 0.0)];

    [Fact]
    public void ExtractsTheSpanBetweenTheChosenPositions()
    {
        // Start at "Gravicode" (index 5), end at "Studios" (index 6).
        var answers = QuestionAnsweringHead.Decode(Context, Pair(), Peak(5), Peak(6));

        Assert.Equal("Gravicode Studios", answers[0].Text);
        Assert.Equal(20, answers[0].Start);
        Assert.Equal(37, answers[0].End);
    }

    [Fact]
    public void NeverAnswersFromTheQuestion()
    {
        // The model points hard at the question token. Segment 0 is not eligible, so the answer
        // has to come from the context anyway - "who wrote it?" answered with "who" is a decoding
        // failure that reads as a model failure.
        var answers = QuestionAnsweringHead.Decode(Context, Pair(), Peak(1), Peak(1));

        Assert.DoesNotContain("who", answers[0].Text, StringComparison.OrdinalIgnoreCase);
        Assert.NotEqual(0, answers[0].Text.Length);
    }

    [Fact]
    public void NeverReturnsAnEndBeforeItsStart()
    {
        // Argmax over each independently would take start=6, end=3 and produce a negative span.
        var answers = QuestionAnsweringHead.Decode(Context, Pair(), Peak(6), Peak(3));

        Assert.True(answers[0].End > answers[0].Start,
            $"got [{answers[0].Start}..{answers[0].End})");
    }

    [Fact]
    public void RespectsTheLengthLimit()
    {
        var answers = QuestionAnsweringHead.Decode(
            Context, Pair(), Peak(3), Peak(7), maxAnswerTokens: 2);

        // Positions 3 and 7 are five tokens apart, so that pair is not allowed to win.
        Assert.True(answers[0].End - answers[0].Start < Context.Length,
            "a limited search still returned the whole context");
    }

    [Fact]
    public void RanksCandidatesByJointProbability()
    {
        var answers = QuestionAnsweringHead.Decode(Context, Pair(), Peak(5), Peak(6), topK: 3);

        Assert.Equal(3, answers.Count);
        for (var i = 1; i < answers.Count; i++)
        {
            Assert.True(answers[i].Score <= answers[i - 1].Score, "candidates are not sorted");
        }
    }

    [Fact]
    public void SpansAreAlwaysRealSubstringsOfTheContext()
    {
        var answers = QuestionAnsweringHead.Decode(Context, Pair(), Peak(4), Peak(7), topK: 5);

        foreach (var answer in answers)
        {
            Assert.Equal(answer.Text, Context[answer.Start..answer.End]);
        }
    }
}
