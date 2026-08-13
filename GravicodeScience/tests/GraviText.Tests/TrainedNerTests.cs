using Gravicode.Science.GraviText.Tasks;
using Xunit;

namespace Gravicode.Science.Tests.GraviText;

/// <summary>
/// Tests for the trained named-entity recogniser.
/// </summary>
/// <remarks>
/// Scored on held-out sentences and on whole entities, not on tokens: token accuracy is dominated
/// by the <c>O</c> tag and a model that predicts nothing scores above 80%. The claim that separates
/// this from the rule-based recogniser is generalisation to unseen names, so that is tested with
/// names that appear nowhere in the corpus.
/// </remarks>
public class TrainedNerTests
{
    private static string CorpusPath => Path.Combine(
        AppContext.BaseDirectory, "..", "..", "..", "..", "..", "datasets", "ner_conll.txt");

    private static IReadOnlyList<TaggedSentence> Corpus() => TaggedSentence.LoadConll(CorpusPath);

    private static (IReadOnlyList<TaggedSentence> Train, IReadOnlyList<TaggedSentence> Test) Split()
    {
        var all = Corpus();
        var cut = (int)(all.Count * 0.75);
        return ([.. all.Take(cut)], [.. all.Skip(cut)]);
    }

    private static TrainedNer Fitted()
    {
        var (train, _) = Split();
        return new TrainedNer().Fit(train);
    }

    [Fact]
    public void TheCorpusLoadsWithTheExpectedShape()
    {
        var corpus = Corpus();

        Assert.Equal(420, corpus.Count);
        Assert.All(corpus, s => Assert.NotEmpty(s.Tokens));

        // Every tag is either O or a well-formed BIO tag over the three types.
        var tags = corpus.SelectMany(s => s.Tags).Distinct().ToHashSet();
        Assert.Contains("O", tags);
        Assert.All(tags, tag => Assert.True(
            tag == "O" || tag.StartsWith("B-", StringComparison.Ordinal)
                       || tag.StartsWith("I-", StringComparison.Ordinal),
            $"'{tag}' is not a BIO tag"));
    }

    [Fact]
    public void HeldOutEntityF1IsAboveNinetyPercent()
    {
        // The headline number, on sentences the model never saw. Entity-level, so the boundaries
        // and the type all have to be right.
        var (train, test) = Split();
        var ner = new TrainedNer().Fit(train);

        var score = ner.Evaluate(test);

        Assert.True(score.F1 > 0.90, $"held-out entity F1 was {score.F1:P1} ({score})");
        Assert.True(score.Precision > 0.90, $"precision was {score.Precision:P1}");
        Assert.True(score.Recall > 0.90, $"recall was {score.Recall:P1}");
    }

    [Fact]
    public void TokenAccuracyIsReportedButIsNotTheInterestingNumber()
    {
        // Pinning the gap deliberately: a model can score far higher on tokens than on entities,
        // which is why the entity measure is the one to judge by.
        var (train, test) = Split();
        var ner = new TrainedNer().Fit(train);

        var tokenAccuracy = ner.TokenAccuracy(test);
        Assert.True(tokenAccuracy > 0.95, $"token accuracy was {tokenAccuracy:P2}");

        // An all-O baseline scores this well on tokens and zero on entities.
        var allO = test.SelectMany(s => s.Tags).Count(t => t == "O");
        var total = test.Sum(s => s.Tokens.Count);
        Assert.True((double)allO / total > 0.5,
            "the corpus is not O-dominated, so this test's premise does not hold");
    }

    [Fact]
    public void UnseenNamesAreRecognisedFromTheirShapeAndContext()
    {
        // The claim that separates a trained tagger from a gazetteer. None of these names appears
        // anywhere in the corpus, so anything found came from the features rather than from memory.
        var ner = Fitted();
        var corpus = string.Join(' ', Corpus().SelectMany(s => s.Words));

        foreach (var name in new[] { "Kartika", "Wirawan", "Zulkarnain" })
        {
            Assert.DoesNotContain(name, corpus, StringComparison.Ordinal);

            var entities = ner.Recognize($"{name} tinggal di Bandung .");
            Assert.Contains(entities, e => e.Type == "PER" && e.Text.Contains(name, StringComparison.Ordinal));
        }
    }

    [Fact]
    public void MultiTokenEntitiesAreKeptWhole()
    {
        // The reason BIO tagging exists. A two-word name must come back as one entity, not two.
        var ner = Fitted();
        var entities = ner.Recognize("Rina Pratama bekerja di Bukit Asam .");

        var person = Assert.Single(entities, e => e.Type == "PER");
        Assert.Equal("Rina Pratama", person.Text);

        var organisation = Assert.Single(entities, e => e.Type == "ORG");
        Assert.Equal("Bukit Asam", organisation.Text);
    }

    [Fact]
    public void EntityTypesAreDistinguishedByContext()
    {
        // Both are capitalised single words in the same position; only the surrounding words say
        // which is a place and which is a company.
        var ner = Fitted();

        var location = ner.Recognize("Budi tinggal di Surabaya .");
        Assert.Contains(location, e => e.Type == "LOC" && e.Text == "Surabaya");

        var organisation = ner.Recognize("Budi bekerja di Tokopedia .");
        Assert.Contains(organisation, e => e.Type == "ORG" && e.Text == "Tokopedia");
    }

    [Fact]
    public void ASentenceWithNoEntitiesProducesNone()
    {
        // The failure mode a capitalisation heuristic has: tagging the sentence-initial word.
        var ner = Fitted();
        Assert.Empty(ner.Recognize("saya pergi ke sana kemarin ."));
    }

    [Fact]
    public void EveryPredictedTagSequenceObeysTheBioScheme()
    {
        // What the CRF's constraints buy. An I-X may never open a sequence, follow an O, or follow
        // a B-Y of a different type — and this checks every sentence in the corpus, not a sample.
        var ner = Fitted();

        foreach (var sentence in Corpus())
        {
            var tags = ner.Tag(sentence.Words);

            for (var i = 0; i < tags.Count; i++)
            {
                if (!tags[i].StartsWith("I-", StringComparison.Ordinal)) continue;

                Assert.True(i > 0, $"'{sentence}' opened with the continuation tag {tags[i]}");

                var previous = tags[i - 1];
                var valid = (previous.StartsWith("B-", StringComparison.Ordinal)
                             || previous.StartsWith("I-", StringComparison.Ordinal))
                            && previous[2..] == tags[i][2..];

                Assert.True(valid, $"'{sentence}': {tags[i]} followed {previous}");
            }
        }
    }

    [Fact]
    public void TrainingIsDeterministic()
    {
        var (train, test) = Split();

        var first = new TrainedNer().Fit(train).Evaluate(test);
        var second = new TrainedNer().Fit(train).Evaluate(test);

        Assert.Equal(first.F1, second.F1, 12);
    }

    [Fact]
    public void TheFeatureSetIsBoundedAndTheLabelSetIsComplete()
    {
        var ner = Fitted();

        Assert.Contains("O", ner.Labels);
        Assert.Contains("B-PER", ner.Labels);
        Assert.Contains("B-LOC", ner.Labels);
        Assert.Contains("B-ORG", ner.Labels);

        // Shape features collapse many words onto the same pattern, so the vocabulary stays far
        // smaller than the token count.
        var tokens = Corpus().Sum(s => s.Tokens.Count);
        Assert.True(ner.FeatureCount < tokens,
            $"{ner.FeatureCount} features for {tokens} tokens suggests the features are memorising");
    }

    [Fact]
    public void ScoringIsStrictAboutBoundaries()
    {
        // An entity counts only when its start, end and type all match. A model that finds the
        // right type over the wrong span scores nothing, which is what the CoNLL measure means.
        var gold = new TaggedSentence(
        [
            new TaggedToken("Rina", "B-PER"),
            new TaggedToken("Pratama", "I-PER"),
            new TaggedToken("hadir", "O"),
        ]);

        var ner = Fitted();
        var score = ner.Evaluate([gold]);

        // Either the model got the whole span or it scored zero; there is no partial credit.
        Assert.True(score.F1 is 0.0 or 1.0, $"strict scoring produced a partial F1 of {score.F1}");
    }

    [Fact]
    public void UsingTheModelBeforeFittingThrows()
    {
        Assert.Throws<InvalidOperationException>(() => new TrainedNer().Tag(["Budi"]));
        Assert.Throws<ArgumentException>(() => new TrainedNer().Fit([]));
    }

    [Fact]
    public void AnEmptySentenceTagsToNothing()
    {
        Assert.Empty(Fitted().Tag([]));
    }
}
