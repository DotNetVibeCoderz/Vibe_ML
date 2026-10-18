using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviLearn.ModelSelection;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Tasks;
using Matplotlib.Net.Artists;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Office;

/// <summary>Sorting customer reviews into happy and unhappy, in English and Indonesian.</summary>
public sealed class CustomerReviewSentiment : Showcase
{
    public override string Title => "Is a trained sentiment model worth the labelling?";
    public override Domain Domain => Domain.Office;
    public override GraviLibrary[] Libraries => [GraviLibrary.GraviText, GraviLibrary.GraviFrame, GraviLibrary.GraviLearn];
    public override string Summary => "A word-list baseline against a classifier trained on labelled reviews, scored by cross-validation.";
    public override string Scenario =>
        "A customer-experience team reads hundreds of reviews a week in English and Bahasa Indonesia and needs " +
        "them sorted before anyone replies. A sentiment word list works out of the box; a classifier trained on " +
        "reviews the team has labelled learns their own vocabulary, but only if there are enough labels. Five-fold " +
        "cross-validation on the 80 reviews in datasets/imdb_reviews.csv shows which wins at that size, per language.";

    public override void Run(ShowcaseOutput output)
    {
        var reviews = DataFrame.ReadCsv(Datasets.Path("imdb_reviews.csv"));
        var text = Enumerable.Range(0, reviews.RowCount).Select(i => reviews.Text("review")[i] ?? "").ToArray();
        var truth = Enumerable.Range(0, reviews.RowCount).Select(i => reviews.Text("sentiment")[i] ?? "").ToArray();
        var language = Enumerable.Range(0, reviews.RowCount).Select(i => reviews.Text("language")[i] ?? "").ToArray();

        // Baseline: the built-in bilingual sentiment lexicon with negation handling. No training.
        var lexicon = new SentimentAnalyzer();
        var lexiconSays = text.Select(t => lexicon.AnalyzeWithLexicon(t).Label).ToArray();

        // Trained: five folds, stratified so each fold has both labels; every review is predicted by a
        // model that never saw it.
        var trainedSays = new string[text.Length];
        var codes = NdArray.FromValues(truth.Select(t => t == "positive" ? 1.0 : 0.0));
        foreach (var (train, test) in Selection.StratifiedKFold(codes, folds: 5, seed: 3))
        {
            var model = new TextClassifier(iterations: 80).Train(train.Select(i => text[i]).ToList(), train.Select(i => truth[i]).ToList());
            foreach (var i in test) trainedSays[i] = model.Predict(text[i]).Label;
        }

        double Accuracy(string[] said, string? lang) =>
            Enumerable.Range(0, text.Length).Where(i => lang is null || language[i] == lang).Average(i => said[i] == truth[i] ? 1.0 : 0.0);
        string[] groups = ["English", "Indonesian", "all"];
        string?[] codesOf = ["en", "id", null];
        var lexiconAcc = codesOf.Select(c => Accuracy(lexiconSays, c) * 100).ToArray();
        var trainedAcc = codesOf.Select(c => Accuracy(trainedSays, c) * 100).ToArray();
        var undecided = lexiconSays.Count(l => l == "neutral");

        // What the trained model leans on, read straight from its coefficients.
        var full = new TextClassifier(iterations: 80).Train(text, truth);
        var positive = full.TopFeatures("positive", 7);
        var negative = full.TopFeatures("negative", 7);

        var figure = GalleryCharts.Figure(10.5, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        var x = Enumerable.Range(0, groups.Length).Select(i => (double)i).ToArray();
        left.Bar(x.Select(v => v - 0.2).ToArray(), lexiconAcc, 0.4, label: "word list (no training)", color: Color.Parse("#8A94A3"));
        left.Bar(x.Select(v => v + 0.2).ToArray(), trainedAcc, 0.4, label: "trained classifier", color: GalleryCharts.Of(GraviLibrary.GraviText));
        left.SetXTicks(x, groups).SetYLim(0, 115);
        left.SetTitle("Accuracy, 5-fold cross-validation").SetYLabel("% of reviews labelled correctly");
        left.Legend(LegendLocation.UpperLeft, columns: 2);

        var terms = negative.Reverse().Concat(positive).ToArray();
        var weights = negative.Reverse().Select(t => -t.Weight).Concat(positive.Select(t => t.Weight)).ToArray();
        var rows = Enumerable.Range(0, terms.Length).Select(i => (double)i).ToArray();
        right.BarH(rows, weights, colors: weights.Select(w => w < 0 ? Color.Parse("#D9363E") : Color.Parse("#2E9E68")).ToArray());
        right.SetYTicks(rows, terms.Select(t => t.Term).ToArray());
        right.SetTitle("Words that move the verdict").SetXLabel("← negative      weight      positive →");
        output.Chart(figure);

        output.Metric("Reviews", $"{text.Length} ({language.Count(l => l == "en")} English, {language.Count(l => l == "id")} Indonesian)");
        output.Metric("Word list accuracy", $"{lexiconAcc[2]:F0}%", $"{undecided} reviews had no sentiment words and came out neutral");
        output.Metric("Trained accuracy", $"{trainedAcc[2]:F0}%", "each review predicted by a model that never saw it");
        output.Line(trainedAcc[2] > lexiconAcc[2]
            ? $"Training on labelled reviews lifts accuracy from {lexiconAcc[2]:F0}% to {trainedAcc[2]:F0}%: the labelling paid for itself."
            : $"With only {text.Length * 4 / 5} labelled reviews per fold the trained model reaches {trainedAcc[2]:F0}% against the word list's " +
              $"{lexiconAcc[2]:F0}%: start with the word list, and keep labelling until a trained model overtakes it.");
    }
}
