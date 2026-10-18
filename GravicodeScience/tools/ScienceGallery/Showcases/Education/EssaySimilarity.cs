using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Linguistics;
using Gravicode.Science.GraviText.Vectorization;
using ScienceGallery.Models;
using ScienceGallery.Services;

namespace ScienceGallery.Showcases.Education;

/// <summary>Flagging essays that are suspiciously alike, with TF-IDF vectors and cosine similarity.</summary>
public sealed class EssaySimilarity : Showcase
{
    public override string Title => "Which essays were copied from each other?";
    public override Domain Domain => Domain.Education;
    public override GraviLibrary[] Libraries => [GraviLibrary.GraviText, GraviLibrary.GraviNum];
    public override string Summary => "Turn each essay into TF-IDF weights and compare every pair by cosine similarity.";
    public override string Scenario =>
        "A teacher sets ten students the same short essay: \"Should every school teach coding?\". Everyone uses " +
        "the same vocabulary, so shared words alone prove nothing. TF-IDF weighs each word by how unusual it is " +
        "across the class, so two essays only score high together when they share the same distinctive phrasing. " +
        "Pairs above the threshold go to the teacher for a human read, not straight to a penalty.";

    public override void Run(ShowcaseOutput output)
    {
        string[] students = ["Ayu", "Bima", "Citra", "Dewi", "Eko", "Fajar", "Gita", "Hana", "Indra", "Joko"];
        string[] essays =
        [
            "Coding teaches children to break a big problem into small steps. That habit helps in mathematics, science and even essay writing. Schools should offer it to everyone, not only to students who already own a computer.",
            "I think coding should be optional. Many students struggle with reading and arithmetic, and those basics matter more. A coding club after school would serve the students who are truly interested.",
            "Programming is the literacy of the digital economy. Factories, hospitals and farms all run on software now. If schools ignore it, only wealthy families will be able to give their children this advantage.",
            "Coding teaches kids to split a large problem into little steps. This habit helps in maths, science and even writing essays. Schools ought to offer it to everybody, not just students who already have a computer.",
            "Learning to code builds patience. A program rarely works the first time, and debugging shows students that failure is part of learning. That lesson is useful long after the last line of code.",
            "Our teachers are already overloaded. Adding coding to the timetable without training teachers first would produce bad lessons and bored students. Train the teachers, then teach the subject.",
            "Programming is the literacy of the modern digital economy. Hospitals, factories and farms all run on software today. If schools ignore it, only rich families will give their kids this advantage.",
            "Games, apps and websites are things students use every day. Coding lets them understand and build those things instead of only consuming them. Creativity grows when students make something real.",
            "Not every student will become a programmer, just as not every student becomes a writer. We still teach writing to everyone because it helps people think clearly, and coding is similar.",
            "Robotics lessons at my school were the most exciting part of the week. We built a line-following robot and learned about sensors, motors and loops. Every school deserves lessons like that.",
        ];

        // Stop words and stemming make "teaches"/"teach" and "the"/"a" stop counting as evidence.
        var vectorizer = new TfidfVectorizer(new VectorizerOptions
        {
            StopWords = StopWords.English,
            Stemmer = PorterStemmer.Stem,
            MinNGram = 1,
            MaxNGram = 2,            // word pairs catch copied phrasing, not just copied words
        }, sublinearTf: true);
        var matrix = vectorizer.FitTransform(essays);

        var n = essays.Length;
        var similarity = new double[n, n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
                similarity[i, j] = Similarity.Cosine(matrix, i, j);

        // Every distinct pair, most similar first. The threshold sits far above the class's typical
        // overlap: a value most honest pairs never reach.
        var pairs = (from i in Enumerable.Range(0, n) from j in Enumerable.Range(i + 1, n - i - 1)
                     select (A: i, B: j, Score: similarity[i, j])).OrderByDescending(p => p.Score).ToList();
        var typical = Statistics.Median(NdArray.FromValues(pairs.Select(p => p.Score)));
        const double threshold = 0.3;
        var flagged = pairs.Where(p => p.Score >= threshold).ToList();

        var figure = GalleryCharts.Figure(7.6, 6.2);
        var ax = figure.AddSubplot();
        ax.Heatmap(similarity, students, students, colormap: "viridis", annotate: true, annotationFormat: "0.00");
        ax.AddColorbar("cosine similarity");
        ax.SetTitle("Essay-to-essay similarity (TF-IDF, words and word pairs)");
        output.Chart(figure);

        output.Metric("Vocabulary", $"{matrix.Shape[1]} terms", "stemmed words and two-word phrases, stop words removed");
        output.Metric("Typical pair similarity", typical.ToString("F2"), "median over all 45 pairs");
        output.Metric("Pairs over threshold", $"{flagged.Count} (≥ {threshold:F2})");
        foreach (var pair in flagged)
        {
            // The evidence a teacher can check: the phrases both essays lean on hardest.
            var shared = Enumerable.Range(0, vectorizer.FeatureCount)
                .Where(t => vectorizer.Vocabulary[t].Contains(' '))
                .OrderByDescending(t => matrix[pair.A, t] * matrix[pair.B, t])
                .TakeWhile(t => matrix[pair.A, t] * matrix[pair.B, t] > 0).Take(4)
                .Select(t => $"\"{vectorizer.Vocabulary[t]}\"");
            output.Line($"{students[pair.A]} and {students[pair.B]}: similarity {pair.Score:F2}, " +
                        $"{pair.Score / typical:F0}× the class median. Shared phrases (stemmed): {string.Join(", ", shared)}.");
        }
    }
}
