using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviLearn.Clustering;
using Gravicode.Science.GraviLearn.Decomposition;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Linguistics;
using Gravicode.Science.GraviText.Vectorization;
using Matplotlib.Net.Artists;
using ScienceGallery.Models;
using ScienceGallery.Services;

namespace ScienceGallery.Showcases.Office;

/// <summary>Grouping a shared inbox into topics nobody labelled, with TF-IDF and k-means.</summary>
public sealed class EmailTopicClusters : Showcase
{
    public override string Title => "Sort a shared inbox into topics nobody labelled";
    public override Domain Domain => Domain.Office;
    public override GraviLibrary[] Libraries => [GraviLibrary.GraviText, GraviLibrary.GraviLearn, GraviLibrary.GraviNum];
    public override string Summary => "TF-IDF vectors, k-means clusters and the top terms of each cluster name the topics.";
    public override string Scenario =>
        "The office's shared mailbox gets a mix of IT problems, supplier invoices, meeting requests and HR " +
        "questions, and nobody has time to tag them. Clustering the TF-IDF vectors of the emails finds the " +
        "groups without any labels; the heaviest terms of each cluster centre tell the office manager what each " +
        "group is about, so routing rules can be written for the four queues.";

    public override void Run(ShowcaseOutput output)
    {
        string[][] topics =
        [
            ["my laptop will not connect to the vpn", "the password reset link has expired", "the printer on floor three is jammed again",
             "outlook keeps crashing when I open attachments", "I need access to the shared drive", "the wifi in the meeting room is very slow",
             "my screen flickers after the update", "please install the software licence on my laptop"],
            ["the invoice for last month is attached", "this payment is overdue by thirty days", "please confirm the bank transfer",
             "the purchase order number is missing on the invoice", "a reminder of the outstanding balance on our invoice",
             "remittance advice for the payment is attached", "a credit note for the returned goods", "please update our billing address for invoices"],
            ["can we move the meeting to thursday", "calendar invite for the quarterly review meeting", "the agenda for tomorrow's call is attached",
             "please book the large room for the workshop", "the client call is rescheduled to three pm", "minutes from today's meeting",
             "are you free next week for a short call", "the video link for the workshop on friday"],
            ["annual leave request for next month", "I am on sick leave today", "my payslip shows the wrong overtime hours",
             "the onboarding checklist for our new employee", "enrolment in the leadership training course", "an update to the health insurance policy",
             "the deadline for the performance review form", "a question about the remote work policy"],
        ];
        string[] openers = ["hi team", "dear all", "good morning", "hello"];
        string[] closers = ["thanks", "best regards", "please let me know as soon as possible", "many thanks in advance"];

        // 140 emails: a greeting, two or three sentences from one topic, a sign-off.
        var rng = new GraviRandom(23);
        var emails = new List<string>();
        var truth = new List<int>();
        for (var i = 0; i < 140; i++)
        {
            var topic = rng.Next(topics.Length);
            var body = rng.Choice(topics[topic].Length, 2 + rng.Next(2), replace: false).Select(k => topics[topic][k]);
            emails.Add($"{openers[rng.Next(openers.Length)]}, {string.Join(". ", body)}. {closers[rng.Next(closers.Length)]}");
            truth.Add(topic);
        }

        var vectorizer = new TfidfVectorizer(new VectorizerOptions { StopWords = StopWords.English, MinDocumentFrequency = 2 });
        var x = vectorizer.FitTransform(emails);

        var kmeans = new KMeans(clusters: 4, restarts: 10, seed: 1);
        var labels = kmeans.FitPredict(x).ToArray().Select(v => (int)v).ToArray();

        // Name each cluster by the heaviest terms of its centre.
        var names = Enumerable.Range(0, 4).Select(c => string.Join(", ",
            Enumerable.Range(0, vectorizer.FeatureCount).OrderByDescending(t => kmeans.Centroids[c, t]).Take(3)
                      .Select(t => vectorizer.Vocabulary[t]))).ToArray();
        // Purity: the share of emails whose cluster's majority topic is their own topic.
        var purity = Enumerable.Range(0, 4).Sum(c => Enumerable.Range(0, 4).Max(t =>
            labels.Where((l, i) => l == c && truth[i] == t).Count())) / (double)emails.Count;
        var silhouette = Metrics.SilhouetteScore(x, NdArray.FromValues(labels.Select(l => (double)l)));

        // Two principal components are enough to see the separation on a page.
        var projected = new PrincipalComponentAnalysis(components: 2).FitTransform(x);
        var figure = GalleryCharts.Figure(11, 5.2);
        var ax = figure.AddSubplot();
        for (var c = 0; c < 4; c++)
        {
            var members = Enumerable.Range(0, emails.Count).Where(i => labels[i] == c).ToArray();
            ax.Scatter(members.Select(i => projected[i, 0]).ToArray(), members.Select(i => projected[i, 1]).ToArray(),
                       $"{names[c]} ({members.Length})", sizes: [36], alpha: 0.8);
        }
        ax.SetTitle("Inbox emails, coloured by k-means cluster").SetXLabel("principal component 1").SetYLabel("principal component 2");
        ax.Legend(LegendLocation.OutsideRight, title: "cluster (top terms)");
        output.Chart(figure);

        output.Metric("Emails / terms", $"{emails.Count} / {vectorizer.FeatureCount}", "terms in at least two emails, stop words removed");
        output.Metric("Cluster purity", $"{purity * 100:F0}%", "against the topic each email was really about");
        output.Metric("Silhouette score", silhouette.ToString("F2"), "above 0 means emails sit closer to their own cluster");
        output.Line("The cluster names come from the centroids alone: " + string.Join("; ", names.Select((n, c) => $"cluster {c + 1} → {n}")) + ".");
    }
}
