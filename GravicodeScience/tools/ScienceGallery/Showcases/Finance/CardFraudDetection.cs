using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviLearn.Anomaly;
using Gravicode.Science.GraviLearn.Preprocessing;
using Gravicode.Science.GraviNum;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Finance;

/// <summary>Learning what normal card use looks like, and flagging what does not, with a one-class SVM.</summary>
public sealed class CardFraudDetection : Showcase
{
    public override string Title => "Flag card payments that do not look like the cardholder";
    public override Domain Domain => Domain.Finance;
    public override GraviLibrary[] Libraries => [GraviLibrary.GraviLearn, GraviLibrary.GraviNum];
    public override string Summary => "A one-class SVM learns normal spending and ranks new payments by how unusual they are.";
    public override string Scenario =>
        "A card issuer sees thousands of payments for every confirmed fraud, and new fraud patterns appear " +
        "faster than they can be labelled. A one-class SVM needs only normal history: it learns the region where " +
        "ordinary spending lives (amount, time of day, distance from home, recent activity) and scores how far " +
        "each new payment falls outside it. The fraud team reviews the highest scores first; a fixed amount " +
        "limit is shown for comparison.";

    public override void Run(ShowcaseOutput output)
    {
        var rng = new GraviRandom(55);
        // One payment: amount (Rp), hour of day, distance from home (km), payments in the past 24 hours.
        (double Amount, double Hour, double Km, double Recent) Normal() => (
            Math.Exp(rng.Normal(Math.Log(180_000), 0.9)),
            Math.Clamp(rng.Normal(14, 3.5), 6, 23.5),
            Math.Exp(rng.Normal(Math.Log(4), 0.8)),
            rng.Poisson(2.5));
        (double Amount, double Hour, double Km, double Recent) Fraud() => (
            Math.Exp(rng.Normal(Math.Log(rng.NextBernoulli(0.5) ? 2_500_000 : 400_000), 0.6)),
            rng.NextBernoulli(0.6) ? rng.Uniform(0, 5) : rng.Uniform(6, 24),
            Math.Exp(rng.Normal(Math.Log(rng.NextBernoulli(0.5) ? 300 : 15), 0.7)),
            rng.Poisson(rng.NextBernoulli(0.5) ? 9 : 3));
        // Features the model sees. Hour goes on a circle, so 23:00 and 01:00 are neighbours; amounts and
        // distances on a log scale, because a Rp 10,000 difference matters only for small payments.
        double[] Features((double Amount, double Hour, double Km, double Recent) p) =>
            [Math.Log(p.Amount), Math.Sin(2 * Math.PI * p.Hour / 24), Math.Cos(2 * Math.PI * p.Hour / 24), Math.Log(p.Km), p.Recent];
        NdArray Matrix(IEnumerable<double[]> rows) { var list = rows.ToList(); return new NdArray(list.SelectMany(r => r).ToArray(), list.Count, 5); }

        var history = Enumerable.Range(0, 1200).Select(_ => Normal()).ToList();
        var incoming = Enumerable.Range(0, 1000).Select(_ => (Payment: Normal(), Fraud: false))
            .Concat(Enumerable.Range(0, 25).Select(_ => (Payment: Fraud(), Fraud: true))).ToList();

        var scaler = new StandardScaler();
        var train = scaler.FitTransform(Matrix(history.Select(Features)));
        var detector = new OneClassSvm(nu: 0.02).Fit(train);      // allow ~2% of normal history outside the boundary
        var test = scaler.Transform(Matrix(incoming.Select(p => Features(p.Payment))));
        var score = detector.DecisionFunction(test).ToArray().Select(v => -v).ToArray();   // higher = more unusual
        var flagged = detector.Predict(test).ToArray().Select(v => v < 0).ToArray();
        var truth = NdArray.FromValues(incoming.Select(p => p.Fraud ? 1.0 : 0.0));

        var auc = Metrics.RocAucScore(truth, NdArray.FromValues(score));
        var (fpr, tpr, _) = Metrics.RocCurve(truth, NdArray.FromValues(score));
        var caught = incoming.Where((p, i) => p.Fraud && flagged[i]).Count();
        var falseAlarms = incoming.Where((p, i) => !p.Fraud && flagged[i]).Count();
        // The rule it replaces: block anything over Rp 2 million.
        var ruleHits = incoming.Count(p => p.Fraud && p.Payment.Amount > 2_000_000);
        var ruleAlarms = incoming.Count(p => !p.Fraud && p.Payment.Amount > 2_000_000);

        var figure = GalleryCharts.Figure(11, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        var normalIdx = Enumerable.Range(0, incoming.Count).Where(i => !incoming[i].Fraud).ToArray();
        var fraudIdx = Enumerable.Range(0, incoming.Count).Where(i => incoming[i].Fraud).ToArray();
        left.Scatter(normalIdx.Select(i => incoming[i].Payment.Hour).ToArray(), normalIdx.Select(i => incoming[i].Payment.Amount / 1000).ToArray(),
                     "normal payment (colour: anomaly score)", sizes: [12], colorValues: normalIdx.Select(i => score[i]).ToArray(), colormap: "viridis", alpha: 0.8);
        left.Scatter(fraudIdx.Select(i => incoming[i].Payment.Hour).ToArray(), fraudIdx.Select(i => incoming[i].Payment.Amount / 1000).ToArray(),
                     "confirmed fraud", Color.Parse("#D9363E"), sizes: [55], marker: MarkerStyle.Cross);
        left.SetYScale("log").SetYLim(8, incoming.Max(p => p.Payment.Amount) / 1000 * 2).SetXLim(0, 24).SetTitle("Incoming payments").SetXLabel("hour of day").SetYLabel("amount (Rp thousand)");
        left.Legend(LegendLocation.LowerLeft);

        right.Plot(fpr, tpr, $"one-class SVM (AUC {auc:F2})", GalleryCharts.Of(GraviLibrary.GraviLearn));
        right.Scatter([ruleAlarms / 1000.0], [ruleHits / 25.0], "rule: amount > Rp 2 million", Color.Parse("#E8A317"), sizes: [90], marker: MarkerStyle.Square);
        right.Scatter([falseAlarms / 1000.0], [caught / 25.0], "SVM at its own boundary", Color.Parse("#D9363E"), sizes: [90], marker: MarkerStyle.Diamond);
        right.Plot([0, 1], [0, 1], color: Color.Parse("#C9D0D8"), style: LineStyle.Dashed, lineWidth: 1.0);
        right.SetTitle("Frauds caught against false alarms").SetXLabel("share of normal payments flagged").SetYLabel("share of frauds caught");
        right.Legend(LegendLocation.LowerRight);
        output.Chart(figure);

        output.Metric("Frauds caught", $"{caught} of 25", $"with {falseAlarms} of 1,000 normal payments flagged");
        output.Metric("Amount rule", $"{ruleHits} of 25", $"with {ruleAlarms} false alarms");
        output.Metric("ROC AUC", auc.ToString("F3"), "ranking quality, for choosing a different review budget");
        output.Line($"Trained without a single fraud example, the SVM catches {caught * 4}% of frauds at a {falseAlarms / 10.0:F1}% review rate; the amount " +
                    $"limit catches {ruleHits * 4}%, because night-time payments far from home are often small.");
    }
}
