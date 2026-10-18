using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviLearn.Explain;
using Gravicode.Science.GraviNum;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Games;

/// <summary>Elo ratings for an online game's ladder: tuning K and checking that the win probabilities are honest.</summary>
public sealed class EloMatchPrediction : Showcase
{
    public override string Title => "Are the matchmaker's win chances honest?";
    public override Domain Domain => Domain.Games;
    public override GraviLibrary[] Libraries => [GraviLibrary.GraviLearn, GraviLibrary.GraviFrame, GraviLibrary.GraviNum];
    public override string Summary => "Run Elo over a season of matches, tune K by Brier score, and check calibration of the predicted odds.";
    public override string Scenario =>
        "A competitive game shows players their chance of winning before each match and pairs them by Elo " +
        "rating. The update size K is a trade-off: too small and ratings lag behind improving players, too large " +
        "and they bounce around on luck. Replaying a season of 3,000 matches for several K values and scoring the " +
        "predictions with the Brier score picks K; a calibration curve then checks that \"65% to win\" really wins " +
        "about 65% of the time.";

    public override void Run(ShowcaseOutput output)
    {
        // 24 players with hidden true skill; the result of each match follows the Elo logistic law on
        // that skill. Two players improve steadily through the season, as real players do.
        const int players = 24, matches = 3000;
        var rng = new GraviRandom(64);
        var skill = Enumerable.Range(0, players).Select(_ => rng.Normal(1500, 180)).ToArray();
        static double WinChance(double a, double b) => 1 / (1 + Math.Pow(10, (b - a) / 400));
        var schedule = new List<(int A, int B, double Win)>();
        for (var m = 0; m < matches; m++)
        {
            skill[0] += 0.08; skill[1] += 0.12;                                 // the improvers
            var a = rng.Next(players);
            var b = (a + 1 + rng.Next(players - 1)) % players;
            schedule.Add((a, b, rng.NextBernoulli(WinChance(skill[a], skill[b])) ? 1 : 0));
        }
        var log = DataFrame.FromColumns(new Dictionary<string, double[]>
        {
            ["match"] = Enumerable.Range(0, matches).Select(m => (double)m).ToArray(),
            ["player_a"] = schedule.Select(s => (double)s.A).ToArray(),
            ["player_b"] = schedule.Select(s => (double)s.B).ToArray(),
            ["a_won"] = schedule.Select(s => s.Win).ToArray(),
        });

        // Replay the season with update size K, predicting each match before seeing its result.
        (double[] Predicted, double[][] History) Replay(double k)
        {
            var rating = Enumerable.Repeat(1500.0, players).ToArray();
            var predicted = new double[matches];
            var history = Enumerable.Range(0, players).Select(_ => new double[matches]).ToArray();
            var (pa, pb, won) = (log.Numeric("player_a").Values, log.Numeric("player_b").Values, log.Numeric("a_won").Values);
            for (var m = 0; m < matches; m++)
            {
                int a = (int)pa[m], b = (int)pb[m];
                predicted[m] = WinChance(rating[a], rating[b]);
                rating[a] += k * (won[m] - predicted[m]);
                rating[b] -= k * (won[m] - predicted[m]);
                for (var p = 0; p < players; p++) history[p][m] = rating[p];
            }
            return (predicted, history);
        }

        // Score the second half of the season, once ratings have had time to settle.
        var secondHalf = Enumerable.Range(matches / 2, matches / 2).ToArray();
        var outcomes = NdArray.FromValues(secondHalf.Select(m => schedule[m].Win));
        double[] ks = [4, 8, 16, 24, 32, 48, 64, 96];
        var brier = ks.Select(k =>
        {
            var predicted = Replay(k).Predicted;
            return Calibration.BrierScore(NdArray.FromValues(secondHalf.Select(m => predicted[m])), outcomes);
        }).ToArray();
        var bestK = ks[Array.IndexOf(brier, brier.Min())];
        var (bestPredicted, history) = Replay(bestK);
        var probabilities = NdArray.FromValues(secondHalf.Select(m => bestPredicted[m]));
        var curve = Calibration.Curve(probabilities, outcomes, bins: 8);
        var ece = Calibration.ExpectedError(probabilities, outcomes, bins: 8);
        var finalRatings = NdArray.FromValues(history.Select(h => h[^1]));
        var agreement = Statistics.Correlation(finalRatings, NdArray.FromValues(skill));

        var figure = GalleryCharts.Figure(11, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        left.Plot([0, 1], [0, 1], "perfectly honest", Color.Parse("#8A94A3"), style: LineStyle.Dashed, lineWidth: 1.2);
        left.Plot(curve.Select(c => c.MeanPredicted).ToArray(), curve.Select(c => c.ObservedFraction).ToArray(), $"Elo, K = {bestK}",
                  marker: MarkerStyle.Circle, markerSize: 6);
        left.SetXLim(0, 1).SetYLim(0, 1).SetAspect(1.0);
        left.SetTitle("Calibration, second half of the season").SetXLabel("predicted chance to win").SetYLabel("share actually won");
        left.Legend(LegendLocation.UpperLeft);

        var matchAxis = Enumerable.Range(0, matches).Select(m => (double)m).ToArray();
        var shown = new[] { 1, Array.IndexOf(skill, skill.Max()), Array.IndexOf(skill, skill.Min()) }.Distinct().ToArray();
        foreach (var p in shown)
        {
            var line = right.Plot(matchAxis, history[p], p == 1 ? "improving player" : p == shown[1] ? "strongest player" : "weakest player", lineWidth: 1.3);
            right.AxHLine(skill[p], line.Color, LineStyle.Dotted, 1.0);
        }
        var span = shown.SelectMany(p => history[p].Append(skill[p])).ToArray();
        right.SetYLim(span.Min() - 40, span.Max() + 40);
        right.SetTitle($"Ratings through the season (K = {bestK}; dotted: true skill at the end)").SetXLabel("match").SetYLabel("Elo rating");
        right.Legend(LegendLocation.LowerLeft);
        output.Chart(figure);

        output.Metric("Best K", bestK.ToString(), $"Brier {brier.Min():F4}; K = {ks[0]} gives {brier[0]:F4}, K = {ks[^1]} gives {brier[^1]:F4}");
        output.Metric("Calibration error (ECE)", $"{ece * 100:F1} points", "average gap between promised and actual win rates");
        output.Metric("Ratings vs true skill", $"r = {agreement:F2}", $"{players} players after {matches:N0} matches");
        output.Line($"With K = {bestK} the displayed odds are off by {ece * 100:F1} percentage points on average, so a \"65% to win\" banner can be trusted. " +
                    "Smaller K misjudges the improving players; larger K chases luck.");
    }
}
