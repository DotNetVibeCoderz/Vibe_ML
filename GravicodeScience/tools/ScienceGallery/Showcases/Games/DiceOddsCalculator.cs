using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Games;

/// <summary>Exact odds for tabletop dice rolls by convolution, checked by simulation and a goodness-of-fit test.</summary>
public sealed class DiceOddsCalculator : Showcase
{
    public override string Title => "What are the exact odds of this dice roll?";
    public override Domain Domain => Domain.Games;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Convolve single-die distributions for exact sums, then confirm with a million simulated rolls.";
    public override string Scenario =>
        "A tabletop game designer balances an attack of 3d6 + 2d8 + 2 against an armour class of 20, and " +
        "compares it with an \"advantage\" attack that rolls twice and keeps the better result. The distribution " +
        "of a sum of dice is the convolution of the individual dice, so the exact odds come from a few short " +
        "convolutions. A million simulated rolls and a chi-square test confirm them, which is the check a " +
        "designer runs before trusting a homemade formula.";

    public override void Run(ShowcaseOutput output)
    {
        double[] Die(int sides) => Enumerable.Repeat(1.0 / sides, sides).ToArray();   // P(face) for faces 1..sides

        // Sum of independent dice = convolution of their distributions. Index 0 of the result is the
        // smallest possible total (the number of dice); +2 is the attack bonus.
        var attack = new[] { Die(6), Die(6), Die(6), Die(8), Die(8) }.Aggregate((a, b) => Signal.Convolve(a, b));
        var lowest = 5 + 2;
        var totals = Enumerable.Range(lowest, attack.Length).Select(t => (double)t).ToArray();
        double Beat(double[] p, int target) => p.Where((_, i) => totals[i] >= target).Sum();

        // Advantage: roll the whole attack twice, keep the better. P(max ≤ t) = P(single ≤ t)².
        var cumulative = attack.Select((_, i) => attack.Take(i + 1).Sum()).ToArray();
        var advantage = cumulative.Select((c, i) => c * c - (i == 0 ? 0 : cumulative[i - 1] * cumulative[i - 1])).ToArray();

        // A million simulated attacks.
        var rng = new GraviRandom(20);
        const int rolls = 1_000_000;
        var counts = new double[attack.Length];
        for (var r = 0; r < rolls; r++)
        {
            var total = 2;
            for (var d = 0; d < 3; d++) total += 1 + rng.Next(6);
            for (var d = 0; d < 2; d++) total += 1 + rng.Next(8);
            counts[total - lowest]++;
        }
        // Chi-square goodness of fit of the simulation to the exact distribution (sparse tails pooled).
        var expected = attack.Select(p => p * rolls).ToArray();
        var keep = Enumerable.Range(0, attack.Length).Where(i => expected[i] >= 5).ToArray();
        var fit = Stats.Chisquare(keep.Select(i => counts[i]).ToArray(), keep.Select(i => expected[i] * counts.Sum() / keep.Sum(j => expected[j])).ToArray());

        var figure = GalleryCharts.Figure(10.5, 4.8);
        var ax = figure.AddSubplot();
        ax.Bar(totals, counts.Select(c => c / rolls * 100).ToArray(), 0.8, label: "1,000,000 simulated rolls", color: Color.Parse("#C9D0D8"));
        ax.Plot(totals, attack.Select(p => p * 100).ToArray(), "exact: 3d6 + 2d8 + 2", marker: MarkerStyle.Circle, markerSize: 4);
        ax.Plot(totals, advantage.Select(p => p * 100).ToArray(), "exact: with advantage", marker: MarkerStyle.Circle, markerSize: 4);
        ax.AxVLine(19.5, Color.Parse("#D9363E"), LineStyle.Dashed, 1.4, "armour class 20");
        ax.SetTitle("Distribution of the attack total").SetXLabel("total").SetYLabel("probability (%)");
        ax.Legend(LegendLocation.UpperLeft);
        output.Chart(figure);

        var mean = totals.Zip(attack, (t, p) => t * p).Sum();
        output.Metric("Chance to hit AC 20", $"{Beat(attack, 20) * 100:F2}%", $"simulated {counts.Where((_, i) => totals[i] >= 20).Sum() / rolls * 100:F2}%");
        output.Metric("With advantage", $"{Beat(advantage, 20) * 100:F2}%", $"+{(Beat(advantage, 20) - Beat(attack, 20)) * 100:F1} points");
        output.Metric("Average total", mean.ToString("F2"), $"range {totals[0]}–{totals[^1]}");
        output.Metric("Chi-square p", fit.Pvalue.ToString("F3"), "simulation vs exact; large p means they agree");
        output.Line($"Advantage lifts the hit chance from {Beat(attack, 20) * 100:F0}% to {Beat(advantage, 20) * 100:F0}% and cuts misses from " +
                    $"{(1 - Beat(attack, 20)) * 100:F0}% to {(1 - Beat(advantage, 20)) * 100:F0}%: against this armour class it is worth {(Beat(advantage, 20) > Beat(attack, 17) ? "more" : "less")} than a +3 bonus " +
                    $"({Beat(attack, 17) * 100:F0}%).");
    }
}
