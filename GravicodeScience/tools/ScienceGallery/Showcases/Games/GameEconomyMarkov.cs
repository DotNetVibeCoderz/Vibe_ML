using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviProb;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Games;

/// <summary>A Markov model of player engagement: lifetime, lifetime value and its uncertainty.</summary>
public sealed class GameEconomyMarkov : Showcase
{
    public override string Title => "How long do players stay, and what is a new player worth?";
    public override Domain Domain => Domain.Games;
    public override GraviLibrary[] Libraries => [GraviLibrary.GraviProb, GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Monthly state transitions as an absorbing Markov chain, with Dirichlet uncertainty on every estimate.";
    public override string Scenario =>
        "A free-to-play studio tracks each player's state month to month: new, casual, engaged, paying, or gone. " +
        "Counting the moves between states gives a Markov chain in which \"gone\" is absorbing. The fundamental " +
        "matrix gives the expected months a new player stays and, with each state's monthly revenue, their " +
        "lifetime value; the chain's second eigenvalue gives how fast the active base decays. Each row of counts " +
        "is uncertain, so Dirichlet posteriors turn the estimates into ranges a finance team can plan with.";

    public override void Run(ShowcaseOutput output)
    {
        string[] states = ["new", "casual", "engaged", "paying", "gone"];
        double[] revenue = [0.05, 0.10, 0.60, 6.50, 0];                  // USD per player per month in each state
        // Last quarter's observed moves (row: this month, column: next month) across the player base.
        double[,] counts =
        {
            { 0, 2100, 950, 120, 4830 },     // new players: most leave in the first month
            { 0, 5200, 1300, 90, 2410 },
            { 0, 900, 3900, 520, 780 },
            { 0, 60, 380, 1350, 210 },
            { 0, 0, 0, 0, 1 },               // gone stays gone
        };
        const int transient = 4;

        // Point estimate: each row of counts, normalised.
        NdArray Transition(Func<int, double[]> row)
        {
            var t = NdArray.Zeros(5, 5);
            for (var i = 0; i < 5; i++) { var r = row(i); for (var j = 0; j < 5; j++) t[i, j] = r[j]; }
            return t;
        }
        double[] Counts(int i) => Enumerable.Range(0, 5).Select(j => counts[i, j]).ToArray();
        var estimate = Transition(i => Counts(i).Select(c => c / Counts(i).Sum()).ToArray());

        // Expected months spent in each state before leaving: the fundamental matrix N = (I − Q)⁻¹,
        // where Q is the transition matrix among the four active states.
        (double Months, double Value, double[] PerState) Lifetime(NdArray t)
        {
            var iMinusQ = NdArray.Zeros(transient, transient);
            for (var i = 0; i < transient; i++)
                for (var j = 0; j < transient; j++) iMinusQ[i, j] = (i == j ? 1 : 0) - t[i, j];
            var n = LinAlg.Inverse(iMinusQ);
            var perState = Enumerable.Range(0, transient).Select(j => n[0, j]).ToArray();
            return (perState.Sum(), perState.Select((m, j) => m * revenue[j]).Sum(), perState);
        }
        var (months, value, monthsIn) = Lifetime(estimate);

        // How quickly the active base decays: the largest eigenvalue of Q below 1 is the monthly survival
        // factor of a long-standing cohort.
        var q = NdArray.Zeros(transient, transient);
        for (var i = 0; i < transient; i++) for (var j = 0; j < transient; j++) q[i, j] = estimate[i, j];
        var decay = Linalg.Eig(q).Values.Max(v => v.Magnitude);
        var halfLife = Math.Log(0.5) / Math.Log(decay);

        // Uncertainty: each row's probabilities have a Dirichlet posterior (flat prior + observed counts).
        // Sample whole transition matrices and recompute the lifetime value for each.
        var rng = new GraviRandom(9);
        var posteriors = Enumerable.Range(0, transient).Select(i => Dirichlet.Uniform(5).Posterior(Counts(i))).ToArray();
        var values = Enumerable.Range(0, 2000).Select(_ =>
        {
            var draw = posteriors.Select(p => p.Sample(rng).ToArray()).ToArray();
            return Lifetime(Transition(i => i < transient ? draw[i] : [0, 0, 0, 0, 1])).Value;
        }).OrderBy(v => v).ToArray();
        var (low, high) = (values[50], values[1949]);                    // central 95%

        // A cohort of 10,000 new players, month by month: shares in each state.
        var horizon = 24;
        var share = new double[5][];
        for (var s = 0; s < 5; s++) share[s] = new double[horizon + 1];
        var current = NdArray.FromValues([1.0, 0, 0, 0, 0]);
        for (var m = 0; m <= horizon; m++)
        {
            for (var s = 0; s < 5; s++) share[s][m] = current.At(s) * 100;
            current = LinAlg.Dot(estimate.T, current);
        }

        var figure = GalleryCharts.Figure(11, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        var monthsAxis = Enumerable.Range(0, horizon + 1).Select(m => (double)m).ToArray();
        left.StackPlot(monthsAxis, share.Take(transient).ToArray(), states.Take(transient).ToArray());
        left.SetYLim(0, 100).SetXLim(0, horizon);
        left.SetTitle("A cohort of new players, month by month").SetXLabel("months since install").SetYLabel("% still active, by state");
        left.Legend(LegendLocation.UpperRight);

        right.Hist(values, bins: 40, label: "posterior draws", color: GalleryCharts.Of(GraviLibrary.GraviProb), alpha: 0.6);
        right.AxVLine(value, Color.Parse(GalleryCharts.Ink), LineStyle.Solid, 1.4, $"point estimate ${value:F2}");
        right.AxVSpan(low, high, Color.Parse("#E8A317"), 0.15, "95% credible range");
        right.SetTitle("Lifetime value of a new player").SetXLabel("USD").SetYLabel("draws");
        right.Legend(LegendLocation.UpperRight);
        output.Chart(figure);

        output.Metric("Expected active months", months.ToString("F2"), "for a newly installed player");
        output.Metric("Lifetime value", $"${value:F2}", $"95% range ${low:F2}–${high:F2}");
        output.Metric("Active-base half-life", $"{halfLife:F1} months", $"monthly survival factor {decay:F3} (largest eigenvalue of Q)");
        output.Metric("Still active after a year", $"{share.Take(transient).Sum(s => s[12]):F1}%");
        output.Line($"A new player spends {monthsIn[3] / months * 100:F0}% of their active months paying, yet those months bring " +
                    $"{monthsIn[3] * revenue[3] / value * 100:F0}% of their lifetime value. User acquisition pays off below about ${low:F2} per install, " +
                    "the bottom of the credible range.");
    }
}
