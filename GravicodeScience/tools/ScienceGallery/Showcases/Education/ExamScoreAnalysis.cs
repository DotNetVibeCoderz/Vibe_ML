using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using ScienceGallery.Models;
using ScienceGallery.Services;

namespace ScienceGallery.Showcases.Education;

/// <summary>Did the new teaching method help? Comparing two classes' exam scores properly.</summary>
public sealed class ExamScoreAnalysis : Showcase
{
    public override string Title => "Did the new teaching method raise exam scores?";
    public override Domain Domain => Domain.Education;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Compare two classes with a Welch t-test, a rank test and an effect size, not just two averages.";
    public override string Scenario =>
        "A school tried project-based learning in one class and kept the usual lessons in another. Averages alone " +
        "cannot say whether a five-point gap is real or luck. The analysis checks the score distributions, runs a " +
        "Welch t-test (no equal-variance assumption) and a Mann–Whitney test (no normality assumption), and reports " +
        "Cohen's d so the head of department sees how big the effect is, not only whether it exists.";

    public override void Run(ShowcaseOutput output)
    {
        var rng = new GraviRandom(2024);
        double[] Scores(int n, double mean, double sd) =>
            Enumerable.Range(0, n).Select(_ => Math.Clamp(Math.Round(rng.Normal(mean, sd)), 0, 100)).ToArray();

        var usual = Scores(34, 68, 11);
        var project = Scores(31, 74, 9);

        var welch = Stats.TtestInd(project, usual, equalVar: false);
        var ranks = Stats.MannWhitneyU(project, usual);
        var (du, dp) = (Stats.Describe(usual), Stats.Describe(project));
        var pooled = Math.Sqrt(((usual.Length - 1) * du.Variance + (project.Length - 1) * dp.Variance) / (usual.Length + project.Length - 2));
        var cohensD = (dp.Mean - du.Mean) / pooled;

        // Smooth score distributions for the chart: a Gaussian kernel density per class.
        var grid = Enumerable.Range(0, 201).Select(i => 30 + i * 0.35).ToArray();
        var kdeUsual = new GaussianKde(usual).Evaluate(grid);
        var kdeProject = new GaussianKde(project).Evaluate(grid);

        var figure = GalleryCharts.Figure(8.5, 4.6);
        var ax = figure.AddSubplot();
        ax.FillBetween(grid, kdeUsual, label: "usual lessons", alpha: 0.25);
        ax.FillBetween(grid, kdeProject, label: "project-based", alpha: 0.25);
        ax.Plot(grid, kdeUsual);
        ax.Plot(grid, kdeProject);
        ax.AxVLine(du.Mean, label: $"means {du.Mean:F1} and {dp.Mean:F1}");
        ax.AxVLine(dp.Mean);
        ax.SetTitle("Exam score distributions").SetXLabel("score").SetYLabel("density");
        ax.Legend();
        output.Chart(figure);

        output.Metric("Mean difference", $"{dp.Mean - du.Mean:+0.0;-0.0} points");
        output.Metric("Welch t-test p", welch.Pvalue.ToString("F4"), $"t = {welch.Statistic:F2}");
        output.Metric("Mann–Whitney p", ranks.Pvalue.ToString("F4"), "does not assume normal scores");
        output.Metric("Cohen's d", cohensD.ToString("F2"), cohensD switch { < 0.2 => "negligible", < 0.5 => "small", < 0.8 => "medium", _ => "large" });
        output.Line(welch.Pvalue < 0.05
            ? "Both tests agree the gap is unlikely to be chance, and the effect size says it is worth acting on."
            : "The gap could plausibly be chance; collect another term of data before changing the curriculum.");
    }
}
