using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Health;

/// <summary>Kaplan–Meier survival curves for two treatments, with censored follow-up and a log-rank test.</summary>
public sealed class SurvivalAfterTreatment : Showcase
{
    public override string Title => "Do patients stay relapse-free longer on the new therapy?";
    public override Domain Domain => Domain.Health;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Kaplan–Meier estimates handle patients still well at the end of follow-up; the log-rank test compares arms.";
    public override string Scenario =>
        "An oncology study follows 90 patients on each of two therapies for up to three years and records when " +
        "the disease returns. Many patients are still relapse-free when the study closes, or move away, so their " +
        "true relapse time is unknown: it is censored. Dropping them would make both therapies look worse; the " +
        "Kaplan–Meier estimator uses them for as long as they were observed. The log-rank test then asks whether " +
        "the two curves differ by more than chance.";

    public override void Run(ShowcaseOutput output)
    {
        // Relapse times follow Weibull laws (months): the new therapy stretches the time scale by about 40%.
        var rng = new GraviRandom(77);
        (double[] Time, bool[] Censored) Arm(double scaleMonths)
        {
            const int patients = 90;
            var time = new double[patients];
            var censored = new bool[patients];
            for (var i = 0; i < patients; i++)
            {
                var relapse = scaleMonths * Math.Pow(-Math.Log(1 - rng.NextDouble()), 1 / 1.3);
                var enrolled = rng.Uniform(0, 12);                     // patients joined over the first year
                var followUp = Math.Min(36 - enrolled, rng.NextBernoulli(0.1) ? rng.Uniform(3, 30) : 99);   // study end or lost
                time[i] = Math.Round(Math.Min(relapse, followUp), 1);
                censored[i] = relapse > followUp;
            }
            return (time, censored);
        }
        var standard = Arm(20);
        var therapy = Arm(28);

        var kmStandard = Stats.Ecdf(CensoredData.RightCensored(standard.Time, standard.Censored));
        var kmTherapy = Stats.Ecdf(CensoredData.RightCensored(therapy.Time, therapy.Censored));
        var test = Stats.Logrank(CensoredData.RightCensored(therapy.Time, therapy.Censored),
                                 CensoredData.RightCensored(standard.Time, standard.Censored));

        // Median relapse-free time: the first time the survival curve drops to one half.
        double Median(EmpiricalDistributionFunction sf)
        {
            var i = Array.FindIndex(sf.Probabilities, p => p <= 0.5);
            return i < 0 ? double.NaN : sf.Quantiles[i];
        }

        var figure = GalleryCharts.Figure(9.5, 4.8);
        var ax = figure.AddSubplot();
        foreach (var (km, name, color) in new[] { (kmStandard, "standard therapy", GalleryCharts.Cycle[0]), (kmTherapy, "new therapy", GalleryCharts.Cycle[1]) })
        {
            var x = new[] { 0.0 }.Concat(km.Sf.Quantiles).ToArray();
            var y = new[] { 1.0 }.Concat(km.Sf.Probabilities).ToArray();
            var (low, high) = km.Sf.ConfidenceInterval(0.95, "log-log");
            var band = Enumerable.Range(0, km.Sf.Quantiles.Length).Where(i => double.IsFinite(low.Probabilities[i]) && double.IsFinite(high.Probabilities[i])).ToArray();
            ax.FillBetween(band.Select(i => km.Sf.Quantiles[i]).ToArray(), band.Select(i => low.Probabilities[i]).ToArray(),
                           band.Select(i => high.Probabilities[i]).ToArray(), color: color, alpha: 0.12);
            ax.Step(x, y, StepWhere.Post, $"{name} (median {Median(km.Sf):F1} months)", color);
        }
        ax.AxHLine(0.5, Color.Parse("#8A94A3"), LineStyle.Dotted, 1.0);
        ax.SetYLim(0, 1.02).SetXLim(0, 36);
        ax.SetTitle("Relapse-free survival (Kaplan–Meier, 95% bands)").SetXLabel("months since treatment").SetYLabel("share still relapse-free");
        ax.Legend(LegendLocation.UpperRight);
        output.Chart(figure);

        output.Metric("Patients", $"{standard.Time.Length} + {therapy.Time.Length}",
                      $"{standard.Censored.Count(c => c) + therapy.Censored.Count(c => c)} censored (still well, or lost to follow-up)");
        output.Metric("Relapse-free at 2 years", $"{kmStandard.Sf.Evaluate(24) * 100:F0}% vs {kmTherapy.Sf.Evaluate(24) * 100:F0}%", "standard vs new");
        output.Metric("Log-rank p", test.Pvalue < 0.001 ? "< 0.001" : test.Pvalue.ToString("F3"), $"statistic {test.Statistic:F2}");
        output.Line(test.Pvalue < 0.05
            ? $"The curves separate early and stay apart; the log-rank test (p {(test.Pvalue < 0.001 ? "< 0.001" : $"= {test.Pvalue:F3}")}) says a gap this size is unlikely by chance."
            : $"The curves differ, but with p = {test.Pvalue:F2} the gap could still be chance; a larger study is needed.");
    }
}
