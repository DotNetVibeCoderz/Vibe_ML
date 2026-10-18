using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviProb;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;
using Visual.Net.Core.Statistics;

namespace ScienceGallery.Showcases.Health;

/// <summary>A Bayesian reading of a two-arm trial: how likely is it that the new treatment is better, and by how much?</summary>
public sealed class ClinicalTrialBayesian : Showcase
{
    public override string Title => "How sure are we that the new treatment works better?";
    public override Domain Domain => Domain.Health;
    public override GraviLibrary[] Libraries => [GraviLibrary.GraviProb, GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Sample the posterior response rates of both arms with NUTS and read off the probability of benefit.";
    public override string Scenario =>
        "A phase II trial gives a new drug to 118 patients and standard care to 120. A p-value only says whether " +
        "the data would be surprising if the drugs were equal; the trial committee wants to know how probable it " +
        "is that the new drug is better, and how likely the benefit is to be big enough to matter clinically. A " +
        "Bayesian model answers both directly. The sampler is checked against the exact Beta posterior, which " +
        "this simple model happens to have.";

    public override void Run(ShowcaseOutput output)
    {
        const int controlN = 120, controlResponders = 42, treatedN = 118, treatedResponders = 57;
        const double meaningful = 0.05;            // the committee's minimum clinically important difference

        // Flat priors on both response rates; a binomial likelihood for each arm.
        var model = new BayesianModel()
            .AddDistribution("control", Distribution.Beta(1, 1))
            .AddDistribution("treated", Distribution.Beta(1, 1))
            .AddObservation("control arm", DistributionSpec.Binomial(controlN, "control"), controlResponders)
            .AddObservation("treated arm", DistributionSpec.Binomial(treatedN, "treated"), treatedResponders);
        var posterior = model.SampleNUTS(iterations: 1500, chains: 4, seed: 11);

        var control = posterior["control"].ToArray();
        var treated = posterior["treated"].ToArray();
        var difference = treated.Zip(control, (t, c) => t - c).ToArray();   // draws are paired, so this is the joint posterior
        var pBetter = difference.Average(d => d > 0 ? 1.0 : 0.0);
        var pMeaningful = difference.Average(d => d > meaningful ? 1.0 : 0.0);
        var hdi = posterior.HighestDensityInterval("treated", 0.95);

        // Independent check: with Beta(1,1) priors the posteriors are exactly Beta, and
        // P(treated > control) = ∫ f_treated(x) F_control(x) dx, by adaptive quadrature.
        var exactControl = Stats.Beta(1 + controlResponders, 1 + controlN - controlResponders);
        var exactTreated = Stats.Beta(1 + treatedResponders, 1 + treatedN - treatedResponders);
        var exactBetter = Integrate.Quad(x => exactTreated.Pdf(x) * exactControl.Cdf(x), 0, 1).Value;

        var figure = GalleryCharts.Figure(10.5, 4.6);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        var grid = NdArray.Linspace(0.15, 0.7, 300).ToArray();
        left.Hist(control, bins: 50, norm: PlotStats.HistNorm.Density, label: "standard care (NUTS draws)", alpha: 0.45);
        left.Hist(treated, bins: 50, norm: PlotStats.HistNorm.Density, label: "new drug (NUTS draws)", alpha: 0.45);
        left.Plot(grid, grid.Select(exactControl.Pdf).ToArray(), "exact Beta posteriors", Color.Parse(GalleryCharts.Ink), lineWidth: 1.2);
        left.Plot(grid, grid.Select(exactTreated.Pdf).ToArray(), color: Color.Parse(GalleryCharts.Ink), lineWidth: 1.2);
        left.SetTitle("Response rate, each arm").SetXLabel("probability of responding").SetYLabel("posterior density");
        left.Legend(LegendLocation.UpperRight);

        right.Hist(difference, bins: 60, norm: PlotStats.HistNorm.Density, label: "new drug − standard care", color: GalleryCharts.Of(GraviLibrary.GraviProb), alpha: 0.6);
        right.AxVLine(0, Color.Parse("#5F6B7A"), LineStyle.Solid, 1.0, "no difference");
        right.AxVLine(meaningful, Color.Parse("#D9363E"), LineStyle.Dashed, 1.2, $"clinically meaningful (+{meaningful * 100:F0} points)");
        right.SetTitle("Difference in response rate").SetXLabel("difference").SetYLabel("posterior density");
        right.Legend(LegendLocation.UpperLeft);
        output.Chart(figure);

        output.Metric("P(new drug is better)", $"{pBetter * 100:F1}%", $"exact: {exactBetter * 100:F1}%");
        output.Metric("P(benefit > 5 points)", $"{pMeaningful * 100:F1}%", "the committee's threshold");
        output.Metric("New drug response, 95% HDI", $"{hdi.Lower * 100:F0}%–{hdi.Upper * 100:F0}%");
        output.Metric("Convergence (R-hat)", $"{Math.Max(posterior.RHat("control"), posterior.RHat("treated")):F3}",
                      $"{posterior.EffectiveSampleSize("treated"):F0} effective draws of the treated rate");
        output.Line($"There is a {pBetter * 100:F0}% probability the new drug is better and a {pMeaningful * 100:F0}% probability its benefit is " +
                    $"clinically meaningful; {(pMeaningful > 0.8 ? "strong enough to move to phase III" : "promising, but not yet enough to justify a phase III trial on its own")}.");
    }
}
