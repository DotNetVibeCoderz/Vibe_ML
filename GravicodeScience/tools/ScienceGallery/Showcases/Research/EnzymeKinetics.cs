using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;

namespace ScienceGallery.Showcases.Research;

/// <summary>Michaelis–Menten parameters from replicate assays, by nonlinear fit and by the old double-reciprocal plot.</summary>
public sealed class EnzymeKinetics : Showcase
{
    public override string Title => "How fast is this enzyme, and how tightly does it bind?";
    public override Domain Domain => Domain.Research;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviFrame, GraviLibrary.GraviNum];
    public override string Summary => "Fit Vmax and Km with error bars, and see why the Lineweaver–Burk shortcut misleads.";
    public override string Scenario =>
        "A biochemistry lab measures the initial reaction rate of an enzyme at eight substrate concentrations, " +
        "three replicates each. The Michaelis–Menten curve v = Vmax·S/(Km + S) gives the maximum rate and the " +
        "substrate level at half that rate. Textbooks still linearise it as 1/v against 1/S, but that plot " +
        "magnifies the noise of the slowest, least reliable measurements; a direct nonlinear fit does not.";

    public override void Run(ShowcaseOutput output)
    {
        const double vmaxTrue = 42.0, kmTrue = 1.8;      // µmol/min/mg and mM
        double[] levels = [0.15, 0.3, 0.6, 1.0, 2.0, 4.0, 8.0, 16.0];   // substrate, mM

        // Three replicate assays per concentration, with 5% plate-reader noise plus a small floor.
        var rng = new GraviRandom(8);
        var rows = levels.SelectMany(s => Enumerable.Range(0, 3).Select(_ =>
            (S: s, V: vmaxTrue * s / (kmTrue + s) * (1 + rng.Normal(0, 0.05)) + rng.Normal(0, 0.4)))).ToArray();
        var assays = DataFrame.FromColumns(new Dictionary<string, double[]>
        {
            ["substrate_mM"] = rows.Select(r => r.S).ToArray(),
            ["rate"] = rows.Select(r => r.V).ToArray(),
        });

        // Nonlinear least squares on every replicate.
        var s = assays.Numeric("substrate_mM").Values;
        var v = assays.Numeric("rate").Values;
        var fit = Optimize.CurveFit((x, p) => p[0] * x / (p[1] + x), s, v, p0: [v.Max(), 1.0]);
        var (vmax, km) = (fit.Popt[0], fit.Popt[1]);

        // Lineweaver–Burk: 1/v = (Km/Vmax)(1/S) + 1/Vmax, a straight line through the replicate means.
        var means = assays.GroupBy("substrate_mM").Mean("rate");
        var sMean = means.Numeric("substrate_mM").Values;
        var vMean = means.Numeric("rate").Values;
        var spread = assays.GroupBy("substrate_mM").Std("rate").Numeric("rate").Values;
        var line = Stats.Linregress(sMean.Select(x => 1 / x).ToArray(), vMean.Select(x => 1 / x).ToArray());
        var vmaxLb = 1 / line.Intercept;
        var kmLb = line.Slope * vmaxLb;

        var curve = NdArray.Linspace(0, 17, 200).ToArray();
        var figure = GalleryCharts.Figure(10, 4.6);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        left.ErrorBar(sMean, vMean, spread, label: "mean ± sd of 3 assays", connect: false);
        left.Plot(curve, curve.Select(x => vmax * x / (km + x)).ToArray(), "Michaelis–Menten fit");
        left.AxHLine(vmax, label: $"Vmax = {vmax:F1}");
        left.SetTitle("Direct fit").SetXLabel("substrate (mM)").SetYLabel("rate (µmol/min/mg)");
        left.Legend(LegendLocation.LowerRight);

        var inverse = sMean.Select(x => 1 / x).ToArray();
        right.Scatter(inverse, vMean.Select(x => 1 / x).ToArray(), "1/v of the means", sizes: [40]);
        right.Plot([0, inverse.Max()], [line.Intercept, line.Intercept + line.Slope * inverse.Max()], "least-squares line", style: LineStyle.Dashed);
        right.SetTitle("Lineweaver–Burk (double reciprocal)").SetXLabel("1/S (1/mM)").SetYLabel("1/v");
        right.Legend(LegendLocation.UpperLeft);
        output.Chart(figure);

        output.Metric("Vmax (direct fit)", $"{vmax:F1} ± {fit.Perr[0]:F1} µmol/min/mg", $"true value {vmaxTrue}");
        output.Metric("Km (direct fit)", $"{km:F2} ± {fit.Perr[1]:F2} mM", $"true value {kmTrue}");
        output.Metric("Lineweaver–Burk estimate", $"Vmax {vmaxLb:F1}, Km {kmLb:F2}", "same data, linearised");
        output.Line($"The double-reciprocal plot puts the lowest concentration at 1/S = {inverse.Max():F1}, far from the others, " +
                    $"so its noise steers the line: Km comes out {Math.Abs(kmLb - kmTrue) / Math.Abs(km - kmTrue + 1e-12):F1}× further from the truth than the direct fit.");
    }
}
