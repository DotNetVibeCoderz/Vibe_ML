using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Research;

/// <summary>Finding and fitting the bands of a Raman spectrum to measure a mineral mixture.</summary>
public sealed class SpectrumPeakFitting : Showcase
{
    public override string Title => "How much calcite is in this rock sample?";
    public override Domain Domain => Domain.Research;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Detect Raman bands with find_peaks, then fit Lorentzian line shapes on a sloping baseline.";
    public override string Scenario =>
        "A geology lab records the Raman spectrum of a powdered rock that should be quartz and calcite. Each mineral " +
        "has its own bands: quartz at 464 cm⁻¹, calcite at 282, 712 and 1086 cm⁻¹. Peak finding locates them " +
        "automatically; a least-squares fit of Lorentzian band shapes then gives each band's exact position, width " +
        "and area, and the area ratio of the strongest bands tracks the mixture's composition.";

    public override void Run(ShowcaseOutput output)
    {
        // A Lorentzian band: height h, centre c, full width at half maximum w.
        static double Band(double x, double h, double c, double w) => h / (1 + Math.Pow(2 * (x - c) / w, 2));

        // The measured spectrum: four bands on a fluorescence background that slopes upward, plus
        // detector noise. Wavenumbers in cm⁻¹, intensities in counts.
        var shift = NdArray.Linspace(150, 1200, 1400).ToArray();
        (double H, double C, double W)[] truth = [(420, 282, 14), (1650, 464, 9), (260, 712, 10), (1250, 1086, 7)];
        var rng = new GraviRandom(5);
        var counts = shift.Select(x => 300 + 0.25 * x + truth.Sum(b => Band(x, b.H, b.C, b.W)) + rng.Normal(0, 18)).ToArray();

        // Peaks must stand well clear of the noise (prominence) and be a few samples wide.
        var found = Signal.FindPeaks(counts, prominence: 120.0, width: 3.0);
        var guesses = found.Peaks;

        // One model for the whole spectrum: a straight baseline plus one Lorentzian per detected peak.
        // Parameters: [offset, slope, h1, c1, w1, h2, c2, w2, ...].
        double Model(double x, double[] p)
        {
            var y = p[0] + p[1] * x;
            for (var k = 2; k < p.Length; k += 3) y += Band(x, p[k], p[k + 1], p[k + 2]);
            return y;
        }
        var start = new List<double> { counts.Min(), 0 };
        for (var i = 0; i < guesses.Length; i++)
            start.AddRange([found.Properties.Prominences![i], shift[guesses[i]], found.Properties.Widths![i] * (shift[1] - shift[0])]);
        var fit = Optimize.CurveFit(Model, shift, counts, p0: start.ToArray());

        var bands = Enumerable.Range(0, guesses.Length).Select(i => (
            H: fit.Popt[2 + 3 * i], C: fit.Popt[3 + 3 * i], W: Math.Abs(fit.Popt[4 + 3 * i]), CErr: fit.Perr[3 + 3 * i])).ToArray();
        // The area of a Lorentzian is (π/2)·h·w.
        double Area((double H, double C, double W, double CErr) b) => Math.PI / 2 * b.H * b.W;
        var quartz = bands.MinBy(b => Math.Abs(b.C - 464));
        var calcite = bands.MinBy(b => Math.Abs(b.C - 1086));
        var residual = counts.Select((y, i) => y - Model(shift[i], fit.Popt)).ToArray();

        var figure = GalleryCharts.Figure(9.5, 5.6);
        var (top, bottom) = (figure.AddSubplot(2, 1, 1), figure.AddSubplot(2, 1, 2));
        top.Plot(shift, counts, "measured", Color.Parse("#8A94A3"), lineWidth: 1.0);
        top.Plot(shift, shift.Select(x => Model(x, fit.Popt)).ToArray(), "fitted model", Color.Parse("#D9363E"), lineWidth: 1.6);
        top.Plot(shift, shift.Select(x => fit.Popt[0] + fit.Popt[1] * x).ToArray(), "baseline", style: LineStyle.Dashed, lineWidth: 1.2);
        foreach (var b in bands)
            top.Text(b.C, Model(b.C, fit.Popt) + 60, $"{b.C:F1}", fontSize: 11, align: TextAlign.Center);
        top.SetTitle("Raman spectrum of a quartz–calcite powder").SetYLabel("counts");
        top.Legend(LegendLocation.UpperRight);
        bottom.Plot(shift, residual, "measured − model", lineWidth: 0.9);
        bottom.AxHLine(0, Color.Parse("#5F6B7A"), LineStyle.Solid, 1.0);
        bottom.SetXLabel("Raman shift (cm⁻¹)").SetYLabel("residual");
        output.Chart(figure);

        output.Metric("Bands found", guesses.Length.ToString(), "find_peaks with prominence ≥ 120 counts");
        output.Metric("Quartz band", $"{quartz.C:F2} ± {quartz.CErr:F2} cm⁻¹", $"FWHM {quartz.W:F1} cm⁻¹");
        output.Metric("Calcite main band", $"{calcite.C:F2} ± {calcite.CErr:F2} cm⁻¹", $"FWHM {calcite.W:F1} cm⁻¹");
        output.Metric("Calcite / quartz band area", (Area(calcite) / Area(quartz)).ToString("F2"),
            $"true ratio {truth[3].H * truth[3].W / (truth[1].H * truth[1].W):F2}");
        output.Line($"Residuals scatter around zero with sd {Math.Sqrt(residual.Select(r => r * r).Average()):F0} counts, the detector noise, " +
                    "so the four-band model explains everything in the spectrum; a missed band would show up as a bump.");
    }
}
