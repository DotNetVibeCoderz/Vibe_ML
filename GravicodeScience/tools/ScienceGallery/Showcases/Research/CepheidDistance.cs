using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Research;

/// <summary>A Cepheid's period from irregular night-time observations, and from it the distance to the star.</summary>
public sealed class CepheidDistance : Showcase
{
    public override string Title => "How far away is this pulsating star?";
    public override Domain Domain => Domain.Research;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "A Lomb–Scargle periodogram finds a Cepheid's period in gappy data; Leavitt's law turns it into a distance.";
    public override string Scenario =>
        "Cepheid variable stars brighten and fade with a period that reveals their true luminosity, which is how " +
        "Henrietta Leavitt and Edwin Hubble measured the size of the universe. An observatory gets one or two " +
        "measurements on clear nights only, so the data are irregular with long gaps and an ordinary FFT cannot be " +
        "used. The Lomb–Scargle periodogram handles uneven sampling directly; the period then gives the absolute " +
        "magnitude, and comparing it with the observed brightness gives the distance.";

    public override void Run(ShowcaseOutput output)
    {
        const double truePeriod = 12.43, meanMagnitude = 14.20;   // days, apparent V magnitude

        // A Cepheid's light curve is a skewed wave: a quick rise to maximum, a slow decline.
        static double Shape(double phase) =>
            -0.38 * Math.Sin(2 * Math.PI * phase) - 0.14 * Math.Sin(4 * Math.PI * phase) - 0.06 * Math.Sin(6 * Math.PI * phase);

        // 200 nights of a season; about half are clear, and a measurement is taken at a random hour.
        var rng = new GraviRandom(19);
        var nights = Enumerable.Range(0, 200).Where(_ => rng.NextBernoulli(0.45)).ToArray();
        var time = nights.Select(n => n + rng.Uniform(-0.15, 0.15)).ToArray();
        var magnitude = time.Select(t => meanMagnitude + Shape(t / truePeriod) + rng.Normal(0, 0.03)).ToArray();

        // Search periods from 1.5 to 60 days. Lombscargle takes angular frequencies (radians per day).
        var periods = NdArray.Linspace(Math.Log(1.5), Math.Log(60), 6000).ToArray().Select(Math.Exp).ToArray();
        var power = Signal.Lombscargle(time, magnitude, periods.Select(p => 2 * Math.PI / p).ToArray(),
                                       floatingMean: true, normalize: LombscargleNormalization.Normalize);
        var best = periods[Array.IndexOf(power, power.Max())];
        // Refine on a fine grid around the peak: precision is limited by the span of the observations.
        var fine = NdArray.Linspace(best * 0.99, best * 1.01, 2001).ToArray();
        var finePower = Signal.Lombscargle(time, magnitude, fine.Select(p => 2 * Math.PI / p).ToArray(), floatingMean: true);
        var period = fine[Array.IndexOf(finePower, finePower.Max())];

        // Leavitt's period–luminosity law in the V band (Benedict et al. 2007), then the distance modulus
        // m − M = 5 log10(d / 10 pc). Interstellar dust is ignored here, so this is an upper bound.
        var absolute = -2.43 * (Math.Log10(period) - 1) - 4.05;
        var mean = magnitude.Average();
        var parsecs = Math.Pow(10, (mean - absolute) / 5 + 1);

        var figure = GalleryCharts.Figure(10, 4.6);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        left.Plot(periods, power, "Lomb–Scargle power", lineWidth: 1.2);
        left.AxVLine(period, Color.Parse("#D9363E"), label: $"P = {period:F2} d");
        left.SetXScale("log").SetXLim(1.5, 60).SetTitle("Periodogram of the irregular data").SetXLabel("trial period (days)").SetYLabel("normalised power");
        left.Legend(LegendLocation.UpperRight);

        var phase = time.Select(t => t / period % 1.0).ToArray();
        right.Scatter(phase.Concat(phase.Select(p => p + 1)).ToArray(), magnitude.Concat(magnitude).ToArray(),
                      $"{time.Length} nights, folded", sizes: [16]);
        right.InvertYAxis();                                       // astronomers' magnitudes: brighter is smaller
        right.SetTitle($"Light curve folded at {period:F2} days").SetXLabel("phase (two cycles)").SetYLabel("V magnitude");
        output.Chart(figure);

        output.Metric("Observations", $"{time.Length} over {time.Max() - time.Min():F0} days", "clear nights only");
        output.Metric("Period", $"{period:F3} days", $"true {truePeriod} days");
        output.Metric("Absolute magnitude", absolute.ToString("F2"), "from Leavitt's law");
        output.Metric("Distance", $"{parsecs / 1000:F1} kpc", $"{parsecs * 3.2616 / 1000:F0} thousand light years");
        output.Line($"The star looks {Math.Pow(10, (mean - absolute) / 2.5) / 1e6:F0} million times fainter than it would from 10 parsecs. " +
                    $"At {parsecs / 1000:F0} kpc it lies {(parsecs > 15_000 ? "outside" : "inside")} the Milky Way's disk (about 30 kpc across); " +
                    "for comparison, the Large Magellanic Cloud, where Leavitt found her Cepheids, is 50 kpc away.");
    }
}
