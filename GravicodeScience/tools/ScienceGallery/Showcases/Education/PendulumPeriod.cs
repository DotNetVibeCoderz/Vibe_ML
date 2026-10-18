using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Education;

/// <summary>The exact pendulum period from an elliptic integral, against the small-angle formula.</summary>
public sealed class PendulumPeriod : Showcase
{
    public override string Title => "How far can a pendulum swing before T = 2π√(L/g) fails?";
    public override Domain Domain => Domain.Education;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "The exact period via the complete elliptic integral K, checked by simulating the swing.";
    public override string Scenario =>
        "Every physics student learns that a pendulum's period does not depend on how far it swings, and every lab " +
        "report finds that it does. The exact period is 4√(L/g)·K(sin²(θ₀/2)), an elliptic integral. Here it is " +
        "compared with the small-angle formula across all amplitudes and confirmed by integrating the full " +
        "equation of motion, so a teacher can tell students how large an angle their stopwatch experiment tolerates.";

    public override void Run(ShowcaseOutput output)
    {
        const double length = 1.0, g = 9.81;                         // a one-metre pendulum
        var smallAngle = 2 * Math.PI * Math.Sqrt(length / g);

        // SciPy's convention: EllipK takes the parameter m = k², here sin²(θ₀/2).
        double Exact(double amplitudeDeg) =>
            4 * Math.Sqrt(length / g) * Special.EllipK(Math.Pow(Math.Sin(amplitudeDeg * Math.PI / 360), 2));

        var amplitudes = NdArray.Linspace(1, 170, 300).ToArray();
        var ratio = amplitudes.Select(a => Exact(a) / smallAngle).ToArray();

        // Independent check: integrate θ'' = -(g/L) sin θ from rest and time the swing to the bottom.
        // From rest at the top, the first pass through θ = 0 takes exactly a quarter period.
        var simulatedAt = new[] { 10.0, 30, 60, 90, 120, 150, 165 };
        var simulated = simulatedAt.Select(a =>
        {
            var bottom = new OdeEvent((t, s) => s[0], terminal: 1, direction: -1);
            var run = Integrate.SolveIvp((t, s, ds) => { ds[0] = s[1]; ds[1] = -g / length * Math.Sin(s[0]); },
                0, 20, [a * Math.PI / 180, 0], OdeMethod.Dop853, events: [bottom], rtol: 1e-11, atol: 1e-12);
            return 4 * run.TEvents![0][0] / smallAngle;
        }).ToArray();

        // The largest amplitude at which the small-angle formula is still within 1% (and 5%).
        var onePercent = amplitudes[Array.FindIndex(ratio, r => r > 1.01) - 1];
        var fivePercent = amplitudes[Array.FindIndex(ratio, r => r > 1.05) - 1];

        var figure = GalleryCharts.Figure(8.5, 4.6);
        var ax = figure.AddSubplot();
        ax.Plot(amplitudes, ratio, "exact: elliptic integral");
        ax.Scatter(simulatedAt, simulated, "simulated swing (ODE)", sizes: [60], marker: MarkerStyle.Square);
        ax.AxHLine(1.0, Color.Parse("#5F6B7A"), LineStyle.Solid, 1.0, "small-angle formula");
        ax.AxVLine(onePercent, Color.Parse("#E8A317"), label: $"1% error at {onePercent:F0}°");
        ax.SetTitle($"Period of a {length:F0} m pendulum relative to 2π√(L/g) = {smallAngle:F3} s")
          .SetXLabel("release angle (degrees)").SetYLabel("true period / small-angle period");
        ax.Legend(LegendLocation.UpperLeft);
        output.Chart(figure);

        output.Metric("Small-angle period", $"{smallAngle:F3} s");
        output.Metric("Exact period at 90°", $"{Exact(90):F3} s", $"{(Exact(90) / smallAngle - 1) * 100:F1}% longer");
        output.Metric("Formula within 1% up to", $"{onePercent:F0}°", $"within 5% up to {fivePercent:F0}°");
        output.Metric("Worst ODE vs exact mismatch", $"{simulated.Zip(simulatedAt, (s, a) => Math.Abs(s - Exact(a) / smallAngle)).Max():E1}",
            "two independent routes to the same period");
        output.Line($"A class releasing the bob at {onePercent:F0}° or less can trust T = 2π√(L/g) to 1%; at 60° the " +
                    $"pendulum is already {(Exact(60) / smallAngle - 1) * 100:F1}% slow, larger than a stopwatch error over ten swings.");
    }
}
