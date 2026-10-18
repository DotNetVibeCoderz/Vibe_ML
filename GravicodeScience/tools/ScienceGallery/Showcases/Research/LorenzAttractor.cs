using Gravicode.Science.SciNet;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Research;

/// <summary>Weather in three equations: integrating the Lorenz system and watching chaos appear.</summary>
public sealed class LorenzAttractor : Showcase
{
    public override string Title => "Chaos in a weather model";
    public override Domain Domain => Domain.Research;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Integrate Lorenz's convection equations and see two almost identical starts drift apart.";
    public override string Scenario =>
        "Edward Lorenz cut a weather model down to three equations and found that a rounding difference in the " +
        "starting point changed the forecast completely. Climate and physics courses use this system to show " +
        "sensitive dependence on initial conditions. Here an adaptive 8th-order solver integrates two runs that " +
        "start one millionth apart, and the chart shows where they part company.";

    public override void Run(ShowcaseOutput output)
    {
        // dx/dt = s(y - x), dy/dt = x(r - z) - y, dz/dt = xy - bz with Lorenz's classic constants.
        static void Lorenz(double t, ReadOnlySpan<double> u, Span<double> du)
        {
            du[0] = 10 * (u[1] - u[0]);
            du[1] = u[0] * (28 - u[2]) - u[1];
            du[2] = u[0] * u[1] - 8.0 / 3.0 * u[2];
        }

        var times = Enumerable.Range(0, 4001).Select(i => i * 0.01).ToArray();
        var a = Integrate.SolveIvp(Lorenz, 0, 40, [1.0, 1.0, 1.0], OdeMethod.Dop853, rtol: 1e-10, atol: 1e-12, tEval: times);
        var b = Integrate.SolveIvp(Lorenz, 0, 40, [1.000001, 1.0, 1.0], OdeMethod.Dop853, rtol: 1e-10, atol: 1e-12, tEval: times);

        // Distance between the two runs; the first time it exceeds 1 is when the "forecasts" disagree.
        var gap = times.Select((_, i) => Math.Sqrt(
            Math.Pow(a.Y[0, i] - b.Y[0, i], 2) + Math.Pow(a.Y[1, i] - b.Y[1, i], 2) + Math.Pow(a.Y[2, i] - b.Y[2, i], 2))).ToArray();
        var parted = times[Array.FindIndex(gap, d => d > 1.0)];

        var figure = GalleryCharts.Figure(9, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        left.Plot(a.Y.Row(0).ToArray(), a.Y.Row(2).ToArray(), "start (1, 1, 1)", lineWidth: 0.7);
        left.SetTitle("The attractor (x against z)").SetXLabel("x").SetYLabel("z");

        right.Plot(times, a.Y.Row(0).ToArray(), "start x = 1", lineWidth: 1.1);
        right.Plot(times, b.Y.Row(0).ToArray(), "start x = 1.000001", lineWidth: 1.1, style: LineStyle.Dashed);
        right.AxVLine(parted, Color.Parse(GalleryCharts.Ink), label: $"they part at t = {parted:F1}");
        right.SetTitle("Two forecasts, one millionth apart").SetXLabel("time").SetYLabel("x");
        right.Legend();
        output.Chart(figure);

        output.Metric("Solver steps", a.T.Length.ToString("N0"), "evaluations reported at 4,001 fixed times");
        output.Metric("Right-hand side calls", a.Nfev.ToString("N0"));
        output.Metric("Forecasts disagree after", $"t = {parted:F2}", "from a difference of 1e-6 in x");
        output.Line("Up to that point the two curves are indistinguishable; after it they are unrelated. " +
                    "No tolerance fixes this: the system itself amplifies every difference.");
    }
}
