using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Industry;

/// <summary>Finding the coolant temperature at which a batch reactor runs away, with an adaptive ODE solver.</summary>
public sealed class ReactorCooling : Showcase
{
    public override string Title => "How warm can the coolant get before the reactor runs away?";
    public override Domain Domain => Domain.Industry;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Simulate an exothermic batch reaction across coolant temperatures and find the runaway cliff.";
    public override string Scenario =>
        "A process-safety engineer must set the highest coolant temperature a jacketed batch reactor may run " +
        "with. The reaction releases heat, heat speeds the reaction up (Arrhenius), and above a critical point " +
        "the jacket cannot keep up: the temperature spikes within seconds after an hour of calm. An adaptive " +
        "solver takes long steps through the calm and tiny ones through the spike. Sweeping the coolant " +
        "temperature shows how sharp the cliff is and where the operating limit belongs.";

    public override void Run(ShowcaseOutput output)
    {
        // A first-order exothermic reaction A -> B in a cooled batch reactor (concentration normalised).
        const double kRef = 2e-4, tRef = 330, activation = 12_000;   // 1/s at 330 K; E/R in kelvin
        const double adiabaticRise = 180;                            // K, if no heat were removed
        const double cooling = 2.5e-3;                               // 1/s, jacket heat-transfer rate UA/(ρ cp V)
        const double limit = 420;                                    // K, product decomposes above this
        double Rate(double kelvin) => kRef * Math.Exp(activation * (1 / tRef - 1 / kelvin));

        OdeResult Batch(double coolant, double[]? times = null) => Integrate.SolveIvp((t, u, du) =>
        {
            var r = Rate(u[1]) * u[0];
            du[0] = -r;                                              // reactant used up
            du[1] = adiabaticRise * r - cooling * (u[1] - coolant);  // heat released minus heat removed
        }, 0, 4 * 3600, [1.0, coolant], OdeMethod.Rk45, rtol: 1e-7, atol: 1e-9, tEval: times);

        // Sweep the coolant temperature and record the hottest point of each batch.
        var coolants = NdArray.Linspace(315, 345, 61).ToArray();
        var peaks = coolants.Select(c => Batch(c).Y.Row(1).ToArray().Max()).ToArray();
        var cliff = coolants[Array.FindIndex(peaks, p => p > limit)];
        // The steepest point of the curve: where one degree of coolant changes the peak the most.
        var steepest = Enumerable.Range(1, coolants.Length - 1).MaxBy(i => peaks[i] - peaks[i - 1]);
        var jump = (peaks[steepest] - peaks[steepest - 1]) / (coolants[steepest] - coolants[steepest - 1]);

        // How the solver paced itself through the runaway batch: the step sizes it chose.
        var runaway = Batch(cliff);
        var steps = runaway.T.Zip(runaway.T.Skip(1), (a, b) => b - a).ToArray();

        var times = NdArray.Linspace(0, 4 * 3600, 800).ToArray();
        var figure = GalleryCharts.Figure(10.5, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        foreach (var c in new[] { cliff - 6, cliff - 3, cliff - 1, cliff })
        {
            var run = Batch(c, times);
            left.Plot(times.Select(t => t / 60).ToArray(), run.Y.Row(1).ToArray().Select(k => k - 273.15).ToArray(), $"coolant {c - 273.15:F1} °C");
        }
        left.AxHLine(limit - 273.15, Color.Parse("#D9363E"), label: "decomposition limit");
        left.SetTitle("Batch temperature").SetXLabel("time (minutes)").SetYLabel("temperature (°C)");
        left.Legend(LegendLocation.UpperRight);

        right.Plot(coolants.Select(c => c - 273.15).ToArray(), peaks.Select(p => p - 273.15).ToArray(), "hottest point of the batch", marker: MarkerStyle.Circle, markerSize: 4);
        right.AxHLine(limit - 273.15, Color.Parse("#D9363E"), label: "decomposition limit");
        right.AxVLine(cliff - 273.15, Color.Parse("#E8A317"), LineStyle.Dotted, 1.6, $"runaway from {cliff - 273.15:F1} °C");
        right.SetTitle("Peak temperature against coolant").SetXLabel("coolant temperature (°C)").SetYLabel("peak temperature (°C)");
        right.Legend(LegendLocation.UpperLeft);
        output.Chart(figure);

        output.Metric("Runaway coolant temperature", $"{cliff - 273.15:F1} °C", $"first batch to pass {limit - 273.15:F0} °C");
        output.Metric("Steepest sensitivity", $"{jump:F0} K per K", $"near {coolants[steepest] - 273.15:F1} °C coolant");
        output.Metric("Solver steps, runaway batch", $"{steps.Length}", $"from {steps.Min():F2} s in the spike to {steps.Max() / 60:F0} min in the calm");
        output.Line($"Near the cliff one degree of coolant moves the peak by up to {jump:F0} degrees, so a thermostat error of a degree is the " +
                    $"difference between a normal batch and a runaway. Set the operating limit well below {cliff - 273.15:F1} °C.");
    }
}
