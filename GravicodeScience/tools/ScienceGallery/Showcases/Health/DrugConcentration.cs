using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Health;

/// <summary>Comparing dosing schedules with a two-compartment pharmacokinetic model and a therapeutic window.</summary>
public sealed class DrugConcentration : Showcase
{
    public override string Title => "Twice or three times a day: which schedule keeps the drug in range?";
    public override Domain Domain => Domain.Health;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Simulate oral doses through gut, blood and tissue compartments and score time in the therapeutic window.";
    public override string Scenario =>
        "A hospital pharmacist must choose between 750 mg every 12 hours and 500 mg every 8 hours of an " +
        "antibiotic: the same daily dose. Too low and the drug stops working; too high and side effects start. " +
        "A two-compartment model (blood exchanging with tissue, plus absorption from the gut) predicts the " +
        "plasma concentration over three days, and the schedules are compared by the share of time spent " +
        "inside the therapeutic window and by the peaks they reach.";

    public override void Run(ShowcaseOutput output)
    {
        // Population parameters (per hour): absorption ka, elimination ke, blood<->tissue exchange k12/k21,
        // central volume 30 L and 85% bioavailability. Window: effective above 6, toxic above 20 mg/L.
        const double ka = 1.1, ke = 0.25, k12 = 0.4, k21 = 0.3, volume = 30, bioavailability = 0.85;
        const double effective = 6, toxic = 20, hours = 72;

        // State: drug amount (mg) in the gut, the blood (central) and the tissue (peripheral).
        static void Kinetics(double t, ReadOnlySpan<double> a, Span<double> da)
        {
            da[0] = -ka * a[0];
            da[1] = ka * a[0] - (ke + k12) * a[1] + k21 * a[2];
            da[2] = k12 * a[1] - k21 * a[2];
        }

        // Integrate dose by dose: between doses the ODE runs freely; at each dose the gut receives the
        // absorbed part of the tablet.
        (double[] T, double[] C) Schedule(double dose, double interval)
        {
            var (times, conc) = (new List<double>(), new List<double>());
            double[] state = [0, 0, 0];
            for (var start = 0.0; start < hours - 1e-9; start += interval)
            {
                state[0] += bioavailability * dose;
                var end = Math.Min(start + interval, hours);
                var grid = NdArray.Linspace(start, end, (int)((end - start) * 20) + 1).ToArray();
                var leg = Integrate.SolveIvp(Kinetics, start, end, state, rtol: 1e-8, atol: 1e-8, tEval: grid);
                times.AddRange(grid);
                conc.AddRange(leg.Y.Row(1).ToArray().Select(a => a / volume));
                state = [leg.Y[0, grid.Length - 1], leg.Y[1, grid.Length - 1], leg.Y[2, grid.Length - 1]];
            }
            return (times.ToArray(), conc.ToArray());
        }

        var twice = Schedule(750, 12);
        var thrice = Schedule(500, 8);
        // Judge only the last 24 hours, once each schedule has reached its steady rhythm.
        double InWindow((double[] T, double[] C) s) => s.T.Where((t, i) => t >= hours - 24 && s.C[i] >= effective && s.C[i] <= toxic).Count()
                                                      / (double)s.T.Count(t => t >= hours - 24) * 100;
        double Trough((double[] T, double[] C) s) => s.C.Where((_, i) => s.T[i] >= hours - 24).Min();
        double Peak((double[] T, double[] C) s) => s.C.Where((_, i) => s.T[i] >= hours - 24).Max();
        // Total exposure over the last day: the area under the curve, by Simpson's rule.
        double Auc((double[] T, double[] C) s)
        {
            var last = Enumerable.Range(0, s.T.Length).Where(i => s.T[i] >= hours - 24).ToArray();
            var distinct = last.GroupBy(i => s.T[i]).Select(g => g.First()).ToArray();
            return Integrate.Simpson(distinct.Select(i => s.C[i]).ToArray(), distinct.Select(i => s.T[i]).ToArray());
        }

        var figure = GalleryCharts.Figure(10, 4.8);
        var ax = figure.AddSubplot();
        ax.AxHSpan(effective, toxic, Color.Parse("#2E9E68"), 0.12, "therapeutic window");
        ax.Plot(twice.T, twice.C, "750 mg every 12 h");
        ax.Plot(thrice.T, thrice.C, "500 mg every 8 h");
        ax.AxVSpan(hours - 24, hours, Color.Parse("#E4E8ED"), 0.5, "steady state, scored");
        ax.SetTitle("Plasma concentration over three days").SetXLabel("hours since first dose").SetYLabel("concentration (mg/L)");
        ax.Legend(LegendLocation.UpperLeft, columns: 2);
        ax.SetYLim(0, Math.Max(Peak(twice), Peak(thrice)) * 1.35);
        output.Chart(figure);

        output.Metric("Time in window, 12-hourly", $"{InWindow(twice):F0}%", $"trough {Trough(twice):F1}, peak {Peak(twice):F1} mg/L");
        output.Metric("Time in window, 8-hourly", $"{InWindow(thrice):F0}%", $"trough {Trough(thrice):F1}, peak {Peak(thrice):F1} mg/L");
        output.Metric("Daily exposure (AUC)", $"{Auc(twice):F0} vs {Auc(thrice):F0} mg·h/L", "same daily dose, so nearly the same AUC");
        var better = InWindow(thrice) > InWindow(twice) ? "500 mg every 8 hours" : "750 mg every 12 hours";
        output.Line($"Both schedules deliver the same exposure, but {better} spends more of the day in the window: " +
                    "smaller, more frequent doses flatten the peaks and lift the troughs. The cost is a third daily dose to remember.");
    }
}
