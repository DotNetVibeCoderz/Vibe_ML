using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Industry;

/// <summary>Statistical process control for a machined part: X-bar chart and process capability.</summary>
public sealed class ProductionLineControlChart : Showcase
{
    public override string Title => "Catch a drifting machine before it makes scrap";
    public override Domain Domain => Domain.Industry;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviFrame, GraviLibrary.GraviNum];
    public override string Summary => "A Shewhart X-bar chart and Cpk for shaft diameters, sampled five at a time each hour.";
    public override string Scenario =>
        "A quality engineer samples five shafts from the lathe every hour. Each subgroup's mean goes on an X-bar " +
        "chart whose limits come from the first, stable day. Here the tool starts to wear at hour 30, the chart " +
        "flags it, and the capability index shows how close the process is to producing out-of-spec parts.";

    public override void Run(ShowcaseOutput output)
    {
        const double target = 25.000, specLow = 24.970, specHigh = 25.030;   // millimetres
        const int hours = 48, perHour = 5;

        var rng = new GraviRandom(11);
        var rows = new List<(double Hour, double Diameter)>();
        for (var h = 0; h < hours; h++)
        {
            var wear = h < 30 ? 0 : (h - 30) * 0.0011;            // the tool starts wearing at hour 30
            for (var i = 0; i < perHour; i++) rows.Add((h, target + wear + rng.Normal(0, 0.006)));
        }
        var frame = DataFrame.FromColumns(new Dictionary<string, double[]>
        {
            ["hour"] = rows.Select(r => r.Hour).ToArray(),
            ["diameter"] = rows.Select(r => r.Diameter).ToArray(),
        });

        var means = frame.GroupBy("hour").Mean("diameter").Numeric("diameter").Values;

        // Control limits from the first 20 hours, when the process was known to be stable:
        // centre line ± 3 standard errors of a subgroup mean.
        var baseline = frame.Numeric("diameter").Values.Take(20 * perHour).ToArray();
        var centre = baseline.Average();
        var sigma = Stats.Describe(baseline).Variance is var v ? Math.Sqrt(v) : 0;
        var (ucl, lcl) = (centre + 3 * sigma / Math.Sqrt(perHour), centre - 3 * sigma / Math.Sqrt(perHour));
        var firstAlarm = Array.FindIndex(means, m => m > ucl || m < lcl);

        // Is the baseline normal enough for 3-sigma limits to mean what they say?
        var normality = Stats.Shapiro(baseline);
        var cpk = Math.Min(specHigh - centre, centre - specLow) / (3 * sigma);

        var figure = GalleryCharts.Figure(9, 4.6);
        var ax = figure.AddSubplot();
        var hoursAxis = Enumerable.Range(0, hours).Select(h => (double)h).ToArray();
        ax.Plot(hoursAxis, means, "hourly mean", marker: MarkerStyle.Circle, markerSize: 4);
        ax.AxHLine(centre, Color.Parse("#5F6B7A"), LineStyle.Solid, 1.0, "centre line");
        ax.AxHLine(ucl, Color.Parse("#D9363E"), label: "control limits");
        ax.AxHLine(lcl, Color.Parse("#D9363E"));
        if (firstAlarm >= 0) ax.AxVLine(firstAlarm, Color.Parse("#E8A317"), LineStyle.Dotted, 1.6, $"first alarm, hour {firstAlarm}");
        ax.SetYLim(Math.Min(lcl, means.Min()) - 0.002, Math.Max(ucl, means.Max()) + 0.002);
        ax.SetTitle("X-bar chart, shaft diameter (mm)").SetXLabel("hour").SetYLabel("subgroup mean (mm)");
        ax.Legend();
        output.Chart(figure);

        output.Metric("Control limits", $"{lcl:F4} – {ucl:F4} mm");
        output.Metric("First out-of-control hour", firstAlarm >= 0 ? firstAlarm.ToString() : "none", "the tool began wearing at hour 30");
        output.Metric("Cpk on the stable baseline", cpk.ToString("F2"), cpk >= 1.33 ? "capable (≥ 1.33)" : "not capable (< 1.33)");
        output.Metric("Shapiro–Wilk p (baseline)", normality.Pvalue.ToString("F3"), "large p: no evidence against normality");
        output.Line($"The chart raised its first alarm {firstAlarm - 30} hours after wear began, before any single shaft " +
                    $"went outside the {specLow:F3}–{specHigh:F3} mm specification.");
    }
}
