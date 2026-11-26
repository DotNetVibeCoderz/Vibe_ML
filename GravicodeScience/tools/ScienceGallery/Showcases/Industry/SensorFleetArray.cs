using Gravicode.Science.DaskNet;
using Gravicode.Science.GraviNum;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Industry;

/// <summary>A day of readings from a plant's pressure sensors as a chunked DaskArray: per-sensor statistics,
/// drift against the commissioning baseline, and spikes.</summary>
public sealed class SensorFleetArray : Showcase
{
    public override string Title => "Which of 48 pressure sensors are drifting or spiking?";
    public override Domain Domain => Domain.Industry;
    public override GraviLibrary[] Libraries => [GraviLibrary.DaskNet, GraviLibrary.GraviNum];
    public override string Summary => "4.1 million readings in 12 chunks: per-sensor statistics, broadcast against a baseline row, z-score spikes.";
    public override string Scenario =>
        "A process plant logs 48 pressure transmitters once a second. Maintenance wants two lists each morning: " +
        "transmitters whose average has walked away from the value measured at commissioning (calibration drift), " +
        "and transmitters that throw spikes (a loose connector or a failing diaphragm). The day's matrix is " +
        "split into chunks of two hours; column statistics are reduced chunk by chunk and tree-combined, and the " +
        "baseline row is broadcast against every chunk exactly as NumPy would broadcast it.";

    public override void Run(ShowcaseOutput output)
    {
        const int seconds = 86_400, sensors = 48, chunkRows = 7_200;    // one day at 1 Hz, two-hour chunks
        var rng = new GraviRandom(48);

        // Commissioning baseline (bar) and each transmitter's noise level.
        var baseline = Enumerable.Range(0, sensors).Select(_ => Math.Round(rng.Uniform(4, 12), 2)).ToArray();
        var noise = Enumerable.Range(0, sensors).Select(_ => rng.Uniform(0.02, 0.06)).ToArray();
        int[] drifting = [7, 30, 41];                                    // calibration has walked off
        int[] spiking = [12, 30];                                        // loose connector, failing diaphragm
        var data = new double[seconds * sensors];
        for (var t = 0; t < seconds; t++)
            for (var s = 0; s < sensors; s++)
            {
                var value = baseline[s] + noise[s] * rng.Normal();
                if (Array.IndexOf(drifting, s) >= 0) value += 0.18 * (s == 41 ? -1 : 1) * t / seconds;   // ramps over the day
                if (Array.IndexOf(spiking, s) >= 0 && rng.NextBernoulli(0.0004)) value += rng.Uniform(0.4, 0.9);
                data[t * sensors + s] = value;
            }
        var readings = new NdArray(data, seconds, sensors);

        // The chunked array: 12 blocks of 7,200 rows × 48 sensors. Every operation below builds a graph.
        var x = DaskArray.FromArray(readings, chunkRows, sensors);
        var commissioning = DaskArray.FromArray(new NdArray(baseline, sensors), sensors);
        var mean = x.Mean(axis: 0);
        var std = x.Std(axis: 0, ddof: 1);
        var drift = mean - commissioning;                                // [48] - [48]
        var z = (x - mean) / std;                                        // [86400, 48] against [48]: broadcast per row
        var spikes = z.Map(v => v > 6 ? 1.0 : 0.0).Sum(axis: 0);         // per sensor, z above 6
        var worstPerSecond = z.Abs().Max(axis: 1);                       // across the fleet, per second

        var driftValues = drift.Compute().ToArray();                     // thread-pool scheduler, block by block
        var spikeCounts = spikes.Compute().ToArray();
        var stdValues = std.Compute().ToArray();
        var fleetMax = worstPerSecond.Compute().ToArray();

        // The same column means on the whole array in GraviNum: the chunked answer must agree.
        var direct = readings.Mean(0).ToArray();
        var agreement = driftValues.Select((d, s) => Math.Abs(d + baseline[s] - direct[s])).Max();

        var figure = GalleryCharts.Figure(11, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        var sx = Enumerable.Range(0, sensors).Select(s => (double)s).ToArray();
        var limit = 0.05;
        left.Bar(sx, driftValues, 0.8,
                 colors: driftValues.Select(d => Math.Abs(d) > limit ? Color.Parse("#D9363E") : GalleryCharts.Of(GraviLibrary.DaskNet)).ToArray());
        left.AxHLine(limit, Color.Parse("#8A94A3"), LineStyle.Dotted, 1.0, $"±{limit} bar recalibration limit");
        left.AxHLine(-limit, Color.Parse("#8A94A3"), LineStyle.Dotted, 1.0);
        left.SetTitle("Day mean minus commissioning baseline").SetXLabel("sensor").SetYLabel("bar");
        left.Legend(LegendLocation.LowerLeft);

        // Fleet-wide worst |z| per minute, so 86,400 seconds become 1,440 points.
        var minutes = Enumerable.Range(0, seconds / 60).Select(m => fleetMax.Skip(m * 60).Take(60).Max()).ToArray();
        right.Plot(Enumerable.Range(0, minutes.Length).Select(m => m / 60.0).ToArray(), minutes, "largest |z| in the fleet",
                   GalleryCharts.Of(GraviLibrary.GraviNum), lineWidth: 1.0);
        right.AxHLine(6, Color.Parse("#D9363E"), LineStyle.Dashed, 1.2, "spike threshold z = 6");
        right.SetTitle("Worst reading across all sensors, per minute").SetXLabel("hour of day").SetYLabel("|z|");
        right.Legend(LegendLocation.UpperRight);
        output.Chart(figure);

        var flaggedDrift = Enumerable.Range(0, sensors).Where(s => Math.Abs(driftValues[s]) > limit).ToArray();
        var flaggedSpikes = Enumerable.Range(0, sensors).Where(s => spikeCounts[s] >= 5).ToArray();
        output.Metric("Readings", $"{(long)seconds * sensors:N0}", $"{x.BlockCount} chunks of {chunkRows:N0} × {sensors}");
        output.Metric("Drifting sensors", string.Join(", ", flaggedDrift.Select(s => $"#{s}")), $"mean moved more than {limit} bar");
        output.Metric("Spiking sensors", string.Join(", ", flaggedSpikes.Select(s => $"#{s}")),
                      string.Join(", ", flaggedSpikes.Select(s => $"#{s}: {spikeCounts[s]:0} spikes")));
        output.Metric("Agreement with GraviNum", $"{agreement:E1}", "column means, chunked vs whole array");
        output.Line($"Sensor #30 is on both lists: its spikes inflate its standard deviation to {stdValues[30]:F3} bar against a fleet median of " +
                    $"{stdValues.Order().ElementAt(sensors / 2):F3}, so a drift check based on σ alone would understate it. " +
                    "Drift is a slow ramp, invisible per second and obvious in the day mean.");
    }
}
