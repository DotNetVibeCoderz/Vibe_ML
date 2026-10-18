using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Health;

/// <summary>Cleaning a noisy ECG and detecting heartbeats to get heart rate and heart-rate variability.</summary>
public sealed class HeartRateFromEcg : Showcase
{
    public override string Title => "Heart rate and variability from a noisy ECG";
    public override Domain Domain => Domain.Health;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Band-pass filter a one-minute ECG, find the R peaks, and compute BPM, SDNN and RMSSD.";
    public override string Scenario =>
        "A wearable chest strap records a one-minute ECG while the wearer breathes slowly. The trace carries " +
        "baseline wander from breathing, 50 Hz mains hum and muscle noise. A zero-phase Butterworth band-pass " +
        "removes them without shifting the beats; peak detection with a refractory distance finds each R wave, " +
        "and the intervals between beats give heart rate and the variability measures clinicians use as a " +
        "marker of recovery and stress.";

    public override void Run(ShowcaseOutput output)
    {
        const double fs = 250;                                       // Hz, typical for wearables
        const double seconds = 60;
        var n = (int)(fs * seconds);
        var t = Enumerable.Range(0, n).Select(i => i / fs).ToArray();

        // Beat times: 68 bpm on average, speeding up on each in-breath (respiratory sinus arrhythmia, 0.2 Hz).
        var rng = new GraviRandom(30);
        var beats = new List<double>();
        for (var b = 0.4; b < seconds; b += 60.0 / (68 + 6 * Math.Sin(2 * Math.PI * 0.2 * b) + rng.Normal(0, 1.2))) beats.Add(b);

        // Each beat is a sum of Gaussians: P wave, Q, the tall R, S and the T wave (millivolts).
        (double Offset, double Width, double Height)[] waves = [(-0.2, 0.025, 0.12), (-0.03, 0.01, -0.12), (0, 0.011, 1.1), (0.03, 0.012, -0.25), (0.25, 0.045, 0.3)];
        var clean = new double[n];
        foreach (var beat in beats)
            for (var i = Math.Max(0, (int)((beat - 0.35) * fs)); i < Math.Min(n, (int)((beat + 0.5) * fs)); i++)
                foreach (var w in waves)
                    clean[i] += w.Height * Math.Exp(-Math.Pow(t[i] - beat - w.Offset, 2) / (2 * w.Width * w.Width));
        var raw = clean.Select((v, i) => v + 0.35 * Math.Sin(2 * Math.PI * 0.2 * t[i])     // breathing wander
                                        + 0.15 * Math.Sin(2 * Math.PI * 50 * t[i])          // mains hum
                                        + rng.Normal(0, 0.06)).ToArray();                    // muscle noise

        // 0.5–40 Hz band-pass, run forwards and backwards so the R peaks are not delayed.
        var bandPass = Signal.ButterSos(4, new[] { 0.5, 40.0 }, FilterType.Bandpass, fs: fs);
        var ecg = Signal.Sosfiltfilt(bandPass, raw);

        // R peaks: tall, prominent, and at least 0.3 s apart (no heart beats faster than 200 bpm).
        var found = Signal.FindPeaks(ecg, prominence: 0.6, distance: 0.3 * fs);
        var rTimes = found.Peaks.Select(i => t[i]).ToArray();
        var rr = rTimes.Zip(rTimes.Skip(1), (a, b) => (b - a) * 1000).ToArray();             // ms
        var bpm = 60_000 / rr.Average();
        var sdnn = Math.Sqrt(rr.Select(v => Math.Pow(v - rr.Average(), 2)).Sum() / (rr.Length - 1));
        var rmssd = Math.Sqrt(rr.Zip(rr.Skip(1), (a, b) => (b - a) * (b - a)).Average());
        var matched = beats.Count(b => rTimes.Any(r => Math.Abs(r - b) < 0.05));

        var figure = GalleryCharts.Figure(10, 5.8);
        var (top, bottom) = (figure.AddSubplot(2, 1, 1), figure.AddSubplot(2, 1, 2));
        var window = Enumerable.Range(0, n).Where(i => t[i] >= 10 && t[i] < 16).ToArray();
        top.Plot(window.Select(i => t[i]).ToArray(), window.Select(i => raw[i] + 1.2).ToArray(), "raw (offset +1.2 mV)", Color.Parse("#8A94A3"), lineWidth: 0.9);
        top.Plot(window.Select(i => t[i]).ToArray(), window.Select(i => ecg[i]).ToArray(), "filtered", lineWidth: 1.2);
        var shownPeaks = found.Peaks.Where(i => t[i] >= 10 && t[i] < 16).ToArray();
        top.Scatter(shownPeaks.Select(i => t[i]).ToArray(), shownPeaks.Select(i => ecg[i]).ToArray(), "R peak", Color.Parse("#D9363E"), sizes: [50]);
        top.SetTitle("Six seconds of the recording").SetXLabel("time (s)").SetYLabel("mV");
        top.Legend(LegendLocation.OutsideRight);

        bottom.Plot(rTimes.Skip(1).ToArray(), rr.Select(v => 60_000 / v).ToArray(), "beat-to-beat heart rate", marker: MarkerStyle.Circle, markerSize: 3);
        bottom.AxHLine(bpm, Color.Parse("#5F6B7A"), label: $"mean {bpm:F0} bpm");
        bottom.SetTitle("Tachogram: heart rate rises with each breath in").SetXLabel("time (s)").SetYLabel("bpm");
        bottom.Legend(LegendLocation.OutsideRight);
        output.Chart(figure);

        output.Metric("Heart rate", $"{bpm:F1} bpm", $"{rTimes.Length} beats in {seconds:F0} s");
        output.Metric("SDNN", $"{sdnn:F0} ms", "overall variability of the beat intervals");
        output.Metric("RMSSD", $"{rmssd:F0} ms", "beat-to-beat variability, vagal tone");
        output.Metric("Beats detected", $"{matched} of {beats.Count}", "within 50 ms of the true beat");
        output.Line($"The heart rate swings by about {rr.Select(v => 60_000 / v).Max() - rr.Select(v => 60_000 / v).Min():F0} bpm with breathing; " +
                    "an RMSSD in the tens of milliseconds is typical of a rested adult.");
    }
}
