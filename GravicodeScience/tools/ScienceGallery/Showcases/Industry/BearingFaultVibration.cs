using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Industry;

/// <summary>Envelope analysis of a pump motor's vibration to find which part of a bearing is failing.</summary>
public sealed class BearingFaultVibration : Showcase
{
    public override string Title => "Which bearing race is cracking in this motor?";
    public override Domain Domain => Domain.Industry;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Band-pass, Hilbert envelope and Welch spectrum reveal a bearing's defect frequency.";
    public override string Scenario =>
        "A maintenance engineer records one second of vibration from a pump motor running at 1,797 rpm. A crack " +
        "in a bearing race sends a tiny impact every time a ball rolls over it, too weak to see in the ordinary " +
        "spectrum, but each impact rings the housing's resonance. Band-pass filtering around that resonance and " +
        "taking the envelope recovers the impact rate, and comparing it with the bearing's defect frequencies " +
        "says which part to replace at the next planned stop.";

    public override void Run(ShowcaseOutput output)
    {
        const double fs = 20_000;                          // Hz, accelerometer sampling rate
        const double shaft = 1797 / 60.0;                  // Hz, shaft rotation
        // Defect frequencies of a 6205 deep-groove bearing, as multiples of shaft speed (from its geometry).
        var outerRace = 3.585 * shaft;                     // BPFO
        var innerRace = 5.415 * shaft;                     // BPFI
        var ballSpin = 2.357 * shaft;                      // BSF

        // The signal: shaft imbalance and its harmonics, an outer-race crack whose impacts ring a 3.4 kHz
        // housing resonance, and broadband noise much larger than the impacts.
        var rng = new GraviRandom(4);
        var n = (int)fs;
        var t = Enumerable.Range(0, n).Select(i => i / fs).ToArray();
        var x = new double[n];
        for (var i = 0; i < n; i++)
            x[i] = 0.8 * Math.Sin(2 * Math.PI * shaft * t[i]) + 0.3 * Math.Sin(4 * Math.PI * shaft * t[i]) + rng.Normal(0, 0.6);
        for (var impact = 0.004; impact < 1.0; impact += 1 / outerRace * (1 + rng.Normal(0, 0.01)))   // slight slip
            for (int i = (int)(impact * fs), k = 0; i < n && k < 120; i++, k++)
                x[i] += 0.9 * Math.Exp(-k / 25.0) * Math.Sin(2 * Math.PI * 3400 * k / fs);

        // The plain spectrum: shaft lines dominate; the bearing hides in the noise floor.
        var (f, raw) = Signal.Welch(x, fs: fs, nperseg: 4096);

        // Envelope analysis: keep only the resonance band, then the magnitude of the analytic signal
        // traces the impacts' outline, and its spectrum shows how often they happen.
        var band = Signal.ButterSos(6, new[] { 2800.0, 4000.0 }, FilterType.Bandpass, fs: fs);
        var filtered = Signal.Sosfiltfilt(band, x);
        var envelope = Signal.Hilbert(filtered).Select(c => c.Magnitude).ToArray();
        var mean = envelope.Average();
        var (fe, env) = Signal.Welch(envelope.Select(v => v - mean).ToArray(), fs: fs, nperseg: 8192);

        // The strongest envelope line below 500 Hz, and the defect frequency it matches.
        var low = Enumerable.Range(0, fe.Length).Where(i => fe[i] > 20 && fe[i] < 500).ToArray();
        var peak = fe[low.MaxBy(i => env[i])];
        (string Part, double Hz)[] candidates = [("outer race", outerRace), ("inner race", innerRace), ("rolling element", ballSpin), ("shaft", shaft)];
        var match = candidates.MinBy(c => Math.Abs(c.Hz - peak));

        var figure = GalleryCharts.Figure(10.5, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        left.Plot(f, raw.Select(p => 10 * Math.Log10(p)).ToArray(), "vibration PSD", lineWidth: 1.0);
        left.AxVSpan(2800, 4000, Color.Parse("#E8A317"), 0.2, "band kept for the envelope");
        left.SetXLim(0, 6000).SetTitle("Raw spectrum").SetXLabel("frequency (Hz)").SetYLabel("power (dB/Hz)");
        left.Legend(LegendLocation.UpperRight);

        var shown = Enumerable.Range(0, fe.Length).Where(i => fe[i] <= 600).ToArray();
        right.Plot(shown.Select(i => fe[i]).ToArray(), shown.Select(i => env[i] * 1e3).ToArray(), "envelope spectrum", lineWidth: 1.2);
        right.AxVLine(outerRace, Color.Parse("#D9363E"), LineStyle.Dashed, 1.2, $"outer race {outerRace:F1} Hz");
        right.AxVLine(2 * outerRace, Color.Parse("#D9363E"), LineStyle.Dotted, 1.0);
        right.AxVLine(innerRace, Color.Parse("#4F46D6"), LineStyle.Dashed, 1.2, $"inner race {innerRace:F1} Hz");
        right.AxVLine(ballSpin, Color.Parse("#2E9E68"), LineStyle.Dashed, 1.2, $"ball spin {ballSpin:F1} Hz");
        right.SetXLim(0, 600).SetTitle("Envelope spectrum").SetXLabel("frequency (Hz)").SetYLabel("power (×10⁻³)");
        right.Legend(LegendLocation.UpperRight);
        output.Chart(figure);

        output.Metric("Shaft speed", $"{shaft:F2} Hz", "1,797 rpm");
        output.Metric("Envelope peak", $"{peak:F1} Hz", $"{peak / shaft:F2} × shaft speed");
        output.Metric("Diagnosis", match.Part, $"defect frequency {match.Hz:F1} Hz");
        output.Line($"The impacts repeat at {peak:F1} Hz, within {Math.Abs(peak - match.Hz):F1} Hz of the {match.Part} defect frequency " +
                    $"(the spectrum resolves {fe[1] - fe[0]:F1} Hz), " +
                    "with its harmonic visible too. Order a replacement bearing now and fit it at the next planned stop.");
    }
}
