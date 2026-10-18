using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Gravicode.Science.SciNet.IO;
using Matplotlib.Net.Artists;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Utility;

/// <summary>Removing mains hum and a whistle from a voice-note recording, file in and file out.</summary>
public sealed class AudioNoiseFilter : Showcase
{
    public override string Title => "Clean the hum and whistle out of a recording";
    public override Domain Domain => Domain.Utility;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Read a WAV file, notch out a 1 kHz whistle, high-pass away 50 Hz hum, and write the result.";
    public override string Scenario =>
        "A podcaster's recording picked up 50 Hz mains hum from a cheap cable and a 1 kHz whistle from a " +
        "monitor. Both are narrow and steady, so they show as horizontal lines in a spectrogram, while the " +
        "melody moves. A narrow band-stop filter and a high-pass filter remove them with almost no effect on the " +
        "music in between. The audio goes through real WAV files, as it would in an editing tool.";

    public override void Run(ShowcaseOutput output)
    {
        const int rate = 16_000;                             // Hz
        const double seconds = 3;
        var n = (int)(rate * seconds);
        var t = Enumerable.Range(0, n).Select(i => i / (double)rate).ToArray();

        // The wanted sound: a short melody of plucked notes (A4, C5, E5, A5 ...), each with harmonics.
        double[] notes = [440, 523.25, 659.25, 880, 659.25, 523.25];
        var melody = t.Select(x =>
        {
            var k = (int)(x / 0.5);
            var f = notes[k % notes.Length];
            var since = x - 0.5 * k;
            return 0.4 * Math.Exp(-3 * since) * (Math.Sin(2 * Math.PI * f * x) + 0.4 * Math.Sin(4 * Math.PI * f * x) + 0.2 * Math.Sin(6 * Math.PI * f * x));
        }).ToArray();
        var rng = new GraviRandom(3);
        var noisy = melody.Select((m, i) => m + 0.12 * Math.Sin(2 * Math.PI * 50 * t[i]) + 0.08 * Math.Sin(2 * Math.PI * 100 * t[i])   // hum
                                              + 0.10 * Math.Sin(2 * Math.PI * 1000 * t[i]) + rng.Normal(0, 0.01)).ToArray();   // whistle, hiss

        // Through a real file and back, as an audio tool would: 16-bit PCM in the temp folder.
        var input = Path.Combine(Path.GetTempPath(), "gallery-voice-note.wav");
        var cleaned = Path.Combine(Path.GetTempPath(), "gallery-voice-note-clean.wav");
        WavFile.Write(input, rate, NdArray.FromValues(noisy.Select(v => v / 1.2)), WavSampleFormat.Int16, normalized: true);
        var recording = WavFile.Read(input, normalize: true);
        var audio = recording.Data.ToArray();

        // A 4th-order band-stop 960–1040 Hz for the whistle, then an 8th-order high-pass at 150 Hz for the
        // hum and its harmonic. Second-order sections keep high orders stable; filtfilt keeps phase intact.
        var whistle = Signal.ButterSos(4, new[] { 960.0, 1040.0 }, FilterType.Bandstop, fs: rate);
        var hum = Signal.ButterSos(8, 150.0, FilterType.Highpass, fs: rate);
        var filtered = Signal.Sosfiltfilt(hum, Signal.Sosfiltfilt(whistle, audio));
        WavFile.Write(cleaned, rate, NdArray.FromValues(filtered), WavSampleFormat.Int16, normalized: true);

        // How much closer to the clean melody? Signal-to-noise ratio before and after, in dB.
        double Snr(double[] x) => 10 * Math.Log10(melody.Sum(m => m * m) / x.Select((v, i) => Math.Pow(v * 1.2 - melody[i], 2)).Sum());
        double Tone(double[] x, double hz)
        {
            var (f, p) = Signal.Welch(x, fs: rate, nperseg: 4096);
            return 10 * Math.Log10(p[Array.IndexOf(f, f.MinBy(v => Math.Abs(v - hz)))]);
        }

        // Spectrograms before and after, 0–2.5 kHz, in decibels.
        double[,] Picture(double[] x, out double[] times, out double[] freqs)
        {
            var (f, tt, sxx, _) = Signal.Spectrogram(x, fs: rate, nperseg: 512, noverlap: 384);
            var rows = Enumerable.Range(0, f.Length).Where(i => f[i] <= 2500).ToArray();
            var image = new double[rows.Length, tt.Length];
            for (var r = 0; r < rows.Length; r++)
                for (var c = 0; c < tt.Length; c++) image[r, c] = 10 * Math.Log10(sxx![rows[r], c] + 1e-12);
            (times, freqs) = (tt, rows.Select(i => f[i]).ToArray());
            return image;
        }
        var before = Picture(audio, out var tb, out var fb);
        var after = Picture(filtered, out _, out _);

        var figure = GalleryCharts.Figure(11, 4.6);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        var extent = (tb[0], tb[^1], fb[0], fb[^1]);
        var norm = new Normalize(-110, -20);
        left.ImShow(before, colormap: "magma", norm: norm, extent: extent, origin: ImageOrigin.Lower);
        left.SetTitle("Recording: hum and 1 kHz whistle").SetXLabel("time (s)").SetYLabel("frequency (Hz)");
        right.ImShow(after, colormap: "magma", norm: norm, extent: extent, origin: ImageOrigin.Lower);
        right.AddColorbar("power (dB/Hz)");
        right.SetTitle("After band-stop and high-pass").SetXLabel("time (s)");
        output.Chart(figure);

        output.Metric("File", $"{recording.SampleRate:N0} Hz, {recording.BitsPerSample}-bit, {new FileInfo(input).Length / 1024} KB", "written and read back with WavFile");
        output.Metric("Whistle at 1 kHz", $"{Tone(audio, 1000) - Tone(filtered, 1000):F0} dB quieter");
        output.Metric("Hum at 50 Hz", $"{Tone(audio, 50) - Tone(filtered, 50):F0} dB quieter");
        output.Metric("Signal-to-noise ratio", $"{Snr(audio):F1} → {Snr(filtered):F1} dB", "against the clean melody");
        output.Line($"The cleaned file is at {cleaned}. The melody's notes (440–1,760 Hz) pass untouched; only the two steady lines are gone.");
    }
}
