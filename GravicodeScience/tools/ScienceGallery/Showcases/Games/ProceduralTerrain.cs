using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Games;

/// <summary>Generating an island map by spectral synthesis: coloured noise shaped in the Fourier domain.</summary>
public sealed class ProceduralTerrain : Showcase
{
    public override string Title => "Generate a believable island for a game map";
    public override Domain Domain => Domain.Games;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Shape white noise by 1/f^β in the 2-D Fourier domain, then add a coastline falloff and smoothing.";
    public override string Scenario =>
        "An open-world game needs a new island every play-through. Real landscapes have a power spectrum close " +
        "to 1/f^β: large features are tall, small ones are short. Filtering random noise to that spectrum with a " +
        "2-D FFT gives realistic hills in one step, a radial falloff turns it into an island, and a light " +
        "Gaussian filter takes the edge off the smallest bumps. The seed is all a save file needs to store.";

    public override void Run(ShowcaseOutput output)
    {
        const int size = 256;                         // grid cells, 40 m each: a 10 km island
        const double beta = 2.4;                      // spectral exponent; real terrain is roughly 2–3
        const int seed = 2024;

        // 1. White noise, into the frequency domain.
        var rng = new GraviRandom(seed);
        var spectrum = Fft.Fft2(rng.Normal(0, 1, size, size));

        // 2. Scale every frequency's amplitude by 1/f^(β/2), so power falls off as 1/f^β.
        var freq = Fft.FftFreq(size);
        var shaped = new Complex[size * size];
        for (var r = 0; r < size; r++)
            for (var c = 0; c < size; c++)
            {
                var f = Math.Sqrt(freq[r] * freq[r] + freq[c] * freq[c]);
                shaped[r * size + c] = f == 0 ? Complex.Zero : spectrum[r, c] / Math.Pow(f, beta / 2);
            }
        var noise = Fft.Ifft2(new ComplexNdArray(shaped, size, size)).Real();

        // 3. Island shape: push the edges down with a radial falloff, smooth a little, scale to metres.
        var height = new double[size, size];
        var (lo, hi) = (noise.ToArray().Min(), noise.ToArray().Max());
        for (var r = 0; r < size; r++)
            for (var c = 0; c < size; c++)
            {
                var d = Math.Sqrt(Math.Pow((r - size / 2.0) / (size / 2.0), 2) + Math.Pow((c - size / 2.0) / (size / 2.0), 2));
                height[r, c] = (noise[r, c] - lo) / (hi - lo) - 1.1 * d * d;
            }
        var smooth = NdImage.GaussianFilter(NdArray.FromArray(height), sigma: 1.2);
        var top = smooth.ToArray().Max();
        var seaLevel = 0.15;
        var elevation = smooth.Map(h => (h - seaLevel) / (top - seaLevel) * 1800).To2DArray();   // metres above sea level

        // Check the spectrum we asked for: radially averaged power against frequency, slope on log-log axes.
        var power = new double[size / 2];
        var counts = new int[size / 2];
        var made = Fft.Fft2(noise);
        for (var r = 0; r < size; r++)
            for (var c = 0; c < size; c++)
            {
                var k = (int)Math.Round(Math.Sqrt(freq[r] * freq[r] + freq[c] * freq[c]) * size);
                if (k is > 0 and < size / 2) { power[k] += Math.Pow(made[r, c].Magnitude, 2); counts[k]++; }
            }
        var ks = Enumerable.Range(2, size / 2 - 2).ToArray();
        var fit = Stats.Linregress(ks.Select(k => Math.Log10(k / (double)size)).ToArray(), ks.Select(k => Math.Log10(power[k] / counts[k])).ToArray());

        var land = elevation.Cast<double>().Count(h => h > 0) * 100.0 / (size * size);
        var figure = GalleryCharts.Figure(11, 5.2);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        var km = Enumerable.Range(0, size).Select(i => i * 0.04).ToArray();
        // matplotlib's "terrain" colormap turns from sea to land 22% of the way up, so put 0 m there.
        left.ImShow(elevation, colormap: "terrain", norm: new Normalize(-1800 * 0.22 / 0.78, 1800), extent: (0, size * 0.04, 0, size * 0.04), origin: ImageOrigin.Lower);
        left.AddColorbar("elevation (m)");                     // before the contour, so it describes the image
        left.Contour(km, km, elevation, levels: [0], color: Color.Parse(GalleryCharts.Ink));
        left.SetTitle($"Island from seed {seed} (black line: coast)").SetXLabel("km").SetYLabel("km");

        right.Plot(ks.Select(k => k / (double)size).ToArray(), ks.Select(k => power[k] / counts[k]).ToArray(), "generated terrain", marker: MarkerStyle.Circle, markerSize: 3);
        right.Plot(ks.Select(k => k / (double)size).ToArray(), ks.Select(k => Math.Pow(10, fit.Intercept) * Math.Pow(k / (double)size, fit.Slope)).ToArray(),
                   $"fitted slope {fit.Slope:F2}", Color.Parse("#D9363E"), style: LineStyle.Dashed);
        var spectrumValues = ks.Select(k => power[k] / counts[k]).ToArray();
        right.SetXScale("log").SetYScale("log").SetXLim(1.5 / size, 0.6).SetYLim(spectrumValues.Min() / 2, spectrumValues.Max() * 3).SetTitle("Power spectrum: big features tall, small ones short")
             .SetXLabel("spatial frequency (cycles per cell)").SetYLabel("power");
        right.Legend(LegendLocation.UpperRight);
        output.Chart(figure);

        output.Metric("Map", $"{size} × {size} cells", "40 m per cell, about 10 × 10 km");
        output.Metric("Land", $"{land:F0}% of the map", $"highest peak {elevation.Cast<double>().Max():F0} m");
        output.Metric("Spectral slope", fit.Slope.ToString("F2"), $"asked for −{beta}");
        output.Line($"The measured slope of {fit.Slope:F2} matches the −{beta} requested, so the hills have the size statistics of real terrain; " +
                    "a different seed gives a different island with the same character.");
    }
}
