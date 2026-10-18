using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Utility;

/// <summary>Choosing the right filter for a noisy photo, judged by PSNR against the clean original.</summary>
public sealed class PhotoDenoise : Showcase
{
    public override string Title => "Rescue a grainy photo with dead pixels";
    public override Domain Domain => Domain.Utility;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Gaussian, uniform and median filters compared by PSNR on sensor grain plus salt-and-pepper pixels.";
    public override string Scenario =>
        "A photo from an old phone sensor has fine grain everywhere and scattered stuck pixels, fully white or " +
        "black. A blur averages grain away but smears the stuck pixels into grey blotches; a median filter " +
        "throws outliers away and keeps edges sharp. Comparing each filter's peak signal-to-noise ratio (PSNR) " +
        "against the clean original says which one to apply, and at what strength.";

    public override void Run(ShowcaseOutput output)
    {
        // A clean 240 × 240 test scene, brightness 0–1: a sky gradient, a sun, hills and a house.
        const int size = 240;
        var clean = new double[size, size];
        for (var r = 0; r < size; r++)
            for (var c = 0; c < size; c++)
            {
                var v = 0.75 - 0.35 * r / size;                                               // sky
                if (Math.Pow(c - 175, 2) + Math.Pow(r - 55, 2) < 22 * 22) v = 0.97;            // sun
                if (r > 150 + 25 * Math.Sin(c / 30.0)) v = 0.32 + 0.1 * Math.Sin(c / 7.0) * Math.Sin(r / 9.0);   // hills
                if (c is > 50 and < 110 && r is > 130 and < 190) v = 0.55;                    // house
                if (c is > 70 and < 88 && r is > 160 and < 190) v = 0.15;                     // door
                if (r > 100 && r < 131 && Math.Abs(c - 80) < (r - 100) * 1.0) v = 0.22;      // roof
                clean[r, c] = v;
            }

        // The sensor: Gaussian grain plus 4% stuck pixels.
        var rng = new GraviRandom(13);
        var noisy = new double[size, size];
        for (var r = 0; r < size; r++)
            for (var c = 0; c < size; c++)
                noisy[r, c] = rng.NextBernoulli(0.04) ? (rng.NextBernoulli(0.5) ? 1.0 : 0.0) : Math.Clamp(clean[r, c] + rng.Normal(0, 0.07), 0, 1);

        // PSNR = 10 log10(peak² / mean squared error); peak is 1 for 0–1 images. Higher is better.
        double Psnr(NdArray image)
        {
            var mse = 0.0;
            for (var r = 0; r < size; r++)
                for (var c = 0; c < size; c++) mse += Math.Pow(image[r, c] - clean[r, c], 2);
            return 10 * Math.Log10(1 / (mse / (size * size)));
        }
        var photo = NdArray.FromArray(noisy);
        var candidates = new (string Name, NdArray Result)[]
        {
            ("no filter", photo),
            ("Gaussian σ = 1", NdImage.GaussianFilter(photo, sigma: 1.0)),
            ("Gaussian σ = 2", NdImage.GaussianFilter(photo, sigma: 2.0)),
            ("uniform 3 × 3", NdImage.UniformFilter(photo, 3)),
            ("median 3 × 3", NdImage.MedianFilter(photo, 3)),
            ("median 5 × 5", NdImage.MedianFilter(photo, 5)),
        };
        var scores = candidates.Select(c => (c.Name, Psnr: Psnr(c.Result), c.Result)).ToArray();
        var best = scores.MaxBy(s => s.Psnr);

        var figure = GalleryCharts.Figure(10.5, 5.4);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        left.ImShow(noisy, colormap: "gray", norm: new Normalize(0, 1));
        left.SetTitle($"From the sensor: PSNR {scores[0].Psnr:F1} dB").AxisOff();
        right.ImShow(best.Result.To2DArray(), colormap: "gray", norm: new Normalize(0, 1));
        right.SetTitle($"{best.Name}: PSNR {best.Psnr:F1} dB").AxisOff();
        output.Chart(figure);

        foreach (var s in scores.Skip(1).OrderByDescending(s => s.Psnr).Take(4))
            output.Metric(s.Name, $"{s.Psnr:F1} dB", $"{s.Psnr - scores[0].Psnr:+0.0;-0.0} dB on the raw photo");
        var gaussian = scores.Where(s => s.Name.StartsWith("Gaussian")).MaxBy(s => s.Psnr);
        output.Line($"The {best.Name} wins by {best.Psnr - gaussian.Psnr:F1} dB over the best Gaussian blur: a blur spreads each stuck pixel over " +
                    "its neighbours, while a median simply ignores it. Each +3 dB halves the remaining error energy.");
    }
}
