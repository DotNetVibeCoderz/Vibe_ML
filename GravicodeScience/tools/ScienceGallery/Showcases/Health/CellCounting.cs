using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Health;

/// <summary>Counting cell nuclei in a microscope image, splitting touching cells from distance-transform seeds.</summary>
public sealed class CellCounting : Showcase
{
    public override string Title => "Count the cells in a microscope image, even the touching ones";
    public override Domain Domain => Domain.Health;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Smooth, threshold and label nuclei, then split touching ones using the peaks of the distance transform.";
    public override string Scenario =>
        "A lab technician counts stained nuclei in fluorescence images to measure how fast a cell culture grows. " +
        "Thresholding and labelling connected regions is quick but counts two touching nuclei as one. The " +
        "distance transform peaks once near the centre of every nucleus. Using those peaks as seeds and giving " +
        "every foreground pixel to its nearest seed cuts touching nuclei apart between their centres.";

    public override void Run(ShowcaseOutput output)
    {
        // A synthetic 256 × 256 fluorescence image: 48 round-ish nuclei, some touching, on a dim, noisy background.
        const int size = 256;
        var rng = new GraviRandom(21);
        var nuclei = Enumerable.Range(0, 48).Select(_ => (X: rng.Uniform(12, size - 12), Y: rng.Uniform(12, size - 12), R: rng.Uniform(6, 10))).ToArray();
        var image = new double[size, size];
        for (var r = 0; r < size; r++)
            for (var c = 0; c < size; c++)
            {
                var signal = nuclei.Max(n => 1 / (1 + Math.Exp((Math.Sqrt((c - n.X) * (c - n.X) + (r - n.Y) * (r - n.Y)) - n.R) * 1.5)));
                image[r, c] = 0.1 + 0.8 * signal + rng.Normal(0, 0.12);
            }

        // 1. Smooth away pixel noise; 2. threshold half-way between background and nuclei.
        var smooth = NdImage.GaussianFilter(NdArray.FromArray(image), sigma: 1.5);
        var mask = NdImage.BinaryFillHoles(smooth.Map(v => v > 0.5 ? 1 : 0));
        var (_, naiveCount) = NdImage.Label(mask);

        // 3. Distance to the background: each nucleus becomes a hill with its top near the centre.
        var distance = NdImage.DistanceTransformEdt(mask);
        // 4. Seeds: the highest point within an 11-pixel window, well inside a nucleus. A light blur of the
        //    distance map first removes the ripples a jagged edge leaves, which would otherwise give one
        //    nucleus two seeds; dilating the seeds then merges any that still sit side by side.
        var hills = NdImage.GaussianFilter(distance, sigma: 1.0);
        var localMax = NdImage.MaximumFilter(hills, 11);
        var seeds = NdArray.Zeros(size, size);
        for (var i = 0; i < seeds.Size; i++)
            if (hills.At(i) >= 3 && hills.At(i) == localMax.At(i)) seeds.SetAt(i, 1);
        var (markers, seedCount) = NdImage.Label(NdImage.MaximumFilter(seeds, 5));
        // 5. Give every foreground pixel to its nearest seed. The distance transform of "not a seed" with
        //    indices returned says, for each pixel, where the closest seed pixel is.
        var nearest = NdImage.DistanceTransformEdt(seeds.Map(v => 1 - v), null, returnDistances: false, returnIndices: true).Indices!;
        var regions = NdArray.Zeros(size, size);
        for (var r = 0; r < size; r++)
            for (var c = 0; c < size; c++)
                if (mask[r, c] > 0) regions[r, c] = markers[(int)nearest[0, r, c], (int)nearest[1, r, c]];
        var areas = Enumerable.Range(1, seedCount).Select(l => regions.ToArray().Count(v => v == l)).ToArray();
        var centres = NdImage.CenterOfMass(mask, regions, Enumerable.Range(1, seedCount).ToArray());

        var figure = GalleryCharts.Figure(10.5, 5.2);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        left.ImShow(image, colormap: "gray");
        left.SetTitle($"Fluorescence image: {naiveCount} blobs by plain labelling").AxisOff();

        var labelled = regions.To2DArray();
        for (var r = 0; r < size; r++)
            for (var c = 0; c < size; c++)
                labelled[r, c] = labelled[r, c] == 0 ? double.NaN : (labelled[r, c] * 37) % 101;   // scatter neighbouring labels across the colormap
        right.ImShow(labelled, colormap: "turbo");
        right.Scatter(centres.Select(p => p[1]).ToArray(), centres.Select(p => p[0]).ToArray(), "nucleus centre", Color.Parse("#FFFFFF"), sizes: [14]);
        right.SetTitle($"Split from distance peaks: {seedCount} nuclei").AxisOff();
        output.Chart(figure);

        output.Metric("Nuclei placed", nuclei.Length.ToString(), "the ground truth of this synthetic image");
        output.Metric("Plain labelling", naiveCount.ToString(), $"{nuclei.Length - naiveCount} lost to touching neighbours");
        output.Metric("Seeded split", seedCount.ToString(), $"error {Math.Abs(seedCount - nuclei.Length) * 100.0 / nuclei.Length:F0}%");
        output.Metric("Median nucleus area", $"{Statistics.Median(NdArray.FromValues(areas.Select(a => (double)a))):F0} px", "useful to flag debris and clumps");
        output.Line($"Splitting touching nuclei moved the count from {naiveCount} to {seedCount}, against {nuclei.Length} placed. " +
                    "Heavily overlapping nuclei can still share one seed, which is why growth curves compare counts made the same way.");
    }
}
