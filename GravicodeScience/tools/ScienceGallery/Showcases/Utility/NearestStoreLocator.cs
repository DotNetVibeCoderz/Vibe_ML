using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Utility;

/// <summary>Nearest-store lookups with a k-d tree, Voronoi service areas, and where an eleventh store helps most.</summary>
public sealed class NearestStoreLocator : Showcase
{
    public override string Title => "Which store is nearest, and where should the next one open?";
    public override Domain Domain => Domain.Utility;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "A k-d tree answers 3,000 nearest-store queries; Voronoi cells show each store's catchment.";
    public override string Scenario =>
        "A pharmacy chain's app shows customers their nearest branch, and the expansion team wants to know " +
        "which neighbourhoods are badly served. A k-d tree finds the nearest of the ten stores for every " +
        "customer address in milliseconds. The Voronoi diagram of the stores draws each one's catchment area, " +
        "and testing candidate sites for an eleventh store shows where it would cut travel the most.";

    public override void Run(ShowcaseOutput output)
    {
        const double width = 20, height = 15;                              // km, the city's bounding box
        var rng = new GraviRandom(40);

        // Customers live mostly in five neighbourhoods, the rest spread across the city.
        (double X, double Y, double Spread, int Homes)[] neighbourhoods =
            [(4, 4, 1.4, 700), (9, 11, 1.8, 650), (15, 5, 1.2, 500), (16.5, 12, 1.0, 450), (10, 6, 2.2, 400)];
        var homes = new List<double[]>();
        foreach (var n in neighbourhoods)
            for (var i = 0; i < n.Homes; i++)
                homes.Add([Math.Clamp(rng.Normal(n.X, n.Spread), 0, width), Math.Clamp(rng.Normal(n.Y, n.Spread), 0, height)]);
        for (var i = 0; i < 300; i++) homes.Add([rng.Uniform(0, width), rng.Uniform(0, height)]);
        double[][] stores = [[3, 3], [5.5, 5], [2, 11], [8, 9], [11, 12.5], [10, 4], [13.5, 2.5], [14, 8], [18.5, 3], [6, 14]];

        // One k-d tree over the stores; one batched query for every home.
        var tree = new KDTree(stores);
        var homeMatrix = new double[homes.Count, 2];
        for (var i = 0; i < homes.Count; i++) (homeMatrix[i, 0], homeMatrix[i, 1]) = (homes[i][0], homes[i][1]);
        var (distance, nearest) = tree.Query(homeMatrix, k: 1);
        var km = distance.ToArray();
        var load = Enumerable.Range(0, stores.Length).Select(s => Enumerable.Range(0, homes.Count).Count(i => nearest[i, 0] == s)).ToArray();

        // Candidate sites on a 500 m grid: each customer would use the new store if it were closer.
        var best = (X: 0.0, Y: 0.0, Mean: double.MaxValue);
        for (var cx = 0.5; cx < width; cx += 0.5)
            for (var cy = 0.5; cy < height; cy += 0.5)
            {
                var mean = 0.0;
                for (var i = 0; i < homes.Count; i++)
                    mean += Math.Min(km[i], Math.Sqrt(Math.Pow(homes[i][0] - cx, 2) + Math.Pow(homes[i][1] - cy, 2)));
                mean /= homes.Count;
                if (mean < best.Mean) best = (cx, cy, mean);
            }

        // Catchment areas: the Voronoi cell of each store, clipped to the city.
        var cells = new Voronoi(tree.Data).FinitePolygons(0, width, 0, height);
        var figure = GalleryCharts.Figure(10, 6.4);
        var ax = figure.AddSubplot();
        var busiest = load.Max();
        for (var s = 0; s < stores.Length; s++)
        {
            var polygon = cells[s];
            ax.AddPatch(new PatchArtist
            {
                Vertices = Enumerable.Range(0, polygon.Shape[0]).Select(k => (polygon[k, 0], polygon[k, 1])).ToArray(),
                FaceColor = GalleryCharts.Of(GraviLibrary.GraviText).WithAlpha(0.08f + 0.5f * load[s] / busiest),
                EdgeColor = Color.Parse("#FFFFFF"), EdgeWidth = 2,
            });
        }
        ax.Scatter(homes.Select(h => h[0]).ToArray(), homes.Select(h => h[1]).ToArray(), "customer", Color.Parse("#5F6B7A"), sizes: [3], alpha: 0.5);
        ax.Scatter(stores.Select(s => s[0]).ToArray(), stores.Select(s => s[1]).ToArray(), "store (cell shade: customers served)", Color.Parse(GalleryCharts.Ink), sizes: [70], marker: MarkerStyle.Square);
        for (var s = 0; s < stores.Length; s++) ax.Text(stores[s][0] + 0.25, stores[s][1] + 0.25, load[s].ToString(), fontSize: 11);
        ax.Scatter([best.X], [best.Y], "best site for store 11", Color.Parse("#D9363E"), sizes: [260], marker: MarkerStyle.Star);
        ax.SetXLim(0, width).SetYLim(0, height).SetAspect(1.0);
        ax.SetTitle("Store catchments (Voronoi) and customer homes").SetXLabel("km east").SetYLabel("km north");
        ax.Legend(LegendLocation.OutsideRight);
        output.Chart(figure);

        output.Metric("Customers", homes.Count.ToString("N0"), $"{stores.Length} stores");
        output.Metric("Mean distance to nearest store", $"{km.Average():F2} km", $"{km.Count(d => d <= 2) * 100.0 / km.Length:F0}% within 2 km");
        output.Metric("Busiest store", $"{busiest:N0} customers", $"store at ({stores[Array.IndexOf(load, busiest)][0]}, {stores[Array.IndexOf(load, busiest)][1]}) km");
        output.Metric("With store 11", $"{best.Mean:F2} km mean", $"at ({best.X:F2}, {best.Y:F2}) km, {(1 - best.Mean / km.Average()) * 100:F0}% less travel");
        var relieved = tree.Query([best.X, best.Y]).Index;
        output.Line($"The new site would take customers from the store at ({stores[relieved][0]}, {stores[relieved][1]}) km, which now serves " +
                    $"{load[relieved]:N0}: {(double)load[relieved] / homes.Count * stores.Length:F1}× an equal share.");
    }
}
