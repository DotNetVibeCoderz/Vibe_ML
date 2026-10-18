using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Sparse;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Industry;

/// <summary>Shortest walking distances through warehouse aisles, then a short tour through a pick list.</summary>
public sealed class WarehousePickRoute : Showcase
{
    public override string Title => "Walk less on every warehouse pick list";
    public override Domain Domain => Domain.Industry;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Dijkstra through the aisle network, then nearest-neighbour and 2-opt to order the picks.";
    public override string Scenario =>
        "A picker in a distribution centre walks the aisles with a list of 14 items, usually in the order the " +
        "system printed them. Shelves block straight lines, so the true distance between two shelves is a path " +
        "along the aisles. Dijkstra's algorithm on the aisle graph gives every walking distance; a nearest-neighbour " +
        "tour improved by 2-opt then orders the list. Over a shift of fifty lists the saving is kilometres.";

    public override void Run(ShowcaseOutput output)
    {
        // The floor: 8 aisles, 4 m apart, each 21 m long with a cross aisle at the front (y = 0) and back
        // (y = 21). One graph node per metre of aisle; the packing desk sits 2 m left of aisle 0.
        const int aisles = 8, length = 22;
        int Node(int aisle, int y) => aisle * length + y;
        var depot = aisles * length;
        var position = new (double X, double Y)[depot + 1];
        var (from, to, metres) = (new List<int>(), new List<int>(), new List<double>());
        void Walkway(int a, int b, double m) { from.Add(a); to.Add(b); metres.Add(m); from.Add(b); to.Add(a); metres.Add(m); }
        for (var a = 0; a < aisles; a++)
            for (var y = 0; y < length; y++)
            {
                position[Node(a, y)] = (4 * a, y);
                if (y > 0) Walkway(Node(a, y - 1), Node(a, y), 1);
                if (a > 0 && (y == 0 || y == length - 1)) Walkway(Node(a - 1, y), Node(a, y), 4);
            }
        position[depot] = (-2, 0);
        Walkway(depot, Node(0, 0), 2);
        var floor = CsrMatrix.FromTriplets(depot + 1, depot + 1, from.ToArray(), to.ToArray(), metres.ToArray());

        // Today's pick list, in the order it was printed.
        var rng = new GraviRandom(9);
        var picks = Enumerable.Range(0, 14).Select(_ => Node(rng.Next(aisles), 1 + rng.Next(length - 2))).Distinct().ToArray();
        var stops = new[] { depot }.Concat(picks).ToArray();

        // Walking distance between every pair of stops (and how to walk it), one Dijkstra per stop.
        var paths = CsGraph.Dijkstra(floor, directed: false, indices: stops, returnPredecessors: true);
        double Distance(int i, int j) => paths.Distances[i, stops[j]];
        double TourLength(IReadOnlyList<int> tour) => tour.Zip(tour.Skip(1)).Sum(p => Distance(p.First, p.Second));

        var printed = Enumerable.Range(0, stops.Length).Append(0).ToList();

        // Nearest neighbour: always walk to the closest stop not yet visited...
        var tour = new List<int> { 0 };
        var left = Enumerable.Range(1, stops.Length - 1).ToHashSet();
        while (left.Count > 0)
        {
            var next = left.MinBy(j => Distance(tour[^1], j));
            tour.Add(next); left.Remove(next);
        }
        tour.Add(0);
        var nearest = TourLength(tour);
        // ...then 2-opt: reverse any stretch of the tour whose reversal shortens it, until none does.
        for (var improved = true; improved;)
        {
            improved = false;
            for (var i = 1; i < tour.Count - 2; i++)
                for (var k = i + 1; k < tour.Count - 1; k++)
                {
                    var delta = Distance(tour[i - 1], tour[k]) + Distance(tour[i], tour[k + 1])
                              - Distance(tour[i - 1], tour[i]) - Distance(tour[k], tour[k + 1]);
                    if (delta < -1e-9) { tour.Reverse(i, k - i + 1); improved = true; }
                }
        }

        // Expand the tour back into floor nodes, following the predecessor chains.
        List<(double X, double Y)> Walk(IReadOnlyList<int> order)
        {
            var points = new List<(double, double)>();
            for (var leg = 0; leg + 1 < order.Count; leg++)
            {
                var chain = new List<int>();
                for (var node = stops[order[leg + 1]]; node != stops[order[leg]]; node = paths.Predecessors![order[leg], node]) chain.Add(node);
                chain.Add(stops[order[leg]]);
                chain.Reverse();
                points.AddRange(chain.Select(c => position[c]));
            }
            return points;
        }

        var figure = GalleryCharts.Figure(11, 4.6);
        var (a1, a2) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        foreach (var (ax, order, name) in new[] { (a1, (IReadOnlyList<int>)printed, "As printed"), (a2, tour, "Optimised") })
        {
            for (var a = 0; a + 1 < aisles; a++)
                ax.AddPatch(new PatchArtist
                {
                    Vertices = [(4 * a + 0.8, 1), (4 * a + 3.2, 1), (4 * a + 3.2, 20), (4 * a + 0.8, 20)],
                    FaceColor = Color.Parse("#E4E8ED"),
                });
            var walk = Walk(order);
            ax.Plot(walk.Select(p => p.X).ToArray(), walk.Select(p => p.Y).ToArray(), "route",
                    name == "As printed" ? Color.Parse("#8A94A3") : GalleryCharts.Of(GraviLibrary.GraviText), lineWidth: 1.8);
            ax.Scatter(picks.Select(p => position[p].X).ToArray(), picks.Select(p => position[p].Y).ToArray(), "pick", Color.Parse("#D9363E"), sizes: [50]);
            ax.Scatter([position[depot].X], [position[depot].Y], "packing desk", Color.Parse(GalleryCharts.Ink), sizes: [110], marker: MarkerStyle.Square);
            ax.SetTitle($"{name}: {TourLength(order):F0} m").SetXLabel("metres").SetAspect(1.0);
        }
        a2.Legend(LegendLocation.OutsideRight);
        a1.SetYLabel("metres");
        output.Chart(figure);

        var optimised = TourLength(tour);
        output.Metric("Picks", (stops.Length - 1).ToString(), $"{aisles} aisles, {floor.Nnz / 2} walkway segments");
        output.Metric("As printed", $"{TourLength(printed):F0} m");
        output.Metric("Nearest neighbour", $"{nearest:F0} m", "greedy: always the closest next pick");
        output.Metric("After 2-opt", $"{optimised:F0} m", $"{(1 - optimised / TourLength(printed)) * 100:F0}% shorter than as printed");
        output.Line($"At 50 lists a shift this saves about {(TourLength(printed) - optimised) * 50 / 1000:F1} km of walking per picker per shift.");
    }
}
