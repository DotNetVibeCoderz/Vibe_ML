using Gravicode.Science.GraviGraph;
using Gravicode.Science.GraviGraph.Algorithms;
using Gravicode.Science.GraviNum;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Education;

/// <summary>Finding friendship circles in a class survey, to form study groups and spot isolated students.</summary>
public sealed class StudyGroupsFromFriendships : Showcase
{
    public override string Title => "Form study groups from who sits with whom";
    public override Domain Domain => Domain.Education;
    public override GraviLibrary[] Libraries => [GraviLibrary.GraviGraph, GraviLibrary.GraviNum];
    public override string Summary => "Community detection on a class friendship survey, with students drawn in a spectral layout.";
    public override string Scenario =>
        "At the start of term a homeroom teacher asks each of 32 students to name the classmates they work well " +
        "with. The answers form a friendship network. Label propagation finds the natural circles, modularity says " +
        "how clear-cut they are, and students with only one or no connections are listed so the teacher can place " +
        "them deliberately rather than leave them to be picked last.";

    public override void Run(ShowcaseOutput output)
    {
        string[] names =
        [
            "Ayu", "Bima", "Citra", "Dimas", "Eka", "Farah", "Galih", "Hana",
            "Intan", "Joko", "Kirana", "Lukas", "Maya", "Nanda", "Oki", "Putri",
            "Raka", "Sari", "Tegar", "Umi", "Vino", "Wulan", "Yoga", "Zahra",
            "Adit", "Bella", "Cahyo", "Dina", "Edo", "Fitri", "Gilang", "Hesti",
        ];
        // Four friendship circles of eight (the teacher does not know this). Friends within a circle
        // name each other often, across circles rarely. Two students are new and know almost nobody.
        var rng = new GraviRandom(17);
        var survey = new Graph(directed: false);
        foreach (var name in names) survey.AddNode(name);
        var newcomers = new HashSet<int> { 7, 22 };
        for (var a = 0; a < names.Length; a++)
            for (var b = a + 1; b < names.Length; b++)
            {
                var p = a / 8 == b / 8 ? 0.55 : 0.03;
                if (newcomers.Contains(a) || newcomers.Contains(b)) p *= 0.12;
                if (rng.NextBernoulli(p)) survey.AddEdge(a, b);
            }

        var circles = GraphAlgorithms.LabelPropagation(survey, seed: 4);
        var modularity = GraphAlgorithms.Modularity(survey, circles);
        var isolated = Enumerable.Range(0, names.Length).Where(i => survey.Degree(i) <= 1).ToArray();
        var central = GraphAlgorithms.BetweennessCentrality(survey).ToArray();
        var bridge = Array.IndexOf(central, central.Max());

        // Layout for the drawing. Start from a spectral layout - the Laplacian eigenvectors with the
        // smallest non-zero eigenvalues place linked students close together (SymmetricEigen sorts
        // descending, so those are the last columns; the very last is constant and skipped) - then
        // relax it with a few hundred force-directed steps: friends attract, everyone repels.
        var eigen = Decomposition.SymmetricEigen(survey.Laplacian());
        var n = names.Length;
        var px = eigen.Vectors.Column(n - 2).ToArray();
        var py = eigen.Vectors.Column(n - 3).ToArray();
        const double spacing = 0.12;
        for (var step = 0; step < 300; step++)
        {
            var (fx, fy) = (new double[n], new double[n]);
            for (var i = 0; i < n; i++)
                for (var j = 0; j < n; j++)
                {
                    if (i == j) continue;
                    var (dx, dy) = (px[i] - px[j], py[i] - py[j]);
                    var d2 = Math.Max(dx * dx + dy * dy, 1e-6);
                    var push = spacing * spacing / d2;                       // repulsion ~ 1/d
                    var pull = survey.HasEdge(i, j) ? Math.Sqrt(d2) / spacing : 0;  // spring ~ d²
                    fx[i] += dx * (push - pull); fy[i] += dy * (push - pull);
                }
            var temperature = 0.02 * (1 - step / 300.0);                    // cool down so it settles
            for (var i = 0; i < n; i++)
            {
                var length = Math.Max(Math.Sqrt(fx[i] * fx[i] + fy[i] * fy[i]), 1e-9);
                px[i] += fx[i] / length * Math.Min(length, temperature);
                py[i] += fy[i] / length * Math.Min(length, temperature);
            }
        }

        var figure = GalleryCharts.Figure(8.6, 6.2);
        var ax = figure.AddSubplot();
        foreach (var edge in survey.Edges())
            ax.Plot([px[edge.Source], px[edge.Target]], [py[edge.Source], py[edge.Target]],
                    color: Color.Parse("#C9D0D8"), lineWidth: 0.8);
        var groupCount = circles.Max() + 1;
        for (var c = 0; c < groupCount; c++)
        {
            var members = Enumerable.Range(0, n).Where(i => circles[i] == c).ToArray();
            ax.Scatter(members.Select(i => px[i]).ToArray(), members.Select(i => py[i]).ToArray(),
                       $"group {c + 1} ({members.Length})", sizes: [110], edgeColor: Color.Parse("#FFFFFF"), edgeWidth: 1.2);
        }
        for (var i = 0; i < n; i++) ax.Text(px[i], py[i] + 0.025, names[i], fontSize: 10, align: TextAlign.Center);
        ax.SetTitle("Class friendship network, coloured by detected circle").AxisOff();
        ax.Legend(LegendLocation.OutsideRight);
        output.Chart(figure);

        output.Metric("Friendships named", survey.EdgeCount.ToString());
        output.Metric("Circles found", groupCount.ToString(), "label propagation, no group count given");
        output.Metric("Modularity", modularity.ToString("F2"), modularity > 0.4 ? "strong, well-separated circles" : "weak structure");
        output.Metric("Best connector", names[bridge], "highest betweenness: links circles together");
        output.Line(isolated.Length == 0
            ? "Every student has at least two connections."
            : $"Place deliberately: {string.Join(" and ", isolated.Select(i => names[i]))} named at most one classmate " +
              $"and would otherwise drift. Pairing them with {names[bridge]}, who bridges circles, is a natural start.");
    }
}
