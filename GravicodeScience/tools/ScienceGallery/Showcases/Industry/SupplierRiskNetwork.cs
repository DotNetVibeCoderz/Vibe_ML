using Gravicode.Science.GraviGraph;
using Gravicode.Science.GraviGraph.Algorithms;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Industry;

/// <summary>Stress-testing a multi-tier supply network: which single supplier failure stops the most production?</summary>
public sealed class SupplierRiskNetwork : Showcase
{
    public override string Title => "Which supplier could stop all three factories?";
    public override Domain Domain => Domain.Industry;
    public override GraviLibrary[] Libraries => [GraviLibrary.GraviGraph, GraviLibrary.GraviNum];
    public override string Summary => "Simulate each supplier failing through a four-tier network and compare with betweenness centrality.";
    public override string Scenario =>
        "An electronics maker knows its direct suppliers but not who supplies them. After mapping three tiers " +
        "upstream, the supply-chain team removes each supplier in turn and lets the shortage cascade: a company " +
        "stops when every source of one of its inputs has stopped. The revenue each failure would halt shows " +
        "where a second source is worth paying for. Graph centrality is compared too, because it is the usual " +
        "shortcut and it does not know about redundancy.";

    public override void Run(ShowcaseOutput output)
    {
        // name, tier (3 = raw material ... 0 = our factories), what it makes, what it needs.
        (string Name, int Tier, string Makes, string[] Needs)[] companies =
        [
            ("WaferCo A", 3, "wafer", []), ("WaferCo B", 3, "wafer", []), ("CopperMine", 3, "copper", []),
            ("Smelter", 3, "copper", []), ("ResinWorks", 3, "resin", []),
            ("Foundry 1", 2, "chip", ["wafer"]), ("Foundry 2", 2, "chip", ["wafer"]), ("PCB Fab A", 2, "pcb", []),
            ("PCB Fab B", 2, "pcb", []), ("Molder 1", 2, "housing", ["resin"]), ("Molder 2", 2, "housing", ["resin"]),
            ("CableCo", 2, "cable", ["copper"]), ("CableCo B", 2, "cable", ["copper"]),
            ("Board A", 1, "board", ["chip", "pcb"]), ("Board B", 1, "board", ["chip", "pcb"]), ("Board C", 1, "board", ["chip", "pcb"]),
            ("Case A", 1, "enclosure", ["housing"]), ("Case B", 1, "enclosure", ["housing"]),
            ("Harness A", 1, "harness", ["cable"]), ("Harness B", 1, "harness", ["cable"]),
            ("Plant Bekasi", 0, "product", ["board", "enclosure", "harness"]),
            ("Plant Batam", 0, "product", ["board", "enclosure", "harness"]),
            ("Plant Cikarang", 0, "product", ["board", "enclosure", "harness"]),
        ];
        var revenue = new Dictionary<string, double> { ["Plant Bekasi"] = 420, ["Plant Batam"] = 260, ["Plant Cikarang"] = 310 };  // Rp billion/month

        // Who supplies whom: an edge supplier -> customer, for every supplier that has qualified.
        (string From, string To)[] contracts =
        [
            ("WaferCo A", "Foundry 1"), ("WaferCo B", "Foundry 1"), ("WaferCo A", "Foundry 2"), ("WaferCo B", "Foundry 2"),
            ("CopperMine", "CableCo"), ("Smelter", "CableCo"), ("ResinWorks", "Molder 1"), ("ResinWorks", "Molder 2"),
            ("Foundry 1", "Board A"), ("Foundry 2", "Board A"), ("Foundry 1", "Board B"), ("Foundry 2", "Board B"), ("Foundry 2", "Board C"),
            ("PCB Fab A", "Board A"), ("PCB Fab B", "Board A"), ("PCB Fab A", "Board B"), ("PCB Fab B", "Board C"),
            ("Molder 1", "Case A"), ("Molder 2", "Case A"), ("Molder 2", "Case B"),
            ("CableCo", "Harness A"), ("CableCo", "Harness B"), ("CableCo B", "Harness B"), ("Smelter", "CableCo B"),
            ("Board A", "Plant Bekasi"), ("Board B", "Plant Bekasi"), ("Board B", "Plant Batam"), ("Board C", "Plant Cikarang"),
            ("Board A", "Plant Cikarang"), ("Case A", "Plant Bekasi"), ("Case A", "Plant Batam"), ("Case B", "Plant Cikarang"),
            ("Case B", "Plant Batam"), ("Harness A", "Plant Bekasi"), ("Harness B", "Plant Batam"), ("Harness B", "Plant Cikarang"),
        ];
        var network = new Graph(directed: true);
        foreach (var c in companies) network.AddNode(c.Name, label: c.Tier);
        var index = companies.Select((c, i) => (c.Name, i)).ToDictionary(p => p.Name, p => p.i);
        foreach (var (from, to) in contracts) network.AddEdge(index[from], index[to]);

        // Knock one company out and let the shortage cascade: a company stops when, for some input it
        // needs, every supplier of that input has stopped. Repeat until nothing more changes.
        double RevenueLost(int failed)
        {
            var down = new HashSet<int> { failed };
            for (var changed = true; changed;)
            {
                changed = false;
                for (var i = 0; i < companies.Length; i++)
                {
                    if (down.Contains(i)) continue;
                    var starved = companies[i].Needs.Any(input =>
                        network.Predecessors(i).Where(p => companies[p.Source].Makes == input).All(p => down.Contains(p.Source)));
                    if (starved) { down.Add(i); changed = true; }
                }
            }
            return down.Where(i => companies[i].Tier == 0).Sum(i => revenue[companies[i].Name]);
        }
        var suppliers = Enumerable.Range(0, companies.Length).Where(i => companies[i].Tier > 0).ToArray();
        var impact = suppliers.ToDictionary(i => i, RevenueLost);
        var betweenness = GraphAlgorithms.BetweennessCentrality(network).ToArray();
        var worst = suppliers.MaxBy(i => impact[i]);
        var mostCentral = suppliers.MaxBy(i => betweenness[i]);
        // Where the shortcut misleads: the most central supplier whose failure would stop nothing.
        var falseAlarm = suppliers.Where(i => impact[i] == 0).MaxBy(i => betweenness[i]);
        var centralityRank = suppliers.OrderByDescending(i => betweenness[i]).ToList().IndexOf(falseAlarm) + 1;
        var totalRevenue = revenue.Values.Sum();

        // Draw the tiers left to right, raw materials first; marker size shows revenue at risk.
        var x = companies.Select(c => 3.0 - c.Tier).ToArray();
        var y = new double[companies.Length];
        foreach (var tier in companies.GroupBy(c => c.Tier))
        {
            var members = tier.Select(c => index[c.Name]).ToArray();
            for (var k = 0; k < members.Length; k++) y[members[k]] = (members.Length - 1) / 2.0 - k;
        }
        var figure = GalleryCharts.Figure(10.5, 6.2);
        var ax = figure.AddSubplot();
        foreach (var edge in network.Edges())
            ax.Plot([x[edge.Source], x[edge.Target]], [y[edge.Source], y[edge.Target]], color: Color.Parse("#C9D0D8"), lineWidth: 0.9);
        ax.Scatter(suppliers.Select(i => x[i]).ToArray(), suppliers.Select(i => y[i]).ToArray(), "supplier (size: revenue halted if it fails)",
                   sizes: suppliers.Select(i => 40 + 900 * impact[i] / totalRevenue).ToArray(),
                   colorValues: suppliers.Select(i => impact[i]).ToArray(), colormap: "YlOrRd", edgeColor: Color.Parse("#5F6B7A"));
        ax.AddColorbar("revenue halted (Rp billion / month)");
        var plants = Enumerable.Range(0, companies.Length).Where(i => companies[i].Tier == 0).ToArray();
        ax.Scatter(plants.Select(i => x[i]).ToArray(), plants.Select(i => y[i]).ToArray(), "our factories", Color.Parse(GalleryCharts.Ink),
                   sizes: [160], marker: MarkerStyle.Square);
        for (var i = 0; i < companies.Length; i++)
            ax.Text(x[i], y[i] + 0.32, companies[i].Name, fontSize: 9.5, align: TextAlign.Center);
        ax.SetXTicks([0, 1, 2, 3], ["tier 3: materials", "tier 2: parts", "tier 1: modules", "assembly"]);
        ax.YAxis.TickLabelsVisible = false;
        ax.SetTitle("Supply network and the cost of losing each supplier");
        ax.Legend(LegendLocation.OutsideBottom, columns: 2);
        output.Chart(figure);

        output.Metric("Companies / contracts", $"{companies.Length} / {contracts.Length}");
        output.Metric("Single worst failure", companies[worst].Name, $"halts Rp {impact[worst]:F0} billion/month, {impact[worst] / totalRevenue * 100:F0}% of revenue");
        output.Metric("Highest betweenness", companies[mostCentral].Name, $"halts Rp {impact[mostCentral]:F0} billion/month if it fails");
        output.Metric("Suppliers that stop nothing", suppliers.Count(i => impact[i] == 0).ToString(), "a second source covers them");
        output.Line($"{companies[worst].Name} is a tier-{companies[worst].Tier} supplier" +
                    (companies[worst].Tier > 1 ? " the company has no contract with" : "") +
                    $", yet losing it stops {impact[worst] / totalRevenue * 100:F0}% of output.");
        output.Line($"Centrality alone ranks {companies[falseAlarm].Name} {centralityRank} of {suppliers.Length}, because many paths cross it; " +
                    "losing it stops nothing, because every customer it serves has a second source. Centrality counts paths, not alternatives.");
    }
}
