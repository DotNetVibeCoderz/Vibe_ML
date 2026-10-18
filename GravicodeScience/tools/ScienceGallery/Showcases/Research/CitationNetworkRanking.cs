using Gravicode.Science.GraviGraph;
using Gravicode.Science.GraviGraph.Algorithms;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using ScienceGallery.Models;
using ScienceGallery.Services;

namespace ScienceGallery.Showcases.Research;

/// <summary>Ranking papers in the Cora citation network by PageRank rather than by raw citation counts.</summary>
public sealed class CitationNetworkRanking : Showcase
{
    public override string Title => "Which papers does the field actually build on?";
    public override Domain Domain => Domain.Research;
    public override GraviLibrary[] Libraries => [GraviLibrary.GraviGraph, GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "PageRank over 5,429 citations between 2,708 machine-learning papers, against plain citation counts.";
    public override string Scenario =>
        "A literature review starts by finding the papers everyone builds on. Counting citations treats a citation " +
        "from an obscure paper the same as one from a landmark. PageRank, the algorithm behind early web search, " +
        "lets each paper pass its own importance along its references, so a paper cited by influential papers " +
        "ranks higher. On the real Cora dataset it shows which topic areas hold the most influence.";

    public override void Run(ShowcaseOutput output)
    {
        // Edges are stored citing -> cited, so importance flows towards the papers being cited.
        var cora = Graph.Load(Datasets.Path("cora_graph.json"));
        var rank = GraphAlgorithms.PageRank(cora, damping: 0.85).ToArray();
        var citations = Enumerable.Range(0, cora.NodeCount).Select(i => (double)cora.InDegree(i)).ToArray();

        // Do the two orderings agree? Spearman's rank correlation compares the orderings, not the values.
        var agreement = Stats.Spearmanr(citations, rank);
        var topByRank = Enumerable.Range(0, cora.NodeCount).OrderByDescending(i => rank[i]).Take(10).ToArray();
        var topByCount = Enumerable.Range(0, cora.NodeCount).OrderByDescending(i => citations[i]).Take(10).ToHashSet();
        var promoted = topByRank.Where(i => !topByCount.Contains(i)).ToArray();

        // Each topic's share of all papers against its share of total PageRank.
        var topics = cora.Classes.Select(c => c.Replace('_', ' ')).ToArray();
        var paperShare = topics.Select((_, k) => cora.NodeLabels.Count(l => l == k) / (double)cora.NodeCount).ToArray();
        var rankShare = topics.Select((_, k) => Enumerable.Range(0, cora.NodeCount).Where(i => cora.NodeLabels[i] == k).Sum(i => rank[i]) / rank.Sum()).ToArray();

        var figure = GalleryCharts.Figure(10.5, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        left.Scatter(citations.Select(c => c + 1).ToArray(), rank, "papers", sizes: [10], alpha: 0.5);
        left.Scatter(promoted.Select(i => citations[i] + 1).ToArray(), promoted.Select(i => rank[i]).ToArray(),
                     "top 10 by PageRank, not by count", sizes: [70], edgeColor: GalleryCharts.Of(GraviLibrary.GraviGraph), edgeWidth: 1.5);
        left.SetXScale("log").SetYScale("log")
            .SetXLim(0.7, (citations.Max() + 1) * 1.5).SetYLim(rank.Min() * 0.7, rank.Max() * 1.8);
        left.SetTitle("Citations against PageRank").SetXLabel("times cited + 1").SetYLabel("PageRank");
        left.Legend(LegendLocation.UpperLeft);

        var positions = Enumerable.Range(0, topics.Length).Select(i => (double)i).ToArray();
        right.BarH(positions.Select(p => p + 0.2).ToArray(), paperShare.Select(v => v * 100).ToArray(), 0.4, label: "share of papers");
        right.BarH(positions.Select(p => p - 0.2).ToArray(), rankShare.Select(v => v * 100).ToArray(), 0.4, label: "share of PageRank");
        right.SetYTicks(positions, topics);
        right.SetTitle("Influence by topic").SetXLabel("percent");
        right.Legend(LegendLocation.CenterRight);
        output.Chart(figure);

        var leader = topByRank[0];
        var mostOver = Enumerable.Range(0, topics.Length).MaxBy(k => rankShare[k] / paperShare[k]);
        output.Metric("Papers / citations", $"{cora.NodeCount:N0} / {cora.EdgeCount:N0}");
        output.Metric("Rank agreement (Spearman ρ)", agreement.Statistic.ToString("F2"), "PageRank vs citation count over all papers");
        output.Metric("Top paper", $"#{cora.NodeNames[leader]}", $"{topics[cora.NodeLabels[leader]]}, cited {citations[leader]:F0} times");
        output.Metric("Promoted into the top 10", promoted.Length.ToString(), "by who cites them, not how often");
        output.Line($"{topics[mostOver]} papers hold {rankShare[mostOver] * 100:F1}% of the influence with {paperShare[mostOver] * 100:F1}% " +
                    "of the papers: their citations come from papers that are themselves well cited.");
    }
}
