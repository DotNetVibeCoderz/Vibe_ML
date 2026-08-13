using System.Diagnostics;
using Gravicode.Science.GraviGraph;
using Gravicode.Science.GraviGraph.Algorithms;
using Gravicode.Science.GraviGraph.Embeddings;
using Gravicode.Science.GraviGraph.Neural;
using Gravicode.Science.GraviNum;

Console.WriteLine(GraviInfo.Banner("GraviGraph"));

var datasets = Resolve("datasets") ?? throw new DirectoryNotFoundException("datasets directory not found.");
var screenshots = ResolveScreenshots();
var watch = new Stopwatch();

// ---------------------------------------------------------------- load
Section("1. The Cora citation network");

watch.Start();
var cora = Graph.Load(Path.Combine(datasets, "cora_graph.json"));
watch.Stop();

Console.WriteLine($"  {cora}");
Console.WriteLine($"  loaded in {watch.ElapsedMilliseconds} ms");
Console.WriteLine($"  node features: {cora.NodeFeatures!.Shape[0]} x {cora.NodeFeatures.Shape[1]} bag-of-words");
Console.WriteLine($"  classes      : {string.Join(", ", cora.Classes)}");

var degrees = cora.Degrees();
Console.WriteLine($"  degree       : mean {Statistics.Mean(degrees):F2}, median {Statistics.Median(degrees):F0}, max {Statistics.Max(degrees):F0}");

var labelCounts = cora.NodeLabels.GroupBy(l => l).OrderBy(g => g.Key);
foreach (var group in labelCounts)
    Console.WriteLine($"    {cora.Classes[group.Key],-24}{group.Count(),5} papers");
Console.WriteLine();

// ---------------------------------------------------------------- pagerank
Section("2. PageRank");

watch.Restart();
var rank = GraphAlgorithms.PageRank(cora);
watch.Stop();

Console.WriteLine($"  computed in {watch.ElapsedMilliseconds} ms, total mass {rank.Sum():F6}");
Console.WriteLine($"  {"rank",5}{"node",8}{"score",12}{"degree",8}  topic");
Console.WriteLine("  " + new string('-', 62));

var ranked = Enumerable.Range(0, cora.NodeCount)
    .OrderByDescending(i => rank.At(i))
    .Take(10)
    .ToArray();

for (var i = 0; i < ranked.Length; i++)
{
    var node = ranked[i];
    Console.WriteLine($"  {i + 1,5}{node,8}{rank.At(node),12:F6}{cora.Degree(node),8}  {cora.Classes[cora.NodeLabels[node]]}");
}
Console.WriteLine();

Console.WriteLine("  Personalised PageRank seeded on the top paper stays inside its topic:");
var personalised = GraphAlgorithms.PersonalizedPageRank(cora, [ranked[0]]);
var topicOfSeed = cora.NodeLabels[ranked[0]];
var neighbourhood = Enumerable.Range(0, cora.NodeCount)
    .OrderByDescending(i => personalised.At(i))
    .Take(20)
    .Count(i => cora.NodeLabels[i] == topicOfSeed);
Console.WriteLine($"    {neighbourhood} of the top 20 share the seed's topic ({cora.Classes[topicOfSeed]})");
Console.WriteLine();

// ---------------------------------------------------------------- centrality
Section("3. Centrality measures");

var smaller = cora.Subgraph(Enumerable.Range(0, 600).ToArray());
Console.WriteLine($"  Working on a {smaller.NodeCount}-node subgraph (betweenness is O(VE)).");

foreach (var (name, compute) in new (string, Func<Graph, NdArray>)[]
         {
             ("degree", GraphAlgorithms.DegreeCentrality),
             ("closeness", GraphAlgorithms.ClosenessCentrality),
             ("betweenness", g => GraphAlgorithms.BetweennessCentrality(g)),
             ("eigenvector", g => GraphAlgorithms.EigenvectorCentrality(g)),
         })
{
    watch.Restart();
    var centrality = compute(smaller);
    watch.Stop();
    Console.WriteLine($"    {name,-14}top node {Statistics.ArgMax(centrality),5}   " +
                      $"max {Statistics.Max(centrality):F5}   {watch.ElapsedMilliseconds,5} ms");
}
Console.WriteLine();

// ---------------------------------------------------------------- structure
Section("4. Structure");

var (components, componentOf) = GraphAlgorithms.ConnectedComponents(cora);
var largest = componentOf.GroupBy(c => c).Max(g => g.Count());
Console.WriteLine($"  weakly connected      : {components} components (largest holds {largest} nodes, {(double)largest / cora.NodeCount:P1})");

var (strong, _) = GraphAlgorithms.StronglyConnectedComponents(cora);
Console.WriteLine($"  strongly connected    : {strong} components (citations rarely point both ways)");
Console.WriteLine($"  average clustering    : {GraphAlgorithms.AverageClusteringCoefficient(cora):F4}");
Console.WriteLine($"  triangles             : {GraphAlgorithms.TriangleCounts(cora).Sum() / 3}");

var shortest = GraphAlgorithms.ShortestPath(cora, ranked[0], ranked[1]);
Console.WriteLine($"  path {ranked[0]} -> {ranked[1]}: {(shortest.Count == 0 ? "no path" : string.Join(" -> ", shortest))}");
Console.WriteLine();

// ---------------------------------------------------------------- communities
Section("5. Community detection");

watch.Restart();
var communities = GraphAlgorithms.LabelPropagation(cora, seed: 42);
watch.Stop();

var found = communities.Distinct().Count();
Console.WriteLine($"  label propagation found {found} communities in {watch.ElapsedMilliseconds} ms");
Console.WriteLine($"  modularity of the discovered partition: {GraphAlgorithms.Modularity(cora, communities):F4}");
Console.WriteLine($"  modularity of the true topic labels   : {GraphAlgorithms.Modularity(cora, cora.NodeLabels):F4}");
Console.WriteLine();

// ---------------------------------------------------------------- embeddings
Section("6. Node embeddings");

// Random walks need to move both ways along a citation, so the undirected view is used.
var (_, coraComponent) = GraphAlgorithms.ConnectedComponents(cora);
var biggestComponent = coraComponent.Select((c, i) => (Component: c, Node: i))
    .GroupBy(t => t.Component)
    .MaxBy(g => g.Count())!
    .Select(t => t.Node)
    .Take(600)
    .ToArray();

var sample = cora.Subgraph(biggestComponent).AsUndirected();
Console.WriteLine($"  working on the largest component: {sample.NodeCount} nodes, {sample.EdgeCount} edges");

watch.Restart();
var deepWalk = new DeepWalk(dimensions: 64, walksPerNode: 6, walkLength: 20, epochs: 3, seed: 42).Train(sample);
watch.Stop();
Console.WriteLine($"  DeepWalk : {deepWalk.NodeCount} x {deepWalk.Dimensions} in {watch.ElapsedMilliseconds} ms");

watch.Restart();
var node2vec = new Node2Vec(dimensions: 64, p: 1.0, q: 0.5, walksPerNode: 6, walkLength: 20, epochs: 3, seed: 42)
    .Train(sample);
watch.Stop();
Console.WriteLine($"  node2vec : {node2vec.NodeCount} x {node2vec.Dimensions} in {watch.ElapsedMilliseconds} ms (q=0.5 explores outward)");

double SameTopicAdvantage(NodeEmbeddings embedding)
{
    double same = 0, different = 0;
    int sameCount = 0, differentCount = 0;
    for (var i = 0; i < Math.Min(250, embedding.NodeCount); i++)
        for (var j = i + 1; j < Math.Min(250, embedding.NodeCount); j++)
        {
            var score = embedding.Similarity(i, j);
            if (sample.NodeLabels[i] == sample.NodeLabels[j]) { same += score; sameCount++; }
            else { different += score; differentCount++; }
        }
    return same / sameCount - different / differentCount;
}

Console.WriteLine($"  same-topic cosine advantage: DeepWalk {SameTopicAdvantage(deepWalk):F4}, node2vec {SameTopicAdvantage(node2vec):F4}");
Console.WriteLine("  (positive means papers on the same topic really do end up closer)");
Console.WriteLine();

// ---------------------------------------------------------------- gnn
Section("7. Graph neural networks");

var rng = new GraviRandom(42);
var order = rng.Permutation(cora.NodeCount);
var train = order.Take(140).ToArray();            // 20 labelled papers per class
var validation = order.Skip(140).Take(500).ToArray();
var test = order.Skip(1708).ToArray();

Console.WriteLine($"  semi-supervised split: {train.Length} labelled, {validation.Length} validation, {test.Length} test");
Console.WriteLine($"  {"model",-26}{"train",10}{"test",10}{"time",10}");
Console.WriteLine("  " + new string('-', 56));

watch.Restart();
var gcn = new GraphConvolutionalNetwork(hiddenSize: 16, learningRate: 0.05, epochs: 60, seed: 42)
    .Train(cora, train, validation);
watch.Stop();
Console.WriteLine($"  {"GCN",-26}{gcn.Score(cora, train),10:P1}{gcn.Score(cora, test),10:P1}{watch.ElapsedMilliseconds,8} ms");

watch.Restart();
var sage = new GraphSage(hiddenSize: 16, learningRate: 0.05, epochs: 60, seed: 42).Train(cora, train, validation);
watch.Stop();
Console.WriteLine($"  {"GraphSAGE",-26}{sage.Score(cora, train),10:P1}{sage.Score(cora, test),10:P1}{watch.ElapsedMilliseconds,8} ms");

Console.WriteLine();
Console.WriteLine($"  A majority-class baseline would score {cora.NodeLabels.GroupBy(l => l).Max(g => g.Count()) / (double)cora.NodeCount:P1}.");
Console.WriteLine($"  GCN loss fell from {gcn.History!.Loss[0]:F4} to {gcn.History.Loss[^1]:F4} over {gcn.History.Loss.Count} epochs.");
Console.WriteLine();

// ---------------------------------------------------------------- chart
Section("8. Network visualisation");

// A small, dense subgraph laid out by a force-directed simulation.
var core = Enumerable.Range(0, cora.NodeCount)
    .OrderByDescending(i => cora.Degree(i))
    .Take(120)
    .ToArray();
var view = cora.Subgraph(core);
var positions = ForceDirectedLayout(view, seed: 42);

var plot = new ScottPlot.Plot();
foreach (var edge in view.Edges())
{
    var line = plot.Add.Line(positions[edge.Source].X, positions[edge.Source].Y,
        positions[edge.Target].X, positions[edge.Target].Y);
    line.LineWidth = 0.6f;
    line.Color = ScottPlot.Colors.Gray.WithAlpha(0.35);
}

foreach (var group in Enumerable.Range(0, view.NodeCount).GroupBy(i => view.NodeLabels[i]))
{
    var xs = group.Select(i => positions[i].X).ToArray();
    var ys = group.Select(i => positions[i].Y).ToArray();
    var scatter = plot.Add.ScatterPoints(xs, ys);
    scatter.MarkerSize = 9;
    scatter.LegendText = cora.Classes[group.Key];
}

plot.Title("GraviGraph - Cora, 120 highest-degree papers coloured by topic");
plot.ShowLegend();
plot.HideGrid();
plot.SavePng(Path.Combine(screenshots, "gravigraph_network.png"), 1100, 850);
Console.WriteLine($"  saved {Path.Combine(screenshots, "gravigraph_network.png")}");

Console.WriteLine();
Console.WriteLine(GraviInfo.Attribution);
return;

/// <summary>
/// A Fruchterman-Reingold layout: edges pull connected nodes together, every pair pushes apart,
/// and the maximum step size cools over time so the layout settles instead of oscillating.
/// </summary>
static (double X, double Y)[] ForceDirectedLayout(Graph graph, int iterations = 260, int seed = 42)
{
    var rng = new GraviRandom(seed);
    var n = graph.NodeCount;
    var position = new (double X, double Y)[n];
    for (var i = 0; i < n; i++) position[i] = (rng.Uniform(-1, 1), rng.Uniform(-1, 1));

    var k = Math.Sqrt(1.0 / n);
    var temperature = 0.15;

    for (var step = 0; step < iterations; step++)
    {
        var displacement = new (double X, double Y)[n];

        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
            {
                var dx = position[i].X - position[j].X;
                var dy = position[i].Y - position[j].Y;
                var distance = Math.Max(Math.Sqrt(dx * dx + dy * dy), 1e-4);
                var repulsion = k * k / distance;

                displacement[i] = (displacement[i].X + dx / distance * repulsion,
                                   displacement[i].Y + dy / distance * repulsion);
                displacement[j] = (displacement[j].X - dx / distance * repulsion,
                                   displacement[j].Y - dy / distance * repulsion);
            }

        foreach (var edge in graph.Edges())
        {
            var dx = position[edge.Source].X - position[edge.Target].X;
            var dy = position[edge.Source].Y - position[edge.Target].Y;
            var distance = Math.Max(Math.Sqrt(dx * dx + dy * dy), 1e-4);
            var attraction = distance * distance / k;

            displacement[edge.Source] = (displacement[edge.Source].X - dx / distance * attraction,
                                         displacement[edge.Source].Y - dy / distance * attraction);
            displacement[edge.Target] = (displacement[edge.Target].X + dx / distance * attraction,
                                         displacement[edge.Target].Y + dy / distance * attraction);
        }

        for (var i = 0; i < n; i++)
        {
            var magnitude = Math.Max(Math.Sqrt(displacement[i].X * displacement[i].X + displacement[i].Y * displacement[i].Y), 1e-9);
            var limited = Math.Min(magnitude, temperature);
            position[i] = (position[i].X + displacement[i].X / magnitude * limited,
                           position[i].Y + displacement[i].Y / magnitude * limited);
        }
        temperature *= 0.985;
    }
    return position;
}

static void Section(string title)
    => Console.WriteLine($"--- {title} " + new string('-', Math.Max(0, 60 - title.Length)));

static string? Resolve(string relative)
{
    var directory = new DirectoryInfo(AppContext.BaseDirectory);
    for (var depth = 0; directory is not null && depth < 12; depth++)
    {
        var candidate = Path.Combine(directory.FullName, relative);
        if (Directory.Exists(candidate)) return candidate;
        directory = directory.Parent;
    }
    return null;
}

static string ResolveScreenshots()
{
    var found = Resolve(Path.Combine("docs", "screenshots"));
    if (found is not null) return found;
    var fallback = Path.Combine(Environment.CurrentDirectory, "screenshots");
    Directory.CreateDirectory(fallback);
    return fallback;
}
