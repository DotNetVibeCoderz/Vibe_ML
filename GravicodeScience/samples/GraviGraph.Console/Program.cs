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

// ---------------------------------------------------------------- v0.4: heterogeneous
Section("10. Heterogeneous graphs");

var shop = new HeterogeneousGraph();
shop.AddNodeType("user", 4);
shop.AddNodeType("item", 5);

shop.AddEdge("user", "viewed", "item", 0, 0);
shop.AddEdge("user", "viewed", "item", 0, 1);
shop.AddEdge("user", "viewed", "item", 1, 1);
shop.AddEdge("user", "viewed", "item", 2, 3);
shop.AddEdge("user", "bought", "item", 1, 2, weight: 1.0);
shop.AddEdge("user", "bought", "item", 3, 4, weight: 1.0);

Console.WriteLine($"  {shop}");
Console.WriteLine("  Node indices are LOCAL to their type - user 0 and item 0 are different nodes,");
Console.WriteLine("  which is what lets each type carry a different feature width:");

shop.SetFeatures("user", new GraviRandom(3).StandardNormal(4, 6));
shop.SetFeatures("item", new GraviRandom(5).StandardNormal(5, 3));
Console.WriteLine($"    user features are {shop.Features("user")!.Shape[1]} wide, " +
                  $"item features are {shop.Features("item")!.Shape[1]} wide");

// Rating scores and timestamps: two numbers per edge, which a scalar weight cannot hold.
var bought = new EdgeType("user", "bought", "item");
shop.SetEdgeFeatures(bought, NdArray.FromArray(new double[,] { { 5, 1710 }, { 3, 1840 } }));
Console.WriteLine($"    edge features on '{bought}': {shop.EdgeFeatures(bought)!.Shape[0]} edges x " +
                  $"{shop.EdgeFeatures(bought)!.Shape[1]} attributes");

shop.AddReverseEdges(new EdgeType("user", "viewed", "item"));
Console.WriteLine("  Messages only flow along edge direction, so items could never inform users.");
Console.WriteLine($"    added '{new EdgeType("item", "rev_viewed", "user")}' as a SEPARATE relation -");
Console.WriteLine("    'user views item' and 'item is viewed by user' deserve different weights");

var rgcn = new RelationalConvolution(shop,
    new Dictionary<string, int> { ["user"] = 6, ["item"] = 3 }, outputSize: 8, new GraviRandom(7));

var messages = rgcn.Forward(shop, new Dictionary<string, NdArray>
{
    ["user"] = shop.Features("user")!,
    ["item"] = shop.Features("item")!,
});

Console.WriteLine($"  R-GCN layer -> user [{Shapes.Describe(messages["user"].Shape)}], " +
                  $"item [{Shapes.Describe(messages["item"].Shape)}]");
Console.WriteLine("  One weight matrix per relation, and in-degree normalised PER RELATION - a user");
Console.WriteLine("  with a thousand views and three purchases would otherwise lose the purchases,");
Console.WriteLine("  and the purchases are the signal.");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: temporal
Section("11. Temporal graphs");

var events = new TemporalGraph();
events.AddEdge(0, 1, time: 1);
events.AddEdge(1, 2, time: 2);
events.AddEdge(2, 3, time: 0);      // happened BEFORE anything arrived at node 2

Console.WriteLine("  Three events: 0->1 at t=1, 1->2 at t=2, 2->3 at t=0.");
Console.WriteLine($"  Statically, 0 reaches 3 through 1 and 2: edge 2->3 exists = {events.Collapse().HasEdge(2, 3)}");

var reachable = events.TemporallyReachable(0);
Console.WriteLine($"  Temporally, 0 reaches: [{string.Join(", ", reachable.Keys.Order())}]");
Console.WriteLine("    node 3 is NOT reachable - the 2->3 edge fired before anything got to node 2,");
Console.WriteLine("    so nothing could have travelled that way. Information, money and disease all");
Console.WriteLine("    obey that ordering, and a static analysis overstates every one of them.");
Console.WriteLine($"  TemporalEfficiency = {events.TemporalEfficiency():F3} (1.0 would mean ordering never mattered)");

// A recency-weighted embedding: recent interactions should say more than old ones.
var decayGraph = new TemporalGraph(directed: false);
decayGraph.AddEdge(0, 1, time: 0);        // old
decayGraph.AddEdge(0, 2, time: 100);      // recent

var signals = NdArray.Zeros(3, 1);
signals[1, 0] = 1.0;       // what the old neighbour says
signals[2, 0] = -1.0;      // what the recent neighbour says

var decayed = decayGraph.TimeDecayedFeatures(signals, asOf: 100, halfLife: 10);
Console.WriteLine($"  Time-decayed embedding of node 0 = {decayed[0, 0]:F4}");
Console.WriteLine("    negative, so the recent neighbour won. A static aggregation weights a");
Console.WriteLine("    year-old interaction the same as yesterday's, which is why collapsed-graph");
Console.WriteLine("    recommenders keep suggesting what somebody liked once, long ago.");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: classification
Section("12. Graph-level pooling and classification");

var pooled = NdArray.FromArray(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
Console.WriteLine("  A readout turns a variable number of node vectors into one graph vector.");
Console.WriteLine($"    Mean    = [{string.Join(", ", GraphPooling.Pool(pooled, PoolingKind.Mean).ToArray())}]");
Console.WriteLine($"    Sum     = [{string.Join(", ", GraphPooling.Pool(pooled, PoolingKind.Sum).ToArray())}]");
Console.WriteLine($"    Max     = [{string.Join(", ", GraphPooling.Pool(pooled, PoolingKind.Max).ToArray())}]");
Console.WriteLine("  Every one is a symmetric aggregate, because graph nodes have no canonical");
Console.WriteLine("  numbering - a readout sensitive to order would depend on how the file was written.");
Console.WriteLine("  Mean is size-invariant (judge composition); sum is not (size itself matters).");

var shapes = new List<Graph>();
var shapeLabels = new List<int>();
for (var n = 5; n <= 12; n++)
{
    shapes.Add(Graph.Cycle(n));
    shapeLabels.Add(0);
    shapes.Add(Graph.Complete(n));
    shapeLabels.Add(1);
}

var shapeClassifier = new GraphClassifier(inputSize: 2, hiddenSize: 16, layers: 2)
    .Fit(shapes, shapeLabels);

Console.WriteLine($"  Trained to tell cycles from complete graphs: {shapeClassifier.Accuracy(shapes, shapeLabels):P1}");
Console.WriteLine($"    a 20-node cycle    -> class {shapeClassifier.Predict(Graph.Cycle(20))} (expected 0)");
Console.WriteLine($"    a 20-node complete -> class {shapeClassifier.Predict(Graph.Complete(20))} (expected 1)");
Console.WriteLine("    both sizes are outside the training range, so it generalised on structure");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: sampling
Section("13. Neighbourhood sampling");

// A hub with two thousand neighbours: full-batch message passing would pull all of them in.
var hub = new Graph(2001);
for (var i = 1; i <= 2000; i++) hub.AddEdge(0, i);

var block = NeighborSampler.Sample(hub, [0], [5], new GraviRandom(3));
Console.WriteLine($"  Node 0 has {hub.Degree(0)} neighbours. Sampling with fan-out 5:");
Console.WriteLine($"    the batch touches {block.NodeCount} nodes and {block.EdgeCount} edges");
Console.WriteLine("  The problem is not memory, it is neighbourhood explosion: a two-layer GNN on a");
Console.WriteLine("  graph of average degree 100 touches 10,000 nodes per target, three layers a million.");

var citation = Graph.Random(500, 0.02, seed: 11);
var citationFeatures = new GraviRandom(13).StandardNormal(500, 8);

var batchBlock = NeighborSampler.Sample(citation, [0, 1, 2], [10, 5], new GraviRandom(17));
var gathered = NeighborSampler.GatherFeatures(batchBlock, citationFeatures);

Console.WriteLine($"  On a 500-node graph, a 3-target batch with fan-out [10, 5]:");
Console.WriteLine($"    {batchBlock.NodeCount} of 500 feature rows need to be resident");
Console.WriteLine($"    layers: {string.Join(", ", batchBlock.Layers.Select(l => l.Length + " edges"))}");

var sageRng = new GraviRandom(19);
NdArray[] sageWeights = [sageRng.StandardNormal(16, 12) * 0.1, sageRng.StandardNormal(24, 4) * 0.1];
var sageOut = NeighborSampler.Aggregate(batchBlock, gathered, sageWeights);

Console.WriteLine($"    aggregated -> [{Shapes.Describe(sageOut.Shape)}], one row per target");
Console.WriteLine("  A SAGE layer takes TWICE its feature width: self and neighbourhood are");
Console.WriteLine("  concatenated rather than averaged together, which is what lets the model tell a");
Console.WriteLine("  node apart from its surroundings.");
Console.WriteLine();

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
