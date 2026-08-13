using Gravicode.Science.GraviGraph;
using Gravicode.Science.GraviGraph.Algorithms;
using Gravicode.Science.GraviGraph.Embeddings;
using Gravicode.Science.GraviGraph.Neural;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.GraviGraph;

public class GraphStructureTests
{
    [Fact]
    public void AddEdge_GrowsTheNodeSetAndCountsEdgesOnce()
    {
        var graph = new Graph();
        graph.AddEdge(0, 1).AddEdge(1, 2);

        Assert.Equal(3, graph.NodeCount);
        Assert.Equal(2, graph.EdgeCount);
    }

    [Fact]
    public void UndirectedEdges_AreVisibleFromBothEnds()
    {
        var graph = new Graph();
        graph.AddEdge(0, 1);

        Assert.True(graph.HasEdge(0, 1));
        Assert.True(graph.HasEdge(1, 0));
        Assert.Equal(1, graph.Degree(0));
        Assert.Equal(1, graph.Degree(1));
    }

    [Fact]
    public void DirectedEdges_AreOneWay()
    {
        var graph = new Graph(directed: true);
        graph.AddEdge(0, 1);

        Assert.True(graph.HasEdge(0, 1));
        Assert.False(graph.HasEdge(1, 0));
        Assert.Equal(1, graph.OutDegree(0));
        Assert.Equal(0, graph.OutDegree(1));
        Assert.Equal(1, graph.InDegree(1));
    }

    [Fact]
    public void DenseAndSparseAdjacencyAgree()
    {
        var graph = Graph.Random(30, 0.2, seed: 3);
        Assert.True(UFunc.AllClose(graph.ToDenseAdjacency(), graph.ToSparseAdjacency().ToDense(), 1e-12));
    }

    [Fact]
    public void NormalizedAdjacency_HasRowsThatSumToRoughlyOne()
    {
        var graph = Graph.Complete(5);
        var propagation = graph.ToSparseAdjacency(addSelfLoops: true, symmetricNormalize: true);
        var rowSums = propagation.RowSums();

        // A regular graph is normalised to exactly one; the check is that the scale is right.
        for (var i = 0; i < 5; i++) Assert.Equal(1.0, rowSums.At(i), 9);
    }

    [Fact]
    public void Laplacian_HasZeroRowSums()
    {
        var laplacian = Graph.Cycle(6).Laplacian();
        for (var i = 0; i < 6; i++)
        {
            var sum = 0.0;
            for (var j = 0; j < 6; j++) sum += laplacian[i, j];
            Assert.Equal(0.0, sum, 9);
        }
    }

    [Fact]
    public void Density_MatchesTheDefinition()
    {
        Assert.Equal(1.0, Graph.Complete(5).Density, 9);
        Assert.Equal(0.0, new Graph(5).Density, 9);
    }

    [Fact]
    public void Subgraph_KeepsOnlyInternalEdges()
    {
        var graph = new Graph();
        graph.AddEdge(0, 1).AddEdge(1, 2).AddEdge(2, 3);

        var sub = graph.Subgraph([1, 2]);
        Assert.Equal(2, sub.NodeCount);
        Assert.Equal(1, sub.EdgeCount);
        Assert.True(sub.HasEdge(0, 1));
    }

    [Fact]
    public void ScaleFree_ProducesASkewedDegreeDistribution()
    {
        var graph = Graph.ScaleFree(200, edgesPerNode: 2, seed: 5);
        var degrees = graph.Degrees();

        Assert.Equal(200, graph.NodeCount);
        // Preferential attachment creates hubs, so the max degree far exceeds the mean.
        Assert.True(Statistics.Max(degrees) > 4 * Statistics.Mean(degrees));
    }

    [Fact]
    public void Communities_LabelNodesByTheirBlock()
    {
        var graph = Graph.Communities(3, 20, seed: 7);
        Assert.Equal(60, graph.NodeCount);
        Assert.Equal(3, graph.Classes.Count);
        Assert.Equal(0, graph.NodeLabels[0]);
        Assert.Equal(2, graph.NodeLabels[59]);
    }

    [Fact]
    public void DenseAdjacency_RefusesToAllocateAnAbsurdMatrix()
    {
        var graph = new Graph(20_000);
        Assert.Throws<InvalidOperationException>(() => graph.ToDenseAdjacency());
    }

    [Fact]
    public void JsonRoundTrip_PreservesStructureLabelsAndFeatures()
    {
        var graph = Graph.Communities(2, 10, seed: 9);
        graph.NodeFeatures = new GraviRandom(9).Integers(0, 2, 20, 6);

        var path = Path.Combine(Path.GetTempPath(), $"gravi-{Guid.NewGuid():N}.json");
        try
        {
            graph.Save(path, "test", "round trip");
            var loaded = Graph.Load(path);

            Assert.Equal(graph.NodeCount, loaded.NodeCount);
            Assert.Equal(graph.EdgeCount, loaded.EdgeCount);
            Assert.Equal(graph.NodeLabels, loaded.NodeLabels);
            Assert.True(UFunc.AllClose(graph.NodeFeatures!, loaded.NodeFeatures!, 1e-12));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }
}

public class AlgorithmTests
{
    private static Graph PathGraph(int nodes)
    {
        var graph = new Graph();
        for (var i = 0; i + 1 < nodes; i++) graph.AddEdge(i, i + 1);
        return graph;
    }

    [Fact]
    public void BreadthFirstSearch_VisitsInDistanceOrder()
    {
        var order = GraphAlgorithms.BreadthFirstSearch(PathGraph(5), 0);
        Assert.Equal([0, 1, 2, 3, 4], order);
    }

    [Fact]
    public void HopDistances_MarkUnreachableNodes()
    {
        var graph = PathGraph(4);
        graph.AddNode();   // isolated

        var distances = GraphAlgorithms.HopDistances(graph, 0);
        Assert.Equal(0, distances[0]);
        Assert.Equal(3, distances[3]);
        Assert.Equal(-1, distances[4]);
    }

    [Fact]
    public void Dijkstra_FindsTheCheapestPathNotTheShortestHopCount()
    {
        var graph = new Graph();
        graph.AddEdge(0, 1, 10.0);
        graph.AddEdge(0, 2, 1.0);
        graph.AddEdge(2, 1, 1.0);

        var (distances, _) = GraphAlgorithms.ShortestPaths(graph, 0);
        Assert.Equal(2.0, distances[1], 9);
        Assert.Equal([0, 2, 1], GraphAlgorithms.ShortestPath(graph, 0, 1));
    }

    [Fact]
    public void Dijkstra_RejectsNegativeWeights()
    {
        var graph = new Graph();
        graph.AddEdge(0, 1, -5.0);
        Assert.Throws<InvalidOperationException>(() => GraphAlgorithms.ShortestPaths(graph, 0));
    }

    [Fact]
    public void PageRank_SumsToOne()
    {
        var graph = Graph.ScaleFree(120, seed: 11);
        var rank = GraphAlgorithms.PageRank(graph);
        Assert.Equal(1.0, rank.Sum(), 8);
    }

    [Fact]
    public void PageRank_IsUniformOnASymmetricRing()
    {
        var rank = GraphAlgorithms.PageRank(Graph.Cycle(8));
        for (var i = 0; i < 8; i++) Assert.Equal(0.125, rank.At(i), 8);
    }

    [Fact]
    public void PageRank_RanksTheHubHighest()
    {
        // A star: node 0 is cited by everyone else.
        var graph = new Graph(directed: true);
        for (var i = 1; i <= 6; i++) graph.AddEdge(i, 0);

        var rank = GraphAlgorithms.PageRank(graph);
        var top = Statistics.ArgMax(rank);
        Assert.Equal(0, top);
    }

    [Fact]
    public void PageRank_HandlesDanglingNodesWithoutLosingMass()
    {
        var graph = new Graph(directed: true);
        graph.AddEdge(0, 1);
        graph.AddEdge(1, 2);
        graph.AddNode();   // dangling: no outgoing edges

        var rank = GraphAlgorithms.PageRank(graph);
        Assert.Equal(1.0, rank.Sum(), 8);
    }

    [Fact]
    public void PersonalizedPageRank_FavoursTheSeedNeighbourhood()
    {
        var graph = Graph.Communities(2, 15, internalP: 0.5, externalP: 0.01, seed: 13);
        var rank = GraphAlgorithms.PersonalizedPageRank(graph, [0]);

        var firstCommunity = Enumerable.Range(0, 15).Sum(i => rank.At(i));
        var secondCommunity = Enumerable.Range(15, 15).Sum(i => rank.At(i));
        Assert.True(firstCommunity > secondCommunity);
    }

    [Fact]
    public void DegreeCentrality_IsOneForEveryNodeOfACompleteGraph()
    {
        var centrality = GraphAlgorithms.DegreeCentrality(Graph.Complete(6));
        for (var i = 0; i < 6; i++) Assert.Equal(1.0, centrality.At(i), 9);
    }

    [Fact]
    public void BetweennessCentrality_PeaksAtTheMiddleOfAPath()
    {
        var centrality = GraphAlgorithms.BetweennessCentrality(PathGraph(5));
        Assert.Equal(2, Statistics.ArgMax(centrality));
        Assert.Equal(0.0, centrality.At(0), 9);
    }

    [Fact]
    public void ClosenessCentrality_PeaksAtTheCentreOfAStar()
    {
        var graph = new Graph();
        for (var i = 1; i <= 5; i++) graph.AddEdge(0, i);

        var centrality = GraphAlgorithms.ClosenessCentrality(graph);
        Assert.Equal(0, Statistics.ArgMax(centrality));
    }

    [Fact]
    public void EigenvectorCentrality_IsUniformOnARing()
    {
        var centrality = GraphAlgorithms.EigenvectorCentrality(Graph.Cycle(7));
        for (var i = 1; i < 7; i++) Assert.Equal(centrality.At(0), centrality.At(i), 6);
    }

    [Fact]
    public void ConnectedComponents_AreWeakOnADirectedGraph()
    {
        // A one-way chain: following only out-edges from node 2 reaches nothing, so a forward-only
        // traversal would report three components instead of one.
        var graph = new Graph(directed: true);
        graph.AddEdge(0, 1).AddEdge(1, 2);

        var (count, component) = GraphAlgorithms.ConnectedComponents(graph);
        Assert.Equal(1, count);
        Assert.Equal(component[0], component[2]);
    }

    [Fact]
    public void StronglyConnectedComponents_RequireMutualReachability()
    {
        var graph = new Graph(directed: true);
        graph.AddEdge(0, 1).AddEdge(1, 0);   // a mutual pair
        graph.AddEdge(1, 2);                 // one-way out to node 2

        var (count, component) = GraphAlgorithms.StronglyConnectedComponents(graph);
        Assert.Equal(2, count);
        Assert.Equal(component[0], component[1]);
        Assert.NotEqual(component[0], component[2]);
    }

    [Fact]
    public void AsUndirected_MakesEveryEdgeTraversableBothWays()
    {
        var directed = new Graph(directed: true);
        directed.AddEdge(0, 1).AddEdge(1, 2);
        directed.NodeLabels[2] = 7;

        var undirected = directed.AsUndirected();

        Assert.False(undirected.Directed);
        Assert.Equal(directed.NodeCount, undirected.NodeCount);
        Assert.Equal(directed.EdgeCount, undirected.EdgeCount);
        Assert.True(undirected.HasEdge(1, 0));
        Assert.Equal(7, undirected.NodeLabels[2]);
        Assert.Equal(3, GraphAlgorithms.BreadthFirstSearch(undirected, 2).Count);
    }

    [Fact]
    public void ConnectedComponents_SeparatesDisjointPieces()
    {
        var graph = new Graph();
        graph.AddEdge(0, 1);
        graph.AddEdge(2, 3);
        graph.AddNode();

        var (count, component) = GraphAlgorithms.ConnectedComponents(graph);
        Assert.Equal(3, count);
        Assert.Equal(component[0], component[1]);
        Assert.NotEqual(component[0], component[2]);
    }

    [Fact]
    public void TriangleCounts_AreExactOnKnownShapes()
    {
        Assert.Equal([1, 1, 1], GraphAlgorithms.TriangleCounts(Graph.Complete(3)));
        Assert.Equal([0, 0, 0, 0], GraphAlgorithms.TriangleCounts(Graph.Cycle(4)));
    }

    [Fact]
    public void ClusteringCoefficient_IsOneForACliqueAndZeroForATree()
    {
        Assert.Equal(1.0, GraphAlgorithms.AverageClusteringCoefficient(Graph.Complete(5)), 9);
        Assert.Equal(0.0, GraphAlgorithms.AverageClusteringCoefficient(PathGraph(5)), 9);
    }

    [Fact]
    public void TopologicalSort_OrdersADagAndRejectsACycle()
    {
        var dag = new Graph(directed: true);
        dag.AddEdge(0, 1).AddEdge(1, 2).AddEdge(0, 2);
        Assert.Equal([0, 1, 2], GraphAlgorithms.TopologicalSort(dag));

        var cyclic = new Graph(directed: true);
        cyclic.AddEdge(0, 1).AddEdge(1, 0);
        Assert.Empty(GraphAlgorithms.TopologicalSort(cyclic));
    }

    [Fact]
    public void LabelPropagation_RecoversPlantedCommunities()
    {
        var graph = Graph.Communities(3, 25, internalP: 0.5, externalP: 0.01, seed: 17);
        var communities = GraphAlgorithms.LabelPropagation(graph, seed: 17);

        // Nodes inside a planted community should share a discovered label.
        var agreement = 0;
        for (var i = 0; i < graph.NodeCount; i++)
            for (var j = i + 1; j < graph.NodeCount; j++)
            {
                var samePlanted = graph.NodeLabels[i] == graph.NodeLabels[j];
                var sameFound = communities[i] == communities[j];
                if (samePlanted == sameFound) agreement++;
            }

        var pairs = graph.NodeCount * (graph.NodeCount - 1) / 2;
        Assert.True((double)agreement / pairs > 0.9);
    }

    [Fact]
    public void Modularity_IsPositiveForAGoodPartition()
    {
        var graph = Graph.Communities(3, 20, internalP: 0.5, externalP: 0.01, seed: 19);
        Assert.True(GraphAlgorithms.Modularity(graph, graph.NodeLabels) > 0.3);
    }
}

public class EmbeddingTests
{
    [Fact]
    public void RandomWalks_ProduceWalksOfTheRequestedShape()
    {
        var graph = Graph.ScaleFree(50, seed: 21);
        var walks = RandomWalks.Uniform(graph, walksPerNode: 3, walkLength: 10, seed: 21);

        Assert.Equal(150, walks.Count);
        Assert.All(walks, w => Assert.True(w.Count <= 10 && w.Count >= 1));
    }

    [Fact]
    public void BiasedWalks_StayValidForAnyPandQ()
    {
        var graph = Graph.Communities(2, 15, seed: 23);
        foreach (var (p, q) in new[] { (1.0, 1.0), (0.25, 4.0), (4.0, 0.25) })
        {
            var walks = RandomWalks.Biased(graph, p, q, walksPerNode: 2, walkLength: 8, seed: 23);
            Assert.Equal(60, walks.Count);
            foreach (var walk in walks)
                foreach (var node in walk)
                    Assert.InRange(int.Parse(node), 0, graph.NodeCount - 1);
        }
    }

    [Fact]
    public void DeepWalk_PlacesSameCommunityNodesCloserThanCrossCommunityOnes()
    {
        var graph = Graph.Communities(2, 25, internalP: 0.5, externalP: 0.005, seed: 27);
        var embeddings = new DeepWalk(dimensions: 32, walksPerNode: 10, walkLength: 20, epochs: 5, seed: 27)
            .Train(graph);

        Assert.Equal(50, embeddings.NodeCount);
        Assert.Equal(32, embeddings.Dimensions);

        double within = 0, across = 0;
        int withinCount = 0, acrossCount = 0;
        for (var i = 0; i < 50; i++)
            for (var j = i + 1; j < 50; j++)
            {
                var score = embeddings.Similarity(i, j);
                if (graph.NodeLabels[i] == graph.NodeLabels[j]) { within += score; withinCount++; }
                else { across += score; acrossCount++; }
            }

        Assert.True(within / withinCount > across / acrossCount,
            $"within {within / withinCount:F3} should beat across {across / acrossCount:F3}");
    }

    [Fact]
    public void Node2Vec_ProducesAnEmbeddingForEveryNode()
    {
        var graph = Graph.Communities(2, 20, seed: 29);
        var embeddings = new Node2Vec(dimensions: 24, p: 1.0, q: 0.5, walksPerNode: 6, walkLength: 15,
            epochs: 3, seed: 29).Train(graph);

        Assert.Equal(40, embeddings.NodeCount);
        Assert.Equal(24, embeddings.Dimensions);
        Assert.Equal(5, embeddings.MostSimilar(0, top: 5).Count);
    }
}

public class GnnTests
{
    /// <summary>
    /// A community graph whose node features are a noisy one-hot of the community, so a GNN that
    /// works at all should reach high accuracy while a feature-blind model cannot.
    /// </summary>
    private static (Graph Graph, int[] Train, int[] Test) CommunityTask(int seed = 31)
    {
        var graph = Graph.Communities(3, 30, internalP: 0.35, externalP: 0.02, seed: seed);
        var rng = new GraviRandom(seed);

        var features = NdArray.Zeros(graph.NodeCount, 6);
        for (var i = 0; i < graph.NodeCount; i++)
        {
            for (var j = 0; j < 6; j++) features[i, j] = rng.Normal(0, 0.6);
            features[i, graph.NodeLabels[i]] += 1.0;
        }
        graph.NodeFeatures = features;

        var order = rng.Permutation(graph.NodeCount);
        return (graph, order.Take(45).ToArray(), order.Skip(45).ToArray());
    }

    [Fact]
    public void Gcn_TrainsAndGeneralisesToHeldOutNodes()
    {
        var (graph, train, test) = CommunityTask();
        var model = new GraphConvolutionalNetwork(hiddenSize: 16, learningRate: 0.03, epochs: 120, seed: 31)
            .Train(graph, train, test);

        Assert.True(model.IsTrained);
        Assert.Equal(3, model.ClassCount);

        var accuracy = model.Score(graph, test);
        Assert.True(accuracy > 0.8, $"test accuracy = {accuracy:F3}");
    }

    [Fact]
    public void Gcn_LossFallsOverTraining()
    {
        var (graph, train, _) = CommunityTask();
        var model = new GraphConvolutionalNetwork(epochs: 80, learningRate: 0.03, seed: 31).Train(graph, train);

        var history = model.History!;
        Assert.Equal(80, history.Loss.Count);
        Assert.True(history.Loss[^1] < history.Loss[0], $"{history.Loss[0]:F4} -> {history.Loss[^1]:F4}");
    }

    [Fact]
    public void Gcn_ProbabilitiesAreNormalised()
    {
        var (graph, train, _) = CommunityTask();
        var model = new GraphConvolutionalNetwork(epochs: 40, seed: 31).Train(graph, train);

        var probabilities = model.PredictProbabilities();
        for (var i = 0; i < graph.NodeCount; i++)
        {
            var total = 0.0;
            for (var c = 0; c < model.ClassCount; c++) total += probabilities[i, c];
            Assert.Equal(1.0, total, 8);
        }
    }

    [Fact]
    public void Gcn_ExposesAHiddenEmbedding()
    {
        var (graph, train, _) = CommunityTask();
        var model = new GraphConvolutionalNetwork(hiddenSize: 12, epochs: 30, seed: 31).Train(graph, train);

        var embedding = model.NodeEmbeddings();
        Assert.Equal(graph.NodeCount, embedding.Shape[0]);
        Assert.Equal(12, embedding.Shape[1]);
    }

    [Fact]
    public void Gcn_RefusesToPredictBeforeTraining()
    {
        var model = new GraphConvolutionalNetwork();
        Assert.Throws<InvalidOperationException>(() => model.Predict());
    }

    [Fact]
    public void Gcn_RequiresNodeFeatures()
    {
        var graph = Graph.Communities(2, 10, seed: 33);
        Assert.Throws<ArgumentException>(() => new GraphConvolutionalNetwork().Train(graph));
    }

    [Fact]
    public void GraphSage_TrainsAndGeneralises()
    {
        var (graph, train, test) = CommunityTask(seed: 35);
        var model = new GraphSage(hiddenSize: 16, learningRate: 0.03, epochs: 120, seed: 35).Train(graph, train, test);

        var accuracy = model.Score(graph, test);
        Assert.True(accuracy > 0.8, $"test accuracy = {accuracy:F3}");
        Assert.True(model.History!.Loss[^1] < model.History.Loss[0]);
    }

    [Fact]
    public void GraphSage_ClassifiesAGraphItNeverSaw()
    {
        var (graph, train, _) = CommunityTask(seed: 37);
        var model = new GraphSage(hiddenSize: 16, learningRate: 0.03, epochs: 120, seed: 37).Train(graph, train);

        // A fresh graph drawn from the same distribution: transductive models cannot do this.
        var (unseen, _, _) = CommunityTask(seed: 137);
        var predictions = model.PredictInductive(unseen, unseen.NodeFeatures!);

        var correct = predictions.Where((p, i) => p == unseen.NodeLabels[i]).Count();
        Assert.Equal(unseen.NodeCount, predictions.Length);
        Assert.True((double)correct / unseen.NodeCount > 0.5,
            $"inductive accuracy = {(double)correct / unseen.NodeCount:F3}");
    }

    [Fact]
    public void Gat_TrainsAndGeneralises()
    {
        var (graph, train, test) = CommunityTask(seed: 39);
        var model = new GraphAttentionNetwork(hiddenSize: 8, heads: 4, learningRate: 0.03, epochs: 80, seed: 39)
            .Train(graph, train, test);

        var accuracy = model.Score(graph, test);
        Assert.True(accuracy > 0.75, $"test accuracy = {accuracy:F3}");
        Assert.True(model.History!.Loss[^1] < model.History.Loss[0]);
    }

    [Fact]
    public void Gat_AttentionCoefficientsFormADistributionPerNode()
    {
        var (graph, train, _) = CommunityTask(seed: 41);
        var model = new GraphAttentionNetwork(hiddenSize: 8, heads: 2, epochs: 10, seed: 41).Train(graph, train);

        var head = model.AttentionWeights[0];
        for (var node = 0; node < graph.NodeCount; node++)
        {
            var total = head.Where(kv => kv.Key.Source == node).Sum(kv => kv.Value);
            Assert.Equal(1.0, total, 8);
        }
    }

    [Fact]
    public void AllThreeArchitecturesLearnTheSameTask()
    {
        var (graph, train, test) = CommunityTask(seed: 43);

        var gcn = new GraphConvolutionalNetwork(epochs: 100, learningRate: 0.03, seed: 43).Train(graph, train);
        var sage = new GraphSage(epochs: 100, learningRate: 0.03, seed: 43).Train(graph, train);
        var gat = new GraphAttentionNetwork(epochs: 60, learningRate: 0.03, seed: 43).Train(graph, train);

        // A majority-class baseline scores about a third on three balanced communities.
        Assert.True(gcn.Score(graph, test) > 0.6);
        Assert.True(sage.Score(graph, test) > 0.6);
        Assert.True(gat.Score(graph, test) > 0.6);
    }
}

public class CoraDatasetTests
{
    private static string? CoraPath()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        for (var depth = 0; directory is not null && depth < 12; depth++)
        {
            var candidate = Path.Combine(directory.FullName, "datasets", "cora_graph.json");
            if (File.Exists(candidate)) return candidate;
            directory = directory.Parent;
        }
        return null;
    }

    [Fact]
    public void Cora_LoadsWithTheExpectedShape()
    {
        var path = CoraPath();
        Assert.NotNull(path);

        var graph = Graph.Load(path!);
        Assert.Equal(2708, graph.NodeCount);
        Assert.Equal(5429, graph.EdgeCount);
        Assert.Equal(7, graph.Classes.Count);
        Assert.Equal(1433, graph.NodeFeatures!.Shape[1]);
    }

    [Fact]
    public void Cora_HasTheKnownLargestConnectedComponent()
    {
        var graph = Graph.Load(CoraPath()!);
        var (_, component) = GraphAlgorithms.ConnectedComponents(graph);
        var largest = component.GroupBy(c => c).Max(g => g.Count());

        // The published figure for Cora's largest weakly connected component is 2485 of 2708.
        Assert.Equal(2485, largest);
    }

    [Fact]
    public void Cora_PageRankIdentifiesHighlyCitedPapers()
    {
        var graph = Graph.Load(CoraPath()!);
        var rank = GraphAlgorithms.PageRank(graph);

        Assert.Equal(1.0, rank.Sum(), 6);
        // The most central paper should clearly outrank the median one.
        Assert.True(Statistics.Max(rank) > 10 * Statistics.Median(rank));
    }

    [Fact]
    public void Cora_GcnBeatsTheMajorityClassBaseline()
    {
        var graph = Graph.Load(CoraPath()!);
        var rng = new GraviRandom(47);
        var order = rng.Permutation(graph.NodeCount);
        var train = order.Take(140).ToArray();          // the standard Cora split: 20 per class
        var test = order.Skip(1708).ToArray();

        var model = new GraphConvolutionalNetwork(hiddenSize: 16, learningRate: 0.05, epochs: 60, seed: 47)
            .Train(graph, train, features: graph.NodeFeatures);

        // The largest Cora class covers about 30% of nodes; published GCN accuracy is around 0.81.
        var accuracy = model.Score(graph, test);
        Assert.True(accuracy > 0.55, $"Cora GCN test accuracy = {accuracy:F3}");
    }
}
