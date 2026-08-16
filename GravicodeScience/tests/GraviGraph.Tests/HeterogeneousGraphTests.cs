using Gravicode.Science.GraviGraph;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.GraviGraph;

/// <summary>
/// Tests for typed graphs, temporal graphs, graph classification and neighbourhood sampling.
/// </summary>
/// <remarks>
/// The claims worth pinning are the ones that distinguish these from the homogeneous static case:
/// that per-relation weights actually differ, that temporal reachability is strictly narrower than
/// static reachability, that a readout is permutation-invariant, and that sampling bounds the work
/// per target regardless of how large the graph is.
/// </remarks>
public class HeterogeneousGraphTests
{
    /// <summary>A small bipartite recommendation graph: users, films, and two relations.</summary>
    private static HeterogeneousGraph Recommendations()
    {
        var graph = new HeterogeneousGraph();

        graph.AddNodeType("user", 3);
        graph.AddNodeType("film", 4);

        graph.AddEdge("user", "watched", "film", 0, 0);
        graph.AddEdge("user", "watched", "film", 0, 1);
        graph.AddEdge("user", "watched", "film", 1, 1);
        graph.AddEdge("user", "rated", "film", 1, 2, weight: 5.0);
        graph.AddEdge("user", "rated", "film", 2, 3, weight: 2.0);

        return graph;
    }

    // ------------------------------------------------------- heterogeneous graph

    [Fact]
    public void NodeIndicesAreLocalToTheirType()
    {
        // The property that lets each type have its own feature dimension, and the thing that
        // separates a typed graph from a graph with a label column.
        var graph = Recommendations();

        Assert.Equal(3, graph.CountOf("user"));
        Assert.Equal(4, graph.CountOf("film"));
        Assert.Equal(7, graph.NodeCount);

        // User 0 and film 0 are different nodes despite sharing an index.
        graph.SetFeatures("user", NdArray.Zeros(3, 2));
        graph.SetFeatures("film", NdArray.Zeros(4, 5));

        Assert.Equal(2, graph.Features("user")!.Shape[1]);
        Assert.Equal(5, graph.Features("film")!.Shape[1]);
    }

    [Fact]
    public void RelationsWithTheSameNameButDifferentEndpointsAreDistinct()
    {
        // (user, rates, film) and (critic, rates, film) are different relations that share a verb,
        // and conflating them means learning one set of weights for two behaviours.
        var graph = new HeterogeneousGraph();
        graph.AddEdge("user", "rates", "film", 0, 0);
        graph.AddEdge("critic", "rates", "film", 0, 0);

        Assert.Equal(2, graph.EdgeTypes.Count);
        Assert.Single(graph.Edges(new EdgeType("user", "rates", "film")));
        Assert.Single(graph.Edges(new EdgeType("critic", "rates", "film")));
    }

    [Fact]
    public void EdgeFeaturesCarryWhatAWeightCannot()
    {
        var graph = Recommendations();
        var rated = new EdgeType("user", "rated", "film");

        // Two numbers per edge — a score and a timestamp — which no scalar weight can hold.
        graph.SetEdgeFeatures(rated, NdArray.FromArray(new double[,] { { 5, 100 }, { 2, 200 } }));

        Assert.Equal([2, 2], graph.EdgeFeatures(rated)!.Shape.ToArray());
        Assert.Equal(100.0, graph.EdgeFeatures(rated)![0, 1]);
    }

    [Fact]
    public void AnEdgeFeatureMatrixMustMatchTheEdgeCount()
    {
        var graph = Recommendations();
        Assert.Throws<ArgumentException>(() => graph.SetEdgeFeatures(
            new EdgeType("user", "rated", "film"), NdArray.Zeros(9, 2)));
    }

    [Fact]
    public void ReverseEdgesAreASeparateRelation()
    {
        // Message passing only moves along edge direction, so without the reverse relation films
        // could never inform users. It is separate rather than symmetric because "user rates film"
        // and "film is rated by user" deserve different weights.
        var graph = Recommendations();
        graph.AddReverseEdges(new EdgeType("user", "watched", "film"));

        var reverse = new EdgeType("film", "rev_watched", "user");

        Assert.Equal(3, graph.Edges(reverse).Count);
        Assert.Contains(graph.Edges(reverse), e => e is { Source: 0, Target: 0 });
        Assert.NotEqual(graph.Edges(new EdgeType("user", "watched", "film")).Count
                        + graph.Edges(reverse).Count, graph.Edges(reverse).Count);
    }

    [Fact]
    public void FlatteningOffsetsEachTypeIntoItsOwnBlock()
    {
        var graph = Recommendations();
        var flat = graph.ToHomogeneous(out var offsets);

        Assert.Equal(7, flat.NodeCount);
        Assert.Equal(5, flat.EdgeCount);

        // Distinct blocks, so no two typed nodes collide.
        Assert.NotEqual(offsets["user"], offsets["film"]);
        Assert.True(flat.HasEdge(offsets["user"] + 0, offsets["film"] + 0));
    }

    [Fact]
    public void RelationalConvolutionLearnsSeparateWeightsPerRelation()
    {
        // The whole argument for R-GCN: a "watched" edge and a "rated" edge get different
        // transformations, which a homogeneous GCN cannot express.
        var graph = Recommendations();
        graph.SetFeatures("user", NdArray.Ones(3, 4));
        graph.SetFeatures("film", NdArray.Ones(4, 3));

        var sizes = new Dictionary<string, int> { ["user"] = 4, ["film"] = 3 };
        var layer = new RelationalConvolution(graph, sizes, outputSize: 6, new GraviRandom(5));

        var watched = layer.Weights(new EdgeType("user", "watched", "film"));
        var rated = layer.Weights(new EdgeType("user", "rated", "film"));

        var identical = true;
        for (var i = 0; i < watched.Size && identical; i++)
            identical = Math.Abs(watched.At(i) - rated.At(i)) < 1e-12;

        Assert.False(identical, "the two relations share a weight matrix");
    }

    [Fact]
    public void RelationalConvolutionProducesOneRepresentationPerNode()
    {
        var graph = Recommendations();
        graph.SetFeatures("user", NdArray.Ones(3, 4));
        graph.SetFeatures("film", NdArray.Ones(4, 3));

        var sizes = new Dictionary<string, int> { ["user"] = 4, ["film"] = 3 };
        var layer = new RelationalConvolution(graph, sizes, outputSize: 6, new GraviRandom(7));

        var inputs = new Dictionary<string, NdArray>
        {
            ["user"] = graph.Features("user")!,
            ["film"] = graph.Features("film")!,
        };

        var outputs = layer.Forward(graph, inputs);

        Assert.Equal([3, 6], outputs["user"].Shape.ToArray());
        Assert.Equal([4, 6], outputs["film"].Shape.ToArray());
    }

    [Fact]
    public void AnIsolatedNodeKeepsItsOwnFeaturesThroughTheSelfLoop()
    {
        // Without a self-loop an isolated node's representation is exactly zero, and the node
        // becomes indistinguishable from every other isolated one.
        var graph = new HeterogeneousGraph();
        graph.AddNodeType("a", 2);
        graph.AddEdge("a", "links", "a", 0, 0);

        var features = NdArray.Ones(2, 3);
        graph.SetFeatures("a", features);

        var layer = new RelationalConvolution(graph,
            new Dictionary<string, int> { ["a"] = 3 }, outputSize: 4, new GraviRandom(11));

        var outputs = layer.Forward(graph, new Dictionary<string, NdArray> { ["a"] = features });

        var allZero = true;
        for (var d = 0; d < 4 && allZero; d++) allZero = outputs["a"][1, d] == 0.0;

        Assert.False(allZero, "the isolated node's representation collapsed to zero");
    }

    // ---------------------------------------------------------- temporal graph

    /// <summary>0→1 at t=1, 1→2 at t=2, and 2→3 at t=0 — the ordering that breaks static paths.</summary>
    private static TemporalGraph Ordered()
    {
        var graph = new TemporalGraph();
        graph.AddEdge(0, 1, time: 1);
        graph.AddEdge(1, 2, time: 2);
        graph.AddEdge(2, 3, time: 0);
        return graph;
    }

    [Fact]
    public void TemporalReachabilityRespectsEdgeOrdering()
    {
        // The claim that makes timestamps worth keeping. Statically 0 reaches 3 through 1 and 2;
        // temporally it cannot, because the 2→3 edge happened before anything arrived at 2.
        var graph = Ordered();
        var reachable = graph.TemporallyReachable(0);

        Assert.Contains(1, reachable.Keys);
        Assert.Contains(2, reachable.Keys);
        Assert.DoesNotContain(3, reachable.Keys);

        // And the static view does reach it, so the test is not vacuous.
        Assert.True(graph.Collapse().HasEdge(2, 3));
    }

    [Fact]
    public void TemporalReachabilityRecordsTheEarliestArrival()
    {
        var graph = new TemporalGraph();
        graph.AddEdge(0, 1, time: 5);
        graph.AddEdge(0, 1, time: 2);
        graph.AddEdge(1, 2, time: 3);

        var reachable = graph.TemporallyReachable(0);

        Assert.Equal(2.0, reachable[1]);
        // Arriving at 1 at t=2 makes the t=3 edge usable; had only the t=5 edge existed it would not.
        Assert.Equal(3.0, reachable[2]);
    }

    [Fact]
    public void AMaximumGapBreaksLongPauses()
    {
        var graph = new TemporalGraph();
        graph.AddEdge(0, 1, time: 0);
        graph.AddEdge(1, 2, time: 100);

        Assert.Contains(2, graph.TemporallyReachable(0).Keys);
        Assert.DoesNotContain(2, graph.TemporallyReachable(0, maxGap: 10).Keys);
    }

    [Fact]
    public void TemporalEfficiencyIsBelowOneWhenOrderingMatters()
    {
        // A single number for how much a static analysis overstates reachability.
        var efficiency = Ordered().TemporalEfficiency();

        Assert.True(efficiency is > 0 and < 1,
            $"efficiency came out at {efficiency:F4}, so ordering apparently made no difference");
    }

    [Fact]
    public void SnapshotsSelectTheEdgesInTheirWindow()
    {
        var graph = Ordered();

        Assert.Equal(1, graph.Snapshot(0, 0).EdgeCount);
        Assert.Equal(2, graph.Snapshot(0, 1).EdgeCount);
        Assert.Equal(3, graph.Collapse().EdgeCount);
    }

    [Fact]
    public void SnapshotUpToDoesNotIncludeLaterEdges()
    {
        // The cut that stops a link predictor from being trained on its own test set.
        var graph = Ordered();
        Assert.Equal(2, graph.SnapshotUpTo(1).EdgeCount);
    }

    [Fact]
    public void WindowsCoverEveryEdgeExactlyOnce()
    {
        var graph = new TemporalGraph();
        for (var t = 0; t < 10; t++) graph.AddEdge(t, t + 1, time: t);

        var windows = graph.Windows(4);
        Assert.Equal(4, windows.Count);

        // The last window's bound is inclusive, or the final edge would be dropped.
        Assert.Equal(10, windows.Sum(w => w.EdgeCount));
    }

    [Fact]
    public void TimeDecayedFeaturesFavourRecentInteractions()
    {
        // The reason a collapsed graph keeps recommending what someone liked once, long ago.
        var graph = new TemporalGraph(directed: false);
        graph.AddEdge(0, 1, time: 0);       // old
        graph.AddEdge(0, 2, time: 100);     // recent

        var features = NdArray.Zeros(3, 1);
        features[1, 0] = 1.0;      // the old neighbour's signal
        features[2, 0] = -1.0;     // the recent neighbour's

        var decayed = graph.TimeDecayedFeatures(features, asOf: 100, halfLife: 10);

        Assert.True(decayed[0, 0] < 0,
            $"node 0 came out at {decayed[0, 0]:F4}, so the old interaction still dominates");
    }

    [Fact]
    public void EdgesComeBackInTimeOrderRegardlessOfInsertionOrder()
    {
        var graph = new TemporalGraph();
        graph.AddEdge(0, 1, time: 5);
        graph.AddEdge(1, 2, time: 1);
        graph.AddEdge(2, 3, time: 3);

        var times = graph.Edges.Select(e => e.Time).ToArray();
        Assert.Equal([1.0, 3.0, 5.0], times);
    }

    // ------------------------------------------------------- graph classification

    [Fact]
    public void PoolingIsInvariantToNodeOrder()
    {
        // Graph nodes have no canonical numbering, so a readout sensitive to permutation makes the
        // model's output depend on how the file happened to be written.
        var features = NdArray.FromArray(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var shuffled = NdArray.FromArray(new double[,] { { 5, 6 }, { 1, 2 }, { 3, 4 } });

        foreach (var kind in new[] { PoolingKind.Mean, PoolingKind.Sum, PoolingKind.Max, PoolingKind.MeanMax })
        {
            var a = GraphPooling.Pool(features, kind);
            var b = GraphPooling.Pool(shuffled, kind);

            for (var d = 0; d < a.Size; d++)
                Assert.Equal(a.At(d), b.At(d), 12);
        }
    }

    [Fact]
    public void PoolingKindsComputeWhatTheyClaim()
    {
        var features = NdArray.FromArray(new double[,] { { 1, 2 }, { 3, 4 } });

        Assert.Equal([2.0, 3.0], GraphPooling.Pool(features, PoolingKind.Mean).ToArray());
        Assert.Equal([4.0, 6.0], GraphPooling.Pool(features, PoolingKind.Sum).ToArray());
        Assert.Equal([3.0, 4.0], GraphPooling.Pool(features, PoolingKind.Max).ToArray());
        Assert.Equal([2.0, 3.0, 3.0, 4.0], GraphPooling.Pool(features, PoolingKind.MeanMax).ToArray());
    }

    [Fact]
    public void MeanIsSizeInvariantAndSumIsNot()
    {
        // The modelling decision the choice encodes: mean judges composition, sum judges magnitude.
        var small = NdArray.FromArray(new double[,] { { 1, 1 } });
        var large = NdArray.FromArray(new double[,] { { 1, 1 }, { 1, 1 }, { 1, 1 } });

        Assert.Equal(GraphPooling.Pool(small, PoolingKind.Mean).ToArray(),
            GraphPooling.Pool(large, PoolingKind.Mean).ToArray());

        Assert.NotEqual(GraphPooling.Pool(small, PoolingKind.Sum).ToArray(),
            GraphPooling.Pool(large, PoolingKind.Sum).ToArray());
    }

    [Fact]
    public void AttentionPoolWeightsSumToOneAndFavourRelevantNodes()
    {
        var features = NdArray.FromArray(new double[,] { { 5, 0 }, { 0, 0 }, { 0, 0 } });
        var (pooled, weights) = GraphPooling.AttentionPool(features, NdArray.FromValues([1.0, 0.0]));

        Assert.Equal(1.0, weights.ToArray().Sum(), 10);
        Assert.True(weights.At(0) > weights.At(1), "the relevant node was not up-weighted");
        Assert.Equal(2, pooled.Size);
    }

    [Fact]
    public void GraphClassifierSeparatesCyclesFromCompleteGraphs()
    {
        // Two families whose structure differs sharply — a cycle has degree 2 everywhere, a
        // complete graph has degree n−1 — so a structural model should separate them cleanly.
        var graphs = new List<Graph>();
        var labels = new List<int>();

        for (var n = 5; n <= 12; n++)
        {
            graphs.Add(Graph.Cycle(n));
            labels.Add(0);

            graphs.Add(Graph.Complete(n));
            labels.Add(1);
        }

        var classifier = new GraphClassifier(inputSize: 2, hiddenSize: 16, layers: 2).Fit(graphs, labels);

        Assert.True(classifier.Accuracy(graphs, labels) > 0.9,
            $"accuracy was {classifier.Accuracy(graphs, labels):P1}");

        // And it generalises to sizes it never saw.
        Assert.Equal(0, classifier.Predict(Graph.Cycle(20)));
        Assert.Equal(1, classifier.Predict(Graph.Complete(20)));
    }

    [Fact]
    public void TheEmbeddingHasTheSameWidthForGraphsOfEverySize()
    {
        // The whole point of a readout: a fixed-size vector from a variable-sized graph.
        var classifier = new GraphClassifier(inputSize: 2, hiddenSize: 8, layers: 2);

        var small = classifier.Embed(Graph.Cycle(4));
        var large = classifier.Embed(Graph.Cycle(400));

        Assert.Equal(small.Size, large.Size);
    }

    [Fact]
    public void ClassifyingBeforeFittingThrows()
    {
        Assert.Throws<InvalidOperationException>(
            () => new GraphClassifier(2).Predict(Graph.Cycle(5)));
    }

    // --------------------------------------------------------- neighbour sampling

    [Fact]
    public void SamplingBoundsTheWorkPerTargetRegardlessOfDegree()
    {
        // The actual problem: neighbourhood explosion. A hub with a thousand neighbours must not
        // pull a thousand nodes into the batch.
        var graph = new Graph(2001);
        for (var i = 1; i <= 2000; i++) graph.AddEdge(0, i);

        var block = NeighborSampler.Sample(graph, [0], [5], new GraviRandom(3));

        Assert.True(block.NodeCount <= 6, $"the block pulled in {block.NodeCount} nodes");
        Assert.True(block.EdgeCount <= 5, $"the block pulled in {block.EdgeCount} edges");
    }

    [Fact]
    public void TheBlockGrowsWithTheFanOutAndTheNumberOfHops()
    {
        var graph = Graph.Random(200, 0.1, seed: 5);

        var shallow = NeighborSampler.Sample(graph, [0], [3], new GraviRandom(7));
        var deep = NeighborSampler.Sample(graph, [0], [3, 3], new GraviRandom(7));

        Assert.True(deep.NodeCount >= shallow.NodeCount,
            "a second hop should not shrink the block");
        Assert.Equal(2, deep.Layers.Count);
    }

    [Fact]
    public void TargetsAreLocatableInTheSampledBlock()
    {
        var graph = Graph.Random(100, 0.1, seed: 11);
        int[] targets = [3, 17, 42];

        var block = NeighborSampler.Sample(graph, targets, [4, 4], new GraviRandom(13));

        Assert.Equal(targets.Length, block.TargetPositions.Length);
        for (var i = 0; i < targets.Length; i++)
            Assert.Equal(targets[i], block.Nodes[block.TargetPositions[i]]);
    }

    [Fact]
    public void SamplingWithoutReplacementNeverRepeatsANeighbour()
    {
        var graph = new Graph(51);
        for (var i = 1; i <= 50; i++) graph.AddEdge(0, i);

        var block = NeighborSampler.Sample(graph, [0], [10], new GraviRandom(17), replace: false);
        var sources = block.Layers[0].Select(e => e.Source).ToArray();

        Assert.Equal(sources.Length, sources.Distinct().Count());
    }

    [Fact]
    public void ANodeWithFewerNeighboursThanTheCapKeepsThemAll()
    {
        var graph = new Graph(4);
        graph.AddEdge(0, 1);
        graph.AddEdge(0, 2);

        var block = NeighborSampler.Sample(graph, [0], [10], new GraviRandom(19));
        Assert.Equal(2, block.Layers[0].Length);
    }

    [Fact]
    public void GatheringFeaturesTouchesOnlyTheBlocksNodes()
    {
        // The memory claim: whatever the graph's size, only these rows need to be resident.
        var graph = Graph.Random(500, 0.02, seed: 23);
        var features = new GraviRandom(29).StandardNormal(500, 8);

        var block = NeighborSampler.Sample(graph, [0, 1], [3, 3], new GraviRandom(31));
        var gathered = NeighborSampler.GatherFeatures(block, features);

        Assert.Equal(block.NodeCount, gathered.Shape[0]);
        Assert.True(gathered.Shape[0] < 500, "the block covered the whole graph, so nothing was saved");

        for (var i = 0; i < block.NodeCount; i++)
            for (var d = 0; d < 8; d++)
                Assert.Equal(features[block.Nodes[i], d], gathered[i, d]);
    }

    [Fact]
    public void AggregationProducesOneRepresentationPerTarget()
    {
        var graph = Graph.Random(200, 0.05, seed: 37);
        var features = new GraviRandom(41).StandardNormal(200, 4);

        int[] targets = [0, 5, 9];
        var block = NeighborSampler.Sample(graph, targets, [4, 4], new GraviRandom(43));
        var gathered = NeighborSampler.GatherFeatures(block, features);

        // A SAGE layer takes twice its input width, because self and neighbourhood are concatenated.
        var rng = new GraviRandom(47);
        NdArray[] weights = [rng.StandardNormal(8, 6) * 0.1, rng.StandardNormal(12, 3) * 0.1];

        var output = NeighborSampler.Aggregate(block, gathered, weights);

        Assert.Equal([targets.Length, 3], output.Shape.ToArray());
    }

    [Fact]
    public void AWeightMatrixOfTheWrongWidthIsRejectedWithAnExplanation()
    {
        // The mistake everyone makes with SAGE: forgetting that the concatenation doubles the width.
        var graph = Graph.Random(50, 0.1, seed: 53);
        var block = NeighborSampler.Sample(graph, [0], [3], new GraviRandom(59));
        var features = new GraviRandom(61).StandardNormal(block.NodeCount, 4);

        var error = Assert.Throws<ArgumentException>(
            () => NeighborSampler.Aggregate(block, features, [NdArray.Zeros(4, 6)]));

        Assert.Contains("twice", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void BatchesCoverEveryNodeExactlyOnce()
    {
        var nodes = Enumerable.Range(0, 97).ToArray();
        var batches = NeighborSampler.Batches(nodes, 10, new GraviRandom(67)).ToArray();

        Assert.Equal(10, batches.Length);
        Assert.Equal(97, batches.Sum(b => b.Length));
        Assert.Equal(nodes, batches.SelectMany(b => b).Order().ToArray());
    }

    [Fact]
    public void MalformedSamplingRequestsAreRejected()
    {
        var graph = Graph.Cycle(10);

        Assert.Throws<ArgumentException>(() => NeighborSampler.Sample(graph, [0], []));
        Assert.Throws<ArgumentOutOfRangeException>(() => NeighborSampler.Sample(graph, [99], [3]));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => NeighborSampler.Batches([1, 2, 3], 0).ToArray());
    }
}
