# GraviGraph

*[Bahasa Indonesia](id/GraviGraph.md)* · Graph ML for .NET — structures, algorithms, node embeddings and GNNs.

## Graph structures

The primary representation is an adjacency list, because graph work is overwhelmingly
neighbourhood-local: PageRank, BFS and message passing all iterate a node's neighbours, which
costs `O(degree)` here and `O(nodes)` on a dense matrix. Real networks are sparse — Cora has 2,708
nodes and 5,429 edges, so a dense matrix would be 99.9% zeros.

```csharp
using Gravicode.Science.GraviGraph;

var g = new Graph(directed: false);
g.AddNode("paper-1", label: 0);
g.AddEdge(0, 1, weight: 2.5);

g.NodeCount;  g.EdgeCount;  g.Density;
g.Neighbors(0);  g.Predecessors(0);
g.Degree(0);  g.InDegree(0);  g.OutDegree(0);  g.WeightedDegree(0);
g.HasEdge(0, 1);  g.Edges();  g.Degrees();

g.NodeFeatures = matrix;      // (nodes, features)
g.NodeLabels;  g.NodeNames;  g.Classes;
```

### Matrix views

```csharp
g.ToSparseAdjacency();                                            // CSR
g.ToSparseAdjacency(addSelfLoops: true, symmetricNormalize: true); // the GCN propagation matrix
g.ToDenseAdjacency();                                             // guarded — throws above ~50M entries
g.Laplacian();                                                    // D - A
```

### Transformations and generators

```csharp
g.Subgraph([1, 5, 9]);        // induced, renumbered from zero
g.AsUndirected();             // every edge traversable both ways

Graph.Random(nodes: 100, p: 0.05, seed: 42);
Graph.ScaleFree(nodes: 1000, edgesPerNode: 3);   // preferential attachment — power-law degrees
Graph.Communities(communities: 3, sizePerCommunity: 50, internalP: 0.4, externalP: 0.01);
Graph.Cycle(10);  Graph.Complete(10);
```

### Loading and saving

```csharp
var cora = Graph.Load("datasets/cora_graph.json");
g.Save("graph.json", name: "mine", description: "...");
Graph.LoadEdgeList("edges.txt", directed: false);
```

Node features are stored as the **indices of the non-zero entries**. Cora's features are
1,433-dimensional but average about eighteen non-zeros, so the sparse form is roughly eighty times
smaller and loads correspondingly faster.

## Algorithms

```csharp
using Gravicode.Science.GraviGraph.Algorithms;

GraphAlgorithms.BreadthFirstSearch(g, start);
GraphAlgorithms.DepthFirstSearch(g, start);
GraphAlgorithms.HopDistances(g, start);              // -1 for unreachable

GraphAlgorithms.ShortestPaths(g, start);             // Dijkstra, binary heap
GraphAlgorithms.ShortestPath(g, start, end);

GraphAlgorithms.PageRank(g, damping: 0.85);
GraphAlgorithms.PersonalizedPageRank(g, seeds: [42]);

GraphAlgorithms.DegreeCentrality(g);
GraphAlgorithms.ClosenessCentrality(g);
GraphAlgorithms.BetweennessCentrality(g);            // Brandes, O(VE)
GraphAlgorithms.EigenvectorCentrality(g);

GraphAlgorithms.ConnectedComponents(g);              // weak on a directed graph
GraphAlgorithms.StronglyConnectedComponents(g);      // Kosaraju
GraphAlgorithms.TriangleCounts(g);
GraphAlgorithms.ClusteringCoefficients(g);
GraphAlgorithms.TopologicalSort(dag);                // empty when a cycle exists
GraphAlgorithms.LabelPropagation(g);
GraphAlgorithms.Modularity(g, communities);
```

### Two things worth knowing

**PageRank redistributes dangling mass.** A node with no outgoing edges would otherwise leak
probability out of the system and stop the scores summing to one. That detail is the difference
between a correct implementation and one whose scores quietly shrink.

**Connectivity on a directed graph means *weak* connectivity.** `ConnectedComponents` follows
edges in both directions. Following only out-edges would split Cora into more than 1,500
fragments simply because citations point one way; with both directions the largest component holds
**2,485 of 2,708 nodes**, which is the published figure. Use `StronglyConnectedComponents` when
mutual reachability is what you actually want.

`BetweennessCentrality` is `O(VE)` even with Brandes' algorithm — an algorithmic limit, not an
implementation one. Restrict it to a subgraph on large networks.

## Node embeddings

```csharp
using Gravicode.Science.GraviGraph.Embeddings;

var deepWalk = new DeepWalk(dimensions: 128, walksPerNode: 10, walkLength: 80, epochs: 5)
    .Train(graph.AsUndirected());

var node2vec = new Node2Vec(dimensions: 128, p: 1.0, q: 0.5, walksPerNode: 10)
    .Train(graph.AsUndirected());

embeddings.Similarity(a, b);
embeddings.MostSimilar(node, top: 10);
embeddings.LinkScore(source, target);      // link prediction
embeddings.Save("nodes.vec");

RandomWalks.Uniform(g, walksPerNode: 10, walkLength: 80);
RandomWalks.Biased(g, p: 1.0, q: 0.5, walksPerNode: 10, walkLength: 80);
```

DeepWalk's insight is that node sequences from random walks have the same power-law statistics as
natural language, so a word embedding model applies directly. node2vec adds a second-order bias:

| | Effect |
|---|---|
| `q < 1` | Walks push outward — depth-first — and embeddings capture **communities** |
| `q > 1` | Walks stay local — breadth-first — and embeddings capture **structural roles** |
| `p` | How eagerly the walk backtracks |

**Use `AsUndirected()` first on a citation or follower graph.** Walking only along the direction of
citation strands most walks after a step or two.

## Graph neural networks

```csharp
using Gravicode.Science.GraviGraph.Neural;

var gcn = new GraphConvolutionalNetwork(hiddenSize: 16, learningRate: 0.01, epochs: 200,
        dropout: 0.5, weightDecay: 5e-4, seed: 42)
    .Train(graph, trainMask, validationMask);

gcn.Predict();  gcn.PredictProbabilities();  gcn.NodeEmbeddings();
gcn.Score(graph, testMask);
gcn.History;    // loss and accuracy per epoch

new GraphSage(hiddenSize: 16, epochs: 200).Train(graph, trainMask);
new GraphAttentionNetwork(hiddenSize: 8, heads: 4, epochs: 200).Train(graph, trainMask);
```

All three are **fully trained**, and all three get their gradients from the
[autodiff tape](GraviNum.md#automatic-differentiation): each layer is written forwards and the
backward pass is derived from it. GAT's attention parameters are genuinely learned, including
through the softmax over each node's incident edges — the one derivation in this library that was
hardest to do by hand, and is now not done by hand at all.

### Writing your own layer

The tape is what makes a fourth architecture cheap: write the forward pass and the gradient
follows.

```csharp
using Gravicode.Science.GraviGraph.Neural;
using Gravicode.Science.GraviNum.Autodiff;

var propagation = graph.ToSparseAdjacency(addSelfLoops: true, symmetricNormalize: true);

var w = Tensor.Parameter(GnnMath.Glorot(features, classes, rng));
var adam = new TapeAdam(w, weightDecay: 5e-4);

for (var epoch = 0; epoch < 200; epoch++)
{
    var logits = GnnTape.Convolve(propagation, Tensor.Constant(x), w, bias);
    var loss = TensorOps.SoftmaxCrossEntropy(logits, graph.NodeLabels, trainMask);

    loss.Backward();      // no derivation anywhere
    adam.Step(0.01);
}
```

`GnnTape` supplies `Convolve`, `SageLayer`, `AttentionLayer`, `SegmentSoftmax`, `Dropout`,
`MeanAggregator`, `EdgeList` and `Descend`. Underneath them sit the graph-shaped tape operations:
`SparseMatMul` for message passing, `Gather` and `SegmentSum` for edge-level work, plus
`ConcatColumns`, `LeakyRelu` and a masked `SoftmaxCrossEntropy`. Check any new gradient with
`GradientCheck` before trusting it.

`Gather` and `SegmentSum` are adjoints of one another — gathering forwards is summing backwards —
which is why attention needs no bespoke kernel. `SegmentSoftmax` is *composed* from them rather
than written as its own operation, so it inherits their verified gradients instead of needing the
softmax Jacobian derived again.

> **A hand-derived gradient was wrong for the entire life of the GCN.** The backward pass omitted
> the dropout mask on the hidden layer, so with dropout active the gradient was about **40%**
> off — and the model still trained to a plausible 71% on Cora, which is exactly why it went
> unnoticed. The tape found it immediately, because a forward pass has nowhere to hide a missing
> term. Test accuracy with the corrected gradient is 69.3%; the older, higher number came from
> a broken gradient that happened to act as an odd regulariser.

### How they differ

| | Neighbour weighting | Setting |
|---|---|---|
| **GCN** | Fixed: `1/sqrt(d_i d_j)` — structure only | Transductive |
| **GraphSAGE** | Learned separately for self and neighbourhood mean | **Inductive** |
| **GAT** | Learned from content: `softmax(LeakyReLU(a·[Wh_i ‖ Wh_j]))` | Transductive |

GraphSAGE keeps `[h_self ; mean(h_neighbours)]` in separate halves of the layer input, which lets
it weight "what I am" against "what surrounds me" independently — and because the aggregator is
defined per node rather than over a fixed normalised adjacency matrix, the same weights apply to
nodes never seen during training:

```csharp
var predictions = sage.PredictInductive(unseenGraph, unseenFeatures);
```

Two layers is the usual depth. Each layer mixes in one more hop, and beyond about three hops every
node's representation converges toward the graph average — the over-smoothing problem.

Training is transductive for GCN and GAT: the whole graph is seen every epoch, but the loss is
computed only on the labelled nodes in `trainMask`. On Cora with 140 labelled papers (20 per class)
and 60 epochs, against a 30.2% majority baseline:

| | Test accuracy |
|---|---:|
| GCN | 69.3% |
| GraphSAGE | 70.3% |
| **GAT** | **72.1%** |

The published GCN figure is ~81%. The remaining gap is the absent learning-rate schedule, no early
stopping, and 60 epochs — not the gradient, which is now checked against finite differences.
Attention earning its keep over a fixed degree normalisation is the expected ordering, and it is
reassuring to see it after the move to the tape rather than before.

## Common mistakes

| Symptom | Cause |
|---|---|
| Thousands of "components" on a citation graph | Expected for strong connectivity — use `ConnectedComponents` for weak |
| Random walks return almost immediately | The graph is directed; call `AsUndirected()` |
| `ToDenseAdjacency` throws | The graph is too large — use `ToSparseAdjacency` |
| GNN training does nothing | The graph has no `NodeFeatures`; pass them explicitly |

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
