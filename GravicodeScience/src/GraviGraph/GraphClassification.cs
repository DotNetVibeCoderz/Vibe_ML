using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviGraph;

/// <summary>How node representations are combined into one vector for the whole graph.</summary>
public enum PoolingKind
{
    /// <summary>Average over nodes. Size-invariant, and the safe default.</summary>
    Mean,

    /// <summary>Sum over nodes. Keeps size information, and scales with it.</summary>
    Sum,

    /// <summary>Element-wise maximum. Detects whether a feature is present anywhere.</summary>
    Max,

    /// <summary>Mean and max concatenated, so both signals survive.</summary>
    MeanMax,
}

/// <summary>
/// Readout functions: collapsing a variable-sized set of node vectors into one graph vector.
/// </summary>
/// <remarks>
/// <para>
/// Node classification has one representation per node and needs no readout. Graph classification —
/// is this molecule toxic, is this program malicious — needs a single vector per graph, and graphs
/// have different numbers of nodes, so the readout has to accept any size and produce a fixed one.
/// </para>
/// <para>
/// <b>The readout must not depend on node order.</b> Graph nodes have no canonical numbering, so a
/// readout that is sensitive to permutation makes the model's output depend on how the file happened
/// to be written. Every function here is a symmetric aggregate for exactly that reason, and it is
/// why concatenating node vectors — the obvious way to get a fixed size — is not an option.
/// </para>
/// <para>
/// The choice between them is a real modelling decision. Mean is invariant to graph size, which is
/// right when a large molecule and a small one should be judged on composition; sum is not, which is
/// right when size itself is informative. Max asks whether a feature appears at all, which is what
/// detects a single unusual substructure that an average would dilute away.
/// </para>
/// </remarks>
public static class GraphPooling
{
    /// <summary>Collapses node representations into one graph vector.</summary>
    public static NdArray Pool(NdArray nodeFeatures, PoolingKind kind = PoolingKind.Mean)
    {
        ArgumentNullException.ThrowIfNull(nodeFeatures);
        if (nodeFeatures.Rank != 2)
            throw new ArgumentException("Node features must be a rank 2 array.", nameof(nodeFeatures));

        var (nodes, width) = (nodeFeatures.Shape[0], nodeFeatures.Shape[1]);

        if (nodes == 0)
            return NdArray.Zeros(kind == PoolingKind.MeanMax ? width * 2 : width);

        return kind switch
        {
            PoolingKind.Mean => Aggregate(nodeFeatures, nodes, width, divide: true),
            PoolingKind.Sum => Aggregate(nodeFeatures, nodes, width, divide: false),
            PoolingKind.Max => Maximum(nodeFeatures, nodes, width),
            PoolingKind.MeanMax => Concatenate(
                Aggregate(nodeFeatures, nodes, width, divide: true), Maximum(nodeFeatures, nodes, width)),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };
    }

    /// <summary>
    /// Pools with attention: nodes are weighted by how relevant the model thinks they are.
    /// </summary>
    /// <param name="nodeFeatures">Node representations.</param>
    /// <param name="gate">A vector scoring each node's relevance from its features.</param>
    /// <remarks>
    /// A mean treats every node as equally informative, which for a molecule with one reactive group
    /// and forty inert atoms is plainly false. Attention pooling learns which nodes to listen to, and
    /// the weights are readable afterwards — they say which part of the graph drove the prediction,
    /// which a mean cannot.
    /// </remarks>
    public static (NdArray Pooled, NdArray Weights) AttentionPool(NdArray nodeFeatures, NdArray gate)
    {
        ArgumentNullException.ThrowIfNull(nodeFeatures);
        ArgumentNullException.ThrowIfNull(gate);

        var (nodes, width) = (nodeFeatures.Shape[0], nodeFeatures.Shape[1]);

        if (gate.Size != width)
            throw new ArgumentException(
                $"The gate has {gate.Size} entries but the features are {width} wide.", nameof(gate));

        if (nodes == 0) return (NdArray.Zeros(width), NdArray.Zeros(0));

        var scores = new double[nodes];
        for (var i = 0; i < nodes; i++)
        {
            var score = 0.0;
            for (var d = 0; d < width; d++) score += nodeFeatures[i, d] * gate.At(d);
            scores[i] = score;
        }

        var weights = MathUtil.Softmax(scores);
        var pooled = NdArray.Zeros(width);

        for (var i = 0; i < nodes; i++)
            for (var d = 0; d < width; d++)
                pooled.SetAt(d, pooled.At(d) + weights[i] * nodeFeatures[i, d]);

        return (pooled, NdArray.FromValues(weights));
    }

    private static NdArray Aggregate(NdArray features, int nodes, int width, bool divide)
    {
        var result = NdArray.Zeros(width);

        for (var i = 0; i < nodes; i++)
            for (var d = 0; d < width; d++)
                result.SetAt(d, result.At(d) + features[i, d]);

        if (divide)
            for (var d = 0; d < width; d++) result.SetAt(d, result.At(d) / nodes);

        return result;
    }

    private static NdArray Maximum(NdArray features, int nodes, int width)
    {
        var result = NdArray.Zeros(width);

        for (var d = 0; d < width; d++)
        {
            var best = double.NegativeInfinity;
            for (var i = 0; i < nodes; i++) best = Math.Max(best, features[i, d]);
            result.SetAt(d, best);
        }

        return result;
    }

    private static NdArray Concatenate(NdArray a, NdArray b)
    {
        var result = NdArray.Zeros(a.Size + b.Size);
        for (var i = 0; i < a.Size; i++) result.SetAt(i, a.At(i));
        for (var i = 0; i < b.Size; i++) result.SetAt(a.Size + i, b.At(i));
        return result;
    }
}

/// <summary>
/// Classifies whole graphs: message passing, a readout, then a linear head.
/// </summary>
/// <remarks>
/// <para>
/// The architecture differs from node classification in exactly one place — the readout — and that
/// one change alters what the model is for. A node classifier answers questions about positions in a
/// single large graph; this answers questions about many small graphs, each of which is one example.
/// </para>
/// <para>
/// Message passing here uses fixed random projections rather than learned ones, with only the final
/// classifier trained. That is a real architecture, not a shortcut: it is the graph analogue of a
/// random-features model, it trains in closed form, and it is a genuinely strong baseline — a
/// learned GNN that cannot beat it is not learning anything the structure did not already give away.
/// A fully trained version belongs on the autodiff tape alongside <see cref="GnnTape"/>.
/// </para>
/// </remarks>
public sealed class GraphClassifier
{
    private readonly NdArray[] _propagationWeights;
    private readonly PoolingKind _pooling;
    private NdArray _classifier = NdArray.Zeros(0, 0);
    private double[] _classes = [];

    /// <summary>Creates a classifier.</summary>
    /// <param name="inputSize">Node feature dimension.</param>
    /// <param name="hiddenSize">Width of each message-passing layer.</param>
    /// <param name="layers">
    /// How many rounds of message passing. Each round widens a node's receptive field by one hop;
    /// beyond three or four the representations tend to converge on each other, which is
    /// over-smoothing and shows up as accuracy falling with depth.
    /// </param>
    /// <param name="pooling">How node representations become a graph vector.</param>
    /// <param name="rng">Source of randomness for the projections.</param>
    public GraphClassifier(int inputSize, int hiddenSize = 32, int layers = 2,
        PoolingKind pooling = PoolingKind.MeanMax, GraviRandom? rng = null)
    {
        if (layers < 1) throw new ArgumentOutOfRangeException(nameof(layers));

        rng ??= new GraviRandom(42);
        _pooling = pooling;

        _propagationWeights = new NdArray[layers];
        var width = inputSize;

        for (var i = 0; i < layers; i++)
        {
            var limit = Math.Sqrt(6.0 / (width + hiddenSize));
            var weights = NdArray.Zeros(width, hiddenSize);
            for (var j = 0; j < weights.Size; j++) weights.SetAt(j, (rng.NextDouble() * 2 - 1) * limit);

            _propagationWeights[i] = weights;
            width = hiddenSize;
        }

        HiddenSize = hiddenSize;
    }

    /// <summary>Width of each message-passing layer.</summary>
    public int HiddenSize { get; }

    /// <summary>The class labels seen during training.</summary>
    public IReadOnlyList<double> Classes => _classes;

    /// <summary>True once <see cref="Fit"/> has run.</summary>
    public bool IsFitted { get; private set; }

    /// <summary>
    /// The graph-level representation: message passing followed by the readout.
    /// </summary>
    /// <remarks>
    /// Each layer averages over neighbours <em>and</em> keeps the node's own features, which is the
    /// self-loop every GCN needs — without it a node's own information is discarded at every layer
    /// and only its neighbourhood survives.
    /// </remarks>
    public NdArray Embed(Graph graph, NdArray? nodeFeatures = null)
    {
        ArgumentNullException.ThrowIfNull(graph);

        // With no features, the degree is the only structural signal available, and it is a real
        // one — many graph datasets are classified well from degree alone.
        var x = nodeFeatures ?? DegreeFeatures(graph);

        foreach (var weights in _propagationWeights)
        {
            var projected = LinAlg.Dot(x, weights);
            var next = NdArray.Zeros(graph.NodeCount, HiddenSize);

            for (var node = 0; node < graph.NodeCount; node++)
            {
                var neighbours = graph.Neighbors(node);
                var divisor = neighbours.Count + 1;      // the +1 is the self-loop

                for (var d = 0; d < HiddenSize; d++)
                {
                    var sum = projected[node, d];
                    foreach (var (target, weight) in neighbours) sum += weight * projected[target, d];
                    next[node, d] = Math.Max(0, sum / divisor);
                }
            }

            x = next;
        }

        return GraphPooling.Pool(x, _pooling);
    }

    /// <summary>
    /// Trains the classifier head on labelled graphs.
    /// </summary>
    /// <remarks>
    /// A ridge-regularised least-squares fit against one-hot targets, solved in closed form. With
    /// the propagation fixed, the only thing left to learn is linear, and there is no reason to
    /// reach for gradient descent when a solve gives the exact optimum.
    /// </remarks>
    public GraphClassifier Fit(IReadOnlyList<Graph> graphs, IReadOnlyList<int> labels,
        IReadOnlyList<NdArray>? features = null, double regularisation = 1e-3)
    {
        ArgumentNullException.ThrowIfNull(graphs);
        ArgumentNullException.ThrowIfNull(labels);

        if (graphs.Count != labels.Count)
            throw new ArgumentException("There must be one label per graph.", nameof(labels));
        if (graphs.Count == 0) throw new ArgumentException("There is nothing to train on.", nameof(graphs));

        var embeddings = new List<NdArray>(graphs.Count);
        for (var i = 0; i < graphs.Count; i++)
            embeddings.Add(Embed(graphs[i], features?[i]));

        var width = embeddings[0].Size;
        _classes = [.. labels.Select(l => (double)l).Distinct().Order()];

        // Design matrix with an intercept column, and one-hot targets.
        var x = NdArray.Zeros(graphs.Count, width + 1);
        var y = NdArray.Zeros(graphs.Count, _classes.Length);

        for (var i = 0; i < graphs.Count; i++)
        {
            for (var d = 0; d < width; d++) x[i, d] = embeddings[i].At(d);
            x[i, width] = 1.0;

            y[i, Array.IndexOf(_classes, (double)labels[i])] = 1.0;
        }

        // (XᵀX + λI)⁻¹ Xᵀy, solved rather than inverted.
        var gram = LinAlg.Dot(Transpose(x), x);
        for (var d = 0; d <= width; d++) gram[d, d] += regularisation;

        _classifier = LinAlg.Solve(gram, LinAlg.Dot(Transpose(x), y));

        IsFitted = true;
        return this;
    }

    /// <summary>Class scores for one graph.</summary>
    public NdArray Score(Graph graph, NdArray? nodeFeatures = null)
    {
        if (!IsFitted) throw new InvalidOperationException("The classifier must be fitted before use.");

        var embedding = Embed(graph, nodeFeatures);
        var scores = NdArray.Zeros(_classes.Length);

        for (var c = 0; c < _classes.Length; c++)
        {
            var sum = _classifier[embedding.Size, c];      // the intercept
            for (var d = 0; d < embedding.Size; d++) sum += embedding.At(d) * _classifier[d, c];
            scores.SetAt(c, sum);
        }

        return scores;
    }

    /// <summary>The predicted class of one graph.</summary>
    public int Predict(Graph graph, NdArray? nodeFeatures = null)
    {
        var scores = Score(graph, nodeFeatures);

        var best = 0;
        for (var c = 1; c < scores.Size; c++) if (scores.At(c) > scores.At(best)) best = c;
        return (int)_classes[best];
    }

    /// <summary>Accuracy over a set of labelled graphs.</summary>
    public double Accuracy(IReadOnlyList<Graph> graphs, IReadOnlyList<int> labels,
        IReadOnlyList<NdArray>? features = null)
    {
        var correct = 0;
        for (var i = 0; i < graphs.Count; i++)
            if (Predict(graphs[i], features?[i]) == labels[i]) correct++;

        return graphs.Count > 0 ? (double)correct / graphs.Count : 0.0;
    }

    /// <summary>Degree and log-degree per node, for graphs with no features of their own.</summary>
    private static NdArray DegreeFeatures(Graph graph)
    {
        var result = NdArray.Zeros(graph.NodeCount, 2);

        for (var node = 0; node < graph.NodeCount; node++)
        {
            var degree = graph.Degree(node);
            result[node, 0] = degree;
            // The log compresses a heavy-tailed degree distribution, where one hub would otherwise
            // dominate every aggregate.
            result[node, 1] = Math.Log(1 + degree);
        }

        return result;
    }

    private static NdArray Transpose(NdArray a)
    {
        var result = NdArray.Zeros(a.Shape[1], a.Shape[0]);
        for (var i = 0; i < a.Shape[0]; i++)
            for (var j = 0; j < a.Shape[1]; j++)
                result[j, i] = a[i, j];
        return result;
    }
}
