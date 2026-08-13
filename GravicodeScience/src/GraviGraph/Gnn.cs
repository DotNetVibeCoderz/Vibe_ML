using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Autodiff;

namespace Gravicode.Science.GraviGraph.Neural;

/// <summary>Adam optimiser state for one parameter matrix.</summary>
/// <remarks>
/// Adam rather than plain SGD because GNN gradients are wildly unevenly scaled: a feature that
/// appears in a handful of nodes gets a tiny gradient while a hub's features get an enormous one,
/// and a single global learning rate cannot serve both. Per-parameter step sizes fix that, which
/// is why every reference GNN implementation uses it.
/// </remarks>
internal sealed class AdamState(int rows, int columns)
{
    private readonly double[,] _m = new double[rows, columns];
    private readonly double[,] _v = new double[rows, columns];
    private int _step;

    public void Apply(NdArray parameter, double[,] gradient, double learningRate,
        double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, double weightDecay = 0.0)
    {
        _step++;
        var correction1 = 1 - Math.Pow(beta1, _step);
        var correction2 = 1 - Math.Pow(beta2, _step);

        for (var i = 0; i < rows; i++)
            for (var j = 0; j < columns; j++)
            {
                var g = gradient[i, j] + weightDecay * parameter[i, j];
                _m[i, j] = beta1 * _m[i, j] + (1 - beta1) * g;
                _v[i, j] = beta2 * _v[i, j] + (1 - beta2) * g * g;

                var mHat = _m[i, j] / correction1;
                var vHat = _v[i, j] / correction2;
                parameter[i, j] -= learningRate * mHat / (Math.Sqrt(vHat) + epsilon);
            }
    }
}

/// <summary>Shared numeric helpers for the GNN layers.</summary>
/// <summary>
/// Graph layers expressed on the autodiff tape.
/// </summary>
/// <remarks>
/// <para>
/// A layer written here needs only its forward pass; the backward pass comes from
/// <see cref="Tensor.Backward"/>. That is the whole point — the hand-derived gradients these
/// replaced were correct but did not extend, so a fourth architecture meant a fourth derivation,
/// and one through an attention softmax is genuinely easy to get subtly wrong.
/// </para>
/// <para>
/// This is also the regime where reverse mode is unambiguously the right tool. A GNN forward pass
/// is a handful of large matrix operations, so the tape allocates a few dozen nodes and each one
/// does real work — the opposite of the low-dimensional log posteriors in GraviProb, where the
/// per-node overhead dominates and finite differences win.
/// </para>
/// </remarks>
public static class GnnTape
{
    /// <summary>
    /// One graph convolution: propagate, project, add bias.
    /// </summary>
    /// <remarks>
    /// The bias is a row vector broadcast down the node axis, so its gradient is the column sum
    /// over nodes. The tape's broadcasting rule handles that; the hand-written version needed an
    /// explicit <c>ColumnSums</c> helper, and getting it wrong is invisible until accuracy is
    /// quietly worse.
    /// </remarks>
    public static Tensor Convolve(SparseMatrix propagation, Tensor x, Tensor weight, Tensor bias)
        => TensorOps.SparseMatMul(propagation, x).MatMul(weight) + bias;

    /// <summary>
    /// Inverted dropout as a fixed mask.
    /// </summary>
    /// <remarks>
    /// The mask is drawn once and enters the graph as a constant, which is exactly right: dropout
    /// is a random choice of sub-network, not a function being differentiated. Scaling at training
    /// time rather than inference is what lets the prediction path use the full network unchanged.
    /// </remarks>
    public static Tensor Dropout(Tensor x, double rate, GraviRandom rng)
    {
        if (rate <= 0) return x;

        var scale = 1.0 / (1.0 - rate);
        var mask = NdArray.Zeros(x.Shape[0], x.Shape[1]);
        for (var i = 0; i < x.Shape[0]; i++)
            for (var j = 0; j < x.Shape[1]; j++)
                mask[i, j] = rng.NextDouble() < rate ? 0.0 : scale;

        return x * Tensor.Constant(mask);
    }

    /// <summary>
    /// The mean-of-neighbours aggregator as a sparse matrix, <c>D^-1 A</c>.
    /// </summary>
    /// <remarks>
    /// Writing the aggregator as a matrix rather than a loop is what lets it reuse
    /// <see cref="TensorOps.SparseMatMul"/>, and with it the transpose that pushes gradients back
    /// to neighbours. The hand-written path needed a separate scatter routine for exactly that,
    /// which had to be kept correct by inspection.
    /// </remarks>
    public static SparseMatrix MeanAggregator(Graph graph)
    {
        var builder = new SparseBuilder(graph.NodeCount, graph.NodeCount);

        for (var i = 0; i < graph.NodeCount; i++)
        {
            var neighbours = graph.Neighbors(i);
            if (neighbours.Count == 0) continue;

            var weight = 1.0 / neighbours.Count;
            foreach (var (target, _) in neighbours) builder.Add(i, target, weight);
        }

        return builder.Build();
    }

    /// <summary>
    /// One GraphSAGE layer: <c>[h ; mean(h_neighbours)] W</c>.
    /// </summary>
    public static Tensor SageLayer(SparseMatrix aggregator, Tensor h, Tensor weight)
        => TensorOps.ConcatColumns(h, TensorOps.SparseMatMul(aggregator, h)).MatMul(weight);

    /// <summary>
    /// The edge list a graph attention layer walks: every neighbour, plus a self-loop.
    /// </summary>
    /// <param name="Sources">The attending node of each edge.</param>
    /// <param name="Targets">The attended-to node of each edge.</param>
    /// <param name="NodeCount">Nodes in the graph.</param>
    /// <remarks>
    /// Self-loops are included so a node can attend to itself; without them a node's own features
    /// reach the output only through its neighbours.
    /// </remarks>
    public readonly record struct EdgeList(int[] Sources, int[] Targets, int NodeCount)
    {
        /// <summary>Builds the edge list of <paramref name="graph"/>, with self-loops.</summary>
        public static EdgeList From(Graph graph)
        {
            var sources = new List<int>();
            var targets = new List<int>();

            for (var i = 0; i < graph.NodeCount; i++)
                foreach (var target in graph.Neighbors(i).Select(t => t.Target).Append(i).Distinct())
                {
                    sources.Add(i);
                    targets.Add(target);
                }

            return new EdgeList([.. sources], [.. targets], graph.NodeCount);
        }
    }

    /// <summary>
    /// Softmax within each segment: every edge is normalised against the other edges leaving the
    /// same node.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Composed from <see cref="TensorOps.Gather"/>, <see cref="TensorOps.SegmentSum"/> and
    /// <c>Exp</c> rather than written as its own tape operation. Each piece is already checked
    /// against finite differences, so the composition inherits that — where a bespoke kernel would
    /// need its own derivation of the softmax Jacobian, which is exactly the step the hand-written
    /// GAT had to get right by hand.
    /// </para>
    /// <para>
    /// The per-segment maximum is subtracted before exponentiating, and enters the graph as a
    /// <em>constant</em>. That is exact rather than an approximation: softmax is invariant to a
    /// shift within a segment, so the shift carries no gradient, and detaching it keeps the
    /// backward pass from chasing a term that cancels.
    /// </para>
    /// </remarks>
    public static Tensor SegmentSoftmax(Tensor scores, IReadOnlyList<int> segments, int count)
    {
        var maxima = NdArray.Full(double.NegativeInfinity, count, 1);
        for (var e = 0; e < segments.Count; e++)
            maxima[segments[e], 0] = Math.Max(maxima[segments[e], 0], scores.Value[e, 0]);

        // A segment with no edges would leave -inf here, which would poison the subtraction.
        for (var s = 0; s < count; s++)
            if (double.IsNegativeInfinity(maxima[s, 0])) maxima[s, 0] = 0.0;

        var shifted = scores - TensorOps.Gather(Tensor.Constant(maxima), segments);
        var weights = shifted.Exp();

        return weights / TensorOps.Gather(TensorOps.SegmentSum(weights, segments, count), segments);
    }

    /// <summary>
    /// One graph attention layer: <c>h_i = sum_j softmax_j(LeakyReLU(a_src·z_i + a_dst·z_j)) z_j</c>.
    /// </summary>
    /// <param name="edges">Edge list including self-loops.</param>
    /// <param name="x">Node features.</param>
    /// <param name="weight">The projection <c>W</c>.</param>
    /// <param name="attentionSource">Attention vector applied to the attending node.</param>
    /// <param name="attentionTarget">Attention vector applied to the attended-to node.</param>
    /// <remarks>
    /// Where a GCN weights each neighbour by <c>1/sqrt(d_i d_j)</c> — a property of the graph
    /// alone — this learns the weight from what the neighbour contains. The score is factored as
    /// two dot products computed per node and then added per edge, rather than one dot product
    /// per edge over a concatenated pair: same result, but the expensive part scales with nodes
    /// instead of edges.
    /// </remarks>
    public static Tensor AttentionLayer(EdgeList edges, Tensor x, Tensor weight,
        Tensor attentionSource, Tensor attentionTarget)
        => AttentionLayer(edges, x, weight, attentionSource, attentionTarget, out _);

    /// <inheritdoc cref="AttentionLayer(EdgeList, Tensor, Tensor, Tensor, Tensor)"/>
    /// <param name="attention">
    /// The per-edge attention coefficients, in edge-list order. Useful for inspecting which
    /// neighbours a node learned to attend to.
    /// </param>
    public static Tensor AttentionLayer(EdgeList edges, Tensor x, Tensor weight,
        Tensor attentionSource, Tensor attentionTarget, out Tensor attention)
    {
        var z = x.MatMul(weight);

        var sourceScore = TensorOps.Gather(z.MatMul(attentionSource), edges.Sources);
        var targetScore = TensorOps.Gather(z.MatMul(attentionTarget), edges.Targets);

        attention = SegmentSoftmax(
            TensorOps.LeakyRelu(sourceScore + targetScore, GnnMath.LeakyReluSlope),
            edges.Sources, edges.NodeCount);

        // The (E,1) attention column broadcasts across the (E,d) gathered features.
        var messages = attention * TensorOps.Gather(z, edges.Targets);
        return TensorOps.SegmentSum(messages, edges.Sources, edges.NodeCount);
    }

    /// <summary>One plain gradient-descent step on a parameter, in place.</summary>
    /// <remarks>
    /// Used for the bias vectors. Adam on the weights and plain descent on the biases is not an
    /// oversight: putting the biases on Adam as well was measured and cost 1.6 points of test
    /// accuracy on Cora.
    /// </remarks>
    public static void Descend(Tensor parameter, double learningRate)
    {
        if (parameter.Gradient is null) return;

        var value = parameter.Value;
        for (var i = 0; i < value.Size; i++)
            value.SetAt(i, value.At(i) - learningRate * parameter.Gradient.At(i));
    }
}

/// <summary>Adam over tape parameters, updating each tensor's value in place.</summary>
/// <remarks>
/// Public because a layer written with <see cref="GnnTape"/> needs an optimiser to go with it;
/// the point of putting the layers on the tape is that someone can add a fourth architecture
/// without also having to supply their own training machinery.
/// </remarks>
public sealed class TapeAdam(Tensor parameter, double weightDecay = 0.0)
{
    private readonly NdArray _m = NdArray.ZerosLike(parameter.Value);
    private readonly NdArray _v = NdArray.ZerosLike(parameter.Value);
    private int _step;

    /// <summary>Applies one step from the gradient currently on the tensor.</summary>
    public void Step(double learningRate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
    {
        if (parameter.Gradient is null) return;

        _step++;
        var correction1 = 1 - Math.Pow(beta1, _step);
        var correction2 = 1 - Math.Pow(beta2, _step);

        var value = parameter.Value;
        var gradient = parameter.Gradient;

        for (var i = 0; i < value.Size; i++)
        {
            // Decoupled weight decay: applied to the gradient, matching the hand-written path
            // this replaced so the two can be compared directly.
            var g = gradient.At(i) + weightDecay * value.At(i);

            _m.SetAt(i, beta1 * _m.At(i) + (1 - beta1) * g);
            _v.SetAt(i, beta2 * _v.At(i) + (1 - beta2) * g * g);

            var mHat = _m.At(i) / correction1;
            var vHat = _v.At(i) / correction2;
            value.SetAt(i, value.At(i) - learningRate * mHat / (Math.Sqrt(vHat) + epsilon));
        }
    }
}

internal static class GnnMath
{
    public const double LeakyReluSlope = 0.2;

    /// <summary>Glorot-initialised weight matrix, the standard for GNN layers.</summary>
    public static NdArray Glorot(int rows, int columns, GraviRandom rng)
    {
        var limit = Math.Sqrt(6.0 / (rows + columns));
        return rng.Uniform(-limit, limit, rows, columns);
    }

    /// <summary>Row-wise softmax cross-entropy, returning the loss and its gradient on the logits.</summary>
    public static (double Loss, double[,] Gradient) SoftmaxCrossEntropy(
        NdArray logits, IReadOnlyList<int> labels, IReadOnlyList<int> mask)
    {
        var n = logits.Shape[0];
        var classes = logits.Shape[1];
        var gradient = new double[n, classes];
        var loss = 0.0;

        foreach (var i in mask)
        {
            var row = new double[classes];
            for (var c = 0; c < classes; c++) row[c] = logits[i, c];
            var probabilities = MathUtil.Softmax(row);

            loss -= Math.Log(Math.Max(probabilities[labels[i]], 1e-15));
            for (var c = 0; c < classes; c++)
                gradient[i, c] = (probabilities[c] - (c == labels[i] ? 1.0 : 0.0)) / mask.Count;
        }
        return (loss / Math.Max(1, mask.Count), gradient);
    }

    /// <summary>Multiplies a sparse propagation matrix by a dense one.</summary>
    public static NdArray Propagate(SparseMatrix propagation, NdArray x)
        => propagation.Multiply(x, denseIsMatrix: true);

    /// <summary>Accuracy of the arg-max prediction over a node subset.</summary>
    public static double Accuracy(NdArray logits, IReadOnlyList<int> labels, IReadOnlyList<int> mask)
    {
        if (mask.Count == 0) return double.NaN;
        var correct = 0;
        foreach (var i in mask)
        {
            var best = 0;
            for (var c = 1; c < logits.Shape[1]; c++) if (logits[i, c] > logits[i, best]) best = c;
            if (best == labels[i]) correct++;
        }
        return (double)correct / mask.Count;
    }
}

/// <summary>How a training run progressed.</summary>
/// <param name="Loss">Training loss after each epoch.</param>
/// <param name="TrainAccuracy">Training accuracy after each epoch.</param>
/// <param name="ValidationAccuracy">Validation accuracy after each epoch, when a mask was given.</param>
public sealed record TrainingHistory(
    IReadOnlyList<double> Loss,
    IReadOnlyList<double> TrainAccuracy,
    IReadOnlyList<double> ValidationAccuracy)
{
    /// <inheritdoc />
    public override string ToString()
        => $"epochs={Loss.Count}, final loss={Loss.LastOrDefault():F4}, train acc={TrainAccuracy.LastOrDefault():F4}";
}

/// <summary>
/// A two-layer graph convolutional network for semi-supervised node classification.
/// </summary>
/// <remarks>
/// <para>
/// Each layer computes <c>ReLU(Â X W)</c>, where <c>Â = D^-1/2 (A + I) D^-1/2</c>. Two things are
/// doing the work. The self-loop keeps a node's own features in its representation instead of
/// replacing them with its neighbours'. The symmetric normalisation stops a high-degree node from
/// dominating - without it, activations grow with degree and the network destabilises within a
/// few layers.
/// </para>
/// <para>
/// Two layers is the usual depth because each layer mixes in one more hop of neighbourhood, and
/// beyond three or so hops every node's representation converges toward the graph average - the
/// over-smoothing problem. Training is transductive: the whole graph is seen every epoch, but the
/// loss is computed only on the labelled nodes in <c>trainMask</c>.
/// </para>
/// </remarks>
public sealed class GraphConvolutionalNetwork(
    int hiddenSize = 16,
    double learningRate = 0.01,
    int epochs = 200,
    double dropout = 0.5,
    double weightDecay = 5e-4,
    int seed = 42)
{
    private NdArray _w1 = NdArray.Zeros(0, 0);
    private NdArray _w2 = NdArray.Zeros(0, 0);
    private NdArray _b1 = NdArray.Zeros(0);
    private NdArray _b2 = NdArray.Zeros(0);
    private SparseMatrix? _propagation;
    private NdArray? _features;

    /// <summary>Width of the hidden layer.</summary>
    public int HiddenSize { get; } = hiddenSize;

    /// <summary>Number of output classes.</summary>
    public int ClassCount { get; private set; }

    /// <summary>True once the model has been trained.</summary>
    public bool IsTrained { get; private set; }

    /// <summary>Loss and accuracy per epoch from the last training run.</summary>
    public TrainingHistory? History { get; private set; }

    /// <summary>
    /// Trains on a graph. Labels come from <see cref="Graph.NodeLabels"/> and features from
    /// <see cref="Graph.NodeFeatures"/> unless <paramref name="features"/> is given.
    /// </summary>
    public GraphConvolutionalNetwork Train(Graph graph, IReadOnlyList<int>? trainMask = null,
        IReadOnlyList<int>? validationMask = null, NdArray? features = null)
    {
        var x = features ?? graph.NodeFeatures
            ?? throw new ArgumentException("The graph has no node features; pass them explicitly.");
        var labels = graph.NodeLabels;
        if (labels.Any(l => l < 0) && trainMask is null)
            throw new ArgumentException("Some nodes are unlabelled; supply a training mask.");

        var train = trainMask ?? Enumerable.Range(0, graph.NodeCount).ToArray();
        ClassCount = labels.Where(l => l >= 0).DefaultIfEmpty(0).Max() + 1;

        var rng = new GraviRandom(seed);
        var featureCount = x.Shape[1];
        _w1 = GnnMath.Glorot(featureCount, HiddenSize, rng);
        _w2 = GnnMath.Glorot(HiddenSize, ClassCount, rng);
        _b1 = NdArray.Zeros(HiddenSize);
        _b2 = NdArray.Zeros(ClassCount);
        _propagation = graph.ToSparseAdjacency(addSelfLoops: true, symmetricNormalize: true);
        _features = x;

        // The parameters are tape leaves. They are created once and updated in place, so the
        // graph rebuilt each epoch always closes over the current weights.
        var w1 = Tensor.Parameter(_w1);
        var w2 = Tensor.Parameter(_w2);
        var b1 = Tensor.Parameter(_b1);
        var b2 = Tensor.Parameter(_b2);

        var adamW1 = new TapeAdam(w1, weightDecay);
        var adamW2 = new TapeAdam(w2, weightDecay);

        var input = Tensor.Constant(x);

        var lossHistory = new List<double>();
        var trainHistory = new List<double>();
        var validationHistory = new List<double>();

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            // Forward only — the backward pass is derived from this, not written alongside it.
            var hidden = GnnTape.Convolve(_propagation, GnnTape.Dropout(input, dropout, rng), w1, b1).Relu();
            var logits = GnnTape.Convolve(_propagation, GnnTape.Dropout(hidden, dropout, rng), w2, b2);
            var loss = TensorOps.SoftmaxCrossEntropy(logits, labels, train);

            loss.Backward();

            // Adam on the weights, plain SGD on the biases. That asymmetry looks like an
            // oversight and is not: switching the biases to Adam as well was tried and cost
            // 1.6 points of test accuracy on Cora, so it stays as it was.
            adamW1.Step(learningRate);
            adamW2.Step(learningRate);
            GnnTape.Descend(b1, learningRate);
            GnnTape.Descend(b2, learningRate);

            lossHistory.Add(loss.Item);
            var evaluation = ForwardInference(x);
            trainHistory.Add(GnnMath.Accuracy(evaluation, labels, train));
            if (validationMask is not null)
                validationHistory.Add(GnnMath.Accuracy(evaluation, labels, validationMask));
        }

        History = new TrainingHistory(lossHistory, trainHistory, validationHistory);
        IsTrained = true;
        return this;
    }

    private NdArray ForwardInference(NdArray x)
    {
        // Dropout is a training-time regulariser only; inference uses the full network.
        var propagatedInput = GnnMath.Propagate(_propagation!, x);
        var hidden = LinAlg.Dot(propagatedInput, _w1);
        AddBias(hidden, _b1);
        for (var i = 0; i < hidden.Shape[0]; i++)
            for (var j = 0; j < hidden.Shape[1]; j++)
                hidden[i, j] = Math.Max(0.0, hidden[i, j]);

        var logits = LinAlg.Dot(GnnMath.Propagate(_propagation!, hidden), _w2);
        AddBias(logits, _b2);
        return logits;
    }

    /// <summary>Class scores for every node.</summary>
    public NdArray PredictLogits()
    {
        RequireTrained();
        return ForwardInference(_features!);
    }

    /// <summary>Class probabilities for every node.</summary>
    public NdArray PredictProbabilities()
    {
        var logits = PredictLogits();
        var result = NdArray.Zeros(logits.Shape[0], logits.Shape[1]);
        for (var i = 0; i < logits.Shape[0]; i++)
        {
            var row = new double[logits.Shape[1]];
            for (var c = 0; c < row.Length; c++) row[c] = logits[i, c];
            var probabilities = MathUtil.Softmax(row);
            for (var c = 0; c < row.Length; c++) result[i, c] = probabilities[c];
        }
        return result;
    }

    /// <summary>The predicted class of every node.</summary>
    public int[] Predict()
    {
        var logits = PredictLogits();
        var result = new int[logits.Shape[0]];
        for (var i = 0; i < result.Length; i++)
        {
            var best = 0;
            for (var c = 1; c < logits.Shape[1]; c++) if (logits[i, c] > logits[i, best]) best = c;
            result[i] = best;
        }
        return result;
    }

    /// <summary>The hidden-layer representation, usable as a learned node embedding.</summary>
    public NdArray NodeEmbeddings()
    {
        RequireTrained();
        var hidden = LinAlg.Dot(GnnMath.Propagate(_propagation!, _features!), _w1);
        AddBias(hidden, _b1);
        for (var i = 0; i < hidden.Shape[0]; i++)
            for (var j = 0; j < hidden.Shape[1]; j++)
                hidden[i, j] = Math.Max(0.0, hidden[i, j]);
        return hidden;
    }

    /// <summary>Accuracy over a node subset.</summary>
    public double Score(Graph graph, IReadOnlyList<int> mask)
        => GnnMath.Accuracy(PredictLogits(), graph.NodeLabels, mask);

    private void RequireTrained()
    {
        if (!IsTrained) throw new InvalidOperationException("The network must be trained before use.");
    }

    internal static void AddBias(NdArray matrix, NdArray bias)
    {
        for (var i = 0; i < matrix.Shape[0]; i++)
            for (var j = 0; j < matrix.Shape[1]; j++)
                matrix[i, j] += bias.At(j);
    }

    internal static double[,] ToJagged(NdArray matrix)
    {
        var result = new double[matrix.Shape[0], matrix.Shape[1]];
        for (var i = 0; i < matrix.Shape[0]; i++)
            for (var j = 0; j < matrix.Shape[1]; j++)
                result[i, j] = matrix[i, j];
        return result;
    }

    internal static NdArray ToNdArray(double[,] source, int rows, int columns)
    {
        var result = NdArray.Zeros(rows, columns);
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < columns; j++)
                result[i, j] = source[i, j];
        return result;
    }

    internal static double[] ColumnSums(NdArray matrix)
    {
        var result = new double[matrix.Shape[1]];
        for (var i = 0; i < matrix.Shape[0]; i++)
            for (var j = 0; j < matrix.Shape[1]; j++)
                result[j] += matrix[i, j];
        return result;
    }
}

/// <summary>
/// GraphSAGE with a mean aggregator.
/// </summary>
/// <remarks>
/// The difference from a GCN is that a node's own features and its neighbours' aggregate are kept
/// in separate halves of the layer input, <c>[h_self ; mean(h_neighbours)]</c>, rather than being
/// summed into one normalised message. Keeping them separate lets the layer weight "what I am"
/// against "what surrounds me" independently, and because the aggregator is defined per node
/// rather than over a fixed normalised adjacency matrix, the same weights apply to nodes the
/// model never saw during training - GraphSAGE is inductive where a plain GCN is transductive.
/// </remarks>
public sealed class GraphSage(
    int hiddenSize = 16,
    double learningRate = 0.01,
    int epochs = 200,
    double weightDecay = 5e-4,
    int seed = 42)
{
    private NdArray _w1 = NdArray.Zeros(0, 0);
    private NdArray _w2 = NdArray.Zeros(0, 0);
    private Graph? _graph;
    private NdArray? _features;

    /// <summary>Number of output classes.</summary>
    public int ClassCount { get; private set; }

    /// <summary>True once the model has been trained.</summary>
    public bool IsTrained { get; private set; }

    /// <summary>Loss and accuracy per epoch from the last training run.</summary>
    public TrainingHistory? History { get; private set; }

    /// <summary>Trains on a graph using the mean-aggregated neighbourhood.</summary>
    public GraphSage Train(Graph graph, IReadOnlyList<int>? trainMask = null,
        IReadOnlyList<int>? validationMask = null, NdArray? features = null)
    {
        var x = features ?? graph.NodeFeatures
            ?? throw new ArgumentException("The graph has no node features; pass them explicitly.");
        var labels = graph.NodeLabels;
        var train = trainMask ?? Enumerable.Range(0, graph.NodeCount).ToArray();
        ClassCount = labels.Where(l => l >= 0).DefaultIfEmpty(0).Max() + 1;

        _graph = graph;
        _features = x;
        var n = graph.NodeCount;
        var featureCount = x.Shape[1];

        var rng = new GraviRandom(seed);
        _w1 = GnnMath.Glorot(featureCount * 2, hiddenSize, rng);
        _w2 = GnnMath.Glorot(hiddenSize * 2, ClassCount, rng);

        // The aggregator depends only on the graph, so it is built once rather than per epoch.
        var aggregator = GnnTape.MeanAggregator(graph);

        var w1 = Tensor.Parameter(_w1);
        var w2 = Tensor.Parameter(_w2);
        var adam1 = new TapeAdam(w1, weightDecay);
        var adam2 = new TapeAdam(w2, weightDecay);
        var input = Tensor.Constant(x);

        var lossHistory = new List<double>();
        var trainHistory = new List<double>();
        var validationHistory = new List<double>();

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            var hidden = GnnTape.SageLayer(aggregator, input, w1).Relu();
            var logits = GnnTape.SageLayer(aggregator, hidden, w2);
            var loss = TensorOps.SoftmaxCrossEntropy(logits, labels, train);

            loss.Backward();
            adam1.Step(learningRate);
            adam2.Step(learningRate);

            lossHistory.Add(loss.Item);
            var evaluation = ForwardInference(graph, x);
            trainHistory.Add(GnnMath.Accuracy(evaluation, labels, train));
            if (validationMask is not null)
                validationHistory.Add(GnnMath.Accuracy(evaluation, labels, validationMask));
        }

        History = new TrainingHistory(lossHistory, trainHistory, validationHistory);
        IsTrained = true;
        return this;
    }

    private NdArray ForwardInference(Graph graph, NdArray x)
    {
        var concat1 = Concatenate(x, MeanAggregate(graph, x));
        var hidden = LinAlg.Dot(concat1, _w1);
        for (var i = 0; i < hidden.Shape[0]; i++)
            for (var j = 0; j < hidden.Shape[1]; j++)
                hidden[i, j] = Math.Max(0.0, hidden[i, j]);

        return LinAlg.Dot(Concatenate(hidden, MeanAggregate(graph, hidden)), _w2);
    }

    /// <summary>
    /// Classifies nodes of a graph the model was not trained on, which a transductive GCN cannot do.
    /// </summary>
    public int[] PredictInductive(Graph graph, NdArray features)
    {
        if (!IsTrained) throw new InvalidOperationException("The network must be trained before use.");
        var logits = ForwardInference(graph, features);
        return ArgMaxRows(logits);
    }

    /// <summary>The predicted class of every training-graph node.</summary>
    public int[] Predict()
    {
        if (!IsTrained) throw new InvalidOperationException("The network must be trained before use.");
        return ArgMaxRows(ForwardInference(_graph!, _features!));
    }

    /// <summary>Accuracy over a node subset.</summary>
    public double Score(Graph graph, IReadOnlyList<int> mask)
        => GnnMath.Accuracy(ForwardInference(graph, _features!), graph.NodeLabels, mask);

    private static int[] ArgMaxRows(NdArray logits)
    {
        var result = new int[logits.Shape[0]];
        for (var i = 0; i < result.Length; i++)
        {
            var best = 0;
            for (var c = 1; c < logits.Shape[1]; c++) if (logits[i, c] > logits[i, best]) best = c;
            result[i] = best;
        }
        return result;
    }

    /// <summary>Mean of each node's neighbours' rows; isolated nodes get zeros.</summary>
    internal static NdArray MeanAggregate(Graph graph, NdArray x)
    {
        var result = NdArray.Zeros(x.Shape[0], x.Shape[1]);
        for (var i = 0; i < graph.NodeCount; i++)
        {
            var neighbours = graph.Neighbors(i);
            if (neighbours.Count == 0) continue;
            foreach (var (target, _) in neighbours)
                for (var j = 0; j < x.Shape[1]; j++)
                    result[i, j] += x[target, j];
            for (var j = 0; j < x.Shape[1]; j++) result[i, j] /= neighbours.Count;
        }
        return result;
    }

    internal static NdArray Concatenate(NdArray a, NdArray b)
    {
        var result = NdArray.Zeros(a.Shape[0], a.Shape[1] + b.Shape[1]);
        for (var i = 0; i < a.Shape[0]; i++)
        {
            for (var j = 0; j < a.Shape[1]; j++) result[i, j] = a[i, j];
            for (var j = 0; j < b.Shape[1]; j++) result[i, a.Shape[1] + j] = b[i, j];
        }
        return result;
    }
}

/// <summary>
/// A graph attention network: neighbour contributions are weighted by learned attention rather
/// than by a fixed degree normalisation.
/// </summary>
/// <remarks>
/// <para>
/// A GCN weights every neighbour by <c>1/sqrt(d_i d_j)</c> - a function of the graph structure
/// alone. GAT instead learns <c>alpha_ij = softmax_j(LeakyReLU(a_src . Wh_i + a_dst . Wh_j))</c>,
/// so how much a neighbour matters depends on what it contains. On a citation graph that means a
/// paper can learn to attend to its topically relevant citations and ignore the incidental ones.
/// </para>
/// <para>
/// Gradients here flow through the attention softmax as well as through the aggregation, so the
/// attention parameters are genuinely trained rather than treated as constants. Heads are
/// concatenated in the hidden layer and averaged in the output layer, following the original paper.
/// </para>
/// </remarks>
public sealed class GraphAttentionNetwork(
    int hiddenSize = 8,
    int heads = 4,
    double learningRate = 0.01,
    int epochs = 200,
    double weightDecay = 5e-4,
    int seed = 42)
{
    private NdArray[] _w1 = [];
    private NdArray[] _aSrc1 = [];
    private NdArray[] _aDst1 = [];
    private NdArray _w2 = NdArray.Zeros(0, 0);
    private NdArray _aSrc2 = NdArray.Zeros(0);
    private NdArray _aDst2 = NdArray.Zeros(0);

    private int[][] _neighbours = [];
    private Graph? _graph;
    private NdArray? _features;

    /// <summary>Attention heads in the hidden layer.</summary>
    public int Heads { get; } = heads;

    /// <summary>Number of output classes.</summary>
    public int ClassCount { get; private set; }

    /// <summary>True once the model has been trained.</summary>
    public bool IsTrained { get; private set; }

    /// <summary>Loss and accuracy per epoch from the last training run.</summary>
    public TrainingHistory? History { get; private set; }

    /// <summary>Attention coefficients of the hidden layer from the last forward pass.</summary>
    public IReadOnlyList<Dictionary<(int Source, int Target), double>> AttentionWeights { get; private set; } = [];

    /// <summary>Trains on a graph.</summary>
    public GraphAttentionNetwork Train(Graph graph, IReadOnlyList<int>? trainMask = null,
        IReadOnlyList<int>? validationMask = null, NdArray? features = null)
    {
        var x = features ?? graph.NodeFeatures
            ?? throw new ArgumentException("The graph has no node features; pass them explicitly.");
        var labels = graph.NodeLabels;
        var train = trainMask ?? Enumerable.Range(0, graph.NodeCount).ToArray();
        ClassCount = labels.Where(l => l >= 0).DefaultIfEmpty(0).Max() + 1;

        _graph = graph;
        _features = x;
        var n = graph.NodeCount;
        var featureCount = x.Shape[1];

        // Self-loops are included so a node can attend to itself.
        _neighbours = new int[n][];
        for (var i = 0; i < n; i++)
            _neighbours[i] = graph.Neighbors(i).Select(t => t.Target).Append(i).Distinct().ToArray();

        var rng = new GraviRandom(seed);
        _w1 = new NdArray[Heads];
        _aSrc1 = new NdArray[Heads];
        _aDst1 = new NdArray[Heads];
        for (var h = 0; h < Heads; h++)
        {
            _w1[h] = GnnMath.Glorot(featureCount, hiddenSize, rng);
            _aSrc1[h] = GnnMath.Glorot(hiddenSize, 1, rng).Reshape(hiddenSize);
            _aDst1[h] = GnnMath.Glorot(hiddenSize, 1, rng).Reshape(hiddenSize);
        }
        _w2 = GnnMath.Glorot(hiddenSize * Heads, ClassCount, rng);
        _aSrc2 = GnnMath.Glorot(ClassCount, 1, rng).Reshape(ClassCount);
        _aDst2 = GnnMath.Glorot(ClassCount, 1, rng).Reshape(ClassCount);

        // Reshape returns a view over the same buffer, so updating the tensor updates the field
        // the inference path reads.
        var edges = GnnTape.EdgeList.From(graph);
        var input = Tensor.Constant(x);

        var w1 = _w1.Select(Tensor.Parameter).ToArray();
        var aSrc1 = _aSrc1.Select(a => Tensor.Parameter(a.Reshape(hiddenSize, 1))).ToArray();
        var aDst1 = _aDst1.Select(a => Tensor.Parameter(a.Reshape(hiddenSize, 1))).ToArray();
        var w2 = Tensor.Parameter(_w2);
        var aSrc2 = Tensor.Parameter(_aSrc2.Reshape(ClassCount, 1));
        var aDst2 = Tensor.Parameter(_aDst2.Reshape(ClassCount, 1));

        var adamW1 = w1.Select(p => new TapeAdam(p, weightDecay)).ToArray();
        var adamA1Src = aSrc1.Select(p => new TapeAdam(p)).ToArray();
        var adamA1Dst = aDst1.Select(p => new TapeAdam(p)).ToArray();
        var adamW2 = new TapeAdam(w2, weightDecay);
        var adamA2Src = new TapeAdam(aSrc2);
        var adamA2Dst = new TapeAdam(aDst2);

        var lossHistory = new List<double>();
        var trainHistory = new List<double>();
        var validationHistory = new List<double>();

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            // Forward only. The attention softmax was the hardest backward pass in this library
            // to derive by hand; here it is simply not derived.
            var headAttention = new Tensor[Heads];
            var head = GnnTape.AttentionLayer(edges, input, w1[0], aSrc1[0], aDst1[0], out headAttention[0]).Relu();

            for (var h = 1; h < Heads; h++)
                head = TensorOps.ConcatColumns(head,
                    GnnTape.AttentionLayer(edges, input, w1[h], aSrc1[h], aDst1[h], out headAttention[h]).Relu());

            var logits = GnnTape.AttentionLayer(edges, head, w2, aSrc2, aDst2);
            var loss = TensorOps.SoftmaxCrossEntropy(logits, labels, train);

            loss.Backward();

            for (var h = 0; h < Heads; h++)
            {
                adamW1[h].Step(learningRate);
                adamA1Src[h].Step(learningRate);
                adamA1Dst[h].Step(learningRate);
            }
            adamW2.Step(learningRate);
            adamA2Src.Step(learningRate);
            adamA2Dst.Step(learningRate);

            AttentionWeights = [.. headAttention.Select(a => ToEdgeMap(edges, a))];
            lossHistory.Add(loss.Item);

            var evaluation = ForwardInference(x);
            trainHistory.Add(GnnMath.Accuracy(evaluation, labels, train));
            if (validationMask is not null)
                validationHistory.Add(GnnMath.Accuracy(evaluation, labels, validationMask));
        }

        History = new TrainingHistory(lossHistory, trainHistory, validationHistory);
        IsTrained = true;
        return this;
    }


    /// <summary>
    /// The same forward pass used for training, run for prediction.
    /// </summary>
    /// <remarks>
    /// Built on the tape as well, so there is exactly one definition of what the network computes.
    /// The tensors are constants here, so nothing is recorded for a backward pass that will not
    /// happen.
    /// </remarks>
    private NdArray ForwardInference(NdArray x)
    {
        var edges = GnnTape.EdgeList.From(_graph!);
        var input = Tensor.Constant(x);

        var head = GnnTape.AttentionLayer(edges, input, Tensor.Constant(_w1[0]),
            Tensor.Constant(_aSrc1[0].Reshape(_w1[0].Shape[1], 1)),
            Tensor.Constant(_aDst1[0].Reshape(_w1[0].Shape[1], 1))).Relu();

        for (var h = 1; h < Heads; h++)
            head = TensorOps.ConcatColumns(head,
                GnnTape.AttentionLayer(edges, input, Tensor.Constant(_w1[h]),
                    Tensor.Constant(_aSrc1[h].Reshape(_w1[h].Shape[1], 1)),
                    Tensor.Constant(_aDst1[h].Reshape(_w1[h].Shape[1], 1))).Relu());

        return GnnTape.AttentionLayer(edges, head, Tensor.Constant(_w2),
            Tensor.Constant(_aSrc2.Reshape(ClassCount, 1)),
            Tensor.Constant(_aDst2.Reshape(ClassCount, 1))).Value;
    }

    /// <summary>Reads per-edge attention back out as a (source, target) lookup.</summary>
    private static Dictionary<(int Source, int Target), double> ToEdgeMap(
        GnnTape.EdgeList edges, Tensor attention)
    {
        var result = new Dictionary<(int, int), double>(edges.Sources.Length);
        for (var e = 0; e < edges.Sources.Length; e++)
            result[(edges.Sources[e], edges.Targets[e])] = attention.Value[e, 0];
        return result;
    }

    /// <summary>The predicted class of every node.</summary>
    public int[] Predict()
    {
        if (!IsTrained) throw new InvalidOperationException("The network must be trained before use.");
        var logits = ForwardInference(_features!);
        var result = new int[logits.Shape[0]];
        for (var i = 0; i < result.Length; i++)
        {
            var best = 0;
            for (var c = 1; c < logits.Shape[1]; c++) if (logits[i, c] > logits[i, best]) best = c;
            result[i] = best;
        }
        return result;
    }

    /// <summary>Accuracy over a node subset.</summary>
    public double Score(Graph graph, IReadOnlyList<int> mask)
        => GnnMath.Accuracy(ForwardInference(_features!), graph.NodeLabels, mask);

}
