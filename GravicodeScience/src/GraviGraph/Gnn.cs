using Gravicode.Science.GraviNum;

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

        var adam1 = new AdamState(featureCount, HiddenSize);
        var adam2 = new AdamState(HiddenSize, ClassCount);
        var n = graph.NodeCount;

        var lossHistory = new List<double>();
        var trainHistory = new List<double>();
        var validationHistory = new List<double>();

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            // ---- forward ----
            var xDropped = ApplyDropout(x, dropout, rng, training: true);
            var propagatedInput = GnnMath.Propagate(_propagation, xDropped);
            var hiddenPre = LinAlg.Dot(propagatedInput, _w1);
            AddBias(hiddenPre, _b1);

            var hidden = NdArray.Zeros(n, HiddenSize);
            for (var i = 0; i < n; i++)
                for (var j = 0; j < HiddenSize; j++)
                    hidden[i, j] = Math.Max(0.0, hiddenPre[i, j]);

            var hiddenDropped = ApplyDropout(hidden, dropout, rng, training: true);
            var propagatedHidden = GnnMath.Propagate(_propagation, hiddenDropped);
            var logits = LinAlg.Dot(propagatedHidden, _w2);
            AddBias(logits, _b2);

            var (loss, dLogits) = GnnMath.SoftmaxCrossEntropy(logits, labels, train);

            // ---- backward ----
            var dLogitsArray = ToNdArray(dLogits, n, ClassCount);

            // dW2 = (A_hat H1)^T dZ
            var gradientW2 = ToJagged(LinAlg.Dot(propagatedHidden.T, dLogitsArray));
            var gradientB2 = ColumnSums(dLogitsArray);

            // dH1 = A_hat^T dZ W2^T, then through ReLU.
            var dPropagatedHidden = LinAlg.Dot(dLogitsArray, _w2.T);
            var dHidden = GnnMath.Propagate(_propagation, dPropagatedHidden);
            for (var i = 0; i < n; i++)
                for (var j = 0; j < HiddenSize; j++)
                    if (hiddenPre[i, j] <= 0) dHidden[i, j] = 0.0;

            var gradientW1 = ToJagged(LinAlg.Dot(propagatedInput.T, dHidden));
            var gradientB1 = ColumnSums(dHidden);

            adam1.Apply(_w1, gradientW1, learningRate, weightDecay: weightDecay);
            adam2.Apply(_w2, gradientW2, learningRate, weightDecay: weightDecay);
            for (var j = 0; j < HiddenSize; j++) _b1.SetAt(j, _b1.At(j) - learningRate * gradientB1[j]);
            for (var c = 0; c < ClassCount; c++) _b2.SetAt(c, _b2.At(c) - learningRate * gradientB2[c]);

            lossHistory.Add(loss);
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

    internal static NdArray ApplyDropout(NdArray x, double rate, GraviRandom rng, bool training)
    {
        if (!training || rate <= 0) return x;
        var scale = 1.0 / (1.0 - rate);
        var result = NdArray.Zeros(x.Shape[0], x.Shape[1]);
        for (var i = 0; i < x.Shape[0]; i++)
            for (var j = 0; j < x.Shape[1]; j++)
                // Inverted dropout: scale at training time so inference needs no adjustment.
                result[i, j] = rng.NextDouble() < rate ? 0.0 : x[i, j] * scale;
        return result;
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

        var adam1 = new AdamState(featureCount * 2, hiddenSize);
        var adam2 = new AdamState(hiddenSize * 2, ClassCount);

        var lossHistory = new List<double>();
        var trainHistory = new List<double>();
        var validationHistory = new List<double>();

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            var concat1 = Concatenate(x, MeanAggregate(graph, x));
            var hiddenPre = LinAlg.Dot(concat1, _w1);
            var hidden = NdArray.Zeros(n, hiddenSize);
            for (var i = 0; i < n; i++)
                for (var j = 0; j < hiddenSize; j++)
                    hidden[i, j] = Math.Max(0.0, hiddenPre[i, j]);

            var concat2 = Concatenate(hidden, MeanAggregate(graph, hidden));
            var logits = LinAlg.Dot(concat2, _w2);

            var (loss, dLogits) = GnnMath.SoftmaxCrossEntropy(logits, labels, train);
            var dLogitsArray = GraphConvolutionalNetwork.ToNdArray(dLogits, n, ClassCount);

            var gradientW2 = GraphConvolutionalNetwork.ToJagged(LinAlg.Dot(concat2.T, dLogitsArray));

            // The gradient reaching the hidden layer arrives through both halves of concat2:
            // directly as the self part, and spread over neighbours as the aggregate part.
            var dConcat2 = LinAlg.Dot(dLogitsArray, _w2.T);
            var dHidden = NdArray.Zeros(n, hiddenSize);
            for (var i = 0; i < n; i++)
                for (var j = 0; j < hiddenSize; j++)
                    dHidden[i, j] = dConcat2[i, j];

            var dAggregate = NdArray.Zeros(n, hiddenSize);
            for (var i = 0; i < n; i++)
                for (var j = 0; j < hiddenSize; j++)
                    dAggregate[i, j] = dConcat2[i, hiddenSize + j];

            var scattered = ScatterMean(graph, dAggregate);
            for (var i = 0; i < n; i++)
                for (var j = 0; j < hiddenSize; j++)
                {
                    dHidden[i, j] += scattered[i, j];
                    if (hiddenPre[i, j] <= 0) dHidden[i, j] = 0.0;
                }

            var gradientW1 = GraphConvolutionalNetwork.ToJagged(LinAlg.Dot(concat1.T, dHidden));

            adam1.Apply(_w1, gradientW1, learningRate, weightDecay: weightDecay);
            adam2.Apply(_w2, gradientW2, learningRate, weightDecay: weightDecay);

            lossHistory.Add(loss);
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

    /// <summary>Transpose of <see cref="MeanAggregate"/>, used to push gradients back to neighbours.</summary>
    internal static NdArray ScatterMean(Graph graph, NdArray gradient)
    {
        var result = NdArray.Zeros(gradient.Shape[0], gradient.Shape[1]);
        for (var i = 0; i < graph.NodeCount; i++)
        {
            var neighbours = graph.Neighbors(i);
            if (neighbours.Count == 0) continue;
            foreach (var (target, _) in neighbours)
                for (var j = 0; j < gradient.Shape[1]; j++)
                    result[target, j] += gradient[i, j] / neighbours.Count;
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

        var adamW1 = Enumerable.Range(0, Heads).Select(_ => new AdamState(featureCount, hiddenSize)).ToArray();
        var adamA1Src = Enumerable.Range(0, Heads).Select(_ => new AdamState(hiddenSize, 1)).ToArray();
        var adamA1Dst = Enumerable.Range(0, Heads).Select(_ => new AdamState(hiddenSize, 1)).ToArray();
        var adamW2 = new AdamState(hiddenSize * Heads, ClassCount);
        var adamA2Src = new AdamState(ClassCount, 1);
        var adamA2Dst = new AdamState(ClassCount, 1);

        var lossHistory = new List<double>();
        var trainHistory = new List<double>();
        var validationHistory = new List<double>();

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            // ---- forward, hidden layer: one attention head at a time, then concatenate ----
            var headOutputs = new NdArray[Heads];
            var headStates = new AttentionState[Heads];
            for (var h = 0; h < Heads; h++)
            {
                headStates[h] = AttentionForward(x, _w1[h], _aSrc1[h], _aDst1[h]);
                headOutputs[h] = Relu(headStates[h].Output, out headStates[h].PreActivation);
            }

            var concatenated = ConcatenateHeads(headOutputs);

            // ---- forward, output layer: a single averaged head ----
            var outputState = AttentionForward(concatenated, _w2, _aSrc2, _aDst2);
            var logits = outputState.Output;

            var (loss, dLogitsRaw) = GnnMath.SoftmaxCrossEntropy(logits, labels, train);
            var dLogits = GraphConvolutionalNetwork.ToNdArray(dLogitsRaw, n, ClassCount);

            // ---- backward ----
            var (dConcat, gradientW2, gradientASrc2, gradientADst2) =
                AttentionBackward(concatenated, outputState, dLogits, _w2, _aSrc2, _aDst2);

            for (var h = 0; h < Heads; h++)
            {
                var dHead = NdArray.Zeros(n, hiddenSize);
                for (var i = 0; i < n; i++)
                    for (var j = 0; j < hiddenSize; j++)
                    {
                        var value = dConcat[i, h * hiddenSize + j];
                        // Back through the ReLU applied to this head's output.
                        dHead[i, j] = headStates[h].PreActivation![i, j] > 0 ? value : 0.0;
                    }

                var (_, gradientW1, gradientASrc1, gradientADst1) =
                    AttentionBackward(x, headStates[h], dHead, _w1[h], _aSrc1[h], _aDst1[h]);

                adamW1[h].Apply(_w1[h], gradientW1, learningRate, weightDecay: weightDecay);
                adamA1Src[h].Apply(_aSrc1[h].Reshape(hiddenSize, 1), ToColumn(gradientASrc1), learningRate);
                adamA1Dst[h].Apply(_aDst1[h].Reshape(hiddenSize, 1), ToColumn(gradientADst1), learningRate);
            }

            adamW2.Apply(_w2, gradientW2, learningRate, weightDecay: weightDecay);
            adamA2Src.Apply(_aSrc2.Reshape(ClassCount, 1), ToColumn(gradientASrc2), learningRate);
            adamA2Dst.Apply(_aDst2.Reshape(ClassCount, 1), ToColumn(gradientADst2), learningRate);

            AttentionWeights = headStates.Select(s => s.Attention).ToList();
            lossHistory.Add(loss);

            var evaluation = ForwardInference(x);
            trainHistory.Add(GnnMath.Accuracy(evaluation, labels, train));
            if (validationMask is not null)
                validationHistory.Add(GnnMath.Accuracy(evaluation, labels, validationMask));
        }

        History = new TrainingHistory(lossHistory, trainHistory, validationHistory);
        IsTrained = true;
        return this;
    }

    /// <summary>Intermediate values one attention layer needs for its backward pass.</summary>
    private sealed class AttentionState
    {
        public NdArray Projected = NdArray.Zeros(0, 0);
        public NdArray Output = NdArray.Zeros(0, 0);
        public NdArray? PreActivation;
        public Dictionary<(int Source, int Target), double> Attention = [];
        public Dictionary<(int Source, int Target), double> RawScores = [];
    }

    private AttentionState AttentionForward(NdArray x, NdArray w, NdArray aSrc, NdArray aDst)
    {
        var n = x.Shape[0];
        var d = w.Shape[1];
        var z = LinAlg.Dot(x, w);

        var state = new AttentionState { Projected = z, Output = NdArray.Zeros(n, d) };

        // Pre-computing a_src . z_i and a_dst . z_j turns the per-edge score into one addition.
        var srcScore = new double[n];
        var dstScore = new double[n];
        for (var i = 0; i < n; i++)
            for (var k = 0; k < d; k++)
            {
                srcScore[i] += z[i, k] * aSrc.At(k);
                dstScore[i] += z[i, k] * aDst.At(k);
            }

        for (var i = 0; i < n; i++)
        {
            var neighbours = _neighbours[i];
            var scores = new double[neighbours.Length];
            for (var t = 0; t < neighbours.Length; t++)
            {
                var raw = srcScore[i] + dstScore[neighbours[t]];
                state.RawScores[(i, neighbours[t])] = raw;
                scores[t] = raw > 0 ? raw : GnnMath.LeakyReluSlope * raw;
            }

            var attention = MathUtil.Softmax(scores);
            for (var t = 0; t < neighbours.Length; t++)
            {
                state.Attention[(i, neighbours[t])] = attention[t];
                for (var k = 0; k < d; k++)
                    state.Output[i, k] += attention[t] * z[neighbours[t], k];
            }
        }
        return state;
    }

    private (NdArray InputGradient, double[,] WeightGradient, double[] SrcGradient, double[] DstGradient)
        AttentionBackward(NdArray x, AttentionState state, NdArray dOutput, NdArray w, NdArray aSrc, NdArray aDst)
    {
        var n = x.Shape[0];
        var d = w.Shape[1];
        var z = state.Projected;

        var dz = NdArray.Zeros(n, d);
        var gradientSrc = new double[d];
        var gradientDst = new double[d];

        for (var i = 0; i < n; i++)
        {
            var neighbours = _neighbours[i];

            // Gradient of the loss with respect to each attention coefficient.
            var dAlpha = new double[neighbours.Length];
            for (var t = 0; t < neighbours.Length; t++)
            {
                var j = neighbours[t];
                var accumulator = 0.0;
                for (var k = 0; k < d; k++)
                {
                    accumulator += dOutput[i, k] * z[j, k];
                    // Aggregation path: the neighbour's projection is scaled by alpha.
                    dz[j, k] += state.Attention[(i, j)] * dOutput[i, k];
                }
                dAlpha[t] = accumulator;
            }

            // Softmax Jacobian: dE_t = alpha_t * (dAlpha_t - sum_k alpha_k dAlpha_k).
            var weighted = 0.0;
            for (var t = 0; t < neighbours.Length; t++)
                weighted += state.Attention[(i, neighbours[t])] * dAlpha[t];

            for (var t = 0; t < neighbours.Length; t++)
            {
                var j = neighbours[t];
                var alpha = state.Attention[(i, j)];
                var dScore = alpha * (dAlpha[t] - weighted);

                // Back through LeakyReLU.
                var raw = state.RawScores[(i, j)];
                var dRaw = dScore * (raw > 0 ? 1.0 : GnnMath.LeakyReluSlope);

                for (var k = 0; k < d; k++)
                {
                    gradientSrc[k] += dRaw * z[i, k];
                    gradientDst[k] += dRaw * z[j, k];
                    dz[i, k] += dRaw * aSrc.At(k);
                    dz[j, k] += dRaw * aDst.At(k);
                }
            }
        }

        var weightGradient = GraphConvolutionalNetwork.ToJagged(LinAlg.Dot(x.T, dz));
        var inputGradient = LinAlg.Dot(dz, w.T);
        return (inputGradient, weightGradient, gradientSrc, gradientDst);
    }

    private NdArray ForwardInference(NdArray x)
    {
        var headOutputs = new NdArray[Heads];
        for (var h = 0; h < Heads; h++)
            headOutputs[h] = Relu(AttentionForward(x, _w1[h], _aSrc1[h], _aDst1[h]).Output, out _);

        return AttentionForward(ConcatenateHeads(headOutputs), _w2, _aSrc2, _aDst2).Output;
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

    private static NdArray Relu(NdArray x, out NdArray? preActivation)
    {
        preActivation = x.Copy();
        var result = NdArray.Zeros(x.Shape[0], x.Shape[1]);
        for (var i = 0; i < x.Shape[0]; i++)
            for (var j = 0; j < x.Shape[1]; j++)
                result[i, j] = Math.Max(0.0, x[i, j]);
        return result;
    }

    private static NdArray ConcatenateHeads(NdArray[] heads)
    {
        var n = heads[0].Shape[0];
        var d = heads[0].Shape[1];
        var result = NdArray.Zeros(n, d * heads.Length);
        for (var h = 0; h < heads.Length; h++)
            for (var i = 0; i < n; i++)
                for (var j = 0; j < d; j++)
                    result[i, h * d + j] = heads[h][i, j];
        return result;
    }

    private static double[,] ToColumn(double[] values)
    {
        var result = new double[values.Length, 1];
        for (var i = 0; i < values.Length; i++) result[i, 0] = values[i];
        return result;
    }
}
