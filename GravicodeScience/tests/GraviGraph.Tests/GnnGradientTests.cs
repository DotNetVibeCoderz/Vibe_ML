using Gravicode.Science.GraviGraph;
using Gravicode.Science.GraviGraph.Neural;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Autodiff;
using Xunit;

namespace Gravicode.Science.Tests.GraviGraph;

/// <summary>
/// Checks the GNN layers' gradients against central finite differences on the loss itself.
/// </summary>
/// <remarks>
/// <para>
/// These exist because the hand-derived backward passes they replaced were not all correct. The
/// GCN one omitted the dropout mask on the hidden layer's backward path, so whenever dropout was
/// active the gradient was about 40% wrong — and the model still trained to a plausible-looking
/// 71% on Cora, which is exactly why nobody noticed.
/// </para>
/// <para>
/// Finite differences share no code with the tape, which is what makes them a real check. Dropout
/// masks are fixed rather than redrawn: dropout picks a random sub-network, and a gradient can
/// only be verified against the same sub-network it was computed on.
/// </para>
/// </remarks>
public class GnnGradientTests
{
    private const int Nodes = 6;
    private const int Features = 4;
    private const int Hidden = 3;
    private const int Classes = 3;

    private static readonly int[] Labels = [0, 1, 2, 1, 0, 2];
    private static readonly int[] Mask = [0, 1, 3, 5];

    /// <summary>A small symmetric propagation matrix, the shape a GCN actually walks.</summary>
    private static SparseMatrix Propagation()
    {
        var builder = new SparseBuilder(Nodes, Nodes);
        foreach (var (i, j) in new[] { (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3) })
        {
            builder.Add(i, j, 0.4);
            builder.Add(j, i, 0.4);
        }
        for (var i = 0; i < Nodes; i++) builder.Add(i, i, 0.5);
        return builder.Build();
    }

    /// <summary>An inverted-dropout mask, drawn once so the sub-network stays fixed.</summary>
    private static NdArray FixedDropoutMask(int rows, int columns, int seed)
    {
        var rng = new GraviRandom(seed);
        var mask = NdArray.Zeros(rows, columns);
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < columns; j++)
                mask[i, j] = rng.NextDouble() < 0.5 ? 0.0 : 2.0;
        return mask;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GcnGradient_AgreesWithFiniteDifferences(bool useDropout)
    {
        var rng = new GraviRandom(5);
        var propagation = Propagation();

        var x = rng.StandardNormal(Nodes, Features);
        var w1 = rng.StandardNormal(Features, Hidden);
        var w2 = rng.StandardNormal(Hidden, Classes);
        var b1 = rng.StandardNormal(Hidden);
        var b2 = rng.StandardNormal(Classes);

        var maskInput = FixedDropoutMask(Nodes, Features, 11);
        var maskHidden = FixedDropoutMask(Nodes, Hidden, 12);

        // The forward pass written plainly. This is the definition the gradient must match.
        double Loss(NdArray weight1)
        {
            var dropped = useDropout ? UFunc.Multiply(x, maskInput) : x;
            var hiddenPre = LinAlg.Dot(propagation.Multiply(dropped, denseIsMatrix: true), weight1);
            for (var i = 0; i < Nodes; i++)
                for (var j = 0; j < Hidden; j++)
                    hiddenPre[i, j] += b1.At(j);

            var hidden = NdArray.Zeros(Nodes, Hidden);
            for (var i = 0; i < Nodes; i++)
                for (var j = 0; j < Hidden; j++)
                    hidden[i, j] = Math.Max(0.0, hiddenPre[i, j]);

            var hiddenDropped = useDropout ? UFunc.Multiply(hidden, maskHidden) : hidden;
            var logits = LinAlg.Dot(propagation.Multiply(hiddenDropped, denseIsMatrix: true), w2);
            for (var i = 0; i < Nodes; i++)
                for (var c = 0; c < Classes; c++)
                    logits[i, c] += b2.At(c);

            var total = 0.0;
            foreach (var i in Mask)
            {
                var max = double.NegativeInfinity;
                for (var c = 0; c < Classes; c++) max = Math.Max(max, logits[i, c]);
                var sum = 0.0;
                for (var c = 0; c < Classes; c++) sum += Math.Exp(logits[i, c] - max);
                total -= logits[i, Labels[i]] - max - Math.Log(sum);
            }
            return total / Mask.Length;
        }

        // The same forward pass on the tape, which yields the gradient without deriving it.
        var input = Tensor.Constant(x);
        var weight1 = Tensor.Parameter(w1.Copy());

        var layer1 = useDropout ? input * Tensor.Constant(maskInput) : input;
        var hiddenTensor = (TensorOps.SparseMatMul(propagation, layer1).MatMul(weight1)
                            + Tensor.Constant(b1)).Relu();
        var layer2 = useDropout ? hiddenTensor * Tensor.Constant(maskHidden) : hiddenTensor;
        var logitsTensor = TensorOps.SparseMatMul(propagation, layer2).MatMul(Tensor.Constant(w2))
                           + Tensor.Constant(b2);

        TensorOps.SoftmaxCrossEntropy(logitsTensor, Labels, Mask).Backward();

        const double step = 1e-6;
        for (var i = 0; i < Features; i++)
            for (var j = 0; j < Hidden; j++)
            {
                var up = w1.Copy();
                up[i, j] += step;
                var down = w1.Copy();
                down[i, j] -= step;

                var numeric = (Loss(up) - Loss(down)) / (2 * step);
                var analytic = weight1.Gradient![i, j];

                Assert.True(Math.Abs(analytic - numeric) / Math.Max(1.0, Math.Abs(numeric)) < 1e-6,
                    $"dropout={useDropout}, dW1[{i},{j}]: tape {analytic:G8} vs finite difference {numeric:G8}");
            }
    }

    [Fact]
    public void SageLayerGradient_AgreesWithFiniteDifferences()
    {
        var graph = new Graph(directed: false);
        for (var i = 0; i < Nodes; i++) graph.AddNode();
        foreach (var (i, j) in new[] { (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0) })
            graph.AddEdge(i, j);

        var rng = new GraviRandom(9);
        var aggregator = GnnTape.MeanAggregator(graph);
        var x = Tensor.Constant(rng.StandardNormal(Nodes, Features));
        var weights = Tensor.Constant(rng.StandardNormal(Nodes, Classes));

        // A GraphSAGE layer takes [self ; mean(neighbours)], so the weight matrix is twice as
        // wide as the feature count.
        var result = GradientCheck.Check(
            w => (GnnTape.SageLayer(aggregator, x, w) * weights).Sum(),
            rng.StandardNormal(Features * 2, Classes));

        Assert.True(result.Passed(1e-6), result.ToString());
    }

    /// <summary>A small undirected graph with one isolated node.</summary>
    private static Graph AttentionGraph()
    {
        var graph = new Graph(directed: false);
        for (var i = 0; i < Nodes; i++) graph.AddNode();
        foreach (var (i, j) in new[] { (0, 1), (1, 2), (2, 3), (3, 0), (0, 2) })
            graph.AddEdge(i, j);
        // Nodes 4 and 5 are left with only their self-loops.
        return graph;
    }

    [Fact]
    public void AttentionLayerGradient_FlowsToTheProjectionWeights()
    {
        var rng = new GraviRandom(13);
        var edges = GnnTape.EdgeList.From(AttentionGraph());

        var x = Tensor.Constant(rng.StandardNormal(Nodes, Features));
        var aSrc = Tensor.Constant(rng.StandardNormal(Hidden, 1));
        var aDst = Tensor.Constant(rng.StandardNormal(Hidden, 1));
        var weights = Tensor.Constant(rng.StandardNormal(Nodes, Hidden));

        var result = GradientCheck.Check(
            w => (GnnTape.AttentionLayer(edges, x, w, aSrc, aDst) * weights).Sum(),
            rng.StandardNormal(Features, Hidden));

        Assert.True(result.Passed(1e-6), result.ToString());
    }

    [Fact]
    public void AttentionLayerGradient_FlowsThroughTheAttentionSoftmax()
    {
        // This is the path the hand-derived version had to work out by hand: the loss reaches
        // a_src only through LeakyReLU and then the per-node softmax over incident edges.
        var rng = new GraviRandom(17);
        var edges = GnnTape.EdgeList.From(AttentionGraph());

        var x = Tensor.Constant(rng.StandardNormal(Nodes, Features));
        var w = Tensor.Constant(rng.StandardNormal(Features, Hidden));
        var aDst = Tensor.Constant(rng.StandardNormal(Hidden, 1));
        var weights = Tensor.Constant(rng.StandardNormal(Nodes, Hidden));

        var source = GradientCheck.Check(
            a => (GnnTape.AttentionLayer(edges, x, w, a, aDst) * weights).Sum(),
            rng.StandardNormal(Hidden, 1));
        Assert.True(source.Passed(1e-6), $"a_src: {source}");

        var aSrc = Tensor.Constant(rng.StandardNormal(Hidden, 1));
        var target = GradientCheck.Check(
            a => (GnnTape.AttentionLayer(edges, x, w, aSrc, a) * weights).Sum(),
            rng.StandardNormal(Hidden, 1));
        Assert.True(target.Passed(1e-6), $"a_dst: {target}");
    }

    [Fact]
    public void SegmentSoftmax_SumsToOneWithinEachNodesEdges()
    {
        var edges = GnnTape.EdgeList.From(AttentionGraph());
        var rng = new GraviRandom(23);
        var scores = Tensor.Constant(rng.StandardNormal(edges.Sources.Length, 1));

        var attention = GnnTape.SegmentSoftmax(scores, edges.Sources, edges.NodeCount);

        var totals = new double[edges.NodeCount];
        for (var e = 0; e < edges.Sources.Length; e++)
        {
            totals[edges.Sources[e]] += attention.Value[e, 0];
            Assert.InRange(attention.Value[e, 0], 0.0, 1.0);
        }

        // Every node has at least a self-loop, so every segment is non-empty.
        foreach (var total in totals) Assert.Equal(1.0, total, 12);
    }

    [Fact]
    public void SegmentSoftmax_SurvivesScoresThatWouldOverflowExp()
    {
        // Without the per-segment max shift, exp(800) is infinity and the result is NaN.
        int[] segments = [0, 0, 1, 1];
        var scores = Tensor.Constant(NdArray.FromArray(new double[,] { { 800.0 }, { 802.0 }, { -5.0 }, { -5.0 } }));

        var attention = GnnTape.SegmentSoftmax(scores, segments, 2);

        Assert.Equal(1.0 / (1 + Math.Exp(2.0)), attention.Value[0, 0], 10);
        Assert.Equal(Math.Exp(2.0) / (1 + Math.Exp(2.0)), attention.Value[1, 0], 10);
        Assert.Equal(0.5, attention.Value[2, 0], 10);
        Assert.Equal(0.5, attention.Value[3, 0], 10);
    }

    [Fact]
    public void GatherAndSegmentSum_AreAdjoints()
    {
        // <Gather(a), b> must equal <a, SegmentSum(b)> for any a and b — the defining property
        // of an adjoint pair, and a check that neither backward rule drifted from the other.
        int[] indices = [0, 2, 2, 1];
        var rng = new GraviRandom(29);

        var a = rng.StandardNormal(3, 2);
        var b = rng.StandardNormal(4, 2);

        var gathered = TensorOps.Gather(Tensor.Constant(a), indices).Value;
        var scattered = TensorOps.SegmentSum(Tensor.Constant(b), indices, 3).Value;

        var left = 0.0;
        for (var e = 0; e < 4; e++)
            for (var j = 0; j < 2; j++) left += gathered[e, j] * b[e, j];

        var right = 0.0;
        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 2; j++) right += a[i, j] * scattered[i, j];

        Assert.Equal(left, right, 12);
    }

    [Fact]
    public void MeanAggregator_AveragesNeighboursAndLeavesIsolatedNodesAtZero()
    {
        var graph = new Graph(directed: false);
        for (var i = 0; i < 4; i++) graph.AddNode();
        graph.AddEdge(0, 1);
        graph.AddEdge(0, 2);
        // Node 3 is isolated.

        var aggregator = GnnTape.MeanAggregator(graph);
        var x = NdArray.FromArray(new double[,] { { 1.0 }, { 4.0 }, { 10.0 }, { 99.0 } });
        var aggregated = aggregator.Multiply(x, denseIsMatrix: true);

        Assert.Equal(7.0, aggregated[0, 0], 12);    // mean of nodes 1 and 2
        Assert.Equal(1.0, aggregated[1, 0], 12);    // only neighbour is node 0
        Assert.Equal(1.0, aggregated[2, 0], 12);
        Assert.Equal(0.0, aggregated[3, 0], 12);    // no neighbours contribute nothing, not NaN
    }
}
