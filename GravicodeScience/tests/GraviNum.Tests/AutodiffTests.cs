using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Autodiff;
using Xunit;

namespace GraviNum.Tests;

/// <summary>
/// Tests for the reverse-mode tape.
/// </summary>
/// <remarks>
/// Most of these check the tape against central finite differences, which share none of its code.
/// Where a derivative has a closed form worth pinning — <c>d/dx x² = 2x</c>, the softmax that falls
/// out of log-sum-exp — that is asserted directly instead, because agreeing with finite differences
/// only says the two agree, not that either is right.
/// </remarks>
public class AutodiffTests
{
    private static NdArray Values(params double[] values) => NdArray.FromValues(values);

    // ---------------------------------------------------------------- closed forms

    [Fact]
    public void Polynomial_MatchesTheDerivativeByHand()
    {
        // f(x) = 3x^2 + 2x + 1  ->  f'(x) = 6x + 2
        var x = Tensor.Parameter(4.0);
        var f = (Tensor.Constant(3.0) * x.Pow(2) + Tensor.Constant(2.0) * x + Tensor.Constant(1.0)).Sum();

        f.Backward();

        Assert.Equal(57.0, f.Item, 10);
        Assert.Equal(26.0, x.Gradient!.At(0), 10);
    }

    [Fact]
    public void Sigmoid_DerivativeIsSTimesOneMinusS()
    {
        var x = Tensor.Parameter(0.7);
        var s = x.Sigmoid();
        s.Sum().Backward();

        var value = s.Item;
        Assert.Equal(value * (1 - value), x.Gradient!.At(0), 12);
    }

    [Fact]
    public void Exp_IsItsOwnDerivative()
    {
        var x = Tensor.Parameter(Values(-1.0, 0.0, 2.0));
        var y = x.Exp();
        y.Sum().Backward();

        for (var i = 0; i < 3; i++)
            Assert.Equal(y.Value.At(i), x.Gradient!.At(i), 10);
    }

    [Fact]
    public void LogSumExp_GradientIsTheSoftmax()
    {
        var x = Tensor.Parameter(Values(1.0, 2.0, 3.0));
        x.LogSumExp().Backward();

        var total = Math.Exp(1.0) + Math.Exp(2.0) + Math.Exp(3.0);
        for (var i = 0; i < 3; i++)
            Assert.Equal(Math.Exp(i + 1.0) / total, x.Gradient!.At(i), 10);

        // The gradient of a log-sum-exp is a probability vector, so it sums to one.
        Assert.Equal(1.0, Statistics.Sum(x.Gradient!), 12);
    }

    [Fact]
    public void LogSumExp_SurvivesValuesThatWouldOverflowExp()
    {
        // exp(1000) is infinity, so the naive log(sum(exp(x))) returns infinity and a NaN gradient.
        var x = Tensor.Parameter(Values(1000.0, 1001.0));
        var y = x.LogSumExp();
        y.Backward();

        Assert.True(double.IsFinite(y.Item), "log-sum-exp overflowed");
        Assert.Equal(1001.0 + Math.Log(1 + Math.Exp(-1.0)), y.Item, 10);
        Assert.True(double.IsFinite(x.Gradient!.At(0)));
        Assert.Equal(1.0, Statistics.Sum(x.Gradient!), 12);
    }

    // ---------------------------------------------------------------- against finite differences

    public static TheoryData<string, Func<Tensor, Tensor>, double[]> ScalarFunctions() => new()
    {
        { "sum of squares", t => (t * t).Sum(), [0.5, -1.5, 2.25] },
        { "log", t => t.Log().Sum(), [0.3, 1.0, 4.0] },
        { "exp", t => t.Exp().Sum(), [-2.0, 0.0, 1.5] },
        { "sqrt", t => t.Sqrt().Sum(), [0.25, 1.0, 9.0] },
        { "tanh", t => t.Tanh().Sum(), [-1.0, 0.0, 2.0] },
        { "sigmoid", t => t.Sigmoid().Sum(), [-3.0, 0.0, 3.0] },
        { "softplus", t => t.Softplus().Sum(), [-2.0, 0.5, 4.0] },
        { "relu", t => t.Relu().Sum(), [-2.0, 0.5, 4.0] },
        { "abs", t => t.Abs().Sum(), [-2.0, 0.5, 4.0] },
        { "pow 1.5", t => t.Pow(1.5).Sum(), [0.4, 1.0, 3.0] },
        { "log-sum-exp", t => t.LogSumExp(), [0.2, 1.7, -0.9] },
        { "quotient chain", t => (t / (t + Tensor.Constant(3.0))).Sum(), [0.5, 2.0, -1.0] },
        { "mean of logs", t => t.Log().Mean(), [1.0, 2.0, 3.0] },
        { "composed", t => (t.Sigmoid() * t.Tanh()).Log().Abs().Sum(), [0.8, 1.4, 2.2] },
    };

    [Theory]
    [MemberData(nameof(ScalarFunctions))]
    public void Gradients_AgreeWithFiniteDifferences(string name, Func<Tensor, Tensor> f, double[] at)
    {
        var result = GradientCheck.Check(f, Values(at));
        Assert.True(result.Passed(1e-6), $"{name}: {result}");
    }

    [Fact]
    public void MatMulGradient_AgreesWithFiniteDifferences()
    {
        var rng = new GraviRandom(7);
        var b = Tensor.Constant(rng.StandardNormal(4, 3));

        // Reduce with a non-uniform weighting, or every column would contribute identically and
        // a transposed backward rule would still pass.
        var weights = Tensor.Constant(rng.StandardNormal(2, 3));

        var result = GradientCheck.Check(
            a => (a.MatMul(b) * weights).Sum(),
            rng.StandardNormal(2, 4));

        Assert.True(result.Passed(1e-6), result.ToString());
    }

    [Fact]
    public void TransposeAndReshape_PreserveGradients()
    {
        var rng = new GraviRandom(11);
        var weights = Tensor.Constant(rng.StandardNormal(3, 2));

        var transposed = GradientCheck.Check(a => (a.T() * weights).Sum(), rng.StandardNormal(2, 3));
        Assert.True(transposed.Passed(1e-6), $"transpose: {transposed}");

        var reshaped = GradientCheck.Check(
            a => (a.Reshape(3, 2) * weights).Sum(), rng.StandardNormal(2, 3));
        Assert.True(reshaped.Passed(1e-6), $"reshape: {reshaped}");
    }

    // ---------------------------------------------------------------- GNN building blocks

    [Fact]
    public void SparseMatMulGradient_AgreesWithFiniteDifferences()
    {
        // A deliberately asymmetric sparse matrix: if the backward rule used A instead of A^T,
        // a symmetric one would hide the mistake completely.
        var builder = new SparseBuilder(4, 4);
        builder.Add(0, 1, 2.0);
        builder.Add(1, 0, -1.5);
        builder.Add(1, 3, 0.5);
        builder.Add(2, 2, 3.0);
        builder.Add(3, 0, 1.0);
        var a = builder.Build();

        var rng = new GraviRandom(31);
        var weights = Tensor.Constant(rng.StandardNormal(4, 2));

        var result = GradientCheck.Check(
            x => (TensorOps.SparseMatMul(a, x) * weights).Sum(),
            rng.StandardNormal(4, 2));

        Assert.True(result.Passed(1e-6), result.ToString());
    }

    [Fact]
    public void ConcatColumnsGradient_RoutesEachHalfBackToItsOwnOperand()
    {
        var left = Tensor.Parameter(NdArray.FromArray(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }));
        var right = Tensor.Parameter(NdArray.FromArray(new double[,] { { 5.0 }, { 6.0 } }));

        // Weight every output column differently, so a backward pass that mixed up the split
        // would produce visibly wrong numbers rather than a plausible average.
        var weights = Tensor.Constant(NdArray.FromArray(new double[,] { { 10.0, 20.0, 30.0 }, { 40.0, 50.0, 60.0 } }));
        (TensorOps.ConcatColumns(left, right) * weights).Sum().Backward();

        Assert.Equal([2, 3], TensorOps.ConcatColumns(left, right).Shape);

        Assert.Equal(10.0, left.Gradient![0, 0], 12);
        Assert.Equal(20.0, left.Gradient![0, 1], 12);
        Assert.Equal(40.0, left.Gradient![1, 0], 12);
        Assert.Equal(50.0, left.Gradient![1, 1], 12);

        Assert.Equal(30.0, right.Gradient![0, 0], 12);
        Assert.Equal(60.0, right.Gradient![1, 0], 12);
    }

    [Fact]
    public void ConcatColumnsGradient_AgreesWithFiniteDifferences()
    {
        // Everything the function closes over is drawn once, outside it. GradientCheck evaluates
        // f many times, so a lambda that draws its own randomness is not the same function twice
        // and the finite differences measure noise.
        var rng = new GraviRandom(41);
        var other = Tensor.Constant(rng.StandardNormal(3, 2));
        var projection = Tensor.Constant(rng.StandardNormal(4, 4)).T();
        var weights = Tensor.Constant(rng.StandardNormal(3, 4));

        var result = GradientCheck.Check(
            a => (TensorOps.ConcatColumns(a, other).MatMul(projection) * weights).Sum(),
            rng.StandardNormal(3, 2));

        Assert.True(result.Passed(1e-6), result.ToString());
    }

    [Fact]
    public void SoftmaxCrossEntropyGradient_AgreesWithFiniteDifferences()
    {
        int[] labels = [2, 0, 1, 1, 0];
        int[] mask = [0, 2, 4];

        var rng = new GraviRandom(37);
        var result = GradientCheck.Check(
            logits => TensorOps.SoftmaxCrossEntropy(logits, labels, mask),
            rng.StandardNormal(5, 3));

        Assert.True(result.Passed(1e-6), result.ToString());
    }

    [Fact]
    public void SoftmaxCrossEntropy_IgnoresRowsOutsideTheMask()
    {
        int[] labels = [0, 1, 2];
        int[] mask = [0, 2];

        var logits = Tensor.Parameter(NdArray.FromArray(new double[,]
        {
            { 2.0, 0.1, 0.1 },
            { 0.0, 5.0, 0.0 },   // unmasked: contributes nothing, gets no gradient
            { 0.1, 0.1, 2.0 },
        }));

        TensorOps.SoftmaxCrossEntropy(logits, labels, mask).Backward();

        for (var c = 0; c < 3; c++)
            Assert.Equal(0.0, logits.Gradient![1, c], 12);

        // The masked rows do get one, and each row's gradient sums to zero because softmax
        // probabilities and the one-hot target both sum to one.
        for (var i = 0; i < 3; i += 2)
        {
            var rowSum = 0.0;
            for (var c = 0; c < 3; c++) rowSum += logits.Gradient![i, c];
            Assert.Equal(0.0, rowSum, 12);
        }
    }

    [Fact]
    public void SoftmaxCrossEntropy_SurvivesLogitsThatWouldOverflowExp()
    {
        // exp(1000) is infinity. Shifting out the row maximum is what keeps this finite.
        var logits = Tensor.Parameter(NdArray.FromArray(new double[,] { { 1000.0, 1002.0 } }));
        var loss = TensorOps.SoftmaxCrossEntropy(logits, [1], [0]);
        loss.Backward();

        Assert.True(double.IsFinite(loss.Item), "loss overflowed");
        Assert.Equal(Math.Log(1 + Math.Exp(-2.0)), loss.Item, 10);
        Assert.True(double.IsFinite(logits.Gradient![0, 0]));
    }

    [Fact]
    public void SoftmaxCrossEntropy_OnAConfidentCorrectPredictionIsNearlyZero()
    {
        var logits = Tensor.Constant(NdArray.FromArray(new double[,] { { 20.0, 0.0, 0.0 } }));
        Assert.True(TensorOps.SoftmaxCrossEntropy(logits, [0], [0]).Item < 1e-8);
    }

    // ---------------------------------------------------------------- graph structure

    [Fact]
    public void BroadcastGradient_SumsOverTheBroadcastAxis()
    {
        // A bias of shape [3] added to a batch of shape [4, 3] influences four outputs, so its
        // gradient is the sum over the batch — not one row, and not the mean. Getting this wrong
        // scales the gradient by the batch size and still looks like it trains.
        var batch = Tensor.Constant(NdArray.Ones(4, 3));
        var bias = Tensor.Parameter(Values(0.5, -1.0, 2.0));

        (batch + bias).Sum().Backward();

        Assert.Equal([3], bias.Gradient!.Shape.ToArray());
        for (var i = 0; i < 3; i++)
            Assert.Equal(4.0, bias.Gradient!.At(i), 12);
    }

    [Fact]
    public void GradientsAccumulateWhenATensorIsUsedTwice()
    {
        // f(x) = x * x through two separate edges into the same node. If the tape overwrote
        // instead of accumulating, this would come back as x rather than 2x.
        var x = Tensor.Parameter(3.0);
        (x * x).Sum().Backward();

        Assert.Equal(6.0, x.Gradient!.At(0), 10);
    }

    [Fact]
    public void DiamondGraph_CountsEveryPath()
    {
        // y = (x + x^2) * x^3, so dy/dx = 4x^3 + 5x^4. At x = 2 that is 32 + 80 = 112.
        var x = Tensor.Parameter(2.0);
        var y = (x + x.Pow(2)) * x.Pow(3);
        y.Sum().Backward();

        Assert.Equal(112.0, x.Gradient!.At(0), 8);
    }

    [Fact]
    public void ConstantsDoNotReceiveGradients()
    {
        var parameter = Tensor.Parameter(2.0);
        var constant = Tensor.Constant(5.0);

        (parameter * constant).Sum().Backward();

        Assert.Equal(5.0, parameter.Gradient!.At(0), 10);
        Assert.Null(constant.Gradient);
    }

    [Fact]
    public void ADeepChainDoesNotOverflowTheStack()
    {
        // A recursive topological sort dies here; the traversal is iterative for this reason.
        var x = Tensor.Parameter(0.01);
        var y = x;
        for (var i = 0; i < 50_000; i++) y += Tensor.Constant(0.0);

        y.Sum().Backward();
        Assert.Equal(1.0, x.Gradient!.At(0), 10);
    }

    [Fact]
    public void BackwardFromANonScalarIsRejected()
    {
        var x = Tensor.Parameter(Values(1.0, 2.0));
        var error = Assert.Throws<InvalidOperationException>(() => (x * x).Backward());
        Assert.Contains("scalar", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void BackwardTwiceGivesTheSameAnswer()
    {
        // Each pass clears first, so running twice must not double the gradient.
        var x = Tensor.Parameter(1.5);
        var y = x.Pow(3).Sum();

        y.Backward();
        var first = x.Gradient!.At(0);
        y.Backward();

        Assert.Equal(first, x.Gradient!.At(0), 12);
        Assert.Equal(3 * 1.5 * 1.5, first, 10);
    }

    // ---------------------------------------------------------------- something end to end

    [Fact]
    public void GradientDescentOnLeastSquaresRecoversTheKnownCoefficients()
    {
        // y = 2x1 - 3x2 + 1, fitted with no derivation by hand anywhere.
        var rng = new GraviRandom(3);
        var x = rng.StandardNormal(200, 2);
        var y = NdArray.Zeros(200, 1);
        for (var i = 0; i < 200; i++)
            y[i, 0] = 2 * x[i, 0] - 3 * x[i, 1] + 1;

        var features = Tensor.Constant(x);
        var target = Tensor.Constant(y);
        var weights = Tensor.Parameter(NdArray.Zeros(2, 1));
        var bias = Tensor.Parameter(NdArray.Zeros(1));

        for (var step = 0; step < 600; step++)
        {
            var error = features.MatMul(weights) + bias - target;
            var loss = (error * error).Mean();
            loss.Backward();

            weights = Tensor.Parameter(UFunc.Subtract(
                weights.Value, UFunc.MultiplyScalar(weights.Gradient!, 0.1)));
            bias = Tensor.Parameter(UFunc.Subtract(
                bias.Value, UFunc.MultiplyScalar(bias.Gradient!, 0.1)));
        }

        Assert.Equal(2.0, weights.Value[0, 0], 4);
        Assert.Equal(-3.0, weights.Value[1, 0], 4);
        Assert.Equal(1.0, bias.Value.At(0), 4);
    }
}
