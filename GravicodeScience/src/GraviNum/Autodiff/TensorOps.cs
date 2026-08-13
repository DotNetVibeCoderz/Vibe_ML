namespace Gravicode.Science.GraviNum.Autodiff;

/// <summary>
/// Differentiable operations on <see cref="Tensor"/>.
/// </summary>
/// <remarks>
/// Each operation computes its forward value with the ordinary <see cref="UFunc"/> and
/// <see cref="LinAlg"/> kernels — the tape adds no numeric machinery of its own — and pairs it
/// with the rule for pushing a gradient back to its inputs.
/// </remarks>
public static class TensorOps
{
    // ================================================================ broadcasting

    /// <summary>
    /// Sums a gradient back down to <paramref name="shape"/>, undoing whatever broadcasting the
    /// forward operation did.
    /// </summary>
    /// <remarks>
    /// This is the step reverse mode is most often wrong about. If a bias of shape <c>[3]</c> is
    /// added to a batch of shape <c>[64, 3]</c>, the forward broadcast copies it 64 times, so the
    /// bias influenced 64 outputs and its gradient is the <em>sum</em> over that axis — not one
    /// row of it, and not the mean. Getting this wrong produces gradients that are quietly a
    /// factor of the batch size out, which trains to something plausible and wrong.
    /// </remarks>
    internal static NdArray ReduceToShape(NdArray gradient, int[] shape)
    {
        if (SameShape(gradient.Shape, shape)) return gradient;

        var result = gradient;

        // Leading axes that the target does not have at all: sum them away.
        while (result.Rank > shape.Length) result = Statistics.Sum(result, axis: 0);

        // Axes the target has as length 1 but that were broadcast wider: sum, keeping the axis.
        for (var axis = 0; axis < shape.Length; axis++)
        {
            if (shape[axis] != 1 || result.Shape[axis] == 1) continue;
            result = Statistics.Sum(result, axis).Reshape(WithAxisAsOne(result.Shape, axis));
        }

        return SameShape(result.Shape, shape) ? result : result.Reshape(shape);
    }

    private static bool SameShape(ReadOnlySpan<int> a, ReadOnlySpan<int> b) => a.SequenceEqual(b);

    private static int[] WithAxisAsOne(ReadOnlySpan<int> shape, int axis)
    {
        var result = shape.ToArray();
        result[axis] = 1;
        return result;
    }

    // ================================================================ arithmetic

    /// <summary>Element-wise sum, with broadcasting.</summary>
    public static Tensor Add(Tensor a, Tensor b)
    {
        var value = UFunc.Add(a.Value, b.Value);
        return Tensor.Derived(value, [a, b], g =>
        {
            // Addition passes the gradient through untouched to both sides.
            a.AccumulateGradient(ReduceToShape(g, a.Shape));
            b.AccumulateGradient(ReduceToShape(g, b.Shape));
        });
    }

    /// <summary>Element-wise difference, with broadcasting.</summary>
    public static Tensor Subtract(Tensor a, Tensor b)
    {
        var value = UFunc.Subtract(a.Value, b.Value);
        return Tensor.Derived(value, [a, b], g =>
        {
            a.AccumulateGradient(ReduceToShape(g, a.Shape));
            b.AccumulateGradient(ReduceToShape(UFunc.MultiplyScalar(g, -1.0), b.Shape));
        });
    }

    /// <summary>Element-wise product, with broadcasting.</summary>
    public static Tensor Multiply(Tensor a, Tensor b)
    {
        var value = UFunc.Multiply(a.Value, b.Value);
        return Tensor.Derived(value, [a, b], g =>
        {
            a.AccumulateGradient(ReduceToShape(UFunc.Multiply(g, b.Value), a.Shape));
            b.AccumulateGradient(ReduceToShape(UFunc.Multiply(g, a.Value), b.Shape));
        });
    }

    /// <summary>Element-wise quotient, with broadcasting.</summary>
    public static Tensor Divide(Tensor a, Tensor b)
    {
        var value = UFunc.Divide(a.Value, b.Value);
        return Tensor.Derived(value, [a, b], g =>
        {
            a.AccumulateGradient(ReduceToShape(UFunc.Divide(g, b.Value), a.Shape));

            // d(a/b)/db = -a/b^2, which is -(value/b) — reusing the forward result costs one
            // multiply instead of squaring b again.
            var wrtB = UFunc.MultiplyScalar(UFunc.Divide(UFunc.Multiply(g, value), b.Value), -1.0);
            b.AccumulateGradient(ReduceToShape(wrtB, b.Shape));
        });
    }

    /// <summary>Negation.</summary>
    public static Tensor Negate(Tensor a)
    {
        var value = UFunc.MultiplyScalar(a.Value, -1.0);
        return Tensor.Derived(value, [a], g => a.AccumulateGradient(UFunc.MultiplyScalar(g, -1.0)));
    }

    // ================================================================ elementwise functions

    /// <summary>Natural exponential.</summary>
    public static Tensor Exp(Tensor a)
    {
        var value = UFunc.Exp(a.Value);
        // d(e^x) = e^x, already computed.
        return Tensor.Derived(value, [a], g => a.AccumulateGradient(UFunc.Multiply(g, value)));
    }

    /// <summary>Natural logarithm.</summary>
    public static Tensor Log(Tensor a)
    {
        var value = UFunc.Log(a.Value);
        return Tensor.Derived(value, [a], g => a.AccumulateGradient(UFunc.Divide(g, a.Value)));
    }

    /// <summary>Square root.</summary>
    public static Tensor Sqrt(Tensor a)
    {
        var value = UFunc.Sqrt(a.Value);
        return Tensor.Derived(value, [a], g =>
            a.AccumulateGradient(UFunc.Divide(g, UFunc.MultiplyScalar(value, 2.0))));
    }

    /// <summary>Raises each element to a constant power.</summary>
    public static Tensor Pow(Tensor a, double exponent)
    {
        var value = UFunc.PowerScalar(a.Value, exponent);
        return Tensor.Derived(value, [a], g =>
        {
            var derivative = UFunc.MultiplyScalar(UFunc.PowerScalar(a.Value, exponent - 1.0), exponent);
            a.AccumulateGradient(UFunc.Multiply(g, derivative));
        });
    }

    /// <summary>Hyperbolic tangent.</summary>
    public static Tensor Tanh(Tensor a)
    {
        var value = UFunc.Tanh(a.Value);
        return Tensor.Derived(value, [a], g =>
        {
            // 1 - tanh^2, from the forward result.
            var derivative = UFunc.Subtract(NdArray.Ones([.. value.Shape]), UFunc.Multiply(value, value));
            a.AccumulateGradient(UFunc.Multiply(g, derivative));
        });
    }

    /// <summary>Logistic sigmoid.</summary>
    public static Tensor Sigmoid(Tensor a)
    {
        var value = Map(a.Value, StableSigmoid);
        return Tensor.Derived(value, [a], g =>
        {
            var derivative = UFunc.Multiply(value, UFunc.Subtract(NdArray.Ones([.. value.Shape]), value));
            a.AccumulateGradient(UFunc.Multiply(g, derivative));
        });
    }

    /// <summary>Rectified linear unit.</summary>
    public static Tensor Relu(Tensor a)
    {
        var value = Map(a.Value, x => x > 0 ? x : 0.0);
        return Tensor.Derived(value, [a], g =>
        {
            var mask = Map(a.Value, x => x > 0 ? 1.0 : 0.0);
            a.AccumulateGradient(UFunc.Multiply(g, mask));
        });
    }

    /// <summary>Absolute value.</summary>
    public static Tensor Abs(Tensor a)
    {
        var value = Map(a.Value, Math.Abs);
        return Tensor.Derived(value, [a], g =>
        {
            // The derivative at exactly zero does not exist; 0 is the usual subgradient choice.
            var sign = Map(a.Value, x => x > 0 ? 1.0 : x < 0 ? -1.0 : 0.0);
            a.AccumulateGradient(UFunc.Multiply(g, sign));
        });
    }

    /// <summary><c>log(1 + exp(x))</c>, computed without overflowing for large <c>x</c>.</summary>
    public static Tensor Softplus(Tensor a)
    {
        var value = Map(a.Value, x => x > 30 ? x : Math.Log(1.0 + Math.Exp(x)));
        return Tensor.Derived(value, [a], g =>
            a.AccumulateGradient(UFunc.Multiply(g, Map(a.Value, StableSigmoid))));
    }

    // ================================================================ reductions

    /// <summary>Sum of every element, as a one-element tensor.</summary>
    public static Tensor Sum(Tensor a)
    {
        var value = NdArray.Scalar(Statistics.Sum(a.Value));
        return Tensor.Derived(value, [a], g =>
        {
            // Every element contributed once, so each gets the same incoming gradient.
            var scale = g.At(0);
            a.AccumulateGradient(NdArray.Full(scale, [.. a.Shape]));
        });
    }

    /// <summary>Mean of every element, as a one-element tensor.</summary>
    public static Tensor Mean(Tensor a)
    {
        var count = a.Size;
        var value = NdArray.Scalar(Statistics.Sum(a.Value) / count);
        return Tensor.Derived(value, [a], g =>
            a.AccumulateGradient(NdArray.Full(g.At(0) / count, [.. a.Shape])));
    }

    /// <summary>Sum along one axis.</summary>
    public static Tensor Sum(Tensor a, int axis)
    {
        var value = Statistics.Sum(a.Value, axis);
        return Tensor.Derived(value, [a], g =>
        {
            // Re-insert the reduced axis so the gradient broadcasts back over it.
            var expanded = g.Reshape([.. WithAxisAsOne(a.Shape, axis)]);
            a.AccumulateGradient(expanded.BroadcastTo([.. a.Shape]).Copy());
        });
    }

    /// <summary>
    /// <c>log(sum(exp(x)))</c> over every element, computed by shifting out the maximum.
    /// </summary>
    /// <remarks>
    /// Written directly rather than composed from <see cref="Log"/> and <see cref="Exp"/> because
    /// the naive form overflows as soon as any element exceeds about 710. The gradient is the
    /// softmax, which falls out of the same shifted exponentials.
    /// </remarks>
    public static Tensor LogSumExp(Tensor a)
    {
        var max = Statistics.Max(a.Value);
        var shifted = UFunc.Exp(UFunc.AddScalar(a.Value, -max));
        var total = Statistics.Sum(shifted);
        var value = NdArray.Scalar(max + Math.Log(total));

        return Tensor.Derived(value, [a], g =>
        {
            var softmax = UFunc.MultiplyScalar(shifted, g.At(0) / total);
            a.AccumulateGradient(softmax);
        });
    }

    // ================================================================ linear algebra

    /// <summary>Matrix product.</summary>
    public static Tensor MatMul(Tensor a, Tensor b)
    {
        var value = LinAlg.Dot(a.Value, b.Value);
        return Tensor.Derived(value, [a, b], g =>
        {
            a.AccumulateGradient(LinAlg.Dot(g, b.Value.T));
            b.AccumulateGradient(LinAlg.Dot(a.Value.T, g));
        });
    }

    /// <summary>
    /// Multiplies a constant sparse matrix by a dense tensor: <c>A x</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the message-passing step of a graph neural network. The sparse operand is the
    /// normalised adjacency, which is structure rather than a parameter, so no gradient flows to
    /// it — only to <paramref name="x"/>, and that gradient is <c>A^T g</c>.
    /// </para>
    /// <para>
    /// The transpose is taken explicitly rather than assumed away. A symmetrically normalised
    /// adjacency is its own transpose, so it would be free to skip; a directed or row-normalised
    /// one is not, and silently training the wrong gradient is the kind of bug that still looks
    /// like it converges.
    /// </para>
    /// </remarks>
    public static Tensor SparseMatMul(SparseMatrix a, Tensor x)
    {
        var value = a.Multiply(x.Value, denseIsMatrix: true);

        // Built once here rather than per backward pass: training calls this every epoch.
        var transpose = a.Transpose();

        return Tensor.Derived(value, [x], g =>
            x.AccumulateGradient(transpose.Multiply(g, denseIsMatrix: true)));
    }

    /// <summary>
    /// Mean softmax cross-entropy over the rows named by <paramref name="mask"/>.
    /// </summary>
    /// <param name="logits">Row per sample, column per class.</param>
    /// <param name="labels">Class index per row; rows outside the mask are ignored.</param>
    /// <param name="mask">Rows that contribute to the loss — the training set in a GNN.</param>
    /// <remarks>
    /// <para>
    /// Fused rather than composed from <c>Log</c> and <c>Exp</c>, for the same two reasons every
    /// framework fuses it. Numerically, the softmax and the log cancel, so the fused form shifts
    /// out the row maximum and never exponentiates a large positive number. Structurally, the
    /// gradient collapses to <c>(p - onehot) / |mask|</c>, which is one subtraction per element
    /// instead of a graph of intermediate tensors.
    /// </para>
    /// <para>
    /// Masking is what makes this usable for semi-supervised node classification: the forward pass
    /// runs over the whole graph, because every node's features inform its neighbours, while only
    /// the labelled nodes contribute to the loss.
    /// </para>
    /// </remarks>
    public static Tensor SoftmaxCrossEntropy(Tensor logits, IReadOnlyList<int> labels, IReadOnlyList<int> mask)
    {
        if (logits.Shape.Length != 2)
            throw new ArgumentException($"SoftmaxCrossEntropy expects a rank 2 tensor, got rank {logits.Shape.Length}.");

        var rows = logits.Shape[0];
        var classes = logits.Shape[1];
        var count = Math.Max(1, mask.Count);

        var probabilities = new double[rows, classes];
        var loss = 0.0;

        foreach (var i in mask)
        {
            var max = double.NegativeInfinity;
            for (var c = 0; c < classes; c++) max = Math.Max(max, logits.Value[i, c]);

            var total = 0.0;
            for (var c = 0; c < classes; c++)
            {
                var e = Math.Exp(logits.Value[i, c] - max);
                probabilities[i, c] = e;
                total += e;
            }

            for (var c = 0; c < classes; c++) probabilities[i, c] /= total;

            // log p[label] = (logit - max) - log(sum exp(logit - max)), with nothing large exponentiated.
            loss -= logits.Value[i, labels[i]] - max - Math.Log(total);
        }

        return Tensor.Derived(NdArray.Scalar(loss / count), [logits], g =>
        {
            var scale = g.At(0) / count;
            var gradient = NdArray.Zeros(rows, classes);

            // Rows outside the mask keep a zero gradient: they took no part in the loss.
            foreach (var i in mask)
                for (var c = 0; c < classes; c++)
                    gradient[i, c] = scale * (probabilities[i, c] - (c == labels[i] ? 1.0 : 0.0));

            logits.AccumulateGradient(gradient);
        });
    }

    /// <summary>
    /// Selects rows by index: <c>result[e] = a[rows[e]]</c>.
    /// </summary>
    /// <remarks>
    /// The edge-list half of a graph layer. An attention model needs each edge's endpoint
    /// features side by side, which means reading the same node row once per incident edge — so
    /// the backward pass has to <em>add</em> each edge's gradient back into its node, not assign
    /// it. <see cref="SegmentSum"/> is this operation's adjoint: gathering forwards is summing
    /// backwards, and vice versa.
    /// </remarks>
    public static Tensor Gather(Tensor a, IReadOnlyList<int> rows)
    {
        if (a.Shape.Length != 2) throw new ArgumentException("Gather expects a rank 2 tensor.");

        var columns = a.Shape[1];
        var value = NdArray.Zeros(rows.Count, columns);

        for (var e = 0; e < rows.Count; e++)
            for (var j = 0; j < columns; j++)
                value[e, j] = a.Value[rows[e], j];

        return Tensor.Derived(value, [a], g =>
        {
            var gradient = NdArray.Zeros(a.Shape[0], columns);
            for (var e = 0; e < rows.Count; e++)
                for (var j = 0; j < columns; j++)
                    gradient[rows[e], j] += g[e, j];

            a.AccumulateGradient(gradient);
        });
    }

    /// <summary>
    /// Adds rows into groups: <c>result[segments[e]] += a[e]</c>.
    /// </summary>
    /// <param name="a">One row per edge.</param>
    /// <param name="segments">Which output row each input row belongs to.</param>
    /// <param name="count">Number of output rows; groups with no members stay zero.</param>
    /// <remarks>
    /// The scatter half of a graph layer, and the exact adjoint of <see cref="Gather"/> — which is
    /// why its backward pass is a gather. Segments that no edge points at come out as zero rather
    /// than undefined, which is what an isolated node should contribute.
    /// </remarks>
    public static Tensor SegmentSum(Tensor a, IReadOnlyList<int> segments, int count)
    {
        if (a.Shape.Length != 2) throw new ArgumentException("SegmentSum expects a rank 2 tensor.");
        if (segments.Count != a.Shape[0])
            throw new ArgumentException($"Got {segments.Count} segment ids for {a.Shape[0]} rows.");

        var columns = a.Shape[1];
        var value = NdArray.Zeros(count, columns);

        for (var e = 0; e < segments.Count; e++)
            for (var j = 0; j < columns; j++)
                value[segments[e], j] += a.Value[e, j];

        return Tensor.Derived(value, [a], g =>
        {
            var gradient = NdArray.Zeros(segments.Count, columns);
            for (var e = 0; e < segments.Count; e++)
                for (var j = 0; j < columns; j++)
                    gradient[e, j] = g[segments[e], j];

            a.AccumulateGradient(gradient);
        });
    }

    /// <summary>Leaky rectified linear unit.</summary>
    /// <remarks>
    /// GAT scores its edges through this rather than a plain ReLU: a hard zero would kill the
    /// gradient for every edge scoring negative, and on a sparse graph that is most of them.
    /// </remarks>
    public static Tensor LeakyRelu(Tensor a, double slope = 0.2)
    {
        var value = Map(a.Value, x => x > 0 ? x : slope * x);
        return Tensor.Derived(value, [a], g =>
        {
            var mask = Map(a.Value, x => x > 0 ? 1.0 : slope);
            a.AccumulateGradient(UFunc.Multiply(g, mask));
        });
    }

    /// <summary>
    /// Takes a contiguous block of columns: <c>a[:, start .. start + count]</c>.
    /// </summary>
    /// <remarks>
    /// The inverse of <see cref="ConcatColumns"/>, and what lets multi-head attention split one
    /// wide projection into per-head slices. The backward pass writes the gradient back into its
    /// own columns and leaves the rest at zero, which is correct precisely because the untaken
    /// columns had no influence on the result.
    /// </remarks>
    public static Tensor SliceColumns(Tensor a, int start, int count)
    {
        if (a.Shape.Length != 2) throw new ArgumentException("SliceColumns expects a rank 2 tensor.");
        if (start < 0 || count < 0 || start + count > a.Shape[1])
            throw new ArgumentOutOfRangeException(nameof(start),
                $"Columns [{start}, {start + count}) do not fit in a matrix {a.Shape[1]} wide.");

        var rows = a.Shape[0];
        var value = NdArray.Zeros(rows, count);
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < count; j++)
                value[i, j] = a.Value[i, start + j];

        return Tensor.Derived(value, [a], g =>
        {
            var gradient = NdArray.Zeros(rows, a.Shape[1]);
            for (var i = 0; i < rows; i++)
                for (var j = 0; j < count; j++)
                    gradient[i, start + j] = g[i, j];

            a.AccumulateGradient(gradient);
        });
    }

    /// <summary>
    /// Joins two matrices side by side: <c>[a b]</c>.
    /// </summary>
    /// <remarks>
    /// GraphSAGE keeps "what I am" and "what surrounds me" in separate halves of the layer input
    /// rather than summing them, so the layer can weight the two independently. The backward pass
    /// is the inverse: each operand takes back the slice of the gradient that came from its own
    /// columns.
    /// </remarks>
    public static Tensor ConcatColumns(Tensor a, Tensor b)
    {
        if (a.Shape.Length != 2 || b.Shape.Length != 2)
            throw new ArgumentException("ConcatColumns expects two rank 2 tensors.");
        if (a.Shape[0] != b.Shape[0])
            throw new ArgumentException($"Row counts differ: {a.Shape[0]} and {b.Shape[0]}.");

        var rows = a.Shape[0];
        var left = a.Shape[1];
        var right = b.Shape[1];

        var value = NdArray.Zeros(rows, left + right);
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < left; j++) value[i, j] = a.Value[i, j];
            for (var j = 0; j < right; j++) value[i, left + j] = b.Value[i, j];
        }

        return Tensor.Derived(value, [a, b], g =>
        {
            var gradientA = NdArray.Zeros(rows, left);
            var gradientB = NdArray.Zeros(rows, right);

            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < left; j++) gradientA[i, j] = g[i, j];
                for (var j = 0; j < right; j++) gradientB[i, j] = g[i, left + j];
            }

            a.AccumulateGradient(gradientA);
            b.AccumulateGradient(gradientB);
        });
    }

    /// <summary>Matrix transpose.</summary>
    public static Tensor Transpose(Tensor a)
    {
        var value = a.Value.T.Copy();
        return Tensor.Derived(value, [a], g => a.AccumulateGradient(g.T.Copy()));
    }

    /// <summary>Reshapes without moving data.</summary>
    public static Tensor Reshape(Tensor a, params int[] shape)
    {
        var value = a.Value.Reshape(shape);
        return Tensor.Derived(value, [a], g => a.AccumulateGradient(g.Reshape([.. a.Shape])));
    }

    /// <summary>One element, as a one-element tensor.</summary>
    public static Tensor At(Tensor a, int flatIndex)
    {
        var value = NdArray.Scalar(a.Value.At(flatIndex));
        return Tensor.Derived(value, [a], g =>
        {
            // Only the selected element was used, so everything else gets zero.
            var full = NdArray.Zeros([.. a.Shape]);
            full.SetAt(flatIndex, g.At(0));
            a.AccumulateGradient(full);
        });
    }

    // ================================================================ helpers

    private static NdArray Map(NdArray a, Func<double, double> f)
    {
        var source = a.AsContiguous().ToArray();
        var result = new double[source.Length];
        for (var i = 0; i < source.Length; i++) result[i] = f(source[i]);
        return new NdArray(result, [.. a.Shape]);
    }

    /// <summary>Sigmoid written so neither tail overflows.</summary>
    private static double StableSigmoid(double x)
    {
        if (x >= 0)
        {
            var z = Math.Exp(-x);
            return 1.0 / (1.0 + z);
        }

        var e = Math.Exp(x);
        return e / (1.0 + e);
    }
}
