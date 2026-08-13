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
