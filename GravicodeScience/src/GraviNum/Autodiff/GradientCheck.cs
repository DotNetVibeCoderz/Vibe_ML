namespace Gravicode.Science.GraviNum.Autodiff;

/// <summary>The worst disagreement found between an analytic gradient and a numeric one.</summary>
/// <param name="MaxRelativeError">Largest relative difference over all checked entries.</param>
/// <param name="Analytic">The autodiff gradient at the worst entry.</param>
/// <param name="Numeric">The finite-difference gradient at the worst entry.</param>
/// <param name="Index">Flat index of the worst entry.</param>
public readonly record struct GradientCheckResult(
    double MaxRelativeError, double Analytic, double Numeric, int Index)
{
    /// <summary>Whether the gradients agree to <paramref name="tolerance"/>.</summary>
    public bool Passed(double tolerance = 1e-6) => MaxRelativeError <= tolerance;

    /// <inheritdoc />
    public override string ToString() =>
        $"max relative error {MaxRelativeError:E3} at [{Index}] (autodiff {Analytic:G8}, numeric {Numeric:G8})";
}

/// <summary>
/// Checks a tape gradient against central finite differences.
/// </summary>
/// <remarks>
/// <para>
/// This is how every gradient in the library is verified. Finite differences are too slow and too
/// imprecise to <em>use</em> — that is exactly why the tape exists — but they depend on nothing
/// the tape does, which makes them a genuinely independent check rather than a restatement.
/// </para>
/// <para>
/// Central differences are accurate to <c>O(h²)</c> and lose roughly half of double precision to
/// cancellation, so agreement to about <c>1e-6</c> relative is as much as can be asked; a
/// mismatch far above that is a real bug in a backward rule.
/// </para>
/// </remarks>
public static class GradientCheck
{
    /// <summary>
    /// Compares the autodiff gradient of <paramref name="f"/> at <paramref name="input"/> with a
    /// central finite-difference estimate.
    /// </summary>
    /// <param name="f">Builds a one-element tensor from the parameter being differentiated.</param>
    /// <param name="input">The point to check at.</param>
    /// <param name="step">Finite-difference step; the default trades truncation against cancellation.</param>
    public static GradientCheckResult Check(Func<Tensor, Tensor> f, NdArray input, double step = 1e-5)
    {
        var parameter = Tensor.Parameter(input.Copy());
        var output = f(parameter);

        if (output.Size != 1)
            throw new ArgumentException($"The function must return a single value, got {output.Size}.");

        output.Backward();
        var analytic = parameter.Gradient
            ?? throw new InvalidOperationException("No gradient reached the input; it is not used by f.");

        var worst = 0.0;
        var worstIndex = 0;
        var worstAnalytic = 0.0;
        var worstNumeric = 0.0;

        for (var i = 0; i < input.Size; i++)
        {
            var forward = input.Copy();
            forward.SetAt(i, forward.At(i) + step);

            var backward = input.Copy();
            backward.SetAt(i, backward.At(i) - step);

            var numeric = (f(Tensor.Constant(forward)).Item - f(Tensor.Constant(backward)).Item) / (2 * step);
            var exact = analytic.At(i);

            // Relative to the larger magnitude, with a floor so two near-zero values agreeing
            // does not divide by nothing.
            var scale = Math.Max(1.0, Math.Max(Math.Abs(numeric), Math.Abs(exact)));
            var error = Math.Abs(numeric - exact) / scale;

            if (error <= worst) continue;
            worst = error;
            worstIndex = i;
            worstAnalytic = exact;
            worstNumeric = numeric;
        }

        return new GradientCheckResult(worst, worstAnalytic, worstNumeric, worstIndex);
    }

    /// <summary>Checks a scalar function of a scalar.</summary>
    public static GradientCheckResult Check(Func<Tensor, Tensor> f, double input, double step = 1e-5)
        => Check(f, NdArray.Scalar(input), step);
}
