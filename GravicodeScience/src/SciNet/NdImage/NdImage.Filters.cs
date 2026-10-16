using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.NdImageSupport;

namespace Gravicode.Science.SciNet;

/// <summary>
/// scipy.ndimage for .NET: n-dimensional filters, spline interpolation, morphology, distance
/// transforms and region measurements over GraviNum <see cref="NdArray"/> images.
/// </summary>
/// <remarks>
/// <para>
/// Boundary handling reproduces scipy's to the bit: the extension index maps are those of
/// <c>NI_ExtendLine</c>, and 1-D correlation sums in scipy's order (including its symmetric-kernel
/// pairing), so results agree with scipy 1.17.1 to rounding rather than approximately. The arrays
/// are double-only, so boolean images are 0/1 doubles and any non-zero value reads as true.
/// </para>
/// <para>
/// Output is always a new float64 array; scipy keeps the input dtype (an integer image filtered
/// by scipy is truncated), so compare against scipy with float input.
/// </para>
/// </remarks>
public static partial class NdImage
{
    // ------------------------------------------------------------------ 1-D correlation

    /// <summary>
    /// Correlates along one axis: <c>out[k] = Σ_j w[j] · in[k + j − len/2 − origin]</c> (scipy.ndimage.correlate1d).
    /// </summary>
    /// <param name="input">Input array of any rank.</param>
    /// <param name="weights">1-D weights.</param>
    /// <param name="axis">Axis to filter along (negative counts from the end).</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value for <see cref="NdImageMode.Constant"/>.</param>
    /// <param name="origin">Shift of the kernel centre, in <c>[-(len/2), (len-1)/2]</c>.</param>
    public static NdArray Correlate1d(NdArray input, double[] weights, int axis = -1, NdImageMode mode = NdImageMode.Reflect,
        double cval = 0.0, int origin = 0)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(weights);
        if (weights.Length < 1) throw new ArgumentException("No filter weights given.");
        var shape = ShapeOf(input);
        var ax = NormalizeAxis(axis, shape.Length);
        var data = input.ToArray();
        Correlate1dInPlace(data, shape, weights, ax, FilterMode(mode), cval, origin);
        return new NdArray(data, shape);
    }

    /// <summary>
    /// Convolves along one axis (scipy.ndimage.convolve1d): the weights are reversed and the origin
    /// negated, with the extra shift scipy applies to even-length kernels.
    /// </summary>
    /// <param name="input">Input array of any rank.</param>
    /// <param name="weights">1-D weights.</param>
    /// <param name="axis">Axis to filter along.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value for <see cref="NdImageMode.Constant"/>.</param>
    /// <param name="origin">Kernel origin.</param>
    public static NdArray Convolve1d(NdArray input, double[] weights, int axis = -1, NdImageMode mode = NdImageMode.Reflect,
        double cval = 0.0, int origin = 0)
    {
        ArgumentNullException.ThrowIfNull(weights);
        var w = weights.Reverse().ToArray();
        var o = -origin;
        if ((w.Length & 1) == 0) o -= 1;
        return Correlate1d(input, w, axis, mode, cval, o);
    }

    internal static void Correlate1dInPlace(double[] data, int[] shape, double[] weights, int axis, NdImageMode mode, double cval, int origin)
    {
        var size1 = weights.Length / 2;
        var size2 = weights.Length - size1 - 1;
        CheckOrigins([weights.Length], [origin]);
        AlongAxis(data, data, shape, axis, size1 + origin, size2 - origin, mode, cval, new NdImageCorrelateLineOp(weights));
    }

    // ------------------------------------------------------------------ n-D correlation

    /// <summary>
    /// n-D correlation with a weights array of the same rank (scipy.ndimage.correlate). As in
    /// scipy, weights with magnitude ≤ DBL_EPSILON are left out of the footprint entirely.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="weights">Weights, same rank as the input.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value for <see cref="NdImageMode.Constant"/>.</param>
    /// <param name="origin">Per-axis origin (one value broadcasts); null for zero.</param>
    public static NdArray Correlate(NdArray input, NdArray weights, NdImageMode mode = NdImageMode.Reflect, double cval = 0.0,
        int[]? origin = null)
        => CorrelateOrConvolve(input, weights, mode, cval, origin, convolve: false);

    /// <summary>
    /// n-D convolution (scipy.ndimage.convolve): correlation with the weights reversed on every
    /// axis and the origin negated (shifted by one more for even extents).
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="weights">Weights, same rank as the input.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value for <see cref="NdImageMode.Constant"/>.</param>
    /// <param name="origin">Per-axis origin; null for zero.</param>
    public static NdArray Convolve(NdArray input, NdArray weights, NdImageMode mode = NdImageMode.Reflect, double cval = 0.0,
        int[]? origin = null)
        => CorrelateOrConvolve(input, weights, mode, cval, origin, convolve: true);

    private static NdArray CorrelateOrConvolve(NdArray input, NdArray weights, NdImageMode mode, double cval, int[]? origin, bool convolve)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(weights);
        var shape = ShapeOf(input);
        var rank = shape.Length;
        if (weights.Rank != rank) throw new ArgumentException($"Weights have rank {weights.Rank}; the input has rank {rank}.");
        var wshape = ShapeOf(weights);
        var w = weights.ToArray();
        var origins = Expand(origin, rank, 0, nameof(origin));
        if (convolve)
        {
            Array.Reverse(w); // reversing the flat buffer reverses every axis
            for (var a = 0; a < rank; a++)
            {
                origins[a] = -origins[a];
                if ((wshape[a] & 1) == 0) origins[a] -= 1;
            }
        }
        CheckOrigins(wshape, origins);
        var fp = new bool[w.Length];
        var nz = new List<double>();
        for (var i = 0; i < w.Length; i++)
        {
            if (Math.Abs(w[i]) > 2.220446049250313e-16) { fp[i] = true; nz.Add(w[i]); }
        }
        var data = input.ToArray();
        if (nz.Count == 0) return new NdArray(new double[data.Length], shape);
        var result = FootprintFilter(data, shape, wshape, fp, origins, FilterMode(mode), cval, new NdImageCorrelateWindowOp(nz.ToArray()));
        return new NdArray(result, shape);
    }

    // ------------------------------------------------------------------ Gaussian

    /// <summary>
    /// The sampled Gaussian (or its <paramref name="order"/>-th derivative) exactly as scipy's
    /// <c>_gaussian_kernel1d</c>: normalised by a pairwise sum, derivatives from the polynomial
    /// recurrence <c>q ← (D + P) q</c>. Not reversed — <see cref="GaussianFilter1d"/> reverses it.
    /// </summary>
    /// <param name="sigma">Standard deviation.</param>
    /// <param name="order">Derivative order, 0 or more.</param>
    /// <param name="radius">Half-width; the kernel has <c>2·radius + 1</c> taps.</param>
    public static double[] GaussianKernel1d(double sigma, int order, int radius)
    {
        if (order < 0) throw new ArgumentException("Order must be non-negative.");
        if (radius < 0) throw new ArgumentException("Radius must be non-negative.");
        var len = 2 * radius + 1;
        var sigma2 = sigma * sigma;
        var phi = new double[len];
        var factor = -0.5 / sigma2;
        for (var i = 0; i < len; i++)
        {
            double x = i - radius;
            phi[i] = Math.Exp(factor * (x * x));
        }
        var sum = PairwiseSum(phi);
        for (var i = 0; i < len; i++) phi[i] /= sum;
        if (order == 0) return phi;

        var q = new double[order + 1];
        q[0] = 1.0;
        var p = 1.0 / -sigma2;
        var next = new double[order + 1];
        for (var it = 0; it < order; it++)
        {
            for (var i = 0; i <= order; i++)
            {
                var v = 0.0;
                if (i >= 1) v += p * q[i - 1];
                if (i + 1 <= order) v += (i + 1) * q[i + 1];
                next[i] = v;
            }
            (q, next) = (next, q);
        }
        var result = new double[len];
        for (var i = 0; i < len; i++)
        {
            double x = i - radius;
            var poly = 0.0;
            var xp = 1.0;
            for (var k = 0; k <= order; k++)
            {
                poly += xp * q[k];
                xp *= x;
            }
            result[i] = poly * phi[i];
        }
        return result;
    }

    /// <summary>
    /// 1-D Gaussian filter or derivative (scipy.ndimage.gaussian_filter1d). The radius is
    /// <c>(int)(truncate·sigma + 0.5)</c> unless <paramref name="radius"/> is given.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="sigma">Standard deviation in samples.</param>
    /// <param name="axis">Axis to filter along.</param>
    /// <param name="order">0 smooths; 1–3 give derivatives of the Gaussian.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value for <see cref="NdImageMode.Constant"/>.</param>
    /// <param name="truncate">Kernel half-width in standard deviations.</param>
    /// <param name="radius">Explicit half-width, overriding <paramref name="truncate"/>.</param>
    public static NdArray GaussianFilter1d(NdArray input, double sigma, int axis = -1, int order = 0,
        NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, double truncate = 4.0, int? radius = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var shape = ShapeOf(input);
        var ax = NormalizeAxis(axis, shape.Length);
        var data = input.ToArray();
        Gaussian1dInPlace(data, shape, sigma, ax, order, FilterMode(mode), cval, truncate, radius);
        return new NdArray(data, shape);
    }

    private static void Gaussian1dInPlace(double[] data, int[] shape, double sigma, int axis, int order, NdImageMode mode,
        double cval, double truncate, int? radius)
    {
        var lw = radius ?? (int)(truncate * sigma + 0.5);
        if (lw < 0) throw new ArgumentException("Radius must be a non-negative integer.");
        var w = GaussianKernel1d(sigma, order, lw);
        Array.Reverse(w);
        Correlate1dInPlace(data, shape, w, axis, mode, cval, 0);
    }

    /// <summary>
    /// Separable n-D Gaussian filter (scipy.ndimage.gaussian_filter): one 1-D pass per axis whose
    /// sigma exceeds 1e-15, in axis order.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="sigma">Standard deviation per axis (one value broadcasts).</param>
    /// <param name="order">Derivative order per axis (null for zero).</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value for <see cref="NdImageMode.Constant"/>.</param>
    /// <param name="truncate">Kernel half-width in standard deviations.</param>
    /// <param name="radius">Explicit per-axis half-widths (null to use <paramref name="truncate"/>).</param>
    public static NdArray GaussianFilter(NdArray input, double[] sigma, int[]? order = null, NdImageMode mode = NdImageMode.Reflect,
        double cval = 0.0, double truncate = 4.0, int[]? radius = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var shape = ShapeOf(input);
        var rank = shape.Length;
        var s = Expand(sigma, rank, 0.0, nameof(sigma));
        var o = Expand(order, rank, 0, nameof(order));
        var r = radius is null ? null : Expand(radius, rank, 0, nameof(radius));
        var data = input.ToArray();
        var m = FilterMode(mode);
        for (var a = 0; a < rank; a++)
            if (s[a] > 1e-15)
                Gaussian1dInPlace(data, shape, s[a], a, o[a], m, cval, truncate, r?[a]);
        return new NdArray(data, shape);
    }

    /// <summary>Isotropic Gaussian filter (same sigma and derivative order on every axis).</summary>
    /// <param name="input">Input array.</param>
    /// <param name="sigma">Standard deviation.</param>
    /// <param name="order">Derivative order applied on every axis.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value for <see cref="NdImageMode.Constant"/>.</param>
    /// <param name="truncate">Kernel half-width in standard deviations.</param>
    /// <param name="radius">Explicit half-width.</param>
    public static NdArray GaussianFilter(NdArray input, double sigma, int order = 0, NdImageMode mode = NdImageMode.Reflect,
        double cval = 0.0, double truncate = 4.0, int? radius = null)
        => GaussianFilter(input, [sigma], [order], mode, cval, truncate, radius is null ? null : [radius.Value]);

    /// <summary>
    /// Gradient magnitude with Gaussian derivatives (scipy.ndimage.gaussian_gradient_magnitude):
    /// <c>sqrt(Σ_axis (∂G/∂axis * input)²)</c>.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="sigma">Standard deviation per axis (one value broadcasts).</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value for <see cref="NdImageMode.Constant"/>.</param>
    /// <param name="truncate">Kernel half-width in standard deviations.</param>
    public static NdArray GaussianGradientMagnitude(NdArray input, double[] sigma, NdImageMode mode = NdImageMode.Reflect,
        double cval = 0.0, double truncate = 4.0)
    {
        ArgumentNullException.ThrowIfNull(input);
        var rank = input.Rank;
        return GenericGradientMagnitude(input, (x, axis, m, c) =>
        {
            var order = new int[rank];
            order[axis] = 1;
            return GaussianFilter(x, sigma, order, m, c, truncate);
        }, mode, cval);
    }

    /// <summary>Isotropic overload of <see cref="GaussianGradientMagnitude(NdArray, double[], NdImageMode, double, double)"/>.</summary>
    /// <param name="input">Input array.</param>
    /// <param name="sigma">Standard deviation.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="truncate">Kernel half-width in standard deviations.</param>
    public static NdArray GaussianGradientMagnitude(NdArray input, double sigma, NdImageMode mode = NdImageMode.Reflect,
        double cval = 0.0, double truncate = 4.0)
        => GaussianGradientMagnitude(input, [sigma], mode, cval, truncate);

    /// <summary>Laplacian of Gaussian (scipy.ndimage.gaussian_laplace): sum of second Gaussian derivatives.</summary>
    /// <param name="input">Input array.</param>
    /// <param name="sigma">Standard deviation per axis (one value broadcasts).</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="truncate">Kernel half-width in standard deviations.</param>
    public static NdArray GaussianLaplace(NdArray input, double[] sigma, NdImageMode mode = NdImageMode.Reflect,
        double cval = 0.0, double truncate = 4.0)
    {
        ArgumentNullException.ThrowIfNull(input);
        var rank = input.Rank;
        return GenericLaplace(input, (x, axis, m, c) =>
        {
            var order = new int[rank];
            order[axis] = 2;
            return GaussianFilter(x, sigma, order, m, c, truncate);
        }, mode, cval);
    }

    /// <summary>Isotropic overload of <see cref="GaussianLaplace(NdArray, double[], NdImageMode, double, double)"/>.</summary>
    /// <param name="input">Input array.</param>
    /// <param name="sigma">Standard deviation.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="truncate">Kernel half-width in standard deviations.</param>
    public static NdArray GaussianLaplace(NdArray input, double sigma, NdImageMode mode = NdImageMode.Reflect,
        double cval = 0.0, double truncate = 4.0)
        => GaussianLaplace(input, [sigma], mode, cval, truncate);

    // ------------------------------------------------------------------ derivatives

    /// <summary>
    /// Sum of second derivatives along every axis (scipy.ndimage.generic_laplace), accumulated
    /// in axis order.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="derivative2">Second derivative along one axis.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    public static NdArray GenericLaplace(NdArray input, NdImageDerivative derivative2, NdImageMode mode = NdImageMode.Reflect, double cval = 0.0)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(derivative2);
        var shape = ShapeOf(input);
        var acc = derivative2(input, 0, mode, cval).ToArray();
        for (var a = 1; a < shape.Length; a++)
        {
            var t = derivative2(input, a, mode, cval).AsContiguous().AsSpan();
            for (var i = 0; i < acc.Length; i++) acc[i] += t[i];
        }
        return new NdArray(acc, shape);
    }

    /// <summary>
    /// <c>sqrt(Σ_axis derivative(input, axis)²)</c> (scipy.ndimage.generic_gradient_magnitude).
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="derivative">First derivative along one axis.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    public static NdArray GenericGradientMagnitude(NdArray input, NdImageDerivative derivative, NdImageMode mode = NdImageMode.Reflect, double cval = 0.0)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(derivative);
        var shape = ShapeOf(input);
        var acc = derivative(input, 0, mode, cval).ToArray();
        for (var i = 0; i < acc.Length; i++) acc[i] *= acc[i];
        for (var a = 1; a < shape.Length; a++)
        {
            var t = derivative(input, a, mode, cval).AsContiguous().AsSpan();
            for (var i = 0; i < acc.Length; i++) acc[i] += t[i] * t[i];
        }
        for (var i = 0; i < acc.Length; i++) acc[i] = Math.Sqrt(acc[i]);
        return new NdArray(acc, shape);
    }

    /// <summary>Sobel derivative along <paramref name="axis"/>: <c>[-1, 0, 1]</c> there, <c>[1, 2, 1]</c> smoothing on every other axis.</summary>
    /// <param name="input">Input array.</param>
    /// <param name="axis">Derivative axis.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    public static NdArray Sobel(NdArray input, int axis = -1, NdImageMode mode = NdImageMode.Reflect, double cval = 0.0)
        => DerivativeWithSmoothing(input, axis, mode, cval, [1.0, 2.0, 1.0]);

    /// <summary>Prewitt derivative along <paramref name="axis"/>: <c>[-1, 0, 1]</c> there, <c>[1, 1, 1]</c> on every other axis.</summary>
    /// <param name="input">Input array.</param>
    /// <param name="axis">Derivative axis.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    public static NdArray Prewitt(NdArray input, int axis = -1, NdImageMode mode = NdImageMode.Reflect, double cval = 0.0)
        => DerivativeWithSmoothing(input, axis, mode, cval, [1.0, 1.0, 1.0]);

    private static NdArray DerivativeWithSmoothing(NdArray input, int axis, NdImageMode mode, double cval, double[] smooth)
    {
        ArgumentNullException.ThrowIfNull(input);
        var shape = ShapeOf(input);
        var ax = NormalizeAxis(axis, shape.Length);
        var m = FilterMode(mode);
        var data = input.ToArray();
        Correlate1dInPlace(data, shape, [-1.0, 0.0, 1.0], ax, m, cval, 0);
        for (var a = 0; a < shape.Length; a++)
            if (a != ax) Correlate1dInPlace(data, shape, smooth, a, m, cval, 0);
        return new NdArray(data, shape);
    }

    /// <summary>Discrete Laplacian: <c>[1, -2, 1]</c> along each axis, summed (scipy.ndimage.laplace).</summary>
    /// <param name="input">Input array.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    public static NdArray Laplace(NdArray input, NdImageMode mode = NdImageMode.Reflect, double cval = 0.0)
        => GenericLaplace(input, static (x, axis, m, c) => Correlate1d(x, [1.0, -2.0, 1.0], axis, m, c), mode, cval);

    // ------------------------------------------------------------------ uniform

    /// <summary>
    /// Moving average along one axis (scipy.ndimage.uniform_filter1d), O(1) per sample via a running sum.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Window length (≥ 1).</param>
    /// <param name="axis">Axis to filter along.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Window origin.</param>
    public static NdArray UniformFilter1d(NdArray input, int size, int axis = -1, NdImageMode mode = NdImageMode.Reflect,
        double cval = 0.0, int origin = 0)
    {
        ArgumentNullException.ThrowIfNull(input);
        var shape = ShapeOf(input);
        var ax = NormalizeAxis(axis, shape.Length);
        var data = input.ToArray();
        Uniform1dInPlace(data, shape, size, ax, FilterMode(mode), cval, origin);
        return new NdArray(data, shape);
    }

    private static void Uniform1dInPlace(double[] data, int[] shape, int size, int axis, NdImageMode mode, double cval, int origin)
    {
        if (size < 1) throw new ArgumentException("Incorrect filter size.");
        CheckOrigins([size], [origin]);
        var before = size / 2 + origin;
        AlongAxis(data, data, shape, axis, before, size - 1 - before, mode, cval, new NdImageUniformLineOp(size));
    }

    /// <summary>Separable n-D moving average (scipy.ndimage.uniform_filter); axes with size 1 are skipped.</summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Window size per axis (one value broadcasts).</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray UniformFilter(NdArray input, int[] size, NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int[]? origin = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var shape = ShapeOf(input);
        var rank = shape.Length;
        var sizes = Expand(size, rank, 1, nameof(size));
        var origins = Expand(origin, rank, 0, nameof(origin));
        var data = input.ToArray();
        var m = FilterMode(mode);
        for (var a = 0; a < rank; a++)
            if (sizes[a] > 1) Uniform1dInPlace(data, shape, sizes[a], a, m, cval, origins[a]);
        return new NdArray(data, shape);
    }

    /// <summary>Isotropic overload of <see cref="UniformFilter(NdArray, int[], NdImageMode, double, int[])"/>.</summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Window size on every axis.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Origin on every axis.</param>
    public static NdArray UniformFilter(NdArray input, int size, NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int origin = 0)
        => UniformFilter(input, [size], mode, cval, [origin]);

    // ------------------------------------------------------------------ minimum / maximum

    /// <summary>Moving minimum along one axis (scipy.ndimage.minimum_filter1d), van Herk–Gil–Werman O(1) per sample.</summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Window length.</param>
    /// <param name="axis">Axis to filter along.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Window origin.</param>
    public static NdArray MinimumFilter1d(NdArray input, int size, int axis = -1, NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int origin = 0)
        => MinMax1d(input, size, axis, mode, cval, origin, minimum: true);

    /// <summary>Moving maximum along one axis (scipy.ndimage.maximum_filter1d), van Herk–Gil–Werman O(1) per sample.</summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Window length.</param>
    /// <param name="axis">Axis to filter along.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Window origin.</param>
    public static NdArray MaximumFilter1d(NdArray input, int size, int axis = -1, NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int origin = 0)
        => MinMax1d(input, size, axis, mode, cval, origin, minimum: false);

    private static NdArray MinMax1d(NdArray input, int size, int axis, NdImageMode mode, double cval, int origin, bool minimum)
    {
        ArgumentNullException.ThrowIfNull(input);
        var shape = ShapeOf(input);
        var ax = NormalizeAxis(axis, shape.Length);
        var data = input.ToArray();
        MinMax1dInPlace(data, shape, size, ax, FilterMode(mode), cval, origin, minimum);
        return new NdArray(data, shape);
    }

    private static void MinMax1dInPlace(double[] data, int[] shape, int size, int axis, NdImageMode mode, double cval, int origin, bool minimum)
    {
        if (size < 1) throw new ArgumentException("Incorrect filter size.");
        CheckOrigins([size], [origin]);
        var before = size / 2 + origin;
        AlongAxis(data, data, shape, axis, before, size - 1 - before, mode, cval, new NdImageMinMaxLineOp(size, minimum));
    }

    /// <summary>
    /// Minimum filter (scipy.ndimage.minimum_filter). A rectangular window (a <paramref name="size"/>,
    /// or a footprint that is all true) runs as separable O(1)-per-sample 1-D passes; any other
    /// footprint is evaluated directly.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Window size per axis; ignored when <paramref name="footprint"/> is given.</param>
    /// <param name="footprint">Structuring element (non-zero = included), same rank as the input.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray MinimumFilter(NdArray input, int[]? size = null, NdArray? footprint = null, NdImageMode mode = NdImageMode.Reflect,
        double cval = 0.0, int[]? origin = null)
        => MinOrMaxFilter(input, size, footprint, null, mode, cval, origin, minimum: true);

    /// <summary>Isotropic overload of the minimum filter.</summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Window size on every axis.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Origin on every axis.</param>
    public static NdArray MinimumFilter(NdArray input, int size, NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int origin = 0)
        => MinOrMaxFilter(input, [size], null, null, mode, cval, [origin], minimum: true);

    /// <summary>
    /// Maximum filter (scipy.ndimage.maximum_filter); separable for rectangular windows, direct
    /// for arbitrary footprints.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Window size per axis; ignored when <paramref name="footprint"/> is given.</param>
    /// <param name="footprint">Structuring element, same rank as the input.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray MaximumFilter(NdArray input, int[]? size = null, NdArray? footprint = null, NdImageMode mode = NdImageMode.Reflect,
        double cval = 0.0, int[]? origin = null)
        => MinOrMaxFilter(input, size, footprint, null, mode, cval, origin, minimum: false);

    /// <summary>Isotropic overload of the maximum filter.</summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Window size on every axis.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Origin on every axis.</param>
    public static NdArray MaximumFilter(NdArray input, int size, NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int origin = 0)
        => MinOrMaxFilter(input, [size], null, null, mode, cval, [origin], minimum: false);

    /// <summary>scipy's <c>_min_or_max_filter</c>: chooses the separable path or the footprint/structure path.</summary>
    internal static NdArray MinOrMaxFilter(NdArray input, int[]? size, NdArray? footprint, NdArray? structure,
        NdImageMode mode, double cval, int[]? origin, bool minimum)
    {
        ArgumentNullException.ThrowIfNull(input);
        var shape = ShapeOf(input);
        var rank = shape.Length;
        var origins = Expand(origin, rank, 0, nameof(origin));
        var m = FilterMode(mode);
        bool separable;
        bool[]? fp = null;
        int[]? fshape = null;
        double[]? sv = null;
        if (structure is null)
        {
            if (footprint is null)
            {
                if (size is null) throw new ArgumentException("No footprint or filter size provided.");
                separable = true;
            }
            else
            {
                fp = ToMask(footprint);
                fshape = ShapeOf(footprint);
                if (!fp.Any(b => b)) throw new ArgumentException("All-zero footprint is not supported.");
                if (fp.All(b => b)) { size = fshape; separable = true; }
                else separable = false;
            }
        }
        else
        {
            sv = structure.ToArray();
            fshape = ShapeOf(structure);
            fp = footprint is null ? Enumerable.Repeat(true, sv.Length).ToArray() : ToMask(footprint);
            if (footprint is not null && !ShapeOf(footprint).SequenceEqual(fshape))
                throw new ArgumentException("Structure and footprint shapes must match.");
            separable = false;
        }

        var data = input.ToArray();
        if (separable)
        {
            var sizes = Expand(size, rank, 1, nameof(size));
            for (var a = 0; a < rank; a++)
                if (sizes[a] > 1) MinMax1dInPlace(data, shape, sizes[a], a, m, cval, origins[a], minimum);
            return new NdArray(data, shape);
        }
        if (fshape!.Length != rank) throw new ArgumentException("Footprint rank must equal the input rank.");
        CheckOrigins(fshape, origins);
        double[]? svals = null;
        if (sv is not null)
        {
            var list = new List<double>();
            for (var i = 0; i < fp!.Length; i++) if (fp[i]) list.Add(sv[i]);
            svals = list.ToArray();
        }
        var result = FootprintFilter(data, shape, fshape, fp!, origins, m, cval, new NdImageMinMaxWindowOp(minimum, svals));
        return new NdArray(result, shape);
    }

    // ------------------------------------------------------------------ rank / median / percentile

    /// <summary>
    /// Rank filter (scipy.ndimage.rank_filter): the <paramref name="rank"/>-th smallest value under
    /// the footprint; negative ranks count from the top. Rank 0 and the top rank delegate to the
    /// minimum/maximum filters, as scipy does.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="rank">Order statistic (0-based; negative counts from the largest).</param>
    /// <param name="size">Window size per axis (one value broadcasts); ignored when a footprint is given.</param>
    /// <param name="footprint">Structuring element, same rank as the input.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray RankFilter(NdArray input, int rank, int[]? size = null, NdArray? footprint = null,
        NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int[]? origin = null)
        => RankCore(input, _ => rank, size, footprint, mode, cval, origin);

    /// <summary>Median filter (scipy.ndimage.median_filter): rank <c>count / 2</c>, the upper median for even counts.</summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Window size per axis (one value broadcasts); ignored when a footprint is given.</param>
    /// <param name="footprint">Structuring element, same rank as the input.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray MedianFilter(NdArray input, int[]? size = null, NdArray? footprint = null,
        NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int[]? origin = null)
        => RankCore(input, count => count / 2, size, footprint, mode, cval, origin);

    /// <summary>Isotropic overload of <see cref="MedianFilter(NdArray, int[], NdArray, NdImageMode, double, int[])"/>.</summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Window size on every axis.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Origin on every axis.</param>
    public static NdArray MedianFilter(NdArray input, int size, NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int origin = 0)
        => MedianFilter(input, [size], null, mode, cval, [origin]);

    /// <summary>
    /// Percentile filter (scipy.ndimage.percentile_filter): rank <c>(int)(count · p / 100)</c>,
    /// clamped to the top for p = 100; negative percentiles count from 100.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="percentile">Percentile in [-100, 100].</param>
    /// <param name="size">Window size per axis (one value broadcasts); ignored when a footprint is given.</param>
    /// <param name="footprint">Structuring element, same rank as the input.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray PercentileFilter(NdArray input, double percentile, int[]? size = null, NdArray? footprint = null,
        NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int[]? origin = null)
    {
        var p = percentile < 0.0 ? percentile + 100.0 : percentile;
        if (p < 0.0 || p > 100.0) throw new ArgumentException("Invalid percentile.");
        return RankCore(input, count => p == 100.0 ? count - 1 : (int)(count * p / 100.0), size, footprint, mode, cval, origin);
    }

    private static NdArray RankCore(NdArray input, Func<int, int> rankOf, int[]? size, NdArray? footprint,
        NdImageMode mode, double cval, int[]? origin)
    {
        ArgumentNullException.ThrowIfNull(input);
        var shape = ShapeOf(input);
        var r = shape.Length;
        var (fshape, fp) = ResolveFootprint(size, footprint, r);
        var origins = Expand(origin, r, 0, nameof(origin));
        CheckOrigins(fshape, origins);
        var count = fp.Count(b => b);
        if (count == 0) throw new ArgumentException("All-zero footprint is not supported.");
        var k = rankOf(count);
        if (k < 0) k += count;
        if (k < 0 || k >= count) throw new ArgumentException("Rank not within filter footprint size.");
        var fpArray = new NdArray(fp.Select(b => b ? 1.0 : 0.0).ToArray(), fshape);
        if (k == 0) return MinOrMaxFilter(input, null, fpArray, null, mode, cval, origins, minimum: true);
        if (k == count - 1) return MinOrMaxFilter(input, null, fpArray, null, mode, cval, origins, minimum: false);
        var result = FootprintFilter(input.ToArray(), shape, fshape, fp, origins, FilterMode(mode), cval, new NdImageRankWindowOp(k));
        return new NdArray(result, shape);
    }

    private static (int[] Shape, bool[] Footprint) ResolveFootprint(int[]? size, NdArray? footprint, int rank)
    {
        if (footprint is not null)
        {
            if (footprint.Rank != rank) throw new ArgumentException("Footprint rank must equal the input rank.");
            return (ShapeOf(footprint), ToMask(footprint));
        }
        if (size is null) throw new ArgumentException("No footprint or filter size provided.");
        var sizes = Expand(size, rank, 1, nameof(size));
        foreach (var s in sizes) if (s < 1) throw new ArgumentException("Filter sizes must be positive.");
        return (sizes, Enumerable.Repeat(true, SizeOf(sizes)).ToArray());
    }

    // ------------------------------------------------------------------ generic

    /// <summary>
    /// Applies <paramref name="function"/> to the values under the footprint at every position
    /// (scipy.ndimage.generic_filter). Runs in parallel over rows for large inputs, so the
    /// function must be thread-safe.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="function">Receives the footprint values in row-major footprint order.</param>
    /// <param name="size">Window size per axis (one value broadcasts); ignored when a footprint is given.</param>
    /// <param name="footprint">Structuring element, same rank as the input.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray GenericFilter(NdArray input, NdImageFootprintFunction function, int[]? size = null, NdArray? footprint = null,
        NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int[]? origin = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(function);
        var shape = ShapeOf(input);
        var (fshape, fp) = ResolveFootprint(size, footprint, shape.Length);
        var origins = Expand(origin, shape.Length, 0, nameof(origin));
        CheckOrigins(fshape, origins);
        var result = FootprintFilter(input.ToArray(), shape, fshape, fp, origins, FilterMode(mode), cval, new NdImageGenericWindowOp(function));
        return new NdArray(result, shape);
    }

    /// <summary>
    /// Applies a line function along one axis (scipy.ndimage.generic_filter1d). The function sees
    /// each line extended by <c>filterSize/2 + origin</c> samples before and the rest after, and
    /// writes a line of the original length. Must be thread-safe.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="function">The line filter.</param>
    /// <param name="filterSize">Length of the filter (sets the extension).</param>
    /// <param name="axis">Axis to filter along.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Filter origin.</param>
    public static NdArray GenericFilter1d(NdArray input, NdImageLineFunction function, int filterSize, int axis = -1,
        NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int origin = 0)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(function);
        if (filterSize < 1) throw new ArgumentException("Invalid filter size.");
        CheckOrigins([filterSize], [origin]);
        var shape = ShapeOf(input);
        var ax = NormalizeAxis(axis, shape.Length);
        var data = input.ToArray();
        var before = filterSize / 2 + origin;
        AlongAxis(data, data, shape, ax, before, filterSize - 1 - before, FilterMode(mode), cval, new NdImageGenericLineOp(function));
        return new NdArray(data, shape);
    }
}
