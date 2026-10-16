using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Fft
{
    // ---------------------------------------------------------------- 1-D real transforms

    /// <summary>
    /// The DFT of a real sequence, keeping only the <c>n/2 + 1</c> non-negative-frequency bins —
    /// scipy's <c>rfft</c>.
    /// </summary>
    /// <remarks>
    /// The other bins are conjugates of these and carry no information. For even n this costs
    /// about half a complex transform of the same length (see <see cref="RfftPlan"/>).
    /// </remarks>
    /// <param name="x">Real input.</param>
    /// <param name="n">Number of input points used (truncate / zero-pad).</param>
    /// <param name="norm">Normalisation.</param>
    public static Complex[] Rfft(double[] x, int? n = null, FftNorm norm = FftNorm.Backward)
    {
        ArgumentNullException.ThrowIfNull(x);
        var len = ResolveLength(n, x.Length);
        var result = new Complex[len / 2 + 1];
        RfftPlan.Get(len).Forward(PadReal(x, len), result);
        FftComplexOps.Scale(result, NormFactor(norm, len, inverse: false));
        return result;
    }

    /// <summary>
    /// Inverse of <see cref="Rfft(double[], int?, FftNorm)"/> — scipy's <c>irfft</c>.
    /// </summary>
    /// <param name="x">Half spectrum. It is truncated or zero-padded to <c>n/2 + 1</c> bins.</param>
    /// <param name="n">Length of the real output; defaults to <c>2 (m − 1)</c> for m input bins.
    /// Pass it explicitly for odd lengths — the half spectrum alone cannot tell 2m−2 from 2m−1.</param>
    /// <param name="norm">Normalisation.</param>
    public static double[] Irfft(Complex[] x, int? n = null, FftNorm norm = FftNorm.Backward)
    {
        ArgumentNullException.ThrowIfNull(x);
        var len = ResolveLength(n, 2 * (x.Length - 1));
        var bins = len / 2 + 1;
        ReadOnlySpan<Complex> spectrum = x;
        if (x.Length != bins)
        {
            var resized = new Complex[bins];
            x.AsSpan(0, Math.Min(bins, x.Length)).CopyTo(resized);
            spectrum = resized;
        }
        var result = new double[len];
        RfftPlan.Get(len).Inverse(spectrum, result);
        FftComplexOps.Scale(result, NormFactor(norm, len, inverse: true));
        return result;
    }

    /// <summary>
    /// The FFT of a Hermitian-symmetric signal given by its first half, which is a real spectrum —
    /// scipy's <c>hfft</c>. Equals <c>Irfft(conj(x), n)</c> with the normalisation direction swapped.
    /// </summary>
    public static double[] Hfft(Complex[] x, int? n = null, FftNorm norm = FftNorm.Backward)
    {
        ArgumentNullException.ThrowIfNull(x);
        var conj = (Complex[])x.Clone();
        FftComplexOps.Conjugate(conj);
        return Irfft(conj, n, SwapDirection(norm));
    }

    /// <summary>Inverse of <see cref="Hfft(Complex[], int?, FftNorm)"/> — scipy's <c>ihfft</c>.</summary>
    public static Complex[] Ihfft(double[] x, int? n = null, FftNorm norm = FftNorm.Backward)
    {
        var result = Rfft(x, n, SwapDirection(norm));
        FftComplexOps.Conjugate(result);
        return result;
    }

    // ---------------------------------------------------------------- along an axis

    /// <summary>Real FFT along one axis — scipy's <c>rfft(x, n, axis)</c>; the axis shrinks to n/2 + 1.</summary>
    /// <param name="x">Real input array.</param>
    /// <param name="n">Number of input points along the axis (truncate / zero-pad).</param>
    /// <param name="axis">Axis to transform.</param>
    /// <param name="norm">Normalisation.</param>
    /// <param name="workers">Thread cap; null = all cores when worthwhile.</param>
    public static ComplexNdArray Rfft(NdArray x, int? n = null, int axis = -1, FftNorm norm = FftNorm.Backward, int? workers = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var shape = x.Shape.ToArray();
        var data = RfftAxis(x.ToArray(), shape, axis, n, norm, FftAxes.ResolveWorkers(workers));
        return new ComplexNdArray(data, shape);
    }

    /// <summary>Inverse real FFT along one axis — scipy's <c>irfft(x, n, axis)</c>.</summary>
    public static NdArray Irfft(ComplexNdArray x, int? n = null, int axis = -1, FftNorm norm = FftNorm.Backward, int? workers = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var shape = x.Shape.ToArray();
        var data = IrfftAxis(x.ToArray(), shape, axis, n, norm, FftAxes.ResolveWorkers(workers));
        return new NdArray(data, shape);
    }

    /// <summary>Hermitian FFT along one axis — scipy's <c>hfft(x, n, axis)</c>.</summary>
    public static NdArray Hfft(ComplexNdArray x, int? n = null, int axis = -1, FftNorm norm = FftNorm.Backward, int? workers = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var data = x.ToArray();
        FftComplexOps.Conjugate(data);
        var shape = x.Shape.ToArray();
        return new NdArray(IrfftAxis(data, shape, axis, n, SwapDirection(norm), FftAxes.ResolveWorkers(workers)), shape);
    }

    /// <summary>Inverse Hermitian FFT along one axis — scipy's <c>ihfft(x, n, axis)</c>.</summary>
    public static ComplexNdArray Ihfft(NdArray x, int? n = null, int axis = -1, FftNorm norm = FftNorm.Backward, int? workers = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var shape = x.Shape.ToArray();
        var data = RfftAxis(x.ToArray(), shape, axis, n, SwapDirection(norm), FftAxes.ResolveWorkers(workers));
        FftComplexOps.Conjugate(data);
        return new ComplexNdArray(data, shape);
    }

    internal static Complex[] RfftAxis(double[] data, int[] shape, int axis, int? n, FftNorm norm, int workers)
    {
        var ax = FftAxes.NormalizeAxis(axis, shape.Length);
        var len = ResolveLength(n, shape[ax]);
        data = FftAxes.Resize(data, shape, ax, len);
        var (outer, inner) = FftAxes.Split(shape, ax);
        var bins = len / 2 + 1;
        var result = new Complex[outer * bins * inner];
        var plan = RfftPlan.Get(len);
        var scale = NormFactor(norm, len, inverse: false);
        FftAxes.MapLines<double, Complex>(data, result, outer, len, bins, inner, workers, (line, output) =>
        {
            plan.Forward(line, output);
            FftComplexOps.Scale(output, scale);
        });
        shape[ax] = bins;
        return result;
    }

    internal static double[] IrfftAxis(Complex[] data, int[] shape, int axis, int? n, FftNorm norm, int workers)
    {
        var ax = FftAxes.NormalizeAxis(axis, shape.Length);
        var len = ResolveLength(n, 2 * (shape[ax] - 1));
        var bins = len / 2 + 1;
        data = FftAxes.Resize(data, shape, ax, bins);
        var (outer, inner) = FftAxes.Split(shape, ax);
        var result = new double[outer * len * inner];
        var plan = RfftPlan.Get(len);
        var scale = NormFactor(norm, len, inverse: true);
        FftAxes.MapLines<Complex, double>(data, result, outer, bins, len, inner, workers, (line, output) =>
        {
            plan.Inverse(line, output);
            FftComplexOps.Scale(output, scale);
        });
        shape[ax] = len;
        return result;
    }

    // ---------------------------------------------------------------- 2-D and n-D

    /// <summary>2-D real FFT over the last two axes by default — scipy's <c>rfft2</c>.</summary>
    public static ComplexNdArray Rfft2(NdArray x, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, int? workers = null)
        => RfftN(x, s, axes ?? [-2, -1], norm, workers);

    /// <summary>Inverse of <see cref="Rfft2"/> — scipy's <c>irfft2</c>.</summary>
    public static NdArray Irfft2(ComplexNdArray x, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, int? workers = null)
        => IrfftN(x, s, axes ?? [-2, -1], norm, workers);

    /// <summary>
    /// n-D real FFT — scipy's <c>rfftn</c>: a real transform along the last of
    /// <paramref name="axes"/> (which shrinks to n/2 + 1), then complex transforms along the rest.
    /// </summary>
    /// <param name="x">Real input.</param>
    /// <param name="s">Input lengths along the transformed axes (truncate / zero-pad).</param>
    /// <param name="axes">Axes to transform; defaults to the last <c>s.Length</c>, or all.</param>
    /// <param name="norm">Normalisation, per axis.</param>
    /// <param name="workers">Thread cap; null = all cores when worthwhile.</param>
    public static ComplexNdArray RfftN(NdArray x, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, int? workers = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var shape = x.Shape.ToArray();
        var (ax, sizes) = FftAxes.ResolveShapeAndAxes(shape, s, axes);
        if (ax.Length == 0) throw new ArgumentException("At least one axis is required.", nameof(axes));
        var w = FftAxes.ResolveWorkers(workers);
        var data = RfftAxis(x.ToArray(), shape, ax[^1], sizes[^1], norm, w);
        for (var i = ax.Length - 2; i >= 0; i--)
            data = TransformAxis(data, shape, ax[i], sizes[i], norm, inverse: false, w);
        return new ComplexNdArray(data, shape);
    }

    /// <summary>
    /// Inverse of <see cref="RfftN"/> — scipy's <c>irfftn</c>. The output length along the last
    /// axis defaults to <c>2 (m − 1)</c>; pass <paramref name="s"/> to recover an odd length.
    /// </summary>
    public static NdArray IrfftN(ComplexNdArray x, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, int? workers = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var shape = x.Shape.ToArray();
        var (ax, sizes) = FftAxes.ResolveShapeAndAxes(shape, s, axes);
        if (ax.Length == 0) throw new ArgumentException("At least one axis is required.", nameof(axes));
        if (s is null || s[^1] == -1) sizes[^1] = 2 * (shape[ax[^1]] - 1);
        var w = FftAxes.ResolveWorkers(workers);
        var data = x.ToArray();
        for (var i = 0; i < ax.Length - 1; i++)
            data = TransformAxis(data, shape, ax[i], sizes[i], norm, inverse: true, w);
        var real = IrfftAxis(data, shape, ax[^1], sizes[^1], norm, w);
        return new NdArray(real, shape);
    }
}
