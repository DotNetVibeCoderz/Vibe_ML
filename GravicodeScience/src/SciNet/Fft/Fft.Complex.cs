using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Normalisation of a transform pair, as scipy.fft's <c>norm</c> keyword.
/// </summary>
public enum FftNorm
{
    /// <summary>Forward unscaled, inverse scaled by 1/n (the default, numpy's convention).</summary>
    Backward,

    /// <summary>Both directions scaled by 1/√n, making the transform unitary.</summary>
    Ortho,

    /// <summary>Forward scaled by 1/n, inverse unscaled.</summary>
    Forward,
}

/// <summary>
/// Discrete Fourier and trigonometric transforms — the counterpart of <c>scipy.fft</c>.
/// </summary>
/// <remarks>
/// <para>
/// Everything here runs on <see cref="FftPlan"/> and <see cref="RfftPlan"/>: cached mixed-radix
/// plans (radix 2, 3, 4, 5 and a generic odd butterfly, SIMD where it measured faster) with
/// Bluestein for lengths that have a large prime factor. Any length is exact — nothing is
/// zero-padded unless you pass <c>n</c> — because padding changes the spectrum.
/// </para>
/// <para>
/// <b>Naming.</b> C# forbids a member named like its class, so scipy's <c>fft</c> is
/// <see cref="Fft1D(Complex[], int?, FftNorm)"/>; every other function keeps its scipy name
/// (<c>Ifft</c>, <c>Fft2</c>, <c>Rfft</c>, <c>Dct</c>, ...). The in-place span entry points are
/// <see cref="Forward(Span{Complex}, FftNorm)"/> and <see cref="Inverse(Span{Complex}, FftNorm)"/>.
/// </para>
/// <para>
/// <b>Threads.</b> Transforms along an axis of an n-D array run their independent lines in
/// parallel once the array is large enough to pay for it. <c>workers</c> follows scipy: a positive
/// value caps the thread count, -1 means all cores; unlike scipy the default (null) is "all cores
/// when worthwhile" rather than one.
/// </para>
/// </remarks>
public static partial class Fft
{
    // ---------------------------------------------------------------- normalisation

    /// <summary>Scale applied after an unnormalised transform of length n.</summary>
    internal static double NormFactor(FftNorm norm, long n, bool inverse) => norm switch
    {
        FftNorm.Backward => inverse ? 1.0 / n : 1.0,
        FftNorm.Ortho => 1.0 / Math.Sqrt(n),
        FftNorm.Forward => inverse ? 1.0 : 1.0 / n,
        _ => throw new ArgumentException($"Unknown norm {norm}.", nameof(norm)),
    };

    internal static FftNorm SwapDirection(FftNorm norm) => norm switch
    {
        FftNorm.Backward => FftNorm.Forward,
        FftNorm.Forward => FftNorm.Backward,
        _ => norm,
    };

    private static int ResolveLength(int? n, int actual)
    {
        var len = n ?? actual;
        if (len < 1) throw new ArgumentException($"Invalid number of data points ({len}) specified.", nameof(n));
        return len;
    }

    // ---------------------------------------------------------------- span API (allocation-free)

    /// <summary>
    /// In-place forward FFT of <paramref name="data"/>. Allocation-free after the plan for this
    /// length is cached (scratch comes from the shared array pool).
    /// </summary>
    public static void Forward(Span<Complex> data, FftNorm norm = FftNorm.Backward)
    {
        if (data.Length == 0) throw new ArgumentException("Cannot transform an empty sequence.", nameof(data));
        FftPlan.Get(data.Length).Execute(data, data, inverse: false);
        FftComplexOps.Scale(data, NormFactor(norm, data.Length, inverse: false));
    }

    /// <summary>In-place inverse FFT; with the default norm, <c>Inverse(Forward(x)) == x</c>.</summary>
    public static void Inverse(Span<Complex> data, FftNorm norm = FftNorm.Backward)
    {
        if (data.Length == 0) throw new ArgumentException("Cannot transform an empty sequence.", nameof(data));
        FftPlan.Get(data.Length).Execute(data, data, inverse: true);
        FftComplexOps.Scale(data, NormFactor(norm, data.Length, inverse: true));
    }

    /// <summary>Out-of-place forward FFT into a caller-owned buffer of the same length.</summary>
    public static void Forward(ReadOnlySpan<Complex> input, Span<Complex> output, FftNorm norm = FftNorm.Backward)
    {
        if (input.Length == 0) throw new ArgumentException("Cannot transform an empty sequence.", nameof(input));
        FftPlan.Get(input.Length).Execute(input, output, inverse: false);
        FftComplexOps.Scale(output, NormFactor(norm, input.Length, inverse: false));
    }

    /// <summary>Out-of-place inverse FFT into a caller-owned buffer of the same length.</summary>
    public static void Inverse(ReadOnlySpan<Complex> input, Span<Complex> output, FftNorm norm = FftNorm.Backward)
    {
        if (input.Length == 0) throw new ArgumentException("Cannot transform an empty sequence.", nameof(input));
        FftPlan.Get(input.Length).Execute(input, output, inverse: true);
        FftComplexOps.Scale(output, NormFactor(norm, input.Length, inverse: true));
    }

    // ---------------------------------------------------------------- 1-D arrays

    /// <summary>
    /// The 1-D discrete Fourier transform, <c>X_k = Σ_j x_j exp(-2πi jk/n)</c> — scipy's
    /// <c>fft(x, n, norm=...)</c>.
    /// </summary>
    /// <param name="x">Input sequence.</param>
    /// <param name="n">Transform length: the input is truncated or zero-padded to it. Defaults to
    /// the input length.</param>
    /// <param name="norm">Normalisation; see <see cref="FftNorm"/>.</param>
    public static Complex[] Fft1D(Complex[] x, int? n = null, FftNorm norm = FftNorm.Backward)
        => Transform1D(x, n, norm, inverse: false);

    /// <summary>The DFT of a real sequence: computed with the real transform, then mirrored.</summary>
    public static Complex[] Fft1D(double[] x, int? n = null, FftNorm norm = FftNorm.Backward)
    {
        ArgumentNullException.ThrowIfNull(x);
        var len = ResolveLength(n, x.Length);
        var result = new Complex[len];
        RealToFullSpectrum(PadReal(x, len), result);
        FftComplexOps.Scale(result, NormFactor(norm, len, inverse: false));
        return result;
    }

    /// <summary>The inverse DFT — scipy's <c>ifft</c>; with the default norm it undoes <see cref="Fft1D(Complex[], int?, FftNorm)"/>.</summary>
    public static Complex[] Ifft(Complex[] x, int? n = null, FftNorm norm = FftNorm.Backward)
        => Transform1D(x, n, norm, inverse: true);

    private static Complex[] Transform1D(Complex[] x, int? n, FftNorm norm, bool inverse)
    {
        ArgumentNullException.ThrowIfNull(x);
        var len = ResolveLength(n, x.Length);
        var buffer = new Complex[len];
        x.AsSpan(0, Math.Min(len, x.Length)).CopyTo(buffer);
        FftPlan.Get(len).Execute(buffer, buffer, inverse);
        FftComplexOps.Scale(buffer, NormFactor(norm, len, inverse));
        return buffer;
    }

    private static ReadOnlySpan<double> PadReal(double[] x, int len)
    {
        if (len == x.Length) return x;
        var padded = new double[len];
        x.AsSpan(0, Math.Min(len, x.Length)).CopyTo(padded);
        return padded;
    }

    /// <summary>Full n-point spectrum of a real line: half from the real plan, half by symmetry.</summary>
    private static void RealToFullSpectrum(ReadOnlySpan<double> line, Span<Complex> output)
    {
        var len = line.Length;
        var h = len / 2;
        RfftPlan.Get(len).Forward(line, output[..(h + 1)]);
        for (var k = h + 1; k < len; k++) output[k] = Complex.Conjugate(output[len - k]);
    }

    // ---------------------------------------------------------------- along an axis

    /// <summary>
    /// The 1-D DFT along one axis of an n-D array — scipy's <c>fft(x, n, axis, norm, workers=...)</c>.
    /// </summary>
    /// <param name="x">Input array.</param>
    /// <param name="n">Length along <paramref name="axis"/> after truncation or zero-padding.</param>
    /// <param name="axis">Axis to transform; negative counts from the end.</param>
    /// <param name="norm">Normalisation.</param>
    /// <param name="workers">Thread cap; null = all cores when the array is large enough.</param>
    public static ComplexNdArray Fft1D(ComplexNdArray x, int? n = null, int axis = -1, FftNorm norm = FftNorm.Backward, int? workers = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var shape = x.Shape.ToArray();
        var data = TransformAxis(x.ToArray(), shape, axis, n, norm, inverse: false, FftAxes.ResolveWorkers(workers));
        return new ComplexNdArray(data, shape);
    }

    /// <summary>The 1-D DFT of a real array along one axis (full, not half, spectrum).</summary>
    public static ComplexNdArray Fft1D(NdArray x, int? n = null, int axis = -1, FftNorm norm = FftNorm.Backward, int? workers = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var shape = x.Shape.ToArray();
        var ax = FftAxes.NormalizeAxis(axis, shape.Length);
        var len = ResolveLength(n, shape[ax]);
        var data = FftAxes.Resize(x.ToArray(), shape, ax, len);
        var (outer, inner) = FftAxes.Split(shape, ax);
        var result = new Complex[data.Length];
        var scale = NormFactor(norm, len, inverse: false);
        FftAxes.MapLines<double, Complex>(data, result, outer, len, len, inner, FftAxes.ResolveWorkers(workers), (line, output) =>
        {
            RealToFullSpectrum(line, output);
            FftComplexOps.Scale(output, scale);
        });
        return new ComplexNdArray(result, shape);
    }

    /// <summary>The 1-D inverse DFT along one axis — scipy's <c>ifft(x, n, axis, ...)</c>.</summary>
    public static ComplexNdArray Ifft(ComplexNdArray x, int? n = null, int axis = -1, FftNorm norm = FftNorm.Backward, int? workers = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var shape = x.Shape.ToArray();
        var data = TransformAxis(x.ToArray(), shape, axis, n, norm, inverse: true, FftAxes.ResolveWorkers(workers));
        return new ComplexNdArray(data, shape);
    }

    /// <summary>The inverse DFT of a real array along one axis.</summary>
    public static ComplexNdArray Ifft(NdArray x, int? n = null, int axis = -1, FftNorm norm = FftNorm.Backward, int? workers = null)
        => Ifft(ComplexNdArray.FromParts(x), n, axis, norm, workers);

    /// <summary>Complex-to-complex transform of every line along an axis; updates <paramref name="shape"/> for n.</summary>
    internal static Complex[] TransformAxis(Complex[] data, int[] shape, int axis, int? n, FftNorm norm, bool inverse, int workers)
    {
        var ax = FftAxes.NormalizeAxis(axis, shape.Length);
        var len = ResolveLength(n, shape[ax]);
        data = FftAxes.Resize(data, shape, ax, len);
        var (outer, inner) = FftAxes.Split(shape, ax);
        var result = new Complex[data.Length];
        var plan = FftPlan.Get(len);
        var scale = NormFactor(norm, len, inverse);
        FftAxes.MapLines<Complex, Complex>(data, result, outer, len, len, inner, workers, (line, output) =>
        {
            plan.Execute(line, output, inverse);
            FftComplexOps.Scale(output, scale);
        });
        return result;
    }

    // ---------------------------------------------------------------- 2-D and n-D

    /// <summary>2-D DFT over <paramref name="axes"/> (default the last two) — scipy's <c>fft2</c>.</summary>
    /// <param name="x">Input array of rank ≥ 2 (or ≥ the number of axes).</param>
    /// <param name="s">Output length along each transformed axis (truncate / zero-pad).</param>
    /// <param name="axes">Axes to transform; default <c>[-2, -1]</c>.</param>
    /// <param name="norm">Normalisation, applied per axis.</param>
    /// <param name="workers">Thread cap; null = all cores when worthwhile.</param>
    public static ComplexNdArray Fft2(ComplexNdArray x, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, int? workers = null)
        => FftN(x, s, axes ?? [-2, -1], norm, workers);

    /// <summary>2-D DFT of a real array.</summary>
    public static ComplexNdArray Fft2(NdArray x, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, int? workers = null)
        => FftN(ComplexNdArray.FromParts(x), s, axes ?? [-2, -1], norm, workers);

    /// <summary>2-D inverse DFT — scipy's <c>ifft2</c>.</summary>
    public static ComplexNdArray Ifft2(ComplexNdArray x, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, int? workers = null)
        => IfftN(x, s, axes ?? [-2, -1], norm, workers);

    /// <summary>
    /// n-D DFT — scipy's <c>fftn(x, s, axes)</c>: a 1-D transform along each axis in turn. Axes
    /// default to the last <c>s.Length</c> axes, or all of them.
    /// </summary>
    public static ComplexNdArray FftN(ComplexNdArray x, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, int? workers = null)
        => TransformN(x, s, axes, norm, inverse: false, workers);

    /// <summary>n-D DFT of a real array.</summary>
    public static ComplexNdArray FftN(NdArray x, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, int? workers = null)
        => TransformN(ComplexNdArray.FromParts(x), s, axes, norm, inverse: false, workers);

    /// <summary>n-D inverse DFT — scipy's <c>ifftn</c>.</summary>
    public static ComplexNdArray IfftN(ComplexNdArray x, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, int? workers = null)
        => TransformN(x, s, axes, norm, inverse: true, workers);

    private static ComplexNdArray TransformN(ComplexNdArray x, int[]? s, int[]? axes, FftNorm norm, bool inverse, int? workers)
    {
        ArgumentNullException.ThrowIfNull(x);
        var shape = x.Shape.ToArray();
        var (ax, sizes) = FftAxes.ResolveShapeAndAxes(shape, s, axes);
        var w = FftAxes.ResolveWorkers(workers);
        var data = x.ToArray();
        // Last axis first, as scipy does; the order does not change the result.
        for (var i = ax.Length - 1; i >= 0; i--)
            data = TransformAxis(data, shape, ax[i], sizes[i], norm, inverse, w);
        return new ComplexNdArray(data, shape);
    }
}
