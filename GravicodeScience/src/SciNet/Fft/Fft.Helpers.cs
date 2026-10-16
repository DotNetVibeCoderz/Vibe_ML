using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Fft
{
    // ---------------------------------------------------------------- frequencies

    /// <summary>
    /// Sample frequencies of the bins of an n-point DFT — scipy's <c>fftfreq(n, d)</c>:
    /// <c>[0, 1, …, ⌈n/2⌉−1, −⌊n/2⌋, …, −1] / (d n)</c>.
    /// </summary>
    /// <param name="n">Window length.</param>
    /// <param name="d">Sample spacing (inverse of the sampling rate).</param>
    public static double[] FftFreq(int n, double d = 1.0)
    {
        if (n < 1) throw new ArgumentException("n must be positive.", nameof(n));
        var result = new double[n];
        var scale = 1.0 / (n * d);
        var positive = (n - 1) / 2 + 1;
        for (var i = 0; i < positive; i++) result[i] = i * scale;
        for (var i = positive; i < n; i++) result[i] = (i - n) * scale;
        return result;
    }

    /// <summary>Sample frequencies of the <c>n/2 + 1</c> bins of <see cref="Rfft(double[], int?, FftNorm)"/> — scipy's <c>rfftfreq</c>.</summary>
    public static double[] RfftFreq(int n, double d = 1.0)
    {
        if (n < 1) throw new ArgumentException("n must be positive.", nameof(n));
        var result = new double[n / 2 + 1];
        var scale = 1.0 / (n * d);
        for (var i = 0; i < result.Length; i++) result[i] = i * scale;
        return result;
    }

    // ---------------------------------------------------------------- shifts

    /// <summary>Moves the zero-frequency bin to the centre — scipy's <c>fftshift</c>.</summary>
    public static double[] FftShift(double[] x) => Roll1D(x, (x ?? throw new ArgumentNullException(nameof(x))).Length / 2);

    /// <summary>Moves the zero-frequency bin to the centre.</summary>
    public static Complex[] FftShift(Complex[] x) => Roll1D(x, (x ?? throw new ArgumentNullException(nameof(x))).Length / 2);

    /// <summary>Undoes <see cref="FftShift(double[])"/> (they differ for odd lengths) — scipy's <c>ifftshift</c>.</summary>
    public static double[] IfftShift(double[] x) => Roll1D(x, -((x ?? throw new ArgumentNullException(nameof(x))).Length / 2));

    /// <summary>Undoes <see cref="FftShift(Complex[])"/>.</summary>
    public static Complex[] IfftShift(Complex[] x) => Roll1D(x, -((x ?? throw new ArgumentNullException(nameof(x))).Length / 2));

    /// <summary>Centres the zero frequency along <paramref name="axes"/> (default all).</summary>
    public static NdArray FftShift(NdArray x, int[]? axes = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var shape = x.Shape.ToArray();
        return new NdArray(RollAxes(x.ToArray(), shape, axes, inverse: false), shape);
    }

    /// <summary>Inverse of <see cref="FftShift(NdArray, int[])"/>.</summary>
    public static NdArray IfftShift(NdArray x, int[]? axes = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var shape = x.Shape.ToArray();
        return new NdArray(RollAxes(x.ToArray(), shape, axes, inverse: true), shape);
    }

    /// <summary>Centres the zero frequency along <paramref name="axes"/> (default all).</summary>
    public static ComplexNdArray FftShift(ComplexNdArray x, int[]? axes = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var shape = x.Shape.ToArray();
        return new ComplexNdArray(RollAxes(x.ToArray(), shape, axes, inverse: false), shape);
    }

    /// <summary>Inverse of <see cref="FftShift(ComplexNdArray, int[])"/>.</summary>
    public static ComplexNdArray IfftShift(ComplexNdArray x, int[]? axes = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var shape = x.Shape.ToArray();
        return new ComplexNdArray(RollAxes(x.ToArray(), shape, axes, inverse: true), shape);
    }

    private static T[] Roll1D<T>(T[] x, int shift)
    {
        var n = x.Length;
        var result = new T[n];
        if (n == 0) return result;
        var s = ((shift % n) + n) % n;
        Array.Copy(x, 0, result, s, n - s);
        Array.Copy(x, n - s, result, 0, s);
        return result;
    }

    private static T[] RollAxes<T>(T[] data, int[] shape, int[]? axes, bool inverse)
    {
        var ax = axes ?? Enumerable.Range(0, shape.Length).ToArray();
        foreach (var a in ax)
        {
            var axis = FftAxes.NormalizeAxis(a, shape.Length);
            var len = shape[axis];
            if (len <= 1) continue;
            var shift = inverse ? len - len / 2 : len / 2;
            var (outer, inner) = FftAxes.Split(shape, axis);
            var result = new T[data.Length];
            var block = len * inner;
            var tail = (len - shift) * inner;
            for (var o = 0; o < outer; o++)
            {
                var b = o * block;
                Array.Copy(data, b, result, b + shift * inner, tail);
                Array.Copy(data, b + tail, result, b, shift * inner);
            }
            data = result;
        }
        return data;
    }

    // ---------------------------------------------------------------- fast lengths

    /// <summary>
    /// The smallest length ≥ <paramref name="target"/> that this engine transforms fastest —
    /// scipy's <c>next_fast_len(target, real)</c>, with identical results.
    /// </summary>
    /// <remarks>
    /// Real transforms get a 5-smooth length (2^a 3^b 5^c), complex ones an 11-smooth length —
    /// the same sets scipy uses. Radix 7 and 11 run through the generic odd butterfly here, which
    /// is still far cheaper than padding to the next 5-smooth size or falling back to Bluestein.
    /// </remarks>
    /// <param name="target">Minimum length (0 returns 0).</param>
    /// <param name="real">True for real-input transforms.</param>
    public static int NextFastLen(int target, bool real = false)
    {
        if (target < 0) throw new ArgumentException("Target cannot be negative.", nameof(target));
        if (target <= 6 || (!real && target <= 12)) return target;
        var result = FftMath.NextSmooth(target, real ? 5 : 11);
        if (result > int.MaxValue) throw new ArgumentException($"No fast length ≥ {target} fits in an int.", nameof(target));
        return (int)result;
    }

    /// <summary>The largest fast length ≤ <paramref name="target"/> — scipy's <c>prev_fast_len</c>.</summary>
    public static int PrevFastLen(int target, bool real = false)
    {
        if (target < 1) throw new ArgumentException("Target must be positive.", nameof(target));
        return (int)FftMath.PrevSmooth(target, real ? 5 : 11);
    }

    // ---------------------------------------------------------------- convolution

    /// <summary>
    /// Full linear convolution of two real sequences through the real FFT (length
    /// <c>n + m − 1</c>) — the kernel behind scipy's <c>fftconvolve(a, b, mode="full")</c>.
    /// </summary>
    /// <remarks>
    /// Both inputs are zero-padded to at least <c>n + m − 1</c> — which is what makes the result
    /// linear instead of circular — and then on to the next 5-smooth length, which is never more
    /// than a few percent longer and keeps the transform off the Bluestein path. Rounding error is
    /// relative to the largest output, so outputs that should be exactly zero come back as
    /// ~1e-16 · max|y|, not 0.
    /// </remarks>
    public static double[] Convolve(double[] a, double[] b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.Length == 0 || b.Length == 0) return [];
        var outLen = a.Length + b.Length - 1;
        var size = NextFastLen(outLen, real: true);
        var plan = RfftPlan.Get(size);
        var bins = size / 2 + 1;

        var buffer = new double[size];
        var fa = new Complex[bins];
        var fb = new Complex[bins];
        a.CopyTo(buffer, 0);
        plan.Forward(buffer, fa);
        Array.Clear(buffer);
        b.CopyTo(buffer, 0);
        plan.Forward(buffer, fb);
        FftComplexOps.MultiplyInPlace(fa, fb);
        plan.Inverse(fa, buffer);

        var result = new double[outLen];
        var scale = 1.0 / size;
        for (var i = 0; i < outLen; i++) result[i] = buffer[i] * scale;
        return result;
    }

    /// <summary>
    /// Full cross-correlation <c>c_k = Σ_n a_{n+k−(m−1)} b_n</c> (length <c>n + m − 1</c>) — scipy's
    /// <c>signal.correlate(a, b, mode="full", method="fft")</c>. Lag zero sits at index m − 1.
    /// </summary>
    public static double[] Correlate(double[] a, double[] b)
    {
        ArgumentNullException.ThrowIfNull(b);
        var reversed = (double[])b.Clone();
        Array.Reverse(reversed);
        return Convolve(a, reversed);
    }

    /// <summary>
    /// Full n-D linear convolution through the real n-D FFT, over <paramref name="axes"/>
    /// (default all). Along transformed axes the output length is <c>n + m − 1</c>; any other axis
    /// must have the same length in both inputs and is convolved element by element.
    /// </summary>
    public static NdArray Convolve(NdArray a, NdArray b, int[]? axes = null, int? workers = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.Rank != b.Rank) throw new ArgumentException("Inputs must have the same rank.", nameof(b));
        var rank = a.Rank;
        var ax = (axes ?? Enumerable.Range(0, rank).ToArray()).Select(v => FftAxes.NormalizeAxis(v, rank)).ToArray();
        if (ax.Length == 0) throw new ArgumentException("At least one axis is required.", nameof(axes));
        for (var d = 0; d < rank; d++)
            if (!ax.Contains(d) && a.Shape[d] != b.Shape[d])
                throw new ArgumentException($"Axis {d} is not convolved, so both inputs need the same length there.", nameof(b));
        if (a.Size == 0 || b.Size == 0) throw new ArgumentException("Inputs must not be empty.");

        var full = ax.Select(d => a.Shape[d] + b.Shape[d] - 1).ToArray();
        var fast = new int[ax.Length];
        for (var i = 0; i < ax.Length; i++)
            fast[i] = i == ax.Length - 1 ? NextFastLen(full[i], real: true) : NextFastLen(full[i]);

        var fa = RfftN(a, fast, ax, FftNorm.Backward, workers);
        var fb = RfftN(b, fast, ax, FftNorm.Backward, workers);
        var product = fa.Data;
        var other = fb.Data;
        for (var i = 0; i < product.Length; i++) product[i] *= other[i];
        var y = IrfftN(fa, fast, ax, FftNorm.Backward, workers);

        var shape = y.Shape.ToArray();
        var data = y.AsContiguous().ToArray();
        for (var i = 0; i < ax.Length; i++) data = FftAxes.Resize(data, shape, ax[i], full[i]);
        return new NdArray(data, shape);
    }
}
