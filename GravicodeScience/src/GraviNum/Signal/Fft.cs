using System.Numerics;

namespace Gravicode.Science.GraviNum.Signal;

/// <summary>
/// The discrete Fourier transform, and the operations that are cheapest through it.
/// </summary>
/// <remarks>
/// <para>
/// Two algorithms sit behind one API. A power-of-two length uses iterative radix-2 Cooley-Tukey,
/// which is the textbook <c>O(n log n)</c> transform. <b>Any other length uses Bluestein's
/// algorithm</b>, which re-expresses the DFT as a convolution and evaluates that with a
/// power-of-two transform — so it is still <c>O(n log n)</c>.
/// </para>
/// <para>
/// That second path is the point. Most hand-rolled FFTs handle only powers of two and leave the
/// caller to pad, but zero-padding a signal is not a neutral act: it changes the spectrum, smearing
/// each peak across neighbouring bins. A library that silently padded would return a plausible
/// answer to a question nobody asked. Padding is sometimes what you want, and then it should be
/// your decision, made in your code.
/// </para>
/// </remarks>
public static class Fft
{
    /// <summary>Transforms a complex sequence in place.</summary>
    /// <remarks>Length is unrestricted; a non-power-of-two goes through Bluestein.</remarks>
    public static Complex[] Forward(Complex[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        var result = (Complex[])values.Clone();
        Transform(result, inverse: false);
        return result;
    }

    /// <summary>
    /// The inverse transform, scaled so that <c>Inverse(Forward(x))</c> returns <c>x</c>.
    /// </summary>
    /// <remarks>
    /// The 1/n normalisation goes here rather than on the forward transform, and rather than
    /// 1/sqrt(n) on both. That is the NumPy convention; the choice is arbitrary but has to be
    /// stated, because a spectrum's amplitudes mean different things under each one.
    /// </remarks>
    public static Complex[] Inverse(Complex[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        var result = (Complex[])values.Clone();
        Transform(result, inverse: true);

        var scale = 1.0 / result.Length;
        for (var i = 0; i < result.Length; i++) result[i] *= scale;
        return result;
    }

    /// <summary>
    /// The transform of a real signal, returning only the <c>n/2 + 1</c> distinct bins.
    /// </summary>
    /// <remarks>
    /// A real signal has a conjugate-symmetric spectrum: bin <c>n-k</c> is the conjugate of bin
    /// <c>k</c>, so the upper half carries no information the lower half does not. Returning it
    /// anyway is what makes people think the transform found twice as much structure as it did.
    /// </remarks>
    public static Complex[] ForwardReal(IReadOnlyList<double> signal)
    {
        ArgumentNullException.ThrowIfNull(signal);

        var complex = new Complex[signal.Count];
        for (var i = 0; i < signal.Count; i++) complex[i] = new Complex(signal[i], 0);

        var full = Forward(complex);
        var kept = signal.Count / 2 + 1;
        return full[..kept];
    }

    /// <summary>Reconstructs a real signal of length <paramref name="length"/> from its half spectrum.</summary>
    public static double[] InverseReal(IReadOnlyList<Complex> spectrum, int length)
    {
        ArgumentNullException.ThrowIfNull(spectrum);
        if (length <= 0) throw new ArgumentOutOfRangeException(nameof(length));

        // Rebuild the discarded upper half by conjugate symmetry before inverting.
        var full = new Complex[length];
        for (var k = 0; k < length; k++)
        {
            if (k < spectrum.Count) full[k] = spectrum[k];
            else full[k] = Complex.Conjugate(spectrum[length - k]);
        }

        var inverted = Inverse(full);
        var result = new double[length];

        // The imaginary parts are zero up to rounding; taking Real rather than Magnitude keeps
        // the sign, which Magnitude would silently discard.
        for (var i = 0; i < length; i++) result[i] = inverted[i].Real;
        return result;
    }

    /// <summary>Magnitude of each bin of a real signal's spectrum.</summary>
    public static NdArray Magnitude(IReadOnlyList<double> signal)
    {
        var spectrum = ForwardReal(signal);
        var result = NdArray.Zeros(spectrum.Length);
        for (var i = 0; i < spectrum.Length; i++) result.SetAt(i, spectrum[i].Magnitude);
        return result;
    }

    /// <summary>
    /// Frequency, in hertz, of each bin returned by <see cref="ForwardReal"/>.
    /// </summary>
    /// <param name="length">Length of the original signal.</param>
    /// <param name="sampleRate">Samples per second.</param>
    /// <remarks>
    /// Worth having rather than leaving as an exercise: bin <c>k</c> is at
    /// <c>k * sampleRate / n</c>, and getting that wrong is the most common way a correct
    /// transform produces a wrong answer.
    /// </remarks>
    public static NdArray FrequencyBins(int length, double sampleRate = 1.0)
    {
        var count = length / 2 + 1;
        var result = NdArray.Zeros(count);
        for (var k = 0; k < count; k++) result.SetAt(k, k * sampleRate / length);
        return result;
    }

    /// <summary>
    /// Linear convolution of two real sequences, evaluated through the transform.
    /// </summary>
    /// <remarks>
    /// Direct convolution is <c>O(n m)</c>; going through the transform is <c>O(N log N)</c> and
    /// wins decisively once both sequences are long. Both are zero-padded to at least
    /// <c>n + m - 1</c> first, which is what makes the result <em>linear</em> rather than circular:
    /// without the padding the tail of the convolution wraps around and corrupts the start.
    /// </remarks>
    public static double[] Convolve(IReadOnlyList<double> a, IReadOnlyList<double> b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.Count == 0 || b.Count == 0) return [];

        var resultLength = a.Count + b.Count - 1;
        var size = NextPowerOfTwo(resultLength);

        var fa = new Complex[size];
        var fb = new Complex[size];
        for (var i = 0; i < a.Count; i++) fa[i] = new Complex(a[i], 0);
        for (var i = 0; i < b.Count; i++) fb[i] = new Complex(b[i], 0);

        Transform(fa, inverse: false);
        Transform(fb, inverse: false);

        for (var i = 0; i < size; i++) fa[i] *= fb[i];

        Transform(fa, inverse: true);

        var result = new double[resultLength];
        for (var i = 0; i < resultLength; i++) result[i] = fa[i].Real / size;
        return result;
    }

    /// <summary>
    /// The discrete Fourier transform computed straight from the definition.
    /// </summary>
    /// <remarks>
    /// <c>O(n²)</c> and far too slow for real use. It exists as the independent reference the fast
    /// transforms are checked against: it shares no code with them, so agreement means something.
    /// </remarks>
    public static Complex[] DirectDft(Complex[] values, bool inverse = false)
    {
        var n = values.Length;
        var result = new Complex[n];
        var sign = inverse ? 1.0 : -1.0;

        for (var k = 0; k < n; k++)
        {
            var sum = Complex.Zero;
            for (var t = 0; t < n; t++)
            {
                var angle = sign * 2.0 * Math.PI * t * k / n;
                sum += values[t] * new Complex(Math.Cos(angle), Math.Sin(angle));
            }
            result[k] = sum;
        }

        if (!inverse) return result;
        for (var i = 0; i < n; i++) result[i] /= n;
        return result;
    }

    // ---------------------------------------------------------------- implementation

    private static void Transform(Complex[] values, bool inverse)
    {
        var n = values.Length;
        if (n <= 1) return;

        if (IsPowerOfTwo(n)) Radix2(values, inverse);
        else Bluestein(values, inverse);
    }

    private static bool IsPowerOfTwo(int n) => n > 0 && (n & (n - 1)) == 0;

    private static int NextPowerOfTwo(int n)
    {
        var result = 1;
        while (result < n) result <<= 1;
        return result;
    }

    /// <summary>Iterative radix-2 Cooley-Tukey, in place.</summary>
    /// <remarks>
    /// Bit-reversal first, then log2(n) butterfly passes. Iterative rather than recursive: the
    /// recursion is easier to read but allocates a pair of arrays per level, and this is the
    /// routine everything else in the file calls.
    /// </remarks>
    private static void Radix2(Complex[] values, bool inverse)
    {
        var n = values.Length;

        for (int i = 1, j = 0; i < n; i++)
        {
            var bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) (values[i], values[j]) = (values[j], values[i]);
        }

        for (var length = 2; length <= n; length <<= 1)
        {
            var angle = (inverse ? 2.0 : -2.0) * Math.PI / length;
            var step = new Complex(Math.Cos(angle), Math.Sin(angle));

            for (var start = 0; start < n; start += length)
            {
                var w = Complex.One;
                for (var k = 0; k < length / 2; k++)
                {
                    var even = values[start + k];
                    var odd = values[start + k + length / 2] * w;
                    values[start + k] = even + odd;
                    values[start + k + length / 2] = even - odd;
                    w *= step;
                }
            }
        }
    }

    /// <summary>
    /// Bluestein's algorithm: the DFT of any length, as a convolution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The identity is <c>t k = (t² + k² - (k - t)²) / 2</c>. Substituting it into the DFT's
    /// exponent splits the kernel into a factor depending on <c>t</c>, one depending on <c>k</c>,
    /// and one depending on their difference — and a sum over a difference is a convolution, which
    /// a power-of-two transform evaluates quickly.
    /// </para>
    /// <para>
    /// The angles use <c>t² mod 2n</c> rather than <c>t²</c>. For a long signal <c>t²</c> overflows
    /// the precision of a double long before it overflows the integer, so the phase would drift and
    /// the result would degrade silently as the input grew.
    /// </para>
    /// </remarks>
    private static void Bluestein(Complex[] values, bool inverse)
    {
        var n = values.Length;
        var size = NextPowerOfTwo(n * 2 + 1);
        var sign = inverse ? 1.0 : -1.0;

        var chirp = new Complex[n];
        for (var i = 0; i < n; i++)
        {
            var index = (int)((long)i * i % (2L * n));
            var angle = sign * Math.PI * index / n;
            chirp[i] = new Complex(Math.Cos(angle), Math.Sin(angle));
        }

        var a = new Complex[size];
        var b = new Complex[size];

        for (var i = 0; i < n; i++) a[i] = values[i] * chirp[i];

        b[0] = chirp[0];
        for (var i = 1; i < n; i++) b[i] = b[size - i] = Complex.Conjugate(chirp[i]);

        Radix2(a, inverse: false);
        Radix2(b, inverse: false);
        for (var i = 0; i < size; i++) a[i] *= b[i];
        Radix2(a, inverse: true);

        for (var i = 0; i < n; i++) values[i] = a[i] / size * chirp[i];
    }
}
