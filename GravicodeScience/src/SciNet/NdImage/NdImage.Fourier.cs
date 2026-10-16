using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class NdImage
{
    /// <summary>
    /// Multiplies a full complex spectrum by the Fourier transform of a Gaussian
    /// (scipy.ndimage.fourier_gaussian with <c>n = -1</c>): <c>exp(-2π² σ² f²)</c> per axis, with
    /// <c>f = k/n</c> in FFT order.
    /// </summary>
    /// <param name="input">Spectrum, as returned by a complex FFT over every axis.</param>
    /// <param name="sigma">Standard deviation per axis (one value broadcasts).</param>
    public static ComplexNdArray FourierGaussian(ComplexNdArray input, double[] sigma)
    {
        ArgumentNullException.ThrowIfNull(input);
        var s = NdImageSupport.Expand(sigma, input.Rank, 0.0, nameof(sigma));
        return FourierMultiply(input, (axis, n, k) =>
        {
            var f = Frequency(k, n);
            return Math.Exp(-2.0 * Math.PI * Math.PI * s[axis] * s[axis] * f * f);
        });
    }

    /// <summary>
    /// Multiplies a full complex spectrum by the transform of a box of the given size
    /// (scipy.ndimage.fourier_uniform with <c>n = -1</c>): <c>sin(π·size·f)/(π·size·f)</c> per axis.
    /// </summary>
    /// <param name="input">Spectrum.</param>
    /// <param name="size">Box size per axis (one value broadcasts).</param>
    public static ComplexNdArray FourierUniform(ComplexNdArray input, double[] size)
    {
        ArgumentNullException.ThrowIfNull(input);
        var s = NdImageSupport.Expand(size, input.Rank, 1.0, nameof(size));
        return FourierMultiply(input, (axis, n, k) =>
        {
            var x = Math.PI * s[axis] * Frequency(k, n);
            return x == 0 ? 1.0 : Math.Sin(x) / x;
        });
    }

    /// <summary>
    /// Applies a (sub-pixel, periodic) shift in the Fourier domain (scipy.ndimage.fourier_shift with
    /// <c>n = -1</c>): multiplies by <c>exp(-2πi · shift · f)</c> per axis.
    /// </summary>
    /// <param name="input">Spectrum.</param>
    /// <param name="shift">Shift per axis in samples (one value broadcasts).</param>
    public static ComplexNdArray FourierShift(ComplexNdArray input, double[] shift)
    {
        ArgumentNullException.ThrowIfNull(input);
        var s = NdImageSupport.Expand(shift, input.Rank, 0.0, nameof(shift));
        var rank = input.Rank;
        var shape = input.Shape.ToArray();
        var factors = new Complex[rank][];
        for (var a = 0; a < rank; a++)
        {
            factors[a] = new Complex[shape[a]];
            for (var k = 0; k < shape[a]; k++)
            {
                var phase = -2.0 * Math.PI * s[a] * Frequency(k, shape[a]);
                factors[a][k] = new Complex(Math.Cos(phase), Math.Sin(phase));
            }
        }
        var data = input.ToArray();
        var counter = new int[rank];
        for (var i = 0; i < data.Length; i++)
        {
            var f = Complex.One;
            for (var a = 0; a < rank; a++) f *= factors[a][counter[a]];
            data[i] *= f;
            Increment(counter, shape);
        }
        return new ComplexNdArray(data, shape);
    }

    private static double Frequency(int k, int n) => (k < (n + 1) / 2 ? k : k - n) / (double)n;

    private static ComplexNdArray FourierMultiply(ComplexNdArray input, Func<int, int, int, double> factor)
    {
        var rank = input.Rank;
        var shape = input.Shape.ToArray();
        var tables = new double[rank][];
        for (var a = 0; a < rank; a++)
        {
            tables[a] = new double[shape[a]];
            for (var k = 0; k < shape[a]; k++) tables[a][k] = factor(a, shape[a], k);
        }
        var data = input.ToArray();
        var counter = new int[rank];
        for (var i = 0; i < data.Length; i++)
        {
            var f = 1.0;
            for (var a = 0; a < rank; a++) f *= tables[a][counter[a]];
            data[i] *= f;
            Increment(counter, shape);
        }
        return new ComplexNdArray(data, shape);
    }

    private static void Increment(int[] counter, int[] shape)
    {
        for (var a = counter.Length - 1; a >= 0; a--)
        {
            if (++counter[a] < shape[a]) return;
            counter[a] = 0;
        }
    }
}
