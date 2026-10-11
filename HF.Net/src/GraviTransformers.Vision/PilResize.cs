using System.Runtime.CompilerServices;

namespace Gravicode.HFNet.GraviTransformers.Vision;

/// <summary>The resampling filters PIL names, by the numbers <c>preprocessor_config.json</c> stores.</summary>
public enum Resample
{
    /// <summary>PIL <c>NEAREST</c> (0).</summary>
    Nearest = 0,

    /// <summary>PIL <c>LANCZOS</c> (1), three lobes.</summary>
    Lanczos = 1,

    /// <summary>PIL <c>BILINEAR</c> (2) - ViT's.</summary>
    Bilinear = 2,

    /// <summary>PIL <c>BICUBIC</c> (3) - CLIP's.</summary>
    Bicubic = 3,

    /// <summary>PIL <c>BOX</c> (4).</summary>
    Box = 4,

    /// <summary>PIL <c>HAMMING</c> (5).</summary>
    Hamming = 5,
}

/// <summary>
/// PIL's <c>Image.resize</c> for 8-bit RGB, reproduced to the byte.
/// </summary>
/// <remarks>
/// <para>
/// Hugging Face's image processors resize through PIL, and a model sees whatever PIL produced. Other
/// libraries' "bicubic" differs in three places that each move pixels by a level or more: PIL widens
/// the kernel by the scale factor when shrinking (so it averages rather than samples), its cubic uses
/// <c>a = -0.5</c>, and it works in fixed point with 22 fractional bits, rounding to bytes between
/// the horizontal and the vertical pass. This follows <c>libImaging/Resample.c</c> step for step.
/// </para>
/// </remarks>
internal static class PilResize
{
    private const int PrecisionBits = 32 - 8 - 2;

    /// <summary>Resizes interleaved RGB bytes.</summary>
    /// <param name="source">Row-major <c>[height, width, 3]</c>.</param>
    /// <param name="width">The source width.</param>
    /// <param name="height">The source height.</param>
    /// <param name="newWidth">The target width.</param>
    /// <param name="newHeight">The target height.</param>
    /// <param name="filter">The PIL filter.</param>
    internal static byte[] Resize(byte[] source, int width, int height, int newWidth, int newHeight, Resample filter)
    {
        if (filter == Resample.Nearest) return Nearest(source, width, height, newWidth, newHeight);

        var (support, kernel) = Filter(filter);
        var current = source;
        var currentWidth = width;

        // Horizontal first, then vertical, as ImagingResample does - with a byte-rounded image between.
        if (newWidth != width)
        {
            current = Horizontal(current, currentWidth, height, newWidth, support, kernel);
            currentWidth = newWidth;
        }

        if (newHeight != height)
        {
            current = Vertical(current, currentWidth, height, newHeight, support, kernel);
        }

        return ReferenceEquals(current, source) ? (byte[])source.Clone() : current;
    }

    private static (double Support, Func<double, double> Kernel) Filter(Resample filter) => filter switch
    {
        Resample.Box => (0.5, x => x > -0.5 && x <= 0.5 ? 1.0 : 0.0),
        Resample.Bilinear => (1.0, x => Math.Abs(x) < 1 ? 1 - Math.Abs(x) : 0),
        Resample.Hamming => (1.0, x =>
        {
            x = Math.Abs(x);
            if (x == 0) return 1;
            if (x >= 1) return 0;
            x *= Math.PI;
            return Math.Sin(x) / x * (0.54f + 0.46f * Math.Cos(x));   // float literals in PIL too
        }),
        Resample.Bicubic => (2.0, x =>
        {
            const double a = -0.5;
            x = Math.Abs(x);
            if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
            if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
            return 0;
        }),
        Resample.Lanczos => (3.0, x => x >= -3 && x < 3 ? Sinc(x) * Sinc(x / 3) : 0),
        _ => throw new ArgumentOutOfRangeException(nameof(filter)),
    };

    private static double Sinc(double x)
    {
        if (x == 0) return 1;
        x *= Math.PI;
        return Math.Sin(x) / x;
    }

    /// <summary><c>precompute_coeffs</c> followed by <c>normalize_coeffs_8bpc</c>.</summary>
    private static (int[] Bounds, int[] Coefficients, int Size) Coefficients(int inSize, int outSize, double support, Func<double, double> kernel)
    {
        var scale = (double)inSize / outSize;
        var filterScale = Math.Max(scale, 1.0);
        support *= filterScale;

        var size = (int)Math.Ceiling(support) * 2 + 1;
        var bounds = new int[outSize * 2];
        var fixedPoint = new int[outSize * size];
        var weights = new double[size];

        for (var xx = 0; xx < outSize; xx++)
        {
            var center = (xx + 0.5) * scale;
            var inverse = 1.0 / filterScale;

            var xmin = (int)(center - support + 0.5);
            if (xmin < 0) xmin = 0;

            var xmax = (int)(center + support + 0.5);
            if (xmax > inSize) xmax = inSize;
            xmax -= xmin;

            var total = 0.0;
            for (var x = 0; x < xmax; x++)
            {
                var w = kernel((x + xmin - center + 0.5) * inverse);
                weights[x] = w;
                total += w;
            }

            for (var x = 0; x < xmax; x++)
            {
                var k = total != 0 ? weights[x] / total : weights[x];
                fixedPoint[xx * size + x] = k < 0
                    ? (int)(-0.5 + k * (1 << PrecisionBits))
                    : (int)(0.5 + k * (1 << PrecisionBits));
            }

            bounds[xx * 2] = xmin;
            bounds[xx * 2 + 1] = xmax;
        }

        return (bounds, fixedPoint, size);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte Clip8(int value)
    {
        if (value >= 1 << PrecisionBits << 8) return 255;
        if (value <= 0) return 0;
        return (byte)(value >> PrecisionBits);
    }

    private static byte[] Horizontal(byte[] source, int width, int height, int newWidth, double support, Func<double, double> kernel)
    {
        var (bounds, k, size) = Coefficients(width, newWidth, support, kernel);
        var result = new byte[newWidth * height * 3];

        for (var y = 0; y < height; y++)
        {
            for (var xx = 0; xx < newWidth; xx++)
            {
                var xmin = bounds[xx * 2];
                var xmax = bounds[xx * 2 + 1];
                int r = 1 << (PrecisionBits - 1), g = r, b = r;

                for (var x = 0; x < xmax; x++)
                {
                    var at = (y * width + x + xmin) * 3;
                    var weight = k[xx * size + x];
                    r += source[at] * weight;
                    g += source[at + 1] * weight;
                    b += source[at + 2] * weight;
                }

                var target = (y * newWidth + xx) * 3;
                result[target] = Clip8(r);
                result[target + 1] = Clip8(g);
                result[target + 2] = Clip8(b);
            }
        }

        return result;
    }

    private static byte[] Vertical(byte[] source, int width, int height, int newHeight, double support, Func<double, double> kernel)
    {
        var (bounds, k, size) = Coefficients(height, newHeight, support, kernel);
        var result = new byte[width * newHeight * 3];

        for (var yy = 0; yy < newHeight; yy++)
        {
            var ymin = bounds[yy * 2];
            var ymax = bounds[yy * 2 + 1];

            for (var x = 0; x < width; x++)
            {
                int r = 1 << (PrecisionBits - 1), g = r, b = r;

                for (var y = 0; y < ymax; y++)
                {
                    var at = ((y + ymin) * width + x) * 3;
                    var weight = k[yy * size + y];
                    r += source[at] * weight;
                    g += source[at + 1] * weight;
                    b += source[at + 2] * weight;
                }

                var target = (yy * width + x) * 3;
                result[target] = Clip8(r);
                result[target + 1] = Clip8(g);
                result[target + 2] = Clip8(b);
            }
        }

        return result;
    }

    /// <summary>PIL's nearest: the source pixel under each target pixel's centre.</summary>
    private static byte[] Nearest(byte[] source, int width, int height, int newWidth, int newHeight)
    {
        var result = new byte[newWidth * newHeight * 3];
        for (var y = 0; y < newHeight; y++)
        {
            var sy = Math.Min(height - 1, (int)((y + 0.5) * height / newHeight));
            for (var x = 0; x < newWidth; x++)
            {
                var sx = Math.Min(width - 1, (int)((x + 0.5) * width / newWidth));
                Array.Copy(source, (sy * width + sx) * 3, result, (y * newWidth + x) * 3, 3);
            }
        }

        return result;
    }
}
