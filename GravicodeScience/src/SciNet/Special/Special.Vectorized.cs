using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Special
{
    /// <summary>Element count above which <see cref="Apply(NdArray, Func{double, double})"/> runs in parallel.</summary>
    /// <remarks>
    /// Special functions cost 20–500 ns each, so the work per element is large and the crossover is
    /// low compared with a plain add; below it the thread hand-off costs more than it saves.
    /// </remarks>
    public const int ParallelThreshold = 4096;

    /// <summary>
    /// Applies a scalar function to every element, returning a new contiguous array of the same
    /// shape (the vectorised form of any function in this class). Runs in parallel chunks above
    /// <see cref="ParallelThreshold"/> elements; the input is never modified.
    /// </summary>
    public static NdArray Apply(NdArray x, Func<double, double> f)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(f);
        var src = x.AsContiguous();
        var output = new double[src.Size];
        Map(src.AsSpan(), output, f);
        return new NdArray(output, x.Shape.ToArray());
    }

    /// <summary>
    /// Applies <paramref name="f"/> element-wise from <paramref name="input"/> into
    /// <paramref name="output"/>; parallel above <see cref="ParallelThreshold"/> elements.
    /// </summary>
    /// <remarks>
    /// The spans are pinned and handed to the workers as pointers: a span cannot be captured by a
    /// lambda, and copying it to an array to get round that costs an extra pass over memory.
    /// </remarks>
    public static unsafe void Map(ReadOnlySpan<double> input, Span<double> output, Func<double, double> f)
    {
        ArgumentNullException.ThrowIfNull(f);
        if (output.Length < input.Length) throw new ArgumentException("output is shorter than input.", nameof(output));
        var n = input.Length;
        if (n < ParallelThreshold)
        {
            for (var i = 0; i < n; i++) output[i] = f(input[i]);
            return;
        }
        var chunks = Math.Min(Environment.ProcessorCount * 4, (n + 1023) / 1024);
        var size = (n + chunks - 1) / chunks;
        fixed (double* pin = input, pout = output)
        {
            var ip = (nint)pin;
            var op = (nint)pout;
            Parallel.For(0, chunks, c =>
            {
                var src = (double*)ip;
                var dst = (double*)op;
                var lo = c * size;
                var hi = Math.Min(n, lo + size);
                for (var i = lo; i < hi; i++) dst[i] = f(src[i]);
            });
        }
    }

    /// <summary>Element-wise <see cref="Gamma(double)"/>.</summary>
    public static NdArray Gamma(NdArray x) => Apply(x, Gamma);

    /// <summary>Element-wise <see cref="GammaLn(double)"/>.</summary>
    public static NdArray GammaLn(NdArray x) => Apply(x, GammaLn);

    /// <summary>Element-wise <see cref="Digamma(double)"/>.</summary>
    public static NdArray Digamma(NdArray x) => Apply(x, Digamma);

    /// <summary>Element-wise <see cref="Erf(double)"/>.</summary>
    public static NdArray Erf(NdArray x) => Apply(x, Erf);

    /// <summary>Element-wise <see cref="Erfc(double)"/>.</summary>
    public static NdArray Erfc(NdArray x) => Apply(x, Erfc);

    /// <summary>Element-wise <see cref="Expit(double)"/>.</summary>
    public static NdArray Expit(NdArray x) => Apply(x, Expit);

    /// <summary>Element-wise <see cref="Logit(double)"/>.</summary>
    public static NdArray Logit(NdArray x) => Apply(x, Logit);

    /// <summary>Element-wise <see cref="J0(double)"/>.</summary>
    public static NdArray J0(NdArray x) => Apply(x, J0);

    /// <summary>Element-wise <see cref="I0(double)"/>.</summary>
    public static NdArray I0(NdArray x) => Apply(x, I0);

    /// <summary>Element-wise <see cref="Ndtr(double)"/>.</summary>
    public static NdArray Ndtr(NdArray x) => Apply(x, Ndtr);

    /// <summary>Element-wise <see cref="Ndtri(double)"/>.</summary>
    public static NdArray Ndtri(NdArray x) => Apply(x, Ndtri);
}
