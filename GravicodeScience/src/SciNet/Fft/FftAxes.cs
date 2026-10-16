using System.Buffers;

namespace Gravicode.Science.SciNet;

/// <summary>A transform of one line: <paramref name="input"/> and <paramref name="output"/> never alias.</summary>
internal delegate void FftLineOp<TIn, TOut>(ReadOnlySpan<TIn> input, Span<TOut> output);

/// <summary>
/// Applies a 1-D operation along one axis of a row-major buffer: every line, optionally in
/// parallel, with strided lines gathered in blocks so each cache line read is used in full.
/// </summary>
internal static class FftAxes
{
    /// <summary>Below this many elements per call, threads cost more than they save.</summary>
    private const long ParallelThreshold = 1 << 15;

    /// <summary>Strided lines are gathered this many at a time (8 complex = two cache lines).</summary>
    private const int Block = 8;

    public static int NormalizeAxis(int axis, int rank)
    {
        var a = axis < 0 ? axis + rank : axis;
        if (a < 0 || a >= rank) throw new ArgumentException($"Axis {axis} is out of range for an array of rank {rank}.", nameof(axis));
        return a;
    }

    /// <summary>
    /// scipy's <c>workers</c>: null means "all cores when worthwhile", a positive value caps the
    /// thread count, and a negative value counts back from the core count (-1 = all cores).
    /// </summary>
    public static int ResolveWorkers(int? workers)
    {
        if (workers is null) return Environment.ProcessorCount;
        var w = workers.Value;
        if (w == 0) throw new ArgumentException("workers must not be 0.", nameof(workers));
        if (w > 0) return w;
        var resolved = Environment.ProcessorCount + 1 + w;
        if (resolved < 1) throw new ArgumentException($"workers value {w} is out of range for {Environment.ProcessorCount} cores.", nameof(workers));
        return resolved;
    }

    public static (int Outer, int Inner) Split(int[] shape, int axis)
    {
        var outer = 1;
        for (var i = 0; i < axis; i++) outer *= shape[i];
        var inner = 1;
        for (var i = axis + 1; i < shape.Length; i++) inner *= shape[i];
        return (outer, inner);
    }

    /// <summary>Zero-pads or truncates <paramref name="data"/> along <paramref name="axis"/>; updates the shape.</summary>
    public static T[] Resize<T>(T[] data, int[] shape, int axis, int newLength)
    {
        var old = shape[axis];
        if (old == newLength) return data;
        var (outer, inner) = Split(shape, axis);
        var result = new T[(long)outer * newLength * inner > int.MaxValue
            ? throw new ArgumentException("Resized array is too large.")
            : outer * newLength * inner];
        var keep = Math.Min(old, newLength) * inner;
        for (var o = 0; o < outer; o++)
            Array.Copy(data, o * old * inner, result, o * newLength * inner, keep);
        shape[axis] = newLength;
        return result;
    }

    /// <summary>
    /// For every line along the axis, reads <paramref name="inLength"/> elements of
    /// <paramref name="source"/> and writes <paramref name="outLength"/> elements of
    /// <paramref name="destination"/>.
    /// </summary>
    public static void MapLines<TIn, TOut>(
        TIn[] source, TOut[] destination, int outer, int inLength, int outLength, int inner,
        int workers, FftLineOp<TIn, TOut> op)
    {
        if (outer == 0 || inner == 0) return;

        var blocksPerOuter = inner == 1 ? 1 : (inner + Block - 1) / Block;
        var items = outer * blocksPerOuter;
        var total = (long)outer * inner * Math.Max(inLength, outLength);
        var threads = total >= ParallelThreshold ? Math.Min(workers, items) : 1;

        if (threads <= 1)
        {
            Run(0, items);
            return;
        }

        // A few chunks per thread so a throttled core does not hold everybody up.
        var chunks = Math.Min(items, threads * 4);
        Parallel.For(0, chunks, new ParallelOptions { MaxDegreeOfParallelism = threads }, c =>
        {
            var start = (int)((long)items * c / chunks);
            var end = (int)((long)items * (c + 1) / chunks);
            Run(start, end);
        });

        void Run(int start, int end)
        {
            if (inner == 1)
            {
                for (var o = start; o < end; o++)
                    op(source.AsSpan(o * inLength, inLength), destination.AsSpan(o * outLength, outLength));
                return;
            }

            var inBlock = ArrayPool<TIn>.Shared.Rent(Block * inLength);
            var outBlock = ArrayPool<TOut>.Shared.Rent(Block * outLength);
            try
            {
                for (var item = start; item < end; item++)
                {
                    var o = item / blocksPerOuter;
                    var j0 = (item % blocksPerOuter) * Block;
                    var b = Math.Min(Block, inner - j0);
                    var inBase = o * inLength * inner + j0;
                    var outBase = o * outLength * inner + j0;

                    for (var t = 0; t < inLength; t++)
                    {
                        var row = inBase + t * inner;
                        for (var q = 0; q < b; q++) inBlock[q * inLength + t] = source[row + q];
                    }
                    for (var q = 0; q < b; q++)
                        op(inBlock.AsSpan(q * inLength, inLength), outBlock.AsSpan(q * outLength, outLength));
                    for (var t = 0; t < outLength; t++)
                    {
                        var row = outBase + t * inner;
                        for (var q = 0; q < b; q++) destination[row + q] = outBlock[q * outLength + t];
                    }
                }
            }
            finally
            {
                ArrayPool<TIn>.Shared.Return(inBlock);
                ArrayPool<TOut>.Shared.Return(outBlock);
            }
        }
    }

    /// <summary>
    /// Resolves scipy's <c>s</c>/<c>axes</c> pair for an n-D transform: axes default to the last
    /// <c>len(s)</c> axes (or all), sizes default to the input shape, and -1 in <c>s</c> means
    /// "the input's size".
    /// </summary>
    public static (int[] Axes, int[] Sizes) ResolveShapeAndAxes(int[] shape, int[]? s, int[]? axes)
    {
        var rank = shape.Length;
        int[] ax;
        if (axes is not null)
        {
            ax = new int[axes.Length];
            for (var i = 0; i < axes.Length; i++) ax[i] = NormalizeAxis(axes[i], rank);
            if (ax.Distinct().Count() != ax.Length) throw new ArgumentException("All axes must be unique.", nameof(axes));
        }
        else
        {
            var count = s?.Length ?? rank;
            if (count > rank) throw new ArgumentException("s has more entries than the array has axes.", nameof(s));
            ax = new int[count];
            for (var i = 0; i < count; i++) ax[i] = rank - count + i;
        }

        var sizes = new int[ax.Length];
        if (s is not null)
        {
            if (s.Length != ax.Length) throw new ArgumentException("s and axes must have the same length.", nameof(s));
            for (var i = 0; i < ax.Length; i++)
            {
                sizes[i] = s[i] == -1 ? shape[ax[i]] : s[i];
                if (sizes[i] < 1) throw new ArgumentException($"Invalid number of data points ({s[i]}) specified.", nameof(s));
            }
        }
        else
        {
            for (var i = 0; i < ax.Length; i++) sizes[i] = shape[ax[i]];
        }
        return (ax, sizes);
    }
}
