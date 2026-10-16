using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>A 1-D operation applied to every line along one axis by <see cref="NdImageSupport.AlongAxis{TOp}"/>.</summary>
internal interface INdImageLineOp
{
    /// <summary>Scratch doubles the operation needs per line of length <paramref name="n"/>.</summary>
    int ScratchLength(int n);

    /// <summary>
    /// Filters one line. <paramref name="ext"/> holds the line extended by the engine's
    /// <c>before</c>/<c>after</c> boundary samples; <paramref name="output"/> has the line's length.
    /// </summary>
    void Apply(ReadOnlySpan<double> ext, Span<double> output, Span<double> scratch);
}

/// <summary>
/// An operation over a padded n-D neighbourhood, evaluated for one output row (the last axis) at
/// a time by <see cref="NdImageSupport.FootprintFilter{TOp}"/>.
/// </summary>
internal interface INdImageWindowOp
{
    /// <summary>Scratch doubles needed per worker for a footprint of <paramref name="count"/> points.</summary>
    int ScratchLength(int count);

    /// <summary>
    /// Computes <paramref name="output"/>[i] from <c>padded[baseIndex + i + offsets[j]]</c>.
    /// </summary>
    void Row(double[] padded, int baseIndex, int[] offsets, Span<double> output, Span<double> scratch);
}

/// <summary>
/// Shared machinery for the ndimage port: boundary extension (bit-faithful to scipy's
/// <c>NI_ExtendLine</c>), the separable line engine, and whole-array padding for footprint filters.
/// </summary>
internal static class NdImageSupport
{
    /// <summary>Below this many elements the parallel paths cost more than they save.</summary>
    internal const int ParallelThreshold = 1 << 15;

    /// <summary>Lines gathered together when a pass runs along a non-final axis.</summary>
    private const int Block = 32;

    internal static int[] ShapeOf(NdArray a) => a.Shape.ToArray();

    internal static int SizeOf(int[] shape)
    {
        long s = 1;
        foreach (var d in shape) s *= d;
        if (s > int.MaxValue) throw new ArgumentException("Array is too large.");
        return (int)s;
    }

    internal static int[] StridesOf(int[] shape)
    {
        var s = new int[shape.Length];
        var acc = 1;
        for (var a = shape.Length - 1; a >= 0; a--) { s[a] = acc; acc *= shape[a]; }
        return s;
    }

    internal static int NormalizeAxis(int axis, int rank)
    {
        var a = axis < 0 ? axis + rank : axis;
        if (a < 0 || a >= rank) throw new ArgumentException($"Axis {axis} is out of range for rank {rank}.");
        return a;
    }

    /// <summary>Filters read the grid-* names as their classic counterparts (scipy's <c>is_filter=True</c>).</summary>
    internal static NdImageMode FilterMode(NdImageMode mode) => mode switch
    {
        NdImageMode.GridMirror => NdImageMode.Reflect,
        NdImageMode.GridConstant => NdImageMode.Constant,
        NdImageMode.GridWrap => NdImageMode.Wrap,
        _ => mode,
    };

    internal static int[] Expand(int[]? values, int rank, int fill, string name)
    {
        if (values is null) return Enumerable.Repeat(fill, rank).ToArray();
        if (values.Length == 1 && rank != 1) return Enumerable.Repeat(values[0], rank).ToArray();
        if (values.Length != rank) throw new ArgumentException($"{name} needs {rank} entries, got {values.Length}.");
        return (int[])values.Clone();
    }

    internal static double[] Expand(double[]? values, int rank, double fill, string name)
    {
        if (values is null) return Enumerable.Repeat(fill, rank).ToArray();
        if (values.Length == 1 && rank != 1) return Enumerable.Repeat(values[0], rank).ToArray();
        if (values.Length != rank) throw new ArgumentException($"{name} needs {rank} entries, got {values.Length}.");
        return (double[])values.Clone();
    }

    /// <summary>
    /// Maps an out-of-range line index for a filter-mode extension; -1 means "use cval".
    /// The periodic forms reproduce NI_ExtendLine's repeated copying for extensions longer than the line.
    /// </summary>
    internal static int ExtendIndex(int i, int n, NdImageMode mode)
    {
        if ((uint)i < (uint)n) return i;
        switch (mode)
        {
            case NdImageMode.Nearest:
                return i < 0 ? 0 : n - 1;
            case NdImageMode.Wrap:
            case NdImageMode.GridWrap:
                {
                    var r = i % n;
                    return r < 0 ? r + n : r;
                }
            case NdImageMode.Reflect:
            case NdImageMode.GridMirror:
                {
                    var p = 2 * n;
                    var r = i % p;
                    if (r < 0) r += p;
                    return r < n ? r : p - 1 - r;
                }
            case NdImageMode.Mirror:
                {
                    if (n == 1) return 0;
                    var p = 2 * n - 2;
                    var r = i % p;
                    if (r < 0) r += p;
                    return r < n ? r : p - r;
                }
            default:
                return -1;
        }
    }

    /// <summary>Fills the <paramref name="before"/> and <paramref name="after"/> margins around the line held in the middle of <paramref name="buf"/>.</summary>
    internal static void ExtendLine(Span<double> buf, int before, int n, int after, NdImageMode mode, double cval)
    {
        if (before == 0 && after == 0) return;
        if (mode is NdImageMode.Constant or NdImageMode.GridConstant)
        {
            buf[..before].Fill(cval);
            buf.Slice(before + n, after).Fill(cval);
            return;
        }
        if (mode == NdImageMode.Nearest)
        {
            buf[..before].Fill(buf[before]);
            buf.Slice(before + n, after).Fill(buf[before + n - 1]);
            return;
        }
        for (var k = 1; k <= before; k++)
            buf[before - k] = buf[before + ExtendIndex(-k, n, mode)];
        for (var k = 0; k < after; k++)
            buf[before + n + k] = buf[before + ExtendIndex(n + k, n, mode)];
    }

    /// <summary>
    /// Applies <paramref name="op"/> to every line along <paramref name="axis"/>. Lines along the
    /// last axis are contiguous and copied directly; along any other axis, <see cref="Block"/>
    /// neighbouring lines are gathered together so each read of the source touches whole cache
    /// lines. Every line is read in full before its output is written, so <paramref name="src"/>
    /// and <paramref name="dst"/> may be the same array.
    /// </summary>
    internal static void AlongAxis<TOp>(double[] src, double[] dst, int[] shape, int axis, int before, int after,
        NdImageMode mode, double cval, TOp op) where TOp : struct, INdImageLineOp
    {
        var n = shape[axis];
        if (n == 0) return;
        var outer = 1;
        for (var a = 0; a < axis; a++) outer *= shape[a];
        var inner = 1;
        for (var a = axis + 1; a < shape.Length; a++) inner *= shape[a];
        if (outer == 0 || inner == 0) return;
        var extLen = before + n + after;
        var scratchLen = op.ScratchLength(n);
        var total = (long)outer * n * inner;

        if (inner == 1)
        {
            var lines = outer;
            void Lines(int lo, int hi)
            {
                var ext = new double[extLen];
                var scratch = new double[scratchLen];
                var outLine = new double[n];
                for (var line = lo; line < hi; line++)
                {
                    var off = line * n;
                    src.AsSpan(off, n).CopyTo(ext.AsSpan(before));
                    ExtendLine(ext, before, n, after, mode, cval);
                    op.Apply(ext, outLine, scratch);
                    outLine.AsSpan().CopyTo(dst.AsSpan(off, n));
                }
            }
            Run(lines, total, Lines);
            return;
        }

        var blocksPerOuter = (inner + Block - 1) / Block;
        var items = outer * blocksPerOuter;
        void Blocks(int lo, int hi)
        {
            var ext = new double[Block * extLen];
            var outLines = new double[Block * n];
            var scratch = new double[scratchLen];
            for (var item = lo; item < hi; item++)
            {
                var o = item / blocksPerOuter;
                var i0 = item % blocksPerOuter * Block;
                var bw = Math.Min(Block, inner - i0);
                var baseIndex = o * n * inner + i0;
                for (var k = 0; k < n; k++)
                {
                    var row = baseIndex + k * inner;
                    var t = before + k;
                    for (var b = 0; b < bw; b++) ext[b * extLen + t] = src[row + b];
                }
                for (var b = 0; b < bw; b++)
                {
                    var e = ext.AsSpan(b * extLen, extLen);
                    ExtendLine(e, before, n, after, mode, cval);
                    op.Apply(e, outLines.AsSpan(b * n, n), scratch);
                }
                for (var k = 0; k < n; k++)
                {
                    var row = baseIndex + k * inner;
                    for (var b = 0; b < bw; b++) dst[row + b] = outLines[b * n + k];
                }
            }
        }
        Run(items, total, Blocks);
    }

    /// <summary>Runs <paramref name="body"/> over [0, count), in parallel chunks when the work is large enough.</summary>
    internal static void Run(int count, long work, Action<int, int> body)
    {
        if (count <= 1 || work < ParallelThreshold)
        {
            body(0, count);
            return;
        }
        var chunks = Math.Min(count, Environment.ProcessorCount * 4);
        Parallel.For(0, chunks, c =>
        {
            var lo = (int)((long)count * c / chunks);
            var hi = (int)((long)count * (c + 1) / chunks);
            body(lo, hi);
        });
    }

    /// <summary>
    /// Pads the whole array by per-axis extension. Mapping each coordinate independently is what
    /// scipy's border offsets do too, including Constant (any coordinate outside gives cval).
    /// </summary>
    internal static double[] Pad(double[] src, int[] shape, int[] before, int[] after, NdImageMode mode, double cval, out int[] padShape)
    {
        var rank = shape.Length;
        padShape = new int[rank];
        var maps = new int[rank][];
        for (var a = 0; a < rank; a++)
        {
            padShape[a] = shape[a] + before[a] + after[a];
            var m = new int[padShape[a]];
            for (var p = 0; p < m.Length; p++) m[p] = ExtendIndex(p - before[a], shape[a], mode);
            maps[a] = m;
        }
        var strides = StridesOf(shape);
        var psize = SizeOf(padShape);
        var result = new double[psize];
        var last = padShape[rank - 1];
        var rows = last == 0 ? 0 : psize / last;
        var lastMap = maps[rank - 1];
        var ps = padShape;
        Run(rows, psize, (lo, hi) =>
        {
            for (var r = lo; r < hi; r++)
            {
                var srcBase = 0;
                var rem = r;
                var outside = false;
                for (var a = rank - 2; a >= 0; a--)
                {
                    var c = rem % ps[a];
                    rem /= ps[a];
                    var m = maps[a][c];
                    if (m < 0) { outside = true; break; }
                    srcBase += m * strides[a];
                }
                var dst = result.AsSpan(r * last, last);
                if (outside) { dst.Fill(cval); continue; }
                for (var p = 0; p < last; p++)
                {
                    var m = lastMap[p];
                    dst[p] = m < 0 ? cval : src[srcBase + m];
                }
            }
        });
        return result;
    }

    /// <summary>
    /// Evaluates a footprint operation everywhere: pads once, turns each true footprint position
    /// into a linear offset in the padded array, then streams output rows. Offsets are in the
    /// footprint's row-major order, which is the order scipy feeds the values in.
    /// </summary>
    internal static double[] FootprintFilter<TOp>(double[] src, int[] shape, int[] fshape, bool[] footprint, int[] origin,
        NdImageMode mode, double cval, TOp op) where TOp : struct, INdImageWindowOp
    {
        var rank = shape.Length;
        var before = new int[rank];
        var after = new int[rank];
        for (var a = 0; a < rank; a++)
        {
            before[a] = fshape[a] / 2 + origin[a];
            after[a] = fshape[a] - 1 - before[a];
        }
        var padded = Pad(src, shape, before, after, mode, cval, out var pshape);
        var pstrides = StridesOf(pshape);
        var offsets = FootprintOffsets(fshape, footprint, pstrides);
        var size = SizeOf(shape);
        var result = new double[size];
        if (size == 0) return result;
        var last = shape[rank - 1];
        var rows = size / last;
        var scratchLen = op.ScratchLength(offsets.Length);
        Run(rows, (long)size * Math.Max(1, offsets.Length), (lo, hi) =>
        {
            var scratch = new double[scratchLen];
            for (var r = lo; r < hi; r++)
            {
                var pbase = 0;
                var rem = r;
                for (var a = rank - 2; a >= 0; a--)
                {
                    var c = rem % shape[a];
                    rem /= shape[a];
                    pbase += c * pstrides[a];
                }
                op.Row(padded, pbase, offsets, result.AsSpan(r * last, last), scratch);
            }
        });
        return result;
    }

    /// <summary>Linear offsets (in an array with <paramref name="strides"/>) of the true footprint entries, row-major.</summary>
    internal static int[] FootprintOffsets(int[] fshape, bool[] footprint, int[] strides)
    {
        var rank = fshape.Length;
        var list = new List<int>();
        var coord = new int[rank];
        for (var i = 0; i < footprint.Length; i++)
        {
            if (footprint[i])
            {
                var off = 0;
                for (var a = 0; a < rank; a++) off += coord[a] * strides[a];
                list.Add(off);
            }
            for (var a = rank - 1; a >= 0; a--)
            {
                if (++coord[a] < fshape[a]) break;
                coord[a] = 0;
            }
        }
        return list.ToArray();
    }

    internal static void CheckOrigins(int[] fshape, int[] origin)
    {
        for (var a = 0; a < fshape.Length; a++)
        {
            var lo = -(fshape[a] / 2);
            var hi = (fshape[a] - 1) / 2;
            if (origin[a] < lo || origin[a] > hi)
                throw new ArgumentException($"Origin {origin[a]} on axis {a} is outside [{lo}, {hi}] for a footprint of size {fshape[a]}.");
        }
    }

    /// <summary>numpy's pairwise summation (8 accumulators, blocks of 128), so normalised kernels match scipy to the bit where exp agrees.</summary>
    internal static double PairwiseSum(ReadOnlySpan<double> a)
    {
        var n = a.Length;
        if (n < 8)
        {
            var s = 0.0;
            for (var i = 0; i < n; i++) s += a[i];
            return s;
        }
        if (n <= 128)
        {
            Span<double> r = stackalloc double[8];
            for (var j = 0; j < 8; j++) r[j] = a[j];
            var i = 8;
            for (; i < n - n % 8; i += 8)
                for (var j = 0; j < 8; j++) r[j] += a[i + j];
            var res = (r[0] + r[1] + (r[2] + r[3])) + (r[4] + r[5] + (r[6] + r[7]));
            for (; i < n; i++) res += a[i];
            return res;
        }
        var n2 = n / 2;
        n2 -= n2 % 8;
        return PairwiseSum(a[..n2]) + PairwiseSum(a[n2..]);
    }

    /// <summary>Reads a boolean-as-double array: any non-zero (including NaN) is true.</summary>
    internal static bool[] ToMask(NdArray a)
    {
        var d = a.AsContiguous().AsSpan();
        var m = new bool[d.Length];
        for (var i = 0; i < d.Length; i++) m[i] = d[i] != 0;
        return m;
    }

    internal static NdArray FromMask(bool[] m, int[] shape)
    {
        var d = new double[m.Length];
        for (var i = 0; i < m.Length; i++) d[i] = m[i] ? 1.0 : 0.0;
        return new NdArray(d, shape);
    }
}
