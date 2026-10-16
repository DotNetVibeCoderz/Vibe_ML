using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>A line transform whose output length may differ from its input length.</summary>
internal delegate void SignalResizeKernel(ReadOnlySpan<double> src, Span<double> dst);

/// <summary>
/// Runs a 1-D transform along one axis of an n-D array, gathering each (possibly strided) line
/// into a contiguous buffer first so kernels always see plain spans. Lines run in parallel once
/// there is enough total work.
/// </summary>
internal static class SignalAxisLines
{
    public static NdArray Map(NdArray x, int axis, int outLength, SignalResizeKernel kernel)
    {
        ArgumentNullException.ThrowIfNull(x);
        var rank = x.Rank;
        if (rank == 0) throw new ArgumentException("Input must have at least one dimension.", nameof(x));
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentException("axis out of range.", nameof(axis));
        var shape = x.Shape.ToArray();
        var len = shape[axis];
        var src = x.AsContiguous().ToArray();
        var outer = 1;
        for (var i = 0; i < axis; i++) outer *= shape[i];
        var inner = 1;
        for (var i = axis + 1; i < rank; i++) inner *= shape[i];
        var oshape = (int[])shape.Clone();
        oshape[axis] = outLength;
        var dst = new double[(long)outer * inner * outLength];
        var lines = outer * inner;
        void Line(int li, double[]? tin, double[]? tout)
        {
            var o = li / inner;
            var iIn = li % inner;
            var sBase = o * len * inner + iIn;
            var dBase = o * outLength * inner + iIn;
            if (inner == 1)
            {
                kernel(src.AsSpan(sBase, len), dst.AsSpan(dBase, outLength));
                return;
            }
            for (var k = 0; k < len; k++) tin![k] = src[sBase + k * inner];
            kernel(tin, tout);
            for (var k = 0; k < outLength; k++) dst[dBase + k * inner] = tout![k];
        }
        if (lines > 1 && (long)lines * Math.Max(len, outLength) >= 1 << 16)
            Parallel.For(0, lines, () => (new double[len], new double[outLength]),
                (li, _, buf) => { Line(li, buf.Item1, buf.Item2); return buf; }, _ => { });
        else
        {
            var tin = new double[len];
            var tout = new double[outLength];
            for (var li = 0; li < lines; li++) Line(li, tin, tout);
        }
        return new NdArray(dst, oshape);
    }

    public static int Length(NdArray x, int axis)
    {
        ArgumentNullException.ThrowIfNull(x);
        var a = axis < 0 ? axis + x.Rank : axis;
        if (a < 0 || a >= x.Rank) throw new ArgumentException("axis out of range.", nameof(axis));
        return x.Shape[a];
    }
}
