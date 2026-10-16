using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Gravicode.Science.SciNet;

/// <summary>
/// 1-D correlation with scipy's symmetry detection (<c>NI_Correlate1D</c>). A kernel whose mirrored
/// taps agree to within an absolute DBL_EPSILON is treated as
/// symmetric (or antisymmetric) and summed in pairs, in the same order scipy uses, so the
/// floating-point result matches scipy's rather than merely being close to it.
/// </summary>
internal readonly struct NdImageCorrelateLineOp : INdImageLineOp
{
    private readonly double[] _w;
    private readonly int _symmetric;

    public NdImageCorrelateLineOp(double[] weights)
    {
        _w = weights;
        _symmetric = Symmetry(weights);
    }

    /// <summary>1 symmetric, -1 antisymmetric, 0 neither — scipy's test, including its absolute DBL_EPSILON tolerance.</summary>
    internal static int Symmetry(double[] w)
    {
        var size = w.Length;
        if ((size & 1) == 0) return 0;
        var c = size / 2;
        var eps = 2.220446049250313e-16;
        var sym = true;
        for (var i = 1; i <= c; i++)
            if (Math.Abs(w[c + i] - w[c - i]) > eps) { sym = false; break; }
        if (sym) return 1;
        for (var i = 1; i <= c; i++)
            if (Math.Abs(w[c + i] + w[c - i]) > eps) return 0;
        return -1;
    }

    public int ScratchLength(int n) => 0;

    public void Apply(ReadOnlySpan<double> ext, Span<double> output, Span<double> scratch)
    {
        var w = _w;
        var size = w.Length;
        var size1 = size / 2;
        var size2 = size - size1 - 1;
        var n = output.Length;
        ref var e = ref MemoryMarshal.GetReference(ext);
        ref var o = ref MemoryMarshal.GetReference(output);
        var vc = Vector<double>.Count;
        var k = 0;
        if (Vector.IsHardwareAccelerated && n >= vc)
        {
            if (_symmetric != 0)
            {
                var wc = new Vector<double>(w[size1]);
                for (; k <= n - vc; k += vc)
                {
                    var c = (nuint)(k + size1);
                    var acc = Vector.LoadUnsafe(ref e, c) * wc;
                    for (var j = size1; j >= 1; j--)
                    {
                        var l = Vector.LoadUnsafe(ref e, c - (nuint)j);
                        var r = Vector.LoadUnsafe(ref e, c + (nuint)j);
                        acc += (_symmetric > 0 ? l + r : l - r) * new Vector<double>(w[size1 - j]);
                    }
                    acc.StoreUnsafe(ref o, (nuint)k);
                }
            }
            else
            {
                var wl = new Vector<double>(w[size - 1]);
                for (; k <= n - vc; k += vc)
                {
                    var acc = Vector.LoadUnsafe(ref e, (nuint)(k + size - 1)) * wl;
                    for (var j = 0; j < size - 1; j++)
                        acc += Vector.LoadUnsafe(ref e, (nuint)(k + j)) * new Vector<double>(w[j]);
                    acc.StoreUnsafe(ref o, (nuint)k);
                }
            }
        }
        for (; k < n; k++)
        {
            double acc;
            if (_symmetric > 0)
            {
                var c = k + size1;
                acc = ext[c] * w[size1];
                for (var j = size1; j >= 1; j--) acc += (ext[c - j] + ext[c + j]) * w[size1 - j];
            }
            else if (_symmetric < 0)
            {
                var c = k + size1;
                acc = ext[c] * w[size1];
                for (var j = size1; j >= 1; j--) acc += (ext[c - j] - ext[c + j]) * w[size1 - j];
            }
            else
            {
                acc = ext[k + size1 + size2] * w[size - 1];
                for (var j = 0; j < size - 1; j++) acc += ext[k + j] * w[j];
            }
            output[k] = acc;
        }
    }
}

/// <summary>
/// Running mean (<c>NI_UniformFilter1D</c>): O(1) per sample whatever the size, with scipy's
/// update order (<c>tmp += (in - out) / size</c>) so the accumulated rounding is the same.
/// </summary>
internal readonly struct NdImageUniformLineOp(int size) : INdImageLineOp
{
    public int ScratchLength(int n) => 0;

    public void Apply(ReadOnlySpan<double> ext, Span<double> output, Span<double> scratch)
    {
        var tmp = 0.0;
        for (var i = 0; i < size; i++) tmp += ext[i];
        tmp /= size;
        output[0] = tmp;
        for (var k = 1; k < output.Length; k++)
        {
            tmp += (ext[k + size - 1] - ext[k - 1]) / size;
            output[k] = tmp;
        }
    }
}

/// <summary>
/// Sliding minimum or maximum by van Herk / Gil–Werman: block-wise prefix and suffix extrema make
/// every output one comparison, so the cost does not grow with the window.
/// </summary>
internal readonly struct NdImageMinMaxLineOp(int size, bool minimum) : INdImageLineOp
{
    public int ScratchLength(int n) => 2 * (n + size - 1);

    public void Apply(ReadOnlySpan<double> ext, Span<double> output, Span<double> scratch)
    {
        var n = output.Length;
        if (size == 1) { ext[..n].CopyTo(output); return; }
        var m = n + size - 1;
        var g = scratch[..m];
        var h = scratch.Slice(m, m);
        if (minimum)
        {
            for (var i = 0; i < m; i++) g[i] = i % size == 0 ? ext[i] : Math.Min(g[i - 1], ext[i]);
            for (var i = m - 1; i >= 0; i--) h[i] = (i % size == size - 1 || i == m - 1) ? ext[i] : Math.Min(h[i + 1], ext[i]);
            for (var k = 0; k < n; k++) output[k] = Math.Min(h[k], g[k + size - 1]);
        }
        else
        {
            for (var i = 0; i < m; i++) g[i] = i % size == 0 ? ext[i] : Math.Max(g[i - 1], ext[i]);
            for (var i = m - 1; i >= 0; i--) h[i] = (i % size == size - 1 || i == m - 1) ? ext[i] : Math.Max(h[i + 1], ext[i]);
            for (var k = 0; k < n; k++) output[k] = Math.Max(h[k], g[k + size - 1]);
        }
    }
}

/// <summary>Calls a user line function (<c>generic_filter1d</c>).</summary>
internal readonly struct NdImageGenericLineOp(NdImageLineFunction function) : INdImageLineOp
{
    public int ScratchLength(int n) => 0;

    public void Apply(ReadOnlySpan<double> ext, Span<double> output, Span<double> scratch) => function(ext, output);
}

/// <summary>Weighted sum over the footprint (<c>NI_Correlate</c>): accumulated from zero in footprint order.</summary>
internal readonly struct NdImageCorrelateWindowOp(double[] weights) : INdImageWindowOp
{
    public int ScratchLength(int count) => 0;

    public void Row(double[] padded, int baseIndex, int[] offsets, Span<double> output, Span<double> scratch)
    {
        var count = offsets.Length;
        for (var i = 0; i < output.Length; i++)
        {
            var b = baseIndex + i;
            var tmp = 0.0;
            for (var j = 0; j < count; j++) tmp += padded[b + offsets[j]] * weights[j];
            output[i] = tmp;
        }
    }
}

/// <summary>Minimum / maximum over a footprint, optionally of <c>value ∓ structure</c> (grey morphology).</summary>
internal readonly struct NdImageMinMaxWindowOp(bool minimum, double[]? structure) : INdImageWindowOp
{
    public int ScratchLength(int count) => 0;

    public void Row(double[] padded, int baseIndex, int[] offsets, Span<double> output, Span<double> scratch)
    {
        var count = offsets.Length;
        var s = structure;
        for (var i = 0; i < output.Length; i++)
        {
            var b = baseIndex + i;
            var tmp = padded[b + offsets[0]];
            if (s is not null) tmp = minimum ? tmp - s[0] : tmp + s[0];
            for (var j = 1; j < count; j++)
            {
                var t2 = padded[b + offsets[j]];
                if (s is not null) t2 = minimum ? t2 - s[j] : t2 + s[j];
                if (minimum) { if (t2 < tmp) tmp = t2; }
                else if (t2 > tmp) tmp = t2;
            }
            output[i] = tmp;
        }
    }
}

/// <summary>Order statistic over a footprint: gathers the values and runs an in-place quickselect.</summary>
internal readonly struct NdImageRankWindowOp(int rank) : INdImageWindowOp
{
    public int ScratchLength(int count) => count;

    public void Row(double[] padded, int baseIndex, int[] offsets, Span<double> output, Span<double> scratch)
    {
        var count = offsets.Length;
        var buf = scratch[..count];
        for (var i = 0; i < output.Length; i++)
        {
            var b = baseIndex + i;
            for (var j = 0; j < count; j++) buf[j] = padded[b + offsets[j]];
            output[i] = NdImageSelect.Select(buf, rank);
        }
    }
}

/// <summary>Calls a user function on the footprint values (<c>generic_filter</c>).</summary>
internal readonly struct NdImageGenericWindowOp(NdImageFootprintFunction function) : INdImageWindowOp
{
    public int ScratchLength(int count) => count;

    public void Row(double[] padded, int baseIndex, int[] offsets, Span<double> output, Span<double> scratch)
    {
        var count = offsets.Length;
        var buf = scratch[..count];
        for (var i = 0; i < output.Length; i++)
        {
            var b = baseIndex + i;
            for (var j = 0; j < count; j++) buf[j] = padded[b + offsets[j]];
            output[i] = function(buf);
        }
    }
}

/// <summary>Selection of the k-th smallest value.</summary>
internal static class NdImageSelect
{
    /// <summary>
    /// Returns the <paramref name="k"/>-th smallest element, permuting <paramref name="a"/>.
    /// Small windows are insertion-sorted (cheaper than partitioning below ~16 values).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static double Select(Span<double> a, int k)
    {
        var lo = 0;
        var hi = a.Length - 1;
        while (hi - lo > 16)
        {
            // median-of-three pivot
            var mid = lo + (hi - lo) / 2;
            if (a[mid] < a[lo]) (a[mid], a[lo]) = (a[lo], a[mid]);
            if (a[hi] < a[lo]) (a[hi], a[lo]) = (a[lo], a[hi]);
            if (a[hi] < a[mid]) (a[hi], a[mid]) = (a[mid], a[hi]);
            var pivot = a[mid];
            var i = lo;
            var j = hi;
            while (i <= j)
            {
                while (a[i] < pivot) i++;
                while (pivot < a[j]) j--;
                if (i <= j)
                {
                    (a[i], a[j]) = (a[j], a[i]);
                    i++;
                    j--;
                }
            }
            if (k <= j) hi = j;
            else if (k >= i) lo = i;
            else return a[k];
        }
        for (var i = lo + 1; i <= hi; i++)
        {
            var v = a[i];
            var j = i - 1;
            while (j >= lo && a[j] > v) { a[j + 1] = a[j]; j--; }
            a[j + 1] = v;
        }
        return a[k];
    }
}
