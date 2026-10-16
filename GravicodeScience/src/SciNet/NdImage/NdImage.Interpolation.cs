using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.NdImageSupport;

namespace Gravicode.Science.SciNet;

/// <summary>
/// In-place B-spline prefilter of one line (scipy <c>ni_splines.c</c>): gain, then for each pole
/// a causal and an anti-causal first-order recursion whose initial values depend on the mode.
/// </summary>
internal readonly struct NdImageSplineLineOp(int order, NdImageMode mode) : INdImageLineOp
{
    public int ScratchLength(int n) => 0;

    public void Apply(ReadOnlySpan<double> ext, Span<double> output, Span<double> scratch)
    {
        ext[..output.Length].CopyTo(output);
        NdImageSpline.FilterLine(output, order, mode);
    }
}

/// <summary>B-spline poles, prefilter and interpolation weights, transcribed from scipy's <c>ni_splines.c</c>.</summary>
internal static class NdImageSpline
{
    internal static int Poles(int order, Span<double> poles)
    {
        switch (order)
        {
            case 2:
                poles[0] = Math.Sqrt(8.0) - 3.0;
                return 1;
            case 3:
                poles[0] = Math.Sqrt(3.0) - 2.0;
                return 1;
            case 4:
                poles[0] = Math.Sqrt(664.0 - Math.Sqrt(438976.0)) + Math.Sqrt(304.0) - 19.0;
                poles[1] = Math.Sqrt(664.0 + Math.Sqrt(438976.0)) - Math.Sqrt(304.0) - 19.0;
                return 2;
            case 5:
                poles[0] = Math.Sqrt(67.5 - Math.Sqrt(4436.25)) + Math.Sqrt(26.25) - 6.5;
                poles[1] = Math.Sqrt(67.5 + Math.Sqrt(4436.25)) - Math.Sqrt(26.25) - 6.5;
                return 2;
            default:
                return 0;
        }
    }

    /// <summary>
    /// Converts samples to B-spline coefficients. The boundary initialisation is what makes the
    /// spline honour the mode: Nearest and Reflect use the half-sample symmetric start, GridWrap
    /// the periodic one, and Mirror, Wrap, Constant and GridConstant the whole-sample symmetric
    /// one (scipy prepads Nearest and GridConstant before this runs).
    /// </summary>
    internal static void FilterLine(Span<double> c, int order, NdImageMode mode)
    {
        var n = c.Length;
        if (order < 2 || n <= 1) return;
        Span<double> zs = stackalloc double[2];
        var np = Poles(order, zs);
        var gain = 1.0;
        for (var i = 0; i < np; i++) gain *= (1.0 - zs[i]) * (1.0 - 1.0 / zs[i]);
        for (var i = 0; i < n; i++) c[i] *= gain;
        for (var p = 0; p < np; p++)
        {
            var z = zs[p];
            switch (mode)
            {
                case NdImageMode.GridWrap: CausalWrap(c, z); break;
                case NdImageMode.Nearest:
                case NdImageMode.Reflect:
                case NdImageMode.GridMirror: CausalReflect(c, z); break;
                default: CausalMirror(c, z); break;
            }
            for (var i = 1; i < n; i++) c[i] += z * c[i - 1];
            switch (mode)
            {
                case NdImageMode.GridWrap: AntiCausalWrap(c, z); break;
                case NdImageMode.Nearest:
                case NdImageMode.Reflect:
                case NdImageMode.GridMirror: c[n - 1] *= z / (z - 1); break;
                default: c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1); break;
            }
            for (var i = n - 2; i >= 0; i--) c[i] = z * (c[i + 1] - c[i]);
        }
    }

    private static void CausalMirror(Span<double> c, double z)
    {
        var n = c.Length;
        var zi = z;
        var zn1 = Math.Pow(z, n - 1);
        c[0] = c[0] + zn1 * c[n - 1];
        for (var i = 1; i < n - 1; i++)
        {
            c[0] += zi * (c[i] + zn1 * c[n - 1 - i]);
            zi *= z;
        }
        c[0] /= 1 - zn1 * zn1;
    }

    private static void CausalReflect(Span<double> c, double z)
    {
        var n = c.Length;
        var zi = z;
        var zn = Math.Pow(z, n);
        var c0 = c[0];
        c[0] = c[0] + zn * c[n - 1];
        for (var i = 1; i < n; i++)
        {
            c[0] += zi * (c[i] + zn * c[n - 1 - i]);
            zi *= z;
        }
        c[0] *= z / (1 - zn * zn);
        c[0] += c0;
    }

    private static void CausalWrap(Span<double> c, double z)
    {
        var n = c.Length;
        var zi = z;
        for (var i = 1; i < n; i++)
        {
            c[0] += zi * c[n - i];
            zi *= z;
        }
        c[0] /= 1 - zi;
    }

    private static void AntiCausalWrap(Span<double> c, double z)
    {
        var n = c.Length;
        var zi = z;
        for (var i = 0; i < n - 1; i++)
        {
            c[n - 1] += zi * c[i];
            zi *= z;
        }
        c[n - 1] *= z / (zi - 1);
    }

    /// <summary>
    /// The <c>order + 1</c> B-spline weights for a sample at <paramref name="x"/> (scipy's
    /// <c>get_spline_interpolation_weights</c>); the last weight is 1 minus the others.
    /// </summary>
    internal static void Weights(double x, int order, Span<double> w)
    {
        if (order == 0) { w[0] = 1.0; return; }
        x -= Math.Floor((order & 1) != 0 ? x : x + 0.5);
        var y = x;
        var z = 1.0 - x;
        double t;
        switch (order)
        {
            case 1:
                w[0] = 1.0 - x;
                break;
            case 2:
                w[1] = 0.75 - x * x;
                y = 0.5 - x;
                w[0] = 0.5 * y * y;
                break;
            case 3:
                w[1] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
                w[2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0;
                y = z * z * z;
                w[0] = y / 6.0;
                break;
            case 4:
                t = x * x;
                w[2] = t * (t * 0.25 - 0.625) + 115.0 / 192.0;
                y = 1.0 + x;
                w[1] = y * (y * (y * (5.0 - y) / 6.0 - 1.25) + 5.0 / 24.0) + 55.0 / 96.0;
                w[3] = z * (z * (z * (5.0 - z) / 6.0 - 1.25) + 5.0 / 24.0) + 55.0 / 96.0;
                t = 0.5 - x;
                t *= t;
                w[0] = t * t / 24.0;
                break;
            case 5:
                t = y * y;
                w[2] = t * (t * (0.25 - y / 12.0) - 0.5) + 0.55;
                t = z * z;
                w[3] = t * (t * (0.25 - z / 12.0) - 0.5) + 0.55;
                y += 1.0;
                w[1] = y * (y * (y * (y * (y / 24.0 - 0.375) + 1.25) - 1.75) + 0.625) + 0.425;
                z += 1.0;
                w[4] = z * (z * (z * (z * (z / 24.0 - 0.375) + 1.25) - 1.75) + 0.625) + 0.425;
                t = 1.0 - x;
                y = t * t;
                w[0] = t * y * y / 120.0;
                break;
        }
        t = 0.0;
        for (var i = 0; i < order; i++) t += w[i];
        w[order] = 1.0 - t;
    }

    /// <summary>
    /// Maps a coordinate outside <c>[0, n-1]</c> back inside (scipy's <c>map_coordinate</c>).
    /// Constant returns -1, which the caller turns into cval; note Wrap's period is n-1.
    /// </summary>
    internal static double MapCoordinate(double x, int len, NdImageMode mode)
    {
        if (x < 0)
        {
            switch (mode)
            {
                case NdImageMode.Mirror:
                    if (len <= 1) return 0;
                    {
                        long s2 = 2L * len - 2;
                        x = s2 * (long)(-x / s2) + x;
                        return x <= 1 - len ? x + s2 : -x;
                    }
                case NdImageMode.Reflect:
                    if (len <= 1) return 0;
                    {
                        long s2 = 2L * len;
                        if (x < -s2) x = s2 * (long)(-x / s2) + x;
                        return x < -len ? x + s2 : (x > -1e-15 ? 1e-15 : -x) - 1;
                    }
                case NdImageMode.Wrap:
                    if (len <= 1) return 0;
                    {
                        long sz = len - 1;
                        return x + sz * ((long)(-x / sz) + 1);
                    }
                case NdImageMode.GridWrap:
                    if (len <= 1) return 0;
                    return x + len * ((long)((-1 - x) / len) + 1);
                case NdImageMode.Nearest:
                    return 0;
                default:
                    return -1;
            }
        }
        if (x > len - 1)
        {
            switch (mode)
            {
                case NdImageMode.Mirror:
                    if (len <= 1) return 0;
                    {
                        long s2 = 2L * len - 2;
                        x -= s2 * (long)(x / s2);
                        return x >= len ? s2 - x : x;
                    }
                case NdImageMode.Reflect:
                    if (len <= 1) return 0;
                    {
                        long s2 = 2L * len;
                        x -= s2 * (long)(x / s2);
                        return x >= len ? s2 - x - 1 : x;
                    }
                case NdImageMode.Wrap:
                    if (len <= 1) return 0;
                    {
                        long sz = len - 1;
                        return x - sz * (long)(x / sz);
                    }
                case NdImageMode.GridWrap:
                    if (len <= 1) return 0;
                    return x - len * (long)(x / len);
                case NdImageMode.Nearest:
                    return len - 1;
                default:
                    return -1;
            }
        }
        return x;
    }

    /// <summary>
    /// Fills the taps for one axis: <paramref name="offsets"/> receives the index of each tap times
    /// the axis stride (or -1 for a cval tap under GridConstant). Returns false when the whole
    /// sample is cval (Constant mode outside the grid). Taps falling outside the grid are mirrored
    /// for Mirror, Constant and Wrap — scipy does this for Wrap too, not wrapping them.
    /// </summary>
    internal static bool AxisTaps(double cc, int n, int stride, int order, NdImageMode mode, Span<int> offsets, Span<double> w)
    {
        var raw = mode is NdImageMode.GridConstant or NdImageMode.Nearest;
        if (!raw) cc = MapCoordinate(cc, n, mode);
        if (!(cc > -1.0 || raw)) return false;
        var start = (order & 1) != 0 ? (long)Math.Floor(cc) - order / 2 : (long)Math.Floor(cc + 0.5) - order / 2;
        var tapMode = mode switch
        {
            NdImageMode.Constant or NdImageMode.Wrap => NdImageMode.Mirror,
            NdImageMode.GridMirror => NdImageMode.Reflect,
            _ => mode,
        };
        for (var t = 0; t <= order; t++)
        {
            var i = start + t;
            int idx;
            if (i >= 0 && i < n) idx = (int)i;
            else if (tapMode == NdImageMode.GridConstant) { offsets[t] = -1; continue; }
            else idx = ExtendIndex((int)Math.Clamp(i, int.MinValue / 2, int.MaxValue / 2), n, tapMode);
            offsets[t] = idx * stride;
        }
        Weights(cc, order, w);
        return true;
    }

    /// <summary>
    /// Tensor-product sum over the taps, in scipy's order: each coefficient times the per-axis
    /// weights in axis order, accumulated over taps in row-major order.
    /// </summary>
    internal static double Combine(double[] c, int rank, int taps, ReadOnlySpan<int> off, ReadOnlySpan<double> w, bool gridConstant, double cval, Span<int> counter)
    {
        if (rank == 1)
        {
            var t = 0.0;
            for (var a = 0; a < taps; a++)
            {
                var o = off[a];
                var v = gridConstant && o < 0 ? cval : c[o];
                t += v * w[a];
            }
            return t;
        }
        if (rank == 2)
        {
            var t = 0.0;
            for (var a = 0; a < taps; a++)
            {
                var o0 = off[a];
                var w0 = w[a];
                for (var b = 0; b < taps; b++)
                {
                    var o1 = off[taps + b];
                    var v = gridConstant && (o0 < 0 || o1 < 0) ? cval : c[o0 + o1];
                    t += v * w0 * w[taps + b];
                }
            }
            return t;
        }
        counter[..rank].Clear();
        var total = 0.0;
        while (true)
        {
            var idx = 0;
            var outside = false;
            var coeff = 1.0;
            for (var a = 0; a < rank; a++)
            {
                var o = off[a * taps + counter[a]];
                if (gridConstant && o < 0) outside = true;
                idx += o;
            }
            coeff = outside ? cval : c[idx];
            for (var a = 0; a < rank; a++) coeff *= w[a * taps + counter[a]];
            total += coeff;
            var ax = rank - 1;
            for (; ax >= 0; ax--)
            {
                if (++counter[ax] < taps) break;
                counter[ax] = 0;
            }
            if (ax < 0) break;
        }
        return total;
    }
}

public static partial class NdImage
{
    /// <summary>
    /// B-spline prefilter along one axis (scipy.ndimage.spline_filter1d): turns samples into the
    /// coefficients an order-<paramref name="order"/> spline interpolates. Orders 0 and 1 are the identity.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="order">Spline order, 0–5.</param>
    /// <param name="axis">Axis to filter along.</param>
    /// <param name="mode">Boundary condition the spline should satisfy (default Mirror, as scipy).</param>
    public static NdArray SplineFilter1d(NdArray input, int order = 3, int axis = -1, NdImageMode mode = NdImageMode.Mirror)
    {
        ArgumentNullException.ThrowIfNull(input);
        CheckOrder(order);
        var shape = ShapeOf(input);
        var ax = NormalizeAxis(axis, shape.Length);
        var data = input.ToArray();
        if (order > 1) AlongAxis(data, data, shape, ax, 0, 0, mode, 0.0, new NdImageSplineLineOp(order, mode));
        return new NdArray(data, shape);
    }

    /// <summary>Multidimensional B-spline prefilter (scipy.ndimage.spline_filter): <see cref="SplineFilter1d"/> along every axis.</summary>
    /// <param name="input">Input array.</param>
    /// <param name="order">Spline order, 0–5.</param>
    /// <param name="mode">Boundary condition (default Mirror).</param>
    public static NdArray SplineFilter(NdArray input, int order = 3, NdImageMode mode = NdImageMode.Mirror)
    {
        ArgumentNullException.ThrowIfNull(input);
        CheckOrder(order);
        var shape = ShapeOf(input);
        var data = input.ToArray();
        SplineFilterInPlace(data, shape, order, mode);
        return new NdArray(data, shape);
    }

    private static void SplineFilterInPlace(double[] data, int[] shape, int order, NdImageMode mode)
    {
        if (order < 2) return;
        for (var a = 0; a < shape.Length; a++)
            AlongAxis(data, data, shape, a, 0, 0, mode, 0.0, new NdImageSplineLineOp(order, mode));
    }

    private static void CheckOrder(int order)
    {
        if (order < 0 || order > 5) throw new ArgumentException("Spline order must be in 0..5.");
    }

    /// <summary>Spline coefficients ready for sampling, plus the padding scipy adds for Nearest / GridConstant.</summary>
    private sealed class NdImageSplineSource
    {
        public required double[] Coefficients;
        public required int[] Dims;
        public required int[] Strides;
        public required int Order;
        public required NdImageMode Mode;
        public required double Cval;
        public required int Pad;

        public int Rank => Dims.Length;
    }

    /// <summary>
    /// scipy's prefilter step: Nearest and GridConstant have no exact spline boundary, so the input
    /// is first padded by 12 samples (edge or cval) and the spline filtered in that larger domain.
    /// </summary>
    private static NdImageSplineSource PrepareSpline(NdArray input, int order, NdImageMode mode, double cval, bool prefilter)
    {
        CheckOrder(order);
        var shape = ShapeOf(input);
        if (shape.Contains(0)) throw new ArgumentException("Cannot interpolate an empty array.");
        var data = input.ToArray();
        var npad = 0;
        if (prefilter && order > 1)
        {
            if (mode is NdImageMode.Nearest or NdImageMode.GridConstant)
            {
                npad = 12;
                var pad = Enumerable.Repeat(npad, shape.Length).ToArray();
                data = Pad(data, shape, pad, pad, mode == NdImageMode.Nearest ? NdImageMode.Nearest : NdImageMode.Constant, cval, out var ps);
                shape = ps;
            }
            SplineFilterInPlace(data, shape, order, mode);
        }
        return new NdImageSplineSource
        {
            Coefficients = data,
            Dims = shape,
            Strides = StridesOf(shape),
            Order = order,
            Mode = mode == NdImageMode.GridMirror ? NdImageMode.Reflect : mode,
            Cval = cval,
            Pad = npad,
        };
    }

    /// <summary>Per-worker scratch for sampling.</summary>
    private sealed class NdImageSampler(NdImageSplineSource s)
    {
        private readonly int[] _off = new int[s.Rank * (s.Order + 1)];
        private readonly double[] _w = new double[s.Rank * (s.Order + 1)];
        private readonly int[] _counter = new int[s.Rank];

        public double Sample(ReadOnlySpan<double> coord)
        {
            var taps = s.Order + 1;
            for (var a = 0; a < s.Rank; a++)
            {
                if (!NdImageSpline.AxisTaps(coord[a] + s.Pad, s.Dims[a], s.Strides[a], s.Order, s.Mode,
                        _off.AsSpan(a * taps, taps), _w.AsSpan(a * taps, taps)))
                    return s.Cval;
            }
            return NdImageSpline.Combine(s.Coefficients, s.Rank, taps, _off, _w, s.Mode == NdImageMode.GridConstant, s.Cval, _counter);
        }
    }

    /// <summary>
    /// Samples the input at arbitrary coordinates (scipy.ndimage.map_coordinates).
    /// </summary>
    /// <param name="input">Input array of rank r.</param>
    /// <param name="coordinates">Shape <c>(r, ...)</c>: <c>coordinates[a, ...]</c> is the axis-a coordinate of each output point.</param>
    /// <param name="order">Spline order, 0–5.</param>
    /// <param name="mode">Boundary mode (interpolation meaning; see <see cref="NdImageMode"/>).</param>
    /// <param name="cval">Value outside the grid for Constant / GridConstant.</param>
    /// <param name="prefilter">Apply the spline prefilter first (set false if the input already holds coefficients).</param>
    /// <returns>An array of shape <c>coordinates.Shape[1..]</c>.</returns>
    public static NdArray MapCoordinates(NdArray input, NdArray coordinates, int order = 3, NdImageMode mode = NdImageMode.Constant,
        double cval = 0.0, bool prefilter = true)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(coordinates);
        var rank = input.Rank;
        var cshape = ShapeOf(coordinates);
        if (cshape[0] != rank) throw new ArgumentException($"Coordinates must have {rank} rows (one per input axis).");
        var outShape = cshape.Length > 1 ? cshape[1..] : [1];
        var src = PrepareSpline(input, order, mode, cval, prefilter);
        var coords = coordinates.ToArray();
        var npts = coords.Length / rank;
        var result = new double[npts];
        Run(npts, (long)npts * (order + 1) * (order + 1) * 4, (lo, hi) =>
        {
            var sampler = new NdImageSampler(src);
            var c = new double[rank];
            for (var p = lo; p < hi; p++)
            {
                for (var a = 0; a < rank; a++) c[a] = coords[a * npts + p];
                result[p] = sampler.Sample(c);
            }
        });
        return new NdArray(result, outShape);
    }

    /// <summary>
    /// Applies an arbitrary mapping from output to input coordinates (scipy.ndimage.geometric_transform).
    /// The mapping is called once per output point, serially, so it need not be thread-safe.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="mapping">Receives an output coordinate, returns the input coordinate (length = input rank).</param>
    /// <param name="outputShape">Output shape (defaults to the input shape).</param>
    /// <param name="order">Spline order, 0–5.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Value outside the grid for Constant / GridConstant.</param>
    /// <param name="prefilter">Apply the spline prefilter first.</param>
    public static NdArray GeometricTransform(NdArray input, Func<ReadOnlySpan<double>, double[]> mapping, int[]? outputShape = null,
        int order = 3, NdImageMode mode = NdImageMode.Constant, double cval = 0.0, bool prefilter = true)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(mapping);
        var oshape = outputShape ?? ShapeOf(input);
        var src = PrepareSpline(input, order, mode, cval, prefilter);
        var size = SizeOf(oshape);
        var result = new double[size];
        var sampler = new NdImageSampler(src);
        var oc = new double[oshape.Length];
        var counter = new int[oshape.Length];
        for (var p = 0; p < size; p++)
        {
            for (var a = 0; a < oshape.Length; a++) oc[a] = counter[a];
            var ic = mapping(oc);
            if (ic is null || ic.Length != src.Rank) throw new ArgumentException($"The mapping must return {src.Rank} coordinates.");
            result[p] = sampler.Sample(ic);
            for (var a = oshape.Length - 1; a >= 0; a--)
            {
                if (++counter[a] < oshape[a]) break;
                counter[a] = 0;
            }
        }
        return new NdArray(result, oshape);
    }

    /// <summary>
    /// Affine transform (scipy.ndimage.affine_transform): output point <c>o</c> samples the input at
    /// <c>matrix · o + offset</c>. The matrix may be <c>r×r</c>, homogeneous <c>r×(r+1)</c> or
    /// <c>(r+1)×(r+1)</c> (its last column is the offset), or a length-r vector meaning a diagonal
    /// matrix — that last case takes scipy's separable zoom/shift path.
    /// </summary>
    /// <param name="input">Input array of rank r.</param>
    /// <param name="matrix">The transform (see summary). For an r×m matrix the output has rank m.</param>
    /// <param name="offset">Offset (one value broadcasts); ignored for a homogeneous matrix.</param>
    /// <param name="outputShape">Output shape (defaults to the input shape).</param>
    /// <param name="order">Spline order, 0–5.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Value outside the grid for Constant / GridConstant.</param>
    /// <param name="prefilter">Apply the spline prefilter first.</param>
    public static NdArray AffineTransform(NdArray input, NdArray matrix, double[]? offset = null, int[]? outputShape = null,
        int order = 3, NdImageMode mode = NdImageMode.Constant, double cval = 0.0, bool prefilter = true)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(matrix);
        var rank = input.Rank;
        var oshape = outputShape ?? ShapeOf(input);
        var off = Expand(offset, rank, 0.0, nameof(offset));
        var m = matrix.ToArray();
        if (matrix.Rank == 1)
        {
            if (m.Length != rank) throw new ArgumentException("A diagonal matrix needs one entry per input axis.");
            if (oshape.Length != rank) throw new ArgumentException("Output rank must equal the input rank for a diagonal matrix.");
            var shift = new double[rank];
            for (var a = 0; a < rank; a++) shift[a] = off[a] / m[a];
            return ZoomShift(input, m, shift, oshape, order, mode, cval, prefilter, gridMode: false);
        }
        if (matrix.Rank != 2) throw new ArgumentException("Matrix must be 1-D or 2-D.");
        var rows = matrix.Shape[0];
        var cols = matrix.Shape[1];
        if (cols == rank + 1 && (rows == rank || rows == rank + 1))
        {
            if (rows == rank + 1)
            {
                for (var j = 0; j < rank; j++)
                    if (m[rank * cols + j] != 0) throw new ArgumentException("Homogeneous matrix's last row must be [0, ..., 0, 1].");
                if (m[rank * cols + rank] != 1) throw new ArgumentException("Homogeneous matrix's last row must be [0, ..., 0, 1].");
            }
            var mm = new double[rank * rank];
            for (var i = 0; i < rank; i++)
            {
                off[i] = m[i * cols + rank];
                for (var j = 0; j < rank; j++) mm[i * rank + j] = m[i * cols + j];
            }
            m = mm;
            cols = rank;
            rows = rank;
        }
        if (rows != rank) throw new ArgumentException("Matrix must have one row per input axis.");
        if (cols != oshape.Length) throw new ArgumentException("Matrix must have one column per output axis.");
        var src = PrepareSpline(input, order, mode, cval, prefilter);
        var orank = oshape.Length;
        var size = SizeOf(oshape);
        var result = new double[size];
        var last = orank == 0 ? 1 : oshape[orank - 1];
        var nrows = last == 0 ? 0 : size / last;
        Run(nrows, (long)size * (order + 1) * (order + 1) * 4, (lo, hi) =>
        {
            var sampler = new NdImageSampler(src);
            var oc = new int[orank];
            var ic = new double[rank];
            for (var r = lo; r < hi; r++)
            {
                var rem = r;
                for (var a = orank - 2; a >= 0; a--) { oc[a] = rem % oshape[a]; rem /= oshape[a]; }
                for (var k = 0; k < last; k++)
                {
                    oc[orank - 1] = k;
                    for (var h = 0; h < rank; h++)
                    {
                        // offset first, then the products: scipy's summation order, which decides
                        // which side of a .5 boundary an order-0 sample lands on
                        var cc = off[h];
                        for (var l = 0; l < orank; l++) cc += oc[l] * m[h * orank + l];
                        ic[h] = cc;
                    }
                    result[r * last + k] = sampler.Sample(ic);
                }
            }
        });
        return new NdArray(result, oshape);
    }

    /// <summary>
    /// Shifts the array by a sub-pixel amount (scipy.ndimage.shift): output[k] samples input[k − shift].
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="shift">Shift per axis (one value broadcasts).</param>
    /// <param name="order">Spline order, 0–5.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Value outside the grid for Constant / GridConstant.</param>
    /// <param name="prefilter">Apply the spline prefilter first.</param>
    public static NdArray Shift(NdArray input, double[] shift, int order = 3, NdImageMode mode = NdImageMode.Constant,
        double cval = 0.0, bool prefilter = true)
    {
        ArgumentNullException.ThrowIfNull(input);
        var s = Expand(shift, input.Rank, 0.0, nameof(shift));
        for (var a = 0; a < s.Length; a++) s[a] = -s[a];
        return ZoomShift(input, null, s, ShapeOf(input), order, mode, cval, prefilter, gridMode: false);
    }

    /// <summary>
    /// Resamples to <c>round(shape · zoom)</c> (scipy.ndimage.zoom, banker's rounding as Python's
    /// <c>round</c>). By default the corner samples align (<c>(n_in − 1)/(n_out − 1)</c>); with
    /// <paramref name="gridMode"/> the pixel edges align instead (<c>n_in / n_out</c>, sampled at
    /// pixel centres) — pair it with GridConstant or GridWrap, as scipy advises.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="zoom">Zoom factor per axis (one value broadcasts).</param>
    /// <param name="order">Spline order, 0–5.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Value outside the grid for Constant / GridConstant.</param>
    /// <param name="prefilter">Apply the spline prefilter first.</param>
    /// <param name="gridMode">Align pixel edges rather than pixel centres.</param>
    public static NdArray Zoom(NdArray input, double[] zoom, int order = 3, NdImageMode mode = NdImageMode.Constant,
        double cval = 0.0, bool prefilter = true, bool gridMode = false)
    {
        ArgumentNullException.ThrowIfNull(input);
        var rank = input.Rank;
        var z = Expand(zoom, rank, 1.0, nameof(zoom));
        var ishape = ShapeOf(input);
        var oshape = new int[rank];
        var factors = new double[rank];
        for (var a = 0; a < rank; a++)
        {
            oshape[a] = (int)Math.Round(ishape[a] * z[a], MidpointRounding.ToEven);
            double div = gridMode ? oshape[a] : oshape[a] - 1;
            double nom = gridMode ? ishape[a] : ishape[a] - 1;
            factors[a] = div != 0 ? nom / div : 1.0;
        }
        return ZoomShift(input, factors, null, oshape, order, mode, cval, prefilter, gridMode);
    }

    /// <summary>
    /// scipy's <c>NI_ZoomShift</c>: the transform is separable, so every axis's taps and weights are
    /// tabulated once per output index and reused for every point on that slice.
    /// </summary>
    private static NdArray ZoomShift(NdArray input, double[]? zoom, double[]? shift, int[] oshape, int order, NdImageMode mode,
        double cval, bool prefilter, bool gridMode)
    {
        var src = PrepareSpline(input, order, mode, cval, prefilter);
        var rank = src.Rank;
        if (oshape.Length != rank) throw new ArgumentException("Output rank must equal the input rank.");
        var taps = order + 1;
        var zero = new bool[rank][];
        var offs = new int[rank][];
        var ws = new double[rank][];
        for (var a = 0; a < rank; a++)
        {
            var n = oshape[a];
            zero[a] = new bool[n];
            offs[a] = new int[n * taps];
            ws[a] = new double[n * taps];
            for (var k = 0; k < n; k++)
            {
                double cc = k;
                if (shift is not null) cc += shift[a];
                if (zoom is not null)
                {
                    if (gridMode)
                    {
                        cc += 0.5;
                        cc *= zoom[a];
                        cc -= 0.5;
                    }
                    else cc *= zoom[a];
                }
                cc += src.Pad;
                zero[a][k] = !NdImageSpline.AxisTaps(cc, src.Dims[a], src.Strides[a], order, src.Mode,
                    offs[a].AsSpan(k * taps, taps), ws[a].AsSpan(k * taps, taps));
            }
        }
        var size = SizeOf(oshape);
        var result = new double[size];
        if (size == 0) return new NdArray(result, oshape);
        var last = oshape[rank - 1];
        var nrows = size / last;
        var gridConstant = src.Mode == NdImageMode.GridConstant;
        Run(nrows, (long)size * taps * taps * 2, (lo, hi) =>
        {
            var off = new int[rank * taps];
            var w = new double[rank * taps];
            var counter = new int[rank];
            var oc = new int[rank];
            for (var r = lo; r < hi; r++)
            {
                var rem = r;
                var rowZero = false;
                for (var a = rank - 2; a >= 0; a--)
                {
                    oc[a] = rem % oshape[a];
                    rem /= oshape[a];
                    if (zero[a][oc[a]]) rowZero = true;
                    Array.Copy(offs[a], oc[a] * taps, off, a * taps, taps);
                    Array.Copy(ws[a], oc[a] * taps, w, a * taps, taps);
                }
                var outRow = result.AsSpan(r * last, last);
                if (rowZero) { outRow.Fill(cval); continue; }
                var la = rank - 1;
                for (var k = 0; k < last; k++)
                {
                    if (zero[la][k]) { outRow[k] = cval; continue; }
                    Array.Copy(offs[la], k * taps, off, la * taps, taps);
                    Array.Copy(ws[la], k * taps, w, la * taps, taps);
                    outRow[k] = NdImageSpline.Combine(src.Coefficients, rank, taps, off, w, gridConstant, cval, counter);
                }
            }
        });
        return new NdArray(result, oshape);
    }

    /// <summary>
    /// Rotates in the plane of two axes (scipy.ndimage.rotate). With <paramref name="reshape"/> the
    /// output grows to hold the whole rotated input; arrays of rank above 2 are rotated plane by
    /// plane, so the prefilter never runs along the untouched axes.
    /// </summary>
    /// <param name="input">Input array, rank ≥ 2.</param>
    /// <param name="angle">Angle in degrees.</param>
    /// <param name="axes">The two axes of the rotation plane (default (1, 0), as scipy).</param>
    /// <param name="reshape">Grow the output to contain the rotated input.</param>
    /// <param name="order">Spline order, 0–5.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Value outside the grid for Constant / GridConstant.</param>
    /// <param name="prefilter">Apply the spline prefilter first.</param>
    public static NdArray Rotate(NdArray input, double angle, (int, int)? axes = null, bool reshape = true, int order = 3,
        NdImageMode mode = NdImageMode.Constant, double cval = 0.0, bool prefilter = true)
    {
        ArgumentNullException.ThrowIfNull(input);
        var rank = input.Rank;
        if (rank < 2) throw new ArgumentException("Input array should be at least 2-D.");
        var (ax0, ax1) = axes ?? (1, 0);
        ax0 = NormalizeAxis(ax0, rank);
        ax1 = NormalizeAxis(ax1, rank);
        if (ax0 == ax1) throw new ArgumentException("Rotation axes must differ.");
        if (ax0 > ax1) (ax0, ax1) = (ax1, ax0);
        var (c, s) = CosSinDegrees(angle);
        var ishape = ShapeOf(input);
        int iy = ishape[ax0], ix = ishape[ax1];
        int oy, ox;
        if (reshape)
        {
            // rot · [[0, 0, iy, iy], [0, ix, 0, ix]], then ptp + 0.5 truncated
            double[] r0 = [0, s * ix, c * iy, c * iy + s * ix];
            double[] r1 = [0, c * ix, -s * iy, -s * iy + c * ix];
            oy = (int)(r0.Max() - r0.Min() + 0.5);
            ox = (int)(r1.Max() - r1.Min() + 0.5);
        }
        else
        {
            oy = iy;
            ox = ix;
        }
        var ocy = (oy - 1) / 2.0;
        var ocx = (ox - 1) / 2.0;
        var outCenter0 = c * ocy + s * ocx;
        var outCenter1 = -s * ocy + c * ocx;
        var offset = new[] { (iy - 1) / 2.0 - outCenter0, (ix - 1) / 2.0 - outCenter1 };
        var rot = new NdArray([c, s, -s, c], 2, 2);
        if (rank == 2)
            return AffineTransform(input, rot, offset, [oy, ox], order, mode, cval, prefilter);

        var oshape = (int[])ishape.Clone();
        oshape[ax0] = oy;
        oshape[ax1] = ox;
        var src = input.ToArray();
        var istr = StridesOf(ishape);
        var ostr = StridesOf(oshape);
        var result = new double[SizeOf(oshape)];
        var others = Enumerable.Range(0, rank).Where(a => a != ax0 && a != ax1).ToArray();
        var planes = others.Aggregate(1, (p, a) => p * ishape[a]);
        var plane = new double[iy * ix];
        for (var p = 0; p < planes; p++)
        {
            var rem = p;
            int ibase = 0, obase = 0;
            for (var q = others.Length - 1; q >= 0; q--)
            {
                var a = others[q];
                var cidx = rem % ishape[a];
                rem /= ishape[a];
                ibase += cidx * istr[a];
                obase += cidx * ostr[a];
            }
            for (var y = 0; y < iy; y++)
                for (var x = 0; x < ix; x++)
                    plane[y * ix + x] = src[ibase + y * istr[ax0] + x * istr[ax1]];
            var rotated = AffineTransform(new NdArray(plane, iy, ix), rot, offset, [oy, ox], order, mode, cval, prefilter).ToArray();
            for (var y = 0; y < oy; y++)
                for (var x = 0; x < ox; x++)
                    result[obase + y * ostr[ax0] + x * ostr[ax1]] = rotated[y * ox + x];
        }
        return new NdArray(result, oshape);
    }

    /// <summary>
    /// Cosine and sine of an angle in degrees, exact at multiples of 90° (as scipy's
    /// <c>cosdg</c>/<c>sindg</c>), so a quarter turn is an exact permutation rather than one
    /// polluted by <c>cos(π/2) ≈ 6e-17</c>.
    /// </summary>
    internal static (double Cos, double Sin) CosSinDegrees(double angle)
    {
        var x = angle % 360.0;
        if (x < 0) x += 360.0;
        if (x == 0) return (1, 0);
        if (x == 90) return (0, 1);
        if (x == 180) return (-1, 0);
        if (x == 270) return (0, -1);
        var r = x * (Math.PI / 180.0);
        return (Math.Cos(r), Math.Sin(r));
    }
}
