using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.NdImageSupport;

namespace Gravicode.Science.SciNet;

public static partial class NdImage
{
    /// <summary>
    /// Exact Euclidean distance from every non-zero element to the nearest zero element
    /// (scipy.ndimage.distance_transform_edt).
    /// </summary>
    /// <param name="input">Image; non-zero is foreground.</param>
    /// <param name="sampling">Pixel spacing per axis (one value broadcasts); null for unit spacing.</param>
    public static NdArray DistanceTransformEdt(NdArray input, double[]? sampling = null)
        => DistanceTransformEdt(input, sampling, returnDistances: true, returnIndices: false).Distances!;

    /// <summary>
    /// Exact Euclidean distance transform with optional feature transform. Uses the separable
    /// Voronoi algorithm of Maurer, Qi and Raghavan (2003) exactly as scipy's
    /// <c>NI_EuclideanFeatureTransform</c> does — one pass per axis in axis order with the same
    /// tie-breaking — so the returned indices, not just the distances, agree with scipy.
    /// </summary>
    /// <remarks>
    /// An image with no background has no nearest zero; like scipy, the result is then computed from
    /// the placeholder feature (-1, 0, …), which is meaningless but deterministic.
    /// </remarks>
    /// <param name="input">Image; non-zero is foreground.</param>
    /// <param name="sampling">Pixel spacing per axis; null for unit spacing.</param>
    /// <param name="returnDistances">Compute the distance map.</param>
    /// <param name="returnIndices">Return the coordinates of each element's nearest background element.</param>
    public static NdImageDistanceResult DistanceTransformEdt(NdArray input, double[]? sampling, bool returnDistances, bool returnIndices)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (!returnDistances && !returnIndices) throw new ArgumentException("At least one of returnDistances/returnIndices must be true.");
        var shape = ShapeOf(input);
        var rank = shape.Length;
        var samp = sampling is null ? null : Expand(sampling, rank, 1.0, nameof(sampling));
        var size = SizeOf(shape);
        var strides = StridesOf(shape);
        var fg = ToMask(input);
        // ft[a * size + x] = axis-a coordinate of x's nearest background element
        var ft = new int[rank * size];
        for (var x = 0; x < size; x++)
        {
            if (fg[x]) { ft[x] = -1; continue; }
            var rem = x;
            for (var a = rank - 1; a >= 0; a--) { ft[a * size + x] = rem % shape[a]; rem /= shape[a]; }
        }
        for (var d = 0; d < rank; d++) VoronoiPass(ft, shape, strides, size, d, samp);

        NdArray? dist = null, idx = null;
        if (returnDistances)
        {
            var dt = new double[size];
            Run(size, size, (lo, hi) =>
            {
                var coord = new int[rank];
                for (var x = lo; x < hi; x++)
                {
                    var rem = x;
                    var acc = 0.0;
                    for (var a = rank - 1; a >= 0; a--) { coord[a] = rem % shape[a]; rem /= shape[a]; }
                    for (var a = 0; a < rank; a++)
                    {
                        double t = ft[a * size + x] - coord[a];
                        if (samp is not null) t *= samp[a];
                        acc += t * t;
                    }
                    dt[x] = Math.Sqrt(acc);
                }
            });
            dist = new NdArray(dt, shape);
        }
        if (returnIndices) idx = new NdArray(ft.Select(v => (double)v).ToArray(), [rank, .. shape]);
        return new NdImageDistanceResult(dist, idx);
    }

    /// <summary>One axis of the feature transform: every line along <paramref name="d"/> gets its lower envelope of parabolas.</summary>
    private static void VoronoiPass(int[] ft, int[] shape, int[] strides, int size, int d, double[]? samp)
    {
        var rank = shape.Length;
        var len = shape[d];
        var stride = strides[d];
        var lines = size / len;
        Run(lines, (long)size * rank, (lo, hi) =>
        {
            var f = new int[len * rank];
            var g = new int[len];
            var coor = new int[rank];
            for (var line = lo; line < hi; line++)
            {
                // line base: enumerate all coordinates except axis d
                var rem = line;
                var b = 0;
                for (var a = rank - 1; a >= 0; a--)
                {
                    if (a == d) { coor[a] = 0; continue; }
                    coor[a] = rem % shape[a];
                    rem /= shape[a];
                    b += coor[a] * strides[a];
                }
                VoronoiLine(ft, size, b, stride, len, rank, d, coor, f, g, samp);
            }
        });
    }

    /// <summary>scipy's <c>_VoronoiFT</c>, transcribed: builds the stack of candidate sites, then sweeps it.</summary>
    private static void VoronoiLine(int[] ft, int size, int b, int stride, int len, int rank, int d, int[] coor, int[] f, int[] g, double[]? samp)
    {
        for (var i = 0; i < len; i++)
            for (var j = 0; j < rank; j++)
                f[i * rank + j] = ft[j * size + b + i * stride];
        var l = -1;
        for (var i = 0; i < len; i++)
        {
            if (f[i * rank] < 0) continue;
            double fd = f[i * rank + d];
            var wR = 0.0;
            for (var j = 0; j < rank; j++)
            {
                if (j == d) continue;
                double tw = f[i * rank + j] - coor[j];
                if (samp is not null) tw *= samp[j];
                wR += tw * tw;
            }
            while (l >= 1)
            {
                var idx1 = g[l];
                double f1 = f[idx1 * rank + d];
                var idx2 = g[l - 1];
                var a = f1 - f[idx2 * rank + d];
                var bb = fd - f1;
                if (samp is not null) { a *= samp[d]; bb *= samp[d]; }
                var c = a + bb;
                double uR = 0.0, vR = 0.0;
                for (var j = 0; j < rank; j++)
                {
                    if (j == d) continue;
                    double cc = coor[j];
                    var tu = f[idx2 * rank + j] - cc;
                    var tv = f[idx1 * rank + j] - cc;
                    if (samp is not null) { tu *= samp[j]; tv *= samp[j]; }
                    uR += tu * tu;
                    vR += tv * tv;
                }
                if (c * vR - bb * uR - a * wR - a * bb * c <= 0.0) break;
                --l;
            }
            ++l;
            g[l] = i;
        }
        var maxl = l;
        if (maxl < 0) return;
        l = 0;
        for (var i = 0; i < len; i++)
        {
            var delta1 = Dist2(f, g[l], i, rank, d, coor, samp);
            while (l < maxl)
            {
                var delta2 = Dist2(f, g[l + 1], i, rank, d, coor, samp);
                if (delta1 <= delta2) break;
                delta1 = delta2;
                ++l;
            }
            var src = g[l];
            for (var j = 0; j < rank; j++) ft[j * size + b + i * stride] = f[src * rank + j];
        }
    }

    private static double Dist2(int[] f, int site, int i, int rank, int d, int[] coor, double[]? samp)
    {
        var s = 0.0;
        for (var j = 0; j < rank; j++)
        {
            double t = j == d ? f[site * rank + j] - i : f[site * rank + j] - coor[j];
            if (samp is not null) t *= samp[j];
            s += t * t;
        }
        return s;
    }

    /// <summary>
    /// Chamfer (city-block or chessboard) distance transform (scipy.ndimage.distance_transform_cdt):
    /// a forward raster pass with the half of the 3×…×3 metric before the centre, then the same pass
    /// over the reversed array. Elements unreachable from any background stay −1, as in scipy.
    /// </summary>
    /// <param name="input">Image; non-zero is foreground.</param>
    /// <param name="metric">Taxicab or Chessboard.</param>
    public static NdArray DistanceTransformCdt(NdArray input, NdImageDistanceMetric metric = NdImageDistanceMetric.Chessboard)
        => DistanceTransformCdt(input, metric, returnDistances: true, returnIndices: false).Distances!;

    /// <summary>Chamfer distance transform with optional feature transform.</summary>
    /// <param name="input">Image; non-zero is foreground.</param>
    /// <param name="metric">Taxicab or Chessboard.</param>
    /// <param name="returnDistances">Compute the distance map.</param>
    /// <param name="returnIndices">Return the coordinates of each element's nearest background element.</param>
    public static NdImageDistanceResult DistanceTransformCdt(NdArray input, NdImageDistanceMetric metric, bool returnDistances, bool returnIndices)
    {
        ArgumentNullException.ThrowIfNull(input);
        var rank = input.Rank;
        var structure = metric switch
        {
            NdImageDistanceMetric.Taxicab => GenerateBinaryStructure(rank, 1),
            NdImageDistanceMetric.Chessboard => GenerateBinaryStructure(rank, rank),
            _ => throw new ArgumentException("The chamfer transform supports Taxicab and Chessboard metrics."),
        };
        return DistanceTransformCdt(input, structure, returnDistances, returnIndices);
    }

    /// <summary>Chamfer distance transform with a custom 3×…×3 metric structure.</summary>
    /// <param name="input">Image; non-zero is foreground.</param>
    /// <param name="metric">Structure of extent 3 on every axis; its non-zero entries are unit steps.</param>
    /// <param name="returnDistances">Compute the distance map.</param>
    /// <param name="returnIndices">Return the coordinates of each element's nearest background element.</param>
    public static NdImageDistanceResult DistanceTransformCdt(NdArray input, NdArray metric, bool returnDistances = true, bool returnIndices = false)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(metric);
        if (!returnDistances && !returnIndices) throw new ArgumentException("At least one of returnDistances/returnIndices must be true.");
        var shape = ShapeOf(input);
        var rank = shape.Length;
        if (metric.Rank != rank || metric.Shape.ToArray().Any(s => s != 3)) throw new ArgumentException("Metric structure must be of size 3 in every direction and match the input rank.");
        var size = SizeOf(shape);
        var fg = ToMask(input);
        var dt = new int[size];
        for (var x = 0; x < size; x++) dt[x] = fg[x] ? -1 : 0;
        var ftIdx = new int[size];
        for (var x = 0; x < size; x++) ftIdx[x] = x;
        var sm = ToMask(metric);
        var half = sm.Length / 2;
        var deltas = new List<int[]>();
        for (var i = 0; i < half; i++)
        {
            if (!sm[i]) continue;
            var dl = new int[rank];
            var rem = i;
            for (var a = rank - 1; a >= 0; a--) { dl[a] = rem % 3 - 1; rem /= 3; }
            deltas.Add(dl);
        }
        ChamferPass(dt, ftIdx, shape, deltas, forward: true);
        ChamferPass(dt, ftIdx, shape, deltas, forward: false);
        NdArray? dist = returnDistances ? new NdArray(dt.Select(v => (double)v).ToArray(), shape) : null;
        NdArray? idx = null;
        if (returnIndices)
        {
            var r = new double[rank * size];
            for (var x = 0; x < size; x++)
            {
                var rem = ftIdx[x];
                for (var a = rank - 1; a >= 0; a--) { r[a * size + x] = rem % shape[a]; rem /= shape[a]; }
            }
            idx = new NdArray(r, [rank, .. shape]);
        }
        return new NdImageDistanceResult(dist, idx);
    }

    /// <summary>
    /// scipy's <c>NI_DistanceTransformOnePass</c>. The backward pass runs over the reversed array,
    /// which in original coordinates means raster order backwards with negated offsets — the
    /// offsets keep their order, which is what decides ties.
    /// </summary>
    private static void ChamferPass(int[] dt, int[] ft, int[] shape, List<int[]> deltas, bool forward)
    {
        var rank = shape.Length;
        var size = dt.Length;
        var strides = StridesOf(shape);
        var sign = forward ? 1 : -1;
        var coord = new int[rank];
        for (var step = 0; step < size; step++)
        {
            var x = forward ? step : size - 1 - step;
            var value = dt[x];
            if (value == 0) continue;
            var rem = x;
            for (var a = rank - 1; a >= 0; a--) { coord[a] = rem % shape[a]; rem /= shape[a]; }
            var min = value;
            var minIdx = x;
            foreach (var dl in deltas)
            {
                var y = 0;
                var inside = true;
                for (var a = 0; a < rank; a++)
                {
                    var v = coord[a] + sign * dl[a];
                    if (v < 0 || v >= shape[a]) { inside = false; break; }
                    y += v * strides[a];
                }
                if (!inside) continue;
                var tt = dt[y];
                if (tt >= 0 && (min < 0 || tt + 1 < min))
                {
                    min = tt + 1;
                    minIdx = y;
                }
            }
            dt[x] = min;
            ft[x] = ft[minIdx];
        }
    }

    /// <summary>
    /// Brute-force distance transform (scipy.ndimage.distance_transform_bf): every foreground
    /// element is compared with every background element that touches the foreground. O(n·m);
    /// use it as a reference. <paramref name="sampling"/> applies to the Euclidean metric only, as
    /// in scipy.
    /// </summary>
    /// <param name="input">Image; non-zero is foreground.</param>
    /// <param name="metric">Distance metric.</param>
    /// <param name="sampling">Pixel spacing per axis (Euclidean only).</param>
    public static NdArray DistanceTransformBf(NdArray input, NdImageDistanceMetric metric = NdImageDistanceMetric.Euclidean, double[]? sampling = null)
        => DistanceTransformBf(input, metric, sampling, returnDistances: true, returnIndices: false).Distances!;

    /// <summary>Brute-force distance transform with optional feature transform; ties go to the background element latest in raster order, as in scipy.</summary>
    /// <param name="input">Image; non-zero is foreground.</param>
    /// <param name="metric">Distance metric.</param>
    /// <param name="sampling">Pixel spacing per axis (Euclidean only).</param>
    /// <param name="returnDistances">Compute the distance map.</param>
    /// <param name="returnIndices">Return the coordinates of each element's nearest background element.</param>
    public static NdImageDistanceResult DistanceTransformBf(NdArray input, NdImageDistanceMetric metric, double[]? sampling, bool returnDistances,
        bool returnIndices)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (!returnDistances && !returnIndices) throw new ArgumentException("At least one of returnDistances/returnIndices must be true.");
        var shape = ShapeOf(input);
        var rank = shape.Length;
        var samp = metric == NdImageDistanceMetric.Euclidean && sampling is not null ? Expand(sampling, rank, 1.0, nameof(sampling)) : null;
        var size = SizeOf(shape);
        var fg = ToMask(input);
        var dil = DilateMask(fg, shape, GenerateBinaryStructure(rank, rank), 1, null, false, null, false);
        // background elements adjacent to the foreground, latest raster position first (scipy's list order)
        var border = new List<int[]>();
        var borderFlat = new List<int>();
        for (var x = size - 1; x >= 0; x--)
        {
            if (fg[x] || !dil[x]) continue;
            border.Add(Coordinates(x, shape));
            borderFlat.Add(x);
        }
        var dist = new double[size];
        var feat = new int[size];
        var borderArr = border.ToArray();
        var flatArr = borderFlat.ToArray();
        Run(size, (long)size * Math.Max(1, borderArr.Length), (lo, hi) =>
        {
            for (var x = lo; x < hi; x++)
            {
                if (!fg[x]) { dist[x] = 0; feat[x] = x; continue; }
                var c = Coordinates(x, shape);
                var best = metric == NdImageDistanceMetric.Euclidean ? double.MaxValue : double.PositiveInfinity;
                var bi = 0;
                for (var k = 0; k < borderArr.Length; k++)
                {
                    var p = borderArr[k];
                    double dd = 0;
                    for (var a = 0; a < rank; a++)
                    {
                        double t = Math.Abs(p[a] - c[a]);
                        switch (metric)
                        {
                            case NdImageDistanceMetric.Euclidean:
                                if (samp is not null) t *= samp[a];
                                dd += t * t;
                                break;
                            case NdImageDistanceMetric.Taxicab:
                                dd += t;
                                break;
                            default:
                                if (t > dd) dd = t;
                                break;
                        }
                    }
                    if (dd < best) { best = dd; bi = flatArr[k]; }
                }
                dist[x] = metric == NdImageDistanceMetric.Euclidean ? Math.Sqrt(best) : best;
                feat[x] = bi;
            }
        });
        NdArray? idx = null;
        if (returnIndices)
        {
            var r = new double[rank * size];
            for (var x = 0; x < size; x++)
            {
                var c = Coordinates(feat[x], shape);
                for (var a = 0; a < rank; a++) r[a * size + x] = c[a];
            }
            idx = new NdArray(r, [rank, .. shape]);
        }
        return new NdImageDistanceResult(returnDistances ? new NdArray(dist, shape) : null, idx);
    }

    private static int[] Coordinates(int x, int[] shape)
    {
        var c = new int[shape.Length];
        for (var a = shape.Length - 1; a >= 0; a--) { c[a] = x % shape[a]; x /= shape[a]; }
        return c;
    }
}
