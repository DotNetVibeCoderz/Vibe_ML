using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.NdImageSupport;

namespace Gravicode.Science.SciNet;

public static partial class NdImage
{
    // ------------------------------------------------------------------ labelling

    /// <summary>
    /// Labels connected components of the non-zero elements (scipy.ndimage.label). Components are
    /// numbered 1..Count in the raster order of their first element, which is scipy's numbering.
    /// </summary>
    /// <remarks>
    /// Union-find over the half of the structure that precedes the centre in raster order (links to
    /// the smaller root, path halving), then one relabelling sweep. The root of every component is
    /// its first element in raster order, so numbering roots as they are met reproduces scipy.
    /// </remarks>
    /// <param name="input">Image; non-zero is foreground.</param>
    /// <param name="structure">
    /// Connectivity: a centrosymmetric 3×…×3 element (default: connectivity 1, i.e. 4-connected in 2-D;
    /// use <c>GenerateBinaryStructure(2, 2)</c> for 8-connectivity).
    /// </param>
    /// <returns>The label image and the number of components.</returns>
    public static (NdArray Labels, int Count) Label(NdArray input, NdArray? structure = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var shape = ShapeOf(input);
        var rank = shape.Length;
        structure ??= GenerateBinaryStructure(rank, 1);
        if (structure.Rank != rank) throw new ArgumentException("Structure and input must have equal rank.");
        if (structure.Shape.ToArray().Any(s => s != 3)) throw new ArgumentException("Structure dimensions must be equal to 3.");
        var sm = ToMask(structure);
        for (var i = 0; i < sm.Length; i++)
            if (sm[i] != sm[sm.Length - 1 - i]) throw new ArgumentException("Structure must be centrosymmetric.");
        var size = SizeOf(shape);
        var fg = ToMask(input);
        var strides = StridesOf(shape);
        var deltas = new List<int[]>();
        for (var i = 0; i < sm.Length / 2; i++)
        {
            if (!sm[i]) continue;
            var dl = new int[rank];
            var rem = i;
            for (var a = rank - 1; a >= 0; a--) { dl[a] = rem % 3 - 1; rem /= 3; }
            deltas.Add(dl);
        }
        var dlin = deltas.Select(dl => dl.Select((v, a) => v * strides[a]).Sum()).ToArray();
        var parent = new int[size];
        if (rank == 2) Label2d(fg, parent, shape[0], shape[1], deltas);
        else LabelNd(fg, parent, shape, deltas, dlin);
        // A root is the first element of its component in raster order, so it is numbered before
        // any element that points at it and the label array can double as the root -> number map.
        var labels = new double[size];
        var count = 0;
        for (var x = 0; x < size; x++)
        {
            if (!fg[x]) continue;
            var r = parent[x] == x ? x : Find(parent, x);
            labels[x] = r == x ? ++count : labels[r];
        }
        return (new NdArray(labels, shape), count);
    }

    /// <summary>
    /// 2-D scan in horizontal bands labelled in parallel, then stitched along the band seams.
    /// Within a band the first foreground neighbour simply adopts its root and later ones are
    /// unioned only when their root differs. Every union links the larger root under the smaller,
    /// so whatever the band order, each component's root ends up as its first element in raster
    /// order — the property the numbering relies on.
    /// </summary>
    private static void Label2d(bool[] fg, int[] parent, int rows, int cols, List<int[]> deltas)
    {
        var dr = deltas.Select(d => d[0]).ToArray();
        var dc = deltas.Select(d => d[1]).ToArray();
        var dl = deltas.Select(d => d[0] * cols + d[1]).ToArray();
        var bands = (long)rows * cols < ParallelThreshold * 4 ? 1 : Math.Min(rows, Environment.ProcessorCount * 2);
        var bandStart = new int[bands + 1];
        for (var b = 0; b <= bands; b++) bandStart[b] = (int)((long)rows * b / bands);

        void Scan(int r0, int r1)
        {
            var nd = dl.Length;
            for (var i = r0; i < r1; i++)
            {
                var rowBase = i * cols;
                for (var j = 0; j < cols; j++)
                {
                    var x = rowBase + j;
                    if (!fg[x]) continue;
                    var root = -1;
                    for (var k = 0; k < nd; k++)
                    {
                        var ni = i + dr[k];
                        var nj = j + dc[k];
                        if (ni < r0 || nj < 0 || nj >= cols) continue;
                        var y = x + dl[k];
                        if (!fg[y]) continue;
                        var ry = Find(parent, y);
                        if (root < 0) root = ry;
                        else if (ry != root)
                        {
                            if (ry < root) { parent[root] = ry; root = ry; }
                            else parent[ry] = root;
                        }
                    }
                    parent[x] = root < 0 ? x : root;
                }
            }
        }

        if (bands == 1) { Scan(0, rows); return; }
        Parallel.For(0, bands, b => Scan(bandStart[b], bandStart[b + 1]));
        for (var b = 1; b < bands; b++)
        {
            var i = bandStart[b];
            for (var j = 0; j < cols; j++)
            {
                var x = i * cols + j;
                if (!fg[x]) continue;
                for (var k = 0; k < dl.Length; k++)
                {
                    if (dr[k] >= 0) continue; // only links reaching into the band above
                    var nj = j + dc[k];
                    if (nj < 0 || nj >= cols) continue;
                    var y = x + dl[k];
                    if (fg[y]) Union(parent, x, y);
                }
            }
        }
    }

    private static void LabelNd(bool[] fg, int[] parent, int[] shape, List<int[]> deltas, int[] dlin)
    {
        var rank = shape.Length;
        var size = fg.Length;
        var coord = new int[rank];
        for (var x = 0; x < size; x++)
        {
            if (fg[x])
            {
                parent[x] = x;
                var interior = true;
                for (var a = 0; a < rank; a++)
                    if (coord[a] == 0 || coord[a] == shape[a] - 1) { interior = false; break; }
                for (var k = 0; k < dlin.Length; k++)
                {
                    if (!interior)
                    {
                        var ok = true;
                        var dl = deltas[k];
                        for (var a = 0; a < rank; a++)
                        {
                            var v = coord[a] + dl[a];
                            if (v < 0 || v >= shape[a]) { ok = false; break; }
                        }
                        if (!ok) continue;
                    }
                    var y = x + dlin[k];
                    if (fg[y]) Union(parent, x, y);
                }
            }
            for (var a = rank - 1; a >= 0; a--)
            {
                if (++coord[a] < shape[a]) break;
                coord[a] = 0;
            }
        }
    }

    private static int Find(int[] parent, int x)
    {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    private static void Union(int[] parent, int a, int b)
    {
        var ra = Find(parent, a);
        var rb = Find(parent, b);
        if (ra == rb) return;
        if (ra < rb) parent[rb] = ra;
        else parent[ra] = rb;
    }

    /// <summary>
    /// Bounding boxes of labelled objects (scipy.ndimage.find_objects): entry <c>i</c> describes label
    /// <c>i + 1</c> as per-axis half-open <c>(Start, Stop)</c> ranges, or null if that label is absent.
    /// </summary>
    /// <param name="labels">Integer label image (values ≤ 0 are background).</param>
    /// <param name="maxLabel">Number of labels to report; 0 means the largest label present.</param>
    public static IReadOnlyList<(int Start, int Stop)[]?> FindObjects(NdArray labels, int maxLabel = 0)
    {
        ArgumentNullException.ThrowIfNull(labels);
        var shape = ShapeOf(labels);
        var rank = shape.Length;
        var d = labels.ToArray();
        if (maxLabel < 1)
        {
            maxLabel = 0;
            foreach (var v in d) if (v > maxLabel) maxLabel = (int)v;
        }
        var lo = new int[maxLabel][];
        var hi = new int[maxLabel][];
        var coord = new int[rank];
        for (var x = 0; x < d.Length; x++)
        {
            var l = (int)d[x];
            if (l >= 1 && l <= maxLabel)
            {
                if (lo[l - 1] is null)
                {
                    lo[l - 1] = (int[])coord.Clone();
                    hi[l - 1] = coord.Select(c => c + 1).ToArray();
                }
                else
                {
                    for (var a = 0; a < rank; a++)
                    {
                        if (coord[a] < lo[l - 1][a]) lo[l - 1][a] = coord[a];
                        if (coord[a] + 1 > hi[l - 1][a]) hi[l - 1][a] = coord[a] + 1;
                    }
                }
            }
            for (var a = rank - 1; a >= 0; a--)
            {
                if (++coord[a] < shape[a]) break;
                coord[a] = 0;
            }
        }
        var result = new (int, int)[]?[maxLabel];
        for (var i = 0; i < maxLabel; i++)
            if (lo[i] is not null)
                result[i] = Enumerable.Range(0, rank).Select(a => (lo[i][a], hi[i][a])).ToArray();
        return result;
    }

    // ------------------------------------------------------------------ grouping

    /// <summary>
    /// Element indices of each requested group, in raster order. With no labels the single group is
    /// everything; with labels but no index it is every element whose label is &gt; 0; otherwise one
    /// group per index value (label equal to it) — scipy's <c>_stats</c> conventions.
    /// </summary>
    private static List<int>[] Groups(NdArray input, NdArray? labels, IReadOnlyList<double>? index, out double[] values, out int[] shape)
    {
        ArgumentNullException.ThrowIfNull(input);
        values = input.ToArray();
        shape = ShapeOf(input);
        if (labels is null)
        {
            if (index is not null) throw new ArgumentException("An index needs labels.");
            return [Enumerable.Range(0, values.Length).ToList()];
        }
        if (!ShapeOf(labels).SequenceEqual(shape)) throw new ArgumentException("Input and labels must have the same shape.");
        var l = labels.ToArray();
        if (index is null)
        {
            var g = new List<int>();
            for (var i = 0; i < l.Length; i++) if (l[i] > 0) g.Add(i);
            return [g];
        }
        var groups = new List<int>[index.Count];
        var lookup = new Dictionary<double, List<int>>();
        for (var k = 0; k < index.Count; k++)
        {
            if (!lookup.TryGetValue(index[k], out var list)) lookup[index[k]] = list = new List<int>();
            groups[k] = list;
        }
        for (var i = 0; i < l.Length; i++)
            if (lookup.TryGetValue(l[i], out var list)) list.Add(i);
        return groups;
    }

    private static IReadOnlyList<double> AsIndex(int[] index) => index.Select(i => (double)i).ToArray();

    private static double GroupSum(double[] v, List<int> g, bool pairwise)
    {
        if (pairwise)
        {
            var tmp = new double[g.Count];
            for (var i = 0; i < tmp.Length; i++) tmp[i] = v[g[i]];
            return PairwiseSum(tmp);
        }
        var s = 0.0;
        foreach (var i in g) s += v[i];
        return s;
    }

    private static double GroupVariance(double[] v, List<int> g, bool pairwise)
    {
        if (g.Count == 0) return double.NaN;
        var mean = GroupSum(v, g, pairwise) / g.Count;
        var tmp = new double[g.Count];
        for (var i = 0; i < tmp.Length; i++) { var c = v[g[i]] - mean; tmp[i] = c * c; }
        if (pairwise) return PairwiseSum(tmp) / g.Count;
        var s = 0.0;
        foreach (var t in tmp) s += t;
        return s / g.Count;
    }

    // ------------------------------------------------------------------ sums and moments

    /// <summary>Sum of the input (scipy.ndimage.sum_labels): of everything, or of elements whose label is &gt; 0.</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Optional label image of the same shape.</param>
    public static double SumLabels(NdArray input, NdArray? labels = null)
    {
        var g = Groups(input, labels, null, out var v, out _);
        return GroupSum(v, g[0], pairwise: true);
    }

    /// <summary>Sum of the input per label in <paramref name="index"/>; 0 for absent labels.</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Label image of the same shape.</param>
    /// <param name="index">Labels to report.</param>
    public static double[] SumLabels(NdArray input, NdArray labels, int[] index)
    {
        var g = Groups(input, labels, AsIndex(index), out var v, out _);
        return g.Select(x => GroupSum(v, x, false)).ToArray();
    }

    /// <summary>Mean of the input (scipy.ndimage.mean), of everything or of labels &gt; 0.</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Optional label image.</param>
    public static double Mean(NdArray input, NdArray? labels = null)
    {
        var g = Groups(input, labels, null, out var v, out _);
        return GroupSum(v, g[0], true) / g[0].Count;
    }

    /// <summary>Mean per label; NaN for absent labels.</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Label image.</param>
    /// <param name="index">Labels to report.</param>
    public static double[] Mean(NdArray input, NdArray labels, int[] index)
    {
        var g = Groups(input, labels, AsIndex(index), out var v, out _);
        return g.Select(x => GroupSum(v, x, false) / x.Count).ToArray();
    }

    /// <summary>Population variance (ddof 0) via centred sums (scipy.ndimage.variance).</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Optional label image.</param>
    public static double Variance(NdArray input, NdArray? labels = null)
    {
        var g = Groups(input, labels, null, out var v, out _);
        return GroupVariance(v, g[0], true);
    }

    /// <summary>Population variance per label; NaN for absent labels.</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Label image.</param>
    /// <param name="index">Labels to report.</param>
    public static double[] Variance(NdArray input, NdArray labels, int[] index)
    {
        var g = Groups(input, labels, AsIndex(index), out var v, out _);
        return g.Select(x => GroupVariance(v, x, false)).ToArray();
    }

    /// <summary>Population standard deviation (scipy.ndimage.standard_deviation).</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Optional label image.</param>
    public static double StandardDeviation(NdArray input, NdArray? labels = null) => Math.Sqrt(Variance(input, labels));

    /// <summary>Population standard deviation per label; NaN for absent labels.</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Label image.</param>
    /// <param name="index">Labels to report.</param>
    public static double[] StandardDeviation(NdArray input, NdArray labels, int[] index)
        => Variance(input, labels, index).Select(Math.Sqrt).ToArray();

    // ------------------------------------------------------------------ order statistics

    /// <summary>Minimum (scipy.ndimage.minimum), of everything or of labels &gt; 0.</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Optional label image.</param>
    public static double Minimum(NdArray input, NdArray? labels = null) => Extremum(input, labels, null, true)[0].Value;

    /// <summary>Minimum per label; 0 for absent labels, as scipy.</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Label image.</param>
    /// <param name="index">Labels to report.</param>
    public static double[] Minimum(NdArray input, NdArray labels, int[] index)
        => Extremum(input, labels, AsIndex(index), true).Select(e => e.Value).ToArray();

    /// <summary>Maximum (scipy.ndimage.maximum), of everything or of labels &gt; 0.</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Optional label image.</param>
    public static double Maximum(NdArray input, NdArray? labels = null) => Extremum(input, labels, null, false)[0].Value;

    /// <summary>Maximum per label; 0 for absent labels, as scipy.</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Label image.</param>
    /// <param name="index">Labels to report.</param>
    public static double[] Maximum(NdArray input, NdArray labels, int[] index)
        => Extremum(input, labels, AsIndex(index), false).Select(e => e.Value).ToArray();

    /// <summary>Coordinates of the first minimum in raster order (scipy.ndimage.minimum_position).</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Optional label image.</param>
    public static int[] MinimumPosition(NdArray input, NdArray? labels = null) => Extremum(input, labels, null, true)[0].Position;

    /// <summary>Coordinates of the first minimum per label; all zeros for absent labels, as scipy.</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Label image.</param>
    /// <param name="index">Labels to report.</param>
    public static int[][] MinimumPosition(NdArray input, NdArray labels, int[] index)
        => Extremum(input, labels, AsIndex(index), true).Select(e => e.Position).ToArray();

    /// <summary>Coordinates of the first maximum in raster order (scipy.ndimage.maximum_position).</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Optional label image.</param>
    public static int[] MaximumPosition(NdArray input, NdArray? labels = null) => Extremum(input, labels, null, false)[0].Position;

    /// <summary>Coordinates of the first maximum per label; all zeros for absent labels.</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Label image.</param>
    /// <param name="index">Labels to report.</param>
    public static int[][] MaximumPosition(NdArray input, NdArray labels, int[] index)
        => Extremum(input, labels, AsIndex(index), false).Select(e => e.Position).ToArray();

    /// <summary>Minimum, maximum and their positions (scipy.ndimage.extrema).</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Optional label image.</param>
    public static NdImageExtrema Extrema(NdArray input, NdArray? labels = null)
    {
        var mn = Extremum(input, labels, null, true)[0];
        var mx = Extremum(input, labels, null, false)[0];
        return new NdImageExtrema(mn.Value, mx.Value, mn.Position, mx.Position);
    }

    /// <summary>Extrema per label.</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Label image.</param>
    /// <param name="index">Labels to report.</param>
    public static NdImageExtrema[] Extrema(NdArray input, NdArray labels, int[] index)
    {
        var mn = Extremum(input, labels, AsIndex(index), true);
        var mx = Extremum(input, labels, AsIndex(index), false);
        return mn.Select((e, k) => new NdImageExtrema(e.Value, mx[k].Value, e.Position, mx[k].Position)).ToArray();
    }

    private static (double Value, int[] Position)[] Extremum(NdArray input, NdArray? labels, IReadOnlyList<double>? index, bool minimum)
    {
        var groups = Groups(input, labels, index, out var v, out var shape);
        var result = new (double, int[])[groups.Length];
        for (var k = 0; k < groups.Length; k++)
        {
            var g = groups[k];
            if (g.Count == 0) { result[k] = (0.0, new int[shape.Length]); continue; }
            var best = g[0];
            foreach (var i in g)
                if (minimum ? v[i] < v[best] : v[i] > v[best]) best = i;
            result[k] = (v[best], Coordinates(best, shape));
        }
        return result;
    }

    /// <summary>Median (scipy.ndimage.median), the mean of the two middle values for even counts.</summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Optional label image.</param>
    public static double Median(NdArray input, NdArray? labels = null)
    {
        var g = Groups(input, labels, null, out var v, out _);
        return GroupMedian(v, g[0]);
    }

    /// <summary>
    /// Median per label. Absent labels give NaN — scipy 1.17 returns an arbitrary leftover value
    /// there, which is not worth reproducing.
    /// </summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Label image.</param>
    /// <param name="index">Labels to report.</param>
    public static double[] Median(NdArray input, NdArray labels, int[] index)
    {
        var g = Groups(input, labels, AsIndex(index), out var v, out _);
        return g.Select(x => GroupMedian(v, x)).ToArray();
    }

    private static double GroupMedian(double[] v, List<int> g)
    {
        if (g.Count == 0) return double.NaN;
        var tmp = g.Select(i => v[i]).ToArray();
        Array.Sort(tmp);
        var n = tmp.Length;
        return (n & 1) == 1 ? tmp[n / 2] : (tmp[n / 2 - 1] + tmp[n / 2]) / 2.0;
    }

    /// <summary>Intensity-weighted centroid (scipy.ndimage.center_of_mass), one coordinate per axis.</summary>
    /// <param name="input">Values (weights).</param>
    /// <param name="labels">Optional label image.</param>
    public static double[] CenterOfMass(NdArray input, NdArray? labels = null)
    {
        var g = Groups(input, labels, null, out var v, out var shape);
        return Centroid(v, g[0], shape, true);
    }

    /// <summary>Centroid per label; NaN for absent labels.</summary>
    /// <param name="input">Values (weights).</param>
    /// <param name="labels">Label image.</param>
    /// <param name="index">Labels to report.</param>
    public static double[][] CenterOfMass(NdArray input, NdArray labels, int[] index)
    {
        var g = Groups(input, labels, AsIndex(index), out var v, out var shape);
        return g.Select(x => Centroid(v, x, shape, false)).ToArray();
    }

    private static double[] Centroid(double[] v, List<int> g, int[] shape, bool pairwise)
    {
        var rank = shape.Length;
        var norm = GroupSum(v, g, pairwise);
        var result = new double[rank];
        var tmp = new double[g.Count];
        for (var a = 0; a < rank; a++)
        {
            var stride = 1;
            for (var b = a + 1; b < rank; b++) stride *= shape[b];
            for (var i = 0; i < g.Count; i++) tmp[i] = v[g[i]] * (g[i] / stride % shape[a]);
            double s;
            if (pairwise) s = PairwiseSum(tmp);
            else { s = 0; foreach (var t in tmp) s += t; }
            result[a] = s / norm;
        }
        return result;
    }

    // ------------------------------------------------------------------ histogram / comprehension

    /// <summary>
    /// Histogram of the values (scipy.ndimage.histogram, i.e. numpy.histogram with a range): equal bins
    /// over <c>[min, max]</c>, the last bin closed, values outside dropped.
    /// </summary>
    /// <param name="input">Values.</param>
    /// <param name="min">Lower edge of the first bin.</param>
    /// <param name="max">Upper edge of the last bin.</param>
    /// <param name="bins">Number of bins.</param>
    /// <param name="labels">Optional label image (elements with label &gt; 0 are counted).</param>
    public static long[] Histogram(NdArray input, double min, double max, int bins, NdArray? labels = null)
    {
        var g = Groups(input, labels, null, out var v, out _);
        return Hist(v, g[0], min, max, bins);
    }

    /// <summary>Histogram per label; null for absent labels, as scipy.</summary>
    /// <param name="input">Values.</param>
    /// <param name="min">Lower edge of the first bin.</param>
    /// <param name="max">Upper edge of the last bin.</param>
    /// <param name="bins">Number of bins.</param>
    /// <param name="labels">Label image.</param>
    /// <param name="index">Labels to report.</param>
    public static long[]?[] Histogram(NdArray input, double min, double max, int bins, NdArray labels, int[] index)
    {
        var g = Groups(input, labels, AsIndex(index), out var v, out _);
        return g.Select(x => x.Count == 0 ? null : Hist(v, x, min, max, bins)).ToArray();
    }

    private static long[] Hist(double[] v, List<int> g, double min, double max, int bins)
    {
        if (bins < 1) throw new ArgumentException("bins must be positive.");
        if (!(max > min)) throw new ArgumentException("max must be greater than min.");
        var counts = new long[bins];
        var norm = bins / (max - min);
        var edges = new double[bins + 1];
        for (var i = 0; i <= bins; i++) edges[i] = min + (max - min) * i / bins;
        edges[bins] = max;
        foreach (var i in g)
        {
            var x = v[i];
            if (!(x >= min && x <= max)) continue;
            var k = (int)((x - min) * norm);
            if (k == bins) k--;
            // numpy's correction for the rounding of (x - min) * norm against the linspace edges
            if (x < edges[k]) k--;
            else if (x >= edges[k + 1] && k != bins - 1) k++;
            counts[k]++;
        }
        return counts;
    }

    /// <summary>
    /// Applies <paramref name="func"/> to the values of each labelled group, given in raster order
    /// (scipy.ndimage.labeled_comprehension); absent labels give <paramref name="defaultValue"/>.
    /// </summary>
    /// <param name="input">Values.</param>
    /// <param name="labels">Label image.</param>
    /// <param name="index">Labels to report.</param>
    /// <param name="func">Reduction over one group's values.</param>
    /// <param name="defaultValue">Result for labels with no elements.</param>
    public static double[] LabeledComprehension(NdArray input, NdArray labels, int[] index, Func<double[], double> func, double defaultValue)
    {
        ArgumentNullException.ThrowIfNull(func);
        var g = Groups(input, labels, AsIndex(index), out var v, out _);
        return g.Select(x => x.Count == 0 ? defaultValue : func(x.Select(i => v[i]).ToArray())).ToArray();
    }

    /// <summary>
    /// For every distinct value, the coordinates where it occurs (scipy.ndimage.value_indices):
    /// one index array per axis, in raster order, keyed in ascending value order.
    /// </summary>
    /// <param name="arr">Array of (typically integer) values.</param>
    /// <param name="ignoreValue">A value to leave out (e.g. the background).</param>
    public static IReadOnlyDictionary<double, int[][]> ValueIndices(NdArray arr, double? ignoreValue = null)
    {
        ArgumentNullException.ThrowIfNull(arr);
        var shape = ShapeOf(arr);
        var v = arr.ToArray();
        var lists = new SortedDictionary<double, List<int>>();
        for (var i = 0; i < v.Length; i++)
        {
            if (ignoreValue is not null && v[i] == ignoreValue.Value) continue;
            if (!lists.TryGetValue(v[i], out var l)) lists[v[i]] = l = new List<int>();
            l.Add(i);
        }
        var result = new SortedDictionary<double, int[][]>();
        foreach (var (key, l) in lists)
        {
            var axes = new int[shape.Length][];
            for (var a = 0; a < shape.Length; a++) axes[a] = new int[l.Count];
            for (var k = 0; k < l.Count; k++)
            {
                var c = Coordinates(l[k], shape);
                for (var a = 0; a < shape.Length; a++) axes[a][k] = c[a];
            }
            result[key] = axes;
        }
        return result;
    }

    // ------------------------------------------------------------------ watershed

    /// <summary>
    /// Watershed from markers by the image foresting transform (scipy.ndimage.watershed_ift): each
    /// element takes the label of the marker reachable with the smallest maximum step
    /// <c>|input[p] − input[q]|</c> along the path.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A literal transcription of scipy's <c>NI_WatershedIFT</c> bucket queue — its initial order
    /// (positive markers pushed to the front, negative to the back), its re-queueing (elements
    /// labelled from a negative marker go to the back of their bucket, others to the front) and its
    /// doubly-linked-list bookkeeping — because ties between equally cheap paths are decided by
    /// queue order, and a "cleaner" priority queue labels plateaus differently.
    /// </para>
    /// <para>
    /// scipy finds neighbours by adding a flat offset and only checks that the result lies inside
    /// the array, so in 2-D and above a neighbour step off the end of a row wraps onto the next
    /// row. Measured against scipy 1.17.1 on hundreds of random cases, that wrap is what reproduces
    /// its labels, so it is the default here. Pass <paramref name="exactNeighbours"/> = true for
    /// the geometrically correct neighbourhood (results then differ from scipy near the edges).
    /// </para>
    /// </remarks>
    /// <param name="input">Cost image of integers in [0, 65535] (scipy accepts uint8/uint16 only).</param>
    /// <param name="markers">Integer markers; 0 is unlabelled, negative markers are processed after positive ones.</param>
    /// <param name="structure">Connectivity (default: connectivity 1).</param>
    /// <param name="exactNeighbours">Check neighbour bounds per axis instead of reproducing scipy's row wrap.</param>
    public static NdArray WatershedIft(NdArray input, NdArray markers, NdArray? structure = null, bool exactNeighbours = false)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(markers);
        var shape = ShapeOf(input);
        var rank = shape.Length;
        if (!ShapeOf(markers).SequenceEqual(shape)) throw new ArgumentException("Input and markers must have equal shape.");
        structure ??= GenerateBinaryStructure(rank, 1);
        if (structure.Rank != rank || structure.Shape.ToArray().Any(s => s != 3))
            throw new ArgumentException("Structure dimensions must be equal to 3 and match the input rank.");
        var size = SizeOf(shape);
        var inv = input.ToArray();
        var val = new int[size];
        var maxval = 0;
        for (var i = 0; i < size; i++)
        {
            var x = inv[i];
            if (x < 0 || x > 65535 || x != Math.Floor(x)) throw new ArgumentException("Input must hold integers in [0, 65535] (scipy accepts uint8/uint16 only).");
            val[i] = (int)x;
            if (val[i] > maxval) maxval = val[i];
        }
        var mk = markers.ToArray();
        var output = new double[size];
        for (var i = 0; i < size; i++) output[i] = (int)mk[i];

        var cost = new int[size];
        var next = new int[size];
        var prev = new int[size];
        var done = new bool[size];
        var first = new int[maxval + 2];
        var last = new int[maxval + 2];
        Array.Fill(first, -1);
        Array.Fill(last, -1);
        Array.Fill(next, -1);
        Array.Fill(prev, -1);
        for (var ll = 0; ll < size; ll++)
        {
            var label = (int)mk[ll];
            if (label != 0)
            {
                cost[ll] = 0;
                if (first[0] < 0)
                {
                    first[0] = ll;
                    next[ll] = -1;
                    prev[ll] = -1;
                    last[0] = ll;
                }
                else if (label > 0)
                {
                    next[ll] = first[0];
                    prev[ll] = -1;
                    prev[first[0]] = ll;
                    first[0] = ll;
                }
                else
                {
                    next[ll] = -1;
                    prev[ll] = last[0];
                    next[last[0]] = ll;
                    last[0] = ll;
                }
            }
            else cost[ll] = maxval + 1;
        }

        var sm = ToMask(structure);
        var deltas = new List<int[]>();
        for (var k = 0; k < sm.Length; k++)
        {
            if (!sm[k] || k == sm.Length / 2) continue;
            var dl = new int[rank];
            var rem = k;
            for (var a = rank - 1; a >= 0; a--) { dl[a] = rem % 3 - 1; rem /= 3; }
            deltas.Add(dl);
        }
        var strides = StridesOf(shape);
        var flat = deltas.Select(dl => dl.Select((d, a) => d * strides[a]).Sum()).ToArray();
        var coord = new int[rank];
        for (var jj = 0; jj <= maxval; jj++)
        {
            while (first[jj] >= 0)
            {
                var v = first[jj];
                first[jj] = next[v];
                if (first[jj] >= 0) prev[first[jj]] = -1;
                prev[v] = -1;
                next[v] = -1;
                done[v] = true;
                var rem = v;
                for (var a = rank - 1; a >= 0; a--) { coord[a] = rem % shape[a]; rem /= shape[a]; }
                for (var k = 0; k < deltas.Count; k++)
                {
                    int p;
                    if (exactNeighbours)
                    {
                        var dl = deltas[k];
                        p = 0;
                        var outside = false;
                        for (var a = 0; a < rank; a++)
                        {
                            var c = coord[a] + dl[a];
                            if (c < 0 || c >= shape[a]) { outside = true; break; }
                            p += c * strides[a];
                        }
                        if (outside) continue;
                    }
                    else
                    {
                        p = v + flat[k];
                        if (p < 0 || p >= size) continue;
                    }
                    if (done[p]) continue;
                    var wvp = Math.Abs(val[p] - val[v]);
                    var pcost = cost[p];
                    var max = cost[v] > wvp ? cost[v] : wvp;
                    if (max >= pcost) continue;
                    cost[p] = max;
                    output[p] = output[v];
                    if (next[p] >= 0 || prev[p] >= 0)
                    {
                        var pr = prev[p];
                        var nx = next[p];
                        if (first[pcost] == p) first[pcost] = nx;
                        if (last[pcost] == p) last[pcost] = pr;
                        if (pr >= 0) next[pr] = nx;
                        if (nx >= 0) prev[nx] = pr;
                    }
                    if (output[p] < 0)
                    {
                        prev[p] = last[max];
                        next[p] = -1;
                        if (last[max] >= 0) next[last[max]] = p;
                        last[max] = p;
                        if (first[max] < 0) first[max] = p;
                    }
                    else
                    {
                        next[p] = first[max];
                        prev[p] = -1;
                        if (first[max] >= 0) prev[first[max]] = p;
                        first[max] = p;
                        if (last[max] < 0) last[max] = p;
                    }
                }
            }
        }
        return new NdArray(output, shape);
    }
}
