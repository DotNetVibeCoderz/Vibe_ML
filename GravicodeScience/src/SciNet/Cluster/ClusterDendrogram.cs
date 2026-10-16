using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Truncation of <see cref="Cluster.Dendrogram"/>.</summary>
public enum DendrogramTruncateMode
{
    /// <summary>No truncation.</summary>
    None,
    /// <summary>Show only the last p merged clusters as leaves.</summary>
    LastP,
    /// <summary>Show no more than p levels below the root.</summary>
    Level,
}

/// <summary>Child ordering of <see cref="Cluster.Dendrogram"/>.</summary>
public enum DendrogramSort
{
    /// <summary>Keep the order of Z.</summary>
    None,
    /// <summary>Smaller (by count or distance) child first.</summary>
    Ascending,
    /// <summary>Larger child first.</summary>
    Descending,
}

/// <summary>
/// The layout scipy's <c>dendrogram</c> returns: one U-shaped link per merge with x positions
/// <see cref="Icoord"/> (leaves sit at 5, 15, 25, ...) and heights <see cref="Dcoord"/>.
/// </summary>
/// <param name="Icoord">Four x coordinates per link.</param>
/// <param name="Dcoord">Four heights per link.</param>
/// <param name="Ivl">Leaf labels left to right ("(count)" for a truncated cluster).</param>
/// <param name="Leaves">Node id of each leaf left to right (cluster ids ≥ n when truncated).</param>
/// <param name="ColorList">Matplotlib colour name of each link.</param>
/// <param name="LeavesColorList">Colour of the link above each leaf (null when not determinable, as scipy).</param>
public sealed record DendrogramResult(
    List<double[]> Icoord, List<double[]> Dcoord, List<string> Ivl, List<int> Leaves,
    List<string> ColorList, List<string?> LeavesColorList);

public static partial class Cluster
{
    private static readonly string[] LinkLineColors = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"];

    /// <summary>
    /// Dendrogram layout of a hierarchy, identical to scipy's <c>dendrogram(..., no_plot=True)</c>
    /// dictionary (nothing is drawn; feed the coordinates to ScottPlot or similar).
    /// </summary>
    /// <param name="z">Linkage matrix.</param>
    /// <param name="p">Truncation parameter.</param>
    /// <param name="truncateMode">Truncation mode.</param>
    /// <param name="colorThreshold">Links below this height are coloured per cluster; null means
    /// 0.7 × the maximum height (scipy's default); 0 or less colours everything above-threshold.</param>
    /// <param name="labels">One label per observation.</param>
    /// <param name="countSort">Order children by count.</param>
    /// <param name="distanceSort">Order children by height (ignored when <paramref name="countSort"/> is set).</param>
    /// <param name="showLeafCounts">Label truncated clusters "(count)".</param>
    /// <param name="leafLabelFunc">Custom leaf label from node id.</param>
    /// <param name="linkColorFunc">Custom link colour from node id.</param>
    /// <param name="aboveThresholdColor">Colour of links at or above the threshold.</param>
    public static DendrogramResult Dendrogram(NdArray z, int p = 30, DendrogramTruncateMode truncateMode = DendrogramTruncateMode.None,
        double? colorThreshold = null, IReadOnlyList<string>? labels = null, DendrogramSort countSort = DendrogramSort.None,
        DendrogramSort distanceSort = DendrogramSort.None, bool showLeafCounts = true, Func<int, string>? leafLabelFunc = null,
        Func<int, string>? linkColorFunc = null, string aboveThresholdColor = "C0")
    {
        var a = ZBuffer(z, out var n);
        if (labels is not null && labels.Count != n) throw new ArgumentException("Dimensions of Z and labels must be consistent.");
        double pp = p;
        if (truncateMode == DendrogramTruncateMode.LastP && (p > n || p == 0)) pp = n;
        if (truncateMode == DendrogramTruncateMode.Level && p <= 0) pp = double.PositiveInfinity;
        var maxH = 0.0;
        for (var i = 0; i < n - 1; i++) maxH = Math.Max(maxH, a[4 * i + 2]);
        var threshold = colorThreshold ?? 0.7 * maxH;

        var st = new DendroState
        {
            Z = a, N = n, P = pp, Mode = truncateMode, Threshold = threshold, Labels = labels,
            CountSort = countSort, DistanceSort = distanceSort, ShowLeafCounts = showLeafCounts,
            LeafLabel = leafLabelFunc, LinkColor = linkColorFunc, Above = aboveThresholdColor,
        };
        Calc(st, 2 * n - 2, 0.0, 0);

        var leavesColor = new List<string?>(new string?[st.Leaves.Count]);
        for (var k = 0; k < st.Icoord.Count; k++)
            for (var e = 0; e < 4; e++)
            {
                double xi = st.Icoord[k][e], yi = st.Dcoord[k][e];
                if (yi == 0.0 && PyMod(xi, 5) == 0 && PyMod(xi, 2) == 1)
                {
                    var leaf = ((int)xi - 5) / 10;
                    leavesColor[leaf] = st.Colors[k];
                }
            }
        return new DendrogramResult(st.Icoord, st.Dcoord, st.Ivl, st.Leaves, st.Colors, leavesColor);
    }

    private static double PyMod(double x, double m)
    {
        var r = x % m;
        if (r != 0 && (r < 0) != (m < 0)) r += m;
        return r;
    }

    private sealed class DendroState
    {
        public double[] Z = [];
        public int N;
        public double P;
        public DendrogramTruncateMode Mode;
        public double Threshold;
        public IReadOnlyList<string>? Labels;
        public DendrogramSort CountSort, DistanceSort;
        public bool ShowLeafCounts;
        public Func<int, string>? LeafLabel, LinkColor;
        public string Above = "C0";
        public readonly List<double[]> Icoord = [], Dcoord = [];
        public readonly List<string> Ivl = [], Colors = [];
        public readonly List<int> Leaves = [];
        public int CurrentColor;
        public bool BelowThreshold;
    }

    private static void AppendSingleton(DendroState st, int i)
    {
        st.Leaves.Add(i);
        st.Ivl.Add(st.LeafLabel?.Invoke(i) ?? (st.Labels is not null ? st.Labels[i] : i.ToString(System.Globalization.CultureInfo.InvariantCulture)));
    }

    private static void AppendNonSingleton(DendroState st, int i)
    {
        st.Leaves.Add(i);
        if (st.LeafLabel is not null) st.Ivl.Add(st.LeafLabel(i));
        else st.Ivl.Add(st.ShowLeafCounts ? "(" + ((long)st.Z[4 * (i - st.N) + 3]).ToString(System.Globalization.CultureInfo.InvariantCulture) + ")" : "");
    }

    /// <summary>Transcription of scipy's _dendrogram_calculate_info; returns (x of the link top, width, height, max distance).</summary>
    private static (double Iv, double W, double H, double Md) Calc(DendroState st, int i, double iv, int level)
    {
        var n = st.N; var z = st.Z;
        if (st.Mode == DendrogramTruncateMode.LastP)
        {
            if (2 * n - st.P > i && i >= n)
            {
                AppendNonSingleton(st, i);
                return (iv + 5.0, 10.0, 0.0, z[4 * (i - n) + 2]);
            }
            if (i < n) { AppendSingleton(st, i); return (iv + 5.0, 10.0, 0.0, 0.0); }
        }
        else if (st.Mode == DendrogramTruncateMode.Level)
        {
            if (i > n && level > st.P)
            {
                AppendNonSingleton(st, i);
                return (iv + 5.0, 10.0, 0.0, z[4 * (i - n) + 2]);
            }
            if (i < n) { AppendSingleton(st, i); return (iv + 5.0, 10.0, 0.0, 0.0); }
        }
        if (i < n) { AppendSingleton(st, i); return (iv + 5.0, 10.0, 0.0, 0.0); }

        int aa = (int)z[4 * (i - n)], ab = (int)z[4 * (i - n) + 1];
        double na = aa >= n ? z[4 * (aa - n) + 3] : 1, da = aa >= n ? z[4 * (aa - n) + 2] : 0.0;
        double nb = ab >= n ? z[4 * (ab - n) + 3] : 1, db = ab >= n ? z[4 * (ab - n) + 2] : 0.0;
        int ua = aa, ub = ab;
        if (st.CountSort == DendrogramSort.Ascending) { if (na > nb) { ua = ab; ub = aa; } }
        else if (st.CountSort == DendrogramSort.Descending) { if (na > nb) { ua = aa; ub = ab; } else { ua = ab; ub = aa; } }
        else if (st.DistanceSort == DendrogramSort.Ascending) { if (da > db) { ua = ab; ub = aa; } }
        else if (st.DistanceSort == DendrogramSort.Descending) { if (da > db) { ua = aa; ub = ab; } else { ua = ab; ub = aa; } }

        var (uiva, uwa, uah, uamd) = Calc(st, ua, iv, level + 1);
        var h = z[4 * (i - n) + 2];
        string c;
        if (h >= st.Threshold || st.Threshold <= 0)
        {
            c = st.Above;
            if (st.BelowThreshold) st.CurrentColor = (st.CurrentColor + 1) % LinkLineColors.Length;
            st.BelowThreshold = false;
        }
        else
        {
            st.BelowThreshold = true;
            c = LinkLineColors[st.CurrentColor];
        }
        var (uivb, uwb, ubh, ubmd) = Calc(st, ub, iv + uwa, level + 1);
        var maxDist = Math.Max(Math.Max(uamd, ubmd), h);
        st.Icoord.Add([uiva, uiva, uivb, uivb]);
        st.Dcoord.Add([uah, h, h, ubh]);
        st.Colors.Add(st.LinkColor?.Invoke(i) ?? c);
        return ((uiva + uivb) / 2, uwa + uwb, h, maxDist);
    }
}
