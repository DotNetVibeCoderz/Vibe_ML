using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.NdImageAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Labelling, region statistics and watershed against scipy 1.17.1 (values in <see cref="NdImageExpectedMeasurements"/>).</summary>
public class NdImageMeasurementTests
{
    private static double[] E(string key) => NdImageExpectedMeasurements.Values[key];

    private static readonly int[] Idx = [1, 2, 3, 5, 99];

    /// <summary>Label image ((5i + 3j) mod 7) on the 7×9 grid; 0 is background.</summary>
    private static NdArray L()
    {
        var d = new double[63];
        for (var i = 0; i < 7; i++)
            for (var j = 0; j < 9; j++)
                d[i * 9 + j] = (i * 5 + j * 3) % 7;
        return new NdArray(d, 7, 9);
    }

    private static double[] FlattenObjects(IReadOnlyList<(int Start, int Stop)[]?> objects)
        => objects.SelectMany(o => o!.SelectMany(s => new double[] { s.Start, s.Stop })).ToArray();

    [Fact]
    public void Label_NumbersComponentsLikeScipy()
    {
        foreach (var (name, img, s) in new[]
                 {
                     ("l4", NdImageMorphologyTests.B2(), (NdArray?)null),
                     ("l8", NdImageMorphologyTests.B2(), NdImage.GenerateBinaryStructure(2, 2)),
                     ("l3c1", NdImageMorphologyTests.B3(), null),
                     ("l3c3", NdImageMorphologyTests.B3(), NdImage.GenerateBinaryStructure(3, 3)),
                 })
        {
            var (labels, count) = NdImage.Label(img, s);
            Assert.Equal((int)E(name + "_n")[0], count);
            Close(E(name), labels, 0, name);
            Close(E(name + "_fo"), FlattenObjects(NdImage.FindObjects(labels)), 0, name + " objects");
        }
    }

    [Fact]
    public void Label_LargeRandomImage_AgreesWithFloodFill()
    {
        var rng = new Random(3);
        int rows = 400, cols = 450; // large enough for the banded parallel scan
        var d = new double[rows * cols];
        for (var i = 0; i < d.Length; i++) d[i] = rng.NextDouble() < 0.55 ? 1 : 0;
        var (labels, count) = NdImage.Label(new NdArray(d, rows, cols), NdImage.GenerateBinaryStructure(2, 2));
        var lab = labels.ToArray();
        // Reference: BFS flood fill numbering components in raster order of their first pixel.
        var refLab = new int[d.Length];
        var next = 0;
        var queue = new Queue<int>();
        for (var start = 0; start < d.Length; start++)
        {
            if (d[start] == 0 || refLab[start] != 0) continue;
            refLab[start] = ++next;
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                var p = queue.Dequeue();
                int pi = p / cols, pj = p % cols;
                for (var di = -1; di <= 1; di++)
                    for (var dj = -1; dj <= 1; dj++)
                    {
                        int qi = pi + di, qj = pj + dj;
                        if (qi < 0 || qi >= rows || qj < 0 || qj >= cols) continue;
                        var q = qi * cols + qj;
                        if (d[q] == 0 || refLab[q] != 0) continue;
                        refLab[q] = next;
                        queue.Enqueue(q);
                    }
            }
        }
        Assert.Equal(next, count);
        for (var i = 0; i < d.Length; i++) Assert.Equal(refLab[i], (int)lab[i]);
    }

    [Fact]
    public void Statistics_MatchScipy()
    {
        var a = A2();
        var l = L();
        Close(E("sum_all"), [NdImage.SumLabels(a)], 1e-14, "sum all");
        Close(E("sum_pos"), [NdImage.SumLabels(a, l)], 1e-14, "sum labels>0");
        Close(E("sum_idx"), NdImage.SumLabels(a, l, Idx), 1e-14, "sum per label");
        Close(E("mean_pos"), [NdImage.Mean(a, l)], 1e-14, "mean");
        Close(E("mean_idx"), NdImage.Mean(a, l, Idx), 1e-14, "mean per label");
        Close(E("var_all"), [NdImage.Variance(a)], 1e-14, "variance");
        Close(E("var_idx"), NdImage.Variance(a, l, Idx), 1e-14, "variance per label");
        Close(E("std_idx"), NdImage.StandardDeviation(a, l, Idx), 1e-14, "std per label");
        Close(E("min_idx"), NdImage.Minimum(a, l, Idx), 0, "min");
        Close(E("max_idx"), NdImage.Maximum(a, l, Idx), 0, "max");
        Close(E("median_idx"), NdImage.Median(a, l, Idx[..4]), 0, "median");
        Close(E("median_all"), [NdImage.Median(a)], 0, "median all");
        Close(E("min_all"), [NdImage.Minimum(a)], 0, "min all");
        Close(E("minpos_idx"), NdImage.MinimumPosition(a, l, Idx).SelectMany(p => p.Select(v => (double)v)).ToArray(), 0, "min positions");
        Close(E("maxpos_idx"), NdImage.MaximumPosition(a, l, Idx).SelectMany(p => p.Select(v => (double)v)).ToArray(), 0, "max positions");
        Close(E("maxpos_all"), NdImage.MaximumPosition(a).Select(v => (double)v).ToArray(), 0, "max position");
        Close(E("minpos_pos"), NdImage.MinimumPosition(a, l).Select(v => (double)v).ToArray(), 0, "min position");
        var shifted = new NdArray(a.ToArray().Select(v => v + 3).ToArray(), 7, 9);
        Close(E("com_all"), NdImage.CenterOfMass(shifted), 1e-14, "centre of mass");
        Close(E("com_idx"), NdImage.CenterOfMass(shifted, l, Idx[..4]).SelectMany(c => c).ToArray(), 1e-14, "centre of mass per label");
        var a3 = A3();
        var mask3 = new NdArray(a3.ToArray().Select(v => v > 0 ? 1.0 : 0.0).ToArray(), 4, 5, 6);
        Close(E("com3"), NdImage.CenterOfMass(new NdArray(a3.ToArray().Select(v => v + 5).ToArray(), 4, 5, 6), mask3), 1e-14, "3-D centre");
        var ext = NdImage.Extrema(a, l, [2, 6]);
        double[] flatExt = [ext[0].Minimum, ext[1].Minimum, ext[0].Maximum, ext[1].Maximum,
            .. ext[0].MinimumPosition, .. ext[1].MinimumPosition, .. ext[0].MaximumPosition, .. ext[1].MaximumPosition];
        Close(E("ext_idx"), flatExt, 0, "extrema");
    }

    [Fact]
    public void HistogramAndComprehension_MatchScipy()
    {
        var a = A2();
        Close(E("hist_all"), NdImage.Histogram(a, -2, 3.5, 7).Select(v => (double)v).ToArray(), 0, "histogram");
        var h = NdImage.Histogram(a, -1, 3, 5, L(), [1, 4]);
        Close(E("hist_idx"), h.SelectMany(x => x!).Select(v => (double)v).ToArray(), 0, "histogram per label");
        Assert.Null(NdImage.Histogram(a, -1, 3, 5, L(), [99])[0]);
        Close(E("comp"), NdImage.LabeledComprehension(a, L(), Idx, v => v.Max() - v.Min(), -7.0), 0, "comprehension");
    }

    [Fact]
    public void ValueIndices_GroupsCoordinates()
    {
        var arr = Arr(new double[,] { { 1, 0, 2 }, { 2, 1, 0 } });
        var vi = NdImage.ValueIndices(arr, ignoreValue: 0);
        // scipy 1.17.1: {1: ([0, 1], [0, 1]), 2: ([0, 1], [2, 0])}
        Assert.Equal(new[] { 1.0, 2.0 }, vi.Keys.ToArray());
        Assert.Equal(new[] { 0, 1 }, vi[1][0]);
        Assert.Equal(new[] { 0, 1 }, vi[1][1]);
        Assert.Equal(new[] { 0, 1 }, vi[2][0]);
        Assert.Equal(new[] { 2, 0 }, vi[2][1]);
    }

    [Fact]
    public void WatershedIft_MatchesScipy()
    {
        var w = new double[99];
        for (var i = 0; i < 9; i++)
            for (var j = 0; j < 11; j++)
                w[i * 11 + j] = Math.Abs(i - 4) * 3 + Math.Abs(j - 5) * 2 + i * j % 4;
        var mk = new double[99];
        mk[1 * 11 + 1] = 1;
        mk[7 * 11 + 9] = 2;
        mk[4 * 11 + 5] = -1;
        mk[0 * 11 + 10] = 3;
        var img = new NdArray(w, 9, 11);
        var markers = new NdArray(mk, 9, 11);
        Close(E("ws"), NdImage.WatershedIft(img, markers), 0, "watershed");
        Close(E("ws8"), NdImage.WatershedIft(img, markers, NdImage.GenerateBinaryStructure(2, 2)), 0, "watershed 8");
        var flat = new double[42];
        for (var i = 0; i < 6; i++) flat[i * 7 + 3] = 5;
        var mk2 = new double[42];
        mk2[2 * 7 + 0] = 1;
        mk2[3 * 7 + 6] = 2;
        mk2[0 * 7 + 3] = -1;
        Close(E("wsflat"), NdImage.WatershedIft(new NdArray(flat, 6, 7), new NdArray(mk2, 6, 7)), 0, "watershed plateau");
    }

    [Fact]
    public void WatershedIft_RowWrapIsScipysAndExactNeighboursIsOptIn()
    {
        // scipy 1.17.1: watershed_ift([[2,1],[1,0]], [[0,0],[-1,1]]) = [[-1,-1],[-1,1]] - element (0,1) is labelled from
        // (1,0), which is only its neighbour because the flat offset -1 wraps onto the previous row.
        var img = Arr(new double[,] { { 2, 1 }, { 1, 0 } });
        var mk = Arr(new double[,] { { 0, 0 }, { -1, 1 } });
        Close([-1, -1, -1, 1], NdImage.WatershedIft(img, mk), 0, "scipy-compatible");
        // With true 4-neighbours (0,1) is reached first from the +1 marker at cost 1 and keeps that label.
        Close([-1, 1, -1, 1], NdImage.WatershedIft(img, mk, exactNeighbours: true), 0, "exact");
    }
}
