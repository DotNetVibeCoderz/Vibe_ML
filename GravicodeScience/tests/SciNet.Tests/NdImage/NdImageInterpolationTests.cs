using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.NdImageAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Spline interpolation against scipy 1.17.1 for every mode and order (values in <see cref="NdImageExpectedInterpolation"/>).</summary>
public class NdImageInterpolationTests
{
    private static double[] E(string key) => NdImageExpectedInterpolation.Values[key];

    public static TheoryData<NdImageMode> Modes() => NdImageAssert.Modes();

    private static double[] Linspace(double a, double b, int n) => Enumerable.Range(0, n).Select(i => a + (b - a) * i / (n - 1)).ToArray();

    /// <summary>The coordinate grids of the scipy script: meshgrid(linspace, linspace, indexing='ij'), deliberately straying outside.</summary>
    private static NdArray Grid(params double[][] axes)
    {
        var rank = axes.Length;
        var shape = axes.Select(a => a.Length).ToArray();
        var n = shape.Aggregate(1, (p, s) => p * s);
        var d = new double[rank * n];
        var counter = new int[rank];
        for (var p = 0; p < n; p++)
        {
            for (var a = 0; a < rank; a++) d[a * n + p] = axes[a][counter[a]];
            for (var a = rank - 1; a >= 0; a--)
            {
                if (++counter[a] < shape[a]) break;
                counter[a] = 0;
            }
        }
        return new NdArray(d, [rank, .. shape]);
    }

    private static NdArray C2() => Grid(Linspace(-3.3, 10.2, 5), Linspace(-2.6, 12.4, 6));
    private static NdArray C3() => Grid(Linspace(-1.2, 4.7, 3), Linspace(-0.4, 5.9, 4), Linspace(-2.1, 6.8, 3));
    private static NdArray M() => Arr(new double[,] { { 0.9, 0.2 }, { -0.3, 1.1 } });

    [Theory]
    [MemberData(nameof(Modes))]
    public void MapCoordinates_AllOrders_MatchScipy(NdImageMode m)
    {
        var n = Name(m);
        for (var o = 0; o <= 5; o++)
            Close(E($"map_{o}_{n}"), NdImage.MapCoordinates(A2(), C2(), o, m, 1.5), 1e-11, $"order {o}");
        foreach (var o in new[] { 0, 1, 3, 5 })
            Close(E($"map3d_{o}_{n}"), NdImage.MapCoordinates(A3(), C3(), o, m, 1.5), 1e-11, $"3-D order {o}");
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void AffineAndShift_AllOrders_MatchScipy(NdImageMode m)
    {
        var n = Name(m);
        for (var o = 0; o <= 5; o++)
        {
            Close(E($"affine_{o}_{n}"), NdImage.AffineTransform(A2(), M(), [0.5, -1.0], [6, 10], o, m, 1.5), 1e-11, $"affine order {o}");
            Close(E($"shift_{o}_{n}"), NdImage.Shift(A2(), [0.3, -1.7], o, m, 1.5), 1e-11, $"shift order {o}");
        }
        var h = Arr(new double[,] { { 0.9, 0.2, 0.5 }, { -0.3, 1.1, -1.0 }, { 0, 0, 1 } });
        Close(E($"homog_{n}"), NdImage.AffineTransform(A2(), h, outputShape: [6, 10], order: 3, mode: m, cval: 1.5), 1e-11, "homogeneous");
        Close(E($"diag_{n}"), NdImage.AffineTransform(A2(), new NdArray([0.8, 1.3], 2), [0.4, -0.6], order: 3, mode: m, cval: 1.5), 1e-11, "diagonal");
        Close(E($"geo_{n}"), NdImage.GeometricTransform(A2(), x => [x[0] * 0.8 + 0.3, x[1] * 1.1 - 0.5], [5, 8], 3, m, 1.5), 1e-11, "geometric");
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void ZoomAndRotate_MatchScipy(NdImageMode m)
    {
        var n = Name(m);
        foreach (var o in new[] { 0, 1, 3, 5 })
        {
            var z = NdImage.Zoom(A2(), [1.5, 0.7], o, m, 1.5);
            Assert.Equal(new[] { 10, 6 }, z.Shape.ToArray());
            Close(E($"zoom_{o}_{n}"), z, 1e-11, $"zoom order {o}");
            Close(E($"zoomg_{o}_{n}"), NdImage.Zoom(A2(), [1.5, 0.7], o, m, 1.5, gridMode: true), 1e-11, $"grid zoom order {o}");
            var r = NdImage.Rotate(A2(), 30, order: o, mode: m, cval: 1.5);
            Assert.Equal(new[] { 11, 11 }, r.Shape.ToArray());
            Close(E($"rot_{o}_{n}"), r, 1e-11, $"rotate order {o}");
        }
        Close(E($"rot3d_{n}"), NdImage.Rotate(A3(), 25, (0, 2), reshape: false, mode: m, cval: 1.5), 1e-11, "3-D rotate");
        var r3 = NdImage.Rotate(A3(), -70, (2, 1), reshape: true, order: 1, mode: m, cval: 1.5);
        Assert.Equal(new[] { 4, 7, 7 }, r3.Shape.ToArray());
        Close(E($"rot3dr_{n}"), r3, 1e-11, "3-D rotate reshape");
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void SplineFilter_MatchesScipy(NdImageMode m)
    {
        var n = Name(m);
        for (var o = 2; o <= 5; o++)
            Close(E($"spline_{o}_{n}"), NdImage.SplineFilter(A2(), o, m), 1e-12, $"order {o}");
        Close(E($"spline1d_{n}"), NdImage.SplineFilter1d(A3(), 3, 1, m), 1e-12, "1-D along axis 1");
    }

    [Fact]
    public void Rotate_QuarterTurnIsExactAndOddAnglesMatch()
    {
        var r = NdImage.Rotate(A2(), 90);
        Assert.Equal(new[] { 9, 7 }, r.Shape.ToArray());
        Close(E("rot90"), r, 1e-12, "90 degrees");
        Close(E("rot_small"), NdImage.Rotate(A2(), -12.5, order: 1), 1e-11, "-12.5 degrees");
    }

    [Fact]
    public void Interpolation_AtGridPointsReturnsTheSamples()
    {
        // A spline through the samples must pass through them: shifting by zero is the identity for every order.
        foreach (var o in new[] { 0, 1, 2, 3, 4, 5 })
            Close(A2().ToArray(), NdImage.Shift(A2(), [0.0, 0.0], o, NdImageMode.Mirror), 1e-12, $"order {o}");
    }

    [Fact]
    public void LargeZoom_ParallelPathMatchesMapCoordinates()
    {
        var img = NdImageFilterTests.RandomImage(64, 80, 9);
        var z = NdImage.Zoom(img, [2.0, 2.0], 3, NdImageMode.Reflect).ToArray();
        // Same points through the general (non-separable) path.
        var rows = 128; var cols = 160;
        var coords = new double[2 * rows * cols];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
            {
                coords[i * cols + j] = i * (63.0 / 127.0);
                coords[rows * cols + i * cols + j] = j * (79.0 / 159.0);
            }
        var viaMap = NdImage.MapCoordinates(img, new NdArray(coords, 2, rows, cols), 3, NdImageMode.Reflect);
        Close(viaMap.ToArray(), z, 1e-12, "zoom vs map_coordinates");
    }
}
