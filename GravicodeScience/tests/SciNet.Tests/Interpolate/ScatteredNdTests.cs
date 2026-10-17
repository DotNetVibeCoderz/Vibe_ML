using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.InterpolateAssert;

namespace Gravicode.Science.Tests.SciNet;

public class ScatteredNdTests
{
    /// <summary>The integer LCG of the Spatial tests, reproducible bit-for-bit in Python.</summary>
    private static double[] Lcg(int n, int d, long seed)
    {
        var s = seed;
        var r = new double[n * d];
        for (var i = 0; i < r.Length; i++)
        {
            s = (s * 1103515245 + 12345) % 2147483648;
            r[i] = (s % 10007) / 10007.0;
        }
        return r;
    }

    [Fact]
    public void GriddataLinear3D_MatchesScipy()
    {
        // scipy 1.17.1: pts = lcg_points(60, 3, 21); v = sin(3x) + y z - 2 z²;
        // xi = lcg_points(8, 3, 99) * 0.8 + 0.1; griddata(pts, v, xi, method='linear').
        // Random points have a unique Delaunay triangulation, so the values must agree.
        var p = Lcg(60, 3, 21);
        var v = new double[60];
        for (var i = 0; i < 60; i++) { double x = p[3 * i], y = p[3 * i + 1], z = p[3 * i + 2]; v[i] = Math.Sin(3 * x) + y * z - 2 * z * z; }
        var xi = Lcg(8, 3, 99);
        for (var i = 0; i < xi.Length; i++) xi[i] = xi[i] * 0.8 + 0.1;
        var r = Interpolate.Griddata(new NdArray(p, 60, 3), new NdArray(v, 60), new NdArray(xi, 8, 3), GriddataMethod.Linear).ToArray();
        AllClose([0.12159236379131116, double.NaN, -0.323124361803851, 0.5002765311417467, 0.4379685244272402, 0.2690069699043547, 0.6459326442220578, 0.5301887733644005], r, 1e-12, "griddata linear 3-D");
    }

    [Fact]
    public void Linear4D_ReproducesAffineFunctions_VectorValued()
    {
        var p = Lcg(80, 4, 5);
        var vals = new double[80 * 2];
        for (var i = 0; i < 80; i++)
        {
            double a = p[4 * i], b = p[4 * i + 1], c = p[4 * i + 2], d = p[4 * i + 3];
            vals[2 * i] = 1 + 2 * a - b + 0.5 * c + 3 * d;
            vals[2 * i + 1] = -a + 4 * d;
        }
        var ip = new LinearNDInterpolator(new NdArray(p, 80, 4), new NdArray(vals, 80, 2), fillValue: -7);
        Assert.Equal(4, ip.NDim);
        var q = Lcg(200, 4, 17);
        for (var i = 0; i < q.Length; i++) q[i] = 1.2 * q[i] - 0.1;
        var r = ip.Evaluate(new NdArray(q, 200, 4)).ToArray();
        var inside = ip.Triangulation.FindSimplex(new NdArray(q, 200, 4));
        var hits = 0;
        for (var i = 0; i < 200; i++)
        {
            if (inside[i] < 0) { Assert.Equal(-7.0, r[2 * i]); Assert.Equal(-7.0, r[2 * i + 1]); continue; }
            hits++;
            double a = q[4 * i], b = q[4 * i + 1], c = q[4 * i + 2], d = q[4 * i + 3];
            Assert.True(Math.Abs(r[2 * i] - (1 + 2 * a - b + 0.5 * c + 3 * d)) < 1e-12, $"component 0 at {i}");
            Assert.True(Math.Abs(r[2 * i + 1] - (-a + 4 * d)) < 1e-12, $"component 1 at {i}");
        }
        Assert.True(hits > 20 && hits < 200, $"{hits} queries inside the hull");
    }

    [Fact]
    public void CloughTocher_Rejects3D()
    {
        var p = Lcg(20, 3, 4);
        Assert.Throws<NotSupportedException>(() => new CloughTocher2DInterpolator(new NdArray(p, 20, 3), new NdArray(new double[20], 20)));
        Assert.Throws<ArgumentException>(() => Interpolate.Griddata(new NdArray(p, 20, 3), new NdArray(new double[20], 20), new NdArray(new double[3], 1, 3), GriddataMethod.Cubic));
    }
}
