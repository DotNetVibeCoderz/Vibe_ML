using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.InterpolateAssert;

namespace Gravicode.Science.Tests.SciNet;

// Expected values: scipy 1.17.1.
public class PolynomialInterpolatorTests
{
    private static readonly double[] Xi = [0.0, 0.5, 1.3, 2.0, 2.7, 3.5, 4.0];
    private static readonly double[] Yi = [0.0, 0.629425538604203, 1.353558185417193, 1.5092974268256816, 1.2373798802338298, 0.6992167723103802, 0.44319750469207175];
    private static readonly double[] Xe = [-0.5, 0.2, 1.0, 2.2, 3.9, 4.5];

    [Fact]
    public void Barycentric_And_Krogh_MatchScipy()
    {
        var b = new BarycentricInterpolator(Xi, Yi);
        AllClose([-0.6268275604262656, 0.2585372078773789, 1.1415731509084812, 1.4685588778191812, 0.48257838070762366, 0.3572322017190009], b.Evaluate(Xe), 1e-12, "bary");
        AllClose([1.1680676318081469, 1.2800842588129477, 0.8400737853596612, -0.2882457784088666, -0.42767971402634763, 0.015418840992280754], b.Derivative(Xe, 1), 1e-10, "bary d1");
        AllClose([-0.8547880912310091, -1.0051066716265442, -0.5351455543674455, 0.5831704370134022, 0.5671929340188411, -0.5349602632650183], b.Derivative(Xe, 3), 1e-8, "bary d3");
        var k = new KroghInterpolator(Xi, Yi);
        AllClose([-0.626827560426266, 0.2585372078773789, 1.1415731509084812, 1.4685588778191812, 0.4825783807076243, 0.3572322017190005], k.Evaluate(Xe), 1e-12, "krogh");
        AllClose([1.4685588778191812, -0.28824577840886717, -0.8094385307296696, 0.583170437013404, 0.8163455781026177, -0.5149122636521094, -0.8056789311521166], k.Derivatives(2.2), 1e-9, "krogh derivatives");
        var kh = new KroghInterpolator([0, 0, 1, 1], [1, 2, 3, -1]);
        AllClose([1.125, 1.496, 3.0, -12.024000000000006, -123.527, -202.625], kh.Evaluate(Xe), 1e-12, "hermite krogh");
        Close(2, kh.Derivative(0.0), 1e-12, "f'(0)");
        Close(-1, kh.Derivative(1.0), 1e-12, "f'(1)");
        var b2 = new BarycentricInterpolator(Xi[..4], Yi[..4]);
        b2.AddXi(Xi[4..], Yi[4..]);
        AllClose(b.Evaluate(Xe), b2.Evaluate(Xe), 1e-12, "add_xi");
    }

    [Fact]
    public void Lagrange_Taylor_Pade_MatchScipy()
    {
        AllClose([0.15555555555555559, -1.611111111111111, 3.455555555555555, 1.0], Interpolate.Lagrange([0, 1, 2.5, 4], [1, 3, 2, -1]), 1e-13, "lagrange");
        AllClose([0.5189432678089376, 1.3733577544354079, 2.7103997895881005, 2.718281828459045], Interpolate.ApproximateTaylorPolynomial(Math.Exp, 1.0, 3, 0.5), 1e-9, "taylor");
        var (p, q) = Interpolate.Pade([1, 1, 1.0 / 2, 1.0 / 6, 1.0 / 24, 1.0 / 120], 2);
        AllClose([0.016666666666666663, 0.15000000000000002, 0.6, 1.0], p, 1e-13, "pade p");
        AllClose([0.05, -0.4, 1.0], q, 1e-13, "pade q");
    }

    [Fact]
    public void FloaterHormann_And_AAA_MatchScipy()
    {
        var fh = new FloaterHormannInterpolator(Xi, Yi, d: 2);
        AllClose([1.5384615384615383, -3.333333333333333, 3.767660910518053, -3.9455782312925165, 3.767660910518053, -3.333333333333334, 1.5384615384615388], fh.Weights, 1e-12, "fh weights");
        AllClose([-0.6577290173749748, 0.25964520922780826, 1.1413985707091339, 1.4694870378405913, 0.48413398438955707, 0.3283487209693314], fh.Evaluate(Xe), 1e-12, "fh");
        var xa = new double[50];
        var ya = new double[50];
        for (var i = 0; i < 50; i++) { xa[i] = -1 + 2.0 * i / 49; ya[i] = Math.Tan(1.2 * xa[i]); }
        var r = new AAA(xa, ya);
        Assert.Equal(8, r.SupportPoints.Length);
        AllClose([-2.1758751312648705, -0.3764028516420176, 0.14868128299925149, 1.3242199577554346], r.Evaluate([-0.95, -0.3, 0.123, 0.77]), 1e-10, "aaa values");
        var poles = r.Poles().Where(z => Math.Abs(z.Imaginary) < 1e-8 && Math.Abs(z.Real) < 2).Select(z => z.Real).OrderBy(v => v).ToArray();
        AllClose([-1.3089969391092515, 1.3089969390842655], poles, 1e-7, "real poles near the data (tan(1.2x) has them at +-pi/2.4)");
    }
}
