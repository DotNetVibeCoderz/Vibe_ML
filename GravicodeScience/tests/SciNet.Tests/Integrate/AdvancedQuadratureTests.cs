using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class AdvancedQuadratureTests
{
    [Fact]
    public void TanhSinh_EndpointSingularities()
    {
        var r = Integrate.TanhSinh(x => Math.Log(x) / Math.Sqrt(x), 0, 1);
        Assert.True(r.Success, $"status {r.Status}");
        IntegrateAssert.Close(-4.0, r.Integral, 1e-12, "log/sqrt");
        Assert.True(Math.Abs(r.Integral + 4) <= Math.Max(r.Error, 1e-15) * 10, "error estimate must cover the true error");
        // Both ends singular. The integrand's own 1 - x^2 cancels catastrophically next to ±1, so
        // neither we nor scipy 1.17.1 (3.1415926318289125) get more than ~8 digits from it.
        var c = Integrate.TanhSinh(x => 1 / Math.Sqrt(1 - x * x), -1, 1);
        IntegrateAssert.Close(Math.PI, c.Integral, 1e-7, "arcsine");
        // At a zero endpoint the abscissae carry full relative precision (x = 1e-300 is representable,
        // 1 - 1e-300 is not), so there the singular integral is resolved to rounding.
        var d = Integrate.TanhSinh(x => Math.Pow(x, -0.9), 0, 1);
        IntegrateAssert.Close(10.0, d.Integral, 1e-11, "x^-0.9");
    }

    [Fact]
    public void TanhSinh_InfiniteAndReversed()
    {
        IntegrateAssert.Close(Math.Sqrt(Math.PI), Integrate.TanhSinh(x => Math.Exp(-x * x), double.NegativeInfinity, double.PositiveInfinity).Integral, 1e-12, "gauss");
        IntegrateAssert.Close(1.0, Integrate.TanhSinh(x => Math.Exp(-x), 0, double.PositiveInfinity).Integral, 1e-12, "exp");
        IntegrateAssert.Close(1.0, Integrate.TanhSinh(Math.Exp, double.NegativeInfinity, 0).Integral, 1e-12, "exp neg");
        IntegrateAssert.Close(-(Math.E - 1), Integrate.TanhSinh(Math.Exp, 1, 0).Integral, 1e-13, "reversed");
        IntegrateAssert.Close(Math.PI / 2, Integrate.TanhSinh(x => 1 / (1 + x * x), 0, double.PositiveInfinity).Integral, 1e-12, "atan");
    }

    [Fact]
    public void Cubature_MatchesScipy()
    {
        // scipy 1.17.1: cubature(exp(-|x|^2) * [1, 2], [0,0], [1,2]) -> [0.65875967, 1.31751934]
        var exact = 0.7468241328124270 * 0.8820813907624215; // erf-based closed form
        var r = Integrate.Cubature(x => { var v = Math.Exp(-(x[0] * x[0] + x[1] * x[1])); return [v, 2 * v]; }, [0, 0], [1, 2]);
        Assert.True(r.Converged);
        IntegrateAssert.Rel(exact, r.Estimate[0], 1e-12, "gk21");
        IntegrateAssert.Rel(2 * exact, r.Estimate[1], 1e-12, "gk21 2");
        var exact3 = 0.06235931799348826; // ∫ cos(x+y+z) over the unit cube
        var g = Integrate.Cubature(x => [Math.Cos(x[0] + x[1] + x[2])], [0, 0, 0], [1, 1, 1], CubatureRule.GenzMalik);
        Assert.True(g.Converged);
        IntegrateAssert.Close(exact3, g.Estimate[0], 1e-9, "genz-malik");
        var k = Integrate.Cubature(x => [Math.Cos(x[0] + x[1] + x[2])], [0, 0, 0], [1, 1, 1], CubatureRule.GaussKronrod15);
        IntegrateAssert.Close(exact3, k.Estimate[0], 1e-13, "gk15");
        var inf = Integrate.Cubature(x => [Math.Exp(-(x[0] * x[0] + x[1] * x[1]))], [double.NegativeInfinity, 0], [double.PositiveInfinity, double.PositiveInfinity], rtol: 1e-10);
        IntegrateAssert.Rel(Math.PI / 2, inf.Estimate[0], 1e-9, "infinite");
    }

    [Fact]
    public void QmcQuad_StandardErrorCoversTruth()
    {
        // ∫ over [0,1]^3 of x y z = 1/8.
        var r = Integrate.QmcQuad(x => x[0] * x[1] * x[2], [0, 0, 0], [1, 1, 1], nPoints: 2048, seed: 7);
        IntegrateAssert.Close(0.125, r.Integral, Math.Max(6 * r.StandardError, 1e-6), "xyz");
        Assert.True(r.StandardError < 1e-3, $"QMC should beat MC: se {r.StandardError}");
        var again = Integrate.QmcQuad(x => x[0] * x[1] * x[2], [0, 0, 0], [1, 1, 1], nPoints: 2048, seed: 7);
        Assert.Equal(again.Integral, r.Integral);
        var g = Integrate.QmcQuad(x => Math.Exp(-x[0] * x[0] - x[1] * x[1]), [-3, -3], [3, 3], seed: 1);
        IntegrateAssert.Close(Math.PI * Math.Pow(0.9999779095030014, 2), g.Integral, Math.Max(6 * g.StandardError, 1e-4), "gauss2d");
    }
}
