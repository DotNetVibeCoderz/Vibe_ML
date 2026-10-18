using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// Mathieu functions against a 40-digit reference built from the definitions in mpmath
/// (MathieuReference.Generated.cs, gen_mathieu.py), not against scipy: scipy's characteristic
/// values are 3-175% wrong for q ≥ 5000, and its ce/se lose six digits at m = 15, q = 1000.
/// </summary>
public class SpecialMathieuTests
{
    public static TheoryData<int> Indices()
    {
        var d = new TheoryData<int>();
        for (var i = 0; i < MathieuReference.Cases.Length; i++) d.Add(i);
        return d;
    }

    [Theory]
    [MemberData(nameof(Indices))]
    public void MatchesHighPrecisionReference(int index)
    {
        var c = MathieuReference.Cases[index];
        (double V, double D) got = c.Function switch
        {
            "a" => (Special.MathieuA(c.M, c.Q), 0.0),
            "b" => (Special.MathieuB(c.M, c.Q), 0.0),
            "ce" => Special.MathieuCem(c.M, c.Q, c.X),
            "se" => Special.MathieuSem(c.M, c.Q, c.X),
            "mc1" => Special.MathieuModCem1(c.M, c.Q, c.X),
            "mc2" => Special.MathieuModCem2(c.M, c.Q, c.X),
            "ms1" => Special.MathieuModSem1(c.M, c.Q, c.X),
            "ms2" => Special.MathieuModSem2(c.M, c.Q, c.X),
            _ => throw new ArgumentException(c.Function),
        };
        // Characteristic values to a few ulps of their size (the matrix is graded, so small values
        // keep relative accuracy too); functions relative to their scale, with an absolute floor for
        // values that vanish by symmetry.
        var (rel, abs) = c.Function switch
        {
            "a" or "b" => (2e-13, 1e-15),
            "ce" or "se" => (2e-11, 1e-13 * Math.Max(1, c.M)),
            _ => (1e-10, 1e-13),
        };
        Check(c, "value", c.Value, got.V, rel, abs);
        if (c.Function is not ("a" or "b")) Check(c, "derivative", c.Derivative, got.D, rel, abs * Math.Max(1, c.M));
    }

    private static void Check(MathieuCase c, string what, double expected, double actual, double rel, double abs)
        => Assert.True(Math.Abs(actual - expected) <= rel * Math.Abs(expected) + abs,
            $"{c.Function}(m={c.M}, q={c.Q}, x={c.X}) {what}: expected {expected:R}, got {actual:R}");

    [Fact]
    public void WronskianOfTheRadialFunctionsIsTwoOverPi()
    {
        // W{Mc1, Mc2} = W{Ms1, Ms2} = 2/π (DLMF 28.20.21): pins both normalisations together.
        foreach (var m in new[] { 0, 1, 2, 4, 7 })
            foreach (var q in new[] { 0.3, 4.0, 25.0 })
                foreach (var z in new[] { 0.2, 0.9, 1.7 })
                {
                    var (c1, dc1) = Special.MathieuModCem1(m, q, z);
                    var (c2, dc2) = Special.MathieuModCem2(m, q, z);
                    Assert.True(Math.Abs(c1 * dc2 - dc1 * c2 - 2 / Math.PI) < 1e-10, $"Mc m={m} q={q} z={z}: {c1 * dc2 - dc1 * c2}");
                    if (m == 0) continue;
                    var (s1, ds1) = Special.MathieuModSem1(m, q, z);
                    var (s2, ds2) = Special.MathieuModSem2(m, q, z);
                    Assert.True(Math.Abs(s1 * ds2 - ds1 * s2 - 2 / Math.PI) < 1e-10, $"Ms m={m} q={q} z={z}: {s1 * ds2 - ds1 * s2}");
                }
    }

    [Fact]
    public void AngularFunctionsSolveMathieusEquationAndAreNormalised()
    {
        foreach (var (m, q) in new[] { (0, 2.0), (3, 7.5), (6, 40.0), (1, -3.0) })
        {
            var a = Special.MathieuA(m, q);
            // y'' + (a - 2q cos 2x) y = 0, y'' by central differences of the exact derivative.
            foreach (var x in new[] { 11.0, 47.0, 101.0 })
            {
                const double h = 1e-4;
                var (y, _) = Special.MathieuCem(m, q, x);
                var dPlus = Special.MathieuCem(m, q, x + h * 180 / Math.PI).Derivative;
                var dMinus = Special.MathieuCem(m, q, x - h * 180 / Math.PI).Derivative;
                var ypp = (dPlus - dMinus) / (2 * h);
                var residual = ypp + (a - 2 * q * Math.Cos(2 * x * Math.PI / 180)) * y;
                Assert.True(Math.Abs(residual) < 1e-6 * (1 + Math.Abs(a)), $"m={m} q={q} x={x}: residual {residual}");
            }
            // ∫₀^{2π} ce² = π.
            var integral = Integrate.Quad(t => Math.Pow(Special.MathieuCem(m, q, t * 180 / Math.PI).Value, 2), 0, 2 * Math.PI).Value;
            Assert.True(Math.Abs(integral - Math.PI) < 1e-10, $"m={m} q={q}: {integral}");
        }
    }

    [Fact]
    public void LimitsSymmetriesAndInvalidArguments()
    {
        for (var m = 0; m < 6; m++) Assert.Equal(m * m, Special.MathieuA(m, 0.0), 12);
        Assert.Equal(1 / Math.Sqrt(2), Special.MathieuCem(0, 0, 30).Value, 15);
        Assert.Equal(Math.Cos(3 * 0.7), Special.MathieuCem(3, 0, 0.7 * 180 / Math.PI).Value, 14);
        Assert.Equal(Math.Sin(2 * 0.7), Special.MathieuSem(2, 0, 0.7 * 180 / Math.PI).Value, 14);
        // a_{2n+1}(−q) = b_{2n+1}(q); a_{2n}(−q) = a_{2n}(q); b_{2n+2}(−q) = b_{2n+2}(q).
        Assert.Equal(Special.MathieuB(3, 4.5), Special.MathieuA(3, -4.5));
        Assert.Equal(Special.MathieuA(4, 4.5), Special.MathieuA(4, -4.5));
        Assert.Equal(Special.MathieuB(4, 4.5), Special.MathieuB(4, -4.5));
        // Period: ce_1 has period 360°, and the derivative is with respect to radians.
        Assert.Equal(Special.MathieuCem(1, 1, 40).Value, Special.MathieuCem(1, 1, 400).Value, 13);
        Assert.True(double.IsNaN(Special.MathieuA(1.5, 1)));
        Assert.True(double.IsNaN(Special.MathieuB(0, 1)));
        Assert.True(double.IsNaN(Special.MathieuCem(-1, 1, 3).Value));
        Assert.True(double.IsNaN(Special.MathieuModCem1(1, -1, 0.5).Value));
    }
}
