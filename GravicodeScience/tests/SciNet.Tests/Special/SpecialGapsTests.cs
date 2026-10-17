using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// Degree trigonometry, AGM, Struve, Kelvin, hyperu and Wright's Bessel function. Tables are
/// scipy 1.17.1 values; the identities at the end are closed forms checked independently.
/// </summary>
public class SpecialGapsTests
{
    [Fact]
    public void Sindg_MatchesScipy()
    {
        double[][] rows =
        [
            [30.0, 0.49999999999999994],
            [180.0, -0.0],
            [-45.0, -0.7071067811865476],
            [100030.0, -0.7660444431189781],
            [270.0, -1.0],
            [33.3, 0.5490228179981317],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Sindg(a[0])], 1e-15, "sindg");
    }

    [Fact]
    public void Cosdg_MatchesScipy()
    {
        double[][] rows =
        [
            [60.0, 0.49999999999999994],
            [90.0, -0.0],
            [-120.0, -0.49999999999999994],
            [450.0, -0.0],
            [33.3, 0.8358073613682703],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Cosdg(a[0])], 1e-15, "cosdg");
    }

    [Fact]
    public void Tandg_MatchesScipy()
    {
        double[][] rows =
        [
            [45.0, 1.0],
            [135.0, -1.0],
            [90.0, double.PositiveInfinity],
            [-30.0, -0.5773502691896257],
            [33.3, 0.6568772224012792],
            [180.0, 0.0],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Tandg(a[0])], 1e-15, "tandg");
    }

    [Fact]
    public void Cotdg_MatchesScipy()
    {
        double[][] rows =
        [
            [45.0, 1.0],
            [180.0, double.PositiveInfinity],
            [60.0, 0.5773502691896257],
            [-30.0, -1.7320508075688767],
            [33.3, 1.5223545068961313],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Cotdg(a[0])], 1e-15, "cotdg");
    }

    [Fact]
    public void Radian_MatchesScipy()
    {
        double[][] rows =
        [
            [30.0, 15.0, 10.0, 0.5280105800963957],
            [-1.0, 0.0, 0.0, -0.017453292519943295],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Radian(a[0], a[1], a[2])], 1e-15, "radian");
    }

    [Fact]
    public void Agm_MatchesScipy()
    {
        double[][] rows =
        [
            [1.0, 2.0, 1.4567910310469068],
            [24.0, 6.0, 13.458171481725614],
            [1e-300, 1.0, 0.0022694061941578216],
            [1e+300, 1.0, 2.269406194157821e+297],
            [-3.0, -5.0, -3.9362355036495553],
            [0.5, 0.5, 0.5],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Agm(a[0], a[1])], 1e-14, "agm");
    }

    [Fact]
    public void StruveH_MatchesScipy()
    {
        double[][] rows =
        [
            [0.0, 1.0, 0.5686566270482881],
            [1.0, 2.5, 0.8631542066565352],
            [2.5, 10.0, 3.123463196577328],
            [0.5, 30.0, 0.12320283348011285],
            [-1.5, 3.0, -0.4777182150870922],
            [3.0, 0.1, 6.060800286971544e-07],
            [1.0, -2.0, 0.6467637282835621],
            [10.0, 5.0, 0.0015857834421077023],
            [0.0, 100.0, -0.07087875168964737],
            [-0.7, 4.0, -0.4003297444286919],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.StruveH(a[0], a[1])], 1e-12, "struve");
    }

    [Fact]
    public void StruveL_MatchesScipy()
    {
        double[][] rows =
        [
            [0.0, 1.0, 0.710243185937891],
            [1.0, 2.5, 1.9880307722359414],
            [2.5, 10.0, 2025.477444203101],
            [0.5, 30.0, 778366068840.3009],
            [-1.5, 3.0, 3.0994834567256357],
            [3.0, 0.1, 6.065291431839885e-07],
            [1.0, -2.0, 1.1027597873677157],
            [10.0, 5.0, 0.0032677172565631785],
            [-0.7, 4.0, 10.54383089777456],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.StruveL(a[0], a[1])], 1e-12, "modstruve");
    }

    [Fact]
    public void Ber_MatchesScipy()
    {
        double[][] rows =
        [
            [0.5, 0.9990234639908382],
            [3.0, -0.2213802495986939],
            [9.9, 133.43446026231672],
            [15.0, -2967.2545346357465],
            [-2.0, 0.7517341827138083],
            [50.0, -117623968512356.98],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Ber(a[0])], 1e-12, "ber");
    }

    [Fact]
    public void Bei_MatchesScipy()
    {
        double[][] rows =
        [
            [0.5, 0.06249321838219946],
            [3.0, 1.9375867852660427],
            [9.9, 43.45915293258391],
            [15.0, -2952.7078873448645],
            [50.0, -50192646254461.41],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Bei(a[0])], 1e-12, "bei");
    }

    [Fact]
    public void Ker_MatchesScipy()
    {
        double[][] rows =
        [
            [0.5, 0.8559058721186341],
            [3.0, -0.06702923330379883],
            [9.9, 0.0001627507044787912],
            [15.0, -1.514347207267156e-08],
            [50.0, -2.9150770893968664e-17],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Ker(a[0])], 1e-12, "ker");
    }

    [Fact]
    public void Kei_MatchesScipy()
    {
        double[][] rows =
        [
            [0.5, -0.6715816950943675],
            [3.0, -0.0511218840459868],
            [9.9, -0.0003209832147204271],
            [15.0, 7.962894398377203e-06],
            [0.0, -0.7853981633974483],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Kei(a[0])], 1e-12, "kei");
    }

    [Fact]
    public void Berp_MatchesScipy()
    {
        double[][] rows =
        [
            [0.5, -0.007812076147507734],
            [3.0, -1.5698466322294042],
            [15.0, 91.05533316965173],
            [-2.0, 0.49306712470943914],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Berp(a[0])], 1e-12, "berp");
    }

    [Fact]
    public void Beip_MatchesScipy()
    {
        double[][] rows =
        [
            [0.5, 0.24991862111621022],
            [3.0, 0.8804823240578614],
            [15.0, -4087.755236845389],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Beip(a[0])], 1e-12, "beip");
    }

    [Fact]
    public void Kerp_MatchesScipy()
    {
        double[][] rows =
        [
            [0.5, -1.8197997533173529],
            [3.0, 0.021476186867733515],
            [15.0, 5.644678075956919e-06],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Kerp(a[0])], 1e-12, "kerp");
    }

    [Fact]
    public void Keip_MatchesScipy()
    {
        double[][] rows =
        [
            [0.5, 0.33320379160332986],
            [3.0, 0.09204305048030394],
            [15.0, -5.882222803057011e-06],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Keip(a[0])], 1e-12, "keip");
    }

    [Fact]
    public void Hyperu_MatchesScipy()
    {
        double[][] rows =
        [
            [1.0, 1.0, 1.0, 0.5963473623231941],
            [0.5, 1.5, 2.0, 0.7071067811865475],
            [2.0, 3.0, 0.5, 4.0],
            [-2.0, 1.0, 3.0, -0.9999999999999994],
            [1.5, 0.3, 4.0, 0.07079879754603124],
            [0.2, 1.0, 0.5, 1.098920979405165],
            [3.0, 7.0, 15.0, 0.0005267489711934156],
            [0.7, 0.2, 0.0, 1.3136925546812048],
            [2.0, 2.5, 20.0, 0.0023871257108963513],
            [0.5, -1.5, 1.0, 0.5342020585529919],
            [1.0, 4.0, 6.0, 0.23148148148148145],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Hyperu(a[0], a[1], a[2])], 1e-09, "hyperu");
    }

    [Fact]
    public void WrightBessel_MatchesScipy()
    {
        double[][] rows =
        [
            [0.5, 1.0, 2.0, 6.690627940507143],
            [1.0, 0.5, 10.0, 157.44031078010775],
            [0.0, 1.0, 3.0, 20.085536923187668],
            [2.0, 0.0, 1.5, 1.692229527461609],
            [0.1, 2.0, 100.0, 8.300602728576021e+36],
            [3.0, 1.0, 0.01, 1.0016667361115703],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.WrightBessel(a[0], a[1], a[2])], 1e-13, "wright_bessel");
    }

    [Fact]
    public void LogWrightBessel_MatchesScipy()
    {
        double[][] rows =
        [
            [0.5, 1.0, 2.0, 1.9007077322924444],
            [0.2, 1.0, 500.0, 275.56589673979414],
            [1.0, 2.0, 1000.0, 56.79324781295563],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.LogWrightBessel(a[0], a[1], a[2])], 1e-13, "log_wright_bessel");
    }

    [Fact]
    public void DegreeFunctions_AreExactAtSpecialAngles()
    {
        Assert.Equal(0.0, Special.Sindg(360));
        Assert.Equal(1.0, Special.Sindg(90));
        Assert.Equal(0.0, Special.Cosdg(270), 0.0);
        Assert.Equal(-1.0, Special.Tandg(-45));
        Assert.Equal(double.PositiveInfinity, Special.Tandg(-90)); // scipy 1.17.1: unsigned poles
        Assert.True(double.IsNaN(Special.Agm(-1, 2)));
    }

    [Fact]
    public void Struve_HalfOrderClosedForms()
    {
        // DLMF 11.4.5/11.4.7: H_{1/2}(x) = √(2/(πx))(1 − cos x), L_{1/2}(x) = √(2/(πx))(cosh x − 1).
        foreach (var x in new[] { 0.3, 2.0, 7.5, 25.0 })
        {
            var c = Math.Sqrt(2 / (Math.PI * x));
            SpecialAssert.Close(c * (1 - Math.Cos(x)), Special.StruveH(0.5, x), 1e-12, $"H(1/2, {x})");
            SpecialAssert.Close(c * (Math.Cosh(x) - 1), Special.StruveL(0.5, x), 1e-12, $"L(1/2, {x})");
        }
    }

    [Fact]
    public void Kelvin_MatchesBesselOfRotatedArgument()
    {
        // ber x + i bei x = J0(x e^{3πi/4}) = I0(x e^{πi/4}); check through the power series of I0.
        foreach (var x in new[] { 0.7, 2.0, 5.0 })
        {
            var z = System.Numerics.Complex.FromPolarCoordinates(x, Math.PI / 4);
            var s = System.Numerics.Complex.Zero;
            var t = System.Numerics.Complex.One;
            for (var k = 1; k < 60; k++) { s += t; t *= z * z / 4 / (k * k); }
            SpecialAssert.Close(s.Real, Special.Ber(x), 1e-13, "ber");
            SpecialAssert.Close(s.Imaginary, Special.Bei(x), 1e-13, "bei");
            var (be, ke, _, kep) = Special.Kelvin(-x);
            Assert.Equal(s.Real, be.Real, 1e-13);
            Assert.True(double.IsNaN(ke.Real) && double.IsNaN(kep.Imaginary));
        }
    }

    [Fact]
    public void Hyperu_Identities()
    {
        // U(a, a + 1, x) = x^(−a); U(a, b, x) = x^(1−b) U(a − b + 1, 2 − b, x).
        foreach (var (a, x) in new[] { (0.5, 1.3), (2.0, 3.0), (1.7, 0.4) })
            SpecialAssert.Close(Math.Pow(x, -a), Special.Hyperu(a, a + 1, x), 1e-9, $"U({a},{a + 1},{x})");
        SpecialAssert.Close(Math.Pow(2.5, -0.6) * Special.Hyperu(0.5 - 1.6 + 1, 2 - 1.6, 2.5), Special.Hyperu(0.5, 1.6, 2.5), 1e-9, "Kummer");
        Assert.True(double.IsNaN(Special.Hyperu(1, 1, -1)));
        Assert.Equal(double.PositiveInfinity, Special.Hyperu(1, 2, 0));
    }

    [Fact]
    public void WrightBessel_ReducesToKnownFunctions()
    {
        // a = 0: e^x/Γ(b); a = 1, b = 1: I0(2√x); a = 1, b = ν + 1: x^(−ν/2) I_ν(2√x).
        SpecialAssert.Close(Math.Exp(2.5) / Special.Gamma(1.5), Special.WrightBessel(0, 1.5, 2.5), 1e-14, "a=0");
        SpecialAssert.Close(Special.I0(2 * Math.Sqrt(3.0)), Special.WrightBessel(1, 1, 3), 1e-14, "I0");
        SpecialAssert.Close(Math.Pow(5.0, -0.35) * Special.Iv(0.7, 2 * Math.Sqrt(5.0)), Special.WrightBessel(1, 1.7, 5), 1e-13, "Iv");
        Assert.True(double.IsNaN(Special.WrightBessel(-1, 1, 1)));
    }
}
