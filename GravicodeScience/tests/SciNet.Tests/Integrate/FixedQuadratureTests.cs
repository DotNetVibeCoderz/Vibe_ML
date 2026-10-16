using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class FixedQuadratureTests
{
    private static readonly double[] Y5 = [1, 3, -2, 4, 0.5];
    private static readonly double[] Y6 = [1, 3, -2, 4, 0.5, 2.5];
    private static readonly double[] X5 = [0, 0.3, 1.0, 1.2, 2.0];
    private static readonly double[] X6 = [0, 0.3, 1.0, 1.2, 2.0, 2.7];

    [Fact]
    public void Simpson_MatchesScipy_OddEvenUneven()
    {
        // scipy 1.17.1 simpson(...)
        IntegrateAssert.Close(8.5, Integrate.Simpson(Y5), 1e-14, "odd");
        IntegrateAssert.Close(9.541666666666666, Integrate.Simpson(Y6), 1e-14, "even");
        IntegrateAssert.Close(6.780753968253969, Integrate.Simpson(Y5, X5), 1e-14, "odd uneven");
        IntegrateAssert.Close(7.555128968253969, Integrate.Simpson(Y6, X6), 1e-14, "even uneven");
        IntegrateAssert.Close(4.770833333333333, Integrate.Simpson(Y6, dx: 0.5), 1e-14, "dx");
        IntegrateAssert.Close(0.6, Integrate.Simpson(Y5.AsSpan(0, 2), X5.AsSpan(0, 2)), 1e-15, "two points");
    }

    [Fact]
    public void Simpson_EvenN_IsExactForQuadratics()
    {
        double[] x = [0, 0.4, 0.9, 1.0, 1.7, 2.3];
        var y = x.Select(t => 3 * t * t - t + 2).ToArray();
        IntegrateAssert.Close(2.3 * 2.3 * 2.3 - 2.3 * 2.3 / 2 + 4.6, Integrate.Simpson(y, x), 1e-13, "quadratic");
    }

    [Fact]
    public void Cumulative_MatchScipy()
    {
        double[] e1 = [2.583333333333333, 3.666666666666666, 5.458333333333332, 8.499999999999998, 9.541666666666664];
        Check(e1, Integrate.CumulativeSimpson(Y6), "cumsimpson");
        double[] e2 = [2.0, 2.662142857142857, 3.8015873015873014, 4.047420634920634, 8.780753968253968, 9.555128968253968];
        Check(e2, Integrate.CumulativeSimpson(Y6, X6, initial: 2.0), "cumsimpson x initial");
        double[] e3 = [0.662142857142857, 1.8015873015873014, 2.0474206349206345, 6.780753968253968];
        Check(e3, Integrate.CumulativeSimpson(Y5, X5), "cumsimpson odd");
        double[] e4 = [0.0, 0.6, 0.95, 1.15, 2.95, 4.0];
        Check(e4, Integrate.CumulativeTrapezoid(Y6, X6, initial: 0), "cumtrapz");
        Assert.Throws<ArgumentException>(() => Integrate.CumulativeTrapezoid(Y6, initial: 1.0));
    }

    private static void Check(double[] e, double[] a, string what)
    {
        Assert.Equal(e.Length, a.Length);
        for (var i = 0; i < e.Length; i++) IntegrateAssert.Close(e[i], a[i], 1e-13, $"{what}[{i}]");
    }

    [Fact]
    public void Romb_Romberg_FixedQuad()
    {
        var y = Enumerable.Range(0, 17).Select(i => Math.Exp(i / 16.0)).ToArray();
        IntegrateAssert.Close(1.7182818284590784, Integrate.Romb(y, 1.0 / 16), 1e-15, "romb");
        Assert.Throws<ArgumentException>(() => Integrate.Romb(new double[6]));
        var r = Integrate.Romberg(Math.Exp, 0, 1);
        Assert.True(r.Converged);
        IntegrateAssert.Close(Math.E - 1, r.Value, 1e-12, "romberg");
        IntegrateAssert.Close(0.8414709848078964, Integrate.FixedQuad(Math.Cos, 0, 1, 7), 1e-15, "fixed_quad");
        // High-order Gauss-Legendre is exact for degree 2n-1.
        IntegrateAssert.Close(2.0 / 199, Integrate.FixedQuad(x => Math.Pow(x, 198), -1, 1, 100), 1e-14, "GL100");
    }

    [Fact]
    public void NewtonCotes_MatchesScipyTable()
    {
        var (w4, b4) = Integrate.NewtonCotes(4);
        double[] e4 = [14.0 / 45, 64.0 / 45, 24.0 / 45, 64.0 / 45, 14.0 / 45];
        for (var i = 0; i < 5; i++) IntegrateAssert.Close(e4[i], w4[i], 1e-16, $"w4[{i}]");
        IntegrateAssert.Close(-8.0 / 945, b4, 1e-18, "B4");
        var (w15, b15) = Integrate.NewtonCotes(15);
        // N = 15 is past scipy's exact table, so scipy inverts a Vandermonde matrix (7.605640623150975);
        // the exact rational value (checked with sympy) is 7.605640623153353.
        IntegrateAssert.Close(7.605640623153353, w15[3], 1e-14, "w15[3] exact");
        IntegrateAssert.Close(7.605640623150975, w15[3], 1e-11, "w15[3] scipy");
        // scipy's floating-point B for N = 15 (-0.006429006036388896) loses 6 digits to cancellation;
        // sympy's exact value is -0.006428992862647135.
        IntegrateAssert.Rel(-0.006428992862647135, b15, 1e-14, "B15 exact");
        IntegrateAssert.Rel(-0.006429006036388896, b15, 1e-5, "B15 scipy");
        var (wu, bu) = Integrate.NewtonCotes([0, 1.5, 2, 3]);
        double[] eu = [0.5, 2.0, 0.0, 0.5];
        for (var i = 0; i < 4; i++) IntegrateAssert.Close(eu[i], wu[i], 1e-13, $"wu[{i}]");
        IntegrateAssert.Close(-0.0843750000000001, bu, 1e-13, "Bu");
    }

    [Fact]
    public void NdArray_AlongAxis()
    {
        var y = new NdArray(Enumerable.Range(0, 12).Select(i => (double)i * i).ToArray(), 3, 4);
        var s = Integrate.Simpson(y, axis: 0).ToArray();
        double[] es = [42.66666666666667, 60.66666666666667, 82.66666666666667, 108.66666666666667];
        for (var i = 0; i < 4; i++) IntegrateAssert.Close(es[i], s[i], 1e-12, $"simpson axis0[{i}]");
        var t = Integrate.Trapezoid(y, axis: 1).ToArray();
        double[] et = [9.5, 93.5, 273.5];
        for (var i = 0; i < 3; i++) IntegrateAssert.Close(et[i], t[i], 1e-12, $"trapz axis1[{i}]");
    }
}
