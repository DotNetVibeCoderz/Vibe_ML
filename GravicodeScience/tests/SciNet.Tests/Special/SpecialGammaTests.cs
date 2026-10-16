using System.Numerics;
using Gravicode.Science.SciNet;
using Xunit;
using R = Gravicode.Science.Tests.SciNet.SpecialReference;

namespace Gravicode.Science.Tests.SciNet;

public class SpecialGammaTests
{
    // Reference tables: SciPy 1.17.1, with mpmath 40-digit values where SciPy itself is off (see
    // the comments in SpecialReference.cs). Tolerances are a little above the measured worst case.

    [Fact]
    public void Gamma_MatchesReference() => SpecialAssert.Table(R.Gamma, a => Special.Gamma(a[0]), 3e-15, "Gamma");

    [Fact]
    public void GammaLn_MatchesReference_IncludingNearOneAndTwo()
        => SpecialAssert.Table(R.GammaLn, a => Special.GammaLn(a[0]), 5e-15, "GammaLn");

    [Fact]
    public void Digamma_MatchesReference_IncludingItsPositiveRoot()
        => SpecialAssert.Table(R.Digamma, a => Special.Digamma(a[0]), 2e-15, "Digamma");

    [Fact]
    public void Polygamma_MatchesReference() => SpecialAssert.Table(R.Polygamma, a => Special.Polygamma((int)a[0], a[1]), 2e-15, "Polygamma");

    [Fact]
    public void Beta_MatchesReference() => SpecialAssert.Table(R.Beta, a => Special.Beta(a[0], a[1]), 1e-13, "Beta");

    [Fact]
    public void BetaLn_MatchesReference_ForHugeArguments() => SpecialAssert.Table(R.BetaLn, a => Special.BetaLn(a[0], a[1]), 4e-15, "BetaLn");

    [Fact]
    public void Poch_MatchesReference() => SpecialAssert.Table(R.Poch, a => Special.Poch(a[0], a[1]), 1e-14, "Poch");

    [Fact]
    public void Binom_MatchesReference() => SpecialAssert.Table(R.Binom, a => Special.Binom(a[0], a[1]), 4e-15, "Binom");

    [Fact]
    public void Zeta_MatchesReference_OnBothSidesOfThePole() => SpecialAssert.Table(R.Zeta, a => Special.Zeta(a[0]), 3e-15, "Zeta");

    [Fact]
    public void HurwitzZeta_MatchesReference() => SpecialAssert.Table(R.HurwitzZeta, a => Special.Zeta(a[0], a[1]), 2e-15, "HurwitzZeta");

    [Fact]
    public void Zetac_MatchesReference_ForLargeArguments() => SpecialAssert.Table(R.Zetac, a => Special.Zetac(a[0]), 2e-15, "Zetac");

    [Fact]
    public void Gamma_IsExactAtIntegers()
    {
        var f = BigInteger.One;
        for (var n = 1; n <= 170; n++)
        {
            if (n > 1) f *= n - 1;
            // (double)BigInteger rounds; parse the decimal form for the correctly rounded value.
            var exact = double.Parse(f.ToString(), System.Globalization.CultureInfo.InvariantCulture);
            Assert.Equal(exact, Special.Gamma(n));
        }
    }

    [Fact]
    public void Gamma_SatisfiesReflectionAndRecurrence()
    {
        foreach (var x in new[] { 0.1, 0.37, 0.5, 0.83, 1.7, 3.3 })
        {
            // Γ(x)Γ(1-x) = π / sin(πx)
            SpecialAssert.Close(Math.PI / Math.Sin(Math.PI * x), Special.Gamma(x) * Special.Gamma(1 - x), 4e-15, $"reflection at {x}");
            SpecialAssert.Close(x * Special.Gamma(x), Special.Gamma(x + 1), 4e-15, $"recurrence at {x}");
        }
    }

    [Fact]
    public void Gamma_EdgeValuesFollowScipy()
    {
        Assert.Equal(double.PositiveInfinity, Special.Gamma(0.0));
        Assert.Equal(double.NegativeInfinity, Special.Gamma(-0.0));
        Assert.True(double.IsNaN(Special.Gamma(-3)));
        Assert.Equal(double.PositiveInfinity, Special.Gamma(172));
        Assert.Equal(double.PositiveInfinity, Special.GammaLn(-2));
        Assert.True(double.IsNaN(Special.GammaSgn(-1)));
        Assert.Equal(-1.0, Special.GammaSgn(-0.5));
        Assert.Equal(1.0, Special.GammaSgn(-1.5));
        Assert.Equal(-1.0, Special.GammaSgn(-0.0));
        Assert.Equal(0.0, Special.RGamma(-4));
        Assert.Equal(0.0, Special.RGamma(0));
        Assert.Equal(double.NegativeInfinity, Special.Digamma(0.0));
        Assert.Equal(double.PositiveInfinity, Special.Digamma(-0.0));
        Assert.True(double.IsNaN(Special.Digamma(-2)));
    }

    [Fact]
    public void Digamma_ClosedFormsAndRecurrence()
    {
        SpecialAssert.Close(-0.57721566490153286061, Special.Digamma(1), 1e-16, "ψ(1) = -γ");
        SpecialAssert.Close(-0.57721566490153286061 - 2 * Math.Log(2), Special.Digamma(0.5), 5e-16, "ψ(1/2)");
        foreach (var x in new[] { 0.2, 2.5, 11.0, 40.3 })
            SpecialAssert.Close(Special.Digamma(x) + 1 / x, Special.Digamma(x + 1), 3e-15, $"ψ recurrence at {x}");
        SpecialAssert.Close(Math.PI * Math.PI / 6, Special.Polygamma(1, 1), 2e-16, "ψ'(1) = π²/6");
    }

    [Fact]
    public void RiemannZeta_KnownValues()
    {
        SpecialAssert.Close(Math.PI * Math.PI / 6, Special.Zeta(2), 2e-16, "ζ(2)");
        SpecialAssert.Close(Math.Pow(Math.PI, 4) / 90, Special.Zeta(4), 5e-16, "ζ(4)");
        SpecialAssert.Close(-1.0 / 12, Special.Zeta(-1), 2e-16, "ζ(-1)");
        Assert.Equal(-0.5, Special.Zeta(0));
        Assert.Equal(0.0, Special.Zeta(-4));
        Assert.Equal(double.PositiveInfinity, Special.Zeta(1));
        Assert.True(double.IsNaN(Special.Zeta(0.5, 1.0))); // Hurwitz form is defined for x > 1 only, as in SciPy
    }

    [Fact]
    public void Beta_IsSymmetricAndMatchesGammaRatio()
    {
        foreach (var (a, b) in new[] { (0.3, 2.2), (5.0, 7.0), (1.5, 30.25) })
        {
            Assert.Equal(Special.Beta(a, b), Special.Beta(b, a));
            SpecialAssert.Close(Special.Gamma(a) * Special.Gamma(b) / Special.Gamma(a + b), Special.Beta(a, b), 5e-15, $"B({a},{b})");
        }
        Assert.Equal(double.PositiveInfinity, Special.Beta(-1, 2));
        Assert.Equal(0.0, Special.Beta(-0.5, -0.5));
    }

    [Fact]
    public void Combinatorics_MatchScipy()
    {
        Assert.Equal(10.0, Special.Comb(5, 2));
        Assert.Equal(0.0, Special.Comb(5, 7));
        Assert.Equal(0.0, Special.Comb(-1, 2));
        Assert.Equal(12.375, Special.Comb(5.5, 2));
        Assert.Equal(15.0, Special.Comb(5, 2, repetition: true));
        Assert.Equal(BigInteger.Parse("100891344545564193334812497256"), Special.CombExact(100, 50));
        Assert.Equal(new BigInteger(15), Special.CombExact(5, 2, repetition: true));
        Assert.Equal(20.0, Special.Perm(5, 2));
        Assert.Equal(24.75, Special.Perm(5.5, 2));
        Assert.Equal(0.0, Special.Perm(5, 7));
        Assert.Equal(new BigInteger(6720), Special.PermExact(8, 5));
        Assert.Equal(BigInteger.Parse("15511210043330985984000000"), Special.FactorialExact(25));
        Assert.Equal(0.0, Special.Factorial(-1));
        SpecialAssert.Close(3.3233509704478426, Special.Factorial(2.5), 1e-15, "2.5!"); // scipy.special.factorial(2.5)
        Assert.Equal(105.0, Special.Factorial2(7));
        Assert.Equal(48.0, Special.Factorial2(6));
        Assert.Equal(1.0, Special.Factorial2(0));
        Assert.Equal(0.0, Special.Factorial2(-1));
        Assert.Equal(280.0, Special.FactorialK(10, 3));
        Assert.Equal(new BigInteger(280), Special.FactorialKExact(10, 3));
        // Gamma-form double factorial agrees with the exact product beyond the direct-product range.
        SpecialAssert.Close((double)Special.Factorial2Exact(61), Special.Factorial2(61), 1e-14, "61!!");
        SpecialAssert.Close((double)Special.Factorial2Exact(80), Special.Factorial2(80), 1e-14, "80!!");
    }
}
