using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class ConstantsTests
{
    internal static void Near(double expected, double actual, double absTol)
        => Assert.True(Math.Abs(actual - expected) <= absTol, $"expected {expected:R}, got {actual:R}");

    [Fact]
    public void SiDefiningConstants_AreExact()
    {
        // The 2019 SI fixes these values; they must be the exact decimal literals.
        Assert.Equal(299792458.0, Constants.C);
        Assert.Equal(299792458.0, Constants.SpeedOfLight);
        Assert.Equal(6.62607015e-34, Constants.H);
        Assert.Equal(6.62607015e-34, Constants.Planck);
        Assert.Equal(1.602176634e-19, Constants.ElementaryCharge);
        Assert.Equal(1.380649e-23, Constants.K);
        Assert.Equal(1.380649e-23, Constants.Boltzmann);
        Assert.Equal(6.02214076e23, Constants.NA);
        Assert.Equal(6.02214076e23, Constants.Avogadro);
        Assert.Equal(9.80665, Constants.StandardGravity);
        Assert.Equal(273.15, Constants.ZeroCelsius);
        Assert.Equal(0.0, Constants.Precision("speed of light in vacuum"));
        Assert.Equal(0.0, Constants.Precision("Planck constant"));
        ConstantsTests.Near(Constants.H / (2 * Math.PI), Constants.Hbar, 1e-49);
        ConstantsTests.Near(Constants.NA * Constants.K, Constants.R, 1e-14);
    }

    [Fact]
    public void Codata2022_ValuesMatchScipy()
    {
        // scipy.constants 1.17.1 (CODATA 2022)
        Assert.Equal(445, Constants.PhysicalConstants.Count);
        Assert.Equal(Constants.PhysicalConstantCount, Constants.PhysicalConstants.Count);
        Assert.Equal(9.1093837139e-31, Constants.Value("electron mass"));
        Assert.Equal("kg", Constants.Unit("electron mass"));
        Assert.Equal(1836.152673426, Constants.Value("proton-electron mass ratio"));
        ConstantsTests.Near(2.2474266964325848e-05, Constants.Precision("Newtonian constant of gravitation"), 1e-20);
        ConstantsTests.Near(1.5073959909534913e-10, Constants.Precision("fine-structure constant"), 1e-24);
        Assert.Equal(10973731.568157, Constants.Rydberg);
        Assert.Equal(1.25663706127e-06, Constants.Mu0);
        Assert.Equal(Constants.ElectronMass, Constants.Value("electron mass"));
        Assert.Equal(Constants.G, Constants.Value("Newtonian constant of gravitation"));
        Assert.Equal(Constants.Alpha, Constants.Value("fine-structure constant"));
        Assert.Equal("", Constants.Unit("fine-structure constant"));
        Assert.Throws<KeyNotFoundException>(() => Constants.Value("no such constant"));
    }

    [Fact]
    public void Find_IsCaseInsensitiveAndSorted()
    {
        // scipy.constants.find('Boltzmann')
        Assert.Equal(new[]
        {
            "Boltzmann constant", "Boltzmann constant in Hz/K", "Boltzmann constant in eV/K",
            "Boltzmann constant in inverse meter per kelvin", "Stefan-Boltzmann constant",
        }, Constants.Find("boltzmann"));
        Assert.Equal(355, Constants.Find().Count); // len(scipy.constants.find())
        Assert.Equal(Constants.CurrentConstantCount, Constants.Find().Count);
        // Obsolete spellings remain readable with their old values, as in SciPy's physical_constants,
        // but Find does not offer them.
        Assert.True(Constants.PhysicalConstants.ContainsKey("Boltzmann constant in inverse meters per kelvin"));
        Assert.DoesNotContain("Boltzmann constant in inverse meters per kelvin", Constants.Find("boltzmann"));
        Assert.Empty(Constants.Find("unobtainium"));
    }

    [Fact]
    public void UnitFactors_AndPrefixes()
    {
        Assert.Equal(1e3, Constants.Kilo);
        Assert.Equal(1e30, Constants.Quetta);
        Assert.Equal(1e-30, Constants.Quecto);
        Assert.Equal(1024.0, Constants.Kibi);
        Assert.Equal(1208925819614629174706176.0, Constants.Yobi); // 2^80
        ConstantsTests.Near(1609.344, Constants.Mile, 1e-9);
        ConstantsTests.Near(0.0254, Constants.Inch, 1e-15);
        ConstantsTests.Near(0.45359237, Constants.Pound, 1e-15);
        Assert.Equal(101325.0, Constants.Atm);
        Assert.Equal(3600.0, Constants.Hour);
        ConstantsTests.Near(Math.PI / 180, Constants.Degree, 1e-17);
        Assert.Equal(1e-3, Constants.Liter);
        Assert.Equal(Constants.Liter, Constants.Litre);
        ConstantsTests.Near(745.6998715822701, Constants.Hp, 1e-9);  // scipy.constants.hp
        Assert.Equal(4.184, Constants.Calorie);
        Assert.Equal(Constants.ElementaryCharge, Constants.EV);
        ConstantsTests.Near(5.0 / 9.0, Constants.DegreeFahrenheit, 1e-15);
        Assert.Equal(Math.PI, Constants.Pi);
        ConstantsTests.Near((1 + Math.Sqrt(5)) / 2, Constants.GoldenRatio, 1e-15);
    }

    [Fact]
    public void ConvertTemperature_AcceptsScipySpellings()
    {
        ConstantsTests.Near(212.0, Constants.ConvertTemperature(100, "Celsius", "Fahrenheit"), 1e-12);
        ConstantsTests.Near(-40.0, Constants.ConvertTemperature(-40, "C", "F"), 1e-12);
        ConstantsTests.Near(540.0, Constants.ConvertTemperature(300, "K", "R"), 1e-12);
        ConstantsTests.Near(373.15, Constants.ConvertTemperature(100, "c", "kelvin"), 1e-12);
        ConstantsTests.Near(0.0, Constants.ConvertTemperature(491.67, "rankine", "celsius"), 1e-12);
        ConstantsTests.Near(-273.15, Constants.ConvertTemperature(0, "k", "C"), 1e-12);
        Assert.Throws<ArgumentException>(() => Constants.ConvertTemperature(1, "Reaumur", "C"));
        foreach (var from in Enum.GetValues<TemperatureScale>())
            foreach (var to in Enum.GetValues<TemperatureScale>())
                foreach (var v in new[] { -100.0, 0.0, 36.6, 1e4 })
                {
                    var back = Constants.ConvertTemperature(Constants.ConvertTemperature(v, from, to), to, from);
                    Assert.True(Math.Abs(back - v) <= 1e-12 * Math.Max(1, Math.Abs(v)), $"{from}->{to}->{from} of {v} gave {back}");
                }
        var arr = Constants.ConvertTemperature([0.0, 100.0], TemperatureScale.Celsius, TemperatureScale.Kelvin);
        Assert.Equal(new[] { 273.15, 373.15 }, arr);
    }

    [Fact]
    public void WavelengthFrequency_Conversions()
    {
        ConstantsTests.Near(599584916000000.0, Constants.Lambda2Nu(500e-9), 1e-1); // scipy.constants.lambda2nu(500e-9)
        Assert.Equal(1.0, Constants.Lambda2Nu(Constants.C));
        ConstantsTests.Near(500e-9, Constants.Nu2Lambda(Constants.Lambda2Nu(500e-9)), 1e-22);
    }
}
