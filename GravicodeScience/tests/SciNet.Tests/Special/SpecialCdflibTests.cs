using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// Noncentral CDFs and the cdflib-style inverses. Tables are scipy 1.17.1 values (whose cdflib
/// search stops at ~1e-10 relative, hence the tolerances); the round trips check SciNet's own
/// inverses to near machine precision.
/// </summary>
public class SpecialCdflibTests
{
    [Fact]
    public void Chndtr_MatchesScipy()
    {
        double[][] rows =
        [
            [5.0, 3.0, 2.0, 0.5934051800831556],
            [0.5, 1.0, 0.3, 0.45854655631597047],
            [40.0, 10.0, 20.0, 0.8433561681387522],
            [3.0, 2.5, 0.0, 0.694150370554182],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Chndtr(a[0], a[1], a[2])], 1e-12, "chndtr");
    }

    [Fact]
    public void Chndtrix_MatchesScipy()
    {
        double[][] rows =
        [
            [0.5, 3.0, 2.0, 4.137515123399119],
            [0.01, 4.0, 10.0, 2.346661564001438],
            [0.99, 2.0, 1.0, 12.848597211167942],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Chndtrix(a[0], a[1], a[2])], 1e-08, "chndtrix");
    }

    [Fact]
    public void Chndtridf_MatchesScipy()
    {
        double[][] rows =
        [
            [5.0, 0.3, 2.0, 5.774818064525707],
            [10.0, 0.7, 1.0, 7.366271428535948],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Chndtridf(a[0], a[1], a[2])], 1e-08, "chndtridf");
    }

    [Fact]
    public void Chndtrinc_MatchesScipy()
    {
        double[][] rows =
        [
            [5.0, 3.0, 0.5, 2.8985299934839217],
            [10.0, 2.0, 0.2, 14.891173618006075],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Chndtrinc(a[0], a[1], a[2])], 1e-08, "chndtrinc");
    }

    [Fact]
    public void Nctdtr_MatchesScipy()
    {
        double[][] rows =
        [
            [5.0, 1.0, 2.0, 0.7780746626162148],
            [10.0, -1.5, 0.3, 0.9631696454007748],
            [3.0, 2.0, -1.0, 0.003005247595398459],
            [20.0, 0.5, 4.0, 0.9984285827025396],
            [1.5, 3.0, 3.0, 0.37748434364721795],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Nctdtr(a[0], a[1], a[2])], 1e-11, "nctdtr");
    }

    [Fact]
    public void Nctdtrit_MatchesScipy()
    {
        double[][] rows =
        [
            [5.0, 1.0, 0.5, 1.0528510409473966],
            [10.0, -1.0, 0.05, -3.0417418149697637],
            [8.0, 2.0, 0.9, 3.8760700143020976],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Nctdtrit(a[0], a[1], a[2])], 1e-08, "nctdtrit");
    }

    [Fact]
    public void Nctdtridf_MatchesScipy()
    {
        double[][] rows =
        [
            [0.5, 1.0, 2.0, 0.4373627315353053],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Nctdtridf(a[0], a[1], a[2])], 1e-07, "nctdtridf");
    }

    [Fact]
    public void Ncfdtr_MatchesScipy()
    {
        double[][] rows =
        [
            [3.0, 7.0, 2.0, 1.5, 0.496450453642787],
            [1.0, 20.0, 5.0, 3.0, 0.3058505698228582],
            [10.0, 4.0, 0.5, 0.8, 0.3298193054729719],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Ncfdtr(a[0], a[1], a[2], a[3])], 1e-11, "ncfdtr");
    }

    [Fact]
    public void Ncfdtri_MatchesScipy()
    {
        double[][] rows =
        [
            [3.0, 7.0, 2.0, 0.5, 1.5133165298689033],
            [5.0, 12.0, 1.0, 0.9, 2.863079269600173],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Ncfdtri(a[0], a[1], a[2], a[3])], 1e-08, "ncfdtri");
    }

    [Fact]
    public void NcfInverses_AgreeWithScipysForwardCdf()
    {
        // scipy 1.17.1's ncfdtridfd/ncfdtridfn/ncfdtrinc solve with cdflib's own noncentral-F series,
        // which disagrees with scipy's (Boost) ncfdtr by ~1e-6: ncfdtr(3, 7.709110801985059, 2, 1.5)
        // = 0.50000082. The expected values here are those at which scipy's ncfdtr returns 0.5 to
        // within 5e-16 (checked with scipy 1.17.1).
        SpecialAssert.Close(7.708932236738083, Special.Ncfdtridfd(3, 0.5, 2, 1.5), 1e-9, "ncfdtridfd");
        SpecialAssert.Close(3.1125150246160684, Special.Ncfdtridfn(0.5, 7, 2, 1.5), 1e-9, "ncfdtridfn");
        SpecialAssert.Close(1.9609626611684163, Special.Ncfdtrinc(3, 7, 0.5, 1.5), 1e-9, "ncfdtrinc");
        // nctdtrinc likewise: scipy's nctdtr(5, 1.8929610084247586, 2) = 0.4999999934.
        SpecialAssert.Close(1.8929609891169885, Special.Nctdtrinc(5, 0.5, 2), 1e-9, "nctdtrinc");
    }

    [Fact]
    public void Gdtrix_MatchesScipy()
    {
        double[][] rows =
        [
            [2.0, 3.0, 0.4, 1.1425384520016904],
            [0.5, 1.5, 0.99, 11.344866730144373],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Gdtrix(a[0], a[1], a[2])], 1e-12, "gdtrix");
    }

    [Fact]
    public void Gdtria_MatchesScipy()
    {
        double[][] rows =
        [
            [0.4, 3.0, 2.0, 1.1425384520016904],
            [0.9, 2.0, 5.0, 0.7779440339734858],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Gdtria(a[0], a[1], a[2])], 1e-10, "gdtria");
    }

    [Fact]
    public void Gdtrib_MatchesScipy()
    {
        double[][] rows =
        [
            [1.0, 0.3, 2.0, 3.102879671786032],
            [2.0, 0.8, 1.5, 2.004310441256648],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Gdtrib(a[0], a[1], a[2])], 1e-08, "gdtrib");
    }

    [Fact]
    public void Stdtridf_MatchesScipy()
    {
        double[][] rows =
        [
            [0.8, 1.0, 2.6387438984983635],
            [0.2, -1.5, 0.866277680133654],
            [0.95, 2.0, 5.176135682782311],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Stdtridf(a[0], a[1])], 1e-07, "stdtridf");
    }

    [Fact]
    public void Fdtridfd_MatchesScipy()
    {
        double[][] rows =
        [
            [3.0, 0.5, 1.2, 1.7503601024659923],
            [5.0, 0.8, 2.0, 6.70460333106414],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Fdtridfd(a[0], a[1], a[2])], 1e-07, "fdtridfd");
    }

    [Fact]
    public void Btdtria_MatchesScipy()
    {
        double[][] rows =
        [
            [0.5, 2.0, 0.3, 1.024930689471517],
            [0.1, 5.0, 0.6, 13.796821398769485],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Btdtria(a[0], a[1], a[2])], 1e-08, "btdtria");
    }

    [Fact]
    public void Btdtrib_MatchesScipy()
    {
        double[][] rows =
        [
            [2.0, 0.5, 0.3, 4.246702175718102],
            [3.0, 0.9, 0.4, 9.493291152498735],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Btdtrib(a[0], a[1], a[2])], 1e-08, "btdtrib");
    }

    [Fact]
    public void Bdtrik_MatchesScipy()
    {
        double[][] rows =
        [
            [0.5, 10.0, 0.5, 4.5],
            [0.3, 20.0, 0.3, 4.372619074262194],
            [0.0001, 10.0, 0.5, 0.0],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Bdtrik(a[0], a[1], a[2])], 1e-08, "bdtrik");
    }

    [Fact]
    public void Bdtrin_MatchesScipy()
    {
        double[][] rows =
        [
            [3.0, 0.5, 0.4, 8.83836263960212],
            [5.0, 0.2, 0.3, 24.805628692640962],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Bdtrin(a[0], a[1], a[2])], 1e-08, "bdtrin");
    }

    [Fact]
    public void Pdtrik_MatchesScipy()
    {
        double[][] rows =
        [
            [0.5, 5.0, 4.329370268245592],
            [0.9, 2.5, 4.095387225454171],
            [1e-09, 5.0, 0.0],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Pdtrik(a[0], a[1])], 1e-08, "pdtrik");
    }

    [Fact]
    public void Nbdtrik_MatchesScipy()
    {
        double[][] rows =
        [
            [0.5, 3.0, 0.4, 3.3402135731035525],
            [0.9, 5.0, 0.6, 5.987820536165971],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Nbdtrik(a[0], a[1], a[2])], 1e-08, "nbdtrik");
    }

    [Fact]
    public void Nbdtrin_MatchesScipy()
    {
        double[][] rows =
        [
            [2.0, 0.5, 0.4, 2.104311093747927],
            [4.0, 0.3, 0.5, 6.729169596817234],
        ];
        SpecialAssert.Table(rows, 1, a => [Special.Nbdtrin(a[0], a[1], a[2])], 1e-08, "nbdtrin");
    }

    [Fact]
    public void Inverses_RoundTripThroughTheirCdfs()
    {
        SpecialAssert.Close(0.37, Special.Chndtr(Special.Chndtrix(0.37, 4.5, 3.2), 4.5, 3.2), 1e-13, "chndtrix");
        SpecialAssert.Close(0.63, Special.Nctdtr(7, 1.2, Special.Nctdtrit(7, 1.2, 0.63)), 1e-12, "nctdtrit");
        SpecialAssert.Close(0.81, Special.Ncfdtr(4, 9, 1.5, Special.Ncfdtri(4, 9, 1.5, 0.81)), 1e-12, "ncfdtri");
        SpecialAssert.Close(0.25, Special.Gdtr(2.0, Special.Gdtrib(2.0, 0.25, 1.7), 1.7), 1e-13, "gdtrib");
        SpecialAssert.Close(0.3, Special.BetaInc(Special.Btdtria(0.3, 2.5, 0.4), 2.5, 0.4), 1e-13, "btdtria");
        // Continuous extension of the binomial: at k = 4.5 of n = 10, p = 0.5 the CDF is exactly 1/2.
        SpecialAssert.Close(4.5, Special.Bdtrik(0.5, 10, 0.5), 1e-13, "bdtrik symmetry");
        // With nc = 0 the noncentral CDFs are the central ones.
        SpecialAssert.Close(Special.Chdtr(3, 2.2), Special.Chndtr(2.2, 3, 0), 1e-14, "chndtr nc=0");
        SpecialAssert.Close(Special.Stdtr(6, 0.7), Special.Nctdtr(6, 0, 0.7), 1e-13, "nctdtr nc=0");
        SpecialAssert.Close(Special.Fdtr(3, 8, 1.1), Special.Ncfdtr(3, 8, 0, 1.1), 1e-13, "ncfdtr nc=0");
        // Unattainable: P(T ≤ 1) lies in (0.5, Φ(1)) for every df.
        Assert.True(double.IsNaN(Special.Stdtridf(0.3, 1.0)));
        Assert.True(double.IsNaN(Special.Nctdtridf(0.2, 0.5, 0.1))); // scipy returns its search bound, -1e100
    }
}
