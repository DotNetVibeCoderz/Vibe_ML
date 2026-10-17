using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpatialAssert;

namespace Gravicode.Science.Tests.SciNet;

public class RotationExtrasTests
{
    private static double[] RotvecFixture()
    {
        var rv = Lcg(6, 3, 2);
        for (var i = 0; i < rv.Length; i++) rv[i] = (rv[i] - 0.5) * 3;
        return rv;
    }

    private static readonly double[] T = [0, 0.5, 1.7, 2.0, 3.1, 4.0];
    private static readonly double[] Q = [0.2, 1.0, 1.9, 2.5, 3.9, -0.5, 4.5];

    private static RotationSpline Spline() => new(T, Rotation.FromRotvec(new NdArray(RotvecFixture(), 6, 3)));

    [Fact]
    public void RotationSpline_MatchesScipy()
    {
        // scipy 1.17.1: rv = (lcg_points(6, 3, 2) - 0.5) * 3; RotationSpline(t, from_rotvec(rv)) at
        // q (two of them outside the key times: extrapolated).
        var s = Spline();
        Close([0.3008490820680549, 0.4992385464231205, 0.1873138664458244, 0.7906732694490527, 0.20714829585937386, 0.8427926404176141, 0.32077991080218743, 0.37932887789648195, -0.14264347852072948, 0.091080654689835, -0.3759723432481296, 0.9110444278350469, 0.15425059850234638, -0.28952306083323504, 0.006718834140259337, 0.9446364419051482, -0.1332612476309363, -0.13398427357831127, 0.5441656489947119, 0.8174187426085253, 0.28970395192841936, 0.01110209566306128, -0.062004433523595345, 0.9550412629475035, -0.33931057595643715, -0.13585884060609726, 0.6672825713403598, 0.6489566075314093],
            s.Evaluate(Q).AsQuat(canonical: true).ToArray(), 1e-10, "rotations");
        Close([-1.2703200474150118, 1.2228568708644543, 1.1840256825847237, 1.595336286547284, 0.7625438889539871, -0.36829899560120655, 0.8578069139701991, -5.431406507090918, -1.323613760494103, 2.8055462243303304, 2.2640892724176713, 1.1790833196530417, -2.490366069505388, 0.15781425961079837, -0.756465537943072, 1.315027616861929, 2.6765779727230874, -2.8493570802544066, 1.678958840329526, -0.5349467201483664, 3.219327747846851],
            s.Evaluate(Q, 1).ToArray(), 1e-9, "angular rate");
        Close([0.6266053283179988, 0.20228245897452465, -0.42708216398456245, 0.3298873371331005, -4.069190164060175, -1.8521669215011383, 2.1831152187254936, 8.790928212502575, 6.278263742032498, 1.8417966684108014, 4.2714357524505795, 0.9244106255590374, 3.218899683040902, 0.1915225365002613, 2.762100708126303, -5.122932251526783, -5.779748158117547, 12.441361676807169, 6.712917896789471, -2.129126872248604, 13.215687269764528],
            s.Evaluate(Q, 2).ToArray(), 1e-8, "angular acceleration");
    }

    [Fact]
    public void RotationSpline_PassesThroughKeys_AndTwoKeysIsConstantRate()
    {
        var rots = Rotation.FromRotvec(new NdArray(RotvecFixture(), 6, 3));
        var s = Spline();
        Assert.All(s.Evaluate(T).ApproxEqual(rots, 1e-12), Assert.True);
        // scipy 1.17.1: RotationSpline(t[:2], rots[:2]) at 0.3
        var s2 = new RotationSpline(T[..2], rots[..2]);
        Close([0.2797314229939519, 0.5357583018308903, 0.2541448197427681, 0.7550654167732591], s2.Evaluate([0.3]).AsQuat(canonical: true).ToArray(), 1e-12, "2 keys");
        Close([-1.045631333102628, 1.3046815024633953, 0.9845428060029852], s2.Evaluate([0.3], 1).ToArray(), 1e-12, "2 keys rate");
        Assert.Throws<ArgumentException>(() => new RotationSpline([0, 0], rots[..2]));
    }

    [Theory]
    [InlineData("I", 60)]
    [InlineData("O", 24)]
    [InlineData("T", 12)]
    [InlineData("D5", 10)]
    [InlineData("C7", 7)]
    public void CreateGroup_IsClosedGroupOfRightOrder(string name, int order)
    {
        var g = Rotation.CreateGroup(name);
        Assert.Equal(order, g.Count);
        // Closure: every product is (up to sign) an element of the group.
        var q = g.AsQuat().ToArray();
        for (var a = 0; a < order; a++)
            for (var b = 0; b < order; b++)
            {
                var prod = (g[a] * g[b]).AsQuat().ToArray();
                var found = false;
                for (var c = 0; c < order && !found; c++)
                {
                    var dot = prod[0] * q[4 * c] + prod[1] * q[4 * c + 1] + prod[2] * q[4 * c + 2] + prod[3] * q[4 * c + 3];
                    found = Math.Abs(Math.Abs(dot) - 1) < 1e-12;
                }
                Assert.True(found, $"{name}: product {a}*{b} not in the group");
            }
    }

    [Fact]
    public void CreateGroup_D3AboutX_MatchesScipy()
    {
        // scipy 1.17.1: Rotation.create_group('D3', axis='X').as_quat()
        Close([0.0, 0.0, 0.0, 1.0, 0.8660254037844386, 0.0, 0.0, 0.5000000000000001, -0.8660254037844387, -0.0, -0.0, 0.4999999999999999, 0.0, 1.0, 0.0, 6.123031769111886e-17, 0.0, 0.5000000000000001, 0.8660254037844386, 6.123031769111886e-17, 0.0, -0.4999999999999998, 0.8660254037844388, 6.123031769111886e-17],
            Rotation.CreateGroup("D3", "X").AsQuat().ToArray(), 1e-15, "D3");
        Assert.Throws<ArgumentException>(() => Rotation.CreateGroup("Q"));
        Assert.Throws<ArgumentException>(() => Rotation.CreateGroup("C0"));
    }

    [Fact]
    public void Reduce_MatchesScipy()
    {
        var p = Rotation.FromRotvec(new NdArray(RotvecFixture(), 6, 3));
        // scipy 1.17.1: p.reduce(left=create_group('O'), return_indices=True)
        var (r, li, ri) = p.Reduce(left: Rotation.CreateGroup("O"));
        Assert.Equal(new[] { 16, 11, 16, 14, 11, 15 }, li);
        Assert.Null(ri);
        Close([0.25389890890147504, -0.2636556901185642, 0.24921920352172236, 0.8966107347800061, -0.15275374173037334, 0.04694114112418388, -0.30739063366047104, 0.9380691989374472],
            r.AsQuat(canonical: true).ToArray()[..8], 1e-12, "reduced");
        // scipy 1.17.1: p.reduce(left=create_group('T'), right=create_group('D3'), return_indices=True)
        var (r2, l2, r2i) = p.Reduce(Rotation.CreateGroup("T"), Rotation.CreateGroup("D3"));
        // scipy picks (5,2) (8,5) (11,1) (1,5) (6,3) (1,4); every element has two exactly tied
        // pairs here (T and D3 share a half-turn), and scipy's choice between them follows
        // rounding in the last bit, so the pair is checked for optimality, not identity.
        (int, int)[][] ties = [[(5, 2), (8, 5)], [(5, 2), (8, 5)], [(6, 4), (11, 1)], [(1, 5), (2, 2)], [(6, 3), (11, 0)], [(1, 4), (2, 1)]];
        for (var i = 0; i < 6; i++) Assert.Contains((l2![i], r2i![i]), ties[i]);
        var g1 = Rotation.CreateGroup("T"); var g2 = Rotation.CreateGroup("D3");
        for (var i = 0; i < 6; i++) Assert.True((g1[l2![i]] * p[i] * g2[r2i![i]]).ApproxEqual(r2[i], 1e-12)[0]);
        Close([0.7497022461455375, 0.5223512998933574, 0.5030152032981532, 0.5491974645927552, 0.4981922719942677, 0.6063572652943243], r2.Magnitude(), 1e-12, "magnitudes");
    }
}
