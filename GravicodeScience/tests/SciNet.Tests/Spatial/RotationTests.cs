using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpatialAssert;

namespace Gravicode.Science.Tests.SciNet;

public class RotationTests
{
    private static readonly double[] Q = [0.1, -0.4, 0.3, 0.8];

    private static void SameRotation(double[] expectedQuat, Rotation r, double tol, string what)
    {
        var q = r.AsQuat().ToArray();
        var dot = Math.Abs(expectedQuat.Zip(q, (a, b) => a * b).Sum()) / Math.Sqrt(expectedQuat.Sum(x => x * x));
        Assert.True(1 - dot <= tol, $"{what}: quaternions differ ({string.Join(",", q)})");
    }

    [Fact]
    public void FromEuler_MatchesScipy()
    {
        // scipy 1.17.1
        Close([0.03813457647485015, 0.18930785741199999, 0.2392983377447303, 0.9515485246437885],
            Rotation.FromEuler("xyz", [10, 20, 30], degrees: true).AsQuat().ToArray(), 1e-15, "xyz");
        Close([0.2392983377447303, 0.18930785741199999, 0.03813457647485015, 0.9515485246437885],
            Rotation.FromEuler("ZYX", [10, 20, 30], degrees: true).AsQuat().ToArray(), 1e-15, "ZYX");
        Close([0.2306493589716389, -0.515385288854499, -0.6200584751460557, 0.5447075955390057],
            Rotation.FromEuler("zxz", [0.3, 1.2, -2.0]).AsQuat().ToArray(), 1e-15, "zxz");
    }

    [Fact]
    public void AsEuler_MatchesScipy()
    {
        var r = Rotation.FromQuat(Q);
        Close([-0.1418970546041639, -0.8911225078866528, 0.7853981633974482], r.AsEuler("xyz").ToArray(), 1e-14, "xyz");
        Close([0.6747409422235526, -0.08900636238054727, -0.8960553845713439], r.AsEuler("ZXY").ToArray(), 1e-14, "ZXY");
        Close([-3.027800646446085, 0.8992181501769813, -2.5378433201923567], r.AsEuler("zyz").ToArray(), 1e-14, "zyz");
        Close([1.0516502125483738, 1.110242335113574, -0.8029402234548508], r.AsEuler("XZX").ToArray(), 1e-14, "XZX");
        Close([-0.8960553845713439, -0.08900636238054727, 0.6747409422235526], r.AsEuler("yxz").ToArray(), 1e-14, "yxz");
        var g = Rotation.FromEuler("xyz", [0.3, Math.PI / 2, 0.2]).AsEuler("xyz", false, out var locked);
        Assert.True(locked);
        Close([0.10000000000000003, 1.5707963267948966, 0.0], g.ToArray(), 1e-12, "gimbal");
    }

    [Fact]
    public void EulerRoundTrips_AllTwelveSequences_BothConventions()
    {
        var seqs = new[] { "xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "xyx", "xzx", "yxy", "yzy", "zxz", "zyz" };
        var rots = Rotation.Random(50, seed: 3);
        foreach (var s0 in seqs)
            foreach (var seq in new[] { s0, s0.ToUpperInvariant() })
            {
                var angles = rots.AsEuler(seq, false, out var locked);
                Assert.False(locked);
                var back = Rotation.FromEuler(seq, angles);
                Assert.All(back.ApproxEqual(rots, 1e-12), Assert.True);
                var a = angles.ToArray();
                var symmetric = seq[0] == seq[2] || char.ToLower(seq[0]) == char.ToLower(seq[2]);
                for (var i = 0; i < 50; i++)
                {
                    if (symmetric) Assert.InRange(a[3 * i + 1], 0, Math.PI);
                    else Assert.InRange(a[3 * i + 1], -Math.PI / 2, Math.PI / 2);
                }
            }
    }

    [Fact]
    public void Conversions_MatchScipy()
    {
        var r = Rotation.FromQuat(Q);
        Close([0.22257238818802236, -0.8902895527520894, 0.6677171645640669], r.AsRotvec().ToArray(), 1e-15, "rotvec");
        Close([0.05718588386558223, -0.22874353546232892, 0.17155765159674669], r.AsMrp().ToArray(), 1e-15, "mrp");
        Close([0.4444444444444445, -0.6222222222222221, -0.6444444444444445, 0.4444444444444443, 0.7777777777777779, -0.4444444444444444, 0.7777777777777778, -0.08888888888888882, 0.6222222222222222],
            r.AsMatrix().ToArray(), 1e-15, "matrix");
        SameRotation(Q, Rotation.FromMatrix(r.AsMatrix()), 1e-15, "matrix round trip");
        SameRotation(Q, Rotation.FromRotvec(r.AsRotvec()), 1e-15, "rotvec round trip");
        SameRotation(Q, Rotation.FromMrp(r.AsMrp()), 1e-15, "mrp round trip");
        // Non-orthogonal input is projected onto the nearest rotation first.
        var m = new NdArray([0.9, 0.1, 0.0, -0.1, 1.1, 0.05, 0.02, 0.0, 0.95], 3, 3);
        Close([-0.012389875964958365, -0.006055382340252688, -0.04993639308887169, 0.9986571884102972], Rotation.FromMatrix(m).AsQuat().ToArray(), 1e-14, "from_matrix");
        Assert.Throws<ArgumentException>(() => Rotation.FromMatrix(new NdArray([-1.0, 0, 0, 0, 1, 0, 0, 0, 1], 3, 3)));
        // Scalar-first and canonical.
        var sf = Rotation.FromQuat([0.8, 0.1, -0.4, 0.3], scalarFirst: true);
        SameRotation(Q, sf, 1e-15, "scalar first");
        var neg = Rotation.FromQuat([-0.1, 0.4, -0.3, -0.8]).AsQuat(canonical: true).ToArray();
        Close(Q.Select(x => x / Math.Sqrt(0.9)).ToArray(), neg, 1e-15, "canonical");
        // Small angles take the Taylor branch.
        var tiny = Rotation.FromRotvec([1e-5, -2e-5, 3e-6]);
        Close([1e-5, -2e-5, 3e-6], tiny.AsRotvec().ToArray(), 1e-20, "tiny rotvec");
    }

    [Fact]
    public void Composition_IsAssociative_AndApplyAgreesWithMatrix()
    {
        var a = Rotation.Random(10, 1); var b = Rotation.Random(10, 2); var c = Rotation.Random(10, 3);
        Assert.All(((a * b) * c).ApproxEqual(a * (b * c), 1e-13), Assert.True);
        Assert.All((a * a.Inv()).Magnitude(), m => Assert.True(m < 1e-7));
        var v = new NdArray([0.3, -1.2, 2.0], 3);
        var lhs = (a * b).Apply(v).ToArray();
        var rhs = a.Apply(b.Apply(v)).ToArray();
        Close(lhs, rhs, 1e-14, "apply composition");
        var m = a[4].AsMatrix().ToArray();
        var rv = a[4].Apply([0.3, -1.2, 2.0]);
        for (var i = 0; i < 3; i++) Close(m[3 * i] * 0.3 - m[3 * i + 1] * 1.2 + m[3 * i + 2] * 2.0, rv[i], 1e-15, "matrix apply");
        Close([0.3, -1.2, 2.0], a[4].Apply(rv, inverse: true), 1e-14, "inverse apply");
        Assert.Equal(3, a[2..5].Count);
        Assert.Equal(20, Rotation.Concatenate([a, b]).Count);
        Close(1.0, Rotation.FromRotvec([0, 0, 0.5]).Pow(2).Magnitude()[0], 1e-15, "pow");
    }

    [Fact]
    public void AlignVectors_MatchesScipy()
    {
        var a = new NdArray([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0.0], 4, 3);
        var truth = Rotation.FromRotvec([0.2, -0.5, 0.3]);
        var b = truth.Inv().Apply(a).ToArray();
        double[] noise = [0.01, 0, 0, 0, -0.02, 0, 0, 0, 0.01, 0.005, 0, 0];
        for (var i = 0; i < 12; i++) b[i] += noise[i];
        var (rot, rssd) = Rotation.AlignVectors(a, new NdArray(b, 4, 3));
        // scipy 1.17.1
        SameRotation([0.09915766247210538, -0.24406157726454755, 0.1482489696381873, 0.9532176810558148], rot, 1e-14, "kabsch");
        Close(0.023761721654423373, rssd, 1e-12, "rssd");
        var (r1, s1) = Rotation.AlignVectors(new NdArray([0, 0, 2.0], 1, 3), new NdArray([1, 0, 0.0], 1, 3));
        SameRotation([0.0, -0.7071067811865475, 0.0, 0.7071067811865476], r1, 1e-15, "single pair");
        Close(1.0, s1, 1e-15, "single rssd");
        var (r2, _) = Rotation.AlignVectors(new NdArray([1, 0, 0.0], 1, 3), new NdArray([-1, 0, 0.0], 1, 3));
        SameRotation([0.0, 0.0, -1.0, 6.123031769111886e-17], r2, 1e-15, "antiparallel");
    }

    [Fact]
    public void MeanMagnitudeSlerpDavenport_MatchScipy()
    {
        var rs = Rotation.FromQuat(new NdArray([0.1, 0.2, 0.3, 0.9, 0.0, 0.1, -0.2, 0.95, 0.3, -0.1, 0.0, 0.9], 3, 4));
        Close([0.14552059541442666, 0.06939944016563693, 0.03466863551810233, 0.9863090589294211], rs.Mean().AsQuat(canonical: true).ToArray(), 1e-13, "mean");
        Close([0.07527669296021992, 0.10556870098515236, -0.03550170837346145, 0.9909229523892593], rs.Mean([1, 2, 0.5]).AsQuat(canonical: true).ToArray(), 1e-13, "weighted mean");
        Close([0.788002053283779, 0.46233570294079707, 0.6757792223317213], rs.Magnitude(), 1e-15, "magnitude");

        var s = new Slerp([0, 1, 3], rs);
        var e = s.Evaluate([0.5, 2.0, 3.0]).AsQuat().ToArray();
        Close([0.053240773996468975, 0.15965240630068098, 0.05338060537392091, 0.9842901198917792,
               0.1610051451936444, -0.0012109886479615625, -0.10491478616650646, 0.9813606698718388,
               0.3144854510165755, -0.10482848367219186, -2.0816681711721685e-17, 0.9434563530497265], e, 1e-14, "slerp");
        Assert.All(s.Evaluate([0.0, 1.0]).ApproxEqual(rs[..2], 1e-14), Assert.True);
        Assert.Throws<ArgumentException>(() => s.Evaluate(3.5));

        var ax = new NdArray([1, 0, 0, 0, 1, 0, 1, 0, 0.0], 3, 3);
        Close([-2.3737365502497476, 1.110242335113574, 2.62244653934327], Rotation.FromQuat(Q).AsDavenport(ax, "extrinsic").ToArray(), 1e-14, "davenport");
        Close([0.38165590209504824, 0.19470917115432523, -0.03946950299855746, 0.90270109637546],
            Rotation.FromDavenport(ax, "intrinsic", [0.2, 0.4, 0.6]).AsQuat().ToArray(), 1e-15, "from davenport");
    }

    [Fact]
    public void ProcrustesAndSlerp_MatchScipy()
    {
        var a = new NdArray([1, 2, 3, 4.0, 5, 7, 2, 1], 4, 2);
        var b = new NdArray([2, 1, 4, 3.0, 7, 5, 1, 1], 4, 2);
        var (_, m2, d) = SpatialFunctions.Procrustes(a, b);
        Close(0.016281512605042014, d, 1e-14, "disparity");
        Close([-0.24063335546990536, -0.2836035975181029, 0.08021111848996845, 0.09453453250603426, 0.3867321784337764, 0.6474183135261742, -0.2263099414538395, -0.45834924851410563], m2.ToArray(), 1e-14, "mtx2");
        var (r, sc) = SpatialFunctions.OrthogonalProcrustes(a, b);
        Close([0.009345386270320192, 0.9999563309242352, 0.9999563309242352, -0.00934538627032017], r.ToArray(), 1e-14, "R");
        Close(107.00467279516347, sc, 1e-11, "scale");
        var g = SpatialFunctions.GeometricSlerp([1, 0, 0], [0, 1, 0], [0, 0.25, 1]).ToArray();
        Close([1.0, 0.0, 0.0, 0.9238795325112867, 0.3826834323650898, 0.0, 6.123233995736766e-17, 1.0, 0.0], g, 1e-15, "geometric slerp");
    }
}
