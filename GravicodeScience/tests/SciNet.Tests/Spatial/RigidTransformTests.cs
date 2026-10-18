using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>N-d rotation arrays and RigidTransform against scipy 1.17.1 (TransformReference.Generated.cs).</summary>
public class RigidTransformTests
{
    private static NdArray Ref(string name)
    {
        var (shape, data) = TransformReference.Values[name];
        return new NdArray((double[])data.Clone(), shape);
    }

    private static void Same(string name, NdArray actual, double tol = 1e-13)
    {
        var (shape, data) = TransformReference.Values[name];
        Assert.Equal(shape, actual.Shape.ToArray());
        var a = actual.AsContiguous().ToArray();
        for (var i = 0; i < data.Length; i++)
            Assert.True(Math.Abs(a[i] - data[i]) <= tol * Math.Max(1, Math.Abs(data[i])), $"{name}[{i}]: expected {data[i]:R}, got {a[i]:R}");
    }

    /// <summary>Quaternions compared up to the overall sign of each one (q and âˆ’q are the same rotation).</summary>
    private static void SameRotation(string name, Rotation actual, double tol = 1e-12)
    {
        var (shape, data) = TransformReference.Values[name];
        Assert.Equal(shape[..^1], actual.Shape);
        var a = actual.AsQuat().AsContiguous().ToArray();
        for (var i = 0; i < data.Length; i += 4)
        {
            var dot = a[i] * data[i] + a[i + 1] * data[i + 1] + a[i + 2] * data[i + 2] + a[i + 3] * data[i + 3];
            var s = dot < 0 ? -1.0 : 1.0;
            for (var k = 0; k < 4; k++)
                Assert.True(Math.Abs(s * a[i + k] - data[i + k]) <= tol, $"{name}[{i / 4}]: expected {data[i + k]:R}, got {s * a[i + k]:R}");
        }
    }

    // ------------------------------------------------------------------ N-d rotations

    [Fact]
    public void NdRotationConversions()
    {
        var r = Rotation.FromQuat(Ref("rot.quat_in"));
        Assert.Equal([2, 3], r.Shape);
        Assert.False(r.Single);
        Assert.Equal(2, r.Length);
        Assert.Equal(6, r.Count);
        Same("rot.quat", r.AsQuat());
        Same("rot.quat_canonical_scalar_first", r.AsQuat(canonical: true, scalarFirst: true));
        Same("rot.matrix", r.AsMatrix());
        Same("rot.rotvec", r.AsRotvec());
        Same("rot.mrp", r.AsMrp());
        Same("rot.euler_zyx", r.AsEuler("zyx"), 1e-12);
        Same("rot.euler_XYZ_deg", r.AsEuler("XYZ", degrees: true), 1e-12);
        Same("rot.magnitude", new NdArray(r.Magnitude(), 2, 3));
        Same("rot.inv", r.Inv().AsQuat());
        SameRotation("rot.pow_0.3", r.Pow(0.3));
        SameRotation("rot.from_matrix_roundtrip", Rotation.FromMatrix(Ref("rot.matrix")));
        SameRotation("rot.from_euler_xyz", Rotation.FromEuler("xyz", Ref("rot.euler_in")));
        SameRotation("rot.from_davenport", Rotation.FromDavenport(new NdArray([1.0, 0, 0, 0, 1, 0, 0, 0, 1], 3, 3), "e", Ref("rot.euler_in")));
    }

    [Fact]
    public void NdRotationBroadcastingAndIndexing()
    {
        var r = Rotation.FromQuat(Ref("rot.quat_in"));
        Assert.Equal([3], r[0].Shape);
        Assert.True(r[1][2].Single);
        Same("rot.compose_r_r0", (r * r[0]).AsQuat());
        var r31 = Rotation.FromQuat(Ref("rot.r31_in"));
        var r4 = Rotation.FromQuat(Ref("rot.r4_in"));
        Same("rot.compose_31_4", (r31 * r4).AsQuat());
        Same("rot.approx_31_4", new NdArray(r31.ApproxEqual(r4, atol: 1.0).Select(b => b ? 1.0 : 0.0).ToArray(), 3, 4));
        Same("rot.apply_bcast", r.Apply(Ref("rot.vec_in")));
        Same("rot.apply_inverse_bcast", r.Apply(Ref("rot.vec_in"), inverse: true));
        Same("rot.concat", Rotation.Concatenate([r, r[..1]]).AsQuat());
        Assert.Throws<ArgumentException>(() => Rotation.Concatenate([r, r4]));
        Assert.Throws<ArgumentException>(() => r * r4);
        // Assignment along the first axis, as scipy's __setitem__.
        var copy = Rotation.FromQuat(Ref("rot.quat_in"));
        copy[0] = r[1];
        Assert.Equal(r[1].AsQuat().ToArray(), copy[0].AsQuat().ToArray());
        Assert.Equal(r[1].AsQuat().ToArray(), copy[1].AsQuat().ToArray());
        Assert.Equal([2, 0], Rotation.Identity([2, 0]).Shape);
        Assert.Equal([2, 2], Rotation.Random([2, 2], seed: 3).Shape);
        Assert.Equal([3, 3], r.Take([1, 0, 1]).Shape);
    }

    [Fact]
    public void NdRotationMeans()
    {
        var r = Rotation.FromQuat(Ref("rot.quat_in"));
        SameRotation("rot.mean_all", r.Mean());
        Assert.True(r.Mean().Single);
        SameRotation("rot.mean_axis0", r.Mean(axis: [0]));
        SameRotation("rot.mean_axis1", r.Mean(axis: [-1]));
        SameRotation("rot.mean_weighted_axis0", r.Mean(Ref("rot.weights").ToArray(), axis: [0]));
    }

    // ------------------------------------------------------------------ rigid transforms

    private static RigidTransform Tf() => RigidTransform.FromComponents(Ref("tf.t2"), Rotation.FromQuat(Ref("tf.rq2")));

    [Fact]
    public void RepresentationsMatchScipy()
    {
        var tf = Tf();
        Assert.Equal([2], tf.Shape);
        Same("tf.matrix", tf.AsMatrix());
        Same("tf.exp_coords", tf.AsExpCoords(), 1e-12);
        Same("tf.dual_quat", tf.AsDualQuat());
        Same("tf.dual_quat_scalar_first", tf.AsDualQuat(scalarFirst: true));
        Same("tf.translation", tf.Translation);
        SameRotation("tf.rotation_quat", tf.Rotation);
        var (t, r) = tf.AsComponents();
        Same("tf.translation", t);
        Assert.Equal([2], r.Shape);
        Same("tf.from_exp", RigidTransform.FromExpCoords(Ref("tf.exp_in")).AsMatrix(), 1e-13);
        Same("tf.from_exp_roundtrip", RigidTransform.FromExpCoords(Ref("tf.exp_in")).AsExpCoords(), 1e-12);
        Same("tf.from_dual_quat", RigidTransform.FromDualQuat(Ref("tf.dq_in")).AsMatrix());
        Same("tf.from_dual_quat_scalar_first", RigidTransform.FromDualQuat(Ref("tf.dq_in"), scalarFirst: true).AsMatrix());
        Same("tf.normalize_dual_quat", RigidTransform.NormalizeDualQuaternion(Ref("tf.dq_in")));
        Same("tf.nonortho", RigidTransform.FromMatrix(Ref("tf.nonortho_in")).AsMatrix());
    }

    [Fact]
    public void AlgebraMatchesScipy()
    {
        var tf = Tf();
        var tf1 = RigidTransform.FromMatrix(Ref("tf1.matrix_in"));
        Assert.True(tf1.Single);
        Same("tf.inv", tf.Inv().AsMatrix());
        Same("tf.pow_0.5", tf.Pow(0.5).AsMatrix(), 1e-12);
        Same("tf.pow_2.5", tf.Pow(2.5).AsMatrix(), 1e-12);
        Same("tf.pow_-1", tf.Pow(-1).AsMatrix());
        Same("tf.compose_tf_tf1", (tf * tf1).AsMatrix());
        Same("tf.compose_tf1_tf", (tf1 * tf).AsMatrix());
        Same("tf.apply_single_many", tf[0].Apply(Ref("tf.points")));
        var points = Ref("tf.points").AsContiguous().ToArray();
        Same("tf.apply_pairwise", tf.Apply(new NdArray(points[..6], 2, 3)));
        Same("tf.apply_one_point", tf.Apply(new NdArray(points[..3], 3)));
        Same("tf.apply_inverse", tf.Apply(new NdArray(points[..6], 2, 3), inverse: true));
        Same("tf.mean", tf.Mean().AsMatrix(), 1e-12);
        Same("tf.mean_weighted", tf.Mean([0.25, 1.75]).AsMatrix(), 1e-12);
        var t23 = RigidTransform.FromMatrix(Ref("tf23.matrix_in"));
        Assert.Equal([2, 3], t23.Shape);
        Same("tf23.mean_axis1", t23.Mean(axis: [1]).AsMatrix(), 1e-12);
        Same("tf23.compose_bcast", (t23 * tf1).AsMatrix());
        var m = Ref("tf23.matrix_in").AsContiguous().ToArray();
        var first = new double[32];
        Array.Copy(m, 0, first, 0, 16);
        Array.Copy(m, 48, first, 16, 16);
        Same("tf23.compose_31", (RigidTransform.FromMatrix(new NdArray(first, 2, 1, 4, 4)) * t23[0]).AsMatrix());
    }

    [Fact]
    public void GroupLawsHold()
    {
        var a = Tf();
        var b = RigidTransform.FromExpCoords(new NdArray([0.1, 0.2, -0.3, 1, 2, 3, -0.5, 0.4, 0.9, 0, -1, 0.5], 2, 6));
        var p = new NdArray([0.3, -1.2, 2.0], 3);
        // (a b) p = a (b p); aâ»Â¹ a = I; aâ¿ via exp coords composes: a^0.5 a^0.5 = a.
        var lhs = (a * b).Apply(p).ToArray();
        var rhs = a.Apply(b.Apply(p)).ToArray();
        for (var i = 0; i < lhs.Length; i++) Assert.True(Math.Abs(lhs[i] - rhs[i]) < 1e-13);
        var id = (a.Inv() * a).AsMatrix().ToArray();
        var eye = RigidTransform.Identity(2).AsMatrix().ToArray();
        for (var i = 0; i < id.Length; i++) Assert.True(Math.Abs(id[i] - eye[i]) < 1e-14);
        var half = a.Pow(0.5);
        var sq = (half * half).AsMatrix().ToArray();
        var am = a.AsMatrix().ToArray();
        for (var i = 0; i < am.Length; i++) Assert.True(Math.Abs(sq[i] - am[i]) < 1e-13);
        // Exponential coordinates and dual quaternions round-trip.
        var e = b.AsExpCoords().ToArray();
        Assert.Equal([0.1, 0.2, -0.3, 1, 2, 3, -0.5, 0.4, 0.9, 0, -1, 0.5], e.Select(x => Math.Round(x, 12)).ToArray());
        var dq = RigidTransform.FromDualQuat(a.AsDualQuat()).AsMatrix().ToArray();
        for (var i = 0; i < am.Length; i++) Assert.True(Math.Abs(dq[i] - am[i]) < 1e-14);
        // Indexing, assignment and concatenation along the first axis.
        var c = RigidTransform.Concatenate([a, b[1], a]);
        Assert.Equal([5], c.Shape);
        c[0] = b[0];
        Assert.Equal(b[0].AsMatrix().ToArray(), c[0].AsMatrix().ToArray());
        Assert.Equal([2], c[1..3].Shape);
        Assert.Equal([3, 0], RigidTransform.Identity([3, 0]).Shape);
        Assert.Throws<ArgumentException>(() => RigidTransform.FromMatrix(new NdArray([1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.5, 1], 4, 4)));
        Assert.Throws<InvalidOperationException>(() => RigidTransform.Identity()[0]);
    }
}
