"""Generates TransformReference.Generated.cs: N-d Rotation and RigidTransform values from scipy 1.17.

Run from this folder:  python gen_transforms.py
Inputs are drawn here with a fixed seed and written out with the outputs, so the C# tests rebuild
exactly the same objects (SciNet's random rotations come from a different generator).
"""
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R, RigidTransform as T
from scipy.spatial.transform import _rigid_transform

rng = np.random.default_rng(20251018)
out = {}


def put(name, a):
    out[name] = np.asarray(a, dtype=float)


# ---------------------------------------------------------------- N-d rotations
q = rng.normal(size=(2, 3, 4))
put("rot.quat_in", q)
r = R.from_quat(q)
put("rot.quat", r.as_quat())
put("rot.quat_canonical_scalar_first", r.as_quat(canonical=True, scalar_first=True))
put("rot.matrix", r.as_matrix())
put("rot.rotvec", r.as_rotvec())
put("rot.mrp", r.as_mrp())
put("rot.euler_zyx", r.as_euler("zyx"))
put("rot.euler_XYZ_deg", r.as_euler("XYZ", degrees=True))
put("rot.magnitude", r.magnitude())
put("rot.inv", r.inv().as_quat())
put("rot.pow_0.3", (r ** 0.3).as_quat())
put("rot.mean_all", r.mean().as_quat())
put("rot.mean_axis0", r.mean(axis=0).as_quat())
put("rot.mean_axis1", r.mean(axis=-1).as_quat())
w = rng.uniform(0.2, 2.0, size=(2, 3))
put("rot.weights", w)
put("rot.mean_weighted_axis0", r.mean(weights=w, axis=0).as_quat())
put("rot.compose_r_r0", (r * r[0]).as_quat())                       # (2,3) * (3,)
r31 = R.from_quat(rng.normal(size=(3, 1, 4)))
r4 = R.from_quat(rng.normal(size=(4, 4)))
put("rot.r31_in", r31.as_quat())
put("rot.r4_in", r4.as_quat())
put("rot.compose_31_4", (r31 * r4).as_quat())                        # (3,1) * (4,) -> (3,4)
put("rot.approx_31_4", r31.approx_equal(r4, atol=1.0).astype(float))
v = rng.normal(size=(2, 1, 3))
put("rot.vec_in", v)
put("rot.apply_bcast", r.apply(v))                                   # (2,3) with (2,1) -> (2,3,3)
put("rot.apply_inverse_bcast", r.apply(v, inverse=True))
put("rot.concat", R.concatenate([r, r[:1]]).as_quat())
ang = rng.normal(size=(2, 2, 3))
put("rot.euler_in", ang)
put("rot.from_euler_xyz", R.from_euler("xyz", ang).as_quat())
put("rot.from_davenport", R.from_davenport(np.eye(3), "e", ang).as_quat())
m = r.as_matrix()
put("rot.from_matrix_roundtrip", R.from_matrix(m).as_quat())

# ---------------------------------------------------------------- rigid transforms
t2 = rng.normal(size=(2, 3))
rq2 = rng.normal(size=(2, 4))
put("tf.t2", t2)
put("tf.rq2", rq2)
tf = T.from_components(t2, R.from_quat(rq2))
put("tf.matrix", tf.as_matrix())
put("tf.exp_coords", tf.as_exp_coords())
put("tf.dual_quat", tf.as_dual_quat())
put("tf.dual_quat_scalar_first", tf.as_dual_quat(scalar_first=True))
put("tf.translation", tf.translation)
put("tf.rotation_quat", tf.rotation.as_quat())
put("tf.inv", tf.inv().as_matrix())
put("tf.pow_0.5", (tf ** 0.5).as_matrix())
put("tf.pow_2.5", (tf ** 2.5).as_matrix())
put("tf.pow_-1", (tf ** -1).as_matrix())
p = rng.normal(size=(5, 3))
put("tf.points", p)
put("tf.apply_single_many", tf[0].apply(p))
put("tf.apply_pairwise", tf.apply(p[:2]))
put("tf.apply_one_point", tf.apply(p[0]))
put("tf.apply_inverse", tf.apply(p[:2], inverse=True))
tf1 = T.from_components(rng.normal(size=3), R.from_rotvec(rng.normal(size=3)))
put("tf1.matrix_in", tf1.as_matrix())
put("tf.compose_tf_tf1", (tf * tf1).as_matrix())
put("tf.compose_tf1_tf", (tf1 * tf).as_matrix())
e = np.array([[0.3, -0.2, 0.5, 1.0, 2.0, -1.0],
              [2e-4, -1e-4, 3e-4, 0.5, -0.4, 0.2],     # below the 1e-3 Taylor switch
              [0.0, 0.0, 0.0, 1.5, 0.0, -2.0],
              [2.9, 1.0, -0.5, -3.0, 0.1, 0.7]])        # angle close to pi
put("tf.exp_in", e)
te = T.from_exp_coords(e)
put("tf.from_exp", te.as_matrix())
put("tf.from_exp_roundtrip", te.as_exp_coords())
dq = rng.normal(size=(3, 8))
put("tf.dq_in", dq)
put("tf.from_dual_quat", T.from_dual_quat(dq).as_matrix())
put("tf.from_dual_quat_scalar_first", T.from_dual_quat(dq, scalar_first=True).as_matrix())
put("tf.normalize_dual_quat", _rigid_transform.normalize_dual_quaternion(dq))
wt = np.array([0.25, 1.75])
put("tf.mean", tf.mean().as_matrix())
put("tf.mean_weighted", tf.mean(weights=wt).as_matrix())
t23 = T.from_components(rng.normal(size=(2, 3, 3)), R.from_quat(rng.normal(size=(2, 3, 4))))
put("tf23.matrix_in", t23.as_matrix())
put("tf23.mean_axis1", t23.mean(axis=1).as_matrix())
put("tf23.compose_bcast", (t23 * tf1).as_matrix())
put("tf23.compose_31", (T.from_matrix(t23.as_matrix()[:, :1]) * t23[0]).as_matrix())  # (2,1) * (3,) -> (2,3)
nonortho = np.diag([2.0, 2.0, 2.0, 1.0])
nonortho[:3, 3] = [1, 2, 3]
put("tf.nonortho_in", nonortho)
put("tf.nonortho", T.from_matrix(nonortho).as_matrix())


def lit(v):
    v = float(v)
    if np.isnan(v):
        return "double.NaN"
    if np.isinf(v):
        return "double.PositiveInfinity" if v > 0 else "double.NegativeInfinity"
    s = repr(v)
    return s if ("." in s or "e" in s) else s + ".0"


lines = [f"// <auto-generated> Reference values generated from scipy {scipy.__version__} (numpy {np.__version__}) by gen_transforms.py. Do not edit. </auto-generated>",
         "namespace Gravicode.Science.Tests.SciNet;",
         "internal static class TransformReference",
         "{",
         "    public static readonly Dictionary<string, (int[] Shape, double[] Data)> Values = new()",
         "    {"]
for k, a in out.items():
    lines.append(f"        [\"{k}\"] = ([{', '.join(str(d) for d in a.shape)}], [{', '.join(lit(x) for x in a.ravel())}]),")
lines += ["    };", "}"]
with open("TransformReference.Generated.cs", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(lines) + "\n")
