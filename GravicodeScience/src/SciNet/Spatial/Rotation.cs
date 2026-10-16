using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Spatial;

/// <summary>
/// A rotation or a stack of rotations in 3-D (scipy.spatial.transform.Rotation), stored as unit
/// quaternions in scalar-last order (x, y, z, w) like scipy.
/// </summary>
/// <remarks>
/// <para>The algorithms are scipy's: matrix → quaternion by Shepperd's largest-pivot method
/// after projecting a non-orthogonal input onto the nearest rotation (polar factor U Vᵀ of the
/// SVD); Euler angles by Bernardes and Bewley (2022), with scipy's gimbal-lock rule (third angle
/// set to zero, reported through <c>gimbalLock</c> rather than a warning); the mean by Markley's
/// quaternion averaging (dominant eigenvector of Σ w q qᵀ); <see cref="AlignVectors"/> by Kabsch
/// with the reflection correction, and the exact minimal rotation for a single vector pair.</para>
/// <para>Small-angle Taylor branches for rotation vectors match scipy's thresholds (1e-3 rad) so
/// round trips agree to the last bits.</para>
/// <para>A <see cref="Single"/> rotation returns shapes without the leading stack axis
/// ((4), (3, 3), (3)), as scipy does; a stack of n returns (n, 4), (n, 3, 3), (n, 3).</para>
/// <para>Not implemented: <c>RotationSpline</c>, <c>reduce</c> with symmetry groups,
/// <c>create_group</c>, and infinite weights in <see cref="AlignVectors"/>.</para>
/// </remarks>
public sealed class Rotation
{
    private readonly double[] _q;

    /// <summary>True for a single rotation (not a stack of one).</summary>
    public bool Single { get; }

    /// <summary>Number of rotations (1 for a single rotation).</summary>
    public int Count => _q.Length / 4;

    private Rotation(double[] quats, bool single)
    {
        _q = quats;
        Single = single;
    }

    // ================================================================== construction

    private static double[] Normalise(double[] q)
    {
        for (var i = 0; i < q.Length; i += 4)
        {
            var n = Math.Sqrt(q[i] * q[i] + q[i + 1] * q[i + 1] + q[i + 2] * q[i + 2] + q[i + 3] * q[i + 3]);
            if (n == 0) throw new ArgumentException("Found zero norm quaternions in `quat`.");
            for (var k = 0; k < 4; k++) q[i + k] /= n;
        }
        return q;
    }

    private static double[] Rows(NdArray a, int width, out bool single, string name)
    {
        ArgumentNullException.ThrowIfNull(a, name);
        if (a.Rank == 1)
        {
            if (a.Shape[0] != width) throw new ArgumentException($"Expected `{name}` to have shape ({width},) or (N, {width}).", name);
            single = true;
        }
        else if (a.Rank == 2)
        {
            if (a.Shape[1] != width) throw new ArgumentException($"Expected `{name}` to have shape ({width},) or (N, {width}).", name);
            single = false;
        }
        else throw new ArgumentException($"Expected `{name}` to have shape ({width},) or (N, {width}).", name);
        return a.AsContiguous().ToArray();
    }

    /// <summary>From one quaternion, scalar-last (x, y, z, w) unless <paramref name="scalarFirst"/>; normalised.</summary>
    public static Rotation FromQuat(double[] quat, bool scalarFirst = false)
    {
        ArgumentNullException.ThrowIfNull(quat);
        return FromQuat(new NdArray((double[])quat.Clone(), quat.Length), scalarFirst);
    }

    /// <summary>From quaternions of shape (4) or (n, 4); each is normalised.</summary>
    public static Rotation FromQuat(NdArray quat, bool scalarFirst = false)
    {
        var q = Rows(quat, 4, out var single, nameof(quat));
        if (scalarFirst)
            for (var i = 0; i < q.Length; i += 4) (q[i], q[i + 1], q[i + 2], q[i + 3]) = (q[i + 1], q[i + 2], q[i + 3], q[i]);
        return new Rotation(Normalise(q), single);
    }

    /// <summary>
    /// From rotation matrices, shape (3, 3) or (n, 3, 3). A matrix that is not orthogonal (to
    /// 1e-12) is replaced by its nearest rotation U Vᵀ; a non-positive determinant throws.
    /// </summary>
    public static Rotation FromMatrix(NdArray matrix, bool assumeValid = false)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        bool single;
        if (matrix.Rank == 2 && matrix.Shape[0] == 3 && matrix.Shape[1] == 3) single = true;
        else if (matrix.Rank == 3 && matrix.Shape[1] == 3 && matrix.Shape[2] == 3) single = false;
        else throw new ArgumentException("Expected `matrix` to have shape (3, 3) or (N, 3, 3).", nameof(matrix));
        var m = matrix.AsContiguous().ToArray();
        var n = m.Length / 9;
        var q = new double[4 * n];
        Span<double> r = stackalloc double[9];
        for (var i = 0; i < n; i++)
        {
            m.AsSpan(9 * i, 9).CopyTo(r);
            if (!assumeValid)
            {
                var det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
                if (!(det > 0)) throw new ArgumentException($"Non-positive determinant (left-handed or null coordinate frame) in rotation matrix {i}.");
                if (!IsOrthogonal(r))
                {
                    var svd = Decomposition.Svd(new NdArray(r.ToArray(), 3, 3));
                    LinAlg.Dot(svd.U, svd.V.T.Copy()).AsContiguous().AsSpan().CopyTo(r);
                }
            }
            MatrixToQuat(r, q.AsSpan(4 * i, 4));
        }
        return new Rotation(q, single);
    }

    private static bool IsOrthogonal(ReadOnlySpan<double> r)
    {
        for (var a = 0; a < 3; a++)
            for (var b = 0; b < 3; b++)
            {
                var g = r[3 * a] * r[3 * b] + r[3 * a + 1] * r[3 * b + 1] + r[3 * a + 2] * r[3 * b + 2];
                var e = a == b ? 1.0 : 0.0;
                if (Math.Abs(g - e) > 1e-12 + 1e-5 * Math.Abs(e)) return false; // numpy isclose(atol=1e-12)
            }
        return true;
    }

    /// <summary>Shepperd's method: pick the largest of trace and the diagonal as the pivot.</summary>
    private static void MatrixToQuat(ReadOnlySpan<double> m, Span<double> q)
    {
        var tr = m[0] + m[4] + m[8];
        var choice = 0; var best = m[0];
        if (m[4] > best) { best = m[4]; choice = 1; }
        if (m[8] > best) { best = m[8]; choice = 2; }
        if (tr > best) choice = 3;
        switch (choice)
        {
            case 0: q[0] = 1 - tr + 2 * m[0]; q[1] = m[3] + m[1]; q[2] = m[6] + m[2]; q[3] = m[7] - m[5]; break;
            case 1: q[0] = m[3] + m[1]; q[1] = 1 - tr + 2 * m[4]; q[2] = m[7] + m[5]; q[3] = m[2] - m[6]; break;
            case 2: q[0] = m[6] + m[2]; q[1] = m[7] + m[5]; q[2] = 1 - tr + 2 * m[8]; q[3] = m[3] - m[1]; break;
            default: q[0] = m[7] - m[5]; q[1] = m[2] - m[6]; q[2] = m[3] - m[1]; q[3] = 1 + tr; break;
        }
        var n = Math.Sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        for (var k = 0; k < 4; k++) q[k] /= n;
    }

    /// <summary>From rotation vectors (axis × angle), shape (3) or (n, 3).</summary>
    public static Rotation FromRotvec(NdArray rotvec, bool degrees = false)
    {
        var v = Rows(rotvec, 3, out var single, nameof(rotvec));
        var n = v.Length / 3;
        var q = new double[4 * n];
        for (var i = 0; i < n; i++)
        {
            double x = v[3 * i], y = v[3 * i + 1], z = v[3 * i + 2];
            if (degrees) { x *= Math.PI / 180; y *= Math.PI / 180; z *= Math.PI / 180; }
            RotvecToQuat(x, y, z, q.AsSpan(4 * i, 4));
        }
        return new Rotation(q, single);
    }

    /// <summary>From one rotation vector.</summary>
    public static Rotation FromRotvec(double[] rotvec, bool degrees = false) => FromRotvec(new NdArray((double[])rotvec.Clone(), rotvec.Length), degrees);

    private static void RotvecToQuat(double x, double y, double z, Span<double> q)
    {
        var angle = Math.Sqrt(x * x + y * y + z * z);
        double scale;
        if (angle <= 1e-3)
        {
            var a2 = angle * angle;
            scale = 0.5 - a2 / 48 + a2 * a2 / 3840;
        }
        else scale = Math.Sin(angle / 2) / angle;
        q[0] = x * scale; q[1] = y * scale; q[2] = z * scale; q[3] = Math.Cos(angle / 2);
    }

    /// <summary>From modified Rodrigues parameters, shape (3) or (n, 3).</summary>
    public static Rotation FromMrp(NdArray mrp)
    {
        var v = Rows(mrp, 3, out var single, nameof(mrp));
        var n = v.Length / 3;
        var q = new double[4 * n];
        for (var i = 0; i < n; i++)
        {
            double x = v[3 * i], y = v[3 * i + 1], z = v[3 * i + 2];
            var s = x * x + y * y + z * z + 1;
            q[4 * i] = 2 * x / s; q[4 * i + 1] = 2 * y / s; q[4 * i + 2] = 2 * z / s; q[4 * i + 3] = (2 - s) / s;
        }
        return new Rotation(q, single);
    }

    private static int[] ParseSeq(string seq, int? required, out bool intrinsic)
    {
        ArgumentNullException.ThrowIfNull(seq);
        if (seq.Length < 1 || seq.Length > 3 || (required is { } r && seq.Length != r))
            throw new ArgumentException(required is { } rr ? $"Expected {rr} axes, got {seq}." : $"Expected axis specification to be a non-empty string of up to 3 characters, got {seq}");
        intrinsic = seq.All(c => c is 'X' or 'Y' or 'Z');
        var extrinsic = seq.All(c => c is 'x' or 'y' or 'z');
        if (!intrinsic && !extrinsic) throw new ArgumentException($"Expected axes from `seq` to be from ['x', 'y', 'z'] or ['X', 'Y', 'Z'], got {seq}");
        for (var i = 0; i + 1 < seq.Length; i++)
            if (seq[i] == seq[i + 1]) throw new ArgumentException($"Expected consecutive axes to be different, got {seq}");
        return seq.ToLowerInvariant().Select(c => c - 'x').ToArray();
    }

    /// <summary>
    /// From Euler angles. Upper-case axes ("XYZ") are intrinsic (rotating frame), lower-case
    /// ("xyz") extrinsic (fixed frame); 1 to 3 axes. <paramref name="angles"/> holds one angle per
    /// axis for a single rotation.
    /// </summary>
    public static Rotation FromEuler(string seq, double[] angles, bool degrees = false)
    {
        ArgumentNullException.ThrowIfNull(angles);
        return FromEulerCore(seq, angles, true, degrees);
    }

    /// <summary>From Euler angles for a stack: <paramref name="angles"/> has shape (n, len(seq)).</summary>
    public static Rotation FromEuler(string seq, NdArray angles, bool degrees = false)
    {
        ArgumentNullException.ThrowIfNull(angles);
        if (angles.Rank == 1) return FromEulerCore(seq, angles.ToArray(), true, degrees);
        if (angles.Rank != 2 || angles.Shape[1] != seq.Length)
            throw new ArgumentException("Expected last dimension of `angles` to match number of sequence axes specified.");
        return FromEulerCore(seq, angles.AsContiguous().ToArray(), false, degrees);
    }

    private static Rotation FromEulerCore(string seq, double[] angles, bool single, bool degrees)
    {
        var axes = ParseSeq(seq, null, out var intrinsic);
        var m = axes.Length;
        if (angles.Length % m != 0 || (single && angles.Length != m))
            throw new ArgumentException("Expected last dimension of `angles` to match number of sequence axes specified.");
        var n = angles.Length / m;
        var q = new double[4 * n];
        Span<double> e = stackalloc double[4];
        Span<double> t = stackalloc double[4];
        for (var i = 0; i < n; i++)
        {
            var qi = q.AsSpan(4 * i, 4);
            for (var k = 0; k < m; k++)
            {
                var a = angles[i * m + k] * (degrees ? Math.PI / 180 : 1.0);
                e.Clear(); e[3] = Math.Cos(a / 2); e[axes[k]] = Math.Sin(a / 2);
                if (k == 0) { e.CopyTo(qi); continue; }
                if (intrinsic) Compose(qi, e, t); else Compose(e, qi, t);
                t.CopyTo(qi);
            }
        }
        return new Rotation(q, single);
    }

    /// <summary>
    /// From Davenport angles about arbitrary axes (each consecutive pair orthogonal),
    /// <paramref name="axes"/> shape (k, 3), <paramref name="order"/> "e"/"extrinsic" or "i"/"intrinsic".
    /// </summary>
    public static Rotation FromDavenport(NdArray axes, string order, double[] angles, bool degrees = false)
    {
        ArgumentNullException.ThrowIfNull(axes); ArgumentNullException.ThrowIfNull(angles);
        var extrinsic = order switch
        {
            "e" or "extrinsic" => true,
            "i" or "intrinsic" => false,
            _ => throw new ArgumentException("order should be 'e'/'extrinsic' for extrinsic sequences or 'i'/'intrinsic' for intrinsic sequences."),
        };
        var ax = NormalisedAxes(axes, null);
        var k = ax.Length / 3;
        if (angles.Length != k) throw new ArgumentException("Expected `angles` to match number of axes.");
        Span<double> q = [0, 0, 0, 1];
        Span<double> qi = stackalloc double[4];
        Span<double> t = stackalloc double[4];
        for (var i = 0; i < k; i++)
        {
            var a = angles[i] * (degrees ? Math.PI / 180 : 1.0);
            RotvecToQuat(a * ax[3 * i], a * ax[3 * i + 1], a * ax[3 * i + 2], qi);
            if (extrinsic) Compose(qi, q, t); else Compose(q, qi, t);
            t.CopyTo(q);
        }
        return new Rotation(q.ToArray(), true);
    }

    private static double[] NormalisedAxes(NdArray axes, int? required)
    {
        if (axes.Rank != 2 || axes.Shape[1] != 3) throw new ArgumentException("Axes must be vectors of length 3.");
        var k = axes.Shape[0];
        if (k < 1 || k > 3 || (required is { } r && k != r)) throw new ArgumentException($"Expected up to 3 axes, got {k}.");
        var ax = axes.AsContiguous().ToArray();
        for (var i = 0; i < k; i++)
        {
            var n = Math.Sqrt(ax[3 * i] * ax[3 * i] + ax[3 * i + 1] * ax[3 * i + 1] + ax[3 * i + 2] * ax[3 * i + 2]);
            for (var c = 0; c < 3; c++) ax[3 * i + c] /= n;
        }
        for (var i = 0; i + 1 < k; i++)
            if (Math.Abs(ax[3 * i] * ax[3 * i + 3] + ax[3 * i + 1] * ax[3 * i + 4] + ax[3 * i + 2] * ax[3 * i + 5]) > 1e-7)
                throw new ArgumentException("Consecutive axes must be orthogonal.");
        return ax;
    }

    /// <summary>The identity rotation (single), or a stack of <paramref name="num"/> identities.</summary>
    public static Rotation Identity(int? num = null)
    {
        var n = num ?? 1;
        var q = new double[4 * n];
        for (var i = 0; i < n; i++) q[4 * i + 3] = 1;
        return new Rotation(q, num is null);
    }

    /// <summary>
    /// Uniformly distributed random rotations (normalised 4-D Gaussian samples). Seeded through
    /// GraviRandom, so the draws differ from numpy's for the same seed.
    /// </summary>
    public static Rotation Random(int? num = null, int seed = 0)
    {
        var rng = new GraviRandom(seed);
        var n = num ?? 1;
        var q = new double[4 * n];
        for (var i = 0; i < q.Length; i++) q[i] = rng.Normal();
        return new Rotation(Normalise(q), num is null);
    }

    /// <summary>Joins rotations (single or stacks) into one stack.</summary>
    public static Rotation Concatenate(IEnumerable<Rotation> rotations)
    {
        ArgumentNullException.ThrowIfNull(rotations);
        var all = new List<double>();
        foreach (var r in rotations) all.AddRange(r._q);
        return new Rotation([.. all], false);
    }

    /// <summary>
    /// The rotation R that best maps <paramref name="b"/> onto <paramref name="a"/> (a ≈ R b),
    /// minimising Σ wᵢ‖aᵢ - R bᵢ‖², and the root of that weighted sum (rssd).
    /// </summary>
    /// <remarks>Kabsch: SVD of B = Σ w a bᵀ, with the last singular vector flipped when
    /// det(U Vᵀ) &lt; 0 so the result is a rotation, never a reflection. A single vector pair has
    /// no unique answer; like scipy, the minimal rotation between the normalised vectors is
    /// returned (with a deliberate π flip when they are antiparallel).</remarks>
    public static (Rotation Rotation, double Rssd) AlignVectors(NdArray a, NdArray b, double[]? weights = null)
    {
        var av = Rows(a, 3, out _, nameof(a));
        var bv = Rows(b, 3, out _, nameof(b));
        if (av.Length != bv.Length) throw new ArgumentException("Expected inputs `a` and `b` to have same shapes.");
        var n = av.Length / 3;
        var w = weights ?? Enumerable.Repeat(1.0, n).ToArray();
        if (w.Length != n && !(n == 1 && w.Length == 1)) throw new ArgumentException("Expected `weights` to have number of values equal to number of input vectors.");
        foreach (var x in w)
        {
            if (x < 0) throw new ArgumentException("`weights` may not contain negative values");
            if (double.IsPositiveInfinity(x)) throw new ArgumentException("Infinite weights are not supported.");
        }
        if (n == 1) return AlignSingle(av, bv, w[0]);

        var bm = new double[9];
        for (var i = 0; i < n; i++)
            for (var r = 0; r < 3; r++)
                for (var c = 0; c < 3; c++) bm[3 * r + c] += w[i] * av[3 * i + r] * bv[3 * i + c];
        var svd = Decomposition.Svd(new NdArray(bm, 3, 3));
        var u = svd.U.AsContiguous().ToArray();
        var vt = svd.V.T.Copy().ToArray();
        var s = svd.SingularValues.ToArray();
        var c3 = Mul3(u, vt);
        if (Det3(c3) < 0)
        {
            s[2] = -s[2];
            for (var r = 0; r < 3; r++) u[3 * r + 2] = -u[3 * r + 2];
            c3 = Mul3(u, vt);
        }
        var ssd = 0.0;
        for (var i = 0; i < n; i++)
            for (var k = 0; k < 3; k++) ssd += w[i] * (bv[3 * i + k] * bv[3 * i + k] + av[3 * i + k] * av[3 * i + k]);
        ssd -= 2 * (s[0] + s[1] + s[2]);
        var q = new double[4];
        MatrixToQuat(c3, q);
        return (new Rotation(q, true), Math.Sqrt(Math.Max(ssd, 0)));
    }

    private static (Rotation, double) AlignSingle(double[] a, double[] b, double weight)
    {
        var na = Math.Sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        var nb = Math.Sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
        if (na == 0 || nb == 0) throw new ArgumentException("Cannot align zero length primary vectors");
        double[] ap = [a[0] / na, a[1] / na, a[2] / na], bp = [b[0] / nb, b[1] / nb, b[2] / nb];
        double[] cross = [bp[1] * ap[2] - bp[2] * ap[1], bp[2] * ap[0] - bp[0] * ap[2], bp[0] * ap[1] - bp[1] * ap[0]];
        var crossNorm = Math.Sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
        var theta = Math.Atan2(crossNorm, ap[0] * bp[0] + ap[1] * bp[1] + ap[2] * bp[2]);
        const double tolerance = 1e-3;
        double[] qFlip = [0, 0, 0, 1];
        if (Math.PI - theta < tolerance)
        {
            double[] r;
            if (crossNorm == 0)
            {
                // Antiparallel: any axis orthogonal to a will do; scipy's choice.
                var i = Math.Abs(ap[0]) <= Math.Abs(ap[1]) && Math.Abs(ap[0]) <= Math.Abs(ap[2]) ? 0 : Math.Abs(ap[1]) <= Math.Abs(ap[2]) ? 1 : 2;
                r = new double[3];
                r[(i + 2) % 3] = ap[(i + 1) % 3];
                r[(i + 1) % 3] = -ap[(i + 2) % 3];
            }
            else r = cross;
            var rn = Math.Sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            RotvecToQuat(r[0] / rn * Math.PI, r[1] / rn * Math.PI, r[2] / rn * Math.PI, qFlip);
            theta = Math.PI - theta;
            cross = [-cross[0], -cross[1], -cross[2]];
        }
        double scale;
        if (Math.Abs(theta) < tolerance)
        {
            var t2 = theta * theta;
            scale = 1 + t2 / 6 + t2 * t2 * 7 / 360;
        }
        else scale = theta / Math.Sin(theta);
        var qr = new double[4];
        RotvecToQuat(cross[0] * scale, cross[1] * scale, cross[2] * scale, qr);
        var q = new double[4];
        Compose(qr, qFlip, q);
        var rot = new Rotation(q, true);
        var est = rot.ApplyCore(b, false);
        var rssd = 0.0;
        // A single pair is aligned exactly in direction; only the length mismatch remains.
        for (var k = 0; k < 3; k++) rssd += (a[k] - est[k]) * (a[k] - est[k]);
        return (rot, Math.Sqrt(weight * rssd));
    }

    // ================================================================== conversion

    /// <summary>
    /// Quaternions, shape (4) for a single rotation or (n, 4). <paramref name="canonical"/> picks
    /// the sign with w &gt; 0 (ties broken on x, then y, then z), as scipy.
    /// </summary>
    public NdArray AsQuat(bool canonical = false, bool scalarFirst = false)
    {
        var q = (double[])_q.Clone();
        for (var i = 0; i < q.Length; i += 4)
        {
            if (canonical && NeedsFlip(q.AsSpan(i, 4))) for (var k = 0; k < 4; k++) q[i + k] = -q[i + k];
            if (scalarFirst) (q[i], q[i + 1], q[i + 2], q[i + 3]) = (q[i + 3], q[i], q[i + 1], q[i + 2]);
        }
        return Single ? new NdArray(q, 4) : new NdArray(q, Count, 4);
    }

    private static bool NeedsFlip(ReadOnlySpan<double> q)
    {
        if (q[3] != 0) return q[3] < 0;
        if (q[0] != 0) return q[0] < 0;
        if (q[1] != 0) return q[1] < 0;
        return q[2] < 0;
    }

    /// <summary>Rotation matrices, shape (3, 3) or (n, 3, 3).</summary>
    public NdArray AsMatrix()
    {
        var m = new double[9 * Count];
        for (var i = 0; i < Count; i++) QuatToMatrix(_q.AsSpan(4 * i, 4), m.AsSpan(9 * i, 9));
        return Single ? new NdArray(m, 3, 3) : new NdArray(m, Count, 3, 3);
    }

    private static void QuatToMatrix(ReadOnlySpan<double> q, Span<double> m)
    {
        double x = q[0], y = q[1], z = q[2], w = q[3];
        double x2 = x * x, y2 = y * y, z2 = z * z, w2 = w * w;
        double xy = x * y, zw = z * w, xz = x * z, yw = y * w, yz = y * z, xw = x * w;
        m[0] = x2 - y2 - z2 + w2; m[1] = 2 * (xy - zw); m[2] = 2 * (xz + yw);
        m[3] = 2 * (xy + zw); m[4] = -x2 + y2 - z2 + w2; m[5] = 2 * (yz - xw);
        m[6] = 2 * (xz - yw); m[7] = 2 * (yz + xw); m[8] = -x2 - y2 + z2 + w2;
    }

    /// <summary>Rotation vectors (axis × angle, angle in [0, π]), shape (3) or (n, 3).</summary>
    public NdArray AsRotvec(bool degrees = false)
    {
        var r = new double[3 * Count];
        Span<double> q = stackalloc double[4];
        for (var i = 0; i < Count; i++)
        {
            _q.AsSpan(4 * i, 4).CopyTo(q);
            if (NeedsFlip(q)) for (var k = 0; k < 4; k++) q[k] = -q[k];
            var axNorm = Math.Sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
            var angle = 2 * Math.Atan2(axNorm, q[3]);
            double scale;
            if (angle <= 1e-3)
            {
                var a2 = angle * angle;
                scale = 2 + a2 / 12 + 7 * a2 * a2 / 2880;
            }
            else scale = angle / Math.Sin(angle / 2);
            if (degrees) scale *= 180 / Math.PI;
            for (var k = 0; k < 3; k++) r[3 * i + k] = scale * q[k];
        }
        return Single ? new NdArray(r, 3) : new NdArray(r, Count, 3);
    }

    /// <summary>Modified Rodrigues parameters of the shorter rotation (norm ≤ 1), shape (3) or (n, 3).</summary>
    public NdArray AsMrp()
    {
        var r = new double[3 * Count];
        for (var i = 0; i < Count; i++)
        {
            var sign = _q[4 * i + 3] < 0 ? -1.0 : 1.0;
            var den = 1 + sign * _q[4 * i + 3];
            for (var k = 0; k < 3; k++) r[3 * i + k] = sign * _q[4 * i + k] / den;
        }
        return Single ? new NdArray(r, 3) : new NdArray(r, Count, 3);
    }

    /// <summary>Euler angles for a 3-axis sequence, each in [-π, π) (second in [0, π] or [-π/2, π/2]).</summary>
    public NdArray AsEuler(string seq, bool degrees = false) => AsEuler(seq, degrees, out _);

    /// <summary>
    /// Euler angles; <paramref name="gimbalLock"/> reports whether any rotation was at a
    /// singularity, where the third angle is set to zero (scipy warns instead).
    /// </summary>
    public NdArray AsEuler(string seq, bool degrees, out bool gimbalLock)
    {
        var axes = ParseSeq(seq, 3, out var intrinsic);
        var extrinsic = !intrinsic;
        if (!extrinsic) Array.Reverse(axes);
        int i = axes[0], j = axes[1], k = axes[2];
        var symmetric = i == k;
        if (symmetric) k = 3 - i - j;
        var sign = (double)((i - j) * (j - k) * (k - i) / 2);
        var r = new double[3 * Count];
        gimbalLock = false;
        for (var n = 0; n < Count; n++)
        {
            var q = _q.AsSpan(4 * n, 4);
            double a, b, c, d;
            if (symmetric) { a = q[3]; b = q[i]; c = q[j]; d = q[k] * sign; }
            else { a = q[3] - q[j]; b = q[i] + q[k] * sign; c = q[j] + q[3]; d = q[k] * sign - q[i]; }
            gimbalLock |= GetAngles(extrinsic, symmetric, sign, Math.PI / 2, a, b, c, d, r.AsSpan(3 * n, 3));
        }
        if (degrees) for (var t = 0; t < r.Length; t++) r[t] *= 180 / Math.PI;
        return Single ? new NdArray(r, 3) : new NdArray(r, Count, 3);
    }

    /// <summary>Bernardes and Bewley (2022), transcribed from scipy's _get_angles; returns true at a singularity.</summary>
    private static bool GetAngles(bool extrinsic, bool symmetric, double sign, double lamb,
        double a, double b, double c, double d, Span<double> angles)
    {
        const double eps = 1e-7;
        var halfSum = Math.Atan2(b, a);
        var halfDiff = Math.Atan2(d, c);
        angles.Clear();
        angles[1] = 2 * Math.Atan2(Hypot(c, d), Hypot(a, b));
        var first = extrinsic ? 0 : 2;
        var third = extrinsic ? 2 : 0;
        var case1 = Math.Abs(angles[1]) <= eps;
        var case2 = Math.Abs(angles[1] - Math.PI) <= eps;
        var case0 = !(case1 || case2);
        angles[0] = case1 ? 2 * halfSum : 2 * halfDiff * (extrinsic ? -1 : 1);
        if (case0) angles[first] = halfSum - halfDiff;
        var a3 = case0 ? halfSum + halfDiff : angles[third];
        if (!symmetric)
        {
            a3 *= sign;
            angles[1] -= lamb;
        }
        angles[third] = a3;
        for (var t = 0; t < 3; t++) angles[t] = FloorMod(angles[t] + Math.PI, 2 * Math.PI) - Math.PI;
        return !case0;
    }

    private static double Hypot(double x, double y) => double.Hypot(x, y);

    /// <summary>Python's float modulo (result has the sign of the divisor).</summary>
    private static double FloorMod(double x, double m)
    {
        var r = x % m;
        if (r != 0 && (r < 0) != (m < 0)) r += m;
        return r;
    }

    /// <summary>Davenport angles about three axes (consecutive pairs orthogonal), shape (3) or (n, 3).</summary>
    public NdArray AsDavenport(NdArray axes, string order, bool degrees = false)
    {
        ArgumentNullException.ThrowIfNull(axes);
        var extrinsic = order switch
        {
            "e" or "extrinsic" => true,
            "i" or "intrinsic" => false,
            _ => throw new ArgumentException("order should be 'e'/'extrinsic' or 'i'/'intrinsic'."),
        };
        var ax = NormalisedAxes(axes, 3);
        double[] n1 = ax[0..3], n2 = ax[3..6], n3 = ax[6..9];
        if (!extrinsic) (n1, n3) = (n3, n1);
        var nCross = Cross(n1, n2);
        var lamb = Math.Atan2(Dot(n3, nCross), Dot(n3, n1));
        var correct = lamb < 0;
        if (correct)
        {
            n2 = [-n2[0], -n2[1], -n2[2]];
            lamb = -lamb;
            nCross = [-nCross[0], -nCross[1], -nCross[2]];
        }
        double[] quatLamb = [Math.Sin(lamb / 2) * n2[0], Math.Sin(lamb / 2) * n2[1], Math.Sin(lamb / 2) * n2[2], Math.Cos(lamb / 2)];
        var r = new double[3 * Count];
        var qt = new double[4];
        for (var n = 0; n < Count; n++)
        {
            Compose(quatLamb, _q.AsSpan(4 * n, 4), qt);
            var a = qt[3];
            var b = qt[0] * n1[0] + qt[1] * n1[1] + qt[2] * n1[2];
            var c = qt[0] * n2[0] + qt[1] * n2[1] + qt[2] * n2[2];
            var d = qt[0] * nCross[0] + qt[1] * nCross[1] + qt[2] * nCross[2];
            var dst = r.AsSpan(3 * n, 3);
            GetAngles(extrinsic, false, 1, lamb, a, b, c, d, dst);
            if (correct) dst[1] = -dst[1];
        }
        if (degrees) for (var t = 0; t < r.Length; t++) r[t] *= 180 / Math.PI;
        return Single ? new NdArray(r, 3) : new NdArray(r, Count, 3);
    }

    private static double[] Cross(double[] u, double[] v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    private static double Dot(double[] u, double[] v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];

    // ================================================================== algebra

    /// <summary>Quaternion product p ⊗ q (apply q first, then p), scipy's compose_quat.</summary>
    private static void Compose(ReadOnlySpan<double> p, ReadOnlySpan<double> q, Span<double> r)
    {
        var cx = p[1] * q[2] - p[2] * q[1];
        var cy = p[2] * q[0] - p[0] * q[2];
        var cz = p[0] * q[1] - p[1] * q[0];
        var x = p[3] * q[0] + q[3] * p[0] + cx;
        var y = p[3] * q[1] + q[3] * p[1] + cy;
        var z = p[3] * q[2] + q[3] * p[2] + cz;
        var w = p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2];
        r[0] = x; r[1] = y; r[2] = z; r[3] = w;
    }

    /// <summary>Composition: <c>(p * q).Apply(v) == p.Apply(q.Apply(v))</c>. Stacks must have
    /// equal length or one of them a single rotation (or a stack of one).</summary>
    public static Rotation operator *(Rotation p, Rotation q)
    {
        ArgumentNullException.ThrowIfNull(p); ArgumentNullException.ThrowIfNull(q);
        if (p.Count != q.Count && p.Count != 1 && q.Count != 1)
            throw new ArgumentException($"Expected equal number of rotations in both or a single rotation in either object, got {p.Count} rotations in first and {q.Count} rotations in second object.");
        var n = Math.Max(p.Count, q.Count);
        var r = new double[4 * n];
        for (var i = 0; i < n; i++)
            Compose(p._q.AsSpan(4 * (p.Count == 1 ? 0 : i), 4), q._q.AsSpan(4 * (q.Count == 1 ? 0 : i), 4), r.AsSpan(4 * i, 4));
        return new Rotation(r, p.Single && q.Single);
    }

    /// <summary>The inverse rotation(s).</summary>
    public Rotation Inv()
    {
        var q = (double[])_q.Clone();
        for (var i = 0; i < q.Length; i += 4) { q[i] = -q[i]; q[i + 1] = -q[i + 1]; q[i + 2] = -q[i + 2]; }
        return new Rotation(q, Single);
    }

    /// <summary>Rotation composed with itself <paramref name="n"/> times (fractional allowed), via the rotation vector.</summary>
    public Rotation Pow(double n)
    {
        if (n == 0) return Single ? Identity() : Identity(Count);
        if (n == -1) return Inv();
        if (n == 1) return new Rotation((double[])_q.Clone(), Single);
        var rv = AsRotvec().AsContiguous().ToArray();
        for (var i = 0; i < rv.Length; i++) rv[i] *= n;
        return FromRotvec(Single ? new NdArray(rv, 3) : new NdArray(rv, Count, 3));
    }

    /// <summary>Rotation angle(s) in [0, π].</summary>
    public double[] Magnitude()
    {
        var m = new double[Count];
        for (var i = 0; i < Count; i++)
        {
            var s = Math.Sqrt(_q[4 * i] * _q[4 * i] + _q[4 * i + 1] * _q[4 * i + 1] + _q[4 * i + 2] * _q[4 * i + 2]);
            m[i] = 2 * Math.Atan2(s, Math.Abs(_q[4 * i + 3]));
        }
        return m;
    }

    /// <summary>Whether each rotation is within <paramref name="atol"/> (radians unless
    /// <paramref name="degrees"/>) of the other, measured by the angle of other·this⁻¹.</summary>
    public bool[] ApproxEqual(Rotation other, double atol = 1e-8, bool degrees = false)
    {
        var tol = degrees ? atol * Math.PI / 180 : atol;
        return (other * Inv()).Magnitude().Select(a => a < tol).ToArray();
    }

    /// <summary>
    /// The rotation minimising the weighted sum of squared chordal distances (Markley et al.
    /// 2007): the dominant eigenvector of Σ wᵢ qᵢ qᵢᵀ. Insensitive to quaternion sign.
    /// </summary>
    public Rotation Mean(double[]? weights = null)
    {
        if (Count == 0) throw new ArgumentException("Mean of an empty rotation set is undefined.");
        if (weights is not null)
        {
            if (weights.Length != Count) throw new ArgumentException("Expected `weights` to have one value per rotation.");
            if (weights.Any(w => w < 0)) throw new ArgumentException("`weights` must be non-negative.");
        }
        var k = new double[16];
        for (var i = 0; i < Count; i++)
        {
            var w = weights?[i] ?? 1.0;
            for (var r = 0; r < 4; r++)
                for (var c = 0; c < 4; c++) k[4 * r + c] += w * _q[4 * i + r] * _q[4 * i + c];
        }
        var eig = Decomposition.SymmetricEigen(new NdArray(k, 4, 4));
        var vals = eig.Values.ToArray();
        var vecs = eig.Vectors.AsContiguous().ToArray();
        var best = 0;
        for (var i = 1; i < 4; i++) if (vals[i] > vals[best]) best = i;
        var q = new double[4];
        for (var r = 0; r < 4; r++) q[r] = vecs[4 * r + best];
        return new Rotation(Normalise(q), true);
    }

    /// <summary>
    /// Rotates vectors: shape (3) or (m, 3). A single rotation rotates every vector; a stack of
    /// n rotates one vector n ways, or n vectors pairwise.
    /// </summary>
    public NdArray Apply(NdArray vectors, bool inverse = false)
    {
        var v = Rows(vectors, 3, out var singleVec, nameof(vectors));
        var m = v.Length / 3;
        if (Count != 1 && m != 1 && Count != m)
            throw new ArgumentException($"Expected equal numbers of rotations and vectors, or a single rotation, or a single vector, got {Count} rotations and {m} vectors.");
        var r = ApplyCore(v, inverse);
        var n = r.Length / 3;
        return Single && singleVec ? new NdArray(r, 3) : new NdArray(r, n, 3);
    }

    /// <summary>Rotates one vector.</summary>
    public double[] Apply(double[] vector, bool inverse = false)
    {
        ArgumentNullException.ThrowIfNull(vector);
        if (vector.Length != 3) throw new ArgumentException("Expected a vector of length 3.", nameof(vector));
        return ApplyCore(vector, inverse);
    }

    private double[] ApplyCore(double[] v, bool inverse)
    {
        var m = v.Length / 3;
        var n = Math.Max(m, Count);
        var r = new double[3 * n];
        Span<double> mat = stackalloc double[9];
        var lastRot = -1;
        for (var i = 0; i < n; i++)
        {
            var ri = Count == 1 ? 0 : i;
            if (ri != lastRot) { QuatToMatrix(_q.AsSpan(4 * ri, 4), mat); lastRot = ri; }
            var vi = m == 1 ? 0 : i;
            double x = v[3 * vi], y = v[3 * vi + 1], z = v[3 * vi + 2];
            for (var k = 0; k < 3; k++)
                r[3 * i + k] = inverse
                    ? mat[k] * x + mat[3 + k] * y + mat[6 + k] * z
                    : mat[3 * k] * x + mat[3 * k + 1] * y + mat[3 * k + 2] * z;
        }
        return r;
    }

    /// <summary>The i-th rotation of a stack (negative counts from the end), as a single rotation.</summary>
    public Rotation this[int index]
    {
        get
        {
            if (Single) throw new InvalidOperationException("Single rotation is not subscriptable.");
            if (index < 0) index += Count;
            if (index < 0 || index >= Count) throw new IndexOutOfRangeException($"Index {index} is out of range for {Count} rotations.");
            return new Rotation(_q[(4 * index)..(4 * index + 4)], true);
        }
    }

    /// <summary>A sub-stack.</summary>
    public Rotation this[Range range]
    {
        get
        {
            if (Single) throw new InvalidOperationException("Single rotation is not subscriptable.");
            var (off, len) = range.GetOffsetAndLength(Count);
            return new Rotation(_q[(4 * off)..(4 * (off + len))], false);
        }
    }

    /// <summary>The rotations at the given indices, as a stack.</summary>
    public Rotation Take(int[] indices)
    {
        ArgumentNullException.ThrowIfNull(indices);
        var q = new double[4 * indices.Length];
        for (var i = 0; i < indices.Length; i++) Array.Copy(_q, 4 * indices[i], q, 4 * i, 4);
        return new Rotation(q, false);
    }

    private static double[] Mul3(double[] a, double[] b)
    {
        var r = new double[9];
        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 3; j++)
                r[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
        return r;
    }

    private static double Det3(double[] r)
        => r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);

    /// <inheritdoc />
    public override string ToString() => Single ? $"Rotation(quat=[{string.Join(", ", _q.Select(x => x.ToString("G6")))}])" : $"Rotation[{Count}]";
}

/// <summary>
/// Spherical linear interpolation of rotations with a constant angular velocity between key
/// times (scipy.spatial.transform.Slerp).
/// </summary>
public sealed class Slerp
{
    private readonly double[] _times;
    private readonly Rotation _rotations;
    private readonly double[] _rotvecs;

    /// <summary>Key times (strictly increasing) and one rotation per time (at least two).</summary>
    public Slerp(double[] times, Rotation rotations)
    {
        ArgumentNullException.ThrowIfNull(times); ArgumentNullException.ThrowIfNull(rotations);
        if (rotations.Single || rotations.Count <= 1) throw new ArgumentException("`rotations` must be a sequence of at least 2 rotations.");
        if (times.Length != rotations.Count) throw new ArgumentException("Expected number of rotations to be equal to number of timestamps given.");
        for (var i = 1; i < times.Length; i++) if (!(times[i] > times[i - 1])) throw new ArgumentException("Times must be in strictly increasing order.");
        _times = (double[])times.Clone();
        _rotations = rotations[..^1];
        _rotvecs = (_rotations.Inv() * rotations[1..]).AsRotvec().AsContiguous().ToArray();
    }

    /// <summary>The interpolated rotation at one time in [times[0], times[^1]].</summary>
    public Rotation Evaluate(double time) => Evaluate([time])[0];

    /// <summary>The interpolated rotations at several times, as a stack.</summary>
    public Rotation Evaluate(double[] times)
    {
        ArgumentNullException.ThrowIfNull(times);
        var rv = new double[3 * times.Length];
        var idx = new int[times.Length];
        for (var i = 0; i < times.Length; i++)
        {
            var t = times[i];
            // searchsorted(side='left') - 1, with t == times[0] mapped to segment 0.
            var ind = Array.BinarySearch(_times, t);
            ind = ind >= 0 ? ind - 1 : ~ind - 1;
            if (t == _times[0]) ind = 0;
            if (ind < 0 || ind > _rotations.Count - 1 || double.IsNaN(t))
                throw new ArgumentException($"Interpolation times must be within the range [{_times[0]}, {_times[^1]}], both inclusive.");
            var alpha = (t - _times[ind]) / (_times[ind + 1] - _times[ind]);
            for (var k = 0; k < 3; k++) rv[3 * i + k] = _rotvecs[3 * ind + k] * alpha;
            idx[i] = ind;
        }
        return _rotations.Take(idx) * Rotation.FromRotvec(new NdArray(rv, times.Length, 3));
    }
}
