using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Spatial;

/// <summary>
/// Rigid transforms in 3-D — rotation then translation, the proper Euclidean group SE(3)
/// (scipy.spatial.transform.RigidTransform, scipy 1.16+). Stored as 4×4 homogeneous matrices
/// [[R, t], [0, 0, 0, 1]], in an array of any <see cref="Shape"/>.
/// </summary>
/// <remarks>
/// <para>Composition reads right to left: <c>(a * b).Apply(p) == a.Apply(b.Apply(p))</c>. Shapes
/// broadcast like numpy in <c>*</c> and <see cref="Apply(NdArray, bool)"/>; indexing works along the first axis.</para>
/// <para>The algorithms are scipy's: <see cref="FromMatrix"/> re-orthonormalises the rotation block
/// through a quaternion (with an SVD only if it is not orthogonal to 1e-12), and so does every
/// composition; exponential coordinates use the closed-form SE(3) exponential and logarithm with
/// scipy's Taylor branches below an angle of 1e-3; dual quaternions are normalised so the real part is a
/// unit quaternion and the dual part orthogonal to it.</para>
/// </remarks>
public sealed class RigidTransform
{
    private readonly double[] _m; // 16 per transform, row-major
    private readonly int[] _shape;

    private RigidTransform(double[] matrices, int[] shape)
    {
        _m = matrices;
        _shape = shape;
    }

    /// <summary>True for a single transform (shape ()).</summary>
    public bool Single => _shape.Length == 0;

    /// <summary>The array shape of the transforms.</summary>
    public int[] Shape => (int[])_shape.Clone();

    /// <summary>Total number of transforms.</summary>
    public int Count => _m.Length / 16;

    /// <summary>The length of the first axis (scipy's <c>len</c>).</summary>
    public int Length => Single ? throw new InvalidOperationException("Single transform has no len.") : _shape[0];

    private static NdArray Shaped(double[] data, int[] lead, params int[] trailing) => new(data, [.. lead, .. trailing]);

    private static double[] Trailing(NdArray a, int[] trailing, string name, out int[] lead)
    {
        ArgumentNullException.ThrowIfNull(a, name);
        var r = trailing.Length;
        if (a.Rank < r || !a.Shape[^r..].ToArray().SequenceEqual(trailing))
            throw new ArgumentException($"Expected `{name}` to have shape (..., {string.Join(", ", trailing)}), got ({string.Join(", ", a.Shape.ToArray())}).", name);
        lead = a.Shape[..^r].ToArray();
        return a.AsContiguous().ToArray();
    }

    // ================================================================== construction

    /// <summary>
    /// From homogeneous matrices of shape (..., 4, 4). The last row must be exactly [0, 0, 0, 1];
    /// the rotation block is replaced by the nearest rotation (a non-positive determinant throws).
    /// </summary>
    public static RigidTransform FromMatrix(NdArray matrix)
    {
        var m = Trailing(matrix, [4, 4], nameof(matrix), out var lead);
        var n = m.Length / 16;
        for (var i = 0; i < n; i++)
        {
            var o = 16 * i;
            if (m[o + 12] != 0 || m[o + 13] != 0 || m[o + 14] != 0 || m[o + 15] != 1)
                throw new ArgumentException($"Expected last row of transformation matrix {i} to be exactly [0, 0, 0, 1], got [{m[o + 12]}, {m[o + 13]}, {m[o + 14]}, {m[o + 15]}].");
        }
        NormaliseRotations(m);
        return new RigidTransform(m, lead);
    }

    /// <summary>Re-orthonormalises every rotation block through a quaternion (scipy's from_matrix normalisation).</summary>
    private static void NormaliseRotations(double[] m)
    {
        var n = m.Length / 16;
        if (n == 0) return;
        var r = new double[9 * n];
        for (var i = 0; i < n; i++)
            for (var a = 0; a < 3; a++)
                for (var b = 0; b < 3; b++) r[9 * i + 3 * a + b] = m[16 * i + 4 * a + b];
        var rm = Rotation.FromMatrix(new NdArray(r, n, 3, 3)).AsMatrix().AsContiguous().ToArray();
        for (var i = 0; i < n; i++)
            for (var a = 0; a < 3; a++)
                for (var b = 0; b < 3; b++) m[16 * i + 4 * a + b] = rm[9 * i + 3 * a + b];
    }

    /// <summary>A pure rotation (zero translation), with the rotation's shape.</summary>
    public static RigidTransform FromRotation(Rotation rotation)
    {
        ArgumentNullException.ThrowIfNull(rotation);
        var rm = rotation.AsMatrix().AsContiguous().ToArray();
        var n = rm.Length / 9;
        var m = new double[16 * n];
        for (var i = 0; i < n; i++)
        {
            for (var a = 0; a < 3; a++)
                for (var b = 0; b < 3; b++) m[16 * i + 4 * a + b] = rm[9 * i + 3 * a + b];
            m[16 * i + 15] = 1;
        }
        return new RigidTransform(m, rotation.Shape);
    }

    /// <summary>A pure translation by vectors of shape (..., 3).</summary>
    public static RigidTransform FromTranslation(NdArray translation)
    {
        var t = Trailing(translation, [3], nameof(translation), out var lead);
        var n = t.Length / 3;
        var m = new double[16 * n];
        for (var i = 0; i < n; i++)
        {
            var o = 16 * i;
            m[o] = m[o + 5] = m[o + 10] = m[o + 15] = 1;
            m[o + 3] = t[3 * i]; m[o + 7] = t[3 * i + 1]; m[o + 11] = t[3 * i + 2];
        }
        return new RigidTransform(m, lead);
    }

    /// <summary>A pure translation by one vector.</summary>
    public static RigidTransform FromTranslation(double[] translation)
    {
        ArgumentNullException.ThrowIfNull(translation);
        return FromTranslation(new NdArray((double[])translation.Clone(), translation.Length));
    }

    /// <summary>Rotation followed by translation: <c>FromTranslation(t) * FromRotation(r)</c>.</summary>
    public static RigidTransform FromComponents(NdArray translation, Rotation rotation)
        => FromTranslation(translation) * FromRotation(rotation);

    /// <summary>Rotation followed by translation by one vector.</summary>
    public static RigidTransform FromComponents(double[] translation, Rotation rotation)
        => FromTranslation(translation) * FromRotation(rotation);

    /// <summary>
    /// From exponential coordinates (..., 6): a rotation vector followed by the translational part
    /// of the twist, mapped through the SE(3) exponential.
    /// </summary>
    public static RigidTransform FromExpCoords(NdArray expCoords)
    {
        var e = Trailing(expCoords, [6], nameof(expCoords), out var lead);
        var n = e.Length / 6;
        var rv = new double[3 * n];
        for (var i = 0; i < n; i++) for (var k = 0; k < 3; k++) rv[3 * i + k] = e[6 * i + k];
        var rm = Rotation.FromRotvec(new NdArray(rv, n, 3)).AsMatrix().AsContiguous().ToArray();
        var m = new double[16 * n];
        Span<double> v = stackalloc double[9];
        for (var i = 0; i < n; i++)
        {
            ExpTranslationTransform(rv[3 * i], rv[3 * i + 1], rv[3 * i + 2], v);
            var o = 16 * i;
            for (var a = 0; a < 3; a++)
            {
                for (var b = 0; b < 3; b++) m[o + 4 * a + b] = rm[9 * i + 3 * a + b];
                m[o + 4 * a + 3] = v[3 * a] * e[6 * i + 3] + v[3 * a + 1] * e[6 * i + 4] + v[3 * a + 2] * e[6 * i + 5];
            }
            m[o + 15] = 1;
        }
        return new RigidTransform(m, lead);
    }

    /// <summary>From one set of exponential coordinates.</summary>
    public static RigidTransform FromExpCoords(double[] expCoords)
    {
        ArgumentNullException.ThrowIfNull(expCoords);
        return FromExpCoords(new NdArray((double[])expCoords.Clone(), expCoords.Length));
    }

    /// <summary>
    /// From unit dual quaternions (..., 8): real part (a rotation quaternion) then dual part, each
    /// scalar-last unless <paramref name="scalarFirst"/>. The input is normalised first.
    /// </summary>
    public static RigidTransform FromDualQuat(NdArray dualQuat, bool scalarFirst = false)
    {
        var d = Trailing(dualQuat, [8], nameof(dualQuat), out var lead);
        var n = d.Length / 8;
        var m = new double[16 * n];
        Span<double> real = stackalloc double[4];
        Span<double> dual = stackalloc double[4];
        Span<double> inv = stackalloc double[4];
        Span<double> t = stackalloc double[4];
        Span<double> rm = stackalloc double[9];
        for (var i = 0; i < n; i++)
        {
            for (var k = 0; k < 4; k++) { real[k] = d[8 * i + k]; dual[k] = d[8 * i + 4 + k]; }
            if (scalarFirst) { Roll(real, -1); Roll(dual, -1); }
            NormaliseDualQuaternion(real, dual);
            // from_quat normalises again (a no-op up to rounding), then t = 2 (dual ⊗ real⁻¹).
            var rn = Math.Sqrt(real[0] * real[0] + real[1] * real[1] + real[2] * real[2] + real[3] * real[3]);
            for (var k = 0; k < 4; k++) real[k] /= rn;
            inv[0] = -real[0]; inv[1] = -real[1]; inv[2] = -real[2]; inv[3] = real[3];
            ComposeQuat(dual, inv, t);
            QuatToMatrix(real, rm);
            var o = 16 * i;
            for (var a = 0; a < 3; a++)
            {
                for (var b = 0; b < 3; b++) m[o + 4 * a + b] = rm[3 * a + b];
                m[o + 4 * a + 3] = 2.0 * t[a];
            }
            m[o + 15] = 1;
        }
        return new RigidTransform(m, lead);
    }

    /// <summary>From one dual quaternion.</summary>
    public static RigidTransform FromDualQuat(double[] dualQuat, bool scalarFirst = false)
    {
        ArgumentNullException.ThrowIfNull(dualQuat);
        return FromDualQuat(new NdArray((double[])dualQuat.Clone(), dualQuat.Length), scalarFirst);
    }

    /// <summary>
    /// scipy.spatial.transform.normalize_dual_quaternion: scales the real part to unit norm (a zero
    /// real part becomes the identity) and removes from the dual part its component along the real part.
    /// </summary>
    public static NdArray NormalizeDualQuaternion(NdArray dualQuat)
    {
        var d = Trailing(dualQuat, [8], nameof(dualQuat), out var lead);
        var n = d.Length / 8;
        Span<double> real = stackalloc double[4];
        Span<double> dual = stackalloc double[4];
        for (var i = 0; i < n; i++)
        {
            for (var k = 0; k < 4; k++) { real[k] = d[8 * i + k]; dual[k] = d[8 * i + 4 + k]; }
            NormaliseDualQuaternion(real, dual);
            for (var k = 0; k < 4; k++) { d[8 * i + k] = real[k]; d[8 * i + 4 + k] = dual[k]; }
        }
        return Shaped(d, lead, 8);
    }

    private static void NormaliseDualQuaternion(Span<double> real, Span<double> dual)
    {
        var norm = Math.Sqrt(real[0] * real[0] + real[1] * real[1] + real[2] * real[2] + real[3] * real[3]);
        if (norm == 0) { real.Clear(); real[3] = 1; norm = 1; }
        for (var k = 0; k < 4; k++) { real[k] /= norm; dual[k] /= norm; }
        var dot = real[0] * dual[0] + real[1] * dual[1] + real[2] * dual[2] + real[3] * dual[3];
        for (var k = 0; k < 4; k++) dual[k] -= dot * real[k];
    }

    private static void Roll(Span<double> q, int shift)
    {
        if (shift == -1) { var x = q[0]; q[0] = q[1]; q[1] = q[2]; q[2] = q[3]; q[3] = x; }
        else { var x = q[3]; q[3] = q[2]; q[2] = q[1]; q[1] = q[0]; q[0] = x; }
    }

    /// <summary>The identity transform, or <paramref name="num"/> of them.</summary>
    public static RigidTransform Identity(int? num = null) => Identity(num is { } n ? [n] : []);

    /// <summary>An array of identity transforms of the given shape.</summary>
    public static RigidTransform Identity(int[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        var n = Rotation.Product(shape);
        var m = new double[16 * n];
        for (var i = 0; i < n; i++) m[16 * i] = m[16 * i + 5] = m[16 * i + 10] = m[16 * i + 15] = 1;
        return new RigidTransform(m, (int[])shape.Clone());
    }

    /// <summary>Joins transforms along the first axis (a single transform counts as an array of one).</summary>
    public static RigidTransform Concatenate(IEnumerable<RigidTransform> transforms)
    {
        ArgumentNullException.ThrowIfNull(transforms);
        var all = new List<double>();
        int[]? trailing = null;
        var length = 0;
        foreach (var t in transforms)
        {
            var shape = t.Single ? [1] : t._shape;
            if (trailing is null) trailing = shape[1..];
            else if (!shape[1..].SequenceEqual(trailing))
                throw new ArgumentException("All transforms must have the same shape after the first axis to be concatenated.");
            length += shape[0];
            all.AddRange(t._m);
        }
        return new RigidTransform([.. all], [length, .. trailing ?? []]);
    }

    // ================================================================== representations

    /// <summary>The 4×4 matrices, shape (..., 4, 4).</summary>
    public NdArray AsMatrix() => Shaped((double[])_m.Clone(), _shape, 4, 4);

    /// <summary>The translation vectors, shape (..., 3).</summary>
    public NdArray Translation
    {
        get
        {
            var t = new double[3 * Count];
            for (var i = 0; i < Count; i++) { t[3 * i] = _m[16 * i + 3]; t[3 * i + 1] = _m[16 * i + 7]; t[3 * i + 2] = _m[16 * i + 11]; }
            return Shaped(t, _shape, 3);
        }
    }

    private double[] RotationBlocks()
    {
        var r = new double[9 * Count];
        for (var i = 0; i < Count; i++)
            for (var a = 0; a < 3; a++)
                for (var b = 0; b < 3; b++) r[9 * i + 3 * a + b] = _m[16 * i + 4 * a + b];
        return r;
    }

    /// <summary>The rotations, with this transform's shape.</summary>
    public Rotation Rotation => Rotation.FromMatrix(Shaped(RotationBlocks(), _shape, 3, 3));

    /// <summary>Translation and rotation (scipy's <c>as_components</c>).</summary>
    public (NdArray Translation, Rotation Rotation) AsComponents() => (Translation, Rotation);

    private double[] OrthogonalQuats() => Rotation.FromMatrix(Shaped(RotationBlocks(), [Count], 3, 3), assumeValid: true).RawQuats;

    /// <summary>Exponential coordinates (..., 6): rotation vector, then the translational part of the twist.</summary>
    public NdArray AsExpCoords()
    {
        var q = OrthogonalQuats();
        var rv = Rotation.FromRawQuats(q, [Count]).AsRotvec().AsContiguous().ToArray();
        var e = new double[6 * Count];
        Span<double> v = stackalloc double[9];
        for (var i = 0; i < Count; i++)
        {
            LogTranslationTransform(rv[3 * i], rv[3 * i + 1], rv[3 * i + 2], v);
            double tx = _m[16 * i + 3], ty = _m[16 * i + 7], tz = _m[16 * i + 11];
            for (var k = 0; k < 3; k++)
            {
                e[6 * i + k] = rv[3 * i + k];
                e[6 * i + 3 + k] = v[3 * k] * tx + v[3 * k + 1] * ty + v[3 * k + 2] * tz;
            }
        }
        return Shaped(e, _shape, 6);
    }

    /// <summary>Unit dual quaternions (..., 8): the rotation quaternion, then ½ (t, 0) ⊗ q.</summary>
    public NdArray AsDualQuat(bool scalarFirst = false)
    {
        var q = OrthogonalQuats();
        var d = new double[8 * Count];
        Span<double> pure = stackalloc double[4];
        Span<double> dual = stackalloc double[4];
        Span<double> real = stackalloc double[4];
        for (var i = 0; i < Count; i++)
        {
            for (var k = 0; k < 4; k++) real[k] = q[4 * i + k];
            pure[0] = _m[16 * i + 3]; pure[1] = _m[16 * i + 7]; pure[2] = _m[16 * i + 11]; pure[3] = 0;
            ComposeQuat(pure, real, dual);
            for (var k = 0; k < 4; k++) dual[k] *= 0.5;
            if (scalarFirst) { Roll(real, 1); Roll(dual, 1); }
            for (var k = 0; k < 4; k++) { d[8 * i + k] = real[k]; d[8 * i + 4 + k] = dual[k]; }
        }
        return Shaped(d, _shape, 8);
    }

    // ================================================================== algebra

    /// <summary>Composition: <c>(a * b).Apply(p) == a.Apply(b.Apply(p))</c>, broadcasting the shapes.
    /// The product's rotation blocks are re-orthonormalised, as scipy does.</summary>
    public static RigidTransform operator *(RigidTransform a, RigidTransform b)
    {
        ArgumentNullException.ThrowIfNull(a); ArgumentNullException.ThrowIfNull(b);
        var shape = Rotation.Broadcast(a._shape, b._shape, "transforms");
        var ai = Rotation.BroadcastIndex(a._shape, shape);
        var bi = Rotation.BroadcastIndex(b._shape, shape);
        var m = new double[16 * ai.Length];
        for (var i = 0; i < ai.Length; i++) MatMul4(a._m.AsSpan(16 * ai[i], 16), b._m.AsSpan(16 * bi[i], 16), m.AsSpan(16 * i, 16));
        NormaliseRotations(m);
        return new RigidTransform(m, shape);
    }

    private static void MatMul4(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<double> r)
    {
        for (var i = 0; i < 4; i++)
            for (var j = 0; j < 4; j++)
                r[4 * i + j] = x[4 * i] * y[j] + x[4 * i + 1] * y[4 + j] + x[4 * i + 2] * y[8 + j] + x[4 * i + 3] * y[12 + j];
    }

    /// <summary>The inverse transforms: [Rᵀ, −Rᵀ t].</summary>
    public RigidTransform Inv() => new(InvMatrices(_m), _shape);

    private static double[] InvMatrices(double[] src)
    {
        var m = new double[src.Length];
        for (var i = 0; i < src.Length / 16; i++)
        {
            var o = 16 * i;
            for (var a = 0; a < 3; a++)
                for (var b = 0; b < 3; b++) m[o + 4 * a + b] = src[o + 4 * b + a];
            for (var a = 0; a < 3; a++)
                m[o + 4 * a + 3] = -(m[o + 4 * a] * src[o + 3] + m[o + 4 * a + 1] * src[o + 7] + m[o + 4 * a + 2] * src[o + 11]);
            m[o + 15] = 1;
        }
        return m;
    }

    /// <summary>The transform applied <paramref name="n"/> times (fractional allowed), through exponential coordinates.</summary>
    public RigidTransform Pow(double n)
    {
        if (n == 0) return Identity(_shape);
        if (n == -1) return Inv();
        if (n == 1) return new RigidTransform((double[])_m.Clone(), _shape);
        var e = AsExpCoords().AsContiguous().ToArray();
        for (var i = 0; i < e.Length; i++) e[i] *= n;
        return FromExpCoords(Shaped(e, _shape, 6));
    }

    /// <summary>
    /// Transforms points of shape (..., 3): R p + t (or the inverse). The transform shape and the
    /// points' leading shape broadcast; a single transform and a single point give shape (3).
    /// </summary>
    public NdArray Apply(NdArray vectors, bool inverse = false)
    {
        var v = Trailing(vectors, [3], nameof(vectors), out var lead);
        var m = inverse ? InvMatrices(_m) : _m;
        var shape = Rotation.Broadcast(_shape, lead, "transforms and vectors");
        var ti = Rotation.BroadcastIndex(_shape, shape);
        var vi = Rotation.BroadcastIndex(lead, shape);
        var r = new double[3 * ti.Length];
        for (var i = 0; i < ti.Length; i++)
        {
            var o = 16 * ti[i];
            double x = v[3 * vi[i]], y = v[3 * vi[i] + 1], z = v[3 * vi[i] + 2];
            for (var k = 0; k < 3; k++) r[3 * i + k] = m[o + 4 * k] * x + m[o + 4 * k + 1] * y + m[o + 4 * k + 2] * z + m[o + 4 * k + 3];
        }
        return Shaped(r, shape, 3);
    }

    /// <summary>Transforms one point.</summary>
    public double[] Apply(double[] vector, bool inverse = false)
    {
        ArgumentNullException.ThrowIfNull(vector);
        return Apply(new NdArray((double[])vector.Clone(), vector.Length), inverse).AsContiguous().ToArray();
    }

    /// <summary>
    /// The mean transform: the rotation mean (Markley's quaternion average) and the arithmetic mean
    /// of the translations, both weighted, over <paramref name="axis"/> (null: all axes).
    /// </summary>
    public RigidTransform Mean(double[]? weights = null, int[]? axis = null)
    {
        if (Count == 0) throw new ArgumentException("Mean of an empty rotation set is undefined.");
        if (weights is not null)
        {
            if (weights.Length != Count) throw new ArgumentException("Expected `weights` to match transform shape.");
            if (weights.Any(w => w < 0)) throw new ArgumentException("`weights` must be non-negative.");
        }
        var rot = Rotation.FromRawQuats(OrthogonalQuats(), _shape).Mean(weights, axis);
        var rm = rot.AsMatrix().AsContiguous().ToArray();
        var (outShape, groups) = Rotation.ReductionGroups(_shape, axis);
        var m = new double[16 * groups.Length];
        for (var g = 0; g < groups.Length; g++)
        {
            double sx = 0, sy = 0, sz = 0, sw = 0;
            foreach (var i in groups[g])
            {
                var w = weights?[i] ?? 1.0;
                sx += w * _m[16 * i + 3]; sy += w * _m[16 * i + 7]; sz += w * _m[16 * i + 11]; sw += w;
            }
            var o = 16 * g;
            for (var a = 0; a < 3; a++)
                for (var b = 0; b < 3; b++) m[o + 4 * a + b] = rm[9 * g + 3 * a + b];
            m[o + 3] = sx / sw; m[o + 7] = sy / sw; m[o + 11] = sz / sw; m[o + 15] = 1;
        }
        return new RigidTransform(m, outShape);
    }

    // ================================================================== indexing

    private int BlockSize => Single ? 1 : Rotation.Product(_shape[1..]);

    private (int Start, int Block) Block(int index)
    {
        if (Single) throw new InvalidOperationException("Single transform is not subscriptable.");
        var len = _shape[0];
        if (index < 0) index += len;
        if (index < 0 || index >= len) throw new IndexOutOfRangeException($"Index {index} is out of range for {len} transforms.");
        var block = BlockSize;
        return (index * block, block);
    }

    /// <summary>The i-th element along the first axis; setting it overwrites those transforms.</summary>
    public RigidTransform this[int index]
    {
        get
        {
            var (start, block) = Block(index);
            return new RigidTransform(_m[(16 * start)..(16 * (start + block))], _shape[1..]);
        }
        set
        {
            ArgumentNullException.ThrowIfNull(value);
            var (start, block) = Block(index);
            if (!value._shape.SequenceEqual(_shape[1..])) throw new ArgumentException("The assigned transform must have the shape of one element of the first axis.");
            Array.Copy(value._m, 0, _m, 16 * start, 16 * block);
        }
    }

    /// <summary>A slice along the first axis.</summary>
    public RigidTransform this[Range range]
    {
        get
        {
            if (Single) throw new InvalidOperationException("Single transform is not subscriptable.");
            var (off, len) = range.GetOffsetAndLength(_shape[0]);
            var block = BlockSize;
            return new RigidTransform(_m[(16 * off * block)..(16 * (off + len) * block)], [len, .. _shape[1..]]);
        }
    }

    /// <summary>The elements at the given indices along the first axis.</summary>
    public RigidTransform Take(int[] indices)
    {
        ArgumentNullException.ThrowIfNull(indices);
        if (Single) throw new InvalidOperationException("Single transform is not subscriptable.");
        var block = BlockSize;
        var m = new double[16 * block * indices.Length];
        for (var i = 0; i < indices.Length; i++)
        {
            var (start, _) = Block(indices[i]);
            Array.Copy(_m, 16 * start, m, 16 * block * i, 16 * block);
        }
        return new RigidTransform(m, [indices.Length, .. _shape[1..]]);
    }

    // ================================================================== helpers

    private static void ComposeQuat(ReadOnlySpan<double> p, ReadOnlySpan<double> q, Span<double> r)
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

    private static void QuatToMatrix(ReadOnlySpan<double> q, Span<double> m)
    {
        double x = q[0], y = q[1], z = q[2], w = q[3];
        double x2 = x * x, y2 = y * y, z2 = z * z, w2 = w * w;
        double xy = x * y, zw = z * w, xz = x * z, yw = y * w, yz = y * z, xw = x * w;
        m[0] = x2 - y2 - z2 + w2; m[1] = 2 * (xy - zw); m[2] = 2 * (xz + yw);
        m[3] = 2 * (xy + zw); m[4] = -x2 + y2 - z2 + w2; m[5] = 2 * (yz - xw);
        m[6] = 2 * (xz - yw); m[7] = 2 * (yz + xw); m[8] = -x2 - y2 + z2 + w2;
    }

    private static void Skew(double x, double y, double z, Span<double> s)
    {
        s[0] = 0; s[1] = -z; s[2] = y;
        s[3] = z; s[4] = 0; s[5] = -x;
        s[6] = -y; s[7] = x; s[8] = 0;
    }

    /// <summary>V = I + k1 [ω]× + k2 [ω]×², the left Jacobian of SO(3) (SE(3) exponential).</summary>
    private static void ExpTranslationTransform(double x, double y, double z, Span<double> v)
    {
        var angle = Math.Sqrt(x * x + y * y + z * z);
        double k1, k2;
        if (angle < 1e-3)
        {
            var a2 = angle * angle;
            k1 = 0.5 - a2 / 24 + a2 * a2 / 720;
            k2 = 1.0 / 6 - a2 / 120 + a2 * a2 / 5040;
        }
        else
        {
            k1 = (1.0 - Math.Cos(angle)) / (angle * angle);
            k2 = (angle - Math.Sin(angle)) / (angle * angle * angle);
        }
        Combine(x, y, z, 1.0, k1, k2, v);
    }

    /// <summary>V⁻¹ = I − ½ [ω]× + k [ω]×² (SE(3) logarithm).</summary>
    private static void LogTranslationTransform(double x, double y, double z, Span<double> v)
    {
        var angle = Math.Sqrt(x * x + y * y + z * z);
        double k;
        if (angle < 1e-3)
        {
            var a2 = angle * angle;
            k = 1.0 / 12 + a2 / 720 + a2 * a2 / 30240;
        }
        else k = (1 - 0.5 * angle / Math.Tan(0.5 * angle)) / (angle * angle);
        Combine(x, y, z, 1.0, -0.5, k, v);
    }

    private static void Combine(double x, double y, double z, double c0, double c1, double c2, Span<double> v)
    {
        Span<double> s = stackalloc double[9];
        Skew(x, y, z, s);
        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 3; j++)
            {
                var s2 = s[3 * i] * s[j] + s[3 * i + 1] * s[3 + j] + s[3 * i + 2] * s[6 + j];
                v[3 * i + j] = (i == j ? c0 : 0.0) + c1 * s[3 * i + j] + c2 * s2;
            }
    }

    /// <inheritdoc />
    public override string ToString() => Single ? "RigidTransform.from_matrix(" + string.Join(", ", _m.Select(x => x.ToString("G6"))) + ")" : $"RigidTransform[{string.Join(", ", _shape)}]";
}
