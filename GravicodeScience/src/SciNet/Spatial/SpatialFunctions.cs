using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Spatial;

/// <summary>Procrustes analysis and spherical interpolation (scipy.spatial.procrustes, scipy.linalg.orthogonal_procrustes, scipy.spatial.geometric_slerp).</summary>
public static class SpatialFunctions
{
    /// <summary>
    /// Procrustes analysis of two configurations of the same shape (n points × d): both are
    /// centred and scaled to unit Frobenius norm, then <paramref name="data2"/> is rotated and
    /// scaled onto <paramref name="data1"/>.
    /// </summary>
    /// <returns>The standardised data1, the transformed data2, and the disparity
    /// sum((mtx1 - mtx2)^2), which is symmetric in the two inputs.</returns>
    public static (NdArray Mtx1, NdArray Mtx2, double Disparity) Procrustes(NdArray data1, NdArray data2)
    {
        ArgumentNullException.ThrowIfNull(data1); ArgumentNullException.ThrowIfNull(data2);
        if (data1.Rank != 2 || data2.Rank != 2) throw new ArgumentException("Input matrices must be two-dimensional.");
        if (data1.Shape[0] != data2.Shape[0] || data1.Shape[1] != data2.Shape[1]) throw new ArgumentException("Input matrices must be of same shape.");
        if (data1.Size == 0) throw new ArgumentException("Input matrices must be >0 rows and >0 cols.");
        int n = data1.Shape[0], d = data1.Shape[1];
        var a = Standardise(data1.AsContiguous().ToArray(), n, d);
        var b = Standardise(data2.AsContiguous().ToArray(), n, d);
        var (r, s) = OrthogonalProcrustes(new NdArray(a, n, d), new NdArray(b, n, d));
        var rr = r.AsContiguous().ToArray();
        var b2 = new double[n * d];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < d; j++)
            {
                var t = 0.0;
                for (var k = 0; k < d; k++) t += b[i * d + k] * rr[j * d + k]; // mtx2 @ R.T
                b2[i * d + j] = t * s;
            }
        var disparity = 0.0;
        for (var i = 0; i < a.Length; i++) disparity += (a[i] - b2[i]) * (a[i] - b2[i]);
        return (new NdArray(a, n, d), new NdArray(b2, n, d), disparity);
    }

    private static double[] Standardise(double[] m, int n, int d)
    {
        for (var j = 0; j < d; j++)
        {
            var mean = 0.0;
            for (var i = 0; i < n; i++) mean += m[i * d + j];
            mean /= n;
            for (var i = 0; i < n; i++) m[i * d + j] -= mean;
        }
        var norm = 0.0;
        foreach (var v in m) norm += v * v;
        norm = Math.Sqrt(norm);
        if (norm == 0) throw new ArgumentException("Input matrices must contain >1 unique points.");
        for (var i = 0; i < m.Length; i++) m[i] /= norm;
        return m;
    }

    /// <summary>
    /// The orthogonal matrix R minimising ‖A R - B‖_F (scipy.linalg.orthogonal_procrustes), and
    /// the scale: the sum of the singular values of AᵀB.
    /// </summary>
    /// <remarks>R = U Vᵀ from the SVD of AᵀB (Schönemann 1966). R may be a reflection; use
    /// <see cref="Rotation.AlignVectors(NdArray, NdArray, double[])"/> for a proper rotation.</remarks>
    public static (NdArray R, double Scale) OrthogonalProcrustes(NdArray a, NdArray b)
    {
        ArgumentNullException.ThrowIfNull(a); ArgumentNullException.ThrowIfNull(b);
        if (a.Rank != 2) throw new ArgumentException($"expected ndim to be 2, but observed {a.Rank}");
        if (a.Shape[0] != b.Shape[0] || a.Shape[1] != b.Shape[1]) throw new ArgumentException("The shapes of A and B differ.");
        SpatialInput.RequireFinite(a.AsContiguous().AsSpan(), "A");
        SpatialInput.RequireFinite(b.AsContiguous().AsSpan(), "B");
        var m = LinAlg.Dot(a.T.Copy(), b);
        var svd = Decomposition.Svd(m);
        var r = LinAlg.Dot(svd.U, svd.V.T.Copy());
        var scale = 0.0;
        foreach (var s in svd.SingularValues.ToArray()) scale += s;
        return (r, scale);
    }

    /// <summary>
    /// Spherical linear interpolation between two unit vectors along the great circle
    /// (scipy.spatial.geometric_slerp); one row per value of <paramref name="t"/>.
    /// </summary>
    /// <remarks>Uses the orthonormal basis {start, normalised component of end orthogonal to
    /// start} and the angle atan2(|end⊥|, start·end), the same construction as scipy's QR route.
    /// Antipodal endpoints have no unique great circle; the path then depends on rounding, as
    /// in scipy (which only warns).</remarks>
    public static NdArray GeometricSlerp(double[] start, double[] end, double[] t)
    {
        ArgumentNullException.ThrowIfNull(start); ArgumentNullException.ThrowIfNull(end); ArgumentNullException.ThrowIfNull(t);
        if (start.Length != end.Length) throw new ArgumentException("The dimensions of start and end must match (have same size).");
        if (start.Length < 2) throw new ArgumentException("The start and end coordinates must both be in at least two-dimensional space.");
        var d = start.Length;
        var result = new double[t.Length * d];
        if (start.AsSpan().SequenceEqual(end))
        {
            for (var i = 0; i < t.Length; i++) start.CopyTo(result, i * d);
            return new NdArray(result, t.Length, d);
        }
        foreach (var v in new[] { start, end })
        {
            var nrm = Math.Sqrt(v.Sum(x => x * x));
            if (Math.Abs(nrm - 1.0) > 1e-9) throw new ArgumentException("start and end are not on a unit n-sphere.");
        }
        var c = 0.0;
        for (var k = 0; k < d; k++) c += start[k] * end[k];
        var q1 = new double[d];
        for (var k = 0; k < d; k++) q1[k] = end[k] - c * start[k];
        var s = Math.Sqrt(q1.Sum(x => x * x));
        if (s > 0) for (var k = 0; k < d; k++) q1[k] /= s;
        var omega = Math.Atan2(s, c);
        for (var i = 0; i < t.Length; i++)
        {
            double sn = Math.Sin(t[i] * omega), cs = Math.Cos(t[i] * omega);
            for (var k = 0; k < d; k++) result[i * d + k] = start[k] * cs + q1[k] * sn;
        }
        return new NdArray(result, t.Length, d);
    }
}
