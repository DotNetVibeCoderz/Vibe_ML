namespace Gravicode.Science.SciNet.Spatial;

/// <summary>
/// The d-dimensional orientation predicate behind the N-d hull and Delaunay code, with the same
/// two-stage design as the 2-D/3-D predicates: a floating-point determinant with a forward error
/// bound, and an exact expansion-arithmetic determinant when the bound cannot certify the sign.
/// </summary>
/// <remarks>
/// <para>Both stages expand the determinant by minors over column subsets (a dynamic programme
/// over bit masks, D·2^(D-1) products) rather than by elimination. Elimination has no clean
/// a-priori error bound; the minor expansion does, because the matching <em>permanent</em> (the
/// same expansion over absolute values) bounds every rounding error. The bound used,
/// <c>(2D(D+3)+8)·u·permanent</c>, is deliberately loose: a loose bound only sends more calls
/// to the exact stage, while a tight wrong one gives a wrong sign.</para>
/// <para>For the lifted (Delaunay) predicate the last column of row i is
/// <c>|p_i − q|² + e_i − e_q</c>, not <c>|p_i|² − |q|²</c>: the two differ by a linear
/// combination of the other columns, so the determinant is the same, but the first form is an
/// exact sum of squares of exact differences and does not cancel catastrophically away from the
/// origin. <c>e</c> is an "excess height", zero on the paraboloid; Qhull's <c>Qz</c> point above
/// the paraboloid is the only point with a non-zero one.</para>
/// </remarks>
internal static partial class SpatialPredicates
{
    /// <summary>Largest hull dimension the N-d predicate supports (2^D minors on the stack).</summary>
    public const int MaxNdDimension = 12;

    /// <summary>
    /// <c>det</c> of the D×D matrix whose row i is <c>P[rows[i]] − q</c> (with the lifted column
    /// when <paramref name="lifted"/>), D = rows.Length. The sign is exact.
    /// </summary>
    /// <param name="p">Point buffer, row-major with stride <paramref name="d"/>.</param>
    /// <param name="d">Coordinates per point; D = d (+1 when lifted).</param>
    /// <param name="lifted">Append the paraboloid column.</param>
    /// <param name="excess">Excess height per point (lifted only; null means all zero).</param>
    /// <param name="rows">The D point indices forming the matrix rows.</param>
    /// <param name="q">The query point's coordinates (d values).</param>
    /// <param name="qExcess">The query's excess height.</param>
    public static double OrientNd(double[] p, int d, bool lifted, double[]? excess, ReadOnlySpan<int> rows, ReadOnlySpan<double> q, double qExcess)
    {
        var D = rows.Length;
        Calls++;
        if (D == 4 || D == 3)
        {
            // Unrolled filters for the common cases (3-D hulls and walks, 3-D Delaunay, 4-D
            // hulls): the minor-expansion loop below costs ~200 ns a call, these ~20.
            Span<double> e = stackalloc double[16];
            Span<double> ae = stackalloc double[16];
            for (var i = 0; i < D; i++)
            {
                var o = rows[i] * d;
                double sq = 0;
                for (var j = 0; j < d; j++)
                {
                    var t = p[o + j] - q[j];
                    e[i * 4 + j] = t; ae[i * 4 + j] = Math.Abs(t);
                    sq += t * t;
                }
                if (lifted)
                {
                    var ex = (excess is null ? 0 : excess[rows[i]]) - qExcess;
                    e[i * 4 + d] = sq + ex; ae[i * 4 + d] = sq + Math.Abs(ex);
                }
            }
            double fdet, fper;
            if (D == 3)
            {
                double m0 = e[5] * e[10] - e[6] * e[9], m1 = e[4] * e[10] - e[6] * e[8], m2 = e[4] * e[9] - e[5] * e[8];
                double p0 = ae[5] * ae[10] + ae[6] * ae[9], p1 = ae[4] * ae[10] + ae[6] * ae[8], p2 = ae[4] * ae[9] + ae[5] * ae[8];
                fdet = e[0] * m0 - e[1] * m1 + e[2] * m2;
                fper = ae[0] * p0 + ae[1] * p1 + ae[2] * p2;
            }
            else
            {
                // Laplace expansion by complementary 2×2 minors of rows (0, 1) and (2, 3).
                double a01 = e[0] * e[5] - e[1] * e[4], a02 = e[0] * e[6] - e[2] * e[4], a03 = e[0] * e[7] - e[3] * e[4];
                double a12 = e[1] * e[6] - e[2] * e[5], a13 = e[1] * e[7] - e[3] * e[5], a23 = e[2] * e[7] - e[3] * e[6];
                double b01 = e[8] * e[13] - e[9] * e[12], b02 = e[8] * e[14] - e[10] * e[12], b03 = e[8] * e[15] - e[11] * e[12];
                double b12 = e[9] * e[14] - e[10] * e[13], b13 = e[9] * e[15] - e[11] * e[13], b23 = e[10] * e[15] - e[11] * e[14];
                fdet = a01 * b23 - a02 * b13 + a03 * b12 + a12 * b03 - a13 * b02 + a23 * b01;
                double c01 = ae[0] * ae[5] + ae[1] * ae[4], c02 = ae[0] * ae[6] + ae[2] * ae[4], c03 = ae[0] * ae[7] + ae[3] * ae[4];
                double c12 = ae[1] * ae[6] + ae[2] * ae[5], c13 = ae[1] * ae[7] + ae[3] * ae[5], c23 = ae[2] * ae[7] + ae[3] * ae[6];
                double d01 = ae[8] * ae[13] + ae[9] * ae[12], d02 = ae[8] * ae[14] + ae[10] * ae[12], d03 = ae[8] * ae[15] + ae[11] * ae[12];
                double d12 = ae[9] * ae[14] + ae[10] * ae[13], d13 = ae[9] * ae[15] + ae[11] * ae[13], d23 = ae[10] * ae[15] + ae[11] * ae[14];
                fper = c01 * d23 + c02 * d13 + c03 * d12 + c12 * d03 + c13 * d02 + c23 * d01;
            }
            var bnd = (2.0 * D * (D + 3) + 8) * Epsilon * fper;
            if (fdet > bnd || -fdet > bnd) return fdet;
            return OrientNdExact(p, d, lifted, excess, rows, q, qExcess);
        }
        Span<double> m = stackalloc double[D * D];
        Span<double> a = stackalloc double[D * D];
        for (var i = 0; i < D; i++)
        {
            var o = rows[i] * d;
            double s = 0, sa = 0;
            for (var j = 0; j < d; j++)
            {
                var t = p[o + j] - q[j];
                m[i * D + j] = t;
                a[i * D + j] = Math.Abs(t);
                s += t * t;
            }
            if (lifted)
            {
                var ex = (excess is null ? 0 : excess[rows[i]]) - qExcess;
                sa = s + Math.Abs(ex);
                s += ex;
                m[i * D + d] = s;
                a[i * D + d] = sa;
            }
        }
        var size = 1 << D;
        Span<double> det = size <= 1024 ? stackalloc double[size] : new double[size];
        Span<double> per = size <= 1024 ? stackalloc double[size] : new double[size];
        det[0] = 1; per[0] = 1;
        for (var mask = 1; mask < size; mask++)
        {
            var k = System.Numerics.BitOperations.PopCount((uint)mask) - 1; // row used
            double sd = 0, sp = 0;
            var c = 0;
            for (var bits = mask; bits != 0; bits &= bits - 1, c++)
            {
                var j = System.Numerics.BitOperations.TrailingZeroCount(bits);
                var sub = mask & ~(1 << j);
                var term = m[k * D + j] * det[sub];
                sd += ((k + c) & 1) == 0 ? term : -term;
                sp += a[k * D + j] * per[sub];
            }
            det[mask] = sd; per[mask] = sp;
        }
        var result = det[size - 1];
        var bound = (2.0 * D * (D + 3) + 8) * Epsilon * per[size - 1];
        if (result > bound || -result > bound) return result;
        return OrientNdExact(p, d, lifted, excess, rows, q, qExcess);
    }

    /// <summary>The exact stage of <see cref="OrientNd"/>: the same minor expansion over expansions.</summary>
    internal static double OrientNdExact(double[] p, int d, bool lifted, double[]? excess, ReadOnlySpan<int> rows, ReadOnlySpan<double> q, double qExcess)
    { ExactCalls++; return OrientNdExact0(p, d, lifted, excess, rows, q, qExcess); }
    internal static long ExactCalls, Calls;
    internal static double OrientNdExact0(double[] p, int d, bool lifted, double[]? excess, ReadOnlySpan<int> rows, ReadOnlySpan<double> q, double qExcess)
    {
        var D = rows.Length;
        Calls++;
        var m = new double[D * D][];
        for (var i = 0; i < D; i++)
        {
            var o = rows[i] * d;
            double[] s = [];
            for (var j = 0; j < d; j++)
            {
                var t = Diff(p[o + j], q[j]);
                m[i * D + j] = t;
                if (lifted) s = Sum(s, Mul(t, t));
            }
            if (lifted)
            {
                var ex = (excess is null ? 0 : excess[rows[i]]) - qExcess; // 0 or ±Δ, exact
                if (ex != 0) s = Grow(s, ex);
                m[i * D + d] = s;
            }
        }
        var size = 1 << D;
        var det = new double[size][];
        det[0] = [1.0];
        for (var mask = 1; mask < size; mask++)
        {
            var k = System.Numerics.BitOperations.PopCount((uint)mask) - 1;
            double[] acc = [];
            var c = 0;
            for (var bits = mask; bits != 0; bits &= bits - 1, c++)
            {
                var j = System.Numerics.BitOperations.TrailingZeroCount(bits);
                var sub = mask & ~(1 << j);
                if (det[sub].Length == 0 || m[k * D + j].Length == 0) continue;
                var term = Mul(m[k * D + j], det[sub]);
                acc = Sum(acc, ((k + c) & 1) == 0 ? term : Negate(term));
            }
            det[mask] = acc;
        }
        return Top(det[size - 1]);
    }
}
