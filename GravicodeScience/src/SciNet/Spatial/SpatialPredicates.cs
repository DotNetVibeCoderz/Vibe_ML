namespace Gravicode.Science.SciNet.Spatial;

/// <summary>
/// Robust geometric predicates: orientation and in-circle tests whose <em>sign</em> is exact.
/// </summary>
/// <remarks>
/// <para>Each predicate first evaluates the determinant in plain floating point together with
/// Shewchuk's forward error bound ("Adaptive Precision Floating-Point Arithmetic and Fast Robust
/// Geometric Predicates", 1997). When the result is larger than the bound its sign is certainly
/// right, which is the overwhelming majority of calls on random data. Otherwise the determinant
/// is recomputed exactly with floating-point expansions (sums of non-overlapping doubles).</para>
/// <para>The exact stage here is the simple one: every coordinate difference is turned into a
/// two-term expansion and the products are expanded in full, rather than Shewchuk's staged B/C
/// refinements. It allocates and is perhaps 10-50x slower than his code on a degenerate call,
/// but it is short enough to check by eye, and degenerate calls only dominate on inputs such as
/// lattices, where correctness is the point. Products use <see cref="Math.FusedMultiplyAdd"/>
/// for the exact low word, so the stage is exact barring underflow below ~1e-300.</para>
/// <para>Why it matters: with a naive in-circle test, four cocircular grid points produce
/// inconsistent answers depending on the order of the arguments, and incremental Delaunay
/// construction then corrupts its own mesh (crossing edges, or a cavity that is not
/// star-shaped).</para>
/// </remarks>
internal static partial class SpatialPredicates
{
    private const double Epsilon = 1.1102230246251565e-16; // 2^-53
    private const double CcwErrBoundA = (3.0 + 16.0 * Epsilon) * Epsilon;
    private const double IccErrBoundA = (10.0 + 96.0 * Epsilon) * Epsilon;
    private const double O3dErrBoundA = (7.0 + 56.0 * Epsilon) * Epsilon;

    /// <summary>
    /// Positive when a, b, c are in counter-clockwise order, negative when clockwise, zero when
    /// collinear. The magnitude is approximately twice the signed triangle area.
    /// </summary>
    public static double Orient2D(double ax, double ay, double bx, double by, double cx, double cy)
    {
        var detLeft = (ax - cx) * (by - cy);
        var detRight = (ay - cy) * (bx - cx);
        var det = detLeft - detRight;
        double detSum;
        if (detLeft > 0)
        {
            if (detRight <= 0) return det;
            detSum = detLeft + detRight;
        }
        else if (detLeft < 0)
        {
            if (detRight >= 0) return det;
            detSum = -detLeft - detRight;
        }
        else return det;
        var bound = CcwErrBoundA * detSum;
        if (det >= bound || -det >= bound) return det;
        return Orient2DExact(ax, ay, bx, by, cx, cy);
    }

    /// <summary>
    /// Positive when d lies strictly inside the circle through a, b, c (given counter-clockwise),
    /// negative outside, zero when the four points are cocircular.
    /// </summary>
    public static double InCircle(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy)
    {
        var adx = ax - dx; var bdx = bx - dx; var cdx = cx - dx;
        var ady = ay - dy; var bdy = by - dy; var cdy = cy - dy;
        var bdxcdy = bdx * cdy; var cdxbdy = cdx * bdy; var alift = adx * adx + ady * ady;
        var cdxady = cdx * ady; var adxcdy = adx * cdy; var blift = bdx * bdx + bdy * bdy;
        var adxbdy = adx * bdy; var bdxady = bdx * ady; var clift = cdx * cdx + cdy * cdy;
        var det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady);
        var permanent = (Math.Abs(bdxcdy) + Math.Abs(cdxbdy)) * alift
                        + (Math.Abs(cdxady) + Math.Abs(adxcdy)) * blift
                        + (Math.Abs(adxbdy) + Math.Abs(bdxady)) * clift;
        var bound = IccErrBoundA * permanent;
        if (det > bound || -det > bound) return det;
        return InCircleExact(ax, ay, bx, by, cx, cy, dx, dy);
    }

    /// <summary>
    /// The determinant <c>det[a-d; b-d; c-d]</c>: negative when d lies on the side of the plane
    /// through a, b, c that the right-handed normal <c>(b-a)x(c-a)</c> points to, positive on the
    /// other side, zero when coplanar.
    /// </summary>
    public static double Orient3D(ReadOnlySpan<double> a, ReadOnlySpan<double> b, ReadOnlySpan<double> c, ReadOnlySpan<double> d)
    {
        double adx = a[0] - d[0], bdx = b[0] - d[0], cdx = c[0] - d[0];
        double ady = a[1] - d[1], bdy = b[1] - d[1], cdy = c[1] - d[1];
        double adz = a[2] - d[2], bdz = b[2] - d[2], cdz = c[2] - d[2];
        double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
        double cdxady = cdx * ady, adxcdy = adx * cdy;
        double adxbdy = adx * bdy, bdxady = bdx * ady;
        var det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);
        var permanent = (Math.Abs(bdxcdy) + Math.Abs(cdxbdy)) * Math.Abs(adz)
                        + (Math.Abs(cdxady) + Math.Abs(adxcdy)) * Math.Abs(bdz)
                        + (Math.Abs(adxbdy) + Math.Abs(bdxady)) * Math.Abs(cdz);
        var bound = O3dErrBoundA * permanent;
        if (det > bound || -det > bound) return det;
        return Orient3DExact(a, b, c, d);
    }

    // ---------------------------------------------------------------- exact stage

    internal static double Orient2DExact(double ax, double ay, double bx, double by, double cx, double cy)
    {
        var acx = Diff(ax, cx); var bcy = Diff(by, cy);
        var acy = Diff(ay, cy); var bcx = Diff(bx, cx);
        return Top(Sum(Mul(acx, bcy), Negate(Mul(acy, bcx))));
    }

    internal static double InCircleExact(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy)
    {
        var adx = Diff(ax, dx); var bdx = Diff(bx, dx); var cdx = Diff(cx, dx);
        var ady = Diff(ay, dy); var bdy = Diff(by, dy); var cdy = Diff(cy, dy);
        var bc = Sum(Mul(bdx, cdy), Negate(Mul(cdx, bdy)));
        var ca = Sum(Mul(cdx, ady), Negate(Mul(adx, cdy)));
        var ab = Sum(Mul(adx, bdy), Negate(Mul(bdx, ady)));
        var alift = Sum(Mul(adx, adx), Mul(ady, ady));
        var blift = Sum(Mul(bdx, bdx), Mul(bdy, bdy));
        var clift = Sum(Mul(cdx, cdx), Mul(cdy, cdy));
        return Top(Sum(Sum(Mul(alift, bc), Mul(blift, ca)), Mul(clift, ab)));
    }

    internal static double Orient3DExact(ReadOnlySpan<double> a, ReadOnlySpan<double> b, ReadOnlySpan<double> c, ReadOnlySpan<double> d)
    {
        var adx = Diff(a[0], d[0]); var bdx = Diff(b[0], d[0]); var cdx = Diff(c[0], d[0]);
        var ady = Diff(a[1], d[1]); var bdy = Diff(b[1], d[1]); var cdy = Diff(c[1], d[1]);
        var adz = Diff(a[2], d[2]); var bdz = Diff(b[2], d[2]); var cdz = Diff(c[2], d[2]);
        var bc = Sum(Mul(bdx, cdy), Negate(Mul(cdx, bdy)));
        var ca = Sum(Mul(cdx, ady), Negate(Mul(adx, cdy)));
        var ab = Sum(Mul(adx, bdy), Negate(Mul(bdx, ady)));
        return Top(Sum(Sum(Mul(adz, bc), Mul(bdz, ca)), Mul(cdz, ab)));
    }

    /// <summary>The most significant component; its sign is the sign of the expansion.</summary>
    private static double Top(double[] e) => e.Length == 0 ? 0.0 : e[^1];

    private static void TwoSum(double a, double b, out double x, out double y)
    {
        x = a + b;
        var bv = x - a;
        var av = x - bv;
        y = (a - av) + (b - bv);
    }

    private static double[] Diff(double a, double b)
    {
        var x = a - b;
        var bv = a - x;
        var av = x + bv;
        var y = (a - av) + (bv - b);
        if (y == 0) return x == 0 ? [] : [x];
        return [y, x];
    }

    private static double[] Negate(double[] e)
    {
        var r = new double[e.Length];
        for (var i = 0; i < e.Length; i++) r[i] = -e[i];
        return r;
    }

    /// <summary>Shewchuk's GROW-EXPANSION with zero elimination: e + b, exactly.</summary>
    private static double[] Grow(double[] e, double b)
    {
        var h = new double[e.Length + 1];
        var count = 0;
        var q = b;
        foreach (var ei in e)
        {
            TwoSum(q, ei, out var sum, out var err);
            if (err != 0) h[count++] = err;
            q = sum;
        }
        if (q != 0 || count == 0) h[count++] = q;
        if (count == 1 && h[0] == 0) return [];
        Array.Resize(ref h, count);
        return h;
    }

    private static double[] Sum(double[] e, double[] f)
    {
        var h = e;
        foreach (var fi in f) h = Grow(h, fi);
        return h;
    }

    /// <summary>Shewchuk's SCALE-EXPANSION with zero elimination: e * b, exactly.</summary>
    private static double[] Scale(double[] e, double b)
    {
        if (e.Length == 0 || b == 0) return [];
        var h = new double[2 * e.Length];
        var count = 0;
        var q = e[0] * b;
        var lo = Math.FusedMultiplyAdd(e[0], b, -q);
        if (lo != 0) h[count++] = lo;
        for (var i = 1; i < e.Length; i++)
        {
            var p1 = e[i] * b;
            var p0 = Math.FusedMultiplyAdd(e[i], b, -p1);
            TwoSum(q, p0, out var sum, out var err);
            if (err != 0) h[count++] = err;
            // Fast-Two-Sum is valid here: |p1| >= |sum| (Shewchuk 1997, Theorem 19).
            var x = p1 + sum;
            var bv = x - p1;
            var y = sum - bv;
            if (y != 0) h[count++] = y;
            q = x;
        }
        if (q != 0 || count == 0) h[count++] = q;
        if (count == 1 && h[0] == 0) return [];
        Array.Resize(ref h, count);
        return h;
    }

    private static double[] Mul(double[] e, double[] f)
    {
        double[] result = [];
        foreach (var fi in f) result = Sum(result, Scale(e, fi));
        return result;
    }
}
