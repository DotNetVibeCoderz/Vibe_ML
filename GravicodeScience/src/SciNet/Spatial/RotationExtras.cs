using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Spatial;

public sealed partial class Rotation
{
    /// <summary>The quaternions as a fresh (n, 4) buffer, scalar-last.</summary>
    internal double[] QuatBuffer => (double[])_q.Clone();

    /// <summary>
    /// Reduces each rotation p to <c>q = l · p · r</c> of smallest magnitude, l and r taken
    /// from <paramref name="left"/> and <paramref name="right"/> (scipy's <c>reduce</c>), e.g.
    /// symmetry groups from <see cref="CreateGroup"/>.
    /// </summary>
    /// <remarks>The smallest magnitude is the largest |scalar part| of the composed quaternion;
    /// ties go to the first (l, r) in left-major order. Indices are null for an omitted side,
    /// as in scipy.</remarks>
    public (Rotation Reduced, int[]? LeftIndices, int[]? RightIndices) Reduce(Rotation? left = null, Rotation? right = null)
    {
        if (left is null && right is null) return (new Rotation((double[])_q.Clone(), Single), null, null);
        var lq = left?._q ?? [0, 0, 0, 1];
        var rq = right?._q ?? [0, 0, 0, 1];
        int nl = lq.Length / 4, nr = rq.Length / 4;
        var outQ = new double[_q.Length];
        var li = new int[Count]; var ri = new int[Count];
        Span<double> lp = stackalloc double[4];
        Span<double> q = stackalloc double[4];
        for (var i = 0; i < Count; i++)
        {
            var p = _q.AsSpan(4 * i, 4);
            var best = -1.0;
            for (var a = 0; a < nl; a++)
            {
                Compose(lq.AsSpan(4 * a, 4), p, lp);
                for (var b = 0; b < nr; b++)
                {
                    var r = rq.AsSpan(4 * b, 4);
                    // Scalar part of lp · r only: no need for the full product here.
                    var w = Math.Abs(lp[3] * r[3] - lp[0] * r[0] - lp[1] * r[1] - lp[2] * r[2]);
                    if (w > best) { best = w; li[i] = a; ri[i] = b; }
                }
            }
            Compose(lq.AsSpan(4 * li[i], 4), p, lp);
            Compose(lp, rq.AsSpan(4 * ri[i], 4), q);
            q.CopyTo(outQ.AsSpan(4 * i, 4));
        }
        return (new Rotation(outQ, Single), left is null ? null : li, right is null ? null : ri);
    }

    /// <summary>
    /// A 3-D rotation group (scipy's <c>create_group</c>): "I" (icosahedral, 60), "O"
    /// (octahedral, 24), "T" (tetrahedral, 12), "Dn" (dicyclic, 2n) or "Cn" (cyclic, n), the
    /// latter two about <paramref name="axis"/>. Elements are in scipy's order.
    /// </summary>
    public static Rotation CreateGroup(string group, string axis = "Z")
    {
        ArgumentNullException.ThrowIfNull(group);
        ArgumentNullException.ThrowIfNull(axis);
        var ax = axis.ToLowerInvariant() switch
        {
            "x" => 0, "y" => 1, "z" => 2,
            _ => throw new ArgumentException("`axis` must be one of x, y, z, X, Y, Z"),
        };
        if (group is "I" or "O" or "T")
            return group switch { "I" => Icosahedral(), "O" => Octahedral(), _ => Tetrahedral() };
        if (group.Length >= 2 && group[0] is 'C' or 'D' && group[1..].All(char.IsAsciiDigit))
        {
            var order = int.Parse(group[1..], System.Globalization.CultureInfo.InvariantCulture);
            if (order < 1) throw new ArgumentException("Group order must be positive");
            return group[0] == 'C' ? Cyclic(order, ax) : Dicyclic(order, ax);
        }
        throw new ArgumentException("`group` must be one of 'I', 'O', 'T', 'Dn', 'Cn'");
    }

    private static double[] TetrahedralQuats()
    {
        const double c = 0.5;
        return
        [
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
            c, -c, -c, c, c, -c, c, c, c, c, -c, c, c, c, c, c,
            c, -c, -c, -c, c, -c, c, -c, c, c, -c, -c, c, c, c, -c,
        ];
    }

    private static Rotation Tetrahedral() => FromQuat(new NdArray(TetrahedralQuats(), 12, 4));

    private static Rotation Octahedral()
    {
        var c = Math.Sqrt(2) / 2;
        double[] g2 =
        [
            c, 0, 0, c, 0, c, 0, c, 0, 0, c, c, 0, 0, -c, c, 0, -c, 0, c, -c, 0, 0, c,
            0, c, c, 0, 0, -c, c, 0, c, 0, c, 0, -c, 0, c, 0, c, c, 0, 0, -c, c, 0, 0,
        ];
        return FromQuat(new NdArray([.. TetrahedralQuats(), .. g2], 24, 4));
    }

    private static Rotation Icosahedral()
    {
        var phi = (1 + Math.Sqrt(5)) / 2;
        double a = 0.5, b = 0.5 / phi, c = phi / 2;
        // scipy's table: (sign pattern) rows over the four placements of the zero.
        double[] g2 =
        [
            a, b, c, 0, a, b, -c, 0, a, c, 0, b, a, c, 0, -b, a, -b, c, 0, a, -b, -c, 0, a, -c, 0, b, a, -c, 0, -b,
            a, 0, b, c, a, 0, b, -c, a, 0, -b, c, a, 0, -b, -c,
            b, a, 0, c, b, a, 0, -c, b, c, a, 0, b, c, -a, 0, b, -a, 0, c, b, -a, 0, -c, b, -c, a, 0, b, -c, -a, 0,
            b, 0, c, a, b, 0, c, -a, b, 0, -c, a, b, 0, -c, -a,
            c, a, b, 0, c, a, -b, 0, c, b, 0, a, c, b, 0, -a, c, -a, b, 0, c, -a, -b, 0, c, -b, 0, a, c, -b, 0, -a,
            c, 0, a, b, c, 0, a, -b, c, 0, -a, b, c, 0, -a, -b,
            0, a, c, b, 0, a, c, -b, 0, a, -c, b, 0, a, -c, -b, 0, b, a, c, 0, b, a, -c, 0, b, -a, c, 0, b, -a, -c,
            0, c, b, a, 0, c, b, -a, 0, c, -b, a, 0, c, -b, -a,
        ];
        return FromQuat(new NdArray([.. TetrahedralQuats(), .. g2], 60, 4));
    }

    private static double[] CyclicRotvecs(int n, int axis)
    {
        var rv = new double[3 * n];
        for (var k = 0; k < n; k++) rv[3 * k + axis] = 2 * Math.PI / n * k; // np.linspace(0, 2π, n, endpoint=False): step × k
        return rv;
    }

    private static Rotation Cyclic(int n, int axis) => FromRotvec(new NdArray(CyclicRotvecs(n, axis), n, 3));

    private static Rotation Dicyclic(int n, int axis)
    {
        // scipy: g1 = cyclic(...).as_rotvec(); g2 = π·(0, cos θ, sin θ) rolled by axis.
        var g1 = Cyclic(n, axis).AsRotvec().AsContiguous().ToArray();
        var rv = new double[6 * n];
        Array.Copy(g1, rv, 3 * n);
        for (var k = 0; k < n; k++)
        {
            var theta = Math.PI / n * k;
            double[] v = [0, Math.PI * Math.Cos(theta), Math.PI * Math.Sin(theta)];
            for (var j = 0; j < 3; j++) rv[3 * (n + k) + (j + axis) % 3] = v[j];
        }
        return FromRotvec(new NdArray(rv, 2 * n, 3));
    }
}

/// <summary>
/// Interpolates rotations with continuous angular rate and acceleration
/// (scipy.spatial.transform.RotationSpline), a port of scipy's algorithm.
/// </summary>
/// <remarks>
/// <para>Between key times the rotation is <c>R_i · exp(v(t))</c> with v a cubic in the
/// rotation vector, as in Fakhrutdinov's construction scipy uses. The interior angular rates
/// make the angular acceleration continuous; the condition is nonlinear in them, so scipy (and
/// this port) iterates a linearised block-tridiagonal system (solved as a 5/5-banded system)
/// at most 10 times to a relative 1e-9. The end rates are the first/last segments' mean rates.
/// Evaluation outside the key times extrapolates the end cubics, as scipy's PPoly does.</para>
/// </remarks>
public sealed class RotationSpline
{
    private const int MaxIter = 10;
    private const double Tol = 1e-9;

    private readonly double[] _times;
    private readonly Rotation _rotations;
    private readonly double[] _coeff; // (segments, 4, 3): c0 t³ + c1 t² + c2 t + c3 per coordinate

    /// <summary>Key times (strictly increasing) and one rotation per time (at least two).</summary>
    public RotationSpline(double[] times, Rotation rotations)
    {
        ArgumentNullException.ThrowIfNull(times); ArgumentNullException.ThrowIfNull(rotations);
        if (rotations.Single) throw new ArgumentException("`rotations` must be a sequence of rotations.");
        if (rotations.Count == 1) throw new ArgumentException("`rotations` must contain at least 2 rotations.");
        if (times.Length != rotations.Count)
            throw new ArgumentException($"Expected number of rotations to be equal to number of timestamps given, got {rotations.Count} rotations and {times.Length} timestamps.");
        var n = times.Length;
        var m = n - 1;
        var dt = new double[m];
        for (var i = 0; i < m; i++)
        {
            dt[i] = times[i + 1] - times[i];
            if (!(dt[i] > 0)) throw new ArgumentException("Values in `times` must be in a strictly increasing order.");
        }
        _times = (double[])times.Clone();
        _rotations = rotations;
        var rotvecs = (rotations[..^1].Inv() * rotations[1..]).AsRotvec().AsContiguous().ToArray();
        var rates = new double[3 * m];
        for (var i = 0; i < m; i++) for (var j = 0; j < 3; j++) rates[3 * i + j] = rotvecs[3 * i + j] / dt[i];
        double[] rotvecsDot;
        if (n == 2) rotvecsDot = (double[])rates.Clone();
        else (rates, rotvecsDot) = SolveForAngularRates(dt, rates, rotvecs);

        _coeff = new double[m * 12];
        for (var i = 0; i < m; i++)
        {
            var h = dt[i];
            for (var j = 0; j < 3; j++)
            {
                double rv = rotvecs[3 * i + j], w = rates[3 * i + j], rd = rotvecsDot[3 * i + j];
                _coeff[12 * i + j] = (-2 * rv + h * w + h * rd) / (h * h * h);
                _coeff[12 * i + 3 + j] = (3 * rv - 2 * h * w - h * rd) / (h * h);
                _coeff[12 * i + 6 + j] = w;
                _coeff[12 * i + 9 + j] = 0;
            }
        }
    }

    /// <summary>The key times.</summary>
    public double[] Times => (double[])_times.Clone();

    private (double[] Rates, double[] RotvecsDot) SolveForAngularRates(double[] dt, double[] rates, double[] rotvecs)
    {
        var m = dt.Length;          // segments
        var first = rates[..3];
        var a = new double[9 * m];
        var aInv = new double[9 * m];
        for (var i = 0; i < m; i++)
        {
            RateToRotvecDot(rotvecs.AsSpan(3 * i, 3), a.AsSpan(9 * i, 9));
            RotvecDotToRate(rotvecs.AsSpan(3 * i, 3), aInv.AsSpan(9 * i, 9));
        }
        // Banded matrix (scipy layout, l = u = 5) of size 3(m − 1): diagonal blocks d_i·I,
        // sub-diagonal blocks 2 A⁻¹_{i+1}/dt_{i+1}, super-diagonal blocks 2 A_{i+1}/dt_{i+1}.
        var nb = m - 1;
        var size = 3 * nb;
        var ab = new double[11 * size];
        void Set(int r, int c, double v) => ab[(5 + r - c) * size + c] = v;
        for (var i = 0; i < nb; i++)
        {
            var di = 4 * (1 / dt[i] + 1 / dt[i + 1]);
            for (var k = 0; k < 3; k++) Set(3 * i + k, 3 * i + k, di);
        }
        for (var i = 0; i < nb - 1; i++)
        {
            var s = 2 / dt[i + 1];
            for (var r = 0; r < 3; r++)
                for (var c = 0; c < 3; c++)
                {
                    Set(3 * (i + 1) + r, 3 * i + c, s * aInv[9 * (i + 1) + 3 * r + c]);
                    Set(3 * i + r, 3 * (i + 1) + c, s * a[9 * (i + 1) + 3 * r + c]);
                }
        }
        var abNd = new NdArray(ab, 11, size);
        var b0 = new double[size];
        for (var i = 0; i < nb; i++)
            for (var j = 0; j < 3; j++)
                b0[3 * i + j] = 6 * (rotvecs[3 * i + j] / (dt[i] * dt[i]) + rotvecs[3 * (i + 1) + j] / (dt[i + 1] * dt[i + 1]));
        for (var j = 0; j < 3; j++)
        {
            double s0 = 0, s1 = 0;
            for (var k = 0; k < 3; k++)
            {
                s0 += aInv[3 * j + k] * first[k];
                s1 += a[9 * (m - 1) + 3 * j + k] * rates[3 * (m - 1) + k];
            }
            b0[j] -= 2 / dt[0] * s0;
            b0[3 * (nb - 1) + j] -= 2 / dt[m - 1] * s1;
        }
        var rotvecsDot = new double[3 * m];
        var b = new double[size];
        Span<double> nl = stackalloc double[3];
        for (var iter = 0; iter < MaxIter; iter++)
        {
            MatVec(a, rates, rotvecsDot, m);
            for (var i = 0; i < nb; i++)
            {
                NonlinearTerm(rotvecs.AsSpan(3 * i, 3), rotvecsDot.AsSpan(3 * i, 3), nl);
                for (var j = 0; j < 3; j++) b[3 * i + j] = b0[3 * i + j] - nl[j];
            }
            var sol = Linalg.SolveBanded(5, 5, abNd, new NdArray((double[])b.Clone(), size)).AsContiguous().ToArray();
            var converged = true;
            for (var k = 0; k < size; k++)
            {
                if (!(Math.Abs(sol[k] - rates[k]) < Tol * (1 + Math.Abs(sol[k])))) converged = false;
                rates[k] = sol[k];
            }
            if (converged) break;
        }
        MatVec(a, rates, rotvecsDot, m);
        var outRates = new double[3 * m];
        Array.Copy(first, outRates, 3);
        Array.Copy(rates, 0, outRates, 3, 3 * (m - 1));
        return (outRates, rotvecsDot);
    }

    private static void MatVec(double[] a, double[] x, double[] y, int m)
    {
        for (var i = 0; i < m; i++)
            for (var r = 0; r < 3; r++)
                y[3 * i + r] = a[9 * i + 3 * r] * x[3 * i] + a[9 * i + 3 * r + 1] * x[3 * i + 1] + a[9 * i + 3 * r + 2] * x[3 * i + 2];
    }

    private static void Skew(ReadOnlySpan<double> x, Span<double> s)
    {
        s[0] = 0; s[1] = -x[2]; s[2] = x[1];
        s[3] = x[2]; s[4] = 0; s[5] = -x[0];
        s[6] = -x[1]; s[7] = x[0]; s[8] = 0;
    }

    private static void IdentityPlus(double c1, double c2, ReadOnlySpan<double> rv, Span<double> result)
    {
        Span<double> s = stackalloc double[9];
        Skew(rv, s);
        for (var r = 0; r < 3; r++)
            for (var c = 0; c < 3; c++)
            {
                double ss = s[3 * r] * s[c] + s[3 * r + 1] * s[3 + c] + s[3 * r + 2] * s[6 + c];
                result[3 * r + c] = (r == c ? 1 : 0) + c1 * s[3 * r + c] + c2 * ss;
            }
    }

    /// <summary>scipy's _angular_rate_to_rotvec_dot_matrix.</summary>
    private static void RateToRotvecDot(ReadOnlySpan<double> rv, Span<double> result)
    {
        var nm = Math.Sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
        var k = nm > 1e-4 ? (1 - 0.5 * nm / Math.Tan(0.5 * nm)) / (nm * nm) : 1.0 / 12 + 1.0 / 720 * nm * nm;
        IdentityPlus(0.5, k, rv, result);
    }

    /// <summary>scipy's _rotvec_dot_to_angular_rate_matrix.</summary>
    private static void RotvecDotToRate(ReadOnlySpan<double> rv, Span<double> result)
    {
        var nm = Math.Sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
        double k1, k2;
        if (nm > 1e-4) { k1 = (1 - Math.Cos(nm)) / (nm * nm); k2 = (nm - Math.Sin(nm)) / (nm * nm * nm); }
        else { k1 = 0.5 - nm * nm / 24; k2 = 1.0 / 6 - nm * nm / 120; }
        IdentityPlus(-k1, k2, rv, result);
    }

    /// <summary>scipy's _angular_acceleration_nonlinear_term.</summary>
    private static void NonlinearTerm(ReadOnlySpan<double> rv, ReadOnlySpan<double> rd, Span<double> result)
    {
        var nm = Math.Sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
        var dp = rv[0] * rd[0] + rv[1] * rd[1] + rv[2] * rd[2];
        Span<double> cp = stackalloc double[3];
        Span<double> ccp = stackalloc double[3];
        Span<double> dccp = stackalloc double[3];
        Cross(rv, rd, cp); Cross(rv, cp, ccp); Cross(rd, cp, dccp);
        double k1, k2, k3;
        if (nm > 1e-4)
        {
            k1 = (-nm * Math.Sin(nm) - 2 * (Math.Cos(nm) - 1)) / Math.Pow(nm, 4);
            k2 = (-2 * nm + 3 * Math.Sin(nm) - nm * Math.Cos(nm)) / Math.Pow(nm, 5);
            k3 = (nm - Math.Sin(nm)) / (nm * nm * nm);
        }
        else
        {
            k1 = 1.0 / 12 - nm * nm / 180;
            k2 = -1.0 / 60 + nm * nm / 12604;
            k3 = 1.0 / 6 - nm * nm / 120;
        }
        for (var j = 0; j < 3; j++) result[j] = dp * (k1 * cp[j] + k2 * ccp[j]) + k3 * dccp[j];
    }

    private static void Cross(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> r)
    {
        r[0] = a[1] * b[2] - a[2] * b[1];
        r[1] = a[2] * b[0] - a[0] * b[2];
        r[2] = a[0] * b[1] - a[1] * b[0];
    }

    private int Segment(double t)
    {
        // searchsorted(side='right') − 1, clamped to the segments (PPoly extrapolation).
        var i = Array.BinarySearch(_times, t);
        if (i >= 0) { while (i + 1 < _times.Length && _times[i + 1] == t) i++; }
        else i = ~i - 1;
        return Math.Clamp(i, 0, _times.Length - 2);
    }

    private void Poly(double t, int seg, int derivative, Span<double> v)
    {
        var x = t - _times[seg];
        var o = 12 * seg;
        for (var j = 0; j < 3; j++)
        {
            double c0 = _coeff[o + j], c1 = _coeff[o + 3 + j], c2 = _coeff[o + 6 + j], c3 = _coeff[o + 9 + j];
            v[j] = derivative switch
            {
                0 => ((c0 * x + c1) * x + c2) * x + c3,
                1 => (3 * c0 * x + 2 * c1) * x + c2,
                _ => 6 * c0 * x + 2 * c1,
            };
        }
    }

    /// <summary>The interpolated rotations at <paramref name="times"/>, as a stack.</summary>
    public Rotation Evaluate(double[] times)
    {
        ArgumentNullException.ThrowIfNull(times);
        var rv = new double[3 * times.Length];
        var idx = new int[times.Length];
        for (var i = 0; i < times.Length; i++)
        {
            idx[i] = Segment(times[i]);
            Poly(times[i], idx[i], 0, rv.AsSpan(3 * i, 3));
        }
        return _rotations.Take(idx) * Rotation.FromRotvec(new NdArray(rv, times.Length, 3));
    }

    /// <summary>The interpolated rotation at one time.</summary>
    public Rotation Evaluate(double time) => Evaluate([time])[0];

    /// <summary>Angular rate (order 1) or angular acceleration (order 2) at each time, shape (n, 3).</summary>
    public NdArray Evaluate(double[] times, int order)
    {
        ArgumentNullException.ThrowIfNull(times);
        if (order == 0) throw new ArgumentException("Use Evaluate(times) for rotations (order 0).");
        if (order is not (1 or 2)) throw new ArgumentException("`order` must be 0, 1 or 2.");
        var r = new double[3 * times.Length];
        Span<double> v = stackalloc double[3];
        Span<double> vd = stackalloc double[3];
        Span<double> vdd = stackalloc double[3];
        Span<double> mat = stackalloc double[9];
        Span<double> nl = stackalloc double[3];
        for (var i = 0; i < times.Length; i++)
        {
            var seg = Segment(times[i]);
            Poly(times[i], seg, 0, v);
            Poly(times[i], seg, 1, vd);
            RotvecDotToRate(v, mat);
            if (order == 1)
            {
                for (var j = 0; j < 3; j++) r[3 * i + j] = mat[3 * j] * vd[0] + mat[3 * j + 1] * vd[1] + mat[3 * j + 2] * vd[2];
            }
            else
            {
                Poly(times[i], seg, 2, vdd);
                NonlinearTerm(v, vd, nl);
                for (var j = 0; j < 3; j++) r[3 * i + j] = mat[3 * j] * vdd[0] + mat[3 * j + 1] * vdd[1] + mat[3 * j + 2] * vdd[2] + nl[j];
            }
        }
        return new NdArray(r, times.Length, 3);
    }
}
