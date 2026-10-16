using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.InterpolateSupport;

namespace Gravicode.Science.SciNet;

/// <summary>What a <see cref="SplineBoundaryCondition"/> prescribes.</summary>
public enum SplineBoundaryKind
{
    /// <summary>No derivative condition: the first two (last two) pieces form one polynomial
    /// (for <c>Interpolate.MakeInterpSpline</c>,
    /// "no conditions at this end").</summary>
    NotAKnot,
    /// <summary>Periodic: value and derivatives match at both ends.</summary>
    Periodic,
    /// <summary>Prescribed derivative values (clamped, natural, or custom).</summary>
    Derivatives,
}

/// <summary>
/// One end condition of an interpolating spline (scipy's <c>bc_type</c> entries:
/// <c>'not-a-knot'</c>, <c>'periodic'</c>, <c>'clamped'</c>, <c>'natural'</c>, or
/// <c>(order, value)</c> pairs).
/// </summary>
public sealed class SplineBoundaryCondition
{
    private SplineBoundaryCondition(SplineBoundaryKind kind, (int Order, double[] Value)[] derivatives)
    {
        Kind = kind;
        Derivatives = derivatives;
    }

    /// <summary>The kind of condition.</summary>
    public SplineBoundaryKind Kind { get; }

    /// <summary>(order, value) pairs for <see cref="SplineBoundaryKind.Derivatives"/>; an empty
    /// value array means zero for every component.</summary>
    public IReadOnlyList<(int Order, double[] Value)> Derivatives { get; }

    /// <summary>Not-a-knot (scipy's default).</summary>
    public static SplineBoundaryCondition NotAKnot { get; } = new(SplineBoundaryKind.NotAKnot, []);

    /// <summary>Periodic (must be given for both ends).</summary>
    public static SplineBoundaryCondition Periodic { get; } = new(SplineBoundaryKind.Periodic, []);

    /// <summary>First derivative zero.</summary>
    public static SplineBoundaryCondition Clamped { get; } = new(SplineBoundaryKind.Derivatives, [(1, [])]);

    /// <summary>Second derivative zero.</summary>
    public static SplineBoundaryCondition Natural { get; } = new(SplineBoundaryKind.Derivatives, [(2, [])]);

    /// <summary>A prescribed derivative of the given order (one value per component).</summary>
    public static SplineBoundaryCondition Derivative(int order, params double[] value) =>
        new(SplineBoundaryKind.Derivatives, [(order, (double[])value.Clone())]);

    /// <summary>Several prescribed derivatives at one end (for higher-degree
    /// <c>MakeInterpSpline</c>), scalar-valued.</summary>
    public static SplineBoundaryCondition FromDerivatives(params (int Order, double Value)[] derivatives) =>
        new(SplineBoundaryKind.Derivatives, derivatives.Select(d => (d.Order, new[] { d.Value })).ToArray());

    /// <summary>Several prescribed derivatives at one end, vector-valued.</summary>
    public static SplineBoundaryCondition FromDerivatives(params (int Order, double[] Value)[] derivatives) =>
        new(SplineBoundaryKind.Derivatives, derivatives.Select(d => (d.Order, (double[])d.Value.Clone())).ToArray());

    internal double[] ValueFor(int index, int nt)
    {
        var v = Derivatives[index].Value;
        if (v.Length == 0) return new double[nt];
        if (v.Length == nt) return v;
        if (v.Length == 1) return Enumerable.Repeat(v[0], nt).ToArray();
        throw new ArgumentException($"Derivative value has {v.Length} components, expected {nt}.");
    }
}

/// <summary>A pair of end conditions; converts implicitly from one condition (both ends) or a
/// (start, end) tuple, mirroring scipy's string-or-pair <c>bc_type</c>.</summary>
public readonly record struct SplineBoundary(SplineBoundaryCondition Start, SplineBoundaryCondition End)
{
    /// <summary>The same condition at both ends.</summary>
    public static implicit operator SplineBoundary(SplineBoundaryCondition both) => new(both, both);

    /// <summary>Different conditions at the two ends.</summary>
    public static implicit operator SplineBoundary((SplineBoundaryCondition Start, SplineBoundaryCondition End) pair) => new(pair.Start, pair.End);
}

/// <summary>
/// Piecewise cubic Hermite interpolant with given values and first derivatives
/// (scipy.interpolate.CubicHermiteSpline). The result is a <see cref="PPoly"/>.
/// </summary>
public class CubicHermiteSpline : PPoly
{
    private protected sealed record Prepared(double[] X, double[] Dx, double[] Y, int N, int[] Rest, int Axis);

    /// <summary>Scalar data: y and dydx have the length of x.</summary>
    public CubicHermiteSpline(double[] x, double[] y, double[] dydx, InterpolateExtrapolate? extrapolate = null)
        : this(Prepare(x, new NdArray(Copy(y, nameof(y)), y.Length), 0), Copy(dydx, nameof(dydx)), extrapolate ?? InterpolateExtrapolate.Extrapolate) { }

    /// <summary>N-D data: x runs along <paramref name="axis"/> of y; dydx has y's shape.</summary>
    public CubicHermiteSpline(double[] x, NdArray y, NdArray dydx, int axis = 0, InterpolateExtrapolate? extrapolate = null)
        : this(Prepare(x, y, axis), PrepareDerivative(y, dydx, axis), extrapolate ?? InterpolateExtrapolate.Extrapolate) { }

    private protected CubicHermiteSpline(Prepared p, double[] dydx, InterpolateExtrapolate extrapolate)
        : base(HermiteCoefficients(p, dydx), 4, p.N - 1, p.Rest, p.X, extrapolate, p.Axis) { }

    private static double[] Copy(double[] v, string name)
    {
        ArgumentNullException.ThrowIfNull(v, name);
        return (double[])v.Clone();
    }

    private static double[] PrepareDerivative(NdArray y, NdArray dydx, int axis)
    {
        if (!y.Shape.SequenceEqual(dydx.Shape)) throw new ArgumentException("The shapes of `y` and `dydx` must be identical.");
        var d = MoveAxisToFront(dydx, NormalizeAxis(axis, y.Rank), out _, out _);
        foreach (var v in d) if (!double.IsFinite(v)) throw new ArgumentException("`dydx` must contain only finite values.");
        return d;
    }

    /// <summary>scipy's <c>prepare_input</c>: validates x and y and moves the axis to the front.</summary>
    private protected static Prepared Prepare(double[] x, NdArray y, int axis)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        axis = NormalizeAxis(axis, y.Rank);
        if (x.Length < 2) throw new ArgumentException("`x` must contain at least 2 elements.", nameof(x));
        if (x.Length != y.Shape[axis]) throw new ArgumentException($"The length of `y` along `axis`={axis} doesn't match the length of `x`");
        var xs = CopyFinite(x, "x");
        var yy = MoveAxisToFront(y, axis, out var n, out var rest);
        foreach (var v in yy) if (!double.IsFinite(v)) throw new ArgumentException("`y` must contain only finite values.", nameof(y));
        var dx = new double[n - 1];
        for (var i = 0; i < n - 1; i++)
        {
            dx[i] = xs[i + 1] - xs[i];
            if (!(dx[i] > 0)) throw new ArgumentException("`x` must be strictly increasing sequence.", nameof(x));
        }
        return new Prepared(xs, dx, yy, n, rest, axis);
    }

    private static double[] HermiteCoefficients(Prepared p, double[] d)
    {
        int n = p.N, m = n - 1, nt = Product(p.Rest);
        var y = p.Y;
        var c = new double[4 * m * nt];
        for (var i = 0; i < m; i++)
        {
            var dx = p.Dx[i];
            for (var j = 0; j < nt; j++)
            {
                var y0 = y[i * nt + j];
                var slope = (y[(i + 1) * nt + j] - y0) / dx;
                var d0 = d[i * nt + j];
                var t = (d0 + d[(i + 1) * nt + j] - 2 * slope) / dx;
                c[(0 * m + i) * nt + j] = t / dx;
                c[(1 * m + i) * nt + j] = (slope - d0) / dx - t;
                c[(2 * m + i) * nt + j] = d0;
                c[(3 * m + i) * nt + j] = y0;
            }
        }
        return c;
    }
}

/// <summary>
/// PCHIP: monotonicity-preserving piecewise cubic Hermite interpolation
/// (scipy.interpolate.PchipInterpolator; Fritsch–Butland harmonic-mean slopes with Moler's
/// one-sided, shape-preserving end slopes).
/// </summary>
public class PchipInterpolator : CubicHermiteSpline
{
    /// <summary>Scalar data.</summary>
    public PchipInterpolator(double[] x, double[] y, InterpolateExtrapolate? extrapolate = null)
        : this(x, new NdArray((double[])y.Clone(), y.Length), 0, extrapolate) { }

    /// <summary>N-D data along <paramref name="axis"/>.</summary>
    public PchipInterpolator(double[] x, NdArray y, int axis = 0, InterpolateExtrapolate? extrapolate = null)
        : this(Prepare(x, y, axis), extrapolate) { }

    private PchipInterpolator(Prepared p, InterpolateExtrapolate? extrapolate)
        : base(p, FindDerivatives(p), extrapolate ?? InterpolateExtrapolate.Extrapolate) { }

    private static double EdgeCase(double h0, double h1, double m0, double m1)
    {
        var d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
        var mask = Math.Sign(d) != Math.Sign(m0);
        var mask2 = Math.Sign(m0) != Math.Sign(m1) && Math.Abs(d) > 3.0 * Math.Abs(m0);
        if (mask) return 0.0;
        if (mask2) return 3.0 * m0;
        return d;
    }

    private static double[] FindDerivatives(Prepared p)
    {
        int n = p.N, nt = Product(p.Rest);
        var y = p.Y;
        var hk = p.Dx;
        var dk = new double[n * nt];
        var mk = new double[n - 1];
        for (var j = 0; j < nt; j++)
        {
            for (var i = 0; i < n - 1; i++) mk[i] = (y[(i + 1) * nt + j] - y[i * nt + j]) / hk[i];
            if (n == 2)
            {
                dk[j] = mk[0]; dk[nt + j] = mk[0];
                continue;
            }
            for (var i = 1; i < n - 1; i++)
            {
                double m0 = mk[i - 1], m1 = mk[i];
                var cond = Math.Sign(m1) != Math.Sign(m0) || m1 == 0 || m0 == 0;
                if (cond) { dk[i * nt + j] = 0; continue; }
                var w1 = 2 * hk[i] + hk[i - 1];
                var w2 = hk[i] + 2 * hk[i - 1];
                var whmean = (w1 / m0 + w2 / m1) / (w1 + w2);
                dk[i * nt + j] = 1.0 / whmean;
            }
            dk[j] = EdgeCase(hk[0], hk[1], mk[0], mk[1]);
            dk[(n - 1) * nt + j] = EdgeCase(hk[n - 2], hk[n - 3], mk[n - 2], mk[n - 3]);
        }
        return dk;
    }
}

/// <summary>Slope rule of <see cref="Akima1DInterpolator"/>.</summary>
public enum AkimaMethod
{
    /// <summary>Akima's original 1970 weights.</summary>
    Akima,
    /// <summary>Modified Akima (MATLAB's <c>makima</c>): adds |m_i + m_{i+1}|/2 to the weights,
    /// avoiding overshoot when several consecutive slopes are equal.</summary>
    Makima,
}

/// <summary>
/// Akima / modified-Akima cubic interpolation (scipy.interpolate.Akima1DInterpolator).
/// Extrapolation is off by default, as in scipy.
/// </summary>
public class Akima1DInterpolator : CubicHermiteSpline
{
    /// <summary>Scalar data.</summary>
    public Akima1DInterpolator(double[] x, double[] y, AkimaMethod method = AkimaMethod.Akima, InterpolateExtrapolate? extrapolate = null)
        : this(x, new NdArray((double[])y.Clone(), y.Length), 0, method, extrapolate) { }

    /// <summary>N-D data along <paramref name="axis"/>.</summary>
    public Akima1DInterpolator(double[] x, NdArray y, int axis = 0, AkimaMethod method = AkimaMethod.Akima, InterpolateExtrapolate? extrapolate = null)
        : this(Prepare(x, y, axis), method, extrapolate) { }

    private Akima1DInterpolator(Prepared p, AkimaMethod method, InterpolateExtrapolate? extrapolate)
        : base(p, Slopes(p, method), extrapolate ?? InterpolateExtrapolate.None) { }

    private static double[] Slopes(Prepared p, AkimaMethod method)
    {
        int n = p.N, nt = Product(p.Rest);
        var y = p.Y;
        var t = new double[n * nt];
        if (n == 2)
        {
            for (var j = 0; j < nt; j++)
            {
                var mk = (y[nt + j] - y[j]) / p.Dx[0];
                t[j] = mk; t[nt + j] = mk;
            }
            return t;
        }
        // m has n+3 rows: slopes in 2..n, two extrapolated slopes at each end.
        var m = new double[(n + 3) * nt];
        for (var i = 0; i < n - 1; i++)
            for (var j = 0; j < nt; j++) m[(i + 2) * nt + j] = (y[(i + 1) * nt + j] - y[i * nt + j]) / p.Dx[i];
        for (var j = 0; j < nt; j++)
        {
            m[1 * nt + j] = 2.0 * m[2 * nt + j] - m[3 * nt + j];
            m[0 * nt + j] = 2.0 * m[1 * nt + j] - m[2 * nt + j];
            m[(n + 1) * nt + j] = 2.0 * m[n * nt + j] - m[(n - 1) * nt + j];
            m[(n + 2) * nt + j] = 2.0 * m[(n + 1) * nt + j] - m[n * nt + j];
        }
        var f1 = new double[n * nt];
        var f2 = new double[n * nt];
        var f12 = new double[n * nt];
        var mmax = double.NegativeInfinity;
        for (var i = 0; i < n; i++)
            for (var j = 0; j < nt; j++)
            {
                t[i * nt + j] = 0.5 * (m[(i + 3) * nt + j] + m[i * nt + j]);
                // dm[q] = |m[q+1] - m[q]|; f1 = dm[i+2], f2 = dm[i]
                var dm2 = Math.Abs(m[(i + 3) * nt + j] - m[(i + 2) * nt + j]);
                var dm0 = Math.Abs(m[(i + 1) * nt + j] - m[i * nt + j]);
                double a = dm2, b = dm0;
                if (method == AkimaMethod.Makima)
                {
                    // pm[q] = |m[q+1] + m[q]|
                    a += 0.5 * Math.Abs(m[(i + 3) * nt + j] + m[(i + 2) * nt + j]);
                    b += 0.5 * Math.Abs(m[(i + 1) * nt + j] + m[i * nt + j]);
                }
                f1[i * nt + j] = a; f2[i * nt + j] = b;
                var s = a + b;
                f12[i * nt + j] = s;
                if (s > mmax) mmax = s;
            }
        // scipy compares against the maximum over *all* components.
        const double breakMult = 1e-9;
        for (var i = 0; i < n; i++)
            for (var j = 0; j < nt; j++)
            {
                var idx = i * nt + j;
                if (f12[idx] > breakMult * mmax)
                    t[idx] = m[(i + 1) * nt + j] + f2[idx] / f12[idx] * (m[(i + 2) * nt + j] - m[(i + 1) * nt + j]);
            }
        return t;
    }

    /// <summary>Not supported, as in scipy: an extended Akima interpolant would not be one.</summary>
    public override void Extend(NdArray c, double[] x) =>
        throw new NotSupportedException("Extending a 1-D Akima interpolator is not yet implemented");
}

/// <summary>
/// Cubic spline interpolation, C² (scipy.interpolate.CubicSpline): first derivatives at the
/// nodes come from one tridiagonal (banded, partially pivoted) solve shared by every value
/// component; periodic splines solve the cyclic system by the condensation scipy uses.
/// </summary>
public class CubicSpline : CubicHermiteSpline
{
    /// <summary>Scalar data. <paramref name="bcType"/> defaults to not-a-knot at both ends;
    /// extrapolation defaults to periodic for periodic splines and on otherwise.</summary>
    public CubicSpline(double[] x, double[] y, SplineBoundary? bcType = null, InterpolateExtrapolate? extrapolate = null)
        : this(x, new NdArray((double[])y.Clone(), y.Length), 0, bcType, extrapolate) { }

    /// <summary>N-D data along <paramref name="axis"/>.</summary>
    public CubicSpline(double[] x, NdArray y, int axis = 0, SplineBoundary? bcType = null, InterpolateExtrapolate? extrapolate = null)
        : this(Prepare(x, y, axis), bcType ?? SplineBoundaryCondition.NotAKnot, extrapolate) { }

    private CubicSpline(Prepared p, SplineBoundary bc, InterpolateExtrapolate? extrapolate)
        : base(p, Slopes(p, bc), extrapolate ?? (bc.Start.Kind == SplineBoundaryKind.Periodic ? InterpolateExtrapolate.Periodic : InterpolateExtrapolate.Extrapolate)) { }

    private enum EndKind { NotAKnot, Periodic, First, Second }

    private static (EndKind Kind, double[] Value) Validate(SplineBoundaryCondition c, int nt)
    {
        switch (c.Kind)
        {
            case SplineBoundaryKind.NotAKnot: return (EndKind.NotAKnot, []);
            case SplineBoundaryKind.Periodic: return (EndKind.Periodic, []);
            default:
                if (c.Derivatives.Count != 1) throw new ArgumentException("CubicSpline takes exactly one (order, value) per end.");
                var order = c.Derivatives[0].Order;
                if (order != 1 && order != 2) throw new ArgumentException("The specified derivative order must be 1 or 2.");
                return (order == 1 ? EndKind.First : EndKind.Second, c.ValueFor(0, nt));
        }
    }

    private static double[] Slopes(Prepared p, SplineBoundary bc)
    {
        int n = p.N, nt = Product(p.Rest);
        var x = p.X; var dx = p.Dx; var y = p.Y;
        if (bc.Start is null || bc.End is null) throw new ArgumentNullException(nameof(bc));
        var periodicStart = bc.Start.Kind == SplineBoundaryKind.Periodic;
        var periodicEnd = bc.End.Kind == SplineBoundaryKind.Periodic;
        if (periodicStart != periodicEnd)
            throw new ArgumentException("'periodic' `bc_type` is defined for both curve ends and cannot be used with other boundary conditions.");
        if (periodicStart)
        {
            for (var j = 0; j < nt; j++)
            {
                double a = y[j], b = y[(n - 1) * nt + j];
                if (!(Math.Abs(a - b) <= 1e-15 + 1e-15 * Math.Abs(b)))
                    throw new ArgumentException($"The first and last `y` point along axis {p.Axis} must be identical (within machine precision) when bc_type='periodic'.");
            }
        }
        var start = Validate(bc.Start, nt);
        var end = Validate(bc.End, nt);
        var s = new double[n * nt];
        if (nt == 0) return s;
        var slope = new double[(n - 1) * nt];
        for (var i = 0; i < n - 1; i++)
            for (var j = 0; j < nt; j++) slope[i * nt + j] = (y[(i + 1) * nt + j] - y[i * nt + j]) / dx[i];
        double Sl(int i, int j) => slope[(i < 0 ? i + n - 1 : i) * nt + j];
        double Dx(int i) => dx[i < 0 ? i + n - 1 : i];

        if (n == 2)
        {
            if (start.Kind is EndKind.NotAKnot or EndKind.Periodic) start = (EndKind.First, Enumerable.Range(0, nt).Select(j => slope[j]).ToArray());
            if (end.Kind is EndKind.NotAKnot or EndKind.Periodic) end = (EndKind.First, Enumerable.Range(0, nt).Select(j => slope[j]).ToArray());
        }

        if (n == 3 && start.Kind == EndKind.NotAKnot && end.Kind == EndKind.NotAKnot)
        {
            var a = new double[9];
            a[0] = 1; a[1] = 1;
            a[3] = dx[1]; a[4] = 2 * (dx[0] + dx[1]); a[5] = dx[0];
            a[7] = 1; a[8] = 1;
            var b = new double[3 * nt];
            for (var j = 0; j < nt; j++)
            {
                b[j] = 2 * slope[j];
                b[nt + j] = 3 * (dx[0] * slope[nt + j] + dx[1] * slope[j]);
                b[2 * nt + j] = 2 * slope[nt + j];
            }
            return DenseSolve(a, 3, b, nt);
        }
        if (n == 3 && start.Kind == EndKind.Periodic)
        {
            for (var j = 0; j < nt; j++)
            {
                double num = 0, den = 0;
                for (var i = 0; i < 2; i++) { num += slope[i * nt + j] / dx[i]; den += 1.0 / dx[i]; }
                var t = num / den;
                for (var i = 0; i < 3; i++) s[i * nt + j] = t;
            }
            return s;
        }

        if (start.Kind == EndKind.Periodic)
        {
            // (n-1) unknowns; condensed (n-2) system solved twice (scipy's approach).
            var nc = n - 2;
            var ac = new BandedLu(nc, 1, 1);
            for (var i = 0; i < nc; i++)
            {
                ac[i, i] = i == 0 ? 2 * (Dx(-1) + dx[0]) : 2 * (dx[i - 1] + dx[i]);
                if (i + 1 < nc) ac[i, i + 1] = i == 0 ? Dx(-1) : dx[i - 1];
                if (i >= 1) ac[i, i - 1] = dx[i];
            }
            ac.Factor();
            // b over the n-1 system rows
            var bfull = new double[(n - 1) * nt];
            for (var i = 1; i < n - 1; i++)
                for (var j = 0; j < nt; j++) bfull[i * nt + j] = 3 * (dx[i] * slope[(i - 1) * nt + j] + dx[i - 1] * slope[i * nt + j]);
            for (var j = 0; j < nt; j++)
            {
                bfull[j] = 3 * (dx[0] * Sl(-1, j) + Dx(-1) * slope[j]);
                bfull[(n - 2) * nt + j] = 3 * (Dx(-1) * Sl(-2, j) + Dx(-2) * Sl(-1, j));
            }
            var s1 = bfull.AsSpan(0, nc * nt).ToArray();
            ac.Solve(s1, nt);
            var s2 = new double[nc];
            s2[0] = -dx[0];
            s2[nc - 1] = -Dx(-3);
            ac.Solve(s2, 1);
            double am10 = Dx(-2), am1m2 = Dx(-1), am1m1 = 2 * (Dx(-1) + Dx(-2));
            for (var j = 0; j < nt; j++)
            {
                var sm1 = (bfull[(n - 2) * nt + j] - am10 * s1[j] - am1m2 * s1[(nc - 1) * nt + j]) /
                          (am1m1 + am10 * s2[0] + am1m2 * s2[nc - 1]);
                for (var i = 0; i < nc; i++) s[i * nt + j] = s1[i * nt + j] + sm1 * s2[i];
                s[(n - 2) * nt + j] = sm1;
                s[(n - 1) * nt + j] = s[j];
            }
            return s;
        }

        var lu = new BandedLu(n, 1, 1);
        var rhs = new double[n * nt];
        for (var i = 1; i < n - 1; i++)
        {
            lu[i, i] = 2 * (dx[i - 1] + dx[i]);
            lu[i, i + 1] = dx[i - 1];
            lu[i, i - 1] = dx[i];
            for (var j = 0; j < nt; j++) rhs[i * nt + j] = 3 * (dx[i] * slope[(i - 1) * nt + j] + dx[i - 1] * slope[i * nt + j]);
        }
        switch (start.Kind)
        {
            case EndKind.NotAKnot:
            {
                lu[0, 0] = dx[1];
                var d = x[2] - x[0];
                lu[0, 1] = d;
                for (var j = 0; j < nt; j++)
                    rhs[j] = ((dx[0] + 2 * d) * dx[1] * slope[j] + dx[0] * dx[0] * slope[nt + j]) / d;
                break;
            }
            case EndKind.First:
                lu[0, 0] = 1; lu[0, 1] = 0;
                for (var j = 0; j < nt; j++) rhs[j] = start.Value[j];
                break;
            case EndKind.Second:
                lu[0, 0] = 2 * dx[0]; lu[0, 1] = dx[0];
                for (var j = 0; j < nt; j++) rhs[j] = -0.5 * start.Value[j] * dx[0] * dx[0] + 3 * (y[nt + j] - y[j]);
                break;
        }
        switch (end.Kind)
        {
            case EndKind.NotAKnot:
            {
                lu[n - 1, n - 1] = Dx(-2);
                var d = x[n - 1] - x[n - 3];
                lu[n - 1, n - 2] = d;
                for (var j = 0; j < nt; j++)
                    rhs[(n - 1) * nt + j] = (Dx(-1) * Dx(-1) * Sl(-2, j) + (2 * d + Dx(-1)) * Dx(-2) * Sl(-1, j)) / d;
                break;
            }
            case EndKind.First:
                lu[n - 1, n - 1] = 1; lu[n - 1, n - 2] = 0;
                for (var j = 0; j < nt; j++) rhs[(n - 1) * nt + j] = end.Value[j];
                break;
            case EndKind.Second:
                lu[n - 1, n - 1] = 2 * Dx(-1); lu[n - 1, n - 2] = Dx(-1);
                for (var j = 0; j < nt; j++)
                    rhs[(n - 1) * nt + j] = 0.5 * end.Value[j] * Dx(-1) * Dx(-1) + 3 * (y[(n - 1) * nt + j] - y[(n - 2) * nt + j]);
                break;
        }
        lu.Factor();
        lu.Solve(rhs, nt);
        return rhs;
    }
}
