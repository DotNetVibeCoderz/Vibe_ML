using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.InterpolateSupport;

namespace Gravicode.Science.SciNet;

/// <summary>
/// The evaluation surface shared by the 1-D piecewise interpolants (<see cref="PPoly"/>,
/// <see cref="BPoly"/>, <see cref="BSpline"/> and the splines built on them): scalar, array,
/// span and n-D overloads, scipy's <c>axis</c> placement of the result, and threaded batches.
/// </summary>
/// <remarks>
/// <para>Values may be vector-valued: every point produces nt = product of
/// <see cref="ValueShape"/> numbers. The overloads returning <c>double</c> or <c>double[]</c>
/// with one entry per point require nt == 1 and throw otherwise, so a vector-valued
/// interpolant is never silently read as its first component.</para>
/// <para>Evaluation is read-only and thread-safe. Batches above a few hundred thousand flops
/// are split across threads; each thread keeps its own interval hint, so sorted queries find
/// their interval in O(1).</para>
/// </remarks>
public abstract class PiecewiseInterpolantBase
{
    private protected int _nt;
    private protected int[] _rest;

    private protected PiecewiseInterpolantBase(int[] rest, InterpolateExtrapolate extrapolate, int axis)
    {
        _rest = rest;
        _nt = Product(rest);
        Extrapolate = extrapolate;
        Axis = axis;
    }

    /// <summary>Extrapolation mode used when a call does not override it.</summary>
    public InterpolateExtrapolate Extrapolate { get; set; }

    /// <summary>The interpolation axis of the original data (scipy's <c>axis</c>); n-D
    /// evaluation results put the x dimensions at this position.</summary>
    public int Axis { get; }

    /// <summary>Shape of the value at one point (empty for a scalar interpolant).</summary>
    public int[] ValueShape => (int[])_rest.Clone();

    private protected void RequireScalar()
    {
        if (_nt != 1) throw new InvalidOperationException("This interpolant is vector-valued; use the NdArray or *Values overloads.");
    }

    /// <summary>Evaluates the interpolant (or its <paramref name="nu"/>-th derivative) at x.</summary>
    public double Evaluate(double x, int nu = 0, InterpolateExtrapolate? extrapolate = null)
    {
        RequireScalar();
        Span<double> o = stackalloc double[1];
        CheckNu(nu);
        EvalChunk([x], o, nu, extrapolate ?? Extrapolate);
        return o[0];
    }

    /// <summary>Evaluates at every point of <paramref name="x"/> (scalar-valued interpolants).</summary>
    public double[] Evaluate(double[] x, int nu = 0, InterpolateExtrapolate? extrapolate = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        RequireScalar();
        var res = new double[x.Length];
        EvaluateBatch(x, res, nu, extrapolate ?? Extrapolate);
        return res;
    }

    /// <summary>Evaluates into a caller-provided buffer of length x.Length × nt, serially and
    /// without allocating.</summary>
    public void Evaluate(ReadOnlySpan<double> x, Span<double> destination, int nu = 0, InterpolateExtrapolate? extrapolate = null)
    {
        if (destination.Length < x.Length * _nt) throw new ArgumentException("destination is too short.", nameof(destination));
        CheckNu(nu);
        EvalChunk(x, destination, nu, extrapolate ?? Extrapolate);
    }

    /// <summary>All value components at one point (length = product of <see cref="ValueShape"/>).</summary>
    public double[] EvaluateValues(double x, int nu = 0, InterpolateExtrapolate? extrapolate = null)
    {
        CheckNu(nu);
        var o = new double[_nt];
        EvalChunk([x], o, nu, extrapolate ?? Extrapolate);
        return o;
    }

    /// <summary>
    /// Evaluates at an array of points of any shape. The result has shape
    /// <c>ValueShape[:axis] + x.shape + ValueShape[axis:]</c>, as scipy's <c>__call__</c>.
    /// </summary>
    public NdArray Evaluate(NdArray x, int nu = 0, InterpolateExtrapolate? extrapolate = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var xs = x.ToArray();
        var res = new double[xs.Length * _nt];
        EvaluateBatch(xs, res, nu, extrapolate ?? Extrapolate);
        return PlaceResult(res, x.Shape, _rest, Axis);
    }

    private protected virtual void CheckNu(int nu)
    {
        if (nu < 0) throw new ArgumentException("Order of derivative to evaluate must be non-negative.", nameof(nu));
    }

    private void EvaluateBatch(double[] x, double[] res, int nu, InterpolateExtrapolate e)
    {
        CheckNu(nu);
        var nt = _nt;
        ForChunks(x.Length, CostPerPoint + 4L * nt, (s, en) =>
            EvalChunk(x.AsSpan(s, en - s), res.AsSpan(s * nt, (en - s) * nt), nu, e));
    }

    /// <summary>Rough flop count of one evaluation, for the threading threshold.</summary>
    private protected abstract long CostPerPoint { get; }

    /// <summary>Evaluates a contiguous run of points; output is (len, nt) row-major.</summary>
    private protected abstract void EvalChunk(ReadOnlySpan<double> xs, Span<double> output, int nu, InterpolateExtrapolate mode);
}
