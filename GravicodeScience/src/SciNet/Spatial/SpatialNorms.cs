using System.Runtime.CompilerServices;

namespace Gravicode.Science.SciNet.Spatial;

/// <summary>
/// A Minkowski norm in "p-th power" form, as a struct so generic kernels specialise per norm.
/// </summary>
/// <remarks>
/// Like cKDTree, tree searches compare <c>sum |d_i|^p</c> rather than its p-th root — the root
/// is monotone, so the comparison is unchanged and one <c>Math.Pow</c> per candidate is saved.
/// The root is taken once, on the results. For p = infinity the "power" form is the max itself.
/// </remarks>
internal interface ISpatialNorm
{
    /// <summary>Folds one absolute coordinate difference into the accumulator.</summary>
    double Add(double accumulator, double absDiff);

    /// <summary>Distance in power form between two rows.</summary>
    double Point(ReadOnlySpan<double> a, ReadOnlySpan<double> b);

    /// <summary>Converts a real distance into power form.</summary>
    double ToPower(double distance);

    /// <summary>Converts a power-form distance into a real distance.</summary>
    double FromPower(double power);
}

internal readonly struct SpatialNormL2 : ISpatialNorm
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Add(double accumulator, double absDiff) => accumulator + absDiff * absDiff;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Point(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var s = 0.0;
        for (var i = 0; i < a.Length; i++) { var d = a[i] - b[i]; s += d * d; }
        return s;
    }

    public double ToPower(double distance) => double.IsPositiveInfinity(distance) ? distance : distance * distance;
    public double FromPower(double power) => Math.Sqrt(power);
}

internal readonly struct SpatialNormL1 : ISpatialNorm
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Add(double accumulator, double absDiff) => accumulator + absDiff;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Point(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var s = 0.0;
        for (var i = 0; i < a.Length; i++) s += Math.Abs(a[i] - b[i]);
        return s;
    }

    public double ToPower(double distance) => distance;
    public double FromPower(double power) => power;
}

internal readonly struct SpatialNormLInf : ISpatialNorm
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Add(double accumulator, double absDiff) => Math.Max(accumulator, absDiff);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Point(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var s = 0.0;
        for (var i = 0; i < a.Length; i++) s = Math.Max(s, Math.Abs(a[i] - b[i]));
        return s;
    }

    public double ToPower(double distance) => distance;
    public double FromPower(double power) => power;
}

internal readonly struct SpatialNormLp(double p) : ISpatialNorm
{
    private readonly double _p = p;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Add(double accumulator, double absDiff) => accumulator + Math.Pow(absDiff, _p);

    public double Point(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var s = 0.0;
        for (var i = 0; i < a.Length; i++) s += Math.Pow(Math.Abs(a[i] - b[i]), _p);
        return s;
    }

    public double ToPower(double distance) => double.IsPositiveInfinity(distance) ? distance : Math.Pow(distance, _p);
    public double FromPower(double power) => Math.Pow(power, 1.0 / _p);
}
