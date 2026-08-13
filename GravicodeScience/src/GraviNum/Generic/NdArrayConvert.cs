using System.Numerics;

namespace Gravicode.Science.GraviNum.Generic;

/// <summary>
/// Converts between <see cref="NdArray"/> and <see cref="NdArray{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// The six libraries are written against the <c>double</c> <see cref="NdArray"/>. Rather than
/// migrate them all at once — a change that would touch every file and every test in the solution —
/// the generic core sits alongside, and code that wants it converts at the boundary.
/// </para>
/// <para>
/// Conversions <b>copy</b>. There is no way to reinterpret a <c>double[]</c> as a <c>float[]</c>
/// without changing every value, so hiding the cost behind a cast would be a lie about what the
/// call does. Going to <c>float</c> also loses precision — about sixteen significant digits become
/// about seven — which is the trade the caller is choosing to make.
/// </para>
/// </remarks>
public static class NdArrayConvert
{
    /// <summary>Copies a double-precision array into the generic type.</summary>
    public static NdArray<T> To<T>(NdArray source) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(source);

        var flat = source.AsContiguous().ToArray();
        var data = new T[flat.Length];
        for (var i = 0; i < flat.Length; i++) data[i] = T.CreateTruncating(flat[i]);

        return new NdArray<T>(data, source.Shape.ToArray());
    }

    /// <summary>Copies a generic array back to double precision.</summary>
    /// <remarks>
    /// Widening from <c>float</c> is exact — every float32 is representable as a double — so this
    /// direction loses nothing. It does not recover the digits the narrowing threw away.
    /// </remarks>
    public static NdArray ToDouble<T>(NdArray<T> source) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(source);

        var flat = source.AsContiguous().ToArray();
        var data = new double[flat.Length];
        for (var i = 0; i < flat.Length; i++) data[i] = double.CreateTruncating(flat[i]);

        return new NdArray(data, source.Shape.ToArray());
    }

    /// <summary>Shorthand for <see cref="To{T}"/> with <see cref="float"/>.</summary>
    public static NdArray<float> ToSingle(NdArray source) => To<float>(source);

    /// <summary>
    /// The relative precision of <typeparamref name="T"/> — the smallest step visible next to one.
    /// </summary>
    /// <remarks>
    /// Worth having in the open. Tests that pin a result to a tolerance need to know which width
    /// they are checking, and a tolerance written for <c>double</c> silently over-asserts on
    /// <c>float</c>.
    /// </remarks>
    public static T Epsilon<T>() where T : unmanaged, IFloatingPointIeee754<T> => T.Epsilon;

    /// <summary>A tolerance appropriate to <typeparamref name="T"/> for comparing computed values.</summary>
    /// <remarks>
    /// Roughly the square root of the machine epsilon, which is the usual allowance once a value
    /// has been through a few operations: about 1e-8 for <c>double</c> and 3e-4 for <c>float</c>.
    /// </remarks>
    public static double ComparisonTolerance<T>() where T : unmanaged, IFloatingPointIeee754<T>
        => typeof(T) == typeof(float) ? 1e-4 : 1e-9;
}
