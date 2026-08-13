using System.Numerics;
using System.Runtime.CompilerServices;

namespace Gravicode.Science.GraviNum;

/// <summary>
/// Element-wise ("universal") functions over <see cref="NdArray"/>.
/// </summary>
/// <remarks>
/// Every binary op goes through the same three-tier dispatch: a <see cref="Vector{T}"/> SIMD loop
/// when both operands are contiguous and identically shaped, a <see cref="Parallel"/> partitioned
/// version of that loop once the array is large enough to pay for the threads, and a strided
/// broadcast walk otherwise. Callers never pick a tier by hand.
/// </remarks>
public static class UFunc
{
    /// <summary>Element count above which element-wise work is spread across threads.</summary>
    public const int ParallelThreshold = 24_000;

    /// <summary>Vector width used by the SIMD kernels on this machine.</summary>
    public static int VectorWidth => Vector<double>.Count;

    /// <summary>True when the runtime exposes hardware SIMD for <see cref="double"/>.</summary>
    public static bool IsSimdAccelerated => Vector.IsHardwareAccelerated;

    // ---------------------------------------------------------------- kernels

    /// <summary>Applies <paramref name="scalar"/> to every element, vectorising with <paramref name="vector"/>.</summary>
    public static NdArray Unary(
        NdArray a,
        Func<double, double> scalar,
        Func<Vector<double>, Vector<double>>? vector = null)
    {
        var result = NdArray.Zeros(a.Shape.ToArray());
        var dst = result.AsSpan();

        if (a.IsContiguous)
        {
            var src = a.AsSpan();
            RunUnary(src, dst, scalar, vector);
        }
        else
        {
            for (var i = 0; i < a.Size; i++) dst[i] = scalar(a.At(i));
        }
        return result;
    }

    private static void RunUnary(
        ReadOnlySpan<double> src, Span<double> dst,
        Func<double, double> scalar, Func<Vector<double>, Vector<double>>? vector)
    {
        var n = src.Length;
        if (vector is not null && Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
        {
            var width = Vector<double>.Count;
            var i = 0;
            for (; i <= n - width; i += width)
            {
                var v = new Vector<double>(src.Slice(i, width));
                vector(v).CopyTo(dst.Slice(i, width));
            }
            for (; i < n; i++) dst[i] = scalar(src[i]);
            return;
        }

        for (var i = 0; i < n; i++) dst[i] = scalar(src[i]);
    }

    /// <summary>
    /// Applies a binary op with NumPy broadcast semantics.
    /// </summary>
    public static NdArray Binary(
        NdArray a, NdArray b,
        Func<double, double, double> scalar,
        Func<Vector<double>, Vector<double>, Vector<double>>? vector = null)
    {
        // Fast path: identical shapes, both contiguous.
        if (a.IsContiguous && b.IsContiguous && SameShape(a, b))
        {
            var result = NdArray.Zeros(a.Shape.ToArray());
            RunBinaryContiguous(a.AsSpan(), b.AsSpan(), result.AsSpan(), scalar, vector);
            return result;
        }

        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var va = a.BroadcastTo(shape);
        var vb = b.BroadcastTo(shape);
        var output = NdArray.Zeros(shape);
        var dst = output.AsSpan();

        var size = output.Size;
        if (size >= ParallelThreshold)
        {
            var buffer = output.Buffer;
            Parallel.For(0, Environment.ProcessorCount, worker =>
            {
                var (from, to) = Partition(size, worker);
                for (var i = from; i < to; i++) buffer[i] = scalar(va.At(i), vb.At(i));
            });
        }
        else
        {
            for (var i = 0; i < size; i++) dst[i] = scalar(va.At(i), vb.At(i));
        }
        return output;
    }

    private static void RunBinaryContiguous(
        ReadOnlySpan<double> left, ReadOnlySpan<double> right, Span<double> dst,
        Func<double, double, double> scalar,
        Func<Vector<double>, Vector<double>, Vector<double>>? vector)
    {
        var n = left.Length;

        if (n >= ParallelThreshold && vector is not null && Vector.IsHardwareAccelerated)
        {
            // Copy to arrays so the lambda can capture them; spans cannot cross the closure.
            var la = left.ToArray();
            var ra = right.ToArray();
            var oa = new double[n];
            Parallel.For(0, Environment.ProcessorCount, worker =>
            {
                var (from, to) = Partition(n, worker);
                SimdBinary(la.AsSpan(from, to - from), ra.AsSpan(from, to - from), oa.AsSpan(from, to - from), scalar, vector);
            });
            oa.AsSpan().CopyTo(dst);
            return;
        }

        if (vector is not null && Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
        {
            SimdBinary(left, right, dst, scalar, vector);
            return;
        }

        for (var i = 0; i < n; i++) dst[i] = scalar(left[i], right[i]);
    }

    private static void SimdBinary(
        ReadOnlySpan<double> left, ReadOnlySpan<double> right, Span<double> dst,
        Func<double, double, double> scalar,
        Func<Vector<double>, Vector<double>, Vector<double>> vector)
    {
        var width = Vector<double>.Count;
        var n = left.Length;
        var i = 0;
        for (; i <= n - width; i += width)
        {
            var l = new Vector<double>(left.Slice(i, width));
            var r = new Vector<double>(right.Slice(i, width));
            vector(l, r).CopyTo(dst.Slice(i, width));
        }
        for (; i < n; i++) dst[i] = scalar(left[i], right[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (int From, int To) Partition(int total, int worker)
    {
        var workers = Environment.ProcessorCount;
        var chunk = (total + workers - 1) / workers;
        var from = worker * chunk;
        var to = Math.Min(from + chunk, total);
        return (Math.Min(from, total), to);
    }

    private static bool SameShape(NdArray a, NdArray b)
    {
        if (a.Rank != b.Rank) return false;
        for (var i = 0; i < a.Rank; i++) if (a.Shape[i] != b.Shape[i]) return false;
        return true;
    }

    // ---------------------------------------------------------------- arithmetic

    /// <summary>Element-wise sum.</summary>
    public static NdArray Add(NdArray a, NdArray b) => Binary(a, b, static (x, y) => x + y, static (x, y) => x + y);

    /// <summary>Element-wise difference.</summary>
    public static NdArray Subtract(NdArray a, NdArray b) => Binary(a, b, static (x, y) => x - y, static (x, y) => x - y);

    /// <summary>Element-wise product (Hadamard).</summary>
    public static NdArray Multiply(NdArray a, NdArray b) => Binary(a, b, static (x, y) => x * y, static (x, y) => x * y);

    /// <summary>Element-wise quotient.</summary>
    public static NdArray Divide(NdArray a, NdArray b) => Binary(a, b, static (x, y) => x / y, static (x, y) => x / y);

    /// <summary>Element-wise minimum.</summary>
    public static NdArray Minimum(NdArray a, NdArray b)
        => Binary(a, b, Math.Min, static (x, y) => Vector.Min(x, y));

    /// <summary>Element-wise maximum.</summary>
    public static NdArray Maximum(NdArray a, NdArray b)
        => Binary(a, b, Math.Max, static (x, y) => Vector.Max(x, y));

    /// <summary>Element-wise power.</summary>
    public static NdArray Power(NdArray a, NdArray b) => Binary(a, b, Math.Pow);

    /// <summary>Element-wise remainder.</summary>
    public static NdArray Modulo(NdArray a, NdArray b) => Binary(a, b, static (x, y) => x % y);

    /// <summary>Adds a scalar to every element.</summary>
    public static NdArray AddScalar(NdArray a, double s) => Unary(a, x => x + s, v => v + new Vector<double>(s));

    /// <summary>Scales every element.</summary>
    public static NdArray MultiplyScalar(NdArray a, double s) => Unary(a, x => x * s, v => v * new Vector<double>(s));

    /// <summary>Raises every element to <paramref name="exponent"/>.</summary>
    public static NdArray PowerScalar(NdArray a, double exponent) => Unary(a, x => Math.Pow(x, exponent));

    // ---------------------------------------------------------------- math

    /// <summary>Element-wise negation.</summary>
    public static NdArray Negate(NdArray a) => Unary(a, static x => -x, static v => -v);

    /// <summary>Element-wise absolute value.</summary>
    public static NdArray Abs(NdArray a) => Unary(a, Math.Abs, static v => Vector.Abs(v));

    /// <summary>Element-wise square root.</summary>
    public static NdArray Sqrt(NdArray a) => Unary(a, Math.Sqrt, static v => Vector.SquareRoot(v));

    /// <summary>Element-wise square.</summary>
    public static NdArray Square(NdArray a) => Unary(a, static x => x * x, static v => v * v);

    /// <summary>Element-wise natural exponential.</summary>
    public static NdArray Exp(NdArray a) => Unary(a, Math.Exp);

    /// <summary>Element-wise natural logarithm.</summary>
    public static NdArray Log(NdArray a) => Unary(a, Math.Log);

    /// <summary>Element-wise base-2 logarithm.</summary>
    public static NdArray Log2(NdArray a) => Unary(a, Math.Log2);

    /// <summary>Element-wise base-10 logarithm.</summary>
    public static NdArray Log10(NdArray a) => Unary(a, Math.Log10);

    /// <summary>Element-wise <c>log(1 + x)</c>, accurate for small inputs.</summary>
    public static NdArray Log1p(NdArray a) => Unary(a, x => Math.Log(1.0 + x));

    /// <summary>Element-wise sine.</summary>
    public static NdArray Sin(NdArray a) => Unary(a, Math.Sin);

    /// <summary>Element-wise cosine.</summary>
    public static NdArray Cos(NdArray a) => Unary(a, Math.Cos);

    /// <summary>Element-wise tangent.</summary>
    public static NdArray Tan(NdArray a) => Unary(a, Math.Tan);

    /// <summary>Element-wise hyperbolic tangent.</summary>
    public static NdArray Tanh(NdArray a) => Unary(a, Math.Tanh);

    /// <summary>Element-wise logistic sigmoid.</summary>
    public static NdArray Sigmoid(NdArray a) => Unary(a, MathUtil.Sigmoid);

    /// <summary>Element-wise rectified linear unit.</summary>
    public static NdArray Relu(NdArray a) => Unary(a, static x => x > 0 ? x : 0, v => Vector.Max(v, Vector<double>.Zero));

    /// <summary>Element-wise floor.</summary>
    public static NdArray Floor(NdArray a) => Unary(a, Math.Floor);

    /// <summary>Element-wise ceiling.</summary>
    public static NdArray Ceiling(NdArray a) => Unary(a, Math.Ceiling);

    /// <summary>Element-wise rounding to <paramref name="digits"/> decimals.</summary>
    public static NdArray Round(NdArray a, int digits = 0) => Unary(a, x => Math.Round(x, digits));

    /// <summary>Element-wise sign (-1, 0 or 1).</summary>
    public static NdArray Sign(NdArray a) => Unary(a, static x => Math.Sign(x));

    /// <summary>Element-wise reciprocal.</summary>
    public static NdArray Reciprocal(NdArray a) => Unary(a, static x => 1.0 / x, static v => Vector<double>.One / v);

    /// <summary>Clamps every element into <c>[min, max]</c>.</summary>
    public static NdArray Clip(NdArray a, double min, double max)
        => Unary(a, x => Math.Clamp(x, min, max),
            v => Vector.Min(Vector.Max(v, new Vector<double>(min)), new Vector<double>(max)));

    // ---------------------------------------------------------------- comparison

    /// <summary>Element-wise equality, as 1.0 / 0.0.</summary>
    public static NdArray Equal(NdArray a, NdArray b, double tolerance = 0)
        => Binary(a, b, (x, y) => Math.Abs(x - y) <= tolerance ? 1.0 : 0.0);

    /// <summary>Element-wise "greater than", as 1.0 / 0.0.</summary>
    public static NdArray Greater(NdArray a, NdArray b) => Binary(a, b, static (x, y) => x > y ? 1.0 : 0.0);

    /// <summary>Element-wise "less than", as 1.0 / 0.0.</summary>
    public static NdArray Less(NdArray a, NdArray b) => Binary(a, b, static (x, y) => x < y ? 1.0 : 0.0);

    /// <summary>True when every pair of elements agrees within <paramref name="tolerance"/>.</summary>
    public static bool AllClose(NdArray a, NdArray b, double tolerance = 1e-9)
    {
        if (!Shapes.CanBroadcast(a.Shape, b.Shape)) return false;
        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var va = a.BroadcastTo(shape);
        var vb = b.BroadcastTo(shape);
        for (var i = 0; i < va.Size; i++)
        {
            var x = va.At(i);
            var y = vb.At(i);
            if (double.IsNaN(x) && double.IsNaN(y)) continue;
            if (Math.Abs(x - y) > tolerance) return false;
        }
        return true;
    }

    /// <summary>Chooses element-wise between two arrays based on a mask.</summary>
    public static NdArray Select(bool[] condition, NdArray ifTrue, NdArray ifFalse)
    {
        var shape = Shapes.Broadcast(ifTrue.Shape, ifFalse.Shape);
        var a = ifTrue.BroadcastTo(shape);
        var b = ifFalse.BroadcastTo(shape);
        var result = NdArray.Zeros(shape);
        for (var i = 0; i < result.Size; i++) result.SetAt(i, condition[i] ? a.At(i) : b.At(i));
        return result;
    }
}
