using System.Numerics;

namespace Gravicode.Science.GraviNum.Compute;

/// <summary>
/// The always-available backend: <see cref="Vector{T}"/> SIMD kernels spread over the thread pool.
/// </summary>
public sealed class CpuBackend : IComputeBackend
{
    /// <inheritdoc />
    public string Name =>
        $"CPU ({(Vector.IsHardwareAccelerated ? $"SIMD x{Vector<double>.Count}" : "scalar")}, {Environment.ProcessorCount} threads)";

    /// <inheritdoc />
    public bool IsGpu => false;

    /// <inheritdoc />
    public int EfficientThreshold => 0;

    /// <inheritdoc />
    public double[] Add(double[] a, double[] b)
    {
        var result = new double[a.Length];
        RunVectorised(a, b, result, static (x, y) => x + y, static (x, y) => x + y);
        return result;
    }

    /// <inheritdoc />
    public double[] Multiply(double[] a, double[] b)
    {
        var result = new double[a.Length];
        RunVectorised(a, b, result, static (x, y) => x * y, static (x, y) => x * y);
        return result;
    }

    /// <inheritdoc />
    public double[] Axpy(double alpha, double[] x, double[] y)
    {
        var result = (double[])y.Clone();
        LinAlg.AxpySpan(alpha, x, result);
        return result;
    }

    /// <inheritdoc />
    public double[] MatMul(double[] a, double[] b, int m, int k, int n)
    {
        var c = new double[(long)m * n <= int.MaxValue ? m * n : throw new ArgumentException("Result too large.")];
        Parallel.For(0, m, i =>
        {
            var row = c.AsSpan(i * n, n);
            for (var p = 0; p < k; p++)
            {
                var aik = a[i * k + p];
                if (aik == 0.0) continue;
                LinAlg.AxpySpan(aik, b.AsSpan(p * n, n), row);
            }
        });
        return c;
    }

    /// <inheritdoc />
    public double Sum(double[] a)
    {
        var acc = 0.0;
        var i = 0;
        if (Vector.IsHardwareAccelerated && a.Length >= Vector<double>.Count)
        {
            var width = Vector<double>.Count;
            var vsum = Vector<double>.Zero;
            for (; i <= a.Length - width; i += width) vsum += new Vector<double>(a.AsSpan(i, width));
            acc = Vector.Sum(vsum);
        }
        for (; i < a.Length; i++) acc += a[i];
        return acc;
    }

    /// <inheritdoc />
    public double Dot(double[] a, double[] b) => LinAlg.InnerSpan(a, b);

    private static void RunVectorised(
        double[] a, double[] b, double[] result,
        Func<double, double, double> scalar,
        Func<Vector<double>, Vector<double>, Vector<double>> vector)
    {
        var n = a.Length;
        if (n >= UFunc.ParallelThreshold)
        {
            Parallel.For(0, Environment.ProcessorCount, worker =>
            {
                var chunk = (n + Environment.ProcessorCount - 1) / Environment.ProcessorCount;
                var from = worker * chunk;
                var to = Math.Min(from + chunk, n);
                if (from >= to) return;
                Kernel(a.AsSpan(from, to - from), b.AsSpan(from, to - from), result.AsSpan(from, to - from), scalar, vector);
            });
            return;
        }
        Kernel(a, b, result, scalar, vector);
    }

    private static void Kernel(
        ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result,
        Func<double, double, double> scalar,
        Func<Vector<double>, Vector<double>, Vector<double>> vector)
    {
        var i = 0;
        if (Vector.IsHardwareAccelerated && a.Length >= Vector<double>.Count)
        {
            var width = Vector<double>.Count;
            for (; i <= a.Length - width; i += width)
                vector(new Vector<double>(a.Slice(i, width)), new Vector<double>(b.Slice(i, width)))
                    .CopyTo(result.Slice(i, width));
        }
        for (; i < a.Length; i++) result[i] = scalar(a[i], b[i]);
    }

    /// <inheritdoc />
    public void Dispose() { }
}
