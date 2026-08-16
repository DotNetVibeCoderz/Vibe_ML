namespace Gravicode.Science.GraviNum.Compute;

/// <summary>
/// A hardware target that can execute the numeric kernels the ecosystem leans on.
/// </summary>
/// <remarks>
/// Every performance-sensitive operation in Gravicode.Science is expressed against this interface
/// so a caller can move work between CPU SIMD and the GPU without changing algorithm code. The
/// contract is deliberately narrow: these are the kernels where dispatch overhead is worth paying.
/// </remarks>
public interface IComputeBackend : IDisposable
{
    /// <summary>Human-readable device name, e.g. <c>CPU (SIMD x4, 16 threads)</c>.</summary>
    string Name { get; }

    /// <summary>True when the work actually runs on a GPU.</summary>
    bool IsGpu { get; }

    /// <summary>
    /// Element count below which this backend is not worth using. GPU backends report a large
    /// value because the transfer cost dominates for small arrays.
    /// </summary>
    int EfficientThreshold { get; }

    /// <summary>Element-wise sum of two equally sized buffers.</summary>
    double[] Add(double[] a, double[] b);

    /// <summary>Element-wise product of two equally sized buffers.</summary>
    double[] Multiply(double[] a, double[] b);

    /// <summary>Element-wise <c>alpha * x + y</c>.</summary>
    double[] Axpy(double alpha, double[] x, double[] y);

    /// <summary>Row-major matrix product of an <c>m×k</c> and a <c>k×n</c> matrix.</summary>
    double[] MatMul(double[] a, double[] b, int m, int k, int n);

    /// <summary>Sum of every element.</summary>
    double Sum(double[] a);

    /// <summary>Inner product of two equally sized buffers.</summary>
    double Dot(double[] a, double[] b);
}
