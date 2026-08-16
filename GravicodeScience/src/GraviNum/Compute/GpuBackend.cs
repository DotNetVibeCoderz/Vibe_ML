using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Gravicode.Science.GraviNum.Compute;

/// <summary>
/// GPU execution through ILGPU, which targets CUDA, OpenCL or a JIT-compiled CPU accelerator.
/// </summary>
/// <remarks>
/// Construction is the risky part - a machine may have no accelerator, or one without float64
/// support - so it is funnelled through <see cref="TryCreate"/>, which never throws. Kernels are
/// compiled once on first use and cached for the lifetime of the backend, because ILGPU's JIT
/// costs far more than any single launch.
/// <para>
/// Transfers dominate for small problems: <see cref="EfficientThreshold"/> reports the size below
/// which the CPU backend wins, and <see cref="Compute.Best"/> uses it to route work automatically.
/// </para>
/// </remarks>
public sealed class GpuBackend : IComputeBackend
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;

    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>> _addKernel;
    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>> _mulKernel;
    private readonly Action<Index1D, double, ArrayView<double>, ArrayView<double>, ArrayView<double>> _axpyKernel;
    private readonly Action<Index2D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int> _matMulKernel;

    private bool _disposed;

    private GpuBackend(Context context, Accelerator accelerator)
    {
        _context = context;
        _accelerator = accelerator;

        _addKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(AddKernel);
        _mulKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(MultiplyKernel);
        _axpyKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, double, ArrayView<double>, ArrayView<double>, ArrayView<double>>(AxpyKernel);
        _matMulKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int>(MatMulKernel);
    }

    /// <summary>
    /// Attempts to bring up a GPU backend. Returns <c>false</c> instead of throwing when no
    /// usable accelerator exists, which is the normal case on CI machines and laptops.
    /// </summary>
    /// <param name="backend">The initialised backend, or <c>null</c>.</param>
    /// <param name="preferCpuAccelerator">
    /// When true, ILGPU's software accelerator is acceptable. Useful for testing kernels without
    /// GPU hardware; useless for performance.
    /// </param>
    public static bool TryCreate(out GpuBackend? backend, bool preferCpuAccelerator = false)
    {
        backend = null;
        Context? context = null;
        Accelerator? accelerator = null;

        try
        {
            context = Context.Create(builder => builder.Default().EnableAlgorithms());

            var device = context.GetPreferredDevice(preferCPU: preferCpuAccelerator);
            if (device is null) { context.Dispose(); return false; }

            accelerator = device.CreateAccelerator(context);
            backend = new GpuBackend(context, accelerator);
            return true;
        }
        catch (Exception)
        {
            // Missing drivers, an unsupported device, a JIT failure: all mean "no GPU today".
            accelerator?.Dispose();
            context?.Dispose();
            backend = null;
            return false;
        }
    }

    /// <inheritdoc />
    public string Name => $"{_accelerator.AcceleratorType} - {_accelerator.Name} ({_accelerator.MemorySize / (1024 * 1024)} MB)";

    /// <inheritdoc />
    public bool IsGpu => _accelerator.AcceleratorType != AcceleratorType.CPU;

    /// <inheritdoc />
    public int EfficientThreshold => IsGpu ? 250_000 : int.MaxValue;

    /// <summary>The maximum number of threads the device can run in one group.</summary>
    public int MaxGroupSize => _accelerator.MaxNumThreadsPerGroup;

    /// <summary>Device memory in bytes.</summary>
    public long MemoryBytes => _accelerator.MemorySize;

    /// <summary>The accelerator kind actually selected (CUDA, OpenCL or CPU).</summary>
    public string DeviceKind => _accelerator.AcceleratorType.ToString();

    // ---------------------------------------------------------------- kernels

    private static void AddKernel(Index1D i, ArrayView<double> a, ArrayView<double> b, ArrayView<double> c)
        => c[i] = a[i] + b[i];

    private static void MultiplyKernel(Index1D i, ArrayView<double> a, ArrayView<double> b, ArrayView<double> c)
        => c[i] = a[i] * b[i];

    private static void AxpyKernel(Index1D i, double alpha, ArrayView<double> x, ArrayView<double> y, ArrayView<double> result)
        => result[i] = alpha * x[i] + y[i];

    private static void MatMulKernel(Index2D index, ArrayView<double> a, ArrayView<double> b, ArrayView<double> c, int k, int n)
    {
        // index.X walks the output rows, index.Y the output columns.
        var acc = 0.0;
        for (var p = 0; p < k; p++) acc += a[index.X * k + p] * b[p * n + index.Y];
        c[index.X * n + index.Y] = acc;
    }

    // ---------------------------------------------------------------- operations

    /// <inheritdoc />
    public double[] Add(double[] a, double[] b) => RunBinary(a, b, _addKernel);

    /// <inheritdoc />
    public double[] Multiply(double[] a, double[] b) => RunBinary(a, b, _mulKernel);

    /// <inheritdoc />
    public double[] Axpy(double alpha, double[] x, double[] y)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        using var dx = _accelerator.Allocate1D(x);
        using var dy = _accelerator.Allocate1D(y);
        using var dr = _accelerator.Allocate1D<double>(x.Length);
        _axpyKernel(x.Length, alpha, dx.View, dy.View, dr.View);
        _accelerator.Synchronize();
        return dr.GetAsArray1D();
    }

    private double[] RunBinary(double[] a, double[] b, Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>> kernel)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (a.Length != b.Length) throw new ArgumentException("Buffers must be the same length.");

        using var da = _accelerator.Allocate1D(a);
        using var db = _accelerator.Allocate1D(b);
        using var dc = _accelerator.Allocate1D<double>(a.Length);
        kernel(a.Length, da.View, db.View, dc.View);
        _accelerator.Synchronize();
        return dc.GetAsArray1D();
    }

    /// <inheritdoc />
    public double[] MatMul(double[] a, double[] b, int m, int k, int n)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        using var da = _accelerator.Allocate1D(a);
        using var db = _accelerator.Allocate1D(b);
        using var dc = _accelerator.Allocate1D<double>((long)m * n);
        _matMulKernel(new Index2D(m, n), da.View, db.View, dc.View, k, n);
        _accelerator.Synchronize();
        return dc.GetAsArray1D();
    }

    /// <inheritdoc />
    public double Sum(double[] a)
    {
        // Reductions of this size are latency-bound; the round trip costs more than the sum.
        var acc = 0.0;
        foreach (var v in a) acc += v;
        return acc;
    }

    /// <inheritdoc />
    public double Dot(double[] a, double[] b)
    {
        var products = Multiply(a, b);
        return Sum(products);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _accelerator.Dispose();
        _context.Dispose();
    }
}
