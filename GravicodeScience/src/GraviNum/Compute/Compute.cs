namespace Gravicode.Science.GraviNum.Compute;

/// <summary>
/// The entry point for hardware selection: exposes the CPU backend, lazily brings up a GPU one,
/// and routes work to whichever is actually faster for a given problem size.
/// </summary>
/// <remarks>
/// GPU initialisation happens once, on first access to <see cref="Gpu"/>, and failure is a normal
/// outcome rather than an error - a machine with no accelerator simply reports
/// <see cref="IsGpuAvailable"/> as false and everything keeps running on the CPU path.
/// </remarks>
public static class Compute
{
    private static readonly Lazy<GpuBackend?> LazyGpu = new(() =>
    {
        GpuBackend.TryCreate(out var backend);
        return backend;
    }, isThreadSafe: true);

    /// <summary>The SIMD + thread-pool CPU backend. Always available.</summary>
    public static IComputeBackend Cpu { get; } = new CpuBackend();

    /// <summary>The GPU backend, or <c>null</c> when no usable accelerator was found.</summary>
    public static IComputeBackend? Gpu => LazyGpu.Value;

    /// <summary>True when a GPU backend came up successfully.</summary>
    public static bool IsGpuAvailable => LazyGpu.Value is not null;

    /// <summary>
    /// Backend override. Set it to force every routed operation onto one device.
    /// </summary>
    public static IComputeBackend? Preferred { get; set; }

    /// <summary>
    /// Opts in to automatic GPU dispatch for large problems. Off by default.
    /// </summary>
    /// <remarks>
    /// Automatic GPU selection is opt-in because a GPU is not reliably faster for this workload.
    /// Everything here is float64, and integrated GPUs run double precision at a small fraction
    /// of their single-precision rate - measured on an Intel UHD 620, a 1024x1024 product takes
    /// about 940 ms on the GPU against 132 ms on the CPU. A discrete compute card reverses that,
    /// but the library cannot tell which one it is looking at, and silently choosing a backend
    /// that is seven times slower is worse than not choosing at all.
    /// <para>
    /// Benchmark <c>benchmarks/GraviNum.Benchmark</c> on the target machine, then either enable
    /// this flag or set <see cref="Preferred"/> explicitly.
    /// </para>
    /// </remarks>
    public static bool AutomaticGpuDispatch { get; set; }

    /// <summary>
    /// Picks the backend for a problem of <paramref name="elementCount"/> elements. Returns the
    /// CPU unless <see cref="Preferred"/> is set, or <see cref="AutomaticGpuDispatch"/> is enabled
    /// and the problem is large enough to hide the transfer cost.
    /// </summary>
    public static IComputeBackend Best(int elementCount)
    {
        if (Preferred is not null) return Preferred;
        if (!AutomaticGpuDispatch) return Cpu;

        var gpu = LazyGpu.Value;
        if (gpu is not null && gpu.IsGpu && elementCount >= gpu.EfficientThreshold) return gpu;
        return Cpu;
    }

    /// <summary>A one-line description of every backend available on this machine.</summary>
    public static string DescribeDevices()
    {
        var gpu = LazyGpu.Value;
        return gpu is null
            ? $"CPU: {Cpu.Name}; GPU: not available"
            : $"CPU: {Cpu.Name}; GPU: {gpu.Name}";
    }

    /// <summary>
    /// Matrix product routed through the best available backend.
    /// </summary>
    public static NdArray Dot(NdArray a, NdArray b, IComputeBackend? backend = null)
    {
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException("Compute.Dot expects two rank 2 arrays.");
        if (a.Shape[1] != b.Shape[0])
            throw new InvalidOperationException(
                $"Shapes ({Shapes.Describe(a.Shape)}) and ({Shapes.Describe(b.Shape)}) are not aligned.");

        var m = a.Shape[0];
        var k = a.Shape[1];
        var n = b.Shape[1];
        var chosen = backend ?? Best(m * n);

        var result = chosen.MatMul(a.AsContiguous().ToArray(), b.AsContiguous().ToArray(), m, k, n);
        return new NdArray(result, m, n);
    }

    /// <summary>Element-wise sum routed through the best available backend.</summary>
    public static NdArray Add(NdArray a, NdArray b, IComputeBackend? backend = null)
    {
        var chosen = backend ?? Best(a.Size);
        if (chosen is CpuBackend) return UFunc.Add(a, b);
        return new NdArray(chosen.Add(a.ToArray(), b.ToArray()), a.Shape.ToArray());
    }

    /// <summary>Element-wise product routed through the best available backend.</summary>
    public static NdArray Multiply(NdArray a, NdArray b, IComputeBackend? backend = null)
    {
        var chosen = backend ?? Best(a.Size);
        if (chosen is CpuBackend) return UFunc.Multiply(a, b);
        return new NdArray(chosen.Multiply(a.ToArray(), b.ToArray()), a.Shape.ToArray());
    }

    /// <summary>Releases the GPU backend if one was created.</summary>
    public static void ReleaseGpu()
    {
        if (LazyGpu.IsValueCreated) LazyGpu.Value?.Dispose();
    }
}
