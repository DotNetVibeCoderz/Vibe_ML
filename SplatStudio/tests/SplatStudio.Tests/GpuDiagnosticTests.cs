using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SplatStudio.Infrastructure.Splatting;
using Xunit.Abstractions;

namespace SplatStudio.Tests;

/// <summary>
/// Reproduces GpuSplatEngine's initialisation one step at a time so a failure names the
/// exact stage (context, accelerator, or kernel JIT) instead of surfacing as a bare
/// "GPU unavailable". Kept in the suite because ILGPU/driver mismatches are the most
/// likely way this engine breaks on a new machine.
/// </summary>
public class GpuDiagnosticTests
{
    private readonly ITestOutputHelper _output;

    public GpuDiagnosticTests(ITestOutputHelper output) => _output = output;

    [GpuFact]
    public void Each_initialisation_stage_succeeds()
    {
        using var context = Context.Create(b => b.Cuda().OpenCL().EnableAlgorithms());
        _output.WriteLine($"Context created. Devices: {context.Devices.Length}");
        foreach (var d in context.Devices)
            _output.WriteLine($"  - {d.Name} [{d.AcceleratorType}]");

        var device = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda)
                     ?? context.Devices.First(d => d.AcceleratorType == AcceleratorType.OpenCL);

        using var accelerator = device.CreateAccelerator(context);
        _output.WriteLine($"Accelerator: {accelerator.Name} / {accelerator.AcceleratorType}, " +
                          $"{accelerator.MemorySize / (1024 * 1024)} MB");

        var depthKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<uint>, ArrayView<float>, int, int>(GpuSplatEngine.DepthKernel);
        _output.WriteLine("Depth kernel compiled.");

        var emitKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<uint>, ArrayView<float>, ArrayView<GpuSplatRecord>,
            ArrayView<int>, int, int, float, float>(GpuSplatEngine.BlurAndEmitKernel);
        _output.WriteLine("Emit kernel compiled.");

        // A 4x4 all-white image: every pixel is opaque, so every pixel must emit a splat.
        const int w = 4, h = 4;
        var pixels = Enumerable.Repeat(0xFFFFFFFFu, w * h).ToArray();

        using var pixelBuffer = accelerator.Allocate1D<uint>(pixels.Length);
        using var depthBuffer = accelerator.Allocate1D<float>(pixels.Length);
        using var outputBuffer = accelerator.Allocate1D<GpuSplatRecord>(pixels.Length);
        using var counterBuffer = accelerator.Allocate1D<int>(1);

        pixelBuffer.CopyFromCPU(pixels);
        counterBuffer.MemSetToZero();

        depthKernel(pixels.Length, pixelBuffer.View, depthBuffer.View, w, h);
        emitKernel(pixels.Length, pixelBuffer.View, depthBuffer.View, outputBuffer.View,
            counterBuffer.View, w, h, 0.5f, 0.6f);
        accelerator.Synchronize();

        var counter = new int[1];
        counterBuffer.CopyToCPU(counter);
        _output.WriteLine($"Splats emitted: {counter[0]} (expected {w * h})");

        Assert.Equal(w * h, counter[0]);
    }
}
