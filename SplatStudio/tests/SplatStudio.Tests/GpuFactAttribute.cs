using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace SplatStudio.Tests;

/// <summary>
/// A <see cref="FactAttribute"/> that skips instead of failing when the machine has no
/// CUDA/OpenCL device. The GPU engine is an optional accelerator, not a requirement, so a
/// GPU-less CI agent should report these as skipped rather than red — but on a machine that
/// does have a device they run for real.
/// </summary>
public sealed class GpuFactAttribute : FactAttribute
{
    public GpuFactAttribute()
    {
        if (!GpuProbe.IsAvailable)
            Skip = $"No CUDA/OpenCL device available ({GpuProbe.Detail}).";
    }
}

/// <summary>Probes once per test run for a usable accelerator.</summary>
public static class GpuProbe
{
    private static readonly Lazy<(bool Available, string Detail)> Probe = new(() =>
    {
        try
        {
            using var context = Context.Create(b => b.Cuda().OpenCL().EnableAlgorithms());
            var device = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda)
                         ?? context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.OpenCL);

            return device is null
                ? (false, "no CUDA or OpenCL device enumerated")
                : (true, $"{device.Name} [{device.AcceleratorType}]");
        }
        catch (Exception ex)
        {
            return (false, ex.Message);
        }
    });

    public static bool IsAvailable => Probe.Value.Available;
    public static string Detail => Probe.Value.Detail;
}
