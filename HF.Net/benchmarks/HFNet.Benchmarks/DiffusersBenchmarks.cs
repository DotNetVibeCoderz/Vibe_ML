using BenchmarkDotNet.Attributes;
using Gravicode.HFNet.GraviDiffusers;
using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.Benchmarks;

/// <summary>GraviDiffusers: pipeline latency, and what the scheduler itself costs per step.</summary>
/// <remarks>
/// The pipeline is <c>optimum-internal-testing/tiny-stable-diffusion-onnx</c>, a 9 MB export with
/// random weights: the image is noise, but every stage runs, so the measurement is of the pipeline's
/// own overhead around the networks. A full Stable Diffusion run is dominated by the UNet in ONNX
/// Runtime. This build uses the CPU package of ONNX Runtime; a GPU needs the CUDA or DirectML one.
/// </remarks>
[BenchmarkCategory("GraviDiffusers")]
public class DiffusersBenchmarks
{
    private DiffusionPipeline _pipeline = null!;
    private NdArray _sample = null!;
    private NdArray _noise = null!;

    [GlobalSetup]
    public void Setup()
    {
        _pipeline = DiffusionPipeline.FromPretrained("optimum-internal-testing/tiny-stable-diffusion-onnx", sampler: Sampler.Euler);

        var random = new GraviRandom(2);
        _sample = new NdArray([.. Enumerable.Range(0, 4 * 64 * 64).Select(_ => random.Normal(0, 1))], 1, 4, 64, 64);
        _noise = new NdArray([.. Enumerable.Range(0, 4 * 64 * 64).Select(_ => random.Normal(0, 1))], 1, 4, 64, 64);
    }

    [GlobalCleanup]
    public void Cleanup() => _pipeline.Dispose();

    [Benchmark(Baseline = true, Description = "Tiny pipeline, 10 steps, 64x64, with guidance")]
    public int Generate()
    {
        using var image = _pipeline.Generate("a lighthouse at dawn", new GenerationOptions(Steps: 10, Width: 64, Height: 64, Seed: 1));
        return image.Width;
    }

    [Benchmark(Description = "25 DDIM steps on a 64x64 SD latent")]
    public NdArray Ddim() => RunScheduler(new DdimScheduler());

    [Benchmark(Description = "25 Euler steps on a 64x64 SD latent")]
    public NdArray Euler() => RunScheduler(new EulerScheduler());

    private NdArray RunScheduler(IScheduler scheduler)
    {
        scheduler.SetTimesteps(25);
        var sample = _sample;
        for (var step = 0; step < 25; step++) sample = scheduler.Step(_noise, step, scheduler.ScaleInput(sample, step));
        return sample;
    }
}
