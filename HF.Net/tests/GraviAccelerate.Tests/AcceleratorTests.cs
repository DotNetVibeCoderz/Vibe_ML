using Gravicode.HFNet.GraviAccelerate;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.HFNet.GraviAccelerate.Tests;

/// <summary>Tests for device selection and sharding.</summary>
public sealed class AcceleratorTests
{
    [Fact]
    public void AlwaysReportsACpuDevice()
    {
        var devices = Accelerator.Devices();

        Assert.Contains(devices, d => d.Kind == DeviceKind.Cpu && d.Available);
    }

    [Fact]
    public void ProbingForAGpuNeverThrows()
    {
        // ILGPU initialisation can fail on a machine with a broken driver, and a missing
        // accelerator is a normal condition rather than an error worth propagating.
        var exception = Record.Exception(() => Accelerator.Describe());

        Assert.Null(exception);
    }

    [Fact]
    public void ExplicitCpuStaysOnTheCpuHoweverLargeTheWork()
    {
        var accelerator = new Accelerator(DeviceKind.Cpu);

        Assert.Same(
            accelerator.Backend(1),
            accelerator.Backend(Accelerator.GpuThreshold * 100));
    }

    [Fact]
    public void AutoKeepsSmallWorkOnTheCpu()
    {
        // Below the threshold the kernel launch and the round trip cost more than the arithmetic,
        // so Auto must not reach for an accelerator even when one is present.
        var accelerator = new Accelerator(DeviceKind.Auto);

        Assert.Equal(
            Gravicode.Science.GraviNum.Compute.Compute.Cpu,
            accelerator.Backend(1024));
    }

    [Fact]
    public void WorkerCountDefaultsToTheProcessorCount()
        => Assert.Equal(Environment.ProcessorCount, new Accelerator().Workers);

    [Fact]
    public void WorkerCountIsAtLeastOne()
        => Assert.Equal(1, new Accelerator(workers: 0).Workers);

    [Fact]
    public void PartitionCoversEveryItemExactlyOnce()
    {
        var shards = new Accelerator(workers: 3).Partition(10);

        Assert.Equal(10, shards.Sum(s => s.Count));

        var covered = shards.SelectMany(s => s.Indices).OrderBy(i => i).ToList();
        Assert.Equal(Enumerable.Range(0, 10), covered);
    }

    [Fact]
    public void PartitionProducesUnevenShardsWhenItMust()
    {
        // 10 over 3 workers cannot be even, which is exactly why gradient averaging has to be
        // weighted by shard size.
        var shards = new Accelerator(workers: 3).Partition(10);

        Assert.Contains(shards, s => s.Count != shards[0].Count);
    }

    [Fact]
    public void ParallelGradientWeightedByShardSizeEqualsTheGlobalMean()
    {
        // The property that makes data parallelism correct. An unweighted average of per-worker
        // means is not the global mean once the shards differ in size, and nothing downstream
        // notices - the model just trains to something slightly wrong.
        var values = Enumerable.Range(0, 10).Select(i => (double)i).ToArray();

        var accelerator = new Accelerator(workers: 3);

        var averaged = accelerator.ParallelGradient(values.Length, shard =>
        {
            var total = 0.0;
            foreach (var index in shard.Indices) total += values[index];
            return new NdArray([total / shard.Count], 1);
        });

        Assert.True(Math.Abs(averaged.At(0) - values.Average()) < 1e-12,
            $"got {averaged.At(0)}, expected {values.Average()}");
    }

    [Fact]
    public void ParallelGradientIsReproducibleAcrossRuns()
    {
        // Collected in worker order rather than completion order: floating-point addition is not
        // associative, so arrival order would make the answer depend on thread scheduling.
        var accelerator = new Accelerator(workers: 4);

        NdArray Run() => accelerator.ParallelGradient(1000, shard =>
        {
            var total = 0.0;
            foreach (var index in shard.Indices) total += Math.Sin(index) * 1e-8;
            return new NdArray([total], 1);
        });

        Assert.Equal(Run().At(0), Run().At(0));
    }

    [Fact]
    public void MeasureReturnsTheBestOfSeveralRuns()
    {
        var calls = 0;
        var elapsed = Accelerator.Measure(() => calls++, iterations: 5, warmup: 3);

        Assert.Equal(8, calls);
        Assert.True(elapsed >= TimeSpan.Zero);
    }

    [Fact]
    public void DotAgreesWithTheCpuBackend()
    {
        var a = new NdArray([1, 2, 3, 4], 2, 2);
        var b = new NdArray([5, 6, 7, 8], 2, 2);

        var result = new Accelerator(DeviceKind.Cpu).Dot(a, b);

        // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
        Assert.Equal([19, 22, 43, 50], result.ToArray());
    }

    [Fact]
    public void TrainingReportComputesThroughput()
    {
        var report = new TrainingReport(2, 1000, TimeSpan.FromSeconds(2), 4, DeviceKind.Cpu, 0.9);

        Assert.Equal(500, report.Throughput);
    }

    [Fact]
    public void TrainingReportHandlesAZeroDuration()
    {
        var report = new TrainingReport(1, 10, TimeSpan.Zero, 1, DeviceKind.Cpu, 1.0);

        Assert.Equal(0, report.Throughput);
    }
}
