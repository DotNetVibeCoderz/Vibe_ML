using BenchmarkDotNet.Attributes;
using Gravicode.HFNet.GraviAccelerate;
using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.Benchmarks;

/// <summary>GraviAccelerate: training speed as data-parallel workers are added.</summary>
/// <remarks>
/// One logistic-regression gradient over 400,000 rows of 32 features, sharded across workers and
/// averaged by shard size - the reduction a multi-device trainer performs. The ratio column is the
/// scaling; past the physical core count, hyperthreads add little to arithmetic this dense.
/// </remarks>
[BenchmarkCategory("GraviAccelerate")]
public class AccelerateBenchmarks
{
    private const int Rows = 400_000, Features = 32;

    private double[] _x = [];
    private double[] _y = [];
    private readonly double[] _weights = new double[Features];
    private Accelerator _accelerator = null!;

    [Params(1, 2, 4, 8)]
    public int Workers { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = new GraviRandom(1);
        _x = new double[Rows * Features];
        _y = new double[Rows];
        for (var i = 0; i < _x.Length; i++) _x[i] = random.Normal(0, 1);
        for (var i = 0; i < Rows; i++) _y[i] = _x[i * Features] > 0 ? 1 : 0;
        _accelerator = new Accelerator(DeviceKind.Cpu, Workers);
    }

    [Benchmark(Description = "Data-parallel gradient, 400k rows")]
    public NdArray Gradient() => _accelerator.ParallelGradient(Rows, shard =>
    {
        var gradient = new double[Features];
        for (var r = shard.Start; r < shard.Start + shard.Count; r++)
        {
            var z = 0.0;
            for (var f = 0; f < Features; f++) z += _x[r * Features + f] * _weights[f];
            var error = 1 / (1 + Math.Exp(-z)) - _y[r];
            for (var f = 0; f < Features; f++) gradient[f] += error * _x[r * Features + f];
        }

        for (var f = 0; f < Features; f++) gradient[f] /= shard.Count;
        return new NdArray(gradient, Features);
    });
}
