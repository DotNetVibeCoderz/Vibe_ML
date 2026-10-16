using System.Numerics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Gravicode.Science.SciNet.Spatial;

BenchmarkSwitcher.FromAssembly(typeof(FftEngineBenchmark).Assembly).Run(args, DefaultConfig.Instance
    .AddJob(Job.ShortRun.WithWarmupCount(3).WithIterationCount(5))
    .WithOptions(ConfigOptions.DisableOptimizationsValidator));
return;

/// <summary>
/// SciNet's FFT engine against GraviNum's, and its SIMD path against its own scalar path.
/// </summary>
/// <remarks>
/// GraviNum's transform is radix-2 plus Bluestein; SciNet's is a cached mixed-radix Stockham plan.
/// Length 1000 is the interesting row: it is not a power of two, so GraviNum takes the Bluestein
/// detour and SciNet does not. The SIMD/scalar pair isolates what the vector butterflies buy,
/// which shrinks as the transform becomes memory-bound.
/// </remarks>
[MemoryDiagnoser]
public class FftEngineBenchmark
{
    private Complex[] _signal = [];
    private Complex[] _buffer = [];

    /// <summary>Transform length.</summary>
    [Params(1000, 1024, 65536)]
    public int Length { get; set; }

    /// <summary>Builds a deterministic complex signal.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var rng = new GraviRandom(42);
        _signal = Enumerable.Range(0, Length).Select(_ => new Complex(rng.Normal(), rng.Normal())).ToArray();
        _buffer = new Complex[Length];
    }

    /// <summary>GraviNum's FFT.</summary>
    [Benchmark(Baseline = true)]
    public Complex[] GraviNum() => Gravicode.Science.GraviNum.Signal.Fft.Forward(_signal);

    /// <summary>SciNet's cached plan, SIMD butterflies.</summary>
    [Benchmark]
    public Complex[] SciNetSimd()
    {
        FftPlan.SimdEnabled = true;
        FftPlan.Get(Length).Execute(_signal, _buffer);
        return _buffer;
    }

    /// <summary>SciNet's cached plan, scalar reference butterflies.</summary>
    [Benchmark]
    public Complex[] SciNetScalar()
    {
        FftPlan.SimdEnabled = false;
        FftPlan.Get(Length).Execute(_signal, _buffer);
        FftPlan.SimdEnabled = true;
        return _buffer;
    }
}

/// <summary>
/// An explicit Runge-Kutta solver on a non-stiff system and BDF on a stiff one.
/// </summary>
/// <remarks>
/// The right-hand side is an allocation-free delegate, so what is measured is the solver's own
/// bookkeeping. SciPy pays a Python call per evaluation here, which is where most of the gap in
/// the cross-stack comparison comes from.
/// </remarks>
[MemoryDiagnoser]
public class OdeSolverBenchmark
{
    private static void Lorenz(double t, ReadOnlySpan<double> y, Span<double> dy)
    {
        dy[0] = 10 * (y[1] - y[0]);
        dy[1] = y[0] * (28 - y[2]) - y[1];
        dy[2] = y[0] * y[1] - 8.0 / 3.0 * y[2];
    }

    private static void VanDerPol(double t, ReadOnlySpan<double> y, Span<double> dy)
    {
        dy[0] = y[1];
        dy[1] = 1000 * (1 - y[0] * y[0]) * y[1] - y[0];
    }

    /// <summary>Lorenz over 0..100 with RK45.</summary>
    [Benchmark(Baseline = true)]
    public int LorenzRk45() => Integrate.SolveIvp(Lorenz, 0, 100, [1.0, 1.0, 1.0], OdeMethod.Rk45, rtol: 1e-9, atol: 1e-9).T.Length;

    /// <summary>Lorenz over 0..100 with DOP853.</summary>
    [Benchmark]
    public int LorenzDop853() => Integrate.SolveIvp(Lorenz, 0, 100, [1.0, 1.0, 1.0], OdeMethod.Dop853, rtol: 1e-9, atol: 1e-9).T.Length;

    /// <summary>Stiff Van der Pol with BDF.</summary>
    [Benchmark]
    public int VanDerPolBdf() => Integrate.SolveIvp(VanDerPol, 0, 3000, [2.0, 0.0], OdeMethod.Bdf).T.Length;

    /// <summary>Stiff Van der Pol with Radau IIA.</summary>
    [Benchmark]
    public int VanDerPolRadau() => Integrate.SolveIvp(VanDerPol, 0, 3000, [2.0, 0.0], OdeMethod.Radau).T.Length;
}

/// <summary>
/// Single-threaded against all-core execution for the kernels that parallelise over rows.
/// </summary>
/// <remarks>
/// On this 4-core laptop the parallel speed-up is real but well short of 4x - the machine
/// throttles under sustained all-core load. Compare the two columns in one run only.
/// </remarks>
[MemoryDiagnoser]
public class ParallelKernelBenchmark
{
    private NdArray _points = NdArray.Zeros(1, 1);
    private NdArray _image = NdArray.Zeros(1, 1);
    private NdArray _values = NdArray.Zeros(1);

    /// <summary>Builds the inputs.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var rng = new GraviRandom(7);
        _points = rng.Uniform(0, 1, 2000, 64);
        _image = rng.Uniform(0, 1, 1024, 1024);
        _values = rng.Uniform(-4, 4, 1_000_000);
    }

    /// <summary>Euclidean cdist 2000x2000 in 64-D on one thread.</summary>
    [Benchmark(Baseline = true)]
    public NdArray CdistSerial() => Distance.Cdist(_points, _points, workers: 1);

    /// <summary>The same on every core.</summary>
    [Benchmark]
    public NdArray CdistParallel() => Distance.Cdist(_points, _points, workers: -1);

    /// <summary>Gaussian filter, sigma 3, 1024x1024.</summary>
    [Benchmark]
    public NdArray GaussianFilter() => NdImage.GaussianFilter(_image, 3.0);

    /// <summary>erf over a million values (vectorised entry point, parallel above a threshold).</summary>
    [Benchmark]
    public NdArray ErfMillion() => Special.Erf(_values);
}
