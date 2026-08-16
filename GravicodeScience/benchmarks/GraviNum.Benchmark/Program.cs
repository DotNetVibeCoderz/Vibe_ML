using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Compute;

BenchmarkSwitcher.FromAssembly(typeof(MatrixProductBenchmark).Assembly).Run(args, BenchmarkConfig.Create());
return;

/// <summary>A short-running configuration so the whole suite finishes in minutes, not hours.</summary>
public static class BenchmarkConfig
{
    public static IConfig Create() => DefaultConfig.Instance
        .AddJob(Job.ShortRun.WithWarmupCount(3).WithIterationCount(5))
        .WithOptions(ConfigOptions.DisableOptimizationsValidator);
}

/// <summary>
/// The headline comparison: a dense matrix product on the CPU's SIMD units against the same
/// product on the GPU through ILGPU.
/// </summary>
/// <remarks>
/// The result is not a single winner. The GPU has far more arithmetic throughput, but every run
/// pays to copy both operands across the bus and the result back, and that transfer is fixed
/// while the arithmetic grows as n-cubed. Small matrices therefore lose on the GPU and large ones
/// win; the crossover is what this benchmark measures, and it is the number
/// <see cref="Compute.Best"/> encodes.
/// </remarks>
[MemoryDiagnoser]
public class MatrixProductBenchmark
{
    private NdArray _left = NdArray.Zeros(1, 1);
    private NdArray _right = NdArray.Zeros(1, 1);
    private double[] _leftFlat = [];
    private double[] _rightFlat = [];
    private IComputeBackend? _gpu;

    /// <summary>Square matrix dimension under test.</summary>
    [Params(128, 256, 512, 1024)]
    public int Size { get; set; }

    /// <summary>Allocates the operands and brings up the GPU backend if one exists.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var rng = new GraviRandom(42);
        _left = rng.StandardNormal(Size, Size);
        _right = rng.StandardNormal(Size, Size);
        _leftFlat = _left.ToArray();
        _rightFlat = _right.ToArray();
        _gpu = Compute.Gpu;

        Console.WriteLine($"[setup] {Compute.DescribeDevices()}");
    }

    /// <summary>The shipped CPU path: four-row register blocking, SIMD inner loop, threaded rows.</summary>
    [Benchmark(Baseline = true)]
    public double SimdBlocked() => LinAlg.Dot(_left, _right)[0, 0];

    /// <summary>The same product through the compute-backend abstraction on the CPU.</summary>
    [Benchmark]
    public double CpuBackend() => Compute.Cpu.MatMul(_leftFlat, _rightFlat, Size, Size, Size)[0];

    /// <summary>The GPU path, including both transfers. Skipped when no accelerator is present.</summary>
    [Benchmark]
    public double GpuBackend()
    {
        if (_gpu is null) return SimdBlocked();
        return _gpu.MatMul(_leftFlat, _rightFlat, Size, Size, Size)[0];
    }

    /// <summary>Releases the accelerator.</summary>
    [GlobalCleanup]
    public void Cleanup() => Compute.ReleaseGpu();
}

/// <summary>
/// What vectorisation and register blocking are actually worth, against the textbook triple loop.
/// </summary>
/// <remarks>
/// Kept to small matrices and in its own class: the naive loop is roughly twenty times slower, so
/// including it at 1024 would dominate the whole suite's runtime for no extra information.
/// </remarks>
[MemoryDiagnoser]
public class NaiveVersusSimdBenchmark
{
    private NdArray _left = NdArray.Zeros(1, 1);
    private NdArray _right = NdArray.Zeros(1, 1);
    private double[] _leftFlat = [];
    private double[] _rightFlat = [];

    /// <summary>Square matrix dimension under test.</summary>
    [Params(64, 128, 256)]
    public int Size { get; set; }

    /// <summary>Allocates the operands.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var rng = new GraviRandom(42);
        _left = rng.StandardNormal(Size, Size);
        _right = rng.StandardNormal(Size, Size);
        _leftFlat = _left.ToArray();
        _rightFlat = _right.ToArray();
    }

    /// <summary>The textbook triple loop, single-threaded and scalar.</summary>
    [Benchmark(Baseline = true)]
    public double NaiveTripleLoop()
    {
        var result = new double[Size * Size];
        for (var i = 0; i < Size; i++)
            for (var j = 0; j < Size; j++)
            {
                var accumulator = 0.0;
                for (var k = 0; k < Size; k++) accumulator += _leftFlat[i * Size + k] * _rightFlat[k * Size + j];
                result[i * Size + j] = accumulator;
            }
        return result[0];
    }

    /// <summary>The shipped implementation.</summary>
    [Benchmark]
    public double SimdBlocked() => LinAlg.Dot(_left, _right)[0, 0];
}

/// <summary>Element-wise work, where the CPU is memory-bound and the GPU is transfer-bound.</summary>
[MemoryDiagnoser]
public class ElementWiseBenchmark
{
    private NdArray _a = NdArray.Zeros(1);
    private NdArray _b = NdArray.Zeros(1);
    private double[] _aFlat = [];
    private double[] _bFlat = [];
    private IComputeBackend? _gpu;

    /// <summary>Number of elements in each operand.</summary>
    [Params(10_000, 1_000_000, 10_000_000)]
    public int Length { get; set; }

    /// <summary>Allocates the operands.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var rng = new GraviRandom(42);
        _a = rng.StandardNormal(Length);
        _b = rng.StandardNormal(Length);
        _aFlat = _a.ToArray();
        _bFlat = _b.ToArray();
        _gpu = Compute.Gpu;
    }

    /// <summary>Scalar loop baseline.</summary>
    [Benchmark(Baseline = true)]
    public double Scalar()
    {
        var result = new double[Length];
        for (var i = 0; i < Length; i++) result[i] = _aFlat[i] + _bFlat[i];
        return result[0];
    }

    /// <summary>The vectorised ufunc path.</summary>
    [Benchmark]
    public double SimdUFunc() => UFunc.Add(_a, _b).At(0);

    /// <summary>The GPU path, transfers included.</summary>
    [Benchmark]
    public double Gpu()
    {
        if (_gpu is null) return SimdUFunc();
        return _gpu.Add(_aFlat, _bFlat)[0];
    }

    /// <summary>Releases the accelerator.</summary>
    [GlobalCleanup]
    public void Cleanup() => Compute.ReleaseGpu();
}

/// <summary>
/// Decomposition cost, which is what dominates any workload that solves systems repeatedly.
/// </summary>
[MemoryDiagnoser]
public class DecompositionBenchmark
{
    private NdArray _matrix = NdArray.Zeros(1, 1);
    private NdArray _symmetric = NdArray.Zeros(1, 1);

    /// <summary>Square matrix dimension under test.</summary>
    [Params(64, 128, 256)]
    public int Size { get; set; }

    /// <summary>Builds a well-conditioned matrix and a symmetric positive-definite one.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var rng = new GraviRandom(42);
        _matrix = rng.StandardNormal(Size, Size) + NdArray.Eye(Size) * Size;
        _symmetric = LinAlg.Dot(_matrix, _matrix.T) + NdArray.Eye(Size) * Size;
    }

    /// <summary>LU with partial pivoting, the cheapest general factorisation.</summary>
    [Benchmark(Baseline = true)]
    public double Lu() => Decomposition.Lu(_matrix).Upper[0, 0];

    /// <summary>Householder QR, more stable and roughly twice the work of LU.</summary>
    [Benchmark]
    public double Qr() => Decomposition.Qr(_matrix).R[0, 0];

    /// <summary>Cholesky, about half the cost of LU but only valid for SPD matrices.</summary>
    [Benchmark]
    public double Cholesky() => Decomposition.Cholesky(_symmetric)[0, 0];

    /// <summary>One-sided Jacobi SVD, the most expensive and most informative.</summary>
    [Benchmark]
    public double Svd() => Decomposition.Svd(_matrix).SingularValues.At(0);

    /// <summary>Cyclic Jacobi eigen decomposition of the symmetric matrix.</summary>
    [Benchmark]
    public double SymmetricEigen() => Decomposition.SymmetricEigen(_symmetric).Values.At(0);
}

/// <summary>Sparse versus dense products at varying density, showing where CSR stops paying off.</summary>
[MemoryDiagnoser]
public class SparseBenchmark
{
    private NdArray _dense = NdArray.Zeros(1, 1);
    private SparseMatrix? _sparse;
    private NdArray _vector = NdArray.Zeros(1);

    /// <summary>Fraction of entries that are non-zero.</summary>
    [Params(0.01, 0.05, 0.25)]
    public double Density { get; set; }

    /// <summary>Builds a 2000x2000 matrix at the requested density.</summary>
    [GlobalSetup]
    public void Setup()
    {
        const int size = 2000;
        var rng = new GraviRandom(42);
        _dense = NdArray.Zeros(size, size);

        var nonZeros = (int)(size * size * Density);
        for (var k = 0; k < nonZeros; k++) _dense[rng.Next(size), rng.Next(size)] = rng.Normal();

        _sparse = SparseMatrix.FromDense(_dense);
        _vector = rng.StandardNormal(size);
    }

    /// <summary>The dense product, which touches every entry regardless of density.</summary>
    [Benchmark(Baseline = true)]
    public double Dense() => LinAlg.Dot(_dense, _vector).At(0);

    /// <summary>The CSR product, whose cost tracks the non-zero count.</summary>
    [Benchmark]
    public double Sparse() => _sparse!.Multiply(_vector).At(0);
}
