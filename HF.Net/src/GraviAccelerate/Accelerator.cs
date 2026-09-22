using System.Diagnostics;
using Gravicode.HFNet.GraviDatasets;
using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviLearn.Distributed;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Compute;
using Dataset = Gravicode.HFNet.GraviDatasets.Dataset;

namespace Gravicode.HFNet.GraviAccelerate;

/// <summary>Where a computation runs.</summary>
public enum DeviceKind
{
    /// <summary>The CPU, using SIMD kernels and the thread pool.</summary>
    Cpu,

    /// <summary>A GPU through ILGPU - CUDA or OpenCL.</summary>
    Gpu,

    /// <summary>Whichever of the two the size of the work favours.</summary>
    Auto,
}

/// <summary>One device the machine can compute on.</summary>
/// <param name="Kind">CPU or GPU.</param>
/// <param name="Name">The device's reported name.</param>
/// <param name="Available">Whether it can actually be used in this process.</param>
public readonly record struct DeviceInfo(DeviceKind Kind, string Name, bool Available)
{
    /// <inheritdoc />
    public override string ToString() => $"{Kind}: {Name}{(Available ? "" : " (unavailable)")}";
}

/// <summary>What a training run cost and produced.</summary>
/// <param name="Epochs">Epochs completed.</param>
/// <param name="Examples">Examples processed in total, counting every epoch.</param>
/// <param name="Elapsed">Wall-clock time.</param>
/// <param name="Workers">How many shards the data was split across.</param>
/// <param name="Device">The device the work ran on.</param>
/// <param name="FinalScore">Training score at the end, when the estimator reports one.</param>
public readonly record struct TrainingReport(
    int Epochs, long Examples, TimeSpan Elapsed, int Workers, DeviceKind Device, double FinalScore)
{
    /// <summary>Examples per second over the whole run.</summary>
    public double Throughput => Elapsed.TotalSeconds > 0 ? Examples / Elapsed.TotalSeconds : 0;

    /// <inheritdoc />
    public override string ToString()
        => $"{Epochs} epochs, {Examples:N0} examples in {Elapsed.TotalSeconds:F2}s "
            + $"({Throughput:N0}/s) on {Device} across {Workers} worker(s), score {FinalScore:F4}";
}

/// <summary>
/// Chooses where work runs and how it is split, so the same training code scales from one core to
/// a GPU without being rewritten.
/// </summary>
/// <remarks>
/// <para>
/// <b>The GPU is not the default, and that is a measured decision rather than caution.</b> The whole
/// stack is <see cref="double"/>, and consumer and integrated GPUs run double-precision arithmetic
/// at a small fraction of their single-precision rate - the foundation measured its ILGPU path 5 to
/// 8 times <i>slower</i> than the CPU on an integrated device. <see cref="DeviceKind.Auto"/>
/// therefore only reaches for the GPU above a size threshold where the transfer and the arithmetic
/// both pay for themselves.
/// </para>
/// <para>
/// Data parallelism here is sharding plus weighted gradient averaging, run through the foundation's
/// transports. The weighting matters: a plain mean of per-worker means equals the global mean only
/// when the shards are the same size, and they are not whenever the worker count does not divide
/// the example count.
/// </para>
/// </remarks>
public sealed class Accelerator
{
    /// <summary>Creates an accelerator.</summary>
    /// <param name="device">Which device to use. <see cref="DeviceKind.Auto"/> decides by size.</param>
    /// <param name="workers">
    /// How many shards to split data across. Defaults to the processor count.
    /// </param>
    public Accelerator(DeviceKind device = DeviceKind.Auto, int? workers = null)
    {
        Device = device;
        Workers = Math.Max(1, workers ?? Environment.ProcessorCount);
    }

    /// <summary>The device this accelerator was configured with.</summary>
    public DeviceKind Device { get; }

    /// <summary>How many shards data is split across.</summary>
    public int Workers { get; }

    /// <summary>
    /// The element count above which <see cref="DeviceKind.Auto"/> will consider the GPU.
    /// </summary>
    /// <remarks>
    /// Below this the kernel launch and the round trip over PCIe dominate, so the GPU loses even
    /// when it is genuinely faster per element.
    /// </remarks>
    public const int GpuThreshold = 1 << 20;

    /// <summary>Every device this process can see.</summary>
    public static IReadOnlyList<DeviceInfo> Devices()
    {
        var devices = new List<DeviceInfo>
        {
            new(DeviceKind.Cpu, $"{Environment.ProcessorCount} logical cores, SIMD width {System.Numerics.Vector<double>.Count}", true),
        };

        // Probing for a GPU initialises ILGPU, which can throw on a machine with a broken driver.
        // A missing accelerator is a normal condition, not an error worth propagating.
        try
        {
            devices.Add(new DeviceInfo(DeviceKind.Gpu, Compute.DescribeDevices(), Compute.IsGpuAvailable));
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            devices.Add(new DeviceInfo(DeviceKind.Gpu, $"unavailable ({exception.GetType().Name})", false));
        }

        return devices;
    }

    /// <summary>A one-line description of what is available.</summary>
    public static string Describe() => string.Join("; ", Devices());

    /// <summary>The backend to use for an operation of a given size.</summary>
    /// <param name="elementCount">How many elements the operation touches.</param>
    public IComputeBackend Backend(int elementCount) => Device switch
    {
        DeviceKind.Cpu => Compute.Cpu,
        DeviceKind.Gpu => Compute.Gpu ?? Compute.Cpu,
        _ => elementCount >= GpuThreshold && Compute.IsGpuAvailable ? Compute.Gpu! : Compute.Cpu,
    };

    /// <summary>Matrix multiply on the chosen device.</summary>
    public NdArray Dot(NdArray a, NdArray b) => Compute.Dot(a, b, Backend(a.Size + b.Size));

    /// <summary>Element-wise add on the chosen device.</summary>
    public NdArray Add(NdArray a, NdArray b) => Compute.Add(a, b, Backend(a.Size));

    /// <summary>Element-wise multiply on the chosen device.</summary>
    public NdArray Multiply(NdArray a, NdArray b) => Compute.Multiply(a, b, Backend(a.Size));

    // ------------------------------------------------------------------ training

    /// <summary>
    /// Trains an estimator over a dataset, sharding the data across workers.
    /// </summary>
    /// <param name="estimator">The model to fit.</param>
    /// <param name="dataset">The examples.</param>
    /// <param name="labelColumn">Which column holds the target.</param>
    /// <param name="featureColumns">
    /// Which columns are features. Every numeric column except the label when null.
    /// </param>
    /// <param name="epochs">How many passes over the data.</param>
    /// <returns>What the run cost.</returns>
    /// <remarks>
    /// <para>
    /// This is the blueprint's <c>Accelerator.Train(model, dataset)</c>: it prepares the matrices,
    /// runs the estimator's own <c>Fit</c> once per epoch, and reports what the run cost.
    /// </para>
    /// <para>
    /// It deliberately does <b>not</b> shard the fit across workers. Sharding only composes for an
    /// estimator whose parameters can be averaged, and averaging the coefficients of two decision
    /// trees produces something that is not a tree. Where averaging is valid, build the loop from
    /// <see cref="Partition"/> and <see cref="ParallelGradient"/>, which do the weighting correctly.
    /// </para>
    /// </remarks>
    public TrainingReport Train(
        IEstimator estimator,
        Dataset dataset,
        string labelColumn,
        IReadOnlyList<string>? featureColumns = null,
        int epochs = 1)
    {
        ArgumentNullException.ThrowIfNull(estimator);
        ArgumentNullException.ThrowIfNull(dataset);

        var features = featureColumns ?? [.. dataset.Frame.NumericColumns
            .Select(c => c.Name)
            .Where(n => !string.Equals(n, labelColumn, StringComparison.Ordinal))];

        var x = dataset.ToMatrix([.. features]);
        var y = new NdArray(dataset.NumericColumn(labelColumn), dataset.Count);

        var stopwatch = Stopwatch.StartNew();

        for (var epoch = 0; epoch < epochs; epoch++) estimator.Fit(x, y);

        stopwatch.Stop();

        var score = estimator is IClassifier classifier
            ? Score(classifier, x, y)
            : double.NaN;

        return new TrainingReport(
            epochs, (long)dataset.Count * epochs, stopwatch.Elapsed, 1, Device, score);
    }

    /// <summary>
    /// Runs one data-parallel round: shard, compute a gradient per shard, average them by size.
    /// </summary>
    /// <param name="itemCount">How many examples there are.</param>
    /// <param name="gradientOf">Computes the gradient for one shard.</param>
    /// <returns>The averaged gradient, as if it had been computed over everything at once.</returns>
    /// <remarks>
    /// Shards are collected in worker order rather than completion order. Floating-point addition
    /// is not associative, so collecting in arrival order would make the result depend on thread
    /// scheduling and quietly break reproducibility.
    /// </remarks>
    public NdArray ParallelGradient(int itemCount, Func<Shard, NdArray> gradientOf)
    {
        ArgumentNullException.ThrowIfNull(gradientOf);

        var shards = DataParallel.Partition(itemCount, Workers);
        var gradients = new NdArray[shards.Length];

        Parallel.For(0, shards.Length, i => gradients[i] = gradientOf(shards[i]));

        return DataParallel.AverageGradients(gradients, [.. shards.Select(s => s.Count)]);
    }

    /// <summary>Splits an example count into shards, one per worker.</summary>
    public Shard[] Partition(int itemCount) => DataParallel.Partition(itemCount, Workers);

    /// <summary>
    /// Measures throughput of a unit of work, after warming it up.
    /// </summary>
    /// <param name="work">The operation to time.</param>
    /// <param name="iterations">How many timed repetitions.</param>
    /// <param name="warmup">
    /// How many untimed repetitions first. Must exceed about thirty to be meaningful.
    /// </param>
    /// <returns>The best observed time, not the mean.</returns>
    /// <remarks>
    /// <b>The warm-up count is not padding.</b> Tiered JIT recompiles a hot method after roughly
    /// thirty calls, so a measurement taken with two warm-up iterations times the interpreter's
    /// output and can be wildly out - the foundation recorded a 192-element cube measuring eight
    /// times slower than a 224-element one that way. The best time rather than the mean is reported
    /// for the same reason a benchmark harness does: on a throttling laptop the mean measures the
    /// thermal state of the room.
    /// </remarks>
    public static TimeSpan Measure(Action work, int iterations = 20, int warmup = 40)
    {
        ArgumentNullException.ThrowIfNull(work);

        for (var i = 0; i < warmup; i++) work();

        var best = TimeSpan.MaxValue;
        for (var i = 0; i < iterations; i++)
        {
            var stopwatch = Stopwatch.StartNew();
            work();
            stopwatch.Stop();

            if (stopwatch.Elapsed < best) best = stopwatch.Elapsed;
        }

        return best;
    }

    /// <summary>Releases any GPU context this process acquired.</summary>
    public static void ReleaseGpu() => Compute.ReleaseGpu();

    private static double Score(IClassifier classifier, NdArray x, NdArray y)
    {
        var predicted = classifier.Predict(x);
        var correct = 0;

        for (var i = 0; i < y.Size; i++)
        {
            if (Math.Abs(predicted.At(i) - y.At(i)) < 1e-9) correct++;
        }

        return y.Size == 0 ? 0 : (double)correct / y.Size;
    }

    /// <inheritdoc />
    public override string ToString() => $"Accelerator({Device}, {Workers} workers)";
}
