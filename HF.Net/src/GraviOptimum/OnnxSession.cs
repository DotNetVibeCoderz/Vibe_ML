using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Gravicode.HFNet.GraviHub;
using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviOptimum;

/// <summary>Which ONNX Runtime execution provider to run on.</summary>
public enum ExecutionTarget
{
    /// <summary>The CPU provider, always present.</summary>
    Cpu,

    /// <summary>NVIDIA CUDA. Needs the CUDA runtime and the matching ORT GPU package.</summary>
    Cuda,

    /// <summary>DirectML, which reaches any DirectX 12 GPU on Windows.</summary>
    DirectML,

    /// <summary>Intel oneDNN, which accelerates the CPU path on Intel hardware.</summary>
    OneDnn,

    /// <summary>Try the accelerators in turn and fall back to the CPU.</summary>
    Auto,
}

/// <summary>A tensor's name and shape on a session's input or output.</summary>
/// <param name="Name">The graph's name for it.</param>
/// <param name="Shape">Its dimensions; -1 marks one the model leaves symbolic.</param>
/// <param name="ElementType">The element type the graph declares.</param>
public readonly record struct TensorSpec(string Name, int[] Shape, string ElementType)
{
    /// <inheritdoc />
    public override string ToString()
        => $"{Name}: {ElementType}[{string.Join(",", Shape.Select(d => d < 0 ? "?" : d.ToString()))}]";
}

/// <summary>What a latency measurement found.</summary>
/// <param name="Target">The provider that was measured.</param>
/// <param name="Best">The fastest observed run.</param>
/// <param name="Median">The median run.</param>
/// <param name="Iterations">How many timed runs were taken.</param>
public readonly record struct LatencyReport(ExecutionTarget Target, TimeSpan Best, TimeSpan Median, int Iterations)
{
    /// <summary>Runs per second implied by the best time.</summary>
    public double PeakThroughput => Best.TotalSeconds > 0 ? 1 / Best.TotalSeconds : 0;

    /// <inheritdoc />
    public override string ToString()
        => $"{Target}: best {Best.TotalMilliseconds:F2} ms, median {Median.TotalMilliseconds:F2} ms ({Iterations} runs)";
}

/// <summary>
/// An ONNX model loaded into ONNX Runtime, with the execution provider chosen explicitly.
/// </summary>
/// <remarks>
/// <para>
/// This is the fast inference path. The managed encoder in GraviTransformers is
/// <see cref="double"/> end to end and exists so that a model can be inspected, loaded and
/// understood in pure .NET; a production request should go through an ONNX export running on
/// single-precision kernels that were written for the hardware.
/// </para>
/// <para>
/// <b>An execution provider that is asked for and missing is reported, not silently dropped.</b>
/// ORT's default behaviour is to fall back to the CPU, which is why a "CUDA" deployment so often
/// turns out to have been running on the CPU for months. <see cref="ActualTarget"/> says what was
/// actually registered.
/// </para>
/// </remarks>
public sealed class OnnxSession : IDisposable
{
    private readonly InferenceSession _session;
    private bool _disposed;

    private OnnxSession(InferenceSession session, string path, ExecutionTarget requested, ExecutionTarget actual)
    {
        _session = session;
        Path = path;
        RequestedTarget = requested;
        ActualTarget = actual;

        Inputs = [.. session.InputMetadata.Select(p => Describe(p.Key, p.Value))];
        Outputs = [.. session.OutputMetadata.Select(p => Describe(p.Key, p.Value))];
    }

    /// <summary>The file this session was created from.</summary>
    public string Path { get; }

    /// <summary>The provider that was asked for.</summary>
    public ExecutionTarget RequestedTarget { get; }

    /// <summary>The provider that was actually registered.</summary>
    public ExecutionTarget ActualTarget { get; }

    /// <summary>Whether the requested provider is the one in use.</summary>
    public bool RanOnRequestedTarget => RequestedTarget is ExecutionTarget.Auto || RequestedTarget == ActualTarget;

    /// <summary>The graph's inputs.</summary>
    public IReadOnlyList<TensorSpec> Inputs { get; }

    /// <summary>The graph's outputs.</summary>
    public IReadOnlyList<TensorSpec> Outputs { get; }

    /// <summary>Opens a local ONNX file.</summary>
    /// <param name="path">Path to the <c>.onnx</c> file.</param>
    /// <param name="target">Which provider to run on.</param>
    public static OnnxSession Open(string path, ExecutionTarget target = ExecutionTarget.Auto)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path)) throw new FileNotFoundException("ONNX model not found.", path);

        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
        };

        var actual = Register(options, target);

        // A provider that was named and could not be registered is a configuration error the caller
        // has to know about; Auto is allowed to settle for the CPU because that is what it means.
        if (target is not (ExecutionTarget.Auto or ExecutionTarget.Cpu) && actual != target)
        {
            options.Dispose();
            throw new NotSupportedException(
                $"Execution provider {target} could not be registered. The base Microsoft.ML.OnnxRuntime "
                + "package ships the CPU provider only - add Microsoft.ML.OnnxRuntime.Gpu for CUDA or "
                + "Microsoft.ML.OnnxRuntime.DirectML for DirectML. Use ExecutionTarget.Auto to accept a fallback.");
        }

        return new OnnxSession(new InferenceSession(path, options), path, target, actual);
    }

    /// <summary>Downloads a Hub model's ONNX export and opens it.</summary>
    /// <param name="repoId">A model id.</param>
    /// <param name="fileName">
    /// Which file to use. Defaults to the conventional <c>onnx/model.onnx</c>.
    /// </param>
    /// <param name="target">Which provider to run on.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    public static OnnxSession FromPretrained(
        string repoId,
        string? fileName = null,
        ExecutionTarget target = ExecutionTarget.Auto,
        string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var info = Hub.ModelInfo(repoId, revision);
        var candidates = info.FilesWithExtension(".onnx");

        if (candidates.Count == 0)
        {
            throw new HubException(
                $"'{repoId}' publishes no .onnx file. Export it first, for example with "
                + "optimum-cli, and push the result.") { RepoId = repoId };
        }

        var chosen = fileName is not null
            ? candidates.FirstOrDefault(f => f.Path == fileName)
            // Prefer the plain export over the quantised variants beside it: a caller who has not
            // said otherwise wants the reference numerics, not int8.
            : candidates.FirstOrDefault(f => f.Path.EndsWith("model.onnx", StringComparison.Ordinal));

        var file = chosen.Path is not null ? chosen : candidates[0];
        var local = Hub.DownloadFile(repoId, file.Path, revision);

        // An ONNX graph above the protobuf limit keeps its weights in a sibling .onnx_data file,
        // which the runtime opens by relative path - so it has to be fetched too.
        foreach (var companion in info.Files)
        {
            if (companion.Path.EndsWith(".onnx_data", StringComparison.Ordinal)
                || companion.Path.EndsWith(".onnx.data", StringComparison.Ordinal))
            {
                Hub.DownloadFile(repoId, companion.Path, revision);
            }
        }

        return Open(local, target);
    }

    /// <summary>Runs the model.</summary>
    /// <param name="inputs">Input tensors, keyed by graph input name.</param>
    /// <returns>Output tensors, keyed by graph output name.</returns>
    public Dictionary<string, NdArray> Run(IReadOnlyDictionary<string, NdArray> inputs)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(inputs);

        var feeds = new List<NamedOnnxValue>(inputs.Count);

        foreach (var (name, array) in inputs)
        {
            if (!_session.InputMetadata.TryGetValue(name, out var metadata))
            {
                throw new ArgumentException(
                    $"'{name}' is not an input of this model. It takes: {string.Join(", ", Inputs)}.",
                    nameof(inputs));
            }

            feeds.Add(ToNamedValue(name, array, metadata.ElementType));
        }

        using var results = _session.Run(feeds);

        var outputs = new Dictionary<string, NdArray>(StringComparer.Ordinal);
        foreach (var result in results) outputs[result.Name] = ToNdArray(result);

        return outputs;
    }

    /// <summary>Runs a model that takes one input and returns one output.</summary>
    /// <param name="input">The single input tensor.</param>
    public NdArray Run(NdArray input)
    {
        var name = Inputs.Count == 1
            ? Inputs[0].Name
            : throw new InvalidOperationException(
                $"This model has {Inputs.Count} inputs ({string.Join(", ", Inputs.Select(i => i.Name))}); "
                + "use the dictionary overload.");

        return Run(new Dictionary<string, NdArray> { [name] = input }).Values.First();
    }

    /// <summary>Times the model on a fixed input.</summary>
    /// <param name="inputs">The input to run.</param>
    /// <param name="iterations">How many timed runs.</param>
    /// <param name="warmup">How many untimed runs first.</param>
    /// <remarks>
    /// The warm-up is not optional. The first call builds the execution plan and allocates the
    /// arena, and on a GPU provider it also compiles kernels - timing it reports setup cost as if
    /// it were inference cost, which is how ONNX exports get reported as slower than they are.
    /// </remarks>
    public LatencyReport Measure(
        IReadOnlyDictionary<string, NdArray> inputs, int iterations = 20, int warmup = 5)
    {
        for (var i = 0; i < warmup; i++) Run(inputs);

        var samples = new List<TimeSpan>(iterations);
        for (var i = 0; i < iterations; i++)
        {
            var stopwatch = Stopwatch.StartNew();
            Run(inputs);
            stopwatch.Stop();
            samples.Add(stopwatch.Elapsed);
        }

        samples.Sort();
        return new LatencyReport(ActualTarget, samples[0], samples[samples.Count / 2], iterations);
    }

    private static ExecutionTarget Register(SessionOptions options, ExecutionTarget target)
    {
        // Each provider lives in its own NuGet package and throws when that package is absent, so
        // registration is attempted rather than assumed.
        foreach (var candidate in Order(target))
        {
            try
            {
                switch (candidate)
                {
                    case ExecutionTarget.Cuda:
                        options.AppendExecutionProvider_CUDA();
                        return ExecutionTarget.Cuda;

                    case ExecutionTarget.DirectML:
                        options.AppendExecutionProvider_DML();
                        return ExecutionTarget.DirectML;

                    case ExecutionTarget.OneDnn:
                        options.AppendExecutionProvider_Dnnl();
                        return ExecutionTarget.OneDnn;

                    case ExecutionTarget.Cpu:
                        return ExecutionTarget.Cpu;
                }
            }
            catch (Exception exception) when (
                exception is EntryPointNotFoundException or DllNotFoundException
                    or OnnxRuntimeException or TypeLoadException or NotSupportedException)
            {
                // Not built into this runtime package; try the next one.
            }
        }

        return ExecutionTarget.Cpu;
    }

    private static IEnumerable<ExecutionTarget> Order(ExecutionTarget target) => target switch
    {
        ExecutionTarget.Auto => [ExecutionTarget.Cuda, ExecutionTarget.DirectML, ExecutionTarget.Cpu],
        _ => [target, ExecutionTarget.Cpu],
    };

    private static TensorSpec Describe(string name, NodeMetadata metadata)
        => new(name, metadata.Dimensions, metadata.ElementType?.Name ?? "unknown");

    /// <summary>
    /// Converts an <see cref="NdArray"/> to the element type the graph declares.
    /// </summary>
    /// <remarks>
    /// The narrowing to <c>float</c> or <c>long</c> is where the two worlds meet: everything in the
    /// Gravicode stack is <c>double</c>, and essentially every ONNX graph wants <c>float</c> for
    /// activations and <c>int64</c> for token ids. Feeding a double tensor to a float input throws
    /// inside the runtime with a message that names neither the tensor nor the caller.
    /// </remarks>
    private static NamedOnnxValue ToNamedValue(string name, NdArray array, Type elementType)
    {
        var shape = array.Shape.ToArray();
        var values = array.AsContiguous().ToArray();

        if (elementType == typeof(float))
        {
            var data = new float[values.Length];
            for (var i = 0; i < values.Length; i++) data[i] = (float)values[i];
            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(data, shape));
        }

        if (elementType == typeof(long))
        {
            var data = new long[values.Length];
            for (var i = 0; i < values.Length; i++) data[i] = (long)Math.Round(values[i]);
            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(data, shape));
        }

        if (elementType == typeof(int))
        {
            var data = new int[values.Length];
            for (var i = 0; i < values.Length; i++) data[i] = (int)Math.Round(values[i]);
            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<int>(data, shape));
        }

        if (elementType == typeof(double))
        {
            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<double>(values, shape));
        }

        throw new NotSupportedException($"Input '{name}' has element type {elementType.Name}, which is not supported.");
    }

    private static NdArray ToNdArray(DisposableNamedOnnxValue value)
    {
        switch (value.ElementType)
        {
            case TensorElementType.Float:
            {
                var tensor = value.AsTensor<float>();
                var data = new double[tensor.Length];
                var i = 0;
                foreach (var element in tensor) data[i++] = element;
                return new NdArray(data, [.. tensor.Dimensions]);
            }

            case TensorElementType.Double:
            {
                var tensor = value.AsTensor<double>();
                return new NdArray([.. tensor], [.. tensor.Dimensions]);
            }

            case TensorElementType.Int64:
            {
                var tensor = value.AsTensor<long>();
                var data = new double[tensor.Length];
                var i = 0;
                foreach (var element in tensor) data[i++] = element;
                return new NdArray(data, [.. tensor.Dimensions]);
            }

            case TensorElementType.Int32:
            {
                var tensor = value.AsTensor<int>();
                var data = new double[tensor.Length];
                var i = 0;
                foreach (var element in tensor) data[i++] = element;
                return new NdArray(data, [.. tensor.Dimensions]);
            }

            default:
                throw new NotSupportedException(
                    $"Output '{value.Name}' has element type {value.ElementType}, which is not supported.");
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _session.Dispose();
    }

    /// <inheritdoc />
    public override string ToString()
        => $"OnnxSession({System.IO.Path.GetFileName(Path)} on {ActualTarget}: "
            + $"{Inputs.Count} in, {Outputs.Count} out)";
}
