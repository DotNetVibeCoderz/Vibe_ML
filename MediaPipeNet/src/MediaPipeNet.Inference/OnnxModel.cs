using System.Collections.Concurrent;
using System.Diagnostics;
using MediaPipeNet.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MediaPipeNet.Inference;

/// <summary>Element type of a model tensor.</summary>
public enum TensorDataType
{
    /// <summary>32-bit float (image and audio models).</summary>
    Float32,
    /// <summary>32-bit integer (token ids, masks).</summary>
    Int32,
    /// <summary>64-bit integer.</summary>
    Int64,
}

/// <summary>Name, fixed shape and element type of a model input or output (dynamic dimensions are resolved to 1).</summary>
/// <param name="Name">Tensor name in the ONNX graph.</param>
/// <param name="Shape">Concrete shape used for the preallocated buffer.</param>
/// <param name="ElementType">Element type.</param>
public sealed record TensorSpec(string Name, IReadOnlyList<int> Shape, TensorDataType ElementType = TensorDataType.Float32)
{
    /// <summary>Total number of elements.</summary>
    public int ElementCount { get; } = Shape.Aggregate(1, (a, d) => a * d);

    /// <inheritdoc />
    public override string ToString() => $"{Name}[{string.Join('x', Shape)}]{(ElementType == TensorDataType.Float32 ? "" : ":" + ElementType)}";
}

/// <summary>An input of <see cref="OnnxModel.RunDynamic"/>: a name, a float[] / int[] / long[] buffer and its shape.</summary>
/// <param name="Name">Input name.</param>
/// <param name="Data">Row-major data.</param>
/// <param name="Shape">Concrete shape.</param>
public readonly record struct DynamicTensor(string Name, Array Data, long[] Shape);

/// <summary>
/// A loaded ONNX model. The <see cref="InferenceSession"/> is created once (with the best
/// available execution provider) and reused for every call. Inputs and outputs live in
/// preallocated buffers inside pooled <see cref="InferenceContext"/> objects, so steady-state
/// inference performs no managed allocations and several threads can run the model concurrently.
/// </summary>
/// <example>
/// <code>
/// using var model = OnnxModel.Load("face_detection_short_range.onnx");
/// using var ctx = model.RentContext();
/// FillInput(ctx.GetInput(0));
/// ctx.Run();
/// ReadOnlySpan&lt;float&gt; scores = ctx.GetOutput(1);
/// </code>
/// </example>
public sealed class OnnxModel : IDisposable
{
    private readonly ConcurrentBag<InferenceContext> _pool = [];
    private readonly InferenceOptions _options;
    private readonly KeyValuePair<string, object?>[] _tags;
    private int _pooled;
    private bool _disposed;

    private OnnxModel(string name, InferenceSession session, ExecutionProvider provider, InferenceOptions options)
    {
        Name = name;
        Session = session;
        Provider = provider;
        _options = options;
        Inputs = session.InputMetadata.Select(kv => CreateSpec(kv.Key, kv.Value)).ToArray();
        Outputs = session.OutputMetadata.Select(kv => CreateSpec(kv.Key, kv.Value)).ToArray();
        _tags = [new("model", name), new("provider", provider.ToString())];
    }

    /// <summary>Logical model name (used in logs and metrics).</summary>
    public string Name { get; }

    /// <summary>The execution provider the session runs on.</summary>
    public ExecutionProvider Provider { get; }

    /// <summary>The underlying ONNX Runtime session (for advanced scenarios).</summary>
    public InferenceSession Session { get; }

    /// <summary>Model inputs, in graph order.</summary>
    public IReadOnlyList<TensorSpec> Inputs { get; }

    /// <summary>Model outputs, in graph order.</summary>
    public IReadOnlyList<TensorSpec> Outputs { get; }

    /// <summary>The options the model was loaded with.</summary>
    public InferenceOptions Options => _options;

    /// <summary>Loads a model from a file.</summary>
    public static OnnxModel Load(string path, InferenceOptions? options = null, ILogger? logger = null, string? name = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path)) throw new ModelNotFoundException($"Model file not found: {path}");
        return Create(name ?? Path.GetFileNameWithoutExtension(path), so => new InferenceSession(path, so), options, logger);
    }

    /// <summary>Loads a model from memory.</summary>
    public static OnnxModel Load(byte[] model, string name, InferenceOptions? options = null, ILogger? logger = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        return Create(name, so => new InferenceSession(model, so), options, logger);
    }

    private static OnnxModel Create(string name, Func<SessionOptions, InferenceSession> factory, InferenceOptions? options, ILogger? logger)
    {
        options ??= InferenceOptions.Default;
        logger ??= NullLogger.Instance;
        EnsureQuietEnvironment();
        var candidates = ExecutionProviderSelector.GetCandidates(options.Provider, options.FallbackToCpu || options.Provider == ExecutionProvider.Auto);
        Exception? failure = null;
        foreach (var provider in candidates)
        {
            SessionOptions? so = null;
            try
            {
                so = ExecutionProviderSelector.CreateSessionOptions(provider, options);
                InferenceSession session;
                if (provider == ExecutionProvider.DirectML)
                {
                    // DirectML sessions must not be created (or run) concurrently on the same device.
                    lock (InferenceContext.DirectMLGate) session = factory(so);
                }
                else
                {
                    session = factory(so);
                }
                ExecutionProviderSelector.LogSelection(logger, name, provider, failure);
                return new OnnxModel(name, session, provider, options);
            }
            catch (Exception e) when (e is OnnxRuntimeException or EntryPointNotFoundException or DllNotFoundException or InvalidOperationException)
            {
                failure = e;
                logger.LogDebug(e, "Model {Model}: execution provider {Provider} unavailable", name, provider);
            }
            finally
            {
                so?.Dispose();
            }
        }
        throw new MediaPipeException(
            $"Could not create an ONNX Runtime session for '{name}'. Make sure a native ONNX Runtime package " +
            "(Gravicode.MediaPipeNet, Gravicode.MediaPipeNet.DirectML or Gravicode.MediaPipeNet.Cuda) is referenced.", failure ?? new InvalidOperationException());
    }

    private static int s_envConfigured;

    // Session-level severity does not cover messages logged while the graph is loaded (e.g. deprecated
    // attributes in converted models), which go to the process-wide environment logger.
    private static void EnsureQuietEnvironment()
    {
        if (Interlocked.Exchange(ref s_envConfigured, 1) == 0) OrtEnv.Instance().EnvLogLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
    }

    private static TensorSpec CreateSpec(string name, NodeMetadata metadata)
    {
        var type = metadata.ElementDataType switch
        {
            TensorElementType.Float => TensorDataType.Float32,
            TensorElementType.Int32 => TensorDataType.Int32,
            TensorElementType.Int64 => TensorDataType.Int64,
            var other => throw new NotSupportedException($"Tensor '{name}' has element type {other}; float32, int32 and int64 are supported."),
        };
        return new TensorSpec(name, metadata.Dimensions.Select(d => d > 0 ? d : 1).ToArray(), type);
    }

    /// <summary>Index of the input with the given name.</summary>
    public int GetInputIndex(string name) => IndexOf(Inputs, name);

    /// <summary>Index of the output with the given name.</summary>
    public int GetOutputIndex(string name) => IndexOf(Outputs, name);

    /// <summary>Index of the first input whose name contains <paramref name="fragment"/> (case-insensitive), or -1.</summary>
    public int FindInput(string fragment) => Find(Inputs, fragment);

    /// <summary>Index of the first output whose name contains <paramref name="fragment"/> (case-insensitive), or -1.</summary>
    public int FindOutput(string fragment) => Find(Outputs, fragment);

    private static int Find(IReadOnlyList<TensorSpec> specs, string fragment)
    {
        for (int i = 0; i < specs.Count; i++)
            if (specs[i].Name.Contains(fragment, StringComparison.OrdinalIgnoreCase)) return i;
        return -1;
    }

    private static int IndexOf(IReadOnlyList<TensorSpec> specs, string name)
    {
        for (int i = 0; i < specs.Count; i++)
            if (specs[i].Name == name) return i;
        throw new KeyNotFoundException($"Tensor '{name}' not found. Available: {string.Join(", ", specs)}");
    }

    /// <summary>
    /// Rents an I/O context (preallocated input and output buffers). Dispose it to return it to the
    /// pool. Contexts are not thread-safe; rent one per concurrent caller.
    /// </summary>
    public InferenceContext RentContext()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_pool.TryTake(out var ctx))
        {
            Interlocked.Decrement(ref _pooled);
            ctx.Reset();
            return ctx;
        }
        return new InferenceContext(this);
    }

    /// <summary>
    /// Runs the model once on inputs of any (dynamic) shape, e.g. a variable number of tokens. Unlike
    /// <see cref="RentContext"/> this allocates per call; outputs are returned as new float arrays with their shapes.
    /// </summary>
    public IReadOnlyList<(float[] Data, long[] Shape)> RunDynamic(IReadOnlyList<DynamicTensor> inputs)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(inputs);
        long start = Stopwatch.GetTimestamp();
        var values = new OrtValue[inputs.Count];
        try
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                var input = inputs[i];
                values[i] = input.Data switch
                {
                    int[] ints => OrtValue.CreateTensorValueFromMemory(ints, input.Shape),
                    long[] longs => OrtValue.CreateTensorValueFromMemory(longs, input.Shape),
                    float[] floats => OrtValue.CreateTensorValueFromMemory(floats, input.Shape),
                    _ => throw new NotSupportedException($"Input '{input.Name}': only float[], int[] and long[] are supported."),
                };
            }
            using var runOptions = new RunOptions();
            var names = inputs.Select(x => x.Name).ToArray();
            var outputNames = Outputs.Select(o => o.Name).ToArray();
            IDisposableReadOnlyCollection<OrtValue> outputs;
            if (Provider == ExecutionProvider.DirectML)
            {
                lock (InferenceContext.DirectMLGate) outputs = Session.Run(runOptions, names, values, outputNames);
            }
            else
            {
                outputs = Session.Run(runOptions, names, values, outputNames);
            }
            using (outputs)
            {
                var result = outputs.Select(o => (o.GetTensorDataAsSpan<float>().ToArray(), o.GetTensorTypeAndShape().Shape)).ToArray();
                RecordRun(start);
                return result;
            }
        }
        finally
        {
            foreach (var v in values) v?.Dispose();
        }
    }

    internal void Return(InferenceContext context)
    {
        if (_disposed || Interlocked.Increment(ref _pooled) > _options.MaxPooledContexts)
        {
            if (!_disposed) Interlocked.Decrement(ref _pooled);
            context.Release();
            return;
        }
        _pool.Add(context);
    }

    internal void RecordRun(long startTimestamp)
    {
        MediaPipeTelemetry.InferenceDuration.Record(Stopwatch.GetElapsedTime(startTimestamp).TotalMilliseconds, _tags);
    }

    /// <summary>Releases the session and all pooled buffers.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        while (_pool.TryTake(out var ctx)) ctx.Release();
        Session.Dispose();
    }
}

/// <summary>
/// Preallocated input and output buffers for one <see cref="OnnxModel"/>. Fill the inputs, call
/// <see cref="Run"/>, read the outputs; then dispose to return the context to the model's pool.
/// </summary>
/// <remarks>
/// With <see cref="InferenceOptions.UseIoBinding"/> the buffers are bound to the session once
/// (<see cref="OrtIoBinding"/>) and every <see cref="Run"/> reuses the binding, skipping the per-call
/// name and value marshaling.
/// </remarks>
public sealed class InferenceContext : IDisposable
{
    private readonly OnnxModel _model;
    private readonly Array[] _inputs;
    private readonly Array[] _outputs;
    private readonly OrtValue[] _inputValues;
    private readonly OrtValue[] _outputValues;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly RunOptions _runOptions = new();
    private readonly OrtIoBinding? _binding;
    private int _returned;
    internal static readonly Lock DirectMLGate = new();

    internal InferenceContext(OnnxModel model)
    {
        _model = model;
        _inputNames = model.Inputs.Select(s => s.Name).ToArray();
        _outputNames = model.Outputs.Select(s => s.Name).ToArray();
        _inputs = model.Inputs.Select(Allocate).ToArray();
        _outputs = model.Outputs.Select(Allocate).ToArray();
        _inputValues = new OrtValue[_inputs.Length];
        _outputValues = new OrtValue[_outputs.Length];
        for (int i = 0; i < _inputs.Length; i++) _inputValues[i] = CreateValue(_inputs[i], model.Inputs[i]);
        for (int i = 0; i < _outputs.Length; i++) _outputValues[i] = CreateValue(_outputs[i], model.Outputs[i]);

        if (model.Options.UseIoBinding)
        {
            _binding = model.Session.CreateIoBinding();
            for (int i = 0; i < _inputs.Length; i++) _binding.BindInput(_inputNames[i], _inputValues[i]);
            for (int i = 0; i < _outputs.Length; i++) _binding.BindOutput(_outputNames[i], _outputValues[i]);
        }
    }

    private static Array Allocate(TensorSpec spec) => spec.ElementType switch
    {
        TensorDataType.Int32 => new int[spec.ElementCount],
        TensorDataType.Int64 => new long[spec.ElementCount],
        _ => new float[spec.ElementCount],
    };

    private static OrtValue CreateValue(Array buffer, TensorSpec spec)
    {
        long[] shape = spec.Shape.Select(d => (long)d).ToArray();
        return buffer switch
        {
            int[] ints => OrtValue.CreateTensorValueFromMemory(ints, shape),
            long[] longs => OrtValue.CreateTensorValueFromMemory(longs, shape),
            _ => OrtValue.CreateTensorValueFromMemory((float[])buffer, shape),
        };
    }

    /// <summary>The model this context belongs to.</summary>
    public OnnxModel Model => _model;

    /// <summary>True when this context runs through an <see cref="OrtIoBinding"/>.</summary>
    public bool UsesIoBinding => _binding is not null;

    /// <summary>Writable float buffer of input <paramref name="index"/>.</summary>
    public Span<float> GetInput(int index) => As<float>(_inputs[index], _model.Inputs[index]);

    /// <summary>Writable float buffer of the named input.</summary>
    public Span<float> GetInput(string name) => GetInput(_model.GetInputIndex(name));

    /// <summary>Writable int32 buffer of input <paramref name="index"/>.</summary>
    public Span<int> GetInputInt32(int index) => As<int>(_inputs[index], _model.Inputs[index]);

    /// <summary>Writable int64 buffer of input <paramref name="index"/>.</summary>
    public Span<long> GetInputInt64(int index) => As<long>(_inputs[index], _model.Inputs[index]);

    /// <summary>Float buffer of output <paramref name="index"/>, valid after <see cref="Run"/>.</summary>
    public ReadOnlySpan<float> GetOutput(int index) => As<float>(_outputs[index], _model.Outputs[index]);

    /// <summary>Float buffer of the named output, valid after <see cref="Run"/>.</summary>
    public ReadOnlySpan<float> GetOutput(string name) => GetOutput(_model.GetOutputIndex(name));

    /// <summary>Int32 buffer of output <paramref name="index"/>, valid after <see cref="Run"/>.</summary>
    public ReadOnlySpan<int> GetOutputInt32(int index) => As<int>(_outputs[index], _model.Outputs[index]);

    private static T[] As<T>(Array buffer, TensorSpec spec) =>
        buffer as T[] ?? throw new InvalidOperationException($"Tensor {spec.Name} is {spec.ElementType}, not {typeof(T).Name}.");

    /// <summary>Runs the model synchronously on the current input buffers.</summary>
    public void Run()
    {
        long start = Stopwatch.GetTimestamp();
        if (_model.Provider == ExecutionProvider.DirectML)
        {
            // DirectML does not support concurrent Run calls; serialize them process-wide.
            lock (DirectMLGate) RunCore();
        }
        else
        {
            RunCore();
        }
        _model.RecordRun(start);
    }

    private void RunCore()
    {
        if (_binding is not null)
        {
            _model.Session.RunWithBinding(_runOptions, _binding);
            return;
        }
        _model.Session.Run(_runOptions, _inputNames, _inputValues, _outputNames, _outputValues);
    }

    /// <summary>Returns the context to its model's pool.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _returned, 1) == 0) _model.Return(this);
    }

    internal void Reset() => _returned = 0;

    internal void Release()
    {
        _binding?.Dispose();
        foreach (var v in _inputValues) v.Dispose();
        foreach (var v in _outputValues) v.Dispose();
        _runOptions.Dispose();
    }
}
