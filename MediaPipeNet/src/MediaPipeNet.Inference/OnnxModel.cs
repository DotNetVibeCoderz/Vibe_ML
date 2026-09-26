using System.Collections.Concurrent;
using System.Diagnostics;
using MediaPipeNet.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MediaPipeNet.Inference;

/// <summary>Name and fixed shape of a model input or output (dynamic dimensions are resolved to 1).</summary>
/// <param name="Name">Tensor name in the ONNX graph.</param>
/// <param name="Shape">Concrete shape used for the preallocated buffer.</param>
public sealed record TensorSpec(string Name, IReadOnlyList<int> Shape)
{
    /// <summary>Total number of elements.</summary>
    public int ElementCount { get; } = Shape.Aggregate(1, (a, d) => a * d);

    /// <inheritdoc />
    public override string ToString() => $"{Name}[{string.Join('x', Shape)}]";
}

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

    private static TensorSpec CreateSpec(string name, NodeMetadata metadata)
    {
        if (metadata.ElementDataType != TensorElementType.Float)
            throw new NotSupportedException($"Tensor '{name}' has element type {metadata.ElementDataType}; only float32 tensors are supported.");
        return new TensorSpec(name, metadata.Dimensions.Select(d => d > 0 ? d : 1).ToArray());
    }

    /// <summary>Index of the input with the given name.</summary>
    public int GetInputIndex(string name) => IndexOf(Inputs, name);

    /// <summary>Index of the output with the given name.</summary>
    public int GetOutputIndex(string name) => IndexOf(Outputs, name);

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
public sealed class InferenceContext : IDisposable
{
    private readonly OnnxModel _model;
    private readonly float[][] _inputs;
    private readonly float[][] _outputs;
    private readonly OrtValue[] _inputValues;
    private readonly OrtValue[] _outputValues;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly RunOptions _runOptions = new();
    private int _returned;
    internal static readonly Lock DirectMLGate = new();

    internal InferenceContext(OnnxModel model)
    {
        _model = model;
        _inputNames = model.Inputs.Select(s => s.Name).ToArray();
        _outputNames = model.Outputs.Select(s => s.Name).ToArray();
        _inputs = model.Inputs.Select(s => new float[s.ElementCount]).ToArray();
        _outputs = model.Outputs.Select(s => new float[s.ElementCount]).ToArray();
        _inputValues = new OrtValue[_inputs.Length];
        _outputValues = new OrtValue[_outputs.Length];
        for (int i = 0; i < _inputs.Length; i++)
            _inputValues[i] = OrtValue.CreateTensorValueFromMemory(_inputs[i], model.Inputs[i].Shape.Select(d => (long)d).ToArray());
        for (int i = 0; i < _outputs.Length; i++)
            _outputValues[i] = OrtValue.CreateTensorValueFromMemory(_outputs[i], model.Outputs[i].Shape.Select(d => (long)d).ToArray());
    }

    /// <summary>The model this context belongs to.</summary>
    public OnnxModel Model => _model;

    /// <summary>Writable buffer of input <paramref name="index"/>.</summary>
    public Span<float> GetInput(int index) => _inputs[index];

    /// <summary>Writable buffer of the named input.</summary>
    public Span<float> GetInput(string name) => _inputs[_model.GetInputIndex(name)];

    /// <summary>Buffer of output <paramref name="index"/>, valid after <see cref="Run"/>.</summary>
    public ReadOnlySpan<float> GetOutput(int index) => _outputs[index];

    /// <summary>Buffer of the named output, valid after <see cref="Run"/>.</summary>
    public ReadOnlySpan<float> GetOutput(string name) => _outputs[_model.GetOutputIndex(name)];

    /// <summary>Runs the model synchronously on the current input buffers.</summary>
    public void Run()
    {
        long start = Stopwatch.GetTimestamp();
        if (_model.Provider == ExecutionProvider.DirectML)
        {
            // DirectML does not support concurrent Run calls; serialize them process-wide.
            lock (DirectMLGate) _model.Session.Run(_runOptions, _inputNames, _inputValues, _outputNames, _outputValues);
        }
        else
        {
            _model.Session.Run(_runOptions, _inputNames, _inputValues, _outputNames, _outputValues);
        }
        _model.RecordRun(start);
    }

    /// <summary>Returns the context to its model's pool.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _returned, 1) == 0) _model.Return(this);
    }

    internal void Reset() => _returned = 0;

    internal void Release()
    {
        foreach (var v in _inputValues) v.Dispose();
        foreach (var v in _outputValues) v.Dispose();
        _runOptions.Dispose();
    }
}
