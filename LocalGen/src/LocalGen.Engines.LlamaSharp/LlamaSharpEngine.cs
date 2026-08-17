using LLama;
using LLama.Common;
using LLama.Native;
using LocalGen.Core;
using LocalGen.Core.Engines;
using LocalGen.Core.Models;
using Microsoft.Extensions.Logging;

namespace LocalGen.Engines.LlamaSharp;

/// <summary>
/// The default LocalGen backend: llama.cpp through LLamaSharp. Serves GGUF weights on CPU or
/// GPU and is the only backend that supports GBNF-constrained output.
/// </summary>
public sealed class LlamaSharpEngine : IInferenceEngine
{
    private readonly ILoggerFactory _loggerFactory;
    private readonly ILogger<LlamaSharpEngine> _logger;

    // Native library loading is process-wide and one-shot in llama.cpp, so the probe result is
    // cached: asking twice cannot produce a different answer.
    private EngineAvailability? _availability;
    private readonly SemaphoreSlim _probeLock = new(1, 1);

    public LlamaSharpEngine(ILoggerFactory loggerFactory)
    {
        _loggerFactory = loggerFactory;
        _logger = loggerFactory.CreateLogger<LlamaSharpEngine>();

        // Done here rather than in the DI extension so the callback is installed before the
        // first probe touches the native library and it starts writing to stderr.
        ServiceCollectionExtensions.RedirectNativeLogging(loggerFactory);
    }

    public EngineDescriptor Descriptor { get; } = new()
    {
        Kind = EngineKind.LlamaSharp,
        DisplayName = "LlamaSharp (llama.cpp)",
        Description =
            "Runs GGUF models through llama.cpp. Broadest model coverage, runs on CPU or GPU, " +
            "and the only backend with grammar-constrained decoding.",
        Recommendation =
            "Recommended for most users. Pick the CUDA build for NVIDIA GPUs, Vulkan for AMD/Intel, " +
            "or CPU when no GPU is available.",
        Capabilities = new EngineCapabilities
        {
            SupportsStreaming = true,
            SupportsEmbeddings = true,
            SupportsGrammar = true,
            SupportsToolCalling = true,
            SupportsVision = true,
            SupportsMultiGpu = true,
            Formats = [ModelFormat.Gguf],
            Devices = [DeviceKind.Auto, DeviceKind.Cpu, DeviceKind.Cuda, DeviceKind.Vulkan, DeviceKind.Metal]
        }
    };

    public async ValueTask<EngineAvailability> ProbeAsync(CancellationToken cancellationToken = default)
    {
        if (_availability is not null)
        {
            return _availability;
        }

        await _probeLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return _availability ??= Probe();
        }
        finally
        {
            _probeLock.Release();
        }
    }

    private EngineAvailability Probe()
    {
        try
        {
            // Reading system info forces the loader to resolve a backend, which is the only
            // reliable way to learn whether one is actually present on this machine.
            var systemInfo = ReadSystemInfo();

            // The accelerators come from the registered ggml devices, not from the system-info
            // banner: current llama.cpp builds report only CPU feature flags there, so a machine
            // with a working CUDA backend still shows a banner that mentions no GPU at all.
            var accelerators = ReadAccelerators();
            var acceleratorNames = accelerators.Select(static a => a.Name).ToList();

            var devices = new List<DeviceKind> { DeviceKind.Cpu };

            if (NativeApi.llama_supports_gpu_offload())
            {
                if (acceleratorNames.Any(static name => name.Contains("CUDA", StringComparison.OrdinalIgnoreCase)))
                {
                    devices.Add(DeviceKind.Cuda);
                }

                if (acceleratorNames.Any(static name => name.Contains("Vulkan", StringComparison.OrdinalIgnoreCase)))
                {
                    devices.Add(DeviceKind.Vulkan);
                }

                if (acceleratorNames.Any(static name => name.Contains("Metal", StringComparison.OrdinalIgnoreCase)))
                {
                    devices.Add(DeviceKind.Metal);
                }
            }

            if (devices.Count > 1)
            {
                devices.Insert(0, DeviceKind.Auto);
            }

            // The device list is more useful in the UI than the CPU banner alone.
            if (accelerators.Count > 0)
            {
                systemInfo = $"{string.Join(", ", acceleratorNames)} | {systemInfo}";
            }

            _logger.LogInformation("LlamaSharp backend ready. Devices: {Devices}", string.Join(", ", devices));

            if (accelerators.Count > 1)
            {
                _logger.LogInformation(
                    "{Count} accelerators registered ({Names}); tensor splitting is available.",
                    accelerators.Count, string.Join(", ", acceleratorNames));
            }

            return new EngineAvailability
            {
                IsAvailable = true,
                AvailableDevices = devices,
                Accelerators = accelerators,
                Version = systemInfo.Split('\n').FirstOrDefault()?.Trim() ?? "llama.cpp"
            };
        }
        catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException or BadImageFormatException)
        {
            _logger.LogWarning(ex, "LlamaSharp native backend could not be loaded.");
            return EngineAvailability.Unavailable(
                "The llama.cpp native library is missing. Install a LLamaSharp.Backend.* package " +
                "matching this machine (Cpu, Cuda12, or Vulkan).");
        }
    }

    /// <summary>Reads llama.cpp's system banner, which lists the CPU feature flags in the build.</summary>
    private static string ReadSystemInfo()
    {
        var pointer = NativeApi.llama_print_system_info();
        return pointer == IntPtr.Zero
            ? string.Empty
            : System.Runtime.InteropServices.Marshal.PtrToStringAnsi(pointer) ?? string.Empty;
    }

    /// <summary>
    /// Lists the accelerator devices ggml registered — <c>CUDA0</c>, <c>Vulkan0</c> and so on.
    /// </summary>
    /// <remarks>
    /// This is the authoritative source for what the loaded native library can actually use, and
    /// its length is what decides whether a tensor split is meaningful: a two-card machine whose
    /// second GPU is masked by <c>CUDA_VISIBLE_DEVICES</c>, or whose driver only bound one, shows
    /// one device here and the split has nowhere to go. Each device is identified by its buffer
    /// type name, which is the only device string LLamaSharp surfaces. The CPU device is filtered
    /// out so the caller is left with accelerators alone.
    /// </remarks>
    private List<AcceleratorDevice> ReadAccelerators()
    {
        var accelerators = new List<AcceleratorDevice>();

        try
        {
            var count = (int)NativeApi.ggml_backend_dev_count();

            for (var i = 0; i < count; i++)
            {
                var device = NativeApi.ggml_backend_dev_get((UIntPtr)i);

                if (device == IntPtr.Zero)
                {
                    continue;
                }

                var bufferType = NativeApi.ggml_backend_dev_buffer_type(device);

                if (bufferType == IntPtr.Zero)
                {
                    continue;
                }

                var namePointer = NativeApi.ggml_backend_buft_name(bufferType);
                var name = namePointer == IntPtr.Zero
                    ? null
                    : System.Runtime.InteropServices.Marshal.PtrToStringAnsi(namePointer);

                if (!string.IsNullOrWhiteSpace(name) &&
                    !name.StartsWith("CPU", StringComparison.OrdinalIgnoreCase))
                {
                    // The index is the accelerator's position among accelerators, not among all
                    // ggml devices: that is what llama.cpp's tensor split addresses.
                    accelerators.Add(new AcceleratorDevice
                    {
                        Index = accelerators.Count,
                        Name = name
                    });
                }
            }
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
        {
            // An older native library without the device API; the CPU fallback still holds.
            _logger.LogDebug(ex, "ggml device enumeration is unavailable in this native build.");
        }

        return accelerators;
    }

    public bool CanServe(ModelDescriptor model) =>
        model.Format == ModelFormat.Gguf ||
        (model.Format == ModelFormat.Unknown &&
         model.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase));

    public async ValueTask<IModelSession> LoadAsync(
        ModelDescriptor model,
        ModelLoadOptions options,
        CancellationToken cancellationToken = default)
    {
        var availability = await ProbeAsync(cancellationToken).ConfigureAwait(false);
        if (!availability.IsAvailable)
        {
            throw new EngineNotAvailableException(Descriptor.DisplayName, availability.Reason);
        }

        if (!File.Exists(model.Path))
        {
            throw new ModelLoadException(model.Id, $"weights not found at '{model.Path}'");
        }

        var device = ResolveDevice(options.Device, availability.AvailableDevices);
        var parameters = BuildModelParams(model, options, device, availability.Accelerators);

        _logger.LogInformation(
            "Loading {Model} on {Device} (ctx={Context}, gpuLayers={GpuLayers})",
            model.Id, device, parameters.ContextSize, parameters.GpuLayerCount);

        try
        {
            var weights = await LLamaWeights
                .LoadFromFileAsync(parameters, cancellationToken)
                .ConfigureAwait(false);

            var projector = await LoadProjectorAsync(model, weights, device, cancellationToken)
                .ConfigureAwait(false);

            // Batching is skipped for a vision model even when it is switched on: an image is
            // encoded by the projector before any of it reaches the batch, so the work that
            // dominates a vision request is the part batching cannot overlap — and mixing the two
            // would mean maintaining a second multimodal path for no throughput gain.
            if (options.BatchedInference && !options.EmbeddingMode && projector is null)
            {
                return new LlamaSharpBatchedSession(
                    model,
                    weights,
                    parameters,
                    device,
                    options.MaxSequences,
                    _loggerFactory.CreateLogger<LlamaSharpBatchedSession>());
            }

            if (options.BatchedInference && projector is not null)
            {
                _logger.LogInformation(
                    "{Model} is a vision model; serving it serialised rather than batched.", model.Id);
            }

            return new LlamaSharpSession(
                model,
                weights,
                projector,
                parameters,
                device,
                _loggerFactory.CreateLogger<LlamaSharpSession>());
        }
        catch (Exception ex) when (ex is not OperationCanceledException and not ModelLoadException)
        {
            throw new ModelLoadException(model.Id, ex.Message, ex);
        }
    }

    /// <summary>
    /// Loads the multimodal projector beside a vision model, or returns null when it has none.
    /// </summary>
    /// <remarks>
    /// A GGUF vision model is two files: the language weights and an <c>mmproj</c> that encodes
    /// images into the same embedding space. A missing or unreadable projector is not fatal — the
    /// language weights are perfectly usable on their own, and refusing to load them would turn a
    /// half-downloaded vision model into no model at all. The session simply reports no vision
    /// and images are described in words.
    /// </remarks>
    private async Task<MtmdWeights?> LoadProjectorAsync(
        ModelDescriptor model,
        LLamaWeights weights,
        DeviceKind device,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrEmpty(model.ProjectorPath) || !File.Exists(model.ProjectorPath))
        {
            return null;
        }

        try
        {
            var parameters = MtmdContextParams.Default();
            parameters.UseGpu = device != DeviceKind.Cpu;

            // The marker has to be the one LocalGen writes into the prompt, so it is taken from
            // the native library rather than hard-coded on either side.
            parameters.MediaMarker = NativeApi.MtmdDefaultMarker();

            var projector = await MtmdWeights
                .LoadFromFileAsync(model.ProjectorPath, weights, parameters, cancellationToken)
                .ConfigureAwait(false);

            _logger.LogInformation(
                "Loaded multimodal projector for {Model} (vision={Vision}, audio={Audio}).",
                model.Id, projector.SupportsVision, projector.SupportsAudio);

            return projector;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(
                ex, "Projector at {Path} could not be loaded; {Model} will run without vision.",
                model.ProjectorPath, model.Id);

            return null;
        }
    }

    /// <summary>
    /// Turns the requested device into one the loaded backend can actually use, falling back to
    /// CPU rather than failing when a GPU was asked for but no GPU backend is installed.
    /// </summary>
    private DeviceKind ResolveDevice(DeviceKind requested, IReadOnlyList<DeviceKind> available)
    {
        if (requested == DeviceKind.Auto)
        {
            foreach (var preferred in (DeviceKind[])[DeviceKind.Cuda, DeviceKind.Metal, DeviceKind.Vulkan])
            {
                if (available.Contains(preferred))
                {
                    return preferred;
                }
            }

            return DeviceKind.Cpu;
        }

        if (requested != DeviceKind.Cpu && !available.Contains(requested))
        {
            _logger.LogWarning(
                "Device {Requested} is not available with the installed backend; falling back to CPU.",
                requested);
            return DeviceKind.Cpu;
        }

        return requested;
    }

    private ModelParams BuildModelParams(
        ModelDescriptor model,
        ModelLoadOptions options,
        DeviceKind device,
        IReadOnlyList<AcceleratorDevice> accelerators)
    {
        var parameters = new ModelParams(model.Path)
        {
            UseMemorymap = options.UseMemoryMap,
            UseMemoryLock = options.UseMemoryLock,
            Embeddings = options.EmbeddingMode
        };

        if (options.ContextSize is > 0)
        {
            parameters.ContextSize = (uint)options.ContextSize.Value;
        }
        else if (model.ContextLength > 0)
        {
            parameters.ContextSize = (uint)model.ContextLength;
        }

        // A GPU is only worth using if layers are actually offloaded to it; 0 layers on CUDA is
        // strictly slower than plain CPU because of the transfer overhead.
        parameters.GpuLayerCount = device == DeviceKind.Cpu
            ? 0
            : options.GpuLayerCount ?? int.MaxValue;

        if (options.ThreadCount is > 0)
        {
            parameters.Threads = options.ThreadCount.Value;
        }

        if (options.BatchSize is > 0)
        {
            parameters.BatchSize = (uint)options.BatchSize.Value;
        }

        if (device != DeviceKind.Cpu)
        {
            ApplyMultiGpu(parameters, options, accelerators);
        }

        if (options.EmbeddingMode)
        {
            // Mean pooling gives one vector per input, which is what an embeddings API returns.
            parameters.PoolingType = LLamaPoolingType.Mean;
        }

        return parameters;
    }

    /// <summary>
    /// Spreads the model over the accelerators present, following the configured split.
    /// </summary>
    /// <remarks>
    /// The resolution happens against the devices ggml actually registered rather than against the
    /// configuration alone, because the two disagree in exactly the case this feature exists for:
    /// a host whose second card is not visible to the build that is loaded. Warnings are logged
    /// rather than thrown — a wrong split should still load the model, on fewer GPUs.
    /// </remarks>
    private void ApplyMultiGpu(
        ModelParams parameters,
        ModelLoadOptions options,
        IReadOnlyList<AcceleratorDevice> accelerators)
    {
        var plan = TensorSplitPlan.Create(options.TensorSplit, accelerators.Count);

        foreach (var warning in plan.Warnings)
        {
            _logger.LogWarning("Multi-GPU configuration: {Warning}", warning);
        }

        if (!plan.IsAutomatic)
        {
            for (var i = 0; i < plan.Fractions.Count && i < parameters.TensorSplits.Length; i++)
            {
                parameters.TensorSplits[i] = plan.Fractions[i];
            }

            _logger.LogInformation("Tensor split: {Split}", plan.Describe(accelerators));
        }

        if (options.MainGpu is { } mainGpu)
        {
            if (accelerators.Count > 0 && mainGpu >= accelerators.Count)
            {
                _logger.LogWarning(
                    "MainGpu is set to {MainGpu} but only {Count} accelerator(s) are visible; using device 0.",
                    mainGpu, accelerators.Count);
            }
            else
            {
                parameters.MainGpu = mainGpu;
            }
        }

        parameters.SplitMode = options.SplitMode switch
        {
            GpuSplitMode.None => LLama.Native.GPUSplitMode.None,
            GpuSplitMode.Layer => LLama.Native.GPUSplitMode.Layer,
            GpuSplitMode.Row => LLama.Native.GPUSplitMode.Row,

            // Auto leaves the field unset so llama.cpp keeps its own default, which is Layer.
            _ => null
        };

        if (options.SplitMode == GpuSplitMode.Row && accelerators.Count > 1)
        {
            _logger.LogInformation(
                "Row split mode: every layer is computed across all {Count} devices, which needs a fast " +
                "link between them to beat layer splitting.",
                accelerators.Count);
        }
    }
}
