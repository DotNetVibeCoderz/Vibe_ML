using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;

namespace MediaPipeNet.Inference;

/// <summary>
/// Discovers which ONNX Runtime execution providers the loaded native runtime offers and
/// configures <see cref="SessionOptions"/> for a requested provider.
/// </summary>
public static class ExecutionProviderSelector
{
    private static readonly Lazy<IReadOnlyList<ExecutionProvider>> s_available = new(Probe);

    /// <summary>The auto-selection order.</summary>
    public static IReadOnlyList<ExecutionProvider> PreferenceOrder { get; } =
        [ExecutionProvider.Cuda, ExecutionProvider.DirectML, ExecutionProvider.CoreML, ExecutionProvider.Cpu];

    /// <summary>
    /// Providers compiled into the native ONNX Runtime library that is loaded in this process.
    /// A listed provider can still fail at session creation (e.g. CUDA without a CUDA runtime).
    /// </summary>
    public static IReadOnlyList<ExecutionProvider> GetAvailableProviders() => s_available.Value;

    /// <summary>The native ONNX Runtime version, or null when the native library cannot be loaded.</summary>
    public static string? GetRuntimeVersion()
    {
        try { return OrtEnv.Instance().GetVersionString(); }
        catch (Exception e) when (e is DllNotFoundException or TypeInitializationException or EntryPointNotFoundException) { return null; }
    }

    /// <summary>Returns the ordered list of providers to try for <paramref name="requested"/>.</summary>
    public static IReadOnlyList<ExecutionProvider> GetCandidates(ExecutionProvider requested, bool fallbackToCpu)
    {
        var available = GetAvailableProviders();
        if (requested == ExecutionProvider.Auto)
            return PreferenceOrder.Where(available.Contains).DefaultIfEmpty(ExecutionProvider.Cpu).ToArray();
        return requested == ExecutionProvider.Cpu || !fallbackToCpu
            ? [requested]
            : [requested, ExecutionProvider.Cpu];
    }

    /// <summary>Creates session options configured for <paramref name="provider"/>.</summary>
    public static SessionOptions CreateSessionOptions(ExecutionProvider provider, InferenceOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var so = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
        };
        if (options.IntraOpThreads > 0) so.IntraOpNumThreads = options.IntraOpThreads;
        if (options.InterOpThreads > 0) so.InterOpNumThreads = options.InterOpThreads;
        if (options.EnableProfiling) so.EnableProfiling = true;
        try
        {
            switch (provider)
            {
                case ExecutionProvider.Cuda:
                    so.AppendExecutionProvider_CUDA(options.DeviceId);
                    break;
                case ExecutionProvider.DirectML:
                    // DirectML requires memory patterns off and sequential execution.
                    so.EnableMemoryPattern = false;
                    so.AppendExecutionProvider_DML(options.DeviceId);
                    break;
                case ExecutionProvider.CoreML:
                    so.AppendExecutionProvider("CoreML", new Dictionary<string, string>());
                    break;
            }
        }
        catch
        {
            so.Dispose();
            throw;
        }
        return so;
    }

    /// <summary>Maps an ONNX Runtime provider name to <see cref="ExecutionProvider"/>.</summary>
    public static ExecutionProvider? FromOrtName(string name) => name switch
    {
        "CUDAExecutionProvider" => ExecutionProvider.Cuda,
        "DmlExecutionProvider" => ExecutionProvider.DirectML,
        "CoreMLExecutionProvider" => ExecutionProvider.CoreML,
        "CPUExecutionProvider" => ExecutionProvider.Cpu,
        _ => null,
    };

    private static IReadOnlyList<ExecutionProvider> Probe()
    {
        try
        {
            return OrtEnv.Instance().GetAvailableProviders()
                .Select(FromOrtName)
                .Where(p => p is not null)
                .Select(p => p!.Value)
                .Distinct()
                .ToArray();
        }
        catch (Exception e) when (e is DllNotFoundException or TypeInitializationException or EntryPointNotFoundException)
        {
            return [];
        }
    }

    internal static void LogSelection(ILogger logger, string model, ExecutionProvider provider, Exception? previousFailure)
    {
        if (previousFailure is not null)
            logger.LogWarning(previousFailure, "Model {Model}: preferred execution provider failed, using {Provider}", model, provider);
        else
            logger.LogDebug("Model {Model} runs on {Provider}", model, provider);
    }
}
