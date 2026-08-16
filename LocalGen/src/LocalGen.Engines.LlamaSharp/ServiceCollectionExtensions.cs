using LLama.Native;
using LocalGen.Core.Engines;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;

namespace LocalGen.Engines.LlamaSharp;

public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Routes llama.cpp's native logging into the .NET logging pipeline.
    /// </summary>
    /// <remarks>
    /// Left alone, the native library writes its build banner and graph allocation details
    /// straight to stderr, which corrupts CLI output and the desktop console. Redirecting it
    /// means those lines land in the log store at Debug level, where they are useful for
    /// diagnostics but invisible during ordinary use.
    /// </remarks>
    public static void RedirectNativeLogging(ILoggerFactory loggerFactory)
    {
        var logger = loggerFactory.CreateLogger("llama.cpp");

        NativeLogConfig.llama_log_set((level, message) =>
        {
            var text = message.TrimEnd('\n');

            if (text.Length == 0)
            {
                return;
            }

            switch (level)
            {
                case LLamaLogLevel.Error:
                    logger.LogError("{Message}", text);
                    break;

                case LLamaLogLevel.Warning:
                    logger.LogWarning("{Message}", text);
                    break;

                default:
                    // Info from llama.cpp is load-time detail, which is Debug for LocalGen's purposes.
                    logger.LogDebug("{Message}", text);
                    break;
            }
        });
    }

    /// <summary>
    /// Registers the llama.cpp backend. Engines are registered as an enumerable so the runtime
    /// can pick between all installed backends at request time.
    /// </summary>
    /// <param name="preferGpu">
    /// Enables CUDA/Vulkan selection with CPU auto-fallback. Only has an effect when a GPU
    /// backend package is installed; the native loader is configured once per process.
    /// </param>
    public static IServiceCollection AddLlamaSharpEngine(
        this IServiceCollection services,
        bool preferGpu = true)
    {
        ConfigureNativeLibrary(preferGpu);
        services.TryAddEnumerable(ServiceDescriptor.Singleton<IInferenceEngine, LlamaSharpEngine>());

        // llama.cpp is the only backend that can quantize, so the capability arrives with it.
        services.TryAddSingleton<Core.Models.IModelQuantizer, LlamaSharpQuantizer>();

        return services;
    }

    /// <summary>
    /// llama.cpp resolves its native library once per process and throws if reconfigured after
    /// loading, so this is a no-op when the library is already up.
    /// </summary>
    private static void ConfigureNativeLibrary(bool preferGpu)
    {
        if (NativeLibraryConfig.LLama.LibraryHasLoaded)
        {
            return;
        }

        try
        {
            // LLamaSharp decides whether to use CUDA by comparing the machine's CUDA toolkit
            // version against the backends it ships. A machine whose primary toolkit is newer
            // than the shipped backend — CUDA 13 against a CUDA 12 build — is judged
            // incompatible and silently falls back to CPU, even though the driver runs those
            // binaries perfectly well. Naming the library outright skips that judgement.
            var accelerated = preferGpu ? FindAcceleratedLibrary() : null;

            if (accelerated is not null)
            {
                NativeLibraryConfig.LLama.WithLibrary(accelerated);
                return;
            }

            NativeLibraryConfig.All
                .WithCuda(preferGpu)
                .WithVulkan(preferGpu)
                .WithAutoFallback(true);
        }
        catch (InvalidOperationException)
        {
            // Raced with another loader; the existing configuration stands.
        }
    }

    /// <summary>
    /// Locates a GPU-capable native build next to the application, most preferred first.
    /// </summary>
    /// <remarks>
    /// Returns null when none is present, which is the ordinary case for a CPU-only build — the
    /// caller then falls back to LLamaSharp's own detection.
    /// </remarks>
    private static string? FindAcceleratedLibrary()
    {
        var runtimes = Path.Combine(AppContext.BaseDirectory, "runtimes");

        if (!Directory.Exists(runtimes))
        {
            return null;
        }

        var identifier = OperatingSystem.IsWindows()
            ? "win-x64"
            : OperatingSystem.IsMacOS() ? "osx-arm64" : "linux-x64";

        var library = OperatingSystem.IsWindows() ? "llama.dll"
            : OperatingSystem.IsMacOS() ? "libllama.dylib" : "libllama.so";

        foreach (var backend in (string[])["cuda12", "cuda11", "vulkan"])
        {
            var candidate = Path.Combine(runtimes, identifier, "native", backend, library);

            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return null;
    }
}
