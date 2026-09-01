using System.Diagnostics;
using LLama.Native;
using LocalGen.Core;
using LocalGen.Core.Models;
using Microsoft.Extensions.Logging;

namespace LocalGen.Engines.LlamaSharp;

/// <summary>
/// Quantizes GGUF weights through llama.cpp.
/// </summary>
/// <remarks>
/// Useful for two things: shrinking a model that was published only at full precision, and
/// trading quality for memory on a machine where the published quantization does not fit. The
/// conversion is CPU-bound and single-file — it reads the whole source and writes a new one, so
/// there must be room for both.
/// </remarks>
public sealed class LlamaSharpQuantizer : IModelQuantizer
{
    private readonly ILogger<LlamaSharpQuantizer> _logger;

    public LlamaSharpQuantizer(ILogger<LlamaSharpQuantizer> logger) => _logger = logger;

    /// <summary>
    /// The quantizations worth offering, smallest first.
    /// </summary>
    /// <remarks>
    /// llama.cpp accepts many more, including the IQ family, but those need an importance matrix
    /// to be worth using and produce poor results without one. Offering them here would invite a
    /// bad result, so the list stays with the K-quants and the legacy formats.
    /// </remarks>
    public IReadOnlyList<Core.Models.Quantization> Supported { get; } =
    [
        Find("Q2_K"),
        Find("Q3_K_M"),
        Find("Q4_K_M"),
        Find("Q5_K_M"),
        Find("Q6_K"),
        Find("Q8_0"),
        Find("F16")
    ];

    private static readonly Dictionary<string, LLamaFtype> FileTypes =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ["Q2_K"] = LLamaFtype.MOSTLY_Q2_K,
            ["Q3_K_S"] = LLamaFtype.MOSTLY_Q3_K_S,
            ["Q3_K_M"] = LLamaFtype.MOSTLY_Q3_K_M,
            ["Q3_K_L"] = LLamaFtype.MOSTLY_Q3_K_L,
            ["Q4_0"] = LLamaFtype.MOSTLY_Q4_0,
            ["Q4_1"] = LLamaFtype.MOSTLY_Q4_1,
            ["Q4_K_S"] = LLamaFtype.MOSTLY_Q4_K_S,
            ["Q4_K_M"] = LLamaFtype.MOSTLY_Q4_K_M,
            ["Q5_0"] = LLamaFtype.MOSTLY_Q5_0,
            ["Q5_1"] = LLamaFtype.MOSTLY_Q5_1,
            ["Q5_K_S"] = LLamaFtype.MOSTLY_Q5_K_S,
            ["Q5_K_M"] = LLamaFtype.MOSTLY_Q5_K_M,
            ["Q6_K"] = LLamaFtype.MOSTLY_Q6_K,
            ["Q8_0"] = LLamaFtype.MOSTLY_Q8_0,
            ["F16"] = LLamaFtype.MOSTLY_F16,
            ["BF16"] = LLamaFtype.MOSTLY_BF16,
            ["F32"] = LLamaFtype.ALL_F32
        };

    public bool CanProduce(string quantization) => FileTypes.ContainsKey(quantization.Trim());

    public async Task<QuantizationResult> QuantizeAsync(
        QuantizationRequest request,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(request.SourcePath))
        {
            throw new LocalGenException($"No model file at '{request.SourcePath}'.");
        }

        if (!FileTypes.TryGetValue(request.Quantization.Trim(), out var fileType))
        {
            throw new LocalGenException(
                $"'{request.Quantization}' is not a quantization LocalGen can produce. " +
                $"Choose one of: {string.Join(", ", Supported.Select(static q => q.Name))}");
        }

        var source = new FileInfo(request.SourcePath);

        // Reading and writing whole models at once, so the disk has to hold both. The source size
        // is a safe upper bound for the output, which is always smaller when quantizing down.
        EnsureDiskSpace(request.TargetPath, source.Length);

        var directory = Path.GetDirectoryName(Path.GetFullPath(request.TargetPath));

        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        _logger.LogInformation(
            "Quantizing {Source} to {Quantization} → {Target}",
            source.Name, request.Quantization, Path.GetFileName(request.TargetPath));

        var stopwatch = Stopwatch.StartNew();

        // Blocking and CPU-bound; kept off the calling thread so a request or the UI stays alive.
        await Task.Run(() =>
        {
            // LLamaSharp installs its DLL import resolver from NativeApi's static constructor.
            // Reading default parameters calls into the native library through a different type,
            // which would run before that constructor and fail to find 'llama' — so NativeApi is
            // touched first to force the loader.
            _ = NativeApi.llama_max_devices();

            var parameters = LLamaModelQuantizeParams.Default();

            parameters.ftype = fileType;
            parameters.nthread = request.ThreadCount ?? 0;
            parameters.allow_requantize = request.AllowRequantize;
            parameters.quantize_output_tensor = request.QuantizeOutputTensor;

            var status = NativeApi.llama_model_quantize(
                request.SourcePath,
                request.TargetPath,
                ref parameters);

            if (status != 0)
            {
                // A non-zero return usually means the source is already quantized and
                // allow_requantize was left off, which is the case worth naming.
                throw new LocalGenException(
                    $"llama.cpp could not quantize '{source.Name}' (status {status}). " +
                    "If the source is already quantized, requantizing needs to be allowed " +
                    "explicitly — though converting the original weights gives a better result.");
            }
        }, cancellationToken).ConfigureAwait(false);

        cancellationToken.ThrowIfCancellationRequested();

        var target = new FileInfo(request.TargetPath);

        _logger.LogInformation(
            "Quantized to {Target}: {SourceMb:N0} MB → {TargetMb:N0} MB in {Elapsed:N0}s",
            target.Name,
            source.Length / 1024d / 1024,
            target.Length / 1024d / 1024,
            stopwatch.Elapsed.TotalSeconds);

        return new QuantizationResult
        {
            TargetPath = request.TargetPath,
            Quantization = request.Quantization,
            SourceBytes = source.Length,
            TargetBytes = target.Length,
            Duration = stopwatch.Elapsed
        };
    }

    /// <summary>
    /// Fails early when the volume cannot hold the output, rather than after minutes of work.
    /// </summary>
    private void EnsureDiskSpace(string targetPath, long required)
    {
        try
        {
            var root = Path.GetPathRoot(Path.GetFullPath(targetPath));

            if (string.IsNullOrEmpty(root))
            {
                return;
            }

            var available = new DriveInfo(root).AvailableFreeSpace;

            if (available < required)
            {
                throw new LocalGenException(
                    $"Not enough space on {root}: {available / 1024 / 1024:N0} MB free, " +
                    $"but up to {required / 1024 / 1024:N0} MB is needed.");
            }
        }
        catch (Exception ex) when (ex is ArgumentException or IOException or UnauthorizedAccessException)
        {
            // A path whose volume cannot be inspected — a network share, say. Not worth blocking on.
            _logger.LogDebug(ex, "Could not check free space for {Path}", targetPath);
        }
    }

    private static Core.Models.Quantization Find(string name) =>
        Core.Models.Quantization.Known.FirstOrDefault(
            q => string.Equals(q.Name, name, StringComparison.OrdinalIgnoreCase))
        ?? new Core.Models.Quantization(name, 0, string.Empty);
}
