using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using Microsoft.Extensions.Logging;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SplatStudio.Application.Abstractions;
using SplatStudio.Domain.Enums;

namespace SplatStudio.Infrastructure.Splatting;

/// <summary>
/// GPU-accelerated sibling of <see cref="LocalHeuristicSplatEngine"/>. It runs the
/// exact same 2.5D depth heuristic — inverse-luminance blended with a centre-weighted
/// radial prior, box-blurred, then one Gaussian per pixel — but evaluates every pixel
/// in parallel on the GPU through ILGPU, which JIT-compiles the kernels below to PTX
/// and runs them on the CUDA device.
///
/// This does NOT make the output more accurate: it is the same approximation, just
/// faster, so the honesty caveat on <see cref="LocalHeuristicSplatEngine"/> applies
/// here unchanged. The win is throughput — it makes a much larger
/// <c>Splatting:MaxPoints</c> budget practical, since cost per point drops sharply.
///
/// Device selection prefers CUDA and falls back to OpenCL. If neither is present the
/// engine reports <see cref="IsAvailable"/> = false and the DI registration silently
/// falls back to the CPU engine, so a GPU-less machine still runs the app.
/// </summary>
public sealed class GpuSplatEngine : IGaussianSplatEngine, IDisposable
{
    private readonly ILogger<GpuSplatEngine> _logger;
    private readonly object _gate = new();

    private Context? _context;
    private Accelerator? _accelerator;
    private Action<Index1D, ArrayView<uint>, ArrayView<float>, int, int>? _depthKernel;
    private Action<Index1D, ArrayView<uint>, ArrayView<float>, ArrayView<GpuSplatRecord>, ArrayView<int>, int, int, float, float>? _emitKernel;
    private bool _disposed;

    public SplatEngineType EngineType => SplatEngineType.Gpu;

    /// <summary>Human-readable name of the device the kernels run on, for logs and the UI.</summary>
    public string DeviceName { get; private set; } = "unavailable";

    /// <summary>False when no CUDA/OpenCL device could be initialised on this machine.</summary>
    public bool IsAvailable { get; private set; }

    public GpuSplatEngine(ILogger<GpuSplatEngine> logger)
    {
        _logger = logger;
        TryInitialize();
    }

    private void TryInitialize()
    {
        try
        {
            _context = Context.Create(b => b.Cuda().OpenCL().EnableAlgorithms());

            // Prefer a real CUDA device; fall back to OpenCL before giving up.
            var device = _context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda)
                         ?? _context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.OpenCL);

            if (device is null)
            {
                _logger.LogInformation("GPU splat engine: no CUDA or OpenCL device found, engine unavailable.");
                _context.Dispose();
                _context = null;
                return;
            }

            _accelerator = device.CreateAccelerator(_context);
            DeviceName = $"{_accelerator.Name} ({_accelerator.AcceleratorType})";

            _depthKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<uint>, ArrayView<float>, int, int>(DepthKernel);
            _emitKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<uint>, ArrayView<float>, ArrayView<GpuSplatRecord>, ArrayView<int>, int, int, float, float>(BlurAndEmitKernel);

            IsAvailable = true;
            _logger.LogInformation("GPU splat engine initialised on {Device} ({Memory} MB).",
                DeviceName, _accelerator.MemorySize / (1024 * 1024));
        }
        catch (Exception ex)
        {
            // A missing driver, a headless VM, or an ILGPU/PTX mismatch all land here.
            _logger.LogWarning(ex, "GPU splat engine could not be initialised — falling back to the CPU engine.");
            _accelerator?.Dispose();
            _context?.Dispose();
            _accelerator = null;
            _context = null;
            IsAvailable = false;
        }
    }

    public async Task<GaussianSplatGenerationResult> GenerateAsync(
        Stream imageStream,
        int maxOutputPoints,
        CancellationToken ct = default)
    {
        if (!IsAvailable)
            return new GaussianSplatGenerationResult(false, null, 0, "GPU engine is not available on this machine.");

        try
        {
            using var image = await Image.LoadAsync<Rgba32>(imageStream, ct);

            var (targetWidth, targetHeight) = ComputeTargetSize(image.Width, image.Height, maxOutputPoints);
            image.Mutate(c => c.Resize(new ResizeOptions
            {
                Size = new Size(targetWidth, targetHeight),
                Mode = ResizeMode.Stretch,
                Sampler = KnownResamplers.Lanczos3
            }));

            int w = image.Width, h = image.Height;
            var pixels = new Rgba32[w * h];
            image.CopyPixelDataTo(pixels);
            var packed = MemoryMarshal.Cast<Rgba32, uint>(pixels).ToArray();

            var (bytes, count) = RunKernels(packed, w, h, ct);
            return new GaussianSplatGenerationResult(true, bytes, count, null);
        }
        catch (Exception ex)
        {
            return new GaussianSplatGenerationResult(false, null, 0, ex.Message);
        }
    }

    /// <summary>
    /// The accelerator has a single default stream and the engine is a singleton, so
    /// launches are serialised. In practice only the single conversion worker calls
    /// this, so the lock is never contended.
    /// </summary>
    private (byte[] Bytes, int Count) RunKernels(uint[] packedPixels, int w, int h, CancellationToken ct)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var accelerator = _accelerator!;
            int pixelCount = w * h;

            var spacing = MathF.Max(2f / w, 2f / h);
            var baseScale = spacing * 0.65f;
            const float depthRangeWorldUnits = 0.6f;

            using var pixelBuffer = accelerator.Allocate1D<uint>(pixelCount);
            using var depthBuffer = accelerator.Allocate1D<float>(pixelCount);
            using var outputBuffer = accelerator.Allocate1D<GpuSplatRecord>(pixelCount);
            using var counterBuffer = accelerator.Allocate1D<int>(1);

            pixelBuffer.CopyFromCPU(packedPixels);
            counterBuffer.MemSetToZero();

            ct.ThrowIfCancellationRequested();

            _depthKernel!(pixelCount, pixelBuffer.View, depthBuffer.View, w, h);
            _emitKernel!(pixelCount, pixelBuffer.View, depthBuffer.View, outputBuffer.View,
                counterBuffer.View, w, h, baseScale, depthRangeWorldUnits);

            accelerator.Synchronize();
            ct.ThrowIfCancellationRequested();

            var emitted = new int[1];
            counterBuffer.CopyToCPU(emitted);
            var count = Math.Clamp(emitted[0], 0, pixelCount);

            if (count == 0)
                return (Array.Empty<byte>(), 0);

            var records = new GpuSplatRecord[count];
            outputBuffer.View.SubView(0, count).CopyToCPU(records);

            // The record layout is byte-identical to the file format, so this is a
            // reinterpret rather than a conversion.
            return (MemoryMarshal.AsBytes<GpuSplatRecord>(records).ToArray(), count);
        }
    }

    // ---- Kernels ---------------------------------------------------------------
    // These run on the GPU: static, allocation-free, no exceptions, blittable args only.

    /// <summary>Per-pixel pseudo-depth in [0,1] — mirrors LocalHeuristicSplatEngine.ComputeDepthField.</summary>
    internal static void DepthKernel(Index1D index, ArrayView<uint> pixels, ArrayView<float> depth, int w, int h)
    {
        int i = index;
        if (i >= w * h) return;

        int x = i % w;
        int y = i / w;

        uint p = pixels[i];
        float r = (p & 0xFF) / 255f;
        float g = ((p >> 8) & 0xFF) / 255f;
        float b = ((p >> 16) & 0xFF) / 255f;

        float luminance = 0.299f * r + 0.587f * g + 0.114f * b;

        float nx = (x / (float)IntrinsicMath.Max(1, w - 1)) * 2f - 1f;
        float ny = (y / (float)IntrinsicMath.Max(1, h - 1)) * 2f - 1f;
        float radial = XMath.Sqrt(nx * nx + ny * ny) / 1.41421356f; // 0 centre, 1 corner
        float centreBias = 1f - radial;

        float d = 0.55f * luminance + 0.45f * centreBias;
        depth[i] = XMath.Clamp(d, 0f, 1f);
    }

    /// <summary>
    /// Fuses the 3x3 box blur and the splat emission into one pass — the blur reads
    /// neighbours straight from the depth buffer the previous kernel wrote, so there is
    /// no need to round-trip an intermediate blurred buffer through global memory.
    ///
    /// Fully transparent pixels are dropped, which makes the output count data-dependent;
    /// an atomic counter compacts the survivors. That means emission order differs from
    /// the CPU engine's row-major order, which is harmless — the format carries no
    /// ordering semantics and the viewer does not depth-sort.
    /// </summary>
    internal static void BlurAndEmitKernel(
        Index1D index,
        ArrayView<uint> pixels,
        ArrayView<float> depth,
        ArrayView<GpuSplatRecord> output,
        ArrayView<int> counter,
        int w,
        int h,
        float baseScale,
        float depthRangeWorldUnits)
    {
        int i = index;
        if (i >= w * h) return;

        uint p = pixels[i];
        byte a = (byte)((p >> 24) & 0xFF);
        if (a < 8) return; // skip fully transparent source pixels

        int x = i % w;
        int y = i / w;

        float sum = 0f;
        int n = 0;
        for (int dy = -1; dy <= 1; dy++)
        {
            int yy = y + dy;
            if (yy < 0 || yy >= h) continue;
            for (int dx = -1; dx <= 1; dx++)
            {
                int xx = x + dx;
                if (xx < 0 || xx >= w) continue;
                sum += depth[yy * w + xx];
                n++;
            }
        }
        float d = n > 0 ? sum / n : depth[i];

        float nx = (x / (float)IntrinsicMath.Max(1, w - 1)) * 2f - 1f;
        // Flip Y: image rows run top-to-bottom, world Y runs bottom-to-top.
        float ny = 1f - (y / (float)IntrinsicMath.Max(1, h - 1)) * 2f;
        float z = (d - 0.5f) * depthRangeWorldUnits;

        int slot = Atomic.Add(ref counter[0], 1);
        if (slot >= output.IntLength) return;

        output[slot] = new GpuSplatRecord
        {
            PosX = nx,
            PosY = ny,
            PosZ = z,
            ScaleX = baseScale,
            ScaleY = baseScale,
            ScaleZ = baseScale * 0.55f,
            R = (byte)(p & 0xFF),
            G = (byte)((p >> 8) & 0xFF),
            B = (byte)((p >> 16) & 0xFF),
            A = a,
            // Identity rotation in the format's 0..255 encoding.
            RotX = 128,
            RotY = 128,
            RotZ = 128,
            RotW = 255
        };
    }

    private static (int width, int height) ComputeTargetSize(int width, int height, int maxOutputPoints)
    {
        var aspect = (double)width / height;
        var targetHeight = (int)Math.Sqrt(Math.Max(64, maxOutputPoints) / aspect);
        var targetWidth = (int)(targetHeight * aspect);
        // The GPU path affords a much larger ceiling than the CPU engine's 512.
        return (Math.Clamp(targetWidth, 8, 2048), Math.Clamp(targetHeight, 8, 2048));
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            _accelerator?.Dispose();
            _context?.Dispose();
            _accelerator = null;
            _context = null;
            IsAvailable = false;
        }
    }
}
