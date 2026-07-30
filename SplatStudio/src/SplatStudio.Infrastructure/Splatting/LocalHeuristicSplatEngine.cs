using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SplatStudio.Application.Abstractions;
using SplatStudio.Domain.Enums;

namespace SplatStudio.Infrastructure.Splatting;

/// <summary>
/// Fully offline, CPU-only, single-image "image -> 3D world" engine.
///
/// IMPORTANT (documented honestly, also see README "How conversion works"):
/// genuine 3D Gaussian Splatting is trained from many photos of the same
/// subject taken from different angles, via Structure-from-Motion (COLMAP)
/// followed by gradient-descent optimisation of millions of Gaussians —
/// a GPU-hours-scale job. A single flat photo does not contain enough
/// information to reconstruct true geometry. What this engine does instead
/// is a fast, deterministic "2.5D" reconstruction: it estimates a per-pixel
/// pseudo-depth from luminance and a center-weighted radial prior, then
/// emits one Gaussian splat per (downsampled) pixel positioned in that
/// depth field. The result is a real, freely-orbitable 3D point-cloud
/// world — convincing for portraits/objects with a clear focal subject —
/// but it is not a substitute for true multi-view reconstruction.
///
/// For production-grade quality, implement <see cref="IGaussianSplatEngine"/>
/// against a real trainer and register it instead (see
/// <see cref="ExternalApiSplatEngine"/> for the integration point), for
/// example nerfstudio/gsplat, Luma AI, or KIRI Engine.
/// </summary>
public class LocalHeuristicSplatEngine : IGaussianSplatEngine
{
    public SplatEngineType EngineType => SplatEngineType.LocalHeuristic;

    public async Task<GaussianSplatGenerationResult> GenerateAsync(
        Stream imageStream,
        int maxOutputPoints,
        CancellationToken ct = default)
    {
        try
        {
            using var image = await Image.LoadAsync<Rgba32>(imageStream, ct);

            var (targetWidth, targetHeight) = ComputeTargetSize(image.Width, image.Height, maxOutputPoints);

            image.Mutate(ctx => ctx.Resize(new ResizeOptions
            {
                Size = new Size(targetWidth, targetHeight),
                Mode = ResizeMode.Stretch,
                Sampler = KnownResamplers.Lanczos3
            }));

            int w = image.Width, h = image.Height;
            var depth = ComputeDepthField(image, w, h);
            depth = BoxBlur(depth, w, h, radius: 1);

            var points = BuildPoints(image, depth, w, h);
            var bytes = SplatFileWriter.Write(points);

            return new GaussianSplatGenerationResult(true, bytes, points.Count, null);
        }
        catch (Exception ex)
        {
            return new GaussianSplatGenerationResult(false, null, 0, ex.Message);
        }
    }

    private static (int width, int height) ComputeTargetSize(int width, int height, int maxOutputPoints)
    {
        var aspect = (double)width / height;
        var targetHeight = (int)Math.Sqrt(Math.Max(64, maxOutputPoints) / aspect);
        var targetWidth = (int)(targetHeight * aspect);
        return (Math.Clamp(targetWidth, 8, 512), Math.Clamp(targetHeight, 8, 512));
    }

    /// <summary>Per-pixel pseudo-depth in [0,1]: a blend of inverse-luminance and a center-weighted radial prior.</summary>
    private static float[] ComputeDepthField(Image<Rgba32> image, int w, int h)
    {
        var depth = new float[w * h];

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                var ny = (y / (float)Math.Max(1, h - 1)) * 2f - 1f;

                for (int x = 0; x < w; x++)
                {
                    var pixel = row[x];
                    var luminance = (0.299f * pixel.R + 0.587f * pixel.G + 0.114f * pixel.B) / 255f;

                    var nx = (x / (float)Math.Max(1, w - 1)) * 2f - 1f;
                    var radialDistance = MathF.Sqrt(nx * nx + ny * ny) / MathF.Sqrt(2f); // 0 center, 1 corner
                    var centerBias = 1f - radialDistance;

                    // Brighter and more central pixels are pulled toward the camera.
                    depth[y * w + x] = Math.Clamp(0.55f * luminance + 0.45f * centerBias, 0f, 1f);
                }
            }
        });

        return depth;
    }

    private static float[] BoxBlur(float[] source, int w, int h, int radius)
    {
        var result = new float[source.Length];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float sum = 0;
                int count = 0;
                for (int dy = -radius; dy <= radius; dy++)
                {
                    var yy = y + dy;
                    if (yy < 0 || yy >= h) continue;
                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        var xx = x + dx;
                        if (xx < 0 || xx >= w) continue;
                        sum += source[yy * w + xx];
                        count++;
                    }
                }
                result[y * w + x] = count > 0 ? sum / count : source[y * w + x];
            }
        }
        return result;
    }

    private static List<SplatPoint> BuildPoints(Image<Rgba32> image, float[] depth, int w, int h)
    {
        var points = new List<SplatPoint>(w * h);
        var spacing = MathF.Max(2f / w, 2f / h);
        var baseScale = spacing * 0.65f;
        const float depthRangeWorldUnits = 0.6f;

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                // Flip Y: image rows go top-to-bottom, world Y should go bottom-to-top.
                var ny = 1f - (y / (float)Math.Max(1, h - 1)) * 2f;

                for (int x = 0; x < w; x++)
                {
                    var pixel = row[x];
                    if (pixel.A < 8) continue; // skip fully transparent source pixels

                    var nx = (x / (float)Math.Max(1, w - 1)) * 2f - 1f;
                    var d = depth[y * w + x];
                    var z = (d - 0.5f) * depthRangeWorldUnits;

                    points.Add(new SplatPoint(
                        Position: new Vector3(nx, ny, z),
                        Scale: new Vector3(baseScale, baseScale, baseScale * 0.55f),
                        Color: (pixel.R, pixel.G, pixel.B, pixel.A),
                        Rotation: SplatFileWriter.IdentityRotation));
                }
            }
        });

        return points;
    }
}
