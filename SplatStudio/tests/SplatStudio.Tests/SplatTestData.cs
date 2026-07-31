using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;

namespace SplatStudio.Tests;

/// <summary>One decoded splat, parsed back out of the 32-byte-per-point .splat layout.</summary>
public readonly record struct DecodedSplat(
    float X, float Y, float Z,
    float ScaleX, float ScaleY, float ScaleZ,
    byte R, byte G, byte B, byte A,
    byte RotX, byte RotY, byte RotZ, byte RotW);

public static class SplatTestData
{
    public const int BytesPerPoint = 32;

    /// <summary>
    /// Reads a .splat buffer back into structured points, so tests assert against the
    /// format the browser actually consumes rather than against internal types.
    /// </summary>
    public static List<DecodedSplat> Decode(byte[] splatBytes)
    {
        Assert.True(splatBytes.Length % BytesPerPoint == 0,
            $"A .splat file must be a whole number of {BytesPerPoint}-byte records.");

        var points = new List<DecodedSplat>(splatBytes.Length / BytesPerPoint);
        for (int o = 0; o + BytesPerPoint <= splatBytes.Length; o += BytesPerPoint)
        {
            points.Add(new DecodedSplat(
                BitConverter.ToSingle(splatBytes, o + 0),
                BitConverter.ToSingle(splatBytes, o + 4),
                BitConverter.ToSingle(splatBytes, o + 8),
                BitConverter.ToSingle(splatBytes, o + 12),
                BitConverter.ToSingle(splatBytes, o + 16),
                BitConverter.ToSingle(splatBytes, o + 20),
                splatBytes[o + 24], splatBytes[o + 25], splatBytes[o + 26], splatBytes[o + 27],
                splatBytes[o + 28], splatBytes[o + 29], splatBytes[o + 30], splatBytes[o + 31]));
        }
        return points;
    }

    /// <summary>
    /// A deterministic synthetic photo: a bright off-centre sphere over a dark vignetted
    /// background. Shaped like the single-clear-subject images the heuristic is tuned for,
    /// so depth output is meaningful rather than uniform.
    /// </summary>
    public static byte[] CreateTestJpeg(int width = 512, int height = 512)
    {
        using var image = new Image<Rgba32>(width, height);

        float cx = width * 0.45f, cy = height * 0.42f;
        float radius = MathF.Min(width, height) * 0.28f;

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < width; x++)
                {
                    float dx = x - cx, dy = y - cy;
                    float dist = MathF.Sqrt(dx * dx + dy * dy);

                    if (dist < radius)
                    {
                        // Lambert-ish shading so the sphere has an internal gradient.
                        float shade = MathF.Max(0f, 1f - dist / radius);
                        byte v = (byte)Math.Clamp(60 + shade * 195f, 0, 255);
                        row[x] = new Rgba32(v, (byte)(v * 0.8f), (byte)(v * 0.55f), 255);
                    }
                    else
                    {
                        float t = Math.Clamp(dist / (radius * 3f), 0f, 1f);
                        byte v = (byte)Math.Clamp(40 * (1f - t), 0, 255);
                        row[x] = new Rgba32(v, v, (byte)(v + 8), 255);
                    }
                }
            }
        });

        using var ms = new MemoryStream();
        image.Save(ms, new JpegEncoder { Quality = 92 });
        return ms.ToArray();
    }

    /// <summary>A PNG with a fully transparent border, to exercise the alpha-skip path.</summary>
    public static byte[] CreateTransparentBorderPng(int size = 256)
    {
        using var image = new Image<Rgba32>(size, size);
        int margin = size / 4;

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < size; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < size; x++)
                {
                    bool inside = x >= margin && x < size - margin && y >= margin && y < size - margin;
                    row[x] = inside
                        ? new Rgba32(220, 120, 200, 255)
                        : new Rgba32(0, 0, 0, 0);
                }
            }
        });

        using var ms = new MemoryStream();
        image.SaveAsPng(ms);
        return ms.ToArray();
    }
}
