using System.Numerics;

namespace SplatStudio.Infrastructure.Splatting;

/// <summary>One Gaussian splat: a 3D position, an anisotropic scale, a color and a rotation.</summary>
public readonly record struct SplatPoint(
    Vector3 Position,
    Vector3 Scale,
    (byte R, byte G, byte B, byte A) Color,
    (byte X, byte Y, byte Z, byte W) Rotation);

/// <summary>
/// Writes the compact, widely-supported binary ".splat" layout: 32 bytes
/// per point — 3 float32 position, 3 float32 scale, 4 uint8 RGBA color,
/// 4 uint8 quaternion (component*127.5+127.5, so 0..255 maps to -1..1).
/// This is the format consumed by wwwroot/js/splat-viewer.js in the browser.
/// </summary>
public static class SplatFileWriter
{
    public const int BytesPerPoint = 32;

    public static byte[] Write(IReadOnlyList<SplatPoint> points)
    {
        var buffer = new byte[points.Count * BytesPerPoint];
        var span = buffer.AsSpan();

        for (int i = 0; i < points.Count; i++)
        {
            var p = points[i];
            var offset = i * BytesPerPoint;

            BitConverter.TryWriteBytes(span.Slice(offset, 4), p.Position.X);
            BitConverter.TryWriteBytes(span.Slice(offset + 4, 4), p.Position.Y);
            BitConverter.TryWriteBytes(span.Slice(offset + 8, 4), p.Position.Z);

            BitConverter.TryWriteBytes(span.Slice(offset + 12, 4), p.Scale.X);
            BitConverter.TryWriteBytes(span.Slice(offset + 16, 4), p.Scale.Y);
            BitConverter.TryWriteBytes(span.Slice(offset + 20, 4), p.Scale.Z);

            span[offset + 24] = p.Color.R;
            span[offset + 25] = p.Color.G;
            span[offset + 26] = p.Color.B;
            span[offset + 27] = p.Color.A;

            span[offset + 28] = p.Rotation.X;
            span[offset + 29] = p.Rotation.Y;
            span[offset + 30] = p.Rotation.Z;
            span[offset + 31] = p.Rotation.W;
        }

        return buffer;
    }

    /// <summary>Identity rotation encoded in the 0..255 byte range used by the format.</summary>
    public static readonly (byte X, byte Y, byte Z, byte W) IdentityRotation = (128, 128, 128, 255);
}
