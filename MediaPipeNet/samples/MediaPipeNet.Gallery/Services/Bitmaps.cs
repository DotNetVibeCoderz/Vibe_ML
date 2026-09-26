using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using MediaPipeNet.Imaging;

namespace MediaPipeNet.Gallery.Services;

/// <summary>Conversions between MediaPipe.NET images and Avalonia bitmaps.</summary>
public static class Bitmaps
{
    /// <summary>Copies an <see cref="MPImage"/> into a (reused when possible) writeable bitmap.</summary>
    public static WriteableBitmap ToBitmap(MPImage image, WriteableBitmap? reuse = null)
    {
        var bmp = reuse is not null && reuse.PixelSize.Width == image.Width && reuse.PixelSize.Height == image.Height
            ? reuse
            : new WriteableBitmap(new PixelSize(image.Width, image.Height), new Vector(96, 96), Avalonia.Platform.PixelFormat.Rgba8888, AlphaFormat.Unpremul);
        using var fb = bmp.Lock();
        var src = image.AsBytes();
        int rowBytes = image.Width * 4;
        unsafe
        {
            for (int y = 0; y < image.Height; y++)
            {
                var dst = new Span<byte>((byte*)fb.Address + (long)y * fb.RowBytes, rowBytes);
                src.Slice(y * rowBytes, rowBytes).CopyTo(dst);
            }
        }
        return bmp;
    }

    /// <summary>Builds a tinted RGBA bitmap from a [0,1] mask (alpha = value × opacity).</summary>
    public static WriteableBitmap MaskToBitmap(ReadOnlySpan<float> mask, int width, int height, byte r, byte g, byte b, float opacity)
    {
        var bmp = new WriteableBitmap(new PixelSize(width, height), new Vector(96, 96), Avalonia.Platform.PixelFormat.Rgba8888, AlphaFormat.Unpremul);
        using var fb = bmp.Lock();
        unsafe
        {
            for (int y = 0; y < height; y++)
            {
                byte* row = (byte*)fb.Address + (long)y * fb.RowBytes;
                for (int x = 0; x < width; x++)
                {
                    float v = mask[y * width + x];
                    row[4 * x] = r;
                    row[4 * x + 1] = g;
                    row[4 * x + 2] = b;
                    row[4 * x + 3] = (byte)Math.Clamp(v * opacity * 255f, 0, 255);
                }
            }
        }
        return bmp;
    }
}
