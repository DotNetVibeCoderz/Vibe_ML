using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace MediaPipeNet.Imaging;

/// <summary>Layout of raw interleaved 8-bit pixel data.</summary>
public enum PixelFormat
{
    /// <summary>Red, green, blue, alpha.</summary>
    Rgba32,
    /// <summary>Blue, green, red, alpha (Windows / Avalonia / Skia native order).</summary>
    Bgra32,
    /// <summary>Red, green, blue.</summary>
    Rgb24,
    /// <summary>Blue, green, red (OpenCV's default order).</summary>
    Bgr24,
    /// <summary>Single 8-bit luminance channel.</summary>
    Gray8,
}

/// <summary>
/// An RGBA image held in a pooled, contiguous pixel buffer — the input type of every MediaPipe.NET
/// task (the equivalent of MediaPipe's <c>mp.Image</c>).
/// </summary>
/// <remarks>
/// The pixel buffer is rented from <see cref="ArrayPool{T}"/>; dispose the frame to return it.
/// For video, reuse one frame and refill it with <see cref="CopyFrom(ReadOnlySpan{byte}, int, int, PixelFormat, int)"/>
/// to avoid per-frame allocations. Instances are not thread-safe for concurrent writes but may be
/// read concurrently.
/// </remarks>
public sealed class MPImage : IDisposable
{
    private Rgba32[]? _buffer;

    /// <summary>Width in pixels.</summary>
    public int Width { get; private set; }

    /// <summary>Height in pixels.</summary>
    public int Height { get; private set; }

    /// <summary>The image size.</summary>
    public ImageSize Size => new(Width, Height);

    /// <summary>Number of pixels.</summary>
    public int PixelCount => Width * Height;

    /// <summary>True once the frame was disposed.</summary>
    public bool IsDisposed => _buffer is null;

    private MPImage(int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        Width = width;
        Height = height;
        _buffer = ArrayPool<Rgba32>.Shared.Rent(width * height);
    }

    /// <summary>Creates a black, fully transparent frame.</summary>
    public static MPImage Create(int width, int height)
    {
        var f = new MPImage(width, height);
        f.GetPixelSpan().Clear();
        return f;
    }

    /// <summary>Read-only view of the pixels in row-major order.</summary>
    public ReadOnlySpan<Rgba32> Pixels => GetPixelSpan();

    /// <summary>Mutable view of the pixels in row-major order.</summary>
    public Span<Rgba32> GetPixelSpan()
    {
        ObjectDisposedException.ThrowIf(_buffer is null, this);
        return _buffer.AsSpan(0, Width * Height);
    }

    /// <summary>Read-only view of one row of pixels.</summary>
    public ReadOnlySpan<Rgba32> GetRow(int y) => GetPixelSpan().Slice(y * Width, Width);

    /// <summary>Gets a single pixel.</summary>
    public Rgba32 this[int x, int y] => GetPixelSpan()[y * Width + x];

    /// <summary>The pixels as raw RGBA bytes.</summary>
    public ReadOnlySpan<byte> AsBytes() => MemoryMarshal.AsBytes(Pixels);

    // ---------------------------------------------------------------- loading

    /// <summary>Decodes an image file (JPEG, PNG, BMP, GIF, WebP, TIFF...), honoring EXIF orientation.</summary>
    public static MPImage Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using var image = Image.Load<Rgba32>(path);
        image.Mutate(static x => x.AutoOrient());
        return FromImage(image);
    }

    /// <summary>Decodes an image file asynchronously.</summary>
    public static async Task<MPImage> LoadAsync(string path, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using var image = await Image.LoadAsync<Rgba32>(path, cancellationToken).ConfigureAwait(false);
        image.Mutate(static x => x.AutoOrient());
        return FromImage(image);
    }

    /// <summary>Decodes an encoded image from a stream.</summary>
    public static MPImage Load(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using var image = Image.Load<Rgba32>(stream);
        image.Mutate(static x => x.AutoOrient());
        return FromImage(image);
    }

    /// <summary>Decodes an encoded image from a stream asynchronously.</summary>
    public static async Task<MPImage> LoadAsync(Stream stream, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using var image = await Image.LoadAsync<Rgba32>(stream, cancellationToken).ConfigureAwait(false);
        image.Mutate(static x => x.AutoOrient());
        return FromImage(image);
    }

    /// <summary>Decodes an encoded image (JPEG/PNG/BMP bytes).</summary>
    public static MPImage Load(ReadOnlySpan<byte> encoded)
    {
        using var image = Image.Load<Rgba32>(encoded);
        image.Mutate(static x => x.AutoOrient());
        return FromImage(image);
    }

    /// <summary>Copies an ImageSharp image into a new frame.</summary>
    public static MPImage FromImage(Image<Rgba32> image)
    {
        ArgumentNullException.ThrowIfNull(image);
        var frame = new MPImage(image.Width, image.Height);
        image.CopyPixelDataTo(frame.GetPixelSpan());
        return frame;
    }

    /// <summary>Copies an ImageSharp image of any pixel type into a new frame.</summary>
    public static MPImage FromImage(Image image)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (image is Image<Rgba32> rgba) return FromImage(rgba);
        using var converted = image.CloneAs<Rgba32>();
        return FromImage(converted);
    }

    /// <summary>Creates a frame from raw interleaved 8-bit pixel data.</summary>
    /// <param name="data">Pixel bytes.</param>
    /// <param name="width">Width in pixels.</param>
    /// <param name="height">Height in pixels.</param>
    /// <param name="format">Channel layout of <paramref name="data"/>.</param>
    /// <param name="stride">Bytes per row; 0 means tightly packed.</param>
    public static MPImage FromPixelData(ReadOnlySpan<byte> data, int width, int height, PixelFormat format, int stride = 0)
    {
        var frame = new MPImage(width, height);
        frame.CopyFromCore(data, format, stride);
        return frame;
    }

    /// <summary>Creates a frame from RGBA pixels.</summary>
    public static MPImage FromPixels(ReadOnlySpan<Rgba32> pixels, int width, int height)
    {
        if (pixels.Length < width * height) throw new ArgumentException("Not enough pixels for the given size.", nameof(pixels));
        var frame = new MPImage(width, height);
        pixels[..(width * height)].CopyTo(frame.GetPixelSpan());
        return frame;
    }

    /// <summary>
    /// Refills this frame from raw pixel data, reallocating only when the new image is larger than
    /// the current buffer. Ideal for reusing one frame across video frames.
    /// </summary>
    public void CopyFrom(ReadOnlySpan<byte> data, int width, int height, PixelFormat format, int stride = 0)
    {
        EnsureSize(width, height);
        CopyFromCore(data, format, stride);
    }

    /// <summary>Refills this frame with the pixels of another frame.</summary>
    public void CopyFrom(MPImage other)
    {
        ArgumentNullException.ThrowIfNull(other);
        EnsureSize(other.Width, other.Height);
        other.Pixels.CopyTo(GetPixelSpan());
    }

    private void EnsureSize(int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ObjectDisposedException.ThrowIf(_buffer is null, this);
        if (_buffer.Length < width * height)
        {
            ArrayPool<Rgba32>.Shared.Return(_buffer);
            _buffer = ArrayPool<Rgba32>.Shared.Rent(width * height);
        }
        Width = width;
        Height = height;
    }

    private void CopyFromCore(ReadOnlySpan<byte> data, PixelFormat format, int stride)
    {
        int bpp = format switch
        {
            PixelFormat.Rgba32 or PixelFormat.Bgra32 => 4,
            PixelFormat.Rgb24 or PixelFormat.Bgr24 => 3,
            PixelFormat.Gray8 => 1,
            _ => throw new ArgumentOutOfRangeException(nameof(format)),
        };
        int rowBytes = Width * bpp;
        if (stride == 0) stride = rowBytes;
        if (stride < rowBytes) throw new ArgumentException("Stride is smaller than one row of pixels.", nameof(stride));
        if (data.Length < stride * (Height - 1) + rowBytes) throw new ArgumentException("Pixel data is too small for the given size.", nameof(data));

        var dst = GetPixelSpan();
        for (int y = 0; y < Height; y++)
        {
            var src = data.Slice(y * stride, rowBytes);
            var row = dst.Slice(y * Width, Width);
            switch (format)
            {
                case PixelFormat.Rgba32:
                    MemoryMarshal.Cast<byte, Rgba32>(src).CopyTo(row);
                    break;
                case PixelFormat.Bgra32:
                    for (int x = 0, i = 0; x < row.Length; x++, i += 4) row[x] = new Rgba32(src[i + 2], src[i + 1], src[i], src[i + 3]);
                    break;
                case PixelFormat.Rgb24:
                    for (int x = 0, i = 0; x < row.Length; x++, i += 3) row[x] = new Rgba32(src[i], src[i + 1], src[i + 2], 255);
                    break;
                case PixelFormat.Bgr24:
                    for (int x = 0, i = 0; x < row.Length; x++, i += 3) row[x] = new Rgba32(src[i + 2], src[i + 1], src[i], 255);
                    break;
                case PixelFormat.Gray8:
                    for (int x = 0; x < row.Length; x++) row[x] = new Rgba32(src[x], src[x], src[x], 255);
                    break;
            }
        }
    }

    // ---------------------------------------------------------------- export

    /// <summary>Copies the frame into a new ImageSharp image.</summary>
    public Image<Rgba32> ToImage() => Image.LoadPixelData<Rgba32>(Pixels, Width, Height);

    /// <summary>Writes the pixels in another channel order (e.g. BGRA for Avalonia/Skia bitmaps).</summary>
    public void CopyTo(Span<byte> destination, PixelFormat format, int stride = 0)
    {
        int bpp = format is PixelFormat.Rgba32 or PixelFormat.Bgra32 ? 4 : format is PixelFormat.Gray8 ? 1 : 3;
        if (stride == 0) stride = Width * bpp;
        var src = Pixels;
        for (int y = 0; y < Height; y++)
        {
            var row = src.Slice(y * Width, Width);
            var dst = destination.Slice(y * stride, Width * bpp);
            for (int x = 0, i = 0; x < row.Length; x++, i += bpp)
            {
                var p = row[x];
                switch (format)
                {
                    case PixelFormat.Rgba32: dst[i] = p.R; dst[i + 1] = p.G; dst[i + 2] = p.B; dst[i + 3] = p.A; break;
                    case PixelFormat.Bgra32: dst[i] = p.B; dst[i + 1] = p.G; dst[i + 2] = p.R; dst[i + 3] = p.A; break;
                    case PixelFormat.Rgb24: dst[i] = p.R; dst[i + 1] = p.G; dst[i + 2] = p.B; break;
                    case PixelFormat.Bgr24: dst[i] = p.B; dst[i + 1] = p.G; dst[i + 2] = p.R; break;
                    case PixelFormat.Gray8: dst[i] = (byte)((p.R * 77 + p.G * 150 + p.B * 29) >> 8); break;
                }
            }
        }
    }

    /// <summary>Encodes the frame as PNG.</summary>
    public void SaveAsPng(string path)
    {
        using var image = ToImage();
        image.SaveAsPng(path);
    }

    /// <summary>Encodes the frame as JPEG.</summary>
    public void SaveAsJpeg(string path, int quality = 90)
    {
        using var image = ToImage();
        image.SaveAsJpeg(path, new SixLabors.ImageSharp.Formats.Jpeg.JpegEncoder { Quality = quality });
    }

    /// <summary>Creates an independent copy of the frame.</summary>
    public MPImage Clone()
    {
        var copy = new MPImage(Width, Height);
        Pixels.CopyTo(copy.GetPixelSpan());
        return copy;
    }

    /// <summary>Creates a horizontally mirrored copy (useful for selfie-view webcams).</summary>
    public MPImage FlipHorizontal()
    {
        var copy = new MPImage(Width, Height);
        var src = Pixels;
        var dst = copy.GetPixelSpan();
        for (int y = 0; y < Height; y++)
        {
            var s = src.Slice(y * Width, Width);
            var d = dst.Slice(y * Width, Width);
            s.CopyTo(d);
            d.Reverse();
        }
        return copy;
    }

    /// <summary>Returns the pixel buffer to the pool.</summary>
    public void Dispose()
    {
        var buffer = Interlocked.Exchange(ref _buffer, null);
        if (buffer is not null) ArrayPool<Rgba32>.Shared.Return(buffer);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal ref readonly Rgba32 PixelRef(int x, int y) => ref _buffer![y * Width + x];
}
