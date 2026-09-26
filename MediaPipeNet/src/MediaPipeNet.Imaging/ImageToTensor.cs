using System.Runtime.CompilerServices;
using SixLabors.ImageSharp.PixelFormats;

namespace MediaPipeNet.Imaging;

/// <summary>How pixels outside the source image are filled when a region of interest extends past its edges.</summary>
public enum BorderMode
{
    /// <summary>Outside pixels are black (they become <see cref="ImageToTensorOptions.RangeMin"/> after normalization).</summary>
    Zero,
    /// <summary>Outside pixels repeat the nearest edge pixel.</summary>
    Replicate,
}

/// <summary>Options for <see cref="ImageToTensor"/> conversions.</summary>
/// <param name="Width">Tensor width in pixels.</param>
/// <param name="Height">Tensor height in pixels.</param>
/// <param name="RangeMin">Value that a 0 channel maps to.</param>
/// <param name="RangeMax">Value that a 255 channel maps to.</param>
/// <param name="KeepAspectRatio">Letterbox the ROI instead of stretching it to the tensor's aspect ratio.</param>
/// <param name="BorderMode">How pixels outside the image are filled.</param>
/// <param name="Antialias">Average several samples per tensor pixel when downscaling strongly (reduces aliasing).</param>
public readonly record struct ImageToTensorOptions(
    int Width,
    int Height,
    float RangeMin = 0f,
    float RangeMax = 1f,
    bool KeepAspectRatio = false,
    BorderMode BorderMode = BorderMode.Zero,
    bool Antialias = true);

/// <summary>
/// Describes how a model input tensor maps back onto the source image: the (possibly padded)
/// rotated ROI that was sampled, plus the letterbox padding. Use it to project model outputs
/// (landmarks, keypoints, masks) back to image coordinates.
/// </summary>
/// <param name="Roi">The rotated rectangle actually sampled, including letterbox padding, in normalized image coordinates.</param>
/// <param name="Padding">Letterbox padding as fractions of the tensor.</param>
/// <param name="ImageWidth">Source image width.</param>
/// <param name="ImageHeight">Source image height.</param>
/// <param name="TensorWidth">Tensor width.</param>
/// <param name="TensorHeight">Tensor height.</param>
public readonly record struct TensorMapping(NormalizedRect Roi, LetterboxPadding Padding, int ImageWidth, int ImageHeight, int TensorWidth, int TensorHeight)
{
    /// <summary>
    /// Maps a point in normalized tensor coordinates ([0,1] across the whole tensor) to normalized
    /// image coordinates, applying the ROI rotation in pixel space.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public (float X, float Y) TensorToImage(float u, float v)
    {
        float rw = Roi.Width * ImageWidth, rh = Roi.Height * ImageHeight;
        float lx = (u - 0.5f) * rw, ly = (v - 0.5f) * rh;
        float cos = MathF.Cos(Roi.Rotation), sin = MathF.Sin(Roi.Rotation);
        float x = Roi.XCenter * ImageWidth + lx * cos - ly * sin;
        float y = Roi.YCenter * ImageHeight + lx * sin + ly * cos;
        return (x / ImageWidth, y / ImageHeight);
    }

    /// <summary>Maps a point in normalized image coordinates to normalized tensor coordinates.</summary>
    public (float U, float V) ImageToTensor(float x, float y)
    {
        float rw = Roi.Width * ImageWidth, rh = Roi.Height * ImageHeight;
        float dx = x * ImageWidth - Roi.XCenter * ImageWidth, dy = y * ImageHeight - Roi.YCenter * ImageHeight;
        float cos = MathF.Cos(Roi.Rotation), sin = MathF.Sin(Roi.Rotation);
        float lx = dx * cos + dy * sin, ly = -dx * sin + dy * cos;
        return (lx / rw + 0.5f, ly / rh + 0.5f);
    }

    /// <summary>Scales a depth value expressed in tensor-normalized units to image-normalized units.</summary>
    public float ScaleZ(float z) => z * Roi.Width;
}

/// <summary>
/// Converts an <see cref="MPImage"/> region of interest into a float tensor in NHWC/RGB layout,
/// mirroring MediaPipe's <c>ImageToTensorCalculator</c>: rotated crop, resize (bilinear),
/// optional letterboxing and value-range normalization — in a single pass without intermediate images.
/// </summary>
public static class ImageToTensor
{
    // Below this many destination pixels the work is done on the calling thread.
    private const int ParallelThreshold = 160 * 160;

    /// <summary>
    /// Expands <paramref name="roi"/> so its pixel aspect ratio matches the tensor's, as MediaPipe
    /// does when <c>keep_aspect_ratio</c> is set, and reports the resulting padding.
    /// </summary>
    public static NormalizedRect PadRoi(in NormalizedRect roi, int imageWidth, int imageHeight, int tensorWidth, int tensorHeight, out LetterboxPadding padding)
    {
        float tensorAspect = (float)tensorHeight / tensorWidth;
        float roiW = roi.Width * imageWidth, roiH = roi.Height * imageHeight;
        float roiAspect = roiH / roiW;
        float hPad = 0, vPad = 0;
        if (tensorAspect > roiAspect)
        {
            float newH = roiW * tensorAspect;
            vPad = (1f - roiH / newH) / 2f;
            roiH = newH;
        }
        else
        {
            float newW = roiH / tensorAspect;
            hPad = (1f - roiW / newW) / 2f;
            roiW = newW;
        }
        padding = new LetterboxPadding(hPad, vPad, hPad, vPad);
        return roi with { Width = roiW / imageWidth, Height = roiH / imageHeight };
    }

    /// <summary>Converts the whole image.</summary>
    public static TensorMapping Convert(MPImage image, in ImageToTensorOptions options, Span<float> destination) =>
        Convert(image, NormalizedRect.FullImage, options, destination);

    /// <summary>
    /// Samples <paramref name="roi"/> from <paramref name="image"/> into <paramref name="destination"/>
    /// (length <c>Width * Height * 3</c>, NHWC, RGB) and returns the mapping back to the image.
    /// </summary>
    public static unsafe TensorMapping Convert(MPImage image, in NormalizedRect roi, in ImageToTensorOptions options, Span<float> destination)
    {
        ArgumentNullException.ThrowIfNull(image);
        int dw = options.Width, dh = options.Height;
        if (destination.Length < dw * dh * 3)
            throw new ArgumentException($"Destination needs {dw * dh * 3} floats but has {destination.Length}.", nameof(destination));

        var sampled = roi;
        var padding = LetterboxPadding.None;
        if (options.KeepAspectRatio)
            sampled = PadRoi(roi, image.Width, image.Height, dw, dh, out padding);

        int iw = image.Width, ih = image.Height;
        float cx = sampled.XCenter * iw, cy = sampled.YCenter * ih;
        float rw = sampled.Width * iw, rh = sampled.Height * ih;
        float cos = MathF.Cos(sampled.Rotation), sin = MathF.Sin(sampled.Rotation);

        // Source position of destination pixel (u, v): C + R * (((u+.5)/dw - .5) * rw, ((v+.5)/dh - .5) * rh),
        // shifted by -0.5 so that integer coordinates address pixel centers for bilinear sampling.
        var p = new SamplerParams
        {
            DuX = rw / dw * cos,
            DuY = rw / dw * sin,
            DvX = -rh / dh * sin,
            DvY = rh / dh * cos,
            Scale = (options.RangeMax - options.RangeMin) / 255f,
            Offset = options.RangeMin,
            Replicate = options.BorderMode == BorderMode.Replicate,
            ImageWidth = iw,
            ImageHeight = ih,
            DestWidth = dw,
            Samples = options.Antialias ? Math.Clamp((int)MathF.Round(MathF.Max(rw / dw, rh / dh)), 1, 4) : 1,
        };
        float ox = (0.5f / dw - 0.5f) * rw, oy = (0.5f / dh - 0.5f) * rh;
        p.X0 = cx + ox * cos - oy * sin - 0.5f;
        p.Y0 = cy + ox * sin + oy * cos - 0.5f;

        fixed (Rgba32* src = image.Pixels)
        fixed (float* dst = destination)
        {
            var srcPtr = (nint)src;
            var dstPtr = (nint)dst;
            if (dw * dh < ParallelThreshold)
            {
                for (int v = 0; v < dh; v++) SampleRow(in p, (Rgba32*)srcPtr, (float*)dstPtr, v);
            }
            else
            {
                Parallel.For(0, dh, v => SampleRow(in p, (Rgba32*)srcPtr, (float*)dstPtr, v));
            }
        }
        return new TensorMapping(sampled, padding, iw, ih, dw, dh);
    }

    private struct SamplerParams
    {
        public float X0, Y0, DuX, DuY, DvX, DvY, Scale, Offset;
        public bool Replicate;
        public int ImageWidth, ImageHeight, DestWidth, Samples;
    }

    private static unsafe void SampleRow(in SamplerParams p, Rgba32* src, float* dst, int v)
    {
        float sx = p.X0 + v * p.DvX, sy = p.Y0 + v * p.DvY;
        float* o = dst + (long)v * p.DestWidth * 3;
        float scale = p.Scale, offset = p.Offset;
        int n = p.Samples;
        if (n <= 1)
        {
            for (int u = 0; u < p.DestWidth; u++, sx += p.DuX, sy += p.DuY, o += 3)
            {
                Bilinear(in p, src, sx, sy, out float r, out float g, out float b);
                o[0] = r * scale + offset;
                o[1] = g * scale + offset;
                o[2] = b * scale + offset;
            }
            return;
        }
        // Supersampling: average n x n bilinear taps spread over the destination pixel's footprint,
        // which removes the aliasing of plain bilinear sampling on strong downscales.
        float inv = 1f / (n * n);
        for (int u = 0; u < p.DestWidth; u++, sx += p.DuX, sy += p.DuY, o += 3)
        {
            float r = 0, g = 0, b = 0;
            for (int j = 0; j < n; j++)
            {
                float fv = (j + 0.5f) / n - 0.5f;
                for (int i = 0; i < n; i++)
                {
                    float fu = (i + 0.5f) / n - 0.5f;
                    Bilinear(in p, src, sx + fu * p.DuX + fv * p.DvX, sy + fu * p.DuY + fv * p.DvY, out float tr, out float tg, out float tb);
                    r += tr; g += tg; b += tb;
                }
            }
            o[0] = r * inv * scale + offset;
            o[1] = g * inv * scale + offset;
            o[2] = b * inv * scale + offset;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Bilinear(in SamplerParams p, Rgba32* src, float sx, float sy, out float r, out float g, out float b)
    {
        int iw = p.ImageWidth, ih = p.ImageHeight;
        float fxf = MathF.Floor(sx), fyf = MathF.Floor(sy);
        int x0 = (int)fxf, y0 = (int)fyf;
        float fx = sx - fxf, fy = sy - fyf;
        if ((uint)x0 < (uint)(iw - 1) && (uint)y0 < (uint)(ih - 1))
        {
            Rgba32* p00 = src + (long)y0 * iw + x0;
            Rgba32* p10 = p00 + iw;
            float w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy;
            r = p00->R * w00 + p00[1].R * w01 + p10->R * w10 + p10[1].R * w11;
            g = p00->G * w00 + p00[1].G * w01 + p10->G * w10 + p10[1].G * w11;
            b = p00->B * w00 + p00[1].B * w01 + p10->B * w10 + p10[1].B * w11;
        }
        else
        {
            SampleBorder(src, iw, ih, x0, y0, fx, fy, p.Replicate, out r, out g, out b);
        }
    }
    private static unsafe void SampleBorder(Rgba32* src, int iw, int ih, int x0, int y0, float fx, float fy, bool replicate, out float r, out float g, out float b)
    {
        r = g = b = 0;
        for (int j = 0; j < 2; j++)
        {
            for (int i = 0; i < 2; i++)
            {
                int x = x0 + i, y = y0 + j;
                float w = (i == 0 ? 1 - fx : fx) * (j == 0 ? 1 - fy : fy);
                if (w == 0) continue;
                if ((uint)x >= (uint)iw || (uint)y >= (uint)ih)
                {
                    if (!replicate) continue;
                    x = Math.Clamp(x, 0, iw - 1);
                    y = Math.Clamp(y, 0, ih - 1);
                }
                Rgba32 px = src[(long)y * iw + x];
                r += px.R * w; g += px.G * w; b += px.B * w;
            }
        }
    }
}
