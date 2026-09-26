namespace MediaPipeNet.Imaging;

/// <summary>
/// Projects single-channel model outputs (segmentation masks, heatmaps) from tensor space back onto
/// the source image, undoing the ROI rotation, crop and letterboxing.
/// </summary>
public static class TensorWarp
{
    /// <summary>
    /// Resamples a <paramref name="tensorWidth"/> x <paramref name="tensorHeight"/> single-channel
    /// tensor into <paramref name="destination"/> (<paramref name="outputWidth"/> x <paramref name="outputHeight"/>,
    /// covering the whole source image). Pixels outside the ROI receive <paramref name="outsideValue"/>.
    /// </summary>
    public static void ProjectToImage(
        ReadOnlySpan<float> tensor, int tensorWidth, int tensorHeight,
        in TensorMapping mapping,
        Span<float> destination, int outputWidth, int outputHeight,
        float outsideValue = 0f)
    {
        if (tensor.Length < tensorWidth * tensorHeight) throw new ArgumentException("Tensor is smaller than its declared size.", nameof(tensor));
        if (destination.Length < outputWidth * outputHeight) throw new ArgumentException("Destination is too small.", nameof(destination));

        var roi = mapping.Roi;
        // Work in source-image pixel units so rotation is applied isotropically.
        float iw = mapping.ImageWidth, ih = mapping.ImageHeight;
        float sx = iw / outputWidth, sy = ih / outputHeight;
        float cx = roi.XCenter * iw, cy = roi.YCenter * ih;
        float rw = roi.Width * iw, rh = roi.Height * ih;
        float cos = MathF.Cos(roi.Rotation), sin = MathF.Sin(roi.Rotation);

        var src = tensor.ToArray(); // lambda capture; tensors are small (<= 512x512)
        var dst = new float[outputWidth * outputHeight];
        Parallel.For(0, outputHeight, y =>
        {
            float py = (y + 0.5f) * sy - cy;
            int row = y * outputWidth;
            for (int x = 0; x < outputWidth; x++)
            {
                float px = (x + 0.5f) * sx - cx;
                float lx = px * cos + py * sin, ly = -px * sin + py * cos;
                float u = (lx / rw + 0.5f) * tensorWidth - 0.5f;
                float v = (ly / rh + 0.5f) * tensorHeight - 0.5f;
                dst[row + x] = Bilinear(src, tensorWidth, tensorHeight, u, v, outsideValue);
            }
        });
        dst.CopyTo(destination);
    }

    /// <summary>Bilinearly resizes a single-channel float image.</summary>
    public static void Resize(ReadOnlySpan<float> source, int sourceWidth, int sourceHeight, Span<float> destination, int width, int height)
    {
        float sx = (float)sourceWidth / width, sy = (float)sourceHeight / height;
        for (int y = 0; y < height; y++)
        {
            float v = (y + 0.5f) * sy - 0.5f;
            for (int x = 0; x < width; x++)
            {
                float u = (x + 0.5f) * sx - 0.5f;
                destination[y * width + x] = BilinearClamp(source, sourceWidth, sourceHeight, u, v);
            }
        }
    }

    private static float Bilinear(float[] t, int w, int h, float u, float v, float outside)
    {
        if (u < -0.5f || v < -0.5f || u > w - 0.5f || v > h - 0.5f) return outside;
        return BilinearClamp(t, w, h, u, v);
    }

    private static float BilinearClamp(ReadOnlySpan<float> t, int w, int h, float u, float v)
    {
        u = Math.Clamp(u, 0, w - 1);
        v = Math.Clamp(v, 0, h - 1);
        int x0 = (int)u, y0 = (int)v;
        int x1 = Math.Min(x0 + 1, w - 1), y1 = Math.Min(y0 + 1, h - 1);
        float fx = u - x0, fy = v - y0;
        float a = t[y0 * w + x0], b = t[y0 * w + x1], c = t[y1 * w + x0], d = t[y1 * w + x1];
        return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
    }
}
