using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Processing;
using SplatStudio.Application.Abstractions;

namespace SplatStudio.Infrastructure.Imaging;

/// <summary>
/// ImageSharp-backed implementation of <see cref="IImageProcessingService"/>.
/// Reads the upload exactly once: decodes it, records its true pixel
/// dimensions, then produces an aspect-preserving JPEG thumbnail for the
/// gallery grid (full-resolution originals are never displayed inline).
/// </summary>
public class ImageProcessingService : IImageProcessingService
{
    public async Task<ProcessedImageResult> ProcessUploadAsync(Stream imageStream, int thumbnailMaxDimension, CancellationToken ct = default)
    {
        using var image = await Image.LoadAsync(imageStream, ct);
        var width = image.Width;
        var height = image.Height;

        using var thumb = image.Clone(ctx => ctx.Resize(new ResizeOptions
        {
            Mode = ResizeMode.Max,
            Size = new Size(thumbnailMaxDimension, thumbnailMaxDimension)
        }));

        using var ms = new MemoryStream();
        await thumb.SaveAsync(ms, new JpegEncoder { Quality = 82 }, ct);

        return new ProcessedImageResult(width, height, ms.ToArray(), "image/jpeg");
    }
}
