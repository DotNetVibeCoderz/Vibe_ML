using Gravicode.HFNet.GraviTransformers.Vision;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Xunit;

namespace Gravicode.HFNet.GraviTransformers.Tests;

/// <summary>
/// The image processor reads files through a restricted decoder set, which keeps ImageSharp 3.1.x's
/// TIFF and ICC advisories off the path an untrusted image takes. These tests pin that the
/// restriction holds and that it costs nothing for the formats models are actually fed.
/// </summary>
public sealed class ImageDecodingTests
{
    private static Image<Rgb24> Gradient()
    {
        var image = new Image<Rgb24>(10, 7);
        for (var y = 0; y < image.Height; y++)
            for (var x = 0; x < image.Width; x++)
                image[x, y] = new Rgb24((byte)(x * 25), (byte)(y * 36), (byte)(x * y * 3));
        return image;
    }

    [Fact]
    public void PngReadsToTheSameTensorAsTheDecodedImage()
    {
        using var image = Gradient();
        using var png = new MemoryStream();
        image.SaveAsPng(png);
        png.Position = 0;

        var processor = ImageProcessor.ViT;
        Assert.Equal(processor.Convert(image).ToArray(), processor.Read(png).ToArray());
    }

    [Fact]
    public void TiffIsRefused()
    {
        using var image = Gradient();
        using var tiff = new MemoryStream();
        image.SaveAsTiff(tiff);
        tiff.Position = 0;

        Assert.Throws<UnknownImageFormatException>(() => ImageProcessor.ViT.Read(tiff));
    }
}
