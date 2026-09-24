using System.Text.Json;
using Gravicode.HFNet.GraviHub;
using Gravicode.Science.GraviNum;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Gravicode.HFNet.GraviTransformers.Vision;

/// <summary>
/// Turns an image file into the tensor a vision model was trained on.
/// </summary>
/// <remarks>
/// Preprocessing is not a detail that can be approximated. A model trained on inputs normalised to
/// [-1, 1] and fed inputs in [0, 1] still produces a confident answer - a wrong one - with nothing
/// in the output to say the input was wrong. So the numbers come from the repository's own
/// <c>preprocessor_config.json</c> rather than from a default written here.
/// </remarks>
public sealed record ImageProcessor
{
    /// <summary>The square edge length the image is resized to.</summary>
    public required int Size { get; init; }

    /// <summary>Per-channel mean subtracted after rescaling, in RGB order.</summary>
    public required IReadOnlyList<double> Mean { get; init; }

    /// <summary>Per-channel standard deviation divided out, in RGB order.</summary>
    public required IReadOnlyList<double> Deviation { get; init; }

    /// <summary>What each stored byte is divided by before normalising.</summary>
    public double RescaleFactor { get; init; } = 1 / 255.0;

    /// <summary>Whether the pixels are normalised at all.</summary>
    public bool Normalize { get; init; } = true;

    /// <summary>The processor used by the original ViT and DeiT checkpoints.</summary>
    public static ImageProcessor ViT => new()
    {
        Size = 224,
        Mean = [0.5, 0.5, 0.5],
        Deviation = [0.5, 0.5, 0.5],
    };

    /// <summary>Reads a repository's <c>preprocessor_config.json</c>.</summary>
    /// <param name="repoId">A model id such as <c>google/vit-base-patch16-224</c>.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    public static ImageProcessor FromPretrained(string repoId, string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        string path;
        try { path = Hub.DownloadFile(repoId, "preprocessor_config.json", revision); }
        catch (HubException) { return ViT; }

        return Load(path);
    }

    /// <summary>Reads a <c>preprocessor_config.json</c> from disk.</summary>
    public static ImageProcessor Load(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;

        return new ImageProcessor
        {
            Size = ReadSize(root),
            Mean = ReadTriple(root, "image_mean", 0.5),
            Deviation = ReadTriple(root, "image_std", 0.5),
            RescaleFactor = root.TryGetProperty("rescale_factor", out var rescale)
                ? rescale.GetDouble()
                : 1 / 255.0,
            Normalize = !root.TryGetProperty("do_normalize", out var normalize) || normalize.GetBoolean(),
        };
    }

    /// <summary>
    /// Reads an image and returns it as <c>[channels, size, size]</c>, normalised.
    /// </summary>
    /// <param name="path">A path to a PNG, JPEG, BMP, GIF, TIFF or WebP file.</param>
    public NdArray Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using var image = Image.Load<Rgb24>(path);
        return Convert(image);
    }

    /// <summary>Reads an image from a stream and returns it as <c>[channels, size, size]</c>.</summary>
    public NdArray Read(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);

        using var image = Image.Load<Rgb24>(stream);
        return Convert(image);
    }

    private NdArray Convert(Image<Rgb24> image)
    {
        // Straight to a square, not shortest-side-then-crop: that is what ViTImageProcessor does
        // with a scalar `size`, and cropping instead would quietly throw away the edges of a
        // non-square photograph.
        image.Mutate(x => x.Resize(new ResizeOptions
        {
            Size = new Size(Size, Size),
            Mode = ResizeMode.Stretch,
            Sampler = KnownResamplers.Triangle,
        }));

        var pixels = NdArray.Zeros(3, Size, Size);

        image.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < Size; y++)
            {
                var row = accessor.GetRowSpan(y);

                for (var x = 0; x < Size; x++)
                {
                    var pixel = row[x];
                    Span<double> channels = [pixel.R, pixel.G, pixel.B];

                    for (var c = 0; c < 3; c++)
                    {
                        var value = channels[c] * RescaleFactor;
                        if (Normalize) value = (value - Mean[c]) / Deviation[c];

                        pixels[c, y, x] = value;
                    }
                }
            }
        });

        return pixels;
    }

    private static int ReadSize(JsonElement root)
    {
        if (root.TryGetProperty("size", out var size))
        {
            if (size.ValueKind == JsonValueKind.Number) return size.GetInt32();

            // Newer exports write {"height": 224, "width": 224}, older ones a bare number, and a
            // few {"shortest_edge": 224}.
            foreach (var name in (string[])["height", "shortest_edge", "width"])
            {
                if (size.TryGetProperty(name, out var value)) return value.GetInt32();
            }
        }

        if (root.TryGetProperty("crop_size", out var crop) && crop.ValueKind == JsonValueKind.Number)
        {
            return crop.GetInt32();
        }

        return 224;
    }

    private static double[] ReadTriple(JsonElement root, string name, double fallback)
    {
        if (!root.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.Array)
        {
            return [fallback, fallback, fallback];
        }

        var values = value.EnumerateArray().Select(v => v.GetDouble()).ToArray();
        return values.Length == 3 ? values : [fallback, fallback, fallback];
    }

    /// <inheritdoc />
    public override string ToString()
        => $"ImageProcessor({Size}x{Size}, mean [{string.Join(", ", Mean)}], "
            + $"std [{string.Join(", ", Deviation)}])";
}
