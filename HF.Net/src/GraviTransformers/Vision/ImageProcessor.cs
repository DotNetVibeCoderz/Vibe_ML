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
    /// <summary>
    /// The square edge length the image is resized to - or, with <see cref="ShortestEdge"/>, the
    /// length its shorter side is resized to.
    /// </summary>
    public required int Size { get; init; }

    /// <summary>
    /// Whether <see cref="Size"/> is the shorter side, the aspect ratio kept (CLIP), rather than both
    /// sides (ViT).
    /// </summary>
    public bool ShortestEdge { get; init; }

    /// <summary>The square cut from the centre after resizing, or <c>null</c> for none.</summary>
    public int? CropSize { get; init; }

    /// <summary>The filter the resize uses, as PIL numbers it.</summary>
    /// <remarks>
    /// ViT is bilinear, CLIP bicubic. The resize is PIL's own algorithm reproduced to the byte (see
    /// <see cref="PilResize"/>), because that is what the model saw in training.
    /// </remarks>
    public Resample Resample { get; init; } = Resample.Bilinear;

    /// <summary>The edge length of the tensor this produces.</summary>
    public int OutputSize => CropSize ?? Size;

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

    /// <summary>The processor CLIP checkpoints publish: shortest edge 224, bicubic, centre crop, CLIP's own statistics.</summary>
    public static ImageProcessor Clip => new()
    {
        Size = 224,
        ShortestEdge = true,
        CropSize = 224,
        Resample = Resample.Bicubic,
        Mean = [0.48145466, 0.4578275, 0.40821073],
        Deviation = [0.26862954, 0.26130258, 0.27577711],
    };

    /// <summary>Reads a <c>preprocessor_config.json</c> from disk.</summary>
    public static ImageProcessor Load(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;

        // A bare number means a square to ViT's processor but the shorter side to CLIP's, which
        // reads sizes with default_to_square=False - and OpenAI's own configs write a bare 224.
        var type = root.TryGetProperty("image_processor_type", out var kind) ? kind.GetString()
            : root.TryGetProperty("feature_extractor_type", out var legacy) ? legacy.GetString()
            : null;
        var squareByDefault = type is null || !type.StartsWith("CLIP", StringComparison.Ordinal);

        var shortest = root.TryGetProperty("size", out var sizeElement)
            && (sizeElement.ValueKind == JsonValueKind.Object
                ? sizeElement.TryGetProperty("shortest_edge", out _)
                : !squareByDefault);
        var crop = (!root.TryGetProperty("do_center_crop", out var doCrop) || doCrop.GetBoolean())
            && root.TryGetProperty("crop_size", out var cropElement)
            ? cropElement.ValueKind == JsonValueKind.Number
                ? cropElement.GetInt32()
                : cropElement.TryGetProperty("height", out var cropHeight) ? cropHeight.GetInt32() : (int?)null
            : null;

        return new ImageProcessor
        {
            Size = ReadSize(root),
            ShortestEdge = shortest,
            CropSize = crop,
            Resample = root.TryGetProperty("resample", out var resample) && resample.ValueKind == JsonValueKind.Number
                ? (Resample)resample.GetInt32()
                : Resample.Bilinear,
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

    /// <summary>Converts a decoded image to <c>[channels, size, size]</c>, normalised.</summary>
    /// <remarks>
    /// Every step is the Hugging Face processor's: PIL's resize on the 8-bit image, an integer
    /// centre crop, then rescaling and normalising in float32, which is the precision the reference
    /// hands the model.
    /// </remarks>
    public NdArray Convert(Image<Rgb24> image)
    {
        ArgumentNullException.ThrowIfNull(image);

        var width = image.Width;
        var height = image.Height;
        var bytes = new byte[width * height * 3];
        image.CopyPixelDataTo(bytes);

        // The target size: both sides, or the shorter one with the longer scaled to match -
        // transformers' get_resize_output_image_size, which truncates the longer side.
        int newWidth, newHeight;
        if (ShortestEdge)
        {
            var (shorter, longer) = width <= height ? (width, height) : (height, width);
            var scaled = (int)((long)Size * longer / shorter);
            (newWidth, newHeight) = width <= height ? (Size, scaled) : (scaled, Size);
        }
        else
        {
            (newWidth, newHeight) = (Size, Size);
        }

        if (newWidth != width || newHeight != height)
        {
            bytes = PilResize.Resize(bytes, width, height, newWidth, newHeight, Resample);
            (width, height) = (newWidth, newHeight);
        }

        var edge = OutputSize;
        var top = CropSize is null ? 0 : (height - edge) / 2;
        var left = CropSize is null ? 0 : (width - edge) / 2;

        if (top < 0 || left < 0 || (CropSize is null && (width != edge || height != edge)))
        {
            throw new InvalidOperationException(
                $"A {width}x{height} image cannot be cut to {edge}x{edge}; the crop is larger than the resized image.");
        }

        var mean = Mean.Select(m => (float)m).ToArray();
        var deviation = Deviation.Select(d => (float)d).ToArray();
        var values = new double[3 * edge * edge];
        var plane = edge * edge;

        for (var y = 0; y < edge; y++)
        {
            for (var x = 0; x < edge; x++)
            {
                var at = ((top + y) * width + left + x) * 3;
                for (var c = 0; c < 3; c++)
                {
                    var value = (float)(bytes[at + c] * RescaleFactor);
                    if (Normalize) value = (value - mean[c]) / deviation[c];

                    values[c * plane + y * edge + x] = value;
                }
            }
        }

        return new NdArray(values, 3, edge, edge);
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
        => $"ImageProcessor({(ShortestEdge ? $"shortest edge {Size}" : $"{Size}x{Size}")}"
            + $"{(CropSize is { } c ? $", crop {c}" : "")}, {Resample}, mean [{string.Join(", ", Mean)}], "
            + $"std [{string.Join(", ", Deviation)}])";
}
