using System.Text.Json;
using Gravicode.HFNet.GraviHub.Io;
using Gravicode.HFNet.GraviTransformers.Vision;
using Gravicode.Science.GraviNum;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Xunit;

namespace Gravicode.HFNet.GraviTransformers.Tests;

/// <summary>
/// Builds a tiny ViT on disk so the loader can be exercised without the network.
/// </summary>
/// <remarks>
/// Small enough to reason about by hand - eight-wide hidden states, four patches - and shaped
/// exactly like a real checkpoint, prefix and all, so the naming is under test rather than assumed.
/// </remarks>
internal sealed class TinyVit : IDisposable
{
    internal const int Hidden = 8;
    internal const int Heads = 2;
    internal const int Intermediate = 16;
    internal const int Layers = 2;
    internal const int ImageSize = 8;
    internal const int PatchSize = 4;
    internal const int Patches = 4;
    internal const int PatchValues = 3 * PatchSize * PatchSize;

    private TinyVit(string directory) => Directory = directory;

    internal string Directory { get; }

    /// <summary>Writes a checkpoint whose every weight is zero except the ones named.</summary>
    /// <param name="classes">How many classes the head predicts, or 0 for no head.</param>
    /// <param name="mutate">Chance to set specific tensors before they are written.</param>
    /// <param name="positions">How many position embeddings to write, for the mismatch case.</param>
    internal static TinyVit Write(
        int classes = 3,
        Action<Dictionary<string, NdArray>>? mutate = null,
        int positions = 1 + Patches)
    {
        var directory = Path.Combine(Path.GetTempPath(), "hfnet-vit-" + Guid.NewGuid().ToString("N")[..10]);
        System.IO.Directory.CreateDirectory(directory);

        var tensors = new Dictionary<string, NdArray>(StringComparer.Ordinal)
        {
            ["vit.embeddings.cls_token"] = NdArray.Zeros(1, 1, Hidden),
            ["vit.embeddings.position_embeddings"] = NdArray.Zeros(1, positions, Hidden),
            ["vit.embeddings.patch_embeddings.projection.weight"] = NdArray.Zeros(Hidden, 3, PatchSize, PatchSize),
            ["vit.embeddings.patch_embeddings.projection.bias"] = NdArray.Zeros(Hidden),
            ["vit.layernorm.weight"] = Ones(Hidden),
            ["vit.layernorm.bias"] = NdArray.Zeros(Hidden),
        };

        for (var layer = 0; layer < Layers; layer++)
        {
            var block = $"vit.encoder.layer.{layer}";

            foreach (var name in (string[])["query", "key", "value"])
            {
                tensors[$"{block}.attention.attention.{name}.weight"] = NdArray.Zeros(Hidden, Hidden);
                tensors[$"{block}.attention.attention.{name}.bias"] = NdArray.Zeros(Hidden);
            }

            tensors[$"{block}.attention.output.dense.weight"] = NdArray.Zeros(Hidden, Hidden);
            tensors[$"{block}.attention.output.dense.bias"] = NdArray.Zeros(Hidden);

            tensors[$"{block}.intermediate.dense.weight"] = NdArray.Zeros(Intermediate, Hidden);
            tensors[$"{block}.intermediate.dense.bias"] = NdArray.Zeros(Intermediate);
            tensors[$"{block}.output.dense.weight"] = NdArray.Zeros(Hidden, Intermediate);
            tensors[$"{block}.output.dense.bias"] = NdArray.Zeros(Hidden);

            tensors[$"{block}.layernorm_before.weight"] = Ones(Hidden);
            tensors[$"{block}.layernorm_before.bias"] = NdArray.Zeros(Hidden);
            tensors[$"{block}.layernorm_after.weight"] = Ones(Hidden);
            tensors[$"{block}.layernorm_after.bias"] = NdArray.Zeros(Hidden);
        }

        if (classes > 0)
        {
            tensors["classifier.weight"] = NdArray.Zeros(classes, Hidden);
            tensors["classifier.bias"] = NdArray.Zeros(classes);
        }

        mutate?.Invoke(tensors);
        SafeTensors.Write(Path.Combine(directory, "model.safetensors"), tensors);

        var labels = Enumerable.Range(0, classes)
            .ToDictionary(i => i.ToString(), i => $"class {i}");

        File.WriteAllText(Path.Combine(directory, "config.json"), JsonSerializer.Serialize(new
        {
            model_type = "vit",
            architectures = new[] { "ViTForImageClassification" },
            hidden_size = Hidden,
            num_hidden_layers = Layers,
            num_attention_heads = Heads,
            intermediate_size = Intermediate,
            image_size = ImageSize,
            patch_size = PatchSize,
            num_channels = 3,
            layer_norm_eps = 1e-12,
            id2label = labels,
        }));

        return new TinyVit(directory);
    }

    internal static NdArray Ones(int size)
    {
        var result = NdArray.Zeros(size);
        for (var i = 0; i < size; i++) result.SetAt(i, 1.0);

        return result;
    }

    /// <summary>A picture where every value is distinct, so a scrambled layout cannot hide.</summary>
    internal static NdArray Ramp()
    {
        var pixels = NdArray.Zeros(3, ImageSize, ImageSize);
        for (var i = 0; i < pixels.Size; i++) pixels.SetAt(i, i + 1);

        return pixels;
    }

    public void Dispose()
    {
        try { System.IO.Directory.Delete(Directory, recursive: true); }
        catch (IOException) { /* a mapped file can outlive the test on Windows */ }
    }
}

/// <summary>Tests for the vision encoder.</summary>
public sealed class VisionTests
{
    // ------------------------------------------------------------------ config

    [Fact]
    public void Config_reads_the_dimensions_a_vision_model_needs()
    {
        using var model = TinyVit.Write();
        var config = VisionConfig.Load(Path.Combine(model.Directory, "config.json"));

        Assert.Equal("vit", config.ModelType);
        Assert.Equal(TinyVit.Hidden, config.HiddenSize);
        Assert.Equal(TinyVit.Layers, config.Layers);
        Assert.Equal(2, config.Grid);
        Assert.Equal(TinyVit.Patches, config.Patches);
        Assert.Equal(3, config.LabelCount);
        Assert.Equal("class 1", config.Label(1));
    }

    [Fact]
    public void Config_refuses_an_image_that_does_not_divide_into_patches()
    {
        var path = Path.GetTempFileName();
        File.WriteAllText(path, """
            {"model_type":"vit","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,
             "intermediate_size":16,"image_size":10,"patch_size":4}
            """);

        var error = Assert.Throws<InvalidDataException>(() => VisionConfig.Load(path));
        Assert.Contains("does not divide", error.Message, StringComparison.Ordinal);

        File.Delete(path);
    }

    [Fact]
    public void Config_throws_rather_than_defaulting_a_missing_dimension()
    {
        var path = Path.GetTempFileName();
        File.WriteAllText(path, """{"model_type":"vit","hidden_size":8,"image_size":8,"patch_size":4}""");

        // A guessed layer count builds a model of the wrong depth, which loads and returns noise.
        Assert.Throws<InvalidDataException>(() => VisionConfig.Load(path));

        File.Delete(path);
    }

    [Fact]
    public void A_windowed_backbone_is_refused_by_name()
    {
        var config = new VisionConfig
        {
            ModelType = "swin",
            HiddenSize = 96,
            Layers = 4,
            Heads = 3,
            IntermediateSize = 384,
            ImageSize = 224,
            PatchSize = 4,
        };

        var error = Assert.Throws<NotSupportedException>(() => VisionConfig.Require(config, "microsoft/swin-tiny"));
        Assert.Contains("swin", error.Message, StringComparison.Ordinal);
        Assert.Contains("ONNX", error.Message, StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------ image processor

    [Fact]
    public void The_processor_reads_its_numbers_from_the_repository()
    {
        var path = Path.GetTempFileName();
        File.WriteAllText(path, """
            {"size":{"height":256,"width":256},"image_mean":[0.485,0.456,0.406],
             "image_std":[0.229,0.224,0.225],"do_normalize":true}
            """);

        var processor = ImageProcessor.Load(path);

        Assert.Equal(256, processor.Size);
        Assert.Equal(0.485, processor.Mean[0], 6);
        Assert.Equal(0.225, processor.Deviation[2], 6);

        File.Delete(path);
    }

    [Fact]
    public void A_scalar_size_and_a_shortest_edge_both_read_as_the_edge_length()
    {
        var scalar = Path.GetTempFileName();
        File.WriteAllText(scalar, """{"size":224}""");
        Assert.Equal(224, ImageProcessor.Load(scalar).Size);
        File.Delete(scalar);

        var shortest = Path.GetTempFileName();
        File.WriteAllText(shortest, """{"size":{"shortest_edge":384}}""");
        Assert.Equal(384, ImageProcessor.Load(shortest).Size);
        File.Delete(shortest);
    }

    [Fact]
    public void Normalisation_maps_the_extremes_to_minus_one_and_one()
    {
        var processor = ImageProcessor.ViT;

        var path = Path.Combine(Path.GetTempPath(), $"hfnet-black-{Guid.NewGuid():N}.png");
        using (var image = new Image<Rgb24>(4, 4))
        {
            image.Save(path);
        }

        // Every pixel is black, which under mean 0.5 / std 0.5 is exactly -1 everywhere.
        var pixels = processor.Read(path);

        Assert.Equal(3, pixels.Shape[0]);
        Assert.Equal(processor.Size, pixels.Shape[1]);
        for (var i = 0; i < pixels.Size; i++) Assert.Equal(-1.0, pixels.At(i), 9);

        File.Delete(path);
    }

    // ------------------------------------------------------------------ loading

    [Fact]
    public void A_checkpoint_loads_with_its_prefix_and_reports_what_it_is()
    {
        using var directory = TinyVit.Write();
        using var model = VisionTransformer.Open(directory.Directory);

        Assert.Equal(TinyVit.Layers, model.Config.Layers);
        Assert.Equal(TinyVit.Patches, model.Config.Patches);
        Assert.True(model.HasClassificationHead);
        Assert.Equal(["class 0", "class 1", "class 2"], model.Labels);
    }

    [Fact]
    public void A_position_count_that_does_not_match_the_patch_grid_is_refused()
    {
        // Silently truncating or tiling these shifts every patch's position and produces a model
        // that runs and is wrong.
        using var directory = TinyVit.Write(positions: 1 + TinyVit.Patches + 1);

        var error = Assert.Throws<InvalidDataException>(() => VisionTransformer.Open(directory.Directory));
        Assert.Contains("position embeddings", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void A_feature_extractor_says_so_instead_of_inventing_classes()
    {
        using var directory = TinyVit.Write(classes: 0);
        using var model = VisionTransformer.Open(directory.Directory);

        Assert.False(model.HasClassificationHead);

        var error = Assert.Throws<NotSupportedException>(() => model.Classify("nothing.png"));
        Assert.Contains("Embed", error.Message, StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------ embeddings

    [Fact]
    public void The_patch_is_flattened_channel_major()
    {
        // The projection picks out one input element: output 0 reads the first value of the
        // flattened patch, output 1 the second, and so on. If the flattening ran (y, x, channel)
        // instead of (channel, y, x) these read different pixels, and nothing downstream would
        // ever say so.
        using var directory = TinyVit.Write(mutate: tensors =>
        {
            var projection = NdArray.Zeros(TinyVit.Hidden, 3, TinyVit.PatchSize, TinyVit.PatchSize);
            for (var o = 0; o < TinyVit.Hidden; o++) projection.SetAt(o * TinyVit.PatchValues + o, 1.0);

            tensors["vit.embeddings.patch_embeddings.projection.weight"] = projection;
        });

        using var model = VisionTransformer.Open(directory.Directory);

        var pixels = TinyVit.Ramp();
        var sequence = model.Embeddings(pixels);

        Assert.Equal(1 + TinyVit.Patches, sequence.Shape[0]);

        // The top-left patch, at row 1. Its channel-major flattening is
        // [c0y0x0, c0y0x1, c0y0x2, c0y0x3, c0y1x0, ...] - the first four of row 0, then row 1.
        var expected = new[]
        {
            pixels[0, 0, 0], pixels[0, 0, 1], pixels[0, 0, 2], pixels[0, 0, 3],
            pixels[0, 1, 0], pixels[0, 1, 1], pixels[0, 1, 2], pixels[0, 1, 3],
        };

        for (var d = 0; d < TinyVit.Hidden; d++) Assert.Equal(expected[d], sequence[1, d], 9);
    }

    [Fact]
    public void Position_embeddings_land_on_the_rows_they_belong_to()
    {
        using var directory = TinyVit.Write(mutate: tensors =>
        {
            var positions = NdArray.Zeros(1, 1 + TinyVit.Patches, TinyVit.Hidden);
            for (var row = 0; row < 1 + TinyVit.Patches; row++)
            {
                for (var d = 0; d < TinyVit.Hidden; d++) positions[0, row, d] = row * 100 + d;
            }

            tensors["vit.embeddings.position_embeddings"] = positions;
        });

        using var model = VisionTransformer.Open(directory.Directory);

        // The projection is zero, so whatever is left is the position embedding alone.
        var sequence = model.Embeddings(TinyVit.Ramp());

        for (var row = 0; row < 1 + TinyVit.Patches; row++)
        {
            for (var d = 0; d < TinyVit.Hidden; d++)
            {
                Assert.Equal(row * 100 + d, sequence[row, d], 9);
            }
        }
    }

    [Fact]
    public void The_class_token_is_the_first_row_and_carries_no_patch()
    {
        using var directory = TinyVit.Write(mutate: tensors =>
        {
            var token = NdArray.Zeros(1, 1, TinyVit.Hidden);
            for (var d = 0; d < TinyVit.Hidden; d++) token[0, 0, d] = 7.0 + d;

            tensors["vit.embeddings.cls_token"] = token;
        });

        using var model = VisionTransformer.Open(directory.Directory);
        var sequence = model.Embeddings(TinyVit.Ramp());

        for (var d = 0; d < TinyVit.Hidden; d++) Assert.Equal(7.0 + d, sequence[0, d], 9);
    }

    // ------------------------------------------------------------------ the block

    [Fact]
    public void A_block_with_zero_weights_is_the_identity_which_is_what_pre_norm_means()
    {
        // The whole reason this is not a mode of TransformerModel. With every projection zeroed,
        // a PRE-norm block returns its input untouched - the residual is added to nothing. A
        // POST-norm block returns LayerNorm(input), which for a ramp is emphatically not the input.
        // Both load the same parameters without complaint, so this is the only thing that
        // distinguishes them.
        using var directory = TinyVit.Write(mutate: tensors =>
        {
            var positions = NdArray.Zeros(1, 1 + TinyVit.Patches, TinyVit.Hidden);
            for (var row = 0; row < 1 + TinyVit.Patches; row++)
            {
                for (var d = 0; d < TinyVit.Hidden; d++) positions[0, row, d] = row + d * 0.5;
            }

            tensors["vit.embeddings.position_embeddings"] = positions;

            // The final norm would hide the result, so make it the identity too.
            tensors["vit.layernorm.weight"] = TinyVit.Ones(TinyVit.Hidden);
        });

        using var model = VisionTransformer.Open(directory.Directory);

        var input = model.Embeddings(TinyVit.Ramp());
        var output = model.Forward(TinyVit.Ramp());

        // Forward ends with vit.layernorm, so compare against the normalised input rather than the
        // input: what is being asserted is that the twelve blocks in between changed nothing.
        var expected = Normalize(input);

        for (var i = 0; i < expected.Size; i++) Assert.Equal(expected.At(i), output.At(i), 9);
    }

    [Fact]
    public void Classification_returns_a_distribution_over_the_checkpoints_own_labels()
    {
        using var directory = TinyVit.Write(mutate: tensors =>
        {
            var head = NdArray.Zeros(3, TinyVit.Hidden);
            for (var d = 0; d < TinyVit.Hidden; d++) head[1, d] = 1.0;

            tensors["classifier.weight"] = head;

            var bias = NdArray.Zeros(3);
            bias[1] = 5.0;
            tensors["classifier.bias"] = bias;
        });

        using var model = VisionTransformer.Open(directory.Directory);

        var image = Path.Combine(Path.GetTempPath(), $"hfnet-grey-{Guid.NewGuid():N}.png");
        using (var picture = new Image<Rgb24>(TinyVit.ImageSize, TinyVit.ImageSize))
        {
            picture.Save(image);
        }

        var predictions = model.Classify(image, topK: 0);

        Assert.Equal(3, predictions.Count);
        Assert.Equal(1.0, predictions.Sum(p => p.Score), 9);
        Assert.Equal("class 1", predictions[0].Label);

        File.Delete(image);
    }

    // ------------------------------------------------------------------ the inner loop

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(768)]
    public void The_vectorised_dot_product_agrees_with_the_naive_one(int length)
    {
        // Lengths that are not a multiple of the vector width are where a vectorised loop drops
        // the tail, and the error is small enough to look like rounding.
        var random = new GraviRandom(11);
        var a = new double[length];
        var b = new double[length];

        for (var i = 0; i < length; i++)
        {
            a[i] = random.NextDouble() * 2 - 1;
            b[i] = random.NextDouble() * 2 - 1;
        }

        var naive = 0.0;
        for (var i = 0; i < length; i++) naive += a[i] * b[i];

        Assert.Equal(naive, Simd.Dot(a, b), 9);
    }

    /// <summary>Layer-normalises every row, as the model's final norm does with unit scale.</summary>
    private static NdArray Normalize(NdArray input)
    {
        var rows = input.Shape[0];
        var width = input.Shape[1];
        var result = NdArray.Zeros(rows, width);

        for (var row = 0; row < rows; row++)
        {
            var mean = 0.0;
            for (var d = 0; d < width; d++) mean += input[row, d] / width;

            var variance = 0.0;
            for (var d = 0; d < width; d++) variance += Math.Pow(input[row, d] - mean, 2) / width;

            var scale = Math.Sqrt(variance + 1e-12);
            for (var d = 0; d < width; d++) result[row, d] = (input[row, d] - mean) / scale;
        }

        return result;
    }
}
