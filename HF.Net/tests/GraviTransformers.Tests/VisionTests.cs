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
            hidden_act = "gelu_new",
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
        Assert.Equal("gelu_new", config.Activation);
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

    // ------------------------------------------------------------------ other resolutions

    /// <summary>
    /// A square grid of <paramref name="side"/> x <paramref name="side"/> vectors, row-major from
    /// row <paramref name="first"/>, holding <c>sin(1.3r + 0.7c + d) + 0.1rc</c>.
    /// </summary>
    /// <remarks>The same formula the torch script that produced the expected values used.</remarks>
    private static NdArray Grid(int side, int hidden, int first = 0)
    {
        var table = NdArray.Zeros(first + side * side, hidden);
        for (var r = 0; r < side; r++)
        {
            for (var c = 0; c < side; c++)
            {
                for (var d = 0; d < hidden; d++)
                {
                    table[first + r * side + c, d] = Math.Sin(1.3 * r + 0.7 * c + d) + 0.1 * r * c;
                }
            }
        }

        return table;
    }

    // torch.nn.functional.interpolate(mode="bicubic", align_corners=False) in float64, on Grid(4, 2).
    private static readonly double[][] TorchFourToSevenByFive =
    [
        [-0.138344337914452, 0.837901636450479], [0.366161221473413, 1.038176511050195],
        [0.887623135172901, 1.003169105045596], [1.045154466917928, 0.557637819739853],
        [0.939793504456064, 0.069966193495694], [0.327194093982027, 0.887205252973058],
        [0.685771103702955, 0.868467960851306], [0.949583927725750, 0.617547890073066],
        [0.847372734050131, 0.134495566486271], [0.592575090318251, -0.273243630529556],
        [0.922808754348798, 0.835096357951033], [1.041532645704685, 0.526634142754484],
        [0.927464402646189, 0.024347588117142], [0.493633002720283, -0.433525074854440],
        [0.076505130129552, -0.667559509108421], [0.961725307568081, 0.225868116984342],
        [0.775931166049338, -0.160477780366145], [0.371383323443420, -0.560028756563350],
        [-0.092192815201812, -0.643021065067064], [-0.376905221305188, -0.498885207927974],
        [0.454540788875486, -0.515537884070937], [0.125397049710861, -0.705787673061764],
        [-0.267039099719494, -0.702822540182702], [-0.424564975678817, -0.289219780964883],
        [-0.360051466641985, 0.208667545564811], [-0.293342645766566, -0.892825721261757],
        [-0.465838751251713, -0.698765267387754], [-0.491731819470918, -0.209657405246590],
        [-0.154767190851871, 0.528563147564928], [0.280805309243480, 1.140080934625529],
        [-0.809606922828562, -1.081156655546249], [-0.838276412783467, -0.613536362291646],
        [-0.577807642697494, 0.198616460705764], [0.103120365856776, 1.118139635065698],
        [0.779046823753084, 1.765688700405297],
    ];

    private static readonly double[][] TorchFourToTwoByThree =
    [
        [0.580603382642157, 0.885332369331180], [0.982751786288936, 0.484542300326449],
        [0.553034475344842, -0.298518162150177], [-0.170019112194266, -0.804463424708930],
        [-0.471313427385433, -0.364025723442963], [0.003814630451076, 0.756015540560574],
    ];

    // The same, on Grid(2, 8) - TinyVit's own position grid - resized to 3x3.
    private static readonly double[][] TorchTwoToThree =
    [
        [-0.144073535298013, 0.831812433552864, 1.043626670319237, 0.296628142549504, -0.722396148285054, -1.076559968662437, -0.440246735437817, 0.601520099270912],
        [0.264442267012718, 0.953300127448922, 0.761707815704698, -0.134185580375530, -0.910699804059677, -0.853911259171356, -0.016031071987096, 0.832597577478352],
        [0.672958069323449, 1.074787821344980, 0.479788961090158, -0.564999303300564, -1.099003459834299, -0.631262549680274, 0.408184591463626, 1.063675055685791],
        [0.451833045458328, 0.808969668738501, 0.418352877962594, -0.360886050851649, -0.812318440196852, -0.520899433196162, 0.245441679061593, 0.782134412129742],
        [0.654268324870141, 0.704990253874238, 0.130532279402084, -0.540951586065305, -0.692102173323714, -0.183952329514409, 0.516307322417405, 0.764861287898460],
        [0.856703604281955, 0.601010839009975, -0.157288319158427, -0.721017121278960, -0.571885906450576, 0.152994774167344, 0.787172965773217, 0.747588163667177],
        [1.047739626214670, 0.786126903924137, -0.206920914394051, -1.018400244252804, -0.902240732108649, 0.034761102270114, 0.931130093561003, 0.962748724988572],
        [1.044094382727565, 0.456680380299553, -0.500643256900532, -0.947717591755080, -0.473504542587750, 0.486006600142538, 1.048645716821906, 0.697124998318568],
        [1.040449139240461, 0.127233856674969, -0.794365599407012, -0.877034939257357, -0.044768353066851, 0.937252098014963, 1.166161340082807, 0.431501271648562],
    ];

    [Theory]
    [InlineData(7, 5)]
    [InlineData(2, 3)]
    public void Bicubic_resizing_agrees_with_torch(int rows, int columns)
    {
        // Pinned to torch rather than to a formula: the kernel constant (-0.75, not -0.5), the
        // half-pixel source coordinate and the edge clamping each move these values by a few
        // hundredths, which is exactly the size of error that would otherwise pass for "close".
        // A non-square target catches rows and columns swapped; 2x3 is a downscale.
        var expected = rows == 7 ? TorchFourToSevenByFive : TorchFourToTwoByThree;
        var actual = VisionTransformer.Bicubic(Grid(4, 2), 0, 4, 2, rows, columns);

        Assert.Equal(expected.Length * 2, actual.Length);

        for (var i = 0; i < expected.Length; i++)
        {
            for (var d = 0; d < 2; d++)
            {
                var error = Math.Abs(expected[i][d] - actual[i * 2 + d]);
                Assert.True(error < 1e-12, $"position {i}, dimension {d}: {actual[i * 2 + d]} vs torch {expected[i][d]}");
            }
        }
    }

    [Fact]
    public void Resizing_to_the_same_grid_changes_nothing()
    {
        // With the half-pixel convention every tap lands exactly on a source point, so the kernel
        // weights are [0, 1, 0, 0]. The trained resolution must never pay for an interpolation
        // that could shift a value in the last bit.
        var grid = Grid(4, 2);
        var actual = VisionTransformer.Bicubic(grid, 0, 4, 2, 4, 4);

        for (var i = 0; i < actual.Length; i++) Assert.Equal(grid.At(i), actual[i], 12);
    }

    [Fact]
    public void A_larger_image_gets_interpolated_positions_and_keeps_the_class_position()
    {
        using var directory = TinyVit.Write(mutate: tensors =>
        {
            var positions = NdArray.Zeros(1, 1 + TinyVit.Patches, TinyVit.Hidden);
            var patches = Grid(2, TinyVit.Hidden, first: 1);

            for (var d = 0; d < TinyVit.Hidden; d++) positions[0, 0, d] = 40 + d;
            for (var i = TinyVit.Hidden; i < patches.Size; i++) positions.SetAt(i, patches.At(i));

            tensors["vit.embeddings.position_embeddings"] = positions;
        });

        using var model = VisionTransformer.Open(directory.Directory);

        // 12px in 4px patches is a 3x3 grid against the 2x2 the checkpoint was trained on. The
        // projection is zero, so what comes out is the position table alone.
        var sequence = model.Embeddings(NdArray.Zeros(3, 12, 12));

        Assert.Equal(1 + 9, sequence.Shape[0]);

        // The class token's position is not part of the grid and must not be resampled with it.
        for (var d = 0; d < TinyVit.Hidden; d++) Assert.Equal(40 + d, sequence[0, d], 12);

        for (var i = 0; i < 9; i++)
        {
            for (var d = 0; d < TinyVit.Hidden; d++)
            {
                // 1e-6, not 1e-12: the grid went through a float32 checkpoint on the way in. The
                // arithmetic itself is pinned to 1e-12 by the test above.
                var error = Math.Abs(TorchTwoToThree[i][d] - sequence[1 + i, d]);
                Assert.True(error < 1e-6,$"patch {i}, dimension {d}: {sequence[1 + i, d]} vs torch {TorchTwoToThree[i][d]}");
            }
        }
    }

    [Fact]
    public void A_rectangular_image_runs_with_patches_in_row_major_order()
    {
        using var directory = TinyVit.Write(classes: 0);
        using var model = VisionTransformer.Open(directory.Directory);

        // 8 tall, 16 wide: two rows of four patches.
        var output = model.Forward(NdArray.Zeros(3, 8, 16));

        Assert.Equal(1 + 2 * 4, output.Shape[0]);
        Assert.Equal(TinyVit.Hidden, output.Shape[1]);
    }

    [Fact]
    public void An_image_size_is_chosen_at_load_and_reaches_the_processor()
    {
        using var directory = TinyVit.Write();
        using var model = VisionTransformer.Open(directory.Directory, imageSize: 16);

        Assert.Equal(16, model.Processor.Size);

        var image = Path.Combine(Path.GetTempPath(), $"hfnet-grey-{Guid.NewGuid():N}.png");
        using (var picture = new Image<Rgb24>(5, 7)) picture.Save(image);

        try
        {
            // A 5x7 picture is stretched to 16x16, a 4x4 grid - which only runs if the positions
            // were interpolated from the checkpoint's 2x2.
            Assert.Equal(3, model.Classify(image, topK: 0).Count);
        }
        finally
        {
            File.Delete(image);
        }
    }

    [Theory]
    [InlineData(10)]
    [InlineData(0)]
    public void An_image_size_that_is_not_a_whole_number_of_patches_is_refused(int size)
    {
        using var directory = TinyVit.Write();

        var error = Assert.Throws<ArgumentOutOfRangeException>(
            () => VisionTransformer.Open(directory.Directory, imageSize: size));
        Assert.Contains("multiple of the 4px patch", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Pixels_that_do_not_divide_into_patches_are_refused_at_the_forward_pass()
    {
        using var directory = TinyVit.Write(classes: 0);
        using var model = VisionTransformer.Open(directory.Directory);

        var error = Assert.Throws<ArgumentException>(() => model.Forward(NdArray.Zeros(3, 8, 10)));
        Assert.Contains("multiple of 4px", error.Message, StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------ activation

    [Theory]
    // x, torch.nn.functional.gelu(x), gelu(x, approximate="tanh") - float64.
    [InlineData(-3.0, -0.0040496940948903104, -0.0036373920817729943)]
    [InlineData(-1.0, -0.15865525393145702, -0.15880800939172329)]
    [InlineData(-0.5, -0.15426876936299344, -0.15428599017485606)]
    [InlineData(0.3, 0.1853734266566858, 0.18537092354275922)]
    [InlineData(1.0, 0.84134474606854304, 0.84119199060827676)]
    [InlineData(2.5, 2.4844758366855597, 2.4849157339100012)]
    [InlineData(4.0, 3.9998733150326675, 3.9999297540518075)]
    public void Gelu_is_the_exact_one_and_gelu_new_the_tanh_one(double x, double exact, double tanh)
    {
        // The two differ by up to 4e-4 here - the size of a rounding error to the eye, and enough
        // over twelve blocks to move a probability in the third decimal place.
        Assert.True(Math.Abs(Activation.For("gelu")(x) - exact) < 1e-13, "gelu");
        Assert.True(Math.Abs(Activation.For("gelu_new")(x) - tanh) < 1e-13, "gelu_new");
        Assert.True(Math.Abs(Activation.For("gelu_pytorch_tanh")(x) - tanh) < 1e-13, "tanh");
    }

    [Fact]
    public void An_unknown_activation_is_refused_by_name()
    {
        var error = Assert.Throws<NotSupportedException>(() => Activation.For("swish"));
        Assert.Contains("'swish'", error.Message, StringComparison.Ordinal);
        Assert.Contains("ONNX", error.Message, StringComparison.Ordinal);
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
