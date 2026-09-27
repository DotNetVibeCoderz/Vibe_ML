using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Tests;

namespace MediaPipeNet.Tasks.Tests;

/// <summary>Reduced-precision model variants (FP16, weight-only INT8) against the float32 models on real images.</summary>
public class PrecisionTests
{
    private static BaseOptions With(ModelPrecision precision) =>
        Fixtures.Base with { Inference = new InferenceOptions { Provider = ExecutionProvider.Cpu, Precision = precision } };

    [Theory]
    [InlineData(ModelPrecision.Float16, 0.002f)]
    [InlineData(ModelPrecision.Int8, 0.004f)]
    public void Face_landmarker_variants_match_float(ModelPrecision precision, float tolerance)
    {
        using var image = Fixtures.Load("portrait.jpg");
        using var reference = FaceLandmarker.Create(new() { BaseOptions = Fixtures.Base });
        using var variant = FaceLandmarker.Create(new() { BaseOptions = With(precision) });
        var expected = reference.Detect(image).Faces[0].Landmarks;
        var actual = variant.Detect(image).Faces.Should().ContainSingle().Subject.Landmarks;
        Golden.MeanError(actual, expected).Should().BeLessThan(tolerance);
    }

    [Theory]
    [InlineData(ModelPrecision.Float16)]
    [InlineData(ModelPrecision.Int8)]
    public void Object_detector_variants_find_the_same_objects(ModelPrecision precision)
    {
        using var image = Fixtures.Load("cats_and_dogs.jpg");
        using var reference = ObjectDetector.Create(new() { BaseOptions = Fixtures.Base });
        using var variant = ObjectDetector.Create(new() { BaseOptions = With(precision) });
        var expected = reference.Detect(image).Detections;
        var actual = variant.Detect(image).Detections;
        actual.Select(d => d.TopCategory.CategoryName).Should().Equal(expected.Select(d => d.TopCategory.CategoryName));
        for (int i = 0; i < expected.Count; i++)
            RectF.IntersectionOverUnion(actual[i].BoundingBox, expected[i].BoundingBox).Should().BeGreaterThan(0.95f);
    }

    [Fact]
    public void Segmenter_fp16_matches_float()
    {
        using var image = Fixtures.Load("portrait.jpg");
        using var reference = ImageSegmenter.Create(new() { BaseOptions = Fixtures.Base, Model = SegmenterModel.SelfieMulticlass, OutputCategoryMask = true });
        using var variant = ImageSegmenter.Create(new() { BaseOptions = With(ModelPrecision.Float16), Model = SegmenterModel.SelfieMulticlass, OutputCategoryMask = true });
        var a = reference.Segment(image).CategoryMask!.Data;
        var b = variant.Segment(image).CategoryMask!.Data;
        a.Zip(b).Count(p => p.First != p.Second).Should().BeLessThan(a.Length / 200); // < 0.5% of pixels
    }

    [Fact]
    public void Catalog_selects_variants_and_falls_back_to_float()
    {
        ModelCatalog.QuantizedVariants.Should().NotBeEmpty().And.OnlyContain(v => v.PackageId == ModelCatalog.QuantizedPackage);
        var fp16 = ModelCatalog.GetVariant(ModelCatalog.FaceLandmarksDetector, ModelPrecision.Float16)!;
        fp16.FileName.Should().Be("face_landmarks_detector.fp16.onnx");
        fp16.Precision.Should().Be(ModelPrecision.Float16);
        ModelCatalog.GetVariant(ModelCatalog.FaceLandmarksDetector, ModelPrecision.Float32).Should().BeNull();
        // BERT already stores int8 weights: no variant, so the float model is used.
        ModelCatalog.GetVariant(ModelCatalog.BertClassifier, ModelPrecision.Int8).Should().BeNull();
        ModelLoader.SelectVariant(With(ModelPrecision.Int8), ModelCatalog.BertClassifier).Should().BeSameAs(ModelCatalog.BertClassifier);
        foreach (var v in ModelCatalog.QuantizedVariants)
        {
            var file = new FileInfo(Path.Combine(TestPaths.Models, "quantized", v.FileName));
            file.Exists.Should().BeTrue(v.FileName);
            file.Length.Should().Be(v.SizeBytes, v.FileName);
        }
    }

    [Fact]
    public async Task Missing_variant_falls_back_to_the_float_model()
    {
        // A model directory without the quantized sub-folder and no store: the float file is used.
        var dir = Directory.CreateTempSubdirectory("mpnet-precision").FullName;
        try
        {
            File.Copy(Path.Combine(TestPaths.Models, "face_detection_short_range.onnx"), Path.Combine(dir, "face_detection_short_range.onnx"));
            File.Copy(Path.Combine(TestPaths.Models, "face_detection_full_range.onnx"), Path.Combine(dir, "face_detection_full_range.onnx"));
            var options = new BaseOptions
            {
                ModelDirectory = dir,
                ModelStore = new ModelStore([]),
                Inference = new InferenceOptions { Precision = ModelPrecision.Int8 },
            };
            var path = await ModelLoader.ResolvePathAsync(options, ModelCatalog.FaceDetectionFullRange);
            Path.GetFileName(path).Should().Be("face_detection_full_range.onnx");
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }
}
