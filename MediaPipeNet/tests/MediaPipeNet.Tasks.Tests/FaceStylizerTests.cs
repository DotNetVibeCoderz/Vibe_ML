using System.Text.Json;
using FluentAssertions;
using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Tests;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using Xunit;

namespace MediaPipeNet.Tasks.Tests;

/// <summary>
/// FaceStylizer against MediaPipe 0.10.21 (tests/assets/golden/mediapipe_face_stylizer_reference.json). The
/// generator injects noise, so outputs are compared at 32×32 against the average of 16 MediaPipe runs.
/// </summary>
public class FaceStylizerTests
{
    private static readonly Lazy<JsonElement> s_golden = new(() =>
        JsonDocument.Parse(File.ReadAllText(TestPaths.GoldenFaceStylizer)).RootElement.GetProperty("results"));

    private static float[] Grid(MPImage image, int grid)
    {
        int block = image.Width / grid;
        var result = new float[grid * grid * 3];
        for (int y = 0; y < image.Height; y++)
            for (int x = 0; x < image.Width; x++)
            {
                var p = image[x, y];
                int i = ((y / block) * grid + x / block) * 3;
                result[i] += p.R;
                result[i + 1] += p.G;
                result[i + 2] += p.B;
            }
        for (int i = 0; i < result.Length; i++) result[i] /= block * block;
        return result;
    }

    private static float DistanceToGolden(MPImage stylized, string imageName)
    {
        var expected = s_golden.Value.GetProperty(imageName);
        int grid = expected.GetProperty("grid").GetInt32();
        var mean = expected.GetProperty("mean").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        var actual = Grid(stylized, grid);
        return actual.Zip(mean, (a, b) => MathF.Abs(a - b)).Average();
    }

    [Fact]
    public void Stylized_portrait_matches_mediapipe()
    {
        using var stylizer = FaceStylizer.Create(new() { BaseOptions = Fixtures.Base, OutputFaceAlignment = true });
        using var image = Fixtures.Load("portrait.jpg");
        using var result = stylizer.Stylize(image);

        result.HasFace.Should().BeTrue();
        result.StylizedImage!.Width.Should().Be(s_golden.Value.GetProperty("portrait.jpg").GetProperty("size")[1].GetInt32());
        result.FaceAlignment!.Size.Should().Be(result.StylizedImage.Size);
        result.Face!.Landmarks.Should().HaveCount(FaceLandmarker.LandmarkCount);
        // Mean |Δ| per 8-bit channel at 32×32; MediaPipe's own runs spread ~0.2 around their average, the rest is
        // a sub-pixel landmark difference moving the rounded crop centre by one pixel.
        DistanceToGolden(result.StylizedImage, "portrait.jpg").Should().BeLessThan(3.5f);
    }

    [Fact]
    public void Alignment_undoes_in_plane_rotation()
    {
        using var stylizer = FaceStylizer.Create(new() { BaseOptions = Fixtures.Base });
        using var upright = Fixtures.Load("portrait.jpg");
        using var rotatedSharp = upright.ToImage();
        rotatedSharp.Mutate(c => c.Rotate(RotateMode.Rotate90));
        using var rotated = MPImage.FromImage(rotatedSharp);

        using var a = stylizer.Stylize(upright);
        using var b = stylizer.Stylize(rotated);
        a.FaceRect!.Value.Rotation.Should().BeApproximately(0f, 0.15f);
        MathF.Abs(b.FaceRect!.Value.Rotation).Should().BeApproximately(MathF.PI / 2, 0.15f);
        // MediaPipe itself lands at 8.6 (90°) / 5.1 (270°) from its upright average: the rotated face yields slightly
        // different landmarks and resampling, but the stylized face stays upright and recognisably the same.
        DistanceToGolden(b.StylizedImage!, "portrait.jpg").Should().BeLessThan(10f);
    }

    [Fact]
    public void Composite_pastes_the_face_back_inside_the_crop_only()
    {
        using var stylizer = FaceStylizer.Create(new() { BaseOptions = Fixtures.Base });
        using var image = Fixtures.Load("portrait.jpg");
        using var result = stylizer.Stylize(image);
        using var composite = result.Composite(image)!;

        composite.Size.Should().Be(image.Size);
        var rect = result.FaceRect!.Value;
        int cx = (int)(rect.XCenter * image.Width), cy = (int)(rect.YCenter * image.Height);
        composite[cx, cy].Should().NotBe(image[cx, cy]);           // the face centre is redrawn
        composite[5, image.Height - 5].Should().Be(image[5, image.Height - 5]); // far outside the crop: untouched
        FaceStylizerResult.Empty.Composite(image).Should().BeNull();
    }

    [Theory]
    [InlineData(ModelPrecision.Float16, 3.5f)]
    [InlineData(ModelPrecision.Int8, 5f)]
    public void Reduced_precision_variants_stay_close_to_mediapipe(ModelPrecision precision, float tolerance)
    {
        var options = Fixtures.Base with { Inference = new InferenceOptions { Provider = ExecutionProvider.Cpu, Precision = precision } };
        using var stylizer = FaceStylizer.Create(new() { BaseOptions = options });
        using var image = Fixtures.Load("portrait.jpg");
        using var result = stylizer.Stylize(image);
        float distance = DistanceToGolden(result.StylizedImage!, "portrait.jpg");
        Console.WriteLine($"{precision}: {distance:F2}");
        distance.Should().BeLessThan(tolerance);
    }

    [Fact]
    public void No_face_gives_empty_result()
    {
        using var stylizer = FaceStylizer.Create(new() { BaseOptions = Fixtures.Base });
        using var image = Fixtures.Load("burger.jpg");
        using var result = stylizer.Stylize(image);
        result.HasFace.Should().BeFalse();
        result.FaceRect.Should().BeNull();
    }

    [Fact]
    public void Face_rect_follows_mediapipe_face_to_rect()
    {
        // Upright face in a 1000×1000 image: eyes at y=400 (x=400 and 600), mouth at y=600.
        var landmarks = new NormalizedLandmark[FaceLandmarker.LandmarkCount];
        void Set(int i, float x, float y) => landmarks[i] = new NormalizedLandmark(x / 1000f, y / 1000f);
        Set(33, 380, 400); Set(133, 420, 400); Set(263, 620, 400); Set(362, 580, 400); Set(61, 460, 600); Set(291, 540, 600);
        var rect = FaceStylizer.ComputeFaceRect(landmarks, 1000, 1000);
        rect.XCenter.Should().BeApproximately(0.5f, 1e-4f);
        rect.YCenter.Should().BeApproximately(0.42f, 1e-4f);        // eye centre + 0.1 × eye-to-mouth
        rect.Width.Should().BeApproximately(0.8f, 1e-4f);           // max(200 × 3.6, 200 × 4.0) = 800 px
        rect.Height.Should().Be(rect.Width);
        rect.Rotation.Should().BeApproximately(0f, 1e-5f);
    }
}
