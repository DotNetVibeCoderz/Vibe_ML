using MediaPipeNet.Imaging;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Tasks.Vision.Processing;

namespace MediaPipeNet.Tasks.Tests;

public class AnchorTests
{
    [Fact]
    public void Blaze_anchor_counts_match_models()
    {
        SsdAnchors.Generate(SsdDetectorSpec.FaceShortRange.Anchors).Should().HaveCount(896);
        SsdAnchors.Generate(SsdDetectorSpec.Palm.Anchors).Should().HaveCount(2016);
        SsdAnchors.Generate(SsdDetectorSpec.Pose.Anchors).Should().HaveCount(2254);
        var first = SsdAnchors.Generate(SsdDetectorSpec.FaceShortRange.Anchors)[0];
        first.Should().Be(new Anchor(0.5f / 16, 0.5f / 16, 1, 1));
        var sized = SsdAnchors.Generate(new SsdAnchorOptions { InputWidth = 64, InputHeight = 64, Strides = [32], FixedAnchorSize = false, AspectRatios = [1f, 2f] });
        sized.Should().HaveCount(2 * 2 * 3);
        sized[1].Width.Should().BeGreaterThan(sized[1].Height);
    }

    [Fact]
    public void EfficientDet_anchors_cover_all_levels()
    {
        var anchors = SsdAnchors.GenerateEfficientDet(320);
        anchors.Should().HaveCount(19206);
        anchors[0].XCenter.Should().BeApproximately(4f / 320, 1e-6f);
        anchors[0].Width.Should().BeApproximately(3 * 8f / 320, 1e-6f);
    }
}

public class DecodingTests
{
    [Fact]
    public void Decodes_boxes_keypoints_and_scores()
    {
        var o = new DetectionDecoderOptions { NumCoords = 6, NumKeypoints = 1, XScale = 10, YScale = 10, WScale = 10, HScale = 10 };
        Anchor[] anchors = [new(0.5f, 0.5f, 1, 1), new(0.2f, 0.2f, 1, 1)];
        float[] boxes = [1, 0, 2, 4, 0, 1, /* anchor 1 */ 0, 0, 1, 1, 0, 0];
        float[] scores = [5f, -5f];
        var output = new List<RawDetection>();
        DetectionDecoder.Decode(boxes, scores, anchors, o, output);
        output.Should().ContainSingle();
        var d = output[0];
        d.XMin.Should().BeApproximately(0.6f - 0.1f, 1e-5f);
        d.YMax.Should().BeApproximately(0.5f + 0.2f, 1e-5f);
        d.Keypoint(0).Should().Be((0.5f, 0.6f));
        d.Score.Should().BeApproximately(DetectionDecoder.Sigmoid(5), 1e-6f);
        d.ToDetection(100, 100, "x", ["kp"]).Keypoints[0].Label.Should().Be("kp");

        var yxhw = o with { ReverseOutputOrder = false, ApplyExponentialOnBoxSize = true, SigmoidScore = false, NumKeypoints = 0 };
        output.Clear();
        DetectionDecoder.Decode(boxes, [0.9f, 0.1f], anchors, yxhw, output, minScore: 0.5f);
        output[0].Width.Should().BeApproximately(MathF.Exp(0.4f), 1e-5f);
    }

    [Fact]
    public void Weighted_nms_merges_and_hard_nms_suppresses()
    {
        var a = new RawDetection { XMin = 0, YMin = 0, XMax = 1, YMax = 1, Score = 0.9f, Keypoints = [0, 0] };
        var b = new RawDetection { XMin = 0.1f, YMin = 0, XMax = 1.1f, YMax = 1, Score = 0.3f, Keypoints = [1, 1] };
        var c = new RawDetection { XMin = 5, YMin = 5, XMax = 6, YMax = 6, Score = 0.5f, ClassId = 1 };
        var merged = NonMaxSuppression.Weighted([a, b, c], 0.3f);
        merged.Should().HaveCount(2);
        merged[0].XMin.Should().BeApproximately(0.3f * 0.1f / 1.2f, 1e-5f);
        merged[0].Keypoints![0].Should().BeApproximately(0.25f, 1e-5f);
        merged[0].Score.Should().Be(0.9f);
        NonMaxSuppression.Hard([a, b, c], 0.3f).Should().HaveCount(2);
        NonMaxSuppression.Hard([a, b with { ClassId = 2 }], 0.3f, perClass: true).Should().HaveCount(2);
        NonMaxSuppression.Hard([a, b, c], 0.3f, maxResults: 1).Should().ContainSingle();
    }

    [Fact]
    public void Map_to_image_undoes_letterbox()
    {
        var map = new TensorMapping(new NormalizedRect(0.5f, 0.5f, 1f, 2f), new LetterboxPadding(0, 0.25f, 0, 0.25f), 200, 100, 64, 64);
        var d = new RawDetection { XMin = 0.25f, YMin = 0.25f, XMax = 0.75f, YMax = 0.75f }.MapToImage(map);
        d.YMin.Should().BeApproximately(0, 1e-5f);
        d.YMax.Should().BeApproximately(1, 1e-5f);
        d.Box.Width.Should().BeApproximately(0.5f, 1e-5f);
    }
}

public class RoiTests
{
    [Fact]
    public void Transform_scales_shifts_and_squares()
    {
        var r = RoiCalculator.Transform(new NormalizedRect(0.5f, 0.5f, 0.2f, 0.1f), 100, 200, 2, 2, 0, -0.5f, squareLong: true);
        r.YCenter.Should().BeApproximately(0.45f, 1e-5f);
        (r.Width * 100).Should().BeApproximately(r.Height * 200, 1e-3f);
        r.Width.Should().BeApproximately(0.4f, 1e-5f);
        var rotated = RoiCalculator.Transform(new NormalizedRect(0.5f, 0.5f, 0.2f, 0.2f, MathF.PI / 2), 100, 100, shiftY: -0.5f);
        rotated.XCenter.Should().BeApproximately(0.6f, 1e-5f);
    }

    [Fact]
    public void Detection_and_alignment_rects()
    {
        var d = new RawDetection { XMin = 0.4f, YMin = 0.4f, XMax = 0.6f, YMax = 0.6f, Keypoints = [0.5f, 0.6f, 0.5f, 0.4f] };
        var rect = RoiCalculator.FromDetection(d, 100, 100, 0, 1, MathF.PI / 2);
        rect.Rotation.Should().BeApproximately(0, 1e-5f);
        rect.Width.Should().BeApproximately(0.2f, 1e-5f);
        var align = RoiCalculator.FromAlignmentPoints(0.5f, 0.5f, 0.5f, 0.3f, 100, 200, MathF.PI / 2);
        (align.Width * 100).Should().BeApproximately(80, 1e-3f);
        RoiCalculator.Overlap(rect, rect).Should().BeApproximately(1, 1e-5f);
    }

    [Fact]
    public void Hand_landmarks_to_rect_is_upright_for_vertical_hand()
    {
        var lm = new NormalizedLandmark[21];
        for (int i = 0; i < 21; i++) lm[i] = new NormalizedLandmark(0.5f + (i % 5 - 2) * 0.02f, 0.8f - i * 0.02f);
        lm[0] = new NormalizedLandmark(0.5f, 0.9f);
        var rect = RoiCalculator.FromHandLandmarks(lm, 100, 100);
        MathF.Abs(rect.Rotation).Should().BeLessThan(0.3f);
        rect.YCenter.Should().BeLessThan(0.9f);
        var face = RoiCalculator.FromLandmarkBounds(lm, 100, 100, 0, 1, 0);
        face.Width.Should().BeGreaterThan(0);
    }

    [Fact]
    public void One_euro_filter_smooths_jitter()
    {
        var f = new OneEuroFilter(1, 0);
        f.Filter(0, 0).Should().Be(0);
        f.Filter(10, 0.01).Should().BeLessThan(10);
        f.Reset();
        f.Filter(5, 1).Should().Be(5);
        var smoother = new LandmarkSmoother(1);
        var arr = new[] { new NormalizedLandmark(0.5f, 0.5f) };
        smoother.Apply(arr, 0, 100, 100, 50);
        arr[0] = new NormalizedLandmark(0.6f, 0.5f);
        smoother.Apply(arr, 16, 100, 100, 50);
        arr[0].X.Should().BeInRange(0.5f, 0.6f);
        smoother.Reset();
    }

    [Fact]
    public void Labels_and_connections()
    {
        Labels.Coco.Should().HaveCount(90);
        Labels.ImageNet.Should().HaveCount(1000);
        Labels.Gestures.Should().HaveCount(8).And.Contain("Victory");
        Connections.Hand.Should().HaveCount(21);
        Connections.Pose.Should().HaveCount(35);
        Connections.FaceContours.Should().OnlyContain(e => e.From < 478 && e.To < 478);
        FaceLandmarker.BlendshapeNames.Should().HaveCount(52);
    }
}
