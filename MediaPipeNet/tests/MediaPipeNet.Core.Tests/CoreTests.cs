using MediaPipeNet.Diagnostics;
using MediaPipeNet.Serialization;

namespace MediaPipeNet.Core.Tests;

public class TimestampTests
{
    [Fact]
    public void Conversions_round_trip()
    {
        Timestamp.FromMilliseconds(1500).Value.Should().Be(1_500_000);
        Timestamp.FromSeconds(1.5).Milliseconds.Should().Be(1500);
        Timestamp.FromTimeSpan(TimeSpan.FromMilliseconds(2)).Value.Should().Be(2000);
        new Timestamp(3_000_000).Seconds.Should().Be(3);
    }

    [Fact]
    public void Special_values_are_ordered_and_named()
    {
        Timestamp.PreStream.Should().BeLessThan(Timestamp.Min);
        Timestamp.Max.Should().BeLessThan(Timestamp.PostStream);
        Timestamp.PostStream.Should().BeLessThan(Timestamp.Done);
        Timestamp.Done.IsSpecialValue.Should().BeTrue();
        new Timestamp(5).IsRangeValue.Should().BeTrue();
        Timestamp.Done.ToString().Should().Be("Timestamp::Done");
        Timestamp.Unset.ToString().Should().Contain("Unset");
        new Timestamp(42).ToString().Should().Be("42us");
    }

    [Fact]
    public void Next_allowed_saturates()
    {
        new Timestamp(7).NextAllowedInStream().Should().Be(new Timestamp(8));
        Timestamp.Max.NextAllowedInStream().Should().Be(Timestamp.Done);
        Timestamp.Maximum(new(1), new(2)).Should().Be(new Timestamp(2));
        Timestamp.Minimum(new(1), new(2)).Should().Be(new Timestamp(1));
        (new Timestamp(1) <= new Timestamp(1)).Should().BeTrue();
        (new Timestamp(2) >= new Timestamp(1)).Should().BeTrue();
    }
}

public class GeometryTests
{
    [Fact]
    public void Iou_of_identical_and_disjoint_boxes()
    {
        var a = new RectF(0, 0, 10, 10);
        RectF.IntersectionOverUnion(a, a).Should().Be(1);
        RectF.IntersectionOverUnion(a, new RectF(20, 20, 5, 5)).Should().Be(0);
        RectF.IntersectionOverUnion(a, new RectF(5, 0, 10, 10)).Should().BeApproximately(50f / 150f, 1e-5f);
    }

    [Fact]
    public void Rect_helpers()
    {
        var r = RectF.FromLtrb(1, 2, 5, 10);
        r.Width.Should().Be(4);
        r.CenterX.Should().Be(3);
        r.CenterY.Should().Be(6);
        r.Scale(2, 3).Should().Be(new RectF(2, 6, 8, 24));
        new RectF(-5, -5, 20, 20).Clamp(10, 10).Should().Be(new RectF(0, 0, 10, 10));
        new RectF(0, 0, -1, 5).IsEmpty.Should().BeTrue();
        RectF.Intersect(new RectF(0, 0, 1, 1), new RectF(2, 2, 1, 1)).IsEmpty.Should().BeTrue();
    }

    [Fact]
    public void Normalized_rect_corners_follow_rotation()
    {
        var r = new NormalizedRect(0.5f, 0.5f, 0.5f, 0.5f);
        r.ToPixelBounds(100, 100).Should().Be(new RectF(25, 25, 50, 50));
        var corners = (r with { Rotation = MathF.PI / 2 }).GetCorners(100, 100);
        // Top-left corner rotated 90° clockwise lands at top-right.
        corners[0].X.Should().BeApproximately(0.75f, 1e-5f);
        corners[0].Y.Should().BeApproximately(0.25f, 1e-5f);
    }

    [Fact]
    public void Letterbox_padding_removal()
    {
        var p = new LetterboxPadding(0.25f, 0, 0.25f, 0);
        p.ContentWidth.Should().Be(0.5f);
        p.Remove(0.5f, 0.5f).Should().Be((0.5f, 0.5f));
        p.Remove(0.25f, 0f).Should().Be((0f, 0f));
        LetterboxPadding.None.ContentHeight.Should().Be(1);
    }

    [Theory]
    [InlineData(0f, 0f)]
    [InlineData(4f, 4f - 2 * MathF.PI)]
    [InlineData(-4f, -4f + 2 * MathF.PI)]
    public void Normalize_radians(float input, float expected) =>
        Angles.NormalizeRadians(input).Should().BeApproximately(expected, 1e-5f);

    [Fact]
    public void Rotation_matches_mediapipe_convention()
    {
        // Vector pointing straight up (y decreasing) with a 90° target needs no rotation.
        Angles.ComputeRotation(0, 10, 0, 0, MathF.PI / 2).Should().BeApproximately(0, 1e-5f);
        // Vector pointing right needs a quarter turn.
        Angles.ComputeRotation(0, 0, 10, 0, MathF.PI / 2).Should().BeApproximately(MathF.PI / 2, 1e-5f);
        Angles.RadiansToDegrees(Angles.DegreesToRadians(30)).Should().BeApproximately(30, 1e-4f);
    }
}

public class ResultTypeTests
{
    [Fact]
    public void Detection_and_category_helpers()
    {
        var d = new Detection(new RectF(1, 2, 3, 4), [new Category(3, 0.9f, "cat")], [new NormalizedKeypoint(0.1f, 0.2f, "eye")]);
        d.Score.Should().Be(0.9f);
        d.TopCategory.CategoryName.Should().Be("cat");
        d.TopCategory.ToString().Should().Contain("cat");
        new NormalizedLandmark(0.5f, 0.25f).ToPixel(200, 100).Should().Be((100f, 25f));
    }

    [Fact]
    public void Json_uses_camel_case_and_skips_nulls()
    {
        var d = new Detection(new RectF(1, 2, 3, 4), [new Category(0, 0.5f)], []);
        var json = MediaPipeJson.Serialize(d);
        json.Should().Contain("\"boundingBox\"").And.Contain("\"categories\"").And.NotContain("categoryName").And.NotContain("topCategory");
        var back = MediaPipeJson.Deserialize<Detection>(json)!;
        back.BoundingBox.Should().Be(d.BoundingBox);
        MediaPipeJson.Serialize(d, indented: true).Should().Contain("\n");
    }

    [Fact]
    public void Exceptions_carry_messages()
    {
        new MediaPipeException("x").Message.Should().Be("x");
        new ModelNotFoundException("m", new IOException()).InnerException.Should().BeOfType<IOException>();
        new MediaPipeException().Should().NotBeNull();
        new ModelNotFoundException().Should().NotBeNull();
    }
}

public class TelemetryTests
{
    [Fact]
    public void Frame_rate_counter_measures_rate()
    {
        var c = new FrameRateCounter(5);
        c.Rate.Should().Be(0);
        for (int i = 0; i < 5; i++)
        {
            c.Tick();
            Thread.Sleep(10);
        }
        c.Rate.Should().BeInRange(10, 200);
        c.Reset();
        c.Rate.Should().Be(0);
    }

    [Fact]
    public void Instruments_are_exposed()
    {
        MediaPipeTelemetry.Meter.Name.Should().Be("MediaPipeNet");
        MediaPipeTelemetry.InferenceDuration.Name.Should().Be("mediapipenet.inference.duration");
        MediaPipeTelemetry.ActivitySource.Name.Should().Be("MediaPipeNet");
    }
}
