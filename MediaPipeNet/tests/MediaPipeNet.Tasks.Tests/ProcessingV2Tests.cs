using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Tasks.Vision.Processing;

namespace MediaPipeNet.Tasks.Tests;

/// <summary>Unit tests of the 0.2 post-processing ports.</summary>
public class ProcessingV2Tests
{
    [Fact]
    public void Segmentation_smoother_keeps_confident_pixels_and_damps_uncertain_ones()
    {
        var smoother = new SegmentationSmoother(0.7f);
        float[] first = [1f, 0f, 0.5f];
        smoother.Apply(first); // stored as is
        first.Should().Equal(1f, 0f, 0.5f);
        float[] second = [0f, 1f, 0.55f];
        smoother.Apply(second);
        second[0].Should().BeApproximately(0f, 1e-4f);     // a confident new value (almost) wins outright
        second[1].Should().BeApproximately(1f, 1e-4f);
        second[2].Should().BeApproximately(0.55f + (0.5f - 0.55f) * 0.7f * 0.98f, 0.01f); // uncertain: mostly the previous value
        smoother.Reset();
        float[] third = [0.3f];
        smoother.Apply(third);
        third.Should().Equal(0.3f);
    }

    [Fact]
    public void Heatmap_refinement_moves_landmarks_to_the_heat_centroid()
    {
        const int size = 16;
        var heatmap = new float[size * size * 2];
        Array.Fill(heatmap, -10f);
        heatmap[(9 * size + 11) * 2] = 10f;           // landmark 0: peak at (11, 9)
        Span<float> xy = [10f / size, 8f / size, 3f / size, 3f / size];
        HeatmapRefinement.Refine(xy, heatmap, size, size);
        xy[0].Should().BeApproximately(11f / size, 1e-3f);
        xy[1].Should().BeApproximately(9f / size, 1e-3f);
        xy[2].Should().Be(3f / size);                  // landmark 1 has no confident heat: unchanged
    }

    [Fact]
    public void Roi_tracking_keeps_a_stable_roi_and_recrops_after_large_motion()
    {
        var landmarks = new[] { new NormalizedLandmark(0.5f, 0.5f), new NormalizedLandmark(0.52f, 0.48f) };
        var previous = new NormalizedRect(0.5f, 0.5f, 0.2f, 0.2f);
        var close = new NormalizedRect(0.51f, 0.5f, 0.21f, 0.2f);
        RoiTracking.Select(landmarks, previous, close, 640, 480, RoiTrackingRequirements.Hand).Should().Be(previous);
        var far = new NormalizedRect(0.7f, 0.5f, 0.2f, 0.2f);
        RoiTracking.Select(landmarks, previous, far, 640, 480, RoiTrackingRequirements.Hand).Should().Be(far);
        var rotated = close with { Rotation = 1.2f };
        RoiTracking.Select(landmarks, previous, rotated, 640, 480, RoiTrackingRequirements.Hand).Should().Be(rotated);
        RoiTracking.Select(null, null, close, 640, 480, RoiTrackingRequirements.Face).Should().Be(close);
    }

    [Fact]
    public void Holistic_video_mode_tracks_hands_across_frames()
    {
        using var holistic = HolisticLandmarker.Create(new() { BaseOptions = Fixtures.Base, RunningMode = RunningMode.Video });
        using var image = Fixtures.Load("pose.jpg");
        var first = holistic.DetectForVideo(image, 0);
        var second = holistic.DetectForVideo(image, 33);
        first.LeftHand.Should().NotBeNull();
        second.LeftHand.Should().NotBeNull();
        // The second frame reuses the first frame's hand ROI (ROI tracking), so results stay put.
        Golden.MeanError(second.LeftHand!.Landmarks, first.LeftHand!.Landmarks).Should().BeLessThan(0.01f);
        holistic.ResetTracking();
        holistic.DetectForVideo(image, 66).Pose.Should().NotBeNull();
    }

    [Fact]
    public void Pose_mask_is_smoothed_in_video_mode()
    {
        using var pose = PoseLandmarker.Create(new() { BaseOptions = Fixtures.Base, RunningMode = RunningMode.Video, OutputSegmentationMasks = true });
        using var image = Fixtures.Load("pose.jpg");
        var a = pose.DetectForVideo(image, 0).Poses[0].SegmentationMask!;
        var b = pose.DetectForVideo(image, 33).Poses[0].SegmentationMask!;
        b.Coverage().Should().BeApproximately(a.Coverage(), 0.02f);
    }
}
