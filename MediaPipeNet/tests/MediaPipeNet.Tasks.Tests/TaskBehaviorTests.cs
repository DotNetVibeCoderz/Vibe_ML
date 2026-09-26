using System.Collections.Concurrent;
using MediaPipeNet.Extensions.DependencyInjection;
using MediaPipeNet.Framework.Config;
using MediaPipeNet.Imaging;
using MediaPipeNet.Serialization;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Tasks.Vision.Graph;
using MediaPipeNet.Visualization;
using Microsoft.Extensions.DependencyInjection;
using SixLabors.ImageSharp;

namespace MediaPipeNet.Tasks.Tests;

public class RunningModeTests
{
    [Fact]
    public void Methods_require_the_matching_mode()
    {
        using var image = Fixtures.Load("portrait.jpg");
        using var imageMode = FaceDetector.Create(new() { BaseOptions = Fixtures.Base });
        imageMode.Invoking(d => d.DetectForVideo(image, 1)).Should().Throw<InvalidOperationException>();
        using var videoMode = FaceDetector.Create(new() { BaseOptions = Fixtures.Base, RunningMode = RunningMode.Video });
        videoMode.Invoking(d => d.Detect(image)).Should().Throw<InvalidOperationException>();
        videoMode.DetectForVideo(image, 10).Detections.Should().ContainSingle();
        videoMode.Invoking(d => d.DetectForVideo(image, 10)).Should().Throw<ArgumentException>().WithMessage("*monotonic*");
        FluentActions.Invoking(() => FaceDetector.Create(new() { BaseOptions = Fixtures.Base, RunningMode = RunningMode.LiveStream }))
            .Should().Throw<ArgumentException>().WithMessage("*ResultCallback*");
        FluentActions.Invoking(() => FaceDetector.Create(new() { BaseOptions = Fixtures.Base, ResultCallback = (_, _, _) => { } }))
            .Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Disposed_tasks_throw()
    {
        var detector = FaceDetector.Create(new() { BaseOptions = Fixtures.Base });
        detector.Dispose();
        using var image = Fixtures.Load("burger.jpg");
        detector.Invoking(d => d.Detect(image)).Should().Throw<ObjectDisposedException>();
        detector.Dispose();
    }

    [Fact]
    public async Task Live_stream_delivers_results_through_callback()
    {
        var results = new ConcurrentQueue<(int Hands, long Ts)>();
        using var done = new SemaphoreSlim(0);
        using var landmarker = await HandLandmarker.CreateAsync(new()
        {
            BaseOptions = Fixtures.Base,
            RunningMode = RunningMode.LiveStream,
            ResultCallback = (r, frame, ts) =>
            {
                frame.Width.Should().BeGreaterThan(0);
                results.Enqueue((r.Hands.Count, ts));
                done.Release();
            },
        });
        using var image = Fixtures.Load("victory.jpg");
        int accepted = 0;
        for (int i = 0; i < 10; i++)
            if (landmarker.DetectLiveStream(image, i * 33L)) accepted++;
        for (int i = 0; i < accepted; i++) (await done.WaitAsync(TimeSpan.FromSeconds(30))).Should().BeTrue();
        results.Should().HaveCount(accepted);
        results.Should().OnlyContain(r => r.Hands == 1);
        results.Select(r => r.Ts).Should().BeInAscendingOrder();
        (accepted + landmarker.DroppedFrames).Should().Be(10);
    }

    [Fact]
    public void Video_mode_tracks_between_frames()
    {
        using var image = Fixtures.Load("victory.jpg");
        using var video = HandLandmarker.Create(new() { BaseOptions = Fixtures.Base, RunningMode = RunningMode.Video });
        var first = video.DetectForVideo(image, 0).Hands[0];
        var second = video.DetectForVideo(image, 33).Hands[0];
        // The second frame reuses the landmark-derived ROI, so results stay close to the first.
        Golden.MeanError(first.Landmarks, second.Landmarks).Should().BeLessThan(0.02f);
        video.ResetTracking();
        video.DetectForVideo(image, 66).Hands.Should().ContainSingle();
    }

    [Fact]
    public async Task Async_apis_and_processing_options()
    {
        using var classifier = await ImageClassifier.CreateAsync(new() { BaseOptions = Fixtures.Base, MaxResults = 2 });
        using var image = Fixtures.Load("burger.jpg");
        (await classifier.ClassifyAsync(image)).Categories.Should().HaveCount(2);
        using var allow = ImageClassifier.Create(new() { BaseOptions = Fixtures.Base, CategoryAllowlist = new HashSet<string> { "bagel" } });
        allow.Classify(image).Categories.Should().ContainSingle().Which.CategoryName.Should().Be("bagel");
        using var rotated = Fixtures.Load("burger.jpg");
        classifier.Classify(rotated, new ImageProcessingOptions(RotationDegrees: 90)).Categories.Should().NotBeEmpty();
        FluentActions.Invoking(() => classifier.Classify(rotated, new ImageProcessingOptions(RotationDegrees: 45))).Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Object_detector_filters()
    {
        using var image = Fixtures.Load("cats_and_dogs.jpg");
        using var cats = ObjectDetector.Create(new() { BaseOptions = Fixtures.Base, CategoryAllowlist = new HashSet<string> { "cat" } });
        cats.Detect(image).Detections.Should().OnlyContain(d => d.TopCategory.CategoryName == "cat");
        using var one = ObjectDetector.Create(new() { BaseOptions = Fixtures.Base, MaxResults = 1, CategoryDenylist = new HashSet<string> { "cat" } });
        one.Detect(image).Detections.Should().ContainSingle().Which.TopCategory.CategoryName.Should().Be("dog");
    }

    [Fact]
    public void Holistic_combines_all_parts()
    {
        using var holistic = HolisticLandmarker.Create(new() { BaseOptions = Fixtures.Base });
        using var image = Fixtures.Load("pose.jpg");
        var r = holistic.Detect(image);
        r.Pose.Should().NotBeNull();
        r.Face.Should().NotBeNull();
        r.LeftHand.Should().NotBeNull();
        r.RightHand.Should().NotBeNull();
        // The person's right hand is on the image's left side.
        r.RightHand!.Landmarks[0].X.Should().BeLessThan(r.LeftHand!.Landmarks[0].X);
    }

    [Fact]
    public void Segmenter_and_pose_video_modes()
    {
        using var image = Fixtures.Load("pose.jpg");
        using var seg = ImageSegmenter.Create(new() { BaseOptions = Fixtures.Base, RunningMode = RunningMode.Video });
        seg.SegmentForVideo(image, 0);
        seg.SegmentForVideo(image, 33).ConfidenceMask.Coverage().Should().BeInRange(0.02f, 0.3f);
        using var pose = PoseLandmarker.Create(new() { BaseOptions = Fixtures.Base, RunningMode = RunningMode.Video, Model = PoseModel.Full });
        pose.DetectForVideo(image, 0).Poses.Should().ContainSingle();
        pose.DetectForVideo(image, 33).Poses.Should().ContainSingle();
    }
}

public class SerializationTests
{
    [Fact]
    public void Results_round_trip_through_json()
    {
        using var image = Fixtures.Load("victory.jpg");
        using var gestures = GestureRecognizer.Create(new() { BaseOptions = Fixtures.Base });
        var result = gestures.Recognize(image);
        var json = result.ToJson(indented: true);
        json.Should().Contain("\"gestures\"").And.Contain("Victory").And.Contain("\"handedness\"");
        var back = MediaPipeJson.Deserialize<GestureRecognitionResult>(json)!;
        back.Hands[0].TopGesture.CategoryName.Should().Be("Victory");
        back.Hands[0].Hand.Landmarks.Should().HaveCount(21);
    }

    [Fact]
    public void Segmentation_mask_serializes_compactly()
    {
        var mask = new SegmentationMask(2, 2, [0f, 0.25f, 0.75f, 1f]);
        var json = new SegmentationResult(mask).ToJson();
        json.Should().Contain("float32-base64");
        var back = MediaPipeJson.Deserialize<SegmentationResult>(json)!;
        back.ConfidenceMask.Data.Should().Equal(mask.Data);
        mask.ToBytes().Should().Equal(0, 64, 191, 255);
        mask.ToBytes(0.5f).Should().Equal(0, 0, 255, 255);
        mask[1, 1].Should().Be(1);
        FaceDetectionResult.Empty.ToJson().Should().Contain("detections");
    }
}

public class IntegrationTests
{
    [Fact]
    public void Dependency_injection_registers_singletons()
    {
        var services = new ServiceCollection();
        services.AddLogging();
        services.AddMediaPipeNet(o => { o.ModelDirectory = MediaPipeNet.Tests.TestPaths.Models; o.AllowModelDownload = false; })
            .AddFaceDetector(o => o with { MinDetectionConfidence = 0.6f })
            .AddImageClassifier();
        using var sp = services.BuildServiceProvider();
        var detector = sp.GetRequiredService<FaceDetector>();
        detector.Options.MinDetectionConfidence.Should().Be(0.6f);
        sp.GetRequiredService<FaceDetector>().Should().BeSameAs(detector);
        using var image = Fixtures.Load("portrait.jpg");
        detector.Detect(image).Detections.Should().ContainSingle();
        sp.GetRequiredService<ImageClassifier>().Classify(image).Categories.Should().NotBeEmpty();
    }

    [Fact]
    public async Task Live_stream_processor_processes_every_frame_from_files()
    {
        await using var source = new ImageFileFrameSource([MediaPipeNet.Tests.TestPaths.Image("victory.jpg"), MediaPipeNet.Tests.TestPaths.Image("pointing_up.jpg")]);
        using var gestures = GestureRecognizer.Create(new() { BaseOptions = Fixtures.Base, RunningMode = RunningMode.Video });
        await using var processor = new LiveStreamProcessor<GestureRecognitionResult>(source, (img, ts) => gestures.RecognizeForVideo(img, ts));
        var names = new List<string>();
        processor.ResultReady += (_, r) => names.Add(r.Result.Hands[0].TopGesture.CategoryName!);
        await processor.RunAsync();
        names.Should().Equal("Victory", "Pointing_Up");
        processor.Stats.FramesProcessed.Should().Be(2);
        processor.Stats.FramesDropped.Should().Be(0);
    }

    [Fact]
    public async Task Live_stream_processor_drops_frames_for_live_sources()
    {
        using var frame = Fixtures.Load("victory.jpg");
        var frames = Enumerable.Repeat(frame, 30).ToArray();
        await using var source = new MemoryFrameSource(frames, frameRate: 200, realTime: true);
        await using var processor = new LiveStreamProcessor<int>(source, (img, ts) => { Thread.Sleep(20); return img.Width; }, new() { MirrorFrames = true });
        int processed = 0;
        processor.ResultReady += (_, _) => processed++;
        await processor.RunAsync();
        processor.Stats.FramesCaptured.Should().Be(30);
        processor.Stats.FramesDropped.Should().BeGreaterThan(0);
        processed.Should().BeLessThan(30).And.BeGreaterThan(0);
    }

    [Fact]
    public async Task Vision_calculators_run_inside_a_pbtxt_graph()
    {
        var registry = new CalculatorRegistry().AddVisionCalculators(Fixtures.Base);
        var config = GraphConfig.ParsePbtxt("""
            input_stream: "image"
            output_stream: "faces"
            output_stream: "objects"
            node { calculator: "FaceDetectorCalculator" input_stream: "IMAGE:image" output_stream: "RESULT:faces" }
            node {
              calculator: "ObjectDetectorCalculator"
              input_stream: "IMAGE:image"
              output_stream: "RESULT:objects"
              options { score_threshold: 0.5 max_results: 3 }
            }
            """);
        var graph = config.Build(registry);
        var faces = new ConcurrentQueue<FaceDetectionResult>();
        var objects = new ConcurrentQueue<ObjectDetectionResult>();
        graph.ObserveOutputStream<FaceDetectionResult>("faces", p => faces.Enqueue(p.Value));
        graph.ObserveOutputStream<ObjectDetectionResult>("objects", p => objects.Enqueue(p.Value));
        await graph.StartAsync();
        using var image = Fixtures.Load("portrait.jpg");
        graph.AddPacket("image", image, 0);
        graph.AddPacket("image", image, 1);
        await graph.CloseAsync();
        faces.Should().HaveCount(2).And.OnlyContain(f => f.Detections.Count == 1);
        objects.Should().HaveCount(2).And.OnlyContain(o => o.Detections.Any(d => d.TopCategory.CategoryName == "person"));
        registry.Names.Should().HaveCount(9);
    }

    [Fact]
    public void Visualization_renders_every_result_type()
    {
        using var portrait = Fixtures.Load("portrait.jpg");
        using var pose = Fixtures.Load("pose.jpg");
        using var canvas = portrait.ToImage();
        using (var t = FaceDetector.Create(new() { BaseOptions = Fixtures.Base })) ResultRenderer.Render(canvas, t.Detect(portrait));
        using (var t = FaceLandmarker.Create(new() { BaseOptions = Fixtures.Base })) ResultRenderer.Render(canvas, t.Detect(portrait));
        using (var t = ImageSegmenter.Create(new() { BaseOptions = Fixtures.Base })) ResultRenderer.Render(canvas, t.Segment(portrait));
        using (var t = ObjectDetector.Create(new() { BaseOptions = Fixtures.Base })) ResultRenderer.Render(canvas, t.Detect(portrait));
        using var poseCanvas = pose.ToImage();
        using (var t = PoseLandmarker.Create(new() { BaseOptions = Fixtures.Base, OutputSegmentationMasks = true })) ResultRenderer.Render(poseCanvas, t.Detect(pose));
        using (var t = HolisticLandmarker.Create(new() { BaseOptions = Fixtures.Base })) ResultRenderer.Render(poseCanvas, t.Detect(pose));
        using var hand = Fixtures.Load("victory.jpg");
        using var handCanvas = hand.ToImage();
        using (var t = HandLandmarker.Create(new() { BaseOptions = Fixtures.Base })) ResultRenderer.Render(handCanvas, t.Detect(hand));
        using (var t = GestureRecognizer.Create(new() { BaseOptions = Fixtures.Base })) ResultRenderer.Render(handCanvas, t.Recognize(hand));
        using (var t = ImageSegmenter.Create(new() { BaseOptions = Fixtures.Base }))
        {
            var mask = t.Segment(portrait).ConfidenceMask;
            SegmentationMaskOverlay.BlurBackground(canvas, mask, 4);
            SegmentationMaskOverlay.ReplaceBackground(canvas, mask, Color.Green);
            using var bg = hand.ToImage();
            SegmentationMaskOverlay.ReplaceBackground(canvas, mask, bg);
        }
        canvas[10, 10].G.Should().BeGreaterThan(0);
        FluentActions.Invoking(() => SegmentationMaskOverlay.Overlay(handCanvas, new SegmentationMask(1, 1, [0f]), Color.Red)).Should().Throw<ArgumentException>();
    }
}
