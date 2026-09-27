using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Tasks.Vision;

namespace MediaPipeNet.Tasks.Tests;

/// <summary>Batch processing, I/O binding and concurrent image-mode calls.</summary>
public class BatchAndBindingTests
{
    private static readonly string[] s_images = ["thumb_up.jpg", "victory.jpg", "pointing_up.jpg", "portrait.jpg"];

    [Fact]
    public void Batch_results_match_sequential_results_in_order()
    {
        using var landmarker = HandLandmarker.Create(new() { BaseOptions = Fixtures.Base });
        var images = s_images.Select(Fixtures.Load).ToArray();
        try
        {
            var sequential = images.Select(i => landmarker.Detect(i)).ToArray();
            var batch = landmarker.ProcessBatch([.. images, .. images], maxDegreeOfParallelism: 4);
            batch.Should().HaveCount(images.Length * 2);
            for (int i = 0; i < batch.Count; i++)
            {
                var expected = sequential[i % images.Length];
                batch[i].Hands.Should().HaveCount(expected.Hands.Count);
                for (int h = 0; h < expected.Hands.Count; h++)
                    Golden.MeanError(batch[i].Hands[h].Landmarks, expected.Hands[h].Landmarks).Should().BeLessThan(1e-5f);
            }
        }
        finally
        {
            foreach (var i in images) i.Dispose();
        }
    }

    [Fact]
    public async Task Batch_async_supports_cancellation()
    {
        using var detector = FaceDetector.Create(new() { BaseOptions = Fixtures.Base });
        using var image = Fixtures.Load("portrait.jpg");
        var result = await detector.ProcessBatchAsync([image, image]);
        result.Should().HaveCount(2).And.OnlyContain(r => r.Detections.Count == 1);
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();
        var act = () => detector.ProcessBatchAsync(Enumerable.Repeat(image, 64).ToArray(), cancellationToken: cts.Token);
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public void Io_binding_produces_identical_results()
    {
        using var image = Fixtures.Load("portrait.jpg");
        using var plain = FaceLandmarker.Create(new() { BaseOptions = Fixtures.Base });
        using var bound = FaceLandmarker.Create(new()
        {
            BaseOptions = Fixtures.Base with { Inference = new InferenceOptions { Provider = ExecutionProvider.Cpu, UseIoBinding = true } },
        });
        var a = plain.Detect(image).Faces[0].Landmarks;
        var b = bound.Detect(image).Faces[0].Landmarks;
        Golden.MeanError(a, b).Should().BeLessThan(1e-4f);
        // Repeated runs reuse the binding.
        Golden.MeanError(bound.Detect(image).Faces[0].Landmarks, b).Should().Be(0f);
    }

    [Fact]
    public void Image_mode_calls_are_thread_safe()
    {
        using var pose = PoseLandmarker.Create(new() { BaseOptions = Fixtures.Base });
        using var image = Fixtures.Load("pose.jpg");
        var reference = pose.Detect(image).Poses[0].Landmarks;
        Parallel.For(0, 16, _ =>
        {
            var r = pose.Detect(image);
            r.Poses.Should().ContainSingle();
            Golden.MeanError(r.Poses[0].Landmarks, reference).Should().BeLessThan(1e-5f);
        });
    }

    [Fact]
    public void Dynamic_run_accepts_variable_shapes()
    {
        using var model = OnnxModel.Load(Path.Combine(MediaPipeNet.Tests.TestPaths.Models, "language_detector.onnx"));
        foreach (int tokens in new[] { 3, 17, 90 })
        {
            var ids = Enumerable.Range(1, 4 * tokens).ToArray();
            var outputs = model.RunDynamic([new DynamicTensor(model.Inputs[0].Name, ids, [4, tokens])]);
            outputs[0].Shape.Should().Equal(1, 111);
            outputs[0].Data.Sum().Should().BeApproximately(1f, 1e-4f); // softmax
        }
    }
}
