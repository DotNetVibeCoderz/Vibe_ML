using MediaPipeNet.Framework;
using MediaPipeNet.Framework.Config;
using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;

namespace MediaPipeNet.Tasks.Vision.Graph;

/// <summary>
/// A graph node that runs a vision task on every <c>IMAGE</c> packet and emits the result on
/// <c>RESULT</c>. The task is created when the graph opens (in video mode, so tracking and smoothing
/// work across packets) and disposed when the graph closes.
/// </summary>
/// <typeparam name="TTask">Task type.</typeparam>
/// <typeparam name="TResult">Result type.</typeparam>
public sealed class VisionTaskNode<TTask, TResult> : ICalculatorNode where TTask : IDisposable
{
    private readonly Func<CancellationToken, Task<TTask>> _factory;
    private readonly Func<TTask, MPImage, long, TResult> _run;
    private TTask? _task;
    private long _lastMs = long.MinValue;

    /// <summary>Creates the node.</summary>
    /// <param name="factory">Creates the task (called once, when the graph starts).</param>
    /// <param name="run">Runs the task on one frame (frame, timestamp in ms).</param>
    public VisionTaskNode(Func<CancellationToken, Task<TTask>> factory, Func<TTask, MPImage, long, TResult> run)
    {
        _factory = factory ?? throw new ArgumentNullException(nameof(factory));
        _run = run ?? throw new ArgumentNullException(nameof(run));
    }

    /// <inheritdoc />
    public void GetContract(CalculatorContract contract) => contract.AddInput<MPImage>("IMAGE").AddOutput<TResult>("RESULT");

    /// <inheritdoc />
    public async ValueTask OpenAsync(CalculatorContext context, CancellationToken cancellationToken) =>
        _task = await _factory(cancellationToken).ConfigureAwait(false);

    /// <inheritdoc />
    public ValueTask ProcessAsync(CalculatorContext context, CancellationToken cancellationToken)
    {
        var image = context.GetInput<MPImage>("IMAGE");
        long ms = Math.Max(context.InputTimestamp.Value / 1000, _lastMs + 1);
        _lastMs = ms;
        var result = _run(_task!, image, ms);
        if (result is not null) context.Send("RESULT", result);
        return ValueTask.CompletedTask;
    }

    /// <inheritdoc />
    public ValueTask CloseAsync(CalculatorContext context, CancellationToken cancellationToken)
    {
        _task?.Dispose();
        return ValueTask.CompletedTask;
    }
}

/// <summary>
/// Runs an arbitrary ONNX image model inside a graph: converts the <c>IMAGE</c> packet to a tensor
/// (resize/letterbox/normalize), runs inference and emits all outputs as <c>float[][]</c> on <c>TENSORS</c>.
/// </summary>
/// <param name="modelPath">ONNX model file (NHWC float input).</param>
/// <param name="rangeMin">Normalized value of black.</param>
/// <param name="rangeMax">Normalized value of white.</param>
/// <param name="keepAspectRatio">Letterbox instead of stretching.</param>
/// <param name="inference">Execution options.</param>
public sealed class InferenceNode(string modelPath, float rangeMin = 0f, float rangeMax = 1f, bool keepAspectRatio = false, InferenceOptions? inference = null)
    : CalculatorNode, IDisposable
{
    private OnnxModel? _model;

    /// <inheritdoc />
    public override void GetContract(CalculatorContract contract) => contract.AddInput<MPImage>("IMAGE").AddOutput<float[][]>("TENSORS");

    /// <inheritdoc />
    protected override void Open(CalculatorContext context) => _model = OnnxModel.Load(modelPath, inference, context.Logger);

    /// <inheritdoc />
    protected override void Process(CalculatorContext context)
    {
        var image = context.GetInput<MPImage>("IMAGE");
        var spec = _model!.Inputs[0];
        using var ctx = _model.RentContext();
        ImageToTensor.Convert(image, new ImageToTensorOptions(spec.Shape[2], spec.Shape[1], rangeMin, rangeMax, keepAspectRatio), ctx.GetInput(0));
        ctx.Run();
        var outputs = new float[_model.Outputs.Count][];
        for (int i = 0; i < outputs.Length; i++) outputs[i] = ctx.GetOutput(i).ToArray();
        context.Send("TENSORS", outputs);
    }

    /// <inheritdoc />
    protected override void Close(CalculatorContext context) => Dispose();

    /// <inheritdoc />
    public void Dispose()
    {
        _model?.Dispose();
        _model = null;
    }
}

/// <summary>Registers the vision calculators in a <see cref="CalculatorRegistry"/>.</summary>
public static class VisionCalculators
{
    /// <summary>
    /// Registers <c>FaceDetectorCalculator</c>, <c>FaceLandmarkerCalculator</c>,
    /// <c>HandLandmarkerCalculator</c>, <c>GestureRecognizerCalculator</c>, <c>PoseLandmarkerCalculator</c>,
    /// <c>HolisticLandmarkerCalculator</c>, <c>ImageSegmenterCalculator</c>, <c>ObjectDetectorCalculator</c>
    /// and <c>ImageClassifierCalculator</c>. Each takes an <c>IMAGE</c> input and produces a <c>RESULT</c>
    /// output; common options (<c>min_detection_confidence</c>, <c>num_hands</c>, <c>num_faces</c>,
    /// <c>max_results</c>, <c>score_threshold</c>) are read from the node's <c>options</c> block.
    /// </summary>
    public static CalculatorRegistry AddVisionCalculators(this CalculatorRegistry registry, BaseOptions? baseOptions = null)
    {
        ArgumentNullException.ThrowIfNull(registry);
        var b = baseOptions ?? BaseOptions.Default;
        var video = RunningMode.Video;
        return registry
            .Register("FaceDetectorCalculator", c => new VisionTaskNode<FaceDetector, FaceDetectionResult>(
                ct => FaceDetector.CreateAsync(new() { BaseOptions = b, RunningMode = video, MinDetectionConfidence = c.GetOption("min_detection_confidence", 0.5f) }, ct),
                (t, i, ts) => t.DetectForVideo(i, ts)))
            .Register("FaceLandmarkerCalculator", c => new VisionTaskNode<FaceLandmarker, FaceLandmarkResult>(
                ct => FaceLandmarker.CreateAsync(new() { BaseOptions = b, RunningMode = video, NumFaces = c.GetOption("num_faces", 1), OutputFaceBlendshapes = c.GetOption("output_face_blendshapes", false) }, ct),
                (t, i, ts) => t.DetectForVideo(i, ts)))
            .Register("HandLandmarkerCalculator", c => new VisionTaskNode<HandLandmarker, HandLandmarkResult>(
                ct => HandLandmarker.CreateAsync(new() { BaseOptions = b, RunningMode = video, NumHands = c.GetOption("num_hands", 2) }, ct),
                (t, i, ts) => t.DetectForVideo(i, ts)))
            .Register("GestureRecognizerCalculator", c => new VisionTaskNode<GestureRecognizer, GestureRecognitionResult>(
                ct => GestureRecognizer.CreateAsync(new() { BaseOptions = b, RunningMode = video, Hands = new() { NumHands = c.GetOption("num_hands", 2) } }, ct),
                (t, i, ts) => t.RecognizeForVideo(i, ts)))
            .Register("PoseLandmarkerCalculator", c => new VisionTaskNode<PoseLandmarker, PoseLandmarkResult>(
                ct => PoseLandmarker.CreateAsync(new() { BaseOptions = b, RunningMode = video, NumPoses = c.GetOption("num_poses", 1) }, ct),
                (t, i, ts) => t.DetectForVideo(i, ts)))
            .Register("HolisticLandmarkerCalculator", _ => new VisionTaskNode<HolisticLandmarker, HolisticResult>(
                ct => HolisticLandmarker.CreateAsync(new() { BaseOptions = b, RunningMode = video }, ct),
                (t, i, ts) => t.DetectForVideo(i, ts)))
            .Register("ImageSegmenterCalculator", _ => new VisionTaskNode<ImageSegmenter, SegmentationResult>(
                ct => ImageSegmenter.CreateAsync(new() { BaseOptions = b, RunningMode = video }, ct),
                (t, i, ts) => t.SegmentForVideo(i, ts)))
            .Register("ObjectDetectorCalculator", c => new VisionTaskNode<ObjectDetector, ObjectDetectionResult>(
                ct => ObjectDetector.CreateAsync(new() { BaseOptions = b, RunningMode = video, ScoreThreshold = c.GetOption("score_threshold", 0.3f), MaxResults = c.GetOption("max_results", -1) }, ct),
                (t, i, ts) => t.DetectForVideo(i, ts)))
            .Register("ImageClassifierCalculator", c => new VisionTaskNode<ImageClassifier, ClassificationResult>(
                ct => ImageClassifier.CreateAsync(new() { BaseOptions = b, RunningMode = video, MaxResults = c.GetOption("max_results", 5) }, ct),
                (t, i, ts) => t.ClassifyForVideo(i, ts)));
    }
}
