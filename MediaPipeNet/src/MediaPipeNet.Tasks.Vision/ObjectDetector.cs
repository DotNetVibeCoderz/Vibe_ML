using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Tasks.Vision.Processing;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Options of <see cref="ObjectDetector"/>.</summary>
public sealed record ObjectDetectorOptions : VisionTaskOptions<ObjectDetectionResult>
{
    /// <summary>Minimum score for a detection to be reported. Default 0.3.</summary>
    public float ScoreThreshold { get; init; } = 0.3f;

    /// <summary>Maximum detections to return (-1 = all). Default -1.</summary>
    public int MaxResults { get; init; } = -1;

    /// <summary>IoU above which overlapping boxes (of any class) are suppressed. Default 0.5.</summary>
    public float NmsThreshold { get; init; } = 0.5f;

    /// <summary>If set, only these labels are reported (e.g. "person", "car").</summary>
    public IReadOnlySet<string>? CategoryAllowlist { get; init; }

    /// <summary>If set, these labels are never reported.</summary>
    public IReadOnlySet<string>? CategoryDenylist { get; init; }
}

/// <summary>
/// General object detection with EfficientDet-Lite0: bounding boxes, labels and scores for the
/// 80 COCO object classes (person, car, dog, cup, ...).
/// </summary>
public sealed class ObjectDetector : VisionTaskBase<ObjectDetectionResult>
{
    private const int InputSize = 320;
    private readonly OnnxModel _model;
    private readonly Anchor[] _anchors;
    private readonly int _boxesOutput, _scoresOutput, _numClasses;
    private readonly bool[] _classEnabled;
    private readonly DetectionDecoderOptions _decoder;

    private ObjectDetector(ObjectDetectorOptions options, OnnxModel model)
        : base(nameof(ObjectDetector), options.RunningMode, options.BaseOptions, options.ResultCallback, options.MaxInFlightFrames)
    {
        Options = options;
        _model = model;
        _anchors = SsdAnchors.GenerateEfficientDet(InputSize);
        for (int i = 0; i < model.Outputs.Count; i++)
        {
            if (model.Outputs[i].Shape[^1] == 4) _boxesOutput = i;
            else _scoresOutput = i;
        }
        _numClasses = model.Outputs[_scoresOutput].Shape[^1];
        var labels = Labels.Coco;
        _classEnabled = new bool[_numClasses];
        for (int c = 0; c < _numClasses; c++)
        {
            string label = c < labels.Count ? labels[c] : "???";
            _classEnabled[c] = label != "???"
                               && (options.CategoryAllowlist is null || options.CategoryAllowlist.Contains(label))
                               && (options.CategoryDenylist is null || !options.CategoryDenylist.Contains(label));
        }
        _decoder = new DetectionDecoderOptions
        {
            NumClasses = _numClasses, NumCoords = 4, ReverseOutputOrder = false, ApplyExponentialOnBoxSize = true,
            SigmoidScore = false, // the exported model already applies the sigmoid
        };
        CompleteInitialization();
    }

    /// <summary>The options the task was created with.</summary>
    public ObjectDetectorOptions Options { get; }

    /// <summary>Creates the task (resolving the model synchronously).</summary>
    public static ObjectDetector Create(ObjectDetectorOptions? options = null)
    {
        options ??= new ObjectDetectorOptions();
        return new ObjectDetector(options, ModelLoader.Load(options.BaseOptions, ModelCatalog.EfficientDetLite0));
    }

    /// <summary>Creates the task, downloading the model asynchronously when needed.</summary>
    public static async Task<ObjectDetector> CreateAsync(ObjectDetectorOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new ObjectDetectorOptions();
        return new ObjectDetector(options, await ModelLoader.LoadAsync(options.BaseOptions, ModelCatalog.EfficientDetLite0, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Detects objects in a still image.</summary>
    public ObjectDetectionResult Detect(MPImage image, ImageProcessingOptions? processingOptions = null) => RunImage(image, processingOptions);

    /// <summary>Detects objects in a still image on a worker thread.</summary>
    public Task<ObjectDetectionResult> DetectAsync(MPImage image, ImageProcessingOptions? processingOptions = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Detect(image, processingOptions), cancellationToken);

    /// <summary>Detects objects in a video frame.</summary>
    public ObjectDetectionResult DetectForVideo(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunVideo(image, timestampMs, processingOptions);

    /// <summary>Submits a live-stream frame. Returns false if it was dropped.</summary>
    public bool DetectLiveStream(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunLiveStream(image, timestampMs, processingOptions);

    /// <inheritdoc />
    protected override ObjectDetectionResult Process(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs)
    {
        var raw = new List<RawDetection>();
        TensorMapping mapping;
        using (var ctx = _model.RentContext())
        {
            mapping = ImageToTensor.Convert(image, options?.ToRoi() ?? NormalizedRect.FullImage,
                new ImageToTensorOptions(InputSize, InputSize, -1f, 1f, KeepAspectRatio: false), ctx.GetInput(0));
            ctx.Run();
            DetectionDecoder.Decode(ctx.GetOutput(_boxesOutput), ctx.GetOutput(_scoresOutput), _anchors, _decoder, raw,
                Options.ScoreThreshold, c => _classEnabled[c]);
        }
        var kept = NonMaxSuppression.Hard(raw, Options.NmsThreshold, Options.MaxResults, perClass: false);
        var labels = Labels.Coco;
        var detections = new Detection[kept.Count];
        for (int i = 0; i < kept.Count; i++)
        {
            var d = kept[i].MapToImage(mapping);
            d.XMin = Math.Clamp(d.XMin, 0, 1); d.YMin = Math.Clamp(d.YMin, 0, 1);
            d.XMax = Math.Clamp(d.XMax, 0, 1); d.YMax = Math.Clamp(d.YMax, 0, 1);
            detections[i] = d.ToDetection(image.Width, image.Height, labels[d.ClassId]);
        }
        return detections.Length == 0 ? ObjectDetectionResult.Empty : new ObjectDetectionResult(detections);
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (disposing) _model.Dispose();
    }
}
