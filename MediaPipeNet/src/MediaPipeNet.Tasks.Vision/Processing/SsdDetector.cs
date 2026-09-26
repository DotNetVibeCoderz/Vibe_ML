using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;

namespace MediaPipeNet.Tasks.Vision.Processing;

/// <summary>Configuration of a BlazeFace-style single-shot detector.</summary>
/// <param name="InputWidth">Model input width.</param>
/// <param name="InputHeight">Model input height.</param>
/// <param name="RangeMin">Normalized value of a black pixel.</param>
/// <param name="RangeMax">Normalized value of a white pixel.</param>
/// <param name="Anchors">Anchor generation options.</param>
/// <param name="Decoder">Tensor decoding options.</param>
/// <param name="NmsThreshold">IoU above which detections are merged.</param>
public sealed record SsdDetectorSpec(int InputWidth, int InputHeight, float RangeMin, float RangeMax,
    SsdAnchorOptions Anchors, DetectionDecoderOptions Decoder, float NmsThreshold = 0.3f)
{
    /// <summary>BlazeFace short range (face_detection_short_range.onnx).</summary>
    public static SsdDetectorSpec FaceShortRange { get; } = new(128, 128, -1f, 1f,
        new SsdAnchorOptions { InputWidth = 128, InputHeight = 128, Strides = [8, 16, 16, 16] },
        new DetectionDecoderOptions { NumCoords = 16, NumKeypoints = 6, XScale = 128, YScale = 128, WScale = 128, HScale = 128 });

    /// <summary>BlazePalm (palm_detection.onnx, 192×192).</summary>
    public static SsdDetectorSpec Palm { get; } = new(192, 192, 0f, 1f,
        new SsdAnchorOptions { InputWidth = 192, InputHeight = 192, Strides = [8, 16, 16, 16] },
        new DetectionDecoderOptions { NumCoords = 18, NumKeypoints = 7, XScale = 192, YScale = 192, WScale = 192, HScale = 192 });

    /// <summary>BlazePose detector (pose_detection.onnx, 224×224).</summary>
    public static SsdDetectorSpec Pose { get; } = new(224, 224, -1f, 1f,
        new SsdAnchorOptions { InputWidth = 224, InputHeight = 224, Strides = [8, 16, 32, 32, 32] },
        new DetectionDecoderOptions { NumCoords = 12, NumKeypoints = 4, XScale = 224, YScale = 224, WScale = 224, HScale = 224 });
}

/// <summary>
/// Runs a BlazeFace-family SSD model: letterboxed image-to-tensor, inference, anchor decoding,
/// weighted NMS and projection back to normalized image coordinates.
/// </summary>
public sealed class SsdDetector : IDisposable
{
    private readonly OnnxModel _model;
    private readonly Anchor[] _anchors;
    private readonly int _boxesOutput;
    private readonly int _scoresOutput;
    private readonly bool _ownsModel;

    /// <summary>Creates the detector over a loaded model.</summary>
    public SsdDetector(OnnxModel model, SsdDetectorSpec spec, bool ownsModel = true)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        Spec = spec ?? throw new ArgumentNullException(nameof(spec));
        _ownsModel = ownsModel;
        _anchors = SsdAnchors.Generate(spec.Anchors);
        _boxesOutput = -1;
        for (int i = 0; i < model.Outputs.Count; i++)
        {
            if (model.Outputs[i].Shape[^1] == spec.Decoder.NumCoords) _boxesOutput = i;
            else _scoresOutput = i;
        }
        if (_boxesOutput < 0) throw new MediaPipeException($"Model '{model.Name}' has no output with {spec.Decoder.NumCoords} values per anchor.");
        if (model.Outputs[_boxesOutput].ElementCount != _anchors.Length * spec.Decoder.NumCoords)
            throw new MediaPipeException($"Model '{model.Name}' produces {model.Outputs[_boxesOutput].ElementCount / spec.Decoder.NumCoords} anchors, expected {_anchors.Length}.");
    }

    /// <summary>The detector configuration.</summary>
    public SsdDetectorSpec Spec { get; }

    /// <summary>The underlying model.</summary>
    public OnnxModel Model => _model;

    /// <summary>Detects objects inside <paramref name="roi"/>; results are in normalized image coordinates.</summary>
    public List<RawDetection> Detect(MPImage image, in NormalizedRect roi, float minScore, int maxResults = -1)
    {
        using var ctx = _model.RentContext();
        var mapping = ImageToTensor.Convert(image, roi,
            new ImageToTensorOptions(Spec.InputWidth, Spec.InputHeight, Spec.RangeMin, Spec.RangeMax, KeepAspectRatio: true),
            ctx.GetInput(0));
        ctx.Run();
        var raw = new List<RawDetection>();
        DetectionDecoder.Decode(ctx.GetOutput(_boxesOutput), ctx.GetOutput(_scoresOutput), _anchors, Spec.Decoder, raw, minScore);
        var kept = NonMaxSuppression.Weighted(raw, Spec.NmsThreshold, maxResults);
        for (int i = 0; i < kept.Count; i++) kept[i] = kept[i].MapToImage(mapping);
        return kept;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_ownsModel) _model.Dispose();
    }
}
