using MediaPipeNet.Imaging;

namespace MediaPipeNet.Tasks.Vision.Processing;

/// <summary>
/// A detection in normalized coordinates (tensor space right after decoding, image space after
/// <see cref="MapToImage"/>). Keypoints are stored as interleaved (x, y) pairs.
/// </summary>
public struct RawDetection
{
    /// <summary>Left edge.</summary>
    public float XMin;
    /// <summary>Top edge.</summary>
    public float YMin;
    /// <summary>Right edge.</summary>
    public float XMax;
    /// <summary>Bottom edge.</summary>
    public float YMax;
    /// <summary>Score.</summary>
    public float Score;
    /// <summary>Class index.</summary>
    public int ClassId;
    /// <summary>Interleaved keypoints (x0, y0, x1, y1, ...), or null.</summary>
    public float[]? Keypoints;

    /// <summary>Box width.</summary>
    public readonly float Width => XMax - XMin;

    /// <summary>Box height.</summary>
    public readonly float Height => YMax - YMin;

    /// <summary>Number of keypoints.</summary>
    public readonly int KeypointCount => Keypoints is null ? 0 : Keypoints.Length / 2;

    /// <summary>Keypoint <paramref name="i"/>.</summary>
    public readonly (float X, float Y) Keypoint(int i) => (Keypoints![2 * i], Keypoints[2 * i + 1]);

    /// <summary>The box as a rectangle.</summary>
    public readonly RectF Box => RectF.FromLtrb(XMin, YMin, XMax, YMax);

    /// <summary>
    /// Maps the detection from tensor space to normalized image space (undoing letterbox, crop and
    /// rotation). A rotated box becomes the axis-aligned bounds of its mapped corners.
    /// </summary>
    public readonly RawDetection MapToImage(in TensorMapping mapping)
    {
        var (x0, y0) = mapping.TensorToImage(XMin, YMin);
        var (x1, y1) = mapping.TensorToImage(XMax, YMin);
        var (x2, y2) = mapping.TensorToImage(XMax, YMax);
        var (x3, y3) = mapping.TensorToImage(XMin, YMax);
        var r = this;
        r.XMin = MathF.Min(MathF.Min(x0, x1), MathF.Min(x2, x3));
        r.XMax = MathF.Max(MathF.Max(x0, x1), MathF.Max(x2, x3));
        r.YMin = MathF.Min(MathF.Min(y0, y1), MathF.Min(y2, y3));
        r.YMax = MathF.Max(MathF.Max(y0, y1), MathF.Max(y2, y3));
        if (Keypoints is not null)
        {
            r.Keypoints = new float[Keypoints.Length];
            for (int i = 0; i < Keypoints.Length; i += 2)
                (r.Keypoints[i], r.Keypoints[i + 1]) = mapping.TensorToImage(Keypoints[i], Keypoints[i + 1]);
        }
        return r;
    }

    /// <summary>Converts to a public <see cref="Detection"/> (pixel box, normalized keypoints).</summary>
    public readonly Detection ToDetection(int imageWidth, int imageHeight, string? label = null, IReadOnlyList<string>? keypointNames = null)
    {
        var box = RectF.FromLtrb(XMin * imageWidth, YMin * imageHeight, XMax * imageWidth, YMax * imageHeight);
        var kps = new NormalizedKeypoint[KeypointCount];
        for (int i = 0; i < kps.Length; i++)
            kps[i] = new NormalizedKeypoint(Keypoints![2 * i], Keypoints[2 * i + 1], keypointNames is not null && i < keypointNames.Count ? keypointNames[i] : null);
        return new Detection(box, [new Category(ClassId, Score, label)], kps);
    }
}

/// <summary>Options of <see cref="DetectionDecoder"/>, mirroring MediaPipe's <c>TensorsToDetectionsCalculatorOptions</c>.</summary>
public sealed record DetectionDecoderOptions
{
    /// <summary>Number of classes in the score tensor.</summary>
    public int NumClasses { get; init; } = 1;
    /// <summary>Values per box in the regressor tensor.</summary>
    public required int NumCoords { get; init; }
    /// <summary>Offset of the box within the per-anchor values.</summary>
    public int BoxCoordOffset { get; init; }
    /// <summary>Offset of the first keypoint.</summary>
    public int KeypointCoordOffset { get; init; } = 4;
    /// <summary>Number of keypoints.</summary>
    public int NumKeypoints { get; init; }
    /// <summary>Values per keypoint.</summary>
    public int NumValuesPerKeypoint { get; init; } = 2;
    /// <summary>X scale (usually the input width).</summary>
    public float XScale { get; init; } = 1;
    /// <summary>Y scale.</summary>
    public float YScale { get; init; } = 1;
    /// <summary>Width scale.</summary>
    public float WScale { get; init; } = 1;
    /// <summary>Height scale.</summary>
    public float HScale { get; init; } = 1;
    /// <summary>Box values are (x, y, w, h) instead of (y, x, h, w).</summary>
    public bool ReverseOutputOrder { get; init; } = true;
    /// <summary>Box sizes are log-encoded.</summary>
    public bool ApplyExponentialOnBoxSize { get; init; }
    /// <summary>Apply a sigmoid to raw scores.</summary>
    public bool SigmoidScore { get; init; } = true;
    /// <summary>Clip raw scores to ±this before the sigmoid (0 = no clipping).</summary>
    public float ScoreClippingThreshold { get; init; } = 100;
    /// <summary>Only the best class per anchor is considered.</summary>
    public float MinScoreThreshold { get; init; } = 0.5f;
}

/// <summary>Decodes raw SSD tensors into detections (MediaPipe's <c>TensorsToDetectionsCalculator</c>).</summary>
public static class DetectionDecoder
{
    /// <summary>
    /// Decodes <paramref name="boxes"/> (anchors × NumCoords) and <paramref name="scores"/>
    /// (anchors × NumClasses) into <paramref name="output"/>, keeping anchors above the score threshold.
    /// </summary>
    public static void Decode(ReadOnlySpan<float> boxes, ReadOnlySpan<float> scores, ReadOnlySpan<Anchor> anchors,
        DetectionDecoderOptions o, List<RawDetection> output, float? minScore = null, Func<int, bool>? classFilter = null)
    {
        ArgumentNullException.ThrowIfNull(o);
        ArgumentNullException.ThrowIfNull(output);
        float threshold = minScore ?? o.MinScoreThreshold;
        for (int i = 0; i < anchors.Length; i++)
        {
            float best = float.NegativeInfinity;
            int bestClass = -1;
            var row = scores.Slice(i * o.NumClasses, o.NumClasses);
            for (int c = 0; c < row.Length; c++)
            {
                if (classFilter is not null && !classFilter(c)) continue;
                if (row[c] > best) { best = row[c]; bestClass = c; }
            }
            if (bestClass < 0) continue;
            float score = best;
            if (o.SigmoidScore)
            {
                if (o.ScoreClippingThreshold > 0) score = Math.Clamp(score, -o.ScoreClippingThreshold, o.ScoreClippingThreshold);
                score = Sigmoid(score);
            }
            if (score < threshold) continue;

            var a = anchors[i];
            int b = i * o.NumCoords + o.BoxCoordOffset;
            float xc, yc, w, h;
            if (o.ReverseOutputOrder) { xc = boxes[b]; yc = boxes[b + 1]; w = boxes[b + 2]; h = boxes[b + 3]; }
            else { yc = boxes[b]; xc = boxes[b + 1]; h = boxes[b + 2]; w = boxes[b + 3]; }
            xc = xc / o.XScale * a.Width + a.XCenter;
            yc = yc / o.YScale * a.Height + a.YCenter;
            if (o.ApplyExponentialOnBoxSize)
            {
                h = MathF.Exp(h / o.HScale) * a.Height;
                w = MathF.Exp(w / o.WScale) * a.Width;
            }
            else
            {
                h = h / o.HScale * a.Height;
                w = w / o.WScale * a.Width;
            }
            var d = new RawDetection
            {
                XMin = xc - w / 2, YMin = yc - h / 2, XMax = xc + w / 2, YMax = yc + h / 2,
                Score = score, ClassId = bestClass,
            };
            if (o.NumKeypoints > 0)
            {
                d.Keypoints = new float[o.NumKeypoints * 2];
                for (int k = 0; k < o.NumKeypoints; k++)
                {
                    int off = i * o.NumCoords + o.KeypointCoordOffset + k * o.NumValuesPerKeypoint;
                    float kx, ky;
                    if (o.ReverseOutputOrder) { kx = boxes[off]; ky = boxes[off + 1]; }
                    else { ky = boxes[off]; kx = boxes[off + 1]; }
                    d.Keypoints[2 * k] = kx / o.XScale * a.Width + a.XCenter;
                    d.Keypoints[2 * k + 1] = ky / o.YScale * a.Height + a.YCenter;
                }
            }
            output.Add(d);
        }
    }

    /// <summary>The logistic function.</summary>
    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}

/// <summary>Non-maximum suppression (MediaPipe's <c>NonMaxSuppressionCalculator</c>).</summary>
public static class NonMaxSuppression
{
    /// <summary>
    /// Weighted NMS: overlapping detections are merged into a score-weighted average box (and
    /// keypoints), which stabilizes BlazeFace/BlazePalm/BlazePose outputs.
    /// </summary>
    public static List<RawDetection> Weighted(List<RawDetection> detections, float iouThreshold, int maxResults = -1)
    {
        ArgumentNullException.ThrowIfNull(detections);
        var remaining = detections.OrderByDescending(d => d.Score).ToList();
        var result = new List<RawDetection>();
        var candidates = new List<RawDetection>();
        var rest = new List<RawDetection>();
        while (remaining.Count > 0 && (maxResults < 0 || result.Count < maxResults))
        {
            var top = remaining[0];
            candidates.Clear();
            rest.Clear();
            foreach (var d in remaining)
            {
                if (RectF.IntersectionOverUnion(top.Box, d.Box) > iouThreshold) candidates.Add(d);
                else rest.Add(d);
            }
            var merged = top;
            if (candidates.Count > 1)
            {
                float total = 0, xmin = 0, ymin = 0, xmax = 0, ymax = 0;
                float[]? kps = top.Keypoints is null ? null : new float[top.Keypoints.Length];
                foreach (var c in candidates)
                {
                    total += c.Score;
                    xmin += c.XMin * c.Score; ymin += c.YMin * c.Score;
                    xmax += c.XMax * c.Score; ymax += c.YMax * c.Score;
                    if (kps is not null)
                        for (int k = 0; k < kps.Length; k++) kps[k] += c.Keypoints![k] * c.Score;
                }
                merged.XMin = xmin / total; merged.YMin = ymin / total;
                merged.XMax = xmax / total; merged.YMax = ymax / total;
                if (kps is not null)
                {
                    for (int k = 0; k < kps.Length; k++) kps[k] /= total;
                    merged.Keypoints = kps;
                }
            }
            result.Add(merged);
            (remaining, rest) = (rest, remaining);
        }
        return result;
    }

    /// <summary>Classic (hard) NMS, optionally per class.</summary>
    public static List<RawDetection> Hard(List<RawDetection> detections, float iouThreshold, int maxResults = -1, bool perClass = true)
    {
        ArgumentNullException.ThrowIfNull(detections);
        var sorted = detections.OrderByDescending(d => d.Score).ToList();
        var result = new List<RawDetection>();
        foreach (var d in sorted)
        {
            if (maxResults >= 0 && result.Count >= maxResults) break;
            bool suppressed = false;
            foreach (var kept in result)
            {
                if (perClass && kept.ClassId != d.ClassId) continue;
                if (RectF.IntersectionOverUnion(kept.Box, d.Box) > iouThreshold) { suppressed = true; break; }
            }
            if (!suppressed) result.Add(d);
        }
        return result;
    }
}
