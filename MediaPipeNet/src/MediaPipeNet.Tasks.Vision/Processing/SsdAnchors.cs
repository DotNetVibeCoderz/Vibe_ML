namespace MediaPipeNet.Tasks.Vision.Processing;

/// <summary>An SSD anchor in normalized tensor coordinates.</summary>
/// <param name="XCenter">Center X.</param>
/// <param name="YCenter">Center Y.</param>
/// <param name="Width">Width.</param>
/// <param name="Height">Height.</param>
public readonly record struct Anchor(float XCenter, float YCenter, float Width, float Height);

/// <summary>Options of <see cref="SsdAnchors.Generate"/>, mirroring MediaPipe's <c>SsdAnchorsCalculatorOptions</c>.</summary>
public sealed record SsdAnchorOptions
{
    /// <summary>Model input width.</summary>
    public required int InputWidth { get; init; }
    /// <summary>Model input height.</summary>
    public required int InputHeight { get; init; }
    /// <summary>Smallest anchor scale.</summary>
    public float MinScale { get; init; } = 0.1484375f;
    /// <summary>Largest anchor scale.</summary>
    public float MaxScale { get; init; } = 0.75f;
    /// <summary>Anchor center offset inside a feature-map cell (X).</summary>
    public float AnchorOffsetX { get; init; } = 0.5f;
    /// <summary>Anchor center offset inside a feature-map cell (Y).</summary>
    public float AnchorOffsetY { get; init; } = 0.5f;
    /// <summary>Feature-map strides, one per layer.</summary>
    public required IReadOnlyList<int> Strides { get; init; }
    /// <summary>Aspect ratios per location.</summary>
    public IReadOnlyList<float> AspectRatios { get; init; } = [1.0f];
    /// <summary>Aspect ratio of the extra interpolated-scale anchor (â‰¤ 0 disables it).</summary>
    public float InterpolatedScaleAspectRatio { get; init; } = 1.0f;
    /// <summary>Emit unit-size anchors (the box regressor predicts absolute sizes).</summary>
    public bool FixedAnchorSize { get; init; } = true;
}

/// <summary>Generates SSD anchors exactly like MediaPipe's <c>SsdAnchorsCalculator</c>.</summary>
public static class SsdAnchors
{
    /// <summary>Generates the anchors.</summary>
    public static Anchor[] Generate(SsdAnchorOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var anchors = new List<Anchor>();
        int numLayers = options.Strides.Count;
        int layer = 0;
        while (layer < numLayers)
        {
            var heights = new List<float>();
            var widths = new List<float>();
            var ratios = new List<float>();
            var scales = new List<float>();
            int last = layer;
            while (last < numLayers && options.Strides[last] == options.Strides[layer])
            {
                float scale = Scale(options.MinScale, options.MaxScale, last, numLayers);
                foreach (var ar in options.AspectRatios)
                {
                    ratios.Add(ar);
                    scales.Add(scale);
                }
                if (options.InterpolatedScaleAspectRatio > 0)
                {
                    float next = last == numLayers - 1 ? 1.0f : Scale(options.MinScale, options.MaxScale, last + 1, numLayers);
                    scales.Add(MathF.Sqrt(scale * next));
                    ratios.Add(options.InterpolatedScaleAspectRatio);
                }
                last++;
            }
            for (int i = 0; i < ratios.Count; i++)
            {
                float sq = MathF.Sqrt(ratios[i]);
                heights.Add(scales[i] / sq);
                widths.Add(scales[i] * sq);
            }
            int stride = options.Strides[layer];
            int fmH = (int)MathF.Ceiling((float)options.InputHeight / stride);
            int fmW = (int)MathF.Ceiling((float)options.InputWidth / stride);
            for (int y = 0; y < fmH; y++)
            {
                for (int x = 0; x < fmW; x++)
                {
                    for (int a = 0; a < heights.Count; a++)
                    {
                        float xc = (x + options.AnchorOffsetX) / fmW;
                        float yc = (y + options.AnchorOffsetY) / fmH;
                        anchors.Add(options.FixedAnchorSize ? new Anchor(xc, yc, 1f, 1f) : new Anchor(xc, yc, widths[a], heights[a]));
                    }
                }
            }
            layer = last;
        }
        return [.. anchors];
    }

    /// <summary>
    /// Generates EfficientDet multi-level anchors (levels 3â€“7, 3 octave scales Ã— aspect ratios
    /// 1, 2, 0.5, anchor scale 3 — verified against MediaPipe's own detections) in normalized (cx, cy, w, h) form, as exported by TFLite Model Maker.
    /// </summary>
    public static Anchor[] GenerateEfficientDet(int inputSize, int minLevel = 3, int maxLevel = 7, int numScales = 3, float anchorScale = 3f)
    {
        float[] aspects = [1.0f, 2.0f, 0.5f];
        var anchors = new List<Anchor>();
        // Feature sizes: repeatedly ceil-halved from the input size.
        var feat = new List<int> { inputSize };
        for (int l = 1; l <= maxLevel; l++) feat.Add((feat[^1] - 1) / 2 + 1);
        for (int level = minLevel; level <= maxLevel; level++)
        {
            int fs = feat[level];
            float stride = (float)inputSize / fs;
            for (int y = 0; y < fs; y++)
            {
                for (int x = 0; x < fs; x++)
                {
                    float cy = stride / 2 + y * stride, cx = stride / 2 + x * stride;
                    for (int octave = 0; octave < numScales; octave++)
                    {
                        foreach (var aspect in aspects)
                        {
                            float baseSize = anchorScale * stride * MathF.Pow(2f, (float)octave / numScales);
                            float ax = MathF.Sqrt(aspect), ay = 1f / ax;
                            anchors.Add(new Anchor(cx / inputSize, cy / inputSize, baseSize * ax / inputSize, baseSize * ay / inputSize));
                        }
                    }
                }
            }
        }
        return [.. anchors];
    }

    private static float Scale(float min, float max, int index, int count) =>
        count == 1 ? (min + max) * 0.5f : min + (max - min) * index / (count - 1f);
}
