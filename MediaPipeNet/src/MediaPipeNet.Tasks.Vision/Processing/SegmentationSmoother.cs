namespace MediaPipeNet.Tasks.Vision.Processing;

/// <summary>
/// Temporal smoothing of probability masks, ported from MediaPipe's <c>SegmentationSmoothingCalculator</c>:
/// each pixel is blended with the previous frame in proportion to how uncertain the new value is
/// (a polynomial approximation of the binary entropy), so confident pixels react immediately while
/// flickering edges settle down.
/// </summary>
/// <param name="combineWithPreviousRatio">How much of the previous mask to blend in at maximum uncertainty (0..1).</param>
public sealed class SegmentationSmoother(float combineWithPreviousRatio = 0.7f)
{
    private const float C1 = 5.68842f, C2 = -0.748699f, C3 = -57.8051f, C4 = 291.309f, C5 = -624.717f;
    private float[]? _previous;

    /// <summary>The blend ratio.</summary>
    public float CombineWithPreviousRatio { get; } = Math.Clamp(combineWithPreviousRatio, 0f, 1f);

    /// <summary>
    /// Smooths <paramref name="mask"/> in place against the previous call's output (the first call, or a
    /// call with a different mask size, only stores the mask).
    /// </summary>
    public void Apply(Span<float> mask)
    {
        var prev = _previous;
        if (prev is not null && prev.Length == mask.Length && CombineWithPreviousRatio > 0)
        {
            float ratio = CombineWithPreviousRatio;
            for (int i = 0; i < mask.Length; i++)
            {
                float p = mask[i];
                float t = p - 0.5f, x = t * t;
                float uncertainty = 1f - MathF.Min(1f, x * (C1 + x * (C2 + x * (C3 + x * (C4 + x * C5)))));
                mask[i] = p + (prev[i] - p) * (uncertainty * ratio);
            }
        }
        if (prev is null || prev.Length != mask.Length) _previous = prev = new float[mask.Length];
        mask.CopyTo(prev);
    }

    /// <summary>Forgets the previous mask.</summary>
    public void Reset() => _previous = null;
}
