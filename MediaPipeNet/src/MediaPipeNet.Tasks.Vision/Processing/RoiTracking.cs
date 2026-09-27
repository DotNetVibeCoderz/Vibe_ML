namespace MediaPipeNet.Tasks.Vision.Processing;

/// <summary>Thresholds of <see cref="RoiTracking.Select"/> (MediaPipe's <c>RoiTrackingCalculatorOptions</c>).</summary>
/// <param name="RotationDegrees">Maximum rotation difference between the previous and the re-crop rect.</param>
/// <param name="Translation">Maximum center shift, as a fraction of the re-crop rect size.</param>
/// <param name="Scale">Maximum size change, as a fraction of the re-crop rect size.</param>
/// <param name="RecropRectMargin">Margin added to the re-crop rect when checking that every previous landmark lies inside it.</param>
public readonly record struct RoiTrackingRequirements(float RotationDegrees, float Translation, float Scale, float RecropRectMargin)
{
    /// <summary>Holistic hand tracking (40°, 0.2, 0.4, margin −0.1).</summary>
    public static RoiTrackingRequirements Hand => new(40f, 0.2f, 0.4f, -0.1f);

    /// <summary>Holistic face tracking (15°, 0.1, 0.3, margin −0.2).</summary>
    public static RoiTrackingRequirements Face => new(15f, 0.1f, 0.3f, -0.2f);
}

/// <summary>
/// Chooses between the ROI derived from the previous frame's landmarks and a freshly computed re-crop
/// ROI, ported from MediaPipe's <c>RoiTrackingCalculator</c>: the previous ROI is kept while it agrees
/// with the re-crop ROI (rotation, translation and scale within limits, previous landmarks inside the
/// re-crop ROI), which keeps the crop stable from frame to frame.
/// </summary>
public static class RoiTracking
{
    /// <summary>Returns <paramref name="previousRect"/> when tracking holds, otherwise <paramref name="recropRect"/>.</summary>
    public static NormalizedRect Select(IReadOnlyList<NormalizedLandmark>? previousLandmarks, NormalizedRect? previousRect,
        in NormalizedRect recropRect, int imageWidth, int imageHeight, in RoiTrackingRequirements requirements)
    {
        if (previousLandmarks is null || previousRect is not { } prev) return recropRect;
        bool keep = RectRequirementsSatisfied(prev, recropRect, imageWidth, imageHeight, requirements)
                    & LandmarksRequirementsSatisfied(previousLandmarks, recropRect, imageWidth, imageHeight, requirements.RecropRectMargin);
        return keep ? prev : recropRect;
    }

    private static bool RectRequirementsSatisfied(in NormalizedRect prev, in NormalizedRect recrop, int w, int h, in RoiTrackingRequirements req)
    {
        float rotation = -recrop.Rotation;
        float cos = MathF.Cos(rotation), sin = MathF.Sin(rotation);
        float prevX = prev.XCenter * w * cos - prev.YCenter * h * sin;
        float prevY = prev.XCenter * w * sin + prev.YCenter * h * cos;
        float recropX = recrop.XCenter * w * cos - recrop.YCenter * h * sin;
        float recropY = recrop.XCenter * w * sin + recrop.YCenter * h * cos;
        float recropW = recrop.Width * w, recropH = recrop.Height * h;

        float rotationDiff = (prev.Rotation - recrop.Rotation) / MathF.PI * 180f;
        if (rotationDiff > 180f) rotationDiff -= 360f;
        if (rotationDiff < -180f) rotationDiff += 360f;

        return MathF.Abs(rotationDiff) <= req.RotationDegrees
               && MathF.Abs(prevX - recropX) <= recropW * req.Translation
               && MathF.Abs(prevY - recropY) <= recropH * req.Translation
               && MathF.Abs(prev.Width * w - recropW) <= recropW * req.Scale
               && MathF.Abs(prev.Height * h - recropH) <= recropH * req.Scale;
    }

    private static bool LandmarksRequirementsSatisfied(IReadOnlyList<NormalizedLandmark> landmarks, in NormalizedRect recrop, int w, int h, float margin)
    {
        float rotation = -recrop.Rotation;
        float cos = MathF.Cos(rotation), sin = MathF.Sin(rotation);
        float rectX = recrop.XCenter * w * cos - recrop.YCenter * h * sin;
        float rectY = recrop.XCenter * w * sin + recrop.YCenter * h * cos;
        float halfW = recrop.Width * w * (1f + margin) * 0.5f, halfH = recrop.Height * h * (1f + margin) * 0.5f;
        foreach (var l in landmarks)
        {
            float x = l.X * w * cos - l.Y * h * sin;
            float y = l.X * w * sin + l.Y * h * cos;
            if (!(rectX - halfW < x && x < rectX + halfW && rectY - halfH < y && y < rectY + halfH)) return false;
        }
        return true;
    }
}
