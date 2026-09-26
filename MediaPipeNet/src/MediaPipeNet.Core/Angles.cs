namespace MediaPipeNet;

/// <summary>Angle helpers used by region-of-interest calculations.</summary>
public static class Angles
{
    /// <summary>Normalizes an angle in radians to the range [-π, π).</summary>
    public static float NormalizeRadians(float angle) =>
        angle - 2f * MathF.PI * MathF.Floor((angle + MathF.PI) / (2f * MathF.PI));

    /// <summary>Converts degrees to radians.</summary>
    public static float DegreesToRadians(float degrees) => degrees * (MathF.PI / 180f);

    /// <summary>Converts radians to degrees.</summary>
    public static float RadiansToDegrees(float radians) => radians * (180f / MathF.PI);

    /// <summary>
    /// The rotation (radians) that brings the vector from (x0, y0) to (x1, y1) — in pixel
    /// coordinates — to <paramref name="targetAngle"/>, exactly as MediaPipe's
    /// <c>DetectionsToRectsCalculator</c> computes it.
    /// </summary>
    public static float ComputeRotation(float x0, float y0, float x1, float y1, float targetAngle) =>
        NormalizeRadians(targetAngle - MathF.Atan2(-(y1 - y0), x1 - x0));
}
