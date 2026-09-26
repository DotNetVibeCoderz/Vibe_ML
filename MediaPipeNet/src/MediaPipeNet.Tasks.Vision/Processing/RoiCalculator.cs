namespace MediaPipeNet.Tasks.Vision.Processing;

/// <summary>
/// Region-of-interest math ported from MediaPipe's <c>DetectionsToRectsCalculator</c>,
/// <c>AlignmentPointsRectsCalculator</c>, <c>RectTransformationCalculator</c> and
/// <c>HandLandmarksToRectCalculator</c>. All rectangles are in normalized image coordinates;
/// rotations are computed in pixel space.
/// </summary>
public static class RoiCalculator
{
    /// <summary>Rect from a detection's box, rotated so the keypoint vector start→end points at <paramref name="targetAngle"/>.</summary>
    public static NormalizedRect FromDetection(in RawDetection d, int imageWidth, int imageHeight, int startKeypoint, int endKeypoint, float targetAngle)
    {
        var (x0, y0) = d.Keypoint(startKeypoint);
        var (x1, y1) = d.Keypoint(endKeypoint);
        float rotation = Angles.ComputeRotation(x0 * imageWidth, y0 * imageHeight, x1 * imageWidth, y1 * imageHeight, targetAngle);
        return new NormalizedRect((d.XMin + d.XMax) / 2, (d.YMin + d.YMax) / 2, d.Width, d.Height, rotation);
    }

    /// <summary>
    /// Rect centered on (<paramref name="x0"/>, <paramref name="y0"/>) whose side is twice the distance to
    /// (<paramref name="x1"/>, <paramref name="y1"/>) (BlazePose alignment points).
    /// </summary>
    public static NormalizedRect FromAlignmentPoints(float x0, float y0, float x1, float y1, int imageWidth, int imageHeight, float targetAngle)
    {
        float px0 = x0 * imageWidth, py0 = y0 * imageHeight, px1 = x1 * imageWidth, py1 = y1 * imageHeight;
        float size = 2f * MathF.Sqrt((px1 - px0) * (px1 - px0) + (py1 - py0) * (py1 - py0));
        float rotation = Angles.ComputeRotation(px0, py0, px1, py1, targetAngle);
        return new NormalizedRect(x0, y0, size / imageWidth, size / imageHeight, rotation);
    }

    /// <summary>Scales, shifts (in the rect's rotated frame) and optionally squares a rect.</summary>
    public static NormalizedRect Transform(in NormalizedRect rect, int imageWidth, int imageHeight,
        float scaleX = 1, float scaleY = 1, float shiftX = 0, float shiftY = 0, bool squareLong = false)
    {
        float w = rect.Width, h = rect.Height, r = rect.Rotation;
        float cx = rect.XCenter, cy = rect.YCenter;
        if (r == 0)
        {
            cx += w * shiftX;
            cy += h * shiftY;
        }
        else
        {
            float xs = (imageWidth * w * shiftX * MathF.Cos(r) - imageHeight * h * shiftY * MathF.Sin(r)) / imageWidth;
            float ys = (imageWidth * w * shiftX * MathF.Sin(r) + imageHeight * h * shiftY * MathF.Cos(r)) / imageHeight;
            cx += xs;
            cy += ys;
        }
        if (squareLong)
        {
            float longSide = MathF.Max(w * imageWidth, h * imageHeight);
            w = longSide / imageWidth;
            h = longSide / imageHeight;
        }
        return new NormalizedRect(cx, cy, w * scaleX, h * scaleY, r);
    }

    /// <summary>
    /// Axis-aligned bounds of the landmarks, rotated so landmark start→end points at
    /// <paramref name="targetAngle"/> (face landmarks → next-frame ROI).
    /// </summary>
    public static NormalizedRect FromLandmarkBounds(IReadOnlyList<NormalizedLandmark> landmarks, int imageWidth, int imageHeight, int start, int end, float targetAngle)
    {
        float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
        foreach (var l in landmarks)
        {
            minX = MathF.Min(minX, l.X); maxX = MathF.Max(maxX, l.X);
            minY = MathF.Min(minY, l.Y); maxY = MathF.Max(maxY, l.Y);
        }
        float rotation = Angles.ComputeRotation(landmarks[start].X * imageWidth, landmarks[start].Y * imageHeight,
            landmarks[end].X * imageWidth, landmarks[end].Y * imageHeight, targetAngle);
        return new NormalizedRect((minX + maxX) / 2, (minY + maxY) / 2, maxX - minX, maxY - minY, rotation);
    }

    // Landmarks used by HandLandmarksToRectCalculator: wrist, thumb CMC..IP, and the MCP/PIP joints of each finger.
    private static readonly int[] s_handPartial = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];

    /// <summary>Next-frame hand ROI from 21 hand landmarks (MediaPipe's <c>HandLandmarksToRectCalculator</c>).</summary>
    public static NormalizedRect FromHandLandmarks(IReadOnlyList<NormalizedLandmark> lm, int imageWidth, int imageHeight)
    {
        float x0 = lm[0].X * imageWidth, y0 = lm[0].Y * imageHeight;
        float x1 = (lm[5].X + lm[13].X) / 2 * imageWidth, y1 = (lm[5].Y + lm[13].Y) / 2 * imageHeight;
        x1 = (x1 + lm[9].X * imageWidth) / 2;
        y1 = (y1 + lm[9].Y * imageHeight) / 2;
        float rotation = Angles.ComputeRotation(x0, y0, x1, y1, MathF.PI / 2);
        float reverse = Angles.NormalizeRadians(-rotation);

        float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
        foreach (int i in s_handPartial)
        {
            float x = lm[i].X * imageWidth, y = lm[i].Y * imageHeight;
            minX = MathF.Min(minX, x); maxX = MathF.Max(maxX, x);
            minY = MathF.Min(minY, y); maxY = MathF.Max(maxY, y);
        }
        float acx = (minX + maxX) / 2, acy = (minY + maxY) / 2;
        float pMinX = float.MaxValue, pMinY = float.MaxValue, pMaxX = float.MinValue, pMaxY = float.MinValue;
        float cos = MathF.Cos(reverse), sin = MathF.Sin(reverse);
        foreach (int i in s_handPartial)
        {
            float ox = lm[i].X * imageWidth - acx, oy = lm[i].Y * imageHeight - acy;
            float px = ox * cos - oy * sin, py = ox * sin + oy * cos;
            pMinX = MathF.Min(pMinX, px); pMaxX = MathF.Max(pMaxX, px);
            pMinY = MathF.Min(pMinY, py); pMaxY = MathF.Max(pMaxY, py);
        }
        float pcx = (pMinX + pMaxX) / 2, pcy = (pMinY + pMaxY) / 2;
        float cx = pcx * MathF.Cos(rotation) - pcy * MathF.Sin(rotation) + acx;
        float cy = pcx * MathF.Sin(rotation) + pcy * MathF.Cos(rotation) + acy;
        var rect = new NormalizedRect(cx / imageWidth, cy / imageHeight, (pMaxX - pMinX) / imageWidth, (pMaxY - pMinY) / imageHeight, rotation);
        return Transform(rect, imageWidth, imageHeight, 2.0f, 2.0f, 0, -0.1f, squareLong: true);
    }

    /// <summary>IoU of the axis-aligned bounds of two (possibly rotated) rects.</summary>
    public static float Overlap(in NormalizedRect a, in NormalizedRect b)
    {
        static RectF Bounds(in NormalizedRect r) => new(r.XCenter - r.Width / 2, r.YCenter - r.Height / 2, r.Width, r.Height);
        return RectF.IntersectionOverUnion(Bounds(a), Bounds(b));
    }
}
