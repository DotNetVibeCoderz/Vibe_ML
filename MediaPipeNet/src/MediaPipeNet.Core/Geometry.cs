using System.Text.Json.Serialization;

namespace MediaPipeNet;

/// <summary>An axis-aligned rectangle with floating point coordinates (pixels unless stated otherwise).</summary>
/// <param name="X">Left edge.</param>
/// <param name="Y">Top edge.</param>
/// <param name="Width">Width.</param>
/// <param name="Height">Height.</param>
public readonly record struct RectF(float X, float Y, float Width, float Height)
{
    /// <summary>Right edge.</summary>
    [JsonIgnore] public float Right => X + Width;

    /// <summary>Bottom edge.</summary>
    [JsonIgnore] public float Bottom => Y + Height;

    /// <summary>Horizontal center.</summary>
    [JsonIgnore] public float CenterX => X + Width * 0.5f;

    /// <summary>Vertical center.</summary>
    [JsonIgnore] public float CenterY => Y + Height * 0.5f;

    /// <summary>Area of the rectangle (zero for degenerate rectangles).</summary>
    [JsonIgnore] public float Area => Width > 0 && Height > 0 ? Width * Height : 0f;

    /// <summary>True when the rectangle has no area.</summary>
    [JsonIgnore] public bool IsEmpty => Width <= 0 || Height <= 0;

    /// <summary>Creates a rectangle from its edges.</summary>
    public static RectF FromLtrb(float left, float top, float right, float bottom) => new(left, top, right - left, bottom - top);

    /// <summary>The intersection of two rectangles (empty when they do not overlap).</summary>
    public static RectF Intersect(in RectF a, in RectF b)
    {
        float l = MathF.Max(a.X, b.X), t = MathF.Max(a.Y, b.Y);
        float r = MathF.Min(a.Right, b.Right), btm = MathF.Min(a.Bottom, b.Bottom);
        return r > l && btm > t ? FromLtrb(l, t, r, btm) : default;
    }

    /// <summary>Intersection-over-union of two rectangles, in [0, 1].</summary>
    public static float IntersectionOverUnion(in RectF a, in RectF b)
    {
        float inter = Intersect(a, b).Area;
        if (inter <= 0) return 0f;
        float union = a.Area + b.Area - inter;
        return union > 0 ? inter / union : 0f;
    }

    /// <summary>Scales every coordinate, e.g. to convert normalized coordinates to pixels.</summary>
    public RectF Scale(float sx, float sy) => new(X * sx, Y * sy, Width * sx, Height * sy);

    /// <summary>Clamps the rectangle to <c>[0, width] x [0, height]</c>.</summary>
    public RectF Clamp(float width, float height)
    {
        float l = Math.Clamp(X, 0, width), t = Math.Clamp(Y, 0, height);
        float r = Math.Clamp(Right, 0, width), b = Math.Clamp(Bottom, 0, height);
        return FromLtrb(l, t, r, b);
    }
}

/// <summary>
/// A rotated rectangle in normalized image coordinates ([0, 1] relative to image width/height),
/// mirroring <c>mediapipe::NormalizedRect</c>. Used as a region of interest (ROI) for landmark models.
/// </summary>
/// <param name="XCenter">Center X, normalized by image width.</param>
/// <param name="YCenter">Center Y, normalized by image height.</param>
/// <param name="Width">Width, normalized by image width.</param>
/// <param name="Height">Height, normalized by image height.</param>
/// <param name="Rotation">Clockwise rotation in radians.</param>
public readonly record struct NormalizedRect(float XCenter, float YCenter, float Width, float Height, float Rotation = 0f)
{
    /// <summary>The rectangle covering the whole image with no rotation.</summary>
    public static NormalizedRect FullImage { get; } = new(0.5f, 0.5f, 1f, 1f);

    /// <summary>Converts the rectangle to pixel units (rotation is ignored).</summary>
    public RectF ToPixelBounds(int imageWidth, int imageHeight) =>
        new((XCenter - Width * 0.5f) * imageWidth, (YCenter - Height * 0.5f) * imageHeight, Width * imageWidth, Height * imageHeight);

    /// <summary>Returns the four corners in normalized coordinates, clockwise from top-left.</summary>
    public (float X, float Y)[] GetCorners(int imageWidth, int imageHeight)
    {
        float cos = MathF.Cos(Rotation), sin = MathF.Sin(Rotation);
        float hw = Width * imageWidth * 0.5f, hh = Height * imageHeight * 0.5f;
        float cx = XCenter * imageWidth, cy = YCenter * imageHeight;
        var corners = new (float X, float Y)[4];
        ReadOnlySpan<(float, float)> offsets = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)];
        for (int i = 0; i < 4; i++)
        {
            var (dx, dy) = offsets[i];
            corners[i] = ((cx + dx * cos - dy * sin) / imageWidth, (cy + dx * sin + dy * cos) / imageHeight);
        }
        return corners;
    }
}

/// <summary>
/// Padding (as fractions of the destination tensor) added when an image was letterboxed into a
/// model input to preserve its aspect ratio.
/// </summary>
/// <param name="Left">Left padding fraction.</param>
/// <param name="Top">Top padding fraction.</param>
/// <param name="Right">Right padding fraction.</param>
/// <param name="Bottom">Bottom padding fraction.</param>
public readonly record struct LetterboxPadding(float Left, float Top, float Right, float Bottom)
{
    /// <summary>No padding.</summary>
    public static LetterboxPadding None => default;

    /// <summary>Maps a normalized coordinate inside the padded tensor back to the unpadded content.</summary>
    public (float X, float Y) Remove(float x, float y)
    {
        float sx = 1f - Left - Right, sy = 1f - Top - Bottom;
        return ((x - Left) / sx, (y - Top) / sy);
    }

    /// <summary>Horizontal scale factor of the content inside the tensor.</summary>
    public float ContentWidth => 1f - Left - Right;

    /// <summary>Vertical scale factor of the content inside the tensor.</summary>
    public float ContentHeight => 1f - Top - Bottom;
}

/// <summary>Size of an image in pixels.</summary>
/// <param name="Width">Width in pixels.</param>
/// <param name="Height">Height in pixels.</param>
public readonly record struct ImageSize(int Width, int Height);
