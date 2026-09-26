using Avalonia.Media;

namespace MediaPipeNet.Gallery.Controls;

/// <summary>A line in normalized image coordinates.</summary>
public readonly record struct OverlayLine(float X1, float Y1, float X2, float Y2, Color Color, double Thickness = 2);

/// <summary>A point in normalized image coordinates.</summary>
public readonly record struct OverlayPoint(float X, float Y, Color Color, double Radius = 3);

/// <summary>A labeled box (normalized). Four corners allow rotated regions of interest.</summary>
public sealed record OverlayBox((float X, float Y)[] Corners, string? Label, Color Color, bool Dashed = false)
{
    public static OverlayBox FromRect(RectF r, string? label, Color color, bool dashed = false) =>
        new([(r.X, r.Y), (r.Right, r.Y), (r.Right, r.Bottom), (r.X, r.Bottom)], label, color, dashed);
}

/// <summary>A single-channel mask covering the whole image.</summary>
public sealed record OverlayMask(float[] Data, int Width, int Height, Color Color, float Opacity = 0.55f);

/// <summary>Everything drawn on top of the stage image for one result.</summary>
public sealed class Overlay
{
    public static readonly Color Coral = Color.Parse("#FF5A36");
    public static readonly Color Mint = Color.Parse("#35E0B5");
    public static readonly Color Paper = Color.Parse("#FFF3E0");
    public static readonly Color Cobalt = Color.Parse("#6D8BFF");
    public static readonly Color Amber = Color.Parse("#FFC53D");
    public static readonly Color Violet = Color.Parse("#9D7BFF");

    public List<OverlayLine> Lines { get; } = [];
    public List<OverlayPoint> Points { get; } = [];
    public List<OverlayBox> Boxes { get; } = [];
    public OverlayMask? Mask { get; set; }

    /// <summary>Adds landmarks and their connections.</summary>
    public void AddSkeleton(IReadOnlyList<NormalizedLandmark> landmarks, IReadOnlyList<(int From, int To)> connections,
        Color line, Color point, double thickness = 2.2, double radius = 3.2, bool points = true, float minVisibility = 0.5f)
    {
        bool Visible(NormalizedLandmark l) => (l.Visibility ?? 1f) >= minVisibility;
        foreach (var (a, b) in connections)
        {
            if (a >= landmarks.Count || b >= landmarks.Count || !Visible(landmarks[a]) || !Visible(landmarks[b])) continue;
            Lines.Add(new OverlayLine(landmarks[a].X, landmarks[a].Y, landmarks[b].X, landmarks[b].Y, line, thickness));
        }
        if (!points) return;
        foreach (var l in landmarks)
            if (Visible(l)) Points.Add(new OverlayPoint(l.X, l.Y, point, radius));
    }

    /// <summary>Adds a rotated region of interest as a dashed quad.</summary>
    public void AddRoi(NormalizedRect roi, int imageWidth, int imageHeight, Color color) =>
        Boxes.Add(new OverlayBox(roi.GetCorners(imageWidth, imageHeight), null, color, Dashed: true));
}
