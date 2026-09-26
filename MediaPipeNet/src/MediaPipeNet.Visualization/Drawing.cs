using MediaPipeNet.Tasks.Vision;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace MediaPipeNet.Visualization;

/// <summary>Colors and sizes used when drawing.</summary>
/// <param name="Color">Stroke/fill color.</param>
/// <param name="Thickness">Line thickness in pixels (0 = scale with image size).</param>
/// <param name="Radius">Point radius in pixels (0 = scale with image size).</param>
public sealed record DrawingStyle(Color Color, float Thickness = 0, float Radius = 0)
{
    /// <summary>Landmark points (warm white).</summary>
    public static DrawingStyle Points { get; } = new(Color.ParseHex("#FFF3E0"));

    /// <summary>Connections (MediaPipe teal).</summary>
    public static DrawingStyle Lines { get; } = new(Color.ParseHex("#00BFA5"));

    /// <summary>Bounding boxes (amber).</summary>
    public static DrawingStyle Boxes { get; } = new(Color.ParseHex("#FFB300"));

    internal float ThicknessFor(Image image) => Thickness > 0 ? Thickness : MathF.Max(1.5f, MathF.Max(image.Width, image.Height) / 400f);

    internal float RadiusFor(Image image) => Radius > 0 ? Radius : MathF.Max(2f, MathF.Max(image.Width, image.Height) / 260f);
}

/// <summary>Draws landmarks and their connections.</summary>
public static class LandmarkDrawer
{
    /// <summary>Draws landmarks (normalized coordinates) and optional connections.</summary>
    public static void Draw(Image<Rgba32> image, IReadOnlyList<NormalizedLandmark> landmarks,
        IReadOnlyList<(int From, int To)>? connections = null, DrawingStyle? points = null, DrawingStyle? lines = null,
        float minVisibility = 0.5f)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(landmarks);
        points ??= DrawingStyle.Points;
        lines ??= DrawingStyle.Lines;
        int w = image.Width, h = image.Height;
        bool Visible(NormalizedLandmark l) => (l.Visibility ?? 1f) >= minVisibility;
        float t = lines.ThicknessFor(image), r = points.RadiusFor(image);
        image.Mutate(ctx =>
        {
            if (connections is not null)
            {
                foreach (var (a, b) in connections)
                {
                    if (a >= landmarks.Count || b >= landmarks.Count || !Visible(landmarks[a]) || !Visible(landmarks[b])) continue;
                    ctx.DrawLine(lines.Color, t, new PointF(landmarks[a].X * w, landmarks[a].Y * h), new PointF(landmarks[b].X * w, landmarks[b].Y * h));
                }
            }
            if (points.Color.ToPixel<Rgba32>().A == 0) return;
            foreach (var l in landmarks)
            {
                if (!Visible(l)) continue;
                ctx.Fill(points.Color, new EllipsePolygon(l.X * w, l.Y * h, r));
            }
        });
    }

    /// <summary>Draws a hand skeleton.</summary>
    public static void DrawHand(Image<Rgba32> image, HandLandmarks hand, DrawingStyle? points = null, DrawingStyle? lines = null)
    {
        ArgumentNullException.ThrowIfNull(hand);
        Draw(image, hand.Landmarks, Connections.Hand, points, lines);
    }

    /// <summary>Draws a pose skeleton (landmarks with visibility below 0.5 are skipped).</summary>
    public static void DrawPose(Image<Rgba32> image, PoseLandmarks pose, DrawingStyle? points = null, DrawingStyle? lines = null)
    {
        ArgumentNullException.ThrowIfNull(pose);
        Draw(image, pose.Landmarks, Connections.Pose, points, lines);
    }

    /// <summary>Draws the face contours (and, optionally, every mesh point).</summary>
    public static void DrawFace(Image<Rgba32> image, FaceLandmarks face, bool drawAllPoints = false, DrawingStyle? lines = null)
    {
        ArgumentNullException.ThrowIfNull(face);
        var pointStyle = drawAllPoints ? DrawingStyle.Points with { Radius = 1f } : DrawingStyle.Points with { Color = Color.Transparent };
        Draw(image, face.Landmarks, Connections.FaceContours, pointStyle, lines);
    }
}

/// <summary>Draws detection boxes with labels and keypoints.</summary>
public static class BoundingBoxDrawer
{
    private static readonly Lazy<Font?> s_font = new(() =>
    {
        foreach (var name in new[] { "Segoe UI", "Arial", "DejaVu Sans", "Liberation Sans", "Helvetica" })
            if (SystemFonts.TryGet(name, out var family)) return family.CreateFont(14, FontStyle.Bold);
        return SystemFonts.Families.FirstOrDefault() is { Name: not null } f ? f.CreateFont(14, FontStyle.Bold) : null;
    });

    /// <summary>Draws one detection.</summary>
    public static void Draw(Image<Rgba32> image, Detection detection, DrawingStyle? style = null, bool drawLabel = true, bool drawKeypoints = true)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(detection);
        style ??= DrawingStyle.Boxes;
        var b = detection.BoundingBox;
        float t = style.ThicknessFor(image);
        image.Mutate(ctx =>
        {
            ctx.Draw(style.Color, t, new RectangularPolygon(b.X, b.Y, b.Width, b.Height));
            if (drawKeypoints)
                foreach (var k in detection.Keypoints)
                    ctx.Fill(DrawingStyle.Lines.Color, new EllipsePolygon(k.X * image.Width, k.Y * image.Height, style.RadiusFor(image)));
            if (drawLabel && s_font.Value is { } baseFont)
            {
                var font = new Font(baseFont, MathF.Max(12, image.Height / 45f));
                var c = detection.Categories.Count > 0 ? detection.Categories[0] : null;
                string text = c is null ? "" : $"{c.CategoryName ?? "face"} {c.Score:P0}";
                var size = TextMeasurer.MeasureSize(text, new TextOptions(font));
                float y = MathF.Max(0, b.Y - size.Height - 6);
                ctx.Fill(style.Color, new RectangularPolygon(b.X, y, size.Width + 10, size.Height + 6));
                ctx.DrawText(text, font, Color.Black, new PointF(b.X + 5, y + 3));
            }
        });
    }

    /// <summary>Draws several detections.</summary>
    public static void DrawAll(Image<Rgba32> image, IEnumerable<Detection> detections, DrawingStyle? style = null)
    {
        ArgumentNullException.ThrowIfNull(detections);
        foreach (var d in detections) Draw(image, d, style);
    }
}

/// <summary>Composites segmentation masks onto images.</summary>
public static class SegmentationMaskOverlay
{
    /// <summary>Tints the masked area with <paramref name="color"/> at <paramref name="opacity"/>.</summary>
    public static void Overlay(Image<Rgba32> image, SegmentationMask mask, Color color, float opacity = 0.5f)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(mask);
        var tint = color.ToPixel<Rgba32>();
        Composite(image, mask, (x, y, a, px) => Lerp(px, tint, a * opacity));
    }

    /// <summary>Replaces the background (mask ≈ 0) with a solid color.</summary>
    public static void ReplaceBackground(Image<Rgba32> image, SegmentationMask mask, Color background)
    {
        var bg = background.ToPixel<Rgba32>();
        Composite(image, mask, (x, y, a, px) => Lerp(bg, px, a));
    }

    /// <summary>Replaces the background with another image (resized to fit).</summary>
    public static void ReplaceBackground(Image<Rgba32> image, SegmentationMask mask, Image<Rgba32> background)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(background);
        using var bg = background.Clone(c => c.Resize(image.Width, image.Height));
        Composite(image, mask, (x, y, a, px) => Lerp(bg[x, y], px, a));
    }

    /// <summary>Blurs the background (portrait mode).</summary>
    public static void BlurBackground(Image<Rgba32> image, SegmentationMask mask, float sigma = 12f)
    {
        ArgumentNullException.ThrowIfNull(image);
        using var blurred = image.Clone(c => c.GaussianBlur(sigma));
        Composite(image, mask, (x, y, a, px) => Lerp(blurred[x, y], px, a));
    }

    private static void Composite(Image<Rgba32> image, SegmentationMask mask, Func<int, int, float, Rgba32, Rgba32> blend)
    {
        ArgumentNullException.ThrowIfNull(mask);
        if (mask.Width != image.Width || mask.Height != image.Height)
            throw new ArgumentException("Mask and image sizes differ.", nameof(mask));
        image.ProcessPixelRows(rows =>
        {
            for (int y = 0; y < rows.Height; y++)
            {
                var row = rows.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++) row[x] = blend(x, y, mask.Data[y * mask.Width + x], row[x]);
            }
        });
    }

    private static Rgba32 Lerp(Rgba32 a, Rgba32 b, float t)
    {
        t = Math.Clamp(t, 0f, 1f);
        return new Rgba32((byte)(a.R + (b.R - a.R) * t), (byte)(a.G + (b.G - a.G) * t), (byte)(a.B + (b.B - a.B) * t), 255);
    }
}

/// <summary>One-call rendering of any task result onto an image.</summary>
public static class ResultRenderer
{
    /// <summary>Draws face detections.</summary>
    public static void Render(Image<Rgba32> image, FaceDetectionResult result) => BoundingBoxDrawer.DrawAll(image, result.Detections);

    /// <summary>Draws object detections.</summary>
    public static void Render(Image<Rgba32> image, ObjectDetectionResult result) => BoundingBoxDrawer.DrawAll(image, result.Detections);

    /// <summary>Draws face meshes.</summary>
    public static void Render(Image<Rgba32> image, FaceLandmarkResult result)
    {
        foreach (var f in result.Faces) LandmarkDrawer.DrawFace(image, f, drawAllPoints: true);
    }

    /// <summary>Draws hand skeletons.</summary>
    public static void Render(Image<Rgba32> image, HandLandmarkResult result)
    {
        foreach (var h in result.Hands) LandmarkDrawer.DrawHand(image, h);
    }

    /// <summary>Draws hand skeletons with gesture boxes.</summary>
    public static void Render(Image<Rgba32> image, GestureRecognitionResult result)
    {
        foreach (var h in result.Hands)
        {
            LandmarkDrawer.DrawHand(image, h.Hand);
            var xs = h.Hand.Landmarks.Select(l => l.X * image.Width).ToArray();
            var ys = h.Hand.Landmarks.Select(l => l.Y * image.Height).ToArray();
            var box = RectF.FromLtrb(xs.Min(), ys.Min(), xs.Max(), ys.Max());
            BoundingBoxDrawer.Draw(image, new Detection(box, [h.TopGesture], []), drawKeypoints: false);
        }
    }

    /// <summary>Draws pose skeletons (and masks when present).</summary>
    public static void Render(Image<Rgba32> image, PoseLandmarkResult result)
    {
        foreach (var p in result.Poses)
        {
            if (p.SegmentationMask is { } m) SegmentationMaskOverlay.Overlay(image, m, Color.ParseHex("#7C4DFF"), 0.35f);
            LandmarkDrawer.DrawPose(image, p);
        }
    }

    /// <summary>Draws a holistic result.</summary>
    public static void Render(Image<Rgba32> image, HolisticResult result)
    {
        if (result.Pose is { } p) LandmarkDrawer.DrawPose(image, p);
        if (result.Face is { } f) LandmarkDrawer.DrawFace(image, f);
        if (result.LeftHand is { } l) LandmarkDrawer.DrawHand(image, l, lines: DrawingStyle.Lines with { Color = Color.ParseHex("#FF7043") });
        if (result.RightHand is { } r) LandmarkDrawer.DrawHand(image, r, lines: DrawingStyle.Lines with { Color = Color.ParseHex("#42A5F5") });
    }

    /// <summary>Draws a segmentation mask overlay.</summary>
    public static void Render(Image<Rgba32> image, SegmentationResult result) =>
        SegmentationMaskOverlay.Overlay(image, result.ConfidenceMask, Color.ParseHex("#7C4DFF"), 0.5f);
}
