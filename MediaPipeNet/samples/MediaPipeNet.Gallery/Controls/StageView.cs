using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using MediaPipeNet.Gallery.Services;

namespace MediaPipeNet.Gallery.Controls;

/// <summary>
/// The viewfinder: a dark stage that fits the image, draws the model's overlay as crisp vectors,
/// frames the picture with registration marks and prints an instrument readout underneath.
/// </summary>
public sealed class StageView : Control
{
    public static readonly StyledProperty<Bitmap?> ImageProperty = AvaloniaProperty.Register<StageView, Bitmap?>(nameof(Image));
    public static readonly StyledProperty<Overlay?> OverlayProperty = AvaloniaProperty.Register<StageView, Overlay?>(nameof(Overlay));
    public static readonly StyledProperty<string?> ReadoutProperty = AvaloniaProperty.Register<StageView, string?>(nameof(Readout));
    public static readonly StyledProperty<string?> MessageProperty = AvaloniaProperty.Register<StageView, string?>(nameof(Message));

    private static readonly IBrush Stage = new SolidColorBrush(Color.Parse("#12151B"));
    private static readonly IBrush Grid = new SolidColorBrush(Color.Parse("#1A1F27"));
    private static readonly IBrush StageInk = new SolidColorBrush(Color.Parse("#AEB8C6"));
    private static readonly Pen MarkPen = new(new SolidColorBrush(Color.Parse("#AEB8C6")), 1.5);
    private static readonly Typeface Mono = new(new FontFamily("avares://MediaPipeNet.Gallery/Assets/Fonts#JetBrains Mono"));
    private static readonly Typeface LabelFace = new(new FontFamily("fonts:Inter#Inter"), FontStyle.Normal, FontWeight.SemiBold);

    private WriteableBitmap? _maskBitmap;
    private OverlayMask? _maskSource;

    static StageView()
    {
        AffectsRender<StageView>(ImageProperty, OverlayProperty, ReadoutProperty, MessageProperty);
    }

    public Bitmap? Image { get => GetValue(ImageProperty); set => SetValue(ImageProperty, value); }
    public Overlay? Overlay { get => GetValue(OverlayProperty); set => SetValue(OverlayProperty, value); }
    public string? Readout { get => GetValue(ReadoutProperty); set => SetValue(ReadoutProperty, value); }
    public string? Message { get => GetValue(MessageProperty); set => SetValue(MessageProperty, value); }

    public override void Render(DrawingContext context)
    {
        var bounds = new Rect(Bounds.Size);
        context.DrawRectangle(Stage, null, bounds);
        const double readoutHeight = 30;
        var area = bounds.Deflate(new Thickness(28, 28, 28, 28 + readoutHeight));
        DrawGrid(context, bounds);

        var image = Image;
        if (image is null || area.Width < 10 || area.Height < 10)
        {
            if (Message is { } msg) DrawCentered(context, msg, bounds);
            DrawReadout(context, bounds, readoutHeight);
            return;
        }

        // Fit the image inside the stage (uniform scale, centered).
        double scale = Math.Min(area.Width / image.Size.Width, area.Height / image.Size.Height);
        var size = new Size(image.Size.Width * scale, image.Size.Height * scale);
        var rect = new Rect(area.X + (area.Width - size.Width) / 2, area.Y + (area.Height - size.Height) / 2, size.Width, size.Height);
        context.DrawImage(image, new Rect(image.Size), rect);

        if (Overlay is { } overlay) DrawOverlay(context, overlay, rect);
        DrawRegistrationMarks(context, rect);
        if (Message is { } m) DrawBadge(context, m, rect);
        DrawReadout(context, bounds, readoutHeight);
    }

    private static void DrawGrid(DrawingContext context, Rect bounds)
    {
        // A faint dot lattice — the stage reads as a measuring surface, not a void.
        const double step = 24;
        for (double y = step; y < bounds.Height; y += step)
            for (double x = step; x < bounds.Width; x += step)
                context.DrawRectangle(Grid, null, new Rect(x, y, 1.2, 1.2));
    }

    private void DrawOverlay(DrawingContext context, Overlay overlay, Rect rect)
    {
        Point P(float x, float y) => new(rect.X + x * rect.Width, rect.Y + y * rect.Height);

        if (overlay.Mask is { } mask)
        {
            if (!ReferenceEquals(mask, _maskSource))
            {
                _maskBitmap?.Dispose();
                _maskBitmap = Bitmaps.MaskToBitmap(mask.Data, mask.Width, mask.Height, mask.Color.R, mask.Color.G, mask.Color.B, mask.Opacity);
                _maskSource = mask;
            }
            context.DrawImage(_maskBitmap!, new Rect(_maskBitmap!.Size), rect);
        }

        foreach (var box in overlay.Boxes)
        {
            var pen = new Pen(new SolidColorBrush(box.Color), box.Dashed ? 1.3 : 2.2) { DashStyle = box.Dashed ? new DashStyle([4, 3], 0) : null };
            var pts = box.Corners.Select(c => P(c.X, c.Y)).ToArray();
            var geo = new PolylineGeometry(pts.Append(pts[0]), false);
            context.DrawGeometry(null, pen, geo);
            if (box.Label is { } label)
            {
                var ft = new FormattedText(label, CultureInfo.CurrentCulture, FlowDirection.LeftToRight, LabelFace, 12.5, Brushes.Black);
                double lx = pts.Min(p => p.X), ly = pts.Min(p => p.Y) - ft.Height - 6;
                if (ly < rect.Y) ly = pts.Min(p => p.Y) + 2;
                context.DrawRectangle(new SolidColorBrush(box.Color), null, new Rect(lx - 1, ly, ft.Width + 12, ft.Height + 5), 4, 4);
                context.DrawText(ft, new Point(lx + 5, ly + 2.5));
            }
        }

        var penCache = new Dictionary<(Color, double), Pen>();
        foreach (var l in overlay.Lines)
        {
            if (!penCache.TryGetValue((l.Color, l.Thickness), out var pen))
                penCache[(l.Color, l.Thickness)] = pen = new Pen(new SolidColorBrush(l.Color), l.Thickness, lineCap: PenLineCap.Round);
            context.DrawLine(pen, P(l.X1, l.Y1), P(l.X2, l.Y2));
        }

        if (!AppSettings.Current.ShowLandmarkPoints) return;
        var brushCache = new Dictionary<Color, IBrush>();
        var outline = new Pen(new SolidColorBrush(Color.FromArgb(160, 18, 21, 27)), 1);
        foreach (var p in overlay.Points)
        {
            if (!brushCache.TryGetValue(p.Color, out var brush)) brushCache[p.Color] = brush = new SolidColorBrush(p.Color);
            context.DrawEllipse(brush, outline, P(p.X, p.Y), p.Radius, p.Radius);
        }
    }

    private static void DrawRegistrationMarks(DrawingContext context, Rect r)
    {
        const double len = 16, gap = 7;
        foreach (var (x, y, dx, dy) in new[] { (r.Left, r.Top, 1, 1), (r.Right, r.Top, -1, 1), (r.Right, r.Bottom, -1, -1), (r.Left, r.Bottom, 1, -1) })
        {
            double ox = x - dx * gap, oy = y - dy * gap;
            context.DrawLine(MarkPen, new Point(ox, oy), new Point(ox + dx * len, oy));
            context.DrawLine(MarkPen, new Point(ox, oy), new Point(ox, oy + dy * len));
        }
        // Center registration tick on each edge.
        context.DrawLine(MarkPen, new Point(r.Center.X, r.Top - gap - 5), new Point(r.Center.X, r.Top - gap + 1));
        context.DrawLine(MarkPen, new Point(r.Center.X, r.Bottom + gap - 1), new Point(r.Center.X, r.Bottom + gap + 5));
        context.DrawLine(MarkPen, new Point(r.Left - gap - 5, r.Center.Y), new Point(r.Left - gap + 1, r.Center.Y));
        context.DrawLine(MarkPen, new Point(r.Right + gap - 1, r.Center.Y), new Point(r.Right + gap + 5, r.Center.Y));
    }

    private void DrawReadout(DrawingContext context, Rect bounds, double height)
    {
        if (string.IsNullOrEmpty(Readout)) return;
        var ft = new FormattedText(Readout, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, Mono, 11.5, StageInk);
        context.DrawText(ft, new Point(28, bounds.Bottom - height + (height - ft.Height) / 2 - 6));
    }

    private static void DrawCentered(DrawingContext context, string text, Rect bounds)
    {
        var ft = new FormattedText(text, CultureInfo.CurrentCulture, FlowDirection.LeftToRight, LabelFace, 14, StageInk)
        {
            MaxTextWidth = Math.Max(100, bounds.Width - 80),
            TextAlignment = TextAlignment.Center,
        };
        context.DrawText(ft, new Point((bounds.Width - ft.Width) / 2, (bounds.Height - ft.Height) / 2));
    }

    private static void DrawBadge(DrawingContext context, string text, Rect rect)
    {
        var ft = new FormattedText(text, CultureInfo.CurrentCulture, FlowDirection.LeftToRight, LabelFace, 12.5, Brushes.White);
        var badge = new Rect(rect.X + 12, rect.Y + 12, ft.Width + 20, ft.Height + 10);
        context.DrawRectangle(new SolidColorBrush(Color.FromArgb(200, 18, 21, 27)), null, badge, 6, 6);
        context.DrawText(ft, new Point(badge.X + 10, badge.Y + 5));
    }
}
