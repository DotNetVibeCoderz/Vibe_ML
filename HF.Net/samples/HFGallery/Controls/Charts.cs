using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

namespace HFGallery.Controls;

/// <summary>One labelled magnitude, with an optional category for its colour.</summary>
/// <param name="Label">What the bar is.</param>
/// <param name="Value">Its magnitude, in whatever unit the chart's <c>Unit</c> names.</param>
/// <param name="Category">
/// Index into the categorical palette, or -1 to take the sequential ramp by rank.
/// </param>
public readonly record struct Datum(string Label, double Value, int Category = -1);

/// <summary>Shared drawing helpers and the palettes, resolved from the active theme.</summary>
internal static class Palette
{
    /// <summary>Looks a brush up in the merged theme dictionaries.</summary>
    /// <remarks>
    /// The variant has to be passed explicitly: the one-argument lookup searches the default
    /// variant only, so every token defined under <c>ThemeDictionaries</c> - which is all of the
    /// ink, the grid and the sequential ramp - silently resolves to the fallback grey instead.
    /// </remarks>
    internal static IBrush Resource(Control host, string key, string fallback = "#888888")
        => host.TryFindResource(key, host.ActualThemeVariant, out var value) && value is IBrush brush
            ? brush
            : new SolidColorBrush(Color.Parse(fallback));

    /// <summary>
    /// The categorical hue for a series, in fixed order.
    /// </summary>
    /// <remarks>
    /// A seventh series does not get a generated hue - it wraps to the muted ink, which reads as
    /// "other" rather than as a repeat of series one. Cycling the palette makes two different
    /// things the same colour, which is worse than admitting the chart ran out of slots.
    /// </remarks>
    internal static IBrush Categorical(Control host, int index)
        => index is >= 0 and < 6
            ? Resource(host, $"Cat{index + 1}")
            : Resource(host, "TextMutedBrush");

    /// <summary>The sequential step for a rank, darkest first.</summary>
    internal static IBrush Sequential(Control host, int rank, int total)
    {
        if (total <= 1) return Resource(host, "Seq4");

        // Ranks map onto the five steps from the strongest downwards, so the top bar is the most
        // saturated and the eye lands on it first.
        var step = 5 - (int)Math.Round((double)rank / Math.Max(1, total - 1) * 4);
        return Resource(host, $"Seq{Math.Clamp(step, 1, 5)}");
    }

    internal static FormattedText Text(Control host, string text, double size, IBrush brush, bool mono = false)
    {
        var family = host.TryFindResource(mono ? "MonoFont" : "UiFont", out var f) && f is FontFamily found
            ? found
            : FontFamily.Default;

        return new FormattedText(text, System.Globalization.CultureInfo.InvariantCulture,
            FlowDirection.LeftToRight, new Typeface(family), size, brush);
    }
}

/// <summary>
/// Horizontal bars: the form for comparing magnitudes across a handful of named things.
/// </summary>
/// <remarks>
/// Horizontal rather than vertical because the labels are words - "POSITIVE", an entity type, a
/// candidate token - and horizontal bars give a label as much room as it needs without rotating it
/// forty-five degrees.
/// </remarks>
public sealed class BarChart : Control
{
    public static readonly StyledProperty<IReadOnlyList<Datum>?> ItemsProperty =
        AvaloniaProperty.Register<BarChart, IReadOnlyList<Datum>?>(nameof(Items));

    public static readonly StyledProperty<string> UnitProperty =
        AvaloniaProperty.Register<BarChart, string>(nameof(Unit), "%");

    public static readonly StyledProperty<double> MaximumProperty =
        AvaloniaProperty.Register<BarChart, double>(nameof(Maximum), double.NaN);

    static BarChart()
    {
        AffectsRender<BarChart>(ItemsProperty, UnitProperty, MaximumProperty);
        AffectsMeasure<BarChart>(ItemsProperty);
    }

    /// <summary>The bars, in the order they should read.</summary>
    public IReadOnlyList<Datum>? Items
    {
        get => GetValue(ItemsProperty);
        set => SetValue(ItemsProperty, value);
    }

    /// <summary>Suffix for the value label - <c>%</c>, <c>ms</c>, or empty.</summary>
    public string Unit
    {
        get => GetValue(UnitProperty);
        set => SetValue(UnitProperty, value);
    }

    /// <summary>Upper bound of the scale. NaN takes the largest value present.</summary>
    public double Maximum
    {
        get => GetValue(MaximumProperty);
        set => SetValue(MaximumProperty, value);
    }

    private const double RowHeight = 30;
    private const double LabelWidth = 168;
    private const double ValueWidth = 66;

    protected override Size MeasureOverride(Size available)
        => new(available.Width, (Items?.Count ?? 0) * RowHeight);

    public override void Render(DrawingContext context)
    {
        var items = Items;
        if (items is null || items.Count == 0) return;

        var maximum = double.IsNaN(Maximum) ? items.Max(i => i.Value) : Maximum;
        if (maximum <= 0) maximum = 1;

        var trackLeft = LabelWidth;
        var trackWidth = Math.Max(20, Bounds.Width - LabelWidth - ValueWidth);

        var text = Palette.Resource(this, "TextBaseBrush");
        var muted = Palette.Resource(this, "TextMutedBrush");
        var grid = Palette.Resource(this, "GridBrush");

        for (var i = 0; i < items.Count; i++)
        {
            var item = items[i];
            var y = i * RowHeight;
            var mid = y + RowHeight / 2;

            // The track, so a short bar still reads against a scale rather than floating.
            context.DrawRectangle(grid, null, new RoundedRect(
                new Rect(trackLeft, mid - 5, trackWidth, 10), 5));

            var width = Math.Max(3, trackWidth * (item.Value / maximum));
            var fill = item.Category >= 0
                ? Palette.Categorical(this, item.Category)
                : Palette.Sequential(this, i, items.Count);

            // 4px rounded data-end, square against the baseline: the bar grows from the axis and
            // the rounding marks where the value stops, not where the track does.
            context.DrawRectangle(fill, null, new RoundedRect(
                new Rect(trackLeft, mid - 5, width, 10),
                new CornerRadius(2, 4, 4, 2)));

            var label = Palette.Text(this, item.Label, 12.5, text);
            label.MaxTextWidth = LabelWidth - 12;

            // One line, always: without the cap a long label wraps, grows past the 30px row and
            // overlaps the bar below it, which is how a chart starts lying about which value is
            // which.
            label.MaxLineCount = 1;
            label.Trimming = TextTrimming.CharacterEllipsis;
            context.DrawText(label, new Point(0, mid - label.Height / 2));

            // Direct-labelled, because six rows is few enough that a reader wants the number and
            // an axis would be more furniture than help.
            var value = Palette.Text(this, Format(item.Value), 12, muted, mono: true);
            context.DrawText(value, new Point(
                trackLeft + trackWidth + ValueWidth - value.Width - 4, mid - value.Height / 2));
        }
    }

    private string Format(double value) => Unit switch
    {
        "%" => $"{value:P1}",
        "" => $"{value:N0}",
        _ => $"{value:N2} {Unit}",
    };
}

/// <summary>A point in an embedding map, after projection to two dimensions.</summary>
/// <param name="X">Projected first component.</param>
/// <param name="Y">Projected second component.</param>
/// <param name="Label">What the point is.</param>
/// <param name="Category">Index into the categorical palette.</param>
public readonly record struct Point2(double X, double Y, string Label, int Category);

/// <summary>
/// A scatter of sentences projected into two dimensions.
/// </summary>
/// <remarks>
/// Every pair of categories can end up adjacent here, which is why the palette was validated with
/// all pairs rather than only neighbours - in a bar chart only the bars beside each other need to
/// be separable, in a scatter any two points can touch.
/// </remarks>
public sealed class ScatterChart : Control
{
    public static readonly StyledProperty<IReadOnlyList<Point2>?> ItemsProperty =
        AvaloniaProperty.Register<ScatterChart, IReadOnlyList<Point2>?>(nameof(Items));

    static ScatterChart() => AffectsRender<ScatterChart>(ItemsProperty);

    /// <summary>The projected points.</summary>
    public IReadOnlyList<Point2>? Items
    {
        get => GetValue(ItemsProperty);
        set => SetValue(ItemsProperty, value);
    }

    public override void Render(DrawingContext context)
    {
        var items = Items;
        if (items is null || items.Count == 0) return;

        const double Pad = 34;
        var width = Math.Max(40, Bounds.Width - Pad * 2);
        var height = Math.Max(40, Bounds.Height - Pad * 2);

        var minX = items.Min(p => p.X);
        var maxX = items.Max(p => p.X);
        var minY = items.Min(p => p.Y);
        var maxY = items.Max(p => p.Y);

        double SpanX(double v) => maxX - minX < 1e-9 ? 0.5 : (v - minX) / (maxX - minX);
        double SpanY(double v) => maxY - minY < 1e-9 ? 0.5 : (v - minY) / (maxY - minY);

        var grid = new Pen(Palette.Resource(this, "GridBrush"), 1);
        var surface = Palette.Resource(this, "PanelGround");
        var muted = Palette.Resource(this, "TextMutedBrush");

        for (var i = 0; i <= 4; i++)
        {
            var x = Pad + width * i / 4.0;
            var y = Pad + height * i / 4.0;
            context.DrawLine(grid, new Point(x, Pad), new Point(x, Pad + height));
            context.DrawLine(grid, new Point(Pad, y), new Point(Pad + width, y));
        }

        foreach (var point in items)
        {
            var centre = new Point(Pad + SpanX(point.X) * width, Pad + height - SpanY(point.Y) * height);
            var fill = Palette.Categorical(this, point.Category);

            // A 2px surface ring, so two overlapping points stay two points rather than one blob.
            context.DrawEllipse(fill, new Pen(surface, 2), centre, 7, 7);
        }

        // Labelled selectively: every point labelled is unreadable, so the extremes carry the
        // names and hovering is left to do the rest.
        foreach (var point in items.OrderByDescending(p => Math.Abs(p.X) + Math.Abs(p.Y)).Take(4))
        {
            var centre = new Point(Pad + SpanX(point.X) * width, Pad + height - SpanY(point.Y) * height);
            var label = Palette.Text(this, Shorten(point.Label), 11, muted);

            // Flipped to the left when it would run past the plot, rather than clamped: a clamped
            // label slides under its own point and ends up naming the wrong one.
            var x = centre.X + 11 + label.Width <= Pad + width
                ? centre.X + 11
                : centre.X - 11 - label.Width;

            context.DrawText(label, new Point(Math.Max(2, x), centre.Y - label.Height / 2));
        }
    }

    private static string Shorten(string text)
        => text.Length <= 26 ? text : text[..24] + "…";
}

/// <summary>One named series of points, for <see cref="LineChart"/>.</summary>
/// <param name="Name">The series name, shown in the legend.</param>
/// <param name="Values">Y values, evenly spaced along X.</param>
/// <param name="Category">Index into the categorical palette.</param>
public readonly record struct Series(string Name, IReadOnlyList<double> Values, int Category);

/// <summary>
/// Lines over an evenly spaced axis - the form for change across an ordered variable.
/// </summary>
public sealed class LineChart : Control
{
    public static readonly StyledProperty<IReadOnlyList<Series>?> SeriesProperty =
        AvaloniaProperty.Register<LineChart, IReadOnlyList<Series>?>(nameof(Series));

    public static readonly StyledProperty<string> XLabelProperty =
        AvaloniaProperty.Register<LineChart, string>(nameof(XLabel), "");

    static LineChart() => AffectsRender<LineChart>(SeriesProperty, XLabelProperty);

    /// <summary>The series to draw.</summary>
    public IReadOnlyList<Series>? Series
    {
        get => GetValue(SeriesProperty);
        set => SetValue(SeriesProperty, value);
    }

    /// <summary>What the horizontal axis counts.</summary>
    public string XLabel
    {
        get => GetValue(XLabelProperty);
        set => SetValue(XLabelProperty, value);
    }

    public override void Render(DrawingContext context)
    {
        var series = Series;
        if (series is null || series.Count == 0) return;

        const double Left = 46;
        const double Right = 14;
        const double Top = 14;
        const double Bottom = 40;

        var width = Math.Max(40, Bounds.Width - Left - Right);
        var height = Math.Max(40, Bounds.Height - Top - Bottom);

        var maximum = series.Max(s => s.Values.Count == 0 ? 0 : s.Values.Max());
        var minimum = series.Min(s => s.Values.Count == 0 ? 0 : s.Values.Min());
        if (maximum - minimum < 1e-12) maximum = minimum + 1;

        var grid = new Pen(Palette.Resource(this, "GridBrush"), 1);
        var axis = new Pen(Palette.Resource(this, "AxisBrush"), 1);
        var muted = Palette.Resource(this, "TextMutedBrush");

        for (var i = 0; i <= 4; i++)
        {
            var y = Top + height - height * i / 4.0;
            context.DrawLine(grid, new Point(Left, y), new Point(Left + width, y));

            var value = minimum + (maximum - minimum) * i / 4.0;
            var label = Palette.Text(this, $"{value:0.##}", 10.5, muted, mono: true);
            context.DrawText(label, new Point(Left - label.Width - 8, y - label.Height / 2));
        }

        context.DrawLine(axis, new Point(Left, Top + height), new Point(Left + width, Top + height));

        foreach (var one in series)
        {
            if (one.Values.Count < 2) continue;

            var geometry = new StreamGeometry();
            using (var draw = geometry.Open())
            {
                for (var i = 0; i < one.Values.Count; i++)
                {
                    var x = Left + width * i / (one.Values.Count - 1.0);
                    var y = Top + height - (one.Values[i] - minimum) / (maximum - minimum) * height;

                    if (i == 0) draw.BeginFigure(new Point(x, y), false);
                    else draw.LineTo(new Point(x, y));
                }

                draw.EndFigure(false);
            }

            context.DrawGeometry(null, new Pen(Palette.Categorical(this, one.Category), 2)
            {
                LineJoin = PenLineJoin.Round,
                LineCap = PenLineCap.Round,
            }, geometry);
        }

        // Legend: always present for two or more series, so identity is never colour alone.
        var legendX = Left + 4;
        var legendY = Top + height + 14;

        foreach (var one in series)
        {
            context.DrawRectangle(Palette.Categorical(this, one.Category), null,
                new RoundedRect(new Rect(legendX, legendY + 4, 14, 3), 1.5));

            var label = Palette.Text(this, one.Name, 11.5, muted);
            context.DrawText(label, new Point(legendX + 20, legendY));
            legendX += 20 + label.Width + 18;
        }

        if (XLabel.Length > 0)
        {
            var label = Palette.Text(this, XLabel, 10.5, Palette.Resource(this, "TextFaintBrush"));
            context.DrawText(label, new Point(Left + width - label.Width, legendY));
        }
    }
}

/// <summary>One rectangle in a treemap.</summary>
/// <param name="Label">What it is.</param>
/// <param name="Value">Its size.</param>
public readonly record struct Slice(string Label, double Value);

/// <summary>
/// A treemap: part-to-whole where the parts differ by orders of magnitude.
/// </summary>
/// <remarks>
/// Chosen over a pie for the same reason a pie is usually wrong - a checkpoint's largest tensor is
/// a thousand times its smallest, and angles at that ratio are unreadable while areas still are.
/// Sequential colour by rank, because the encoded quantity is magnitude, not identity.
/// </remarks>
public sealed class Treemap : Control
{
    public static readonly StyledProperty<IReadOnlyList<Slice>?> ItemsProperty =
        AvaloniaProperty.Register<Treemap, IReadOnlyList<Slice>?>(nameof(Items));

    static Treemap() => AffectsRender<Treemap>(ItemsProperty);

    /// <summary>The slices, largest first.</summary>
    public IReadOnlyList<Slice>? Items
    {
        get => GetValue(ItemsProperty);
        set => SetValue(ItemsProperty, value);
    }

    public override void Render(DrawingContext context)
    {
        var items = Items?.Where(i => i.Value > 0).OrderByDescending(i => i.Value).ToList();
        if (items is null || items.Count == 0) return;

        var surface = Palette.Resource(this, "PanelGround");
        var total = items.Sum(i => i.Value);

        // Squarified-ish: slice alternately along the longer edge, which keeps the rectangles
        // close enough to square that their areas stay comparable by eye.
        var rect = new Rect(0, 0, Bounds.Width, Bounds.Height);
        var remaining = total;

        for (var i = 0; i < items.Count; i++)
        {
            var item = items[i];
            var share = remaining <= 0 ? 0 : item.Value / remaining;

            Rect cell;
            if (rect.Width >= rect.Height)
            {
                var w = i == items.Count - 1 ? rect.Width : rect.Width * share;
                cell = new Rect(rect.X, rect.Y, w, rect.Height);
                rect = new Rect(rect.X + w, rect.Y, Math.Max(0, rect.Width - w), rect.Height);
            }
            else
            {
                var h = i == items.Count - 1 ? rect.Height : rect.Height * share;
                cell = new Rect(rect.X, rect.Y, rect.Width, h);
                rect = new Rect(rect.X, rect.Y + h, rect.Width, Math.Max(0, rect.Height - h));
            }

            remaining -= item.Value;

            // A 2px surface gap between fills, so adjacent cells stay separate shapes.
            var inner = cell.Deflate(1);
            if (inner.Width <= 2 || inner.Height <= 2) continue;

            context.DrawRectangle(Palette.Sequential(this, i, items.Count), null,
                new RoundedRect(inner, 3));

            if (inner.Width < 62 || inner.Height < 26) continue;

            // Labels go on the cells with room for them; the rest are left to the tooltip.
            var ink = i < items.Count / 2
                ? Palette.Resource(this, "AppGround")
                : Palette.Resource(this, "TextHighBrush");

            var label = Palette.Text(this, item.Label, 11, ink);
            label.MaxTextWidth = inner.Width - 12;
            label.Trimming = TextTrimming.CharacterEllipsis;
            context.DrawText(label, new Point(inner.X + 6, inner.Y + 5));

            var value = Palette.Text(this, $"{item.Value / (1024 * 1024.0):0.#} MB", 10.5, ink, mono: true);
            if (inner.Height > 42) context.DrawText(value, new Point(inner.X + 6, inner.Y + 21));
        }
    }
}

/// <summary>
/// A swatch-and-name key, in the same fixed order the categorical palette is assigned in.
/// </summary>
/// <remarks>
/// Present whenever two or more categories are on screen, even where the marks are also directly
/// labelled: direct labels go missing as soon as a mark is too small to hold one, and the legend is
/// what keeps identity from resting on colour alone.
/// </remarks>
public sealed class Legend : Control
{
    /// <summary>The category names, in palette order.</summary>
    public static readonly StyledProperty<IReadOnlyList<string>?> ItemsProperty =
        AvaloniaProperty.Register<Legend, IReadOnlyList<string>?>(nameof(Items));

    static Legend()
    {
        AffectsRender<Legend>(ItemsProperty);
        AffectsMeasure<Legend>(ItemsProperty);
    }

    /// <inheritdoc cref="ItemsProperty" />
    public IReadOnlyList<string>? Items
    {
        get => GetValue(ItemsProperty);
        set => SetValue(ItemsProperty, value);
    }

    protected override Size MeasureOverride(Size available)
        => new(available.Width, Items is { Count: > 0 } ? 18 : 0);

    public override void Render(DrawingContext context)
    {
        if (Items is not { Count: > 0 } items) return;

        var ink = Palette.Resource(this, "TextMutedBrush", "#8894A3");
        var x = 0.0;

        for (var i = 0; i < items.Count; i++)
        {
            var label = Palette.Text(this, items[i], 11, ink);
            if (x + 12 + label.Width > Bounds.Width) break;

            context.DrawRectangle(
                Palette.Categorical(this, i), null,
                new RoundedRect(new Rect(x, 4.5, 8, 8), 2));

            context.DrawText(label, new Point(x + 12, 2));
            x += 12 + label.Width + 16;
        }
    }
}
