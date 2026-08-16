using System.Collections.Specialized;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

namespace LocalGen.Desktop.Controls;

/// <summary>
/// A compact line plot of recent values, drawn directly rather than through a charting library.
/// </summary>
/// <remarks>
/// This is the readout strip's signature: a live trace of generation throughput that makes the
/// machine's work visible. It is deliberately unlabelled — the number beside it carries the
/// value, and the trace carries the shape.
/// </remarks>
public sealed class Sparkline : Control
{
    public static readonly StyledProperty<System.Collections.IEnumerable?> ValuesProperty =
        AvaloniaProperty.Register<Sparkline, System.Collections.IEnumerable?>(nameof(Values));

    public static readonly StyledProperty<IBrush?> StrokeProperty =
        AvaloniaProperty.Register<Sparkline, IBrush?>(nameof(Stroke));

    public static readonly StyledProperty<IBrush?> FillProperty =
        AvaloniaProperty.Register<Sparkline, IBrush?>(nameof(Fill));

    public static readonly StyledProperty<double> StrokeThicknessProperty =
        AvaloniaProperty.Register<Sparkline, double>(nameof(StrokeThickness), 1.5);

    private INotifyCollectionChanged? _observed;

    static Sparkline()
    {
        AffectsRender<Sparkline>(ValuesProperty, StrokeProperty, FillProperty, StrokeThicknessProperty);
        ValuesProperty.Changed.AddClassHandler<Sparkline>((sparkline, e) => sparkline.OnValuesChanged(e));
    }

    public System.Collections.IEnumerable? Values
    {
        get => GetValue(ValuesProperty);
        set => SetValue(ValuesProperty, value);
    }

    public IBrush? Stroke
    {
        get => GetValue(StrokeProperty);
        set => SetValue(StrokeProperty, value);
    }

    public IBrush? Fill
    {
        get => GetValue(FillProperty);
        set => SetValue(FillProperty, value);
    }

    public double StrokeThickness
    {
        get => GetValue(StrokeThicknessProperty);
        set => SetValue(StrokeThicknessProperty, value);
    }

    /// <summary>Re-subscribes so appended samples repaint without the caller raising anything.</summary>
    private void OnValuesChanged(AvaloniaPropertyChangedEventArgs e)
    {
        if (_observed is not null)
        {
            _observed.CollectionChanged -= OnCollectionChanged;
        }

        _observed = e.NewValue as INotifyCollectionChanged;

        if (_observed is not null)
        {
            _observed.CollectionChanged += OnCollectionChanged;
        }
    }

    private void OnCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e) =>
        Avalonia.Threading.Dispatcher.UIThread.Post(InvalidateVisual);

    public override void Render(DrawingContext context)
    {
        if (Values is null || Bounds.Width <= 1 || Bounds.Height <= 1)
        {
            return;
        }

        var samples = Values.Cast<object>()
            .Select(static value => Convert.ToDouble(value))
            .ToArray();

        // A single point has no shape to draw.
        if (samples.Length < 2)
        {
            return;
        }

        var maximum = samples.Max();
        var minimum = samples.Min();

        // A flat series would divide by zero; give it a nominal range so it renders as a level line.
        var range = maximum - minimum;
        if (range < double.Epsilon)
        {
            range = maximum > 0 ? maximum : 1;
            minimum = 0;
        }

        var width = Bounds.Width;
        var height = Bounds.Height;
        var step = width / (samples.Length - 1);

        // Inset by the stroke width so the line is not clipped at the top and bottom edges.
        var inset = StrokeThickness;
        var plotHeight = Math.Max(height - inset * 2, 1);

        var points = new Point[samples.Length];

        for (var i = 0; i < samples.Length; i++)
        {
            var normalized = (samples[i] - minimum) / range;
            points[i] = new Point(i * step, inset + (1 - normalized) * plotHeight);
        }

        var geometry = new StreamGeometry();

        using (var draw = geometry.Open())
        {
            draw.BeginFigure(points[0], isFilled: false);

            for (var i = 1; i < points.Length; i++)
            {
                draw.LineTo(points[i]);
            }

            draw.EndFigure(false);
        }

        if (Fill is not null)
        {
            // The fill closes the same path down to the baseline, giving the trace weight
            // without a second stroke competing with it.
            var area = new StreamGeometry();

            using (var draw = area.Open())
            {
                draw.BeginFigure(new Point(points[0].X, height), isFilled: true);

                foreach (var point in points)
                {
                    draw.LineTo(point);
                }

                draw.LineTo(new Point(points[^1].X, height));
                draw.EndFigure(true);
            }

            context.DrawGeometry(Fill, null, area);
        }

        if (Stroke is not null)
        {
            context.DrawGeometry(null, new Pen(Stroke, StrokeThickness, lineCap: PenLineCap.Round), geometry);
        }
    }
}
