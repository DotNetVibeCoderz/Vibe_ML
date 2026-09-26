using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using MediaPipeNet.Framework;

namespace MediaPipeNet.Gallery.Controls;

/// <summary>Draws a calculator graph left-to-right in topological layers, like a signal-flow schematic.</summary>
public sealed class GraphDiagram : Control
{
    private static readonly Typeface Mono = new(new FontFamily("avares://MediaPipeNet.Gallery/Assets/Fonts#JetBrains Mono"));
    private static readonly Typeface Label = new(new FontFamily("fonts:Inter#Inter"), FontStyle.Normal, FontWeight.SemiBold);
    private IReadOnlyList<(string From, string To, string Stream)> _edges = [];
    private Func<string, string>? _typeOf;
    private HashSet<string> _active = [];

    public void SetGraph(CalculatorGraph graph)
    {
        _edges = graph.GetEdges();
        _typeOf = graph.GetNodeType;
        InvalidateVisual();
    }

    /// <summary>Highlights nodes that produced output in the last run.</summary>
    public void SetActive(IEnumerable<string> nodes)
    {
        _active = [.. nodes];
        InvalidateVisual();
    }

    public override void Render(DrawingContext context)
    {
        var bounds = new Rect(Bounds.Size);
        context.DrawRectangle(new SolidColorBrush(Color.Parse("#12151B")), null, bounds, 10, 10);
        if (_edges.Count == 0) return;

        // Layer = longest path from any input.
        var nodes = _edges.SelectMany(e => new[] { e.From, e.To }).Distinct().ToList();
        var layer = nodes.ToDictionary(n => n, _ => 0);
        for (int pass = 0; pass < nodes.Count; pass++)
            foreach (var (from, to, _) in _edges)
                layer[to] = Math.Max(layer[to], layer[from] + 1);
        int layers = layer.Values.Max() + 1;
        var columns = Enumerable.Range(0, layers).Select(l => nodes.Where(n => layer[n] == l).ToList()).ToList();

        const double boxH = 46;
        double boxW = Math.Min(176, (bounds.Width - 40) / layers - 28);
        double colGap = (bounds.Width - 40 - boxW) / Math.Max(1, layers - 1);
        var pos = new Dictionary<string, Point>();
        for (int c = 0; c < layers; c++)
        {
            var col = columns[c];
            double rowGap = (bounds.Height - 20) / col.Count;
            for (int r = 0; r < col.Count; r++)
                pos[col[r]] = new Point(20 + c * colGap, 10 + rowGap * r + (rowGap - boxH) / 2);
        }

        var wire = new Pen(new SolidColorBrush(Color.Parse("#4B5566")), 1.6);
        var ink = new SolidColorBrush(Color.Parse("#AEB8C6"));
        foreach (var (from, to, stream) in _edges)
        {
            var a = pos[from] + new Point(boxW, boxH / 2);
            var b = pos[to] + new Point(0, boxH / 2);
            var geo = new StreamGeometry();
            using (var g = geo.Open())
            {
                g.BeginFigure(a, false);
                double mx = (a.X + b.X) / 2;
                g.CubicBezierTo(new Point(mx, a.Y), new Point(mx, b.Y), b);
            }
            context.DrawGeometry(null, wire, geo);
            context.DrawEllipse(new SolidColorBrush(Color.Parse("#35E0B5")), null, b, 3, 3);
            var ft = new FormattedText(stream, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, Mono, 10, ink);
            context.DrawText(ft, new Point((a.X + b.X) / 2 - ft.Width / 2, (a.Y + b.Y) / 2 - ft.Height - 2));
        }

        foreach (var (name, p) in pos)
        {
            bool io = name.StartsWith("input:", StringComparison.Ordinal) || name.StartsWith("output:", StringComparison.Ordinal);
            bool active = _active.Contains(name);
            var fill = new SolidColorBrush(Color.Parse(io ? "#1C222B" : active ? "#223066" : "#1E2530"));
            var border = new Pen(new SolidColorBrush(Color.Parse(io ? "#4B5566" : active ? "#6D8BFF" : "#3A4452")), active ? 1.8 : 1.2)
            {
                DashStyle = io ? new DashStyle([3, 3], 0) : null,
            };
            var rect = new Rect(p, new Size(boxW, boxH));
            context.DrawRectangle(fill, border, rect, 8, 8);
            string title = io ? name[(name.IndexOf(':') + 1)..] : name;
            string sub = io ? name[..name.IndexOf(':')] : _typeOf?.Invoke(name) ?? "";
            var t1 = new FormattedText(title, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, Label, 12.5, Brushes.White);
            var t2 = new FormattedText(sub, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, Mono, 10, ink);
            context.DrawText(t1, new Point(p.X + 12, p.Y + 8));
            context.DrawText(t2, new Point(p.X + 12, p.Y + 28));
        }
    }
}
