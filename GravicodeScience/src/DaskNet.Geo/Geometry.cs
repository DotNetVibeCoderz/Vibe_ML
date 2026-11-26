using System.Globalization;
using System.Text;

namespace Gravicode.Science.DaskNet.Geo;

/// <summary>An axis-aligned bounding box, in the geometry's own x/y (longitude/latitude) units.</summary>
public readonly record struct Bounds(double MinX, double MinY, double MaxX, double MaxY)
{
    /// <summary>Whether the point lies inside or on the box.</summary>
    public bool Contains(double x, double y) => x >= MinX && x <= MaxX && y >= MinY && y <= MaxY;

    /// <summary>Whether the boxes overlap or touch.</summary>
    public bool Intersects(Bounds o) => MinX <= o.MaxX && o.MinX <= MaxX && MinY <= o.MaxY && o.MinY <= MaxY;
}

/// <summary>
/// A polygon or multipolygon parsed from WKT (<c>POLYGON ((x y, ...), (hole...))</c>,
/// <c>MULTIPOLYGON (((...)), ((...)))</c>), in planar x/y coordinates - for geographic data, x is
/// longitude and y latitude, as in WKT. Enough geometry for point-in-polygon joins; buffers and
/// overlays are out of scope (they need a full topology suite).
/// </summary>
public sealed class Polygon
{
    /// <summary>Each part's rings: the first is the shell, the rest are holes. Rings are closed
    /// (first point repeated last).</summary>
    public IReadOnlyList<IReadOnlyList<(double X, double Y)[]>> Parts { get; }

    /// <summary>The bounding box of every shell.</summary>
    public Bounds Bounds { get; }

    private Polygon(List<List<(double X, double Y)[]>> parts)
    {
        if (parts.Count == 0) throw new FormatException("A polygon needs at least one ring.");
        Parts = parts;
        double minX = double.PositiveInfinity, minY = double.PositiveInfinity, maxX = double.NegativeInfinity, maxY = double.NegativeInfinity;
        foreach (var part in parts)
            foreach (var (x, y) in part[0])
            {
                minX = Math.Min(minX, x); maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y); maxY = Math.Max(maxY, y);
            }
        Bounds = new Bounds(minX, minY, maxX, maxY);
    }

    /// <summary>
    /// Whether the point is inside or on the boundary. Boundary points count as inside, like
    /// <c>ST_Intersects</c>, so a point on a shared edge matches both neighbours. Inside a hole
    /// is outside.
    /// </summary>
    public bool Contains(double x, double y)
    {
        if (!Bounds.Contains(x, y)) return false;
        foreach (var part in Parts)
        {
            var state = Locate(part[0], x, y);
            if (state == 0) continue;
            if (state == 2) return true;
            var inHole = false;
            for (var h = 1; h < part.Count && !inHole; h++)
            {
                var s = Locate(part[h], x, y);
                if (s == 2) return true;          // a hole's edge is still the boundary
                inHole = s == 1;
            }
            if (!inHole) return true;
        }
        return false;
    }

    /// <summary>0 outside, 1 strictly inside, 2 on the ring.</summary>
    private static int Locate((double X, double Y)[] ring, double x, double y)
    {
        var inside = false;
        for (int i = 0, j = ring.Length - 1; i < ring.Length; j = i++)
        {
            var (xi, yi) = ring[i];
            var (xj, yj) = ring[j];
            var cross = (xj - xi) * (y - yi) - (yj - yi) * (x - xi);
            if (cross == 0 && x >= Math.Min(xi, xj) && x <= Math.Max(xi, xj) && y >= Math.Min(yi, yj) && y <= Math.Max(yi, yj))
                return 2;
            if (yi > y != yj > y && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
        }
        return inside ? 1 : 0;
    }

    /// <summary>Planar area (shells minus holes), in squared coordinate units.</summary>
    public double Area()
    {
        double total = 0;
        foreach (var part in Parts)
        {
            total += Math.Abs(Shoelace(part[0]));
            for (var h = 1; h < part.Count; h++) total -= Math.Abs(Shoelace(part[h]));
        }
        return total;
    }

    private static double Shoelace((double X, double Y)[] ring)
    {
        double s = 0;
        for (var i = 0; i < ring.Length - 1; i++) s += ring[i].X * ring[i + 1].Y - ring[i + 1].X * ring[i].Y;
        return s / 2;
    }

    /// <summary>The geometry as WKT.</summary>
    public string ToWkt()
    {
        static string N(double v) => v.ToString("R", CultureInfo.InvariantCulture);
        var sb = new StringBuilder(Parts.Count == 1 ? "POLYGON " : "MULTIPOLYGON (");
        for (var p = 0; p < Parts.Count; p++)
        {
            if (p > 0) sb.Append(", ");
            sb.Append('(');
            for (var r = 0; r < Parts[p].Count; r++)
            {
                if (r > 0) sb.Append(", ");
                sb.Append('(').AppendJoin(", ", Parts[p][r].Select(c => N(c.X) + " " + N(c.Y))).Append(')');
            }
            sb.Append(')');
        }
        if (Parts.Count > 1) sb.Append(')');
        return sb.ToString();
    }

    /// <inheritdoc/>
    public override string ToString() => ToWkt();

    /// <summary>A rectangle as a polygon.</summary>
    public static Polygon Box(double minX, double minY, double maxX, double maxY)
        => new([[[(minX, minY), (maxX, minY), (maxX, maxY), (minX, maxY), (minX, minY)]]]);

    /// <summary>Parses <c>POLYGON</c> or <c>MULTIPOLYGON</c> WKT (2-D; a Z or M ordinate is ignored).</summary>
    public static Polygon Parse(string wkt)
    {
        ArgumentNullException.ThrowIfNull(wkt);
        var p = new Reader(wkt);
        var word = p.Word().ToUpperInvariant();
        if (word is not ("POLYGON" or "MULTIPOLYGON"))
            throw new FormatException($"Expected POLYGON or MULTIPOLYGON, found '{word}'.");
        if (p.PeekWord()?.ToUpperInvariant() is "Z" or "M" or "ZM") p.Word();
        if (p.PeekWord()?.ToUpperInvariant() == "EMPTY") throw new FormatException("An empty polygon has nothing to contain.");
        var parts = new List<List<(double X, double Y)[]>>();
        if (word == "POLYGON") parts.Add(p.Polygon());
        else
        {
            p.Expect('(');
            do parts.Add(p.Polygon()); while (p.Comma());
            p.Expect(')');
        }
        p.End();
        return new Polygon(parts);
    }

    /// <summary>Parses, returning false instead of throwing.</summary>
    public static bool TryParse(string? wkt, out Polygon? polygon)
    {
        polygon = null;
        if (string.IsNullOrWhiteSpace(wkt)) return false;
        try { polygon = Parse(wkt); return true; }
        catch (FormatException) { return false; }
    }

    private sealed class Reader(string s)
    {
        private int _i;

        private void Skip() { while (_i < s.Length && char.IsWhiteSpace(s[_i])) _i++; }

        public string Word()
        {
            Skip();
            var start = _i;
            while (_i < s.Length && char.IsLetter(s[_i])) _i++;
            return s[start.._i];
        }

        public string? PeekWord()
        {
            var save = _i;
            var w = Word();
            _i = save;
            return w.Length == 0 ? null : w;
        }

        public void Expect(char c)
        {
            Skip();
            if (_i >= s.Length || s[_i] != c) throw new FormatException($"Expected '{c}' at position {_i} of the WKT.");
            _i++;
        }

        public bool Comma()
        {
            Skip();
            if (_i < s.Length && s[_i] == ',') { _i++; return true; }
            return false;
        }

        public void End()
        {
            Skip();
            if (_i != s.Length) throw new FormatException($"Unexpected text at position {_i} of the WKT.");
        }

        private double Number()
        {
            Skip();
            var start = _i;
            while (_i < s.Length && (char.IsDigit(s[_i]) || s[_i] is '-' or '+' or '.' or 'e' or 'E')) _i++;
            if (start == _i) throw new FormatException($"Expected a number at position {_i} of the WKT.");
            return double.Parse(s.AsSpan(start, _i - start), NumberStyles.Float, CultureInfo.InvariantCulture);
        }

        public List<(double X, double Y)[]> Polygon()
        {
            Expect('(');
            var rings = new List<(double X, double Y)[]>();
            do rings.Add(Ring()); while (Comma());
            Expect(')');
            return rings;
        }

        private (double X, double Y)[] Ring()
        {
            Expect('(');
            var points = new List<(double X, double Y)>();
            do
            {
                var x = Number();
                var y = Number();
                Skip();
                while (_i < s.Length && s[_i] != ',' && s[_i] != ')') { Number(); Skip(); }   // Z / M
                points.Add((x, y));
            } while (Comma());
            Expect(')');
            if (points[0] != points[^1]) points.Add(points[0]);
            if (points.Count < 4) throw new FormatException("A ring needs at least three distinct points.");
            return [.. points];
        }
    }
}
