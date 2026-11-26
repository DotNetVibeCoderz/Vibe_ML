using System.Globalization;
using NetTopologySuite.Geometries;

namespace Gravicode.Science.DaskNet.Geo;

/// <summary>
/// A coordinate reference system a WKT column can be in: WGS 84 longitude/latitude
/// (EPSG:4326), Web Mercator (EPSG:3857) or a WGS 84 UTM zone (EPSG:326xx north, 327xx south).
/// Geometry operations given one work in metres on the ground (see <see cref="GeometryFrame"/>),
/// and <see cref="GeometryFrame.ToCrs"/> converts between them.
/// </summary>
/// <remarks>
/// Everything is on the WGS 84 ellipsoid with a 6th-order Krüger series transverse Mercator - the
/// algorithm PROJ's <c>tmerc</c> uses - accurate to well under a millimetre within a UTM zone and
/// to millimetres thousands of kilometres from its central meridian. Other systems are refused
/// rather than approximated; convert with PROJ (pyproj, ProjNet) first.
/// </remarks>
public sealed class GeoCrs : IEquatable<GeoCrs>
{
    private enum Kind { Geographic, WebMercator, Utm }

    private readonly Kind _kind;
    private readonly int _zone;
    private readonly bool _north;

    private GeoCrs(int epsg, Kind kind, int zone = 0, bool north = true)
    {
        Epsg = epsg;
        _kind = kind;
        _zone = zone;
        _north = north;
    }

    /// <summary>WGS 84 longitude/latitude in degrees, x = longitude (EPSG:4326, as GeoJSON and
    /// geopandas order it).</summary>
    public static GeoCrs Wgs84 { get; } = new(4326, Kind.Geographic);

    /// <summary>Web Mercator (EPSG:3857), the projection of web map tiles. Its metres are only
    /// ground metres at the equator; operations given it still measure on the ground.</summary>
    public static GeoCrs WebMercator { get; } = new(3857, Kind.WebMercator);

    /// <summary>A WGS 84 UTM zone (1-60).</summary>
    public static GeoCrs Utm(int zone, bool north = true)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(zone, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(zone, 60);
        return new GeoCrs((north ? 32600 : 32700) + zone, Kind.Utm, zone, north);
    }

    /// <summary>The UTM zone a longitude/latitude falls in (Norway and Svalbard's exceptions ignored).</summary>
    public static GeoCrs UtmFor(double longitude, double latitude)
        => Utm(Math.Clamp((int)Math.Floor((Normalize(longitude) + 180) / 6) + 1, 1, 60), latitude >= 0);

    /// <summary>The system with this EPSG code: 4326, 3857, 32601-32660 or 32701-32760.</summary>
    public static GeoCrs FromEpsg(int code) => code switch
    {
        4326 => Wgs84,
        3857 or 900913 => WebMercator,
        >= 32601 and <= 32660 => Utm(code - 32600, north: true),
        >= 32701 and <= 32760 => Utm(code - 32700, north: false),
        _ => throw new NotSupportedException($"EPSG:{code} is not supported; supported are 4326, 3857 and the WGS 84 UTM zones (32601-32660, 32701-32760)."),
    };

    /// <summary>Parses <c>EPSG:4326</c> (or a bare code).</summary>
    public static GeoCrs Parse(string text)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(text);
        var code = text.Trim();
        if (code.StartsWith("EPSG:", StringComparison.OrdinalIgnoreCase)) code = code[5..];
        return int.TryParse(code, NumberStyles.None, CultureInfo.InvariantCulture, out var epsg)
            ? FromEpsg(epsg)
            : throw new FormatException($"'{text}' is not an EPSG code (EPSG:4326, say).");
    }

    /// <summary>The EPSG code.</summary>
    public int Epsg { get; }

    /// <summary>True for longitude/latitude.</summary>
    public bool IsGeographic => _kind == Kind.Geographic;

    /// <inheritdoc/>
    public override string ToString() => "EPSG:" + Epsg.ToString(CultureInfo.InvariantCulture);

    /// <inheritdoc/>
    public bool Equals(GeoCrs? other) => other is not null && other.Epsg == Epsg;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as GeoCrs);

    /// <inheritdoc/>
    public override int GetHashCode() => Epsg;

    private const double R = 6378137.0;                           // WGS 84 semi-major axis; Web Mercator's sphere

    /// <summary>Longitude/latitude (degrees) of a coordinate in this system.</summary>
    internal (double Lon, double Lat) ToGeographic(double x, double y)
    {
        switch (_kind)
        {
            case Kind.Geographic:
                if (!(Math.Abs(y) <= 90)) throw new ArgumentException($"Latitude {y} is outside [-90, 90]: these coordinates are not EPSG:4326 longitude/latitude.");
                return (x, y);
            case Kind.WebMercator:
                return (x / R * 180 / Math.PI, (Math.PI / 2 - 2 * Math.Atan(Math.Exp(-y / R))) * 180 / Math.PI);
            default:
                var (lat, lon) = TransverseMercator.Inverse((x - 500_000) / 0.9996, (y - (_north ? 0 : 10_000_000)) / 0.9996, CentralMeridian);
                return (lon, lat);
        }
    }

    /// <summary>A longitude/latitude (degrees) in this system.</summary>
    internal (double X, double Y) FromGeographic(double lon, double lat)
    {
        switch (_kind)
        {
            case Kind.Geographic:
                return (lon, lat);
            case Kind.WebMercator:
                return (lon * Math.PI / 180 * R, R * Math.Log(Math.Tan(Math.PI / 4 + lat * Math.PI / 360)));
            default:
                var (x, y) = TransverseMercator.Forward(lat, lon, CentralMeridian);
                return (500_000 + 0.9996 * x, (_north ? 0 : 10_000_000) + 0.9996 * y);
        }
    }

    private double CentralMeridian => _zone * 6 - 183;

    internal static double Normalize(double lon) => lon - 360 * Math.Floor((lon + 180) / 360);

    /// <summary>A copy of <paramref name="g"/> with every coordinate mapped.</summary>
    internal static Geometry Transform(Geometry g, Func<double, double, (double, double)> map)
    {
        var copy = g.Copy();
        copy.Apply(new Mapper(map));
        copy.GeometryChanged();
        return copy;
    }

    private sealed class Mapper(Func<double, double, (double, double)> map) : IEntireCoordinateSequenceFilter
    {
        public bool Done => false;
        public bool GeometryChanged => true;

        public void Filter(CoordinateSequence seq)
        {
            for (var i = 0; i < seq.Count; i++)
            {
                var (x, y) = map(seq.GetX(i), seq.GetY(i));
                seq.SetX(i, x);
                seq.SetY(i, y);
            }
        }
    }
}

/// <summary>
/// A geometry in a transverse Mercator centred on its own centroid (scale 1 there): the plane in
/// which a metre is a ground metre, to within (distance from the centroid / Earth radius)^2 / 2 -
/// a part in a million 9 km away, a part in ten thousand at 90 km.
/// </summary>
internal sealed class LocalProjection
{
    private readonly GeoCrs _crs;
    private readonly double _lon0;
    private readonly double _y0;

    public LocalProjection(Geometry g, GeoCrs crs, out Geometry projected)
    {
        _crs = crs;
        var geographic = crs.IsGeographic ? g : GeoCrs.Transform(g, (x, y) => crs.ToGeographic(x, y));
        var box = geographic.EnvelopeInternal;
        if (!(Math.Abs(box.MinY) <= 90 && Math.Abs(box.MaxY) <= 90))
            throw new ArgumentException($"Latitudes {box.MinY}..{box.MaxY} are outside [-90, 90]: these coordinates are not {crs} as declared.");
        var c = geographic.Centroid.Coordinate;
        _lon0 = c.X;
        _y0 = TransverseMercator.Forward(c.Y, c.X, _lon0).Y;
        projected = GeoCrs.Transform(geographic, (lon, lat) =>
        {
            var (x, y) = TransverseMercator.Forward(lat, lon, _lon0);
            return (x, y - _y0);
        });
    }

    /// <summary>Back from the local plane into the original system.</summary>
    public Geometry Back(Geometry local) => GeoCrs.Transform(local, (x, y) =>
    {
        var (lat, lon) = TransverseMercator.Inverse(x, y + _y0, _lon0);
        return _crs.FromGeographic(lon, lat);
    });
}

/// <summary>
/// Transverse Mercator on WGS 84 by Krüger's series to sixth order in n (Karney 2011, "Transverse
/// Mercator with an accuracy of a few nanometers"), with scale 1, no false origin, x east of the
/// central meridian and y north of the equator. Latitudes and longitudes are in degrees.
/// </summary>
internal static class TransverseMercator
{
    private const double A = 6378137.0;
    private const double F = 1 / 298.257223563;
    private static readonly double N = F / (2 - F);
    private static readonly double E = Math.Sqrt(F * (2 - F));
    private static readonly double Rectifying = A / (1 + N) * (1 + N * N / 4 + Math.Pow(N, 4) / 64 + Math.Pow(N, 6) / 256);

    private static readonly double[] Alpha =
    [
        N / 2 - 2 * N * N / 3 + 5 * Math.Pow(N, 3) / 16 + 41 * Math.Pow(N, 4) / 180 - 127 * Math.Pow(N, 5) / 288 + 7891 * Math.Pow(N, 6) / 37800,
        13 * N * N / 48 - 3 * Math.Pow(N, 3) / 5 + 557 * Math.Pow(N, 4) / 1440 + 281 * Math.Pow(N, 5) / 630 - 1983433 * Math.Pow(N, 6) / 1935360,
        61 * Math.Pow(N, 3) / 240 - 103 * Math.Pow(N, 4) / 140 + 15061 * Math.Pow(N, 5) / 26880 + 167603 * Math.Pow(N, 6) / 181440,
        49561 * Math.Pow(N, 4) / 161280 - 179 * Math.Pow(N, 5) / 168 + 6601661 * Math.Pow(N, 6) / 7257600,
        34729 * Math.Pow(N, 5) / 80640 - 3418889 * Math.Pow(N, 6) / 1995840,
        212378941 * Math.Pow(N, 6) / 319334400,
    ];

    private static readonly double[] Beta =
    [
        N / 2 - 2 * N * N / 3 + 37 * Math.Pow(N, 3) / 96 - Math.Pow(N, 4) / 360 - 81 * Math.Pow(N, 5) / 512 + 96199 * Math.Pow(N, 6) / 604800,
        N * N / 48 + Math.Pow(N, 3) / 15 - 437 * Math.Pow(N, 4) / 1440 + 46 * Math.Pow(N, 5) / 105 - 1118711 * Math.Pow(N, 6) / 3870720,
        17 * Math.Pow(N, 3) / 480 - 37 * Math.Pow(N, 4) / 840 - 209 * Math.Pow(N, 5) / 4480 + 5569 * Math.Pow(N, 6) / 90720,
        4397 * Math.Pow(N, 4) / 161280 - 11 * Math.Pow(N, 5) / 504 - 830251 * Math.Pow(N, 6) / 7257600,
        4583 * Math.Pow(N, 5) / 161280 - 108847 * Math.Pow(N, 6) / 3991680,
        20648693 * Math.Pow(N, 6) / 638668800,
    ];

    public static (double X, double Y) Forward(double latitude, double longitude, double centralMeridian)
    {
        var phi = latitude * Math.PI / 180;
        var lambda = GeoCrs.Normalize(longitude - centralMeridian) * Math.PI / 180;
        var sin = Math.Sin(phi);
        // Conformal latitude, as its tangent; at the poles the limit is exact.
        var t = Math.Abs(latitude) >= 90 ? Math.Sign(latitude) * double.MaxValue
            : Math.Sinh(Math.Atanh(sin) - E * Math.Atanh(E * sin));
        var xiP = Math.Atan2(t, Math.Cos(lambda));
        var etaP = Math.Atanh(Math.Sin(lambda) / Math.Sqrt(1 + t * t));
        double xi = xiP, eta = etaP;
        for (var j = 1; j <= 6; j++)
        {
            xi += Alpha[j - 1] * Math.Sin(2 * j * xiP) * Math.Cosh(2 * j * etaP);
            eta += Alpha[j - 1] * Math.Cos(2 * j * xiP) * Math.Sinh(2 * j * etaP);
        }
        return (Rectifying * eta, Rectifying * xi);
    }

    public static (double Lat, double Lon) Inverse(double x, double y, double centralMeridian)
    {
        var xi = y / Rectifying;
        var eta = x / Rectifying;
        double xiP = xi, etaP = eta;
        for (var j = 1; j <= 6; j++)
        {
            xiP -= Beta[j - 1] * Math.Sin(2 * j * xi) * Math.Cosh(2 * j * eta);
            etaP -= Beta[j - 1] * Math.Cos(2 * j * xi) * Math.Sinh(2 * j * eta);
        }
        var lambda = Math.Atan2(Math.Sinh(etaP), Math.Cos(xiP));
        // tan of the conformal latitude, then Newton's method for tan(phi) (Karney 2011, eq. 19-21).
        var tauP = Math.Sin(xiP) / Math.Sqrt(Math.Sinh(etaP) * Math.Sinh(etaP) + Math.Cos(xiP) * Math.Cos(xiP));
        var e2m = 1 - E * E;
        var tau = tauP / e2m;
        for (var i = 0; i < 6; i++)
        {
            var sigma = Math.Sinh(E * Math.Atanh(E * tau / Math.Sqrt(1 + tau * tau)));
            var taui = tau * Math.Sqrt(1 + sigma * sigma) - sigma * Math.Sqrt(1 + tau * tau);
            var delta = (tauP - taui) / Math.Sqrt(1 + taui * taui) * (1 + e2m * tau * tau) / (e2m * Math.Sqrt(1 + tau * tau));
            tau += delta;
            if (Math.Abs(delta) < 1e-15 * Math.Max(1, Math.Abs(tau))) break;
        }
        return (Math.Atan(tau) * 180 / Math.PI, GeoCrs.Normalize(centralMeridian + lambda * 180 / Math.PI));
    }
}
