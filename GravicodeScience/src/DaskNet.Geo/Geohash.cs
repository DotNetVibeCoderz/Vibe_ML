namespace Gravicode.Science.DaskNet.Geo;

/// <summary>
/// Geohashes: a base-32 string naming a latitude/longitude cell, where a shared prefix means a
/// shared enclosing cell. Sorting by geohash therefore keeps nearby points together, which is
/// what <see cref="GeoFrame.SpatialPartition"/> uses.
/// </summary>
public static class Geohash
{
    private const string Alphabet = "0123456789bcdefghjkmnpqrstuvwxyz";

    /// <summary>The geohash of a point, <paramref name="precision"/> characters long (1-12).</summary>
    public static string Encode(double latitude, double longitude, int precision = 7)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(precision, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(precision, 12);
        if (!(latitude is >= -90 and <= 90) || !(longitude is >= -180 and <= 180))
            throw new ArgumentOutOfRangeException(nameof(latitude), $"({latitude}, {longitude}) is not a latitude/longitude.");
        double latLo = -90, latHi = 90, lonLo = -180, lonHi = 180;
        var chars = new char[precision];
        var even = true;
        for (var c = 0; c < precision; c++)
        {
            var index = 0;
            for (var bit = 0; bit < 5; bit++, even = !even)
            {
                index <<= 1;
                if (even)
                {
                    var mid = (lonLo + lonHi) / 2;
                    if (longitude >= mid) { index |= 1; lonLo = mid; } else lonHi = mid;
                }
                else
                {
                    var mid = (latLo + latHi) / 2;
                    if (latitude >= mid) { index |= 1; latLo = mid; } else latHi = mid;
                }
            }
            chars[c] = Alphabet[index];
        }
        return new string(chars);
    }

    /// <summary>The cell a geohash names: its bounds and centre.</summary>
    public static GeohashCell Decode(string hash)
    {
        ArgumentException.ThrowIfNullOrEmpty(hash);
        double latLo = -90, latHi = 90, lonLo = -180, lonHi = 180;
        var even = true;
        foreach (var ch in hash.ToLowerInvariant())
        {
            var index = Alphabet.IndexOf(ch);
            if (index < 0) throw new FormatException($"'{ch}' is not a geohash character.");
            for (var bit = 4; bit >= 0; bit--, even = !even)
            {
                var on = (index >> bit & 1) == 1;
                if (even)
                {
                    var mid = (lonLo + lonHi) / 2;
                    if (on) lonLo = mid; else lonHi = mid;
                }
                else
                {
                    var mid = (latLo + latHi) / 2;
                    if (on) latLo = mid; else latHi = mid;
                }
            }
        }
        return new GeohashCell(latLo, latHi, lonLo, lonHi);
    }

    /// <summary>The eight cells around <paramref name="hash"/> (fewer at the poles), same precision.</summary>
    public static string[] Neighbors(string hash)
    {
        var cell = Decode(hash);
        var dLat = cell.LatitudeMax - cell.LatitudeMin;
        var dLon = cell.LongitudeMax - cell.LongitudeMin;
        var result = new List<string>();
        for (var i = -1; i <= 1; i++)
            for (var j = -1; j <= 1; j++)
            {
                if (i == 0 && j == 0) continue;
                var lat = cell.Latitude + i * dLat;
                if (lat is < -90 or > 90) continue;
                var lon = cell.Longitude + j * dLon;
                lon = lon > 180 ? lon - 360 : lon < -180 ? lon + 360 : lon;
                var h = Encode(lat, lon, hash.Length);
                if (!result.Contains(h)) result.Add(h);
            }
        return [.. result];
    }
}

/// <summary>A geohash cell.</summary>
public readonly record struct GeohashCell(double LatitudeMin, double LatitudeMax, double LongitudeMin, double LongitudeMax)
{
    /// <summary>The centre's latitude.</summary>
    public double Latitude => (LatitudeMin + LatitudeMax) / 2;

    /// <summary>The centre's longitude.</summary>
    public double Longitude => (LongitudeMin + LongitudeMax) / 2;
}

/// <summary>Great-circle distances on a spherical Earth.</summary>
public static class GeoDistance
{
    /// <summary>The mean Earth radius used throughout, in kilometres (IUGG).</summary>
    public const double EarthRadiusKm = 6371.0088;

    /// <summary>Haversine distance in kilometres.</summary>
    public static double Haversine(double lat1, double lon1, double lat2, double lon2)
    {
        const double rad = Math.PI / 180;
        var dLat = (lat2 - lat1) * rad;
        var dLon = (lon2 - lon1) * rad;
        var a = Math.Sin(dLat / 2) * Math.Sin(dLat / 2)
              + Math.Cos(lat1 * rad) * Math.Cos(lat2 * rad) * Math.Sin(dLon / 2) * Math.Sin(dLon / 2);
        return 2 * EarthRadiusKm * Math.Asin(Math.Min(1, Math.Sqrt(a)));
    }
}
