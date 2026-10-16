using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Spatial;

/// <summary>
/// Converts the point-set shapes the public API accepts into one contiguous row-major buffer.
/// </summary>
/// <remarks>
/// Every spatial kernel walks raw <c>double[]</c> rows. The <see cref="NdArray"/> indexer does
/// stride maths and bounds checks per element, which the repository measured at ~4x in KMeans'
/// inner loop; copying once up front is cheaper than paying that on every distance.
/// </remarks>
internal static class SpatialInput
{
    /// <summary>Copies an (n, d) array (or a rank-1 array as n points in 1-D) into a fresh buffer.</summary>
    public static double[] Copy(NdArray x, out int n, out int d, string name = "x")
    {
        ArgumentNullException.ThrowIfNull(x, name);
        if (x.Rank == 1) { n = x.Shape[0]; d = 1; }
        else if (x.Rank == 2) { n = x.Shape[0]; d = x.Shape[1]; }
        else throw new ArgumentException($"{name} must be a 1-D or 2-D array of points, got rank {x.Rank}.", name);
        var buffer = new double[n * d];
        x.CopyTo(buffer);
        return buffer;
    }

    /// <summary>Copies a rectangular array of points into a fresh buffer.</summary>
    public static double[] Copy(double[,] x, out int n, out int d)
    {
        ArgumentNullException.ThrowIfNull(x);
        n = x.GetLength(0); d = x.GetLength(1);
        var buffer = new double[n * d];
        var k = 0;
        for (var i = 0; i < n; i++)
            for (var j = 0; j < d; j++) buffer[k++] = x[i, j];
        return buffer;
    }

    /// <summary>Copies a jagged array of points (all rows the same length) into a fresh buffer.</summary>
    public static double[] Copy(double[][] x, out int n, out int d)
    {
        ArgumentNullException.ThrowIfNull(x);
        n = x.Length;
        d = n == 0 ? 0 : x[0].Length;
        var buffer = new double[n * d];
        for (var i = 0; i < n; i++)
        {
            if (x[i].Length != d) throw new ArgumentException("All points must have the same dimension.", nameof(x));
            x[i].CopyTo(buffer, i * d);
        }
        return buffer;
    }

    /// <summary>Converts a rectangular array to an NdArray.</summary>
    public static NdArray ToNd(double[,] x) => new(Copy(x, out var n, out var d), n, d);

    /// <summary>Converts a jagged array to an NdArray.</summary>
    public static NdArray ToNd(double[][] x) => new(Copy(x, out var n, out var d), n, d);

    /// <summary>Throws if any coordinate is NaN or infinite.</summary>
    public static void RequireFinite(ReadOnlySpan<double> data, string what)
    {
        foreach (var v in data)
            if (!double.IsFinite(v)) throw new ArgumentException($"{what} must contain only finite values.");
    }
}
