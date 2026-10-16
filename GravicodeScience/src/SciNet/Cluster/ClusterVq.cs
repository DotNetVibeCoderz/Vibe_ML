using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;

namespace Gravicode.Science.SciNet;

/// <summary>Initialisation of <see cref="Cluster.KMeans2(NdArray, int, int, double, KMeansInit, KMeansMissing, int)"/>.</summary>
public enum KMeansInit
{
    /// <summary>k draws from a Gaussian with the data's mean and covariance.</summary>
    Random,
    /// <summary>k distinct observations chosen at random.</summary>
    Points,
    /// <summary>k-means++ careful seeding (Arthur and Vassilvitskii 2007).</summary>
    PlusPlus,
}

/// <summary>What <see cref="Cluster.KMeans2(NdArray, int, int, double, KMeansInit, KMeansMissing, int)"/> does about an empty cluster.</summary>
public enum KMeansMissing
{
    /// <summary>Keep the empty cluster's previous centroid (scipy warns; here nothing is printed).</summary>
    Warn,
    /// <summary>Throw an <see cref="InvalidOperationException"/>.</summary>
    Raise,
}

public static partial class Cluster
{
    private const long VqParallelWork = 1L << 17;

    private static double[] ObsBuffer(NdArray obs, out int n, out int d, out bool oneD)
    {
        ArgumentNullException.ThrowIfNull(obs);
        oneD = obs.Rank == 1;
        if (obs.Rank > 2) throw new ArgumentException("Input of rank > 2 is not supported.");
        var buf = obs.AsContiguous().ToArray();
        n = obs.Shape[0];
        d = oneD ? 1 : obs.Shape[1];
        return buf;
    }

    /// <summary>
    /// Scales each feature (column) by its standard deviation (ddof 0) so all have unit
    /// variance; columns with zero deviation are left unchanged, as in scipy.
    /// </summary>
    public static NdArray Whiten(NdArray obs)
    {
        var x = ObsBuffer(obs, out var n, out var d, out var oneD);
        SpatialInput.RequireFinite(x, "obs");
        for (var j = 0; j < d; j++)
        {
            double mean = 0;
            for (var i = 0; i < n; i++) mean += x[i * d + j];
            mean /= n;
            double ss = 0;
            for (var i = 0; i < n; i++) { var t = x[i * d + j] - mean; ss += t * t; }
            var sd = Math.Sqrt(ss / n);
            if (sd == 0) sd = 1;
            for (var i = 0; i < n; i++) x[i * d + j] /= sd;
        }
        return oneD ? new NdArray(x, n) : new NdArray(x, n, d);
    }

    /// <summary>
    /// Assigns each observation to its nearest code (Euclidean; the first on ties) and returns
    /// the codes and the distances to them.
    /// </summary>
    public static (int[] Codes, double[] Distortion) Vq(NdArray obs, NdArray codeBook)
    {
        var x = ObsBuffer(obs, out var n, out var d, out _);
        var c = ObsBuffer(codeBook, out var k, out var dc, out _);
        if (dc != d) throw new ArgumentException("Observation and code_book should have the same rank.");
        SpatialInput.RequireFinite(x, "obs"); SpatialInput.RequireFinite(c, "code_book");
        var codes = new int[n];
        var dist = new double[n];
        Assign(x, n, d, c, k, codes, dist);
        for (var i = 0; i < n; i++) dist[i] = Math.Sqrt(dist[i]);
        return (codes, dist);
    }

    /// <summary>Nearest code per row (squared distances); rows in parallel, SIMD distances.</summary>
    private static void Assign(double[] x, int n, int d, double[] c, int k, int[] codes, double[] sq)
    {
        void Row(int i)
        {
            var row = x.AsSpan(i * d, d);
            var best = double.PositiveInfinity; var arg = 0;
            for (var j = 0; j < k; j++)
            {
                var s = Distance.SqEuclideanSimd(row, c.AsSpan(j * d, d));
                if (s < best) { best = s; arg = j; }
            }
            codes[i] = arg; sq[i] = best;
        }
        if ((long)n * k * d < VqParallelWork) for (var i = 0; i < n; i++) Row(i);
        else Parallel.For(0, n, Row);
    }

    private static (double[] Means, bool[] HasMembers) UpdateMeans(double[] x, int n, int d, int[] codes, int k)
    {
        var means = new double[k * d];
        var counts = new int[k];
        for (var i = 0; i < n; i++)
        {
            var c = codes[i];
            counts[c]++;
            for (var j = 0; j < d; j++) means[c * d + j] += x[i * d + j];
        }
        var has = new bool[k];
        for (var c = 0; c < k; c++)
        {
            has[c] = counts[c] > 0;
            if (has[c]) for (var j = 0; j < d; j++) means[c * d + j] /= counts[c];
        }
        return (means, has);
    }

    /// <summary>scipy's _kmeans: Lloyd iterations until the mean distortion changes by at most thresh.</summary>
    private static (double[] Book, int K, double Distortion) KMeansRaw(double[] x, int n, int d, double[] guess, int k, double thresh)
    {
        var book = guess; var kk = k;
        var diff = double.PositiveInfinity;
        double prev = double.PositiveInfinity, cur = double.PositiveInfinity;
        var codes = new int[n];
        var sq = new double[n];
        while (diff > thresh)
        {
            Assign(x, n, d, book, kk, codes, sq);
            var mean = 0.0;
            for (var i = 0; i < n; i++) mean += Math.Sqrt(sq[i]);
            mean /= n;
            prev = cur; cur = mean;
            var (means, has) = UpdateMeans(x, n, d, codes, kk);
            // Codes that lost all members are dropped, as scipy does.
            var keep = has.Count(h => h);
            var nb = new double[keep * d];
            var t = 0;
            for (var c = 0; c < kk; c++)
                if (has[c]) { Array.Copy(means, c * d, nb, t * d, d); t++; }
            book = nb; kk = keep;
            diff = Math.Abs(prev - cur);
        }
        return (book, kk, cur);
    }

    /// <summary>
    /// k-means with scipy's stopping rule: iterate until the mean distance to the nearest code
    /// changes by at most <paramref name="thresh"/>, restart <paramref name="iter"/> times from k
    /// random observations, keep the lowest distortion. Codes that lose all members are removed,
    /// so fewer than k may be returned.
    /// </summary>
    public static (NdArray CodeBook, double Distortion) KMeans(NdArray obs, int k, int iter = 20, double thresh = 1e-5, int seed = 0)
    {
        if (iter < 1) throw new ArgumentException($"iter must be at least 1, got {iter}");
        if (k < 1) throw new ArgumentException($"Asked for {k} clusters.");
        var x = ObsBuffer(obs, out var n, out var d, out var oneD);
        SpatialInput.RequireFinite(x, "obs");
        if (k > n) throw new ArgumentException("Cannot pick more initial codes than observations.");
        var rng = new GraviRandom(seed);
        double[] bestBook = []; var bestK = 0; var bestDist = double.PositiveInfinity;
        for (var it = 0; it < iter; it++)
        {
            var idx = rng.Choice(n, k, replace: false);
            var guess = new double[k * d];
            for (var c = 0; c < k; c++) Array.Copy(x, idx[c] * d, guess, c * d, d);
            var (book, kk, dist) = KMeansRaw(x, n, d, guess, k, thresh);
            if (dist < bestDist) { bestBook = book; bestK = kk; bestDist = dist; }
        }
        return (oneD ? new NdArray(bestBook, bestK) : new NdArray(bestBook, bestK, d), bestDist);
    }

    /// <summary>k-means from a given initial code book (scipy's <c>kmeans(obs, guess)</c>): one run, no restarts.</summary>
    public static (NdArray CodeBook, double Distortion) KMeans(NdArray obs, NdArray guess, double thresh = 1e-5)
    {
        var x = ObsBuffer(obs, out var n, out var d, out var oneD);
        var g = ObsBuffer(guess, out var k, out var dg, out _);
        if (dg != d) throw new ArgumentException("The code book must have the same dimension as the observations.");
        if (k < 1) throw new ArgumentException("Asked for 0 clusters.");
        var (book, kk, dist) = KMeansRaw(x, n, d, g, k, thresh);
        return (oneD ? new NdArray(book, kk) : new NdArray(book, kk, d), dist);
    }

    /// <summary>
    /// k-means for exactly <paramref name="iter"/> Lloyd iterations (scipy's <c>kmeans2</c>),
    /// starting from <paramref name="minit"/>. Returns the centroids and each observation's label.
    /// </summary>
    /// <remarks>Random draws use GraviRandom, so a given seed does not reproduce numpy's
    /// initial centroids; the algorithm and the result for a given initialisation are scipy's.
    /// <paramref name="thresh"/> is unused, as in scipy.</remarks>
    public static (NdArray Centroids, int[] Labels) KMeans2(NdArray data, int k, int iter = 10, double thresh = 1e-5,
        KMeansInit minit = KMeansInit.Random, KMeansMissing missing = KMeansMissing.Warn, int seed = 0)
    {
        if (iter < 1) throw new ArgumentException($"Invalid iter ({iter}), must be a positive integer.");
        if (k < 1) throw new ArgumentException($"Cannot ask kmeans2 for {k} clusters (k was {k})");
        var x = ObsBuffer(data, out var n, out var d, out var oneD);
        if (x.Length == 0) throw new ArgumentException("Empty input is not supported.");
        SpatialInput.RequireFinite(x, "data");
        var rng = new GraviRandom(seed);
        var book = minit switch
        {
            KMeansInit.Points => InitPoints(x, n, d, k, rng),
            KMeansInit.PlusPlus => InitPlusPlus(x, n, d, k, rng),
            KMeansInit.Random => InitRandom(x, n, d, k, rng),
            _ => throw new ArgumentException($"Unknown init method {minit}"),
        };
        return KMeans2Core(x, n, d, oneD, book, k, iter, missing);
    }

    /// <summary>kmeans2 from a given initial code book (scipy's <c>minit='matrix'</c>).</summary>
    public static (NdArray Centroids, int[] Labels) KMeans2(NdArray data, NdArray initial, int iter = 10,
        KMeansMissing missing = KMeansMissing.Warn)
    {
        if (iter < 1) throw new ArgumentException($"Invalid iter ({iter}), must be a positive integer.");
        var x = ObsBuffer(data, out var n, out var d, out var oneD);
        var g = ObsBuffer(initial, out var k, out var dg, out _);
        if (dg != d || data.Rank != initial.Rank) throw new ArgumentException("k array doesn't match data dimension");
        return KMeans2Core(x, n, d, oneD, g, k, iter, missing);
    }

    private static (NdArray, int[]) KMeans2Core(double[] x, int n, int d, bool oneD, double[] book, int k, int iter, KMeansMissing missing)
    {
        var codes = new int[n];
        var sq = new double[n];
        for (var it = 0; it < iter; it++)
        {
            Assign(x, n, d, book, k, codes, sq);
            var (means, has) = UpdateMeans(x, n, d, codes, k);
            if (has.Any(h => !h))
            {
                if (missing == KMeansMissing.Raise) throw new InvalidOperationException("One of the clusters is empty. Re-run kmeans with a different initialization.");
                for (var c = 0; c < k; c++) if (!has[c]) Array.Copy(book, c * d, means, c * d, d);
            }
            book = means;
        }
        return (oneD ? new NdArray(book, k) : new NdArray(book, k, d), codes);
    }

    private static double[] InitPoints(double[] x, int n, int d, int k, GraviRandom rng)
    {
        if (k > n) throw new ArgumentException("Cannot pick more initial points than observations.");
        var idx = rng.Choice(n, k, replace: false);
        var book = new double[k * d];
        for (var c = 0; c < k; c++) Array.Copy(x, idx[c] * d, book, c * d, d);
        return book;
    }

    private static double[] InitPlusPlus(double[] x, int n, int d, int k, GraviRandom rng)
    {
        var book = new double[k * d];
        var d2 = new double[n];
        Array.Fill(d2, double.PositiveInfinity);
        for (var c = 0; c < k; c++)
        {
            int idx;
            if (c == 0) idx = rng.Next(n);
            else
            {
                var prev = book.AsSpan((c - 1) * d, d);
                var total = 0.0;
                for (var i = 0; i < n; i++) { d2[i] = Math.Min(d2[i], Distance.SqEuclideanSimd(x.AsSpan(i * d, d), prev)); total += d2[i]; }
                var r = rng.NextDouble();
                var cum = 0.0;
                idx = n - 1;
                for (var i = 0; i < n; i++) { cum += d2[i] / total; if (cum >= r) { idx = i; break; } }
            }
            Array.Copy(x, idx * d, book, c * d, d);
        }
        return book;
    }

    /// <summary>scipy's _krandinit: Gaussian with the data's mean and covariance (SVD route when d &gt; n).</summary>
    private static double[] InitRandom(double[] x, int n, int d, int k, GraviRandom rng)
    {
        var mu = new double[d];
        for (var i = 0; i < n; i++) for (var j = 0; j < d; j++) mu[j] += x[i * d + j];
        for (var j = 0; j < d; j++) mu[j] /= n;
        var centred = new double[n * d];
        for (var i = 0; i < n; i++) for (var j = 0; j < d; j++) centred[i * d + j] = x[i * d + j] - mu[j];
        double[] transform; int rows;
        if (d > n)
        {
            var svd = Decomposition.Svd(new NdArray(centred, n, d));
            var s = svd.SingularValues.ToArray();
            var vt = svd.V.T.Copy().ToArray(); // (r, d)
            rows = s.Length;
            transform = new double[rows * d];
            for (var r = 0; r < rows; r++) for (var j = 0; j < d; j++) transform[r * d + j] = s[r] * vt[r * d + j] / Math.Sqrt(n - 1.0);
        }
        else
        {
            var cov = new double[d * d];
            for (var i = 0; i < n; i++)
                for (var a = 0; a < d; a++)
                    for (var b = 0; b < d; b++) cov[a * d + b] += centred[i * d + a] * centred[i * d + b];
            for (var t = 0; t < cov.Length; t++) cov[t] /= n - 1;
            var l = Decomposition.Cholesky(new NdArray(cov, d, d)).AsContiguous().ToArray();
            rows = d;
            transform = new double[d * d]; // L^T
            for (var a = 0; a < d; a++) for (var b = 0; b < d; b++) transform[a * d + b] = l[b * d + a];
        }
        var book = new double[k * d];
        for (var c = 0; c < k; c++)
        {
            var z = new double[rows];
            for (var r = 0; r < rows; r++) z[r] = rng.Normal();
            for (var j = 0; j < d; j++)
            {
                var v = mu[j];
                for (var r = 0; r < rows; r++) v += z[r] * transform[r * d + j];
                book[c * d + j] = v;
            }
        }
        return book;
    }
}
