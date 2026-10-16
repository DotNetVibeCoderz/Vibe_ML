using System.Numerics;
using System.Runtime.InteropServices;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Spatial;

/// <summary>The built-in metrics of <see cref="Distance.Pdist(NdArray, DistanceMetric, double, double[], double[], NdArray, double?, int)"/> and <see cref="Distance.Cdist(NdArray, NdArray, DistanceMetric, double, double[], double[], NdArray, double?, int)"/>.</summary>
public enum DistanceMetric
{
    /// <summary>sum|u-v| / sum|u+v|.</summary>
    BrayCurtis,
    /// <summary>sum |u-v| / (|u|+|v|), 0/0 terms skipped.</summary>
    Canberra,
    /// <summary>max |u-v|.</summary>
    Chebyshev,
    /// <summary>sum |u-v| (Manhattan).</summary>
    Cityblock,
    /// <summary>1 - centred cosine similarity.</summary>
    Correlation,
    /// <summary>1 - u·v / (|u||v|).</summary>
    Cosine,
    /// <summary>Boolean Dice dissimilarity.</summary>
    Dice,
    /// <summary>sqrt(sum (u-v)^2).</summary>
    Euclidean,
    /// <summary>Fraction of coordinates that differ.</summary>
    Hamming,
    /// <summary>Boolean Jaccard-Needham dissimilarity.</summary>
    Jaccard,
    /// <summary>Jensen-Shannon distance between probability vectors.</summary>
    JensenShannon,
    /// <summary>Boolean Kulczynski 1 (removed from scipy 1.17; kept with its old definition).</summary>
    Kulczynski1,
    /// <summary>sqrt((u-v) VI (u-v)^T).</summary>
    Mahalanobis,
    /// <summary>(sum w|u-v|^p)^(1/p).</summary>
    Minkowski,
    /// <summary>Boolean Rogers-Tanimoto dissimilarity.</summary>
    RogersTanimoto,
    /// <summary>Boolean Russell-Rao dissimilarity.</summary>
    RussellRao,
    /// <summary>Standardised Euclidean, sqrt(sum (u-v)^2 / V).</summary>
    SEuclidean,
    /// <summary>Boolean Sokal-Michener (removed from scipy 1.17; equal to Rogers-Tanimoto).</summary>
    SokalMichener,
    /// <summary>Boolean Sokal-Sneath dissimilarity.</summary>
    SokalSneath,
    /// <summary>sum (u-v)^2.</summary>
    SqEuclidean,
    /// <summary>Boolean Yule dissimilarity.</summary>
    Yule,
}

/// <summary>
/// Distance computations (scipy.spatial.distance): per-pair metrics, condensed and rectangular
/// distance matrices, and conversions between them.
/// </summary>
/// <remarks>
/// <para><b>Euclidean family.</b> <see cref="Pdist(NdArray, DistanceMetric, double, double[], double[], NdArray, double?, int)"/>
/// and <see cref="Cdist(NdArray, NdArray, DistanceMetric, double, double[], double[], NdArray, double?, int)"/>
/// compute Euclidean and squared Euclidean distances <em>directly</em> from the differences, like
/// scipy, not through <c>|a|² + |b|² - 2a·b</c>. The expansion would turn the work into a matrix
/// product, but it cancels catastrophically for nearby points far from the origin: two points
/// 1e-4 apart at coordinates ~1e4 lose every significant digit. The direct kernel instead
/// compares one row of A with four rows of B per pass (one load of A feeds four
/// <see cref="Vector{T}"/> accumulators), walks B in L2-sized blocks, and runs rows in parallel.</para>
/// <para><b>Boolean metrics</b> treat any non-zero value as true, as scipy's pdist/cdist do.
/// <b>Hamming</b> compares the values themselves, again as scipy does.</para>
/// <para><b>Weights.</b> Where scipy accepts <c>w</c>, so does this class, with scipy's
/// definitions (for example weighted Hamming divides by the total weight, and weighted
/// Chebyshev ignores coordinates of zero weight).</para>
/// </remarks>
public static class Distance
{
    private const long ParallelWork = 1L << 18;

    // ================================================================== per-pair metrics

    private static void Same(ReadOnlySpan<double> u, ReadOnlySpan<double> v)
    {
        if (u.Length != v.Length) throw new ArgumentException("The vectors must have equal lengths.");
    }

    private static void SameW(ReadOnlySpan<double> u, ReadOnlySpan<double> w)
    {
        if (!w.IsEmpty && w.Length != u.Length) throw new ArgumentException("'w' should have the same length as 'u' and 'v'.");
    }

    /// <summary>Squared Euclidean distance (SIMD when unweighted).</summary>
    public static double SqEuclidean(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
    {
        Same(u, v); SameW(u, w);
        if (w.IsEmpty) return SqEuclideanSimd(u, v);
        var s = 0.0;
        for (var i = 0; i < u.Length; i++) { var d = u[i] - v[i]; s += w[i] * d * d; }
        return s;
    }

    /// <summary>Euclidean distance, optionally weighted: sqrt(sum w (u-v)^2).</summary>
    public static double Euclidean(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
        => Math.Sqrt(SqEuclidean(u, v, w));

    /// <summary>Standardised Euclidean distance with per-coordinate variances <paramref name="variance"/>.</summary>
    public static double SEuclidean(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> variance)
    {
        Same(u, v);
        if (variance.Length != u.Length) throw new ArgumentException("V must have the same dimension as u and v.");
        var s = 0.0;
        for (var i = 0; i < u.Length; i++) { var d = u[i] - v[i]; s += d * d / variance[i]; }
        return Math.Sqrt(s);
    }

    /// <summary>Minkowski distance (sum w|u-v|^p)^(1/p); p = +inf gives the (weighted) Chebyshev distance.</summary>
    public static double Minkowski(ReadOnlySpan<double> u, ReadOnlySpan<double> v, double p = 2, ReadOnlySpan<double> w = default)
    {
        Same(u, v); SameW(u, w);
        if (!(p > 0)) throw new ArgumentException("p must be greater than 0.", nameof(p));
        if (double.IsPositiveInfinity(p)) return Chebyshev(u, v, w);
        if (p == 2) return Euclidean(u, v, w);
        if (p == 1) return Cityblock(u, v, w);
        var s = 0.0;
        for (var i = 0; i < u.Length; i++)
        {
            var t = Math.Pow(Math.Abs(u[i] - v[i]), p);
            s += w.IsEmpty ? t : w[i] * t;
        }
        return Math.Pow(s, 1.0 / p);
    }

    /// <summary>City block (Manhattan) distance, optionally weighted.</summary>
    public static double Cityblock(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
    {
        Same(u, v); SameW(u, w);
        if (w.IsEmpty) return CityblockSimd(u, v);
        var s = 0.0;
        for (var i = 0; i < u.Length; i++) s += w[i] * Math.Abs(u[i] - v[i]);
        return s;
    }

    /// <summary>Chebyshev distance; with weights, coordinates of zero weight are ignored (scipy).</summary>
    public static double Chebyshev(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
    {
        Same(u, v); SameW(u, w);
        if (w.IsEmpty) return ChebyshevSimd(u, v);
        var s = 0.0;
        for (var i = 0; i < u.Length; i++)
            if (w[i] > 0) s = Math.Max(s, Math.Abs(u[i] - v[i]));
        return s;
    }

    /// <summary>Cosine distance 1 - u·v/(|u||v|), clipped to [0, 2]; NaN for a zero vector.</summary>
    public static double Cosine(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
        => CorrelationCore(u, v, w, centered: false);

    /// <summary>Correlation distance: cosine distance of the (weighted-)mean-centred vectors.</summary>
    public static double Correlation(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
        => CorrelationCore(u, v, w, centered: true);

    private static double CorrelationCore(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w, bool centered)
    {
        Same(u, v); SameW(u, w);
        var n = u.Length;
        double wsum = 0;
        if (!w.IsEmpty) foreach (var x in w) wsum += x;
        double um = 0, vm = 0;
        if (centered)
        {
            for (var i = 0; i < n; i++)
            {
                var wi = w.IsEmpty ? 1.0 / n : w[i] / wsum;
                um += wi * u[i]; vm += wi * v[i];
            }
        }
        double uv = 0, uu = 0, vv = 0;
        for (var i = 0; i < n; i++)
        {
            var a = u[i] - um; var b = v[i] - vm;
            var wi = w.IsEmpty ? 1.0 : w[i] / wsum;
            uv += a * b * wi; uu += a * a * wi; vv += b * b * wi;
        }
        var d = 1.0 - uv / Math.Sqrt(uu * vv);
        return double.IsNaN(d) ? d : Math.Clamp(d, 0.0, 2.0);
    }

    /// <summary>Canberra distance sum w|u-v|/(|u|+|v|); terms with 0/0 contribute nothing.</summary>
    public static double Canberra(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
    {
        Same(u, v); SameW(u, w);
        var s = 0.0;
        for (var i = 0; i < u.Length; i++)
        {
            var den = Math.Abs(u[i]) + Math.Abs(v[i]);
            if (den == 0) continue;
            var t = Math.Abs(u[i] - v[i]) / den;
            s += w.IsEmpty ? t : w[i] * t;
        }
        return s;
    }

    /// <summary>Bray-Curtis dissimilarity sum w|u-v| / sum w|u+v|.</summary>
    public static double BrayCurtis(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
    {
        Same(u, v); SameW(u, w);
        double num = 0, den = 0;
        for (var i = 0; i < u.Length; i++)
        {
            var wi = w.IsEmpty ? 1.0 : w[i];
            num += wi * Math.Abs(u[i] - v[i]);
            den += wi * Math.Abs(u[i] + v[i]);
        }
        return num / den;
    }

    /// <summary>Mahalanobis distance with inverse covariance <paramref name="vi"/> (d × d).</summary>
    public static double Mahalanobis(ReadOnlySpan<double> u, ReadOnlySpan<double> v, NdArray vi)
    {
        Same(u, v);
        ArgumentNullException.ThrowIfNull(vi);
        var n = u.Length;
        if (vi.Rank != 2 || vi.Shape[0] != n || vi.Shape[1] != n) throw new ArgumentException("VI must be a d x d matrix.", nameof(vi));
        return MahalanobisCore(u, v, vi.AsContiguous().AsSpan());
    }

    private static double MahalanobisCore(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> vi)
    {
        var n = u.Length;
        Span<double> delta = n <= 256 ? stackalloc double[n] : new double[n];
        for (var i = 0; i < n; i++) delta[i] = u[i] - v[i];
        var m = 0.0;
        for (var i = 0; i < n; i++)
        {
            var row = 0.0;
            for (var j = 0; j < n; j++) row += vi[j * n + i] * delta[j];
            m += row * delta[i];
        }
        return Math.Sqrt(m);
    }

    /// <summary>
    /// Jensen-Shannon distance sqrt(JS divergence) between two non-negative vectors, each
    /// normalised to sum to one; natural logarithm unless <paramref name="logBase"/> is given.
    /// </summary>
    public static double JensenShannon(ReadOnlySpan<double> p, ReadOnlySpan<double> q, double? logBase = null)
    {
        Same(p, q);
        double sp = 0, sq = 0;
        for (var i = 0; i < p.Length; i++) { sp += p[i]; sq += q[i]; }
        var js = 0.0;
        for (var i = 0; i < p.Length; i++)
        {
            var a = p[i] / sp; var b = q[i] / sq;
            var m = 0.5 * (a + b);
            js += RelEntr(a, m) + RelEntr(b, m);
        }
        if (logBase is { } lb) js /= Math.Log(lb);
        return Math.Sqrt(js / 2.0);
    }

    private static double RelEntr(double x, double y)
    {
        if (double.IsNaN(x) || double.IsNaN(y)) return double.NaN;
        if (x > 0 && y > 0) return x * Math.Log(x / y);
        if (x == 0 && y >= 0) return 0;
        return double.PositiveInfinity;
    }

    /// <summary>Fraction (or weighted fraction) of coordinates where u and v differ.</summary>
    public static double Hamming(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
    {
        Same(u, v); SameW(u, w);
        double s = 0, total = 0;
        for (var i = 0; i < u.Length; i++)
        {
            var wi = w.IsEmpty ? 1.0 : w[i];
            total += wi;
            if (u[i] != v[i]) s += wi;
        }
        return s / total;
    }

    /// <summary>Weighted counts of (false,false), (false,true), (true,false), (true,true) pairs.</summary>
    private static (double Nff, double Nft, double Ntf, double Ntt) Correspond(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w)
    {
        Same(u, v); SameW(u, w);
        double nff = 0, nft = 0, ntf = 0, ntt = 0;
        for (var i = 0; i < u.Length; i++)
        {
            var wi = w.IsEmpty ? 1.0 : w[i];
            var a = u[i] != 0; var b = v[i] != 0;
            if (a) { if (b) ntt += wi; else ntf += wi; }
            else { if (b) nft += wi; else nff += wi; }
        }
        return (nff, nft, ntf, ntt);
    }

    /// <summary>Jaccard-Needham dissimilarity; 0 when both vectors are entirely false.</summary>
    public static double Jaccard(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
    {
        var (_, nft, ntf, ntt) = Correspond(u, v, w);
        var nonzero = nft + ntf + ntt;
        return nonzero == 0 ? 0.0 : (nft + ntf) / nonzero;
    }

    /// <summary>Dice dissimilarity (ntf+nft)/(2ntt+ntf+nft); NaN for two all-false vectors (as scipy's pdist).</summary>
    public static double Dice(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
    {
        var (_, nft, ntf, ntt) = Correspond(u, v, w);
        return (ntf + nft) / (2.0 * ntt + ntf + nft);
    }

    /// <summary>Kulczynski 1 dissimilarity ntt/(ntf+nft) (scipy &lt; 1.17 definition).</summary>
    public static double Kulczynski1(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
    {
        var (_, nft, ntf, ntt) = Correspond(u, v, w);
        return ntt / (ntf + nft);
    }

    /// <summary>Rogers-Tanimoto dissimilarity R/(ntt+nff+R), R = 2(ntf+nft).</summary>
    public static double RogersTanimoto(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
    {
        var (nff, nft, ntf, ntt) = Correspond(u, v, w);
        var r = 2.0 * (ntf + nft);
        return r / (ntt + nff + r);
    }

    /// <summary>Sokal-Michener dissimilarity (scipy &lt; 1.17); algebraically equal to Rogers-Tanimoto.</summary>
    public static double SokalMichener(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
        => RogersTanimoto(u, v, w);

    /// <summary>Russell-Rao dissimilarity (n - ntt)/n, n the length or total weight.</summary>
    public static double RussellRao(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
    {
        var (nff, nft, ntf, ntt) = Correspond(u, v, w);
        var n = nff + nft + ntf + ntt;
        return (n - ntt) / n;
    }

    /// <summary>Sokal-Sneath dissimilarity R/(ntt+R), R = 2(ntf+nft); NaN for two all-false vectors.</summary>
    public static double SokalSneath(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
    {
        var (_, nft, ntf, ntt) = Correspond(u, v, w);
        var r = 2.0 * (ntf + nft);
        return r / (ntt + r);
    }

    /// <summary>Yule dissimilarity 2·ntf·nft/(ntt·nff + ntf·nft); 0 when ntf·nft = 0.</summary>
    public static double Yule(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w = default)
    {
        var (nff, nft, ntf, ntt) = Correspond(u, v, w);
        var halfR = ntf * nft;
        return halfR == 0 ? 0.0 : 2.0 * halfR / (ntt * nff + halfR);
    }

    /// <summary>
    /// Directed Hausdorff distance from <paramref name="u"/> to <paramref name="v"/> (rows are
    /// points), with the indices of the pair that realises it.
    /// </summary>
    /// <remarks>Taha and Hanbury's (2015) early-break algorithm, as scipy: the rows are visited
    /// in a random order (seeded) so the inner loop usually stops after a few candidates. The
    /// distance is exact; when several pairs realise it, which pair is reported depends on the
    /// shuffle, and the shuffle is GraviRandom's, not numpy's.</remarks>
    public static (double Distance, int IndexU, int IndexV) DirectedHausdorff(NdArray u, NdArray v, int seed = 0)
    {
        var a = SpatialInput.Copy(u, out var na, out var d, nameof(u));
        var b = SpatialInput.Copy(v, out var nb, out var d2, nameof(v));
        if (d != d2) throw new ArgumentException("u and v need to have the same number of columns.");
        if (na == 0 || nb == 0) throw new ArgumentException("u and v must not be empty.");
        var rng = new GraviRandom(seed);
        var pu = rng.Permutation(na);
        var pv = rng.Permutation(nb);
        double cmax = 0;
        int iRet = 0, jRet = 0;
        for (var ii = 0; ii < na; ii++)
        {
            var i = pu[ii];
            var noBreak = true;
            var cmin = double.PositiveInfinity;
            int iStore = 0, jStore = 0;
            var ra = a.AsSpan(i * d, d);
            for (var jj = 0; jj < nb; jj++)
            {
                var j = pv[jj];
                var dist = SqEuclideanSimd(ra, b.AsSpan(j * d, d));
                if (dist < cmax) { noBreak = false; break; }
                if (dist < cmin) { cmin = dist; iStore = i; jStore = j; }
            }
            if (!double.IsPositiveInfinity(cmin) && cmin > cmax && noBreak)
            {
                cmax = cmin; iRet = iStore; jRet = jStore;
            }
        }
        return (Math.Sqrt(cmax), iRet, jRet);
    }

    // ================================================================== SIMD kernels

    /// <summary>Scalar reference for the SIMD squared-Euclidean kernel (kept for the tests).</summary>
    internal static double SqEuclideanScalar(ReadOnlySpan<double> u, ReadOnlySpan<double> v)
    {
        var s = 0.0;
        for (var i = 0; i < u.Length; i++) { var d = u[i] - v[i]; s += d * d; }
        return s;
    }

    internal static double SqEuclideanSimd(ReadOnlySpan<double> u, ReadOnlySpan<double> v)
    {
        var n = u.Length;
        var i = 0;
        var s = 0.0;
        var width = Vector<double>.Count;
        if (Vector.IsHardwareAccelerated && n >= width)
        {
            ref var ru = ref MemoryMarshal.GetReference(u);
            ref var rv = ref MemoryMarshal.GetReference(v);
            var acc = Vector<double>.Zero;
            for (; i <= n - width; i += width)
            {
                var diff = Vector.LoadUnsafe(ref ru, (nuint)i) - Vector.LoadUnsafe(ref rv, (nuint)i);
                acc += diff * diff;
            }
            s = Vector.Sum(acc);
        }
        for (; i < n; i++) { var d = u[i] - v[i]; s += d * d; }
        return s;
    }

    private static double CityblockSimd(ReadOnlySpan<double> u, ReadOnlySpan<double> v)
    {
        var n = u.Length;
        var i = 0;
        var s = 0.0;
        var width = Vector<double>.Count;
        if (Vector.IsHardwareAccelerated && n >= width)
        {
            ref var ru = ref MemoryMarshal.GetReference(u);
            ref var rv = ref MemoryMarshal.GetReference(v);
            var acc = Vector<double>.Zero;
            for (; i <= n - width; i += width)
                acc += Vector.Abs(Vector.LoadUnsafe(ref ru, (nuint)i) - Vector.LoadUnsafe(ref rv, (nuint)i));
            s = Vector.Sum(acc);
        }
        for (; i < n; i++) s += Math.Abs(u[i] - v[i]);
        return s;
    }

    private static double ChebyshevSimd(ReadOnlySpan<double> u, ReadOnlySpan<double> v)
    {
        var n = u.Length;
        var i = 0;
        var s = 0.0;
        var width = Vector<double>.Count;
        if (Vector.IsHardwareAccelerated && n >= width)
        {
            ref var ru = ref MemoryMarshal.GetReference(u);
            ref var rv = ref MemoryMarshal.GetReference(v);
            var acc = Vector<double>.Zero;
            for (; i <= n - width; i += width)
                acc = Vector.Max(acc, Vector.Abs(Vector.LoadUnsafe(ref ru, (nuint)i) - Vector.LoadUnsafe(ref rv, (nuint)i)));
            for (var k = 0; k < width; k++) s = Math.Max(s, acc[k]);
        }
        for (; i < n; i++) s = Math.Max(s, Math.Abs(u[i] - v[i]));
        return s;
    }

    private static double DotSimd(ReadOnlySpan<double> u, ReadOnlySpan<double> v)
    {
        var n = u.Length;
        var i = 0;
        var s = 0.0;
        var width = Vector<double>.Count;
        if (Vector.IsHardwareAccelerated && n >= width)
        {
            ref var ru = ref MemoryMarshal.GetReference(u);
            ref var rv = ref MemoryMarshal.GetReference(v);
            var acc = Vector<double>.Zero;
            for (; i <= n - width; i += width)
                acc += Vector.LoadUnsafe(ref ru, (nuint)i) * Vector.LoadUnsafe(ref rv, (nuint)i);
            s = Vector.Sum(acc);
        }
        for (; i < n; i++) s += u[i] * v[i];
        return s;
    }

    /// <summary>
    /// Squared distances from row <paramref name="a"/> to rows [j0, j1) of B, written to
    /// <paramref name="dst"/>. Four B rows share each load of A.
    /// </summary>
    private static void SqEuclideanRowBlock(ReadOnlySpan<double> a, double[] b, int d, int j0, int j1, Span<double> dst)
    {
        var width = Vector<double>.Count;
        var j = j0;
        if (Vector.IsHardwareAccelerated && d >= width)
        {
            ref var ra = ref MemoryMarshal.GetReference(a);
            ref var rb = ref MemoryMarshal.GetArrayDataReference(b);
            var vecEnd = d - d % width;
            for (; j + 4 <= j1; j += 4)
            {
                var o0 = (nuint)(j * d); var o1 = o0 + (nuint)d; var o2 = o1 + (nuint)d; var o3 = o2 + (nuint)d;
                Vector<double> s0 = Vector<double>.Zero, s1 = s0, s2 = s0, s3 = s0;
                for (var k = 0; k < vecEnd; k += width)
                {
                    var va = Vector.LoadUnsafe(ref ra, (nuint)k);
                    var t0 = va - Vector.LoadUnsafe(ref rb, o0 + (nuint)k); s0 += t0 * t0;
                    var t1 = va - Vector.LoadUnsafe(ref rb, o1 + (nuint)k); s1 += t1 * t1;
                    var t2 = va - Vector.LoadUnsafe(ref rb, o2 + (nuint)k); s2 += t2 * t2;
                    var t3 = va - Vector.LoadUnsafe(ref rb, o3 + (nuint)k); s3 += t3 * t3;
                }
                double r0 = Vector.Sum(s0), r1 = Vector.Sum(s1), r2 = Vector.Sum(s2), r3 = Vector.Sum(s3);
                for (var k = vecEnd; k < d; k++)
                {
                    var x = a[k];
                    var e0 = x - b[(int)o0 + k]; r0 += e0 * e0;
                    var e1 = x - b[(int)o1 + k]; r1 += e1 * e1;
                    var e2 = x - b[(int)o2 + k]; r2 += e2 * e2;
                    var e3 = x - b[(int)o3 + k]; r3 += e3 * e3;
                }
                dst[j - j0] = r0; dst[j - j0 + 1] = r1; dst[j - j0 + 2] = r2; dst[j - j0 + 3] = r3;
            }
        }
        for (; j < j1; j++) dst[j - j0] = SqEuclideanSimd(a, b.AsSpan(j * d, d));
    }

    // ================================================================== matrices

    private static int Workers(int workers) => workers == -1 ? Environment.ProcessorCount : Math.Max(1, workers);

    private static void Rows(int count, long work, int workers, Action<int> body)
    {
        var w = Workers(workers);
        if (w == 1 || work < ParallelWork || count < 2)
        {
            for (var i = 0; i < count; i++) body(i);
            return;
        }
        Parallel.For(0, count, new ParallelOptions { MaxDegreeOfParallelism = w }, body);
    }

    /// <summary>Condensed index of pair (i, j), i &lt; j, in an n-point condensed matrix.</summary>
    internal static long CondensedIndex(long n, long i, long j) => n * i - i * (i + 1) / 2 + (j - i - 1);

    private static Func<ReadOnlySpan<double>, ReadOnlySpan<double>, double> Kernel(
        DistanceMetric metric, double p, double[]? w, double[]? v, double[]? vi, double? logBase)
    {
        switch (metric)
        {
            case DistanceMetric.Euclidean: return (a, b) => Euclidean(a, b, w);
            case DistanceMetric.SqEuclidean: return (a, b) => SqEuclidean(a, b, w);
            case DistanceMetric.Minkowski: return (a, b) => Minkowski(a, b, p, w);
            case DistanceMetric.Cityblock: return (a, b) => Cityblock(a, b, w);
            case DistanceMetric.Chebyshev: return (a, b) => Chebyshev(a, b, w);
            case DistanceMetric.Cosine: return (a, b) => Cosine(a, b, w);
            case DistanceMetric.Correlation: return (a, b) => Correlation(a, b, w);
            case DistanceMetric.Canberra: return (a, b) => Canberra(a, b, w);
            case DistanceMetric.BrayCurtis: return (a, b) => BrayCurtis(a, b, w);
            case DistanceMetric.Hamming: return (a, b) => Hamming(a, b, w);
            case DistanceMetric.Jaccard: return (a, b) => Jaccard(a, b, w);
            case DistanceMetric.Dice: return (a, b) => Dice(a, b, w);
            case DistanceMetric.Kulczynski1: return (a, b) => Kulczynski1(a, b, w);
            case DistanceMetric.RogersTanimoto: return (a, b) => RogersTanimoto(a, b, w);
            case DistanceMetric.SokalMichener: return (a, b) => SokalMichener(a, b, w);
            case DistanceMetric.RussellRao: return (a, b) => RussellRao(a, b, w);
            case DistanceMetric.SokalSneath: return (a, b) => SokalSneath(a, b, w);
            case DistanceMetric.Yule: return (a, b) => Yule(a, b, w);
            case DistanceMetric.JensenShannon: return (a, b) => JensenShannon(a, b, logBase);
            case DistanceMetric.SEuclidean: return (a, b) => SEuclidean(a, b, v);
            case DistanceMetric.Mahalanobis: return (a, b) => MahalanobisCore(a, b, vi);
            default: throw new ArgumentException($"Unknown metric {metric}.", nameof(metric));
        }
    }

    /// <summary>Column variances with ddof = 1 (scipy's default V for seuclidean).</summary>
    private static double[] ColumnVariance(double[] x, int n, int d)
    {
        var mean = new double[d];
        for (var i = 0; i < n; i++) for (var j = 0; j < d; j++) mean[j] += x[i * d + j];
        for (var j = 0; j < d; j++) mean[j] /= n;
        var var = new double[d];
        for (var i = 0; i < n; i++) for (var j = 0; j < d; j++) { var t = x[i * d + j] - mean[j]; var[j] += t * t; }
        for (var j = 0; j < d; j++) var[j] /= n - 1;
        return var;
    }

    /// <summary>inv(cov(X)) with ddof = 1, scipy's default VI for mahalanobis.</summary>
    private static double[] InverseCovariance(double[] x, int n, int d)
    {
        if (n <= d) throw new ArgumentException($"The number of observations ({n}) is too small; the covariance matrix is singular. For observations with {d} dimensions, at least {d + 1} observations are required.");
        var mean = new double[d];
        for (var i = 0; i < n; i++) for (var j = 0; j < d; j++) mean[j] += x[i * d + j];
        for (var j = 0; j < d; j++) mean[j] /= n;
        var cov = new double[d * d];
        for (var i = 0; i < n; i++)
            for (var a = 0; a < d; a++)
            {
                var ta = x[i * d + a] - mean[a];
                for (var b = 0; b < d; b++) cov[a * d + b] += ta * (x[i * d + b] - mean[b]);
            }
        for (var k = 0; k < cov.Length; k++) cov[k] /= n - 1;
        var inv = LinAlg.Inverse(new NdArray(cov, d, d));
        // scipy passes inv(cov).T; for a symmetric covariance the transpose is the same matrix.
        return inv.Transpose().Copy().ToArray();
    }

    private static double[] CheckWeights(double[]? w, int d)
    {
        if (w is null) return [];
        if (w.Length != d) throw new ArgumentException("Weights must have the same size as the vectors.", nameof(w));
        foreach (var x in w) if (x < 0) throw new ArgumentException("Input weights should be all non-negative.", nameof(w));
        return w;
    }

    /// <summary>
    /// Pairwise distances between the rows of <paramref name="x"/> as a condensed vector of
    /// length n(n-1)/2: entry <c>n*i - i*(i+1)/2 + (j-i-1)</c> holds d(x_i, x_j) for i &lt; j.
    /// </summary>
    /// <param name="x">Observations, shape (n, d).</param>
    /// <param name="metric">Metric.</param>
    /// <param name="p">Minkowski p.</param>
    /// <param name="w">Optional per-coordinate weights (metrics that accept them).</param>
    /// <param name="v">SEuclidean variances; default: column variances of x (ddof 1).</param>
    /// <param name="vi">Mahalanobis inverse covariance; default: inverse sample covariance of x.</param>
    /// <param name="logBase">Jensen-Shannon logarithm base (default e).</param>
    /// <param name="workers">Threads, -1 for all cores; small problems run serially.</param>
    public static double[] Pdist(NdArray x, DistanceMetric metric = DistanceMetric.Euclidean, double p = 2,
        double[]? w = null, double[]? v = null, NdArray? vi = null, double? logBase = null, int workers = -1)
    {
        var data = SpatialInput.Copy(x, out var n, out var d, nameof(x));
        if (x.Rank != 2) throw new ArgumentException("A 2-dimensional array must be passed.", nameof(x));
        var wv = CheckWeights(w, d);
        var result = new double[(long)n * (n - 1) / 2];
        if (n < 2) return result;
        long work = (long)n * n / 2 * Math.Max(d, 1);

        if ((metric == DistanceMetric.Euclidean || metric == DistanceMetric.SqEuclidean || (metric == DistanceMetric.Minkowski && p == 2)) && wv.Length == 0)
        {
            var root = metric != DistanceMetric.SqEuclidean;
            Rows(n - 1, work, workers, i =>
            {
                var off = (int)CondensedIndex(n, i, i + 1);
                var dst = result.AsSpan(off, n - i - 1);
                SqEuclideanRowBlock(data.AsSpan(i * d, d), data, d, i + 1, n, dst);
                if (root) for (var k = 0; k < dst.Length; k++) dst[k] = Math.Sqrt(dst[k]);
            });
            return result;
        }
        if ((metric == DistanceMetric.Cosine || metric == DistanceMetric.Correlation) && wv.Length == 0)
        {
            var prepared = PrepareCosine(data, n, d, metric == DistanceMetric.Correlation, out var norms);
            Rows(n - 1, work, workers, i =>
            {
                var a = prepared.AsSpan(i * d, d);
                var off = CondensedIndex(n, i, i + 1);
                for (var j = i + 1; j < n; j++)
                    result[off + j - i - 1] = CosineFromDot(DotSimd(a, prepared.AsSpan(j * d, d)), norms[i], norms[j]);
            });
            return result;
        }

        var vv = metric == DistanceMetric.SEuclidean ? v ?? ColumnVariance(data, n, d) : null;
        var vim = metric == DistanceMetric.Mahalanobis ? (vi is null ? InverseCovariance(data, n, d) : vi.AsContiguous().ToArray()) : null;
        if (vim is not null && vim.Length != d * d) throw new ArgumentException("VI must be a d x d matrix.", nameof(vi));
        var kernel = Kernel(metric, p, wv, vv, vim, logBase);
        return PdistWith(data, n, d, kernel, workers, result);
    }

    /// <summary>Pairwise distances with a caller-supplied metric.</summary>
    public static double[] Pdist(NdArray x, Func<ReadOnlySpan<double>, ReadOnlySpan<double>, double> metric, int workers = -1)
    {
        ArgumentNullException.ThrowIfNull(metric);
        var data = SpatialInput.Copy(x, out var n, out var d, nameof(x));
        var result = new double[(long)n * (n - 1) / 2];
        return n < 2 ? result : PdistWith(data, n, d, metric, workers, result);
    }

    /// <summary>Pairwise distances between the rows of a rectangular array.</summary>
    public static double[] Pdist(double[,] x, DistanceMetric metric = DistanceMetric.Euclidean, double p = 2, double[]? w = null, int workers = -1)
        => Pdist(SpatialInput.ToNd(x), metric, p, w, workers: workers);

    private static double[] PdistWith(double[] data, int n, int d, Func<ReadOnlySpan<double>, ReadOnlySpan<double>, double> kernel, int workers, double[] result)
    {
        long work = (long)n * n / 2 * Math.Max(d, 1) * 4;
        Rows(n - 1, work, workers, i =>
        {
            var a = data.AsSpan(i * d, d);
            var off = CondensedIndex(n, i, i + 1);
            for (var j = i + 1; j < n; j++) result[off + j - i - 1] = kernel(a, data.AsSpan(j * d, d));
        });
        return result;
    }

    private static double[] PrepareCosine(double[] data, int n, int d, bool center, out double[] norms)
    {
        var prepared = center ? new double[data.Length] : data;
        norms = new double[n];
        for (var i = 0; i < n; i++)
        {
            var row = data.AsSpan(i * d, d);
            var dst = prepared.AsSpan(i * d, d);
            if (center)
            {
                var mean = 0.0;
                foreach (var t in row) mean += t;
                mean /= d;
                for (var k = 0; k < d; k++) dst[k] = row[k] - mean;
            }
            norms[i] = Math.Sqrt(DotSimd(dst, dst));
        }
        return prepared;
    }

    private static double CosineFromDot(double dot, double na, double nb)
    {
        var r = 1.0 - dot / (na * nb);
        return double.IsNaN(r) ? r : Math.Clamp(r, 0.0, 2.0);
    }

    /// <summary>
    /// Distances between each row of <paramref name="xa"/> and each row of <paramref name="xb"/>,
    /// shape (na, nb).
    /// </summary>
    /// <param name="xa">First set, shape (na, d).</param>
    /// <param name="xb">Second set, shape (nb, d).</param>
    /// <param name="metric">Metric.</param>
    /// <param name="p">Minkowski p.</param>
    /// <param name="w">Optional per-coordinate weights.</param>
    /// <param name="v">SEuclidean variances; default: column variances of both sets stacked.</param>
    /// <param name="vi">Mahalanobis inverse covariance; default from both sets stacked.</param>
    /// <param name="logBase">Jensen-Shannon logarithm base.</param>
    /// <param name="workers">Threads, -1 for all cores.</param>
    public static NdArray Cdist(NdArray xa, NdArray xb, DistanceMetric metric = DistanceMetric.Euclidean, double p = 2,
        double[]? w = null, double[]? v = null, NdArray? vi = null, double? logBase = null, int workers = -1)
    {
        var a = SpatialInput.Copy(xa, out var na, out var d, nameof(xa));
        var b = SpatialInput.Copy(xb, out var nb, out var db, nameof(xb));
        if (xa.Rank != 2 || xb.Rank != 2) throw new ArgumentException("XA and XB must be 2-dimensional arrays.");
        if (d != db) throw new ArgumentException("XA and XB must have the same number of columns.");
        var wv = CheckWeights(w, d);
        var result = new double[(long)na * nb];
        long work = (long)na * nb * Math.Max(d, 1);

        if ((metric == DistanceMetric.Euclidean || metric == DistanceMetric.SqEuclidean || (metric == DistanceMetric.Minkowski && p == 2)) && wv.Length == 0)
        {
            var root = metric != DistanceMetric.SqEuclidean;
            // Walk B in blocks of ~128 KB so a block stays in L2 while every A row passes over it.
            var block = Math.Max(16, (16384 / Math.Max(d, 1)) & ~3);
            var chunk = Math.Max(1, Math.Min(64, na / (4 * Environment.ProcessorCount) + 1));
            var chunks = (na + chunk - 1) / chunk;
            Rows(chunks, work, workers, c =>
            {
                var i0 = c * chunk; var i1 = Math.Min(na, i0 + chunk);
                for (var j0 = 0; j0 < nb; j0 += block)
                {
                    var j1 = Math.Min(nb, j0 + block);
                    for (var i = i0; i < i1; i++)
                    {
                        var dst = result.AsSpan(i * nb + j0, j1 - j0);
                        SqEuclideanRowBlock(a.AsSpan(i * d, d), b, d, j0, j1, dst);
                        if (root) for (var k = 0; k < dst.Length; k++) dst[k] = Math.Sqrt(dst[k]);
                    }
                }
            });
            return new NdArray(result, na, nb);
        }
        if ((metric == DistanceMetric.Cosine || metric == DistanceMetric.Correlation) && wv.Length == 0)
        {
            var center = metric == DistanceMetric.Correlation;
            var pa = PrepareCosine(a, na, d, center, out var nra);
            var pb = PrepareCosine(b, nb, d, center, out var nrb);
            Rows(na, work, workers, i =>
            {
                var ra = pa.AsSpan(i * d, d);
                for (var j = 0; j < nb; j++) result[(long)i * nb + j] = CosineFromDot(DotSimd(ra, pb.AsSpan(j * d, d)), nra[i], nrb[j]);
            });
            return new NdArray(result, na, nb);
        }

        double[]? vv = null, vim = null;
        if (metric is DistanceMetric.SEuclidean or DistanceMetric.Mahalanobis && (metric == DistanceMetric.SEuclidean ? v is null : vi is null))
        {
            var stacked = new double[a.Length + b.Length];
            a.CopyTo(stacked, 0); b.CopyTo(stacked, a.Length);
            if (metric == DistanceMetric.SEuclidean) vv = ColumnVariance(stacked, na + nb, d);
            else vim = InverseCovariance(stacked, na + nb, d);
        }
        if (metric == DistanceMetric.SEuclidean) vv ??= v;
        if (metric == DistanceMetric.Mahalanobis)
        {
            vim ??= vi!.AsContiguous().ToArray();
            if (vim.Length != d * d) throw new ArgumentException("VI must be a d x d matrix.", nameof(vi));
        }
        var kernel = Kernel(metric, p, wv, vv, vim, logBase);
        return CdistWith(a, na, b, nb, d, kernel, workers, result);
    }

    /// <summary>Rectangular distance matrix with a caller-supplied metric.</summary>
    public static NdArray Cdist(NdArray xa, NdArray xb, Func<ReadOnlySpan<double>, ReadOnlySpan<double>, double> metric, int workers = -1)
    {
        ArgumentNullException.ThrowIfNull(metric);
        var a = SpatialInput.Copy(xa, out var na, out var d, nameof(xa));
        var b = SpatialInput.Copy(xb, out var nb, out var db, nameof(xb));
        if (d != db) throw new ArgumentException("XA and XB must have the same number of columns.");
        return CdistWith(a, na, b, nb, d, metric, workers, new double[(long)na * nb]);
    }

    /// <summary>Rectangular distance matrix between two rectangular arrays.</summary>
    public static NdArray Cdist(double[,] xa, double[,] xb, DistanceMetric metric = DistanceMetric.Euclidean, double p = 2, double[]? w = null, int workers = -1)
        => Cdist(SpatialInput.ToNd(xa), SpatialInput.ToNd(xb), metric, p, w, workers: workers);

    private static NdArray CdistWith(double[] a, int na, double[] b, int nb, int d,
        Func<ReadOnlySpan<double>, ReadOnlySpan<double>, double> kernel, int workers, double[] result)
    {
        long work = (long)na * nb * Math.Max(d, 1) * 4;
        Rows(na, work, workers, i =>
        {
            var ra = a.AsSpan(i * d, d);
            for (var j = 0; j < nb; j++) result[(long)i * nb + j] = kernel(ra, b.AsSpan(j * d, d));
        });
        return new NdArray(result, na, nb);
    }

    /// <summary>
    /// Minkowski distance matrix between two point sets (scipy.spatial.distance_matrix).
    /// </summary>
    public static NdArray DistanceMatrix(NdArray x, NdArray y, double p = 2)
        => Cdist(x, y, DistanceMetric.Minkowski, p);

    // ================================================================== condensed <-> square

    /// <summary>Converts a condensed distance vector into the symmetric square matrix with a zero diagonal.</summary>
    public static NdArray Squareform(double[] condensed, bool checks = true)
    {
        ArgumentNullException.ThrowIfNull(condensed);
        var n = NumObsY(condensed.LongLength);
        var m = new double[n * n];
        long k = 0;
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
            {
                var v = condensed[k++];
                m[i * n + j] = v; m[j * n + i] = v;
            }
        return new NdArray(m, n, n);
    }

    /// <summary>
    /// Converts a square distance matrix into its condensed upper-triangle vector. With
    /// <paramref name="checks"/>, the matrix must be exactly symmetric with an exactly zero
    /// diagonal (scipy's checks use zero tolerance too).
    /// </summary>
    public static double[] Squareform(NdArray matrix, bool checks = true)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        if (matrix.Rank != 2 || matrix.Shape[0] != matrix.Shape[1]) throw new ArgumentException("The matrix argument must be square.", nameof(matrix));
        var n = matrix.Shape[0];
        var m = matrix.AsContiguous().AsSpan();
        if (checks) IsValidDm(matrix, throwOnError: true);
        var result = new double[(long)n * (n - 1) / 2];
        long k = 0;
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++) result[k++] = m[i * n + j];
        return result;
    }

    /// <summary>True when the matrix is square, exactly symmetric and has an exactly zero diagonal (scipy's tolerance-zero checks).</summary>
    public static bool IsValidDm(NdArray matrix, bool throwOnError = false)
    {
        string? error = null;
        if (matrix.Rank != 2 || matrix.Shape[0] != matrix.Shape[1]) error = "Distance matrix must be 2-dimensional and square.";
        else
        {
            var n = matrix.Shape[0];
            var m = matrix.AsContiguous().AsSpan();
            for (var i = 0; i < n && error is null; i++)
            {
                if (m[i * n + i] != 0) error = "Distance matrix diagonal must be zero.";
                for (var j = i + 1; j < n && error is null; j++)
                    if (m[i * n + j] != m[j * n + i]) error = "Distance matrix must be symmetric.";
            }
        }
        if (error is not null && throwOnError) throw new ArgumentException(error, nameof(matrix));
        return error is null;
    }

    /// <summary>True when the length is n(n-1)/2 for some integer n.</summary>
    public static bool IsValidY(double[] condensed, bool throwOnError = false)
    {
        try { NumObsY(condensed.LongLength); return true; }
        catch (ArgumentException) when (!throwOnError) { return false; }
    }

    /// <summary>Number of observations behind a condensed matrix of the given length.</summary>
    public static int NumObsY(long length)
    {
        if (length == 0) return 1;
        var n = (long)Math.Ceiling(Math.Sqrt(length * 2.0));
        if (n * (n - 1) / 2 != length)
            throw new ArgumentException("Incompatible vector size. It must be a binomial coefficient n choose 2 for some integer n >= 2.");
        return (int)n;
    }
}
