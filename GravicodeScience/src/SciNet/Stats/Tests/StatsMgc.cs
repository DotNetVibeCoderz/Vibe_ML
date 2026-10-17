using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Result of <see cref="Stats.MultiscaleGraphcorr(NdArray, NdArray, Func{NdArray, NdArray}?, bool, int, int?, bool)"/> (scipy's <c>MGCResult</c>).</summary>
/// <param name="Statistic">The MGC statistic, in [−1, 1].</param>
/// <param name="Pvalue">Permutation p-value, (count + 1)/(reps + 1).</param>
/// <param name="MgcMap">The local correlation at every neighbourhood scale (k, l).</param>
/// <param name="OptScale">The optimal scale, 1-based as scipy (and R's MGC).</param>
/// <param name="NullDistribution">The statistic under each permutation.</param>
public sealed record MgcResult(double Statistic, double Pvalue, NdArray MgcMap, (int K, int L) OptScale, double[] NullDistribution);

public static partial class Stats
{
    /// <summary>MGC of two 1-D samples (n paired observations); see the <see cref="NdArray"/> overload.</summary>
    public static MgcResult MultiscaleGraphcorr(double[] x, double[] y, int reps = 1000, int? seed = null, bool parallel = false)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        return MultiscaleGraphcorr(new NdArray((double[])x.Clone(), x.Length, 1), new NdArray((double[])y.Clone(), y.Length, 1), reps: reps, seed: seed, parallel: parallel);
    }

    /// <summary>
    /// Multiscale Graph Correlation (Vogelstein et al. 2019), scipy's <c>multiscale_graphcorr</c>: an
    /// independence test that computes distance correlations restricted to every pair of
    /// k- and l-nearest-neighbourhoods, then reports the largest smoothed local correlation and a
    /// permutation p-value.
    /// </summary>
    /// <param name="x">Shape (n, p).</param>
    /// <param name="y">Shape (n, q); or (m, p) for the two-sample test (rows are then pooled and y
    /// becomes the group label, as scipy's <c>_two_sample_transform</c>).</param>
    /// <param name="computeDistance">Maps an (n, p) array to its (n, n) distance matrix; Euclidean when null.</param>
    /// <param name="precomputed">x and y are already distance matrices (scipy's <c>compute_distance=None</c>).</param>
    /// <param name="reps">Permutations for the p-value (scipy's default 1000).</param>
    /// <param name="seed">Seed of the permutations.</param>
    /// <param name="parallel">Run permutations on several threads; the result does not depend on it.</param>
    /// <remarks>
    /// Memory and time are O(n²) per statistic (distance, rank and centred matrices plus an
    /// (n × n) local-covariance table), so the test is for n up to a few thousand; each permutation
    /// re-ranks nothing (y's ranks permute with it) but recomputes the O(n²) covariance table. The
    /// p-value is Monte Carlo and differs from scipy's run to run; the statistic, map and optimal
    /// scale are deterministic and match.
    /// </remarks>
    public static MgcResult MultiscaleGraphcorr(NdArray x, NdArray y, Func<NdArray, NdArray>? computeDistance = null, bool precomputed = false,
        int reps = 1000, int? seed = null, bool parallel = false)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (x.Shape.Length == 1) x = new NdArray(x.AsContiguous().ToArray(), x.Shape[0], 1);
        if (y.Shape.Length == 1) y = new NdArray(y.AsContiguous().ToArray(), y.Shape[0], 1);
        if (x.Shape.Length != 2 || y.Shape.Length != 2) throw new ArgumentException("x and y must be 1-D or 2-D arrays.");
        foreach (var v in x.AsContiguous().AsSpan()) if (!double.IsFinite(v)) throw new ArgumentException("Inputs contain NaN or infinities.", nameof(x));
        foreach (var v in y.AsContiguous().AsSpan()) if (!double.IsFinite(v)) throw new ArgumentException("Inputs contain NaN or infinities.", nameof(y));
        if (reps < 0) throw new ArgumentException("Number of reps must be an integer greater than 0.", nameof(reps));
        int nx = x.Shape[0], ny = y.Shape[0];
        if (nx != ny)
        {
            if (x.Shape[1] != y.Shape[1]) throw new ArgumentException("Shape mismatch, x and y must have shape [n, p] and [n, q] or have shape [n, p] and [m, p].");
            if (precomputed) throw new ArgumentException("Cannot run if inputs are distance matrices");
            var p = x.Shape[1];
            var pooled = new double[(nx + ny) * p];
            x.AsContiguous().AsSpan().CopyTo(pooled);
            y.AsContiguous().AsSpan().CopyTo(pooled.AsSpan(nx * p));
            var label = new double[nx + ny];
            for (var i = nx; i < nx + ny; i++) label[i] = 1;
            x = new NdArray(pooled, nx + ny, p);
            y = new NdArray(label, nx + ny, 1);
        }
        if (x.Shape[0] < 5) throw new ArgumentException("MGC requires at least 5 samples to give reasonable results.");
        double[] dx, dy;
        var n = x.Shape[0];
        if (precomputed)
        {
            if (x.Shape[1] != n || y.Shape[1] != n) throw new ArgumentException("Distance matrices must be square.");
            dx = x.AsContiguous().ToArray();
            dy = y.AsContiguous().ToArray();
        }
        else if (computeDistance != null)
        {
            dx = computeDistance(x).AsContiguous().ToArray();
            dy = computeDistance(y).AsContiguous().ToArray();
            if (dx.Length != n * n || dy.Length != n * n) throw new ArgumentException("computeDistance must return an (n, n) matrix.", nameof(computeDistance));
        }
        else
        {
            dx = MgcEuclidean(x);
            dy = MgcEuclidean(y);
        }
        var (stat, map, mapRows, mapCols, opt) = MgcStat(dx, dy, n);
        var master = new GraviRandom(seed ?? Random.Shared.Next());
        var seeds = new ulong[reps];
        for (var b = 0; b < reps; b++) seeds[b] = master.NextUInt64();
        var nulls = new double[reps];
        void One(int b)
        {
            var rng = new GraviRandom(seeds[b]);
            var perm = new int[n];
            for (var i = 0; i < n; i++) perm[i] = i;
            for (var i = n - 1; i > 0; i--) { var j = rng.Next(i + 1); (perm[i], perm[j]) = (perm[j], perm[i]); }
            var py = new double[n * n];
            for (var i = 0; i < n; i++)
                for (var j = 0; j < n; j++) py[i * n + j] = dy[perm[i] * n + perm[j]];
            nulls[b] = MgcStat(dx, py, n).Stat;
        }
        if (parallel) Parallel.For(0, reps, One);
        else for (var b = 0; b < reps; b++) One(b);
        var count = 0;
        foreach (var v in nulls) if (v >= stat) count++;
        var pv = reps == 0 ? double.NaN : (count + 1.0) / (reps + 1.0);
        return new MgcResult(stat, pv, new NdArray(map, mapRows, mapCols), opt, nulls);
    }

    private static double[] MgcEuclidean(NdArray a)
    {
        int n = a.Shape[0], p = a.Shape[1];
        var s = a.AsContiguous().ToArray();
        var d = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
            {
                var t = 0.0;
                for (var k = 0; k < p; k++) { var u = s[i * p + k] - s[j * p + k]; t += u * u; }
                d[i * n + j] = d[j * n + i] = Math.Sqrt(t);
            }
        return d;
    }

    /// <summary>Column-wise dense ranks of a distance matrix, 0-based (ties share a rank and no rank is skipped, so a
    /// label distance in the two-sample test has just two ranks and the map two columns), as [i, j] = rank of d[i, j] within column j.</summary>
    private static int[] MgcColumnRanks(double[] d, int n)
    {
        var r = new int[n * n];
        var idx = new int[n];
        var col = new double[n];
        for (var j = 0; j < n; j++)
        {
            for (var i = 0; i < n; i++) { idx[i] = i; col[i] = d[i * n + j]; }
            Array.Sort((double[])col.Clone(), idx);
            var rank = 0;
            for (var a = 0; a < n;)
            {
                var b = a;
                while (b + 1 < n && col[idx[b + 1]] == col[idx[a]]) b++;
                for (var t = a; t <= b; t++) r[idx[t] * n + j] = rank; // dense rank, 0-based
                rank++;
                a = b + 1;
            }
        }
        return r;
    }

    /// <summary>The 'mgc' centring: subtract column sums/(n − 1), zero the diagonal.</summary>
    private static double[] MgcCenter(double[] d, int n)
    {
        var c = new double[n * n];
        var colMean = new double[n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++) colMean[j] += d[i * n + j];
        for (var j = 0; j < n; j++) colMean[j] /= n - 1;
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++) c[i * n + j] = i == j ? 0.0 : d[i * n + j] - colMean[j];
        return c;
    }

    /// <summary>
    /// scipy's <c>_local_covariance</c>: entry (k, l) sums a[i, j]·b[j, i] over pairs with rank_a[i, j] ≤ k and
    /// rank_b[j, i] ≤ l — scattered into a table and then cumulatively summed both ways — minus the
    /// product of the matching local sums of a and b over n².
    /// </summary>
    private static double[] MgcLocalCov(double[] a, double[] b, int[] ra, int[] rb, int n, int rows, int cols)
    {
        var t = new double[rows * cols];
        var ex = new double[rows];
        var ey = new double[cols];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                int k = ra[i * n + j], l = rb[j * n + i];
                double av = a[i * n + j], bv = b[j * n + i];
                t[k * cols + l] += av * bv;
                ex[k] += av;
                ey[l] += bv;
            }
        for (var r = 1; r < rows; r++)
            for (var c = 0; c < cols; c++) t[r * cols + c] += t[(r - 1) * cols + c];
        for (var r = 0; r < rows; r++)
            for (var c = 1; c < cols; c++) t[r * cols + c] += t[r * cols + c - 1];
        for (var r = 1; r < rows; r++) ex[r] += ex[r - 1];
        for (var c = 1; c < cols; c++) ey[c] += ey[c - 1];
        // Local centring: the neighbourhood means change with the scale, so they are subtracted per (k, l).
        double nn = (double)n * n;
        for (var r = 0; r < rows; r++)
            for (var c = 0; c < cols; c++) t[r * cols + c] -= ex[r] * ey[c] / nn;
        return t;
    }

    private static (double Stat, double[] Map, int Rows, int Cols, (int, int) Opt) MgcStat(double[] dx, double[] dy, int n)
    {
        var rx = MgcColumnRanks(dx, n);
        var ry = MgcColumnRanks(dy, n);
        var cx = MgcCenter(dx, n);
        var cy = MgcCenter(dy, n);
        int rows = 0, cols = 0;
        foreach (var v in rx) rows = Math.Max(rows, v + 1);
        foreach (var v in ry) cols = Math.Max(cols, v + 1);
        var cov = MgcLocalCov(cx, cy, rx, ry, n, rows, cols);
        var vx = MgcLocalCov(cx, cx, rx, rx, n, rows, rows);
        var vy = MgcLocalCov(cy, cy, ry, ry, n, cols, cols);
        var map = new double[rows * cols];
        for (var k = 0; k < rows; k++)
        {
            var a = vx[k * rows + k];
            for (var l = 0; l < cols; l++)
            {
                var b = vy[l * cols + l];
                if (a <= 0 || b <= 0) { map[k * cols + l] = 0; continue; }
                var c = cov[k * cols + l] / Math.Sqrt(a * b);
                map[k * cols + l] = double.IsNaN(c) ? c : Math.Min(c, 1.0);
            }
        }
        var stat = map[rows * cols - 1];
        if (rows == 1 || cols == 1) return (stat, map, rows, cols, (rows, cols));
        // Threshold (scipy's _threshold_mgc_map): Beta approximation of the null, then the largest
        // 4-connected region above it.
        var samp = n - 1;
        var perSig = 1 - 0.02 / samp;
        var shape = samp * (samp - 3) / 4.0 - 0.5;
        var thr = new BetaDistribution(shape, shape).Ppf(perSig) * 2 - 1;
        thr = Math.Max(thr, stat);
        var labels = new int[rows * cols];
        var sizes = new List<int> { 0 };
        var stack = new Stack<int>();
        for (var s = 0; s < labels.Length; s++)
        {
            if (labels[s] != 0 || !(map[s] > thr)) continue;
            var lab = sizes.Count;
            sizes.Add(0);
            labels[s] = lab;
            stack.Push(s);
            while (stack.Count > 0)
            {
                var c = stack.Pop();
                sizes[lab]++;
                int r = c / cols, q = c % cols;
                void Visit(int rr, int qq)
                {
                    if (rr < 0 || rr >= rows || qq < 0 || qq >= cols) return;
                    var id = rr * cols + qq;
                    if (labels[id] == 0 && map[id] > thr) { labels[id] = lab; stack.Push(id); }
                }
                Visit(r - 1, q); Visit(r + 1, q); Visit(r, q - 1); Visit(r, q + 1);
            }
        }
        var opt = (rows, cols);
        if (sizes.Count == 1) return (stat, map, rows, cols, opt);
        var best = 1;
        for (var l = 2; l < sizes.Count; l++) if (sizes[l] > sizes[best]) best = l;
        // _smooth_mgc_map: only a region large enough counts.
        if (sizes[best] >= Math.Ceiling(0.02 * Math.Max(rows, cols)) * Math.Min(rows, cols))
        {
            var maxCorr = double.NegativeInfinity;
            for (var s = 0; s < map.Length; s++) if (labels[s] == best) maxCorr = Math.Max(maxCorr, map[s]);
            if (maxCorr >= stat)
            {
                stat = maxCorr;
                var last = -1;
                for (var s = 0; s < map.Length; s++) if (labels[s] == best && map[s] >= maxCorr) last = s;
                opt = (last / cols + 1, last % cols + 1);
            }
        }
        return (stat, map, rows, cols, opt);
    }
}
