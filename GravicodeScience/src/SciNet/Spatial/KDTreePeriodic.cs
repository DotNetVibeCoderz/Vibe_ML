using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Spatial;

public sealed partial class KDTree
{
    private readonly double[]? _box; // per dimension: period L, or +inf for an open dimension

    /// <summary>Periodic box sizes (scipy's <c>boxsize</c>), null for an ordinary tree.</summary>
    public double[]? BoxSize => _box is null ? null : (double[])_box.Clone();

    /// <summary>
    /// Builds a tree with periodic (toroidal) boundaries, scipy's <c>boxsize</c>.
    /// </summary>
    /// <param name="data">Points, shape (n, m); in every periodic dimension each coordinate must
    /// lie in [0, L), as scipy requires.</param>
    /// <param name="boxsize">Period per dimension (length m), or one value for all; a value of
    /// +∞ leaves that dimension open.</param>
    /// <param name="leafSize">Maximum points per leaf.</param>
    /// <param name="compactNodes">Shrink each node's box to its data.</param>
    /// <param name="balancedTree">Median splits when true, sliding midpoint when false.</param>
    /// <remarks>
    /// <para>Distances use the minimum image: per periodic coordinate, <c>min(|x−y|, L−|x−y|)</c>,
    /// with query points wrapped into [0, L) first, as cKDTree does.</para>
    /// <para><b>How.</b> The tree itself is the ordinary one over the wrapped data. A query is
    /// answered by querying the images <c>x + sL</c>, s ∈ {−1, 0, 1} per periodic dimension, but
    /// only the images the search radius can reach from x (a shift is needed only when x lies
    /// within the radius of a face of the box); a k-nearest query first searches x itself and
    /// uses its k-th distance as that radius. Candidates are merged with their minimum-image
    /// distances. This leaves the non-periodic kernels exactly as they were, at the price of up
    /// to 3^m searches for a point near a corner of the box with a large radius.</para>
    /// </remarks>
    public KDTree(NdArray data, double[] boxsize, int leafSize = 10, bool compactNodes = true, bool balancedTree = true)
        : this(data, leafSize, compactNodes, balancedTree)
    {
        ArgumentNullException.ThrowIfNull(boxsize);
        if (boxsize.Length == 1 && M != 1) { var b = boxsize[0]; boxsize = new double[M]; Array.Fill(boxsize, b); }
        if (boxsize.Length != M) throw new ArgumentException($"boxsize must have one value per dimension ({M}).", nameof(boxsize));
        foreach (var b in boxsize)
            if (!(b > 0)) throw new ArgumentException("boxsize values must be positive (or +inf for an open dimension).", nameof(boxsize));
        for (var i = 0; i < N; i++)
            for (var j = 0; j < M; j++)
            {
                var v = _data[i * M + j];
                if (double.IsFinite(boxsize[j]) && (v < 0 || v >= boxsize[j]))
                    throw new ArgumentException(v < 0 ? "Negative input data are outside of the periodic box." : "Some input data are greater than the size of the periodic box.");
            }
        _box = (double[])boxsize.Clone();
    }

    /// <summary>Periodic tree over a rectangular array (see the NdArray overload).</summary>
    public KDTree(double[,] data, double[] boxsize, int leafSize = 10, bool compactNodes = true, bool balancedTree = true)
        : this(SpatialInput.ToNd(data), boxsize, leafSize, compactNodes, balancedTree) { }

    private void Wrap(ReadOnlySpan<double> x, Span<double> w)
    {
        var box = _box!;
        for (var j = 0; j < M; j++)
        {
            var v = x[j];
            var L = box[j];
            if (double.IsFinite(L))
            {
                v -= Math.Floor(v / L) * L;
                if (v >= L) v -= L; // rounding of the floor product
                if (v < 0) v = 0;
            }
            w[j] = v;
        }
    }

    /// <summary>Minimum-image distance between a (wrapped) point and data row j.</summary>
    internal double PeriodicDistance(ReadOnlySpan<double> x, int j, double p)
    {
        Span<double> w = stackalloc double[M];
        Wrap(x, w);
        return PeriodicDistanceWrapped(w, j, p);
    }

    private double PeriodicDistanceWrapped(ReadOnlySpan<double> w, int row, double p)
    {
        var box = _box!;
        double acc = 0;
        for (var j = 0; j < M; j++)
        {
            var d = Math.Abs(w[j] - _data[row * M + j]);
            if (double.IsFinite(box[j]) && box[j] - d < d) d = box[j] - d;
            if (p == 2) acc += d * d;
            else if (p == 1) acc += d;
            else if (double.IsPositiveInfinity(p)) acc = Math.Max(acc, d);
            else acc += Math.Pow(d, p);
        }
        return p == 2 ? Math.Sqrt(acc) : p == 1 || double.IsPositiveInfinity(p) ? acc : Math.Pow(acc, 1 / p);
    }

    /// <summary>Calls <paramref name="visit"/> on every image of w that the radius can reach.</summary>
    private void ForEachImage(ReadOnlySpan<double> w, double radius, bool skipIdentity, Action<double[]> visit)
    {
        var box = _box!;
        var m = M;
        var shifts = new List<int>[m];
        for (var j = 0; j < m; j++)
        {
            shifts[j] = [0];
            if (!double.IsFinite(box[j])) continue;
            if (w[j] <= radius) shifts[j].Add(1);
            if (w[j] >= box[j] - radius) shifts[j].Add(-1);
        }
        var img = new double[m];
        var pick = new int[m];
        while (true)
        {
            var identity = true;
            for (var j = 0; j < m; j++)
            {
                var s = shifts[j][pick[j]];
                img[j] = w[j] + s * box[j];
                if (s != 0) identity = false;
            }
            if (!(identity && skipIdentity)) visit((double[])img.Clone());
            var t = 0;
            while (t < m && ++pick[t] == shifts[t].Count) { pick[t] = 0; t++; }
            if (t == m) return;
        }
    }

    private void PeriodicBall(ReadOnlySpan<double> x, double r, double p, double eps, List<int> result)
    {
        Span<double> w = stackalloc double[M];
        Wrap(x, w);
        var seen = new HashSet<int>();
        var tmp = new List<int>();
        ForEachImage(w, r * (1 + eps), false, img =>
        {
            tmp.Clear();
            BallFlat(img, r, p, eps, tmp);
            foreach (var j in tmp)
                if (seen.Add(j)) result.Add(j);
        });
    }

    private void PeriodicKnn(double[] queries, int q, int k, double eps, double p, double ub, double[] dist, int[] idx, int workers)
    {
        var m = M;
        var w = Workers(workers);
        void One(int i)
        {
            var wq = new double[m];
            Wrap(queries.AsSpan(i * m, m), wq);
            // Search x itself, then every image within its k-th distance.
            var d0 = new double[k]; var i0 = new int[k];
            KnnFlat(wq, k, eps, p, ub, d0, i0);
            var radius = Math.Min(d0[k - 1], ub);
            var cand = new Dictionary<int, double>();
            void Take(int[] ids)
            {
                foreach (var j in ids)
                    if (j < N && !cand.ContainsKey(j)) cand[j] = PeriodicDistanceWrapped(wq, j, p);
            }
            Take(i0);
            var bound = double.IsPositiveInfinity(radius) ? radius : Math.BitIncrement(radius);
            ForEachImage(wq, radius, true, img =>
            {
                var di = new double[k]; var ii = new int[k];
                KnnFlat(img, k, eps, p, bound, di, ii);
                Take(ii);
            });
            var sorted = cand.Where(e => e.Value < ub).OrderBy(e => e.Value).ThenBy(e => e.Key).Take(k).ToArray();
            for (var t = 0; t < k; t++)
            {
                if (t < sorted.Length) { dist[i * k + t] = sorted[t].Value; idx[i * k + t] = sorted[t].Key; }
                else { dist[i * k + t] = double.PositiveInfinity; idx[i * k + t] = N; }
            }
        }
        if (w == 1 || q < ParallelThreshold) for (var i = 0; i < q; i++) One(i);
        else Parallel.For(0, q, new ParallelOptions { MaxDegreeOfParallelism = w }, One);
    }

    /// <summary>The ordinary (non-periodic) k-nearest search of one point.</summary>
    private void KnnFlat(double[] x, int k, double eps, double p, double ub, double[] dist, int[] idx)
    {
        var epsFac = EpsFactor(p, eps);
        var heap = new KnnHeap(k);
        if (p == 2) KnnOne(new SpatialNormL2(), x, heap, epsFac, new SpatialNormL2().ToPower(ub), dist, idx);
        else if (p == 1) KnnOne(new SpatialNormL1(), x, heap, epsFac, ub, dist, idx);
        else if (double.IsPositiveInfinity(p)) KnnOne(new SpatialNormLInf(), x, heap, epsFac, ub, dist, idx);
        else { var nrm = new SpatialNormLp(p); KnnOne(nrm, x, heap, epsFac, nrm.ToPower(ub), dist, idx); }
    }

    // ------------------------------------------------------------------ weighted / periodic counts

    /// <summary>
    /// Weighted pair counts (scipy's <c>count_neighbors</c> with <c>weights</c>): the sum of
    /// <c>selfWeights[i] · otherWeights[j]</c> over pairs within each radius; a null side counts
    /// each of its points once. Works with periodic trees.
    /// </summary>
    /// <remarks>Each point of this tree takes one ball query of the other tree at the largest
    /// radius and bins its neighbours, so the cost grows with the number of pairs within that
    /// radius — unlike the unweighted overload, which credits whole nodes at once.</remarks>
    public double[] CountNeighbors(KDTree other, double[] r, double[]? selfWeights, double[]? otherWeights, double p = 2, bool cumulative = true)
    {
        ArgumentNullException.ThrowIfNull(other);
        ArgumentNullException.ThrowIfNull(r);
        CheckP(p);
        if (other.M != M) throw new ArgumentException("Both trees must have the same dimension.", nameof(other));
        foreach (var rr in r) if (double.IsNaN(rr)) throw new ArgumentException("Radii must not be NaN.", nameof(r));
        if (selfWeights is not null && selfWeights.Length != N) throw new ArgumentException("selfWeights must have one value per point of this tree.");
        if (otherWeights is not null && otherWeights.Length != other.N) throw new ArgumentException("otherWeights must have one value per point of the other tree.");
        return CountNeighborsGeneral(other, r, selfWeights, otherWeights, p, cumulative);
    }

    private double[] CountNeighborsGeneral(KDTree other, double[] r, double[]? sw, double[]? ow, double p, bool cumulative)
    {
        var nr = r.Length;
        var order = new int[nr];
        for (var i = 0; i < nr; i++) order[i] = i;
        var sorted = (double[])r.Clone();
        Array.Sort(sorted, order);
        var rmax = nr == 0 ? -1 : sorted[^1];
        var diff = new double[nr + 1];
        if (rmax >= 0 && N > 0 && other.N > 0)
        {
            var m = M;
            var gate = new object();
            Parallel.For(0, N, () => (new double[nr + 1], new List<int>()), (i, _, state) =>
            {
                var (local, list) = state;
                list.Clear();
                var x = _data.AsSpan(i * m, m);
                other.Ball(x, rmax, p, 0, list);
                var wi = sw?[i] ?? 1.0;
                foreach (var j in list)
                {
                    var d = other._box is null ? Minkowski(x, other._data.AsSpan(j * m, m), p) : other.PeriodicDistance(x, j, p);
                    var b = Array.BinarySearch(sorted, d);
                    if (b < 0) b = ~b;
                    else while (b > 0 && sorted[b - 1] == d) b--;
                    local[b] += wi * (ow?[j] ?? 1.0);
                }
                return state;
            }, state => { lock (gate) for (var t = 0; t <= nr; t++) diff[t] += state.Item1[t]; });
        }
        var cum = new double[nr];
        double run = 0;
        for (var i = 0; i < nr; i++) { run += diff[i]; cum[i] = run; }
        var result = new double[nr];
        for (var i = 0; i < nr; i++) result[order[i]] = cum[i];
        if (!cumulative)
        {
            var bins = new double[nr];
            for (var i = 0; i < nr; i++) bins[i] = i == 0 ? result[0] : result[i] - result[i - 1];
            return bins;
        }
        return result;
    }
}
