using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Signal
{
    /// <summary>
    /// 1-D median filter with zero padding (<c>scipy.signal.medfilt</c>).
    /// </summary>
    /// <remarks>
    /// A sorted copy of the window slides along the signal: each step removes the outgoing sample
    /// and inserts the incoming one by binary search plus a block move, so a kernel of k costs
    /// O(log k) comparisons and one short memmove per sample instead of a fresh selection.
    /// Like scipy, the window reaches past the ends into zeros, so the first and last
    /// <c>k/2</c> outputs are pulled towards 0. NaN inputs give unspecified results.
    /// </remarks>
    /// <param name="volume">Signal.</param>
    /// <param name="kernelSize">Odd window length.</param>
    public static double[] Medfilt(double[] volume, int kernelSize = 3)
    {
        ArgumentNullException.ThrowIfNull(volume);
        if (kernelSize < 1 || kernelSize % 2 != 1) throw new ArgumentException("Each element of kernel_size should be odd.", nameof(kernelSize));
        var y = new double[volume.Length];
        SlidingMedian(volume, y, kernelSize);
        return y;
    }

    /// <summary>
    /// n-D median filter with zero padding (<c>scipy.signal.medfilt</c>); <paramref name="kernelSize"/>
    /// gives an odd length per axis (default 3 everywhere).
    /// </summary>
    public static NdArray Medfilt(NdArray volume, int[]? kernelSize = null)
    {
        ArgumentNullException.ThrowIfNull(volume);
        var rank = volume.Rank;
        var k = kernelSize ?? Enumerable.Repeat(3, rank).ToArray();
        if (k.Length == 1 && rank > 1) k = Enumerable.Repeat(k[0], rank).ToArray();
        if (k.Length != rank) throw new ArgumentException("kernel_size must have one entry per dimension.", nameof(kernelSize));
        foreach (var v in k) if (v < 1 || v % 2 != 1) throw new ArgumentException("Each element of kernel_size should be odd.", nameof(kernelSize));
        if (rank == 1) { var r = Medfilt(volume.AsContiguous().ToArray(), k[0]); return new NdArray(r, r.Length); }
        if (rank == 2) return Medfilt2d(volume, k[0], k[1]);
        var size = 1;
        foreach (var v in k) size *= v;
        var mask = new double[size];
        Array.Fill(mask, 1.0);
        return OrderFilter(volume, new NdArray(mask, k), size / 2);
    }

    /// <summary>
    /// 2-D median filter with zero padding (<c>scipy.signal.medfilt2d</c>), square kernel.
    /// </summary>
    public static NdArray Medfilt2d(NdArray input, int kernelSize = 3) => Medfilt2d(input, kernelSize, kernelSize);

    /// <summary>
    /// 2-D median filter with zero padding and a <paramref name="kRows"/> × <paramref name="kCols"/>
    /// kernel. Each row of outputs slides a sorted window right one column at a time (remove a
    /// column, insert a column); rows run in parallel.
    /// </summary>
    public static NdArray Medfilt2d(NdArray input, int kRows, int kCols)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.Rank != 2) throw new ArgumentException("medfilt2d needs a 2-D array.", nameof(input));
        if (kRows < 1 || kCols < 1 || kRows % 2 != 1 || kCols % 2 != 1) throw new ArgumentException("Each element of kernel_size should be odd.");
        var rows = input.Shape[0];
        var cols = input.Shape[1];
        var a = input.AsContiguous().ToArray();
        var y = new double[a.Length];
        var hr = kRows / 2;
        var hc = kCols / 2;
        var k = kRows * kCols;
        double At(int i, int j) => i < 0 || i >= rows || j < 0 || j >= cols ? 0.0 : a[i * cols + j];
        void Row(int i, double[] win)
        {
            var cnt = 0;
            for (var di = -hr; di <= hr; di++)
                for (var dj = -hc; dj <= hc; dj++) win[cnt++] = At(i + di, dj);
            Array.Sort(win, 0, k);
            for (var j = 0; j < cols; j++)
            {
                y[i * cols + j] = win[k / 2];
                if (j == cols - 1) break;
                for (var di = -hr; di <= hr; di++)
                {
                    SortedReplace(win, k, At(i + di, j - hc), At(i + di, j + hc + 1));
                }
            }
        }
        if ((long)rows * cols * k > 1 << 16 && rows > 1)
            Parallel.For(0, rows, () => new double[k], (i, _, w) => { Row(i, w); return w; }, _ => { });
        else
        {
            var w = new double[k];
            for (var i = 0; i < rows; i++) Row(i, w);
        }
        return new NdArray(y, rows, cols);
    }

    /// <summary>
    /// Rank-order filter (<c>scipy.signal.order_filter</c>): each output is the
    /// <paramref name="rank"/>-th smallest of the input values under the non-zero entries of
    /// <paramref name="domain"/>, centred on the output, with zero padding.
    /// </summary>
    /// <param name="a">Input array.</param>
    /// <param name="domain">Mask with an odd length on every axis; non-zero entries select the neighbours.</param>
    /// <param name="rank">0 = minimum, count − 1 = maximum.</param>
    public static NdArray OrderFilter(NdArray a, NdArray domain, int rank)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(domain);
        var r = a.Rank;
        if (domain.Rank != r) throw new ArgumentException("domain must have the same rank as a.", nameof(domain));
        var ds = domain.Shape.ToArray();
        foreach (var v in ds) if (v % 2 != 1) throw new ArgumentException("Each dimension of domain argument should have an odd number of elements.", nameof(domain));
        var dm = domain.AsContiguous().ToArray();
        var shape = a.Shape.ToArray();
        var data = a.AsContiguous().ToArray();
        // Offsets of the selected neighbours, one int[] per neighbour.
        var offs = new List<int[]>();
        for (var f = 0; f < dm.Length; f++)
        {
            if (dm[f] == 0) continue;
            var o = new int[r];
            var rem = f;
            for (var d = r - 1; d >= 0; d--) { o[d] = rem % ds[d] - ds[d] / 2; rem /= ds[d]; }
            offs.Add(o);
        }
        var cnt = offs.Count;
        if (cnt == 0) throw new ArgumentException("domain selects no elements.", nameof(domain));
        if (rank < 0 || rank >= cnt) throw new ArgumentException("rank not within filter footprint size.", nameof(rank));
        var strides = new int[r];
        var acc = 1;
        for (var d = r - 1; d >= 0; d--) { strides[d] = acc; acc *= shape[d]; }
        var y = new double[data.Length];
        void Point(int p, double[] buf, int[] idx)
        {
            var rem = p;
            for (var d = r - 1; d >= 0; d--) { idx[d] = rem % shape[d]; rem /= shape[d]; }
            for (var q = 0; q < cnt; q++)
            {
                var o = offs[q];
                var lin = 0;
                var inside = true;
                for (var d = 0; d < r; d++)
                {
                    var c = idx[d] + o[d];
                    if (c < 0 || c >= shape[d]) { inside = false; break; }
                    lin += c * strides[d];
                }
                buf[q] = inside ? data[lin] : 0.0;
            }
            y[p] = QuickSelect(buf, cnt, rank);
        }
        if ((long)data.Length * cnt > 1 << 16)
            Parallel.For(0, data.Length, () => (new double[cnt], new int[r]), (p, _, st) => { Point(p, st.Item1, st.Item2); return st; }, _ => { });
        else
        {
            var buf = new double[cnt];
            var idx = new int[r];
            for (var p = 0; p < data.Length; p++) Point(p, buf, idx);
        }
        return new NdArray(y, shape);
    }

    /// <summary>
    /// Adaptive Wiener noise filter (<c>scipy.signal.wiener</c>): local mean and variance over a
    /// window (zero padded, windows centred as in <c>correlate(..., 'same')</c>), then
    /// <c>m + (1 − σ²_noise/σ²_local)(x − m)</c>, or the local mean where the local variance is
    /// below the noise. Window sums are separable running sums, O(1) per sample and axis.
    /// </summary>
    /// <param name="im">Input array.</param>
    /// <param name="mysize">Window length per axis (one value applies to all); default 3.</param>
    /// <param name="noise">Noise power; default the mean local variance.</param>
    public static NdArray Wiener(NdArray im, int[]? mysize = null, double? noise = null)
    {
        ArgumentNullException.ThrowIfNull(im);
        var rank = im.Rank;
        var k = mysize ?? Enumerable.Repeat(3, rank).ToArray();
        if (k.Length == 1 && rank > 1) k = Enumerable.Repeat(k[0], rank).ToArray();
        if (k.Length != rank) throw new ArgumentException("mysize must have one entry per dimension.", nameof(mysize));
        var shape = im.Shape.ToArray();
        var x = im.AsContiguous().ToArray();
        var size = 1.0;
        foreach (var v in k) size *= v;
        var sq = new double[x.Length];
        for (var i = 0; i < x.Length; i++) sq[i] = x[i] * x[i];
        var mean = BoxSum(x, shape, k);
        var msq = BoxSum(sq, shape, k);
        var lvar = new double[x.Length];
        for (var i = 0; i < x.Length; i++)
        {
            mean[i] /= size;
            lvar[i] = msq[i] / size - mean[i] * mean[i];
        }
        double nz;
        if (noise is { } given) nz = given;
        else
        {
            nz = 0;
            foreach (var v in lvar) nz += v;
            nz /= lvar.Length;
        }
        var y = new double[x.Length];
        for (var i = 0; i < x.Length; i++)
        {
            var res = (x[i] - mean[i]) * (1 - nz / lvar[i]) + mean[i];
            y[i] = lvar[i] < nz ? mean[i] : res;
        }
        return new NdArray(y, shape);
    }

    /// <summary>1-D <see cref="Wiener(NdArray, int[], double?)"/>.</summary>
    public static double[] Wiener(double[] im, int mysize = 3, double? noise = null)
        => Wiener(new NdArray((double[])im.Clone(), im.Length), [mysize], noise).AsContiguous().ToArray();

    // ------------------------------------------------------------------ helpers

    private static double[] BoxSum(double[] data, int[] shape, int[] k)
    {
        var cur = (double[])data.Clone();
        var rank = shape.Length;
        for (var axis = 0; axis < rank; axis++)
        {
            var len = shape[axis];
            var inner = 1;
            for (var d = axis + 1; d < rank; d++) inner *= shape[d];
            var outer = cur.Length / (len * inner);
            var after = (k[axis] - 1) / 2;        // samples ahead of the centre
            var before = k[axis] - 1 - after;     // samples behind
            var next = new double[cur.Length];
            var line = new double[len + 1];
            for (var o = 0; o < outer; o++)
                for (var ii = 0; ii < inner; ii++)
                {
                    var b = o * len * inner + ii;
                    line[0] = 0;
                    for (var t = 0; t < len; t++) line[t + 1] = line[t] + cur[b + t * inner];
                    for (var t = 0; t < len; t++)
                    {
                        var lo = Math.Max(0, t - before);
                        var hi = Math.Min(len - 1, t + after);
                        next[b + t * inner] = lo > hi ? 0 : line[hi + 1] - line[lo];
                    }
                }
            cur = next;
        }
        return cur;
    }

    private static void SlidingMedian(ReadOnlySpan<double> x, Span<double> y, int k)
    {
        var n = x.Length;
        if (n == 0) return;
        var h = k / 2;
        double At(ReadOnlySpan<double> s, int i) => i < 0 || i >= s.Length ? 0.0 : s[i];
        if (k <= 7)
        {
            Span<double> w = stackalloc double[k];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < k; j++) w[j] = At(x, i - h + j);
                w.Sort();
                y[i] = w[h];
            }
            return;
        }
        var xs = x.ToArray();
        var ys = new double[n];
        // Independent chunks, each seeded with a freshly sorted window, run in parallel.
        var chunks = (long)n * k < 1 << 18 ? 1 : Math.Min(Environment.ProcessorCount * 2, Math.Max(1, n / (4 * k)));
        var per = (n + chunks - 1) / chunks;
        Parallel.For(0, chunks, new ParallelOptions { MaxDegreeOfParallelism = chunks == 1 ? 1 : -1 }, c =>
        {
            var s0 = c * per;
            var s1 = Math.Min(n, s0 + per);
            if (s0 >= s1) return;
            var win = new double[k];
            for (var j = 0; j < k; j++) win[j] = At(xs, s0 + j - h);
            Array.Sort(win);
            for (var i = s0; i < s1; i++)
            {
                ys[i] = win[h];
                if (i == s1 - 1) break;
                SortedReplace(win, k, At(xs, i - h), At(xs, i + h + 1));
            }
        });
        ys.CopyTo(y);
    }

    /// <summary>
    /// Replaces one occurrence of <paramref name="outgoing"/> by <paramref name="incoming"/>, keeping
    /// the first <paramref name="k"/> entries sorted: binary-search the outgoing slot, then slide the
    /// incoming value towards its place insertion-sort style (one pass, no second search, no block
    /// copy call — on random data the slide is a third of the window on average).
    /// </summary>
    private static void SortedReplace(double[] win, int k, double outgoing, double incoming)
    {
        if (outgoing.Equals(incoming)) return;
        var r = Array.BinarySearch(win, 0, k, outgoing);
        if (r < 0)
        {
            // NaN (BinarySearch cannot find it): fall back to a linear scan.
            for (var i = 0; i < k; i++) if (win[i].Equals(outgoing)) { r = i; break; }
            if (r < 0) return;
        }
        var p = r;
        if (incoming > outgoing)
            while (p + 1 < k && win[p + 1] < incoming) { win[p] = win[p + 1]; p++; }
        else
            while (p > 0 && win[p - 1] > incoming) { win[p] = win[p - 1]; p--; }
        win[p] = incoming;
    }

    /// <summary>Hoare/Floyd–Rivest-style selection of the k-th smallest of the first n entries (reorders them).</summary>
    private static double QuickSelect(double[] a, int n, int k)
    {
        int lo = 0, hi = n - 1;
        while (hi > lo)
        {
            var pivot = a[(lo + hi) >>> 1];
            int i = lo, j = hi;
            while (i <= j)
            {
                while (a[i] < pivot) i++;
                while (a[j] > pivot) j--;
                if (i <= j) { (a[i], a[j]) = (a[j], a[i]); i++; j--; }
            }
            if (k <= j) hi = j;
            else if (k >= i) lo = i;
            else return a[k];
        }
        return a[k];
    }
}
