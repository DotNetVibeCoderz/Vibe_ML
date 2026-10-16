using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Index handling at the ends for <see cref="Signal.ArgRelMax(double[], int, ArgRelMode)"/>.</summary>
public enum ArgRelMode
{
    /// <summary>Indices are clipped to the ends, so an end sample is compared with itself (and fails a strict comparison).</summary>
    Clip,
    /// <summary>The signal is treated as periodic.</summary>
    Wrap,
}

public static partial class Signal
{
    // ================================================================== find_peaks

    /// <summary>
    /// Local maxima filtered by height, threshold, distance, prominence, width and plateau size
    /// (<c>scipy.signal.find_peaks</c>), applied in scipy's order so the same peaks survive.
    /// </summary>
    /// <remarks>
    /// <para>A peak is a sample (or the middle of a flat plateau, rounded down) strictly higher
    /// than its neighbours; the two end samples are never peaks.</para>
    /// <para>Prominences are computed in parallel over peaks — each walks outwards independently —
    /// which is where the time goes on long noisy signals. The distance filter visits peaks in
    /// order of decreasing height; ties are broken by a <em>stable</em> sort (later index first),
    /// whereas scipy's quicksort leaves tie order unspecified, so with exactly equal heights the
    /// kept peak can differ.</para>
    /// </remarks>
    /// <param name="x">Signal.</param>
    /// <param name="height">Required peak height.</param>
    /// <param name="threshold">Required vertical distance to both neighbours.</param>
    /// <param name="distance">Minimal horizontal distance (≥ 1) between peaks; smaller peaks are removed first.</param>
    /// <param name="prominence">Required prominence.</param>
    /// <param name="width">Required width at <paramref name="relHeight"/> of the prominence.</param>
    /// <param name="wlen">Window length limiting the prominence search (rounded up; must exceed 1).</param>
    /// <param name="relHeight">Relative height at which the width is measured.</param>
    /// <param name="plateauSize">Required plateau size.</param>
    public static FindPeaksResult FindPeaks(double[] x, PeakCondition? height = null, PeakCondition? threshold = null, double? distance = null,
        PeakCondition? prominence = null, PeakCondition? width = null, double? wlen = null, double relHeight = 0.5, PeakCondition? plateauSize = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (distance is < 1) throw new ArgumentException("`distance` must be greater or equal to 1.", nameof(distance));
        var n = x.Length;
        var (peaks, leftEdges, rightEdges) = LocalMaxima1D(x);
        int[]? plateau = null, le = null, re = null;
        double[]? heights = null, lth = null, rth = null, prom = null, wh = null, wd = null, lip = null, rip = null;
        int[]? lb = null, rb = null;

        void Keep(bool[] keep)
        {
            peaks = Filter(peaks, keep);
            plateau = Filter(plateau, keep); le = Filter(le, keep); re = Filter(re, keep);
            heights = Filter(heights, keep); lth = Filter(lth, keep); rth = Filter(rth, keep);
            prom = Filter(prom, keep); lb = Filter(lb, keep); rb = Filter(rb, keep);
            wh = Filter(wh, keep); wd = Filter(wd, keep); lip = Filter(lip, keep); rip = Filter(rip, keep);
        }

        if (plateauSize is not null)
        {
            plateau = new int[peaks.Length];
            for (var i = 0; i < peaks.Length; i++) plateau[i] = rightEdges[i] - leftEdges[i] + 1;
            le = leftEdges;
            re = rightEdges;
            var keep = new bool[peaks.Length];
            for (var i = 0; i < peaks.Length; i++) keep[i] = plateauSize.Accepts(plateau[i], peaks[i], n);
            Keep(keep);
        }
        if (height is not null)
        {
            heights = new double[peaks.Length];
            for (var i = 0; i < peaks.Length; i++) heights[i] = x[peaks[i]];
            var keep = new bool[peaks.Length];
            for (var i = 0; i < peaks.Length; i++) keep[i] = height.Accepts(heights[i], peaks[i], n);
            Keep(keep);
        }
        if (threshold is not null)
        {
            lth = new double[peaks.Length];
            rth = new double[peaks.Length];
            var keep = new bool[peaks.Length];
            for (var i = 0; i < peaks.Length; i++)
            {
                var p = peaks[i];
                lth[i] = x[p] - x[p - 1];
                rth[i] = x[p] - x[p + 1];
                var lo = threshold.LowerAt(p, n);
                var hi = threshold.UpperAt(p, n);
                keep[i] = (lo is null || lo.Value <= Math.Min(lth[i], rth[i])) && (hi is null || Math.Max(lth[i], rth[i]) <= hi.Value);
            }
            Keep(keep);
        }
        if (distance is { } dist)
            Keep(SelectByPeakDistance(peaks, x, dist));
        if (prominence is not null || width is not null)
        {
            var w = WlenAsExpected(wlen);
            (prom, lb, rb) = PeakProminencesCore(x, peaks, w);
        }
        if (prominence is not null)
        {
            var keep = new bool[peaks.Length];
            for (var i = 0; i < peaks.Length; i++) keep[i] = prominence.Accepts(prom![i], peaks[i], n);
            Keep(keep);
        }
        if (width is not null)
        {
            (wd, wh, lip, rip) = PeakWidthsCore(x, peaks, relHeight, prom!, lb!, rb!);
            var keep = new bool[peaks.Length];
            for (var i = 0; i < peaks.Length; i++) keep[i] = width.Accepts(wd[i], peaks[i], n);
            Keep(keep);
        }
        var props = new PeakProperties
        {
            PeakHeights = heights, LeftThresholds = lth, RightThresholds = rth,
            Prominences = prom, LeftBases = lb, RightBases = rb,
            WidthHeights = wh, Widths = wd, LeftIps = lip, RightIps = rip,
            PlateauSizes = plateau, LeftEdges = le, RightEdges = re,
        };
        return new FindPeaksResult(peaks, props);
    }

    /// <summary>
    /// Prominence of each peak (<c>scipy.signal.peak_prominences</c>): its height above the higher
    /// of the two lowest points reached before the signal climbs above it on either side.
    /// </summary>
    /// <param name="x">Signal.</param>
    /// <param name="peaks">Peak indices.</param>
    /// <param name="wlen">Optional window (samples, rounded up, &gt; 1) centred on each peak that limits the search.</param>
    public static (double[] Prominences, int[] LeftBases, int[] RightBases) PeakProminences(double[] x, int[] peaks, double? wlen = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(peaks);
        return PeakProminencesCore(x, peaks, WlenAsExpected(wlen));
    }

    /// <summary>
    /// Width of each peak at <paramref name="relHeight"/> of its prominence
    /// (<c>scipy.signal.peak_widths</c>), with linearly interpolated intersection points.
    /// </summary>
    /// <param name="x">Signal.</param>
    /// <param name="peaks">Peak indices.</param>
    /// <param name="relHeight">0 = at the peak, 1 = at the prominence base.</param>
    /// <param name="prominenceData">Output of <see cref="PeakProminences"/>; computed when null.</param>
    /// <param name="wlen">Window for the prominence computation when <paramref name="prominenceData"/> is null.</param>
    public static (double[] Widths, double[] WidthHeights, double[] LeftIps, double[] RightIps) PeakWidths(double[] x, int[] peaks, double relHeight = 0.5,
        (double[] Prominences, int[] LeftBases, int[] RightBases)? prominenceData = null, double? wlen = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(peaks);
        var pd = prominenceData ?? PeakProminencesCore(x, peaks, WlenAsExpected(wlen));
        return PeakWidthsCore(x, peaks, relHeight, pd.Prominences, pd.LeftBases, pd.RightBases);
    }

    private static long WlenAsExpected(double? wlen)
    {
        if (wlen is null) return -1;
        if (!(wlen.Value > 1)) throw new ArgumentException($"`wlen` must be larger than 1, was {wlen}.", nameof(wlen));
        return (long)Math.Ceiling(wlen.Value);
    }

    private static T[]? Filter<T>(T[]? a, bool[] keep)
    {
        if (a is null) return null;
        var c = 0;
        foreach (var k in keep) if (k) c++;
        var r = new T[c];
        var j = 0;
        for (var i = 0; i < a.Length; i++) if (keep[i]) r[j++] = a[i];
        return r;
    }

    /// <summary>scipy's <c>_local_maxima_1d</c>: strict maxima and flat plateaus, midpoint rounded down.</summary>
    private static (int[] Mid, int[] Left, int[] Right) LocalMaxima1D(double[] x)
    {
        var mid = new List<int>();
        var left = new List<int>();
        var right = new List<int>();
        var iMax = x.Length - 1;
        var i = 1;
        while (i < iMax)
        {
            if (x[i - 1] < x[i])
            {
                var ahead = i + 1;
                while (ahead < iMax && x[ahead] == x[i]) ahead++;
                if (x[ahead] < x[i])
                {
                    left.Add(i);
                    right.Add(ahead - 1);
                    mid.Add((i + ahead - 1) / 2);
                    i = ahead;
                }
            }
            i++;
        }
        return (mid.ToArray(), left.ToArray(), right.ToArray());
    }

    private static bool[] SelectByPeakDistance(int[] peaks, double[] x, double distance)
    {
        var size = peaks.Length;
        var d = Math.Ceiling(distance);
        var keep = new bool[size];
        Array.Fill(keep, true);
        var heights = new double[size];
        var order = new int[size];
        for (var i = 0; i < size; i++) { heights[i] = x[peaks[i]]; order[i] = i; }
        // Stable ascending sort by height; the loop below walks it backwards.
        Array.Sort((double[])heights.Clone(), order);
        StableFix(order, heights);
        for (var i = size - 1; i >= 0; i--)
        {
            var j = order[i];
            if (!keep[j]) continue;
            var k = j - 1;
            while (k >= 0 && peaks[j] - peaks[k] < d) { keep[k] = false; k--; }
            k = j + 1;
            while (k < size && peaks[k] - peaks[j] < d) { keep[k] = false; k++; }
        }
        return keep;
    }

    /// <summary>Array.Sort is introsort (unstable); re-sort runs of equal keys by index.</summary>
    private static void StableFix(int[] order, double[] keys)
    {
        var i = 0;
        while (i < order.Length)
        {
            var j = i + 1;
            while (j < order.Length && keys[order[j]] == keys[order[i]]) j++;
            if (j - i > 1) Array.Sort(order, i, j - i);
            i = j;
        }
    }

    private static (double[] Prominences, int[] LeftBases, int[] RightBases) PeakProminencesCore(double[] x, int[] peaks, long wlen)
    {
        var m = peaks.Length;
        var prom = new double[m];
        var lb = new int[m];
        var rb = new int[m];
        var n = x.Length;
        foreach (var p in peaks)
            if (p < 0 || p > n - 1) throw new ArgumentException($"peak {p} is not a valid index for `x`.", nameof(peaks));
        void One(int k)
        {
            var peak = peaks[k];
            long iMin = 0, iMax = n - 1;
            if (wlen >= 2)
            {
                iMin = Math.Max(peak - wlen / 2, iMin);
                iMax = Math.Min(peak + wlen / 2, iMax);
            }
            var xp = x[peak];
            var i = peak;
            lb[k] = peak;
            var leftMin = xp;
            while (iMin <= i && x[i] <= xp)
            {
                if (x[i] < leftMin) { leftMin = x[i]; lb[k] = i; }
                i--;
            }
            i = peak;
            rb[k] = peak;
            var rightMin = xp;
            while (i <= iMax && x[i] <= xp)
            {
                if (x[i] < rightMin) { rightMin = x[i]; rb[k] = i; }
                i++;
            }
            prom[k] = xp - Math.Max(leftMin, rightMin);
        }
        if (m > 4096) Parallel.For(0, (m + 1023) / 1024, c => { for (var k = c * 1024; k < Math.Min(m, c * 1024 + 1024); k++) One(k); });
        else for (var k = 0; k < m; k++) One(k);
        return (prom, lb, rb);
    }

    private static (double[] Widths, double[] WidthHeights, double[] LeftIps, double[] RightIps) PeakWidthsCore(double[] x, int[] peaks, double relHeight,
        double[] prominences, int[] leftBases, int[] rightBases)
    {
        if (relHeight < 0) throw new ArgumentException("`rel_height` must be greater or equal to 0.0.", nameof(relHeight));
        var m = peaks.Length;
        if (prominences.Length != m || leftBases.Length != m || rightBases.Length != m)
            throw new ArgumentException("arrays in `prominence_data` must have the same shape as `peaks`.");
        var widths = new double[m];
        var wh = new double[m];
        var lip = new double[m];
        var rip = new double[m];
        for (var p = 0; p < m; p++)
        {
            var iMin = leftBases[p];
            var iMax = rightBases[p];
            var peak = peaks[p];
            if (!(0 <= iMin && iMin <= peak && peak <= iMax && iMax < x.Length))
                throw new ArgumentException($"prominence data is invalid for peak {peak}.");
            var height = wh[p] = x[peak] - prominences[p] * relHeight;
            var i = peak;
            while (iMin < i && height < x[i]) i--;
            double left = i;
            if (x[i] < height) left += (height - x[i]) / (x[i + 1] - x[i]);
            i = peak;
            while (i < iMax && height < x[i]) i++;
            double right = i;
            if (x[i] < height) right -= (height - x[i]) / (x[i - 1] - x[i]);
            widths[p] = right - left;
            lip[p] = left;
            rip[p] = right;
        }
        return (widths, wh, lip, rip);
    }

    // ================================================================== argrelextrema

    /// <summary>
    /// Indices of relative extrema (<c>scipy.signal.argrelextrema</c>): samples for which
    /// <paramref name="comparator"/>(x[i], x[i ± k]) holds for every k = 1…<paramref name="order"/>.
    /// Unlike <see cref="FindPeaks"/>, plateaus never qualify under a strict comparator.
    /// </summary>
    public static int[] ArgRelExtrema(double[] data, Func<double, double, bool> comparator, int order = 1, ArgRelMode mode = ArgRelMode.Clip)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(comparator);
        if (order < 1) throw new ArgumentException("Order must be an int >= 1.", nameof(order));
        var n = data.Length;
        var res = new List<int>();
        for (var i = 0; i < n; i++)
        {
            var ok = true;
            for (var s = 1; s <= order && ok; s++)
            {
                ok = comparator(data[i], data[Take(i + s, n, mode)]) && comparator(data[i], data[Take(i - s, n, mode)]);
            }
            if (ok) res.Add(i);
        }
        return res.ToArray();
    }

    /// <summary>Relative maxima (<c>scipy.signal.argrelmax</c>).</summary>
    public static int[] ArgRelMax(double[] data, int order = 1, ArgRelMode mode = ArgRelMode.Clip)
        => ArgRelExtrema(data, static (a, b) => a > b, order, mode);

    /// <summary>Relative minima (<c>scipy.signal.argrelmin</c>).</summary>
    public static int[] ArgRelMin(double[] data, int order = 1, ArgRelMode mode = ArgRelMode.Clip)
        => ArgRelExtrema(data, static (a, b) => a < b, order, mode);

    /// <summary>
    /// Relative extrema along <paramref name="axis"/> of an n-D array; returns one index array per
    /// dimension (numpy's <c>nonzero</c> layout, row-major order).
    /// </summary>
    public static int[][] ArgRelExtrema(NdArray data, Func<double, double, bool> comparator, int axis = 0, int order = 1, ArgRelMode mode = ArgRelMode.Clip)
    {
        var mask = BoolRelExtrema(data, comparator, axis, order, mode);
        var shape = data.Shape.ToArray();
        var rank = shape.Length;
        var hits = new List<int>();
        for (var i = 0; i < mask.Length; i++) if (mask[i]) hits.Add(i);
        var result = new int[rank][];
        for (var d = 0; d < rank; d++) result[d] = new int[hits.Count];
        for (var h = 0; h < hits.Count; h++)
        {
            var rem = hits[h];
            for (var d = rank - 1; d >= 0; d--) { result[d][h] = rem % shape[d]; rem /= shape[d]; }
        }
        return result;
    }

    private static bool[] BoolRelExtrema(NdArray data, Func<double, double, bool> comparator, int axis, int order, ArgRelMode mode)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(comparator);
        if (order < 1) throw new ArgumentException("Order must be an int >= 1.", nameof(order));
        var rank = data.Rank;
        var ax = axis < 0 ? axis + rank : axis;
        if (ax < 0 || ax >= rank) throw new ArgumentException("axis out of range.", nameof(axis));
        var shape = data.Shape.ToArray();
        var len = shape[ax];
        var inner = 1;
        for (var d = ax + 1; d < rank; d++) inner *= shape[d];
        var v = data.AsContiguous().ToArray();
        var mask = new bool[v.Length];
        for (var idx = 0; idx < v.Length; idx++)
        {
            var pos = idx / inner % len;
            var baseIdx = idx - pos * inner;
            var ok = true;
            for (var s = 1; s <= order && ok; s++)
                ok = comparator(v[idx], v[baseIdx + Take(pos + s, len, mode) * inner]) && comparator(v[idx], v[baseIdx + Take(pos - s, len, mode) * inner]);
            mask[idx] = ok;
        }
        return mask;
    }

    private static int Take(int i, int n, ArgRelMode mode)
        => mode == ArgRelMode.Wrap ? ((i % n) + n) % n : Math.Clamp(i, 0, n - 1);
}
