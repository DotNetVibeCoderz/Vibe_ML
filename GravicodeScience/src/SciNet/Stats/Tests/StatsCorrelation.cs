using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Result of the matrix forms of the correlation functions: coefficients and p-values per pair.</summary>
/// <param name="Statistic">k×k matrix of correlation coefficients.</param>
/// <param name="Pvalue">k×k matrix of p-values.</param>
public sealed record CorrelationMatrixResult(NdArray Statistic, NdArray Pvalue);

/// <summary>scipy.stats correlation and simple regression.</summary>
public static partial class Stats
{
    /// <summary>
    /// Pearson correlation coefficient and the exact p-value under bivariate normality (r ~ Beta(n/2 − 1,
    /// n/2 − 1) on [−1, 1]), scipy's <c>pearsonr</c>. The result's
    /// <see cref="PearsonRResult.ConfidenceInterval(double)"/> uses the Fisher transformation. A constant
    /// input yields NaN.
    /// </summary>
    public static PearsonRResult Pearsonr(double[] x, double[] y, Alternative alternative = Alternative.TwoSided)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (x.Length != y.Length) throw new ArgumentException("`x` and `y` must have the same length along `axis`.");
        var n = x.Length;
        if (n < 2) throw new ArgumentException("`x` and `y` must have length at least 2.");
        var r = PearsonCoefficient(x, y);
        double p;
        if (double.IsNaN(r)) p = double.NaN;
        else if (n == 2) { r = Math.Round(r); p = 1.0; }
        else
        {
            var ab = n / 2.0 - 1;
            p = alternative switch
            {
                Alternative.Less => Special.BetaInc(ab, ab, Math.Clamp((1 + r) / 2, 0, 1)),
                Alternative.Greater => Special.BetaInc(ab, ab, Math.Clamp((1 - r) / 2, 0, 1)),
                _ => 2 * Special.BetaInc(ab, ab, Math.Clamp((1 - Math.Abs(r)) / 2, 0, 1)),
            };
        }
        return new PearsonRResult(r, p, n, alternative) { X = x, Y = y };
    }

    /// <summary>Point-biserial correlation (Pearson's r with a dichotomous x), scipy's <c>pointbiserialr</c>.</summary>
    public static SignificanceResult Pointbiserialr(double[] x, double[] y)
    {
        var r = Pearsonr(x, y);
        return new SignificanceResult(r.Statistic, r.Pvalue);
    }

    internal static double PearsonCoefficient(double[] x, double[] y)
    {
        var n = x.Length;
        if (x.All(v => v == x[0]) || y.All(v => v == y[0])) return double.NaN;
        var xm = StatsTestMath.Mean(x);
        var ym = StatsTestMath.Mean(y);
        // Scale by the largest deviation before squaring (scipy) so huge or tiny data cannot overflow.
        double xmax = 0, ymax = 0;
        for (var i = 0; i < n; i++)
        {
            xmax = Math.Max(xmax, Math.Abs(x[i] - xm));
            ymax = Math.Max(ymax, Math.Abs(y[i] - ym));
        }
        double sxx = 0, syy = 0, sxy = 0;
        for (var i = 0; i < n; i++)
        {
            var a = (x[i] - xm) / xmax;
            var b = (y[i] - ym) / ymax;
            sxx += a * a;
            syy += b * b;
            sxy += a * b;
        }
        return Math.Clamp(sxy / (Math.Sqrt(sxx) * Math.Sqrt(syy)), -1.0, 1.0);
    }

    /// <summary>
    /// Spearman rank correlation with the t-approximation p-value (n − 2 df), scipy's <c>spearmanr</c>.
    /// </summary>
    public static SignificanceResult Spearmanr(double[] a, double[] b, Alternative alternative = Alternative.TwoSided)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.Length != b.Length) throw new ArgumentException("a and b must have the same length.");
        var n = a.Length;
        if (n <= 1 || a.All(v => v == a[0]) || b.All(v => v == b[0])) return new SignificanceResult(double.NaN, double.NaN);
        if (a.Any(double.IsNaN) || b.Any(double.IsNaN)) return new SignificanceResult(double.NaN, double.NaN);
        double[] ra = null!, rb = null!;
        if (n >= 100_000) Parallel.Invoke(() => ra = Rankdata(a), () => rb = Rankdata(b));
        else { ra = Rankdata(a); rb = Rankdata(b); }
        var rs = CorrFromCov(ra, rb);
        return new SignificanceResult(rs, SpearmanP(rs, n - 2, alternative));
    }

    private static double SpearmanP(double rs, int dof, Alternative alternative)
    {
        var t = rs * Math.Sqrt(Math.Max(dof / ((rs + 1.0) * (1.0 - rs)), 0));
        if (double.IsNaN(t) && Math.Abs(rs) == 1) t = rs * double.PositiveInfinity;
        return StatsTestMath.TPvalue(t, dof, alternative);
    }

    /// <summary>numpy's corrcoef of two variables: cov / (σx σy), clipped to [−1, 1].</summary>
    private static double CorrFromCov(double[] x, double[] y)
    {
        var n = x.Length;
        var xm = StatsTestMath.Mean(x);
        var ym = StatsTestMath.Mean(y);
        double sxx = 0, syy = 0, sxy = 0;
        for (var i = 0; i < n; i++)
        {
            var a = x[i] - xm;
            var b = y[i] - ym;
            sxx += a * a;
            syy += b * b;
            sxy += a * b;
        }
        return Math.Clamp(sxy / Math.Sqrt(sxx) / Math.Sqrt(syy), -1.0, 1.0);
    }

    /// <summary>
    /// Spearman correlation matrix of the columns of <paramref name="data"/> (observations in rows),
    /// scipy's <c>spearmanr(a)</c> for 2-D input. Ranking and the k(k−1)/2 correlations run in parallel.
    /// </summary>
    public static CorrelationMatrixResult Spearmanr(NdArray data, Alternative alternative = Alternative.TwoSided)
    {
        var (nobs, k, d) = StatsTestMath.Matrix(data, nameof(data));
        var ranks = new double[k][];
        Parallel.For(0, k, j =>
        {
            var col = new double[nobs];
            for (var i = 0; i < nobs; i++) col[i] = d[i * k + j];
            ranks[j] = Rankdata(col);
        });
        var rs = new double[k * k];
        var ps = new double[k * k];
        Parallel.For(0, k, i =>
        {
            for (var j = i; j < k; j++)
            {
                var r = i == j ? 1.0 : CorrFromCov(ranks[i], ranks[j]);
                rs[i * k + j] = rs[j * k + i] = r;
                var p = SpearmanP(r, nobs - 2, alternative);
                ps[i * k + j] = ps[j * k + i] = p;
            }
        });
        return new CorrelationMatrixResult(new NdArray(rs, k, k), new NdArray(ps, k, k));
    }

    /// <summary>
    /// Kendall's τ-b (or τ-c) rank correlation, scipy's <c>kendalltau</c>. Discordant pairs are counted
    /// by merge sort in O(n log n). <paramref name="method"/> <c>"auto"</c> uses the exact null
    /// distribution when there are no ties and n ≤ 33 (or the statistic is extreme), the normal
    /// approximation with the full tie-corrected variance otherwise.
    /// </summary>
    public static SignificanceResult Kendalltau(double[] x, double[] y, string method = "auto", string variant = "b", Alternative alternative = Alternative.TwoSided)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (x.Length != y.Length) throw new ArgumentException("Array shapes are incompatible for broadcasting.");
        var size = x.Length;
        if (size == 0 || x.Any(double.IsNaN) || y.Any(double.IsNaN)) return new SignificanceResult(double.NaN, double.NaN);
        // Dense ranks of y (and its tie runs) from one sort; the x sort runs concurrently on large input.
        var ky = (double[])y.Clone();
        var iy = new int[size];
        var kx = (double[])x.Clone();
        var idx = new int[size];
        for (var i = 0; i < size; i++) { iy[i] = i; idx[i] = i; }
        if (size >= 50_000) Parallel.Invoke(() => Array.Sort(ky, iy), () => Array.Sort(kx, idx));
        else { Array.Sort(ky, iy); Array.Sort(kx, idx); }
        var yRank = new int[size];
        long ytie = 0;
        double y0 = 0, y1 = 0;
        var denseY = 0;
        for (int start0 = 0; start0 < size;)
        {
            var end = start0 + 1;
            while (end < size && ky[end] == ky[start0]) end++;
            denseY++;
            for (var q = start0; q < end; q++) yRank[iy[q]] = denseY;
            AddTie(end - start0, ref ytie, ref y0, ref y1);
            start0 = end;
        }
        // Order by x, breaking ties by y rank, and record the x tie runs.
        var xs = new int[size];
        var ys = new int[size];
        long xtie = 0;
        double x0 = 0, x1 = 0;
        var denseX = 0;
        for (int start0 = 0; start0 < size;)
        {
            var end = start0 + 1;
            while (end < size && kx[end] == kx[start0]) end++;
            denseX++;
            if (end - start0 > 1) Array.Sort(idx, start0, end - start0, Comparer<int>.Create((i, j) => yRank[i].CompareTo(yRank[j])));
            for (var q = start0; q < end; q++) { xs[q] = denseX; ys[q] = yRank[idx[q]]; }
            AddTie(end - start0, ref xtie, ref x0, ref x1);
            start0 = end;
        }
        var dis = CountInversions((int[])ys.Clone());
        long ntie = 0;
        var start = 0;
        for (var i = 1; i <= size; i++)
        {
            if (i == size || xs[i] != xs[i - 1] || ys[i] != ys[i - 1])
            {
                long cnt = i - start;
                ntie += cnt * (cnt - 1) / 2;
                start = i;
            }
        }
        long tot = (long)size * (size - 1) / 2;
        if (xtie == tot || ytie == tot) return new SignificanceResult(double.NaN, double.NaN);
        var conMinusDis = (double)(tot - xtie - ytie + ntie - 2 * dis);
        double tau;
        if (variant == "b") tau = conMinusDis / Math.Sqrt(tot - xtie) / Math.Sqrt(tot - ytie);
        else if (variant == "c")
        {
            double minclasses = Math.Min(denseX, denseY);
            tau = 2 * conMinusDis / ((double)size * size * (minclasses - 1) / minclasses);
        }
        else throw new ArgumentException($"Unknown variant of the method chosen: {variant}. variant must be 'b' or 'c'.", nameof(variant));
        tau = Math.Min(1.0, Math.Max(-1.0, tau));
        var m = method.ToLowerInvariant();
        if (m == "exact" && (xtie != 0 || ytie != 0)) throw new ArgumentException("Ties found, exact method cannot be used.", nameof(method));
        if (m == "auto")
            m = xtie == 0 && ytie == 0 && (size <= 33 || Math.Min(dis, tot - dis) <= 1) ? "exact" : "asymptotic";
        double p;
        if (xtie == 0 && ytie == 0 && m == "exact") p = StatsRankDistributions.KendallPExact(size, tot - dis, alternative);
        else if (m == "asymptotic")
        {
            var mm = size * (size - 1.0);
            var var = (mm * (2.0 * size + 5) - x1 - y1) / 18 + 2.0 * xtie * ytie / mm + x0 * y0 / (9 * mm * (size - 2));
            p = StatsTestMath.NormPvalue(conMinusDis / Math.Sqrt(var), alternative);
        }
        else throw new ArgumentException($"Unknown method {method} specified.  Use 'auto', 'exact' or 'asymptotic'.", nameof(method));
        return new SignificanceResult(tau, p);
    }

    private static void AddTie(long c, ref long tie, ref double t0, ref double t1)
    {
        if (c <= 1) return;
        tie += c * (c - 1) / 2;
        t0 += c * (c - 1.0) * (c - 2);
        t1 += c * (c - 1.0) * (2 * c + 5);
    }

    /// <summary>Number of pairs i &lt; j with a[i] &gt; a[j] (strict), by bottom-up merge sort.</summary>
    private static long CountInversions(int[] a)
    {
        var n = a.Length;
        var buf = new int[n];
        long inv = 0;
        for (var width = 1; width < n; width *= 2)
        {
            for (var lo = 0; lo < n; lo += 2 * width)
            {
                int mid = Math.Min(lo + width, n), hi = Math.Min(lo + 2 * width, n);
                int i = lo, j = mid, k = lo;
                while (i < mid && j < hi)
                {
                    if (a[i] <= a[j]) buf[k++] = a[i++];
                    else { inv += mid - i; buf[k++] = a[j++]; }
                }
                while (i < mid) buf[k++] = a[i++];
                while (j < hi) buf[k++] = a[j++];
            }
            (a, buf) = (buf, a);
        }
        return inv;
    }

    /// <summary>
    /// Somers' D(Y|X) for two ordinal sequences, scipy's <c>somersd(x, y)</c>, with its asymptotic test.
    /// </summary>
    public static SomersDResult Somersd(double[] x, double[] y, Alternative alternative = Alternative.TwoSided)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (x.Length != y.Length) throw new ArgumentException("Rankings must be of equal length.");
        var (_, table) = Crosstab(x, y);
        return Somersd(table, alternative);
    }

    /// <summary>Somers' D(Y|X) from an r×c contingency table (rows: x categories), scipy's <c>somersd(table)</c>.</summary>
    public static SomersDResult Somersd(NdArray table, Alternative alternative = Alternative.TwoSided)
    {
        var (r, c, A) = StatsTestMath.Matrix(table, nameof(table));
        if (A.Any(v => v < 0)) throw new ArgumentException("All elements of the contingency table must be non-negative.", nameof(table));
        if (r <= 1 || c <= 1) return new SomersDResult(double.NaN, double.NaN, table);
        var (P, Q, S2) = ConcordanceSums(A, r, c);
        var na = A.Sum();
        var sri2 = 0.0;
        for (var i = 0; i < r; i++)
        {
            var s = 0.0;
            for (var j = 0; j < c; j++) s += A[i * c + j];
            sri2 += s * s;
        }
        var d = (P - Q) / (na * na - sri2);
        var S = S2 - (P - Q) * (P - Q) / na;
        var z = (P - Q) / Math.Sqrt(4 * S);
        return new SomersDResult(d, StatsTestMath.NormPvalue(z, alternative), table);
    }

    /// <summary>Twice the concordant (P) and discordant (Q) pair counts and Σ A_ij (A⁺ − A⁻)².</summary>
    private static (double P, double Q, double S) ConcordanceSums(double[] A, int r, int c)
    {
        // 2-D prefix sums make each quadrant sum O(1).
        var pre = new double[(r + 1) * (c + 1)];
        for (var i = 0; i < r; i++)
            for (var j = 0; j < c; j++)
                pre[(i + 1) * (c + 1) + j + 1] = A[i * c + j] + pre[i * (c + 1) + j + 1] + pre[(i + 1) * (c + 1) + j] - pre[i * (c + 1) + j];
        double Block(int i0, int i1, int j0, int j1) // rows [i0, i1), cols [j0, j1)
            => i1 <= i0 || j1 <= j0 ? 0 : pre[i1 * (c + 1) + j1] - pre[i0 * (c + 1) + j1] - pre[i1 * (c + 1) + j0] + pre[i0 * (c + 1) + j0];
        double P = 0, Q = 0, S = 0;
        for (var i = 0; i < r; i++)
            for (var j = 0; j < c; j++)
            {
                var a = A[i * c + j];
                if (a == 0) continue;
                var con = Block(0, i, 0, j) + Block(i + 1, r, j + 1, c);
                var dis = Block(i + 1, r, 0, j) + Block(0, i, j + 1, c);
                P += a * con;
                Q += a * dis;
                S += a * (con - dis) * (con - dis);
            }
        return (P, Q, S);
    }

    /// <summary>
    /// Least-squares line of y on x with the t-test of the slope, scipy's <c>linregress</c>.
    /// </summary>
    public static LinregressResult Linregress(double[] x, double[] y, Alternative alternative = Alternative.TwoSided)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (x.Length == 0 || y.Length == 0) throw new ArgumentException("Inputs must not be empty.");
        if (x.Length != y.Length) throw new ArgumentException("x and y must have the same length.");
        var n = x.Length;
        if (n > 1 && x.Max() == x.Min()) throw new ArgumentException("Cannot calculate a linear regression if all x values are identical");
        var xmean = StatsTestMath.Mean(x);
        var ymean = StatsTestMath.Mean(y);
        double ssxm = 0, ssym = 0, ssxym = 0;
        for (var i = 0; i < n; i++)
        {
            var a = x[i] - xmean;
            var b = y[i] - ymean;
            ssxm += a * a;
            ssym += b * b;
            ssxym += a * b;
        }
        ssxm /= n; ssym /= n; ssxym /= n;
        double r;
        if (ssxm == 0 || ssym == 0) r = ssxym == 0 ? double.NaN : 0.0;
        else r = Math.Clamp(ssxym / Math.Sqrt(ssxm * ssym), -1.0, 1.0);
        var slope = ssxym / ssxm;
        var intercept = ymean - slope * xmean;
        double p, se, ise;
        if (n == 2)
        {
            p = y[0] == y[1] ? 1.0 : 0.0;
            se = 0;
            ise = 0;
        }
        else
        {
            const double tiny = 1.0e-20;
            double df = n - 2;
            var t = r * Math.Sqrt(df / ((1.0 - r + tiny) * (1.0 + r + tiny)));
            p = StatsTestMath.TPvalue(t, df, alternative);
            se = Math.Sqrt((1 - r * r) * ssym / ssxm / df);
            ise = se * Math.Sqrt(ssxm + xmean * xmean);
        }
        return new LinregressResult(slope, intercept, r, p, se, ise);
    }

    /// <summary>
    /// Theil–Sen estimator: the median of all pairwise slopes, with Sen's (1968) confidence interval,
    /// scipy's <c>theilslopes</c>. <paramref name="method"/> <c>"separate"</c> takes the intercept as
    /// median(y) − slope·median(x); <c>"joint"</c> as median(y − slope·x). x defaults to 0…n−1.
    /// </summary>
    public static TheilslopesResult Theilslopes(double[] y, double[]? x = null, double alpha = 0.95, string method = "separate")
    {
        ArgumentNullException.ThrowIfNull(y);
        if (method is not ("joint" or "separate")) throw new ArgumentException($"method must be either 'joint' or 'separate'.'{method}' is invalid.", nameof(method));
        var n = y.Length;
        x ??= Enumerable.Range(0, n).Select(i => (double)i).ToArray();
        if (x.Length != n) throw new ArgumentException("Array shapes are incompatible for broadcasting.");
        if (n < 2) throw new ArgumentException("`x` and `y` must have length at least 2.");
        var slopes = new List<double>(n * (n - 1) / 2);
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                var dx = x[i] - x[j];
                if (dx > 0) slopes.Add((y[i] - y[j]) / dx);
            }
        var s = slopes.ToArray();
        Array.Sort(s);
        var medslope = s.Length == 0 ? double.NaN : StatsTestMath.Median(s);
        double medinter;
        if (method == "joint")
        {
            var res = new double[n];
            for (var i = 0; i < n; i++) res[i] = y[i] - medslope * x[i];
            medinter = StatsTestMath.Median(res);
        }
        else medinter = StatsTestMath.Median(y) - medslope * StatsTestMath.Median(x);
        if (alpha > 0.5) alpha = 1 - alpha;
        var z = Special.Ndtri(alpha / 2);
        double ny = n;
        var sigsq = 1 / 18.0 * (ny * (ny - 1) * (2 * ny + 5));
        foreach (var k in StatsTestMath.TieCounts(StatsTestMath.Sorted(x))) if (k > 1) sigsq -= 1 / 18.0 * k * (k - 1.0) * (2 * k + 5);
        foreach (var k in StatsTestMath.TieCounts(StatsTestMath.Sorted(y))) if (k > 1) sigsq -= 1 / 18.0 * k * (k - 1.0) * (2 * k + 5);
        double lo = double.NaN, hi = double.NaN;
        var nt = s.Length;
        var sigma = Math.Sqrt(sigsq);
        if (nt > 0 && !double.IsNaN(sigma))
        {
            var ru = (int)Math.Min(Math.Round((nt - z * sigma) / 2, MidpointRounding.ToEven), nt - 1);
            var rl = (int)Math.Max(Math.Round((nt + z * sigma) / 2, MidpointRounding.ToEven) - 1, 0);
            if (rl >= 0 && rl < nt && ru >= 0 && ru < nt) { lo = s[rl]; hi = s[ru]; }
        }
        return new TheilslopesResult(medslope, medinter, lo, hi);
    }

    /// <summary>
    /// Siegel's repeated-medians slope (50% breakdown), scipy's <c>siegelslopes</c>. <paramref name="method"/>
    /// <c>"hierarchical"</c> takes the intercept as median(y − slope·x); <c>"separate"</c> uses repeated
    /// medians of the pairwise intercepts.
    /// </summary>
    public static SiegelslopesResult Siegelslopes(double[] y, double[]? x = null, string method = "hierarchical")
    {
        ArgumentNullException.ThrowIfNull(y);
        if (method is not ("hierarchical" or "separate")) throw new ArgumentException("method can only be 'hierarchical' or 'separate'", nameof(method));
        var n = y.Length;
        x ??= Enumerable.Range(0, n).Select(i => (double)i).ToArray();
        if (x.Length != n) throw new ArgumentException("Array shapes are incompatible for broadcasting.");
        if (n < 2) throw new ArgumentException("`x` and `y` must have length at least 2.");
        var slopes = new double[n];
        var inters = new double[n];
        Parallel.For(0, n, j =>
        {
            var sj = new List<double>(n);
            var ij = new List<double>(n);
            for (var i = 0; i < n; i++)
            {
                var dx = x[j] - x[i];
                if (dx == 0) continue;
                sj.Add((y[j] - y[i]) / dx);
                if (method == "separate") ij.Add((y[i] * x[j] - y[j] * x[i]) / dx);
            }
            slopes[j] = sj.Count == 0 ? double.NaN : StatsTestMath.Median(sj.ToArray());
            if (method == "separate") inters[j] = ij.Count == 0 ? double.NaN : StatsTestMath.Median(ij.ToArray());
        });
        var medslope = StatsTestMath.Median(slopes);
        double medinter;
        if (method == "separate") medinter = StatsTestMath.Median(inters);
        else
        {
            var res = new double[n];
            for (var i = 0; i < n; i++) res[i] = y[i] - medslope * x[i];
            medinter = StatsTestMath.Median(res);
        }
        return new SiegelslopesResult(medslope, medinter);
    }

    /// <summary>
    /// Chatterjee's ξ correlation (2021), a consistent measure of whether y is a function of x, scipy's
    /// <c>chatterjeexi</c>, with its asymptotic one-sided test of independence. Ties in x are broken by
    /// position, so results with tied x depend on input order (as in scipy, where the sort is unstable).
    /// </summary>
    public static SignificanceResult Chatterjeexi(double[] x, double[] y, bool yContinuous = false)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (x.Length != y.Length) throw new ArgumentException("x and y must have the same length.");
        var n = x.Length;
        var idx = Enumerable.Range(0, n).ToArray();
        var keys = (double[])x.Clone();
        Array.Sort(keys, idx);
        var ys = idx.Select(i => y[i]).ToArray();
        var r = Rankdata(ys, "max");
        var l = Rankdata(ys.Select(v => -v).ToArray(), "max");
        var num = 0.0;
        for (var i = 1; i < n; i++) num += Math.Abs(r[i] - r[i - 1]);
        double xi, std;
        if (yContinuous)
        {
            xi = 1 - 3 * num / ((double)n * n - 1);
            std = Math.Sqrt(2.0 / 5) / Math.Sqrt(n);
        }
        else
        {
            var den = 0.0;
            for (var i = 0; i < n; i++) den += (n - l[i]) * l[i];
            xi = 1 - n * num / (2 * den);
            var u = StatsTestMath.Sorted(r);
            double an = 0, bn = 0, cn = 0, dn = 0, v = 0;
            double nn = n;
            for (var i = 1; i <= n; i++)
            {
                var ui = u[i - 1];
                v += ui;
                an += (2 * nn - 2 * i + 1) * ui * ui;
                bn += (v + (nn - i) * ui) * (v + (nn - i) * ui);
                cn += (2 * nn - 2 * i + 1) * ui;
                dn += l[i - 1] * (nn - l[i - 1]);
            }
            an /= Math.Pow(nn, 4);
            bn /= Math.Pow(nn, 5);
            cn /= Math.Pow(nn, 3);
            dn /= Math.Pow(nn, 3);
            var tau2 = (an - 2 * bn + cn * cn) / (dn * dn);
            std = Math.Sqrt(tau2) / Math.Sqrt(nn);
        }
        return new SignificanceResult(xi, Special.Ndtr(-(xi / std)));
    }
    /// <summary>
    /// Weighted Kendall's τ (Vigna 2015), scipy's <c>weightedtau</c>: exchanges are weighted by the
    /// importance of the two elements, weigher(r) = 1/(r + 1) of their rank by default (additively),
    /// so disagreements among the top-ranked items count most. With <paramref name="rank"/> true the
    /// result averages the τ ranked by decreasing (x, y) and by decreasing (y, x); false ranks by
    /// position. The p-value is NaN, as scipy (no null distribution is known).
    /// </summary>
    /// <remarks>Computed from the pair-sum definition in O(n²) (parallel over rows), not with Vigna's
    /// O(n log n) merge algorithm scipy uses; the value is the same.</remarks>
    public static SignificanceResult Weightedtau(double[] x, double[] y, bool rank = true, Func<int, double>? weigher = null, bool additive = true)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (x.Length != y.Length) throw new ArgumentException("Array shapes are incompatible for broadcasting.");
        if (x.Length == 0) return new SignificanceResult(double.NaN, double.NaN);
        double tau;
        if (rank) tau = (WeightedTauCore(x, y, LexRank(x, y), weigher, additive) + WeightedTauCore(y, x, LexRank(y, x), weigher, additive)) / 2;
        else tau = WeightedTauCore(x, y, Enumerable.Range(0, x.Length).ToArray(), weigher, additive);
        return new SignificanceResult(tau, double.NaN);
    }

    /// <summary>Weighted τ with an explicit importance rank per element (0 = most important).</summary>
    public static SignificanceResult Weightedtau(double[] x, double[] y, int[] rank, Func<int, double>? weigher = null, bool additive = true)
    {
        ArgumentNullException.ThrowIfNull(rank);
        if (rank.Length != x.Length) throw new ArgumentException($"All inputs to `weightedtau` must be of the same size, found x-size {x.Length} and rank-size {rank.Length}", nameof(rank));
        return new SignificanceResult(WeightedTauCore(x, y, rank, weigher, additive), double.NaN);
    }

    /// <summary>Rank 0 for the largest (a, b) in lexicographic order, as numpy's <c>lexsort((b, a))[::-1]</c>.</summary>
    private static int[] LexRank(double[] a, double[] b)
    {
        var n = a.Length;
        var order = Enumerable.Range(0, n).ToArray();
        // lexsort is stable ascending; reversing it puts later equal elements first.
        Array.Sort(order, (i, j) =>
        {
            var c = a[j].CompareTo(a[i]);
            if (c != 0) return c;
            c = b[j].CompareTo(b[i]);
            return c != 0 ? c : j.CompareTo(i);
        });
        var r = new int[n];
        for (var k = 0; k < n; k++) r[order[k]] = k;
        return r;
    }

    private static double WeightedTauCore(double[] x, double[] y, int[] rank, Func<int, double>? weigher, bool additive)
    {
        var n = x.Length;
        var w = new double[n];
        for (var i = 0; i < n; i++) w[i] = weigher?.Invoke(rank[i]) ?? 1.0 / (rank[i] + 1);
        double num = 0, dx = 0, dy = 0;
        var lockObj = new object();
        Parallel.For(0, n, () => (0.0, 0.0, 0.0), (i, _, acc) =>
        {
            for (var j = i + 1; j < n; j++)
            {
                var wij = additive ? w[i] + w[j] : w[i] * w[j];
                var sx = Math.Sign(x[i] - x[j]);
                var sy = Math.Sign(y[i] - y[j]);
                acc.Item1 += wij * sx * sy;
                acc.Item2 += wij * sx * sx;
                acc.Item3 += wij * sy * sy;
            }
            return acc;
        }, acc => { lock (lockObj) { num += acc.Item1; dx += acc.Item2; dy += acc.Item3; } });
        if (dx == 0 || dy == 0) return double.NaN;
        return num / Math.Sqrt(dx) / Math.Sqrt(dy);
    }
}
