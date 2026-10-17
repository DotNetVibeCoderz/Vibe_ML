using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// A continuous distribution defined by a histogram, scipy's <c>rv_histogram</c>: the density is constant
/// on each bin, the CDF piecewise linear and the quantile function its linear interpolation.
/// </summary>
public sealed class RvHistogramDistribution : ContinuousDistribution
{
    private readonly double[] _bins, _pdf, _cdf;
    private readonly double[] _counts;
    private readonly bool _density;

    /// <summary>
    /// From bin <paramref name="counts"/> and their <paramref name="edges"/> (one more than the counts,
    /// increasing). With <paramref name="density"/> false (scipy's default) the counts are per bin and the
    /// density is count/(total·width); with true they are proportional to the density itself.
    /// </summary>
    public RvHistogramDistribution(double[] counts, double[] edges, bool density = false, double loc = 0.0, double scale = 1.0) : base(loc, scale)
    {
        ArgumentNullException.ThrowIfNull(counts);
        ArgumentNullException.ThrowIfNull(edges);
        if (counts.Length == 0 || edges.Length != counts.Length + 1) throw new ArgumentException("edges must have one more element than counts.", nameof(edges));
        for (var i = 0; i < counts.Length; i++)
        {
            if (!(counts[i] >= 0) || double.IsInfinity(counts[i])) throw new ArgumentException("counts must be non-negative and finite.", nameof(counts));
            if (!(edges[i + 1] > edges[i]) || !double.IsFinite(edges[i]) || !double.IsFinite(edges[i + 1])) throw new ArgumentException("edges must be finite and strictly increasing.", nameof(edges));
        }
        _counts = (double[])counts.Clone();
        _bins = (double[])edges.Clone();
        _density = density;
        var k = counts.Length;
        _pdf = new double[k];
        double norm = 0;
        for (var i = 0; i < k; i++) norm += density ? counts[i] * (edges[i + 1] - edges[i]) : counts[i];
        if (!(norm > 0)) throw new ArgumentException("The histogram must have positive mass.", nameof(counts));
        for (var i = 0; i < k; i++) _pdf[i] = density ? counts[i] / norm : counts[i] / (norm * (edges[i + 1] - edges[i]));
        _cdf = new double[k + 1];
        for (var i = 0; i < k; i++) _cdf[i + 1] = _cdf[i] + _pdf[i] * (edges[i + 1] - edges[i]);
    }

    /// <inheritdoc />
    public override string Name => "rv_histogram";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new RvHistogramDistribution(_counts, _bins, _density, loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => _bins[0];
    /// <inheritdoc />
    protected override double Upper0 => _bins[^1];

    /// <summary>The bin edges.</summary>
    public double[] Edges => (double[])_bins.Clone();

    /// <inheritdoc />
    protected override double Pdf0(double z)
    {
        // searchsorted(bins, z, side='right'): the bin whose left edge is the largest ≤ z.
        var i = Array.BinarySearch(_bins, z);
        var idx = i >= 0 ? i : ~i - 1;
        while (i >= 0 && idx + 1 < _bins.Length && _bins[idx + 1] == z) idx++;
        return idx >= 0 && idx < _pdf.Length ? _pdf[idx] : 0.0;
    }

    /// <inheritdoc />
    protected override double Cdf0(double z)
    {
        var i = Array.BinarySearch(_bins, z);
        if (i >= 0) return _cdf[i];
        var b = ~i - 1;
        return _cdf[b] + _pdf[b] * (z - _bins[b]);
    }

    /// <inheritdoc />
    protected override double Sf0(double z) => 1.0 - Cdf0(z);

    /// <inheritdoc />
    protected override double Ppf0(double q)
    {
        // np.interp(q, cdf, bins): first segment whose upper CDF reaches q.
        var k = _pdf.Length;
        for (var i = 0; i < k; i++)
        {
            if (_cdf[i + 1] < q) continue;
            var w = _cdf[i + 1] - _cdf[i];
            return w > 0 ? _bins[i] + (q - _cdf[i]) / w * (_bins[i + 1] - _bins[i]) : _bins[i + 1];
        }
        return _bins[^1];
    }

    /// <inheritdoc />
    protected override double Isf0(double q) => Ppf0(1 - q);

    /// <inheritdoc />
    protected override double Munp0(int n)
    {
        var s = 0.0;
        for (var i = 0; i < _pdf.Length; i++)
            s += _pdf[i] * (Math.Pow(_bins[i + 1], n + 1) - Math.Pow(_bins[i], n + 1)) / (n + 1);
        return s;
    }

    /// <inheritdoc />
    protected override double Entropy0()
    {
        var s = 0.0;
        for (var i = 0; i < _pdf.Length; i++)
            if (_pdf[i] > 0) s -= _pdf[i] * Math.Log(_pdf[i]) * (_bins[i + 1] - _bins[i]);
        return s;
    }
}

/// <summary>
/// A user-defined discrete distribution on arbitrary support points, scipy's <c>rv_discrete(values=(xk, pk))</c>.
/// </summary>
/// <remarks>Points need not be integers. Quantiles follow scipy: <c>Ppf(q)</c> is the smallest xₖ with
/// CDF ≥ q and <c>Ppf(0)</c> = x₀ − 1; the survival function sums the upper tail directly.</remarks>
public sealed class RvDiscreteDistribution : IUnivariateDistribution
{
    private readonly double[] _x, _p, _cum, _tail;

    /// <summary>From distinct values <paramref name="xk"/> and probabilities <paramref name="pk"/> summing to 1.</summary>
    public RvDiscreteDistribution(double[] xk, double[] pk)
    {
        ArgumentNullException.ThrowIfNull(xk);
        ArgumentNullException.ThrowIfNull(pk);
        if (xk.Length == 0 || xk.Length != pk.Length) throw new ArgumentException("xk and pk must have the same non-zero length.", nameof(pk));
        if (pk.Any(v => !(v >= 0 && v <= 1))) throw new ArgumentException("All elements of pk must be non-negative.", nameof(pk));
        if (Math.Abs(pk.Sum() - 1) > 1e-8 + 1e-5) throw new ArgumentException("The sum of provided pk is not 1.", nameof(pk));
        if (xk.Any(v => !double.IsFinite(v))) throw new ArgumentException("xk must be finite.", nameof(xk));
        var idx = Enumerable.Range(0, xk.Length).OrderBy(i => xk[i]).ToArray();
        _x = idx.Select(i => xk[i]).ToArray();
        _p = idx.Select(i => pk[i]).ToArray();
        for (var i = 1; i < _x.Length; i++) if (_x[i] == _x[i - 1]) throw new ArgumentException("xk may not contain duplicate values.", nameof(xk));
        var k = _x.Length;
        _cum = new double[k];
        _tail = new double[k]; // P(X > x_i)
        var c = 0.0;
        for (var i = 0; i < k; i++) { c += _p[i]; _cum[i] = c; }
        var t = 0.0;
        for (var i = k - 1; i >= 0; i--) { _tail[i] = t; t += _p[i]; }
    }

    /// <summary>The support points, sorted.</summary>
    public double[] Xk => (double[])_x.Clone();
    /// <summary>Their probabilities.</summary>
    public double[] Pk => (double[])_p.Clone();

    /// <inheritdoc />
    public string Name => "rv_discrete";
    /// <inheritdoc />
    public bool IsDiscrete => true;
    /// <inheritdoc />
    public (double Lower, double Upper) Support() => (_x[0], _x[^1]);

    private int Index(double x) => Array.BinarySearch(_x, x);
    private int LastLe(double x) { var i = Index(x); return i >= 0 ? i : ~i - 1; }

    /// <summary>P(X = x).</summary>
    public double Pmf(double x) { var i = Index(x); return i >= 0 ? _p[i] : 0.0; }
    /// <summary>log P(X = x).</summary>
    public double LogPmf(double x) => Math.Log(Pmf(x));
    /// <inheritdoc />
    public double Density(double x) => Pmf(x);
    /// <inheritdoc />
    public double LogDensity(double x) => LogPmf(x);
    /// <inheritdoc />
    public double Cdf(double x) { if (double.IsNaN(x)) return double.NaN; var i = LastLe(x); return i < 0 ? 0.0 : Math.Min(_cum[i], 1.0); }
    /// <inheritdoc />
    public double LogCdf(double x) => Math.Log(Cdf(x));
    /// <inheritdoc />
    public double Sf(double x) { if (double.IsNaN(x)) return double.NaN; var i = LastLe(x); return i < 0 ? 1.0 : _tail[i]; }
    /// <inheritdoc />
    public double LogSf(double x) => Math.Log(Sf(x));

    /// <inheritdoc />
    public double Ppf(double q)
    {
        if (!(q >= 0 && q <= 1)) return double.NaN;
        if (q == 0) return _x[0] - 1;
        for (var i = 0; i < _x.Length; i++) if (_cum[i] >= q) return _x[i];
        return _x[^1];
    }

    /// <inheritdoc />
    public double Isf(double q)
    {
        if (!(q >= 0 && q <= 1)) return double.NaN;
        if (q == 1) return _x[0] - 1;
        for (var i = 0; i < _x.Length; i++) if (_tail[i] <= q) return _x[i];
        return _x[^1];
    }

    /// <inheritdoc />
    public double Moment(int n)
    {
        if (n < 0) throw new ArgumentException("Moment order must be non-negative.", nameof(n));
        var s = 0.0;
        for (var i = 0; i < _x.Length; i++) s += Math.Pow(_x[i], n) * _p[i];
        return s;
    }

    /// <inheritdoc />
    public DistributionMoments Stats()
    {
        var m = Moment(1);
        double m2 = 0, m3 = 0, m4 = 0;
        for (var i = 0; i < _x.Length; i++)
        {
            var d = _x[i] - m;
            m2 += d * d * _p[i];
            m3 += d * d * d * _p[i];
            m4 += d * d * d * d * _p[i];
        }
        return new DistributionMoments(m, m2, m3 / Math.Pow(m2, 1.5), m4 / (m2 * m2) - 3);
    }

    /// <inheritdoc />
    public double Mean() => Moment(1);
    /// <inheritdoc />
    public double Var() => Stats().Variance;
    /// <inheritdoc />
    public double Std() => Math.Sqrt(Var());
    /// <inheritdoc />
    public double Median() => Ppf(0.5);
    /// <inheritdoc />
    public double Skew() => Stats().Skewness;
    /// <inheritdoc />
    public double Kurtosis() => Stats().Kurtosis;
    /// <inheritdoc />
    public double Entropy() => -_p.Sum(v => v > 0 ? v * Math.Log(v) : 0.0);

    /// <inheritdoc />
    public (double Lower, double Upper) Interval(double confidence)
    {
        if (!(confidence >= 0 && confidence <= 1)) throw new ArgumentException("confidence must lie in [0, 1].", nameof(confidence));
        var a = 0.5 * (1 - confidence);
        return (Ppf(a), Isf(a));
    }

    /// <inheritdoc />
    public double Rvs(GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        var u = StatsRandom.U(rng);
        int lo = 0, hi = _cum.Length - 1;
        while (lo < hi) { var mid = (lo + hi) >>> 1; if (_cum[mid] >= u) hi = mid; else lo = mid + 1; }
        return _x[lo];
    }

    /// <inheritdoc />
    public double[] Rvs(int size, GraviRandom rng)
    {
        var r = new double[size];
        for (var i = 0; i < size; i++) r[i] = Rvs(rng);
        return r;
    }

    /// <summary>Draws with a seed (null: fresh).</summary>
    public double[] Rvs(int size, int? seed = null) => Rvs(size, StatsDistributions.MakeRng(seed));
}

/// <summary>
/// Contingency tables with fixed row and column sums, scipy's <c>random_table</c> (the multivariate
/// hypergeometric over tables, i.e. Fisher's exact-test null).
/// </summary>
public sealed class RandomTableDistribution
{
    /// <summary>Tables with the given non-negative integer row and column sums (equal totals).</summary>
    public RandomTableDistribution(int[] row, int[] col)
    {
        ArgumentNullException.ThrowIfNull(row);
        ArgumentNullException.ThrowIfNull(col);
        if (row.Length == 0 || col.Length == 0 || row.Any(v => v < 0) || col.Any(v => v < 0)) throw new ArgumentException("row and col must be non-empty non-negative counts.");
        if (row.Sum() != col.Sum()) throw new ArgumentException("row and col must have the same sum.", nameof(col));
        Row = (int[])row.Clone();
        Col = (int[])col.Clone();
        N = row.Sum();
    }

    /// <summary>Row sums.</summary>
    public int[] Row { get; }
    /// <summary>Column sums.</summary>
    public int[] Col { get; }
    /// <summary>Total count.</summary>
    public int N { get; }

    /// <summary>log P(table) = Σ log rᵢ! + Σ log cⱼ! − log n! − Σ log xᵢⱼ!; −∞ for a table with other margins.</summary>
    public double LogPmf(NdArray x)
    {
        ArgumentNullException.ThrowIfNull(x);
        int r = Row.Length, c = Col.Length;
        if (x.Shape.Length != 2 || x.Shape[0] != r || x.Shape[1] != c) throw new ArgumentException("x must have shape (len(row), len(col)).", nameof(x));
        var t = x.AsContiguous().ToArray();
        var rs = new double[r];
        var cs = new double[c];
        var s = -Special.GammaLn(N + 1);
        for (var i = 0; i < r; i++)
            for (var j = 0; j < c; j++)
            {
                var v = t[i * c + j];
                if (v < 0 || v != Math.Floor(v)) return double.NegativeInfinity;
                rs[i] += v;
                cs[j] += v;
                s -= Special.GammaLn(v + 1);
            }
        for (var i = 0; i < r; i++) { if (rs[i] != Row[i]) return double.NegativeInfinity; s += Special.GammaLn(Row[i] + 1); }
        for (var j = 0; j < c; j++) { if (cs[j] != Col[j]) return double.NegativeInfinity; s += Special.GammaLn(Col[j] + 1); }
        return s;
    }

    /// <summary>P(table).</summary>
    public double Pmf(NdArray x) => Math.Exp(LogPmf(x));

    /// <summary>Expected table rᵢcⱼ/n.</summary>
    public NdArray Mean()
    {
        int r = Row.Length, c = Col.Length;
        var m = new double[r * c];
        for (var i = 0; i < r; i++)
            for (var j = 0; j < c; j++) m[i * c + j] = N == 0 ? 0 : (double)Row[i] * Col[j] / N;
        return new NdArray(m, r, c);
    }

    /// <summary>
    /// <paramref name="size"/> random tables. <c>"boyett"</c> shuffles n column labels and tallies them
    /// against the row labels (O(n)); <c>"patefield"</c> draws row by row from the conditional
    /// multivariate hypergeometric (O(r·c), better for large n) — the same distribution, so either is
    /// exact; scipy picks by size and so does a null <paramref name="method"/> here. The random streams
    /// differ from scipy's.
    /// </summary>
    public NdArray[] Rvs(int size, GraviRandom rng, string? method = null)
    {
        ArgumentNullException.ThrowIfNull(rng);
        int r = Row.Length, c = Col.Length;
        var m = method?.ToLowerInvariant() ?? (N > 10 * r * c ? "patefield" : "boyett");
        if (m != "boyett" && m != "patefield") throw new ArgumentException("method must be 'boyett' or 'patefield'.", nameof(method));
        var result = new NdArray[size];
        var labels = new int[N];
        for (var s = 0; s < size; s++)
        {
            var t = new double[r * c];
            if (m == "boyett")
            {
                var o = 0;
                for (var j = 0; j < c; j++) for (var k = 0; k < Col[j]; k++) labels[o++] = j;
                for (var i = N - 1; i > 0; i--) { var k = rng.Next(i + 1); (labels[i], labels[k]) = (labels[k], labels[i]); }
                o = 0;
                for (var i = 0; i < r; i++) for (var k = 0; k < Row[i]; k++) t[i * c + labels[o++]]++;
            }
            else
            {
                var remaining = (int[])Col.Clone();
                for (var i = 0; i < r - 1; i++)
                {
                    var draw = new MultivariateHypergeomDistribution(remaining, Row[i]).Rvs(1, rng).AsContiguous().ToArray();
                    for (var j = 0; j < c; j++) { t[i * c + j] = draw[j]; remaining[j] -= (int)draw[j]; }
                }
                for (var j = 0; j < c; j++) t[(r - 1) * c + j] = remaining[j];
            }
            result[s] = new NdArray(t, r, c);
        }
        return result;
    }

    /// <summary>Draws with a seed (null: fresh).</summary>
    public NdArray[] Rvs(int size, int? seed = null, string? method = null) => Rvs(size, StatsDistributions.MakeRng(seed), method);
}

public static partial class Stats
{
    /// <summary>Distribution from a histogram, scipy's <c>rv_histogram((counts, edges), density=…)</c>.</summary>
    public static RvHistogramDistribution RvHistogram(double[] counts, double[] edges, bool density = false) => new(counts, edges, density);

    /// <summary>Discrete distribution on given points, scipy's <c>rv_discrete(values=(xk, pk))</c>.</summary>
    public static RvDiscreteDistribution RvDiscrete(double[] xk, double[] pk) => new(xk, pk);

    /// <summary>Random contingency tables with fixed margins, scipy's <c>random_table</c>.</summary>
    public static RandomTableDistribution RandomTable(int[] row, int[] col) => new(row, col);

    /// <summary>
    /// A random correlation matrix with the given eigenvalues, scipy's <c>random_correlation.rvs</c>
    /// (Davies &amp; Higham 2000): a Haar-random orthogonal similarity of diag(eigs), then Givens rotations
    /// that set the diagonal to one without changing the spectrum. The eigenvalues must be
    /// non-negative and sum to their count (within <paramref name="tol"/>).
    /// </summary>
    public static NdArray RandomCorrelation(double[] eigs, int? seed = null, double tol = 1e-13, double diagTol = 1e-7)
    {
        ArgumentNullException.ThrowIfNull(eigs);
        var d = eigs.Length;
        if (d < 1) throw new ArgumentException("Array 'eigs' must be a vector of length greater than 1.", nameof(eigs));
        if (Math.Abs(eigs.Sum() - d) > tol) throw new ArgumentException("Sum of eigenvalues must equal dimensionality.", nameof(eigs));
        if (eigs.Any(v => v < -tol)) throw new ArgumentException("All eigenvalues must be non-negative.", nameof(eigs));
        var o = RandomMatrixGroups.OrthoGroup(d, StatsDistributions.MakeRng(seed)).AsContiguous().ToArray();
        var m = new double[d * d];
        for (var i = 0; i < d; i++)
            for (var j = 0; j < d; j++)
            {
                var s = 0.0;
                for (var k = 0; k < d; k++) s += o[i * d + k] * eigs[k] * o[j * d + k];
                m[i * d + j] = s;
            }
        for (var i = 0; i < d - 1; i++)
        {
            if (m[i * d + i] == 1) continue;
            var j = i + 1;
            if (m[i * d + i] > 1) { for (j = i + 1; j < d; j++) if (m[j * d + j] < 1) break; }
            else { for (j = i + 1; j < d; j++) if (m[j * d + j] > 1) break; }
            if (j == d) j = d - 1; // as scipy's loop variable, which stays at the last index
            var (c, s) = GivensToOne(m[i * d + i], m[j * d + j], m[i * d + j]);
            // Rows i, j then columns i, j: x' = c·x − s·y, y' = s·x + c·y (BLAS drot with −s).
            for (var k = 0; k < d; k++)
            {
                double x = m[i * d + k], y = m[j * d + k];
                m[i * d + k] = c * x - s * y;
                m[j * d + k] = s * x + c * y;
            }
            for (var k = 0; k < d; k++)
            {
                double x = m[k * d + i], y = m[k * d + j];
                m[k * d + i] = c * x - s * y;
                m[k * d + j] = s * x + c * y;
            }
        }
        for (var i = 0; i < d; i++)
            if (Math.Abs(m[i * d + i] - 1) > diagTol) throw new ArithmeticException("Failed to generate a valid correlation matrix");
        return new NdArray(m, d, d);
    }

    private static (double C, double S) GivensToOne(double aii, double ajj, double aij)
    {
        var aiid = aii - 1;
        var ajjd = ajj - 1;
        if (ajjd == 0) return (0, 1);
        var dd = Math.Sqrt(Math.Max(aij * aij - aiid * ajjd, 0));
        var t = (aij + Math.CopySign(dd, aij)) / ajjd;
        var c = 1 / Math.Sqrt(1 + t * t);
        var s = c == 0 ? 1.0 : c * t;
        return (c, s);
    }
}
