using System.Runtime.Intrinsics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Kernel density estimation with Gaussian kernels, scipy's <c>gaussian_kde</c>: the density is the
/// weighted mixture Σ wᵢ N(x; xᵢ, Σ) with Σ = factor² × the (weighted) data covariance.
/// </summary>
/// <remarks>
/// <para>Evaluation is exact (every kernel at every point) and runs in parallel over the evaluation
/// points; the 1-D kernel sum is vectorised with <see cref="Vector256"/> and checked against the scalar
/// path in the tests. <see cref="EvaluateBinned"/> is the explicitly approximate alternative for very
/// large 1-D problems: linear binning plus a truncated discrete convolution.</para>
/// <para>Data are d×n as in scipy (one column per observation); a <c>double[]</c> is 1-D data.</para>
/// </remarks>
public sealed class GaussianKde
{
    private readonly double[] _data;       // d×n row-major (dimension-major, as scipy's dataset)
    private readonly double[] _weights;    // normalised to sum 1
    private readonly double[] _dataCov;    // d×d
    private readonly double[] _dataChol;   // lower Cholesky factor of the data covariance
    private double[] _chol = [];           // lower Cholesky factor of the kernel covariance
    private double[] _whitened = [];       // n×d: L⁻¹ xᵢ, for the kernel sums
    private double _logNorm;               // log of (2π)^{d/2} det(L)
    private Func<GaussianKde, double>? _bwFunc;

    /// <summary>Creates an estimator from d×n data (rows are dimensions, columns observations).</summary>
    /// <param name="dataset">The data, shape (d, n), or (n) for 1-D data.</param>
    /// <param name="bwMethod"><c>"scott"</c> (default) or <c>"silverman"</c>.</param>
    /// <param name="weights">Optional non-negative observation weights.</param>
    public GaussianKde(NdArray dataset, string bwMethod = "scott", double[]? weights = null)
        : this(dataset, weights) => SetBandwidth(bwMethod);

    /// <summary>Creates an estimator with a fixed bandwidth factor.</summary>
    public GaussianKde(NdArray dataset, double bwFactor, double[]? weights = null)
        : this(dataset, weights) => SetBandwidth(bwFactor);

    /// <summary>Creates an estimator whose bandwidth factor is computed from the estimator (scipy's callable bw_method).</summary>
    public GaussianKde(NdArray dataset, Func<GaussianKde, double> bwMethod, double[]? weights = null)
        : this(dataset, weights) => SetBandwidth(bwMethod);

    /// <summary>1-D data.</summary>
    public GaussianKde(double[] dataset, string bwMethod = "scott", double[]? weights = null)
        : this(new NdArray((double[])dataset.Clone(), 1, dataset.Length), bwMethod, weights) { }

    /// <summary>1-D data with a fixed bandwidth factor.</summary>
    public GaussianKde(double[] dataset, double bwFactor, double[]? weights = null)
        : this(new NdArray((double[])dataset.Clone(), 1, dataset.Length), bwFactor, weights) { }

    private GaussianKde(NdArray dataset, double[]? weights)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        if (dataset.Rank == 1) dataset = new NdArray(dataset.AsContiguous().ToArray(), 1, dataset.Size);
        if (dataset.Rank != 2) throw new ArgumentException("dataset must be 1-D or 2-D (d, n).", nameof(dataset));
        D = dataset.Shape[0];
        N = dataset.Shape[1];
        if (N <= 1) throw new ArgumentException("`dataset` input should have multiple elements.", nameof(dataset));
        _data = dataset.AsContiguous().ToArray();
        if (_data.Any(v => !double.IsFinite(v))) throw new ArgumentException("dataset must be finite.", nameof(dataset));
        _weights = new double[N];
        if (weights is null)
            for (var i = 0; i < N; i++) _weights[i] = 1.0 / N;
        else
        {
            if (weights.Length != N) throw new ArgumentException("`weights` input should be of length n", nameof(weights));
            var s = 0.0;
            foreach (var w in weights)
            {
                if (!(w >= 0) || double.IsInfinity(w)) throw new ArgumentException("weights must be non-negative and finite.", nameof(weights));
                s += w;
            }
            for (var i = 0; i < N; i++) _weights[i] = weights[i] / s;
        }
        var sw2 = 0.0;
        foreach (var w in _weights) sw2 += w * w;
        Neff = 1.0 / sw2;
        if (D > N) throw new ArgumentException("Number of dimensions is greater than number of samples. This results in a singular data covariance matrix.", nameof(dataset));
        // numpy's cov with aweights and ddof 1: Σ w (x − x̄)(x − x̄)ᵀ / (1 − Σw²) for normalised w.
        _dataCov = new double[D * D];
        var mean = new double[D];
        for (var a = 0; a < D; a++)
            for (var i = 0; i < N; i++) mean[a] += _weights[i] * _data[a * N + i];
        var fact = 1 - sw2;
        for (var a = 0; a < D; a++)
            for (var b = a; b < D; b++)
            {
                var s = 0.0;
                for (var i = 0; i < N; i++) s += _weights[i] * (_data[a * N + i] - mean[a]) * (_data[b * N + i] - mean[b]);
                _dataCov[a * D + b] = _dataCov[b * D + a] = s / fact;
            }
        _dataChol = CholeskyLower(_dataCov, D);
    }

    /// <summary>Number of dimensions d.</summary>
    public int D { get; }

    /// <summary>Number of observations n.</summary>
    public int N { get; }

    /// <summary>Effective sample size 1/Σwᵢ² (n when unweighted).</summary>
    public double Neff { get; }

    /// <summary>The normalised weights.</summary>
    public double[] Weights => (double[])_weights.Clone();

    /// <summary>The data, d×n.</summary>
    public NdArray Dataset => new((double[])_data.Clone(), D, N);

    /// <summary>The bandwidth factor multiplying the data covariance's square root.</summary>
    public double Factor { get; private set; }

    /// <summary>The kernel covariance, factor² × the data covariance (d×d).</summary>
    public NdArray Covariance { get; private set; } = null!;

    /// <summary>The inverse of <see cref="Covariance"/>.</summary>
    public NdArray InvCov
    {
        get
        {
            var inv = LinAlg.Inverse(Covariance);
            return inv;
        }
    }

    /// <summary>Scott's factor n_eff^(−1/(d+4)).</summary>
    public double ScottsFactor() => Math.Pow(Neff, -1.0 / (D + 4));

    /// <summary>Silverman's factor (n_eff (d+2)/4)^(−1/(d+4)).</summary>
    public double SilvermanFactor() => Math.Pow(Neff * (D + 2.0) / 4.0, -1.0 / (D + 4));

    /// <summary>Sets the bandwidth rule by name (<c>"scott"</c> or <c>"silverman"</c>) and recomputes the kernel.</summary>
    public void SetBandwidth(string bwMethod)
    {
        _bwFunc = bwMethod.ToLowerInvariant() switch
        {
            "scott" => k => k.ScottsFactor(),
            "silverman" => k => k.SilvermanFactor(),
            _ => throw new ArgumentException("`bw_method` should be 'scott', 'silverman', a scalar or a callable.", nameof(bwMethod)),
        };
        ComputeCovariance();
    }

    /// <summary>Sets a fixed bandwidth factor and recomputes the kernel.</summary>
    public void SetBandwidth(double factor)
    {
        if (!(factor > 0) || double.IsInfinity(factor)) throw new ArgumentException("The bandwidth factor must be positive and finite.", nameof(factor));
        _bwFunc = _ => factor;
        ComputeCovariance();
    }

    /// <summary>Sets a bandwidth rule computed from the estimator and recomputes the kernel.</summary>
    public void SetBandwidth(Func<GaussianKde, double> bwMethod)
    {
        ArgumentNullException.ThrowIfNull(bwMethod);
        _bwFunc = bwMethod;
        ComputeCovariance();
    }

    private void ComputeCovariance()
    {
        Factor = _bwFunc!(this);
        var f2 = Factor * Factor;
        var cov = new double[D * D];
        for (var i = 0; i < cov.Length; i++) cov[i] = _dataCov[i] * f2;
        Covariance = new NdArray(cov, D, D);
        _chol = new double[D * D];
        for (var i = 0; i < _chol.Length; i++) _chol[i] = _dataChol[i] * Factor;
        _logNorm = 0;
        for (var a = 0; a < D; a++) _logNorm += Math.Log(_chol[a * D + a] * Math.Sqrt(2 * Math.PI));
        // Whiten the data once: every kernel becomes a standard normal in these coordinates.
        _whitened = new double[N * D];
        var col = new double[D];
        for (var i = 0; i < N; i++)
        {
            for (var a = 0; a < D; a++) col[a] = _data[a * N + i];
            ForwardSolve(_chol, D, col);
            for (var a = 0; a < D; a++) _whitened[i * D + a] = col[a];
        }
    }

    // ------------------------------------------------------------------ evaluation

    /// <summary>The density at m points given as a d×m array (or a 1-D array when d = 1).</summary>
    public double[] Evaluate(NdArray points) => EvaluateCore(PointsToRows(points), log: false);

    /// <summary>The density at 1-D points.</summary>
    public double[] Evaluate(double[] points)
    {
        if (D != 1) throw new ArgumentException($"points have dimension 1, dataset has dimension {D}", nameof(points));
        return EvaluateCore(points, log: false);
    }

    /// <summary>The density (scipy's <c>pdf</c>, an alias of <c>evaluate</c>).</summary>
    public double[] Pdf(double[] points) => Evaluate(points);

    /// <summary>The density at d-dimensional points.</summary>
    public double[] Pdf(NdArray points) => Evaluate(points);

    /// <summary>The log density, computed with log-sum-exp so it stays finite far in the tails.</summary>
    public double[] LogPdf(double[] points)
    {
        if (D != 1) throw new ArgumentException($"points have dimension 1, dataset has dimension {D}", nameof(points));
        return EvaluateCore(points, log: true);
    }

    /// <summary>The log density at d-dimensional points.</summary>
    public double[] LogPdf(NdArray points) => EvaluateCore(PointsToRows(points), log: true);

    private double[] PointsToRows(NdArray points)
    {
        ArgumentNullException.ThrowIfNull(points);
        var p = points.AsContiguous().ToArray();
        int d, m;
        if (points.Rank == 1) { d = 1; m = p.Length; if (D != 1) { d = p.Length; m = 1; } }
        else { d = points.Shape[0]; m = points.Shape[1]; }
        if (d != D)
        {
            if (points.Rank == 2 && points.Shape[1] == D && points.Shape[0] != D)
                throw new ArgumentException($"points have dimension {d}, dataset has dimension {D}; pass points as a (d, m) array.", nameof(points));
            throw new ArgumentException($"points have dimension {d}, dataset has dimension {D}", nameof(points));
        }
        // To point-major m×d.
        var rows = new double[m * D];
        for (var a = 0; a < D; a++)
            for (var j = 0; j < m; j++) rows[j * D + a] = p[a * m + j];
        return rows;
    }

    private double[] EvaluateCore(double[] rows, bool log)
    {
        var m = rows.Length / D;
        var result = new double[m];
        // Whiten the points; then the kernel at (point, xᵢ) is exp(−½|zₚ − zᵢ|²).
        var z = new double[rows.Length];
        Array.Copy(rows, z, rows.Length);
        for (var j = 0; j < m; j++) ForwardSolve(_chol, D, z.AsSpan(j * D, D));
        void One(int j)
        {
            if (log) result[j] = LogKernelSum(z.AsSpan(j * D, D)) - _logNorm;
            else result[j] = (D == 1 ? KernelSum1d(z[j]) : KernelSum(z.AsSpan(j * D, D))) * Math.Exp(-_logNorm);
        }
        if ((long)m * N >= 1 << 16) Parallel.For(0, m, One);
        else for (var j = 0; j < m; j++) One(j);
        return result;
    }

    private double KernelSum(ReadOnlySpan<double> zp)
    {
        var s = 0.0;
        for (var i = 0; i < N; i++)
        {
            var q = 0.0;
            for (var a = 0; a < D; a++)
            {
                var dlt = zp[a] - _whitened[i * D + a];
                q += dlt * dlt;
            }
            s += _weights[i] * Math.Exp(-0.5 * q);
        }
        return s;
    }

    /// <summary>The 1-D kernel sum, vectorised over observations.</summary>
    private double KernelSum1d(double zp)
    {
        var i = 0;
        var s = 0.0;
        if (Vector256.IsHardwareAccelerated && N >= 8)
        {
            var acc = Vector256<double>.Zero;
            var vz = Vector256.Create(zp);
            var half = Vector256.Create(-0.5);
            var w = _whitened.AsSpan();
            var wt = _weights.AsSpan();
            for (; i + 4 <= N; i += 4)
            {
                var d = vz - Vector256.Create(w.Slice(i, 4));
                acc += Vector256.Create(wt.Slice(i, 4)) * Vector256.Exp(half * d * d);
            }
            s = Vector256.Sum(acc);
        }
        for (; i < N; i++)
        {
            var d = zp - _whitened[i];
            s += _weights[i] * Math.Exp(-0.5 * d * d);
        }
        return s;
    }

    private double LogKernelSum(ReadOnlySpan<double> zp)
    {
        var max = double.NegativeInfinity;
        var terms = new double[N];
        for (var i = 0; i < N; i++)
        {
            var q = 0.0;
            for (var a = 0; a < D; a++)
            {
                var dlt = zp[a] - _whitened[i * D + a];
                q += dlt * dlt;
            }
            var t = Math.Log(_weights[i]) - 0.5 * q;
            terms[i] = t;
            if (t > max) max = t;
        }
        if (double.IsNegativeInfinity(max)) return max;
        var s = 0.0;
        foreach (var t in terms) s += Math.Exp(t - max);
        return max + Math.Log(s);
    }

    /// <summary>
    /// Approximate 1-D evaluation for large problems: the weights are linearly binned onto
    /// <paramref name="gridSize"/> points spanning the data ±8 bandwidths, convolved with the kernel
    /// truncated at 8 bandwidths, and interpolated at <paramref name="points"/>. O(n + g·w + m) instead of
    /// O(n·m); the relative error is typically 1e-4 or better where the density is not tiny. Not a
    /// replacement for <see cref="Evaluate(double[])"/> when exact values matter.
    /// </summary>
    public double[] EvaluateBinned(double[] points, int gridSize = 4096)
    {
        ArgumentNullException.ThrowIfNull(points);
        if (D != 1) throw new ArgumentException("EvaluateBinned supports 1-D data only.");
        if (gridSize < 16) throw new ArgumentException("gridSize must be at least 16.", nameof(gridSize));
        var h = Math.Sqrt(Covariance.AsContiguous().ToArray()[0]);
        double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
        foreach (var v in _data) { lo = Math.Min(lo, v); hi = Math.Max(hi, v); }
        lo -= 8 * h;
        hi += 8 * h;
        var delta = (hi - lo) / (gridSize - 1);
        var grid = new double[gridSize];
        for (var i = 0; i < N; i++)
        {
            var t = (_data[i] - lo) / delta;
            var k = Math.Min((int)t, gridSize - 2);
            var f = t - k;
            grid[k] += _weights[i] * (1 - f);
            grid[k + 1] += _weights[i] * f;
        }
        var half = Math.Min(gridSize - 1, (int)Math.Ceiling(8 * h / delta));
        var kernel = new double[2 * half + 1];
        var norm = 1.0 / (h * Math.Sqrt(2 * Math.PI));
        for (var j = -half; j <= half; j++)
        {
            var u = j * delta / h;
            kernel[j + half] = norm * Math.Exp(-0.5 * u * u);
        }
        var dens = new double[gridSize];
        Parallel.For(0, gridSize, g =>
        {
            var s = 0.0;
            var a = Math.Max(0, g - half);
            var b = Math.Min(gridSize - 1, g + half);
            for (var q = a; q <= b; q++) s += grid[q] * kernel[g - q + half];
            dens[g] = s;
        });
        var r = new double[points.Length];
        var outside = new List<int>();
        for (var j = 0; j < points.Length; j++)
        {
            var t = (points[j] - lo) / delta;
            if (!(t >= 0 && t <= gridSize - 1)) { outside.Add(j); continue; }
            var k = Math.Min((int)t, gridSize - 2);
            var f = t - k;
            r[j] = dens[k] * (1 - f) + dens[k + 1] * f;
        }
        // Points more than 8 bandwidths beyond the data: the density there is below 1e-14 of its peak,
        // but evaluate it exactly (in one parallel batch) rather than return a made-up zero.
        if (outside.Count > 0)
        {
            var exact = Evaluate(outside.Select(j => points[j]).ToArray());
            for (var q = 0; q < outside.Count; q++) r[outside[q]] = exact[q];
        }
        return r;
    }

    // ------------------------------------------------------------------ integrals

    /// <summary>∫ kde(x) · N(x; mean, cov) dx = Σ wᵢ N(mean; xᵢ, Σ + cov), scipy's <c>integrate_gaussian</c>.</summary>
    public double IntegrateGaussian(double[] mean, NdArray cov)
    {
        ArgumentNullException.ThrowIfNull(mean);
        ArgumentNullException.ThrowIfNull(cov);
        if (mean.Length != D) throw new ArgumentException($"mean does not have dimension {D}", nameof(mean));
        var c = cov.AsContiguous().ToArray();
        if (c.Length != D * D) throw new ArgumentException($"covariance does not have dimension {D}", nameof(cov));
        var k = Covariance.AsContiguous().ToArray();
        var sum = new double[D * D];
        for (var i = 0; i < sum.Length; i++) sum[i] = k[i] + c[i];
        return GaussianMixtureAt(sum, mean, Enumerable.Range(0, N), i => _weights[i], i => Column(i));
    }

    /// <summary>∫ over [low, high] of a 1-D density, Σ wᵢ [Φ((high − xᵢ)/h) − Φ((low − xᵢ)/h)].</summary>
    public double IntegrateBox1d(double low, double high)
    {
        if (D != 1) throw new ArgumentException("integrate_box_1d() only handles 1D pdfs");
        var h = Math.Sqrt(Covariance.AsContiguous().ToArray()[0]);
        var s = 0.0;
        for (var i = 0; i < N; i++)
            s += _weights[i] * (Special.Ndtr((high - _data[i]) / h) - Special.Ndtr((low - _data[i]) / h));
        return s;
    }

    /// <summary>
    /// ∫ over the box [low, high] (scipy's <c>integrate_box</c>): exact for d = 1 and d = 2 (Genz's
    /// bivariate normal algorithm, ~1e-15), otherwise a weighted sum of multivariate normal box
    /// probabilities by Genz's quasi-Monte Carlo algorithm (absolute error ≈ 1e-6). scipy uses the
    /// randomised algorithm for every d ≥ 2, so its own values carry ~1e-5 error.
    /// </summary>
    public double IntegrateBox(double[] lowBounds, double[] highBounds, int? maxpts = null, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(lowBounds);
        ArgumentNullException.ThrowIfNull(highBounds);
        if (lowBounds.Length != D || highBounds.Length != D) throw new ArgumentException($"bounds must have dimension {D}.");
        if (D == 1) return IntegrateBox1d(lowBounds[0], highBounds[0]);
        if (D == 2)
        {
            // Exact bivariate normal box probabilities (Genz's BVNU), not Monte Carlo.
            var cv = Covariance.AsContiguous().ToArray();
            double s1 = Math.Sqrt(cv[0]), s2 = Math.Sqrt(cv[3]);
            var rho = cv[1] / (s1 * s2);
            var acc = 0.0;
            for (var i = 0; i < N; i++)
            {
                double l1 = (lowBounds[0] - _data[i]) / s1, h1 = (highBounds[0] - _data[i]) / s1;
                double l2 = (lowBounds[1] - _data[N + i]) / s2, h2 = (highBounds[1] - _data[N + i]) / s2;
                var p = StatsBivariateNormal.Upper(l1, l2, rho) - StatsBivariateNormal.Upper(h1, l2, rho)
                      - StatsBivariateNormal.Upper(l1, h2, rho) + StatsBivariateNormal.Upper(h1, h2, rho);
                acc += _weights[i] * Math.Max(p, 0);
            }
            return acc;
        }
        var mvn = new MultivariateNormalDistribution(new double[D], Covariance);
        var s = 0.0;
        for (var i = 0; i < N; i++)
        {
            var lo = new double[D];
            var hi = new double[D];
            for (var a = 0; a < D; a++) { lo[a] = lowBounds[a] - _data[a * N + i]; hi[a] = highBounds[a] - _data[a * N + i]; }
            s += _weights[i] * mvn.Cdf(hi, lo, maxpts, 1e-6, 1e-6, seed);
        }
        return s;
    }

    /// <summary>∫ kde(x) · other(x) dx for another KDE of the same dimension, scipy's <c>integrate_kde</c>.</summary>
    public double IntegrateKde(GaussianKde other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (other.D != D) throw new ArgumentException("KDEs are not the same dimensionality", nameof(other));
        var a = Covariance.AsContiguous().ToArray();
        var b = other.Covariance.AsContiguous().ToArray();
        var sum = new double[D * D];
        for (var i = 0; i < sum.Length; i++) sum[i] = a[i] + b[i];
        var chol = CholeskyLower(sum, D);
        var logNorm = 0.0;
        for (var q = 0; q < D; q++) logNorm += Math.Log(chol[q * D + q] * Math.Sqrt(2 * Math.PI));
        var total = 0.0;
        var lockObj = new object();
        Parallel.For(0, other.N, () => 0.0, (j, _, local) =>
        {
            var diff = new double[D];
            for (var i = 0; i < N; i++)
            {
                for (var q = 0; q < D; q++) diff[q] = _data[q * N + i] - other._data[q * other.N + j];
                ForwardSolve(chol, D, diff);
                var e = 0.0;
                foreach (var v in diff) e += v * v;
                local += _weights[i] * other._weights[j] * Math.Exp(-0.5 * e - logNorm);
            }
            return local;
        }, local => { lock (lockObj) total += local; });
        return total;
    }

    private double GaussianMixtureAt(double[] cov, double[] at, IEnumerable<int> idx, Func<int, double> w, Func<int, double[]> center)
    {
        var chol = CholeskyLower(cov, D);
        var logNorm = 0.0;
        for (var q = 0; q < D; q++) logNorm += Math.Log(chol[q * D + q] * Math.Sqrt(2 * Math.PI));
        var s = 0.0;
        var diff = new double[D];
        foreach (var i in idx)
        {
            var c = center(i);
            for (var q = 0; q < D; q++) diff[q] = at[q] - c[q];
            ForwardSolve(chol, D, diff);
            var e = 0.0;
            foreach (var v in diff) e += v * v;
            s += w(i) * Math.Exp(-0.5 * e - logNorm);
        }
        return s;
    }

    private double[] Column(int i)
    {
        var c = new double[D];
        for (var a = 0; a < D; a++) c[a] = _data[a * N + i];
        return c;
    }

    // ------------------------------------------------------------------ sampling and marginals

    /// <summary>
    /// Draws <paramref name="size"/> points from the estimated density (an observation chosen by weight
    /// plus N(0, Σ) noise), returned d×size as scipy.
    /// </summary>
    public NdArray Resample(int size, int? seed = null)
    {
        if (size < 0) throw new ArgumentException("size must be non-negative.", nameof(size));
        var rng = new GraviRandom(seed ?? Random.Shared.Next());
        var cum = new double[N];
        var acc = 0.0;
        for (var i = 0; i < N; i++) { acc += _weights[i]; cum[i] = acc; }
        var outp = new double[D * size];
        var z = new double[D];
        for (var j = 0; j < size; j++)
        {
            var u = rng.NextDouble() * acc;
            var pos = Array.BinarySearch(cum, u);
            var idx = Math.Min(pos < 0 ? ~pos : pos, N - 1);
            for (var a = 0; a < D; a++) z[a] = rng.Normal();
            for (var a = 0; a < D; a++)
            {
                var s = 0.0;
                for (var q = 0; q <= a; q++) s += _chol[a * D + q] * z[q];
                outp[a * size + j] = _data[a * N + idx] + s;
            }
        }
        return new NdArray(outp, D, size);
    }

    /// <summary>
    /// The KDE of the marginal distribution over the given dimensions (scipy's <c>marginal</c>), keeping
    /// the weights and this estimator's bandwidth factor.
    /// </summary>
    public GaussianKde Marginal(params int[] dimensions)
    {
        ArgumentNullException.ThrowIfNull(dimensions);
        var dims = dimensions.Select(v => v < 0 ? v + D : v).ToArray();
        if (dims.Any(v => v < 0 || v >= D) || dims.Distinct().Count() != dims.Length)
            throw new ArgumentException("Dimensions must be unique, in-range integers.", nameof(dimensions));
        var data = new double[dims.Length * N];
        for (var k = 0; k < dims.Length; k++) Array.Copy(_data, dims[k] * N, data, k * N, N);
        var f = Factor;
        return new GaussianKde(new NdArray(data, dims.Length, N), f, _weights);
    }

    // ------------------------------------------------------------------ small dense helpers

    private static double[] CholeskyLower(double[] a, int n)
    {
        var l = new double[n * n];
        for (var j = 0; j < n; j++)
        {
            var s = a[j * n + j];
            for (var k = 0; k < j; k++) s -= l[j * n + k] * l[j * n + k];
            if (!(s > 0)) throw new ArgumentException("The data covariance matrix is not positive definite (the data are degenerate in some direction).");
            l[j * n + j] = Math.Sqrt(s);
            for (var i = j + 1; i < n; i++)
            {
                var t = a[i * n + j];
                for (var k = 0; k < j; k++) t -= l[i * n + k] * l[j * n + k];
                l[i * n + j] = t / l[j * n + j];
            }
        }
        return l;
    }

    private static void ForwardSolve(double[] l, int n, Span<double> b)
    {
        for (var i = 0; i < n; i++)
        {
            var s = b[i];
            for (var k = 0; k < i; k++) s -= l[i * n + k] * b[k];
            b[i] = s / l[i * n + i];
        }
    }
}
