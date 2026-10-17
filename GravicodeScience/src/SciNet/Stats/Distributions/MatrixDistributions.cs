using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Small dense helpers for the matrix-variate distributions (row-major, no allocations beyond the results).</summary>
internal static class StatsMatrixMath
{
    public static double[] Square(NdArray a, string name, out int n)
    {
        ArgumentNullException.ThrowIfNull(a, name);
        if (a.Shape.Length != 2 || a.Shape[0] != a.Shape[1] || a.Shape[0] == 0) throw new ArgumentException($"{name} must be a non-empty square matrix.", name);
        n = a.Shape[0];
        return a.AsContiguous().ToArray();
    }

    public static double[] Identity(int n)
    {
        var r = new double[n * n];
        for (var i = 0; i < n; i++) r[i * n + i] = 1;
        return r;
    }

    /// <summary>C = A·B for A (m×k), B (k×n).</summary>
    public static double[] Mul(double[] a, double[] b, int m, int k, int n)
    {
        var c = new double[m * n];
        for (var i = 0; i < m; i++)
            for (var p = 0; p < k; p++)
            {
                var v = a[i * k + p];
                if (v == 0) continue;
                for (var j = 0; j < n; j++) c[i * n + j] += v * b[p * n + j];
            }
        return c;
    }

    public static double[] Transpose(double[] a, int m, int n)
    {
        var t = new double[m * n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++) t[j * m + i] = a[i * n + j];
        return t;
    }

    /// <summary>log|det A| and its sign by partial-pivot LU.</summary>
    public static (double LogAbsDet, int Sign) SlogDet(double[] m, int n)
    {
        var a = (double[])m.Clone();
        var sign = 1;
        var logdet = 0.0;
        for (var c = 0; c < n; c++)
        {
            var p = c;
            for (var r = c + 1; r < n; r++) if (Math.Abs(a[r * n + c]) > Math.Abs(a[p * n + c])) p = r;
            if (a[p * n + c] == 0) return (double.NegativeInfinity, 0);
            if (p != c) { for (var k = 0; k < n; k++) (a[p * n + k], a[c * n + k]) = (a[c * n + k], a[p * n + k]); sign = -sign; }
            var piv = a[c * n + c];
            if (piv < 0) sign = -sign;
            logdet += Math.Log(Math.Abs(piv));
            for (var r = c + 1; r < n; r++)
            {
                var f = a[r * n + c] / piv;
                if (f == 0) continue;
                for (var k = c; k < n; k++) a[r * n + k] -= f * a[c * n + k];
            }
        }
        return (logdet, sign);
    }
}

/// <summary>
/// The matrix normal distribution, scipy's <c>matrix_normal</c>: X (m×n) with vec(X) ~ N(vec(M), V ⊗ U),
/// U the m×m row covariance and V the n×n column covariance.
/// </summary>
public sealed class MatrixNormalDistribution
{
    private readonly double[] _mean, _rowChol, _colChol;

    /// <summary>Matrix normal with mean <paramref name="mean"/> (m×n; zeros when null) and positive-definite row/column covariances.</summary>
    public MatrixNormalDistribution(NdArray? mean, NdArray rowcov, NdArray colcov)
    {
        var u = StatsMatrixMath.Square(rowcov, nameof(rowcov), out var m);
        var v = StatsMatrixMath.Square(colcov, nameof(colcov), out var n);
        Rows = m;
        Cols = n;
        if (mean != null && (mean.Shape.Length != 2 || mean.Shape[0] != m || mean.Shape[1] != n)) throw new ArgumentException("mean must have shape (rows of rowcov, columns of colcov).", nameof(mean));
        _mean = mean?.AsContiguous().ToArray() ?? new double[m * n];
        RowCov = u;
        ColCov = v;
        _rowChol = StatsMultivariate.CholeskyLower(u, m, nameof(rowcov));
        _colChol = StatsMultivariate.CholeskyLower(v, n, nameof(colcov));
    }

    /// <summary>Number of rows m.</summary>
    public int Rows { get; }
    /// <summary>Number of columns n.</summary>
    public int Cols { get; }
    internal double[] RowCov { get; }
    internal double[] ColCov { get; }

    /// <summary>The mean matrix.</summary>
    public NdArray Mean => new((double[])_mean.Clone(), Rows, Cols);

    /// <summary>log density: −(mn/2) log 2π − (n/2) log|U| − (m/2) log|V| − ½ tr(V⁻¹(X − M)ᵀU⁻¹(X − M)).</summary>
    public double LogPdf(NdArray x)
    {
        ArgumentNullException.ThrowIfNull(x);
        int m = Rows, n = Cols;
        if (x.Shape.Length != 2 || x.Shape[0] != m || x.Shape[1] != n) throw new ArgumentException("x has the wrong shape.", nameof(x));
        var d = x.AsContiguous().ToArray();
        for (var i = 0; i < d.Length; i++) d[i] -= _mean[i];
        var uinvD = StatsMultivariate.CholeskySolve(_rowChol, m, d, n);           // U⁻¹ D (m×n)
        var vinvDt = StatsMultivariate.CholeskySolve(_colChol, n, StatsMatrixMath.Transpose(d, m, n), m); // V⁻¹ Dᵀ (n×m)
        // tr(V⁻¹ Dᵀ U⁻¹ D) = Σ_ij (V⁻¹Dᵀ)_ji (U⁻¹D)_ij
        var tr = 0.0;
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++) tr += vinvDt[j * m + i] * uinvD[i * n + j];
        return -0.5 * m * n * Math.Log(2 * Math.PI) - 0.5 * n * StatsMultivariate.LogDetFromCholesky(_rowChol, m)
               - 0.5 * m * StatsMultivariate.LogDetFromCholesky(_colChol, n) - 0.5 * tr;
    }

    /// <summary>Density.</summary>
    public double Pdf(NdArray x) => Math.Exp(LogPdf(x));

    /// <summary>Differential entropy ½mn(1 + log 2π) + (n/2) log|U| + (m/2) log|V|.</summary>
    public double Entropy() => 0.5 * Rows * Cols * (1 + Math.Log(2 * Math.PI)) + 0.5 * Cols * StatsMultivariate.LogDetFromCholesky(_rowChol, Rows)
                               + 0.5 * Rows * StatsMultivariate.LogDetFromCholesky(_colChol, Cols);

    /// <summary><paramref name="size"/> draws M + A Z Bᵀ with A Aᵀ = U, B Bᵀ = V.</summary>
    public NdArray[] Rvs(int size, GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        int m = Rows, n = Cols;
        var r = new NdArray[size];
        var bt = StatsMatrixMath.Transpose(_colChol, n, n);
        for (var s = 0; s < size; s++)
        {
            var z = new double[m * n];
            for (var i = 0; i < z.Length; i++) z[i] = StatsRandom.Normal(rng);
            var x = StatsMatrixMath.Mul(StatsMatrixMath.Mul(_rowChol, z, m, m, n), bt, m, n, n);
            for (var i = 0; i < x.Length; i++) x[i] += _mean[i];
            r[s] = new NdArray(x, m, n);
        }
        return r;
    }

    /// <summary>Draws with a seed (null: fresh).</summary>
    public NdArray[] Rvs(int size, int? seed = null) => Rvs(size, StatsDistributions.MakeRng(seed));
}

/// <summary>
/// The matrix t distribution, scipy's <c>matrix_t</c> (Gupta &amp; Nagar 2000, §4.2): for X (m×n),
/// log f = log Γ_n((ν+m+n−1)/2) − log Γ_n((ν+n−1)/2) − (mn/2) log π − (n/2) log|U| − (m/2) log|V|
/// − ((ν+m+n−1)/2) log|I_n + V⁻¹(X − M)ᵀU⁻¹(X − M)|.
/// </summary>
public sealed class MatrixTDistribution
{
    private readonly double[] _mean, _rowChol, _colChol;

    /// <summary>Matrix t with mean (zeros when null), row and column spread matrices and degrees of freedom df &gt; 0.</summary>
    public MatrixTDistribution(NdArray? mean, NdArray rowSpread, NdArray colSpread, double df = 1.0)
    {
        var u = StatsMatrixMath.Square(rowSpread, nameof(rowSpread), out var m);
        var v = StatsMatrixMath.Square(colSpread, nameof(colSpread), out var n);
        if (!(df > 0) || double.IsInfinity(df)) throw new ArgumentException("Degrees of freedom must be positive.", nameof(df));
        Rows = m;
        Cols = n;
        Df = df;
        if (mean != null && (mean.Shape.Length != 2 || mean.Shape[0] != m || mean.Shape[1] != n)) throw new ArgumentException("mean must have shape (rows of rowSpread, columns of colSpread).", nameof(mean));
        _mean = mean?.AsContiguous().ToArray() ?? new double[m * n];
        _rowSpread = u;
        _colSpread = v;
        _rowChol = StatsMultivariate.CholeskyLower(u, m, nameof(rowSpread));
        _colChol = StatsMultivariate.CholeskyLower(v, n, nameof(colSpread));
    }

    private readonly double[] _rowSpread, _colSpread;

    /// <summary>Number of rows m.</summary>
    public int Rows { get; }
    /// <summary>Number of columns n.</summary>
    public int Cols { get; }
    /// <summary>Degrees of freedom ν.</summary>
    public double Df { get; }

    /// <summary>Log density (see the class remarks).</summary>
    public double LogPdf(NdArray x)
    {
        ArgumentNullException.ThrowIfNull(x);
        int m = Rows, n = Cols;
        if (x.Shape.Length != 2 || x.Shape[0] != m || x.Shape[1] != n) throw new ArgumentException("x has the wrong shape.", nameof(x));
        var d = x.AsContiguous().ToArray();
        for (var i = 0; i < d.Length; i++) d[i] -= _mean[i];
        var uinvD = StatsMultivariate.CholeskySolve(_rowChol, m, d, n);            // U⁻¹D (m×n)
        var dtUinvD = StatsMatrixMath.Mul(StatsMatrixMath.Transpose(d, m, n), uinvD, n, m, n); // DᵀU⁻¹D (n×n)
        var arg = StatsMultivariate.CholeskySolve(_colChol, n, dtUinvD, n);        // V⁻¹DᵀU⁻¹D
        for (var i = 0; i < n; i++) arg[i * n + i] += 1;
        var (logdet, _) = StatsMatrixMath.SlogDet(arg, n);
        double nu = Df;
        return -((nu + m + n - 1) / 2) * logdet
               + StatsMultivariate.MultiGammaLn((nu + m + n - 1) / 2, n) - StatsMultivariate.MultiGammaLn((nu + n - 1) / 2, n)
               - m * n / 2.0 * Math.Log(Math.PI) - n / 2.0 * StatsMultivariate.LogDetFromCholesky(_rowChol, m)
               - m / 2.0 * StatsMultivariate.LogDetFromCholesky(_colChol, n);
    }

    /// <summary>Density.</summary>
    public double Pdf(NdArray x) => Math.Exp(LogPdf(x));

    /// <summary>The mean M, defined for ν &gt; 1 (NaN otherwise).</summary>
    public NdArray Mean() => new(Df > 1 ? (double[])_mean.Clone() : Enumerable.Repeat(double.NaN, _mean.Length).ToArray(), Rows, Cols);

    /// <summary>
    /// <paramref name="size"/> draws, as scipy: the Cholesky factor of an inverse-Wishart(ν + p − 1, Σ) draw
    /// on the smaller dimension p times a Gaussian matrix times the Cholesky factor of the other spread.
    /// </summary>
    public NdArray[] Rvs(int size, GraviRandom rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        int m = Rows, n = Cols;
        var r = new NdArray[size];
        var rowIw = m <= n ? new InvWishartDistribution(Df + m - 1, new NdArray(_rowSpread, m, m)) : null;
        var colIw = m > n ? new InvWishartDistribution(Df + n - 1, new NdArray(_colSpread, n, n)) : null;
        for (var s = 0; s < size; s++)
        {
            var a = rowIw != null ? StatsMultivariate.CholeskyLower(rowIw.Rvs(1, rng)[0].AsContiguous().ToArray(), m, "draw") : _rowChol;
            var b = colIw != null ? StatsMultivariate.CholeskyLower(colIw.Rvs(1, rng)[0].AsContiguous().ToArray(), n, "draw") : _colChol;
            var z = new double[m * n];
            for (var i = 0; i < z.Length; i++) z[i] = StatsRandom.Normal(rng);
            var x = StatsMatrixMath.Mul(StatsMatrixMath.Mul(a, z, m, m, n), StatsMatrixMath.Transpose(b, n, n), m, n, n);
            for (var i = 0; i < x.Length; i++) x[i] += _mean[i];
            r[s] = new NdArray(x, m, n);
        }
        return r;
    }

    /// <summary>Draws with a seed (null: fresh).</summary>
    public NdArray[] Rvs(int size, int? seed = null) => Rvs(size, StatsDistributions.MakeRng(seed));
}

public static partial class Stats
{
    /// <summary>Matrix normal, scipy's <c>matrix_normal</c>.</summary>
    public static MatrixNormalDistribution MatrixNormal(NdArray? mean, NdArray rowcov, NdArray colcov) => new(mean, rowcov, colcov);

    /// <summary>Matrix t, scipy's <c>matrix_t</c>.</summary>
    public static MatrixTDistribution MatrixT(NdArray? mean, NdArray rowSpread, NdArray colSpread, double df = 1.0) => new(mean, rowSpread, colSpread, df);
}
