using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// A covariance matrix in a factored form, scipy's <c>stats.Covariance</c>: the representation decides
/// what is cheap, so a multivariate normal built from a precision matrix, a diagonal, a Cholesky
/// factor or an eigendecomposition never re-factorises Σ. Every form provides <see cref="Whiten(double[])"/>
/// (x ↦ y with Cov(y) = I on the support), <see cref="Colorize(double[])"/> (its inverse direction),
/// <see cref="LogPdet"/> and <see cref="Rank"/>.
/// </summary>
/// <remarks>Vectors are row vectors as in scipy: whitening an (n, d) array whitens each row.</remarks>
public abstract class Covariance
{
    /// <summary>Dimension d.</summary>
    public abstract int Dim { get; }

    /// <summary>Rank of Σ (d unless a diagonal/eigen form has zero entries).</summary>
    public abstract int Rank { get; }

    /// <summary>Log of the pseudo-determinant (product of the positive eigenvalues).</summary>
    public abstract double LogPdet { get; }

    /// <summary>The dense covariance matrix (computed on demand).</summary>
    public abstract NdArray CovarianceMatrix { get; }

    /// <summary>Whitens one vector.</summary>
    public abstract double[] Whiten(double[] x);

    /// <summary>Colours one vector (maps standard normal draws to N(0, Σ)).</summary>
    public abstract double[] Colorize(double[] x);

    /// <summary>True when x lies in the support (the range of Σ); always true for full-rank forms.</summary>
    public virtual bool InSupport(ReadOnlySpan<double> x) => true;

    /// <summary>Whitens each row of an (n, d) array.</summary>
    public NdArray Whiten(NdArray x) => MapRows(x, Whiten);

    /// <summary>Colours each row of an (n, d) array.</summary>
    public NdArray Colorize(NdArray x) => MapRows(x, Colorize);

    private NdArray MapRows(NdArray x, Func<double[], double[]> f)
    {
        ArgumentNullException.ThrowIfNull(x);
        var d = Dim;
        var a = x.AsContiguous().ToArray();
        if (x.Shape.Length == 1) return new NdArray(f(Check(a)), d);
        if (x.Shape.Length != 2 || x.Shape[1] != d) throw new ArgumentException("x must have shape (n, d).", nameof(x));
        var n = x.Shape[0];
        var r = new double[n * d];
        for (var i = 0; i < n; i++) f(a[(i * d)..((i + 1) * d)]).CopyTo(r, i * d);
        return new NdArray(r, n, d);
    }

    /// <summary>Validates a vector's length.</summary>
    protected double[] Check(double[] x)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Length != Dim) throw new ArgumentException("x has the wrong dimension.", nameof(x));
        return x;
    }

    /// <summary>Σ represented by its precision matrix P = Σ⁻¹ (and optionally Σ itself), scipy's <c>Covariance.from_precision</c>.</summary>
    public static CovViaPrecision FromPrecision(NdArray precision, NdArray? covariance = null) => new(precision, covariance);

    /// <summary>Diagonal Σ (zeros allowed: a singular, lower-rank covariance), scipy's <c>Covariance.from_diagonal</c>.</summary>
    public static CovViaDiagonal FromDiagonal(double[] diagonal) => new(diagonal);

    /// <summary>Σ = L Lᵀ from its lower Cholesky factor, scipy's <c>Covariance.from_cholesky</c>.</summary>
    public static CovViaCholesky FromCholesky(NdArray cholesky) => new(cholesky);

    /// <summary>Σ = V diag(w) Vᵀ from an eigendecomposition (zero eigenvalues allowed), scipy's <c>Covariance.from_eigendecomposition</c>.</summary>
    public static CovViaEigendecomposition FromEigendecomposition(double[] eigenvalues, NdArray eigenvectors) => new(eigenvalues, eigenvectors);
}

/// <summary>Covariance from a precision matrix: whitening is x·R with P = R Rᵀ (lower Cholesky), no inverse needed.</summary>
public sealed class CovViaPrecision : Covariance
{
    private readonly double[] _cholP;
    private readonly double[]? _cov;
    private readonly int _d;

    internal CovViaPrecision(NdArray precision, NdArray? covariance)
    {
        var p = StatsMatrixMath.Square(precision, nameof(precision), out _d);
        _cholP = StatsMultivariate.CholeskyLower(p, _d, nameof(precision));
        if (covariance != null)
        {
            _cov = StatsMatrixMath.Square(covariance, nameof(covariance), out var dc);
            if (dc != _d) throw new ArgumentException("precision and covariance must have the same shape.", nameof(covariance));
        }
        LogPdet = -StatsMultivariate.LogDetFromCholesky(_cholP, _d);
    }

    /// <inheritdoc />
    public override int Dim => _d;
    /// <inheritdoc />
    public override int Rank => _d;
    /// <inheritdoc />
    public override double LogPdet { get; }
    /// <inheritdoc />
    public override NdArray CovarianceMatrix => new(_cov != null ? (double[])_cov.Clone() : StatsMultivariate.CholeskySolve(_cholP, _d, StatsMatrixMath.Identity(_d), _d), _d, _d);

    /// <inheritdoc />
    public override double[] Whiten(double[] x)
    {
        Check(x);
        var r = new double[_d];
        for (var j = 0; j < _d; j++)
        {
            var s = 0.0;
            for (var i = j; i < _d; i++) s += x[i] * _cholP[i * _d + j];
            r[j] = s;
        }
        return r;
    }

    /// <inheritdoc />
    public override double[] Colorize(double[] x)
    {
        // y with y·R = x, i.e. Rᵀ yᵀ = xᵀ (Rᵀ upper triangular): back substitution.
        Check(x);
        var y = new double[_d];
        for (var j = _d - 1; j >= 0; j--)
        {
            var s = x[j];
            for (var i = j + 1; i < _d; i++) s -= y[i] * _cholP[i * _d + j];
            y[j] = s / _cholP[j * _d + j];
        }
        return y;
    }
}

/// <summary>Diagonal covariance; zero entries make it singular, with the support the coordinate subspace of the positive ones.</summary>
public sealed class CovViaDiagonal : Covariance
{
    private readonly double[] _d;

    internal CovViaDiagonal(double[] diagonal)
    {
        ArgumentNullException.ThrowIfNull(diagonal);
        if (diagonal.Length == 0 || diagonal.Any(v => !(v >= 0) || double.IsInfinity(v))) throw new ArgumentException("All elements of the diagonal must be non-negative and finite.", nameof(diagonal));
        _d = (double[])diagonal.Clone();
        Rank = _d.Count(v => v > 0);
        LogPdet = _d.Where(v => v > 0).Sum(Math.Log);
    }

    /// <inheritdoc />
    public override int Dim => _d.Length;
    /// <inheritdoc />
    public override int Rank { get; }
    /// <inheritdoc />
    public override double LogPdet { get; }
    /// <inheritdoc />
    public override NdArray CovarianceMatrix
    {
        get
        {
            var n = _d.Length;
            var m = new double[n * n];
            for (var i = 0; i < n; i++) m[i * n + i] = _d[i];
            return new NdArray(m, n, n);
        }
    }
    /// <inheritdoc />
    public override double[] Whiten(double[] x) { Check(x); return x.Select((v, i) => _d[i] > 0 ? v / Math.Sqrt(_d[i]) : 0.0).ToArray(); }
    /// <inheritdoc />
    public override double[] Colorize(double[] x) { Check(x); return x.Select((v, i) => v * Math.Sqrt(_d[i])).ToArray(); }
    /// <inheritdoc />
    public override bool InSupport(ReadOnlySpan<double> x)
    {
        var scale = 0.0;
        foreach (var v in x) scale = Math.Max(scale, Math.Abs(v));
        for (var i = 0; i < _d.Length; i++) if (_d[i] == 0 && Math.Abs(x[i]) > 1e-8 * Math.Max(1.0, scale)) return false;
        return true;
    }
}

/// <summary>Covariance from its lower Cholesky factor L: whitening solves L y = x, colouring is L x.</summary>
public sealed class CovViaCholesky : Covariance
{
    private readonly double[] _l;
    private readonly int _n;

    internal CovViaCholesky(NdArray cholesky)
    {
        var l = StatsMatrixMath.Square(cholesky, nameof(cholesky), out _n);
        for (var i = 0; i < _n; i++)
        {
            if (!(l[i * _n + i] != 0) || !double.IsFinite(l[i * _n + i])) throw new ArgumentException("The Cholesky factor must have a non-zero finite diagonal.", nameof(cholesky));
            for (var j = i + 1; j < _n; j++) l[i * _n + j] = 0; // scipy reads only the lower triangle
        }
        _l = l;
        var s = 0.0;
        for (var i = 0; i < _n; i++) s += Math.Log(Math.Abs(l[i * _n + i]));
        LogPdet = 2 * s;
    }

    /// <inheritdoc />
    public override int Dim => _n;
    /// <inheritdoc />
    public override int Rank => _n;
    /// <inheritdoc />
    public override double LogPdet { get; }
    /// <inheritdoc />
    public override NdArray CovarianceMatrix => new(StatsMatrixMath.Mul(_l, StatsMatrixMath.Transpose(_l, _n, _n), _n, _n, _n), _n, _n);
    /// <inheritdoc />
    public override double[] Whiten(double[] x)
    {
        Check(x);
        var y = new double[_n];
        for (var i = 0; i < _n; i++)
        {
            var s = x[i];
            for (var k = 0; k < i; k++) s -= _l[i * _n + k] * y[k];
            y[i] = s / _l[i * _n + i];
        }
        return y;
    }
    /// <inheritdoc />
    public override double[] Colorize(double[] x)
    {
        Check(x);
        var y = new double[_n];
        for (var i = 0; i < _n; i++)
        {
            var s = 0.0;
            for (var k = 0; k <= i; k++) s += _l[i * _n + k] * x[k];
            y[i] = s;
        }
        return y;
    }
}

/// <summary>Covariance from an eigendecomposition V diag(w) Vᵀ; zero eigenvalues give a singular covariance supported on the span of the others.</summary>
public sealed class CovViaEigendecomposition : Covariance
{
    private readonly double[] _w, _v;
    private readonly int _n;

    internal CovViaEigendecomposition(double[] eigenvalues, NdArray eigenvectors)
    {
        ArgumentNullException.ThrowIfNull(eigenvalues);
        _v = StatsMatrixMath.Square(eigenvectors, nameof(eigenvectors), out _n);
        if (eigenvalues.Length != _n) throw new ArgumentException("The shapes of eigenvalues and eigenvectors must be compatible.", nameof(eigenvalues));
        if (eigenvalues.Any(v => !(v >= 0) || double.IsInfinity(v))) throw new ArgumentException("All eigenvalues must be non-negative and finite.", nameof(eigenvalues));
        _w = (double[])eigenvalues.Clone();
        Rank = _w.Count(v => v > 0);
        LogPdet = _w.Where(v => v > 0).Sum(Math.Log);
    }

    /// <inheritdoc />
    public override int Dim => _n;
    /// <inheritdoc />
    public override int Rank { get; }
    /// <inheritdoc />
    public override double LogPdet { get; }
    /// <inheritdoc />
    public override NdArray CovarianceMatrix
    {
        get
        {
            var m = new double[_n * _n];
            for (var i = 0; i < _n; i++)
                for (var j = 0; j < _n; j++)
                {
                    var s = 0.0;
                    for (var k = 0; k < _n; k++) s += _v[i * _n + k] * _w[k] * _v[j * _n + k];
                    m[i * _n + j] = s;
                }
            return new NdArray(m, _n, _n);
        }
    }
    /// <inheritdoc />
    public override double[] Whiten(double[] x)
    {
        Check(x);
        var y = new double[_n];
        for (var k = 0; k < _n; k++)
        {
            if (!(_w[k] > 0)) continue;
            var s = 0.0;
            for (var i = 0; i < _n; i++) s += x[i] * _v[i * _n + k];
            y[k] = s / Math.Sqrt(_w[k]);
        }
        return y;
    }
    /// <inheritdoc />
    public override double[] Colorize(double[] x)
    {
        Check(x);
        var y = new double[_n];
        for (var i = 0; i < _n; i++)
        {
            var s = 0.0;
            for (var k = 0; k < _n; k++) s += _v[i * _n + k] * Math.Sqrt(_w[k]) * x[k];
            y[i] = s;
        }
        return y;
    }
    /// <inheritdoc />
    public override bool InSupport(ReadOnlySpan<double> x)
    {
        var scale = 0.0;
        foreach (var v in x) scale = Math.Max(scale, Math.Abs(v));
        for (var k = 0; k < _n; k++)
        {
            if (_w[k] > 0) continue;
            var s = 0.0;
            for (var i = 0; i < _n; i++) s += x[i] * _v[i * _n + k];
            if (Math.Abs(s) > 1e-8 * Math.Max(1.0, scale)) return false;
        }
        return true;
    }
}
