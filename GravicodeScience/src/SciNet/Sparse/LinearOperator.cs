using System.Buffers;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Sparse;

/// <summary>
/// Anything that can multiply a vector: the one capability every Krylov solver and eigensolver in
/// <see cref="SparseLinalg"/> needs. The analogue of <c>scipy.sparse.linalg.LinearOperator</c>.
/// </summary>
/// <remarks>
/// Both products write into a caller-supplied buffer so a solver can run thousands of iterations
/// without allocating. Operators here are real, so the adjoint is the transpose.
/// </remarks>
public interface ILinearOperator
{
    /// <summary>Number of rows (length of <c>A x</c>).</summary>
    int Rows { get; }

    /// <summary>Number of columns (length of <c>x</c>).</summary>
    int Columns { get; }

    /// <summary>Computes <c>y = A x</c>. <paramref name="y"/> is overwritten, not accumulated into.</summary>
    void MatVec(ReadOnlySpan<double> x, Span<double> y);

    /// <summary>Computes <c>y = Aᵀ x</c>. <paramref name="y"/> is overwritten, not accumulated into.</summary>
    void RMatVec(ReadOnlySpan<double> x, Span<double> y);
}

/// <summary>An allocation-free matrix–vector product: write <c>A x</c> (or <c>Aᵀ x</c>) into <c>y</c>.</summary>
public delegate void MatVecFunction(ReadOnlySpan<double> x, Span<double> y);

/// <summary>
/// A linear operator defined by delegates, plus the composition algebra of
/// <c>scipy.sparse.linalg.LinearOperator</c>: sums, products, scaling, transpose and powers.
/// </summary>
/// <remarks>
/// Composites never materialise a matrix; a product applies its factors one after the other
/// through a pooled temporary. That is what lets a shift-invert operator or a normal-equations
/// operator <c>AᵀA</c> cost two sparse products per application instead of a dense fill-in.
/// </remarks>
public class LinearOperator : ILinearOperator
{
    private readonly MatVecFunction _matVec;
    private readonly MatVecFunction? _rMatVec;

    /// <summary>Builds an operator from allocation-free product delegates.</summary>
    /// <param name="rows">Number of rows.</param>
    /// <param name="columns">Number of columns.</param>
    /// <param name="matVec">Writes <c>A x</c> into its second argument.</param>
    /// <param name="rMatVec">Writes <c>Aᵀ x</c>; optional, but solvers that need it (BiCG, QMR, LSQR) throw without it.</param>
    public LinearOperator(int rows, int columns, MatVecFunction matVec, MatVecFunction? rMatVec = null)
    {
        if (rows < 0 || columns < 0) throw new ArgumentException("Operator dimensions must be non-negative.");
        Rows = rows;
        Columns = columns;
        _matVec = matVec ?? throw new ArgumentNullException(nameof(matVec));
        _rMatVec = rMatVec;
    }

    /// <summary>Builds an operator from allocating delegates (convenient, slower in a tight loop).</summary>
    public LinearOperator(int rows, int columns, Func<double[], double[]> matVec, Func<double[], double[]>? rMatVec = null)
        : this(rows, columns, Wrap(matVec, rows)!, Wrap(rMatVec, columns))
    {
    }

    private static MatVecFunction? Wrap(Func<double[], double[]>? f, int outLength)
    {
        if (f is null) return null;
        return (x, y) =>
        {
            var r = f(x.ToArray());
            if (r.Length != outLength) throw new InvalidOperationException($"Operator returned {r.Length} entries, expected {outLength}.");
            r.CopyTo(y);
        };
    }

    /// <inheritdoc />
    public int Rows { get; }

    /// <inheritdoc />
    public int Columns { get; }

    /// <summary>True when a transpose product was supplied.</summary>
    public bool HasRMatVec => _rMatVec is not null;

    /// <inheritdoc />
    public void MatVec(ReadOnlySpan<double> x, Span<double> y)
    {
        CheckLengths(x.Length, y.Length, Columns, Rows);
        _matVec(x, y);
    }

    /// <inheritdoc />
    public void RMatVec(ReadOnlySpan<double> x, Span<double> y)
    {
        if (_rMatVec is null) throw new NotSupportedException("This LinearOperator was built without a transpose product (rMatVec).");
        CheckLengths(x.Length, y.Length, Rows, Columns);
        _rMatVec(x, y);
    }

    private static void CheckLengths(int xLen, int yLen, int expectX, int expectY)
    {
        if (xLen != expectX) throw new ArgumentException($"Input has {xLen} entries, expected {expectX}.");
        if (yLen != expectY) throw new ArgumentException($"Output has {yLen} entries, expected {expectY}.");
    }

    /// <summary>The <paramref name="n"/>×<paramref name="n"/> identity.</summary>
    public static LinearOperator Identity(int n)
        => new(n, n, (MatVecFunction)((x, y) => x.CopyTo(y)), (x, y) => x.CopyTo(y));

    /// <summary>Wraps any operator (no copy). Returns it unchanged when it already is a <see cref="LinearOperator"/>.</summary>
    public static LinearOperator AsLinearOperator(ILinearOperator a)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a is LinearOperator lo) return lo;
        return new LinearOperator(a.Rows, a.Columns, (MatVecFunction)a.MatVec, a.RMatVec);
    }

    /// <summary>Wraps a dense 2-D array as an operator (row-major products, no copy of a contiguous array).</summary>
    public static LinearOperator AsLinearOperator(NdArray dense)
    {
        if (dense.Rank != 2) throw new ArgumentException("A dense operator must be a rank 2 array.");
        var a = dense.AsContiguous().ToArray();
        int m = dense.Shape[0], n = dense.Shape[1];
        return new LinearOperator(m, n,
            (MatVecFunction)((x, y) =>
            {
                for (var i = 0; i < m; i++) y[i] = LinAlg.InnerSpan(a.AsSpan(i * n, n), x);
            }),
            (x, y) =>
            {
                y.Clear();
                for (var i = 0; i < m; i++) LinAlg.AxpySpan(x[i], a.AsSpan(i * n, n), y);
            });
    }

    /// <summary><c>A + B</c> without forming either.</summary>
    public static LinearOperator Sum(ILinearOperator a, ILinearOperator b)
    {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            throw new ArgumentException($"Cannot add a {a.Rows}x{a.Columns} operator to a {b.Rows}x{b.Columns} one.");
        return new LinearOperator(a.Rows, a.Columns,
            (MatVecFunction)((x, y) => AddInto(a.MatVec, b.MatVec, x, y)),
            (x, y) => AddInto(a.RMatVec, b.RMatVec, x, y));
    }

    private static void AddInto(MatVecFunction f, MatVecFunction g, ReadOnlySpan<double> x, Span<double> y)
    {
        f(x, y);
        var tmp = ArrayPool<double>.Shared.Rent(y.Length);
        try
        {
            var t = tmp.AsSpan(0, y.Length);
            g(x, t);
            for (var i = 0; i < y.Length; i++) y[i] += t[i];
        }
        finally { ArrayPool<double>.Shared.Return(tmp); }
    }

    /// <summary><c>A B</c>: applies <paramref name="b"/> then <paramref name="a"/>.</summary>
    public static LinearOperator Product(ILinearOperator a, ILinearOperator b)
    {
        if (a.Columns != b.Rows)
            throw new ArgumentException($"Cannot multiply a {a.Rows}x{a.Columns} operator by a {b.Rows}x{b.Columns} one.");
        var inner = a.Columns;
        return new LinearOperator(a.Rows, b.Columns,
            (MatVecFunction)((x, y) => Chain(b.MatVec, a.MatVec, inner, x, y)),
            (x, y) => Chain(a.RMatVec, b.RMatVec, inner, x, y));
    }

    private static void Chain(MatVecFunction first, MatVecFunction second, int inner, ReadOnlySpan<double> x, Span<double> y)
    {
        var tmp = ArrayPool<double>.Shared.Rent(Math.Max(inner, 1));
        try
        {
            var t = tmp.AsSpan(0, inner);
            first(x, t);
            second(t, y);
        }
        finally { ArrayPool<double>.Shared.Return(tmp); }
    }

    /// <summary><c>alpha A</c>.</summary>
    public static LinearOperator Scaled(ILinearOperator a, double alpha)
        => new(a.Rows, a.Columns,
            (MatVecFunction)((x, y) => { a.MatVec(x, y); Scale(y, alpha); }),
            (x, y) => { a.RMatVec(x, y); Scale(y, alpha); });

    private static void Scale(Span<double> y, double alpha)
    {
        for (var i = 0; i < y.Length; i++) y[i] *= alpha;
    }

    /// <summary><c>Aᵀ</c>: swaps the two products.</summary>
    public static LinearOperator Transpose(ILinearOperator a)
        => new(a.Columns, a.Rows, (MatVecFunction)a.RMatVec, a.MatVec);

    /// <summary><c>Aᴴ</c>; identical to <see cref="Transpose(ILinearOperator)"/> for these real operators.</summary>
    public static LinearOperator Adjoint(ILinearOperator a) => Transpose(a);

    /// <summary><c>Aᵖ</c> for a square operator, applied as <paramref name="p"/> successive products.</summary>
    public static LinearOperator Power(ILinearOperator a, int p)
    {
        if (a.Rows != a.Columns) throw new ArgumentException("Only a square operator has powers.");
        if (p < 0) throw new ArgumentException("Negative powers would need an inverse.");
        if (p == 0) return Identity(a.Rows);
        var n = a.Rows;
        return new LinearOperator(n, n,
            (MatVecFunction)((x, y) => Repeat(a.MatVec, p, n, x, y)),
            (x, y) => Repeat(a.RMatVec, p, n, x, y));
    }

    private static void Repeat(MatVecFunction f, int p, int n, ReadOnlySpan<double> x, Span<double> y)
    {
        var tmp = ArrayPool<double>.Shared.Rent(Math.Max(n, 1));
        try
        {
            var t = tmp.AsSpan(0, n);
            f(x, y);
            for (var i = 1; i < p; i++)
            {
                y.CopyTo(t);
                f(t, y);
            }
        }
        finally { ArrayPool<double>.Shared.Return(tmp); }
    }

    /// <summary>Sum of two operators.</summary>
    public static LinearOperator operator +(LinearOperator a, LinearOperator b) => Sum(a, b);

    /// <summary>Difference of two operators.</summary>
    public static LinearOperator operator -(LinearOperator a, LinearOperator b) => Sum(a, Scaled(b, -1.0));

    /// <summary>Negation.</summary>
    public static LinearOperator operator -(LinearOperator a) => Scaled(a, -1.0);

    /// <summary>Operator product <c>A B</c>.</summary>
    public static LinearOperator operator *(LinearOperator a, LinearOperator b) => Product(a, b);

    /// <summary>Scaling.</summary>
    public static LinearOperator operator *(double alpha, LinearOperator a) => Scaled(a, alpha);

    /// <summary>Scaling.</summary>
    public static LinearOperator operator *(LinearOperator a, double alpha) => Scaled(a, alpha);

    /// <summary>Applies the operator to a vector, allocating the result.</summary>
    public static double[] operator *(LinearOperator a, double[] x) => a.Dot(x);

    /// <summary>The transpose of this operator.</summary>
    public LinearOperator T => Transpose(this);

    /// <summary>The adjoint of this operator.</summary>
    public LinearOperator H => Adjoint(this);
}

/// <summary>Allocating conveniences over <see cref="ILinearOperator"/>.</summary>
public static class LinearOperatorExtensions
{
    /// <summary><c>A x</c> as a new array.</summary>
    public static double[] Dot(this ILinearOperator a, ReadOnlySpan<double> x)
    {
        var y = new double[a.Rows];
        a.MatVec(x, y);
        return y;
    }

    /// <summary><c>A x</c> as a new array.</summary>
    public static double[] Dot(this ILinearOperator a, double[] x) => a.Dot((ReadOnlySpan<double>)x);

    /// <summary><c>Aᵀ x</c> as a new array.</summary>
    public static double[] RDot(this ILinearOperator a, ReadOnlySpan<double> x)
    {
        var y = new double[a.Columns];
        a.RMatVec(x, y);
        return y;
    }

    /// <summary><c>A X</c> for a dense rank 2 <paramref name="x"/>, one column at a time.</summary>
    public static NdArray MatMat(this ILinearOperator a, NdArray x)
    {
        if (x.Rank == 1) return new NdArray(a.Dot(x.ToArray()), a.Rows);
        if (x.Rank != 2 || x.Shape[0] != a.Columns)
            throw new ArgumentException($"Cannot apply a {a.Rows}x{a.Columns} operator to an array of shape ({string.Join(", ", x.Shape.ToArray())}).");
        int n = x.Shape[1], m = a.Rows, k = a.Columns;
        var src = x.AsContiguous().ToArray();
        var outBuf = new double[m * n];
        var col = new double[k];
        var res = new double[m];
        for (var j = 0; j < n; j++)
        {
            for (var i = 0; i < k; i++) col[i] = src[i * n + j];
            a.MatVec(col, res);
            for (var i = 0; i < m; i++) outBuf[i * n + j] = res[i];
        }
        return new NdArray(outBuf, m, n);
    }

    /// <summary>Wraps the operator as a <see cref="LinearOperator"/> so the composition operators apply.</summary>
    public static LinearOperator AsLinearOperator(this ILinearOperator a) => LinearOperator.AsLinearOperator(a);

    /// <summary>Materialises the operator as a dense array by applying it to each unit vector.</summary>
    /// <remarks>O(n) products; meant for tests and small projected problems, not for large operators.</remarks>
    public static NdArray ToDenseArray(this ILinearOperator a)
    {
        int m = a.Rows, n = a.Columns;
        var buf = new double[m * n];
        var e = new double[n];
        var col = new double[m];
        for (var j = 0; j < n; j++)
        {
            e[j] = 1.0;
            a.MatVec(e, col);
            e[j] = 0.0;
            for (var i = 0; i < m; i++) buf[i * n + j] = col[i];
        }
        return new NdArray(buf, m, n);
    }
}
