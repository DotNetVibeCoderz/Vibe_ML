using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Linalg
{
    // ================================================================ special matrices

    /// <summary>Block-diagonal matrix from the given blocks, scipy's <c>block_diag</c>.</summary>
    public static NdArray BlockDiag(params NdArray[] blocks)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        int rows = 0, cols = 0;
        foreach (var b in blocks)
        {
            var (r, c) = BlockShape(b);
            rows += r; cols += c;
        }
        var outp = new double[rows * cols];
        int r0 = 0, c0 = 0;
        foreach (var b in blocks)
        {
            var (r, c) = BlockShape(b);
            var s = b.AsContiguous().AsSpan();
            for (var i = 0; i < r; i++)
                s.Slice(i * c, c).CopyTo(outp.AsSpan((r0 + i) * cols + c0, c));
            r0 += r; c0 += c;
        }
        return new NdArray(outp, rows, cols);
    }

    private static (int, int) BlockShape(NdArray b) => b.Rank switch
    {
        0 => (1, 1),
        1 => (1, b.Size),
        2 => (b.Shape[0], b.Shape[1]),
        _ => throw new ArgumentException("BlockDiag: blocks must be at most 2-D."),
    };

    /// <summary>Circulant matrix with first column c: <c>C[i, j] = c[(i − j) mod n]</c>.</summary>
    public static NdArray Circulant(double[] c)
    {
        ArgumentNullException.ThrowIfNull(c);
        var n = c.Length;
        var r = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++) r[i * n + j] = c[((i - j) % n + n) % n];
        return new NdArray(r, n, n);
    }

    /// <summary>
    /// Frobenius companion matrix of the polynomial with coefficients a (highest degree first),
    /// scipy's <c>companion</c>: first row <c>−a[1:]/a[0]</c>, ones on the subdiagonal.
    /// </summary>
    public static NdArray Companion(double[] a)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Length < 2) throw new ArgumentException("Companion: need at least two coefficients.");
        if (a[0] == 0.0) throw new ArgumentException("Companion: the leading coefficient must not be zero.");
        var n = a.Length - 1;
        var r = new double[n * n];
        for (var j = 0; j < n; j++) r[j] = -a[j + 1] / a[0];
        for (var i = 1; i < n; i++) r[i * n + i - 1] = 1.0;
        return new NdArray(r, n, n);
    }

    /// <summary>
    /// Matrix A such that <c>A @ v</c> is the convolution of a with v (length n), scipy's
    /// <c>convolution_matrix</c>; mode "full" (default), "valid" or "same" as <c>np.convolve</c>.
    /// </summary>
    public static NdArray ConvolutionMatrix(double[] a, int n, string mode = "full")
    {
        ArgumentNullException.ThrowIfNull(a);
        if (n <= 0) throw new ArgumentException("ConvolutionMatrix: n must be positive.");
        if (a.Length == 0) throw new ArgumentException("ConvolutionMatrix: a must not be empty.");
        if (mode is not ("full" or "valid" or "same")) throw new ArgumentException("ConvolutionMatrix: mode must be 'full', 'valid' or 'same'.");
        var len = a.Length;
        var az = new double[len + n - 1];
        var raz = new double[len + n - 1];
        for (var i = 0; i < len; i++) { az[i] = a[i]; raz[i] = a[len - 1 - i]; }
        double[] col0, row0;
        if (mode == "full")
        {
            col0 = az;
            row0 = raz[^n..];
        }
        else
        {
            int tb, te;
            if (mode == "same")
            {
                var trim = Math.Min(n, len) - 1;
                tb = trim / 2; te = trim - tb;
            }
            else
            {
                tb = Math.Min(n, len) - 1; te = tb;
            }
            col0 = az[tb..(az.Length - te)];
            row0 = raz[(raz.Length - n - tb)..(raz.Length - tb)];
        }
        return Toeplitz(col0, row0);
    }

    /// <summary>
    /// The n×n DFT matrix <c>m[j, k] = exp(−2πi jk/n)</c>, scipy's <c>dft</c>; scale "sqrtn" divides
    /// by √n (unitary), "n" by n.
    /// </summary>
    public static ComplexNdArray Dft(int n, string? scale = null)
    {
        if (n < 0) throw new ArgumentException("Dft: n must be non-negative.");
        if (scale is not (null or "sqrtn" or "n")) throw new ArgumentException("Dft: scale must be null, 'sqrtn' or 'n'.");
        var f = scale switch { "sqrtn" => 1.0 / Math.Sqrt(n), "n" => 1.0 / n, _ => 1.0 };
        var m = new Complex[n * n];
        for (var j = 0; j < n; j++)
            for (var k = 0; k < n; k++)
            {
                // Reduce jk mod n in integers so large n keeps the angle exact.
                var e = (long)j * k % n;
                m[j * n + k] = Complex.FromPolarCoordinates(f, -2.0 * Math.PI * e / n);
            }
        return new ComplexNdArray(m, n, n);
    }

    /// <summary>Symmetric Fiedler matrix <c>F[i, j] = |a[i] − a[j]|</c>, scipy's <c>fiedler</c>.</summary>
    public static NdArray Fiedler(double[] a)
    {
        ArgumentNullException.ThrowIfNull(a);
        var n = a.Length;
        var r = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++) r[i * n + j] = Math.Abs(a[i] - a[j]);
        return new NdArray(r, n, n);
    }

    /// <summary>
    /// Fiedler's pentadiagonal companion matrix of a polynomial (highest degree first), scipy's
    /// <c>fiedler_companion</c>; same eigenvalues as <see cref="Companion"/>, but banded.
    /// </summary>
    public static NdArray FiedlerCompanion(double[] a)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Length <= 2)
        {
            if (a.Length == 2) return new NdArray([-(a[1] / a[0])], 1, 1);
            return NdArray.Zeros(0, 0);
        }
        if (a[0] == 0.0) throw new ArgumentException("FiedlerCompanion: leading coefficient is zero.");
        var b = new double[a.Length];
        for (var i = 0; i < a.Length; i++) b[i] = a[i] / a[0];
        var n = b.Length - 1;
        var c = new double[n * n];
        for (int i = 3, j = 1; i < n && j < n - 2; i += 2, j += 2) c[i * n + j] = 1.0;
        for (int i = 2, j = 1, k = 3; i < n && j < n - 1 && k < b.Length; i += 2, j += 2, k += 2) c[i * n + j] = -b[k];
        for (int i = 0, j = 2; i < n - 2 && j < n; i += 2, j += 2) c[i * n + j] = 1.0;
        for (int i = 0, j = 1, k = 2; i < n - 1 && j < n && k < b.Length; i += 2, j += 2, k += 2) c[i * n + j] = -b[k];
        c[0] = -b[1];
        c[n] = 1.0;
        return new NdArray(c, n, n);
    }

    /// <summary>Sylvester-construction Hadamard matrix of order n (a power of two), scipy's <c>hadamard</c>.</summary>
    public static NdArray Hadamard(int n)
    {
        if (n < 1 || (n & (n - 1)) != 0) throw new ArgumentException("Hadamard: n must be a positive power of 2.");
        var r = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++) r[i * n + j] = (BitOperations.PopCount((uint)(i & j)) & 1) == 0 ? 1.0 : -1.0;
        return new NdArray(r, n, n);
    }

    /// <summary>
    /// Hankel matrix with first column c and last row r (default zeros), scipy's <c>hankel</c>;
    /// <c>r[0]</c> is ignored in favour of <c>c[^1]</c>.
    /// </summary>
    public static NdArray Hankel(double[] c, double[]? r = null)
    {
        ArgumentNullException.ThrowIfNull(c);
        r ??= new double[c.Length];
        var m = c.Length;
        var n = r.Length;
        var h = new double[m * n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
            {
                var k = i + j;
                h[i * n + j] = k < m ? c[k] : r[k - m + 1];
            }
        return new NdArray(h, m, n);
    }

    /// <summary>
    /// Helmert matrix of order n, scipy's <c>helmert</c>: orthogonal, first row constant
    /// <c>1/√n</c>; without <paramref name="full"/> that row is dropped ((n−1)×n).
    /// </summary>
    public static NdArray Helmert(int n, bool full = false)
    {
        if (n < 1) throw new ArgumentException("Helmert: n must be positive.");
        var h = new double[n * n];
        for (var j = 0; j < n; j++) h[j] = 1.0 / Math.Sqrt(n);
        for (var i = 1; i < n; i++)
        {
            var d = Math.Sqrt((double)i * (i + 1));
            for (var j = 0; j < i; j++) h[i * n + j] = 1.0 / d;
            h[i * n + i] = -i / d;
        }
        if (full) return new NdArray(h, n, n);
        var r = new double[(n - 1) * n];
        Array.Copy(h, n, r, 0, r.Length);
        return new NdArray(r, n - 1, n);
    }

    /// <summary>Hilbert matrix <c>H[i, j] = 1/(i + j + 1)</c>, scipy's <c>hilbert</c>.</summary>
    public static NdArray Hilbert(int n)
    {
        var h = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++) h[i * n + j] = 1.0 / (i + j + 1);
        return new NdArray(h, n, n);
    }

    /// <summary>
    /// Exact inverse of the Hilbert matrix, scipy's <c>invhilbert(n, exact=True)</c>, computed in
    /// arbitrary-precision integers and rounded once to double. The entries are integers; beyond
    /// n ≈ 14 they exceed 2⁵³ and are no longer exactly representable (but are correctly rounded).
    /// </summary>
    public static NdArray InvHilbert(int n) => BigToNd(InvHilbertExact(n), n);

    /// <summary>The exact integer entries of the inverse Hilbert matrix.</summary>
    public static BigInteger[,] InvHilbertExact(int n)
    {
        if (n < 0) throw new ArgumentException("InvHilbert: n must be non-negative.");
        var r = new BigInteger[n, n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j <= i; j++)
            {
                var s = (i + j) % 2 == 0 ? BigInteger.One : BigInteger.MinusOne;
                var c = Binom(i + j, i);
                var v = s * (i + j + 1) * Binom(n + i, n - j - 1) * Binom(n + j, n - i - 1) * c * c;
                r[i, j] = v;
                r[j, i] = v;
            }
        return r;
    }

    /// <summary>
    /// Pascal matrix, scipy's <c>pascal</c>: kind "symmetric" <c>C(i+j, i)</c>, "lower"
    /// <c>C(i, j)</c>, "upper" its transpose. Rounded from exact integers.
    /// </summary>
    public static NdArray Pascal(int n, string kind = "symmetric")
    {
        if (kind is not ("symmetric" or "lower" or "upper")) throw new ArgumentException("Pascal: kind must be 'symmetric', 'lower' or 'upper'.");
        var r = new BigInteger[n, n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
                r[i, j] = kind switch
                {
                    "symmetric" => Binom(i + j, i),
                    "lower" => j <= i ? Binom(i, j) : BigInteger.Zero,
                    _ => i <= j ? Binom(j, i) : BigInteger.Zero,
                };
        return BigToNd(r, n);
    }

    /// <summary>Exact inverse of the Pascal matrix, scipy's <c>invpascal</c>.</summary>
    public static NdArray Invpascal(int n, string kind = "symmetric")
    {
        if (kind is not ("symmetric" or "lower" or "upper")) throw new ArgumentException("Invpascal: kind must be 'symmetric', 'lower' or 'upper'.");
        var r = new BigInteger[n, n];
        if (kind == "symmetric")
        {
            for (var i = 0; i < n; i++)
                for (var j = 0; j <= i; j++)
                {
                    var v = BigInteger.Zero;
                    for (var k = 0; k < n - i; k++) v += Binom(i + k, k) * Binom(i + k, i + k - j);
                    if ((i - j) % 2 != 0) v = -v;
                    r[i, j] = v;
                    r[j, i] = v;
                }
        }
        else
        {
            for (var i = 0; i < n; i++)
                for (var j = 0; j < n; j++)
                {
                    var p = kind == "lower" ? (j <= i ? Binom(i, j) : 0) : (i <= j ? Binom(j, i) : 0);
                    r[i, j] = Math.Abs(i - j) % 2 == 0 ? p : -p;
                }
        }
        return BigToNd(r, n);
    }

    private static BigInteger Binom(int n, int k)
    {
        if (k < 0 || k > n) return BigInteger.Zero;
        k = Math.Min(k, n - k);
        var r = BigInteger.One;
        for (var i = 1; i <= k; i++) r = r * (n - k + i) / i;
        return r;
    }

    private static NdArray BigToNd(BigInteger[,] r, int n)
    {
        var d = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++) d[i * n + j] = (double)r[i, j];
        return new NdArray(d, n, n);
    }

    /// <summary>
    /// Leslie population matrix from fecundities f (length n) and survival rates s (length n−1),
    /// scipy's <c>leslie</c>.
    /// </summary>
    public static NdArray Leslie(double[] f, double[] s)
    {
        ArgumentNullException.ThrowIfNull(f);
        ArgumentNullException.ThrowIfNull(s);
        if (f.Length != s.Length + 1) throw new ArgumentException("Leslie: len(s) must be len(f) - 1.");
        if (s.Length == 0) throw new ArgumentException("Leslie: s must not be empty.");
        var n = f.Length;
        var a = new double[n * n];
        Array.Copy(f, a, n);
        for (var i = 1; i < n; i++) a[i * n + i - 1] = s[i - 1];
        return new NdArray(a, n, n);
    }

    /// <summary>
    /// Toeplitz matrix with first column c and first row r (default r = c, symmetric), scipy's
    /// <c>toeplitz</c>; <c>r[0]</c> is ignored.
    /// </summary>
    public static NdArray Toeplitz(double[] c, double[]? r = null)
    {
        ArgumentNullException.ThrowIfNull(c);
        r ??= c;
        var m = c.Length;
        var n = r.Length;
        var t = new double[m * n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++) t[i * n + j] = i >= j ? c[i - j] : r[j - i];
        return new NdArray(t, m, n);
    }

    /// <summary>Kronecker product (delegates to GraviNum <c>LinAlg.Kron</c>).</summary>
    public static NdArray Kron(NdArray a, NdArray b) => LinAlg.Kron(a, b);

    /// <summary>Lower triangle of m with entries above the k-th diagonal zeroed (numpy <c>tril</c>).</summary>
    public static NdArray Tril(NdArray m, int k = 0) => Tri(m, k, lower: true);

    /// <summary>Upper triangle of m with entries below the k-th diagonal zeroed (numpy <c>triu</c>).</summary>
    public static NdArray Triu(NdArray m, int k = 0) => Tri(m, k, lower: false);

    private static NdArray Tri(NdArray m, int k, bool lower)
    {
        LinalgKernels.RequireMatrix(m, lower ? nameof(Tril) : nameof(Triu));
        var r = m.ToArray();
        var rows = m.Shape[0];
        var cols = m.Shape[1];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                if (lower ? j - i > k : j - i < k) r[i * cols + j] = 0.0;
        return new NdArray(r, rows, cols);
    }

    /// <summary>
    /// Vandermonde matrix (numpy <c>vander</c>): columns are powers of x, decreasing by default
    /// (<c>V[i, j] = x[i]^(N−1−j)</c>), increasing with <paramref name="increasing"/>.
    /// </summary>
    public static NdArray Vandermonde(double[] x, int? columns = null, bool increasing = false)
    {
        ArgumentNullException.ThrowIfNull(x);
        var m = x.Length;
        var n = columns ?? m;
        var v = new double[m * n];
        for (var i = 0; i < m; i++)
        {
            var p = 1.0;
            for (var j = 0; j < n; j++)
            {
                var col = increasing ? j : n - 1 - j;
                v[i * n + col] = p;
                p *= x[i];
            }
        }
        return new NdArray(v, m, n);
    }

    // ================================================================ norms and structure

    /// <summary>Frobenius norm of a matrix, 2-norm of a vector (overflow-safe scaled sum of squares).</summary>
    public static double Norm(NdArray a)
    {
        ArgumentNullException.ThrowIfNull(a);
        return LinalgKernels.FrobeniusNorm(a.AsContiguous().AsSpan());
    }

    /// <summary>
    /// Vector or matrix norm, scipy's <c>norm(a, ord)</c>. Vectors: any p, ±∞ (max/min |x|), 0
    /// (count of non-zeros). Matrices: 1/−1 (max/min column sum), 2/−2 (max/min singular
    /// value), ±∞ (max/min row sum).
    /// </summary>
    public static double Norm(NdArray a, double ord)
    {
        ArgumentNullException.ThrowIfNull(a);
        var s = a.AsContiguous().AsSpan();
        if (a.Rank == 1)
        {
            if (double.IsPositiveInfinity(ord)) { var m = 0.0; foreach (var v in s) m = Math.Max(m, Math.Abs(v)); return m; }
            if (double.IsNegativeInfinity(ord)) { var m = double.PositiveInfinity; foreach (var v in s) m = Math.Min(m, Math.Abs(v)); return m; }
            if (ord == 0) { var c = 0; foreach (var v in s) if (v != 0) c++; return c; }
            if (ord == 2) return LinalgKernels.FrobeniusNorm(s);
            if (ord == 1) { var t = 0.0; foreach (var v in s) t += Math.Abs(v); return t; }
            var acc = 0.0;
            foreach (var v in s) acc += Math.Pow(Math.Abs(v), ord);
            return Math.Pow(acc, 1.0 / ord);
        }
        LinalgKernels.RequireMatrix(a, nameof(Norm));
        var rows = a.Shape[0];
        var cols = a.Shape[1];
        if (ord == 1 || ord == -1)
        {
            var sums = new double[cols];
            for (var i = 0; i < rows; i++)
                for (var j = 0; j < cols; j++) sums[j] += Math.Abs(s[i * cols + j]);
            return ord == 1 ? sums.Max() : sums.Min();
        }
        if (double.IsInfinity(ord))
        {
            var sums = new double[rows];
            for (var i = 0; i < rows; i++)
                for (var j = 0; j < cols; j++) sums[i] += Math.Abs(s[i * cols + j]);
            return ord > 0 ? sums.Max() : sums.Min();
        }
        if (ord == 2 || ord == -2)
        {
            var sv = Svdvals(a);
            return ord == 2 ? sv[0] : sv[^1];
        }
        throw new ArgumentException($"Norm: invalid order {ord} for a matrix.");
    }

    /// <summary>Matrix norm by name: "fro" (Frobenius) or "nuc" (nuclear, sum of singular values).</summary>
    public static double Norm(NdArray a, string ord)
    {
        ArgumentNullException.ThrowIfNull(a);
        return ord switch
        {
            "fro" => LinalgKernels.FrobeniusNorm(a.AsContiguous().AsSpan()),
            "nuc" => Svdvals(a).Sum(),
            _ => throw new ArgumentException("Norm: string order must be 'fro' or 'nuc'."),
        };
    }

    /// <summary>
    /// Lower and upper bandwidth, scipy's <c>bandwidth</c>: the furthest non-zero sub- and
    /// superdiagonal.
    /// </summary>
    public static (int Lower, int Upper) Bandwidth(NdArray a)
    {
        LinalgKernels.RequireMatrix(a, nameof(Bandwidth));
        var rows = a.Shape[0];
        var cols = a.Shape[1];
        var s = a.AsContiguous().AsSpan();
        int lo = 0, up = 0;
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
            {
                if (s[i * cols + j] == 0.0) continue;
                if (i - j > lo) lo = i - j;
                if (j - i > up) up = j - i;
            }
        return (lo, up);
    }

    private static (int Lower, int Upper) Bandwidth(double[] a, int n) => Bandwidth(new NdArray(a, n, n));

    /// <summary>
    /// Whether a square matrix is symmetric, scipy's <c>issymmetric</c>: exact comparison by
    /// default, otherwise <c>|a_ij − a_ji| ≤ atol + rtol·|a_ji|</c>.
    /// </summary>
    public static bool IsSymmetric(NdArray a, double atol = 0.0, double rtol = 0.0)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1]) return false;
        var n = a.Shape[0];
        var s = a.AsContiguous().AsSpan();
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
            {
                var x = s[i * n + j];
                var y = s[j * n + i];
                if (atol == 0 && rtol == 0 ? x != y : Math.Abs(x - y) > atol + rtol * Math.Abs(y)) return false;
            }
        return true;
    }

    /// <summary>Hermitian test, scipy's <c>ishermitian</c>; for a real matrix the same as <see cref="IsSymmetric"/>.</summary>
    public static bool IsHermitian(NdArray a, double atol = 0.0, double rtol = 0.0) => IsSymmetric(a, atol, rtol);
}
