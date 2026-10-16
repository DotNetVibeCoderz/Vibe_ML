using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Small dense linear-algebra kernels on row-major <c>double[]</c> buffers, shared by the
/// optimisers. The matrices inside an optimiser are small (n up to a few hundred) and touched every
/// iteration, so these avoid <see cref="NdArray"/> headers and the indexer; anything large or
/// one-off (SVD for covariance, eigen decomposition) goes through GraviNum.
/// </summary>
internal static class OptimizeDense
{
    public static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        double s = 0;
        for (var i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }

    /// <summary>Euclidean norm with scaling so huge or tiny entries do not overflow/underflow.</summary>
    public static double Norm2(ReadOnlySpan<double> a)
    {
        double scale = 0, ssq = 1;
        for (var i = 0; i < a.Length; i++)
        {
            var v = a[i];
            if (v == 0) continue;
            var av = Math.Abs(v);
            if (double.IsNaN(av)) return double.NaN;
            if (double.IsInfinity(av)) return double.PositiveInfinity;
            if (scale < av) { var r = scale / av; ssq = 1 + ssq * r * r; scale = av; }
            else { var r = av / scale; ssq += r * r; }
        }
        return scale * Math.Sqrt(ssq);
    }

    public static double NormInf(ReadOnlySpan<double> a)
    {
        double m = 0;
        for (var i = 0; i < a.Length; i++) { var v = Math.Abs(a[i]); if (v > m || double.IsNaN(v)) m = v; }
        return m;
    }

    /// <summary>y += alpha * x.</summary>
    public static void Axpy(double alpha, ReadOnlySpan<double> x, Span<double> y)
    {
        for (var i = 0; i < x.Length; i++) y[i] += alpha * x[i];
    }

    /// <summary>y = A x for a row-major (m, n) matrix.</summary>
    public static void MatVec(ReadOnlySpan<double> a, int m, int n, ReadOnlySpan<double> x, Span<double> y)
    {
        for (var i = 0; i < m; i++)
        {
            var row = a.Slice(i * n, n);
            double s = 0;
            for (var j = 0; j < n; j++) s += row[j] * x[j];
            y[i] = s;
        }
    }

    /// <summary>y = Aᵀ x for a row-major (m, n) matrix.</summary>
    public static void MatTVec(ReadOnlySpan<double> a, int m, int n, ReadOnlySpan<double> x, Span<double> y)
    {
        y[..n].Clear();
        for (var i = 0; i < m; i++)
        {
            var xi = x[i];
            if (xi == 0) continue;
            var row = a.Slice(i * n, n);
            for (var j = 0; j < n; j++) y[j] += row[j] * xi;
        }
    }

    /// <summary>C = A B for row-major A (m,k), B (k,n).</summary>
    public static double[] MatMul(double[] a, int m, int k, double[] b, int n)
    {
        var c = new double[m * n];
        for (var i = 0; i < m; i++)
        {
            var ci = c.AsSpan(i * n, n);
            for (var p = 0; p < k; p++)
            {
                var aip = a[i * k + p];
                if (aip == 0) continue;
                var bp = b.AsSpan(p * n, n);
                for (var j = 0; j < n; j++) ci[j] += aip * bp[j];
            }
        }
        return c;
    }

    /// <summary>AᵀA for row-major A (m, n); result (n, n) symmetric.</summary>
    public static double[] Gram(ReadOnlySpan<double> a, int m, int n)
    {
        var g = new double[n * n];
        for (var r = 0; r < m; r++)
        {
            var row = a.Slice(r * n, n);
            for (var i = 0; i < n; i++)
            {
                var ri = row[i];
                if (ri == 0) continue;
                var gi = g.AsSpan(i * n, n);
                for (var j = i; j < n; j++) gi[j] += ri * row[j];
            }
        }
        for (var i = 0; i < n; i++) for (var j = 0; j < i; j++) g[i * n + j] = g[j * n + i];
        return g;
    }

    public static double[] Transpose(ReadOnlySpan<double> a, int m, int n)
    {
        var t = new double[m * n];
        for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) t[j * m + i] = a[i * n + j];
        return t;
    }

    public static double[] Identity(int n)
    {
        var e = new double[n * n];
        for (var i = 0; i < n; i++) e[i * n + i] = 1;
        return e;
    }

    /// <summary>
    /// In-place Cholesky of a symmetric (n, n) matrix: the lower triangle becomes L, the strict upper
    /// triangle is zeroed. Returns false (leaving <paramref name="a"/> partly overwritten) when the
    /// matrix is not numerically positive definite.
    /// </summary>
    public static bool Cholesky(Span<double> a, int n)
    {
        for (var j = 0; j < n; j++)
        {
            var d = a[j * n + j];
            for (var k = 0; k < j; k++) d -= a[j * n + k] * a[j * n + k];
            if (!(d > 0) || double.IsInfinity(d)) return false;
            d = Math.Sqrt(d);
            a[j * n + j] = d;
            for (var i = j + 1; i < n; i++)
            {
                var s = a[i * n + j];
                for (var k = 0; k < j; k++) s -= a[i * n + k] * a[j * n + k];
                a[i * n + j] = s / d;
            }
            for (var i = 0; i < j; i++) a[i * n + j] = 0;
        }
        return true;
    }

    /// <summary>Solves L Lᵀ x = b in place given the factor from <see cref="Cholesky"/>.</summary>
    public static void CholeskySolve(ReadOnlySpan<double> l, int n, Span<double> b)
    {
        for (var i = 0; i < n; i++)
        {
            var s = b[i];
            for (var k = 0; k < i; k++) s -= l[i * n + k] * b[k];
            b[i] = s / l[i * n + i];
        }
        for (var i = n - 1; i >= 0; i--)
        {
            var s = b[i];
            for (var k = i + 1; k < n; k++) s -= l[k * n + i] * b[k];
            b[i] = s / l[i * n + i];
        }
    }

    /// <summary>
    /// LU with partial pivoting, in place. Returns false for an exactly singular pivot. <paramref name="piv"/>
    /// holds the row taken at each step.
    /// </summary>
    public static bool LuFactor(Span<double> a, int n, Span<int> piv)
    {
        var ok = true;
        for (var k = 0; k < n; k++)
        {
            var p = k; var best = Math.Abs(a[k * n + k]);
            for (var i = k + 1; i < n; i++) { var v = Math.Abs(a[i * n + k]); if (v > best) { best = v; p = i; } }
            piv[k] = p;
            if (p != k) for (var j = 0; j < n; j++) (a[k * n + j], a[p * n + j]) = (a[p * n + j], a[k * n + j]);
            var d = a[k * n + k];
            if (d == 0) { ok = false; continue; }
            for (var i = k + 1; i < n; i++)
            {
                var f = a[i * n + k] / d;
                a[i * n + k] = f;
                if (f == 0) continue;
                for (var j = k + 1; j < n; j++) a[i * n + j] -= f * a[k * n + j];
            }
        }
        return ok;
    }

    public static void LuSolve(ReadOnlySpan<double> lu, int n, ReadOnlySpan<int> piv, Span<double> b)
    {
        for (var k = 0; k < n; k++) if (piv[k] != k) (b[k], b[piv[k]]) = (b[piv[k]], b[k]);
        for (var i = 0; i < n; i++)
        {
            var s = b[i];
            for (var k = 0; k < i; k++) s -= lu[i * n + k] * b[k];
            b[i] = s;
        }
        for (var i = n - 1; i >= 0; i--)
        {
            var s = b[i];
            for (var k = i + 1; k < n; k++) s -= lu[i * n + k] * b[k];
            b[i] = s / lu[i * n + i];
        }
    }

    /// <summary>
    /// Solves a square system by LU; returns null when singular. Neither input is modified.
    /// </summary>
    public static double[]? Solve(ReadOnlySpan<double> a, int n, ReadOnlySpan<double> b)
    {
        var lu = a.ToArray();
        Span<int> piv = n <= 256 ? stackalloc int[n] : new int[n];
        if (!LuFactor(lu, n, piv)) return null;
        var x = b.ToArray();
        LuSolve(lu, n, piv, x);
        for (var i = 0; i < n; i++) if (!double.IsFinite(x[i])) return null;
        return x;
    }

    /// <summary>
    /// Householder QR with column pivoting (LINPACK/MINPACK <c>qrfac</c> style) of a row-major (m, n)
    /// matrix, in place. On return the upper triangle of the first min(m,n) rows holds R, the
    /// Householder vectors are stored below the diagonal with scalars in <paramref name="tau"/>,
    /// and <paramref name="perm"/> maps column j of R to original column perm[j].
    /// </summary>
    public static void QrPivot(Span<double> a, int m, int n, Span<int> perm, Span<double> tau, bool pivot = true)
    {
        var k = Math.Min(m, n);
        Span<double> norms = n <= 256 ? stackalloc double[n] : new double[n];
        Span<double> norms0 = n <= 256 ? stackalloc double[n] : new double[n];
        for (var j = 0; j < n; j++)
        {
            perm[j] = j;
            double s = 0;
            for (var i = 0; i < m; i++) s += a[i * n + j] * a[i * n + j];
            norms[j] = norms0[j] = Math.Sqrt(s);
        }
        for (var j = 0; j < k; j++)
        {
            if (pivot)
            {
                var p = j;
                for (var c = j + 1; c < n; c++) if (norms[c] > norms[p]) p = c;
                if (p != j)
                {
                    for (var i = 0; i < m; i++) (a[i * n + j], a[i * n + p]) = (a[i * n + p], a[i * n + j]);
                    (perm[j], perm[p]) = (perm[p], perm[j]);
                    (norms[j], norms[p]) = (norms[p], norms[j]);
                    (norms0[j], norms0[p]) = (norms0[p], norms0[j]);
                }
            }
            // Householder vector for column j, rows j..m-1.
            double alpha = 0;
            for (var i = j; i < m; i++) alpha += a[i * n + j] * a[i * n + j];
            alpha = Math.Sqrt(alpha);
            if (alpha == 0) { tau[j] = 0; continue; }
            var x0 = a[j * n + j];
            var beta = x0 >= 0 ? -alpha : alpha;
            var v0 = x0 - beta;
            for (var i = j + 1; i < m; i++) a[i * n + j] /= v0;
            tau[j] = (beta - x0) / beta;
            a[j * n + j] = beta;
            // Apply H = I - tau v vᵀ (v0 = 1) to remaining columns.
            for (var c = j + 1; c < n; c++)
            {
                var s = a[j * n + c];
                for (var i = j + 1; i < m; i++) s += a[i * n + j] * a[i * n + c];
                s *= tau[j];
                a[j * n + c] -= s;
                for (var i = j + 1; i < m; i++) a[i * n + c] -= s * a[i * n + j];
                if (pivot && norms[c] != 0)
                {
                    var t = a[j * n + c] / norms[c];
                    t = Math.Max(0, 1 - t * t);
                    var t2 = norms[c] / norms0[c];
                    if (0.05 * t * t2 * t2 <= 2.2e-16)
                    {
                        double ss = 0;
                        for (var i = j + 1; i < m; i++) ss += a[i * n + c] * a[i * n + c];
                        norms[c] = norms0[c] = Math.Sqrt(ss);
                    }
                    else norms[c] *= Math.Sqrt(t);
                }
            }
        }
    }

    /// <summary>Overwrites b (length m) with Qᵀ b using the reflectors from <see cref="QrPivot"/>.</summary>
    public static void ApplyQt(ReadOnlySpan<double> qr, int m, int n, ReadOnlySpan<double> tau, Span<double> b)
    {
        var k = Math.Min(m, n);
        for (var j = 0; j < k; j++)
        {
            if (tau[j] == 0) continue;
            var s = b[j];
            for (var i = j + 1; i < m; i++) s += qr[i * n + j] * b[i];
            s *= tau[j];
            b[j] -= s;
            for (var i = j + 1; i < m; i++) b[i] -= s * qr[i * n + j];
        }
    }

    /// <summary>Overwrites b (length m) with Q b using the reflectors from <see cref="QrPivot"/>.</summary>
    public static void ApplyQ(ReadOnlySpan<double> qr, int m, int n, ReadOnlySpan<double> tau, Span<double> b)
    {
        var k = Math.Min(m, n);
        for (var j = k - 1; j >= 0; j--)
        {
            if (tau[j] == 0) continue;
            var s = b[j];
            for (var i = j + 1; i < m; i++) s += qr[i * n + j] * b[i];
            s *= tau[j];
            b[j] -= s;
            for (var i = j + 1; i < m; i++) b[i] -= s * qr[i * n + j];
        }
    }

    /// <summary>
    /// Minimum-norm least-squares solution of A x ≈ b (A row-major (m, n)) through GraviNum's SVD,
    /// discarding singular values below <c>rcond * s_max</c> (rcond defaults to eps·max(m, n), as
    /// <c>numpy.linalg.lstsq</c>).
    /// </summary>
    public static double[] LstsqSvd(double[] a, int m, int n, ReadOnlySpan<double> b, double rcond = -1)
    {
        var svd = Decomposition.Svd(new NdArray((double[])a.Clone(), m, n));
        var u = svd.U.AsContiguous().ToArray();
        var s = svd.SingularValues.ToArray();
        var v = svd.V.AsContiguous().ToArray();
        var k = s.Length;
        var uc = u.Length / m;
        var vc = v.Length / n;
        if (rcond < 0) rcond = 2.220446049250313e-16 * Math.Max(m, n);
        var smax = k > 0 ? s[0] : 0;
        var x = new double[n];
        for (var c = 0; c < k; c++)
        {
            if (s[c] <= rcond * smax || s[c] == 0) continue;
            double proj = 0;
            for (var i = 0; i < m; i++) proj += u[i * uc + c] * b[i];
            proj /= s[c];
            for (var j = 0; j < n; j++) x[j] += v[j * vc + c] * proj;
        }
        return x;
    }

    /// <summary>Symmetrises an (n, n) matrix in place by averaging with its transpose.</summary>
    public static void Symmetrize(Span<double> a, int n)
    {
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
            {
                var v = 0.5 * (a[i * n + j] + a[j * n + i]);
                a[i * n + j] = v; a[j * n + i] = v;
            }
    }

    /// <summary>Row-major copy of a 2-D <see cref="NdArray"/>, validating its shape.</summary>
    public static double[] ToMatrix(NdArray a, int rows, int cols, string name)
    {
        if (a.Rank == 1 && rows == 1 && a.Size == cols) return a.ToArray();
        if (a.Rank != 2 || a.Shape[0] != rows || a.Shape[1] != cols)
            throw new ArgumentException($"{name} must have shape ({rows}, {cols}).");
        return a.ToArray();
    }
}
