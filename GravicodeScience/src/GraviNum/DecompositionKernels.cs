using System.Runtime.CompilerServices;

namespace Gravicode.Science.GraviNum;

/// <summary>
/// Dense factorisation kernels working on flat row-major buffers.
/// </summary>
/// <remarks>
/// <para>
/// These are the transform-and-iterate algorithms behind <see cref="Decomposition.Svd"/> and
/// <see cref="Decomposition.SymmetricEigen"/>. Both reduce the matrix to a condensed form with a
/// fixed number of Householder reflections and then run a shifted QR iteration on that form. The
/// Jacobi methods they replaced sweep the whole matrix repeatedly instead, which is why they cost
/// one to two orders of magnitude more; the Jacobi versions remain public as the reference the
/// tests compare against.
/// </para>
/// <para>
/// Everything here works on <c>double[]</c> rather than <see cref="NdArray"/> for the same reason
/// Lloyd's loop in <c>KMeans</c> does: the indexer's stride arithmetic and bounds checks dominate
/// once they sit inside an O(n³) loop.
/// </para>
/// </remarks>
internal static class DecompositionKernels
{
    /// <summary>Roughly <c>2^-52</c>: the point at which an off-diagonal counts as zero.</summary>
    private const double Epsilon = 2.220446049250313e-16;

    /// <summary>Underflow guard for the SVD's negligibility tests.</summary>
    private const double Tiny = 1.0e-300;

    /// <summary><c>sqrt(a² + b²)</c> without squaring either argument into an overflow.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double Hypot(double a, double b)
    {
        var absA = Math.Abs(a);
        var absB = Math.Abs(b);

        if (absA > absB)
        {
            var r = absB / absA;
            return absA * Math.Sqrt(1.0 + r * r);
        }

        if (absB == 0.0) return 0.0;

        var q = absA / absB;
        return absB * Math.Sqrt(1.0 + q * q);
    }

    // ================================================================ symmetric eigen

    /// <summary>
    /// Reduces a symmetric matrix to tridiagonal form by Householder reflections, accumulating
    /// the orthogonal transform in place.
    /// </summary>
    /// <param name="z">
    /// On entry the symmetric matrix, row-major <c>n×n</c>. On exit the accumulated transform.
    /// </param>
    /// <param name="d">Receives the diagonal of the tridiagonal form.</param>
    /// <param name="e">Receives the off-diagonal, in <c>e[1..n-1]</c>.</param>
    /// <param name="n">Matrix order.</param>
    /// <remarks>This is the classic <c>tred2</c>; only the lower triangle of <paramref name="z"/> is read.</remarks>
    public static void Tridiagonalize(double[] z, double[] d, double[] e, int n)
    {
        for (var j = 0; j < n; j++) d[j] = z[(n - 1) * n + j];

        // One reflection per column, working up from the last, each zeroing everything in the
        // row beyond the subdiagonal.
        for (var i = n - 1; i > 0; i--)
        {
            var scale = 0.0;
            var h = 0.0;
            for (var k = 0; k < i; k++) scale += Math.Abs(d[k]);

            if (scale == 0.0)
            {
                // The row is already zero; nothing to reflect.
                e[i] = d[i - 1];
                for (var j = 0; j < i; j++)
                {
                    d[j] = z[(i - 1) * n + j];
                    z[i * n + j] = 0.0;
                    z[j * n + i] = 0.0;
                }
                d[i] = h;
                continue;
            }

            for (var k = 0; k < i; k++)
            {
                d[k] /= scale;
                h += d[k] * d[k];
            }

            var f = d[i - 1];
            var g = Math.Sqrt(h);
            if (f > 0) g = -g;

            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for (var j = 0; j < i; j++) e[j] = 0.0;

            // Apply the reflection from both sides at once, exploiting symmetry.
            for (var j = 0; j < i; j++)
            {
                f = d[j];
                z[j * n + i] = f;
                g = e[j] + z[j * n + j] * f;
                for (var k = j + 1; k <= i - 1; k++)
                {
                    g += z[k * n + j] * d[k];
                    e[k] += z[k * n + j] * f;
                }
                e[j] = g;
            }

            f = 0.0;
            for (var j = 0; j < i; j++)
            {
                e[j] /= h;
                f += e[j] * d[j];
            }

            var hh = f / (h + h);
            for (var j = 0; j < i; j++) e[j] -= hh * d[j];

            for (var j = 0; j < i; j++)
            {
                f = d[j];
                g = e[j];
                for (var k = j; k <= i - 1; k++) z[k * n + j] -= f * e[k] + g * d[k];
                d[j] = z[(i - 1) * n + j];
                z[i * n + j] = 0.0;
            }

            d[i] = h;
        }

        // Unwind the reflections to turn the stored vectors into the transform itself.
        for (var i = 0; i < n - 1; i++)
        {
            z[(n - 1) * n + i] = z[i * n + i];
            z[i * n + i] = 1.0;

            var h = d[i + 1];
            if (h != 0.0)
            {
                for (var k = 0; k <= i; k++) d[k] = z[k * n + i + 1] / h;
                for (var j = 0; j <= i; j++)
                {
                    var g = 0.0;
                    for (var k = 0; k <= i; k++) g += z[k * n + i + 1] * z[k * n + j];
                    for (var k = 0; k <= i; k++) z[k * n + j] -= g * d[k];
                }
            }

            for (var k = 0; k <= i; k++) z[k * n + i + 1] = 0.0;
        }

        for (var j = 0; j < n; j++)
        {
            d[j] = z[(n - 1) * n + j];
            z[(n - 1) * n + j] = 0.0;
        }
        z[(n - 1) * n + n - 1] = 1.0;
        e[0] = 0.0;
    }

    /// <summary>
    /// Diagonalises a symmetric tridiagonal matrix by implicit QL iteration with Wilkinson
    /// shifts, updating the accumulated transform so its columns become the eigenvectors.
    /// </summary>
    /// <param name="d">Diagonal on entry, eigenvalues on exit.</param>
    /// <param name="e">Off-diagonal in <c>e[1..n-1]</c>; destroyed.</param>
    /// <param name="z">The transform from <see cref="Tridiagonalize"/>; eigenvectors on exit.</param>
    /// <param name="n">Matrix order.</param>
    /// <remarks>
    /// <para>
    /// This is <c>tql2</c>. The shift makes convergence cubic for a well-separated eigenvalue, so
    /// in practice a couple of iterations deflate each one — against the whole-matrix sweeps a
    /// Jacobi method needs.
    /// </para>
    /// <para>
    /// Eigenvalues come out in <em>no particular order</em>: they emerge as each block deflates,
    /// which has nothing to do with their magnitude. Callers that promise an order must sort, and
    /// must carry the columns of <paramref name="z"/> along with the values.
    /// </para>
    /// </remarks>
    public static void TridiagonalQl(double[] d, double[] e, double[] z, int n)
    {
        for (var i = 1; i < n; i++) e[i - 1] = e[i];
        e[n - 1] = 0.0;

        var f = 0.0;
        var tst1 = 0.0;

        for (var l = 0; l < n; l++)
        {
            tst1 = Math.Max(tst1, Math.Abs(d[l]) + Math.Abs(e[l]));

            // Find the end of the block that still couples to l.
            var m = l;
            while (m < n && Math.Abs(e[m]) > Epsilon * tst1) m++;

            if (m > l)
            {
                do
                {
                    // Wilkinson shift from the trailing 2x2.
                    var g = d[l];
                    var p = (d[l + 1] - g) / (2.0 * e[l]);
                    var r = Hypot(p, 1.0);
                    if (p < 0) r = -r;

                    d[l] = e[l] / (p + r);
                    d[l + 1] = e[l] * (p + r);

                    var dl1 = d[l + 1];
                    var h = g - d[l];
                    for (var i = l + 2; i < n; i++) d[i] -= h;
                    f += h;

                    // Chase the bulge back down the block.
                    p = d[m];
                    var c = 1.0;
                    var c2 = c;
                    var c3 = c;
                    var el1 = e[l + 1];
                    var s = 0.0;
                    var s2 = 0.0;

                    for (var i = m - 1; i >= l; i--)
                    {
                        c3 = c2;
                        c2 = c;
                        s2 = s;
                        g = c * e[i];
                        h = c * p;
                        r = Hypot(p, e[i]);
                        e[i + 1] = s * r;
                        s = e[i] / r;
                        c = p / r;
                        p = c * d[i] - s * g;
                        d[i + 1] = h + s * (c * g + s * d[i]);

                        for (var k = 0; k < n; k++)
                        {
                            h = z[k * n + i + 1];
                            z[k * n + i + 1] = s * z[k * n + i] + c * h;
                            z[k * n + i] = c * z[k * n + i] - s * h;
                        }
                    }

                    p = -s * s2 * c3 * el1 * e[l] / dl1;
                    e[l] = s * p;
                    d[l] = c * p;
                }
                while (Math.Abs(e[l]) > Epsilon * tst1);
            }

            d[l] += f;
            e[l] = 0.0;
        }
    }

    // ================================================================ SVD

    /// <summary>
    /// Singular value decomposition by Householder bidiagonalisation followed by an implicit
    /// shifted QR iteration on the bidiagonal form (Golub–Kahan–Reinsch).
    /// </summary>
    /// <param name="a">The matrix, row-major <c>m×n</c>; destroyed.</param>
    /// <param name="m">Rows; must be at least <paramref name="n"/>.</param>
    /// <param name="n">Columns.</param>
    /// <param name="u">Receives the left factor, row-major <c>m×n</c>. Ignored when <paramref name="wantU"/> is false.</param>
    /// <param name="s">Receives the singular values, descending, length <paramref name="n"/>.</param>
    /// <param name="v">Receives the right factor, row-major <c>n×n</c>. Ignored when <paramref name="wantV"/> is false.</param>
    /// <param name="wantU">Whether the left factor is needed.</param>
    /// <param name="wantV">Whether the right factor is needed.</param>
    /// <remarks>
    /// The rotations that build <paramref name="u"/> cost <c>O(m n²)</c>, which for a tall matrix
    /// is nearly the whole run — PCA on 20 000×20 never looks at <c>U</c>. Skipping a factor drops
    /// only its accumulation loops; the singular values are computed the same way either way,
    /// because the rotations that produce them also update <c>s</c> and <c>e</c>.
    /// </remarks>
    public static void GolubKahanSvd(
        double[] a, int m, int n, double[] u, double[] s, double[] v, bool wantU = true, bool wantV = true)
    {
        var e = new double[n];
        var work = new double[m];

        var nct = Math.Min(m - 1, n);          // column reflections
        var nrt = Math.Max(0, Math.Min(n - 2, m)); // row reflections

        // ---- reduce to bidiagonal form ------------------------------------------------------
        for (var k = 0; k < Math.Max(nct, nrt); k++)
        {
            if (k < nct)
            {
                s[k] = 0;
                for (var i = k; i < m; i++) s[k] = Hypot(s[k], a[i * n + k]);

                if (s[k] != 0.0)
                {
                    if (a[k * n + k] < 0.0) s[k] = -s[k];
                    for (var i = k; i < m; i++) a[i * n + k] /= s[k];
                    a[k * n + k] += 1.0;
                }
                s[k] = -s[k];
            }

            for (var j = k + 1; j < n; j++)
            {
                if (k < nct && s[k] != 0.0)
                {
                    var t = 0.0;
                    for (var i = k; i < m; i++) t += a[i * n + k] * a[i * n + j];
                    t = -t / a[k * n + k];
                    for (var i = k; i < m; i++) a[i * n + j] += t * a[i * n + k];
                }
                e[j] = a[k * n + j];
            }

            if (wantU && k < nct)
                for (var i = k; i < m; i++) u[i * n + k] = a[i * n + k];

            if (k >= nrt) continue;

            e[k] = 0;
            for (var i = k + 1; i < n; i++) e[k] = Hypot(e[k], e[i]);

            if (e[k] != 0.0)
            {
                if (e[k + 1] < 0.0) e[k] = -e[k];
                for (var i = k + 1; i < n; i++) e[i] /= e[k];
                e[k + 1] += 1.0;
            }
            e[k] = -e[k];

            if (k + 1 < m && e[k] != 0.0)
            {
                for (var i = k + 1; i < m; i++) work[i] = 0.0;
                for (var j = k + 1; j < n; j++)
                    for (var i = k + 1; i < m; i++)
                        work[i] += e[j] * a[i * n + j];

                for (var j = k + 1; j < n; j++)
                {
                    var t = -e[j] / e[k + 1];
                    for (var i = k + 1; i < m; i++) a[i * n + j] += t * work[i];
                }
            }

            if (wantV)
                for (var i = k + 1; i < n; i++) v[i * n + k] = e[i];
        }

        // ---- close off the bidiagonal -------------------------------------------------------
        var p = Math.Min(n, m + 1);
        if (nct < n) s[nct] = a[nct * n + nct];
        if (m < p) s[p - 1] = 0.0;
        if (nrt + 1 < p) e[nrt] = a[nrt * n + p - 1];
        e[p - 1] = 0.0;

        // ---- expand the stored reflections into U and V -------------------------------------
        if (wantU) ExpandLeftFactor(u, s, m, n, nct);
        if (wantV) ExpandRightFactor(v, e, n, nrt);

        // ---- QR iteration on the bidiagonal -------------------------------------------------
        var pp = p - 1;
        while (p > 0)
        {
            // Classify the trailing block: which of the four cases applies decides whether we
            // deflate a negligible value, split the block, take a QR step, or accept convergence.
            int k;
            for (k = p - 2; k >= 0; k--)
            {
                if (Math.Abs(e[k]) <= Tiny + Epsilon * (Math.Abs(s[k]) + Math.Abs(s[k + 1])))
                {
                    e[k] = 0.0;
                    break;
                }
            }

            int kase;
            if (k == p - 2)
            {
                kase = 4;
            }
            else
            {
                int ks;
                for (ks = p - 1; ks >= k; ks--)
                {
                    if (ks == k) break;
                    var t = (ks != p ? Math.Abs(e[ks]) : 0.0)
                          + (ks != k + 1 ? Math.Abs(e[ks - 1]) : 0.0);
                    if (Math.Abs(s[ks]) <= Tiny + Epsilon * t)
                    {
                        s[ks] = 0.0;
                        break;
                    }
                }

                if (ks == k) kase = 3;
                else if (ks == p - 1) kase = 1;
                else { kase = 2; k = ks; }
            }
            k++;

            switch (kase)
            {
                case 1: DeflateNegligibleValue(s, e, v, n, p, k, wantV); break;
                case 2: SplitAtNegligibleValue(s, e, u, m, n, p, k, wantU); break;
                case 3: QrStep(s, e, u, v, m, n, p, k, wantU, wantV); break;
                default:
                    Converge(s, v, u, m, n, pp, ref k, wantU, wantV);
                    p--;
                    break;
            }
        }

        // The iteration leaves the values descending already; the wrapper relies on that.
    }

    /// <summary>Turns the column reflectors stored in <paramref name="u"/> into the factor itself.</summary>
    private static void ExpandLeftFactor(double[] u, double[] s, int m, int n, int nct)
    {
        for (var j = nct; j < n; j++)
        {
            for (var i = 0; i < m; i++) u[i * n + j] = 0.0;
            u[j * n + j] = 1.0;
        }

        for (var k = nct - 1; k >= 0; k--)
        {
            if (s[k] == 0.0)
            {
                for (var i = 0; i < m; i++) u[i * n + k] = 0.0;
                u[k * n + k] = 1.0;
                continue;
            }

            for (var j = k + 1; j < n; j++)
            {
                var t = 0.0;
                for (var i = k; i < m; i++) t += u[i * n + k] * u[i * n + j];
                t = -t / u[k * n + k];
                for (var i = k; i < m; i++) u[i * n + j] += t * u[i * n + k];
            }

            for (var i = k; i < m; i++) u[i * n + k] = -u[i * n + k];
            u[k * n + k] += 1.0;
            for (var i = 0; i < k - 1; i++) u[i * n + k] = 0.0;
        }
    }

    /// <summary>Turns the row reflectors stored in <paramref name="v"/> into the factor itself.</summary>
    private static void ExpandRightFactor(double[] v, double[] e, int n, int nrt)
    {
        for (var k = n - 1; k >= 0; k--)
        {
            if (k < nrt && e[k] != 0.0)
            {
                for (var j = k + 1; j < n; j++)
                {
                    var t = 0.0;
                    for (var i = k + 1; i < n; i++) t += v[i * n + k] * v[i * n + j];
                    t = -t / v[(k + 1) * n + k];
                    for (var i = k + 1; i < n; i++) v[i * n + j] += t * v[i * n + k];
                }
            }

            for (var i = 0; i < n; i++) v[i * n + k] = 0.0;
            v[k * n + k] = 1.0;
        }
    }

    /// <summary>Rotates away a negligible trailing singular value (case 1).</summary>
    private static void DeflateNegligibleValue(
        double[] s, double[] e, double[] v, int n, int p, int k, bool wantV)
    {
        var f = e[p - 2];
        e[p - 2] = 0.0;

        for (var j = p - 2; j >= k; j--)
        {
            var t = Hypot(s[j], f);
            var cs = s[j] / t;
            var sn = f / t;
            s[j] = t;

            if (j != k)
            {
                f = -sn * e[j - 1];
                e[j - 1] = cs * e[j - 1];
            }

            if (!wantV) continue;
            for (var i = 0; i < n; i++)
            {
                t = cs * v[i * n + j] + sn * v[i * n + p - 1];
                v[i * n + p - 1] = -sn * v[i * n + j] + cs * v[i * n + p - 1];
                v[i * n + j] = t;
            }
        }
    }

    /// <summary>Splits the block at a negligible singular value (case 2).</summary>
    private static void SplitAtNegligibleValue(
        double[] s, double[] e, double[] u, int m, int n, int p, int k, bool wantU)
    {
        var f = e[k - 1];
        e[k - 1] = 0.0;

        for (var j = k; j < p; j++)
        {
            var t = Hypot(s[j], f);
            var cs = s[j] / t;
            var sn = f / t;
            s[j] = t;
            f = -sn * e[j];
            e[j] = cs * e[j];

            if (!wantU) continue;
            for (var i = 0; i < m; i++)
            {
                t = cs * u[i * n + j] + sn * u[i * n + k - 1];
                u[i * n + k - 1] = -sn * u[i * n + j] + cs * u[i * n + k - 1];
                u[i * n + j] = t;
            }
        }
    }

    /// <summary>One implicit shifted QR step on the bidiagonal block (case 3).</summary>
    private static void QrStep(
        double[] s, double[] e, double[] u, double[] v, int m, int n, int p, int k, bool wantU, bool wantV)
    {
        // Scale before forming the shift, so a large or tiny block does not overflow the squares.
        var scale = Math.Max(
            Math.Max(Math.Max(Math.Abs(s[p - 1]), Math.Abs(s[p - 2])), Math.Abs(e[p - 2])),
            Math.Max(Math.Abs(s[k]), Math.Abs(e[k])));

        var sp = s[p - 1] / scale;
        var spm1 = s[p - 2] / scale;
        var epm1 = e[p - 2] / scale;
        var sk = s[k] / scale;
        var ek = e[k] / scale;

        var b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
        var c = sp * epm1 * (sp * epm1);

        var shift = 0.0;
        if (b != 0.0 || c != 0.0)
        {
            shift = Math.Sqrt(b * b + c);
            if (b < 0.0) shift = -shift;
            shift = c / (b + shift);
        }

        var f = (sk + sp) * (sk - sp) + shift;
        var g = sk * ek;

        // Chase the bulge down the bidiagonal, one Givens pair per column.
        for (var j = k; j < p - 1; j++)
        {
            var t = Hypot(f, g);
            var cs = f / t;
            var sn = g / t;
            if (j != k) e[j - 1] = t;

            f = cs * s[j] + sn * e[j];
            e[j] = cs * e[j] - sn * s[j];
            g = sn * s[j + 1];
            s[j + 1] = cs * s[j + 1];

            if (wantV)
                for (var i = 0; i < n; i++)
                {
                    t = cs * v[i * n + j] + sn * v[i * n + j + 1];
                    v[i * n + j + 1] = -sn * v[i * n + j] + cs * v[i * n + j + 1];
                    v[i * n + j] = t;
                }

            t = Hypot(f, g);
            cs = f / t;
            sn = g / t;
            s[j] = t;
            f = cs * e[j] + sn * s[j + 1];
            s[j + 1] = -sn * e[j] + cs * s[j + 1];
            g = sn * e[j + 1];
            e[j + 1] = cs * e[j + 1];

            if (!wantU || j >= m - 1) continue;
            for (var i = 0; i < m; i++)
            {
                t = cs * u[i * n + j] + sn * u[i * n + j + 1];
                u[i * n + j + 1] = -sn * u[i * n + j] + cs * u[i * n + j + 1];
                u[i * n + j] = t;
            }
        }

        e[p - 2] = f;
    }

    /// <summary>Makes the converged value positive and bubbles it into descending order (case 4).</summary>
    private static void Converge(
        double[] s, double[] v, double[] u, int m, int n, int pp, ref int k, bool wantU, bool wantV)
    {
        if (s[k] <= 0.0)
        {
            // A negative value is made positive by flipping the sign of its right vector, which
            // keeps U S V^T equal to the original matrix.
            s[k] = s[k] < 0.0 ? -s[k] : 0.0;
            if (wantV)
                for (var i = 0; i <= pp; i++) v[i * n + k] = -v[i * n + k];
        }

        while (k < pp)
        {
            if (s[k] >= s[k + 1]) break;

            (s[k], s[k + 1]) = (s[k + 1], s[k]);

            if (wantV && k < n - 1)
                for (var i = 0; i < n; i++)
                    (v[i * n + k], v[i * n + k + 1]) = (v[i * n + k + 1], v[i * n + k]);

            if (wantU && k < m - 1)
                for (var i = 0; i < m; i++)
                    (u[i * n + k], u[i * n + k + 1]) = (u[i * n + k + 1], u[i * n + k]);

            k++;
        }
    }
}
