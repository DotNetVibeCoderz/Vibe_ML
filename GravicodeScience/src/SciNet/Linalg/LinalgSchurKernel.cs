using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Householder Hessenberg reduction, the Francis double-shift QR iteration with Schur-vector
/// accumulation, 2×2 block standardisation, and the real-to-complex Schur conversion.
/// </summary>
/// <remarks>
/// <para>
/// GraviNum's <c>Decomposition.Eigenvalues</c> reduces by Gaussian elimination (not orthogonal)
/// and throws the transformation away, so it can give eigenvalues but neither a Schur form nor
/// eigenvectors. Everything that scipy.linalg builds on <c>gees</c> — Schur, Eig, Sqrtm, Logm,
/// Funm, Sylvester, Lyapunov — needs the orthogonal factor, which is why this kernel exists.
/// </para>
/// <para>
/// The iteration follows EISPACK <c>hqr2</c> (via JAMA), including Wilkinson's exceptional shift
/// at iteration 10 and MATLAB's at 30. Two things it does not do on its own and that are added
/// here: it leaves "negligible" subdiagonals non-zero rather than setting them to zero, and its
/// complex 2×2 blocks are in no particular form. Both are fixed after convergence — the first by
/// zeroing every subdiagonal outside a complex block, the second with LAPACK's <c>dlanv2</c>
/// standardisation (equal diagonal, off-diagonals of opposite sign).
/// </para>
/// </remarks>
internal static class LinalgSchurKernel
{
    private const double Eps = LinalgKernels.Eps;

    /// <summary>
    /// Reduces <paramref name="h"/> (row-major n×n) to upper Hessenberg form in place by Householder
    /// reflections, accumulating <c>A = Q H Qᵀ</c> into <paramref name="q"/> when it is not null.
    /// </summary>
    public static void Hessenberg(double[] h, double[]? q, int n)
    {
        if (q is not null)
        {
            Array.Clear(q);
            for (var i = 0; i < n; i++) q[i * n + i] = 1.0;
        }

        var v = new double[n];
        var w = new double[n];
        for (var k = 0; k < n - 2; k++)
        {
            // Reflector for column k below the subdiagonal.
            var scale = 0.0;
            for (var i = k + 1; i < n; i++) scale += Math.Abs(h[i * n + k]);
            if (scale == 0.0) continue;

            var sigma = 0.0;
            for (var i = k + 1; i < n; i++)
            {
                v[i] = h[i * n + k] / scale;
                sigma += v[i] * v[i];
            }
            var norm = Math.Sqrt(sigma);
            var alpha = v[k + 1] > 0 ? -norm : norm;
            v[k + 1] -= alpha;
            var vtv = 0.0;
            for (var i = k + 1; i < n; i++) vtv += v[i] * v[i];
            if (vtv == 0.0) continue;
            var beta = 2.0 / vtv;

            // Left: rows k+1.., columns k.. :  H <- (I - beta v v^T) H, done as w = v^T H then rank-1.
            var width = n - k;
            var wSpan = w.AsSpan(0, width);
            wSpan.Clear();
            for (var i = k + 1; i < n; i++)
                LinAlg.AxpySpan(v[i], h.AsSpan(i * n + k, width), wSpan);
            for (var i = k + 1; i < n; i++)
                LinAlg.AxpySpan(-beta * v[i], wSpan, h.AsSpan(i * n + k, width));

            // Right: all rows, columns k+1.. :  H <- H (I - beta v v^T); likewise Q.
            var vTail = v.AsSpan(k + 1, n - k - 1);
            ApplyRight(h, n, k + 1, vTail, beta);
            if (q is not null) ApplyRight(q, n, k + 1, vTail, beta);

            h[(k + 1) * n + k] = alpha * scale;
            for (var i = k + 2; i < n; i++) h[i * n + k] = 0.0;
        }
    }

    private static void ApplyRight(double[] m, int n, int start, ReadOnlySpan<double> v, double beta)
    {
        var len = v.Length;
        if (n >= 192)
        {
            var vv = v.ToArray();
            Parallel.For(0, n, i =>
            {
                var row = m.AsSpan(i * n + start, len);
                var s = LinAlg.InnerSpan(row, vv);
                if (s != 0.0) LinAlg.AxpySpan(-beta * s, vv, row);
            });
            return;
        }
        for (var i = 0; i < n; i++)
        {
            var row = m.AsSpan(i * n + start, len);
            var s = LinAlg.InnerSpan(row, v);
            if (s != 0.0) LinAlg.AxpySpan(-beta * s, v, row);
        }
    }

    /// <summary>
    /// Real Schur decomposition <c>A = Z T Zᵀ</c>. On entry <paramref name="t"/> holds A; on exit the
    /// quasi-upper-triangular T with standardised 2×2 blocks. Returns the eigenvalues in diagonal
    /// order (a complex pair as <c>a + ib</c> then <c>a - ib</c>).
    /// </summary>
    /// <param name="t">Matrix in, Schur form out (row-major n×n).</param>
    /// <param name="z">Receives the orthogonal Schur vectors, or null to skip accumulating them.</param>
    /// <param name="n">Order.</param>
    public static (double[] Re, double[] Im) RealSchur(double[] t, double[]? z, int n)
    {
        var wr = new double[n];
        var wi = new double[n];
        if (n == 0) return (wr, wi);

        // Accumulating Z needs a buffer; with none requested, a scratch one keeps hqr2's code
        // path single. The cost is O(n^3) extra only when the caller did not ask.
        var zz = z ?? new double[n * n];
        Hessenberg(t, zz, n);
        // The iteration rotates pairs/triples of Z's columns; on a row-major buffer those are
        // strided, so it works on Zᵀ, where each rotation streams through contiguous rows.
        var accumulate = z is not null;
        if (accumulate) TransposeInPlace(zz, n);
        Francis(t, zz, n, wr, wi, accumulate);
        Clean(t, zz, n, wr, wi, accumulate);
        if (accumulate) TransposeInPlace(zz, n);
        return (wr, wi);
    }

    private static void TransposeInPlace(double[] a, int n)
    {
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++) (a[i * n + j], a[j * n + i]) = (a[j * n + i], a[i * n + j]);
    }

    /// <summary>The hqr2 iteration on a Hessenberg matrix, full Schur form (not just the active block).</summary>
    private static void Francis(double[] H, double[] V, int nn, double[] d, double[] e, bool accumulate)
    {
        var n = nn - 1;
        const int low = 0;
        var high = nn - 1;
        var exshift = 0.0;
        double p = 0, q = 0, r = 0, s = 0, z = 0;
        double w, x, y;

        var norm = 0.0;
        for (var i = 0; i < nn; i++)
            for (var j = Math.Max(i - 1, 0); j < nn; j++) norm += Math.Abs(H[i * nn + j]);

        var iter = 0;
        var totalIter = 0;
        var maxTotal = 100 * Math.Max(nn, 10);
        while (n >= low)
        {
            // Look for a single small subdiagonal element.
            var l = n;
            while (l > low)
            {
                s = Math.Abs(H[(l - 1) * nn + l - 1]) + Math.Abs(H[l * nn + l]);
                if (s == 0.0) s = norm;
                if (Math.Abs(H[l * nn + l - 1]) < Eps * s) break;
                l--;
            }

            if (l == n)
            {
                // One root.
                H[n * nn + n] += exshift;
                d[n] = H[n * nn + n];
                e[n] = 0.0;
                n--;
                iter = 0;
            }
            else if (l == n - 1)
            {
                // Two roots.
                w = H[n * nn + n - 1] * H[(n - 1) * nn + n];
                p = (H[(n - 1) * nn + n - 1] - H[n * nn + n]) / 2.0;
                q = p * p + w;
                z = Math.Sqrt(Math.Abs(q));
                H[n * nn + n] += exshift;
                H[(n - 1) * nn + n - 1] += exshift;
                x = H[n * nn + n];

                if (q >= 0)
                {
                    // Real pair: rotate the block to triangular.
                    z = p >= 0 ? p + z : p - z;
                    d[n - 1] = x + z;
                    d[n] = d[n - 1];
                    if (z != 0.0) d[n] = x - w / z;
                    e[n - 1] = 0.0;
                    e[n] = 0.0;
                    x = H[n * nn + n - 1];
                    s = Math.Abs(x) + Math.Abs(z);
                    p = x / s;
                    q = z / s;
                    r = Math.Sqrt(p * p + q * q);
                    p /= r;
                    q /= r;

                    for (var j = n - 1; j < nn; j++)
                    {
                        z = H[(n - 1) * nn + j];
                        H[(n - 1) * nn + j] = q * z + p * H[n * nn + j];
                        H[n * nn + j] = q * H[n * nn + j] - p * z;
                    }
                    for (var i = 0; i <= n; i++)
                    {
                        z = H[i * nn + n - 1];
                        H[i * nn + n - 1] = q * z + p * H[i * nn + n];
                        H[i * nn + n] = q * H[i * nn + n] - p * z;
                    }
                    if (accumulate)
                        for (var i = low; i <= high; i++)
                        {
                            z = V[(n - 1) * nn + i];
                            V[(n - 1) * nn + i] = q * z + p * V[(n) * nn + i];
                            V[(n) * nn + i] = q * V[(n) * nn + i] - p * z;
                        }
                    H[n * nn + n - 1] = 0.0;
                }
                else
                {
                    d[n - 1] = x + p;
                    d[n] = x + p;
                    e[n - 1] = z;
                    e[n] = -z;
                }
                n -= 2;
                iter = 0;
            }
            else
            {
                // No convergence yet: form the shift.
                x = H[n * nn + n];
                y = 0.0;
                w = 0.0;
                if (l < n)
                {
                    y = H[(n - 1) * nn + n - 1];
                    w = H[n * nn + n - 1] * H[(n - 1) * nn + n];
                }

                if (iter == 10)
                {
                    // Wilkinson's original ad hoc shift.
                    exshift += x;
                    for (var i = low; i <= n; i++) H[i * nn + i] -= x;
                    s = Math.Abs(H[n * nn + n - 1]) + Math.Abs(H[(n - 1) * nn + n - 2]);
                    x = y = 0.75 * s;
                    w = -0.4375 * s * s;
                }

                if (iter == 30)
                {
                    // MATLAB's ad hoc shift.
                    s = (y - x) / 2.0;
                    s = s * s + w;
                    if (s > 0)
                    {
                        s = Math.Sqrt(s);
                        if (y < x) s = -s;
                        s = x - w / ((y - x) / 2.0 + s);
                        for (var i = low; i <= n; i++) H[i * nn + i] -= s;
                        exshift += s;
                        x = y = w = 0.964;
                    }
                }

                iter++;
                if (++totalIter > maxTotal || iter > 200)
                    throw new LinalgException("Schur: the QR iteration did not converge.");

                // Look for two consecutive small subdiagonal elements.
                var m = n - 2;
                while (m >= l)
                {
                    z = H[m * nn + m];
                    r = x - z;
                    s = y - z;
                    p = (r * s - w) / H[(m + 1) * nn + m] + H[m * nn + m + 1];
                    q = H[(m + 1) * nn + m + 1] - z - r - s;
                    r = H[(m + 2) * nn + m + 1];
                    s = Math.Abs(p) + Math.Abs(q) + Math.Abs(r);
                    p /= s;
                    q /= s;
                    r /= s;
                    if (m == l) break;
                    if (Math.Abs(H[m * nn + m - 1]) * (Math.Abs(q) + Math.Abs(r)) <
                        Eps * (Math.Abs(p) * (Math.Abs(H[(m - 1) * nn + m - 1]) + Math.Abs(z) + Math.Abs(H[(m + 1) * nn + m + 1]))))
                        break;
                    m--;
                }

                for (var i = m + 2; i <= n; i++)
                {
                    H[i * nn + i - 2] = 0.0;
                    if (i > m + 2) H[i * nn + i - 3] = 0.0;
                }

                // Double QR step on rows l..n and columns m..n.
                for (var k = m; k <= n - 1; k++)
                {
                    var notlast = k != n - 1;
                    if (k != m)
                    {
                        p = H[k * nn + k - 1];
                        q = H[(k + 1) * nn + k - 1];
                        r = notlast ? H[(k + 2) * nn + k - 1] : 0.0;
                        x = Math.Abs(p) + Math.Abs(q) + Math.Abs(r);
                        if (x == 0.0) continue;
                        p /= x;
                        q /= x;
                        r /= x;
                    }

                    s = Math.Sqrt(p * p + q * q + r * r);
                    if (p < 0) s = -s;
                    if (s == 0) continue;

                    if (k != m) H[k * nn + k - 1] = -s * x;
                    else if (l != m) H[k * nn + k - 1] = -H[k * nn + k - 1];
                    p += s;
                    x = p / s;
                    y = q / s;
                    z = r / s;
                    q /= p;
                    r /= p;

                    for (var j = k; j < nn; j++)
                    {
                        p = H[k * nn + j] + q * H[(k + 1) * nn + j];
                        if (notlast)
                        {
                            p += r * H[(k + 2) * nn + j];
                            H[(k + 2) * nn + j] -= p * z;
                        }
                        H[k * nn + j] -= p * x;
                        H[(k + 1) * nn + j] -= p * y;
                    }

                    var top = Math.Min(n, k + 3);
                    for (var i = 0; i <= top; i++)
                    {
                        p = x * H[i * nn + k] + y * H[i * nn + k + 1];
                        if (notlast)
                        {
                            p += z * H[i * nn + k + 2];
                            H[i * nn + k + 2] -= p * r;
                        }
                        H[i * nn + k] -= p;
                        H[i * nn + k + 1] -= p * q;
                    }

                    if (accumulate)
                        for (var i = low; i <= high; i++)
                        {
                            p = x * V[(k) * nn + i] + y * V[(k + 1) * nn + i];
                            if (notlast)
                            {
                                p += z * V[(k + 2) * nn + i];
                                V[(k + 2) * nn + i] -= p * r;
                            }
                            V[(k) * nn + i] -= p;
                            V[(k + 1) * nn + i] -= p * q;
                        }
                }
            }
        }
    }

    /// <summary>
    /// Zeroes everything below the quasi-triangle and standardises each complex 2×2 block with
    /// <c>dlanv2</c>, recomputing the eigenvalues from the final blocks.
    /// </summary>
    private static void Clean(double[] T, double[] Z, int n, double[] wr, double[] wi, bool accumulate)
    {
        // Which subdiagonals belong to a complex block: hqr2 reported a positive imaginary part
        // at the first index of each pair.
        var block = new bool[n];
        for (var i = 0; i + 1 < n; i++)
            if (wi[i] > 0 && wi[i + 1] < 0) { block[i] = true; i++; }

        for (var i = 1; i < n; i++)
        {
            if (!block[i - 1]) T[i * n + i - 1] = 0.0;
            for (var j = 0; j < i - 1; j++) T[i * n + j] = 0.0;
        }

        for (var k = 0; k < n; k++)
        {
            if (!block[k])
            {
                wr[k] = T[k * n + k];
                wi[k] = 0.0;
                continue;
            }

            var a = T[k * n + k];
            var b = T[k * n + k + 1];
            var c = T[(k + 1) * n + k];
            var d = T[(k + 1) * n + k + 1];
            Lanv2(ref a, ref b, ref c, ref d, out var cs, out var sn);

            // T <- G^T T G with G = [cs -sn; sn cs] acting on indices k, k+1.
            for (var j = k + 2; j < n; j++)
            {
                var t1 = T[k * n + j];
                var t2 = T[(k + 1) * n + j];
                T[k * n + j] = cs * t1 + sn * t2;
                T[(k + 1) * n + j] = -sn * t1 + cs * t2;
            }
            for (var i = 0; i < k; i++)
            {
                var t1 = T[i * n + k];
                var t2 = T[i * n + k + 1];
                T[i * n + k] = cs * t1 + sn * t2;
                T[i * n + k + 1] = -sn * t1 + cs * t2;
            }
            if (accumulate)
                for (var i = 0; i < n; i++)
                {
                    var z1 = Z[k * n + i];
                    var z2 = Z[(k + 1) * n + i];
                    Z[k * n + i] = cs * z1 + sn * z2;
                    Z[(k + 1) * n + i] = -sn * z1 + cs * z2;
                }

            T[k * n + k] = a;
            T[k * n + k + 1] = b;
            T[(k + 1) * n + k] = c;
            T[(k + 1) * n + k + 1] = d;

            if (c == 0.0)
            {
                wr[k] = a; wi[k] = 0.0;
                wr[k + 1] = d; wi[k + 1] = 0.0;
            }
            else
            {
                var im = Math.Sqrt(Math.Abs(b)) * Math.Sqrt(Math.Abs(c));
                wr[k] = a; wi[k] = im;
                wr[k + 1] = d; wi[k + 1] = -im;
            }
            k++;
        }
    }

    /// <summary>
    /// LAPACK <c>dlanv2</c>: the Schur factorisation of a real 2×2 nonsymmetric matrix in
    /// standardised form, <c>[a b; c d] = [cs -sn; sn cs] [aa bb; cc dd] [cs sn; -sn cs]</c>.
    /// </summary>
    internal static void Lanv2(ref double a, ref double b, ref double c, ref double d, out double cs, out double sn)
    {
        const double multpl = 4.0;
        if (c == 0.0)
        {
            cs = 1.0; sn = 0.0;
        }
        else if (b == 0.0)
        {
            // Swap rows and columns.
            cs = 0.0; sn = 1.0;
            (a, d) = (d, a);
            b = -c;
            c = 0.0;
        }
        else if (a - d == 0.0 && Math.Sign(b) != Math.Sign(c))
        {
            cs = 1.0; sn = 0.0;
        }
        else
        {
            var temp = a - d;
            var p = 0.5 * temp;
            var bcmax = Math.Max(Math.Abs(b), Math.Abs(c));
            var bcmis = Math.Min(Math.Abs(b), Math.Abs(c)) * Math.Sign(b) * Math.Sign(c);
            var scale = Math.Max(Math.Abs(p), bcmax);
            var z = p / scale * p + bcmax / scale * bcmis;

            if (z >= multpl * Eps)
            {
                // Real eigenvalues: compute a and d.
                z = p + Math.CopySign(Math.Sqrt(scale) * Math.Sqrt(z), p);
                a = d + z;
                d -= bcmax / z * bcmis;
                var tau = Hypot(c, z);
                cs = z / tau;
                sn = c / tau;
                b -= c;
                c = 0.0;
            }
            else
            {
                // Complex or nearly equal real eigenvalues: make the diagonal equal.
                var sigma = b + c;
                var tau = Hypot(sigma, temp);
                cs = Math.Sqrt(0.5 * (1.0 + Math.Abs(sigma) / tau));
                sn = -(p / (tau * cs)) * (sigma >= 0 ? 1.0 : -1.0);

                var aa = a * cs + b * sn;
                var bb = -a * sn + b * cs;
                var cc = c * cs + d * sn;
                var dd = -c * sn + d * cs;

                a = aa * cs + cc * sn;
                b = bb * cs + dd * sn;
                c = -aa * sn + cc * cs;
                d = -bb * sn + dd * cs;

                temp = 0.5 * (a + d);
                a = temp;
                d = temp;

                if (c != 0.0)
                {
                    if (b != 0.0)
                    {
                        if (Math.Sign(b) == Math.Sign(c))
                        {
                            // Real eigenvalues: reduce to upper triangular.
                            var sab = Math.Sqrt(Math.Abs(b));
                            var sac = Math.Sqrt(Math.Abs(c));
                            p = Math.CopySign(sab * sac, c);
                            tau = 1.0 / Math.Sqrt(Math.Abs(b + c));
                            a = temp + p;
                            d = temp - p;
                            b -= c;
                            c = 0.0;
                            var cs1 = sab * tau;
                            var sn1 = sac * tau;
                            temp = cs * cs1 - sn * sn1;
                            sn = cs * sn1 + sn * cs1;
                            cs = temp;
                        }
                    }
                    else
                    {
                        b = -c;
                        c = 0.0;
                        temp = cs;
                        cs = -sn;
                        sn = temp;
                    }
                }
            }
        }
    }

    private static double Hypot(double x, double y)
    {
        var ax = Math.Abs(x);
        var ay = Math.Abs(y);
        var w = Math.Max(ax, ay);
        var z = Math.Min(ax, ay);
        if (z == 0.0) return w;
        var r = z / w;
        return w * Math.Sqrt(1.0 + r * r);
    }

    /// <summary>
    /// Converts a real Schur form (standardised blocks) into a complex upper-triangular Schur form,
    /// scipy's <c>rsf2csf</c>. Each complex block is rotated so its upper-left entry is the
    /// eigenvalue with positive imaginary part.
    /// </summary>
    public static (Complex[] T, Complex[] Z) ToComplexSchur(ReadOnlySpan<double> tReal, ReadOnlySpan<double> zReal, int n)
    {
        var T = LinalgKernels.ToComplex(tReal);
        var Z = LinalgKernels.ToComplex(zReal);
        for (var m = n - 1; m > 0; m--)
        {
            var sub = T[m * n + m - 1];
            if (sub == Complex.Zero) continue;

            // Eigenvalues of the 2x2 block; take the one with positive imaginary part.
            var a = T[(m - 1) * n + m - 1];
            var b = T[(m - 1) * n + m];
            var c = sub;
            var d = T[m * n + m];
            var half = (a + d) / 2.0;
            var disc = Complex.Sqrt((a - d) * (a - d) / 4.0 + b * c);
            var lambda = half + disc;
            if (lambda.Imaginary < 0) lambda = half - disc;
            var mu = lambda - d;

            var r = Math.Sqrt(mu.Magnitude * mu.Magnitude + c.Magnitude * c.Magnitude);
            var cc = mu / r;
            var s = c / r;
            // G = [[conj(cc), s], [-s, cc]]
            var g00 = Complex.Conjugate(cc);
            var g01 = s;
            var g10 = -s;
            var g11 = cc;

            for (var j = m - 1; j < n; j++)
            {
                var x0 = T[(m - 1) * n + j];
                var x1 = T[m * n + j];
                T[(m - 1) * n + j] = g00 * x0 + g01 * x1;
                T[m * n + j] = g10 * x0 + g11 * x1;
            }

            // Right-multiply by G^H = [[cc, -conj(s)], [conj(s), conj(cc)]].
            var h00 = cc;
            var h01 = -Complex.Conjugate(s);
            var h10 = Complex.Conjugate(s);
            var h11 = Complex.Conjugate(cc);
            for (var i = 0; i <= m; i++)
            {
                var x0 = T[i * n + m - 1];
                var x1 = T[i * n + m];
                T[i * n + m - 1] = x0 * h00 + x1 * h10;
                T[i * n + m] = x0 * h01 + x1 * h11;
            }
            for (var i = 0; i < n; i++)
            {
                var x0 = Z[i * n + m - 1];
                var x1 = Z[i * n + m];
                Z[i * n + m - 1] = x0 * h00 + x1 * h10;
                Z[i * n + m] = x0 * h01 + x1 * h11;
            }
            T[m * n + m - 1] = Complex.Zero;
        }
        return (T, Z);
    }

    /// <summary>Complex Schur form of a real matrix: <c>A = Z T Z^H</c>, T upper triangular.</summary>
    public static (Complex[] T, Complex[] Z) ComplexSchur(NdArray a)
    {
        var n = a.Shape[0];
        var t = a.ToArray();
        var z = new double[n * n];
        RealSchur(t, z, n);
        return ToComplexSchur(t, z, n);
    }

    /// <summary>
    /// Eigenvectors of an upper-triangular complex matrix by back-substitution (the <c>trevc</c>
    /// idea), one per diagonal entry, as columns of a row-major n×n array.
    /// </summary>
    /// <remarks>
    /// A defective matrix makes the denominators <c>T[i,i] - λ</c> exactly zero; like LAPACK they are
    /// perturbed to a small multiple of eps·|λ| rather than divided by, which returns a vector close
    /// to the (single) true eigenvector instead of NaN.
    /// </remarks>
    public static Complex EigenvectorOfTriangular(ReadOnlySpan<Complex> T, int n, int k, Span<Complex> x)
    {
        x.Clear();
        var lambda = T[k * n + k];
        x[k] = Complex.One;
        var smin = Math.Max(Eps * (Math.Abs(lambda.Real) + Math.Abs(lambda.Imaginary)), 1e-300);
        for (var i = k - 1; i >= 0; i--)
        {
            var s = Complex.Zero;
            for (var j = i + 1; j <= k; j++) s += T[i * n + j] * x[j];
            var den = T[i * n + i] - lambda;
            if (Math.Abs(den.Real) + Math.Abs(den.Imaginary) < smin) den = smin;
            x[i] = -s / den;
            // Rescale to keep growth from overflowing on near-defective matrices.
            var mag = Complex.Abs(x[i]);
            if (mag > 1e150)
                for (var j = i; j <= k; j++) x[j] /= mag;
        }
        return lambda;
    }

    /// <summary>
    /// Parlett–Reinsch balancing (scaling only, powers of two), LAPACK <c>gebal</c> with
    /// <c>job='S'</c>. Returns D such that the balanced matrix is <c>D⁻¹ A D</c>.
    /// </summary>
    public static double[] Balance(double[] a, int n)
    {
        var scale = new double[n];
        Array.Fill(scale, 1.0);
        const double radix = 2.0;
        const double sqrdx = radix * radix;
        var done = false;
        var sweeps = 0;
        while (!done && sweeps++ < 100)
        {
            done = true;
            for (var i = 0; i < n; i++)
            {
                double r = 0, c = 0;
                for (var j = 0; j < n; j++)
                {
                    if (j == i) continue;
                    c += Math.Abs(a[j * n + i]);
                    r += Math.Abs(a[i * n + j]);
                }
                if (c == 0.0 || r == 0.0) continue;
                var g = r / radix;
                var f = 1.0;
                var s = c + r;
                while (c < g) { f *= radix; c *= sqrdx; }
                g = r * radix;
                while (c > g) { f /= radix; c /= sqrdx; }
                if ((c + r) / f < 0.95 * s)
                {
                    done = false;
                    g = 1.0 / f;
                    scale[i] *= f;
                    for (var j = 0; j < n; j++) a[i * n + j] *= g;
                    for (var j = 0; j < n; j++) a[j * n + i] *= f;
                }
            }
        }
        return scale;
    }
}
