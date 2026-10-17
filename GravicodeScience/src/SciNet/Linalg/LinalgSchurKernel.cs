using System.Numerics;
using System.Runtime.Intrinsics;
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
    /// <param name="z">Receives the orthogonal Schur vectors, or null to skip accumulating them.
    /// When null only the eigenvalues are wanted, and <paramref name="t"/> is left holding the
    /// diagonal blocks but not a valid Schur form (LAPACK's <c>wantt = false</c>).</param>
    /// <param name="n">Order.</param>
    public static (double[] Re, double[] Im) RealSchur(double[] t, double[]? z, int n)
    {
        var wr = new double[n];
        var wi = new double[n];
        if (n == 0) return (wr, wi);

        if (LinalgNativeLapack.ShouldUse(n))
        {
            // dgees: same standardised 2×2 blocks (dlanv2), same eigenvalue order convention.
            var info = LinalgNativeLapack.Gees(t, n, z, wr, wi);
            if (info != 0) throw new LinalgException($"Schur: LAPACK dgees failed (info = {info}).");
            return (wr, wi);
        }

        Hessenberg(t, z, n);
        Francis(t, z, n, wr, wi, wantT: z is not null);
        Clean(t, z, n, wr, wi);
        return (wr, wi);
    }

    /// <summary>Rows × reflectors below which a deferred update is applied on one thread.</summary>
    private const long DeferredParallelWork = 6_000;

    /// <summary>
    /// The hqr2 iteration on a Hessenberg matrix: the full Schur form when <paramref name="wantT"/>,
    /// otherwise only the active diagonal block (eigenvalues only).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each double-shift sweep applies a chain of 3×3 reflectors <c>P_k</c>, each from the left to
    /// rows k..k+2 and from the right to columns k..k+2. On a row-major buffer the right-hand
    /// application walks three columns down every row — a cache line touched per row per
    /// reflector — and the Schur vectors get the same treatment. That walk was most of the runtime
    /// once the matrix stopped fitting in L2.
    /// </para>
    /// <para>
    /// The bulge chase itself only reads the band around the diagonal, so the right-hand update
    /// is split: rows <c>i ≥ k − 1</c> get <c>P_k</c> at once (the chase needs them), rows above
    /// get every reflector of the sweep afterwards, each row walking left to right through its own
    /// contiguous memory, rows in parallel. Row i receives exactly the reflectors with
    /// <c>k ≥ i + 2</c> late, and nothing reads or writes row i between their original time and the
    /// end of the sweep, so the arithmetic — and the result, bit for bit — is unchanged. The Schur
    /// vectors are never read by the chase, so all of their updates are deferred the same way;
    /// keeping Z row-major (not transposed, as the previous version did) makes those row walks too.
    /// </para>
    /// </remarks>
    private static void Francis(double[] H, double[]? Z, int nn, double[] d, double[] e, bool wantT)
    {
        var n = nn - 1;
        const int low = 0;
        var exshift = 0.0;
        double p = 0, q = 0, r = 0, s = 0, z = 0;
        double w, x, y;

        var norm = 0.0;
        for (var i = 0; i < nn; i++)
            for (var j = Math.Max(i - 1, 0); j < nn; j++) norm += Math.Abs(H[i * nn + j]);

        // Reflector k of the current sweep: (x, y, z, q, r) and 0 = skipped, 2 = last (two rows), 3.
        var refl = new double[5 * nn];
        var kind = new byte[nn];

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
                    if (Z is not null)
                        for (var i = 0; i < nn; i++)
                        {
                            z = Z[i * nn + n - 1];
                            Z[i * nn + n - 1] = q * z + p * Z[i * nn + n];
                            Z[i * nn + n] = q * Z[i * nn + n] - p * z;
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

                // Only the active block [l, n] when the Schur form is not wanted.
                var colEnd = wantT ? nn : n + 1;
                var rowStart = wantT ? 0 : l;

                // Double QR step on rows l..n and columns m..n, in chunks of reflectors: within a
                // chunk the left updates go only as far right as the chase will read before the
                // chunk ends (column ke + 3); the columns beyond take the whole chunk afterwards,
                // in parallel stripes. Per element the arithmetic and its order are unchanged.
                for (var ks = m; ks <= n - 1; ks += SweepChunk)
                {
                var ke = Math.Min(n - 1, ks + SweepChunk - 1);
                var colLimit = Math.Min(colEnd, ke + 4);
                for (var k = ks; k <= ke; k++)
                {
                    kind[k] = 0;
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

                    kind[k] = notlast ? (byte)3 : (byte)2;
                    refl[5 * k] = x; refl[5 * k + 1] = y; refl[5 * k + 2] = z;
                    refl[5 * k + 3] = q; refl[5 * k + 4] = r;

                    LeftReflect(H, nn, k, k, colLimit, x, y, z, q, r, notlast);

                    // Right: rows the chase still reads now (i ≥ k − 1), the rest after the sweep.
                    var top = Math.Min(n, k + 3);
                    for (var i = Math.Max(rowStart, k - 1); i <= top; i++)
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
                }
                if (colLimit < colEnd) DeferredColumns(H, nn, ks, ke, colLimit, colEnd, refl, kind);
                }

                // Deferred right-hand updates: row i takes reflectors k ≥ max(m, i + 2).
                DeferredRows(H, nn, rowStart, n - 2, m, n - 1, 2, refl, kind);
                if (Z is not null) DeferredRows(Z, nn, 0, nn, m, n - 1, int.MinValue / 2, refl, kind);
            }
        }
    }

    /// <summary>
    /// Left application of reflector k to rows k..k+2 (k..k+1 when last), columns k..colEnd-1,
    /// vectorised over the columns with the same per-element arithmetic as the scalar loop.
    /// </summary>
    private static unsafe void LeftReflect(double[] H, int nn, int k, int colStart, int colEnd,
        double x, double y, double z, double q, double r, bool notlast)
    {
        fixed (double* hp = H)
        {
            var r0 = hp + (long)k * nn;
            var r1 = r0 + nn;
            var j = colStart;
            if (notlast)
            {
                var r2 = r1 + nn;
                var vq = Vector256.Create(q); var vr = Vector256.Create(r);
                var vx = Vector256.Create(x); var vy = Vector256.Create(y); var vz = Vector256.Create(z);
                for (; j <= colEnd - 4; j += 4)
                {
                    var h0 = Vector256.Load(r0 + j);
                    var h1 = Vector256.Load(r1 + j);
                    var h2 = Vector256.Load(r2 + j);
                    var pv = h0 + vq * h1 + vr * h2;
                    Vector256.Store(h2 - pv * vz, r2 + j);
                    Vector256.Store(h0 - pv * vx, r0 + j);
                    Vector256.Store(h1 - pv * vy, r1 + j);
                }
                for (; j < colEnd; j++)
                {
                    var pp = r0[j] + q * r1[j] + r * r2[j];
                    r2[j] -= pp * z;
                    r0[j] -= pp * x;
                    r1[j] -= pp * y;
                }
            }
            else
            {
                var vq = Vector256.Create(q); var vx = Vector256.Create(x); var vy = Vector256.Create(y);
                for (; j <= colEnd - 4; j += 4)
                {
                    var h0 = Vector256.Load(r0 + j);
                    var h1 = Vector256.Load(r1 + j);
                    var pv = h0 + vq * h1;
                    Vector256.Store(h0 - pv * vx, r0 + j);
                    Vector256.Store(h1 - pv * vy, r1 + j);
                }
                for (; j < colEnd; j++)
                {
                    var pp = r0[j] + q * r1[j];
                    r0[j] -= pp * x;
                    r1[j] -= pp * y;
                }
            }
        }
    }

    /// <summary>Reflectors per chunk of a sweep (see the chunk loop in <see cref="Francis"/>).</summary>
    private const int SweepChunk = 48;

    /// <summary>
    /// The left-hand updates of reflectors ks..ke on columns [c0, c1), which the chase does not
    /// read during the chunk: column stripes in parallel, each applying the reflectors in order.
    /// </summary>
    private static void DeferredColumns(double[] H, int nn, int ks, int ke, int c0, int c1, double[] refl, byte[] kind)
    {
        var work = (long)(c1 - c0) * (ke - ks + 1);
        if (work < 2 * DeferredParallelWork || Environment.ProcessorCount == 1)
        {
            Stripe(c0, c1);
            return;
        }
        var stripes = Math.Min(2 * Environment.ProcessorCount, Math.Max(1, (c1 - c0) / 32));
        var width = ((c1 - c0 + stripes - 1) / stripes + 3) / 4 * 4;
        stripes = (c1 - c0 + width - 1) / width;
        Parallel.For(0, stripes, t => Stripe(c0 + t * width, Math.Min(c1, c0 + (t + 1) * width)));

        void Stripe(int s0, int s1)
        {
            for (var k = ks; k <= ke; k++)
            {
                var kk = kind[k];
                if (kk == 0) continue;
                var o = 5 * k;
                LeftReflect(H, nn, k, s0, s1, refl[o], refl[o + 1], refl[o + 2], refl[o + 3], refl[o + 4], kk == 3);
            }
        }
    }

    /// <summary>
    /// Applies the sweep's reflectors from the right to rows [rowStart, rowEnd) of M: row i takes
    /// reflectors max(kFirst, i + offset) .. kLast in order, walking along its own memory.
    /// </summary>
    private static void DeferredRows(double[] M, int nn, int rowStart, int rowEnd, int kFirst, int kLast,
        int offset, double[] refl, byte[] kind)
    {
        if (rowEnd <= rowStart || kLast < kFirst) return;
        var work = (long)(rowEnd - rowStart) * (kLast - kFirst + 1);
        if (work < DeferredParallelWork || Environment.ProcessorCount == 1)
        {
            WalkRange(M, nn, rowStart, rowEnd, kFirst, kLast, offset, refl, kind);
            return;
        }
        var chunks = Math.Min(rowEnd - rowStart, 4 * Environment.ProcessorCount);
        var per = (rowEnd - rowStart + chunks - 1) / chunks;
        Parallel.For(0, chunks, c =>
        {
            var i0 = rowStart + c * per;
            var i1 = Math.Min(rowEnd, i0 + per);
            WalkRange(M, nn, i0, i1, kFirst, kLast, offset, refl, kind);
        });
    }

    /// <summary>
    /// Rows [i0, i1) four at a time: each row's walk is a dependent chain (a reflector reads the
    /// value the previous one just wrote), latency-bound at roughly 18 cycles per reflector on its
    /// own. Four independent rows in one loop let the core overlap the chains.
    /// </summary>
    private static void WalkRange(double[] M, int nn, int i0, int i1, int kFirst, int kLast, int offset,
        double[] refl, byte[] kind)
    {
        var i = i0;
        for (; i + 4 <= i1; i += 4)
        {
            // Rows start at different reflectors when offset applies: catch the first three up
            // to the fourth's start on their own, then walk all four together.
            var start = Math.Max(kFirst, i + 3 + offset);
            for (var r = 0; r < 3; r++)
                WalkRow(M, nn, i + r, Math.Max(kFirst, i + r + offset), Math.Min(kLast, start - 1), refl, kind);
            if (start <= kLast) WalkRows4(M, nn, i, start, kLast, refl, kind);
        }
        for (; i < i1; i++) WalkRow(M, nn, i, Math.Max(kFirst, i + offset), kLast, refl, kind);
    }

    private static unsafe void WalkRows4(double[] M, int nn, int i, int kFrom, int kTo, double[] refl, byte[] kind)
    {
        fixed (double* mp = M, rp = refl)
        {
            var r0 = mp + (long)i * nn; var r1 = r0 + nn; var r2 = r1 + nn; var r3 = r2 + nn;
            var k = kFrom;
            // Rolling window of columns k, k+1, k+2 per row, held in registers.
            double a0 = r0[k], a1 = r0[k + 1], b0 = r1[k], b1 = r1[k + 1];
            double c0 = r2[k], c1 = r2[k + 1], d0 = r3[k], d1 = r3[k + 1];
            double a2 = 0, b2 = 0, c2 = 0, d2 = 0;
            if (k + 2 < nn) { a2 = r0[k + 2]; b2 = r1[k + 2]; c2 = r2[k + 2]; d2 = r3[k + 2]; }
            for (; k <= kTo; k++)
            {
                var kk = kind[k];
                if (kk != 0)
                {
                    var co = rp + 5 * k;
                    double x = co[0], y = co[1], q = co[3];
                    if (kk == 3)
                    {
                        double z = co[2], rr = co[4];
                        var pa = x * a0 + y * a1; pa += z * a2; a2 -= pa * rr; a0 -= pa; a1 -= pa * q;
                        var pb = x * b0 + y * b1; pb += z * b2; b2 -= pb * rr; b0 -= pb; b1 -= pb * q;
                        var pc = x * c0 + y * c1; pc += z * c2; c2 -= pc * rr; c0 -= pc; c1 -= pc * q;
                        var pd = x * d0 + y * d1; pd += z * d2; d2 -= pd * rr; d0 -= pd; d1 -= pd * q;
                    }
                    else
                    {
                        var pa = x * a0 + y * a1; a0 -= pa; a1 -= pa * q;
                        var pb = x * b0 + y * b1; b0 -= pb; b1 -= pb * q;
                        var pc = x * c0 + y * c1; c0 -= pc; c1 -= pc * q;
                        var pd = x * d0 + y * d1; d0 -= pd; d1 -= pd * q;
                    }
                }
                r0[k] = a0; r1[k] = b0; r2[k] = c0; r3[k] = d0;
                a0 = a1; b0 = b1; c0 = c1; d0 = d1;
                a1 = a2; b1 = b2; c1 = c2; d1 = d2;
                if (k + 3 < nn) { a2 = r0[k + 3]; b2 = r1[k + 3]; c2 = r2[k + 3]; d2 = r3[k + 3]; }
            }
            // k = kTo + 1: flush the two columns still held (k ≤ n ≤ nn − 1 always exists).
            r0[k] = a0; r1[k] = b0; r2[k] = c0; r3[k] = d0;
            if (k + 1 < nn) { r0[k + 1] = a1; r1[k + 1] = b1; r2[k + 1] = c1; r3[k + 1] = d1; }
        }
    }

    private static unsafe void WalkRow(double[] M, int nn, int i, int kFrom, int kTo, double[] refl, byte[] kind)
    {
        fixed (double* mp = M, rp = refl)
        {
            var row = mp + (long)i * nn;
            for (var k = kFrom; k <= kTo; k++)
            {
                var kk = kind[k];
                if (kk == 0) continue;
                var c = rp + 5 * k;
                var p = c[0] * row[k] + c[1] * row[k + 1];
                if (kk == 3)
                {
                    p += c[2] * row[k + 2];
                    row[k + 2] -= p * c[4];
                }
                row[k] -= p;
                row[k + 1] -= p * c[3];
            }
        }
    }

    /// <summary>
    /// Zeroes everything below the quasi-triangle and standardises each complex 2×2 block with
    /// <c>dlanv2</c>, recomputing the eigenvalues from the final blocks.
    /// </summary>
    private static void Clean(double[] T, double[]? Z, int n, double[] wr, double[] wi)
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
            if (Z is not null)
                for (var i = 0; i < n; i++)
                {
                    var z1 = Z[i * n + k];
                    var z2 = Z[i * n + k + 1];
                    Z[i * n + k] = cs * z1 + sn * z2;
                    Z[i * n + k + 1] = -sn * z1 + cs * z2;
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
