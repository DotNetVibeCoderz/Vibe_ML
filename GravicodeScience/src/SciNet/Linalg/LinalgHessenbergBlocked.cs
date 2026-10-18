using System.Runtime.Intrinsics;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Blocked Householder reduction to upper Hessenberg form (LAPACK <c>dgehrd</c>/<c>dlahr2</c>) and
/// blocked formation of the orthogonal factor (<c>dorghr</c>), on row-major buffers.
/// </summary>
/// <remarks>
/// <para>
/// The unblocked reduction (<see cref="LinalgSchurKernel.Hessenberg"/>) updates the whole trailing
/// matrix with two rank-1 corrections per column, so every reflector streams the n² array through
/// memory twice more. Here <c>nb</c> reflectors are gathered first (<c>dlahr2</c>: the panel is
/// reduced with only the products the next reflector needs, and <c>Y = A V T</c> is built
/// alongside), then the trailing matrix takes them as two matrix products: from the right
/// <c>A ← A − Y Vᵀ</c>, from the left <c>A ← (I − V T Vᵀ)ᵀ A</c>. Only the <c>A v</c> product
/// inside the panel stays level 2 — it needs the reflector just computed — and that one runs over
/// contiguous rows, in parallel when large.
/// </para>
/// <para>
/// Q is built backwards, as <c>dorgqr</c> does: starting from the identity, the last panel's block
/// reflector is applied first, so each product only touches the trailing block that is not still
/// the identity. That is about 4n³/3 flops against the 2n³ of accumulating forwards.
/// </para>
/// <para>
/// The same reflectors as the unblocked version in exact arithmetic (the vectors are identical up
/// to rounding); H agrees to rounding, not bit for bit.
/// </para>
/// </remarks>
internal static unsafe class LinalgHessenbergBlocked
{
    /// <summary>Panel width (LAPACK's <c>ILAENV(1, 'DGEHRD')</c> is 32 too).</summary>
    internal const int BlockSize = 32;

    /// <summary>
    /// Trailing columns reduced unblocked (LAPACK's crossover <c>NX</c>). Below this many columns
    /// in total the whole reduction is unblocked.
    /// </summary>
    internal const int Crossover = 64;

    /// <summary>
    /// Reduces <paramref name="a"/> (row-major n×n) to upper Hessenberg form in place, zeroing below
    /// the subdiagonal, and writes <c>Q</c> with <c>A = Q H Qᵀ</c> into <paramref name="q"/> when it
    /// is not null.
    /// </summary>
    public static void Reduce(double[] a, double[]? q, int n)
    {
        if (n <= 2)
        {
            if (q is not null)
            {
                Array.Clear(q);
                for (var i = 0; i < n; i++) q[i * n + i] = 1.0;
            }
            return;
        }
        var tau = new double[n];
        // Panels: start columns (1-based) and their T factors, kept for building Q.
        var panelStarts = new List<int>();
        var panelT = new List<double[]>();
        var nb = BlockSize;
        var nh = n; // ilo = 1, ihi = n
        fixed (double* pa = a, pt = tau)
        {
            var ihi = n;
            var i = 1;
            if (nb < nh && Crossover < nh)
            {
                var y = new double[n * nb];
                var vd = new double[n * nb];
                var w = new double[nb * n];
                fixed (double* py = y, pv = vd, pw = w)
                {
                    for (i = 1; i <= ihi - 1 - Crossover; i += nb)
                    {
                        var ib = Math.Min(nb, ihi - i);
                        var t = new double[ib * ib];
                        fixed (double* ptt = t)
                        {
                            Lahr2(ihi, i, ib, pa + (i - 1), n, pt + (i - 1), ptt, ib, py, ib);
                            Trailing(pa, n, ihi, i, ib, py, ib, ptt, pv, pw);
                        }
                        panelStarts.Add(i);
                        panelT.Add(t);
                    }
                }
            }
            // Unblocked tail: dgehd2 on columns i .. ihi-1.
            Gehd2(n, i, ihi, pa, n, pt);

            if (q is not null)
                fixed (double* pq = q)
                    FormQ(pa, n, pt, i, panelStarts, panelT, pq);

            // Clear the reflectors below the subdiagonal.
            for (var r = 2; r < n; r++)
                new Span<double>(pa + (long)r * n, r - 1).Clear();
        }
    }

    // ------------------------------------------------------------------ panel (dlahr2)

    /// <summary>
    /// LAPACK <c>dlahr2</c> with row-major storage. <paramref name="A"/> points at element
    /// (1, K) of the full matrix, i.e. local column 1 is the panel's first column; rows are
    /// global. Y is N×NB (row stride <paramref name="ldy"/>), T is NB×NB.
    /// </summary>
    private static void Lahr2(int N, int K, int NB, double* A, int lda, double* tau, double* T, int ldt, double* Y, int ldy)
    {
        if (N <= 1) return;
        double ei = 0;
        for (var I = 1; I <= NB; I++)
        {
            if (I > 1)
            {
                // Update A(K+1:N, I): A(:,I) -= Y(K+1:N, 1:I-1) * A(K+I-1, 1:I-1)^T
                Gemv(false, N - K, I - 1, -1.0, P(Y, ldy, K + 1, 1), ldy, P(A, lda, K + I - 1, 1), 1, 1.0, P(A, lda, K + 1, I), lda);

                // Apply I - V T^T V^T from the left, using the last column of T as workspace w.
                //   b1 = A(K+1:K+I-1, I), b2 = A(K+I:N, I)
                //   w := V1^T b1
                Copy(I - 1, P(A, lda, K + 1, I), lda, P(T, ldt, 1, NB), ldt);
                Trmv(lower: true, trans: true, unit: true, I - 1, P(A, lda, K + 1, 1), lda, P(T, ldt, 1, NB), ldt);
                //   w += V2^T b2
                Gemv(true, N - K - I + 1, I - 1, 1.0, P(A, lda, K + I, 1), lda, P(A, lda, K + I, I), lda, 1.0, P(T, ldt, 1, NB), ldt);
                //   w := T^T w
                Trmv(lower: false, trans: true, unit: false, I - 1, T, ldt, P(T, ldt, 1, NB), ldt);
                //   b2 -= V2 w
                Gemv(false, N - K - I + 1, I - 1, -1.0, P(A, lda, K + I, 1), lda, P(T, ldt, 1, NB), ldt, 1.0, P(A, lda, K + I, I), lda);
                //   b1 -= V1 w
                Trmv(lower: true, trans: false, unit: true, I - 1, P(A, lda, K + 1, 1), lda, P(T, ldt, 1, NB), ldt);
                Axpy(I - 1, -1.0, P(T, ldt, 1, NB), ldt, P(A, lda, K + 1, I), lda);

                *P(A, lda, K + I - 1, I - 1) = ei;
            }

            // Reflector H(I) annihilating A(K+I+1:N, I).
            var len = N - K - I + 1;
            ref var alpha = ref *P(A, lda, K + I, I);
            LinalgMultishiftQr.Larfg(len, ref alpha, P(A, lda, Math.Min(K + I + 1, N), I), lda, out tau[I - 1]);
            ei = *P(A, lda, K + I, I);
            *P(A, lda, K + I, I) = 1.0;

            // Y(K+1:N, I) = A(K+1:N, I+1:N-K+1... ) v   (the level-2 product with the trailing matrix)
            Gemv(false, N - K, N - K - I + 1, 1.0, P(A, lda, K + 1, I + 1), lda, P(A, lda, K + I, I), lda, 0.0, P(Y, ldy, K + 1, I), ldy);
            // T(1:I-1, I) = A(K+I:N, 1:I-1)^T v
            Gemv(true, N - K - I + 1, I - 1, 1.0, P(A, lda, K + I, 1), lda, P(A, lda, K + I, I), lda, 0.0, P(T, ldt, 1, I), ldt);
            // Y(K+1:N, I) -= Y(K+1:N, 1:I-1) T(1:I-1, I)
            Gemv(false, N - K, I - 1, -1.0, P(Y, ldy, K + 1, 1), ldy, P(T, ldt, 1, I), ldt, 1.0, P(Y, ldy, K + 1, I), ldy);
            Scal(N - K, tau[I - 1], P(Y, ldy, K + 1, I), ldy);

            // T(1:I, I)
            Scal(I - 1, -tau[I - 1], P(T, ldt, 1, I), ldt);
            Trmv(lower: false, trans: false, unit: false, I - 1, T, ldt, P(T, ldt, 1, I), ldt);
            *P(T, ldt, I, I) = tau[I - 1];
        }
        *P(A, lda, K + NB, NB) = ei;
        // Y(1:K, 1:NB) is formed by the caller with matrix products (see Trailing).
    }

    /// <summary>
    /// Everything <c>dgehrd</c> does after <c>dlahr2</c> for the panel at column i (1-based) of
    /// width ib: the top rows of Y, the right update of all rows and the left update of the
    /// trailing columns, as matrix products against an explicit copy of V.
    /// </summary>
    private static void Trailing(double* a, int n, int ihi, int i, int ib, double* y, int ldy, double* t, double* vd, double* w)
    {
        // Explicit V: rows i+1 .. ihi (m of them), columns 1..ib; reflector c has its unit at
        // global row i + c and is zero above it. vd is row-major m × ib.
        var m = ihi - i;
        for (var r = 0; r < m; r++)
        {
            var grow = i + 1 + r; // global row (1-based)
            var vr = vd + (long)r * ib;
            for (var c = 1; c <= ib; c++)
            {
                var unitRow = i + c;
                // Column c of the panel is global column i + c - 1.
                vr[c - 1] = grow < unitRow ? 0.0 : grow == unitRow ? 1.0 : *P(a, n, grow, i + c - 1);
            }
        }

        // Y(1:i, :) = A(1:i, i+1:ihi) V T.
        if (i > 0)
        {
            Gemm(false, false, i, ib, m, 1.0, P(a, n, 1, i + 1), n, vd, ib, 0.0, y, ldy);
            TrmmRightUpper(i, ib, t, ib, y, ldy, w);
        }

        // Right: rows 1..i, columns i+1..ihi:      A -= Y(1:i, :) V^T
        //        rows i+1..ihi, columns i+ib..ihi: A -= Y(i+1:ihi, :) V(ib.., :)^T
        // (rows below i in the panel columns were updated inside the panel).
        if (i > 0)
            Gemm(false, true, i, m, ib, -1.0, y, ldy, vd, ib, 1.0, P(a, n, 1, i + 1), n);
        var trailCols = ihi - i - ib + 1;
        if (trailCols > 0)
            Gemm(false, true, m, trailCols, ib, -1.0, P(y, ldy, i + 1, 1), ldy, vd + (long)(ib - 1) * ib, ib, 1.0, P(a, n, i + 1, i + ib), n);

        // Left: C = A(i+1:ihi, i+ib:n) ← (I − V T Vᵀ)ᵀ C = C − V Tᵀ (Vᵀ C).
        var nc = n - i - ib + 1;
        if (nc > 0)
        {
            var c = P(a, n, i + 1, i + ib);
            Gemm(true, false, ib, nc, m, 1.0, vd, ib, c, n, 0.0, w, nc);
            TrmmLeftUpperTrans(ib, nc, t, ib, w, nc);
            Gemm(false, false, m, nc, ib, -1.0, vd, ib, w, nc, 1.0, c, n);
        }
    }

    /// <summary>B (m×k) ← B T with T upper triangular k×k, through scratch <paramref name="w"/>.</summary>
    private static void TrmmRightUpper(int m, int k, double* t, int ldt, double* b, int ldb, double* w)
    {
        // Row by row: b_r ← b_r T, computed right to left so each entry reads unchanged values.
        for (var r = 0; r < m; r++)
        {
            var br = b + (long)r * ldb;
            for (var j = k - 1; j >= 0; j--)
            {
                var s = 0.0;
                for (var p = 0; p <= j; p++) s += br[p] * t[p * ldt + j];
                br[j] = s;
            }
        }
    }

    /// <summary>W (k×nc) ← Tᵀ W with T upper triangular k×k.</summary>
    private static void TrmmLeftUpperTrans(int k, int nc, double* t, int ldt, double* w, int ldw)
    {
        // Row j of Tᵀ W is sum_{p ≤ j} T[p, j] W_p: go bottom up so W_p (p < j) is still original.
        for (var j = k - 1; j >= 0; j--)
        {
            var wj = new Span<double>(w + (long)j * ldw, nc);
            var tjj = t[j * ldt + j];
            for (var q = 0; q < nc; q++) wj[q] *= tjj;
            for (var p = 0; p < j; p++)
            {
                var tpj = t[p * ldt + j];
                if (tpj != 0.0) LinalgBlocked.Axpy(tpj, w + (long)p * ldw, w + (long)j * ldw, nc);
            }
        }
    }

    // ------------------------------------------------------------------ unblocked (dgehd2)

    /// <summary>
    /// LAPACK <c>dgehd2</c> on the row-major n×n matrix at <paramref name="a"/> (row stride
    /// <paramref name="lda"/>), columns ilo..ihi-1 (1-based). Reflector vectors stay below the
    /// subdiagonal; <c>tau[i-1]</c> receives reflector i's scalar.
    /// </summary>
    internal static void Gehd2(int n, int ilo, int ihi, double* a, int lda, double* tau)
    {
        var work = new double[n];
        fixed (double* pw = work)
            for (var i = ilo; i <= ihi - 1; i++)
            {
                ref var alpha = ref *P(a, lda, i + 1, i);
                LinalgMultishiftQr.Larfg(ihi - i, ref alpha, P(a, lda, Math.Min(i + 2, n), i), lda, out tau[i - 1]);
                var v = P(a, lda, i + 1, i);
                var t = tau[i - 1];
                // Right: A(1:ihi, i+1:ihi) ← A H;  left: A(i+1:ihi, i+1:n) ← H A.
                LinalgMultishiftQr.ApplyRight(ihi, ihi - i, v, lda, t, P(a, lda, 1, i + 1), lda, pw);
                LinalgMultishiftQr.ApplyLeft(ihi - i, n - i, v, lda, t, P(a, lda, i + 1, i + 1), lda, pw);
            }
    }

    // ------------------------------------------------------------------ Q (dorghr)

    /// <summary>
    /// Builds Q = H(1) … H(n−2) from the reflectors stored below the subdiagonal of
    /// <paramref name="a"/>: unblocked tail reflectors first (from <paramref name="tailStart"/>),
    /// then each panel's block reflector, last panel first.
    /// </summary>
    private static void FormQ(double* a, int n, double* tau, int tailStart, List<int> panelStarts, List<double[]> panelT, double* q)
    {
        new Span<double>(q, n * n).Clear();
        for (var i = 0; i < n; i++) q[(long)i * n + i] = 1.0;

        var work = new double[n];
        fixed (double* pw = work)
        {
            // Q ← H(j) Q for the tail, j = n-1 down to tailStart (reflector j acts on rows j+1..n;
            // Q is still the identity outside rows/columns ≥ j+1, so only that block changes).
            for (var j = n - 1; j >= tailStart; j--)
            {
                var t = tau[j - 1];
                if (t == 0.0) continue;
                LinalgMultishiftQr.ApplyLeft(n - j, n - j, P(a, n, j + 1, j), n, t, P(q, n, j + 1, j + 1), n, pw);
            }
        }

        if (panelStarts.Count == 0) return;
        var nb = BlockSize;
        var vd = new double[n * nb];
        var w = new double[nb * n];
        fixed (double* pv = vd, pw = w)
            for (var b = panelStarts.Count - 1; b >= 0; b--)
            {
                var i = panelStarts[b];
                var tArr = panelT[b];
                var ib = (int)Math.Round(Math.Sqrt(tArr.Length));
                var m = n - i; // rows i+1..n
                for (var r = 0; r < m; r++)
                {
                    var grow = i + 1 + r;
                    var vr = pv + (long)r * ib;
                    for (var c = 1; c <= ib; c++)
                    {
                        var unitRow = i + c;
                        vr[c - 1] = grow < unitRow ? 0.0 : grow == unitRow ? 1.0 : *P(a, n, grow, i + c - 1);
                    }
                }
                // Q(i+1:n, i+1:n) ← (I − V T Vᵀ) Q(...).
                var qs = P(q, n, i + 1, i + 1);
                fixed (double* pt = tArr)
                {
                    Gemm(true, false, ib, m, m, 1.0, pv, ib, qs, n, 0.0, pw, m);
                    TrmmLeftUpper(ib, m, pt, ib, pw, m);
                    Gemm(false, false, m, m, ib, -1.0, pv, ib, pw, m, 1.0, qs, n);
                }
            }
    }

    /// <summary>W (k×nc) ← T W with T upper triangular k×k.</summary>
    private static void TrmmLeftUpper(int k, int nc, double* t, int ldt, double* w, int ldw)
    {
        // Row j of T W is sum_{p ≥ j} T[j, p] W_p: go top down so W_p (p > j) is still original.
        for (var j = 0; j < k; j++)
        {
            var wj = w + (long)j * ldw;
            var tjj = t[j * ldt + j];
            for (var q = 0; q < nc; q++) wj[q] *= tjj;
            for (var p = j + 1; p < k; p++)
            {
                var tjp = t[j * ldt + p];
                if (tjp != 0.0) LinalgBlocked.Axpy(tjp, w + (long)p * ldw, wj, nc);
            }
        }
    }

    // ------------------------------------------------------------------ level-1/2 helpers

    /// <summary>Address of element (i, j), 1-based, of the row-major matrix at p.</summary>
    private static double* P(double* p, int ld, int i, int j) => p + (long)(i - 1) * ld + (j - 1);

    private static void Gemm(bool ta, bool tb, int m, int n, int k, double alpha, double* a, int lda, double* b, int ldb, double beta, double* c, int ldc)
        => LinalgMultishiftQr.Gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

    private static void Copy(int n, double* x, int incx, double* y, int incy)
    {
        for (var i = 0; i < n; i++) y[(long)i * incy] = x[(long)i * incx];
    }

    private static void Scal(int n, double alpha, double* x, int incx)
    {
        for (var i = 0; i < n; i++) x[(long)i * incx] *= alpha;
    }

    private static void Axpy(int n, double alpha, double* x, int incx, double* y, int incy)
    {
        for (var i = 0; i < n; i++) y[(long)i * incy] += alpha * x[(long)i * incx];
    }

    /// <summary>Work (rows × columns) above which the non-transposed product runs rows in parallel.</summary>
    private const long GemvParallelWork = 40_000;

    /// <summary>
    /// <c>y ← alpha·op(A)·x + beta·y</c>, A row-major m×n. Without transpose each entry of y is a dot
    /// product over a contiguous row; with transpose y accumulates row axpys.
    /// </summary>
    private static void Gemv(bool trans, int m, int n, double alpha, double* a, int lda, double* x, int incx, double beta, double* y, int incy)
    {
        if (m <= 0 || n <= 0)
        {
            var len = trans ? n : m;
            if (beta != 1.0) for (var i = 0; i < len; i++) y[(long)i * incy] = beta == 0.0 ? 0.0 : beta * y[(long)i * incy];
            return;
        }
        if (!trans)
        {
            // Gather x contiguous once.
            var xs = new double[n];
            for (var j = 0; j < n; j++) xs[j] = x[(long)j * incx];
            if ((long)m * n >= GemvParallelWork && Environment.ProcessorCount > 1)
            {
                var aS = (nint)a; var yS = (nint)y;
                var chunks = Math.Min(m, 2 * Environment.ProcessorCount);
                var per = (m + chunks - 1) / chunks;
                Parallel.For(0, chunks, c =>
                {
                    var r0 = c * per;
                    var r1 = Math.Min(m, r0 + per);
                    fixed (double* px = xs)
                        for (var r = r0; r < r1; r++)
                        {
                            var s = Dot((double*)aS + (long)r * lda, px, n);
                            var yp = (double*)yS + (long)r * incy;
                            *yp = beta == 0.0 ? alpha * s : beta * *yp + alpha * s;
                        }
                });
                return;
            }
            fixed (double* px = xs)
                for (var r = 0; r < m; r++)
                {
                    var s = Dot(a + (long)r * lda, px, n);
                    var yp = y + (long)r * incy;
                    *yp = beta == 0.0 ? alpha * s : beta * *yp + alpha * s;
                }
        }
        else
        {
            var acc = new double[n];
            fixed (double* pacc = acc)
            {
                for (var r = 0; r < m; r++)
                {
                    var xr = x[(long)r * incx];
                    if (xr != 0.0) LinalgBlocked.Axpy(xr, a + (long)r * lda, pacc, n);
                }
                for (var j = 0; j < n; j++)
                {
                    var yp = y + (long)j * incy;
                    *yp = beta == 0.0 ? alpha * pacc[j] : beta * *yp + alpha * pacc[j];
                }
            }
        }
    }

    private static double Dot(double* a, double* b, int n)
    {
        var i = 0;
        var acc0 = Vector256<double>.Zero;
        var acc1 = Vector256<double>.Zero;
        for (; i <= n - 8; i += 8)
        {
            acc0 += Vector256.Load(a + i) * Vector256.Load(b + i);
            acc1 += Vector256.Load(a + i + 4) * Vector256.Load(b + i + 4);
        }
        var s = Vector256.Sum(acc0 + acc1);
        for (; i < n; i++) s += a[i] * b[i];
        return s;
    }

    /// <summary><c>x ← op(A) x</c> for triangular A (n×n, row-major at <paramref name="a"/>).</summary>
    private static void Trmv(bool lower, bool trans, bool unit, int n, double* a, int lda, double* x, int incx)
    {
        if (n <= 0) return;
        double A(int i, int j) => a[(long)i * lda + j];
        ref double X(int i) => ref x[(long)i * incx];
        // Effective matrix M = op(A); M is upper when (upper, no trans) or (lower, trans).
        var effUpper = lower == trans;
        if (effUpper)
        {
            // x_i = sum_{j ≥ i} M(i,j) x_j: ascending i reads only unchanged x_j.
            for (var i = 0; i < n; i++)
            {
                var s = unit ? X(i) : (trans ? A(i, i) : A(i, i)) * X(i);
                for (var j = i + 1; j < n; j++) s += (trans ? A(j, i) : A(i, j)) * X(j);
                X(i) = s;
            }
        }
        else
        {
            // x_i = sum_{j ≤ i} M(i,j) x_j: descending i.
            for (var i = n - 1; i >= 0; i--)
            {
                var s = unit ? X(i) : A(i, i) * X(i);
                for (var j = 0; j < i; j++) s += (trans ? A(j, i) : A(i, j)) * X(j);
                X(i) = s;
            }
        }
    }
}
