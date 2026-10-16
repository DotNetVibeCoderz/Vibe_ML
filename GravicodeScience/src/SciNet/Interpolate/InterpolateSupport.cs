using System.Runtime.CompilerServices;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>How a piecewise interpolant behaves outside its base interval (scipy's
/// <c>extrapolate</c> argument: <c>True</c>, <c>False</c> or <c>'periodic'</c>).</summary>
public enum InterpolateExtrapolate
{
    /// <summary>Out-of-range points evaluate to NaN (scipy <c>extrapolate=False</c>).</summary>
    None,
    /// <summary>The first and last pieces are continued (scipy <c>extrapolate=True</c>).</summary>
    Extrapolate,
    /// <summary>The argument is wrapped into the base interval (scipy <c>'periodic'</c>).</summary>
    Periodic,
}

/// <summary>Shared kernels of the Interpolate module: searches, banded and dense solves,
/// B-spline basis evaluation and the axis bookkeeping that maps scipy's <c>axis</c> argument
/// onto a contiguous (n, trailing) buffer.</summary>
internal static class InterpolateSupport
{
    // ================================================================ validation

    public static double[] CopyFinite(ReadOnlySpan<double> v, string name)
    {
        var a = v.ToArray();
        foreach (var e in a)
            if (!double.IsFinite(e)) throw new ArgumentException($"`{name}` must contain only finite values.", name);
        return a;
    }

    public static void RequireStrictlyIncreasing(ReadOnlySpan<double> x, string name)
    {
        for (var i = 1; i < x.Length; i++)
            if (!(x[i] > x[i - 1])) throw new ArgumentException($"`{name}` must be a strictly increasing sequence.", name);
    }

    public static void RequireNonDecreasing(ReadOnlySpan<double> x, string name)
    {
        for (var i = 1; i < x.Length; i++)
            if (!(x[i] >= x[i - 1])) throw new ArgumentException($"`{name}` must be non-decreasing.", name);
    }

    public static int Product(ReadOnlySpan<int> s)
    {
        var p = 1;
        foreach (var e in s) p *= e;
        return p;
    }

    // ================================================================ searching

    /// <summary>numpy <c>searchsorted(a, v, side='left')</c>: the first i with a[i] &gt;= v.</summary>
    public static int SearchLeft(ReadOnlySpan<double> a, double v)
    {
        int lo = 0, hi = a.Length;
        while (lo < hi)
        {
            var mid = (lo + hi) >>> 1;
            if (a[mid] < v) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    /// <summary>numpy <c>searchsorted(a, v, side='right')</c>: the first i with a[i] &gt; v.</summary>
    public static int SearchRight(ReadOnlySpan<double> a, double v)
    {
        int lo = 0, hi = a.Length;
        while (lo < hi)
        {
            var mid = (lo + hi) >>> 1;
            if (a[mid] <= v) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    /// <summary>
    /// The interval i with x[i] &lt;= v &lt; x[i+1] of an ascending breakpoint vector (scipy's
    /// <c>find_interval_ascending</c>): v == x[last] belongs to the last interval, points outside
    /// clamp to the end intervals when <paramref name="extrapolate"/> and give -1 otherwise, as
    /// does NaN. <paramref name="hint"/> is the previous answer; sorted queries then cost O(1).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int FindIntervalAscending(ReadOnlySpan<double> x, double v, int hint, bool extrapolate)
    {
        var nx = x.Length;
        double a = x[0], b = x[nx - 1];
        if (!(v >= a && v <= b))
        {
            if (double.IsNaN(v) || !extrapolate) return -1;
            return v < a ? 0 : nx - 2;
        }
        if (v == b) return nx - 2;
        if ((uint)hint < (uint)(nx - 1))
        {
            if (x[hint] <= v)
            {
                if (v < x[hint + 1]) return hint;
                if (hint + 2 < nx && v < x[hint + 2] && x[hint + 1] <= v) return hint + 1;
            }
        }
        // Largest i with x[i] <= v; v < b so i <= nx - 2.
        return SearchRight(x, v) - 1;
    }

    /// <summary>The descending-breakpoint analogue of <see cref="FindIntervalAscending"/>:
    /// x[i] &gt;= v &gt; x[i+1], with v == x[last] in the last interval.</summary>
    public static int FindIntervalDescending(ReadOnlySpan<double> x, double v, int hint, bool extrapolate)
    {
        var nx = x.Length;
        double a = x[0], b = x[nx - 1];
        if (!(v <= a && v >= b))
        {
            if (double.IsNaN(v) || !extrapolate) return -1;
            return v > a ? 0 : nx - 2;
        }
        if (v == b) return nx - 2;
        if ((uint)hint < (uint)(nx - 1) && x[hint] >= v && v > x[hint + 1]) return hint;
        // Largest i with x[i] >= v.
        int lo = 0, hi = nx;
        while (lo < hi)
        {
            var mid = (lo + hi) >>> 1;
            if (x[mid] >= v) lo = mid + 1; else hi = mid;
        }
        return Math.Min(lo - 1, nx - 2);
    }

    // ================================================================ B-spline basis

    /// <summary>
    /// scipy's <c>_dierckx::find_interval</c>: the l in [k, n-1] with t[l] &lt;= x &lt; t[l+1]
    /// (x == t[n] belongs to the last interval); -1 for NaN or, without extrapolation, for x
    /// outside [t[k], t[n]].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int FindSplineInterval(ReadOnlySpan<double> t, int k, double x, int prevL, bool extrapolate)
    {
        var n = t.Length - k - 1;
        double tb = t[k], te = t[n];
        if (double.IsNaN(x)) return -1;
        if ((x < tb || x > te) && !extrapolate) return -1;
        var l = k < prevL && prevL < n ? prevL : k;
        while (x < t[l] && l != k) l--;
        l++;
        while (l != n && x >= t[l]) l++;
        return l - 1;
    }

    /// <summary>
    /// The k+1 B-splines (or their <paramref name="nu"/>-th derivatives) that are non-zero on
    /// interval l, written to <c>h[0..k]</c>; <paramref name="work"/> needs k+1 slots. A port of
    /// scipy's <c>_deBoor_D</c>: Cox–de Boor up to order k-nu, then nu differentiation sweeps.
    /// </summary>
    public static void DeBoorD(ReadOnlySpan<double> t, double x, int k, int l, int nu, Span<double> h, Span<double> work)
    {
        h[0] = 1.0;
        for (var j = 1; j <= k - nu; j++)
        {
            h[..j].CopyTo(work);
            h[0] = 0.0;
            for (var n = 1; n <= j; n++)
            {
                var ind = l + n;
                double xb = t[ind], xa = t[ind - j];
                if (xb == xa) { h[n] = 0.0; continue; }
                var w = work[n - 1] / (xb - xa);
                h[n - 1] += w * (xb - x);
                h[n] = w * (x - xa);
            }
        }
        for (var j = k - nu + 1; j <= k; j++)
        {
            h[..j].CopyTo(work);
            h[0] = 0.0;
            for (var n = 1; n <= j; n++)
            {
                var ind = l + n;
                double xb = t[ind], xa = t[ind - j];
                if (xb == xa) { h[n] = 0.0; continue; }
                var w = j * work[n - 1] / (xb - xa);
                h[n - 1] -= w;
                h[n] = w;
            }
        }
    }

    /// <summary>FITPACK's <c>fpbspl</c> (0-based): the k+1 non-zero B-splines at x on interval
    /// l (t[l] &lt;= x &lt; t[l+1]). Kept separate from <see cref="DeBoorD"/> because the FITPACK
    /// port reproduces Dierckx's operation order.</summary>
    public static void Fpbspl(ReadOnlySpan<double> t, int k, double x, int l, Span<double> h)
    {
        Span<double> hh = stackalloc double[20];
        h[0] = 1.0;
        for (var j = 1; j <= k; j++)
        {
            for (var i = 0; i < j; i++) hh[i] = h[i];
            h[0] = 0.0;
            for (var i = 1; i <= j; i++)
            {
                // Fortran: li = l+i, lj = li-j with l 1-based; here l is 0-based so t index
                // li-1 = l+i (0-based), lj-1 = l+i-j.
                var li = l + i;
                var lj = li - j;
                if (t[li] == t[lj]) { h[i] = 0.0; continue; }
                var f = hh[i - 1] / (t[li] - t[lj]);
                h[i - 1] += f * (t[li] - x);
                h[i] = f * (x - t[lj]);
            }
        }
    }

    /// <summary>Evaluates a spline (or its nu-th derivative; negative nu is not allowed) with
    /// coefficient columns c (n, nt) at one point, writing nt values.</summary>
    public static void EvaluateSplinePoint(ReadOnlySpan<double> t, ReadOnlySpan<double> c, int nt, int k, double x, int nu,
        bool extrapolate, ref int hint, Span<double> output, Span<double> h, Span<double> work)
    {
        var l = FindSplineInterval(t, k, x, hint, extrapolate);
        if (l < 0)
        {
            output[..nt].Fill(double.NaN);
            return;
        }
        hint = l;
        DeBoorD(t, x, k, l, nu, h, work);
        if (nt == 1)
        {
            var s = 0.0;
            for (var a = 0; a <= k; a++) s += c[l + a - k] * h[a];
            output[0] = s;
            return;
        }
        output[..nt].Clear();
        for (var a = 0; a <= k; a++)
        {
            var row = c.Slice((l + a - k) * nt, nt);
            var ha = h[a];
            for (var j = 0; j < nt; j++) output[j] += row[j] * ha;
        }
    }

    // ================================================================ banded LU

    /// <summary>
    /// LU factorisation with partial pivoting of a band matrix with kl sub- and ku
    /// super-diagonals (the algorithm of LAPACK's <c>dgbtrf</c>, without blocking). Row i stores
    /// columns i-kl .. i+ku+kl: the extra kl columns hold the fill-in that row swaps create.
    /// </summary>
    internal sealed class BandedLu
    {
        private readonly double[] _a;
        private readonly int[] _piv;
        private readonly int _n, _kl, _ku, _w;

        public BandedLu(int n, int kl, int ku)
        {
            _n = n; _kl = kl; _ku = ku; _w = 2 * kl + ku + 1;
            _a = new double[n * _w];
            _piv = new int[n];
        }

        public int N => _n;

        /// <summary>Element (i, j); |j - i| must lie inside the band.</summary>
        public double this[int i, int j]
        {
            get => _a[i * _w + (j - i + _kl)];
            set => _a[i * _w + (j - i + _kl)] = value;
        }

        public void Add(int i, int j, double v) => _a[i * _w + (j - i + _kl)] += v;

        /// <summary>Factorises in place; throws when a pivot is exactly zero (singular).</summary>
        public void Factor()
        {
            int n = _n, kl = _kl, w = _w, ku = _ku;
            var a = _a;
            for (var c = 0; c < n; c++)
            {
                var rmax = Math.Min(n - 1, c + kl);
                var p = c;
                var best = Math.Abs(a[c * w + kl]);
                for (var r = c + 1; r <= rmax; r++)
                {
                    var v = Math.Abs(a[r * w + (c - r + kl)]);
                    if (v > best) { best = v; p = r; }
                }
                _piv[c] = p;
                if (best == 0) throw new InvalidOperationException("Banded matrix is singular.");
                var jmax = Math.Min(n - 1, c + ku + kl);
                if (p != c)
                {
                    for (var j = c; j <= jmax; j++)
                        (a[c * w + (j - c + kl)], a[p * w + (j - p + kl)]) = (a[p * w + (j - p + kl)], a[c * w + (j - c + kl)]);
                }
                var piv = a[c * w + kl];
                for (var r = c + 1; r <= rmax; r++)
                {
                    var ir = r * w + (c - r + kl);
                    var f = a[ir] / piv;
                    a[ir] = f;
                    if (f == 0) continue;
                    var baseR = r * w - r + kl;
                    var baseC = c * w - c + kl;
                    for (var j = c + 1; j <= jmax; j++) a[baseR + j] -= f * a[baseC + j];
                }
            }
        }

        /// <summary>Solves A X = B in place for B of shape (n, nrhs), row-major.</summary>
        public void Solve(Span<double> b, int nrhs)
        {
            int n = _n, kl = _kl, w = _w, ku = _ku;
            var a = _a;
            // Forward: apply the interchanges and L as they were generated.
            for (var c = 0; c < n; c++)
            {
                var p = _piv[c];
                if (p != c)
                    for (var q = 0; q < nrhs; q++) (b[c * nrhs + q], b[p * nrhs + q]) = (b[p * nrhs + q], b[c * nrhs + q]);
                var rmax = Math.Min(n - 1, c + kl);
                for (var r = c + 1; r <= rmax; r++)
                {
                    var f = a[r * w + (c - r + kl)];
                    if (f == 0) continue;
                    for (var q = 0; q < nrhs; q++) b[r * nrhs + q] -= f * b[c * nrhs + q];
                }
            }
            // Backward with U (bandwidth ku + kl).
            for (var i = n - 1; i >= 0; i--)
            {
                var jmax = Math.Min(n - 1, i + ku + kl);
                var baseI = i * w - i + kl;
                for (var q = 0; q < nrhs; q++)
                {
                    var s = b[i * nrhs + q];
                    for (var j = i + 1; j <= jmax; j++) s -= a[baseI + j] * b[j * nrhs + q];
                    b[i * nrhs + q] = s / a[baseI + i];
                }
            }
        }
    }

    // ================================================================ dense LU

    /// <summary>In-place LU with partial pivoting of a row-major n×n matrix.</summary>
    public static void LuFactor(double[] a, int n, int[] piv)
    {
        for (var c = 0; c < n; c++)
        {
            var p = c;
            var best = Math.Abs(a[c * n + c]);
            for (var r = c + 1; r < n; r++)
            {
                var v = Math.Abs(a[r * n + c]);
                if (v > best) { best = v; p = r; }
            }
            piv[c] = p;
            if (best == 0) throw new InvalidOperationException("Matrix is singular.");
            if (p != c)
                for (var j = 0; j < n; j++) (a[c * n + j], a[p * n + j]) = (a[p * n + j], a[c * n + j]);
            var d = a[c * n + c];
            for (var r = c + 1; r < n; r++)
            {
                var f = a[r * n + c] / d;
                a[r * n + c] = f;
                if (f == 0) continue;
                var ar = a.AsSpan(r * n + c + 1, n - c - 1);
                var ac = a.AsSpan(c * n + c + 1, n - c - 1);
                for (var j = 0; j < ar.Length; j++) ar[j] -= f * ac[j];
            }
        }
    }

    /// <summary>Solves with the factors of <see cref="LuFactor"/>; b is (n, nrhs) row-major.</summary>
    public static void LuSolve(double[] lu, int n, int[] piv, Span<double> b, int nrhs)
    {
        for (var c = 0; c < n; c++)
        {
            var p = piv[c];
            if (p != c)
                for (var q = 0; q < nrhs; q++) (b[c * nrhs + q], b[p * nrhs + q]) = (b[p * nrhs + q], b[c * nrhs + q]);
        }
        for (var i = 0; i < n; i++)
            for (var j = 0; j < i; j++)
            {
                var f = lu[i * n + j];
                if (f == 0) continue;
                for (var q = 0; q < nrhs; q++) b[i * nrhs + q] -= f * b[j * nrhs + q];
            }
        for (var i = n - 1; i >= 0; i--)
        {
            for (var j = i + 1; j < n; j++)
            {
                var f = lu[i * n + j];
                if (f == 0) continue;
                for (var q = 0; q < nrhs; q++) b[i * nrhs + q] -= f * b[j * nrhs + q];
            }
            var d = lu[i * n + i];
            for (var q = 0; q < nrhs; q++) b[i * nrhs + q] /= d;
        }
    }

    /// <summary>
    /// Blocked right-looking LU with partial pivoting (the algorithm of LAPACK's
    /// <c>dgetrf</c>, panel width 64): the trailing update, which is nearly all the work, runs as
    /// threaded, vectorised row updates over a cache-resident panel of U. Same pivots and same
    /// factors as <see cref="LuFactor"/> up to rounding; about an order of magnitude faster
    /// for n in the thousands.
    /// </summary>
    public static void LuFactorBlocked(double[] a, int n, int[] piv)
    {
        const int nb = 64;
        if (n < 2 * nb) { LuFactor(a, n, piv); return; }
        for (var k0 = 0; k0 < n; k0 += nb)
        {
            var kb = Math.Min(nb, n - k0);
            var kend = k0 + kb;
            // Panel: unblocked LU of columns k0..kend-1 over rows k0..n-1, full-row swaps.
            for (var c = k0; c < kend; c++)
            {
                var p = c;
                var best = Math.Abs(a[c * n + c]);
                for (var r = c + 1; r < n; r++)
                {
                    var v = Math.Abs(a[r * n + c]);
                    if (v > best) { best = v; p = r; }
                }
                piv[c] = p;
                if (best == 0) throw new InvalidOperationException("Matrix is singular.");
                if (p != c)
                    for (var j = 0; j < n; j++) (a[c * n + j], a[p * n + j]) = (a[p * n + j], a[c * n + j]);
                var d = a[c * n + c];
                var cc = c;
                void PanelRows(int r0, int r1)
                {
                    for (var r = r0; r < r1; r++)
                    {
                        var f = a[r * n + cc] / d;
                        a[r * n + cc] = f;
                        if (f == 0) continue;
                        for (var j = cc + 1; j < kend; j++) a[r * n + j] -= f * a[cc * n + j];
                    }
                }
                var rows = n - c - 1;
                if (rows * (long)(kend - c) > 200_000)
                    ForChunks(rows, kend - c, (s0, e0) => PanelRows(c + 1 + s0, c + 1 + e0));
                else PanelRows(c + 1, n);
            }
            if (kend >= n) break;
            var w = n - kend;
            // U12 = L11^-1 A12 (unit lower).
            for (var c = k0; c < kend; c++)
            {
                var uc = a.AsSpan(c * n + kend, w);
                for (var r = c + 1; r < kend; r++)
                {
                    var f = a[r * n + c];
                    if (f != 0) Axpy(-f, uc, a.AsSpan(r * n + kend, w));
                }
            }
            // A22 -= L21 U12. Large updates go through GraviNum's register-blocked matrix
            // product; small ones are fused row updates.
            var mrows = n - kend;
            if ((long)mrows * w * kb >= 4_000_000)
            {
                var l21 = new double[mrows * kb];
                for (var r = 0; r < mrows; r++) Array.Copy(a, (kend + r) * n + k0, l21, r * kb, kb);
                var u12 = new double[kb * w];
                for (var c = 0; c < kb; c++) Array.Copy(a, (k0 + c) * n + kend, u12, c * w, w);
                var prod = LinAlg.Dot(new NdArray(l21, mrows, kb), new NdArray(u12, kb, w)).AsContiguous().ToArray();
                ForChunks(mrows, w, (s0, e0) =>
                {
                    for (var r = s0; r < e0; r++) Axpy(-1.0, prod.AsSpan(r * w, w), a.AsSpan((kend + r) * n + kend, w));
                });
                continue;
            }
            ForChunks(mrows, (long)kb * w, (s0, e0) =>
            {
                for (var r = kend + s0; r < kend + e0; r++)
                {
                    var row = a.AsSpan(r * n + kend, w);
                    var c = k0;
                    // Four panel columns per sweep: the row is loaded and stored once per four
                    // multiply-adds instead of once per one.
                    for (; c + 3 < kend; c += 4)
                        Axpy4(-a[r * n + c], -a[r * n + c + 1], -a[r * n + c + 2], -a[r * n + c + 3],
                            a.AsSpan(c * n + kend, w), a.AsSpan((c + 1) * n + kend, w),
                            a.AsSpan((c + 2) * n + kend, w), a.AsSpan((c + 3) * n + kend, w), row);
                    for (; c < kend; c++)
                    {
                        var f = a[r * n + c];
                        if (f != 0) Axpy(-f, a.AsSpan(c * n + kend, w), row);
                    }
                }
            });
        }
    }

    /// <summary>y += a0 x0 + a1 x1 + a2 x2 + a3 x3, vectorised.</summary>
    public static void Axpy4(double a0, double a1, double a2, double a3, ReadOnlySpan<double> x0, ReadOnlySpan<double> x1,
        ReadOnlySpan<double> x2, ReadOnlySpan<double> x3, Span<double> y)
    {
        var i = 0;
        var vc = System.Numerics.Vector<double>.Count;
        if (System.Numerics.Vector.IsHardwareAccelerated && y.Length >= vc)
        {
            var v0 = new System.Numerics.Vector<double>(a0);
            var v1 = new System.Numerics.Vector<double>(a1);
            var v2 = new System.Numerics.Vector<double>(a2);
            var v3 = new System.Numerics.Vector<double>(a3);
            var X0 = System.Runtime.InteropServices.MemoryMarshal.Cast<double, System.Numerics.Vector<double>>(x0);
            var X1 = System.Runtime.InteropServices.MemoryMarshal.Cast<double, System.Numerics.Vector<double>>(x1);
            var X2 = System.Runtime.InteropServices.MemoryMarshal.Cast<double, System.Numerics.Vector<double>>(x2);
            var X3 = System.Runtime.InteropServices.MemoryMarshal.Cast<double, System.Numerics.Vector<double>>(x3);
            var Y = System.Runtime.InteropServices.MemoryMarshal.Cast<double, System.Numerics.Vector<double>>(y);
            for (var k = 0; k < Y.Length; k++) Y[k] += v0 * X0[k] + v1 * X1[k] + v2 * X2[k] + v3 * X3[k];
            i = Y.Length * vc;
        }
        for (; i < y.Length; i++) y[i] += a0 * x0[i] + a1 * x1[i] + a2 * x2[i] + a3 * x3[i];
    }

    /// <summary>y += alpha x, vectorised.</summary>
    public static void Axpy(double alpha, ReadOnlySpan<double> x, Span<double> y)
    {
        var i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated && x.Length >= System.Numerics.Vector<double>.Count)
        {
            var va = new System.Numerics.Vector<double>(alpha);
            var vx = System.Runtime.InteropServices.MemoryMarshal.Cast<double, System.Numerics.Vector<double>>(x);
            var vy = System.Runtime.InteropServices.MemoryMarshal.Cast<double, System.Numerics.Vector<double>>(y);
            for (var k = 0; k < vx.Length; k++) vy[k] += va * vx[k];
            i = vx.Length * System.Numerics.Vector<double>.Count;
        }
        for (; i < x.Length; i++) y[i] += alpha * x[i];
    }

    /// <summary>Dense solve A X = B (A copied), B (n, nrhs) row-major, returned as a new array.</summary>
    public static double[] DenseSolve(double[] a, int n, double[] b, int nrhs)
    {
        var lu = (double[])a.Clone();
        var piv = new int[n];
        LuFactorBlocked(lu, n, piv);
        var x = (double[])b.Clone();
        LuSolve(lu, n, piv, x, nrhs);
        return x;
    }

    // ================================================================ axis bookkeeping

    /// <summary>
    /// Moves <paramref name="axis"/> of y to the front and returns the contiguous (n, nt) buffer
    /// plus the shape of the remaining axes in order (scipy's <c>np.moveaxis(y, axis, 0)</c>).
    /// </summary>
    public static double[] MoveAxisToFront(NdArray y, int axis, out int n, out int[] rest)
    {
        var rank = y.Rank;
        if (rank == 0) throw new ArgumentException("y must have at least one dimension.");
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentException($"axis {axis} is out of range for a {rank}-D array.");
        var shape = y.Shape.ToArray();
        n = shape[axis];
        rest = new int[rank - 1];
        for (int i = 0, j = 0; i < rank; i++) if (i != axis) rest[j++] = shape[i];
        if (axis == 0) return y.ToArray();
        var perm = new int[rank];
        perm[0] = axis;
        for (int i = 0, j = 1; i < rank; i++) if (i != axis) perm[j++] = i;
        return y.Transpose(perm).ToArray();
    }

    /// <summary>Normalises an axis argument against a rank.</summary>
    public static int NormalizeAxis(int axis, int rank)
    {
        var a = axis < 0 ? axis + rank : axis;
        if (a < 0 || a >= Math.Max(rank, 1)) throw new ArgumentException($"axis {axis} is out of range for a {rank}-D array.");
        return a;
    }

    /// <summary>
    /// Builds the result of evaluating at points of shape <paramref name="xShape"/> from a
    /// (nx, nt) buffer: shape xShape + rest, then the x axes moved to position
    /// <paramref name="axis"/> (scipy's output transpose).
    /// </summary>
    public static NdArray PlaceResult(double[] flat, ReadOnlySpan<int> xShape, int[] rest, int axis)
    {
        var shape = new int[xShape.Length + rest.Length];
        xShape.CopyTo(shape);
        rest.CopyTo(shape, xShape.Length);
        var arr = new NdArray(flat, shape);
        if (axis == 0 || rest.Length == 0) return arr;
        var xn = xShape.Length;
        var nd = shape.Length;
        // l = l[xn : xn+axis] + l[:xn] + l[xn+axis:]
        var perm = new int[nd];
        var p = 0;
        for (var i = xn; i < xn + axis; i++) perm[p++] = i;
        for (var i = 0; i < xn; i++) perm[p++] = i;
        for (var i = xn + axis; i < nd; i++) perm[p++] = i;
        return arr.Transpose(perm).Copy();
    }

    // ================================================================ misc maths

    /// <summary>Binomial coefficient as a double (exact for the small arguments used here).</summary>
    public static double Binom(int n, int k)
    {
        if (k < 0 || k > n) return 0;
        k = Math.Min(k, n - k);
        var r = 1.0;
        for (var i = 1; i <= k; i++) r = r * (n - k + i) / i;
        return Math.Round(r);
    }

    /// <summary>n! as a double.</summary>
    public static double Factorial(int n)
    {
        var r = 1.0;
        for (var i = 2; i <= n; i++) r *= i;
        return r;
    }

    /// <summary>Python's floor-mod for doubles (result has the sign of the divisor).</summary>
    public static double PyMod(double a, double b)
    {
        var r = a % b;
        if (r != 0 && (r < 0) != (b < 0)) r += b;
        return r;
    }

    /// <summary>Should a batch of <paramref name="count"/> evaluations each costing about
    /// <paramref name="costPerItem"/> flops be split across threads?</summary>
    public static bool UseParallel(long count, long costPerItem = 16) =>
        Environment.ProcessorCount > 1 && count * costPerItem >= 200_000;

    /// <summary>Runs body(start, end) over [0, count) in chunks, in parallel when worthwhile.</summary>
    public static void ForChunks(int count, long costPerItem, Action<int, int> body)
    {
        if (!UseParallel(count, costPerItem))
        {
            body(0, count);
            return;
        }
        var chunks = Math.Min(Environment.ProcessorCount * 4, Math.Max(1, count / 2048));
        var size = (count + chunks - 1) / chunks;
        Parallel.For(0, chunks, ci =>
        {
            var s = ci * size;
            var e = Math.Min(count, s + size);
            if (s < e) body(s, e);
        });
    }
}
