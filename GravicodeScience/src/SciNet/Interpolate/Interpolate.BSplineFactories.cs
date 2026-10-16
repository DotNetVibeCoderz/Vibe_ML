using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.InterpolateSupport;

namespace Gravicode.Science.SciNet;

/// <summary>
/// scipy.interpolate: one-dimensional and multivariate interpolation, splines, smoothing
/// splines and FITPACK-compatible curve fitting.
/// </summary>
public static partial class Interpolate
{
    // ================================================================ knot helpers

    /// <summary>scipy's <c>_not_a_knot</c>: knots at the data (odd k) or midpoints (even k),
    /// dropping the ones next to the ends, with k+1-fold boundary knots.</summary>
    internal static double[] NotAKnotKnots(double[] x, int k)
    {
        double[] inner;
        int k2;
        if (k % 2 == 1)
        {
            k2 = (k + 1) / 2;
            inner = x;
        }
        else
        {
            k2 = k / 2;
            inner = new double[x.Length - 1];
            for (var i = 0; i < inner.Length; i++) inner[i] = (x[i + 1] + x[i]) / 2;
        }
        var mid = Math.Max(0, inner.Length - 2 * k2);
        var t = new double[2 * (k + 1) + mid];
        for (var i = 0; i <= k; i++) { t[i] = x[0]; t[^(i + 1)] = x[^1]; }
        Array.Copy(inner, k2, t, k + 1, mid);
        return t;
    }

    /// <summary>scipy's <c>_augknt</c>: data points as knots with k extra copies of each end.</summary>
    internal static double[] AugmentedKnots(double[] x, int k)
    {
        var t = new double[x.Length + 2 * k];
        for (var i = 0; i < k; i++) { t[i] = x[0]; t[^(i + 1)] = x[^1]; }
        Array.Copy(x, 0, t, k, x.Length);
        return t;
    }

    /// <summary>scipy's <c>_periodic_knots</c>.</summary>
    internal static double[] PeriodicKnots(double[] x, int k)
    {
        var xc = (double[])x.Clone();
        var n = xc.Length;
        if (k % 2 == 0)
        {
            var d0 = new double[n - 1];
            for (var i = 0; i < n - 1; i++) d0[i] = xc[i + 1] - xc[i];
            for (var i = 1; i < n - 1; i++) xc[i] -= d0[i - 1] / 2;
        }
        var dx = new double[n - 1];
        for (var i = 0; i < n - 1; i++) dx[i] = xc[i + 1] - xc[i];
        var t = new double[n + 2 * k];
        Array.Copy(xc, 0, t, k, n);
        var len = t.Length;
        for (var i = 0; i < k; i++)
        {
            t[k - i - 1] = t[k - i] - dx[n - 1 - 1 - (i % (n - 1))];
            t[len - k + i] = t[len - k + i - 1] + dx[i % (n - 1)];
        }
        return t;
    }

    // ================================================================ make_interp_spline

    /// <summary>
    /// Interpolating B-spline of degree k (scipy.interpolate.make_interp_spline). Default knots
    /// are not-a-knot; with derivative end conditions they are the data points with k-fold
    /// ends. The collocation system is banded and solved with partial pivoting (LAPACK
    /// <c>gbsv</c>'s algorithm); periodic splines solve the cyclic system with a Woodbury
    /// correction of the banded part.
    /// </summary>
    public static BSpline MakeInterpSpline(double[] x, double[] y, int k = 3, double[]? t = null, SplineBoundary? bcType = null) =>
        MakeInterpSpline(x, new NdArray((double[])(y ?? throw new ArgumentNullException(nameof(y))).Clone(), y.Length), k, t, bcType, 0);

    /// <summary>N-D overload: y's <paramref name="axis"/> runs along x.</summary>
    public static BSpline MakeInterpSpline(double[] x, NdArray y, int k = 3, double[]? t = null, SplineBoundary? bcType = null, int axis = 0)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        var bc = bcType ?? SplineBoundaryCondition.NotAKnot;
        var periodic = bc.Start.Kind == SplineBoundaryKind.Periodic || bc.End.Kind == SplineBoundaryKind.Periodic;
        if (periodic && (bc.Start.Kind != SplineBoundaryKind.Periodic || bc.End.Kind != SplineBoundaryKind.Periodic))
            throw new ArgumentException("'periodic' must be used at both ends.");
        var derivL = bc.Start.Kind == SplineBoundaryKind.Derivatives ? bc.Start : null;
        var derivR = bc.End.Kind == SplineBoundaryKind.Derivatives ? bc.End : null;

        axis = NormalizeAxis(axis, y.Rank);
        var xs = CopyFinite(x, "x");
        var yy = MoveAxisToFront(y, axis, out var ny, out var rest);
        foreach (var v in yy) if (!double.IsFinite(v)) throw new ArgumentException("Array must not contain infs or nans.", nameof(y));
        var nt = Product(rest);
        if (xs.Length != ny) throw new ArgumentException($"Shapes of x ({xs.Length}) and y ({ny}) are incompatible");
        for (var i = 1; i < xs.Length; i++)
        {
            if (xs[i] == xs[i - 1]) throw new ArgumentException("Expect x to not have duplicates");
            if (xs[i] < xs[i - 1]) throw new ArgumentException("Expect x to be a 1D strictly increasing sequence.");
        }
        if (periodic)
            for (var j = 0; j < nt; j++)
                if (!(Math.Abs(yy[j] - yy[(ny - 1) * nt + j]) <= 1e-15 + 1e-5 * Math.Abs(yy[(ny - 1) * nt + j])))
                    throw new ArgumentException("First and last points does not match while periodic case expected");
        if (k < 0) throw new ArgumentException("Expect non-negative k.");

        if (k == 0)
        {
            if (t is not null || derivL is not null || derivR is not null)
                throw new ArgumentException("Too much info for k=0: t and bc_type can only be None.");
            return BSpline.Raw([.. xs, xs[^1]], yy, ny, rest, 0, InterpolateExtrapolate.Extrapolate, axis);
        }
        if (k == 1 && t is null)
        {
            if (derivL is not null || derivR is not null) throw new ArgumentException("Too much info for k=1: bc_type can only be None.");
            return BSpline.Raw([xs[0], .. xs, xs[^1]], yy, ny, rest, 1, InterpolateExtrapolate.Extrapolate, axis);
        }
        if (periodic && t is not null)
            throw new NotSupportedException("For periodic case t is constructed automatically and can not be passed manually");
        if (t is null)
        {
            if (derivL is null && derivR is null) t = periodic ? PeriodicKnots(xs, k) : NotAKnotKnots(xs, k);
            else t = AugmentedKnots(xs, k);
        }
        else
        {
            t = CopyFinite(t, "t");
            RequireNonDecreasing(t, "t");
        }
        if (t.Length < xs.Length + k + 1) throw new ArgumentException($"Got {t.Length} knots, need at least {xs.Length + k + 1}.");
        if (xs[0] < t[k] || xs[^1] > t[t.Length - k]) throw new ArgumentException($"Out of bounds w/ x = [{xs[0]}, ..., {xs[^1]}].");

        if (periodic) return MakePeriodicSpline(xs, yy, nt, rest, t, k, axis);

        var nleft = derivL?.Derivatives.Count ?? 0;
        var nright = derivR?.Derivatives.Count ?? 0;
        if (derivL is not null) foreach (var d in derivL.Derivatives) if (d.Order < 0 || d.Order > k) throw new ArgumentException($"Bad boundary conditions at {xs[0]}.");
        if (derivR is not null) foreach (var d in derivR.Derivatives) if (d.Order < 0 || d.Order > k) throw new ArgumentException($"Bad boundary conditions at {xs[^1]}.");
        var n = xs.Length;
        var ncoef = t.Length - k - 1;
        if (ncoef - n != nleft + nright)
            throw new ArgumentException($"The number of derivatives at boundaries does not match: expected {ncoef - n}, got {nleft}+{nright}");

        var lu = new BandedLu(ncoef, k, k);
        var h = new double[k + 1];
        var w = new double[k + 1];
        var rhs = new double[ncoef * nt];
        void DerivRows(double xval, SplineBoundaryCondition cond, int offset)
        {
            var left = FindSplineInterval(t, k, xval, k, false);
            for (var row = 0; row < cond.Derivatives.Count; row++)
            {
                DeBoorD(t, xval, k, left, cond.Derivatives[row].Order, h, w);
                for (var a = 0; a <= k; a++) lu[offset + row, left - k + a] = h[a];
                var val = cond.ValueFor(row, nt);
                for (var q = 0; q < nt; q++) rhs[(offset + row) * nt + q] = val[q];
            }
        }
        var hint = k;
        for (var j = 0; j < n; j++)
        {
            var left = FindSplineInterval(t, k, xs[j], hint, false);
            hint = left;
            DeBoorD(t, xs[j], k, left, 0, h, w);
            for (var a = 0; a <= k; a++) lu[j + nleft, left - k + a] = h[a];
            for (var q = 0; q < nt; q++) rhs[(j + nleft) * nt + q] = yy[j * nt + q];
        }
        if (derivL is not null) DerivRows(xs[0], derivL, 0);
        if (derivR is not null) DerivRows(xs[^1], derivR, ncoef - nright);
        try { lu.Factor(); }
        catch (InvalidOperationException) { throw new InvalidOperationException("Colocation matrix is singular."); }
        lu.Solve(rhs, nt);
        return BSpline.Raw(t, rhs, ncoef, rest, k, InterpolateExtrapolate.Extrapolate, axis);
    }

    private static BSpline MakePeriodicSpline(double[] x, double[] y, int nt, int[] rest, double[] t, int k, int axis)
    {
        var n = x.Length;
        var ncoef = n + k - 1;
        var c = new double[ncoef * nt];
        Span<double> h = stackalloc double[k + 1];
        Span<double> w = stackalloc double[k + 1];
        if (n <= k)
        {
            // scipy's _make_interp_per_full_matr: collocation plus derivative continuity rows.
            var mat = new double[ncoef * ncoef];
            for (var i = 0; i < k - 1; i++)
            {
                DeBoorD(t, x[0], k, k, i + 1, h, w);
                for (var a = 0; a <= k; a++) mat[i * ncoef + a] += h[a];
                DeBoorD(t, x[^1], k, n + k - 1, i + 1, h, w);
                for (var a = 0; a < k; a++) mat[i * ncoef + ncoef - k + a] -= h[a];
            }
            for (var i = 0; i < n; i++)
            {
                var left = x[i] == t[k] ? k : SearchLeft(t, x[i]) - 1;
                DeBoorD(t, x[i], k, left, 0, h, w);
                for (var a = 0; a <= k; a++) mat[(i + k - 1) * ncoef + left - k + a] = h[a];
            }
            var b = new double[ncoef * nt];
            for (var i = 0; i < n; i++)
                for (var q = 0; q < nt; q++) b[(i + k - 1) * nt + q] = y[i * nt + q];
            return BSpline.Raw(t, DenseSolve(mat, ncoef, b, nt), ncoef, rest, k, InterpolateExtrapolate.Periodic, axis);
        }
        // Unknowns cc[0..N), N = n-1; c_full[j] = cc[(j - kul) mod N].
        var N = n - 1;
        var kul = k / 2;
        var entries = new List<(int Row, int Col, double Value)>();
        var hint = k;
        for (var i = 0; i < N; i++)
        {
            var left = FindSplineInterval(t, k, x[i], hint, false);
            hint = left;
            DeBoorD(t, x[i], k, left, 0, h, w);
            for (var a = 0; a <= k; a++)
            {
                var j = left - k + a;
                var col = ((j - kul) % N + N) % N;
                entries.Add((i, col, h[a]));
            }
        }
        var rhs = new double[N * nt];
        for (var i = 0; i < N; i++)
            for (var q = 0; q < nt; q++) rhs[i * nt + q] = y[i * nt + q];
        var cc = SolveCyclic(N, k, entries, rhs, nt);
        for (var j = 0; j < ncoef; j++)
        {
            var src = ((j - kul) % N + N) % N;
            for (var q = 0; q < nt; q++) c[j * nt + q] = cc[src * nt + q];
        }
        return BSpline.Raw(t, c, ncoef, rest, k, InterpolateExtrapolate.Periodic, axis);
    }

    /// <summary>
    /// Solves a "cyclic banded" system: entries within bandwidth <paramref name="bw"/> go to a
    /// banded LU, the wrap-around corners become a low-rank Woodbury correction (scipy's
    /// <c>_woodbury_algorithm</c> idea). Small systems are solved densely.
    /// </summary>
    private static double[] SolveCyclic(int N, int bw, List<(int Row, int Col, double Value)> entries, double[] rhs, int nt)
    {
        if (N <= Math.Max(4 * bw + 4, 64))
        {
            var mat = new double[N * N];
            foreach (var (r, c, v) in entries) mat[r * N + c] += v;
            return DenseSolve(mat, N, rhs, nt);
        }
        var lu = new BandedLu(N, bw, bw);
        var outCols = new SortedDictionary<int, double[]>();
        foreach (var (r, c, v) in entries)
        {
            if (Math.Abs(r - c) <= bw) lu.Add(r, c, v);
            else
            {
                if (!outCols.TryGetValue(c, out var col)) outCols[c] = col = new double[N];
                col[r] += v;
            }
        }
        lu.Factor();
        var y = (double[])rhs.Clone();
        lu.Solve(y, nt);
        if (outCols.Count == 0) return y;
        var cols = outCols.Keys.ToArray();
        var r2 = cols.Length;
        // Z = B^-1 U (N x r2)
        var z = new double[N * r2];
        for (var a = 0; a < r2; a++)
        {
            var u = outCols[cols[a]];
            for (var i = 0; i < N; i++) z[i * r2 + a] = u[i];
        }
        lu.Solve(z, r2);
        // H = I + V^T Z, V^T picks rows cols[]
        var hm = new double[r2 * r2];
        for (var a = 0; a < r2; a++)
            for (var b = 0; b < r2; b++) hm[a * r2 + b] = (a == b ? 1 : 0) + z[cols[a] * r2 + b];
        // rhs2 = V^T y
        var vy = new double[r2 * nt];
        for (var a = 0; a < r2; a++)
            for (var q = 0; q < nt; q++) vy[a * nt + q] = y[cols[a] * nt + q];
        var s = DenseSolve(hm, r2, vy, nt);
        for (var i = 0; i < N; i++)
            for (var q = 0; q < nt; q++)
            {
                var acc = 0.0;
                for (var a = 0; a < r2; a++) acc += z[i * r2 + a] * s[a * nt + q];
                y[i * nt + q] -= acc;
            }
        return y;
    }

    // ================================================================ make_lsq_spline

    /// <summary>
    /// Weighted least-squares B-spline with given knots (scipy.interpolate.make_lsq_spline,
    /// method "qr"): Givens QR of the banded observation matrix, as FITPACK. Throws when the
    /// knots violate the Schoenberg–Whitney conditions (the system is singular).
    /// </summary>
    public static BSpline MakeLsqSpline(double[] x, double[] y, double[] t, int k = 3, double[]? w = null) =>
        MakeLsqSpline(x, new NdArray((double[])(y ?? throw new ArgumentNullException(nameof(y))).Clone(), y.Length), t, k, w, 0);

    /// <summary>N-D overload: y's <paramref name="axis"/> runs along x.</summary>
    public static BSpline MakeLsqSpline(double[] x, NdArray y, double[] t, int k = 3, double[]? w = null, int axis = 0)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(t);
        axis = NormalizeAxis(axis, y.Rank);
        var xs = CopyFinite(x, "x");
        var ts = CopyFinite(t, "t");
        var yy = MoveAxisToFront(y, axis, out var ny, out var rest);
        var nt = Product(rest);
        var ww = w is null ? Enumerable.Repeat(1.0, xs.Length).ToArray() : CopyFinite(w, "w");
        if (k < 0) throw new ArgumentException("Expect non-negative k.");
        if (xs.Length < k + 1) throw new ArgumentException("Need more x points.");
        RequireNonDecreasing(ts, "t");
        if (xs.Length != ny) throw new ArgumentException($"Shapes of x ({xs.Length}) and y ({ny}) are incompatible");
        if (xs.Length != ww.Length) throw new ArgumentException("Shapes of x and w are incompatible");
        RequireNonDecreasing(xs, "x");
        if (k > 0)
            foreach (var v in xs)
                if (v < ts[k] || v > ts[ts.Length - k]) throw new ArgumentException($"Out of bounds w/ x = {v}.");
        if (k == 0) throw new NotSupportedException("k = 0 least-squares splines are not supported.");
        var c = InterpolateFitpack.LeastSquares(xs, yy, nt, ww, ts, k, out _);
        var n = ts.Length - k - 1;
        var cc = new double[n * nt];
        Array.Copy(c, cc, n * nt);
        return BSpline.Raw(ts, cc, n, rest, k, InterpolateExtrapolate.Extrapolate, axis);
    }

    // ================================================================ make_smoothing_spline

    /// <summary>
    /// Penalised cubic smoothing spline minimising Σ wᵢ(yᵢ - g(xᵢ))² + λ ∫ g''² (scipy's
    /// <c>make_smoothing_spline</c>, Woltring's natural-spline basis). With
    /// <paramref name="lam"/> null, λ minimises the generalised cross-validation score, found
    /// with a port of scipy's bounded Brent minimiser on [0, n], so λ agrees with scipy to the
    /// minimiser's tolerance (1e-5 absolute) and usually far better.
    /// </summary>
    public static BSpline MakeSmoothingSpline(double[] x, double[] y, double[]? w = null, double? lam = null) =>
        MakeSmoothingSpline(x, y, w, lam, out _);

    /// <summary>As <see cref="MakeSmoothingSpline(double[], double[], double[], double?)"/>,
    /// also returning the λ used.</summary>
    public static BSpline MakeSmoothingSpline(double[] x, double[] y, double[]? w, double? lam, out double lambda)
    {
        var res = MakeSmoothingSpline(x, new NdArray((double[])(y ?? throw new ArgumentNullException(nameof(y))).Clone(), y.Length), w, lam, 0, out var lams);
        lambda = lams[0];
        return res;
    }

    /// <summary>N-D overload; λ is chosen per column when not given.</summary>
    public static BSpline MakeSmoothingSpline(double[] x, NdArray y, double[]? w, double? lam, int axis, out double[] lambdas)
    {
        ArgumentNullException.ThrowIfNull(x);
        var n = x.Length;
        var xs = CopyFinite(x, "x");
        for (var i = 1; i < n; i++) if (!(xs[i] - xs[i - 1] > 0)) throw new ArgumentException("``x`` should be an ascending array");
        axis = NormalizeAxis(axis, y.Rank);
        var yy = MoveAxisToFront(y, axis, out var ny, out var rest);
        if (ny != n) throw new ArgumentException("``x`` should be 1D and match y along axis");
        var nt = Product(rest);
        var ww = w is null ? Enumerable.Repeat(1.0, n).ToArray() : (double[])w.Clone();
        if (ww.Length != n) throw new ArgumentException("w must have the length of x");
        foreach (var v in ww) if (!(v > 0)) throw new ArgumentException("Invalid vector of weights");
        if (n <= 4) throw new ArgumentException("``x`` and ``y`` length must be at least 5");
        if (lam is < 0) throw new ArgumentException("Regularization parameter should be non-negative");

        var t = new double[n + 6];
        for (var i = 0; i < 3; i++) { t[i] = xs[0]; t[n + 3 + i] = xs[^1]; }
        Array.Copy(xs, 0, t, 3, n);

        // Dense-ish access to the B-spline design matrix rows (n x (n+2)), k = 3.
        var xb = new double[n, 4];
        var off = new int[n];
        {
            Span<double> h = stackalloc double[4];
            Span<double> wk = stackalloc double[4];
            var hint = 3;
            for (var i = 0; i < n; i++)
            {
                var l = FindSplineInterval(t, 3, xs[i], hint, true);
                hint = l;
                DeBoorD(t, xs[i], 3, l, 0, h, wk);
                off[i] = l - 3;
                for (var a = 0; a < 4; a++) xb[i, a] = h[a];
            }
        }
        double Xb(int r, int c)
        {
            var a = c - off[r];
            return a >= 0 && a < 4 ? xb[r, a] : 0.0;
        }
        // X in (2,2) banded storage: X[2 + i - j, j] = A[i, j].
        var X = new double[5, n];
        for (var i = 1; i < 4; i++)
            for (var r = 0; r < n - 4; r++) X[i, 2 + r] = Xb(i + r, 3 + r);
        X[1, 1] = Xb(0, 0);
        X[2, 0] = (xs[2] + xs[1] - 2 * xs[0]) * Xb(0, 0);
        X[2, 1] = Xb(1, 1) + Xb(1, 2);
        X[3, 0] = (xs[2] - xs[0]) * Xb(1, 1);
        X[3, 1] = Xb(2, 2);
        X[1, n - 2] = Xb(n - 3, n - 1);
        X[1, n - 1] = (xs[^1] - xs[^3]) * Xb(n - 2, n);
        X[2, n - 2] = Xb(n - 2, n - 1) + Xb(n - 2, n);
        X[2, n - 1] = (2 * xs[^1] - xs[^2] - xs[^3]) * Xb(n - 1, n + 1);
        X[3, n - 2] = Xb(n - 1, n + 1);

        var wE = new double[5, n];
        {
            var d = DividedDiff(xs, 0, 3);
            for (var r = 0; r < 3; r++) wE[2 + r, 0] = d[r] / ww[r];
            d = DividedDiff(xs, 0, 4);
            for (var r = 0; r < 4; r++) wE[1 + r, 1] = d[r] / ww[r];
            for (var j = 2; j < n - 2; j++)
            {
                d = DividedDiff(xs, j - 2, 5);
                for (var r = 0; r < 5; r++) wE[r, j] = (xs[j + 2] - xs[j - 2]) * d[r] / ww[j - 2 + r];
            }
            d = DividedDiff(xs, n - 4, 4);
            for (var r = 0; r < 4; r++) wE[r, n - 2] = -d[r] / ww[n - 4 + r];
            d = DividedDiff(xs, n - 3, 3);
            for (var r = 0; r < 3; r++) wE[r, n - 1] = d[r] / ww[n - 3 + r];
            for (var r = 0; r < 5; r++)
                for (var j = 0; j < n; j++) wE[r, j] *= 6;
        }

        lambdas = new double[nt];
        var cnat = new double[n * nt];
        double[,]? xtwx = null, xte = null;
        for (var q = 0; q < nt; q++)
        {
            var col = new double[n];
            for (var i = 0; i < n; i++) col[i] = yy[i * nt + q];
            double lq;
            if (lam is double lv) lq = lv;
            else
            {
                xtwx ??= BandedXtWY(X, ww, X);
                xte ??= BandedXtWY(X, ww, wE);
                var cx = X; var cw = wE; var a1 = xtwx; var a2 = xte;
                lq = FminBound(l => Gcv(l, cx, a1, cw, a2, col), 0, n);
            }
            lambdas[q] = lq;
            var sol = SolveBanded22(X, wE, lq, col);
            for (var i = 0; i < n; i++) cnat[i * nt + q] = sol[i];
        }
        // Back to the B-spline basis (Woltring's (2.2.10)).
        var cb = new double[(n + 2) * nt];
        for (var q = 0; q < nt; q++)
        {
            double c0 = cnat[q], c1 = cnat[nt + q], cm0 = cnat[(n - 1) * nt + q], cm1 = cnat[(n - 2) * nt + q];
            cb[q] = c0 * (t[5] + t[4] - 2 * t[3]) + c1;
            cb[nt + q] = c0 * (t[5] - t[3]) + c1;
            for (var i = 1; i < n - 1; i++) cb[(i + 1) * nt + q] = cnat[i * nt + q];
            var L = t.Length;
            cb[n * nt + q] = cm0 * (t[L - 4] - t[L - 6]) + cm1;
            cb[(n + 1) * nt + q] = cm0 * (2 * t[L - 4] - t[L - 5] - t[L - 6]) + cm1;
        }
        return BSpline.Raw(t, cb, n + 2, rest, 3, InterpolateExtrapolate.Extrapolate, axis);
    }

    private static double[] DividedDiff(double[] x, int start, int count)
    {
        var res = new double[count];
        for (var i = 0; i < count; i++)
        {
            var pp = 1.0;
            for (var k = 0; k < count; k++)
                if (k != i) pp *= x[start + i] - x[start + k];
            res[i] = 1.0 / pp;
        }
        return res;
    }

    private static double[] SolveBanded22(double[,] X, double[,] wE, double lam, double[] y)
    {
        var n = y.Length;
        var lu = new BandedLu(n, 2, 2);
        for (var j = 0; j < n; j++)
            for (var r = 0; r < 5; r++)
            {
                var i = r - 2 + j;
                if (i < 0 || i >= n) continue;
                lu[i, j] = X[r, j] + lam * wE[r, j];
            }
        lu.Factor();
        var b = (double[])y.Clone();
        lu.Solve(b, 1);
        return b;
    }

    /// <summary>scipy's compute_banded_symmetric_XT_W_Y: Xᵀ W Y in (3, 0) upper banded storage.</summary>
    private static double[,] BandedXtWY(double[,] X, double[] w, double[,] Y)
    {
        var n = X.GetLength(1);
        var wy = (double[,])Y.Clone();
        for (var r = 0; r < 5; r++)
            for (var j = 0; j < n; j++)
            {
                var i = r - 2 + j;
                if (i >= 0 && i < n) wy[r, j] *= w[i];
            }
        var res = new double[4, n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < Math.Min(n - i, 4); j++)
            {
                var s = 0.0;
                for (var r = 0; r < 5 - j; r++) s += X[j + r, i] * wy[r, i + j];
                res[3 - j, i + j] = s;
            }
        return res;
    }

    private static double Gcv(double lam, double[,] X, double[,] xtwx, double[,] wE, double[,] xte, double[] y)
    {
        var n = X.GetLength(1);
        var c = SolveBanded22(X, wE, lam, y);
        var numer = 0.0;
        for (var i = 0; i < n; i++)
        {
            var r = 0.0;
            for (var j = Math.Max(0, i - n + 3); j < Math.Min(5, i + 3); j++)
            {
                var col = i + 2 - j;
                r += wE[j, col] * c[col];
            }
            var v = lam * r;
            numer += v * v;
        }
        numer /= n;
        // lhs = XtWX + lam * XtE, (4, n) upper banded symmetric
        var lhs = new double[4, n];
        for (var r = 0; r < 4; r++)
            for (var j = 0; j < n; j++) lhs[r, j] = xtwx[r, j] + lam * xte[r, j];
        var bInv = BandedInverseBand(lhs);
        var tr = 0.0;
        for (var r = 0; r < 4; r++)
            for (var j = 0; j < n; j++)
            {
                var v = bInv[r, j] * xtwx[r, j];
                tr += r < 3 ? 2 * v : v;
            }
        var denom = (1 - tr / n) * (1 - tr / n);
        return numer / denom;
    }

    /// <summary>scipy's <c>compute_b_inv</c>: the band of the inverse of a symmetric positive
    /// definite (3, 0)-banded matrix from its banded Cholesky factor (Hutchinson–de Hoog).</summary>
    private static double[,] BandedInverseBand(double[,] A)
    {
        var n = A.GetLength(1);
        // Upper banded Cholesky A = UᵀU, storage U[3 + i - j, j] = U(i, j).
        var U = new double[4, n];
        for (var j = 0; j < n; j++)
        {
            for (var i = Math.Max(0, j - 3); i <= j; i++)
            {
                var s = A[3 + i - j, j];
                for (var p = Math.Max(0, j - 3); p < i; p++) s -= U[3 + p - i, i] * U[3 + p - j, j];
                if (i == j)
                {
                    if (!(s > 0)) throw new ArgumentException("Seems like the problem is ill-posed");
                    U[3, j] = Math.Sqrt(s);
                }
                else U[3 + i - j, j] = s / U[3, i];
            }
        }
        for (var i = 2; i < 5; i++)
            for (var j = i - 1; j < n; j++) U[4 - i, j] /= U[3, j - i + 1];
        var D = new double[n];
        for (var j = 0; j < n; j++) { D[j] = 1.0 / (U[3, j] * U[3, j]); U[3, j] = 1.0; }
        var B = new double[4, n];
        for (var i = n - 1; i >= 0; i--)
            for (var j = Math.Min(3, n - i - 1); j >= 0; j--)
            {
                var rng = Math.Min(3, n - i - 1);
                var sum = 0.0;
                if (j == 0)
                {
                    for (var k = 1; k <= rng; k++) sum -= U[3 - k, i + k] * B[3 - k, i + k];
                    sum += D[i];
                    B[3, i] = sum;
                }
                else
                {
                    for (var k = 1; k <= rng; k++)
                    {
                        var diag = Math.Abs(k - j);
                        var ind = i + Math.Min(k, j);
                        sum -= U[3 - k, i + k] * B[3 - diag, ind + diag];
                    }
                    B[3 - j, i + j] = sum;
                }
            }
        for (var j = 0; j < n; j++) B[0, j] = 0;
        return B;
    }

    /// <summary>A port of scipy's <c>_minimize_scalar_bounded</c> (Brent's fminbound) with
    /// xatol = 1e-5 and maxiter = 500, step for step, so λ matches scipy's choice.</summary>
    internal static double FminBound(Func<double, double> func, double x1, double x2, double xatol = 1e-5, int maxfun = 500)
    {
        var sqrtEps = Math.Sqrt(2.2e-16);
        var goldenMean = 0.5 * (3.0 - Math.Sqrt(5.0));
        double a = x1, b = x2;
        var fulc = a + goldenMean * (b - a);
        double nfc = fulc, xf = fulc;
        double rat = 0, e = 0;
        var x = xf;
        var fx = func(x);
        var num = 1;
        double ffulc = fx, fnfc = fx;
        var xm = 0.5 * (a + b);
        var tol1 = sqrtEps * Math.Abs(xf) + xatol / 3.0;
        var tol2 = 2.0 * tol1;
        while (Math.Abs(xf - xm) > tol2 - 0.5 * (b - a))
        {
            var golden = true;
            if (Math.Abs(e) > tol1)
            {
                golden = false;
                var r = (xf - nfc) * (fx - ffulc);
                var q = (xf - fulc) * (fx - fnfc);
                var p = (xf - fulc) * q - (xf - nfc) * r;
                q = 2.0 * (q - r);
                if (q > 0.0) p = -p;
                q = Math.Abs(q);
                r = e;
                e = rat;
                if (Math.Abs(p) < Math.Abs(0.5 * q * r) && p > q * (a - xf) && p < q * (b - xf))
                {
                    rat = (p + 0.0) / q;
                    x = xf + rat;
                    if (x - a < tol2 || b - x < tol2)
                    {
                        var si0 = Math.Sign(xm - xf) + (xm - xf == 0 ? 1 : 0);
                        rat = tol1 * si0;
                    }
                }
                else golden = true;
            }
            if (golden)
            {
                e = xf >= xm ? a - xf : b - xf;
                rat = goldenMean * e;
            }
            var si = Math.Sign(rat) + (rat == 0 ? 1 : 0);
            x = xf + si * Math.Max(Math.Abs(rat), tol1);
            var fu = func(x);
            num++;
            if (fu <= fx)
            {
                if (x >= xf) a = xf; else b = xf;
                fulc = nfc; ffulc = fnfc;
                nfc = xf; fnfc = fx;
                xf = x; fx = fu;
            }
            else
            {
                if (x < xf) a = x; else b = x;
                if (fu <= fnfc || nfc == xf)
                {
                    fulc = nfc; ffulc = fnfc;
                    nfc = x; fnfc = fu;
                }
                else if (fu <= ffulc || fulc == xf || fulc == nfc)
                {
                    fulc = x; ffulc = fu;
                }
            }
            xm = 0.5 * (a + b);
            tol1 = sqrtEps * Math.Abs(xf) + xatol / 3.0;
            tol2 = 2.0 * tol1;
            if (num >= maxfun) break;
        }
        if (double.IsNaN(xf) || double.IsNaN(fx)) throw new ArithmeticException("Unable to find minimum of the GCV function.");
        return xf;
    }
}
