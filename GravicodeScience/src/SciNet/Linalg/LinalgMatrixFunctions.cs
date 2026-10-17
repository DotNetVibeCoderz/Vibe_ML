using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Linalg
{
    // ================================================================ expm

    private static readonly double[] Pade3 = [120, 60, 12, 1];
    private static readonly double[] Pade5 = [30240, 15120, 3360, 420, 30, 1];
    private static readonly double[] Pade7 = [17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1];
    private static readonly double[] Pade9 = [17643225600, 8821612800, 2075673600, 302702400, 30270240, 2162160, 110880, 3960, 90, 1];
    private static readonly double[] Pade13 =
    [
        64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000,
        10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1,
    ];

    /// <summary>
    /// Matrix exponential, scipy's <c>expm</c>: the scaling-and-squaring algorithm of
    /// Al-Mohy &amp; Higham (2009) with Padé degrees 3, 5, 7, 9 and 13.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The degree and the number of squarings are chosen from <c>d_k = ‖A^k‖₁^{1/k}</c> rather than
    /// from ‖A‖₁ alone (Higham 2005 used the latter). For a non-normal matrix ‖A^k‖^{1/k} can be far
    /// smaller than ‖A‖, and over-scaling is what made the 2005 algorithm lose accuracy on badly
    /// scaled examples: every unnecessary squaring doubles the rounding error. The
    /// <c>ℓ(A, m)</c> correction then adds squarings only when the backward-error bound demands it.
    /// Norms of powers are exact below n = 200 (as scipy) and estimated by Hager's method above.
    /// </para>
    /// <para>
    /// Upper-triangular input gets the exact diagonal and first superdiagonal recomputed at every
    /// squaring (Al-Mohy–Higham Code Fragment 2.1). Without that, the classic badly scaled 2×2
    /// triangular examples lose most of their digits in the off-diagonal.
    /// </para>
    /// </remarks>
    public static NdArray Expm(NdArray a)
    {
        LinalgKernels.RequireSquare(a, nameof(Expm));
        var n = a.Shape[0];
        if (n == 0) return NdArray.Zeros(0, 0);
        var data = a.ToArray();
        LinalgKernels.RequireFinite(data, nameof(Expm));
        if (n == 1) return new NdArray([Math.Exp(data[0])], 1, 1);
        return ExpmCore(data, n);
    }

    private static NdArray ExpmCore(double[] data, int n)
    {
        var triangular = LinalgKernels.IsUpperTriangular(data, n);
        var exact = n < 200;
        // |A| and ‖A‖₁ feed every ℓ(A, m) test; computed once rather than per degree tried.
        var abs = new double[data.Length];
        for (var i = 0; i < abs.Length; i++) abs[i] = Math.Abs(data[i]);
        var norm1 = LinalgKernels.OneNorm(data, n, n);

        var a2 = LinalgKernels.Mul(data, data, n, n, n);
        var a4 = LinalgKernels.Mul(a2, a2, n, n, n);
        var a6 = LinalgKernels.Mul(a2, a4, n, n, n);
        var d4 = Math.Pow(LinalgKernels.OneNorm(a4, n, n), 0.25);
        var d6 = Math.Pow(LinalgKernels.OneNorm(a6, n, n), 1.0 / 6.0);
        var eta1 = Math.Max(d4, d6);

        if (eta1 <= 1.495585217958292e-2 && Ell(abs, norm1, n, 3, 0) == 0)
            return PadeSolve(data, [a2], Pade3, n, triangular);
        if (eta1 <= 2.539398330063230e-1 && Ell(abs, norm1, n, 5, 0) == 0)
            return PadeSolve(data, [a2, a4], Pade5, n, triangular);

        // A⁸ is kept when it was formed exactly: degree 9 needs it as a term.
        var a8 = exact ? LinalgKernels.Mul(a4, a4, n, n, n) : null;
        var d8 = Math.Pow(a8 is not null ? LinalgKernels.OneNorm(a8, n, n) : PowerNormEstimate(n, a4, a4), 1.0 / 8.0);
        var eta3 = Math.Max(d6, d8);
        if (eta3 <= 9.504178996162932e-1 && Ell(abs, norm1, n, 7, 0) == 0)
            return PadeSolve(data, [a2, a4, a6], Pade7, n, triangular);
        if (eta3 <= 2.097847961257068 && Ell(abs, norm1, n, 9, 0) == 0)
            return PadeSolve(data, [a2, a4, a6, a8 ?? LinalgKernels.Mul(a4, a4, n, n, n)], Pade9, n, triangular);

        var d10 = Math.Pow(exact ? LinalgKernels.OneNorm(LinalgKernels.Mul(a4, a6, n, n, n), n, n) : PowerNormEstimate(n, a4, a6), 0.1);
        var eta4 = Math.Max(d8, d10);
        var eta5 = Math.Min(eta3, eta4);
        const double theta13 = 4.25;
        var s = eta5 == 0 ? 0 : Math.Max((int)Math.Ceiling(Math.Log2(eta5 / theta13)), 0);
        s += Ell(abs, norm1, n, 13, s);
        if (s > 1000) throw new LinalgException("Expm: scaling exponent overflow; the matrix norm is too large.");

        // The scaling 2^-s is folded into the Padé coefficients and the products' alpha instead of
        // copying four scaled matrices: a power of two scales exactly, so the result is the same.
        var c1 = Math.Pow(2.0, -s);
        var c2 = c1 * c1;
        var c4 = c2 * c2;
        var c6 = c4 * c2;
        var b = Pade13;
        var w1 = Lin3(b[13] * c6, a6, b[11] * c4, a4, b[9] * c2, a2);
        var w2 = Lin3(b[7] * c6, a6, b[5] * c4, a4, b[3] * c2, a2);
        LinalgKernels.AddDiagonal(w2, n, b[1]);
        var z1 = Lin3(b[12] * c6, a6, b[10] * c4, a4, b[8] * c2, a2);
        var v = Lin3(b[6] * c6, a6, b[4] * c4, a4, b[2] * c2, a2);
        LinalgKernels.AddDiagonal(v, n, b[0]);
        var u = new double[n * n];
        unsafe
        {
            fixed (double* p6 = a6, pw1 = w1, pw2 = w2, pz1 = z1, pv = v, pa = data, pu = u)
            {
                LinalgGemm.Gemm(n, n, n, c6, p6, n, pw1, n, 1.0, pw2, n);   // W = A6s W1 + W2
                LinalgGemm.Gemm(n, n, n, c1, pa, n, pw2, n, 0.0, pu, n);    // U = As W
                LinalgGemm.Gemm(n, n, n, c6, p6, n, pz1, n, 1.0, pv, n);    // V = A6s Z1 + Z2
            }
        }
        return Square(SolvePq(u, v, n), s, n, triangular, data);
    }

    private static NdArray PadeSolve(double[] a, double[][] evenPowers, double[] b, int n, bool triangular)
    {
        // U = A * sum b_{2k+1} A^{2k}, V = sum b_{2k} A^{2k}.
        var uInner = new double[n * n];
        var v = new double[n * n];
        LinalgKernels.AddDiagonal(uInner, n, b[1]);
        LinalgKernels.AddDiagonal(v, n, b[0]);
        for (var k = 0; k < evenPowers.Length; k++)
        {
            LinAlg.AxpySpan(b[2 * k + 3], evenPowers[k], uInner);
            LinAlg.AxpySpan(b[2 * k + 2], evenPowers[k], v);
        }
        var u = LinalgKernels.Mul(a, uInner, n, n, n);
        return Square(SolvePq(u, v, n), 0, n, triangular, a);
    }

    /// <summary>R = (V - U)⁻¹ (V + U), overwriting both buffers; returns R.</summary>
    private static double[] SolvePq(double[] u, double[] v, int n)
    {
        // Q = V - U into u's buffer, P = V + U into v's.
        for (var i = 0; i < u.Length; i++)
        {
            var ui = u[i];
            var vi = v[i];
            u[i] = vi - ui;
            v[i] = vi + ui;
        }
        LinalgKernels.RequireFinite(u, nameof(Solve));
        // Deliberately managed even when a LAPACK is installed: numpy/scipy's OpenBLAS dgesv
        // measured 60 ms against 7 ms for this LU + blocked solve at n = 400 (row-major LAPACKE
        // transposes in and out, and its threads compete with .NET's pool).
        var swaps = new int[n];
        var zero = LinalgKernels.LuInPlace(u, n, swaps);
        if (zero >= 0) throw new LinalgSingularMatrixException($"Solve: matrix is exactly singular (zero pivot at {zero}).");
        LinalgKernels.LuSolveInPlace(u, swaps, n, v, n, transpose: false);
        return v;
    }

    private static NdArray Square(double[] r, int s, int n, bool triangular, double[] t)
    {
        if (triangular) FixTriangular(r, t, n, Math.Pow(2.0, -s));
        for (var i = s - 1; i >= 0; i--)
        {
            r = LinalgKernels.Mul(r, r, n, n, n);
            if (triangular) FixTriangular(r, t, n, Math.Pow(2.0, -i));
        }
        return new NdArray(r, n, n);
    }

    /// <summary>Exact diagonal and superdiagonal of exp(scale·T) for upper-triangular T (Higham eq. 10.42).</summary>
    private static void FixTriangular(Span<double> x, double[] t, int n, double scale)
    {
        for (var k = 0; k < n; k++) x[k * n + k] = Math.Exp(scale * t[k * n + k]);
        for (var k = 0; k < n - 1; k++)
        {
            var l1 = scale * t[k * n + k];
            var l2 = scale * t[(k + 1) * n + k + 1];
            var t12 = scale * t[k * n + k + 1];
            var half = (l2 - l1) / 2.0;
            var sinch = half == 0.0 ? 1.0 : Math.Sinh(half) / half;
            x[k * n + k + 1] = t12 * Math.Exp((l1 + l2) / 2.0) * sinch;
        }
    }

    /// <summary>
    /// The ℓ(2⁻ˢA, m) correction of Al-Mohy–Higham (2009, eq. 5.1): extra squarings needed so the
    /// Padé backward-error bound holds, from the exact 1-norm of |A|^{2m+1} (a non-negative
    /// matrix, whose 1-norm is the largest entry of 1ᵀ|A|^{2m+1} — n² work per power).
    /// </summary>
    /// <param name="abs">|A| element-wise, row-major.</param>
    /// <param name="norm1">‖A‖₁.</param>
    /// <param name="n">Order.</param>
    /// <param name="m">Padé degree.</param>
    /// <param name="s">Squarings already chosen.</param>
    private static unsafe int Ell(double[] abs, double norm1, int n, int m, int s)
    {
        double c = m switch
        {
            3 => 100800.0,
            5 => 10059033600.0,
            7 => 4487938430976000.0,
            9 => 5914384781877410214400.0,
            _ => 113250775606021113483283660800000000.0,
        };
        if (norm1 == 0.0) return 0;
        var p = 2 * m + 1;
        var v = new double[n];
        Array.Fill(v, 1.0);
        var next = new double[n];
        var logScale = 0.0; // keep v normalised so |A|^27 cannot overflow
        fixed (double* pa = abs, pn = next)
        {
            for (var k = 0; k < p; k++)
            {
                Array.Clear(next);
                // Four rows per pass over `next`: a quarter of the loads and stores of plain axpys.
                var i = 0;
                for (; i + 4 <= n; i += 4)
                    AbsRowsAccumulate(pa + (long)i * n, n, v[i], v[i + 1], v[i + 2], v[i + 3], pn);
                for (; i < n; i++)
                    if (v[i] != 0.0) LinalgBlocked.Axpy(v[i], pa + (long)i * n, pn, n);
                var mx = 0.0;
                foreach (var x in next) mx = Math.Max(mx, x);
                if (mx == 0.0) return 0;
                for (var q = 0; q < n; q++) v[q] = next[q] / mx;
                logScale += Math.Log2(mx);
            }
        }
        // log2(alpha) with alpha = ||2^-s|A||^p||_1 / (||2^-s A||_1 * c)
        var log2Alpha = logScale - p * s - (Math.Log2(norm1) - s) - Math.Log2(c);
        var value = (int)Math.Ceiling((log2Alpha + 53.0) / (2.0 * m));
        return Math.Max(value, 0);
    }

    /// <summary><c>y += v0 r0 + v1 r1 + v2 r2 + v3 r3</c> for four consecutive rows of stride n.</summary>
    private static unsafe void AbsRowsAccumulate(double* r0, int n, double v0, double v1, double v2, double v3, double* y)
    {
        var r1 = r0 + n; var r2 = r1 + n; var r3 = r2 + n;
        var w0 = System.Runtime.Intrinsics.Vector256.Create(v0);
        var w1 = System.Runtime.Intrinsics.Vector256.Create(v1);
        var w2 = System.Runtime.Intrinsics.Vector256.Create(v2);
        var w3 = System.Runtime.Intrinsics.Vector256.Create(v3);
        var j = 0;
        for (; j <= n - 4; j += 4)
        {
            var acc = System.Runtime.Intrinsics.Vector256.Load(y + j)
                + w0 * System.Runtime.Intrinsics.Vector256.Load(r0 + j)
                + w1 * System.Runtime.Intrinsics.Vector256.Load(r1 + j)
                + w2 * System.Runtime.Intrinsics.Vector256.Load(r2 + j)
                + w3 * System.Runtime.Intrinsics.Vector256.Load(r3 + j);
            System.Runtime.Intrinsics.Vector256.Store(acc, y + j);
        }
        for (; j < n; j++) y[j] += v0 * r0[j] + v1 * r1[j] + v2 * r2[j] + v3 * r3[j];
    }

    /// <summary>
    /// Hager/Higham lower-bound estimate of ‖F₁F₂‖₁ for two commuting powers of A (A⁸ = A⁴A⁴,
    /// A¹⁰ = A⁴A⁶), using only matrix-vector products.
    /// </summary>
    /// <remarks>
    /// Applying the already-formed powers costs two matrix-vector products per application instead
    /// of p products with A itself: the estimate is of the same matrix, four to five times cheaper.
    /// </remarks>
    private static double PowerNormEstimate(int n, double[] f1, double[] f2)
    {
        double[] MatVec(double[] m, double[] x, bool transpose)
        {
            var z = new double[n];
            if (!transpose)
                for (var i = 0; i < n; i++) z[i] = LinAlg.InnerSpan(m.AsSpan(i * n, n), x);
            else
                for (var i = 0; i < n; i++) if (x[i] != 0.0) LinAlg.AxpySpan(x[i], m.AsSpan(i * n, n), z);
            return z;
        }
        double[] Apply(double[] x, bool transpose)
            => transpose ? MatVec(f2, MatVec(f1, x, true), true) : MatVec(f1, MatVec(f2, x, false), false);

        var xv = new double[n];
        Array.Fill(xv, 1.0 / n);
        var est = 0.0;
        var lastJ = -1;
        for (var iter = 0; iter < 5; iter++)
        {
            var y = Apply(xv, false);
            var newEst = 0.0;
            foreach (var t in y) newEst += Math.Abs(t);
            if (iter > 0 && newEst <= est) break;
            est = newEst;
            var xi = new double[n];
            for (var i = 0; i < n; i++) xi[i] = y[i] >= 0 ? 1.0 : -1.0;
            var z = Apply(xi, true);
            var j = 0;
            for (var i = 1; i < n; i++) if (Math.Abs(z[i]) > Math.Abs(z[j])) j = i;
            if (j == lastJ) break;
            lastJ = j;
            Array.Clear(xv);
            xv[j] = 1.0;
        }
        // dlacon's alternating-sign test vector guards against the estimator's known traps.
        var alt = new double[n];
        for (var i = 0; i < n; i++) alt[i] = (i % 2 == 0 ? 1 : -1) * (1.0 + (double)i / Math.Max(n - 1, 1));
        var ya = Apply(alt, false);
        var altEst = 0.0;
        foreach (var t in ya) altEst += Math.Abs(t);
        altEst = 2.0 * altEst / (3.0 * n);
        return Math.Max(est, altEst);
    }

    private static double[] Lin3(double c1, double[] x1, double c2, double[] x2, double c3, double[] x3)
    {
        var r = new double[x1.Length];
        for (var i = 0; i < r.Length; i++) r[i] = c1 * x1[i] + c2 * x2[i] + c3 * x3[i];
        return r;
    }

    private static NdArray Scaled(NdArray a, double f)
    {
        var src = a.AsContiguous().AsSpan();
        var r = new double[src.Length];
        for (var i = 0; i < r.Length; i++) r[i] = src[i] * f;
        return new NdArray(r, a.Shape.ToArray());
    }

    private static NdArray Lin(double c1, NdArray x1, double c2, NdArray x2, double c3, NdArray x3)
    {
        var a = x1.AsContiguous().AsSpan();
        var b = x2.AsContiguous().AsSpan();
        var c = x3.AsContiguous().AsSpan();
        var r = new double[a.Length];
        for (var i = 0; i < r.Length; i++) r[i] = c1 * a[i] + c2 * b[i] + c3 * c[i];
        return new NdArray(r, x1.Shape.ToArray());
    }

    // ================================================================ expm Frechet

    /// <summary>
    /// The matrix exponential and its Fréchet derivative in direction E, scipy's
    /// <c>expm_frechet</c>: <c>L(A, E) = d/dt exp(A + tE)|₀</c>.
    /// </summary>
    /// <remarks>
    /// Al-Mohy &amp; Higham (2009), "Computing the Fréchet derivative of the matrix exponential",
    /// Algorithm 6.4: Padé degree from ‖A‖₁ (Table 6.1 thresholds), differentiating U and V
    /// alongside, then the squaring phase <c>L ← R L + L R</c>. Same order of cost as about three
    /// expm's, against eight for the 2n block-matrix trick it replaces.
    /// </remarks>
    public static (NdArray Expm, NdArray Frechet) ExpmFrechet(NdArray a, NdArray e)
    {
        LinalgKernels.RequireSquare(a, nameof(ExpmFrechet));
        LinalgKernels.RequireSquare(e, nameof(ExpmFrechet));
        var n = a.Shape[0];
        if (e.Shape[0] != n) throw new ArgumentException("ExpmFrechet: a and e must have the same shape.");
        var A = a.AsContiguous().Copy();
        var E = e.AsContiguous().Copy();
        LinalgKernels.RequireFinite(A.AsSpan(), nameof(ExpmFrechet));
        var norm = LinalgKernels.OneNorm(A);
        var s = 0;
        double[] b;
        int m;
        if (norm <= 1.08e-2) { m = 3; b = Pade3; }
        else if (norm <= 2.00e-1) { m = 5; b = Pade5; }
        else if (norm <= 7.83e-1) { m = 7; b = Pade7; }
        else if (norm <= 1.78) { m = 9; b = Pade9; }
        else
        {
            m = 13; b = Pade13;
            s = Math.Max(0, (int)Math.Ceiling(Math.Log2(norm / 4.74)));
            var f = Math.Pow(2.0, -s);
            A = Scaled(A, f);
            E = Scaled(E, f);
        }

        var I = NdArray.Eye(n);
        NdArray U, V, Lu, Lv;
        var A2 = LinalgKernels.Mul(A, A);
        var M2 = LinalgKernels.Combine(1.0, LinalgKernels.Mul(A, E), 1.0, LinalgKernels.Mul(E, A));
        if (m == 13)
        {
            var A4 = LinalgKernels.Mul(A2, A2);
            var M4 = LinalgKernels.Combine(1.0, LinalgKernels.Mul(A2, M2), 1.0, LinalgKernels.Mul(M2, A2));
            var A6 = LinalgKernels.Mul(A2, A4);
            var M6 = LinalgKernels.Combine(1.0, LinalgKernels.Mul(A4, M2), 1.0, LinalgKernels.Mul(M4, A2));
            var W1 = Lin(b[13], A6, b[11], A4, b[9], A2);
            var W2 = Lin(b[7], A6, b[5], A4, b[3], A2); LinalgKernels.AddDiagonal(W2.AsSpan(), n, b[1]);
            var Z1 = Lin(b[12], A6, b[10], A4, b[8], A2);
            var Z2 = Lin(b[6], A6, b[4], A4, b[2], A2); LinalgKernels.AddDiagonal(Z2.AsSpan(), n, b[0]);
            var W = LinalgKernels.Combine(1.0, LinalgKernels.Mul(A6, W1), 1.0, W2);
            U = LinalgKernels.Mul(A, W);
            V = LinalgKernels.Combine(1.0, LinalgKernels.Mul(A6, Z1), 1.0, Z2);
            var Lw1 = Lin(b[13], M6, b[11], M4, b[9], M2);
            var Lw2 = Lin(b[7], M6, b[5], M4, b[3], M2);
            var Lz1 = Lin(b[12], M6, b[10], M4, b[8], M2);
            var Lz2 = Lin(b[6], M6, b[4], M4, b[2], M2);
            var Lw = Sum3(LinalgKernels.Mul(A6, Lw1), LinalgKernels.Mul(M6, W1), Lw2);
            Lu = LinalgKernels.Combine(1.0, LinalgKernels.Mul(A, Lw), 1.0, LinalgKernels.Mul(E, W));
            Lv = Sum3(LinalgKernels.Mul(A6, Lz1), LinalgKernels.Mul(M6, Z1), Lz2);
        }
        else
        {
            // Even powers and their derivatives: P_k = A^{2k}, M_k = L(A^{2k}).
            var powers = new List<NdArray> { I, A2 };
            var derivs = new List<NdArray> { NdArray.Zeros(n, n), M2 };
            while (powers.Count <= (m - 1) / 2)
            {
                var k = powers.Count;            // building A^{2k} = A^{2(k-1)} A^2
                var prev = powers[k - 1];
                var dprev = derivs[k - 1];
                powers.Add(LinalgKernels.Mul(prev, A2));
                derivs.Add(LinalgKernels.Combine(1.0, LinalgKernels.Mul(dprev, A2), 1.0, LinalgKernels.Mul(prev, M2)));
            }
            var uIn = NdArray.Zeros(n, n);
            var vv = NdArray.Zeros(n, n);
            var luIn = NdArray.Zeros(n, n);
            var lvv = NdArray.Zeros(n, n);
            for (var k = 0; k < powers.Count; k++)
            {
                LinAlg.AxpySpan(b[2 * k + 1], powers[k].AsSpan(), uIn.AsSpan());
                LinAlg.AxpySpan(b[2 * k], powers[k].AsSpan(), vv.AsSpan());
                LinAlg.AxpySpan(b[2 * k + 1], derivs[k].AsSpan(), luIn.AsSpan());
                LinAlg.AxpySpan(b[2 * k], derivs[k].AsSpan(), lvv.AsSpan());
            }
            U = LinalgKernels.Mul(A, uIn);
            V = vv;
            Lu = LinalgKernels.Combine(1.0, LinalgKernels.Mul(A, luIn), 1.0, LinalgKernels.Mul(E, uIn));
            Lv = lvv;
        }

        var q = LinalgKernels.Combine(-1.0, U, 1.0, V);
        var lu = LuFactor(q);
        var R = LuSolve(lu, LinalgKernels.Combine(1.0, U, 1.0, V));
        var rhs = LinalgKernels.Combine(1.0, LinalgKernels.Combine(1.0, Lu, 1.0, Lv), 1.0,
            LinalgKernels.Mul(LinalgKernels.Combine(1.0, Lu, -1.0, Lv), R)); // d(Q⁻¹P) = Q⁻¹(dP − dQ·R)
        var L = LuSolve(lu, rhs);
        for (var k = 0; k < s; k++)
        {
            L = LinalgKernels.Combine(1.0, LinalgKernels.Mul(R, L), 1.0, LinalgKernels.Mul(L, R));
            R = LinalgKernels.Mul(R, R);
        }
        return (R, L);
    }

    private static NdArray Sum3(NdArray x, NdArray y, NdArray z)
    {
        var r = LinalgKernels.Combine(1.0, x, 1.0, y);
        LinAlg.AxpySpan(1.0, z.AsContiguous().AsSpan(), r.AsSpan());
        return r;
    }

    // ================================================================ Schur-based functions

    private static readonly double[] LogPadeTheta =
    [
        3.65e-8, 3.759e-4, 8.196e-3, 3.783e-2, 9.290e-2, 1.6563e-1, 2.4572e-1, 3.2530e-1,
        3.9976e-1, 4.6699e-1, 5.2647e-1, 5.7848e-1, 6.2371e-1, 6.6296e-1, 6.9702e-1, 7.2663e-1,
    ];

    /// <summary>
    /// Principal matrix logarithm, scipy's <c>logm</c>, for real matrices whose principal logarithm
    /// is real — no eigenvalue on the closed negative real axis. Use <see cref="LogmComplex"/> otherwise.
    /// </summary>
    /// <remarks>
    /// Inverse scaling and squaring on the complex Schur form (Higham 2001, "Evaluating Padé
    /// approximants of the matrix logarithm"): square roots of the triangle until ‖T − I‖₁ is small,
    /// then the [m/m] Padé approximant of log(I + X) evaluated as its Gauss–Legendre partial-fraction
    /// sum, then multiplied back by 2ᵏ. Degree thresholds were recomputed for <em>relative</em> error
    /// ≤ u (the published ones bound absolute error, which is not what matters once the result is
    /// multiplied by 2ᵏ). An extra square root is taken only when it lowers the degree by more than
    /// one. Not the later Al-Mohy–Higham (2012) refinements; very ill-conditioned inputs may lose a
    /// few more digits than scipy.
    /// </remarks>
    /// <exception cref="LinalgException">A is singular, or its principal logarithm is not real.</exception>
    public static NdArray Logm(NdArray a)
    {
        var (f, n) = LogmCore(a, nameof(Logm));
        return RealOrThrow(f, n, nameof(Logm), "has an eigenvalue on the negative real axis; use LogmComplex");
    }

    /// <summary>Principal matrix logarithm as a complex matrix (any non-singular real input).</summary>
    public static ComplexNdArray LogmComplex(NdArray a)
    {
        var (f, n) = LogmCore(a, nameof(LogmComplex));
        return new ComplexNdArray(f, n, n);
    }

    private static (Complex[] F, int N) LogmCore(NdArray a, string caller)
    {
        LinalgKernels.RequireSquare(a, caller);
        var n = a.Shape[0];
        LinalgKernels.RequireFinite(a.AsContiguous().AsSpan(), caller);
        var (t, z) = LinalgSchurKernel.ComplexSchur(a);
        var l = LogTriangular(t, n, caller);
        return (LinalgKernels.CSandwich(z, l, n), n);
    }

    internal static Complex[] LogTriangular(Complex[] t, int n, string caller)
    {
        for (var i = 0; i < n; i++)
            if (t[i * n + i] == Complex.Zero) throw new LinalgSingularMatrixException($"{caller}: matrix is singular, its logarithm does not exist.");

        var k = 0;
        int m;
        var cur = (Complex[])t.Clone();
        while (true)
        {
            var x = DistanceFromIdentity(cur, n);
            if (x <= LogPadeTheta[^1])
            {
                m = DegreeFor(x);
                var mHalf = DegreeFor(x / 2.0);
                if (m - mHalf <= 1 || k >= 60) break;
            }
            if (k >= 100) throw new LinalgException($"{caller}: square roots failed to approach the identity.");
            cur = SqrtTriangular(cur, n, caller);
            k++;
        }

        // log(I + X) ≈ sum_j w_j X (I + x_j X)^{-1}, Gauss–Legendre on [0, 1].
        var xm = (Complex[])cur.Clone();
        for (var i = 0; i < n; i++) xm[i * n + i] -= 1.0;
        var (nodes, weights) = GaussLegendre01(m);
        var result = new Complex[n * n];
        var col = new Complex[n];
        var shifted = new Complex[n * n];
        for (var q = 0; q < m; q++)
        {
            // (I + x_j X) Y = X  ->  Y = (I + x_j X)^{-1} X, which commutes with X.
            for (var i = 0; i < n * n; i++) shifted[i] = nodes[q] * xm[i];
            for (var i = 0; i < n; i++) shifted[i * n + i] += 1.0;
            for (var c = 0; c < n; c++)
            {
                for (var i = 0; i < n; i++) col[i] = xm[i * n + c];
                LinalgKernels.CUpperSolve(shifted, n, col, Complex.Zero);
                for (var i = 0; i < n; i++) result[i * n + c] += weights[q] * col[i];
            }
        }
        var scale = Math.Pow(2.0, k);
        for (var i = 0; i < result.Length; i++) result[i] *= scale;
        return result;
    }

    private static int DegreeFor(double x)
    {
        for (var i = 0; i < LogPadeTheta.Length; i++) if (x <= LogPadeTheta[i]) return i + 1;
        return LogPadeTheta.Length;
    }

    private static double DistanceFromIdentity(Complex[] t, int n)
    {
        var best = 0.0;
        for (var j = 0; j < n; j++)
        {
            var s = 0.0;
            for (var i = 0; i <= j; i++) s += Complex.Abs(i == j ? t[i * n + j] - 1.0 : t[i * n + j]);
            best = Math.Max(best, s);
        }
        return best;
    }

    private static (double[] Nodes, double[] Weights) GaussLegendre01(int m)
    {
        var nodes = new double[m];
        var weights = new double[m];
        for (var i = 0; i < m; i++)
        {
            var z = Math.Cos(Math.PI * (i + 0.75) / (m + 0.5));
            double pp = 0;
            for (var it = 0; it < 100; it++)
            {
                double p1 = 1, p2 = 0;
                for (var j = 1; j <= m; j++)
                {
                    var p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
                }
                pp = m * (z * p1 - p2) / (z * z - 1.0);
                var z1 = z;
                z = z1 - p1 / pp;
                if (Math.Abs(z - z1) < 1e-16) break;
            }
            nodes[i] = (1.0 - z) / 2.0;
            weights[i] = 1.0 / ((1.0 - z * z) * pp * pp);
        }
        return (nodes, weights);
    }

    /// <summary>
    /// Principal square root of an upper-triangular complex matrix (Björck–Hammarling recurrence).
    /// </summary>
    internal static Complex[] SqrtTriangular(Complex[] t, int n, string caller)
    {
        var r = new Complex[n * n];
        for (var i = 0; i < n; i++) r[i * n + i] = Complex.Sqrt(t[i * n + i]);
        var tnorm = 0.0;
        foreach (var v in t) tnorm = Math.Max(tnorm, Complex.Abs(v));
        for (var j = 1; j < n; j++)
            for (var i = j - 1; i >= 0; i--)
            {
                var s = t[i * n + j];
                for (var k = i + 1; k < j; k++) s -= r[i * n + k] * r[k * n + j];
                var den = r[i * n + i] + r[j * n + j];
                if (den == Complex.Zero)
                {
                    if (Complex.Abs(s) <= 1e3 * LinalgKernels.Eps * Math.Max(tnorm, 1e-300)) { r[i * n + j] = 0; continue; }
                    throw new LinalgException($"{caller}: the matrix is singular with a non-trivial Jordan block at 0 and has no square root.");
                }
                r[i * n + j] = s / den;
            }
        return r;
    }

    /// <summary>
    /// Principal matrix square root, scipy's <c>sqrtm</c>, for real matrices whose principal root is
    /// real (no negative real eigenvalues). Use <see cref="SqrtmComplex"/> otherwise.
    /// </summary>
    /// <remarks>
    /// Schur method: the complex Schur form (converted from the real Schur form), the Björck–Hammarling
    /// recurrence on the triangle, and back-transformation — not Denman–Beavers. The real Schur
    /// quasi-triangular variant (Higham 1987) would avoid complex arithmetic; this one costs about
    /// four times the flops of that in the recurrence but shares its accuracy. A real root is
    /// recovered as the real part, after checking the imaginary part is rounding-level.
    /// </remarks>
    public static NdArray Sqrtm(NdArray a)
    {
        var (f, n) = SqrtmCore(a, nameof(Sqrtm));
        return RealOrThrow(f, n, nameof(Sqrtm), "has a negative real eigenvalue; use SqrtmComplex");
    }

    /// <summary>Principal matrix square root as a complex matrix.</summary>
    public static ComplexNdArray SqrtmComplex(NdArray a)
    {
        var (f, n) = SqrtmCore(a, nameof(SqrtmComplex));
        return new ComplexNdArray(f, n, n);
    }

    private static (Complex[] F, int N) SqrtmCore(NdArray a, string caller)
    {
        LinalgKernels.RequireSquare(a, caller);
        var n = a.Shape[0];
        LinalgKernels.RequireFinite(a.AsContiguous().AsSpan(), caller);
        var (t, z) = LinalgSchurKernel.ComplexSchur(a);
        return (LinalgKernels.CSandwich(z, SqrtTriangular(t, n, caller), n), n);
    }

    private static NdArray RealOrThrow(Complex[] f, int n, string caller, string reason)
    {
        if (LinalgKernels.RelativeImaginary(f) > 1e-8)
            throw new LinalgException($"{caller}: the principal result is not real — the matrix {reason}.");
        return new NdArray(LinalgKernels.RealPart(f), n, n);
    }

    /// <summary>
    /// Fractional matrix power <c>A^t</c>, scipy's <c>fractional_matrix_power</c>, real case.
    /// Integer t uses repeated squaring (and the inverse for negative t).
    /// </summary>
    /// <remarks>
    /// Non-integer t is computed as <c>exp(t·log A)</c> from <see cref="LogmComplex"/> and
    /// <see cref="Expm"/>, not by the Higham–Lin Schur–Padé algorithm scipy uses; its error grows
    /// with ‖t·log A‖, so expect a few digits fewer than scipy when that is large.
    /// </remarks>
    public static NdArray FractionalMatrixPower(NdArray a, double t)
    {
        LinalgKernels.RequireSquare(a, nameof(FractionalMatrixPower));
        if (t == Math.Floor(t) && Math.Abs(t) < int.MaxValue)
        {
            var p = (int)t;
            return p >= 0 ? LinAlg.MatrixPower(a, p) : LinAlg.MatrixPower(Inv(a), -p);
        }
        var f = FractionalPowerCore(a, t);
        return RealOrThrow(f, a.Shape[0], nameof(FractionalMatrixPower), "has a negative real eigenvalue; use FractionalMatrixPowerComplex");
    }

    /// <summary>Fractional matrix power as a complex matrix (principal branch).</summary>
    public static ComplexNdArray FractionalMatrixPowerComplex(NdArray a, double t)
    {
        LinalgKernels.RequireSquare(a, nameof(FractionalMatrixPowerComplex));
        var n = a.Shape[0];
        return new ComplexNdArray(FractionalPowerCore(a, t), n, n);
    }

    private static Complex[] FractionalPowerCore(NdArray a, double t)
    {
        var (log, n) = LogmCore(a, nameof(FractionalMatrixPower));
        for (var i = 0; i < log.Length; i++) log[i] *= t;
        if (LinalgKernels.RelativeImaginary(log) <= 1e-12)
        {
            var e = Expm(new NdArray(LinalgKernels.RealPart(log), n, n));
            return LinalgKernels.ToComplex(e.AsSpan());
        }
        // Complex exponential through the real 2n embedding [[Re, -Im], [Im, Re]].
        var big = Expm(LinalgKernels.RealEmbedding(log, n));
        return LinalgKernels.FromRealEmbedding(big.AsSpan(), n);
    }

    /// <summary>
    /// <c>f(A)</c> for a general real matrix by the Schur–Parlett recurrence on the complex Schur
    /// form, scipy's <c>funm</c>; the result must be real (use <see cref="FunmComplex"/> otherwise).
    /// </summary>
    /// <remarks>
    /// The plain Parlett recurrence divides by <c>T[j,j] − T[i,i]</c>, so it needs <em>distinct</em>
    /// eigenvalues and becomes inaccurate as two approach each other — the same limitation as
    /// scipy's <c>funm</c>, which is not the blocked Davies–Higham algorithm either. An exactly
    /// repeated eigenvalue throws. For symmetric matrices use <see cref="FunmSymmetric"/>, which
    /// has no such limit; for exp/log/sqrt use the dedicated functions.
    /// </remarks>
    public static NdArray Funm(NdArray a, Func<Complex, Complex> f)
    {
        var (r, n) = FunmCore(a, f, nameof(Funm));
        return RealOrThrow(r, n, nameof(Funm), "maps to a non-real result under f; use FunmComplex");
    }

    /// <summary>Schur–Parlett <c>f(A)</c> as a complex matrix.</summary>
    public static ComplexNdArray FunmComplex(NdArray a, Func<Complex, Complex> f)
    {
        var (r, n) = FunmCore(a, f, nameof(FunmComplex));
        return new ComplexNdArray(r, n, n);
    }

    private static (Complex[] F, int N) FunmCore(NdArray a, Func<Complex, Complex> f, string caller)
    {
        ArgumentNullException.ThrowIfNull(f);
        LinalgKernels.RequireSquare(a, caller);
        var n = a.Shape[0];
        var (t, z) = LinalgSchurKernel.ComplexSchur(a);
        var F = new Complex[n * n];
        for (var i = 0; i < n; i++) F[i * n + i] = f(t[i * n + i]);
        for (var p = 1; p < n; p++)
            for (var i = 0; i + p < n; i++)
            {
                var j = i + p;
                var s = t[i * n + j] * (F[j * n + j] - F[i * n + i]);
                for (var k = i + 1; k < j; k++)
                    s += t[i * n + k] * F[k * n + j] - F[i * n + k] * t[k * n + j];
                var den = t[j * n + j] - t[i * n + i];
                if (den == Complex.Zero)
                {
                    if (s == Complex.Zero && t[i * n + j] == Complex.Zero) { F[i * n + j] = 0; continue; }
                    throw new LinalgException($"{caller}: repeated eigenvalue; the Parlett recurrence is undefined. Use FunmSymmetric or a dedicated function.");
                }
                F[i * n + j] = s / den;
            }
        return (LinalgKernels.CSandwich(z, F, n), n);
    }

    /// <summary>
    /// <c>f(A)</c> for a symmetric matrix as <c>V diag(f(λ)) Vᵀ</c>. Exact up to the eigensolver's
    /// accuracy for any f and any eigenvalue multiplicity. Only the lower triangle is read.
    /// </summary>
    public static NdArray FunmSymmetric(NdArray a, Func<double, double> f)
    {
        ArgumentNullException.ThrowIfNull(f);
        var e = Eigh(a);
        var n = a.Shape[0];
        var v = e.Vectors.AsContiguous().AsSpan();
        var scaled = new double[n * n];
        var fv = new double[n];
        for (var j = 0; j < n; j++) fv[j] = f(e.Values[j]);
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++) scaled[i * n + j] = v[i * n + j] * fv[j];
        return LinalgKernels.Mul(new NdArray(scaled, n, n), e.Vectors.T);
    }

    // ================================================================ trig / hyperbolic

    /// <summary>
    /// Matrix cosine, scipy's <c>cosm</c>. Computed in real arithmetic as the top-left block of
    /// <c>exp([[0, A], [−A, 0]]) = [[cos A, sin A], [−sin A, cos A]]</c> — scipy's
    /// <c>Re(expm(iA))</c> without complex arithmetic.
    /// </summary>
    public static NdArray Cosm(NdArray a) => CosSin(a, nameof(Cosm)).Cos;

    /// <summary>Matrix sine, scipy's <c>sinm</c> (see <see cref="Cosm"/>).</summary>
    public static NdArray Sinm(NdArray a) => CosSin(a, nameof(Sinm)).Sin;

    /// <summary>Matrix tangent <c>cos(A)⁻¹ sin(A)</c>, scipy's <c>tanm</c>.</summary>
    public static NdArray Tanm(NdArray a)
    {
        var (c, s) = CosSin(a, nameof(Tanm));
        return Solve(c, s, LinalgMatrixStructure.General);
    }

    /// <summary>Matrix hyperbolic cosine <c>(e^A + e^{−A})/2</c>, scipy's <c>coshm</c>.</summary>
    public static NdArray Coshm(NdArray a)
    {
        var (ep, em) = ExpPair(a, nameof(Coshm));
        return LinalgKernels.Combine(0.5, ep, 0.5, em);
    }

    /// <summary>Matrix hyperbolic sine <c>(e^A − e^{−A})/2</c>, scipy's <c>sinhm</c>.</summary>
    public static NdArray Sinhm(NdArray a)
    {
        var (ep, em) = ExpPair(a, nameof(Sinhm));
        return LinalgKernels.Combine(0.5, ep, -0.5, em);
    }

    /// <summary>Matrix hyperbolic tangent <c>cosh(A)⁻¹ sinh(A)</c>, scipy's <c>tanhm</c>.</summary>
    public static NdArray Tanhm(NdArray a)
    {
        var (ep, em) = ExpPair(a, nameof(Tanhm));
        return Solve(LinalgKernels.Combine(0.5, ep, 0.5, em), LinalgKernels.Combine(0.5, ep, -0.5, em), LinalgMatrixStructure.General);
    }

    private static (NdArray Cos, NdArray Sin) CosSin(NdArray a, string caller)
    {
        LinalgKernels.RequireSquare(a, caller);
        var n = a.Shape[0];
        var src = a.AsContiguous().AsSpan();
        var m = 2 * n;
        var big = new double[m * m];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                big[i * m + n + j] = src[i * n + j];
                big[(n + i) * m + j] = -src[i * n + j];
            }
        var e = Expm(new NdArray(big, m, m)).AsSpan();
        var c = new double[n * n];
        var s = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                c[i * n + j] = e[i * m + j];
                s[i * n + j] = e[i * m + n + j];
            }
        return (new NdArray(c, n, n), new NdArray(s, n, n));
    }

    private static (NdArray Plus, NdArray Minus) ExpPair(NdArray a, string caller)
    {
        LinalgKernels.RequireSquare(a, caller);
        return (Expm(a), Expm(Scaled(a.AsContiguous(), -1.0)));
    }

    /// <summary>
    /// Matrix sign function, scipy's <c>signm</c>: <c>sign(A)</c> maps each eigenvalue to ±1 by
    /// the sign of its real part.
    /// </summary>
    /// <remarks>
    /// Newton iteration <c>X ← (μX + (μX)⁻¹)/2</c> with determinant scaling
    /// <c>μ = |det X|^{−1/n}</c> (Byers 1987; Higham, Functions of Matrices §5.5), switched off once
    /// close so the final steps converge quadratically. Undefined when A is singular or has an
    /// eigenvalue on the imaginary axis; the iteration then fails to converge and this throws.
    /// </remarks>
    public static NdArray Signm(NdArray a)
    {
        LinalgKernels.RequireSquare(a, nameof(Signm));
        return SignNewton(a.AsContiguous().Copy(), nameof(Signm));
    }

    internal static NdArray SignNewton(NdArray x, string caller)
    {
        var n = x.Shape[0];
        var scaling = true;
        for (var iter = 0; iter < 100; iter++)
        {
            var (lu, swaps, zero) = LinalgKernels.LuFactor(x);
            if (zero >= 0) throw new LinalgSingularMatrixException($"{caller}: singular iterate; the matrix has an eigenvalue on the imaginary axis or is singular.");
            var mu = 1.0;
            if (scaling)
            {
                var logDet = 0.0;
                for (var i = 0; i < n; i++) logDet += Math.Log(Math.Abs(lu[i * n + i]));
                mu = Math.Exp(-logDet / n);
            }
            var inv = LinalgKernels.IdentityBuffer(n);
            LinalgKernels.LuSolveInPlace(lu, swaps, n, inv, n, transpose: false);
            var xs = x.AsSpan();
            var next = new double[n * n];
            for (var i = 0; i < next.Length; i++) next[i] = 0.5 * (mu * xs[i] + inv[i] / mu);
            var diff = 0.0;
            for (var i = 0; i < next.Length; i++) diff = Math.Max(diff, Math.Abs(next[i] - xs[i]));
            var size = 0.0;
            foreach (var v in next) size = Math.Max(size, Math.Abs(v));
            x = new NdArray(next, n, n);
            if (diff <= 1e-2 * size) scaling = false;
            if (diff <= 10 * n * LinalgKernels.Eps * size) return x;
        }
        throw new LinalgException($"{caller}: Newton iteration did not converge (eigenvalue on or near the imaginary axis?).");
    }
}
