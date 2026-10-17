using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.LinalgAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// The level-3 kernels behind the dense routines (packed GEMM, recursive LU, striped triangular
/// solves, split complex products, recursive triangular Sylvester), each checked against the
/// plain implementation it replaced or a naive loop written here, and the routines that use them
/// checked against independent identities at sizes large enough to take the blocked and parallel
/// paths.
/// </summary>
public class LinalgKernelPathTests
{
    private static double[] Random(int count, int seed, double scale = 1.0)
    {
        var r = new GraviRandom(seed);
        var a = new double[count];
        for (var i = 0; i < count; i++) a[i] = (r.NextDouble() * 2 - 1) * scale;
        return a;
    }

    private static double[] NaiveMul(double[] a, int lda, double[] b, int ldb, int m, int k, int n)
    {
        var c = new double[m * n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
            {
                var s = 0.0;
                for (var p = 0; p < k; p++) s += a[i * lda + p] * b[p * ldb + j];
                c[i * n + j] = s;
            }
        return c;
    }

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(3, 5, 7)]
    [InlineData(17, 9, 33)]
    [InlineData(64, 64, 64)]
    [InlineData(130, 300, 70)]   // ragged tiles, more than one k slice, parallel
    [InlineData(5, 600, 2100)]   // wide: columns split across tasks and Nc panels
    public void Gemm_MatchesNaiveProductWithAlphaBetaAndStrides(int m, int k, int n)
    {
        // Operands embedded in larger buffers so the leading dimensions differ from the widths.
        int lda = k + 3, ldb = n + 5, ldc = n + 2;
        var a = Random(m * lda, 1);
        var b = Random(k * ldb, 2);
        var c = Random(m * ldc, 3);
        var c0 = (double[])c.Clone();
        const double alpha = -1.5, beta = 0.25;
        unsafe
        {
            fixed (double* pa = a, pb = b, pc = c)
                LinalgGemm.Gemm(m, n, k, alpha, pa, lda, pb, ldb, beta, pc, ldc);
        }
        var ab = NaiveMul(a, lda, b, ldb, m, k, n);
        for (var i = 0; i < m; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var expected = alpha * ab[i * n + j] + beta * c0[i * ldc + j];
                Assert.True(Math.Abs(expected - c[i * ldc + j]) <= 1e-12 * Math.Max(1, k),
                    $"C[{i},{j}] = {c[i * ldc + j]:R}, expected {expected:R}");
            }
            for (var j = n; j < ldc; j++) Assert.Equal(c0[i * ldc + j], c[i * ldc + j]); // padding untouched
        }
    }

    [Fact]
    public void Gemm_BetaZeroIgnoresNaNInC()
    {
        var a = Random(20 * 20, 4);
        var c = new double[400];
        Array.Fill(c, double.NaN);
        unsafe
        {
            fixed (double* pa = a, pc = c) LinalgGemm.Gemm(20, 20, 20, 1.0, pa, 20, pa, 20, 0.0, pc, 20);
        }
        Close(NaiveMul(a, 20, a, 20, 20, 20, 20), c, 1e-13, "beta = 0");
    }

    [Theory]
    [InlineData(64)]
    [InlineData(97)]
    [InlineData(257)]
    public void RecursiveLu_SamePivotsAsUnblockedAndReconstructs(int n)
    {
        var a = Random(n * n, 10 + n);
        var blocked = (double[])a.Clone();
        var plain = (double[])a.Clone();
        var sb = new int[n];
        var sp = new int[n];
        Assert.Equal(-1, LinalgBlocked.Lu(blocked, n, sb));
        Assert.Equal(-1, LinalgKernels.LuInPlaceUnblocked(plain, n, sp));
        Assert.Equal(sp, sb);
        Close(plain, blocked, 1e-11, "packed LU");

        // P A = L U with P replayed from the swaps.
        var perm = LinalgKernels.SwapsToPermutation(sb);
        var l = new double[n * n];
        var u = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                if (j < i) l[i * n + j] = blocked[i * n + j];
                else u[i * n + j] = blocked[i * n + j];
                if (i == j) l[i * n + j] = 1.0;
            }
        var lu = NaiveMul(l, n, u, n, n, n, n);
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
                Assert.True(Math.Abs(lu[i * n + j] - a[perm[i] * n + j]) <= 1e-12 * n, $"(PA - LU)[{i},{j}]");
    }

    [Fact]
    public void RecursiveLu_ReportsFirstZeroPivotLikeUnblocked()
    {
        const int n = 80;
        var a = Random(n * n, 21);
        // Column 30 equal to column 10 times two: rank deficient, exact zero pivot at the end.
        for (var i = 0; i < n; i++) a[i * n + 30] = 2 * a[i * n + 10];
        for (var j = 0; j < n; j++) { a[5 * n + j] = 0; }
        var blocked = (double[])a.Clone();
        var plain = (double[])a.Clone();
        var zb = LinalgBlocked.Lu(blocked, n, new int[n]);
        var zp = LinalgKernels.LuInPlaceUnblocked(plain, n, new int[n]);
        Assert.True(zb >= 0 && zp >= 0, $"zero pivots {zb}, {zp}");
        Assert.Throws<LinalgSingularMatrixException>(() => Linalg.Solve(new NdArray(a, n, n), new NdArray(Random(n * 20, 22), n, 20), LinalgMatrixStructure.General));
    }

    [Theory]
    [InlineData(70, 1)]
    [InlineData(70, 40)]
    [InlineData(150, 150)]
    [InlineData(200, 33)]
    public void BlockedTriangularSolves_MatchUnblocked(int n, int k)
    {
        var t = Random(n * n, 30);
        for (var i = 0; i < n; i++) t[i * n + i] = 3.0 + i % 5; // well conditioned diagonal
        var b = Random(n * k, 31);
        var lb = (double[])b.Clone();
        var lp = (double[])b.Clone();
        LinalgBlocked.LowerUnitSolve(t, n, lb, k);
        LinalgKernels.TriangularSolveInPlace(t, n, lp, k, lower: true, transpose: false, unitDiagonal: true);
        Close(lp, lb, 1e-10, "unit lower");
        var ub = (double[])b.Clone();
        var up = (double[])b.Clone();
        LinalgBlocked.UpperSolve(t, n, ub, k);
        LinalgKernels.TriangularSolveInPlace(t, n, up, k, lower: false, transpose: false, unitDiagonal: false);
        Close(up, ub, 1e-10, "upper");
    }

    [Fact]
    public void Solve_ManyRightHandSides_ResidualSmall()
    {
        const int n = 180, k = 120;
        var a = new NdArray(Random(n * n, 40), n, n);
        var b = new NdArray(Random(n * k, 41), n, k);
        var x = Linalg.Solve(a, b, LinalgMatrixStructure.General);
        Assert.True(MaxAbs(Sub(Mul(a, x), b)) <= 1e-10, "A X - B");
        var inv = Linalg.Inv(a);
        Identity(Mul(a, inv), 1e-10, "A A⁻¹");
    }

    [Theory]
    [InlineData(3, 4, 5)]
    [InlineData(40, 33, 50)]
    public void ComplexProduct_SplitMatchesNaive(int m, int k, int n)
    {
        var re = Random(m * k + k * n, 50);
        var im = Random(m * k + k * n, 51);
        var a = new Complex[m * k];
        var b = new Complex[k * n];
        for (var i = 0; i < a.Length; i++) a[i] = new Complex(re[i], im[i]);
        for (var i = 0; i < b.Length; i++) b[i] = new Complex(re[a.Length + i], im[a.Length + i]);
        var fast = LinalgKernels.CMul(a, b, m, k, n);
        var naive = LinalgKernels.CMulNaive(a, b, m, k, n);
        for (var i = 0; i < fast.Length; i++)
            Assert.True((fast[i] - naive[i]).Magnitude <= 1e-12 * k, $"C[{i}]");
    }

    [Theory]
    [InlineData(70, 45)]
    [InlineData(20, 90)]
    [InlineData(1, 40)]
    public void Sylvester_RecursiveSolve_ResidualSmall(int m, int n)
    {
        var a = new NdArray(Random(m * m, 60), m, m);
        var bData = Random(n * n, 61);
        for (var i = 0; i < n; i++) bData[i * n + i] += 4.0; // keep −eig(B) away from eig(A)
        var b = new NdArray(bData, n, n);
        var q = new NdArray(Random(m * n, 62), m, n);
        var x = Linalg.SolveSylvester(a, b, q);
        var residual = Sub(Add(Mul(a, x), Mul(x, b)), q);
        Assert.True(MaxAbs(residual) <= 1e-10 * Math.Max(1, MaxAbs(x)), $"AX + XB - Q = {MaxAbs(residual):E2}");
    }

    [Fact]
    public void Sylvester_SharedEigenvalue_StillThrows()
    {
        // B = −A: every eigenvalue of A meets one of −B.
        var a = new NdArray(Random(50 * 50, 63), 50, 50);
        var negA = new NdArray(a.ToArray().Select(v => -v).ToArray(), 50, 50);
        Assert.Throws<LinalgSingularMatrixException>(() => Linalg.SolveSylvester(a, negA, new NdArray(Random(2500, 64), 50, 50)));
    }

    [Theory]
    [InlineData(150)] // exact power norms
    [InlineData(230)] // estimated power norms
    public void Expm_SymmetricMatchesEigenDecomposition(int n)
    {
        // exp(S) = V diag(e^λ) Vᵀ — an independent route through the symmetric eigensolver.
        var r = Random(n * n, 70 + n, 0.3);
        var s = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++) s[i * n + j] = r[i * n + j] + r[j * n + i];
        var sa = new NdArray(s, n, n);
        var expected = Linalg.FunmSymmetric(sa, Math.Exp);
        var actual = Linalg.Expm(sa);
        Close(expected, actual, 1e-11, "expm vs eigh");
    }

    [Fact]
    public void Expm_TimesExpmOfNegativeIsIdentity()
    {
        const int n = 210;
        var a = new NdArray(Random(n * n, 80, 0.15), n, n);
        var neg = new NdArray(a.ToArray().Select(v => -v).ToArray(), n, n);
        Identity(Mul(Linalg.Expm(a), Linalg.Expm(neg)), 1e-11, "e^A e^-A");
    }

    [Theory]
    [InlineData(120)]
    [InlineData(260)]
    public void Schur_LargeMatrix_FactorisationAndEigenvaluesAgree(int n)
    {
        var a = new NdArray(Random(n * n, 90 + n), n, n);
        var s = Linalg.Schur(a);
        Orthogonal(s.Z, 1e-12 * n, "Z");
        Assert.True(MaxAbs(Sub(Mul(Mul(s.Z, s.T), s.Z.T), a)) <= 1e-12 * n, "Z T Zᵀ − A");
        var t = s.T.ToArray();
        for (var i = 2; i < n; i++)
            for (var j = 0; j < i - 1; j++) Assert.Equal(0.0, t[i * n + j]);

        // Eigvals takes the eigenvalues-only path (active block only, no Z); same spectrum.
        var w = Linalg.Eigvals(a).OrderBy(z => z.Real).ThenBy(z => z.Imaginary).ToArray();
        var e = Linalg.Eig(a);
        var we = e.Values.OrderBy(z => z.Real).ThenBy(z => z.Imaginary).ToArray();
        for (var i = 0; i < n; i++) Assert.True((w[i] - we[i]).Magnitude <= 1e-9, $"λ[{i}] {w[i]} vs {we[i]}");

        // A V = V Λ for every eigenpair.
        var v = e.Vectors;
        var ad = a.ToArray();
        for (var k = 0; k < n; k += 7)
            for (var i = 0; i < n; i++)
            {
                Complex acc = 0;
                for (var j = 0; j < n; j++) acc += ad[i * n + j] * v[j, k];
                Assert.True((acc - v[i, k] * e.Values[k]).Magnitude <= 1e-10, $"(AV − VΛ)[{i},{k}]");
            }
    }

    [Fact]
    public void NativeLapack_WhenPresent_AgreesWithManagedKernels()
    {
        Assert.SkipUnless(LinalgNativeLapack.IsAvailable, "No LAPACK with LAPACKE_dgees/dgeev/dgehrd found (set GRAVICODE_BLAS to test this).");
        const int n = 90;
        var a = Random(n * n, 99);

        var tn = (double[])a.Clone();
        var zn = new double[n * n];
        var wrN = new double[n];
        var wiN = new double[n];
        Assert.Equal(0, LinalgNativeLapack.Gees(tn, n, zn, wrN, wiN));
        var zNd = new NdArray(zn, n, n);
        Assert.True(MaxAbs(Sub(Mul(Mul(zNd, new NdArray(tn, n, n)), zNd.T), new NdArray(a, n, n))) <= 1e-11, "native Z T Zᵀ − A");

        // Managed kernel directly (Linalg.Schur would route native here).
        var tm = (double[])a.Clone();
        var (wrM, wiM) = RealSchurManaged(tm, n);
        var native = wrN.Zip(wiN, (x, y) => new Complex(x, y)).OrderBy(z => z.Real).ThenBy(z => z.Imaginary).ToArray();
        var managed = wrM.Zip(wiM, (x, y) => new Complex(x, y)).OrderBy(z => z.Real).ThenBy(z => z.Imaginary).ToArray();
        for (var i = 0; i < n; i++) Assert.True((native[i] - managed[i]).Magnitude <= 1e-10, $"λ[{i}]");

        var h = (double[])a.Clone();
        var q = new double[n * n];
        Assert.Equal(0, LinalgNativeLapack.Hessenberg(h, n, q));
        var qNd = new NdArray(q, n, n);
        Assert.True(MaxAbs(Sub(Mul(Mul(qNd, new NdArray(h, n, n)), qNd.T), new NdArray(a, n, n))) <= 1e-11, "native Q H Qᵀ − A");
    }

    private static (double[] Re, double[] Im) RealSchurManaged(double[] t, int n)
    {
        var enabled = Gravicode.Science.GraviNum.Compute.NativeLapack.Enabled;
        // Only this kernel call is affected; ShouldUse is read once at entry.
        Gravicode.Science.GraviNum.Compute.NativeLapack.Enabled = false;
        try { return LinalgSchurKernel.RealSchur(t, new double[n * n], n); }
        finally { Gravicode.Science.GraviNum.Compute.NativeLapack.Enabled = enabled; }
    }
}
