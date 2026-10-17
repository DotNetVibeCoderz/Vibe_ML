using System.Numerics;

namespace Gravicode.Science.SciNet;

/// <summary>
/// The triangular Sylvester equation <c>R Y + Y S = F</c> (R, S complex upper triangular), the
/// middle stage of Bartels–Stewart.
/// </summary>
/// <remarks>
/// <para>
/// Recursive blocking after Jonsson &amp; Kågström (2002, "RECSY"): split the larger of R and S
/// in half, solve the half that does not depend on the other, subtract its contribution from the
/// other half's right-hand side with one matrix product, recurse. Blocks of 32×32 or smaller are
/// solved element by element. The column-by-column substitution this replaced did all
/// O(mn(m+n)) of its work in scalar <see cref="Complex"/> arithmetic and was, at n = 400, about as
/// slow as both Schur decompositions together.
/// </para>
/// <para>
/// Everything is kept split into real and imaginary arrays so each complex product is four real
/// products through <see cref="LinalgGemm"/>. Uniqueness fails when <c>R[i,i] + S[j,j]</c>
/// vanishes; that is tested relative to <c>max(max|R[i,i]|, |S[j,j]|)</c>, as the previous
/// column solve did, and throws.
/// </para>
/// </remarks>
internal static unsafe class LinalgSylvesterKernel
{
    private const int Leaf = 32;

    public static Complex[] SolveTriangular(Complex[] r, int m, Complex[] s, int n, Complex[] f, string caller)
    {
        var rr = new double[m * m]; var ri = new double[m * m];
        var sr = new double[n * n]; var si = new double[n * n];
        var yr = new double[m * n]; var yi = new double[m * n];
        for (var i = 0; i < rr.Length; i++) { rr[i] = r[i].Real; ri[i] = r[i].Imaginary; }
        for (var i = 0; i < sr.Length; i++) { sr[i] = s[i].Real; si[i] = s[i].Imaginary; }
        for (var i = 0; i < yr.Length; i++) { yr[i] = f[i].Real; yi[i] = f[i].Imaginary; }
        var maxR = 0.0;
        for (var i = 0; i < m; i++) maxR = Math.Max(maxR, Complex.Abs(r[i * m + i]));

        fixed (double* prr = rr, pri = ri, psr = sr, psi = si, pyr = yr, pyi = yi)
        {
            var ctx = new Context(prr, pri, m, psr, psi, n, pyr, pyi, maxR, caller);
            Solve(ref ctx, 0, m, 0, n);
        }

        var y = new Complex[m * n];
        for (var i = 0; i < y.Length; i++) y[i] = new Complex(yr[i], yi[i]);
        return y;
    }

    private readonly struct Context(double* rr, double* ri, int m, double* sr, double* si, int n,
        double* yr, double* yi, double maxR, string caller)
    {
        public readonly double* Rr = rr, Ri = ri, Sr = sr, Si = si, Yr = yr, Yi = yi;
        public readonly int M = m, N = n;
        public readonly double MaxR = maxR;
        public readonly string Caller = caller;
    }

    /// <summary>Solves for the block of Y with rows [r0, r0+mb) and columns [c0, c0+nb).</summary>
    private static void Solve(ref Context c, int r0, int mb, int c0, int nb)
    {
        if (mb <= 0 || nb <= 0) return;
        if (mb <= Leaf && nb <= Leaf)
        {
            Base(ref c, r0, mb, c0, nb);
            return;
        }
        if (mb >= nb)
        {
            // Bottom rows first (R is upper triangular), then the top rows' right-hand side
            // loses R[top, bottom] Y[bottom].
            var h = mb / 2;
            Solve(ref c, r0 + h, mb - h, c0, nb);
            SubtractProduct(c.Rr + (long)r0 * c.M + r0 + h, c.Ri + (long)r0 * c.M + r0 + h, c.M,
                c.Yr + (long)(r0 + h) * c.N + c0, c.Yi + (long)(r0 + h) * c.N + c0, c.N,
                c.Yr + (long)r0 * c.N + c0, c.Yi + (long)r0 * c.N + c0, c.N, h, nb, mb - h);
            Solve(ref c, r0, h, c0, nb);
        }
        else
        {
            // Left columns first (S is upper triangular), then the right columns lose Y[left] S[left, right].
            var h = nb / 2;
            Solve(ref c, r0, mb, c0, h);
            SubtractProduct(c.Yr + (long)r0 * c.N + c0, c.Yi + (long)r0 * c.N + c0, c.N,
                c.Sr + (long)c0 * c.N + c0 + h, c.Si + (long)c0 * c.N + c0 + h, c.N,
                c.Yr + (long)r0 * c.N + c0 + h, c.Yi + (long)r0 * c.N + c0 + h, c.N, mb, nb - h, h);
            Solve(ref c, r0, mb, c0 + h, nb - h);
        }
    }

    /// <summary>C ← C − A B for complex split operands (A rows×k, B k×cols).</summary>
    private static void SubtractProduct(double* ar, double* ai, int lda, double* br, double* bi, int ldb,
        double* cr, double* ci, int ldc, int rows, int cols, int k)
    {
        LinalgGemm.Gemm(rows, cols, k, -1.0, ar, lda, br, ldb, 1.0, cr, ldc);
        LinalgGemm.Gemm(rows, cols, k, 1.0, ai, lda, bi, ldb, 1.0, cr, ldc);
        LinalgGemm.Gemm(rows, cols, k, -1.0, ar, lda, bi, ldb, 1.0, ci, ldc);
        LinalgGemm.Gemm(rows, cols, k, -1.0, ai, lda, br, ldb, 1.0, ci, ldc);
    }

    /// <summary>Element-wise substitution: rows bottom to top, columns left to right.</summary>
    private static void Base(ref Context c, int r0, int mb, int c0, int nb)
    {
        int m = c.M, n = c.N;
        var r1 = r0 + mb;
        var c1 = c0 + nb;
        for (var i = r1 - 1; i >= r0; i--)
        {
            var rowR = (long)i * m;
            var rowY = (long)i * n;
            for (var j = c0; j < c1; j++)
            {
                var accR = c.Yr[rowY + j];
                var accI = c.Yi[rowY + j];
                for (var k = i + 1; k < r1; k++)
                {
                    double ar = c.Rr[rowR + k], ai = c.Ri[rowR + k];
                    double yr = c.Yr[(long)k * n + j], yi = c.Yi[(long)k * n + j];
                    accR -= ar * yr - ai * yi;
                    accI -= ar * yi + ai * yr;
                }
                for (var l = c0; l < j; l++)
                {
                    double yr = c.Yr[rowY + l], yi = c.Yi[rowY + l];
                    double sr = c.Sr[(long)l * n + j], si = c.Si[(long)l * n + j];
                    accR -= yr * sr - yi * si;
                    accI -= yr * si + yi * sr;
                }
                var d = new Complex(c.Rr[rowR + i] + c.Sr[(long)j * n + j], c.Ri[rowR + i] + c.Si[(long)j * n + j]);
                var scale = Math.Max(c.MaxR, Complex.Abs(new Complex(c.Sr[(long)j * n + j], c.Si[(long)j * n + j])));
                if (Complex.Abs(d) <= 1e-14 * Math.Max(scale, 1e-300))
                    throw new LinalgSingularMatrixException($"{c.Caller}: the equation is singular (eigenvalues sum to zero); no unique solution.");
                var q = new Complex(accR, accI) / d;
                c.Yr[rowY + j] = q.Real;
                c.Yi[rowY + j] = q.Imaginary;
            }
        }
    }
}
