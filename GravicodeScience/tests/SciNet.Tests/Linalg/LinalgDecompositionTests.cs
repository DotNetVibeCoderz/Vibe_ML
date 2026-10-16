using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.LinalgAssert;

namespace Gravicode.Science.Tests.SciNet;

public class LinalgDecompositionTests
{
    [Fact]
    public void QrPivoting_ReconstructsAndRevealsRank()
    {
        // Rank 2 matrix: third column = first + second.
        var a = M(4, 3, 1, 2, 3, 4, 5, 9, 7, 8, 15, 1, 0, 1);
        var r = Linalg.QrPivoting(a);
        Orthogonal(r.Q, 1e-13, "Q");
        Close(a.T.Copy().Take(r.P).T.Copy(), Mul(r.Q, r.R), 1e-12, "A[:,P] = QR");
        for (var k = 0; k + 1 < 3; k++) Assert.True(Math.Abs(r.R[k, k]) >= Math.Abs(r.R[k + 1, k + 1]) - 1e-12);
        Assert.True(Math.Abs(r.R[2, 2]) < 1e-12, "rank deficiency shows on the diagonal");
        // scipy 1.17.1: abs(diag(R)) of qr(a, pivoting=True)
        Close([17.776388834631177, 0.6820482925847114], [Math.Abs(r.R[0, 0]), Math.Abs(r.R[1, 1])], 1e-12, "|diag R|");

        var e = Linalg.QrPivoting(a, economic: true);
        Assert.Equal([4, 3], e.Q.Shape.ToArray());
        Assert.Equal([3, 3], e.R.Shape.ToArray());
    }

    [Theory]
    [InlineData(4, 4)]
    [InlineData(3, 5)]
    [InlineData(5, 3)]
    public void Rq_ReconstructsWithUpperTrapezoidalR(int m, int n)
    {
        var a = Random(m, n, 100 + m * 10 + n);
        var r = Linalg.Rq(a);
        Orthogonal(r.Q, 1e-13, "Q");
        Close(a, Mul(r.R, r.Q), 1e-13, "A = RQ");
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                if (j < i + n - m) Assert.True(Math.Abs(r.R[i, j]) < 1e-14, $"R[{i},{j}] = {r.R[i, j]}");
        var e = Linalg.Rq(a, economic: true);
        Close(a, Mul(e.R, e.Q), 1e-13, "economic");
    }

    [Fact]
    public void Hessenberg_ReconstructsAndIsHessenberg()
    {
        var a = Random(7, 7, 5);
        var h = Linalg.Hessenberg(a, calcQ: true);
        Orthogonal(h.Q!, 1e-13, "Q");
        Close(a, Mul(Mul(h.Q!, h.H), h.Q!.T), 1e-13, "Q H Qᵀ");
        for (var i = 2; i < 7; i++) for (var j = 0; j < i - 1; j++) Assert.Equal(0.0, h.H[i, j]);
        Assert.Null(Linalg.Hessenberg(a).Q);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void Schur_ReconstructsWithStandardisedBlocks(int seed)
    {
        var n = 8;
        var a = Random(n, n, seed);
        var s = Linalg.Schur(a);
        Orthogonal(s.Z, 1e-12, "Z");
        Close(a, Mul(Mul(s.Z, s.T), s.Z.T), 1e-12, "Z T Zᵀ");
        var blocks = 0;
        for (var i = 1; i < n; i++)
        {
            for (var j = 0; j < i - 1; j++) Assert.Equal(0.0, s.T[i, j]);
            if (s.T[i, i - 1] != 0)
            {
                blocks++;
                Assert.Equal(s.T[i - 1, i - 1], s.T[i, i]);              // equal diagonal
                Assert.True(s.T[i - 1, i] * s.T[i, i - 1] < 0);        // b c < 0
                if (i + 1 < n) Assert.Equal(0.0, s.T[i + 1, i]);       // blocks do not overlap
            }
        }
        Assert.True(blocks > 0, "random 8x8 should have a complex pair");

        var c = Linalg.SchurComplex(a);
        var t = c.T.ToArray();
        var z = c.Z.ToArray();
        for (var i = 1; i < n; i++) for (var j = 0; j < i; j++) Assert.Equal(Complex.Zero, t[i * n + j]);
        var zh = c.Z.ConjugateTranspose().ToArray();
        var rec = CMul(CMul(z, t, n), zh, n);
        var ad = a.ToArray();
        for (var i = 0; i < n * n; i++) Assert.True(Complex.Abs(rec[i] - ad[i]) < 1e-12, $"complex Schur [{i}]");
    }

    [Fact]
    public void Schur_And_Eig_OnA150Matrix()
    {
        const int n = 150;
        var a = Random(n, n, 77);
        var s = Linalg.Schur(a);
        Assert.True(MaxAbs(Sub(a, Mul(Mul(s.Z, s.T), s.Z.T))) < 1e-11, "Z T Zᵀ");
        Assert.True(MaxAbs(Sub(Mul(s.Z.T, s.Z), NdArray.Eye(n))) < 1e-12, "ZᵀZ");
        // Trace and the sum of eigenvalues agree.
        var w = Linalg.Eigvals(a);
        var trace = 0.0;
        for (var i = 0; i < n; i++) trace += a[i, i];
        Assert.True(Math.Abs(w.Aggregate(Complex.Zero, (x, y) => x + y).Real - trace) < 1e-10);
    }

    [Fact]
    public void Schur_OfRotationAndJordanBlock()
    {
        // Pure rotation: eigenvalues ±i, a single standardised block.
        var s = Linalg.Schur(M(2, 2, 0, -1, 1, 0));
        Assert.Equal(0.0, s.T[0, 0]);
        Assert.True(Math.Abs(Math.Abs(s.T[0, 1] * s.T[1, 0]) - 1) < 1e-14);
        // A Jordan block is already triangular.
        var j = M(3, 3, 2, 1, 0, 0, 2, 1, 0, 0, 2);
        var sj = Linalg.Schur(j);
        Close(j, Mul(Mul(sj.Z, sj.T), sj.Z.T), 1e-14, "jordan");
    }

    [Fact]
    public void Eig_MatchesScipyAndSatisfiesAvEqualsLambdaV()
    {
        var r = Linalg.Eig(A4);
        // scipy 1.17.1 eigvals(A4), sorted by (real, imag)
        Complex[] expected =
        [
            3.1960692553304044, 5.948801222386055,
            new(7.927564761141769, -4.914044197630075), new(7.927564761141769, 4.914044197630075),
        ];
        var got = r.Values.OrderBy(v => v.Real).ThenBy(v => v.Imaginary).ToArray();
        for (var i = 0; i < 4; i++) Assert.True(Complex.Abs(expected[i] - got[i]) < 1e-12, $"λ{i}: {got[i]}");

        var n = 4;
        var v = r.Vectors.ToArray();
        var ad = A4.ToArray();
        for (var k = 0; k < n; k++)
        {
            var norm = 0.0;
            for (var i = 0; i < n; i++)
            {
                var av = Complex.Zero;
                for (var j = 0; j < n; j++) av += ad[i * n + j] * v[j * n + k];
                Assert.True(Complex.Abs(av - r.Values[k] * v[i * n + k]) < 1e-12, $"residual col {k}");
                norm += v[i * n + k].Magnitude * v[i * n + k].Magnitude;
            }
            Assert.True(Math.Abs(norm - 1) < 1e-13, "unit norm");
        }
        // Conjugate pairs are adjacent, positive imaginary part first, vectors conjugate.
        var p = Array.FindIndex(r.Values, z => z.Imaginary > 0);
        Assert.Equal(Complex.Conjugate(r.Values[p]), r.Values[p + 1]);
        for (var i = 0; i < n; i++) Assert.Equal(Complex.Conjugate(v[i * n + p]), v[i * n + p + 1]);
        Assert.Equal(r.Values, Linalg.Eigvals(A4));
    }

    [Fact]
    public void Eig_BadlyScaledMatrixBenefitsFromBalancing()
    {
        // Similar to diag(1,2,3) by a diagonal scaling with entries spanning 1e-8..1e8.
        double[] d = [1e-8, 1, 1e8];
        var b = M(3, 3, 1, 2, 3, 0, 2, 4, 0, 0, 3);
        var a = NdArray.Zeros(3, 3);
        for (var i = 0; i < 3; i++) for (var j = 0; j < 3; j++) a[i, j] = b[i, j] * d[i] / d[j];
        var w = Linalg.Eigvals(a).Select(z => z.Real).OrderBy(x => x).ToArray();
        Close([1, 2, 3], w, 1e-12, "eigenvalues");
    }

    [Fact]
    public void Eigh_StandardGeneralisedAndSubsets()
    {
        var s = M(3, 3, 4, 1, 2, 1, 5, 3, 2, 3, 6);
        var g = M(3, 3, 4, 1, 0.5, 1, 3, 0.2, 0.5, 0.2, 2);
        var e = Linalg.Eigh(s, g);
        // scipy 1.17.1 eigh(S, G) eigenvalues
        Close([0.7198807599863909, 1.3003304643769427, 3.512433209643242], e.Values, 1e-12, "generalised λ");
        Identity(Mul(Mul(e.Vectors.T, g), e.Vectors), 1e-12, "VᵀBV = I");
        Close(Mul(Mul(s, e.Vectors), NdArray.Eye(3)), Mul(Mul(g, e.Vectors), NdArray.Diag(e.Values)), 1e-12, "A V = B V Λ");

        var std = Linalg.Eigh(s);
        Assert.True(std.Values[0] < std.Values[1] && std.Values[1] < std.Values[2], "ascending");
        Close(Mul(s, std.Vectors), Mul(std.Vectors, NdArray.Diag(std.Values)), 1e-12, "A V = V Λ");
        var sub = Linalg.Eigh(s, subsetByIndex: (1, 2));
        Close(std.Values[1..], sub.Values, 1e-14, "subset by index");
        Assert.Equal([3, 2], sub.Vectors.Shape.ToArray());
        var byValue = Linalg.Eigh(s, subsetByValue: (std.Values[0], 100));
        Assert.Equal(2, byValue.Values.Length);
        Close(std.Values, Linalg.Eigvalsh(s), 1e-14, "eigvalsh");
    }

    [Fact]
    public void EighTridiagonal_MatchesScipyAndDense()
    {
        double[] d = [2, 3, 1, 4];
        double[] e = [1, -0.5, 2];
        var r = Linalg.EighTridiagonal(d, e);
        // scipy 1.17.1 eigh_tridiagonal
        Close([-0.07733643776477617, 1.4157430318632365, 3.631478971677628, 5.03011443422391], r.Values, 1e-13, "λ");
        var t = NdArray.Zeros(4, 4);
        for (var i = 0; i < 4; i++) t[i, i] = d[i];
        for (var i = 0; i < 3; i++) { t[i, i + 1] = e[i]; t[i + 1, i] = e[i]; }
        Close(Mul(t, r.Vectors), Mul(r.Vectors, NdArray.Diag(r.Values)), 1e-13, "T V = V Λ");
        Orthogonal(r.Vectors, 1e-13, "V");
        Close(r.Values, Linalg.EigvalsBanded(M(2, 4, 0, 1, -0.5, 2, 2, 3, 1, 4)), 1e-13, "banded upper");
    }

    [Theory]
    [InlineData(5, 3)]
    [InlineData(3, 5)]
    public void Svd_FullAndThin_Reconstruct(int m, int n)
    {
        var a = Random(m, n, 40 + m);
        var full = Linalg.Svd(a);
        Assert.Equal([m, m], full.U.Shape.ToArray());
        Assert.Equal([n, n], full.Vh.Shape.ToArray());
        Orthogonal(full.U, 1e-13, "U");
        Orthogonal(full.Vh.T.Copy(), 1e-13, "V");
        Close(a, Mul(Mul(full.U, Linalg.Diagsvd(full.S, m, n)), full.Vh), 1e-13, "U Σ Vh");
        var thin = Linalg.Svd(a, fullMatrices: false);
        var k = Math.Min(m, n);
        Close(a, Mul(Mul(thin.U, Linalg.Diagsvd(thin.S, k, k)), thin.Vh), 1e-13, "thin");
        Close(full.S, Linalg.Svdvals(a), 1e-14, "svdvals");
    }

    [Fact]
    public void Polar_NullSpace_Orth_SubspaceAngles()
    {
        var a = M(3, 3, 1, -1, 0, 2, 3, 1, 0, 1, 4);
        var p = Linalg.Polar(a);
        Orthogonal(p.U, 1e-13, "U");
        Close(a, Mul(p.U, p.P), 1e-13, "A = U P");
        Close(p.P, p.P.T.Copy(), 1e-13, "P symmetric");
        Assert.All(Linalg.Eigvalsh(p.P), l => Assert.True(l > 0));
        var left = Linalg.Polar(a, "left");
        Close(a, Mul(left.P, left.U), 1e-13, "A = P U");

        var rd = M(3, 4, 1, 2, 3, 4, 2, 4, 6, 8, 1, 0, 1, 0);
        var ns = Linalg.NullSpace(rd);
        Assert.Equal([4, 2], ns.Shape.ToArray());
        Assert.True(MaxAbs(Mul(rd, ns)) < 1e-13, "A N = 0");
        Orthogonal(ns, 1e-13, "N");
        var o = Linalg.Orth(rd);
        Assert.Equal([3, 2], o.Shape.ToArray());

        var angles = Linalg.SubspaceAngles(M(4, 2, 1, 0, 0, 1, 1, 1, 0, 2), M(4, 2, 1, 1, 1, 0, 0, 1, 1, -1));
        // scipy 1.17.1 subspace_angles
        Close([0.7423985899137626, 0.30188792690906435], angles, 1e-12, "angles");
        // Tiny angle: arccos would return 0 or garbage; arcsin keeps it.
        var tiny = Linalg.SubspaceAngles(M(3, 1, 1, 0, 0), M(3, 1, 1, 1e-10, 0));
        Close([1e-10], tiny, 1e-14, "tiny angle");
    }
}
