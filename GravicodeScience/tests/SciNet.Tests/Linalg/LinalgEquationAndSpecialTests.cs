using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.LinalgAssert;

namespace Gravicode.Science.Tests.SciNet;

public class LinalgEquationTests
{
    private static readonly NdArray Q = M(3, 3, 2, 0.5, 0, 0.5, 1, 0.2, 0, 0.2, 3);

    [Fact]
    public void Sylvester_MatchesScipyAndResidual()
    {
        var b = M(3, 3, 1, 2, 0, 0, 3, 1, 1, 0, 2);
        var q = new NdArray(Enumerable.Range(0, 12).Select(i => (double)i).ToArray(), 4, 3);
        var x = Linalg.SolveSylvester(A4, b, q);
        Close(q, Add(Mul(A4, x), Mul(x, b)), 1e-12, "AX + XB = Q");
        // scipy 1.17.1 solve_sylvester
        Close([-0.5012543267783099, -0.02750364732564046, -0.14907202146951332, 0.8514174142285893, 0.42470065628041165, 0.957637059564033, 1.1398389217231524, 0.744054312414693, 1.2564402139781383, 1.0727798540316946, 0.7667937283274097, 1.1935898937642142],
            x.ToArray(), 1e-12, "scipy");
        Assert.Throws<LinalgSingularMatrixException>(() => Linalg.SolveSylvester(M(1, 1, 1), M(1, 1, -1), M(1, 1, 1)));
    }

    [Fact]
    public void Lyapunov_ContinuousAndDiscrete_MatchScipy()
    {
        var a = M(3, 3, -3, 1, 0.5, 0.2, -2, 1, 0, -1, -4);
        var x = Linalg.SolveContinuousLyapunov(a, Q);
        Close(Q, Add(Mul(a, x), Mul(x, a.T)), 1e-12, "AX + XAᵀ = Q");
        Close([-0.39442388094559916, -0.1798714407797778, -0.006800404114036833, -0.1798714407797778, -0.29082958383685426, -0.04568487951775265, -0.006800404114036778, -0.04568487951775265, -0.3635787801205619],
            x.ToArray(), 1e-12, "scipy 1.17.1");

        var ad = M(3, 3, 0.5, 0.2, 0, -0.1, 0.3, 0.4, 0.2, 0, -0.6);
        var xd = Linalg.SolveDiscreteLyapunov(ad, Q);
        Assert.True(MaxAbs(Add(Sub(Mul(Mul(ad, xd), ad.T), xd), Q)) < 1e-12, "AXAᵀ − X + Q = 0");
        Close([2.8971944798167173, 0.5249008883530294, 0.3112405231606839, 0.5249008883530294, 1.6978920547983125, -0.7824766382758015, 0.3112405231606839, -0.7824766382758015, 4.751859458803287],
            xd.ToArray(), 1e-12, "scipy 1.17.1");
        // λ = -1 gives λλ̄ = 1: the Stein equation is singular and must be refused, not solved.
        Assert.Throws<LinalgSingularMatrixException>(() => Linalg.SolveDiscreteLyapunov(M(2, 2, -1, 0.3, 0, 0.5), M(2, 2, 1, 0, 0, 1)));
    }

    private static NdArray CareResidual(NdArray a, NdArray b, NdArray q, NdArray r, NdArray x)
        => Add(Sub(Add(Mul(a.T, x), Mul(x, a)), Mul(Mul(Mul(x, b), Linalg.Inv(r)), Mul(b.T, x))), q);

    [Fact]
    public void ContinuousAre_ResidualStabilityAndScipy()
    {
        var a = M(3, 3, 0, 1, 0, 0, 0, 1, -1, 2, 1.5);
        var b = M(3, 1, 0, 0, 1);
        var r = M(1, 1, 0.5);
        var x = Linalg.SolveContinuousAre(a, b, Q, r);
        Assert.True(MaxAbs(CareResidual(a, b, Q, r, x)) < 1e-11, "CARE residual");
        Close([6.6839155195950255, 6.148162230697642, 0.6180339887498976, 6.148162230697642, 11.034677689531222, 3.7655310937860857, 0.6180339887498976, 3.7655310937860857, 3.1641315402823618],
            x.ToArray(), 1e-11, "scipy 1.17.1");
        // Closed loop A − B R⁻¹ Bᵀ X is Hurwitz.
        var closed = Sub(a, Mul(Mul(b, Linalg.Inv(r)), Mul(b.T, x)));
        Assert.All(Linalg.Eigvals(closed), l => Assert.True(l.Real < 0, $"unstable {l}"));

        var e = M(3, 3, 2, 0, 0.5, 0, 1, 0, 0.1, 0, 1);
        var s = M(3, 1, 0.1, 0, 0.2);
        Close([2.8500045505526916, 3.153610233015561, 0.12923698991174562, 3.153610233015561, 8.142543685731342, 3.1572228056034346, 0.12923698991174562, 3.1572228056034346, 3.034210240387325],
            Linalg.SolveContinuousAre(a, b, Q, r, e, s).ToArray(), 1e-11, "with e and s, scipy 1.17.1");
    }

    [Fact]
    public void DiscreteAre_ResidualStabilityAndScipy()
    {
        var a = M(3, 3, 1.1, 0.5, 0, 0, 0.9, 0.3, 0.2, 0, 1.2);
        var b = M(3, 2, 0, 1, 1, 0, 0.5, 0.5);
        var r = M(2, 2, 1, 0.2, 0.2, 2);
        var x = Linalg.SolveDiscreteAre(a, b, Q, r);
        var btx = Mul(b.T, x);
        var k = Linalg.Solve(Add(r, Mul(btx, b)), Mul(btx, a));
        var res = Add(Sub(Sub(Mul(Mul(a.T, x), a), x), Mul(Mul(a.T, Mul(x, b)), k)), Q);
        Assert.True(MaxAbs(res) < 1e-11, "DARE residual");
        Close([3.894799091716634, 2.8114156036436126, -2.2007898393659096, 2.8114156036436126, 6.659755399216865, -6.959094573647807, -2.2007898393659096, -6.959094573647807, 15.229484881696187],
            x.ToArray(), 1e-11, "scipy 1.17.1");
        Assert.All(Linalg.Eigvals(Sub(a, Mul(b, k))), l => Assert.True(l.Magnitude < 1, $"unstable {l}"));

        var e = M(3, 3, 2, 0, 0.5, 0, 1, 0, 0.1, 0, 1);
        var s = M(3, 2, 0.1, 0, 0, 0.1, 0.05, 0);
        Close([0.7011809063425423, 0.5601092843555378, -0.9039447087847163, 0.5601092843555378, 3.2577601021160447, -3.1485875907603114, -0.9039447087847163, -3.1485875907603114, 10.732752383251945],
            Linalg.SolveDiscreteAre(a, b, Q, r, e, s).ToArray(), 1e-11, "with e and s, scipy 1.17.1");
    }
}

public class LinalgSpecialMatrixTests
{
    [Fact]
    public void SpecialMatrices_MatchScipy()
    {
        // All expected arrays from scipy 1.17.1.
        Close([-2, -3, 1, 0, 0, 1, 0, 0, 0, 0, 0, -4, 0, -5, 1, 0, 1, 0, 0, 0, 0, 0, 0, -6, 0],
            Linalg.FiedlerCompanion([1, 2, 3, 4, 5, 6]).ToArray(), 0, "fiedler_companion");
        Close([0.7071067811865475, -0.7071067811865475, 0, 0, 0.4082482904638631, 0.4082482904638631, -0.8164965809277261, 0, 0.2886751345948129, 0.2886751345948129, 0.2886751345948129, -0.8660254037844387],
            Linalg.Helmert(4).ToArray(), 1e-15, "helmert");
        Orthogonal(Linalg.Helmert(5, full: true), 1e-15, "helmert full");
        Close([16, -120, 240, -140, -120, 1200, -2700, 1680, 240, -2700, 6480, -4200, -140, 1680, -4200, 2800],
            Linalg.InvHilbert(4).ToArray(), 0, "invhilbert");
        Identity(Mul(Linalg.Hilbert(6), Linalg.InvHilbert(6)), 1e-6, "H H⁻¹");
        Close([4, -6, 4, -1, -6, 14, -11, 3, 4, -11, 10, -3, -1, 3, -3, 1], Linalg.Invpascal(4).ToArray(), 0, "invpascal");
        Close([1, 0, 0, 0, -1, 1, 0, 0, 1, -2, 1, 0, -1, 3, -3, 1], Linalg.Invpascal(4, "lower").ToArray(), 0, "invpascal lower");
        Identity(Mul(Linalg.Pascal(5, "upper"), Linalg.Invpascal(5, "upper")), 0, "pascal upper");
        Identity(Mul(Linalg.Pascal(6), Linalg.Invpascal(6)), 0, "pascal symmetric");
        Close([2, 1, 0, 0, 0, 3, 2, 1, 0, 0, 4, 3, 2, 1, 0, 0, 4, 3, 2, 1, 0, 0, 4, 3, 2],
            Linalg.ConvolutionMatrix([1, 2, 3, 4], 5, "same").ToArray(), 0, "convolution same");
        Close([3, 2, 1, 0, 0, 0, 3, 2, 1, 0, 0, 0, 3, 2, 1], Linalg.ConvolutionMatrix([1, 2, 3], 5, "valid").ToArray(), 0, "convolution valid");
        Close([1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6], Linalg.Hankel([1, 2, 3], [3, 4, 5, 6]).ToArray(), 0, "hankel");

        var full = Linalg.ConvolutionMatrix([1, 2, 3], 4);
        Close([1, 4, 10, 16, 17, 12], Mul(full, LinalgAssert.V(1, 2, 3, 4)).ToArray(), 0, "full convolution");

        var comp = Linalg.Companion([1, -6, 11, -6]); // roots 1, 2, 3
        Close([1, 2, 3], Linalg.Eigvals(comp).Select(z => z.Real).OrderBy(v => v).ToArray(), 1e-12, "companion roots");
        Close([1, 2, 3], Linalg.Eigvals(Linalg.FiedlerCompanion([1, -6, 11, -6])).Select(z => z.Real).OrderBy(v => v).ToArray(), 1e-12, "fiedler roots");

        var h = Linalg.Hadamard(8);
        var eightI = new double[64];
        for (var i = 0; i < 8; i++) eightI[i * 9] = 8;
        Close(eightI, Mul(h, h.T).ToArray(), 0, "H Hᵀ = 8 I");
        Assert.Throws<ArgumentException>(() => Linalg.Hadamard(6));

        var dft = Linalg.Dft(5, "sqrtn");
        var prod = CMul(dft.ConjugateTranspose().ToArray(), dft.ToArray(), 5);
        for (var i = 0; i < 5; i++) for (var j = 0; j < 5; j++)
            Assert.True(Complex.Abs(prod[i * 5 + j] - (i == j ? 1 : 0)) < 1e-14, "unitary DFT");
        var x = new Complex[] { 1, 2, 3, 4, 5 };
        var spectrum = Gravicode.Science.GraviNum.Signal.Fft.Forward(x);
        var dft1 = Linalg.Dft(5).ToArray();
        for (var i = 0; i < 5; i++)
        {
            var s = Complex.Zero;
            for (var j = 0; j < 5; j++) s += dft1[i * 5 + j] * x[j];
            Assert.True(Complex.Abs(s - spectrum[i]) < 1e-12, "DFT matrix = FFT");
        }

        Close([0, 1, 3, 1, 0, 2, 3, 2, 0], Linalg.Fiedler([1, 2, 4]).ToArray(), 0, "fiedler");
        Close([1, 2, 3, 0.5, 0, 0, 0, 0.25, 0], Linalg.Leslie([1, 2, 3], [0.5, 0.25]).ToArray(), 0, "leslie");
        var bd = Linalg.BlockDiag(M(1, 2, 1, 2), M(2, 1, 3, 4));
        Close([1, 2, 0, 0, 0, 3, 0, 0, 4], bd.ToArray(), 0, "block_diag");
        Close([1, 1, 1, 1, 2, 4, 1, 3, 9], Linalg.Vandermonde([1, 2, 3], increasing: true).ToArray(), 0, "vander");
        Close([4, 2, 1, 1, 4, 2, 2, 1, 4], Linalg.Circulant([4, 1, 2]).ToArray(), 0, "circulant");
        Close([1, 2, 3, 0, 5, 6, 0, 0, 9],
            Linalg.Triu(M(3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9)).ToArray(), 0, "triu");
        Close([1, 0, 0, 4, 5, 0, 7, 8, 9], Linalg.Tril(M(3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9)).ToArray(), 0, "tril");
        Close([0, 2, 3, 0, 0, 6, 0, 0, 0], Linalg.Triu(M(3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9), 1).ToArray(), 0, "triu k=1");
    }

    [Fact]
    public void Norms_Bandwidth_Symmetry_MatchScipy()
    {
        // scipy 1.17.1 norm(A4, ord)
        Close(17.0, Linalg.Norm(A4, 1), 1e-15);
        Close(10.0, Linalg.Norm(A4, -1), 1e-15);
        Close(10.830459127509027, Linalg.Norm(A4, 2), 1e-13);
        Close(2.509646317210538, Linalg.Norm(A4, -2), 1e-13);
        Close(16.0, Linalg.Norm(A4, double.PositiveInfinity), 1e-15);
        Close(10.0, Linalg.Norm(A4, double.NegativeInfinity), 1e-15);
        Close(15.874507866387544, Linalg.Norm(A4, "fro"), 1e-14);
        Close(15.874507866387544, Linalg.Norm(A4), 1e-14);
        Close(29.15488282298732, Linalg.Norm(A4, "nuc"), 1e-13);
        Close(22.392304845413257, Linalg.Norm(V(3, -4, 0, 1), 0.5), 1e-14);
        Close(3.0, Linalg.Norm(V(3, -4, 0, 1), 0), 0);
        Close(5.0, Linalg.Norm(V(3, -4)), 1e-15);
        Close(1e300 * Math.Sqrt(2), Linalg.Norm(V(1e300, 1e300)), 1e-15, "no overflow");
        Assert.Equal((2, 1), Linalg.Bandwidth(M(3, 3, 1, 2, 0, 0, 1, 0, 5, 0, 1)));
        Assert.True(Linalg.IsSymmetric(M(2, 2, 1, 2, 2, 1)));
        Assert.False(Linalg.IsSymmetric(M(2, 2, 1, 2, 2 + 1e-12, 1)));
        Assert.True(Linalg.IsHermitian(M(2, 2, 1, 2, 2 + 1e-12, 1), atol: 1e-10));
    }
}
