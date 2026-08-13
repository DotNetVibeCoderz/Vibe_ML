using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.GraviNum;

public class LinAlgTests
{
    private static NdArray NaiveMatMul(NdArray a, NdArray b)
    {
        var result = NdArray.Zeros(a.Shape[0], b.Shape[1]);
        for (var i = 0; i < a.Shape[0]; i++)
            for (var j = 0; j < b.Shape[1]; j++)
            {
                var acc = 0.0;
                for (var k = 0; k < a.Shape[1]; k++) acc += a[i, k] * b[k, j];
                result[i, j] = acc;
            }
        return result;
    }

    [Fact]
    public void Dot_MatchesTheTextbookTripleLoop()
    {
        var rng = new GraviRandom(7);
        var a = rng.StandardNormal(37, 23);
        var b = rng.StandardNormal(23, 19);

        Assert.True(UFunc.AllClose(LinAlg.Dot(a, b), NaiveMatMul(a, b), 1e-9));
    }

    [Fact]
    public void Dot_TakesTheParallelPathForLargeMatrices()
    {
        // Above ParallelRowThreshold the implementation switches strategy; results must not change.
        var rng = new GraviRandom(11);
        var a = rng.StandardNormal(80, 40);
        var b = rng.StandardNormal(40, 30);

        Assert.True(UFunc.AllClose(LinAlg.Dot(a, b), NaiveMatMul(a, b), 1e-9));
    }

    [Fact]
    public void Dot_HandlesMatrixVectorAndVectorVector()
    {
        var a = NdArray.FromArray(new double[,] { { 1, 2 }, { 3, 4 } });
        var v = NdArray.FromValues([1.0, 1.0]);

        var mv = LinAlg.Dot(a, v);
        Assert.Equal(new[] { 3.0, 7.0 }, mv.ToArray());

        var inner = LinAlg.Dot(v, v);
        Assert.Equal(2.0, inner.At(0));
    }

    [Fact]
    public void Dot_RejectsMisalignedShapes()
    {
        var a = NdArray.Ones(2, 3);
        var b = NdArray.Ones(4, 2);
        Assert.Throws<InvalidOperationException>(() => LinAlg.Dot(a, b));
    }

    [Fact]
    public void Determinant_MatchesTheClosedFormForSmallMatrices()
    {
        var a = NdArray.FromArray(new double[,] { { 4, 7 }, { 2, 6 } });
        Assert.Equal(10.0, LinAlg.Determinant(a), 9);

        var b = NdArray.FromArray(new double[,] { { 6, 1, 1 }, { 4, -2, 5 }, { 2, 8, 7 } });
        Assert.Equal(-306.0, LinAlg.Determinant(b), 9);
    }

    [Fact]
    public void Inverse_TimesOriginalIsIdentity()
    {
        var rng = new GraviRandom(3);
        var a = rng.StandardNormal(6, 6) + NdArray.Eye(6) * 6.0;
        var inv = LinAlg.Inverse(a);
        Assert.True(UFunc.AllClose(LinAlg.Dot(a, inv), NdArray.Eye(6), 1e-8));
    }

    [Fact]
    public void Solve_RecoversTheKnownSolution()
    {
        var a = NdArray.FromArray(new double[,] { { 3, 2, -1 }, { 2, -2, 4 }, { -1, 0.5, -1 } });
        var expected = NdArray.FromValues([1.0, -2.0, -2.0]);
        var b = LinAlg.Dot(a, expected);

        Assert.True(UFunc.AllClose(LinAlg.Solve(a, b), expected, 1e-9));
    }

    [Fact]
    public void LuFactorisation_ReconstructsThePermutedMatrix()
    {
        var rng = new GraviRandom(21);
        var a = rng.StandardNormal(7, 7);
        var lu = Decomposition.Lu(a);

        var pa = LinAlg.Dot(lu.PermutationMatrix(), a);
        var lUpper = LinAlg.Dot(lu.Lower, lu.Upper);
        Assert.True(UFunc.AllClose(pa, lUpper, 1e-9));
    }

    [Fact]
    public void QrFactorisation_HasOrthonormalQAndReconstructsA()
    {
        var rng = new GraviRandom(5);
        var a = rng.StandardNormal(8, 4);
        var qr = Decomposition.Qr(a);

        Assert.True(UFunc.AllClose(LinAlg.Dot(qr.Q.T, qr.Q), NdArray.Eye(4), 1e-9));
        Assert.True(UFunc.AllClose(LinAlg.Dot(qr.Q, qr.R), a, 1e-9));
    }

    [Fact]
    public void Cholesky_ReconstructsAPositiveDefiniteMatrix()
    {
        var rng = new GraviRandom(9);
        var m = rng.StandardNormal(5, 5);
        var spd = LinAlg.Dot(m, m.T) + NdArray.Eye(5) * 5.0;

        var l = Decomposition.Cholesky(spd);
        Assert.True(UFunc.AllClose(LinAlg.Dot(l, l.T), spd, 1e-8));
    }

    [Fact]
    public void Cholesky_RejectsNonPositiveDefiniteInput()
    {
        var notSpd = NdArray.FromArray(new double[,] { { 1, 2 }, { 2, 1 } });
        Assert.Throws<InvalidOperationException>(() => Decomposition.Cholesky(notSpd));
    }

    [Fact]
    public void Svd_ReconstructsTheMatrixAndOrdersSingularValues()
    {
        var rng = new GraviRandom(13);
        var a = rng.StandardNormal(9, 5);
        var svd = Decomposition.Svd(a);

        Assert.True(UFunc.AllClose(svd.Reconstruct(), a, 1e-8));

        for (var i = 1; i < svd.SingularValues.Size; i++)
            Assert.True(svd.SingularValues.At(i - 1) >= svd.SingularValues.At(i));

        Assert.True(UFunc.AllClose(LinAlg.Dot(svd.V.T, svd.V), NdArray.Eye(5), 1e-8));
    }

    [Fact]
    public void Svd_WorksWhenThereAreMoreColumnsThanRows()
    {
        var rng = new GraviRandom(17);
        var a = rng.StandardNormal(3, 7);
        var svd = Decomposition.Svd(a);
        Assert.True(UFunc.AllClose(svd.Reconstruct(), a, 1e-8));
    }

    [Fact]
    public void SymmetricEigen_SatisfiesTheEigenEquation()
    {
        var rng = new GraviRandom(23);
        var m = rng.StandardNormal(6, 6);
        var symmetric = (m + m.T) * 0.5;

        var eigen = Decomposition.SymmetricEigen(symmetric);

        // A v = lambda v, column by column.
        for (var k = 0; k < 6; k++)
        {
            var v = eigen.Vectors.Column(k).Copy();
            var av = LinAlg.Dot(symmetric, v);
            var lv = v * eigen.Values.At(k);
            Assert.True(UFunc.AllClose(av, lv, 1e-7));
        }

        // Eigenvalues come back in descending order.
        for (var i = 1; i < 6; i++)
            Assert.True(eigen.Values.At(i - 1) >= eigen.Values.At(i));
    }

    [Fact]
    public void SymmetricEigen_RecoversKnownEigenvalues()
    {
        var a = NdArray.FromArray(new double[,] { { 2, 0, 0 }, { 0, 3, 4 }, { 0, 4, 9 } });
        var eigen = Decomposition.SymmetricEigen(a);
        var values = eigen.Values.ToArray().OrderBy(x => x).ToArray();

        // Eigenvalues of the 2x2 block are 1 and 11; the isolated entry contributes 2.
        Assert.Equal(1.0, values[0], 8);
        Assert.Equal(2.0, values[1], 8);
        Assert.Equal(11.0, values[2], 8);
    }

    // ---------------------------------------------------------------- fast path vs reference
    //
    // Svd and SymmetricEigen use Householder reduction plus a shifted QR/QL iteration.
    // SvdJacobi and SymmetricEigenJacobi compute the same factorisations by rotating pairs
    // until nothing changes — a completely different route to the same answer, which is what
    // makes them a real check rather than a restatement.

    [Theory]
    [InlineData(9, 5)]
    [InlineData(5, 9)]
    [InlineData(12, 12)]
    public void Svd_AgreesWithTheJacobiReference(int rows, int columns)
    {
        var rng = new GraviRandom(101);
        var a = rng.StandardNormal(rows, columns);

        var fast = Decomposition.Svd(a);
        var reference = Decomposition.SvdJacobi(a);

        Assert.True(UFunc.AllClose(fast.SingularValues, reference.SingularValues, 1e-9),
            $"singular values differ for {rows}x{columns}");
    }

    [Fact]
    public void SymmetricEigen_AgreesWithTheJacobiReference()
    {
        var rng = new GraviRandom(103);
        var m = rng.StandardNormal(20, 20);
        var symmetric = (m + m.T) * 0.5;

        var fast = Decomposition.SymmetricEigen(symmetric);
        var reference = Decomposition.SymmetricEigenJacobi(symmetric);

        Assert.True(UFunc.AllClose(fast.Values, reference.Values, 1e-9));
    }

    [Fact]
    public void SymmetricEigen_OrdersRepeatedEigenvaluesWithoutLosingAny()
    {
        // The QL iteration deflates blocks in convergence order, not magnitude order, so the
        // wrapper has to sort. A repeated eigenvalue is where an off-by-one in that sort shows:
        // it silently drops one copy and duplicates a neighbour.
        var a = NdArray.FromArray(new double[,]
        {
            { 5, 0, 0, 0 },
            { 0, -2, 0, 0 },
            { 0, 0, 5, 0 },
            { 0, 0, 0, 0 },
        });

        var eigen = Decomposition.SymmetricEigen(a);

        Assert.Equal(5.0, eigen.Values.At(0), 10);
        Assert.Equal(5.0, eigen.Values.At(1), 10);
        Assert.Equal(0.0, eigen.Values.At(2), 10);
        Assert.Equal(-2.0, eigen.Values.At(3), 10);

        // The eigenvectors must still be sorted alongside their values.
        for (var k = 0; k < 4; k++)
        {
            var v = eigen.Vectors.Column(k).Copy();
            Assert.True(UFunc.AllClose(LinAlg.Dot(a, v), v * eigen.Values.At(k), 1e-9));
        }
    }

    [Fact]
    public void Svd_HandlesRankDeficientAndZeroMatrices()
    {
        // Every row identical: one nonzero singular value, and the bidiagonal iteration has to
        // deflate the other three rather than divide by a zero pivot.
        var rankOne = NdArray.Zeros(6, 4);
        for (var i = 0; i < 6; i++)
            for (var j = 0; j < 4; j++)
                rankOne[i, j] = j + 1;

        var svd = Decomposition.Svd(rankOne);
        Assert.True(UFunc.AllClose(svd.Reconstruct(), rankOne, 1e-8));
        for (var i = 1; i < 4; i++)
            Assert.True(svd.SingularValues.At(i) < 1e-9, $"s[{i}] = {svd.SingularValues.At(i)}");

        var zeros = Decomposition.Svd(NdArray.Zeros(5, 3));
        for (var i = 0; i < 3; i++)
            Assert.Equal(0.0, zeros.SingularValues.At(i), 10);
    }

    [Theory]
    [InlineData(20, 6)]
    [InlineData(6, 20)]
    [InlineData(14, 14)]
    public void PartialSvd_MatchesTheFullFactorisation(int rows, int columns)
    {
        // Skipping a factor must change only what is computed, never the answer — including for
        // a wide matrix, where the transpose swaps which factor is the one being skipped.
        var rng = new GraviRandom(109);
        var a = rng.StandardNormal(rows, columns);
        var full = Decomposition.Svd(a);

        Assert.True(UFunc.AllClose(Decomposition.SingularValues(a), full.SingularValues, 1e-12));

        var (values, v) = Decomposition.SvdRightVectors(a);
        Assert.True(UFunc.AllClose(values, full.SingularValues, 1e-12));
        Assert.True(UFunc.AllClose(UFunc.Abs(v), UFunc.Abs(full.V), 1e-12));
    }

    [Fact]
    public void SymmetricEigen_ProducesOrthonormalVectors()
    {
        var rng = new GraviRandom(107);
        var m = rng.StandardNormal(16, 16);
        var symmetric = (m + m.T) * 0.5;

        var eigen = Decomposition.SymmetricEigen(symmetric);
        Assert.True(UFunc.AllClose(LinAlg.Dot(eigen.Vectors.T, eigen.Vectors), NdArray.Eye(16), 1e-9));
    }

    [Fact]
    public void Eigenvalues_OfANonSymmetricMatrixIncludeComplexPairs()
    {
        // Rotation by 90 degrees has eigenvalues +i and -i.
        var rotation = NdArray.FromArray(new double[,] { { 0, -1 }, { 1, 0 } });
        var (re, im) = Decomposition.Eigenvalues(rotation);

        Assert.Equal(0.0, re.At(0), 8);
        Assert.Equal(0.0, re.At(1), 8);
        Assert.Equal(1.0, Math.Abs(im.At(0)), 8);
        Assert.Equal(1.0, Math.Abs(im.At(1)), 8);
    }

    [Fact]
    public void Eigenvalues_OfATriangularMatrixAreItsDiagonal()
    {
        var a = NdArray.FromArray(new double[,] { { 1, 2, 3 }, { 0, 5, 6 }, { 0, 0, 9 } });
        var (re, _) = Decomposition.Eigenvalues(a);
        var values = re.ToArray().OrderBy(x => x).ToArray();

        Assert.Equal(1.0, values[0], 7);
        Assert.Equal(5.0, values[1], 7);
        Assert.Equal(9.0, values[2], 7);
    }

    [Fact]
    public void PseudoInverse_SolvesAnOverdeterminedSystem()
    {
        // y = 2x + 1 sampled at four points; least squares must recover the coefficients.
        var a = NdArray.FromArray(new double[,] { { 1, 1 }, { 2, 1 }, { 3, 1 }, { 4, 1 } });
        var y = NdArray.FromValues([3.0, 5.0, 7.0, 9.0]);

        var coefficients = LinAlg.LeastSquares(a, y);
        Assert.Equal(2.0, coefficients.At(0), 7);
        Assert.Equal(1.0, coefficients.At(1), 7);
    }

    [Fact]
    public void MatrixRank_CountsIndependentRows()
    {
        var rankTwo = NdArray.FromArray(new double[,] { { 1, 2, 3 }, { 2, 4, 6 }, { 1, 0, 1 } });
        Assert.Equal(2, LinAlg.MatrixRank(rankTwo));
        Assert.Equal(3, LinAlg.MatrixRank(NdArray.Eye(3)));
    }

    [Fact]
    public void Norms_MatchTheirDefinitions()
    {
        var v = NdArray.FromValues([3.0, -4.0]);
        Assert.Equal(5.0, LinAlg.Norm(v), 12);
        Assert.Equal(7.0, LinAlg.Norm(v, 1), 12);
        Assert.Equal(4.0, LinAlg.Norm(v, double.PositiveInfinity), 12);
    }

    [Fact]
    public void MatrixPower_MatchesRepeatedMultiplication()
    {
        var a = NdArray.FromArray(new double[,] { { 1, 1 }, { 1, 0 } });
        var fifth = LinAlg.MatrixPower(a, 5);

        // Powers of this matrix hold Fibonacci numbers.
        Assert.Equal(8.0, fifth[0, 0], 9);
        Assert.Equal(5.0, fifth[0, 1], 9);
    }

    [Fact]
    public void TraceAndDiagonal_ReadTheMainDiagonal()
    {
        var a = NdArray.FromArray(new double[,] { { 1, 2 }, { 3, 4 } });
        Assert.Equal(5.0, LinAlg.Trace(a));
        Assert.Equal(new[] { 1.0, 4.0 }, LinAlg.Diagonal(a).ToArray());
    }
}
