using Gravicode.Science.GraviNum;
using Xunit;

namespace GraviNum.Tests;

/// <summary>
/// Tests for Einstein summation.
/// </summary>
/// <remarks>
/// Each case is pinned against the operation it is meant to reproduce — <c>LinAlg.Dot</c>, a
/// transpose, a trace computed by hand — so a passing test says the notation means what it should,
/// not merely that the evaluator is self-consistent.
/// </remarks>
public class EinsumTests
{
    [Fact]
    public void MatrixProductMatchesDot()
    {
        var rng = new GraviRandom(3);
        var a = rng.StandardNormal(4, 5);
        var b = rng.StandardNormal(5, 3);

        var expected = LinAlg.Dot(a, b);
        var actual = Einsum.Evaluate("ij,jk->ik", a, b);

        Assert.Equal([4, 3], actual.Shape.ToArray());
        Assert.True(UFunc.AllClose(actual, expected, 1e-12));
    }

    [Fact]
    public void ContractingOverTheFirstAxisIsATransposedProduct()
    {
        // The case the notation earns its keep on: "ji,jk" says which axes meet, where
        // Dot(a.T, b) makes the reader work it out.
        var rng = new GraviRandom(5);
        var a = rng.StandardNormal(5, 4);
        var b = rng.StandardNormal(5, 3);

        var expected = LinAlg.Dot(a.T, b);
        var actual = Einsum.Evaluate("ji,jk->ik", a, b);

        Assert.True(UFunc.AllClose(actual, expected, 1e-12));
    }

    [Fact]
    public void TransposeAndDiagonalAndTrace()
    {
        var a = NdArray.FromArray(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });

        var transposed = Einsum.Evaluate("ij->ji", a);
        Assert.Equal([3, 3], transposed.Shape.ToArray());
        Assert.Equal(4.0, transposed[0, 1]);
        Assert.Equal(2.0, transposed[1, 0]);

        // A letter repeated inside one operand selects the diagonal, because both axes take the
        // same index — it falls out of the rule rather than being a special case.
        var diagonal = Einsum.Evaluate("ii->i", a);
        Assert.Equal([1.0, 5.0, 9.0], diagonal.ToArray());

        // And with nothing kept, the diagonal is summed: the trace.
        Assert.Equal(15.0, Einsum.Evaluate("ii->", a).At(0), 12);
    }

    [Fact]
    public void ReductionsAlongEachAxis()
    {
        var a = NdArray.FromArray(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });

        Assert.Equal([6.0, 15.0], Einsum.Evaluate("ij->i", a).ToArray());
        Assert.Equal([5.0, 7.0, 9.0], Einsum.Evaluate("ij->j", a).ToArray());
        Assert.Equal(21.0, Einsum.Evaluate("ij->", a).At(0), 12);
    }

    [Fact]
    public void InnerAndOuterProducts()
    {
        var x = NdArray.FromValues([1.0, 2.0, 3.0]);
        var y = NdArray.FromValues([4.0, 5.0]);

        var outer = Einsum.Evaluate("i,j->ij", x, y);
        Assert.Equal([3, 2], outer.Shape.ToArray());
        Assert.Equal(8.0, outer[1, 0], 12);      // 2 * 4
        Assert.Equal(15.0, outer[2, 1], 12);     // 3 * 5

        var z = NdArray.FromValues([2.0, 0.5, 1.0]);
        Assert.Equal(1 * 2 + 2 * 0.5 + 3 * 1.0, Einsum.Evaluate("i,i->", x, z).At(0), 12);
    }

    [Fact]
    public void FrobeniusInnerProductOfTwoMatrices()
    {
        var rng = new GraviRandom(7);
        var a = rng.StandardNormal(4, 6);
        var b = rng.StandardNormal(4, 6);

        var expected = 0.0;
        for (var i = 0; i < a.Size; i++) expected += a.At(i) * b.At(i);

        Assert.Equal(expected, Einsum.Evaluate("ij,ij->", a, b).At(0), 10);
    }

    [Fact]
    public void BatchedMatrixProduct()
    {
        // Rank 3 is where writing this out by hand stops being readable.
        var rng = new GraviRandom(11);
        var a = rng.StandardNormal(3, 4, 5);
        var b = rng.StandardNormal(3, 5, 2);

        var actual = Einsum.Evaluate("bij,bjk->bik", a, b);
        Assert.Equal([3, 4, 2], actual.Shape.ToArray());

        // Check each batch against an ordinary matrix product.
        for (var batch = 0; batch < 3; batch++)
        {
            var left = NdArray.Zeros(4, 5);
            var right = NdArray.Zeros(5, 2);
            for (var i = 0; i < 4; i++)
                for (var j = 0; j < 5; j++)
                    left[i, j] = a.At(batch * 20 + i * 5 + j);
            for (var j = 0; j < 5; j++)
                for (var k = 0; k < 2; k++)
                    right[j, k] = b.At(batch * 10 + j * 2 + k);

            var expected = LinAlg.Dot(left, right);
            for (var i = 0; i < 4; i++)
                for (var k = 0; k < 2; k++)
                    Assert.Equal(expected[i, k], actual.At(batch * 8 + i * 2 + k), 10);
        }
    }

    [Fact]
    public void TheOutputIsInferredWhenNoArrowIsGiven()
    {
        // NumPy's rule: letters appearing exactly once, alphabetically. So "ij,jk" contracts j.
        var rng = new GraviRandom(13);
        var a = rng.StandardNormal(3, 4);
        var b = rng.StandardNormal(4, 2);

        Assert.True(UFunc.AllClose(
            Einsum.Evaluate("ij,jk", a, b),
            Einsum.Evaluate("ij,jk->ik", a, b), 1e-12));
    }

    [Fact]
    public void MismatchedAxisLengthsAreRejected()
    {
        // The mistake that matters: if j is 5 in one operand and 4 in the other, the contraction
        // is meaningless. Without the check it would run over whichever came first.
        var a = NdArray.Zeros(3, 5);
        var b = NdArray.Zeros(4, 2);

        var error = Assert.Throws<ArgumentException>(() => Einsum.Evaluate("ij,jk->ik", a, b));
        Assert.Contains("j", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void SubscriptsThatDoNotMatchTheOperandsAreRejected()
    {
        var a = NdArray.Zeros(3, 4);

        // Wrong rank.
        Assert.Throws<ArgumentException>(() => Einsum.Evaluate("i->i", a));

        // Wrong number of operands.
        Assert.Throws<ArgumentException>(() => Einsum.Evaluate("ij,jk->ik", a));

        // Not a letter.
        Assert.Throws<ArgumentException>(() => Einsum.Evaluate("i1->i", a));
    }

    [Fact]
    public void SpacesInTheSubscriptStringAreIgnored()
    {
        var rng = new GraviRandom(17);
        var a = rng.StandardNormal(3, 3);

        Assert.True(UFunc.AllClose(
            Einsum.Evaluate("ij -> ji", a), Einsum.Evaluate("ij->ji", a), 1e-12));
    }
}
