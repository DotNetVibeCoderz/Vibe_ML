using System.Numerics;
using Gravicode.Science.GraviNum;
using Xunit;

namespace GraviNum.Tests;

/// <summary>
/// Tests for the complex array.
/// </summary>
/// <remarks>
/// The interesting failures here are all sign errors that produce plausible numbers: a conjugate
/// left off an inner product, a Hermitian transpose done as a plain one. So the pins are the
/// identities those mistakes break — <c>⟨a, a⟩</c> being real and non-negative, <c>(AB)ᴴ = BᴴAᴴ</c>,
/// Parseval's theorem — rather than the arithmetic alone.
/// </remarks>
public class ComplexNdArrayTests
{
    private static ComplexNdArray Matrix(params Complex[] values)
        => new(values, 2, 2);

    [Fact]
    public void ElementwiseArithmeticMatchesTheHandComputation()
    {
        var a = ComplexNdArray.FromValues(new Complex(1, 2), new Complex(3, -1));
        var b = ComplexNdArray.FromValues(new Complex(0, 1), new Complex(2, 2));

        // (1+2i)(0+1i) = -2 + i, and (3-i)(2+2i) = 6 + 6i - 2i + 2 = 8 + 4i.
        var product = a * b;
        Assert.Equal(new Complex(-2, 1), product[0]);
        Assert.Equal(new Complex(8, 4), product[1]);

        var sum = a + b;
        Assert.Equal(new Complex(1, 3), sum[0]);

        var difference = a - b;
        Assert.Equal(new Complex(1, 1), difference[0]);
    }

    [Fact]
    public void DivisionInvertsMultiplication()
    {
        var a = ComplexNdArray.FromValues(new Complex(3, 4), new Complex(-1, 7));
        var b = ComplexNdArray.FromValues(new Complex(1, -2), new Complex(0.5, 0.25));

        Assert.True((a * b / b).AllClose(a, 1e-12));
    }

    [Fact]
    public void MagnitudeAndPhaseDescribeThePolarForm()
    {
        // 3 + 4i has magnitude 5 exactly, which is the point of picking it.
        var a = ComplexNdArray.FromValues(new Complex(3, 4), new Complex(0, -2));

        Assert.Equal(5.0, a.Magnitude().At(0), 12);
        Assert.Equal(2.0, a.Magnitude().At(1), 12);
        Assert.Equal(-Math.PI / 2, a.Phase().At(1), 12);

        // Power is the squared magnitude, computed without the round trip through a square root.
        Assert.Equal(25.0, a.Power().At(0), 12);
    }

    [Fact]
    public void MagnitudeSurvivesValuesThatWouldOverflowTheNaiveFormula()
    {
        // sqrt(re² + im²) overflows here even though the answer is perfectly representable — the
        // reason this goes through Complex.Abs rather than the obvious expression.
        var a = ComplexNdArray.FromValues(new Complex(1e200, 1e200));

        var magnitude = a.Magnitude().At(0);
        Assert.False(double.IsInfinity(magnitude), "the magnitude overflowed");
        Assert.Equal(Math.Sqrt(2) * 1e200, magnitude, 10);
    }

    [Fact]
    public void ConjugationFlipsOnlyTheImaginaryPart()
    {
        var a = ComplexNdArray.FromValues(new Complex(1, 2), new Complex(-3, -4));
        var conjugate = a.Conjugate();

        Assert.Equal(new Complex(1, -2), conjugate[0]);
        Assert.Equal(new Complex(-3, 4), conjugate[1]);

        // And conjugating twice is the identity.
        Assert.True(conjugate.Conjugate().AllClose(a, 1e-15));
    }

    [Fact]
    public void TheHermitianInnerProductOfAVectorWithItselfIsItsSquaredNorm()
    {
        // The identity a missing conjugate breaks: without it, ⟨[i], [i]⟩ comes out as -1, and a
        // "norm" of sqrt(-1) is how the bug announces itself several functions later.
        var a = ComplexNdArray.FromValues(new Complex(0, 1), new Complex(3, 4));

        var inner = ComplexNdArray.Inner(a, a);

        Assert.Equal(0.0, inner.Imaginary, 12);
        Assert.True(inner.Real > 0);
        Assert.Equal(1 + 25.0, inner.Real, 12);
        Assert.Equal(Math.Sqrt(26), a.Norm(), 12);
    }

    [Fact]
    public void TheInnerProductIsConjugateSymmetric()
    {
        // ⟨a, b⟩ = conj(⟨b, a⟩), which a plain unconjugated dot product does not satisfy.
        var a = ComplexNdArray.FromValues(new Complex(1, 2), new Complex(-1, 0.5));
        var b = ComplexNdArray.FromValues(new Complex(0, -1), new Complex(2, 3));

        var forward = ComplexNdArray.Inner(a, b);
        var backward = ComplexNdArray.Inner(b, a);

        Assert.Equal(forward.Real, backward.Real, 12);
        Assert.Equal(forward.Imaginary, -backward.Imaginary, 12);
    }

    [Fact]
    public void ConjugateTransposeDiffersFromPlainTranspose()
    {
        var a = Matrix(new Complex(1, 1), new Complex(2, -2), new Complex(3, 0), new Complex(0, 4));

        var plain = a.Transpose();
        var hermitian = a.ConjugateTranspose();

        Assert.Equal(new Complex(2, -2), plain[1, 0]);
        Assert.Equal(new Complex(2, 2), hermitian[1, 0]);
    }

    [Fact]
    public void AHermitianTransposeMakesTheGramMatrixPositiveSemiDefinite()
    {
        // AᴴA has a real, non-negative diagonal; AᵀA generally does not. This is exactly the
        // mistake that produces complex "variances" and it survives every shape check.
        var a = new ComplexNdArray(
            [new Complex(1, 2), new Complex(0, -1), new Complex(3, 1), new Complex(-2, 2)], 2, 2);

        var gram = ComplexNdArray.Dot(a.ConjugateTranspose(), a);

        for (var i = 0; i < 2; i++)
        {
            Assert.Equal(0.0, gram[i, i].Imaginary, 12);
            Assert.True(gram[i, i].Real >= 0, $"diagonal entry {i} was {gram[i, i].Real}");
        }

        // And it is Hermitian: G[i,j] = conj(G[j,i]).
        Assert.Equal(gram[0, 1].Real, gram[1, 0].Real, 12);
        Assert.Equal(gram[0, 1].Imaginary, -gram[1, 0].Imaginary, 12);
    }

    [Fact]
    public void TheProductOfHermitianTransposesReversesOrder()
    {
        // (AB)ᴴ = BᴴAᴴ. Getting the conjugation right but the order wrong passes shape checks and
        // fails this.
        var a = new ComplexNdArray(
            [new Complex(1, 1), new Complex(2, 0), new Complex(0, 3), new Complex(-1, 1),
             new Complex(2, -2), new Complex(1, 0)], 2, 3);

        var b = new ComplexNdArray(
            [new Complex(0, 1), new Complex(1, 1), new Complex(2, 0), new Complex(-1, 2),
             new Complex(1, -1), new Complex(3, 0)], 3, 2);

        var left = ComplexNdArray.Dot(a, b).ConjugateTranspose();
        var right = ComplexNdArray.Dot(b.ConjugateTranspose(), a.ConjugateTranspose());

        Assert.True(left.AllClose(right, 1e-12));
    }

    [Fact]
    public void ComplexMatrixProductAgreesWithTheRealOneWhenNothingIsImaginary()
    {
        // Pins the complex kernel against LinAlg.Dot, which is independently tested.
        var rng = new GraviRandom(9);
        var a = rng.StandardNormal(5, 4);
        var b = rng.StandardNormal(4, 3);

        var expected = LinAlg.Dot(a, b);
        var actual = ComplexNdArray.Dot(ComplexNdArray.FromParts(a), ComplexNdArray.FromParts(b));

        Assert.Equal([5, 3], actual.Shape.ToArray());
        for (var i = 0; i < expected.Size; i++)
        {
            Assert.Equal(expected.At(i), actual[i].Real, 10);
            Assert.Equal(0.0, actual[i].Imaginary, 12);
        }
    }

    [Fact]
    public void MatrixProductIsAssociative()
    {
        var rng = new GraviRandom(11);
        var a = ComplexNdArray.FromParts(rng.StandardNormal(3, 4), rng.StandardNormal(3, 4));
        var b = ComplexNdArray.FromParts(rng.StandardNormal(4, 2), rng.StandardNormal(4, 2));
        var c = ComplexNdArray.FromParts(rng.StandardNormal(2, 3), rng.StandardNormal(2, 3));

        var left = ComplexNdArray.Dot(ComplexNdArray.Dot(a, b), c);
        var right = ComplexNdArray.Dot(a, ComplexNdArray.Dot(b, c));

        Assert.True(left.AllClose(right, 1e-10));
    }

    [Fact]
    public void MismatchedShapesAreRejected()
    {
        var a = ComplexNdArray.Zeros(2, 3);
        var b = ComplexNdArray.Zeros(2, 2);

        Assert.Throws<ArgumentException>(() => a + b);
        Assert.Throws<ArgumentException>(() => ComplexNdArray.Dot(a, ComplexNdArray.Zeros(2, 4)));
        Assert.Throws<ArgumentException>(() => new ComplexNdArray(new Complex[5], 2, 3));
    }

    // ------------------------------------------------------------------- signal

    [Fact]
    public void TheTransformRoundTrips()
    {
        var rng = new GraviRandom(13);
        var signal = ComplexNdArray.FromParts(rng.StandardNormal(64), rng.StandardNormal(64));

        Assert.True(signal.Fft().Ifft().AllClose(signal, 1e-10));
    }

    [Fact]
    public void ParsevalsTheoremHolds()
    {
        // Energy is conserved by the transform: Σ|x|² = (1/N) Σ|X|². An independent check on the
        // scaling convention, which is the thing every FFT implementation gets to choose and can
        // therefore get wrong without any test noticing.
        var rng = new GraviRandom(17);
        var signal = ComplexNdArray.FromParts(rng.StandardNormal(128), rng.StandardNormal(128));

        var timeEnergy = 0.0;
        for (var i = 0; i < signal.Size; i++) timeEnergy += signal.Power().At(i);

        var spectrum = signal.Fft();
        var frequencyEnergy = 0.0;
        for (var i = 0; i < spectrum.Size; i++) frequencyEnergy += spectrum.Power().At(i);

        Assert.Equal(timeEnergy, frequencyEnergy / signal.Size, 8);
    }

    [Fact]
    public void ThePureToneLandsInASingleBin()
    {
        // The most direct statement of what the transform means, and one that a wrong sign in the
        // twiddle factor moves to the mirrored bin.
        const int n = 64;
        const int frequency = 5;

        var data = new Complex[n];
        for (var i = 0; i < n; i++)
            data[i] = Complex.FromPolarCoordinates(1.0, 2 * Math.PI * frequency * i / n);

        var spectrum = new ComplexNdArray(data, n).Fft();

        Assert.Equal(n, Complex.Abs(spectrum[frequency]), 8);
        for (var k = 0; k < n; k++)
            if (k != frequency)
                Assert.True(Complex.Abs(spectrum[k]) < 1e-8, $"bin {k} held {Complex.Abs(spectrum[k]):E3}");
    }

    [Fact]
    public void TheTwoDimensionalTransformRoundTrips()
    {
        var rng = new GraviRandom(19);
        var image = ComplexNdArray.FromParts(rng.StandardNormal(16, 8), rng.StandardNormal(16, 8));

        Assert.True(image.Fft2().Ifft2().AllClose(image, 1e-10));
    }

    [Fact]
    public void TheTwoDimensionalTransformOfAConstantIsASinglePeak()
    {
        // A flat image has no variation at any non-zero frequency, so all its energy is at DC.
        var data = new Complex[8 * 8];
        Array.Fill(data, Complex.One);

        var spectrum = new ComplexNdArray(data, 8, 8).Fft2();

        Assert.Equal(64.0, spectrum[0, 0].Real, 8);
        for (var i = 0; i < 8; i++)
            for (var j = 0; j < 8; j++)
                if (i != 0 || j != 0)
                    Assert.True(Complex.Abs(spectrum[i, j]) < 1e-9, $"bin ({i}, {j}) was not empty");
    }

    [Fact]
    public void SeparabilityMeansRowsAndColumnsCanBeTransformedInEitherOrder()
    {
        // Why the 2-D transform is two passes of 1-D rather than a double sum. If the order
        // mattered, the factorisation would be wrong.
        var rng = new GraviRandom(23);
        var image = ComplexNdArray.FromParts(rng.StandardNormal(8, 8), rng.StandardNormal(8, 8));

        var byRowsFirst = image.Fft2();
        var byColumnsFirst = image.Transpose().Fft2().Transpose();

        Assert.True(byRowsFirst.AllClose(byColumnsFirst, 1e-10));
    }

    // -------------------------------------------------------------------- shape

    [Fact]
    public void ReshapeInfersASingleMinusOne()
    {
        var a = ComplexNdArray.Zeros(12);
        Assert.Equal([3, 4], a.Reshape(3, -1).Shape.ToArray());
        Assert.Equal([2, 6], a.Reshape(-1, 6).Shape.ToArray());
    }

    [Fact]
    public void ReshapeCopiesRatherThanAliasing()
    {
        // Deliberately unlike NdArray, whose reshape is a view. Documenting it with a test so the
        // difference is a decision rather than a surprise.
        var a = ComplexNdArray.FromValues(Complex.One, Complex.Zero, Complex.Zero, Complex.One);
        var reshaped = a.Reshape(2, 2);

        reshaped[0, 0] = new Complex(9, 9);
        Assert.Equal(Complex.One, a[0]);
    }

    [Fact]
    public void PartsRoundTripThroughRealAndImaginary()
    {
        var rng = new GraviRandom(29);
        var real = rng.StandardNormal(3, 4);
        var imaginary = rng.StandardNormal(3, 4);

        var complex = ComplexNdArray.FromParts(real, imaginary);

        Assert.True(UFunc.AllClose(complex.Real(), real, 1e-15));
        Assert.True(UFunc.AllClose(complex.Imaginary(), imaginary, 1e-15));
        Assert.Equal([3, 4], complex.Shape.ToArray());
    }

    [Fact]
    public void APurelyRealArrayHasZeroImaginaryPart()
    {
        var complex = ComplexNdArray.FromParts(NdArray.FromValues([1.0, 2.0, 3.0]));

        for (var i = 0; i < 3; i++) Assert.Equal(0.0, complex[i].Imaginary);
    }

    [Fact]
    public void MatrixOperationsRefuseNonMatrices()
    {
        var a = ComplexNdArray.Zeros(2, 2, 2);

        Assert.Throws<InvalidOperationException>(() => a.Transpose());
        Assert.Throws<InvalidOperationException>(() => a.ConjugateTranspose());
        Assert.Throws<InvalidOperationException>(() => a.Fft2());
        Assert.Throws<InvalidOperationException>(() => a[0, 0]);
    }
}
