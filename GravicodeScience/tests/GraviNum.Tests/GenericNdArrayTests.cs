using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Generic;
using Xunit;

namespace GraviNum.Tests;

/// <summary>
/// Tests for the generic <see cref="NdArray{T}"/> and its kernels.
/// </summary>
/// <remarks>
/// The generic path is pinned against the <c>double</c> implementation the rest of the library
/// already trusts, and every numeric test runs at both widths. Tolerances differ by width on
/// purpose: a threshold written for <c>double</c> silently over-asserts on <c>float</c>, which
/// carries about seven significant digits rather than sixteen.
/// </remarks>
public class GenericNdArrayTests
{
    // ---------------------------------------------------------------- structure

    [Fact]
    public void ViewsShareOneBufferAtBothWidths()
    {
        // The defining property of the double NdArray, and it has to survive genericisation:
        // reshaping and transposing are views, so writing through one is visible through another.
        var a = new NdArray<float>([1f, 2f, 3f, 4f, 5f, 6f], 2, 3);
        var reshaped = a.Reshape(3, 2);

        reshaped[0, 1] = 99f;
        Assert.Equal(99f, a[0, 1]);

        var copy = a.Copy();
        copy[0, 0] = -1f;
        Assert.Equal(1f, a[0, 0]);      // Copy is the escape hatch
    }

    [Fact]
    public void TransposeIsAViewAndReadsCorrectly()
    {
        var a = new NdArray<double>([1, 2, 3, 4, 5, 6], 2, 3);
        var t = a.T2;

        Assert.Equal([3, 2], t.Shape.ToArray());
        Assert.False(t.IsContiguous);
        Assert.Equal(2.0, t[1, 0]);
        Assert.Equal(6.0, t[2, 1]);

        // A strided view still enumerates in logical row-major order.
        Assert.Equal([1, 4, 2, 5, 3, 6], t.ToArray());
    }

    [Fact]
    public void EyeAndFullBuildTheExpectedContents()
    {
        var eye = NdArray<float>.Eye(3);
        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 3; j++)
                Assert.Equal(i == j ? 1f : 0f, eye[i, j]);

        var full = NdArray<double>.Full(2.5, 2, 2);
        for (var i = 0; i < full.Size; i++) Assert.Equal(2.5, full.At(i));
    }

    [Fact]
    public void AsSpanRefusesAStridedView()
    {
        var a = new NdArray<double>([1, 2, 3, 4], 2, 2);
        Assert.Throws<InvalidOperationException>(() => a.T2.AsSpan());
    }

    [Fact]
    public void VectorWidthDoublesForSinglePrecision()
    {
        // This is the mechanism the whole exercise rests on: the same generic source compiles to
        // twice the lanes for float, which is half of where the measured speedup comes from.
        Assert.Equal(2 * UFunc<double>.VectorWidth, UFunc<float>.VectorWidth);
    }

    // ---------------------------------------------------------------- agreement with the double path

    [Fact]
    public void DoubleKernelsMatchTheNonGenericImplementation()
    {
        // Same algorithm, same width — so this should agree to the last bit, and any difference
        // is a transcription error rather than rounding.
        var rng = new GraviRandom(17);
        var a = rng.StandardNormal(500, 4);
        var b = rng.StandardNormal(500, 4);

        var expected = UFunc.Add(a, b);
        var actual = UFunc<double>.Add(NdArrayConvert.To<double>(a), NdArrayConvert.To<double>(b));

        for (var i = 0; i < expected.Size; i++)
            Assert.Equal(expected.At(i), actual.At(i));
    }

    [Fact]
    public void GenericDotMatchesTheNonGenericOne()
    {
        var rng = new GraviRandom(19);
        var a = rng.StandardNormal(70, 50);
        var b = rng.StandardNormal(50, 40);

        var expected = LinAlg.Dot(a, b);
        var actual = UFunc<double>.Dot(NdArrayConvert.To<double>(a), NdArrayConvert.To<double>(b));

        for (var i = 0; i < expected.Size; i++)
            Assert.True(Math.Abs(expected.At(i) - actual.At(i)) < 1e-9,
                $"element {i}: {expected.At(i):G17} vs {actual.At(i):G17}");
    }

    [Theory]
    [InlineData(64, 40, 32)]
    [InlineData(100, 73, 57)]    // none of the dimensions divide the tile size
    [InlineData(129, 65, 33)]
    public void SingleAndDoubleProductsAgreeToSinglePrecision(int m, int k, int n)
    {
        var rng = new GraviRandom(23);
        var a = rng.StandardNormal(m, k);
        var b = rng.StandardNormal(k, n);

        var wide = UFunc<double>.Dot(NdArrayConvert.To<double>(a), NdArrayConvert.To<double>(b));
        var narrow = UFunc<float>.Dot(NdArrayConvert.ToSingle(a), NdArrayConvert.ToSingle(b));

        var worst = 0.0;
        var scale = 0.0;
        for (var i = 0; i < wide.Size; i++)
        {
            worst = Math.Max(worst, Math.Abs(wide.At(i) - narrow.At(i)));
            scale = Math.Max(scale, Math.Abs(wide.At(i)));
        }

        Assert.True(worst / Math.Max(1.0, scale) < 1e-5,
            $"{m}x{k}x{n}: relative difference {worst / Math.Max(1.0, scale):E2}");
    }

    // ---------------------------------------------------------------- kernels at both widths

    [Fact]
    public void ElementwiseKernelsAgreeAtBothWidths()
    {
        var rng = new GraviRandom(29);

        // Above the parallel threshold, so the pinned-pointer path is the one under test.
        var a = rng.StandardNormal(60_000);
        var b = UFunc.AddScalar(UFunc.Abs(rng.StandardNormal(60_000)), 1.0);

        Check(UFunc<double>.Subtract, UFunc<float>.Subtract, "subtract");
        Check(UFunc<double>.Multiply, UFunc<float>.Multiply, "multiply");
        Check(UFunc<double>.Divide, UFunc<float>.Divide, "divide");

        void Check(Func<NdArray<double>, NdArray<double>, NdArray<double>> wide,
            Func<NdArray<float>, NdArray<float>, NdArray<float>> narrow, string name)
        {
            var w = wide(NdArrayConvert.To<double>(a), NdArrayConvert.To<double>(b));
            var s = narrow(NdArrayConvert.ToSingle(a), NdArrayConvert.ToSingle(b));

            var worst = 0.0;
            var scale = 0.0;
            for (var i = 0; i < w.Size; i++)
            {
                worst = Math.Max(worst, Math.Abs(w.At(i) - s.At(i)));
                scale = Math.Max(scale, Math.Abs(w.At(i)));
            }

            Assert.True(worst / Math.Max(1.0, scale) < 1e-5, $"{name}: {worst:E2}");
        }
    }

    [Fact]
    public void UnaryKernelsHandleARaggedTailPastTheVectorWidth()
    {
        // A length that is not a multiple of the vector width is where an off-by-one hides.
        var a = new NdArray<float>([1f, 4f, 9f, 16f, 25f, 36f, 49f], 7);
        var roots = UFunc<float>.Sqrt(a);

        for (var i = 0; i < 7; i++)
            Assert.Equal(MathF.Sqrt(a.At(i)), roots.At(i), 5);
    }

    [Fact]
    public void SumIsPairwiseAndStaysAccurateInSinglePrecision()
    {
        // A million values of 0.1f: a naive running total drifts badly here because the
        // accumulator quickly dwarfs each addend. Pairwise summation keeps the error logarithmic.
        const int n = 1_000_000;
        var data = new float[n];
        Array.Fill(data, 0.1f);

        var total = UFunc<float>.Sum(new NdArray<float>(data, n));

        Assert.True(Math.Abs(total - 100_000f) < 1f,
            $"pairwise sum drifted to {total:F2}, expected 100000");
    }

    [Fact]
    public void MeanMatchesTheDoublePath()
    {
        var rng = new GraviRandom(31);
        var a = rng.StandardNormal(5000);

        var expected = Statistics.Mean(a);
        var actual = UFunc<double>.Mean(NdArrayConvert.To<double>(a));

        Assert.True(Math.Abs(expected - actual) < 1e-12, $"{expected:G17} vs {actual:G17}");
    }

    [Fact]
    public void MismatchedShapesAreRejectedRatherThanBroadcast()
    {
        // The double UFunc broadcasts; these kernels deliberately do not, and saying so is better
        // than quietly producing a differently shaped result than the caller expected.
        var a = NdArray<double>.Zeros(3, 4);
        var b = NdArray<double>.Zeros(4);

        var error = Assert.Throws<ArgumentException>(() => UFunc<double>.Add(a, b));
        Assert.Contains("broadcast", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    // ---------------------------------------------------------------- conversion

    [Fact]
    public void WideningBackFromSingleIsExact()
    {
        // Every float32 is representable as a double, so this direction loses nothing — it just
        // does not recover what the narrowing threw away.
        var original = new NdArray<float>([1.5f, -2.25f, 0.125f], 3);
        var widened = NdArrayConvert.ToDouble(original);

        for (var i = 0; i < 3; i++) Assert.Equal((double)original.At(i), widened.At(i));
    }

    [Fact]
    public void NarrowingCostsPrecisionAndSaysSo()
    {
        // The trade the caller is making, pinned so nobody assumes float is free.
        var precise = NdArray.FromValues([1.0 + 1e-10]);
        var narrowed = NdArrayConvert.ToSingle(precise);

        Assert.Equal(1.0f, narrowed.At(0));                       // the digit is gone
        Assert.NotEqual(1.0, precise.At(0));                      // it was there in double

        Assert.True(NdArrayConvert.ComparisonTolerance<float>()
            > NdArrayConvert.ComparisonTolerance<double>());
    }

    [Fact]
    public void ConversionPreservesShapeAndStridedContents()
    {
        var rng = new GraviRandom(37);
        var a = rng.StandardNormal(6, 4);
        var strided = a.T;      // non-contiguous on the way in

        var converted = NdArrayConvert.To<double>(strided);

        Assert.Equal([4, 6], converted.Shape.ToArray());
        for (var i = 0; i < 4; i++)
            for (var j = 0; j < 6; j++)
                Assert.Equal(strided[i, j], converted[i, j]);
    }
}
