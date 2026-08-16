using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Single;
using Xunit;

namespace GraviNum.Tests;

/// <summary>
/// Tests for the single-precision prototype kernels.
/// </summary>
/// <remarks>
/// These are pinned against the <c>double</c> path with a tolerance appropriate to float32, which
/// carries roughly seven decimal digits against double's sixteen. Asserting equality to 1e-12 here
/// would be asserting that float is double.
/// </remarks>
public class SingleKernelTests
{
    [Fact]
    public void AddMatchesTheDoublePathWithinSinglePrecision()
    {
        var rng = new GraviRandom(31);

        // Above the parallel threshold, so the pinned-pointer path is the one being checked.
        const int n = 100_000;
        var a = rng.StandardNormal(n);
        var b = rng.StandardNormal(n);

        var expected = UFunc.Add(a, b);

        var af = SingleKernels.ToSingle(a.ToArray());
        var bf = SingleKernels.ToSingle(b.ToArray());
        var result = new float[n];
        SingleKernels.Add(af, bf, result);

        var worst = 0.0;
        for (var i = 0; i < n; i++) worst = Math.Max(worst, Math.Abs(expected.At(i) - result[i]));

        Assert.True(worst < 1e-5, $"largest difference from the double path was {worst:E2}");
    }

    [Fact]
    public void AddHandlesALengthBelowTheParallelThreshold()
    {
        // The short path skips Parallel.For entirely; a ragged tail past the vector width is
        // where an off-by-one would hide.
        var a = new float[7] { 1, 2, 3, 4, 5, 6, 7 };
        var b = new float[7] { 10, 20, 30, 40, 50, 60, 70 };
        var result = new float[7];

        SingleKernels.Add(a, b, result);

        for (var i = 0; i < 7; i++) Assert.Equal(a[i] + b[i], result[i], 5);
    }

    [Fact]
    public void AddRejectsMismatchedLengths()
    {
        Assert.Throws<ArgumentException>(() => SingleKernels.Add(new float[4], new float[5], new float[4]));
    }

    [Theory]
    [InlineData(64, 64, 64)]
    [InlineData(100, 73, 57)]
    [InlineData(129, 65, 33)]
    public void MultiplyMatchesANaiveTripleLoop(int m, int k, int n)
    {
        var rng = new GraviRandom(37);
        var a = SingleKernels.ToSingle(rng.StandardNormal(m, k).ToArray());
        var b = SingleKernels.ToSingle(rng.StandardNormal(k, n).ToArray());
        var c = new float[(long)m * n];

        SingleKernels.Multiply(a, b, c, m, n, k);

        var worst = 0.0;
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
            {
                var acc = 0.0;
                for (var p = 0; p < k; p++) acc += (double)a[i * k + p] * b[p * n + j];
                worst = Math.Max(worst, Math.Abs(acc - c[i * n + j]));
            }

        // Accumulating in float loses digits as k grows; the naive reference sums in double.
        Assert.True(worst < 1e-3, $"max difference from the naive product was {worst:E2} at {m}x{k}x{n}");
    }

    [Fact]
    public void SinglePrecisionCarriesFewerDigitsThanDouble()
    {
        // Worth pinning rather than assuming: this is the cost side of the trade the prototype
        // exists to measure, and it is why the float path is opt-in rather than the default.
        const float value = 1.0f;
        Assert.Equal(value, value + 1e-8f);          // lost entirely in float
        Assert.NotEqual(1.0, 1.0 + 1e-8);            // still visible in double
    }
}
