using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Compute;
using Xunit;

namespace Gravicode.Science.Tests.GraviNum;

/// <summary>
/// Every matrix-product path — the automatic dispatch, the blocked kernel, the packed kernel on one
/// thread and in parallel — checked against a triple loop written here, over the shapes where tiled
/// kernels go wrong: ragged edges, degenerate dimensions, and operands that are views.
/// </summary>
public class MatMulKernelTests
{
    private static readonly MatMulKernel[] Kernels =
        [MatMulKernel.Auto, MatMulKernel.Blocked, MatMulKernel.PackedSerial, MatMulKernel.Packed];

    /// <summary>Independent of every kernel under test: indexer reads, one sum per element.</summary>
    private static double[,] Naive(NdArray a, NdArray b)
    {
        int m = a.Shape[0], k = a.Shape[1], n = b.Shape[1];
        var c = new double[m, n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
            {
                var s = 0.0;
                for (var p = 0; p < k; p++) s += a[i, p] * b[p, j];
                c[i, j] = s;
            }
        return c;
    }

    private static void AssertMatches(double[,] expected, NdArray actual, int k, string label)
    {
        Assert.Equal(expected.GetLength(0), actual.Shape[0]);
        Assert.Equal(expected.GetLength(1), actual.Shape[1]);
        // Standard-normal entries: the rounding error of a k-term sum is ~sqrt(k)·eps·|terms|.
        var tolerance = 1e-13 * Math.Max(1, k);
        for (var i = 0; i < expected.GetLength(0); i++)
            for (var j = 0; j < expected.GetLength(1); j++)
                Assert.True(Math.Abs(expected[i, j] - actual[i, j]) <= tolerance,
                    $"{label}: C[{i},{j}] = {actual[i, j]}, expected {expected[i, j]}");
    }

    private static void WithManagedPaths(Action body)
    {
        var previous = NativeBlas.Enabled;
        NativeBlas.Enabled = false;
        try { body(); }
        finally { NativeBlas.Enabled = previous; }
    }

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(1, 40, 1)]
    [InlineData(1, 37, 65)]     // row vector · matrix
    [InlineData(65, 37, 1)]     // matrix · column
    [InlineData(2, 3, 2)]
    [InlineData(4, 4, 4)]
    [InlineData(17, 17, 17)]
    [InlineData(63, 63, 63)]
    [InlineData(65, 65, 65)]
    [InlineData(127, 127, 127)]
    [InlineData(129, 129, 129)]
    [InlineData(257, 257, 257)] // two k slices, ragged
    [InlineData(3, 500, 7)]
    [InlineData(500, 3, 9)]
    [InlineData(129, 260, 257)]
    [InlineData(300, 20, 4)]    // n equal to the vector width
    [InlineData(5, 600, 33)]
    [InlineData(1000, 16, 5)]
    public void EveryPathMatchesATripleLoop(int m, int k, int n)
    {
        var rng = new GraviRandom(m * 10007 + k * 101 + n);
        var a = rng.StandardNormal(m, k);
        var b = rng.StandardNormal(k, n);
        var expected = Naive(a, b);

        WithManagedPaths(() =>
        {
            foreach (var kernel in Kernels)
                AssertMatches(expected, LinAlg.MatMul(a, b, kernel), k, $"{kernel} {m}x{k}x{n}");
            AssertMatches(expected, LinAlg.Dot(a, b), k, $"Dot {m}x{k}x{n}");
        });
    }

    [Theory]
    [InlineData(17, 33, 9)]
    [InlineData(65, 129, 63)]
    [InlineData(130, 70, 140)]
    [InlineData(70, 90, 1)]     // one column: contiguous goes to mat-vec, a view to the packed kernel
    public void TransposedViewsAreMultipliedInPlace(int m, int k, int n)
    {
        // a.T and b.T are views with a unit row stride: the packers take a different branch for
        // them, and Dot no longer copies them to contiguous storage first.
        var rng = new GraviRandom(5);
        var aT = rng.StandardNormal(k, m).T;
        var bT = rng.StandardNormal(n, k).T;
        var a = aT.Copy();
        var b = bT.Copy();
        Assert.False(aT.IsContiguous);

        WithManagedPaths(() =>
        {
            foreach (var kernel in Kernels)
            {
                AssertMatches(Naive(a, b), LinAlg.MatMul(aT, b, kernel), k, $"{kernel} Aᵀ");
                AssertMatches(Naive(a, b), LinAlg.MatMul(a, bT, kernel), k, $"{kernel} Bᵀ");
                AssertMatches(Naive(a, b), LinAlg.MatMul(aT, bT, kernel), k, $"{kernel} AᵀBᵀ");
            }
        });
    }

    [Fact]
    public void ATransposedMatrixTimesAVectorMatchesATripleLoop()
    {
        var rng = new GraviRandom(31);
        var aT = rng.StandardNormal(90, 70).T;     // 70 x 90 view
        var x = rng.StandardNormal(90);
        var expected = Naive(aT, x.Reshape(90, 1));
        WithManagedPaths(() =>
        {
            var y = LinAlg.Dot(aT, x);
            Assert.Equal([70], y.Shape.ToArray());
            AssertMatches(expected, y.Reshape(70, 1), 90, "Aᵀx");
        });
    }

    [Fact]
    public void StridedAndReversedSlicesAreMultipliedCorrectly()
    {
        // Every other row and every third column, starting mid-buffer; then a reversed view,
        // whose negative strides exercise the pointer arithmetic in the packers.
        var rng = new GraviRandom(11);
        var bigA = rng.StandardNormal(150, 200);
        var bigB = rng.StandardNormal(260, 90);
        var a = bigA.Slice(Slice.Range(3, 143, 2), Slice.Range(1, 200, 3));   // 70 x 67
        var b = bigB.Slice(Slice.Range(5, 72), Slice.Range(2, 90));           // 67 x 88
        var reversed = bigB.Slice(Slice.Reversed, Slice.Reversed).Slice(Slice.Range(0, 67), Slice.All);
        Assert.False(a.IsContiguous);
        Assert.Equal(67, a.Shape[1]);

        WithManagedPaths(() =>
        {
            foreach (var kernel in Kernels)
            {
                AssertMatches(Naive(a, b), LinAlg.MatMul(a, b, kernel), 67, $"{kernel} strided");
                AssertMatches(Naive(a, reversed), LinAlg.MatMul(a, reversed, kernel), 67, $"{kernel} reversed");
            }
        });
    }

    [Fact]
    public void TheParallelPackedKernelIsBitIdenticalToTheSerialOne()
    {
        // Parallelism only ever divides C, never k, so every element is summed in the same order
        // however the work is split. Bit equality, not closeness, is the claim.
        var rng = new GraviRandom(3);
        foreach (var (m, k, n) in new[] { (200, 200, 200), (129, 600, 75), (1000, 64, 300) })
        {
            var a = rng.StandardNormal(m, k);
            var b = rng.StandardNormal(k, n);
            var serial = LinAlg.MatMul(a, b, MatMulKernel.PackedSerial).ToArray();
            for (var run = 0; run < 3; run++)
                Assert.Equal(serial, LinAlg.MatMul(a, b, MatMulKernel.Packed).ToArray());
        }
    }

    [Fact]
    public void DotIsDeterministicRunToRun()
    {
        var rng = new GraviRandom(17);
        var a = rng.StandardNormal(300, 257);
        var b = rng.StandardNormal(257, 129);
        WithManagedPaths(() =>
        {
            var first = LinAlg.Dot(a, b).ToArray();
            for (var run = 0; run < 5; run++) Assert.Equal(first, LinAlg.Dot(a, b).ToArray());
        });
    }

    [Fact]
    public void TheReferenceKernelIsATripleLoop()
    {
        var rng = new GraviRandom(23);
        var a = rng.StandardNormal(13, 9).T;   // 9 x 13 view, to prove it reads through strides
        var b = rng.StandardNormal(13, 4);
        AssertMatches(Naive(a, b), LinAlg.MatMul(a, b, MatMulKernel.Reference), 13, "Reference");
    }

    [Fact]
    public void EmptyAndOneColumnProductsKeepTheirShape()
    {
        var a = NdArray.Ones(5, 0);
        var b = NdArray.Ones(0, 3);
        var c = LinAlg.Dot(a, b);
        Assert.Equal([5, 3], c.Shape.ToArray());
        Assert.All(c.ToArray(), v => Assert.Equal(0.0, v));

        var column = LinAlg.Dot(NdArray.Ones(4, 6), NdArray.Ones(6, 1));
        Assert.Equal([4, 1], column.Shape.ToArray());
        Assert.All(column.ToArray(), v => Assert.Equal(6.0, v));
    }

    [Fact]
    public void DispatchFollowsTheMeasuredRules()
    {
        Assert.False(PackedMatMul.ShouldUse(8, 8, 8));       // tiny: packing cannot pay
        Assert.True(PackedMatMul.ShouldUse(16, 16, 16));
        Assert.True(PackedMatMul.ShouldUse(200, 200, 200));  // the size this kernel exists for
        Assert.False(PackedMatMul.ShouldUse(1, 500, 500));   // vector · matrix
        var w = System.Numerics.Vector<double>.Count;
        Assert.False(PackedMatMul.ShouldUse(500, w, 500));   // C exactly one vector wide (m, n, k order)
        Assert.True(PackedMatMul.ShouldUse(500, w + 1, 500));
    }

    [Fact]
    public void GemmHonoursAlphaBetaAndLeadingDimension()
    {
        // The pointer API updates a block of a larger matrix in place: C ← alpha·A·B + beta·C.
        var rng = new GraviRandom(29);
        const int m = 37, k = 41, n = 19, ldc = 30;
        var a = rng.StandardNormal(m, k).ToArray();
        var b = rng.StandardNormal(k, n).ToArray();
        var c = rng.StandardNormal(m, ldc).ToArray();
        var original = (double[])c.Clone();
        unsafe
        {
            fixed (double* pa = a, pb = b, pc = c)
                PackedMatMul.Gemm(m, n, k, -0.5, pa, k, 1, pb, n, 1, 2.0, pc, ldc);
        }
        for (var i = 0; i < m; i++)
            for (var j = 0; j < ldc; j++)
            {
                if (j >= n)
                {
                    Assert.Equal(original[i * ldc + j], c[i * ldc + j]);   // outside the block: untouched
                    continue;
                }
                var s = 0.0;
                for (var p = 0; p < k; p++) s += a[i * k + p] * b[p * n + j];
                var expected = -0.5 * s + 2.0 * original[i * ldc + j];
                Assert.True(Math.Abs(expected - c[i * ldc + j]) < 1e-12, $"C[{i},{j}]");
            }
    }
}
