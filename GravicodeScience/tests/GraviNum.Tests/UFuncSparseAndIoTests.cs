using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Compute;
using Gravicode.Science.GraviNum.Io;
using Xunit;

namespace Gravicode.Science.Tests.GraviNum;

public class UFuncTests
{
    [Fact]
    public void ArithmeticOperators_WorkElementWise()
    {
        var a = NdArray.FromValues([1.0, 2.0, 3.0]);
        var b = NdArray.FromValues([4.0, 5.0, 6.0]);

        Assert.Equal(new[] { 5.0, 7.0, 9.0 }, (a + b).ToArray());
        Assert.Equal(new[] { -3.0, -3.0, -3.0 }, (a - b).ToArray());
        Assert.Equal(new[] { 4.0, 10.0, 18.0 }, (a * b).ToArray());
        Assert.Equal(new[] { 2.0, 4.0, 6.0 }, (a * 2.0).ToArray());
        Assert.Equal(new[] { -1.0, -2.0, -3.0 }, (-a).ToArray());
    }

    [Fact]
    public void SimdAndScalarPathsAgree()
    {
        // Sizes deliberately straddle the vector width and the parallel threshold.
        foreach (var n in new[] { 1, 3, 7, 8, 1023, UFunc.ParallelThreshold + 17 })
        {
            var rng = new GraviRandom(n);
            var a = rng.StandardNormal(n);
            var b = rng.StandardNormal(n);

            var vectorised = UFunc.Add(a, b);
            for (var i = 0; i < n; i++)
                Assert.Equal(a.At(i) + b.At(i), vectorised.At(i), 12);
        }
    }

    [Fact]
    public void MathFunctions_MatchSystemMath()
    {
        var a = NdArray.FromValues([0.25, 1.0, 4.0]);
        Assert.Equal(new[] { 0.5, 1.0, 2.0 }, UFunc.Sqrt(a).ToArray());
        Assert.Equal(Math.Log(4.0), UFunc.Log(a).At(2), 12);
        Assert.Equal(Math.Exp(1.0), UFunc.Exp(a).At(1), 12);
    }

    [Fact]
    public void Clip_BoundsEveryElement()
    {
        var a = NdArray.FromValues([-5.0, 0.5, 10.0]);
        Assert.Equal(new[] { 0.0, 0.5, 1.0 }, UFunc.Clip(a, 0, 1).ToArray());
    }

    [Fact]
    public void Sigmoid_IsCentredAtOneHalf()
    {
        var a = NdArray.FromValues([-100.0, 0.0, 100.0]);
        var s = UFunc.Sigmoid(a);
        Assert.Equal(0.0, s.At(0), 12);
        Assert.Equal(0.5, s.At(1), 12);
        Assert.Equal(1.0, s.At(2), 12);
    }

    [Fact]
    public void BinaryOpsOnStridedViews_RespectBroadcasting()
    {
        var a = NdArray.Arange(12).Reshape(3, 4);
        var column = a.Slice(Slice.All, Slice.Range(0, 1));   // 3 x 1, strided
        var result = a - column;

        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 4; j++)
                Assert.Equal(a[i, j] - a[i, 0], result[i, j], 12);
    }

    [Fact]
    public void AllClose_ComparesWithTolerance()
    {
        var a = NdArray.FromValues([1.0, 2.0]);
        var b = NdArray.FromValues([1.0 + 1e-12, 2.0]);
        Assert.True(UFunc.AllClose(a, b));
        Assert.False(UFunc.AllClose(a, b, 1e-15));
    }
}

public class SparseMatrixTests
{
    private static NdArray SampleDense() => NdArray.FromArray(new double[,]
    {
        { 1, 0, 0, 2 },
        { 0, 0, 3, 0 },
        { 0, 0, 0, 0 },
        { 4, 5, 0, 6 },
    });

    [Fact]
    public void FromDense_StoresOnlyNonZeros()
    {
        var sparse = SparseMatrix.FromDense(SampleDense());
        Assert.Equal(6, sparse.NonZeroCount);
        Assert.Equal(4, sparse.Rows);
        Assert.Equal(4, sparse.Columns);
        Assert.Equal(6.0 / 16.0, sparse.Density, 12);
    }

    [Fact]
    public void ToDense_RoundTrips()
    {
        var dense = SampleDense();
        Assert.True(UFunc.AllClose(SparseMatrix.FromDense(dense).ToDense(), dense));
    }

    [Fact]
    public void Indexer_ReadsStoredAndMissingEntries()
    {
        var sparse = SparseMatrix.FromDense(SampleDense());
        Assert.Equal(3.0, sparse[1, 2]);
        Assert.Equal(0.0, sparse[2, 2]);
    }

    [Fact]
    public void MatrixVectorProduct_MatchesTheDenseResult()
    {
        var dense = SampleDense();
        var sparse = SparseMatrix.FromDense(dense);
        var v = NdArray.FromValues([1.0, 2.0, 3.0, 4.0]);

        Assert.True(UFunc.AllClose(sparse.Multiply(v), LinAlg.Dot(dense, v), 1e-12));
    }

    [Fact]
    public void MatrixMatrixProduct_MatchesTheDenseResult()
    {
        var dense = SampleDense();
        var sparse = SparseMatrix.FromDense(dense);
        var rng = new GraviRandom(5);
        var b = rng.StandardNormal(4, 3);

        Assert.True(UFunc.AllClose(sparse.Multiply(b, denseIsMatrix: true), LinAlg.Dot(dense, b), 1e-12));
    }

    [Fact]
    public void Transpose_MatchesTheDenseTranspose()
    {
        var dense = SampleDense();
        var transposed = SparseMatrix.FromDense(dense).Transpose();
        Assert.True(UFunc.AllClose(transposed.ToDense(), dense.T, 1e-12));
    }

    [Fact]
    public void FromTriplets_SumsDuplicateCoordinates()
    {
        var sparse = SparseMatrix.FromTriplets(2, 2, [(0, 0, 1.0), (0, 0, 2.0), (1, 1, 5.0)]);
        Assert.Equal(3.0, sparse[0, 0]);
        Assert.Equal(5.0, sparse[1, 1]);
        Assert.Equal(2, sparse.NonZeroCount);
    }

    [Fact]
    public void Builder_ProducesTheSameMatrix()
    {
        var built = new SparseBuilder(2, 2).Add(0, 1, 7.0).Add(1, 0, 8.0).Build();
        Assert.Equal(7.0, built[0, 1]);
        Assert.Equal(8.0, built[1, 0]);
    }

    [Fact]
    public void LargeSparseProduct_TakesTheParallelPathCorrectly()
    {
        // Above 512 rows the multiply parallelises; the answer must be unchanged.
        var rng = new GraviRandom(9);
        var dense = NdArray.Zeros(700, 50);
        for (var i = 0; i < 700; i++)
            for (var k = 0; k < 3; k++)
                dense[i, rng.Next(50)] = rng.Normal();

        var sparse = SparseMatrix.FromDense(dense);
        var v = rng.StandardNormal(50);
        Assert.True(UFunc.AllClose(sparse.Multiply(v), LinAlg.Dot(dense, v), 1e-10));
    }
}

public class IoTests : IDisposable
{
    private readonly string _directory = Directory.CreateTempSubdirectory("gravinum-io").FullName;

    private string Path(string name) => System.IO.Path.Combine(_directory, name);

    [Fact]
    public void BinaryRoundTrip_PreservesShapeAndValues()
    {
        var rng = new GraviRandom(3);
        var original = rng.StandardNormal(5, 7);
        var path = Path("array.gnb");

        NdIO.SaveBinary(original, path);
        var loaded = NdIO.LoadBinary(path);

        Assert.Equal(original.Rank, loaded.Rank);
        Assert.Equal(original.Shape[1], loaded.Shape[1]);
        Assert.True(UFunc.AllClose(original, loaded, 0));
    }

    [Fact]
    public void CsvRoundTrip_PreservesValues()
    {
        var original = NdArray.FromArray(new double[,] { { 1.5, 2.5 }, { -3.25, 4.125 } });
        var path = Path("array.csv");

        NdIO.SaveCsv(original, path);
        var loaded = NdIO.LoadCsv(path);

        Assert.True(UFunc.AllClose(original, loaded, 1e-12));
    }

    [Fact]
    public void JsonRoundTrip_PreservesShapeAndValues()
    {
        var original = NdArray.Arange(24).Reshape(2, 3, 4);
        var json = NdIO.ToJson(original);
        var loaded = NdIO.FromJson(json);

        Assert.Equal(3, loaded.Rank);
        Assert.True(UFunc.AllClose(original, loaded, 0));
    }

    [Fact]
    public void MemoryMappedArray_ReadsBackWhatItWrote()
    {
        var path = Path("mapped.gmm");
        var rng = new GraviRandom(5);
        var original = rng.StandardNormal(64, 16);

        using (var mapped = MemoryMappedArray.Persist(original, path)) { }

        using var reopened = MemoryMappedArray.Open(path);
        Assert.Equal(1024, reopened.Length);
        Assert.Equal(new[] { 64, 16 }, reopened.Shape);
        Assert.True(UFunc.AllClose(reopened.ToNdArray(), original, 0));

        // Row access must not require materialising the whole file.
        var row = reopened.ReadRow(7);
        for (var j = 0; j < 16; j++) Assert.Equal(original[7, j], row.At(j), 12);
    }

    [Fact]
    public void MemoryMappedArray_SupportsRandomWrites()
    {
        var path = Path("writable.gmm");
        using (var mapped = MemoryMappedArray.Create(path, 10, 10))
        {
            mapped.Set(3, 4, 42.0);
            mapped.Flush();
        }

        using var reopened = MemoryMappedArray.Open(path);
        Assert.Equal(42.0, reopened.Get(3, 4));
    }

    [Fact]
    public void MemoryMappedArray_StreamsInChunks()
    {
        var path = Path("chunks.gmm");
        using (MemoryMappedArray.Persist(NdArray.Arange(1000), path)) { }

        using var mapped = MemoryMappedArray.Open(path);
        var total = mapped.Chunks(chunkSize: 128).Sum(chunk => chunk.Sum());
        Assert.Equal(499_500.0, total, 6);
    }

    public void Dispose()
    {
        try { Directory.Delete(_directory, recursive: true); } catch (IOException) { }
        GC.SuppressFinalize(this);
    }
}

public class ComputeBackendTests
{
    [Fact]
    public void CpuBackend_IsAlwaysAvailableAndDescribesItself()
    {
        Assert.False(Compute.Cpu.IsGpu);
        Assert.Contains("CPU", Compute.Cpu.Name);
    }

    [Fact]
    public void CpuBackend_MatMulMatchesLinAlg()
    {
        var rng = new GraviRandom(7);
        var a = rng.StandardNormal(12, 9);
        var b = rng.StandardNormal(9, 5);

        var viaBackend = new NdArray(Compute.Cpu.MatMul(a.ToArray(), b.ToArray(), 12, 9, 5), 12, 5);
        Assert.True(UFunc.AllClose(viaBackend, LinAlg.Dot(a, b), 1e-10));
    }

    [Fact]
    public void Best_FallsBackToCpuForSmallProblems()
    {
        Assert.Same(Compute.Cpu, Compute.Best(100));
    }

    [Fact]
    public void DescribeDevices_NeverThrowsEvenWithoutAGpu()
    {
        var description = Compute.DescribeDevices();
        Assert.False(string.IsNullOrWhiteSpace(description));
        Assert.Contains("CPU", description);
    }

    [Fact]
    public void ComputeDot_AgreesWithLinAlgDot()
    {
        // Routes through whichever backend Best() picks; the answer must not depend on that.
        var rng = new GraviRandom(11);
        var a = rng.StandardNormal(40, 30);
        var b = rng.StandardNormal(30, 20);

        Assert.True(UFunc.AllClose(Compute.Dot(a, b), LinAlg.Dot(a, b), 1e-9));
    }
}
